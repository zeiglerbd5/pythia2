"""
Integrated Production System: Ensemble Prediction + RL Position Sizing

This module combines:
1. Volatility-filtered ensemble spike predictor (50% precision)
2. RL position sizer v4 (70.2% win rate, 10.61 profit factor)

Pipeline: Signal -> Volatility Filter -> Ensemble Prediction -> RL Sizing -> Execution

Production-ready with clear configuration and execution flow.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import duckdb
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

# ML imports
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProductionConfig:
    """Production system configuration."""
    # Database
    db_path: str = "/Users/bz/Pythia2/full_pythia.duckdb"

    # Volatility filter (top 50% volatility = filter P50+)
    volatility_percentile_threshold: float = 50.0
    volatility_col: str = 'volatility_4h'

    # Ensemble settings
    use_orderbook_features: bool = True
    ensemble_method: str = 'weighted_voting'
    min_ensemble_confidence: float = 0.1  # Min probability to consider trade (lower to let RL decide)

    # RL position sizing
    rl_model_path: str = "/Users/bz/Pythia2/models/rl_position_sizer_v4.pt"
    position_sizes: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    # Trading parameters
    initial_capital: float = 10000.0
    max_position_pct: float = 20.0
    max_concurrent_positions: int = 3
    stop_loss_pct: float = 3.0
    trailing_stop_activation: float = 2.0  # Trail activates at +2%
    max_hold_hours: int = 24
    fee_pct: float = 0.1
    slippage_pct: float = 0.1

    # Fixed TP for baseline comparison only (integrated uses ratcheting trail)
    take_profit_pct: float = 6.0

    # Ratcheting trailing stop (replaces fixed TP)
    # Format: (gain_threshold_pct, trail_distance_pct)
    # Trail distance increases/tightens based on profit level
    ratchet_levels: List[Tuple[float, float]] = field(default_factory=lambda: [
        (2.0, 1.2),   # 2-6%:   1.2% trail (activation zone)
        (6.0, 1.8),   # 6-10%:  1.8% trail (former TP zone - liberal)
        (10.0, 1.2),  # 10-15%: 1.2% trail (tightening)
        (15.0, 4.0),  # 15%+:   4.0% trail (lock in big gains)
    ])


@dataclass
class RLConfigV4:
    """RL agent v4 configuration."""
    state_dim: int = 12
    n_actions: int = 5
    position_sizes: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    hidden_dim: int = 128


# =============================================================================
# ORDERBOOK FEATURE COMPUTER
# =============================================================================

class OrderbookFeatureComputer:
    """Computes orderbook microstructure features."""

    FEATURE_NAMES = [
        'spread_bps', 'imbalance_l3', 'imbalance_l5', 'imbalance_l10',
        'depth_ratio', 'log_depth', 'large_order_imbalance',
        'spread_volatility', 'spread_trend',
    ]

    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path, read_only=True)
        self._cache = {}

    def compute_features_at_time(
        self,
        symbol: str,
        timestamp: datetime,
        lookback_minutes: int = 30
    ) -> Dict[str, float]:
        """Compute orderbook features at a specific time."""
        cache_key = f"{symbol}_{timestamp.isoformat()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            query = f"""
                SELECT timestamp, bids, asks, best_bid, best_ask, mid_price, spread_bps
                FROM order_book_snapshots
                WHERE symbol = '{symbol}'
                  AND timestamp >= '{timestamp - timedelta(minutes=lookback_minutes)}'
                  AND timestamp <= '{timestamp}'
                ORDER BY timestamp DESC
                LIMIT 20
            """
            snapshots = self.conn.execute(query).df()

            if len(snapshots) == 0:
                return self._empty_features()

            latest = snapshots.iloc[0]
            bids = json.loads(latest['bids']) if isinstance(latest['bids'], str) else latest['bids']
            asks = json.loads(latest['asks']) if isinstance(latest['asks'], str) else latest['asks']

            features = {'spread_bps': latest['spread_bps']}

            # Volume imbalance at multiple levels
            for n_levels in [3, 5, 10]:
                bid_sizes = [b[1] for b in bids[:n_levels]]
                ask_sizes = [a[1] for a in asks[:n_levels]]
                total = sum(bid_sizes) + sum(ask_sizes)
                features[f'imbalance_l{n_levels}'] = (sum(bid_sizes) - sum(ask_sizes)) / total if total > 0 else 0

            # Depth features
            bid_prices = [b[0] for b in bids[:10]]
            bid_sizes = [b[1] for b in bids[:10]]
            ask_prices = [a[0] for a in asks[:10]]
            ask_sizes = [a[1] for a in asks[:10]]

            bid_depth = sum(p * s for p, s in zip(bid_prices, bid_sizes))
            ask_depth = sum(p * s for p, s in zip(ask_prices, ask_sizes))
            features['depth_ratio'] = bid_depth / ask_depth if ask_depth > 0 else 1.0
            features['log_depth'] = np.log10(max(bid_depth + ask_depth, 1))

            # Large order imbalance
            if bid_sizes and ask_sizes:
                p90 = np.percentile(bid_sizes + ask_sizes, 90)
                large_bids = sum(1 for s in bid_sizes if s > p90)
                large_asks = sum(1 for s in ask_sizes if s > p90)
                features['large_order_imbalance'] = large_bids - large_asks
            else:
                features['large_order_imbalance'] = 0

            # Spread dynamics
            if len(snapshots) > 1:
                spreads = snapshots['spread_bps'].values
                features['spread_volatility'] = np.std(spreads)
                features['spread_trend'] = spreads[0] - spreads[-1]
            else:
                features['spread_volatility'] = 0
                features['spread_trend'] = 0

            self._cache[cache_key] = features
            return features

        except Exception:
            return self._empty_features()

    def _empty_features(self) -> Dict[str, float]:
        return {name: None for name in self.FEATURE_NAMES}

    def batch_compute(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Compute orderbook features for all signals."""
        features_list = []
        for idx, row in signals_df.iterrows():
            feat = self.compute_features_at_time(row['symbol'], row['timestamp'])
            features_list.append(feat)
        return pd.DataFrame(features_list)


# =============================================================================
# ENSEMBLE SPIKE PREDICTOR
# =============================================================================

class EnsembleSpikePredictor:
    """Volatility-filtered ensemble spike predictor."""

    CATALYST_FEATURES = [
        'event_priority', 'sentiment_score', 'log_usd_value',
        'volatility_4h', 'momentum_4h', 'volume_ratio', 'rsi_proxy',
    ]

    ORDERBOOK_FEATURES = OrderbookFeatureComputer.FEATURE_NAMES

    BOOL_FEATURES = ['is_bearish_flow', 'is_bullish_flow', 'has_direction']

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.models = {}
        self.model_weights = {}
        self.volatility_threshold = None

        if config.use_orderbook_features:
            self.ob_computer = OrderbookFeatureComputer(config.db_path)
        else:
            self.ob_computer = None

    def fit_volatility_filter(self, df: pd.DataFrame) -> float:
        """Compute volatility threshold from data."""
        self.volatility_threshold = np.percentile(
            df[self.config.volatility_col].dropna(),
            self.config.volatility_percentile_threshold
        )
        logger.info(f"Volatility threshold (P{self.config.volatility_percentile_threshold}): {self.volatility_threshold:.4f}")
        return self.volatility_threshold

    def filter_by_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to high-volatility periods."""
        if self.volatility_threshold is None:
            raise ValueError("Must call fit_volatility_filter first")
        return df[df[self.config.volatility_col] >= self.volatility_threshold].copy()

    def prepare_features(self, df: pd.DataFrame, add_orderbook: bool = True) -> pd.DataFrame:
        """Prepare feature matrix."""
        data = df.copy()

        # Add orderbook features if enabled
        if add_orderbook and self.ob_computer is not None:
            ob_features = self.ob_computer.batch_compute(data)
            data = pd.concat([data.reset_index(drop=True), ob_features], axis=1)

        # Select features
        feature_cols = self.CATALYST_FEATURES.copy()
        if self.config.use_orderbook_features:
            feature_cols.extend(self.ORDERBOOK_FEATURES)

        for col in self.BOOL_FEATURES:
            if col in data.columns:
                feature_cols.append(col)
                data[col] = data[col].astype(int)

        available = [f for f in feature_cols if f in data.columns]
        X = data[available].copy()

        # Fill NaN with median
        for col in X.columns:
            if X[col].isna().any():
                median = X[col].median()
                X[col] = X[col].fillna(median if not pd.isna(median) else 0)

        return X

    def _get_models(self, n_pos: int, n_neg: int) -> Dict[str, Any]:
        """Get dictionary of models."""
        scale_pos = n_neg / n_pos if n_pos > 0 else 1
        models = {}

        if HAS_LIGHTGBM:
            models['lightgbm'] = lgb.LGBMClassifier(
                objective='binary', num_leaves=31, learning_rate=0.05,
                n_estimators=100, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=5, scale_pos_weight=scale_pos, verbose=-1, random_state=42
            )

        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBClassifier(
                objective='binary:logistic', max_depth=5, learning_rate=0.05,
                n_estimators=100, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos, eval_metric='auc',
                use_label_encoder=False, verbosity=0, random_state=42
            )

        models['randomforest'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=5, class_weight={0: 1, 1: scale_pos},
            random_state=42, n_jobs=-1
        )

        models['gradboost'] = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        )

        return models

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train ensemble models."""
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos

        logger.info(f"Training ensemble on {len(X_train)} samples ({n_pos} positive)")

        self.models = self._get_models(n_pos, n_neg)

        for name, model in self.models.items():
            model.fit(X_train, y_train)

        # Compute weights from validation performance
        if X_val is not None and y_val is not None:
            aucs = {}
            for name, model in self.models.items():
                try:
                    proba = model.predict_proba(X_val)[:, 1]
                    aucs[name] = roc_auc_score(y_val, proba) if y_val.sum() > 0 else 0.5
                except:
                    aucs[name] = 0.5

            adjusted = {k: max(v - 0.5, 0.01) for k, v in aucs.items()}
            total = sum(adjusted.values())
            self.model_weights = {k: v / total for k, v in adjusted.items()}
        else:
            self.model_weights = {name: 1.0 / len(self.models) for name in self.models}

        logger.info(f"Model weights: {self.model_weights}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble probability predictions."""
        predictions = []
        weights = []

        for name, model in self.models.items():
            proba = model.predict_proba(X)[:, 1]
            predictions.append(proba)
            weights.append(self.model_weights.get(name, 1.0))

        return np.average(np.array(predictions), axis=0, weights=np.array(weights))


# =============================================================================
# RL POSITION SIZER
# =============================================================================

if HAS_TORCH:
    class QNetworkV4(nn.Module):
        """Q-network for RL position sizing."""

        def __init__(self, state_dim: int, n_actions: int, hidden_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.bn1 = nn.LayerNorm(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = nn.LayerNorm(hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.bn3 = nn.LayerNorm(hidden_dim // 2)
            self.out = nn.Linear(hidden_dim // 2, n_actions)

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
            return self.out(x)


class RLPositionSizer:
    """RL-based position sizer using trained DQN agent."""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.rl_config = RLConfigV4()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else None
        self.q_net = None

    def load_model(self, path: str = None):
        """Load trained RL model."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required for RL position sizing")

        path = path or self.config.rl_model_path

        self.q_net = QNetworkV4(
            self.rl_config.state_dim,
            self.rl_config.n_actions,
            self.rl_config.hidden_dim
        ).to(self.device)

        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.q_net.eval()

        logger.info(f"Loaded RL model from {path}")

    def build_state(
        self,
        signal: pd.Series,
        in_position: bool = False,
        hold_hours: float = 0,
        unrealized_pnl: float = 0,
        trailing_active: bool = False,
        n_trades: int = 0,
        total_pnl: float = 0,
    ) -> np.ndarray:
        """Build state vector for RL agent."""
        # Compute momentum rank (approximation)
        momentum = signal.get('momentum_4h', 0)
        momentum_rank = 0.5 + (np.clip(momentum, -3, 3) / 6)  # Approximate rank

        state = np.array([
            signal.get('y_pred_proba', signal.get('ensemble_proba', 0.5)),
            min(signal.get('volatility_4h', 0.5), 3.0) / 3.0,
            np.clip(momentum, -5, 5) / 5.0,
            momentum_rank,
            min(signal.get('volume_ratio', 1.0), 3.0) / 3.0,
            signal.get('rsi_proxy', 50) / 100.0,
            float(in_position),
            min(hold_hours / 24.0, 1.0),
            np.clip(unrealized_pnl / 10.0, -1, 1),
            float(trailing_active),
            min(n_trades / 10.0, 1.0),
            total_pnl / self.config.initial_capital,
        ], dtype=np.float32)

        return state

    def get_position_size(self, state: np.ndarray) -> Tuple[int, float]:
        """Get position size from RL agent."""
        if self.q_net is None:
            raise ValueError("Must load model first")

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            action = q_values.argmax(dim=1).item()

        return action, self.config.position_sizes[action]


# =============================================================================
# INTEGRATED BACKTESTER
# =============================================================================

@dataclass
class Trade:
    """Trade record."""
    symbol: str
    entry_time: datetime
    entry_price: float
    position_size_usd: float
    position_size_pct: float
    pred_prob: float
    rl_action: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    max_favorable: float = 0.0
    trailing_activated: bool = False


@dataclass
class BacktestResult:
    """Backtest results."""
    system_name: str = ""
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    avg_pnl_per_trade: float = 0.0
    avg_position_pct: float = 0.0
    n_signals: int = 0
    n_filtered: int = 0
    n_skipped_by_rl: int = 0
    exits_tp: int = 0
    exits_sl: int = 0
    exits_trail: int = 0
    exits_time: int = 0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class IntegratedBacktester:
    """
    Integrated backtester combining:
    1. Volatility filter
    2. Ensemble spike predictor
    3. RL position sizer
    """

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.conn = duckdb.connect(config.db_path, read_only=True)

        self.ensemble = EnsembleSpikePredictor(config)
        self.rl_sizer = RLPositionSizer(config)

    def _get_price(self, symbol: str, time: datetime) -> Optional[float]:
        """Get price from database."""
        try:
            query = f"""
                SELECT close FROM ohlcv
                WHERE symbol = '{symbol}' AND timeframe = '5m'
                  AND timestamp >= '{time - timedelta(minutes=30)}'
                  AND timestamp <= '{time + timedelta(minutes=30)}'
                ORDER BY ABS(EPOCH(timestamp) - EPOCH(TIMESTAMP '{time}'))
                LIMIT 1
            """
            result = self.conn.execute(query).fetchone()
            return result[0] if result else None
        except:
            return None

    def _get_price_extremes(
        self, symbol: str, start: datetime, end: datetime
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get max high and min low in time range."""
        try:
            query = f"""
                SELECT MAX(high), MIN(low) FROM ohlcv
                WHERE symbol = '{symbol}' AND timeframe = '5m'
                  AND timestamp >= '{start}' AND timestamp <= '{end}'
            """
            result = self.conn.execute(query).fetchone()
            return (result[0], result[1]) if result else (None, None)
        except:
            return (None, None)

    def run_integrated_backtest(
        self,
        whale_features_csv: str,
        target_col: str = 'spike_10pct_24h',
        train_ratio: float = 0.6,
    ) -> BacktestResult:
        """
        Run full integrated backtest.

        Pipeline:
        1. Load data
        2. Split into train/test
        3. Train ensemble on volatility-filtered train data
        4. Load RL model
        5. Run backtest on test data
        """
        result = BacktestResult(system_name="Integrated (Ensemble + Vol Filter + RL)")

        # Load data
        df = pd.read_csv(whale_features_csv, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        result.n_signals = len(df)

        logger.info(f"Loaded {len(df)} signals")

        # Train/test split
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

        # Fit volatility filter on training data
        self.ensemble.fit_volatility_filter(train_df)

        # Filter training data
        train_filtered = self.ensemble.filter_by_volatility(train_df)
        logger.info(f"Filtered train: {len(train_filtered)} ({len(train_filtered)/len(train_df)*100:.1f}%)")

        # Prepare features and train ensemble
        X_train = self.ensemble.prepare_features(train_filtered, add_orderbook=True)
        y_train = train_filtered[target_col].astype(int)

        # Split for validation
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
        y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

        self.ensemble.train(X_tr, y_tr, X_val, y_val)

        # Load RL model
        self.rl_sizer.load_model()

        # Filter test data by volatility
        test_filtered = self.ensemble.filter_by_volatility(test_df)
        result.n_filtered = len(test_filtered)
        logger.info(f"Filtered test: {len(test_filtered)} ({len(test_filtered)/len(test_df)*100:.1f}%)")

        # Prepare test features
        X_test = self.ensemble.prepare_features(test_filtered, add_orderbook=True)

        # Get ensemble predictions
        test_filtered = test_filtered.reset_index(drop=True)
        test_filtered['ensemble_proba'] = self.ensemble.predict_proba(X_test)

        logger.info(f"Ensemble predictions: mean={test_filtered['ensemble_proba'].mean():.3f}")

        # Run backtest
        self._execute_backtest(test_filtered, result)

        return result

    def _execute_backtest(self, signals_df: pd.DataFrame, result: BacktestResult):
        """Execute backtest with RL position sizing."""
        capital = self.config.initial_capital
        peak_capital = capital
        open_positions: Dict[str, Trade] = {}
        equity_curve = [capital]
        position_sizes = []

        signals_df = signals_df.sort_values('timestamp')

        for idx, signal in signals_df.iterrows():
            signal_time = signal['timestamp']
            symbol = signal['symbol']

            # Check exits for open positions
            closed = self._check_exits(open_positions, signal_time, result)
            for trade in closed:
                capital += trade.pnl_usd + trade.position_size_usd
                result.trades.append(trade)

            # Update equity
            current_equity = capital + sum(t.position_size_usd for t in open_positions.values())
            equity_curve.append(current_equity)
            peak_capital = max(peak_capital, current_equity)

            # Skip if at max positions or already in symbol
            if len(open_positions) >= self.config.max_concurrent_positions:
                continue
            if symbol in open_positions:
                continue

            # Check ensemble confidence
            ensemble_prob = signal.get('ensemble_proba', 0)
            if ensemble_prob < self.config.min_ensemble_confidence:
                continue

            # Get RL decision
            state = self.rl_sizer.build_state(
                signal,
                n_trades=len(result.trades),
                total_pnl=sum(t.pnl_usd for t in result.trades),
            )
            rl_action, position_size_pct = self.rl_sizer.get_position_size(state)

            if rl_action == 0:
                result.n_skipped_by_rl += 1
                continue

            # Calculate position size
            max_pos = capital * (self.config.max_position_pct / 100)
            position_value = max_pos * position_size_pct

            if position_value < 50:
                continue

            # Get entry price
            entry_price = self._get_price(symbol, signal_time + timedelta(minutes=5))
            if entry_price is None:
                continue

            # Apply slippage
            entry_price *= (1 + self.config.slippage_pct / 100)

            # Create trade
            trade = Trade(
                symbol=symbol,
                entry_time=signal_time,
                entry_price=entry_price,
                position_size_usd=position_value,
                position_size_pct=position_size_pct,
                pred_prob=ensemble_prob,
                rl_action=rl_action,
            )
            open_positions[symbol] = trade
            capital -= position_value
            position_sizes.append(position_size_pct)

        # Close remaining positions
        if len(signals_df) > 0:
            end_time = signals_df['timestamp'].max() + timedelta(hours=1)
            for symbol, trade in list(open_positions.items()):
                self._close_trade(trade, end_time, "end", result)
                capital += trade.pnl_usd + trade.position_size_usd
                result.trades.append(trade)

        equity_curve.append(capital)
        result.equity_curve = equity_curve

        # Calculate metrics
        self._calc_metrics(result, position_sizes)

    def _check_exits(
        self,
        positions: Dict[str, Trade],
        current_time: datetime,
        result: BacktestResult,
    ) -> List[Trade]:
        """Check and execute exits with trailing stops."""
        closed = []

        for symbol, trade in list(positions.items()):
            max_price, min_price = self._get_price_extremes(
                symbol, trade.entry_time, current_time
            )
            current_price = self._get_price(symbol, current_time)

            if current_price is None:
                continue

            # Calculate returns
            pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
            max_pnl = ((max_price - trade.entry_price) / trade.entry_price) * 100 if max_price else pnl_pct
            min_pnl = ((min_price - trade.entry_price) / trade.entry_price) * 100 if min_price else pnl_pct

            trade.max_favorable = max(trade.max_favorable, max_pnl)
            hold_hours = (current_time - trade.entry_time).total_seconds() / 3600

            # Check trailing stop activation
            if trade.max_favorable >= self.config.trailing_stop_activation:
                trade.trailing_activated = True

            # Exit conditions
            exit_reason = None
            exit_pnl = None

            # SL hit first
            if min_pnl <= -self.config.stop_loss_pct:
                exit_reason = "sl"
                exit_pnl = -self.config.stop_loss_pct
            # Ratcheting trailing stop (replaces fixed TP)
            elif trade.trailing_activated:
                # Determine current trail distance based on max gain (ratcheting)
                current_trail = 1.2  # Default for activation zone (2-6%)
                for level_pct, trail_dist in self.config.ratchet_levels:
                    if trade.max_favorable >= level_pct:
                        current_trail = trail_dist

                trailing_level = trade.max_favorable - current_trail
                if pnl_pct <= trailing_level:
                    exit_reason = "trail"
                    exit_pnl = trailing_level
            # Time limit
            elif hold_hours >= self.config.max_hold_hours:
                exit_reason = "time"
                exit_pnl = pnl_pct

            if exit_reason:
                self._close_trade(trade, current_time, exit_reason, result, exit_pnl)
                closed.append(trade)
                del positions[symbol]

        return closed

    def _close_trade(
        self,
        trade: Trade,
        exit_time: datetime,
        reason: str,
        result: BacktestResult,
        override_pnl: float = None,
    ):
        """Close a trade."""
        exit_price = self._get_price(trade.symbol, exit_time)
        if exit_price is None:
            exit_price = trade.entry_price

        exit_price *= (1 - self.config.slippage_pct / 100)

        if override_pnl is not None:
            gross_pnl = override_pnl
        else:
            gross_pnl = ((exit_price - trade.entry_price) / trade.entry_price) * 100

        net_pnl = gross_pnl - (2 * self.config.fee_pct)

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl_pct = net_pnl
        trade.pnl_usd = trade.position_size_usd * (net_pnl / 100)

        if reason == "tp":
            result.exits_tp += 1
        elif reason == "sl":
            result.exits_sl += 1
        elif reason == "trail":
            result.exits_trail += 1
        elif reason == "time":
            result.exits_time += 1

    def _calc_metrics(self, result: BacktestResult, position_sizes: List[float]):
        """Calculate performance metrics."""
        if not result.trades:
            return

        result.n_trades = len(result.trades)

        wins = [t for t in result.trades if t.pnl_pct > 0]
        losses = [t for t in result.trades if t.pnl_pct <= 0]

        result.n_wins = len(wins)
        result.win_rate = len(wins) / len(result.trades) * 100

        result.total_pnl = sum(t.pnl_usd for t in result.trades)
        result.total_pnl_pct = result.total_pnl / self.config.initial_capital * 100
        result.avg_pnl_per_trade = result.total_pnl / result.n_trades

        gross_profit = sum(t.pnl_usd for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        if position_sizes:
            result.avg_position_pct = np.mean(position_sizes) * 100

        # Max drawdown
        equity = pd.Series(result.equity_curve)
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        result.max_drawdown = abs(drawdown.min()) * 100

        # Sharpe ratio
        returns = equity.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            result.sharpe = (returns.mean() / returns.std()) * np.sqrt(252)


# =============================================================================
# BASELINE BACKTESTER (for comparison)
# =============================================================================

class BaselineBacktester:
    """Baseline backtester: Single LightGBM + fixed sizing."""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.conn = duckdb.connect(config.db_path, read_only=True)
        self.model = None

    def _get_price(self, symbol: str, time: datetime) -> Optional[float]:
        try:
            query = f"""
                SELECT close FROM ohlcv
                WHERE symbol = '{symbol}' AND timeframe = '5m'
                  AND timestamp >= '{time - timedelta(minutes=30)}'
                  AND timestamp <= '{time + timedelta(minutes=30)}'
                ORDER BY ABS(EPOCH(timestamp) - EPOCH(TIMESTAMP '{time}'))
                LIMIT 1
            """
            result = self.conn.execute(query).fetchone()
            return result[0] if result else None
        except:
            return None

    def _get_price_extremes(
        self, symbol: str, start: datetime, end: datetime
    ) -> Tuple[Optional[float], Optional[float]]:
        try:
            query = f"""
                SELECT MAX(high), MIN(low) FROM ohlcv
                WHERE symbol = '{symbol}' AND timeframe = '5m'
                  AND timestamp >= '{start}' AND timestamp <= '{end}'
            """
            result = self.conn.execute(query).fetchone()
            return (result[0], result[1]) if result else (None, None)
        except:
            return (None, None)

    def run_baseline_backtest(
        self,
        whale_features_csv: str,
        target_col: str = 'spike_10pct_24h',
        train_ratio: float = 0.6,
        fixed_position_pct: float = 0.5,
        min_prob_threshold: float = 0.3,
    ) -> BacktestResult:
        """Run baseline backtest with single model and fixed sizing."""
        result = BacktestResult(system_name="Baseline (LightGBM + Fixed 50%)")

        # Load data
        df = pd.read_csv(whale_features_csv, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        result.n_signals = len(df)

        # Train/test split
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # Prepare features (no volatility filter, no orderbook)
        feature_cols = [
            'event_priority', 'sentiment_score', 'log_usd_value',
            'volatility_4h', 'momentum_4h', 'volume_ratio', 'rsi_proxy',
        ]
        bool_cols = ['is_bearish_flow', 'is_bullish_flow', 'has_direction']

        for col in bool_cols:
            if col in train_df.columns:
                feature_cols.append(col)
                train_df[col] = train_df[col].astype(int)
                test_df[col] = test_df[col].astype(int)

        available = [f for f in feature_cols if f in train_df.columns]

        X_train = train_df[available].copy()
        y_train = train_df[target_col].astype(int)
        X_test = test_df[available].copy()

        # Fill NaN
        for col in X_train.columns:
            median = X_train[col].median()
            X_train[col] = X_train[col].fillna(median if not pd.isna(median) else 0)
            X_test[col] = X_test[col].fillna(median if not pd.isna(median) else 0)

        # Train single LightGBM
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos

        if HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                objective='binary', num_leaves=31, learning_rate=0.05,
                n_estimators=100, scale_pos_weight=n_neg/n_pos if n_pos > 0 else 1,
                verbose=-1, random_state=42
            )
            self.model.fit(X_train, y_train)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

        # Get predictions
        test_df = test_df.reset_index(drop=True)
        test_df['pred_proba'] = self.model.predict_proba(X_test)[:, 1]

        result.n_filtered = len(test_df)  # No filter

        # Run backtest
        self._execute_baseline_backtest(test_df, result, fixed_position_pct, min_prob_threshold)

        return result

    def _execute_baseline_backtest(
        self,
        signals_df: pd.DataFrame,
        result: BacktestResult,
        fixed_position_pct: float,
        min_prob: float,
    ):
        """Execute baseline backtest."""
        capital = self.config.initial_capital
        peak_capital = capital
        open_positions: Dict[str, Trade] = {}
        equity_curve = [capital]

        signals_df = signals_df.sort_values('timestamp')

        for idx, signal in signals_df.iterrows():
            signal_time = signal['timestamp']
            symbol = signal['symbol']

            # Check exits
            closed = self._check_exits_baseline(open_positions, signal_time, result)
            for trade in closed:
                capital += trade.pnl_usd + trade.position_size_usd
                result.trades.append(trade)

            current_equity = capital + sum(t.position_size_usd for t in open_positions.values())
            equity_curve.append(current_equity)
            peak_capital = max(peak_capital, current_equity)

            if len(open_positions) >= self.config.max_concurrent_positions:
                continue
            if symbol in open_positions:
                continue

            # Simple threshold
            if signal.get('pred_proba', 0) < min_prob:
                continue

            # Fixed position size
            max_pos = capital * (self.config.max_position_pct / 100)
            position_value = max_pos * fixed_position_pct

            if position_value < 50:
                continue

            entry_price = self._get_price(symbol, signal_time + timedelta(minutes=5))
            if entry_price is None:
                continue

            entry_price *= (1 + self.config.slippage_pct / 100)

            trade = Trade(
                symbol=symbol,
                entry_time=signal_time,
                entry_price=entry_price,
                position_size_usd=position_value,
                position_size_pct=fixed_position_pct,
                pred_prob=signal.get('pred_proba', 0),
                rl_action=-1,  # No RL
            )
            open_positions[symbol] = trade
            capital -= position_value

        # Close remaining
        if len(signals_df) > 0:
            end_time = signals_df['timestamp'].max() + timedelta(hours=1)
            for symbol, trade in list(open_positions.items()):
                self._close_trade_baseline(trade, end_time, "end", result)
                capital += trade.pnl_usd + trade.position_size_usd
                result.trades.append(trade)

        equity_curve.append(capital)
        result.equity_curve = equity_curve

        # Calculate metrics
        self._calc_metrics_baseline(result)

    def _check_exits_baseline(
        self,
        positions: Dict[str, Trade],
        current_time: datetime,
        result: BacktestResult,
    ) -> List[Trade]:
        """Check exits without trailing stops."""
        closed = []

        for symbol, trade in list(positions.items()):
            max_price, min_price = self._get_price_extremes(
                symbol, trade.entry_time, current_time
            )
            current_price = self._get_price(symbol, current_time)

            if current_price is None:
                continue

            pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
            max_pnl = ((max_price - trade.entry_price) / trade.entry_price) * 100 if max_price else pnl_pct
            min_pnl = ((min_price - trade.entry_price) / trade.entry_price) * 100 if min_price else pnl_pct

            hold_hours = (current_time - trade.entry_time).total_seconds() / 3600

            exit_reason = None
            exit_pnl = None

            if max_pnl >= self.config.take_profit_pct:
                exit_reason = "tp"
                exit_pnl = self.config.take_profit_pct
            elif min_pnl <= -self.config.stop_loss_pct:
                exit_reason = "sl"
                exit_pnl = -self.config.stop_loss_pct
            elif hold_hours >= self.config.max_hold_hours:
                exit_reason = "time"
                exit_pnl = pnl_pct

            if exit_reason:
                self._close_trade_baseline(trade, current_time, exit_reason, result, exit_pnl)
                closed.append(trade)
                del positions[symbol]

        return closed

    def _close_trade_baseline(
        self,
        trade: Trade,
        exit_time: datetime,
        reason: str,
        result: BacktestResult,
        override_pnl: float = None,
    ):
        exit_price = self._get_price(trade.symbol, exit_time) or trade.entry_price
        exit_price *= (1 - self.config.slippage_pct / 100)

        if override_pnl is not None:
            gross_pnl = override_pnl
        else:
            gross_pnl = ((exit_price - trade.entry_price) / trade.entry_price) * 100

        net_pnl = gross_pnl - (2 * self.config.fee_pct)

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl_pct = net_pnl
        trade.pnl_usd = trade.position_size_usd * (net_pnl / 100)

        if reason == "tp":
            result.exits_tp += 1
        elif reason == "sl":
            result.exits_sl += 1
        elif reason == "time":
            result.exits_time += 1

    def _calc_metrics_baseline(self, result: BacktestResult):
        if not result.trades:
            return

        result.n_trades = len(result.trades)
        wins = [t for t in result.trades if t.pnl_pct > 0]
        losses = [t for t in result.trades if t.pnl_pct <= 0]

        result.n_wins = len(wins)
        result.win_rate = len(wins) / len(result.trades) * 100

        result.total_pnl = sum(t.pnl_usd for t in result.trades)
        result.total_pnl_pct = result.total_pnl / self.config.initial_capital * 100
        result.avg_pnl_per_trade = result.total_pnl / result.n_trades

        gross_profit = sum(t.pnl_usd for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        result.avg_position_pct = 50.0  # Fixed

        equity = pd.Series(result.equity_curve)
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        result.max_drawdown = abs(drawdown.min()) * 100

        returns = equity.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            result.sharpe = (returns.mean() / returns.std()) * np.sqrt(252)


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_full_comparison():
    """Run full comparison between baseline and integrated system."""
    print("=" * 80)
    print("PRODUCTION SYSTEM COMPARISON")
    print("Baseline (LightGBM + Fixed) vs Integrated (Ensemble + Vol Filter + RL)")
    print("=" * 80)

    config = ProductionConfig()
    whale_csv = "/Users/bz/Pythia2/whale_features.csv"

    # Run baseline
    print("\n" + "-" * 80)
    print("1. BASELINE SYSTEM")
    print("-" * 80)

    baseline = BaselineBacktester(config)
    baseline_result = baseline.run_baseline_backtest(whale_csv)

    print(f"\n  System: {baseline_result.system_name}")
    print(f"  Signals: {baseline_result.n_signals}")
    print(f"  Trades: {baseline_result.n_trades}")
    print(f"  Win Rate: {baseline_result.win_rate:.1f}%")
    print(f"  Total PnL: ${baseline_result.total_pnl:+,.2f} ({baseline_result.total_pnl_pct:+.1f}%)")
    print(f"  Avg PnL/Trade: ${baseline_result.avg_pnl_per_trade:+.2f}")
    print(f"  Profit Factor: {baseline_result.profit_factor:.2f}")
    print(f"  Max Drawdown: {baseline_result.max_drawdown:.1f}%")
    print(f"  Sharpe Ratio: {baseline_result.sharpe:.2f}")
    print(f"  Exits: TP={baseline_result.exits_tp}, SL={baseline_result.exits_sl}, Time={baseline_result.exits_time}")

    # Run integrated system
    print("\n" + "-" * 80)
    print("2. INTEGRATED SYSTEM")
    print("-" * 80)

    integrated = IntegratedBacktester(config)
    integrated_result = integrated.run_integrated_backtest(whale_csv)

    print(f"\n  System: {integrated_result.system_name}")
    print(f"  Signals: {integrated_result.n_signals}")
    print(f"  After Vol Filter: {integrated_result.n_filtered}")
    print(f"  Skipped by RL: {integrated_result.n_skipped_by_rl}")
    print(f"  Trades: {integrated_result.n_trades}")
    print(f"  Win Rate: {integrated_result.win_rate:.1f}%")
    print(f"  Total PnL: ${integrated_result.total_pnl:+,.2f} ({integrated_result.total_pnl_pct:+.1f}%)")
    print(f"  Avg PnL/Trade: ${integrated_result.avg_pnl_per_trade:+.2f}")
    print(f"  Profit Factor: {integrated_result.profit_factor:.2f}")
    print(f"  Max Drawdown: {integrated_result.max_drawdown:.1f}%")
    print(f"  Sharpe Ratio: {integrated_result.sharpe:.2f}")
    print(f"  Avg Position Size: {integrated_result.avg_position_pct:.1f}%")
    print(f"  Exits: TP={integrated_result.exits_tp}, SL={integrated_result.exits_sl}, "
          f"Trail={integrated_result.exits_trail}, Time={integrated_result.exits_time}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'Baseline':>15} {'Integrated':>15} {'Improvement':>15}")
    print("-" * 70)

    metrics = [
        ("Trades", baseline_result.n_trades, integrated_result.n_trades, None),
        ("Win Rate (%)", baseline_result.win_rate, integrated_result.win_rate,
         integrated_result.win_rate - baseline_result.win_rate),
        ("Total PnL ($)", baseline_result.total_pnl, integrated_result.total_pnl,
         integrated_result.total_pnl - baseline_result.total_pnl),
        ("Avg PnL/Trade ($)", baseline_result.avg_pnl_per_trade, integrated_result.avg_pnl_per_trade,
         integrated_result.avg_pnl_per_trade - baseline_result.avg_pnl_per_trade),
        ("Profit Factor", baseline_result.profit_factor, integrated_result.profit_factor,
         integrated_result.profit_factor - baseline_result.profit_factor),
        ("Max Drawdown (%)", baseline_result.max_drawdown, integrated_result.max_drawdown,
         baseline_result.max_drawdown - integrated_result.max_drawdown),  # Lower is better
        ("Sharpe Ratio", baseline_result.sharpe, integrated_result.sharpe,
         integrated_result.sharpe - baseline_result.sharpe),
    ]

    for name, base_val, int_val, improvement in metrics:
        if improvement is not None:
            sign = "+" if improvement >= 0 else ""
            print(f"{name:<25} {base_val:>15.2f} {int_val:>15.2f} {sign}{improvement:>14.2f}")
        else:
            print(f"{name:<25} {base_val:>15} {int_val:>15} {'N/A':>15}")

    # Save results
    print("\n" + "-" * 80)
    print("SAVING RESULTS")
    print("-" * 80)

    # Save trade details
    if integrated_result.trades:
        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'position_size_usd': t.position_size_usd,
            'position_size_pct': t.position_size_pct,
            'pred_prob': t.pred_prob,
            'rl_action': t.rl_action,
            'exit_reason': t.exit_reason,
            'pnl_pct': t.pnl_pct,
            'pnl_usd': t.pnl_usd,
            'trailing_activated': t.trailing_activated,
        } for t in integrated_result.trades])
        trades_df.to_csv("/Users/bz/Pythia2/integrated_backtest_trades.csv", index=False)
        print("  Trades saved to: /Users/bz/Pythia2/integrated_backtest_trades.csv")

    # Save equity curves
    equity_df = pd.DataFrame({
        'step': range(len(integrated_result.equity_curve)),
        'integrated_equity': integrated_result.equity_curve,
    })
    if len(baseline_result.equity_curve) == len(integrated_result.equity_curve):
        equity_df['baseline_equity'] = baseline_result.equity_curve
    equity_df.to_csv("/Users/bz/Pythia2/equity_curves.csv", index=False)
    print("  Equity curves saved to: /Users/bz/Pythia2/equity_curves.csv")

    return baseline_result, integrated_result


if __name__ == "__main__":
    run_full_comparison()
