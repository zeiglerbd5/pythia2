#!/usr/bin/env python3
"""
Backtest Event Classifier Strategies

Tests XGBoost event classifier models with various entry thresholds and exit strategies.
Uses historical OHLCV data from FeatureBuffer (SQLite) to simulate paper trading.

Usage:
    python scripts/backtest_event_classifier.py
    python scripts/backtest_event_classifier.py --model models/xgboost_full_dataset/model.pkl
    python scripts/backtest_event_classifier.py --threshold 0.85 --stop-loss 0.02
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import argparse
import sqlite3
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

FEATURE_BUFFER_PATH = '/Users/bz/Pythia2/data/feature_buffer.db'

# Default model paths to test
DEFAULT_MODELS = [
    '/Users/bz/Pythia2/models/event_classifier_xgb.pkl',
    '/Users/bz/Pythia2/models/xgboost_full_dataset/model.pkl',
]

# Event classifier feature columns (26 features)
FEATURE_COLS = [
    'natr_14', 'bb_width_20', 'bb_position', 'rsi_14',
    'returns_1hr', 'returns_6hr', 'returns_12hr',
    'momentum_5', 'momentum_20',
    'dist_from_24hr_high', 'dist_from_24hr_low',
    'hl_range', 'body_ratio_avg_1hr', 'range_compression',
    'volume_vs_ma20', 'volume_trend_6hr', 'obv_slope_1hr', 'vroc_12',
    'vol_ratio_20_60', 'vol_ratio_60_240', 'vol_acceleration', 'vol_price_divergence',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
]


@dataclass
class Position:
    """Tracks a single position"""
    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: float
    position_size: float = 2500.0
    peak_price: float = 0.0
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl: float = 0.0

    def __post_init__(self):
        self.peak_price = self.entry_price


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    model_path: str
    threshold: float = 0.80
    stop_loss_pct: float = 0.01  # 1% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    trailing_stop_pct: float = 0.02  # 2% trailing stop after activation
    trailing_activation_pct: float = 0.05  # Activate trailing at 5% profit
    max_hold_minutes: int = 180  # 3 hours max hold
    position_size: float = 2500.0
    max_positions: int = 4
    min_volume_ratio: float = 1.0  # Minimum vol_ratio_20_60 to enter
    require_momentum: bool = False  # Require positive momentum to enter


@dataclass
class BacktestResult:
    """Backtest results"""
    config: BacktestConfig
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[Dict] = field(default_factory=list)


class EventClassifierBacktester:
    """Backtests event classifier strategies on historical data"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self._load_model()

    def _load_model(self):
        """Load the XGBoost model and scaler"""
        try:
            model_data = joblib.load(self.config.model_path)
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.scaler = model_data.get('scaler')
                self.feature_cols = model_data.get('feature_cols', FEATURE_COLS)
            else:
                self.model = model_data
                self.feature_cols = FEATURE_COLS
            logger.info(f"Loaded model: {Path(self.config.model_path).name} ({len(self.feature_cols)} features)")
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_path}: {e}")
            raise

    def load_ohlcv_data(self) -> pd.DataFrame:
        """Load OHLCV data from FeatureBuffer SQLite"""
        conn = sqlite3.connect(FEATURE_BUFFER_PATH)

        query = """
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM ohlcv
            ORDER BY symbol, timestamp
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)  # Remove timezone for easier comparison
        logger.info(f"Loaded {len(df)} OHLCV rows for {df['symbol'].nunique()} symbols")

        return df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute event classifier features from OHLCV data"""
        features_list = []

        for symbol in df['symbol'].unique():
            sym_df = df[df['symbol'] == symbol].copy()
            sym_df = sym_df.sort_values('timestamp').reset_index(drop=True)

            if len(sym_df) < 1440:  # Need at least 24 hours
                continue

            # Compute features for each row (after warmup)
            for i in range(1440, len(sym_df)):
                try:
                    features = self._compute_single_features(sym_df, i)
                    if features:
                        features['symbol'] = symbol
                        features['timestamp'] = sym_df.iloc[i]['timestamp']
                        features['close'] = sym_df.iloc[i]['close']
                        features_list.append(features)
                except Exception as e:
                    continue

        if not features_list:
            logger.warning("No features computed!")
            return pd.DataFrame()

        result = pd.DataFrame(features_list)
        logger.info(f"Computed features for {len(result)} data points across {result['symbol'].nunique()} symbols")
        return result

    def _compute_single_features(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        """Compute features for a single timestamp"""
        try:
            close = df['close'].iloc[:idx+1]
            high = df['high'].iloc[:idx+1]
            low = df['low'].iloc[:idx+1]
            volume = df['volume'].iloc[:idx+1]

            current_close = close.iloc[-1]

            # NATR (14-period)
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr_14 = tr.iloc[-14:].mean()
            natr_14 = (atr_14 / current_close) * 100 if current_close > 0 else 0

            # Bollinger Bands (20-period)
            sma_20 = close.iloc[-20:].mean()
            std_20 = close.iloc[-20:].std()
            bb_width_20 = (2 * std_20 / sma_20) if sma_20 > 0 else 0
            bb_position = (current_close - (sma_20 - 2*std_20)) / (4*std_20) if std_20 > 0 else 0.5

            # RSI (14-period)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).iloc[-14:].mean()
            loss = (-delta.where(delta < 0, 0)).iloc[-14:].mean()
            rs = gain / loss if loss > 0 else 100
            rsi_14 = 100 - (100 / (1 + rs))

            # Returns at horizons
            returns_1hr = (current_close / close.iloc[-60] - 1) if len(close) > 60 else 0
            returns_6hr = (current_close / close.iloc[-360] - 1) if len(close) > 360 else 0
            returns_12hr = (current_close / close.iloc[-720] - 1) if len(close) > 720 else 0

            # Momentum
            momentum_5 = (current_close / close.iloc[-5] - 1) if len(close) > 5 else 0
            momentum_20 = (current_close / close.iloc[-20] - 1) if len(close) > 20 else 0

            # Distance from 24hr high/low
            high_24hr = high.iloc[-1440:].max()
            low_24hr = low.iloc[-1440:].min()
            dist_from_24hr_high = (current_close - high_24hr) / high_24hr if high_24hr > 0 else 0
            dist_from_24hr_low = (current_close - low_24hr) / low_24hr if low_24hr > 0 else 0

            # HL Range
            hl_range = (high_24hr - low_24hr) / current_close if current_close > 0 else 0

            # Body ratio
            body = (close - df['open'].iloc[:idx+1]).abs()
            wick = high - low
            body_ratio = body / (wick + 1e-10)
            body_ratio_avg_1hr = body_ratio.iloc[-60:].mean() if len(body_ratio) >= 60 else 0.5

            # Range compression
            range_20 = (high.rolling(20).max() - low.rolling(20).min()) / current_close
            range_60 = (high.rolling(60).max() - low.rolling(60).min()) / current_close
            r20 = range_20.iloc[-1] if pd.notna(range_20.iloc[-1]) else 0
            r60 = range_60.iloc[-1] if pd.notna(range_60.iloc[-1]) else 1
            range_compression = r20 / r60 if r60 > 0 else 1

            # Volume features
            vol_ma_20 = volume.iloc[-20:].mean()
            volume_vs_ma20 = volume.iloc[-1] / vol_ma_20 if vol_ma_20 > 0 else 1

            vol_ma_60 = volume.iloc[-60:].mean()
            vol_ma_360 = volume.iloc[-360:].mean() if len(volume) >= 360 else vol_ma_60
            volume_trend_6hr = vol_ma_60 / vol_ma_360 if vol_ma_360 > 0 else 1

            # OBV slope
            obv = (volume * np.sign(close.diff())).cumsum()
            obv_slope_1hr = (obv.iloc[-1] - obv.iloc[-60]) / 60 if len(obv) >= 60 else 0

            # VROC
            vroc_12 = (volume.iloc[-1] / volume.iloc[-12] - 1) if len(volume) > 12 and volume.iloc[-12] > 0 else 0

            # Volume ratios
            vol_20 = volume.iloc[-20:].mean()
            vol_60 = volume.iloc[-60:].mean()
            vol_240 = volume.iloc[-240:].mean() if len(volume) >= 240 else vol_60
            vol_ratio_20_60 = vol_20 / vol_60 if vol_60 > 0 else 1
            vol_ratio_60_240 = vol_60 / vol_240 if vol_240 > 0 else 1

            # Volume acceleration
            vol_5 = volume.iloc[-5:].mean()
            vol_10 = volume.iloc[-10:].mean()
            vol_acceleration = (vol_5 / vol_10 - 1) if vol_10 > 0 else 0

            # Vol-price divergence
            price_change = returns_1hr
            vol_change = (vol_60 / vol_ma_360 - 1) if vol_ma_360 > 0 else 0
            vol_price_divergence = vol_change - price_change

            # Time features
            ts = df['timestamp'].iloc[idx]
            hour = ts.hour
            dow = ts.dayofweek
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            dow_sin = np.sin(2 * np.pi * dow / 7)
            dow_cos = np.cos(2 * np.pi * dow / 7)

            return {
                'natr_14': natr_14,
                'bb_width_20': bb_width_20,
                'bb_position': bb_position,
                'rsi_14': rsi_14,
                'returns_1hr': returns_1hr,
                'returns_6hr': returns_6hr,
                'returns_12hr': returns_12hr,
                'momentum_5': momentum_5,
                'momentum_20': momentum_20,
                'dist_from_24hr_high': dist_from_24hr_high,
                'dist_from_24hr_low': dist_from_24hr_low,
                'hl_range': hl_range,
                'body_ratio_avg_1hr': body_ratio_avg_1hr,
                'range_compression': range_compression,
                'volume_vs_ma20': volume_vs_ma20,
                'volume_trend_6hr': volume_trend_6hr,
                'obv_slope_1hr': obv_slope_1hr,
                'vroc_12': vroc_12,
                'vol_ratio_20_60': vol_ratio_20_60,
                'vol_ratio_60_240': vol_ratio_60_240,
                'vol_acceleration': vol_acceleration,
                'vol_price_divergence': vol_price_divergence,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'dow_sin': dow_sin,
                'dow_cos': dow_cos,
            }
        except Exception as e:
            return None

    def get_prediction(self, features: Dict) -> float:
        """Get model prediction probability for features"""
        try:
            # Extract features in correct order
            X = np.array([[features.get(col, 0) for col in self.feature_cols]])

            # Handle NaN/Inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale if scaler available
            if self.scaler:
                X = self.scaler.transform(X)

            # Get probability
            prob = self.model.predict_proba(X)[0][1]
            return float(prob)
        except Exception as e:
            return 0.0

    def run_backtest(self, features_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> BacktestResult:
        """Run the backtest simulation"""
        result = BacktestResult(config=self.config)

        open_positions: List[Position] = []
        closed_positions: List[Position] = []
        equity_curve = []
        starting_capital = 100000.0
        cash = starting_capital

        # Sort by timestamp
        features_df = features_df.sort_values('timestamp')

        # Create price lookup
        price_lookup = {}
        for _, row in ohlcv_df.iterrows():
            key = (row['symbol'], row['timestamp'])
            price_lookup[key] = row['close']

        # Group features by timestamp for processing
        for timestamp in features_df['timestamp'].unique():
            ts_features = features_df[features_df['timestamp'] == timestamp]

            # Check exits first
            positions_to_close = []
            for pos in open_positions:
                # Get current price
                current_price = None
                for _, row in ts_features[ts_features['symbol'] == pos.symbol].iterrows():
                    current_price = row['close']
                    break

                if current_price is None:
                    # Try to get from OHLCV
                    key = (pos.symbol, timestamp)
                    current_price = price_lookup.get(key)

                if current_price is None:
                    continue

                # Update peak price
                if current_price > pos.peak_price:
                    pos.peak_price = current_price

                pct_change = (current_price - pos.entry_price) / pos.entry_price
                peak_pct = (pos.peak_price - pos.entry_price) / pos.entry_price
                hold_minutes = (timestamp - pos.entry_time).total_seconds() / 60

                exit_reason = None

                # Check stop loss
                if pct_change <= -self.config.stop_loss_pct:
                    exit_reason = "stop_loss"
                    pos.exit_price = pos.entry_price * (1 - self.config.stop_loss_pct)

                # Check take profit
                elif pct_change >= self.config.take_profit_pct:
                    exit_reason = "take_profit"
                    pos.exit_price = current_price

                # Check max hold time
                elif hold_minutes >= self.config.max_hold_minutes:
                    exit_reason = "max_hold"
                    pos.exit_price = current_price

                # Check trailing stop
                elif peak_pct >= self.config.trailing_activation_pct:
                    trailing_stop = pos.peak_price * (1 - self.config.trailing_stop_pct)
                    if current_price <= trailing_stop:
                        exit_reason = "trailing_stop"
                        pos.exit_price = trailing_stop

                if exit_reason:
                    pos.exit_time = timestamp
                    pos.exit_reason = exit_reason
                    pos.pnl = pos.quantity * (pos.exit_price - pos.entry_price)
                    positions_to_close.append(pos)

            # Close positions
            for pos in positions_to_close:
                open_positions.remove(pos)
                closed_positions.append(pos)
                cash += pos.position_size + pos.pnl

            # Check entries (if we have capacity)
            if len(open_positions) < self.config.max_positions:
                for _, row in ts_features.iterrows():
                    if len(open_positions) >= self.config.max_positions:
                        break

                    # Skip if already in position
                    if any(p.symbol == row['symbol'] for p in open_positions):
                        continue

                    # Check volume ratio filter
                    if row.get('vol_ratio_20_60', 0) < self.config.min_volume_ratio:
                        continue

                    # Check momentum filter
                    if self.config.require_momentum and row.get('momentum_5', 0) <= 0:
                        continue

                    # Get prediction
                    features = {col: row.get(col, 0) for col in self.feature_cols}
                    prob = self.get_prediction(features)

                    if prob >= self.config.threshold:
                        # Enter position
                        if cash >= self.config.position_size:
                            entry_price = row['close']
                            quantity = self.config.position_size / entry_price

                            pos = Position(
                                symbol=row['symbol'],
                                entry_price=entry_price,
                                entry_time=timestamp,
                                quantity=quantity,
                                position_size=self.config.position_size
                            )
                            open_positions.append(pos)
                            cash -= self.config.position_size

            # Track equity
            open_value = sum(p.position_size for p in open_positions)
            total_equity = cash + open_value
            equity_curve.append({'timestamp': timestamp, 'equity': total_equity})

        # Close any remaining positions at last price
        for pos in open_positions:
            last_price = features_df[features_df['symbol'] == pos.symbol]['close'].iloc[-1] if len(features_df[features_df['symbol'] == pos.symbol]) > 0 else pos.entry_price
            pos.exit_price = last_price
            pos.exit_time = features_df['timestamp'].max()
            pos.exit_reason = "end_of_backtest"
            pos.pnl = pos.quantity * (pos.exit_price - pos.entry_price)
            closed_positions.append(pos)

        # Calculate metrics
        result.total_trades = len(closed_positions)

        if result.total_trades > 0:
            pnls = [p.pnl for p in closed_positions]
            result.total_pnl = sum(pnls)
            result.avg_pnl_per_trade = result.total_pnl / result.total_trades

            wins = [p for p in closed_positions if p.pnl > 0]
            losses = [p for p in closed_positions if p.pnl <= 0]

            result.winning_trades = len(wins)
            result.losing_trades = len(losses)
            result.win_rate = len(wins) / result.total_trades if result.total_trades > 0 else 0

            result.avg_win = np.mean([p.pnl for p in wins]) if wins else 0
            result.avg_loss = np.mean([p.pnl for p in losses]) if losses else 0

            # Max drawdown
            if equity_curve:
                equities = pd.Series([e['equity'] for e in equity_curve])
                rolling_max = equities.expanding().max()
                drawdown = (equities - rolling_max) / rolling_max
                result.max_drawdown = drawdown.min()

            # Sharpe ratio (simplified)
            if len(pnls) > 1:
                result.sharpe_ratio = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0

            # Store trade details
            for p in closed_positions:
                result.trades.append({
                    'symbol': p.symbol,
                    'entry_price': p.entry_price,
                    'entry_time': p.entry_time,
                    'exit_price': p.exit_price,
                    'exit_time': p.exit_time,
                    'exit_reason': p.exit_reason,
                    'pnl': p.pnl,
                    'pct_return': (p.exit_price - p.entry_price) / p.entry_price * 100
                })

        return result


def run_parameter_sweep(ohlcv_df: pd.DataFrame, features_df: pd.DataFrame):
    """Run backtest with various parameter combinations"""

    results = []

    # Models to test
    models = []
    for path in DEFAULT_MODELS:
        if Path(path).exists():
            models.append(path)

    if not models:
        logger.error("No models found!")
        return

    # Parameter grid
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    stop_losses = [0.01, 0.02, 0.03]
    take_profits = [0.05, 0.10, 0.15]
    min_vol_ratios = [1.0, 1.5, 2.0]

    total_configs = len(models) * len(thresholds) * len(stop_losses) * len(take_profits) * len(min_vol_ratios)
    logger.info(f"Running {total_configs} configurations...")

    config_num = 0
    for model_path in models:
        for threshold in thresholds:
            for stop_loss in stop_losses:
                for take_profit in take_profits:
                    for min_vol_ratio in min_vol_ratios:
                        config_num += 1

                        config = BacktestConfig(
                            model_path=model_path,
                            threshold=threshold,
                            stop_loss_pct=stop_loss,
                            take_profit_pct=take_profit,
                            min_volume_ratio=min_vol_ratio
                        )

                        try:
                            backtester = EventClassifierBacktester(config)
                            result = backtester.run_backtest(features_df, ohlcv_df)

                            results.append({
                                'model': Path(model_path).name,
                                'threshold': threshold,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'min_vol_ratio': min_vol_ratio,
                                'trades': result.total_trades,
                                'win_rate': result.win_rate,
                                'total_pnl': result.total_pnl,
                                'avg_pnl': result.avg_pnl_per_trade,
                                'avg_win': result.avg_win,
                                'avg_loss': result.avg_loss,
                                'max_dd': result.max_drawdown,
                                'sharpe': result.sharpe_ratio
                            })

                            if config_num % 10 == 0:
                                logger.info(f"  Progress: {config_num}/{total_configs}")

                        except Exception as e:
                            logger.warning(f"Config failed: {e}")
                            continue

    return results


def print_results(results: List[Dict]):
    """Print results table sorted by total P&L"""
    if not results:
        logger.warning("No results to display")
        return

    df = pd.DataFrame(results)

    # Filter to configs with at least 5 trades
    df = df[df['trades'] >= 5]

    if len(df) == 0:
        logger.warning("No configurations had >= 5 trades")
        return

    # Sort by total P&L
    df = df.sort_values('total_pnl', ascending=False)

    print("\n" + "=" * 120)
    print("TOP 20 CONFIGURATIONS BY P&L (min 5 trades)")
    print("=" * 120)
    print(f"{'Model':<25} {'Thresh':>6} {'SL':>5} {'TP':>5} {'VolR':>5} {'Trades':>6} {'WinR':>6} {'P&L':>10} {'AvgPnL':>8} {'MaxDD':>7}")
    print("-" * 120)

    for _, row in df.head(20).iterrows():
        print(f"{row['model']:<25} {row['threshold']:>6.2f} {row['stop_loss']:>5.1%} {row['take_profit']:>5.1%} {row['min_vol_ratio']:>5.1f} {row['trades']:>6} {row['win_rate']:>6.1%} ${row['total_pnl']:>9.0f} ${row['avg_pnl']:>7.0f} {row['max_dd']:>7.1%}")

    print("\n" + "=" * 120)
    print("TOP 20 CONFIGURATIONS BY WIN RATE (min 10 trades)")
    print("=" * 120)

    df_wr = df[df['trades'] >= 10].sort_values('win_rate', ascending=False)
    print(f"{'Model':<25} {'Thresh':>6} {'SL':>5} {'TP':>5} {'VolR':>5} {'Trades':>6} {'WinR':>6} {'P&L':>10} {'AvgPnL':>8} {'MaxDD':>7}")
    print("-" * 120)

    for _, row in df_wr.head(20).iterrows():
        print(f"{row['model']:<25} {row['threshold']:>6.2f} {row['stop_loss']:>5.1%} {row['take_profit']:>5.1%} {row['min_vol_ratio']:>5.1f} {row['trades']:>6} {row['win_rate']:>6.1%} ${row['total_pnl']:>9.0f} ${row['avg_pnl']:>7.0f} {row['max_dd']:>7.1%}")

    # Save full results
    output_path = '/Users/bz/Pythia2/data/backtest_results.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"\nFull results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Backtest Event Classifier Strategies')
    parser.add_argument('--model', type=str, help='Specific model path to test')
    parser.add_argument('--threshold', type=float, default=0.80, help='Entry threshold')
    parser.add_argument('--stop-loss', type=float, default=0.01, help='Stop loss percentage')
    parser.add_argument('--take-profit', type=float, default=0.10, help='Take profit percentage')
    parser.add_argument('--sweep', action='store_true', help='Run full parameter sweep')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("EVENT CLASSIFIER BACKTEST")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading OHLCV data...")
    config = BacktestConfig(
        model_path=args.model or DEFAULT_MODELS[0],
        threshold=args.threshold,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit
    )

    backtester = EventClassifierBacktester(config)
    ohlcv_df = backtester.load_ohlcv_data()

    # Compute features
    logger.info("\nComputing features (this may take a while)...")
    features_df = backtester.compute_features(ohlcv_df)

    if features_df.empty:
        logger.error("No features computed. Check data availability.")
        return

    if args.sweep:
        # Run parameter sweep
        logger.info("\nRunning parameter sweep...")
        results = run_parameter_sweep(ohlcv_df, features_df)
        print_results(results)
    else:
        # Run single backtest
        logger.info(f"\nRunning backtest with threshold={args.threshold}, stop_loss={args.stop_loss}, take_profit={args.take_profit}")
        result = backtester.run_backtest(features_df, ohlcv_df)

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Model: {Path(config.model_path).name}")
        print(f"Threshold: {config.threshold}")
        print(f"Stop Loss: {config.stop_loss_pct:.1%}")
        print(f"Take Profit: {config.take_profit_pct:.1%}")
        print("-" * 60)
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.1%}")
        print(f"Total P&L: ${result.total_pnl:,.2f}")
        print(f"Avg P&L per Trade: ${result.avg_pnl_per_trade:,.2f}")
        print(f"Avg Win: ${result.avg_win:,.2f}")
        print(f"Avg Loss: ${result.avg_loss:,.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.1%}")
        print("=" * 60)

        if result.trades:
            print("\nSample Trades:")
            for trade in result.trades[:10]:
                print(f"  {trade['symbol']}: {trade['pct_return']:+.1f}% ({trade['exit_reason']})")


if __name__ == "__main__":
    main()
