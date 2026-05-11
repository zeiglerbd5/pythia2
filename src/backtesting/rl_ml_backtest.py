"""
RL-Enhanced ML Backtest

Integrates the RL position sizing agent with the existing ML backtest framework.
Uses the trained RL agent to dynamically adjust position sizes based on:
- Spike probability from LightGBM
- Market context features
- Current portfolio state
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import duckdb
from loguru import logger

from src.models.rl_position_sizer import (
    PositionSizingAgent,
    RLConfig,
    HAS_TORCH
)


@dataclass
class RLTrade:
    """Represents a single RL-enhanced trade."""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    position_size: float = 0.0
    position_size_pct: float = 0.0  # RL-determined size
    pred_probability: float = 0.0
    rl_action: int = 0


@dataclass
class RLBacktestConfig:
    """RL backtest configuration."""
    initial_capital: float = 10000.0
    max_position_size_pct: float = 20.0  # Max per trade
    max_positions: int = 3
    max_portfolio_exposure: float = 0.6  # Max 60% of capital in positions

    # ML threshold (minimum to consider)
    min_pred_probability: float = 0.2

    # Exit parameters
    take_profit_pct: float = 10.0
    stop_loss_pct: float = 5.0
    max_hold_hours: int = 24

    # Costs
    trading_fee_pct: float = 0.1
    slippage_pct: float = 0.1

    # RL model path
    rl_model_path: Optional[str] = None


@dataclass
class RLBacktestResult:
    """RL backtest results with detailed metrics."""
    total_pnl_usd: float = 0.0
    total_pnl_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0

    # Exit breakdown
    exits_take_profit: int = 0
    exits_stop_loss: int = 0
    exits_time_limit: int = 0

    # RL-specific metrics
    avg_position_size_pct: float = 0.0
    position_size_std: float = 0.0
    skipped_by_rl: int = 0  # Signals RL chose to skip

    trades: List[RLTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    position_size_history: List[float] = field(default_factory=list)


class RLMLBacktester:
    """
    Backtester using RL agent for position sizing.

    Combines:
    1. LightGBM spike predictions for signal generation
    2. RL agent for position sizing decisions
    """

    def __init__(self, db_path: str = "full_pythia.duckdb"):
        self.conn = duckdb.connect(db_path)
        self.rl_agent: Optional[PositionSizingAgent] = None
        self.rl_config: Optional[RLConfig] = None

    def load_rl_agent(self, model_path: str, config: Optional[RLConfig] = None):
        """Load trained RL agent."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.rl_config = config or RLConfig()
        self.rl_agent = PositionSizingAgent(self.rl_config)
        self.rl_agent.load(model_path)
        logger.info(f"Loaded RL agent from {model_path}")

    def _build_rl_state(
        self,
        signal: pd.Series,
        in_position: bool,
        hold_duration: float,
        current_drawdown: float,
    ) -> np.ndarray:
        """Build state vector for RL agent."""
        state = np.array([
            signal.get('y_pred_proba', 0.5),  # Spike probability
            min(signal.get('volatility_4h', 0.5), 5.0) / 5.0,
            np.clip(signal.get('momentum_4h', 0.0), -10, 10) / 10.0,
            min(signal.get('volume_ratio', 1.0), 3.0) / 3.0,
            signal.get('rsi_proxy', 50.0) / 100.0,
            float(in_position),
            hold_duration / 24.0,  # Normalized hold duration
            current_drawdown / 10.0,
        ], dtype=np.float32)
        return state

    def run(
        self,
        predictions_csv: str,
        features_csv: str,
        config: RLBacktestConfig,
    ) -> RLBacktestResult:
        """
        Run backtest using RL agent for position sizing.

        Args:
            predictions_csv: CSV with model predictions
            features_csv: CSV with features
            config: Backtest configuration

        Returns:
            RLBacktestResult with performance metrics
        """
        result = RLBacktestResult()

        # Check RL agent loaded
        if self.rl_agent is None:
            if config.rl_model_path:
                self.load_rl_agent(config.rl_model_path)
            else:
                logger.warning("No RL agent loaded. Using fixed position sizing.")

        # Load and merge data
        preds = pd.read_csv(predictions_csv, parse_dates=['timestamp'])
        features = pd.read_csv(features_csv, parse_dates=['timestamp'])

        signals = preds.merge(
            features[['timestamp', 'symbol', 'price_at_signal', 'volatility_4h',
                     'momentum_4h', 'volume_ratio', 'rsi_proxy']],
            on=['timestamp', 'symbol'],
            how='left'
        )

        # Fill missing values
        signals = signals.fillna({
            'volatility_4h': 0.5,
            'momentum_4h': 0.0,
            'volume_ratio': 1.0,
            'rsi_proxy': 50.0,
        })

        # Filter by minimum probability
        signals = signals[signals['y_pred_proba'] >= config.min_pred_probability]
        signals = signals.sort_values('timestamp')

        logger.info(f"Processing {len(signals)} signals (threshold: {config.min_pred_probability})")

        if len(signals) == 0:
            return result

        # Initialize state
        capital = config.initial_capital
        peak_capital = capital
        max_drawdown = 0.0
        equity_curve = [capital]
        position_sizes = []
        open_positions: Dict[str, RLTrade] = {}

        # Process signals chronologically
        for idx, signal in signals.iterrows():
            signal_time = signal['timestamp']
            symbol = signal['symbol']

            # Check exits first
            closed = self._check_exits(
                open_positions, signal_time, config, result
            )
            for trade in closed:
                capital += trade.pnl_usd + trade.position_size
                result.trades.append(trade)

            # Update drawdown
            if capital > peak_capital:
                peak_capital = capital
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)

            # Skip if already have position in this symbol
            if symbol in open_positions:
                continue

            # Check portfolio constraints
            current_exposure = sum(t.position_size for t in open_positions.values())
            if len(open_positions) >= config.max_positions:
                continue
            if current_exposure >= capital * config.max_portfolio_exposure:
                continue

            # Get RL position sizing decision
            if self.rl_agent is not None:
                state = self._build_rl_state(
                    signal,
                    in_position=False,
                    hold_duration=0,
                    current_drawdown=drawdown * 100,
                )
                position_size_pct = self.rl_agent.get_position_size(state)

                # Skip if RL says 0% position
                if position_size_pct <= 0:
                    result.skipped_by_rl += 1
                    continue
            else:
                # Fallback: scale by prediction confidence
                position_size_pct = min(signal['y_pred_proba'], 1.0)

            # Record position size decision
            position_sizes.append(position_size_pct)
            result.position_size_history.append(position_size_pct)

            # Calculate actual position size
            max_position = capital * (config.max_position_size_pct / 100)
            position_value = max_position * position_size_pct

            # Ensure minimum position
            if position_value < 50:
                continue

            # Enter position
            trade = self._try_entry(signal, position_value, position_size_pct, config)
            if trade:
                open_positions[symbol] = trade
                capital -= trade.position_size
                equity_curve.append(capital + sum(t.position_size for t in open_positions.values()))

        # Close remaining positions at end
        if signals is not None and len(signals) > 0:
            end_time = signals['timestamp'].max() + timedelta(hours=1)
            for symbol, trade in list(open_positions.items()):
                self._close_position(trade, end_time, "end_of_backtest", config, result)
                capital += trade.pnl_usd + trade.position_size
                result.trades.append(trade)

        equity_curve.append(capital)
        result.equity_curve = equity_curve
        result.max_drawdown_pct = max_drawdown * 100

        # Calculate metrics
        self._calculate_metrics(result, config.initial_capital, position_sizes)

        return result

    def _try_entry(
        self,
        signal: pd.Series,
        position_value: float,
        position_size_pct: float,
        config: RLBacktestConfig,
    ) -> Optional[RLTrade]:
        """Try to enter a position."""
        symbol = signal['symbol']
        signal_time = signal['timestamp']

        # Get entry price (5 min after signal)
        entry_price = self._get_price_at_time(symbol, signal_time + timedelta(minutes=5))
        if entry_price is None:
            return None

        # Apply slippage
        entry_price *= (1 + config.slippage_pct / 100)

        return RLTrade(
            symbol=symbol,
            entry_time=signal_time,
            entry_price=entry_price,
            position_size=position_value,
            position_size_pct=position_size_pct,
            pred_probability=signal['y_pred_proba'],
        )

    def _check_exits(
        self,
        open_positions: Dict[str, RLTrade],
        current_time: datetime,
        config: RLBacktestConfig,
        result: RLBacktestResult,
    ) -> List[RLTrade]:
        """Check and close positions that hit exit conditions."""
        closed = []

        for symbol, trade in list(open_positions.items()):
            current_price = self._get_price_at_time(symbol, current_time)
            if current_price is None:
                continue

            pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100

            exit_reason = None

            # Take profit
            if pnl_pct >= config.take_profit_pct:
                exit_reason = "take_profit"
            # Stop loss
            elif pnl_pct <= -config.stop_loss_pct:
                exit_reason = "stop_loss"
            # Time limit
            else:
                hold_hours = (current_time - trade.entry_time).total_seconds() / 3600
                if hold_hours >= config.max_hold_hours:
                    exit_reason = "time_limit"

            if exit_reason:
                self._close_position(trade, current_time, exit_reason, config, result)
                closed.append(trade)
                del open_positions[symbol]

        return closed

    def _close_position(
        self,
        trade: RLTrade,
        exit_time: datetime,
        reason: str,
        config: RLBacktestConfig,
        result: RLBacktestResult,
    ):
        """Close a position and calculate PnL."""
        exit_price = self._get_price_at_time(trade.symbol, exit_time)
        if exit_price is None:
            exit_price = trade.entry_price

        # Apply slippage (unfavorable)
        exit_price *= (1 - config.slippage_pct / 100)

        gross_pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        net_pnl_pct = gross_pnl_pct - (2 * config.trading_fee_pct)

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl_pct = net_pnl_pct
        trade.pnl_usd = trade.position_size * (net_pnl_pct / 100)

        if reason == "take_profit":
            result.exits_take_profit += 1
        elif reason == "stop_loss":
            result.exits_stop_loss += 1
        elif reason == "time_limit":
            result.exits_time_limit += 1

    def _get_price_at_time(self, symbol: str, time: datetime) -> Optional[float]:
        """Get price at or near given time."""
        try:
            query = f"""
                SELECT close
                FROM ohlcv
                WHERE symbol = '{symbol}'
                  AND timeframe = '5m'
                  AND timestamp >= '{time - timedelta(minutes=30)}'
                  AND timestamp <= '{time + timedelta(minutes=30)}'
                ORDER BY ABS(EPOCH(timestamp) - EPOCH(TIMESTAMP '{time}'))
                LIMIT 1
            """
            result = self.conn.execute(query).fetchone()
            return result[0] if result else None
        except Exception:
            return None

    def _calculate_metrics(
        self,
        result: RLBacktestResult,
        initial_capital: float,
        position_sizes: List[float],
    ):
        """Calculate final performance metrics."""
        if not result.trades:
            return

        result.total_trades = len(result.trades)

        wins = [t for t in result.trades if t.pnl_pct > 0]
        losses = [t for t in result.trades if t.pnl_pct <= 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(result.trades) * 100 if result.trades else 0

        result.avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
        result.avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0

        result.total_pnl_usd = sum(t.pnl_usd for t in result.trades)
        result.total_pnl_pct = (result.total_pnl_usd / initial_capital) * 100

        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Position size metrics
        if position_sizes:
            result.avg_position_size_pct = np.mean(position_sizes) * 100
            result.position_size_std = np.std(position_sizes) * 100

        # Sharpe ratio
        if len(result.equity_curve) > 1:
            returns = pd.Series(result.equity_curve).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)


def run_comparison():
    """
    Run comparison between RL-enhanced and baseline backtest.
    """
    print("=" * 70)
    print("RL vs BASELINE BACKTEST COMPARISON")
    print("=" * 70)

    backtester = RLMLBacktester()

    # Configuration
    config = RLBacktestConfig(
        initial_capital=10000,
        max_position_size_pct=15,
        max_positions=3,
        min_pred_probability=0.2,
        take_profit_pct=10,
        stop_loss_pct=5,
        max_hold_hours=24,
    )

    # Run RL-enhanced backtest
    print("\n" + "-" * 70)
    print("RL-ENHANCED BACKTEST")
    print("-" * 70)

    try:
        backtester.load_rl_agent("models/rl_position_sizer.pt")
        rl_result = backtester.run(
            predictions_csv="spike_predictions.csv",
            features_csv="whale_features.csv",
            config=config,
        )

        print(f"\nRL Results:")
        print(f"  Trades:         {rl_result.total_trades}")
        print(f"  Win Rate:       {rl_result.win_rate:.1f}%")
        print(f"  Total PnL:      ${rl_result.total_pnl_usd:+,.2f} ({rl_result.total_pnl_pct:+.1f}%)")
        print(f"  Profit Factor:  {rl_result.profit_factor:.2f}")
        print(f"  Max Drawdown:   {rl_result.max_drawdown_pct:.1f}%")
        print(f"  Sharpe Ratio:   {rl_result.sharpe_ratio:.2f}")
        print(f"\n  Avg Position:   {rl_result.avg_position_size_pct:.1f}%")
        print(f"  Skipped by RL:  {rl_result.skipped_by_rl}")
        print(f"\n  Exits: TP={rl_result.exits_take_profit}, SL={rl_result.exits_stop_loss}, Time={rl_result.exits_time_limit}")

    except FileNotFoundError:
        print("  RL model not found. Train first with: python -m src.models.rl_position_sizer")
        rl_result = None

    # Run baseline (no RL)
    print("\n" + "-" * 70)
    print("BASELINE BACKTEST (No RL)")
    print("-" * 70)

    backtester_baseline = RLMLBacktester()
    baseline_result = backtester_baseline.run(
        predictions_csv="spike_predictions.csv",
        features_csv="whale_features.csv",
        config=config,
    )

    print(f"\nBaseline Results:")
    print(f"  Trades:         {baseline_result.total_trades}")
    print(f"  Win Rate:       {baseline_result.win_rate:.1f}%")
    print(f"  Total PnL:      ${baseline_result.total_pnl_usd:+,.2f} ({baseline_result.total_pnl_pct:+.1f}%)")
    print(f"  Profit Factor:  {baseline_result.profit_factor:.2f}")
    print(f"  Max Drawdown:   {baseline_result.max_drawdown_pct:.1f}%")
    print(f"  Sharpe Ratio:   {baseline_result.sharpe_ratio:.2f}")
    print(f"\n  Exits: TP={baseline_result.exits_take_profit}, SL={baseline_result.exits_stop_loss}, Time={baseline_result.exits_time_limit}")

    # Comparison
    if rl_result:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)

        pnl_diff = rl_result.total_pnl_usd - baseline_result.total_pnl_usd
        pnl_improvement = (
            pnl_diff / abs(baseline_result.total_pnl_usd) * 100
            if baseline_result.total_pnl_usd != 0 else 0
        )

        print(f"\n  PnL Difference:     ${pnl_diff:+,.2f} ({pnl_improvement:+.1f}%)")
        print(f"  Win Rate Diff:      {rl_result.win_rate - baseline_result.win_rate:+.1f}pp")
        print(f"  Trade Count Diff:   {rl_result.total_trades - baseline_result.total_trades:+d}")

        # Per-trade analysis
        if rl_result.trades:
            print("\n  RL Trade Details:")
            for trade in rl_result.trades[:5]:
                print(f"    {trade.symbol}: size={trade.position_size_pct*100:.0f}%, "
                      f"PnL={trade.pnl_pct:+.1f}%, exit={trade.exit_reason}")

    return rl_result, baseline_result


if __name__ == "__main__":
    run_comparison()
