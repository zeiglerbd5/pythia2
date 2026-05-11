"""
RL Integrated Backtest

Uses the trained RL agent with real price data from the database.
This provides a realistic evaluation of the RL position sizing strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import duckdb
from loguru import logger
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.rl_position_sizer_v3 import DQNAgentV3, RLConfigV3, HAS_TORCH


@dataclass
class Trade:
    """Trade record."""
    symbol: str
    entry_time: datetime
    entry_price: float
    position_size: float
    position_size_pct: float
    pred_prob: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 10000.0
    max_position_pct: float = 20.0  # Max 20% per trade
    max_positions: int = 3
    min_pred_prob: float = 0.2  # Min probability to consider

    take_profit_pct: float = 5.0
    stop_loss_pct: float = 3.0
    max_hold_hours: int = 24

    fee_pct: float = 0.1
    slippage_pct: float = 0.1


@dataclass
class BacktestResult:
    """Backtest results."""
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    avg_position_pct: float = 0.0
    n_skipped_by_rl: int = 0

    exits_tp: int = 0
    exits_sl: int = 0
    exits_time: int = 0
    exits_agent: int = 0

    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class RLIntegratedBacktester:
    """Backtester using RL agent with real price data."""

    def __init__(self, db_path: str = "full_pythia.duckdb"):
        self.conn = duckdb.connect(db_path)
        self.agent = None
        self.config_rl = None

    def load_agent(self, model_path: str):
        """Load trained RL agent."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        self.config_rl = RLConfigV3()
        self.agent = DQNAgentV3(self.config_rl)
        self.agent.load(model_path)
        logger.info(f"Loaded RL agent from {model_path}")

    def _build_state(
        self,
        signal: pd.Series,
        in_position: bool,
        hold_hours: float,
        unrealized_pnl: float,
        n_trades: int,
        total_pnl: float,
        capital: float,
    ) -> np.ndarray:
        """Build state vector for RL agent."""
        return np.array([
            signal.get('y_pred_proba', 0.5),
            min(signal.get('volatility_4h', 0.5), 3.0) / 3.0,
            np.clip(signal.get('momentum_4h', 0), -5, 5) / 5.0,
            min(signal.get('volume_ratio', 1.0), 3.0) / 3.0,
            signal.get('rsi_proxy', 50) / 100.0,
            float(in_position),
            min(hold_hours / 24.0, 1.0),
            np.clip(unrealized_pnl / 10.0, -1, 1),
            min(n_trades / 10.0, 1.0),
            total_pnl / capital,
        ], dtype=np.float32)

    def _get_price(self, symbol: str, time: datetime) -> Optional[float]:
        """Get price at time from database."""
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
        except:
            return None

    def _get_max_price(self, symbol: str, start: datetime, end: datetime) -> Optional[float]:
        """Get maximum high price in time range."""
        try:
            query = f"""
                SELECT MAX(high)
                FROM ohlcv
                WHERE symbol = '{symbol}'
                  AND timeframe = '5m'
                  AND timestamp >= '{start}'
                  AND timestamp <= '{end}'
            """
            result = self.conn.execute(query).fetchone()
            return result[0] if result else None
        except:
            return None

    def _get_min_price(self, symbol: str, start: datetime, end: datetime) -> Optional[float]:
        """Get minimum low price in time range."""
        try:
            query = f"""
                SELECT MIN(low)
                FROM ohlcv
                WHERE symbol = '{symbol}'
                  AND timeframe = '5m'
                  AND timestamp >= '{start}'
                  AND timestamp <= '{end}'
            """
            result = self.conn.execute(query).fetchone()
            return result[0] if result else None
        except:
            return None

    def run(
        self,
        predictions_csv: str,
        features_csv: str,
        config: BacktestConfig,
    ) -> BacktestResult:
        """Run integrated backtest."""
        result = BacktestResult()

        # Load data
        preds = pd.read_csv(predictions_csv, parse_dates=['timestamp'])
        features = pd.read_csv(features_csv, parse_dates=['timestamp'])

        signals = preds.merge(
            features[['timestamp', 'symbol', 'price_at_signal', 'volatility_4h',
                     'momentum_4h', 'volume_ratio', 'rsi_proxy']],
            on=['timestamp', 'symbol'],
            how='left'
        ).fillna({
            'volatility_4h': 0.5,
            'momentum_4h': 0.0,
            'volume_ratio': 1.0,
            'rsi_proxy': 50.0,
        })

        # Filter by probability
        signals = signals[signals['y_pred_proba'] >= config.min_pred_prob]
        signals = signals.sort_values('timestamp')

        logger.info(f"Processing {len(signals)} signals")

        # State
        capital = config.initial_capital
        peak_capital = capital
        open_positions: Dict[str, Trade] = {}
        equity_curve = [capital]
        position_sizes = []

        # Process each signal
        for _, signal in signals.iterrows():
            signal_time = signal['timestamp']
            symbol = signal['symbol']

            # Check exits for open positions
            closed = self._check_exits(open_positions, signal_time, config, result)
            for trade in closed:
                capital += trade.pnl_usd + trade.position_size
                result.trades.append(trade)

            # Update equity
            current_equity = capital + sum(t.position_size for t in open_positions.values())
            equity_curve.append(current_equity)
            if current_equity > peak_capital:
                peak_capital = current_equity

            # Skip if at max positions or already in symbol
            if len(open_positions) >= config.max_positions:
                continue
            if symbol in open_positions:
                continue

            # Get RL decision
            if self.agent is not None:
                state = self._build_state(
                    signal,
                    in_position=False,
                    hold_hours=0,
                    unrealized_pnl=0,
                    n_trades=len(result.trades),
                    total_pnl=sum(t.pnl_usd for t in result.trades),
                    capital=config.initial_capital,
                )
                action = self.agent.select_action(state, training=False)

                if action == 0:
                    result.n_skipped_by_rl += 1
                    continue

                position_size_pct = self.config_rl.position_sizes[action]
            else:
                # Fallback: use prediction probability
                position_size_pct = signal['y_pred_proba']

            # Calculate position size
            max_pos = capital * (config.max_position_pct / 100)
            position_value = max_pos * position_size_pct

            if position_value < 50:
                continue

            # Get entry price
            entry_price = self._get_price(symbol, signal_time + timedelta(minutes=5))
            if entry_price is None:
                continue

            # Apply slippage
            entry_price *= (1 + config.slippage_pct / 100)

            # Create trade
            trade = Trade(
                symbol=symbol,
                entry_time=signal_time,
                entry_price=entry_price,
                position_size=position_value,
                position_size_pct=position_size_pct,
                pred_prob=signal['y_pred_proba'],
            )
            open_positions[symbol] = trade
            capital -= position_value
            position_sizes.append(position_size_pct)

        # Close remaining positions
        if signals is not None and len(signals) > 0:
            end_time = signals['timestamp'].max() + timedelta(hours=1)
            for symbol, trade in list(open_positions.items()):
                self._close_trade(trade, end_time, "end", config, result)
                capital += trade.pnl_usd + trade.position_size
                result.trades.append(trade)

        equity_curve.append(capital)
        result.equity_curve = equity_curve

        # Calculate metrics
        self._calc_metrics(result, config, position_sizes)

        return result

    def _check_exits(
        self,
        positions: Dict[str, Trade],
        current_time: datetime,
        config: BacktestConfig,
        result: BacktestResult,
    ) -> List[Trade]:
        """Check and execute exits."""
        closed = []

        for symbol, trade in list(positions.items()):
            # Get price range since entry
            max_price = self._get_max_price(symbol, trade.entry_time, current_time)
            min_price = self._get_min_price(symbol, trade.entry_time, current_time)
            current_price = self._get_price(symbol, current_time)

            if current_price is None:
                continue

            # Calculate returns
            pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
            max_pnl = ((max_price - trade.entry_price) / trade.entry_price) * 100 if max_price else pnl_pct
            min_pnl = ((min_price - trade.entry_price) / trade.entry_price) * 100 if min_price else pnl_pct

            hold_hours = (current_time - trade.entry_time).total_seconds() / 3600

            # Check exit conditions
            exit_reason = None
            exit_pnl = None

            # TP hit
            if max_pnl >= config.take_profit_pct:
                exit_reason = "tp"
                exit_pnl = config.take_profit_pct
            # SL hit
            elif min_pnl <= -config.stop_loss_pct:
                exit_reason = "sl"
                exit_pnl = -config.stop_loss_pct
            # Time limit
            elif hold_hours >= config.max_hold_hours:
                exit_reason = "time"
                exit_pnl = pnl_pct

            if exit_reason:
                self._close_trade(trade, current_time, exit_reason, config, result, exit_pnl)
                closed.append(trade)
                del positions[symbol]

        return closed

    def _close_trade(
        self,
        trade: Trade,
        exit_time: datetime,
        reason: str,
        config: BacktestConfig,
        result: BacktestResult,
        override_pnl: Optional[float] = None,
    ):
        """Close a trade."""
        exit_price = self._get_price(trade.symbol, exit_time)
        if exit_price is None:
            exit_price = trade.entry_price

        # Apply slippage
        exit_price *= (1 - config.slippage_pct / 100)

        if override_pnl is not None:
            gross_pnl = override_pnl
        else:
            gross_pnl = ((exit_price - trade.entry_price) / trade.entry_price) * 100

        net_pnl = gross_pnl - (2 * config.fee_pct)

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl_pct = net_pnl
        trade.pnl_usd = trade.position_size * (net_pnl / 100)

        if reason == "tp":
            result.exits_tp += 1
        elif reason == "sl":
            result.exits_sl += 1
        elif reason == "time":
            result.exits_time += 1

    def _calc_metrics(self, result: BacktestResult, config: BacktestConfig, position_sizes: List[float]):
        """Calculate performance metrics."""
        if not result.trades:
            return

        result.n_trades = len(result.trades)

        wins = [t for t in result.trades if t.pnl_pct > 0]
        losses = [t for t in result.trades if t.pnl_pct <= 0]

        result.n_wins = len(wins)
        result.n_losses = len(losses)
        result.win_rate = len(wins) / len(result.trades) * 100

        result.total_pnl = sum(t.pnl_usd for t in result.trades)
        result.total_pnl_pct = result.total_pnl / config.initial_capital * 100

        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        if position_sizes:
            result.avg_position_pct = np.mean(position_sizes) * 100

        # Max drawdown
        equity = pd.Series(result.equity_curve)
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        result.max_drawdown = abs(drawdown.min()) * 100

        # Sharpe
        returns = equity.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            result.sharpe = (returns.mean() / returns.std()) * np.sqrt(252)


def main():
    """Run integrated backtest."""
    print("=" * 70)
    print("RL INTEGRATED BACKTEST")
    print("=" * 70)

    backtester = RLIntegratedBacktester()

    config = BacktestConfig(
        initial_capital=10000,
        max_position_pct=20,
        max_positions=3,
        min_pred_prob=0.2,
        take_profit_pct=5.0,
        stop_loss_pct=3.0,
        max_hold_hours=24,
        fee_pct=0.1,
        slippage_pct=0.1,
    )

    # RL backtest
    print("\n" + "-" * 70)
    print("RL-ENHANCED BACKTEST")
    print("-" * 70)

    try:
        backtester.load_agent("models/rl_position_sizer_v3.pt")

        rl_result = backtester.run(
            predictions_csv="spike_predictions.csv",
            features_csv="whale_features.csv",
            config=config,
        )

        print(f"\n  Trades:        {rl_result.n_trades}")
        print(f"  Win Rate:      {rl_result.win_rate:.1f}%")
        print(f"  Total PnL:     ${rl_result.total_pnl:+,.2f} ({rl_result.total_pnl_pct:+.1f}%)")
        print(f"  Profit Factor: {rl_result.profit_factor:.2f}")
        print(f"  Max Drawdown:  {rl_result.max_drawdown:.1f}%")
        print(f"  Sharpe Ratio:  {rl_result.sharpe:.2f}")
        print(f"  Avg Position:  {rl_result.avg_position_pct:.1f}%")
        print(f"  Skipped by RL: {rl_result.n_skipped_by_rl}")
        print(f"\n  Exits: TP={rl_result.exits_tp}, SL={rl_result.exits_sl}, Time={rl_result.exits_time}")

        if rl_result.trades:
            print(f"\n  Sample Trades:")
            for t in rl_result.trades[:5]:
                print(f"    {t.symbol}: pos={t.position_size_pct*100:.0f}%, "
                      f"pnl={t.pnl_pct:+.1f}%, exit={t.exit_reason}")

    except FileNotFoundError:
        print("  RL model not found. Train first.")
        rl_result = None

    # Baseline (no RL)
    print("\n" + "-" * 70)
    print("BASELINE BACKTEST (No RL)")
    print("-" * 70)

    backtester_base = RLIntegratedBacktester()
    baseline_result = backtester_base.run(
        predictions_csv="spike_predictions.csv",
        features_csv="whale_features.csv",
        config=config,
    )

    print(f"\n  Trades:        {baseline_result.n_trades}")
    print(f"  Win Rate:      {baseline_result.win_rate:.1f}%")
    print(f"  Total PnL:     ${baseline_result.total_pnl:+,.2f} ({baseline_result.total_pnl_pct:+.1f}%)")
    print(f"  Profit Factor: {baseline_result.profit_factor:.2f}")
    print(f"  Max Drawdown:  {baseline_result.max_drawdown:.1f}%")
    print(f"\n  Exits: TP={baseline_result.exits_tp}, SL={baseline_result.exits_sl}, Time={baseline_result.exits_time}")

    # Comparison
    if rl_result:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)

        pnl_diff = rl_result.total_pnl - baseline_result.total_pnl
        improvement = (
            pnl_diff / abs(baseline_result.total_pnl) * 100
            if baseline_result.total_pnl != 0 else 0
        )

        print(f"\n  RL PnL:       ${rl_result.total_pnl:+,.2f}")
        print(f"  Baseline PnL: ${baseline_result.total_pnl:+,.2f}")
        print(f"  Difference:   ${pnl_diff:+,.2f}")

        if improvement != 0:
            print(f"  Improvement:  {improvement:+.1f}%")

        print(f"\n  RL Win Rate:       {rl_result.win_rate:.1f}%")
        print(f"  Baseline Win Rate: {baseline_result.win_rate:.1f}%")

    return rl_result, baseline_result


if __name__ == "__main__":
    main()
