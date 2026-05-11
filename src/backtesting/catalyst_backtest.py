"""
Catalyst Signal Backtesting Framework

Simulates trading based on catalyst signals with realistic:
- Entry timing (next candle after signal)
- Exit logic (take profit, stop loss, time-based)
- Position sizing
- Transaction costs

Outputs: PnL, Sharpe ratio, drawdown, win rate, trade log
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger
import json


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    position_size: float = 0.0

    # Signal info
    signal_source: str = ""
    signal_type: str = ""
    signal_title: str = ""


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    # Capital
    initial_capital: float = 10000.0
    position_size_pct: float = 5.0  # % of capital per trade
    max_positions: int = 5

    # Entry
    entry_delay_minutes: int = 5  # Wait N minutes after signal

    # Exit
    take_profit_pct: float = 10.0  # Take profit at X%
    stop_loss_pct: float = 5.0     # Stop loss at X%
    max_hold_hours: int = 24       # Max hold time

    # Costs
    trading_fee_pct: float = 0.1   # 0.1% per trade (entry + exit)
    slippage_pct: float = 0.1      # 0.1% slippage

    # Filters
    min_signal_priority: float = 0.5
    signal_types: Optional[List[str]] = None  # Filter by type
    signal_sources: Optional[List[str]] = None  # Filter by source


@dataclass
class BacktestResult:
    """Backtest results."""
    # Performance
    total_pnl_usd: float = 0.0
    total_pnl_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0

    # By exit reason
    exits_take_profit: int = 0
    exits_stop_loss: int = 0
    exits_time_limit: int = 0

    # Trade log
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class CatalystBacktester:
    """
    Backtests trading strategies based on catalyst signals.
    """

    def __init__(self, db_path: str = "full_pythia.duckdb"):
        self.conn = duckdb.connect(db_path)
        self.config: Optional[BacktestConfig] = None

    def run(self, config: BacktestConfig) -> BacktestResult:
        """
        Run backtest with given configuration.

        Args:
            config: Backtest configuration

        Returns:
            BacktestResult with performance metrics and trade log
        """
        self.config = config
        result = BacktestResult()

        # Load signals
        signals = self._load_signals()
        logger.info(f"Loaded {len(signals)} signals for backtesting")

        if len(signals) == 0:
            return result

        # Initialize state
        capital = config.initial_capital
        equity_curve = [capital]
        open_positions: Dict[str, Trade] = {}  # symbol -> Trade

        # Process signals chronologically
        for idx, signal in signals.iterrows():
            signal_time = signal["timestamp"]
            symbol = signal["symbol"]

            # Check and close any positions that hit exit conditions
            closed = self._check_exits(open_positions, signal_time, result)
            for trade in closed:
                capital += trade.pnl_usd + trade.position_size
                result.trades.append(trade)

            # Skip if we already have position in this symbol
            if symbol in open_positions:
                continue

            # Skip if at max positions
            if len(open_positions) >= config.max_positions:
                continue

            # Try to enter new position
            trade = self._try_entry(signal, capital)
            if trade:
                open_positions[symbol] = trade
                capital -= trade.position_size

            equity_curve.append(capital + sum(t.position_size for t in open_positions.values()))

        # Close any remaining positions at end
        end_time = signals["timestamp"].max() + timedelta(hours=1)
        for symbol, trade in list(open_positions.items()):
            self._close_position(trade, end_time, "end_of_backtest", result)
            capital += trade.pnl_usd + trade.position_size
            result.trades.append(trade)

        equity_curve.append(capital)
        result.equity_curve = equity_curve

        # Calculate final metrics
        self._calculate_metrics(result, config.initial_capital)

        return result

    def _load_signals(self) -> pd.DataFrame:
        """Load signals from database based on config filters."""
        query = """
            SELECT
                symbol,
                timestamp,
                source,
                event_type,
                title,
                event_priority as priority
            FROM news_signals
            WHERE timestamp >= '2025-12-10'
              AND symbol != 'UNKNOWN-USD'
        """

        params = []

        if self.config.min_signal_priority:
            query += f" AND event_priority >= {self.config.min_signal_priority}"

        if self.config.signal_types:
            types_str = ", ".join(f"'{t}'" for t in self.config.signal_types)
            query += f" AND event_type IN ({types_str})"

        if self.config.signal_sources:
            sources_str = ", ".join(f"'{s}'" for s in self.config.signal_sources)
            query += f" AND source IN ({sources_str})"

        query += " ORDER BY timestamp"

        return self.conn.execute(query).df()

    def _try_entry(self, signal: pd.Series, available_capital: float) -> Optional[Trade]:
        """Try to enter a position based on signal."""
        symbol = signal["symbol"]
        signal_time = signal["timestamp"]

        # Calculate entry time (delay after signal)
        entry_time = signal_time + timedelta(minutes=self.config.entry_delay_minutes)

        # Get entry price (close of candle at entry time)
        entry_price = self._get_price_at_time(symbol, entry_time)
        if entry_price is None:
            return None

        # Calculate position size
        position_size = min(
            available_capital * (self.config.position_size_pct / 100),
            available_capital * 0.95  # Leave some buffer
        )

        if position_size < 10:  # Minimum $10 position
            return None

        # Apply slippage to entry
        entry_price *= (1 + self.config.slippage_pct / 100)

        return Trade(
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            position_size=position_size,
            signal_source=signal["source"],
            signal_type=signal["event_type"],
            signal_title=signal["title"][:100] if signal["title"] else "",
        )

    def _check_exits(
        self,
        open_positions: Dict[str, Trade],
        current_time: datetime,
        result: BacktestResult
    ) -> List[Trade]:
        """Check if any open positions should be closed."""
        closed = []

        for symbol, trade in list(open_positions.items()):
            # Get current price
            current_price = self._get_price_at_time(symbol, current_time)
            if current_price is None:
                continue

            pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100

            # Check take profit
            if pnl_pct >= self.config.take_profit_pct:
                self._close_position(trade, current_time, "take_profit", result)
                closed.append(trade)
                del open_positions[symbol]
                continue

            # Check stop loss
            if pnl_pct <= -self.config.stop_loss_pct:
                self._close_position(trade, current_time, "stop_loss", result)
                closed.append(trade)
                del open_positions[symbol]
                continue

            # Check time limit
            hold_time = (current_time - trade.entry_time).total_seconds() / 3600
            if hold_time >= self.config.max_hold_hours:
                self._close_position(trade, current_time, "time_limit", result)
                closed.append(trade)
                del open_positions[symbol]
                continue

        return closed

    def _close_position(
        self,
        trade: Trade,
        exit_time: datetime,
        reason: str,
        result: BacktestResult
    ):
        """Close a position and calculate PnL."""
        exit_price = self._get_price_at_time(trade.symbol, exit_time)
        if exit_price is None:
            exit_price = trade.entry_price  # Fallback

        # Apply slippage to exit
        exit_price *= (1 - self.config.slippage_pct / 100)

        # Calculate PnL
        gross_pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100

        # Subtract fees (entry + exit)
        net_pnl_pct = gross_pnl_pct - (2 * self.config.trading_fee_pct)

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl_pct = net_pnl_pct
        trade.pnl_usd = trade.position_size * (net_pnl_pct / 100)

        # Update exit reason counters
        if reason == "take_profit":
            result.exits_take_profit += 1
        elif reason == "stop_loss":
            result.exits_stop_loss += 1
        elif reason == "time_limit":
            result.exits_time_limit += 1

    def _get_price_at_time(self, symbol: str, time: datetime) -> Optional[float]:
        """Get price at or near given time."""
        try:
            result = self.conn.execute(f"""
                SELECT close
                FROM ohlcv
                WHERE symbol = '{symbol}'
                  AND timeframe = '5m'
                  AND timestamp >= '{time - timedelta(minutes=30)}'
                  AND timestamp <= '{time + timedelta(minutes=30)}'
                ORDER BY ABS(EPOCH(timestamp) - EPOCH(TIMESTAMP '{time}'))
                LIMIT 1
            """).fetchone()

            return result[0] if result else None
        except:
            return None

    def _calculate_metrics(self, result: BacktestResult, initial_capital: float):
        """Calculate final performance metrics."""
        if not result.trades:
            return

        result.total_trades = len(result.trades)

        # Win/loss
        wins = [t for t in result.trades if t.pnl_pct > 0]
        losses = [t for t in result.trades if t.pnl_pct <= 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(result.trades) * 100 if result.trades else 0

        # Average win/loss
        result.avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
        result.avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0

        # PnL
        result.total_pnl_usd = sum(t.pnl_usd for t in result.trades)
        result.total_pnl_pct = (result.total_pnl_usd / initial_capital) * 100

        # Profit factor
        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio (annualized, assuming daily returns)
        if len(result.equity_curve) > 1:
            returns = pd.Series(result.equity_curve).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        # Max drawdown
        equity = pd.Series(result.equity_curve)
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        result.max_drawdown_pct = abs(drawdown.min()) * 100


def run_backtest(
    config: Optional[BacktestConfig] = None,
    db_path: str = "full_pythia.duckdb"
) -> BacktestResult:
    """Convenience function to run backtest."""
    if config is None:
        config = BacktestConfig()

    backtester = CatalystBacktester(db_path=db_path)
    return backtester.run(config)


def print_results(result: BacktestResult, config: BacktestConfig):
    """Print formatted backtest results."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Initial Capital:    ${config.initial_capital:,.0f}")
    print(f"  Position Size:      {config.position_size_pct}%")
    print(f"  Take Profit:        {config.take_profit_pct}%")
    print(f"  Stop Loss:          {config.stop_loss_pct}%")
    print(f"  Max Hold:           {config.max_hold_hours}h")
    print(f"  Min Signal Priority: {config.min_signal_priority}")

    print("\n" + "-" * 70)
    print("Performance:")
    print(f"  Total PnL:          ${result.total_pnl_usd:,.2f} ({result.total_pnl_pct:+.1f}%)")
    print(f"  Sharpe Ratio:       {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:       {result.max_drawdown_pct:.1f}%")
    print(f"  Profit Factor:      {result.profit_factor:.2f}")

    print("\n" + "-" * 70)
    print("Trade Statistics:")
    print(f"  Total Trades:       {result.total_trades}")
    print(f"  Win Rate:           {result.win_rate:.1f}%")
    print(f"  Avg Win:            {result.avg_win_pct:+.2f}%")
    print(f"  Avg Loss:           {result.avg_loss_pct:+.2f}%")

    print("\n" + "-" * 70)
    print("Exit Reasons:")
    print(f"  Take Profit:        {result.exits_take_profit}")
    print(f"  Stop Loss:          {result.exits_stop_loss}")
    print(f"  Time Limit:         {result.exits_time_limit}")

    if result.trades:
        print("\n" + "-" * 70)
        print("Top 10 Trades:")
        sorted_trades = sorted(result.trades, key=lambda t: t.pnl_pct, reverse=True)
        for trade in sorted_trades[:10]:
            print(
                f"  {trade.entry_time.strftime('%Y-%m-%d %H:%M')} | "
                f"{trade.symbol:12} | {trade.signal_type:15} | "
                f"{trade.pnl_pct:+.1f}% | {trade.exit_reason}"
            )

    print("\n" + "=" * 70)


def main():
    """Run backtest with different configurations."""
    import argparse

    parser = argparse.ArgumentParser(description="Backtest catalyst signals")
    parser.add_argument("--capital", type=float, default=10000)
    parser.add_argument("--position-size", type=float, default=5.0)
    parser.add_argument("--take-profit", type=float, default=10.0)
    parser.add_argument("--stop-loss", type=float, default=5.0)
    parser.add_argument("--max-hold", type=int, default=24)
    parser.add_argument("--min-priority", type=float, default=0.5)
    parser.add_argument("--signal-types", nargs="+", default=None)
    parser.add_argument("--signal-sources", nargs="+", default=None)
    parser.add_argument("--db", default="full_pythia.duckdb")
    args = parser.parse_args()

    config = BacktestConfig(
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss,
        max_hold_hours=args.max_hold,
        min_signal_priority=args.min_priority,
        signal_types=args.signal_types,
        signal_sources=args.signal_sources,
    )

    logger.info("Running backtest...")
    result = run_backtest(config, db_path=args.db)
    print_results(result, config)

    # Save trade log
    if result.trades:
        trades_df = pd.DataFrame([
            {
                "symbol": t.symbol,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl_pct": t.pnl_pct,
                "pnl_usd": t.pnl_usd,
                "exit_reason": t.exit_reason,
                "signal_source": t.signal_source,
                "signal_type": t.signal_type,
            }
            for t in result.trades
        ])
        trades_df.to_csv("backtest_trades.csv", index=False)
        print(f"\nTrade log saved to: backtest_trades.csv")


if __name__ == "__main__":
    main()
