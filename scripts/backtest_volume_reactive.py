#!/usr/bin/env python3
"""
Backtest for volume-reactive spike trading strategy.

Strategy:
1. Detect volume explosion (3x+ 24h average)
2. Price up >5% in 24h
3. Apply FOMO filters (momentum, volatility)
4. Use ratcheting trailing stops

Uses DuckDB copy to avoid locking issues with live collector.
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Configuration
@dataclass
class BacktestConfig:
    # Entry filters
    volume_multiple_threshold: float = 3.0  # 3x normal volume
    price_change_threshold: float = 0.05    # +5% in 24h
    max_momentum: float = 5.0               # FOMO filter
    max_volatility: float = 0.6             # Volatility filter

    # Position sizing
    position_size_usd: float = 1000.0
    max_positions: int = 3
    starting_capital: float = 10000.0

    # Exit rules - ratcheting trailing stops
    stop_loss_pct: float = 0.03  # 3% stop loss
    ratchet_levels: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.02, 0.012),   # 2-6%: 1.2% trail
        (0.06, 0.018),   # 6-10%: 1.8% trail
        (0.10, 0.012),   # 10-15%: 1.2% trail
        (0.15, 0.04),    # 15%+: 4% trail
    ])
    max_hold_minutes: int = 1440  # 24 hours


@dataclass
class Position:
    symbol: str
    entry_time: datetime
    entry_price: float
    size_usd: float
    quantity: float
    highest_price: float
    trailing_activated: bool = False


@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    hold_minutes: int


class VolumeReactiveBacktest:
    def __init__(self, db_path: str, config: BacktestConfig = None):
        self.db_path = db_path
        self.config = config or BacktestConfig()
        self.conn = duckdb.connect(db_path, read_only=True)

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.cash = self.config.starting_capital
        self.signals_seen = 0
        self.signals_filtered = 0

        # Cache for volume stats
        self.volume_cache: Dict[str, Dict] = {}

    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load OHLCV data for backtest period."""
        print(f"Loading data from {start_date} to {end_date}...")

        query = f"""
            SELECT
                symbol,
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                volume * close as volume_usd
            FROM ohlcv
            WHERE timestamp >= '{start_date}'
              AND timestamp <= '{end_date}'
            ORDER BY symbol, timestamp
        """
        df = self.conn.execute(query).fetchdf()

        # Remove duplicate timestamps per symbol
        df = df.drop_duplicates(subset=['symbol', 'timestamp'])

        print(f"Loaded {len(df):,} candles for {df['symbol'].nunique()} symbols")
        return df

    def calculate_volume_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling 24h volume stats per symbol."""
        print("Calculating volume statistics...")

        # Group by symbol and calculate rolling stats
        results = []
        for symbol in df['symbol'].unique():
            sym_df = df[df['symbol'] == symbol].copy()
            sym_df = sym_df.sort_values('timestamp')

            if len(sym_df) < 1440:  # Need at least 24h of data
                continue

            # Rolling 24h USD volume (1440 minutes)
            sym_df['volume_24h'] = sym_df['volume_usd'].rolling(window=1440, min_periods=100).sum()
            sym_df['volume_avg'] = sym_df['volume_usd'].rolling(window=1440*2, min_periods=1440).mean() * 1440
            sym_df['volume_multiple'] = sym_df['volume_24h'] / sym_df['volume_avg'].replace(0, np.nan)

            # Price change 24h
            sym_df['price_24h_ago'] = sym_df['close'].shift(1440)
            sym_df['price_change_24h'] = (sym_df['close'] - sym_df['price_24h_ago']) / sym_df['price_24h_ago']

            # Volatility (ATR-based)
            sym_df['tr'] = np.maximum(
                sym_df['high'] - sym_df['low'],
                np.maximum(
                    abs(sym_df['high'] - sym_df['close'].shift(1)),
                    abs(sym_df['low'] - sym_df['close'].shift(1))
                )
            )
            sym_df['atr_4h'] = sym_df['tr'].rolling(window=240, min_periods=60).mean()
            sym_df['volatility'] = sym_df['atr_4h'] / sym_df['close']

            # Momentum (price acceleration)
            sym_df['ret_5m'] = sym_df['close'].pct_change(5)
            sym_df['momentum'] = abs(sym_df['ret_5m']) / sym_df['volatility'].replace(0, np.nan)

            results.append(sym_df)

        combined = pd.concat(results, ignore_index=True)
        print(f"Calculated stats for {len(results)} symbols")
        return combined

    def check_entry_signal(self, row: pd.Series) -> Tuple[bool, str]:
        """Check if row qualifies as entry signal."""
        # Volume explosion
        if pd.isna(row['volume_multiple']) or row['volume_multiple'] < self.config.volume_multiple_threshold:
            return False, "low_volume"

        # Price up > threshold
        if pd.isna(row['price_change_24h']) or row['price_change_24h'] < self.config.price_change_threshold:
            return False, "price_not_up"

        # FOMO filter - extreme momentum
        if not pd.isna(row['momentum']) and row['momentum'] > self.config.max_momentum:
            return False, "fomo_momentum"

        # Volatility filter
        if not pd.isna(row['volatility']) and row['volatility'] > self.config.max_volatility:
            return False, "high_volatility"

        return True, "passed"

    def get_trail_distance(self, gain_pct: float) -> float:
        """Get trailing stop distance based on current gain."""
        for threshold, trail in reversed(self.config.ratchet_levels):
            if gain_pct >= threshold:
                return trail
        return self.config.stop_loss_pct  # Default to stop loss

    def check_exit(self, pos: Position, current_price: float, current_time: datetime) -> Tuple[bool, str, float]:
        """Check if position should be exited."""
        gain_pct = (current_price - pos.entry_price) / pos.entry_price

        # Update highest price
        if current_price > pos.highest_price:
            pos.highest_price = current_price

        # Check stop loss
        if gain_pct <= -self.config.stop_loss_pct:
            return True, "stop_loss", current_price

        # Check trailing stop (only if in profit)
        if gain_pct > 0.02:  # Trail activates at +2%
            pos.trailing_activated = True
            trail_distance = self.get_trail_distance(gain_pct)
            trail_price = pos.highest_price * (1 - trail_distance)
            if current_price <= trail_price:
                return True, f"trailing_stop_{gain_pct:.1%}", current_price

        # Check max hold time
        hold_minutes = (current_time - pos.entry_time).total_seconds() / 60
        if hold_minutes >= self.config.max_hold_minutes:
            return True, "timeout", current_price

        return False, "", 0

    def run(self, start_date: str, end_date: str):
        """Run the backtest."""
        # Load and prepare data
        df = self.load_data(start_date, end_date)
        df = self.calculate_volume_stats(df)
        df = df.sort_values('timestamp')

        print(f"\nRunning backtest from {start_date} to {end_date}...")
        print(f"Config: {self.config.volume_multiple_threshold}x volume, {self.config.price_change_threshold:.0%} price threshold")
        print(f"Position size: ${self.config.position_size_usd}, Max positions: {self.config.max_positions}")
        print("-" * 60)

        # Track signals by type
        signal_reasons = defaultdict(int)

        # Process each row
        last_checked = {}  # Avoid checking same symbol multiple times per minute

        for idx, row in df.iterrows():
            symbol = row['symbol']
            timestamp = row['timestamp']
            price = row['close']

            # Skip if we already checked this symbol this minute
            check_key = (symbol, timestamp)
            if check_key in last_checked:
                continue
            last_checked[check_key] = True

            # Check exits for open positions
            if symbol in self.positions:
                pos = self.positions[symbol]
                should_exit, reason, exit_price = self.check_exit(pos, price, timestamp)
                if should_exit:
                    pnl_usd = (exit_price - pos.entry_price) * pos.quantity
                    pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                    hold_minutes = int((timestamp - pos.entry_time).total_seconds() / 60)

                    trade = Trade(
                        symbol=symbol,
                        entry_time=pos.entry_time,
                        exit_time=timestamp,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        size_usd=pos.size_usd,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        exit_reason=reason,
                        hold_minutes=hold_minutes,
                    )
                    self.trades.append(trade)
                    self.cash += pos.size_usd + pnl_usd
                    del self.positions[symbol]

            # Check for entry signals
            if symbol not in self.positions and len(self.positions) < self.config.max_positions:
                is_signal, reason = self.check_entry_signal(row)

                if is_signal:
                    self.signals_seen += 1

                    # Enter position
                    if self.cash >= self.config.position_size_usd:
                        quantity = self.config.position_size_usd / price
                        pos = Position(
                            symbol=symbol,
                            entry_time=timestamp,
                            entry_price=price,
                            size_usd=self.config.position_size_usd,
                            quantity=quantity,
                            highest_price=price,
                        )
                        self.positions[symbol] = pos
                        self.cash -= self.config.position_size_usd
                else:
                    if row['volume_multiple'] >= self.config.volume_multiple_threshold:
                        signal_reasons[reason] += 1
                        self.signals_filtered += 1

        # Close any remaining positions at last price
        for symbol, pos in list(self.positions.items()):
            last_row = df[df['symbol'] == symbol].iloc[-1]
            exit_price = last_row['close']
            pnl_usd = (exit_price - pos.entry_price) * pos.quantity
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price

            trade = Trade(
                symbol=symbol,
                entry_time=pos.entry_time,
                exit_time=last_row['timestamp'],
                entry_price=pos.entry_price,
                exit_price=exit_price,
                size_usd=pos.size_usd,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                exit_reason="end_of_backtest",
                hold_minutes=int((last_row['timestamp'] - pos.entry_time).total_seconds() / 60),
            )
            self.trades.append(trade)

        self.print_results(signal_reasons)

    def print_results(self, signal_reasons: Dict[str, int]):
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        if not self.trades:
            print("No trades executed!")
            return

        # Calculate stats
        total_pnl = sum(t.pnl_usd for t in self.trades)
        wins = [t for t in self.trades if t.pnl_usd > 0]
        losses = [t for t in self.trades if t.pnl_usd <= 0]

        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t.pnl_usd for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_usd for t in losses]) if losses else 0

        # Exit reasons
        exit_reasons = defaultdict(int)
        for t in self.trades:
            exit_reasons[t.exit_reason] += 1

        print(f"\nTrade Summary:")
        print(f"  Total Trades:    {len(self.trades)}")
        print(f"  Wins:            {len(wins)} ({win_rate:.1f}%)")
        print(f"  Losses:          {len(losses)}")
        print(f"  Avg Win:         ${avg_win:,.2f}")
        print(f"  Avg Loss:        ${avg_loss:,.2f}")

        print(f"\nP&L Summary:")
        print(f"  Starting Capital: ${self.config.starting_capital:,.2f}")
        print(f"  Total P&L:        ${total_pnl:,.2f}")
        print(f"  Return:           {total_pnl/self.config.starting_capital*100:.1f}%")
        print(f"  Final Capital:    ${self.config.starting_capital + total_pnl:,.2f}")

        print(f"\nSignal Stats:")
        print(f"  Signals Seen:     {self.signals_seen}")
        print(f"  Filtered Out:     {self.signals_filtered}")

        print(f"\nFilter Breakdown:")
        for reason, count in sorted(signal_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

        print(f"\nExit Reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

        # Best and worst trades
        sorted_trades = sorted(self.trades, key=lambda t: t.pnl_pct, reverse=True)
        print(f"\nTop 5 Trades:")
        for t in sorted_trades[:5]:
            print(f"  {t.symbol}: {t.pnl_pct:+.1%} (${t.pnl_usd:+,.2f}) - {t.exit_reason}")

        print(f"\nWorst 5 Trades:")
        for t in sorted_trades[-5:]:
            print(f"  {t.symbol}: {t.pnl_pct:+.1%} (${t.pnl_usd:+,.2f}) - {t.exit_reason}")

        # Average hold time
        avg_hold = np.mean([t.hold_minutes for t in self.trades])
        print(f"\nAverage Hold Time: {avg_hold:.0f} minutes ({avg_hold/60:.1f} hours)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Volume-reactive strategy backtest")
    parser.add_argument("--db", default="/Users/bz/Pythia2/pythia_backtest_copy.duckdb", help="DuckDB path")
    parser.add_argument("--start", default="2026-03-19", help="Start date")
    parser.add_argument("--end", default="2026-03-23", help="End date")
    parser.add_argument("--volume-mult", type=float, default=3.0, help="Volume multiple threshold")
    parser.add_argument("--price-change", type=float, default=0.05, help="Price change threshold")
    args = parser.parse_args()

    config = BacktestConfig(
        volume_multiple_threshold=args.volume_mult,
        price_change_threshold=args.price_change,
    )

    backtest = VolumeReactiveBacktest(args.db, config)
    backtest.run(args.start, args.end)


if __name__ == "__main__":
    main()
