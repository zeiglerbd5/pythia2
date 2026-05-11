#!/usr/bin/env python3
"""
Backtest NATR + Order Book Entry Strategy

Uses historical data from data/pythia.duckdb (Sep 24-28, 2025)

Strategy:
- Entry: NATR normalized >= 0.8 AND any order book feature >= 0.8 within 6 min window
- Exit: -1% stop loss, +20%/+30% partial exits, trailing stop at +15%, max 180 min hold
"""

import duckdb
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class BacktestPosition:
    """Tracks a single position through its lifecycle"""
    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: float
    position_size: float = 2500.0
    peak_price: float = 0.0
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0
    partial_exits: List[int] = field(default_factory=list)
    remaining_quantity: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    pnl: float = 0.0

    def __post_init__(self):
        self.peak_price = self.entry_price
        self.remaining_quantity = self.quantity


class StrategyBacktester:
    """
    Backtests the NATR + Order Book entry strategy.

    Entry: NATR >= 0.8 normalized AND any order book feature >= 0.8 within 6-min window
    Exit: Stop loss at -1%, partial exits at +20%/+30%, trailing stop at +15%, max 180 min
    """

    def __init__(self, db_path: str = "data/pythia.duckdb"):
        self.db_path = db_path
        self.starting_capital = 10000.0
        self.position_size = 2500.0
        self.max_positions = 4

        # Exit parameters
        self.stop_loss_pct = -0.01  # -1%
        self.partial_exit_1_pct = 0.20  # +20%
        self.partial_exit_2_pct = 0.30  # +30%
        self.trailing_stop_activation = 0.15  # +15%
        self.trailing_stop_distance = 0.065  # 6.5%
        self.max_hold_minutes = 180

        # Results
        self.open_positions: Dict[str, BacktestPosition] = {}
        self.closed_positions: List[BacktestPosition] = []
        self.trade_log: List[Dict] = []

        # Price cache for exit simulation
        self.candle_data: Dict[str, pd.DataFrame] = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load features and order book data efficiently (separate queries)"""
        conn = duckdb.connect(self.db_path, read_only=True)

        print("Loading features table...")
        features_df = conn.execute('''
            SELECT symbol, timestamp, natr
            FROM features
            WHERE natr IS NOT NULL
            ORDER BY timestamp, symbol
        ''').fetchdf()
        print(f"  Loaded {len(features_df):,} feature rows")

        print("Loading order book features (aggregating to minute)...")
        ob_df = conn.execute('''
            SELECT
                symbol,
                DATE_TRUNC('minute', CAST(timestamp AS TIMESTAMP)) as timestamp,
                AVG(spread_percentage) as spread_percentage,
                AVG(bid_ask_ratio) as bid_ask_ratio,
                AVG(large_bid_orders) as large_bid_orders,
                AVG(large_ask_orders) as large_ask_orders
            FROM order_book_features
            GROUP BY symbol, DATE_TRUNC('minute', CAST(timestamp AS TIMESTAMP))
            ORDER BY timestamp, symbol
        ''').fetchdf()
        print(f"  Loaded {len(ob_df):,} order book rows (aggregated)")

        # Load candles for price simulation
        print("Loading candles for price simulation...")
        candles_df = conn.execute('''
            SELECT symbol, timestamp, open, high, low, close
            FROM candles
            ORDER BY symbol, timestamp
        ''').fetchdf()
        print(f"  Loaded {len(candles_df):,} candle rows")

        # Cache candles by symbol for fast lookup
        for symbol in candles_df['symbol'].unique():
            self.candle_data[symbol] = candles_df[candles_df['symbol'] == symbol].set_index('timestamp')

        conn.close()
        return features_df, ob_df

    def merge_data(self, features_df: pd.DataFrame, ob_df: pd.DataFrame) -> pd.DataFrame:
        """Merge features with order book data on symbol and timestamp"""
        print("Merging features with order book data...")

        # Inner join - only keep rows with both NATR and order book data
        merged = pd.merge(
            features_df,
            ob_df,
            on=['symbol', 'timestamp'],
            how='inner'
        )

        # Calculate large_order_imbalance
        total_orders = merged['large_bid_orders'] + merged['large_ask_orders']
        merged['large_order_imbalance'] = np.where(
            total_orders > 0,
            (merged['large_bid_orders'] - merged['large_ask_orders']) / total_orders,
            0
        )

        print(f"  Merged data: {len(merged):,} rows")
        return merged

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalize features and map to 0-1 scale"""
        print("Normalizing features...")

        features = ['natr', 'spread_percentage', 'bid_ask_ratio', 'large_order_imbalance']

        for feat in features:
            if feat in df.columns:
                mean = df[feat].mean()
                std = df[feat].std()
                if std > 0:
                    z = (df[feat] - mean) / std
                    z_clipped = np.clip(z, -3, 3)
                    df[f'{feat}_norm'] = (z_clipped + 3) / 6
                else:
                    df[f'{feat}_norm'] = 0.5
                print(f"  {feat}: mean={mean:.4f}, std={std:.4f}")

        return df

    def get_price_at_time(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """Get price for a symbol at a given time from candles"""
        if symbol not in self.candle_data:
            return None

        candles = self.candle_data[symbol]

        # Find closest candle at or before timestamp
        if timestamp in candles.index:
            return candles.loc[timestamp, 'close']

        # Find closest prior candle
        prior = candles[candles.index <= timestamp]
        if len(prior) > 0:
            return prior.iloc[-1]['close']

        return None

    def get_price_range(self, symbol: str, start: datetime, end: datetime) -> Tuple[float, float, float]:
        """Get high/low/close prices for a symbol over a time range"""
        if symbol not in self.candle_data:
            return None, None, None

        candles = self.candle_data[symbol]
        period = candles[(candles.index >= start) & (candles.index <= end)]

        if len(period) == 0:
            return None, None, None

        return period['high'].max(), period['low'].min(), period.iloc[-1]['close']

    def check_exits(self, timestamp: datetime) -> List[str]:
        """Check and execute exits for all open positions"""
        symbols_to_close = []

        for symbol, pos in list(self.open_positions.items()):
            # Get current price range
            high, low, close = self.get_price_range(
                symbol,
                pos.entry_time,
                timestamp
            )

            if high is None:
                continue

            current_price = close
            pct_change = (current_price - pos.entry_price) / pos.entry_price

            # Update peak price
            if high > pos.peak_price:
                pos.peak_price = high

            exit_reason = None
            exit_price = current_price

            # Check stop loss
            if pct_change <= self.stop_loss_pct:
                exit_reason = "stop_loss"
                exit_price = pos.entry_price * (1 + self.stop_loss_pct)

            # Check max hold time
            elif (timestamp - pos.entry_time).total_seconds() / 60 >= self.max_hold_minutes:
                exit_reason = "max_hold"

            # Check trailing stop
            elif pos.trailing_stop_active:
                peak_pct = (pos.peak_price - pos.entry_price) / pos.entry_price
                trailing_price = pos.peak_price * (1 - self.trailing_stop_distance)
                if current_price <= trailing_price:
                    exit_reason = "trailing_stop"
                    exit_price = trailing_price

            # Check partial exits
            elif 1 not in pos.partial_exits and pct_change >= self.partial_exit_1_pct:
                # Partial exit 1: sell 25%
                pos.partial_exits.append(1)
                partial_qty = pos.quantity * 0.25
                partial_pnl = partial_qty * (current_price - pos.entry_price)
                pos.remaining_quantity -= partial_qty
                self.trade_log.append({
                    'type': 'partial_exit_1',
                    'symbol': symbol,
                    'timestamp': str(timestamp),
                    'price': current_price,
                    'quantity': partial_qty,
                    'pnl': partial_pnl
                })

            elif 2 not in pos.partial_exits and pct_change >= self.partial_exit_2_pct:
                # Partial exit 2: sell 25%
                pos.partial_exits.append(2)
                partial_qty = pos.quantity * 0.25
                partial_pnl = partial_qty * (current_price - pos.entry_price)
                pos.remaining_quantity -= partial_qty
                self.trade_log.append({
                    'type': 'partial_exit_2',
                    'symbol': symbol,
                    'timestamp': str(timestamp),
                    'price': current_price,
                    'quantity': partial_qty,
                    'pnl': partial_pnl
                })

            # Activate trailing stop at +15%
            if not pos.trailing_stop_active and pct_change >= self.trailing_stop_activation:
                pos.trailing_stop_active = True
                pos.trailing_stop_price = current_price * (1 - self.trailing_stop_distance)

            # Execute full exit
            if exit_reason:
                pos.exit_price = exit_price
                pos.exit_time = timestamp
                pos.exit_reason = exit_reason
                pos.pnl = pos.remaining_quantity * (exit_price - pos.entry_price)

                self.closed_positions.append(pos)
                symbols_to_close.append(symbol)

                self.trade_log.append({
                    'type': 'exit',
                    'symbol': symbol,
                    'timestamp': str(timestamp),
                    'price': exit_price,
                    'reason': exit_reason,
                    'pnl': pos.pnl,
                    'pct_return': (exit_price - pos.entry_price) / pos.entry_price * 100
                })

        # Remove closed positions
        for symbol in symbols_to_close:
            del self.open_positions[symbol]

        return symbols_to_close

    def run_backtest(self) -> Dict:
        """Run the full backtest simulation"""
        print("=" * 60)
        print("NATR + Order Book Strategy Backtest")
        print("=" * 60)

        # Load and prepare data
        features_df, ob_df = self.load_data()
        merged_df = self.merge_data(features_df, ob_df)
        merged_df = self.normalize_features(merged_df)

        # Sort chronologically
        merged_df = merged_df.sort_values('timestamp')

        # Track 6-minute windows for signals
        signal_tracker: Dict[str, List[Tuple]] = defaultdict(list)

        # Process each timestamp chronologically
        print("\nRunning backtest simulation...")
        timestamps = sorted(merged_df['timestamp'].unique())
        total_ts = len(timestamps)

        entry_count = 0

        for i, ts in enumerate(timestamps):
            if i % 10000 == 0:
                print(f"  Processing {i:,}/{total_ts:,} timestamps ({i/total_ts*100:.1f}%)")

            # First check exits for open positions
            self.check_exits(ts)

            # Get all data for this timestamp
            ts_data = merged_df[merged_df['timestamp'] == ts]

            for _, row in ts_data.iterrows():
                symbol = row['symbol']

                # Record feature hits >= 0.8
                for feat, norm_col in [
                    ('natr', 'natr_norm'),
                    ('spread_percentage', 'spread_percentage_norm'),
                    ('bid_ask_ratio', 'bid_ask_ratio_norm'),
                    ('large_order_imbalance', 'large_order_imbalance_norm')
                ]:
                    if norm_col in row and pd.notna(row[norm_col]) and row[norm_col] >= 0.8:
                        signal_tracker[symbol].append((ts, feat, row[norm_col]))

                # Clean old signals (> 6 min)
                cutoff = ts - timedelta(minutes=6)
                signal_tracker[symbol] = [
                    s for s in signal_tracker[symbol] if s[0] >= cutoff
                ]

                # Check for entry signal
                hits = signal_tracker[symbol]
                has_natr = any(h[1] == 'natr' for h in hits)
                has_orderbook = any(h[1] in ['spread_percentage', 'bid_ask_ratio',
                                              'large_order_imbalance'] for h in hits)

                if has_natr and has_orderbook:
                    # Can we open a position?
                    if symbol not in self.open_positions and len(self.open_positions) < self.max_positions:
                        # Get entry price from candles
                        entry_price = self.get_price_at_time(symbol, ts)

                        if entry_price and entry_price > 0:
                            quantity = self.position_size / entry_price

                            pos = BacktestPosition(
                                symbol=symbol,
                                entry_price=entry_price,
                                entry_time=ts,
                                quantity=quantity,
                                position_size=self.position_size
                            )

                            self.open_positions[symbol] = pos
                            entry_count += 1

                            # Get trigger features
                            trigger_features = [h[1] for h in hits]

                            self.trade_log.append({
                                'type': 'entry',
                                'symbol': symbol,
                                'timestamp': str(ts),
                                'price': entry_price,
                                'quantity': quantity,
                                'triggers': trigger_features
                            })

                            # Clear signal tracker for this symbol after entry
                            signal_tracker[symbol] = []

        print(f"\n  Total entries: {entry_count}")
        print(f"  Closed positions: {len(self.closed_positions)}")
        print(f"  Still open at end: {len(self.open_positions)}")

        return self.calculate_results()

    def calculate_results(self) -> Dict:
        """Calculate and return backtest results"""
        results = {
            'total_trades': len(self.closed_positions),
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_rate': 0.0,
            'avg_hold_minutes': 0.0,
            'max_drawdown': 0.0,
            'exit_reasons': defaultdict(int),
            'trades_by_symbol': defaultdict(int)
        }

        hold_times = []
        equity_curve = [self.starting_capital]
        peak_equity = self.starting_capital

        for pos in self.closed_positions:
            results['trades_by_symbol'][pos.symbol] += 1
            results['exit_reasons'][pos.exit_reason] += 1

            if pos.pnl > 0:
                results['winning_trades'] += 1
                results['gross_profit'] += pos.pnl
            else:
                results['losing_trades'] += 1
                results['gross_loss'] += abs(pos.pnl)

            results['total_pnl'] += pos.pnl

            if pos.exit_time and pos.entry_time:
                hold_minutes = (pos.exit_time - pos.entry_time).total_seconds() / 60
                hold_times.append(hold_minutes)

            # Track equity curve
            new_equity = equity_curve[-1] + pos.pnl
            equity_curve.append(new_equity)

            if new_equity > peak_equity:
                peak_equity = new_equity

            drawdown = (peak_equity - new_equity) / peak_equity
            if drawdown > results['max_drawdown']:
                results['max_drawdown'] = drawdown

        # Calculate averages
        if results['total_trades'] > 0:
            results['win_rate'] = results['winning_trades'] / results['total_trades'] * 100

            if results['winning_trades'] > 0:
                results['avg_win'] = results['gross_profit'] / results['winning_trades']

            if results['losing_trades'] > 0:
                results['avg_loss'] = results['gross_loss'] / results['losing_trades']

            if hold_times:
                results['avg_hold_minutes'] = np.mean(hold_times)

        results['final_capital'] = self.starting_capital + results['total_pnl']
        results['return_pct'] = results['total_pnl'] / self.starting_capital * 100

        return results

    def print_results(self, results: Dict):
        """Print formatted results"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        print(f"\nCapital: ${self.starting_capital:,.2f} -> ${results['final_capital']:,.2f}")
        print(f"Total P&L: ${results['total_pnl']:,.2f} ({results['return_pct']:.2f}%)")
        print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")

        print(f"\n--- Trade Statistics ---")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Winning: {results['winning_trades']} | Losing: {results['losing_trades']}")
        print(f"Gross Profit: ${results['gross_profit']:,.2f}")
        print(f"Gross Loss: ${results['gross_loss']:,.2f}")
        print(f"Avg Win: ${results['avg_win']:,.2f}")
        print(f"Avg Loss: ${results['avg_loss']:,.2f}")
        print(f"Avg Hold Time: {results['avg_hold_minutes']:.1f} minutes")

        print(f"\n--- Exit Reasons ---")
        for reason, count in sorted(results['exit_reasons'].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

        print(f"\n--- Top Symbols by Trade Count ---")
        top_symbols = sorted(results['trades_by_symbol'].items(), key=lambda x: -x[1])[:10]
        for symbol, count in top_symbols:
            print(f"  {symbol}: {count} trades")

    def save_results(self, results: Dict, output_path: str = "backtest_results.json"):
        """Save results and trade log to JSON"""
        output = {
            'results': {k: v if not isinstance(v, defaultdict) else dict(v)
                       for k, v in results.items()},
            'trade_log': self.trade_log,
            'parameters': {
                'starting_capital': self.starting_capital,
                'position_size': self.position_size,
                'max_positions': self.max_positions,
                'stop_loss_pct': self.stop_loss_pct,
                'partial_exit_1_pct': self.partial_exit_1_pct,
                'partial_exit_2_pct': self.partial_exit_2_pct,
                'trailing_stop_activation': self.trailing_stop_activation,
                'trailing_stop_distance': self.trailing_stop_distance,
                'max_hold_minutes': self.max_hold_minutes
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to {output_path}")


def main():
    import sys

    # Use default or provided db path
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/pythia.duckdb"

    # Check if database exists
    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    # Run backtest
    backtester = StrategyBacktester(db_path)
    results = backtester.run_backtest()

    # Print and save results
    backtester.print_results(results)
    backtester.save_results(results)


if __name__ == "__main__":
    main()
