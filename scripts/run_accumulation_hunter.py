#!/usr/bin/env python3
"""
Real-time Accumulation Hunter Monitor v6.0

Runs alongside the collector to detect stealth accumulation patterns
using order book data. Enters on confirmed breakout.

This runs independently of Breakout Hunter v5.1 - both systems can
operate simultaneously.

v6.1: Now uses SQLite (feature_buffer.db) for concurrent read access
      while collector writes to DuckDB.
"""

import logging
import json
import signal
import time
import sys
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.accumulation_hunter import AccumulationHunter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AccumulationMonitor:
    """Real-time monitor for accumulation patterns"""

    def __init__(self, db_path: str = 'data/feature_buffer.db',
                 check_interval: int = 60,
                 state_file: str = 'accumulation_hunter_state.json'):
        """
        Initialize the monitor.

        Args:
            db_path: Path to SQLite database (feature_buffer.db)
            check_interval: Seconds between checks
            state_file: Path to state file for dashboard
        """
        self.db_path = db_path
        self.check_interval = check_interval
        self.state_file = state_file
        self.hunter = AccumulationHunter(paper_trading=True)
        self.running = True
        self.last_baseline_update = None

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received, stopping...")
        self.running = False

    def get_connection(self):
        """Get a SQLite connection"""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def update_baselines(self):
        """Update baseline metrics for all symbols"""
        conn = self.get_connection()

        try:
            # Get latest timestamp to determine data range
            cursor = conn.execute("SELECT MAX(timestamp) as latest FROM order_book_snapshots")
            row = cursor.fetchone()
            latest_ts = row[0] if row else None

            if latest_ts is None:
                logger.warning("No order book data available")
                return

            # Calculate cutoff (48 hours before latest)
            from dateutil import parser as dateutil_parser
            if isinstance(latest_ts, str):
                latest_dt = dateutil_parser.isoparse(latest_ts)
            else:
                latest_dt = latest_ts
            cutoff_48h = (latest_dt - timedelta(hours=48)).isoformat()

            # Get all symbols with data in last 48 hours
            cursor = conn.execute("""
                SELECT DISTINCT symbol FROM order_book_snapshots
                WHERE timestamp > ?
            """, (cutoff_48h,))
            symbols = [row[0] for row in cursor.fetchall()]

            logger.info(f"Updating baselines for {len(symbols)} symbols (data as of {latest_ts})...")

            for symbol in symbols:
                try:
                    # Get order book data from first half of available data (baseline)
                    cursor = conn.execute("""
                        SELECT bids, asks, mid_price, timestamp
                        FROM order_book_snapshots
                        WHERE symbol = ?
                          AND bids IS NOT NULL
                        ORDER BY timestamp
                        LIMIT 50
                    """, (symbol,))
                    baseline_rows = cursor.fetchall()

                    if len(baseline_rows) < 10:
                        continue

                    # Calculate baseline depths
                    bid_depths = []
                    ask_depths = []
                    bars = []

                    for row in baseline_rows:
                        try:
                            bids = json.loads(row['bids']) if row['bids'] else []
                            asks = json.loads(row['asks']) if row['asks'] else []
                            bid_depth = sum(float(b[1]) for b in bids[:10])
                            ask_depth = sum(float(a[1]) for a in asks[:10])
                            if ask_depth > 0:
                                bid_depths.append(bid_depth)
                                ask_depths.append(ask_depth)
                                bars.append(bid_depth / ask_depth)
                        except (json.JSONDecodeError, TypeError, IndexError):
                            continue

                    if not bid_depths:
                        continue

                    # Get baseline volume from trades (estimate from first hour of data)
                    cursor = conn.execute("""
                        SELECT MIN(timestamp) FROM order_book_snapshots WHERE symbol = ?
                    """, (symbol,))
                    min_ts_row = cursor.fetchone()
                    if min_ts_row and min_ts_row[0]:
                        min_ts = dateutil_parser.isoparse(min_ts_row[0]) if isinstance(min_ts_row[0], str) else min_ts_row[0]
                        vol_cutoff = (min_ts + timedelta(hours=24)).isoformat()
                        cursor = conn.execute("""
                            SELECT SUM(size * price) as volume
                            FROM trades
                            WHERE symbol = ?
                              AND timestamp > ?
                              AND timestamp < ?
                        """, (symbol, min_ts.isoformat(), vol_cutoff))
                        vol_data = cursor.fetchone()
                        volume = vol_data[0] if vol_data and vol_data[0] else 0
                    else:
                        volume = 0

                    hours = max(len(baseline_rows) / 4, 1)  # Estimate hours covered
                    avg_hourly_volume = volume / hours

                    # Update baseline
                    self.hunter.update_baseline(symbol, {
                        'avg_bid_depth': np.mean(bid_depths),
                        'avg_ask_depth': np.mean(ask_depths),
                        'avg_bar': np.mean(bars),
                        'avg_volume': max(avg_hourly_volume, 1),  # Prevent division by zero
                    })

                except Exception as e:
                    logger.debug(f"Error updating baseline for {symbol}: {e}")
                    continue

            self.last_baseline_update = datetime.now()
            logger.info(f"Baselines updated for {len(self.hunter.baseline_data)} symbols")

        finally:
            conn.close()

    def check_for_signals(self):
        """Check for accumulation signals"""
        conn = self.get_connection()

        try:
            from dateutil import parser as dateutil_parser

            # Get latest timestamp to work with historical or live data
            cursor = conn.execute("SELECT MAX(timestamp) as latest FROM order_book_snapshots")
            row = cursor.fetchone()
            latest_ts = row[0] if row else None

            if latest_ts is None:
                logger.warning("No order book data available")
                return

            # Parse latest timestamp
            if isinstance(latest_ts, str):
                latest_dt = dateutil_parser.isoparse(latest_ts)
            else:
                latest_dt = latest_ts
            current_time = latest_dt

            # Calculate cutoffs
            cutoff_1h = (latest_dt - timedelta(hours=1)).isoformat()
            cutoff_3h = (latest_dt - timedelta(hours=3)).isoformat()

            # Get latest order book snapshots (most recent per symbol)
            cursor = conn.execute("""
                SELECT symbol, timestamp, bids, asks, mid_price
                FROM order_book_snapshots
                WHERE timestamp > ?
                ORDER BY symbol, timestamp DESC
            """, (cutoff_1h,))
            snapshot_rows = cursor.fetchall()

            if len(snapshot_rows) == 0:
                logger.warning("No recent order book data available")
                return

            # Convert to DataFrame and get latest per symbol
            snapshot_data = [dict(row) for row in snapshot_rows]
            latest_snapshots = pd.DataFrame(snapshot_data)
            latest_by_symbol = latest_snapshots.groupby('symbol').first().reset_index()

            # Get recent volume from trades (last 3 hours relative to latest data)
            cursor = conn.execute("""
                SELECT symbol,
                       SUM(size * price) as volume,
                       MIN(price) as low_price,
                       MAX(price) as high_price
                FROM trades
                WHERE timestamp > ?
                GROUP BY symbol
            """, (cutoff_3h,))
            volume_rows = cursor.fetchall()
            volume_data = [dict(row) for row in volume_rows]
            recent_volume = pd.DataFrame(volume_data) if volume_data else pd.DataFrame()

            volume_by_symbol = {row['symbol']: row for _, row in recent_volume.iterrows()} if len(recent_volume) > 0 else {}

            # Get earliest prices for comparison (baseline period)
            cursor = conn.execute("SELECT MIN(timestamp) FROM order_book_snapshots")
            min_ts_row = cursor.fetchone()
            if min_ts_row and min_ts_row[0]:
                min_ts = dateutil_parser.isoparse(min_ts_row[0]) if isinstance(min_ts_row[0], str) else min_ts_row[0]
                baseline_cutoff = (min_ts + timedelta(hours=1)).isoformat()
                cursor = conn.execute("""
                    SELECT symbol, mid_price
                    FROM order_book_snapshots
                    WHERE timestamp < ?
                """, (baseline_cutoff,))
                baseline_rows = cursor.fetchall()
                baseline_data = [dict(row) for row in baseline_rows]
                if baseline_data:
                    prices_baseline = pd.DataFrame(baseline_data).groupby('symbol').first()
                else:
                    prices_baseline = pd.DataFrame()
            else:
                prices_baseline = pd.DataFrame()

            # Process each symbol
            for _, row in latest_by_symbol.iterrows():
                symbol = row['symbol']

                # Skip if no baseline
                if symbol not in self.hunter.baseline_data:
                    continue

                try:
                    # Parse order book
                    bids = json.loads(row['bids']) if row['bids'] else []
                    asks = json.loads(row['asks']) if row['asks'] else []

                    if not bids or not asks:
                        continue

                    # Calculate depths
                    bid_depth = sum(float(b[1]) for b in bids[:10])
                    ask_depth = sum(float(a[1]) for a in asks[:10])

                    if ask_depth == 0:
                        continue

                    # Get volume
                    vol_info = volume_by_symbol.get(symbol)
                    if vol_info is not None and isinstance(vol_info, dict):
                        current_volume = vol_info.get('volume', 0) or 0
                    elif hasattr(vol_info, 'volume'):
                        current_volume = vol_info['volume'] or 0
                    else:
                        current_volume = 0
                    hourly_volume = current_volume / 3  # Convert 3h to hourly

                    # Get baseline price
                    try:
                        price_baseline = prices_baseline.loc[symbol, 'mid_price'] if symbol in prices_baseline.index else row['mid_price']
                    except:
                        price_baseline = row['mid_price']

                    # Compute metrics
                    metrics = self.hunter.compute_accumulation_metrics(
                        symbol=symbol,
                        current_bid_depth=bid_depth,
                        current_ask_depth=ask_depth,
                        current_volume=hourly_volume,
                        current_price=row['mid_price'],
                        price_24h_ago=price_baseline,
                    )

                    if not metrics:
                        continue

                    # Check for new accumulation
                    if symbol not in self.hunter.watch_list:
                        self.hunter.check_for_accumulation(symbol, metrics, current_time)
                    else:
                        # Update existing signal
                        self.hunter.update_accumulation_state(symbol, metrics, current_time)

                        # Check for breakout entry
                        vol_ratio = metrics['volume_ratio']
                        self.hunter.check_entry_conditions(
                            symbol, row['mid_price'], vol_ratio, current_time
                        )

                except (json.JSONDecodeError, TypeError, KeyError, IndexError) as e:
                    logger.debug(f"Error processing {symbol}: {e}")
                    continue

            # Check exit conditions for active trades
            for symbol in list(self.hunter.active_trades.keys()):
                try:
                    # Get latest price
                    price_row = latest_by_symbol[latest_by_symbol['symbol'] == symbol]
                    if len(price_row) > 0:
                        current_price = price_row.iloc[0]['mid_price']
                        exit_reason = self.hunter.check_exit_conditions(
                            symbol, current_price, current_time
                        )
                        if exit_reason:
                            self.hunter.exit_trade(symbol, current_price, current_time, exit_reason)
                except Exception as e:
                    logger.error(f"Error checking exit for {symbol}: {e}")

        finally:
            conn.close()

    def run(self):
        """Main run loop"""
        logger.info("=" * 60)
        logger.info("ACCUMULATION HUNTER v6.0 - Stealth Pattern Detection")
        logger.info("=" * 60)
        logger.info(f"Detection: Vol>{self.hunter.VOLUME_ANOMALY_THRESHOLD}x, "
                   f"Price<{self.hunter.PRICE_FLAT_MAX*100}%, "
                   f"BAR>{self.hunter.BAR_WATCH}x (watch), "
                   f">{self.hunter.BAR_STRONG}x (strong)")
        logger.info(f"Entry: Breakout>{self.hunter.BREAKOUT_PRICE_PCT*100}%, "
                   f"Vol>{self.hunter.BREAKOUT_VOLUME_RATIO}x")
        lock_str = ", ".join([f"+{t*100:.0f}%→{l*100:.0f}%" for t, l in self.hunter.PROFIT_LOCK_LEVELS])
        logger.info(f"Exit: SL={self.hunter.INITIAL_STOP_LOSS_PCT*100:.0f}%, "
                   f"Locks=[{lock_str}], "
                   f"Trail={self.hunter.TRAIL_STOP_PCT*100:.0f}%@+{self.hunter.TRAIL_TRIGGER_PCT*100:.0f}%")
        logger.info(f"Position: ${self.hunter.POSITION_SIZE_USD} x {self.hunter.MAX_POSITIONS} max")
        logger.info(f"Checking every {self.check_interval} seconds...")
        logger.info("")

        # Initial baseline update
        self.update_baselines()

        while self.running:
            try:
                # Update baselines every hour
                if (self.last_baseline_update is None or
                    datetime.now() - self.last_baseline_update > timedelta(hours=1)):
                    self.update_baselines()

                # Check for signals
                self.check_for_signals()

                # Print status
                stats = self.hunter.get_stats()
                watch_count = len(self.hunter.watch_list)
                active = len(self.hunter.active_trades)

                # Count by status
                accumulating = sum(1 for s in self.hunter.watch_list.values()
                                  if s.status in ('accumulating', 'ready'))

                if watch_count > 0 or active > 0:
                    status = f"Watch: {watch_count} ({accumulating} accumulating) | Active: {active}"
                    if stats['total_trades'] > 0:
                        status += f" | Completed: {stats['total_trades']} ({stats['win_rate']:.0f}% win)"
                    logger.info(status)

                    # Show top signals
                    if watch_count > 0:
                        top_signals = sorted(
                            self.hunter.watch_list.values(),
                            key=lambda x: x.signal_strength,
                            reverse=True
                        )[:3]
                        for sig in top_signals:
                            logger.info(f"  {sig.symbol}: Score={sig.signal_strength:.0f} | "
                                       f"Status={sig.status} | Hours={sig.accumulation_hours:.1f} | "
                                       f"BAR={sig.bar_multiple:.1f}x")

                # Save state for dashboard
                self.hunter.save_state(self.state_file)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()

            # Wait for next check
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)

        # Final status
        logger.info("\nFinal Status:")
        self.hunter.print_status()
        self.hunter.save_state(self.state_file)


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Accumulation Hunter Monitor v6.1 (SQLite)')
    parser.add_argument('--db', default='data/feature_buffer.db',
                       help='Path to SQLite database (feature_buffer.db)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds')
    parser.add_argument('--state-file', default='accumulation_hunter_state.json',
                       help='Path to state file for dashboard')
    args = parser.parse_args()

    monitor = AccumulationMonitor(
        db_path=args.db,
        check_interval=args.interval,
        state_file=args.state_file
    )
    monitor.run()


if __name__ == '__main__':
    main()
