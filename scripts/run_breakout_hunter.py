#!/usr/bin/env python3
"""
Real-time Breakout Hunter Monitor

Runs alongside the collector to detect and alert on breakout signals.
Currently in monitoring/alert mode - can be extended for live trading.
"""

import asyncio
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import signal

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.breakout_hunter import BreakoutHunter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BreakoutMonitor:
    """Real-time monitor for breakout signals"""

    def __init__(self, db_path: str = 'data/feature_buffer.db',
                 check_interval: int = 60):
        """
        Initialize the monitor.

        Args:
            db_path: Path to feature buffer database
            check_interval: Seconds between checks
        """
        self.db_path = db_path
        self.check_interval = check_interval
        self.hunter = BreakoutHunter(paper_trading=True)
        self.last_hour_checked = None
        self.running = True

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received, stopping...")
        self.running = False

    def get_recent_hourly_data(self) -> pd.DataFrame:
        """Get recent hourly OHLCV data from database"""
        conn = sqlite3.connect(self.db_path)

        # Get last 48 hours of data
        query = '''
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE timestamp > datetime('now', '-48 hours')
        ORDER BY symbol, timestamp
        '''

        df = pd.read_sql(query, conn)
        conn.close()

        if len(df) == 0:
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        df['hour'] = df['timestamp'].dt.floor('h')

        # Aggregate to hourly
        hourly = df.groupby(['symbol', 'hour']).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        hourly = hourly.rename(columns={'hour': 'timestamp'})

        return hourly

    def check_for_signals(self):
        """Check for new breakout signals using T+2 resumption pattern"""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)

        # Get fresh data
        hourly = self.get_recent_hourly_data()

        if len(hourly) == 0:
            logger.warning("No data available")
            return

        # Update volume data for all symbols
        for symbol in hourly['symbol'].unique():
            sym_data = hourly[hourly['symbol'] == symbol].sort_values('timestamp')
            self.hunter.update_hourly_data(symbol, sym_data)

        # Get the most recent COMPLETED hour (exclude current incomplete hour)
        # Data timestamps are in UTC, so compare against current UTC hour
        current_utc_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
        completed_hourly = hourly[hourly['timestamp'] < current_utc_hour]

        if len(completed_hourly) == 0:
            logger.warning("No completed hourly data available yet")
            return

        latest_hour = completed_hourly['timestamp'].max()

        # If we already checked this hour, only check exits
        if self.last_hour_checked and latest_hour <= self.last_hour_checked:
            current_time = datetime.now()

            for symbol in list(self.hunter.active_trades.keys()):
                sym_data = hourly[hourly['symbol'] == symbol]
                if len(sym_data) > 0:
                    current_price = sym_data['close'].iloc[-1]
                    exit_reason = self.hunter.check_exit_conditions(symbol, current_price, current_time)
                    if exit_reason:
                        self.hunter.exit_trade(symbol, current_price, current_time, exit_reason)
            return

        # New hour - check for breakouts
        self.last_hour_checked = latest_hour
        logger.info(f"Checking hour: {latest_hour}")

        # Get candles for the latest COMPLETED hour
        latest_candles = completed_hourly[completed_hourly['timestamp'] == latest_hour]

        # Check for new T+0 breakouts
        for _, row in latest_candles.iterrows():
            candle = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'timestamp': row['timestamp']
            }
            self.hunter.check_for_breakout(row['symbol'], candle)

        # Update T+1 data for pending signals (v5.0: T+1 must be positive)
        for symbol in list(self.hunter.pending_signals.keys()):
            signal = self.hunter.pending_signals[symbol]
            if signal.status == 'pending_t1':
                hours_since = (latest_hour - signal.signal_time).total_seconds() / 3600
                if hours_since >= 1:
                    sym_candle = latest_candles[latest_candles['symbol'] == symbol]
                    if len(sym_candle) > 0:
                        row = sym_candle.iloc[0]
                        self.hunter.update_t1_data(symbol, row['close'], latest_hour, row['volume'])

        # Check T+2 confirmations (v5.0: need T+2 > 5% AND vol continuation > 1x)
        for symbol in list(self.hunter.pending_signals.keys()):
            signal = self.hunter.pending_signals[symbol]
            if signal.status == 'pending_t2':
                hours_since = (latest_hour - signal.signal_time).total_seconds() / 3600
                if hours_since >= 2:
                    sym_candle = latest_candles[latest_candles['symbol'] == symbol]
                    if len(sym_candle) > 0:
                        row = sym_candle.iloc[0]
                        # Calculate post-signal volume (last 3 hours)
                        sym_data = hourly[hourly['symbol'] == symbol].sort_values('timestamp')
                        post_vol = sym_data.tail(3)['volume'].sum() if len(sym_data) >= 3 else 0
                        self.hunter.check_confirmation(
                            symbol, row['close'], latest_hour,
                            t2_open=row['open'], post_signal_volume=post_vol
                        )

        # Check exit conditions for active trades
        for symbol in list(self.hunter.active_trades.keys()):
            sym_candle = latest_candles[latest_candles['symbol'] == symbol]
            if len(sym_candle) > 0:
                row = sym_candle.iloc[0]
                exit_reason = self.hunter.check_exit_conditions(symbol, row['close'], latest_hour)
                if exit_reason:
                    self.hunter.exit_trade(symbol, row['close'], latest_hour, exit_reason)

    def run(self):
        """Main run loop"""
        version = "5.3"  # T+2 entry with enhanced exits
        entry_mode = "T+1" if self.hunter.ENTER_ON_T1 else "T+2"
        logger.info("=" * 60)
        logger.info(f"BREAKOUT HUNTER v{version} - Enter on {entry_mode}")
        logger.info("=" * 60)
        logger.info(f"Entry: T+0>{self.hunter.INITIAL_MOVE_PCT}%, "
                   f"Vol>{self.hunter.VOLUME_THRESHOLD}x, "
                   f"T+1>{self.hunter.T1_MIN_RETURN_PCT}% → ENTER")
        lock_str = ", ".join([f"+{t*100:.0f}%→{l*100:.1f}%" for t, l in self.hunter.PROFIT_LOCK_LEVELS])
        trail_str = f"{self.hunter.TRAIL_STOP_PCT*100:.0f}%@+{self.hunter.TRAIL_TRIGGER_PCT*100:.0f}%"
        for thresh, pct in self.hunter.TRAIL_TIGHTEN_LEVELS:
            trail_str += f", {pct*100:.0f}%@+{thresh*100:.0f}%"
        logger.info(f"Exit: SL={self.hunter.INITIAL_STOP_LOSS_PCT*100:.0f}%, "
                   f"Locks=[{lock_str}], Trail=[{trail_str}]")
        logger.info(f"Checking every {self.check_interval} seconds...")
        logger.info("")

        while self.running:
            try:
                self.check_for_signals()

                # Print status every check
                stats = self.hunter.get_stats()
                pending = len(self.hunter.pending_signals)
                active = len(self.hunter.active_trades)

                if pending > 0 or active > 0:
                    status = f"Pending: {pending} | Active: {active}"
                    if stats['total_trades'] > 0:
                        status += f" | Completed: {stats['total_trades']} ({stats['win_rate']:.0f}% win)"
                    logger.info(status)

                # Save state for dashboard
                self.hunter.save_state()

            except Exception as e:
                logger.error(f"Error: {e}")

            # Wait for next check
            for _ in range(self.check_interval):
                if not self.running:
                    break
                import time
                time.sleep(1)

        # Final status
        logger.info("\nFinal Status:")
        self.hunter.print_status()


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Breakout Hunter Monitor')
    parser.add_argument('--db', default='data/feature_buffer.db',
                       help='Path to feature buffer database')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds')
    args = parser.parse_args()

    monitor = BreakoutMonitor(db_path=args.db, check_interval=args.interval)
    monitor.run()


if __name__ == '__main__':
    main()
