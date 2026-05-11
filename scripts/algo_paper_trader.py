#!/usr/bin/env python3
"""
Algorithmic Paper Trader - NATR + Order Book Entry Strategy

Entry Logic:
- NATR normalized >= 0.8 AND any order book feature >= 0.8 within 6-minute window
- Order book features: bid_ask_spread_pct, order_book_depth_ratio, large_order_imbalance

Exit Strategy (from live_paper_trader.py):
- Stop Loss: -1%
- Partial Take Profit: 25% at +20%, 25% at +30%
- Trailing Stop: Activates at +15%, trails by 6.5%
- Max Hold: 180 minutes

Usage:
    python scripts/algo_paper_trader.py
"""

import duckdb
import pandas as pd
import numpy as np
import time
import json
import signal
import sys
import shutil
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/algo_paper_trader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Position:
    """Represents an open or closed position"""
    symbol: str
    entry_price: float
    entry_time: datetime
    quantity: float
    position_size: float  # Dollar amount invested

    # Tracking
    peak_price: float = 0.0
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""

    # Exit management
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0
    partial_exits: List[int] = field(default_factory=list)
    remaining_quantity: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self):
        self.peak_price = self.entry_price
        self.remaining_quantity = self.quantity

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def pnl_pct(self, current_price: float) -> float:
        """Calculate P&L percentage"""
        return (current_price - self.entry_price) / self.entry_price

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason,
            'position_size': self.position_size,
            'quantity': self.quantity,
            'realized_pnl': self.realized_pnl,
            'partial_exits': self.partial_exits,
        }


@dataclass
class Portfolio:
    """Portfolio state"""
    starting_capital: float = 10000.0
    cash: float = 10000.0
    position_size: float = 2500.0
    max_positions: int = 4

    open_positions: List[Position] = field(default_factory=list)
    closed_positions: List[Position] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.closed_positions)

    @property
    def total_pnl_pct(self) -> float:
        return (self.total_pnl / self.starting_capital) * 100

    def can_open_position(self) -> bool:
        return (len(self.open_positions) < self.max_positions and
                self.cash >= self.position_size)

    def to_dict(self) -> dict:
        return {
            'starting_capital': self.starting_capital,
            'cash': self.cash,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'open_positions': len(self.open_positions),
            'closed_positions': len(self.closed_positions),
            'positions': [p.to_dict() for p in self.closed_positions],
        }


# ============================================================================
# Feature Window Tracker
# ============================================================================

class FeatureWindowTracker:
    """
    Tracks feature hits within a rolling time window.
    Entry signal fires when NATR AND any order book feature both hit >= 0.8
    within the window.
    """

    ORDERBOOK_FEATURES = ['bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance']

    def __init__(self, window_minutes: int = 6, threshold: float = 0.8):
        self.window_minutes = window_minutes
        self.threshold = threshold
        # {symbol: [(timestamp, feature_name, normalized_value), ...]}
        self.feature_hits: Dict[str, List[Tuple[datetime, str, float]]] = defaultdict(list)
        # Track when we last triggered for each symbol (avoid repeat triggers)
        self.last_trigger: Dict[str, datetime] = {}
        self.cooldown_minutes = 30  # Don't re-trigger same symbol within 30 min

    def record_hit(self, symbol: str, timestamp: datetime, feature: str, normalized_value: float):
        """Record a feature value if it exceeds threshold"""
        if normalized_value >= self.threshold:
            self.feature_hits[symbol].append((timestamp, feature, normalized_value))
            self._cleanup_old(symbol, timestamp)

    def _cleanup_old(self, symbol: str, current_time: datetime):
        """Remove hits older than the window"""
        cutoff = current_time - timedelta(minutes=self.window_minutes)
        self.feature_hits[symbol] = [
            h for h in self.feature_hits[symbol] if h[0] >= cutoff
        ]

    def check_entry_signal(self, symbol: str, current_time: datetime) -> Optional[dict]:
        """
        Check if both NATR and order book feature hit within window.
        Returns signal details if triggered, None otherwise.
        """
        self._cleanup_old(symbol, current_time)

        # Check cooldown
        if symbol in self.last_trigger:
            if current_time - self.last_trigger[symbol] < timedelta(minutes=self.cooldown_minutes):
                return None

        hits = self.feature_hits[symbol]

        # Find NATR hits
        natr_hits = [h for h in hits if h[1] == 'NATR']

        # Find order book hits
        orderbook_hits = [h for h in hits if h[1] in self.ORDERBOOK_FEATURES]

        if natr_hits and orderbook_hits:
            # Get the details
            best_natr = max(natr_hits, key=lambda x: x[2])
            best_orderbook = max(orderbook_hits, key=lambda x: x[2])

            # Record trigger
            self.last_trigger[symbol] = current_time

            # Clear hits for this symbol
            self.feature_hits[symbol] = []

            return {
                'natr_value': best_natr[2],
                'natr_time': best_natr[0],
                'orderbook_feature': best_orderbook[1],
                'orderbook_value': best_orderbook[2],
                'orderbook_time': best_orderbook[0],
            }

        return None


# ============================================================================
# Algo Paper Trader
# ============================================================================

class AlgoPaperTrader:
    """
    Main paper trading engine.
    Reads features from database, normalizes them, checks for entry signals,
    and manages positions with the exit strategy.
    """

    FEATURES_TO_TRACK = ['NATR', 'bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance']

    def __init__(
        self,
        db_path: str = "market_data.duckdb",
        starting_capital: float = 10000.0,
        position_size: float = 2500.0,
        max_positions: int = 4,
        window_minutes: int = 6,
        threshold: float = 0.8,
        scan_interval: int = 60,  # seconds
    ):
        self.db_path = db_path
        self.scan_interval = scan_interval

        self.portfolio = Portfolio(
            starting_capital=starting_capital,
            cash=starting_capital,
            position_size=position_size,
            max_positions=max_positions,
        )

        self.feature_tracker = FeatureWindowTracker(
            window_minutes=window_minutes,
            threshold=threshold
        )

        # For z-score normalization (rolling stats)
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.stats_lookback_hours = 1  # Use 1 hour of data for normalization stats

        # Running state
        self.running = False
        self.start_time = None
        self.scan_count = 0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("=" * 70)
        logger.info("ALGO PAPER TRADER INITIALIZED")
        logger.info("=" * 70)
        logger.info(f"  Starting Capital: ${starting_capital:,.2f}")
        logger.info(f"  Position Size: ${position_size:,.2f}")
        logger.info(f"  Max Positions: {max_positions}")
        logger.info(f"  Entry Window: {window_minutes} minutes")
        logger.info(f"  Entry Threshold: {threshold} normalized")
        logger.info(f"  Scan Interval: {scan_interval} seconds")
        logger.info("=" * 70)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("\nShutdown signal received...")
        self.running = False

    def _ensure_collector_running(self) -> bool:
        """Check if collector is running, start it if not."""
        import subprocess
        import os

        # Check if collector process exists
        result = subprocess.run(
            ['pgrep', '-f', 'integrated_collector'],
            capture_output=True
        )

        if result.returncode != 0:
            # Collector not running - start it
            logger.warning("Collector not running - starting it now...")
            subprocess.Popen(
                [sys.executable, '-m', 'src.data_ingestion.integrated_collector'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            # Wait for collector to initialize and start collecting data
            logger.info("Waiting 30s for collector to initialize...")
            time.sleep(30)
            logger.info("Collector started")
        else:
            logger.info("Collector already running")

        return True

    def _copy_database(self) -> str:
        """Copy main database to temp location to avoid lock conflicts.

        Uses unique filenames and validates copies to handle race conditions
        with the collector writing to the source database.
        """
        import uuid
        import os

        # Use unique filename to avoid corrupting existing copy
        unique_id = uuid.uuid4().hex[:8]
        temp_path = f"/tmp/market_data_paper_trader_{unique_id}.duckdb"

        try:
            # Copy to unique temp file
            shutil.copy2(self.db_path, temp_path)

            # Validate the copy by trying to open it
            try:
                test_conn = duckdb.connect(temp_path, read_only=True)
                test_conn.execute("SELECT 1").fetchone()
                test_conn.close()
                logger.debug(f"Copied and validated database to {temp_path}")

                # Clean up old temp files (keep only last 3)
                self._cleanup_old_temp_files()

                return temp_path
            except Exception as validate_err:
                logger.warning(f"Copy validation failed: {validate_err}")
                # Remove corrupted copy
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise validate_err

        except Exception as e:
            logger.warning(f"Failed to copy database: {e}")
            # Clean up failed copy
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise e

    def _cleanup_old_temp_files(self):
        """Remove old temporary database copies"""
        import glob
        import os

        temp_files = glob.glob("/tmp/market_data_paper_trader_*.duckdb")
        if len(temp_files) > 3:
            # Sort by modification time, remove oldest
            temp_files.sort(key=os.path.getmtime)
            for old_file in temp_files[:-3]:
                try:
                    os.remove(old_file)
                except:
                    pass

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get read-only database connection from temp copy with retry"""
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                temp_path = self._copy_database()
                return duckdb.connect(temp_path, read_only=True)
            except Exception as e:
                last_error = e
                logger.warning(f"Database copy attempt {attempt + 1} failed: {e}")
                time.sleep(1)  # Wait before retry

        # All retries failed
        raise last_error

    def _get_symbols(self) -> List[str]:
        """Get all symbols from features table"""
        conn = self._get_connection()
        try:
            result = conn.execute("""
                SELECT DISTINCT symbol FROM features
                WHERE timestamp > now() - INTERVAL '1 hour'
            """).fetchall()
            return [r[0] for r in result if r[0]]
        finally:
            conn.close()

    def _query_recent_features(self, minutes: int = 6) -> pd.DataFrame:
        """Query recent features from database"""
        conn = self._get_connection()
        try:
            query = f"""
                SELECT symbol, timestamp, NATR, bid_ask_spread_pct,
                       order_book_depth_ratio, large_order_imbalance
                FROM features
                WHERE timestamp > now() - INTERVAL '{minutes} minutes'
                ORDER BY timestamp DESC
            """
            return conn.execute(query).fetchdf()
        except Exception as e:
            logger.error(f"Error querying features: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def _update_normalization_stats(self):
        """Update feature statistics for z-score normalization"""
        conn = self._get_connection()
        try:
            for feature in self.FEATURES_TO_TRACK:
                query = f"""
                    SELECT AVG({feature}) as mean, STDDEV({feature}) as std
                    FROM features
                    WHERE timestamp > now() - INTERVAL '{self.stats_lookback_hours} hours'
                    AND {feature} IS NOT NULL
                """
                result = conn.execute(query).fetchone()
                if result and result[0] is not None and result[1] is not None:
                    self.feature_stats[feature] = {
                        'mean': result[0],
                        'std': max(result[1], 1e-10)  # Avoid division by zero
                    }
        except Exception as e:
            logger.error(f"Error updating normalization stats: {e}")
        finally:
            conn.close()

    def _normalize_value(self, feature: str, value: float) -> float:
        """
        Normalize a feature value to 0-1 scale using z-score.
        Same logic as Spike Visualizer:
        - Calculate z-score
        - Clip to [-3, +3]
        - Map to [0, 1] via: (z + 3) / 6
        """
        if feature not in self.feature_stats:
            return 0.5  # Default to middle if no stats

        stats = self.feature_stats[feature]
        z_score = (value - stats['mean']) / stats['std']
        z_clipped = np.clip(z_score, -3, 3)
        normalized = (z_clipped + 3) / 6

        return normalized

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        conn = self._get_connection()
        try:
            # Try candles first (more accurate)
            result = conn.execute(f"""
                SELECT close FROM candles
                WHERE symbol = '{symbol}'
                ORDER BY timestamp DESC
                LIMIT 1
            """).fetchone()

            if result:
                return result[0]

            # Fall back to tickers
            result = conn.execute(f"""
                SELECT price FROM tickers
                WHERE symbol = '{symbol}'
                ORDER BY timestamp DESC
                LIMIT 1
            """).fetchone()

            return result[0] if result else None
        finally:
            conn.close()

    def _scan_for_entries(self):
        """Scan all symbols for entry signals"""
        # Update normalization stats periodically
        if self.scan_count % 10 == 0:  # Every 10 scans
            self._update_normalization_stats()

        # Query recent features
        features_df = self._query_recent_features(minutes=self.feature_tracker.window_minutes)

        if features_df.empty:
            return

        now = datetime.now(timezone.utc)
        signals_found = 0

        # Process each row
        for _, row in features_df.iterrows():
            symbol = row['symbol']
            if not symbol:
                continue

            timestamp = row['timestamp']
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            # Record hits for each feature
            for feature in self.FEATURES_TO_TRACK:
                if feature in row and pd.notna(row[feature]):
                    normalized = self._normalize_value(feature, row[feature])
                    self.feature_tracker.record_hit(symbol, timestamp, feature, normalized)

            # Check for entry signal (only if we can open position)
            if self.portfolio.can_open_position():
                # Skip if already in position
                if any(p.symbol == symbol for p in self.portfolio.open_positions):
                    continue

                signal = self.feature_tracker.check_entry_signal(symbol, now)
                if signal:
                    signals_found += 1
                    self._enter_position(symbol, signal)

        if signals_found > 0:
            logger.info(f"Found {signals_found} entry signal(s) this scan")

    def _enter_position(self, symbol: str, signal: dict):
        """Enter a new position"""
        price = self._get_current_price(symbol)
        if not price:
            logger.warning(f"Could not get price for {symbol}, skipping entry")
            return

        quantity = self.portfolio.position_size / price

        position = Position(
            symbol=symbol,
            entry_price=price,
            entry_time=datetime.now(timezone.utc),
            quantity=quantity,
            position_size=self.portfolio.position_size,
        )

        self.portfolio.open_positions.append(position)
        self.portfolio.cash -= self.portfolio.position_size

        logger.info("=" * 50)
        logger.info(f"ENTRY: {symbol}")
        logger.info(f"  Price: ${price:.6f}")
        logger.info(f"  Size: ${self.portfolio.position_size:.2f} ({quantity:.6f} units)")
        logger.info(f"  Signal: NATR={signal['natr_value']:.2f}, {signal['orderbook_feature']}={signal['orderbook_value']:.2f}")
        logger.info(f"  Open Positions: {len(self.portfolio.open_positions)}/{self.portfolio.max_positions}")
        logger.info("=" * 50)

    def _monitor_positions(self):
        """Monitor open positions and apply exit strategy"""
        positions_to_close = []

        for position in self.portfolio.open_positions:
            current_price = self._get_current_price(position.symbol)
            if not current_price:
                continue

            pnl_pct = position.pnl_pct(current_price)
            hold_minutes = (datetime.now(timezone.utc) - position.entry_time).total_seconds() / 60

            # Update peak price
            if current_price > position.peak_price:
                position.peak_price = current_price

            # 1. STOP LOSS (-1%)
            if pnl_pct <= -0.01:
                position.exit_price = current_price
                position.exit_time = datetime.now(timezone.utc)
                position.exit_reason = "Stop Loss (-1%)"
                position.realized_pnl = (current_price - position.entry_price) * position.remaining_quantity
                positions_to_close.append(position)
                self._log_exit(position, current_price, pnl_pct)
                continue

            # 2. PARTIAL TAKE PROFITS
            take_profit_levels = [
                (0.20, 0.25, 0, '+20%'),  # 25% at +20%
                (0.30, 0.25, 1, '+30%'),  # 25% at +30%
            ]

            for tp_pct, exit_portion, level_id, label in take_profit_levels:
                if pnl_pct >= tp_pct and level_id not in position.partial_exits:
                    exit_quantity = position.quantity * exit_portion
                    exit_value = exit_quantity * current_price

                    position.partial_exits.append(level_id)
                    position.remaining_quantity -= exit_quantity
                    position.realized_pnl += (current_price - position.entry_price) * exit_quantity
                    self.portfolio.cash += exit_value

                    logger.info(f"PARTIAL EXIT {label}: {position.symbol} - Sold {exit_portion*100:.0f}% at ${current_price:.6f}")

            # 3. TRAILING STOP (activate at +15%, trail by 6.5%)
            if pnl_pct >= 0.15 and not position.trailing_stop_active:
                position.trailing_stop_active = True
                position.trailing_stop_price = current_price * 0.935
                logger.info(f"TRAILING STOP ACTIVATED: {position.symbol} at ${position.trailing_stop_price:.6f}")

            if position.trailing_stop_active:
                new_trailing = current_price * 0.935
                if new_trailing > position.trailing_stop_price:
                    position.trailing_stop_price = new_trailing

                if current_price <= position.trailing_stop_price:
                    position.exit_price = current_price
                    position.exit_time = datetime.now(timezone.utc)
                    position.exit_reason = "Trailing Stop"
                    position.realized_pnl += (current_price - position.entry_price) * position.remaining_quantity
                    self.portfolio.cash += current_price * position.remaining_quantity
                    positions_to_close.append(position)
                    self._log_exit(position, current_price, pnl_pct)
                    continue

            # 4. MAX HOLD TIME (180 minutes)
            if hold_minutes >= 180:
                position.exit_price = current_price
                position.exit_time = datetime.now(timezone.utc)
                position.exit_reason = f"Max Hold Time ({hold_minutes:.0f}min)"
                position.realized_pnl += (current_price - position.entry_price) * position.remaining_quantity
                self.portfolio.cash += current_price * position.remaining_quantity
                positions_to_close.append(position)
                self._log_exit(position, current_price, pnl_pct)
                continue

        # Move closed positions
        for position in positions_to_close:
            self.portfolio.open_positions.remove(position)
            self.portfolio.closed_positions.append(position)

    def _log_exit(self, position: Position, current_price: float, pnl_pct: float):
        """Log position exit"""
        logger.info("=" * 50)
        logger.info(f"EXIT: {position.symbol} - {position.exit_reason}")
        logger.info(f"  Entry: ${position.entry_price:.6f}")
        logger.info(f"  Exit: ${current_price:.6f}")
        logger.info(f"  P&L: {pnl_pct*100:+.2f}% (${position.realized_pnl:+.2f})")
        logger.info("=" * 50)

    def _log_status(self):
        """Log current portfolio status"""
        logger.info("-" * 50)
        logger.info(f"SCAN #{self.scan_count} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Cash: ${self.portfolio.cash:,.2f}")
        logger.info(f"  Open: {len(self.portfolio.open_positions)}/{self.portfolio.max_positions}")
        logger.info(f"  Closed: {len(self.portfolio.closed_positions)}")
        logger.info(f"  Total P&L: ${self.portfolio.total_pnl:+,.2f} ({self.portfolio.total_pnl_pct:+.2f}%)")

        # Log open positions
        for pos in self.portfolio.open_positions:
            price = self._get_current_price(pos.symbol)
            if price:
                pnl = pos.pnl_pct(price)
                hold_min = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 60
                logger.info(f"    {pos.symbol}: {pnl*100:+.2f}% | ${price:.6f} | {hold_min:.0f}min")

    def _save_results(self):
        """Save trading results to JSON"""
        results = {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': datetime.now(timezone.utc).isoformat(),
            'total_scans': self.scan_count,
            'portfolio': self.portfolio.to_dict(),
        }

        filename = f"logs/algo_paper_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(filename).parent.mkdir(exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {filename}")

    def run(self):
        """Main trading loop"""
        self.running = True
        self.start_time = datetime.now(timezone.utc)

        # Ensure collector is running before we start
        self._ensure_collector_running()

        logger.info("\n" + "=" * 70)
        logger.info("STARTING LIVE PAPER TRADING")
        logger.info("=" * 70)

        # Initial stats update
        self._update_normalization_stats()
        logger.info(f"Loaded normalization stats for {len(self.feature_stats)} features")

        try:
            while self.running:
                self.scan_count += 1

                # Scan for entries
                self._scan_for_entries()

                # Monitor open positions
                self._monitor_positions()

                # Log status
                self._log_status()

                # Wait for next scan
                time.sleep(self.scan_interval)

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)

        finally:
            # Final summary
            logger.info("\n" + "=" * 70)
            logger.info("TRADING SESSION COMPLETE")
            logger.info("=" * 70)
            logger.info(f"  Duration: {(datetime.now(timezone.utc) - self.start_time).total_seconds() / 60:.1f} minutes")
            logger.info(f"  Total Scans: {self.scan_count}")
            logger.info(f"  Trades Closed: {len(self.portfolio.closed_positions)}")
            logger.info(f"  Total P&L: ${self.portfolio.total_pnl:+,.2f} ({self.portfolio.total_pnl_pct:+.2f}%)")
            logger.info(f"  Final Cash: ${self.portfolio.cash:,.2f}")

            # Save results
            self._save_results()


# ============================================================================
# Main
# ============================================================================

def main():
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    trader = AlgoPaperTrader(
        db_path="market_data.duckdb",
        starting_capital=10000.0,
        position_size=2500.0,
        max_positions=4,
        window_minutes=6,
        threshold=0.8,
        scan_interval=60,  # Scan every minute
    )

    trader.run()


if __name__ == "__main__":
    main()
