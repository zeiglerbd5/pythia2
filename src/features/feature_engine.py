"""
Real-Time Feature Engineering Engine

Coordinates all feature calculation modules and manages rolling windows.

Per implementation guide:
- 5-minute primary timeframe (optimal for scalping)
- 50-100 bar lookback windows (4-8 hours on 5m)
- Multi-timeframe support (1m, 5m, 15m)
- Real-time calculation with database integration

Calculates 30-40 features after Boruta selection from 100+ candidates.
"""

import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import json

# Paper trades SQLite database for visualizer
try:
    import paper_trades
except ImportError:
    paper_trades = None

import pandas as pd
import numpy as np
from loguru import logger
import joblib

from .microstructure import calculate_microstructure_features
from .volume_indicators import calculate_volume_features
from .price_indicators import calculate_price_features
from .ohlcv_aggregator import OHLCVAggregator, candles_to_dataframe
from .whale_signals import WhaleSignalBuffer, WhaleTransaction

# Import CoinbaseAuth for REST API price fetches
try:
    from ..data_ingestion.coinbase_auth import CoinbaseAuth
except ImportError:
    CoinbaseAuth = None

import sqlite3
import threading


# ============================================================================
# Feature Buffer - Persistent Rolling Window for Zero-Warmup Restarts
# ============================================================================

class FeatureBuffer:
    """
    SQLite-backed rolling feature buffer for zero-warmup restarts.

    Maintains a 90-minute rolling window of calculated features per symbol.
    On collector restart, features are loaded from this buffer to populate
    normalization stats, eliminating the 60-minute warmup period.

    Thread-safe for concurrent writes from feature calculation.
    """

    # Features to persist (entry signal features + key indicators)
    FEATURE_COLUMNS = [
        'natr', 'bid_ask_spread_pct', 'order_book_depth_ratio',
        'large_order_imbalance', 'volume_zscore', 'volume_zscore_5m',
        'returns_5m', 'BB_width', 'RSI_14', 'VWAP_distance'
    ]

    def __init__(self, db_path: str = None, retention_minutes: int = 90):
        """
        Initialize feature buffer.

        Args:
            db_path: Path to SQLite database (default: data/feature_buffer.db)
            retention_minutes: How long to keep features (default: 90 min)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "feature_buffer.db"

        self.db_path = Path(db_path)
        self.retention_minutes = retention_minutes
        self._lock = threading.Lock()

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

        logger.info(f"FeatureBuffer initialized at {self.db_path} (retention: {retention_minutes}min)")

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new connection (SQLite connections are thread-local)."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Initialize database schema."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        natr REAL,
                        bid_ask_spread_pct REAL,
                        order_book_depth_ratio REAL,
                        large_order_imbalance REAL,
                        volume_zscore REAL,
                        volume_zscore_5m REAL,
                        returns_5m REAL,
                        BB_width REAL,
                        RSI_14 REAL,
                        VWAP_distance REAL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_features_timestamp
                    ON features(timestamp)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_features_symbol
                    ON features(symbol)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp
                    ON features(symbol, timestamp)
                """)
                # OHLCV table for V5 event classifier (needs 1500 candles of history)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ohlcv (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        UNIQUE(symbol, timestamp)
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp
                    ON ohlcv(symbol, timestamp DESC)
                """)

                # Order book snapshots table for Accumulation Hunter (48h retention)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS order_book_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        bids TEXT,
                        asks TEXT,
                        mid_price REAL,
                        UNIQUE(symbol, timestamp)
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_obs_symbol_timestamp
                    ON order_book_snapshots(symbol, timestamp DESC)
                """)

                # Trades table for volume calculations (3h retention)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        price REAL NOT NULL,
                        size REAL NOT NULL,
                        side TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp
                    ON trades(symbol, timestamp DESC)
                """)

                conn.commit()
            finally:
                conn.close()

    def write_features(self, symbol: str, timestamp: datetime, features: Dict[str, float]):
        """
        Write features for a symbol to the buffer.

        Args:
            symbol: Trading pair
            timestamp: Feature timestamp
            features: Dictionary of feature values
        """
        with self._lock:
            conn = self._get_conn()
            try:
                # Extract values for configured columns
                values = {col: features.get(col) for col in self.FEATURE_COLUMNS}

                # Insert new row
                cols = ', '.join(['timestamp', 'symbol'] + self.FEATURE_COLUMNS)
                placeholders = ', '.join(['?'] * (2 + len(self.FEATURE_COLUMNS)))

                row_values = [
                    timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                    symbol
                ] + [values.get(col) for col in self.FEATURE_COLUMNS]

                conn.execute(f"INSERT INTO features ({cols}) VALUES ({placeholders})", row_values)
                conn.commit()

            finally:
                conn.close()

    def write_features_batch(self, batch: List[Tuple[str, datetime, Dict[str, float]]]):
        """
        Write multiple feature rows in a single transaction.

        Args:
            batch: List of (symbol, timestamp, features_dict) tuples
        """
        if not batch:
            return

        with self._lock:
            conn = self._get_conn()
            try:
                cols = ', '.join(['timestamp', 'symbol'] + self.FEATURE_COLUMNS)
                placeholders = ', '.join(['?'] * (2 + len(self.FEATURE_COLUMNS)))

                rows = []
                for symbol, timestamp, features in batch:
                    row_values = [
                        timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                        symbol
                    ] + [features.get(col) for col in self.FEATURE_COLUMNS]
                    rows.append(row_values)

                conn.executemany(f"INSERT INTO features ({cols}) VALUES ({placeholders})", rows)
                conn.commit()

            finally:
                conn.close()

    def prune_old(self):
        """Remove features older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.retention_minutes)

        with self._lock:
            conn = self._get_conn()
            try:
                result = conn.execute(
                    "DELETE FROM features WHERE timestamp < ?",
                    (cutoff.isoformat(),)
                )
                deleted = result.rowcount
                conn.commit()

                if deleted > 0:
                    logger.debug(f"[FEATURE_BUFFER] Pruned {deleted} old feature rows")

            finally:
                conn.close()

    def load_recent_features(self, minutes: int = 90) -> Dict[str, List[Dict]]:
        """
        Load recent features for all symbols.

        Args:
            minutes: How many minutes of history to load

        Returns:
            Dict mapping symbol -> list of feature dicts with timestamps
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)

        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT * FROM features
                    WHERE timestamp >= ?
                    ORDER BY symbol, timestamp
                """, (cutoff.isoformat(),)).fetchall()

                # Group by symbol
                result: Dict[str, List[Dict]] = defaultdict(list)
                for row in rows:
                    symbol = row['symbol']
                    features = {col: row[col] for col in self.FEATURE_COLUMNS}
                    features['timestamp'] = row['timestamp']
                    result[symbol].append(features)

                logger.info(f"[FEATURE_BUFFER] Loaded {len(rows)} feature rows for {len(result)} symbols")
                return dict(result)

            finally:
                conn.close()

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        with self._lock:
            conn = self._get_conn()
            try:
                total = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
                symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM features").fetchone()[0]
                oldest = conn.execute("SELECT MIN(timestamp) FROM features").fetchone()[0]
                newest = conn.execute("SELECT MAX(timestamp) FROM features").fetchone()[0]

                return {
                    'total_rows': total,
                    'symbols': symbols,
                    'oldest': oldest,
                    'newest': newest,
                    'retention_minutes': self.retention_minutes
                }
            finally:
                conn.close()

    # ========================================================================
    # OHLCV Storage for V5 Event Classifier
    # ========================================================================

    def write_ohlcv(self, symbol: str, timestamp: datetime, ohlcv: Dict[str, float]):
        """
        Write a single OHLCV candle to the buffer.

        Args:
            symbol: Trading pair
            timestamp: Candle timestamp
            ohlcv: Dict with open, high, low, close, volume
        """
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv (timestamp, symbol, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                    symbol,
                    ohlcv.get('open', 0),
                    ohlcv.get('high', 0),
                    ohlcv.get('low', 0),
                    ohlcv.get('close', 0),
                    ohlcv.get('volume', 0)
                ))
                conn.commit()
            finally:
                conn.close()

    def write_ohlcv_batch(self, batch: List[Tuple[str, datetime, Dict[str, float]]]):
        """
        Write multiple OHLCV candles in a single transaction.

        Args:
            batch: List of (symbol, timestamp, ohlcv_dict) tuples
        """
        if not batch:
            return

        with self._lock:
            conn = self._get_conn()
            try:
                rows = []
                for symbol, timestamp, ohlcv in batch:
                    rows.append((
                        timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                        symbol,
                        ohlcv.get('open', 0),
                        ohlcv.get('high', 0),
                        ohlcv.get('low', 0),
                        ohlcv.get('close', 0),
                        ohlcv.get('volume', 0)
                    ))
                conn.executemany("""
                    INSERT OR REPLACE INTO ohlcv (timestamp, symbol, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, rows)
                conn.commit()
            finally:
                conn.close()

    def load_ohlcv_history(self, symbol: str, limit: int = 1500) -> Optional[pd.DataFrame]:
        """
        Load OHLCV history for a symbol.

        Args:
            symbol: Trading pair
            limit: Maximum candles to return (default 1500 for V5)

        Returns:
            DataFrame with timestamp, open, high, low, close, volume columns
            or None if insufficient data
        """
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (symbol, limit)).fetchall()

                if not rows:
                    return None

                # Convert to DataFrame
                data = [{
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                } for row in rows]

                df = pd.DataFrame(data)
                # Reverse to chronological order
                df = df.iloc[::-1].reset_index(drop=True)
                return df

            finally:
                conn.close()

    def prune_old_ohlcv(self, max_candles_per_symbol: int = 1500):
        """
        Remove old OHLCV data, keeping only the most recent candles per symbol.

        Args:
            max_candles_per_symbol: Max candles to retain per symbol
        """
        with self._lock:
            conn = self._get_conn()
            try:
                # Get all symbols
                symbols = conn.execute("SELECT DISTINCT symbol FROM ohlcv").fetchall()

                total_deleted = 0
                for (symbol,) in symbols:
                    # Delete all but the most recent N candles for this symbol
                    result = conn.execute("""
                        DELETE FROM ohlcv
                        WHERE symbol = ? AND id NOT IN (
                            SELECT id FROM ohlcv
                            WHERE symbol = ?
                            ORDER BY timestamp DESC
                            LIMIT ?
                        )
                    """, (symbol, symbol, max_candles_per_symbol))
                    total_deleted += result.rowcount

                conn.commit()

                if total_deleted > 0:
                    logger.debug(f"[FEATURE_BUFFER] Pruned {total_deleted} old OHLCV rows")

            finally:
                conn.close()

    def get_ohlcv_stats(self) -> Dict:
        """Get OHLCV buffer statistics."""
        with self._lock:
            conn = self._get_conn()
            try:
                total = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
                symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv").fetchone()[0]
                oldest = conn.execute("SELECT MIN(timestamp) FROM ohlcv").fetchone()[0]
                newest = conn.execute("SELECT MAX(timestamp) FROM ohlcv").fetchone()[0]

                # Get per-symbol counts
                symbol_counts = conn.execute("""
                    SELECT symbol, COUNT(*) as cnt FROM ohlcv
                    GROUP BY symbol ORDER BY cnt DESC LIMIT 5
                """).fetchall()

                return {
                    'total_candles': total,
                    'symbols': symbols,
                    'oldest': oldest,
                    'newest': newest,
                    'top_symbols': [(row['symbol'], row['cnt']) for row in symbol_counts]
                }
            finally:
                conn.close()

    # ========================================================================
    # Order Book Snapshots for Accumulation Hunter (48h retention)
    # ========================================================================

    def write_order_book_snapshot(self, symbol: str, timestamp: datetime,
                                   bids: str, asks: str, mid_price: float):
        """
        Write a single order book snapshot.

        Args:
            symbol: Trading pair
            timestamp: Snapshot timestamp
            bids: JSON string of bid levels
            asks: JSON string of ask levels
            mid_price: Mid price
        """
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO order_book_snapshots
                    (timestamp, symbol, bids, asks, mid_price)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                    symbol,
                    bids,
                    asks,
                    mid_price
                ))
                conn.commit()
            finally:
                conn.close()

    def write_order_book_batch(self, batch: List[Tuple[str, datetime, str, str, float]]):
        """
        Write multiple order book snapshots in a single transaction.

        Args:
            batch: List of (symbol, timestamp, bids_json, asks_json, mid_price) tuples
        """
        if not batch:
            return

        with self._lock:
            conn = self._get_conn()
            try:
                rows = []
                for symbol, timestamp, bids, asks, mid_price in batch:
                    rows.append((
                        timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                        symbol,
                        bids,
                        asks,
                        mid_price
                    ))
                conn.executemany("""
                    INSERT OR REPLACE INTO order_book_snapshots
                    (timestamp, symbol, bids, asks, mid_price)
                    VALUES (?, ?, ?, ?, ?)
                """, rows)
                conn.commit()
            finally:
                conn.close()

    def prune_old_order_books(self, retention_hours: int = 48):
        """
        Remove order book snapshots older than retention period.

        Args:
            retention_hours: Hours of data to retain (default 48)
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)

        with self._lock:
            conn = self._get_conn()
            try:
                result = conn.execute(
                    "DELETE FROM order_book_snapshots WHERE timestamp < ?",
                    (cutoff.isoformat(),)
                )
                deleted = result.rowcount
                conn.commit()

                if deleted > 0:
                    logger.debug(f"[FEATURE_BUFFER] Pruned {deleted} old order book rows")

            finally:
                conn.close()

    # ========================================================================
    # Trades for Volume Calculations (3h retention)
    # ========================================================================

    def write_trade(self, symbol: str, timestamp: datetime,
                    price: float, size: float, side: str = None):
        """
        Write a single trade.

        Args:
            symbol: Trading pair
            timestamp: Trade timestamp
            price: Trade price
            size: Trade size
            side: 'buy' or 'sell' (optional)
        """
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO trades (timestamp, symbol, price, size, side)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                    symbol,
                    price,
                    size,
                    side
                ))
                conn.commit()
            finally:
                conn.close()

    def write_trades_batch(self, batch: List[Tuple[str, datetime, float, float, str]]):
        """
        Write multiple trades in a single transaction.

        Args:
            batch: List of (symbol, timestamp, price, size, side) tuples
        """
        if not batch:
            return

        with self._lock:
            conn = self._get_conn()
            try:
                rows = []
                for symbol, timestamp, price, size, side in batch:
                    rows.append((
                        timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                        symbol,
                        price,
                        size,
                        side
                    ))
                conn.executemany("""
                    INSERT INTO trades (timestamp, symbol, price, size, side)
                    VALUES (?, ?, ?, ?, ?)
                """, rows)
                conn.commit()
            finally:
                conn.close()

    def prune_old_trades(self, retention_hours: int = 3):
        """
        Remove trades older than retention period.

        Args:
            retention_hours: Hours of data to retain (default 3)
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)

        with self._lock:
            conn = self._get_conn()
            try:
                result = conn.execute(
                    "DELETE FROM trades WHERE timestamp < ?",
                    (cutoff.isoformat(),)
                )
                deleted = result.rowcount
                conn.commit()

                if deleted > 0:
                    logger.debug(f"[FEATURE_BUFFER] Pruned {deleted} old trade rows")

            finally:
                conn.close()

    def get_order_book_stats(self) -> Dict:
        """Get order book buffer statistics."""
        with self._lock:
            conn = self._get_conn()
            try:
                total = conn.execute("SELECT COUNT(*) FROM order_book_snapshots").fetchone()[0]
                symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM order_book_snapshots").fetchone()[0]
                oldest = conn.execute("SELECT MIN(timestamp) FROM order_book_snapshots").fetchone()[0]
                newest = conn.execute("SELECT MAX(timestamp) FROM order_book_snapshots").fetchone()[0]

                return {
                    'total_snapshots': total,
                    'symbols': symbols,
                    'oldest': oldest,
                    'newest': newest
                }
            finally:
                conn.close()


# ============================================================================
# Paper Trading Data Classes
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

    # Step-down exit strategy fields
    step_floor: float = 0.0  # Current floor price (ratchets down)
    initial_drawdown_used: bool = False  # Whether first 5% step has occurred
    local_high_after_step: float = 0.0  # Track local high after a step for 2% calculation

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
            # Tracking fields for open positions
            'remaining_quantity': self.remaining_quantity,
            'peak_price': self.peak_price,
            'trailing_stop_active': self.trailing_stop_active,
            'trailing_stop_price': self.trailing_stop_price,
            # Step-down exit strategy fields
            'step_floor': self.step_floor,
            'initial_drawdown_used': self.initial_drawdown_used,
            'local_high_after_step': self.local_high_after_step,
        }


@dataclass
class Portfolio:
    """Portfolio state for paper trading"""
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

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get open position for a symbol"""
        for pos in self.open_positions:
            if pos.symbol == symbol:
                return pos
        return None

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


@dataclass
class TradingStrategy:
    """
    Paper trading strategy with configurable entry filters and exit strategies.

    Entry filters:
    - Strategy B (no ret5m): V3 + vol_d + natr_d
    - Strategy C (step_down): V3 + vol_d + ret5m, with step-down exit logic
    - Strategy D (both filters): V3 + vol_d + natr_d + ret5m, with 5-min stop delay

    Exit strategies:
    - "current": 1% stop (skip if V3>85%, max -8%), 15% trailing, partial TPs, timeout extend if V3>80%+profitable
    - "step_down": 5% initial drawdown, 2% step, ratcheting floor, shakeout detection (OB depth>0.8 holds), -12% max loss
    """
    name: str
    portfolio: Portfolio
    require_natr: bool = True
    require_ret5m: bool = True
    max_ret5m: float = 1.0  # Cap ret5m to avoid entering late (1.0 = no cap)
    exit_strategy: str = "current"  # "current" or "step_down"
    stop_delay_minutes: int = 0  # Delay stop loss for first N minutes after entry
    # Step-down variants
    probation_minutes: int = 0  # Tighter tolerance during first N minutes (0 = disabled)
    probation_tolerance: float = 0.03  # Drawdown tolerance during probation (e.g., 3%)
    ob_bail_threshold: float = 0.0  # Bail if OB drops below this while in drawdown (0 = disabled)
    # V4 Reactive strategy parameters
    entry_type: str = "v3"  # "v3" (ML-based), "reactive" (L2 signal-based), or "volume"
    bar_multiple_threshold: float = 3.0  # BAR must be this multiple of baseline
    ask_collapse_threshold: float = 0.5  # Ask depth must collapse below this fraction
    max_price_change_entry: float = 0.02  # Max price change from baseline at entry (predictive filter)
    min_reactive_conditions: int = 3  # How many of the 3 conditions must be met (2 or 3)
    # C5 Volume strategy parameters
    volume_multiple_threshold: float = 3.0  # 24h volume must be this multiple of 30-day avg
    min_price_change_entry: float = 0.05  # Min price change to enter (catch momentum)
    # Simple stop loss and take profit (checked before other exit logic)
    simple_stop_loss: float = 0.0  # Exit if down this % from entry (0 = disabled, e.g., 0.025 = 2.5%)
    simple_take_profit: float = 0.0  # Exit if up this % from entry (0 = disabled, e.g., 0.15 = 15%)
    # Max hold time
    max_hold_minutes: int = 180  # Default 3 hours, C4 uses 1440 (24 hours)
    # V5 Event classifier parameters
    event_classifier_threshold: float = 0.8  # Prediction probability threshold for entry
    event_classifier_min_vol_ratio: float = 1.0  # Min vol_ratio_20_60 to enter (1.0 = disabled, legacy)
    # V5 Two-tier entry system (replaces vol_ratio filter)
    event_classifier_volume_spike_threshold: float = 3.0  # Tier 1: immediate entry if volume spike >= this
    event_classifier_momentum_threshold: float = 0.03  # Tier 1: immediate entry if 15-min momentum >= this (3%)
    event_classifier_watch_mode: bool = True  # Enable watch mode for Tier 2 entries
    event_classifier_watch_trigger_pct: float = 0.02  # Tier 2: enter if price up this % from watch price (2%)
    event_classifier_watch_timeout_min: int = 60  # Remove from watch after this many minutes
    event_classifier_watch_bail_pct: float = 0.02  # Remove from watch if price drops this % (2%)
    event_classifier_model_path: Optional[str] = None  # Custom model path (None = use default)
    cooldowns: Dict[str, datetime] = field(default_factory=dict)

    def get_cooldown(self, symbol: str) -> Optional[datetime]:
        return self.cooldowns.get(symbol)

    def set_cooldown(self, symbol: str, timestamp: datetime):
        self.cooldowns[symbol] = timestamp


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
            time_since = current_time - self.last_trigger[symbol]
            if time_since < timedelta(minutes=self.cooldown_minutes):
                return None

        hits = self.feature_hits[symbol]

        # Find NATR hits
        natr_hits = [h for h in hits if h[1] == 'natr']

        # Find order book hits
        orderbook_hits = [h for h in hits if h[1] in self.ORDERBOOK_FEATURES]

        if natr_hits and orderbook_hits:
            # Get the best values
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

        # Debug: log why signal didn't fire when we have both types of hits
        if natr_hits and orderbook_hits:
            # This should never happen - if we have both, signal should have fired!
            from loguru import logger
            logger.warning(f"[SIG-BUG] {symbol}: BOTH conditions met but signal didn't fire! natr_hits={len(natr_hits)} ob_hits={len(orderbook_hits)}")
        elif natr_hits or orderbook_hits:
            # Log when only one condition is met (for debugging)
            from loguru import logger
            logger.info(f"[SIG-PARTIAL] {symbol}: natr={len(natr_hits)} ob={len(orderbook_hits)} hits={len(hits)}")

        return None


class NormalizationStats:
    """
    Tracks rolling z-score statistics for feature normalization.
    Uses 1-hour lookback window by default.
    """

    def __init__(self, lookback_minutes: int = 60):
        self.lookback_minutes = lookback_minutes
        # {feature: [(timestamp, value), ...]}
        self.history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        # Cached stats
        self._stats_cache: Dict[str, Dict[str, float]] = {}
        self._last_update: Optional[datetime] = None
        self._update_interval = 10  # Update every N feature calculations
        self._calc_count = 0

    def add_value(self, feature: str, timestamp: datetime, value: float):
        """Add a value to the history"""
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            self.history[feature].append((timestamp, value))
            self._cleanup_old(feature, timestamp)

    def _cleanup_old(self, feature: str, current_time: datetime):
        """Remove values older than lookback window"""
        cutoff = current_time - timedelta(minutes=self.lookback_minutes)
        self.history[feature] = [
            (t, v) for t, v in self.history[feature] if t >= cutoff
        ]

    def get_stats(self, feature: str) -> Tuple[float, float]:
        """Get mean and std for a feature"""
        if feature in self._stats_cache:
            return self._stats_cache[feature]['mean'], self._stats_cache[feature]['std']

        values = [v for _, v in self.history[feature]]
        if len(values) < 10:
            return 0.0, 1.0  # Default to prevent division by zero

        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            std = 1.0

        return mean, std

    def normalize(self, feature: str, value: float) -> float:
        """
        Normalize a value using z-score to 0-1 scale.
        Z-score is clipped to [-3, +3] then mapped to [0, 1].
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.5  # Return middle value for missing data

        mean, std = self.get_stats(feature)
        z_score = (value - mean) / std
        z_clipped = np.clip(z_score, -3, 3)
        normalized = (z_clipped + 3) / 6
        return float(normalized)

    def update_cache(self):
        """Update cached statistics"""
        self._calc_count += 1
        if self._calc_count % self._update_interval == 0:
            for feature in self.history:
                values = [v for _, v in self.history[feature]]
                if len(values) >= 10:
                    self._stats_cache[feature] = {
                        'mean': np.mean(values),
                        'std': max(np.std(values), 0.0001)
                    }


class RollingBaseline:
    """
    Tracks rolling baselines for L2 signals used in V4 reactive strategy.

    Calculates 60-minute rolling means for:
    - bid_ask_ratio (bid_depth / ask_depth)
    - bid_depth
    - ask_depth
    - price

    Used to detect relative spikes vs baseline for predictive entry signals.
    """

    def __init__(self, window_minutes: int = 60, min_samples: int = 10):
        self.window_minutes = window_minutes
        self.min_samples = min_samples

        # {symbol: [(timestamp, bid_ask_ratio, bid_depth, ask_depth, price), ...]}
        self._history: Dict[str, List[Tuple[datetime, float, float, float, float]]] = defaultdict(list)

        # Cached baselines for quick lookup
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._last_update: Dict[str, datetime] = {}

    def update(self, symbol: str, timestamp: datetime, bid_depth: float,
               ask_depth: float, price: float):
        """
        Record new L2 data point for baseline calculation.

        Called each time order book data is updated.
        """
        if not bid_depth or not ask_depth or ask_depth == 0:
            return

        bid_ask_ratio = bid_depth / ask_depth

        self._history[symbol].append((timestamp, bid_ask_ratio, bid_depth, ask_depth, price))
        self._cleanup_old(symbol, timestamp)

        # Recalculate baseline periodically (every 30 seconds)
        should_update = (
            symbol not in self._last_update or
            (timestamp - self._last_update[symbol]).total_seconds() >= 30
        )

        if should_update:
            self._recalculate_baseline(symbol)
            self._last_update[symbol] = timestamp

    def _cleanup_old(self, symbol: str, current_time: datetime):
        """Remove records older than window."""
        cutoff = current_time - timedelta(minutes=self.window_minutes)
        self._history[symbol] = [
            r for r in self._history[symbol] if r[0] >= cutoff
        ]

    def _recalculate_baseline(self, symbol: str):
        """Calculate rolling means for the baseline."""
        records = self._history[symbol]

        if len(records) < self.min_samples:
            return

        bar_values = [r[1] for r in records]
        bid_values = [r[2] for r in records]
        ask_values = [r[3] for r in records]
        price_values = [r[4] for r in records if r[4] > 0]

        self._baselines[symbol] = {
            'bid_ask_ratio': np.mean(bar_values),
            'bid_depth': np.mean(bid_values),
            'ask_depth': np.mean(ask_values),
            'price': np.mean(price_values) if price_values else 0.0,
            'samples': len(records)
        }

    def get_baseline(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get baseline values for a symbol."""
        return self._baselines.get(symbol)

    def check_reactive_signal(self, symbol: str, current_bid_depth: float,
                              current_ask_depth: float, current_price: float,
                              bar_multiple_threshold: float = 3.0,
                              ask_collapse_threshold: float = 0.5,
                              max_price_change: float = 0.02,
                              min_conditions: int = 3) -> Optional[Dict]:
        """
        Check if current L2 values indicate a predictive spike signal.

        Conditions:
        1. BAR (bid_ask_ratio) >= bar_multiple_threshold * baseline
        2. Ask depth <= ask_collapse_threshold * baseline
        3. Price has NOT already moved > max_price_change (predictive filter)

        Args:
            min_conditions: How many of the 3 conditions must be met (2 or 3)

        Returns signal dict if triggered, None otherwise.
        """
        baseline = self.get_baseline(symbol)

        if not baseline or baseline.get('samples', 0) < self.min_samples:
            return None

        if not current_bid_depth or not current_ask_depth or current_ask_depth == 0:
            return None

        current_bar = current_bid_depth / current_ask_depth
        baseline_bar = baseline['bid_ask_ratio']
        baseline_ask = baseline['ask_depth']
        baseline_price = baseline['price']

        if baseline_bar == 0 or baseline_ask == 0 or baseline_price == 0:
            return None

        # Calculate multiples
        bar_multiple = current_bar / baseline_bar
        ask_collapse = current_ask_depth / baseline_ask
        price_change = (current_price - baseline_price) / baseline_price if baseline_price > 0 else 0

        # Check each condition
        bar_condition = bar_multiple >= bar_multiple_threshold
        ask_condition = ask_collapse <= ask_collapse_threshold
        price_condition = abs(price_change) <= max_price_change

        # BAR is mandatory - Ask signal alone is not predictive (75 false signals vs 1 fluke win)
        if not bar_condition:
            return None

        # Only count BAR + Price (Ask kept for logging but doesn't trigger entries)
        conditions_met = sum([bar_condition, price_condition])

        # Need at least min_conditions to trigger
        if conditions_met >= min_conditions:
            return {
                'symbol': symbol,
                'bar_multiple': bar_multiple,
                'ask_collapse': ask_collapse,
                'price_change': price_change,
                'current_bar': current_bar,
                'baseline_bar': baseline_bar,
                'current_ask_depth': current_ask_depth,
                'baseline_ask_depth': baseline_ask,
                'current_price': current_price,
                'baseline_price': baseline_price,
                'conditions_met': conditions_met,
                'bar_condition': bar_condition,
                'ask_condition': ask_condition,
                'price_condition': price_condition,
            }

        return None


class WatchMode:
    """
    Tracks symbols that need increased L2 polling for C4 reactive strategy.

    When moderate signals appear (V3 > 70%, BAR > 2x baseline, or ask collapse),
    the symbol enters watch mode for faster L2 polling (5s vs 30s).
    This gives C4 more opportunities to catch brief signal alignments.

    Watch mode expires after 15 minutes or when signal fades.
    """

    def __init__(self, watch_duration: int = 900, max_watched: int = 3):
        """
        Initialize watch mode tracker.

        Args:
            watch_duration: How long to keep a symbol in watch mode (seconds)
            max_watched: Maximum symbols to watch at once (to respect rate limits)
        """
        self.watch_duration = watch_duration
        self.max_watched = max_watched
        self.fast_poll_interval = 5  # seconds

        # {symbol: {'entered': timestamp, 'reason': str, 'last_signal': timestamp}}
        self._watched: Dict[str, Dict] = {}

    def should_watch(self, symbol: str, v3_confidence: float = 0.0,
                     bar_multiple: float = 0.0, ask_ratio: float = 1.0) -> Tuple[bool, Optional[str]]:
        """
        Check if a symbol should enter watch mode based on early signals.

        Args:
            symbol: Trading pair
            v3_confidence: V3 model prediction (0-1)
            bar_multiple: Current BAR / baseline BAR
            ask_ratio: Current ask depth / baseline ask depth

        Returns:
            (should_watch, reason) tuple
        """
        if v3_confidence >= 0.70:
            return True, f"V3={v3_confidence:.1%}"
        if bar_multiple >= 2.0:
            return True, f"BAR={bar_multiple:.1f}x"
        if ask_ratio <= 0.70:
            return True, f"Ask={ask_ratio:.0%}"
        return False, None

    def add_to_watch(self, symbol: str, reason: str) -> bool:
        """
        Add a symbol to watch mode.

        Args:
            symbol: Trading pair
            reason: Why it's being watched (for logging)

        Returns:
            True if added, False if already watching max symbols
        """
        import time
        now = time.time()

        # Already watching this symbol? Just update the signal time
        if symbol in self._watched:
            self._watched[symbol]['last_signal'] = now
            self._watched[symbol]['reason'] = reason
            return True

        # At max capacity? Don't add more
        if len(self._watched) >= self.max_watched:
            return False

        self._watched[symbol] = {
            'entered': now,
            'reason': reason,
            'last_signal': now
        }
        logger.info(f"WATCH MODE: {symbol} added ({reason})")
        return True

    def remove_from_watch(self, symbol: str):
        """Remove a symbol from watch mode."""
        if symbol in self._watched:
            del self._watched[symbol]
            logger.info(f"WATCH MODE: {symbol} removed")

    def is_watched(self, symbol: str) -> bool:
        """Check if a symbol is currently in watch mode."""
        import time
        if symbol not in self._watched:
            return False

        # Auto-expire after duration
        now = time.time()
        if now - self._watched[symbol]['entered'] > self.watch_duration:
            self.remove_from_watch(symbol)
            return False

        return True

    def get_watched_symbols(self) -> List[str]:
        """Get list of currently watched symbols (for fast polling)."""
        import time
        now = time.time()

        # Clean up expired entries
        expired = [s for s, data in self._watched.items()
                   if now - data['entered'] > self.watch_duration]
        for s in expired:
            self.remove_from_watch(s)

        return list(self._watched.keys())

    def get_watch_info(self, symbol: str) -> Optional[Dict]:
        """Get watch mode info for a symbol."""
        if self.is_watched(symbol):
            return self._watched.get(symbol)
        return None


class FeatureEngine:
    """
    Real-time feature calculation engine.

    Manages:
    - OHLCV aggregation from trades
    - Rolling window buffers (50-100 bars per guide)
    - Multi-timeframe feature calculation
    - Database writing

    Per guide: Focus on 5m primary with 15m confirmation and 1m entry timing.
    """

    def __init__(
        self,
        db_manager,
        symbols: List[str],
        primary_timeframe: str = '5m',
        timeframes: Optional[List[str]] = None,
        lookback_bars: int = 100,
        xgboost_model_path: Optional[str] = None,
        prediction_threshold: float = 0.7
    ):
        """
        Initialize feature engine.

        Args:
            db_manager: DuckDBManager instance
            symbols: List of trading pairs to track
            primary_timeframe: Primary analysis timeframe (default: 5m per guide)
            timeframes: Additional timeframes (default: 1m, 5m, 15m)
            lookback_bars: Rolling window size (default: 100 per guide)
            xgboost_model_path: Path to XGBoost model for live predictions (optional)
            prediction_threshold: Probability threshold for spike alerts (default: 0.7)
        """
        self.db_manager = db_manager
        self.symbols = symbols
        self.primary_timeframe = primary_timeframe
        self.timeframes = timeframes or ['1m', '5m', '15m']
        self.lookback_bars = lookback_bars
        self.prediction_threshold = prediction_threshold

        # Thread pool for CPU-bound feature calculations
        # This allows parallel processing of symbols despite Python's GIL
        # Reduced from 8 to 4 workers to lower CPU contention
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._loop = None  # Will be set when first async method runs

        # OHLCV aggregator
        self.aggregator = OHLCVAggregator(timeframes=self.timeframes)

        # Rolling window buffers per symbol per timeframe
        # {symbol: {timeframe: DataFrame}}
        self.ohlcv_buffers: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)

        # Order book manager reference (set via set_order_book_manager after init)
        self.order_book_manager = None

        # Cache for latest order book data per symbol
        self._order_book_cache: Dict[str, dict] = {}

        # Volume history for large_order_imbalance calculation (90th percentile threshold)
        # {symbol: {'buy': [volumes], 'sell': [volumes]}}
        self._volume_history: Dict[str, Dict[str, list]] = defaultdict(lambda: {'buy': [], 'sell': []})
        self._volume_history_size = 50  # Rolling window for percentile calculation

        # News signal cache for news-based entry strategy
        # {symbol: NewsSignal} - most recent news signal per symbol
        self._news_signals: Dict[str, dict] = {}
        self._news_signal_ttl_minutes = 60  # News signals valid for 1 hour

        # Whale signal buffer for whale-derived features
        # Tracks whale transactions for net flow, exchange pressure, activity z-score
        self.whale_buffer = WhaleSignalBuffer(ttl_minutes=120)

        # Feature calculation flags
        self.calculation_enabled = {
            'microstructure': True,
            'volume': True,
            'price': True,
        }

        # XGBoost models for live predictions (V1 for momentum, V3 for slow/large spikes)
        self.xgboost_model_v1 = None
        self.xgboost_model_v3 = None
        self.feature_columns = [
            'returns', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14', 'NATR',
            'BB_width', 'BB_squeeze', 'VWAP_distance', 'volume_zscore', 'volume_roc',
            'OBV', 'trade_count', 'buy_sell_ratio', 'roll_measure', 'order_flow_imbalance',
            'vpin', 'bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance',
            'returns_5m', 'volume_zscore_5m', 'returns_15m', 'volume_zscore_15m'
        ]

        # Home run filter thresholds (from Dec 7-8 backtest analysis)
        # Best filter: V3 >= 93% AND V1 >= 92% AND BB_width > 0.07
        # Results: 40% home run rate, 14.2% avg return
        self.v3_threshold = 0.93
        self.v1_threshold = 0.92
        self.bb_width_threshold = 0.07

        # Fast & Big spike threshold (separate from Slow & Large)
        self.fast_va_threshold = 0.95  # High confidence only

        # V1 model disabled - only using V3 for paper trading entry
        # This speeds up processing to handle all 300+ symbols
        # v1_model_path = Path('models/xgboost_slow_large_v1.pkl')
        logger.info("V1 model disabled (only V3 needed for paper trading entry)")

        # Load V3 model (slow & large spike detection)
        v3_model_path = Path('models/xgboost_slow_large_v3_model.pkl')
        if v3_model_path.exists():
            try:
                self.xgboost_model_v3 = joblib.load(v3_model_path)
                logger.info(f"Loaded V3 model from {v3_model_path}")
            except Exception as e:
                logger.error(f"Error loading V3 model: {e}")
        else:
            logger.warning(f"V3 model not found: {v3_model_path}")

        # Fast models disabled - only using V3 for paper trading entry
        # This speeds up processing to handle all 300+ symbols
        self.xgboost_model_fast_va = None
        self.xgboost_model_fast_vb = None
        self.xgboost_model_fast_vc = None
        self.fast_vb_features = None
        self.fast_vc_features = None
        self.fast_vc_threshold = 0.85
        logger.info("Fast models (vA, vB, vC) disabled (only V3 needed for paper trading entry)")

        # Log filter configuration - V3 only for paper trading
        if self.xgboost_model_v3:
            logger.info(f"V3 + ACCELERATION ENTRY ACTIVE:")
            logger.info(f"  V3 >= 70% (model confidence)")
            logger.info(f"  volume_zscore_delta > 0.5 (volume accelerating)")
            logger.info(f"  returns_5m > 0 (price moving up)")
            logger.info(f"  NATR_delta > 0.05 (volatility accelerating)")

        # Legacy: keep xgboost_model for backward compatibility
        self.xgboost_model = self.xgboost_model_v3

        # Statistics
        self.stats = {
            'trades_processed': 0,
            'candles_completed': 0,
            'features_calculated': 0,
            'predictions_made': 0,
            'spikes_predicted': 0,
        }

        # Alert log file
        self.alert_log_file = "spike_alerts.json"

        # Paper trading components
        self.paper_trading_enabled = True

        # XGBoost Model Comparison Test (Jan 31, 2026)
        # Comparing 3 models trained with different precision/recall tradeoffs:
        # - X18: Best F1 (0.757), balanced (scale_pos_weight=0.45)
        # - P25: High precision (95% @0.8), balanced recall (51%) - slow learner
        # - P3: Maximum precision (100% @0.9), lower recall (33%) - heavy FP penalty
        # OLD STRATEGIES DISABLED - X31/X32/X33/X34 retired
        # Using new volume-reactive system via run_spike_paper_trader.py instead
        self.strategies: List[TradingStrategy] = []

        # Backward compatibility: reference first strategy's portfolio (or create empty one)
        self.portfolio = self.strategies[0].portfolio if self.strategies else Portfolio(starting_capital=100000.0, cash=100000.0, max_positions=50)

        self.feature_tracker = FeatureWindowTracker(window_minutes=6, threshold=0.8)
        self.norm_stats: Dict[str, NormalizationStats] = {}  # Per-symbol normalization
        self._price_cache: Dict[str, float] = {}  # Latest prices from ticker/trades

        # V4 Reactive strategy: Rolling baseline for L2 signals
        self.rolling_baseline = RollingBaseline(window_minutes=60, min_samples=10)

        # V5 Watch mode: {symbol: {'watch_price': float, 'watch_time': datetime, 'prob': float, 'strategy': str}}
        self._v5_watchlist: Dict[str, Dict] = {}

        # Watch mode: faster L2 polling for symbols with early signals
        self.watch_mode = WatchMode(watch_duration=900, max_watched=3)

        # Ignition Watchlist: Symbols with high model probability but bearish depth
        # Wait for depth to flip bullish (> 1.0) before entering
        # {symbol: {'timestamp': datetime, 'prob': float, 'strategy': str, 'initial_depth': float, 'entry_price': float}}
        self._ignition_watchlist: Dict[str, dict] = {}
        self._watchlist_max_age_hours = 12  # Remove from watchlist after 12 hours

        # C5 Volume: Cache of volume explosion signals from VolumeScanner
        # Updated by collector callback, checked by C5 strategy
        self._volume_signals: Dict[str, dict] = {}  # {symbol: {volume_multiple, price_change, timestamp}}

        # V5 Event Classifier: XGBoost models for spike prediction
        # Supports multiple models - one per strategy if specified
        self._event_classifiers: Dict[str, dict] = {}  # {model_path: {'model': ..., 'scaler': ..., 'features': ...}}
        self._default_classifier_path = '/Users/bz/Pythia2/models/event_classifier_xgb.pkl'

        # Collect all model paths needed (default + strategy-specific)
        model_paths_to_load = {self._default_classifier_path}
        for strategy in self.strategies:
            if strategy.event_classifier_model_path:
                model_paths_to_load.add(strategy.event_classifier_model_path)

        # Load all needed models
        try:
            for model_path in model_paths_to_load:
                if Path(model_path).exists():
                    model_data = joblib.load(model_path)
                    self._event_classifiers[model_path] = {
                        'model': model_data['model'],
                        'scaler': model_data['scaler'],
                        'features': model_data['feature_cols']
                    }
                    model_name = Path(model_path).stem
                    logger.info(f"Loaded event classifier: {model_name} ({len(model_data['feature_cols'])} features)")
                else:
                    logger.warning(f"Event classifier model not found at {model_path}")
        except Exception as e:
            logger.warning(f"Could not load event classifiers: {e}")

        # Feature buffer: persistent rolling window for zero-warmup restarts
        # 360 min (6h) retention supports the loading detector's 6h lookback
        self.feature_buffer = FeatureBuffer(retention_minutes=360)
        self._feature_buffer_write_counter = 0
        self._feature_buffer_prune_interval = 100  # Prune every 100 writes
        self._feature_buffer_vacuum_counter = 0
        self._feature_buffer_vacuum_interval = 50  # VACUUM every 50 prunes (~5000 writes)

        # REST API for fetching fresh prices (avoids stale cache issues)
        try:
            self._coinbase_auth = CoinbaseAuth.from_env() if CoinbaseAuth else None
        except Exception as e:
            logger.warning(f"Could not initialize CoinbaseAuth for REST prices: {e}")
            self._coinbase_auth = None
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Lock to prevent concurrent position monitoring (fixes duplicate exits)
        self._position_lock = asyncio.Lock()

        self._last_status_log: Optional[datetime] = None
        self._status_log_interval = 60  # Log status every 60 seconds
        self._predictions_at_last_log = 0  # Track predictions per minute
        self._warmup_minutes = 60  # Don't trade until normalization window is full
        self._first_feature_time: Optional[datetime] = None  # Track when we started

        # Features to track for entry signals
        self.entry_features = ['natr', 'bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance']

        # Feature delta tracking for acceleration-based entry
        self._prev_features: Dict[str, Dict[str, float]] = {}  # {symbol: {feature: value}}

        # Cache for latest features per symbol (for paper trading visualization)
        self.latest_features: Dict[str, Dict] = {}

        # Cache for latest V3 predictions (for exit logic)
        self._v3_cache: Dict[str, float] = {}

        logger.info(
            f"FeatureEngine initialized",
            extra={
                "symbols": len(symbols),
                "primary_timeframe": primary_timeframe,
                "timeframes": self.timeframes,
                "lookback_bars": lookback_bars,
                "xgboost_enabled": self.xgboost_model is not None,
                "paper_trading": self.paper_trading_enabled
            }
        )

        if self.paper_trading_enabled:
            # Try to load existing state from JSON
            self._load_paper_trading_state()

            logger.info(f"MULTI-STRATEGY PAPER TRADING ENABLED:")
            for strat in self.strategies:
                open_pos = len(strat.portfolio.open_positions)
                closed_pos = len(strat.portfolio.closed_positions)
                if strat.entry_type == "reactive":
                    # L2 reactive strategy - show conditions
                    cond_str = f"{strat.min_reactive_conditions}of3"
                    entry_desc = f"L2 {cond_str} (BAR≥{strat.bar_multiple_threshold:.0f}x, Ask≤{strat.ask_collapse_threshold:.0%}, Δpx≤{strat.max_price_change_entry:.0%})"
                else:
                    # V3 ML-based strategy
                    natr_filter = "NATR" if strat.require_natr else "-"
                    ret5m_filter = "ret5m" if strat.require_ret5m else "-"
                    entry_desc = f"V3 + vol_d + {natr_filter} + {ret5m_filter}"
                logger.info(f"  [{strat.name}] ${strat.portfolio.cash:,.0f} | {entry_desc} | Open: {open_pos} Closed: {closed_pos}")

    def set_order_book_manager(self, order_book_manager):
        """
        Set the order book manager reference for live order book features.

        Args:
            order_book_manager: OrderBookManager instance from websocket_manager
        """
        self.order_book_manager = order_book_manager
        logger.info("Order book manager connected to feature engine")

    async def load_historical_data(self, lookback_minutes: int = 500):
        """
        Load historical candles from database to pre-fill OHLCV buffers.

        This dramatically reduces warmup time from hours to seconds by
        loading existing candle data instead of waiting for new trades.

        Also backfills the FeatureBuffer SQLite with 1500 1-min candles
        for the V5 event classifier so it's ready immediately after restart.

        Args:
            lookback_minutes: How many minutes of history to load (default 500 = ~8 hours)
        """
        if not self.db_manager:
            logger.warning("No database manager - cannot load historical data")
            return

        # Use naive datetime (no timezone) to match DB storage format
        start_time = datetime.now() - timedelta(minutes=lookback_minutes)
        symbols_loaded = 0
        total_candles = 0

        # V5 needs 1500 1-min candles - load ALL available from DuckDB (no time filter)
        v5_min_candles = 1500
        has_feature_buffer = hasattr(self, 'feature_buffer') and self.feature_buffer
        v5_symbols_filled = 0
        v5_debug_count = 0  # Limit debug logging

        logger.info(f"Loading historical data for {len(self.symbols)} symbols (last {lookback_minutes} minutes)...")

        for symbol in self.symbols:
            try:
                for timeframe in self.timeframes:
                    # Query database for historical candles
                    df = self.db_manager.get_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=start_time
                    )

                    if df.empty or len(df) < 10:
                        continue

                    # Set timestamp as index (matching live candle format)
                    if 'timestamp' in df.columns:
                        df.set_index('timestamp', inplace=True)

                    # Normalize to tz-naive (remove timezone for consistent comparison)
                    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)

                    # Trim to lookback window
                    if len(df) > self.lookback_bars:
                        df = df.iloc[-self.lookback_bars:]

                    # Store in buffer
                    self.ohlcv_buffers[symbol][timeframe] = df
                    total_candles += len(df)

                # Count symbol as loaded if it has primary timeframe data
                if self.primary_timeframe in self.ohlcv_buffers.get(symbol, {}):
                    if len(self.ohlcv_buffers[symbol][self.primary_timeframe]) >= 50:
                        symbols_loaded += 1

                # V5 backfill: load 1-min candles from DuckDB into FeatureBuffer SQLite
                if has_feature_buffer:
                    existing = self.feature_buffer.load_ohlcv_history(symbol, limit=v5_min_candles)
                    existing_count = 0 if existing is None else len(existing)
                    if existing_count < v5_min_candles:
                        # Try multiple timeframe names (schema may vary)
                        df_v5 = pd.DataFrame()
                        for tf_name in ['1m', '60', '1min', 'ONE_MINUTE']:
                            df_v5 = self.db_manager.get_ohlcv(
                                symbol=symbol,
                                timeframe=tf_name,
                            )
                            if not df_v5.empty:
                                break
                        if df_v5.empty and v5_debug_count < 5:
                            logger.info(f"[V5 BACKFILL] {symbol}: no 1m candles in DuckDB (existing SQLite: {existing_count})")
                            v5_debug_count += 1
                        if not df_v5.empty and len(df_v5) > existing_count:
                            # Set timestamp as column (not index) for batch write
                            if 'timestamp' not in df_v5.columns and isinstance(df_v5.index, pd.DatetimeIndex):
                                df_v5 = df_v5.reset_index()

                            batch = []
                            for _, row in df_v5.iterrows():
                                ts = row.get('timestamp')
                                if ts is None:
                                    continue
                                if isinstance(ts, pd.Timestamp):
                                    ts = ts.to_pydatetime()
                                if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                                    ts = ts.replace(tzinfo=None)
                                batch.append((symbol, ts, {
                                    'open': float(row.get('open', 0)),
                                    'high': float(row.get('high', 0)),
                                    'low': float(row.get('low', 0)),
                                    'close': float(row.get('close', 0)),
                                    'volume': float(row.get('volume', 0)),
                                }))
                            if batch:
                                self.feature_buffer.write_ohlcv_batch(batch)
                                v5_symbols_filled += 1

            except Exception as e:
                logger.debug(f"Error loading historical data for {symbol}: {e}")

        logger.info(f"[HISTORICAL] Loaded {total_candles} candles for {symbols_loaded} symbols (ready for predictions)")
        if has_feature_buffer:
            logger.info(f"[V5 BACKFILL] Populated FeatureBuffer for {v5_symbols_filled} symbols from DuckDB (1500 candles)")

    async def load_normalization_stats(self, lookback_minutes: int = 60):
        """
        Load historical features to pre-populate normalization stats.

        This eliminates the 60-minute warmup period for paper trading by
        loading existing feature values from the database.

        Args:
            lookback_minutes: How many minutes of history to load (default 60)
        """
        if not self.db_manager:
            logger.warning("No database manager - cannot load normalization stats")
            return

        # Use naive datetime (no timezone) to match DB storage format
        start_time = datetime.now() - timedelta(minutes=lookback_minutes)
        symbols_loaded = 0
        total_values = 0

        logger.info(f"Loading normalization stats for {len(self.symbols)} symbols (last {lookback_minutes} minutes)...")

        for symbol in self.symbols:
            try:
                # Query features for this symbol
                df = self.db_manager.get_features(
                    symbol=symbol,
                    timeframe=self.primary_timeframe,
                    start_time=start_time
                )

                if df.empty or len(df) < 10:
                    continue

                # Create normalization stats for this symbol if needed
                if symbol not in self.norm_stats:
                    self.norm_stats[symbol] = NormalizationStats(lookback_minutes=60)

                symbol_stats = self.norm_stats[symbol]

                # Load each entry feature's history
                for feature in self.entry_features:
                    if feature in df.columns:
                        for _, row in df.iterrows():
                            timestamp = row.get('timestamp')
                            value = row.get(feature)
                            if timestamp is not None and value is not None and not pd.isna(value):
                                # Ensure timestamp is datetime
                                if isinstance(timestamp, str):
                                    timestamp = pd.to_datetime(timestamp)
                                if timestamp.tzinfo is None:
                                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                                symbol_stats.add_value(feature, timestamp, float(value))
                                total_values += 1

                # Update cached stats
                symbol_stats.update_cache()
                symbols_loaded += 1

            except Exception as e:
                logger.debug(f"Error loading normalization stats for {symbol}: {e}")

        # Mark warmup as complete if we loaded enough data
        if symbols_loaded > 0:
            self._first_feature_time = datetime.now(timezone.utc) - timedelta(minutes=self._warmup_minutes + 1)
            logger.info(f"[NORMALIZATION] Loaded {total_values} values for {symbols_loaded} symbols (warmup skipped)")
        else:
            logger.warning("[NORMALIZATION] No historical features found - warmup period will apply")

    def load_features_from_buffer(self) -> int:
        """
        Load features from persistent buffer to populate normalization stats.

        Called on startup to enable zero-warmup paper trading.
        Reads from the SQLite feature buffer and populates NormalizationStats
        for each symbol.

        Returns:
            Number of symbols successfully loaded
        """
        try:
            # Load recent features from buffer
            recent_features = self.feature_buffer.load_recent_features(minutes=90)

            if not recent_features:
                logger.warning("[FEATURE_BUFFER] No recent features in buffer - warmup will apply")
                return 0

            symbols_loaded = 0
            total_values = 0

            for symbol, feature_list in recent_features.items():
                if not feature_list:
                    continue

                # Create normalization stats for this symbol if needed
                if symbol not in self.norm_stats:
                    self.norm_stats[symbol] = NormalizationStats(lookback_minutes=60)

                symbol_stats = self.norm_stats[symbol]

                # Load each feature's history
                for feature_row in feature_list:
                    timestamp_str = feature_row.get('timestamp')
                    if not timestamp_str:
                        continue

                    # Parse timestamp
                    try:
                        timestamp = pd.to_datetime(timestamp_str)
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                    except Exception:
                        continue

                    # Add each entry feature to normalization stats
                    for feature in self.entry_features:
                        value = feature_row.get(feature)
                        if value is not None and not (isinstance(value, float) and np.isnan(value)):
                            symbol_stats.add_value(feature, timestamp, float(value))
                            total_values += 1

                # Update cached stats
                symbol_stats.update_cache()
                symbols_loaded += 1

            # Mark warmup as complete if we loaded enough data
            if symbols_loaded > 0:
                self._first_feature_time = datetime.now(timezone.utc) - timedelta(minutes=self._warmup_minutes + 1)
                logger.info(f"[FEATURE_BUFFER] Loaded {total_values} values for {symbols_loaded} symbols (WARMUP SKIPPED)")
                return symbols_loaded
            else:
                logger.warning("[FEATURE_BUFFER] No usable features in buffer - warmup will apply")
                return 0

        except Exception as e:
            logger.error(f"[FEATURE_BUFFER] Error loading from buffer: {e}")
            return 0

    async def backfill_candles_from_api(self, symbols: Optional[List[str]] = None,
                                        count: int = 60) -> int:
        """
        Fetch historical candles from Coinbase API for zero-warmup startup.

        This fetches the last N 1-minute candles directly from Coinbase,
        bypassing the need to wait for trades to accumulate. Useful for:
        - Cold starts when database has no data
        - New symbols discovered dynamically
        - Low-volume coins that don't generate enough trades

        Args:
            symbols: List of symbols to backfill (default: all symbols)
            count: Number of candles to fetch per symbol (default: 60)

        Returns:
            Number of symbols successfully backfilled
        """
        symbols_to_backfill = symbols or self.symbols
        backfilled = 0

        if not self._http_session:
            self._http_session = aiohttp.ClientSession()

        logger.info(f"[BACKFILL] Fetching {count} candles from API for {len(symbols_to_backfill)} symbols...")

        for symbol in symbols_to_backfill:
            try:
                url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
                params = {'granularity': 60}  # 1-minute candles

                async with self._http_session.get(url, params=params, timeout=5) as resp:
                    if resp.status != 200:
                        continue

                    data = await resp.json()

                    if not data or len(data) < 10:
                        continue

                    # Coinbase returns: [timestamp, open, high, low, close, volume]
                    # Most recent first - reverse to chronological order
                    candles = list(reversed(data[:count]))

                    # Convert to DataFrame with proper format
                    rows = []
                    for candle in candles:
                        timestamp, open_p, high, low, close, volume = candle
                        rows.append({
                            'open': float(open_p),
                            'high': float(high),
                            'low': float(low),
                            'close': float(close),
                            'volume': float(volume),
                        })

                    if rows:
                        df = pd.DataFrame(rows)
                        # Create timestamp index from candle timestamps
                        timestamps = [datetime.fromtimestamp(c[0], tz=timezone.utc).replace(tzinfo=None)
                                      for c in candles]
                        df.index = pd.DatetimeIndex(timestamps)

                        # Store in 1m buffer
                        self.ohlcv_buffers[symbol]['1m'] = df
                        backfilled += 1

                # Small delay to respect rate limits
                await asyncio.sleep(0.05)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.debug(f"Error backfilling {symbol}: {e}")

        logger.info(f"[BACKFILL] Fetched candles for {backfilled}/{len(symbols_to_backfill)} symbols from API")
        return backfilled

    async def backfill_single_symbol(self, symbol: str, count: int = 60) -> bool:
        """
        Backfill candles for a single symbol (for dynamic symbol discovery).

        Args:
            symbol: Trading pair to backfill
            count: Number of candles to fetch

        Returns:
            True if successful
        """
        result = await self.backfill_candles_from_api(symbols=[symbol], count=count)
        return result > 0

    def update_order_book_cache(self, symbol: str, snapshot: dict):
        """
        Update cached order book data for a symbol.

        Called by websocket manager when new L2 data arrives.

        Args:
            symbol: Trading pair
            snapshot: Order book snapshot with keys: best_bid, best_ask, bid_depth, ask_depth
        """
        self._order_book_cache[symbol] = snapshot

        # Also update price cache from order book
        if 'best_bid' in snapshot and 'best_ask' in snapshot:
            best_bid = snapshot.get('best_bid', 0)
            best_ask = snapshot.get('best_ask', 0)
            if best_bid and best_ask and best_bid > 0 and best_ask > 0:
                self._price_cache[symbol] = (best_bid + best_ask) / 2

        # Update rolling baseline for V4 reactive strategy
        bid_depth = snapshot.get('bid_depth', 0)
        ask_depth = snapshot.get('ask_depth', 0)
        price = self._price_cache.get(symbol, 0)
        if bid_depth and ask_depth and price:
            self.rolling_baseline.update(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                price=price
            )

            # Watch mode: trigger on early L2 signals for faster polling
            baseline = self.rolling_baseline.get_baseline(symbol)
            if baseline and baseline.get('samples', 0) >= 10:
                baseline_bar = baseline.get('bid_ask_ratio', 1.0)
                baseline_ask = baseline.get('ask_depth', ask_depth)

                if baseline_bar > 0 and baseline_ask > 0:
                    current_bar = bid_depth / ask_depth
                    bar_multiple = current_bar / baseline_bar
                    ask_ratio = ask_depth / baseline_ask

                    should_watch, reason = self.watch_mode.should_watch(
                        symbol, bar_multiple=bar_multiple, ask_ratio=ask_ratio
                    )
                    if should_watch:
                        self.watch_mode.add_to_watch(symbol, reason)

    def update_price(self, symbol: str, price: float):
        """
        Update cached price for a symbol.

        Called from trade processing.

        Args:
            symbol: Trading pair
            price: Latest trade price
        """
        self._price_cache[symbol] = price

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol from cache"""
        return self._price_cache.get(symbol)

    def update_volume_signal(self, symbol: str, volume_multiple: float, price_change: float, timestamp: datetime):
        """
        Update volume explosion signal for a symbol.

        Called by the collector when VolumeScanner detects a signal.
        C5_volume strategy checks this cache for entry signals.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            volume_multiple: Current 24h volume / 30-day avg daily volume
            price_change: 24h price change as decimal (e.g., 0.15 for +15%)
            timestamp: When the signal was detected
        """
        self._volume_signals[symbol] = {
            'volume_multiple': volume_multiple,
            'price_change': price_change,
            'timestamp': timestamp,
        }

    def get_volume_signal(self, symbol: str) -> Optional[dict]:
        """Get cached volume signal for a symbol if still fresh (< 5 min old)"""
        signal = self._volume_signals.get(symbol)
        if not signal:
            return None

        # Signal expires after 5 minutes
        age = (datetime.now(timezone.utc) - signal['timestamp']).total_seconds()
        if age > 300:
            return None

        return signal

    def add_whale_transaction(self, tx: WhaleTransaction):
        """
        Add a whale transaction to the buffer.

        Called by the collector when a whale alert is received.

        Args:
            tx: WhaleTransaction from whale alert
        """
        self.whale_buffer.add_transaction(tx)

    def _calculate_whale_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate whale-derived features for a symbol.

        Features:
        - whale_net_flow_1h: Net USD flow (outflows - inflows), z-score normalized
        - whale_exchange_pressure_1h: Ratio of inflows to total (>0.5 = selling pressure)
        - whale_activity_zscore: Transaction count vs baseline
        - whale_largest_move_recency: Minutes since last $10M+ move, normalized
        - whale_btc_eth_pressure: Aggregate BTC+ETH exchange pressure

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            Dict of whale feature values
        """
        # Net flow: normalize by dividing by $100M and clip to [-3, 3]
        net_flow_raw = self.whale_buffer.get_net_flow_usd(symbol, 60)
        whale_net_flow = np.clip(net_flow_raw / 1e8, -3, 3) / 3  # Normalize to [-1, 1]

        # Exchange pressure: already in [0, 1] range
        whale_exchange_pressure = self.whale_buffer.get_exchange_pressure(symbol, 60)

        # Activity z-score: clip to [-3, 3] and normalize to [-1, 1]
        activity_zscore_raw = self.whale_buffer.get_activity_zscore(symbol, 60)
        whale_activity_zscore = np.clip(activity_zscore_raw, -3, 3) / 3

        # Largest move recency: cap at 60 min and normalize to [0, 1]
        recency_raw = self.whale_buffer.get_largest_move_recency(symbol, 10_000_000)
        whale_largest_move_recency = min(60, recency_raw) / 60

        # Market leader pressure: already in [0, 1] range
        whale_btc_eth_pressure = self.whale_buffer.get_market_leader_pressure(60)

        return {
            'whale_net_flow_1h': whale_net_flow,
            'whale_exchange_pressure_1h': whale_exchange_pressure,
            'whale_activity_zscore': whale_activity_zscore,
            'whale_largest_move_recency': whale_largest_move_recency,
            'whale_btc_eth_pressure': whale_btc_eth_pressure,
        }

    async def _fetch_fresh_price(self, symbol: str) -> Optional[float]:
        """
        Fetch fresh price from Coinbase REST API.

        Uses the ticker endpoint to get the latest price, avoiding stale cache issues.
        Falls back to cache if REST fails.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')

        Returns:
            Latest price or None if unavailable
        """
        if not self._coinbase_auth:
            logger.debug(f"No auth available, using cached price for {symbol}")
            return self._price_cache.get(symbol)

        try:
            # Create session lazily
            if self._http_session is None or self._http_session.closed:
                self._http_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=5)
                )

            path = f"/api/v3/brokerage/products/{symbol}/ticker"
            url = f"https://api.coinbase.com{path}"
            headers = self._coinbase_auth.get_auth_headers(method="GET", path=path)

            async with self._http_session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Try trades array first
                    trades = data.get('trades', [])
                    if trades:
                        price = float(trades[0].get('price', 0))
                        if price > 0:
                            self._price_cache[symbol] = price
                            return price
                    # Fall back to ticker price
                    price_str = data.get('price')
                    if price_str:
                        price = float(price_str)
                        if price > 0:
                            self._price_cache[symbol] = price
                            return price
                else:
                    logger.debug(f"REST price fetch failed for {symbol}: {response.status}")

        except asyncio.TimeoutError:
            logger.debug(f"REST price fetch timed out for {symbol}")
        except Exception as e:
            logger.debug(f"REST price fetch error for {symbol}: {e}")

        # Fall back to cache
        return self._price_cache.get(symbol)

    def _get_order_book_features(self, symbol: str) -> dict:
        """
        Get order book features for a symbol from cache or manager.

        Returns:
            dict with bid_ask_spread_pct and order_book_depth_ratio
        """
        features = {
            'bid_ask_spread_pct': np.nan,
            'order_book_depth_ratio': np.nan,
        }

        # Try cache first (fastest)
        if symbol in self._order_book_cache:
            snapshot = self._order_book_cache[symbol]
            best_bid = snapshot.get('best_bid', 0)
            best_ask = snapshot.get('best_ask', 0)
            bid_depth = snapshot.get('bid_depth', 0)
            ask_depth = snapshot.get('ask_depth', 0)

            # Calculate spread as percentage
            if best_bid and best_bid > 0 and best_ask and best_ask > 0:
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                features['bid_ask_spread_pct'] = spread / mid_price

            # Calculate depth ratio (bid_depth / ask_depth, >1 = more buy pressure)
            if ask_depth and ask_depth > 0:
                features['order_book_depth_ratio'] = bid_depth / ask_depth

            return features

        # Fallback: query order book manager directly
        if self.order_book_manager is not None:
            try:
                # Get the order book for this symbol
                if symbol in self.order_book_manager.books:
                    book = self.order_book_manager.books[symbol]
                    best_bid = book.get_best_bid()
                    best_ask = book.get_best_ask()

                    if best_bid and best_ask and best_bid > 0 and best_ask > 0:
                        mid_price = (best_bid + best_ask) / 2
                        spread = best_ask - best_bid
                        features['bid_ask_spread_pct'] = spread / mid_price

                    # Calculate depth ratio from order book levels
                    bid_levels, ask_levels = book.get_depth(levels=5)
                    bid_depth = sum(level.quantity for level in bid_levels)
                    ask_depth = sum(level.quantity for level in ask_levels)

                    if ask_depth > 0:
                        features['order_book_depth_ratio'] = bid_depth / ask_depth

            except Exception as e:
                logger.debug(f"Error getting order book for {symbol}: {e}")

        return features

    def _had_recent_fast_spike(self, symbol: str, threshold: float = 0.15,
                                window_minutes: int = 5, lookback_hours: int = 6,
                                comedown_threshold: float = 0.10) -> bool:
        """
        Check if symbol is in a spike COMEDOWN (price-relative filter).

        Returns True if:
        1. A spike (+threshold% in window_minutes) occurred in last lookback_hours
        2. Current price is >comedown_threshold% BELOW that spike's high

        This allows catching current spikes (price still near high) while blocking
        entry on old spikes where we'd be catching the crash.

        Example:
        - BIRB spiked to $0.40, now at $0.35 (-12% from high) → BLOCK (in comedown)
        - TRIA spiked to $0.025, now at $0.024 (-4% from high) → ALLOW (still running)
        """
        try:
            ohlcv_df = self.feature_buffer.load_ohlcv_history(symbol, limit=lookback_hours * 60)
            if ohlcv_df is None or len(ohlcv_df) < window_minutes:
                return False

            # Find any spike in the lookback period and track its high
            spike_high = None

            for i in range(len(ohlcv_df) - window_minutes):
                window = ohlcv_df.iloc[i:i + window_minutes]
                low_price = window['low'].min()
                high_price = window['high'].max()

                if low_price > 0:
                    move_pct = (high_price - low_price) / low_price
                    if move_pct >= threshold:
                        # Found a spike - track the highest high
                        if spike_high is None or high_price > spike_high:
                            spike_high = high_price

            if spike_high is None:
                return False  # No spike found, allow entry

            # Get current price
            current_price = ohlcv_df.iloc[-1]['close']
            if current_price <= 0:
                return False

            # Check if current price is significantly below spike high
            drop_from_high = (spike_high - current_price) / spike_high

            if drop_from_high >= comedown_threshold:
                logger.debug(
                    f"[SPIKE_FILTER] {symbol}: spike_high=${spike_high:.6f}, "
                    f"current=${current_price:.6f}, drop={drop_from_high:.1%} >= {comedown_threshold:.0%} → BLOCK"
                )
                return True  # In comedown, block entry
            else:
                logger.debug(
                    f"[SPIKE_FILTER] {symbol}: spike_high=${spike_high:.6f}, "
                    f"current=${current_price:.6f}, drop={drop_from_high:.1%} < {comedown_threshold:.0%} → ALLOW"
                )
                return False  # Still near spike high, allow entry

        except Exception as e:
            logger.debug(f"Error checking recent spike for {symbol}: {e}")
            return False

    def _calculate_large_order_imbalance(self, symbol: str, buy_volume: float, sell_volume: float) -> float:
        """
        Calculate large order imbalance based on 90th percentile thresholds.

        Tracks rolling window of buy/sell volumes and returns:
        - +1 if current buy volume > 90th percentile (large buy pressure)
        - -1 if current sell volume > 90th percentile (large sell pressure)
        - 0 otherwise

        Args:
            symbol: Trading pair
            buy_volume: Current candle's buy volume
            sell_volume: Current candle's sell volume

        Returns:
            Large order imbalance (-1, 0, or +1)
        """
        history = self._volume_history[symbol]

        # Add current volumes to history
        history['buy'].append(buy_volume)
        history['sell'].append(sell_volume)

        # Keep rolling window size
        if len(history['buy']) > self._volume_history_size:
            history['buy'] = history['buy'][-self._volume_history_size:]
        if len(history['sell']) > self._volume_history_size:
            history['sell'] = history['sell'][-self._volume_history_size:]

        # Need enough history to calculate percentile
        if len(history['buy']) < 10:
            return 0.0

        # Calculate 90th percentile thresholds
        buy_threshold = np.percentile(history['buy'], 90)
        sell_threshold = np.percentile(history['sell'], 90)

        # Determine if current volumes are "large"
        large_buy = 1 if buy_volume > buy_threshold else 0
        large_sell = 1 if sell_volume > sell_threshold else 0

        return float(large_buy - large_sell)

    async def process_trade(
        self,
        symbol: str,
        price: float,
        size: float,
        side: str,
        timestamp: datetime
    ):
        """
        Process a trade and update features.

        Args:
            symbol: Trading pair
            price: Trade price
            size: Trade size
            side: Trade side ('BUY' or 'SELL')
            timestamp: Trade timestamp
        """
        # Update price cache for paper trading
        self._price_cache[symbol] = price

        # Add to OHLCV aggregator
        completed_candles = self.aggregator.add_trade(
            symbol, price, size, side, timestamp
        )

        self.stats['trades_processed'] += 1

        # Process any completed candles
        if completed_candles:
            await self._process_completed_candles(symbol, completed_candles)

    async def _process_completed_candles(
        self,
        symbol: str,
        candles: List
    ):
        """
        Process completed candles and calculate features.

        Args:
            symbol: Trading pair (may be overridden by candle.symbol)
            candles: List of completed OHLCVCandle objects
        """
        # Collect feature calculation tasks to run in parallel
        feature_tasks = []

        for candle in candles:
            # Use candle's symbol if available, otherwise use parameter
            candle_symbol = candle.symbol if hasattr(candle, 'symbol') and candle.symbol else symbol

            # Convert to DataFrame row
            candle_dict = candle.to_dict()
            candle_df = pd.DataFrame([candle_dict])
            candle_df.set_index('timestamp', inplace=True)

            # Normalize to tz-naive (remove timezone info for consistent comparison)
            if isinstance(candle_df.index, pd.DatetimeIndex) and candle_df.index.tz is not None:
                candle_df.index = candle_df.index.tz_localize(None)

            timeframe = candle.timeframe

            # Initialize buffer if needed
            if timeframe not in self.ohlcv_buffers[candle_symbol]:
                self.ohlcv_buffers[candle_symbol][timeframe] = pd.DataFrame()

            # Normalize buffer timestamps to tz-naive before concat
            buffer = self.ohlcv_buffers[candle_symbol][timeframe]
            if len(buffer) > 0 and isinstance(buffer.index, pd.DatetimeIndex) and buffer.index.tz is not None:
                buffer = buffer.copy()
                buffer.index = buffer.index.tz_localize(None)
            self.ohlcv_buffers[candle_symbol][timeframe] = pd.concat([buffer, candle_df])

            # Trim buffer to lookback window
            if len(self.ohlcv_buffers[candle_symbol][timeframe]) > self.lookback_bars:
                self.ohlcv_buffers[candle_symbol][timeframe] = \
                    self.ohlcv_buffers[candle_symbol][timeframe].iloc[-self.lookback_bars:]

            # Write candle to database
            self.db_manager.queue_candle(
                symbol=candle_symbol,
                timeframe=timeframe,
                data=candle_dict
            )

            # Write 1-minute candles to FeatureBuffer for V5 persistence (needs 1500 candles)
            if timeframe == '1m' and hasattr(self, 'feature_buffer'):
                candle_ts = candle_dict.get('timestamp')
                if candle_ts:
                    self.feature_buffer.write_ohlcv(
                        symbol=candle_symbol,
                        timestamp=candle_ts,
                        ohlcv={
                            'open': candle_dict.get('open', 0),
                            'high': candle_dict.get('high', 0),
                            'low': candle_dict.get('low', 0),
                            'close': candle_dict.get('close', 0),
                            'volume': candle_dict.get('volume', 0)
                        }
                    )

            self.stats['candles_completed'] += 1

            # Feature calculation moved to periodic loop (run_all_predictions)
            # This dramatically reduces per-trade processing load
            # Features are calculated every 60 seconds instead of every candle

    def _compute_features_sync(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        buy_volume: pd.Series,
        sell_volume: pd.Series
    ) -> pd.DataFrame:
        """
        Synchronous CPU-bound feature calculation.

        This runs in a thread pool to avoid blocking the event loop.
        """
        # Calculate price features
        if self.calculation_enabled['price']:
            price_features = calculate_price_features(
                open_, high, low, close, volume,
                rsi_period=14,
                atr_period=14,
                bb_period=20
            )
        else:
            price_features = pd.DataFrame(index=close.index)

        # Calculate volume features
        if self.calculation_enabled['volume']:
            volume_features = calculate_volume_features(
                close, volume,
                spike_window=20,
                spike_threshold=2.0,
                vroc_period=14
            )
        else:
            volume_features = pd.DataFrame(index=close.index)

        # Calculate microstructure features
        if self.calculation_enabled['microstructure']:
            microstructure_features = calculate_microstructure_features(
                prices=close,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                order_book_imbalance=None,
                roll_window=min(100, len(close)),
                vpin_window=min(50, len(close))
            )
        else:
            microstructure_features = pd.DataFrame(index=close.index)

        # Combine all features
        return pd.concat([
            price_features,
            volume_features,
            microstructure_features
        ], axis=1)

    async def _calculate_features(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime
    ):
        """
        Calculate all features for a symbol and timeframe.

        Args:
            symbol: Trading pair
            timeframe: Timeframe string
            timestamp: Feature timestamp
        """
        try:
            # Get OHLCV buffer
            ohlcv = self.ohlcv_buffers[symbol][timeframe].copy()

            if len(ohlcv) < 20:  # Minimum for indicators
                return

            # Ensure index is DatetimeIndex and tz-naive for consistent operations
            if not isinstance(ohlcv.index, pd.DatetimeIndex):
                ohlcv.index = pd.to_datetime(ohlcv.index)
            if ohlcv.index.tz is not None:
                ohlcv.index = ohlcv.index.tz_localize(None)
            # Sort to ensure proper time ordering
            ohlcv = ohlcv.sort_index()

            # Extract OHLCV components
            open_ = ohlcv['open']
            high = ohlcv['high']
            low = ohlcv['low']
            close = ohlcv['close']
            volume = ohlcv['volume']
            # Handle missing buy_volume/sell_volume (not in FeatureBuffer schema)
            if 'buy_volume' in ohlcv.columns:
                buy_volume = ohlcv['buy_volume']
            else:
                # Estimate: assume 50/50 split if not available
                buy_volume = volume * 0.5
            if 'sell_volume' in ohlcv.columns:
                sell_volume = ohlcv['sell_volume']
            else:
                sell_volume = volume * 0.5

            # Run CPU-bound calculations in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            all_features = await loop.run_in_executor(
                self._thread_pool,
                self._compute_features_sync,
                open_, high, low, close, volume, buy_volume, sell_volume
            )

            # Add multi-timeframe features if on primary timeframe (1m)
            if timeframe == self.primary_timeframe:
                # Initialize with NaN
                all_features['returns_5m'] = np.nan
                all_features['volume_zscore_5m'] = np.nan
                all_features['returns_15m'] = np.nan
                all_features['volume_zscore_15m'] = np.nan

                # Get 5m features if available
                if '5m' in self.ohlcv_buffers.get(symbol, {}):
                    buffer_5m = self.ohlcv_buffers[symbol]['5m']
                    if len(buffer_5m) >= 2:
                        # Calculate 5m returns
                        all_features.loc[all_features.index[-1], 'returns_5m'] = (
                            buffer_5m['close'].iloc[-1] / buffer_5m['close'].iloc[-2] - 1
                        )
                        # Calculate 5m volume z-score
                        vol_mean = buffer_5m['volume'].mean()
                        vol_std = buffer_5m['volume'].std()
                        if vol_std > 0:
                            all_features.loc[all_features.index[-1], 'volume_zscore_5m'] = (
                                (buffer_5m['volume'].iloc[-1] - vol_mean) / vol_std
                            )

                # Get 15m features if available
                if '15m' in self.ohlcv_buffers.get(symbol, {}):
                    buffer_15m = self.ohlcv_buffers[symbol]['15m']
                    if len(buffer_15m) >= 2:
                        # Calculate 15m returns
                        all_features.loc[all_features.index[-1], 'returns_15m'] = (
                            buffer_15m['close'].iloc[-1] / buffer_15m['close'].iloc[-2] - 1
                        )
                        # Calculate 15m volume z-score
                        vol_mean = buffer_15m['volume'].mean()
                        vol_std = buffer_15m['volume'].std()
                        if vol_std > 0:
                            all_features.loc[all_features.index[-1], 'volume_zscore_15m'] = (
                                (buffer_15m['volume'].iloc[-1] - vol_mean) / vol_std
                            )

            # Add order book features from live L2 data
            ob_features = self._get_order_book_features(symbol)
            all_features.loc[all_features.index[-1], 'bid_ask_spread_pct'] = ob_features['bid_ask_spread_pct']
            all_features.loc[all_features.index[-1], 'order_book_depth_ratio'] = ob_features['order_book_depth_ratio']

            # Calculate large_order_imbalance from buy/sell volumes (90th percentile threshold)
            latest_buy_vol = buy_volume.iloc[-1] if len(buy_volume) > 0 else 0.0
            latest_sell_vol = sell_volume.iloc[-1] if len(sell_volume) > 0 else 0.0
            all_features.loc[all_features.index[-1], 'large_order_imbalance'] = self._calculate_large_order_imbalance(
                symbol, latest_buy_vol, latest_sell_vol
            )

            # Add whale-derived features from whale alert data
            whale_features = self._calculate_whale_features(symbol)
            for k, v in whale_features.items():
                all_features.loc[all_features.index[-1], k] = v

            # Get latest row (current bar features)
            if len(all_features) > 0:
                latest_features = all_features.iloc[-1].to_dict()

                # Add metadata
                latest_features['symbol'] = symbol
                latest_features['timestamp'] = timestamp
                latest_features['timeframe'] = timeframe

                # Write to database
                self.db_manager.queue_features(symbol, timeframe, latest_features)

                # Cache for paper trading visualization
                self.latest_features[symbol] = latest_features

                # Write to persistent feature buffer for zero-warmup restarts
                if timeframe == self.primary_timeframe:
                    self.feature_buffer.write_features(symbol, timestamp, latest_features)
                    self._feature_buffer_write_counter += 1
                    # Prune old entries periodically
                    if self._feature_buffer_write_counter % self._feature_buffer_prune_interval == 0:
                        self.feature_buffer.prune_old()
                        # Also prune OHLCV to keep max 1500 candles per symbol
                        self.feature_buffer.prune_old_ohlcv(max_candles_per_symbol=1500)
                        # Prune order book snapshots (48h) and trades (3h) for Accumulation Hunter
                        self.feature_buffer.prune_old_order_books(retention_hours=6)
                        self.feature_buffer.prune_old_trades(retention_hours=3)
                        # VACUUM periodically to reclaim disk space
                        self._feature_buffer_vacuum_counter += 1
                        if self._feature_buffer_vacuum_counter % self._feature_buffer_vacuum_interval == 0:
                            try:
                                conn = self.feature_buffer._get_conn()
                                conn.execute("VACUUM")
                                conn.close()
                                logger.info("[FEATURE_BUFFER] VACUUM completed")
                            except Exception as e:
                                logger.debug(f"[FEATURE_BUFFER] VACUUM failed: {e}")

                self.stats['features_calculated'] += 1

                logger.debug(
                    f"Calculated features for {symbol} {timeframe}",
                    extra={
                        "timestamp": timestamp.isoformat(),
                        "num_features": len(latest_features),
                        "buffer_size": len(ohlcv)
                    }
                )

                # Calculate feature deltas for acceleration tracking
                prev = self._prev_features.get(symbol, {})
                deltas = {}
                current_vals = {}
                for feature in ['natr', 'volume_zscore_5m', 'returns_5m']:
                    curr_val = latest_features.get(feature, 0.0)
                    if curr_val is None or (isinstance(curr_val, float) and np.isnan(curr_val)):
                        curr_val = 0.0
                    current_vals[feature] = curr_val
                    prev_val = prev.get(feature, curr_val)
                    deltas[f'{feature}_delta'] = curr_val - prev_val

                # Save current values for next delta calculation
                self._prev_features[symbol] = current_vals

                # Run live prediction if models are loaded and this is the primary timeframe
                if (self.xgboost_model_v1 or self.xgboost_model_v3) and timeframe == self.primary_timeframe:
                    await self._run_prediction(latest_features, symbol, timestamp, timeframe, deltas)

                # Paper trading: process entry signals and manage positions
                if self.paper_trading_enabled and timeframe == self.primary_timeframe:
                    await self._process_paper_trading(latest_features, symbol, timestamp, deltas)

        except Exception as e:
            logger.error(
                f"Error calculating features for {symbol} {timeframe}: {e}",
                exc_info=True
            )

    async def _run_prediction(
        self,
        features_dict: dict,
        symbol: str,
        timestamp: datetime,
        timeframe: str,
        deltas: dict = None
    ):
        """
        Run XGBoost prediction with home run filter.

        Uses dual-model approach:
        - V3: Detects slow & large spike buildups (15 min early warning)
        - V1: Confirms current momentum is building

        Plus volatility filter (BB_width > 0.07) to select high-potential moves.

        Args:
            features_dict: Dictionary of features
            symbol: Trading pair
            timestamp: Feature timestamp
            timeframe: Timeframe string
        """
        try:
            # Extract features in correct order
            features = []
            for col in self.feature_columns:
                val = features_dict.get(col, 0.0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                features.append(val)

            features_array = np.array(features).reshape(1, -1)

            # Get BB_width for volatility filter
            bb_width = features_dict.get('BB_width', 0.0)
            if bb_width is None or (isinstance(bb_width, float) and np.isnan(bb_width)):
                bb_width = 0.0

            # Run V3 prediction only (optimized for speed - handles all 300+ symbols)
            # Offload to thread pool since predict_proba is CPU-bound
            v3_prob = 0.0
            if self.xgboost_model_v3:
                loop = asyncio.get_event_loop()
                v3_prob = await loop.run_in_executor(
                    self._thread_pool,
                    lambda: self.xgboost_model_v3.predict_proba(features_array)[0][1]
                )

            self.stats['predictions_made'] += 1

            # Cache V3 score for exit logic
            self._v3_cache[symbol] = v3_prob

            # Watch mode: trigger on V3 >= 70% for faster L2 polling
            if v3_prob >= 0.70:
                should_watch, reason = self.watch_mode.should_watch(
                    symbol, v3_confidence=v3_prob
                )
                if should_watch:
                    self.watch_mode.add_to_watch(symbol, reason)

            # Extract deltas for display
            vol_d = deltas.get('volume_zscore_5m_delta', 0.0) if deltas else 0.0
            natr_d = deltas.get('natr_delta', 0.0) if deltas else 0.0
            returns_5m = features_dict.get('returns_5m', 0.0)
            if returns_5m is None or (isinstance(returns_5m, float) and np.isnan(returns_5m)):
                returns_5m = 0.0

            # Log predictions periodically (every 100th) or high confidence (V3 >= 70%)
            if self.stats['predictions_made'] % 100 == 0 or v3_prob >= 0.70:
                logger.info(
                    f"[PRED] {symbol}: V3={v3_prob:.1%} "
                    f"vol_d={vol_d:.2f} natr_d={natr_d:.3f} ret5m={returns_5m:.3f}"
                )

            # Log high-confidence predictions that meet V3 threshold (>= 70%)
            if v3_prob >= 0.70:
                # Check if this also meets acceleration criteria for entry
                entry_criteria_met = (
                    vol_d > 0.5 and
                    returns_5m > 0 and
                    natr_d > 0.05
                )

                if entry_criteria_met:
                    logger.warning(
                        f"[V3 ENTRY SIGNAL] {symbol}: V3={v3_prob:.1%} | "
                        f"vol_d={vol_d:.2f} natr_d={natr_d:.3f} ret5m={returns_5m:.3f}"
                    )

        except Exception as e:
            logger.error(f"Error running prediction for {symbol}: {e}")

    async def run_all_predictions(self):
        """
        Run predictions for ALL symbols that have enough data.

        This is called periodically (every minute) to ensure all symbols
        get predictions, not just those with recent trades.

        Uses parallel execution for better performance.
        """
        timestamp = datetime.now(timezone.utc)
        timeframe = self.primary_timeframe

        # Collect eligible symbols
        eligible_symbols = []
        for symbol in list(self.ohlcv_buffers.keys()):
            if timeframe not in self.ohlcv_buffers[symbol]:
                continue
            buffer = self.ohlcv_buffers[symbol][timeframe]
            if len(buffer) >= 50:
                eligible_symbols.append(symbol)

        if not eligible_symbols:
            return 0

        # Run predictions in parallel batches (50 at a time to avoid overwhelming)
        BATCH_SIZE = 50
        symbols_processed = 0

        for i in range(0, len(eligible_symbols), BATCH_SIZE):
            batch = eligible_symbols[i:i + BATCH_SIZE]
            tasks = [
                self._calculate_features(symbol, timeframe, timestamp)
                for symbol in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            symbols_processed += sum(1 for r in results if not isinstance(r, Exception))

        logger.info(f"[PERIODIC] Ran predictions for {symbols_processed} symbols (parallel)")
        return symbols_processed

    async def _save_alert(self, alert: dict):
        """
        Append alert to JSON log file.

        Args:
            alert: Alert dictionary
        """
        try:
            # Load existing alerts
            alerts = []
            alert_path = Path(self.alert_log_file)
            if alert_path.exists():
                with open(alert_path, 'r') as f:
                    alerts = json.load(f)

            # Append new alert
            alerts.append(alert)

            # Save back
            with open(alert_path, 'w') as f:
                json.dump(alerts, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving alert: {e}")

    def get_latest_features(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[pd.Series]:
        """
        Get latest calculated features for a symbol/timeframe.

        Args:
            symbol: Trading pair
            timeframe: Timeframe string

        Returns:
            Series of features or None
        """
        if symbol not in self.ohlcv_buffers:
            return None

        if timeframe not in self.ohlcv_buffers[symbol]:
            return None

        ohlcv = self.ohlcv_buffers[symbol][timeframe]

        if len(ohlcv) == 0:
            return None

        # Return latest row
        return ohlcv.iloc[-1]

    def get_buffer(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV buffer for a symbol/timeframe.

        Args:
            symbol: Trading pair
            timeframe: Timeframe string

        Returns:
            DataFrame or None
        """
        return self.ohlcv_buffers.get(symbol, {}).get(timeframe)

    def add_symbols(self, new_symbols: list):
        """
        Dynamically add new symbols to the feature engine.

        Called when new Coinbase listings are discovered.
        Initializes all necessary data structures for the new symbols.

        Args:
            new_symbols: List of new symbol strings (e.g., ['ZKP-USD', 'BREV-USD'])
        """
        if not new_symbols:
            return

        # Add to symbols list (avoiding duplicates)
        existing = set(self.symbols)
        added = []
        for symbol in new_symbols:
            if symbol not in existing:
                self.symbols.append(symbol)
                added.append(symbol)

        if not added:
            return

        logger.info(f"Feature engine: Added {len(added)} new symbols: {added}")

        # Initialize order book cache for new symbols
        for symbol in added:
            if symbol not in self._order_book_cache:
                self._order_book_cache[symbol] = {}

        # ohlcv_buffers, _volume_history, and rolling_baseline use defaultdict
        # so they auto-initialize for new symbols on first access

        # Log total symbol count
        logger.info(f"Feature engine now tracking {len(self.symbols)} symbols")

    def get_statistics(self) -> dict:
        """Get engine statistics."""
        stats = {
            **self.stats,
            'symbols_tracked': len(self.symbols),
            'buffers_active': sum(len(buffers) for buffers in self.ohlcv_buffers.values()),
        }

        # Add aggregator stats
        stats.update(self.aggregator.get_statistics())

        # Add whale buffer stats
        stats['whale_buffer'] = self.whale_buffer.get_statistics()

        return stats

    def enable_feature_type(self, feature_type: str, enabled: bool):
        """
        Enable/disable specific feature calculation.

        Args:
            feature_type: 'microstructure', 'volume', or 'price'
            enabled: True to enable, False to disable
        """
        if feature_type in self.calculation_enabled:
            self.calculation_enabled[feature_type] = enabled
            logger.info(f"Feature type '{feature_type}' {'enabled' if enabled else 'disabled'}")

    # =========================================================================
    # News Signal Methods
    # =========================================================================

    def update_news_signal(self, signal: dict):
        """
        Update news signal cache with a new signal.

        Called by NewsMonitor when a new signal is detected.

        Args:
            signal: NewsSignal dict with symbol, confidence, event_type, etc.
        """
        symbol = signal.get('symbol')
        if not symbol:
            return

        self._news_signals[symbol] = {
            **signal,
            'received_at': datetime.now(timezone.utc),
        }

        logger.debug(f"News signal cached for {symbol}: {signal.get('event_type')} "
                    f"(conf={signal.get('confidence', 0):.2f})")

    def get_news_signal(self, symbol: str) -> Optional[dict]:
        """
        Get the most recent news signal for a symbol.

        Returns None if no signal exists or if signal is expired.

        Args:
            symbol: Trading pair (e.g., "SOL-USD")

        Returns:
            Signal dict if valid, None otherwise
        """
        signal = self._news_signals.get(symbol)
        if not signal:
            return None

        # Check if signal is expired
        received_at = signal.get('received_at')
        if received_at:
            age_minutes = (datetime.now(timezone.utc) - received_at).total_seconds() / 60
            if age_minutes > self._news_signal_ttl_minutes:
                # Signal expired, remove from cache
                del self._news_signals[symbol]
                return None

        return signal

    def get_active_news_signals(self, min_confidence: float = 0.6) -> List[dict]:
        """
        Get all active (non-expired) news signals above confidence threshold.

        Args:
            min_confidence: Minimum confidence score

        Returns:
            List of active news signals, sorted by confidence descending
        """
        now = datetime.now(timezone.utc)
        active = []

        expired_symbols = []
        for symbol, signal in self._news_signals.items():
            received_at = signal.get('received_at')
            if received_at:
                age_minutes = (now - received_at).total_seconds() / 60
                if age_minutes > self._news_signal_ttl_minutes:
                    expired_symbols.append(symbol)
                    continue

            if signal.get('confidence', 0) >= min_confidence:
                active.append(signal)

        # Clean up expired signals
        for symbol in expired_symbols:
            del self._news_signals[symbol]

        return sorted(active, key=lambda s: s.get('confidence', 0), reverse=True)

    def _check_news_entry(self, symbol: str, features_dict: dict) -> Optional[dict]:
        """
        Check if symbol has a valid news signal for entry.

        This is called during entry checking to see if news-based entry is warranted.

        Args:
            symbol: Trading pair
            features_dict: Current feature values (for additional filtering)

        Returns:
            Entry signal dict if valid news signal exists, None otherwise
        """
        signal = self.get_news_signal(symbol)
        if not signal:
            return None

        confidence = signal.get('confidence', 0)
        event_type = signal.get('event_type', '')

        # High-value events get lower threshold
        if event_type in ['listing', 'whale_move']:
            min_confidence = 0.5
        elif event_type in ['partnership', 'airdrop']:
            min_confidence = 0.6
        else:
            min_confidence = 0.7

        if confidence < min_confidence:
            return None

        # Additional volume/momentum confirmation for lower confidence signals
        if confidence < 0.7:
            volume_zscore = features_dict.get('volume_zscore', 0)
            if volume_zscore < 1.0:  # Need some volume confirmation
                return None

        return {
            'entry_type': 'news',
            'news_event': event_type,
            'news_confidence': confidence,
            'news_source': signal.get('source', 'unknown'),
            'news_title': signal.get('title', '')[:100],
        }

    # =========================================================================
    # Paper Trading Methods
    # =========================================================================

    async def _process_paper_trading(
        self,
        features_dict: dict,
        symbol: str,
        timestamp: datetime,
        deltas: dict
    ):
        """
        Process paper trading: detect entry signals and manage positions.

        Called after each feature calculation on the primary timeframe.

        Args:
            features_dict: Current feature values
            symbol: Trading pair
            timestamp: Current timestamp
            deltas: Pre-computed feature deltas from _calculate_features
        """
        try:
            # Track first feature time for warmup period
            if self._first_feature_time is None:
                self._first_feature_time = timestamp
                logger.info(f"Paper trading warmup started - will begin trading at {timestamp + timedelta(minutes=self._warmup_minutes)}")

            # Get or create per-symbol normalization stats
            if symbol not in self.norm_stats:
                self.norm_stats[symbol] = NormalizationStats(lookback_minutes=60)
            symbol_stats = self.norm_stats[symbol]

            # Update normalization stats for this symbol
            for feature in self.entry_features:
                val = features_dict.get(feature)
                if val is not None:
                    symbol_stats.add_value(feature, timestamp, val)

            symbol_stats.update_cache()

            # Note: deltas are now passed in from _calculate_features to avoid
            # race condition where _prev_features was already updated

            # Normalize features and record hits (data preserved in DB for post-hoc near-miss analysis)
            for feature in self.entry_features:
                raw_val = features_dict.get(feature)
                if raw_val is not None:
                    normalized = symbol_stats.normalize(feature, raw_val)
                    self.feature_tracker.record_hit(symbol, timestamp, feature, normalized)

            # Check for entry signals using V3 + acceleration filter FOR EACH STRATEGY
            v3_prob = self._get_v3_prediction(features_dict)

            # Use lock to prevent concurrent position modifications from multiple symbol trades
            async with self._position_lock:
                # Loop over all strategies - each can independently enter on same symbol
                for strategy in self.strategies:
                    entry_signal = None

                    if strategy.entry_type == "reactive":
                        # V4: Check L2 reactive signals
                        entry_signal = self._check_reactive_entry(symbol, timestamp, strategy)
                    elif strategy.entry_type == "volume":
                        # C5: Check volume explosion signals (with hybrid V3 filter)
                        entry_signal = self._check_volume_entry(symbol, timestamp, strategy, v3_prob)
                    elif strategy.entry_type == "event_classifier":
                        # V5: Check XGBoost event classifier for spike prediction
                        entry_signal = await self._check_event_classifier_entry(symbol, timestamp, strategy)
                    elif strategy.entry_type == "spike_continuation":
                        # X34: Detect massive spike candles and enter for continuation
                        entry_signal = self._check_spike_continuation_entry(symbol, timestamp, strategy)
                    else:
                        # V3: Check ML-based acceleration entry
                        entry_signal = self._check_v3_acceleration_entry(
                            symbol, timestamp, v3_prob, features_dict, deltas, strategy
                        )

                    if entry_signal:
                        await self._enter_position(symbol, entry_signal, timestamp, strategy)
                        # Save state immediately after entry
                        await self._save_paper_trading_state()

                # Check ignition watchlist for depth flip entries (independent of model signal)
                ignition_signals = await self._check_watchlist_ignition(symbol, timestamp)
                for ignition_signal in ignition_signals:
                    ignition_strategy = ignition_signal.get('strategy')
                    if ignition_strategy:
                        await self._enter_position(symbol, ignition_signal, timestamp, ignition_strategy)
                        await self._save_paper_trading_state()

                # Monitor open positions for exits (all strategies)
                await self._monitor_positions(timestamp)

            # Log status periodically
            await self._log_paper_trading_status(timestamp)

            # Log watchlist status periodically (every ~60 symbols processed)
            if self.stats['features_calculated'] % 60 == 0:
                self._log_watchlist_status()

        except Exception as e:
            logger.error(f"Error in paper trading for {symbol}: {e}")

    def _get_v3_prediction(self, features_dict: dict) -> float:
        """Get V3 model prediction for current features"""
        if not self.xgboost_model_v3:
            return 0.0

        try:
            features = []
            for col in self.feature_columns:
                val = features_dict.get(col, 0.0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                features.append(val)

            features_array = np.array(features).reshape(1, -1)
            return float(self.xgboost_model_v3.predict_proba(features_array)[0][1])
        except Exception as e:
            logger.error(f"Error getting V3 prediction: {e}")
            return 0.0

    def _check_v3_acceleration_entry(
        self,
        symbol: str,
        timestamp: datetime,
        v3_prob: float,
        features_dict: dict,
        deltas: dict,
        strategy: TradingStrategy
    ) -> Optional[dict]:
        """
        Check V3 + acceleration entry criteria for a specific strategy.

        Base conditions (always required):
        - V3 >= 0.7
        - volume_zscore_delta > 0.5

        Strategy-dependent conditions:
        - returns_5m > 0 (if strategy.require_ret5m)
        - NATR_delta > 0.05 (if strategy.require_natr)
        """
        # Check strategy-specific cooldown
        last_trigger = strategy.get_cooldown(symbol)
        if last_trigger:
            time_since = timestamp - last_trigger
            if time_since < timedelta(minutes=30):
                return None

        # V3 model confidence threshold (always required)
        if v3_prob < 0.80:
            return None

        # Acceleration conditions
        vol_delta = deltas.get('volume_zscore_5m_delta', 0.0)
        natr_delta = deltas.get('natr_delta', 0.0)
        returns_5m = features_dict.get('returns_5m', 0.0)

        # Handle NaN values
        if returns_5m is None or (isinstance(returns_5m, float) and np.isnan(returns_5m)):
            returns_5m = 0.0

        # vol_delta always required
        if vol_delta <= 0.5:
            return None

        # Dynamic ret5m thresholds based on V3 confidence
        # Higher V3 = trust model more = allow wider ret5m range
        if strategy.require_ret5m:
            if v3_prob >= 0.90:
                min_ret5m, max_ret5m = -0.05, 0.10  # Very high confidence
            elif v3_prob >= 0.85:
                min_ret5m, max_ret5m = -0.02, 0.05  # High confidence
            else:  # 0.80-0.85
                min_ret5m, max_ret5m = -0.01, 0.03  # Good confidence

            if returns_5m < min_ret5m or returns_5m > max_ret5m:
                return None

        # Conditional: NATR check (Strategy A and C skip this)
        if strategy.require_natr and natr_delta <= 0.05:
            return None

        # All conditions met for this strategy - record trigger
        strategy.set_cooldown(symbol, timestamp)

        # Build filter description for logging
        filters_used = ["V3", "vol_d"]
        if strategy.require_natr:
            filters_used.append("NATR")
        if strategy.require_ret5m:
            filters_used.append("ret5m")

        logger.info(f"[{strategy.name}] {symbol}: V3={v3_prob:.1%} vol_d={vol_delta:.2f} "
                    f"NATR_d={natr_delta:.3f} ret5m={returns_5m:.2%} | filters={'+'.join(filters_used)}")

        return {
            'v3_prob': v3_prob,
            'vol_delta': vol_delta,
            'natr_delta': natr_delta,
            'returns_5m': returns_5m,
            'trigger_type': 'V3_ACCELERATION',
            'strategy_name': strategy.name
        }

    def _check_reactive_entry(
        self,
        symbol: str,
        timestamp: datetime,
        strategy: TradingStrategy
    ) -> Optional[dict]:
        """
        Check L2 reactive entry criteria for V4 strategy.

        Entry conditions:
        1. BAR (bid_ask_ratio) >= bar_multiple_threshold * baseline
        2. Ask depth <= ask_collapse_threshold * baseline
        3. Price change <= max_price_change_entry (predictive filter)

        The key insight is that predictive signals appear BEFORE the price moves.
        If price has already moved significantly, the signal is descriptive not predictive.
        """
        # Check strategy-specific cooldown
        last_trigger = strategy.get_cooldown(symbol)
        if last_trigger:
            time_since = timestamp - last_trigger
            if time_since < timedelta(minutes=30):
                return None

        # Get current L2 data from cache
        if symbol not in self._order_book_cache:
            return None

        snapshot = self._order_book_cache[symbol]
        bid_depth = snapshot.get('bid_depth', 0)
        ask_depth = snapshot.get('ask_depth', 0)
        price = self._price_cache.get(symbol, 0)

        if not bid_depth or not ask_depth or not price:
            return None

        # Check for reactive signal using rolling baseline
        signal = self.rolling_baseline.check_reactive_signal(
            symbol=symbol,
            current_bid_depth=bid_depth,
            current_ask_depth=ask_depth,
            current_price=price,
            bar_multiple_threshold=strategy.bar_multiple_threshold,
            ask_collapse_threshold=strategy.ask_collapse_threshold,
            max_price_change=strategy.max_price_change_entry,
            min_conditions=strategy.min_reactive_conditions
        )

        if not signal:
            return None

        # Signal detected - record cooldown
        strategy.set_cooldown(symbol, timestamp)

        # Build condition status string
        cond_status = []
        if signal.get('bar_condition'): cond_status.append(f"BAR={signal['bar_multiple']:.1f}x")
        if signal.get('ask_condition'): cond_status.append(f"Ask={signal['ask_collapse']:.0%}")
        if signal.get('price_condition'): cond_status.append(f"Price={signal['price_change']:.1%}")

        logger.info(
            f"[{strategy.name}] REACTIVE SIGNAL: {symbol} | "
            f"{signal.get('conditions_met', 3)}/3 conditions met: {', '.join(cond_status)}"
        )

        return {
            'bar_multiple': signal['bar_multiple'],
            'ask_collapse': signal['ask_collapse'],
            'price_change': signal['price_change'],
            'current_bar': signal['current_bar'],
            'baseline_bar': signal['baseline_bar'],
            'trigger_type': 'L2_REACTIVE',
            'strategy_name': strategy.name
        }

    def _check_volume_entry(
        self,
        symbol: str,
        timestamp: datetime,
        strategy: TradingStrategy,
        v3_prob: float = 0.0
    ) -> Optional[dict]:
        """
        Check volume explosion entry criteria for C5 strategy.

        Entry conditions:
        1. Volume multiple >= threshold (default 3x normal)
        2. Price change >= min_price_change_entry (default 5%) - confirms momentum
        3. Price change <= max_price_change_entry (default 50%) - not already pumped

        HYBRID V3 FILTER (added 2026-01-14):
        - US hours (17:00-19:59 UTC): No V3 filter - volume signals are reliable
        - After hours (20:00-16:59 UTC): Require V3 >= 70% - filter out fakeouts

        Analysis showed:
        - US hours: 64% win rate without filter
        - After hours: 23% win rate without filter, but V3>=70% improves to profitable
        """
        # HYBRID V3 FILTER: Check time-based V3 requirement
        current_hour = timestamp.hour
        is_us_hours = 14 <= current_hour <= 21  # 9 AM - 4 PM ET

        if not is_us_hours:
            # After hours: require V3 >= 70%
            if v3_prob < 0.70:
                return None
        # Check strategy-specific cooldown (30 min)
        last_trigger = strategy.get_cooldown(symbol)
        if last_trigger:
            time_since = timestamp - last_trigger
            if time_since < timedelta(minutes=30):
                return None

        # Get cached volume signal (must be < 5 min old)
        signal = self.get_volume_signal(symbol)
        if not signal:
            return None

        volume_multiple = signal['volume_multiple']
        price_change = signal['price_change']

        # Check volume threshold
        if volume_multiple < strategy.volume_multiple_threshold:
            return None

        # Check price is moving (min threshold - confirms momentum)
        if price_change < strategy.min_price_change_entry:
            return None

        # Check price hasn't moved too much (max threshold - not exhausted)
        if price_change > strategy.max_price_change_entry:
            return None

        # Signal valid - record cooldown
        strategy.set_cooldown(symbol, timestamp)

        # Log with V3 filter status
        filter_status = "US_HOURS" if is_us_hours else f"V3={v3_prob:.0%}"
        logger.info(
            f"[{strategy.name}] VOLUME SIGNAL: {symbol} | "
            f"Vol={volume_multiple:.1f}x | Price={price_change:+.1%} | {filter_status}"
        )

        return {
            'volume_multiple': volume_multiple,
            'price_change': price_change,
            'trigger_type': 'VOLUME_EXPLOSION',
            'strategy_name': strategy.name,
            'v3_prob': v3_prob,
            'is_us_hours': is_us_hours
        }

    def _check_spike_continuation_entry(
        self,
        symbol: str,
        timestamp: datetime,
        strategy: TradingStrategy
    ) -> Optional[dict]:
        """
        Detect massive spike candles and enter for continuation.
        Triggers when: single candle has >15% move AND >30x volume.

        Key insight: You cannot catch the initial spike with 1-minute candles,
        but you CAN detect it immediately after (when candle closes) and ride
        the continuation.

        Example (BIRB 2026-02-03):
        - 07:00 candle: +22.8% price, 100x volume = unmistakable signal
        - After detection at 07:01, price went from $0.34 → $0.41 (+20% continuation)
        """
        if strategy.get_cooldown(symbol):
            return None
        if strategy.portfolio.get_position(symbol):
            return None

        # Get recent candles
        ohlcv = self.feature_buffer.load_ohlcv_history(symbol, limit=60)
        if ohlcv is None or len(ohlcv) < 30:
            return None

        # Check last completed candle
        last_candle = ohlcv.iloc[-1]
        open_price = last_candle['open']
        close_price = last_candle['close']
        current_vol = last_candle['volume']

        # Must be UP candle
        if close_price <= open_price:
            return None

        candle_move = (close_price - open_price) / open_price if open_price > 0 else 0

        # Calculate avg volume (exclude last candle)
        avg_vol = ohlcv['volume'].iloc[-30:-1].mean()
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0

        # SPIKE DETECTION: >15% single-candle move with >30x volume
        if candle_move >= 0.15 and vol_ratio >= 30.0:
            # Check order book for continuation potential
            ob_features = self._get_order_book_features(symbol)
            depth_ratio = ob_features.get('order_book_depth_ratio', 1.0)

            # Only enter if order book is bullish (depth_ratio > 1.0 = more bids than asks)
            if depth_ratio >= 1.0:
                strategy.set_cooldown(symbol, timestamp)
                logger.warning(
                    f"[{strategy.name}] SPIKE_CONTINUATION_ENTRY: {symbol} | "
                    f"Candle={candle_move:.1%} | Vol={vol_ratio:.0f}x | Depth={depth_ratio:.2f}"
                )
                return {
                    'trigger_type': 'SPIKE_CONTINUATION',
                    'strategy_name': strategy.name,
                    'candle_move': candle_move,
                    'vol_ratio': vol_ratio,
                    'depth_ratio': depth_ratio,
                }

        return None

    async def _check_event_classifier_entry(
        self,
        symbol: str,
        timestamp: datetime,
        strategy: TradingStrategy
    ) -> Optional[dict]:
        """
        Check V5 event classifier for spike prediction using two-tier entry system.

        TIER 1 - Immediate Entry (any of):
        - Volume explosion from VolumeScanner (24h vol >= 3x 30-day avg)
        - V5 prob >= 80% AND (15-min volume spike >= 3x OR 15-min momentum >= 3%)
        - V5 prob >= 80% AND combined (volume >= 2x AND momentum >= 1.5%)

        TIER 2 - Watch Mode:
        - V5 prob >= 95% but no Tier 1 signal
        - Add to watchlist, enter if +2% from watch price
        - Bail if -2% or 60 min timeout
        """
        # Get the model for this strategy (custom or default)
        model_path = strategy.event_classifier_model_path or self._default_classifier_path
        classifier_data = self._event_classifiers.get(model_path)
        if classifier_data is None:
            return None

        # Check strategy-specific cooldown (30 min)
        last_trigger = strategy.get_cooldown(symbol)
        if last_trigger:
            time_since = timestamp - last_trigger
            if time_since < timedelta(minutes=30):
                return None

        # Check if already has position in this symbol
        if strategy.portfolio.get_position(symbol):
            return None

        # PRICE-RELATIVE SPIKE FILTER: Skip if coin is in spike comedown
        # Blocks entry if: (1) had +15% spike in last 6h AND (2) current price >10% below spike high
        # Allows entry if: no spike, or price still near spike high (spike still running)
        if self._had_recent_fast_spike(symbol, threshold=0.15, window_minutes=5, lookback_hours=6, comedown_threshold=0.10):
            logger.info(f"[{strategy.name}] SPIKE_FILTER_REJECT: {symbol} | In spike comedown (>10% below spike high)")
            return None

        # Get current price for watch mode checks
        current_price = self.get_current_price(symbol)
        if current_price is None:
            return None

        # First, check for watch mode breakout (Tier 2 completion)
        if strategy.event_classifier_watch_mode:
            watch_entry = self._check_v5_watch_entry(symbol, current_price, timestamp, strategy)
            if watch_entry:
                strategy.set_cooldown(symbol, timestamp)
                return watch_entry

        try:
            # === TIER 1A: DISABLED - revisit as combo with model tomorrow ===
            # volume_signal = self.get_volume_signal(symbol)
            # if volume_signal:
            #     vol_mult = volume_signal.get('volume_multiple', 0)
            #     price_chg = volume_signal.get('price_change', 0)
            #
            #     if vol_mult >= strategy.event_classifier_volume_spike_threshold:
            #         strategy.set_cooldown(symbol, timestamp)
            #         logger.info(
            #             f"[{strategy.name}] TIER1 VOLUME EXPLOSION: {symbol} | "
            #             f"VolumeMultiple={vol_mult:.1f}x (>={strategy.event_classifier_volume_spike_threshold:.1f}x) | "
            #             f"PriceChange={price_chg:+.1%}"
            #         )
            #         return {
            #             'event_prob': 0.0,  # No V5 model used
            #             'trigger_type': 'TIER1_VOLUME_EXPLOSION',
            #             'strategy_name': strategy.name,
            #             'volume_multiple': vol_mult,
            #             'price_change_24h': price_chg
            #         }

            # === Compute V5 event features ===
            features = await self._compute_event_features(symbol, timestamp)
            if features is None:
                return None

            # Scale features and get prediction using strategy's model
            model = classifier_data['model']
            scaler = classifier_data['scaler']
            feature_cols = classifier_data['features']

            feature_values = [features.get(col, 0.0) for col in feature_cols]
            feature_array = np.array(feature_values).reshape(1, -1)
            scaled_features = scaler.transform(feature_array)
            pred_proba = float(model.predict_proba(scaled_features)[0][1])

            # === MODEL-ONLY ENTRY: Enter if probability >= threshold ===
            # No volume/momentum confirmation needed
            if pred_proba >= strategy.event_classifier_threshold:
                # === BULLISH GATE + IGNITION WATCHLIST ===
                # Get real-time order book state (L2)
                ob_features = self._get_order_book_features(symbol)
                depth_ratio = ob_features.get('order_book_depth_ratio', 1.0)

                # depth_ratio = bid_depth / ask_depth
                # > 1.0 = more bids than asks = buying pressure (bullish)
                # < 1.0 = more asks than bids = selling pressure (bearish)
                is_bullish = depth_ratio is not None and not np.isnan(depth_ratio) and depth_ratio > 1.0

                if is_bullish:
                    # IGNITION: Depth is bullish - check if on watchlist or enter directly
                    watchlist_key = f"{symbol}:{strategy.name}"
                    if watchlist_key in self._ignition_watchlist:
                        watch_info = self._ignition_watchlist.pop(watchlist_key)
                        logger.info(
                            f"[{strategy.name}] IGNITION: {symbol} | "
                            f"WatchProb={watch_info['prob']:.1%} | CurrentProb={pred_proba:.1%} | "
                            f"DepthFlip={watch_info['initial_depth']:.2f}→{depth_ratio:.2f}"
                        )
                    else:
                        logger.info(
                            f"[{strategy.name}] MODEL_ENTRY: {symbol} | Prob={pred_proba:.1%} >= {strategy.event_classifier_threshold:.0%} | DepthRatio={depth_ratio:.2f}"
                        )

                    strategy.set_cooldown(symbol, timestamp)
                    return {
                        'event_prob': pred_proba,
                        'trigger_type': 'MODEL_ONLY',
                        'strategy_name': strategy.name,
                        'features': features,
                    }
                else:
                    # WATCHLIST: High prob but bearish depth - add to watch for ignition
                    watchlist_key = f"{symbol}:{strategy.name}"
                    if watchlist_key not in self._ignition_watchlist:
                        self._ignition_watchlist[watchlist_key] = {
                            'timestamp': timestamp,
                            'prob': pred_proba,
                            'strategy': strategy.name,
                            'initial_depth': depth_ratio,
                            'entry_price': current_price,
                        }
                        logger.info(
                            f"[{strategy.name}] WATCHLIST_ADD: {symbol} | Prob={pred_proba:.1%} | Depth={depth_ratio:.2f}"
                        )
                    return None

            return None

        except Exception as e:
            logger.error(f"Error in event classifier entry check for {symbol}: {e}")
            return None

    async def _check_watchlist_ignition(self, symbol: str, timestamp: datetime) -> List[dict]:
        """
        Check if watchlisted symbol has depth flip - enter if so.

        This is called periodically to check symbols that had high model probability
        but bearish depth at signal time. When depth flips bullish, we enter.

        Returns a list of entry signals (one per strategy that was watching).
        """
        signals = []

        # Find any watchlist entries for this symbol (across all strategies)
        matching_keys = [k for k in self._ignition_watchlist.keys() if k.startswith(f"{symbol}:")]
        if not matching_keys:
            return signals

        # Check depth once (same for all strategies)
        ob_features = self._get_order_book_features(symbol)
        depth_ratio = ob_features.get('order_book_depth_ratio', 1.0)
        is_bullish = depth_ratio is not None and not np.isnan(depth_ratio) and depth_ratio > 1.0

        for watchlist_key in matching_keys:
            watch_info = self._ignition_watchlist.get(watchlist_key)
            if not watch_info:
                continue

            # Check age - remove if too old
            age_hours = (timestamp - watch_info['timestamp']).total_seconds() / 3600
            if age_hours > self._watchlist_max_age_hours:
                self._ignition_watchlist.pop(watchlist_key, None)
                logger.info(f"WATCHLIST_EXPIRE: {symbol} after {age_hours:.1f}h")
                continue

            if is_bullish:
                # IGNITION! Depth flipped bullish
                self._ignition_watchlist.pop(watchlist_key, None)

                # Find the strategy and create entry signal
                for strategy in self.strategies:
                    if strategy.name == watch_info['strategy']:
                        # Check if already has position
                        if strategy.portfolio.get_position(symbol):
                            continue

                        # Check cooldown
                        last_trigger = strategy.get_cooldown(symbol)
                        if last_trigger:
                            time_since = timestamp - last_trigger
                            if time_since < timedelta(minutes=30):
                                continue

                        logger.info(
                            f"[{strategy.name}] IGNITION: {symbol} | "
                            f"WatchProb={watch_info['prob']:.1%} | "
                            f"DepthFlip={watch_info['initial_depth']:.2f}→{depth_ratio:.2f} | "
                            f"WaitTime={age_hours:.1f}h"
                        )
                        strategy.set_cooldown(symbol, timestamp)
                        signals.append({
                            'event_prob': watch_info['prob'],
                            'trigger_type': 'IGNITION',
                            'strategy_name': strategy.name,
                            'strategy': strategy,
                            'watch_time_hours': age_hours,
                            'initial_depth': watch_info['initial_depth'],
                            'current_depth': depth_ratio,
                        })

        return signals

    def _log_watchlist_status(self):
        """Log current watchlist state for monitoring."""
        if self._ignition_watchlist:
            symbols = list(set(k.split(':')[0] for k in self._ignition_watchlist.keys()))
            logger.info(f"[WATCHLIST] {len(self._ignition_watchlist)} entries: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")

    async def _compute_event_features(self, symbol: str, timestamp: datetime) -> Optional[dict]:
        """
        Compute event classifier features from OHLCV history.

        Requires ~25 hours (1500 min) of history for proper feature computation.
        """
        try:
            # Get OHLCV history from database
            ohlcv_df = await self._get_ohlcv_history(symbol, minutes=1500)
            if ohlcv_df is None or len(ohlcv_df) < 1500:
                candles = 0 if ohlcv_df is None else len(ohlcv_df)
                # Log once per symbol to avoid spam
                if not hasattr(self, '_v5_insufficient_logged'):
                    self._v5_insufficient_logged = set()
                if symbol not in self._v5_insufficient_logged:
                    self._v5_insufficient_logged.add(symbol)
                    logger.info(f"[V5] {symbol}: insufficient history ({candles}/1500 candles) - skipping")
                return None

            df = ohlcv_df.copy()
            close = df['close'].iloc[-1]
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            volume = df['volume'].iloc[-1]

            features = {}

            # Returns
            df['returns'] = df['close'].pct_change()

            # NATR (Normalized ATR)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean().iloc[-1]
            features['natr_14'] = (atr_14 / close * 100) if close > 0 else 0

            # Bollinger Band width
            bb_middle = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            bb_width = ((bb_middle.iloc[-1] + 2 * bb_std.iloc[-1]) - (bb_middle.iloc[-1] - 2 * bb_std.iloc[-1])) / bb_middle.iloc[-1]
            features['bb_width_20'] = bb_width if pd.notna(bb_width) else 0

            # Volatility ratios
            vol_20 = df['returns'].rolling(20).std().iloc[-1]
            vol_60 = df['returns'].rolling(60).std().iloc[-1]
            vol_240 = df['returns'].rolling(240).std().iloc[-1]
            features['vol_ratio_20_60'] = (vol_20 / vol_60) if pd.notna(vol_60) and vol_60 > 0 else 1
            features['vol_ratio_60_240'] = (vol_60 / vol_240) if pd.notna(vol_240) and vol_240 > 0 else 1

            # Volatility acceleration
            vol_20_series = df['returns'].rolling(20).std()
            vol_acceleration = (vol_20_series.iloc[-1] - vol_20_series.iloc[-60]) / (vol_20_series.iloc[-60] + 1e-10) if len(vol_20_series) > 60 else 0
            features['vol_acceleration'] = vol_acceleration if pd.notna(vol_acceleration) else 0

            # Volume features
            vol_ma_20 = df['volume'].rolling(20).mean().iloc[-1]
            vol_ma_6hr = df['volume'].rolling(360).mean()
            features['volume_vs_ma20'] = (volume / vol_ma_20) if pd.notna(vol_ma_20) and vol_ma_20 > 0 else 1
            vol_trend = (vol_ma_20 - vol_ma_6hr.iloc[-1]) / (vol_ma_6hr.iloc[-1] + 1e-10) if len(vol_ma_6hr) > 0 and pd.notna(vol_ma_6hr.iloc[-1]) else 0
            features['volume_trend_6hr'] = vol_trend if pd.notna(vol_trend) else 0

            # OBV slope
            obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
            obv_slope = (obv.iloc[-1] - obv.iloc[-60]) / 60 if len(obv) > 60 else 0
            features['obv_slope_1hr'] = obv_slope if pd.notna(obv_slope) else 0

            # VROC
            vroc = (volume - df['volume'].iloc[-12]) / (df['volume'].iloc[-12] + 1e-10) if len(df) > 12 else 0
            features['vroc_12'] = vroc if pd.notna(vroc) else 0

            # Volume-price divergence
            price_range_60 = (df['close'].iloc[-60:].max() - df['close'].iloc[-60:].min()) / close if len(df) >= 60 else 0
            vol_change_60 = (vol_ma_20 - df['volume'].iloc[-60:-40].mean()) / (df['volume'].iloc[-60:-40].mean() + 1e-10) if len(df) >= 60 else 0
            vol_price_divergence = vol_change_60 if (price_range_60 < 0.02 and vol_change_60 > 0.3) else 0
            features['vol_price_divergence'] = vol_price_divergence if pd.notna(vol_price_divergence) else 0

            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rs = gain / (loss + 1e-10)
            features['rsi_14'] = (100 - (100 / (1 + rs))) if pd.notna(rs) else 50

            # Returns at horizons
            features['returns_1hr'] = (close / df['close'].iloc[-60] - 1) if len(df) > 60 else 0
            features['returns_6hr'] = (close / df['close'].iloc[-360] - 1) if len(df) > 360 else 0
            features['returns_12hr'] = (close / df['close'].iloc[-720] - 1) if len(df) > 720 else 0

            # Momentum
            features['momentum_5'] = (close / df['close'].iloc[-5] - 1) if len(df) > 5 else 0
            features['momentum_20'] = (close / df['close'].iloc[-20] - 1) if len(df) > 20 else 0

            # Price position
            high_24hr = df['high'].iloc[-1440:].max() if len(df) >= 1440 else df['high'].max()
            low_24hr = df['low'].iloc[-1440:].min() if len(df) >= 1440 else df['low'].min()
            features['dist_from_24hr_high'] = (close - high_24hr) / high_24hr if high_24hr > 0 else 0
            features['dist_from_24hr_low'] = (close - low_24hr) / low_24hr if low_24hr > 0 else 0

            # BB position
            bb_lower = bb_middle.iloc[-1] - 2 * bb_std.iloc[-1]
            bb_upper = bb_middle.iloc[-1] + 2 * bb_std.iloc[-1]
            bb_range = bb_upper - bb_lower
            features['bb_position'] = (close - bb_lower) / bb_range if pd.notna(bb_range) and bb_range > 0 else 0.5

            # HL range
            features['hl_range'] = (high - low) / close if close > 0 else 0

            # Body ratio
            body = (df['close'] - df['open']).abs()
            wick = df['high'] - df['low']
            body_ratio = body / (wick + 1e-10)
            features['body_ratio_avg_1hr'] = body_ratio.iloc[-60:].mean() if len(body_ratio) >= 60 else 0.5

            # Range compression
            range_20 = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / close
            range_60 = (df['high'].rolling(60).max() - df['low'].rolling(60).min()) / close
            features['range_compression'] = (range_20.iloc[-1] / range_60.iloc[-1]) if pd.notna(range_60.iloc[-1]) and range_60.iloc[-1] > 0 else 1

            # Temporal features
            ts = timestamp
            hour = ts.hour
            dow = ts.weekday()
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

            # Clean up NaN/inf
            for k, v in features.items():
                if pd.isna(v) or np.isinf(v):
                    features[k] = 0

            return features

        except Exception as e:
            logger.error(f"Error computing event features for {symbol}: {e}")
            return None

    async def _get_ohlcv_history(self, symbol: str, minutes: int = 1500) -> Optional[pd.DataFrame]:
        """Get OHLCV history from FeatureBuffer (SQLite) for V5 feature computation."""
        try:
            # Use FeatureBuffer (SQLite) for persistent OHLCV storage
            # This allows V5 to have 1500 candles immediately on restart
            if not hasattr(self, 'feature_buffer'):
                logger.debug(f"No feature_buffer available")
                return None

            df = self.feature_buffer.load_ohlcv_history(symbol, limit=minutes)

            if df is None or len(df) == 0:
                logger.debug(f"No OHLCV data in FeatureBuffer for {symbol}")
                return None

            # Ensure we have required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.debug(f"OHLCV buffer for {symbol} missing required columns")
                return None

            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV history for {symbol}: {e}")
            return None

    def _compute_volume_spike_15m(self, ohlcv_df: pd.DataFrame) -> float:
        """
        Compute volume spike ratio: recent 15-min volume vs 6-hour baseline.
        Returns ratio (e.g., 3.0 = 3x normal volume).
        """
        try:
            if ohlcv_df is None or len(ohlcv_df) < 15:
                return 1.0

            # Last 15 minutes of volume
            recent_volume = ohlcv_df['volume'].iloc[-15:].sum()

            # Average 15-min volume over last 6 hours (24 periods of 15 min = 360 candles)
            if len(ohlcv_df) >= 360:
                total_6hr_volume = ohlcv_df['volume'].iloc[-360:].sum()
                baseline_volume = total_6hr_volume / 24  # Average per 15-min period
            else:
                # Use available data
                n_periods = max(1, len(ohlcv_df) // 15)
                total_volume = ohlcv_df['volume'].sum()
                baseline_volume = total_volume / n_periods

            if baseline_volume > 0:
                return recent_volume / baseline_volume
            return 1.0

        except Exception as e:
            logger.debug(f"Error computing volume spike: {e}")
            return 1.0

    def _compute_volume_spike_1h_gapped(self, ohlcv_df: pd.DataFrame) -> float:
        """
        DEPRECATED - Too slow, detects moves 4+ hours late.
        Kept for reference. Use _compute_volume_surge_5m instead.
        """
        return 1.0  # Disabled

    def _compute_volume_surge_5m(self, ohlcv_df: pd.DataFrame) -> float:
        """
        Ultra-fast volume surge detection: last 2 min vs baseline from 5-20 min ago.

        Catches volume explosions within 2 minutes of onset.

        Timeline (minutes ago):
        -20       -5    -2    NOW
        [=BASELINE=] gap [NOW]
          15 min   3m   2m

        The 3-minute gap keeps baseline clean.
        Only needs 20 minutes of data.

        Returns ratio (e.g., 3.0 = 3x normal volume).
        """
        try:
            if ohlcv_df is None or len(ohlcv_df) < 20:
                # Need at least 20 minutes of data
                return 1.0

            # Current 2 minutes of volume (last 2 candles)
            current_2m_volume = ohlcv_df['volume'].iloc[-2:].sum()

            # Baseline: 15 minutes from 5-20 minutes ago
            # iloc[-20:-5] = from 20 minutes ago to 5 minutes ago
            baseline_15m_volume = ohlcv_df['volume'].iloc[-20:-5].sum()
            baseline_2m_avg = baseline_15m_volume / 7.5  # Average per 2-min period

            if baseline_2m_avg > 0:
                return current_2m_volume / baseline_2m_avg
            return 1.0

        except Exception as e:
            logger.debug(f"Error computing 2m volume surge: {e}")
            return 1.0

    def _compute_price_momentum_15m(self, ohlcv_df: pd.DataFrame) -> float:
        """
        Compute price momentum over last 15 minutes.
        Returns percentage change (e.g., 0.03 = +3%).
        """
        try:
            if ohlcv_df is None or len(ohlcv_df) < 15:
                return 0.0

            price_now = ohlcv_df['close'].iloc[-1]
            price_15m_ago = ohlcv_df['close'].iloc[-15]

            if price_15m_ago > 0:
                return (price_now - price_15m_ago) / price_15m_ago
            return 0.0

        except Exception as e:
            logger.debug(f"Error computing price momentum: {e}")
            return 0.0

    def _check_v5_watch_entry(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
        strategy: TradingStrategy
    ) -> Optional[dict]:
        """
        Check if a watched symbol should trigger entry.
        Returns entry signal if triggered, None otherwise.
        Also handles watch expiration/bail.
        """
        if symbol not in self._v5_watchlist:
            return None

        watch = self._v5_watchlist[symbol]
        watch_price = watch['watch_price']
        watch_time = watch['watch_time']

        # Calculate price change from watch price
        price_change = (current_price - watch_price) / watch_price if watch_price > 0 else 0
        time_in_watch = timestamp - watch_time

        # Entry trigger: price up >= threshold
        if price_change >= strategy.event_classifier_watch_trigger_pct:
            logger.info(
                f"[{strategy.name}] WATCH BREAKOUT: {symbol} | "
                f"Price +{price_change:.1%} from watch (${watch_price:.4f} -> ${current_price:.4f}) | "
                f"Watched {time_in_watch.total_seconds()/60:.0f}min | Prob={watch['prob']:.1%}"
            )
            del self._v5_watchlist[symbol]
            return {
                'event_prob': watch['prob'],
                'trigger_type': 'WATCH_BREAKOUT',
                'strategy_name': strategy.name,
                'watch_price': watch_price,
                'price_change_from_watch': price_change,
                'time_in_watch_min': time_in_watch.total_seconds() / 60
            }

        # Bail: price dropped too much
        if price_change <= -strategy.event_classifier_watch_bail_pct:
            logger.info(
                f"[{strategy.name}] WATCH BAIL: {symbol} | "
                f"Price {price_change:+.1%} (dropped below -{strategy.event_classifier_watch_bail_pct:.0%})"
            )
            del self._v5_watchlist[symbol]
            return None

        # Timeout: watched too long
        if time_in_watch > timedelta(minutes=strategy.event_classifier_watch_timeout_min):
            logger.info(
                f"[{strategy.name}] WATCH TIMEOUT: {symbol} | "
                f"{time_in_watch.total_seconds()/60:.0f}min > {strategy.event_classifier_watch_timeout_min}min limit"
            )
            del self._v5_watchlist[symbol]
            return None

        return None

    def _add_to_v5_watchlist(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        prob: float,
        strategy: TradingStrategy
    ):
        """Add a symbol to the V5 watchlist."""
        self._v5_watchlist[symbol] = {
            'watch_price': price,
            'watch_time': timestamp,
            'prob': prob,
            'strategy': strategy.name
        }
        logger.info(
            f"[{strategy.name}] WATCH ADD: {symbol} | "
            f"Prob={prob:.1%} | Price=${price:.4f} | "
            f"Will trigger at +{strategy.event_classifier_watch_trigger_pct:.0%}"
        )

    async def _enter_position(self, symbol: str, signal: dict, timestamp: datetime, strategy: TradingStrategy):
        """Enter a new paper trading position for a specific strategy"""
        portfolio = strategy.portfolio

        # Check warmup period - don't trade until normalization has enough data
        if self._first_feature_time:
            warmup_elapsed = (timestamp - self._first_feature_time).total_seconds() / 60
            if warmup_elapsed < self._warmup_minutes:
                logger.debug(f"[{strategy.name}] Skipping {symbol} entry - warmup {warmup_elapsed:.1f}/{self._warmup_minutes} min")
                return

        if not portfolio.can_open_position():
            return

        # Check if this strategy already has position for this symbol
        if portfolio.get_position(symbol):
            return

        # Fetch fresh price via REST API (avoids stale cache issues)
        price = await self._fetch_fresh_price(symbol)
        if not price:
            logger.warning(f"[{strategy.name}] Could not get price for {symbol}, skipping entry")
            return

        quantity = portfolio.position_size / price
        # Use current time for entry timestamp, not candle timestamp (bug fix)
        current_time = datetime.now(timezone.utc)

        position = Position(
            symbol=symbol,
            entry_price=price,
            entry_time=current_time,
            quantity=quantity,
            position_size=portfolio.position_size,
        )

        portfolio.open_positions.append(position)
        portfolio.cash -= portfolio.position_size

        # Record to SQLite for visualizer
        if paper_trades:
            try:
                paper_trades.record_entry(
                    strategy=strategy.name,
                    symbol=symbol,
                    entry_price=price,
                    entry_time=current_time,
                    position_size=portfolio.position_size,
                    quantity=quantity
                )
            except Exception as e:
                logger.debug(f"Failed to record entry to SQLite: {e}")

        logger.warning("=" * 60)
        logger.warning(f"📈 [{strategy.name}] ENTRY: {symbol}")
        logger.warning(f"  Price: ${price:.6f}")
        logger.warning(f"  Size: ${portfolio.position_size:.2f} ({quantity:.6f} units)")
        if signal.get('trigger_type') == 'V3_ACCELERATION':
            logger.warning(f"  Signal: V3={signal['v3_prob']:.1%}, vol_d={signal['vol_delta']:.2f}, "
                           f"NATR_d={signal['natr_delta']:.3f}, ret5m={signal['returns_5m']:.2%}")
        else:
            # Legacy NATR-based signal format
            logger.warning(f"  Signal: NATR={signal.get('natr_value', 0):.2f}, "
                           f"{signal.get('orderbook_feature', 'ob')}={signal.get('orderbook_value', 0):.2f}")
        logger.warning(f"  [{strategy.name}] Open: {len(portfolio.open_positions)}/{portfolio.max_positions}")
        logger.warning("=" * 60)

    async def _monitor_positions(self, timestamp: datetime):
        """Monitor open positions for all strategies and apply exit strategy"""
        for strategy in self.strategies:
            await self._monitor_strategy_positions(strategy, timestamp)

    async def _monitor_strategy_positions(self, strategy: TradingStrategy, timestamp: datetime):
        """Monitor open positions for a specific strategy and apply exit strategy"""
        portfolio = strategy.portfolio
        positions_to_close = []
        # Use current time for exit timestamps, not candle timestamp (bug fix)
        current_time = datetime.now(timezone.utc)

        for position in portfolio.open_positions:
            # Fetch fresh price via REST API (critical for accurate exit P&L)
            current_price = await self._fetch_fresh_price(position.symbol)
            if not current_price:
                continue

            # Record price and features for visualizer charting
            if paper_trades:
                try:
                    paper_trades.record_price(position.symbol, current_time, current_price)
                    # Also record current features
                    if position.symbol in self.latest_features:
                        paper_trades.record_features(
                            symbol=position.symbol,
                            timestamp=current_time,
                            features=self.latest_features[position.symbol]
                        )
                    else:
                        logger.debug(f"No features for {position.symbol}, available: {list(self.latest_features.keys())[:5]}")
                except Exception as e:
                    logger.debug(f"Failed to record features: {e}")

            pnl_pct = position.pnl_pct(current_price)
            hold_minutes = (current_time - position.entry_time).total_seconds() / 60

            # Update peak price (all strategies)
            if current_price > position.peak_price:
                position.peak_price = current_price
                # Also update local high after step for step-down strategy
                if position.initial_drawdown_used:
                    position.local_high_after_step = current_price

            # ============================================================
            # SIMPLE STOP LOSS (checked first, before other exit logic)
            # Used by C5_volume for quick exits on momentum failures
            # ============================================================
            if strategy.simple_stop_loss > 0 and pnl_pct <= -strategy.simple_stop_loss:
                position.exit_price = current_price
                position.exit_time = current_time
                position.exit_reason = f"Stop Loss ({strategy.simple_stop_loss*100:.1f}%)"
                position.realized_pnl = (current_price - position.entry_price) * position.remaining_quantity
                portfolio.cash += current_price * position.remaining_quantity
                positions_to_close.append(position)
                self._log_exit(position, current_price, pnl_pct, strategy.name)
                continue

            # ============================================================
            # SIMPLE TAKE PROFIT (checked after stop loss)
            # Used by V5 strategies for clean exits at target
            # ============================================================
            if strategy.simple_take_profit > 0 and pnl_pct >= strategy.simple_take_profit:
                position.exit_price = current_price
                position.exit_time = current_time
                position.exit_reason = f"Take Profit ({strategy.simple_take_profit*100:.0f}%)"
                position.realized_pnl = (current_price - position.entry_price) * position.remaining_quantity
                portfolio.cash += current_price * position.remaining_quantity
                positions_to_close.append(position)
                self._log_exit(position, current_price, pnl_pct, strategy.name)
                continue

            # ============================================================
            # TIERED TRAILING STOP (C4 strategies - lock in gains)
            # Under 5% gain from entry: trail 1% below peak
            # 5%+ gain from entry: trail 2.5% below peak
            # ============================================================
            if strategy.name.startswith("C4"):
                # Calculate gain from entry to peak
                peak_gain_pct = (position.peak_price - position.entry_price) / position.entry_price

                # Only activate trailing stop once we've had at least 1% gain
                if peak_gain_pct >= 0.01:
                    # Determine trail distance based on peak gain
                    if peak_gain_pct >= 0.05:
                        trail_pct = 0.025  # 2.5% trail for 5%+ gains
                    else:
                        trail_pct = 0.01   # 1% trail for under 5% gains

                    # Calculate trailing stop price (trail from peak, not current)
                    trailing_stop = position.peak_price * (1 - trail_pct)

                    # Exit if current price falls below trailing stop
                    if current_price <= trailing_stop:
                        locked_gain_pct = (trailing_stop - position.entry_price) / position.entry_price
                        position.exit_price = current_price
                        position.exit_time = current_time
                        position.exit_reason = f"Trailing Stop ({trail_pct*100:.1f}% from peak {peak_gain_pct*100:.1f}%)"
                        position.realized_pnl = (current_price - position.entry_price) * position.remaining_quantity
                        portfolio.cash += current_price * position.remaining_quantity
                        positions_to_close.append(position)
                        self._log_exit(position, current_price, pnl_pct, strategy.name)
                        continue

            # ============================================================
            # EXIT STRATEGY: STEP-DOWN (Strategy C variants)
            # ============================================================
            if strategy.exit_strategy == "step_down":
                # Calculate drawdown from peak
                drawdown_from_peak = (position.peak_price - current_price) / position.peak_price

                # C3: OB-aware bail - exit early if OB collapses while in drawdown
                if strategy.ob_bail_threshold > 0 and drawdown_from_peak > 0.01:
                    ob_features = self._get_order_book_features(position.symbol)
                    depth_ratio = ob_features.get('order_book_depth_ratio', 1.0)
                    if depth_ratio < strategy.ob_bail_threshold:
                        position.exit_price = current_price
                        position.exit_time = current_time
                        position.exit_reason = f"OB Bail (OB={depth_ratio:.2f}<{strategy.ob_bail_threshold})"
                        position.realized_pnl = (current_price - position.entry_price) * position.remaining_quantity
                        portfolio.cash += current_price * position.remaining_quantity
                        positions_to_close.append(position)
                        self._log_exit(position, current_price, pnl_pct, strategy.name)
                        continue

                # C2: Probation period - tighter tolerance in first N minutes
                in_probation = False
                if strategy.probation_minutes > 0:
                    in_probation = hold_minutes < strategy.probation_minutes
                    if in_probation and drawdown_from_peak >= strategy.probation_tolerance:
                        position.exit_price = current_price
                        position.exit_time = current_time
                        position.exit_reason = f"Probation Exit ({drawdown_from_peak:.1%} > {strategy.probation_tolerance:.0%})"
                        position.realized_pnl = (current_price - position.entry_price) * position.remaining_quantity
                        portfolio.cash += current_price * position.remaining_quantity
                        positions_to_close.append(position)
                        self._log_exit(position, current_price, pnl_pct, strategy.name)
                        continue

                if not position.initial_drawdown_used:
                    # First drawdown logic - 5% tolerance (or skip if in probation)
                    if drawdown_from_peak >= 0.05:
                        position.initial_drawdown_used = True
                        position.step_floor = current_price
                        position.local_high_after_step = current_price
                        logger.info(f"[{strategy.name}] STEP FLOOR SET: {position.symbol} floor=${current_price:.6f} (5% drawdown from peak)")
                        # Don't exit yet - just set the floor
                else:
                    # After first step - check if below floor
                    if current_price < position.step_floor:
                        # Before exiting, check order book for support (shakeout detection)
                        ob_features = self._get_order_book_features(position.symbol)
                        depth_ratio = ob_features.get('order_book_depth_ratio', 0)

                        # Failsafe: always exit if down more than 12% from entry
                        if pnl_pct <= -0.12:
                            position.exit_price = current_price
                            position.exit_time = current_time
                            position.exit_reason = f"Step Down Max Loss (-12%)"
                            position.realized_pnl = (current_price - position.entry_price) * position.remaining_quantity
                            portfolio.cash += current_price * position.remaining_quantity
                            positions_to_close.append(position)
                            self._log_exit(position, current_price, pnl_pct, strategy.name)
                            continue

                        # If strong bid support (depth_ratio > 0.8), hold through the dip
                        if depth_ratio > 0.8:
                            logger.info(f"[{strategy.name}] SHAKEOUT HOLD: {position.symbol} below floor but order_book_depth_ratio={depth_ratio:.2f} (strong bids) - holding")
                            # Don't exit - order book shows support
                        else:
                            # EXIT - broke through floor with weak order book
                            position.exit_price = current_price
                            position.exit_time = current_time
                            position.exit_reason = f"Step Down Exit (OB={depth_ratio:.2f})"
                            position.realized_pnl = (current_price - position.entry_price) * position.remaining_quantity
                            portfolio.cash += current_price * position.remaining_quantity
                            positions_to_close.append(position)
                            self._log_exit(position, current_price, pnl_pct, strategy.name)
                            continue

                    # Check for new step (>2% drop from local high after step)
                    if position.local_high_after_step > 0:
                        drawdown_from_local = (position.local_high_after_step - current_price) / position.local_high_after_step
                        if drawdown_from_local >= 0.02 and current_price > position.step_floor:
                            # New step - update floor to current lower price
                            old_floor = position.step_floor
                            position.step_floor = current_price
                            position.local_high_after_step = current_price
                            logger.info(f"[{strategy.name}] STEP FLOOR LOWERED: {position.symbol} ${old_floor:.6f} -> ${current_price:.6f} (2% step down)")

            # ============================================================
            # EXIT STRATEGY: CURRENT (Strategy B & D)
            # ============================================================
            else:
                # 1. STOP LOSS (-3%) - but skip if V3 > 85% (model says hold)
                #    Also skip if within stop_delay_minutes (let position settle)
                if pnl_pct <= -0.03:
                    v3_score = self._v3_cache.get(position.symbol, 0)

                    # Failsafe: always exit if down more than 8% (even with high V3 or delay)
                    if pnl_pct <= -0.08:
                        position.exit_price = current_price
                        position.exit_time = current_time
                        position.exit_reason = f"Max Loss (-8%) V3={v3_score:.0%}"
                        position.realized_pnl = (current_price - position.entry_price) * position.remaining_quantity
                        portfolio.cash += current_price * position.remaining_quantity
                        positions_to_close.append(position)
                        self._log_exit(position, current_price, pnl_pct, strategy.name)
                        continue

                    # Skip stop loss if within delay period (let initial volatility settle)
                    if strategy.stop_delay_minutes > 0 and hold_minutes < strategy.stop_delay_minutes:
                        logger.info(f"[{strategy.name}] STOP DELAY: {position.symbol} at {pnl_pct:.1%} but only {hold_minutes:.1f}min (delay={strategy.stop_delay_minutes}min) - holding")
                    # Skip stop loss if V3 still high (model says this is going up)
                    elif v3_score > 0.85:
                        logger.info(f"[{strategy.name}] STOP SKIP: {position.symbol} at {pnl_pct:.1%} but V3={v3_score:.1%} - holding")
                    else:
                        position.exit_price = current_price
                        position.exit_time = current_time
                        position.exit_reason = f"Stop Loss (-3%) V3={v3_score:.0%}"
                        position.realized_pnl = (current_price - position.entry_price) * position.remaining_quantity
                        portfolio.cash += current_price * position.remaining_quantity
                        positions_to_close.append(position)
                        self._log_exit(position, current_price, pnl_pct, strategy.name)
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
                        portfolio.cash += exit_value

                        logger.warning(f"📊 [{strategy.name}] PARTIAL {label}: {position.symbol} - Sold {exit_portion*100:.0f}% at ${current_price:.6f}")

                # 3. TRAILING STOP (activate at +15%, trail by 6.5%)
                if pnl_pct >= 0.15 and not position.trailing_stop_active:
                    position.trailing_stop_active = True
                    position.trailing_stop_price = current_price * 0.935
                    logger.info(f"[{strategy.name}] TRAILING STOP ACTIVATED: {position.symbol} at ${position.trailing_stop_price:.6f}")

                if position.trailing_stop_active:
                    new_trailing = current_price * 0.935
                    if new_trailing > position.trailing_stop_price:
                        position.trailing_stop_price = new_trailing

                    if current_price <= position.trailing_stop_price:
                        position.exit_price = current_price
                        position.exit_time = current_time
                        position.exit_reason = "Trailing Stop"
                        position.realized_pnl += (current_price - position.entry_price) * position.remaining_quantity
                        portfolio.cash += current_price * position.remaining_quantity
                        positions_to_close.append(position)
                        self._log_exit(position, current_price, pnl_pct, strategy.name)
                        continue

            # 4. MAX HOLD TIME - uses strategy.max_hold_minutes (default 180, C4 uses 1440)
            if hold_minutes >= strategy.max_hold_minutes:
                v3_score = self._v3_cache.get(position.symbol, 0)
                position.exit_price = current_price
                position.exit_time = current_time
                position.exit_reason = f"Max Hold Time ({hold_minutes:.0f}min) V3={v3_score:.0%}"
                position.realized_pnl += (current_price - position.entry_price) * position.remaining_quantity
                portfolio.cash += current_price * position.remaining_quantity
                positions_to_close.append(position)
                self._log_exit(position, current_price, pnl_pct, strategy.name)
                continue

        # Move closed positions
        for position in positions_to_close:
            portfolio.open_positions.remove(position)
            portfolio.closed_positions.append(position)

            # Record to SQLite for visualizer
            if paper_trades:
                try:
                    paper_trades.record_exit(
                        strategy=strategy.name,
                        symbol=position.symbol,
                        exit_price=position.exit_price,
                        exit_time=position.exit_time,
                        exit_reason=position.exit_reason,
                        realized_pnl=position.realized_pnl
                    )
                except Exception as e:
                    logger.debug(f"Failed to record exit to SQLite: {e}")

        # Save state immediately after any exits
        if positions_to_close:
            await self._save_paper_trading_state()

    def _log_exit(self, position: Position, current_price: float, pnl_pct: float, strategy_name: str = ""):
        """Log position exit"""
        prefix = f"[{strategy_name}] " if strategy_name else ""
        logger.warning("=" * 60)
        logger.warning(f"📉 {prefix}EXIT: {position.symbol} - {position.exit_reason}")
        logger.warning(f"  Entry: ${position.entry_price:.6f}")
        logger.warning(f"  Exit: ${current_price:.6f}")
        logger.warning(f"  P&L: {pnl_pct*100:+.2f}% (${position.realized_pnl:+.2f})")
        logger.warning("=" * 60)

    async def _log_paper_trading_status(self, timestamp: datetime):
        """Log paper trading status periodically and save state to file"""
        # Log every 60 seconds
        if self._last_status_log is None:
            self._last_status_log = timestamp

        elapsed = (timestamp - self._last_status_log).total_seconds()
        if elapsed < self._status_log_interval:
            return

        self._last_status_log = timestamp

        # Calculate predictions per minute
        predictions_this_minute = self.stats['predictions_made'] - self._predictions_at_last_log
        self._predictions_at_last_log = self.stats['predictions_made']

        logger.info("-" * 60)
        logger.info(f"📊 MULTI-STRATEGY STATUS | {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Predictions/min: {predictions_this_minute} (total: {self.stats['predictions_made']})")

        # Summary line for all strategies
        summary_parts = []
        for strat in self.strategies:
            pnl = strat.portfolio.total_pnl
            pnl_pct = strat.portfolio.total_pnl_pct
            open_ct = len(strat.portfolio.open_positions)
            summary_parts.append(f"[{strat.name}] ${pnl:+,.0f} ({pnl_pct:+.1f}%) {open_ct}pos")
        logger.info(f"  {' | '.join(summary_parts)}")

        # Detailed per-strategy status
        for strat in self.strategies:
            portfolio = strat.portfolio
            open_ct = len(portfolio.open_positions)
            closed_ct = len(portfolio.closed_positions)

            if open_ct > 0 or closed_ct > 0:
                logger.info(f"  [{strat.name}] Cash: ${portfolio.cash:,.0f} | Open: {open_ct}/{portfolio.max_positions} | Closed: {closed_ct}")

                # Log open positions for this strategy
                for pos in portfolio.open_positions:
                    price = self.get_current_price(pos.symbol)
                    if price:
                        pnl = pos.pnl_pct(price)
                        hold_min = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 60
                        logger.info(f"    {pos.symbol}: {pnl*100:+.2f}% | {hold_min:.0f}min")

        # Save state to file for external monitor
        await self._save_paper_trading_state()

    def _load_paper_trading_state(self):
        """Load paper trading state from JSON file on startup"""
        state_file = Path('paper_trading_state.json')
        if not state_file.exists():
            logger.info("No existing paper trading state found, starting fresh")
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            strategies_state = state.get('strategies', {})

            for strat in self.strategies:
                if strat.name not in strategies_state:
                    continue

                saved = strategies_state[strat.name]
                portfolio_data = saved.get('portfolio', {})

                # Restore portfolio cash (total_pnl/total_pnl_pct are computed from closed_positions)
                strat.portfolio.cash = portfolio_data.get('cash', strat.portfolio.starting_capital)

                # Restore closed positions
                for pos_data in portfolio_data.get('positions', []):
                    try:
                        entry_time = datetime.fromisoformat(pos_data['entry_time']) if pos_data.get('entry_time') else datetime.now(timezone.utc)
                        exit_time = datetime.fromisoformat(pos_data['exit_time']) if pos_data.get('exit_time') else None

                        pos = Position(
                            symbol=pos_data['symbol'],
                            entry_price=pos_data['entry_price'],
                            entry_time=entry_time,
                            quantity=pos_data['quantity'],
                            position_size=pos_data['position_size'],
                            exit_price=pos_data.get('exit_price', 0),
                            exit_time=exit_time,
                            exit_reason=pos_data.get('exit_reason', ''),
                            realized_pnl=pos_data.get('realized_pnl', 0),
                            partial_exits=pos_data.get('partial_exits', []),
                        )
                        strat.portfolio.closed_positions.append(pos)
                    except Exception as e:
                        logger.warning(f"Failed to restore closed position: {e}")

                # Restore open positions
                for pos_data in saved.get('open_positions', []):
                    try:
                        entry_time = datetime.fromisoformat(pos_data['entry_time']) if pos_data.get('entry_time') else datetime.now(timezone.utc)

                        pos = Position(
                            symbol=pos_data['symbol'],
                            entry_price=pos_data['entry_price'],
                            entry_time=entry_time,
                            quantity=pos_data['quantity'],
                            position_size=pos_data['position_size'],
                            partial_exits=pos_data.get('partial_exits', []),
                        )
                        # Override __post_init__ defaults with saved values
                        pos.remaining_quantity = pos_data.get('remaining_quantity', pos.quantity)
                        pos.peak_price = pos_data.get('peak_price', pos.entry_price)
                        pos.trailing_stop_active = pos_data.get('trailing_stop_active', False)
                        pos.trailing_stop_price = pos_data.get('trailing_stop_price', 0)
                        # Step-down exit strategy fields
                        pos.step_floor = pos_data.get('step_floor', 0)
                        pos.initial_drawdown_used = pos_data.get('initial_drawdown_used', False)
                        pos.local_high_after_step = pos_data.get('local_high_after_step', 0)
                        strat.portfolio.open_positions.append(pos)
                        logger.info(f"  Restored open position: {pos.symbol} @ ${pos.entry_price:.6f}")
                    except Exception as e:
                        logger.warning(f"Failed to restore open position: {e}")

            logger.info(f"Loaded paper trading state from {state_file}")

        except Exception as e:
            logger.error(f"Error loading paper trading state: {e}")
            logger.info("Starting with fresh portfolios")

    async def _save_paper_trading_state(self):
        """Save paper trading state to JSON file for external monitor"""
        try:
            # Build state for all strategies
            strategies_state = {}
            for strat in self.strategies:
                strategies_state[strat.name] = {
                    'portfolio': strat.portfolio.to_dict(),
                    'open_positions': [p.to_dict() for p in strat.portfolio.open_positions],
                    'require_natr': strat.require_natr,
                    'require_ret5m': strat.require_ret5m,
                    'exit_strategy': strat.exit_strategy,
                    'entry_type': strat.entry_type,
                }

            state = {
                'strategies': strategies_state,
                'last_update': datetime.now().isoformat(),
            }

            state_file = Path('paper_trading_state.json')
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving paper trading state: {e}")

    def get_paper_trading_results(self) -> dict:
        """Get paper trading results as dictionary"""
        strategies_results = {}
        for strat in self.strategies:
            strategies_results[strat.name] = {
                'portfolio': strat.portfolio.to_dict(),
                'open_positions': [p.to_dict() for p in strat.portfolio.open_positions],
                'require_natr': strat.require_natr,
                'require_ret5m': strat.require_ret5m,
                'exit_strategy': strat.exit_strategy,
                'entry_type': strat.entry_type,
            }
        return {
            'strategies': strategies_results,
            'normalization_features': len(self.norm_stats),
        }


if __name__ == "__main__":
    # Test feature engine
    from ..data_ingestion.database import DuckDBManager
    import numpy as np

    async def test():
        # Create database manager
        db = DuckDBManager("data/test_features.duckdb")

        # Create feature engine
        engine = FeatureEngine(
            db_manager=db,
            symbols=["BTC-USD", "ETH-USD"],
            primary_timeframe="5m",
            timeframes=["1m", "5m"],
            lookback_bars=100
        )

        # Start database batch writer
        await db.start_batch_writer()

        # Simulate trades
        symbol = "BTC-USD"
        base_price = 45000.0
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        np.random.seed(42)

        print("=== Feature Engine Test ===\n")
        print("Processing 500 trades over 1 hour...\n")

        for i in range(500):
            # Random trade
            price = base_price + np.random.randn() * 50 + (i * 0.1)  # Slight uptrend
            size = np.random.exponential(0.1)
            side = np.random.choice(['BUY', 'SELL'])
            timestamp = base_time + timedelta(seconds=i * 7.2)  # ~7 trades/min

            # Process trade
            await engine.process_trade(symbol, price, size, side, timestamp)

            # Log progress
            if (i + 1) % 100 == 0:
                stats = engine.get_statistics()
                print(f"Progress: {i+1}/500 trades | "
                      f"Candles: {stats['candles_completed']} | "
                      f"Features: {stats['features_calculated']}")

        # Wait for batch to flush
        await asyncio.sleep(6)

        # Print final statistics
        stats = engine.get_statistics()
        print(f"\n=== Final Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Check latest features
        print(f"\n=== Latest Features (5m) ===")
        features_5m = engine.get_latest_features("BTC-USD", "5m")
        if features_5m is not None:
            print(features_5m)

        # Stop database writer
        await db.stop_batch_writer()

        # Query features from database
        features_df = db.get_features("BTC-USD", timeframe="5m", limit=10)
        print(f"\n=== Features in Database (last 10) ===")
        print(features_df)

    asyncio.run(test())
