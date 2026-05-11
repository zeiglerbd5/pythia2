"""
Paper Trades SQLite Database

Lightweight database for paper trading data that supports concurrent access.
The collector writes trade events and price snapshots here.
The visualizer reads from here.
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
import threading

DB_PATH = Path(__file__).parent / "paper_trades.db"

# Thread-local storage for connections
_local = threading.local()


def get_connection() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    if not hasattr(_local, 'conn') or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), timeout=30)
        _local.conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent access
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=30000")
    return _local.conn


def init_db():
    """Initialize the database schema."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            entry_price REAL NOT NULL,
            entry_time TEXT NOT NULL,
            exit_price REAL,
            exit_time TEXT,
            exit_reason TEXT,
            position_size REAL NOT NULL,
            quantity REAL NOT NULL,
            realized_pnl REAL DEFAULT 0,
            is_open INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
        CREATE INDEX IF NOT EXISTS idx_trades_is_open ON trades(is_open);
        CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);

        CREATE TABLE IF NOT EXISTS price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            price REAL NOT NULL,
            UNIQUE(symbol, timestamp)
        );

        CREATE INDEX IF NOT EXISTS idx_prices_symbol_time ON price_snapshots(symbol, timestamp);

        CREATE TABLE IF NOT EXISTS feature_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            features TEXT NOT NULL,  -- JSON blob of feature values
            FOREIGN KEY (trade_id) REFERENCES trades(id)
        );

        CREATE INDEX IF NOT EXISTS idx_features_trade ON feature_snapshots(trade_id);
        CREATE INDEX IF NOT EXISTS idx_features_symbol_time ON feature_snapshots(symbol, timestamp);
    """)
    conn.commit()


def record_entry(strategy: str, symbol: str, entry_price: float, entry_time: datetime,
                 position_size: float, quantity: float) -> int:
    """Record a new trade entry. Returns the trade ID."""
    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO trades (strategy, symbol, entry_price, entry_time, position_size, quantity)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (strategy, symbol, entry_price, entry_time.isoformat(), position_size, quantity))
    conn.commit()
    return cursor.lastrowid


def record_exit(strategy: str, symbol: str, exit_price: float, exit_time: datetime,
                exit_reason: str, realized_pnl: float):
    """Record a trade exit."""
    conn = get_connection()
    conn.execute("""
        UPDATE trades
        SET exit_price = ?, exit_time = ?, exit_reason = ?, realized_pnl = ?,
            is_open = 0, updated_at = CURRENT_TIMESTAMP
        WHERE strategy = ? AND symbol = ? AND is_open = 1
    """, (exit_price, exit_time.isoformat(), exit_reason, realized_pnl, strategy, symbol))
    conn.commit()


def record_price(symbol: str, timestamp: datetime, price: float):
    """Record a price snapshot for charting."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO price_snapshots (symbol, timestamp, price)
            VALUES (?, ?, ?)
        """, (symbol, timestamp.strftime('%Y-%m-%d %H:%M:00'), price))
        conn.commit()
    except Exception:
        pass  # Ignore duplicate key errors


def record_features(symbol: str, timestamp: datetime, features: Dict, trade_id: int = None):
    """Record feature snapshot for visualization."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO feature_snapshots (trade_id, symbol, timestamp, features)
            VALUES (?, ?, ?, ?)
        """, (trade_id, symbol, timestamp.isoformat(), json.dumps(features)))
        conn.commit()
    except Exception as e:
        pass  # Non-critical


def get_feature_data(symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
    """Get feature snapshots for charting."""
    conn = get_connection()
    cursor = conn.execute("""
        SELECT timestamp, features
        FROM feature_snapshots
        WHERE symbol = ?
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
    """, (symbol, start_time.isoformat(), end_time.isoformat()))

    results = []
    for row in cursor.fetchall():
        try:
            features = json.loads(row['features'])
            features['timestamp'] = row['timestamp']
            results.append(features)
        except:
            pass
    return results


def get_trade_features(trade_id: int) -> List[Dict]:
    """Get all feature snapshots for a specific trade."""
    conn = get_connection()
    cursor = conn.execute("""
        SELECT timestamp, features
        FROM feature_snapshots
        WHERE trade_id = ?
        ORDER BY timestamp
    """, (trade_id,))

    results = []
    for row in cursor.fetchall():
        try:
            features = json.loads(row['features'])
            features['timestamp'] = row['timestamp']
            results.append(features)
        except:
            pass
    return results


def get_all_trades() -> List[Dict]:
    """Get all trades for the visualizer."""
    conn = get_connection()
    cursor = conn.execute("""
        SELECT * FROM trades ORDER BY entry_time DESC
    """)
    return [dict(row) for row in cursor.fetchall()]


def get_price_data(symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
    """Get price snapshots for charting."""
    conn = get_connection()
    cursor = conn.execute("""
        SELECT timestamp, price as close
        FROM price_snapshots
        WHERE symbol = ?
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
    """, (symbol, start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')))
    return [dict(row) for row in cursor.fetchall()]


def get_open_trades() -> List[Dict]:
    """Get all open trades."""
    conn = get_connection()
    cursor = conn.execute("""
        SELECT * FROM trades WHERE is_open = 1 ORDER BY entry_time DESC
    """)
    return [dict(row) for row in cursor.fetchall()]


def get_closed_trades() -> List[Dict]:
    """Get all closed trades."""
    conn = get_connection()
    cursor = conn.execute("""
        SELECT * FROM trades WHERE is_open = 0 ORDER BY exit_time DESC
    """)
    return [dict(row) for row in cursor.fetchall()]


def get_trades_by_strategy(strategy: str) -> List[Dict]:
    """Get trades for a specific strategy."""
    conn = get_connection()
    cursor = conn.execute("""
        SELECT * FROM trades WHERE strategy = ? ORDER BY entry_time DESC
    """, (strategy,))
    return [dict(row) for row in cursor.fetchall()]


def close_orphaned_positions(active_symbols: set) -> int:
    """Close all open DB records not in the active positions set.

    Called on startup to reconcile SQLite with in-memory state.
    Orphaned positions get $0 PnL since we can't determine actual exit price.
    Returns count of closed records.
    """
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()

    if not active_symbols:
        # No active positions — close everything
        cursor = conn.execute("""
            UPDATE trades
            SET exit_reason = 'orphaned_restart', realized_pnl = 0,
                is_open = 0, exit_time = ?, updated_at = CURRENT_TIMESTAMP
            WHERE is_open = 1
        """, (now,))
    else:
        # Close only records whose symbol is NOT in active set
        placeholders = ','.join('?' for _ in active_symbols)
        cursor = conn.execute(f"""
            UPDATE trades
            SET exit_reason = 'orphaned_restart', realized_pnl = 0,
                is_open = 0, exit_time = ?, updated_at = CURRENT_TIMESTAMP
            WHERE is_open = 1 AND symbol NOT IN ({placeholders})
        """, (now, *active_symbols))

    conn.commit()
    return cursor.rowcount


def get_stats() -> Dict:
    """Get summary statistics."""
    conn = get_connection()

    stats = {}

    # Overall stats
    cursor = conn.execute("""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN is_open = 1 THEN 1 ELSE 0 END) as open_trades,
            SUM(CASE WHEN is_open = 0 THEN 1 ELSE 0 END) as closed_trades,
            SUM(CASE WHEN is_open = 0 AND realized_pnl > 0 THEN 1 ELSE 0 END) as winners,
            SUM(CASE WHEN is_open = 0 AND realized_pnl <= 0 THEN 1 ELSE 0 END) as losers,
            SUM(CASE WHEN is_open = 0 THEN realized_pnl ELSE 0 END) as total_pnl
        FROM trades
    """)
    row = cursor.fetchone()
    stats['overall'] = dict(row) if row else {}

    # Per-strategy stats
    cursor = conn.execute("""
        SELECT
            strategy,
            COUNT(*) as total_trades,
            SUM(CASE WHEN is_open = 1 THEN 1 ELSE 0 END) as open_trades,
            SUM(CASE WHEN is_open = 0 THEN 1 ELSE 0 END) as closed_trades,
            SUM(CASE WHEN is_open = 0 AND realized_pnl > 0 THEN 1 ELSE 0 END) as winners,
            SUM(CASE WHEN is_open = 0 AND realized_pnl <= 0 THEN 1 ELSE 0 END) as losers,
            SUM(CASE WHEN is_open = 0 THEN realized_pnl ELSE 0 END) as total_pnl
        FROM trades
        GROUP BY strategy
    """)
    stats['by_strategy'] = {row['strategy']: dict(row) for row in cursor.fetchall()}

    return stats


# Initialize on import
init_db()


if __name__ == "__main__":
    # Test
    init_db()
    print(f"Database initialized at {DB_PATH}")
    print(f"Stats: {get_stats()}")
