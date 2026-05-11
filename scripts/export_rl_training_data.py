#!/usr/bin/env python3
"""
Export OHLCV and enhanced feature data from DuckDB to SQLite for RL training.

Run this when the collector is stopped to create a portable training dataset.

Exports:
- OHLCV data (1-minute candles)
- Order book features (imbalance, spread, depth)
- Trade flow features (VPIN, roll measure)
- Aggregated trade volumes (buy/sell pressure)
"""

import argparse
import duckdb
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import sys


def export_data(
    duckdb_path: str,
    sqlite_path: str,
    symbols: list = None,
    days: int = 30,
    timeframe: str = "1m",
    include_features: bool = True,
    include_trades: bool = True,
):
    """Export OHLCV and enhanced feature data from DuckDB to SQLite."""

    print(f"Connecting to DuckDB: {duckdb_path}")
    try:
        duck_conn = duckdb.connect(duckdb_path, read_only=True)
    except Exception as e:
        print(f"ERROR: Cannot connect to DuckDB: {e}")
        print("Make sure the collector is stopped before running this script.")
        return False

    # Get available symbols if not specified
    if not symbols:
        result = duck_conn.execute(f"""
            SELECT symbol, COUNT(*) as cnt
            FROM ohlcv
            WHERE timeframe = '{timeframe}'
            GROUP BY symbol
            HAVING cnt > 10000
            ORDER BY cnt DESC
            LIMIT 50
        """).fetchdf()
        symbols = result['symbol'].tolist()
        print(f"Found {len(symbols)} symbols with sufficient data")

    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    # Convert to ISO format strings for SQL
    start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"Time range: {start_str} to {end_str}")
    print(f"Symbols: {len(symbols)}")
    print(f"Include features: {include_features}")
    print(f"Include trades: {include_trades}")

    # Create SQLite database
    sqlite_path = Path(sqlite_path)
    if sqlite_path.exists():
        sqlite_path.unlink()

    sqlite_conn = sqlite3.connect(str(sqlite_path))

    # Create OHLCV table
    sqlite_conn.execute("""
        CREATE TABLE ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            num_trades INTEGER DEFAULT 0,
            UNIQUE(symbol, timestamp)
        )
    """)
    sqlite_conn.execute("CREATE INDEX idx_ohlcv_symbol_timestamp ON ohlcv(symbol, timestamp)")

    # Create features table for order book and trade flow features
    if include_features:
        sqlite_conn.execute("""
            CREATE TABLE features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                timeframe TEXT DEFAULT '5m',
                order_book_imbalance_l5 REAL,
                bid_ask_ratio REAL,
                bid_ask_spread_pct REAL,
                order_book_depth_ratio REAL,
                large_order_imbalance REAL,
                weighted_mid_price REAL,
                vpin REAL,
                roll_measure REAL,
                volume_spike_ratio REAL,
                rsi_14 REAL,
                vwap REAL,
                vwap_distance_pct REAL,
                atr REAL,
                natr REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                bb_width REAL,
                UNIQUE(symbol, timestamp, timeframe)
            )
        """)
        sqlite_conn.execute("CREATE INDEX idx_features_symbol_timestamp ON features(symbol, timestamp)")

    # Create aggregated trades table for buy/sell volumes
    if include_trades:
        sqlite_conn.execute("""
            CREATE TABLE trade_volumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                buy_volume REAL NOT NULL,
                sell_volume REAL NOT NULL,
                buy_count INTEGER DEFAULT 0,
                sell_count INTEGER DEFAULT 0,
                large_buy_volume REAL DEFAULT 0,
                large_sell_volume REAL DEFAULT 0,
                UNIQUE(symbol, timestamp)
            )
        """)
        sqlite_conn.execute("CREATE INDEX idx_trades_symbol_timestamp ON trade_volumes(symbol, timestamp)")

    total_ohlcv_rows = 0
    total_feature_rows = 0
    total_trade_rows = 0

    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")

        # Export OHLCV
        ohlcv_count = _export_ohlcv(duck_conn, sqlite_conn, symbol, timeframe, start_str, end_str)
        total_ohlcv_rows += ohlcv_count
        print(f"  OHLCV: {ohlcv_count:,} rows")

        # Export features
        if include_features:
            feature_count = _export_features(duck_conn, sqlite_conn, symbol, start_str, end_str)
            total_feature_rows += feature_count
            print(f"  Features: {feature_count:,} rows")

        # Export aggregated trades
        if include_trades:
            trade_count = _export_trade_volumes(duck_conn, sqlite_conn, symbol, start_str, end_str)
            total_trade_rows += trade_count
            print(f"  Trade volumes: {trade_count:,} rows")

    # Verify
    print("\n" + "=" * 50)
    print("=== Export Complete ===")
    print("=" * 50)

    cursor = sqlite_conn.execute("SELECT COUNT(*) FROM ohlcv")
    print(f"OHLCV rows: {cursor.fetchone()[0]:,}")

    if include_features:
        cursor = sqlite_conn.execute("SELECT COUNT(*) FROM features")
        print(f"Feature rows: {cursor.fetchone()[0]:,}")

    if include_trades:
        cursor = sqlite_conn.execute("SELECT COUNT(*) FROM trade_volumes")
        print(f"Trade volume rows: {cursor.fetchone()[0]:,}")

    cursor = sqlite_conn.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv")
    print(f"Symbols: {cursor.fetchone()[0]}")

    print(f"Output: {sqlite_path}")
    print(f"Size: {sqlite_path.stat().st_size / 1024 / 1024:.1f} MB")

    sqlite_conn.close()
    duck_conn.close()

    return True


def _export_ohlcv(duck_conn, sqlite_conn, symbol, timeframe, start_str, end_str):
    """Export OHLCV data for a symbol."""
    try:
        df = duck_conn.execute(f"""
            SELECT
                symbol,
                timestamp,
                open, high, low, close,
                volume,
                COALESCE(num_trades, 0) as num_trades
            FROM ohlcv
            WHERE symbol = '{symbol}'
              AND timeframe = '{timeframe}'
              AND timestamp >= '{start_str}'
              AND timestamp <= '{end_str}'
            ORDER BY timestamp
        """).fetchdf()

        if len(df) == 0:
            return 0

        df['timestamp'] = df['timestamp'].astype(str)
        rows = df.values.tolist()

        sqlite_conn.executemany(
            "INSERT INTO ohlcv (symbol, timestamp, open, high, low, close, volume, num_trades) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows
        )
        sqlite_conn.commit()
        return len(df)

    except Exception as e:
        print(f"    OHLCV ERROR: {e}")
        return 0


def _export_features(duck_conn, sqlite_conn, symbol, start_str, end_str):
    """Export order book and trade flow features for a symbol."""
    try:
        df = duck_conn.execute(f"""
            SELECT
                symbol,
                timestamp,
                timeframe,
                order_book_imbalance_l5,
                bid_ask_ratio,
                bid_ask_spread_pct,
                order_book_depth_ratio,
                large_order_imbalance,
                weighted_mid_price,
                vpin,
                roll_measure,
                volume_spike_ratio,
                rsi_14,
                vwap,
                vwap_distance_pct,
                atr,
                natr,
                bb_upper,
                bb_middle,
                bb_lower,
                bb_width
            FROM features
            WHERE symbol = '{symbol}'
              AND timestamp >= '{start_str}'
              AND timestamp <= '{end_str}'
            ORDER BY timestamp
        """).fetchdf()

        if len(df) == 0:
            return 0

        df['timestamp'] = df['timestamp'].astype(str)
        rows = df.values.tolist()

        sqlite_conn.executemany(
            """INSERT INTO features (
                symbol, timestamp, timeframe,
                order_book_imbalance_l5, bid_ask_ratio, bid_ask_spread_pct,
                order_book_depth_ratio, large_order_imbalance, weighted_mid_price,
                vpin, roll_measure, volume_spike_ratio, rsi_14,
                vwap, vwap_distance_pct, atr, natr,
                bb_upper, bb_middle, bb_lower, bb_width
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows
        )
        sqlite_conn.commit()
        return len(df)

    except Exception as e:
        print(f"    Features ERROR: {e}")
        return 0


def _export_trade_volumes(duck_conn, sqlite_conn, symbol, start_str, end_str):
    """
    Export aggregated buy/sell volumes from trades table.

    Aggregates trades to 1-minute buckets with buy/sell classification.
    """
    try:
        # Check if trades table exists and has data
        check = duck_conn.execute(f"""
            SELECT COUNT(*) FROM trades
            WHERE symbol = '{symbol}'
              AND timestamp >= '{start_str}'
              AND timestamp <= '{end_str}'
        """).fetchone()

        if check[0] == 0:
            return 0

        # Aggregate trades to 1-minute buckets
        # Note: trades table has 'side' column for BUY/SELL classification
        df = duck_conn.execute(f"""
            SELECT
                symbol,
                date_trunc('minute', timestamp) as timestamp,
                SUM(CASE WHEN UPPER(side) = 'BUY' THEN size * price ELSE 0 END) as buy_volume,
                SUM(CASE WHEN UPPER(side) = 'SELL' THEN size * price ELSE 0 END) as sell_volume,
                SUM(CASE WHEN UPPER(side) = 'BUY' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN UPPER(side) = 'SELL' THEN 1 ELSE 0 END) as sell_count,
                SUM(CASE WHEN UPPER(side) = 'BUY' AND size * price > 10000 THEN size * price ELSE 0 END) as large_buy_volume,
                SUM(CASE WHEN UPPER(side) = 'SELL' AND size * price > 10000 THEN size * price ELSE 0 END) as large_sell_volume
            FROM trades
            WHERE symbol = '{symbol}'
              AND timestamp >= '{start_str}'
              AND timestamp <= '{end_str}'
            GROUP BY symbol, date_trunc('minute', timestamp)
            ORDER BY timestamp
        """).fetchdf()

        if len(df) == 0:
            return 0

        df['timestamp'] = df['timestamp'].astype(str)
        rows = df.values.tolist()

        sqlite_conn.executemany(
            """INSERT INTO trade_volumes (
                symbol, timestamp, buy_volume, sell_volume,
                buy_count, sell_count, large_buy_volume, large_sell_volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows
        )
        sqlite_conn.commit()
        return len(df)

    except Exception as e:
        print(f"    Trade volumes ERROR: {e}")
        return 0


def generate_mock_data(
    sqlite_path: str,
    symbols: int = 10,
    days: int = 7,
    include_features: bool = True,
    include_trades: bool = True,
):
    """Generate mock OHLCV and feature data for testing when DuckDB is unavailable."""
    import numpy as np

    print(f"Generating mock training data...")
    print(f"  Symbols: {symbols}")
    print(f"  Days: {days}")
    print(f"  Include features: {include_features}")
    print(f"  Include trades: {include_trades}")

    sqlite_path = Path(sqlite_path)
    if sqlite_path.exists():
        sqlite_path.unlink()

    sqlite_conn = sqlite3.connect(str(sqlite_path))

    # Create OHLCV table
    sqlite_conn.execute("""
        CREATE TABLE ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            num_trades INTEGER DEFAULT 0,
            UNIQUE(symbol, timestamp)
        )
    """)
    sqlite_conn.execute("CREATE INDEX idx_ohlcv_symbol_timestamp ON ohlcv(symbol, timestamp)")

    # Create features table
    if include_features:
        sqlite_conn.execute("""
            CREATE TABLE features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                timeframe TEXT DEFAULT '5m',
                order_book_imbalance_l5 REAL,
                bid_ask_ratio REAL,
                bid_ask_spread_pct REAL,
                order_book_depth_ratio REAL,
                large_order_imbalance REAL,
                weighted_mid_price REAL,
                vpin REAL,
                roll_measure REAL,
                volume_spike_ratio REAL,
                UNIQUE(symbol, timestamp, timeframe)
            )
        """)
        sqlite_conn.execute("CREATE INDEX idx_features_symbol_timestamp ON features(symbol, timestamp)")

    # Create trade volumes table
    if include_trades:
        sqlite_conn.execute("""
            CREATE TABLE trade_volumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                buy_volume REAL NOT NULL,
                sell_volume REAL NOT NULL,
                buy_count INTEGER DEFAULT 0,
                sell_count INTEGER DEFAULT 0,
                large_buy_volume REAL DEFAULT 0,
                large_sell_volume REAL DEFAULT 0,
                UNIQUE(symbol, timestamp)
            )
        """)
        sqlite_conn.execute("CREATE INDEX idx_trades_symbol_timestamp ON trade_volumes(symbol, timestamp)")

    # Generate synthetic symbols
    symbol_list = [f"MOCK{i}-USD" for i in range(symbols)]

    # Time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    minutes = int((end_time - start_time).total_seconds() / 60)

    total_ohlcv = 0
    total_features = 0
    total_trades = 0

    for symbol in symbol_list:
        # Random starting price
        np.random.seed(hash(symbol) % 2**32)
        price = np.random.uniform(1, 100)

        ohlcv_rows = []
        feature_rows = []
        trade_rows = []
        current_time = start_time

        for i in range(minutes):
            # Random walk with occasional spikes
            if np.random.random() < 0.01:  # 1% chance of spike
                change = np.random.choice([-1, 1]) * np.random.uniform(0.01, 0.05)
            else:
                change = np.random.normal(0, 0.001)

            price = price * (1 + change)
            price = max(0.01, price)

            # Generate OHLCV
            open_p = price
            high_p = price * (1 + abs(np.random.normal(0, 0.002)))
            low_p = price * (1 - abs(np.random.normal(0, 0.002)))
            close_p = price * (1 + np.random.normal(0, 0.001))
            volume = np.random.exponential(1000) * price
            num_trades = int(np.random.poisson(50))

            ohlcv_rows.append([
                symbol,
                current_time.isoformat(),
                open_p, high_p, low_p, close_p,
                volume, num_trades
            ])

            # Generate trade volumes
            if include_trades:
                buy_ratio = np.clip(0.5 + (close_p - open_p) / price * 10, 0.2, 0.8)
                buy_volume = volume * buy_ratio
                sell_volume = volume * (1 - buy_ratio)
                buy_count = int(num_trades * buy_ratio)
                sell_count = num_trades - buy_count

                trade_rows.append([
                    symbol,
                    current_time.isoformat(),
                    buy_volume, sell_volume,
                    buy_count, sell_count,
                    buy_volume * 0.1 if buy_volume > 5000 else 0,
                    sell_volume * 0.1 if sell_volume > 5000 else 0,
                ])

            # Generate features (every 5 minutes)
            if include_features and i % 5 == 0:
                feature_rows.append([
                    symbol,
                    current_time.isoformat(),
                    '5m',
                    np.random.randn() * 0.3,  # order_book_imbalance_l5
                    0.9 + np.random.randn() * 0.1,  # bid_ask_ratio
                    0.001 + abs(np.random.randn() * 0.0005),  # bid_ask_spread_pct
                    1 + np.random.randn() * 0.2,  # order_book_depth_ratio
                    np.random.randn() * 0.2,  # large_order_imbalance
                    close_p,  # weighted_mid_price
                    np.clip(abs(np.random.randn() * 0.3), 0, 1),  # vpin
                    abs(np.random.randn() * 0.001),  # roll_measure
                    1 + np.random.randn() * 0.5,  # volume_spike_ratio
                ])

            current_time += timedelta(minutes=1)
            price = close_p

        # Insert OHLCV
        sqlite_conn.executemany(
            "INSERT INTO ohlcv (symbol, timestamp, open, high, low, close, volume, num_trades) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ohlcv_rows
        )
        total_ohlcv += len(ohlcv_rows)

        # Insert features
        if include_features and feature_rows:
            sqlite_conn.executemany(
                """INSERT INTO features (
                    symbol, timestamp, timeframe,
                    order_book_imbalance_l5, bid_ask_ratio, bid_ask_spread_pct,
                    order_book_depth_ratio, large_order_imbalance, weighted_mid_price,
                    vpin, roll_measure, volume_spike_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                feature_rows
            )
            total_features += len(feature_rows)

        # Insert trade volumes
        if include_trades and trade_rows:
            sqlite_conn.executemany(
                """INSERT INTO trade_volumes (
                    symbol, timestamp, buy_volume, sell_volume,
                    buy_count, sell_count, large_buy_volume, large_sell_volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                trade_rows
            )
            total_trades += len(trade_rows)

        sqlite_conn.commit()
        print(f"  {symbol}: {len(ohlcv_rows):,} OHLCV, {len(feature_rows):,} features, {len(trade_rows):,} trades")

    print(f"\n=== Mock Data Generated ===")
    print(f"Output: {sqlite_path}")
    print(f"OHLCV rows: {total_ohlcv:,}")
    if include_features:
        print(f"Feature rows: {total_features:,}")
    if include_trades:
        print(f"Trade volume rows: {total_trades:,}")
    print(f"Symbols: {len(symbol_list)}")
    print(f"Size: {sqlite_path.stat().st_size / 1024 / 1024:.1f} MB")

    sqlite_conn.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export RL training data from DuckDB to SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export 30 days of data with all features
  python export_rl_training_data.py --days 30

  # Export OHLCV only (faster, smaller file)
  python export_rl_training_data.py --no-features --no-trades

  # Generate mock data for testing
  python export_rl_training_data.py --mock --mock-symbols 5 --mock-days 3

  # Export specific symbols
  python export_rl_training_data.py --symbols BTC-USD,ETH-USD,SOL-USD
"""
    )
    parser.add_argument(
        "--duckdb",
        default="/Users/bz/Pythia2/full_pythia.duckdb",
        help="Path to DuckDB database"
    )
    parser.add_argument(
        "--output",
        default="/Users/bz/Pythia2/rl_training_data.db",
        help="Output SQLite path"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to export (default: 30)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols to export (default: auto-select top 50)"
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Skip exporting order book features"
    )
    parser.add_argument(
        "--no-trades",
        action="store_true",
        help="Skip exporting aggregated trade volumes"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate mock data instead (for testing)"
    )
    parser.add_argument(
        "--mock-symbols",
        type=int,
        default=10,
        help="Number of mock symbols to generate (default: 10)"
    )
    parser.add_argument(
        "--mock-days",
        type=int,
        default=7,
        help="Number of days of mock data (default: 7)"
    )

    args = parser.parse_args()

    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]

    if args.mock:
        success = generate_mock_data(
            args.output,
            symbols=args.mock_symbols,
            days=args.mock_days,
            include_features=not args.no_features,
            include_trades=not args.no_trades,
        )
    else:
        success = export_data(
            args.duckdb,
            args.output,
            symbols=symbols,
            days=args.days,
            include_features=not args.no_features,
            include_trades=not args.no_trades,
        )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
