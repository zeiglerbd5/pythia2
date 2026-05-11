#!/usr/bin/env python3
"""
Check Data Readiness for Model Training

Verifies that you have sufficient data to train models.

Requirements:
- 81+ days of continuous data
- Features calculated
- OHLCV candles available
- Minimum sample count

Usage:
    python scripts/check_data_readiness.py --db pythia.duckdb
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from loguru import logger
import pandas as pd


def check_data_readiness(db_path: str, symbols: list = None):
    """
    Check if database has enough data for training.

    Args:
        db_path: Path to DuckDB database
        symbols: List of symbols to check (None = all)
    """
    logger.info("=" * 80)
    logger.info("DATA READINESS CHECK")
    logger.info("=" * 80)

    db_path = Path(db_path)

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Run integrated_collector.py to start collecting data")
        return False

    logger.info(f"Database: {db_path}")
    logger.info(f"Size: {db_path.stat().st_size / 1e9:.2f} GB")
    logger.info("")

    # Connect to database
    conn = duckdb.connect(str(db_path), read_only=True)

    try:
        # Check tables exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        table_names = [t[0] for t in tables]

        logger.info("Available tables:")
        for table in table_names:
            # Count rows
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"  {table}: {count:,} rows")

        logger.info("")

        # Check required tables
        required_tables = ['ohlcv', 'features']  # Minimum required
        missing_tables = [t for t in required_tables if t not in table_names]

        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
            logger.warning("Run feature calculation to create missing tables")
            logger.info("")

        # Check OHLCV data coverage
        if 'ohlcv' in table_names:
            logger.info("OHLCV Data Coverage:")

            # Get symbols
            if symbols is None:
                symbols = conn.execute(
                    "SELECT DISTINCT symbol FROM ohlcv"
                ).fetchall()
                symbols = [s[0] for s in symbols]

            for symbol in symbols:
                for timeframe in ['1m', '5m', '15m']:
                    # Get date range
                    result = conn.execute(f"""
                        SELECT
                            MIN(timestamp) as min_date,
                            MAX(timestamp) as max_date,
                            COUNT(*) as count
                        FROM ohlcv
                        WHERE symbol = '{symbol}'
                        AND timeframe = '{timeframe}'
                    """).fetchone()

                    if result and result[0]:
                        min_date = pd.to_datetime(result[0])
                        max_date = pd.to_datetime(result[1])
                        count = result[2]

                        days = (max_date - min_date).days

                        logger.info(f"  {symbol} @ {timeframe}:")
                        logger.info(f"    Range: {min_date.date()} to {max_date.date()}")
                        logger.info(f"    Days: {days}")
                        logger.info(f"    Candles: {count:,}")

                        # Check if enough for training
                        min_days_required = 81  # 60 + 14 + 7

                        if days >= min_days_required:
                            logger.success(f"    ✓ Sufficient data for training ({days} >= {min_days_required} days)")
                        else:
                            days_needed = min_days_required - days
                            logger.warning(f"    ✗ Need {days_needed} more days of data")

                        logger.info("")

        # Check features
        if 'features' in table_names:
            logger.info("Feature Data Coverage:")

            if symbols is None:
                symbols = conn.execute(
                    "SELECT DISTINCT symbol FROM features"
                ).fetchall()
                symbols = [s[0] for s in symbols]

            for symbol in symbols:
                result = conn.execute(f"""
                    SELECT
                        COUNT(*) as count,
                        MIN(timestamp) as min_date,
                        MAX(timestamp) as max_date
                    FROM features
                    WHERE symbol = '{symbol}'
                    AND timeframe = '5m'
                """).fetchone()

                if result and result[0] > 0:
                    count = result[0]
                    min_date = pd.to_datetime(result[1])
                    max_date = pd.to_datetime(result[2])
                    days = (max_date - min_date).days

                    logger.info(f"  {symbol} @ 5m:")
                    logger.info(f"    Features calculated: {count:,}")
                    logger.info(f"    Date range: {min_date.date()} to {max_date.date()}")
                    logger.info(f"    Days: {days}")

                    # Estimate number of sequences
                    sequence_length = 60 * 12 * 24  # 60 days @ 5min = 17,280 candles
                    forward_window = 14 * 12 * 24   # 14 days @ 5min = 4,032 candles

                    if count > sequence_length + forward_window:
                        max_sequences = count - sequence_length - forward_window
                        logger.info(f"    Estimated samples: ~{max_sequences:,}")

                        if max_sequences >= 500:
                            logger.success(f"    ✓ Enough samples for training")
                        else:
                            logger.warning(f"    ✗ Need more data (minimum 500 samples)")
                    else:
                        logger.warning(f"    ✗ Not enough data for sequences")

                    logger.info("")

        # Summary
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)

        # Check if ready to train
        ready_to_train = False

        if 'ohlcv' in table_names and 'features' in table_names:
            # Check if any symbol has enough data
            for symbol in symbols if symbols else ['BTC-USD']:
                result = conn.execute(f"""
                    SELECT COUNT(*) FROM features
                    WHERE symbol = '{symbol}'
                    AND timeframe = '5m'
                """).fetchone()

                if result and result[0] > 0:
                    sequence_length = 60 * 12 * 24
                    forward_window = 14 * 12 * 24

                    if result[0] > sequence_length + forward_window + 500:
                        ready_to_train = True
                        break

        if ready_to_train:
            logger.success("✓ DATA IS READY FOR TRAINING!")
            logger.info("")
            logger.info("Next steps:")
            logger.info("  1. Run: python scripts/train_model.py --symbol BTC-USD")
            logger.info("  2. Wait for training to complete (~1-2 hours)")
            logger.info("  3. Check evaluation metrics")
        else:
            logger.warning("✗ NOT READY FOR TRAINING YET")
            logger.info("")
            logger.info("Options:")
            logger.info("  1. Backfill historical data:")
            logger.info("     python scripts/backfill_historical_data.py --days 90")
            logger.info("  2. Continue collecting (need 81+ days total)")
            logger.info("  3. Calculate features if missing:")
            logger.info("     python src/data_ingestion/integrated_collector.py")

        logger.info("=" * 80)

    finally:
        conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Check data readiness')
    parser.add_argument('--db', default='data/pythia.duckdb', help='Database path')
    parser.add_argument('--symbols', help='Comma-separated symbols to check')

    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]

    check_data_readiness(args.db, symbols)


if __name__ == "__main__":
    main()
