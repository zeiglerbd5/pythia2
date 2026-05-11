#!/usr/bin/env python3
"""
Forward-fill gaps in candle data using efficient DuckDB SQL.

Creates candles_filled table with:
- Complete minute-by-minute data (no gaps)
- Forward-filled OHLCV (last known price)
- Volume=0 during gaps
- no_activity boolean flag

Strategy:
1. Generate complete time range per symbol
2. LEFT JOIN existing candles
3. Use LAST_VALUE() window function for forward-fill
4. Fast SQL execution (not Python loops)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from loguru import logger


def forward_fill_candles(db_path: str, max_gap_minutes: int = 1440):
    """
    Forward-fill gaps in candle data using DuckDB SQL.

    Args:
        db_path: Path to DuckDB database
        max_gap_minutes: Maximum gap to fill (default: 1 day)
    """
    logger.info(f"Forward-filling candles in {db_path}")
    logger.info(f"Max gap to fill: {max_gap_minutes} minutes")

    conn = duckdb.connect(db_path, read_only=False)

    # Step 1: Get date range per symbol
    logger.info("Step 1/4: Analyzing date ranges per symbol...")

    date_ranges = conn.execute("""
        SELECT
            symbol,
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time,
            COUNT(*) as actual_candles,
            DATEDIFF('minute', MIN(timestamp), MAX(timestamp)) + 1 as expected_candles
        FROM candles
        GROUP BY symbol
    """).fetchall()

    logger.info(f"Found {len(date_ranges)} symbols to process")

    # Step 2: Create filled candles table
    logger.info("Step 2/4: Creating candles_filled table...")

    conn.execute("DROP TABLE IF EXISTS candles_filled")
    conn.execute("""
        CREATE TABLE candles_filled (
            symbol VARCHAR,
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            no_activity BOOLEAN
        )
    """)

    # Step 3: Fill each symbol
    logger.info("Step 3/4: Forward-filling each symbol...")

    total_added = 0

    for idx, (symbol, start_time, end_time, actual, expected) in enumerate(date_ranges):
        missing = expected - actual

        if idx % 10 == 0:
            logger.info(f"  Processing {idx+1}/{len(date_ranges)}: {symbol} (adding {missing} candles)")

        # Generate complete time range and forward-fill
        query = f"""
        WITH RECURSIVE
        -- Generate complete minute range for this symbol
        time_range AS (
            SELECT
                '{symbol}' as symbol,
                TIMESTAMP '{start_time}' as timestamp
            UNION ALL
            SELECT
                '{symbol}',
                timestamp + INTERVAL '1' MINUTE
            FROM time_range
            WHERE timestamp < TIMESTAMP '{end_time}'
        ),
        -- Join with actual candles
        filled AS (
            SELECT
                t.symbol,
                t.timestamp,
                c.open,
                c.high,
                c.low,
                c.close,
                COALESCE(c.volume, 0.0) as volume,
                c.close IS NULL as is_gap
            FROM time_range t
            LEFT JOIN candles c
                ON t.symbol = c.symbol
                AND t.timestamp = c.timestamp
        ),
        -- Forward fill using window functions
        forward_filled AS (
            SELECT
                symbol,
                timestamp,
                LAST_VALUE(close IGNORE NULLS) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) as filled_close,
                volume,
                is_gap
            FROM filled
        )
        SELECT
            symbol,
            timestamp,
            filled_close as open,   -- Use last close as OHLC during gaps
            filled_close as high,
            filled_close as low,
            filled_close as close,
            volume,
            is_gap as no_activity
        FROM forward_filled
        WHERE filled_close IS NOT NULL  -- Skip nulls at start (no data to fill from)
        """

        conn.execute(f"INSERT INTO candles_filled {query}")
        total_added += missing

    # Step 4: Verify results
    logger.info("Step 4/4: Verifying results...")

    filled_stats = conn.execute("""
        SELECT
            COUNT(DISTINCT symbol) as num_symbols,
            COUNT(*) as total_candles,
            SUM(CASE WHEN no_activity THEN 1 ELSE 0 END) as filled_candles,
            SUM(CASE WHEN NOT no_activity THEN 1 ELSE 0 END) as original_candles
        FROM candles_filled
    """).fetchone()

    num_symbols, total, filled, original = filled_stats

    logger.success(f"✓ Forward-filling complete!")
    logger.info(f"  Symbols processed: {num_symbols}")
    logger.info(f"  Original candles: {original:,}")
    logger.info(f"  Filled gaps: {filled:,}")
    logger.info(f"  Total candles: {total:,}")
    logger.info(f"  Data completeness: {(original/total*100):.2f}% → 100%")

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Forward-fill gaps in candle data')
    parser.add_argument('--db', default='data/pythia.duckdb', help='Path to DuckDB database')
    parser.add_argument('--max-gap', type=int, default=1440, help='Maximum gap to fill (minutes)')

    args = parser.parse_args()

    forward_fill_candles(args.db, args.max_gap)
