#!/usr/bin/env python3
"""
Analyze gaps in candle data to understand data quality.

Quick analysis of missing minutes in the time series.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from datetime import timedelta

def analyze_gaps(db_path: str, symbol: str = None):
    """
    Analyze gaps in candle data.

    Args:
        db_path: Path to DuckDB database
        symbol: Specific symbol to analyze (None = summary of all)
    """
    conn = duckdb.connect(db_path, read_only=True)

    if symbol:
        # Detailed gap analysis for one symbol
        query = f"""
        WITH time_diffs AS (
            SELECT
                symbol,
                timestamp,
                LEAD(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as next_timestamp,
                DATEDIFF('minute', timestamp, LEAD(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp)) as gap_minutes
            FROM candles
            WHERE symbol = '{symbol}'
        )
        SELECT
            gap_minutes,
            COUNT(*) as num_gaps,
            COUNT(*) * gap_minutes as total_missing_minutes
        FROM time_diffs
        WHERE gap_minutes > 1
        GROUP BY gap_minutes
        ORDER BY gap_minutes
        """

        results = conn.execute(query).fetchall()

        print(f"\n=== Gap Analysis for {symbol} ===")
        print(f"{'Gap Size (min)':<15} {'Count':<10} {'Missing Minutes':<20}")
        print("-" * 50)

        total_gaps = 0
        total_missing = 0

        for gap_size, count, missing in results:
            print(f"{gap_size:<15} {count:<10} {missing:<20}")
            total_gaps += count
            total_missing += missing

        print("-" * 50)
        print(f"Total gaps: {total_gaps}")
        print(f"Total missing minutes: {total_missing}")

        # Get actual candle count
        candle_count = conn.execute(f"SELECT COUNT(*) FROM candles WHERE symbol = '{symbol}'").fetchone()[0]
        print(f"Actual candles: {candle_count}")
        print(f"Would have after filling: {candle_count + total_missing}")

    else:
        # Summary across all symbols
        query = """
        WITH time_diffs AS (
            SELECT
                symbol,
                timestamp,
                LEAD(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as next_timestamp,
                DATEDIFF('minute', timestamp, LEAD(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp)) as gap_minutes
            FROM candles
        ),
        gap_stats AS (
            SELECT
                symbol,
                COUNT(CASE WHEN gap_minutes > 1 THEN 1 END) as num_gaps,
                SUM(CASE WHEN gap_minutes > 1 THEN gap_minutes - 1 ELSE 0 END) as missing_minutes,
                MAX(gap_minutes) as max_gap_minutes
            FROM time_diffs
            GROUP BY symbol
        )
        SELECT
            symbol,
            num_gaps,
            missing_minutes,
            max_gap_minutes
        FROM gap_stats
        WHERE num_gaps > 0
        ORDER BY missing_minutes DESC
        LIMIT 20
        """

        results = conn.execute(query).fetchall()

        print("\n=== Top 20 Symbols by Missing Data ===")
        print(f"{'Symbol':<15} {'Gaps':<10} {'Missing Min':<15} {'Max Gap (min)':<15}")
        print("-" * 60)

        for symbol, gaps, missing, max_gap in results:
            print(f"{symbol:<15} {gaps:<10} {missing:<15} {max_gap:<15}")

    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze gaps in candle data')
    parser.add_argument('--db', default='data/pythia.duckdb', help='Path to DuckDB database')
    parser.add_argument('--symbol', help='Specific symbol to analyze (optional)')

    args = parser.parse_args()

    analyze_gaps(args.db, args.symbol)
