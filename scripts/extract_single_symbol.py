#!/usr/bin/env python3
"""
Extract single symbol OHLCV data from DuckDB to parquet.
"""
import duckdb
import os

def main():
    # Paths
    source_db = 'data/pythia_snapshot.duckdb'
    output_dir = 'data/nn_training'
    symbol = 'DASH-USD'
    output_file = os.path.join(output_dir, 'dash_ohlcv.parquet')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Connect to source
    conn = duckdb.connect(source_db, read_only=True)

    # Check source data
    count = conn.execute(f"SELECT COUNT(*) FROM ohlcv WHERE symbol = '{symbol}'").fetchone()[0]
    print(f"Source: {count:,} rows for {symbol}")

    # Extract to parquet
    conn.execute(f'''
        COPY (
            SELECT * FROM ohlcv
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
        ) TO '{output_file}' (FORMAT PARQUET)
    ''')

    print(f"Extracted to: {output_file}")

    # Verify
    verify_conn = duckdb.connect(':memory:')
    verify_count = verify_conn.execute(f"SELECT COUNT(*) FROM '{output_file}'").fetchone()[0]
    print(f"Verified: {verify_count:,} rows in output file")

    # Show sample
    print("\nSample data:")
    sample = verify_conn.execute(f"SELECT * FROM '{output_file}' LIMIT 5").fetchdf()
    print(sample)

if __name__ == '__main__':
    main()
