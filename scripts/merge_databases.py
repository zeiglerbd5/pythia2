#!/usr/bin/env python3
"""
Merge archived DuckDB databases into the current one.

Usage:
    python scripts/merge_databases.py /Volumes/LaCie/Pythia_Archives/pythia_march2026.duckdb
    python scripts/merge_databases.py --list  # Show what would be merged
"""

import argparse
import duckdb
from pathlib import Path
from datetime import datetime


def get_table_info(conn, table_name: str) -> dict:
    """Get row count and date range for a table."""
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        # Try to get date range if timestamp column exists
        try:
            min_max = conn.execute(f"""
                SELECT MIN(timestamp), MAX(timestamp)
                FROM {table_name}
            """).fetchone()
            return {
                'count': count,
                'min_date': min_max[0],
                'max_date': min_max[1]
            }
        except:
            return {'count': count, 'min_date': None, 'max_date': None}
    except Exception as e:
        return {'count': 0, 'error': str(e)}


def list_archive_contents(archive_path: str):
    """Show what's in the archive database."""
    print(f"\n{'='*60}")
    print(f"ARCHIVE: {archive_path}")
    print(f"{'='*60}")

    conn = duckdb.connect(archive_path, read_only=True)
    tables = conn.execute("SHOW TABLES").fetchall()

    for (table_name,) in tables:
        info = get_table_info(conn, table_name)
        if 'error' in info:
            print(f"  {table_name}: ERROR - {info['error']}")
        elif info['min_date']:
            print(f"  {table_name}: {info['count']:,} rows ({info['min_date']} to {info['max_date']})")
        else:
            print(f"  {table_name}: {info['count']:,} rows")

    conn.close()


def merge_databases(current_db: str, archive_db: str, dry_run: bool = False):
    """
    Merge archive database into current database.

    Args:
        current_db: Path to current/active database
        archive_db: Path to archived database to merge in
        dry_run: If True, just show what would happen
    """
    print(f"\n{'='*60}")
    print(f"MERGE OPERATION")
    print(f"{'='*60}")
    print(f"Current DB: {current_db}")
    print(f"Archive DB: {archive_db}")
    print(f"Dry run: {dry_run}")
    print()

    # Connect to current database
    conn = duckdb.connect(current_db)

    # Attach archive
    conn.execute(f"ATTACH '{archive_db}' AS archive (READ_ONLY)")

    # Get tables from both
    current_tables = set(t[0] for t in conn.execute("SHOW TABLES").fetchall())
    archive_tables = set(t[0] for t in conn.execute("SHOW ALL TABLES").fetchall()
                        if t[1] == 'archive')

    # Actually get archive tables properly
    archive_tables = conn.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_catalog = 'archive'
    """).fetchall()
    archive_tables = set(t[0] for t in archive_tables)

    print(f"Current DB tables: {current_tables}")
    print(f"Archive DB tables: {archive_tables}")
    print()

    # Tables to merge (exist in both)
    tables_to_merge = ['ohlcv', 'order_book_snapshots', 'trades', 'features',
                       'whale_transactions', 'news_signals']

    for table in tables_to_merge:
        if table not in archive_tables:
            print(f"  {table}: Not in archive, skipping")
            continue

        # Get counts
        archive_info = get_table_info(conn, f"archive.{table}")

        if table in current_tables:
            current_info = get_table_info(conn, table)
            print(f"  {table}:")
            print(f"    Archive: {archive_info['count']:,} rows")
            print(f"    Current: {current_info['count']:,} rows")

            if not dry_run:
                # Insert avoiding duplicates (using timestamp + symbol as key)
                try:
                    if table in ['ohlcv', 'order_book_snapshots', 'trades']:
                        # These have timestamp + symbol
                        conn.execute(f"""
                            INSERT INTO {table}
                            SELECT a.* FROM archive.{table} a
                            WHERE NOT EXISTS (
                                SELECT 1 FROM {table} c
                                WHERE c.timestamp = a.timestamp
                                AND c.symbol = a.symbol
                            )
                        """)
                    else:
                        # Just insert all (may have duplicates)
                        conn.execute(f"""
                            INSERT INTO {table}
                            SELECT * FROM archive.{table}
                        """)

                    new_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    added = new_count - current_info['count']
                    print(f"    Added: {added:,} rows (new total: {new_count:,})")
                except Exception as e:
                    print(f"    ERROR: {e}")
        else:
            print(f"  {table}: Creating table from archive ({archive_info['count']:,} rows)")
            if not dry_run:
                try:
                    conn.execute(f"CREATE TABLE {table} AS SELECT * FROM archive.{table}")
                    print(f"    Created successfully")
                except Exception as e:
                    print(f"    ERROR: {e}")

    conn.execute("DETACH archive")
    conn.close()

    print()
    print("Merge complete!" if not dry_run else "Dry run complete (no changes made)")


def main():
    parser = argparse.ArgumentParser(description="Merge DuckDB databases")
    parser.add_argument("archive", nargs="?", help="Path to archive database to merge")
    parser.add_argument("--current", default="/Users/bz/Pythia2/pythia_march2026.duckdb",
                       help="Path to current database")
    parser.add_argument("--list", action="store_true", help="List archive contents only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be merged")
    args = parser.parse_args()

    if args.list and args.archive:
        list_archive_contents(args.archive)
    elif args.archive:
        merge_databases(args.current, args.archive, dry_run=args.dry_run)
    else:
        # List all archives
        archive_dir = Path("/Volumes/LaCie/Pythia_Archives")
        if archive_dir.exists():
            print("Available archives:")
            for f in archive_dir.glob("*.duckdb"):
                size = f.stat().st_size / (1024**3)
                print(f"  {f.name}: {size:.1f} GB")
        else:
            print("No archives found. Usage: python merge_databases.py <archive.duckdb>")


if __name__ == "__main__":
    main()
