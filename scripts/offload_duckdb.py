#!/usr/bin/env python3
"""
Offload old DuckDB data to LaCie external drive.

Keeps the last 24h in the live database, archives everything older to
a date-stamped DuckDB file on the LaCie. Designed to run via cron daily.

Usage:
    python scripts/offload_duckdb.py                   # dry run
    python scripts/offload_duckdb.py --execute          # actually offload
    python scripts/offload_duckdb.py --execute --days 3 # keep 3 days instead of 1
"""

import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import duckdb

LIVE_DB = "data/pythia.duckdb"
LACIE_DIR = "/Volumes/LaCie/Pythia_Archives"
TABLES_TO_OFFLOAD = [
    "trades",              # Biggest table by far
    "tickers",             # Second biggest
    "order_book_snapshots", # Third biggest (has JSON blobs)
    "ohlcv",
    "features",
    "news_signals",
    "whale_transactions",
]


def check_lacie():
    """Check LaCie is mounted."""
    if not Path(LACIE_DIR).exists():
        print("ERROR: LaCie not mounted at /Volumes/LaCie")
        print("Mount the drive and try again.")
        sys.exit(1)

    free = shutil.disk_usage(LACIE_DIR).free / (1024**3)
    print(f"LaCie free space: {free:.1f} GB")
    return free


def check_live_db():
    """Check live DB size and date ranges."""
    db_path = Path(LIVE_DB)
    if not db_path.exists():
        print(f"ERROR: {LIVE_DB} not found")
        sys.exit(1)

    size_gb = db_path.stat().st_size / (1024**3)
    print(f"Live DB size: {size_gb:.2f} GB")

    db = duckdb.connect(LIVE_DB, read_only=True)
    for table in TABLES_TO_OFFLOAD:
        try:
            r = db.execute(f"SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM {table}").fetchone()
            print(f"  {table:25s}: {r[0]:>12,} rows | {r[1]} → {r[2]}")
        except Exception as e:
            print(f"  {table:25s}: {e}")
    db.close()
    return size_gb


def offload(keep_days: int, execute: bool):
    """Offload old data to LaCie."""
    cutoff = datetime.now() - timedelta(days=keep_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    date_stamp = datetime.now().strftime("%Y%m%d")

    archive_dir = Path(LACIE_DIR) / "auto_offloads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"pythia_offload_{date_stamp}.duckdb"

    print(f"\nCutoff: {cutoff_str} (keeping last {keep_days} day(s))")
    print(f"Archive: {archive_path}")

    if archive_path.exists():
        print(f"Archive {archive_path} already exists — will append to it")

    # Open both databases
    live = duckdb.connect(LIVE_DB)
    archive = duckdb.connect(str(archive_path))

    total_moved = 0

    for table in TABLES_TO_OFFLOAD:
        try:
            # Count rows to offload
            count = live.execute(
                f"SELECT COUNT(*) FROM {table} WHERE timestamp < ?", [cutoff_str]
            ).fetchone()[0]

            if count == 0:
                print(f"  {table}: nothing to offload")
                continue

            print(f"  {table}: {count:,} rows to offload", end="")

            if not execute:
                print(" (dry run)")
                continue

            # Create table in archive if needed (copy schema)
            try:
                archive.execute(f"SELECT 1 FROM {table} LIMIT 0")
            except:
                # Table doesn't exist in archive — create it from live schema
                cols = live.execute(f"DESCRIBE {table}").fetchall()
                col_defs = ", ".join(f"{c[0]} {c[1]}" for c in cols)
                archive.execute(f"CREATE TABLE IF NOT EXISTS {table} ({col_defs})")

            # Copy old rows to archive
            live.execute(f"ATTACH '{archive_path}' AS archive")
            live.execute(f"""
                INSERT INTO archive.{table}
                SELECT * FROM {table}
                WHERE timestamp < ?
            """, [cutoff_str])
            live.execute("DETACH archive")

            # Delete from live
            live.execute(f"DELETE FROM {table} WHERE timestamp < ?", [cutoff_str])

            total_moved += count
            print(f" → moved")

        except Exception as e:
            print(f"  {table}: ERROR - {e}")

    archive.close()

    if execute and total_moved > 0:
        # Checkpoint and compact the live database
        print("\nCompacting live database...")
        live.execute("CHECKPOINT")
        print("Done.")

    live.close()

    # Check resulting sizes
    if execute:
        live_size = Path(LIVE_DB).stat().st_size / (1024**3)
        archive_size = archive_path.stat().st_size / (1024**3) if archive_path.exists() else 0
        disk_free = shutil.disk_usage("/").free / (1024**3)
        print(f"\nResults:")
        print(f"  Live DB:    {live_size:.2f} GB")
        print(f"  Archive:    {archive_size:.2f} GB")
        print(f"  Disk free:  {disk_free:.1f} GB")
        print(f"  Rows moved: {total_moved:,}")
    else:
        print(f"\nDRY RUN — would move {total_moved:,} rows. Run with --execute to proceed.")


def main():
    parser = argparse.ArgumentParser(description="Offload old DuckDB data to LaCie")
    parser.add_argument("--execute", action="store_true", help="Actually perform the offload")
    parser.add_argument("--days", type=int, default=1, help="Days of data to keep in live DB (default: 1)")
    args = parser.parse_args()

    print("=" * 60)
    print("  DuckDB Auto-Offload")
    print("=" * 60)

    check_lacie()
    check_live_db()
    offload(keep_days=args.days, execute=args.execute)


if __name__ == "__main__":
    main()
