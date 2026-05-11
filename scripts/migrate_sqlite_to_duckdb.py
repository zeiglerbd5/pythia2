#!/usr/bin/env python3
"""
Migrate SQLite Database to DuckDB

Migrates WebSocket-collected data from SQLite to DuckDB format.

Usage:
    python scripts/migrate_sqlite_to_duckdb.py --input data.db --output pythia.duckdb
"""

import sqlite3
import duckdb
import argparse
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def migrate_table(
    sqlite_conn: sqlite3.Connection,
    duckdb_conn: duckdb.DuckDBPyConnection,
    table_name: str,
    batch_size: int = 10000
):
    """
    Migrate a table from SQLite to DuckDB.

    Args:
        sqlite_conn: SQLite connection
        duckdb_conn: DuckDB connection
        table_name: Table to migrate
        batch_size: Records per batch
    """
    logger.info(f"Migrating table: {table_name}")

    # Count total rows
    cursor = sqlite_conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = cursor.fetchone()[0]

    logger.info(f"Total rows to migrate: {total_rows:,}")

    if total_rows == 0:
        logger.warning(f"Table {table_name} is empty, skipping")
        return

    # Get column names
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
    columns = [desc[0] for desc in cursor.description]
    column_names = ', '.join(columns)

    # Use DuckDB's INSERT ... FROM statement with SQLite extension
    # This is much faster than manual batch inserts
    logger.info("Attaching SQLite database to DuckDB...")

    try:
        # Attach the SQLite database
        sqlite_path = sqlite_conn.execute("PRAGMA database_list").fetchone()[2]
        duckdb_conn.execute(f"INSTALL sqlite")
        duckdb_conn.execute(f"LOAD sqlite")
        duckdb_conn.execute(f"ATTACH '{sqlite_path}' AS sqlite_db (TYPE sqlite)")

        # Create table with SELECT to infer schema and copy data
        logger.info(f"Creating table and copying data...")
        duckdb_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM sqlite_db.{table_name}")

        migrated = total_rows
        logger.success(f"✓ Migrated {migrated:,} rows for table {table_name}")

    except Exception as e:
        logger.warning(f"Fast migration failed: {e}")
        logger.info("Falling back to batch migration...")

        # Fallback: manual batch migration
        # First, fetch a sample row to create table schema
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
        sample_row = cursor.fetchone()

        if not sample_row:
            logger.warning(f"Could not fetch sample row from {table_name}")
            return

        # Create table using sample data for schema inference
        import pandas as pd
        sample_df = pd.DataFrame([sample_row], columns=columns)
        duckdb_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM sample_df")
        duckdb_conn.execute(f"DELETE FROM {table_name}")  # Remove sample row

        # Migrate data in batches
        offset = 0
        migrated = 0

        while offset < total_rows:
            # Fetch batch from SQLite
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}")
            rows = cursor.fetchall()

            if not rows:
                break

            # Insert into DuckDB
            placeholders = ','.join(['?' for _ in columns])
            insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"

            try:
                duckdb_conn.executemany(insert_sql, rows)
                migrated += len(rows)

                if migrated % 100000 == 0:
                    logger.info(f"Migrated {migrated:,} / {total_rows:,} rows ({migrated/total_rows*100:.1f}%)")

            except Exception as e:
                logger.error(f"Error inserting batch at offset {offset}: {e}")
                break

            offset += batch_size

        logger.success(f"✓ Migrated {migrated:,} rows for table {table_name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Migrate SQLite to DuckDB')
    parser.add_argument('--input', required=True, help='Input SQLite database path')
    parser.add_argument('--output', required=True, help='Output DuckDB database path')
    parser.add_argument('--tables', help='Comma-separated tables to migrate (default: all)')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input database not found: {input_path}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("SQLITE → DUCKDB MIGRATION")
    logger.info("=" * 80)
    logger.info(f"Input:  {input_path} ({input_path.stat().st_size / 1e9:.2f} GB)")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 80)

    # Connect to databases
    sqlite_conn = sqlite3.connect(str(input_path))
    duckdb_conn = duckdb.connect(str(output_path))

    # Attach SQLite database once for all tables (faster)
    logger.info("Attaching SQLite database to DuckDB...")
    try:
        duckdb_conn.execute(f"INSTALL sqlite")
        duckdb_conn.execute(f"LOAD sqlite")
        duckdb_conn.execute(f"ATTACH '{str(input_path)}' AS sqlite_db (TYPE sqlite)")
        use_fast_method = True
        logger.info("✓ SQLite database attached successfully")
    except Exception as e:
        logger.warning(f"Could not attach SQLite database: {e}")
        logger.info("Will use slower batch migration method")
        use_fast_method = False

    # Get list of tables
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    all_tables = [row[0] for row in cursor.fetchall()]

    if args.tables:
        tables_to_migrate = [t.strip() for t in args.tables.split(',')]
    else:
        tables_to_migrate = all_tables

    logger.info(f"Tables to migrate: {tables_to_migrate}")
    logger.info("")

    # Migrate each table
    for table in tables_to_migrate:
        if table in all_tables:
            # Use fast method if SQLite is attached
            if use_fast_method:
                try:
                    cursor = sqlite_conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    total_rows = cursor.fetchone()[0]

                    if total_rows == 0:
                        logger.warning(f"Table {table} is empty, skipping")
                        continue

                    logger.info(f"Migrating table: {table} ({total_rows:,} rows)")
                    logger.info(f"  Using fast method (DuckDB sqlite extension)...")

                    duckdb_conn.execute(f"CREATE TABLE {table} AS SELECT * FROM sqlite_db.{table}")

                    logger.success(f"✓ Migrated {total_rows:,} rows for table {table}")
                except Exception as e:
                    logger.warning(f"Fast migration failed for {table}: {e}")
                    logger.info("Falling back to batch migration for this table")
                    migrate_table(sqlite_conn, duckdb_conn, table, args.batch_size)
            else:
                migrate_table(sqlite_conn, duckdb_conn, table, args.batch_size)
        else:
            logger.warning(f"Table {table} not found in source database")

    # Close connections
    sqlite_conn.close()
    duckdb_conn.close()

    logger.info("")
    logger.info("=" * 80)
    logger.success("MIGRATION COMPLETE")
    logger.info("=" * 80)

    # Show output file size
    if output_path.exists():
        size_gb = output_path.stat().st_size / 1e9
        logger.info(f"Output database: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
