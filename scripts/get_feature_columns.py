#!/usr/bin/env python3
"""
Quick script to get exact feature column order from database.
This matches what train_xgboost_minimal.py uses.
"""

import duckdb

DB_PATH = '/Users/brettzeigler/Pythia/market_data.duckdb'

conn = duckdb.connect(DB_PATH, read_only=True)

# Get column names from features table
schema = conn.execute("DESCRIBE features").fetchall()
all_columns = [row[0] for row in schema]

# Feature columns (exclude metadata) - same as train_xgboost_minimal.py
exclude_cols = ['symbol', 'timestamp', 'timeframe', 'open', 'high', 'low', 'close',
                'volume', 'is_spike', 'spike_return_1m', 'spike_return_3m',
                'spike_return_5m', 'spike_return_10m']
feature_cols = [c for c in all_columns if c not in exclude_cols]

conn.close()

print(f"Feature columns ({len(feature_cols)}):")
print("=" * 60)
for i, col in enumerate(feature_cols, 1):
    print(f"{i:2d}. {col}")

print("\n" + "=" * 60)
print("Python list format:")
print("=" * 60)
print("feature_cols = [")
for col in feature_cols:
    print(f"    '{col}',")
print("]")
