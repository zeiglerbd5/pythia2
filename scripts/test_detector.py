#!/usr/bin/env python3
"""
Test the spike detector on historical data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timedelta
import duckdb
import numpy as np
import joblib

# Load model
model = joblib.load('/Users/brettzeigler/Pythia/models/xgboost_slow_large_v1.pkl')
print(f"✓ Model loaded")

# Feature columns (must match training order)
feature_cols = [
    'returns', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14',
    'NATR', 'BB_width', 'BB_squeeze', 'VWAP_distance', 'volume_zscore',
    'volume_roc', 'OBV', 'trade_count', 'buy_sell_ratio', 'roll_measure',
    'order_flow_imbalance', 'vpin', 'bid_ask_spread_pct',
    'order_book_depth_ratio', 'large_order_imbalance', 'returns_5m',
    'volume_zscore_5m', 'returns_15m', 'volume_zscore_15m',
]

# Query database for latest candle per symbol
conn = duckdb.connect('/Users/brettzeigler/Pythia/market_data.duckdb', read_only=True)

query = f"""
    WITH ranked AS (
        SELECT
            symbol,
            timestamp,
            {', '.join(feature_cols)},
            ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
        FROM features
        WHERE timeframe = '1m'
    )
    SELECT symbol, timestamp, {', '.join(feature_cols)}
    FROM ranked
    WHERE rn = 1
"""

df = conn.execute(query).fetchdf()
conn.close()

print(f"✓ Loaded {len(df)} symbols")

# Run predictions
X = df[feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

probs = model.predict_proba(X)[:, 1]
df['spike_probability'] = probs

# Sort by probability
df = df.sort_values('spike_probability', ascending=False)

# Show top 20
print()
print("=" * 60)
print("TOP 20 SPIKE OPPORTUNITIES (Most Recent Candles)")
print("=" * 60)
print(f"{'Symbol':<15} {'Timestamp':<20} {'Prob':>8} {'NATR':>8}")
print("-" * 60)

for _, row in df.head(20).iterrows():
    symbol = row['symbol']
    timestamp = row['timestamp']
    prob = row['spike_probability']
    natr = row.get('NATR', 0)

    flag = "🚀" if prob >= 0.5 else "  "
    print(f"{flag} {symbol:<12} {str(timestamp):<20} {prob:>7.1%} {natr:>8.3f}")

print()
print(f"Symbols above 50% threshold: {(df['spike_probability'] >= 0.5).sum()}/{len(df)}")
print(f"Symbols above 30% threshold: {(df['spike_probability'] >= 0.3).sum()}/{len(df)}")
print()
print("=" * 60)
