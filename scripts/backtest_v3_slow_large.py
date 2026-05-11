#!/usr/bin/env python3
"""
Backtest V3 (Slow & Large) vs V1/V2 on Dec 7-8 Data

Specifically checks:
- LOKA-USD: +30% over 1 hour starting ~19:00 UTC Dec 7
- LRDS-USD: +40% over 18 hours starting ~11:30 Dec 7

Questions to answer:
1. Does V3 catch these multi-hour runners?
2. Does V3 fire BEFORE V1/V2 (earlier warning)?
3. What's the hit rate on V3's longer-horizon predictions?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
import joblib
from datetime import datetime, timedelta


def main():
    logger.info("=" * 80)
    logger.info("BACKTEST V3 (SLOW & LARGE) vs V1/V2")
    logger.info("=" * 80)
    logger.info("")

    # === CONFIGURATION ===
    DB_PATH = 'market_data.duckdb'
    V1_MODEL_PATH = 'models/xgboost_slow_large_v1.pkl'
    V2_MODEL_PATH = 'models/xgboost_slow_large_v2.pkl'
    V3_MODEL_PATH = 'models/xgboost_slow_large_v3_model.pkl'

    # Time range (from backtest CSV)
    START_TIME = '2025-12-07 20:00:00'
    END_TIME = '2025-12-08 15:00:00'

    logger.info(f"Time range: {START_TIME} to {END_TIME}")
    logger.info("")

    # === STEP 1: Load models ===
    logger.info("Loading models...")
    v1_model = joblib.load(V1_MODEL_PATH)
    v2_model = joblib.load(V2_MODEL_PATH)
    v3_model = joblib.load(V3_MODEL_PATH)
    logger.info(f"✓ V1 model loaded (Fast spike - descriptive)")
    logger.info(f"✓ V2 model loaded (Fast spike - 2min predictive)")
    logger.info(f"✓ V3 model loaded (Slow & Large - 15min predictive)")
    logger.info("")

    # Feature columns
    feature_cols = [
        'returns', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14', 'NATR',
        'BB_width', 'BB_squeeze', 'VWAP_distance', 'volume_zscore', 'volume_roc',
        'OBV', 'trade_count', 'buy_sell_ratio', 'roll_measure', 'order_flow_imbalance',
        'vpin', 'bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance',
        'returns_5m', 'volume_zscore_5m', 'returns_15m', 'volume_zscore_15m'
    ]

    # === STEP 2: Load features ===
    logger.info("Loading features from database...")
    conn = duckdb.connect(DB_PATH, read_only=True)

    query = f"""
        SELECT
            timestamp,
            symbol,
            {', '.join(feature_cols)}
        FROM features
        WHERE timeframe = '1m'
        AND timestamp >= '{START_TIME}'
        AND timestamp <= '{END_TIME}'
        ORDER BY timestamp, symbol
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    if df.empty:
        logger.error("No features found!")
        return

    logger.info(f"Loaded {len(df):,} feature rows")
    logger.info(f"Symbols: {df['symbol'].nunique()}")
    logger.info("")

    # === STEP 3: Run predictions ===
    logger.info("Running predictions with all three models...")

    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    v1_probs = v1_model.predict_proba(X)[:, 1]
    v2_probs = v2_model.predict_proba(X)[:, 1]
    v3_probs = v3_model.predict_proba(X)[:, 1]

    df['v1_prob'] = v1_probs
    df['v2_prob'] = v2_probs
    df['v3_prob'] = v3_probs

    logger.info("✓ Predictions complete")
    logger.info("")

    # === STEP 4: Overall statistics ===
    logger.info("=" * 80)
    logger.info("OVERALL STATISTICS")
    logger.info("=" * 80)
    logger.info("")

    for name, probs in [('V1 (Fast-Descriptive)', v1_probs),
                        ('V2 (Fast-Predictive)', v2_probs),
                        ('V3 (Slow & Large)', v3_probs)]:
        logger.info(f"{name}:")
        logger.info(f"  Mean: {probs.mean()*100:.2f}%")
        logger.info(f"  Max:  {probs.max()*100:.2f}%")
        logger.info(f"  >50%: {(probs > 0.5).sum():,}")
        logger.info(f"  >70%: {(probs > 0.7).sum():,}")
        logger.info(f"  >80%: {(probs > 0.8).sum():,}")
        logger.info("")

    # === STEP 5: Focus on LOKA-USD ===
    logger.info("=" * 80)
    logger.info("LOKA-USD ANALYSIS (30% spike starting ~19:00 Dec 7)")
    logger.info("=" * 80)
    logger.info("")

    loka = df[df['symbol'] == 'LOKA-USD'].copy()
    loka = loka.sort_values('timestamp')

    if len(loka) > 0:
        logger.info(f"LOKA-USD data points: {len(loka)}")
        logger.info(f"Time range: {loka['timestamp'].min()} to {loka['timestamp'].max()}")
        logger.info("")

        # Show predictions during the spike period
        logger.info("Predictions around spike time:")
        logger.info(f"{'Timestamp':<20} {'V1%':>7} {'V2%':>7} {'V3%':>7}  Notes")
        logger.info("-" * 70)

        for _, row in loka.head(60).iterrows():  # First hour
            ts = str(row['timestamp'])
            v1 = row['v1_prob'] * 100
            v2 = row['v2_prob'] * 100
            v3 = row['v3_prob'] * 100

            notes = ""
            if v3 >= 70:
                notes += "🟢 V3 ALERT "
            if v1 >= 70:
                notes += "🔵 V1 "
            if v2 >= 70:
                notes += "🟡 V2 "

            if v1 >= 50 or v2 >= 50 or v3 >= 50:
                logger.info(f"{ts:<20} {v1:6.1f}% {v2:6.1f}% {v3:6.1f}%  {notes}")
    else:
        logger.warning("No LOKA-USD data found in this time range")

    logger.info("")

    # === STEP 6: Focus on LRDS-USD ===
    logger.info("=" * 80)
    logger.info("LRDS-USD ANALYSIS (40% spike over 18 hours)")
    logger.info("=" * 80)
    logger.info("")

    lrds = df[df['symbol'] == 'LRDS-USD'].copy()
    lrds = lrds.sort_values('timestamp')

    if len(lrds) > 0:
        logger.info(f"LRDS-USD data points: {len(lrds)}")

        # Find high V3 predictions
        high_v3 = lrds[lrds['v3_prob'] >= 0.5]
        if len(high_v3) > 0:
            logger.info(f"V3 signals >= 50%: {len(high_v3)}")
            logger.info("")
            logger.info("High V3 signals:")
            for _, row in high_v3.iterrows():
                logger.info(f"  {row['timestamp']} - V3={row['v3_prob']*100:.1f}% V1={row['v1_prob']*100:.1f}%")
        else:
            logger.info("No V3 signals >= 50% for LRDS-USD")
    else:
        logger.warning("No LRDS-USD data found")

    logger.info("")

    # === STEP 7: V3-unique signals (V3 high, V1/V2 low) ===
    logger.info("=" * 80)
    logger.info("V3-UNIQUE SIGNALS (V3 >= 70%, V1 < 50%, V2 < 50%)")
    logger.info("=" * 80)
    logger.info("")

    v3_unique = df[(df['v3_prob'] >= 0.7) &
                   (df['v1_prob'] < 0.5) &
                   (df['v2_prob'] < 0.5)].copy()

    if len(v3_unique) > 0:
        logger.info(f"Found {len(v3_unique)} V3-unique signals")
        logger.info("")

        # Group by symbol
        v3_unique_by_symbol = v3_unique.groupby('symbol').agg({
            'v3_prob': ['count', 'max'],
            'timestamp': 'first'
        }).reset_index()
        v3_unique_by_symbol.columns = ['symbol', 'count', 'max_prob', 'first_signal']
        v3_unique_by_symbol = v3_unique_by_symbol.sort_values('max_prob', ascending=False)

        logger.info("Top 15 symbols with V3-unique signals:")
        logger.info(f"{'Symbol':<15} {'Count':>6} {'Max V3%':>8} {'First Signal':<20}")
        logger.info("-" * 55)

        for _, row in v3_unique_by_symbol.head(15).iterrows():
            logger.info(f"{row['symbol']:<15} {row['count']:>6} {row['max_prob']*100:>7.1f}% {str(row['first_signal']):<20}")
    else:
        logger.info("No V3-unique signals found")

    logger.info("")

    # === STEP 8: Top V3 signals overall ===
    logger.info("=" * 80)
    logger.info("TOP 20 V3 SIGNALS (Highest probability)")
    logger.info("=" * 80)
    logger.info("")

    top_v3 = df.nlargest(20, 'v3_prob')

    logger.info(f"{'Symbol':<15} {'Timestamp':<20} {'V3%':>7} {'V1%':>7} {'V2%':>7}")
    logger.info("-" * 65)

    for _, row in top_v3.iterrows():
        logger.info(
            f"{row['symbol']:<15} {str(row['timestamp']):<20} "
            f"{row['v3_prob']*100:6.1f}% {row['v1_prob']*100:6.1f}% {row['v2_prob']*100:6.1f}%"
        )

    logger.info("")

    # === STEP 9: Save results ===
    output_file = 'backtest_v3_results.csv'
    df_output = df[['timestamp', 'symbol', 'v1_prob', 'v2_prob', 'v3_prob']].copy()
    df_output.to_csv(output_file, index=False)
    logger.info(f"✓ Results saved to {output_file}")

    # === STEP 10: Summary ===
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    v1_alerts = (v1_probs >= 0.7).sum()
    v2_alerts = (v2_probs >= 0.7).sum()
    v3_alerts = (v3_probs >= 0.7).sum()

    logger.info(f"Total alerts (>=70% threshold):")
    logger.info(f"  V1 (Fast-Descriptive): {v1_alerts:,}")
    logger.info(f"  V2 (Fast-Predictive):  {v2_alerts:,}")
    logger.info(f"  V3 (Slow & Large):     {v3_alerts:,}")
    logger.info("")

    # Check overlap
    both_v1_v3 = ((v1_probs >= 0.7) & (v3_probs >= 0.7)).sum()
    only_v3 = ((v1_probs < 0.5) & (v2_probs < 0.5) & (v3_probs >= 0.7)).sum()

    logger.info(f"V3 signals that V1 also caught: {both_v1_v3}")
    logger.info(f"V3-only signals (V1<50%, V2<50%): {only_v3}")
    logger.info("")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
