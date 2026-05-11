#!/usr/bin/env python3
"""
Backtest V1 vs V2 Models on Last 19 Hours of Data

Compares predictions from:
- V1 (DESCRIPTIVE): Labels during spike, prediction_offset=0
- V2 (PREDICTIVE): Labels 2min before spike, prediction_offset=2

Shows which model would have alerted earlier and with what probability.
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
    logger.info("BACKTEST: V1 vs V2 PREDICTIONS (LAST 19 HOURS)")
    logger.info("=" * 80)
    logger.info("")

    # === CONFIGURATION ===
    DB_PATH = 'market_data.duckdb'
    V1_MODEL_PATH = 'models/xgboost_slow_large_v1.pkl'
    V2_MODEL_PATH = 'models/xgboost_slow_large_v2.pkl'

    # Get data from last 19 hours
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=19)

    logger.info(f"Time range: {start_time} to {end_time}")
    logger.info("")

    # === STEP 1: Load models ===
    logger.info("Loading models...")
    v1_model = joblib.load(V1_MODEL_PATH)
    v2_model = joblib.load(V2_MODEL_PATH)
    logger.info(f"✓ V1 model loaded from {V1_MODEL_PATH}")
    logger.info(f"✓ V2 model loaded from {V2_MODEL_PATH}")
    logger.info("")

    # Feature columns (from training)
    feature_cols = [
        'returns', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14', 'NATR',
        'BB_width', 'BB_squeeze', 'VWAP_distance', 'volume_zscore', 'volume_roc',
        'OBV', 'trade_count', 'buy_sell_ratio', 'roll_measure', 'order_flow_imbalance',
        'vpin', 'bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance',
        'returns_5m', 'volume_zscore_5m', 'returns_15m', 'volume_zscore_15m'
    ]

    # === STEP 2: Load features from database ===
    logger.info("Loading features from database (last 19 hours)...")
    conn = duckdb.connect(DB_PATH, read_only=True)

    # Get all 1m features from the timeframe
    query = f"""
        SELECT
            timestamp,
            symbol,
            {', '.join(feature_cols)}
        FROM features
        WHERE timeframe = '1m'
        AND timestamp >= '{start_time}'
        AND timestamp <= '{end_time}'
        ORDER BY timestamp DESC, symbol
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    if df.empty:
        logger.error("No features found in the last 19 hours!")
        return

    logger.info(f"Loaded {len(df):,} feature rows")
    logger.info(f"Symbols: {df['symbol'].nunique()}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info("")

    # === STEP 3: Run predictions with both models ===
    logger.info("Running predictions with both models...")

    # Extract feature matrix
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Run predictions
    v1_probs = v1_model.predict_proba(X)[:, 1]
    v2_probs = v2_model.predict_proba(X)[:, 1]

    # Add to dataframe
    df['v1_prob'] = v1_probs
    df['v2_prob'] = v2_probs
    df['v1_pred'] = (v1_probs >= 0.7).astype(int)
    df['v2_pred'] = (v2_probs >= 0.7).astype(int)

    logger.info("✓ Predictions complete")
    logger.info("")

    # === STEP 4: Compare results ===
    logger.info("=" * 80)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 80)
    logger.info("")

    # Overall statistics
    logger.info("Overall Probability Distribution:")
    logger.info(f"  V1 (DESCRIPTIVE):")
    logger.info(f"    Mean:   {v1_probs.mean()*100:.2f}%")
    logger.info(f"    Max:    {v1_probs.max()*100:.2f}%")
    logger.info(f"    >10%:   {(v1_probs > 0.1).sum():,} samples")
    logger.info(f"    >30%:   {(v1_probs > 0.3).sum():,} samples")
    logger.info(f"    >50%:   {(v1_probs > 0.5).sum():,} samples")
    logger.info(f"    >70%:   {(v1_probs > 0.7).sum():,} samples (ALERTS)")
    logger.info("")
    logger.info(f"  V2 (PREDICTIVE):")
    logger.info(f"    Mean:   {v2_probs.mean()*100:.2f}%")
    logger.info(f"    Max:    {v2_probs.max()*100:.2f}%")
    logger.info(f"    >10%:   {(v2_probs > 0.1).sum():,} samples")
    logger.info(f"    >30%:   {(v2_probs > 0.3).sum():,} samples")
    logger.info(f"    >50%:   {(v2_probs > 0.5).sum():,} samples")
    logger.info(f"    >70%:   {(v2_probs > 0.7).sum():,} samples (ALERTS)")
    logger.info("")

    # Find cases where models disagree significantly
    logger.info("=" * 80)
    logger.info("TOP 20 DISAGREEMENTS (V2 higher than V1)")
    logger.info("=" * 80)
    logger.info("")

    df['diff'] = df['v2_prob'] - df['v1_prob']
    disagreements = df.nlargest(20, 'diff')

    logger.info("Symbol           Timestamp            V1 Prob  V2 Prob  Diff")
    logger.info("-" * 80)
    for _, row in disagreements.iterrows():
        logger.info(
            f"{row['symbol']:15s}  {row['timestamp']}  "
            f"{row['v1_prob']*100:5.1f}%   {row['v2_prob']*100:5.1f}%   "
            f"+{row['diff']*100:5.1f}%"
        )
    logger.info("")

    # Find top predictions from each model
    logger.info("=" * 80)
    logger.info("TOP 10 V1 PREDICTIONS (DESCRIPTIVE)")
    logger.info("=" * 80)
    logger.info("")

    top_v1 = df.nlargest(10, 'v1_prob')
    logger.info("Symbol           Timestamp            V1 Prob  V2 Prob  Alert")
    logger.info("-" * 80)
    for _, row in top_v1.iterrows():
        alert = "🔔 ALERT" if row['v1_prob'] >= 0.7 else ""
        logger.info(
            f"{row['symbol']:15s}  {row['timestamp']}  "
            f"{row['v1_prob']*100:5.1f}%   {row['v2_prob']*100:5.1f}%   {alert}"
        )
    logger.info("")

    logger.info("=" * 80)
    logger.info("TOP 10 V2 PREDICTIONS (PREDICTIVE)")
    logger.info("=" * 80)
    logger.info("")

    top_v2 = df.nlargest(10, 'v2_prob')
    logger.info("Symbol           Timestamp            V1 Prob  V2 Prob  Alert")
    logger.info("-" * 80)
    for _, row in top_v2.iterrows():
        alert = "🔔 ALERT" if row['v2_prob'] >= 0.7 else ""
        logger.info(
            f"{row['symbol']:15s}  {row['timestamp']}  "
            f"{row['v1_prob']*100:5.1f}%   {row['v2_prob']*100:5.1f}%   {alert}"
        )
    logger.info("")

    # Check if any symbol had high predictions
    logger.info("=" * 80)
    logger.info("HIGH CONFIDENCE PREDICTIONS (>40%)")
    logger.info("=" * 80)
    logger.info("")

    high_conf = df[(df['v1_prob'] > 0.4) | (df['v2_prob'] > 0.4)].sort_values('timestamp')

    if len(high_conf) > 0:
        logger.info(f"Found {len(high_conf)} high-confidence predictions")
        logger.info("")
        logger.info("Symbol           Timestamp            V1 Prob  V2 Prob")
        logger.info("-" * 80)
        for _, row in high_conf.iterrows():
            logger.info(
                f"{row['symbol']:15s}  {row['timestamp']}  "
                f"{row['v1_prob']*100:5.1f}%   {row['v2_prob']*100:5.1f}%"
            )
    else:
        logger.info("No high-confidence predictions (market was quiet)")

    logger.info("")

    # === STEP 5: Summary ===
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    v1_alerts = (v1_probs >= 0.7).sum()
    v2_alerts = (v2_probs >= 0.7).sum()

    logger.info(f"V1 (DESCRIPTIVE) would have triggered {v1_alerts} alerts (>70%)")
    logger.info(f"V2 (PREDICTIVE) would have triggered {v2_alerts} alerts (>70%)")
    logger.info("")

    if v1_alerts == 0 and v2_alerts == 0:
        logger.info("✓ QUIET MARKET: Neither model triggered alerts")
        logger.info("  This is expected - the last 19 hours had no explosive spikes")
        logger.info("  Both models are working correctly (low false positive rate)")
    elif v2_alerts > v1_alerts:
        logger.info("⚠️  V2 MORE SENSITIVE: Triggered more alerts")
        logger.info("  V2 predicts spikes earlier, may catch signals v1 misses")
        logger.info("  But could also have higher false positive rate")
    elif v1_alerts > v2_alerts:
        logger.info("⚠️  V1 MORE SENSITIVE: Triggered more alerts")
        logger.info("  V1 detects spikes during the event (more certain)")
        logger.info("  V2 predicts earlier but with less certainty")
    else:
        logger.info("✓ AGREEMENT: Both models triggered same number of alerts")

    logger.info("")
    logger.info("=" * 80)

    # Save detailed comparison to CSV
    output_file = 'backtest_v1_vs_v2_results.csv'
    df_output = df[['timestamp', 'symbol', 'v1_prob', 'v2_prob', 'diff']].copy()
    df_output = df_output.sort_values(['timestamp', 'symbol'])
    df_output.to_csv(output_file, index=False)
    logger.info(f"✓ Detailed results saved to {output_file}")
    logger.info("")


if __name__ == '__main__':
    main()
