#!/usr/bin/env python3
"""
Train XGBoost v2 - PREDICTIVE Model (Not Descriptive)

Key difference from v1:
  - v1: Labels candles DURING spike (prediction_offset=0) - DESCRIPTIVE
  - v2: Labels candles 2min BEFORE spike (prediction_offset=2) - PREDICTIVE

This model predicts spikes 2 minutes earlier, before volume explodes.

Usage:
    python scripts/train_xgboost_v2_predictive.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Import our DatasetBuilder with modified SpikeTargetGenerator
from src.models.dataset import SpikeTargetGenerator


def main():
    logger.info("=" * 80)
    logger.info("XGBOOST V2 - PREDICTIVE MODEL (prediction_offset=2)")
    logger.info("=" * 80)
    logger.info("")

    # === CONFIGURATION ===
    DB_PATH = 'market_data.duckdb'
    TIMEFRAME = '1m'
    START_DATE = '2025-10-06'  # Same training period as v1
    END_DATE = '2025-10-20'

    # Model parameters
    PREDICTION_OFFSET = 2  # KEY CHANGE: Predict 2 minutes BEFORE spike

    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Prediction offset: {PREDICTION_OFFSET} minutes (PREDICTIVE)")
    logger.info("")

    # === STEP 1: Get all symbols ===
    logger.info("Getting all symbols...")

    # Retry logic for database locks (collector may be running)
    import time
    max_retries = 10
    retry_delay = 5

    conn = None
    for attempt in range(max_retries):
        try:
            conn = duckdb.connect(DB_PATH, read_only=True)
            break
        except Exception as e:
            if "lock" in str(e).lower() and attempt < max_retries - 1:
                logger.warning(f"Database locked (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to database: {e}")
                raise

    if conn is None:
        logger.error("Could not connect to database after retries")
        return

    # Get symbols from candles table
    result = conn.execute("""
        SELECT DISTINCT symbol
        FROM candles
        WHERE timestamp >= ?
        AND timestamp <= ?
        ORDER BY symbol
    """, [START_DATE, END_DATE]).fetchall()

    symbols = [r[0] for r in result]
    logger.info(f"Found {len(symbols)} symbols")
    logger.info("")

    # === STEP 2: Generate targets with prediction_offset=2 ===
    logger.info("Generating PREDICTIVE targets (prediction_offset=2)...")
    logger.info("This labels candles 2 minutes BEFORE spikes start")
    logger.info("")

    target_gen = SpikeTargetGenerator(
        price_window=3,
        volume_window=2,
        min_price_spike=0.06,
        min_volume_spike=5.0,
        prediction_offset=PREDICTION_OFFSET  # NEW PARAMETER
    )

    all_features = []
    all_targets = []
    all_symbols = []

    processed = 0
    positive_count = 0

    # Feature columns from v1 model
    feature_cols = [
        'returns', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14', 'NATR',
        'BB_width', 'BB_squeeze', 'VWAP_distance', 'volume_zscore', 'volume_roc',
        'OBV', 'trade_count', 'buy_sell_ratio', 'roll_measure', 'order_flow_imbalance',
        'vpin', 'bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance',
        'returns_5m', 'volume_zscore_5m', 'returns_15m', 'volume_zscore_15m'
    ]

    for i, symbol in enumerate(symbols):
        if (i + 1) % 20 == 0:
            logger.info(f"Processing {i+1}/{len(symbols)} symbols...")

        # Get prices and volumes for target generation
        result = conn.execute(f"""
            SELECT timestamp, close, volume
            FROM candles
            WHERE symbol = '{symbol}'
            AND timestamp >= '{START_DATE}'
            AND timestamp <= '{END_DATE}'
            ORDER BY timestamp
        """).fetchdf()

        if len(result) < 100:  # Skip symbols with insufficient data
            continue

        # Generate targets using MODIFIED SpikeTargetGenerator
        prices = result['close']
        volumes = result['volume']
        targets_series = target_gen.generate_targets(prices, volumes, result['timestamp'])

        # Get features
        timestamps_str = result['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        timestamps_clause = "', '".join(timestamps_str)

        features_query = f"""
            SELECT {', '.join(feature_cols)}
            FROM features
            WHERE symbol = '{symbol}'
            AND timeframe = '1m'
            AND timestamp IN ('{timestamps_clause}')
            ORDER BY timestamp
        """

        features_df = conn.execute(features_query).fetchdf()

        if len(features_df) == 0:
            continue

        # Align features and targets
        if len(features_df) == len(targets_series):
            targets_array = targets_series.values
            features_array = features_df.values

            all_features.append(features_array)
            all_targets.append(targets_array)
            all_symbols.extend([symbol] * len(targets_array))

            positive_count += int(targets_array.sum())
            processed += 1

    conn.close()

    logger.info(f"Processed {processed} symbols")
    logger.info(f"Total positive labels: {positive_count}")
    logger.info("")

    if processed == 0:
        logger.error("No data processed! Exiting.")
        return

    # === STEP 3: Create training dataset ===
    logger.info("Creating training dataset...")

    X = np.vstack(all_features)
    y = np.concatenate(all_targets)

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Total samples: {len(X):,}")
    logger.info(f"Positive samples: {int(y.sum()):,} ({y.mean()*100:.2f}%)")
    logger.info(f"Feature shape: {X.shape}")
    logger.info(f"Memory: {X.nbytes / 1024 / 1024:.1f} MB")
    logger.info("")

    # Downsample negatives to balance dataset (100:1 ratio)
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    n_negatives_to_keep = min(len(pos_indices) * 100, len(neg_indices))
    neg_indices_sampled = np.random.choice(neg_indices, n_negatives_to_keep, replace=False)

    keep_indices = np.concatenate([pos_indices, neg_indices_sampled])
    np.random.shuffle(keep_indices)

    X = X[keep_indices]
    y = y[keep_indices]

    logger.info(f"After downsampling negatives:")
    logger.info(f"Total samples: {len(X):,}")
    logger.info(f"Positive samples: {int(y.sum()):,} ({y.mean()*100:.2f}%)")
    logger.info("")

    # === STEP 4: Train/val/test split ===
    logger.info("Splitting data...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {len(y_train):,} ({int(y_train.sum())} pos, {y_train.mean()*100:.2f}%)")
    logger.info(f"Val:   {len(y_val):,} ({int(y_val.sum())} pos, {y_val.mean()*100:.2f}%)")
    logger.info(f"Test:  {len(y_test):,} ({int(y_test.sum())} pos, {y_test.mean()*100:.2f}%)")
    logger.info("")

    # === STEP 5: Train XGBoost ===
    logger.info("=" * 80)
    logger.info("TRAINING XGBOOST V2 (PREDICTIVE)")
    logger.info("=" * 80)
    logger.info("")

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info(f"scale_pos_weight: {pos_weight:.1f}")
    logger.info("")

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        eval_metric='aucpr',
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )

    logger.info("Training started...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    logger.info("")
    logger.info("Training complete!")
    logger.info("")

    # === STEP 6: Evaluate ===
    logger.info("=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)
    logger.info("")

    # Validation set
    y_val_prob = clf.predict_proba(X_val)[:, 1]

    logger.info("Validation Set:")
    for threshold in [0.3, 0.5, 0.7]:
        y_pred = (y_val_prob >= threshold).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        tp = ((y_pred == 1) & (y_val == 1)).sum()
        fp = ((y_pred == 1) & (y_val == 0)).sum()
        fn = ((y_pred == 0) & (y_val == 1)).sum()

        logger.info(f"  Threshold {threshold:.1f}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (TP={tp}, FP={fp}, FN={fn})")

    logger.info("")

    # Test set
    y_test_prob = clf.predict_proba(X_test)[:, 1]

    logger.info("Test Set:")
    for threshold in [0.3, 0.5, 0.7]:
        y_pred = (y_test_prob >= threshold).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        logger.info(f"  Threshold {threshold:.1f}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    logger.info("")

    # === STEP 7: Feature importance ===
    logger.info("=" * 80)
    logger.info("TOP 15 FEATURES")
    logger.info("=" * 80)
    logger.info("")

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    for i, idx in enumerate(indices):
        logger.info(f"{i+1:2d}. {feature_cols[idx]:25s}: {importances[idx]:.4f}")

    logger.info("")

    # === STEP 8: Compare to v1 ===
    logger.info("=" * 80)
    logger.info("COMPARISON: V1 vs V2")
    logger.info("=" * 80)
    logger.info("")

    val_f1_50 = f1_score(y_val, (y_val_prob >= 0.5).astype(int), zero_division=0)
    test_f1_50 = f1_score(y_test, (y_test_prob >= 0.5).astype(int), zero_division=0)

    logger.info("V1 (DESCRIPTIVE, prediction_offset=0):")
    logger.info("  - Labels candles DURING spike")
    logger.info("  - Validation F1 @ 0.5: ~0.77 (from previous training)")
    logger.info("")
    logger.info("V2 (PREDICTIVE, prediction_offset=2):")
    logger.info("  - Labels candles 2min BEFORE spike")
    logger.info(f"  - Validation F1 @ 0.5: {val_f1_50:.3f}")
    logger.info(f"  - Test F1 @ 0.5: {test_f1_50:.3f}")
    logger.info("")

    if val_f1_50 > 0.50:
        logger.info("✓ SUCCESS: V2 model learned predictive patterns!")
        logger.info("  Model can forecast spikes before they start.")
    elif val_f1_50 > 0.30:
        logger.info("~ PARTIAL: Some predictive ability, but weaker than v1")
        logger.info("  This is expected - predicting BEFORE spike is harder")
    else:
        logger.info("✗ WEAK: Limited predictive power")
        logger.info("  Spikes may be too random to predict 2min in advance")

    logger.info("")

    # === STEP 9: Save model ===
    logger.info("=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)
    logger.info("")

    model_path = 'models/xgboost_slow_large_v2.pkl'
    joblib.dump(clf, model_path)
    logger.info(f"✓ Model saved to {model_path}")
    logger.info("")
    logger.info("To use this model in production:")
    logger.info("  1. Edit src/data_ingestion/integrated_collector.py line 127")
    logger.info("  2. Change: xgboost_model_path='models/xgboost_slow_large_v1.pkl'")
    logger.info("  3. To:     xgboost_model_path='models/xgboost_slow_large_v2.pkl'")
    logger.info("  4. Restart integrated collector")
    logger.info("")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
