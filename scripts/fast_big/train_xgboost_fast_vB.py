#!/usr/bin/env python3
"""
Train XGBoost Fast-vB - WHALE SPARK Detection

This model detects the "spark" before whale spikes (10%+ moves).
Key insight: Label candles T-10 to T-1 BEFORE the spike, not at T=0.

Based on analysis findings:
- BB_width: 7x higher before whales (Cohen's d = 0.89)
- NATR: 6x higher before whales (Cohen's d = 0.83)
- bid_ask_spread: 5x higher (d = 0.35)
- RSI: Lower before whales (d = -0.26)
- vpin: Slightly higher (d = 0.32)

These signals ARE detectable at T-10 to T-1, before the spike starts.

Usage:
    python scripts/fast_big/train_xgboost_fast_vB.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib


def main():
    logger.info("=" * 80)
    logger.info("TRAINING XGBOOST FAST-vB - WHALE SPARK DETECTOR")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Strategy: Detect 'spark' patterns 1-10 minutes BEFORE whale spikes")
    logger.info("Target: 10%+ price moves in next 30 minutes (whale/moby)")
    logger.info("")

    # === CONFIGURATION ===
    DB_PATH = 'market_data_analysis.duckdb'
    MODEL_OUTPUT_PATH = 'models/fast_big/xgboost_fast_vB.pkl'

    # Date range for training
    START_DATE = '2025-10-06'
    END_DATE = '2025-10-20'

    # Spark zone configuration
    SPARK_LOOKBACK = 10  # Label candles T-10 to T-1 before spike
    MIN_WHALE_GAIN = 0.10  # 10% = whale threshold
    MIN_MOBY_GAIN = 0.20   # 20% = moby (extra weight)
    SPIKE_WINDOW = 30      # 30 min forward window

    # Features based on analysis - ordered by discriminative power
    feature_cols = [
        # Strongest discriminators (Cohen's d > 0.5)
        'BB_width',          # d=0.89 - MOST IMPORTANT
        'NATR',              # d=0.83 - 2nd most important

        # Medium discriminators (0.25 < d < 0.5)
        'bid_ask_spread_pct',  # d=0.35
        'vpin',                # d=0.32
        'VWAP_distance',       # d=-0.27 (negative = lower before whales)
        'RSI_14',              # d=-0.26 (negative = more oversold before whales)

        # Weak but significant discriminators
        'order_flow_imbalance',   # d=-0.08
        'order_book_depth_ratio', # d=0.05
        'volume_zscore',          # d=0.04
        'returns',                # d=-0.03

        # Additional features from vA that might help
        'MACD', 'MACD_signal', 'MACD_hist',
        'volume_roc', 'OBV', 'trade_count', 'buy_sell_ratio',
        'roll_measure', 'large_order_imbalance',
        'returns_5m', 'volume_zscore_5m', 'returns_15m', 'volume_zscore_15m'
    ]

    logger.info(f"Training parameters:")
    logger.info(f"  Spark lookback: T-{SPARK_LOOKBACK} to T-1")
    logger.info(f"  Whale threshold: {MIN_WHALE_GAIN*100:.0f}%")
    logger.info(f"  Moby threshold: {MIN_MOBY_GAIN*100:.0f}%")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info("")

    conn = duckdb.connect(DB_PATH, read_only=True)

    # === STEP 1: Find all whale/moby spikes ===
    logger.info("Step 1: Finding whale/moby spikes...")

    spike_query = f"""
    WITH price_data AS (
        SELECT
            symbol,
            timestamp,
            close,
            MAX(close) OVER (
                PARTITION BY symbol
                ORDER BY timestamp
                ROWS BETWEEN 1 FOLLOWING AND {SPIKE_WINDOW} FOLLOWING
            ) as future_max_price
        FROM candles
        WHERE timestamp >= '{START_DATE}'
        AND timestamp <= '{END_DATE}'
    ),
    all_spikes AS (
        SELECT
            symbol,
            timestamp as spike_start,
            close as start_price,
            future_max_price,
            (future_max_price - close) / NULLIF(close, 0) as magnitude
        FROM price_data
        WHERE close > 0
        AND future_max_price >= close * {1 + MIN_WHALE_GAIN}
    )
    SELECT * FROM all_spikes
    ORDER BY symbol, spike_start
    """

    spikes_df = conn.execute(spike_query).fetchdf()
    logger.info(f"Found {len(spikes_df):,} whale+ spike events")

    # Classify magnitude
    spikes_df['is_moby'] = spikes_df['magnitude'] >= MIN_MOBY_GAIN
    moby_count = spikes_df['is_moby'].sum()
    whale_count = len(spikes_df) - moby_count
    logger.info(f"  Whales (10-20%): {whale_count:,}")
    logger.info(f"  Mobys (20%+): {moby_count:,}")

    # === STEP 2: Create spark zone labels ===
    logger.info("")
    logger.info("Step 2: Creating spark zone labels (T-10 to T-1 before each spike)...")

    # For each spike, we want to label the candles T-10 to T-1 before it
    # This is the "spark zone" where we want the model to detect patterns

    spark_times = []
    for _, spike in spikes_df.iterrows():
        symbol = spike['symbol']
        spike_time = spike['spike_start']
        is_moby = spike['is_moby']

        # Label T-10 to T-1 before spike
        for offset in range(1, SPARK_LOOKBACK + 1):
            spark_time = spike_time - pd.Timedelta(minutes=offset)
            spark_times.append({
                'symbol': symbol,
                'timestamp': spark_time,
                'target': 1,
                'is_moby': is_moby,
                'minutes_before_spike': offset
            })

    spark_df = pd.DataFrame(spark_times)
    logger.info(f"Created {len(spark_df):,} positive spark zone labels")

    # === STEP 3: Load features and create training data ===
    logger.info("")
    logger.info("Step 3: Loading features and creating training dataset...")

    # Get all features
    features_query = f"""
    SELECT *
    FROM features
    WHERE timestamp >= '{START_DATE}'
    AND timestamp <= '{END_DATE}'
    """

    features_df = conn.execute(features_query).fetchdf()
    logger.info(f"Loaded {len(features_df):,} feature rows")

    # Join spark labels with features
    # Positive samples: rows in spark zone
    # Negative samples: all other rows

    # Create unique key for joining
    spark_df['key'] = spark_df['symbol'] + '_' + spark_df['timestamp'].astype(str)
    features_df['key'] = features_df['symbol'] + '_' + features_df['timestamp'].astype(str)

    # Mark positive samples
    positive_keys = set(spark_df['key'].unique())
    features_df['target'] = features_df['key'].isin(positive_keys).astype(int)

    # Add moby weight info
    moby_keys = set(spark_df[spark_df['is_moby']]['key'].unique())
    features_df['is_moby'] = features_df['key'].isin(moby_keys)

    positive_count = features_df['target'].sum()
    negative_count = len(features_df) - positive_count
    logger.info(f"Positive samples (spark zone): {positive_count:,}")
    logger.info(f"Negative samples: {negative_count:,}")
    logger.info(f"Imbalance ratio: 1:{negative_count // positive_count}")

    # === STEP 4: Prepare features for training ===
    logger.info("")
    logger.info("Step 4: Preparing features...")

    # Filter to available features
    available_features = [f for f in feature_cols if f in features_df.columns]
    missing_features = [f for f in feature_cols if f not in features_df.columns]

    if missing_features:
        logger.warning(f"Missing features: {missing_features}")

    logger.info(f"Using {len(available_features)} features")

    # Extract X and y
    X = features_df[available_features].copy()
    y = features_df['target'].values

    # Sample weights: moby spark zones get 2x weight
    sample_weights = np.ones(len(y))
    sample_weights[features_df['is_moby'].values] = 2.0

    # Handle missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y distribution: {np.bincount(y)}")

    # === STEP 5: Train/test split ===
    logger.info("")
    logger.info("Step 5: Train/test split...")

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train):,} samples")
    logger.info(f"Test: {len(X_test):,} samples")

    # === STEP 6: Train XGBoost ===
    logger.info("")
    logger.info("Step 6: Training XGBoost classifier...")

    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = negative_count / positive_count

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )

    # === STEP 7: Evaluate ===
    logger.info("")
    logger.info("Step 7: Evaluation...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    logger.info("")
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))

    # Precision at different thresholds
    logger.info("")
    logger.info("Precision at different thresholds:")
    for threshold in [0.5, 0.7, 0.8, 0.9, 0.95]:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        if y_pred_thresh.sum() > 0:
            prec = precision_score(y_test, y_pred_thresh)
            rec = recall_score(y_test, y_pred_thresh)
            n_alerts = y_pred_thresh.sum()
            logger.info(f"  Threshold {threshold:.0%}: Precision={prec:.1%}, Recall={rec:.1%}, Alerts={n_alerts:,}")
        else:
            logger.info(f"  Threshold {threshold:.0%}: No alerts")

    # Feature importance
    logger.info("")
    logger.info("Top 15 Feature Importance:")
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance_df.head(15).iterrows():
        logger.info(f"  {row['feature']:<25s}: {row['importance']:.4f}")

    # === STEP 8: Save model ===
    logger.info("")
    logger.info("Step 8: Saving model...")

    Path(MODEL_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'model': model,
        'features': available_features,
        'threshold': 0.90,  # Default threshold
        'metadata': {
            'spark_lookback': SPARK_LOOKBACK,
            'min_whale_gain': MIN_WHALE_GAIN,
            'training_samples': len(X_train),
            'positive_samples': int(y_train.sum())
        }
    }, MODEL_OUTPUT_PATH)

    logger.info(f"Model saved to: {MODEL_OUTPUT_PATH}")

    # === Summary ===
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Fast-vB detects 'whale spark' patterns in the T-10 to T-1 zone")
    logger.info("before 10%+ price spikes. Use 90%+ confidence for alerts.")
    logger.info("")
    logger.info("Key insight: BB_width and NATR are the strongest predictors,")
    logger.info("showing high volatility BEFORE the spike, not during it.")

    conn.close()


if __name__ == "__main__":
    main()
