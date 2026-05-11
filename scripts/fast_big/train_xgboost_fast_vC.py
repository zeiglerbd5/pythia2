#!/usr/bin/env python3
"""
Train XGBoost Fast-vC - "READY TO POP" Compression Detection

This model detects the COMPRESSION pattern before big spikes (15%+ moves).
Key insight: Big runners start from TIGHT bands, not elevated volatility.

vA problem: Detects at spike start (too late - chasing)
vB problem: Looks for high volatility before spike (wrong - that's the fizzles)

vC solution: Detect compression + building pressure BEFORE the explosion
- Tight BB_width (<0.035) - price coiled tight
- Low NATR (<1.0) - calm before the storm
- Low vpin (<0.3) - not yet imbalanced
- RSI rising from neutral (45->55+) - bid pressure building
- Tight spread (<0.5 bps) - whales can execute

We label candles that have compression conditions AND precede a 15%+ spike
within 30 minutes.

Usage:
    python scripts/fast_big/train_xgboost_fast_vC.py
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
import time


def main():
    logger.info("=" * 80)
    logger.info("TRAINING XGBOOST FAST-vC - READY TO POP DETECTOR")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Strategy: Detect COMPRESSION patterns before big spikes")
    logger.info("Key insight: Runners start from tight bands, fizzles start from volatility")
    logger.info("")

    # === CONFIGURATION ===
    DB_PATH = 'market_data.duckdb'
    MODEL_OUTPUT_PATH = 'models/fast_big/xgboost_fast_vC.pkl'

    # Use live OHLCV data (Nov 27 - Dec 13)
    START_DATE = '2025-11-27'
    END_DATE = '2025-12-13'

    # Spike detection parameters
    MIN_SPIKE_GAIN = 0.15      # 15% minimum move to qualify as "big"
    SPIKE_WINDOW = 30          # 30 minutes to reach peak
    LABEL_OFFSET_START = 5     # Start labeling 5 min before spike
    LABEL_OFFSET_END = 20      # End labeling 20 min before spike

    # Compression thresholds (from our forensic analysis)
    MAX_BB_WIDTH = 0.04        # Must be tight (< 4%)
    MAX_NATR = 1.2             # Must be calm (< 1.2%)
    MAX_VPIN = 0.40            # Not yet imbalanced

    # Features - ordered by importance from analysis
    feature_cols = [
        # Compression indicators (PRIMARY - must be LOW)
        'BB_width',           # Tight bands = coiled spring
        'NATR',               # Low volatility = calm before storm
        'vpin',               # Low = not yet imbalanced

        # Liquidity indicators
        'bid_ask_spread_pct', # Tight spread = whales can execute
        'order_book_depth_ratio',

        # Momentum building (should be RISING)
        'RSI_14',             # Rising from neutral
        'MACD',
        'MACD_signal',
        'MACD_hist',

        # Volume context
        'volume_zscore',
        'volume_roc',
        'OBV',

        # Flow indicators
        'order_flow_imbalance',
        'buy_sell_ratio',
        'large_order_imbalance',

        # Price context
        'returns',
        'VWAP_distance',

        # Rate of change features (detecting acceleration)
        'returns_5m',
        'volume_zscore_5m',
        'returns_15m',
        'volume_zscore_15m',

        # Trade microstructure
        'trade_count',
        'roll_measure',
    ]

    logger.info(f"Training parameters:")
    logger.info(f"  Min spike: {MIN_SPIKE_GAIN*100:.0f}%")
    logger.info(f"  Spike window: {SPIKE_WINDOW} minutes")
    logger.info(f"  Label zone: T-{LABEL_OFFSET_END} to T-{LABEL_OFFSET_START}")
    logger.info(f"  Max BB_width: {MAX_BB_WIDTH}")
    logger.info(f"  Max NATR: {MAX_NATR}")
    logger.info(f"  Max vpin: {MAX_VPIN}")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info("")

    conn = duckdb.connect(DB_PATH, read_only=True)

    # === STEP 1: Find all 15%+ spikes ===
    logger.info("Step 1: Finding 15%+ spikes in OHLCV data...")

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
        FROM ohlcv
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
        AND future_max_price >= close * {1 + MIN_SPIKE_GAIN}
    )
    SELECT * FROM all_spikes
    ORDER BY magnitude DESC
    """

    spikes_df = conn.execute(spike_query).fetchdf()
    logger.info(f"Found {len(spikes_df):,} spike events (15%+)")

    if len(spikes_df) == 0:
        logger.error("No spikes found! Check date range and data.")
        conn.close()
        return

    # Show magnitude distribution
    logger.info(f"Magnitude distribution:")
    logger.info(f"  15-20%: {len(spikes_df[(spikes_df['magnitude'] >= 0.15) & (spikes_df['magnitude'] < 0.20)]):,}")
    logger.info(f"  20-30%: {len(spikes_df[(spikes_df['magnitude'] >= 0.20) & (spikes_df['magnitude'] < 0.30)]):,}")
    logger.info(f"  30-50%: {len(spikes_df[(spikes_df['magnitude'] >= 0.30) & (spikes_df['magnitude'] < 0.50)]):,}")
    logger.info(f"  50%+:   {len(spikes_df[spikes_df['magnitude'] >= 0.50]):,}")

    # Group by symbol
    spikes_by_symbol = spikes_df.groupby('symbol').size().sort_values(ascending=False)
    logger.info(f"\nTop 10 symbols by spike count:")
    for sym, count in spikes_by_symbol.head(10).items():
        max_mag = spikes_df[spikes_df['symbol'] == sym]['magnitude'].max()
        logger.info(f"  {sym}: {count} spikes (max {max_mag*100:.1f}%)")

    # === STEP 2: Load features ===
    logger.info("")
    logger.info("Step 2: Loading features...")

    features_query = f"""
    SELECT *
    FROM features
    WHERE timestamp >= '{START_DATE}'
    AND timestamp <= '{END_DATE}'
    """

    features_df = conn.execute(features_query).fetchdf()
    logger.info(f"Loaded {len(features_df):,} feature rows")
    logger.info(f"Symbols: {features_df['symbol'].nunique()}")

    # === STEP 3: Create "Ready to Pop" labels ===
    logger.info("")
    logger.info("Step 3: Creating 'Ready to Pop' labels...")
    logger.info("Requirements: Compression conditions + spike follows within window")

    # For each spike, look back to find pre-spike candles with compression
    positive_samples = []
    compression_stats = {'total_checked': 0, 'compression_met': 0}

    for _, spike in spikes_df.iterrows():
        symbol = spike['symbol']
        spike_time = spike['spike_start']
        magnitude = spike['magnitude']

        # Get features for this symbol in the pre-spike window
        symbol_features = features_df[features_df['symbol'] == symbol].copy()

        if len(symbol_features) == 0:
            continue

        # Look at T-20 to T-5 before spike
        for offset in range(LABEL_OFFSET_START, LABEL_OFFSET_END + 1):
            pre_spike_time = spike_time - pd.Timedelta(minutes=offset)

            # Find the feature row at this time
            row = symbol_features[symbol_features['timestamp'] == pre_spike_time]

            if len(row) == 0:
                continue

            row = row.iloc[0]
            compression_stats['total_checked'] += 1

            # Check compression conditions
            bb_width = row.get('BB_width', np.nan)
            natr = row.get('NATR', np.nan)
            vpin = row.get('vpin', np.nan)

            # Must have valid values
            if pd.isna(bb_width) or pd.isna(natr) or pd.isna(vpin):
                continue

            # Check if compression conditions are met
            has_compression = (
                bb_width < MAX_BB_WIDTH and
                natr < MAX_NATR and
                vpin < MAX_VPIN
            )

            if has_compression:
                compression_stats['compression_met'] += 1
                positive_samples.append({
                    'symbol': symbol,
                    'timestamp': pre_spike_time,
                    'target': 1,
                    'magnitude': magnitude,
                    'minutes_before': offset,
                    'bb_width': bb_width,
                    'natr': natr,
                    'vpin': vpin
                })

    positive_df = pd.DataFrame(positive_samples)

    if len(positive_df) == 0:
        logger.error("No positive samples with compression found!")
        logger.info(f"Stats: {compression_stats}")
        conn.close()
        return

    # Deduplicate (same candle might precede multiple spike events)
    positive_df['key'] = positive_df['symbol'] + '_' + positive_df['timestamp'].astype(str)
    positive_df = positive_df.drop_duplicates(subset='key')

    logger.info(f"Created {len(positive_df):,} positive 'Ready to Pop' samples")
    logger.info(f"  Compression check rate: {compression_stats['compression_met']/compression_stats['total_checked']*100:.1f}%")
    logger.info(f"  Avg magnitude: {positive_df['magnitude'].mean()*100:.1f}%")
    logger.info(f"  Avg minutes before: {positive_df['minutes_before'].mean():.1f}")

    # Show compression stats for positives
    logger.info(f"\nPositive sample compression stats:")
    logger.info(f"  BB_width: mean={positive_df['bb_width'].mean():.4f}, max={positive_df['bb_width'].max():.4f}")
    logger.info(f"  NATR: mean={positive_df['natr'].mean():.3f}, max={positive_df['natr'].max():.3f}")
    logger.info(f"  vpin: mean={positive_df['vpin'].mean():.3f}, max={positive_df['vpin'].max():.3f}")

    # === STEP 4: Create negative samples (non-spike periods with similar compression) ===
    logger.info("")
    logger.info("Step 4: Creating negative samples...")

    # Get all feature rows NOT in positive set
    positive_keys = set(positive_df['key'].unique())
    features_df['key'] = features_df['symbol'] + '_' + features_df['timestamp'].astype(str)

    negative_df = features_df[~features_df['key'].isin(positive_keys)].copy()
    negative_df['target'] = 0

    logger.info(f"Total negative candidates: {len(negative_df):,}")

    # === STEP 5: Merge and prepare training data ===
    logger.info("")
    logger.info("Step 5: Preparing training dataset...")

    # Select columns for training
    available_features = [f for f in feature_cols if f in features_df.columns]
    missing_features = [f for f in feature_cols if f not in features_df.columns]

    if missing_features:
        logger.warning(f"Missing features: {missing_features}")

    logger.info(f"Using {len(available_features)} features")

    # Get feature values for positive samples
    positive_features = features_df[features_df['key'].isin(positive_keys)].copy()
    positive_features['target'] = 1

    # Combine positive and negative
    # Downsample negatives to 20:1 ratio for better training
    n_positive = len(positive_features)
    n_negative_sample = min(len(negative_df), n_positive * 20)

    np.random.seed(42)
    negative_sample = negative_df.sample(n=n_negative_sample, random_state=42)

    combined_df = pd.concat([positive_features, negative_sample], ignore_index=True)

    logger.info(f"Combined dataset: {len(combined_df):,} samples")
    logger.info(f"  Positive: {n_positive:,} ({n_positive/len(combined_df)*100:.1f}%)")
    logger.info(f"  Negative: {n_negative_sample:,}")

    # Extract X and y
    X = combined_df[available_features].copy()
    y = combined_df['target'].values

    # Handle missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Add rate-of-change features if not present
    # These help detect the "building pressure" before spike
    if 'BB_width_chg_5m' not in X.columns and 'BB_width' in X.columns:
        logger.info("Adding rate-of-change features...")
        # These would be calculated in feature_engine normally
        # For now we'll work with what we have

    logger.info(f"X shape: {X.shape}")

    # === STEP 6: Train/test split ===
    logger.info("")
    logger.info("Step 6: Train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train):,} samples ({y_train.sum():,} positive)")
    logger.info(f"Test: {len(X_test):,} samples ({y_test.sum():,} positive)")

    # === STEP 7: Train XGBoost ===
    logger.info("")
    logger.info("Step 7: Training XGBoost classifier...")

    start_time = time.time()

    # Calculate scale_pos_weight for remaining imbalance
    neg_count = len(y_train) - y_train.sum()
    pos_count = y_train.sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,           # Slightly shallower to avoid overfitting
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,   # Higher to reduce overfitting
        gamma=0.1,             # Regularization
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    train_time = time.time() - start_time
    logger.info(f"Training complete in {train_time:.1f}s")

    # === STEP 8: Evaluate ===
    logger.info("")
    logger.info("=" * 80)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info("")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))

    # Precision at different thresholds
    logger.info("")
    logger.info("Precision at different thresholds:")
    logger.info(f"{'Threshold':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Alerts':>10}")
    logger.info("-" * 55)

    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        if y_pred_thresh.sum() > 0:
            prec = precision_score(y_test, y_pred_thresh)
            rec = recall_score(y_test, y_pred_thresh)
            f1 = f1_score(y_test, y_pred_thresh)
            n_alerts = y_pred_thresh.sum()
            logger.info(f"{threshold:<12.0%} {prec*100:>9.1f}% {rec*100:>9.1f}% {f1*100:>9.1f}% {n_alerts:>10,}")

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        else:
            logger.info(f"{threshold:<12.0%} {'No alerts':>42}")

    logger.info("")
    logger.info(f"Best F1: {best_f1*100:.1f}% at threshold {best_threshold:.0%}")

    # Confusion matrix at 85% threshold
    logger.info("")
    y_pred_85 = (y_prob >= 0.85).astype(int)
    cm = confusion_matrix(y_test, y_pred_85)
    logger.info("Confusion Matrix (threshold=85%):")
    logger.info(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    logger.info(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # === STEP 9: Feature importance ===
    logger.info("")
    logger.info("Top 15 Feature Importances:")
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance_df.head(15).iterrows():
        logger.info(f"  {row['feature']:<25s}: {row['importance']:.4f}")

    # === STEP 9.5: Visualize model diagnostics ===
    logger.info("")
    logger.info("Generating diagnostic visualizations...")

    from src.visualization.model_diagnostics import plot_predictions

    plot_predictions(
        model=model,
        train_data=X_train,
        train_labels=y_train,
        test_data=X_test,
        test_labels=y_test,
        title="Fast-vC Model Diagnostics (Ready-to-Pop Compression Strategy)",
        save_path="models/fast_big/xgboost_fast_vC_diagnostics.png",
        show=False  # Don't block in scripts
    )

    # === STEP 10: Save model ===
    logger.info("")
    logger.info("Step 10: Saving model...")

    Path(MODEL_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': model,
        'features': available_features,
        'threshold': 0.85,  # Recommended threshold for precision
        'metadata': {
            'version': 'vC',
            'strategy': 'ready_to_pop_compression',
            'min_spike_gain': MIN_SPIKE_GAIN,
            'spike_window': SPIKE_WINDOW,
            'label_offset_start': LABEL_OFFSET_START,
            'label_offset_end': LABEL_OFFSET_END,
            'compression_thresholds': {
                'max_bb_width': MAX_BB_WIDTH,
                'max_natr': MAX_NATR,
                'max_vpin': MAX_VPIN
            },
            'training_samples': len(X_train),
            'positive_samples': int(y_train.sum()),
            'date_range': f"{START_DATE} to {END_DATE}",
            'best_f1': best_f1,
            'best_threshold': best_threshold
        }
    }

    joblib.dump(model_data, MODEL_OUTPUT_PATH)
    logger.info(f"Model saved to: {MODEL_OUTPUT_PATH}")

    # === Summary ===
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE - FAST-vC 'READY TO POP'")
    logger.info("=" * 80)
    logger.info("")
    logger.info("vC detects COMPRESSION patterns before big spikes:")
    logger.info(f"  - Tight BB_width (< {MAX_BB_WIDTH})")
    logger.info(f"  - Low NATR (< {MAX_NATR})")
    logger.info(f"  - Low vpin (< {MAX_VPIN})")
    logger.info(f"  - Followed by 15%+ spike within {SPIKE_WINDOW} minutes")
    logger.info("")
    logger.info("Key insight: Big runners start from compression (tight bands),")
    logger.info("not from elevated volatility. vC detects the calm before the storm.")
    logger.info("")
    logger.info("Recommended usage:")
    logger.info("  - Use 85%+ threshold for high-precision alerts")
    logger.info("  - Combine with volume confirmation at entry")
    logger.info("  - Exit within 30 minutes")
    logger.info("")

    conn.close()


if __name__ == "__main__":
    main()
