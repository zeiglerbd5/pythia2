#!/usr/bin/env python3
"""
Train XGBoost V3 - SLOW & LARGE Spike Detection

Target: Multi-hour runners (30-100%+ gains over hours)
NOT: Fast 5-minute spikes

Key changes from V1/V2:
  - price_window: 60 minutes (was 3) - look for sustained moves
  - min_price_spike: 15% (was 6%) - bigger targets
  - volume_window: 30 minutes (was 2) - sustained elevated volume
  - min_volume_spike: 2.0x (was 5x) - lower but sustained
  - prediction_offset: 15 minutes (was 0/2) - earlier warning

Examples of target spikes:
  - LOKA-USD: +30% over 1 hour (Dec 7)
  - MLN-USD: +109% over 24 hours (Oct 19)
  - LRDS-USD: +40% over 18 hours (Dec 7-8)

Usage:
    python scripts/train_xgboost_v3_slow_large.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import time


class SlowLargeSpikeTargetGenerator:
    """
    Generate targets for SLOW & LARGE spikes (multi-hour runners).

    Different from V1/V2 which targeted fast 5-minute spikes.
    This targets sustained moves that build over 1-2+ hours.
    """

    def __init__(
        self,
        price_window: int = 60,      # 60 minutes to look ahead for price
        volume_window: int = 30,      # 30 minutes for volume confirmation
        min_price_spike: float = 0.15,  # 15% minimum move
        min_volume_spike: float = 2.0,  # 2x sustained volume (lower than V1's 5x)
        prediction_offset: int = 15     # 15 minutes early warning
    ):
        """
        Initialize Slow & Large spike detector.

        Args:
            price_window: Minutes to look forward for price move (default: 60)
            volume_window: Minutes to look forward for volume (default: 30)
            min_price_spike: Minimum price increase (default: 15%)
            min_volume_spike: Minimum avg volume multiplier (default: 2x)
            prediction_offset: Minutes to offset prediction (default: 15)
        """
        self.price_window = int(price_window)
        self.volume_window = int(volume_window)
        self.min_price_spike = min_price_spike
        self.min_volume_spike = min_volume_spike
        self.prediction_offset = int(prediction_offset)

        logger.info(
            f"SlowLargeSpikeTargetGenerator initialized",
            extra={
                "price_window": f"{self.price_window} minutes",
                "volume_window": f"{self.volume_window} minutes",
                "min_price_spike": f"{min_price_spike*100:.0f}%",
                "min_volume_spike": f"{min_volume_spike:.1f}x avg",
                "prediction_offset": f"{prediction_offset} minutes"
            }
        )

    def generate_targets(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        timestamps: pd.Series = None
    ) -> pd.Series:
        """
        Generate binary targets for Slow & Large spikes.

        Criteria:
        1. Price rises 15%+ in the next 60 minutes
        2. Volume is 2x+ average over the next 30 minutes
        3. Signal fires 15 minutes BEFORE the move accelerates

        Args:
            prices: Close prices (1-minute candles)
            volumes: Trade volumes (1-minute candles)
            timestamps: Corresponding timestamps (optional)

        Returns:
            Binary series (0/1) indicating pre-spike pattern
        """
        n = len(prices)
        max_window = max(self.price_window, self.volume_window)
        total_lookahead = max_window + self.prediction_offset

        # Convert to numpy
        price_arr = prices.values
        volume_arr = volumes.values

        # Initialize targets
        targets = np.zeros(n, dtype=np.float32)

        # Calculate rolling average volume (for comparison)
        # Use 60-minute lookback for "normal" volume
        lookback = 60
        rolling_avg_volume = np.zeros(n, dtype=np.float32)
        for i in range(lookback, n):
            rolling_avg_volume[i] = volume_arr[i-lookback:i].mean()

        # For each candle, check if a Slow & Large spike follows
        for i in range(lookback, n - total_lookahead):
            # Current price and baseline volume
            current_price = price_arr[i]
            baseline_volume = rolling_avg_volume[i]

            if baseline_volume <= 0 or current_price <= 0:
                continue

            # Look ahead with prediction offset
            # Price check: max price in [i+offset : i+offset+price_window]
            price_start = i + 1 + self.prediction_offset
            price_end = price_start + self.price_window

            if price_end > n:
                continue

            max_future_price = price_arr[price_start:price_end].max()
            price_return = (max_future_price - current_price) / current_price

            # Volume check: average volume in [i+offset : i+offset+volume_window]
            vol_start = i + 1 + self.prediction_offset
            vol_end = vol_start + self.volume_window

            if vol_end > n:
                continue

            avg_future_volume = volume_arr[vol_start:vol_end].mean()
            volume_ratio = avg_future_volume / baseline_volume

            # Check both conditions
            has_price_spike = price_return >= self.min_price_spike
            has_volume_surge = volume_ratio >= self.min_volume_spike

            if has_price_spike and has_volume_surge:
                targets[i] = 1.0

        return pd.Series(targets, index=prices.index)


def main():
    logger.info("=" * 80)
    logger.info("XGBOOST V3 - SLOW & LARGE SPIKE DETECTION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Target: Multi-hour runners (15%+ in 60 min, sustained volume)")
    logger.info("NOT: Fast 5-minute spikes")
    logger.info("")

    # === CONFIGURATION ===
    DB_PATH = 'market_data.duckdb'
    TIMEFRAME = '1m'
    START_DATE = '2025-10-06'
    END_DATE = '2025-10-20'

    # V3 parameters for Slow & Large spikes
    PRICE_WINDOW = 60          # 60 minutes
    VOLUME_WINDOW = 30         # 30 minutes
    MIN_PRICE_SPIKE = 0.15     # 15%
    MIN_VOLUME_SPIKE = 2.0     # 2x average
    PREDICTION_OFFSET = 15     # 15 min early warning

    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info("")
    logger.info("V3 Parameters:")
    logger.info(f"  Price window: {PRICE_WINDOW} minutes")
    logger.info(f"  Volume window: {VOLUME_WINDOW} minutes")
    logger.info(f"  Min price spike: {MIN_PRICE_SPIKE*100:.0f}%")
    logger.info(f"  Min volume spike: {MIN_VOLUME_SPIKE:.1f}x")
    logger.info(f"  Prediction offset: {PREDICTION_OFFSET} minutes")
    logger.info("")

    # === STEP 1: Get all symbols ===
    logger.info("Getting all symbols...")

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

    # === STEP 2: Generate V3 targets ===
    logger.info("Generating SLOW & LARGE targets...")
    logger.info("Looking for 15%+ moves over 60 minutes with 2x sustained volume")
    logger.info("")

    target_gen = SlowLargeSpikeTargetGenerator(
        price_window=PRICE_WINDOW,
        volume_window=VOLUME_WINDOW,
        min_price_spike=MIN_PRICE_SPIKE,
        min_volume_spike=MIN_VOLUME_SPIKE,
        prediction_offset=PREDICTION_OFFSET
    )

    all_features = []
    all_targets = []
    all_symbols = []
    all_timestamps = []

    processed = 0
    positive_count = 0

    # Feature columns
    feature_cols = [
        'returns', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14', 'NATR',
        'BB_width', 'BB_squeeze', 'VWAP_distance', 'volume_zscore', 'volume_roc',
        'OBV', 'trade_count', 'buy_sell_ratio', 'roll_measure', 'order_flow_imbalance',
        'vpin', 'bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance',
        'returns_5m', 'volume_zscore_5m', 'returns_15m', 'volume_zscore_15m'
    ]

    symbols_with_spikes = []

    for i, symbol in enumerate(symbols):
        if (i + 1) % 20 == 0:
            logger.info(f"Processing {i+1}/{len(symbols)} symbols... ({positive_count} positives so far)")

        # Get prices and volumes
        result = conn.execute(f"""
            SELECT timestamp, close, volume
            FROM candles
            WHERE symbol = '{symbol}'
            AND timestamp >= '{START_DATE}'
            AND timestamp <= '{END_DATE}'
            ORDER BY timestamp
        """).fetchdf()

        if len(result) < 200:  # Need more data for 60-min windows
            continue

        # Generate V3 targets
        prices = result['close']
        volumes = result['volume']
        targets_series = target_gen.generate_targets(prices, volumes, result['timestamp'])

        n_positives = int(targets_series.sum())
        if n_positives > 0:
            symbols_with_spikes.append((symbol, n_positives))

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

            positive_count += n_positives
            processed += 1

    conn.close()

    logger.info(f"Processed {processed} symbols")
    logger.info(f"Total positive labels (Slow & Large spikes): {positive_count}")
    logger.info("")

    # Show symbols with most spikes
    if symbols_with_spikes:
        symbols_with_spikes.sort(key=lambda x: x[1], reverse=True)
        logger.info("Top 10 symbols with Slow & Large spikes:")
        for sym, count in symbols_with_spikes[:10]:
            logger.info(f"  {sym}: {count} spikes")
        logger.info("")

    if processed == 0 or positive_count == 0:
        logger.error("No positive samples found! Try adjusting parameters.")
        return

    # === STEP 3: Create training dataset ===
    logger.info("Creating training dataset...")

    X = np.vstack(all_features)
    y = np.concatenate(all_targets)

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Total samples: {len(X):,}")
    logger.info(f"Positive samples: {int(y.sum()):,} ({y.mean()*100:.3f}%)")
    logger.info(f"Feature shape: {X.shape}")
    logger.info("")

    # Downsample negatives (50:1 ratio for rare events)
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    n_negatives_to_keep = min(len(pos_indices) * 50, len(neg_indices))
    neg_indices_sampled = np.random.choice(neg_indices, n_negatives_to_keep, replace=False)

    keep_indices = np.concatenate([pos_indices, neg_indices_sampled])
    np.random.shuffle(keep_indices)

    X = X[keep_indices]
    y = y[keep_indices]

    logger.info(f"After downsampling:")
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
    logger.info("TRAINING XGBOOST V3 (SLOW & LARGE)")
    logger.info("=" * 80)
    logger.info("")

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info(f"scale_pos_weight: {pos_weight:.1f}")
    logger.info("")

    clf = XGBClassifier(
        n_estimators=150,       # More trees for complex patterns
        max_depth=5,            # Slightly deeper
        learning_rate=0.08,     # Slightly lower LR
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

    y_val_prob = clf.predict_proba(X_val)[:, 1]

    logger.info("Validation Set:")
    for threshold in [0.3, 0.5, 0.7, 0.8]:
        y_pred = (y_val_prob >= threshold).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        tp = ((y_pred == 1) & (y_val == 1)).sum()
        fp = ((y_pred == 1) & (y_val == 0)).sum()
        fn = ((y_pred == 0) & (y_val == 1)).sum()

        logger.info(f"  Threshold {threshold:.1f}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (TP={tp}, FP={fp}, FN={fn})")

    logger.info("")

    y_test_prob = clf.predict_proba(X_test)[:, 1]

    logger.info("Test Set:")
    for threshold in [0.3, 0.5, 0.7, 0.8]:
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

    # === STEP 8: Save model ===
    logger.info("=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)
    logger.info("")

    model_path = 'models/xgboost_slow_large_v3.pkl'

    # Save model with metadata
    model_data = {
        'model': clf,
        'feature_cols': feature_cols,
        'params': {
            'price_window': PRICE_WINDOW,
            'volume_window': VOLUME_WINDOW,
            'min_price_spike': MIN_PRICE_SPIKE,
            'min_volume_spike': MIN_VOLUME_SPIKE,
            'prediction_offset': PREDICTION_OFFSET
        },
        'version': 'v3_slow_large'
    }

    joblib.dump(model_data, model_path)
    logger.info(f"✓ Model saved to {model_path}")
    logger.info("")

    # Also save just the model for compatibility
    model_only_path = 'models/xgboost_slow_large_v3_model.pkl'
    joblib.dump(clf, model_only_path)
    logger.info(f"✓ Model-only saved to {model_only_path}")
    logger.info("")

    logger.info("To use this model:")
    logger.info("  1. Update integrated_collector.py to use V3")
    logger.info("  2. Or run backtest_v3.py to compare with V1/V2")
    logger.info("")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
