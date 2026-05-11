#!/usr/bin/env python3
"""
Train XGBoost Fast-vA - FAST & BIG Spike Detection

This is SEPARATE from the Slow & Large system (V1/V2/V3).
Fast & Big uses letter versioning (vA, vB, vC...) to avoid confusion.

Target: Quick explosive moves (10-25%+ gains in 5-30 minutes)
NOT: Slow multi-hour runners (that's the Slow & Large system)

Parameters:
  - price_window: 10 minutes - fast peaks
  - min_price_spike: 10% - big move
  - volume_window: 3 minutes - explosive not sustained
  - min_volume_spike: 5.0x - instant explosion
  - prediction_offset: 2 minutes - quick entry needed

Examples of target spikes (from SPIKE_TYPES.md):
  - CTX-USD: 23.5% peak in 16 min
  - FLOKI-USD: 20.5% peak in 7 min
  - AST-USD: 15.4% peak in 4 min

Trading strategy for Fast & Big:
  - Enter quickly (within 2 min of signal)
  - Tight stop-loss (1-2%)
  - Exit within 30 minutes
  - Target: 10-25% gain

Usage:
    python scripts/fast_big/train_xgboost_fast_vA.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import time


class FastBigSpikeTargetGenerator:
    """
    Generate targets for FAST & BIG spikes (explosive short-term moves).

    Different from V3 which targets slow multi-hour runners.
    This targets explosive moves that peak within 5-30 minutes.
    """

    def __init__(
        self,
        price_window: int = 10,       # 10 minutes to look ahead for price
        volume_window: int = 3,        # 3 minutes for volume explosion
        min_price_spike: float = 0.10, # 10% minimum move
        min_volume_spike: float = 5.0, # 5x instant volume explosion
        prediction_offset: int = 2     # 2 minutes early warning (fast entry needed)
    ):
        """
        Initialize Fast & Big spike detector.

        Args:
            price_window: Minutes to look forward for price move (default: 10)
            volume_window: Minutes to look forward for volume (default: 3)
            min_price_spike: Minimum price increase (default: 10%)
            min_volume_spike: Minimum volume multiplier (default: 5x)
            prediction_offset: Minutes to offset prediction (default: 2)
        """
        self.price_window = int(price_window)
        self.volume_window = int(volume_window)
        self.min_price_spike = min_price_spike
        self.min_volume_spike = min_volume_spike
        self.prediction_offset = int(prediction_offset)

        logger.info(
            f"FastBigSpikeTargetGenerator initialized",
            extra={
                "price_window": f"{self.price_window} minutes",
                "volume_window": f"{self.volume_window} minutes",
                "min_price_spike": f"{min_price_spike*100:.0f}%",
                "min_volume_spike": f"{min_volume_spike:.1f}x",
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
        Generate binary targets for Fast & Big spikes.

        Criteria:
        1. Price rises 10%+ in the next 10 minutes
        2. Volume explodes 5x+ in the next 3 minutes
        3. Signal fires 2 minutes BEFORE the explosion

        Args:
            prices: Close prices (1-minute candles)
            volumes: Trade volumes (1-minute candles)
            timestamps: Corresponding timestamps (optional)

        Returns:
            pd.Series: Binary targets (1 = spike coming, 0 = normal)
        """
        n = len(prices)
        targets = np.zeros(n, dtype=int)

        price_arr = prices.values
        volume_arr = volumes.values

        # Calculate rolling average volume for baseline
        vol_window = min(20, n // 4)
        vol_avg = pd.Series(volume_arr).rolling(window=vol_window, min_periods=1).mean().values

        offset = self.prediction_offset
        price_window = self.price_window
        volume_window = self.volume_window

        positive_count = 0

        for i in range(n - offset - max(price_window, volume_window) - 1):
            # FIX: Compare against price at candle i (NOW), not future_idx
            # We want to predict spikes BEFORE they happen, not during
            current_price = price_arr[i]

            # Look ahead: price should spike within offset + price_window minutes
            future_end = i + offset + price_window
            if future_end < n and current_price > 0:
                # Check max price from NOW to end of window
                future_prices = price_arr[i + 1:future_end + 1]

                if len(future_prices) > 0:
                    max_future_price = np.max(future_prices)
                    price_change = (max_future_price - current_price) / current_price

                    # Check volume explosion in the near future (offset to offset + volume_window)
                    vol_start = i + offset
                    vol_end = vol_start + volume_window
                    if vol_end < n:
                        future_volumes = volume_arr[vol_start:vol_end + 1]
                        avg_vol = vol_avg[i]

                        if len(future_volumes) > 0 and avg_vol > 0:
                            max_future_vol = np.max(future_volumes)
                            vol_ratio = max_future_vol / avg_vol

                            # Both conditions must be met
                            if price_change >= self.min_price_spike and vol_ratio >= self.min_volume_spike:
                                targets[i] = 1
                                positive_count += 1

        logger.info(f"Generated {positive_count} positive targets out of {n} samples ({positive_count/n*100:.3f}%)")
        return pd.Series(targets, index=prices.index)


def main():
    logger.info("=" * 80)
    logger.info("TRAINING XGBOOST FAST-vA - FAST & BIG SPIKE DETECTOR")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Target: Explosive 10-25% moves in 5-30 minutes")
    logger.info("Examples: CTX-USD +23% in 16min, FLOKI-USD +21% in 7min")
    logger.info("")

    # === CONFIGURATION ===
    DB_PATH = 'market_data.duckdb'
    MODEL_OUTPUT_PATH = 'models/fast_big/xgboost_fast_vA.pkl'

    # Date range - same as V3 for comparison
    START_DATE = '2025-10-06'
    END_DATE = '2025-10-20'

    # V4 Fast & Big parameters
    PRICE_WINDOW = 10        # 10 min (fast peak)
    VOLUME_WINDOW = 3        # 3 min (explosive)
    MIN_PRICE_SPIKE = 0.10   # 10% move
    MIN_VOLUME_SPIKE = 5.0   # 5x explosion
    PREDICTION_OFFSET = 2    # 2 min early warning

    logger.info(f"Fast & Big Parameters:")
    logger.info(f"  Price window:     {PRICE_WINDOW} minutes")
    logger.info(f"  Volume window:    {VOLUME_WINDOW} minutes")
    logger.info(f"  Min price spike:  {MIN_PRICE_SPIKE*100:.0f}%")
    logger.info(f"  Min volume spike: {MIN_VOLUME_SPIKE}x")
    logger.info(f"  Prediction offset: {PREDICTION_OFFSET} minutes")
    logger.info("")

    # Feature columns (same as V1/V2/V3)
    feature_cols = [
        'returns', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14', 'NATR',
        'BB_width', 'BB_squeeze', 'VWAP_distance', 'volume_zscore', 'volume_roc',
        'OBV', 'trade_count', 'buy_sell_ratio', 'roll_measure', 'order_flow_imbalance',
        'vpin', 'bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance',
        'returns_5m', 'volume_zscore_5m', 'returns_15m', 'volume_zscore_15m'
    ]

    # === STEP 1: Load OHLCV data ===
    logger.info("Loading OHLCV data from database...")

    conn = duckdb.connect(DB_PATH, read_only=True)

    # Use 'candles' table for historical training data (Oct 6-20)
    # 'ohlcv' table has recent live data only
    # Note: candles table is all 1m data (no timeframe column)
    ohlcv_query = f"""
        SELECT symbol, timestamp, close, volume
        FROM candles
        WHERE timestamp >= '{START_DATE}'
        AND timestamp <= '{END_DATE}'
        ORDER BY symbol, timestamp
    """

    ohlcv_df = conn.execute(ohlcv_query).fetchdf()
    logger.info(f"Loaded {len(ohlcv_df):,} OHLCV rows")
    logger.info(f"Symbols: {ohlcv_df['symbol'].nunique()}")
    logger.info("")

    # === STEP 2: Generate targets per symbol ===
    logger.info("Generating Fast & Big spike targets...")

    target_gen = FastBigSpikeTargetGenerator(
        price_window=PRICE_WINDOW,
        volume_window=VOLUME_WINDOW,
        min_price_spike=MIN_PRICE_SPIKE,
        min_volume_spike=MIN_VOLUME_SPIKE,
        prediction_offset=PREDICTION_OFFSET
    )

    all_targets = []
    symbols_with_spikes = []

    for symbol in ohlcv_df['symbol'].unique():
        symbol_data = ohlcv_df[ohlcv_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('timestamp')

        if len(symbol_data) < 100:
            continue

        targets = target_gen.generate_targets(
            prices=symbol_data['close'],
            volumes=symbol_data['volume'],
            timestamps=symbol_data['timestamp']
        )

        symbol_data['target'] = targets.values
        symbol_data['symbol'] = symbol

        positives = targets.sum()
        if positives > 0:
            symbols_with_spikes.append((symbol, positives))

        all_targets.append(symbol_data[['symbol', 'timestamp', 'target']])

    targets_df = pd.concat(all_targets, ignore_index=True)

    total_positives = targets_df['target'].sum()
    logger.info(f"Total positive targets: {total_positives:,}")
    logger.info(f"Symbols with spikes: {len(symbols_with_spikes)}")
    logger.info("")

    if total_positives == 0:
        logger.error("No positive targets found! Adjust parameters.")
        return

    # Show top symbols
    symbols_with_spikes.sort(key=lambda x: x[1], reverse=True)
    logger.info("Top 10 symbols by spike count:")
    for sym, count in symbols_with_spikes[:10]:
        logger.info(f"  {sym}: {count}")
    logger.info("")

    # === STEP 3: Load features ===
    logger.info("Loading features from database...")

    features_query = f"""
        SELECT
            symbol,
            timestamp,
            {', '.join(feature_cols)}
        FROM features
        WHERE timeframe = '1m'
        AND timestamp >= '{START_DATE}'
        AND timestamp <= '{END_DATE}'
    """

    features_df = conn.execute(features_query).fetchdf()
    conn.close()

    logger.info(f"Loaded {len(features_df):,} feature rows")
    logger.info("")

    # === STEP 4: Merge features with targets ===
    logger.info("Merging features with targets...")

    merged = features_df.merge(
        targets_df,
        on=['symbol', 'timestamp'],
        how='inner'
    )

    logger.info(f"Merged dataset: {len(merged):,} rows")
    logger.info(f"Positive samples: {merged['target'].sum():,} ({merged['target'].mean()*100:.3f}%)")
    logger.info("")

    # === STEP 5: Prepare training data ===
    logger.info("Preparing training data...")

    # Handle missing values
    X = merged[feature_cols].values
    y = merged['target'].values

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Downsample negatives to balance (keep all positives)
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    # Keep 10x negatives for every positive
    n_neg_keep = min(len(neg_indices), len(pos_indices) * 10)
    np.random.seed(42)
    neg_indices_sample = np.random.choice(neg_indices, size=n_neg_keep, replace=False)

    keep_indices = np.concatenate([pos_indices, neg_indices_sample])
    np.random.shuffle(keep_indices)

    X_balanced = X[keep_indices]
    y_balanced = y[keep_indices]

    logger.info(f"Balanced dataset: {len(y_balanced):,} samples")
    logger.info(f"Positive rate: {y_balanced.mean()*100:.2f}%")
    logger.info("")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    logger.info(f"Training set: {len(X_train):,} samples ({y_train.sum():,} positive)")
    logger.info(f"Test set: {len(X_test):,} samples ({y_test.sum():,} positive)")
    logger.info("")

    # === STEP 6: Train XGBoost ===
    logger.info("Training XGBoost model...")
    start_time = time.time()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,  # Already balanced
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    train_time = time.time() - start_time
    logger.info(f"Training complete in {train_time:.1f}s")
    logger.info("")

    # === STEP 7: Evaluate ===
    logger.info("=" * 80)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info("")

    y_proba = model.predict_proba(X_test)[:, 1]

    # Test multiple thresholds
    thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]

    logger.info(f"{'Threshold':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Alerts':>10}")
    logger.info("-" * 55)

    best_f1 = 0
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        alerts = y_pred.sum()

        logger.info(f"{thresh:<12.1f} {prec*100:>9.1f}% {rec*100:>9.1f}% {f1*100:>9.1f}% {alerts:>10}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    logger.info("")
    logger.info(f"Best F1: {best_f1*100:.1f}% at threshold {best_threshold}")
    logger.info("")

    # Confusion matrix at 0.7 threshold
    y_pred_70 = (y_proba >= 0.7).astype(int)
    cm = confusion_matrix(y_test, y_pred_70)
    logger.info("Confusion Matrix (threshold=0.7):")
    logger.info(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    logger.info(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    logger.info("")

    # === STEP 8: Feature importance ===
    logger.info("Top 10 Feature Importances:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']:<25} {row['importance']*100:>6.2f}%")
    logger.info("")

    # === STEP 9: Save model ===
    logger.info("Saving model...")

    model_data = {
        'model': model,
        'feature_columns': feature_cols,
        'thresholds': {
            'recommended': 0.7,
            'high_precision': 0.9,
            'high_recall': 0.5
        },
        'parameters': {
            'price_window': PRICE_WINDOW,
            'volume_window': VOLUME_WINDOW,
            'min_price_spike': MIN_PRICE_SPIKE,
            'min_volume_spike': MIN_VOLUME_SPIKE,
            'prediction_offset': PREDICTION_OFFSET
        },
        'training_info': {
            'date_range': f"{START_DATE} to {END_DATE}",
            'total_samples': len(y_balanced),
            'positive_samples': int(y_balanced.sum()),
            'best_f1': best_f1,
            'best_threshold': best_threshold
        }
    }

    # Save full model with metadata
    joblib.dump(model_data, MODEL_OUTPUT_PATH)
    logger.info(f"Full model saved to {MODEL_OUTPUT_PATH}")

    # Save model only (for compatibility)
    model_only_path = MODEL_OUTPUT_PATH.replace('.pkl', '_model.pkl')
    joblib.dump(model, model_only_path)
    logger.info(f"Model only saved to {model_only_path}")
    logger.info("")

    # === STEP 10: Summary ===
    logger.info("=" * 80)
    logger.info("FAST-vA TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Model detects explosive short-term spikes:")
    logger.info(f"  - 10%+ price move in {PRICE_WINDOW} minutes")
    logger.info(f"  - 5x+ volume explosion in {VOLUME_WINDOW} minutes")
    logger.info(f"  - {PREDICTION_OFFSET} minute early warning")
    logger.info("")
    logger.info("Recommended usage:")
    logger.info("  - Threshold 0.7 for balanced precision/recall")
    logger.info("  - Threshold 0.9 for high-confidence only")
    logger.info("  - Exit within 30 minutes of entry")
    logger.info("  - Tight stop-loss (1-2%)")
    logger.info("")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
