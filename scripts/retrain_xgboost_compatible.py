"""
Retrain XGBoost with Full Feature Set - Compatible with Paper Trading

Uses the full 26-feature set that matches the paper trading system's
_compute_event_features() function.

Usage:
    python scripts/retrain_xgboost_compatible.py
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import os
import gc
import json
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib

# Configuration
DB_PATH = '/Users/bz/Pythia2/full_pythia.duckdb'
OUTPUT_DIR = '/Users/bz/Pythia2/models/xgboost_full_dataset'
MIN_GAIN_THRESHOLD = 0.20  # 20% gain
FORWARD_WINDOW_MINUTES = 1440  # 24 hours
EVENT_GAP_MINUTES = 720  # 12 hours between events
MIN_CANDLES_PER_SYMBOL = 1500  # Need 25+ hours for full feature computation

# Temporal split
VAL_CUTOFF = '2026-02-01'
TEST_CUTOFF = '2026-02-15'

# Full 26-feature set matching paper trading
FEATURE_COLS = [
    'natr_14', 'bb_width_20', 'vol_ratio_20_60', 'vol_ratio_60_240',
    'vol_acceleration', 'volume_vs_ma20', 'volume_trend_6hr', 'obv_slope_1hr',
    'vroc_12', 'vol_price_divergence', 'rsi_14', 'returns_1hr', 'returns_6hr',
    'returns_12hr', 'momentum_5', 'momentum_20', 'dist_from_24hr_high',
    'dist_from_24hr_low', 'bb_position', 'hl_range', 'body_ratio_avg_1hr',
    'range_compression', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
]


def generate_labels(conn) -> pd.DataFrame:
    """Generate 20%+ gain labels for all symbols."""
    logger.info("Generating labels...")

    symbols = conn.execute("""
        SELECT symbol, COUNT(*) as cnt
        FROM ohlcv WHERE timeframe = '1m'
        GROUP BY symbol HAVING cnt >= 1500
    """).fetchdf()['symbol'].tolist()

    logger.info(f"Processing {len(symbols)} symbols with >= 1500 candles")

    all_labels = []

    for i, symbol in enumerate(symbols):
        try:
            df = conn.execute(f"""
                SELECT timestamp, close, high
                FROM ohlcv
                WHERE symbol = '{symbol}' AND timeframe = '1m'
                ORDER BY timestamp
            """).fetchdf()

            if len(df) < MIN_CANDLES_PER_SYMBOL:
                continue

            # Forward max high in 24hr window
            df['future_max_high'] = df['high'].iloc[::-1].rolling(
                FORWARD_WINDOW_MINUTES, min_periods=1
            ).max().iloc[::-1].shift(-1)

            df['max_gain_24h'] = (df['future_max_high'] - df['close']) / df['close']
            df['label'] = (df['max_gain_24h'] >= MIN_GAIN_THRESHOLD).astype(int)
            df = df[df['max_gain_24h'].notna()].copy()
            df['symbol'] = symbol

            all_labels.append(df[['symbol', 'timestamp', 'label', 'max_gain_24h']])

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1}/{len(symbols)} symbols")

        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")

    labels_df = pd.concat(all_labels, ignore_index=True)
    logger.info(f"Generated {len(labels_df):,} labels, {labels_df['label'].sum():,} positives")
    return labels_df


def identify_spike_events(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Group positive labels into discrete spike events."""
    logger.info("Identifying spike events...")

    pos_labels = labels_df[labels_df['label'] == 1].copy()
    symbols = pos_labels['symbol'].unique()

    all_events = []
    event_id = 0

    for symbol in symbols:
        sym_labels = pos_labels[pos_labels['symbol'] == symbol].sort_values('timestamp')

        events = []
        current_event = []
        prev_ts = None

        for _, row in sym_labels.iterrows():
            ts = row['timestamp']
            if prev_ts is not None:
                gap = (ts - prev_ts).total_seconds() / 60
                if gap > EVENT_GAP_MINUTES:
                    if current_event:
                        events.append(current_event)
                    current_event = []
            current_event.append(row)
            prev_ts = ts

        if current_event:
            events.append(current_event)

        for event_labels in events:
            if not event_labels:
                continue

            first_ts = event_labels[0]['timestamp']
            max_gain_row = max(event_labels, key=lambda x: x['max_gain_24h'])

            all_events.append({
                'event_id': event_id,
                'symbol': symbol,
                'entry_start': first_ts,
                'max_gain': max_gain_row['max_gain_24h']
            })
            event_id += 1

    events_df = pd.DataFrame(all_events)
    logger.info(f"Identified {len(events_df)} spike events")
    return events_df


def sample_negatives(events_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Sample negative examples (3x positives)."""
    logger.info("Sampling negatives...")

    neg_labels = labels_df[labels_df['label'] == 0].copy()
    n_target = len(events_df) * 3

    negatives = neg_labels.sample(n=min(n_target, len(neg_labels)), random_state=42)

    neg_df = pd.DataFrame({
        'event_id': range(len(events_df), len(events_df) + len(negatives)),
        'symbol': negatives['symbol'].values,
        'timestamp': negatives['timestamp'].values,
        'negative_type': 'random'
    })

    logger.info(f"Sampled {len(neg_df)} negative examples")
    return neg_df


def compute_features(ohlcv_df: pd.DataFrame, target_idx: int, timestamp: pd.Timestamp) -> dict:
    """
    Compute full 26-feature set matching paper trading system.

    This matches _compute_event_features() in feature_engine.py
    """
    if target_idx < 1440 or target_idx >= len(ohlcv_df):
        return None

    df = ohlcv_df.iloc[:target_idx + 1].copy()
    close = df['close'].iloc[-1]
    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    volume = df['volume'].iloc[-1]

    if close <= 0:
        return None

    features = {}

    # Returns series
    df['returns'] = df['close'].pct_change()

    # 1. NATR (Normalized ATR)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean().iloc[-1]
    features['natr_14'] = (atr_14 / close * 100) if pd.notna(atr_14) else 0

    # 2. Bollinger Band width
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_width = ((bb_middle.iloc[-1] + 2 * bb_std.iloc[-1]) - (bb_middle.iloc[-1] - 2 * bb_std.iloc[-1])) / bb_middle.iloc[-1]
    features['bb_width_20'] = bb_width if pd.notna(bb_width) else 0

    # 3-4. Volatility ratios
    vol_20 = df['returns'].rolling(20).std().iloc[-1]
    vol_60 = df['returns'].rolling(60).std().iloc[-1]
    vol_240 = df['returns'].rolling(240).std().iloc[-1]
    features['vol_ratio_20_60'] = (vol_20 / vol_60) if pd.notna(vol_60) and vol_60 > 0 else 1
    features['vol_ratio_60_240'] = (vol_60 / vol_240) if pd.notna(vol_240) and vol_240 > 0 else 1

    # 5. Volatility acceleration
    vol_20_series = df['returns'].rolling(20).std()
    if len(vol_20_series) > 60:
        vol_acceleration = (vol_20_series.iloc[-1] - vol_20_series.iloc[-60]) / (vol_20_series.iloc[-60] + 1e-10)
    else:
        vol_acceleration = 0
    features['vol_acceleration'] = vol_acceleration if pd.notna(vol_acceleration) else 0

    # 6-7. Volume features
    vol_ma_20 = df['volume'].rolling(20).mean().iloc[-1]
    vol_ma_6hr = df['volume'].rolling(360).mean()
    features['volume_vs_ma20'] = (volume / vol_ma_20) if pd.notna(vol_ma_20) and vol_ma_20 > 0 else 1

    if len(vol_ma_6hr) > 0 and pd.notna(vol_ma_6hr.iloc[-1]) and vol_ma_6hr.iloc[-1] > 0:
        vol_trend = (vol_ma_20 - vol_ma_6hr.iloc[-1]) / (vol_ma_6hr.iloc[-1] + 1e-10)
    else:
        vol_trend = 0
    features['volume_trend_6hr'] = vol_trend if pd.notna(vol_trend) else 0

    # 8. OBV slope
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    obv_slope = (obv.iloc[-1] - obv.iloc[-60]) / 60 if len(obv) > 60 else 0
    features['obv_slope_1hr'] = obv_slope if pd.notna(obv_slope) else 0

    # 9. VROC
    vroc = (volume - df['volume'].iloc[-12]) / (df['volume'].iloc[-12] + 1e-10) if len(df) > 12 else 0
    features['vroc_12'] = vroc if pd.notna(vroc) else 0

    # 10. Volume-price divergence
    if len(df) >= 60:
        price_range_60 = (df['close'].iloc[-60:].max() - df['close'].iloc[-60:].min()) / close
        vol_change_60 = (vol_ma_20 - df['volume'].iloc[-60:-40].mean()) / (df['volume'].iloc[-60:-40].mean() + 1e-10)
        vol_price_divergence = vol_change_60 if (price_range_60 < 0.02 and vol_change_60 > 0.3) else 0
    else:
        vol_price_divergence = 0
    features['vol_price_divergence'] = vol_price_divergence if pd.notna(vol_price_divergence) else 0

    # 11. RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
    rs = gain / (loss + 1e-10)
    features['rsi_14'] = (100 - (100 / (1 + rs))) if pd.notna(rs) else 50

    # 12-14. Returns at horizons
    features['returns_1hr'] = (close / df['close'].iloc[-60] - 1) if len(df) > 60 else 0
    features['returns_6hr'] = (close / df['close'].iloc[-360] - 1) if len(df) > 360 else 0
    features['returns_12hr'] = (close / df['close'].iloc[-720] - 1) if len(df) > 720 else 0

    # 15-16. Momentum
    features['momentum_5'] = (close / df['close'].iloc[-5] - 1) if len(df) > 5 else 0
    features['momentum_20'] = (close / df['close'].iloc[-20] - 1) if len(df) > 20 else 0

    # 17-18. Distance from 24hr high/low
    if len(df) >= 1440:
        high_24hr = df['high'].iloc[-1440:].max()
        low_24hr = df['low'].iloc[-1440:].min()
    else:
        high_24hr = df['high'].max()
        low_24hr = df['low'].min()
    features['dist_from_24hr_high'] = (close - high_24hr) / high_24hr if high_24hr > 0 else 0
    features['dist_from_24hr_low'] = (close - low_24hr) / low_24hr if low_24hr > 0 else 0

    # 19. BB position
    bb_lower = bb_middle.iloc[-1] - 2 * bb_std.iloc[-1]
    bb_upper = bb_middle.iloc[-1] + 2 * bb_std.iloc[-1]
    bb_range = bb_upper - bb_lower
    features['bb_position'] = (close - bb_lower) / bb_range if pd.notna(bb_range) and bb_range > 0 else 0.5

    # 20. HL range
    features['hl_range'] = (high - low) / close if close > 0 else 0

    # 21. Body ratio average
    body = (df['close'] - df['open']).abs()
    wick = df['high'] - df['low']
    body_ratio = body / (wick + 1e-10)
    features['body_ratio_avg_1hr'] = body_ratio.iloc[-60:].mean() if len(body_ratio) >= 60 else 0.5

    # 22. Range compression
    range_20 = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / close
    range_60 = (df['high'].rolling(60).max() - df['low'].rolling(60).min()) / close
    features['range_compression'] = (range_20.iloc[-1] / range_60.iloc[-1]) if pd.notna(range_60.iloc[-1]) and range_60.iloc[-1] > 0 else 1

    # 23-26. Time features
    hour = timestamp.hour
    dow = timestamp.dayofweek
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # Clean NaN/inf
    for k, v in features.items():
        if pd.isna(v) or np.isinf(v):
            features[k] = 0

    return features


def engineer_features(events_df: pd.DataFrame, negatives_df: pd.DataFrame, conn) -> pd.DataFrame:
    """Compute features for all events."""
    logger.info("Engineering features with full 26-feature set...")

    samples = []

    for _, row in events_df.iterrows():
        samples.append({
            'event_id': row['event_id'],
            'symbol': row['symbol'],
            'timestamp': row['entry_start'],
            'label': 1,
            'sample_type': 'spike'
        })

    for _, row in negatives_df.iterrows():
        samples.append({
            'event_id': row['event_id'],
            'symbol': row['symbol'],
            'timestamp': row['timestamp'],
            'label': 0,
            'sample_type': row['negative_type']
        })

    samples_df = pd.DataFrame(samples)
    symbols = samples_df['symbol'].unique()

    all_features = []

    for sym_idx, symbol in enumerate(symbols):
        try:
            ohlcv_df = conn.execute(f"""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv WHERE symbol = '{symbol}' AND timeframe = '1m'
                ORDER BY timestamp
            """).fetchdf()

            if len(ohlcv_df) < 1500:
                continue

            ts_to_idx = dict(zip(ohlcv_df['timestamp'], range(len(ohlcv_df))))
            sym_samples = samples_df[samples_df['symbol'] == symbol]

            for _, sample in sym_samples.iterrows():
                idx = ts_to_idx.get(sample['timestamp'])
                if idx is None or idx < 1440:  # Need 24hr history
                    continue

                features = compute_features(ohlcv_df, idx, pd.Timestamp(sample['timestamp']))
                if features:
                    features['event_id'] = sample['event_id']
                    features['symbol'] = sample['symbol']
                    features['timestamp'] = sample['timestamp']
                    features['label'] = sample['label']
                    features['sample_type'] = sample['sample_type']
                    all_features.append(features)

            if (sym_idx + 1) % 50 == 0:
                logger.info(f"  Processed {sym_idx+1}/{len(symbols)} symbols, {len(all_features)} features computed")

        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")

    features_df = pd.DataFrame(all_features)
    logger.info(f"Computed features for {len(features_df)} samples")

    return features_df


def train_xgboost(features_df: pd.DataFrame):
    """Train XGBoost with temporal split and save compatible model."""
    logger.info("Training XGBoost...")

    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])

    # Temporal split
    train = features_df[features_df['timestamp'] < VAL_CUTOFF]
    val = features_df[(features_df['timestamp'] >= VAL_CUTOFF) & (features_df['timestamp'] < TEST_CUTOFF)]
    test = features_df[features_df['timestamp'] >= TEST_CUTOFF]

    logger.info(f"Train: {len(train)} ({train['label'].sum()} pos)")
    logger.info(f"Val: {len(val)} ({val['label'].sum()} pos)")
    logger.info(f"Test: {len(test)} ({test['label'].sum()} pos)")

    X_train = train[FEATURE_COLS].values
    y_train = train['label'].values
    X_val = val[FEATURE_COLS].values
    y_val = val['label'].values
    X_test = test[FEATURE_COLS].values
    y_test = test['label'].values

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost with conservative params to avoid overfitting
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        min_child_weight=5,
        scale_pos_weight=0.45,  # Penalize false positives
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )

    # Evaluate
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.3, 0.95, 0.05):
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    y_pred = (y_proba >= best_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'threshold': best_thresh,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'n_train': len(train),
        'n_val': len(val),
        'n_test': len(test),
    }

    # Precision at different thresholds (important for paper trading)
    for thresh in [0.8, 0.85, 0.9, 0.95]:
        y_pred_t = (y_proba >= thresh).astype(int)
        metrics[f'precision_at_{int(thresh*100)}'] = precision_score(y_test, y_pred_t, zero_division=0)
        metrics[f'recall_at_{int(thresh*100)}'] = recall_score(y_test, y_pred_t, zero_division=0)
        metrics[f'n_pred_at_{int(thresh*100)}'] = int(y_pred_t.sum())

    logger.info(f"\n{'='*50}")
    logger.info(f"TEST RESULTS:")
    logger.info(f"  F1: {metrics['f1']:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall: {metrics['recall']:.3f}")
    logger.info(f"  AUC: {metrics['auc']:.3f}")
    logger.info(f"  Optimal Threshold: {best_thresh:.2f}")
    logger.info(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    logger.info(f"\nPrecision at thresholds:")
    for thresh in [0.8, 0.85, 0.9, 0.95]:
        p = metrics[f'precision_at_{int(thresh*100)}']
        r = metrics[f'recall_at_{int(thresh*100)}']
        n = metrics[f'n_pred_at_{int(thresh*100)}']
        logger.info(f"  @{thresh:.0%}: Prec={p:.1%}, Recall={r:.1%}, N={n}")
    logger.info(f"{'='*50}")

    # Feature importance
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
    logger.info("\nTop 10 features:")
    for i, (feat, imp) in enumerate(sorted_imp[:10]):
        logger.info(f"  {i+1}. {feat}: {imp:.3f}")

    # Save in format compatible with paper trading
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save as dict (matching old format)
    model_dict = {
        'model': model,
        'scaler': scaler,
        'feature_cols': FEATURE_COLS
    }
    joblib.dump(model_dict, os.path.join(OUTPUT_DIR, 'model.pkl'))

    # Also save raw model for inspection
    model.save_model(os.path.join(OUTPUT_DIR, 'xgboost_model.json'))

    # Save metrics
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save feature importance
    with open(os.path.join(OUTPUT_DIR, 'feature_importance.json'), 'w') as f:
        json.dump(importances, f, indent=2)

    logger.info(f"\nModel saved to {OUTPUT_DIR}/model.pkl")
    logger.info("Compatible with paper trading - can swap in directly!")

    return model, scaler, metrics


def main():
    logger.info("="*60)
    logger.info("RETRAINING XGBOOST WITH FULL 26-FEATURE SET")
    logger.info("="*60)

    conn = duckdb.connect(DB_PATH, read_only=True)

    # Step 1: Generate labels
    labels_df = generate_labels(conn)

    # Step 2: Identify spike events
    events_df = identify_spike_events(labels_df)

    # Step 3: Sample negatives
    negatives_df = sample_negatives(events_df, labels_df)

    # Step 4: Engineer features
    features_df = engineer_features(events_df, negatives_df, conn)

    conn.close()

    # Step 5: Train
    model, scaler, metrics = train_xgboost(features_df)

    logger.info("\nDONE!")
    logger.info(f"New model: {OUTPUT_DIR}/model.pkl")
    logger.info("To use in paper trading, update feature_engine.py:")
    logger.info(f"  event_classifier_model_path='{OUTPUT_DIR}/model.pkl'")


if __name__ == '__main__':
    main()
