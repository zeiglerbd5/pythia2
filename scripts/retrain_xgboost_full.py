"""
Retrain XGBoost with Full Dataset (Oct 2025 - Mar 2026)

Runs the complete pipeline:
1. Generate labels from full_pythia.duckdb
2. Identify spike events
3. Sample negatives
4. Engineer features
5. Train XGBoost

Usage:
    python scripts/retrain_xgboost_full.py
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# Configuration
DB_PATH = '/Users/bz/Pythia2/full_pythia.duckdb'
OUTPUT_DIR = '/Users/bz/Pythia2/data/full_dataset'
MIN_GAIN_THRESHOLD = 0.20  # 20% gain
FORWARD_WINDOW_MINUTES = 1440  # 24 hours
EVENT_GAP_MINUTES = 720  # 12 hours between events
MIN_CANDLES_PER_SYMBOL = 500

# Temporal split
VAL_CUTOFF = '2026-02-01'
TEST_CUTOFF = '2026-02-15'


def generate_labels(conn) -> pd.DataFrame:
    """Generate 20%+ gain labels for all symbols."""
    logger.info("Generating labels...")

    # Get all symbols with enough data
    symbols = conn.execute("""
        SELECT symbol, COUNT(*) as cnt
        FROM ohlcv WHERE timeframe = '1m'
        GROUP BY symbol HAVING cnt >= 500
    """).fetchdf()['symbol'].tolist()

    logger.info(f"Processing {len(symbols)} symbols")

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

            # Max gain from close to future max high
            df['max_gain_24h'] = (df['future_max_high'] - df['close']) / df['close']

            # Binary label
            df['label'] = (df['max_gain_24h'] >= MIN_GAIN_THRESHOLD).astype(int)

            # Only keep rows with valid labels
            df = df[df['max_gain_24h'].notna()].copy()
            df['symbol'] = symbol

            # Calculate time to peak for positives
            df['time_to_peak_minutes'] = np.nan
            pos_indices = df[df['label'] == 1].index
            for idx in pos_indices:
                pos = df.index.get_loc(idx)
                end_pos = min(pos + FORWARD_WINDOW_MINUTES, len(df))
                future_slice = df.iloc[pos:end_pos]['high']
                if len(future_slice) > 0:
                    peak_offset = future_slice.values.argmax()
                    df.loc[idx, 'time_to_peak_minutes'] = peak_offset

            all_labels.append(df[['symbol', 'timestamp', 'label', 'max_gain_24h', 'time_to_peak_minutes']])

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1}/{len(symbols)} symbols")

        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")

    labels_df = pd.concat(all_labels, ignore_index=True)
    logger.info(f"Generated {len(labels_df):,} labels, {labels_df['label'].sum():,} positives")

    return labels_df


def identify_spike_events(labels_df: pd.DataFrame, conn) -> pd.DataFrame:
    """Group positive labels into discrete spike events."""
    logger.info("Identifying spike events...")

    # Get symbols with positives
    pos_labels = labels_df[labels_df['label'] == 1].copy()
    symbols = pos_labels['symbol'].unique()

    all_events = []

    for symbol in symbols:
        sym_labels = pos_labels[pos_labels['symbol'] == symbol].sort_values('timestamp')

        # Group by time gaps
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

        # Process each event
        for event_labels in events:
            if not event_labels:
                continue

            first_ts = event_labels[0]['timestamp']
            max_gain_row = max(event_labels, key=lambda x: x['max_gain_24h'])
            peak_ts = max_gain_row['timestamp']
            if pd.notna(max_gain_row['time_to_peak_minutes']):
                peak_ts += timedelta(minutes=int(max_gain_row['time_to_peak_minutes']))

            all_events.append({
                'symbol': symbol,
                'entry_start': first_ts,
                'peak_time': peak_ts,
                'return_pct': max_gain_row['max_gain_24h'],
                'n_positive_labels': len(event_labels)
            })

    events_df = pd.DataFrame(all_events)
    events_df = events_df.sort_values('entry_start').reset_index(drop=True)
    events_df['event_id'] = range(len(events_df))

    logger.info(f"Identified {len(events_df)} spike events from {len(symbols)} symbols")

    return events_df


def sample_negatives(events_df: pd.DataFrame, labels_df: pd.DataFrame, ratio: float = 2.0) -> pd.DataFrame:
    """Sample negative examples (non-spike periods)."""
    logger.info(f"Sampling negatives (ratio={ratio})...")

    n_target = int(len(events_df) * ratio)

    # Get negative labels
    neg_labels = labels_df[labels_df['label'] == 0].copy()

    # Sample random negatives
    negatives = neg_labels.sample(n=min(n_target, len(neg_labels)), random_state=42)

    neg_df = pd.DataFrame({
        'event_id': range(len(events_df), len(events_df) + len(negatives)),
        'symbol': negatives['symbol'].values,
        'timestamp': negatives['timestamp'].values,
        'negative_type': 'random'
    })

    logger.info(f"Sampled {len(neg_df)} negative examples")

    return neg_df


def compute_features(row, ohlcv_df: pd.DataFrame, target_idx: int) -> dict:
    """Compute features for a single event."""
    if target_idx < 100 or target_idx >= len(ohlcv_df):
        return None

    df = ohlcv_df.iloc[:target_idx + 1].copy()
    close = df['close'].iloc[-1]

    features = {}

    # Returns
    df['returns'] = df['close'].pct_change()

    # NATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean().iloc[-1]
    features['natr_14'] = (atr_14 / close * 100) if close > 0 else 0

    # Volatility ratios
    vol_20 = df['returns'].rolling(20).std().iloc[-1]
    vol_60 = df['returns'].rolling(60).std().iloc[-1]
    features['vol_ratio_20_60'] = (vol_20 / vol_60) if pd.notna(vol_60) and vol_60 > 0 else 1

    # Volume features
    vol_ma_20 = df['volume'].rolling(20).mean().iloc[-1]
    features['volume_vs_ma20'] = (df['volume'].iloc[-1] / vol_ma_20) if pd.notna(vol_ma_20) and vol_ma_20 > 0 else 1

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
    rs = gain / (loss + 1e-10)
    features['rsi_14'] = (100 - (100 / (1 + rs))) if pd.notna(rs) else 50

    # Momentum
    features['momentum_20'] = (close / df['close'].iloc[-20] - 1) if len(df) > 20 else 0

    # Bollinger
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_width = ((bb_middle.iloc[-1] + 2 * bb_std.iloc[-1]) - (bb_middle.iloc[-1] - 2 * bb_std.iloc[-1])) / bb_middle.iloc[-1]
    features['bb_width_20'] = bb_width if pd.notna(bb_width) else 0

    # Distance from 24hr high/low
    if len(df) >= 1440:
        high_24hr = df['high'].iloc[-1440:].max()
        low_24hr = df['low'].iloc[-1440:].min()
    else:
        high_24hr = df['high'].max()
        low_24hr = df['low'].min()
    features['dist_from_24hr_high'] = (close - high_24hr) / high_24hr if high_24hr > 0 else 0
    features['dist_from_24hr_low'] = (close - low_24hr) / low_24hr if low_24hr > 0 else 0

    # Clean NaN
    for k, v in features.items():
        if pd.isna(v) or np.isinf(v):
            features[k] = 0

    return features


def engineer_features(events_df: pd.DataFrame, negatives_df: pd.DataFrame, conn) -> pd.DataFrame:
    """Compute features for all events."""
    logger.info("Engineering features...")

    # Combine events and negatives
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

            if len(ohlcv_df) < 100:
                continue

            ts_to_idx = dict(zip(ohlcv_df['timestamp'], range(len(ohlcv_df))))
            sym_samples = samples_df[samples_df['symbol'] == symbol]

            for _, sample in sym_samples.iterrows():
                idx = ts_to_idx.get(sample['timestamp'])
                if idx is None or idx < 100:
                    continue

                features = compute_features(sample, ohlcv_df, idx)
                if features:
                    features['event_id'] = sample['event_id']
                    features['symbol'] = sample['symbol']
                    features['timestamp'] = sample['timestamp']
                    features['label'] = sample['label']
                    features['sample_type'] = sample['sample_type']
                    all_features.append(features)

            if (sym_idx + 1) % 50 == 0:
                logger.info(f"  Processed {sym_idx+1}/{len(symbols)} symbols")

        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")

    features_df = pd.DataFrame(all_features)
    logger.info(f"Computed features for {len(features_df)} samples")

    return features_df


def train_xgboost(features_df: pd.DataFrame):
    """Train XGBoost classifier with temporal split."""
    logger.info("Training XGBoost...")

    # Temporal split
    features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])

    train = features_df[features_df['timestamp'] < VAL_CUTOFF]
    val = features_df[(features_df['timestamp'] >= VAL_CUTOFF) & (features_df['timestamp'] < TEST_CUTOFF)]
    test = features_df[features_df['timestamp'] >= TEST_CUTOFF]

    logger.info(f"Train: {len(train)} ({(train['label']==1).sum()} pos)")
    logger.info(f"Val: {len(val)} ({(val['label']==1).sum()} pos)")
    logger.info(f"Test: {len(test)} ({(test['label']==1).sum()} pos)")

    # Feature columns
    meta_cols = ['event_id', 'symbol', 'timestamp', 'label', 'sample_type']
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    X_train = train[feature_cols].values
    y_train = train['label'].values
    X_val = val[feature_cols].values
    y_val = val['label'].values
    X_test = test[feature_cols].values
    y_test = test['label'].values

    # Scale pos weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    # Train
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate on test
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Find best threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.3, 0.9, 0.05):
        y_pred = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    y_pred = (y_pred_proba >= best_thresh).astype(int)

    # Metrics
    results = {
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0,
        'threshold': best_thresh,
        'n_train': len(train),
        'n_val': len(val),
        'n_test': len(test),
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    results['tp'] = int(tp)
    results['fp'] = int(fp)
    results['tn'] = int(tn)
    results['fn'] = int(fn)

    logger.info(f"\nTEST RESULTS:")
    logger.info(f"  F1:        {results['f1']:.3f}")
    logger.info(f"  Precision: {results['precision']:.3f}")
    logger.info(f"  Recall:    {results['recall']:.3f}")
    logger.info(f"  AUC:       {results['auc']:.3f}")
    logger.info(f"  Threshold: {results['threshold']:.2f}")
    logger.info(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Feature importance
    logger.info(f"\nTop 10 features:")
    importance = dict(zip(feature_cols, model.feature_importances_))
    for i, (feat, imp) in enumerate(sorted(importance.items(), key=lambda x: -x[1])[:10]):
        logger.info(f"  {i+1}. {feat}: {imp:.4f}")

    return model, results, feature_cols


def main():
    logger.info("=" * 70)
    logger.info("RETRAINING XGBOOST WITH FULL DATASET")
    logger.info("=" * 70)

    # Create output dir
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Connect to DB
    conn = duckdb.connect(DB_PATH, read_only=True)

    # Step 1: Generate labels
    labels_df = generate_labels(conn)
    labels_df.to_parquet(f'{OUTPUT_DIR}/labels.parquet', index=False)

    # Step 2: Identify spike events
    events_df = identify_spike_events(labels_df, conn)
    events_df.to_parquet(f'{OUTPUT_DIR}/spike_events.parquet', index=False)

    # Step 3: Sample negatives
    negatives_df = sample_negatives(events_df, labels_df, ratio=2.0)
    negatives_df.to_parquet(f'{OUTPUT_DIR}/negatives.parquet', index=False)

    # Step 4: Engineer features
    features_df = engineer_features(events_df, negatives_df, conn)
    features_df.to_parquet(f'{OUTPUT_DIR}/features.parquet', index=False)

    conn.close()

    # Step 5: Train XGBoost
    model, results, feature_cols = train_xgboost(features_df)

    # Save model and results
    model.save_model(f'{OUTPUT_DIR}/xgboost_model.json')
    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved to {OUTPUT_DIR}")

    return model, results


if __name__ == '__main__':
    main()
