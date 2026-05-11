#!/usr/bin/env python3
"""
XGBoost Minimal - Direct Database Query
Bypasses DatasetBuilder entirely to avoid memory issues.
Only loads the features we actually need (last timestep).
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import duckdb
import joblib
import os


def main():
    # === CONFIGURATION ===
    DB_PATH = '/Users/brettzeigler/Pythia/market_data.duckdb'
    CATEGORIZED_CSV = '/Users/brettzeigler/Pythia/all_spikes_categorized.csv'
    
    # How many negative samples per positive (to control memory)
    NEG_RATIO = 100  # 100 negatives per positive = manageable dataset
    
    logger.info("=" * 80)
    logger.info("XGBOOST MINIMAL - DIRECT DATABASE QUERY")
    logger.info("=" * 80)
    logger.info("")

    # === STEP 1: Load spike timestamps from CSV ===
    logger.info("Loading Slow & Large spike timestamps...")
    df_spikes = pd.read_csv(CATEGORIZED_CSV)
    df_slow_large = df_spikes[df_spikes['category'] == 'Slow & Large'].copy()
    
    logger.info(f"Found {len(df_slow_large)} Slow & Large spikes")
    logger.info("")
    
    # === STEP 2: Get feature columns from database ===
    logger.info("Connecting to database...")
    conn = duckdb.connect(DB_PATH, read_only=True)
    
    # Get column names from features table
    schema = conn.execute("DESCRIBE features").fetchall()
    all_columns = [row[0] for row in schema]
    
    # Feature columns (exclude metadata)
    exclude_cols = ['symbol', 'timestamp', 'timeframe', 'open', 'high', 'low', 'close', 
                    'volume', 'is_spike', 'spike_return_1m', 'spike_return_3m', 
                    'spike_return_5m', 'spike_return_10m']
    feature_cols = [c for c in all_columns if c not in exclude_cols]
    
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}...")
    logger.info("")
    
    # === STEP 3: Load POSITIVE samples (spikes) ===
    logger.info("Loading positive samples (spike moments)...")
    
    positives = []
    for _, row in df_slow_large.iterrows():
        symbol = row['symbol']
        timestamp = row['timestamp']
        
        # Get features at this exact timestamp
        query = f"""
            SELECT {', '.join(feature_cols)}
            FROM features 
            WHERE symbol = '{symbol}' 
            AND timestamp = '{timestamp}'
            AND timeframe = '1m'
            LIMIT 1
        """
        result = conn.execute(query).fetchone()
        if result:
            positives.append(list(result))
    
    logger.info(f"Loaded {len(positives)} positive samples")
    
    if len(positives) == 0:
        logger.error("No positive samples found! Check timestamp format.")
        conn.close()
        return
    
    # === STEP 4: Load NEGATIVE samples (random non-spike moments) ===
    n_negatives = len(positives) * NEG_RATIO
    logger.info(f"Loading {n_negatives} negative samples (random non-spike moments)...")
    
    # Get random negative samples
    # Exclude timestamps that are spikes
    spike_timestamps = df_slow_large['timestamp'].tolist()
    
    # Sample random rows (will filter out spike timestamps manually)
    query = f"""
        SELECT {', '.join(feature_cols)}, symbol, timestamp
        FROM features
        WHERE timeframe = '1m'
        ORDER BY RANDOM()
        LIMIT {n_negatives * 2}
    """
    result = conn.execute(query).fetchall()

    # Filter out any spike timestamps and extract only feature columns
    spike_set = set(df_slow_large[['symbol', 'timestamp']].itertuples(index=False, name=None))
    negatives = []
    for row in result:
        # Last two columns are symbol, timestamp
        symbol, timestamp = row[-2], row[-1]
        if (symbol, str(timestamp)) not in spike_set:
            # Keep only feature columns (all but last 2)
            negatives.append(list(row[:-2]))
        if len(negatives) >= n_negatives:
            break

    logger.info(f"Loaded {len(negatives)} negative samples (filtered out {len(result) - len(negatives)} spike timestamps)")
    logger.info("")

    conn.close()
    
    # === STEP 5: Create training data ===
    logger.info("Creating training dataset...")
    
    X_pos = np.array(positives, dtype=np.float32)
    X_neg = np.array(negatives, dtype=np.float32)
    
    y_pos = np.ones(len(X_pos), dtype=np.int32)
    y_neg = np.zeros(len(X_neg), dtype=np.int32)
    
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])
    
    # Handle NaN/inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info(f"Total samples: {len(X)} ({len(X_pos)} pos, {len(X_neg)} neg)")
    logger.info(f"Feature shape: {X.shape}")
    logger.info(f"Memory: {X.nbytes / 1024 / 1024:.1f} MB")
    logger.info("")
    
    # === STEP 6: Train/val/test split ===
    logger.info("Splitting data...")
    
    # Stratified split to maintain class balance
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Train: {len(y_train)} ({y_train.sum()} pos)")
    logger.info(f"Val:   {len(y_val)} ({y_val.sum()} pos)")
    logger.info(f"Test:  {len(y_test)} ({y_test.sum()} pos)")
    logger.info("")
    
    # === STEP 7: Train XGBoost ===
    logger.info("=" * 80)
    logger.info("TRAINING XGBOOST")
    logger.info("=" * 80)
    logger.info("")
    
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info(f"scale_pos_weight: {pos_weight:.1f}")
    
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
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    logger.info("")
    logger.info("Training complete!")
    logger.info("")
    
    # === STEP 8: Evaluate ===
    logger.info("=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)
    logger.info("")
    
    y_val_prob = clf.predict_proba(X_val)[:, 1]
    
    # Try multiple thresholds
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        y_pred = (y_val_prob >= threshold).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        tp = ((y_pred == 1) & (y_val == 1)).sum()
        fp = ((y_pred == 1) & (y_val == 0)).sum()
        fn = ((y_pred == 0) & (y_val == 1)).sum()
        
        logger.info(f"Threshold {threshold:.1f}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (TP={tp}, FP={fp}, FN={fn})")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {'precision': prec, 'recall': rec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}
    
    logger.info("")
    logger.info(f"Best F1: {best_f1:.3f} at threshold {best_threshold}")
    logger.info("")
    
    # === STEP 9: Feature Importance ===
    logger.info("=" * 80)
    logger.info("TOP 10 FEATURE IMPORTANCES")
    logger.info("=" * 80)
    logger.info("")
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices):
        logger.info(f"{i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    logger.info("")
    
    # === STEP 10: Validation positive probabilities ===
    logger.info("=" * 80)
    logger.info("VALIDATION POSITIVE PROBABILITIES")
    logger.info("=" * 80)
    logger.info("")
    
    val_pos_probs = y_val_prob[y_val == 1]
    logger.info(f"Min:  {val_pos_probs.min():.4f}")
    logger.info(f"Max:  {val_pos_probs.max():.4f}")
    logger.info(f"Mean: {val_pos_probs.mean():.4f}")
    logger.info(f"Median: {np.median(val_pos_probs):.4f}")
    logger.info(f">0.3: {(val_pos_probs > 0.3).sum()}/{len(val_pos_probs)}")
    logger.info(f">0.5: {(val_pos_probs > 0.5).sum()}/{len(val_pos_probs)}")
    logger.info("")
    
    # === FINAL SUMMARY ===
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"CNN-LSTM (30 epochs):     F1 = 0.000 (complete collapse)")
    logger.info(f"XGBoost (this run):       F1 = {best_f1:.3f} at threshold {best_threshold}")
    logger.info("")
    
    if best_f1 > 0.10:
        logger.info("✓ SUCCESS: XGBoost learned from the features!")
        logger.info("  The CNN-LSTM training was the problem, not the data.")
    elif best_f1 > 0.01:
        logger.info("~ PARTIAL: Some learning, but weak.")
    else:
        logger.info("✗ FAILURE: XGBoost also couldn't learn.")
    
    logger.info("")
    logger.info("=" * 80)

    # === STEP 11: Save model and feature columns ===
    logger.info("")
    logger.info("SAVING MODEL...")
    logger.info("")

    os.makedirs('/Users/brettzeigler/Pythia/models', exist_ok=True)

    model_path = '/Users/brettzeigler/Pythia/models/xgboost_slow_large_v1.pkl'
    joblib.dump(clf, model_path)
    logger.info(f"✓ Model saved to {model_path}")

    # Save feature column order for reference
    feature_cols_path = '/Users/brettzeigler/Pythia/models/feature_columns.txt'
    with open(feature_cols_path, 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    logger.info(f"✓ Feature columns saved to {feature_cols_path}")
    logger.info("")


if __name__ == '__main__':
    main()
