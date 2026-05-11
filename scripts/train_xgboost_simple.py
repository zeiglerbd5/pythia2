#!/usr/bin/env python3
"""
XGBoost Baseline - Memory-Efficient Version
Samples data during loading instead of after.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import joblib

from src.models.dataset import DatasetBuilder
from src.models.spike_filter import SpikeCategoryFilter


def main():
    DB_PATH = '/Users/brettzeigler/Pythia/market_data.duckdb'
    CATEGORIZED_CSV = '/Users/brettzeigler/Pythia/all_spikes_categorized.csv'
    SEQUENCE_LENGTH = 60

    logger.info("=" * 80)
    logger.info("XGBOOST BASELINE - MEMORY EFFICIENT VERSION")
    logger.info("=" * 80)
    logger.info("")

    # Load symbols
    logger.info(f"Loading symbols from: {CATEGORIZED_CSV}")
    df = pd.read_csv(CATEGORIZED_CSV)
    symbols = df[df['category'] == 'Slow & Large']['symbol'].unique().tolist()
    logger.info(f"✓ Loaded {len(symbols)} Slow & Large symbols")
    logger.info("")

    # Sample subset of symbols to reduce memory (e.g., 10 symbols instead of 125)
    np.random.seed(42)
    symbols_sample = np.random.choice(symbols, min(10, len(symbols)), replace=False).tolist()
    logger.info(f"Sampling {len(symbols_sample)} symbols to reduce memory footprint")
    logger.info("")

    # Initialize spike filter
    spike_filter = SpikeCategoryFilter(CATEGORIZED_CSV)
    logger.info("")

    # Build dataset on SAMPLED symbols only
    logger.info(f"Building dataset for {len(symbols_sample)} symbols...")
    builder = DatasetBuilder(
        db_path=DB_PATH,
        sequence_length=SEQUENCE_LENGTH,
        scaler_type='robust'
    )

    dataset = builder.build_dataset_multi_symbol(
        symbols=symbols_sample,
        timeframe='1m',
        normalize=False,
        fit_scaler=False,
        spike_filter=spike_filter
    )

    logger.info(f"Total dataset: {len(dataset)} sequences")
    stats = dataset.get_statistics()
    logger.info(f"  Positives: {stats['positive_samples']} ({stats['positive_ratio']*100:.4f}%)")
    logger.info(f"  Negatives: {stats['negative_samples']}")
    logger.info("")

    # Split dataset
    from src.models.validation import TimeSeriesSplitter
    splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, gap=2)
    train_idx, val_idx, test_idx = splitter.split(len(dataset))

    # Extract ONLY last timestep (not full sequences) - 60x memory reduction!
    logger.info("Extracting ONLY last timestep features (60x memory reduction)...")

    # For training: extract last timestep for sampled indices only
    train_sequences_last = dataset.sequences[train_idx, -1, :].numpy()
    train_labels = dataset.labels[train_idx].numpy()

    # For val/test: extract last timestep directly
    val_sequences_last = dataset.sequences[val_idx, -1, :].numpy()
    val_labels = dataset.labels[val_idx].numpy()

    test_sequences_last = dataset.sequences[test_idx, -1, :].numpy()
    test_labels = dataset.labels[test_idx].numpy()

    logger.info(f"Train: {len(train_labels)} samples ({train_labels.sum()} positive)")
    logger.info(f"Val:   {len(val_labels)} samples ({val_labels.sum()} positive)")
    logger.info(f"Test:  {len(test_labels)} samples ({test_labels.sum()} positive)")
    logger.info(f"Extracted shapes - Train: {train_sequences_last.shape}, Val: {val_sequences_last.shape}, Test: {test_sequences_last.shape}")
    logger.info("")

    # Normalize last timestep features
    logger.info("Normalizing last timestep features...")
    # Reshape to (n_samples, 1, n_features) for normalize_sequences
    train_seq_reshaped = train_sequences_last[:, np.newaxis, :]
    val_seq_reshaped = val_sequences_last[:, np.newaxis, :]
    test_seq_reshaped = test_sequences_last[:, np.newaxis, :]

    train_norm = builder.normalize_sequences(train_seq_reshaped, fit=True)
    val_norm = builder.normalize_sequences(val_seq_reshaped)
    test_norm = builder.normalize_sequences(test_seq_reshaped)

    # Extract back to 2D (n_samples, n_features)
    X_train = train_norm[:, 0, :]
    y_train = train_labels

    X_val = val_norm[:, 0, :]
    y_val = val_labels

    X_test = test_norm[:, 0, :]
    y_test = test_labels

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info("")

    # Calculate class weight
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    logger.info(f"Class imbalance: {pos_weight:.1f}:1")
    logger.info(f"Using scale_pos_weight={pos_weight:.1f}")
    logger.info("")

    # Train XGBoost
    logger.info("=" * 80)
    logger.info("TRAINING XGBOOST")
    logger.info("=" * 80)
    logger.info("")

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        eval_metric='aucpr',
        random_state=42,
        tree_method='hist',
        device='cpu'
    )

    logger.info("Training XGBoost classifier...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    logger.info("")
    logger.info("✓ Training complete")
    logger.info("")

    # Evaluate
    logger.info("=" * 80)
    logger.info("VALIDATION SET EVALUATION")
    logger.info("=" * 80)
    logger.info("")

    y_val_pred = clf.predict(X_val)
    y_val_prob = clf.predict_proba(X_val)[:, 1]

    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)

    logger.info(f"Default threshold (0.5):")
    logger.info(f"  Precision: {precision*100:6.2f}%")
    logger.info(f"  Recall:    {recall*100:6.2f}%")
    logger.info(f"  F1:        {f1*100:6.2f}%")
    logger.info("")

    # Try different thresholds
    logger.info("=" * 80)
    logger.info("THRESHOLD TUNING")
    logger.info("=" * 80)
    logger.info("")

    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        y_pred_thresh = (y_val_prob >= threshold).astype(int)
        prec = precision_score(y_val, y_pred_thresh, zero_division=0)
        rec = recall_score(y_val, y_pred_thresh, zero_division=0)
        f1_thresh = f1_score(y_val, y_pred_thresh, zero_division=0)

        if f1_thresh > best_f1:
            best_f1 = f1_thresh
            best_threshold = threshold

        logger.info(f"Threshold: {threshold:.2f}")
        logger.info(f"  Precision: {prec*100:6.2f}%")
        logger.info(f"  Recall:    {rec*100:6.2f}%")
        logger.info(f"  F1:        {f1_thresh*100:6.2f}%")
        logger.info("")

    logger.info(f"Best F1: {best_f1*100:.2f}% at threshold {best_threshold:.2f}")
    logger.info("")

    # Final comparison
    logger.info("=" * 80)
    logger.info("BASELINE COMPARISON")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"CNN-LSTM (Focal Loss, Balanced Batching):")
    logger.info(f"  Best Val F1: 0.00%")
    logger.info("")
    logger.info(f"XGBoost (This Run, {len(symbols_sample)} symbols):")
    logger.info(f"  Best Val F1: {best_f1*100:.2f}% at threshold {best_threshold:.2f}")
    logger.info("")

    if best_f1 > 0.10:
        logger.info("✓ SUCCESS: XGBoost is learning!")
        logger.info("  Problem was CNN-LSTM training, not the data.")
    elif best_f1 > 0.01:
        logger.info("⚠ PARTIAL: Weak learning detected.")
    else:
        logger.info("✗ FAILURE: XGBoost also failed.")
    logger.info("")

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
