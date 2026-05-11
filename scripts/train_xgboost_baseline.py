#!/usr/bin/env python3
"""
XGBoost Baseline Model for Slow & Large Spike Prediction

Tests if a simpler model can learn from the strong feature signals identified
in feature separability analysis (13 features with Cohen's d > 0.8).

Key findings from separability analysis:
- BB_width: d=8.269 (EXTREME signal)
- NATR: d=7.437
- bid_ask_spread_pct: d=3.528
- 13 total features with |d| > 0.8

If XGBoost works but CNN-LSTM collapsed, the problem is in deep learning training.
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
from src.models.validation import TimeSeriesSplitter
from src.models.spike_filter import SpikeCategoryFilter


def main():
    # Configuration
    DB_PATH = '/Users/brettzeigler/Pythia/market_data.duckdb'
    CATEGORIZED_CSV = '/Users/brettzeigler/Pythia/all_spikes_categorized.csv'
    SEQUENCE_LENGTH = 60

    logger.info("=" * 80)
    logger.info("XGBOOST BASELINE - SLOW & LARGE SPIKE PREDICTION")
    logger.info("=" * 80)
    logger.info("")

    # Load symbols
    logger.info(f"Loading symbols from: {CATEGORIZED_CSV}")
    df = pd.read_csv(CATEGORIZED_CSV)
    symbols = df[df['category'] == 'Slow & Large']['symbol'].unique().tolist()
    logger.info(f"✓ Loaded {len(symbols)} Slow & Large symbols")
    logger.info("")

    # Initialize spike filter
    spike_filter = SpikeCategoryFilter(CATEGORIZED_CSV)
    logger.info("")

    # Build dataset
    logger.info("Building dataset...")
    builder = DatasetBuilder(
        db_path=DB_PATH,
        sequence_length=SEQUENCE_LENGTH,
        scaler_type='robust'
    )

    dataset = builder.build_dataset_multi_symbol(
        symbols=symbols,
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

    # Split dataset (60/20/20)
    splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, gap=2)
    train_idx, val_idx, test_idx = splitter.split(len(dataset))

    logger.info(f"Train indices: {len(train_idx)}")
    logger.info(f"Val indices:   {len(val_idx)}")
    logger.info(f"Test indices:  {len(test_idx)}")
    logger.info("")

    # SAMPLE TRAINING INDICES BEFORE EXTRACTING ARRAYS to avoid memory issues
    # This is critical - we must sample indices first, then only extract those arrays
    logger.info("Sampling training indices to avoid loading full 5.5M dataset...")

    # Get labels for training split only (cheap operation on indices)
    train_labels_full = dataset.labels.numpy()[train_idx]

    pos_mask = train_labels_full == 1
    neg_mask = train_labels_full == 0

    pos_indices_in_train = np.where(pos_mask)[0]
    neg_indices_in_train = np.where(neg_mask)[0]

    logger.info(f"  Full training set: {len(train_labels_full)} samples")
    logger.info(f"    Positives: {len(pos_indices_in_train)}")
    logger.info(f"    Negatives: {len(neg_indices_in_train)}")

    # Sample negatives within training set
    np.random.seed(42)
    if len(neg_indices_in_train) > 50000:
        neg_sample_indices_in_train = np.random.choice(neg_indices_in_train, 50000, replace=False)
        logger.info(f"  Sampling 50,000 negatives from {len(neg_indices_in_train)}")
    else:
        neg_sample_indices_in_train = neg_indices_in_train
        logger.info(f"  Using all {len(neg_indices_in_train)} negatives")

    # Combine positive and sampled negative indices (relative to training split)
    sampled_train_indices_relative = np.concatenate([pos_indices_in_train, neg_sample_indices_in_train])
    np.random.shuffle(sampled_train_indices_relative)

    # Map back to original dataset indices
    sampled_train_idx = train_idx[sampled_train_indices_relative]

    logger.info(f"  Sampled training indices: {len(sampled_train_idx)}")
    logger.info("")

    # Extract ONLY last timestep (not full sequences) - 60x memory reduction!
    logger.info("Extracting ONLY last timestep features (60x memory reduction)...")

    # For training: extract last timestep for sampled indices only
    train_sequences_last = dataset.sequences[sampled_train_idx, -1, :].numpy()
    train_labels = dataset.labels[sampled_train_idx].numpy()

    # For val/test: extract last timestep directly
    val_sequences_last = dataset.sequences[val_idx, -1, :].numpy()
    val_labels = dataset.labels[val_idx].numpy()

    test_sequences_last = dataset.sequences[test_idx, -1, :].numpy()
    test_labels = dataset.labels[test_idx].numpy()

    logger.info(f"Train: {len(train_labels)} samples ({train_labels.sum()} positive)")
    logger.info(f"Val:   {len(val_labels)} samples ({val_labels.sum()} positive)")
    logger.info(f"Test:  {len(test_labels)} samples ({test_labels.sum()} positive)")
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
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Class imbalance: {pos_weight:.1f}:1 (negatives:positives)")
    logger.info(f"Using scale_pos_weight={pos_weight:.1f}")
    logger.info("")

    # Train XGBoost
    logger.info("=" * 80)
    logger.info("TRAINING XGBOOST")
    logger.info("=" * 80)
    logger.info("")

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,  # Shallow to avoid overfitting on 165 positives
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        eval_metric='aucpr',  # Better for imbalanced data
        random_state=42,
        tree_method='hist',  # Faster training
        device='cpu'  # XGBoost doesn't support MPS
    )

    logger.info("Training XGBoost classifier...")
    logger.info(f"  n_estimators: {clf.n_estimators}")
    logger.info(f"  max_depth: {clf.max_depth}")
    logger.info(f"  learning_rate: {clf.learning_rate}")
    logger.info(f"  scale_pos_weight: {clf.scale_pos_weight:.1f}")
    logger.info("")

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    logger.info("")
    logger.info("✓ Training complete")
    logger.info("")

    # Evaluate on validation set
    logger.info("=" * 80)
    logger.info("VALIDATION SET EVALUATION")
    logger.info("=" * 80)
    logger.info("")

    y_val_pred = clf.predict(X_val)
    y_val_prob = clf.predict_proba(X_val)[:, 1]

    # Metrics at default threshold (0.5)
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)

    logger.info(f"Default threshold (0.5):")
    logger.info(f"  Precision: {precision*100:6.2f}%")
    logger.info(f"  Recall:    {recall*100:6.2f}%")
    logger.info(f"  F1:        {f1*100:6.2f}%")
    logger.info("")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        logger.info(f"Confusion Matrix:")
        logger.info(f"  TP: {tp:5d}  FP: {fp:7d}")
        logger.info(f"  FN: {fn:5d}  TN: {tn:7d}")
        logger.info("")

    # Probability distribution on positives
    val_pos_probs = y_val_prob[y_val == 1]
    logger.info(f"Positive sample probabilities:")
    logger.info(f"  Count:  {len(val_pos_probs)}")
    logger.info(f"  Min:    {val_pos_probs.min():.6f}")
    logger.info(f"  Max:    {val_pos_probs.max():.6f}")
    logger.info(f"  Mean:   {val_pos_probs.mean():.6f}")
    logger.info(f"  Median: {np.median(val_pos_probs):.6f}")
    logger.info(f"  Std:    {val_pos_probs.std():.6f}")
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

        cm = confusion_matrix(y_val, y_pred_thresh)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            logger.info(f"  TP: {tp:5d}  FP: {fp:7d}")
            logger.info(f"  FN: {fn:5d}  TN: {tn:7d}")
        logger.info("")

    logger.info(f"Best F1: {best_f1*100:.2f}% at threshold {best_threshold:.2f}")
    logger.info("")

    # Feature importances
    logger.info("=" * 80)
    logger.info("FEATURE IMPORTANCES")
    logger.info("=" * 80)
    logger.info("")

    feature_names = [
        'returns', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14', 'NATR',
        'BB_width', 'BB_squeeze', 'VWAP_distance',
        'volume_zscore', 'volume_roc', 'OBV',
        'trade_count', 'buy_sell_ratio', 'roll_measure', 'order_flow_imbalance',
        'vpin', 'bid_ask_spread_pct',
        'order_book_depth_ratio', 'large_order_imbalance',
        'returns_5m', 'volume_zscore_5m', 'returns_15m', 'volume_zscore_15m'
    ]

    importances = clf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    logger.info("Top 10 most important features:")
    for idx, row in feature_importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']:<25s} {row['importance']:.4f}")
    logger.info("")

    # Compare to feature separability results
    logger.info("Comparing to feature separability (Cohen's d):")
    logger.info("Top features by Cohen's d:")
    logger.info("  1. BB_width (d=8.269)")
    logger.info("  2. NATR (d=7.437)")
    logger.info("  3. bid_ask_spread_pct (d=3.528)")
    logger.info("  4. returns_15m (d=-2.665)")
    logger.info("  5. volume_zscore_15m (d=2.211)")
    logger.info("")

    # Save model
    model_path = Path("models") / "xgboost_baseline"
    model_path.mkdir(parents=True, exist_ok=True)

    model_file = model_path / "xgboost_model.pkl"
    joblib.dump(clf, model_file)
    logger.info(f"Model saved to: {model_file}")

    feature_importance_df.to_csv(model_path / "feature_importances.csv", index=False)
    logger.info(f"Feature importances saved to: {model_path / 'feature_importances.csv'}")
    logger.info("")

    # Final comparison
    logger.info("=" * 80)
    logger.info("BASELINE COMPARISON")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"CNN-LSTM (Focal Loss, Balanced Batching):")
    logger.info(f"  Best Val F1: 0.00%")
    logger.info(f"  Issue: Model collapsed to predicting constant 0.024")
    logger.info("")
    logger.info(f"XGBoost (This Run):")
    logger.info(f"  Best Val F1: {best_f1*100:.2f}% at threshold {best_threshold:.2f}")
    logger.info(f"  Default F1:  {f1*100:.2f}% at threshold 0.50")
    logger.info("")

    if best_f1 > 0.10:
        logger.info("✓ SUCCESS: XGBoost is learning! The problem was CNN-LSTM training.")
        logger.info("  Signal is strong enough for simpler models.")
        logger.info("  Consider using XGBoost or fixing deep learning architecture.")
    elif best_f1 > 0.01:
        logger.info("⚠ PARTIAL: XGBoost shows some learning but performance is weak.")
        logger.info("  May need more training data or feature engineering.")
    else:
        logger.info("✗ FAILURE: XGBoost also failed despite strong signal (d=8.269).")
        logger.info("  This suggests a deeper problem with the task setup.")
    logger.info("")

    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
