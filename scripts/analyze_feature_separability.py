#!/usr/bin/env python3
"""
Feature Separability Analysis

Analyzes whether features actually differ between pre-spike and non-spike moments.
If the model collapsed to predicting the prior (0.024), it means features don't
contain useful signal to distinguish positives from negatives.

This script compares feature statistics for positive vs negative samples to determine
if there's any signal at all.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from src.models.dataset import DatasetBuilder, SpikeDataset
from src.models.validation import TimeSeriesSplitter
from src.models.spike_filter import SpikeCategoryFilter


def cohen_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def analyze_feature_separability(train_dataset):
    """Analyze if features differ between positive and negative samples."""

    logger.info("="*80)
    logger.info("FEATURE SEPARABILITY ANALYSIS")
    logger.info("="*80)
    logger.info("")

    # Extract sequences (already numpy arrays after normalization)
    sequences = train_dataset.sequences if isinstance(train_dataset.sequences, np.ndarray) else train_dataset.sequences.numpy()
    labels = train_dataset.labels if isinstance(train_dataset.labels, np.ndarray) else train_dataset.labels.numpy()

    # Get positive and negative samples
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_sequences = sequences[pos_mask]
    neg_sequences = sequences[neg_mask]

    logger.info(f"Positive samples: {pos_mask.sum()}")
    logger.info(f"Negative samples: {neg_mask.sum()}")
    logger.info("")

    # For analysis, we'll look at:
    # 1. Last timestep (moment before spike)
    # 2. Average over last 5 timesteps (recent context)
    # 3. Max/min over entire sequence (to catch brief signals)

    pos_last = pos_sequences[:, -1, :]  # (n_pos, n_features)
    neg_last = neg_sequences[:, -1, :]  # (n_neg, n_features)

    # Sample negatives to avoid memory issues (use 10k random samples)
    if len(neg_last) > 10000:
        np.random.seed(42)
        neg_sample_idx = np.random.choice(len(neg_last), 10000, replace=False)
        neg_last_sampled = neg_last[neg_sample_idx]
    else:
        neg_last_sampled = neg_last

    logger.info(f"Comparing last timestep features:")
    logger.info(f"  Positives: {len(pos_last)}")
    logger.info(f"  Negatives: {len(neg_last_sampled)} (sampled)")
    logger.info("")

    # Analyze each feature
    logger.info("="*80)
    logger.info("LAST TIMESTEP FEATURE COMPARISON")
    logger.info("="*80)
    logger.info(f"{'Feature':<30} {'Pos Mean':>10} {'Neg Mean':>10} {'Diff':>10} {'Cohen d':>10} {'p-value':>12} {'Signal':<10}")
    logger.info("-"*100)

    results = []

    for i, name in enumerate(train_dataset.feature_names):
        pos_vals = pos_last[:, i]
        neg_vals = neg_last_sampled[:, i]

        pos_mean = pos_vals.mean()
        neg_mean = neg_vals.mean()
        diff = pos_mean - neg_mean

        # Cohen's d effect size
        d = cohen_d(pos_vals, neg_vals)

        # T-test
        t_stat, p_value = stats.ttest_ind(pos_vals, neg_vals)

        # Signal strength classification
        signal = ""
        if abs(d) > 0.8:
            signal = "*** STRONG"
        elif abs(d) > 0.5:
            signal = "** MEDIUM"
        elif abs(d) > 0.2:
            signal = "* WEAK"
        else:
            signal = "NONE"

        logger.info(f"{name:<30} {pos_mean:>10.4f} {neg_mean:>10.4f} {diff:>10.4f} {d:>10.4f} {p_value:>12.6f} {signal:<10}")

        results.append({
            'feature': name,
            'pos_mean': pos_mean,
            'neg_mean': neg_mean,
            'diff': diff,
            'cohen_d': d,
            'p_value': p_value,
            'signal': signal
        })

    logger.info("")

    # Summary statistics
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    results_df = pd.DataFrame(results)

    strong_signal = results_df[results_df['cohen_d'].abs() > 0.8]
    medium_signal = results_df[(results_df['cohen_d'].abs() > 0.5) & (results_df['cohen_d'].abs() <= 0.8)]
    weak_signal = results_df[(results_df['cohen_d'].abs() > 0.2) & (results_df['cohen_d'].abs() <= 0.5)]
    no_signal = results_df[results_df['cohen_d'].abs() <= 0.2]

    logger.info(f"Features with STRONG signal (|d| > 0.8): {len(strong_signal)}")
    if len(strong_signal) > 0:
        for _, row in strong_signal.iterrows():
            logger.info(f"  - {row['feature']}: d={row['cohen_d']:.3f}")

    logger.info("")
    logger.info(f"Features with MEDIUM signal (|d| > 0.5): {len(medium_signal)}")
    if len(medium_signal) > 0:
        for _, row in medium_signal.iterrows():
            logger.info(f"  - {row['feature']}: d={row['cohen_d']:.3f}")

    logger.info("")
    logger.info(f"Features with WEAK signal (|d| > 0.2): {len(weak_signal)}")
    if len(weak_signal) > 0:
        for _, row in weak_signal.iterrows():
            logger.info(f"  - {row['feature']}: d={row['cohen_d']:.3f}")

    logger.info("")
    logger.info(f"Features with NO signal (|d| <= 0.2): {len(no_signal)}")

    logger.info("")

    # Analyze feature sequences over time (last 10 timesteps)
    logger.info("="*80)
    logger.info("TEMPORAL ANALYSIS (Last 10 Timesteps)")
    logger.info("="*80)
    logger.info("")

    # Find features with best separability
    top_features = results_df.nlargest(5, 'cohen_d', keep='all')

    for _, row in top_features.iterrows():
        feat_idx = train_dataset.feature_names.index(row['feature'])
        feat_name = row['feature']

        logger.info(f"{feat_name} (d={row['cohen_d']:.3f}):")
        logger.info("  Timestep: " + " ".join([f"t-{i:2d}" for i in range(9, -1, -1)]))

        # Positive mean over last 10 timesteps
        pos_temporal = pos_sequences[:, -10:, feat_idx].mean(axis=0)
        logger.info("  Pos Mean: " + " ".join([f"{v:5.2f}" for v in pos_temporal]))

        # Negative mean over last 10 timesteps
        neg_sample_idx_full = np.random.choice(len(neg_sequences), min(10000, len(neg_sequences)), replace=False)
        neg_temporal = neg_sequences[neg_sample_idx_full, -10:, feat_idx].mean(axis=0)
        logger.info("  Neg Mean: " + " ".join([f"{v:5.2f}" for v in neg_temporal]))

        logger.info("")

    # Overall conclusion
    logger.info("="*80)
    logger.info("CONCLUSION")
    logger.info("="*80)

    if len(strong_signal) > 0:
        logger.info("✓ STRONG SIGNAL DETECTED")
        logger.info(f"  {len(strong_signal)} features show strong separation (|d| > 0.8)")
        logger.info("  The model SHOULD be able to learn from these features.")
        logger.info("  Problem is likely in model architecture or training dynamics.")
    elif len(medium_signal) > 0:
        logger.info("⚠ MODERATE SIGNAL DETECTED")
        logger.info(f"  {len(medium_signal)} features show moderate separation (|d| > 0.5)")
        logger.info("  Signal exists but is weak. May need:")
        logger.info("  - Simpler models (XGBoost, logistic regression)")
        logger.info("  - More training data (lower thresholds)")
        logger.info("  - Better feature engineering")
    elif len(weak_signal) > 0:
        logger.info("⚠ WEAK SIGNAL DETECTED")
        logger.info(f"  {len(weak_signal)} features show weak separation (|d| > 0.2)")
        logger.info("  Signal is very weak. Recommendations:")
        logger.info("  - Lower spike thresholds to get more obvious spikes")
        logger.info("  - Predict 'spike happening' instead of 'pre-spike'")
        logger.info("  - Use longer prediction windows (5-10 minutes)")
    else:
        logger.info("✗ NO SIGNAL DETECTED")
        logger.info("  Features do not differ between pre-spike and normal moments.")
        logger.info("  The 1-minute pre-spike window is too early - no accumulation visible.")
        logger.info("  CRITICAL: This task may not be solvable with current approach.")
        logger.info("")
        logger.info("  Recommendations:")
        logger.info("  1. Change target to 'spike happening now' (0-1 min ahead)")
        logger.info("  2. Lower thresholds (3x vol, 4% price) to catch gradual builds")
        logger.info("  3. Try 5-10 minute prediction window")

    logger.info("")

    return results_df


def main():
    # Configuration
    DB_PATH = '/Users/brettzeigler/Pythia/market_data.duckdb'
    CATEGORIZED_CSV = '/Users/brettzeigler/Pythia/all_spikes_categorized.csv'
    SEQUENCE_LENGTH = 60

    logger.info("="*80)
    logger.info("FEATURE SEPARABILITY ANALYSIS")
    logger.info("="*80)
    logger.info("")

    # Get 125 symbols (same as training)
    logger.info(f"Loading symbols from: {CATEGORIZED_CSV}")
    df = pd.read_csv(CATEGORIZED_CSV)
    symbols = df[df['category'] == 'Slow & Large']['symbol'].unique().tolist()
    logger.info(f"✓ Loaded {len(symbols)} Slow & Large symbols")
    logger.info("")

    # Initialize spike filter
    spike_filter = SpikeCategoryFilter(CATEGORIZED_CSV)
    logger.info("")

    # Build dataset
    logger.info("Building training dataset...")
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
    logger.info("")

    # Split to get training set
    splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, gap=2)
    train_idx, val_idx, test_idx = splitter.split(len(dataset))

    # Extract training data
    train_sequences = dataset.sequences[train_idx].numpy()
    train_labels = dataset.labels[train_idx].numpy()
    train_timestamps = dataset.timestamps[train_idx]
    train_symbols = dataset.symbols[train_idx]

    # SAMPLE DATA FIRST to avoid memory issues
    # Keep ALL positives (165) + sample 50k negatives
    pos_mask = train_labels == 1
    neg_mask = train_labels == 0

    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    logger.info(f"Full training set: {len(train_labels)} samples")
    logger.info(f"  Positives: {pos_mask.sum()}")
    logger.info(f"  Negatives: {neg_mask.sum()}")
    logger.info("")

    # Sample negatives
    np.random.seed(42)
    if len(neg_indices) > 50000:
        neg_sample_indices = np.random.choice(neg_indices, 50000, replace=False)
        logger.info(f"Sampling 50,000 negatives from {len(neg_indices)} for analysis")
    else:
        neg_sample_indices = neg_indices
        logger.info(f"Using all {len(neg_indices)} negatives")

    # Combine positive and sampled negative indices
    sample_indices = np.concatenate([pos_indices, neg_sample_indices])
    np.random.shuffle(sample_indices)

    # Create sampled dataset
    train_dataset = SpikeDataset(
        sequences=train_sequences[sample_indices],
        labels=train_labels[sample_indices],
        timestamps=train_timestamps[sample_indices],
        symbols=train_symbols[sample_indices],
        feature_names=dataset.feature_names
    )

    logger.info(f"Sampled dataset: {len(train_dataset)} samples")
    logger.info("")

    # Normalize (much faster on 50k samples)
    logger.info("Normalizing sampled data...")
    train_sequences_norm = builder.normalize_sequences(train_dataset.sequences, fit=True)
    train_dataset.sequences = train_sequences_norm  # Keep as numpy for analysis

    train_stats = train_dataset.get_statistics()
    logger.info(f"Normalized training set: {len(train_dataset)} samples")
    logger.info(f"  Positives: {train_stats['positive_samples']}")
    logger.info(f"  Negatives: {train_stats['negative_samples']}")
    logger.info("")

    # Run analysis
    results = analyze_feature_separability(train_dataset)

    # Save results
    output_file = "feature_separability_results.csv"
    results.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
