#!/usr/bin/env python3
"""Quick Feature Separability Test - Hardcoded feature names"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

# HARDCODED feature names (24 features from your setup)
FEATURE_NAMES = [
    'returns', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14', 'NATR',
    'BB_width', 'BB_squeeze', 'VWAP_distance',
    'volume_zscore', 'volume_roc', 'OBV',
    'trade_count', 'buy_sell_ratio', 'roll_measure', 'order_flow_imbalance', 'vpin', 'bid_ask_spread_pct',
    'order_book_depth_ratio', 'large_order_imbalance',
    'returns_5m', 'volume_zscore_5m', 'returns_15m', 'volume_zscore_15m'
]

def cohen_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


# Load sampled data
logger.info("Loading sampled + normalized data from previous run...")
logger.info("This uses the 165 positives + 50k negatives that were already normalized")

# Load from the checkpoint saved during previous run
# We'll use diagnose_model data which is already normalized
from src.models.dataset import DatasetBuilder, SpikeDataset
from src.models.validation import TimeSeriesSplitter
from src.models.spike_filter import SpikeCategoryFilter

DB_PATH = '/Users/brettzeigler/Pythia/market_data.duckdb'
CATEGORIZED_CSV = '/Users/brettzeigler/Pythia/all_spikes_categorized.csv'
SEQUENCE_LENGTH = 60

logger.info(f"Loading symbols from: {CATEGORIZED_CSV}")
df = pd.read_csv(CATEGORIZED_CSV)
symbols = df[df['category'] == 'Slow & Large']['symbol'].unique().tolist()
logger.info(f"✓ Loaded {len(symbols)} Slow & Large symbols")

# Initialize spike filter
spike_filter = SpikeCategoryFilter(CATEGORIZED_CSV)

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

# Split
splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, gap=2)
train_idx, val_idx, test_idx = splitter.split(len(dataset))

# Extract training data
train_sequences = dataset.sequences[train_idx].numpy()
train_labels = dataset.labels[train_idx].numpy()

logger.info(f"Full training set: {len(train_labels)} samples")

# SAMPLE DATA FIRST
pos_mask = train_labels == 1
neg_mask = train_labels == 0

pos_indices = np.where(pos_mask)[0]
neg_indices = np.where(neg_mask)[0]

logger.info(f"  Positives: {pos_mask.sum()}")
logger.info(f"  Negatives: {neg_mask.sum()}")

# Sample 50k negatives
np.random.seed(42)
neg_sample_indices = np.random.choice(neg_indices, min(50000, len(neg_indices)), replace=False)

# Combine
sample_indices = np.concatenate([pos_indices, neg_sample_indices])
sampled_sequences = train_sequences[sample_indices]
sampled_labels = train_labels[sample_indices]

logger.info(f"Sampled: {len(sampled_sequences)} sequences")

# Normalize
logger.info("Normalizing...")
normalized_sequences = builder.normalize_sequences(sampled_sequences, fit=True)

# Extract positives and negatives
pos_mask_sampled = sampled_labels == 1
neg_mask_sampled = sampled_labels == 0

pos_sequences = normalized_sequences[pos_mask_sampled]
neg_sequences = normalized_sequences[neg_mask_sampled]

logger.info(f"Positive samples: {len(pos_sequences)}")
logger.info(f"Negative samples: {len(neg_sequences)}")

# Get last timestep
pos_last = pos_sequences[:, -1, :]  # (n_pos, n_features)
neg_last = neg_sequences[:, -1, :]  # (n_neg, n_features)

# Sample 10k negatives for comparison
if len(neg_last) > 10000:
    neg_sample_idx = np.random.choice(len(neg_last), 10000, replace=False)
    neg_last_sampled = neg_last[neg_sample_idx]
else:
    neg_last_sampled = neg_last

logger.info("")
logger.info("=" * 80)
logger.info("LAST TIMESTEP FEATURE COMPARISON")
logger.info("=" * 80)
logger.info(f"{'Feature':<30} {'Pos Mean':>10} {'Neg Mean':>10} {'Diff':>10} {'Cohen d':>10} {'p-value':>12} {'Signal':<10}")
logger.info("-" * 100)

results = []

for i, name in enumerate(FEATURE_NAMES):
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
logger.info("=" * 80)
logger.info("SUMMARY")
logger.info("=" * 80)

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
logger.info("=" * 80)
logger.info("CONCLUSION")
logger.info("=" * 80)

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

# Save results
output_file = "feature_separability_results.csv"
results_df.to_csv(output_file, index=False)
logger.info(f"Results saved to: {output_file}")
