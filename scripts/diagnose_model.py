#!/usr/bin/env python3
"""
Model Diagnostic Script

Loads trained model checkpoint and runs comprehensive diagnostics:
- Load validation dataset (same 125 symbols, same splits)
- Run inference on all validation samples
- Extract probabilities for validation positives (33 samples)
- Report: min/max/mean probability, threshold counts
- Compute F1 scores at different thresholds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from loguru import logger

from src.models.trainer import ModelTrainer
from src.models.dataset import DatasetBuilder, SpikeDataset
from src.models.validation import TimeSeriesSplitter
from src.models.spike_filter import SpikeCategoryFilter


def load_model_checkpoint(checkpoint_path, device='mps'):
    """Load model from checkpoint"""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    config = checkpoint['model_config']
    n_features = config['n_features']
    sequence_length = config['sequence_length']
    model_type = config['model_type']

    # Initialize trainer with same config
    trainer = ModelTrainer(
        model_type=model_type,
        n_features=n_features,
        sequence_length=sequence_length,
        device=device,
        loss_type='bce',
        checkpoint_dir=Path(checkpoint_path).parent
    )

    # Load weights
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model.eval()

    logger.info(f"✓ Loaded {model_type} model from epoch {checkpoint['epoch']}")
    logger.info(f"  n_features: {n_features}")
    logger.info(f"  sequence_length: {sequence_length}")

    return trainer, config


def load_validation_dataset(db_path, symbols, sequence_length):
    """Load same validation dataset used during training"""
    logger.info("Building validation dataset...")

    # Initialize spike filter (Slow & Large only)
    spike_filter = SpikeCategoryFilter('all_spikes_categorized.csv')

    # Build dataset
    builder = DatasetBuilder(
        db_path=db_path,
        sequence_length=sequence_length,
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

    # Split (60/20/20, gap=2)
    splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, gap=2)
    train_idx, val_idx, test_idx = splitter.split(len(dataset))

    # Extract validation set
    val_dataset = SpikeDataset(
        sequences=dataset.sequences[val_idx].numpy(),
        labels=dataset.labels[val_idx].numpy(),
        timestamps=dataset.timestamps[val_idx],
        symbols=dataset.symbols[val_idx],
        feature_names=dataset.feature_names
    )

    # Normalize using training set scaler
    logger.info("Normalizing sequences...")
    train_sequences = dataset.sequences[train_idx].numpy()
    _ = builder.normalize_sequences(train_sequences, fit=True)
    val_sequences_norm = builder.normalize_sequences(val_dataset.sequences.numpy())
    val_dataset.sequences = torch.FloatTensor(val_sequences_norm)

    val_stats = val_dataset.get_statistics()
    logger.info(f"✓ Validation set: {len(val_dataset)} samples")
    logger.info(f"  Positives: {val_stats['positive_samples']}")
    logger.info(f"  Negatives: {val_stats['negative_samples']}")

    return val_dataset


def run_diagnostics(trainer, val_dataset, device='mps'):
    """Run comprehensive diagnostics"""

    # Create dataloader (batch size 32 for efficiency)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Run inference
    logger.info("")
    logger.info("="*80)
    logger.info("RUNNING INFERENCE ON VALIDATION SET")
    logger.info("="*80)

    all_probs = []
    all_labels = []

    trainer.model.eval()
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            outputs = trainer.model(sequences)  # Returns probabilities

            all_probs.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Extract positive samples
    positive_mask = all_labels == 1
    positive_probs = all_probs[positive_mask]
    negative_probs = all_probs[~positive_mask]

    logger.info(f"Total validation samples: {len(all_labels)}")
    logger.info(f"Positive samples: {positive_mask.sum()}")
    logger.info(f"Negative samples: {(~positive_mask).sum()}")

    # Probability analysis for positives
    logger.info("")
    logger.info("="*80)
    logger.info("POSITIVE SAMPLES PROBABILITY DISTRIBUTION")
    logger.info("="*80)
    logger.info(f"Count:  {len(positive_probs)}")
    logger.info(f"Min:    {positive_probs.min():.6f}")
    logger.info(f"Max:    {positive_probs.max():.6f}")
    logger.info(f"Mean:   {positive_probs.mean():.6f}")
    logger.info(f"Median: {np.median(positive_probs):.6f}")
    logger.info(f"Std:    {positive_probs.std():.6f}")

    # Threshold counts
    logger.info("")
    logger.info("="*80)
    logger.info("POSITIVE SAMPLES AT DIFFERENT THRESHOLDS")
    logger.info("="*80)
    for threshold in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        count = (positive_probs > threshold).sum()
        pct = count / len(positive_probs) * 100
        logger.info(f"  >{threshold:.2f}: {count:2d} / {len(positive_probs)} ({pct:5.1f}%)")

    # F1 scores at different thresholds
    logger.info("")
    logger.info("="*80)
    logger.info("F1 SCORES AT DIFFERENT THRESHOLDS")
    logger.info("="*80)

    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        preds = (all_probs > threshold).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

        logger.info(f"\nThreshold: {threshold:.2f}")
        logger.info(f"  Precision: {precision*100:6.2f}%")
        logger.info(f"  Recall:    {recall*100:6.2f}%")
        logger.info(f"  F1:        {f1*100:6.2f}%")

        # Confusion matrix
        cm = confusion_matrix(all_labels, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            logger.info(f"  TP: {tp:5d}  FP: {fp:7d}")
            logger.info(f"  FN: {fn:5d}  TN: {tn:7d}")

    logger.info("")
    logger.info("="*80)
    logger.info(f"BEST F1: {best_f1*100:.2f}% at threshold {best_threshold:.2f}")
    logger.info("="*80)

    # Probability histogram
    logger.info("")
    logger.info("="*80)
    logger.info("PROBABILITY HISTOGRAM")
    logger.info("="*80)
    bins = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    hist_pos, _ = np.histogram(positive_probs, bins=bins)
    hist_neg, _ = np.histogram(negative_probs, bins=bins)

    logger.info(f"{'Range':<12} {'Positives':>10} {'Negatives':>12} {'Pos %':>8}")
    logger.info("-" * 44)
    for i in range(len(bins)-1):
        pos_pct = (hist_pos[i] / len(positive_probs) * 100) if len(positive_probs) > 0 else 0
        logger.info(f"{bins[i]:.2f}-{bins[i+1]:.2f}     {hist_pos[i]:>10d} {hist_neg[i]:>12d} {pos_pct:>7.1f}%")

    # Top 10 highest probability positives
    logger.info("")
    logger.info("="*80)
    logger.info("TOP 10 HIGHEST PROBABILITY POSITIVES")
    logger.info("="*80)
    positive_indices = np.where(positive_mask)[0]
    top_10_idx = positive_indices[np.argsort(positive_probs)[-10:][::-1]]

    for rank, idx in enumerate(top_10_idx, 1):
        prob = all_probs[idx]
        symbol = val_dataset.symbols[idx]
        timestamp = val_dataset.timestamps[idx]
        logger.info(f"{rank:2d}. {symbol:12s} {timestamp} | Prob: {prob:.6f}")

    # Bottom 10 lowest probability positives
    logger.info("")
    logger.info("="*80)
    logger.info("BOTTOM 10 LOWEST PROBABILITY POSITIVES")
    logger.info("="*80)
    bottom_10_idx = positive_indices[np.argsort(positive_probs)[:10]]

    for rank, idx in enumerate(bottom_10_idx, 1):
        prob = all_probs[idx]
        symbol = val_dataset.symbols[idx]
        timestamp = val_dataset.timestamps[idx]
        logger.info(f"{rank:2d}. {symbol:12s} {timestamp} | Prob: {prob:.6f}")

    return {
        'positive_probs': positive_probs,
        'best_f1': best_f1,
        'best_threshold': best_threshold
    }


def main():
    # Configuration
    DB_PATH = '/Users/brettzeigler/Pythia/market_data.duckdb'
    CHECKPOINT_PATH = '/Users/brettzeigler/Pythia/models/cnn_lstm_improved_20251202_175034/latest_model.pt'
    CATEGORIZED_CSV = '/Users/brettzeigler/Pythia/all_spikes_categorized.csv'

    logger.info("="*80)
    logger.info("MODEL DIAGNOSTICS")
    logger.info("="*80)

    # Get 125 symbols (same as training)
    logger.info(f"Loading symbols from: {CATEGORIZED_CSV}")
    df = pd.read_csv(CATEGORIZED_CSV)
    symbols = df[df['category'] == 'Slow & Large']['symbol'].unique().tolist()
    logger.info(f"✓ Loaded {len(symbols)} Slow & Large symbols")
    logger.info("")

    # Load model
    trainer, config = load_model_checkpoint(CHECKPOINT_PATH, device='mps')
    logger.info("")

    # Load validation dataset
    val_dataset = load_validation_dataset(DB_PATH, symbols, config['sequence_length'])
    logger.info("")

    # Run diagnostics
    results = run_diagnostics(trainer, val_dataset, device='mps')

    logger.info("")
    logger.info("="*80)
    logger.info("DIAGNOSTICS COMPLETE")
    logger.info("="*80)
    logger.info(f"Best F1: {results['best_f1']*100:.2f}% at threshold {results['best_threshold']:.2f}")
    logger.info("")


if __name__ == '__main__':
    main()
