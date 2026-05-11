#!/usr/bin/env python3
"""
Train Spike Detection Model

Complete training pipeline using collected data.

Prerequisites:
- 81+ days of data (60-day sequences + 14-day forward + 7-day buffer)
- Features calculated and stored in DuckDB
- OHLCV candles for target generation

Usage:
    python scripts/train_model.py --symbol BTC-USD --days 90 --timeframe 5m
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from loguru import logger

from src.models.dataset import DatasetBuilder, create_dataloaders
from src.models.smote import SMOTEDatasetWrapper
from src.models.trainer import ModelTrainer
from src.models.metrics import ModelEvaluator
from src.models.validation import TimeSeriesSplitter
from src.utils.config import get_config


async def train_single_model(
    symbol: str,
    days: int,
    timeframe: str = '5m',
    use_smote: bool = True,
    smote_ratio: float = 0.2
):
    """
    Train a single spike detection model.

    Args:
        symbol: Trading pair (e.g., 'BTC-USD')
        days: Days of historical data to use
        timeframe: Timeframe ('1m', '5m', '15m')
        use_smote: Apply SMOTE oversampling
        smote_ratio: SMOTE target ratio (0.1-0.2)
    """
    logger.info("=" * 80)
    logger.info(f"TRAINING SPIKE DETECTION MODEL: {symbol}")
    logger.info("=" * 80)

    # Load configuration
    config = get_config()
    db_path = str(config.get_database_path())

    # Check if database exists
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Please collect data first using integrated_collector.py")
        return

    # Initialize dataset builder
    logger.info("Initializing dataset builder...")

    builder = DatasetBuilder(
        db_path=db_path,
        sequence_length=60,  # 60-day lookback per guide
        forward_window=14,  # 7-14 day forward per guide
        min_spike_threshold=0.15,  # 15% minimum spike
        scaler_type='robust'  # Better for outliers
    )

    # Calculate date ranges
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    logger.info(f"Data range: {start_date.date()} to {end_date.date()}")

    # Check available features (Boruta-selected)
    # For now, use all available features
    # In production, you'd load Boruta-selected features
    feature_columns = None  # None = use all features

    # Build complete dataset
    logger.info(f"Loading data for {symbol}...")

    dataset = builder.build_dataset(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        feature_columns=feature_columns,
        normalize=False,  # We'll normalize per split
        fit_scaler=False
    )

    if dataset is None:
        logger.error(f"Failed to build dataset for {symbol}")
        logger.error("Possible reasons:")
        logger.error("  1. Not enough data (need 81+ days)")
        logger.error("  2. Features not calculated")
        logger.error("  3. Missing OHLCV data")
        return

    logger.info(f"Dataset created: {len(dataset)} samples")

    stats = dataset.get_statistics()
    logger.info(f"Dataset statistics:")
    logger.info(f"  Positive samples: {stats['positive_samples']} ({stats['positive_ratio']*100:.2f}%)")
    logger.info(f"  Negative samples: {stats['negative_samples']}")
    logger.info(f"  Features: {stats['n_features']}")
    logger.info(f"  Sequence length: {stats['sequence_length']}")

    # Check if we have enough data
    if len(dataset) < 500:
        logger.error(f"Not enough samples ({len(dataset)}). Need at least 500.")
        logger.error("Collect more data or reduce sequence_length/forward_window")
        return

    # Split into train/val/test (temporal)
    logger.info("Splitting into train/val/test sets...")

    splitter = TimeSeriesSplitter(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        gap=5  # 5-sample gap to prevent leakage
    )

    train_idx, val_idx, test_idx = splitter.split(len(dataset))

    # Create datasets for each split
    train_dataset = type(dataset)(
        sequences=dataset.sequences[train_idx].numpy(),
        labels=dataset.labels[train_idx].numpy(),
        timestamps=dataset.timestamps[train_idx],
        symbols=dataset.symbols[train_idx],
        feature_names=dataset.feature_names
    )

    val_dataset = type(dataset)(
        sequences=dataset.sequences[val_idx].numpy(),
        labels=dataset.labels[val_idx].numpy(),
        timestamps=dataset.timestamps[val_idx],
        symbols=dataset.symbols[val_idx],
        feature_names=dataset.feature_names
    )

    test_dataset = type(dataset)(
        sequences=dataset.sequences[test_idx].numpy(),
        labels=dataset.labels[test_idx].numpy(),
        timestamps=dataset.timestamps[test_idx],
        symbols=dataset.symbols[test_idx],
        feature_names=dataset.feature_names
    )

    logger.info(f"Split sizes:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val:   {len(val_dataset)} samples")
    logger.info(f"  Test:  {len(test_dataset)} samples")

    # Normalize (fit on training data only)
    logger.info("Normalizing data...")

    train_sequences_norm = builder.normalize_sequences(
        train_dataset.sequences.numpy(),
        fit=True
    )
    val_sequences_norm = builder.normalize_sequences(val_dataset.sequences.numpy())
    test_sequences_norm = builder.normalize_sequences(test_dataset.sequences.numpy())

    # Update datasets with normalized sequences
    train_dataset.sequences = torch.FloatTensor(train_sequences_norm)
    val_dataset.sequences = torch.FloatTensor(val_sequences_norm)
    test_dataset.sequences = torch.FloatTensor(test_sequences_norm)

    # Apply SMOTE to training set (optional)
    if use_smote:
        logger.info(f"Applying SMOTE (target ratio: 1:{int(1/smote_ratio)})...")

        smote_wrapper = SMOTEDatasetWrapper(
            target_ratio=smote_ratio,
            k_neighbors=5,
            random_state=42
        )

        train_dataset = smote_wrapper.fit_resample_dataset(train_dataset)

        logger.info(f"After SMOTE: {len(train_dataset)} training samples")

    # Create data loaders
    logger.info("Creating data loaders...")

    loaders = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=32,
        use_weighted_sampler=True,  # Additional balancing
        num_workers=0  # Set to 0 for macOS
    )

    # Initialize trainer
    logger.info("Initializing model trainer...")

    # Create checkpoint directory
    checkpoint_dir = Path(config.get_models_path()) / f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = ModelTrainer(
        model_type='cnn_lstm',  # or 'gru'
        n_features=train_dataset.n_features,
        sequence_length=train_dataset.sequence_length,
        device='mps',  # Use MPS for Apple Silicon, or 'cuda'/'cpu'
        loss_type='focal',
        focal_alpha=0.25,  # Per guide
        focal_gamma=2.0,   # Per guide
        learning_rate=0.001,
        weight_decay=1e-5,
        checkpoint_dir=str(checkpoint_dir)
    )

    # Train model
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    history = trainer.train(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        epochs=100,
        early_stopping_patience=10,
        verbose=True
    )

    # Save training history
    trainer.save_history()

    # Evaluate on test set
    logger.info("=" * 80)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 80)

    # Load best model
    trainer.load_checkpoint('best_model.pt')

    # Get predictions
    trainer.model.eval()
    all_predictions = []
    all_labels = []
    all_probas = []

    with torch.no_grad():
        for sequences, labels in loaders['test']:
            sequences = sequences.to(trainer.device)
            outputs = trainer.model(sequences)

            predictions = (outputs > 0.5).float().cpu().numpy()
            probas = outputs.cpu().numpy()

            all_predictions.extend(predictions.flatten())
            all_labels.extend(labels.numpy().flatten())
            all_probas.extend(probas.flatten())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)

    # Evaluate
    evaluator = ModelEvaluator()

    evaluation = evaluator.evaluate_model(
        y_true=all_labels,
        y_pred=all_predictions,
        y_proba=all_probas,
        returns=None  # Would need actual forward returns
    )

    evaluator.print_report(evaluation)

    # Save evaluation report
    evaluator.save_report(
        evaluation,
        str(checkpoint_dir / 'evaluation.json')
    )

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {checkpoint_dir}")
    logger.info(f"Best checkpoint: {checkpoint_dir / 'best_model.pt'}")

    # Check if meets targets
    if evaluation['meets_classification_targets']:
        logger.success("✓ Model meets implementation guide targets!")
        logger.success(f"  Accuracy: {evaluation['classification']['accuracy']:.4f} (target: 0.8244)")
        logger.success(f"  Precision: {evaluation['classification']['precision']:.4f} (target: 0.9000)")
    else:
        logger.warning("✗ Model does not meet targets yet")
        logger.warning("Try:")
        logger.warning("  - Collecting more data")
        logger.warning("  - Tuning hyperparameters")
        logger.warning("  - Feature engineering")
        logger.warning("  - Ensemble methods")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train spike detection model')
    parser.add_argument('--symbol', default='BTC-USD', help='Trading pair')
    parser.add_argument('--days', type=int, default=90, help='Days of data to use')
    parser.add_argument('--timeframe', default='5m', help='Timeframe (1m, 5m, 15m)')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE')
    parser.add_argument('--smote-ratio', type=float, default=0.2, help='SMOTE ratio')

    args = parser.parse_args()

    asyncio.run(train_single_model(
        symbol=args.symbol,
        days=args.days,
        timeframe=args.timeframe,
        use_smote=not args.no_smote,
        smote_ratio=args.smote_ratio
    ))


if __name__ == "__main__":
    main()
