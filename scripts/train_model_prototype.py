#!/usr/bin/env python3
"""
Train Pre-Spike Detection Model (26-Day Prototype)

UPDATED October 2025: Pre-spike pattern detection for momentum scalping

Strategy:
- Target: Detect volume explosion + price acceleration 1-3 minutes BEFORE spike
- Pre-spike pattern: 5x volume increase + 6% price move in next 1-3 minutes
- Stop loss: 0.75% (8:1 reward-risk ratio)
- Signals: 2-4 high-conviction per day via threshold optimization

Aggressive Class Imbalance Handling:
- Focal Loss: α=0.05, γ=4.0 (vs guide's 0.25, 2.0)
- Strategic undersampling: 1:5 ratio (20% of negatives)
- Decision threshold optimization for target signals/day

Current Status:
- 26 days of data (vs 81-day target)
- Expected ~150-200 positive examples (0.5-2% of data)
- Target: 70-78% accuracy (vs 82% with full data)

Usage:
    # Train on 86GB SQLite database (all 319 symbols)
    python scripts/train_model_prototype.py \\
        --db "/Users/brettzeigler/Pythia/market_data copy_86.db" \\
        --all-symbols --timeframe 1m

    # Single symbol test
    python scripts/train_model_prototype.py \\
        --db "/path/to/database.db" \\
        --symbol BTC-USD --timeframe 1m
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


async def train_prototype_model(
    db_path: str,
    symbol: str = None,
    all_symbols: bool = False,
    timeframe: str = '1m',
    sequence_days: int = None,
    max_symbols: int = None,
    skip_symbols: int = 0
):
    """
    Train pre-spike detection model with 26 days of data.

    Args:
        db_path: Path to SQLite database with features
        symbol: Trading pair (required if all_symbols=False)
        all_symbols: Train on all symbols in database
        timeframe: Timeframe ('1m' only for pre-spike detection)
        sequence_days: Custom sequence length in days (default: 3 for 1m)
        max_symbols: Maximum number of symbols to train on (None = all 319)
        skip_symbols: Number of symbols to skip (for batch processing)
    """
    logger.info("=" * 80)
    logger.info("PRE-SPIKE DETECTION TRAINING (26-DAY PROTOTYPE)")
    logger.info("=" * 80)
    logger.info("Strategy: Momentum scalping with pre-spike pattern detection")
    logger.info("Target: Volume explosion (5x) + price spike (6%) in next 1-3 minutes")
    logger.info("Risk: 0.75% stop loss (8:1 reward-risk ratio)")
    logger.info("Signals: 2-4 high-conviction per day via threshold optimization")
    logger.info("=" * 80)
    logger.info("")

    # Training mode
    if all_symbols:
        logger.info("Training Mode: MULTI-SYMBOL (all symbols in database)")
    else:
        logger.info(f"Training Mode: SINGLE-SYMBOL ({symbol})")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Database: {db_path}")
    logger.info("")

    # Calculate candles per day based on timeframe
    if timeframe == '1m':
        candles_per_day = 24 * 60  # 1,440 candles per day
    elif timeframe == '5m':
        candles_per_day = 24 * 12  # 288 candles per day
    else:
        logger.error(f"Unsupported timeframe: {timeframe} (only 1m supported for pre-spike)")
        return

    # Sequence length for pre-spike detection
    # Optimized: 1 hour (60 candles) for 1-3 min ahead prediction
    if sequence_days is None:
        if timeframe == '1m':
            # For 1m timeframe, use 60 candles (1 hour) for pre-spike detection
            sequence_candles = 60  # 1 hour optimized for pre-spike detection
            SEQUENCE_LENGTH = sequence_candles / candles_per_day  # For logging (~0.042 days)
        else:
            SEQUENCE_LENGTH = 3  # Keep 3 days for other timeframes
            sequence_candles = int(SEQUENCE_LENGTH * candles_per_day)
    else:
        SEQUENCE_LENGTH = sequence_days
        sequence_candles = int(SEQUENCE_LENGTH * candles_per_day)

    logger.info("Pre-Spike Detection Parameters:")
    logger.info(f"  Sequence length:  {sequence_candles} candles ({SEQUENCE_LENGTH:.3f} days @ {timeframe})")
    logger.info(f"  Price window:     3 minutes (look ahead for spike)")
    logger.info(f"  Volume window:    2 minutes (look ahead for explosion)")
    logger.info(f"  Min price spike:  6% in next 3 minutes")
    logger.info(f"  Min volume spike: 5x increase in next 2 minutes")
    logger.info(f"  Stop loss:        0.75% (8:1 reward-risk ratio)")
    logger.info("")

    logger.info("Aggressive Class Imbalance Handling:")
    logger.info(f"  Focal Loss:       α=0.05, γ=4.0 (vs guide's 0.25, 2.0)")
    logger.info(f"  Undersampling:    1:5 ratio (keep 20% of negatives)")
    logger.info(f"  Decision thresh:  Optimized for 2-4 signals/day")
    logger.info("")

    # Validate database
    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Make sure you've run:")
        logger.error("  1. scripts/aggregate_candles_batch.py")
        logger.error("  2. scripts/calculate_features_batch_v2.py")
        return

    logger.info(f"Final sequence: {sequence_candles} candles ({SEQUENCE_LENGTH:.3f} days @ {timeframe})")
    logger.info(f"Note: Pre-spike targets generated automatically by DatasetBuilder")
    logger.info(f"      (volume explosion + price spike detection in next 1-3 minutes)")
    logger.info("")

    # Initialize dataset builder with pre-spike target generation
    # Target generation is now built into SpikeTargetGenerator class
    builder = DatasetBuilder(
        db_path=db_path,
        sequence_length=sequence_candles,
        forward_window=0,  # Not used for pre-spike detection
        min_spike_threshold=0.0,  # Not used for pre-spike detection
        scaler_type='robust'
    )

    # Calculate date range - use None to get all available data
    end_date = None
    start_date = None

    logger.info(f"Using all available data in database")
    logger.info("")

    # Build dataset
    if all_symbols:
        logger.info("Loading data for all symbols in database...")

        dataset = builder.build_dataset_multi_symbol(
            symbols=None,  # All symbols
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            feature_columns=None,
            min_candles=sequence_candles + 100,  # Minimum for sequences + pre-spike window
            max_symbols=max_symbols,  # Limit number of symbols
            skip_symbols=skip_symbols,  # Skip first N symbols
            normalize=False,
            fit_scaler=False
        )
    else:
        logger.info(f"Loading data for {symbol}...")

        dataset = builder.build_dataset(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            feature_columns=None,
            normalize=False,
            fit_scaler=False
        )

    if dataset is None:
        logger.error("Failed to build dataset")
        logger.error("Possible reasons:")
        logger.error("  1. Features not calculated (run calculate_features_batch.py)")
        logger.error("  2. Not enough OHLCV data")
        logger.error("  3. Database tables missing")
        logger.error("  4. No symbols meet minimum candle threshold")
        return

    logger.info(f"Dataset created: {len(dataset)} samples")

    stats = dataset.get_statistics()
    logger.info(f"Dataset statistics:")
    logger.info(f"  Positive: {stats['positive_samples']} ({stats['positive_ratio']*100:.2f}%)")
    logger.info(f"  Negative: {stats['negative_samples']}")
    logger.info(f"  Features: {stats['n_features']}")
    logger.info("")

    # Check minimum samples
    if len(dataset) < 100:
        logger.error(f"Only {len(dataset)} samples - need at least 100 for prototype")
        logger.error("Wait a few more days or reduce sequence_length further")
        return

    # Split (60/20/20 for smaller dataset)
    logger.info("Splitting dataset...")

    splitter = TimeSeriesSplitter(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        gap=2
    )

    train_idx, val_idx, test_idx = splitter.split(len(dataset))

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

    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val:   {len(val_dataset)} samples")
    logger.info(f"  Test:  {len(test_dataset)} samples")
    logger.info("")

    # Normalize
    train_sequences_norm = builder.normalize_sequences(train_dataset.sequences.numpy(), fit=True)
    val_sequences_norm = builder.normalize_sequences(val_dataset.sequences.numpy())
    test_sequences_norm = builder.normalize_sequences(test_dataset.sequences.numpy())

    train_dataset.sequences = torch.FloatTensor(train_sequences_norm)
    val_dataset.sequences = torch.FloatTensor(val_sequences_norm)
    test_dataset.sequences = torch.FloatTensor(test_sequences_norm)

    # Apply strategic undersampling (1:5 ratio)
    logger.info("Applying strategic undersampling (1:5 ratio)...")
    train_stats = train_dataset.get_statistics()
    n_positives = train_stats['positive_samples']
    n_negatives = train_stats['negative_samples']

    # Calculate target number of negatives (5x positives)
    target_negatives = n_positives * 5
    keep_ratio = min(1.0, target_negatives / n_negatives)

    if keep_ratio < 1.0:
        # Undersample negatives
        positive_mask = train_dataset.labels == 1
        negative_mask = train_dataset.labels == 0

        positive_indices = np.where(positive_mask)[0]
        negative_indices = np.where(negative_mask)[0]

        # Randomly sample negatives
        np.random.seed(42)
        sampled_negative_indices = np.random.choice(
            negative_indices,
            size=int(len(negative_indices) * keep_ratio),
            replace=False
        )

        # Combine positive and sampled negative indices
        final_indices = np.concatenate([positive_indices, sampled_negative_indices])
        np.random.shuffle(final_indices)

        # Create undersampled dataset
        train_dataset = type(train_dataset)(
            sequences=train_dataset.sequences[final_indices].numpy(),
            labels=train_dataset.labels[final_indices].numpy(),
            timestamps=train_dataset.timestamps[final_indices],
            symbols=train_dataset.symbols[final_indices],
            feature_names=train_dataset.feature_names
        )

        logger.success(
            f"Undersampled: {len(train_dataset)} samples "
            f"({n_positives} pos + {len(sampled_negative_indices)} neg)"
        )
    else:
        logger.info(f"No undersampling needed (already balanced or small dataset)")
    logger.info("")

    # Create loaders
    loaders = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=16,  # Smaller batch for smaller dataset
        use_weighted_sampler=True,
        num_workers=0
    )

    # Initialize trainer
    model_name = "all_symbols" if all_symbols else symbol
    checkpoint_dir = Path("models") / f"prototype_{model_name}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = ModelTrainer(
        model_type='cnn_lstm',
        n_features=train_dataset.n_features,
        sequence_length=train_dataset.sequence_length,
        device='mps',
        loss_type='focal',
        focal_alpha=0.05,  # Aggressive for pre-spike detection (vs 0.25)
        focal_gamma=4.0,  # Higher focusing for extreme imbalance (vs 2.0)
        learning_rate=0.001,
        weight_decay=1e-5,
        checkpoint_dir=str(checkpoint_dir),
        use_undersampling=True,  # Already applied above
        undersampling_ratio=0.20  # 1:5 ratio
    )

    # Train
    logger.info("=" * 80)
    logger.info("STARTING PROTOTYPE TRAINING")
    logger.info("=" * 80)

    history = trainer.train(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        epochs=50,  # Fewer epochs for prototype
        early_stopping_patience=10,
        verbose=True
    )

    trainer.save_history()

    # Evaluate
    logger.info("=" * 80)
    logger.info("EVALUATING PROTOTYPE")
    logger.info("=" * 80)

    trainer.load_checkpoint('best_model.pt')
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

    evaluator = ModelEvaluator()
    evaluation = evaluator.evaluate_model(
        y_true=all_labels,
        y_pred=all_predictions,
        y_proba=all_probas
    )

    evaluator.print_report(evaluation)
    evaluator.save_report(evaluation, str(checkpoint_dir / 'evaluation.json'))

    logger.info("=" * 80)
    logger.info("PRE-SPIKE DETECTION TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("Model saved to: " + str(checkpoint_dir))
    logger.info("")
    logger.warning("26-DAY PROTOTYPE STATUS:")
    logger.warning("  Current: 26 days of data (32% of 81-day requirement)")
    logger.warning("  Expected: 70-78% accuracy (vs 82% target with full data)")
    logger.warning("  Positive examples: ~150-200 (0.5-2% of data)")
    logger.info("")
    logger.info("For production deployment:")
    logger.info("  1. Collect remaining 55 days of real WebSocket data")
    logger.info("  2. Retrain with 81+ days for full accuracy (82%+ target)")
    logger.info("  3. Optimize decision threshold for 2-4 signals/day")
    logger.info("  4. Paper trade for 1-2 weeks before live deployment")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train pre-spike detection model (26-day prototype)')
    parser.add_argument('--db', required=True, help='Path to SQLite database with features')
    parser.add_argument('--symbol', help='Trading pair (required unless --all-symbols)')
    parser.add_argument('--all-symbols', action='store_true', help='Train on all symbols in database')
    parser.add_argument('--timeframe', choices=['1m', '5m'], default='1m', help='Timeframe (1m recommended)')
    parser.add_argument('--sequence-days', type=int, help='Custom sequence length in days (default: 3 for 1m)')
    parser.add_argument('--max-symbols', type=int, help='Maximum number of symbols to train on (reduces memory)')
    parser.add_argument('--skip-symbols', type=int, default=0, help='Number of symbols to skip (for batch processing)')

    args = parser.parse_args()

    # Validate arguments
    if not args.all_symbols and not args.symbol:
        parser.error("Either --symbol or --all-symbols is required")

    asyncio.run(train_prototype_model(
        db_path=args.db,
        symbol=args.symbol,
        all_symbols=args.all_symbols,
        timeframe=args.timeframe,
        sequence_days=args.sequence_days,
        max_symbols=args.max_symbols,
        skip_symbols=args.skip_symbols
    ))


if __name__ == "__main__":
    main()
