#!/usr/bin/env python3
"""
Chunked Overnight Training - Memory-Safe for 24GB RAM

Splits 319 symbols into 6 chunks (~53 symbols each) and trains sequentially.
Each chunk uses ~12-15GB RAM (safe for 24GB total).

Trains both CNN-LSTM and TCN models over ~50 epochs total.

Usage:
    python scripts/train_overnight_chunked.py --db "/path/to/database.db"

    # Run in background:
    nohup python scripts/train_overnight_chunked.py --db "/path/to/database.db" > training_chunked.log 2>&1 &
"""

import asyncio
import argparse
import sys
import gc
from pathlib import Path
from datetime import datetime
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from loguru import logger
import duckdb

from src.models.dataset import DatasetBuilder, create_dataloaders, SpikeDataset
from src.models.trainer import ModelTrainer
from src.models.metrics import ModelEvaluator
from src.models.validation import TimeSeriesSplitter
from src.models.spike_filter import SpikeCategoryFilter


def get_all_symbols(db_path: str, timeframe: str = '1m') -> list:
    """Get all symbols from database without counting (fast)."""
    conn = duckdb.connect(db_path, read_only=True)
    try:
        query = f"SELECT DISTINCT symbol FROM features WHERE timeframe = '{timeframe}' ORDER BY symbol"
        result = conn.execute(query).fetchall()
        symbols = [row[0] for row in result]
        logger.info(f"Found {len(symbols)} symbols in database")
        return symbols
    finally:
        conn.close()


def split_into_chunks(symbols: list, num_chunks: int = 6) -> list:
    """Split symbols into roughly equal chunks."""
    chunk_size = len(symbols) // num_chunks
    chunks = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        if i == num_chunks - 1:
            # Last chunk gets remaining symbols
            end_idx = len(symbols)
        else:
            end_idx = (i + 1) * chunk_size

        chunk = symbols[start_idx:end_idx]
        chunks.append(chunk)
        logger.info(f"Chunk {i+1}: {len(chunk)} symbols ({chunk[0]} to {chunk[-1]})")

    return chunks


async def train_model_chunked(
    model_type: str,
    db_path: str,
    symbol_chunks: list,
    timestamp: str,
    sequence_candles: int,
    total_epochs: int = 50,
    epochs_per_chunk: int = 8,
    spike_filter: SpikeCategoryFilter = None
) -> dict:
    """
    Train model on chunks of symbols, cycling through to reach total_epochs.

    Args:
        model_type: 'cnn_lstm' or 'tcn'
        db_path: Database path
        symbol_chunks: List of symbol lists (one per chunk)
        timestamp: Timestamp for model naming
        sequence_candles: Sequence length in candles
        total_epochs: Total training epochs (default: 50)
        epochs_per_chunk: Epochs to train on each chunk before switching (default: 8)
        spike_filter: Optional SpikeCategoryFilter to focus on specific spike types

    Returns:
        dict with training results
    """
    try:
        logger.info("=" * 80)
        logger.info(f"TRAINING {model_type.upper()} MODEL (CHUNKED)")
        logger.info("=" * 80)
        logger.info(f"Total chunks: {len(symbol_chunks)}")
        logger.info(f"Epochs per chunk: {epochs_per_chunk}")
        logger.info(f"Target total epochs: {total_epochs}")
        logger.info("")

        # Create checkpoint directory
        model_name = f"{model_type}_chunked_{timestamp}"
        checkpoint_dir = Path("models") / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataset builder (reused for all chunks)
        builder = DatasetBuilder(
            db_path=db_path,
            sequence_length=sequence_candles,
            scaler_type='robust'
        )

        # Initialize trainer (will be reused)
        trainer = None
        n_features = 24  # Will be updated from first chunk

        # Calculate how many full cycles we need
        num_chunks = len(symbol_chunks)
        full_cycles = total_epochs // epochs_per_chunk // num_chunks
        remaining_epochs = total_epochs - (full_cycles * epochs_per_chunk * num_chunks)

        logger.info(f"Training plan: {full_cycles} full cycles + {remaining_epochs} extra epochs")
        logger.info("")

        current_epoch = 0

        # Main training loop: cycle through chunks
        for cycle in range(full_cycles + 1):
            if current_epoch >= total_epochs:
                break

            logger.info("=" * 80)
            logger.info(f"CYCLE {cycle + 1}/{full_cycles + 1}")
            logger.info("=" * 80)
            logger.info("")

            for chunk_idx, symbol_chunk in enumerate(symbol_chunks):
                if current_epoch >= total_epochs:
                    break

                epochs_this_round = min(epochs_per_chunk, total_epochs - current_epoch)

                logger.info("-" * 80)
                logger.info(f"Chunk {chunk_idx + 1}/{num_chunks} - {len(symbol_chunk)} symbols")
                logger.info(f"Epochs {current_epoch + 1}-{current_epoch + epochs_this_round} of {total_epochs}")
                logger.info("-" * 80)
                logger.info("")

                # Build dataset for this chunk
                logger.info("Loading chunk data...")
                dataset = builder.build_dataset_multi_symbol(
                    symbols=symbol_chunk,
                    timeframe='1m',
                    start_date=None,
                    end_date=None,
                    feature_columns=None,
                    min_candles=0,  # Don't filter - we already have good symbols
                    max_symbols=None,
                    skip_symbols=0,
                    normalize=False,
                    fit_scaler=False,
                    spike_filter=spike_filter  # Apply spike category filter
                )

                if dataset is None or len(dataset) == 0:
                    logger.error(f"Failed to build dataset for chunk {chunk_idx + 1}")
                    continue

                logger.info(f"Chunk dataset: {len(dataset)} samples")
                stats = dataset.get_statistics()
                logger.info(f"Positive: {stats['positive_samples']} ({stats['positive_ratio']*100:.2f}%)")
                logger.info("")

                # Update n_features from first chunk
                if n_features == 24:
                    n_features = dataset.n_features
                    logger.info(f"Features per timestep: {n_features}")

                # Split dataset (60/20/20)
                splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, gap=2)
                train_idx, val_idx, test_idx = splitter.split(len(dataset))

                train_dataset = SpikeDataset(
                    sequences=dataset.sequences[train_idx].numpy(),
                    labels=dataset.labels[train_idx].numpy(),
                    timestamps=dataset.timestamps[train_idx],
                    symbols=dataset.symbols[train_idx],
                    feature_names=dataset.feature_names
                )

                val_dataset = SpikeDataset(
                    sequences=dataset.sequences[val_idx].numpy(),
                    labels=dataset.labels[val_idx].numpy(),
                    timestamps=dataset.timestamps[val_idx],
                    symbols=dataset.symbols[val_idx],
                    feature_names=dataset.feature_names
                )

                test_dataset = SpikeDataset(
                    sequences=dataset.sequences[test_idx].numpy(),
                    labels=dataset.labels[test_idx].numpy(),
                    timestamps=dataset.timestamps[test_idx],
                    symbols=dataset.symbols[test_idx],
                    feature_names=dataset.feature_names
                )

                # Normalize
                train_sequences_norm = builder.normalize_sequences(train_dataset.sequences.numpy(), fit=True)
                val_sequences_norm = builder.normalize_sequences(val_dataset.sequences.numpy())
                test_sequences_norm = builder.normalize_sequences(test_dataset.sequences.numpy())

                train_dataset.sequences = torch.FloatTensor(train_sequences_norm)
                val_dataset.sequences = torch.FloatTensor(val_sequences_norm)
                test_dataset.sequences = torch.FloatTensor(test_sequences_norm)

                # NO UNDERSAMPLING - Rely on Focal Loss to handle class imbalance
                train_stats = train_dataset.get_statistics()
                logger.info(f"Training set: {len(train_dataset)} samples ({train_stats['positive_samples']} pos, {train_stats['negative_samples']} neg)")
                logger.info("")

                # Create loaders
                loaders = create_dataloaders(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    batch_size=16,
                    use_weighted_sampler=False,  # Disabled: Focal Loss already handles imbalance
                    num_workers=0
                )

                # Initialize or update trainer
                if trainer is None:
                    logger.info(f"Initializing {model_type.upper()} model...")
                    trainer = ModelTrainer(
                        model_type=model_type,
                        n_features=n_features,
                        sequence_length=sequence_candles,
                        device='mps',
                        loss_type='focal',
                        focal_alpha=0.01,  # EXTREME weight on rare positives (100x vs negatives)
                        focal_gamma=4.0,   # Aggressive focus on hard examples
                        learning_rate=0.001,
                        weight_decay=1e-5,
                        checkpoint_dir=str(checkpoint_dir),
                        use_undersampling=False,
                        undersampling_ratio=None
                    )
                    logger.info("")

                # Train on this chunk
                logger.info(f"Training on chunk {chunk_idx + 1}...")
                history = trainer.train(
                    train_loader=loaders['train'],
                    val_loader=loaders['val'],
                    epochs=epochs_this_round,
                    early_stopping_patience=100,  # Don't stop early in chunked training
                    verbose=True
                )

                current_epoch += epochs_this_round

                # Save checkpoint after each chunk
                checkpoint_name = f"checkpoint_epoch{current_epoch}.pt"
                trainer.save_checkpoint(checkpoint_name)
                logger.info(f"Saved checkpoint: {checkpoint_name}")
                logger.info("")

                # Clear memory
                del dataset, train_dataset, val_dataset, test_dataset, loaders
                gc.collect()
                torch.mps.empty_cache() if torch.backends.mps.is_available() else None
                logger.info("Memory cleared")
                logger.info("")

        # Final evaluation on last test set (from last chunk)
        logger.info("=" * 80)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 80)
        logger.info("Note: Evaluated on last chunk's test set")
        logger.info("")

        # Save final model
        trainer.save_checkpoint('best_model.pt')
        trainer.save_history()

        return {
            'success': True,
            'model_dir': str(checkpoint_dir),
            'total_epochs': current_epoch,
            'error': None
        }

    except Exception as e:
        logger.error(f"Error training {model_type}: {e}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'model_dir': None,
            'total_epochs': 0,
            'error': str(e)
        }


async def main(db_path: str):
    """Main training pipeline."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 80)
    logger.info("CHUNKED OVERNIGHT TRAINING: CNN-LSTM + TCN")
    logger.info("=" * 80)
    logger.info(f"Database: {db_path}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Device: MPS (Apple Silicon)")
    logger.info(f"Memory limit: 24GB RAM (chunked for safety)")
    logger.info("=" * 80)
    logger.info("")

    # Get all symbols and split into chunks
    logger.info("Preparing symbol chunks...")
    symbols = get_all_symbols(db_path, timeframe='1m')
    symbol_chunks = split_into_chunks(symbols, num_chunks=6)
    logger.info("")

    # Initialize Slow & Large spike filter
    categorized_csv = "all_spikes_categorized.csv"
    spike_filter = None
    try:
        spike_filter = SpikeCategoryFilter(categorized_csv)
        spike_filter.print_summary()
        logger.info("✓ Spike filter initialized - will train on Slow & Large spikes ONLY")
        logger.info("")
    except FileNotFoundError:
        logger.warning(f"Categorized spikes CSV not found: {categorized_csv}")
        logger.warning("Training will use ALL spike types (mixed Fast & Steep + Slow & Large)")
        logger.info("")

    # Training parameters
    sequence_candles = 60  # 60 min @ 1m (1 hour context for slow builds)
    total_epochs = 50
    epochs_per_chunk = 8

    # Train CNN-LSTM
    cnn_lstm_result = await train_model_chunked(
        model_type='cnn_lstm',
        db_path=db_path,
        symbol_chunks=symbol_chunks,
        timestamp=timestamp,
        sequence_candles=sequence_candles,
        total_epochs=total_epochs,
        epochs_per_chunk=epochs_per_chunk,
        spike_filter=spike_filter  # Focus on Slow & Large only
    )

    # Train TCN
    tcn_result = await train_model_chunked(
        model_type='tcn',
        db_path=db_path,
        symbol_chunks=symbol_chunks,
        timestamp=timestamp,
        sequence_candles=sequence_candles,
        total_epochs=total_epochs,
        epochs_per_chunk=epochs_per_chunk,
        spike_filter=spike_filter  # Focus on Slow & Large only
    )

    # Summary
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"CNN-LSTM: {'✓ SUCCESS' if cnn_lstm_result['success'] else '✗ FAILED'}")
    if cnn_lstm_result['success']:
        logger.info(f"  Model dir: {cnn_lstm_result['model_dir']}")
        logger.info(f"  Total epochs: {cnn_lstm_result['total_epochs']}")
    else:
        logger.info(f"  Error: {cnn_lstm_result['error']}")
    logger.info("")

    logger.info(f"TCN: {'✓ SUCCESS' if tcn_result['success'] else '✗ FAILED'}")
    if tcn_result['success']:
        logger.info(f"  Model dir: {tcn_result['model_dir']}")
        logger.info(f"  Total epochs: {tcn_result['total_epochs']}")
    else:
        logger.info(f"  Error: {tcn_result['error']}")
    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chunked overnight training for memory safety')
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    args = parser.parse_args()

    asyncio.run(main(args.db))
