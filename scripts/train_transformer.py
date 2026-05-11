"""
Train Big Mover Transformer (Memory-Optimized)

Trains the transformer model to predict 20%+ gains in 24 hours.
Uses sharded data loading to avoid memory exhaustion.

Usage:
    # First, create shards (run once):
    python scripts/preprocess_shards.py

    # Then train:
    python scripts/train_transformer.py

Features:
- Sharded data loading (streams from disk, never loads full dataset)
- Focal loss for imbalanced classes (alpha=0.80 to upweight positives)
- Gradient accumulation for effective larger batch sizes
- OneCycleLR scheduler
- Early stopping
- MPS (Apple Silicon) acceleration
- Checkpoint saving
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from loguru import logger
import time

from src.models.transformer import BigMoverTransformer, count_parameters
from src.models.focal_loss import FocalLoss
from src.models.sharded_dataset import ShardedDataset, ShardedMapDataset, create_sharded_dataloader


# Configuration - MEMORY OPTIMIZED
CONFIG = {
    # Data (sharded)
    'shard_dir': '/Users/bz/Pythia2/data/shards',
    'output_dir': '/Users/bz/Pythia2/models',

    # Sequence (must match preprocessing)
    'seq_len': 720,  # 12hr of 1-min candles

    # Model architecture
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 4,
    'd_ff': 512,
    'dropout': 0.1,
    'classifier_dropout': 0.2,

    # Training - MEMORY OPTIMIZED
    'batch_size': 32,  # Reduced from 128 (fits in 16GB with MPS)
    'gradient_accumulation_steps': 4,  # Effective batch = 32 * 4 = 128
    'epochs': 50,
    'learning_rate': 5e-5,  # Scaled down for smaller batch
    'weight_decay': 0.01,
    'max_lr': 1.5e-4,  # Scaled down for smaller batch

    # Data loading
    'num_workers': 2,  # Reduced from 4 (less memory duplication)
    'prefetch_factor': 2,

    # Focal loss - FIXED (was 0.25, which downweighted positives!)
    'focal_alpha': 0.80,  # Upweight rare positives (5.5% of data)
    'focal_gamma': 2.0,

    # Class balancing
    'oversample_positive': 2.0,  # 2x oversample positive class in shards

    # Early stopping
    'patience': 10,
    'min_delta': 1e-4,

    # Data split (handled by ShardedDataset)
    'train_ratio': 0.70,
    'val_ratio': 0.15,
}


def train_epoch_with_accumulation(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler=None,
    accumulation_steps: int = 1
) -> dict:
    """
    Train for one epoch with gradient accumulation.

    Gradient accumulation allows effective larger batch sizes
    without the memory cost.
    """
    model.train()
    total_loss = 0
    n_correct = 0
    n_total = 0
    n_pos_correct = 0
    n_pos_total = 0

    optimizer.zero_grad()

    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        output = model(batch_x)
        loss = criterion(output, batch_y)

        # Scale loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Accumulate gradients
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        # Track metrics (use unscaled loss)
        total_loss += loss.item() * accumulation_steps * len(batch_x)

        # Accuracy
        pred = (output > 0.5).float()
        n_correct += (pred == batch_y).sum().item()
        n_total += len(batch_y)

        # Positive class recall
        pos_mask = batch_y == 1
        n_pos_total += pos_mask.sum().item()
        n_pos_correct += ((pred == 1) & pos_mask).sum().item()

    # Handle remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

    return {
        'loss': total_loss / max(n_total, 1),
        'accuracy': n_correct / max(n_total, 1),
        'recall': n_pos_correct / max(n_pos_total, 1)
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        output = model(batch_x)
        loss = criterion(output, batch_y)

        total_loss += loss.item() * len(batch_x)
        all_preds.extend(output.cpu().numpy().flatten())
        all_labels.extend(batch_y.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(all_labels) == 0:
        return {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0,
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    # Metrics at threshold 0.5
    pred_binary = (all_preds > 0.5).astype(int)
    tp = ((pred_binary == 1) & (all_labels == 1)).sum()
    fp = ((pred_binary == 1) & (all_labels == 0)).sum()
    fn = ((pred_binary == 0) & (all_labels == 1)).sum()
    tn = ((pred_binary == 0) & (all_labels == 0)).sum()

    accuracy = (tp + tn) / len(all_labels)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'loss': total_loss / len(all_labels),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


def train(config: dict):
    """Main training function with sharded data loading."""
    logger.info("=" * 70)
    logger.info("BIG MOVER TRANSFORMER TRAINING (Memory-Optimized)")
    logger.info("=" * 70)

    # Check shards exist
    shard_dir = Path(config['shard_dir'])
    if not shard_dir.exists() or not list(shard_dir.glob('shard_*.parquet')):
        logger.error(f"No shards found in {shard_dir}")
        logger.error("Run 'python scripts/preprocess_shards.py' first!")
        return None, None

    # Load shard metadata
    with open(shard_dir / 'metadata.json') as f:
        metadata = json.load(f)

    n_features = len(metadata['feature_cols'])
    seq_len = metadata['seq_len']

    logger.info(f"Shard directory: {shard_dir}")
    logger.info(f"Total shards: {metadata['n_shards']}")
    logger.info(f"Total sequences: {metadata['total_sequences']:,}")
    logger.info(f"Positive rate: {metadata['positive_rate']*100:.2f}%")
    logger.info(f"Features: {n_features}")
    logger.info(f"Sequence length: {seq_len}")

    # Log config
    logger.info(f"\nTraining Config:")
    logger.info(f"  batch_size: {config['batch_size']}")
    logger.info(f"  gradient_accumulation: {config['gradient_accumulation_steps']}")
    logger.info(f"  effective_batch: {config['batch_size'] * config['gradient_accumulation_steps']}")
    logger.info(f"  focal_alpha: {config['focal_alpha']} (upweights positives)")
    logger.info(f"  learning_rate: {config['learning_rate']}")
    logger.info(f"  max_lr: {config['max_lr']}")

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.mps.empty_cache()
        logger.info(f"\nUsing MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"\nUsing CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("\nUsing CPU (WARNING: Training will be slow)")

    # Create data loaders
    logger.info("\nCreating data loaders...")

    train_dataset = ShardedDataset(
        shard_dir=str(shard_dir),
        split='train',
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        normalize=True,
        oversample_positive=config['oversample_positive']
    )

    val_dataset = ShardedMapDataset(
        shard_dir=str(shard_dir),
        split='val',
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        normalize=True
    )

    test_dataset = ShardedMapDataset(
        shard_dir=str(shard_dir),
        split='test',
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        normalize=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=config['prefetch_factor'] if config['num_workers'] > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Model
    model = BigMoverTransformer(
        n_features=n_features,
        seq_len=seq_len,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        classifier_dropout=config['classifier_dropout']
    ).to(device)

    logger.info(f"\nModel parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = FocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Estimate steps per epoch (for scheduler)
    est_steps_per_epoch = len(train_dataset) // config['batch_size']
    total_steps = est_steps_per_epoch * config['epochs'] // config['gradient_accumulation_steps']

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['max_lr'],
        total_steps=max(total_steps, 1000),  # Minimum steps
        pct_start=0.1
    )

    # Early stopping
    best_val_loss = float('inf')
    best_val_f1 = 0
    patience_counter = 0
    best_epoch = 0

    # Training loop
    logger.info(f"\n{'='*70}")
    logger.info("STARTING TRAINING")
    logger.info(f"{'='*70}\n")

    training_start = time.time()

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch_with_accumulation(
            model, train_loader, criterion, optimizer, device,
            scheduler, config['gradient_accumulation_steps']
        )
        train_time = time.time() - epoch_start

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Sync MPS for accurate timing
        if device.type == 'mps':
            torch.mps.synchronize()

        epoch_time = time.time() - epoch_start
        samples_per_sec = len(train_dataset) / train_time if train_time > 0 else 0

        # Log progress
        logger.info(
            f"Epoch {epoch+1:2d}/{config['epochs']} [{epoch_time:.1f}s, {samples_per_sec:.0f} samp/s] | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f}, Recall: {train_metrics['recall']:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}, "
            f"Prec: {val_metrics['precision']:.3f}, Recall: {val_metrics['recall']:.3f}, F1: {val_metrics['f1']:.3f}"
        )

        # Early stopping based on val loss (or F1)
        improved = False
        if val_metrics['loss'] < best_val_loss - config['min_delta']:
            best_val_loss = val_metrics['loss']
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            improved = True
            patience_counter = 0

            # Save best model
            model_path = Path(config['output_dir']) / 'big_mover_transformer_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_f1': best_val_f1,
                'config': config,
                'metadata': metadata
            }, model_path)
            logger.info(f"  -> Saved best model (val_loss: {best_val_loss:.4f}, F1: {best_val_f1:.3f})")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = Path(config['output_dir']) / f'big_mover_transformer_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, ckpt_path)
            logger.info(f"  -> Saved checkpoint: {ckpt_path.name}")

        if patience_counter >= config['patience']:
            logger.info(f"\nEarly stopping at epoch {epoch+1} (no improvement for {config['patience']} epochs)")
            break

    training_time = time.time() - training_start
    logger.info(f"\nTraining completed in {training_time/3600:.1f} hours")
    logger.info(f"Best epoch: {best_epoch + 1} (val_loss: {best_val_loss:.4f}, F1: {best_val_f1:.3f})")

    # Load best model for final test
    logger.info("\n" + "=" * 70)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 70)

    checkpoint = torch.load(Path(config['output_dir']) / 'big_mover_transformer_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device)

    logger.info(f"\nTest Set Results:")
    logger.info(f"  Loss:      {test_metrics['loss']:.4f}")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.3f}")
    logger.info(f"  Precision: {test_metrics['precision']:.3f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.3f}")
    logger.info(f"  F1:        {test_metrics['f1']:.3f}")
    logger.info(f"  TP: {test_metrics['tp']}, FP: {test_metrics['fp']}, "
                f"FN: {test_metrics['fn']}, TN: {test_metrics['tn']}")

    # Save final model
    final_path = Path(config['output_dir']) / 'big_mover_transformer_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metadata': metadata,
        'test_metrics': test_metrics,
        'best_epoch': best_epoch,
        'training_time_hours': training_time / 3600
    }, final_path)
    logger.info(f"\nSaved final model to {final_path}")

    return model, test_metrics


if __name__ == "__main__":
    train(CONFIG)
