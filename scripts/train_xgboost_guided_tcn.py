#!/usr/bin/env python3
"""
Train XGBoost-Guided TCN with Knowledge Distillation

Knowledge transfer from XGBoost V3 to TCN via:
1. Feature attention initialized from XGBoost importance weights
2. Delta layer for acceleration detection (key XGBoost insight)
3. Soft label distillation from XGBoost predictions

Usage:
    python scripts/train_xgboost_guided_tcn.py

Data splits:
    - Train: Sep 24 - Nov 15 (53 days)
    - Val: Nov 16 - Nov 30 (15 days)
    - Test: Dec 1 - Dec 24 (24 days)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import duckdb
import joblib
from loguru import logger
from sklearn.preprocessing import RobustScaler

from src.models.xgboost_guided_tcn import XGBoostGuidedTCN, FEATURE_COLUMNS
from src.models.distillation_loss import DistillationLoss, AdaptiveDistillationLoss
from src.models.dataset import SpikeDatasetWithSoftLabels


# === CONFIGURATION ===
CONFIG = {
    # Paths
    'db_path': str(PROJECT_ROOT / 'market_data.duckdb'),
    'xgboost_model_path': str(PROJECT_ROOT / 'models/xgboost_slow_large_v3.pkl'),
    'output_dir': str(PROJECT_ROOT / 'models'),

    # Temporal splits
    'train_start': '2025-09-24',
    'train_end': '2025-11-15',
    'val_start': '2025-11-16',
    'val_end': '2025-11-30',
    'test_start': '2025-12-01',
    'test_end': '2025-12-24',

    # Model params
    'sequence_length': 30,
    'n_features': 24,
    'num_channels': [64, 64, 64, 64],
    'kernel_size': 3,
    'dropout': 0.3,
    'attention_temperature': 1.0,

    # Training params
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 100,
    'patience': 15,
    'distillation_alpha': 0.5,
    'focal_alpha': 0.05,
    'focal_gamma': 4.0,

    # Spike labeling (same as V3)
    'prediction_offset': 15,
    'min_price_spike': 0.15,
    'price_window': 60,
    'min_volume_spike': 2.0,
    'volume_window': 30,

    # Hardware
    'device': 'mps',
}


class SlowLargeSpikeLabeler:
    """
    Generate spike labels matching XGBoost V3 criteria.

    Labels a candle as positive if, starting 15 minutes later,
    price rises 15%+ over next 60 minutes with 2x sustained volume.
    """

    def __init__(
        self,
        prediction_offset: int = 15,
        price_window: int = 60,
        min_price_spike: float = 0.15,
        volume_window: int = 30,
        min_volume_spike: float = 2.0
    ):
        self.prediction_offset = prediction_offset
        self.price_window = price_window
        self.min_price_spike = min_price_spike
        self.volume_window = volume_window
        self.min_volume_spike = min_volume_spike

    def generate_labels(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> np.ndarray:
        """Generate spike labels for a price/volume series."""
        n = len(prices)
        labels = np.zeros(n, dtype=np.float32)

        # Calculate rolling baseline volume (60 min)
        baseline_volume = pd.Series(volumes).rolling(60, min_periods=30).mean().values

        # Label each candle
        max_idx = n - self.prediction_offset - self.price_window
        for i in range(max_idx):
            # Future window starts after prediction offset
            start_idx = i + 1 + self.prediction_offset
            price_end = start_idx + self.price_window
            vol_end = start_idx + self.volume_window

            if price_end > n or vol_end > n:
                continue

            # Check price spike
            max_future_price = prices[start_idx:price_end].max()
            price_return = (max_future_price - prices[i]) / (prices[i] + 1e-10)

            # Check volume
            avg_future_volume = volumes[start_idx:vol_end].mean()
            baseline = baseline_volume[i] if not np.isnan(baseline_volume[i]) else volumes[:i+1].mean()
            vol_ratio = avg_future_volume / (baseline + 1e-10)

            # Label if both conditions met
            if price_return >= self.min_price_spike and vol_ratio >= self.min_volume_spike:
                labels[i] = 1.0

        return labels


def setup_device(device: str) -> torch.device:
    """Setup computation device."""
    if device == 'mps' and torch.backends.mps.is_available():
        logger.info("Using MPS (Metal Performance Shaders)")
        return torch.device('mps')
    elif device == 'cuda' and torch.cuda.is_available():
        logger.info("Using CUDA")
        return torch.device('cuda')
    else:
        logger.info("Using CPU")
        return torch.device('cpu')


def load_xgboost_model(model_path: str) -> Tuple[object, List[float]]:
    """Load XGBoost model and extract feature importances."""
    logger.info(f"Loading XGBoost model from {model_path}")

    model_data = joblib.load(model_path)

    if isinstance(model_data, dict):
        xgb_model = model_data.get('model', model_data)
    else:
        xgb_model = model_data

    if hasattr(xgb_model, 'feature_importances_'):
        importances = xgb_model.feature_importances_.tolist()
        logger.info(f"Loaded {len(importances)} feature importances")
    else:
        importances = None
        logger.warning("XGBoost model has no feature_importances_")

    return xgb_model, importances


def load_features_from_db(
    db_path: str,
    start_date: str,
    end_date: str,
    feature_cols: List[str],
    min_candles: int = 1000
) -> Dict[str, pd.DataFrame]:
    """Load features from database for all symbols in date range."""
    logger.info(f"Loading features from {start_date} to {end_date}")

    conn = duckdb.connect(db_path, read_only=True)

    try:
        # Get symbols with sufficient data
        query = f"""
            SELECT symbol, COUNT(*) as cnt
            FROM features
            WHERE timeframe = '1m'
              AND timestamp >= '{start_date}'
              AND timestamp <= '{end_date}'
            GROUP BY symbol
            HAVING cnt >= {min_candles}
            ORDER BY cnt DESC
        """
        symbols_df = conn.execute(query).fetchdf()
        symbols = symbols_df['symbol'].tolist()
        logger.info(f"Found {len(symbols)} symbols with >= {min_candles} candles")

        # Load features for each symbol
        symbol_data = {}
        for symbol in symbols:
            cols_str = ', '.join(['timestamp', 'symbol'] + feature_cols)
            query = f"""
                SELECT {cols_str}
                FROM features
                WHERE symbol = '{symbol}'
                  AND timeframe = '1m'
                  AND timestamp >= '{start_date}'
                  AND timestamp <= '{end_date}'
                ORDER BY timestamp ASC
            """
            df = conn.execute(query).fetchdf()

            if len(df) >= min_candles:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                symbol_data[symbol] = df

        logger.info(f"Loaded data for {len(symbol_data)} symbols")
        return symbol_data

    finally:
        conn.close()


def load_candles_from_db(
    db_path: str,
    symbols: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV candles for label generation."""
    logger.info(f"Loading candles for {len(symbols)} symbols")

    conn = duckdb.connect(db_path, read_only=True)

    try:
        symbol_candles = {}
        for symbol in symbols:
            query = f"""
                SELECT timestamp, close, volume
                FROM ohlcv
                WHERE symbol = '{symbol}'
                  AND timeframe = '1m'
                  AND timestamp >= '{start_date}'
                  AND timestamp <= '{end_date}'
                ORDER BY timestamp ASC
            """
            df = conn.execute(query).fetchdf()

            if len(df) > 0:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                symbol_candles[symbol] = df

        return symbol_candles

    finally:
        conn.close()


def generate_xgboost_soft_labels(
    xgb_model: object,
    features_df: pd.DataFrame,
    feature_cols: List[str]
) -> np.ndarray:
    """Generate XGBoost probability predictions for features."""
    X = features_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return xgb_model.predict_proba(X)[:, 1]


def create_sequences(
    features: pd.DataFrame,
    labels: np.ndarray,
    soft_labels: np.ndarray,
    sequence_length: int,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences from features and labels."""
    n = len(features)
    n_features = len(feature_cols)

    # Get feature matrix
    X = features[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Create sequences
    sequences = []
    hard_labels_out = []
    soft_labels_out = []
    timestamps = []
    symbols = []

    symbol_name = features['symbol'].iloc[0] if 'symbol' in features.columns else 'UNKNOWN'

    for i in range(sequence_length, n):
        # Sequence: previous sequence_length candles
        seq = X[i - sequence_length:i]
        sequences.append(seq)

        # Labels at prediction point
        hard_labels_out.append(labels[i])
        soft_labels_out.append(soft_labels[i])
        timestamps.append(features.index[i])
        symbols.append(symbol_name)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(hard_labels_out, dtype=np.float32),
        np.array(soft_labels_out, dtype=np.float32),
        np.array(timestamps),
        np.array(symbols)
    )


def build_dataset(
    features_dict: Dict[str, pd.DataFrame],
    candles_dict: Dict[str, pd.DataFrame],
    xgb_model: object,
    labeler: SlowLargeSpikeLabeler,
    config: dict,
    scaler: Optional[RobustScaler] = None,
    fit_scaler: bool = False
) -> Tuple[SpikeDatasetWithSoftLabels, RobustScaler]:
    """Build dataset from features and candles."""
    all_sequences = []
    all_hard_labels = []
    all_soft_labels = []
    all_timestamps = []
    all_symbols = []

    feature_cols = FEATURE_COLUMNS

    for symbol in features_dict:
        if symbol not in candles_dict:
            continue

        features_df = features_dict[symbol]
        candles_df = candles_dict[symbol]

        # Align timestamps
        common_idx = features_df.index.intersection(candles_df.index)
        if len(common_idx) < config['sequence_length'] + 100:
            continue

        features_df = features_df.loc[common_idx]
        candles_df = candles_df.loc[common_idx]

        # Generate hard labels
        hard_labels = labeler.generate_labels(
            candles_df['close'].values,
            candles_df['volume'].values
        )

        # Generate soft labels (XGBoost predictions)
        soft_labels = generate_xgboost_soft_labels(xgb_model, features_df, feature_cols)

        # Create sequences
        seqs, hard, soft, ts, syms = create_sequences(
            features_df.reset_index(),  # reset index to include timestamp in df
            hard_labels,
            soft_labels,
            config['sequence_length'],
            feature_cols
        )

        if len(seqs) > 0:
            all_sequences.append(seqs)
            all_hard_labels.append(hard)
            all_soft_labels.append(soft)
            all_timestamps.append(ts)
            all_symbols.append(syms)
            logger.debug(f"{symbol}: {len(seqs)} sequences, {hard.sum():.0f} spikes")

    # Concatenate all
    sequences = np.concatenate(all_sequences)
    hard_labels = np.concatenate(all_hard_labels)
    soft_labels = np.concatenate(all_soft_labels)
    timestamps = np.concatenate(all_timestamps)
    symbols = np.concatenate(all_symbols)

    logger.info(f"Total: {len(sequences)} sequences, {hard_labels.sum():.0f} spikes ({hard_labels.mean()*100:.2f}%)")

    # Normalize
    if scaler is None:
        scaler = RobustScaler()

    original_shape = sequences.shape
    sequences_2d = sequences.reshape(-1, sequences.shape[-1])

    if fit_scaler:
        sequences_2d = scaler.fit_transform(sequences_2d)
        logger.info("Fitted scaler on training data")
    else:
        sequences_2d = scaler.transform(sequences_2d)

    sequences = sequences_2d.reshape(original_shape).astype(np.float32)

    # Create dataset
    dataset = SpikeDatasetWithSoftLabels(
        sequences=sequences,
        hard_labels=hard_labels,
        soft_labels=soft_labels,
        timestamps=timestamps,
        symbols=symbols,
        feature_names=feature_cols
    )

    return dataset, scaler


def create_dataloader(
    dataset: SpikeDatasetWithSoftLabels,
    batch_size: int,
    shuffle: bool = False,
    weighted_sampling: bool = False
) -> DataLoader:
    """Create DataLoader with optional weighted sampling."""
    if weighted_sampling:
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_hard = 0.0
    total_soft = 0.0
    correct = 0
    total = 0

    for batch_idx, (sequences, hard_labels, soft_labels) in enumerate(train_loader):
        sequences = sequences.to(device)
        hard_labels = hard_labels.to(device)
        soft_labels = soft_labels.to(device)

        optimizer.zero_grad()

        outputs = model(sequences).squeeze()
        loss, hard_loss, soft_loss = criterion(outputs, hard_labels, soft_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_hard += hard_loss.item()
        total_soft += soft_loss.item()

        preds = (outputs > 0.5).float()
        correct += (preds == hard_labels).sum().item()
        total += hard_labels.size(0)

    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'hard_loss': total_hard / n_batches,
        'soft_loss': total_soft / n_batches,
        'accuracy': correct / total
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, hard_labels, soft_labels in val_loader:
            sequences = sequences.to(device)
            hard_labels = hard_labels.to(device)
            soft_labels = soft_labels.to(device)

            outputs = model(sequences).squeeze()
            loss, _, _ = criterion(outputs, hard_labels, soft_labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(hard_labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    preds_binary = (all_preds > 0.5).astype(float)
    tp = np.sum((preds_binary == 1) & (all_labels == 1))
    fp = np.sum((preds_binary == 1) & (all_labels == 0))
    fn = np.sum((preds_binary == 0) & (all_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'loss': total_loss / len(val_loader),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': np.mean(preds_binary == all_labels)
    }


def save_model(
    model: nn.Module,
    output_path: str,
    config: dict,
    history: dict,
    xgb_importances: List[float],
    scaler: RobustScaler
):
    """Save model with metadata."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'xgb_importances': xgb_importances,
        'scaler_state': {
            'center_': scaler.center_.tolist(),
            'scale_': scaler.scale_.tolist()
        },
        'feature_columns': FEATURE_COLUMNS,
        'saved_at': datetime.now().isoformat()
    }

    torch.save(checkpoint, output_path)
    logger.info(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost-Guided TCN')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--batch-size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=CONFIG['learning_rate'])
    parser.add_argument('--alpha', type=float, default=CONFIG['distillation_alpha'])
    parser.add_argument('--device', type=str, default=CONFIG['device'])
    args = parser.parse_args()

    # Update config with args
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size
    CONFIG['learning_rate'] = args.lr
    CONFIG['distillation_alpha'] = args.alpha
    CONFIG['device'] = args.device

    logger.info("=" * 80)
    logger.info("XGBoost-Guided TCN Training")
    logger.info("=" * 80)
    logger.info(f"Config: {json.dumps(CONFIG, indent=2, default=str)}")

    # Setup device
    device = setup_device(CONFIG['device'])

    # Load XGBoost model
    xgb_model, xgb_importances = load_xgboost_model(CONFIG['xgboost_model_path'])

    # Initialize labeler
    labeler = SlowLargeSpikeLabeler(
        prediction_offset=CONFIG['prediction_offset'],
        price_window=CONFIG['price_window'],
        min_price_spike=CONFIG['min_price_spike'],
        volume_window=CONFIG['volume_window'],
        min_volume_spike=CONFIG['min_volume_spike']
    )

    # Load training data
    logger.info("\n=== Loading Training Data ===")
    train_features = load_features_from_db(
        CONFIG['db_path'],
        CONFIG['train_start'],
        CONFIG['train_end'],
        FEATURE_COLUMNS
    )
    train_candles = load_candles_from_db(
        CONFIG['db_path'],
        list(train_features.keys()),
        CONFIG['train_start'],
        CONFIG['train_end']
    )
    train_dataset, scaler = build_dataset(
        train_features, train_candles, xgb_model, labeler, CONFIG,
        fit_scaler=True
    )

    # Load validation data
    logger.info("\n=== Loading Validation Data ===")
    val_features = load_features_from_db(
        CONFIG['db_path'],
        CONFIG['val_start'],
        CONFIG['val_end'],
        FEATURE_COLUMNS
    )
    val_candles = load_candles_from_db(
        CONFIG['db_path'],
        list(val_features.keys()),
        CONFIG['val_start'],
        CONFIG['val_end']
    )
    val_dataset, _ = build_dataset(
        val_features, val_candles, xgb_model, labeler, CONFIG,
        scaler=scaler
    )

    # Create data loaders
    train_loader = create_dataloader(train_dataset, CONFIG['batch_size'], weighted_sampling=True)
    val_loader = create_dataloader(val_dataset, CONFIG['batch_size'])

    logger.info(f"\nTrain: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")

    # Create model
    logger.info("\n=== Creating Model ===")
    model = XGBoostGuidedTCN(
        n_features=CONFIG['n_features'],
        sequence_length=CONFIG['sequence_length'],
        xgboost_importances=xgb_importances,
        num_channels=CONFIG['num_channels'],
        kernel_size=CONFIG['kernel_size'],
        dropout=CONFIG['dropout'],
        attention_temperature=CONFIG['attention_temperature']
    )
    model = model.to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    # Loss and optimizer
    criterion = DistillationLoss(
        alpha=CONFIG['distillation_alpha'],
        focal_alpha=CONFIG['focal_alpha'],
        focal_gamma=CONFIG['focal_gamma']
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    logger.info("\n=== Training ===")
    history = {
        'train_loss': [], 'train_hard': [], 'train_soft': [], 'train_acc': [],
        'val_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': [],
        'lr': []
    }

    best_f1 = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(CONFIG['epochs']):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_metrics['f1'])
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_hard'].append(train_metrics['hard_loss'])
        history['train_soft'].append(train_metrics['soft_loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['lr'].append(current_lr)

        epoch_time = time.time() - start_time

        logger.info(
            f"Epoch {epoch+1}/{CONFIG['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} (H:{train_metrics['hard_loss']:.4f} S:{train_metrics['soft_loss']:.4f}) | "
            f"Val F1: {val_metrics['f1']:.4f} P:{val_metrics['precision']:.4f} R:{val_metrics['recall']:.4f} | "
            f"LR: {current_lr:.1e} | Time: {epoch_time:.1f}s"
        )

        # Early stopping
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"  ↑ New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"\nRestored best model (F1: {best_f1:.4f})")

    # Save model
    output_path = os.path.join(CONFIG['output_dir'], 'xgboost_guided_tcn_v1.pt')
    save_model(model, output_path, CONFIG, history, xgb_importances, scaler)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)
    logger.info(f"Best Validation F1: {best_f1:.4f}")
    logger.info(f"Model saved to: {output_path}")

    # Show attention weights
    attention_weights = model.get_attention_weights().cpu().numpy()
    logger.info("\nTop 5 Attention Weights (Original Features):")
    for i, (feat, weight) in enumerate(zip(FEATURE_COLUMNS, attention_weights[:24])):
        if i < 5:
            logger.info(f"  {feat}: {weight:.4f}")

    logger.info("\nTop 5 Attention Weights (Delta Features):")
    for i, (feat, weight) in enumerate(zip(FEATURE_COLUMNS, attention_weights[24:])):
        if i < 5:
            logger.info(f"  {feat}_delta: {weight:.4f}")


if __name__ == '__main__':
    main()
