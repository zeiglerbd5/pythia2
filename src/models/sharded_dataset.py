"""
Sharded Dataset for Memory-Efficient Training

Streams data from parquet shards, never loading entire dataset into RAM.
Supports shuffling within and across shards, and weighted sampling for
class imbalance.

Usage:
    from src.models.sharded_dataset import ShardedDataset, create_sharded_dataloader

    dataset = ShardedDataset(
        shard_dir='data/shards',
        normalize=True
    )

    dataloader = create_sharded_dataloader(
        dataset,
        batch_size=32,
        num_workers=2
    )
"""

import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple
from loguru import logger


class ShardedDataset(IterableDataset):
    """
    Memory-efficient dataset that streams from parquet shards.

    Each shard contains sequences for a batch of symbols.
    During iteration:
    1. Shuffles shard order
    2. Loads one shard at a time
    3. Shuffles samples within shard
    4. Yields normalized (features, label) pairs
    5. Frees shard memory before loading next
    """

    def __init__(
        self,
        shard_dir: str,
        split: str = 'train',  # 'train', 'val', or 'test'
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        normalize: bool = True,
        oversample_positive: float = 1.0,  # Multiplier for positive samples
        seed: Optional[int] = None
    ):
        """
        Initialize sharded dataset.

        Args:
            shard_dir: Directory containing shard parquet files and metadata.json
            split: Which split to use ('train', 'val', 'test')
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            normalize: Whether to normalize features using stored stats
            oversample_positive: Factor to oversample positive class (1.0 = no oversampling)
            seed: Random seed for reproducibility
        """
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.normalize = normalize
        self.oversample_positive = oversample_positive
        self.seed = seed

        # Load metadata
        metadata_path = self.shard_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.seq_len = self.metadata['seq_len']
        self.n_features = len(self.metadata['feature_cols'])
        self.norm_mean = np.array(self.metadata['norm_mean'], dtype=np.float32)
        self.norm_std = np.array(self.metadata['norm_std'], dtype=np.float32)

        # Get shard paths
        all_shards = sorted(self.shard_dir.glob('shard_*.parquet'))
        n_shards = len(all_shards)

        # Split shards by index (temporal split)
        n_train = int(n_shards * train_ratio)
        n_val = int(n_shards * val_ratio)

        if split == 'train':
            self.shard_paths = all_shards[:n_train]
        elif split == 'val':
            self.shard_paths = all_shards[n_train:n_train + n_val]
        else:  # test
            self.shard_paths = all_shards[n_train + n_val:]

        # Count samples
        self.n_samples = 0
        self.n_positives = 0
        for shard in self.metadata['shards']:
            shard_path = Path(shard['path'])
            if shard_path in self.shard_paths:
                self.n_samples += shard['n_samples']
                self.n_positives += shard['n_positives']

        logger.info(
            f"ShardedDataset ({split}): {len(self.shard_paths)} shards, "
            f"{self.n_samples:,} samples, {self.n_positives:,} positives "
            f"({100*self.n_positives/max(self.n_samples,1):.2f}%)"
        )

    def __len__(self):
        """Approximate length (exact for non-oversampled)."""
        if self.oversample_positive > 1.0:
            extra = int(self.n_positives * (self.oversample_positive - 1))
            return self.n_samples + extra
        return self.n_samples

    def __iter__(self):
        """Iterate through shards, yielding normalized samples."""
        # Get worker info for distributed loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split shards among workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shard_paths = self.shard_paths[worker_id::num_workers]
        else:
            shard_paths = self.shard_paths

        # Set seed for this epoch
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Shuffle shard order (for training)
        if self.split == 'train':
            shard_paths = random.sample(list(shard_paths), len(shard_paths))

        for shard_path in shard_paths:
            # Load shard
            df = pd.read_parquet(shard_path)

            # Shuffle within shard (for training)
            if self.split == 'train':
                df = df.sample(frac=1).reset_index(drop=True)

            for idx in range(len(df)):
                row = df.iloc[idx]

                # Reconstruct features array
                features = np.array(row['features'], dtype=np.float32)
                features = features.reshape(self.seq_len, self.n_features)

                label = float(row['label'])

                # Normalize
                if self.normalize:
                    features = (features - self.norm_mean) / self.norm_std

                # Handle NaN/inf
                features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

                yield torch.FloatTensor(features), torch.FloatTensor([label])

                # Oversample positives (repeat positive samples)
                if self.split == 'train' and label == 1 and self.oversample_positive > 1.0:
                    # Yield additional copies based on oversample factor
                    extra_copies = int(self.oversample_positive - 1)
                    for _ in range(extra_copies):
                        yield torch.FloatTensor(features), torch.FloatTensor([label])

            # Free memory
            del df


class ShardedMapDataset(torch.utils.data.Dataset):
    """
    Alternative: Map-style dataset that loads shards on demand.

    Maintains an index mapping and loads the appropriate shard when
    a specific index is requested. Better for validation/test where
    you need consistent ordering.
    """

    def __init__(
        self,
        shard_dir: str,
        split: str = 'val',
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        normalize: bool = True
    ):
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.normalize = normalize

        # Load metadata
        with open(self.shard_dir / 'metadata.json') as f:
            self.metadata = json.load(f)

        self.seq_len = self.metadata['seq_len']
        self.n_features = len(self.metadata['feature_cols'])
        self.norm_mean = np.array(self.metadata['norm_mean'], dtype=np.float32)
        self.norm_std = np.array(self.metadata['norm_std'], dtype=np.float32)

        # Get shard paths for this split
        all_shards = sorted(self.shard_dir.glob('shard_*.parquet'))
        n_shards = len(all_shards)
        n_train = int(n_shards * train_ratio)
        n_val = int(n_shards * val_ratio)

        if split == 'train':
            self.shard_paths = all_shards[:n_train]
        elif split == 'val':
            self.shard_paths = all_shards[n_train:n_train + n_val]
        else:
            self.shard_paths = all_shards[n_train + n_val:]

        # Build index: (shard_idx, row_idx) for each sample
        self.index_map = []
        for shard_idx, shard_path in enumerate(self.shard_paths):
            df = pd.read_parquet(shard_path)
            for row_idx in range(len(df)):
                self.index_map.append((shard_idx, row_idx))
            del df

        # Cache for currently loaded shard
        self._cached_shard_idx = None
        self._cached_data = None

        logger.info(
            f"ShardedMapDataset ({split}): {len(self.shard_paths)} shards, "
            f"{len(self.index_map):,} samples"
        )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        shard_idx, row_idx = self.index_map[idx]

        # Load shard if not cached
        if self._cached_shard_idx != shard_idx:
            self._cached_data = pd.read_parquet(self.shard_paths[shard_idx])
            self._cached_shard_idx = shard_idx

        row = self._cached_data.iloc[row_idx]

        # Reconstruct features
        features = np.array(row['features'], dtype=np.float32)
        features = features.reshape(self.seq_len, self.n_features)
        label = float(row['label'])

        if self.normalize:
            features = (features - self.norm_mean) / self.norm_std

        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

        return torch.FloatTensor(features), torch.FloatTensor([label])


def create_sharded_dataloader(
    shard_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 2,
    oversample_positive: float = 1.0,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for sharded data.

    Args:
        shard_dir: Directory containing shards
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        oversample_positive: Factor to oversample positive class
        **kwargs: Additional arguments passed to ShardedDataset

    Returns:
        DataLoader configured for memory-efficient streaming
    """
    if split == 'train':
        dataset = ShardedDataset(
            shard_dir=shard_dir,
            split=split,
            oversample_positive=oversample_positive,
            **kwargs
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )
    else:
        # Use map-style for val/test (consistent ordering)
        dataset = ShardedMapDataset(
            shard_dir=shard_dir,
            split=split,
            **kwargs
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )


if __name__ == "__main__":
    # Test the dataset
    import sys

    shard_dir = '/Users/bz/Pythia2/data/shards'

    if not Path(shard_dir).exists():
        print(f"Shard directory not found: {shard_dir}")
        print("Run preprocess_shards.py first.")
        sys.exit(1)

    print("=== Testing ShardedDataset ===\n")

    # Test training dataset
    train_dataset = ShardedDataset(shard_dir, split='train')
    print(f"Train samples: {len(train_dataset):,}")

    # Test iteration
    print("\nFirst 5 samples:")
    for i, (features, label) in enumerate(train_dataset):
        print(f"  {i}: features {features.shape}, label {label.item():.0f}")
        if i >= 4:
            break

    # Test dataloader
    print("\n=== Testing DataLoader ===")
    train_loader = create_sharded_dataloader(shard_dir, split='train', batch_size=32)

    batch_features, batch_labels = next(iter(train_loader))
    print(f"Batch features: {batch_features.shape}")
    print(f"Batch labels: {batch_labels.shape}")
    print(f"Positive in batch: {batch_labels.sum().item():.0f}")

    print("\n=== Test Complete ===")
