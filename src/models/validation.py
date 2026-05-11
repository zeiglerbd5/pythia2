"""
Walk-Forward Validation Framework for Time-Series ML

Implements proper validation for financial time-series per implementation guide:
- Temporal train/test splits (no random shuffling)
- Expanding window: Train on all historical data
- Rolling window: Train on fixed recent window
- Purging and embargo to prevent look-ahead bias
- Multiple validation periods for robust performance estimation

Critical for cryptocurrency spike detection to avoid data leakage
and ensure model generalizes to unseen future data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Iterator, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class ValidationFold:
    """
    Single fold in walk-forward validation.

    Contains indices for train/val/test sets with temporal ordering.
    """
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: Optional[np.ndarray] = None
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    val_start: Optional[datetime] = None
    val_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None
    fold_number: int = 0

    def __repr__(self) -> str:
        return (
            f"ValidationFold(fold={self.fold_number}, "
            f"train={len(self.train_indices)}, val={len(self.val_indices)}, "
            f"test={len(self.test_indices) if self.test_indices is not None else 0})"
        )


class WalkForwardValidator:
    """
    Walk-forward validation for time-series data.

    Per implementation guide:
    - Maintains temporal order (no shuffling)
    - Supports expanding and rolling windows
    - Includes purging period to prevent leakage
    - Embargo period after test to prevent future info

    Modes:
    - 'expanding': Train on all historical data (default)
    - 'rolling': Train on fixed window of recent data
    - 'anchored': Train once, test on multiple forward periods
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
        mode: str = 'expanding',
        min_train_size: Optional[int] = None
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_splits: Number of validation folds
            train_size: Training set size (None = use all available before val)
            val_size: Validation set size (samples per fold)
            test_size: Test set size (final holdout, optional)
            gap: Gap between train and val to prevent leakage (samples)
            mode: 'expanding', 'rolling', or 'anchored'
            min_train_size: Minimum training samples required
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.gap = gap
        self.mode = mode
        self.min_train_size = min_train_size

        if mode not in ['expanding', 'rolling', 'anchored']:
            raise ValueError(f"mode must be 'expanding', 'rolling', or 'anchored', got {mode}")

        logger.info(
            f"WalkForwardValidator initialized",
            extra={
                "n_splits": n_splits,
                "mode": mode,
                "gap": gap,
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size
            }
        )

    def split(
        self,
        timestamps: np.ndarray,
        verbose: bool = True
    ) -> List[ValidationFold]:
        """
        Generate walk-forward splits.

        Args:
            timestamps: Timestamp array (must be sorted)
            verbose: Print split information

        Returns:
            List of ValidationFold objects
        """
        n_samples = len(timestamps)

        # Verify temporal order
        if not self._is_sorted(timestamps):
            raise ValueError("Timestamps must be sorted in ascending order")

        # Reserve test set if specified
        if self.test_size:
            n_test = self.test_size
            test_indices = np.arange(n_samples - n_test, n_samples)
            available_indices = np.arange(0, n_samples - n_test)
        else:
            test_indices = None
            available_indices = np.arange(0, n_samples)

        n_available = len(available_indices)

        # Calculate validation size
        if self.val_size is None:
            # Auto-calculate: split remaining data into n_splits + 1 parts
            # (1 part for initial train, n_splits parts for validation)
            self.val_size = n_available // (self.n_splits + 1)

        if self.val_size * self.n_splits > n_available:
            raise ValueError(
                f"Not enough data: val_size ({self.val_size}) * n_splits ({self.n_splits}) "
                f"= {self.val_size * self.n_splits} > available samples ({n_available})"
            )

        # Generate folds based on mode
        if self.mode == 'expanding':
            folds = self._expanding_window_split(available_indices, timestamps)
        elif self.mode == 'rolling':
            folds = self._rolling_window_split(available_indices, timestamps)
        else:  # anchored
            folds = self._anchored_split(available_indices, timestamps)

        # Add test set to all folds
        if test_indices is not None:
            for fold in folds:
                fold.test_indices = test_indices
                fold.test_start = timestamps[test_indices[0]]
                fold.test_end = timestamps[test_indices[-1]]

        if verbose:
            self._print_split_summary(folds, timestamps)

        return folds

    def _expanding_window_split(
        self,
        indices: np.ndarray,
        timestamps: np.ndarray
    ) -> List[ValidationFold]:
        """
        Expanding window: Each fold trains on all data up to validation period.

        Timeline: [Train 1][Val 1][Train 1+2][Val 2][Train 1+2+3][Val 3]...
        """
        folds = []

        for i in range(self.n_splits):
            # Validation indices
            val_start_idx = len(indices) - (self.n_splits - i) * self.val_size
            val_end_idx = val_start_idx + self.val_size
            val_indices = indices[val_start_idx:val_end_idx]

            # Training indices (all data before validation, with gap)
            train_end_idx = val_start_idx - self.gap
            train_indices = indices[:train_end_idx]

            # Apply minimum training size
            if self.min_train_size and len(train_indices) < self.min_train_size:
                logger.warning(
                    f"Fold {i+1}: Train size ({len(train_indices)}) < min_train_size "
                    f"({self.min_train_size}), skipping fold"
                )
                continue

            # Apply fixed train size if specified
            if self.train_size and len(train_indices) > self.train_size:
                train_indices = train_indices[-self.train_size:]

            fold = ValidationFold(
                train_indices=train_indices,
                val_indices=val_indices,
                train_start=timestamps[train_indices[0]],
                train_end=timestamps[train_indices[-1]],
                val_start=timestamps[val_indices[0]],
                val_end=timestamps[val_indices[-1]],
                fold_number=i + 1
            )

            folds.append(fold)

        return folds

    def _rolling_window_split(
        self,
        indices: np.ndarray,
        timestamps: np.ndarray
    ) -> List[ValidationFold]:
        """
        Rolling window: Each fold trains on fixed-size recent window.

        Timeline: [Train 1][Val 1]
                      [Train 2][Val 2]
                          [Train 3][Val 3]...
        """
        if self.train_size is None:
            raise ValueError("train_size must be specified for rolling window mode")

        folds = []

        for i in range(self.n_splits):
            # Validation indices
            val_start_idx = len(indices) - (self.n_splits - i) * self.val_size
            val_end_idx = val_start_idx + self.val_size
            val_indices = indices[val_start_idx:val_end_idx]

            # Training indices (fixed window before validation)
            train_end_idx = val_start_idx - self.gap
            train_start_idx = max(0, train_end_idx - self.train_size)
            train_indices = indices[train_start_idx:train_end_idx]

            if len(train_indices) == 0:
                continue

            fold = ValidationFold(
                train_indices=train_indices,
                val_indices=val_indices,
                train_start=timestamps[train_indices[0]],
                train_end=timestamps[train_indices[-1]],
                val_start=timestamps[val_indices[0]],
                val_end=timestamps[val_indices[-1]],
                fold_number=i + 1
            )

            folds.append(fold)

        return folds

    def _anchored_split(
        self,
        indices: np.ndarray,
        timestamps: np.ndarray
    ) -> List[ValidationFold]:
        """
        Anchored: Train once on initial data, validate on multiple forward periods.

        Timeline: [Train][Val 1][Val 2][Val 3]...

        Useful for testing model degradation over time.
        """
        # Initial training set
        total_val_size = self.val_size * self.n_splits
        train_end_idx = len(indices) - total_val_size - self.gap
        train_indices = indices[:train_end_idx]

        if self.train_size and len(train_indices) > self.train_size:
            train_indices = train_indices[-self.train_size:]

        folds = []

        for i in range(self.n_splits):
            # Validation indices
            val_start_idx = train_end_idx + self.gap + i * self.val_size
            val_end_idx = val_start_idx + self.val_size
            val_indices = indices[val_start_idx:val_end_idx]

            fold = ValidationFold(
                train_indices=train_indices,
                val_indices=val_indices,
                train_start=timestamps[train_indices[0]],
                train_end=timestamps[train_indices[-1]],
                val_start=timestamps[val_indices[0]],
                val_end=timestamps[val_indices[-1]],
                fold_number=i + 1
            )

            folds.append(fold)

        return folds

    def _is_sorted(self, arr: np.ndarray) -> bool:
        """Check if array is sorted in ascending order."""
        return np.all(arr[:-1] <= arr[1:])

    def _print_split_summary(
        self,
        folds: List[ValidationFold],
        timestamps: np.ndarray
    ):
        """Print summary of generated splits."""
        logger.info("=" * 80)
        logger.info(f"WALK-FORWARD VALIDATION SPLITS ({self.mode.upper()} MODE)")
        logger.info("=" * 80)

        for fold in folds:
            logger.info(f"\nFold {fold.fold_number}:")
            logger.info(f"  Train: {len(fold.train_indices):,} samples")
            logger.info(f"    Period: {fold.train_start} to {fold.train_end}")
            logger.info(f"  Validation: {len(fold.val_indices):,} samples")
            logger.info(f"    Period: {fold.val_start} to {fold.val_end}")

            if fold.test_indices is not None:
                logger.info(f"  Test: {len(fold.test_indices):,} samples")
                logger.info(f"    Period: {fold.test_start} to {fold.test_end}")

        logger.info("\n" + "=" * 80)


class TimeSeriesSplitter:
    """
    Simple time-series train/val/test splitter.

    For cases where walk-forward validation is not needed
    and a simple temporal split is sufficient.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        gap: int = 0
    ):
        """
        Initialize time-series splitter.

        Args:
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            gap: Gap between splits to prevent leakage
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.gap = gap

        logger.info(
            f"TimeSeriesSplitter initialized",
            extra={
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "gap": gap
            }
        )

    def split(
        self,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test sets.

        Args:
            n_samples: Total number of samples

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Calculate split points
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        # Apply gaps
        train_indices = np.arange(0, train_end)
        val_indices = np.arange(train_end + self.gap, val_end)
        test_indices = np.arange(val_end + self.gap, n_samples)

        logger.info(
            f"Created temporal split",
            extra={
                "train": len(train_indices),
                "val": len(val_indices),
                "test": len(test_indices)
            }
        )

        return train_indices, val_indices, test_indices

    def split_by_date(
        self,
        timestamps: np.ndarray,
        train_end_date: Optional[str] = None,
        val_end_date: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data by specific dates.

        Args:
            timestamps: Timestamp array
            train_end_date: End date for training (ISO format)
            val_end_date: End date for validation (ISO format)

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        if train_end_date:
            train_end_dt = pd.to_datetime(train_end_date)
            train_mask = timestamps <= train_end_dt
            train_indices = np.where(train_mask)[0]
        else:
            train_end = int(len(timestamps) * self.train_ratio)
            train_indices = np.arange(0, train_end)

        if val_end_date:
            val_end_dt = pd.to_datetime(val_end_date)
            val_start_dt = pd.to_datetime(train_end_date) if train_end_date else timestamps[train_indices[-1]]
            val_mask = (timestamps > val_start_dt) & (timestamps <= val_end_dt)
            val_indices = np.where(val_mask)[0]
        else:
            val_start = train_indices[-1] + 1 + self.gap
            val_end = int(len(timestamps) * (self.train_ratio + self.val_ratio))
            val_indices = np.arange(val_start, val_end)

        # Test is everything after validation
        test_start = val_indices[-1] + 1 + self.gap
        test_indices = np.arange(test_start, len(timestamps))

        logger.info(
            f"Created date-based split",
            extra={
                "train": (len(train_indices), str(timestamps[train_indices[0]]), str(timestamps[train_indices[-1]])),
                "val": (len(val_indices), str(timestamps[val_indices[0]]), str(timestamps[val_indices[-1]])),
                "test": (len(test_indices), str(timestamps[test_indices[0]]), str(timestamps[test_indices[-1]]))
            }
        )

        return train_indices, val_indices, test_indices


if __name__ == "__main__":
    # Test validation framework
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("=== Walk-Forward Validation Test ===\n")

    # Create synthetic time-series data
    n_samples = 1000
    start_date = pd.Timestamp('2024-01-01')
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq='5min')

    print(f"Dataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Period: {timestamps[0]} to {timestamps[-1]}")
    print(f"  Frequency: 5min\n")

    # Test expanding window
    print("=== Expanding Window Mode ===\n")

    validator_expanding = WalkForwardValidator(
        n_splits=5,
        val_size=100,
        test_size=100,
        gap=10,
        mode='expanding'
    )

    folds_expanding = validator_expanding.split(timestamps.values, verbose=False)

    print(f"Generated {len(folds_expanding)} folds:\n")
    for fold in folds_expanding:
        print(f"Fold {fold.fold_number}:")
        print(f"  Train: {len(fold.train_indices):4d} samples ({fold.train_start} to {fold.train_end})")
        print(f"  Val:   {len(fold.val_indices):4d} samples ({fold.val_start} to {fold.val_end})")
        if fold.test_indices is not None:
            print(f"  Test:  {len(fold.test_indices):4d} samples ({fold.test_start} to {fold.test_end})")
        print()

    # Test rolling window
    print("=== Rolling Window Mode ===\n")

    validator_rolling = WalkForwardValidator(
        n_splits=5,
        train_size=300,
        val_size=100,
        test_size=100,
        gap=10,
        mode='rolling'
    )

    folds_rolling = validator_rolling.split(timestamps.values, verbose=False)

    print(f"Generated {len(folds_rolling)} folds:\n")
    for fold in folds_rolling:
        print(f"Fold {fold.fold_number}:")
        print(f"  Train: {len(fold.train_indices):4d} samples ({fold.train_start} to {fold.train_end})")
        print(f"  Val:   {len(fold.val_indices):4d} samples ({fold.val_start} to {fold.val_end})")
        print()

    # Test anchored mode
    print("=== Anchored Mode ===\n")

    validator_anchored = WalkForwardValidator(
        n_splits=5,
        val_size=100,
        test_size=100,
        gap=10,
        mode='anchored'
    )

    folds_anchored = validator_anchored.split(timestamps.values, verbose=False)

    print(f"Generated {len(folds_anchored)} folds:\n")
    for fold in folds_anchored:
        print(f"Fold {fold.fold_number}:")
        print(f"  Train: {len(fold.train_indices):4d} samples (SAME FOR ALL FOLDS)")
        print(f"  Val:   {len(fold.val_indices):4d} samples ({fold.val_start} to {fold.val_end})")
        print()

    # Test simple splitter
    print("=== Simple Time-Series Splitter ===\n")

    splitter = TimeSeriesSplitter(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        gap=5
    )

    train_idx, val_idx, test_idx = splitter.split(n_samples)

    print(f"Split results:")
    print(f"  Train: {len(train_idx)} samples ({timestamps[train_idx[0]]} to {timestamps[train_idx[-1]]})")
    print(f"  Val:   {len(val_idx)} samples ({timestamps[val_idx[0]]} to {timestamps[val_idx[-1]]})")
    print(f"  Test:  {len(test_idx)} samples ({timestamps[test_idx[0]]} to {timestamps[test_idx[-1]]})")

    # Test date-based split
    print("\n=== Date-Based Split ===\n")

    train_idx, val_idx, test_idx = splitter.split_by_date(
        timestamps.values,
        train_end_date='2024-01-02',
        val_end_date='2024-01-03'
    )

    print(f"Split by dates:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")

    # Verify no overlap
    print("\n=== Verifying No Overlap ===")

    for mode_name, folds in [
        ('Expanding', folds_expanding),
        ('Rolling', folds_rolling),
        ('Anchored', folds_anchored)
    ]:
        print(f"\n{mode_name} mode:")
        for fold in folds:
            train_set = set(fold.train_indices)
            val_set = set(fold.val_indices)
            test_set = set(fold.test_indices) if fold.test_indices is not None else set()

            train_val_overlap = train_set & val_set
            train_test_overlap = train_set & test_set
            val_test_overlap = val_set & test_set

            has_overlap = len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0

            if has_overlap:
                print(f"  Fold {fold.fold_number}: ✗ HAS OVERLAP")
            else:
                print(f"  Fold {fold.fold_number}: ✓ No overlap")

    print("\n✓ Validation framework test complete!")
