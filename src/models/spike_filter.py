"""
Spike Category Filter for Focusing on Specific Spike Types

This module provides utilities to filter training data to specific spike categories
(e.g., "Slow & Large" only) to improve model specialization.

Based on analysis showing models perform poorly when trying to predict both:
- Fast & Steep: <30 min peaks, 10-25% gains
- Slow & Large: 30min-24hr peaks, 15-135% gains

By focusing on one category, the model learns coherent patterns.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Set, Tuple
from pathlib import Path
from loguru import logger


class SpikeCategoryFilter:
    """
    Filter training targets to specific spike categories.

    Usage:
        # Load categorized spikes
        filter = SpikeCategoryFilter('all_spikes_categorized.csv')

        # During dataset building, filter targets
        filtered_targets = filter.filter_targets(
            targets, timestamps, symbols,
            categories=['Slow & Large']
        )
    """

    def __init__(self, categorized_spikes_path: str):
        """
        Initialize spike category filter.

        Args:
            categorized_spikes_path: Path to CSV with categorized spikes
                Expected columns: symbol, timestamp, category, peak_gain, time_to_peak_min
        """
        self.categorized_path = Path(categorized_spikes_path)

        if not self.categorized_path.exists():
            raise FileNotFoundError(f"Categorized spikes file not found: {categorized_spikes_path}")

        # Load categorized spikes
        self.categorized_df = pd.read_csv(self.categorized_path)
        self.categorized_df['timestamp'] = pd.to_datetime(self.categorized_df['timestamp'])

        # Get unique categories
        self.available_categories = set(self.categorized_df['category'].unique())

        # Build lookup structures for fast filtering
        self._build_lookups()

        logger.info(
            f"SpikeC ategoryFilter initialized",
            extra={
                "total_spikes": len(self.categorized_df),
                "categories": sorted(list(self.available_categories)),
                "category_counts": self.categorized_df['category'].value_counts().to_dict()
            }
        )

    def _build_lookups(self):
        """Build fast lookup structures for filtering."""
        # Category-specific lookups: {category: {(symbol, timestamp_str)}}
        self.category_lookups = {}

        for category in self.available_categories:
            category_df = self.categorized_df[self.categorized_df['category'] == category]

            # Create set of (symbol, timestamp_str) tuples for O(1) lookup
            spike_keys = set(
                zip(
                    category_df['symbol'].values,
                    category_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').values
                )
            )

            self.category_lookups[category] = spike_keys
            logger.debug(f"Category '{category}': {len(spike_keys)} spikes indexed")

    def filter_targets(
        self,
        targets: np.ndarray,
        timestamps: np.ndarray,
        symbols: np.ndarray,
        categories: List[str],
        exclude_unlabeled: bool = True
    ) -> np.ndarray:
        """
        Filter targets to only include specified spike categories.

        Args:
            targets: Binary target array (0/1)
            timestamps: Timestamp array (numpy datetime64 or string)
            symbols: Symbol array
            categories: List of categories to KEEP (e.g., ['Slow & Large'])
            exclude_unlabeled: If True, set non-categorized spikes to 0

        Returns:
            Filtered target array (same shape as input)
        """
        # Validate categories
        invalid_cats = set(categories) - self.available_categories
        if invalid_cats:
            raise ValueError(
                f"Invalid categories: {invalid_cats}. "
                f"Available: {self.available_categories}"
            )

        # Build combined lookup set for all requested categories
        combined_lookup = set()
        for category in categories:
            combined_lookup.update(self.category_lookups[category])

        # Create filtered copy
        filtered_targets = targets.copy()

        # Convert timestamps to string format for lookup
        if timestamps.dtype.kind == 'M':  # numpy datetime64
            timestamp_strs = pd.to_datetime(timestamps).strftime('%Y-%m-%d %H:%M:%S').values
        else:  # Already string-like
            timestamp_strs = pd.to_datetime(timestamps).strftime('%Y-%m-%d %H:%M:%S').values

        # Iterate through positive targets and check if they're in our category
        positive_indices = np.where(targets == 1)[0]

        n_kept = 0
        n_filtered = 0

        for idx in positive_indices:
            symbol = symbols[idx]
            timestamp_str = timestamp_strs[idx]
            key = (symbol, timestamp_str)

            if key not in combined_lookup:
                # This spike is not in our desired categories
                if exclude_unlabeled:
                    filtered_targets[idx] = 0
                    n_filtered += 1
            else:
                n_kept += 1

        logger.info(
            f"Filtered targets to categories: {categories}",
            extra={
                "original_positives": len(positive_indices),
                "kept": n_kept,
                "filtered_out": n_filtered,
                "new_positive_ratio": f"{(filtered_targets.sum() / len(filtered_targets)) * 100:.3f}%"
            }
        )

        return filtered_targets

    def get_category_stats(self, category: Optional[str] = None) -> Dict:
        """
        Get statistics for a category or all categories.

        Args:
            category: Specific category or None for all

        Returns:
            Dictionary with statistics
        """
        if category:
            df = self.categorized_df[self.categorized_df['category'] == category]
        else:
            df = self.categorized_df

        if len(df) == 0:
            return {}

        stats = {
            "n_spikes": len(df),
            "n_symbols": df['symbol'].nunique(),
            "peak_gain": {
                "mean": df['peak_gain'].mean(),
                "median": df['peak_gain'].median(),
                "min": df['peak_gain'].min(),
                "max": df['peak_gain'].max()
            },
            "time_to_peak_min": {
                "mean": df['time_to_peak_min'].mean(),
                "median": df['time_to_peak_min'].median(),
                "min": df['time_to_peak_min'].min(),
                "max": df['time_to_peak_min'].max()
            },
            "top_symbols": df['symbol'].value_counts().head(10).to_dict()
        }

        return stats

    def print_summary(self):
        """Print summary of all categories."""
        print("\n" + "="*80)
        print("SPIKE CATEGORY SUMMARY")
        print("="*80)

        for category in sorted(self.available_categories):
            stats = self.get_category_stats(category)
            print(f"\n{category}:")
            print(f"  Spikes: {stats['n_spikes']}")
            print(f"  Symbols: {stats['n_symbols']}")
            print(f"  Peak Gain: {stats['peak_gain']['median']:.1f}% median, "
                  f"{stats['peak_gain']['min']:.1f}%-{stats['peak_gain']['max']:.1f}% range")
            print(f"  Time to Peak: {stats['time_to_peak_min']['median']:.0f} min median, "
                  f"{stats['time_to_peak_min']['min']:.0f}-{stats['time_to_peak_min']['max']:.0f} min range")
            print(f"  Top Symbols: {', '.join([f'{s} ({n})' for s, n in list(stats['top_symbols'].items())[:5]])}")

        print("\n" + "="*80)


# Convenience function for quick filtering
def filter_to_slow_large(
    targets: np.ndarray,
    timestamps: np.ndarray,
    symbols: np.ndarray,
    categorized_csv_path: str = "all_spikes_categorized.csv"
) -> np.ndarray:
    """
    Quick filter to Slow & Large spikes only.

    Args:
        targets: Binary target array
        timestamps: Timestamp array
        symbols: Symbol array
        categorized_csv_path: Path to categorized spikes CSV

    Returns:
        Filtered target array with only Slow & Large spikes
    """
    filter_obj = SpikeCategoryFilter(categorized_csv_path)
    return filter_obj.filter_targets(
        targets, timestamps, symbols,
        categories=['Slow & Large'],
        exclude_unlabeled=True
    )
