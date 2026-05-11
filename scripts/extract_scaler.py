#!/usr/bin/env python3
"""
Extract and save the RobustScaler from training data.

This recreates the scaler that was used during training so it can be
used for live inference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import argparse
from loguru import logger

from src.models.dataset import DatasetBuilder

def main():
    parser = argparse.ArgumentParser(description='Extract scaler from training data')
    parser.add_argument('--db', required=True, help='Path to database')
    parser.add_argument('--output', required=True, help='Path to save scaler.pkl')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length')
    parser.add_argument('--n-symbols', type=int, default=10, help='Number of symbols to sample')

    args = parser.parse_args()

    logger.info("Creating dataset builder to fit scaler...")
    logger.info(f"Database: {args.db}")
    logger.info(f"Sequence length: {args.sequence_length}")
    logger.info(f"Sampling {args.n_symbols} symbols")

    # Create dataset builder - it will fit the scaler automatically
    dataset_builder = DatasetBuilder(
        db_path=args.db,
        sequence_length=args.sequence_length,
        forward_window=3,
        scaler_type='robust'
    )

    # Load training data to fit scaler
    logger.info("Loading sample data to fit scaler...")
    logger.info("This will take a few minutes...")

    # Build dataset with fit_scaler=True
    dataset = dataset_builder.build_dataset_multi_symbol(
        symbols=None,  # Use all symbols
        timeframe='1m',
        start_date=None,
        end_date=None,
        feature_columns=None,
        min_candles=0,
        max_symbols=20,  # Just sample 20 symbols
        skip_symbols=0,
        normalize=True,  # Enable normalization
        fit_scaler=True  # This fits the scaler
    )

    if dataset is None or len(dataset) == 0:
        logger.error("Failed to build dataset!")
        return

    # Check if scaler was fitted
    if not dataset_builder.scaler_fitted:
        logger.error("Scaler was not fitted!")
        return

    logger.info(f"Dataset built: {len(dataset)} samples")

    # Save scaler
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(dataset_builder.scaler, f)

    logger.info(f"Scaler saved to: {output_path}")
    logger.info(f"Scaler type: {type(dataset_builder.scaler).__name__}")

    # Print scaler stats
    if hasattr(dataset_builder.scaler, 'center_'):
        logger.info(f"Scaler centers shape: {dataset_builder.scaler.center_.shape}")
        logger.info(f"Scaler scales shape: {dataset_builder.scaler.scale_.shape}")

if __name__ == "__main__":
    main()
