"""
Generate Big Mover Labels for Transformer Training

Creates labels for 20%+ gain prediction:
- For each candle, check if price increases 20%+ within next 24 hours
- Uses forward-looking max price (no data leakage in temporal split)

Usage:
    python scripts/generate_big_mover_labels.py

Output:
    data/big_mover_labels.parquet - Labels with columns:
        symbol, timestamp, label, max_gain_24h, time_to_peak
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger


# Configuration
DB_PATH = '/Users/bz/Pythia2/data/pythia.duckdb'
OUTPUT_PATH = '/Users/bz/Pythia2/data/big_mover_labels.parquet'
MIN_GAIN_THRESHOLD = 0.20  # 20% gain
FORWARD_WINDOW_MINUTES = 1440  # 24 hours = 1440 minutes
MIN_CANDLES_PER_SYMBOL = 500  # Need enough history


def generate_labels_for_symbol(conn, symbol: str) -> pd.DataFrame:
    """
    Generate 20%+ gain labels for a single symbol.

    For each candle, we check if the price increases 20%+
    within the next 24 hours (forward-looking).

    Returns DataFrame with columns:
        symbol, timestamp, label, max_gain_24h, time_to_peak_minutes
    """
    # Load 1-minute OHLCV data
    query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = '{symbol}' AND timeframe = '1m'
        ORDER BY timestamp
    """
    df = conn.execute(query).fetchdf()

    if len(df) < MIN_CANDLES_PER_SYMBOL:
        return pd.DataFrame()

    # Calculate forward-looking metrics using vectorized operations
    # For efficiency, we use rolling max shifted backwards

    # Forward max high in 24hr window
    # rolling(1440) looks back, shift(-1440) makes it look forward
    df['future_max_high'] = df['high'].iloc[::-1].rolling(
        FORWARD_WINDOW_MINUTES, min_periods=1
    ).max().iloc[::-1].shift(-1)

    # Calculate max gain from current close to future max high
    df['max_gain_24h'] = (df['future_max_high'] - df['close']) / df['close']

    # Binary label: 1 if gain >= 20%
    df['label'] = (df['max_gain_24h'] >= MIN_GAIN_THRESHOLD).astype(int)

    # Calculate time to peak (for analysis)
    # Find index of max high within each 24hr window
    df['time_to_peak_minutes'] = np.nan
    for i in range(len(df) - FORWARD_WINDOW_MINUTES):
        if df['label'].iloc[i] == 1:
            future_slice = df['high'].iloc[i:i+FORWARD_WINDOW_MINUTES]
            peak_idx = future_slice.idxmax() - df.index[i]
            df.loc[df.index[i], 'time_to_peak_minutes'] = peak_idx

    # Add symbol column
    df['symbol'] = symbol

    # Select output columns
    result = df[['symbol', 'timestamp', 'label', 'max_gain_24h', 'time_to_peak_minutes']].copy()

    # Drop last 24 hours (no future data to label)
    result = result.iloc[:-FORWARD_WINDOW_MINUTES]

    return result


def generate_all_labels():
    """Generate labels for all symbols."""
    logger.info(f"Starting label generation with {MIN_GAIN_THRESHOLD*100:.0f}% threshold")

    conn = duckdb.connect(DB_PATH, read_only=True)

    # Get all symbols with enough data (1-minute candles)
    symbols_query = """
        SELECT symbol, COUNT(*) as cnt
        FROM ohlcv
        WHERE timeframe = '1m'
        GROUP BY symbol
        HAVING COUNT(*) >= ?
        ORDER BY COUNT(*) DESC
    """
    symbols_df = conn.execute(symbols_query, [MIN_CANDLES_PER_SYMBOL]).fetchdf()
    symbols = symbols_df['symbol'].tolist()

    logger.info(f"Found {len(symbols)} symbols with >= {MIN_CANDLES_PER_SYMBOL} candles")

    # Process each symbol
    all_labels = []
    total_positive = 0
    total_samples = 0

    for i, symbol in enumerate(symbols):
        try:
            labels_df = generate_labels_for_symbol(conn, symbol)

            if len(labels_df) > 0:
                n_positive = labels_df['label'].sum()
                total_positive += n_positive
                total_samples += len(labels_df)

                all_labels.append(labels_df)

                if (i + 1) % 50 == 0:
                    pct = 100 * total_positive / total_samples if total_samples > 0 else 0
                    logger.info(
                        f"Progress: {i+1}/{len(symbols)} symbols | "
                        f"{total_samples:,} samples | "
                        f"{total_positive:,} positive ({pct:.2f}%)"
                    )
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    conn.close()

    # Combine all labels
    if not all_labels:
        logger.error("No labels generated!")
        return

    combined = pd.concat(all_labels, ignore_index=True)

    # Summary statistics
    n_positive = combined['label'].sum()
    n_total = len(combined)
    pct_positive = 100 * n_positive / n_total

    logger.info(f"\n=== Label Generation Complete ===")
    logger.info(f"Total samples: {n_total:,}")
    logger.info(f"Positive (20%+ gain): {n_positive:,} ({pct_positive:.2f}%)")
    logger.info(f"Negative: {n_total - n_positive:,} ({100-pct_positive:.2f}%)")

    # Distribution of max gains for positive samples
    positive_gains = combined[combined['label'] == 1]['max_gain_24h']
    if len(positive_gains) > 0:
        logger.info(f"\nPositive sample gain distribution:")
        logger.info(f"  Min: {positive_gains.min()*100:.1f}%")
        logger.info(f"  Median: {positive_gains.median()*100:.1f}%")
        logger.info(f"  Mean: {positive_gains.mean()*100:.1f}%")
        logger.info(f"  Max: {positive_gains.max()*100:.1f}%")

    # Time to peak distribution
    time_to_peak = combined[combined['label'] == 1]['time_to_peak_minutes'].dropna()
    if len(time_to_peak) > 0:
        logger.info(f"\nTime to peak (minutes):")
        logger.info(f"  Min: {time_to_peak.min():.0f}")
        logger.info(f"  Median: {time_to_peak.median():.0f}")
        logger.info(f"  Mean: {time_to_peak.mean():.0f}")
        logger.info(f"  Max: {time_to_peak.max():.0f}")

    # Save to parquet
    combined.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"\nSaved labels to: {OUTPUT_PATH}")

    return combined


def validate_labels(labels_path: str = OUTPUT_PATH):
    """Load and validate generated labels."""
    df = pd.read_parquet(labels_path)

    print(f"\n=== Label Validation ===")
    print(f"Total rows: {len(df):,}")
    print(f"Symbols: {df['symbol'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nPositive rate: {100 * df['label'].mean():.2f}%")

    # Check for data quality issues
    print(f"\nNull checks:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate big mover labels")
    parser.add_argument('--validate-only', action='store_true',
                       help="Only validate existing labels")
    parser.add_argument('--threshold', type=float, default=0.20,
                       help="Gain threshold (default: 0.20 = 20%%)")
    args = parser.parse_args()

    if args.validate_only:
        validate_labels()
    else:
        MIN_GAIN_THRESHOLD = args.threshold
        generate_all_labels()
