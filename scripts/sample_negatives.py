"""
Sample Negative Periods for Event Classifier

Two types of negatives:
1. Quiet periods: Low volatility, baseline volume, no spike within ±24hr
2. False starts: High volatility + volume expansion, but NO spike follows

This prevents the model from just learning "excitement = good" and forces it
to distinguish "excitement that leads to continuation" from "failed breakouts."

Usage:
    python scripts/sample_negatives.py

Output:
    data/negative_samples.parquet - Negative sample timestamps
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import random


# Configuration
LABELS_PATH = '/Users/bz/Pythia2/data/big_mover_labels.parquet'
EVENTS_PATH = '/Users/bz/Pythia2/data/spike_events.parquet'
DB_PATH = '/Users/bz/Pythia2/data/pythia.duckdb'
OUTPUT_PATH = '/Users/bz/Pythia2/data/negative_samples.parquet'

# Sampling parameters
TARGET_QUIET_SAMPLES = 250
TARGET_FALSE_START_SAMPLES = 250
MIN_GAP_FROM_SPIKE_HOURS = 24  # Must be at least 24hr from any spike
MIN_CANDLES_FOR_FEATURES = 100  # Need enough history for indicators

# Random seed for reproducibility
RANDOM_SEED = 42


def compute_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility and volume indicators for regime detection."""
    df = df.copy()

    # Returns and volatility
    df['returns'] = df['close'].pct_change()
    df['vol_20'] = df['returns'].rolling(20).std()
    df['vol_60'] = df['returns'].rolling(60).std()
    df['vol_ratio'] = df['vol_20'] / (df['vol_60'] + 1e-10)

    # Volume indicators
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ma_60'] = df['volume'].rolling(60).mean()
    df['volume_ratio'] = df['volume_ma_20'] / (df['volume_ma_60'] + 1e-10)

    # Price range compression
    df['range_20'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']

    return df


def find_quiet_periods(symbol: str, ohlcv_df: pd.DataFrame, spike_times: list, n_samples: int) -> list:
    """
    Find quiet periods for a symbol.

    Quiet = low volatility, baseline volume, no spike within ±24hr
    """
    if len(ohlcv_df) < MIN_CANDLES_FOR_FEATURES:
        return []

    df = compute_regime_indicators(ohlcv_df)

    # Mark timestamps too close to spikes
    df['near_spike'] = False
    for spike_time in spike_times:
        # Mark 24hr before and after each spike
        mask = (df['timestamp'] >= spike_time - timedelta(hours=MIN_GAP_FROM_SPIKE_HOURS)) & \
               (df['timestamp'] <= spike_time + timedelta(hours=MIN_GAP_FROM_SPIKE_HOURS))
        df.loc[mask, 'near_spike'] = True

    # Find quiet candidates:
    # - Not near any spike
    # - Low volatility (vol_ratio < 1.0 = below average)
    # - Normal volume (volume_ratio between 0.7 and 1.3)
    # - Range compression (range_20 < median)

    range_median = df['range_20'].median()
    vol_median = df['vol_20'].median()

    quiet_mask = (
        ~df['near_spike'] &
        df['vol_ratio'].notna() &
        (df['vol_ratio'] < 1.0) &
        (df['volume_ratio'] > 0.7) &
        (df['volume_ratio'] < 1.3) &
        (df['range_20'] < range_median) &
        (df.index >= MIN_CANDLES_FOR_FEATURES)  # Need enough lookback
    )

    quiet_candidates = df[quiet_mask]['timestamp'].tolist()

    if not quiet_candidates:
        return []

    # Sample with minimum spacing (at least 4 hours apart)
    sampled = []
    candidates = sorted(quiet_candidates)
    random.shuffle(candidates)

    for ts in candidates:
        # Check if far enough from already sampled
        too_close = False
        for s in sampled:
            if abs((ts - s).total_seconds()) < 4 * 3600:  # 4 hours
                too_close = True
                break

        if not too_close:
            sampled.append(ts)

        if len(sampled) >= n_samples:
            break

    return sampled


def find_false_starts(symbol: str, ohlcv_df: pd.DataFrame, spike_times: list, labels_df: pd.DataFrame, n_samples: int) -> list:
    """
    Find false start periods for a symbol.

    False start = elevated volatility + volume expansion, but NO 20%+ spike follows

    These are failed breakouts / bull traps - the hardest negatives.
    """
    if len(ohlcv_df) < MIN_CANDLES_FOR_FEATURES:
        return []

    df = compute_regime_indicators(ohlcv_df)

    # Get negative labels for this symbol (where no 20%+ spike follows)
    sym_labels = labels_df[labels_df['symbol'] == symbol]
    negative_times = set(sym_labels[sym_labels['label'] == 0]['timestamp'].tolist())

    # Mark timestamps too close to actual spikes
    df['near_spike'] = False
    for spike_time in spike_times:
        mask = (df['timestamp'] >= spike_time - timedelta(hours=MIN_GAP_FROM_SPIKE_HOURS)) & \
               (df['timestamp'] <= spike_time + timedelta(hours=MIN_GAP_FROM_SPIKE_HOURS))
        df.loc[mask, 'near_spike'] = True

    # Find false start candidates:
    # - Not near any spike
    # - Has negative label (no 20%+ spike follows)
    # - Elevated volatility (vol_ratio > 1.2)
    # - Volume expansion (volume_ratio > 1.3)

    false_start_mask = (
        ~df['near_spike'] &
        df['vol_ratio'].notna() &
        (df['vol_ratio'] > 1.2) &  # Above-average short-term vol
        (df['volume_ratio'] > 1.3) &  # Volume expanding
        (df['timestamp'].isin(negative_times)) &
        (df.index >= MIN_CANDLES_FOR_FEATURES)
    )

    false_start_candidates = df[false_start_mask]['timestamp'].tolist()

    if not false_start_candidates:
        return []

    # Sample with minimum spacing
    sampled = []
    candidates = sorted(false_start_candidates)
    random.shuffle(candidates)

    for ts in candidates:
        too_close = False
        for s in sampled:
            if abs((ts - s).total_seconds()) < 4 * 3600:
                too_close = True
                break

        if not too_close:
            sampled.append(ts)

        if len(sampled) >= n_samples:
            break

    return sampled


def main():
    logger.info("=" * 60)
    logger.info("SAMPLING NEGATIVE PERIODS")
    logger.info("=" * 60)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load spike events
    logger.info(f"Loading spike events from {EVENTS_PATH}")
    events_df = pd.read_parquet(EVENTS_PATH)
    logger.info(f"Total spike events: {len(events_df)}")

    # Create lookup of spike times per symbol
    spike_times_by_symbol = {}
    for _, row in events_df.iterrows():
        symbol = row['symbol']
        if symbol not in spike_times_by_symbol:
            spike_times_by_symbol[symbol] = []
        # Include both entry_start and peak_time as "spike zone"
        spike_times_by_symbol[symbol].append(row['entry_start'])
        spike_times_by_symbol[symbol].append(row['peak_time'])

    # Load labels
    logger.info(f"Loading labels from {LABELS_PATH}")
    labels_df = pd.read_parquet(LABELS_PATH)

    # Get all symbols (including those without spikes for quiet period sampling)
    all_symbols = labels_df['symbol'].unique()
    symbols_with_spikes = set(events_df['symbol'].unique())

    logger.info(f"Total symbols: {len(all_symbols)}")
    logger.info(f"Symbols with spikes: {len(symbols_with_spikes)}")

    # Connect to DB
    conn = duckdb.connect(DB_PATH, read_only=True)

    # Sample from each symbol proportionally
    n_symbols_for_quiet = min(len(all_symbols), 100)  # Limit symbols to sample from
    n_symbols_for_false_start = len(symbols_with_spikes)

    samples_per_symbol_quiet = max(1, TARGET_QUIET_SAMPLES // n_symbols_for_quiet)
    samples_per_symbol_false_start = max(1, TARGET_FALSE_START_SAMPLES // n_symbols_for_false_start)

    all_quiet = []
    all_false_starts = []

    # Sample quiet periods from all symbols
    logger.info(f"\nSampling quiet periods (target: {TARGET_QUIET_SAMPLES})...")
    sampled_symbols = random.sample(list(all_symbols), n_symbols_for_quiet)

    for i, symbol in enumerate(sampled_symbols):
        try:
            # Load OHLCV
            ohlcv_query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = '{symbol}' AND timeframe = '1m'
                ORDER BY timestamp
            """
            ohlcv_df = conn.execute(ohlcv_query).fetchdf()

            spike_times = spike_times_by_symbol.get(symbol, [])
            quiet_samples = find_quiet_periods(symbol, ohlcv_df, spike_times, samples_per_symbol_quiet)

            for ts in quiet_samples:
                all_quiet.append({
                    'symbol': symbol,
                    'timestamp': ts,
                    'negative_type': 'quiet'
                })

            if (i + 1) % 25 == 0:
                logger.info(f"  Processed {i+1}/{n_symbols_for_quiet} symbols, {len(all_quiet)} quiet samples")

        except Exception as e:
            logger.error(f"Error processing {symbol} for quiet: {e}")

    # Sample false starts from symbols with spikes
    logger.info(f"\nSampling false starts (target: {TARGET_FALSE_START_SAMPLES})...")

    for i, symbol in enumerate(symbols_with_spikes):
        try:
            ohlcv_query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = '{symbol}' AND timeframe = '1m'
                ORDER BY timestamp
            """
            ohlcv_df = conn.execute(ohlcv_query).fetchdf()

            spike_times = spike_times_by_symbol.get(symbol, [])
            false_start_samples = find_false_starts(
                symbol, ohlcv_df, spike_times, labels_df, samples_per_symbol_false_start
            )

            for ts in false_start_samples:
                all_false_starts.append({
                    'symbol': symbol,
                    'timestamp': ts,
                    'negative_type': 'false_start'
                })

            if (i + 1) % 25 == 0:
                logger.info(f"  Processed {i+1}/{n_symbols_for_false_start} symbols, {len(all_false_starts)} false starts")

        except Exception as e:
            logger.error(f"Error processing {symbol} for false starts: {e}")

    conn.close()

    # Combine and assign IDs
    all_negatives = all_quiet + all_false_starts
    negatives_df = pd.DataFrame(all_negatives)

    # Sort by timestamp
    negatives_df = negatives_df.sort_values('timestamp').reset_index(drop=True)
    negatives_df['event_id'] = range(1000, 1000 + len(negatives_df))  # Start from 1000 to distinguish from positives

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("NEGATIVE SAMPLING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total negative samples: {len(negatives_df)}")
    logger.info(f"  Quiet periods: {len(all_quiet)}")
    logger.info(f"  False starts: {len(all_false_starts)}")
    logger.info(f"Symbols represented: {negatives_df['symbol'].nunique()}")

    # Time range
    logger.info(f"\nTime range:")
    logger.info(f"  First: {negatives_df['timestamp'].min()}")
    logger.info(f"  Last: {negatives_df['timestamp'].max()}")

    # Type distribution
    logger.info(f"\nType distribution:")
    print(negatives_df['negative_type'].value_counts())

    # Save
    negatives_df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"\nSaved to: {OUTPUT_PATH}")

    # Show sample
    logger.info(f"\nSample negatives:")
    print(negatives_df.head(10).to_string())

    return negatives_df


if __name__ == "__main__":
    main()
