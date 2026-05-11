"""
Identify Independent Spike Events from Labels

Groups overlapping positive labels into discrete spike events.
Defines event_start based on observable regime shift, not optimal hindsight entry.

Usage:
    python scripts/identify_spike_events.py

Output:
    data/spike_events.parquet - Independent spike events with columns:
        event_id, symbol, entry_start, peak_time, return_pct, event_duration_min
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
LABELS_PATH = '/Users/bz/Pythia2/data/big_mover_labels.parquet'
DB_PATH = '/Users/bz/Pythia2/data/pythia.duckdb'
OUTPUT_PATH = '/Users/bz/Pythia2/data/spike_events.parquet'

# Event grouping: gap > 12 hours = new event
EVENT_GAP_MINUTES = 720  # 12 hours

# Regime detection lookback
REGIME_LOOKBACK_MINUTES = 60  # 1 hour baseline for vol/volume comparison


def find_regime_shift_point(ohlcv_df: pd.DataFrame, peak_idx: int, first_positive_idx: int) -> int:
    """
    Walk backward from peak until regime breaks.

    Regime shift detected when ANY of:
    - Volatility drops below 20-period baseline
    - Volume drops below 20-period baseline
    - Price re-enters prior consolidation range

    Returns index of regime shift point (not first profitable minute).
    """
    if peak_idx <= first_positive_idx:
        return first_positive_idx

    # Compute rolling baselines
    df = ohlcv_df.copy()

    # Volatility: rolling std of returns
    df['returns'] = df['close'].pct_change()
    df['vol_20'] = df['returns'].rolling(20).std()
    df['vol_baseline'] = df['vol_20'].rolling(60).mean()  # Longer-term baseline

    # Volume baseline
    df['vol_ma_20'] = df['volume'].rolling(20).mean()
    df['vol_baseline_60'] = df['volume'].rolling(60).mean()

    # Price range (for consolidation detection)
    df['price_ma_60'] = df['close'].rolling(60).mean()
    df['price_std_60'] = df['close'].rolling(60).std()
    df['price_upper'] = df['price_ma_60'] + 1.5 * df['price_std_60']
    df['price_lower'] = df['price_ma_60'] - 1.5 * df['price_std_60']

    # Walk backward from peak
    regime_start = first_positive_idx

    for i in range(peak_idx - 1, max(first_positive_idx - 60, 60), -1):
        if i >= len(df):
            continue

        row = df.iloc[i]

        # Check if regime has "broken" (returned to baseline)
        vol_elevated = row['vol_20'] > row['vol_baseline'] * 1.2 if pd.notna(row['vol_baseline']) else True
        volume_elevated = row['volume'] > row['vol_baseline_60'] * 1.3 if pd.notna(row['vol_baseline_60']) else True
        price_in_range = row['close'] < row['price_upper'] if pd.notna(row['price_upper']) else True

        # If all indicators show "normal", this is before the regime shift
        if not vol_elevated and not volume_elevated and price_in_range:
            regime_start = i + 1  # The regime started AFTER this point
            break

        regime_start = i

    # Don't go earlier than first_positive - 60 (1 hour before first profitable entry)
    # This prevents anchoring too far back on gradual moves
    min_start = max(first_positive_idx - 60, 0)
    regime_start = max(regime_start, min_start)

    return regime_start


def identify_events_for_symbol(symbol: str, labels_df: pd.DataFrame, conn) -> list:
    """
    Identify independent spike events for a single symbol.

    Returns list of event dicts with:
        symbol, entry_start, peak_time, return_pct, first_positive_time
    """
    # Get positive labels for this symbol, sorted by time
    sym_labels = labels_df[
        (labels_df['symbol'] == symbol) & (labels_df['label'] == 1)
    ].sort_values('timestamp').copy()

    if len(sym_labels) == 0:
        return []

    # Load OHLCV data for this symbol
    ohlcv_query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = '{symbol}' AND timeframe = '1m'
        ORDER BY timestamp
    """
    ohlcv_df = conn.execute(ohlcv_query).fetchdf()

    if len(ohlcv_df) < 100:
        return []

    # Create timestamp index for fast lookup
    ohlcv_df['ts_idx'] = range(len(ohlcv_df))
    ts_to_idx = dict(zip(ohlcv_df['timestamp'], ohlcv_df['ts_idx']))

    # Group consecutive positives into events
    # Gap > EVENT_GAP_MINUTES = new event
    events = []
    current_event_labels = []

    prev_ts = None
    for _, row in sym_labels.iterrows():
        ts = row['timestamp']

        if prev_ts is not None:
            gap_minutes = (ts - prev_ts).total_seconds() / 60

            if gap_minutes > EVENT_GAP_MINUTES:
                # End current event, start new one
                if current_event_labels:
                    events.append(current_event_labels)
                current_event_labels = []

        current_event_labels.append(row)
        prev_ts = ts

    # Don't forget last event
    if current_event_labels:
        events.append(current_event_labels)

    # Process each event
    processed_events = []

    for event_labels in events:
        if not event_labels:
            continue

        # First positive timestamp
        first_positive_ts = event_labels[0]['timestamp']
        first_positive_idx = ts_to_idx.get(first_positive_ts)

        if first_positive_idx is None:
            continue

        # Find peak: max price within 24 hours of first positive
        # Using max_gain_24h from labels to find the actual peak
        max_gain_row = max(event_labels, key=lambda x: x['max_gain_24h'])
        peak_ts = max_gain_row['timestamp'] + timedelta(minutes=max_gain_row['time_to_peak_minutes']) \
                  if pd.notna(max_gain_row['time_to_peak_minutes']) else max_gain_row['timestamp']

        # Get peak info from OHLCV
        peak_idx = ts_to_idx.get(peak_ts)
        if peak_idx is None:
            # Find closest timestamp
            closest_idx = ohlcv_df['timestamp'].searchsorted(peak_ts)
            peak_idx = min(closest_idx, len(ohlcv_df) - 1)

        # Find regime shift point (entry_start)
        regime_start_idx = find_regime_shift_point(ohlcv_df, peak_idx, first_positive_idx)

        entry_start_ts = ohlcv_df.iloc[regime_start_idx]['timestamp']
        entry_price = ohlcv_df.iloc[regime_start_idx]['close']
        peak_price = ohlcv_df.iloc[min(peak_idx, len(ohlcv_df)-1)]['high']

        return_pct = (peak_price - entry_price) / entry_price if entry_price > 0 else 0

        # Event duration
        event_duration = len(event_labels)  # Number of positive labels

        processed_events.append({
            'symbol': symbol,
            'entry_start': entry_start_ts,
            'first_positive_time': first_positive_ts,
            'peak_time': peak_ts,
            'entry_price': entry_price,
            'peak_price': peak_price,
            'return_pct': return_pct,
            'event_duration_min': event_duration,
            'n_positive_labels': len(event_labels)
        })

    return processed_events


def main():
    logger.info("=" * 60)
    logger.info("IDENTIFYING INDEPENDENT SPIKE EVENTS")
    logger.info("=" * 60)

    # Load labels
    logger.info(f"Loading labels from {LABELS_PATH}")
    labels_df = pd.read_parquet(LABELS_PATH)

    n_positive = labels_df['label'].sum()
    n_symbols_with_pos = labels_df[labels_df['label'] == 1]['symbol'].nunique()

    logger.info(f"Total positive labels: {n_positive:,}")
    logger.info(f"Symbols with positives: {n_symbols_with_pos}")

    # Connect to DB for OHLCV data
    conn = duckdb.connect(DB_PATH, read_only=True)

    # Get symbols with positive labels
    symbols_with_positives = labels_df[labels_df['label'] == 1]['symbol'].unique()

    # Process each symbol
    all_events = []

    for i, symbol in enumerate(symbols_with_positives):
        try:
            events = identify_events_for_symbol(symbol, labels_df, conn)
            all_events.extend(events)

            if (i + 1) % 20 == 0:
                logger.info(f"Processed {i+1}/{len(symbols_with_positives)} symbols, {len(all_events)} events found")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    conn.close()

    # Create DataFrame
    events_df = pd.DataFrame(all_events)
    events_df['event_id'] = range(len(events_df))

    # Sort by entry_start time
    events_df = events_df.sort_values('entry_start').reset_index(drop=True)
    events_df['event_id'] = range(len(events_df))  # Re-assign after sort

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPIKE EVENT IDENTIFICATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total independent events: {len(events_df)}")
    logger.info(f"From {n_positive:,} positive labels -> {len(events_df)} events")
    logger.info(f"Average labels per event: {n_positive / len(events_df):.1f}")
    logger.info(f"Symbols represented: {events_df['symbol'].nunique()}")

    # Return distribution
    logger.info(f"\nReturn distribution:")
    logger.info(f"  Min: {events_df['return_pct'].min()*100:.1f}%")
    logger.info(f"  Median: {events_df['return_pct'].median()*100:.1f}%")
    logger.info(f"  Mean: {events_df['return_pct'].mean()*100:.1f}%")
    logger.info(f"  Max: {events_df['return_pct'].max()*100:.1f}%")

    # Time range
    logger.info(f"\nTime range:")
    logger.info(f"  First event: {events_df['entry_start'].min()}")
    logger.info(f"  Last event: {events_df['entry_start'].max()}")

    # Events per symbol distribution
    events_per_sym = events_df.groupby('symbol').size()
    logger.info(f"\nEvents per symbol:")
    logger.info(f"  Min: {events_per_sym.min()}")
    logger.info(f"  Median: {events_per_sym.median():.0f}")
    logger.info(f"  Max: {events_per_sym.max()}")

    # Save
    events_df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"\nSaved to: {OUTPUT_PATH}")

    # Show sample
    logger.info(f"\nSample events:")
    print(events_df[['event_id', 'symbol', 'entry_start', 'return_pct', 'n_positive_labels']].head(10).to_string())

    return events_df


if __name__ == "__main__":
    main()
