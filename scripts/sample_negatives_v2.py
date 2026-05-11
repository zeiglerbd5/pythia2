"""
Sample Negative Periods for Event Classifier - V2

Key insight: The original negatives were 59% "false starts" which look like breakouts.
This made the model good at avoiding failed breakouts, but bad at rejecting
normal boring market activity.

V2 Strategy - Three types of negatives:
1. Random samples (NEW): Truly random timestamps from random coins
2. Quiet periods: Low volatility, baseline volume
3. False starts: High volatility + volume, but no spike follows

Target ratio: 60% random, 25% quiet, 15% false starts

Usage:
    python scripts/sample_negatives_v2.py
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
EVENTS_PATH = '/Users/bz/Pythia2/data/spike_events.parquet'
DB_PATH = '/Users/bz/Pythia2/data/pythia.duckdb'
OUTPUT_PATH = '/Users/bz/Pythia2/data/negative_samples_v2.parquet'

TARGET_RANDOM_SAMPLES = 700
TARGET_QUIET_SAMPLES = 300
TARGET_FALSE_START_SAMPLES = 200

MIN_GAP_FROM_SPIKE_HOURS = 12
MIN_CANDLES_FOR_FEATURES = 100
RANDOM_SEED = 42


def get_all_symbols(conn) -> list:
    """Get all symbols with sufficient data."""
    result = conn.execute("""
        SELECT symbol, COUNT(*) as cnt
        FROM ohlcv
        WHERE timeframe = '1m'
        GROUP BY symbol
        HAVING COUNT(*) > 1000
    """).fetchall()
    return [r[0] for r in result]


def get_spike_times(events_df) -> dict:
    """Create lookup of spike times per symbol."""
    spike_times = {}
    for _, row in events_df.iterrows():
        symbol = row['symbol']
        if symbol not in spike_times:
            spike_times[symbol] = []
        spike_times[symbol].append(row['entry_start'])
        if pd.notna(row.get('peak_time')):
            spike_times[symbol].append(row['peak_time'])
    return spike_times


def is_near_spike(timestamp, spike_times: list, hours: int = 12) -> bool:
    """Check if timestamp is within N hours of any spike."""
    for spike_time in spike_times:
        if abs((timestamp - spike_time).total_seconds()) < hours * 3600:
            return True
    return False


def sample_random_negatives(conn, all_symbols: list, spike_times_by_symbol: dict,
                            target_count: int) -> list:
    """Sample truly random timestamps from random coins."""
    logger.info(f"Sampling {target_count} random negatives...")

    samples = []
    attempts = 0
    max_attempts = target_count * 20

    while len(samples) < target_count and attempts < max_attempts:
        attempts += 1
        symbol = random.choice(all_symbols)

        try:
            time_range = conn.execute(f"""
                SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts, COUNT(*) as cnt
                FROM ohlcv
                WHERE timeframe = '1m' AND symbol = '{symbol}'
            """).fetchone()

            if time_range[2] < MIN_CANDLES_FOR_FEATURES:
                continue

            min_ts, max_ts = time_range[0], time_range[1]
            buffer = timedelta(minutes=MIN_CANDLES_FOR_FEATURES)
            valid_start = min_ts + buffer
            valid_end = max_ts - timedelta(hours=24)

            if valid_start >= valid_end:
                continue

            total_seconds = (valid_end - valid_start).total_seconds()
            random_offset = random.random() * total_seconds
            random_ts = valid_start + timedelta(seconds=random_offset)
            random_ts = random_ts.replace(second=0, microsecond=0)

            symbol_spikes = spike_times_by_symbol.get(symbol, [])
            if is_near_spike(random_ts, symbol_spikes, MIN_GAP_FROM_SPIKE_HOURS):
                continue

            too_close = False
            for existing in samples:
                if existing['symbol'] == symbol:
                    if abs((random_ts - existing['timestamp']).total_seconds()) < 4 * 3600:
                        too_close = True
                        break

            if too_close:
                continue

            samples.append({
                'symbol': symbol,
                'timestamp': random_ts,
                'negative_type': 'random'
            })

            if len(samples) % 100 == 0:
                logger.info(f"  Collected {len(samples)}/{target_count} random samples")

        except Exception as e:
            continue

    logger.info(f"  Final: {len(samples)} random samples")
    return samples


def sample_quiet_periods(conn, all_symbols: list, spike_times_by_symbol: dict,
                         target_count: int) -> list:
    """Sample quiet periods - low volatility, normal volume."""
    logger.info(f"Sampling {target_count} quiet periods...")

    samples = []
    symbols_to_try = random.sample(all_symbols, min(len(all_symbols), 150))
    samples_per_symbol = max(1, target_count // len(symbols_to_try) + 1)

    for symbol in symbols_to_try:
        if len(samples) >= target_count:
            break

        try:
            df = conn.execute(f"""
                WITH base AS (
                    SELECT timestamp, close, volume,
                           AVG(volume) OVER (ORDER BY timestamp ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) as vol_ma60,
                           STDDEV(close/LAG(close) OVER (ORDER BY timestamp) - 1)
                               OVER (ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as volatility_20
                    FROM ohlcv
                    WHERE timeframe = '1m' AND symbol = '{symbol}'
                )
                SELECT timestamp, volatility_20,
                       volume / NULLIF(vol_ma60, 0) as vol_ratio
                FROM base
                WHERE volatility_20 IS NOT NULL
                ORDER BY timestamp
            """).fetchdf()

            if len(df) < MIN_CANDLES_FOR_FEATURES:
                continue

            vol_median = df['volatility_20'].median()
            quiet_mask = (
                (df['volatility_20'] < vol_median * 0.8) &
                (df['vol_ratio'] > 0.5) & (df['vol_ratio'] < 1.5)
            )

            quiet_candidates = df[quiet_mask]['timestamp'].tolist()
            if not quiet_candidates:
                continue

            random.shuffle(quiet_candidates)
            symbol_spikes = spike_times_by_symbol.get(symbol, [])
            symbol_samples = 0

            for ts in quiet_candidates:
                if symbol_samples >= samples_per_symbol:
                    break

                if is_near_spike(ts, symbol_spikes, MIN_GAP_FROM_SPIKE_HOURS):
                    continue

                too_close = False
                for existing in samples:
                    if existing['symbol'] == symbol:
                        if abs((ts - existing['timestamp']).total_seconds()) < 4 * 3600:
                            too_close = True
                            break

                if not too_close:
                    samples.append({
                        'symbol': symbol,
                        'timestamp': ts,
                        'negative_type': 'quiet'
                    })
                    symbol_samples += 1

        except Exception as e:
            continue

    logger.info(f"  Final: {len(samples)} quiet samples")
    return samples


def sample_false_starts(conn, symbols_with_spikes: list, spike_times_by_symbol: dict,
                        target_count: int) -> list:
    """Sample false starts - elevated activity but no spike follows."""
    logger.info(f"Sampling {target_count} false starts...")

    samples = []
    samples_per_symbol = max(1, target_count // len(symbols_with_spikes) + 1)

    for symbol in symbols_with_spikes:
        if len(samples) >= target_count:
            break

        try:
            df = conn.execute(f"""
                WITH base AS (
                    SELECT timestamp, high, low, close, volume,
                           AVG(volume) OVER (ORDER BY timestamp ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) as vol_ma60,
                           MAX(high) OVER (ORDER BY timestamp ROWS BETWEEN 1439 PRECEDING AND CURRENT ROW) as high_24h,
                           MIN(low) OVER (ORDER BY timestamp ROWS BETWEEN 1439 PRECEDING AND CURRENT ROW) as low_24h
                    FROM ohlcv
                    WHERE timeframe = '1m' AND symbol = '{symbol}'
                )
                SELECT timestamp, close,
                       volume / NULLIF(vol_ma60, 0) as vol_ratio,
                       (close - low_24h) / NULLIF(high_24h - low_24h, 0) as price_position
                FROM base
                WHERE vol_ma60 IS NOT NULL
                ORDER BY timestamp
            """).fetchdf()

            if len(df) < MIN_CANDLES_FOR_FEATURES:
                continue

            false_start_mask = (
                (df['vol_ratio'] > 2.0) &
                (df['price_position'] > 0.7)
            )

            candidates = df[false_start_mask]['timestamp'].tolist()
            if not candidates:
                continue

            random.shuffle(candidates)
            symbol_spikes = spike_times_by_symbol.get(symbol, [])
            symbol_samples = 0

            for ts in candidates:
                if symbol_samples >= samples_per_symbol:
                    break

                if is_near_spike(ts, symbol_spikes, MIN_GAP_FROM_SPIKE_HOURS):
                    continue

                too_close = False
                for existing in samples:
                    if existing['symbol'] == symbol:
                        if abs((ts - existing['timestamp']).total_seconds()) < 4 * 3600:
                            too_close = True
                            break

                if not too_close:
                    samples.append({
                        'symbol': symbol,
                        'timestamp': ts,
                        'negative_type': 'false_start'
                    })
                    symbol_samples += 1

        except Exception as e:
            continue

    logger.info(f"  Final: {len(samples)} false start samples")
    return samples


def main():
    logger.info("=" * 60)
    logger.info("NEGATIVE SAMPLING V2 - More Random/Boring Samples")
    logger.info("=" * 60)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    logger.info(f"\nLoading spike events from {EVENTS_PATH}")
    events_df = pd.read_parquet(EVENTS_PATH)
    logger.info(f"Total spike events: {len(events_df)}")

    spike_times_by_symbol = get_spike_times(events_df)
    symbols_with_spikes = list(spike_times_by_symbol.keys())

    conn = duckdb.connect(DB_PATH, read_only=True)

    all_symbols = get_all_symbols(conn)
    logger.info(f"Total symbols with sufficient data: {len(all_symbols)}")
    logger.info(f"Symbols with spikes: {len(symbols_with_spikes)}")

    random_samples = sample_random_negatives(
        conn, all_symbols, spike_times_by_symbol, TARGET_RANDOM_SAMPLES
    )

    quiet_samples = sample_quiet_periods(
        conn, all_symbols, spike_times_by_symbol, TARGET_QUIET_SAMPLES
    )

    false_start_samples = sample_false_starts(
        conn, symbols_with_spikes, spike_times_by_symbol, TARGET_FALSE_START_SAMPLES
    )

    conn.close()

    all_negatives = random_samples + quiet_samples + false_start_samples
    negatives_df = pd.DataFrame(all_negatives)

    negatives_df = negatives_df.sort_values('timestamp').reset_index(drop=True)
    negatives_df['event_id'] = range(2000, 2000 + len(negatives_df))

    logger.info("")
    logger.info("=" * 60)
    logger.info("SAMPLING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total negative samples: {len(negatives_df)}")
    logger.info(f"\nBy type:")
    print(negatives_df['negative_type'].value_counts())

    logger.info(f"\nSymbols represented: {negatives_df['symbol'].nunique()}")
    logger.info(f"Time range: {negatives_df['timestamp'].min()} to {negatives_df['timestamp'].max()}")

    negatives_df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"\nSaved to: {OUTPUT_PATH}")

    try:
        v1 = pd.read_parquet('/Users/bz/Pythia2/data/negative_samples.parquet')
        logger.info(f"\n=== COMPARISON WITH V1 ===")
        logger.info(f"V1 total: {len(v1)}")
        logger.info(f"V2 total: {len(negatives_df)} (+{len(negatives_df) - len(v1)})")
        logger.info(f"\nV1 breakdown:")
        print(v1['negative_type'].value_counts())
        logger.info(f"\nV2 breakdown:")
        print(negatives_df['negative_type'].value_counts())
    except:
        pass

    return negatives_df


if __name__ == "__main__":
    main()
