#!/usr/bin/env python3
"""
High-Frequency Order Book Feature Extraction

Extracts 10-second resolution features from order book + trade data.
Memory-safe: processes one symbol-date at a time.

Usage:
    python scripts/extract_orderbook_features_hf.py \
        --db "market_data copy_86.db" \
        --symbols BTC-USD,ETH-USD,SOL-USD \
        --dates 2024-10-18,2024-10-19,2024-10-20 \
        --window 10 \
        --output data/orderbook_features_3day.npz
"""

import argparse
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import gc
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class HighFreqFeatureExtractor:
    """
    Extract high-frequency features from order book + trades.
    """

    def __init__(self, db_path: str, window_seconds: int = 10):
        self.db_path = db_path
        self.window_seconds = window_seconds
        self.conn = sqlite3.connect(db_path)
        print(f"Connected to database: {db_path}")
        print(f"Window size: {window_seconds} seconds")

    def extract_for_symbol_date(self, symbol: str, date: str):
        """
        Extract features for ONE symbol-date pair.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            date: Date string (e.g., '2024-10-18')

        Returns:
            numpy array: (num_windows, num_features)
        """
        print(f"\n  Querying order book for {symbol} on {date}...")

        # Query 1: Order book snapshots
        ob_query = """
            SELECT
                timestamp,
                bid_depth_10,
                ask_depth_10,
                spread_percentage,
                large_bid_orders,
                large_ask_orders,
                weighted_mid_price
            FROM order_book_features
            WHERE symbol = ?
              AND timestamp >= ?
              AND timestamp < ?
            ORDER BY timestamp
        """

        # Date range
        start_date = f"{date} 00:00:00"
        end_date = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')

        ob_df = pd.read_sql_query(ob_query, self.conn, params=[symbol, start_date, end_date])

        if len(ob_df) == 0:
            print(f"    WARNING: No order book data for {symbol} on {date}")
            return None

        print(f"    Loaded {len(ob_df):,} order book snapshots")

        # Query 2: Trades
        print(f"  Querying trades for {symbol} on {date}...")
        trades_query = """
            SELECT
                timestamp,
                price,
                size,
                side
            FROM trades
            WHERE symbol = ?
              AND timestamp >= ?
              AND timestamp < ?
            ORDER BY timestamp
        """

        trades_df = pd.read_sql_query(trades_query, self.conn, params=[symbol, start_date, end_date])

        if len(trades_df) == 0:
            print(f"    WARNING: No trade data for {symbol} on {date}")
            return None

        print(f"    Loaded {len(trades_df):,} trades")

        # Resample to time windows
        print(f"  Resampling to {self.window_seconds}-second windows...")
        ob_resampled = self._resample_orderbook(ob_df, f'{self.window_seconds}S')
        trades_resampled = self._resample_trades(trades_df, f'{self.window_seconds}S')

        # Merge on timestamp
        merged = pd.merge(ob_resampled, trades_resampled,
                         left_index=True, right_index=True, how='inner')

        if len(merged) == 0:
            print(f"    WARNING: No overlapping data after merge")
            return None

        print(f"    Merged to {len(merged):,} windows")

        # Calculate features
        print(f"  Calculating features...")
        features = self._calculate_features(merged)

        # Add symbol and timestamp columns for later analysis
        features['symbol'] = symbol
        features['timestamp'] = features.index

        print(f"    Calculated {len(features.columns)} features")

        return features

    def _resample_orderbook(self, df, window):
        """Resample order book snapshots to time windows."""
        # Handle mixed timestamp formats (with/without microseconds)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        df = df.set_index('timestamp')

        # Take last value in each window (most recent snapshot)
        resampled = df.resample(window).last()

        # Forward fill missing windows (use last known value)
        resampled = resampled.ffill()

        # Drop any remaining NaNs (start of data)
        resampled = resampled.dropna()

        return resampled

    def _resample_trades(self, df, window):
        """Aggregate trades into time windows."""
        # Handle mixed timestamp formats (with/without microseconds)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        df = df.set_index('timestamp')

        # Separate buy/sell volumes
        df['buy_volume'] = df.apply(lambda x: x['size'] if x['side'] == 'BUY' else 0, axis=1)
        df['sell_volume'] = df.apply(lambda x: x['size'] if x['side'] == 'SELL' else 0, axis=1)

        # Aggregate per window
        agg_dict = {
            'price': ['first', 'last', 'min', 'max'],
            'size': 'sum',
            'buy_volume': 'sum',
            'sell_volume': 'sum',
            'side': 'count'  # Trade count
        }

        resampled = df.resample(window).agg(agg_dict)

        # Flatten column names
        resampled.columns = ['_'.join(col).strip() if col[1] else col[0]
                            for col in resampled.columns.values]

        # Fill missing windows with zeros (no trades)
        resampled = resampled.fillna(0)

        return resampled

    def _calculate_features(self, df):
        """Calculate all features from merged data."""
        features = pd.DataFrame(index=df.index)

        # === 1. ORDER BOOK IMBALANCE ===
        features['bid_ask_imbalance'] = (
            (df['bid_depth_10'] - df['ask_depth_10']) /
            (df['bid_depth_10'] + df['ask_depth_10'] + 1e-9)
        )

        # === 2. DEPTH RATE OF CHANGE ===
        # 5-window lookback = 50 seconds
        features['bid_depth_roc'] = df['bid_depth_10'].pct_change(5)
        features['ask_depth_roc'] = df['ask_depth_10'].pct_change(5)

        # Absolute depth (raw values)
        features['bid_depth'] = df['bid_depth_10']
        features['ask_depth'] = df['ask_depth_10']

        # === 3. SPREAD DYNAMICS ===
        features['spread_pct'] = df['spread_percentage']
        features['spread_roc'] = df['spread_percentage'].pct_change(3)  # 30-sec window

        # === 4. LARGE ORDER IMBALANCE ===
        features['large_order_imbalance'] = (
            (df['large_bid_orders'] - df['large_ask_orders']) /
            (df['large_bid_orders'] + df['large_ask_orders'] + 1)
        )
        features['large_bid_count'] = df['large_bid_orders']
        features['large_ask_count'] = df['large_ask_orders']

        # === 5. ORDER FLOW IMBALANCE (from trades) ===
        features['ofi'] = (
            (df['buy_volume_sum'] - df['sell_volume_sum']) /
            (df['buy_volume_sum'] + df['sell_volume_sum'] + 1e-9)
        )
        features['buy_volume'] = df['buy_volume_sum']
        features['sell_volume'] = df['sell_volume_sum']

        # === 6. TRADE VELOCITY ===
        features['trade_count'] = df['side_count']
        features['trade_velocity_roc'] = features['trade_count'].pct_change(3)

        # === 7. PRICE MOMENTUM ===
        # Use 'price_last' (last trade price in window)
        features['price'] = df['price_last']
        features['price_change_10sec'] = df['price_last'].pct_change(1) * 100  # 10-sec
        features['price_change_30sec'] = df['price_last'].pct_change(3) * 100  # 30-sec
        features['price_change_60sec'] = df['price_last'].pct_change(6) * 100  # 60-sec

        # === 8. PRICE VOLATILITY ===
        # Rolling std over 60 seconds (6 windows)
        features['price_volatility_60sec'] = (
            df['price_last'].pct_change().rolling(6).std() * 100
        )

        # === 9. CUMULATIVE IMBALANCE ===
        # Sum of imbalance over 30 seconds (3 windows)
        features['cumulative_imbalance_30sec'] = (
            features['bid_ask_imbalance'].rolling(3).sum()
        )

        # === 10. IMBALANCE VELOCITY ===
        # Rate of change of imbalance (per second)
        features['imbalance_velocity'] = (
            features['bid_ask_imbalance'].diff() / self.window_seconds
        )

        # === 11. WEIGHTED MID PRICE ===
        features['weighted_mid_price'] = df['weighted_mid_price']
        features['wmp_roc'] = df['weighted_mid_price'].pct_change(3) * 100

        # Fill NaNs from rolling calculations with 0
        features = features.fillna(0)

        # Replace inf values with 0
        features = features.replace([np.inf, -np.inf], 0)

        return features

    def close(self):
        """Close database connection."""
        self.conn.close()


def process_symbol_date_worker(args):
    """
    Worker function for multiprocessing.
    Each worker gets its own database connection.

    Args:
        args: Tuple of (db_path, symbol, date, window_seconds)

    Returns:
        DataFrame with features or None if failed
    """
    db_path, symbol, date, window_seconds = args

    try:
        # Each worker creates its own extractor (with own DB connection)
        extractor = HighFreqFeatureExtractor(db_path, window_seconds)
        features = extractor.extract_for_symbol_date(symbol, date)
        extractor.close()
        return features
    except Exception as e:
        print(f"ERROR processing {symbol} on {date}: {e}")
        return None


def build_sequences(features_list, sequence_length=60, step=6, lookahead=6, spike_threshold=0.05):
    """
    Build sequences and labels from extracted features.

    Args:
        features_list: List of DataFrames (one per symbol-date)
        sequence_length: Number of windows in sequence (60 = 10 min history)
        step: Step size for sliding window (6 = 60 sec apart, reduces overlap)
        lookahead: Windows to look ahead for spike (6 = 60 sec)
        spike_threshold: Min price change to be labeled as spike (0.05 = 5%)

    Returns:
        X: (num_sequences, sequence_length, num_features)
        y: (num_sequences,) binary labels
        metadata: List of (symbol, timestamp) for each sequence
    """
    all_sequences = []
    all_labels = []
    all_metadata = []

    # Get feature columns (excluding metadata)
    feature_cols = [c for c in features_list[0].columns if c not in ['symbol', 'timestamp']]
    num_features = len(feature_cols)

    print(f"\nBuilding sequences...")
    print(f"  Sequence length: {sequence_length} windows ({sequence_length * 10} seconds)")
    print(f"  Step size: {step} windows ({step * 10} seconds)")
    print(f"  Lookahead: {lookahead} windows ({lookahead * 10} seconds)")
    print(f"  Spike threshold: {spike_threshold * 100}%")
    print(f"  Features: {num_features}")

    for features_df in features_list:
        symbol = features_df['symbol'].iloc[0]

        # Extract feature matrix (drop metadata columns)
        features_array = features_df[feature_cols].values.astype(np.float32)
        timestamps = features_df['timestamp'].values

        # Sliding window
        num_sequences = 0
        for i in range(0, len(features_array) - sequence_length - lookahead, step):
            # Extract sequence
            sequence = features_array[i:i+sequence_length]

            # Calculate label (spike in next lookahead windows?)
            price_col_idx = feature_cols.index('price')
            price_now = features_array[i + sequence_length - 1, price_col_idx]

            # Look ahead
            future_prices = features_array[i+sequence_length:i+sequence_length+lookahead, price_col_idx]

            if len(future_prices) == 0 or price_now == 0:
                continue

            max_future_price = np.max(future_prices)
            price_change_pct = (max_future_price - price_now) / price_now

            spike = 1 if price_change_pct >= spike_threshold else 0

            all_sequences.append(sequence)
            all_labels.append(spike)
            all_metadata.append((symbol, timestamps[i + sequence_length - 1]))

            num_sequences += 1

        print(f"    {symbol}: {num_sequences:,} sequences")

    # Convert to numpy arrays
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)

    print(f"\nFinal dataset:")
    print(f"  Shape: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Positive samples: {y.sum()} ({y.mean()*100:.2f}%)")
    print(f"  Negative samples: {(1-y).sum()} ({(1-y.mean())*100:.2f}%)")
    print(f"  Memory: {X.nbytes / 1024**3:.2f} GB")

    return X, y, all_metadata


def main():
    parser = argparse.ArgumentParser(description='Extract high-frequency order book features')
    parser.add_argument('--db', required=True, help='Path to database file')
    parser.add_argument('--symbols', required=True, help='Comma-separated list of symbols')
    parser.add_argument('--dates', required=True, help='Comma-separated list of dates (YYYY-MM-DD)')
    parser.add_argument('--window', type=int, default=10, help='Window size in seconds (default: 10)')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length in windows (default: 60)')
    parser.add_argument('--step', type=int, default=6, help='Step size for sliding window (default: 6)')
    parser.add_argument('--lookahead', type=int, default=6, help='Lookahead windows for labeling (default: 6)')
    parser.add_argument('--spike-threshold', type=float, default=0.05, help='Spike threshold percentage (default: 0.05)')
    parser.add_argument('--output', required=True, help='Output .npz file path')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers (default: 10)')

    args = parser.parse_args()

    # Parse inputs
    symbols = [s.strip() for s in args.symbols.split(',')]
    dates = [d.strip() for d in args.dates.split(',')]

    print("=" * 80)
    print("HIGH-FREQUENCY ORDER BOOK FEATURE EXTRACTION")
    print("=" * 80)
    print(f"Database: {args.db}")
    print(f"Symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    print(f"Dates: {len(dates)} ({', '.join(dates)})")
    print(f"Total combinations: {len(symbols) * len(dates)}")
    print(f"Workers: {args.workers}")
    print("=" * 80)
    print(f"Connected to database: {args.db}")
    print(f"Window size: {args.window} seconds")
    print()

    # Build task list for parallel processing
    tasks = [(args.db, symbol, date, args.window) for symbol in symbols for date in dates]

    # Extract features in parallel
    print(f"Extracting features using {args.workers} parallel workers...")
    with Pool(args.workers) as pool:
        features_list = list(tqdm(
            pool.imap(process_symbol_date_worker, tasks),
            total=len(tasks),
            desc="Processing symbol-date pairs",
            unit="pair"
        ))

    # Filter out None results (failed extractions)
    features_list = [f for f in features_list if f is not None]
    print(f"\nSuccessfully extracted features for {len(features_list)} symbol-date pairs")

    if len(features_list) == 0:
        print("\nERROR: No features extracted. Check your data.")
        return

    # Build sequences
    X, y, metadata = build_sequences(
        features_list,
        sequence_length=args.sequence_length,
        step=args.step,
        lookahead=args.lookahead,
        spike_threshold=args.spike_threshold
    )

    # Save to disk
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        metadata=np.array(metadata, dtype=object),
        feature_names=np.array([c for c in features_list[0].columns if c not in ['symbol', 'timestamp']]),
        config={
            'symbols': symbols,
            'dates': dates,
            'window_seconds': args.window,
            'sequence_length': args.sequence_length,
            'step': args.step,
            'lookahead': args.lookahead,
            'spike_threshold': args.spike_threshold
        }
    )

    file_size_mb = output_path.stat().st_size / 1024**2
    print(f"Saved {file_size_mb:.2f} MB")
    print("\nDONE!")


if __name__ == '__main__':
    main()
