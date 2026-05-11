#!/usr/bin/env python3
"""
Pre-compute Pre-Spike Targets for All Symbols

Runs target generation ONCE and saves to database.
After this, training loads targets instantly from DB instead of computing them.

One-time cost: ~30-60 minutes
Benefit: Training becomes instant forever

Usage:
    python scripts/precompute_targets.py --db "/path/to/database.db"

    # Run in background:
    nohup python scripts/precompute_targets.py --db "/path/to/database.db" > precompute_targets.log 2>&1 &
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import duckdb
from loguru import logger


def generate_targets_vectorized_fast(
    prices: np.ndarray,
    volumes: np.ndarray,
    price_window: int = 3,
    volume_window: int = 2,
    min_price_spike: float = 0.06,
    min_volume_spike: float = 5.0
) -> np.ndarray:
    """
    TRUE vectorized target generation using NumPy's rolling window tricks.

    This is 100x faster than the loop version.
    """
    n = len(prices)
    max_window = max(price_window, volume_window)

    # Initialize targets
    targets = np.zeros(n, dtype=np.float32)

    # Use stride tricks for efficient rolling max without loops
    from numpy.lib.stride_tricks import sliding_window_view

    # Create sliding windows for future values
    if n <= max_window:
        return targets

    # For volume: max of next 2 minutes (volume_window=2)
    volume_windows = sliding_window_view(volumes[1:], window_shape=volume_window)
    max_future_volumes = np.max(volume_windows, axis=1)

    # For price: max of next 3 minutes (price_window=3)
    price_windows = sliding_window_view(prices[1:], window_shape=price_window)
    max_future_prices = np.max(price_windows, axis=1)

    # Pad to original length (last few elements can't have full forward window)
    max_future_volumes_padded = np.pad(max_future_volumes, (0, n - len(max_future_volumes)), constant_values=0)
    max_future_prices_padded = np.pad(max_future_prices, (0, n - len(max_future_prices)), constant_values=0)

    # Vectorized calculations
    volume_ratios = max_future_volumes_padded / (volumes + 1e-10)
    price_returns = (max_future_prices_padded - prices) / (prices + 1e-10)

    # Vectorized conditions
    has_volume_explosion = volume_ratios >= min_volume_spike
    has_price_spike = price_returns >= min_price_spike
    volume_not_zero = volumes > 0

    # Combine conditions (set last max_window elements to 0 since they can't have full forward window)
    targets[:-max_window] = (
        has_volume_explosion[:-max_window] &
        has_price_spike[:-max_window] &
        volume_not_zero[:-max_window]
    ).astype(np.float32)

    return targets


def precompute_targets_for_symbol(
    db_path: str,
    symbol: str,
    timeframe: str = '1m'
) -> pd.DataFrame:
    """
    Generate targets for a single symbol.

    Returns:
        DataFrame with columns: timestamp, symbol, timeframe, target
    """
    conn = duckdb.connect(db_path, read_only=True)

    try:
        # Load prices and volumes from candles table
        query = f"""
            SELECT timestamp, close as price, volume
            FROM candles
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
        """

        df = conn.execute(query).fetchdf()

        if len(df) == 0:
            logger.warning(f"No data for {symbol}")
            return None

        # Generate targets
        prices = df['price'].values
        volumes = df['volume'].values

        targets = generate_targets_vectorized_fast(
            prices=prices,
            volumes=volumes,
            price_window=3,
            volume_window=2,
            min_price_spike=0.06,
            min_volume_spike=5.0
        )

        # Create result dataframe
        result = pd.DataFrame({
            'timestamp': df['timestamp'],
            'symbol': symbol,
            'timeframe': timeframe,
            'target': targets
        })

        n_positive = int(targets.sum())
        positive_pct = (n_positive / len(targets)) * 100

        logger.info(
            f"{symbol}: {len(targets):,} candles, "
            f"{n_positive} pre-spike patterns ({positive_pct:.2f}%)"
        )

        return result

    finally:
        conn.close()


def create_targets_table(db_path: str):
    """Create targets table if it doesn't exist."""
    conn = duckdb.connect(db_path, read_only=False)

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS targets (
                timestamp TEXT,
                symbol TEXT,
                timeframe TEXT,
                target REAL,
                created_at TEXT,
                PRIMARY KEY (timestamp, symbol, timeframe)
            )
        """)

        logger.info("✓ Targets table created/verified")

    finally:
        conn.close()


def save_targets_to_db(db_path: str, targets_df: pd.DataFrame):
    """Save targets to database."""
    conn = duckdb.connect(db_path, read_only=False)

    try:
        # Add created_at timestamp
        targets_df['created_at'] = datetime.now().isoformat()

        # Delete existing targets for this symbol (if any)
        symbol = targets_df['symbol'].iloc[0]
        timeframe = targets_df['timeframe'].iloc[0]
        conn.execute(f"""
            DELETE FROM targets
            WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
        """)

        # Insert new targets
        conn.execute("""
            INSERT INTO targets (timestamp, symbol, timeframe, target, created_at)
            SELECT * FROM targets_df
        """)

        logger.info(f"✓ Saved {len(targets_df):,} targets to database")

    finally:
        conn.close()


def main(db_path: str):
    """Main pre-computation pipeline."""
    logger.info("=" * 80)
    logger.info("PRE-COMPUTING TARGETS FOR ALL SYMBOLS")
    logger.info("=" * 80)
    logger.info(f"Database: {db_path}")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 80)
    logger.info("")

    # Create targets table
    create_targets_table(db_path)
    logger.info("")

    # Get all symbols
    conn = duckdb.connect(db_path, read_only=True)
    try:
        symbols_query = "SELECT DISTINCT symbol FROM candles ORDER BY symbol"
        symbols = conn.execute(symbols_query).fetchdf()['symbol'].tolist()
        logger.info(f"Found {len(symbols)} symbols")
        logger.info("")
    finally:
        conn.close()

    # Process each symbol
    total_targets = 0
    total_positives = 0

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")

        try:
            targets_df = precompute_targets_for_symbol(db_path, symbol, timeframe='1m')

            if targets_df is not None:
                # Save to database
                save_targets_to_db(db_path, targets_df)

                total_targets += len(targets_df)
                total_positives += int(targets_df['target'].sum())

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("PRE-COMPUTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Completed: {datetime.now()}")
    logger.info(f"Symbols processed: {len(symbols)}")
    logger.info(f"Total targets: {total_targets:,}")
    logger.info(f"Pre-spike patterns: {total_positives:,} ({total_positives/total_targets*100:.2f}%)")
    logger.info("")
    logger.info("✓ Training can now load targets instantly from database!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-compute pre-spike targets for all symbols')
    parser.add_argument('--db', required=True, help='Path to database')
    args = parser.parse_args()

    main(args.db)
