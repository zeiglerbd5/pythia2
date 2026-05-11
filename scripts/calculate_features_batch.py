#!/usr/bin/env python3
"""
Batch Feature Calculator

Calculates ML features from existing candles and order book data.

Uses manual implementations from src/features/:
- microstructure.py: Roll measure, VPIN, order flow imbalance
- price_indicators.py: RSI, VWAP, ATR, Bollinger Bands
- volume_indicators.py: OBV, VROC, volume spikes

Writes results to 'features' table in DuckDB for model training.

Usage:
    python scripts/calculate_features_batch.py --db data/pythia.duckdb --timeframe 1m
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import duckdb
from loguru import logger

from src.features.microstructure import calculate_microstructure_features
from src.features.price_indicators import calculate_price_features
from src.features.volume_indicators import calculate_volume_features


class BatchFeatureCalculator:
    """
    Calculate ML features from existing database tables.

    Reads from:
    - candles (or ohlcv): OHLCV data
    - order_book_features: Microstructure data

    Writes to:
    - features: ML-ready features for training
    """

    def __init__(self, db_path: str, table_name: str = 'candles'):
        """
        Initialize calculator.

        Args:
            db_path: Path to DuckDB database
            table_name: Name of candles table (default: 'candles')
        """
        self.db_path = db_path
        self.table_name = table_name

        logger.info(f"BatchFeatureCalculator initialized: {db_path}")
        logger.info(f"Using table: {table_name}")

    def get_all_symbols(self) -> List[str]:
        """Get all symbols from database."""
        conn = duckdb.connect(self.db_path, read_only=True)

        try:
            query = f"SELECT DISTINCT symbol FROM {self.table_name} ORDER BY symbol"
            result = conn.execute(query).fetchall()

            symbols = [r[0] for r in result]

            return symbols

        finally:
            conn.close()

    def load_candles(self, symbol: str) -> pd.DataFrame:
        """
        Load OHLCV candles for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            DataFrame with OHLCV data
        """
        conn = duckdb.connect(self.db_path, read_only=True)

        try:
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {self.table_name}
                WHERE symbol = '{symbol}'
                ORDER BY timestamp ASC
            """

            df = conn.execute(query).df()

            if df.empty:
                return pd.DataFrame()

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            return df

        finally:
            conn.close()

    def load_order_book_features(self, symbol: str) -> pd.DataFrame:
        """
        Load order book features for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            DataFrame with order book data
        """
        conn = duckdb.connect(self.db_path, read_only=True)

        try:
            query = f"""
                SELECT
                    timestamp,
                    bid_price, ask_price,
                    spread_percentage,
                    bid_depth_10, ask_depth_10,
                    bid_ask_ratio,
                    weighted_mid_price
                FROM order_book_features
                WHERE symbol = '{symbol}'
                ORDER BY timestamp ASC
            """

            df = conn.execute(query).df()

            if df.empty:
                return pd.DataFrame()

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            return df

        finally:
            conn.close()

    def calculate_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Calculate all ML features for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            DataFrame with calculated features, or None if insufficient data
        """
        logger.info(f"Calculating features for {symbol}...")

        # Load data
        candles = self.load_candles(symbol)

        if candles.empty:
            logger.warning(f"No candles found for {symbol}")
            return None

        if len(candles) < 100:
            logger.warning(f"Insufficient data for {symbol} ({len(candles)} candles)")
            return None

        # Calculate price-based features
        try:
            price_features = calculate_price_features(
                open_=candles['open'],
                high=candles['high'],
                low=candles['low'],
                close=candles['close'],
                volume=candles['volume']
            )
        except Exception as e:
            logger.error(f"Error calculating price features for {symbol}: {e}")
            return None

        # Calculate volume-based features
        try:
            volume_features = calculate_volume_features(
                close=candles['close'],
                volume=candles['volume']
            )
        except Exception as e:
            logger.error(f"Error calculating volume features for {symbol}: {e}")
            return None

        # Combine features
        features = pd.concat([price_features, volume_features], axis=1)

        # Skip microstructure features for now - order book data format doesn't match
        # (needs buy_volume/sell_volume split, we have bid/ask aggregates)
        logger.info(f"{symbol}: Calculated {len(features.columns)} features (price + volume only)")

        # Add metadata
        features['symbol'] = symbol
        features['timeframe'] = '1m'  # Assuming 1-minute

        # Reset index to have timestamp as column
        features = features.reset_index()

        return features

    def write_features(self, features: pd.DataFrame):
        """
        Write features to database.

        Args:
            features: DataFrame with calculated features
        """
        if features.empty:
            return

        conn = duckdb.connect(self.db_path, read_only=False)

        try:
            # Create features table if doesn't exist
            # Get column names and types from DataFrame
            columns = []
            for col, dtype in features.dtypes.items():
                if col == 'timestamp':
                    sql_type = 'TIMESTAMP'
                elif col in ['symbol', 'timeframe']:
                    sql_type = 'VARCHAR'
                elif dtype == 'float64':
                    sql_type = 'DOUBLE'
                elif dtype == 'int64':
                    sql_type = 'BIGINT'
                else:
                    sql_type = 'VARCHAR'

                columns.append(f"{col} {sql_type}")

            create_table = f"""
                CREATE TABLE IF NOT EXISTS features (
                    {', '.join(columns)}
                )
            """

            conn.execute(create_table)

            # Insert features using DuckDB's DataFrame support
            # Register DataFrame as temporary table
            conn.register('temp_features', features)
            conn.execute("INSERT INTO features SELECT * FROM temp_features")
            conn.unregister('temp_features')

            logger.info(f"Wrote {len(features)} feature rows to database")

        finally:
            conn.close()

    def process_symbol(self, symbol: str):
        """
        Process a single symbol: calculate and write features.

        Args:
            symbol: Trading pair
        """
        features = self.calculate_features(symbol)

        if features is not None:
            self.write_features(features)
            logger.success(f"✓ Processed {symbol}: {len(features)} rows")
        else:
            logger.warning(f"✗ Skipped {symbol}")

    def process_all_symbols(self, symbols: Optional[List[str]] = None):
        """
        Process all symbols in database.

        Args:
            symbols: List of symbols (None = all)
        """
        if symbols is None:
            symbols = self.get_all_symbols()

        logger.info(f"Processing {len(symbols)} symbols...")

        success = 0
        failed = 0

        for i, symbol in enumerate(symbols):
            logger.info(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")

            try:
                self.process_symbol(symbol)
                success += 1
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                failed += 1

        logger.info("=" * 80)
        logger.info("BATCH FEATURE CALCULATION COMPLETE")
        logger.info("=" * 80)
        logger.success(f"✓ Success: {success} symbols")
        if failed > 0:
            logger.warning(f"✗ Failed: {failed} symbols")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Calculate features in batch')
    parser.add_argument('--db', required=True, help='Path to DuckDB database')
    parser.add_argument('--symbols', help='Comma-separated symbols (default: all)')
    parser.add_argument('--timeframe', default='1m', help='Timeframe (default: 1m)')
    parser.add_argument('--table', default='candles', help='Candles table name (default: candles)')

    args = parser.parse_args()

    # Parse symbols
    symbols = None
    if args.symbols:
        if args.symbols.lower() == 'all':
            symbols = None
        else:
            symbols = [s.strip() for s in args.symbols.split(',')]

    # Initialize calculator
    calculator = BatchFeatureCalculator(args.db, table_name=args.table)

    # Process symbols
    calculator.process_all_symbols(symbols)


if __name__ == "__main__":
    main()
