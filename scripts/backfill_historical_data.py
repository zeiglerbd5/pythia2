#!/usr/bin/env python3
"""
Backfill Historical Data from Coinbase

Fetches historical OHLCV data from Coinbase Advanced Trade API
to supplement WebSocket collection for model training.

Per implementation guide:
- Need 81+ days for 60-day sequences + 14-day forward window + validation
- Fetch 1m, 5m, 15m candles
- Calculate features on historical data
- Store in DuckDB alongside WebSocket data

Usage:
    python scripts/backfill_historical_data.py --days 90 --symbols BTC-USD,ETH-USD
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
import time

from src.data_ingestion.coinbase_auth import CoinbaseAuth
from src.data_ingestion.database import DuckDBManager
from src.features.feature_engine import FeatureEngine
from src.utils.config import get_config


class HistoricalDataBackfiller:
    """
    Backfill historical data from Coinbase API.

    Fetches OHLCV candles and calculates features.
    """

    def __init__(
        self,
        auth: CoinbaseAuth,
        db_manager: DuckDBManager,
        feature_engine: Optional[FeatureEngine] = None
    ):
        """
        Initialize backfiller.

        Args:
            auth: Coinbase authentication
            db_manager: Database manager
            feature_engine: Feature engine (optional)
        """
        self.auth = auth
        self.db_manager = db_manager
        self.feature_engine = feature_engine

        # Coinbase API endpoints
        self.base_url = "https://api.coinbase.com/api/v3/brokerage"

        logger.info("HistoricalDataBackfiller initialized")

    async def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        granularity: str = '300'  # 5 minutes = 300 seconds
    ) -> pd.DataFrame:
        """
        Fetch historical candles from Coinbase.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            start: Start datetime
            end: End datetime
            granularity: Candle size in seconds (60=1m, 300=5m, 900=15m)

        Returns:
            DataFrame with OHLCV data
        """
        import aiohttp
        import json

        url = f"{self.base_url}/products/{symbol}/candles"

        # Coinbase limits to 300 candles per request
        # Need to chunk requests
        all_candles = []

        current_start = start
        chunk_size = timedelta(hours=25)  # ~300 5-min candles

        while current_start < end:
            current_end = min(current_start + chunk_size, end)

            # Format timestamps (Unix seconds)
            start_unix = int(current_start.timestamp())
            end_unix = int(current_end.timestamp())

            params = {
                'start': str(start_unix),
                'end': str(end_unix),
                'granularity': granularity
            }

            # Get JWT token
            token = self.auth.generate_token(service="retail_rest_api_proxy")

            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            candles = data.get('candles', [])
                            all_candles.extend(candles)

                            logger.info(
                                f"Fetched {len(candles)} candles for {symbol} "
                                f"({current_start} to {current_end})"
                            )
                        else:
                            error_text = await response.text()
                            logger.error(
                                f"Failed to fetch candles: {response.status} - {error_text}"
                            )

                except Exception as e:
                    logger.error(f"Error fetching candles: {e}")

            current_start = current_end

            # Rate limiting (10 requests/second limit)
            await asyncio.sleep(0.2)

        if not all_candles:
            logger.warning(f"No candles fetched for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_candles)

        # Parse columns
        # Coinbase format: {start, low, high, open, close, volume}
        df['timestamp'] = pd.to_datetime(df['start'], unit='s')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        # Add metadata
        df['symbol'] = symbol

        # Map granularity to timeframe
        granularity_map = {
            '60': '1m',
            '300': '5m',
            '900': '15m'
        }
        df['timeframe'] = granularity_map.get(granularity, '5m')

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Select columns
        df = df[['timestamp', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']]

        logger.info(f"Converted {len(df)} candles to DataFrame for {symbol}")

        return df

    async def backfill_symbol(
        self,
        symbol: str,
        days: int = 90,
        granularities: List[str] = ['300']  # 5m default
    ):
        """
        Backfill historical data for a symbol.

        Args:
            symbol: Trading pair
            days: Number of days to backfill
            granularities: List of granularities (60=1m, 300=5m, 900=15m)
        """
        end = datetime.now()
        start = end - timedelta(days=days)

        logger.info(f"Backfilling {symbol} from {start} to {end} ({days} days)")

        for granularity in granularities:
            logger.info(f"Fetching {granularity}s candles...")

            df = await self.fetch_candles(symbol, start, end, granularity)

            if df.empty:
                logger.warning(f"No data fetched for {symbol} @ {granularity}s")
                continue

            # Write to database
            await self.write_ohlcv(df)

            logger.success(
                f"Backfilled {len(df)} candles for {symbol} @ {granularity}s"
            )

    async def write_ohlcv(self, df: pd.DataFrame):
        """
        Write OHLCV data to database.

        Args:
            df: DataFrame with OHLCV data
        """
        if df.empty:
            return

        # Convert to records
        records = df.to_dict('records')

        # Write in batches
        batch_size = 1000

        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]

            # Queue for writing
            for record in batch:
                await self.db_manager.write_ohlcv(
                    timestamp=record['timestamp'],
                    symbol=record['symbol'],
                    timeframe=record['timeframe'],
                    open=record['open'],
                    high=record['high'],
                    low=record['low'],
                    close=record['close'],
                    volume=record['volume']
                )

        # Flush
        await self.db_manager._flush_all_batches()

        logger.info(f"Wrote {len(records)} OHLCV records to database")

    async def calculate_features(self, symbol: str, timeframe: str = '5m'):
        """
        Calculate features on historical data.

        Args:
            symbol: Trading pair
            timeframe: Timeframe
        """
        if not self.feature_engine:
            logger.warning("No feature engine provided, skipping feature calculation")
            return

        logger.info(f"Calculating features for {symbol} @ {timeframe}")

        # Load OHLCV data from database
        # This is a simplified version - you'd need to implement proper loading
        # For now, just log that this step would happen

        logger.info(
            "Feature calculation would happen here. "
            "You'll need to run the feature engine on the historical OHLCV data."
        )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Backfill historical data')
    parser.add_argument('--days', type=int, default=90, help='Days to backfill')
    parser.add_argument(
        '--symbols',
        type=str,
        default='BTC-USD,ETH-USD',
        help='Comma-separated symbols'
    )
    parser.add_argument(
        '--granularities',
        type=str,
        default='300',  # 5m
        help='Comma-separated granularities (60=1m, 300=5m, 900=15m)'
    )

    args = parser.parse_args()

    symbols = args.symbols.split(',')
    granularities = args.granularities.split(',')

    logger.info("=" * 80)
    logger.info("HISTORICAL DATA BACKFILL")
    logger.info("=" * 80)
    logger.info(f"Days: {args.days}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Granularities: {granularities}")
    logger.info("=" * 80)

    # Initialize components
    config = get_config()
    auth = CoinbaseAuth.from_env()

    db_path = str(config.get_database_path())
    db_manager = DuckDBManager(db_path=db_path)

    # Create backfiller
    backfiller = HistoricalDataBackfiller(
        auth=auth,
        db_manager=db_manager
    )

    # Backfill each symbol
    for symbol in symbols:
        await backfiller.backfill_symbol(
            symbol=symbol.strip(),
            days=args.days,
            granularities=granularities
        )

    logger.success("Backfill complete!")


if __name__ == "__main__":
    asyncio.run(main())
