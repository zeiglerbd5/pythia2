"""
Efficient Historical OHLCV Backfill from Coinbase

Maximizes throughput with async requests while respecting rate limits.
Designed for minimal collector downtime.

Uses AUTHENTICATED API for 30 req/sec (3x faster than public 10 req/sec).

Usage:
    # Stop collector first, then:
    python scripts/backfill_ohlcv.py --days 30
    python scripts/backfill_ohlcv.py --days 90 --symbols-file data/priority_symbols.txt

Rate limit: ~30 req/sec (authenticated) = 8K requests in ~4.5 min (30 days for 118 symbols)
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import asyncio
import aiohttp
import argparse
import duckdb
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import time
import json

from src.data_ingestion.coinbase_auth import CoinbaseAuth


# Configuration
DB_PATH = '/Users/bz/Pythia2/data/pythia.duckdb'
PROGRESS_FILE = '/Users/bz/Pythia2/data/backfill_progress.json'

# Coinbase API - using Advanced Trade API (authenticated)
BASE_URL = 'https://api.coinbase.com/api/v3/brokerage'
CANDLES_PER_REQUEST = 300
GRANULARITY = 'ONE_MINUTE'  # Advanced Trade API format

# Rate limiting - Authenticated API allows 30/sec
MAX_CONCURRENT = 30
RETRY_DELAY = 2.0
MAX_RETRIES = 3

# Batch size for DB writes
DB_BATCH_SIZE = 10000


class RateLimiter:
    """Token bucket rate limiter."""
    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


async def fetch_candles(
    session: aiohttp.ClientSession,
    auth: CoinbaseAuth,
    symbol: str,
    start: datetime,
    end: datetime,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore
) -> list:
    """Fetch candles for a time range with rate limiting and retries."""

    # Advanced Trade API endpoint
    url = f"{BASE_URL}/products/{symbol}/candles"
    request_path = f"/api/v3/brokerage/products/{symbol}/candles"

    # Convert timestamps to Unix seconds
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    params = {
        'granularity': GRANULARITY,
        'start': str(start_ts),
        'end': str(end_ts)
    }

    for attempt in range(MAX_RETRIES):
        await rate_limiter.acquire()

        async with semaphore:
            try:
                # Get fresh auth headers
                headers = auth.get_auth_headers("GET", request_path)

                async with session.get(url, params=params, headers=headers, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Advanced Trade API returns {"candles": [...]}
                        # Each candle: {start, low, high, open, close, volume}
                        return data.get('candles', [])

                    elif resp.status == 429:
                        # Rate limited - back off
                        logger.warning(f"{symbol}: Rate limited, backing off...")
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))

                    elif resp.status == 404:
                        # Symbol doesn't exist or no data
                        return []

                    else:
                        text = await resp.text()
                        logger.warning(f"{symbol}: HTTP {resp.status}: {text[:100]}")
                        await asyncio.sleep(RETRY_DELAY)

            except asyncio.TimeoutError:
                logger.warning(f"{symbol}: Timeout, retrying...")
                await asyncio.sleep(RETRY_DELAY)
            except Exception as e:
                logger.error(f"{symbol}: Error {e}")
                await asyncio.sleep(RETRY_DELAY)

    return []


async def backfill_symbol(
    session: aiohttp.ClientSession,
    auth: CoinbaseAuth,
    symbol: str,
    target_start: datetime,
    current_earliest: datetime,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    progress_callback
) -> list:
    """
    Backfill all candles for a symbol from target_start to current_earliest.

    Returns list of candle tuples: (timestamp, open, high, low, close, volume)
    """
    all_candles = []

    # Work backwards from current_earliest to target_start
    current_end = current_earliest
    chunk_minutes = CANDLES_PER_REQUEST

    requests_made = 0

    while current_end > target_start:
        current_start = max(target_start, current_end - timedelta(minutes=chunk_minutes))

        candles = await fetch_candles(
            session, auth, symbol, current_start, current_end,
            rate_limiter, semaphore
        )

        requests_made += 1

        if candles:
            # Convert from Advanced Trade API format to our format
            for c in candles:
                # Advanced Trade: {start, low, high, open, close, volume}
                ts = datetime.fromtimestamp(int(c['start']), tz=timezone.utc).replace(tzinfo=None)
                all_candles.append({
                    'symbol': symbol,
                    'timestamp': ts,
                    'timeframe': '1m',
                    'open': float(c['open']),
                    'high': float(c['high']),
                    'low': float(c['low']),
                    'close': float(c['close']),
                    'volume': float(c['volume'])
                })

        current_end = current_start

        # Progress callback every 10 requests
        if requests_made % 10 == 0:
            progress_callback(symbol, len(all_candles), requests_made)

    return all_candles


def get_existing_data_ranges(conn) -> dict:
    """Get the earliest timestamp for each symbol in the database."""
    result = conn.execute("""
        SELECT symbol, MIN(timestamp) as earliest, MAX(timestamp) as latest, COUNT(*) as cnt
        FROM ohlcv
        WHERE timeframe = '1m'
        GROUP BY symbol
    """).fetchall()

    return {
        row[0]: {
            'earliest': row[1],
            'latest': row[2],
            'count': row[3]
        }
        for row in result
    }


def get_priority_symbols(conn) -> list:
    """Get symbols that have shown spikes (from labels) or have significant data."""

    # First try to get symbols from spike events if they exist
    try:
        events_df = pd.read_parquet('/Users/bz/Pythia2/data/spike_events.parquet')
        spike_symbols = events_df['symbol'].unique().tolist()
        logger.info(f"Found {len(spike_symbols)} symbols with spike events")
        return spike_symbols
    except FileNotFoundError:
        pass

    # Fall back to symbols with positive labels
    try:
        labels_df = pd.read_parquet('/Users/bz/Pythia2/data/big_mover_labels.parquet')
        pos_symbols = labels_df[labels_df['label'] == 1]['symbol'].unique().tolist()
        logger.info(f"Found {len(pos_symbols)} symbols with positive labels")
        return pos_symbols
    except FileNotFoundError:
        pass

    # Fall back to all symbols with sufficient data
    result = conn.execute("""
        SELECT symbol FROM ohlcv
        WHERE timeframe = '1m'
        GROUP BY symbol
        HAVING COUNT(*) >= 500
        ORDER BY COUNT(*) DESC
    """).fetchall()

    symbols = [r[0] for r in result]
    logger.info(f"Using {len(symbols)} symbols with 500+ candles")
    return symbols


def save_progress(progress: dict):
    """Save progress to file for resume capability."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def load_progress() -> dict:
    """Load progress from file."""
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


async def main(days: int, symbols_file: str = None, dry_run: bool = False):
    logger.info("=" * 60)
    logger.info("OHLCV BACKFILL - AUTHENTICATED API (30 req/sec)")
    logger.info("=" * 60)

    # Load environment
    load_dotenv()

    # Initialize auth
    try:
        auth = CoinbaseAuth.from_env()
        logger.info("Authenticated with Coinbase API")
    except Exception as e:
        logger.error(f"Failed to initialize auth: {e}")
        logger.error("Make sure COINBASE_API_KEY and COINBASE_API_SECRET are set")
        return

    # Connect to database
    conn = duckdb.connect(DB_PATH, read_only=dry_run)

    # Get existing data ranges
    existing_ranges = get_existing_data_ranges(conn)
    logger.info(f"Found existing data for {len(existing_ranges)} symbols")

    # Get priority symbols
    if symbols_file and Path(symbols_file).exists():
        with open(symbols_file) as f:
            symbols = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")
    else:
        symbols = get_priority_symbols(conn)

    # Calculate target start date
    target_start = datetime.utcnow() - timedelta(days=days)
    logger.info(f"Target backfill: {target_start} to now ({days} days)")

    # Calculate work needed
    work_items = []
    total_requests_estimate = 0

    for symbol in symbols:
        if symbol in existing_ranges:
            current_earliest = existing_ranges[symbol]['earliest']
            if current_earliest <= target_start:
                continue  # Already have enough data
        else:
            current_earliest = datetime.utcnow()

        minutes_needed = (current_earliest - target_start).total_seconds() / 60
        requests_needed = int(np.ceil(minutes_needed / CANDLES_PER_REQUEST))

        if requests_needed > 0:
            work_items.append({
                'symbol': symbol,
                'target_start': target_start,
                'current_earliest': current_earliest,
                'requests_needed': requests_needed
            })
            total_requests_estimate += requests_needed

    logger.info(f"\nWork to do:")
    logger.info(f"  Symbols needing backfill: {len(work_items)}")
    logger.info(f"  Total requests estimated: {total_requests_estimate:,}")
    logger.info(f"  Estimated time at 30 req/s: {total_requests_estimate / 30 / 60:.1f} minutes")

    if dry_run:
        logger.info("\nDRY RUN - no data will be fetched")
        conn.close()
        return

    if not work_items:
        logger.info("Nothing to backfill!")
        conn.close()
        return

    # Load previous progress
    progress = load_progress()

    # Initialize rate limiter and semaphore
    rate_limiter = RateLimiter(30)  # 30 requests per second (authenticated)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Stats tracking
    stats = {
        'symbols_completed': 0,
        'candles_fetched': 0,
        'requests_made': 0,
        'start_time': time.time()
    }

    def progress_callback(symbol, candles, requests):
        stats['candles_fetched'] += candles
        stats['requests_made'] = requests
        elapsed = time.time() - stats['start_time']
        rate = stats['requests_made'] / max(elapsed, 1)
        logger.info(f"  {symbol}: {candles:,} candles ({requests} requests, {rate:.1f} req/s)")

    # Create aiohttp session with connection pooling
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, limit_per_host=MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Process symbols
        all_candles = []

        for i, work in enumerate(work_items):
            symbol = work['symbol']

            # Skip if already completed in previous run
            if progress.get(symbol) == 'completed':
                logger.info(f"[{i+1}/{len(work_items)}] {symbol}: Skipping (already completed)")
                continue

            logger.info(f"[{i+1}/{len(work_items)}] {symbol}: Fetching {work['requests_needed']} chunks...")

            candles = await backfill_symbol(
                session, auth, symbol,
                work['target_start'],
                work['current_earliest'],
                rate_limiter, semaphore,
                progress_callback
            )

            if candles:
                all_candles.extend(candles)
                logger.info(f"  {symbol}: Got {len(candles):,} candles")

            # Mark as completed
            progress[symbol] = 'completed'
            save_progress(progress)

            stats['symbols_completed'] += 1

            # Write to DB in batches
            if len(all_candles) >= DB_BATCH_SIZE:
                logger.info(f"Writing batch of {len(all_candles):,} candles to database...")
                df = pd.DataFrame(all_candles)
                conn.execute("""
                    INSERT INTO ohlcv (symbol, timestamp, timeframe, open, high, low, close, volume)
                    SELECT symbol, timestamp, timeframe, open, high, low, close, volume
                    FROM df
                    ON CONFLICT DO NOTHING
                """)
                all_candles = []

        # Write remaining candles
        if all_candles:
            logger.info(f"Writing final batch of {len(all_candles):,} candles to database...")
            df = pd.DataFrame(all_candles)
            conn.execute("""
                INSERT INTO ohlcv (symbol, timestamp, timeframe, open, high, low, close, volume)
                SELECT symbol, timestamp, timeframe, open, high, low, close, volume
                FROM df
                ON CONFLICT DO NOTHING
            """)

    # Final stats
    elapsed = time.time() - stats['start_time']

    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Symbols processed: {stats['symbols_completed']}")
    logger.info(f"Total candles fetched: {stats['candles_fetched']:,}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Average rate: {stats['candles_fetched'] / max(elapsed, 1):.0f} candles/sec")

    # Clean up progress file
    if Path(PROGRESS_FILE).exists():
        Path(PROGRESS_FILE).unlink()

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical OHLCV data")
    parser.add_argument('--days', type=int, default=30, help="Days of history to fetch")
    parser.add_argument('--symbols-file', type=str, help="File with symbols to fetch (one per line)")
    parser.add_argument('--dry-run', action='store_true', help="Calculate work without fetching")
    args = parser.parse_args()

    asyncio.run(main(args.days, args.symbols_file, args.dry_run))
