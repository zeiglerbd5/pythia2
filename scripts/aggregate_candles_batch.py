#!/usr/bin/env python3
"""
Batch Candle Aggregation from Historical Trades

Converts trades table → candles table with:
- 1-minute OHLCV aggregation
- Buy/sell volume split (from trade sides)
- Gap-filling: forward-fill price, volume=0
- Processes all symbols in database

Usage:
    python scripts/aggregate_candles_batch.py --db /path/to/database.db
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import argparse


def aggregate_candles(
    db_path: str,
    fill_gaps: bool = True,
    symbols: list = None
):
    """
    Aggregate trades into 1-minute OHLCV candles.

    Args:
        db_path: Path to SQLite database
        fill_gaps: Forward-fill gaps with last price, volume=0
        symbols: List of symbols to process (None = all)
    """
    logger.info("=" * 80)
    logger.info("BATCH CANDLE AGGREGATION")
    logger.info("=" * 80)
    logger.info(f"Database: {db_path}")
    logger.info(f"Gap filling: {'Enabled' if fill_gaps else 'Disabled'}")
    logger.info("=" * 80)

    conn = duckdb.connect(db_path)

    # Drop and recreate candles table with proper schema
    logger.info("Setting up candles table schema...")
    conn.execute("DROP TABLE IF EXISTS candles")
    conn.execute("""
        CREATE TABLE candles (
            symbol TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume DOUBLE NOT NULL,
            buy_volume DOUBLE,
            sell_volume DOUBLE,
            num_trades INTEGER,
            is_filled BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX idx_candles_symbol_timestamp ON candles(symbol, timestamp)")
    logger.success("✓ Candles table ready with buy/sell volume support")
    print()

    # Get all symbols if not specified
    if symbols is None:
        symbols_df = conn.execute('SELECT DISTINCT symbol FROM trades ORDER BY symbol').df()
        symbols = symbols_df['symbol'].tolist()

    logger.info(f"Processing {len(symbols)} symbols...")
    print()

    total_candles = 0
    total_filled = 0

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] {symbol}...")

        # Load trades for this symbol
        trades = conn.execute(f'''
            SELECT timestamp, price, size, side
            FROM trades
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
        ''').df()

        if trades.empty:
            logger.warning(f"  No trades found for {symbol}")
            continue

        # Parse timestamps
        trades['timestamp'] = pd.to_datetime(trades['timestamp'])
        trades = trades.set_index('timestamp')

        # Separate buy/sell volumes
        trades['buy_volume'] = trades.apply(
            lambda x: x['size'] if x['side'] == 'buy' else 0,
            axis=1
        )
        trades['sell_volume'] = trades.apply(
            lambda x: x['size'] if x['side'] == 'sell' else 0,
            axis=1
        )

        # Aggregate to 1-minute candles
        candles = pd.DataFrame()
        candles['open'] = trades['price'].resample('1min').first()
        candles['high'] = trades['price'].resample('1min').max()
        candles['low'] = trades['price'].resample('1min').min()
        candles['close'] = trades['price'].resample('1min').last()
        candles['volume'] = trades['size'].resample('1min').sum()
        candles['buy_volume'] = trades['buy_volume'].resample('1min').sum()
        candles['sell_volume'] = trades['sell_volume'].resample('1min').sum()
        candles['num_trades'] = trades['price'].resample('1min').count()

        # Remove completely empty candles (no trades at all)
        candles = candles.dropna(subset=['open'])

        original_count = len(candles)

        # Gap filling if enabled
        if fill_gaps and len(candles) > 0:
            # Generate complete minute range
            start_time = candles.index.min()
            end_time = candles.index.max()
            complete_range = pd.date_range(
                start=start_time.floor('min'),
                end=end_time.floor('min'),
                freq='1min'
            )

            # Reindex to complete range
            candles = candles.reindex(complete_range)

            # Forward-fill OHLC with last known price
            candles[['open', 'high', 'low', 'close']] = candles[['open', 'high', 'low', 'close']].ffill()

            # Fill volumes with 0 (no activity)
            candles['volume'] = candles['volume'].fillna(0)
            candles['buy_volume'] = candles['buy_volume'].fillna(0)
            candles['sell_volume'] = candles['sell_volume'].fillna(0)
            candles['num_trades'] = candles['num_trades'].fillna(0)

            # Mark filled candles
            candles['is_filled'] = candles['num_trades'] == 0

            filled_count = len(candles) - original_count
            total_filled += filled_count
        else:
            candles['is_filled'] = False
            filled_count = 0

        # Add metadata
        candles['symbol'] = symbol
        candles = candles.reset_index()
        candles.rename(columns={'index': 'timestamp'}, inplace=True)

        # Write to database using DuckDB
        try:
            # DuckDB requires register + INSERT
            conn.register('temp_candles', candles)
            conn.execute("""
                INSERT INTO candles
                SELECT symbol, timestamp, open, high, low, close, volume,
                       buy_volume, sell_volume, num_trades, is_filled, CURRENT_TIMESTAMP
                FROM temp_candles
            """)
            conn.unregister('temp_candles')
            total_candles += len(candles)

            if fill_gaps:
                logger.success(
                    f"  ✓ {len(candles):,} candles "
                    f"({original_count:,} original + {filled_count:,} filled)"
                )
            else:
                logger.success(f"  ✓ {len(candles):,} candles")

        except Exception as e:
            logger.error(f"  ✗ Error writing candles: {e}")
            continue

    conn.close()

    # Summary
    logger.info("=" * 80)
    logger.success("AGGREGATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Symbols processed: {len(symbols)}")
    logger.info(f"Total candles: {total_candles:,}")
    if fill_gaps:
        logger.info(f"Filled gaps: {total_filled:,} ({total_filled/total_candles*100:.1f}%)")
        logger.info(f"Original candles: {total_candles - total_filled:,}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate trades to candles')
    parser.add_argument(
        '--db',
        required=True,
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--no-fill-gaps',
        action='store_true',
        help='Disable gap filling (not recommended for ML)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated symbols to process (default: all)'
    )

    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]

    aggregate_candles(
        db_path=args.db,
        fill_gaps=not args.no_fill_gaps,
        symbols=symbols
    )
