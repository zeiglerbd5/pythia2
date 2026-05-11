"""
DuckDB Database Manager for Pythia

Implements time-series data storage with batch writing for optimal performance.
Uses DuckDB columnar storage for efficient analytical queries.

Per implementation guide adaptations:
- DuckDB instead of QuestDB (single-machine deployment)
- Batch writing with configurable intervals
- Columnar storage for fast aggregations
- ASOF JOIN capabilities for backtesting
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import deque

import duckdb
import pandas as pd
from loguru import logger


class DuckDBManager:
    """
    Manages DuckDB database for time-series market data storage.

    Features:
    - Async batch writing with configurable intervals
    - Columnar storage optimized for analytical queries
    - Automatic table partitioning by timestamp
    - ASOF JOIN support for backtesting
    """

    def __init__(
        self,
        db_path: str = "data/pythia.duckdb",
        batch_size: int = 1000,
        batch_timeout_seconds: float = 5.0
    ):
        """
        Initialize DuckDB manager.

        Args:
            db_path: Path to DuckDB database file
            batch_size: Number of records to batch before writing
            batch_timeout_seconds: Max seconds to wait before flushing batch
        """
        self.db_path = Path(db_path)
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connection pool (DuckDB supports concurrent reads)
        self.conn = duckdb.connect(str(self.db_path))

        # Batch queues for async writing
        self.ticker_queue: deque = deque()
        self.trade_queue: deque = deque()
        self.orderbook_queue: deque = deque()
        self.candle_queue: deque = deque()
        self.feature_queue: deque = deque()
        self.news_signal_queue: deque = deque()
        self.whale_transaction_queue: deque = deque()

        # Batch writing task
        self._batch_writer_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self.stats = {
            "tickers_written": 0,
            "trades_written": 0,
            "orderbooks_written": 0,
            "candles_written": 0,
            "features_written": 0,
            "news_signals_written": 0,
            "whale_transactions_written": 0,
            "batches_written": 0,
        }

        # Initialize schema
        self.init_schema()

        logger.info(f"DuckDB initialized at {self.db_path}")

    def init_schema(self):
        """
        Initialize database schema.

        Creates tables optimized for time-series queries with indices.
        """
        # Ticker data (real-time best bid/ask updates)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS tickers (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                best_bid DOUBLE,
                best_ask DOUBLE,
                price DOUBLE,
                volume_24h DOUBLE,
                price_change_24h DOUBLE,
                price_change_pct_24h DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            )
        ''')

        # Trade executions (market_trades channel)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                trade_id VARCHAR,
                price DOUBLE NOT NULL,
                size DOUBLE NOT NULL,
                side VARCHAR,  -- 'BUY' or 'SELL'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp
            ON trades(symbol, timestamp)
        ''')

        # Order book snapshots (periodic snapshots for features)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS order_book_snapshots (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                bids JSON,  -- Array of [price, quantity] tuples
                asks JSON,  -- Array of [price, quantity] tuples
                best_bid DOUBLE,
                best_ask DOUBLE,
                mid_price DOUBLE,
                spread DOUBLE,
                spread_bps DOUBLE,
                sequence_num BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            )
        ''')

        # OHLCV candles (multiple timeframes)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                timeframe VARCHAR NOT NULL,  -- '1m', '5m', '15m', '1h'
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                num_trades INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp, timeframe)
            )
        ''')

        # Computed features (from feature engineering pipeline)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS features (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                timeframe VARCHAR NOT NULL DEFAULT '5m',

                -- Order book microstructure (per implementation guide)
                order_book_imbalance_l5 DOUBLE,  -- ρ at L=5 depth
                roll_measure DOUBLE,              -- Roll measure (top predictor)
                vpin DOUBLE,                      -- Volume-synchronized PIN

                -- Volume indicators
                volume DOUBLE,
                volume_spike_ratio DOUBLE,        -- Current / 20-period avg
                obv DOUBLE,                       -- On-balance volume
                vroc DOUBLE,                      -- Volume rate of change

                -- Price indicators
                rsi_14 DOUBLE,
                vwap DOUBLE,
                vwap_std DOUBLE,
                vwap_distance_pct DOUBLE,
                atr DOUBLE,
                natr DOUBLE,                      -- Normalized ATR

                -- Bollinger Bands
                bb_upper DOUBLE,
                bb_middle DOUBLE,
                bb_lower DOUBLE,
                bb_width DOUBLE,

                -- Additional microstructure
                bid_ask_ratio DOUBLE,
                weighted_mid_price DOUBLE,
                large_bid_orders INTEGER,
                large_ask_orders INTEGER,

                -- Order book features for paper trading entry signals
                bid_ask_spread_pct DOUBLE,
                order_book_depth_ratio DOUBLE,
                large_order_imbalance DOUBLE,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp, timeframe)
            )
        ''')

        # Add new columns to existing databases (migrations)
        new_columns = [
            ('bid_ask_spread_pct', 'DOUBLE'),
            ('order_book_depth_ratio', 'DOUBLE'),
            ('large_order_imbalance', 'DOUBLE'),
        ]
        for col_name, col_type in new_columns:
            try:
                self.conn.execute(f"ALTER TABLE features ADD COLUMN {col_name} {col_type}")
                logger.info(f"Added column {col_name} to features table")
            except Exception:
                pass  # Column already exists

        # News signals (from news monitoring system)
        # Migration: drop old table if it has the id column
        try:
            self.conn.execute("SELECT id FROM news_signals LIMIT 1")
            self.conn.execute("DROP TABLE news_signals")
            logger.info("Dropped old news_signals table (had id column)")
        except Exception:
            pass  # Table doesn't exist or doesn't have id column

        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS news_signals (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                source VARCHAR NOT NULL,
                event_type VARCHAR NOT NULL,
                confidence DOUBLE NOT NULL,
                title TEXT,
                url TEXT,
                signal_hash VARCHAR,
                source_credibility DOUBLE,
                entity_certainty DOUBLE,
                event_priority DOUBLE,
                recency_score DOUBLE,
                engagement_score DOUBLE,
                sentiment_score DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Migration: add sentiment_score to existing databases
        try:
            self.conn.execute("ALTER TABLE news_signals ADD COLUMN sentiment_score DOUBLE")
            logger.info("Added sentiment_score column to news_signals table")
        except Exception:
            pass  # Column already exists
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_news_signals_symbol_timestamp
            ON news_signals(symbol, timestamp)
        ''')

        # Whale transactions table (raw whale alert data)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS whale_transactions (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                amount_usd DOUBLE NOT NULL,
                subtype VARCHAR NOT NULL,
                from_name VARCHAR,
                to_name VARCHAR,
                blockchain VARCHAR,
                tx_hash VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_whale_transactions_symbol_timestamp
            ON whale_transactions(symbol, timestamp)
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_whale_transactions_subtype
            ON whale_transactions(subtype)
        ''')

        logger.info("Database schema initialized")

    async def start_batch_writer(self):
        """Start async batch writer task."""
        if self._batch_writer_task is None or self._batch_writer_task.done():
            self._running = True
            self._batch_writer_task = asyncio.create_task(self._batch_writer_loop())
            logger.info("Batch writer started")

    async def stop_batch_writer(self):
        """Stop async batch writer and flush remaining data."""
        self._running = False

        if self._batch_writer_task:
            # Flush remaining data
            await self._flush_all_batches()

            # Cancel task
            if not self._batch_writer_task.done():
                self._batch_writer_task.cancel()
                try:
                    await self._batch_writer_task
                except asyncio.CancelledError:
                    pass

        logger.info("Batch writer stopped")

    async def _batch_writer_loop(self):
        """
        Background task that periodically flushes batch queues.

        Flushes when:
        - Batch size is reached
        - Timeout is exceeded
        """
        last_flush = time.time()

        while self._running:
            try:
                current_time = time.time()
                time_since_flush = current_time - last_flush

                # Check if timeout exceeded
                should_flush = time_since_flush >= self.batch_timeout_seconds

                # Check if any queue is full
                if not should_flush:
                    should_flush = (
                        len(self.ticker_queue) >= self.batch_size or
                        len(self.trade_queue) >= self.batch_size or
                        len(self.orderbook_queue) >= self.batch_size or
                        len(self.candle_queue) >= self.batch_size or
                        len(self.feature_queue) >= self.batch_size or
                        len(self.news_signal_queue) >= self.batch_size or
                        len(self.whale_transaction_queue) >= self.batch_size
                    )

                if should_flush:
                    await self._flush_all_batches()
                    last_flush = time.time()

                # Sleep briefly
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in batch writer loop: {e}")
                await asyncio.sleep(1)

    async def _flush_all_batches(self):
        """Flush all batch queues to database."""
        try:
            # Flush each queue
            if self.ticker_queue:
                await self._flush_tickers()

            if self.trade_queue:
                await self._flush_trades()

            if self.orderbook_queue:
                await self._flush_orderbooks()

            if self.candle_queue:
                await self._flush_candles()

            if self.feature_queue:
                await self._flush_features()

            if self.news_signal_queue:
                await self._flush_news_signals()

            if self.whale_transaction_queue:
                await self._flush_whale_transactions()

            self.stats["batches_written"] += 1

        except Exception as e:
            logger.error(f"Error flushing batches: {e}")

    async def _flush_tickers(self):
        """Flush ticker queue to database."""
        if not self.ticker_queue:
            return

        # Convert queue to DataFrame
        records = []
        while self.ticker_queue:
            records.append(self.ticker_queue.popleft())

        if not records:
            return

        try:
            df = pd.DataFrame(records)

            # Filter out records with NULL symbol or timestamp (NOT NULL in schema)
            initial_count = len(df)
            df = df.dropna(subset=['symbol', 'timestamp'])
            if len(df) < initial_count:
                logger.debug(f"Dropped {initial_count - len(df)} ticker records with NULL required fields")

            if len(df) == 0:
                return

            # Convert empty strings to NaN for numeric columns (DuckDB can't cast '' to DOUBLE)
            numeric_cols = ['best_bid', 'best_ask', 'price', 'volume_24h', 'price_change_24h', 'price_change_pct_24h']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # DuckDB requires ON CONFLICT syntax, not INSERT OR REPLACE
            self.conn.register('tickers_df', df)
            self.conn.execute("""
                INSERT INTO tickers (symbol, timestamp, best_bid, best_ask, price,
                                    volume_24h, price_change_24h, price_change_pct_24h, created_at)
                SELECT symbol, timestamp, best_bid, best_ask, price,
                       volume_24h, price_change_24h, price_change_pct_24h, created_at
                FROM tickers_df
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    best_bid = EXCLUDED.best_bid,
                    best_ask = EXCLUDED.best_ask,
                    price = EXCLUDED.price,
                    volume_24h = EXCLUDED.volume_24h,
                    price_change_24h = EXCLUDED.price_change_24h,
                    price_change_pct_24h = EXCLUDED.price_change_pct_24h,
                    created_at = EXCLUDED.created_at
            """)
            self.conn.unregister('tickers_df')
            self.stats["tickers_written"] += len(df)
            logger.debug(f"Flushed {len(df)} tickers")

        except Exception as e:
            logger.error(f"Error flushing tickers: {e}")

    async def _flush_trades(self):
        """Flush trade queue to database."""
        if not self.trade_queue:
            return

        records = []
        while self.trade_queue:
            records.append(self.trade_queue.popleft())

        if not records:
            return

        try:
            df = pd.DataFrame(records)
            # Explicitly specify columns excluding auto-generated 'id'
            self.conn.execute("""
                INSERT INTO trades (symbol, trade_id, price, size, side, timestamp, created_at)
                SELECT symbol, trade_id, price, size, side, timestamp, created_at FROM df
            """)
            self.stats["trades_written"] += len(records)
            logger.debug(f"Flushed {len(records)} trades")

        except Exception as e:
            logger.error(f"Error flushing trades: {e}")

    async def _flush_orderbooks(self):
        """Flush order book snapshots to database."""
        if not self.orderbook_queue:
            return

        records = []
        while self.orderbook_queue:
            records.append(self.orderbook_queue.popleft())

        if not records:
            return

        try:
            df = pd.DataFrame(records)

            # DuckDB requires ON CONFLICT syntax, not INSERT OR REPLACE
            self.conn.register('orderbooks_df', df)
            self.conn.execute("""
                INSERT INTO order_book_snapshots (symbol, timestamp, bids, asks, best_bid,
                                                  best_ask, mid_price, spread, spread_bps,
                                                  sequence_num, created_at)
                SELECT symbol, timestamp, bids, asks, best_bid, best_ask, mid_price,
                       spread, spread_bps, sequence_num, created_at
                FROM orderbooks_df
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    bids = EXCLUDED.bids,
                    asks = EXCLUDED.asks,
                    best_bid = EXCLUDED.best_bid,
                    best_ask = EXCLUDED.best_ask,
                    mid_price = EXCLUDED.mid_price,
                    spread = EXCLUDED.spread,
                    spread_bps = EXCLUDED.spread_bps,
                    sequence_num = EXCLUDED.sequence_num,
                    created_at = EXCLUDED.created_at
            """)
            self.conn.unregister('orderbooks_df')
            self.stats["orderbooks_written"] += len(records)
            logger.debug(f"Flushed {len(records)} order book snapshots")

        except Exception as e:
            logger.error(f"Error flushing order books: {e}")

    async def _flush_candles(self):
        """Flush candle queue to database."""
        if not self.candle_queue:
            return

        records = []
        while self.candle_queue:
            records.append(self.candle_queue.popleft())

        if not records:
            return

        try:
            df = pd.DataFrame(records)

            # Filter out records with NULL/None symbol, timestamp, or timeframe (all NOT NULL in schema)
            initial_count = len(df)
            df = df.dropna(subset=['symbol', 'timestamp', 'timeframe'])
            if len(df) < initial_count:
                logger.warning(f"Dropped {initial_count - len(df)} candle records with NULL required fields")

            if len(df) == 0:
                logger.warning("No valid candle records to flush after NULL filtering")
                return

            # DuckDB requires ON CONFLICT syntax, not INSERT OR REPLACE
            self.conn.register('candles_df', df)
            self.conn.execute("""
                INSERT INTO ohlcv (symbol, timestamp, timeframe, open, high, low, close,
                                   volume, num_trades, created_at)
                SELECT symbol, timestamp, timeframe, open, high, low, close,
                       volume, num_trades, created_at
                FROM candles_df
                ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    num_trades = EXCLUDED.num_trades,
                    created_at = EXCLUDED.created_at
            """)
            self.conn.unregister('candles_df')
            self.stats["candles_written"] += len(df)
            logger.debug(f"Flushed {len(df)} candles")

        except Exception as e:
            logger.error(f"Error flushing candles: {e}")

    async def _flush_features(self):
        """Flush features queue to database."""
        if not self.feature_queue:
            return

        records = []
        while self.feature_queue:
            records.append(self.feature_queue.popleft())

        if not records:
            return

        try:
            df = pd.DataFrame(records)

            # Normalize column names to match database schema (lowercase)
            # Map any alternative names to the actual schema column names
            column_mapping = {
                'rsi': 'rsi_14',
                'RSI_14': 'rsi_14',
                'NATR': 'natr',
                'MACD': 'macd',
                'MACD_signal': 'macd_signal',
                'MACD_hist': 'macd_hist',
                'BB_width': 'bb_width',
                'BB_squeeze': 'bb_squeeze',
                'VWAP_distance': 'vwap_distance_pct',
                'OBV': 'obv',
            }

            # Rename columns that exist in both the DataFrame and the mapping
            df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

            # Filter to only columns that exist in the database schema
            # Must match actual schema column names (lowercase)
            valid_columns = {
                'symbol', 'timestamp', 'timeframe',
                'order_book_imbalance_l5', 'roll_measure', 'vpin',
                'volume', 'volume_spike_ratio', 'obv', 'vroc',
                'rsi_14', 'vwap', 'vwap_std', 'vwap_distance_pct', 'atr', 'natr',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'bid_ask_ratio', 'weighted_mid_price', 'large_bid_orders', 'large_ask_orders',
                'bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance',
            }

            # Keep only columns that exist in valid_columns
            df = df[[col for col in df.columns if col in valid_columns]]

            # Explicitly cast symbol and timeframe to string to avoid type inference issues
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].astype(str)
            if 'timeframe' in df.columns:
                df['timeframe'] = df['timeframe'].astype(str)

            # Features table has UNIQUE index on (symbol, timestamp, timeframe)
            # Use dynamic column list based on what's actually in the DataFrame
            cols = list(df.columns)
            cols_str = ", ".join(cols)

            # Build ON CONFLICT SET clause for all columns except the key columns
            update_cols = [c for c in cols if c not in ["symbol", "timestamp", "timeframe"]]
            update_str = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])

            # Explicitly register the DataFrame to avoid type inference issues
            self.conn.register('features_df', df)

            query = f"""
                INSERT INTO features ({cols_str})
                SELECT {cols_str} FROM features_df
                ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET {update_str}
            """

            self.conn.execute(query)
            self.conn.unregister('features_df')
            self.stats["features_written"] += len(df)
            logger.debug(f"Flushed {len(df)} features")

        except Exception as e:
            logger.error(f"Error flushing features: {e}")

    async def _flush_news_signals(self):
        """Flush news signals queue to database."""
        if not self.news_signal_queue:
            return

        records = []
        while self.news_signal_queue:
            records.append(self.news_signal_queue.popleft())

        if not records:
            return

        try:
            df = pd.DataFrame(records)

            self.conn.register('news_signals_df', df)
            self.conn.execute("""
                INSERT INTO news_signals (symbol, timestamp, source, event_type, confidence,
                                         title, url, signal_hash, source_credibility,
                                         entity_certainty, event_priority, recency_score,
                                         engagement_score, sentiment_score, created_at)
                SELECT symbol, timestamp, source, event_type, confidence,
                       title, url, signal_hash, source_credibility,
                       entity_certainty, event_priority, recency_score,
                       engagement_score, sentiment_score, created_at
                FROM news_signals_df
            """)
            self.conn.unregister('news_signals_df')
            self.stats["news_signals_written"] += len(df)
            logger.debug(f"Flushed {len(df)} news signals")

        except Exception as e:
            logger.error(f"Error flushing news signals: {e}")

    async def _flush_whale_transactions(self):
        """Flush whale transactions queue to database."""
        if not self.whale_transaction_queue:
            return

        records = []
        while self.whale_transaction_queue:
            records.append(self.whale_transaction_queue.popleft())

        if not records:
            return

        try:
            df = pd.DataFrame(records)

            self.conn.register('whale_transactions_df', df)
            self.conn.execute("""
                INSERT INTO whale_transactions (symbol, timestamp, amount_usd, subtype,
                                               from_name, to_name, blockchain, tx_hash, created_at)
                SELECT symbol, timestamp, amount_usd, subtype,
                       from_name, to_name, blockchain, tx_hash, created_at
                FROM whale_transactions_df
            """)
            self.conn.unregister('whale_transactions_df')
            self.stats["whale_transactions_written"] += len(df)
            logger.debug(f"Flushed {len(df)} whale transactions")

        except Exception as e:
            logger.error(f"Error flushing whale transactions: {e}")

    # Public methods for adding data to queues

    def queue_ticker(self, symbol: str, data: Dict[str, Any]):
        """Add ticker data to batch queue."""
        record = {
            "symbol": symbol,
            "timestamp": pd.to_datetime(data.get("timestamp", datetime.now())),
            "best_bid": data.get("best_bid"),
            "best_ask": data.get("best_ask"),
            "price": data.get("price"),
            "volume_24h": data.get("volume_24h"),
            "price_change_24h": data.get("price_change_24h"),
            "price_change_pct_24h": data.get("price_change_pct_24h"),
            "created_at": datetime.now(),
        }
        self.ticker_queue.append(record)

    def queue_trade(self, symbol: str, data: Dict[str, Any]):
        """Add trade data to batch queue."""
        record = {
            "symbol": symbol,
            "timestamp": pd.to_datetime(data.get("timestamp", datetime.now())),
            "trade_id": data.get("trade_id"),
            "price": float(data.get("price", 0)),
            "size": float(data.get("size", 0)),
            "side": data.get("side"),
            "created_at": datetime.now(),
        }
        self.trade_queue.append(record)

    def queue_orderbook(self, symbol: str, snapshot: Dict[str, Any]):
        """Add order book snapshot to batch queue."""
        record = {
            "symbol": symbol,
            "timestamp": pd.to_datetime(snapshot.get("timestamp", datetime.now())),
            "bids": snapshot.get("bids"),  # JSON array
            "asks": snapshot.get("asks"),  # JSON array
            "best_bid": snapshot.get("best_bid"),
            "best_ask": snapshot.get("best_ask"),
            "mid_price": snapshot.get("mid_price"),
            "spread": snapshot.get("spread"),
            "spread_bps": snapshot.get("spread_bps"),
            "sequence_num": snapshot.get("sequence_num"),
            "created_at": datetime.now(),
        }
        self.orderbook_queue.append(record)

    def queue_candle(self, symbol: str, timeframe: str, data: Dict[str, Any]):
        """Add OHLCV candle to batch queue."""
        record = {
            "symbol": symbol,
            "timestamp": pd.to_datetime(data.get("timestamp", datetime.now())),
            "timeframe": timeframe,
            "open": float(data.get("open", 0)),
            "high": float(data.get("high", 0)),
            "low": float(data.get("low", 0)),
            "close": float(data.get("close", 0)),
            "volume": float(data.get("volume", 0)),
            "num_trades": data.get("num_trades"),
            "created_at": datetime.now(),
        }
        self.candle_queue.append(record)

    def queue_features(self, symbol: str, timeframe: str, features: Dict[str, Any]):
        """Add computed features to batch queue."""
        record = {
            "symbol": symbol,
            "timestamp": pd.to_datetime(features.get("timestamp", datetime.now())),
            "timeframe": timeframe,
            **{k: v for k, v in features.items() if k not in ["symbol", "timestamp", "timeframe"]}
        }
        self.feature_queue.append(record)

    def queue_news_signal(self, signal: Dict[str, Any]):
        """Add news signal to batch queue."""
        record = {
            "symbol": signal.get("symbol"),
            "timestamp": pd.to_datetime(signal.get("timestamp", datetime.now())),
            "source": signal.get("source"),
            "event_type": signal.get("event_type"),
            "confidence": signal.get("confidence"),
            "title": signal.get("title"),
            "url": signal.get("url"),
            "signal_hash": signal.get("signal_hash"),
            "source_credibility": signal.get("source_credibility"),
            "entity_certainty": signal.get("entity_certainty"),
            "event_priority": signal.get("event_priority"),
            "recency_score": signal.get("recency_score"),
            "engagement_score": signal.get("engagement_score"),
            "sentiment_score": signal.get("sentiment_score"),
            "created_at": datetime.now(),
        }
        self.news_signal_queue.append(record)

    def queue_whale_transaction(self, tx: Dict[str, Any]):
        """Add whale transaction to batch queue."""
        record = {
            "symbol": tx.get("symbol"),
            "timestamp": pd.to_datetime(tx.get("timestamp", datetime.now())),
            "amount_usd": tx.get("amount_usd"),
            "subtype": tx.get("subtype"),
            "from_name": tx.get("from_name"),
            "to_name": tx.get("to_name"),
            "blockchain": tx.get("blockchain"),
            "tx_hash": tx.get("tx_hash"),
            "created_at": datetime.now(),
        }
        self.whale_transaction_queue.append(record)

    # Query methods

    def get_recent_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get recent trades for a symbol."""
        query = f"""
            SELECT * FROM trades
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        return self.conn.execute(query, [symbol, limit]).df()

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get OHLCV data for a symbol and timeframe."""
        query = """
            SELECT * FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp"

        return self.conn.execute(query, params).df()

    def get_features(
        self,
        symbol: str,
        timeframe: str = "5m",
        start_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get computed features for ML training."""
        query = """
            SELECT * FROM features
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        return self.conn.execute(query, params).df()

    def get_news_signals(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get news signals, optionally filtered by symbol."""
        query = "SELECT * FROM news_signals WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        return self.conn.execute(query, params).df()

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {**self.stats}

        # Get row counts
        for table in ["tickers", "trades", "order_book_snapshots", "ohlcv", "features", "news_signals", "whale_transactions"]:
            try:
                result = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats[f"{table}_count"] = result[0] if result else 0
            except Exception:
                stats[f"{table}_count"] = 0

        # Queue sizes
        stats["ticker_queue_size"] = len(self.ticker_queue)
        stats["trade_queue_size"] = len(self.trade_queue)
        stats["orderbook_queue_size"] = len(self.orderbook_queue)
        stats["candle_queue_size"] = len(self.candle_queue)
        stats["feature_queue_size"] = len(self.feature_queue)
        stats["news_signal_queue_size"] = len(self.news_signal_queue)

        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()
        logger.info("DuckDB connection closed")


if __name__ == "__main__":
    # Test the database manager
    import asyncio

    async def test():
        db = DuckDBManager("data/test_pythia.duckdb")

        # Start batch writer
        await db.start_batch_writer()

        # Add some test data
        for i in range(10):
            db.queue_ticker("BTC-USD", {
                "timestamp": datetime.now(),
                "best_bid": 45000.0 + i,
                "best_ask": 45001.0 + i,
                "price": 45000.5 + i,
                "volume_24h": 1000000.0,
            })

            db.queue_trade("BTC-USD", {
                "timestamp": datetime.now(),
                "trade_id": f"trade_{i}",
                "price": 45000.0 + i,
                "size": 0.1,
                "side": "BUY" if i % 2 == 0 else "SELL",
            })

            await asyncio.sleep(0.1)

        # Wait for batch to flush
        await asyncio.sleep(6)

        # Stop batch writer
        await db.stop_batch_writer()

        # Check statistics
        stats = db.get_statistics()
        print("\nDatabase Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Query data
        trades = db.get_recent_trades("BTC-USD", limit=5)
        print(f"\nRecent Trades:\n{trades}")

        db.close()

    asyncio.run(test())
