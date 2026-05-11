#!/usr/bin/env python3
"""
Pythia Integrated Data Collector - Phases 1 & 2

Combines WebSocket data ingestion (Phase 1) with real-time feature engineering (Phase 2).

Pipeline:
  WebSocket → Trades → OHLCV Aggregation → Feature Calculation → Database

This is the complete end-to-end data collection and feature engineering system.

Usage:
    python -m src.data_ingestion.integrated_collector
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, List, Set
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateutil_parser
import aiohttp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from dotenv import load_dotenv

from src.data_ingestion.coinbase_auth import CoinbaseAuth
from src.data_ingestion.database import DuckDBManager
from src.data_ingestion.websocket_manager import CoinbaseWebSocketManager
from src.data_ingestion.rest_trade_fetcher import RestTradeFetcher
from src.data_ingestion.rest_orderbook_fetcher import RestOrderBookFetcher
from src.features.feature_engine import FeatureEngine
from src.features.whale_signals import WhaleTransaction
from src.features.volume_scanner import VolumeScanner
from src.strategies.loading_scanner import LoadingScanner
from src.strategies.elite_mover_tracker import elite_mover_loop
from src.signals.symbol_mapper import SymbolMapper
from src.signals.news_monitor import NewsMonitor
from src.signals.sources.exchange_listings import ExchangeListingsSource
from src.signals.sources.whale_alert import WhaleAlertSource
from src.signals.sources.twitter_rss import TwitterRSSSource
from src.signals.sources.reddit import RedditSource
from src.signals.catalyst_detector import CatalystDetector, CatalystSignal
from src.utils.config import get_config
from src.monitoring.logger import setup_logging


class IntegratedCollector:
    """
    Integrated data collection and feature engineering system.

    Combines:
    - Phase 1: WebSocket connectivity, order book tracking, database storage
    - Phase 2: OHLCV aggregation, feature calculation, multi-timeframe analysis

    This is the complete pipeline from raw market data to ML-ready features.
    """

    def __init__(self):
        """Initialize integrated collector."""
        # Load environment variables
        load_dotenv()

        # Load configuration
        self.config = get_config()

        # Setup logging
        setup_logging(
            log_file=str(self.config.get_log_path()),
            log_level=self.config.env.log_level,
            rotation=self.config.config.monitoring.log_rotation,
            retention=self.config.config.monitoring.log_retention,
            structured=self.config.config.monitoring.structured_logging
        )

        logger.info("=" * 80)
        logger.info("INITIALIZING PYTHIA INTEGRATED COLLECTOR (PHASES 1 & 2)")
        logger.info("=" * 80)

        # Initialize components
        self.auth: Optional[CoinbaseAuth] = None
        self.db_manager: Optional[DuckDBManager] = None
        self.feature_engine: Optional[FeatureEngine] = None
        self.ws_manager: Optional[CoinbaseWebSocketManager] = None

        # System state
        self.running = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _filter_pairs(self, pairs: list) -> list:
        """Filter out stablecoins and high-priced coins not useful for spike detection."""
        import requests

        # Stablecoins - no point tracking, they don't spike
        STABLECOINS = {
            'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'USDP', 'GUSD', 'FRAX',
            'LUSD', 'SUSD', 'EURC', 'PYUSD', 'FDUSD', 'USDY', 'PAX', 'CUSD'
        }

        # High-priced coins (>$200) - rarely spike 50%+ in short periods
        MAX_PRICE = 200.0

        filtered = []
        excluded_stable = []
        excluded_price = []

        # Get all product prices in a single API call
        prices = {}
        try:
            r = requests.get(
                "https://api.exchange.coinbase.com/products",
                timeout=10
            )
            if r.status_code == 200:
                for product in r.json():
                    product_id = product.get('id', '')
                    price_str = product.get('price')
                    if price_str and product_id.endswith('-USD'):
                        try:
                            prices[product_id] = float(price_str)
                        except (ValueError, TypeError):
                            pass
                logger.info(f"Fetched prices for {len(prices)} products in single API call")
        except Exception as e:
            logger.warning(f"Could not fetch bulk prices: {e}")

        for pair in pairs:
            base = pair.replace('-USD', '')

            # Skip stablecoins
            if base in STABLECOINS:
                excluded_stable.append(pair)
                continue

            # Check price from bulk data (or keep if no price data)
            price = prices.get(pair, 0)
            if price > MAX_PRICE:
                excluded_price.append((pair, price))
                continue
            filtered.append(pair)

        logger.info(f"Pair filtering: {len(pairs)} → {len(filtered)} pairs")
        logger.info(f"  Excluded stablecoins: {len(excluded_stable)} {excluded_stable}")
        logger.info(f"  Excluded high-priced (>${MAX_PRICE}): {len(excluded_price)}")
        for p, price in excluded_price[:5]:
            logger.info(f"    {p}: ${price:,.0f}")

        return filtered

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

        if self.ws_manager:
            asyncio.create_task(self.ws_manager.stop())

    async def initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing components...")

        # Initialize authentication (optional for public market data)
        try:
            self.auth = CoinbaseAuth.from_env()
            logger.success("✓ Coinbase authentication initialized")

        except Exception as e:
            logger.warning(f"Authentication not available: {e}")
            logger.info("✓ Running in UNAUTHENTICATED mode (public market data only)")
            self.auth = None

        # Initialize database
        try:
            db_path = str(self.config.get_database_path())
            self.db_manager = DuckDBManager(
                db_path=db_path,
                batch_size=self.config.config.database.batch_size,
                batch_timeout_seconds=self.config.config.database.batch_timeout_seconds
            )
            logger.success(f"✓ Database initialized at {db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

        # Initialize feature engine (PHASE 2)
        try:
            all_pairs = self.config.get_all_pairs()

            # Filter out stablecoins and high-priced coins (not useful for spike detection)
            all_pairs = self._filter_pairs(all_pairs)

            self.feature_engine = FeatureEngine(
                db_manager=self.db_manager,
                symbols=all_pairs,
                primary_timeframe='1m',  # XGBoost models trained on 1m features
                timeframes=['1m', '5m', '15m'],
                lookback_bars=self.config.config.features.lookback['indicators'],
            )
            logger.success(f"✓ Feature engine initialized (3 timeframes, {len(all_pairs)} pairs, V1+V3 models)")

        except Exception as e:
            logger.error(f"Failed to initialize feature engine: {e}")
            raise

        # Initialize WebSocket manager with feature engine integration
        try:
            self.ws_manager = IntegratedWebSocketManager(
                auth=self.auth,
                db_manager=self.db_manager,
                feature_engine=self.feature_engine,
                symbols=all_pairs,
                channels=self.config.config.websocket.channels
            )
            logger.success(f"✓ WebSocket manager initialized for {len(all_pairs)} pairs")

        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {e}")
            raise

        logger.success("All components initialized successfully")

    async def start(self):
        """Start the integrated collection system."""
        self._display_startup_info()

        self.running = True

        try:
            # Initialize components
            await self.initialize_components()

            # Start WebSocket manager (includes feature engine)
            logger.info("Starting integrated data collection & feature engineering...")
            await self.ws_manager.start()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        except Exception as e:
            logger.error(f"System error: {e}")
            logger.exception("Full traceback:")

        finally:
            await self.stop()

    async def stop(self):
        """Stop the system gracefully."""
        logger.info("=" * 80)
        logger.info("STOPPING INTEGRATED COLLECTOR")
        logger.info("=" * 80)

        self.running = False

        # Stop WebSocket manager
        if self.ws_manager:
            logger.info("Stopping WebSocket manager...")
            await self.ws_manager.stop()

        # Display final statistics
        self._display_final_statistics()

        logger.success("Integrated collector stopped gracefully")

    def _display_startup_info(self):
        """Display startup information."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("PYTHIA INTEGRATED COLLECTOR - PHASES 1 & 2")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Configuration:")
        logger.info(f"  Environment:      {self.config.config.system.environment}")
        logger.info(f"  Paper Trading:    {self.config.is_paper_trading()}")
        logger.info(f"  Database:         {self.config.get_database_path()}")
        logger.info("")
        logger.info("Trading Pairs:")
        logger.info(f"  Context Pairs:    {len(self.config.config.trading.context_pairs)} (BTC, ETH)")
        logger.info(f"  Target Pairs:     {len(self.config.config.trading.target_pairs)} altcoins")
        logger.info(f"  Total Monitored:  {len(self.config.get_all_pairs())}")
        logger.info("")
        logger.info("PHASE 1 - Data Ingestion:")
        logger.info("  ✓ ES256 JWT authentication (90s refresh)")
        logger.info("  ✓ WebSocket channels: level2, market_trades, ticker, heartbeats")
        logger.info("  ✓ Order book tracking (L=5 depth, snapshot/delta)")
        logger.info("  ✓ Batch database writing (configurable intervals)")
        logger.info("")
        logger.info("PHASE 2 - Feature Engineering:")
        logger.info("  ✓ OHLCV aggregation (1m, 5m, 15m timeframes)")
        logger.info("  ✓ Microstructure: Roll measure (MDA 0.058), VPIN, imbalance")
        logger.info("  ✓ Volume: OBV, VROC, spike detection (>2x avg)")
        logger.info("  ✓ Price: RSI, VWAP±σ, ATR, Bollinger Bands")
        logger.info("  ✓ Rolling windows: 100 bars (8.3 hours @ 5m)")
        logger.info("  ✓ ~30-40 features per bar (after Boruta selection)")
        logger.info("")
        logger.info("=" * 80)
        logger.info("")

    def _display_final_statistics(self):
        """Display final statistics."""
        if not self.ws_manager:
            return

        stats = self.ws_manager.get_statistics()

        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 80)
        logger.info("")
        logger.info("PHASE 1 - WebSocket & Database:")
        logger.info(f"  Messages Processed:     {stats.get('message_count', 0):,}")
        logger.info(f"  Reconnection Attempts:  {stats.get('reconnect_attempts', 0)}")
        logger.info(f"  Order Books Tracked:    {stats.get('order_books', 0)}")

        db_stats = stats.get("database_stats", {})
        logger.info(f"  Tickers Written:        {db_stats.get('tickers_written', 0):,}")
        logger.info(f"  Trades Written:         {db_stats.get('trades_written', 0):,}")
        logger.info(f"  Orderbooks Written:     {db_stats.get('orderbooks_written', 0):,}")

        logger.info("")
        logger.info("PHASE 2 - Feature Engineering:")

        feature_stats = stats.get("feature_stats", {})
        logger.info(f"  Trades Processed:       {feature_stats.get('trades_processed', 0):,}")
        logger.info(f"  Candles Completed:      {feature_stats.get('candles_completed', 0):,}")
        logger.info(f"  Features Calculated:    {feature_stats.get('features_calculated', 0):,}")
        logger.info(f"  Active Buffers:         {feature_stats.get('buffers_active', 0)}")
        logger.info(f"  Features Written:       {db_stats.get('features_written', 0):,}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("")


class IntegratedWebSocketManager(CoinbaseWebSocketManager):
    """
    Extended WebSocket manager with feature engine integration.

    Adds trade processing through feature engine on top of base WebSocket functionality.
    Uses an async queue to decouple WebSocket message receiving from heavy processing,
    preventing ping timeouts.
    """

    def __init__(
        self,
        auth,
        db_manager,
        feature_engine: FeatureEngine,
        symbols,
        channels
    ):
        """
        Initialize integrated WebSocket manager.

        Args:
            auth: CoinbaseAuth instance
            db_manager: DuckDBManager instance
            feature_engine: FeatureEngine instance
            symbols: Trading pair symbols
            channels: WebSocket channels
        """
        super().__init__(auth, db_manager, symbols, channels)

        self.feature_engine = feature_engine

        # Connect order book manager to feature engine for live spread/depth features
        # The orderbook_manager is created in CoinbaseWebSocketManager.__init__
        if hasattr(self, 'orderbook_manager') and self.orderbook_manager:
            self.feature_engine.set_order_book_manager(self.orderbook_manager)
            logger.info("Order book manager connected to feature engine")

        # Async queue to decouple WebSocket receiving from processing
        # This prevents ping timeouts by allowing WebSocket to stay responsive
        self._trade_queue: asyncio.Queue = asyncio.Queue(maxsize=100000)
        self._processor_task: Optional[asyncio.Task] = None
        self._periodic_prediction_task: Optional[asyncio.Task] = None
        self._queue_stats = {
            'enqueued': 0,
            'processed': 0,
            'dropped': 0,
            'max_size_seen': 0,
            'ticker_trades': 0,  # Synthetic trades from ticker
        }

        # Track symbols with recent market_trades data (last 2 minutes)
        # Symbols NOT in this set will get synthetic trades from ticker
        self._symbols_with_trades: Dict[str, datetime] = {}
        self._trade_inactivity_threshold = 120  # seconds

        # Track last ticker price to detect actual price changes
        self._last_ticker_price: Dict[str, float] = {}

        # Track last synthetic trade time per symbol (for periodic updates)
        self._last_synthetic_trade_time: Dict[str, datetime] = {}
        self._synthetic_trade_interval = 30  # Generate synthetic trade at least every 30 seconds

        # REST trade fetcher for symbols with sparse websocket data (requires auth)
        self._rest_fetcher: Optional[RestTradeFetcher] = None
        self._rest_fetcher_task: Optional[asyncio.Task] = None
        if auth:
            self._rest_fetcher = RestTradeFetcher(
                auth=auth,
                symbols=symbols,
                trade_callback=self._process_rest_trade,
                fetch_interval=60,  # Fetch every 60 seconds
                inactivity_threshold=120,  # Fetch if no WS trade for 2 minutes
            )
            logger.info("REST trade fetcher initialized for sparse WS symbols")

        # REST order book fetcher for depth data (public API - no auth required)
        # This provides bid_depth and ask_depth for order_book_depth_ratio feature
        self._orderbook_fetcher: Optional[RestOrderBookFetcher] = None
        self._orderbook_fetcher_task: Optional[asyncio.Task] = None
        self._orderbook_fetcher = RestOrderBookFetcher(
            symbols=symbols,
            depth_callback=self._process_orderbook_depth,
            fetch_interval=30,  # Fetch every 30 seconds
            levels=2,  # level=2 gets top 50 bids/asks from public API
        )
        logger.info("REST order book fetcher initialized (public API, no auth required)")

        # Always enable ticker-based synthetic trades (no auth needed)
        logger.info("Ticker-to-trade synthesis enabled for symbols without market_trades")

        # Dynamic symbol discovery (checks for new Coinbase listings)
        self._symbol_discovery_task: Optional[asyncio.Task] = None
        self._known_symbols: set = set(symbols)  # Track currently monitored symbols
        self._symbol_check_interval = 5 * 60  # Check every 5 minutes (seconds)

        # DuckDB auto-offload (archives old data to LaCie daily)
        self._offload_task: Optional[asyncio.Task] = None
        self._offload_interval_hours = 24
        self._offload_keep_days = 1
        self._offload_lacie_dir = "/Volumes/LaCie/Pythia_Archives/auto_offloads"

        # Fast-poll task for watch mode symbols (C4 reactive)
        self._fast_poll_task: Optional[asyncio.Task] = None
        self._fast_poll_interval = 5  # seconds (vs 30s for regular L2 polling)

        # Volume explosion scanner (detects abnormal 24h volume vs 30-day avg)
        self._volume_scanner: Optional[VolumeScanner] = None
        self._volume_scanner_task: Optional[asyncio.Task] = None
        self._volume_scanner = VolumeScanner(
            symbols=symbols,
            signal_callback=self._process_volume_signal,
            volume_threshold=3.0,  # 3x normal volume triggers signal
            min_volume_usd=100_000,  # Only track coins with $100K+ daily volume
            scan_interval=120,  # Scan every 2 minutes for faster detection
        )
        logger.info("Volume scanner initialized (3x threshold, 2min interval)")

        # Loading scanner (Stage 1 spike detection — pre-trigger loading patterns)
        self._loading_scanner: Optional[LoadingScanner] = None
        self._loading_scanner_task: Optional[asyncio.Task] = None
        self._loading_scanner = LoadingScanner(
            scan_interval=60,          # Every 1 minute
            loading_threshold=9.0,     # Score threshold (tuned: 5.0→42, 7.0→26, 9.0→~5-6 with bot penalty)
            trigger_pct=8.0,           # 8% move triggers Stage 2 (was 5% — too many P2 stops)
            confirm_minutes=15,        # 15min extension check (76% of winners pass, 26% of fizzles)
            callback=self._process_loading_alert,
        )
        logger.info("Loading scanner initialized (1min interval, threshold=7.0)")

        # News monitoring system (detects listings, whale moves, partnerships)
        self._symbol_mapper: Optional[SymbolMapper] = None
        self._news_monitor: Optional[NewsMonitor] = None
        self._news_monitor_task: Optional[asyncio.Task] = None
        self._whale_alert_source: Optional[WhaleAlertSource] = None

        # Initialize symbol mapper with tradeable symbols
        self._symbol_mapper = SymbolMapper(
            tradeable_symbols=symbols,
            use_coingecko=True,  # Fetch comprehensive name mappings
        )

        # Initialize news sources
        self._whale_alert_source = WhaleAlertSource()  # Keep reference for WebSocket management

        # Hook up whale alerts to feature engine buffer for whale-derived features
        if self._whale_alert_source:
            self._whale_alert_source.set_alert_callback(self._process_whale_for_features)
            logger.info("Whale alert callback set for feature engine whale buffer")

        news_sources = [
            ExchangeListingsSource(),
            self._whale_alert_source,
            # TwitterRSSSource(),  # Disabled - Nitter instances are unreliable
            RedditSource(),
        ]

        # Initialize news monitor
        self._news_monitor = NewsMonitor(
            symbol_mapper=self._symbol_mapper,
            sources=news_sources,
            signal_callback=self._process_news_signal,
            scan_interval=30,  # Scan every 30 seconds
            min_confidence=0.5,  # Filter low-quality signals
        )
        logger.info(f"News monitor initialized with {len(news_sources)} sources")

        # Initialize catalyst detector (aggregates signals with smart scoring)
        # Note: CryptoPanic disabled (paid only), using Twitter RSS + whale alerts
        self._catalyst_detector: Optional[CatalystDetector] = None
        self._catalyst_detector_task: Optional[asyncio.Task] = None
        self._catalyst_detector = CatalystDetector(
            enable_cryptopanic=False,  # Paid only now
            enable_calendar=True,
            enable_exchange_listings=True,
            enable_twitter=True,
            enable_whale_alert=False,  # Already handled by our whale_alert_source
            poll_interval_seconds=60,
        )
        # Register callback for high-priority signals
        self._catalyst_detector.register_callback(self._process_catalyst_signal)
        logger.info("Catalyst detector initialized (aggregates and scores signals)")

        logger.info("IntegratedWebSocketManager initialized with feature engine and async queue")

    async def start(self):
        """Override to start the trade processor task."""
        # Load historical data to pre-fill OHLCV buffers (reduces warmup from hours to seconds)
        await self.feature_engine.load_historical_data(lookback_minutes=500)

        # Backfill from API for symbols without sufficient DB data (zero-warmup)
        # This fetches candles directly from Coinbase for cold starts / new symbols
        symbols_needing_backfill = [
            s for s in self.feature_engine.symbols
            if s not in self.feature_engine.ohlcv_buffers
            or '1m' not in self.feature_engine.ohlcv_buffers.get(s, {})
            or len(self.feature_engine.ohlcv_buffers[s].get('1m', [])) < 50
        ]
        if symbols_needing_backfill:
            logger.info(f"[BACKFILL] {len(symbols_needing_backfill)} symbols need API backfill")
            await self.feature_engine.backfill_candles_from_api(symbols=symbols_needing_backfill, count=60)

        # Load normalization stats for zero-warmup paper trading
        # Priority: 1) Feature buffer (SQLite), 2) DB features, 3) Warmup
        buffer_loaded = self.feature_engine.load_features_from_buffer()
        if buffer_loaded == 0:
            # Fallback to DB if buffer is empty (first run or buffer cleared)
            await self.feature_engine.load_normalization_stats(lookback_minutes=60)

        # Start the background trade processor
        self._processor_task = asyncio.create_task(self._trade_processor_loop())
        logger.info("Trade processor task started")

        # DISABLED: Prediction loop not needed - paper trader uses catalyst signals
        # The V3 XGBoost predictions were causing Metal/GPU crashes and aren't used
        # self._periodic_prediction_task = asyncio.create_task(self._periodic_prediction_loop())
        # logger.info("Periodic prediction task started")
        logger.info("Periodic prediction loop DISABLED (using catalyst signals instead)")

        # Start REST trade fetcher for symbols with sparse WS data
        if self._rest_fetcher:
            self._rest_fetcher_task = asyncio.create_task(self._rest_fetcher.start())
            logger.info("REST trade fetcher task started")

        # Start REST order book fetcher for depth data
        if self._orderbook_fetcher:
            self._orderbook_fetcher_task = asyncio.create_task(self._orderbook_fetcher.start())
            logger.info("REST order book fetcher task started")

        # Start dynamic symbol discovery (checks for new Coinbase listings frequently)
        self._symbol_discovery_task = asyncio.create_task(self._symbol_discovery_loop())
        logger.info("Symbol discovery task started (checks every 5 min)")

        # Start fast-poll task for watch mode symbols
        self._fast_poll_task = asyncio.create_task(self._fast_poll_watched_symbols())
        logger.info("Fast-poll task started for watch mode symbols (every 5s)")

        # Start volume explosion scanner
        if self._volume_scanner:
            self._volume_scanner_task = asyncio.create_task(self._volume_scanner.start())
            logger.info("Volume scanner task started (every 5min)")

        # Start news monitor
        if self._news_monitor:
            # Initialize symbol mapper async resources (CoinGecko)
            if self._symbol_mapper:
                await self._symbol_mapper.initialize()
            self._news_monitor_task = asyncio.create_task(self._news_monitor.start())
            logger.info("News monitor task started (every 30s)")

        # Start catalyst detector (aggregates and scores signals from all sources)
        if self._catalyst_detector:
            await self._catalyst_detector.start()
            logger.info("Catalyst detector started (ENTER/PREPARE signals exported to live_signals.json)")

        # Start Whale Alert WebSocket (real-time alerts)
        if self._whale_alert_source and self._whale_alert_source.api_key:
            await self._whale_alert_source.start_websocket()
            logger.info("Whale Alert WebSocket started (real-time alerts)")

        # Start loading scanner (Stage 1 spike detection)
        if self._loading_scanner:
            self._loading_scanner_task = asyncio.create_task(self._loading_scanner.start())
            logger.info("Loading scanner task started (every 60s)")

        # Start DuckDB auto-offload task
        self._offload_task = asyncio.create_task(self._offload_loop())
        logger.info(f"DuckDB auto-offload task started (every {self._offload_interval_hours}h, keep {self._offload_keep_days}d)")

        # Start elite mover tracker (detects 50%+ movers, updates elite_movers.duckdb hourly)
        self._elite_mover_task = asyncio.create_task(elite_mover_loop())
        logger.info("Elite mover tracker started (every 1h)")

        # Call parent start
        await super().start()

    async def stop(self):
        """Override to stop the trade processor task."""
        # Stop REST trade fetcher
        if self._rest_fetcher:
            await self._rest_fetcher.stop()
        if self._rest_fetcher_task and not self._rest_fetcher_task.done():
            self._rest_fetcher_task.cancel()
            try:
                await self._rest_fetcher_task
            except asyncio.CancelledError:
                pass
            logger.info("REST trade fetcher stopped")

        # Stop REST order book fetcher
        if self._orderbook_fetcher:
            await self._orderbook_fetcher.stop()
        if self._orderbook_fetcher_task and not self._orderbook_fetcher_task.done():
            self._orderbook_fetcher_task.cancel()
            try:
                await self._orderbook_fetcher_task
            except asyncio.CancelledError:
                pass
            logger.info("REST order book fetcher stopped")

        # Cancel processor task
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            logger.info(f"Trade processor stopped. Stats: {self._queue_stats}")

        # Cancel periodic prediction task
        if self._periodic_prediction_task and not self._periodic_prediction_task.done():
            self._periodic_prediction_task.cancel()
            try:
                await self._periodic_prediction_task
            except asyncio.CancelledError:
                pass
            logger.info("Periodic prediction task stopped")

        # Cancel symbol discovery task
        if self._symbol_discovery_task and not self._symbol_discovery_task.done():
            self._symbol_discovery_task.cancel()
            try:
                await self._symbol_discovery_task
            except asyncio.CancelledError:
                pass
            logger.info("Symbol discovery task stopped")

        # Cancel loading scanner task
        if self._loading_scanner:
            self._loading_scanner.stop()
        if self._loading_scanner_task and not self._loading_scanner_task.done():
            self._loading_scanner_task.cancel()
            try:
                await self._loading_scanner_task
            except asyncio.CancelledError:
                pass
            logger.info("Loading scanner task stopped")

        # Cancel offload task
        if self._offload_task and not self._offload_task.done():
            self._offload_task.cancel()
            try:
                await self._offload_task
            except asyncio.CancelledError:
                pass
            logger.info("DuckDB offload task stopped")

        # Cancel fast-poll task
        if self._fast_poll_task and not self._fast_poll_task.done():
            self._fast_poll_task.cancel()
            try:
                await self._fast_poll_task
            except asyncio.CancelledError:
                pass
            logger.info("Fast-poll task stopped")

        # Stop volume scanner
        if self._volume_scanner:
            await self._volume_scanner.stop()
        if self._volume_scanner_task and not self._volume_scanner_task.done():
            self._volume_scanner_task.cancel()
            try:
                await self._volume_scanner_task
            except asyncio.CancelledError:
                pass
            logger.info("Volume scanner stopped")

        # Stop Whale Alert WebSocket
        if self._whale_alert_source:
            await self._whale_alert_source.stop_websocket()
            logger.info("Whale Alert WebSocket stopped")

        # Stop news monitor
        if self._news_monitor:
            await self._news_monitor.stop()
        if self._news_monitor_task and not self._news_monitor_task.done():
            self._news_monitor_task.cancel()
            try:
                await self._news_monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("News monitor stopped")

        # Stop catalyst detector
        if self._catalyst_detector:
            await self._catalyst_detector.stop()
            logger.info("Catalyst detector stopped")

        # Call parent stop
        await super().stop()

    def _symbol_needs_ticker_trade(self, symbol: str) -> bool:
        """
        Check if a symbol needs synthetic trades from ticker data.

        Returns True if the symbol hasn't received market_trades in the
        inactivity threshold period.
        """
        last_trade = self._symbols_with_trades.get(symbol)
        if last_trade is None:
            return True  # Never received a trade

        elapsed = (datetime.now(timezone.utc) - last_trade).total_seconds()
        return elapsed > self._trade_inactivity_threshold

    async def _handle_ticker(self, data: dict):
        """
        Override to update feature engine with bid/ask data from ticker.

        Ticker messages include best_bid and best_ask which we use to
        calculate bid_ask_spread_pct for the model.

        Also generates synthetic trades for symbols that aren't receiving
        market_trades data from the websocket.
        """
        # Call parent method for database writing
        await super()._handle_ticker(data)

        # Update feature engine's order book cache with bid/ask from ticker
        events = data.get("events", [])
        for event in events:
            tickers = event.get("tickers", [])
            for ticker in tickers:
                symbol = ticker.get("product_id")
                if not symbol:
                    continue

                best_bid = ticker.get("best_bid")
                best_ask = ticker.get("best_ask")
                price = ticker.get("price")

                if best_bid and best_ask:
                    try:
                        # Get existing cache to preserve REST depth data
                        existing = self.feature_engine._order_book_cache.get(symbol, {})

                        # Get depth from Level 2 if available, otherwise preserve existing
                        # This prevents the ticker handler from overwriting REST depth with zeros
                        bid_depth = existing.get('bid_depth', 0.0)
                        ask_depth = existing.get('ask_depth', 0.0)

                        if self.orderbook_manager:
                            order_book = self.orderbook_manager.get_book(symbol)
                            if order_book and order_book.is_synchronized:
                                # Get top 10 levels of depth
                                bid_levels, ask_levels = order_book.get_depth(levels=10)
                                bid_depth = sum(level.quantity for level in bid_levels)
                                ask_depth = sum(level.quantity for level in ask_levels)

                        # Update feature engine cache (preserves REST depth if L2 not available)
                        self.feature_engine.update_order_book_cache(symbol, {
                            'best_bid': float(best_bid),
                            'best_ask': float(best_ask),
                            'bid_depth': bid_depth,
                            'ask_depth': ask_depth,
                        })
                    except (ValueError, TypeError):
                        pass

                # Generate synthetic trade for symbols without market_trades
                if price and self._symbol_needs_ticker_trade(symbol):
                    try:
                        current_price = float(price)
                        last_price = self._last_ticker_price.get(symbol)
                        now = datetime.now(timezone.utc)

                        # Check if enough time has passed since last synthetic trade
                        last_synth_time = self._last_synthetic_trade_time.get(symbol)
                        time_elapsed = (now - last_synth_time).total_seconds() if last_synth_time else float('inf')

                        # Generate trade if: price changed OR enough time has passed
                        price_changed = last_price is None or abs(current_price - last_price) > 1e-10
                        time_to_generate = time_elapsed >= self._synthetic_trade_interval

                        if price_changed or time_to_generate:
                            self._last_ticker_price[symbol] = current_price
                            self._last_synthetic_trade_time[symbol] = now

                            # Create synthetic trade message
                            synthetic_trade = {
                                "events": [{
                                    "product_id": symbol,
                                    "trades": [{
                                        "price": str(current_price),
                                        "size": "0.001",  # Minimal size
                                        "side": "BUY" if not last_price or current_price >= last_price else "SELL",
                                        "time": now.isoformat()
                                    }]
                                }]
                            }

                            # Enqueue for feature processing
                            try:
                                self._trade_queue.put_nowait(synthetic_trade)
                                self._queue_stats['enqueued'] += 1
                                self._queue_stats['ticker_trades'] += 1
                            except asyncio.QueueFull:
                                pass  # Don't log, queue full is handled elsewhere

                    except (ValueError, TypeError):
                        pass

    async def _handle_market_trades(self, data: dict):
        """
        Override to add feature engine processing via async queue.

        The WebSocket message loop just enqueues trades and returns immediately,
        keeping the connection responsive to pings.
        """
        # Call parent method for database writing (fast operation)
        await super()._handle_market_trades(data)

        # Record that this symbol is receiving real market_trades
        # This prevents synthetic trades from ticker for active symbols
        # Note: product_id is in each trade object, NOT at the event level
        events = data.get("events", [])
        now = datetime.now(timezone.utc)
        for event in events:
            trades = event.get("trades", [])
            for trade in trades:
                symbol = trade.get("product_id")
                if symbol:
                    self._symbols_with_trades[symbol] = now
                    # Also tell REST fetcher if available
                    if self._rest_fetcher:
                        self._rest_fetcher.record_ws_trade(symbol)

        # Enqueue for feature engine processing (non-blocking)
        try:
            self._trade_queue.put_nowait(data)
            self._queue_stats['enqueued'] += 1

            # Track queue size
            current_size = self._trade_queue.qsize()
            if current_size > self._queue_stats['max_size_seen']:
                self._queue_stats['max_size_seen'] = current_size

            # Warn if queue is getting full
            if current_size > 5000 and current_size % 1000 == 0:
                logger.warning(f"Trade queue size: {current_size}/10000")

        except asyncio.QueueFull:
            self._queue_stats['dropped'] += 1
            if self._queue_stats['dropped'] % 100 == 1:
                logger.warning(f"Trade queue full, dropped {self._queue_stats['dropped']} messages")

    async def _trade_processor_loop(self):
        """
        Background task that processes trades from the queue.

        This runs independently of the WebSocket message loop, so heavy
        feature calculation doesn't block ping/pong handling.

        Uses batch processing with asyncio.gather() to process multiple
        trades in parallel, preventing alphabetical bias from sequential processing.
        """
        from datetime import datetime

        logger.info("Trade processor loop started (batch parallel processing)")

        BATCH_SIZE = 200  # Process up to 200 trades in parallel
        BATCH_TIMEOUT = 0.05  # Wait up to 50ms to collect a batch (faster processing)

        while True:
            try:
                # Collect a batch of trades from the queue
                batch = []

                # Wait for first trade
                data = await self._trade_queue.get()
                batch.append(data)

                # Try to get more trades without blocking (up to BATCH_SIZE)
                while len(batch) < BATCH_SIZE:
                    try:
                        data = await asyncio.wait_for(
                            self._trade_queue.get(),
                            timeout=BATCH_TIMEOUT
                        )
                        batch.append(data)
                    except asyncio.TimeoutError:
                        break  # No more trades waiting, process what we have

                # Create tasks for all trades in the batch
                tasks = []
                for data in batch:
                    events = data.get("events", [])
                    for event in events:
                        event_symbol = event.get("product_id")
                        trades = event.get("trades", [])

                        for trade in trades:
                            try:
                                symbol = event_symbol or trade.get("product_id")
                                price = float(trade.get("price", 0))
                                size = float(trade.get("size", 0))
                                side = trade.get("side", "").upper()
                                time_str = trade.get("time")

                                if time_str:
                                    timestamp = dateutil_parser.isoparse(time_str)
                                else:
                                    timestamp = datetime.now(timezone.utc)

                                # Create task (don't await yet)
                                task = self.feature_engine.process_trade(
                                    symbol=symbol,
                                    price=price,
                                    size=size,
                                    side=side,
                                    timestamp=timestamp
                                )
                                tasks.append(task)

                            except Exception as e:
                                logger.error(f"Error preparing trade for processing: {e}")

                # Process all trades in parallel
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Write trades to SQLite for Accumulation Hunter volume calculations
                if self.feature_engine and hasattr(self.feature_engine, 'feature_buffer'):
                    try:
                        trades_batch = []
                        for msg_data in batch:
                            # Extract trades from nested structure (events -> trades)
                            events = msg_data.get("events", [])
                            for event in events:
                                event_symbol = event.get("product_id")
                                trades = event.get("trades", [])
                                for trade in trades:
                                    symbol = event_symbol or trade.get("product_id")
                                    price = float(trade.get("price", 0))
                                    size = float(trade.get("size", 0))
                                    side = trade.get("side", "").upper()
                                    time_str = trade.get("time")
                                    if time_str:
                                        timestamp = dateutil_parser.isoparse(time_str)
                                    else:
                                        timestamp = datetime.now(timezone.utc)
                                    if symbol and price and size:
                                        trades_batch.append((symbol, timestamp, price, size, side))
                        if trades_batch:
                            self.feature_engine.feature_buffer.write_trades_batch(trades_batch)
                    except Exception as e:
                        logger.debug(f"Error writing trades to SQLite: {e}")

                self._queue_stats['processed'] += len(batch)

                # Log stats periodically
                if self._queue_stats['processed'] % 1000 == 0:
                    logger.info(
                        f"[QUEUE] Processed: {self._queue_stats['processed']} | "
                        f"Pending: {self._trade_queue.qsize()} | "
                        f"Max seen: {self._queue_stats['max_size_seen']} | "
                        f"Dropped: {self._queue_stats['dropped']} | "
                        f"Batch: {len(batch)}"
                    )

            except asyncio.CancelledError:
                logger.info("Trade processor loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trade processor loop: {e}")

    async def _periodic_prediction_loop(self):
        """
        Run predictions for all symbols every 60 seconds.

        This ensures all symbols get predictions, not just those
        with recent trades.
        """
        logger.info("Periodic prediction loop started")

        while True:
            try:
                # Wait 60 seconds between prediction runs
                await asyncio.sleep(60)

                # Run predictions for all eligible symbols
                symbols_processed = await self.feature_engine.run_all_predictions()

                logger.info(f"[PERIODIC] Completed prediction cycle: {symbols_processed} symbols")

            except asyncio.CancelledError:
                logger.info("Periodic prediction loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in periodic prediction loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _fast_poll_watched_symbols(self):
        """
        Poll watched symbols every 5 seconds for C4 reactive signals.

        When watch_mode has symbols flagged (due to V3 > 70%, BAR > 2x, etc.),
        this task fetches their L2 data more frequently than the regular
        30-second cycle, giving C4 more opportunities to catch brief signal alignments.
        """
        from datetime import datetime, timezone, timedelta

        logger.info("Fast-poll task started for watch mode symbols")

        while True:
            try:
                # Get currently watched symbols
                watched = self.feature_engine.watch_mode.get_watched_symbols()

                if watched:
                    logger.debug(f"[FAST_POLL] Polling {len(watched)} watched symbols: {watched}")

                    for symbol in watched[:3]:  # Max 3 symbols to avoid rate limits
                        try:
                            # Fetch fresh L2 data
                            if self._orderbook_fetcher:
                                await self._orderbook_fetcher.fetch_single(symbol)

                            # Check for C4 reactive signal (only for reactive strategies)
                            timestamp = datetime.now(timezone.utc)
                            async with self.feature_engine._position_lock:
                                for strategy in self.feature_engine.strategies:
                                    if strategy.entry_type == "reactive":
                                        entry_signal = self.feature_engine._check_reactive_entry(
                                            symbol, timestamp, strategy
                                        )
                                        if entry_signal:
                                            await self.feature_engine._enter_position(
                                                symbol, entry_signal, timestamp, strategy
                                            )

                        except Exception as e:
                            logger.debug(f"[FAST_POLL] Error polling {symbol}: {e}")

                # Wait 5 seconds before next poll
                await asyncio.sleep(self._fast_poll_interval)

            except asyncio.CancelledError:
                logger.info("Fast-poll task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in fast-poll loop: {e}")
                await asyncio.sleep(5)

    async def _process_rest_trade(
        self,
        symbol: str,
        price: float,
        size: float,
        side: str,
        timestamp
    ):
        """
        Process a trade received via REST API (callback for RestTradeFetcher).

        This goes directly to the feature engine, bypassing the WS queue.
        """
        try:
            await self.feature_engine.process_trade(
                symbol=symbol,
                price=price,
                size=size,
                side=side,
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Error processing REST trade for {symbol}: {e}")

    async def _process_orderbook_depth(
        self,
        symbol: str,
        bid_depth: float,
        ask_depth: float,
        best_bid: float,
        best_ask: float,
        bids: list = None,
        asks: list = None
    ):
        """
        Process order book depth from REST API (callback for RestOrderBookFetcher).

        Updates the feature engine's order book cache with depth data AND
        stores full L2 data to database for training.
        """
        try:
            # Update feature engine cache (for real-time features)
            self.feature_engine.update_order_book_cache(symbol, {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
            })

            # Store full L2 to database for training (if we have data)
            if bids and asks:
                mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None
                spread = best_ask - best_bid if best_bid and best_ask else None
                spread_bps = (spread / mid_price * 10000) if mid_price and spread else None

                from datetime import datetime
                snapshot_data = {
                    "timestamp": datetime.now(),
                    "bids": bids,  # List of (price, quantity) tuples
                    "asks": asks,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "mid_price": mid_price,
                    "spread": spread,
                    "spread_bps": spread_bps,
                    "sequence_num": None,  # REST API doesn't provide sequence
                }
                self.db_manager.queue_orderbook(symbol, snapshot_data)

                # Also write to SQLite for Accumulation Hunter (concurrent read access)
                if self.feature_engine and hasattr(self.feature_engine, 'feature_buffer'):
                    import json
                    self.feature_engine.feature_buffer.write_order_book_snapshot(
                        symbol=symbol,
                        timestamp=snapshot_data["timestamp"],
                        bids=json.dumps(bids[:10]) if bids else "[]",  # Top 10 levels
                        asks=json.dumps(asks[:10]) if asks else "[]",
                        mid_price=mid_price or 0.0
                    )

        except Exception as e:
            logger.error(f"Error processing order book depth for {symbol}: {e}")

    async def _process_volume_signal(self, signal):
        """
        Process volume explosion signal from VolumeScanner.

        This is called when a coin shows abnormal volume (e.g., 3x+ normal).
        Updates the feature engine cache so C5_volume strategy can check for entries.

        Args:
            signal: VolumeSignal dataclass with volume_multiple, price_change, etc.
        """
        try:
            # Update feature engine cache for C5_volume strategy
            self.feature_engine.update_volume_signal(
                symbol=signal.symbol,
                volume_multiple=signal.volume_multiple,
                price_change=signal.price_change_24h,
                timestamp=signal.timestamp
            )

            # Add to watch mode if price is moving up (volume + momentum)
            logger.info(f"[VOLUME_CALLBACK] {signal.symbol} price_change={signal.price_change_24h:.1%} (threshold: 5%)")
            if signal.price_change_24h > 0.05:  # Up more than 5%
                # This coin has volume explosion AND is moving up
                logger.warning(f"[SIGNAL_EXPORT] Exporting {signal.symbol} to live_signals.json")
                self.feature_engine.watch_mode.add_to_watch(
                    signal.symbol,
                    f"VOL={signal.volume_multiple:.1f}x +{signal.price_change_24h:.0%}"
                )

                # Export to JSON for paper trader
                prob = min(signal.volume_multiple / 5.0, 1.0)  # Scale 3-5x to 0.6-1.0
                self._export_signal_to_json({
                    'symbol': signal.symbol,
                    'timestamp': signal.timestamp.isoformat() if hasattr(signal.timestamp, 'isoformat') else str(signal.timestamp),
                    'event_type': 'volume_explosion',
                    'confidence': prob,
                    'event_priority': prob,
                    'pred_probability': prob,  # For paper trader compatibility
                    'volume_multiple': signal.volume_multiple,
                    'price_change': signal.price_change_24h,
                })

        except Exception as e:
            logger.error(f"Error processing volume signal for {signal.symbol}: {e}")

    async def _process_loading_alert(self, alert):
        """
        Process loading scanner alert (Stage 1 spike detection).

        Called when a symbol's loading score crosses the threshold,
        indicating potential pre-spike accumulation.
        """
        try:
            logger.warning(
                f"[LOADING ALERT] {alert.symbol} score={alert.score:.1f} "
                f"price=${alert.price_at_alert:.4f} "
                f"vol_trend={alert.components.get('vol_trend', 0):.2f} "
                f"natr={alert.components.get('natr', 0):.2f} "
                f"bb={alert.components.get('bb_width', 0):.4f}"
            )

            # Export to live_signals.json for paper trader
            self._export_signal_to_json({
                'symbol': alert.symbol,
                'timestamp': alert.timestamp.isoformat(),
                'event_type': 'loading_detected',
                'confidence': min(alert.score / 10.0, 1.0),
                'event_priority': min(alert.score / 10.0, 1.0),
                'pred_probability': min(alert.score / 10.0, 1.0),
                'loading_score': alert.score,
                'components': {k: round(v, 4) for k, v in alert.components.items()},
            })

        except Exception as e:
            logger.error(f"Error processing loading alert for {alert.symbol}: {e}")

    async def _process_whale_for_features(self, item):
        """
        Feed raw whale alerts to feature engine buffer.

        This is called by WhaleAlertSource for every whale transaction.
        Converts the NewsItem to WhaleTransaction and adds to the buffer.

        Args:
            item: NewsItem from WhaleAlertSource with raw_data containing transaction details
        """
        try:
            raw = item.raw_data
            if not raw or not raw.get('amount_usd'):
                return

            # Extract symbol - raw_data has 'symbol' with -USD suffix or just the base symbol
            raw_symbol = raw.get('symbol', '')
            if not raw_symbol or raw_symbol == 'UNKNOWN':
                # Try alert_symbol as fallback
                raw_symbol = raw.get('alert_symbol', '')
            if not raw_symbol:
                return

            # Ensure -USD suffix
            symbol = raw_symbol if raw_symbol.endswith('-USD') else f"{raw_symbol}-USD"

            tx = WhaleTransaction(
                timestamp=item.timestamp,
                symbol=symbol,
                amount_usd=raw.get('amount_usd', 0),
                subtype=raw.get('subtype', 'unknown'),
                from_name=raw.get('from_name', 'unknown'),
                to_name=raw.get('to_name', 'unknown'),
                blockchain=raw.get('blockchain', ''),
            )

            # Add to feature engine buffer (for real-time features)
            self.feature_engine.add_whale_transaction(tx)

            # Queue to database for historical analysis
            self.db_manager.queue_whale_transaction({
                'symbol': symbol,
                'timestamp': item.timestamp,
                'amount_usd': raw.get('amount_usd', 0),
                'subtype': raw.get('subtype', 'unknown'),
                'from_name': raw.get('from_name', 'unknown'),
                'to_name': raw.get('to_name', 'unknown'),
                'blockchain': raw.get('blockchain', ''),
                'tx_hash': raw.get('tx_hash', ''),
            })

            logger.debug(f"[WHALE_FEATURES] Added {symbol} ${tx.amount_usd:,.0f} {tx.subtype}")

        except Exception as e:
            logger.error(f"Error processing whale alert for features: {e}")

    async def _process_news_signal(self, signal):
        """
        Process news signal from NewsMonitor.

        This is called when a news event is detected that may affect a tradeable symbol.
        Updates the feature engine cache so news-based entry strategies can check for signals.

        Args:
            signal: NewsSignal with symbol, confidence, event_type, source, etc.
        """
        try:
            # Convert NewsSignal to dict for feature engine
            signal_dict = {
                'symbol': signal.symbol,
                'timestamp': signal.timestamp,
                'source': signal.source,
                'event_type': signal.event_type,
                'confidence': signal.confidence,
                'title': signal.title,
                'url': signal.url,
                'signal_hash': signal.signal_hash,
                'source_credibility': signal.source_credibility,
                'entity_certainty': signal.entity_certainty,
                'event_priority': signal.event_priority,
                'recency_score': signal.recency_score,
                'engagement_score': signal.engagement_score,
                'sentiment_score': signal.sentiment_score,
            }

            # Update feature engine cache for news-based entry
            self.feature_engine.update_news_signal(signal_dict)

            # Queue to database for historical analysis
            self.db_manager.queue_news_signal(signal_dict)

            # Export to JSON for paper trader (can't read DuckDB due to lock)
            self._export_signal_to_json(signal_dict)

            # Add high-confidence signals to watch mode
            if signal.confidence >= 0.7:
                event_emoji = {
                    'listing': 'L',
                    'whale_move': 'W',
                    'partnership': 'P',
                    'sentiment_spike': 'S',
                }.get(signal.event_type, 'N')

                self.feature_engine.watch_mode.add_to_watch(
                    signal.symbol,
                    f"NEWS={event_emoji}:{signal.confidence:.0%}"
                )

        except Exception as e:
            logger.error(f"Error processing news signal for {signal.symbol}: {e}")

    async def _process_catalyst_signal(self, signal: CatalystSignal):
        """
        Process signal from CatalystDetector.

        The CatalystDetector aggregates multiple sources and provides
        action recommendations (ENTER, PREPARE, WATCH, AVOID).
        Only export high-priority actionable signals.
        """
        try:
            # Only export ENTER and PREPARE signals (the actionable ones)
            if signal.action not in ('ENTER', 'PREPARE'):
                logger.debug(f"[CATALYST] Skipping {signal.symbol} ({signal.action}): {signal.headline[:50]}")
                return

            # Convert CatalystSignal to dict for export
            signal_dict = {
                'symbol': signal.symbol,
                'timestamp': signal.detected_at.isoformat(),
                'source': 'catalyst_detector',
                'event_type': signal.catalyst_type,
                'confidence': signal.confidence,
                'event_priority': signal.priority_score,
                'pred_probability': signal.priority_score,  # Paper trader compatibility
                'title': signal.headline,
                'action': signal.action,
                'urgency': signal.urgency,
                'impact_score': signal.impact_score,
                'sources': signal.sources,
                'url': signal.source_urls[0] if signal.source_urls else None,
            }

            # Export to JSON for paper trader
            self._export_signal_to_json(signal_dict)

            # Add to watch mode
            action_emoji = {'ENTER': '🚀', 'PREPARE': '⚡'}.get(signal.action, '📊')
            self.feature_engine.watch_mode.add_to_watch(
                signal.symbol,
                f"CATALYST={action_emoji}{signal.catalyst_type}:{signal.priority_score:.0%}"
            )

            logger.warning(
                f"[CATALYST] {action_emoji} {signal.action} {signal.symbol} | "
                f"type={signal.catalyst_type} priority={signal.priority_score:.2f} | "
                f"{signal.headline[:60]}"
            )

        except Exception as e:
            logger.error(f"Error processing catalyst signal: {e}")

    def _export_signal_to_json(self, signal_dict: dict):
        """
        Export signal to JSON file for paper trader.

        Maintains a rolling list of recent signals that the paper trader
        can read (avoiding DuckDB lock conflicts).
        """
        import json
        # Use absolute path to ensure it works regardless of working directory
        signals_file = Path("/Users/bz/Pythia2/data/live_signals.json")

        try:
            # Load existing signals
            if signals_file.exists():
                with open(signals_file) as f:
                    signals = json.load(f)
            else:
                signals = []

            # Convert timestamp to ISO format string
            sig = signal_dict.copy()
            if hasattr(sig.get('timestamp'), 'isoformat'):
                sig['timestamp'] = sig['timestamp'].isoformat()

            # Add new signal
            signals.append(sig)

            # Keep only last hour of signals
            cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            signals = [
                s for s in signals
                if datetime.fromisoformat(s['timestamp'].replace('Z', '+00:00')) > cutoff
            ]

            # Write back
            signals_file.parent.mkdir(parents=True, exist_ok=True)
            with open(signals_file, 'w') as f:
                json.dump(signals, f, indent=2, default=str)

            logger.info(f"[SIGNAL_EXPORT] Wrote {len(signals)} signals to {signals_file}")

        except Exception as e:
            logger.error(f"Error exporting signal to JSON: {e}")

    async def _offload_loop(self):
        """
        Periodically offload old DuckDB data to LaCie.

        Runs every _offload_interval_hours, keeps _offload_keep_days of data
        in the live database, moves everything older to a date-stamped archive
        on the LaCie drive.
        """
        import shutil
        from pathlib import Path as _Path

        # Wait a bit after startup before first offload check
        await asyncio.sleep(300)  # 5 min

        while self.running:
            try:
                lacie_dir = _Path(self._offload_lacie_dir)

                # Check if LaCie is mounted
                if not lacie_dir.parent.parent.exists():
                    logger.debug("[OFFLOAD] LaCie not mounted, skipping")
                    await asyncio.sleep(self._offload_interval_hours * 3600)
                    continue

                lacie_dir.mkdir(parents=True, exist_ok=True)

                if not self.db_manager or not self.db_manager.conn:
                    await asyncio.sleep(3600)
                    continue

                cutoff = datetime.now() - timedelta(days=self._offload_keep_days)
                cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
                date_stamp = datetime.now().strftime("%Y%m%d")
                archive_path = lacie_dir / f"pythia_offload_{date_stamp}.duckdb"

                tables = ["trades", "tickers", "order_book_snapshots",
                          "ohlcv", "features", "news_signals", "whale_transactions"]

                total_moved = 0
                for table in tables:
                    try:
                        count = self.db_manager.conn.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE timestamp < ?", [cutoff_str]
                        ).fetchone()[0]

                        if count == 0:
                            continue

                        # Create archive and copy
                        self.db_manager.conn.execute(
                            f"ATTACH '{archive_path}' AS archive"
                        )

                        # Create table in archive if needed
                        try:
                            self.db_manager.conn.execute(f"SELECT 1 FROM archive.{table} LIMIT 0")
                        except:
                            cols = self.db_manager.conn.execute(f"DESCRIBE {table}").fetchall()
                            col_defs = ", ".join(f"{c[0]} {c[1]}" for c in cols)
                            self.db_manager.conn.execute(
                                f"CREATE TABLE IF NOT EXISTS archive.{table} ({col_defs})"
                            )

                        self.db_manager.conn.execute(f"""
                            INSERT INTO archive.{table}
                            SELECT * FROM {table} WHERE timestamp < ?
                        """, [cutoff_str])

                        self.db_manager.conn.execute(
                            f"DELETE FROM {table} WHERE timestamp < ?", [cutoff_str]
                        )

                        self.db_manager.conn.execute("DETACH archive")

                        total_moved += count
                        logger.info(f"[OFFLOAD] {table}: archived {count:,} rows")

                    except Exception as e:
                        logger.error(f"[OFFLOAD] Error on {table}: {e}")
                        try:
                            self.db_manager.conn.execute("DETACH archive")
                        except:
                            pass

                if total_moved > 0:
                    self.db_manager.conn.execute("CHECKPOINT")
                    db_size = _Path(self.db_manager.db_path).stat().st_size / (1024**3)
                    disk_free = shutil.disk_usage("/").free / (1024**3)
                    logger.success(
                        f"[OFFLOAD] Archived {total_moved:,} rows to {archive_path.name}. "
                        f"Live DB: {db_size:.2f}GB, Disk free: {disk_free:.1f}GB"
                    )
                else:
                    logger.debug("[OFFLOAD] Nothing to archive")

            except Exception as e:
                logger.error(f"[OFFLOAD] Error in offload loop: {e}")

            await asyncio.sleep(self._offload_interval_hours * 3600)

    async def _symbol_discovery_loop(self):
        """
        Periodically check for new Coinbase USD listings and add them dynamically.

        Runs once per day to catch new coin listings without requiring restart.
        New coins are often the most volatile and best candidates for spike detection.
        """
        # Wait a bit before first check (let system stabilize)
        await asyncio.sleep(60)

        logger.info("Symbol discovery loop started - will check for new listings every 24h")

        while True:
            try:
                new_symbols = await self._fetch_new_symbols()

                if new_symbols:
                    logger.info(f"🆕 Found {len(new_symbols)} new Coinbase listings: {new_symbols}")
                    await self._add_new_symbols(new_symbols)
                else:
                    logger.debug("No new symbols found")

                # Wait 24 hours before next check
                await asyncio.sleep(self._symbol_check_interval)

            except asyncio.CancelledError:
                logger.info("Symbol discovery loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in symbol discovery loop: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour on error

    async def _fetch_new_symbols(self) -> List[str]:
        """
        Fetch all available USD pairs from Coinbase and find new ones.

        Returns list of symbols that are on Coinbase but not yet being monitored.
        """
        # Stablecoins to exclude
        STABLECOINS = {
            'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'USDP', 'GUSD', 'FRAX',
            'LUSD', 'SUSD', 'EURC', 'PYUSD', 'FDUSD', 'USDY', 'PAX', 'CUSD'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.exchange.coinbase.com/products",
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch Coinbase products: {response.status}")
                        return []

                    products = await response.json()

                    # Filter to online USD pairs
                    all_usd_pairs = set()
                    for p in products:
                        if (p.get('quote_currency') == 'USD' and
                            p.get('status') == 'online'):
                            symbol = p.get('id')
                            base = symbol.replace('-USD', '') if symbol else ''

                            # Skip stablecoins
                            if base not in STABLECOINS:
                                all_usd_pairs.add(symbol)

                    # Find new symbols (not in our known set)
                    new_symbols = all_usd_pairs - self._known_symbols

                    return sorted(list(new_symbols))

        except Exception as e:
            logger.error(f"Error fetching Coinbase products: {e}")
            return []

    async def _add_new_symbols(self, new_symbols: List[str]):
        """
        Dynamically add new symbols to all components without restart.

        Updates:
        - WebSocket subscriptions
        - REST order book fetcher
        - Feature engine
        - Known symbols tracking
        """
        if not new_symbols:
            return

        logger.info(f"Adding {len(new_symbols)} new symbols dynamically...")

        # Filter out high-priced coins (>$200 - unlikely to spike 50%+)
        MAX_PRICE = 200.0
        symbols_to_add = []

        async with aiohttp.ClientSession() as session:
            for symbol in new_symbols:
                try:
                    async with session.get(
                        f"https://api.exchange.coinbase.com/products/{symbol}/ticker",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            price = float(data.get('price', 0))
                            if price <= MAX_PRICE:
                                symbols_to_add.append(symbol)
                                logger.info(f"  ✓ {symbol} @ ${price:.2f}")
                            else:
                                logger.info(f"  ✗ {symbol} @ ${price:.0f} (too expensive)")
                        else:
                            # Add anyway if we can't check price
                            symbols_to_add.append(symbol)
                except Exception as e:
                    # Add anyway on error
                    symbols_to_add.append(symbol)
                    logger.debug(f"  ? {symbol} (price check failed: {e})")

        if not symbols_to_add:
            logger.info("No new symbols passed price filter")
            return

        # 1. Add to WebSocket subscriptions
        if self.websocket and self.connected:
            import json
            for channel in self.channels:
                # Subscribe in small batches
                for i in range(0, len(symbols_to_add), 5):
                    batch = symbols_to_add[i:i + 5]
                    if self.auth:
                        subscribe_msg = self.auth.get_websocket_auth_message(channel, batch)
                    else:
                        subscribe_msg = {
                            "type": "subscribe",
                            "product_ids": batch,
                            "channel": channel
                        }
                    try:
                        await self.websocket.send(json.dumps(subscribe_msg))
                        logger.debug(f"Subscribed to {channel} for {batch}")
                    except Exception as e:
                        logger.error(f"Failed to subscribe {batch} to {channel}: {e}")
                    await asyncio.sleep(0.1)

        # 2. Add to symbols list
        self.symbols.extend(symbols_to_add)

        # 3. Add to REST order book fetcher
        if self._orderbook_fetcher:
            self._orderbook_fetcher.symbols.extend(symbols_to_add)

        # 4. Add to feature engine
        if self.feature_engine:
            self.feature_engine.add_symbols(symbols_to_add)

        # 5. Update known symbols set
        self._known_symbols.update(symbols_to_add)

        logger.success(f"✅ Added {len(symbols_to_add)} new symbols: {symbols_to_add}")
        logger.info(f"Total symbols now: {len(self._known_symbols)}")

    def get_statistics(self) -> dict:
        """Get combined statistics."""
        stats = super().get_statistics()

        # Add feature engine statistics
        stats["feature_stats"] = self.feature_engine.get_statistics()

        # Add queue statistics
        stats["queue_stats"] = {
            **self._queue_stats,
            'current_size': self._trade_queue.qsize()
        }

        # Add ticker trade statistics
        symbols_with_real_trades = len(self._symbols_with_trades)
        stats["ticker_trade_stats"] = {
            'symbols_with_real_trades': symbols_with_real_trades,
            'symbols_using_ticker': len(self._last_ticker_price) - symbols_with_real_trades,
            'ticker_trades_generated': self._queue_stats.get('ticker_trades', 0),
        }

        # Add REST fetcher statistics
        if self._rest_fetcher:
            stats["rest_fetcher_stats"] = self._rest_fetcher.get_statistics()

        # Add REST order book fetcher statistics
        if self._orderbook_fetcher:
            stats["orderbook_fetcher_stats"] = self._orderbook_fetcher.get_statistics()

        return stats


async def main():
    """Main entry point."""
    collector = IntegratedCollector()

    try:
        await collector.start()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
