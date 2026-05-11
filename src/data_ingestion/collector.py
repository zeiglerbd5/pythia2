#!/usr/bin/env python3
"""
Pythia Market Data Collector - Main Entry Point

Coordinates WebSocket data ingestion, order book tracking, and database storage.
This is the Phase 1 implementation of the Pythia trading system.

Usage:
    python -m src.data_ingestion.collector

    or

    python src/data_ingestion/collector.py
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from dotenv import load_dotenv

from src.data_ingestion.coinbase_auth import CoinbaseAuth
from src.data_ingestion.database import DuckDBManager
from src.data_ingestion.websocket_manager import CoinbaseWebSocketManager
from src.signals.catalyst_detector import CatalystDetector, CatalystSignal
from src.utils.config import get_config
from src.monitoring.logger import setup_logging


class PythiaDataCollector:
    """
    Main coordinator for Pythia data collection system.

    Phase 1 implementation:
    - Coinbase WebSocket connectivity (level2, market_trades, ticker, heartbeats)
    - ES256 JWT authentication with 90s refresh
    - Order book tracking with snapshot/delta updates
    - DuckDB storage with batch writing
    - Graceful shutdown handling
    """

    def __init__(self):
        """Initialize Pythia data collector."""
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

        logger.info("Initializing Pythia Data Collector")
        logger.info(
            "Configuration loaded",
            extra={
                "environment": self.config.config.system.environment,
                "paper_trading": self.config.is_paper_trading(),
                "context_pairs": len(self.config.config.trading.context_pairs),
                "target_pairs": len(self.config.config.trading.target_pairs),
            }
        )

        # Initialize components
        self.auth: Optional[CoinbaseAuth] = None
        self.db_manager: Optional[DuckDBManager] = None
        self.ws_manager: Optional[CoinbaseWebSocketManager] = None
        self.catalyst_detector: Optional[CatalystDetector] = None

        # System state
        self.running = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.success("Pythia Data Collector initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

        # Trigger async shutdown
        if self.ws_manager:
            asyncio.create_task(self.ws_manager.stop())

    async def _on_catalyst_signal(self, signal: CatalystSignal):
        """Handle incoming catalyst signals - store in database."""
        if not self.db_manager:
            return

        # Convert CatalystSignal to database format
        for symbol in [signal.symbol]:  # Could be multiple symbols in future
            signal_dict = {
                "symbol": symbol,
                "timestamp": signal.detected_at,
                "source": ",".join(signal.sources[:3]),  # Limit to 3 sources
                "event_type": signal.catalyst_type,
                "confidence": signal.confidence,
                "title": signal.headline[:500] if signal.headline else None,
                "url": signal.source_urls[0] if signal.source_urls else None,
                "signal_hash": f"{symbol}:{signal.catalyst_type}:{signal.detected_at.timestamp():.0f}",
                "source_credibility": signal.confidence,  # Use confidence as credibility proxy
                "entity_certainty": signal.confidence,
                "event_priority": signal.priority_score,
                "recency_score": 1.0 if signal.urgency == "IMMEDIATE" else 0.7 if signal.urgency == "SOON" else 0.4,
                "engagement_score": signal.corroborating_signals / 5.0,  # Normalize to 0-1
                "sentiment_score": signal.impact_score,  # Use impact as sentiment proxy
            }
            self.db_manager.queue_news_signal(signal_dict)

        # Log high-priority signals
        if signal.priority_score >= 0.7:
            logger.info(
                f"🚨 HIGH PRIORITY CATALYST: [{signal.catalyst_type}] {signal.symbol} "
                f"- {signal.headline[:60]}... (priority={signal.priority_score:.2f}, action={signal.action})"
            )

    async def initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing components...")

        # Initialize authentication
        try:
            self.auth = CoinbaseAuth.from_env()
            logger.success("Coinbase authentication initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Coinbase auth: {e}")
            raise

        # Initialize database
        try:
            db_path = str(self.config.get_database_path())
            self.db_manager = DuckDBManager(
                db_path=db_path,
                batch_size=self.config.config.database.batch_size,
                batch_timeout_seconds=self.config.config.database.batch_timeout_seconds
            )
            logger.success(f"Database initialized at {db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

        # Initialize WebSocket manager
        try:
            # Get all trading pairs (context + target)
            all_pairs = self.config.get_all_pairs()

            self.ws_manager = CoinbaseWebSocketManager(
                auth=self.auth,
                db_manager=self.db_manager,
                symbols=all_pairs,
                channels=self.config.config.websocket.channels
            )
            logger.success(f"WebSocket manager initialized for {len(all_pairs)} pairs")

        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {e}")
            raise

        # Initialize Catalyst Detector
        try:
            self.catalyst_detector = CatalystDetector(
                enable_cryptopanic=True,
                enable_calendar=True,
                enable_exchange_listings=True,
                enable_twitter=True,
                enable_whale_alert=True,
                poll_interval_seconds=60,
            )
            # Register callback to store signals in database
            self.catalyst_detector.register_callback(self._on_catalyst_signal)
            logger.success("Catalyst detector initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize catalyst detector: {e}")
            # Non-fatal - continue without catalyst detection

        logger.success("All components initialized successfully")

    async def start(self):
        """Start the data collection system."""
        logger.info("=" * 80)
        logger.info("STARTING PYTHIA DATA COLLECTION SYSTEM")
        logger.info("=" * 80)

        self.running = True

        try:
            # Initialize components
            await self.initialize_components()

            # Display startup information
            self._display_startup_info()

            # Start catalyst detector (runs in background)
            if self.catalyst_detector:
                logger.info("Starting catalyst detector...")
                await self.catalyst_detector.start()

            # Start WebSocket manager (this will run until stopped)
            logger.info("Starting WebSocket data collection...")
            await self.ws_manager.start()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        except Exception as e:
            logger.error(f"System error: {e}")
            logger.exception("Full traceback:")

        finally:
            await self.stop()

    async def stop(self):
        """Stop the data collection system gracefully."""
        logger.info("=" * 80)
        logger.info("STOPPING PYTHIA DATA COLLECTION SYSTEM")
        logger.info("=" * 80)

        self.running = False

        # Stop catalyst detector
        if self.catalyst_detector:
            logger.info("Stopping catalyst detector...")
            await self.catalyst_detector.stop()

        # Stop WebSocket manager
        if self.ws_manager:
            logger.info("Stopping WebSocket manager...")
            await self.ws_manager.stop()

        # Display final statistics
        self._display_final_statistics()

        logger.success("Pythia Data Collector stopped gracefully")

    def _display_startup_info(self):
        """Display startup information."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("PYTHIA DATA COLLECTION SYSTEM - PHASE 1")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Configuration:")
        logger.info(f"  Environment:      {self.config.config.system.environment}")
        logger.info(f"  Paper Trading:    {self.config.is_paper_trading()}")
        logger.info(f"  Database:         {self.config.get_database_path()}")
        logger.info(f"  Log File:         {self.config.get_log_path()}")
        logger.info("")
        logger.info("Trading Pairs:")
        logger.info(f"  Context Pairs:    {len(self.config.config.trading.context_pairs)} (BTC, ETH)")
        logger.info(f"  Target Pairs:     {len(self.config.config.trading.target_pairs)} altcoins")
        logger.info(f"  Total Monitored:  {len(self.config.get_all_pairs())}")
        logger.info("")
        logger.info("WebSocket Channels:")
        for channel in self.config.config.websocket.channels:
            logger.info(f"  - {channel}")
        logger.info("")
        logger.info("Data Collection Features:")
        logger.info("  ✓ ES256 JWT authentication with 90s refresh")
        logger.info("  ✓ Level2 order book tracking (guaranteed delivery)")
        logger.info("  ✓ Market trades (250ms batched executions)")
        logger.info("  ✓ Ticker updates (best bid/ask)")
        logger.info("  ✓ Heartbeat monitoring (prevent timeout)")
        logger.info("  ✓ Automatic reconnection with exponential backoff")
        logger.info("  ✓ Batch database writing (configurable intervals)")
        logger.info("  ✓ Order book imbalance calculation (L=5 depth)")
        logger.info("")
        if self.catalyst_detector:
            logger.info("Catalyst Detection:")
            logger.info("  ✓ CryptoPanic news aggregation")
            logger.info("  ✓ CoinMarketCal scheduled events")
            logger.info("  ✓ Binance listing announcements (API)")
            logger.info("  ✓ Twitter/X monitoring (@CoinbaseMarkets, etc.)")
            logger.info("  ✓ Whale Alert large transfers")
            logger.info(f"  Poll interval: {self.catalyst_detector.poll_interval}s")
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
        logger.info("WebSocket:")
        logger.info(f"  Messages Processed:     {stats.get('message_count', 0):,}")
        logger.info(f"  Reconnection Attempts:  {stats.get('reconnect_attempts', 0)}")
        logger.info(f"  Order Books Tracked:    {stats.get('order_books', 0)}")
        logger.info("")

        db_stats = stats.get("database_stats", {})
        logger.info("Database:")
        logger.info(f"  Tickers Written:        {db_stats.get('tickers_written', 0):,}")
        logger.info(f"  Trades Written:         {db_stats.get('trades_written', 0):,}")
        logger.info(f"  Orderbooks Written:     {db_stats.get('orderbooks_written', 0):,}")
        logger.info(f"  Batches Written:        {db_stats.get('batches_written', 0):,}")
        logger.info(f"  Total Tickers:          {db_stats.get('tickers_count', 0):,}")
        logger.info(f"  Total Trades:           {db_stats.get('trades_count', 0):,}")
        logger.info(f"  Total OB Snapshots:     {db_stats.get('order_book_snapshots_count', 0):,}")
        logger.info("")
        logger.info("=" * 80)
        logger.info("")


async def main():
    """Main entry point."""
    collector = PythiaDataCollector()

    try:
        await collector.start()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    # Run the collector
    asyncio.run(main())
