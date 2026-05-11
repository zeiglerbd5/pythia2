"""
Coinbase Advanced Trade WebSocket Manager

Implements reliable WebSocket connectivity based on proven patterns from
project_stephanie, adapted for Coinbase Advanced Trade API with ES256 JWT authentication.

Key features:
- Multiple channel subscriptions (level2, market_trades, ticker, heartbeats)
- ES256 JWT authentication with 90-second refresh
- Automatic reconnection with exponential backoff
- Order book state tracking
- Async batch database writing
- Graceful shutdown handling
"""

import asyncio
import json
import time
import signal
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from collections import defaultdict

import websockets
from websockets.exceptions import ConnectionClosed
from loguru import logger

from .coinbase_auth import CoinbaseAuth
from .order_book import OrderBookManager
from .database import DuckDBManager


class CoinbaseWebSocketManager:
    """
    Manages WebSocket connections to Coinbase Advanced Trade API.

    Per implementation guide:
    - WebSocket endpoint: wss://advanced-trade-ws.coinbase.com
    - Channels: level2 (guaranteed delivery), market_trades, ticker, heartbeats
    - JWT refresh every 90 seconds
    - Exponential backoff reconnection
    - Sub-100ms message processing
    """

    def __init__(
        self,
        auth: CoinbaseAuth,
        db_manager: DuckDBManager,
        symbols: List[str],
        channels: Optional[List[str]] = None
    ):
        """
        Initialize WebSocket manager.

        Args:
            auth: CoinbaseAuth instance for JWT authentication
            db_manager: DuckDBManager for data storage
            symbols: List of trading pair symbols to monitor
            channels: List of channels to subscribe (default: level2, market_trades, ticker, heartbeats)
        """
        self.auth = auth
        self.db_manager = db_manager
        self.symbols = symbols
        self.channels = channels or ["level2", "market_trades", "ticker", "heartbeats"]

        # Coinbase Advanced Trade WebSocket endpoint (per guide)
        self.ws_url = "wss://advanced-trade-ws.coinbase.com"

        # WebSocket state
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        self.connected = False

        # Order book manager (tracks state for all symbols)
        self.orderbook_manager = OrderBookManager()

        # Message statistics
        self.message_count = 0
        self.last_message_time: Optional[datetime] = None
        self.messages_by_type: Dict[str, int] = defaultdict(int)

        # Reconnection settings (per guide and reference implementation)
        self.reconnect_delay_seconds = 5
        self.max_reconnect_delay_seconds = 60
        self.reconnect_backoff_multiplier = 1.5
        self.max_reconnect_attempts = 999999  # Effectively unlimited - Coinbase has occasional HTTP 500s
        self.reconnect_attempts = 0

        # JWT refresh settings (per guide: 90 seconds)
        self.jwt_refresh_interval = 90
        self.last_jwt_refresh = 0

        # Periodic snapshot settings (per guide: every 10 seconds)
        self.snapshot_interval = 10
        self.last_snapshot_time = 0

        # Connection health monitoring
        # Initialize to current time to prevent immediate reconnection on startup
        self.last_heartbeat_time = time.time()
        self.heartbeat_timeout = 90  # seconds

        # Tasks
        self._tasks: List[asyncio.Task] = []

        logger.info(
            f"WebSocketManager initialized",
            extra={
                "symbols": len(symbols),
                "channels": self.channels,
                "endpoint": self.ws_url
            }
        )

    async def start(self):
        """
        Start WebSocket connection and all background tasks.
        """
        logger.info("Starting WebSocket manager")
        self.running = True

        try:
            # Start database batch writer
            await self.db_manager.start_batch_writer()

            # Create main connection task
            connection_task = asyncio.create_task(self._connection_loop())
            self._tasks.append(connection_task)

            # Create JWT refresh task (per guide: every 90s)
            refresh_task = asyncio.create_task(self._jwt_refresh_loop())
            self._tasks.append(refresh_task)

            # Create periodic snapshot task
            snapshot_task = asyncio.create_task(self._snapshot_loop())
            self._tasks.append(snapshot_task)

            # Create health monitoring task
            health_task = asyncio.create_task(self._health_check_loop())
            self._tasks.append(health_task)

            # Wait for all tasks
            await asyncio.gather(*self._tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"WebSocket manager error: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """
        Stop WebSocket connection and cleanup.
        """
        logger.info("Stopping WebSocket manager")
        self.running = False
        self.connected = False

        # Close WebSocket connection
        if self.websocket and self.websocket.state.name != "CLOSED":
            await self.websocket.close()

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop database batch writer and flush remaining data
        await self.db_manager.stop_batch_writer()

        logger.info("WebSocket manager stopped")

    async def _connection_loop(self):
        """
        Main connection loop with automatic reconnection.

        Per guide and reference implementation:
        - Exponential backoff on failures
        - State recovery after reconnection
        """
        while self.running:
            try:
                await self._connect_and_subscribe()

            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                self.connected = False

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connected = False

            if self.running:
                # Calculate reconnection delay with exponential backoff
                delay = min(
                    self.reconnect_delay_seconds * (self.reconnect_backoff_multiplier ** self.reconnect_attempts),
                    self.max_reconnect_delay_seconds
                )

                self.reconnect_attempts += 1

                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
                    self.running = False
                    break

                logger.info(f"Reconnecting in {delay:.1f}s (attempt {self.reconnect_attempts})")
                await asyncio.sleep(delay)

    async def _connect_and_subscribe(self):
        """
        Connect to WebSocket and subscribe to channels.

        Per guide:
        - Subscribe to level2, market_trades, ticker, heartbeats
        - Include JWT token in subscription messages
        """
        logger.info(f"Connecting to {self.ws_url}")

        async with websockets.connect(
            self.ws_url,
            ping_interval=30,
            ping_timeout=60,
            close_timeout=10,
            max_size=50 * 1024 * 1024,  # 50MB - level2 snapshots for many symbols are large
        ) as websocket:
            self.websocket = websocket
            self.connected = True
            self.reconnect_attempts = 0  # Reset on successful connection

            logger.info("WebSocket connected, subscribing to channels")

            # Subscribe to each channel in batches (to avoid message too big error)
            # Coinbase limits message size, so we batch symbols
            batch_size = 5  # Subscribe to max 5 symbols at a time

            for channel in self.channels:
                # Split symbols into batches
                for i in range(0, len(self.symbols), batch_size):
                    symbol_batch = self.symbols[i:i + batch_size]

                    if self.auth:
                        # Authenticated subscription
                        subscribe_msg = self.auth.get_websocket_auth_message(channel, symbol_batch)
                    else:
                        # Unauthenticated subscription (public market data)
                        subscribe_msg = {
                            "type": "subscribe",
                            "product_ids": symbol_batch,
                            "channel": channel
                        }

                    await websocket.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to {channel} for {len(symbol_batch)} symbols: {symbol_batch}")

                    # Small delay between subscriptions
                    await asyncio.sleep(0.1)

            # Process incoming messages
            async for message in websocket:
                if not self.running:
                    break

                try:
                    await self._process_message(message)

                except Exception as e:
                    logger.error(f"Error processing message: {e}")

    async def _process_message(self, message: str):
        """
        Process WebSocket message.

        Routes to appropriate handler based on channel.
        """
        try:
            data = json.loads(message)

        except json.JSONDecodeError:
            logger.warning("Received invalid JSON message")
            return

        # Update statistics
        self.message_count += 1
        self.last_message_time = datetime.now()
        # Update heartbeat time on any message - proves connection is alive
        self.last_heartbeat_time = time.time()

        # Get channel type
        channel = data.get("channel")
        msg_type = data.get("type")

        # Track message types
        self.messages_by_type[f"{channel}:{msg_type}"] += 1

        # Log periodic statistics
        if self.message_count % 1000 == 0:
            logger.info(
                f"Processed {self.message_count} messages",
                extra={
                    "last_message": self.last_message_time.isoformat() if self.last_message_time else None,
                    "message_types": dict(self.messages_by_type)
                }
            )

        # Route to appropriate handler
        try:
            if msg_type == "subscriptions":
                # Subscription confirmation
                logger.info(f"Subscription confirmed: {data}")

            elif msg_type == "error":
                # Error message
                logger.error(f"WebSocket error message: {data}")

            elif channel == "level2":
                await self._handle_level2(data)

            elif channel == "market_trades":
                await self._handle_market_trades(data)

            elif channel == "ticker":
                await self._handle_ticker(data)

            elif channel == "heartbeats":
                await self._handle_heartbeat(data)

            else:
                logger.debug(f"Unknown message type: {channel}:{msg_type}")

        except Exception as e:
            logger.error(f"Error handling {channel}:{msg_type} message: {e}")

    async def _handle_level2(self, data: Dict[str, Any]):
        """
        Handle level2 channel messages (order book updates).

        Per guide:
        - Guaranteed delivery with sequence numbers
        - Snapshot messages initialize state
        - Update messages apply deltas (absolute quantities)
        """
        # Process through order book manager
        success = self.orderbook_manager.process_message(data)

        if not success:
            # Desynchronization detected, need to recover
            # In production, would trigger REST API snapshot fetch
            events = data.get("events", [])
            if events:
                symbol = events[0].get("product_id")
                logger.warning(f"Order book desynchronization detected for {symbol}")

    async def _handle_market_trades(self, data: Dict[str, Any]):
        """
        Handle market_trades channel messages.

        Per guide: Batched every 250ms, reveals execution flow

        Note: product_id is in each trade object, NOT at the event level.
        """
        events = data.get("events", [])

        for event in events:
            trades = event.get("trades", [])

            for trade in trades:
                # product_id is in each trade, not at event level
                symbol = trade.get("product_id")

                trade_data = {
                    "timestamp": trade.get("time"),
                    "trade_id": trade.get("trade_id"),
                    "price": trade.get("price"),
                    "size": trade.get("size"),
                    "side": trade.get("side"),
                }

                self.db_manager.queue_trade(symbol, trade_data)

    async def _handle_ticker(self, data: Dict[str, Any]):
        """
        Handle ticker channel messages.

        Per guide: Updates on every match with automatic batching
        """
        events = data.get("events", [])

        for event in events:
            tickers = event.get("tickers", [])

            for ticker in tickers:
                # product_id might be in different places depending on message format
                symbol = ticker.get("product_id") or event.get("product_id") or data.get("product_id")

                if not symbol:
                    # Try to find symbol in nested structures
                    if "type" in ticker and ticker.get("type") == "ticker":
                        symbol = ticker.get("product_id")

                if not symbol:
                    continue  # Skip tickers without a symbol

                ticker_data = {
                    # timestamp is at the data root level, not in event
                    "timestamp": data.get("timestamp") or event.get("timestamp") or ticker.get("time"),
                    "price": ticker.get("price"),
                    "volume_24h": ticker.get("volume_24_h"),
                    "best_bid": ticker.get("best_bid"),
                    "best_ask": ticker.get("best_ask"),
                }

                self.db_manager.queue_ticker(symbol, ticker_data)

    async def _handle_heartbeat(self, data: Dict[str, Any]):
        """
        Handle heartbeat channel messages.

        Per guide: Prevents 60-90s timeout on illiquid pairs
        """
        self.last_heartbeat_time = time.time()

        # Heartbeat messages provide sequence verification
        heartbeat_counter = data.get("heartbeat_counter")
        if heartbeat_counter and heartbeat_counter % 100 == 0:
            logger.debug(f"Heartbeat: {heartbeat_counter}")

    async def _jwt_refresh_loop(self):
        """
        Background task to refresh JWT token every 90 seconds.

        Per guide:
        - Unsubscribe from channel
        - Wait 100ms
        - Resubscribe with fresh JWT

        Note: Skipped in unauthenticated mode
        """
        if not self.auth:
            # No JWT refresh needed for unauthenticated connections
            return

        while self.running:
            try:
                await asyncio.sleep(self.jwt_refresh_interval)

                if not self.connected or not self.websocket:
                    continue

                logger.info("Refreshing JWT tokens for all channels")

                for channel in self.channels:
                    if channel == "heartbeats":
                        # Heartbeats don't require auth
                        continue

                    # Get refresh messages (unsubscribe + subscribe with new JWT)
                    unsub_msg, sub_msg = self.auth.refresh_websocket_subscription(
                        channel, self.symbols
                    )

                    # Unsubscribe
                    await self.websocket.send(json.dumps(unsub_msg))

                    # Wait 100ms per guide
                    await asyncio.sleep(0.1)

                    # Resubscribe with fresh JWT
                    await self.websocket.send(json.dumps(sub_msg))

                    logger.debug(f"Refreshed JWT for {channel}")

                self.last_jwt_refresh = time.time()

            except Exception as e:
                logger.error(f"Error refreshing JWT: {e}")

    async def _snapshot_loop(self):
        """
        Background task to periodically save order book snapshots.

        Per guide: Every 10 seconds for feature calculation
        """
        while self.running:
            try:
                await asyncio.sleep(self.snapshot_interval)

                if not self.connected:
                    continue

                # Get all order book snapshots
                snapshots = self.orderbook_manager.get_all_snapshots()

                for snapshot in snapshots:
                    # Calculate additional metrics
                    mid_price = snapshot.mid_price
                    spread = snapshot.spread

                    spread_bps = None
                    if mid_price and mid_price > 0 and spread:
                        spread_bps = (spread / mid_price) * 10000  # basis points

                    # Only store if we have actual order book data
                    if snapshot.bids and snapshot.asks:
                        snapshot_data = {
                            "timestamp": datetime.fromtimestamp(snapshot.timestamp),
                            "bids": snapshot.bids[:50],  # All available levels
                            "asks": snapshot.asks[:50],
                            "best_bid": snapshot.bids[0][0],
                            "best_ask": snapshot.asks[0][0],
                            "mid_price": mid_price,
                            "spread": spread,
                            "spread_bps": spread_bps,
                            "sequence_num": snapshot.sequence_num,
                        }
                        self.db_manager.queue_orderbook(snapshot.symbol, snapshot_data)

                self.last_snapshot_time = time.time()

            except Exception as e:
                logger.error(f"Error in snapshot loop: {e}")

    async def _health_check_loop(self):
        """
        Monitor connection health and trigger reconnection if needed.
        """
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                if not self.connected:
                    continue

                # Check if we've received recent heartbeats
                time_since_heartbeat = time.time() - self.last_heartbeat_time

                if time_since_heartbeat > self.heartbeat_timeout:
                    logger.warning(
                        f"No heartbeat received for {time_since_heartbeat:.0f}s, "
                        "closing connection to trigger reconnect"
                    )

                    if self.websocket and self.websocket.state.name != "CLOSED":
                        await self.websocket.close()

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get WebSocket manager statistics.

        Returns:
            Dictionary with connection stats
        """
        stats = {
            "connected": self.connected,
            "running": self.running,
            "message_count": self.message_count,
            "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
            "messages_by_type": dict(self.messages_by_type),
            "reconnect_attempts": self.reconnect_attempts,
            "symbols": len(self.symbols),
            "channels": self.channels,
            "order_books": len(self.orderbook_manager.books),
            "time_since_heartbeat": time.time() - self.last_heartbeat_time if self.last_heartbeat_time else None,
            "time_since_jwt_refresh": time.time() - self.last_jwt_refresh if self.last_jwt_refresh else None,
        }

        # Add order book statistics
        stats["order_book_stats"] = self.orderbook_manager.get_statistics_all()

        # Add database statistics
        stats["database_stats"] = self.db_manager.get_statistics()

        return stats


async def main():
    """
    Test WebSocket manager.

    Usage:
        python -m src.data_ingestion.websocket_manager
    """
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize components
    auth = CoinbaseAuth.from_env()
    db_manager = DuckDBManager("data/test_ws_pythia.duckdb")

    # Test with a few symbols
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

    # Create WebSocket manager
    ws_manager = CoinbaseWebSocketManager(
        auth=auth,
        db_manager=db_manager,
        symbols=symbols
    )

    # Setup graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(ws_manager.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start WebSocket manager
    try:
        await ws_manager.start()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

    finally:
        await ws_manager.stop()

        # Print final statistics
        stats = ws_manager.get_statistics()
        logger.info("Final Statistics:")
        logger.info(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
