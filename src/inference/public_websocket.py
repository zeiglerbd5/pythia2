"""
Public WebSocket Client for Coinbase (No Authentication Required)

Connects to Coinbase public channels for real-time market data.
No API keys or credentials needed.
"""

import asyncio
import json
from typing import List, Callable, Optional
from datetime import datetime
import websockets
from loguru import logger


class PublicWebSocketClient:
    """
    Simple WebSocket client for Coinbase public market data.

    No authentication required - connects to public channels only.
    """

    def __init__(
        self,
        symbols: List[str],
        on_ticker: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
        channels: Optional[List[str]] = None
    ):
        """
        Initialize public WebSocket client.

        Args:
            symbols: List of product IDs (e.g., ['BTC-USD', 'ETH-USD'])
            on_ticker: Callback(symbol, price, volume_24h)
            on_trade: Callback(symbol, price, size, side)
            channels: List of public channels (default: ['ticker', 'market_trades'])
        """
        self.symbols = symbols
        self.on_ticker = on_ticker
        self.on_trade = on_trade
        self.channels = channels or ['ticker', 'market_trades']

        # Coinbase public WebSocket endpoint
        self.ws_url = "wss://ws-feed.exchange.coinbase.com"

        # Connection state
        self.websocket = None
        self.running = False
        self.connected = False

        # Statistics
        self.message_count = 0
        self.ticker_count = 0
        self.trade_count = 0

        logger.info(
            f"PublicWebSocketClient initialized: {len(symbols)} symbols, "
            f"channels={self.channels}"
        )

    async def connect(self):
        """Connect to WebSocket and subscribe to channels."""
        logger.info(f"Connecting to {self.ws_url}")

        try:
            async with websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10
            ) as websocket:
                self.websocket = websocket
                self.connected = True

                # Subscribe to channels (no auth needed for public channels)
                subscribe_message = {
                    "type": "subscribe",
                    "product_ids": self.symbols,
                    "channels": self.channels
                }

                await websocket.send(json.dumps(subscribe_message))
                logger.info(f"Subscribed to {self.channels} for {len(self.symbols)} symbols")

                # Process messages
                async for message in websocket:
                    if not self.running:
                        break

                    try:
                        await self._process_message(message)

                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.connected = False

    async def _process_message(self, message: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            self.message_count += 1

            # Log subscription confirmations
            if msg_type == 'subscriptions':
                logger.info(f"Subscription confirmed: {data}")
                return

            # Handle ticker updates
            if msg_type == 'ticker':
                self.ticker_count += 1
                await self._handle_ticker(data)

            # Handle trade updates
            elif msg_type in ['match', 'last_match']:
                self.trade_count += 1
                await self._handle_trade(data)

            # Log periodic stats
            if self.message_count % 1000 == 0:
                logger.info(
                    f"Messages: {self.message_count} "
                    f"(tickers: {self.ticker_count}, trades: {self.trade_count})"
                )

        except json.JSONDecodeError:
            logger.warning("Invalid JSON message received")

    async def _handle_ticker(self, data: dict):
        """Handle ticker message."""
        if not self.on_ticker:
            return

        try:
            symbol = data.get('product_id')
            price = float(data.get('price', 0))
            volume_24h = float(data.get('volume_24h', 0))

            # Call callback
            self.on_ticker(symbol, price, volume_24h)

        except Exception as e:
            logger.error(f"Error handling ticker: {e}")

    async def _handle_trade(self, data: dict):
        """Handle trade message."""
        if not self.on_trade:
            return

        try:
            symbol = data.get('product_id')
            price = float(data.get('price', 0))
            size = float(data.get('size', 0))
            side = data.get('side', 'buy').upper()

            # Call callback
            self.on_trade(symbol, price, size, side)

        except Exception as e:
            logger.error(f"Error handling trade: {e}")

    async def start(self):
        """Start WebSocket connection and run indefinitely."""
        self.running = True

        while self.running:
            try:
                await self.connect()

            except Exception as e:
                logger.error(f"Connection failed: {e}")

            if self.running:
                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def stop(self):
        """Stop WebSocket connection."""
        logger.info("Stopping WebSocket client")
        self.running = False
        self.connected = False

        if self.websocket and not self.websocket.closed:
            await self.websocket.close()

    def get_statistics(self) -> dict:
        """Get connection statistics."""
        return {
            'connected': self.connected,
            'running': self.running,
            'message_count': self.message_count,
            'ticker_count': self.ticker_count,
            'trade_count': self.trade_count,
            'symbols': len(self.symbols),
            'channels': self.channels,
        }


async def test_public_websocket():
    """Test public WebSocket connection."""

    def on_ticker(symbol, price, volume_24h):
        logger.info(f"TICKER: {symbol} @ ${price:,.2f} (24h vol: ${volume_24h:,.0f})")

    def on_trade(symbol, price, size, side):
        logger.info(f"TRADE: {symbol} {side} {size:.4f} @ ${price:,.2f}")

    # Test with a few symbols
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']

    client = PublicWebSocketClient(
        symbols=symbols,
        on_ticker=on_ticker,
        on_trade=on_trade
    )

    try:
        await client.start()

    except KeyboardInterrupt:
        logger.info("Stopping...")
        await client.stop()

        # Print stats
        stats = client.get_statistics()
        logger.info(f"Final statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(test_public_websocket())
