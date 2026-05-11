"""
Whale Alert Source

Monitors large cryptocurrency transfers using Whale Alert WebSocket API.

Priority: Tier 1 (high value signal)
Cost: $30/month (100 alerts/hour, 2 concurrent connections)
API: https://developer.whale-alert.io/
"""

import os
import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Callable
import aiohttp
from loguru import logger

from .base import BaseSource, NewsItem


class WhaleAlertSource(BaseSource):
    """
    Monitors large cryptocurrency transfers via Whale Alert WebSocket API.

    Large transfers can indicate:
    - Exchange inflows (potential selling pressure)
    - Exchange outflows (accumulation)
    - OTC deals
    - Whale accumulation/distribution

    Plan limits: 100 alerts/hour, 2 concurrent WebSocket connections
    """

    WS_URL = "wss://leviathan.whale-alert.io/ws"

    # Minimum USD value ($100k is API minimum, we use $1M)
    MIN_USD_VALUE = 1_000_000

    # Blockchain mappings to trading symbols
    BLOCKCHAIN_TO_SYMBOL = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "ripple": "XRP",
        "litecoin": "LTC",
        "tron": "TRX",
        "stellar": "XLM",
        "eos": "EOS",
        "tezos": "XTZ",
        "neo": "NEO",
        "binancecoin": "BNB",
        "cardano": "ADA",
        "dogecoin": "DOGE",
        "polkadot": "DOT",
        "solana": "SOL",
        "avalanche": "AVAX",
        "cosmos": "ATOM",
        "algorand": "ALGO",
        "near": "NEAR",
        "fantom": "FTM",
        "polygon": "POL",  # MATIC renamed to POL
        "arbitrum": "ARB",
        "optimism": "OP",
    }

    # Reverse mapping
    SYMBOL_TO_BLOCKCHAIN = {v: k for k, v in BLOCKCHAIN_TO_SYMBOL.items()}

    def __init__(
        self,
        api_key: Optional[str] = None,
        min_usd_value: int = 1_000_000,
        symbols: Optional[List[str]] = None,
        request_timeout: int = 30,
    ):
        """
        Initialize Whale Alert source.

        Args:
            api_key: Whale Alert API key (or from WHALE_ALERT_API_KEY env var)
            min_usd_value: Minimum USD value for whale transactions (min $100k)
            symbols: Optional list of symbols to monitor (e.g., ["BTC", "ETH"])
            request_timeout: Timeout for API requests
        """
        super().__init__(
            rate_limit_per_minute=100,  # Not really used for WebSocket
            request_timeout=request_timeout,
        )

        self.api_key = api_key or os.environ.get("WHALE_ALERT_API_KEY", "")
        self.min_usd_value = max(min_usd_value, 100_000)  # API minimum is $100k
        self.symbols = symbols

        # Convert symbols to blockchains for subscription
        self.blockchains = None
        if symbols:
            self.blockchains = []
            for sym in symbols:
                bc = self.SYMBOL_TO_BLOCKCHAIN.get(sym.upper())
                if bc:
                    self.blockchains.append(bc)

        # WebSocket state
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._running = False
        self._alert_queue: asyncio.Queue = asyncio.Queue()
        self._reconnect_delay = 60  # Start with 60s to avoid rate limits on restart
        self._max_reconnect_delay = 600  # Max 10 minutes (paid tier still has limits)
        self._consecutive_errors = 0
        self._rate_limited = False  # Track if we're currently rate limited

        # Track seen transaction hashes
        self._seen_transactions: set = set()

        # Alert callback for real-time processing
        self._alert_callback: Optional[Callable] = None

        if not self.api_key:
            logger.warning("WHALE_ALERT_API_KEY not set - Whale Alert source disabled")
        else:
            logger.info(f"Whale Alert initialized (min ${min_usd_value:,}, WebSocket mode)")

    @property
    def source_name(self) -> str:
        return "whale_alert"

    @property
    def source_credibility(self) -> float:
        return 0.9  # On-chain data is highly reliable

    def set_alert_callback(self, callback: Callable):
        """Set callback for real-time alert processing."""
        self._alert_callback = callback

    async def start_websocket(self, startup_delay: int = 30):
        """Start the WebSocket connection for real-time alerts.

        Args:
            startup_delay: Seconds to wait before first connection (avoids rate limits on restart)
        """
        if not self.api_key:
            logger.warning("Cannot start WebSocket - no API key")
            return

        self._running = True

        # Delay initial connection to avoid rate limits on restart
        if startup_delay > 0:
            logger.info(f"Whale Alert: waiting {startup_delay}s before connecting (rate limit protection)")
            await asyncio.sleep(startup_delay)

        self._ws_task = asyncio.create_task(self._websocket_loop())
        logger.info("Whale Alert WebSocket started")

    async def stop_websocket(self):
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        logger.info("Whale Alert WebSocket stopped")

    async def _websocket_loop(self):
        """Main WebSocket connection loop with reconnection and exponential backoff."""
        while self._running:
            try:
                await self._connect_and_listen()

                # If we disconnected due to rate limit, don't reset delay
                if not self._rate_limited:
                    self._consecutive_errors = 0
                    self._reconnect_delay = 60  # Reset to base delay
                else:
                    self._consecutive_errors += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_errors += 1
                # Check for rate limit (429)
                if "429" in str(e):
                    logger.warning(f"Whale Alert rate limited (HTTP 429), backing off...")
                    self._rate_limited = True
                    self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
                else:
                    logger.error(f"Whale Alert WebSocket error: {e}")

            if self._running:
                status = "(rate limited)" if self._rate_limited else ""
                logger.info(f"Whale Alert: reconnecting in {self._reconnect_delay}s {status}")
                await asyncio.sleep(self._reconnect_delay)

    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for alerts."""
        url = f"{self.WS_URL}?api_key={self.api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url, heartbeat=30) as ws:
                self._ws = ws
                logger.info("Whale Alert WebSocket connected")

                # Subscribe to alerts
                await self._subscribe(ws)

                # Listen for messages
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        should_continue = await self._handle_message(msg.data)
                        if not should_continue:
                            logger.warning("Whale Alert: disconnecting due to rate limit, will backoff")
                            break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("WebSocket closed by server")
                        break

    async def _subscribe(self, ws: aiohttp.ClientWebSocketResponse):
        """Send subscription message."""
        subscribe_msg = {
            "type": "subscribe_alerts",
            "tx_types": ["transfer"],  # Focus on transfers
            "min_value_usd": self.min_usd_value,
        }

        # Add blockchain filter if specified
        if self.blockchains:
            subscribe_msg["blockchains"] = self.blockchains

        await ws.send_json(subscribe_msg)
        logger.info(f"Subscribed to whale alerts (min ${self.min_usd_value:,})")

    async def _handle_message(self, data: str) -> bool:
        """Handle incoming WebSocket message. Returns False if we should disconnect."""
        try:
            msg = json.loads(data)

            # Check for error responses (rate limit, auth errors)
            error = msg.get("error", "")
            if error:
                if "rate limit" in error.lower():
                    logger.warning(f"🐋 [WHALE_ALERT] Rate limited: {error}")
                    self._rate_limited = True
                    self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
                    return False  # Signal to disconnect and backoff
                elif "not authenticated" in error.lower():
                    logger.warning(f"🐋 [WHALE_ALERT] Auth error (likely rate limit cooldown): {error}")
                    self._rate_limited = True
                    self._reconnect_delay = 300  # Wait 5 min on auth errors
                    return False
                else:
                    logger.warning(f"🐋 [WHALE_ALERT] Error: {error}")
                    return True  # Continue for other errors

            # Log incoming messages (reduce verbosity for alerts)
            msg_type = msg.get("type")
            if msg_type != "alert":
                logger.info(f"[WHALE_WS] Received: {str(msg)[:300]}")

            if msg_type == "alert":
                self._rate_limited = False  # Getting alerts = not rate limited
                await self._process_alert(msg)
            elif msg_type in ("subscribed", "subscribed_alerts"):
                sub_id = msg.get("id", msg.get("channel_id", msg.get("subscription_id")))
                logger.info(f"Subscription confirmed: {sub_id}")
                self._rate_limited = False
            elif msg_type == "error":
                logger.error(f"WebSocket error: {msg.get('message')}")
            else:
                # Log unknown types for debugging
                logger.debug(f"Unknown message type: {msg_type} - {str(msg)[:200]}")

            return True  # Continue listening

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from WebSocket: {e}")
            return True

    async def _process_alert(self, alert: Dict[str, Any]):
        """Process a whale alert."""
        try:
            # Debug: log alert structure on first few
            if len(self._seen_transactions) < 5:
                logger.debug(f"Alert structure: {json.dumps(alert, default=str)[:500]}")

            # Get transaction hash for deduplication
            tx = alert.get("transaction", {})
            if isinstance(tx, dict):
                tx_hash = tx.get("hash", "")
            else:
                # Extract from text if transaction is not a dict
                text = alert.get("text", "")
                if "hash:" in text.lower():
                    tx_hash = text.split("hash:")[-1].strip().split()[0]
                else:
                    tx_hash = f"{alert.get('timestamp', '')}_{alert.get('blockchain', '')}"

            # Skip duplicates
            if tx_hash in self._seen_transactions:
                return
            self._seen_transactions.add(tx_hash)

            # Limit cache size
            if len(self._seen_transactions) > 10000:
                self._seen_transactions = set(list(self._seen_transactions)[-5000:])

            # Create NewsItem
            item = self._transaction_to_news_item(alert)

            if item:
                # Add to queue for fetch_items()
                await self._alert_queue.put(item)

                # Call callback if set
                if self._alert_callback:
                    try:
                        await self._alert_callback(item)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")

                # Log the alert
                logger.warning(f"🐋 [WHALE_ALERT] {item.title}")

        except Exception as e:
            logger.error(f"Error processing whale alert: {e}")

    def _transaction_to_news_item(self, alert: Dict[str, Any]) -> Optional[NewsItem]:
        """Convert WebSocket alert to NewsItem."""
        tx = alert.get("transaction", {})
        if isinstance(tx, str):
            tx = {}

        blockchain = alert.get("blockchain", "").lower()

        # Get symbol from amounts array (more reliable than blockchain mapping)
        amounts = alert.get("amounts", [])
        if amounts and isinstance(amounts[0], dict):
            alert_symbol = amounts[0].get("symbol", "").upper()
            amount = amounts[0].get("amount", 0)
            amount_usd = amounts[0].get("value_usd", 0)
        else:
            alert_symbol = self.BLOCKCHAIN_TO_SYMBOL.get(blockchain)
            amount = 0
            amount_usd = 0

        # Map to our trading symbol format
        symbol = alert_symbol
        if symbol and symbol not in ("USDT", "USDC", "DAI", "PYUSD"):  # Skip stablecoins for trading
            # Add -USD suffix if not present
            if not symbol.endswith("-USD"):
                symbol = f"{symbol}-USD"
        else:
            # For stablecoins, we might still want to track the flow
            symbol = alert_symbol

        timestamp = alert.get("timestamp", 0)

        # Extract from/to - these are simple strings in WebSocket API
        from_name = alert.get("from", "unknown")
        to_name = alert.get("to", "unknown")

        # Determine owner types from names
        known_exchanges = ["binance", "coinbase", "kraken", "bybit", "okx", "huobi",
                          "kucoin", "bitfinex", "gemini", "bitstamp", "crypto.com"]
        from_owner = "exchange" if any(ex in from_name.lower() for ex in known_exchanges) else "unknown"
        to_owner = "exchange" if any(ex in to_name.lower() for ex in known_exchanges) else "unknown"

        # Classify the move
        subtype = self._classify_whale_move(from_owner, to_owner)

        # Build title
        amount_str = f"{amount:,.0f}" if amount >= 1 else f"{amount:.4f}"
        usd_str = f"${amount_usd:,.0f}"
        title = f"{amount_str} {alert_symbol} ({usd_str})"

        if from_name != "unknown wallet" or to_name != "unknown wallet":
            title += f" | {from_name} → {to_name}"

        # Sentiment based on flow direction
        engagement = {"sentiment": "neutral"}
        if subtype == "exchange_inflow":
            engagement["sentiment"] = "bearish"
        elif subtype == "exchange_outflow":
            engagement["sentiment"] = "bullish"

        # Get transaction hash
        tx_hash = ""
        if isinstance(tx, dict):
            tx_hash = tx.get("hash", "")
        if not tx_hash:
            # Try to extract from text
            text = alert.get("text", "")
            if "hash:" in text.lower():
                tx_hash = text.split("hash:")[-1].strip().split()[0]

        return NewsItem(
            source=self.source_name,
            event_type="whale_move",
            title=title,
            content=f"{subtype} | {from_name} → {to_name} | {blockchain}",
            url=f"https://whale-alert.io/transaction/{blockchain}/{tx_hash}" if tx_hash else None,
            timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc) if timestamp else datetime.now(timezone.utc),
            verified=True,
            raw_data={
                "tx_hash": tx_hash,
                "blockchain": blockchain,
                "symbol": symbol,
                "alert_symbol": alert_symbol,
                "amount": amount,
                "amount_usd": amount_usd,
                "from_owner": from_owner,
                "to_owner": to_owner,
                "from_name": from_name,
                "to_name": to_name,
                "subtype": subtype,
            },
            engagement=engagement,
        )

    def _classify_whale_move(self, from_owner: str, to_owner: str) -> str:
        """
        Classify the whale move type.

        Returns:
            - "exchange_inflow": Moving to exchange (potential sell)
            - "exchange_outflow": Moving from exchange (accumulation)
            - "exchange_transfer": Between exchanges
            - "wallet_transfer": Between wallets
        """
        from_is_exchange = from_owner == "exchange"
        to_is_exchange = to_owner == "exchange"

        if from_is_exchange and not to_is_exchange:
            return "exchange_outflow"
        elif not from_is_exchange and to_is_exchange:
            return "exchange_inflow"
        elif from_is_exchange and to_is_exchange:
            return "exchange_transfer"
        else:
            return "wallet_transfer"

    async def fetch_items(self) -> List[NewsItem]:
        """
        Fetch queued whale alerts.

        For WebSocket mode, this drains the alert queue.
        Alerts are pushed in real-time via the WebSocket.
        """
        if not self.api_key:
            return []

        items = []

        # Drain the queue (non-blocking)
        while not self._alert_queue.empty():
            try:
                item = self._alert_queue.get_nowait()
                items.append(item)
            except asyncio.QueueEmpty:
                break

        return items

    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and not self._ws.closed


if __name__ == "__main__":
    # Test the whale alert WebSocket
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    async def test():
        source = WhaleAlertSource(min_usd_value=500_000)  # Lower threshold for testing

        print(f"Source: {source.source_name}")
        print(f"Credibility: {source.source_credibility}")
        print(f"API Key: {'set' if source.api_key else 'NOT SET'}")

        if not source.api_key:
            print("\nSet WHALE_ALERT_API_KEY environment variable to test")
            return

        # Set up callback to print alerts
        async def on_alert(item: NewsItem):
            print(f"\n  ALERT: {item.title}")
            print(f"         {item.content}")
            print(f"         Sentiment: {item.engagement}")

        source.set_alert_callback(on_alert)

        print("\nStarting WebSocket connection...")
        print("Listening for whale alerts (Ctrl+C to stop)...")

        await source.start_websocket()

        try:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            await source.stop_websocket()

        print(f"\nHealth status: {source.get_health_status()}")

    asyncio.run(test())
