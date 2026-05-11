"""
Coinbase REST API Trade Fetcher

Fetches recent trades via REST API for symbols that aren't receiving
websocket trade data. This ensures all configured symbols get feature
calculation even when websocket market_trades channel is unreliable.

The Coinbase Advanced Trade API endpoint:
GET https://api.coinbase.com/api/v3/brokerage/products/{product_id}/ticker
Returns the latest trade info including price, size, time, and 24h stats.
"""

import asyncio
import aiohttp
from datetime import datetime, timezone
from typing import Optional, Dict, Set, List, Callable
from dateutil import parser as dateutil_parser
from collections import defaultdict

from loguru import logger


class RestTradeFetcher:
    """
    Fetches trade data via REST API for symbols with sparse websocket data.

    Tracks which symbols receive websocket trades and periodically fetches
    REST data for symbols that haven't received trades recently.
    """

    BASE_URL = "https://api.coinbase.com/api/v3/brokerage"

    def __init__(
        self,
        auth,
        symbols: List[str],
        trade_callback: Callable,
        fetch_interval: int = 60,  # seconds between REST fetches
        inactivity_threshold: int = 120,  # seconds without WS trade before REST fetch
    ):
        """
        Initialize REST trade fetcher.

        Args:
            auth: CoinbaseAuth instance for API authentication
            symbols: List of symbols to monitor
            trade_callback: Async callback function(symbol, price, size, side, timestamp)
            fetch_interval: Seconds between REST fetch cycles
            inactivity_threshold: Seconds of WS inactivity before triggering REST fetch
        """
        self.auth = auth
        self.symbols = set(symbols)
        self.trade_callback = trade_callback
        self.fetch_interval = fetch_interval
        self.inactivity_threshold = inactivity_threshold

        # Track last websocket trade time per symbol
        self._last_ws_trade: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))

        # Track last REST fetch time per symbol
        self._last_rest_fetch: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))

        # Track last known price per symbol (to detect actual trades vs stale data)
        self._last_price: Dict[str, float] = {}

        # Statistics
        self._stats = {
            'rest_fetches': 0,
            'rest_trades_processed': 0,
            'rest_errors': 0,
            'symbols_with_ws_data': 0,
            'symbols_needing_rest': 0,
        }

        # Running state
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(f"RestTradeFetcher initialized for {len(symbols)} symbols")

    def record_ws_trade(self, symbol: str):
        """
        Record that a websocket trade was received for this symbol.
        Called by the websocket handler whenever a trade arrives.
        """
        self._last_ws_trade[symbol] = datetime.now(timezone.utc)

    def get_symbols_needing_rest(self) -> Set[str]:
        """
        Get symbols that need REST data due to WS inactivity.
        """
        now = datetime.now(timezone.utc)
        threshold = self.inactivity_threshold

        needing_rest = set()
        for symbol in self.symbols:
            last_trade = self._last_ws_trade[symbol]
            # Check if symbol has been inactive
            if last_trade == datetime.min.replace(tzinfo=timezone.utc):
                # Never received WS trade
                needing_rest.add(symbol)
            elif (now - last_trade).total_seconds() > threshold:
                # Haven't received WS trade recently
                needing_rest.add(symbol)

        return needing_rest

    async def start(self):
        """Start the REST fetcher loop."""
        self._running = True
        self._session = aiohttp.ClientSession()

        logger.info(f"REST trade fetcher started (interval={self.fetch_interval}s, threshold={self.inactivity_threshold}s)")

        try:
            while self._running:
                await self._fetch_cycle()
                await asyncio.sleep(self.fetch_interval)
        except asyncio.CancelledError:
            logger.info("REST trade fetcher cancelled")
        finally:
            if self._session:
                await self._session.close()

    async def stop(self):
        """Stop the REST fetcher."""
        self._running = False
        if self._session:
            await self._session.close()
        logger.info(f"REST trade fetcher stopped. Stats: {self._stats}")

    async def _fetch_cycle(self):
        """
        One cycle of REST fetches for inactive symbols.
        """
        symbols_needing_rest = self.get_symbols_needing_rest()

        # Update stats
        self._stats['symbols_with_ws_data'] = len(self.symbols) - len(symbols_needing_rest)
        self._stats['symbols_needing_rest'] = len(symbols_needing_rest)

        if not symbols_needing_rest:
            return

        # Log periodically (every 10 cycles)
        if self._stats['rest_fetches'] % 10 == 0:
            logger.info(
                f"[REST] WS active: {self._stats['symbols_with_ws_data']}, "
                f"needing REST: {len(symbols_needing_rest)}, "
                f"total fetches: {self._stats['rest_fetches']}"
            )

        # Fetch in batches to avoid rate limits
        batch_size = 10
        symbols_list = list(symbols_needing_rest)

        for i in range(0, len(symbols_list), batch_size):
            batch = symbols_list[i:i + batch_size]

            # Fetch batch concurrently
            tasks = [self._fetch_ticker(symbol) for symbol in batch]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Small delay between batches to respect rate limits
            if i + batch_size < len(symbols_list):
                await asyncio.sleep(0.2)

    async def _fetch_ticker(self, symbol: str):
        """
        Fetch ticker data for a single symbol via REST API.

        Uses the ticker endpoint which includes last trade info.
        """
        try:
            path = f"/api/v3/brokerage/products/{symbol}/ticker"
            url = f"https://api.coinbase.com{path}"
            headers = self.auth.get_auth_headers(method="GET", path=path)

            async with self._session.get(url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    if response.status != 404:  # Don't log 404s for invalid symbols
                        logger.warning(f"[REST] {symbol} fetch failed: {response.status} - {error_text[:100]}")
                    self._stats['rest_errors'] += 1
                    return

                data = await response.json()
                self._stats['rest_fetches'] += 1

                # Extract trade info from ticker
                trades = data.get('trades', [])
                if trades:
                    # Process the most recent trade
                    trade = trades[0]
                    await self._process_rest_trade(symbol, trade)
                else:
                    # Fall back to ticker price/size if no trades array
                    price = data.get('price')
                    size = data.get('size', '0.001')  # Default small size
                    trade_time = data.get('time')
                    side = data.get('side', 'BUY')

                    if price and trade_time:
                        await self._process_rest_trade(symbol, {
                            'price': price,
                            'size': size,
                            'time': trade_time,
                            'side': side,
                        })

        except asyncio.TimeoutError:
            logger.warning(f"[REST] {symbol} fetch timed out")
            self._stats['rest_errors'] += 1
        except Exception as e:
            logger.error(f"[REST] {symbol} fetch error: {e}")
            self._stats['rest_errors'] += 1

    async def _process_rest_trade(self, symbol: str, trade: dict):
        """
        Process a trade received via REST API.

        Only processes if it looks like new data (different price from last seen).
        """
        try:
            price = float(trade.get('price', 0))
            size = float(trade.get('size', 0))
            side = trade.get('side', 'BUY').upper()
            time_str = trade.get('time', '')

            if not price or not time_str:
                return

            # Parse timestamp
            try:
                timestamp = dateutil_parser.isoparse(time_str)
            except:
                timestamp = datetime.now(timezone.utc)

            # Check if this looks like new data
            last_price = self._last_price.get(symbol)
            if last_price is not None and abs(price - last_price) < 1e-10:
                # Same price as last fetch, likely stale - skip
                return

            self._last_price[symbol] = price
            self._last_rest_fetch[symbol] = datetime.now(timezone.utc)
            self._stats['rest_trades_processed'] += 1

            # Call the callback to process through feature engine
            await self.trade_callback(
                symbol=symbol,
                price=price,
                size=size,
                side=side,
                timestamp=timestamp
            )

        except Exception as e:
            logger.error(f"[REST] Error processing {symbol} trade: {e}")

    def get_statistics(self) -> dict:
        """Get fetcher statistics."""
        return {
            **self._stats,
            'fetch_interval': self.fetch_interval,
            'inactivity_threshold': self.inactivity_threshold,
            'total_symbols': len(self.symbols),
        }
