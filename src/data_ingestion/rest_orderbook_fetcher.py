"""
Coinbase REST API Order Book Fetcher

Fetches order book depth via REST API to provide bid/ask depth data
for feature calculation. This supplements the ticker data (which only
provides best bid/ask) with actual depth information.

Uses the PUBLIC Coinbase Exchange API (no authentication required):
GET https://api.exchange.coinbase.com/products/{product_id}/book?level=2

Returns bid/ask levels with prices and quantities.
"""

import asyncio
import aiohttp
from datetime import datetime, timezone
from typing import Optional, Dict, List, Callable
from collections import defaultdict

from loguru import logger


class RestOrderBookFetcher:
    """
    Fetches order book depth via REST API for all symbols.

    Polls order book depth periodically to provide bid_depth and ask_depth
    for the order_book_depth_ratio feature.

    Uses the PUBLIC Coinbase Exchange API - no authentication required.
    """

    # Public Exchange API - no auth needed
    BASE_URL = "https://api.exchange.coinbase.com"

    def __init__(
        self,
        symbols: List[str],
        depth_callback: Callable,
        fetch_interval: int = 30,  # seconds between full cycles
        levels: int = 2,  # level=2 gets top 50 bids/asks
        auth=None,  # Not used, kept for compatibility
    ):
        """
        Initialize REST order book fetcher.

        Args:
            symbols: List of symbols to fetch order books for
            depth_callback: Callback function(symbol, bid_depth, ask_depth, best_bid, best_ask)
            fetch_interval: Seconds between fetch cycles
            levels: Book level (2 = top 50 bids/asks)
            auth: Not used (public API), kept for compatibility
        """
        self.symbols = list(symbols)
        self.depth_callback = depth_callback
        self.fetch_interval = fetch_interval
        self.levels = levels

        # Statistics
        self._stats = {
            'cycles_completed': 0,
            'fetches_successful': 0,
            'fetches_failed': 0,
            'symbols_with_depth': 0,
            'total_bid_depth_sum': 0.0,
            'total_ask_depth_sum': 0.0,
        }

        # Running state
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(f"RestOrderBookFetcher initialized for {len(symbols)} symbols (levels={levels})")

    async def start(self):
        """Start the order book fetcher loop."""
        self._running = True
        self._session = aiohttp.ClientSession()

        logger.info(f"REST order book fetcher started (interval={self.fetch_interval}s, levels={self.levels})")

        try:
            while self._running:
                await self._fetch_cycle()
                await asyncio.sleep(self.fetch_interval)
        except asyncio.CancelledError:
            logger.info("REST order book fetcher cancelled")
        finally:
            if self._session:
                await self._session.close()

    async def stop(self):
        """Stop the order book fetcher."""
        self._running = False
        if self._session:
            await self._session.close()
        logger.info(f"REST order book fetcher stopped. Stats: {self._stats}")

    async def _fetch_cycle(self):
        """
        One cycle of order book fetches for all symbols.
        """
        start_time = datetime.now(timezone.utc)
        successful = 0
        failed = 0

        # Fetch in batches to avoid rate limits
        # Coinbase rate limit is ~30 requests/second for authenticated
        batch_size = 20

        for i in range(0, len(self.symbols), batch_size):
            batch = self.symbols[i:i + batch_size]

            # Fetch batch concurrently
            tasks = [self._fetch_order_book(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if result is True:
                    successful += 1
                else:
                    failed += 1

            # Small delay between batches to respect rate limits
            if i + batch_size < len(self.symbols):
                await asyncio.sleep(0.1)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        self._stats['cycles_completed'] += 1
        self._stats['fetches_successful'] += successful
        self._stats['fetches_failed'] += failed
        self._stats['symbols_with_depth'] = successful

        # Log periodically
        if self._stats['cycles_completed'] % 10 == 1:
            logger.info(
                f"[ORDER_BOOK_REST] Cycle {self._stats['cycles_completed']}: "
                f"{successful}/{len(self.symbols)} successful in {elapsed:.1f}s"
            )

    async def _fetch_order_book(self, symbol: str) -> bool:
        """
        Fetch order book depth for a single symbol via public REST API.

        Uses Coinbase Exchange API (no auth required):
        GET https://api.exchange.coinbase.com/products/{product_id}/book?level=2

        Returns True if successful, False otherwise.
        """
        try:
            # Public Exchange API endpoint - no auth needed
            url = f"{self.BASE_URL}/products/{symbol}/book"
            params = {'level': str(self.levels)}

            async with self._session.get(url, params=params, timeout=10) as response:
                if response.status != 200:
                    # Don't log 404s for invalid symbols
                    if response.status != 404:
                        if self._stats['cycles_completed'] < 3:  # Only log first few cycles
                            logger.debug(f"[ORDER_BOOK_REST] {symbol} fetch failed: {response.status}")
                    return False

                data = await response.json()

                # Public API format: {"bids": [[price, size, num_orders], ...], "asks": [...]}
                bids = data.get('bids', [])
                asks = data.get('asks', [])

                if not bids and not asks:
                    return False

                # Calculate total depth (each entry is [price, size, num_orders])
                bid_depth = sum(float(bid[1]) for bid in bids if len(bid) >= 2)
                ask_depth = sum(float(ask[1]) for ask in asks if len(ask) >= 2)

                # Get best bid/ask
                best_bid = float(bids[0][0]) if bids and len(bids[0]) >= 1 else 0.0
                best_ask = float(asks[0][0]) if asks and len(asks[0]) >= 1 else 0.0

                # Convert to (price, quantity) tuples for storage - all 50 levels
                bids_levels = [(float(b[0]), float(b[1])) for b in bids[:50] if len(b) >= 2]
                asks_levels = [(float(a[0]), float(a[1])) for a in asks[:50] if len(a) >= 2]

                # Track totals for stats
                self._stats['total_bid_depth_sum'] += bid_depth
                self._stats['total_ask_depth_sum'] += ask_depth

                # Call callback to update feature engine AND store full L2
                await self.depth_callback(
                    symbol=symbol,
                    bid_depth=bid_depth,
                    ask_depth=ask_depth,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bids=bids_levels,
                    asks=asks_levels
                )

                return True

        except asyncio.TimeoutError:
            return False
        except Exception as e:
            if self._stats['cycles_completed'] < 3:  # Only log first few cycles
                logger.debug(f"[ORDER_BOOK_REST] {symbol} error: {e}")
            return False

    async def fetch_single(self, symbol: str) -> bool:
        """
        Fetch order book for a single symbol on-demand.

        Used by watch mode for fast polling of high-priority symbols.
        This method can be called outside of the regular fetch cycle.

        Args:
            symbol: Trading pair to fetch

        Returns:
            True if successful
        """
        if not self._session:
            self._session = aiohttp.ClientSession()

        return await self._fetch_order_book(symbol)

    def get_statistics(self) -> dict:
        """Get fetcher statistics."""
        return {
            **self._stats,
            'fetch_interval': self.fetch_interval,
            'levels': self.levels,
            'total_symbols': len(self.symbols),
        }
