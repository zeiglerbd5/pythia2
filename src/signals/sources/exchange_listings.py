"""
Exchange Listings Source

Monitors Binance for new listing announcements via their API.
Coinbase listings are monitored via Twitter RSS (@CoinbaseMarkets).

Priority: Tier 1 (highest value signal)
Cost: Free (Binance API)
"""

import asyncio
import re
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Set
import aiohttp
from loguru import logger

from .base import BaseSource, NewsItem


class ExchangeListingsSource(BaseSource):
    """
    Monitors exchange listing announcements.

    Sources:
    - Binance Announcements API (catalogId=48 = New Cryptocurrency Listing)
    - Coinbase listings via Twitter RSS (@CoinbaseMarkets) - handled separately

    Listing announcements are the highest-value signal:
    - Typically cause 20-100%+ price spikes
    - Very short window to trade (minutes to hours)
    - High signal-to-noise ratio
    """

    # Binance announcements API
    BINANCE_API_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
    BINANCE_LISTING_CATALOG_ID = 48  # "New Cryptocurrency Listing" category

    # Keywords indicating listing announcements
    LISTING_KEYWORDS = [
        "now available",
        "listed on",
        "will list",
        "adds support",
        "adding",
        "launches",
        "new asset",
        "trading pairs",
        "perpetual contract",
        "futures listing",
    ]

    DELISTING_KEYWORDS = [
        "delisting",
        "delist",
        "removing",
        "will suspend",
        "trading suspension",
    ]

    # Common crypto symbols to look for
    KNOWN_SYMBOLS = {
        "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "SHIB", "DOT", "LINK",
        "MATIC", "LTC", "UNI", "ATOM", "XLM", "ETC", "FIL", "NEAR", "APT", "ARB",
        "OP", "SUI", "INJ", "IMX", "SEI", "TIA", "PEPE", "WIF", "BONK", "FLOKI",
        "RENDER", "FET", "JASMY", "GRT", "STX", "ALGO", "SAND", "MANA", "AXS",
        "AAVE", "MKR", "SNX", "CRV", "COMP", "YFI", "SUSHI", "1INCH", "ENS",
        "LDO", "RPL", "BLUR", "PYTH", "JTO", "JUP", "W", "STRK", "ZRO",
        "TRUMP", "MELANIA", "FARTCOIN", "AI16Z", "GOAT", "PNUT", "ACT",
    }

    NAME_TO_SYMBOL = {
        "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "ripple": "XRP",
        "dogecoin": "DOGE", "cardano": "ADA", "avalanche": "AVAX", "polkadot": "DOT",
        "chainlink": "LINK", "polygon": "MATIC", "litecoin": "LTC", "uniswap": "UNI",
        "cosmos": "ATOM", "stellar": "XLM", "filecoin": "FIL", "arbitrum": "ARB",
        "optimism": "OP", "injective": "INJ", "aptos": "APT", "sui": "SUI",
        "celestia": "TIA", "render": "RENDER", "aave": "AAVE", "maker": "MKR",
    }

    def __init__(self, request_timeout: int = 30):
        """
        Initialize exchange listings source.

        Uses Binance announcements API for listing detection.
        """
        super().__init__(
            rate_limit_per_minute=10,
            request_timeout=request_timeout,
        )

        # Track seen announcement IDs
        self._seen_ids: Set[int] = set()

        # Cache for deduplication
        self._last_fetch_time: Optional[datetime] = None

    @property
    def source_name(self) -> str:
        return "exchange_listings"

    @property
    def source_credibility(self) -> float:
        return 1.0  # Official exchange announcements

    async def fetch_items(self) -> List[NewsItem]:
        """Fetch listing announcements from Binance API."""
        items = []

        async with aiohttp.ClientSession() as session:
            try:
                binance_items = await self._fetch_binance_listings(session)
                items.extend(binance_items)
                logger.debug(f"[exchange_listings] binance={len(binance_items)} items")
            except Exception as e:
                logger.debug(f"[exchange_listings] binance error: {e}")

        return items

    async def _fetch_binance_listings(self, session: aiohttp.ClientSession) -> List[NewsItem]:
        """Fetch from Binance announcements API."""
        items = []

        try:
            params = {
                "type": 1,
                "catalogId": self.BINANCE_LISTING_CATALOG_ID,
                "pageNo": 1,
                "pageSize": 20,
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; CatalystBot/1.0)",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            }

            async with session.get(
                self.BINANCE_API_URL,
                params=params,
                headers=headers,
                timeout=self.request_timeout,
            ) as response:
                if response.status != 200:
                    logger.debug(f"Binance API returned {response.status}")
                    return []

                data = await response.json()

                # Navigate to articles in the response
                catalogs = data.get("data", {}).get("catalogs", [])
                for catalog in catalogs:
                    for article in catalog.get("articles", []):
                        news_item = self._parse_binance_article(article)
                        if news_item:
                            items.append(news_item)

        except aiohttp.ClientError as e:
            logger.debug(f"Binance API network error: {e}")
        except Exception as e:
            logger.debug(f"Binance API error: {e}")

        return items

    def _parse_binance_article(self, article: Dict) -> Optional[NewsItem]:
        """Parse a Binance announcement article."""
        article_id = article.get("id")
        title = article.get("title", "")
        code = article.get("code", "")
        release_date = article.get("releaseDate", 0)

        # Skip if already seen
        if article_id in self._seen_ids:
            return None
        self._seen_ids.add(article_id)

        # Limit cache size
        if len(self._seen_ids) > 5000:
            self._seen_ids = set(list(self._seen_ids)[-2500:])

        # Parse timestamp (milliseconds)
        try:
            timestamp = datetime.fromtimestamp(release_date / 1000, tz=timezone.utc)
        except:
            timestamp = datetime.now(timezone.utc)

        # Skip old announcements (more than 7 days)
        if (datetime.now(timezone.utc) - timestamp).days > 7:
            return None

        # Classify event type
        event_type = self._classify_event(title.lower())
        if not event_type:
            event_type = "listing"  # Default for this catalog

        # Extract symbols
        symbols = self._extract_symbols(title)

        # Build URL
        url = f"https://www.binance.com/en/support/announcement/{code}"

        return NewsItem(
            source=self.source_name,
            event_type=event_type,
            title=title,
            content=title,  # Binance API only returns title, not full content
            url=url,
            timestamp=timestamp,
            symbols=symbols,
            entity_confidence=0.95 if symbols else 0.5,
            author="Binance",
            verified=True,
            raw_data={
                "article_id": article_id,
                "code": code,
            },
        )

    def _classify_event(self, text: str) -> Optional[str]:
        """
        Classify the event type based on text content.

        Returns:
            Event type string or None if not a relevant event
        """
        text_lower = text.lower()

        # Check for delisting first (higher priority)
        for keyword in self.DELISTING_KEYWORDS:
            if keyword in text_lower:
                return "delisting"

        # Check for listing
        for keyword in self.LISTING_KEYWORDS:
            if keyword in text_lower:
                return "listing"

        return None

    def _extract_symbols(self, content: str) -> List[str]:
        """Extract crypto symbols from content."""
        symbols = set()
        content_upper = content.upper()
        content_lower = content.lower()

        # Look for $SYMBOL cashtag format
        cashtags = re.findall(r'\$([A-Z]{2,10})\b', content_upper)
        for tag in cashtags:
            if tag in self.KNOWN_SYMBOLS:
                symbols.add(f"{tag}-USD")

        # Look for standalone symbols in parentheses (common in listings)
        # e.g., "Katana (KAT)" or "Centrifuge (CFG)"
        # For exchange listings, trust parentheses format even if not in known list
        parens = re.findall(r'\(([A-Z]{2,10})\)', content_upper)
        for sym in parens:
            # Skip common non-crypto terms
            if sym not in {'USD', 'USDT', 'USDC', 'EUR', 'GBP', 'GMT', 'UTC', 'API', 'VIP'}:
                symbols.add(f"{sym}-USD")

        # Look for "XXXUSDT" pattern (Binance perpetual contracts)
        usdt_pairs = re.findall(r'\b([A-Z]{2,10})USDT\b', content_upper)
        for sym in usdt_pairs:
            if sym not in {'USD'}:
                symbols.add(f"{sym}-USD")

        # Look for crypto names
        for name, symbol in self.NAME_TO_SYMBOL.items():
            if name in content_lower:
                symbols.add(f"{symbol}-USD")

        return list(symbols)


if __name__ == "__main__":
    # Test the exchange listings source
    import asyncio

    async def test():
        source = ExchangeListingsSource()

        print(f"Source: {source.source_name}")
        print(f"Credibility: {source.source_credibility}")
        print(f"Health: {source.is_healthy()}")

        print("\nFetching items...")
        items = await source.fetch_with_rate_limit()

        print(f"\nFound {len(items)} items:")
        for item in items[:5]:  # Show first 5
            print(f"  - [{item.event_type}] {item.title[:60]}...")
            print(f"    URL: {item.url}")
            print(f"    Time: {item.timestamp}")
            print()

        print(f"Health status: {source.get_health_status()}")

    asyncio.run(test())
