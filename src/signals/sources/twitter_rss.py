"""
Twitter RSS Source

Monitors key crypto Twitter accounts via Nitter RSS feeds.
Free alternative to Twitter API.

Priority: Tier 2
Cost: Free (Nitter RSS)
"""

import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Set
import aiohttp
from loguru import logger

from .base import BaseSource, NewsItem


class TwitterRSSSource(BaseSource):
    """
    Monitors Twitter/X accounts via Nitter RSS feeds.

    Nitter is a free, open-source Twitter frontend that provides RSS feeds
    for any public Twitter account. No API key required.

    Key accounts monitored:
    - @coinbase, @CoinbaseAssets (listing announcements)
    - @binance, @kaborakis (exchanges)
    - @whale_alert, @unusual_whales (whale tracking)
    - @WatcherGuru, @CoinDesk (news)
    """

    # Nitter instances (multiple for redundancy)
    NITTER_INSTANCES = [
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://nitter.projectsegfau.lt",
    ]

    # Key accounts to monitor (handle -> category)
    DEFAULT_ACCOUNTS = {
        # Exchange official accounts (listings)
        "CoinbaseMarkets": "exchange", # PRIMARY: Coinbase listing announcements
        "coinbase": "exchange",        # Coinbase main
        "CoinbaseAssets": "exchange",  # Coinbase assets (redirects to CoinbaseMarkets)
        "binance": "exchange",         # Binance main
        "krakenfx": "exchange",        # Kraken

        # Whale tracking
        "whale_alert": "whale",
        "unusual_whales": "whale",

        # Major crypto news
        "CoinDesk": "news",
        "Cointelegraph": "news",
        "WatcherGuru": "news",

        # Influential accounts
        "VitalikButerin": "project",
        "solana": "project",
    }

    # Keywords for classification
    LISTING_KEYWORDS = ["listed", "listing", "now available", "will list", "adds", "adding"]
    PARTNERSHIP_KEYWORDS = ["partnership", "partners with", "collaborating", "integration"]
    AIRDROP_KEYWORDS = ["airdrop", "claim", "distribution"]

    # Common crypto symbols to look for (top 100 + common ones)
    KNOWN_SYMBOLS = {
        "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "SHIB", "DOT", "LINK",
        "MATIC", "LTC", "UNI", "ATOM", "XLM", "ETC", "FIL", "NEAR", "APT", "ARB",
        "OP", "SUI", "INJ", "IMX", "SEI", "TIA", "PEPE", "WIF", "BONK", "FLOKI",
        "RENDER", "FET", "JASMY", "GRT", "STX", "ALGO", "SAND", "MANA", "AXS",
        "AAVE", "MKR", "SNX", "CRV", "COMP", "YFI", "SUSHI", "1INCH", "ENS",
        "LDO", "RPL", "BLUR", "PYTH", "JTO", "JUP", "W", "STRK", "ZRO",
        "TRUMP", "MELANIA", "FARTCOIN", "AI16Z", "GOAT", "PNUT", "ACT",
    }

    # Name to symbol mapping for common cryptos mentioned by name
    NAME_TO_SYMBOL = {
        "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "ripple": "XRP",
        "dogecoin": "DOGE", "cardano": "ADA", "avalanche": "AVAX", "polkadot": "DOT",
        "chainlink": "LINK", "polygon": "MATIC", "litecoin": "LTC", "uniswap": "UNI",
        "cosmos": "ATOM", "stellar": "XLM", "filecoin": "FIL", "arbitrum": "ARB",
        "optimism": "OP", "injective": "INJ", "aptos": "APT", "sui": "SUI",
        "celestia": "TIA", "render": "RENDER", "aave": "AAVE", "maker": "MKR",
    }

    def __init__(
        self,
        accounts: Optional[Dict[str, str]] = None,
        nitter_instance: Optional[str] = None,
        request_timeout: int = 30,
    ):
        """
        Initialize Twitter RSS source.

        Args:
            accounts: Dict of {twitter_handle: category} to monitor
            nitter_instance: Specific Nitter instance URL to use
            request_timeout: Timeout for requests
        """
        super().__init__(
            rate_limit_per_minute=30,  # Be gentle with Nitter instances
            request_timeout=request_timeout,
        )

        self.accounts = accounts or self.DEFAULT_ACCOUNTS
        self.nitter_instance = nitter_instance or self.NITTER_INSTANCES[0]
        self._current_instance_idx = 0

        # Track seen tweets
        self._seen_tweets: Set[str] = set()

        logger.info(f"TwitterRSS initialized with {len(self.accounts)} accounts, using {self.nitter_instance}")

    @property
    def source_name(self) -> str:
        return "twitter_rss"

    @property
    def source_credibility(self) -> float:
        return 0.7  # Verified accounts have good signal

    async def fetch_items(self) -> List[NewsItem]:
        """Fetch tweets from all monitored accounts."""
        items = []

        async with aiohttp.ClientSession() as session:
            # Fetch from each account
            for handle, category in self.accounts.items():
                try:
                    account_items = await self._fetch_account(session, handle, category)
                    items.extend(account_items)

                    # Small delay between accounts
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.debug(f"Error fetching @{handle}: {e}")

        return items

    async def _fetch_account(
        self,
        session: aiohttp.ClientSession,
        handle: str,
        category: str
    ) -> List[NewsItem]:
        """Fetch RSS feed for a single account."""
        url = f"{self.nitter_instance}/{handle}/rss"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)",
            }

            async with session.get(url, headers=headers, timeout=self.request_timeout) as response:
                if response.status == 404:
                    logger.debug(f"Account @{handle} not found")
                    return []

                if response.status != 200:
                    # Try fallback instance
                    await self._try_fallback_instance()
                    return []

                text = await response.text()
                return self._parse_rss(text, handle, category)

        except aiohttp.ClientError as e:
            logger.debug(f"Network error for @{handle}: {e}")
            await self._try_fallback_instance()
            return []

    async def _try_fallback_instance(self):
        """Switch to next Nitter instance on failure."""
        self._current_instance_idx = (self._current_instance_idx + 1) % len(self.NITTER_INSTANCES)
        self.nitter_instance = self.NITTER_INSTANCES[self._current_instance_idx]
        logger.info(f"Switched to Nitter instance: {self.nitter_instance}")

    def _parse_rss(self, xml_text: str, handle: str, category: str) -> List[NewsItem]:
        """Parse RSS feed and extract relevant tweets."""
        items = []

        try:
            root = ET.fromstring(xml_text)

            for item in root.findall('.//item'):
                try:
                    news_item = self._parse_tweet(item, handle, category)
                    if news_item:
                        items.append(news_item)
                except Exception as e:
                    logger.debug(f"Error parsing tweet from @{handle}: {e}")

        except ET.ParseError as e:
            logger.debug(f"RSS parse error for @{handle}: {e}")

        return items

    def _parse_tweet(self, item: ET.Element, handle: str, category: str) -> Optional[NewsItem]:
        """Parse a single tweet from RSS item."""
        # Extract tweet content
        title_elem = item.find('title')
        title = title_elem.text if title_elem is not None and title_elem.text else ""

        # Nitter puts the full tweet in description
        desc_elem = item.find('description')
        description = desc_elem.text if desc_elem is not None and desc_elem.text else ""

        # Clean HTML from description
        content = re.sub(r'<[^>]+>', '', description)
        content = content.strip()

        # Get link
        link_elem = item.find('link')
        url = link_elem.text if link_elem is not None and link_elem.text else ""

        # Get publication date
        pub_date_elem = item.find('pubDate')
        timestamp = self._parse_date(pub_date_elem.text if pub_date_elem is not None else None)

        # Generate unique ID
        tweet_id = url.split('/')[-1] if url else f"{handle}:{title[:50]}"

        if tweet_id in self._seen_tweets:
            return None

        self._seen_tweets.add(tweet_id)

        # Limit cache size
        if len(self._seen_tweets) > 10000:
            self._seen_tweets = set(list(self._seen_tweets)[-5000:])

        # Skip old tweets (more than 24 hours)
        if (datetime.now(timezone.utc) - timestamp).total_seconds() > 86400:
            return None

        # Classify event type
        event_type = self._classify_tweet(content, category)

        # Extract crypto symbols from content
        symbols = self._extract_symbols(content)

        # Skip tweets that aren't relevant (no symbols and not a key event type)
        if event_type == "mention" and category not in ["exchange", "whale"]:
            if not symbols:  # Only skip if no symbols detected
                return None

        # Extract engagement from Nitter (if available in RSS)
        engagement = {}

        return NewsItem(
            source=self.source_name,
            event_type=event_type,
            title=title[:200] if title else content[:200],
            content=content,
            url=url,
            timestamp=timestamp,
            symbols=symbols,
            entity_confidence=0.9 if symbols else 0.3,
            author=f"@{handle}",
            verified=category in ["exchange", "whale"],
            engagement=engagement,
            raw_data={
                "handle": handle,
                "category": category,
                "tweet_id": tweet_id,
            },
        )

    def _parse_date(self, date_str: Optional[str]) -> datetime:
        """Parse date from RSS pubDate format."""
        if not date_str:
            return datetime.now(timezone.utc)

        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
            "%Y-%m-%dT%H:%M:%S%z",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

        return datetime.now(timezone.utc)

    def _extract_symbols(self, content: str) -> List[str]:
        """Extract crypto symbols from tweet content."""
        symbols = set()
        content_upper = content.upper()
        content_lower = content.lower()

        # Look for $SYMBOL cashtag format (most reliable)
        cashtags = re.findall(r'\$([A-Z]{2,10})\b', content_upper)
        for tag in cashtags:
            if tag in self.KNOWN_SYMBOLS:
                symbols.add(f"{tag}-USD")

        # Look for standalone symbols (less reliable, require known list)
        words = re.findall(r'\b([A-Z]{2,6})\b', content_upper)
        for word in words:
            if word in self.KNOWN_SYMBOLS and len(word) >= 3:
                symbols.add(f"{word}-USD")

        # Look for crypto names
        for name, symbol in self.NAME_TO_SYMBOL.items():
            if name in content_lower:
                symbols.add(f"{symbol}-USD")

        return list(symbols)

    def _classify_tweet(self, content: str, category: str) -> str:
        """Classify tweet event type."""
        content_lower = content.lower()

        # Check for specific event types
        for keyword in self.LISTING_KEYWORDS:
            if keyword in content_lower:
                return "listing"

        for keyword in self.PARTNERSHIP_KEYWORDS:
            if keyword in content_lower:
                return "partnership"

        for keyword in self.AIRDROP_KEYWORDS:
            if keyword in content_lower:
                return "airdrop"

        # Default based on account category
        if category == "whale":
            return "whale_move"
        elif category == "exchange":
            return "mention"  # Exchange mention without specific event
        else:
            return "mention"


if __name__ == "__main__":
    # Test the Twitter RSS source
    import asyncio

    async def test():
        source = TwitterRSSSource()

        print(f"Source: {source.source_name}")
        print(f"Credibility: {source.source_credibility}")
        print(f"Nitter instance: {source.nitter_instance}")
        print(f"Monitoring {len(source.accounts)} accounts")

        print("\nFetching tweets...")
        items = await source.fetch_with_rate_limit()

        print(f"\nFound {len(items)} relevant tweets:")
        for item in items[:10]:
            print(f"  - [{item.event_type}] @{item.raw_data.get('handle')}: {item.title[:60]}...")
            print(f"    Time: {item.timestamp}")
            print()

        print(f"Health status: {source.get_health_status()}")

    asyncio.run(test())
