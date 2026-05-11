"""
CryptoPanic News Source

Fetches crypto news from CryptoPanic API - a news aggregator that pulls from
multiple sources and categorizes by:
- News type (media, blog, reddit)
- Currencies mentioned
- Community sentiment (bullish/bearish votes)

Free tier: Limited requests per day, but sufficient for catalyst detection.
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import aiohttp
from loguru import logger

from .base import BaseSource, NewsItem


class CryptoPanicSource(BaseSource):
    """
    CryptoPanic API news source.

    Fetches cryptocurrency news with sentiment and currency tags.
    API docs: https://cryptopanic.com/developers/api/
    """

    # Event type mapping for catalyst detection
    CATALYST_KEYWORDS = {
        'listing': ['listing', 'listed', 'lists', 'available on', 'trading live', 'now trading'],
        'partnership': ['partner', 'partnership', 'collaboration', 'integrate', 'integration'],
        'airdrop': ['airdrop', 'free tokens', 'token distribution', 'claim'],
        'upgrade': ['upgrade', 'mainnet', 'v2', 'v3', 'hard fork', 'protocol upgrade'],
        'legal': ['sec', 'lawsuit', 'regulation', 'court', 'ruling', 'settlement'],
        'etf': ['etf', 'exchange traded', 'spot etf'],
        'hack': ['hack', 'exploit', 'breach', 'stolen', 'vulnerability'],
        'burn': ['burn', 'burned', 'burning', 'deflation'],
    }

    # Known crypto symbols for extraction from text
    KNOWN_SYMBOLS = {
        "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "SHIB", "DOT", "LINK",
        "MATIC", "LTC", "UNI", "ATOM", "XLM", "ETC", "FIL", "NEAR", "APT", "ARB",
        "OP", "SUI", "INJ", "IMX", "SEI", "TIA", "PEPE", "WIF", "BONK", "FLOKI",
        "RENDER", "FET", "JASMY", "GRT", "STX", "ALGO", "SAND", "MANA", "AXS",
        "AAVE", "MKR", "SNX", "CRV", "COMP", "YFI", "SUSHI", "1INCH", "ENS",
        "LDO", "RPL", "BLUR", "PYTH", "JTO", "JUP", "W", "STRK", "ZRO",
        "TRUMP", "MELANIA", "FARTCOIN", "AI16Z", "GOAT", "PNUT", "ACT",
        "BNB", "TRX", "TON", "HBAR", "VET", "FTM", "THETA", "EOS", "XTZ",
    }

    # Name to symbol mapping
    NAME_TO_SYMBOL = {
        "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "ripple": "XRP",
        "dogecoin": "DOGE", "cardano": "ADA", "avalanche": "AVAX", "polkadot": "DOT",
        "chainlink": "LINK", "polygon": "MATIC", "litecoin": "LTC", "uniswap": "UNI",
        "cosmos": "ATOM", "stellar": "XLM", "filecoin": "FIL", "arbitrum": "ARB",
        "optimism": "OP", "injective": "INJ", "aptos": "APT", "sui": "SUI",
        "celestia": "TIA", "render": "RENDER", "aave": "AAVE", "maker": "MKR",
        "shiba": "SHIB", "pepe": "PEPE", "bonk": "BONK", "floki": "FLOKI",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        filter_currencies: Optional[List[str]] = None,
        include_news: bool = True,
        include_media: bool = True,
        include_reddit: bool = False,
    ):
        """
        Initialize CryptoPanic source.

        Args:
            api_key: CryptoPanic API key (from env CRYPTOPANIC_API_KEY if not provided)
            filter_currencies: List of currencies to filter (e.g., ['BTC', 'ETH'])
            include_news: Include news posts
            include_media: Include media/video posts
            include_reddit: Include Reddit posts
        """
        # Lower rate limit for free tier (be conservative)
        super().__init__(rate_limit_per_minute=10)

        self.api_key = api_key or os.getenv('CRYPTOPANIC_API_KEY')
        if not self.api_key:
            logger.warning("CryptoPanic API key not set - source will be disabled")

        # API v2 with developer plan path
        self.base_url = "https://cryptopanic.com/api/developer/v2"
        self.filter_currencies = filter_currencies
        self.include_news = include_news
        self.include_media = include_media
        self.include_reddit = include_reddit

        # Track seen items to avoid duplicates
        self._seen_ids: set = set()

    @property
    def source_name(self) -> str:
        return "cryptopanic"

    @property
    def source_credibility(self) -> float:
        # CryptoPanic aggregates from various sources, credibility varies
        # We'll adjust per-item based on the original source
        return 0.7

    def _extract_symbols(self, title: str, content: str) -> List[str]:
        """Extract crypto symbols from text when instruments not provided."""
        import re
        symbols = set()
        text = f"{title} {content}"
        text_upper = text.upper()
        text_lower = text.lower()

        # Look for $SYMBOL cashtag format
        cashtags = re.findall(r'\$([A-Z]{2,10})\b', text_upper)
        for tag in cashtags:
            if tag in self.KNOWN_SYMBOLS:
                symbols.add(f"{tag}-USD")

        # Look for symbols in parentheses e.g., "Bitcoin (BTC)"
        parens = re.findall(r'\(([A-Z]{2,6})\)', text_upper)
        for sym in parens:
            if sym in self.KNOWN_SYMBOLS:
                symbols.add(f"{sym}-USD")

        # Look for crypto names
        for name, symbol in self.NAME_TO_SYMBOL.items():
            if name in text_lower:
                symbols.add(f"{symbol}-USD")

        return list(symbols)

    def _classify_event_type(self, title: str, content: str) -> str:
        """Classify news into catalyst categories."""
        text = f"{title} {content}".lower()

        for event_type, keywords in self.CATALYST_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return event_type

        return "news"  # Generic news

    def _extract_sentiment(self, votes: Dict[str, int]) -> float:
        """Convert vote counts to sentiment score (-1 to 1)."""
        positive = votes.get('positive', 0) + votes.get('liked', 0)
        negative = votes.get('negative', 0) + votes.get('disliked', 0)
        total = positive + negative

        if total == 0:
            return 0.0

        return (positive - negative) / total

    async def fetch_items(self) -> List[NewsItem]:
        """Fetch news from CryptoPanic API."""
        if not self.api_key:
            return []

        items = []

        try:
            # Build request URL
            params = {
                'auth_token': self.api_key,
                'public': 'true',
            }

            # Note: v2 API 'kind' param only accepts single value (news, media, all)
            # Default is 'all' which gets everything

            # Filter by currencies if specified
            if self.filter_currencies:
                params['currencies'] = ','.join(self.filter_currencies)

            url = f"{self.base_url}/posts/"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=self.request_timeout) as response:
                    if response.status != 200:
                        logger.warning(f"CryptoPanic API error: {response.status}")
                        return []

                    data = await response.json()

                    for post in data.get('results', []):
                        # Create unique ID from title + timestamp (id may be None on Developer plan)
                        title = post.get('title', '')
                        pub_at = post.get('published_at', '')
                        post_id = post.get('id') or f"{title[:50]}_{pub_at}"

                        # Skip duplicates
                        if post_id in self._seen_ids:
                            continue
                        self._seen_ids.add(post_id)

                        # Parse timestamp
                        try:
                            timestamp = datetime.fromisoformat(
                                post.get('published_at', '').replace('Z', '+00:00')
                            )
                        except:
                            timestamp = datetime.now(timezone.utc)

                        # Extract currencies from 'instruments' (v2 API)
                        currencies = []
                        for instrument in post.get('instruments', []):
                            code = instrument.get('code')
                            if code:
                                currencies.append(f"{code}-USD")

                        # Get title and content (v2 uses 'description' not 'body')
                        title = post.get('title', '')
                        content = post.get('description', '') or title

                        # If no instruments, try to extract symbols from text
                        if not currencies:
                            currencies = self._extract_symbols(title, content)

                        # Classify event type
                        event_type = self._classify_event_type(title, content)

                        # Extract sentiment from votes
                        votes = post.get('votes', {})
                        sentiment = self._extract_sentiment(votes)

                        # Get source info (v2 has source object)
                        source_obj = post.get('source', {})
                        source_domain = source_obj.get('domain', '')
                        credibility = self._get_domain_credibility(source_domain)

                        item = NewsItem(
                            source=self.source_name,
                            event_type=event_type,
                            title=title,
                            content=content,
                            url=post.get('original_url') or post.get('url'),
                            timestamp=timestamp,
                            symbols=currencies,
                            entity_confidence=0.9 if currencies else 0.3,
                            engagement={
                                'votes': votes,
                                'sentiment': sentiment,
                                'panic_score': post.get('panic_score'),
                            },
                            author=source_obj.get('title', source_domain),
                            verified=source_obj.get('type') in ('feed', 'media'),
                            raw_data=post,
                        )

                        items.append(item)

            # Keep seen_ids from growing too large
            if len(self._seen_ids) > 10000:
                self._seen_ids = set(list(self._seen_ids)[-5000:])

            logger.debug(f"CryptoPanic fetched {len(items)} items")
            return items

        except asyncio.TimeoutError:
            logger.warning("CryptoPanic API timeout")
            return []
        except Exception as e:
            logger.error(f"CryptoPanic fetch error: {e}")
            raise

    def _get_domain_credibility(self, domain: str) -> float:
        """Get credibility score based on source domain."""
        domain = domain.lower()

        # Tier 1: Official sources
        if any(x in domain for x in ['coinbase', 'binance', 'kraken', 'sec.gov']):
            return 1.0

        # Tier 2: Major crypto news
        if any(x in domain for x in ['coindesk', 'cointelegraph', 'decrypt', 'theblock']):
            return 0.85

        # Tier 3: General crypto media
        if any(x in domain for x in ['bitcoinist', 'newsbtc', 'cryptopotato', 'beincrypto']):
            return 0.7

        # Tier 4: Aggregators and blogs
        return 0.5


class CryptoPanicFreeSource(BaseSource):
    """
    Alternative free crypto news source using free-crypto-news API.

    No API key required. Falls back to this if CryptoPanic key not available.
    GitHub: https://github.com/nirholas/free-crypto-news
    """

    def __init__(self):
        super().__init__(rate_limit_per_minute=30)
        self.base_url = "https://api.free-crypto-news.com"  # Placeholder - verify actual URL

    @property
    def source_name(self) -> str:
        return "free_crypto_news"

    @property
    def source_credibility(self) -> float:
        return 0.6

    async def fetch_items(self) -> List[NewsItem]:
        """Fetch from free crypto news API."""
        # TODO: Implement when we verify the actual API endpoint
        logger.debug("Free crypto news source not yet implemented")
        return []
