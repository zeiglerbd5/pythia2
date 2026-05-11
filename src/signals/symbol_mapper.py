"""
Symbol Mapper - Entity Extraction for Cryptocurrency Mentions

Maps text mentions to trading symbols:
- "Solana" -> "SOL-USD"
- "$SOL" -> "SOL-USD"
- "#Bitcoin" -> "BTC-USD"

Uses CoinGecko API for comprehensive name mappings,
filtered to only tradeable symbols from config.
"""

import re
import asyncio
import aiohttp
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from loguru import logger


@dataclass
class SymbolMatch:
    """Represents a matched cryptocurrency symbol in text."""
    symbol: str           # Trading pair (e.g., "SOL-USD")
    matched_text: str     # Original text that was matched
    confidence: float     # Confidence score (0.0-1.0)
    match_type: str       # "ticker", "name", "hashtag", "cashtag"


class SymbolMapper:
    """
    Maps cryptocurrency mentions in text to trading symbols.

    Features:
    - Multiple match types (ticker, name, hashtag, cashtag)
    - CoinGecko API integration for name resolution
    - Filters to only tradeable pairs from config
    - Caches mappings for performance
    """

    COINGECKO_API = "https://api.coingecko.com/api/v3"

    # Common name -> ticker mappings (fallback when CoinGecko unavailable)
    COMMON_NAMES = {
        # Top coins
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "solana": "SOL",
        "ripple": "XRP",
        "cardano": "ADA",
        "dogecoin": "DOGE",
        "polkadot": "DOT",
        "polygon": "MATIC",
        "avalanche": "AVAX",
        "chainlink": "LINK",
        "uniswap": "UNI",
        "litecoin": "LTC",
        "cosmos": "ATOM",
        "stellar": "XLM",
        "monero": "XMR",
        "tron": "TRX",
        "toncoin": "TON",
        "near protocol": "NEAR",
        "near": "NEAR",
        "internet computer": "ICP",
        "hedera": "HBAR",
        "filecoin": "FIL",
        "aptos": "APT",
        "arbitrum": "ARB",
        "optimism": "OP",
        "immutable": "IMX",
        "injective": "INJ",
        "render": "RENDER",
        "the graph": "GRT",
        "aave": "AAVE",
        "maker": "MKR",
        "sui": "SUI",
        "sei": "SEI",
        "celestia": "TIA",
        "stacks": "STX",
        "pepe": "PEPE",
        "bonk": "BONK",
        "shiba inu": "SHIB",
        "shiba": "SHIB",
        "floki": "FLOKI",
        "worldcoin": "WLD",
        "blur": "BLUR",
        "kaspa": "KAS",
        "jupiter": "JUPITER",
        "ondo": "ONDO",
        "pyth": "PYTH",
        "wormhole": "W",
        "pendle": "PENDLE",
        "mantle": "MANTLE",
    }

    # Tickers that need special handling (common words)
    AMBIGUOUS_TICKERS = {"THE", "FOR", "AND", "ARE", "HAS", "HAD", "CAN", "GET", "NEW", "ALL", "ONE", "TWO"}

    def __init__(
        self,
        tradeable_symbols: List[str],
        cache_ttl_hours: int = 24,
        use_coingecko: bool = True,
    ):
        """
        Initialize the symbol mapper.

        Args:
            tradeable_symbols: List of trading pairs from config (e.g., ["SOL-USD", "BTC-USD"])
            cache_ttl_hours: Hours to cache CoinGecko mappings
            use_coingecko: Whether to fetch mappings from CoinGecko
        """
        self.tradeable_symbols = set(tradeable_symbols)
        self.cache_ttl_hours = cache_ttl_hours
        self.use_coingecko = use_coingecko

        # Extract base tickers from trading pairs
        self._tradeable_tickers: Set[str] = set()
        for pair in tradeable_symbols:
            if "-USD" in pair:
                ticker = pair.replace("-USD", "")
                self._tradeable_tickers.add(ticker.upper())

        # Build reverse lookup: name -> ticker
        self._name_to_ticker: Dict[str, str] = {}
        self._ticker_to_names: Dict[str, List[str]] = {}

        # Initialize with common names
        for name, ticker in self.COMMON_NAMES.items():
            if ticker in self._tradeable_tickers:
                self._name_to_ticker[name.lower()] = ticker
                if ticker not in self._ticker_to_names:
                    self._ticker_to_names[ticker] = []
                self._ticker_to_names[ticker].append(name)

        # CoinGecko cache
        self._coingecko_cache: Dict[str, str] = {}
        self._cache_timestamp: Optional[datetime] = None

        # Compiled regex patterns
        self._cashtag_pattern = re.compile(r'\$([A-Za-z]{2,10})(?:\s|$|[.,!?])', re.IGNORECASE)
        self._hashtag_pattern = re.compile(r'#([A-Za-z]{2,15})(?:\s|$|[.,!?])', re.IGNORECASE)
        self._ticker_pattern = re.compile(r'\b([A-Z]{2,10})\b')

        # Statistics
        self.stats = {
            'matches_found': 0,
            'coingecko_lookups': 0,
            'cache_hits': 0,
        }

        logger.info(f"SymbolMapper initialized with {len(self._tradeable_tickers)} tradeable tickers")

    async def initialize(self):
        """
        Initialize async resources and fetch CoinGecko mappings.
        Call this after creation in async context.
        """
        if self.use_coingecko:
            await self._refresh_coingecko_cache()

    async def _refresh_coingecko_cache(self):
        """Fetch coin list from CoinGecko and build name mappings."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.COINGECKO_API}/coins/list"
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"CoinGecko API returned {response.status}")
                        return

                    coins = await response.json()

                    for coin in coins:
                        coin_id = coin.get("id", "").lower()
                        symbol = coin.get("symbol", "").upper()
                        name = coin.get("name", "").lower()

                        # Only add if symbol is tradeable
                        if symbol in self._tradeable_tickers:
                            self._coingecko_cache[coin_id] = symbol
                            self._coingecko_cache[name] = symbol

                            # Add to name lookup
                            self._name_to_ticker[name] = symbol
                            self._name_to_ticker[coin_id] = symbol

                            if symbol not in self._ticker_to_names:
                                self._ticker_to_names[symbol] = []
                            if name not in self._ticker_to_names[symbol]:
                                self._ticker_to_names[symbol].append(name)

                    self._cache_timestamp = datetime.now(timezone.utc)
                    self.stats['coingecko_lookups'] += 1

                    logger.info(f"CoinGecko cache refreshed: {len(self._coingecko_cache)} mappings")

        except Exception as e:
            logger.error(f"Failed to refresh CoinGecko cache: {e}")

    def _is_cache_valid(self) -> bool:
        """Check if CoinGecko cache is still valid."""
        if self._cache_timestamp is None:
            return False
        age = datetime.now(timezone.utc) - self._cache_timestamp
        return age < timedelta(hours=self.cache_ttl_hours)

    def extract_symbols(self, text: str) -> List[SymbolMatch]:
        """
        Extract all cryptocurrency mentions from text.

        Args:
            text: Text to search for mentions

        Returns:
            List of SymbolMatch objects for each unique symbol found
        """
        if not text:
            return []

        matches: Dict[str, SymbolMatch] = {}  # symbol -> best match

        # 1. Search for $TICKER (cashtag) - highest confidence
        for match in self._cashtag_pattern.finditer(text):
            ticker = match.group(1).upper()
            if ticker in self._tradeable_tickers:
                symbol = f"{ticker}-USD"
                if symbol not in matches or matches[symbol].confidence < 0.95:
                    matches[symbol] = SymbolMatch(
                        symbol=symbol,
                        matched_text=match.group(0).strip(),
                        confidence=0.95,
                        match_type="cashtag"
                    )

        # 2. Search for #TICKER (hashtag) - high confidence
        for match in self._hashtag_pattern.finditer(text):
            tag = match.group(1).lower()

            # Check if hashtag matches a ticker
            ticker = tag.upper()
            if ticker in self._tradeable_tickers:
                symbol = f"{ticker}-USD"
                if symbol not in matches or matches[symbol].confidence < 0.85:
                    matches[symbol] = SymbolMatch(
                        symbol=symbol,
                        matched_text=match.group(0).strip(),
                        confidence=0.85,
                        match_type="hashtag"
                    )

            # Check if hashtag matches a name
            elif tag in self._name_to_ticker:
                ticker = self._name_to_ticker[tag]
                symbol = f"{ticker}-USD"
                if symbol not in matches or matches[symbol].confidence < 0.80:
                    matches[symbol] = SymbolMatch(
                        symbol=symbol,
                        matched_text=match.group(0).strip(),
                        confidence=0.80,
                        match_type="hashtag"
                    )

        # 3. Search for full coin names - medium-high confidence
        text_lower = text.lower()
        for name, ticker in self._name_to_ticker.items():
            if len(name) < 3:  # Skip very short names
                continue

            # Look for word boundaries around the name
            pattern = r'\b' + re.escape(name) + r'\b'
            if re.search(pattern, text_lower):
                symbol = f"{ticker}-USD"
                if symbol not in matches or matches[symbol].confidence < 0.75:
                    matches[symbol] = SymbolMatch(
                        symbol=symbol,
                        matched_text=name,
                        confidence=0.75,
                        match_type="name"
                    )

        # 4. Search for standalone tickers - lower confidence (many false positives)
        for match in self._ticker_pattern.finditer(text):
            ticker = match.group(1).upper()

            # Skip ambiguous tickers
            if ticker in self.AMBIGUOUS_TICKERS:
                continue

            # Skip if already matched with higher confidence
            if ticker in self._tradeable_tickers:
                symbol = f"{ticker}-USD"
                if symbol not in matches:
                    matches[symbol] = SymbolMatch(
                        symbol=symbol,
                        matched_text=match.group(0),
                        confidence=0.50,
                        match_type="ticker"
                    )

        self.stats['matches_found'] += len(matches)
        return list(matches.values())

    def extract_primary_symbol(self, text: str) -> Optional[SymbolMatch]:
        """
        Extract the primary (highest confidence) symbol from text.

        Args:
            text: Text to search

        Returns:
            SymbolMatch with highest confidence, or None if no matches
        """
        matches = self.extract_symbols(text)
        if not matches:
            return None
        return max(matches, key=lambda m: m.confidence)

    def is_about_symbol(self, text: str, symbol: str) -> Tuple[bool, float]:
        """
        Check if text is about a specific trading symbol.

        Args:
            text: Text to check
            symbol: Trading pair (e.g., "SOL-USD")

        Returns:
            Tuple of (is_about, confidence)
        """
        matches = self.extract_symbols(text)
        for match in matches:
            if match.symbol == symbol:
                return True, match.confidence
        return False, 0.0

    def get_tradeable_symbols(self) -> Set[str]:
        """Get set of all tradeable symbols."""
        return self.tradeable_symbols.copy()

    def get_ticker_names(self, ticker: str) -> List[str]:
        """Get all known names for a ticker."""
        return self._ticker_to_names.get(ticker.upper(), [])

    def get_statistics(self) -> dict:
        """Get mapper statistics."""
        return {
            **self.stats,
            'tradeable_tickers': len(self._tradeable_tickers),
            'name_mappings': len(self._name_to_ticker),
            'cache_valid': self._is_cache_valid(),
        }


if __name__ == "__main__":
    # Test the symbol mapper
    import asyncio

    async def test():
        # Sample tradeable symbols
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "PEPE-USD", "ARB-USD"]

        mapper = SymbolMapper(symbols, use_coingecko=False)

        # Test cases
        test_texts = [
            "Just bought some $SOL, looking bullish!",
            "Ethereum is about to pump, get in now!",
            "#Bitcoin breaking $100k today",
            "Arbitrum and Optimism are the future of L2",
            "BTC ETH SOL all mooning",
            "This random text has no crypto mentions",
            "DOGE to the moon! Much wow, such gains!",
        ]

        print("Symbol Mapper Test Results:")
        print("-" * 60)

        for text in test_texts:
            matches = mapper.extract_symbols(text)
            print(f"\nText: {text}")
            if matches:
                for m in matches:
                    print(f"  -> {m.symbol} ({m.match_type}, conf={m.confidence:.2f})")
            else:
                print("  -> No matches")

        print(f"\nStatistics: {mapper.get_statistics()}")

    asyncio.run(test())
