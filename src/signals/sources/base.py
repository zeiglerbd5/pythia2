"""
Base Source Adapter

Defines the interface for all news source adapters with:
- Rate limiting
- Health tracking
- Consistent data structures
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from loguru import logger


@dataclass
class NewsItem:
    """
    Standardized news/signal item from any source.

    This is the common format that all source adapters must produce.
    """
    source: str              # Source identifier (e.g., "exchange_listings", "whale_alert")
    event_type: str          # Event category (e.g., "listing", "whale_move", "tweet")
    title: str               # Headline or summary
    content: str             # Full content/description
    url: Optional[str]       # Original URL if available
    timestamp: datetime      # When the event occurred/was published
    raw_data: Dict[str, Any] = field(default_factory=dict)  # Original API response

    # Entity extraction (filled by SymbolMapper)
    symbols: List[str] = field(default_factory=list)  # Matched trading symbols
    entity_confidence: float = 0.0  # Confidence in symbol extraction

    # Source-specific metadata
    engagement: Optional[Dict[str, Any]] = None  # likes, retweets, upvotes, etc.
    author: Optional[str] = None
    verified: bool = False   # Whether the source is verified (e.g., official exchange account)

    def content_hash(self) -> str:
        """Generate a hash for deduplication."""
        import hashlib
        content = f"{self.source}:{self.event_type}:{self.title}:{self.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()


class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    """

    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self._last_request: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request is allowed."""
        async with self._lock:
            now = time.time()
            wait_time = self._last_request + self.interval - now

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            self._last_request = time.time()


class BaseSource(ABC):
    """
    Abstract base class for all news source adapters.

    Subclasses must implement:
    - fetch_items(): Fetch news items from the source
    - source_name: Property returning source identifier
    """

    def __init__(
        self,
        rate_limit_per_minute: int = 60,
        request_timeout: int = 30,
    ):
        """
        Initialize base source.

        Args:
            rate_limit_per_minute: Maximum requests per minute
            request_timeout: Timeout for API requests in seconds
        """
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        self.request_timeout = request_timeout

        # Health tracking
        self._last_success: Optional[datetime] = None
        self._last_error: Optional[str] = None
        self._consecutive_errors: int = 0
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._items_fetched: int = 0

        # Last fetch diagnostics
        self._last_fetch_count: int = 0
        self._last_fetch_status: str = "not_started"

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return unique identifier for this source."""
        pass

    @property
    def source_credibility(self) -> float:
        """
        Return credibility score for this source (0.0-1.0).

        Used in signal scoring:
        - 1.0: Official exchange announcements
        - 0.9: Whale Alert (on-chain data)
        - 0.7: Verified social accounts
        - 0.4: Community/Reddit
        """
        return 0.5  # Default, subclasses should override

    @abstractmethod
    async def fetch_items(self) -> List[NewsItem]:
        """
        Fetch news items from the source.

        Returns:
            List of NewsItem objects
        """
        pass

    async def fetch_with_rate_limit(self) -> List[NewsItem]:
        """
        Fetch items with rate limiting and error handling.

        Returns:
            List of NewsItem objects
        """
        await self.rate_limiter.acquire()
        self._total_requests += 1

        try:
            items = await self.fetch_items()
            self._last_success = datetime.now(timezone.utc)
            self._consecutive_errors = 0
            self._items_fetched += len(items)

            # Track fetch result for diagnostics
            self._last_fetch_count = len(items)
            self._last_fetch_status = "ok"
            return items

        except Exception as e:
            self._total_errors += 1
            self._consecutive_errors += 1
            self._last_error = str(e)
            self._last_fetch_count = 0
            self._last_fetch_status = f"error: {str(e)[:50]}"
            logger.warning(f"[{self.source_name}] Fetch error ({self._consecutive_errors} consecutive): {e}")
            return []

    def is_healthy(self) -> bool:
        """
        Check if source is healthy.

        Returns False if too many consecutive errors.
        """
        return self._consecutive_errors < 5

    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status."""
        return {
            "source": self.source_name,
            "healthy": self.is_healthy(),
            "last_success": self._last_success.isoformat() if self._last_success else None,
            "last_error": self._last_error,
            "consecutive_errors": self._consecutive_errors,
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "items_fetched": self._items_fetched,
            "credibility": self.source_credibility,
            "last_fetch_count": self._last_fetch_count,
            "last_fetch_status": self._last_fetch_status,
        }
