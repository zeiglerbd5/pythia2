"""
News Monitor - Main Coordinator

Follows VolumeScanner pattern:
- Async scan loop (30 second interval)
- Signal callback to FeatureEngine
- Deduplication with content hashing
- Source health tracking
- Orchestrates multiple source adapters
"""

import asyncio
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable, Set, Any
from dataclasses import dataclass, field
from loguru import logger

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .symbol_mapper import SymbolMapper, SymbolMatch
from .sources.base import BaseSource, NewsItem


@dataclass
class NewsSignal:
    """
    Processed news signal ready for trading decisions.

    This is the output of NewsMonitor after scoring and filtering.
    """
    symbol: str                    # Trading pair (e.g., "SOL-USD")
    timestamp: datetime            # Signal timestamp
    source: str                    # Source identifier
    event_type: str                # "listing", "whale_move", "partnership", etc.
    confidence: float              # Final confidence score (0.0-1.0)
    title: str                     # Headline
    url: Optional[str]             # Original URL

    # Scoring breakdown
    source_credibility: float = 0.0
    entity_certainty: float = 0.0
    event_priority: float = 0.0
    recency_score: float = 0.0
    engagement_score: float = 0.0

    # VADER sentiment score (-1.0 bearish to +1.0 bullish)
    sentiment_score: float = 0.0

    # Deduplication
    signal_hash: str = ""

    def __post_init__(self):
        if not self.signal_hash:
            content = f"{self.symbol}:{self.source}:{self.event_type}:{self.title}"
            self.signal_hash = hashlib.md5(content.encode()).hexdigest()[:16]


# Event priority mapping
EVENT_PRIORITY = {
    "listing": 1.0,           # New exchange listings
    "delisting": 0.9,         # Delistings (bearish signal)
    "whale_move": 0.9,        # Large transfers
    "partnership": 0.7,       # Partnership announcements
    "upgrade": 0.7,           # Network upgrades
    "airdrop": 0.6,           # Airdrop announcements
    "sentiment_spike": 0.5,   # Social sentiment surge
    "mention": 0.3,           # General mentions
}


class NewsMonitor:
    """
    Coordinates multiple news sources and produces trading signals.

    Features:
    - Multi-source aggregation
    - Symbol extraction and mapping
    - Signal scoring based on source credibility, entity certainty, etc.
    - Deduplication with content hashing
    - Health monitoring per source
    """

    def __init__(
        self,
        symbol_mapper: SymbolMapper,
        sources: Optional[List[BaseSource]] = None,
        signal_callback: Optional[Callable] = None,
        scan_interval: int = 30,  # 30 seconds per plan
        min_confidence: float = 0.5,
        signal_cooldown: int = 1800,  # 30 min cooldown per symbol
    ):
        """
        Initialize the news monitor.

        Args:
            symbol_mapper: SymbolMapper instance for entity extraction
            sources: List of news source adapters
            signal_callback: Async callback for processed signals
            scan_interval: Seconds between scan cycles
            min_confidence: Minimum confidence for signal emission
            signal_cooldown: Cooldown per symbol in seconds
        """
        self.symbol_mapper = symbol_mapper
        self.sources: List[BaseSource] = sources or []
        self.signal_callback = signal_callback
        self.scan_interval = scan_interval
        self.min_confidence = min_confidence
        self.signal_cooldown = signal_cooldown

        # VADER sentiment analyzer with crypto lexicon updates
        self._sentiment_analyzer = SentimentIntensityAnalyzer()

        # Add crypto-specific terms to VADER lexicon
        # Positive values = bullish, negative = bearish
        crypto_lexicon = {
            # Bullish terms
            'bullish': 2.5,
            'moon': 2.0,
            'mooning': 2.5,
            'pump': 1.5,
            'pumping': 2.0,
            'breakout': 1.8,
            'ath': 2.0,  # all-time high
            'hodl': 1.5,
            'accumulate': 1.2,
            'accumulating': 1.5,
            'undervalued': 1.5,
            'gem': 1.8,
            'bullrun': 2.5,
            'rally': 1.5,
            'surge': 1.8,
            'surging': 2.0,
            'listing': 1.5,
            'listed': 1.2,
            'partnership': 1.2,
            'adoption': 1.5,
            'integration': 1.0,
            'upgrade': 1.0,
            'mainnet': 1.2,
            'airdrop': 1.0,
            'staking': 0.8,

            # Bearish terms
            'bearish': -2.5,
            'dump': -2.0,
            'dumping': -2.5,
            'crash': -2.5,
            'crashing': -2.8,
            'rug': -3.0,
            'rugged': -3.0,
            'rugpull': -3.0,
            'scam': -3.0,
            'hack': -2.5,
            'hacked': -2.8,
            'exploit': -2.0,
            'exploited': -2.5,
            'rekt': -2.5,
            'liquidated': -2.0,
            'liquidation': -1.8,
            'delist': -2.0,
            'delisting': -2.2,
            'sec': -1.0,  # SEC often negative context
            'lawsuit': -2.0,
            'fud': -1.5,
            'ponzi': -3.0,
            'selloff': -2.0,
            'bleeding': -1.8,
            'tanking': -2.2,
            'plunge': -2.0,
            'plunging': -2.2,
        }

        # Update VADER lexicon with crypto terms
        self._sentiment_analyzer.lexicon.update(crypto_lexicon)

        # Deduplication
        self._seen_hashes: Set[str] = set()
        self._hash_expiry: Dict[str, datetime] = {}
        self._hash_ttl_hours = 24

        # Recent signals per symbol (for cooldown)
        self._recent_signals: Dict[str, datetime] = {}

        # Signal cache for FeatureEngine queries
        self._signal_cache: Dict[str, NewsSignal] = {}

        # Statistics
        self._stats = {
            'scans_completed': 0,
            'signals_detected': 0,
            'items_processed': 0,
            'duplicates_filtered': 0,
            'last_scan': None,
        }

        # Running state
        self._running = False
        self._task: Optional[asyncio.Task] = None

        logger.info(f"NewsMonitor initialized with {len(self.sources)} sources, interval={scan_interval}s")

    def add_source(self, source: BaseSource):
        """Add a news source adapter."""
        self.sources.append(source)
        logger.info(f"Added source: {source.source_name}")

    async def start(self):
        """Start the news monitor loop."""
        if self._running:
            logger.warning("NewsMonitor already running")
            return

        self._running = True
        logger.info(f"News monitor starting (interval={self.scan_interval}s, sources={len(self.sources)})")

        try:
            while self._running:
                await self._scan_cycle()
                await asyncio.sleep(self.scan_interval)
        except asyncio.CancelledError:
            logger.info("News monitor cancelled")
        except Exception as e:
            logger.error(f"News monitor error: {e}")
        finally:
            self._running = False

    async def stop(self):
        """Stop the news monitor."""
        self._running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(f"News monitor stopped. Stats: {self._stats}")

    async def _scan_cycle(self):
        """One scan cycle across all sources."""
        start_time = datetime.now(timezone.utc)
        all_items: List[NewsItem] = []

        # Fetch from all healthy sources in parallel
        healthy_sources = [s for s in self.sources if s.is_healthy()]

        if not healthy_sources:
            logger.warning("No healthy sources available")
            return

        tasks = [s.fetch_with_rate_limit() for s in healthy_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Track per-source results for logging
        source_results = {}

        for source, result in zip(healthy_sources, results):
            if isinstance(result, Exception):
                logger.warning(f"Source {source.source_name} failed: {result}")
                source_results[source.source_name] = "ERR"
            elif isinstance(result, list):
                all_items.extend(result)
                source_results[source.source_name] = len(result)

        # Process items
        signals_this_cycle = []

        for item in all_items:
            self._stats['items_processed'] += 1

            # Check for duplicates
            item_hash = item.content_hash()
            if item_hash in self._seen_hashes:
                self._stats['duplicates_filtered'] += 1
                continue

            # Mark as seen
            self._seen_hashes.add(item_hash)
            self._hash_expiry[item_hash] = datetime.now(timezone.utc) + timedelta(hours=self._hash_ttl_hours)

            # Extract symbols
            matches = self.symbol_mapper.extract_symbols(item.title + " " + item.content)

            if not matches:
                continue  # No tradeable symbols mentioned

            # Create signals for each matched symbol
            for match in matches:
                signal = self._create_signal(item, match)

                if signal.confidence >= self.min_confidence:
                    if self._check_cooldown(signal):
                        signals_this_cycle.append(signal)

        # Process signals
        for signal in signals_this_cycle:
            await self._process_signal(signal)

        # Cleanup expired hashes periodically
        if self._stats['scans_completed'] % 60 == 0:  # Every 30 minutes
            self._cleanup_expired_hashes()

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        self._stats['scans_completed'] += 1
        self._stats['last_scan'] = start_time

        # Log summary periodically
        if self._stats['scans_completed'] % 20 == 1:  # Every 10 minutes
            # Build per-source breakdown string
            source_breakdown = " | ".join(
                f"{name}:{count}" for name, count in source_results.items()
            )
            logger.info(
                f"[NEWS_SCAN] Cycle {self._stats['scans_completed']}: "
                f"{len(all_items)} items, {len(signals_this_cycle)} signals in {elapsed:.1f}s "
                f"[{source_breakdown}]"
            )

    def _create_signal(self, item: NewsItem, match: SymbolMatch) -> NewsSignal:
        """
        Create a scored signal from a news item and symbol match.

        Scoring formula (from plan):
        Final Score = (0.3 * Source Credibility) +
                      (0.3 * Entity Certainty) +
                      (0.2 * Event Priority) +
                      (0.1 * Recency) +
                      (0.1 * Engagement)
        """
        # Get source credibility
        source_credibility = 0.5
        for source in self.sources:
            if source.source_name == item.source:
                source_credibility = source.source_credibility
                break

        # Entity certainty from match confidence
        entity_certainty = match.confidence

        # Event priority
        event_priority = EVENT_PRIORITY.get(item.event_type, 0.3)

        # Recency score (higher for more recent)
        age_minutes = (datetime.now(timezone.utc) - item.timestamp).total_seconds() / 60
        if age_minutes <= 5:
            recency_score = 1.0
        elif age_minutes <= 15:
            recency_score = 0.8
        elif age_minutes <= 60:
            recency_score = 0.5
        else:
            recency_score = 0.2

        # Engagement score
        engagement_score = 0.5  # Default
        if item.engagement:
            # Normalize engagement metrics
            likes = item.engagement.get('likes', 0)
            retweets = item.engagement.get('retweets', 0)
            upvotes = item.engagement.get('upvotes', 0)
            total_engagement = likes + retweets * 2 + upvotes

            if total_engagement >= 1000:
                engagement_score = 1.0
            elif total_engagement >= 100:
                engagement_score = 0.7
            elif total_engagement >= 10:
                engagement_score = 0.5
            else:
                engagement_score = 0.3

        # Calculate final confidence
        confidence = (
            0.3 * source_credibility +
            0.3 * entity_certainty +
            0.2 * event_priority +
            0.1 * recency_score +
            0.1 * engagement_score
        )

        # VADER sentiment analysis on title + content
        # Returns compound score from -1.0 (bearish) to +1.0 (bullish)
        text_for_sentiment = f"{item.title} {item.content}"
        sentiment_scores = self._sentiment_analyzer.polarity_scores(text_for_sentiment)
        sentiment_score = sentiment_scores['compound']  # -1 to +1

        return NewsSignal(
            symbol=match.symbol,
            timestamp=item.timestamp,
            source=item.source,
            event_type=item.event_type,
            confidence=confidence,
            title=item.title,
            url=item.url,
            source_credibility=source_credibility,
            entity_certainty=entity_certainty,
            event_priority=event_priority,
            recency_score=recency_score,
            engagement_score=engagement_score,
            sentiment_score=sentiment_score,
        )

    def _check_cooldown(self, signal: NewsSignal) -> bool:
        """Check if signal passes cooldown check."""
        if signal.symbol in self._recent_signals:
            last_signal = self._recent_signals[signal.symbol]
            if (signal.timestamp - last_signal).total_seconds() < self.signal_cooldown:
                return False
        return True

    async def _process_signal(self, signal: NewsSignal):
        """Process a detected signal."""
        self._recent_signals[signal.symbol] = signal.timestamp
        self._signal_cache[signal.symbol] = signal
        self._stats['signals_detected'] += 1

        # Log the signal
        event_emoji = {
            "listing": "📋",
            "whale_move": "🐋",
            "partnership": "🤝",
            "sentiment_spike": "📈",
        }.get(signal.event_type, "📰")

        logger.warning(
            f"{event_emoji} [NEWS_SIGNAL] {signal.symbol}: "
            f"{signal.event_type} | conf={signal.confidence:.2f} | "
            f"src={signal.source} | {signal.title[:50]}..."
        )

        # Call callback if set
        if self.signal_callback:
            try:
                await self.signal_callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    def _cleanup_expired_hashes(self):
        """Remove expired hashes from deduplication set."""
        now = datetime.now(timezone.utc)
        expired = [h for h, exp in self._hash_expiry.items() if exp < now]

        for h in expired:
            self._seen_hashes.discard(h)
            del self._hash_expiry[h]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired hashes")

    # Public methods for querying

    def get_signal(self, symbol: str) -> Optional[NewsSignal]:
        """Get the most recent signal for a symbol."""
        return self._signal_cache.get(symbol)

    def get_active_signals(self, min_confidence: float = 0.6) -> List[NewsSignal]:
        """Get all active signals above confidence threshold."""
        now = datetime.now(timezone.utc)
        active = []

        for signal in self._signal_cache.values():
            # Consider signals active for 1 hour
            if (now - signal.timestamp).total_seconds() < 3600:
                if signal.confidence >= min_confidence:
                    active.append(signal)

        return sorted(active, key=lambda s: s.confidence, reverse=True)

    def get_source_health(self) -> List[Dict[str, Any]]:
        """Get health status for all sources."""
        return [s.get_health_status() for s in self.sources]

    def get_statistics(self) -> dict:
        """Get monitor statistics."""
        return {
            **self._stats,
            'sources_total': len(self.sources),
            'sources_healthy': sum(1 for s in self.sources if s.is_healthy()),
            'active_signals': len([s for s in self._signal_cache.values()
                                   if (datetime.now(timezone.utc) - s.timestamp).total_seconds() < 3600]),
            'seen_hashes': len(self._seen_hashes),
        }


if __name__ == "__main__":
    # Test the news monitor
    import asyncio

    async def test():
        from .symbol_mapper import SymbolMapper

        # Create mapper with sample symbols
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]
        mapper = SymbolMapper(symbols, use_coingecko=False)

        # Create monitor without sources (just test infrastructure)
        monitor = NewsMonitor(
            symbol_mapper=mapper,
            sources=[],
            scan_interval=5,
        )

        print(f"Statistics: {monitor.get_statistics()}")
        print(f"Source health: {monitor.get_source_health()}")

    asyncio.run(test())
