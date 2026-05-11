"""
Catalyst Detector

Unified system for detecting spike catalysts across all data sources.
Aggregates, scores, and prioritizes signals to identify tradeable opportunities.

Catalyst Categories (by priority):
1. Exchange Listings - HIGH impact, predictable
2. Airdrops/Snapshots - HIGH impact, predictable
3. Legal/Regulatory - HIGH impact, semi-predictable
4. Partnerships - MEDIUM impact, variable timing
5. Technical Breakouts - MEDIUM impact, detectable
6. Social/Viral - VARIABLE impact, fast-moving

The detector outputs actionable signals with confidence scores.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from .sources.base import NewsItem
from .sources.cryptopanic import CryptoPanicSource
from .sources.event_calendar import CoinMarketCalSource
from .sources.exchange_listings import ExchangeListingsSource
from .sources.twitter_rss import TwitterRSSSource
from .sources.whale_alert import WhaleAlertSource
from .news_monitor import NewsMonitor


@dataclass
class CatalystSignal:
    """
    A detected catalyst event that may trigger a price spike.

    This is the output of the catalyst detector - actionable trading signals.
    """
    # Core signal data
    symbol: str
    catalyst_type: str  # 'listing', 'airdrop', 'partnership', 'legal', 'social', etc.
    headline: str
    details: str

    # Timing
    detected_at: datetime
    event_time: Optional[datetime] = None  # When the catalyst happens (if scheduled)
    urgency: str = "MONITOR"  # 'IMMEDIATE', 'SOON', 'MONITOR'

    # Scoring
    confidence: float = 0.5  # 0-1 confidence in the signal
    impact_score: float = 0.5  # 0-1 expected price impact
    priority_score: float = 0.5  # Combined score for ranking

    # Source tracking
    sources: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    corroborating_signals: int = 1  # Number of sources confirming

    # Action recommendation
    action: str = "WATCH"  # 'ENTER', 'PREPARE', 'WATCH', 'AVOID'
    entry_window_minutes: Optional[int] = None

    def __post_init__(self):
        """Calculate priority score."""
        self.priority_score = self._calculate_priority()

    def _calculate_priority(self) -> float:
        """
        Calculate overall priority score.

        Factors:
        - Confidence (how sure are we this is real)
        - Impact (how much could price move)
        - Urgency (how soon do we need to act)
        - Corroboration (multiple sources = more reliable)
        """
        urgency_multiplier = {
            'IMMEDIATE': 1.5,
            'SOON': 1.2,
            'MONITOR': 1.0,
        }.get(self.urgency, 1.0)

        corroboration_bonus = min(self.corroborating_signals * 0.1, 0.3)

        return (
            self.confidence * 0.4 +
            self.impact_score * 0.4 +
            corroboration_bonus
        ) * urgency_multiplier


class CatalystDetector:
    """
    Main catalyst detection engine.

    Aggregates signals from multiple sources, deduplicates, scores,
    and outputs prioritized trading signals.
    """

    # Catalyst type impact scores (baseline)
    CATALYST_IMPACT = {
        'listing': 0.95,       # Exchange listings have highest impact
        'airdrop': 0.85,       # Airdrops create strong buying pressure
        'legal': 0.80,         # Legal wins/losses are major
        'etf': 0.90,           # ETF approvals are massive
        'whale_move': 0.70,    # Large whale transfers signal accumulation/distribution
        'partnership': 0.60,   # Partnerships vary in impact
        'upgrade': 0.55,       # Protocol upgrades
        'burn': 0.50,          # Burns are mildly positive
        'social': 0.45,        # Social can be big but unpredictable
        'unlock': 0.40,        # Unlocks often negative (selling pressure)
        'news': 0.30,          # Generic news
        'conference': 0.25,    # Conference appearances
        'mention': 0.20,       # Generic mentions
    }

    # Keywords that boost impact score
    HIGH_IMPACT_KEYWORDS = [
        'binance', 'coinbase', 'kraken',  # Major exchanges
        'sec', 'lawsuit', 'approved',      # Legal
        'etf', 'spot',                     # ETF
        'airdrop', 'snapshot',             # Airdrops
        'partnership', 'collaboration',    # Partnerships
        'breaking', 'just announced',      # Urgency
    ]

    def __init__(
        self,
        enable_cryptopanic: bool = True,
        enable_calendar: bool = True,
        enable_exchange_listings: bool = True,
        enable_twitter: bool = True,
        enable_whale_alert: bool = True,
        poll_interval_seconds: int = 60,
    ):
        """
        Initialize catalyst detector with specified sources.

        Args:
            enable_*: Toggle individual data sources
            poll_interval_seconds: How often to check for new signals
        """
        self.poll_interval = poll_interval_seconds
        self.sources = []
        self._whale_alert_source: Optional[WhaleAlertSource] = None

        # Initialize enabled sources
        if enable_cryptopanic:
            self.sources.append(CryptoPanicSource())

        if enable_calendar:
            self.sources.append(CoinMarketCalSource())

        if enable_exchange_listings:
            self.sources.append(ExchangeListingsSource())

        if enable_twitter:
            self.sources.append(TwitterRSSSource())

        if enable_whale_alert:
            self._whale_alert_source = WhaleAlertSource(min_usd_value=1_000_000)
            self.sources.append(self._whale_alert_source)

        # Signal aggregation
        self._recent_signals: Dict[str, CatalystSignal] = {}  # key = symbol:catalyst_type
        self._signal_history: List[CatalystSignal] = []
        self._seen_news_hashes: Set[str] = set()

        # Callbacks
        self._signal_callbacks: List[callable] = []

        # Running state
        self._running = False
        self._task: Optional[asyncio.Task] = None

        logger.info(f"CatalystDetector initialized with {len(self.sources)} sources")

    def register_callback(self, callback: callable):
        """Register callback for new signals."""
        self._signal_callbacks.append(callback)

    async def start(self):
        """Start the catalyst detection loop."""
        if self._running:
            logger.warning("CatalystDetector already running")
            return

        self._running = True

        # Start Whale Alert WebSocket if enabled
        if self._whale_alert_source and self._whale_alert_source.api_key:
            await self._whale_alert_source.start_websocket(startup_delay=5)

        self._task = asyncio.create_task(self._detection_loop())
        logger.info("CatalystDetector started")

    async def stop(self):
        """Stop the detection loop."""
        self._running = False

        # Stop Whale Alert WebSocket
        if self._whale_alert_source:
            await self._whale_alert_source.stop_websocket()

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("CatalystDetector stopped")

    async def _detection_loop(self):
        """Main detection loop."""
        while self._running:
            try:
                await self._fetch_and_process()
            except Exception as e:
                logger.error(f"Detection loop error: {e}")

            await asyncio.sleep(self.poll_interval)

    async def _fetch_and_process(self):
        """Fetch from all sources and process into signals."""
        all_items: List[NewsItem] = []

        # Fetch from all sources concurrently
        tasks = [source.fetch_with_rate_limit() for source in self.sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Source {self.sources[i].source_name} error: {result}")
            elif isinstance(result, list):
                all_items.extend(result)

        logger.debug(f"Fetched {len(all_items)} items from {len(self.sources)} sources")

        # Process items into signals
        new_signals = self._process_items(all_items)

        # Notify callbacks for high-priority signals
        for signal in new_signals:
            if signal.priority_score >= 0.6:  # Threshold for notification
                for callback in self._signal_callbacks:
                    try:
                        await callback(signal) if asyncio.iscoroutinefunction(callback) else callback(signal)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

    def _process_items(self, items: List[NewsItem]) -> List[CatalystSignal]:
        """Process news items into catalyst signals."""
        new_signals = []

        # Group items by symbol
        symbol_items: Dict[str, List[NewsItem]] = defaultdict(list)
        for item in items:
            # Skip already seen
            content_hash = item.content_hash()
            if content_hash in self._seen_news_hashes:
                continue
            self._seen_news_hashes.add(content_hash)

            for symbol in item.symbols:
                symbol_items[symbol].append(item)

        # Process each symbol's items
        for symbol, items in symbol_items.items():
            # Group by catalyst type
            type_items: Dict[str, List[NewsItem]] = defaultdict(list)
            for item in items:
                type_items[item.event_type].append(item)

            # Create signal for each catalyst type with items
            for catalyst_type, type_items_list in type_items.items():
                signal = self._create_signal(symbol, catalyst_type, type_items_list)
                if signal and signal.priority_score >= 0.3:  # Minimum threshold
                    self._add_or_update_signal(signal)
                    new_signals.append(signal)

        # Cleanup old hashes
        if len(self._seen_news_hashes) > 50000:
            self._seen_news_hashes = set(list(self._seen_news_hashes)[-25000:])

        return new_signals

    def _create_signal(
        self,
        symbol: str,
        catalyst_type: str,
        items: List[NewsItem]
    ) -> Optional[CatalystSignal]:
        """Create a catalyst signal from news items."""
        if not items:
            return None

        # Use most recent item as primary
        items.sort(key=lambda x: x.timestamp, reverse=True)
        primary = items[0]

        # Calculate confidence from source credibility and corroboration
        avg_confidence = sum(
            self._get_source_credibility(item.source) * item.entity_confidence
            for item in items
        ) / len(items)

        # Boost confidence for multiple sources
        corroboration = len(set(item.source for item in items))
        confidence = min(avg_confidence + (corroboration - 1) * 0.1, 1.0)

        # Calculate impact score
        base_impact = self.CATALYST_IMPACT.get(catalyst_type, 0.3)
        impact_boost = self._calculate_impact_boost(primary.title, primary.content)
        impact_score = min(base_impact + impact_boost, 1.0)

        # Determine urgency
        urgency = self._determine_urgency(primary, catalyst_type)

        # Determine action recommendation
        action = self._recommend_action(catalyst_type, confidence, impact_score)

        # Estimate entry window
        entry_window = self._estimate_entry_window(catalyst_type, urgency)

        signal = CatalystSignal(
            symbol=symbol,
            catalyst_type=catalyst_type,
            headline=primary.title,
            details=primary.content,
            detected_at=datetime.now(timezone.utc),
            event_time=primary.timestamp if primary.timestamp > datetime.now(timezone.utc) else None,
            urgency=urgency,
            confidence=confidence,
            impact_score=impact_score,
            sources=[item.source for item in items],
            source_urls=[item.url for item in items if item.url],
            corroborating_signals=corroboration,
            action=action,
            entry_window_minutes=entry_window,
        )

        return signal

    def _get_source_credibility(self, source: str) -> float:
        """Get credibility score for a source."""
        credibility_map = {
            'exchange_listings': 1.0,
            'token_unlocks': 0.95,
            'coinmarketcal': 0.75,
            'cryptopanic': 0.70,
            'twitter_rss': 0.60,
            'reddit': 0.40,
        }
        return credibility_map.get(source, 0.5)

    def _calculate_impact_boost(self, title: str, content: str) -> float:
        """Calculate impact boost from keyword analysis."""
        text = f"{title} {content}".lower()
        boost = 0.0

        for keyword in self.HIGH_IMPACT_KEYWORDS:
            if keyword in text:
                boost += 0.05

        return min(boost, 0.3)  # Cap at 0.3 boost

    def _determine_urgency(self, item: NewsItem, catalyst_type: str) -> str:
        """Determine urgency level."""
        now = datetime.now(timezone.utc)

        # Scheduled events
        if item.timestamp > now:
            time_until = item.timestamp - now
            if time_until < timedelta(hours=1):
                return "IMMEDIATE"
            elif time_until < timedelta(hours=24):
                return "SOON"
            else:
                return "MONITOR"

        # Breaking news
        time_since = now - item.timestamp
        if time_since < timedelta(minutes=30):
            return "IMMEDIATE"
        elif time_since < timedelta(hours=2):
            return "SOON"
        else:
            return "MONITOR"

    def _recommend_action(
        self,
        catalyst_type: str,
        confidence: float,
        impact_score: float
    ) -> str:
        """Recommend trading action."""
        combined = confidence * impact_score

        if catalyst_type in ['listing', 'airdrop', 'etf'] and combined > 0.6:
            return "ENTER"
        elif combined > 0.5:
            return "PREPARE"
        elif combined > 0.3:
            return "WATCH"
        else:
            return "AVOID"

    def _estimate_entry_window(self, catalyst_type: str, urgency: str) -> Optional[int]:
        """Estimate how long the entry window lasts (minutes)."""
        windows = {
            'listing': {'IMMEDIATE': 15, 'SOON': 60, 'MONITOR': 240},
            'airdrop': {'IMMEDIATE': 30, 'SOON': 120, 'MONITOR': 480},
            'legal': {'IMMEDIATE': 30, 'SOON': 120, 'MONITOR': None},
            'social': {'IMMEDIATE': 10, 'SOON': 30, 'MONITOR': 60},
        }

        return windows.get(catalyst_type, {}).get(urgency)

    def _add_or_update_signal(self, signal: CatalystSignal):
        """Add new signal or update existing."""
        key = f"{signal.symbol}:{signal.catalyst_type}"

        existing = self._recent_signals.get(key)
        if existing:
            # Update with newer info, increase corroboration
            signal.corroborating_signals = existing.corroborating_signals + 1
            signal.sources = list(set(existing.sources + signal.sources))

        self._recent_signals[key] = signal
        self._signal_history.append(signal)

        # Cleanup old signals
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-500:]

    def get_active_signals(
        self,
        min_priority: float = 0.3,
        catalyst_types: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
    ) -> List[CatalystSignal]:
        """
        Get current active signals filtered and sorted by priority.

        Args:
            min_priority: Minimum priority score
            catalyst_types: Filter by catalyst types
            symbols: Filter by symbols

        Returns:
            List of signals sorted by priority (highest first)
        """
        signals = list(self._recent_signals.values())

        # Filter
        if min_priority:
            signals = [s for s in signals if s.priority_score >= min_priority]
        if catalyst_types:
            signals = [s for s in signals if s.catalyst_type in catalyst_types]
        if symbols:
            signals = [s for s in signals if s.symbol in symbols]

        # Sort by priority
        signals.sort(key=lambda x: x.priority_score, reverse=True)

        return signals

    def get_health_status(self) -> Dict[str, Any]:
        """Get detector health status."""
        return {
            "running": self._running,
            "sources": [
                {
                    "name": s.source_name,
                    "healthy": s.is_healthy(),
                    "credibility": s.source_credibility,
                }
                for s in self.sources
            ],
            "active_signals": len(self._recent_signals),
            "total_signals_processed": len(self._signal_history),
            "seen_news_count": len(self._seen_news_hashes),
        }


# Convenience function for quick signal check
async def check_catalysts(symbols: Optional[List[str]] = None) -> List[CatalystSignal]:
    """
    One-shot catalyst check.

    Args:
        symbols: Optional list of symbols to check

    Returns:
        List of detected catalyst signals
    """
    detector = CatalystDetector()
    await detector._fetch_and_process()
    return detector.get_active_signals(symbols=symbols)
