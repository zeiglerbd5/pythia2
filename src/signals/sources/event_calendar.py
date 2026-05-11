"""
Crypto Event Calendar Source

Monitors scheduled events that can cause price spikes:
- Token unlocks (vesting releases)
- Airdrops
- Hard forks / upgrades
- Conference appearances
- Earnings/reports (for crypto companies)

Data sources:
- CoinMarketCap token unlocks
- TokenUnlocks.app
- CoinMarketCal
"""

import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import aiohttp
from dataclasses import dataclass
from loguru import logger

from .base import BaseSource, NewsItem


@dataclass
class ScheduledEvent:
    """Represents a scheduled crypto event."""
    symbol: str
    event_type: str  # 'unlock', 'airdrop', 'fork', 'conference', 'launch'
    event_date: datetime
    description: str
    value_usd: Optional[float] = None  # For unlocks: USD value being unlocked
    percentage_of_supply: Optional[float] = None  # % of circulating supply
    source_url: Optional[str] = None
    confidence: float = 0.8


class TokenUnlocksSource(BaseSource):
    """
    Token unlocks calendar source.

    Monitors upcoming token unlocks that may cause selling pressure
    or (inversely) buying opportunities after dumps.

    Large unlocks (>5% of supply) are significant catalysts.
    """

    # Known high-impact unlock events (manually curated, update regularly)
    # In production, this would be fetched from an API
    KNOWN_UNLOCKS = [
        # Example format - in reality, fetch from API
        # {"symbol": "ARB", "date": "2024-03-16", "pct": 3.2, "value_usd": 1.2e9},
    ]

    def __init__(self):
        super().__init__(rate_limit_per_minute=10)
        self._cached_events: List[ScheduledEvent] = []
        self._last_cache_update: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)

    @property
    def source_name(self) -> str:
        return "token_unlocks"

    @property
    def source_credibility(self) -> float:
        return 0.95  # Unlocks are scheduled, highly reliable

    async def fetch_items(self) -> List[NewsItem]:
        """Fetch upcoming token unlock events."""
        items = []

        # Check cache
        now = datetime.now(timezone.utc)
        if self._last_cache_update and (now - self._last_cache_update) < self._cache_ttl:
            events = self._cached_events
        else:
            events = await self._fetch_unlock_calendar()
            self._cached_events = events
            self._last_cache_update = now

        # Convert events happening in next 7 days to NewsItems
        for event in events:
            days_until = (event.event_date - now).days

            if 0 <= days_until <= 7:
                # Determine impact level
                if event.percentage_of_supply and event.percentage_of_supply > 5:
                    impact = "HIGH"
                elif event.percentage_of_supply and event.percentage_of_supply > 2:
                    impact = "MEDIUM"
                else:
                    impact = "LOW"

                title = f"Token Unlock: {event.symbol} - {event.percentage_of_supply:.1f}% of supply"
                content = (
                    f"{event.symbol} unlock in {days_until} days. "
                    f"{event.description}. "
                    f"Impact: {impact}"
                )

                if event.value_usd:
                    content += f". Value: ${event.value_usd/1e6:.1f}M"

                item = NewsItem(
                    source=self.source_name,
                    event_type="unlock",
                    title=title,
                    content=content,
                    url=event.source_url,
                    timestamp=event.event_date,
                    symbols=[f"{event.symbol}-USD"],
                    entity_confidence=0.95,
                    engagement={
                        'days_until': days_until,
                        'impact': impact,
                        'pct_of_supply': event.percentage_of_supply,
                        'value_usd': event.value_usd,
                    },
                    verified=True,
                )
                items.append(item)

        return items

    async def _fetch_unlock_calendar(self) -> List[ScheduledEvent]:
        """
        Fetch unlock calendar from API.

        TODO: Integrate with actual API (TokenUnlocks, CoinMarketCap)
        For now, returns empty list - would need API key or scraping.
        """
        events = []

        # Placeholder: In production, fetch from:
        # - https://token.unlocks.app/api/... (requires subscription)
        # - https://coinmarketcap.com/token-unlocks/ (scraping or API)
        # - https://defillama.com/unlocks (API available)

        try:
            # Try DefiLlama unlocks API (free)
            async with aiohttp.ClientSession() as session:
                url = "https://api.llama.fi/unlocks/upcoming"
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        for item in data.get('unlocks', [])[:50]:  # Limit to 50
                            try:
                                event = ScheduledEvent(
                                    symbol=item.get('symbol', 'UNKNOWN'),
                                    event_type='unlock',
                                    event_date=datetime.fromtimestamp(
                                        item.get('timestamp', 0),
                                        tz=timezone.utc
                                    ),
                                    description=item.get('description', 'Token unlock'),
                                    value_usd=item.get('value'),
                                    percentage_of_supply=item.get('percentOfSupply'),
                                    source_url="https://defillama.com/unlocks",
                                    confidence=0.9,
                                )
                                events.append(event)
                            except Exception as e:
                                logger.debug(f"Error parsing unlock event: {e}")
                                continue
        except Exception as e:
            logger.warning(f"Failed to fetch unlock calendar: {e}")

        logger.debug(f"Fetched {len(events)} upcoming unlock events")
        return events


class AirdropCalendarSource(BaseSource):
    """
    Airdrop calendar source.

    Monitors upcoming and recently announced airdrops.
    Airdrops create buying pressure before snapshot dates.
    """

    def __init__(self):
        super().__init__(rate_limit_per_minute=5)

    @property
    def source_name(self) -> str:
        return "airdrop_calendar"

    @property
    def source_credibility(self) -> float:
        return 0.8

    async def fetch_items(self) -> List[NewsItem]:
        """Fetch upcoming airdrop events."""
        items = []

        # Placeholder: Would fetch from airdrops.io API or scrape
        # For now, return empty - this requires either:
        # 1. Paid API access
        # 2. Web scraping (less reliable)
        # 3. Manual curation

        logger.debug("Airdrop calendar: No API configured yet")
        return items


class CoinMarketCalSource(BaseSource):
    """
    CoinMarketCal event calendar source.

    Comprehensive crypto event calendar including:
    - Exchange listings
    - Partnerships
    - Updates/releases
    - Conferences
    - Airdrops
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(rate_limit_per_minute=10)
        self.api_key = api_key or os.getenv('COINMARKETCAL_API_KEY')
        self.base_url = "https://developers.coinmarketcal.com/v1"

    @property
    def source_name(self) -> str:
        return "coinmarketcal"

    @property
    def source_credibility(self) -> float:
        return 0.75  # Community-submitted events, varies in quality

    async def fetch_items(self) -> List[NewsItem]:
        """Fetch events from CoinMarketCal."""
        if not self.api_key:
            logger.debug("CoinMarketCal API key not configured")
            return []

        items = []

        try:
            headers = {
                'x-api-key': self.api_key,
                'Accept': 'application/json',
            }

            # Get events for next 7 days
            params = {
                'max': 50,
                'dateRangeStart': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                'dateRangeEnd': (datetime.now(timezone.utc) + timedelta(days=7)).strftime('%Y-%m-%d'),
            }

            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/events"
                async with session.get(url, headers=headers, params=params, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(f"CoinMarketCal API error: {response.status}")
                        return []

                    data = await response.json()

                    for event in data.get('body', []):
                        # Extract coins (skip CRYPTO which is "General Event")
                        coins = []
                        for coin in event.get('coins', []):
                            symbol = coin.get('symbol')
                            if symbol and symbol != 'CRYPTO':
                                coins.append(f"{symbol}-USD")

                        # Parse date (format: 2026-03-19T00:00:00Z)
                        try:
                            date_str = event.get('date_event', '')
                            # Handle both with and without microseconds
                            if '.' in date_str:
                                event_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
                            else:
                                event_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                        except:
                            event_date = datetime.now(timezone.utc)

                        # Map category to event type
                        categories = [c.get('name', '') for c in event.get('categories', [])]
                        event_type = self._categorize_event(categories)

                        # Get title
                        title_obj = event.get('title', {})
                        title = title_obj.get('en', '') if isinstance(title_obj, dict) else str(title_obj)

                        # Build content from categories and displayed date
                        content = f"{', '.join(categories)} - {event.get('displayed_date', '')}"

                        item = NewsItem(
                            source=self.source_name,
                            event_type=event_type,
                            title=title or 'Unknown Event',
                            content=content,
                            url=event.get('source'),
                            timestamp=event_date,
                            symbols=coins,
                            entity_confidence=0.9 if coins else 0.3,
                            engagement={
                                'categories': categories,
                                'event_id': event.get('id'),
                            },
                            verified=True,  # CoinMarketCal events are verified
                            raw_data=event,
                        )
                        items.append(item)

                    logger.debug(f"CoinMarketCal fetched {len(items)} events")

        except Exception as e:
            logger.error(f"CoinMarketCal fetch error: {e}")
            raise

        return items

    def _categorize_event(self, categories: List[str]) -> str:
        """Map CoinMarketCal categories to our event types."""
        categories_lower = [c.lower() for c in categories]

        if any('exchange' in c or 'listing' in c for c in categories_lower):
            return 'listing'
        if any('airdrop' in c for c in categories_lower):
            return 'airdrop'
        if any('partnership' in c for c in categories_lower):
            return 'partnership'
        if any('update' in c or 'release' in c or 'upgrade' in c for c in categories_lower):
            return 'upgrade'
        if any('conference' in c or 'event' in c for c in categories_lower):
            return 'conference'
        if any('burn' in c for c in categories_lower):
            return 'burn'

        return 'event'
