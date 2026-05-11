"""
Catalyst Data Backfill Script

Fetches historical catalyst data from available APIs:
- Binance listing announcements (paginated API)
- CoinMarketCal events (date range queries)

Stores in DuckDB for correlation with historical price data.
"""

import asyncio
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import aiohttp
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Import database manager
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_ingestion.database import DuckDBManager


class CatalystBackfiller:
    """Backfills historical catalyst data from available APIs."""

    # Binance announcements API
    BINANCE_API_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
    BINANCE_LISTING_CATALOG_ID = 48  # "New Cryptocurrency Listing"

    # CoinMarketCal API
    COINMARKETCAL_API_URL = "https://developers.coinmarketcal.com/v1/events"

    def __init__(self, db_path: Optional[str] = None):
        """Initialize backfiller."""
        self.db_path = db_path or os.getenv("DATABASE_PATH", "data/pythia.duckdb")
        self.db_manager = DuckDBManager(db_path=self.db_path)

        self.coinmarketcal_api_key = os.getenv("COINMARKETCAL_API_KEY")

        # Track stats
        self.stats = {
            "binance_fetched": 0,
            "coinmarketcal_fetched": 0,
            "total_stored": 0,
        }

    async def backfill_all(
        self,
        days_back: int = 90,
        binance_pages: int = 50,
    ):
        """
        Run full backfill from all sources.

        Args:
            days_back: How many days of CoinMarketCal events to fetch
            binance_pages: How many pages of Binance announcements to fetch
        """
        logger.info(f"Starting catalyst backfill: {days_back} days, {binance_pages} Binance pages")

        async with aiohttp.ClientSession() as session:
            # Backfill in parallel
            await asyncio.gather(
                self._backfill_binance(session, binance_pages),
                self._backfill_coinmarketcal(session, days_back),
            )

        logger.info(f"Backfill complete: {self.stats}")
        return self.stats

    async def _backfill_binance(self, session: aiohttp.ClientSession, max_pages: int = 50):
        """Fetch historical Binance listing announcements."""
        logger.info(f"Backfilling Binance announcements ({max_pages} pages)...")

        total_articles = 0

        for page in range(1, max_pages + 1):
            try:
                params = {
                    "type": 1,
                    "catalogId": self.BINANCE_LISTING_CATALOG_ID,
                    "pageNo": page,
                    "pageSize": 20,
                }

                headers = {
                    "User-Agent": "Mozilla/5.0 (compatible; CatalystBackfill/1.0)",
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate",
                }

                async with session.get(
                    self.BINANCE_API_URL,
                    params=params,
                    headers=headers,
                    timeout=30,
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Binance API error on page {page}: {response.status}")
                        break

                    data = await response.json()

                    catalogs = data.get("data", {}).get("catalogs", [])
                    page_articles = 0

                    for catalog in catalogs:
                        for article in catalog.get("articles", []):
                            self._store_binance_article(article)
                            page_articles += 1
                            total_articles += 1

                    if page_articles == 0:
                        logger.info(f"No more articles at page {page}, stopping")
                        break

                    if page % 10 == 0:
                        logger.info(f"Binance page {page}: {total_articles} total articles")

                    # Be nice to the API
                    await asyncio.sleep(0.2)

            except Exception as e:
                logger.error(f"Error fetching Binance page {page}: {e}")
                break

        self.stats["binance_fetched"] = total_articles
        logger.info(f"Binance backfill complete: {total_articles} articles")

    def _store_binance_article(self, article: Dict[str, Any]):
        """Store a Binance article as a news signal."""
        import re

        article_id = article.get("id")
        title = article.get("title", "")
        code = article.get("code", "")
        release_date = article.get("releaseDate", 0)

        # Parse timestamp
        try:
            timestamp = datetime.fromtimestamp(release_date / 1000, tz=timezone.utc)
        except:
            timestamp = datetime.now(timezone.utc)

        # Extract symbols from title
        symbols = self._extract_symbols(title)

        # Classify event type
        event_type = self._classify_binance_event(title)

        # Build URL
        url = f"https://www.binance.com/en/support/announcement/{code}"

        # Store for each symbol
        for symbol in symbols if symbols else ["UNKNOWN-USD"]:
            signal_dict = {
                "symbol": symbol,
                "timestamp": timestamp,
                "source": "binance_announcements",
                "event_type": event_type,
                "confidence": 0.95,
                "title": title[:500],
                "url": url,
                "signal_hash": f"binance:{article_id}:{symbol}",
                "source_credibility": 1.0,
                "entity_certainty": 0.95 if symbols else 0.3,
                "event_priority": 0.9 if event_type == "listing" else 0.7,
                "recency_score": 1.0,
                "engagement_score": 0.0,
                "sentiment_score": 0.8,
            }
            self.db_manager.queue_news_signal(signal_dict)
            self.stats["total_stored"] += 1

    async def _backfill_coinmarketcal(self, session: aiohttp.ClientSession, days_back: int = 90):
        """Fetch historical CoinMarketCal events."""
        if not self.coinmarketcal_api_key:
            logger.warning("CoinMarketCal API key not set, skipping")
            return

        logger.info(f"Backfilling CoinMarketCal events ({days_back} days)...")

        total_events = 0

        # Fetch in chunks (API may limit date ranges)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)

        # Process in 30-day chunks
        chunk_days = 30
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)

            try:
                headers = {
                    "x-api-key": self.coinmarketcal_api_key,
                    "Accept": "application/json",
                }

                params = {
                    "max": 100,
                    "dateRangeStart": current_start.strftime("%Y-%m-%d"),
                    "dateRangeEnd": current_end.strftime("%Y-%m-%d"),
                }

                async with session.get(
                    self.COINMARKETCAL_API_URL,
                    headers=headers,
                    params=params,
                    timeout=30,
                ) as response:
                    if response.status != 200:
                        logger.warning(f"CoinMarketCal API error: {response.status}")
                        break

                    data = await response.json()
                    events = data.get("body", [])

                    for event in events:
                        self._store_coinmarketcal_event(event)
                        total_events += 1

                    logger.info(
                        f"CoinMarketCal {current_start.date()} to {current_end.date()}: "
                        f"{len(events)} events, {total_events} total"
                    )

                    await asyncio.sleep(0.5)  # Rate limit

            except Exception as e:
                logger.error(f"Error fetching CoinMarketCal: {e}")

            current_start = current_end

        self.stats["coinmarketcal_fetched"] = total_events
        logger.info(f"CoinMarketCal backfill complete: {total_events} events")

    def _store_coinmarketcal_event(self, event: Dict[str, Any]):
        """Store a CoinMarketCal event as a news signal."""
        # Extract coins
        symbols = []
        for coin in event.get("coins", []):
            symbol = coin.get("symbol")
            if symbol and symbol != "CRYPTO":
                symbols.append(f"{symbol}-USD")

        # Parse date
        try:
            date_str = event.get("date_event", "")
            if "." in date_str:
                event_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            else:
                event_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
            event_date = event_date.replace(tzinfo=timezone.utc)
        except:
            event_date = datetime.now(timezone.utc)

        # Map category to event type
        categories = [c.get("name", "") for c in event.get("categories", [])]
        event_type = self._categorize_coinmarketcal_event(categories)

        # Get title
        title_obj = event.get("title", {})
        title = title_obj.get("en", "") if isinstance(title_obj, dict) else str(title_obj)

        # Store for each symbol
        for symbol in symbols if symbols else ["UNKNOWN-USD"]:
            signal_dict = {
                "symbol": symbol,
                "timestamp": event_date,
                "source": "coinmarketcal",
                "event_type": event_type,
                "confidence": 0.8,
                "title": title[:500] if title else ", ".join(categories),
                "url": event.get("source"),
                "signal_hash": f"cmc:{event.get('id')}:{symbol}",
                "source_credibility": 0.75,
                "entity_certainty": 0.9 if symbols else 0.3,
                "event_priority": self._get_event_priority(event_type),
                "recency_score": 1.0,
                "engagement_score": 0.0,
                "sentiment_score": 0.7,
            }
            self.db_manager.queue_news_signal(signal_dict)
            self.stats["total_stored"] += 1

    def _extract_symbols(self, content: str) -> List[str]:
        """Extract crypto symbols from content."""
        import re

        symbols = set()
        content_upper = content.upper()

        # Look for symbols in parentheses (e.g., "Katana (KAT)")
        parens = re.findall(r"\(([A-Z]{2,10})\)", content_upper)
        for sym in parens:
            if sym not in {"USD", "USDT", "USDC", "EUR", "GBP", "GMT", "UTC", "API", "VIP"}:
                symbols.add(f"{sym}-USD")

        # Look for XXXUSDT pattern
        usdt_pairs = re.findall(r"\b([A-Z]{2,10})USDT\b", content_upper)
        for sym in usdt_pairs:
            if sym not in {"USD"}:
                symbols.add(f"{sym}-USD")

        return list(symbols)

    def _classify_binance_event(self, title: str) -> str:
        """Classify Binance announcement type."""
        title_lower = title.lower()

        if "delist" in title_lower:
            return "delisting"
        elif "list" in title_lower or "will add" in title_lower:
            return "listing"
        elif "airdrop" in title_lower:
            return "airdrop"
        elif "futures" in title_lower or "perpetual" in title_lower:
            return "futures_listing"
        elif "margin" in title_lower:
            return "margin_listing"
        else:
            return "announcement"

    def _categorize_coinmarketcal_event(self, categories: List[str]) -> str:
        """Map CoinMarketCal categories to event types."""
        categories_lower = [c.lower() for c in categories]

        if any("exchange" in c or "listing" in c for c in categories_lower):
            return "listing"
        if any("airdrop" in c for c in categories_lower):
            return "airdrop"
        if any("partnership" in c for c in categories_lower):
            return "partnership"
        if any("update" in c or "release" in c or "upgrade" in c for c in categories_lower):
            return "upgrade"
        if any("conference" in c or "event" in c for c in categories_lower):
            return "conference"
        if any("burn" in c for c in categories_lower):
            return "burn"

        return "event"

    def _get_event_priority(self, event_type: str) -> float:
        """Get priority score for event type."""
        priorities = {
            "listing": 0.95,
            "airdrop": 0.85,
            "futures_listing": 0.80,
            "delisting": 0.75,
            "partnership": 0.60,
            "upgrade": 0.55,
            "burn": 0.50,
            "conference": 0.40,
            "event": 0.30,
            "announcement": 0.25,
        }
        return priorities.get(event_type, 0.3)

    async def flush_and_close(self):
        """Flush any remaining data and close."""
        # Force flush the database queues
        await self.db_manager._flush_news_signals()
        logger.info("Database flushed")


async def main():
    """Run the backfill."""
    import argparse

    parser = argparse.ArgumentParser(description="Backfill catalyst data")
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch")
    parser.add_argument("--binance-pages", type=int, default=50, help="Binance pages to fetch")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    args = parser.parse_args()

    backfiller = CatalystBackfiller(db_path=args.db)

    try:
        stats = await backfiller.backfill_all(
            days_back=args.days,
            binance_pages=args.binance_pages,
        )
        await backfiller.flush_and_close()

        print("\n" + "=" * 60)
        print("BACKFILL COMPLETE")
        print("=" * 60)
        print(f"Binance announcements: {stats['binance_fetched']}")
        print(f"CoinMarketCal events:  {stats['coinmarketcal_fetched']}")
        print(f"Total signals stored:  {stats['total_stored']}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nBackfill interrupted")
        await backfiller.flush_and_close()


if __name__ == "__main__":
    asyncio.run(main())
