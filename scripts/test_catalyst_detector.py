#!/usr/bin/env python3
"""
Test script for the catalyst detection pipeline.

Runs a single fetch cycle to verify all sources are working.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.signals.catalyst_detector import CatalystDetector, check_catalysts


async def test_individual_sources():
    """Test each source individually."""
    from src.signals.sources.cryptopanic import CryptoPanicSource
    from src.signals.sources.event_calendar import TokenUnlocksSource, CoinMarketCalSource
    from src.signals.sources.exchange_listings import ExchangeListingsSource
    from src.signals.sources.twitter_rss import TwitterRSSSource

    sources = [
        ("CryptoPanic", CryptoPanicSource()),
        ("TokenUnlocks", TokenUnlocksSource()),
        ("CoinMarketCal", CoinMarketCalSource()),
    ]

    # Check if exchange listings source exists
    try:
        sources.append(("ExchangeListings", ExchangeListingsSource()))
    except Exception as e:
        logger.warning(f"ExchangeListingsSource not available: {e}")

    # Check if twitter source exists
    try:
        sources.append(("TwitterRSS", TwitterRSSSource()))
    except Exception as e:
        logger.warning(f"TwitterRSSSource not available: {e}")

    print("\n" + "="*60)
    print("TESTING INDIVIDUAL SOURCES")
    print("="*60 + "\n")

    for name, source in sources:
        print(f"\n--- {name} ---")
        print(f"  Source name: {source.source_name}")
        print(f"  Credibility: {source.source_credibility}")
        print(f"  Healthy: {source.is_healthy()}")

        try:
            items = await source.fetch_with_rate_limit()
            print(f"  Items fetched: {len(items)}")

            if items:
                # Show first item
                item = items[0]
                print(f"  Sample item:")
                print(f"    - Type: {item.event_type}")
                print(f"    - Title: {item.title[:60]}..." if len(item.title) > 60 else f"    - Title: {item.title}")
                print(f"    - Symbols: {item.symbols[:3]}..." if len(item.symbols) > 3 else f"    - Symbols: {item.symbols}")
        except Exception as e:
            print(f"  ERROR: {e}")


async def test_catalyst_detector():
    """Test the unified catalyst detector."""
    print("\n" + "="*60)
    print("TESTING CATALYST DETECTOR")
    print("="*60 + "\n")

    # Create detector with all sources
    detector = CatalystDetector(
        enable_cryptopanic=True,
        enable_unlocks=True,
        enable_calendar=True,
        enable_exchange_listings=True,
        enable_twitter=True,
    )

    print(f"Sources initialized: {len(detector.sources)}")

    # Run a single fetch cycle
    print("\nRunning fetch cycle...")
    await detector._fetch_and_process()

    # Get signals
    signals = detector.get_active_signals(min_priority=0.0)
    print(f"\nSignals detected: {len(signals)}")

    # Show top signals
    if signals:
        print("\n--- Top Signals ---")
        for signal in signals[:5]:
            print(f"\n  [{signal.symbol}] {signal.catalyst_type.upper()}")
            print(f"    Headline: {signal.headline[:50]}..." if len(signal.headline) > 50 else f"    Headline: {signal.headline}")
            print(f"    Priority: {signal.priority_score:.2f} | Confidence: {signal.confidence:.2f} | Impact: {signal.impact_score:.2f}")
            print(f"    Urgency: {signal.urgency} | Action: {signal.action}")
            print(f"    Sources: {signal.sources}")

    # Health status
    health = detector.get_health_status()
    print("\n--- Health Status ---")
    print(f"  Active signals: {health['active_signals']}")
    print(f"  Total processed: {health['total_signals_processed']}")
    print(f"  Sources:")
    for src in health['sources']:
        status = "✓" if src['healthy'] else "✗"
        print(f"    {status} {src['name']} (credibility: {src['credibility']})")


async def test_quick_check():
    """Test the convenience check_catalysts function."""
    print("\n" + "="*60)
    print("TESTING QUICK CHECK FUNCTION")
    print("="*60 + "\n")

    signals = await check_catalysts()
    print(f"Quick check found {len(signals)} signals")


async def main():
    """Run all tests."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")

    print("\n" + "#"*60)
    print("# CATALYST DETECTION PIPELINE TEST")
    print("#"*60)

    # Check environment variables
    print("\n--- Environment Check ---")
    cryptopanic_key = os.getenv('CRYPTOPANIC_API_KEY')
    coinmarketcal_key = os.getenv('COINMARKETCAL_API_KEY')
    print(f"  CRYPTOPANIC_API_KEY: {'Set' if cryptopanic_key else 'NOT SET'}")
    print(f"  COINMARKETCAL_API_KEY: {'Set' if coinmarketcal_key else 'NOT SET'}")

    if not cryptopanic_key:
        print("\n  Note: CryptoPanic source will be disabled without API key")
        print("  Get a free key at: https://cryptopanic.com/developers/api/")

    # Run tests
    await test_individual_sources()
    await test_catalyst_detector()
    await test_quick_check()

    print("\n" + "#"*60)
    print("# TEST COMPLETE")
    print("#"*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
