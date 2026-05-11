#!/usr/bin/env python3
"""
Catalyst Monitor Runner

Runs the catalyst detection loop and outputs signals to the console.
Can be run standalone or integrated with the trading system.

Usage:
    python scripts/run_catalyst_monitor.py
    python scripts/run_catalyst_monitor.py --interval 30
    python scripts/run_catalyst_monitor.py --min-priority 0.5
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.signals.catalyst_detector import CatalystDetector, CatalystSignal


def format_signal(signal: CatalystSignal) -> str:
    """Format a signal for console output."""
    urgency_colors = {
        "IMMEDIATE": "\033[91m",  # Red
        "SOON": "\033[93m",       # Yellow
        "MONITOR": "\033[92m",    # Green
    }
    action_colors = {
        "ENTER": "\033[91m\033[1m",  # Bold Red
        "PREPARE": "\033[93m",        # Yellow
        "WATCH": "\033[92m",          # Green
        "AVOID": "\033[90m",          # Gray
    }
    reset = "\033[0m"

    urgency_color = urgency_colors.get(signal.urgency, "")
    action_color = action_colors.get(signal.action, "")

    lines = [
        f"\n{'='*60}",
        f"  {action_color}[{signal.action}]{reset} {signal.symbol} - {signal.catalyst_type.upper()}",
        f"{'='*60}",
        f"  Headline: {signal.headline[:70]}{'...' if len(signal.headline) > 70 else ''}",
        f"  Priority: {signal.priority_score:.2f} | Confidence: {signal.confidence:.2f} | Impact: {signal.impact_score:.2f}",
        f"  {urgency_color}Urgency: {signal.urgency}{reset} | Sources: {len(signal.sources)} | Corroborated: {signal.corroborating_signals}x",
    ]

    if signal.entry_window_minutes:
        lines.append(f"  Entry Window: ~{signal.entry_window_minutes} minutes")

    if signal.source_urls:
        lines.append(f"  URL: {signal.source_urls[0]}")

    return "\n".join(lines)


async def signal_callback(signal: CatalystSignal):
    """Callback for new high-priority signals."""
    print(format_signal(signal))

    # Could add: send to Discord, Telegram, or trading engine here
    if signal.action == "ENTER":
        logger.warning(f"🚨 ENTER SIGNAL: {signal.symbol} ({signal.catalyst_type})")


async def main(args):
    """Main entry point."""
    # Configure logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr,
        level=log_level,
        format="<level>{level: <8}</level> | <cyan>{time:HH:mm:ss}</cyan> | {message}"
    )

    # Check API keys
    cryptopanic = os.getenv('CRYPTOPANIC_API_KEY')
    coinmarketcal = os.getenv('COINMARKETCAL_API_KEY')

    print("\n" + "="*60)
    print("CATALYST MONITOR")
    print("="*60)
    print(f"\nAPI Keys:")
    print(f"  CRYPTOPANIC_API_KEY: {'✓ Set' if cryptopanic else '✗ Not set'}")
    print(f"  COINMARKETCAL_API_KEY: {'✓ Set' if coinmarketcal else '✗ Not set'}")
    print(f"\nSettings:")
    print(f"  Poll interval: {args.interval}s")
    print(f"  Min priority: {args.min_priority}")
    print(f"\nStarting monitor...")
    print("-"*60 + "\n")

    # Create detector
    detector = CatalystDetector(
        enable_cryptopanic=True,
        enable_unlocks=True,
        enable_calendar=True,
        enable_exchange_listings=True,
        enable_twitter=True,
        poll_interval_seconds=args.interval,
    )

    # Register callback for high-priority signals
    detector.register_callback(signal_callback)

    # Initial fetch
    await detector._fetch_and_process()

    # Show initial signals
    signals = detector.get_active_signals(min_priority=args.min_priority)
    if signals:
        print(f"\n📊 Initial signals ({len(signals)} found):")
        for signal in signals[:10]:  # Top 10
            print(format_signal(signal))
    else:
        print("\n📊 No signals above threshold currently")

    # Start continuous monitoring
    print(f"\n🔄 Starting continuous monitoring (Ctrl+C to stop)...")

    try:
        await detector.start()

        # Keep running until interrupted
        while True:
            await asyncio.sleep(args.interval)

            # Periodic status update
            health = detector.get_health_status()
            active = len(detector.get_active_signals(min_priority=args.min_priority))
            timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")

            logger.info(f"[{timestamp}] Active signals: {active} | Total processed: {health['total_signals_processed']}")

    except KeyboardInterrupt:
        print("\n\n⏹ Stopping monitor...")
        await detector.stop()
        print("Monitor stopped.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Catalyst Monitor")
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        help="Poll interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--min-priority", "-p",
        type=float,
        default=0.4,
        help="Minimum priority score to display (default: 0.4)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
