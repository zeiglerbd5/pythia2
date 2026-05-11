"""
Volume Explosion Scanner

Detects coins with abnormal 24h volume vs 30-day average.
Uses Coinbase API stats endpoint - no warmup needed.

Key metric: volume_multiple = 24h_volume / avg_daily_volume_30d
Signal when volume_multiple >= threshold (e.g., 3x)
"""

import asyncio
import aiohttp
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from loguru import logger


@dataclass
class VolumeSignal:
    symbol: str
    volume_multiple: float
    volume_24h_usd: float
    avg_daily_usd: float
    price: float
    price_change_24h: float
    timestamp: datetime


class VolumeScanner:
    """
    Scans for volume explosions across all symbols.

    Uses Coinbase stats API which provides volume_30day,
    allowing instant baseline calculation with no warmup.
    """

    BASE_URL = "https://api.exchange.coinbase.com"

    def __init__(
        self,
        symbols: List[str],
        signal_callback: Optional[Callable] = None,
        volume_threshold: float = 3.0,  # 3x normal volume
        min_volume_usd: float = 100_000,  # Minimum $100K daily volume
        scan_interval: int = 300,  # 5 minutes
    ):
        self.symbols = list(symbols)
        self.signal_callback = signal_callback
        self.volume_threshold = volume_threshold
        self.min_volume_usd = min_volume_usd
        self.scan_interval = scan_interval

        # Cache of recent signals (avoid duplicate alerts)
        self._recent_signals: Dict[str, datetime] = {}
        self._signal_cooldown = 1800  # 30 min cooldown per symbol

        # Statistics
        self._stats = {
            'scans_completed': 0,
            'signals_detected': 0,
            'last_scan': None,
        }

        # Current volume data for all symbols
        self._volume_data: Dict[str, VolumeSignal] = {}

        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(f"VolumeScanner initialized: {len(symbols)} symbols, threshold={volume_threshold}x")

    async def start(self):
        """Start the volume scanner loop."""
        self._running = True
        self._session = aiohttp.ClientSession()

        logger.info(f"Volume scanner started (interval={self.scan_interval}s)")

        try:
            while self._running:
                await self._scan_cycle()
                await asyncio.sleep(self.scan_interval)
        except asyncio.CancelledError:
            logger.info("Volume scanner cancelled")
        finally:
            if self._session:
                await self._session.close()

    async def stop(self):
        """Stop the scanner."""
        self._running = False
        if self._session:
            await self._session.close()
        logger.info(f"Volume scanner stopped. Stats: {self._stats}")

    async def _scan_cycle(self):
        """One scan cycle across all symbols."""
        start_time = datetime.now(timezone.utc)
        signals_this_cycle = []

        # Fetch in batches
        batch_size = 30
        for i in range(0, len(self.symbols), batch_size):
            batch = self.symbols[i:i + batch_size]
            tasks = [self._fetch_stats(sym) for sym in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for sym, result in zip(batch, results):
                if isinstance(result, VolumeSignal):
                    self._volume_data[sym] = result

                    # Check if this is a signal
                    if self._is_signal(result):
                        signals_this_cycle.append(result)

            # Rate limiting
            await asyncio.sleep(0.1)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        self._stats['scans_completed'] += 1
        self._stats['last_scan'] = start_time

        # Process signals
        for signal in signals_this_cycle:
            await self._process_signal(signal)

        # Log summary periodically
        if self._stats['scans_completed'] % 12 == 1:  # Every hour
            high_volume = [s for s, v in self._volume_data.items() if v.volume_multiple >= 2.0]
            logger.info(
                f"[VOLUME_SCAN] Cycle {self._stats['scans_completed']}: "
                f"{len(self._volume_data)} symbols scanned in {elapsed:.1f}s, "
                f"{len(high_volume)} with 2x+ volume"
            )

    async def _fetch_stats(self, symbol: str) -> Optional[VolumeSignal]:
        """Fetch stats for a single symbol."""
        try:
            url = f"{self.BASE_URL}/products/{symbol}/stats"
            async with self._session.get(url, timeout=10) as response:
                if response.status != 200:
                    return None

                data = await response.json()

                price = float(data.get('last', 0))
                open_price = float(data.get('open', 0))
                volume_24h = float(data.get('volume', 0))
                volume_30d = float(data.get('volume_30day', 0))

                if not price or not volume_30d:
                    return None

                volume_24h_usd = volume_24h * price
                avg_daily_usd = (volume_30d * price) / 30

                if avg_daily_usd < self.min_volume_usd:
                    return None  # Skip low-volume coins

                volume_multiple = volume_24h_usd / avg_daily_usd if avg_daily_usd > 0 else 0
                price_change = (price - open_price) / open_price if open_price > 0 else 0

                return VolumeSignal(
                    symbol=symbol,
                    volume_multiple=volume_multiple,
                    volume_24h_usd=volume_24h_usd,
                    avg_daily_usd=avg_daily_usd,
                    price=price,
                    price_change_24h=price_change,
                    timestamp=datetime.now(timezone.utc)
                )

        except Exception as e:
            return None

    def _is_signal(self, signal: VolumeSignal) -> bool:
        """Check if this qualifies as a volume explosion signal."""
        # Must meet volume threshold
        if signal.volume_multiple < self.volume_threshold:
            return False

        # Check cooldown
        if signal.symbol in self._recent_signals:
            last_signal = self._recent_signals[signal.symbol]
            if (signal.timestamp - last_signal).total_seconds() < self._signal_cooldown:
                return False

        return True

    async def _process_signal(self, signal: VolumeSignal):
        """Process a volume explosion signal."""
        self._recent_signals[signal.symbol] = signal.timestamp
        self._stats['signals_detected'] += 1

        direction = "📈" if signal.price_change_24h > 0 else "📉"

        logger.warning(
            f"{direction} [VOLUME_EXPLOSION] {signal.symbol}: "
            f"{signal.volume_multiple:.1f}x normal volume | "
            f"24h: ${signal.volume_24h_usd:,.0f} vs avg ${signal.avg_daily_usd:,.0f} | "
            f"Price: {signal.price_change_24h:+.1%}"
        )

        # Call callback if set
        if self.signal_callback:
            await self.signal_callback(signal)

    def get_volume_data(self, symbol: str) -> Optional[VolumeSignal]:
        """Get cached volume data for a symbol."""
        return self._volume_data.get(symbol)

    def get_high_volume_symbols(self, threshold: float = 2.0) -> List[VolumeSignal]:
        """Get all symbols currently above volume threshold."""
        return [
            v for v in self._volume_data.values()
            if v.volume_multiple >= threshold
        ]

    def get_statistics(self) -> dict:
        """Get scanner statistics."""
        return {
            **self._stats,
            'symbols_tracked': len(self._volume_data),
            'current_signals': len([
                v for v in self._volume_data.values()
                if v.volume_multiple >= self.volume_threshold
            ])
        }
