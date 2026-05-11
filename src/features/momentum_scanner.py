"""
Momentum Scanner

Detects coins with strong 1-hour price momentum for spike trading.
Based on backtest analysis showing:
- 6%+ 1h momentum: 56% win rate, avg +0.74% per trade
- Volume-based signals missed most big movers (SYND, BOBA)

Key metric: price_change_1h = (current_price / price_1h_ago - 1) * 100
Signal when price_change_1h >= threshold (default 6%)
"""

import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from loguru import logger
import sqlite3


@dataclass
class MomentumSignal:
    """Signal generated when 1h momentum exceeds threshold."""
    symbol: str
    momentum_1h: float  # Percentage change in last hour
    momentum_4h: float  # Percentage change in last 4 hours (context)
    price: float
    volume_usd_1h: float  # Volume in last hour (for reference)
    timestamp: datetime


class MomentumScanner:
    """
    Scans for momentum breakouts across all symbols.

    Uses local SQLite OHLCV data for fast, zero-latency scanning.
    Designed to complement (not replace) volume scanning.
    """

    def __init__(
        self,
        db_path: str = "data/feature_buffer.db",
        signal_callback: Optional[Callable] = None,
        momentum_threshold_1h: float = 6.0,  # 6%+ triggers signal
        max_momentum_1h: float = 25.0,  # Cap to avoid extreme chase
        scan_interval: int = 60,  # 1 minute
    ):
        self.db_path = db_path
        self.signal_callback = signal_callback
        self.momentum_threshold_1h = momentum_threshold_1h
        self.max_momentum_1h = max_momentum_1h
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

        self._running = False
        self._conn: Optional[sqlite3.Connection] = None

        logger.info(f"MomentumScanner initialized: threshold={momentum_threshold_1h}%+, max={max_momentum_1h}%")

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=30)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    async def start(self):
        """Start the momentum scanner loop."""
        self._running = True

        logger.info(f"Momentum scanner started (interval={self.scan_interval}s)")

        try:
            while self._running:
                await self._scan_cycle()
                await asyncio.sleep(self.scan_interval)
        except asyncio.CancelledError:
            logger.info("Momentum scanner cancelled")
        finally:
            if self._conn:
                self._conn.close()
                self._conn = None

    async def stop(self):
        """Stop the scanner."""
        self._running = False
        if self._conn:
            self._conn.close()
            self._conn = None
        logger.info(f"Momentum scanner stopped. Stats: {self._stats}")

    async def _scan_cycle(self):
        """One scan cycle across all symbols."""
        start_time = datetime.now(timezone.utc)
        signals_this_cycle = []

        try:
            conn = self._get_connection()

            # Get all symbols with recent data
            cursor = conn.execute("""
                SELECT DISTINCT symbol FROM ohlcv
                WHERE timestamp >= datetime('now', '-2 hours')
            """)
            symbols = [row[0] for row in cursor.fetchall()]

            for symbol in symbols:
                try:
                    signal = await self._check_symbol(symbol)
                    if signal:
                        signals_this_cycle.append(signal)
                except Exception as e:
                    logger.debug(f"Error checking {symbol}: {e}")

            self._stats['scans_completed'] += 1
            self._stats['last_scan'] = start_time.isoformat()

            # Process detected signals
            for signal in signals_this_cycle:
                # Check cooldown
                if self._is_on_cooldown(signal.symbol):
                    continue

                # Record signal
                self._recent_signals[signal.symbol] = start_time
                self._stats['signals_detected'] += 1

                logger.warning(
                    f"[MOMENTUM] {signal.symbol}: +{signal.momentum_1h:.1f}% 1h "
                    f"(4h: {signal.momentum_4h:+.1f}%) @ ${signal.price:.6f}"
                )

                # Trigger callback
                if self.signal_callback:
                    try:
                        if asyncio.iscoroutinefunction(self.signal_callback):
                            await self.signal_callback(signal)
                        else:
                            self.signal_callback(signal)
                    except Exception as e:
                        logger.error(f"Signal callback error: {e}")

        except Exception as e:
            logger.error(f"Momentum scan cycle error: {e}")

        if signals_this_cycle:
            logger.info(f"[MOMENTUM] Scan complete: {len(signals_this_cycle)} signals from {len(symbols)} symbols")

    async def _check_symbol(self, symbol: str) -> Optional[MomentumSignal]:
        """Check single symbol for momentum signal."""
        conn = self._get_connection()

        # Get current price
        cursor = conn.execute("""
            SELECT close, volume FROM ohlcv
            WHERE symbol = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (symbol,))
        current = cursor.fetchone()
        if not current:
            return None

        current_price = current[0]

        # Get price 1 hour ago
        cursor = conn.execute("""
            SELECT close FROM ohlcv
            WHERE symbol = ?
              AND timestamp <= datetime('now', '-1 hour')
            ORDER BY timestamp DESC LIMIT 1
        """, (symbol,))
        price_1h = cursor.fetchone()

        # Get price 4 hours ago
        cursor = conn.execute("""
            SELECT close FROM ohlcv
            WHERE symbol = ?
              AND timestamp <= datetime('now', '-4 hours')
            ORDER BY timestamp DESC LIMIT 1
        """, (symbol,))
        price_4h = cursor.fetchone()

        # Get 1h volume
        cursor = conn.execute("""
            SELECT SUM(volume * close) FROM ohlcv
            WHERE symbol = ?
              AND timestamp >= datetime('now', '-1 hour')
        """, (symbol,))
        vol_result = cursor.fetchone()
        volume_1h = vol_result[0] if vol_result and vol_result[0] else 0

        # Calculate momentum
        if not price_1h or price_1h[0] <= 0:
            return None

        momentum_1h = ((current_price / price_1h[0]) - 1) * 100
        momentum_4h = ((current_price / price_4h[0]) - 1) * 100 if price_4h and price_4h[0] > 0 else 0

        # Check threshold
        if momentum_1h < self.momentum_threshold_1h:
            return None

        # Check cap (avoid extreme chase)
        if momentum_1h > self.max_momentum_1h:
            logger.info(f"[MOMENTUM] {symbol} exceeds cap: {momentum_1h:.1f}% > {self.max_momentum_1h}%")
            return None

        return MomentumSignal(
            symbol=symbol,
            momentum_1h=momentum_1h,
            momentum_4h=momentum_4h,
            price=current_price,
            volume_usd_1h=volume_1h,
            timestamp=datetime.now(timezone.utc),
        )

    def _is_on_cooldown(self, symbol: str) -> bool:
        """Check if symbol is on signal cooldown."""
        if symbol in self._recent_signals:
            last_signal = self._recent_signals[symbol]
            elapsed = (datetime.now(timezone.utc) - last_signal).total_seconds()
            if elapsed < self._signal_cooldown:
                return True
        return False

    def get_statistics(self) -> Dict:
        """Get scanner statistics."""
        return {
            **self._stats,
            'recent_signals': len(self._recent_signals),
        }
