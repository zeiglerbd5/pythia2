"""
Real-time Candle Aggregator

Aggregates WebSocket ticker/trade updates into 1-minute OHLCV candles.
Provides completed candles to downstream feature calculator.
"""

import asyncio
from typing import Dict, Optional, Callable
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class PartialCandle:
    """Represents a candle being built from incoming ticks."""
    symbol: str
    timestamp: datetime  # Start of 1-minute period
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    num_trades: int = 0

    def update_from_ticker(self, price: float):
        """Update candle from ticker price update."""
        if self.open is None:
            self.open = price

        if self.high is None or price > self.high:
            self.high = price

        if self.low is None or price < self.low:
            self.low = price

        self.close = price

    def update_from_trade(self, price: float, size: float, side: str):
        """Update candle from trade execution."""
        self.update_from_ticker(price)

        self.volume += size
        self.num_trades += 1

        if side == 'BUY':
            self.buy_volume += size
        else:
            self.sell_volume += size

    def is_complete(self) -> bool:
        """Check if candle has all required OHLC data."""
        return all([
            self.open is not None,
            self.high is not None,
            self.low is not None,
            self.close is not None
        ])

    def to_dict(self) -> Dict:
        """Convert to dictionary for downstream processing."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'num_trades': self.num_trades,
        }


class CandleAggregator:
    """
    Aggregates real-time market data into 1-minute candles.

    Features:
    - Handles ticker and trade updates
    - Automatic candle completion at minute boundaries
    - Callback notification for completed candles
    - Missing candle filling (optional)
    """

    def __init__(
        self,
        symbols: list[str],
        on_candle_complete: Optional[Callable] = None,
        fill_missing: bool = True
    ):
        """
        Initialize candle aggregator.

        Args:
            symbols: List of symbols to track
            on_candle_complete: Callback function(symbol, candle_dict)
            fill_missing: Whether to fill missing candles with previous close
        """
        self.symbols = symbols
        self.on_candle_complete = on_candle_complete
        self.fill_missing = fill_missing

        # Current partial candles being built (one per symbol)
        self.partial_candles: Dict[str, PartialCandle] = {}

        # Last completed candle for each symbol (for filling gaps)
        self.last_candles: Dict[str, Dict] = {}

        # Statistics
        self.candles_completed = 0
        self.candles_filled = 0

        logger.info(f"CandleAggregator initialized for {len(symbols)} symbols")

    def _get_current_minute_timestamp(self) -> datetime:
        """Get timestamp for current 1-minute period (floor to minute)."""
        now = datetime.now(timezone.utc)
        return now.replace(second=0, microsecond=0)

    def _get_partial_candle(self, symbol: str) -> PartialCandle:
        """Get or create partial candle for current minute."""
        current_minute = self._get_current_minute_timestamp()

        # Check if we need to complete previous candle and start new one
        if symbol in self.partial_candles:
            existing = self.partial_candles[symbol]

            if existing.timestamp < current_minute:
                # Previous candle is complete - finalize it
                self._finalize_candle(symbol, existing)

                # Start new candle
                self.partial_candles[symbol] = PartialCandle(
                    symbol=symbol,
                    timestamp=current_minute
                )
        else:
            # First candle for this symbol
            self.partial_candles[symbol] = PartialCandle(
                symbol=symbol,
                timestamp=current_minute
            )

        return self.partial_candles[symbol]

    def _finalize_candle(self, symbol: str, candle: PartialCandle):
        """Finalize and emit completed candle."""
        if not candle.is_complete():
            logger.warning(
                f"Incomplete candle for {symbol} at {candle.timestamp}, "
                f"open={candle.open}, high={candle.high}, low={candle.low}, close={candle.close}"
            )

            # Try to fill from last candle if available
            if self.fill_missing and symbol in self.last_candles:
                last = self.last_candles[symbol]
                if candle.open is None:
                    candle.open = last['close']
                if candle.high is None:
                    candle.high = last['close']
                if candle.low is None:
                    candle.low = last['close']
                if candle.close is None:
                    candle.close = last['close']

                self.candles_filled += 1
                logger.debug(f"Filled incomplete candle for {symbol} with last close={last['close']}")

        # Store as last candle
        candle_dict = candle.to_dict()
        self.last_candles[symbol] = candle_dict

        # Emit via callback
        if self.on_candle_complete:
            try:
                self.on_candle_complete(symbol, candle_dict)
            except Exception as e:
                logger.error(f"Error in candle completion callback: {e}")

        self.candles_completed += 1

        if self.candles_completed % 100 == 0:
            logger.info(
                f"Candles completed: {self.candles_completed} "
                f"(filled: {self.candles_filled})"
            )

    def on_ticker_update(self, symbol: str, price: float, volume_24h: Optional[float] = None):
        """
        Handle ticker update from WebSocket.

        Args:
            symbol: Trading pair symbol
            price: Current price
            volume_24h: 24-hour volume (optional, not used for candles)
        """
        if symbol not in self.symbols:
            return

        candle = self._get_partial_candle(symbol)
        candle.update_from_ticker(price)

    def on_trade_update(self, symbol: str, price: float, size: float, side: str):
        """
        Handle trade update from WebSocket.

        Args:
            symbol: Trading pair symbol
            price: Trade price
            size: Trade size
            side: 'BUY' or 'SELL'
        """
        if symbol not in self.symbols:
            return

        candle = self._get_partial_candle(symbol)
        candle.update_from_trade(price, size, side)

    async def periodic_flush(self, interval_seconds: int = 5):
        """
        Periodically check for completed candles and flush them.

        This ensures candles are completed even if no new data arrives.
        Run this as a background task.

        Args:
            interval_seconds: How often to check for completed candles
        """
        logger.info(f"Starting periodic candle flush (every {interval_seconds}s)")

        while True:
            await asyncio.sleep(interval_seconds)

            current_minute = self._get_current_minute_timestamp()

            # Check all partial candles
            symbols_to_flush = []
            for symbol, candle in self.partial_candles.items():
                if candle.timestamp < current_minute:
                    symbols_to_flush.append(symbol)

            # Flush completed candles
            for symbol in symbols_to_flush:
                candle = self.partial_candles[symbol]
                self._finalize_candle(symbol, candle)

                # Start new candle
                self.partial_candles[symbol] = PartialCandle(
                    symbol=symbol,
                    timestamp=current_minute
                )

    def get_statistics(self) -> Dict:
        """Get aggregator statistics."""
        return {
            'candles_completed': self.candles_completed,
            'candles_filled': self.candles_filled,
            'symbols_tracked': len(self.symbols),
            'active_partial_candles': len(self.partial_candles),
        }
