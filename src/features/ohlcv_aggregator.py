"""
OHLCV Candle Aggregator

Generates OHLCV candles from trade data for multiple timeframes.

Per implementation guide:
- Primary: 5-minute bars (optimal for scalping)
- Confirmation: 15-minute bars (trend alignment)
- Entry: 1-minute bars (precise timing)

Aggregates trades into candles in real-time with support for
multiple concurrent timeframes.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict, deque

import pandas as pd
import numpy as np
from loguru import logger


class OHLCVCandle:
    """
    Represents a single OHLCV candle.
    """

    def __init__(self, timestamp: datetime, timeframe: str, symbol: Optional[str] = None):
        """
        Initialize empty candle.

        Args:
            timestamp: Candle start time
            timeframe: Timeframe string ('1m', '5m', '15m', etc.)
            symbol: Trading pair symbol (optional)
        """
        self.timestamp = timestamp
        self.timeframe = timeframe
        self.symbol = symbol

        # OHLCV data
        self.open: Optional[float] = None
        self.high: Optional[float] = None
        self.low: Optional[float] = None
        self.close: Optional[float] = None
        self.volume: float = 0.0

        # Additional metrics
        self.num_trades: int = 0
        self.buy_volume: float = 0.0
        self.sell_volume: float = 0.0

        # Track first and last trade times
        self.first_trade_time: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None

    def add_trade(self, price: float, size: float, side: str, timestamp: datetime):
        """
        Add a trade to this candle.

        Args:
            price: Trade price
            size: Trade size
            side: Trade side ('BUY' or 'SELL')
            timestamp: Trade timestamp
        """
        # Set open price (first trade)
        if self.open is None:
            self.open = price
            self.first_trade_time = timestamp

        # Update high/low
        if self.high is None or price > self.high:
            self.high = price

        if self.low is None or price < self.low:
            self.low = price

        # Update close (latest trade)
        self.close = price
        self.last_trade_time = timestamp

        # Update volume
        self.volume += size
        self.num_trades += 1

        # Track buy/sell volume
        if side == 'BUY':
            self.buy_volume += size
        elif side == 'SELL':
            self.sell_volume += size

    def is_complete(self) -> bool:
        """Check if candle has at least one trade."""
        return self.open is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            'timestamp': self.timestamp,
            'timeframe': self.timeframe,
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'num_trades': self.num_trades,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
        }

    def __repr__(self) -> str:
        return (
            f"OHLCVCandle({self.timeframe} @ {self.timestamp}: "
            f"O={self.open:.2f} H={self.high:.2f} L={self.low:.2f} "
            f"C={self.close:.2f} V={self.volume:.2f})"
        )


class OHLCVAggregator:
    """
    Real-time OHLCV candle aggregator for multiple timeframes.

    Supports concurrent aggregation of 1m, 5m, 15m, 1h, etc.
    """

    # Timeframe to minutes mapping
    TIMEFRAME_MINUTES = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
    }

    def __init__(self, timeframes: List[str]):
        """
        Initialize aggregator for multiple timeframes.

        Args:
            timeframes: List of timeframes ('1m', '5m', '15m', etc.)
        """
        self.timeframes = timeframes

        # Validate timeframes
        for tf in timeframes:
            if tf not in self.TIMEFRAME_MINUTES:
                raise ValueError(f"Unsupported timeframe: {tf}")

        # Current candles per symbol per timeframe
        # {symbol: {timeframe: OHLCVCandle}}
        self.current_candles: Dict[str, Dict[str, OHLCVCandle]] = defaultdict(dict)

        # Completed candles queue
        self.completed_candles: deque = deque(maxlen=1000)

        logger.info(
            f"OHLCVAggregator initialized",
            extra={"timeframes": timeframes}
        )

    def get_candle_timestamp(self, timestamp: datetime, timeframe: str) -> datetime:
        """
        Get candle start timestamp for a given trade timestamp.

        Rounds down to the nearest timeframe boundary.

        Args:
            timestamp: Trade timestamp
            timeframe: Timeframe string

        Returns:
            Candle start timestamp
        """
        minutes = self.TIMEFRAME_MINUTES[timeframe]

        # Round down to timeframe boundary
        total_minutes = timestamp.hour * 60 + timestamp.minute
        candle_start_minutes = (total_minutes // minutes) * minutes

        candle_start = timestamp.replace(
            hour=candle_start_minutes // 60,
            minute=candle_start_minutes % 60,
            second=0,
            microsecond=0
        )

        return candle_start

    def add_trade(
        self,
        symbol: str,
        price: float,
        size: float,
        side: str,
        timestamp: datetime
    ) -> List[OHLCVCandle]:
        """
        Add a trade and aggregate into candles.

        Args:
            symbol: Trading pair symbol
            price: Trade price
            size: Trade size
            side: Trade side ('BUY' or 'SELL')
            timestamp: Trade timestamp

        Returns:
            List of completed candles (if any candles closed)
        """
        completed = []

        for timeframe in self.timeframes:
            # Get candle timestamp for this timeframe
            candle_ts = self.get_candle_timestamp(timestamp, timeframe)

            # Check if we need to create a new candle
            if timeframe not in self.current_candles[symbol]:
                # First candle for this symbol/timeframe
                candle = OHLCVCandle(candle_ts, timeframe, symbol)
                self.current_candles[symbol][timeframe] = candle

            else:
                current_candle = self.current_candles[symbol][timeframe]

                # Check if trade belongs to new candle
                if candle_ts != current_candle.timestamp:
                    # Complete current candle
                    if current_candle.is_complete():
                        completed.append(current_candle)
                        self.completed_candles.append(current_candle)

                    # Create new candle
                    candle = OHLCVCandle(candle_ts, timeframe, symbol)
                    self.current_candles[symbol][timeframe] = candle

            # Add trade to current candle
            self.current_candles[symbol][timeframe].add_trade(price, size, side, timestamp)

        return completed

    def get_current_candle(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[OHLCVCandle]:
        """
        Get current (incomplete) candle for a symbol and timeframe.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string

        Returns:
            Current candle or None
        """
        return self.current_candles.get(symbol, {}).get(timeframe)

    def get_completed_candles(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[OHLCVCandle]:
        """
        Get completed candles with optional filtering.

        Args:
            symbol: Filter by symbol (optional)
            timeframe: Filter by timeframe (optional)
            limit: Maximum number of candles to return (optional)

        Returns:
            List of completed candles (newest first)
        """
        candles = list(reversed(self.completed_candles))

        # Filter by symbol
        if symbol:
            candles = [c for c in candles if c.to_dict().get('symbol') == symbol]

        # Filter by timeframe
        if timeframe:
            candles = [c for c in candles if c.timeframe == timeframe]

        # Apply limit
        if limit:
            candles = candles[:limit]

        return candles

    def force_close_candles(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None
    ) -> List[OHLCVCandle]:
        """
        Force close current candles (e.g., at end of day).

        Args:
            symbol: Trading pair symbol
            timeframes: Timeframes to close (optional, defaults to all)

        Returns:
            List of closed candles
        """
        if timeframes is None:
            timeframes = self.timeframes

        completed = []

        for timeframe in timeframes:
            candle = self.current_candles.get(symbol, {}).get(timeframe)
            if candle and candle.is_complete():
                completed.append(candle)
                self.completed_candles.append(candle)

                # Remove from current
                del self.current_candles[symbol][timeframe]

        return completed

    def get_statistics(self) -> dict:
        """Get aggregator statistics."""
        stats = {
            'timeframes': self.timeframes,
            'symbols_tracked': len(self.current_candles),
            'completed_candles': len(self.completed_candles),
        }

        # Count current candles per timeframe
        for tf in self.timeframes:
            count = sum(
                1 for candles in self.current_candles.values()
                if tf in candles
            )
            stats[f'current_{tf}'] = count

        return stats


def candles_to_dataframe(candles: List[OHLCVCandle]) -> pd.DataFrame:
    """
    Convert list of candles to pandas DataFrame.

    Args:
        candles: List of OHLCVCandle objects

    Returns:
        DataFrame with OHLCV data
    """
    if not candles:
        return pd.DataFrame()

    data = [c.to_dict() for c in candles]
    df = pd.DataFrame(data)

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    # Sort by timestamp
    df.sort_index(inplace=True)

    return df


if __name__ == "__main__":
    # Test OHLCV aggregator
    import numpy as np

    np.random.seed(42)

    # Create aggregator
    aggregator = OHLCVAggregator(timeframes=['1m', '5m', '15m'])

    symbol = "BTC-USD"
    base_price = 45000.0
    base_time = datetime(2024, 1, 1, 10, 0, 0)

    print("=== OHLCV Aggregator Test ===\n")
    print("Simulating 100 trades over 20 minutes...\n")

    # Simulate trades
    all_completed = []

    for i in range(100):
        # Random trade
        price = base_price + np.random.randn() * 50
        size = np.random.exponential(0.1)
        side = np.random.choice(['BUY', 'SELL'])
        timestamp = base_time + timedelta(seconds=i * 12)  # Trade every 12 seconds

        # Add trade
        completed = aggregator.add_trade(symbol, price, size, side, timestamp)

        if completed:
            all_completed.extend(completed)
            for candle in completed:
                print(f"✓ Completed: {candle}")

    print(f"\n=== Statistics ===")
    stats = aggregator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\n=== Current Candles (incomplete) ===")
    for tf in ['1m', '5m', '15m']:
        candle = aggregator.get_current_candle(symbol, tf)
        if candle:
            print(f"  {tf}: {candle}")

    print(f"\n=== Converting to DataFrame ===")
    if all_completed:
        # Separate by timeframe
        for tf in ['1m', '5m', '15m']:
            tf_candles = [c for c in all_completed if c.timeframe == tf]
            if tf_candles:
                df = candles_to_dataframe(tf_candles)
                print(f"\n{tf} candles ({len(df)} total):")
                print(df[['open', 'high', 'low', 'close', 'volume', 'num_trades']].tail())
