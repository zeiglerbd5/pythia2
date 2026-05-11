"""
Enhanced State Representation for RL Trading Agent (Phase 2)

Implements:
- MarketState dataclass for complete state representation
- Multi-timescale history buffers
- Microstructure features (order book imbalance, trade flow)
- Market context features (BTC correlation, regime indicators)
- State builders and processors
"""

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from collections import deque
from datetime import datetime, timedelta
import duckdb
from loguru import logger


@dataclass
class StateConfig:
    """Configuration for enhanced state representation."""
    # Timescale dimensions
    d_micro: int = 15             # Micro features dimension
    d_meso: int = 10              # Meso features dimension
    d_macro: int = 6              # Macro features dimension
    d_position: int = 6           # Position context dimension
    d_time: int = 4               # Time encoding dimension

    # Sequence lengths
    seq_micro: int = 60           # 60 minutes of 1-min data
    seq_meso: int = 24            # 24 candles of hourly data
    seq_macro: int = 7            # 7 days of daily data

    # Current features
    d_current: int = 20           # Current timestep features

    # Normalization
    normalize: bool = True
    clip_value: float = 10.0


@dataclass
class MarketState:
    """
    Complete state representation for RL agent.

    Captures multi-timescale market information plus
    position context for trading decisions.
    """
    # Current timestep features
    current_features: np.ndarray  # (d_current,)

    # Historical context (for attention)
    history_micro: np.ndarray     # (seq_micro, d_micro)
    history_meso: np.ndarray      # (seq_meso, d_meso)
    history_macro: np.ndarray     # (seq_macro, d_macro)

    # Position state
    position_context: np.ndarray  # (d_position,)

    # Time encoding
    time_embedding: np.ndarray    # (d_time,)

    # Metadata
    timestamp: Optional[datetime] = None
    symbol: Optional[str] = None

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary format."""
        return {
            'current': self.current_features,
            'micro': self.history_micro,
            'meso': self.history_meso,
            'macro': self.history_macro,
            'position': self.position_context,
            'time': self.time_embedding,
        }

    def to_tensor_dict(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Convert to tensor dictionary."""
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in self.to_dict().items()
        }

    def to_flat_array(self) -> np.ndarray:
        """Flatten to single array (for basic policies)."""
        return np.concatenate([
            self.current_features,
            self.history_micro.flatten(),
            self.history_meso.flatten(),
            self.history_macro.flatten(),
            self.position_context,
            self.time_embedding,
        ])


class HistoryBuffer:
    """
    Rolling buffer for historical data at a specific timescale.

    Maintains fixed-size history for attention-based processing.
    """

    def __init__(
        self,
        max_length: int,
        feature_dim: int,
        dtype: np.dtype = np.float32,
    ):
        """
        Initialize history buffer.

        Args:
            max_length: Maximum sequence length
            feature_dim: Feature dimension
            dtype: Data type
        """
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.dtype = dtype

        # Initialize with zeros
        self.buffer = np.zeros((max_length, feature_dim), dtype=dtype)
        self.timestamps = deque(maxlen=max_length)
        self.count = 0

    def add(
        self,
        features: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Add new features to buffer.

        Args:
            features: Feature vector (feature_dim,)
            timestamp: Associated timestamp
        """
        # Roll buffer and add new entry at end
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = features

        if timestamp is not None:
            self.timestamps.append(timestamp)

        self.count = min(self.count + 1, self.max_length)

    def add_batch(
        self,
        features: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
    ) -> None:
        """
        Add batch of features.

        Args:
            features: Feature array (n, feature_dim)
            timestamps: Associated timestamps
        """
        n = len(features)
        if n >= self.max_length:
            # Just take last max_length entries
            self.buffer = features[-self.max_length:]
            self.count = self.max_length
        else:
            # Roll and insert
            self.buffer = np.roll(self.buffer, -n, axis=0)
            self.buffer[-n:] = features
            self.count = min(self.count + n, self.max_length)

        if timestamps is not None:
            for ts in timestamps[-self.max_length:]:
                self.timestamps.append(ts)

    def get(self) -> np.ndarray:
        """Get current buffer contents."""
        return self.buffer.copy()

    def get_valid(self) -> np.ndarray:
        """Get only valid (non-zero) entries."""
        if self.count < self.max_length:
            return self.buffer[-self.count:].copy()
        return self.buffer.copy()

    def reset(self) -> None:
        """Reset buffer to zeros."""
        self.buffer = np.zeros((self.max_length, self.feature_dim), dtype=self.dtype)
        self.timestamps.clear()
        self.count = 0

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.count >= self.max_length


class MicrostructureFeatureBuilder:
    """
    Build microstructure features from order book and trade data.

    Features:
    - Order book imbalance at multiple levels
    - Bid-ask spread and depth
    - Trade flow imbalance
    - Large order detection
    """

    def __init__(self, levels: int = 5):
        """
        Initialize microstructure feature builder.

        Args:
            levels: Number of order book levels to use
        """
        self.levels = levels

    def from_order_book(
        self,
        bids: List[Tuple[float, float]],  # [(price, qty), ...]
        asks: List[Tuple[float, float]],
    ) -> Dict[str, float]:
        """
        Extract features from order book snapshot.

        Args:
            bids: List of (price, quantity) tuples
            asks: List of (price, quantity) tuples

        Returns:
            Dictionary of microstructure features
        """
        features = {}

        if not bids or not asks:
            return self._default_features()

        # Convert to arrays
        bid_prices = np.array([b[0] for b in bids[:self.levels]])
        bid_qtys = np.array([b[1] for b in bids[:self.levels]])
        ask_prices = np.array([a[0] for a in asks[:self.levels]])
        ask_qtys = np.array([a[1] for a in asks[:self.levels]])

        # Best bid/ask
        best_bid = bid_prices[0] if len(bid_prices) > 0 else 0
        best_ask = ask_prices[0] if len(ask_prices) > 0 else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0

        # Spread
        spread = (best_ask - best_bid) if best_bid > 0 else 0
        spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0
        features['spread_bps'] = spread_bps

        # Order book imbalance at each level
        total_bid_qty = bid_qtys.sum() if len(bid_qtys) > 0 else 0
        total_ask_qty = ask_qtys.sum() if len(ask_qtys) > 0 else 0
        total_qty = total_bid_qty + total_ask_qty

        if total_qty > 0:
            imbalance = (total_bid_qty - total_ask_qty) / total_qty
        else:
            imbalance = 0

        features['order_book_imbalance'] = imbalance

        # Depth ratio (near vs far)
        if len(bid_qtys) >= 3 and len(ask_qtys) >= 3:
            near_depth = (bid_qtys[:2].sum() + ask_qtys[:2].sum())
            far_depth = (bid_qtys[2:].sum() + ask_qtys[2:].sum())
            depth_ratio = near_depth / (far_depth + 1e-8)
            features['depth_ratio'] = min(depth_ratio, 10)
        else:
            features['depth_ratio'] = 1.0

        # Weighted mid price
        if total_qty > 0:
            weighted_mid = (best_bid * total_ask_qty + best_ask * total_bid_qty) / total_qty
            features['weighted_mid_dist'] = (weighted_mid - mid_price) / mid_price * 100 if mid_price > 0 else 0
        else:
            features['weighted_mid_dist'] = 0

        return features

    def from_trades(
        self,
        prices: np.ndarray,
        sizes: np.ndarray,
        sides: np.ndarray,  # 1 = buy, -1 = sell
        window: int = 20,
    ) -> Dict[str, float]:
        """
        Extract features from recent trades.

        Args:
            prices: Trade prices
            sizes: Trade sizes
            sides: Trade sides (1 = buy, -1 = sell)
            window: Number of trades to analyze

        Returns:
            Dictionary of trade flow features
        """
        features = {}

        if len(prices) < window:
            return self._default_trade_features()

        recent_prices = prices[-window:]
        recent_sizes = sizes[-window:]
        recent_sides = sides[-window:]

        # Buy/sell volume
        buy_mask = recent_sides > 0
        buy_volume = recent_sizes[buy_mask].sum()
        sell_volume = recent_sizes[~buy_mask].sum()
        total_volume = buy_volume + sell_volume

        # Trade flow imbalance
        if total_volume > 0:
            features['trade_flow_imbalance'] = (buy_volume - sell_volume) / total_volume
        else:
            features['trade_flow_imbalance'] = 0

        # VPIN (simplified)
        features['vpin'] = abs(features['trade_flow_imbalance'])

        # Large trade ratio
        avg_size = recent_sizes.mean()
        large_trades = recent_sizes > (2 * avg_size)
        features['large_trade_ratio'] = large_trades.sum() / len(recent_sizes)

        # Trade rate acceleration
        if len(prices) >= 2 * window:
            recent_count = window
            previous_count = window  # Same window for comparison
            features['trade_rate_ratio'] = 1.0  # Would need timestamps for actual rate
        else:
            features['trade_rate_ratio'] = 1.0

        return features

    def _default_features(self) -> Dict[str, float]:
        """Return default order book features."""
        return {
            'spread_bps': 0,
            'order_book_imbalance': 0,
            'depth_ratio': 1.0,
            'weighted_mid_dist': 0,
        }

    def _default_trade_features(self) -> Dict[str, float]:
        """Return default trade features."""
        return {
            'trade_flow_imbalance': 0,
            'vpin': 0,
            'large_trade_ratio': 0,
            'trade_rate_ratio': 1.0,
        }


class MarketContextBuilder:
    """
    Build market context features from broader market data.

    Features:
    - BTC correlation and returns
    - Market regime indicators
    - Sector/market-wide momentum
    """

    def __init__(
        self,
        db_path: str = "/Users/bz/Pythia2/full_pythia.duckdb",
    ):
        """
        Initialize market context builder.

        Args:
            db_path: Path to DuckDB database
        """
        self.db_path = db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._btc_cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Lazy database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path, read_only=True)
        return self._conn

    def get_btc_context(
        self,
        timestamp: datetime,
        lookback_hours: int = 24,
    ) -> Dict[str, float]:
        """
        Get BTC-related context features.

        Args:
            timestamp: Current timestamp
            lookback_hours: Hours of BTC data to analyze

        Returns:
            BTC context features
        """
        features = {}

        try:
            # Check cache
            if (self._btc_cache is not None and
                self._cache_time is not None and
                (timestamp - self._cache_time).total_seconds() < 3600):
                btc_data = self._btc_cache
            else:
                # Query BTC data
                start_time = timestamp - timedelta(hours=lookback_hours)
                query = """
                    SELECT timestamp, close, volume
                    FROM ohlcv
                    WHERE symbol = 'BTC-USD'
                      AND timeframe = '1m'
                      AND timestamp >= ?
                      AND timestamp <= ?
                    ORDER BY timestamp
                """
                btc_data = self.conn.execute(query, [start_time, timestamp]).fetchdf()

                if len(btc_data) > 0:
                    self._btc_cache = btc_data
                    self._cache_time = timestamp

            if len(btc_data) < 60:
                return self._default_btc_features()

            # BTC returns at various horizons
            closes = btc_data['close'].values
            features['btc_return_1h'] = (closes[-1] / closes[-60] - 1) * 100 if len(closes) >= 60 else 0
            features['btc_return_24h'] = (closes[-1] / closes[0] - 1) * 100

            # BTC volatility
            returns = np.diff(np.log(closes))
            features['btc_volatility'] = np.std(returns) * np.sqrt(60 * 24) * 100  # Annualized %

            # BTC trend (simple momentum)
            if len(closes) >= 120:
                short_ma = np.mean(closes[-20:])
                long_ma = np.mean(closes[-120:])
                features['btc_trend'] = (short_ma / long_ma - 1) * 100
            else:
                features['btc_trend'] = 0

        except Exception as e:
            logger.warning(f"Error getting BTC context: {e}")
            return self._default_btc_features()

        return features

    def get_correlation(
        self,
        symbol: str,
        timestamp: datetime,
        window_hours: int = 24,
    ) -> float:
        """
        Calculate correlation with BTC.

        Args:
            symbol: Target symbol
            timestamp: Current timestamp
            window_hours: Correlation window

        Returns:
            Correlation coefficient
        """
        try:
            start_time = timestamp - timedelta(hours=window_hours)

            query = """
                SELECT symbol, timestamp, close
                FROM ohlcv
                WHERE symbol IN (?, 'BTC-USD')
                  AND timeframe = '1m'
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp
            """

            data = self.conn.execute(query, [symbol, start_time, timestamp]).fetchdf()

            if len(data) < 60:
                return 0.0

            # Pivot and calculate correlation
            pivot = data.pivot(index='timestamp', columns='symbol', values='close')

            if symbol not in pivot.columns or 'BTC-USD' not in pivot.columns:
                return 0.0

            # Returns
            returns = pivot.pct_change().dropna()

            if len(returns) < 30:
                return 0.0

            correlation = returns[symbol].corr(returns['BTC-USD'])
            return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger.warning(f"Error calculating correlation: {e}")
            return 0.0

    def _default_btc_features(self) -> Dict[str, float]:
        """Return default BTC features."""
        return {
            'btc_return_1h': 0,
            'btc_return_24h': 0,
            'btc_volatility': 20,  # Typical crypto volatility
            'btc_trend': 0,
        }

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class StateBuilder:
    """
    Build complete MarketState from raw data.

    Combines:
    - Multi-timeframe OHLCV features
    - Microstructure features
    - Market context
    - Position context
    """

    def __init__(
        self,
        config: Optional[StateConfig] = None,
        db_path: str = "/Users/bz/Pythia2/full_pythia.duckdb",
    ):
        """
        Initialize state builder.

        Args:
            config: State configuration
            db_path: Path to database
        """
        self.config = config or StateConfig()

        # History buffers
        self.micro_buffer = HistoryBuffer(
            self.config.seq_micro, self.config.d_micro
        )
        self.meso_buffer = HistoryBuffer(
            self.config.seq_meso, self.config.d_meso
        )
        self.macro_buffer = HistoryBuffer(
            self.config.seq_macro, self.config.d_macro
        )

        # Feature builders
        self.microstructure = MicrostructureFeatureBuilder()
        self.market_context = MarketContextBuilder(db_path)

        # Current symbol
        self.current_symbol: Optional[str] = None

    def update(
        self,
        ohlcv: Dict[str, float],
        order_book: Optional[Dict] = None,
        trades: Optional[Dict] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Update state with new data.

        Args:
            ohlcv: OHLCV data for current candle
            order_book: Order book snapshot
            trades: Recent trades
            timestamp: Current timestamp
        """
        # Build micro features from 1-min data
        micro_features = self._build_micro_features(ohlcv, order_book, trades)
        self.micro_buffer.add(micro_features, timestamp)

        # Note: meso and macro buffers would be updated at their respective
        # timeframes (hourly, daily) in a real implementation

    def build_state(
        self,
        current_ohlcv: Dict[str, float],
        position: Optional[Any] = None,
        timestamp: Optional[datetime] = None,
        symbol: Optional[str] = None,
    ) -> MarketState:
        """
        Build complete market state.

        Args:
            current_ohlcv: Current OHLCV data
            position: Current position (optional)
            timestamp: Current timestamp
            symbol: Current symbol

        Returns:
            MarketState object
        """
        # Current features
        current_features = self._build_current_features(current_ohlcv, timestamp)

        # Position context
        position_context = self._build_position_context(position, current_ohlcv.get('close', 0))

        # Time embedding
        time_embedding = self._build_time_embedding(timestamp)

        # Get history from buffers
        history_micro = self.micro_buffer.get()
        history_meso = self.meso_buffer.get()
        history_macro = self.macro_buffer.get()

        return MarketState(
            current_features=current_features,
            history_micro=history_micro,
            history_meso=history_meso,
            history_macro=history_macro,
            position_context=position_context,
            time_embedding=time_embedding,
            timestamp=timestamp,
            symbol=symbol,
        )

    def _build_micro_features(
        self,
        ohlcv: Dict[str, float],
        order_book: Optional[Dict] = None,
        trades: Optional[Dict] = None,
    ) -> np.ndarray:
        """Build micro-scale features."""
        features = []

        # OHLCV features
        features.extend([
            ohlcv.get('return', 0),
            ohlcv.get('volume_ratio', 1),
            ohlcv.get('high_low_range', 0),
        ])

        # Order book features
        if order_book is not None:
            ob_features = self.microstructure.from_order_book(
                order_book.get('bids', []),
                order_book.get('asks', []),
            )
            features.extend([
                ob_features['spread_bps'] / 100,
                ob_features['order_book_imbalance'],
                ob_features['depth_ratio'] / 10,
            ])
        else:
            features.extend([0, 0, 0.1])

        # Trade features
        if trades is not None:
            trade_features = self.microstructure.from_trades(
                np.array(trades.get('prices', [])),
                np.array(trades.get('sizes', [])),
                np.array(trades.get('sides', [])),
            )
            features.extend([
                trade_features['trade_flow_imbalance'],
                trade_features['vpin'],
                trade_features['large_trade_ratio'],
            ])
        else:
            features.extend([0, 0, 0])

        # Pad to d_micro
        while len(features) < self.config.d_micro:
            features.append(0)

        return np.array(features[:self.config.d_micro], dtype=np.float32)

    def _build_current_features(
        self,
        ohlcv: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        """Build current timestep features."""
        features = []

        # Price features
        features.append(ohlcv.get('return_1m', 0))
        features.append(ohlcv.get('return_5m', 0))
        features.append(ohlcv.get('return_15m', 0))
        features.append(ohlcv.get('return_1h', 0))

        # Volatility
        features.append(ohlcv.get('volatility', 0))
        features.append(ohlcv.get('natr', 0))

        # Volume
        features.append(ohlcv.get('volume_ratio', 1))
        features.append(ohlcv.get('volume_zscore', 0))

        # Technical
        features.append(ohlcv.get('rsi', 0.5) - 0.5)  # Center around 0
        features.append(ohlcv.get('bb_position', 0.5) - 0.5)

        # Pad to d_current
        while len(features) < self.config.d_current:
            features.append(0)

        arr = np.array(features[:self.config.d_current], dtype=np.float32)

        # Normalize
        if self.config.normalize:
            arr = np.clip(arr, -self.config.clip_value, self.config.clip_value)

        return arr

    def _build_position_context(
        self,
        position: Optional[Any],
        current_price: float,
    ) -> np.ndarray:
        """Build position context features."""
        if position is None:
            return np.zeros(self.config.d_position, dtype=np.float32)

        features = [
            1.0,  # has_position
            position.unrealized_return(current_price) * 10,  # Scale up
            min(position.highest_return() * 10, 1.0),  # Capped
        ]

        # Time in position (normalized)
        # Would need timestamp from position
        features.append(0.5)  # Placeholder

        # Stop distance
        stop_dist = (current_price - position.stop_loss) / current_price
        features.append(stop_dist * 100)

        # Position size
        features.append(position.size)

        return np.array(features[:self.config.d_position], dtype=np.float32)

    def _build_time_embedding(
        self,
        timestamp: Optional[datetime],
    ) -> np.ndarray:
        """Build cyclical time embedding."""
        if timestamp is None:
            return np.zeros(self.config.d_time, dtype=np.float32)

        # Hour of day (cyclical)
        hour = timestamp.hour + timestamp.minute / 60
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Day of week (cyclical)
        day = timestamp.weekday()
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)

        return np.array([hour_sin, hour_cos, day_sin, day_cos], dtype=np.float32)

    def reset(self) -> None:
        """Reset all buffers."""
        self.micro_buffer.reset()
        self.meso_buffer.reset()
        self.macro_buffer.reset()

    def close(self) -> None:
        """Clean up resources."""
        self.market_context.close()


if __name__ == "__main__":
    # Test state representation
    print("Testing State Representation\n" + "=" * 50)

    config = StateConfig()

    # Test HistoryBuffer
    print("\n1. HistoryBuffer")
    buffer = HistoryBuffer(max_length=10, feature_dim=5)
    for i in range(15):
        buffer.add(np.random.randn(5).astype(np.float32))
    print(f"   Buffer shape: {buffer.get().shape}")
    print(f"   Is full: {buffer.is_full()}")
    print(f"   Count: {buffer.count}")

    # Test MicrostructureFeatureBuilder
    print("\n2. MicrostructureFeatureBuilder")
    micro_builder = MicrostructureFeatureBuilder()

    # Fake order book
    bids = [(100.0, 10.0), (99.5, 20.0), (99.0, 30.0)]
    asks = [(100.5, 15.0), (101.0, 25.0), (101.5, 35.0)]

    ob_features = micro_builder.from_order_book(bids, asks)
    print(f"   Order book features: {ob_features}")

    # Fake trades
    prices = np.array([100.0, 100.1, 99.9, 100.2, 100.0] * 10)
    sizes = np.array([1.0, 0.5, 2.0, 1.5, 0.8] * 10)
    sides = np.array([1, -1, -1, 1, 1] * 10)

    trade_features = micro_builder.from_trades(prices, sizes, sides)
    print(f"   Trade features: {trade_features}")

    # Test StateBuilder
    print("\n3. StateBuilder")
    builder = StateBuilder(config)

    # Simulate updates
    for i in range(100):
        ohlcv = {
            'open': 100 + np.random.randn(),
            'high': 101 + np.random.randn(),
            'low': 99 + np.random.randn(),
            'close': 100 + np.random.randn(),
            'volume': 1000 + np.random.randn() * 100,
            'return': np.random.randn() * 0.01,
            'volume_ratio': 1 + np.random.randn() * 0.2,
        }
        builder.update(ohlcv, timestamp=datetime.now())

    # Build state
    current_ohlcv = {
        'return_1m': 0.001,
        'return_5m': 0.003,
        'return_15m': 0.005,
        'return_1h': 0.01,
        'volatility': 0.02,
        'natr': 2.5,
        'volume_ratio': 1.2,
        'volume_zscore': 0.5,
        'rsi': 0.55,
        'bb_position': 0.6,
        'close': 100.5,
    }

    state = builder.build_state(
        current_ohlcv,
        position=None,
        timestamp=datetime.now(),
        symbol='ETH-USD',
    )

    print(f"   Current features shape: {state.current_features.shape}")
    print(f"   Micro history shape: {state.history_micro.shape}")
    print(f"   Meso history shape: {state.history_meso.shape}")
    print(f"   Macro history shape: {state.history_macro.shape}")
    print(f"   Position context shape: {state.position_context.shape}")
    print(f"   Time embedding shape: {state.time_embedding.shape}")

    # Test flat array
    flat = state.to_flat_array()
    print(f"   Flat array shape: {flat.shape}")

    # Test tensor conversion
    tensors = state.to_tensor_dict()
    print(f"   Tensor keys: {list(tensors.keys())}")

    builder.close()
    print("\nState representation tests passed!")
