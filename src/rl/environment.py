"""
Gym-Compatible Trading Environment for RL Agent

Implements a trading environment that:
- Loads data from DuckDB
- Provides OHLCV features + position context as state
- Supports 7 discrete actions with action masking
- Simulates execution with realistic fees
- Structures episodes as 24 hours of 1-minute decisions
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import duckdb
import sqlite3
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Union
from datetime import datetime, timedelta
from enum import IntEnum
from loguru import logger


class Action(IntEnum):
    """Discrete action space for trading."""
    WAIT = 0           # Do nothing, continue observing
    ENTER_LONG = 1     # Open long position (fixed size)
    HOLD = 2           # Maintain current position
    TIGHTEN_STOP = 3   # Move stop loss closer (reduce risk)
    LOOSEN_STOP = 4    # Move stop loss further (give room)
    TAKE_PARTIAL = 5   # Exit 50% of position
    EXIT_ALL = 6       # Close entire position


@dataclass
class Position:
    """Represents an open trading position."""
    entry_price: float
    entry_time: datetime
    size: float  # 1.0 = full position, 0.5 = half position
    stop_loss: float
    highest_price: float  # For tracking max unrealized gain

    def unrealized_return(self, current_price: float) -> float:
        """Calculate unrealized return percentage."""
        return (current_price - self.entry_price) / self.entry_price

    def highest_return(self) -> float:
        """Calculate highest unrealized return seen."""
        return (self.highest_price - self.entry_price) / self.entry_price

    def update_high(self, current_price: float) -> None:
        """Update highest price seen."""
        if current_price > self.highest_price:
            self.highest_price = current_price


@dataclass
class TradeResult:
    """Result of a completed trade."""
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size: float
    return_pct: float
    exit_reason: str  # 'manual', 'stop_loss', 'end_of_episode'
    fees_paid: float
    highest_return: float = 0.0  # v5: Highest unrealized return during position


@dataclass
class EpisodeConfig:
    """Configuration for training episodes."""
    # Episode duration
    episode_length: int = 1440        # 1440 minutes = 24 hours
    step_size_minutes: int = 1        # Decision frequency

    # Data sampling
    symbols: Optional[List[str]] = None  # None = sample from available
    random_start: bool = True         # Random start time in data

    # Position constraints
    max_position_duration: int = 480  # 8 hours max hold
    max_trades_per_episode: int = 10  # Prevent overtrading
    initial_capital: float = 10000.0  # Starting capital

    # Stop loss parameters
    initial_stop_pct: float = 0.02    # 2% initial stop loss
    min_stop_pct: float = 0.005       # 0.5% minimum stop
    max_stop_pct: float = 0.05        # 5% maximum stop
    stop_adjustment_pct: float = 0.005  # 0.5% per adjustment

    # Transaction costs
    fee_rate: float = 0.0055          # 0.55% total fees (maker+taker)
    slippage_pct: float = 0.001       # 0.1% slippage estimate

    # Termination conditions
    terminate_on_drawdown: float = 0.15  # -15% episode loss

    # Lookback for features
    lookback_minutes: int = 60        # Historical context needed

    # =========================================
    # ENHANCED SAMPLING CONFIGURATION
    # =========================================

    # Sampling mode: how to select episode start times
    # - "random": Random non-overlapping windows (original behavior)
    # - "sequential": Sequential episodes with overlap
    # - "event_anchored": Bias toward interesting market events
    sampling_mode: str = "random"

    # Overlap percentage for sequential mode
    # 0.5 = 50% overlap, meaning each episode shares half its data with neighbors
    window_overlap_pct: float = 0.5

    # Extended context lookback (even for shorter episodes)
    # This provides 24-48hr of historical context in features
    # regardless of episode_length
    context_lookback_minutes: int = 1440  # 24hr context

    # Event-anchored sampling parameters
    # When sampling_mode="event_anchored", bias toward these conditions
    event_volume_threshold: float = 3.0    # Volume spike multiplier (3x avg)
    event_volatility_threshold: float = 2.0  # Volatility spike multiplier (2x avg)
    event_bias_probability: float = 0.7    # Probability of sampling near events

    # Sequential episode tracking (for sequential mode)
    _sequential_index: int = 0  # Internal: current position in sequence


class TradingEnvironment(gym.Env):
    """
    Gym-compatible trading environment for cryptocurrency.

    State Space:
        - OHLCV features (normalized)
        - Technical indicators
        - Position context (has_position, unrealized_pnl, time_in_position, etc.)

    Action Space:
        7 discrete actions with masking based on position state

    Reward:
        Configurable - P&L based with risk adjustments (see rewards.py)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        db_path: str = "/Users/bz/Pythia2/full_pythia.duckdb",
        config: Optional[EpisodeConfig] = None,
        reward_calculator: Optional[Any] = None,  # RewardCalculator instance
        feature_extractor: Optional[Any] = None,  # FeatureExtractor instance
        render_mode: Optional[str] = None,
    ):
        """
        Initialize trading environment.

        Args:
            db_path: Path to DuckDB database
            config: Episode configuration
            reward_calculator: Optional reward calculator (uses default if None)
            feature_extractor: Optional feature extractor (uses default if None)
            render_mode: Render mode for visualization
        """
        super().__init__()

        self.db_path = db_path
        self.config = config or EpisodeConfig()
        self.render_mode = render_mode

        # Lazy imports to avoid circular dependencies
        if reward_calculator is None:
            from .rewards import RewardCalculator, RewardConfig
            self.reward_calculator = RewardCalculator(RewardConfig())
        else:
            self.reward_calculator = reward_calculator

        if feature_extractor is None:
            from .features import FeatureExtractor, FeatureConfig
            self.feature_extractor = FeatureExtractor(FeatureConfig())
        else:
            self.feature_extractor = feature_extractor

        # State dimension from feature extractor + position features
        # Position features: 6 base + 5 spike tracking = 11 total
        self._position_feature_dim = 11
        self.state_dim = self.feature_extractor.get_state_dim() + self._position_feature_dim

        # Define spaces
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Database connection (lazy initialization)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._available_symbols: Optional[List[str]] = None
        self._available_time_ranges: Optional[Dict[str, Tuple[datetime, datetime]]] = None

        # Episode state
        self.symbol: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.current_step: int = 0
        self.current_time: Optional[datetime] = None

        # Market data for episode
        self.ohlcv_data: Optional[pd.DataFrame] = None
        self.features_data: Optional[pd.DataFrame] = None

        # Position and trading state
        self.position: Optional[Position] = None
        self.episode_trades: List[TradeResult] = []
        self.episode_returns: List[float] = []
        self.capital: float = self.config.initial_capital

        # Previous state for reward calculation
        self._prev_state: Optional[np.ndarray] = None
        self._prev_price: Optional[float] = None

        # =========================================
        # SPIKE TRACKING STATE (v2)
        # =========================================
        # Tracks spikes to prevent repeated trading on same event
        self.last_trade_time: Optional[datetime] = None
        self.spike_start_time: Optional[datetime] = None
        self.spike_traded: bool = False
        self.trades_this_hour: int = 0
        self._trades_timestamps: List[datetime] = []  # Track recent trade times

        # Spike detection thresholds (rolling averages)
        self._volume_ma: Optional[float] = None
        self._volatility_ma: Optional[float] = None

        logger.info(f"TradingEnvironment initialized with state_dim={self.state_dim}")

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return self.db_path.endswith('.db') and not self.db_path.endswith('.duckdb')

    @property
    def conn(self) -> Union[duckdb.DuckDBPyConnection, sqlite3.Connection]:
        """Lazy database connection."""
        if self._conn is None:
            if self.is_sqlite:
                self._conn = sqlite3.connect(self.db_path)
                self._conn.row_factory = sqlite3.Row
            else:
                self._conn = duckdb.connect(self.db_path, read_only=True)
        return self._conn

    def _execute_query(self, query: str, params: list = None) -> pd.DataFrame:
        """Execute query and return DataFrame, handling both SQLite and DuckDB."""
        if self.is_sqlite:
            return pd.read_sql_query(query, self.conn, params=params)
        else:
            if params:
                return self.conn.execute(query, params).fetchdf()
            return self.conn.execute(query).fetchdf()

    def _execute_one(self, query: str, params: list = None) -> tuple:
        """Execute query and return single row."""
        if self.is_sqlite:
            cursor = self.conn.execute(query, params or [])
            return cursor.fetchone()
        else:
            return self.conn.execute(query, params or []).fetchone()

    def _get_available_symbols(self) -> List[str]:
        """Get list of available symbols with sufficient data."""
        if self._available_symbols is None:
            # Need enough data for at least one full episode plus lookback
            min_records = self.config.episode_length + self.config.lookback_minutes + 100

            # SQLite feature_buffer.db doesn't have timeframe column
            if self.is_sqlite:
                query = f"""
                    SELECT symbol, COUNT(*) as cnt
                    FROM ohlcv
                    GROUP BY symbol
                    HAVING cnt > {min_records}
                    ORDER BY cnt DESC
                """
            else:
                query = f"""
                    SELECT symbol, COUNT(*) as cnt
                    FROM ohlcv
                    WHERE timeframe = '1m'
                    GROUP BY symbol
                    HAVING cnt > {min_records}
                    ORDER BY cnt DESC
                """
            result = self._execute_query(query)
            self._available_symbols = result['symbol'].tolist()
            logger.info(f"Found {len(self._available_symbols)} symbols with sufficient data (>{min_records} records)")
        return self._available_symbols

    def _get_time_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available time range for a symbol."""
        if self._available_time_ranges is None:
            self._available_time_ranges = {}

        if symbol not in self._available_time_ranges:
            if self.is_sqlite:
                query = """
                    SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
                    FROM ohlcv
                    WHERE symbol = ?
                """
            else:
                query = """
                    SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
                    FROM ohlcv
                    WHERE symbol = ? AND timeframe = '1m'
                """
            result = self._execute_one(query, [symbol])
            self._available_time_ranges[symbol] = (
                pd.to_datetime(result[0]),
                pd.to_datetime(result[1])
            )
        return self._available_time_ranges[symbol]

    def _sample_symbol(self) -> str:
        """Sample a random symbol from available symbols."""
        symbols = self.config.symbols or self._get_available_symbols()
        return np.random.choice(symbols)

    def _check_data_availability(self, symbol: str, start_time: datetime) -> int:
        """
        Check how many records exist for a given symbol and time range.

        Returns the number of records found.
        """
        data_start = start_time - timedelta(minutes=self.config.lookback_minutes)
        data_end = start_time + timedelta(minutes=self.config.episode_length)

        if self.is_sqlite:
            query = """
                SELECT COUNT(*) as cnt
                FROM ohlcv
                WHERE symbol = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
            """
            result = self._execute_one(query, [symbol, data_start.isoformat(), data_end.isoformat()])
        else:
            query = """
                SELECT COUNT(*) as cnt
                FROM ohlcv
                WHERE symbol = ?
                  AND timeframe = '1m'
                  AND timestamp >= ?
                  AND timestamp <= ?
            """
            result = self._execute_one(query, [symbol, data_start, data_end])

        return result[0] if result else 0

    def _sample_start_time(self, symbol: str, max_attempts: int = 5) -> datetime:
        """
        Sample start time based on configured sampling mode.

        Modes:
        - random: Random non-overlapping windows
        - sequential: Sequential with overlap
        - event_anchored: Biased toward interesting market events
        """
        if self.config.sampling_mode == "sequential":
            return self._sample_sequential_start_time(symbol)
        elif self.config.sampling_mode == "event_anchored":
            return self._sample_event_anchored_start_time(symbol, max_attempts)
        else:
            return self._sample_random_start_time(symbol, max_attempts)

    def _sample_random_start_time(self, symbol: str, max_attempts: int = 5) -> datetime:
        """Sample a random start time within available data, verifying data exists."""
        min_ts, max_ts = self._get_time_range(symbol)

        # Use extended context lookback for feature calculation
        effective_lookback = max(
            self.config.lookback_minutes,
            self.config.context_lookback_minutes
        )

        # Need enough room for lookback and episode
        required_minutes = effective_lookback + self.config.episode_length
        latest_start = max_ts - timedelta(minutes=required_minutes)
        earliest_start = min_ts + timedelta(minutes=effective_lookback)

        if earliest_start >= latest_start:
            return earliest_start

        # Minimum acceptable data density (at least 50% of expected minutes)
        min_records = int((effective_lookback + self.config.episode_length) * 0.5)

        # Try to find a time range with sufficient data
        time_range = (latest_start - earliest_start).total_seconds()

        for attempt in range(max_attempts):
            random_offset = np.random.uniform(0, time_range)
            candidate_time = earliest_start + timedelta(seconds=random_offset)

            # Check if sufficient data exists for this time range
            record_count = self._check_data_availability(symbol, candidate_time)

            if record_count >= min_records:
                logger.debug(
                    f"Found valid start time for {symbol} on attempt {attempt + 1}: "
                    f"{candidate_time} with {record_count} records"
                )
                return candidate_time
            else:
                logger.debug(
                    f"Insufficient data for {symbol} at {candidate_time}: "
                    f"{record_count} < {min_records} records"
                )

        # Return last candidate anyway - reset will handle the error
        return candidate_time

    def _sample_sequential_start_time(self, symbol: str) -> datetime:
        """
        Sample start time for sequential episodes with overlap.

        Episodes progress through time with configurable overlap,
        allowing the agent to see continuity between episodes.
        """
        min_ts, max_ts = self._get_time_range(symbol)

        effective_lookback = max(
            self.config.lookback_minutes,
            self.config.context_lookback_minutes
        )

        # Calculate step size based on overlap
        # With 50% overlap, each episode starts at half the episode length after previous
        step_size = int(self.config.episode_length * (1 - self.config.window_overlap_pct))
        step_size = max(step_size, 1)  # Ensure at least 1 minute step

        # Calculate total number of possible episodes
        available_range = (max_ts - min_ts).total_seconds() / 60
        total_episodes = max(1, int((available_range - effective_lookback - self.config.episode_length) / step_size))

        # Get current position in sequence
        seq_idx = self.config._sequential_index % total_episodes

        # Calculate start time
        start_offset = effective_lookback + (seq_idx * step_size)
        start_time = min_ts + timedelta(minutes=start_offset)

        # Increment sequence index for next call
        self.config._sequential_index += 1

        logger.debug(
            f"Sequential episode {seq_idx}/{total_episodes} for {symbol}: {start_time}"
        )

        return start_time

    def _sample_event_anchored_start_time(self, symbol: str, max_attempts: int = 10) -> datetime:
        """
        Sample start time biased toward interesting market events.

        Events include:
        - High volume periods (volume spikes)
        - High volatility periods
        - Large price movements

        With configured probability, sample near an event rather than randomly.
        """
        # Decide whether to sample near an event or randomly
        if np.random.random() > self.config.event_bias_probability:
            return self._sample_random_start_time(symbol, max_attempts)

        # Try to find interesting events in the data
        events = self._find_market_events(symbol)

        if events is None or len(events) == 0:
            logger.debug(f"No events found for {symbol}, falling back to random sampling")
            return self._sample_random_start_time(symbol, max_attempts)

        # Sample from events
        effective_lookback = max(
            self.config.lookback_minutes,
            self.config.context_lookback_minutes
        )

        min_ts, max_ts = self._get_time_range(symbol)
        earliest_start = min_ts + timedelta(minutes=effective_lookback)
        latest_start = max_ts - timedelta(minutes=self.config.episode_length)

        # Filter events to valid range
        valid_events = [e for e in events if earliest_start <= e <= latest_start]

        if len(valid_events) == 0:
            return self._sample_random_start_time(symbol, max_attempts)

        # Choose a random event and start episode slightly before it
        event_time = np.random.choice(valid_events)

        # Start 30-60 minutes before the event so agent can learn to anticipate
        lead_time = np.random.randint(30, 61)
        start_time = event_time - timedelta(minutes=lead_time)

        # Ensure within bounds
        start_time = max(earliest_start, min(start_time, latest_start))

        logger.debug(
            f"Event-anchored sampling for {symbol}: event at {event_time}, "
            f"episode starts at {start_time}"
        )

        return start_time

    def _find_market_events(self, symbol: str) -> Optional[List[datetime]]:
        """
        Find interesting market events for a symbol.

        Returns list of timestamps where significant events occurred.
        """
        try:
            # Query for volume and price data to find events
            if self.is_sqlite:
                query = """
                    SELECT timestamp, volume, high, low, close
                    FROM ohlcv
                    WHERE symbol = ?
                    ORDER BY timestamp
                """
                df = pd.read_sql_query(query, self.conn, params=[symbol])
            else:
                query = """
                    SELECT timestamp, volume, high, low, close
                    FROM ohlcv
                    WHERE symbol = ? AND timeframe = '1m'
                    ORDER BY timestamp
                """
                df = self.conn.execute(query, [symbol]).fetchdf()

            if len(df) < 100:
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Calculate rolling averages
            df['volume_ma'] = df['volume'].rolling(60).mean()
            df['range'] = (df['high'] - df['low']) / df['close']
            df['range_ma'] = df['range'].rolling(60).mean()

            # Find volume spikes
            volume_threshold = self.config.event_volume_threshold
            volume_events = df[df['volume'] > df['volume_ma'] * volume_threshold]['timestamp'].tolist()

            # Find volatility spikes
            vol_threshold = self.config.event_volatility_threshold
            volatility_events = df[df['range'] > df['range_ma'] * vol_threshold]['timestamp'].tolist()

            # Combine and deduplicate events (within 30-minute windows)
            all_events = sorted(set(volume_events + volatility_events))

            # Cluster events within 30 minutes
            if len(all_events) == 0:
                return None

            clustered = [all_events[0]]
            for event in all_events[1:]:
                if (event - clustered[-1]).total_seconds() > 1800:  # 30 minutes
                    clustered.append(event)

            logger.debug(
                f"Found {len(clustered)} event clusters for {symbol} "
                f"({len(volume_events)} volume, {len(volatility_events)} volatility)"
            )

            return clustered

        except Exception as e:
            logger.warning(f"Error finding events for {symbol}: {e}")
            return None

    def _load_episode_data(self) -> None:
        """Load OHLCV, order book, and feature data for current episode."""
        # Use extended context lookback for richer historical features
        effective_lookback = max(
            self.config.lookback_minutes,
            self.config.context_lookback_minutes
        )
        data_start = self.start_time - timedelta(minutes=effective_lookback)
        data_end = self.start_time + timedelta(minutes=self.config.episode_length)

        # Load OHLCV data - different query for SQLite vs DuckDB
        if self.is_sqlite:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp
            """
            self.ohlcv_data = pd.read_sql_query(
                query, self.conn,
                params=[self.symbol, data_start.isoformat(), data_end.isoformat()]
            )
        else:
            query = """
                SELECT timestamp, open, high, low, close, volume, num_trades
                FROM ohlcv
                WHERE symbol = ?
                  AND timeframe = '1m'
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp
            """
            self.ohlcv_data = self.conn.execute(
                query,
                [self.symbol, data_start, data_end]
            ).fetchdf()

        if len(self.ohlcv_data) == 0:
            raise ValueError(f"No OHLCV data found for {self.symbol} in range {data_start} to {data_end}")

        # Set timestamp as index
        self.ohlcv_data['timestamp'] = pd.to_datetime(self.ohlcv_data['timestamp'])
        self.ohlcv_data.set_index('timestamp', inplace=True)

        # Load order book features if available (DuckDB only for now)
        order_book_data = None
        if not self.is_sqlite:
            order_book_data = self._load_order_book_features(data_start, data_end)

        # Calculate features with extended context
        self.features_data = self.feature_extractor.calculate_features(
            self.ohlcv_data,
            order_book_data=order_book_data
        )

        logger.debug(
            f"Loaded episode data: {len(self.ohlcv_data)} candles, "
            f"{len(self.features_data)} feature rows, "
            f"context lookback={effective_lookback}min"
        )

    def _load_order_book_features(
        self,
        data_start: datetime,
        data_end: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Load pre-calculated order book features from the features table.

        Returns None if no order book data is available.
        """
        try:
            query = """
                SELECT
                    timestamp,
                    order_book_imbalance_l5,
                    bid_ask_ratio,
                    bid_ask_spread_pct,
                    order_book_depth_ratio,
                    large_order_imbalance,
                    weighted_mid_price,
                    vpin,
                    roll_measure
                FROM features
                WHERE symbol = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp
            """
            df = self.conn.execute(
                query,
                [self.symbol, data_start, data_end]
            ).fetchdf()

            if len(df) == 0:
                logger.debug(f"No order book features found for {self.symbol}")
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            logger.debug(f"Loaded {len(df)} order book feature rows for {self.symbol}")
            return df

        except Exception as e:
            logger.debug(f"Could not load order book features: {e}")
            return None

    def _get_current_price(self) -> float:
        """Get current close price."""
        # Find closest timestamp at or before current_time
        available_times = self.ohlcv_data.index[self.ohlcv_data.index <= self.current_time]
        if len(available_times) == 0:
            return self.ohlcv_data['close'].iloc[0]
        return self.ohlcv_data.loc[available_times[-1], 'close']

    def _get_observation(self) -> np.ndarray:
        """Get current observation (state)."""
        # Get features up to current time
        available_times = self.features_data.index[self.features_data.index <= self.current_time]
        if len(available_times) == 0:
            # Return zeros if no features available yet
            return np.zeros(self.state_dim, dtype=np.float32)

        # Get latest features
        latest_features = self.features_data.loc[available_times[-1]]

        # Add position context
        position_features = self._get_position_features()

        # Combine features
        market_features = latest_features.values.astype(np.float32)
        full_state = np.concatenate([market_features, position_features])

        # Handle NaN values
        full_state = np.nan_to_num(full_state, nan=0.0, posinf=0.0, neginf=0.0)

        return full_state.astype(np.float32)

    def _get_position_features(self) -> np.ndarray:
        """Get position context features including spike tracking."""
        current_price = self._get_current_price()

        # Base position features
        if self.position is None:
            position_features = [
                0.0,  # has_position
                0.0,  # position_return_pct
                0.0,  # time_in_position (normalized)
                0.0,  # highest_return_seen
                0.0,  # current_stop_distance
                0.0,  # position_size
            ]
        else:
            time_in_position = (self.current_time - self.position.entry_time).total_seconds() / 60
            position_features = [
                1.0,  # has_position
                self.position.unrealized_return(current_price),
                min(time_in_position / self.config.max_position_duration, 1.0),  # normalized
                self.position.highest_return(),
                (current_price - self.position.stop_loss) / current_price,  # stop distance
                self.position.size,
            ]

        # =========================================
        # SPIKE TRACKING FEATURES (v2)
        # =========================================
        spike_features = self._get_spike_tracking_features()

        return np.array(position_features + spike_features, dtype=np.float32)

    def _get_spike_tracking_features(self) -> List[float]:
        """
        Get spike tracking features to prevent repeated trading on same event.

        Returns:
            List of spike tracking features:
            - minutes_since_last_trade (normalized 0-1, 1 = 60+ min ago)
            - trades_in_last_hour (normalized 0-1, 1 = 5+ trades)
            - spike_freshness (1.0 = new spike, decays to 0 over 60 min)
            - already_traded_this_spike (binary)
            - is_in_spike (binary - currently elevated volume/volatility)
        """
        # Minutes since last trade (normalized: 0 = just traded, 1 = 60+ min ago)
        if self.last_trade_time is None:
            minutes_since_trade = 1.0  # No trades yet, fully "cooled down"
        else:
            mins = (self.current_time - self.last_trade_time).total_seconds() / 60
            minutes_since_trade = min(mins / 60.0, 1.0)  # Normalize to 60 min

        # Trades in last hour (normalized: 0 = none, 1 = 5+ trades)
        self._update_trades_this_hour()
        trades_normalized = min(self.trades_this_hour / 5.0, 1.0)

        # Detect current spike and compute freshness
        is_spike, spike_freshness = self._detect_current_spike()

        # Already traded this spike (binary)
        already_traded = 1.0 if self.spike_traded else 0.0

        return [
            minutes_since_trade,
            trades_normalized,
            spike_freshness,
            already_traded,
            1.0 if is_spike else 0.0,
        ]

    def _update_trades_this_hour(self) -> None:
        """Update count of trades in the last hour."""
        if self.current_time is None:
            self.trades_this_hour = 0
            return

        one_hour_ago = self.current_time - timedelta(hours=1)
        self._trades_timestamps = [
            t for t in self._trades_timestamps if t > one_hour_ago
        ]
        self.trades_this_hour = len(self._trades_timestamps)

    def _detect_current_spike(self) -> Tuple[bool, float]:
        """
        Detect if we're currently in a spike and compute its freshness.

        Returns:
            (is_spike, freshness) where freshness decays from 1.0 to 0.0 over 60 min
        """
        if self.ohlcv_data is None or self.current_time is None:
            return False, 0.0

        # Get recent data for spike detection
        lookback = 60  # 60 minute rolling window for averages
        current_idx = self.ohlcv_data.index.get_indexer([self.current_time], method='ffill')[0]

        if current_idx < lookback:
            return False, 0.0

        recent_data = self.ohlcv_data.iloc[max(0, current_idx - lookback):current_idx + 1]

        if len(recent_data) < 10:
            return False, 0.0

        # Calculate current vs average volume and volatility
        current_volume = recent_data['volume'].iloc[-1]
        avg_volume = recent_data['volume'].iloc[:-1].mean()

        current_range = (recent_data['high'].iloc[-1] - recent_data['low'].iloc[-1]) / recent_data['close'].iloc[-1]
        avg_range = ((recent_data['high'].iloc[:-1] - recent_data['low'].iloc[:-1]) / recent_data['close'].iloc[:-1]).mean()

        # Detect spike
        volume_spike = current_volume > avg_volume * self.config.event_volume_threshold
        volatility_spike = current_range > avg_range * self.config.event_volatility_threshold

        is_spike = volume_spike or volatility_spike

        # Track spike start time
        if is_spike:
            if self.spike_start_time is None:
                # New spike starting
                self.spike_start_time = self.current_time
                self.spike_traded = False  # Reset traded flag for new spike

            # Calculate freshness (1.0 at start, decays to 0 over 60 minutes)
            spike_age_minutes = (self.current_time - self.spike_start_time).total_seconds() / 60
            freshness = max(0.0, 1.0 - (spike_age_minutes / 60.0))
        else:
            # No spike - reset tracking
            if self.spike_start_time is not None:
                # Spike ended, keep spike_traded flag until next spike
                pass
            self.spike_start_time = None
            freshness = 0.0

        return is_spike, freshness

    def _record_trade(self) -> None:
        """Record that a trade was made (for spike tracking)."""
        if self.current_time is not None:
            self.last_trade_time = self.current_time
            self._trades_timestamps.append(self.current_time)

            # Mark current spike as traded
            if self.spike_start_time is not None:
                self.spike_traded = True

    def _get_spike_context(self) -> Dict[str, float]:
        """
        Get spike tracking context for reward calculation.

        Returns:
            Dict with spike tracking features matching SpikeAwareRewardCalculator expectations
        """
        spike_features = self._get_spike_tracking_features()

        return {
            'minutes_since_last_trade': spike_features[0],
            'trades_in_last_hour': spike_features[1],
            'spike_freshness': spike_features[2],
            'already_traded_this_spike': spike_features[3],
            'is_in_spike': spike_features[4],
        }

    def get_action_mask(self) -> np.ndarray:
        """
        Get valid action mask based on current state.

        Returns:
            Boolean array where True = action is valid
        """
        mask = np.ones(len(Action), dtype=bool)

        if self.position is None:
            # No position - can only WAIT or ENTER_LONG
            mask[Action.HOLD] = False
            mask[Action.TIGHTEN_STOP] = False
            mask[Action.LOOSEN_STOP] = False
            mask[Action.TAKE_PARTIAL] = False
            mask[Action.EXIT_ALL] = False
        else:
            # Have position - can't enter again
            mask[Action.ENTER_LONG] = False

            # Check stop adjustment limits
            current_price = self._get_current_price()
            current_stop_pct = (current_price - self.position.stop_loss) / current_price

            if current_stop_pct <= self.config.min_stop_pct:
                mask[Action.TIGHTEN_STOP] = False
            if current_stop_pct >= self.config.max_stop_pct:
                mask[Action.LOOSEN_STOP] = False

            # Can't take partial if already at minimum size
            if self.position.size <= 0.5:
                mask[Action.TAKE_PARTIAL] = False

        # Check max trades limit
        if len(self.episode_trades) >= self.config.max_trades_per_episode:
            mask[Action.ENTER_LONG] = False

        return mask

    def _execute_action(self, action: int) -> Optional[TradeResult]:
        """
        Execute trading action.

        Returns:
            TradeResult if a trade was closed, None otherwise
        """
        current_price = self._get_current_price()
        trade_result = None

        if action == Action.WAIT:
            pass  # Do nothing

        elif action == Action.ENTER_LONG:
            if self.position is None:
                # Calculate entry price with slippage
                entry_price = current_price * (1 + self.config.slippage_pct)
                stop_loss = entry_price * (1 - self.config.initial_stop_pct)

                self.position = Position(
                    entry_price=entry_price,
                    entry_time=self.current_time,
                    size=1.0,
                    stop_loss=stop_loss,
                    highest_price=entry_price
                )
                # Record trade for spike tracking
                self._record_trade()
                logger.debug(f"Entered long at {entry_price:.2f}, stop at {stop_loss:.2f}")

        elif action == Action.HOLD:
            if self.position is not None:
                self.position.update_high(current_price)

        elif action == Action.TIGHTEN_STOP:
            if self.position is not None:
                new_stop = self.position.stop_loss + (current_price * self.config.stop_adjustment_pct)
                new_stop = min(new_stop, current_price * (1 - self.config.min_stop_pct))
                self.position.stop_loss = new_stop
                logger.debug(f"Tightened stop to {new_stop:.2f}")

        elif action == Action.LOOSEN_STOP:
            if self.position is not None:
                new_stop = self.position.stop_loss - (current_price * self.config.stop_adjustment_pct)
                new_stop = max(new_stop, current_price * (1 - self.config.max_stop_pct))
                self.position.stop_loss = new_stop
                logger.debug(f"Loosened stop to {new_stop:.2f}")

        elif action == Action.TAKE_PARTIAL:
            if self.position is not None and self.position.size > 0.5:
                # Close half position
                exit_price = current_price * (1 - self.config.slippage_pct)
                partial_size = self.position.size * 0.5
                return_pct = (exit_price - self.position.entry_price) / self.position.entry_price
                fees = self.config.fee_rate * 2  # Entry + exit fees

                trade_result = TradeResult(
                    entry_price=self.position.entry_price,
                    exit_price=exit_price,
                    entry_time=self.position.entry_time,
                    exit_time=self.current_time,
                    size=partial_size,
                    return_pct=return_pct - fees,
                    exit_reason='manual_partial',
                    fees_paid=fees * self.position.entry_price * partial_size,
                    highest_return=self.position.highest_return(),
                )
                self.episode_trades.append(trade_result)
                self.episode_returns.append(trade_result.return_pct * partial_size)
                self.position.size *= 0.5
                logger.debug(f"Took partial profit: {return_pct*100:.2f}%")

        elif action == Action.EXIT_ALL:
            if self.position is not None:
                trade_result = self._close_position('manual')

        # Check stop loss
        if self.position is not None and current_price <= self.position.stop_loss:
            trade_result = self._close_position('stop_loss')

        # Update position high watermark
        if self.position is not None:
            self.position.update_high(current_price)

        return trade_result

    def _close_position(self, reason: str) -> Optional[TradeResult]:
        """Close the current position."""
        if self.position is None:
            return None

        current_price = self._get_current_price()

        # Apply slippage (worse for stop loss)
        if reason == 'stop_loss':
            exit_price = self.position.stop_loss * (1 - self.config.slippage_pct)
        else:
            exit_price = current_price * (1 - self.config.slippage_pct)

        return_pct = (exit_price - self.position.entry_price) / self.position.entry_price
        fees = self.config.fee_rate * 2  # Entry + exit fees

        trade_result = TradeResult(
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            entry_time=self.position.entry_time,
            exit_time=self.current_time,
            size=self.position.size,
            return_pct=return_pct - fees,
            exit_reason=reason,
            fees_paid=fees * self.position.entry_price * self.position.size,
            highest_return=self.position.highest_return(),
        )

        self.episode_trades.append(trade_result)
        self.episode_returns.append(trade_result.return_pct * self.position.size)
        self.capital *= (1 + trade_result.return_pct * self.position.size)

        logger.debug(
            f"Closed position ({reason}): {return_pct*100:.2f}% return, "
            f"capital now {self.capital:.2f}"
        )

        self.position = None
        return trade_result

    def _check_termination(self) -> Tuple[bool, bool]:
        """
        Check if episode should terminate.

        Returns:
            (terminated, truncated)
        """
        # Check episode length
        if self.current_step >= self.config.episode_length:
            return False, True

        # Check drawdown
        episode_return = (self.capital - self.config.initial_capital) / self.config.initial_capital
        if episode_return < -self.config.terminate_on_drawdown:
            return True, False

        # Check if we've run out of data
        if self.current_time >= self.ohlcv_data.index[-1]:
            return False, True

        return False, False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode.

        Args:
            seed: Random seed
            options: Optional dict with 'symbol' and/or 'start_time' to override random sampling
                     Can also include 'max_retries' (default 10) for handling data gaps

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        options = options or {}
        max_retries = options.get('max_retries', 10)

        # If both symbol and start_time are specified, don't retry
        fixed_params = 'symbol' in options and 'start_time' in options

        last_error = None
        for attempt in range(max_retries):
            try:
                # Sample or use provided symbol and start time
                self.symbol = options.get('symbol') or self._sample_symbol()

                if 'start_time' in options:
                    self.start_time = pd.to_datetime(options['start_time'])
                elif self.config.random_start:
                    self.start_time = self._sample_start_time(self.symbol)
                else:
                    min_ts, _ = self._get_time_range(self.symbol)
                    self.start_time = min_ts + timedelta(minutes=self.config.lookback_minutes)

                # Reset episode state
                self.current_step = 0
                self.current_time = self.start_time
                self.position = None
                self.episode_trades = []
                self.episode_returns = []
                self.capital = self.config.initial_capital

                # Reset spike tracking state (v2)
                self.last_trade_time = None
                self.spike_start_time = None
                self.spike_traded = False
                self.trades_this_hour = 0
                self._trades_timestamps = []

                # Load data - this may raise ValueError if no data found
                self._load_episode_data()

                # Get initial observation
                observation = self._get_observation()
                self._prev_state = observation
                self._prev_price = self._get_current_price()

                info = {
                    'symbol': self.symbol,
                    'start_time': self.start_time.isoformat(),
                    'episode_length': self.config.episode_length,
                    'sampling_mode': self.config.sampling_mode,
                    'context_lookback': max(
                        self.config.lookback_minutes,
                        self.config.context_lookback_minutes
                    ),
                    'data_points': len(self.ohlcv_data),
                }

                logger.info(
                    f"Episode reset: {self.symbol} starting at {self.start_time} "
                    f"(mode={self.config.sampling_mode}, context={info['context_lookback']}min)"
                )

                return observation, info

            except ValueError as e:
                last_error = e
                if fixed_params:
                    # Don't retry if user specified both symbol and time
                    raise
                logger.warning(f"Reset attempt {attempt + 1}/{max_retries} failed: {e}")
                # Clear cached time range for this symbol so we re-query
                if self._available_time_ranges and self.symbol in self._available_time_ranges:
                    del self._available_time_ranges[self.symbol]
                continue

        # All retries exhausted
        raise RuntimeError(
            f"Failed to reset environment after {max_retries} attempts. "
            f"Last error: {last_error}. "
            f"The database may have insufficient data coverage."
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return next state.

        Args:
            action: Action to take (0-6)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Validate action with mask
        valid_actions = self.get_action_mask()
        if not valid_actions[action]:
            # Default to WAIT or HOLD if invalid
            if self.position is None:
                action = Action.WAIT
            else:
                action = Action.HOLD

        # Store previous state for reward calculation
        prev_state = self._prev_state
        prev_price = self._prev_price
        prev_position = self.position

        # Execute action
        trade_result = self._execute_action(action)

        # Advance time
        self.current_step += 1
        self.current_time = self.start_time + timedelta(minutes=self.current_step)

        # Get next observation
        observation = self._get_observation()
        current_price = self._get_current_price()

        # Build reward calculation kwargs
        reward_kwargs = dict(
            action=action,
            prev_price=prev_price,
            current_price=current_price,
            position=self.position,
            prev_position=prev_position,
            trade_result=trade_result,
            time_in_position=(
                (self.current_time - self.position.entry_time).total_seconds() / 60
                if self.position else 0
            ),
            episode_returns=self.episode_returns,
        )

        # Add spike context for spike-aware reward calculators
        from .rewards import SpikeAwareRewardCalculator, SpikeQualityBonusCalculator
        if isinstance(self.reward_calculator, (SpikeAwareRewardCalculator, SpikeQualityBonusCalculator)):
            reward_kwargs['spike_context'] = self._get_spike_context()

        # Calculate reward
        reward = self.reward_calculator.calculate(**reward_kwargs)

        # Check termination
        terminated, truncated = self._check_termination()

        # Close position at end of episode
        if (terminated or truncated) and self.position is not None:
            end_trade = self._close_position('end_of_episode')

        # Update previous state
        self._prev_state = observation
        self._prev_price = current_price

        # Build info dict
        info = {
            'symbol': self.symbol,
            'step': self.current_step,
            'current_time': self.current_time.isoformat() if self.current_time else None,
            'current_price': current_price,
            'has_position': self.position is not None,
            'position_return': (
                self.position.unrealized_return(current_price) if self.position else 0
            ),
            'episode_return': sum(self.episode_returns),
            'num_trades': len(self.episode_trades),
            'capital': self.capital,
            'action_taken': Action(action).name,
            'trade_closed': trade_result is not None,
            'action_mask': valid_actions,
        }

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render current state."""
        if self.render_mode == "human":
            print(self._get_render_string())
        elif self.render_mode == "ansi":
            return self._get_render_string()
        return None

    def _get_render_string(self) -> str:
        """Get string representation of current state."""
        current_price = self._get_current_price() if self.ohlcv_data is not None else 0

        lines = [
            f"Step: {self.current_step}/{self.config.episode_length}",
            f"Symbol: {self.symbol}",
            f"Time: {self.current_time}",
            f"Price: ${current_price:.2f}",
            f"Capital: ${self.capital:.2f}",
            f"Episode Return: {sum(self.episode_returns)*100:.2f}%",
            f"Trades: {len(self.episode_trades)}",
        ]

        if self.position:
            lines.extend([
                f"Position: LONG @ ${self.position.entry_price:.2f}",
                f"  Unrealized: {self.position.unrealized_return(current_price)*100:.2f}%",
                f"  Stop Loss: ${self.position.stop_loss:.2f}",
            ])
        else:
            lines.append("Position: FLAT")

        return "\n".join(lines)

    def close(self) -> None:
        """Clean up resources."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def make_trading_env(
    db_path: str = "/Users/bz/Pythia2/full_pythia.duckdb",
    config: Optional[EpisodeConfig] = None,
    **kwargs
) -> TradingEnvironment:
    """Factory function to create trading environment."""
    return TradingEnvironment(db_path=db_path, config=config, **kwargs)


if __name__ == "__main__":
    # Test the environment
    import sys

    print("=" * 60)
    print("Testing TradingEnvironment with Enhanced Sampling")
    print("=" * 60)

    # Test with SQLite (mock data) if DuckDB is locked
    db_path = "/Users/bz/Pythia2/rl_training_data.db"
    import os
    if not os.path.exists(db_path):
        print(f"SQLite training data not found at {db_path}")
        print("Using DuckDB instead...")
        db_path = "/Users/bz/Pythia2/full_pythia.duckdb"

    # Test different sampling modes
    for sampling_mode in ["random", "sequential"]:
        print(f"\n{'='*60}")
        print(f"Testing sampling_mode='{sampling_mode}'")
        print(f"{'='*60}")

        config = EpisodeConfig(
            episode_length=480,  # 8 hours
            sampling_mode=sampling_mode,
            window_overlap_pct=0.5,
            context_lookback_minutes=1440,  # 24hr context
        )

        env = TradingEnvironment(
            db_path=db_path,
            config=config,
            render_mode="human"
        )

        try:
            # Reset
            obs, info = env.reset()
            print(f"\nInitial observation shape: {obs.shape}")
            print(f"Initial info:")
            for k, v in info.items():
                print(f"  {k}: {v}")

            # Take a few steps
            for i in range(10):
                # Get valid actions
                mask = env.get_action_mask()
                valid_actions = np.where(mask)[0]

                # Random valid action
                action = np.random.choice(valid_actions)

                obs, reward, terminated, truncated, info = env.step(action)

                print(f"\nStep {i+1}:")
                print(f"  Action: {Action(action).name}")
                print(f"  Reward: {reward:.4f}")
                print(f"  Episode return: {info['episode_return']*100:.2f}%")

                if terminated or truncated:
                    break

            # Test sequential continuity (for sequential mode)
            if sampling_mode == "sequential":
                print("\n--- Testing sequential episode continuity ---")
                for ep in range(3):
                    obs, info = env.reset()
                    print(f"Episode {ep+1}: starts at {info['start_time']}")

            print(f"\n{sampling_mode} mode test completed successfully!")

        except Exception as e:
            print(f"Error with {sampling_mode} mode: {e}")
            import traceback
            traceback.print_exc()
        finally:
            env.close()

    print("\n" + "=" * 60)
    print("All environment tests completed!")
    print("=" * 60)
