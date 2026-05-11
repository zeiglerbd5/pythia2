"""
Entry Timing Environment for RL Agent

This environment focuses ONLY on entry timing. The agent learns:
- WHEN to enter a position (binary decision: wait or enter)

Exits are handled by RULES, not learned:
- Take profit: 12% gain
- Stop loss: 2% loss
- Max hold time: 24 hours (1440 minutes)

This separation addresses the core problem: entry timing is the hard part.
Exit timing can be mechanical with fixed rules.

Key insight from failed experiments:
- Previous 7-action agent learned to spam entries (255 per episode)
- All trades lost ~1.3% due to immediate exits
- 0% win rate despite positive training rewards
- The agent was reward hacking by rapidly cycling through entries

This simplified approach:
- Only 2 actions (WAIT=0, ENTER=1)
- Can only enter when not in position
- Reward is SPARSE - only when trade completes via rule-based exit
- No way to hack the reward by rapid cycling
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


class EntryAction(IntEnum):
    """Binary action space for entry timing."""
    WAIT = 0   # Do nothing, continue observing
    ENTER = 1  # Open long position


class ExitReason(IntEnum):
    """Reasons for position exit (all rule-based)."""
    NONE = 0
    TAKE_PROFIT = 1     # Hit take profit target
    STOP_LOSS = 2       # Hit stop loss
    MAX_HOLD_TIME = 3   # Exceeded max hold duration
    END_OF_EPISODE = 4  # Episode ended while in position


@dataclass
class EntryTimingConfig:
    """Configuration for entry timing environment."""
    # Episode duration
    episode_length: int = 1440        # 24 hours of 1-minute decisions
    step_size_minutes: int = 1        # Decision frequency

    # Data sampling
    symbols: Optional[List[str]] = None  # None = sample from available
    random_start: bool = True         # Random start time in data

    # Position constraints
    max_trades_per_episode: int = 20  # Limit trades to prevent spam
    initial_capital: float = 10000.0  # Starting capital
    cooldown_minutes: int = 30        # Minimum time between entries

    # Exit rules (FIXED - not learned)
    take_profit_pct: float = 0.12     # 12% take profit
    stop_loss_pct: float = 0.02       # 2% stop loss
    max_hold_minutes: int = 1440      # 24 hours max hold

    # Transaction costs
    fee_rate: float = 0.0055          # 0.55% total fees (maker+taker)
    slippage_pct: float = 0.001       # 0.1% slippage estimate

    # Termination conditions
    terminate_on_drawdown: float = 0.20  # -20% episode loss

    # Lookback for features
    lookback_minutes: int = 60        # Historical context needed
    context_lookback_minutes: int = 1440  # 24hr context for features

    # Sampling mode
    sampling_mode: str = "event_anchored"  # Bias toward spikes

    # Event-anchored sampling parameters
    event_volume_threshold: float = 3.0    # Volume spike multiplier
    event_volatility_threshold: float = 2.0  # Volatility spike multiplier
    event_bias_probability: float = 0.7    # Probability of sampling near events


@dataclass
class EntryPosition:
    """Represents an open trading position."""
    entry_price: float
    entry_time: datetime
    stop_loss_price: float
    take_profit_price: float
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
class EntryTradeResult:
    """Result of a completed trade."""
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    return_pct: float        # Net return after fees
    exit_reason: ExitReason
    fees_paid: float
    highest_return: float    # Max unrealized return during position
    hold_duration_minutes: int


class EntryTimingEnvironment(gym.Env):
    """
    Entry Timing Environment.

    The agent ONLY decides WHEN to enter. Exits are automatic:
    - Take profit at 12% gain
    - Stop loss at 2% loss
    - Max hold time of 24 hours

    This makes the problem much simpler and prevents reward hacking
    through rapid entry/exit cycles.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        db_path: str = "/Users/bz/Pythia2/full_pythia.duckdb",
        config: Optional[EntryTimingConfig] = None,
        reward_calculator: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize entry timing environment.

        Args:
            db_path: Path to DuckDB database
            config: Environment configuration
            reward_calculator: Optional reward calculator
            feature_extractor: Optional feature extractor
            render_mode: Render mode for visualization
        """
        super().__init__()

        self.db_path = db_path
        self.config = config or EntryTimingConfig()
        self.render_mode = render_mode

        # Lazy imports to avoid circular dependencies
        if reward_calculator is None:
            from .entry_timing_rewards import EntryTimingRewardCalculator, EntryTimingRewardConfig
            self.reward_calculator = EntryTimingRewardCalculator(EntryTimingRewardConfig())
        else:
            self.reward_calculator = reward_calculator

        if feature_extractor is None:
            from .features import FeatureExtractor, FeatureConfig
            self.feature_extractor = FeatureExtractor(FeatureConfig())
        else:
            self.feature_extractor = feature_extractor

        # State dimension from feature extractor + position features
        # Position features: 8 (simplified from original 11)
        self._position_feature_dim = 8
        self.state_dim = self.feature_extractor.get_state_dim() + self._position_feature_dim

        # Define spaces - BINARY action space
        self.action_space = spaces.Discrete(2)  # WAIT=0, ENTER=1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Database connection (lazy initialization)
        self._conn: Optional[Union[duckdb.DuckDBPyConnection, sqlite3.Connection]] = None
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
        self.position: Optional[EntryPosition] = None
        self.episode_trades: List[EntryTradeResult] = []
        self.capital: float = self.config.initial_capital

        # Cooldown tracking
        self.last_entry_time: Optional[datetime] = None
        self.last_exit_time: Optional[datetime] = None

        # Previous state for reward calculation
        self._prev_price: Optional[float] = None

        # Spike tracking for features
        self.spike_start_time: Optional[datetime] = None
        self.spike_traded: bool = False
        self._volume_ma: Optional[float] = None
        self._volatility_ma: Optional[float] = None

        logger.info(f"EntryTimingEnvironment initialized with state_dim={self.state_dim}")
        logger.info(f"Exit rules: TP={self.config.take_profit_pct*100}%, SL={self.config.stop_loss_pct*100}%, MaxHold={self.config.max_hold_minutes}min")

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
        """Execute query and return DataFrame."""
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
            min_records = self.config.episode_length + self.config.lookback_minutes + 100

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
            logger.info(f"Found {len(self._available_symbols)} symbols with sufficient data")
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

    def _sample_start_time(self, symbol: str, max_attempts: int = 10) -> datetime:
        """Sample start time, biased toward interesting events."""
        min_ts, max_ts = self._get_time_range(symbol)

        effective_lookback = max(
            self.config.lookback_minutes,
            self.config.context_lookback_minutes
        )

        earliest_start = min_ts + timedelta(minutes=effective_lookback)
        latest_start = max_ts - timedelta(minutes=self.config.episode_length)

        if earliest_start >= latest_start:
            return earliest_start

        # Event-anchored sampling
        if self.config.sampling_mode == "event_anchored" and np.random.random() < self.config.event_bias_probability:
            events = self._find_market_events(symbol)
            if events and len(events) > 0:
                valid_events = [e for e in events if earliest_start <= e <= latest_start]
                if valid_events:
                    event_time = np.random.choice(valid_events)
                    lead_time = np.random.randint(30, 61)
                    start_time = event_time - timedelta(minutes=lead_time)
                    return max(earliest_start, min(start_time, latest_start))

        # Random sampling
        time_range = (latest_start - earliest_start).total_seconds()
        random_offset = np.random.uniform(0, time_range)
        return earliest_start + timedelta(seconds=random_offset)

    def _find_market_events(self, symbol: str) -> Optional[List[datetime]]:
        """Find interesting market events for a symbol."""
        try:
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
            df['volume_ma'] = df['volume'].rolling(60).mean()
            df['range'] = (df['high'] - df['low']) / df['close']
            df['range_ma'] = df['range'].rolling(60).mean()

            volume_events = df[df['volume'] > df['volume_ma'] * self.config.event_volume_threshold]['timestamp'].tolist()
            volatility_events = df[df['range'] > df['range_ma'] * self.config.event_volatility_threshold]['timestamp'].tolist()

            all_events = sorted(set(volume_events + volatility_events))

            if len(all_events) == 0:
                return None

            # Cluster events within 30 minutes
            clustered = [all_events[0]]
            for event in all_events[1:]:
                if (event - clustered[-1]).total_seconds() > 1800:
                    clustered.append(event)

            return clustered

        except Exception as e:
            logger.warning(f"Error finding events for {symbol}: {e}")
            return None

    def _load_episode_data(self) -> None:
        """Load OHLCV and feature data for current episode."""
        effective_lookback = max(
            self.config.lookback_minutes,
            self.config.context_lookback_minutes
        )
        data_start = self.start_time - timedelta(minutes=effective_lookback)
        data_end = self.start_time + timedelta(minutes=self.config.episode_length)

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

        self.ohlcv_data['timestamp'] = pd.to_datetime(self.ohlcv_data['timestamp'])
        self.ohlcv_data.set_index('timestamp', inplace=True)

        # Calculate features
        self.features_data = self.feature_extractor.calculate_features(self.ohlcv_data)

        logger.debug(f"Loaded episode data: {len(self.ohlcv_data)} candles, {len(self.features_data)} feature rows")

    def _get_current_price(self) -> float:
        """Get current close price."""
        available_times = self.ohlcv_data.index[self.ohlcv_data.index <= self.current_time]
        if len(available_times) == 0:
            return self.ohlcv_data['close'].iloc[0]
        return self.ohlcv_data.loc[available_times[-1], 'close']

    def _get_state_features_for_reward(self) -> dict:
        """Extract key state features for reward calculation (setup quality estimation)."""
        try:
            available_times = self.features_data.index[self.features_data.index <= self.current_time]
            if len(available_times) == 0:
                return {}

            latest = self.features_data.loc[available_times[-1]]

            # Extract features if they exist (names may vary)
            state_features = {}

            # Look for ATR z-score
            for col in ['atr_zscore', 'atr_z', 'volatility_zscore', 'atr_14_zscore']:
                if col in latest.index:
                    state_features['atr_zscore'] = float(latest[col])
                    break
            if 'atr_zscore' not in state_features:
                # Compute from ATR if available
                if 'atr_14' in latest.index:
                    state_features['atr_zscore'] = 0.0  # Default

            # Look for volume z-score
            for col in ['volume_zscore', 'vol_zscore', 'volume_z', 'volume_ratio']:
                if col in latest.index:
                    val = float(latest[col])
                    # Convert ratio to z-score-like
                    if col == 'volume_ratio':
                        state_features['volume_zscore'] = (val - 1.0) * 2  # Approximate
                    else:
                        state_features['volume_zscore'] = val
                    break
            if 'volume_zscore' not in state_features:
                state_features['volume_zscore'] = 0.0

            # Look for RSI
            for col in ['rsi_14', 'rsi', 'rsi_7']:
                if col in latest.index:
                    state_features['rsi'] = float(latest[col])
                    break
            if 'rsi' not in state_features:
                state_features['rsi'] = 50.0

            return state_features

        except Exception as e:
            logger.debug(f"Error extracting state features: {e}")
            return {}

    def _get_observation(self) -> np.ndarray:
        """Get current observation (state)."""
        available_times = self.features_data.index[self.features_data.index <= self.current_time]
        if len(available_times) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)

        latest_features = self.features_data.loc[available_times[-1]]
        position_features = self._get_position_features()

        market_features = latest_features.values.astype(np.float32)
        full_state = np.concatenate([market_features, position_features])

        full_state = np.nan_to_num(full_state, nan=0.0, posinf=0.0, neginf=0.0)

        return full_state.astype(np.float32)

    def _get_position_features(self) -> np.ndarray:
        """Get position context features."""
        current_price = self._get_current_price()

        # Cooldown feature
        if self.last_exit_time is None:
            cooldown_elapsed = 1.0  # No recent exit, fully cooled
        else:
            minutes_since_exit = (self.current_time - self.last_exit_time).total_seconds() / 60
            cooldown_elapsed = min(minutes_since_exit / self.config.cooldown_minutes, 1.0)

        # Spike detection features
        is_spike, spike_freshness = self._detect_current_spike()

        if self.position is None:
            return np.array([
                0.0,  # has_position
                0.0,  # position_return_pct
                0.0,  # time_in_position (normalized)
                0.0,  # highest_return_seen
                cooldown_elapsed,  # cooldown progress (1.0 = can enter)
                1.0 if is_spike else 0.0,  # is_in_spike
                spike_freshness,  # spike_freshness (1.0 = fresh, 0.0 = stale)
                1.0 if self.spike_traded else 0.0,  # already_traded_this_spike
            ], dtype=np.float32)
        else:
            time_in_position = (self.current_time - self.position.entry_time).total_seconds() / 60
            return np.array([
                1.0,  # has_position
                self.position.unrealized_return(current_price),
                min(time_in_position / self.config.max_hold_minutes, 1.0),  # normalized
                self.position.highest_return(),
                cooldown_elapsed,
                1.0 if is_spike else 0.0,
                spike_freshness,
                1.0 if self.spike_traded else 0.0,
            ], dtype=np.float32)

    def _detect_current_spike(self) -> Tuple[bool, float]:
        """Detect if we're currently in a spike and compute its freshness."""
        if self.ohlcv_data is None or self.current_time is None:
            return False, 0.0

        lookback = 60
        current_idx = self.ohlcv_data.index.get_indexer([self.current_time], method='ffill')[0]

        if current_idx < lookback:
            return False, 0.0

        recent_data = self.ohlcv_data.iloc[max(0, current_idx - lookback):current_idx + 1]

        if len(recent_data) < 10:
            return False, 0.0

        current_volume = recent_data['volume'].iloc[-1]
        avg_volume = recent_data['volume'].iloc[:-1].mean()

        current_range = (recent_data['high'].iloc[-1] - recent_data['low'].iloc[-1]) / recent_data['close'].iloc[-1]
        avg_range = ((recent_data['high'].iloc[:-1] - recent_data['low'].iloc[:-1]) / recent_data['close'].iloc[:-1]).mean()

        volume_spike = current_volume > avg_volume * self.config.event_volume_threshold
        volatility_spike = current_range > avg_range * self.config.event_volatility_threshold

        is_spike = volume_spike or volatility_spike

        if is_spike:
            if self.spike_start_time is None:
                self.spike_start_time = self.current_time
                self.spike_traded = False

            spike_age_minutes = (self.current_time - self.spike_start_time).total_seconds() / 60
            freshness = max(0.0, 1.0 - (spike_age_minutes / 60.0))
        else:
            self.spike_start_time = None
            freshness = 0.0

        return is_spike, freshness

    def get_action_mask(self) -> np.ndarray:
        """Get valid action mask based on current state."""
        mask = np.ones(2, dtype=bool)

        # Can't enter if already in position
        if self.position is not None:
            mask[EntryAction.ENTER] = False

        # Can't enter if in cooldown
        if self.last_exit_time is not None:
            minutes_since_exit = (self.current_time - self.last_exit_time).total_seconds() / 60
            if minutes_since_exit < self.config.cooldown_minutes:
                mask[EntryAction.ENTER] = False

        # Can't enter if max trades reached
        if len(self.episode_trades) >= self.config.max_trades_per_episode:
            mask[EntryAction.ENTER] = False

        return mask

    def _check_exit_conditions(self) -> Optional[ExitReason]:
        """Check if any exit condition is met."""
        if self.position is None:
            return None

        current_price = self._get_current_price()

        # Check take profit
        if current_price >= self.position.take_profit_price:
            return ExitReason.TAKE_PROFIT

        # Check stop loss
        if current_price <= self.position.stop_loss_price:
            return ExitReason.STOP_LOSS

        # Check max hold time
        hold_duration = (self.current_time - self.position.entry_time).total_seconds() / 60
        if hold_duration >= self.config.max_hold_minutes:
            return ExitReason.MAX_HOLD_TIME

        return None

    def _execute_entry(self) -> None:
        """Execute entry into position."""
        current_price = self._get_current_price()

        # Calculate entry price with slippage
        entry_price = current_price * (1 + self.config.slippage_pct)

        # Calculate exit levels
        stop_loss_price = entry_price * (1 - self.config.stop_loss_pct)
        take_profit_price = entry_price * (1 + self.config.take_profit_pct)

        self.position = EntryPosition(
            entry_price=entry_price,
            entry_time=self.current_time,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            highest_price=entry_price,
        )

        self.last_entry_time = self.current_time

        # Mark spike as traded
        if self.spike_start_time is not None:
            self.spike_traded = True

        logger.debug(
            f"Entered at {entry_price:.2f}, "
            f"SL={stop_loss_price:.2f} ({self.config.stop_loss_pct*100}%), "
            f"TP={take_profit_price:.2f} ({self.config.take_profit_pct*100}%)"
        )

    def _execute_exit(self, reason: ExitReason) -> EntryTradeResult:
        """Execute exit from position."""
        current_price = self._get_current_price()

        # Determine exit price based on reason
        if reason == ExitReason.TAKE_PROFIT:
            exit_price = self.position.take_profit_price * (1 - self.config.slippage_pct)
        elif reason == ExitReason.STOP_LOSS:
            exit_price = self.position.stop_loss_price * (1 - self.config.slippage_pct)
        else:
            exit_price = current_price * (1 - self.config.slippage_pct)

        # Calculate return
        gross_return = (exit_price - self.position.entry_price) / self.position.entry_price
        fees = self.config.fee_rate * 2  # Entry + exit fees
        net_return = gross_return - fees

        hold_duration = int((self.current_time - self.position.entry_time).total_seconds() / 60)

        trade_result = EntryTradeResult(
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            entry_time=self.position.entry_time,
            exit_time=self.current_time,
            return_pct=net_return,
            exit_reason=reason,
            fees_paid=fees * self.position.entry_price,
            highest_return=self.position.highest_return(),
            hold_duration_minutes=hold_duration,
        )

        self.episode_trades.append(trade_result)
        self.capital *= (1 + net_return)

        logger.debug(
            f"Exited ({reason.name}): {net_return*100:.2f}% return, "
            f"hold={hold_duration}min, capital={self.capital:.2f}"
        )

        self.last_exit_time = self.current_time
        self.position = None

        return trade_result

    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate."""
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
        """Reset environment for new episode."""
        super().reset(seed=seed)

        options = options or {}
        max_retries = options.get('max_retries', 10)

        fixed_params = 'symbol' in options and 'start_time' in options

        last_error = None
        for attempt in range(max_retries):
            try:
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
                self.capital = self.config.initial_capital

                # Reset cooldown tracking
                self.last_entry_time = None
                self.last_exit_time = None

                # Reset spike tracking
                self.spike_start_time = None
                self.spike_traded = False

                # Load data
                self._load_episode_data()

                # Get initial observation
                observation = self._get_observation()
                self._prev_price = self._get_current_price()

                info = {
                    'symbol': self.symbol,
                    'start_time': self.start_time.isoformat(),
                    'episode_length': self.config.episode_length,
                    'exit_rules': {
                        'take_profit': self.config.take_profit_pct,
                        'stop_loss': self.config.stop_loss_pct,
                        'max_hold_minutes': self.config.max_hold_minutes,
                    },
                    'data_points': len(self.ohlcv_data),
                }

                logger.info(
                    f"Episode reset: {self.symbol} starting at {self.start_time}"
                )

                return observation, info

            except ValueError as e:
                last_error = e
                if fixed_params:
                    raise
                logger.warning(f"Reset attempt {attempt + 1}/{max_retries} failed: {e}")
                continue

        raise RuntimeError(
            f"Failed to reset environment after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return next state."""
        # Validate action with mask
        valid_actions = self.get_action_mask()
        if not valid_actions[action]:
            action = EntryAction.WAIT  # Default to WAIT if invalid

        trade_result = None

        # Execute action (only ENTER matters, WAIT does nothing)
        if action == EntryAction.ENTER and self.position is None:
            self._execute_entry()

        # Check for rule-based exit if in position
        if self.position is not None:
            # Update high watermark
            current_price = self._get_current_price()
            self.position.update_high(current_price)

            # Check exit conditions
            exit_reason = self._check_exit_conditions()
            if exit_reason is not None:
                trade_result = self._execute_exit(exit_reason)

        # Advance time
        self.current_step += 1
        self.current_time = self.start_time + timedelta(minutes=self.current_step)

        # Get next observation
        observation = self._get_observation()
        current_price = self._get_current_price()

        # Get spike context for reward
        is_spike, spike_freshness = self._detect_current_spike()
        spike_context = {
            'is_in_spike': is_spike,
            'spike_freshness': spike_freshness,
            'spike_traded': self.spike_traded,
        }

        # Get state features for setup quality estimation
        state_features = self._get_state_features_for_reward()

        # Calculate reward
        reward = self.reward_calculator.calculate(
            action=action,
            trade_result=trade_result,
            position=self.position,
            spike_context=spike_context,
            episode_trades=self.episode_trades,
            state_features=state_features,
        )

        # Check termination
        terminated, truncated = self._check_termination()

        # Close position at end of episode
        if (terminated or truncated) and self.position is not None:
            trade_result = self._execute_exit(ExitReason.END_OF_EPISODE)
            # Recalculate reward for this trade
            reward += self.reward_calculator.calculate(
                action=action,
                trade_result=trade_result,
                position=None,
                spike_context=spike_context,
                episode_trades=self.episode_trades,
                state_features=state_features,
            )

        # Update previous price
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
            'episode_return': (self.capital - self.config.initial_capital) / self.config.initial_capital,
            'num_trades': len(self.episode_trades),
            'capital': self.capital,
            'action_taken': EntryAction(action).name,
            'trade_closed': trade_result is not None,
            'trade_result': trade_result,
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
            f"Episode Return: {(self.capital - self.config.initial_capital) / self.config.initial_capital * 100:.2f}%",
            f"Trades: {len(self.episode_trades)}",
        ]

        if self.position:
            lines.extend([
                f"Position: LONG @ ${self.position.entry_price:.2f}",
                f"  Unrealized: {self.position.unrealized_return(current_price)*100:.2f}%",
                f"  Stop Loss: ${self.position.stop_loss_price:.2f}",
                f"  Take Profit: ${self.position.take_profit_price:.2f}",
            ])
        else:
            lines.append("Position: FLAT")

        # Trade summary
        if self.episode_trades:
            wins = [t for t in self.episode_trades if t.return_pct > 0]
            win_rate = len(wins) / len(self.episode_trades) * 100
            lines.append(f"Win Rate: {win_rate:.1f}%")

        return "\n".join(lines)

    def close(self) -> None:
        """Clean up resources."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def make_entry_timing_env(
    db_path: str = "/Users/bz/Pythia2/full_pythia.duckdb",
    config: Optional[EntryTimingConfig] = None,
    **kwargs
) -> EntryTimingEnvironment:
    """Factory function to create entry timing environment."""
    return EntryTimingEnvironment(db_path=db_path, config=config, **kwargs)


if __name__ == "__main__":
    # Test the environment
    print("=" * 60)
    print("Testing EntryTimingEnvironment")
    print("=" * 60)

    db_path = "/Users/bz/Pythia2/rl_training_data.db"
    import os
    if not os.path.exists(db_path):
        print(f"SQLite training data not found at {db_path}")
        print("Using DuckDB instead...")
        db_path = "/Users/bz/Pythia2/full_pythia.duckdb"

    config = EntryTimingConfig(
        episode_length=480,  # 8 hours for quick test
        take_profit_pct=0.12,
        stop_loss_pct=0.02,
        max_hold_minutes=1440,
    )

    env = EntryTimingEnvironment(
        db_path=db_path,
        config=config,
        render_mode="human"
    )

    try:
        obs, info = env.reset()
        print(f"\nInitial observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        print(f"Initial info: {info}")

        total_reward = 0
        entries = 0

        # Run through an episode
        for i in range(480):
            mask = env.get_action_mask()

            # Simple strategy: enter on fresh spikes
            if mask[EntryAction.ENTER] and np.random.random() < 0.05:  # 5% entry chance
                action = EntryAction.ENTER
                entries += 1
            else:
                action = EntryAction.WAIT

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if info['trade_closed']:
                print(f"\nStep {i}: Trade closed!")
                print(f"  Result: {info['trade_result'].exit_reason.name}")
                print(f"  Return: {info['trade_result'].return_pct*100:.2f}%")

            if terminated or truncated:
                break

        print(f"\n" + "=" * 60)
        print("Episode Summary")
        print("=" * 60)
        print(f"Total entries: {entries}")
        print(f"Total trades: {len(env.episode_trades)}")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Final capital: ${env.capital:.2f}")
        print(f"Episode return: {(env.capital - config.initial_capital) / config.initial_capital * 100:.2f}%")

        if env.episode_trades:
            wins = [t for t in env.episode_trades if t.return_pct > 0]
            print(f"Win rate: {len(wins) / len(env.episode_trades) * 100:.1f}%")

            # Exit reason breakdown
            reasons = {}
            for t in env.episode_trades:
                reasons[t.exit_reason.name] = reasons.get(t.exit_reason.name, 0) + 1
            print(f"Exit reasons: {reasons}")

        print("\nEntryTimingEnvironment test completed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
