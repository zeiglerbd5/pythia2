# Reinforcement Learning-Based Crypto Trading System Architecture

**Document Version**: 1.0
**Date**: March 17, 2026
**Author**: AI/ML Engineering Analysis

---

## Executive Summary

This document proposes an architecture for a Reinforcement Learning (RL) based crypto trading system designed to "navigate like whitewater rapids" - making continuous micro-decisions based on current market state rather than predicting future prices. The system leverages ~90GB of historical data in the Pythia2 DuckDB database.

**Key Design Principles:**
1. **Reactive, not predictive**: Respond to market state, don't forecast prices
2. **Continuous decisions**: Every timestep is a decision point (hold, adjust, act)
3. **Attention-based state**: Let the model learn WHAT to focus on
4. **Risk-first rewards**: Optimize risk-adjusted returns, not raw P&L
5. **Continuous learning**: Adapt to regime changes without catastrophic forgetting

---

## 1. Data Inventory

### 1.1 Primary Database: full_pythia.duckdb (90GB)

| Table | Description | Key Columns | Estimated Rows |
|-------|-------------|-------------|----------------|
| `trades` | L1 trade executions | symbol, timestamp, price, size, side | ~1.18 billion |
| `order_book_snapshots` | L2 order book depth | symbol, timestamp, bids (JSON), asks (JSON), spread_bps | ~100M+ |
| `ohlcv` | Candlestick data (1m, 5m, 15m) | symbol, timestamp, timeframe, OHLCV, num_trades | ~25M |
| `tickers` | L1 quote data | symbol, timestamp, best_bid/ask, volume_24h | ~100M+ |
| `features` | Pre-computed indicators | symbol, timestamp, vpin, rsi_14, vwap, bb_*, spread_pct | ~10M+ |
| `whale_transactions` | Large transfers | symbol, timestamp, amount_usd, subtype, from/to | ~100K |
| `news_signals` | Whale alerts | symbol, timestamp, confidence, sentiment_score | ~50K |

**Date Range**: October 2025 - March 2026 (~5 months)
**Symbols**: ~380 crypto pairs (BTC-USD, ETH-USD, etc.)

### 1.2 Feature Buffer: feature_buffer.db (SQLite, ~600MB)

Real-time rolling buffer for live trading:
- `features`: Real-time computed indicators (natr, bid_ask_spread_pct, volume_zscore, RSI, VWAP_distance)
- `ohlcv`: Recent 1-minute candles (~565K rows)
- `order_book_snapshots`: Recent L2 snapshots (~25K rows)
- `trades`: Recent trades for aggregation (~2.5M rows)

### 1.3 Available Raw Data Types

```
+------------------+     +------------------+     +------------------+
|   Trade Flow     |     |   Order Book     |     |   Price/Volume   |
+------------------+     +------------------+     +------------------+
| - Individual     |     | - L2 depth       |     | - OHLCV candles  |
|   trades         |     | - Bid/ask arrays |     | - Multiple TFs   |
| - Side (buy/sell)|     | - Spread         |     | - Volume profile |
| - Size           |     | - Imbalance      |     | - VWAP           |
| - Timestamp (ms) |     | - Mid price      |     | - Momentum       |
+------------------+     +------------------+     +------------------+
         |                       |                       |
         v                       v                       v
+--------------------------------------------------------------------+
|                    MARKET STATE REPRESENTATION                      |
+--------------------------------------------------------------------+
```

---

## 2. Previous ML Attempts: Analysis and Learnings

### 2.1 Summary of Existing Models

| Model | Architecture | Target | Best Performance | Issues |
|-------|--------------|--------|------------------|--------|
| CNN-LSTM | Conv1D + LSTM | 15% spike in 60min | ~F1 0.55 | Focal loss, SMOTE |
| Transformer | 4-layer encoder, attention pooling | 20%+ in 24h | Val loss 0.4 | Memory optimized |
| XGBoost v4 | Gradient boosting | 15% spike in 60min | Precision 0.64 | Raw features approach |
| Event Classifier | XGBoost + LR ensemble | Spike/no-spike | F1 ~0.55 | Rolling validation |
| Full Dataset XGB | XGBoost | Binary spike | Prec@90 = 0.96, Recall = 0.16 | High threshold only |

### 2.2 Why Previous Approaches May Have Struggled

**1. Prediction vs. Navigation Paradigm Mismatch**
- All models tried to PREDICT spikes in advance
- Spike prediction is extremely hard (rare events, regime-dependent)
- Even "good" models (96% precision) only catch 16% of spikes

**2. Class Imbalance Hell**
- Spikes are rare (<5% of samples)
- SMOTE, focal loss, class weights all help but don't solve the fundamental issue
- High precision comes at cost of abysmal recall

**3. Feature Leakage Concerns**
- Many features (RSI, MACD, BB) are lagging indicators
- V4 tried "raw" features but still used rolling windows that may leak
- Rolling-origin backtests showed degraded performance vs. random splits

**4. Temporal Non-Stationarity**
- Crypto market regimes change constantly
- Models trained on one regime fail in another
- No mechanism for continuous adaptation

**5. Single-Shot Decisions**
- Binary classification: enter/don't enter
- No concept of position management, stop adjustment, partial exits
- Doesn't match how real trading works

### 2.3 What Worked in the Rule-Based System

The Breakout Hunter v5.3 achieved **94.6% win rate** using:
- **Multi-stage confirmation**: T+0 detect, T+1 filter, T+2 entry
- **Reactive exits**: Trail stops, profit locks based on current state
- **Volume continuation**: Checking if buying pressure persists
- **No prediction**: Just pattern matching on current/recent state

This suggests RL should focus on **reaction and adaptation**, not prediction.

---

## 3. RL Architecture Design

### 3.1 Core Philosophy: The Whitewater Rapids Paradigm

```
Traditional ML:                    RL Navigation:

  "Predict the future"             "React to the current"

  [State] --> [Prediction]         [State] --> [Action] --> [State']
              "Will spike?"                    "What now?"

  Binary decision                  Continuous decisions
  Enter or don't                   Enter, hold, tighten, exit, wait
```

The agent doesn't predict WHERE the market is going. It observes the CURRENT state and decides the best action RIGHT NOW, like a kayaker navigating rapids.

### 3.2 State Space Design

#### 3.2.1 Multi-Timescale Feature Hierarchy

```
                    +------------------------+
                    |    COMPOSITE STATE     |
                    |    (d = 128-256 dims)  |
                    +------------------------+
                              ^
        +---------------------+---------------------+
        |                     |                     |
+---------------+     +---------------+     +---------------+
| Micro State   |     | Meso State    |     | Macro State   |
| (1-5 min)     |     | (1-4 hours)   |     | (24h+)        |
+---------------+     +---------------+     +---------------+
| - Trade flow  |     | - Trend       |     | - Regime      |
| - Book imbal  |     | - Volatility  |     | - BTC corr    |
| - Spread      |     | - Volume prof |     | - Market cap  |
| - Tick moves  |     | - Support/Res |     | - Sector mom  |
+---------------+     +---------------+     +---------------+
```

#### 3.2.2 Feature Groups

**Group A: Microstructure (10-15 features)**
```python
# From L2 order book
order_book_imbalance_l5      # (bid_vol - ask_vol) / total at 5 levels
bid_ask_spread_bps           # Current spread in basis points
spread_zscore                # Spread vs. recent history
depth_ratio                  # Near depth / far depth
large_order_imbalance        # Large orders on bid vs. ask side

# From trade flow
trade_imbalance_1m           # Buy vol - Sell vol, 1 minute
trade_imbalance_5m           # Buy vol - Sell vol, 5 minutes
vpin                         # Volume-synchronized informed trading
large_trade_ratio            # Trades > 2x average size
trade_frequency_ratio        # Current rate / average rate
```

**Group B: Price Dynamics (8-10 features)**
```python
returns_1m, returns_5m, returns_15m, returns_1h  # Multi-scale returns
realized_volatility_5m       # sqrt(sum(returns^2))
natr                         # Normalized ATR
price_position_24h           # (price - low) / (high - low)
distance_from_vwap           # Signed distance from VWAP
momentum_score               # Composite momentum indicator
```

**Group C: Volume Profile (6-8 features)**
```python
volume_zscore                # Current vs. rolling mean
volume_acceleration          # d(volume)/dt
volume_ratio_1h_24h          # Short-term vs. long-term volume
buy_pressure_ratio           # Buy volume / total volume
volume_at_price_position     # Volume profile context
```

**Group D: Market Context (4-6 features)**
```python
btc_returns_1h               # BTC performance (market leader)
correlation_to_btc_24h       # Symbol's correlation to BTC
market_regime_indicator      # Bull/bear/sideways from BTC
hour_of_day_embedding        # Cyclical encoding of time
day_of_week_embedding        # Weekend effects
```

**Group E: Position Context (4-6 features)**
```python
has_position                 # Binary: in trade or not
position_return_pct          # Unrealized P&L
time_in_position             # Normalized duration
highest_return_seen          # Max unrealized gain
current_stop_distance        # Distance from stop loss
```

#### 3.2.3 State Tensor Structure

```python
@dataclass
class MarketState:
    """Complete state representation for RL agent."""

    # Current timestep features
    current_features: torch.Tensor  # (n_features,) ~40 dims

    # Historical context (for attention)
    history_micro: torch.Tensor     # (seq_len_micro, n_micro)  e.g., (60, 15) = 60 minutes
    history_meso: torch.Tensor      # (seq_len_meso, n_meso)    e.g., (24, 10) = 24 hours
    history_macro: torch.Tensor     # (seq_len_macro, n_macro)  e.g., (7, 6) = 7 days

    # Position state
    position_context: torch.Tensor  # (n_position_features,) ~6 dims

    # Time encoding
    time_embedding: torch.Tensor    # (time_dim,) ~16 dims

# Total state dimension: ~40 + attention(60*15 + 24*10 + 7*6) + 6 + 16
# Effective: ~60-80 dims after attention pooling
```

### 3.3 Action Space Design

#### 3.3.1 Option 1: Discrete Actions (Recommended for Phase 1)

```python
class DiscreteActionSpace:
    """
    Discrete action space for initial implementation.

    Actions are mutually exclusive and cover all scenarios.
    """

    # No position actions
    WAIT = 0              # Do nothing, continue observing
    ENTER_LONG = 1        # Open long position (fixed size)

    # In position actions
    HOLD = 2              # Maintain current position
    TIGHTEN_STOP = 3      # Move stop loss closer (reduce risk)
    LOOSEN_STOP = 4       # Move stop loss further (give room)
    TAKE_PARTIAL = 5      # Exit 50% of position
    EXIT_ALL = 6          # Close entire position

    n_actions = 7
```

#### 3.3.2 Option 2: Hybrid Action Space (Phase 2)

```python
class HybridActionSpace:
    """
    Combines discrete action type with continuous parameters.
    """

    # Discrete component: action type
    action_type: int  # One of the 7 discrete actions

    # Continuous components (only used for certain actions)
    position_size: float      # [0, 1] fraction of max position
    stop_adjustment: float    # [-0.05, 0.05] relative stop change
    target_adjustment: float  # [-0.05, 0.05] relative target change
```

#### 3.3.3 Action Masking

```python
def get_valid_actions(state: MarketState) -> torch.Tensor:
    """
    Mask invalid actions based on current state.

    Returns boolean mask of valid actions.
    """
    mask = torch.ones(n_actions, dtype=torch.bool)

    if not state.has_position:
        # Can't do position-management actions without position
        mask[HOLD] = False
        mask[TIGHTEN_STOP] = False
        mask[LOOSEN_STOP] = False
        mask[TAKE_PARTIAL] = False
        mask[EXIT_ALL] = False
    else:
        # Can't enter when already in position
        mask[ENTER_LONG] = False

        # Can't tighten if already at minimum stop
        if state.current_stop_distance <= MIN_STOP:
            mask[TIGHTEN_STOP] = False

    return mask
```

### 3.4 Attention Mechanism

#### 3.4.1 Multi-Head Temporal Attention

The agent should learn WHAT historical information is relevant for the current decision.

```
                    Current State
                         |
                         v
+-------------------+----+----+-------------------+
|                   |         |                   |
v                   v         v                   v
+--------+     +--------+     +--------+     +--------+
| Micro  |     |  Meso  |     | Macro  |     |Position|
| History|     | History|     | History|     | State  |
+--------+     +--------+     +--------+     +--------+
    |               |              |              |
    v               v              v              |
+--------+     +--------+     +--------+         |
|Temporal|     |Temporal|     |Temporal|         |
|Attn    |     |Attn    |     |Attn    |         |
+--------+     +--------+     +--------+         |
    |               |              |              |
    +-------+-------+------+-------+--------------+
            |              |
            v              v
      +-----------+  +-----------+
      |  Feature  |  |  Feature  |
      |  Attn     |  |  Cross    |
      |  (what)   |  |  Attn     |
      +-----------+  +-----------+
            |              |
            +------+-------+
                   |
                   v
            +------------+
            |  Fused     |
            |  State     |
            +------------+
                   |
                   v
            +------------+
            |  Policy    |
            |  Network   |
            +------------+
```

#### 3.4.2 Attention Architecture Code

```python
class MarketAttention(nn.Module):
    """
    Multi-scale attention for market state processing.

    Learns:
    1. Which timesteps in history matter (temporal attention)
    2. Which features matter (feature attention)
    3. How to combine across timescales (cross attention)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_micro: int = 15,
        d_meso: int = 10,
        d_macro: int = 6,
        seq_micro: int = 60,
        seq_meso: int = 24,
        seq_macro: int = 7
    ):
        super().__init__()

        # Project each timescale to common dimension
        self.proj_micro = nn.Linear(d_micro, d_model)
        self.proj_meso = nn.Linear(d_meso, d_model)
        self.proj_macro = nn.Linear(d_macro, d_model)

        # Temporal attention for each timescale
        self.attn_micro = TemporalAttention(d_model, n_heads)
        self.attn_meso = TemporalAttention(d_model, n_heads)
        self.attn_macro = TemporalAttention(d_model, n_heads)

        # Cross-timescale attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )

        # Feature attention (what features to focus on)
        self.feature_attn = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )

        # Output projection
        self.output = nn.Linear(d_model, d_model)

    def forward(
        self,
        micro: torch.Tensor,    # (batch, seq_micro, d_micro)
        meso: torch.Tensor,     # (batch, seq_meso, d_meso)
        macro: torch.Tensor,    # (batch, seq_macro, d_macro)
        current: torch.Tensor   # (batch, d_current)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with attention weight tracking.

        Returns:
            fused: (batch, d_model) - fused state representation
            attention_weights: Dict with all attention patterns
        """
        # Project to common space
        micro_proj = self.proj_micro(micro)
        meso_proj = self.proj_meso(meso)
        macro_proj = self.proj_macro(macro)

        # Temporal attention within each scale
        micro_pooled, attn_micro = self.attn_micro(micro_proj)
        meso_pooled, attn_meso = self.attn_meso(meso_proj)
        macro_pooled, attn_macro = self.attn_macro(macro_proj)

        # Concatenate scale representations
        multi_scale = torch.stack([micro_pooled, meso_pooled, macro_pooled], dim=1)

        # Cross-scale attention
        fused, attn_cross = self.cross_attn(
            multi_scale, multi_scale, multi_scale
        )
        fused = fused.mean(dim=1)  # Pool across scales

        # Feature attention
        combined = torch.cat([micro_pooled, meso_pooled, macro_pooled], dim=-1)
        feature_weights = F.softmax(self.feature_attn(combined), dim=-1)
        fused = fused * feature_weights

        # Output
        output = self.output(fused)

        attention_weights = {
            'micro': attn_micro,      # Which recent minutes mattered
            'meso': attn_meso,        # Which recent hours mattered
            'macro': attn_macro,      # Which recent days mattered
            'cross': attn_cross,      # How scales interact
            'feature': feature_weights # Which features mattered
        }

        return output, attention_weights


class TemporalAttention(nn.Module):
    """Attention pooling over time dimension."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            pooled: (batch, d_model)
            attention_weights: (batch, seq_len)
        """
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)

        pooled, weights = self.attention(query, x, x)
        pooled = pooled.squeeze(1)
        weights = weights.squeeze(1)

        return pooled, weights
```

### 3.5 Reward Function Design (CRITICAL)

The reward function is the most important design decision. It must:
1. Balance profit vs. risk
2. Avoid degenerate strategies (always hold, overtrade)
3. Be dense enough for learning but not so noisy it misleads
4. Account for transaction costs

#### 3.5.1 Reward Components

```python
def compute_reward(
    state: MarketState,
    action: int,
    next_state: MarketState,
    trade_result: Optional[TradeResult] = None
) -> float:
    """
    Compute reward for state-action-next_state transition.

    Reward = R_pnl + R_risk + R_efficiency + R_behavior
    """
    reward = 0.0

    # =========================================
    # Component 1: P&L Reward (Primary Signal)
    # =========================================
    if trade_result is not None and trade_result.closed:
        # Trade completed - reward based on outcome
        net_return = trade_result.return_pct - TRANSACTION_COST

        if net_return > 0:
            # Winning trade: reward proportional to return
            reward += net_return * REWARD_SCALE_WIN
        else:
            # Losing trade: penalize, but less than symmetric
            # (we want the agent to take reasonable risks)
            reward += net_return * REWARD_SCALE_LOSS

    elif state.has_position:
        # In position but not closed - small unrealized P&L signal
        unrealized_return = next_state.position_return_pct - state.position_return_pct
        reward += unrealized_return * REWARD_SCALE_UNREALIZED

    # =========================================
    # Component 2: Risk-Adjusted Reward
    # =========================================
    if state.has_position:
        # Penalize large drawdowns from peak
        drawdown = state.highest_return_seen - state.position_return_pct
        if drawdown > DRAWDOWN_THRESHOLD:
            reward -= drawdown * REWARD_SCALE_DRAWDOWN

        # Reward for keeping volatility bounded
        position_volatility = compute_position_volatility(state)
        if position_volatility < TARGET_VOLATILITY:
            reward += REWARD_VOLATILITY_BONUS

    # =========================================
    # Component 3: Efficiency Rewards
    # =========================================

    # Penalize excessive actions (prevent overtrading)
    if action in [TIGHTEN_STOP, LOOSEN_STOP]:
        reward -= PENALTY_ADJUSTMENT

    # Penalize holding too long without profit
    if action == HOLD and state.time_in_position > MAX_HOLD_TIME:
        if state.position_return_pct < MIN_PROFIT_FOR_HOLD:
            reward -= PENALTY_STALE_POSITION

    # =========================================
    # Component 4: Behavior Shaping
    # =========================================

    # Reward for exiting before hitting stop loss
    if trade_result is not None and trade_result.exit_reason == 'manual':
        if trade_result.return_pct > -STOP_LOSS_PCT:
            reward += REWARD_SMART_EXIT

    # Reward for letting winners run
    if action == HOLD and state.position_return_pct > PROFIT_TARGET:
        reward += REWARD_PATIENCE

    return reward


# Reward hyperparameters
REWARD_SCALE_WIN = 10.0           # Amplify winning trades
REWARD_SCALE_LOSS = 5.0           # Losses hurt less (encourage risk-taking)
REWARD_SCALE_UNREALIZED = 0.1     # Small signal for paper gains
REWARD_SCALE_DRAWDOWN = 2.0       # Penalize giving back gains
REWARD_VOLATILITY_BONUS = 0.01    # Tiny bonus for smooth equity
PENALTY_ADJUSTMENT = 0.001        # Discourage fidgeting
PENALTY_STALE_POSITION = 0.1      # Don't hold forever
REWARD_SMART_EXIT = 1.0           # Reward for exiting above stop
REWARD_PATIENCE = 0.1             # Small bonus for holding winners
```

#### 3.5.2 Sharpe-Based Reward (Alternative)

```python
def compute_sharpe_reward(
    episode_returns: List[float],
    risk_free_rate: float = 0.0,
    annualization: float = np.sqrt(252 * 24)  # Hourly trading
) -> float:
    """
    Reward based on rolling Sharpe ratio.

    This directly optimizes for risk-adjusted returns.
    """
    if len(episode_returns) < 2:
        return 0.0

    returns = np.array(episode_returns)
    excess = returns - risk_free_rate

    mean_excess = np.mean(excess)
    std_excess = np.std(excess) + 1e-8

    sharpe = (mean_excess / std_excess) * annualization

    return np.tanh(sharpe)  # Bound to [-1, 1]
```

#### 3.5.3 Reward Function Variants to Test

| Variant | Description | Pros | Cons |
|---------|-------------|------|------|
| **Sparse** | Reward only on trade close | Clear signal | Slow learning |
| **Dense** | Reward every step | Fast learning | Noisy, may mislead |
| **Shaped** | Potential-based shaping | Best of both | Requires domain knowledge |
| **Sharpe** | Rolling Sharpe ratio | Risk-adjusted | High variance |
| **Hybrid** | Sparse + small dense shaping | Balanced | Hyperparameter sensitive |

**Recommendation**: Start with **Hybrid** - sparse rewards for trade outcomes + small dense rewards for risk management.

---

## 4. Training Pipeline

### 4.1 Episode Structure

```python
@dataclass
class EpisodeConfig:
    """Configuration for training episodes."""

    # Episode duration
    episode_length: int = 1440        # 1440 minutes = 24 hours
    step_size_minutes: int = 1        # Decision frequency

    # Data sampling
    symbols_per_episode: int = 1      # Focus on one symbol
    random_start: bool = True         # Random start time in data

    # Position constraints
    max_position_duration: int = 480  # 8 hours max hold
    max_trades_per_episode: int = 10  # Prevent overtrading

    # Termination conditions
    terminate_on_drawdown: float = 0.15  # -15% episode loss


class TradingEnvironment:
    """
    Custom Gym environment for crypto trading.
    """

    def __init__(
        self,
        db_path: str,
        config: EpisodeConfig,
        feature_config: FeatureConfig
    ):
        self.db = duckdb.connect(db_path, read_only=True)
        self.config = config
        self.feature_builder = FeatureBuilder(feature_config)

        # Gym spaces
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(STATE_DIM,), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        # Sample random symbol and start time
        self.symbol = self._sample_symbol()
        self.start_time = self._sample_start_time()
        self.current_step = 0

        # Reset position state
        self.position = None
        self.episode_trades = []
        self.episode_returns = []

        # Load data window
        self._load_data_window()

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return next state."""
        # Validate action
        valid_actions = self._get_valid_actions()
        if not valid_actions[action]:
            action = WAIT  # Default to wait if invalid

        # Execute action
        trade_result = self._execute_action(action)

        # Advance time
        self.current_step += 1

        # Get next observation
        next_obs = self._get_observation()

        # Compute reward
        reward = compute_reward(
            state=self.prev_state,
            action=action,
            next_state=self.current_state,
            trade_result=trade_result
        )

        # Check termination
        done = self._check_termination()

        # Info for logging
        info = {
            'symbol': self.symbol,
            'step': self.current_step,
            'position': self.position is not None,
            'episode_return': sum(self.episode_returns)
        }

        return next_obs, reward, done, info
```

### 4.2 Efficient Data Loading

With 90GB of data, we need efficient loading strategies.

```python
class EfficientDataLoader:
    """
    Memory-efficient data loading from DuckDB.

    Strategies:
    1. Lazy loading: Only load data for current episode
    2. Caching: Cache recently used symbols
    3. Chunking: Stream large queries
    4. Prefetching: Load next episode's data in background
    """

    def __init__(
        self,
        db_path: str,
        cache_size_mb: int = 1000,  # 1GB cache
        prefetch: bool = True
    ):
        self.db_path = db_path
        self.cache = LRUCache(maxsize_bytes=cache_size_mb * 1024 * 1024)
        self.prefetch = prefetch
        self.executor = ThreadPoolExecutor(max_workers=2)

    def load_episode_data(
        self,
        symbol: str,
        start_time: datetime,
        duration_minutes: int,
        lookback_minutes: int = 60
    ) -> EpisodeData:
        """
        Load all data needed for an episode.

        Args:
            symbol: Trading pair
            start_time: Episode start
            duration_minutes: Episode length
            lookback_minutes: Historical context needed

        Returns:
            EpisodeData with trades, ohlcv, order_book, features
        """
        cache_key = f"{symbol}:{start_time}:{duration_minutes}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Calculate time range
        data_start = start_time - timedelta(minutes=lookback_minutes)
        data_end = start_time + timedelta(minutes=duration_minutes)

        # Load in parallel
        futures = {
            'trades': self.executor.submit(
                self._load_trades, symbol, data_start, data_end
            ),
            'ohlcv': self.executor.submit(
                self._load_ohlcv, symbol, data_start, data_end
            ),
            'order_book': self.executor.submit(
                self._load_order_book, symbol, data_start, data_end
            ),
            'features': self.executor.submit(
                self._load_features, symbol, data_start, data_end
            )
        }

        # Collect results
        data = EpisodeData(
            symbol=symbol,
            start_time=start_time,
            trades=futures['trades'].result(),
            ohlcv=futures['ohlcv'].result(),
            order_book=futures['order_book'].result(),
            features=futures['features'].result()
        )

        self.cache[cache_key] = data

        return data

    def _load_trades(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Load trades with streaming for large results."""
        conn = duckdb.connect(self.db_path, read_only=True)

        query = f"""
            SELECT timestamp, price, size, side
            FROM trades
            WHERE symbol = '{symbol}'
            AND timestamp >= '{start}'
            AND timestamp <= '{end}'
            ORDER BY timestamp
        """

        return conn.execute(query).fetchdf()
```

### 4.3 Training Algorithm Selection

| Algorithm | Type | Pros | Cons | Recommendation |
|-----------|------|------|------|----------------|
| PPO | On-policy | Stable, sample efficient | Sensitive to hyperparams | **Phase 1** |
| SAC | Off-policy | Continuous actions, stable | Complex, more memory | Phase 2 |
| DQN | Off-policy | Simple, proven | Discrete only, overestimation | Baseline |
| A2C | On-policy | Simple, fast | High variance | Not recommended |
| TD3 | Off-policy | Handles overestimation | Continuous only | Phase 2 |

**Recommendation**: Start with **PPO** (Proximal Policy Optimization)
- Stable training
- Works well with discrete actions
- Good balance of sample efficiency and stability
- Well-supported in stable-baselines3

### 4.4 Preventing Overfitting

```python
class AntiOverfitTrainer:
    """
    Training strategies to prevent overfitting to historical data.
    """

    def __init__(self):
        # Data augmentation
        self.augmenters = [
            TimeShiftAugment(max_shift=5),      # Shift timestamps
            NoiseAugment(noise_std=0.01),        # Add feature noise
            DropoutAugment(drop_prob=0.1),       # Random feature dropout
        ]

        # Validation strategy
        self.validation = WalkForwardValidator(
            train_months=3,
            test_months=1,
            step_months=1
        )

    def train(self, agent, env, total_timesteps: int):
        """Train with overfitting prevention."""

        for fold in self.validation.get_folds():
            # Set environment to training period
            env.set_time_range(fold.train_start, fold.train_end)

            # Train
            agent.learn(
                total_timesteps=total_timesteps // self.validation.n_folds,
                callback=ValidationCallback(env, fold.test_start, fold.test_end)
            )

            # Evaluate on held-out period
            test_return = self.evaluate(agent, env, fold.test_start, fold.test_end)

            # Early stop if generalization degrades
            if test_return < EARLY_STOP_THRESHOLD:
                logger.warning(f"Early stopping: test return {test_return}")
                break


class WalkForwardValidator:
    """
    Walk-forward validation for time series.

    +-------+-------+-------+-------+-------+
    |Train  |Train  |Train  | Test  |       |
    +-------+-------+-------+-------+-------+
    |       |Train  |Train  |Train  | Test  |
    +-------+-------+-------+-------+-------+
    """

    def get_folds(self):
        # Implementation...
        pass
```

### 4.5 Curriculum Learning

```python
class CurriculumScheduler:
    """
    Start with easy patterns, gradually increase difficulty.

    Curriculum stages:
    1. High-volatility periods with clear trends
    2. Medium volatility, some noise
    3. Low volatility, choppy markets
    4. Full distribution of market conditions
    """

    def __init__(self):
        self.stages = [
            {
                'name': 'obvious_trends',
                'volatility_filter': lambda v: v > 0.05,  # High vol
                'duration_episodes': 1000
            },
            {
                'name': 'moderate_moves',
                'volatility_filter': lambda v: 0.02 < v < 0.05,
                'duration_episodes': 2000
            },
            {
                'name': 'choppy_markets',
                'volatility_filter': lambda v: v < 0.02,
                'duration_episodes': 2000
            },
            {
                'name': 'full_distribution',
                'volatility_filter': None,  # All data
                'duration_episodes': 5000
            }
        ]

    def get_episode_filter(self, episode_count: int):
        """Get data filter for current curriculum stage."""
        cumulative = 0
        for stage in self.stages:
            cumulative += stage['duration_episodes']
            if episode_count < cumulative:
                return stage['volatility_filter']
        return None  # Full distribution
```

---

## 5. Continuous Learning

### 5.1 Online Learning Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Live Market     |     |   RL Agent        |     |   Experience      |
|   Data Stream     | --> |   (Inference)     | --> |   Buffer          |
+-------------------+     +-------------------+     +-------------------+
                                  |                         |
                                  v                         v
                          +-------------+           +---------------+
                          |   Trading   |           |   Periodic    |
                          |   Actions   |           |   Retraining  |
                          +-------------+           +---------------+
                                                           |
                                                           v
                                                   +---------------+
                                                   |   Updated     |
                                                   |   Model       |
                                                   +---------------+
```

### 5.2 Preventing Catastrophic Forgetting

```python
class ContinualLearner:
    """
    Strategies to prevent forgetting old knowledge when learning new.
    """

    def __init__(self, base_model):
        self.model = base_model

        # Elastic Weight Consolidation (EWC)
        self.fisher_info = None
        self.old_params = None

        # Experience replay buffer with prioritization
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=100000,
            alpha=0.6,
            beta=0.4
        )

        # Historical performance tracking
        self.regime_performance = defaultdict(list)

    def update(self, new_experiences: List[Experience]):
        """
        Update model with new data while preserving old knowledge.
        """
        # Add new experiences to buffer
        for exp in new_experiences:
            self.replay_buffer.add(exp)

        # Sample mix of old and new experiences
        batch = self.replay_buffer.sample(
            batch_size=256,
            beta=self.get_beta()
        )

        # Compute standard loss
        policy_loss, value_loss = self.compute_loss(batch)

        # Add EWC regularization to prevent forgetting
        ewc_loss = self.compute_ewc_loss()

        total_loss = policy_loss + value_loss + EWC_LAMBDA * ewc_loss

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Elastic Weight Consolidation loss.

        Penalizes changes to parameters that were important for past tasks.
        """
        if self.fisher_info is None:
            return torch.tensor(0.0)

        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_info:
                fisher = self.fisher_info[name]
                old_param = self.old_params[name]
                loss += (fisher * (param - old_param).pow(2)).sum()

        return loss

    def consolidate(self):
        """
        After learning a regime, consolidate knowledge.

        Computes Fisher information to identify important weights.
        """
        self.fisher_info = {}
        self.old_params = {}

        for name, param in self.model.named_parameters():
            self.old_params[name] = param.clone().detach()
            self.fisher_info[name] = param.grad.pow(2).detach()
```

### 5.3 Regime Detection and Adaptation

```python
class RegimeDetector:
    """
    Detect market regime changes that may invalidate old learning.
    """

    def __init__(self):
        self.regime_features = [
            'btc_volatility_30d',
            'average_correlation',
            'volume_trend',
            'trend_strength'
        ]

        # Regime clustering model
        self.regime_model = GaussianMixture(n_components=4)

        # Performance tracking per regime
        self.regime_returns = defaultdict(list)

    def detect_regime(self, market_state: MarketState) -> int:
        """Classify current market regime."""
        features = self.extract_regime_features(market_state)
        regime = self.regime_model.predict([features])[0]
        return regime

    def check_regime_change(self, history: List[int], window: int = 100) -> bool:
        """Check if regime has changed significantly."""
        if len(history) < window:
            return False

        recent = history[-window//2:]
        older = history[-window:-window//2]

        recent_dist = Counter(recent)
        older_dist = Counter(older)

        # KL divergence between distributions
        divergence = self.kl_divergence(recent_dist, older_dist)

        return divergence > REGIME_CHANGE_THRESHOLD

    def should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        current_regime = self.detect_regime(self.latest_state)

        # Check if performance in current regime is degrading
        recent_returns = self.regime_returns[current_regime][-50:]

        if len(recent_returns) < 50:
            return False

        avg_return = np.mean(recent_returns)
        historical_avg = np.mean(self.regime_returns[current_regime])

        # Retrain if performance dropped significantly
        if avg_return < historical_avg - PERFORMANCE_DROP_THRESHOLD:
            logger.warning(f"Performance drop in regime {current_regime}: "
                          f"{avg_return:.4f} vs historical {historical_avg:.4f}")
            return True

        return False
```

### 5.4 A/B Testing New Models

```python
class ModelABTester:
    """
    Safely A/B test new model versions in production.
    """

    def __init__(
        self,
        current_model: nn.Module,
        candidate_model: nn.Module,
        test_allocation: float = 0.1  # 10% to candidate
    ):
        self.current = current_model
        self.candidate = candidate_model
        self.allocation = test_allocation

        self.current_trades = []
        self.candidate_trades = []

    def select_model(self, state: MarketState) -> Tuple[nn.Module, str]:
        """Select which model to use for this decision."""
        if random.random() < self.allocation:
            return self.candidate, 'candidate'
        return self.current, 'current'

    def record_trade(self, model_type: str, trade_result: TradeResult):
        """Record trade result for analysis."""
        if model_type == 'current':
            self.current_trades.append(trade_result)
        else:
            self.candidate_trades.append(trade_result)

    def evaluate_candidate(self) -> Dict:
        """Statistical test of candidate vs current."""
        if len(self.candidate_trades) < MIN_TRADES_FOR_TEST:
            return {'status': 'insufficient_data'}

        current_returns = [t.return_pct for t in self.current_trades]
        candidate_returns = [t.return_pct for t in self.candidate_trades]

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(candidate_returns, current_returns)

        # Metrics
        current_sharpe = self.compute_sharpe(current_returns)
        candidate_sharpe = self.compute_sharpe(candidate_returns)

        return {
            'status': 'evaluated',
            'current_sharpe': current_sharpe,
            'candidate_sharpe': candidate_sharpe,
            'improvement': candidate_sharpe - current_sharpe,
            'p_value': p_value,
            'significant': p_value < 0.05 and candidate_sharpe > current_sharpe,
            'recommendation': 'promote' if candidate_sharpe > current_sharpe and p_value < 0.05 else 'keep_current'
        }
```

---

## 6. Implementation Roadmap

### Phase 1: Minimal Viable Agent (4-6 weeks)

**Goal**: Working RL agent that can make basic trading decisions

```
Week 1-2: Environment Setup
- [ ] Implement TradingEnvironment with Gym interface
- [ ] Basic feature extraction (OHLCV-based)
- [ ] Simple state representation (no attention yet)
- [ ] Discrete action space (7 actions)

Week 3-4: Basic Agent
- [ ] PPO agent using stable-baselines3
- [ ] Simple reward function (P&L only)
- [ ] Episode-based training loop
- [ ] Basic logging and visualization

Week 5-6: Baseline Evaluation
- [ ] Walk-forward backtesting
- [ ] Compare to random baseline
- [ ] Compare to buy-and-hold
- [ ] Identify failure modes
```

**Deliverables**:
- `src/rl/environment.py` - Gym environment
- `src/rl/features.py` - Feature extraction
- `src/rl/agent.py` - PPO agent wrapper
- `scripts/train_rl_baseline.py` - Training script
- `notebooks/rl_baseline_analysis.ipynb` - Evaluation

### Phase 2: Attention and State Enhancement (4-6 weeks)

**Goal**: Add attention mechanism and richer state representation

```
Week 7-8: Multi-Scale Features
- [ ] Implement microstructure features (trade flow, order book)
- [ ] Multi-timescale history (micro/meso/macro)
- [ ] Feature normalization and preprocessing

Week 9-10: Attention Architecture
- [ ] Temporal attention per timescale
- [ ] Cross-timescale attention
- [ ] Feature attention
- [ ] Attention weight logging for interpretability

Week 11-12: Enhanced Rewards
- [ ] Implement risk-adjusted rewards
- [ ] Add behavior shaping components
- [ ] Tune reward hyperparameters
- [ ] Compare reward variants
```

**Deliverables**:
- `src/rl/attention.py` - Attention modules
- `src/rl/rewards.py` - Reward functions
- `src/rl/state.py` - Enhanced state representation
- Updated training scripts
- Attention visualization dashboard

### Phase 3: Continuous Learning (4-6 weeks)

**Goal**: System that adapts to changing market conditions

```
Week 13-14: Online Learning Infrastructure
- [ ] Experience replay buffer
- [ ] Periodic retraining pipeline
- [ ] Model checkpointing and versioning

Week 15-16: Anti-Forgetting Mechanisms
- [ ] Implement EWC (Elastic Weight Consolidation)
- [ ] Regime detection
- [ ] Performance monitoring per regime

Week 17-18: Production Pipeline
- [ ] A/B testing framework
- [ ] Automated model promotion
- [ ] Monitoring and alerting
- [ ] Integration with existing paper trading
```

**Deliverables**:
- `src/rl/continual.py` - Continuous learning components
- `src/rl/regime.py` - Regime detection
- `src/rl/ab_test.py` - A/B testing
- `scripts/run_rl_live.py` - Live trading integration
- Monitoring dashboard

### Phase 4: Advanced Improvements (Ongoing)

```
- Hybrid action space (discrete type + continuous params)
- Multi-agent (different agents for different regimes)
- Meta-learning for fast adaptation
- Ensemble of RL agents
- Integration with LLM for news interpretation
```

---

## 7. Technical Recommendations

### 7.1 Library Selection

```python
# requirements_rl.txt

# Core RL
stable-baselines3==2.2.1
gymnasium==0.29.1

# Deep Learning
torch>=2.1.0
torchvision>=0.16.0

# Data
duckdb>=0.9.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0

# Monitoring
wandb>=0.16.0
tensorboard>=2.15.0

# Utilities
loguru>=0.7.0
hydra-core>=1.3.0
```

### 7.2 Hardware Considerations

For M4 Mac Mini with 16GB RAM:
- **Training**: Batch size 32-64, gradient accumulation for effective larger batches
- **Data Loading**: Stream from DuckDB, don't load full dataset
- **MPS Acceleration**: Use `device='mps'` for ~2-3x speedup vs CPU
- **Memory**: Keep experience buffer < 2GB, use lazy episode loading

### 7.3 Key Hyperparameters to Tune

| Parameter | Starting Value | Range to Explore |
|-----------|---------------|------------------|
| Learning rate | 3e-4 | 1e-5 to 1e-3 |
| Discount (gamma) | 0.99 | 0.95 to 0.999 |
| GAE lambda | 0.95 | 0.9 to 0.99 |
| Clip ratio (PPO) | 0.2 | 0.1 to 0.3 |
| Entropy coefficient | 0.01 | 0.001 to 0.1 |
| Episode length | 1440 (24h) | 720 to 2880 |
| d_model (attention) | 128 | 64 to 256 |
| n_attention_heads | 8 | 4 to 16 |

---

## 8. Risks and Challenges

### 8.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Reward hacking | High | High | Multiple reward signals, behavior constraints |
| Overfitting to history | High | High | Walk-forward validation, data augmentation |
| Degenerate policies | Medium | High | Action masking, entropy bonus, reward shaping |
| Catastrophic forgetting | Medium | Medium | EWC, experience replay, regime awareness |
| Slow training | Medium | Low | Curriculum learning, efficient data loading |

### 8.2 Market Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| Regime change | Model trained on bull market fails in bear | Regime detection, continuous adaptation |
| Black swan events | Unprecedented market conditions | Position limits, drawdown stops |
| Execution slippage | Real execution differs from backtest | Realistic simulation, conservative estimates |
| Overconfidence | High backtest performance doesn't transfer | Out-of-sample validation, paper trading |

### 8.3 Honest Assessment

**This will be hard.** RL for trading is notoriously difficult because:

1. **Non-stationary environment**: Markets change constantly
2. **Partial observability**: We can't see all market participants
3. **Delayed and sparse rewards**: Trade outcomes take time
4. **High noise**: Signal-to-noise ratio is very low
5. **Adversarial**: Other traders adapt to your strategy

**However**, the existing rule-based system (Breakout Hunter v5.3 with 94.6% win rate) provides a strong baseline. The RL system should aim to:
1. Learn WHEN to apply which rules
2. Optimize position management dynamically
3. Adapt to regime changes automatically
4. Discover patterns the rules miss

Rather than replacing the rule-based system, the RL agent can **complement** it by handling the continuous decisions (stop adjustment, partial exits, timing) while the rules handle signal generation.

---

## 9. Success Metrics

### 9.1 Training Metrics

- **Episode return distribution**: Should be right-skewed (more wins than losses)
- **Policy entropy**: Should decrease over time but not collapse to zero
- **Value loss**: Should decrease and stabilize
- **Attention weights**: Should show meaningful patterns (not uniform)

### 9.2 Evaluation Metrics

| Metric | Target (Phase 1) | Target (Phase 3) |
|--------|-----------------|-----------------|
| Sharpe Ratio | > 1.0 | > 2.0 |
| Win Rate | > 45% | > 55% |
| Max Drawdown | < 20% | < 15% |
| Profit Factor | > 1.2 | > 1.5 |
| Trades per Day | 1-5 | 2-10 |

### 9.3 Comparison Baselines

1. **Random agent**: Should dramatically outperform
2. **Buy-and-hold**: Should outperform on risk-adjusted basis
3. **Rule-based (v5.3)**: Should match or exceed with less parameter tuning
4. **XGBoost classifier**: Should show better position management

---

## 10. Conclusion

Building an RL-based trading system that "navigates like whitewater rapids" is ambitious but achievable. The key insights from this analysis:

1. **Use existing data infrastructure**: 90GB of multi-resolution data provides rich state information
2. **Learn from rule-based success**: The v5.3 strategy's patterns inform reward design and action space
3. **Attention is key**: Let the model learn what to focus on across timescales
4. **Risk-adjusted rewards**: Optimize Sharpe, not raw returns
5. **Continuous adaptation**: Build regime detection and anti-forgetting from the start
6. **Start simple**: Phase 1 should prove the concept before adding complexity

The proposed architecture balances sophistication with practicality, providing a clear path from minimal viable agent to production-ready continuous learning system.

---

## Appendix A: Code Templates

### A.1 Environment Template

See `/Users/bz/Pythia2/src/rl/environment.py` (to be created)

### A.2 Feature Extraction Template

See `/Users/bz/Pythia2/src/rl/features.py` (to be created)

### A.3 Reward Function Template

See section 3.5 of this document

---

## Appendix B: References

1. **Deep Reinforcement Learning for Automated Stock Trading** - Xiong et al., 2018
2. **Proximal Policy Optimization Algorithms** - Schulman et al., 2017
3. **Overcoming Catastrophic Forgetting in Neural Networks** - Kirkpatrick et al., 2017
4. **Attention Is All You Need** - Vaswani et al., 2017
5. **Advances in Financial Machine Learning** - Lopez de Prado, 2018

---

*Document generated: March 17, 2026*
*Last updated: March 17, 2026*
