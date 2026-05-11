# RL Training v5 Proposal

**Date:** 2026-03-18
**Status:** Proposal
**Baseline:** v4 final reward = 10.8 (broke the 9.5 plateau)

---

## Executive Summary

v4 achieved a breakthrough by using additive bonuses on top of the proven hybrid reward, reaching 10.8 final reward after 500K timesteps. This proposal outlines v5 strategies to push rewards to 12+ while improving real trading metrics (Sharpe ratio, win rate on quality entries).

**Key v5 strategies:**
1. Bonus tuning based on v4 analysis
2. Temporal architecture (LSTM or Transformer)
3. Exit quality bonuses (not just entry quality)
4. Risk-adjusted reward components
5. Curriculum learning for spike intensity

---

## v4 Analysis: What Worked

### Bonus Effectiveness Review

Based on the v4 `SpikeQualityBonusConfig`:

| Bonus | Current Value | Estimated Frequency | Episode Impact |
|-------|---------------|---------------------|----------------|
| `bonus_fresh_spike_entry` | 0.3 * freshness | ~3 entries | +0.6 |
| `bonus_patient_entry` | 0.2 | ~4 entries | +0.8 |
| `bonus_first_trade_on_spike` | 0.4 | ~2 entries | +0.8 |
| `large_move_multiplier` | 1.0 * return | ~1 trade | +0.03 |
| `huge_move_multiplier` | 1.5 * return | ~0.3 trades | +0.02 |
| `quality_multiplier` (freshness) | 0.2 * return | ~5 wins | +0.1 |
| `first_trade_bonus` | 0.1 * return | ~2 wins | +0.02 |

**Estimated total bonus per episode:** +2.3 to +2.5
**Observed improvement over hybrid:** 10.8 - 9.5 = +1.3

The gap between estimated (+2.3) and observed (+1.3) suggests bonuses are not being triggered as often as expected, likely because:
1. Agent may not be finding fresh spikes frequently enough
2. Patient entry threshold (0.6 = 36 min) may be too strict
3. Large moves (>3%) are rarer than expected

---

## v5 Strategy 1: Bonus Tuning

### 1.1 Lower Thresholds to Increase Trigger Rate

```python
SpikeQualityBonusConfig(
    # ENTRY TIMING - lower thresholds for more frequent bonuses
    bonus_fresh_spike_entry=0.35,      # Increased from 0.3
    fresh_spike_threshold=0.6,          # Lowered from 0.7 (trigger earlier)

    bonus_patient_entry=0.25,           # Increased from 0.2
    patient_threshold=0.5,              # Lowered from 0.6 (~30 min vs 36 min)

    bonus_first_trade_on_spike=0.5,     # Increased from 0.4 (high-value action)

    # TRADE QUALITY - lower thresholds, higher multipliers
    large_move_threshold=0.02,          # Lowered from 0.03 (2% vs 3%)
    large_move_multiplier=1.5,          # Increased from 1.0
    huge_move_threshold=0.04,           # Lowered from 0.05 (4% vs 5%)
    huge_move_multiplier=2.0,           # Increased from 1.5

    # QUALITY MULTIPLIER - increase impact
    enable_quality_multiplier=True,
    max_freshness_bonus=0.3,            # Increased from 0.2 (up to 30% extra)
    first_trade_bonus=0.15,             # Increased from 0.1
)
```

**Expected impact:** +0.5 to +1.0 additional reward per episode

### 1.2 New Bonus: Streak Bonus

Reward consecutive winning trades (momentum strategy):

```python
streak_bonus_per_win: float = 0.1       # +0.1 for 2nd consecutive win
streak_bonus_max: float = 0.5           # Cap at +0.5 for 5+ streak
```

**Implementation:**
```python
# Track win streak
if trade_result and trade_result.return_pct > 0:
    self._win_streak += 1
    streak_bonus = min(self._win_streak - 1, 5) * cfg.streak_bonus_per_win
    bonus += streak_bonus
elif trade_result and trade_result.return_pct <= 0:
    self._win_streak = 0
```

---

## v5 Strategy 2: Exit Quality Bonuses

v4 focuses on entry quality but exit timing is equally important. Add bonuses for smart exits.

### 2.1 Exit Timing Bonuses

```python
@dataclass
class ExitQualityBonusConfig:
    # Reward locking in profits before reversal
    bonus_profit_lock: float = 0.2          # Bonus for exiting >2% gain
    profit_lock_threshold: float = 0.02     # 2% unrealized triggers

    # Reward exiting near local high
    bonus_near_high_exit: float = 0.15      # Exit within 1% of position high
    near_high_threshold: float = 0.01       # 1% of entry price

    # Reward quick exit on failed trade (loss minimization)
    bonus_quick_loss_exit: float = 0.1      # Exit loss <1% early
    quick_loss_threshold: float = 0.01
    quick_loss_time_threshold: int = 30     # Within 30 minutes

    # Reward trailing stop discipline
    bonus_trailing_stop_exit: float = 0.1   # Exit via tightened stop
```

**Implementation:**
```python
def _calculate_exit_bonuses(
    self,
    trade_result: TradeResult,
    position_info: dict,
) -> float:
    bonus = 0.0
    cfg = self.exit_config

    if trade_result is None:
        return 0.0

    net_return = trade_result.return_pct
    duration_minutes = (trade_result.exit_time - trade_result.entry_time).total_seconds() / 60

    # Profit lock bonus: exited with >2% gain
    if net_return > cfg.profit_lock_threshold:
        bonus += cfg.bonus_profit_lock

    # Near-high exit: captured most of the move
    highest_return = position_info.get('highest_return', net_return)
    if net_return > 0 and (highest_return - net_return) < cfg.near_high_threshold:
        bonus += cfg.bonus_near_high_exit

    # Quick loss exit: minimized damage on bad trade
    if net_return < 0 and abs(net_return) < cfg.quick_loss_threshold:
        if duration_minutes < cfg.quick_loss_time_threshold:
            bonus += cfg.bonus_quick_loss_exit

    # Trailing stop discipline
    if trade_result.exit_reason == 'stop_loss' and net_return > 0:
        # Stopped out at profit = good trailing stop management
        bonus += cfg.bonus_trailing_stop_exit

    return bonus
```

**Expected impact:** +0.3 to +0.5 per episode

---

## v5 Strategy 3: Temporal Architecture (LSTM/Transformer)

### 3.1 Problem with Current Architecture

The current MLP policy sees only the current timestep features. While multi-timeframe features provide some historical context, the agent cannot learn temporal patterns like:
- Spike acceleration (velocity of volume increase)
- Pre-spike buildup patterns (warning signs before breakout)
- Market regime transitions (trending vs ranging)

### 3.2 LSTM Feature Extractor

Replace `TradingFeaturesExtractor` with a recurrent variant:

```python
class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for temporal pattern learning.

    Maintains hidden state across timesteps within an episode.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        input_dim = int(np.prod(observation_space.shape))

        # Input projection
        self.input_proj = nn.Linear(input_dim, lstm_hidden)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden, features_dim),
            nn.Tanh(),
            nn.LayerNorm(features_dim),
        )

        # Hidden state (reset per episode)
        self.hidden = None

    def reset_hidden(self, batch_size: int = 1):
        """Reset hidden state at episode start."""
        device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(2, batch_size, 128, device=device),
            torch.zeros(2, batch_size, 128, device=device),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Project input
        x = self.input_proj(observations)

        # Handle batched vs sequential input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # LSTM forward
        if self.hidden is None:
            self.reset_hidden(x.size(0))

        lstm_out, self.hidden = self.lstm(x, self.hidden)

        # Detach hidden to prevent BPTT explosion
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        # Output projection
        features = self.output_proj(lstm_out[:, -1, :])

        return features
```

### 3.3 Attention-Based Alternative

Use the existing `MarketAttention` module from `attention.py`:

```python
# In agent.py, modify policy_kwargs
policy_kwargs = {
    "features_extractor_class": AttentionFeaturesExtractor,
    "features_extractor_kwargs": {
        "features_dim": 128,
        "attention_config": AttentionConfig(
            d_model=128,
            n_heads=8,
            seq_micro=60,   # 60 min of micro history
            seq_meso=24,    # 24 hours of meso history
        ),
    },
}
```

**Challenge:** Requires restructuring observation space to include historical sequences. Current environment provides single-timestep observations.

### 3.4 Implementation Plan for Temporal Architecture

**Phase 1: Sequence Buffer**
```python
class SequenceBuffer:
    """Maintains rolling history for LSTM/Attention input."""

    def __init__(self, sequence_length: int = 60, feature_dim: int = 56):
        self.sequence_length = sequence_length
        self.buffer = np.zeros((sequence_length, feature_dim), dtype=np.float32)

    def push(self, observation: np.ndarray):
        """Add new observation, shift buffer."""
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = observation

    def get_sequence(self) -> np.ndarray:
        """Get full sequence for model input."""
        return self.buffer.copy()

    def reset(self):
        """Clear buffer at episode start."""
        self.buffer = np.zeros_like(self.buffer)
```

**Phase 2: Modified Environment**
- Add sequence buffer to environment
- Expand observation space to include history
- Reset buffer on episode reset

**Expected impact:** +0.5 to +1.5 from better pattern recognition (speculative)

---

## v5 Strategy 4: Risk-Adjusted Reward Components

### 4.1 Rolling Sharpe Bonus

Add a Sharpe-based component to reward consistent, risk-adjusted returns:

```python
def _calculate_sharpe_bonus(self, episode_returns: List[float]) -> float:
    """
    Bonus based on rolling Sharpe ratio.

    Encourages consistent profitable trading over wild swings.
    """
    if len(episode_returns) < 5:
        return 0.0

    returns = np.array(episode_returns[-20:])  # Last 20 trades

    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-8

    # Simple Sharpe (no risk-free rate for simplicity)
    sharpe = mean_return / std_return

    # Tanh to bound, scale to reasonable bonus
    sharpe_bonus = np.tanh(sharpe * 2.0) * 0.3  # Max +/- 0.3

    return max(0.0, sharpe_bonus)  # Only positive bonus
```

### 4.2 Drawdown Penalty Exemption

v4 inherits drawdown penalties from hybrid. Consider exempting trades that:
1. Are the first trade on a fresh spike (high conviction)
2. Have been profitable within the last 5 trades

```python
def _should_exempt_drawdown_penalty(
    self,
    is_first_trade_on_spike: bool,
    recent_win_rate: float,
) -> bool:
    """Exempt high-conviction trades from drawdown penalty."""
    return is_first_trade_on_spike or recent_win_rate > 0.6
```

### 4.3 Win Rate Momentum Bonus

```python
def _calculate_winrate_bonus(self, trades: List[TradeResult]) -> float:
    """Bonus for maintaining high win rate."""
    if len(trades) < 5:
        return 0.0

    recent_trades = trades[-10:]
    wins = sum(1 for t in recent_trades if t.return_pct > 0)
    win_rate = wins / len(recent_trades)

    if win_rate > 0.6:  # 60%+ win rate
        return (win_rate - 0.6) * 1.0  # Up to +0.4 bonus
    return 0.0
```

**Expected impact:** +0.3 from Sharpe bonus, +0.2 from win rate bonus

---

## v5 Strategy 5: Curriculum Learning

### 5.1 Spike Intensity Curriculum

Train the agent progressively on harder spike scenarios:

**Stage 1: Easy spikes (0-250K steps)**
- Volume threshold: 2x (lower than normal 3x)
- Longer spike duration (90 min decay vs 60 min)
- More frequent fresh spike opportunities

**Stage 2: Normal spikes (250K-500K steps)**
- Volume threshold: 3x (standard)
- 60 min spike decay
- Normal bonus thresholds

**Stage 3: Hard spikes (500K+ steps)**
- Volume threshold: 4x (only trade big moves)
- 45 min spike decay (faster staleness)
- Tighter bonus thresholds

```python
class CurriculumScheduler:
    """Adjusts environment difficulty based on training progress."""

    def __init__(self):
        self.stage = 0
        self.stage_thresholds = [250_000, 500_000]

    def update(self, timesteps: int) -> dict:
        """Return environment config updates for current stage."""
        if timesteps < self.stage_thresholds[0]:
            return {
                'event_volume_threshold': 2.0,
                'spike_decay_minutes': 90,
            }
        elif timesteps < self.stage_thresholds[1]:
            return {
                'event_volume_threshold': 3.0,
                'spike_decay_minutes': 60,
            }
        else:
            return {
                'event_volume_threshold': 4.0,
                'spike_decay_minutes': 45,
            }
```

---

## v5 Strategy 6: Training Improvements

### 6.1 Longer Training

v4 used 500K timesteps. Recommendations for v5:

| Variant | Timesteps | Expected Duration | Purpose |
|---------|-----------|-------------------|---------|
| v5-quick | 500K | ~20 min | Validate bonus tuning |
| v5-standard | 1M | ~40 min | Full training |
| v5-extended | 2M | ~80 min | Convergence test |

### 6.2 Hyperparameter Adjustments

```python
PPO(
    learning_rate=2e-4,              # Slightly lower (was 3e-4)
    n_steps=4096,                    # Double (was 2048)
    batch_size=512,                  # Double (was 256)
    n_epochs=15,                     # Increase (was 10)
    gamma=0.995,                     # Slightly higher (was 0.99)
    gae_lambda=0.95,                 # Keep same
    clip_range=0.15,                 # Tighter (was 0.2)
    ent_coef=0.015,                  # Lower entropy (was 0.02)
    vf_coef=0.5,                     # Keep same
)
```

### 6.3 Learning Rate Schedule

```python
from stable_baselines3.common.callbacks import BaseCallback

class LearningRateScheduler(BaseCallback):
    """Linear decay with warmup."""

    def __init__(self, warmup_steps: int = 50_000, initial_lr: float = 1e-5):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = 3e-4
        self.final_lr = 1e-5

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.model.total_timesteps

        if self.num_timesteps < self.warmup_steps:
            # Warmup: linear increase
            warmup_progress = self.num_timesteps / self.warmup_steps
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * warmup_progress
        else:
            # Decay: linear decrease
            decay_progress = (self.num_timesteps - self.warmup_steps) / (
                self.model.total_timesteps - self.warmup_steps
            )
            lr = self.target_lr - (self.target_lr - self.final_lr) * decay_progress

        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = lr

        return True
```

---

## Implementation Priority

### High Priority (v5.0 - Quick Wins)

1. **Bonus tuning** (Strategy 1)
   - Lower thresholds
   - Increase multipliers
   - Add streak bonus
   - Expected: +0.5 to +1.0

2. **Exit quality bonuses** (Strategy 2)
   - Profit lock bonus
   - Near-high exit bonus
   - Expected: +0.3 to +0.5

**v5.0 Target Reward: 11.5-12.5**

### Medium Priority (v5.1 - Architecture)

3. **LSTM feature extractor** (Strategy 3)
   - Implement sequence buffer
   - Add recurrent layer
   - Expected: +0.5 to +1.0 (speculative)

4. **Risk-adjusted bonuses** (Strategy 4)
   - Rolling Sharpe bonus
   - Win rate bonus
   - Expected: +0.3 to +0.5

**v5.1 Target Reward: 12.5-13.5**

### Lower Priority (v5.2 - Advanced)

5. **Curriculum learning** (Strategy 5)
   - Spike difficulty progression
   - Expected: Better generalization, not necessarily higher reward

6. **Full attention policy** (Strategy 3.3)
   - Use MarketAttention module
   - Requires observation space restructuring

---

## Experiment Plan

### v5.0 Experiments

```bash
# Experiment 1: Bonus tuning only
python scripts/train_rl_agent.py \
    --reward-type spike_quality_bonus \
    --total-timesteps 500000 \
    --experiment-name v5.0_bonus_tuning

# Experiment 2: Bonus tuning + exit bonuses
python scripts/train_rl_agent.py \
    --reward-type spike_quality_bonus_v5 \
    --total-timesteps 500000 \
    --experiment-name v5.0_with_exit_bonuses

# Experiment 3: Extended training (best config from 1/2)
python scripts/train_rl_agent.py \
    --reward-type spike_quality_bonus_v5 \
    --total-timesteps 1000000 \
    --experiment-name v5.0_extended
```

### Success Criteria

| Metric | v4 Baseline | v5.0 Target | v5.1 Target |
|--------|-------------|-------------|-------------|
| Final Reward | 10.8 | 12.0 | 13.0 |
| Win Rate | ~55% | ~58% | ~60% |
| Quality Entry Rate | ~40% | ~50% | ~60% |
| Avg Return per Trade | ~0.5% | ~0.6% | ~0.7% |

---

## Files to Modify/Create

### v5.0 Changes

1. **`/Users/bz/Pythia2/src/rl/rewards.py`**
   - Update `SpikeQualityBonusConfig` with tuned values
   - Add `ExitQualityBonusConfig` dataclass
   - Add exit bonus calculation to `SpikeQualityBonusCalculator`
   - Add streak tracking

2. **`/Users/bz/Pythia2/scripts/train_rl_agent.py`**
   - Add v5 configuration options
   - Add learning rate scheduler callback

### v5.1 Changes

3. **`/Users/bz/Pythia2/src/rl/agent.py`**
   - Add `LSTMFeaturesExtractor` class
   - Add sequence buffer management

4. **`/Users/bz/Pythia2/src/rl/environment.py`**
   - Add sequence buffer to environment
   - Expand observation space for sequences

---

## Recommended v5.0 Configuration

```python
# /Users/bz/Pythia2/src/rl/rewards.py

@dataclass
class SpikeQualityBonusConfigV5:
    """v5: Tuned bonuses + exit quality + streak rewards."""

    # =========================================
    # ENTRY BONUSES (tuned from v4)
    # =========================================
    bonus_fresh_spike_entry: float = 0.35
    fresh_spike_threshold: float = 0.6

    bonus_patient_entry: float = 0.25
    patient_threshold: float = 0.5

    bonus_first_trade_on_spike: float = 0.5

    # =========================================
    # TRADE QUALITY BONUSES (tuned from v4)
    # =========================================
    large_move_threshold: float = 0.02
    large_move_multiplier: float = 1.5
    huge_move_threshold: float = 0.04
    huge_move_multiplier: float = 2.0

    enable_quality_multiplier: bool = True
    max_freshness_bonus: float = 0.3
    first_trade_bonus: float = 0.15

    # =========================================
    # EXIT QUALITY BONUSES (NEW in v5)
    # =========================================
    enable_exit_bonuses: bool = True
    bonus_profit_lock: float = 0.2
    profit_lock_threshold: float = 0.02

    bonus_near_high_exit: float = 0.15
    near_high_threshold: float = 0.01

    bonus_quick_loss_exit: float = 0.1
    quick_loss_threshold: float = 0.01
    quick_loss_time_threshold: int = 30

    # =========================================
    # STREAK BONUSES (NEW in v5)
    # =========================================
    enable_streak_bonus: bool = True
    streak_bonus_per_win: float = 0.1
    streak_bonus_max: float = 0.5

    # =========================================
    # RISK-ADJUSTED BONUSES (NEW in v5)
    # =========================================
    enable_sharpe_bonus: bool = True
    sharpe_window: int = 20
    sharpe_bonus_scale: float = 0.3

    enable_winrate_bonus: bool = True
    winrate_threshold: float = 0.6
    winrate_bonus_scale: float = 1.0
```

---

## Conclusion

v4 demonstrated that the additive bonus approach works. v5 builds on this foundation with:

1. **More aggressive bonus tuning** - Lower thresholds, higher values
2. **Exit quality rewards** - Complete the entry-to-exit quality incentive loop
3. **Streak and consistency bonuses** - Reward disciplined, repeatable trading
4. **Optional temporal architecture** - Unlock pattern recognition across time

The most impactful quick wins are likely:
- Lowering `fresh_spike_threshold` from 0.7 to 0.6
- Adding exit bonuses (especially `bonus_profit_lock`)
- Adding `streak_bonus_per_win`

These changes require minimal code modifications and should provide 1-2 additional reward points, pushing toward the 12-13 target.

---

## Appendix: v4 to v5 Comparison

| Component | v4 | v5 (Proposed) |
|-----------|----|----|
| `fresh_spike_threshold` | 0.7 | 0.6 |
| `patient_threshold` | 0.6 | 0.5 |
| `bonus_first_trade_on_spike` | 0.4 | 0.5 |
| `large_move_threshold` | 0.03 | 0.02 |
| `large_move_multiplier` | 1.0 | 1.5 |
| `max_freshness_bonus` | 0.2 | 0.3 |
| Exit bonuses | None | Profit lock, near-high, quick-loss |
| Streak bonus | None | 0.1 per win, max 0.5 |
| Sharpe bonus | None | Up to 0.3 |
| Win rate bonus | None | Up to 0.4 |
| Architecture | MLP | MLP (LSTM optional) |

**Estimated v5.0 reward: 12.0-12.5**
**Estimated v5.1 reward (with LSTM): 13.0+**
