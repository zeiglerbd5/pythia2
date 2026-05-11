# RL Training v4 Implementation Plan

**Date:** 2026-03-18
**Author:** Claude (RL Architecture Analysis)

---

## Executive Summary

The v3 `SpikeAwareRewardCalculator` failed catastrophically because it was **penalty-dominated**. The reward function accumulated so many penalties that all actions became bad, pushing rewards to -130 to -143 range. The agent learned to minimize activity rather than trade selectively.

**Solution:** v4 will use the proven hybrid reward as a **stable positive baseline**, then add **small, targeted bonuses** for spike-quality trading. No penalties beyond what hybrid already provides.

---

## Problem Analysis

### v3 Failure Root Cause

Looking at `SpikeAwareRewardConfig`, the penalties compound catastrophically:

| Penalty Source | Value | Frequency | Episode Impact |
|----------------|-------|-----------|----------------|
| `penalty_already_traded_spike` | -0.3 | Every re-entry | ~3-5 per episode = -0.9 to -1.5 |
| `penalty_rapid_reentry` | -0.15 | Most entries | ~5 per episode = -0.75 |
| `penalty_overtrading` | -0.1 | When >3 trades/hr | ~2 per episode = -0.2 |
| `penalty_stop_loss` | -0.3 | On stops | ~3 per episode = -0.9 |
| `drawdown_penalty_weight` | -1.5 * dd | Continuous | ~-5.0 over episode |
| `penalty_stale_position` | -0.05 | Per step when stale | ~50 steps = -2.5 |

**Total per-episode penalty: approximately -10 to -15 from penalties alone**

Meanwhile, the base P&L contribution:
- With `reward_scale_win=15` and `reward_scale_loss=12`
- Average trade return after fees: ~0.3%
- Average P&L reward per episode: ~4-6 (similar to hybrid)

**Net result: 4-6 base reward - 10-15 penalties = -5 to -10 per episode**

The 1440-minute episode length (3x longer than v1/v2's 480) amplified all time-based penalties, making things even worse.

### Why Hybrid Works

The hybrid reward in v1/v2:
- `reward_scale_win=10.0`, `reward_scale_loss=5.0` (asymmetric, encourages trading)
- Small penalties for fidgeting (`penalty_adjustment=0.001`)
- Moderate drawdown penalty (`reward_scale_drawdown=2.0`)
- Patience bonus for holding winners (`reward_patience=0.1`)

Net effect: **Positive baseline around 8-10 per episode**

### The Plateau Problem

v1 and v2 plateau at ~9.5 because:
1. Agent learns to "trade every spike" indiscriminately
2. No differentiation between fresh spike (high probability) vs stale spike (low probability)
3. Equal reward for first trade on spike vs third trade on same spike
4. The agent found a local optimum: "trade whenever spike features are elevated"

---

## v4 Design: Additive Bonus Approach

### Core Principle

**DO NOT modify the base reward calculation. ADD bonuses on top.**

```
v4_reward = hybrid_reward + spike_quality_bonus
```

Where `spike_quality_bonus >= 0` always (no additional penalties).

### Bonus Categories

#### 1. Fresh Spike Entry Bonus

When entering a long position during a fresh spike:
```python
if action == ENTER_LONG and is_in_spike and spike_freshness > 0.7:
    bonus += 0.3 * spike_freshness  # Max +0.3 for very fresh
```

**Rationale:** Reward the agent for timing entries well. Fresh spikes (first 18 minutes) have higher probability of continuation.

#### 2. Patient Trading Bonus

When entering after waiting sufficient time:
```python
if action == ENTER_LONG and minutes_since_last_trade > 0.6:  # >36 minutes
    bonus += 0.2
```

**Rationale:** Encourage selectivity without penalizing rapid trades. Agent learns patience is rewarded, not that speed is punished.

#### 3. First-Entry-on-Spike Bonus

When this is the first trade on a new spike:
```python
if action == ENTER_LONG and not already_traded_this_spike and is_in_spike:
    bonus += 0.4
```

**Rationale:** The highest probability trade is the first one on a fresh spike. Make this the most rewarded entry type.

#### 4. Large Move Capture Bonus

When closing a winning trade that caught a significant move:
```python
if trade_result and trade_result.return_pct > 0:
    if return_pct > 0.03:  # 3%+
        bonus += return_pct * 1.0  # +0.03 to +0.05
    if return_pct > 0.05:  # 5%+
        bonus += return_pct * 1.5  # Additional +0.075+
```

**Rationale:** Breaking the plateau requires incentivizing BIG wins, not just any wins.

#### 5. Quality Trade Multiplier

Apply a multiplier to the base P&L reward based on entry quality:
```python
if trade_result and is_in_spike:
    quality = 1.0 + (spike_freshness_at_entry * 0.2)  # 1.0 to 1.2x
    if first_trade_on_spike:
        quality += 0.1  # Up to 1.3x
    pnl_bonus += base_pnl_reward * (quality - 1.0)
```

**Rationale:** Make good entries more valuable without penalizing mediocre entries.

### Configuration

```python
@dataclass
class SpikeQualityBonusConfig:
    """
    v4: Additive bonuses on top of hybrid reward.
    All values are BONUSES (non-negative).
    """
    # Entry timing bonuses
    bonus_fresh_spike_entry: float = 0.3      # Max bonus for fresh spike entry
    fresh_spike_threshold: float = 0.7         # freshness > 0.7 triggers bonus

    bonus_patient_entry: float = 0.2           # Bonus for waiting before entry
    patient_threshold: float = 0.6             # minutes_since_trade > 36min

    bonus_first_trade_on_spike: float = 0.4    # Bonus for first trade on spike

    # Trade quality bonuses
    large_move_threshold: float = 0.03         # 3% move
    large_move_multiplier: float = 1.0         # Extra reward = return * multiplier
    huge_move_threshold: float = 0.05          # 5% move
    huge_move_multiplier: float = 1.5          # Additional multiplier for huge moves

    # Quality multiplier (scales base P&L reward)
    enable_quality_multiplier: bool = True
    max_freshness_bonus: float = 0.2           # Up to 20% extra for fresh spike
    first_trade_bonus: float = 0.1             # Extra 10% for first trade
```

### Implementation Strategy

```python
class SpikeQualityBonusCalculator:
    """
    v4: Wraps the hybrid RewardCalculator and adds spike-quality bonuses.

    This is a COMPOSITIONAL approach - we don't modify hybrid, we enhance it.
    """

    def __init__(
        self,
        base_config: RewardConfig = None,
        bonus_config: SpikeQualityBonusConfig = None
    ):
        # Use hybrid as base
        base_config = base_config or RewardConfig(reward_type=RewardType.HYBRID)
        self.base_calculator = RewardCalculator(base_config)
        self.bonus_config = bonus_config or SpikeQualityBonusConfig()

        # Track entry quality for exit bonus calculation
        self._entry_spike_freshness: float = 0.0
        self._entry_was_first_trade: bool = False

    def calculate(
        self,
        action: int,
        prev_price: float,
        current_price: float,
        position: Optional[Any] = None,
        prev_position: Optional[Any] = None,
        trade_result: Optional[Any] = None,
        time_in_position: float = 0,
        episode_returns: Optional[List[float]] = None,
        spike_context: Optional[dict] = None,
    ) -> float:
        """Calculate reward with spike-quality bonuses."""

        # 1. Get base hybrid reward (proven stable)
        base_reward = self.base_calculator.calculate(
            action=action,
            prev_price=prev_price,
            current_price=current_price,
            position=position,
            prev_position=prev_position,
            trade_result=trade_result,
            time_in_position=time_in_position,
            episode_returns=episode_returns,
        )

        # 2. Calculate additive bonuses
        bonus = self._calculate_bonuses(
            action=action,
            trade_result=trade_result,
            spike_context=spike_context or {},
        )

        return base_reward + bonus

    def _calculate_bonuses(
        self,
        action: int,
        trade_result: Optional[Any],
        spike_context: dict,
    ) -> float:
        """Calculate all spike-quality bonuses (always >= 0)."""

        bonus = 0.0
        cfg = self.bonus_config

        # Extract spike context
        minutes_since_trade = spike_context.get('minutes_since_last_trade', 1.0)
        spike_freshness = spike_context.get('spike_freshness', 0.0)
        already_traded = spike_context.get('already_traded_this_spike', 0.0)
        is_in_spike = spike_context.get('is_in_spike', 0.0)

        from .environment import Action

        # =========================================
        # ENTRY BONUSES
        # =========================================
        if action == Action.ENTER_LONG:
            # Track entry quality for later
            self._entry_spike_freshness = spike_freshness
            self._entry_was_first_trade = already_traded < 0.5

            # Bonus 1: Fresh spike entry
            if is_in_spike > 0.5 and spike_freshness > cfg.fresh_spike_threshold:
                bonus += cfg.bonus_fresh_spike_entry * spike_freshness

            # Bonus 2: Patient entry
            if minutes_since_trade > cfg.patient_threshold:
                bonus += cfg.bonus_patient_entry

            # Bonus 3: First trade on spike
            if is_in_spike > 0.5 and already_traded < 0.5:
                bonus += cfg.bonus_first_trade_on_spike

        # =========================================
        # EXIT BONUSES
        # =========================================
        if trade_result is not None:
            net_return = trade_result.return_pct

            # Bonus 4: Large move capture
            if net_return > cfg.large_move_threshold:
                bonus += net_return * cfg.large_move_multiplier

            if net_return > cfg.huge_move_threshold:
                bonus += net_return * cfg.huge_move_multiplier

            # Bonus 5: Quality multiplier on base P&L
            if cfg.enable_quality_multiplier and net_return > 0:
                # Use tracked entry quality
                quality_bonus = net_return * cfg.max_freshness_bonus * self._entry_spike_freshness
                if self._entry_was_first_trade:
                    quality_bonus += net_return * cfg.first_trade_bonus
                bonus += quality_bonus

            # Reset entry tracking
            self._entry_spike_freshness = 0.0
            self._entry_was_first_trade = False

        return max(0.0, bonus)  # Ensure non-negative
```

---

## Expected Reward Distribution

### v4 Bonus Budget Per Episode

| Bonus Type | Frequency | Value | Episode Total |
|------------|-----------|-------|---------------|
| Fresh spike entry | ~3 entries | +0.2 avg | +0.6 |
| Patient entry | ~4 entries | +0.2 | +0.8 |
| First trade on spike | ~2 entries | +0.4 | +0.8 |
| Large move capture | ~1 trade | +0.04 | +0.04 |
| Quality multiplier | ~5 wins | +0.02 avg | +0.1 |

**Total bonus per episode: +2.3 (approximate)**

### v4 Expected Reward

```
Base hybrid reward: ~9.5
Spike quality bonus: +2.3
Expected v4 reward: ~11.8
```

This breaks the 9.5 plateau while maintaining stability.

---

## Training Configuration

```python
# Recommended training configuration for v4
EpisodeConfig(
    episode_length=480,              # Back to 8 hours (not 1440!)
    sampling_mode='event_anchored',  # Keep biasing toward interesting events
    context_lookback_minutes=1440,   # 24hr context for features
    max_trades_per_episode=10,
    initial_stop_pct=0.02,
    fee_rate=0.0055,
)

PPO(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,              # Higher entropy for exploration
    policy_kwargs={
        'net_arch': dict(pi=[256, 256], vf=[256, 256]),
    },
)
```

---

## Experiment Plan

### Phase 1: Validation (v4.0)
1. Implement `SpikeQualityBonusCalculator`
2. Run 500K timesteps with event_anchored sampling
3. **Success criteria:** Reward > 9.5, ideally > 10.5

### Phase 2: Tuning (v4.1)
1. Grid search bonus magnitudes:
   - `bonus_fresh_spike_entry`: [0.2, 0.3, 0.4]
   - `bonus_first_trade_on_spike`: [0.3, 0.4, 0.5]
   - `large_move_multiplier`: [0.5, 1.0, 1.5]
2. Run 1M timesteps each
3. Select best configuration

### Phase 3: Long Run (v4.2)
1. Full 2M timestep training
2. Compare to v1/v2 on held-out test data
3. Measure actual trading metrics (win rate, Sharpe, max drawdown)

---

## Files to Modify

1. **`/Users/bz/Pythia2/src/rl/rewards.py`**
   - Add `SpikeQualityBonusConfig` dataclass
   - Add `SpikeQualityBonusCalculator` class
   - Keep existing calculators unchanged

2. **`/Users/bz/Pythia2/scripts/train_rl_agent.py`**
   - Add `--reward-type spike_quality_bonus` option
   - Wire up new calculator in `make_env()`

3. **`/Users/bz/Pythia2/docs/RL_IMPROVEMENTS.md`**
   - Document v4 approach

---

## Key Insights

### Why Bonuses Work Better Than Penalties

1. **Positive gradient flow:** Agent learns "do more of this" not "do less of everything"
2. **Stable baseline:** Hybrid reward provides consistent positive signal
3. **Additive composition:** Bonuses can only improve, never hurt
4. **Exploration friendly:** Agent still explores "bad" actions without catastrophic penalty
5. **Plateau breaking:** Extra reward for quality creates gradient above the plateau

### Why v3 Failed

1. **Penalty stacking:** Multiple overlapping penalties created negative-sum game
2. **Episode length:** 1440 minutes amplified time-based penalties 3x
3. **Symmetric loss scaling:** `reward_scale_loss=12` (close to win=15) discouraged risk-taking
4. **Opportunity cost:** Penalizing inaction + penalizing action = no good moves
5. **Reward landscape:** All actions negative meant agent minimized activity

---

## Command to Run v4

```bash
python scripts/train_rl_agent.py \
    --reward-type spike_quality_bonus \
    --total-timesteps 500000 \
    --n-envs 8 \
    --episode-length 480 \
    --sampling-mode event_anchored \
    --experiment-name v4_spike_quality_bonus
```

---

## Conclusion

v4 represents a fundamental shift in reward philosophy:

- **v3:** "Punish bad behavior" -> Agent learns to do nothing
- **v4:** "Reward good behavior" -> Agent learns to do better

By keeping the proven hybrid reward as a base and adding targeted bonuses for spike-quality trading, v4 should:
1. Maintain the stable ~9.5 baseline
2. Add +2-3 bonus reward for quality trades
3. Break the plateau and reach 11-12+ reward
4. Improve actual trading metrics (win rate on high-quality entries)

The key is **compositional enhancement** rather than **wholesale replacement**.
