# RL System Improvements

## v3: Spike-Aware Reward Function (IMPLEMENTED)

**Problem:** v1 and v2 both plateau at ~9.5 reward. Adding spike tracking features to the state did NOT break the plateau because the reward function did not use these features.

**Root Cause Analysis:**
1. State features (spike_freshness, already_traded_this_spike) were available but NOT incentivized
2. Reward function was blind to spike quality
3. Asymmetric win/loss scaling encouraged gambling rather than selectivity
4. No explicit penalty for re-trading the same spike

**Solution:** New `SpikeAwareRewardCalculator` in `/Users/bz/Pythia2/src/rl/rewards.py`

### Key Changes:

1. **Spike Quality Multiplier**: Trade rewards scaled by spike freshness
   ```python
   quality_multiplier = stale_penalty + (1 - stale_penalty) * spike_freshness
   if already_traded_this_spike:
       quality_multiplier *= 0.5  # Halve reward for repeat trades
   reward *= quality_multiplier
   ```

2. **Re-entry Penalties**: Explicit negative rewards for bad behavior
   ```python
   if action == ENTER_LONG and already_traded_this_spike:
       reward -= 0.5  # Penalty for re-trading same spike

   if minutes_since_last_trade < 0.5:  # < 30 minutes
       reward -= 0.3 * (1 - minutes / threshold)  # Rapid re-entry penalty
   ```

3. **Selectivity Bonuses**: Reward patience and fresh entries
   ```python
   if spike_freshness > 0.8 and is_in_spike:
       reward += 0.5  # Fresh spike entry bonus

   if minutes_since_last_trade > 0.75:  # > 45 minutes
       reward += 0.3  # Patient trade bonus
   ```

4. **More Symmetric P&L Scaling**: Discourage gambling
   ```python
   reward_scale_win = 15.0   # (was 10.0)
   reward_scale_loss = 12.0  # (was 5.0 - now losses hurt more)
   ```

5. **Large Move Bonuses**: Extra reward for catching significant moves
   ```python
   if return > 5%: reward *= 3.0 (huge move)
   if return > 3%: reward *= 2.0 (large move)
   ```

### Configuration: `SpikeAwareRewardConfig`

```python
SpikeAwareRewardConfig(
    # P&L scales
    reward_scale_win=15.0,
    reward_scale_loss=12.0,

    # Spike quality
    spike_freshness_weight=1.0,
    stale_spike_penalty=0.3,

    # Re-entry penalties
    penalty_already_traded_spike=0.5,
    penalty_rapid_reentry=0.3,
    rapid_reentry_threshold=0.5,

    # Selectivity bonuses
    bonus_fresh_spike_entry=0.5,
    bonus_patient_trade=0.3,
    patience_threshold=0.75,

    # Large move bonuses
    large_move_threshold=0.03,
    large_move_bonus=2.0,
    huge_move_threshold=0.05,
    huge_move_bonus=3.0,
)
```

### Usage

```python
# Training with spike-aware rewards (recommended)
python scripts/train_rl_agent.py \
    --reward-type spike_aware \
    --total-timesteps 2000000 \
    --n-envs 8 \
    --experiment-name v3_spike_aware

# Or directly in code:
from src.rl.rewards import SpikeAwareRewardCalculator, SpikeAwareRewardConfig
from src.rl.environment import TradingEnvironment, EpisodeConfig

config = SpikeAwareRewardConfig()
reward_calc = SpikeAwareRewardCalculator(config)

env = TradingEnvironment(
    db_path="...",
    config=EpisodeConfig(sampling_mode='event_anchored'),
    reward_calculator=reward_calc,
)
```

### Expected Improvements

1. **Break plateau**: Different spike qualities get DIFFERENT rewards
2. **Reduce overtrading**: Explicit penalties for re-entry
3. **Encourage selectivity**: Bonuses for patient, high-quality entries
4. **Better risk management**: More symmetric win/loss scaling

---

## v2: Spike Tracking (IMPLEMENTED)

**Problem:** Agent sees spike signal, trades, exits, then re-enters because spike features still elevated.

**Solution:** Added 5 spike-tracking features to state (now 56 total dimensions):

```python
# New features in environment.py _get_spike_tracking_features()
- minutes_since_last_trade   # 0.0 = just traded, 1.0 = 60+ min ago
- trades_in_last_hour        # Normalized: 0.0 = none, 1.0 = 5+ trades
- spike_freshness            # 1.0 at spike start → 0.0 after 60 min
- already_traded_this_spike  # Binary: did we enter during this spike?
- is_in_spike                # Binary: currently elevated volume/volatility
```

**Implementation Details:**
- `_detect_current_spike()`: Uses rolling 60-min averages to detect volume > 3x or volatility > 2x
- `_record_trade()`: Called on ENTER_LONG to mark spike as traded
- `spike_start_time`: Tracks when current spike began
- `spike_traded`: Flag reset when new spike starts

**State Dimension:** 56 (45 market + 6 position + 5 spike tracking)

**Status:** ✅ Implemented and trained

---

## v2 Training Results (2026-03-17)

**Configuration:**
- Network: [256, 256] (4x larger than v1)
- Batch size: 256 (4x larger than v1)
- 8 parallel environments
- 2M timesteps
- Higher entropy (0.02)

**Results:**
- Final reward: 9.53 (similar to v1's 9.59)
- Reward range: 9.15-9.64 throughout training
- Training time: 72 minutes at 468 FPS

**Key Finding:** Spike tracking features alone do NOT break the ~9.5 plateau. The policy can see spike state but still achieves similar reward. This suggests:
1. Reward function needs explicit penalties for bad behaviors
2. Features provide info but don't incentivize different behavior
3. Agent may be ignoring new features since reward signal is the same

**Model saved:** `/Users/bz/Pythia2/models/rl/v2_optimized/ppo_v2_optimized_final.zip`

---

## v1 Training Observations

- Reward stuck at ~9.6 (flat) - agent learned to trade but not selectively
- 1M timesteps, 47 minutes on MPS
- Model saved: `/Users/bz/Pythia2/models/rl/full_run_v1/ppo_trading_final.zip`

---

## Future Improvements (v3+)

### Reward Shaping (HIGH PRIORITY)
Based on v2 results, features alone are not enough. Must modify reward directly:

1. **Re-entry penalty:**
   ```python
   if action == ENTER_LONG and minutes_since_last_exit < 30:
       reward -= 0.5  # Explicit penalty
   ```

2. **Spike freshness scaling:**
   ```python
   trade_reward *= spike_freshness  # Reduce reward for stale spikes
   ```

3. **Sharpe-based component:**
   ```python
   sharpe_bonus = calculate_rolling_sharpe(returns)
   total_reward = trade_reward + 0.1 * sharpe_bonus
   ```

### Exploration
- ✅ Higher entropy (0.02) - tested in v2, didn't break plateau
- Consider curiosity-driven exploration bonus
- Entropy annealing schedule

### Architecture
- Test attention-based policy network (Phase 2 architecture)
- Multi-head attention over different timescales
- Recurrent policy (LSTM) for temporal patterns

