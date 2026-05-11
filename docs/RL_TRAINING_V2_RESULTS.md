# RL Trading Agent v2 - Training Results

**Date:** 2026-03-17
**Duration:** ~72 minutes
**Total Timesteps:** 2,015,232

---

## What's New in v2

### Spike Tracking Features (5 new state dimensions)
```python
# Added to environment.py _get_spike_tracking_features()
- minutes_since_last_trade   # 0.0 = just traded, 1.0 = 60+ min ago
- trades_in_last_hour        # Normalized: 0.0 = none, 1.0 = 5+ trades
- spike_freshness            # 1.0 at spike start → 0.0 after 60 min
- already_traded_this_spike  # Binary: did we enter during this spike?
- is_in_spike                # Binary: currently elevated volume/volatility
```

### Optimized Configuration
- **Larger network:** [256, 256] (vs v1's [64, 64])
- **Larger batch size:** 256 (vs v1's 64)
- **More environments:** 8 parallel (vs v1's 4)
- **More training:** 2M timesteps (vs v1's 1M)
- **Higher entropy:** 0.02 (vs v1's 0.01) for more exploration

---

## Configuration

### Environment Settings
```python
EpisodeConfig(
    episode_length=480,              # 8 hours of 1-minute decisions
    sampling_mode='event_anchored',  # Bias toward volume/volatility spikes
    window_overlap_pct=0.5,          # 50% overlap between episodes
    context_lookback_minutes=1440,   # 24-hour historical context
    max_trades_per_episode=10,
    initial_stop_pct=0.02,           # 2% initial stop loss
    fee_rate=0.0055,                 # 0.55% round-trip fees
)
```

### Feature Configuration
```python
FeatureConfig(
    include_order_book=True,      # Order book imbalance, spread, depth
    include_trade_flow=True,      # VPIN, roll measure, buy/sell pressure
    include_volume_profile=True,  # Volume distribution features
    include_multi_timeframe=True, # 5m, 15m, 1h aggregations
)
# Total state dimension: 56 features (45 market + 6 position + 5 spike tracking)
```

### PPO Hyperparameters
```python
PPO(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,              # 4x larger than v1
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,               # 2x more exploration than v1
    vf_coef=0.5,
    max_grad_norm=0.5,
    device='mps',
    policy_kwargs={
        'net_arch': dict(pi=[256, 256], vf=[256, 256]),  # 4x larger network
    },
)
```

### Training Data
- **Database:** `/Users/bz/Pythia2/rl_training_data.db` (393 MB)
- **OHLCV rows:** 1,128,565
- **Feature rows:** 426,141 (order book, VPIN, technical indicators)
- **Trade volume rows:** 764,843 (buy/sell pressure)
- **Symbols:** 50 cryptocurrencies
- **Time range:** 30 days (2026-02-15 to 2026-03-17)

---

## Results

### Final Metrics
| Metric | v2 Value | v1 Value | Change |
|--------|----------|----------|--------|
| `ep_rew_mean` | 9.53 | 9.59 | -0.06 |
| `ep_len_mean` | 478 steps | 474 steps | +4 |
| `fps` | 468 | 360 | +30% |
| `total_timesteps` | 2,015,232 | 1,007,616 | +100% |
| `training_time` | 72 min | 47 min | +53% |

### Evaluation Progression
| Timesteps | Mean Reward | Trend |
|-----------|-------------|-------|
| 16K | 9.15 | Baseline |
| 81K | 9.54 | ↑ |
| 163K | 9.63 | ↑ |
| 180K | 9.64 | ↑ Peak Early |
| 327K | 9.63 | - |
| 573K | 9.53 | ↓ |
| 622K | 9.36 | ↓ Lowest |
| 655K | 9.58 | ↑ |
| 917K | 9.61 | ↑ |
| 1.25M | 9.51 | ↓ |
| 1.54M | 9.25 | ↓ |
| 1.77M | 9.49 | ↑ |
| 1.92M | 9.44 | - |
| 2M | 9.53 | Final |

### Final Evaluation
```
Eval num_timesteps=2000000, episode_reward=9.49 +/- 0.33
```

### Training Speed
- **Device:** Apple MPS (M-series GPU)
- **Parallel environments:** 8
- **Average FPS:** 468 (30% faster than v1)
- **Total time:** 4,303 seconds (~72 min)

---

## Model Artifacts

```
/Users/bz/Pythia2/models/rl/v2_optimized/
├── ppo_v2_optimized_final.zip   # Final model (1.9 MB)
├── best_model.zip               # Best evaluation score
├── ppo_v2_opt_800000_steps.zip  # Checkpoint
├── ppo_v2_opt_1600000_steps.zip # Checkpoint
├── evaluations.npz              # Evaluation history
└── tensorboard/                 # TensorBoard logs
```

---

## Analysis

### What Worked
1. **Spike tracking features** - Environment now tracks spike freshness and trading history
2. **Larger network** - More capacity for complex patterns
3. **Higher entropy** - More exploration, less premature convergence
4. **Faster training** - 468 FPS vs 360 FPS (30% improvement)
5. **Stable training** - 2M+ timesteps without crashes

### Issues Remaining

#### 1. Flat Reward Curve Persists
- Despite 2x more training and spike tracking, reward still oscillates around 9.4-9.6
- Spike tracking features may not be utilized effectively by the policy
- The reward function itself may need restructuring

#### 2. No Significant Improvement Over v1
- v1 final: 9.59, v2 final: 9.53
- Suggests the fundamental limitation is not feature availability
- Likely need to change reward shaping, not just add features

#### 3. High Variance
- Reward fluctuates ±0.3 throughout training
- Agent may be learning different strategies that have similar expected returns

---

## Comparison: v1 vs v2

| Aspect | v1 | v2 |
|--------|----|----|
| State dim | 51 | 56 (+5 spike tracking) |
| Network | [64, 64] | [256, 256] |
| Batch size | 64 | 256 |
| Parallel envs | 4 | 8 |
| Timesteps | 1M | 2M |
| Entropy coef | 0.01 | 0.02 |
| Final reward | 9.59 | 9.53 |
| Reward range | 8.98-9.74 | 9.15-9.64 |

**Conclusion:** v2 has more features and compute but similar performance. The spike tracking features alone are not sufficient to break out of the 9.5 plateau.

---

## Next Steps (v3+)

### Reward Function Changes (High Priority)
1. **Penalize re-entry explicitly:**
   ```python
   if action == ENTER_LONG and minutes_since_last_exit < 30:
       reward -= 0.5  # Penalty for rapid re-entry
   ```

2. **Scale reward by spike freshness:**
   ```python
   trade_reward *= spike_freshness  # 1.0 for fresh, 0.0 for stale
   ```

3. **Add Sharpe-based component:**
   ```python
   sharpe_bonus = calculate_rolling_sharpe(returns, window=20)
   total_reward = trade_reward + 0.1 * sharpe_bonus
   ```

### Architecture Experiments
- Try attention-based policy (Phase 2 architecture)
- Multi-head attention over different timescales
- Recurrent policy (LSTM/GRU) for sequence memory

### Exploration Strategies
- Curiosity-driven exploration bonus
- Entropy annealing (high early, low late)
- Population-based training

---

## Usage

### Load the trained model:
```python
from stable_baselines3 import PPO
from src.rl.environment import TradingEnvironment, EpisodeConfig
from src.rl.features import FeatureExtractor, FeatureConfig

model = PPO.load("/Users/bz/Pythia2/models/rl/v2_optimized/ppo_v2_optimized_final")

env = TradingEnvironment(
    db_path="/Users/bz/Pythia2/rl_training_data.db",
    config=EpisodeConfig(sampling_mode='event_anchored'),
    feature_extractor=FeatureExtractor(FeatureConfig(
        include_order_book=True,
        include_trade_flow=True,
        include_volume_profile=True,
        include_multi_timeframe=True,
    )),
)

obs, info = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

### View TensorBoard:
```bash
tensorboard --logdir /Users/bz/Pythia2/models/rl/v2_optimized/tensorboard
```
