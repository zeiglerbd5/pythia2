# RL Trading Agent v1 - Training Results

**Date:** 2026-03-17
**Duration:** ~47 minutes
**Total Timesteps:** 1,007,616

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
# Total state dimension: 51 features
```

### PPO Hyperparameters
```python
PPO(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device='mps',  # Apple Silicon GPU
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
| Metric | Value |
|--------|-------|
| `ep_rew_mean` | 9.59 |
| `ep_len_mean` | 474 steps |
| `explained_variance` | 0.586 |
| `entropy_loss` | -0.83 |
| `clip_fraction` | 0.112 |
| `fps` | 360 |
| `n_updates` | 1,220 |

### Evaluation Progression
| Timesteps | Mean Reward | Trend |
|-----------|-------------|-------|
| 100K | 8.98 | Baseline |
| 200K | 9.74 | ↑ Peak |
| 300K | 9.64 | ↓ |
| 400K | 9.54 | ↓ |
| 500K | 9.12 | ↓ Lowest |
| 600K | 9.69 | ↑ |
| 700K | 9.43 | ↓ |
| 800K | 9.65 | ↑ |
| 900K | 9.55 | ↓ |
| 1M | 9.53 | Final |

### Training Speed
- **Device:** Apple MPS (M-series GPU)
- **Parallel environments:** 4
- **Average FPS:** 360-450
- **Total time:** 2,796 seconds (~47 min)

---

## Model Artifacts

```
/Users/bz/Pythia2/models/rl/full_run_v1/
├── ppo_trading_final.zip      # Final model (220 KB)
├── best_model.zip             # Best evaluation score
├── ppo_trading_200000_steps.zip
├── ppo_trading_400000_steps.zip
├── ppo_trading_600000_steps.zip
├── ppo_trading_800000_steps.zip
├── ppo_trading_1000000_steps.zip
├── evaluations.npz            # Evaluation history
└── tensorboard/               # TensorBoard logs
```

---

## Observations

### What Worked
1. **Environment stability** - 1M+ timesteps without crashes
2. **Event-anchored sampling** - Episodes started near interesting market events
3. **MPS GPU acceleration** - 360+ FPS on Apple Silicon
4. **Retry logic** - Handled data gaps gracefully

### Issues Identified

#### 1. Flat Reward Curve
- Reward stayed at ~9.5 throughout training (no improvement)
- Suggests agent learned a simple heuristic early and stopped exploring
- Possibly just trading every spike regardless of quality

#### 2. Repeated Spike Trading (Critical)
- Agent re-enters same spike multiple times
- Missing state features to track "already traded this event"
- Needs cooldown/freshness features

#### 3. Reward Function
- +1% win = +1.1 reward, -2% loss = -0.1 reward
- Asymmetry may encourage overtrading
- No penalty for re-entry within N minutes

---

## Next Steps (v2)

### Priority Fixes
1. **Add spike-tracking features:**
   ```python
   - minutes_since_last_trade
   - trades_in_last_hour
   - spike_age_minutes
   - already_traded_this_spike (binary)
   - spike_freshness (decaying 1.0 → 0.0)
   ```

2. **Reward function improvements:**
   - Penalize re-entry within 30 minutes of exit
   - Scale reward by signal freshness
   - Add Sharpe-based component

3. **Exploration:**
   - Increase entropy coefficient early in training
   - Add curiosity-driven exploration bonus

### Experiments to Run
- Compare event_anchored vs sequential sampling
- Test longer episodes (24hr instead of 8hr)
- Try different reward scaling

---

## Usage

### Load the trained model:
```python
from stable_baselines3 import PPO
from src.rl.environment import TradingEnvironment, EpisodeConfig
from src.rl.features import FeatureExtractor, FeatureConfig

model = PPO.load("/Users/bz/Pythia2/models/rl/full_run_v1/ppo_trading_final")

env = TradingEnvironment(
    db_path="/Users/bz/Pythia2/rl_training_data.db",
    config=EpisodeConfig(sampling_mode='event_anchored'),
    feature_extractor=FeatureExtractor(FeatureConfig(
        include_order_book=True,
        include_trade_flow=True,
    )),
)

obs, info = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

### View TensorBoard:
```bash
tensorboard --logdir /Users/bz/Pythia2/models/rl/full_run_v1/tensorboard
```
