# RL Position Sizer - Iteration 2 Results

## Summary

**Date:** 2026-03-19

**Objective:** Analyze v3 results and implement improvements

## Data Analysis Findings

### Return Distribution
- Mean max_return_4h: 0.88% (overall), 0.43% (test set)
- Only 0.8% of signals hit 5% TP
- 0% of signals hit -3% SL (bullish market regime)
- 96% of signals have positive returns

### Winner vs Loser Characteristics
| Feature | Winners (ret >= 5%) | Neutral | Delta |
|---------|---------------------|---------|-------|
| Momentum | 0.485 | 0.032 | +0.453 |
| Volatility | 0.266 | 0.209 | +0.057 |
| Volume Ratio | 0.910 | 0.964 | -0.054 |

**Key Insight:** Momentum is the strongest differentiator for winning trades

## V3 to V4 Improvements

### Changes Implemented in V4:
1. **Trailing stops** - Lock in gains at 2% with 1% trail distance
2. **Momentum rank feature** - Added percentile ranking for momentum
3. **Better reward shaping** - Penalize premature exits, reward holding
4. **Minimum hold time** - 3 bars before agent can exit
5. **Prioritized replay buffer** - Weight spike events higher
6. **Double DQN** - Use online network to select, target to evaluate
7. **Deeper network** - 128 hidden units with LayerNorm

### Performance Comparison

| Metric | V3 | V4 | Change |
|--------|-----|-----|--------|
| Total PnL | $266.38 | $276.47 | +$10.09 (+3.8%) |
| # Trades | 357 | 94 | -263 |
| Win Rate | 54.3% | 70.2% | +15.9pp |
| Profit Factor | 5.54 | 10.61 | +5.07 (+92%) |
| Avg PnL/Trade | $0.75 | $2.94 | +$2.20 (+293%) |
| Avg Position | 34% | 61.8% | +27.8pp |
| Avg Hold Time | ~1 bar | 6.8 bars | +5.8 bars |

### What V4 Learned:
- Takes MORE high-momentum signals (54/57 vs 17/57)
- Uses larger positions on high-quality signals
- Holds positions longer (avg 6.8 bars vs immediate exit)
- More selective overall (94 trades vs 357)

### Exit Distribution (V4):
- Agent exits: 92 (71% win rate, +0.32% avg)
- Time exits: 1 (0% win rate)
- Trailing stop exits: 1 (+1.88% profit)

## Experiments with Trailing Stop Parameters

| Activation | Trail Distance | PnL | Win Rate | Trail Exits |
|------------|----------------|-----|----------|-------------|
| 2.0% | 1.0% | $276 | 70.2% | 1 |
| 0.5% | 0.3% | $59 | 54.5% | 19 |

**Finding:** Tighter trailing stops hurt performance by exiting too early. The market doesn't provide enough volatility to benefit from tight trails.

## Key Findings

1. **V4 is better than V3** across all metrics except raw trade count
2. **Momentum is critical** - V4 correctly prioritizes high-momentum signals
3. **Trailing stops need room** - 2% activation works better than 0.5%
4. **Quality over quantity** - Fewer trades with larger positions outperforms
5. **Hold time matters** - Minimum hold period prevents early exits

## Files Created/Modified

- `/Users/bz/Pythia2/src/models/rl_position_sizer_v4.py` - New v4 implementation
- `/Users/bz/Pythia2/models/rl_position_sizer_v4.pt` - Final trained model
- `/Users/bz/Pythia2/models/rl_position_sizer_v4_best.pt` - Best training checkpoint

## Next Steps (Iteration 3)

1. **Market regime detection** - Add bull/bear/sideways classification
2. **Adaptive trailing stops** - Adjust parameters based on volatility
3. **Multi-symbol features** - Cross-asset correlation signals
4. **Longer evaluation period** - Test on more diverse market conditions
5. **Risk-adjusted rewards** - Incorporate Sharpe/Sortino into reward function
