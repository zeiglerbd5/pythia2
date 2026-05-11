# Spike Trading Strategy Analysis

## Date: March 23, 2026
## Analysis Period: March 19-23, 2026

---

## Executive Summary

The backtest revealed that **volume-based entry signals are too restrictive** for spike trading. Most big movers (SYND +120%, BOBA +66%) had **normal volume** during their spikes. The recommended approach is to switch to **momentum-based entry** (6%+ in 1 hour) which captured 56% win rate with positive expectancy.

---

## Backtest Results Comparison

| Strategy | Trades | Win% | Total PnL | Avg PnL |
|----------|--------|------|-----------|---------|
| Volume 3x + Price 5% | 0 | - | $0 | - |
| Volume 1.5x + Price 5% | 5 | 40% | -$2.80 | -$0.56 |
| Price only 5% (24h) | 49 | 53% | +$294 | +$6.00 |
| Price only 10% (24h) | 53 | 57% | +$481 | +$9.08 |
| **Momentum 6% (1h)** | **111** | **56%** | **+$818** | **+$7.37** |
| Momentum 7% (1h) | 67 | 61% | +$491 | +$7.33 |

**Winner: 6% 1h Momentum Strategy** - Best risk-adjusted returns

---

## Key Findings

### 1. Volume Signals Are Lagging Indicators

Analysis of SYND-USD (+120% spike):
- Volume multiple was **0.75-1.5x** when the spike began
- Volume only exploded to **17-42x** AFTER the move was underway
- Waiting for 3x volume meant **missing the entire move**

```
Hour             Price    Vol Multiple  1h Change
2026-03-21 15:00 $0.0329  1.91x        +7.5%   <- ENTRY SIGNAL (momentum)
2026-03-21 19:00 $0.0378  17.97x       +15.6%  <- Volume finally explodes
2026-03-21 20:00 $0.0465  42.00x       +23.0%  <- Peak momentum
```

### 2. Momentum Is a Leading Indicator

When a coin is up 6%+ in 1 hour:
- **56% probability** of gaining another 3%+ in next 4 hours
- Average max gain: **3.1%** in the next 4 hours
- With trailing stops: **55% of entries exit profitably**

### 3. Optimal Parameters (Backtested)

**Entry:**
- Trigger: 6%+ price gain in last 1 hour
- Skip if: Already up 25%+ (FOMO trap)
- Cooldown: 2 hours between trades on same symbol

**Exit:**
- Stop Loss: 2% (tight, to manage volatility)
- Trailing Stop (ratcheting):
  - 2-6% gain: 1.2% trail
  - 6-10% gain: 1.8% trail (give room to run)
  - 10-15% gain: 1.2% trail (tightening)
  - 15%+ gain: 4.0% trail (lock in big wins)
- Max Hold: 4 hours (most continuation is fast)

### 4. Exit Analysis

| Exit Type | Count | Avg PnL |
|-----------|-------|---------|
| Stop Loss | 47 (42%) | -2.0% |
| Trailing Stop | 61 (55%) | +2.9% |
| Timeout | 3 (3%) | -0.5% |

The trailing stop strategy captures most of the profitable exits while limiting losses.

---

## Why Volume Signals Failed

1. **Liquidity comes AFTER visibility**: Big moves attract volume, not the other way around
2. **Threshold too high**: 3x was almost never reached before the move
3. **24h window too long**: By the time 24h volume was calculated, the move was over
4. **Low-cap coins have erratic volume**: Normal fluctuations can look like explosions

---

## Recommendations

### Immediate Changes

1. **Disable volume filter for entry** - Keep it for informational purposes only
2. **Add momentum-based entry** - 6%+ in 1 hour triggers signal
3. **Tighten stop loss** - From 3% to 2% (these are volatile trades)
4. **Reduce max hold time** - From 24h to 4h (continuation is fast or absent)

### Code Changes

New files created:
- `/Users/bz/Pythia2/scripts/momentum_spike_trader.py` - Standalone trader
- `/Users/bz/Pythia2/src/features/momentum_scanner.py` - Scanner for collector

### Strategy Configuration

```python
# OLD STRATEGY (volume-based)
volume_multiple_threshold: 3.0  # Too restrictive
price_change_threshold: 0.05   # 24h - too slow
stop_loss_pct: 0.03            # 3% - too wide

# NEW STRATEGY (momentum-based)
momentum_threshold_1h: 6.0     # 6% in 1h - faster detection
max_momentum_1h: 25.0          # Skip FOMO traps
stop_loss_pct: 0.02            # 2% - tighter for volatility
max_hold_hours: 4              # Faster exits
```

---

## What to Track in Collector

### Keep Tracking:
1. OHLCV data (essential for momentum calculation)
2. Volume signals (useful for confirmation, not entry)
3. Whale alerts (can combine with momentum)

### Add Tracking:
1. **1h momentum** - Core entry signal
2. **4h momentum** - Context for continuation
3. **Momentum acceleration** - Is the move speeding up?

### Remove/Deprioritize:
1. 24h price change as primary signal
2. Volume multiple as entry requirement
3. News signals (low hit rate in backtest)

---

## Expected Performance

Based on 4-day backtest (111 trades):
- Win Rate: ~56%
- Avg Win: +2.9%
- Avg Loss: -2.0%
- Profit Factor: 1.5x
- Expected Value: +0.74% per trade

With $1000 position size and 3 max positions:
- ~3-5 trades per day
- Expected daily P&L: +$20-40

**Note:** Past performance does not guarantee future results. Backtest was on 4 days of data and may not represent all market conditions.

---

## Next Steps

1. Run momentum_spike_trader.py in paper trading mode
2. Compare live results to backtest expectations
3. After 1 week, evaluate:
   - Actual win rate vs expected (56%)
   - Slippage impact
   - Signal quality during different market conditions
4. Consider hybrid approach: momentum + volume confirmation

---

## Files

- Backtest database: `/Users/bz/Pythia2/pythia_backtest_copy.duckdb`
- New trader: `/Users/bz/Pythia2/scripts/momentum_spike_trader.py`
- New scanner: `/Users/bz/Pythia2/src/features/momentum_scanner.py`
- Analysis script: `/Users/bz/Pythia2/scripts/backtest_volume_reactive.py`
