# Crypto Price Breakout Detection Strategies

## Technical Documentation v1.0

**Last Updated:** March 2026
**Backtesting Period:** October 2025 - March 2026 (143 days)
**Exchange:** Coinbase

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Breakout Hunter v5.3 Strategy](#breakout-hunter-v53-strategy)
   - [Triple Confirmation Pattern](#triple-confirmation-pattern)
   - [Mathematical Formulas](#mathematical-formulas)
   - [Entry Criteria](#entry-criteria)
   - [Exit System](#exit-system)
   - [Backtest Results](#backtest-results)
   - [Why T+1 is a Filter, Not an Entry Point](#why-t1-is-a-filter-not-an-entry-point)
3. [Accumulation Hunter v6.0 Strategy](#accumulation-hunter-v60-strategy)
   - [Stealth Accumulation Detection](#stealth-accumulation-detection)
   - [BAR Calculation](#bar-buyask-ratio-calculation)
   - [Signal Strength Scoring](#signal-strength-scoring)
   - [Entry and Exit Criteria](#entry-and-exit-criteria)
4. [Visual Reference](#visual-reference)
5. [Parameter Reference Tables](#parameter-reference-tables)
6. [Implementation Notes](#implementation-notes)

---

## Executive Summary

This document describes two complementary crypto trading strategies designed to detect and capitalize on price breakouts:

| Strategy | Approach | Win Rate | Trade Frequency |
|----------|----------|----------|-----------------|
| **Breakout Hunter v5.3** | React to breakouts with triple confirmation | 94.6% | ~1 trade / 2.5 days |
| **Accumulation Hunter v6.0** | Detect accumulation BEFORE breakout | TBD (new) | Varies |

Both strategies are designed for cryptocurrency markets and have been optimized for Coinbase's fee structure (0.55% per trade).

**Key Insight:** The strategies filter out 97%+ of signals, focusing only on the highest-probability setups. Quality over quantity.

---

## Breakout Hunter v5.3 Strategy

### Overview

Breakout Hunter detects large price moves accompanied by volume spikes, then waits for **three confirmations** before entering. This triple-confirmation approach achieved a **94.6% win rate** over 143 days of backtesting.

### Triple Confirmation Pattern

The strategy operates on hourly candles and requires three sequential conditions:

![Breakout Pattern](breakout_pattern.png)

#### T+0: Breakout Detection (Hour 0)

The initial breakout signal requires:
- **Price move > 5%** within the hour
- **Volume > 3x** the 24-hour average

This is the detection phase only - no trade is entered.

#### T+1: Continuation Filter (Hour 1)

T+1 serves as a **filter** to eliminate immediate reversals:
- **Return from T+0 close > 0%** (must be positive)
- If negative, signal is rejected

**Critical:** T+1 is NOT an entry point. See [Why T+1 is a Filter](#why-t1-is-a-filter-not-an-entry-point).

#### T+2: Entry Confirmation (Hour 2)

Final confirmation requires:
- **Return from T+1 close >= 5%** (strong resumption)
- **Volume continuation > 1x** (buying pressure continues)
- **Entry at T+2 OPEN price**

### Mathematical Formulas

#### Volume Ratio Calculation

The volume ratio compares current hourly volume to the 24-hour average:

```
Volume Ratio = V_current / V_avg24

Where:
  V_current = Volume in current hour
  V_avg24 = Average hourly volume over previous 24 hours
```

**Threshold:** Volume Ratio >= 3.0 for T+0 detection

#### Return Calculations

**T+0 Return (Breakout Move):**
```
R_T0 = (Close_T0 - Open_T0) / Open_T0 * 100

Required: R_T0 >= 5%
```

**T+1 Return (Continuation):**
```
R_T1 = (Close_T1 - Close_T0) / Close_T0 * 100

Required: R_T1 > 0%  (positive, any magnitude)
```

**T+2 Return (Resumption):**
```
R_T2 = (Close_T2 - Close_T1) / Close_T1 * 100

Required: R_T2 >= 5%
```

#### Volume Continuation

Measures whether buying pressure continues after the initial spike:

```
Vol_Continuation = V_post / V_pre

Where:
  V_post = Total volume in 3 hours after breakout
  V_pre  = Total volume in 6 hours before breakout

Required: Vol_Continuation >= 1.0
```

### Entry Criteria

| Criterion | Parameter | Value | Description |
|-----------|-----------|-------|-------------|
| T+0 Move | `INITIAL_MOVE_PCT` | 5% | Minimum hourly price move |
| T+0 Volume | `VOLUME_THRESHOLD` | 3.0x | Volume vs 24h average |
| T+1 Return | `T1_MIN_RETURN_PCT` | > 0% | Must be positive (filter) |
| T+2 Return | `T2_RESUMPTION_PCT` | 5% | Strong resumption required |
| Vol Continuation | `VOL_CONTINUATION_MIN` | 1.0x | Post/pre volume ratio |
| Entry Price | - | T+2 Open | Enter at T+2 candle open |

### Exit System

The exit system uses a combination of:
1. Initial stop loss
2. Stepped profit locks
3. Tiered trailing stop
4. Maximum hold time

![Profit Locks](profit_locks.png)

#### Initial Stop Loss

```
Stop_Loss_Price = Entry_Price * (1 - 0.08)

8% initial stop loss from entry
```

#### Stepped Profit Locks

Profit locks ensure winners don't become losers. Once price reaches a threshold, a minimum profit level is locked:

| Peak Gain | Locked Level | Effect |
|-----------|--------------|--------|
| +5% | -2% | Small loss acceptable |
| +10% | 0% (breakeven) | Can't lose money |
| +15% | +5% | Guaranteed profit |
| +25% | +12% | Significant profit locked |
| +40% | +25% | Major profit protected |

**Formula:**
```
If (Highest_Price - Entry_Price) / Entry_Price >= Threshold:
    Stop_Price = max(Stop_Price, Entry_Price * (1 + Lock_Level))
```

#### Tiered Trailing Stop

The trailing stop activates at +20% gain and tightens as gains increase:

![Trailing Stop](trailing_stop.png)

| Peak Gain | Trail Percentage | Effect |
|-----------|-----------------|--------|
| +20% | 10% | Trail activates |
| +30% | 8% | Tighter trail |
| +50% | 6% | Protecting large gains |
| +70% | 5% | Maximum protection |

**Formula:**
```
Trail_Stop_Price = Highest_Price * (1 - Trail_Pct)

Where Trail_Pct is determined by peak gain tier
```

#### Maximum Hold Time

```
Max_Hold = Entry_Time + 48 hours
```

If no other exit condition is met, position is closed after 48 hours.

#### Take Profit Cap

```
Take_Profit_Price = Entry_Price * (1 + 0.80)

Exit if price reaches +80% gain
```

### Backtest Results

**Period:** October 2025 - March 2026 (143 days)

| Metric | Value |
|--------|-------|
| Total Trades | 56 |
| Win Rate | **94.6%** |
| Expected Return | +61.9% per trade |
| Trade Frequency | ~1 every 2.5 days |
| Signals Detected | 2,076 |
| Signals Filtered Out | 97.3% |

#### Exit Analysis

| Exit Type | Count | Avg Return | Total P&L |
|-----------|-------|------------|-----------|
| Trail Stop | 13 | +30.3% | +$9,843 |
| Stop Loss | 6 | -10.9% | -$1,641 |
| Profit Lock | 8 | -2.2% | -$448 |
| Max Hold | 1 | +5.7% | +$142 |

**Key Observation:** Trail stops captured the majority of profits. The strategy succeeds by letting winners run while cutting losers quickly.

### Why T+1 is a Filter, Not an Entry Point

#### The Failed T+1 Entry Experiment

Version 5.2 tested entering at T+1 close instead of waiting for T+2. The hypothesis was that earlier entry would capture more of the move.

**Results were disastrous:**

| Entry Mode | Win Rate | Avg P&L |
|------------|----------|---------|
| T+1 Entry | 33.3% | -2.5% |
| T+2 Entry | 94.6% | +61.9% |

#### Case Study: GODS-USD (March 17, 2026)

GODS-USD demonstrated exactly why T+1 entry fails:

| Hour | Event | Price | Return |
|------|-------|-------|--------|
| T+0 | Breakout | $0.045 | +23.4%, 279x vol |
| T+1 | Peak | $0.060 | +32.9% from T+0 |
| T+2 | Collapse | - | **-16.3%** from T+1 |

- **With T+1 entry:** Enter at $0.060 (the top) -> Stop loss hit -> **-8% loss**
- **With T+2 entry:** Signal filtered out (T+2 < 5%) -> **No loss**

#### T+1 Gain Distribution Analysis

| T+1 Gain | Win Rate | Observation |
|----------|----------|-------------|
| 0-5% | 66.7% | Moderate T+1 = good |
| 5-10% | 50.0% | Mixed results |
| 10-15% | 0% | All losers |
| 15-25% | 0% | All losers |
| 25%+ | 0% | All losers |

**Critical Finding:** When T+1 gain exceeds 10%, win rate drops to 0%. These are "buy the top" traps.

#### Why This Happens

1. **T+1 is often the peak:** 78% of breakouts reverse after T+1
2. **Selection bias:** Large T+1 gains feel like momentum but are exhaustion
3. **No confirmation:** T+1 entry has no proof the move will continue
4. **FOMO trap:** Entering on big green candles = buying what everyone else bought

#### The Solution

T+1 serves as a **filter only**:
- **T+1 positive:** Signal passes to T+2 evaluation
- **T+1 negative:** Signal rejected (immediate reversal)

T+2 confirmation proves the move has legs. This is what produces the 94.6% win rate.

---

## Accumulation Hunter v6.0 Strategy

### Overview

While Breakout Hunter reacts to breakouts, Accumulation Hunter attempts to detect stealth accumulation patterns **12-48 hours BEFORE** the breakout occurs.

The strategy is based on analysis of major price spikes, which revealed detectable signals in order book data 40+ hours before the move.

### Stealth Accumulation Detection

![Accumulation Pattern](accumulation_pattern.png)

Stealth accumulation exhibits these characteristics:
1. **Elevated volume** but **flat price** (buying absorbed by sellers)
2. **Order book imbalance** showing bid-side strength
3. **Ask depth collapse** as sell-side liquidity is absorbed
4. **Pattern persistence** over multiple hours

### BAR (Buy/Ask Ratio) Calculation

BAR measures the imbalance between bid and ask depth in the order book:

```
BAR = Bid_Depth / Ask_Depth

Where:
  Bid_Depth = Sum of top 10 bid sizes
  Ask_Depth = Sum of top 10 ask sizes
```

**BAR Multiple vs Baseline:**
```
BAR_Multiple = Current_BAR / Baseline_BAR

Where Baseline_BAR = Average BAR over previous 24 hours
```

| BAR Multiple | Interpretation |
|--------------|----------------|
| < 2x | Normal |
| 3x | Watch - possible accumulation |
| 5x | Strong signal |
| 8x+ | Critical - high probability breakout |

### Volume Anomaly Detection

```
Volume_Ratio = Current_Volume / Baseline_Volume

Where Baseline_Volume = Average hourly volume over 24h
```

**Accumulation Signal:**
- Volume_Ratio >= 2.0 (elevated buying)
- Price change < 2% (flat - buying absorbed)

The combination of high volume and flat price suggests someone is quietly accumulating.

### Ask Depth Collapse

Tracks reduction in sell-side liquidity:

```
Ask_Collapse_Pct = 1 - (Current_Ask_Depth / Baseline_Ask_Depth)
```

| Collapse % | Interpretation |
|------------|----------------|
| < 30% | Normal fluctuation |
| 50% | Significant - supply being absorbed |
| 80%+ | Critical - very thin sell side |

### Signal Strength Scoring

The strategy combines multiple signals into a composite score (0-100):

| Factor | Max Points | Criteria |
|--------|------------|----------|
| Volume anomaly | 25 | 15 @ 2x, 25 @ 3.5x |
| Price flatness | 15 | 15 @ <1%, 10 @ <2% |
| BAR multiple | 30 | 10 @ 3x, 20 @ 5x, 30 @ 8x |
| Ask collapse | 20 | 10 @ 30%, 15 @ 50%, 20 @ 80% |
| Sell absorption | 10 | BAR >= 3x AND price flat |

**Alert Levels:**

| Score | Level | Action |
|-------|-------|--------|
| < 30 | None | Ignore |
| 30-49 | Watch | Add to watchlist |
| 50-69 | Warning | Active monitoring |
| 70+ | Critical | Ready to enter on breakout |

### Entry and Exit Criteria

#### Entry Process

1. **Detection:** Signal enters watchlist (score >= 30)
2. **Confirmation:** Pattern persists 2+ hours, score >= 50 -> "accumulating"
3. **Ready State:** Score >= 70 -> "ready" for breakout
4. **Entry Trigger:**
   - Price moves > 3% from accumulation range
   - Volume surges > 2x baseline
   - Enter at market

#### Entry Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Min accumulation hours | 2h | Pattern must persist |
| Breakout price move | > 3% | Confirms breakout |
| Breakout volume | > 2x | Confirms demand |
| Signal expiry | 72h | Max time to wait for breakout |

#### Exit Parameters

Accumulation Hunter uses tighter stops than Breakout Hunter since entry is earlier:

| Parameter | Value | vs Breakout Hunter |
|-----------|-------|-------------------|
| Initial stop | 6% | 8% |
| Trail trigger | +15% | +20% |
| Trail percentage | 6% | 10% |
| Position size | $2,000 | $2,500 |
| Max positions | 2 | 3 |

#### Profit Locks

| Peak Gain | Locked Level |
|-----------|--------------|
| +5% | 0% (breakeven) |
| +10% | +3% |
| +20% | +8% |

---

## Visual Reference

### Breakout Pattern (T+0, T+1, T+2)

![Breakout Pattern](breakout_pattern.png)

### Profit Lock System

![Profit Locks](profit_locks.png)

### Tiered Trailing Stop

![Trailing Stop](trailing_stop.png)

### Accumulation Pattern

![Accumulation Pattern](accumulation_pattern.png)

---

## Parameter Reference Tables

### Breakout Hunter v5.3 Parameters

#### Detection Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `VOLUME_THRESHOLD` | 3.0 | T+0 volume vs 24h avg |
| `INITIAL_MOVE_PCT` | 5.0% | T+0 minimum move |
| `T1_MIN_RETURN_PCT` | 0.0% | T+1 must be positive |
| `T2_RESUMPTION_PCT` | 5.0% | T+2 minimum resumption |
| `VOL_CONTINUATION_MIN` | 1.0 | Post/pre volume ratio |

#### Exit Parameters

| Parameter | Value |
|-----------|-------|
| `INITIAL_STOP_LOSS_PCT` | 8% |
| `TRAIL_TRIGGER_PCT` | 20% |
| `TRAIL_STOP_PCT` | 10% (default) |
| `MAX_TAKE_PROFIT_PCT` | 80% |
| `MAX_HOLD_HOURS` | 48 |

#### Profit Lock Levels

| Trigger | Lock |
|---------|------|
| +5% | -2% |
| +10% | 0% |
| +15% | +5% |
| +25% | +12% |
| +40% | +25% |

#### Trail Tighten Levels

| Trigger | Trail % |
|---------|---------|
| +30% | 8% |
| +50% | 6% |
| +70% | 5% |

#### Position Sizing

| Parameter | Value |
|-----------|-------|
| `POSITION_SIZE_USD` | $2,500 |
| `MAX_POSITIONS` | 3 |
| `FEE_RATE` | 0.55% |

### Accumulation Hunter v6.0 Parameters

#### Detection Parameters

| Parameter | Value |
|-----------|-------|
| `VOLUME_ANOMALY_THRESHOLD` | 2.0x |
| `PRICE_FLAT_MAX` | 2% |
| `BAR_WATCH` | 3.0x |
| `BAR_STRONG` | 5.0x |
| `ASK_COLLAPSE_MIN` | 50% |
| `MIN_ACCUMULATION_HOURS` | 2h |

#### Entry Parameters

| Parameter | Value |
|-----------|-------|
| `BREAKOUT_PRICE_PCT` | 3% |
| `BREAKOUT_VOLUME_RATIO` | 2.0x |

#### Exit Parameters

| Parameter | Value |
|-----------|-------|
| `INITIAL_STOP_LOSS_PCT` | 6% |
| `TRAIL_TRIGGER_PCT` | 15% |
| `TRAIL_STOP_PCT` | 6% |
| `MAX_TAKE_PROFIT_PCT` | 80% |
| `MAX_HOLD_HOURS` | 48 |

#### Profit Lock Levels

| Trigger | Lock |
|---------|------|
| +5% | 0% |
| +10% | +3% |
| +20% | +8% |

#### Position Sizing

| Parameter | Value |
|-----------|-------|
| `POSITION_SIZE_USD` | $2,000 |
| `MAX_POSITIONS` | 2 |
| `FEE_RATE` | 0.55% |

---

## Implementation Notes

### Data Requirements

| Strategy | Data Needed | Frequency |
|----------|-------------|-----------|
| Breakout Hunter | OHLCV | Hourly |
| Accumulation Hunter | OHLCV + Order Book | Every 5 min |

### Gap Handling

Both strategies simulate 2% slippage when price gaps past stop levels:

```python
if exit_price < trade.current_stop:
    actual_exit = trade.current_stop * 0.98  # 2% slippage
```

### State Machine Flow

#### Breakout Hunter

```
No Signal -> pending_t1 -> pending_t2 -> confirmed -> closed
                |              |
                v              v
            expired        expired
```

#### Accumulation Hunter

```
No Signal -> watch -> accumulating -> ready -> triggered -> closed
              |           |            |
              v           v            v
           expired     expired      expired
```

### Known Limitations

1. **Paper trading only** - Not tested with real execution/slippage
2. **Coinbase specific** - Fee structure tuned to 0.55%
3. **Gap risk** - Real overnight gaps may exceed simulated 2% slippage
4. **Liquidity** - May not work for large position sizes on illiquid pairs

### Files Reference

| File | Description |
|------|-------------|
| `src/strategies/breakout_hunter.py` | Breakout Hunter implementation |
| `src/strategies/accumulation_hunter.py` | Accumulation Hunter implementation |
| `STRATEGY_RESEARCH.md` | Research log and backtest history |

---

## Appendix: Research Evolution

### Strategy Version History

| Version | Date | Key Change | Win Rate |
|---------|------|------------|----------|
| v3.0 | Pre-Mar 2026 | Initial approach | Unknown |
| v3.2 | Mar 14 | Tighter stops | 16% (full data) |
| v4.0 | Mar 14-15 | T+2 resumption | 53.6% |
| v5.0 | Mar 15 | Triple confirmation | 94.6% |
| v5.1 | Mar 16 | Tiered trailing | 94.6% |
| v5.2 | Mar 16-17 | T+1 entry (failed) | 33.3% |
| v5.3 | Mar 17 | Revert + enhanced exits | 94.6% |

### Key Research Insights

1. **T+1 Return is the #1 Differentiator**
   - Winners: T+1 = +4.2%
   - Losers: T+1 = -4.4%
   - Gap: +8.6%

2. **Volume Continuation Matters**
   - Winners: 43x continuation
   - Losers: 2x continuation

3. **Filtering is Key**
   - 2,076 signals detected
   - 56 passed triple confirmation
   - 94.6% of those were winners

---

*Document generated from source code and research logs. See `/Users/bz/Pythia2/STRATEGY_RESEARCH.md` for complete research history.*
