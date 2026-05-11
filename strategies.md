# Pythia2 Loading Strategy

The result of months of research, backtesting, and live paper trading. This document captures every parameter, threshold, and design decision needed to rebuild the strategy from scratch.

## Overview

Detect coins that are "loading" — accumulating volume, compressing in price, showing bot accumulation — before a large move (50%+). Enter small, confirm the move is real, then scale in. Absorb many small losses waiting for asymmetric winners.

- Target: 1-2 big movers (50%+) per week
- Universe: ~383 Coinbase-listed USD pairs
- Timeframe: intraday to 48h holds
- Capital: $5,000 paper trading (as of Apr 2026)

## Architecture

Three components run inside the integrated collector as async tasks:

1. **Loading Scanner** (`src/strategies/loading_scanner.py`) — scores every symbol every 60s
2. **Paper Trader** (`src/strategies/loading_paper_trader.py`) — manages positions
3. **FeatureBuffer** (`data/feature_buffer.db`) — SQLite with months of 1m candles, persists across restarts (no warmup needed)

State file: `data/loading_trader_state.json`

---

## Loading Score Calculation (v2 — Apr 17 2026)

Computed from a 6-hour window of 1-minute candles per symbol. Calibrated from elite movers (204 events, 50%+) vs fizzler analysis: vol_trend is #1 discriminator (5.0x elite vs 1.6x fizzlers), NATR is #2 (0.60 vs 0.40). Bot net ANTI-correlates with winners (removed from scoring).

### Hard Gate: NATR >= 0.5

Coins with NATR below 0.5 are immediately rejected (score = 0). Blocks 65% of fizzlers while keeping 53% of elite movers. Most fizzlers have NATR 0.3-0.5.

### Volume Signals (max 5.5 pts)

| Component | Formula | Condition | Points |
|-----------|---------|-----------|--------|
| Recent volume vs window avg | mean(last 15m) / mean(full window) | > 2.0x | +2.0 |
| | | > 1.5x | +1.0 |
| Volume trend | mean(last 1h) / mean(first 1h) | > 3.0x | +2.5 |
| | | > 2.0x | +1.5 |
| | | > 1.3x | +0.5 |
| Volume acceleration | mean(last 15m) / mean(prev 15m) | > 1.3x | +1.0 |
| | | > 1.1x | +0.5 |

Volume trend thresholds raised significantly — elite movers have median 5.0x vs fizzlers at 1.6x.

### Volatility Signals (max 3.5 pts)

All symbols reaching this point have NATR >= 0.5 (gate passed).

| Component | Formula | Condition | Points |
|-----------|---------|-----------|--------|
| NATR | ATR(14) / close * 100 | > 1.0 | +2.0 |
| | | > 0.7 | +1.5 |
| | | >= 0.5 (gate) | +1.0 |
| Bollinger Band width | (4 * std(20)) / SMA(20) | > 0.05 | +1.5 |
| | | > 0.03 | +1.0 |
| | | > 0.02 | +0.5 |

### Price Action Signals (max 2.0 pts)

| Component | Formula | Condition | Points |
|-----------|---------|-----------|--------|
| Close position in range | (close - low) / (high - low) | > 0.6 | +0.5 |
| Price range | (max - min) / min * 100 | > 8% | +1.0 |
| | | > 5% | +0.5 |
| 1h momentum | (close / close[-60] - 1) * 100 | 1% to 5% | +0.5 |

### Bot Accumulation — RECORD ONLY (0 pts)

Bot data is still computed and logged for analysis, but **no longer affects the score** (Apr 17 2026). Fizzler analysis showed bot_net anti-correlates with winners: fizzlers have median +7.15% bot_net vs elite movers +0.53%. 70% of fizzlers have positive bot_net vs only 48% of elite events. The buy boost was actively selecting worse trades.

### Repeat Spiker Bonus (max +2.0 pts)

Symbols that have spiked 50%+ at least twice in elite_movers.duckdb get +2.0 points. 56% of elite events are repeat spikers vs only 15% of fizzlers. List reloads from DB every hour.

### Spread Signal (max +1.5 pts)

| Spread | Points |
|--------|--------|
| >= 0.5% | +1.5 |
| >= 0.2% | +0.5 |

### Time-of-Day Signal (max +1.0 / min -0.5)

| Hour (UTC) | Points |
|------------|--------|
| 02:00-03:59 | +1.0 |
| 00:00-01:59, 20:00-21:59 | +0.5 |
| 10:00-11:59 | -0.5 |

### Minimum Price Filter

Coins below $0.0001 are skipped. Only 6 coins affected, only 2 of 190 fifty-percent movers were below this threshold.

**Max theoretical score: 16.0** (volume 5.5 + volatility 3.5 + price action 2.0 + repeat spiker 2.0 + spread 1.5 + time 1.0)

### Alert Threshold

**Score >= 9.0** fires a LOADING ALERT.

With the NATR gate, raised vol_trend thresholds, and removal of bot scoring, we expect significantly fewer but higher-quality entries.

---

## Two-Phase Position Management

### Phase 1: Probe Entry

- **Trigger:** Loading alert fires (score >= 9.0)
- **Size:** $200 (20% of $1,000 full position)
- **Max concurrent positions:** 10

**Phase 1 exits:**

| Condition | Label | Typical Loss |
|-----------|-------|-------------|
| P&L <= -3% | `phase1_stop` | -$6 |
| Loading score < 5.0 for 60 consecutive minutes | `score_fade` | -$5 to +$3 |
| 6 hours elapsed | `phase1_timeout` | varies |

### Phase 2: Scale-in (Confirmation)

- **Trigger:** Position gains 8%+ from Phase 1 entry price AND 15 minutes have passed
- **Size:** Scale to $1,000 total (add $800 at current price)
- **Entry price:** Blended weighted average of Phase 1 and Phase 2 fills

Confirmation window was reduced from 60 minutes to 15 minutes in Run 3 — faster slot cycling, still filters obvious fakes.

**Phase 2 exits:**

| Condition | Label | Notes |
|-----------|-------|-------|
| P&L <= -8% from blended entry | `phase2_stop` | Max loss ~$80 |
| Price drops 8% from peak, after gain >= 15% | `trailing_stop` | Permanently active once triggered |
| 48 hours elapsed | `time_stop` | Catch-all |

### Trailing Stop Details

- **Activation:** Once unrealized P&L >= 15% from blended entry, set `trailing_stop_active = True`
- **Trail:** 8% below the highest price seen since entry
- **Permanent:** Once activated, it stays active even if price dips below 15% gain
- **Optimized from:** 709 backtest events. 8% trail + 15% activation = best median capture (19.9%), only 2.1% negative exits

---

## The Math (Why This Works)

The strategy is designed around asymmetric payoff:

- **Phase 1 loser:** -$3 to -$15 (most common outcome)
- **Phase 2 loser:** up to -$80 (less common, need 8%+ move to enter)
- **Phase 2 winner:** +$500 to +$700 on a 50-70% move (the target)

One big winner pays for 50-100 Phase 1 scratches or 7-10 Phase 2 stops.

**Live results (first 33 hours, Apr 13-14 2026):**
- 52 Phase 1 entries, 6 Phase 2 scale-ins
- CHECK-USD: +$699 (entered Phase 1 at $0.0264, Phase 2 scale, peaked +97%, trailing stop at +69.9%)
- Net P&L: +$1,238 on $5,000 (+24.8%)

---

## Key Research Findings Behind the Design

These findings shaped every threshold and parameter choice.

### What Predicts Spikes

1. **Bot accumulation is the strongest leading signal** — 24.7x higher bot net buy before big movers vs control group. Validated on 100 events + 40 controls from LaCie data.
2. **Volume BUILDUP (gradual increase over 3-6h) is predictive** — this is what the volume trend and acceleration components measure.
3. **Price compression (low volatility) precedes breakouts** — captured by BB width.
4. **Same coins keep spiking** — 77% of slow/large events are repeat spikers. Could add symbol historical win rate as a feature.

### What Does NOT Predict Spikes

1. **Volume spikes are LAGGING** — they explode AFTER the move, not before. The v3.0 strategy based on volume > 3x had 16% win rate and lost $16K.
2. **Raw catalyst signals (whale alerts, news) are not profitable alone** — need ML filtering.
3. **Buy ratio alone doesn't discriminate** — 0.50 for both winners and losers. It's the MAGNITUDE of net accumulation that matters.
4. **Orderbook spread, market context** — no discrimination between winners and fizzles at trigger time.
5. **Whale trades before spikes are often SELLS** — counterintuitive, likely market makers providing liquidity before a move.

### False Positive Analysis (10,270 triggers studied)

At trigger time, winners and fizzles look nearly identical on most features. Best discriminators:
- Symbol historical win rate: 2.12x ratio
- NATR: 1.48x ratio
- Price (lower = better): 0.47x ratio

Best rule combo found: `gain_1h > 0 + win_rate > 15% + NATR > 0.5` = 11.3% precision (4x baseline). Still below the ~17% breakeven threshold for rule-based filtering alone.

### Spike Characteristics (from 3,554 events)

- 50%+ movers: 2.9/day avg (190 events over 65 days, the target population)
- Sub-$0.10 coins spike 3x more often than $100+ coins
- Most spikes cluster 00:00-06:00 UTC, weekends overrepresented
- T+1 return is the #1 differentiator between winners and failures

### Spike Types

1. **Fast & Steep:** Peak within 30 min, 10-25% gains
2. **Slow & Large:** Sustained over hours, 15-70%+ gains (the ones we want)
3. **Quick Fade:** Gains reverse quickly (the losers)

---

## Strategy Evolution

| Version | Logic | Win Rate | Result |
|---------|-------|----------|--------|
| v3.0 | Volume > 3x + price > 10% | 16% | -$16K (volume is lagging) |
| v4.0 | T+0 breakout, skip T+1, enter T+2 if > 8% | 53.6% | +$7,896 |
| v5.0 | T+1 positive + T+2 > 5% + vol continuation > 1x | 94.6% | +61.9% E[return], 56 trades/143 days |
| Loading v1 | Current strategy (this doc) | TBD | +$1,238 in 33h paper trading |

---

## Stop Loss Optimization (from 269 events with 50%+ gain)

| Stop % | Keeps (% of events that survive) | Breakeven Win Rate |
|--------|----------------------------------|--------------------|
| 1.5% | 35% | 5.7% |
| 3.0% | 50% | 10.7% |
| 5.0% | 65% | ~15% |

We use 3% for Phase 1 (cheap probes) and 8% for Phase 2 (give winners room).

---

## Known Issues & Edge Cases

1. **MOG-USD price precision:** Coins priced below ~$0.0001 have tick sizes representing 7%+ moves, making stop losses unreliable. Only 6 coins affected (MOG, PEPE, SHIB, BONK, FLOKI, NOICE). Only 2 of 190 fifty-percent movers were below $0.0001. Recommend filtering these out.

2. **Max positions full:** Can miss big movers when all 5 slots are occupied by mediocre Phase 1 positions (missed TROLL-USD +79% on Apr 10). The 15-min confirmation helps cycle slots faster.

3. **Phase 1 entry volume:** 28-33 entries/day. Most are small losses. This is by design — the cost of scanning.

---

## Data Infrastructure

- **FeatureBuffer:** `data/feature_buffer.db` (SQLite) — 1m candles, months of history, persists across restarts
- **Research DB:** `data/research.duckdb` (5.4GB) — 43-day consolidated dataset (Jan 31 - Apr 3, 2026)
- **Big Movers DB:** `data/big_movers.duckdb` — 605 events of 30%+ moves
- **Collector:** `src/data_ingestion/integrated_collector.py` — Coinbase WebSocket, runs on Mac Mini
- **Auto-offload:** Every 24h, archives data older than 1 day to `/Volumes/LaCie/Pythia_Archives/auto_offloads/`
- **ML Model:** `models/spike_scorer/spike_scorer_v1.joblib` — GBT, AUC 0.837 (not yet integrated into live scanner)

---

## Parameters Quick Reference

```
# Scanner
scan_interval        = 60 seconds
loading_threshold    = 9.0
natr_gate            = 0.5 (hard minimum, score=0 if below)
bot_score_threshold  = 7.0 (run bot analysis if base score >= this)
bot_scoring          = DISABLED (record only, no score impact)
min_price            = 0.0001 (skip coins below this)
spread_boost         = +1.5 (>= 0.5%), +0.5 (>= 0.2%)
time_boost           = +1.0 (02-04 UTC), +0.5 (00-02/20-22), -0.5 (10-12)
repeat_spiker_bonus  = +2.0 (from elite_movers.duckdb, reloads hourly)

# Paper Trader
full_position_size   = $1,000
max_positions        = 10
phase1_pct           = 0.20 ($200)
phase1_stop          = 0.03 (3%)
phase1_timeout       = 6 hours
score_fade_threshold = 5.0 (below this starts fade timer)
score_fade_minutes   = 60
trigger_pct          = 8.0% (gain needed for Phase 2, was 5% — too many P2 stops)
confirm_minutes      = 15
phase2_stop          = 0.08 (8%)
trail_pct            = 0.08 (8% from peak)
trail_activate       = 0.15 (15% gain to activate trail)
time_stop_hours      = 48
starting_capital     = $5,000
```
