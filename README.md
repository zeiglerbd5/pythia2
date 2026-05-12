# Pythia2

**Loading-strategy crypto scalping research platform with neural-net + RL fizzler filter.**

A research-grade, single-machine system that ingests Coinbase order-book and
trade data, identifies coins that are *loading* — accumulating volume,
compressing in price, showing pre-breakout structure — and paper-trades them in
two phases: a small probe entry, then a scale-in when the move confirms.
A separate ML track (XGBoost + CNN-LSTM, with an in-progress reinforcement
learning policy) acts as a *fizzler filter* — rejecting loading signals
that historically faded instead of running.

> **Status: research platform, not a finished product.** This repository
> documents an iterative exploration of a single hypothesis — that pre-breakout
> accumulation patterns are detectable in real-time market data — and the
> supporting machine-learning work that filters those signals. The system has
> never been connected to a live trading account; everything below is paper
> trading or offline backtest. Aggregate, out-of-sample, multi-regime strategy
> performance is deliberately not headlined: samples are small, the backtest
> regime is a single bull cycle, and what's reported in [Results](#results)
> is component-level measurement, not a productionised strategy. Successor to
> a v1 system in the archived [`Pythia-`](https://github.com/zeiglerbd5/Pythia-)
> repo.

## What's working / what isn't

| Status | Component |
|--------|-----------|
| ✅ | Coinbase WebSocket ingest (level2, trades, ticker, heartbeats) with ES256 JWT auth |
| ✅ | DuckDB time-series store + FeatureBuffer persistence across restarts |
| ✅ | Real-time feature engine (~30-40 features) |
| ✅ | Loading scanner v2 (NATR gate, vol-trend discriminator, etc.) |
| ✅ | Two-phase paper trader (probe → confirm → scale) |
| ✅ | 163-day offline backtester (~24.7 M candles) |
| ✅ | XGBoost + CNN-LSTM fizzler filter, **log-only mode** (records predictions, does not yet gate execution) |
| 🔄 | Reinforcement-learning position sizer — iterating offline, not in the live loop |
| 🔄 | Live execution wiring — paper-only today |
| ⏳ | Larger labelled training corpus from continued paper-trading |
| ⏳ | Out-of-sample / multi-regime backtest validation |
| ⏳ | Promote fizzler filter from log-only to gating |

## Results

What has actually been measured, with explicit caveats. None of these
numbers should be read as validated end-to-end strategy performance.

### Component-level measurements (loading scanner)

These are calibrated against ~24.7 M 1-minute candles spanning 163 days
across the Coinbase USD universe. They are **in-sample**: the gates were
tuned on this same window, so the numbers describe selectivity rather
than predictive generalisation.

| Component | Measurement |
|---|---|
| NATR hard gate (≥ 0.5) | Rejects 65 % of fizzlers; retains 53 % of elite movers |
| Volume-trend (1h-over-1h) ratio at signal time | 5.0× for elite movers vs 1.6× for fizzlers — the strongest single discriminator |
| Recent vs window volume ratio | Up to +2.0 score points when last 15 m > 2× window mean |
| Bot-net activity score (v1) | Removed in v2 after measurement — *anti-*correlates with eventual winners on this universe |

### Catalyst-signal research — a deliberately reported negative result

A separate research track tried to use external catalyst signals (whale
transfers, exchange listings) to time entries. Documented in
[`docs/catalyst_spike_prediction.md`](docs/catalyst_spike_prediction.md):

| Approach | Trades | PnL | Notes |
|---|---|---|---|
| Raw whale-transfer signals | 100+ | **−6 % to −16 %** | Stablecoins dilute signal; exchange-direction theory did not hold up |
| LightGBM filter, threshold 0.4 | 11 | +1.2 % | Tiny N; AUC 0.645; precision-at-0.5 only 7.4 % |

The most useful output of this track was the discovery that *market
context features* (4-hour volatility, volume ratio) dominate the model's
importance ranking — whale direction itself is barely informative.

### Iteration discipline — RL position sizer (offline, small N)

Iteration v3 → v4 of the RL agent, measured on the same backtest window.
Numbers are for the *position-sizing* policy applied on top of fixed
entry signals; they are **not** end-to-end strategy returns.

| Metric | v3 | v4 | Δ |
|---|---|---|---|
| Trades | 357 | 94 | quality-over-quantity |
| Win rate | 54.3 % | 70.2 % | +15.9 pp |
| Profit factor | 5.54 | 10.61 | +92 % |
| Avg PnL / trade | $0.75 | $2.94 | +293 % |
| Avg hold time | ~1 bar | 6.8 bars | learned to hold |

v4 became more selective by weighting momentum and enforcing a minimum
hold time. Trade counts are small and the dataset is a single market
regime; treat this as evidence the training loop works, not as
production performance.

### Archive — Breakout Hunter v5.3 reactive strategy

An earlier, *reactive* strategy (the opposite of the loading-scanner
thesis: react to a confirmed breakout rather than anticipate one) was
backtested over 143 days, Oct 2025 – Mar 2026:

| Metric | Value |
|---|---|
| Total trades | 56 |
| Win rate | 94.6 % |
| Expected return / trade | +61.9 % |
| Signals filtered (triple-confirmation gate) | 97.3 % of 2,076 raw signals |

These numbers look extremely strong and they should also be read with
extreme caution: N=56, single bull-market regime, in-sample parameter
tuning, no live execution. The strategy is preserved in the codebase
mainly because the *T+1-entry-fails* finding it produced (33 % win rate
when entering at T+1 vs 95 % at T+2 — see
[`docs/BREAKOUT_DETECTION_STRATEGIES.md`](docs/BREAKOUT_DETECTION_STRATEGIES.md))
directly motivated the loading-scanner redesign in v2.

### What is not validated yet

- End-to-end strategy Sharpe / Sortino / max drawdown
- Out-of-sample performance on a held-out window
- Performance across regimes (only a bull cycle is in the dataset)
- Real fills, slippage, fees against a live order book
- Promotion of the fizzler filter from log-only to execution-gating

These are the open questions the platform exists to answer.

## The idea

Most "spike detection" approaches try to catch a coin already moving.
That's too late: by the time a 5%+ candle prints, the easy entry is gone
and the residual move is too small to overcome fees and slippage.

Pythia2 inverts this. Instead of reacting to moves, it scores *every*
Coinbase USD pair every 60 seconds on a six-hour window for signs of
accumulation — disproportionate recent volume, rising volume trend
relative to baseline, NATR floor met. Coins that cross a score threshold
go on a watchlist; coins that then print a real breakout get a small
probe entry; probes that confirm get scaled into a larger position;
everything else exits on a trailing stop.

The hypothesis being tested is *many small losses absorbed in exchange
for a few asymmetric winners* — 1-2 big movers (50 %+) per week against
~383 candidates. Whether the asymmetry holds out-of-sample is exactly
what the open validation work (above) is for.

## Architecture

```
                       Coinbase WebSocket
                              │
                              ▼
              ┌────────── Integrated Collector ──────────┐
              │  (asyncio, runs continuously)            │
              │                                          │
              │  ┌─ Order-book engine                    │
              │  ├─ Trade & ticker ingest                │
              │  ├─ Feature engine (RSI, VPIN, Roll,     │
              │  │      OBV, VROC, NATR, …)              │
              │  ├─ Loading Scanner (every 60s)          │
              │  ├─ Paper Trader (two-phase entries)     │
              │  └─ FeatureBuffer (SQLite, persists)     │
              └──────────────────────────────────────────┘
                              │
                              ▼
            DuckDB time-series + JSON state files
                              │
                              ▼
              ┌────────── Offline pipelines ─────────────┐
              │  ├─ Backtester (163 days, 24.7M candles) │
              │  ├─ Fizzler filter training              │
              │  │      (XGBoost + CNN-LSTM)             │
              │  └─ RL position sizer (offline only —    │
              │       see docs/RL_*.md)                  │
              └──────────────────────────────────────────┘
```

### Top-level entry points

- `run_production_backtest.py` — main backtester
- `run_filtered_backtest.py` — backtester with fizzler-filter enabled
- `strategy_dashboard.py` — terminal dashboard for live paper trades
- `paper_trade_visualizer.py` — chart trades from `paper_trades.py` output
- `bad_entry_filter.py` — standalone fizzler-filter sanity check

The continuous collector + scanner + paper-trader lives under
`src/data_ingestion/` and `src/strategies/`.

## Loading score

Computed per symbol from a 6-hour window of 1-minute candles. Full
parameter list and calibration history in [`strategies.md`](strategies.md).
The headline component-level numbers are summarised in
[Results](#component-level-measurements-loading-scanner) above.

## Fizzler filter

Trained on labelled outcomes from 163 days of historical 1-minute candles
(~24.7 M rows) across the active universe.

- `xgboost_slow_large_v3_model.pkl` — gradient-boosted baseline.
- `models/fizzler_filter_v2/` — CNN-LSTM trained on order-book + price
  features. Currently runs in **log-only mode** — its predictions are
  recorded alongside paper-trade entries but don't yet gate execution,
  pending a larger labelled training set.
- `models/spike_scorer/` — separate signal-quality scorer used during
  backtests.

## Reinforcement learning (experimental)

A parallel research track in `src/rl/` trains a Stable-Baselines3 agent
to learn position sizing and exit timing on top of the loading-scanner's
entry signals. The agent is **not** in the live paper-trading loop. It
trains offline on the same backtest data the loading-scanner uses, and
the iteration log is in [`docs/RL_TRAINING_V1_RESULTS.md`](docs/RL_TRAINING_V1_RESULTS.md),
[`V2`](docs/RL_TRAINING_V2_RESULTS.md),
[`V4 plan`](docs/RL_V4_PLAN.md),
and [`V5 proposal`](docs/RL_V5_PROPOSAL.md). v3 → v4 numbers are in
[Results](#iteration-discipline--rl-position-sizer-offline-small-n).

## Running it

### Prerequisites

- Python 3.10+
- macOS with Apple Silicon (PyTorch MPS) or Linux with CUDA
- Coinbase Advanced Trade API key (read-only is enough for paper trading)
- TA-Lib system library: `brew install ta-lib` (macOS) or `apt-get install ta-lib`

### Setup

```bash
git clone https://github.com/zeiglerbd5/pythia2.git
cd pythia2

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# Optional: RL components
pip install -r requirements_rl.txt

cp config/.env.example .env
# edit .env with Coinbase API credentials
```

### Common commands

```bash
# Continuous collector + scanner + paper trader (the main loop)
python -m src.data_ingestion.integrated_collector

# Live terminal dashboard (in a second shell)
python strategy_dashboard.py

# Offline backtest on the 163-day dataset
python run_production_backtest.py

# Backtest with the fizzler filter active
python run_filtered_backtest.py
```

## Repository layout

```
pythia2/
├── src/
│   ├── data_ingestion/      # Coinbase WebSocket, order book, integrated collector
│   ├── features/            # Microstructure + price + volume feature engine
│   ├── strategies/          # Loading scanner, paper trader
│   ├── signals/             # Spike detectors, ensemble voting
│   ├── models/              # CNN-LSTM and XGBoost wrappers
│   ├── inference/           # Live-inference glue (loads .pt / .pkl, scales features)
│   ├── execution/           # Order-routing stubs (paper today)
│   ├── risk/                # Position sizing, stops, exposure caps
│   ├── monitoring/          # Health checks, alerts
│   ├── backtesting/         # Offline replay over historical candles
│   ├── rl/                  # Stable-Baselines3 agent + envs (experimental)
│   ├── visualization/       # Charting helpers
│   └── utils/
├── docs/                    # Strategy + RL design notes
├── scripts/                 # One-off analysis and training scripts
├── tests/                   # pytest suite (currently RL-focused)
├── models/                  # Saved model artifacts (.pt, .pkl, scalers)
├── config/                  # config.yaml, .env.example
├── notebooks/               # Exploratory Jupyter work
└── strategies.md            # Authoritative parameter / calibration record
```

## Notes on the v1 (`Pythia-`)

The first version of this system used a single CNN-LSTM ensemble plus an
XGBoost baseline trained on much shorter windows of data, optimised for
*detecting* a spike already in progress rather than *predicting* one
before it loads. It detected 9/10 elite movers on the best test day but
struggled with profitability because Phase-2 stop-losses bled the gains
on the winners. The loading-strategy framing is the v2 redesign that
came out of that analysis. The archived [`Pythia-`](https://github.com/zeiglerbd5/Pythia-) repo
preserves that earlier history.

## License

Released under the [PolyForm Noncommercial License 1.0.0](LICENSE).
You may read, fork, modify, and use the code for personal study,
non-commercial research, and educational purposes. Commercial use of
the software (including running it as part of any revenue-generating
trading activity) requires a separate license — contact the author.
