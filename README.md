# Pythia2

**Loading-strategy crypto scalping with neural-net + RL fizzler filter.**

A real-time, single-machine system that ingests Coinbase order-book and
trade data, identifies coins that are *loading* — accumulating volume,
compressing in price, showing bot accumulation — and paper-trades them in
two phases: a small probe entry, then a scale-in when the move confirms.
A separate ML pipeline (XGBoost + CNN-LSTM, with an in-progress
reinforcement-learning policy) acts as a *fizzler filter* — rejecting
loading signals that historically faded instead of running.

> Status: research / paper-trading prototype. Not connected to a live
> trading account. The loading-scanner + paper-trader runs continuously
> on a Mac mini in a separate process; results inform offline backtests
> and RL training. Successor to a v1 system documented in the archived
> [`Pythia-`](https://github.com/zeiglerbd5/Pythia-) repo.

## The idea

Most "spike detection" approaches try to catch a coin already moving.
That's too late: by the time a 5%+ candle prints, the easy entry is gone
and the residual move is too small to overcome fees and slippage.

Pythia2 inverts this. Instead of reacting to moves, it scores *every*
Coinbase USD pair every 60 seconds on a six-hour window for signs of
accumulation — disproportionate recent volume, rising volume trend
relative to baseline, NATR floor met, bot-net activity pattern. Coins
that cross a score threshold go on a watchlist; coins that then print a
real breakout get a small probe entry; probes that confirm get scaled
into a larger position; everything else exits on a trailing stop.

The economics work out to *many small losses absorbed in exchange for a
few asymmetric winners* — 1-2 big movers (50%+) per week against ~383
candidates. The asymmetry is the whole game.

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
              │  └─ RL agent (Stable-Baselines3, in      │
              │       progress — see docs/RL_*.md)        │
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
parameter list and calibration history in [`strategies.md`](strategies.md);
short version:

| Component                       | Notes |
|---------------------------------|-------|
| **NATR hard gate ≥ 0.5**        | Rejects 65 % of fizzlers, keeps 53 % of elite movers. |
| Recent vs window volume ratio   | Up to +2.0 pts when last 15 m > 2× window mean. |
| Volume trend (1h-over-1h)       | Up to +2.5 pts; #1 discriminator (5.0× elite vs 1.6× fizzlers). |
| Price compression / breakout    | Bollinger squeeze, NATR, breakout candle. |

Bot-net activity was scored in v1 but removed in v2 after calibration —
it *anti*-correlates with eventual winners on this universe.

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

## Reinforcement learning

Parallel research track in `src/rl/`. Stable-Baselines3 agent learns
position-sizing and exit timing on top of the loading-scanner's entry
signals. Iteration history and results in `docs/RL_*.md` —
[V1 results](docs/RL_TRAINING_V1_RESULTS.md),
[V2 results](docs/RL_TRAINING_V2_RESULTS.md),
[V4 plan](docs/RL_V4_PLAN.md),
[V5 proposal](docs/RL_V5_PROPOSAL.md).

The RL agent is **not** in the live paper-trading loop yet; it trains
offline on the same backtest data the loading-scanner uses.

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
│   ├── rl/                  # Stable-Baselines3 agent + envs (in progress)
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

## What's working / what isn't

| Status | Component |
|--------|-----------|
| ✅ | Coinbase WebSocket ingest (level2, trades, ticker, heartbeats) with ES256 JWT auth |
| ✅ | DuckDB time-series store + FeatureBuffer persistence across restarts |
| ✅ | Real-time feature engine (~30-40 features) |
| ✅ | Loading scanner v2 (NATR gate, vol trend, etc.) |
| ✅ | Two-phase paper trader (probe → confirm → scale) |
| ✅ | 163-day backtester (24.7 M candles) |
| ✅ | XGBoost + CNN-LSTM fizzler filter, log-only mode |
| 🔄 | Reinforcement-learning policy (iterating, not in live loop) |
| 🔄 | Live execution wiring (paper-only today) |
| ⏳ | Promote fizzler filter from log-only to gating, once labelled set is larger |
| ⏳ | Bigger labelled training corpus from continued paper-trading |

## Notes on the v1 (`Pythia-`)

The first version of this system used a single CNN-LSTM ensemble plus an
XGBoost baseline trained on much shorter windows of data, optimised for
*detecting* a spike already in progress rather than *predicting* one
before it loads. It detected 9/10 elite movers on the best test day but
struggled with profitability because Phase 2 stop-losses bled the gains
on the winners. The loading-strategy framing is the v2 redesign that
came out of that analysis. The archived [`Pythia-`](https://github.com/zeiglerbd5/Pythia-) repo
preserves that earlier history.

## License

All rights reserved. This source is published for portfolio review and
evaluation only — no use, copying, modification, or redistribution is
permitted without written permission. See [LICENSE](LICENSE).
