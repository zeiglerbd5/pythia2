# Catalyst Spike Prediction System

## Overview

This system predicts cryptocurrency price spikes (10%+ moves) using catalyst signals (whale movements, exchange listings) combined with market context features.

## Architecture

```
Catalyst Sources          Feature Engineering         ML Model            Backtest
---------------          -------------------         --------            --------
Whale Alert    ─┐
Binance API    ─┼─> news_signals table ─> catalyst_features.py ─> spike_predictor.py ─> ml_backtest.py
CoinMarketCal  ─┤                              │
Twitter RSS    ─┘                              │
                                               ▼
                                        Price/Volume Context
                                        (ohlcv table)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/signals/sources/whale_alert.py` | Real-time whale transaction monitoring |
| `src/signals/sources/exchange_listings.py` | Binance listing announcements |
| `src/signals/backfill_catalysts.py` | Historical data backfill |
| `src/signals/analyze_catalyst_impact.py` | Catalyst impact analysis |
| `src/features/catalyst_features.py` | Feature engineering |
| `src/models/spike_predictor.py` | LightGBM spike prediction model |
| `src/backtesting/catalyst_backtest.py` | Raw signal backtesting |
| `src/backtesting/ml_backtest.py` | ML-enhanced backtesting |

## Key Findings

### 1. Raw Catalyst Signals Are Not Profitable
- Backtest of raw whale signals: **-6% to -16% loss**
- Stablecoins dilute signal quality
- Exchange direction theory didn't hold

### 2. Feature Importance (from LightGBM)
| Feature | Importance |
|---------|------------|
| volatility_4h | 38,849 |
| volume_ratio | 15,864 |
| log_usd_value | 14,845 |
| momentum_4h | 13,311 |
| rsi_proxy | 9,642 |
| direction_encoded | 2,162 |

**Key insight:** Market context (volatility, volume) matters more than whale direction.

### 3. ML Filtering Improves Results
| Approach | Trades | PnL |
|----------|--------|-----|
| Raw signals | 100+ | -6% to -16% |
| ML threshold 0.4 | 11 | **+1.2%** |

### 4. Spike Rate by Whale Direction
| Direction | Signals | Spike Rate |
|-----------|---------|------------|
| wallet_to_wallet | 1,288 | 5.2% |
| to_exchange | 652 | 2.6% |
| unknown | 2,586 | 2.5% |
| from_exchange | 623 | 2.1% |

**Unexpected:** Wallet-to-wallet transfers have higher spike rate than exchange flows.

## Model Performance

- **AUC:** 0.645 (better than random 0.5)
- **Precision at 0.5:** 7.4%
- **Best threshold:** 0.4 (54.5% win rate)

## Limitations

1. **Limited data:** Only 2 months of whale signals with price data
2. **Small test set:** Only 2 weeks for walk-forward validation
3. **Low precision:** 7.4% precision means many false positives
4. **No TP hits:** Take-profit (10%) may be too aggressive

## Next Steps

1. **More data:** Continue collecting signals to build larger dataset
2. **Feature engineering:**
   - Add on-chain metrics (exchange reserves, active addresses)
   - Add order book depth features
   - Add cross-asset momentum features
3. **Exit optimization:**
   - Lower take-profit targets (5-7%)
   - Trailing stops
   - Volatility-adjusted exits
4. **Model improvements:**
   - XGBoost/CatBoost comparison
   - Ensemble methods
   - Symbol-specific models

## Running the System

```bash
# 1. Build features from whale signals
python3 src/features/catalyst_features.py

# 2. Train model and run walk-forward validation
python3 src/models/spike_predictor.py

# 3. Run ML-enhanced backtest
python3 src/backtesting/ml_backtest.py

# 4. (Optional) Run raw signal backtest for comparison
python3 run_filtered_backtest.py
```

## Database Tables Used

- `news_signals`: Catalyst signals from all sources
- `ohlcv`: Price/volume data for context and forward returns
