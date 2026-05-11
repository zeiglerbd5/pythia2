# Pythia Training Scripts

Utility scripts for preparing data and training models.

## Data Requirements

To train the spike detection models, you need:

- **Minimum:** 81+ days of continuous data
  - 60 days for sequence lookback
  - 14 days for forward spike detection
  - 7+ days for validation/test sets

- **Data types:**
  - OHLCV candles (1m, 5m, 15m)
  - Calculated features (~30-40 after Boruta)
  - Order book snapshots (for microstructure features)

## Scripts

### 1. Check Data Readiness

Verify you have enough data to train models:

```bash
python scripts/check_data_readiness.py --db data/pythia.duckdb
```

**What it checks:**
- Database exists and has required tables
- OHLCV coverage per symbol/timeframe
- Feature calculation completion
- Estimated number of training samples
- Overall readiness status

**Output:**
- ✓ Ready to train
- ✗ Not ready (with specific reasons and next steps)

### 2. Migrate SQLite to DuckDB

Convert existing SQLite database to DuckDB format:

```bash
python scripts/migrate_sqlite_to_duckdb.py \
  --input /path/to/your/data.db \
  --output data/pythia.duckdb
```

**Options:**
- `--input`: SQLite database path (required)
- `--output`: DuckDB database path (required)
- `--tables`: Specific tables to migrate (optional, default: all)
- `--batch-size`: Records per batch (default: 10000)

**Performance:**
- Handles large databases (40GB+)
- Batched processing to manage memory
- Progress tracking

### 3. Backfill Historical Data

Fetch historical candles from Coinbase API:

```bash
python scripts/backfill_historical_data.py \
  --days 90 \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  --granularities 300
```

**Options:**
- `--days`: Days to backfill (default: 90)
- `--symbols`: Comma-separated symbols (default: BTC-USD,ETH-USD)
- `--granularities`: Candle sizes in seconds (60=1m, 300=5m, 900=15m)

**Notes:**
- Fetches up to 300 candles per request (Coinbase limit)
- Automatically handles pagination
- Rate limited to 5 requests/second
- Writes directly to DuckDB

**Estimated time:**
- 90 days @ 5m candles: ~5-10 minutes per symbol
- Multiple symbols run sequentially

### 4. Train Model

Train a single spike detection model:

```bash
python scripts/train_model.py \
  --symbol BTC-USD \
  --days 90 \
  --timeframe 5m
```

**Options:**
- `--symbol`: Trading pair (default: BTC-USD)
- `--days`: Days of data to use (default: 90)
- `--timeframe`: Timeframe (1m, 5m, 15m; default: 5m)
- `--no-smote`: Disable SMOTE oversampling
- `--smote-ratio`: SMOTE target ratio (default: 0.2 = 1:5)

**What it does:**
1. Loads data from DuckDB
2. Generates spike targets (7-14 day forward)
3. Creates sequences (60-day lookback)
4. Splits train/val/test (70/15/15)
5. Normalizes features (RobustScaler)
6. Applies SMOTE oversampling (1:5 ratio)
7. Trains CNN-LSTM with Focal Loss
8. Evaluates on test set
9. Saves model checkpoints and metrics

**Training time:**
- CPU: 2-4 hours
- MPS (Apple Silicon): 1-2 hours
- CUDA: 30-60 minutes

**Output:**
- Model checkpoints: `models/{symbol}_{timeframe}_{timestamp}/`
- Best model: `best_model.pt`
- Training history: `training_history.json`
- Evaluation report: `evaluation.json`

## Typical Workflow

### Scenario 1: You Have 10 Days of SQLite Data (Current)

```bash
# Step 1: Migrate to DuckDB
python scripts/migrate_sqlite_to_duckdb.py \
  --input /path/to/your/data.db \
  --output data/pythia.duckdb

# Step 2: Backfill 80 more days of historical data
python scripts/backfill_historical_data.py \
  --days 90 \
  --symbols BTC-USD,ETH-USD

# Step 3: Calculate features (if not already done)
python src/data_ingestion/integrated_collector.py
# Let it run for a few minutes to calculate features on historical data

# Step 4: Check readiness
python scripts/check_data_readiness.py

# Step 5: Train model (if ready)
python scripts/train_model.py --symbol BTC-USD
```

### Scenario 2: Starting Fresh with Pythia

```bash
# Step 1: Start collecting data
python src/data_ingestion/integrated_collector.py
# Let run continuously

# Step 2: Backfill historical data while collecting
python scripts/backfill_historical_data.py --days 90

# Step 3: Wait until you have 81+ days total

# Step 4: Train model
python scripts/train_model.py --symbol BTC-USD
```

### Scenario 3: You Already Have 90+ Days

```bash
# Step 1: Check readiness
python scripts/check_data_readiness.py

# Step 2: Train immediately
python scripts/train_model.py --symbol BTC-USD
```

## Target Metrics

Per implementation guide, models should achieve:

**Classification:**
- Accuracy: ≥ 82.44%
- Precision: ≥ 90%
- F1: ≥ 0.80

**Trading:**
- Sharpe Ratio: 1.3 - 1.8
- Win Rate: ≥ 60%

If your model doesn't meet these targets:
1. Collect more data (more symbols, longer timeframe)
2. Tune hyperparameters
3. Try different feature sets
4. Use ensemble methods (train 3-5 models)

## Troubleshooting

**"Not enough data" error:**
- Need 81+ days minimum
- Check with: `python scripts/check_data_readiness.py`
- Backfill historical data

**"Features not calculated" error:**
- Run integrated collector to calculate features
- Features are calculated real-time on WebSocket data

**"Database not found" error:**
- Check database path in `.env` file
- Ensure `DATA_DIR` is set correctly
- Run data collector first

**Training is slow:**
- Use `device='mps'` for Apple Silicon
- Use `device='cuda'` for NVIDIA GPUs
- Reduce batch size if memory issues
- Use GRU model (60% faster than LSTM)

**Poor model performance:**
- Collect more data (90+ days better than 81)
- Try different symbols (some are easier to predict)
- Adjust spike threshold (try 10%, 15%, 20%)
- Use ensemble of 3-5 models
- Check class imbalance (should be 1-5% positive)

## Next Steps After Training

Once you have a trained model with good metrics:

1. **Train Ensemble:**
   - Train 3-5 models with different:
     - Data splits (walk-forward validation)
     - Hyperparameters
     - Random seeds
   - Combine with Sharpe-weighted voting

2. **Phase 4: Signal Generation**
   - Implement signal filters
   - Multi-timeframe confirmation
   - Confidence thresholds

3. **Phase 5: Risk Management**
   - Position sizing
   - Stop loss placement
   - Portfolio limits

4. **Phase 6: Paper Trading**
   - Test with paper money
   - Monitor performance
   - Iterate on strategy
