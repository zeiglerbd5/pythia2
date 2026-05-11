#!/usr/bin/env python3
"""
Train XGBoost V4 - RAW DATA APPROACH

Key difference from V3: No lagging indicators (MACD, RSI, BB).
Train directly on order flow and price data.

Features:
  - Trade flow: buy/sell imbalance, trade count, large trade ratio
  - L1 data: spread, mid-price changes
  - Simple price: returns at multiple windows
  - Volume: acceleration, intensity

Target: 15%+ price increase in next 60 minutes

Hardware: M4 Mac Mini 16GB - use all resources
Approach: Start broad, iterate

Usage:
    python scripts/train_xgboost_v4_raw.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def aggregate_trades_to_1m(conn, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Aggregate raw trades into 1-minute windows with buy/sell breakdown.

    This is the core of V4 - using actual trade flow, not derived indicators.
    """
    query = f"""
    WITH trade_agg AS (
        SELECT
            symbol,
            DATE_TRUNC('minute', timestamp) as minute,
            COUNT(*) as trade_count,
            SUM(size * price) as volume_usd,
            SUM(CASE WHEN side = 'BUY' THEN size * price ELSE 0 END) as buy_volume,
            SUM(CASE WHEN side = 'SELL' THEN size * price ELSE 0 END) as sell_volume,
            AVG(size * price) as avg_trade_size,
            MAX(size * price) as max_trade_size,
            AVG(price) as avg_price,
            MIN(price) as low_price,
            MAX(price) as high_price,
            FIRST(price) as open_price,
            LAST(price) as close_price
        FROM trades
        WHERE symbol = '{symbol}'
        AND timestamp >= '{start_date}'
        AND timestamp <= '{end_date}'
        GROUP BY symbol, DATE_TRUNC('minute', timestamp)
        ORDER BY minute
    )
    SELECT * FROM trade_agg
    """
    return conn.execute(query).fetchdf()


def aggregate_tickers_to_1m(conn, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Aggregate L1 ticker data into 1-minute windows.
    """
    query = f"""
    SELECT
        symbol,
        DATE_TRUNC('minute', timestamp) as minute,
        AVG(best_bid) as avg_bid,
        AVG(best_ask) as avg_ask,
        AVG(best_ask - best_bid) as avg_spread,
        AVG((best_ask - best_bid) / NULLIF((best_ask + best_bid) / 2, 0) * 10000) as spread_bps,
        AVG(price) as avg_price,
        LAST(price) as close_price
    FROM tickers
    WHERE symbol = '{symbol}'
    AND timestamp >= '{start_date}'
    AND timestamp <= '{end_date}'
    AND best_bid IS NOT NULL
    AND best_ask IS NOT NULL
    GROUP BY symbol, DATE_TRUNC('minute', timestamp)
    ORDER BY minute
    """
    return conn.execute(query).fetchdf()


def compute_v4_features(trades_df: pd.DataFrame, tickers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute V4 features from raw trade and ticker data.

    All features are LEADING indicators based on current market state,
    not LAGGING indicators that react to past price moves.
    """
    if len(trades_df) < 20 or len(tickers_df) < 20:
        return pd.DataFrame()

    # Merge on minute
    df = pd.merge(
        trades_df,
        tickers_df[['minute', 'spread_bps', 'avg_bid', 'avg_ask']],
        on='minute',
        how='inner'
    )

    if len(df) < 20:
        return pd.DataFrame()

    df = df.sort_values('minute').reset_index(drop=True)

    # === TRADE FLOW FEATURES (Leading) ===

    # Trade imbalance: (buy - sell) / (buy + sell)
    # Positive = buying pressure, negative = selling pressure
    total_vol = df['buy_volume'] + df['sell_volume']
    df['trade_imbalance'] = (df['buy_volume'] - df['sell_volume']) / total_vol.replace(0, np.nan)

    # Rolling trade imbalance (5m, 15m windows)
    df['trade_imbalance_5m'] = df['trade_imbalance'].rolling(5, min_periods=1).mean()
    df['trade_imbalance_15m'] = df['trade_imbalance'].rolling(15, min_periods=1).mean()

    # Trade intensity (trades per minute vs average)
    avg_trades = df['trade_count'].rolling(60, min_periods=10).mean()
    df['trade_intensity'] = df['trade_count'] / avg_trades.replace(0, np.nan)

    # Trade intensity acceleration
    df['trade_intensity_5m'] = df['trade_count'].rolling(5, min_periods=1).sum()
    df['trade_intensity_15m'] = df['trade_count'].rolling(15, min_periods=1).sum()
    df['trade_accel'] = df['trade_intensity_5m'] / df['trade_intensity_15m'].replace(0, np.nan) * 3  # Normalize

    # Large trade detection (trades > 2x average size)
    avg_size = df['avg_trade_size'].rolling(60, min_periods=10).mean()
    df['large_trade_ratio'] = (df['max_trade_size'] > 2 * avg_size).astype(float)
    df['large_trade_ratio_5m'] = df['large_trade_ratio'].rolling(5, min_periods=1).mean()

    # Volume acceleration
    vol_1m = df['volume_usd']
    vol_5m = df['volume_usd'].rolling(5, min_periods=1).mean()
    vol_30m = df['volume_usd'].rolling(30, min_periods=10).mean()
    df['volume_accel_5m'] = vol_1m / vol_5m.replace(0, np.nan)
    df['volume_accel_30m'] = vol_5m / vol_30m.replace(0, np.nan)

    # === L1 FEATURES (Leading) ===

    # Spread (wider spread = less liquidity = potential for moves)
    df['spread_bps_norm'] = df['spread_bps'] / df['spread_bps'].rolling(60, min_periods=10).mean().replace(0, np.nan)

    # Spread change (narrowing spread can indicate incoming move)
    df['spread_change_5m'] = df['spread_bps'].pct_change(5)

    # === PRICE FEATURES (Simple, not derived) ===

    # Returns at multiple windows
    df['returns_1m'] = df['close_price'].pct_change(1)
    df['returns_5m'] = df['close_price'].pct_change(5)
    df['returns_15m'] = df['close_price'].pct_change(15)
    df['returns_30m'] = df['close_price'].pct_change(30)

    # Price range (volatility proxy)
    df['range_1m'] = (df['high_price'] - df['low_price']) / df['close_price']
    df['range_5m'] = df['range_1m'].rolling(5, min_periods=1).mean()

    # Price vs recent high/low
    df['high_20m'] = df['high_price'].rolling(20, min_periods=5).max()
    df['low_20m'] = df['low_price'].rolling(20, min_periods=5).min()
    df['price_position'] = (df['close_price'] - df['low_20m']) / (df['high_20m'] - df['low_20m']).replace(0, np.nan)

    # === MOMENTUM FEATURES (Simple) ===

    # Consecutive up/down minutes
    df['up_minute'] = (df['returns_1m'] > 0).astype(int)
    df['consecutive_up'] = df['up_minute'].rolling(5, min_periods=1).sum()

    # Buy pressure trend (is buying increasing?)
    df['buy_pct'] = df['buy_volume'] / total_vol.replace(0, np.nan)
    df['buy_pct_5m'] = df['buy_pct'].rolling(5, min_periods=1).mean()
    df['buy_trend'] = df['buy_pct'] - df['buy_pct'].shift(5)

    return df


def generate_spike_targets(df: pd.DataFrame,
                          price_col: str = 'close_price',
                          forward_window: int = 60,
                          min_spike: float = 0.15) -> pd.Series:
    """
    Generate binary targets for price spikes.

    Target = 1 if price rises by min_spike within forward_window minutes.
    """
    targets = pd.Series(0, index=df.index)
    prices = df[price_col].values

    for i in range(len(prices) - forward_window):
        current_price = prices[i]
        future_prices = prices[i+1:i+forward_window+1]
        max_future = np.max(future_prices)

        if (max_future - current_price) / current_price >= min_spike:
            targets.iloc[i] = 1

    return targets


def main():
    logger.info("=" * 80)
    logger.info("XGBOOST V4 - RAW DATA SPIKE DETECTION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Philosophy: Use raw trade flow and L1 data, not lagging indicators")
    logger.info("Goal: Pre-emptive spike detection with low false positive rate")
    logger.info("")

    # === CONFIGURATION ===
    DB_PATH = '/Users/bz/Pythia2/data/pythia.duckdb'
    START_DATE = '2025-12-10'
    END_DATE = '2026-01-05'

    # Target parameters
    FORWARD_WINDOW = 60  # Look 60 minutes ahead
    MIN_SPIKE = 0.15     # 15% minimum move

    # Feature columns (V4 raw features)
    FEATURE_COLS = [
        # Trade flow (leading indicators)
        'trade_imbalance', 'trade_imbalance_5m', 'trade_imbalance_15m',
        'trade_intensity', 'trade_accel',
        'large_trade_ratio_5m',
        'volume_accel_5m', 'volume_accel_30m',

        # L1 features
        'spread_bps', 'spread_bps_norm', 'spread_change_5m',

        # Price features (simple)
        'returns_1m', 'returns_5m', 'returns_15m', 'returns_30m',
        'range_5m', 'price_position',

        # Momentum
        'consecutive_up', 'buy_pct_5m', 'buy_trend'
    ]

    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Target: {MIN_SPIKE*100:.0f}%+ in {FORWARD_WINDOW} minutes")
    logger.info(f"Features: {len(FEATURE_COLS)} raw features")
    logger.info("")

    # === STEP 1: Get symbols with sufficient data ===
    logger.info("Finding symbols with sufficient trade data...")

    conn = duckdb.connect(DB_PATH, read_only=True)

    # Get symbols with at least 10k trades
    symbols_df = conn.execute(f"""
        SELECT symbol, COUNT(*) as trade_count
        FROM trades
        WHERE timestamp >= '{START_DATE}'
        AND timestamp <= '{END_DATE}'
        GROUP BY symbol
        HAVING COUNT(*) >= 10000
        ORDER BY trade_count DESC
    """).fetchdf()

    symbols = symbols_df['symbol'].tolist()
    logger.info(f"Found {len(symbols)} symbols with sufficient data")
    logger.info(f"Top 5 by trades: {symbols[:5]}")
    logger.info("")

    # === STEP 2: Build dataset ===
    logger.info("Building V4 feature dataset...")
    logger.info("This uses raw trade flow and L1 data, NOT derived indicators")
    logger.info("")

    all_features = []
    all_targets = []
    all_symbols = []
    positive_count = 0

    for i, symbol in enumerate(symbols):
        if (i + 1) % 20 == 0:
            logger.info(f"Processing {i+1}/{len(symbols)} symbols... ({positive_count} positives)")

        try:
            # Get aggregated trades
            trades_df = aggregate_trades_to_1m(conn, symbol, START_DATE, END_DATE)
            if len(trades_df) < 100:
                continue

            # Get aggregated tickers
            tickers_df = aggregate_tickers_to_1m(conn, symbol, START_DATE, END_DATE)
            if len(tickers_df) < 100:
                continue

            # Compute V4 features
            features_df = compute_v4_features(trades_df, tickers_df)
            if len(features_df) < 100:
                continue

            # Generate targets
            targets = generate_spike_targets(
                features_df,
                forward_window=FORWARD_WINDOW,
                min_spike=MIN_SPIKE
            )

            n_positives = int(targets.sum())

            # Extract feature matrix
            X = features_df[FEATURE_COLS].values
            y = targets.values

            # Handle NaN/inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            all_features.append(X)
            all_targets.append(y)
            all_symbols.extend([symbol] * len(y))
            positive_count += n_positives

        except Exception as e:
            logger.warning(f"Error processing {symbol}: {e}")
            continue

    conn.close()

    if positive_count == 0:
        logger.error("No positive samples found!")
        return

    # === STEP 3: Create training dataset ===
    logger.info("")
    logger.info("Creating training dataset...")

    X = np.vstack(all_features)
    y = np.concatenate(all_targets)

    logger.info(f"Total samples: {len(X):,}")
    logger.info(f"Positive samples: {int(y.sum()):,} ({y.mean()*100:.3f}%)")
    logger.info(f"Feature shape: {X.shape}")
    logger.info("")

    # === STEP 4: Train/test split ===
    logger.info("Splitting train/test (80/20)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train):,} samples ({y_train.mean()*100:.3f}% positive)")
    logger.info(f"Test: {len(X_test):,} samples ({y_test.mean()*100:.3f}% positive)")
    logger.info("")

    # === STEP 5: Train XGBoost ===
    logger.info("Training XGBoost V4...")
    logger.info("Using all CPU cores on M4 Mac Mini")
    logger.info("")

    # Calculate class weight for imbalanced data
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    logger.info(f"Class imbalance ratio: {scale_pos_weight:.1f}:1")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,  # Use all CPU cores
        random_state=42,
        eval_metric='aucpr',
        early_stopping_rounds=50
    )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # === STEP 6: Evaluate ===
    logger.info("")
    logger.info("=" * 40)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 40)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall: {recall:.3f}")
    logger.info(f"F1 Score: {f1:.3f}")
    logger.info("")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:")
    logger.info(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    logger.info(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    logger.info("")

    # High-confidence predictions (what we care about for trading)
    logger.info("High-confidence threshold analysis:")
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        if y_pred_thresh.sum() > 0:
            prec = precision_score(y_test, y_pred_thresh, zero_division=0)
            rec = recall_score(y_test, y_pred_thresh, zero_division=0)
            n_signals = y_pred_thresh.sum()
            logger.info(f"  Threshold {thresh}: Precision={prec:.3f}, Recall={rec:.3f}, Signals={n_signals:,}")
    logger.info("")

    # Feature importance
    logger.info("Feature Importance (Top 10):")
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    for i in range(min(10, len(FEATURE_COLS))):
        idx = indices[i]
        logger.info(f"  {i+1}. {FEATURE_COLS[idx]}: {importance[idx]:.4f}")
    logger.info("")

    # === STEP 7: Save model ===
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_v4_raw.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        'model': model,
        'feature_cols': FEATURE_COLS,
        'params': {
            'forward_window': FORWARD_WINDOW,
            'min_spike': MIN_SPIKE,
            'start_date': START_DATE,
            'end_date': END_DATE
        }
    }, model_path)

    logger.info(f"Model saved to: {model_path}")
    logger.info("")
    logger.info("V4 Training complete!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Integrate into feature_engine.py for live predictions")
    logger.info("  2. Compare V4 signals to V3 signals on recent movers")
    logger.info("  3. Paper trade to evaluate real-world performance")


if __name__ == "__main__":
    main()
