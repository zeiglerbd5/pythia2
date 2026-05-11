#!/usr/bin/env python3
"""
Fast Backtest Event Classifier Strategies (Vectorized)

Uses vectorized pandas operations for speed.
Tests on a sample of symbols to get quick results.

Usage:
    python scripts/backtest_event_classifier_fast.py --sweep
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import argparse
import sqlite3
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


FEATURE_BUFFER_PATH = '/Users/bz/Pythia2/data/feature_buffer.db'

DEFAULT_MODELS = [
    '/Users/bz/Pythia2/models/event_classifier_xgb.pkl',
    '/Users/bz/Pythia2/models/xgboost_full_dataset/model.pkl',
]

FEATURE_COLS = [
    'natr_14', 'bb_width_20', 'bb_position', 'rsi_14',
    'returns_1hr', 'returns_6hr', 'returns_12hr',
    'momentum_5', 'momentum_20',
    'dist_from_24hr_high', 'dist_from_24hr_low',
    'hl_range', 'body_ratio_avg_1hr', 'range_compression',
    'volume_vs_ma20', 'volume_trend_6hr', 'obv_slope_1hr', 'vroc_12',
    'vol_ratio_20_60', 'vol_ratio_60_240', 'vol_acceleration', 'vol_price_divergence',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
]


@dataclass
class BacktestConfig:
    model_path: str
    threshold: float = 0.80
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.10
    trailing_stop_pct: float = 0.02
    trailing_activation_pct: float = 0.05
    max_hold_minutes: int = 180
    position_size: float = 2500.0
    max_positions: int = 4
    min_volume_ratio: float = 1.0


def load_model(model_path: str):
    """Load model and return model, scaler, feature_cols"""
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict):
        return model_data['model'], model_data.get('scaler'), model_data.get('feature_cols', FEATURE_COLS)
    return model_data, None, FEATURE_COLS


def compute_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features using vectorized operations"""
    df = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_ = df['open']

    # NATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    df['natr_14'] = (atr_14 / close) * 100

    # Bollinger Bands
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df['bb_width_20'] = (2 * std_20 / sma_20)
    df['bb_position'] = (close - (sma_20 - 2*std_20)) / (4*std_20 + 1e-10)

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Returns
    df['returns_1hr'] = close / close.shift(60) - 1
    df['returns_6hr'] = close / close.shift(360) - 1
    df['returns_12hr'] = close / close.shift(720) - 1
    df['momentum_5'] = close / close.shift(5) - 1
    df['momentum_20'] = close / close.shift(20) - 1

    # Price position
    high_24hr = high.rolling(1440).max()
    low_24hr = low.rolling(1440).min()
    df['dist_from_24hr_high'] = (close - high_24hr) / (high_24hr + 1e-10)
    df['dist_from_24hr_low'] = (close - low_24hr) / (low_24hr + 1e-10)
    df['hl_range'] = (high_24hr - low_24hr) / (close + 1e-10)

    # Body ratio
    body = (close - open_).abs()
    wick = high - low + 1e-10
    body_ratio = body / wick
    df['body_ratio_avg_1hr'] = body_ratio.rolling(60).mean()

    # Range compression
    range_20 = (high.rolling(20).max() - low.rolling(20).min()) / (close + 1e-10)
    range_60 = (high.rolling(60).max() - low.rolling(60).min()) / (close + 1e-10)
    df['range_compression'] = range_20 / (range_60 + 1e-10)

    # Volume features
    vol_ma_20 = volume.rolling(20).mean()
    vol_ma_60 = volume.rolling(60).mean()
    vol_ma_240 = volume.rolling(240).mean()
    vol_ma_360 = volume.rolling(360).mean()

    df['volume_vs_ma20'] = volume / (vol_ma_20 + 1e-10)
    df['volume_trend_6hr'] = vol_ma_60 / (vol_ma_360 + 1e-10)

    # OBV slope
    obv = (volume * np.sign(close.diff())).cumsum()
    df['obv_slope_1hr'] = (obv - obv.shift(60)) / 60

    # VROC
    df['vroc_12'] = volume / (volume.shift(12) + 1e-10) - 1

    # Volume ratios
    df['vol_ratio_20_60'] = vol_ma_20 / (vol_ma_60 + 1e-10)
    df['vol_ratio_60_240'] = vol_ma_60 / (vol_ma_240 + 1e-10)

    vol_5 = volume.rolling(5).mean()
    vol_10 = volume.rolling(10).mean()
    df['vol_acceleration'] = vol_5 / (vol_10 + 1e-10) - 1

    # Vol-price divergence
    vol_change = vol_ma_60 / (vol_ma_360 + 1e-10) - 1
    df['vol_price_divergence'] = vol_change - df['returns_1hr']

    # Time features
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)

    # Fill NaN
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def run_backtest(config: BacktestConfig, features_df: pd.DataFrame) -> Dict:
    """Run backtest simulation"""
    model, scaler, feature_cols = load_model(config.model_path)

    trades = []
    open_positions = {}  # symbol -> position dict

    # Sort by timestamp
    features_df = features_df.sort_values('timestamp')
    timestamps = features_df['timestamp'].unique()

    for ts in timestamps:
        ts_data = features_df[features_df['timestamp'] == ts]

        # Check exits for open positions
        symbols_to_close = []
        for symbol, pos in open_positions.items():
            sym_data = ts_data[ts_data['symbol'] == symbol]
            if len(sym_data) == 0:
                continue

            current_price = sym_data['close'].iloc[0]
            pos['peak_price'] = max(pos['peak_price'], current_price)

            pct_change = (current_price - pos['entry_price']) / pos['entry_price']
            peak_pct = (pos['peak_price'] - pos['entry_price']) / pos['entry_price']
            hold_minutes = (ts - pos['entry_time']).total_seconds() / 60

            exit_reason = None
            exit_price = current_price

            if pct_change <= -config.stop_loss_pct:
                exit_reason = "stop_loss"
                exit_price = pos['entry_price'] * (1 - config.stop_loss_pct)
            elif pct_change >= config.take_profit_pct:
                exit_reason = "take_profit"
            elif hold_minutes >= config.max_hold_minutes:
                exit_reason = "max_hold"
            elif peak_pct >= config.trailing_activation_pct:
                trailing_stop = pos['peak_price'] * (1 - config.trailing_stop_pct)
                if current_price <= trailing_stop:
                    exit_reason = "trailing_stop"
                    exit_price = trailing_stop

            if exit_reason:
                pnl = pos['quantity'] * (exit_price - pos['entry_price'])
                trades.append({
                    'symbol': symbol,
                    'entry_price': pos['entry_price'],
                    'entry_time': pos['entry_time'],
                    'exit_price': exit_price,
                    'exit_time': ts,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'pct_return': pct_change * 100
                })
                symbols_to_close.append(symbol)

        for symbol in symbols_to_close:
            del open_positions[symbol]

        # Check entries
        if len(open_positions) < config.max_positions:
            for _, row in ts_data.iterrows():
                if len(open_positions) >= config.max_positions:
                    break

                symbol = row['symbol']
                if symbol in open_positions:
                    continue

                # Volume filter
                if row.get('vol_ratio_20_60', 0) < config.min_volume_ratio:
                    continue

                # Get prediction
                X = np.array([[row.get(col, 0) for col in feature_cols]])
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                if scaler:
                    X = scaler.transform(X)

                try:
                    prob = model.predict_proba(X)[0][1]
                except:
                    continue

                if prob >= config.threshold:
                    entry_price = row['close']
                    quantity = config.position_size / entry_price
                    open_positions[symbol] = {
                        'entry_price': entry_price,
                        'entry_time': ts,
                        'quantity': quantity,
                        'peak_price': entry_price
                    }

    # Close remaining positions
    last_ts = timestamps[-1]
    for symbol, pos in open_positions.items():
        last_data = features_df[(features_df['symbol'] == symbol) & (features_df['timestamp'] == last_ts)]
        if len(last_data) > 0:
            exit_price = last_data['close'].iloc[0]
        else:
            exit_price = pos['entry_price']

        pnl = pos['quantity'] * (exit_price - pos['entry_price'])
        pct_return = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
        trades.append({
            'symbol': symbol,
            'entry_price': pos['entry_price'],
            'entry_time': pos['entry_time'],
            'exit_price': exit_price,
            'exit_time': last_ts,
            'exit_reason': 'end',
            'pnl': pnl,
            'pct_return': pct_return
        })

    # Calculate metrics
    if not trades:
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0}

    total_pnl = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    return {
        'trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / len(trades) if trades else 0,
        'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
        'avg_loss': np.mean([t['pnl'] for t in losses]) if losses else 0,
        'trade_details': trades[:20]  # Keep sample of trades
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    parser.add_argument('--symbols', type=int, default=50, help='Number of symbols to test')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("FAST EVENT CLASSIFIER BACKTEST")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading OHLCV data...")
    conn = sqlite3.connect(FEATURE_BUFFER_PATH)

    # Get top symbols by data count
    symbol_counts = pd.read_sql_query(
        "SELECT symbol, COUNT(*) as cnt FROM ohlcv GROUP BY symbol ORDER BY cnt DESC",
        conn
    )
    top_symbols = symbol_counts.head(args.symbols)['symbol'].tolist()
    logger.info(f"Testing on {len(top_symbols)} symbols with most data")

    symbols_str = "', '".join(top_symbols)
    df = pd.read_sql_query(
        f"SELECT * FROM ohlcv WHERE symbol IN ('{symbols_str}') ORDER BY symbol, timestamp",
        conn
    )
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    logger.info(f"Loaded {len(df)} rows")

    # Compute features per symbol
    logger.info("\nComputing features (vectorized)...")
    all_features = []
    for symbol in top_symbols:
        sym_df = df[df['symbol'] == symbol].copy()
        if len(sym_df) < 1500:
            continue
        sym_features = compute_features_vectorized(sym_df)
        sym_features = sym_features.iloc[1440:]  # Skip warmup
        sym_features['symbol'] = symbol
        all_features.append(sym_features)

    if not all_features:
        logger.error("No features computed!")
        return

    features_df = pd.concat(all_features, ignore_index=True)
    logger.info(f"Features computed: {len(features_df)} rows, {features_df['symbol'].nunique()} symbols")

    # Check models
    models = [m for m in DEFAULT_MODELS if Path(m).exists()]
    if not models:
        logger.error("No models found!")
        return

    if args.sweep:
        logger.info("\nRunning parameter sweep...")

        results = []
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
        stop_losses = [0.01, 0.02, 0.03]
        take_profits = [0.05, 0.10, 0.15]
        vol_ratios = [1.0, 1.5, 2.0]

        total = len(models) * len(thresholds) * len(stop_losses) * len(take_profits) * len(vol_ratios)
        logger.info(f"Testing {total} configurations...")

        count = 0
        for model_path in models:
            model_name = Path(model_path).name
            for thresh in thresholds:
                for sl in stop_losses:
                    for tp in take_profits:
                        for vr in vol_ratios:
                            count += 1
                            config = BacktestConfig(
                                model_path=model_path,
                                threshold=thresh,
                                stop_loss_pct=sl,
                                take_profit_pct=tp,
                                min_volume_ratio=vr
                            )

                            try:
                                result = run_backtest(config, features_df)
                                results.append({
                                    'model': model_name,
                                    'threshold': thresh,
                                    'stop_loss': sl,
                                    'take_profit': tp,
                                    'vol_ratio': vr,
                                    **{k: v for k, v in result.items() if k != 'trade_details'}
                                })
                            except Exception as e:
                                logger.warning(f"Config failed: {e}")

                            if count % 50 == 0:
                                logger.info(f"  Progress: {count}/{total}")

        # Print results
        results_df = pd.DataFrame(results)
        results_df = results_df[results_df['trades'] >= 5]

        if len(results_df) == 0:
            logger.warning("No configurations had >= 5 trades")
            return

        print("\n" + "=" * 130)
        print("TOP 20 CONFIGURATIONS BY P&L (min 5 trades)")
        print("=" * 130)
        print(f"{'Model':<30} {'Thresh':>6} {'SL':>5} {'TP':>5} {'VolR':>5} {'Trades':>6} {'WinR':>6} {'P&L':>10} {'AvgPnL':>8} {'AvgWin':>8} {'AvgLoss':>8}")
        print("-" * 130)

        for _, row in results_df.sort_values('total_pnl', ascending=False).head(20).iterrows():
            print(f"{row['model']:<30} {row['threshold']:>6.2f} {row['stop_loss']:>5.1%} {row['take_profit']:>5.1%} {row['vol_ratio']:>5.1f} {row['trades']:>6} {row['win_rate']:>6.1%} ${row['total_pnl']:>9.0f} ${row['avg_pnl']:>7.0f} ${row['avg_win']:>7.0f} ${row['avg_loss']:>7.0f}")

        print("\n" + "=" * 130)
        print("TOP 20 CONFIGURATIONS BY WIN RATE (min 10 trades)")
        print("=" * 130)

        wr_df = results_df[results_df['trades'] >= 10].sort_values('win_rate', ascending=False)
        print(f"{'Model':<30} {'Thresh':>6} {'SL':>5} {'TP':>5} {'VolR':>5} {'Trades':>6} {'WinR':>6} {'P&L':>10} {'AvgPnL':>8}")
        print("-" * 130)

        for _, row in wr_df.head(20).iterrows():
            print(f"{row['model']:<30} {row['threshold']:>6.2f} {row['stop_loss']:>5.1%} {row['take_profit']:>5.1%} {row['vol_ratio']:>5.1f} {row['trades']:>6} {row['win_rate']:>6.1%} ${row['total_pnl']:>9.0f} ${row['avg_pnl']:>7.0f}")

        # Save results
        results_df.to_csv('/Users/bz/Pythia2/data/backtest_results.csv', index=False)
        logger.info(f"\nResults saved to data/backtest_results.csv")

    else:
        # Single run
        config = BacktestConfig(model_path=models[0])
        result = run_backtest(config, features_df)

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Model: {Path(config.model_path).name}")
        print(f"Trades: {result['trades']}")
        print(f"Win Rate: {result['win_rate']:.1%}")
        print(f"Total P&L: ${result['total_pnl']:,.2f}")
        print(f"Avg P&L: ${result['avg_pnl']:,.2f}")


if __name__ == "__main__":
    main()
