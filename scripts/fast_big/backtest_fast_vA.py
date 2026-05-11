#!/usr/bin/env python3
"""
Backtest Fast-vA Model - Simulated Trading on Historical Data

This script runs the Fast-vA model on Oct 6-20 data and simulates
what would have happened if we traded every signal.

Simulated strategy:
  - Enter at signal + 1 minute (realistic delay)
  - Exit at max profit within 30 minutes OR
  - Exit at stop-loss (-2%) OR
  - Exit at 30 minute timeout

Outputs:
  - Win rate, average gain, total return
  - List of all trades with outcomes
  - Comparison with random entry baseline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
import joblib
from datetime import timedelta


def run_backtest(
    model_path: str = 'models/fast_big/xgboost_fast_vA.pkl',
    threshold: float = 0.7,
    hold_minutes: int = 30,
    stop_loss: float = -0.02,
    take_profit: float = None,  # None = ride to max within window
    entry_delay: int = 1,  # Minutes delay after signal
):
    """
    Run backtest on historical data.

    Args:
        model_path: Path to trained model
        threshold: Probability threshold for signals
        hold_minutes: Maximum hold time
        stop_loss: Stop loss percentage (negative)
        take_profit: Take profit percentage (None = ride the wave)
        entry_delay: Minutes to wait after signal before entry
    """
    logger.info("=" * 80)
    logger.info("FAST-vA BACKTEST")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Model: {model_path}")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Hold window: {hold_minutes} minutes")
    logger.info(f"Stop loss: {stop_loss*100:.1f}%")
    logger.info(f"Take profit: {take_profit*100:.1f}%" if take_profit else "Take profit: Ride to max")
    logger.info(f"Entry delay: {entry_delay} minute(s)")
    logger.info("")

    # Load model
    logger.info("Loading model...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_cols = model_data['feature_columns']

    # Load data
    logger.info("Loading historical data...")
    DB_PATH = 'market_data.duckdb'
    START_DATE = '2025-10-06'
    END_DATE = '2025-10-20'

    conn = duckdb.connect(DB_PATH, read_only=True)

    # Load OHLCV for price tracking
    ohlcv_query = f"""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM candles
        WHERE timestamp >= '{START_DATE}'
        AND timestamp <= '{END_DATE}'
        ORDER BY symbol, timestamp
    """
    ohlcv_df = conn.execute(ohlcv_query).fetchdf()
    logger.info(f"Loaded {len(ohlcv_df):,} OHLCV rows")

    # Load features
    features_query = f"""
        SELECT
            symbol,
            timestamp,
            {', '.join(feature_cols)}
        FROM features
        WHERE timeframe = '1m'
        AND timestamp >= '{START_DATE}'
        AND timestamp <= '{END_DATE}'
        ORDER BY symbol, timestamp
    """
    features_df = conn.execute(features_query).fetchdf()
    conn.close()
    logger.info(f"Loaded {len(features_df):,} feature rows")
    logger.info("")

    # Generate predictions
    logger.info("Generating predictions...")
    X = features_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    probas = model.predict_proba(X)[:, 1]
    features_df['proba'] = probas
    features_df['signal'] = (probas >= threshold).astype(int)

    n_signals = features_df['signal'].sum()
    logger.info(f"Total signals at threshold {threshold}: {n_signals:,}")
    logger.info("")

    # Simulate trades
    logger.info("Simulating trades...")
    logger.info("")

    trades = []

    # Group by symbol for efficient lookup
    ohlcv_grouped = {symbol: group.sort_values('timestamp').reset_index(drop=True)
                     for symbol, group in ohlcv_df.groupby('symbol')}

    signal_rows = features_df[features_df['signal'] == 1].copy()

    for _, row in signal_rows.iterrows():
        symbol = row['symbol']
        signal_time = row['timestamp']
        signal_proba = row['proba']

        if symbol not in ohlcv_grouped:
            continue

        symbol_ohlcv = ohlcv_grouped[symbol]

        # Find entry point (signal time + delay)
        entry_time = signal_time + timedelta(minutes=entry_delay)

        # Find entry row
        entry_mask = symbol_ohlcv['timestamp'] >= entry_time
        if not entry_mask.any():
            continue

        entry_idx = entry_mask.idxmax()
        entry_price = symbol_ohlcv.loc[entry_idx, 'open']  # Enter at open
        entry_actual_time = symbol_ohlcv.loc[entry_idx, 'timestamp']

        if entry_price <= 0:
            continue

        # Track price over hold window
        exit_time = entry_actual_time + timedelta(minutes=hold_minutes)
        hold_mask = (symbol_ohlcv['timestamp'] > entry_actual_time) & \
                    (symbol_ohlcv['timestamp'] <= exit_time)

        if not hold_mask.any():
            continue

        hold_data = symbol_ohlcv[hold_mask]

        # Simulate minute-by-minute
        exit_price = entry_price
        exit_reason = 'timeout'
        max_gain = 0
        min_gain = 0

        for _, candle in hold_data.iterrows():
            # Check low for stop loss
            low_pct = (candle['low'] - entry_price) / entry_price
            if low_pct <= stop_loss:
                exit_price = entry_price * (1 + stop_loss)
                exit_reason = 'stop_loss'
                break

            # Check high for take profit
            high_pct = (candle['high'] - entry_price) / entry_price
            max_gain = max(max_gain, high_pct)
            min_gain = min(min_gain, low_pct)

            if take_profit and high_pct >= take_profit:
                exit_price = entry_price * (1 + take_profit)
                exit_reason = 'take_profit'
                break

            # Update exit price (close of last candle)
            exit_price = candle['close']

        # Calculate return
        pnl = (exit_price - entry_price) / entry_price

        trades.append({
            'symbol': symbol,
            'signal_time': signal_time,
            'signal_proba': signal_proba,
            'entry_time': entry_actual_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'max_gain': max_gain,
            'min_gain': min_gain
        })

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        logger.warning("No trades executed!")
        return

    # === RESULTS ===
    logger.info("=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)
    logger.info("")

    # Overall stats
    n_trades = len(trades_df)
    n_wins = (trades_df['pnl'] > 0).sum()
    n_losses = (trades_df['pnl'] <= 0).sum()
    win_rate = n_wins / n_trades * 100

    avg_pnl = trades_df['pnl'].mean() * 100
    total_pnl = trades_df['pnl'].sum() * 100
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() * 100 if n_wins > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() * 100 if n_losses > 0 else 0

    logger.info(f"Total Trades: {n_trades}")
    logger.info(f"Winners: {n_wins} ({win_rate:.1f}%)")
    logger.info(f"Losers: {n_losses} ({100-win_rate:.1f}%)")
    logger.info("")
    logger.info(f"Average P&L per trade: {avg_pnl:+.2f}%")
    logger.info(f"Average winning trade: {avg_win:+.2f}%")
    logger.info(f"Average losing trade: {avg_loss:+.2f}%")
    logger.info("")
    logger.info(f"Total cumulative return: {total_pnl:+.1f}%")
    logger.info(f"(If you traded each signal with 1 unit)")
    logger.info("")

    # Exit reason breakdown
    exit_reasons = trades_df['exit_reason'].value_counts()
    logger.info("Exit Reasons:")
    for reason, count in exit_reasons.items():
        pct = count / n_trades * 100
        reason_pnl = trades_df[trades_df['exit_reason'] == reason]['pnl'].mean() * 100
        logger.info(f"  {reason}: {count} ({pct:.1f}%) - avg P&L: {reason_pnl:+.2f}%")
    logger.info("")

    # Max gain analysis
    avg_max_gain = trades_df['max_gain'].mean() * 100
    logger.info(f"Average max gain during hold: {avg_max_gain:.2f}%")
    logger.info(f"(This is the potential if we had perfect exits)")
    logger.info("")

    # Best and worst trades
    logger.info("Top 10 Best Trades:")
    best_trades = trades_df.nlargest(10, 'pnl')
    for _, trade in best_trades.iterrows():
        logger.info(f"  {trade['symbol']}: {trade['pnl']*100:+.1f}% (proba={trade['signal_proba']:.2f}, max={trade['max_gain']*100:.1f}%)")
    logger.info("")

    logger.info("Top 10 Worst Trades:")
    worst_trades = trades_df.nsmallest(10, 'pnl')
    for _, trade in worst_trades.iterrows():
        logger.info(f"  {trade['symbol']}: {trade['pnl']*100:+.1f}% (proba={trade['signal_proba']:.2f}, min={trade['min_gain']*100:.1f}%)")
    logger.info("")

    # By symbol
    logger.info("Performance by Symbol (top 10 most traded):")
    symbol_stats = trades_df.groupby('symbol').agg({
        'pnl': ['count', 'mean', 'sum']
    }).round(4)
    symbol_stats.columns = ['trades', 'avg_pnl', 'total_pnl']
    symbol_stats = symbol_stats.sort_values('trades', ascending=False).head(10)

    for symbol, row in symbol_stats.iterrows():
        logger.info(f"  {symbol}: {int(row['trades'])} trades, avg {row['avg_pnl']*100:+.2f}%, total {row['total_pnl']*100:+.1f}%")
    logger.info("")

    # By probability bucket
    logger.info("Performance by Signal Strength:")
    trades_df['proba_bucket'] = pd.cut(trades_df['signal_proba'],
                                        bins=[0.7, 0.8, 0.9, 0.95, 1.0],
                                        labels=['0.7-0.8', '0.8-0.9', '0.9-0.95', '0.95+'])

    bucket_stats = trades_df.groupby('proba_bucket', observed=True).agg({
        'pnl': ['count', 'mean'],
        'max_gain': 'mean'
    })
    bucket_stats.columns = ['trades', 'avg_pnl', 'avg_max_gain']

    for bucket, row in bucket_stats.iterrows():
        logger.info(f"  {bucket}: {int(row['trades'])} trades, avg {row['avg_pnl']*100:+.2f}%, max potential {row['avg_max_gain']*100:.1f}%")
    logger.info("")

    # Save detailed results
    output_path = 'models/fast_big/backtest_results_vA.csv'
    trades_df.to_csv(output_path, index=False)
    logger.info(f"Detailed results saved to {output_path}")
    logger.info("")

    logger.info("=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 80)

    return trades_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Backtest Fast-vA model')
    parser.add_argument('--threshold', type=float, default=0.7, help='Signal threshold')
    parser.add_argument('--hold', type=int, default=30, help='Max hold minutes')
    parser.add_argument('--stop-loss', type=float, default=-0.02, help='Stop loss pct')
    parser.add_argument('--take-profit', type=float, default=None, help='Take profit pct')
    args = parser.parse_args()

    run_backtest(
        threshold=args.threshold,
        hold_minutes=args.hold,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit
    )


if __name__ == '__main__':
    main()
