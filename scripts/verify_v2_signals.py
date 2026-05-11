#!/usr/bin/env python3
"""
Verify V2 Model Signals - Check What Actually Happened

For each high-confidence V2 prediction, look at what happened in the
2-10 minutes after the signal. Did price actually spike?

This tells us if V2's "early warnings" are real alpha or false positives.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from datetime import datetime, timedelta


def main():
    logger.info("=" * 80)
    logger.info("VERIFY V2 SIGNALS - WHAT ACTUALLY HAPPENED?")
    logger.info("=" * 80)
    logger.info("")

    # === CONFIGURATION ===
    DB_PATH = 'market_data.duckdb'
    BACKTEST_CSV = 'backtest_v1_vs_v2_results.csv'

    # Signal thresholds
    V2_THRESHOLD = 0.70  # High confidence V2 signals

    # Outcome windows (minutes after signal)
    WINDOWS = [2, 3, 5, 10]

    # === STEP 1: Load backtest results ===
    logger.info("Loading backtest results...")
    df = pd.read_csv(BACKTEST_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info("")

    # === STEP 2: Find high-confidence V2 signals ===
    logger.info(f"Finding V2 signals >= {V2_THRESHOLD*100:.0f}%...")

    v2_signals = df[df['v2_prob'] >= V2_THRESHOLD].copy()
    v2_signals = v2_signals.sort_values(['v2_prob'], ascending=False)

    logger.info(f"Found {len(v2_signals)} high-confidence V2 signals")
    logger.info("")

    if len(v2_signals) == 0:
        logger.warning("No high-confidence V2 signals found!")
        return

    # === STEP 3: Query candles for outcome analysis ===
    logger.info("Querying candle data for outcome analysis...")

    conn = duckdb.connect(DB_PATH, read_only=True)

    # Get unique symbols from signals
    symbols = v2_signals['symbol'].unique().tolist()
    symbols_clause = "', '".join(symbols)

    # Get time range (signal time to +15 minutes after last signal)
    min_time = v2_signals['timestamp'].min()
    max_time = v2_signals['timestamp'].max() + timedelta(minutes=15)

    # Use ohlcv table (has recent data) instead of candles
    candles_query = f"""
        SELECT
            symbol,
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM ohlcv
        WHERE symbol IN ('{symbols_clause}')
        AND timestamp >= '{min_time}'
        AND timestamp <= '{max_time}'
        AND timeframe = '1m'
        ORDER BY symbol, timestamp
    """

    candles_df = conn.execute(candles_query).fetchdf()
    conn.close()

    logger.info(f"Loaded {len(candles_df):,} candles for {len(symbols)} symbols")
    logger.info("")

    # === STEP 4: Calculate outcomes for each signal ===
    logger.info("=" * 80)
    logger.info("SIGNAL OUTCOMES")
    logger.info("=" * 80)
    logger.info("")

    results = []

    for _, signal in v2_signals.iterrows():
        symbol = signal['symbol']
        signal_time = signal['timestamp']
        v2_prob = signal['v2_prob']
        v1_prob = signal['v1_prob']

        # Get candles for this symbol starting from signal time
        symbol_candles = candles_df[
            (candles_df['symbol'] == symbol) &
            (candles_df['timestamp'] >= signal_time)
        ].sort_values('timestamp')

        if len(symbol_candles) < 2:
            continue

        # Get entry price (close of signal candle or open of next)
        entry_candle = symbol_candles.iloc[0]
        entry_price = entry_candle['close']

        # Calculate returns at each window
        outcome = {
            'symbol': symbol,
            'signal_time': signal_time,
            'v2_prob': v2_prob,
            'v1_prob': v1_prob,
            'entry_price': entry_price,
        }

        for window in WINDOWS:
            # Get candles within window
            window_end = signal_time + timedelta(minutes=window)
            window_candles = symbol_candles[
                (symbol_candles['timestamp'] > signal_time) &
                (symbol_candles['timestamp'] <= window_end)
            ]

            if len(window_candles) > 0:
                max_high = window_candles['high'].max()
                min_low = window_candles['low'].min()
                final_close = window_candles.iloc[-1]['close']
                max_volume = window_candles['volume'].max()

                # Calculate returns
                max_return = (max_high - entry_price) / entry_price * 100
                min_return = (min_low - entry_price) / entry_price * 100
                final_return = (final_close - entry_price) / entry_price * 100

                outcome[f'max_{window}m'] = max_return
                outcome[f'min_{window}m'] = min_return
                outcome[f'final_{window}m'] = final_return
            else:
                outcome[f'max_{window}m'] = None
                outcome[f'min_{window}m'] = None
                outcome[f'final_{window}m'] = None

        results.append(outcome)

    results_df = pd.DataFrame(results)

    # === STEP 5: Display results ===
    logger.info(f"Analyzed {len(results_df)} signals")
    logger.info("")

    # Sort by V2 probability (highest first)
    results_df = results_df.sort_values('v2_prob', ascending=False)

    # Display top signals
    logger.info("TOP 30 V2 SIGNALS AND THEIR OUTCOMES:")
    logger.info("-" * 120)
    logger.info(f"{'Symbol':<12} {'Time':<20} {'V2%':>6} {'V1%':>6} │ {'2m':>7} {'3m':>7} {'5m':>7} {'10m':>7} │ {'Result':<10}")
    logger.info("-" * 120)

    for _, row in results_df.head(30).iterrows():
        # Determine outcome
        max_5m = row.get('max_5m', 0) or 0
        max_10m = row.get('max_10m', 0) or 0

        if max_5m >= 3.0:
            result = "✅ SPIKE!"
        elif max_5m >= 1.5:
            result = "📈 Good"
        elif max_5m >= 0.5:
            result = "➡️ Small"
        elif max_5m <= -1.0:
            result = "❌ Drop"
        else:
            result = "➖ Flat"

        # Format returns
        def fmt_ret(val):
            if val is None:
                return "   N/A"
            return f"{val:+6.2f}%"

        logger.info(
            f"{row['symbol']:<12} {str(row['signal_time']):<20} "
            f"{row['v2_prob']*100:5.1f}% {row['v1_prob']*100:5.1f}% │ "
            f"{fmt_ret(row.get('max_2m'))} {fmt_ret(row.get('max_3m'))} "
            f"{fmt_ret(row.get('max_5m'))} {fmt_ret(row.get('max_10m'))} │ "
            f"{result}"
        )

    logger.info("-" * 120)
    logger.info("")

    # === STEP 6: Summary statistics ===
    logger.info("=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    logger.info("")

    total_signals = len(results_df)

    # Calculate hit rates at different thresholds
    for target in [1.0, 2.0, 3.0, 5.0]:
        hits_5m = (results_df['max_5m'] >= target).sum()
        hits_10m = (results_df['max_10m'] >= target).sum()

        logger.info(f"Signals reaching +{target:.0f}% gain:")
        logger.info(f"  Within 5 min:  {hits_5m:3d}/{total_signals} ({hits_5m/total_signals*100:5.1f}%)")
        logger.info(f"  Within 10 min: {hits_10m:3d}/{total_signals} ({hits_10m/total_signals*100:5.1f}%)")
        logger.info("")

    # Average returns
    logger.info("Average MAX returns after V2 signal:")
    for window in WINDOWS:
        col = f'max_{window}m'
        if col in results_df.columns:
            avg = results_df[col].mean()
            median = results_df[col].median()
            logger.info(f"  {window:2d} min: avg={avg:+.2f}%, median={median:+.2f}%")
    logger.info("")

    # Worst drawdowns
    logger.info("Average MIN returns (drawdown) after V2 signal:")
    for window in WINDOWS:
        col = f'min_{window}m'
        if col in results_df.columns:
            avg = results_df[col].mean()
            worst = results_df[col].min()
            logger.info(f"  {window:2d} min: avg={avg:+.2f}%, worst={worst:+.2f}%")
    logger.info("")

    # === STEP 7: V2-only signals (where V1 was low) ===
    logger.info("=" * 80)
    logger.info("V2 EARLY WARNINGS (V2 high, V1 low)")
    logger.info("=" * 80)
    logger.info("")

    # Signals where V2 >> V1 (early detection)
    early_warnings = results_df[
        (results_df['v2_prob'] >= 0.70) &
        (results_df['v1_prob'] < 0.30)
    ].copy()

    logger.info(f"Found {len(early_warnings)} signals where V2 >= 70% but V1 < 30%")
    logger.info("These are cases where V2 detected something V1 missed:")
    logger.info("")

    if len(early_warnings) > 0:
        logger.info(f"{'Symbol':<12} {'Time':<20} {'V2%':>6} {'V1%':>6} │ {'5m Max':>8} {'10m Max':>8}")
        logger.info("-" * 80)

        for _, row in early_warnings.head(20).iterrows():
            def fmt_ret(val):
                if val is None:
                    return "   N/A"
                return f"{val:+6.2f}%"

            logger.info(
                f"{row['symbol']:<12} {str(row['signal_time']):<20} "
                f"{row['v2_prob']*100:5.1f}% {row['v1_prob']*100:5.1f}% │ "
                f"{fmt_ret(row.get('max_5m')):>8} {fmt_ret(row.get('max_10m')):>8}"
            )

        logger.info("")

        # Stats for early warnings specifically
        if len(early_warnings) > 0:
            avg_5m = early_warnings['max_5m'].mean()
            avg_10m = early_warnings['max_10m'].mean()
            hits_3pct = (early_warnings['max_5m'] >= 3.0).sum()

            logger.info(f"Early warning stats:")
            logger.info(f"  Avg max return (5m):  {avg_5m:+.2f}%")
            logger.info(f"  Avg max return (10m): {avg_10m:+.2f}%")
            logger.info(f"  Hit +3% within 5m:    {hits_3pct}/{len(early_warnings)} ({hits_3pct/len(early_warnings)*100:.1f}%)")

    logger.info("")

    # === STEP 8: Save detailed results ===
    output_file = 'v2_signal_verification.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"✓ Detailed results saved to {output_file}")
    logger.info("")

    # === STEP 9: Final verdict ===
    logger.info("=" * 80)
    logger.info("VERDICT")
    logger.info("=" * 80)
    logger.info("")

    avg_max_5m = results_df['max_5m'].mean()
    hit_rate_3pct = (results_df['max_5m'] >= 3.0).sum() / len(results_df) * 100

    if hit_rate_3pct >= 30:
        logger.info("✅ STRONG: V2 signals show strong predictive power")
        logger.info(f"   {hit_rate_3pct:.1f}% of signals hit +3% within 5 minutes")
        logger.info("   RECOMMENDATION: Deploy V2 for live trading")
    elif hit_rate_3pct >= 15:
        logger.info("📈 MODERATE: V2 signals show moderate predictive power")
        logger.info(f"   {hit_rate_3pct:.1f}% of signals hit +3% within 5 minutes")
        logger.info("   RECOMMENDATION: Deploy with tight risk management")
    elif avg_max_5m >= 1.0:
        logger.info("➡️ WEAK: V2 signals show some edge but not strong")
        logger.info(f"   Average max return: {avg_max_5m:+.2f}%")
        logger.info("   RECOMMENDATION: More testing needed")
    else:
        logger.info("❌ POOR: V2 signals don't show reliable predictive power")
        logger.info(f"   Average max return: {avg_max_5m:+.2f}%")
        logger.info("   RECOMMENDATION: Don't deploy, needs retraining")

    logger.info("")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
