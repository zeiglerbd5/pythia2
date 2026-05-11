#!/usr/bin/env python3
"""
Analyze Small Mover spikes to understand why they trigger detection
but don't deliver meaningful gains.
"""

import sqlite3
import pandas as pd
import numpy as np

def analyze_small_mover(db_path: str, symbol: str, spike_time: str):
    """Analyze a small mover spike in detail."""

    conn = sqlite3.connect(db_path)

    # Get the spike candle + next 10 candles
    query = """
    SELECT
        c.timestamp,
        c.open,
        c.close,
        c.high,
        c.low,
        c.volume,
        f.volume_zscore,
        f.RSI_14
    FROM candles c
    JOIN features f ON c.timestamp = f.timestamp AND c.symbol = f.symbol
    WHERE c.symbol = ?
        AND c.timestamp BETWEEN ? AND datetime(?, '+10 minutes')
        AND f.timeframe = '1m'
    ORDER BY c.timestamp
    """

    df = pd.read_sql_query(query, conn, params=[symbol, spike_time, spike_time])
    conn.close()

    if len(df) == 0:
        return None

    signal_price = df['close'].iloc[0]
    df['gain_pct'] = ((df['close'] / signal_price) - 1) * 100
    df['candle_gain_pct'] = ((df['close'] / df['open']) - 1) * 100

    # Check if it meets spike criteria on signal candle
    signal_candle = df.iloc[0]
    candle_move_pct = ((signal_candle['high'] / signal_candle['open']) - 1) * 100

    # Look at next 3 candles for spike confirmation
    if len(df) >= 4:
        next_3 = df.iloc[1:4]
        max_price_next_3 = next_3['high'].max()
        price_spike_pct = ((max_price_next_3 / signal_price) - 1) * 100
    else:
        price_spike_pct = 0

    peak_gain = df['gain_pct'].max()
    peak_idx = df['gain_pct'].idxmax()
    time_to_peak = peak_idx

    # Check for reversal
    if len(df) >= 2:
        end_gain = df['gain_pct'].iloc[-1]
        reversal_pct = ((end_gain - peak_gain) / peak_gain * 100) if peak_gain > 0 else 0
    else:
        end_gain = 0
        reversal_pct = 0

    return {
        'symbol': symbol,
        'timestamp': spike_time,
        'signal_candle_move': candle_move_pct,
        'price_spike_next_3': price_spike_pct,
        'volume_zscore': signal_candle['volume_zscore'],
        'peak_gain': peak_gain,
        'time_to_peak_candles': time_to_peak,
        'end_gain_10m': end_gain,
        'reversal_pct': reversal_pct,
        'dataframe': df
    }


def analyze_all_small_movers(db_path: str):
    """Compare Small Movers vs successful spikes."""

    conn = sqlite3.connect(db_path)

    # Get all spikes from both periods
    spikes_query = """
    SELECT symbol, timestamp
    FROM targets
    WHERE timeframe = '1m'
        AND target = 1
        AND timestamp >= '2025-10-15'
        AND timestamp < '2025-10-21'
    ORDER BY timestamp
    """

    spikes = pd.read_sql_query(spikes_query, conn)
    conn.close()

    print(f"Analyzing {len(spikes)} total spikes...")
    print()

    # Categorize each
    small_movers = []
    successful_spikes = []

    for idx, row in spikes.iterrows():
        analysis = analyze_small_mover(db_path, row['symbol'], row['timestamp'])
        if analysis:
            if analysis['peak_gain'] < 8:
                small_movers.append(analysis)
            else:
                successful_spikes.append(analysis)

    print(f"Small Movers: {len(small_movers)}")
    print(f"Successful Spikes: {len(successful_spikes)}")
    print()

    # Compare statistics
    sm_df = pd.DataFrame(small_movers)
    ss_df = pd.DataFrame(successful_spikes)

    print("=" * 80)
    print("COMPARISON: Small Movers vs Successful Spikes")
    print("=" * 80)
    print()

    metrics = [
        ('Peak Gain %', 'peak_gain'),
        ('Signal Candle Move %', 'signal_candle_move'),
        ('Price Spike Next 3 Min %', 'price_spike_next_3'),
        ('Volume Z-Score', 'volume_zscore'),
        ('End Gain 10m %', 'end_gain_10m'),
        ('Reversal %', 'reversal_pct')
    ]

    print(f"{'Metric':<30} {'Small Mover':>15} {'Successful':>15} {'Difference':>15}")
    print("-" * 80)

    for label, col in metrics:
        sm_mean = sm_df[col].mean()
        ss_mean = ss_df[col].mean()
        diff = sm_mean - ss_mean

        print(f"{label:<30} {sm_mean:>15.2f} {ss_mean:>15.2f} {diff:>15.2f}")

    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Check reversal rate
    sm_reversal_rate = (sm_df['reversal_pct'] < -50).sum() / len(sm_df) * 100
    ss_reversal_rate = (ss_df['reversal_pct'] < -50).sum() / len(ss_df) * 100

    print(f"Quick Reversal Rate (>50% loss from peak):")
    print(f"  Small Movers: {sm_reversal_rate:.1f}%")
    print(f"  Successful: {ss_reversal_rate:.1f}%")
    print()

    # Check peak timing
    sm_fast_peak = (sm_df['time_to_peak_candles'] <= 3).sum() / len(sm_df) * 100
    ss_fast_peak = (ss_df['time_to_peak_candles'] <= 3).sum() / len(ss_df) * 100

    print(f"Peaks in First 3 Candles:")
    print(f"  Small Movers: {sm_fast_peak:.1f}%")
    print(f"  Successful: {ss_fast_peak:.1f}%")
    print()

    # Volume comparison
    print(f"Volume Z-Score Distribution:")
    print(f"  Small Movers: {sm_df['volume_zscore'].median():.2f} (median)")
    print(f"  Successful: {ss_df['volume_zscore'].median():.2f} (median)")
    print()

    # Show examples
    print("=" * 80)
    print("EXAMPLE SMALL MOVERS")
    print("=" * 80)
    print()

    for i, sm in enumerate(small_movers[:5]):
        print(f"{i+1}. {sm['symbol']} @ {sm['timestamp']}")
        print(f"   Peak: {sm['peak_gain']:.1f}% at candle {sm['time_to_peak_candles']}")
        print(f"   Signal candle: {sm['signal_candle_move']:.1f}%")
        print(f"   Next 3 min: {sm['price_spike_next_3']:.1f}%")
        print(f"   Volume Z-score: {sm['volume_zscore']:.2f}")
        print(f"   Reversal: {sm['reversal_pct']:.1f}%")
        print()

    return sm_df, ss_df


if __name__ == "__main__":
    db_path = "market_data copy_86.db"

    sm_df, ss_df = analyze_all_small_movers(db_path)

    # Save to file
    sm_df.to_csv('small_movers_analysis.csv', index=False)
    print(f"Saved detailed analysis to small_movers_analysis.csv")
