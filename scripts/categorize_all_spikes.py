#!/usr/bin/env python3
"""
Categorize ALL spikes in the database to get comprehensive statistics.
"""

import sqlite3
import pandas as pd
import numpy as np

def analyze_spike_duration(db_path: str, symbol: str, spike_time: str):
    """Analyze a spike across multiple timeframes."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT c.timestamp, c.close, c.volume
    FROM candles c
    WHERE c.symbol = ?
        AND c.timestamp BETWEEN ? AND datetime(?, '+24 hours')
    ORDER BY c.timestamp
    """

    df = pd.read_sql_query(query, conn, params=[symbol, spike_time, spike_time])
    conn.close()

    if len(df) == 0:
        return None

    signal_price = df['close'].iloc[0]
    df['minutes_elapsed'] = (pd.to_datetime(df['timestamp']) - pd.to_datetime(df['timestamp'].iloc[0])).dt.total_seconds() / 60
    df['gain_pct'] = ((df['close'] / signal_price) - 1) * 100

    peak_idx = df['gain_pct'].idxmax()
    peak_gain = df.loc[peak_idx, 'gain_pct']
    time_to_peak_min = df.loc[peak_idx, 'minutes_elapsed']

    gains = {}
    for minutes, label in [(10, '10m'), (60, '1h'), (360, '6h'), (1440, '24h')]:
        window = df[df['minutes_elapsed'] <= minutes]
        if len(window) > 0:
            gains[label] = window['gain_pct'].max()
        else:
            gains[label] = 0

    return {
        'peak_gain': peak_gain,
        'time_to_peak_min': time_to_peak_min,
        'gain_10m': gains.get('10m', 0),
        'gain_1h': gains.get('1h', 0),
        'gain_6h': gains.get('6h', 0),
        'gain_24h': gains.get('24h', 0)
    }


def categorize_spike(analysis: dict) -> str:
    """Categorize spike based on behavior."""
    if analysis is None:
        return "Unknown"

    peak_gain = analysis['peak_gain']
    time_to_peak = analysis['time_to_peak_min']
    gain_1h = analysis['gain_1h']

    if time_to_peak <= 30 and peak_gain >= 8:
        if gain_1h < peak_gain * 0.5:
            return "Fast & Steep (Fade)"
        else:
            return "Fast & Steep"
    elif time_to_peak > 30 and peak_gain >= 15:
        return "Slow & Large"
    elif peak_gain < 8:
        return "Small Mover"
    else:
        return "Other"


def analyze_all_spikes(db_path: str):
    """Analyze all spikes in the database."""

    conn = sqlite3.connect(db_path)

    # Get ALL spikes
    spikes_query = """
    SELECT symbol, timestamp
    FROM targets
    WHERE timeframe = '1m' AND target = 1
    ORDER BY timestamp
    """

    spikes = pd.read_sql_query(spikes_query, conn)
    conn.close()

    print(f"Analyzing {len(spikes)} total spikes...")
    print(f"Date range: {spikes['timestamp'].min()} to {spikes['timestamp'].max()}")
    print()

    # Analyze each spike
    results = []
    for idx, row in spikes.iterrows():
        if idx % 50 == 0:
            print(f"  Progress: {idx+1}/{len(spikes)} ({(idx+1)/len(spikes)*100:.1f}%)")

        analysis = analyze_spike_duration(db_path, row['symbol'], row['timestamp'])
        if analysis:
            category = categorize_spike(analysis)
            results.append({
                'symbol': row['symbol'],
                'timestamp': row['timestamp'],
                'category': category,
                'peak_gain': analysis['peak_gain'],
                'time_to_peak_min': analysis['time_to_peak_min']
            })

    print(f"\nAnalyzed {len(results)}/{len(spikes)} spikes successfully")
    print()

    # Count categories
    categories = {}
    for r in results:
        cat = r['category']
        categories[cat] = categories.get(cat, 0) + 1

    # Print summary
    print("=" * 80)
    print("COMPREHENSIVE SPIKE CATEGORIZATION")
    print(f"Total Spikes: {len(results)}")
    print(f"Date Range: Sept 24 - Oct 20, 2025 (27 days)")
    print("=" * 80)
    print()

    print("Category Breakdown:")
    print("-" * 80)
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(results) * 100
        print(f"{cat:<20} {count:>6} ({pct:>5.1f}%)")

    print()
    fast_steep_pct = categories.get('Fast & Steep', 0) / len(results) * 100
    slow_large_pct = categories.get('Slow & Large', 0) / len(results) * 100
    combined_pct = fast_steep_pct + slow_large_pct

    print(f"{'Two-Type Combined':<20} {categories.get('Fast & Steep', 0) + categories.get('Slow & Large', 0):>6} ({combined_pct:>5.1f}%)")
    print()

    # Statistics by category
    df = pd.DataFrame(results)

    print("=" * 80)
    print("STATISTICS BY CATEGORY")
    print("=" * 80)
    print()

    for cat in sorted(categories.keys()):
        cat_df = df[df['category'] == cat]
        print(f"{cat}:")
        print(f"  Count: {len(cat_df)}")
        print(f"  Peak Gain - Mean: {cat_df['peak_gain'].mean():.1f}%, Median: {cat_df['peak_gain'].median():.1f}%")
        print(f"  Time to Peak - Mean: {cat_df['time_to_peak_min'].mean():.0f} min, Median: {cat_df['time_to_peak_min'].median():.0f} min")
        print()

    # Save to CSV
    df.to_csv('all_spikes_categorized.csv', index=False)
    print(f"Full results saved to: all_spikes_categorized.csv")


if __name__ == "__main__":
    db_path = "market_data copy_86.db"
    analyze_all_spikes(db_path)
