#!/usr/bin/env python3
"""
Categorize spikes into two types:
1. Fast & Steep: Quick gains (few minutes), typically 10-25%
2. Slow & Large: Sustained gains over hours, 15-70%+
"""

import sqlite3
import pandas as pd
from datetime import datetime

def analyze_spike_duration(db_path: str, symbol: str, spike_time: str):
    """
    Analyze a spike across multiple timeframes to determine its type.

    Returns dict with gains at 10min, 1hr, 6hr, 24hr
    """
    conn = sqlite3.connect(db_path)

    # Get candles from signal time to +24 hours
    query = """
    SELECT
        c.timestamp,
        c.close,
        c.volume
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

    # Calculate time to peak and peak gain
    df['minutes_elapsed'] = (pd.to_datetime(df['timestamp']) - pd.to_datetime(df['timestamp'].iloc[0])).dt.total_seconds() / 60
    df['gain_pct'] = ((df['close'] / signal_price) - 1) * 100

    peak_idx = df['gain_pct'].idxmax()
    peak_gain = df.loc[peak_idx, 'gain_pct']
    time_to_peak_min = df.loc[peak_idx, 'minutes_elapsed']

    # Get gains at specific intervals
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
    """
    Categorize spike based on behavior:
    - Fast & Steep: Peak within 30 min, gains 8-25%
    - Slow & Large: Peak after 30 min, sustained gains >15%
    - Quick Fade: Gains reverse quickly
    - Other: Doesn't fit patterns
    """
    if analysis is None:
        return "Unknown"

    peak_gain = analysis['peak_gain']
    time_to_peak = analysis['time_to_peak_min']
    gain_10m = analysis['gain_10m']
    gain_1h = analysis['gain_1h']

    # Fast & Steep: Quick peak, moderate-high gains
    if time_to_peak <= 30 and peak_gain >= 8:
        # Check if it fades quickly
        if gain_1h < peak_gain * 0.5:
            return "Fast & Steep (Fade)"
        else:
            return "Fast & Steep"

    # Slow & Large: Takes time to build, sustains
    elif time_to_peak > 30 and peak_gain >= 15:
        return "Slow & Large"

    # Small movers
    elif peak_gain < 8:
        return "Small Mover"

    else:
        return "Other"


def generate_categorization_report(db_path: str, output_file: str):
    """Generate spike categorization report."""

    conn = sqlite3.connect(db_path)

    # Get all spikes
    spikes_query = """
    SELECT symbol, timestamp
    FROM targets
    WHERE timeframe = '1m'
        AND target = 1
        AND timestamp >= '2025-10-18'
    ORDER BY timestamp
    """

    spikes = pd.read_sql_query(spikes_query, conn)
    conn.close()

    print(f"Analyzing {len(spikes)} spikes for categorization...")
    print()

    # Analyze each spike
    results = []
    for idx, row in spikes.iterrows():
        print(f"  Analyzing spike {idx+1}/{len(spikes)}: {row['symbol']} @ {row['timestamp']}")
        analysis = analyze_spike_duration(db_path, row['symbol'], row['timestamp'])
        if analysis:
            category = categorize_spike(analysis)
            results.append({
                'spike_num': idx + 1,
                'symbol': row['symbol'],
                'timestamp': row['timestamp'],
                'category': category,
                'analysis': analysis
            })

    # Count categories
    categories = {}
    for r in results:
        cat = r['category']
        categories[cat] = categories.get(cat, 0) + 1

    # Generate markdown
    md = []
    md.append("# Spike Type Categorization")
    md.append("")
    md.append(f"Analysis of {len(results)} detected spikes from October 18-20, 2025")
    md.append("")
    md.append("## Categories")
    md.append("")
    md.append("**Fast & Steep**: Quick gains over a few minutes (8-25%), peaks within 30 min")
    md.append("")
    md.append("**Slow & Large**: Sustained gains over hours (15-70%+), peaks after 30 min")
    md.append("")
    md.append("**Small Mover**: Peak gains <8%")
    md.append("")
    md.append("## Summary")
    md.append("")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(results) * 100
        md.append(f"- **{cat}**: {count}/{len(results)} ({pct:.1f}%)")
    md.append("")

    # Group by category
    md.append("## Spikes by Category")
    md.append("")

    for cat in sorted(categories.keys()):
        md.append(f"### {cat}")
        md.append("")

        cat_spikes = [r for r in results if r['category'] == cat]

        for result in cat_spikes:
            a = result['analysis']
            md.append(f"**#{result['spike_num']}: {result['symbol']} @ {result['timestamp']}**")
            md.append(f"- Peak: {a['peak_gain']:.1f}% at {a['time_to_peak_min']:.0f} min")
            md.append(f"- 10m: {a['gain_10m']:.1f}% | 1h: {a['gain_1h']:.1f}% | 6h: {a['gain_6h']:.1f}% | 24h: {a['gain_24h']:.1f}%")
            md.append("")

        md.append("---")
        md.append("")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(md))

    print(f"\nReport written to: {output_file}")
    print(f"\nCategory breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} ({count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    db_path = "market_data copy_86.db"
    output_file = "SPIKE_CATEGORIZATION.md"

    generate_categorization_report(db_path, output_file)
