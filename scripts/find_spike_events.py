#!/usr/bin/env python3
"""
Find Spike Events

Scans trade data to find rapid price spikes (e.g., 20%+ in < 5 minutes).
Outputs timestamps of spike onsets for use in labeling.

Usage:
    python scripts/find_spike_events.py \
        --db "market_data copy_86.db" \
        --symbols ARPA-USD,LOKA-USD \
        --dates 2025-10-13,2025-10-14,2025-10-15 \
        --min-gain 0.20 \
        --max-duration 300 \
        --output data/spike_events.csv
"""

import argparse
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def find_spikes_for_symbol_date(conn, symbol, date, min_gain=0.20, max_duration_sec=300):
    """
    Find all spike events for a symbol on a given date.

    Args:
        conn: Database connection
        symbol: Trading pair
        date: Date string (YYYY-MM-DD)
        min_gain: Minimum price gain to qualify as spike (0.20 = 20%)
        max_duration_sec: Maximum duration for spike (300 = 5 minutes)

    Returns:
        List of dicts with spike info
    """
    # Query all trades for the day
    query = """
        SELECT timestamp, price
        FROM trades
        WHERE symbol = ?
          AND timestamp >= ?
          AND timestamp < ?
        ORDER BY timestamp
    """

    start_date = f"{date}T00:00:00Z"
    end_date = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')

    df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])

    if len(df) == 0:
        return []

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

    spikes = []

    # Sliding window to find rapid price moves
    for i in range(len(df)):
        price_start = df.iloc[i]['price']
        time_start = df.iloc[i]['timestamp']

        # Look ahead for rapid gains
        for j in range(i+1, len(df)):
            price_end = df.iloc[j]['price']
            time_end = df.iloc[j]['timestamp']

            duration_sec = (time_end - time_start).total_seconds()

            # Stop if we exceed max duration
            if duration_sec > max_duration_sec:
                break

            # Check if this is a spike
            gain = (price_end - price_start) / price_start

            if gain >= min_gain:
                # Found a spike! Record it
                spikes.append({
                    'symbol': symbol,
                    'date': date,
                    'spike_start_time': time_start,
                    'spike_peak_time': time_end,
                    'start_price': price_start,
                    'peak_price': price_end,
                    'gain_pct': gain * 100,
                    'duration_sec': duration_sec
                })

                # Skip ahead to avoid finding overlapping spikes
                # (i.e., don't count 20% -> 25% as a separate spike)
                break

    return spikes


def main():
    parser = argparse.ArgumentParser(description='Find rapid price spike events')
    parser.add_argument('--db', required=True, help='Path to database file')
    parser.add_argument('--symbols', required=True, help='Comma-separated list of symbols')
    parser.add_argument('--dates', required=True, help='Comma-separated list of dates (YYYY-MM-DD)')
    parser.add_argument('--min-gain', type=float, default=0.20, help='Minimum gain percentage (default: 0.20 = 20%%)')
    parser.add_argument('--max-duration', type=int, default=300, help='Max spike duration in seconds (default: 300 = 5 min)')
    parser.add_argument('--output', required=True, help='Output CSV file path')

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]
    dates = [d.strip() for d in args.dates.split(',')]

    print("=" * 80)
    print("SPIKE EVENT DETECTION")
    print("=" * 80)
    print(f"Database: {args.db}")
    print(f"Symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    print(f"Dates: {len(dates)} ({', '.join(dates)})")
    print(f"Min gain: {args.min_gain*100}%")
    print(f"Max duration: {args.max_duration} seconds")
    print("=" * 80)
    print()

    conn = sqlite3.connect(args.db)

    all_spikes = []
    total_combinations = len(symbols) * len(dates)
    processed = 0

    for symbol in symbols:
        for date in dates:
            processed += 1
            print(f"[{processed}/{total_combinations}] Scanning {symbol} on {date}...")

            spikes = find_spikes_for_symbol_date(
                conn, symbol, date,
                min_gain=args.min_gain,
                max_duration_sec=args.max_duration
            )

            if spikes:
                print(f"  Found {len(spikes)} spike(s)")
                for spike in spikes:
                    print(f"    {spike['spike_start_time'].strftime('%H:%M:%S')} → "
                          f"{spike['spike_peak_time'].strftime('%H:%M:%S')}: "
                          f"+{spike['gain_pct']:.1f}% in {spike['duration_sec']:.0f}s")
                all_spikes.extend(spikes)
            else:
                print(f"  No spikes found")

    conn.close()

    print()
    print("=" * 80)
    print(f"TOTAL SPIKES FOUND: {len(all_spikes)}")
    print("=" * 80)

    if len(all_spikes) > 0:
        # Convert to DataFrame and save
        df = pd.DataFrame(all_spikes)
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")

        # Summary stats
        print()
        print("Summary:")
        print(f"  Total spikes: {len(df)}")
        print(f"  Avg gain: {df['gain_pct'].mean():.1f}%")
        print(f"  Max gain: {df['gain_pct'].max():.1f}%")
        print(f"  Avg duration: {df['duration_sec'].mean():.0f}s")
        print(f"  Symbols with spikes: {df['symbol'].nunique()}")
    else:
        print("No spikes found matching criteria.")

    print("\nDONE!")


if __name__ == '__main__':
    main()
