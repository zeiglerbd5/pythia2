#!/usr/bin/env python3
"""
Document all spike conditions detected in the order book data.

This script analyzes the database and creates a comprehensive report of:
- Timestamp when spike conditions were detected
- Symbol
- Start price (when conditions detected)
- Peak price reached
- Percentage gain
- Time to peak
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import argparse


def find_spikes(db_path, symbols, dates, spike_threshold=0.05, lookahead_minutes=10):
    """
    Find all spike conditions in the data.

    Args:
        db_path: Path to SQLite database
        symbols: List of symbols to analyze
        dates: List of dates to analyze
        spike_threshold: Minimum percentage gain to be considered a spike (default 5%)
        lookahead_minutes: How far ahead to look for peak (default 10 minutes)

    Returns:
        List of spike events with details
    """
    conn = sqlite3.connect(db_path)

    spike_events = []
    total_combinations = len(symbols) * len(dates)
    processed = 0

    print(f"Analyzing {total_combinations} symbol-date combinations...")
    print(f"Spike threshold: {spike_threshold*100}%")
    print(f"Lookahead window: {lookahead_minutes} minutes")
    print()

    for symbol in symbols:
        for date in dates:
            processed += 1
            if processed % 100 == 0:
                print(f"Progress: {processed}/{total_combinations} ({100*processed/total_combinations:.1f}%)")

            # Query order book data for this symbol-date
            query = """
            SELECT timestamp, bid_price, ask_price
            FROM order_book_features
            WHERE symbol = ? AND DATE(timestamp) = ?
            ORDER BY timestamp
            """

            df = pd.read_sql_query(query, conn, params=(symbol, date))

            if len(df) == 0:
                continue

            # Convert timestamp to datetime and calculate mid_price
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2

            # For each point, look ahead and find peak
            lookahead_seconds = lookahead_minutes * 60

            for i in range(len(df) - 1):
                current_time = df.iloc[i]['timestamp']
                current_price = df.iloc[i]['mid_price']

                if pd.isna(current_price) or current_price == 0:
                    continue

                # Find all prices in lookahead window
                lookahead_end = current_time + timedelta(seconds=lookahead_seconds)
                future_prices = df[
                    (df['timestamp'] > current_time) &
                    (df['timestamp'] <= lookahead_end)
                ]['mid_price']

                if len(future_prices) == 0:
                    continue

                # Find peak
                peak_price = future_prices.max()
                peak_idx = future_prices.idxmax()
                peak_time = df.loc[peak_idx, 'timestamp']

                # Calculate gain
                gain = (peak_price - current_price) / current_price

                # If this is a spike, record it
                if gain >= spike_threshold:
                    time_to_peak = (peak_time - current_time).total_seconds()

                    spike_events.append({
                        'symbol': symbol,
                        'detection_time': current_time.isoformat(),
                        'start_price': float(current_price),
                        'peak_price': float(peak_price),
                        'peak_time': peak_time.isoformat(),
                        'gain_percent': float(gain * 100),
                        'time_to_peak_seconds': float(time_to_peak),
                        'time_to_peak_minutes': float(time_to_peak / 60)
                    })

    conn.close()

    print(f"\nFound {len(spike_events)} spike events")
    return spike_events


def create_spike_report(spike_events, output_path):
    """Create a detailed markdown report of all spikes."""

    if len(spike_events) == 0:
        print("No spike events found!")
        return

    # Convert to DataFrame for analysis
    df = pd.DataFrame(spike_events)

    # Sort by detection time
    df = df.sort_values('detection_time')

    # Create markdown report
    report = []
    report.append("# Spike Events Report")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Total Spike Events**: {len(df)}")
    report.append(f"**Unique Symbols**: {df['symbol'].nunique()}")
    report.append("")

    # Summary statistics
    report.append("## Summary Statistics")
    report.append("")
    report.append(f"- **Average Gain**: {df['gain_percent'].mean():.2f}%")
    report.append(f"- **Median Gain**: {df['gain_percent'].median():.2f}%")
    report.append(f"- **Max Gain**: {df['gain_percent'].max():.2f}%")
    report.append(f"- **Min Gain**: {df['gain_percent'].min():.2f}%")
    report.append("")
    report.append(f"- **Average Time to Peak**: {df['time_to_peak_minutes'].mean():.2f} minutes")
    report.append(f"- **Median Time to Peak**: {df['time_to_peak_minutes'].median():.2f} minutes")
    report.append("")

    # Top symbols by spike count
    report.append("## Top 20 Symbols by Spike Count")
    report.append("")
    symbol_counts = df['symbol'].value_counts().head(20)
    for symbol, count in symbol_counts.items():
        avg_gain = df[df['symbol'] == symbol]['gain_percent'].mean()
        report.append(f"- **{symbol}**: {count} spikes (avg gain: {avg_gain:.2f}%)")
    report.append("")

    # Top 50 largest spikes
    report.append("## Top 50 Largest Spikes")
    report.append("")
    report.append("| Rank | Symbol | Detection Time | Start Price | Peak Price | Gain % | Time to Peak |")
    report.append("|------|--------|----------------|-------------|------------|--------|--------------|")

    top_spikes = df.nlargest(50, 'gain_percent')
    for idx, (_, row) in enumerate(top_spikes.iterrows(), 1):
        detection_time = pd.to_datetime(row['detection_time']).strftime('%Y-%m-%d %H:%M:%S')
        report.append(
            f"| {idx} | {row['symbol']} | {detection_time} | "
            f"${row['start_price']:.6f} | ${row['peak_price']:.6f} | "
            f"{row['gain_percent']:.2f}% | {row['time_to_peak_minutes']:.2f} min |"
        )
    report.append("")

    # All spike events (detailed table)
    report.append("## All Spike Events (Chronological)")
    report.append("")
    report.append("| # | Symbol | Detection Time | Start Price | Peak Price | Peak Time | Gain % | Time to Peak |")
    report.append("|---|--------|----------------|-------------|------------|-----------|--------|--------------|")

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        detection_time = pd.to_datetime(row['detection_time']).strftime('%Y-%m-%d %H:%M:%S')
        peak_time = pd.to_datetime(row['peak_time']).strftime('%Y-%m-%d %H:%M:%S')
        report.append(
            f"| {idx} | {row['symbol']} | {detection_time} | "
            f"${row['start_price']:.6f} | ${row['peak_price']:.6f} | {peak_time} | "
            f"{row['gain_percent']:.2f}% | {row['time_to_peak_minutes']:.2f} min |"
        )

    report.append("")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Report written to: {output_path}")


def save_spike_json(spike_events, output_path):
    """Save spike events as JSON for programmatic access."""
    with open(output_path, 'w') as f:
        json.dump(spike_events, f, indent=2)
    print(f"JSON data written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Document all spike events in order book data')
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--symbols', required=True, help='Comma-separated list of symbols')
    parser.add_argument('--dates', required=True, help='Comma-separated list of dates (YYYY-MM-DD)')
    parser.add_argument('--spike-threshold', type=float, default=0.05,
                        help='Minimum gain threshold (default: 0.05 = 5%%)')
    parser.add_argument('--lookahead', type=int, default=10,
                        help='Lookahead window in minutes (default: 10)')
    parser.add_argument('--output-md', default='SPIKE_EVENTS.md',
                        help='Output markdown report path')
    parser.add_argument('--output-json', default='data/spike_events.json',
                        help='Output JSON data path')

    args = parser.parse_args()

    # Parse symbols and dates
    symbols = [s.strip() for s in args.symbols.split(',')]
    dates = [d.strip() for d in args.dates.split(',')]

    print("="*80)
    print("SPIKE EVENT DOCUMENTATION")
    print("="*80)
    print(f"Database: {args.db}")
    print(f"Symbols: {len(symbols)}")
    print(f"Dates: {len(dates)}")
    print(f"Spike threshold: {args.spike_threshold*100}%")
    print(f"Lookahead: {args.lookahead} minutes")
    print("="*80)
    print()

    # Find all spikes
    spike_events = find_spikes(
        args.db,
        symbols,
        dates,
        args.spike_threshold,
        args.lookahead
    )

    if len(spike_events) > 0:
        # Create report
        create_spike_report(spike_events, args.output_md)

        # Save JSON
        save_spike_json(spike_events, args.output_json)

        print()
        print("DONE!")
    else:
        print("No spikes found with the given parameters.")


if __name__ == '__main__':
    main()
