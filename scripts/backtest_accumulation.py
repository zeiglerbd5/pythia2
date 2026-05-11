#!/usr/bin/env python3
"""
Accumulation Hunter Backtest

Analyzes historical order book data to find accumulation patterns
and tracks their outcomes (spike vs fizzle).

Run overnight - takes several hours on 86GB database.

Usage:
    python scripts/backtest_accumulation.py

Output:
    - Console progress updates
    - accumulation_backtest_results.json (detailed results)
    - accumulation_backtest_summary.txt (human-readable summary)
"""

import duckdb
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Configuration
DB_PATH = '/Users/bz/Pythia2/full_pythia.duckdb'
OUTPUT_DIR = Path('/Users/bz/Pythia2')

# Detection thresholds (same as strategy)
BAR_WATCH = 3.0       # Bid/ask ratio multiple for watch
BAR_STRONG = 5.0      # Strong signal
ASK_COLLAPSE_MIN = 0.50  # 50% ask depth collapse
MIN_HOURS = 2         # Minimum accumulation duration

# Outcome thresholds
SPIKE_THRESHOLD = 0.20    # 20% = spike
MODEST_THRESHOLD = 0.10   # 10% = modest gain
LOOKFORWARD_HOURS = 72    # Check outcomes within 72 hours


def parse_order_book(bids_json, asks_json, top_n=10):
    """Parse order book JSON and calculate depths."""
    try:
        bids = json.loads(bids_json) if bids_json else []
        asks = json.loads(asks_json) if asks_json else []

        bid_depth = sum(float(b[1]) for b in bids[:top_n])
        ask_depth = sum(float(a[1]) for a in asks[:top_n])

        return bid_depth, ask_depth
    except:
        return 0, 0


def find_accumulation_patterns(conn, symbol, progress_callback=None):
    """Find accumulation patterns for a symbol."""
    patterns = []

    # Get hourly aggregated order book data
    query = """
        SELECT
            DATE_TRUNC('hour', timestamp) as hour,
            FIRST(bids ORDER BY timestamp) as bids,
            FIRST(asks ORDER BY timestamp) as asks,
            AVG(mid_price) as price,
            COUNT(*) as samples
        FROM order_book_snapshots
        WHERE symbol = ?
        GROUP BY DATE_TRUNC('hour', timestamp)
        ORDER BY hour
    """

    df = conn.execute(query, [symbol]).fetchdf()

    if len(df) < 48:  # Need at least 48 hours
        return patterns

    # Calculate depths and BAR
    depths = []
    for _, row in df.iterrows():
        bid_depth, ask_depth = parse_order_book(row['bids'], row['asks'])
        if ask_depth > 0:
            depths.append({
                'hour': row['hour'],
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'bar': bid_depth / ask_depth,
                'price': row['price']
            })

    if len(depths) < 24:
        return patterns

    depth_df = pd.DataFrame(depths)

    # Calculate rolling 24h baseline
    depth_df['bar_baseline'] = depth_df['bar'].rolling(24, min_periods=12).mean().shift(1)
    depth_df['ask_baseline'] = depth_df['ask_depth'].rolling(24, min_periods=12).mean().shift(1)

    # Calculate multiples
    depth_df['bar_multiple'] = depth_df['bar'] / depth_df['bar_baseline']
    depth_df['ask_collapse'] = 1 - (depth_df['ask_depth'] / depth_df['ask_baseline'])

    # Drop NaN rows
    depth_df = depth_df.dropna()

    if len(depth_df) < 24:
        return patterns

    # Find accumulation signals
    in_accumulation = False
    accum_start = None
    accum_start_price = None
    max_bar_multiple = 0
    max_ask_collapse = 0

    for i, row in depth_df.iterrows():
        is_signal = (row['bar_multiple'] >= BAR_WATCH and
                     row['ask_collapse'] >= ASK_COLLAPSE_MIN)

        if is_signal and not in_accumulation:
            # Start new accumulation period
            in_accumulation = True
            accum_start = row['hour']
            accum_start_price = row['price']
            max_bar_multiple = row['bar_multiple']
            max_ask_collapse = row['ask_collapse']

        elif is_signal and in_accumulation:
            # Continue accumulation
            max_bar_multiple = max(max_bar_multiple, row['bar_multiple'])
            max_ask_collapse = max(max_ask_collapse, row['ask_collapse'])

        elif not is_signal and in_accumulation:
            # End accumulation period
            accum_end = row['hour']
            duration_hours = (accum_end - accum_start).total_seconds() / 3600

            if duration_hours >= MIN_HOURS:
                # Valid accumulation pattern - check outcome
                future_df = depth_df[depth_df['hour'] > accum_end].head(LOOKFORWARD_HOURS)

                if len(future_df) >= 12:
                    max_price = future_df['price'].max()
                    min_price = future_df['price'].min()
                    end_price = future_df['price'].iloc[-1] if len(future_df) > 0 else accum_start_price

                    max_gain = (max_price - accum_start_price) / accum_start_price
                    max_loss = (min_price - accum_start_price) / accum_start_price
                    final_return = (end_price - accum_start_price) / accum_start_price

                    # Determine outcome
                    if max_gain >= SPIKE_THRESHOLD:
                        outcome = 'spike'
                    elif max_gain >= MODEST_THRESHOLD:
                        outcome = 'modest'
                    else:
                        outcome = 'fizzle'

                    patterns.append({
                        'symbol': symbol,
                        'start_time': accum_start.isoformat(),
                        'end_time': accum_end.isoformat(),
                        'duration_hours': duration_hours,
                        'start_price': accum_start_price,
                        'max_bar_multiple': max_bar_multiple,
                        'max_ask_collapse': max_ask_collapse,
                        'max_gain_pct': max_gain * 100,
                        'max_loss_pct': max_loss * 100,
                        'final_return_pct': final_return * 100,
                        'outcome': outcome,
                    })

            in_accumulation = False
            max_bar_multiple = 0
            max_ask_collapse = 0

    return patterns


def main():
    print("="*70)
    print("ACCUMULATION HUNTER BACKTEST")
    print("="*70)
    print(f"Database: {DB_PATH}")
    print(f"Started: {datetime.now()}")
    print()

    conn = duckdb.connect(DB_PATH, read_only=True)

    # Get all symbols
    symbols = conn.execute("""
        SELECT symbol, COUNT(*) as cnt
        FROM order_book_snapshots
        WHERE bids IS NOT NULL
        GROUP BY symbol
        HAVING COUNT(*) >= 100
        ORDER BY cnt DESC
    """).fetchdf()

    total_symbols = len(symbols)
    print(f"Analyzing {total_symbols} symbols with sufficient data...")
    print()

    all_patterns = []

    for idx, row in symbols.iterrows():
        symbol = row['symbol']

        # Progress update
        if idx % 10 == 0:
            pct = idx / total_symbols * 100
            print(f"[{pct:5.1f}%] Processing {symbol}... ({len(all_patterns)} patterns found)")

        try:
            patterns = find_accumulation_patterns(conn, symbol)
            all_patterns.extend(patterns)
        except Exception as e:
            print(f"  Error on {symbol}: {e}")
            continue

    conn.close()

    print()
    print(f"Analysis complete. Found {len(all_patterns)} accumulation patterns.")
    print()

    # Generate summary
    if len(all_patterns) == 0:
        print("No patterns found!")
        return

    df = pd.DataFrame(all_patterns)

    # Overall stats
    total = len(df)
    spikes = (df['outcome'] == 'spike').sum()
    modest = (df['outcome'] == 'modest').sum()
    fizzles = (df['outcome'] == 'fizzle').sum()

    summary = []
    summary.append("="*70)
    summary.append("BACKTEST RESULTS SUMMARY")
    summary.append("="*70)
    summary.append(f"Total patterns found: {total}")
    summary.append(f"Date range: {df['start_time'].min()} to {df['start_time'].max()}")
    summary.append("")
    summary.append("OUTCOME BREAKDOWN:")
    summary.append(f"  Spike (20%+):  {spikes:4d} ({spikes/total*100:5.1f}%)")
    summary.append(f"  Modest (10%+): {modest:4d} ({modest/total*100:5.1f}%)")
    summary.append(f"  Fizzle (<10%): {fizzles:4d} ({fizzles/total*100:5.1f}%)")
    summary.append("")
    summary.append(f"SUCCESS RATE (10%+ gain): {(spikes+modest)/total*100:.1f}%")
    summary.append(f"FIZZLE RATE:              {fizzles/total*100:.1f}%")
    summary.append("")

    # By signal strength
    summary.append("BY SIGNAL STRENGTH:")
    for bar_thresh in [3.0, 5.0, 8.0]:
        subset = df[df['max_bar_multiple'] >= bar_thresh]
        if len(subset) > 0:
            sub_success = ((subset['outcome'] == 'spike') | (subset['outcome'] == 'modest')).sum()
            summary.append(f"  BAR >= {bar_thresh}x: {len(subset):4d} patterns, {sub_success/len(subset)*100:.1f}% success")

    summary.append("")
    summary.append("TOP 10 SPIKES:")
    for _, row in df.nlargest(10, 'max_gain_pct').iterrows():
        summary.append(f"  {row['symbol']}: BAR {row['max_bar_multiple']:.1f}x, "
                      f"Collapse {row['max_ask_collapse']*100:.0f}% → +{row['max_gain_pct']:.0f}%")

    summary.append("")
    summary.append("WORST 10 FIZZLES:")
    for _, row in df.nsmallest(10, 'max_gain_pct').iterrows():
        summary.append(f"  {row['symbol']}: BAR {row['max_bar_multiple']:.1f}x, "
                      f"Collapse {row['max_ask_collapse']*100:.0f}% → {row['max_gain_pct']:+.0f}%")

    summary.append("")
    summary.append("="*70)
    summary.append(f"Completed: {datetime.now()}")

    # Print summary
    print("\n".join(summary))

    # Save results
    results_file = OUTPUT_DIR / 'accumulation_backtest_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_patterns, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    summary_file = OUTPUT_DIR / 'accumulation_backtest_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("\n".join(summary))
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
