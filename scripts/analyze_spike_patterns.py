#!/usr/bin/env python3
"""
Analyze actual spike patterns to understand their characteristics.
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse


def analyze_spike_characteristics(db_path: str, limit: int = 50):
    """Analyze characteristics of actual spikes in the database."""

    conn = duckdb.connect(db_path, read_only=True)

    print("=" * 80)
    print("SPIKE PATTERN ANALYSIS")
    print("=" * 80)
    print()

    # Get some actual spike examples
    query = """
    SELECT
        symbol,
        timestamp,
        target_label,
        future_returns_1h,
        future_max_return_1h,
        future_min_return_1h,
        close,
        volume,
        volume_delta,
        rsi,
        bb_position
    FROM features
    WHERE timeframe = '1m'
        AND target_label = 1
    ORDER BY future_max_return_1h DESC
    LIMIT ?
    """

    spikes = conn.execute(query, [limit]).fetchdf()

    print(f"Found {len(spikes)} spike examples")
    print()

    if len(spikes) == 0:
        print("No spikes found in database!")
        conn.close()
        return

    # Analyze spike characteristics
    print("SPIKE STATISTICS:")
    print("-" * 80)
    print(f"Future Max Return (1h):")
    print(f"  Mean: {spikes['future_max_return_1h'].mean():.2%}")
    print(f"  Median: {spikes['future_max_return_1h'].median():.2%}")
    print(f"  Min: {spikes['future_max_return_1h'].min():.2%}")
    print(f"  Max: {spikes['future_max_return_1h'].max():.2%}")
    print()

    print(f"Future Returns (1h end point):")
    print(f"  Mean: {spikes['future_returns_1h'].mean():.2%}")
    print(f"  Median: {spikes['future_returns_1h'].median():.2%}")
    print(f"  Min: {spikes['future_returns_1h'].min():.2%}")
    print(f"  Max: {spikes['future_returns_1h'].max():.2%}")
    print()

    print(f"Future Min Return (1h - drawdown):")
    print(f"  Mean: {spikes['future_min_return_1h'].mean():.2%}")
    print(f"  Median: {spikes['future_min_return_1h'].median():.2%}")
    print(f"  Min: {spikes['future_min_return_1h'].min():.2%}")
    print(f"  Max: {spikes['future_min_return_1h'].max():.2%}")
    print()

    # Volume analysis
    print(f"Volume Delta:")
    print(f"  Mean: {spikes['volume_delta'].mean():.2%}")
    print(f"  Median: {spikes['volume_delta'].median():.2%}")
    print()

    # RSI analysis
    print(f"RSI at spike:")
    print(f"  Mean: {spikes['rsi'].mean():.1f}")
    print(f"  Median: {spikes['rsi'].median():.1f}")
    print()

    print("=" * 80)
    print("TOP 10 LARGEST SPIKES:")
    print("=" * 80)
    print()

    for idx, row in spikes.head(10).iterrows():
        print(f"{idx+1}. {row['symbol']} @ {row['timestamp']}")
        print(f"   Max Return: {row['future_max_return_1h']:.2%}")
        print(f"   End Return: {row['future_returns_1h']:.2%}")
        print(f"   Min Return: {row['future_min_return_1h']:.2%}")
        print(f"   Volume Delta: {row['volume_delta']:.2%}")
        print(f"   RSI: {row['rsi']:.1f}")
        print()

    # Now let's get the pre-spike candle data for a few examples
    print("=" * 80)
    print("DETAILED PRE-SPIKE ANALYSIS (Top 4 examples):")
    print("=" * 80)
    print()

    for idx, spike_row in spikes.head(4).iterrows():
        symbol = spike_row['symbol']
        spike_time = spike_row['timestamp']

        print(f"\n{'=' * 80}")
        print(f"SPIKE #{idx+1}: {symbol} @ {spike_time}")
        print(f"Max Return: {spike_row['future_max_return_1h']:.2%}")
        print(f"{'=' * 80}\n")

        # Get the 10 candles BEFORE the spike
        pre_spike_query = """
        SELECT
            timestamp,
            close,
            volume,
            volume_delta,
            rsi,
            macd,
            bb_position,
            target_label,
            future_max_return_1h
        FROM features
        WHERE symbol = ?
            AND timeframe = '1m'
            AND timestamp <= ?
        ORDER BY timestamp DESC
        LIMIT 10
        """

        pre_candles = conn.execute(
            pre_spike_query,
            [symbol, spike_time]
        ).fetchdf()

        if len(pre_candles) > 0:
            pre_candles = pre_candles.sort_values('timestamp')

            print("PRE-SPIKE CANDLES (last 10):")
            print("-" * 80)
            print(f"{'Time':<20} {'Close':>10} {'Vol Δ':>10} {'RSI':>6} {'MACD':>8} {'BB Pos':>8} {'Label':>6}")
            print("-" * 80)

            for _, candle in pre_candles.iterrows():
                label_str = "SPIKE" if candle['target_label'] == 1 else ""
                print(f"{str(candle['timestamp']):<20} "
                      f"{candle['close']:>10.4f} "
                      f"{candle['volume_delta']:>9.1%} "
                      f"{candle['rsi']:>6.1f} "
                      f"{candle['macd']:>8.4f} "
                      f"{candle['bb_position']:>8.2f} "
                      f"{label_str:>6}")

            print()

            # Calculate changes leading up to spike
            if len(pre_candles) >= 2:
                price_changes = pre_candles['close'].pct_change()
                print("PRICE MOMENTUM:")
                print(f"  Last 1 candle: {price_changes.iloc[-1]:.2%}")
                print(f"  Last 2 candles: {price_changes.iloc[-2:].sum():.2%}")
                print(f"  Last 5 candles: {price_changes.iloc[-5:].sum():.2%}")
                print()

                print("VOLUME TREND:")
                print(f"  Avg vol delta (last 5): {pre_candles['volume_delta'].tail(5).mean():.2%}")
                print()

                print("RSI TREND:")
                print(f"  RSI at spike: {pre_candles['rsi'].iloc[-1]:.1f}")
                print(f"  RSI 5 candles before: {pre_candles['rsi'].iloc[-5]:.1f}")
                print(f"  RSI change: {pre_candles['rsi'].iloc[-1] - pre_candles['rsi'].iloc[-5]:.1f}")
                print()

    conn.close()


def compare_spike_vs_nonspike(db_path: str, symbol: str = None):
    """Compare pre-spike patterns vs non-spike patterns."""

    conn = duckdb.connect(db_path, read_only=True)

    print("=" * 80)
    print("SPIKE vs NON-SPIKE COMPARISON")
    print("=" * 80)
    print()

    # Get statistics for spike candles
    if symbol:
        spike_query = """
        SELECT
            volume_delta,
            rsi,
            macd,
            bb_position,
            close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp) as price_change
        FROM features
        WHERE timeframe = '1m'
            AND target_label = 1
            AND symbol = ?
        """
        spike_stats = conn.execute(spike_query, [symbol]).fetchdf()
    else:
        spike_query = """
        SELECT
            volume_delta,
            rsi,
            macd,
            bb_position
        FROM features
        WHERE timeframe = '1m'
            AND target_label = 1
        LIMIT 1000
        """
        spike_stats = conn.execute(spike_query).fetchdf()

    # Get statistics for non-spike candles
    if symbol:
        nonspike_query = """
        SELECT
            volume_delta,
            rsi,
            macd,
            bb_position
        FROM features
        WHERE timeframe = '1m'
            AND target_label = 0
            AND symbol = ?
        LIMIT 10000
        """
        nonspike_stats = conn.execute(nonspike_query, [symbol]).fetchdf()
    else:
        nonspike_query = """
        SELECT
            volume_delta,
            rsi,
            macd,
            bb_position
        FROM features
        WHERE timeframe = '1m'
            AND target_label = 0
        LIMIT 10000
        """
        nonspike_stats = conn.execute(nonspike_query).fetchdf()

    print(f"Spike candles: {len(spike_stats)}")
    print(f"Non-spike candles: {len(nonspike_stats)}")
    print()

    print(f"{'Metric':<20} {'Spike Mean':>15} {'Non-Spike Mean':>15} {'Difference':>15}")
    print("-" * 80)

    for col in ['volume_delta', 'rsi', 'macd', 'bb_position']:
        spike_mean = spike_stats[col].mean()
        nonspike_mean = nonspike_stats[col].mean()
        diff = spike_mean - nonspike_mean

        if col == 'volume_delta':
            print(f"{col:<20} {spike_mean:>14.2%} {nonspike_mean:>14.2%} {diff:>14.2%}")
        else:
            print(f"{col:<20} {spike_mean:>15.4f} {nonspike_mean:>15.4f} {diff:>15.4f}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze spike patterns')
    parser.add_argument('--db', required=True, help='Database path')
    parser.add_argument('--limit', type=int, default=50, help='Number of spike examples to analyze')
    parser.add_argument('--symbol', help='Optional: analyze specific symbol')
    args = parser.parse_args()

    analyze_spike_characteristics(args.db, args.limit)
    print("\n" * 2)
    compare_spike_vs_nonspike(args.db, args.symbol)
