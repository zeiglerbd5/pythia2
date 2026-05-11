"""
Trade Microstructure Analysis

Goal: Detect volume spikes in the first 5-10 seconds of a minute,
before the full candle closes.

Features:
1. Trade rate (trades/sec) vs baseline
2. Volume velocity (USD/sec)
3. Buy/sell imbalance
4. Large trade detection
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import timedelta

DB_PATH = 'data/pythia_snapshot.duckdb'
SYMBOL = 'BTC-USD'

def get_minute_trade_stats(conn, symbol: str = 'BTC-USD'):
    """Get per-minute trade statistics."""
    return conn.execute(f'''
        SELECT
            date_trunc('minute', timestamp) as minute,
            COUNT(*) as total_trades,
            SUM(size * price) as total_usd,
            SUM(CASE WHEN side = 'BUY' THEN size * price ELSE 0 END) as buy_usd,
            SUM(CASE WHEN side = 'SELL' THEN size * price ELSE 0 END) as sell_usd
        FROM trades
        WHERE symbol = '{symbol}'
        GROUP BY 1
        ORDER BY 1
    ''').df()


def get_second_level_data(conn, symbol: str, start_time: str, end_time: str):
    """Get second-by-second trade aggregates."""
    return conn.execute(f'''
        SELECT
            date_trunc('second', timestamp) as second,
            date_trunc('minute', timestamp) as minute,
            EXTRACT(SECOND FROM timestamp)::INT as sec_of_min,
            COUNT(*) as trades,
            SUM(size) as volume,
            SUM(size * price) as usd_volume,
            SUM(CASE WHEN side = 'BUY' THEN size * price ELSE 0 END) as buy_usd,
            SUM(CASE WHEN side = 'SELL' THEN size * price ELSE 0 END) as sell_usd,
            MAX(size * price) as largest_trade_usd
        FROM trades
        WHERE symbol = '{symbol}'
          AND timestamp >= '{start_time}'
          AND timestamp < '{end_time}'
        GROUP BY 1, 2, 3
        ORDER BY 1
    ''').df()


def calculate_early_window_features(second_df: pd.DataFrame, window_seconds: int = 10):
    """
    For each minute, calculate features from the first N seconds
    and compare to the full minute's volume.
    """
    results = []

    for minute, group in second_df.groupby('minute'):
        group = group.sort_values('sec_of_min')

        # Full minute stats
        full_trades = group['trades'].sum()
        full_usd = group['usd_volume'].sum()
        full_buy = group['buy_usd'].sum()
        full_sell = group['sell_usd'].sum()

        if full_usd == 0:
            continue

        # Early window stats (first N seconds)
        early = group[group['sec_of_min'] < window_seconds]
        early_trades = early['trades'].sum() if len(early) > 0 else 0
        early_usd = early['usd_volume'].sum() if len(early) > 0 else 0
        early_buy = early['buy_usd'].sum() if len(early) > 0 else 0
        early_sell = early['sell_usd'].sum() if len(early) > 0 else 0
        early_largest = early['largest_trade_usd'].max() if len(early) > 0 else 0

        # Calculate features
        early_trade_rate = early_trades / window_seconds  # trades per second
        early_volume_rate = early_usd / window_seconds    # USD per second

        # Buy/sell imbalance (-1 = all sell, +1 = all buy)
        early_total = early_buy + early_sell
        early_imbalance = (early_buy - early_sell) / early_total if early_total > 0 else 0

        results.append({
            'minute': minute,
            'full_trades': full_trades,
            'full_usd': full_usd,
            'full_imbalance': (full_buy - full_sell) / (full_buy + full_sell) if (full_buy + full_sell) > 0 else 0,
            'early_trades': early_trades,
            'early_usd': early_usd,
            'early_trade_rate': early_trade_rate,
            'early_volume_rate': early_volume_rate,
            'early_imbalance': early_imbalance,
            'early_largest_trade': early_largest,
            'early_pct_of_full': early_usd / full_usd if full_usd > 0 else 0,
        })

    return pd.DataFrame(results)


def analyze_predictive_power(features_df: pd.DataFrame):
    """Analyze if early-window features predict full-minute volume."""

    # Define volume spike threshold (top 10% of minutes by volume)
    volume_threshold = features_df['full_usd'].quantile(0.90)
    features_df['is_spike'] = features_df['full_usd'] > volume_threshold

    print(f"Volume spike threshold (90th pct): ${volume_threshold:,.0f}")
    print(f"Spike minutes: {features_df['is_spike'].sum()} / {len(features_df)}")

    # For spike minutes, what did early features look like?
    spikes = features_df[features_df['is_spike']]
    non_spikes = features_df[~features_df['is_spike']]

    print("\n=== SPIKE vs NON-SPIKE COMPARISON ===")
    print(f"{'Metric':<25} {'Spikes':<15} {'Non-Spikes':<15} {'Ratio':<10}")
    print("-" * 65)

    metrics = ['early_trade_rate', 'early_volume_rate', 'early_imbalance', 'early_largest_trade']
    for m in metrics:
        spike_mean = spikes[m].mean()
        non_spike_mean = non_spikes[m].mean()
        ratio = spike_mean / non_spike_mean if non_spike_mean != 0 else float('inf')
        print(f"{m:<25} {spike_mean:<15.2f} {non_spike_mean:<15.2f} {ratio:<10.2f}x")

    # Test thresholds
    print("\n=== EARLY DETECTION ACCURACY ===")

    # If early_trade_rate > X, predict spike
    for threshold_pct in [75, 80, 85, 90]:
        threshold = features_df['early_trade_rate'].quantile(threshold_pct / 100)
        predicted_spike = features_df['early_trade_rate'] > threshold

        true_positives = (predicted_spike & features_df['is_spike']).sum()
        false_positives = (predicted_spike & ~features_df['is_spike']).sum()
        actual_spikes = features_df['is_spike'].sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / actual_spikes if actual_spikes > 0 else 0

        print(f"Trade rate > {threshold_pct}th pct ({threshold:.1f}/sec): "
              f"Precision={precision:.1%}, Recall={recall:.1%}")

    return features_df


def main():
    print("=" * 60)
    print("TRADE MICROSTRUCTURE ANALYSIS")
    print("=" * 60)

    conn = duckdb.connect(DB_PATH, read_only=True)

    # Get date range
    date_range = conn.execute(f'''
        SELECT MIN(timestamp), MAX(timestamp)
        FROM trades WHERE symbol = '{SYMBOL}'
    ''').fetchone()
    print(f"\nData range: {date_range[0]} to {date_range[1]}")

    # Get second-level data for analysis window
    # Use a few days of data
    start = '2026-01-29 00:00:00'
    end = '2026-01-31 00:00:00'

    print(f"Analyzing: {start} to {end}")
    print("Loading second-level data...")

    second_df = get_second_level_data(conn, SYMBOL, start, end)
    print(f"Loaded {len(second_df):,} second-level records")

    # Calculate early-window features
    print("\nCalculating early-window features (10-second window)...")
    features_df = calculate_early_window_features(second_df, window_seconds=10)
    print(f"Analyzed {len(features_df):,} minutes")

    # Analyze predictive power
    features_df = analyze_predictive_power(features_df)

    # Show example spike minutes
    print("\n=== EXAMPLE SPIKE MINUTES ===")
    spikes = features_df[features_df['is_spike']].nlargest(5, 'full_usd')
    for _, row in spikes.iterrows():
        print(f"\n{row['minute']}")
        print(f"  Full minute: {row['full_trades']:.0f} trades, ${row['full_usd']:,.0f}")
        print(f"  First 10 sec: {row['early_trades']:.0f} trades, ${row['early_usd']:,.0f} ({row['early_pct_of_full']:.1%} of total)")
        print(f"  Early trade rate: {row['early_trade_rate']:.1f}/sec")
        print(f"  Early imbalance: {row['early_imbalance']:+.2f} (-1=sell, +1=buy)")
        print(f"  Largest early trade: ${row['early_largest_trade']:,.0f}")

    # Correlation analysis
    print("\n=== CORRELATION: EARLY FEATURES vs FULL VOLUME ===")
    correlations = features_df[['early_trade_rate', 'early_volume_rate', 'early_largest_trade', 'full_usd']].corr()
    print(correlations['full_usd'].drop('full_usd').to_string())

    conn.close()

    # Save results
    features_df.to_csv('data/microstructure_features.csv', index=False)
    print("\nSaved features to data/microstructure_features.csv")

    return features_df


if __name__ == '__main__':
    df = main()
