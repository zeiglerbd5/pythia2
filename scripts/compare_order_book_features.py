#!/usr/bin/env python3
"""
Compare order book features between Fast & Steep, Slow & Large, and Small Movers.
"""

import sqlite3
import pandas as pd
import numpy as np

def get_spike_categories(db_path: str):
    """Get spikes from Oct 18-20 and categorize them."""

    conn = sqlite3.connect(db_path)

    query = """
    SELECT symbol, timestamp
    FROM targets
    WHERE timeframe = '1m'
        AND target = 1
        AND timestamp >= '2025-10-18'
        AND timestamp < '2025-10-21'
    ORDER BY timestamp
    """

    spikes = pd.read_sql_query(query, conn)

    # Analyze each to categorize
    categories = {'Fast & Steep': [], 'Slow & Large': [], 'Small Mover': []}

    for _, row in spikes.iterrows():
        # Get peak gain
        candles_query = """
        SELECT c.close
        FROM candles c
        WHERE c.symbol = ?
            AND c.timestamp BETWEEN ? AND datetime(?, '+24 hours')
        ORDER BY c.timestamp
        """

        df = pd.read_sql_query(candles_query, conn, params=[row['symbol'], row['timestamp'], row['timestamp']])

        if len(df) == 0:
            continue

        signal_price = df['close'].iloc[0]
        df['gain_pct'] = ((df['close'] / signal_price) - 1) * 100
        df['minutes_elapsed'] = range(len(df))

        peak_gain = df['gain_pct'].max()
        time_to_peak = df['gain_pct'].idxmax()

        # Categorize
        if time_to_peak <= 30 and peak_gain >= 8:
            categories['Fast & Steep'].append((row['symbol'], row['timestamp'], peak_gain))
        elif time_to_peak > 30 and peak_gain >= 15:
            categories['Slow & Large'].append((row['symbol'], row['timestamp'], peak_gain))
        elif peak_gain < 8:
            categories['Small Mover'].append((row['symbol'], row['timestamp'], peak_gain))

    conn.close()
    return categories


def get_order_book_features(db_path: str, symbol: str, timestamp: str):
    """Get order book features for a specific spike."""

    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        f.buy_sell_ratio,
        f.order_flow_imbalance,
        f.vpin,
        f.bid_ask_spread_pct,
        f.order_book_depth_ratio,
        f.large_order_imbalance,
        f.trade_count,
        f.volume_zscore,
        f.RSI_14
    FROM features f
    WHERE f.symbol = ?
        AND strftime('%Y-%m-%d %H:%M:%S', f.timestamp) = ?
        AND f.timeframe = '1m'
    """

    df = pd.read_sql_query(query, conn, params=[symbol, timestamp])
    conn.close()

    if len(df) == 0:
        return None

    return df.iloc[0].to_dict()


def compare_categories(db_path: str):
    """Compare order book features across spike categories."""

    print("Categorizing spikes...")
    categories = get_spike_categories(db_path)

    print(f"Fast & Steep: {len(categories['Fast & Steep'])}")
    print(f"Slow & Large: {len(categories['Slow & Large'])}")
    print(f"Small Mover: {len(categories['Small Mover'])}")
    print()

    # Collect features for each category
    results = {}

    for cat_name, spikes in categories.items():
        print(f"Analyzing {cat_name}...")
        features_list = []

        for symbol, timestamp, peak_gain in spikes[:10]:  # Sample 10 from each
            features = get_order_book_features(db_path, symbol, timestamp)
            if features:
                features['peak_gain'] = peak_gain
                features['symbol'] = symbol
                features_list.append(features)

        if features_list:
            results[cat_name] = pd.DataFrame(features_list)

    # Print comparison
    print()
    print("=" * 100)
    print("ORDER BOOK FEATURES COMPARISON")
    print("=" * 100)
    print()

    metrics = [
        'buy_sell_ratio',
        'order_flow_imbalance',
        'vpin',
        'bid_ask_spread_pct',
        'order_book_depth_ratio',
        'large_order_imbalance',
        'trade_count',
        'volume_zscore',
        'RSI_14'
    ]

    print(f"{'Metric':<30} {'Fast & Steep':>15} {'Slow & Large':>15} {'Small Mover':>15}")
    print("-" * 100)

    for metric in metrics:
        row = f"{metric:<30}"
        for cat_name in ['Fast & Steep', 'Slow & Large', 'Small Mover']:
            if cat_name in results and metric in results[cat_name].columns:
                mean_val = results[cat_name][metric].mean()
                # Check if feature is populated
                non_zero_pct = (results[cat_name][metric] != 0).sum() / len(results[cat_name]) * 100
                row += f" {mean_val:>10.4f} ({non_zero_pct:>3.0f}%)"
            else:
                row += f" {'N/A':>15}"
        print(row)

    print()
    print("Note: Percentages show how many samples have non-zero values")
    print()

    # Show specific examples
    print("=" * 100)
    print("EXAMPLES")
    print("=" * 100)
    print()

    for cat_name in ['Fast & Steep', 'Small Mover']:
        if cat_name in results:
            print(f"{cat_name}:")
            df = results[cat_name].head(3)
            for idx, row in df.iterrows():
                print(f"  {row['symbol']} (Peak: {row['peak_gain']:.1f}%)")
                print(f"    Buy/Sell: {row['buy_sell_ratio']:.3f} | Trade Count: {row['trade_count']:.0f} | Volume Z: {row['volume_zscore']:.2f}")
                print(f"    Order Flow Imb: {row['order_flow_imbalance']:.3f} | Depth Ratio: {row['order_book_depth_ratio']:.3f}")
            print()


if __name__ == "__main__":
    db_path = "market_data copy_86.db"
    compare_categories(db_path)
