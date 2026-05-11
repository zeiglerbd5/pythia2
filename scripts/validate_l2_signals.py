#!/usr/bin/env python3
"""
L2 Signal Validation Script for V4 Reactive Strategy

Validates whether L2 signals (bid_ask_ratio, ask_depth collapse) are
PREDICTIVE (appear before price spikes) vs DESCRIPTIVE (appear during spikes).

Uses historical data from Sept-Oct 2025 where we have both:
- candles table (price data)
- order_book_features table (L2 data)
"""

import duckdb
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

# Archive database path
DB_PATH = "/Users/bz/Pythia2/market_data.duckdb copy"


def find_spikes(conn, min_gain_pct=20):
    """Find all hourly windows with 20%+ price gains."""
    print(f"\n=== Finding {min_gain_pct}%+ Spikes ===")

    query = f"""
        WITH hourly_stats AS (
            SELECT
                symbol,
                DATE_TRUNC('hour', timestamp) as hour,
                FIRST(open ORDER BY timestamp) as open_price,
                MAX(high) as max_high,
                FIRST(timestamp ORDER BY timestamp) as first_ts,
                LAST(timestamp ORDER BY timestamp) as last_ts
            FROM candles
            GROUP BY symbol, DATE_TRUNC('hour', timestamp)
        )
        SELECT
            symbol,
            hour,
            open_price,
            max_high,
            (max_high / open_price - 1) * 100 as gain_pct,
            first_ts,
            last_ts
        FROM hourly_stats
        WHERE (max_high / open_price - 1) >= {min_gain_pct / 100}
        ORDER BY gain_pct DESC
    """

    spikes = conn.execute(query).fetchall()
    print(f"Found {len(spikes)} spikes of {min_gain_pct}%+")

    return spikes


def analyze_l2_signals_for_spike(conn, symbol, spike_hour, open_price, max_high):
    """
    Analyze L2 signals in the 60 minutes before and during a spike.
    Returns signal analysis including lead time and whether it was predictive.
    """
    # Look at 90 minutes before spike hour to 30 min after spike hour starts
    start_time = spike_hour - timedelta(hours=1.5)
    end_time = spike_hour + timedelta(minutes=30)

    query = f"""
        SELECT
            timestamp::TIMESTAMP as ts,
            bid_ask_ratio,
            bid_depth_10,
            ask_depth_10,
            bid_price,
            ask_price
        FROM order_book_features
        WHERE symbol = '{symbol}'
          AND timestamp::TIMESTAMP >= '{start_time}'
          AND timestamp::TIMESTAMP <= '{end_time}'
        ORDER BY timestamp
    """

    records = conn.execute(query).fetchall()

    if not records:
        return None

    # Calculate rolling baseline (first 30 minutes)
    baseline_records = [r for r in records if r[0] < spike_hour - timedelta(minutes=30)]

    if len(baseline_records) < 10:
        return None

    baseline_bar = statistics.mean([r[1] for r in baseline_records if r[1]])
    baseline_ask_depth = statistics.mean([r[3] for r in baseline_records if r[3]])

    if baseline_bar == 0 or baseline_ask_depth == 0:
        return None

    # Find first significant signal
    signal_time = None
    signal_bar_multiple = None
    signal_ask_collapse = None
    signal_price = None

    for record in records:
        ts, bar, bid_depth, ask_depth, bid_price, ask_price = record

        if not bar or not ask_depth:
            continue

        bar_multiple = bar / baseline_bar
        ask_collapse = ask_depth / baseline_ask_depth

        # Signal conditions: BAR > 3x baseline AND ask depth < 50% baseline
        if bar_multiple > 3 and ask_collapse < 0.5:
            # Check if price hasn't already moved much
            price = (bid_price + ask_price) / 2 if bid_price and ask_price else None

            if price and open_price:
                price_change = (price - open_price) / open_price

                # Predictive filter: price must not have moved more than 5% yet
                if price_change < 0.05:
                    signal_time = ts
                    signal_bar_multiple = bar_multiple
                    signal_ask_collapse = ask_collapse
                    signal_price = price
                    break

    if not signal_time:
        return {
            'found_signal': False,
            'symbol': symbol,
            'spike_hour': spike_hour
        }

    # Calculate lead time (minutes before spike hour)
    lead_time_minutes = (spike_hour - signal_time).total_seconds() / 60

    return {
        'found_signal': True,
        'symbol': symbol,
        'spike_hour': spike_hour,
        'signal_time': signal_time,
        'lead_time_minutes': lead_time_minutes,
        'bar_multiple': signal_bar_multiple,
        'ask_collapse': signal_ask_collapse,
        'signal_price': signal_price,
        'open_price': open_price,
        'max_high': max_high,
        'is_predictive': lead_time_minutes > 5  # Signal came > 5 min before
    }


def analyze_false_positives(conn, known_spike_hours, sample_size=1000):
    """
    Sample random symbol-hours with high BAR signals that are NOT spikes.
    Calculate false positive rate.
    """
    print("\n=== Analyzing False Positives ===")

    # Get symbol-hours with high relative BAR but NOT in spike list
    query = """
        WITH high_bar_hours AS (
            SELECT DISTINCT
                symbol,
                DATE_TRUNC('hour', timestamp::TIMESTAMP) as hour
            FROM order_book_features
            WHERE bid_ask_ratio > 5
        )
        SELECT symbol, hour
        FROM high_bar_hours
        ORDER BY RANDOM()
        LIMIT 2000
    """

    potential_fps = conn.execute(query).fetchall()

    # Filter out known spikes
    spike_set = set(known_spike_hours)
    non_spikes = [(s, h) for s, h in potential_fps if (s, h) not in spike_set][:sample_size]

    print(f"Sampling {len(non_spikes)} high-BAR hours that are NOT known spikes")

    # Check how many of these actually had price gains
    false_positives = 0
    true_negatives = 0

    for symbol, hour in non_spikes[:100]:  # Check first 100
        query = f"""
            SELECT
                FIRST(open ORDER BY timestamp) as open_price,
                MAX(high) as max_high
            FROM candles
            WHERE symbol = '{symbol}'
              AND DATE_TRUNC('hour', timestamp) = '{hour}'
        """
        result = conn.execute(query).fetchone()

        if result and result[0] and result[1]:
            gain = (result[1] - result[0]) / result[0]
            if gain < 0.10:  # Less than 10% gain = not a spike
                false_positives += 1
            else:
                true_negatives += 1  # Actually was a spike we missed

    checked = false_positives + true_negatives
    if checked > 0:
        fp_rate = false_positives / checked
        print(f"False positive rate: {fp_rate:.1%} ({false_positives}/{checked})")

    return false_positives, true_negatives


def main():
    print("=" * 60)
    print("L2 SIGNAL VALIDATION FOR V4 REACTIVE STRATEGY")
    print("=" * 60)

    conn = duckdb.connect(DB_PATH, read_only=True)

    # Find all significant spikes
    spikes = find_spikes(conn, min_gain_pct=20)

    # Analyze L2 signals for each spike
    print("\n=== Analyzing L2 Signals Before Spikes ===")

    results = []
    found_signals = 0
    predictive_signals = 0
    lead_times = []

    for i, spike in enumerate(spikes[:50]):  # Analyze top 50 spikes
        symbol, hour, open_price, max_high, gain_pct, first_ts, last_ts = spike

        result = analyze_l2_signals_for_spike(conn, symbol, hour, open_price, max_high)

        if result:
            results.append(result)

            if result['found_signal']:
                found_signals += 1
                lead_times.append(result['lead_time_minutes'])

                if result['is_predictive']:
                    predictive_signals += 1
                    status = "PREDICTIVE"
                else:
                    status = "DESCRIPTIVE"

                print(f"  {symbol} +{gain_pct:.0f}%: {status} (lead={result['lead_time_minutes']:.0f}min, BAR={result['bar_multiple']:.1f}x)")
            else:
                print(f"  {symbol} +{gain_pct:.0f}%: NO SIGNAL FOUND")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_analyzed = len(results)
    hit_rate = found_signals / total_analyzed if total_analyzed > 0 else 0
    predictive_rate = predictive_signals / found_signals if found_signals > 0 else 0

    print(f"\nTotal spikes analyzed: {total_analyzed}")
    print(f"Spikes with L2 signal: {found_signals} ({hit_rate:.1%} hit rate)")
    print(f"Predictive signals (>5min lead): {predictive_signals} ({predictive_rate:.1%} of signals)")

    if lead_times:
        print(f"\nLead time distribution:")
        print(f"  Min: {min(lead_times):.0f} min")
        print(f"  Max: {max(lead_times):.0f} min")
        print(f"  Median: {statistics.median(lead_times):.0f} min")
        print(f"  Mean: {statistics.mean(lead_times):.0f} min")

    # False positive analysis
    known_spike_hours = [(s[0], s[1]) for s in spikes]
    fp_count, tn_count = analyze_false_positives(conn, known_spike_hours)

    print("\n" + "=" * 60)
    print("VALIDATION VERDICT")
    print("=" * 60)

    if hit_rate >= 0.3 and predictive_rate >= 0.5:
        print("\nRESULT: SIGNALS ARE VIABLE")
        print(f"  - Hit rate {hit_rate:.1%} >= 30% target")
        print(f"  - Predictive rate {predictive_rate:.1%} >= 50% target")
        print("  - Proceed with implementation!")
    else:
        print("\nRESULT: SIGNALS NEED REFINEMENT")
        if hit_rate < 0.3:
            print(f"  - Hit rate {hit_rate:.1%} < 30% target")
        if predictive_rate < 0.5:
            print(f"  - Predictive rate {predictive_rate:.1%} < 50% target")
        print("  - Consider adjusting thresholds or adding signals")

    conn.close()

    return results


if __name__ == "__main__":
    main()
