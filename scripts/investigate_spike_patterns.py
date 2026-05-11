#!/usr/bin/env python3
"""
Investigate whether all 42 detected spikes follow the hypothesized 4-phase structure.

Hypothesis:
1. Warm-up phase (5-15 min): +5-10%, elevated volume
2. Consolidation (optional): Sideways movement
3. Explosive candles: 1-3 minutes with +10-15% each
4. Continuation: Smaller moves maintaining gains
"""

import sqlite3
import pandas as pd
from datetime import datetime

def analyze_spike_phases(db_path: str, symbol: str, spike_time: str):
    """
    Analyze a single spike for the 4-phase structure.

    Returns dict with phase analysis and narrative.
    """
    conn = sqlite3.connect(db_path)

    # Get 15 candles before, current, and 10 after
    # Note: Strip timezone from candles.timestamp for comparison with targets.timestamp
    query = """
    SELECT
        c.timestamp,
        c.close,
        c.volume,
        c.high,
        c.low,
        f.RSI_14,
        f.volume_zscore,
        COALESCE(t.target, 0) as is_spike
    FROM candles c
    JOIN features f ON c.timestamp = f.timestamp AND c.symbol = f.symbol
    LEFT JOIN targets t ON strftime('%Y-%m-%d %H:%M:%S', c.timestamp) = t.timestamp
        AND c.symbol = t.symbol AND t.timeframe = '1m'
    WHERE c.symbol = ?
        AND f.timeframe = '1m'
        AND c.timestamp BETWEEN datetime(?, '-15 minutes') AND datetime(?, '+10 minutes')
    ORDER BY c.timestamp
    """

    df = pd.read_sql_query(query, conn, params=[symbol, spike_time, spike_time])
    conn.close()

    if len(df) == 0:
        return None

    # Find signal candle
    signal_idx = df[df['is_spike'] == 1].index
    if len(signal_idx) == 0:
        return None
    signal_idx = signal_idx[0]

    # Calculate price changes
    df['price_change_pct'] = df['close'].pct_change() * 100
    df['cum_return'] = ((df['close'] / df['close'].iloc[0]) - 1) * 100

    # Get phases
    before_signal = df.iloc[:signal_idx]
    signal_candle = df.iloc[signal_idx]
    after_signal = df.iloc[signal_idx+1:signal_idx+11]

    # Analyze Phase 1: Warm-up (5-15 min before)
    warmup_window = before_signal.tail(15)
    if len(warmup_window) > 0:
        warmup_return = ((warmup_window['close'].iloc[-1] / warmup_window['close'].iloc[0]) - 1) * 100
        warmup_vol_avg = warmup_window['volume_zscore'].mean()
        warmup_vol_elevated = warmup_vol_avg > 0.5
        has_warmup = (5 <= warmup_return <= 15) and warmup_vol_elevated
    else:
        warmup_return = 0
        warmup_vol_elevated = False
        has_warmup = False

    # Analyze Phase 2: Consolidation (last 5 candles before signal)
    consol_window = before_signal.tail(5)
    if len(consol_window) >= 3:
        consol_range = consol_window['close'].max() - consol_window['close'].min()
        consol_pct = (consol_range / consol_window['close'].mean()) * 100
        has_consolidation = consol_pct < 2.0  # Less than 2% range
    else:
        consol_pct = 0
        has_consolidation = False

    # Analyze Phase 3: Explosive candles (1-3 candles after signal with 10-15%+ each)
    explosive_candles = []
    if len(after_signal) > 0:
        for i in range(min(3, len(after_signal))):
            candle_gain = after_signal.iloc[i]['price_change_pct']
            if candle_gain >= 10:
                explosive_candles.append((i+1, candle_gain))
    has_explosive = len(explosive_candles) > 0

    # Analyze Phase 4: Continuation (remaining candles maintain gains)
    if len(after_signal) >= 5:
        peak_price = after_signal['close'].max()
        end_price = after_signal['close'].iloc[-1]
        peak_gain = ((peak_price / signal_candle['close']) - 1) * 100
        end_gain = ((end_price / signal_candle['close']) - 1) * 100
        maintained_pct = (end_gain / peak_gain * 100) if peak_gain > 0 else 0
        has_continuation = maintained_pct >= 50  # Maintained at least 50% of peak gain
    else:
        peak_gain = 0
        end_gain = 0
        maintained_pct = 0
        has_continuation = False

    # Generate narrative
    narrative_parts = []

    if has_warmup:
        narrative_parts.append(f"Warm-up phase present: +{warmup_return:.1f}% over 15min with elevated volume")
    else:
        narrative_parts.append(f"No clear warm-up: +{warmup_return:.1f}% over 15min (expected 5-10%)")

    if has_consolidation:
        narrative_parts.append(f"Consolidation detected: {consol_pct:.1f}% range in last 5 candles")
    else:
        narrative_parts.append(f"No consolidation: {consol_pct:.1f}% range (continued movement)")

    if has_explosive:
        explosive_desc = ", ".join([f"+{gain:.1f}% (candle {idx})" for idx, gain in explosive_candles])
        narrative_parts.append(f"Explosive candles found: {explosive_desc}")
    else:
        max_gain = after_signal['price_change_pct'].max() if len(after_signal) > 0 else 0
        narrative_parts.append(f"No explosive candles: max single candle gain was {max_gain:.1f}%")

    if has_continuation:
        narrative_parts.append(f"Continuation present: maintained {maintained_pct:.0f}% of peak gain")
    else:
        narrative_parts.append(f"No continuation: only maintained {maintained_pct:.0f}% of peak (reversed)")

    return {
        'phases': {
            'warmup': has_warmup,
            'consolidation': has_consolidation,
            'explosive': has_explosive,
            'continuation': has_continuation
        },
        'metrics': {
            'warmup_return': warmup_return,
            'warmup_vol_elevated': warmup_vol_elevated,
            'consol_pct': consol_pct,
            'explosive_candles': explosive_candles,
            'peak_gain': peak_gain,
            'end_gain': end_gain,
            'maintained_pct': maintained_pct
        },
        'narrative': " | ".join(narrative_parts),
        'dataframe': df
    }


def generate_report(db_path: str, output_file: str):
    """Generate the investigation report."""

    conn = sqlite3.connect(db_path)

    # Get all 42 spikes
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

    print(f"Analyzing {len(spikes)} spikes...")

    # Analyze each spike
    results = []
    for idx, row in spikes.iterrows():
        print(f"  Analyzing spike {idx+1}/{len(spikes)}: {row['symbol']} @ {row['timestamp']}")
        analysis = analyze_spike_phases(db_path, row['symbol'], row['timestamp'])
        if analysis:
            results.append({
                'spike_num': idx + 1,
                'symbol': row['symbol'],
                'timestamp': row['timestamp'],
                'analysis': analysis
            })

    # Calculate summary stats
    total = len(results)
    warmup_count = sum(1 for r in results if r['analysis']['phases']['warmup'])
    consol_count = sum(1 for r in results if r['analysis']['phases']['consolidation'])
    explosive_count = sum(1 for r in results if r['analysis']['phases']['explosive'])
    continuation_count = sum(1 for r in results if r['analysis']['phases']['continuation'])
    all_phases_count = sum(1 for r in results if all(r['analysis']['phases'].values()))

    # Generate markdown
    md = []
    md.append("# Spike Pattern Investigation")
    md.append("")
    md.append(f"Analysis of {total} detected spikes from October 18-20, 2025")
    md.append("")
    md.append("## Hypothesis")
    md.append("")
    md.append("**All spikes follow this 4-phase structure:**")
    md.append("1. Warm-up phase (5-15 min): +5-10%, elevated volume")
    md.append("2. Consolidation (optional): Sideways movement")
    md.append("3. Explosive candles: 1-3 minutes with +10-15% each")
    md.append("4. Continuation: Smaller moves maintaining gains")
    md.append("")
    md.append("## Summary Findings")
    md.append("")
    md.append(f"- **Warm-up phase**: {warmup_count}/{total} ({warmup_count/total*100:.1f}%)")
    md.append(f"- **Consolidation**: {consol_count}/{total} ({consol_count/total*100:.1f}%)")
    md.append(f"- **Explosive candles**: {explosive_count}/{total} ({explosive_count/total*100:.1f}%)")
    md.append(f"- **Continuation**: {continuation_count}/{total} ({continuation_count/total*100:.1f}%)")
    md.append(f"- **All 4 phases present**: {all_phases_count}/{total} ({all_phases_count/total*100:.1f}%)")
    md.append("")
    md.append("## Individual Spike Analysis")
    md.append("")

    for result in results:
        spike_num = result['spike_num']
        symbol = result['symbol']
        timestamp = result['timestamp']
        analysis = result['analysis']
        phases = analysis['phases']
        metrics = analysis['metrics']
        narrative = analysis['narrative']

        md.append(f"### Spike #{spike_num}: {symbol} @ {timestamp}")
        md.append("")
        md.append("**Phases Present:**")
        md.append(f"- Warm-up: {'✓' if phases['warmup'] else '✗'}")
        md.append(f"- Consolidation: {'✓' if phases['consolidation'] else '✗'}")
        md.append(f"- Explosive: {'✓' if phases['explosive'] else '✗'}")
        md.append(f"- Continuation: {'✓' if phases['continuation'] else '✗'}")
        md.append("")
        md.append("**What Happened:**")
        md.append(f"{narrative}")
        md.append("")
        md.append("**Key Metrics:**")
        md.append(f"- Warm-up return: {metrics['warmup_return']:.1f}%")
        md.append(f"- Consolidation range: {metrics['consol_pct']:.1f}%")
        md.append(f"- Peak gain: {metrics['peak_gain']:.1f}%")
        md.append(f"- Final gain: {metrics['end_gain']:.1f}%")
        md.append("")
        md.append("---")
        md.append("")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(md))

    print(f"\nReport written to: {output_file}")
    print(f"\nSummary:")
    print(f"  Total spikes: {total}")
    print(f"  All 4 phases: {all_phases_count} ({all_phases_count/total*100:.1f}%)")
    print(f"  Warmup: {warmup_count} ({warmup_count/total*100:.1f}%)")
    print(f"  Explosive: {explosive_count} ({explosive_count/total*100:.1f}%)")


if __name__ == "__main__":
    db_path = "market_data copy_86.db"
    output_file = "SPIKE_INVESTIGATION.md"

    generate_report(db_path, output_file)
