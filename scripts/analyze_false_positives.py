#!/usr/bin/env python3
"""
False Positive & Early Entry Analysis

Core question: At the moment a coin starts moving (e.g. +5% in 1h), what
distinguishes the ones that become slow_large (20%+) from the ones that fizzle?

Approach:
1. Find ALL instances where a coin moves +5% in a 1h window (trigger events)
2. Track forward: what happens in next 6h, 12h, 24h?
3. Label: winner (20%+ forward gain) vs fizzle (<10% forward or reversal)
4. Compare pre-trigger context between winners and fizzles
5. Special focus on repeat spikers vs one-timers
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import duckdb
import pandas as pd
import numpy as np
from datetime import timedelta
from collections import defaultdict

DB_PATH = "data/research.duckdb"
TRIGGER_PCT = 5.0       # Initial move to trigger analysis
TRIGGER_WINDOW = 12      # 12 x 5min = 1 hour
WINNER_THRESHOLD = 20.0  # 20%+ forward = winner
FIZZLE_THRESHOLD = 10.0  # <10% forward = fizzle
FORWARD_BARS = 288       # 24h of 5m bars
COOLDOWN_BARS = 36       # 3h cooldown between triggers per symbol


def find_trigger_events(db):
    """Find all instances where a coin moves +5% within 1 hour."""
    print("Finding trigger events (5%+ in 1h)...")

    # Get all 5m candles
    candles = db.execute("""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE timeframe = '5m'
        ORDER BY symbol, timestamp
    """).df()

    print(f"  Total candles: {len(candles):,}")

    triggers = []
    for symbol, group in candles.groupby('symbol'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        if len(group) < TRIGGER_WINDOW + FORWARD_BARS:
            continue

        close = group['close'].values
        high = group['high'].values
        low = group['low'].values
        volume = group['volume'].values
        timestamps = group['timestamp'].values

        last_trigger_idx = -COOLDOWN_BARS

        for i in range(TRIGGER_WINDOW, len(group) - FORWARD_BARS):
            # Already triggered recently?
            if i - last_trigger_idx < COOLDOWN_BARS:
                continue

            # Check: has price moved +5% from the low of the last 12 bars?
            window_low = low[i-TRIGGER_WINDOW:i].min()
            current = close[i]

            if window_low <= 0:
                continue

            move_pct = (current - window_low) / window_low * 100

            if move_pct >= TRIGGER_PCT:
                # TRIGGER! Now track forward
                entry_price = close[i]
                forward_slice = group.iloc[i:i+FORWARD_BARS]

                # Forward metrics
                fwd_high = forward_slice['high'].values
                fwd_close = forward_slice['close'].values

                # Max gain from entry
                max_fwd_gain = (fwd_high.max() - entry_price) / entry_price * 100
                # Time to max gain
                max_idx = fwd_high.argmax()
                time_to_peak_hours = max_idx * 5 / 60

                # Gains at specific horizons
                gain_1h = (fwd_close[min(11, len(fwd_close)-1)] - entry_price) / entry_price * 100
                gain_3h = (fwd_close[min(35, len(fwd_close)-1)] - entry_price) / entry_price * 100
                gain_6h = (fwd_close[min(71, len(fwd_close)-1)] - entry_price) / entry_price * 100
                gain_12h = (fwd_close[min(143, len(fwd_close)-1)] - entry_price) / entry_price * 100
                gain_24h = (fwd_close[min(287, len(fwd_close)-1)] - entry_price) / entry_price * 100

                # Max drawdown from entry before peak
                pre_peak_low = fwd_close[:max(max_idx, 1)].min()
                max_dd = (pre_peak_low - entry_price) / entry_price * 100

                # Did it close above entry after 1h?
                close_1h_above = gain_1h > 0

                # Pre-trigger context (6h = 72 bars before)
                pre_start = max(0, i - 72)
                pre_slice = group.iloc[pre_start:i]
                pre_close = pre_slice['close'].values
                pre_volume = pre_slice['volume'].values
                pre_high = pre_slice['high'].values
                pre_low = pre_slice['low'].values

                # Pre-trigger features
                if len(pre_close) >= 12:
                    pre_vol_early = pre_volume[:len(pre_volume)//2].mean()
                    pre_vol_late = pre_volume[len(pre_volume)//2:].mean()
                    vol_trend = pre_vol_late / pre_vol_early if pre_vol_early > 0 else 1.0

                    pre_returns = np.diff(pre_close) / pre_close[:-1]
                    pre_volatility = pre_returns.std() * 100

                    # Price compression: vol of last 1h vs vol of full 6h
                    if len(pre_returns) >= 12:
                        vol_1h = pre_returns[-12:].std()
                        vol_6h = pre_returns.std()
                        compression = vol_1h / vol_6h if vol_6h > 0 else 1.0
                    else:
                        compression = 1.0

                    # Momentum: how much did it move in the 3h before trigger
                    pre_momentum_3h = (pre_close[-1] / pre_close[-min(36, len(pre_close))] - 1) * 100

                    # Volume in trigger window vs pre-trigger average
                    trigger_vol = volume[i-TRIGGER_WINDOW:i].mean()
                    pre_avg_vol = pre_volume.mean()
                    trigger_vol_ratio = trigger_vol / pre_avg_vol if pre_avg_vol > 0 else 1.0

                    # How much of the trigger move happened in the LAST 15 min (3 bars)?
                    if i >= 3:
                        last_15m_move = (close[i] - close[i-3]) / close[i-3] * 100
                    else:
                        last_15m_move = 0

                    # Average price level (proxy for market cap)
                    avg_price = pre_close.mean()
                else:
                    vol_trend = compression = pre_volatility = pre_momentum_3h = None
                    trigger_vol_ratio = last_15m_move = avg_price = None

                triggers.append({
                    'symbol': symbol,
                    'trigger_time': timestamps[i],
                    'trigger_move_pct': move_pct,
                    'entry_price': entry_price,
                    'max_fwd_gain': max_fwd_gain,
                    'time_to_peak_hours': time_to_peak_hours,
                    'max_dd_before_peak': max_dd,
                    'gain_1h': gain_1h,
                    'gain_3h': gain_3h,
                    'gain_6h': gain_6h,
                    'gain_12h': gain_12h,
                    'gain_24h': gain_24h,
                    'close_1h_above': close_1h_above,
                    'pre_vol_trend': vol_trend,
                    'pre_compression': compression,
                    'pre_volatility': pre_volatility,
                    'pre_momentum_3h': pre_momentum_3h,
                    'trigger_vol_ratio': trigger_vol_ratio,
                    'last_15m_move': last_15m_move,
                    'avg_price': avg_price,
                })
                last_trigger_idx = i

    triggers_df = pd.DataFrame(triggers)
    print(f"  Total trigger events: {len(triggers_df):,}")
    return triggers_df


def add_spike_history(triggers_df, db):
    """Add historical spike frequency for each symbol (repeat spiker score)."""
    print("Computing spike history per symbol...")

    # Count how many times each symbol appears as a trigger
    sym_counts = triggers_df.groupby('symbol').size().reset_index(name='total_triggers')

    # Count how many resulted in 20%+ moves
    winners = triggers_df[triggers_df['max_fwd_gain'] >= WINNER_THRESHOLD]
    win_counts = winners.groupby('symbol').size().reset_index(name='winner_count')

    sym_stats = sym_counts.merge(win_counts, on='symbol', how='left')
    sym_stats['winner_count'] = sym_stats['winner_count'].fillna(0).astype(int)
    sym_stats['win_rate'] = sym_stats['winner_count'] / sym_stats['total_triggers']

    triggers_df = triggers_df.merge(sym_stats, on='symbol', how='left')
    triggers_df['is_repeat_spiker'] = triggers_df['total_triggers'] >= 10

    return triggers_df


def add_feature_context(triggers_df, db):
    """Pull features at trigger time from research.duckdb."""
    print("Pulling feature context at trigger time...")

    feature_cols = ['rsi_14', 'vpin', 'natr', 'bb_width', 'volume_spike_ratio',
                    'bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance',
                    'roll_measure', 'obv', 'atr', 'vwap_distance_pct']

    feat_data = []
    for idx, row in triggers_df.iterrows():
        ts = row['trigger_time']
        sym = row['symbol']

        result = db.execute("""
            SELECT * FROM features
            WHERE symbol = ? AND timestamp <= ? AND timestamp >= ?
            ORDER BY timestamp DESC LIMIT 1
        """, [sym, ts, ts - timedelta(hours=1)]).df()

        if not result.empty:
            r = result.iloc[0]
            d = {}
            for col in feature_cols:
                d[f'feat_{col}'] = r[col] if col in r.index and pd.notna(r[col]) else None
            feat_data.append(d)
        else:
            feat_data.append({f'feat_{col}': None for col in feature_cols})

        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(triggers_df)}...", flush=True)

    feat_df = pd.DataFrame(feat_data)
    triggers_df = pd.concat([triggers_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    return triggers_df


def analyze(df):
    """The actual analysis."""

    # Label outcomes
    df['outcome'] = 'fizzle'
    df.loc[df['max_fwd_gain'] >= WINNER_THRESHOLD, 'outcome'] = 'winner'
    df.loc[(df['max_fwd_gain'] >= FIZZLE_THRESHOLD) & (df['max_fwd_gain'] < WINNER_THRESHOLD), 'outcome'] = 'moderate'

    winners = df[df['outcome'] == 'winner']
    fizzles = df[df['outcome'] == 'fizzle']
    moderates = df[df['outcome'] == 'moderate']

    print("\n" + "=" * 70)
    print("  FALSE POSITIVE ANALYSIS")
    print("=" * 70)

    print(f"\nTotal triggers (5%+ in 1h): {len(df):,}")
    print(f"  Winners (20%+ forward):   {len(winners):,} ({len(winners)/len(df)*100:.1f}%)")
    print(f"  Moderate (10-20%):         {len(moderates):,} ({len(moderates)/len(df)*100:.1f}%)")
    print(f"  Fizzles (<10%):            {len(fizzles):,} ({len(fizzles)/len(df)*100:.1f}%)")

    # ── REPEAT SPIKERS ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REPEAT SPIKERS vs ONE-TIMERS")
    print("=" * 70)

    repeaters = df[df['is_repeat_spiker'] == True]
    non_rep = df[df['is_repeat_spiker'] == False]

    print(f"\nRepeat spikers (10+ triggers):   {repeaters['symbol'].nunique()} symbols, {len(repeaters):,} triggers")
    print(f"Non-repeat (<10 triggers):        {non_rep['symbol'].nunique()} symbols, {len(non_rep):,} triggers")

    rep_wr = (repeaters['outcome'] == 'winner').mean() * 100
    non_wr = (non_rep['outcome'] == 'winner').mean() * 100
    print(f"\nWin rate (repeat):    {rep_wr:.1f}%")
    print(f"Win rate (non-repeat): {non_wr:.1f}%")

    print(f"\nRepeat spiker forward gains:")
    print(f"  Avg max gain: {repeaters['max_fwd_gain'].mean():.1f}%  median: {repeaters['max_fwd_gain'].median():.1f}%")
    print(f"Non-repeat forward gains:")
    print(f"  Avg max gain: {non_rep['max_fwd_gain'].mean():.1f}%  median: {non_rep['max_fwd_gain'].median():.1f}%")

    # ── WHAT SEPARATES WINNERS FROM FIZZLES ──────────────────────────
    print("\n" + "=" * 70)
    print("  WINNERS vs FIZZLES: FEATURE COMPARISON (medians)")
    print("=" * 70)

    compare_cols = [
        ('trigger_move_pct', 'Initial trigger move (%)'),
        ('last_15m_move', 'Last 15min move (%)'),
        ('pre_vol_trend', 'Volume trend (late/early 6h)'),
        ('trigger_vol_ratio', 'Trigger vol / pre-avg vol'),
        ('pre_compression', 'Price compression'),
        ('pre_volatility', 'Pre-trigger volatility'),
        ('pre_momentum_3h', 'Pre-trigger 3h momentum (%)'),
        ('avg_price', 'Avg coin price ($)'),
        ('total_triggers', 'Historical trigger count'),
        ('winner_count', 'Historical winner count'),
        ('win_rate', 'Historical win rate'),
        ('feat_rsi_14', 'RSI-14'),
        ('feat_vpin', 'VPIN'),
        ('feat_natr', 'NATR'),
        ('feat_bb_width', 'BB width'),
        ('feat_volume_spike_ratio', 'Volume spike ratio'),
        ('feat_bid_ask_spread_pct', 'Bid-ask spread %'),
        ('feat_order_book_depth_ratio', 'OB depth ratio'),
        ('feat_large_order_imbalance', 'Large order imbalance'),
    ]

    print(f"\n{'Feature':40s} {'Winners':>10s} {'Fizzles':>10s} {'Ratio':>8s}  Useful?")
    print("-" * 80)

    useful_features = []
    for col, desc in compare_cols:
        if col not in df.columns:
            continue
        w_med = winners[col].median()
        f_med = fizzles[col].median()
        if pd.notna(w_med) and pd.notna(f_med) and f_med != 0:
            ratio = w_med / f_med
            # Mark as useful if ratio > 1.3 or < 0.7
            useful = "***" if (ratio > 1.3 or ratio < 0.7) else ("*" if (ratio > 1.15 or ratio < 0.85) else "")
            if useful:
                useful_features.append((col, desc, ratio))
            print(f"{desc:40s} {w_med:10.3f} {f_med:10.3f} {ratio:8.2f}x  {useful}")
        elif pd.notna(w_med):
            print(f"{desc:40s} {w_med:10.3f} {'N/A':>10s}")

    # ── TIMING: CAN WE GET IN EARLY ENOUGH? ─────────────────────────
    print("\n" + "=" * 70)
    print("  ENTRY TIMING ANALYSIS")
    print("=" * 70)

    print("\nFor WINNERS, how much of the move is left after trigger?")
    print(f"  Trigger move (entry cost):  {winners['trigger_move_pct'].median():.1f}% (median)")
    print(f"  Max forward gain:           {winners['max_fwd_gain'].median():.1f}% (median)")
    print(f"  Remaining upside:           {(winners['max_fwd_gain'] - winners['trigger_move_pct']).median():.1f}%")
    print(f"  Time to peak:               {winners['time_to_peak_hours'].median():.1f}h (median)")

    print(f"\n  Forward gain distribution for WINNERS:")
    for horizon, col in [('1h', 'gain_1h'), ('3h', 'gain_3h'), ('6h', 'gain_6h'),
                          ('12h', 'gain_12h'), ('24h', 'gain_24h')]:
        med = winners[col].median()
        pct_pos = (winners[col] > 0).mean() * 100
        print(f"    {horizon}: median {med:+.1f}%, positive {pct_pos:.0f}% of the time")

    print(f"\n  Max drawdown before reaching peak (WINNERS):")
    print(f"    Median: {winners['max_dd_before_peak'].median():.1f}%")
    print(f"    p25:    {winners['max_dd_before_peak'].quantile(0.25):.1f}%")
    print(f"    p10:    {winners['max_dd_before_peak'].quantile(0.10):.1f}%")
    print(f"    (This is what your stop loss needs to survive)")

    # ── THE FALSE POSITIVE PROBLEM ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  FALSE POSITIVE DEEP DIVE")
    print("=" * 70)

    # For fizzles: what happens after trigger?
    print("\nFizzle forward trajectory:")
    for horizon, col in [('1h', 'gain_1h'), ('3h', 'gain_3h'), ('6h', 'gain_6h')]:
        med = fizzles[col].median()
        pct_neg = (fizzles[col] < 0).mean() * 100
        print(f"  {horizon}: median {med:+.1f}%, negative {pct_neg:.0f}% of the time")

    print(f"\n  Fizzle max forward gain: {fizzles['max_fwd_gain'].median():.1f}% (median)")
    print(f"  Fizzle max drawdown:     {fizzles['max_dd_before_peak'].median():.1f}%")

    # Key question: can we filter using T+1h behavior?
    print("\n--- T+1h FILTER (like v5.0 triple confirmation) ---")
    for threshold in [0, 1, 2, 3]:
        passes = df[df['gain_1h'] > threshold]
        if len(passes) == 0:
            continue
        wr = (passes['outcome'] == 'winner').mean() * 100
        avg_gain = passes['max_fwd_gain'].mean()
        print(f"  gain_1h > {threshold}%: {len(passes):,} triggers pass, {wr:.1f}% win rate, avg max gain {avg_gain:.1f}%")

    # Combined filter: repeat spiker + T+1h positive
    print("\n--- COMBINED FILTERS ---")
    filters = [
        ("Baseline (all triggers)", df),
        ("Repeat spiker only", df[df['is_repeat_spiker']]),
        ("Repeat + gain_1h > 0", df[(df['is_repeat_spiker']) & (df['gain_1h'] > 0)]),
        ("Repeat + gain_1h > 2%", df[(df['is_repeat_spiker']) & (df['gain_1h'] > 2)]),
        ("Repeat + gain_1h > 0 + vol_trend > 1.5", df[(df['is_repeat_spiker']) & (df['gain_1h'] > 0) & (df['pre_vol_trend'] > 1.5)]),
        ("Repeat + gain_1h > 0 + trigger_vol > 2x", df[(df['is_repeat_spiker']) & (df['gain_1h'] > 0) & (df['trigger_vol_ratio'] > 2)]),
        ("Price < $0.10 + gain_1h > 0", df[(df['avg_price'] < 0.10) & (df['gain_1h'] > 0)]),
        ("Price < $0.10 + repeat + gain_1h > 0", df[(df['avg_price'] < 0.10) & (df['is_repeat_spiker']) & (df['gain_1h'] > 0)]),
    ]

    print(f"\n{'Filter':50s} {'Triggers':>9s} {'WinRate':>8s} {'AvgGain':>8s} {'MedGain':>8s} {'FalsePos':>9s}")
    print("-" * 95)
    for label, subset in filters:
        if len(subset) == 0:
            continue
        wr = (subset['outcome'] == 'winner').mean() * 100
        avg = subset['max_fwd_gain'].mean()
        med = subset['max_fwd_gain'].median()
        fp = (subset['outcome'] == 'fizzle').mean() * 100
        print(f"{label:50s} {len(subset):9,} {wr:7.1f}% {avg:7.1f}% {med:7.1f}% {fp:8.1f}%")

    # ── WHAT MAKES A FIZZLE LOOK DIFFERENT AT TRIGGER TIME ───────────
    print("\n" + "=" * 70)
    print("  STRONGEST DISCRIMINATORS (at trigger time)")
    print("=" * 70)

    if useful_features:
        print("\nFeatures where winners differ from fizzles by >1.3x:")
        for col, desc, ratio in sorted(useful_features, key=lambda x: abs(x[2] - 1), reverse=True):
            direction = "higher" if ratio > 1 else "lower"
            print(f"  {desc:40s}  {ratio:.2f}x ({direction} for winners)")

    # ── TIMING: WHEN WINNERS BECOME IDENTIFIABLE ─────────────────────
    print("\n" + "=" * 70)
    print("  WHEN DO WINNERS BECOME IDENTIFIABLE?")
    print("=" * 70)

    # At each time horizon, what's the best we can separate?
    print("\nIf we wait N minutes after trigger and check if still positive:")
    for wait_bars, label in [(3, '15min'), (6, '30min'), (12, '1h'), (24, '2h'), (36, '3h')]:
        col_approx = None
        if wait_bars <= 12:
            col_approx = 'gain_1h' if wait_bars >= 12 else None

        # We need finer-grained forward data. Use gain_1h as proxy for 1h,
        # and approximate intermediate points.
        # Actually we have gain_1h, gain_3h etc. Let's use what we have.

    # More useful: what % of winners are still positive at 1h vs fizzles
    w_pos_1h = (winners['gain_1h'] > 0).mean() * 100
    f_pos_1h = (fizzles['gain_1h'] > 0).mean() * 100
    w_pos_3h = (winners['gain_3h'] > 0).mean() * 100
    f_pos_3h = (fizzles['gain_3h'] > 0).mean() * 100

    print(f"  At +1h: {w_pos_1h:.0f}% of winners still positive vs {f_pos_1h:.0f}% of fizzles")
    print(f"  At +3h: {w_pos_3h:.0f}% of winners still positive vs {f_pos_3h:.0f}% of fizzles")

    print(f"\n  Winners 1h gain:  median {winners['gain_1h'].median():+.1f}%")
    print(f"  Fizzles 1h gain:  median {fizzles['gain_1h'].median():+.1f}%")
    print(f"  Winners 3h gain:  median {winners['gain_3h'].median():+.1f}%")
    print(f"  Fizzles 3h gain:  median {fizzles['gain_3h'].median():+.1f}%")

    # Save enriched data
    output_path = "data/trigger_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved enriched trigger data to {output_path}")

    return df


def main():
    db = duckdb.connect(DB_PATH, read_only=True)

    # 1. Find all trigger events
    triggers = find_trigger_events(db)

    # 2. Add spike history
    triggers = add_spike_history(triggers, db)

    # 3. Add feature context
    triggers = add_feature_context(triggers, db)

    db.close()

    # 4. Analyze
    analyze(triggers)


if __name__ == "__main__":
    main()
