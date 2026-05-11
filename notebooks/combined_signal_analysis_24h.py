"""
Combined Signal Analysis: Velocity Spikes + 24h Price Moves

Goal: Find conditions under which velocity spikes predict 20%+ moves in 24 hours.
Uses larger time windows and proper signal deduplication.
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_text
import warnings
warnings.filterwarnings('ignore')

DB_PATH = 'data/pythia_snapshot.duckdb'
SPIKES_PATH = 'data/velocity_spike_analysis.csv'

# Configuration
VELOCITY_THRESHOLD = 5.0  # Minimum velocity ratio
PRICE_MOVE_THRESHOLD = 0.20  # 20% move threshold
FORWARD_HOURS = 24  # Look 24 hours ahead
DEDUP_WINDOW_MINUTES = 60  # Group spikes within 60 min as same signal
FEATURE_WINDOW_MINUTES = 15  # Join features within ±15 min

FEATURES_TO_ANALYZE = [
    'vpin',
    'large_order_imbalance',
    'bid_ask_spread_pct',
    'volume_spike_ratio',
]


def load_velocity_spikes(min_velocity: float = VELOCITY_THRESHOLD) -> pd.DataFrame:
    """Load and deduplicate velocity spikes."""
    spikes = pd.read_csv(SPIKES_PATH, parse_dates=['timestamp'])
    spikes = spikes[spikes['velocity_ratio'] >= min_velocity].copy()

    print(f"Raw spikes >= {min_velocity}x: {len(spikes):,}")

    # Deduplicate: for each symbol, group spikes within DEDUP_WINDOW
    # Keep only the first (or strongest) spike in each window
    spikes = spikes.sort_values(['symbol', 'timestamp'])

    deduped = []
    for symbol, group in spikes.groupby('symbol'):
        group = group.sort_values('timestamp')

        last_ts = None
        for _, row in group.iterrows():
            if last_ts is None or (row['timestamp'] - last_ts).total_seconds() > DEDUP_WINDOW_MINUTES * 60:
                deduped.append(row)
                last_ts = row['timestamp']
            # Skip spikes too close to previous one

    spikes = pd.DataFrame(deduped)

    print(f"After {DEDUP_WINDOW_MINUTES}min dedup: {len(spikes):,} unique signals")
    print(f"  Date range: {spikes['timestamp'].min()} to {spikes['timestamp'].max()}")
    print(f"  Symbols: {spikes['symbol'].nunique()}")

    return spikes


def compute_24h_price_changes(spikes: pd.DataFrame) -> pd.DataFrame:
    """Compute 24h forward price change for each spike from OHLCV data."""
    conn = duckdb.connect(DB_PATH, read_only=True)

    print(f"\nComputing {FORWARD_HOURS}h forward price changes...")

    results = []
    symbols = spikes['symbol'].unique()

    for symbol in symbols:
        # Load OHLCV for this symbol
        ohlcv = conn.execute(f'''
            SELECT timestamp, close
            FROM ohlcv
            WHERE symbol = '{symbol}' AND timeframe = '1m'
            ORDER BY timestamp
        ''').df()

        if len(ohlcv) == 0:
            continue

        ohlcv = ohlcv.set_index('timestamp')

        symbol_spikes = spikes[spikes['symbol'] == symbol]

        for _, spike in symbol_spikes.iterrows():
            ts = spike['timestamp']

            # Get price at spike time (or closest)
            try:
                # Find closest price within 5 min
                mask = (ohlcv.index >= ts - pd.Timedelta(minutes=5)) & \
                       (ohlcv.index <= ts + pd.Timedelta(minutes=5))
                nearby = ohlcv[mask]
                if len(nearby) == 0:
                    continue
                price_at_spike = nearby['close'].iloc[0]

                # Get price 24h later
                future_ts = ts + pd.Timedelta(hours=FORWARD_HOURS)
                mask_future = (ohlcv.index >= future_ts - pd.Timedelta(minutes=5)) & \
                              (ohlcv.index <= future_ts + pd.Timedelta(minutes=5))
                future = ohlcv[mask_future]
                if len(future) == 0:
                    continue
                price_future = future['close'].iloc[0]

                # Also get max/min in the 24h window for max move
                window_mask = (ohlcv.index >= ts) & (ohlcv.index <= future_ts)
                window = ohlcv[window_mask]
                if len(window) == 0:
                    continue

                max_price = window['close'].max()
                min_price = window['close'].min()

                # Calculate changes
                price_change_24h = (price_future - price_at_spike) / price_at_spike
                max_up = (max_price - price_at_spike) / price_at_spike
                max_down = (min_price - price_at_spike) / price_at_spike
                max_move = max(abs(max_up), abs(max_down))

                row = spike.to_dict()
                row['price_at_spike'] = price_at_spike
                row['price_24h'] = price_future
                row['price_change_24h'] = price_change_24h
                row['max_up_24h'] = max_up
                row['max_down_24h'] = max_down
                row['max_move_24h'] = max_move
                results.append(row)

            except Exception as e:
                continue

    conn.close()

    df = pd.DataFrame(results)
    print(f"  Computed price changes for {len(df):,} spikes")

    return df


def load_features() -> pd.DataFrame:
    """Load features from DuckDB (1m timeframe)."""
    conn = duckdb.connect(DB_PATH, read_only=True)

    available_features = ['vpin', 'large_order_imbalance', 'bid_ask_spread_pct', 'volume_spike_ratio']
    feature_cols = ', '.join(available_features)

    features = conn.execute(f'''
        SELECT timestamp, symbol, {feature_cols}
        FROM features
        WHERE timeframe = '1m'
    ''').df()
    conn.close()

    print(f"Loaded {len(features):,} feature rows")
    return features


def join_spikes_with_features(spikes: pd.DataFrame, features: pd.DataFrame,
                               window_minutes: int = FEATURE_WINDOW_MINUTES) -> pd.DataFrame:
    """Join spikes with nearest features using time window."""
    results = []
    features_by_symbol = {sym: grp for sym, grp in features.groupby('symbol')}

    print(f"\nJoining with ±{window_minutes} minute window...")

    matched = 0
    unmatched = 0

    for _, spike in spikes.iterrows():
        sym = spike['symbol']
        ts = spike['timestamp']

        if sym not in features_by_symbol:
            unmatched += 1
            continue

        sym_features = features_by_symbol[sym]

        window_start = ts - pd.Timedelta(minutes=window_minutes)
        window_end = ts + pd.Timedelta(minutes=window_minutes)

        nearby = sym_features[
            (sym_features['timestamp'] >= window_start) &
            (sym_features['timestamp'] <= window_end)
        ]

        if len(nearby) == 0:
            unmatched += 1
            continue

        nearby = nearby.copy()
        nearby['time_diff'] = abs((nearby['timestamp'] - ts).dt.total_seconds())
        closest = nearby.loc[nearby['time_diff'].idxmin()]

        row = spike.to_dict()
        for col in FEATURES_TO_ANALYZE:
            if col in closest.index:
                row[col] = closest[col]
        row['feature_timestamp'] = closest['timestamp']
        row['time_diff_seconds'] = closest['time_diff']

        results.append(row)
        matched += 1

    print(f"  Matched: {matched}, Unmatched: {unmatched}")

    if len(results) == 0:
        return pd.DataFrame()

    merged = pd.DataFrame(results)

    # Label TRUE spikes
    merged['is_true_spike'] = merged['max_move_24h'] >= PRICE_MOVE_THRESHOLD
    merged['move_direction'] = np.sign(merged['price_change_24h'])

    print(f"\nJoined data: {len(merged):,} events")
    print(f"  TRUE spikes (>={PRICE_MOVE_THRESHOLD*100:.0f}% max move): {merged['is_true_spike'].sum()}")
    print(f"  FALSE spikes: {(~merged['is_true_spike']).sum()}")
    if len(merged) > 0:
        print(f"  Precision baseline: {merged['is_true_spike'].mean()*100:.2f}%")

    return merged


def compare_distributions(merged: pd.DataFrame):
    """Compare feature distributions between TRUE and FALSE spikes."""
    print("\n" + "=" * 80)
    print("FEATURE DISTRIBUTION COMPARISON: TRUE vs FALSE SPIKES")
    print("=" * 80)

    true_spikes = merged[merged['is_true_spike']]
    false_spikes = merged[~merged['is_true_spike']]

    results = []

    print(f"\n{'Feature':<28} {'TRUE Mean':>12} {'FALSE Mean':>12} {'Ratio':>10} {'P-value':>12} {'Sig':>6}")
    print("-" * 82)

    for col in FEATURES_TO_ANALYZE:
        true_vals = true_spikes[col].dropna()
        false_vals = false_spikes[col].dropna()

        if len(true_vals) == 0 or len(false_vals) == 0:
            continue

        true_mean = true_vals.mean()
        false_mean = false_vals.mean()

        ratio = true_mean / false_mean if false_mean != 0 else float('inf')

        try:
            stat, p_value = stats.mannwhitneyu(true_vals, false_vals, alternative='two-sided')
        except:
            p_value = 1.0

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

        results.append({
            'feature': col,
            'true_mean': true_mean,
            'false_mean': false_mean,
            'ratio': ratio,
            'p_value': p_value,
        })

        print(f"{col:<28} {true_mean:>12.4f} {false_mean:>12.4f} {ratio:>10.2f}x {p_value:>12.4e} {sig:>6}")

    return pd.DataFrame(results)


def analyze_extreme_moves(merged: pd.DataFrame):
    """Show the biggest movers."""
    print("\n" + "=" * 80)
    print(f"TOP MOVERS: Spikes followed by largest {FORWARD_HOURS}h moves")
    print("=" * 80)

    top = merged.nlargest(20, 'max_move_24h')

    print(f"\n{'Symbol':<14} {'Timestamp':<20} {'Vel':>8} {'24h Move':>10} {'Max Move':>10} {'VPIN':>8} {'Spread':>10}")
    print("-" * 92)

    for _, row in top.iterrows():
        ts = str(row['timestamp'])[:16]
        print(f"{row['symbol']:<14} {ts:<20} {row['velocity_ratio']:>8.1f}x "
              f"{row['price_change_24h']:>+10.1%} {row['max_move_24h']:>10.1%} "
              f"{row.get('vpin', 0):>8.3f} {row.get('bid_ask_spread_pct', 0)*100:>10.3f}%")


def grid_search_thresholds(merged: pd.DataFrame):
    """Find best single-feature thresholds."""
    print("\n" + "=" * 80)
    print("GRID SEARCH: Single feature thresholds")
    print("=" * 80)

    results = []
    baseline = merged['is_true_spike'].mean()

    for feature in FEATURES_TO_ANALYZE:
        if feature not in merged.columns:
            continue

        vals = merged[feature].dropna()
        if len(vals) == 0:
            continue

        for pct in [70, 80, 90, 95]:
            # High threshold
            threshold = vals.quantile(pct / 100)
            mask = merged[feature] > threshold
            if mask.sum() > 0:
                precision = merged[mask]['is_true_spike'].mean()
                recall = merged[mask]['is_true_spike'].sum() / merged['is_true_spike'].sum() if merged['is_true_spike'].sum() > 0 else 0
                results.append({
                    'rule': f'{feature} > {threshold:.4f} (p{pct})',
                    'signals': mask.sum(),
                    'precision': precision,
                    'recall': recall,
                    'improvement': precision / baseline if baseline > 0 else 0
                })

            # Low threshold
            low_threshold = vals.quantile((100 - pct) / 100)
            mask = merged[feature] < low_threshold
            if mask.sum() > 0:
                precision = merged[mask]['is_true_spike'].mean()
                recall = merged[mask]['is_true_spike'].sum() / merged['is_true_spike'].sum() if merged['is_true_spike'].sum() > 0 else 0
                results.append({
                    'rule': f'{feature} < {low_threshold:.4f} (p{100-pct})',
                    'signals': mask.sum(),
                    'precision': precision,
                    'recall': recall,
                    'improvement': precision / baseline if baseline > 0 else 0
                })

    results_df = pd.DataFrame(results).sort_values('improvement', ascending=False)

    print(f"\nTop rules by improvement (baseline: {baseline*100:.2f}%):")
    print(f"{'Rule':<45} {'Signals':>8} {'Prec':>10} {'Recall':>10} {'Improv':>10}")
    print("-" * 85)

    for _, row in results_df.head(15).iterrows():
        print(f"{row['rule']:<45} {row['signals']:>8} {row['precision']:>10.2%} "
              f"{row['recall']:>10.2%} {row['improvement']:>10.1f}x")

    return results_df


def test_combined_rules(merged: pd.DataFrame):
    """Test combined rule sets."""
    print("\n" + "=" * 80)
    print("COMBINED RULES TESTING")
    print("=" * 80)

    baseline = merged['is_true_spike'].mean()
    total_true = merged['is_true_spike'].sum()

    rule_sets = [
        ('Velocity >= 10x', {'velocity_ratio': ('>=', 10.0)}),
        ('Velocity >= 15x', {'velocity_ratio': ('>=', 15.0)}),
        ('Wide spread (>0.2%)', {'bid_ask_spread_pct': ('>', 0.002)}),
        ('Wide spread (>0.3%)', {'bid_ask_spread_pct': ('>', 0.003)}),
        ('Low VPIN (<0.15)', {'vpin': ('<', 0.15)}),
        ('Low VPIN (<0.20)', {'vpin': ('<', 0.20)}),
        ('Vel>=10 + spread>0.2%', {'velocity_ratio': ('>=', 10.0), 'bid_ask_spread_pct': ('>', 0.002)}),
        ('Vel>=10 + low VPIN', {'velocity_ratio': ('>=', 10.0), 'vpin': ('<', 0.20)}),
        ('Vel>=7 + spread>0.3%', {'velocity_ratio': ('>=', 7.0), 'bid_ask_spread_pct': ('>', 0.003)}),
        ('Spread>0.2% + low VPIN', {'bid_ask_spread_pct': ('>', 0.002), 'vpin': ('<', 0.25)}),
    ]

    results = []

    for name, rules in rule_sets:
        mask = pd.Series([True] * len(merged), index=merged.index)

        for feature, (op, threshold) in rules.items():
            if feature not in merged.columns:
                continue
            if op == '>=':
                mask = mask & (merged[feature] >= threshold)
            elif op == '>':
                mask = mask & (merged[feature] > threshold)
            elif op == '<':
                mask = mask & (merged[feature] < threshold)
            elif op == '<=':
                mask = mask & (merged[feature] <= threshold)

        signals = mask.sum()
        if signals == 0:
            continue

        true_positives = merged[mask]['is_true_spike'].sum()
        precision = true_positives / signals
        recall = true_positives / total_true if total_true > 0 else 0
        improvement = precision / baseline if baseline > 0 else 0

        results.append({
            'name': name,
            'signals': signals,
            'true_pos': true_positives,
            'precision': precision,
            'recall': recall,
            'improvement': improvement
        })

    results_df = pd.DataFrame(results).sort_values('precision', ascending=False)

    print(f"\n{'Rule':<30} {'Signals':>10} {'TP':>6} {'Precision':>12} {'Recall':>10} {'Improv':>10}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(f"{row['name']:<30} {row['signals']:>10} {row['true_pos']:>6} "
              f"{row['precision']:>12.2%} {row['recall']:>10.2%} {row['improvement']:>10.1f}x")

    return results_df


def build_decision_tree(merged: pd.DataFrame, max_depth: int = 3):
    """Build decision tree to find thresholds."""
    print("\n" + "=" * 80)
    print("DECISION TREE ANALYSIS")
    print("=" * 80)

    feature_cols = [c for c in FEATURES_TO_ANALYZE if c in merged.columns]
    X = merged[feature_cols].copy()
    y = merged['is_true_spike'].astype(int)

    mask = ~X.isna().any(axis=1)
    X = X[mask]
    y = y[mask]

    print(f"\nTraining: {len(X):,} samples, {y.sum()} positive ({y.mean()*100:.2f}%)")

    if len(X) == 0 or y.sum() < 2:
        print("Insufficient data for tree")
        return None, feature_cols

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(2, int(y.sum() * 0.1)),
        class_weight='balanced'
    )
    tree.fit(X, y)

    print(f"\nTree (depth={max_depth}):")
    print("-" * 50)
    print(export_text(tree, feature_names=feature_cols))

    print("\nFeature Importance:")
    for feat, imp in sorted(zip(feature_cols, tree.feature_importances_), key=lambda x: -x[1]):
        if imp > 0:
            bar = '#' * int(imp * 30)
            print(f"  {feat:<25} {imp:.3f} {bar}")

    return tree, feature_cols


def main():
    print("=" * 80)
    print(f"COMBINED SIGNAL ANALYSIS: {PRICE_MOVE_THRESHOLD*100:.0f}% moves in {FORWARD_HOURS}h")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Velocity threshold: >= {VELOCITY_THRESHOLD}x")
    print(f"  Price move threshold: >= {PRICE_MOVE_THRESHOLD*100:.0f}% max move in {FORWARD_HOURS}h")
    print(f"  Dedup window: {DEDUP_WINDOW_MINUTES} minutes")
    print(f"  Feature join window: ±{FEATURE_WINDOW_MINUTES} minutes")

    # Step 1: Load and deduplicate spikes
    print("\n" + "=" * 80)
    print("STEP 1: Load and Deduplicate Velocity Spikes")
    print("=" * 80)
    spikes = load_velocity_spikes()

    # Step 2: Compute 24h price changes
    print("\n" + "=" * 80)
    print("STEP 2: Compute 24h Forward Price Changes")
    print("=" * 80)
    spikes = compute_24h_price_changes(spikes)

    if len(spikes) == 0:
        print("No price data available")
        return None, None, None

    # Quick stats on moves
    print(f"\n24h move statistics:")
    print(f"  Max move mean: {spikes['max_move_24h'].mean()*100:.1f}%")
    print(f"  Max move median: {spikes['max_move_24h'].median()*100:.1f}%")
    print(f"  Spikes with >=20% move: {(spikes['max_move_24h'] >= 0.20).sum()}")
    print(f"  Spikes with >=15% move: {(spikes['max_move_24h'] >= 0.15).sum()}")
    print(f"  Spikes with >=10% move: {(spikes['max_move_24h'] >= 0.10).sum()}")

    # Step 3: Load features
    print("\n" + "=" * 80)
    print("STEP 3: Load Features")
    print("=" * 80)
    features = load_features()

    # Step 4: Join
    print("\n" + "=" * 80)
    print("STEP 4: Join Spikes with Features")
    print("=" * 80)
    merged = join_spikes_with_features(spikes, features)

    if len(merged) == 0:
        print("No data after join")
        return None, None, None

    # Step 5: Compare distributions
    dist_results = compare_distributions(merged)

    # Step 6: Show biggest movers
    analyze_extreme_moves(merged)

    # Step 7: Grid search
    grid_results = grid_search_thresholds(merged)

    # Step 8: Test combined rules
    combined_results = test_combined_rules(merged)

    # Step 9: Decision tree
    tree, feature_cols = build_decision_tree(merged, max_depth=4)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    baseline = merged['is_true_spike'].mean()
    total_true = merged['is_true_spike'].sum()

    print(f"""
Dataset:
  - {len(merged):,} velocity spikes (deduplicated, {DEDUP_WINDOW_MINUTES}min window)
  - {total_true} TRUE spikes (>={PRICE_MOVE_THRESHOLD*100:.0f}% max move in {FORWARD_HOURS}h)
  - Baseline precision: {baseline*100:.2f}%

Best Rules (by precision):
""")

    if len(combined_results) > 0:
        top3 = combined_results.head(3)
        for _, row in top3.iterrows():
            print(f"  {row['name']}: {row['precision']*100:.1f}% precision, "
                  f"{row['signals']} signals, {row['improvement']:.1f}x improvement")

    # Save
    merged.to_csv('data/spikes_24h_analysis.csv', index=False)
    print(f"\nSaved to data/spikes_24h_analysis.csv")

    return merged, dist_results, tree


if __name__ == '__main__':
    merged, dist_results, tree = main()
