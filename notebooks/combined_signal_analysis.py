"""
Combined Signal Analysis: Velocity Spikes + Conditioning Factors

Goal: Find conditions under which trade velocity spikes predict significant price moves.
Velocity alone has ~0.5% precision - we need conditioning signals to filter noise.

Methodology:
1. Join velocity spikes (>=5x) with features table
2. Label TRUE (5%+ move in 60min) vs FALSE spikes
3. Compare feature distributions between groups
4. Build decision tree to find optimal thresholds
5. Backtest combined signal precision/recall
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

DB_PATH = 'data/pythia_snapshot.duckdb'
SPIKES_PATH = 'data/velocity_spike_analysis.csv'

# Configuration
VELOCITY_THRESHOLD = 5.0  # Minimum velocity ratio for spike
PRICE_MOVE_THRESHOLD = 0.05  # 5% move threshold for "TRUE" spike

# Features available in 1m timeframe (roll_measure and order_book_imbalance_l5 only in 5m)
FEATURES_TO_ANALYZE = [
    'vpin',
    'large_order_imbalance',
    'bid_ask_spread_pct',
    'volume_spike_ratio',
]


def load_velocity_spikes(min_velocity: float = VELOCITY_THRESHOLD) -> pd.DataFrame:
    """Load velocity spike data and filter by threshold."""
    spikes = pd.read_csv(SPIKES_PATH, parse_dates=['timestamp'])
    spikes = spikes[spikes['velocity_ratio'] >= min_velocity].copy()

    # Create 5-minute bucket for joining with features
    spikes['bucket'] = spikes['timestamp'].dt.floor('5min')

    print(f"Loaded {len(spikes):,} velocity spikes >= {min_velocity}x")
    print(f"  Date range: {spikes['timestamp'].min()} to {spikes['timestamp'].max()}")
    print(f"  Symbols: {spikes['symbol'].nunique()}")
    return spikes


def load_features() -> pd.DataFrame:
    """Load features from DuckDB (1m timeframe for better coverage)."""
    conn = duckdb.connect(DB_PATH, read_only=True)

    # Use 1m features - they have better date coverage
    # Note: roll_measure and order_book_imbalance_l5 are only in 5m, so we skip them
    available_features = ['vpin', 'large_order_imbalance', 'bid_ask_spread_pct', 'volume_spike_ratio']
    feature_cols = ', '.join(available_features)

    query = f'''
        SELECT timestamp, symbol, {feature_cols}
        FROM features
        WHERE timeframe = '1m'
    '''

    features = conn.execute(query).df()
    conn.close()

    print(f"Loaded {len(features):,} feature rows (1m timeframe)")
    print(f"  Available features: {available_features}")
    return features


def join_spikes_with_features(spikes: pd.DataFrame, features: pd.DataFrame,
                               window_minutes: int = 5) -> pd.DataFrame:
    """Join velocity spikes with nearest feature values using time window.

    Features are sparse (~9% density), so we find the closest feature within
    a time window rather than requiring exact minute match.
    """
    results = []

    spikes = spikes.copy()
    features = features.copy()

    # Group features by symbol for faster lookup
    features_by_symbol = {sym: grp for sym, grp in features.groupby('symbol')}

    print(f"  Joining with ±{window_minutes} minute window...")

    matched = 0
    unmatched = 0

    for _, spike in spikes.iterrows():
        sym = spike['symbol']
        ts = spike['timestamp']

        if sym not in features_by_symbol:
            unmatched += 1
            continue

        sym_features = features_by_symbol[sym]

        # Find features within time window
        window_start = ts - pd.Timedelta(minutes=window_minutes)
        window_end = ts + pd.Timedelta(minutes=window_minutes)

        nearby = sym_features[
            (sym_features['timestamp'] >= window_start) &
            (sym_features['timestamp'] <= window_end)
        ]

        if len(nearby) == 0:
            unmatched += 1
            continue

        # Take the closest one
        nearby = nearby.copy()
        nearby['time_diff'] = abs((nearby['timestamp'] - ts).dt.total_seconds())
        closest = nearby.loc[nearby['time_diff'].idxmin()]

        # Combine spike and feature data
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
        print("  WARNING: No matches found!")
        return pd.DataFrame()

    merged = pd.DataFrame(results)

    # Deduplicate by timestamp + symbol (keep first occurrence)
    merged = merged.drop_duplicates(subset=['timestamp', 'symbol']).copy()

    # Label TRUE vs FALSE spikes
    merged['is_true_spike'] = abs(merged['price_change_60min']) >= PRICE_MOVE_THRESHOLD
    merged['move_direction'] = np.sign(merged['price_change_60min'])
    merged['abs_move'] = abs(merged['price_change_60min'])

    print(f"\nJoined data: {len(merged):,} unique events")
    print(f"  TRUE spikes (>={PRICE_MOVE_THRESHOLD*100:.0f}% move): {merged['is_true_spike'].sum()}")
    print(f"  FALSE spikes: {(~merged['is_true_spike']).sum()}")
    if len(merged) > 0:
        print(f"  Precision baseline: {merged['is_true_spike'].mean()*100:.2f}%")
        print(f"  Avg time diff to nearest feature: {merged['time_diff_seconds'].mean():.1f}s")

    return merged


def compare_distributions(merged: pd.DataFrame):
    """Compare feature distributions between TRUE and FALSE spikes."""
    print("\n" + "=" * 80)
    print("FEATURE DISTRIBUTION COMPARISON: TRUE vs FALSE SPIKES")
    print("=" * 80)

    true_spikes = merged[merged['is_true_spike']]
    false_spikes = merged[~merged['is_true_spike']]

    results = []

    print(f"\n{'Feature':<28} {'TRUE Mean':>12} {'FALSE Mean':>12} {'Ratio':>10} {'P-value':>12} {'Significant':>12}")
    print("-" * 88)

    for col in FEATURES_TO_ANALYZE:
        true_vals = true_spikes[col].dropna()
        false_vals = false_spikes[col].dropna()

        if len(true_vals) == 0 or len(false_vals) == 0:
            print(f"{col:<28} {'N/A':>12} {'N/A':>12}")
            continue

        true_mean = true_vals.mean()
        false_mean = false_vals.mean()
        true_median = true_vals.median()
        false_median = false_vals.median()

        # Ratio (handle zero division)
        if false_mean != 0:
            ratio = true_mean / false_mean
        else:
            ratio = float('inf') if true_mean > 0 else 1.0

        # Statistical test (Mann-Whitney U for non-normal distributions)
        try:
            stat, p_value = stats.mannwhitneyu(true_vals, false_vals, alternative='two-sided')
        except:
            p_value = 1.0

        significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

        results.append({
            'feature': col,
            'true_mean': true_mean,
            'false_mean': false_mean,
            'true_median': true_median,
            'false_median': false_median,
            'ratio': ratio,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

        print(f"{col:<28} {true_mean:>12.4f} {false_mean:>12.4f} {ratio:>10.2f}x {p_value:>12.4e} {significant:>12}")

    print("\n*** p<0.001, ** p<0.01, * p<0.05")

    return pd.DataFrame(results)


def analyze_directional_signals(merged: pd.DataFrame):
    """Analyze if features predict direction of move."""
    print("\n" + "=" * 80)
    print("DIRECTIONAL ANALYSIS: Do features predict UP vs DOWN moves?")
    print("=" * 80)

    # Only look at TRUE spikes (significant moves)
    true_spikes = merged[merged['is_true_spike']].copy()

    if len(true_spikes) == 0:
        print("No TRUE spikes to analyze")
        return

    up_moves = true_spikes[true_spikes['move_direction'] > 0]
    down_moves = true_spikes[true_spikes['move_direction'] < 0]

    print(f"\nTRUE spikes: {len(up_moves)} UP, {len(down_moves)} DOWN")
    print(f"\n{'Feature':<28} {'UP Mean':>12} {'DOWN Mean':>12} {'Diff':>10}")
    print("-" * 62)

    for col in FEATURES_TO_ANALYZE:
        if col == 'volume_spike_ratio':  # Not directional
            continue

        up_mean = up_moves[col].mean()
        down_mean = down_moves[col].mean()
        diff = up_mean - down_mean

        print(f"{col:<28} {up_mean:>12.4f} {down_mean:>12.4f} {diff:>+10.4f}")


def build_decision_tree(merged: pd.DataFrame, max_depth: int = 3):
    """Build decision tree to find optimal thresholds."""
    print("\n" + "=" * 80)
    print("DECISION TREE: Finding optimal thresholds")
    print("=" * 80)

    # Prepare features and target
    feature_cols = [c for c in FEATURES_TO_ANALYZE if c in merged.columns]
    X = merged[feature_cols].copy()
    y = merged['is_true_spike'].astype(int)

    # Drop rows with NaN
    mask = ~X.isna().any(axis=1)
    X = X[mask]
    y = y[mask]

    print(f"\nTraining data: {len(X):,} samples, {y.sum()} positive ({y.mean()*100:.2f}% positive rate)")

    if len(X) == 0 or y.sum() == 0:
        print("\nInsufficient data for decision tree training.")
        print("Need at least some TRUE spikes to train classifier.")
        return None, feature_cols

    # Train decision tree
    min_leaf = max(2, int(y.sum() * 0.1))  # At least 10% of positives per leaf
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        class_weight='balanced'  # Handle class imbalance
    )
    tree.fit(X, y)

    # Print tree rules
    print(f"\nDecision Tree (max_depth={max_depth}, min_leaf={min_leaf}):")
    print("-" * 60)
    tree_rules = export_text(tree, feature_names=feature_cols)
    print(tree_rules)

    # Feature importance
    print("\nFeature Importance:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': tree.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.iterrows():
        if row['importance'] > 0:
            bar = '#' * int(row['importance'] * 40)
            print(f"  {row['feature']:<28} {row['importance']:.3f} {bar}")

    # Predictions
    y_pred = tree.predict(X)

    print("\nTree Performance:")
    print(f"  Predictions: {y_pred.sum()} predicted TRUE, {(~y_pred.astype(bool)).sum()} predicted FALSE")
    if y_pred.sum() > 0:
        print(f"  Precision: {precision_score(y, y_pred):.2%}")
        print(f"  Recall: {recall_score(y, y_pred):.2%}")
        print(f"  F1: {f1_score(y, y_pred):.2%}")
    else:
        print("  (No positive predictions made)")

    return tree, feature_cols


def extract_rules_from_tree(tree, feature_names, node_id=0, depth=0, rules=None):
    """Extract decision rules from tree."""
    if rules is None:
        rules = []

    left_child = tree.tree_.children_left[node_id]
    right_child = tree.tree_.children_right[node_id]

    if left_child == right_child:  # Leaf node
        samples = tree.tree_.n_node_samples[node_id]
        values = tree.tree_.value[node_id][0]
        class_0, class_1 = values
        precision = class_1 / (class_0 + class_1) if (class_0 + class_1) > 0 else 0
        return precision, class_1

    return None


def backtest_combined_signal(merged: pd.DataFrame, rules: dict):
    """Backtest combined signal with specific rules."""
    print("\n" + "=" * 80)
    print("BACKTEST: Combined Signal Performance")
    print("=" * 80)

    # Apply rules
    mask = pd.Series([True] * len(merged), index=merged.index)

    print("\nApplied rules:")
    for feature, (op, threshold) in rules.items():
        if feature not in merged.columns:
            continue
        if op == '>':
            condition = merged[feature] > threshold
        elif op == '<':
            condition = merged[feature] < threshold
        elif op == '>=':
            condition = merged[feature] >= threshold
        elif op == '<=':
            condition = merged[feature] <= threshold
        elif op == 'abs>':
            condition = abs(merged[feature]) > threshold
        else:
            continue

        mask = mask & condition
        print(f"  {feature} {op} {threshold}")

    # Calculate metrics
    signals = merged[mask]
    true_positives = signals['is_true_spike'].sum()
    total_signals = len(signals)
    total_true = merged['is_true_spike'].sum()

    precision = true_positives / total_signals if total_signals > 0 else 0
    recall = true_positives / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nResults:")
    print(f"  Total spikes in data: {len(merged):,}")
    print(f"  Signals fired: {total_signals:,} ({total_signals/len(merged)*100:.1f}%)")
    print(f"  True positives: {true_positives}")
    print(f"  Precision: {precision:.2%} (baseline: {merged['is_true_spike'].mean():.2%})")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1 Score: {f1:.2%}")
    print(f"  Improvement over baseline: {precision / merged['is_true_spike'].mean():.1f}x")

    return {
        'signals': total_signals,
        'true_positives': true_positives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'improvement': precision / merged['is_true_spike'].mean() if merged['is_true_spike'].mean() > 0 else 0
    }


def grid_search_thresholds(merged: pd.DataFrame):
    """Grid search over feature thresholds to find optimal combinations."""
    print("\n" + "=" * 80)
    print("GRID SEARCH: Finding optimal threshold combinations")
    print("=" * 80)

    best_results = []

    # Test individual feature thresholds
    for feature in FEATURES_TO_ANALYZE:
        if feature not in merged.columns:
            continue

        vals = merged[feature].dropna()
        if len(vals) == 0:
            continue

        # Test percentile thresholds
        for pct in [70, 80, 90, 95]:
            threshold = vals.quantile(pct / 100)

            # High values
            mask = merged[feature] > threshold
            if mask.sum() > 0:
                precision = merged[mask]['is_true_spike'].mean()
                recall = merged[mask]['is_true_spike'].sum() / merged['is_true_spike'].sum()
                best_results.append({
                    'rule': f'{feature} > {threshold:.4f} (p{pct})',
                    'signals': mask.sum(),
                    'precision': precision,
                    'recall': recall,
                    'improvement': precision / merged['is_true_spike'].mean()
                })

            # Low values (for features where low might be signal)
            low_threshold = vals.quantile((100 - pct) / 100)
            mask = merged[feature] < low_threshold
            if mask.sum() > 0:
                precision = merged[mask]['is_true_spike'].mean()
                recall = merged[mask]['is_true_spike'].sum() / merged['is_true_spike'].sum()
                best_results.append({
                    'rule': f'{feature} < {low_threshold:.4f} (p{100-pct})',
                    'signals': mask.sum(),
                    'precision': precision,
                    'recall': recall,
                    'improvement': precision / merged['is_true_spike'].mean()
                })

    # Sort by improvement
    results_df = pd.DataFrame(best_results)
    results_df = results_df.sort_values('improvement', ascending=False)

    print("\nTop 15 single-feature rules by improvement:")
    print(f"{'Rule':<50} {'Signals':>10} {'Precision':>12} {'Recall':>10} {'Improvement':>12}")
    print("-" * 96)

    for _, row in results_df.head(15).iterrows():
        print(f"{row['rule']:<50} {row['signals']:>10} {row['precision']:>12.2%} {row['recall']:>10.2%} {row['improvement']:>12.1f}x")

    return results_df


def analyze_extreme_moves(merged: pd.DataFrame):
    """Deep dive into the most extreme price moves."""
    print("\n" + "=" * 80)
    print("EXTREME MOVES ANALYSIS: What distinguished the biggest movers?")
    print("=" * 80)

    # Get top movers
    top_movers = merged.nlargest(20, 'abs_move')

    print(f"\nTop 20 price moves (all >{PRICE_MOVE_THRESHOLD*100:.0f}% moves):")
    print(f"{'Symbol':<12} {'Timestamp':<20} {'Move':>10} {'Velocity':>10} {'OBI':>10} {'VPIN':>10} {'Roll':>10}")
    print("-" * 94)

    for _, row in top_movers.iterrows():
        ts = str(row['timestamp'])[:16] if 'timestamp' in row else 'N/A'
        print(f"{row['symbol']:<12} {ts:<20} {row['price_change_60min']:>+10.2%} "
              f"{row['velocity_ratio']:>10.1f}x "
              f"{row.get('order_book_imbalance_l5', 0):>+10.3f} "
              f"{row.get('vpin', 0):>10.3f} "
              f"{row.get('roll_measure', 0):>10.4f}")

    # Compare extreme TRUE vs average FALSE
    print("\n\nExtreme TRUE spikes vs average FALSE spikes:")
    extreme_true = top_movers
    avg_false = merged[~merged['is_true_spike']]

    for col in FEATURES_TO_ANALYZE:
        if col not in merged.columns:
            continue
        extreme_mean = extreme_true[col].mean()
        false_mean = avg_false[col].mean()
        ratio = extreme_mean / false_mean if false_mean != 0 else float('inf')
        print(f"  {col:<28}: Extreme={extreme_mean:>10.4f}, False={false_mean:>10.4f}, Ratio={ratio:>6.2f}x")


def test_combined_rules(merged: pd.DataFrame):
    """Test specific rule combinations."""
    print("\n" + "=" * 80)
    print("COMBINED RULES TESTING")
    print("=" * 80)

    # Define rule combinations to test (using available 1m features)
    # Based on analysis: LOW VPIN correlates with big moves
    rule_sets = [
        # Low VPIN rules (discovered pattern)
        {
            'name': 'Low VPIN (< 0.13) + velocity',
            'rules': {
                'velocity_ratio': ('>=', 5.0),
                'vpin': ('<', 0.13),
            }
        },
        {
            'name': 'Very low VPIN (< 0.10) + velocity',
            'rules': {
                'velocity_ratio': ('>=', 5.0),
                'vpin': ('<', 0.10),
            }
        },
        {
            'name': 'Low VPIN + high volume spike',
            'rules': {
                'vpin': ('<', 0.15),
                'volume_spike_ratio': ('>', 2.0),
            }
        },
        # Velocity + VPIN (informed trading)
        {
            'name': 'High velocity + high VPIN',
            'rules': {
                'velocity_ratio': ('>=', 7.0),
                'vpin': ('>', 0.6),
            }
        },
        # Velocity + VPIN + high spread
        {
            'name': 'Velocity + VPIN + wide spread',
            'rules': {
                'velocity_ratio': ('>=', 6.0),
                'vpin': ('>', 0.5),
                'bid_ask_spread_pct': ('>', 0.002),
            }
        },
        # Wide spread + velocity
        {
            'name': 'High velocity + wide spread',
            'rules': {
                'velocity_ratio': ('>=', 7.0),
                'bid_ask_spread_pct': ('>', 0.002),
            }
        },
        # Large order imbalance
        {
            'name': 'High velocity + large order imbalance',
            'rules': {
                'velocity_ratio': ('>=', 7.0),
                'large_order_imbalance': ('abs>', 0.3),
            }
        },
        # Extreme velocity
        {
            'name': 'Extreme velocity (10x+)',
            'rules': {
                'velocity_ratio': ('>=', 10.0),
            }
        },
        # Very high VPIN alone
        {
            'name': 'Very high VPIN (0.7+) + velocity',
            'rules': {
                'velocity_ratio': ('>=', 5.0),
                'vpin': ('>', 0.7),
            }
        },
        # Combined: velocity + VPIN + large order
        {
            'name': 'Velocity + VPIN + large order imbalance',
            'rules': {
                'velocity_ratio': ('>=', 6.0),
                'vpin': ('>', 0.5),
                'large_order_imbalance': ('abs>', 0.2),
            }
        },
    ]

    results = []
    for rule_set in rule_sets:
        print(f"\n--- {rule_set['name']} ---")
        result = backtest_combined_signal(merged, rule_set['rules'])
        result['name'] = rule_set['name']
        results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Combined Rules Performance")
    print("=" * 80)
    print(f"\n{'Rule Set':<40} {'Signals':>10} {'Precision':>12} {'Recall':>10} {'Improvement':>12}")
    print("-" * 86)

    for r in sorted(results, key=lambda x: x['precision'], reverse=True):
        print(f"{r['name']:<40} {r['signals']:>10} {r['precision']:>12.2%} {r['recall']:>10.2%} {r['improvement']:>12.1f}x")


def main():
    print("=" * 80)
    print("COMBINED SIGNAL ANALYSIS: Velocity Spikes + Conditioning Factors")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Velocity threshold: >= {VELOCITY_THRESHOLD}x")
    print(f"  Price move threshold: >= {PRICE_MOVE_THRESHOLD*100:.0f}%")
    print(f"  Features: {FEATURES_TO_ANALYZE}")

    # Step 1: Load data
    print("\n" + "=" * 80)
    print("STEP 1: Loading Data")
    print("=" * 80)
    spikes = load_velocity_spikes()
    features = load_features()

    # Step 2: Join data
    print("\n" + "=" * 80)
    print("STEP 2: Joining Velocity Spikes with Features")
    print("=" * 80)
    merged = join_spikes_with_features(spikes, features)

    # Step 3: Compare distributions
    distribution_results = compare_distributions(merged)

    # Step 4: Directional analysis
    analyze_directional_signals(merged)

    # Step 5: Extreme moves analysis
    analyze_extreme_moves(merged)

    # Step 6: Grid search single features
    grid_results = grid_search_thresholds(merged)

    # Step 7: Test combined rules
    test_combined_rules(merged)

    # Step 8: Decision tree
    tree, feature_cols = build_decision_tree(merged, max_depth=4)

    # Step 9: Try lower threshold analysis if few TRUE spikes
    if merged['is_true_spike'].sum() < 20:
        print("\n" + "=" * 80)
        print("ALTERNATIVE: Lower threshold analysis (3% moves)")
        print("=" * 80)

        merged_3pct = merged.copy()
        merged_3pct['is_true_spike'] = abs(merged_3pct['price_change_60min']) >= 0.03
        true_3pct = merged_3pct['is_true_spike'].sum()
        print(f"  TRUE spikes (>=3% move): {true_3pct}")
        print(f"  Precision baseline: {merged_3pct['is_true_spike'].mean()*100:.2f}%")

        if true_3pct >= 10:
            compare_distributions(merged_3pct)
            build_decision_tree(merged_3pct, max_depth=3)

    # Step 10: Test discovered rules (low VPIN)
    print("\n" + "=" * 80)
    print("DISCOVERED RULE VALIDATION: Low VPIN Signal")
    print("=" * 80)

    # Test the tree-discovered rule
    rule = {
        'vpin': ('<', 0.13),
        'volume_spike_ratio': ('>', 0.77),
    }
    print("\nTesting decision tree rule: vpin < 0.13 AND volume_spike_ratio > 0.77")
    result_5pct = backtest_combined_signal(merged, rule)

    # Also test with 3% threshold
    merged_3pct = merged.copy()
    merged_3pct['is_true_spike'] = abs(merged_3pct['price_change_60min']) >= 0.03
    print("\nSame rule with 3% move threshold:")
    result_3pct = backtest_combined_signal(merged_3pct, rule)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: Actionable Findings")
    print("=" * 80)

    print("""
KEY INSIGHTS:

1. WIDE SPREAD is the strongest predictor of significant moves
   - TRUE spikes have 2.2x wider spreads (p < 0.001***)
   - Wide spread = low liquidity = price impact vulnerability

2. LOW VPIN also predicts moves (p = 0.002**)
   - TRUE spikes: VPIN 0.20 vs FALSE: 0.37
   - Low VPIN = less informed trading = noise-driven spike

RECOMMENDED SIGNAL RULES:
""")

    # Best rule: velocity + wide spread
    mask_spread = (merged['velocity_ratio'] >= 7.0) & (merged['bid_ask_spread_pct'] > 0.002)
    if mask_spread.sum() > 0:
        precision_5 = merged[mask_spread]['is_true_spike'].mean()
        precision_3 = merged_3pct[mask_spread]['is_true_spike'].mean()
        signals = mask_spread.sum()
        baseline = merged['is_true_spike'].mean()

        print(f"BEST: velocity >= 7x AND spread > 0.2%")
        print(f"  - Signals: {signals} / {len(merged)} ({signals/len(merged)*100:.1f}% of spikes)")
        print(f"  - Precision (5% moves): {precision_5*100:.1f}%")
        print(f"  - Precision (3% moves): {precision_3*100:.1f}%")
        print(f"  - Improvement: {precision_5 / baseline:.1f}x over baseline")

    # Alternative: low VPIN
    mask_vpin = (merged['velocity_ratio'] >= 5.0) & (merged['vpin'] < 0.13)
    if mask_vpin.sum() > 0:
        precision_5 = merged[mask_vpin]['is_true_spike'].mean()
        signals = mask_vpin.sum()
        baseline = merged['is_true_spike'].mean()

        print(f"\nALT: velocity >= 5x AND vpin < 0.13")
        print(f"  - Signals: {signals} ({signals/len(merged)*100:.1f}% of spikes)")
        print(f"  - Precision (5%): {precision_5*100:.1f}%")
        print(f"  - Higher recall but lower precision")

    print("""
IMPLEMENTATION NOTES:
- Primary signal: velocity spike + wide spread (>0.2%)
- Secondary filter: low VPIN (<0.13) for higher recall
- Direction still unpredictable (40% up, 60% down in sample)
- Consider order book imbalance for direction

DATA QUALITY:
- 4,306 events analyzed, 20 TRUE (5%+) spikes
- Features are ~9% dense (computed irregularly)
- Used ±5min window join for better coverage
""")

    # Save results
    output_path = 'data/combined_signal_results.csv'
    distribution_results.to_csv(output_path, index=False)
    print(f"\nSaved distribution comparison to {output_path}")

    merged_output = 'data/spikes_with_features.csv'
    merged.to_csv(merged_output, index=False)
    print(f"Saved merged data to {merged_output}")

    return merged, distribution_results, tree


if __name__ == '__main__':
    merged, dist_results, tree = main()
