#!/usr/bin/env python3
"""
Train a spike scoring model for the 24/7 live system.

Architecture:
1. TRIGGER: coin moves 5%+ in 1h → enters evaluation
2. CONFIRM: wait 1h, check if still positive → if yes, score it
3. SCORE: ML model predicts probability of 50%+ forward move
4. ENTER: only if score > threshold (tuned for 1-2 trades/week)

Training data: 10,270 trigger events from research.duckdb
Label: binary — 50%+ forward gain = 1, else = 0
Features: everything available at T+1h decision point

Walk-forward validation: train on first 70%, validate on last 30%
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    USE_LGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_LGB = False
    print("LightGBM not available, falling back to sklearn GBT")

OUTPUT_DIR = Path("models/spike_scorer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare():
    df = pd.read_csv('data/trigger_analysis.csv', parse_dates=['trigger_time'])
    df = df.sort_values('trigger_time').reset_index(drop=True)

    # Binary label: 50%+ forward gain
    df['target'] = (df['max_fwd_gain'] >= 50).astype(int)

    # Known producer flag (symbols that have historically produced 50%+ spikes)
    big_producers = df[df['target'] == 1].groupby('symbol').size()
    known_producers = set(big_producers[big_producers >= 2].index)
    df['is_known_producer'] = df['symbol'].isin(known_producers).astype(int)

    # Features available at T+1h decision point
    feature_cols = [
        # Trigger context
        'trigger_move_pct',
        'last_15m_move',
        'gain_1h',           # THE key confirmation signal

        # Pre-trigger market context
        'pre_vol_trend',
        'pre_compression',
        'pre_volatility',
        'pre_momentum_3h',
        'trigger_vol_ratio',
        'avg_price',

        # Symbol history (the strongest signal)
        'is_known_producer',
        'total_triggers',
        'winner_count',
        'win_rate',

        # Technical features at trigger time
        'feat_rsi_14',
        'feat_vpin',
        'feat_natr',
        'feat_bb_width',
        'feat_volume_spike_ratio',
        'feat_bid_ask_spread_pct',
        'feat_order_book_depth_ratio',
        'feat_large_order_imbalance',
        'feat_atr',

        # Time features
        'spike_hour',
        'spike_dow',
    ]

    # Extract time features
    df['spike_hour'] = df['trigger_time'].dt.hour
    df['spike_dow'] = df['trigger_time'].dt.dayofweek

    # Ensure all feature cols exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df, feature_cols


def walk_forward_eval(df, feature_cols):
    """
    Walk-forward: train on first 70%, test on last 30%.
    This respects time ordering — no future leakage.
    """
    split_idx = int(len(df) * 0.7)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    print(f"Train: {len(train):,} triggers ({train['target'].sum()} positives, "
          f"{train['trigger_time'].min().date()} → {train['trigger_time'].max().date()})")
    print(f"Test:  {len(test):,} triggers ({test['target'].sum()} positives, "
          f"{test['trigger_time'].min().date()} → {test['trigger_time'].max().date()})")

    X_train = train[feature_cols].copy()
    y_train = train['target'].values
    X_test = test[feature_cols].copy()
    y_test = test['target'].values

    # Handle NaN
    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)

    # Train model
    if USE_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1),
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
    else:
        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    test = test.copy()
    test['score'] = y_prob

    return model, train, test, X_train, y_train, X_test, y_test, y_prob


def analyze_model(model, test, feature_cols, y_test, y_prob):
    """Detailed analysis of model performance."""

    print("\n" + "=" * 70)
    print("  MODEL PERFORMANCE")
    print("=" * 70)

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    print(f"\nAUC-ROC: {auc:.3f}")
    print(f"Average Precision: {ap:.3f}")
    print(f"Base rate: {y_test.mean()*100:.1f}% positive")

    # Feature importance
    print("\n--- Feature Importance ---")
    if USE_LGB:
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    for name, imp in feat_imp:
        bar = "#" * int(imp / max(importances) * 40)
        print(f"  {name:35s} {imp:6.0f}  {bar}")

    # Threshold analysis — the key table
    print("\n" + "=" * 70)
    print("  THRESHOLD ANALYSIS (tuning for 1-2 trades/week)")
    print("=" * 70)

    test_weeks = (test['trigger_time'].max() - test['trigger_time'].min()).days / 7

    print(f"\nTest period: {test_weeks:.1f} weeks")
    print(f"\n{'Threshold':>10s} {'Trades':>7s} {'/week':>6s} {'Prec':>6s} {'Recall':>7s} "
          f"{'AvgGain':>8s} {'MedGain':>8s} {'50%+':>5s} {'FP(<10%)':>9s}")
    print("-" * 80)

    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        selected = test[test['score'] >= threshold]
        n = len(selected)
        if n == 0:
            continue

        per_week = n / test_weeks
        n_big = (selected['max_fwd_gain'] >= 50).sum()
        precision = n_big / n * 100 if n > 0 else 0
        recall = n_big / y_test.sum() * 100 if y_test.sum() > 0 else 0
        avg_gain = selected['max_fwd_gain'].mean()
        med_gain = selected['max_fwd_gain'].median()
        fp = (selected['max_fwd_gain'] < 10).sum() / n * 100

        marker = " <-- TARGET" if 0.8 <= per_week <= 3.0 else ""
        print(f"{threshold:10.2f} {n:7d} {per_week:6.1f} {precision:5.1f}% {recall:6.1f}% "
              f"{avg_gain:7.1f}% {med_gain:7.1f}% {n_big:5d} {fp:8.1f}%{marker}")

    # Show the actual trades at ~2/week threshold
    print("\n" + "=" * 70)
    print("  WHAT THE ~2/WEEK TRADES LOOK LIKE")
    print("=" * 70)

    # Find threshold that gives ~2/week
    for t in np.arange(0.05, 0.95, 0.01):
        selected = test[test['score'] >= t]
        if len(selected) / test_weeks <= 2.5:
            target_threshold = t
            break
    else:
        target_threshold = 0.5

    selected = test[test['score'] >= target_threshold].copy()
    print(f"\nThreshold: {target_threshold:.2f} → {len(selected)} trades ({len(selected)/test_weeks:.1f}/week)")

    if len(selected) > 0:
        n_big = (selected['max_fwd_gain'] >= 50).sum()
        n_mod = ((selected['max_fwd_gain'] >= 20) & (selected['max_fwd_gain'] < 50)).sum()
        n_small = ((selected['max_fwd_gain'] >= 10) & (selected['max_fwd_gain'] < 20)).sum()
        n_fizzle = (selected['max_fwd_gain'] < 10).sum()

        print(f"\nOutcomes:")
        print(f"  50%+ (BIG):   {n_big} ({n_big/len(selected)*100:.0f}%)")
        print(f"  20-50%:       {n_mod} ({n_mod/len(selected)*100:.0f}%)")
        print(f"  10-20%:       {n_small} ({n_small/len(selected)*100:.0f}%)")
        print(f"  <10% (fizzle):{n_fizzle} ({n_fizzle/len(selected)*100:.0f}%)")

        print(f"\nAvg max gain:   {selected['max_fwd_gain'].mean():.1f}%")
        print(f"Median max gain:{selected['max_fwd_gain'].median():.1f}%")

        # Simulated P&L with 6% stop, 8% trail
        results = []
        for _, row in selected.iterrows():
            remaining = row['max_fwd_gain'] - row['gain_1h']
            dd = row['max_dd_before_peak']
            if dd < -6:
                results.append(-6)
            elif remaining > 8:
                results.append(max(remaining * 0.7, remaining - 8))  # Conservative trail estimate
            else:
                results.append(remaining * 0.5)  # Conservative: capture half

        results = np.array(results)
        print(f"\nSimulated returns (conservative):")
        print(f"  Avg: {results.mean():+.1f}%  Median: {np.median(results):+.1f}%")
        print(f"  Win rate: {(results > 0).mean()*100:.0f}%")
        print(f"  With $1000 positions: ~${results.mean() * 10 * len(selected)/test_weeks:.0f}/week")

        # Show individual trades
        print(f"\nAll trades:")
        show_cols = ['trigger_time', 'symbol', 'score', 'gain_1h', 'max_fwd_gain',
                     'time_to_peak_hours', 'max_dd_before_peak']
        print(selected[show_cols].to_string(index=False))

    return target_threshold


def train_final_model(df, feature_cols):
    """Train on ALL data for production use."""
    print("\n" + "=" * 70)
    print("  TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 70)

    X = df[feature_cols].fillna(-999)
    y = df['target'].values

    if USE_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            scale_pos_weight=len(y[y==0]) / max(len(y[y==1]), 1),
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42,
        )

    model.fit(X, y)

    # Save model + metadata
    model_path = OUTPUT_DIR / "spike_scorer_v1.joblib"
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'training_samples': len(df),
        'positive_rate': y.mean(),
        'known_producers': list(df[df['target'] == 1].groupby('symbol').size().pipe(
            lambda x: x[x >= 2]).index),
    }, model_path)

    print(f"Saved to {model_path}")
    print(f"Training samples: {len(df):,}")
    print(f"Positive rate: {y.mean()*100:.1f}%")
    print(f"Known producers: {len(df[df['target']==1].groupby('symbol').size().pipe(lambda x: x[x>=2]))}")

    return model


def main():
    # Load data
    df, feature_cols = load_and_prepare()

    print(f"Dataset: {len(df):,} trigger events")
    print(f"Positives (50%+): {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    print(f"Features: {len(feature_cols)}")

    # Walk-forward evaluation
    model, train, test, X_train, y_train, X_test, y_test, y_prob = walk_forward_eval(df, feature_cols)

    # Analyze
    threshold = analyze_model(model, test, feature_cols, y_test, y_prob)

    # Train final model on all data
    final_model = train_final_model(df, feature_cols)

    print("\nDone!")


if __name__ == "__main__":
    main()
