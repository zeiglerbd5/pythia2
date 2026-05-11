#!/usr/bin/env python3
"""
Train XGBoost fizzler filter for Pythia2 loading strategy.

Builds a classifier to distinguish winners from fizzlers among loading_v1 alerts.
Goal: filter out fizzlers (73% of alerts) while keeping as many winners as possible.

Key finding from EDA: momentum_1h and close_position are the strongest separators.
Winners tend to enter BEFORE the move (low momentum, low close_position).
Fizzlers tend to enter AFTER price has already moved up.

Data sources:
  - paper_trades.db (SQLite): all loading_v1 trades with outcomes
  - elite_movers.duckdb: entry-time features for Apr 17+ trades
  - backtest_ohlcv.duckdb: 1-min candles to compute features for older trades
  - elite_movers.duckdb symbol_stats: repeat spiker labels
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2/venv/lib/python3.9/site-packages')

import os
import json
import warnings
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

MODEL_DIR = '/Users/bz/Pythia2/models/fizzler_filter'
os.makedirs(MODEL_DIR, exist_ok=True)

# Base features from the scanner
BASE_FEATURES = [
    'vol_trend', 'vol_last_vs_avg', 'vol_accel', 'natr', 'bb_width',
    'momentum_1h', 'price_range', 'close_position', 'bot_net_pct',
    'spread_pct', 'repeat_spiker', 'hour_utc', 'score'
]

# Engineered features we'll add
ENGINEERED_FEATURES = [
    'already_moved',         # momentum_1h * close_position -- captures "already ran"
    'vol_momentum_ratio',    # vol_trend / (1 + abs(momentum_1h))
    'tightness',             # 1 / (1 + bb_width) -- tight BBands = pre-breakout
    'natr_x_range',          # natr * price_range -- volatility explosion indicator
    'is_night_utc',          # hour_utc in [0..8] -- Asian session
    'is_us_open',            # hour_utc in [13..17] -- US market hours
    'high_momentum',         # momentum_1h > 5 (already moved a lot)
    'low_close_pos',         # close_position < 0.3 (near bottom of range)
    'vol_surge',             # vol_last_vs_avg > 3 (strong volume spike)
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

# ---------------------------------------------------------------------------
# Step 1: Load paper trades with outcomes
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading paper trades and building training dataset")
print("=" * 70)

sqlite_con = sqlite3.connect('/Users/bz/Pythia2/paper_trades.db')
trades_df = pd.read_sql_query("""
    SELECT id, symbol, entry_price, entry_time, exit_price, exit_time,
           exit_reason, position_size, realized_pnl, is_open
    FROM trades
    WHERE strategy = 'loading_v1' AND is_open = 0
    ORDER BY entry_time
""", sqlite_con)
sqlite_con.close()

trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

# Deduplicate: keep the phase1 ($200) entry per (symbol, entry_time rounded to minute)
trades_df['entry_minute'] = trades_df['entry_time'].dt.tz_localize(None).dt.floor('min')
trades_df = trades_df.sort_values('position_size').drop_duplicates(
    subset=['symbol', 'entry_minute'], keep='first'
).sort_values('entry_time').reset_index(drop=True)

# Label: winner=1, fizzler=0
trades_df['is_winner'] = (
    (trades_df['exit_reason'] == 'trailing_stop') |
    (trades_df['realized_pnl'] > 0)
).astype(int)

# Drop orphaned_restart (no real outcome)
trades_df = trades_df[trades_df['exit_reason'] != 'orphaned_restart'].reset_index(drop=True)

print(f"Total trades after dedup: {len(trades_df)}")
print(f"Winners: {trades_df['is_winner'].sum()} ({trades_df['is_winner'].mean()*100:.1f}%)")
print(f"Fizzlers: {(1-trades_df['is_winner']).sum()} ({(1-trades_df['is_winner']).mean()*100:.1f}%)")
print(f"Date range: {trades_df['entry_time'].min()} to {trades_df['entry_time'].max()}")
print()
print("Exit reason breakdown:")
for reason, grp in trades_df.groupby('exit_reason'):
    w = grp['is_winner'].sum()
    print(f"  {reason:>20s}: n={len(grp):>4d}, winners={w:>3d}, "
          f"avg_pnl=${grp['realized_pnl'].mean():>8.2f}")

# ---------------------------------------------------------------------------
# Step 2: Join features from elite_movers trades (Apr 17+)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Joining features from elite_movers trades table")
print("=" * 70)

elite_con = duckdb.connect('/Users/bz/Pythia2/data/elite_movers.duckdb', read_only=True)

elite_entries = elite_con.execute("""
    SELECT timestamp, symbol, score, vol_trend, vol_last_vs_avg, vol_accel,
           natr, bb_width, momentum_1h, price_range, close_position,
           bot_net_pct, spread_pct, repeat_spiker, hour_utc
    FROM trades
    WHERE event_type = 'entry'
    ORDER BY timestamp
""").fetchdf()

# Load repeat spiker info from symbol_stats
symbol_stats = elite_con.execute("""
    SELECT symbol, is_repeat_spiker FROM symbol_stats
""").fetchdf()
repeat_spikers = set(symbol_stats[symbol_stats['is_repeat_spiker']]['symbol'].tolist())
elite_con.close()

print(f"Elite entry events: {len(elite_entries)}")
print(f"Repeat spikers in symbol_stats: {len(repeat_spikers)}")

# Match elite entries to paper trades by symbol + timestamp proximity
elite_entries['timestamp'] = pd.to_datetime(elite_entries['timestamp']).dt.tz_localize(None)
elite_entries['entry_minute'] = elite_entries['timestamp'].dt.floor('min')

# Merge
merged = trades_df.merge(
    elite_entries.drop(columns=['timestamp']),
    on=['symbol', 'entry_minute'],
    how='left',
    suffixes=('', '_elite')
)

has_elite = merged['vol_trend'].notna()
print(f"Trades matched to elite features: {has_elite.sum()}")
print(f"Trades needing OHLCV feature computation: {(~has_elite).sum()}")

# ---------------------------------------------------------------------------
# Step 3: Compute features from OHLCV for trades missing elite features
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Computing features from backtest OHLCV for older trades")
print("=" * 70)

needs_features = merged[~has_elite].copy()

if len(needs_features) > 0:
    ohlcv_con = duckdb.connect('/Users/bz/Pythia2/data/backtest_ohlcv.duckdb', read_only=True)

    computed_features = []
    symbols_times = list(zip(needs_features.index, needs_features['symbol'], needs_features['entry_time']))

    for i, (idx, symbol, entry_time) in enumerate(symbols_times):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(symbols_times)}...")

        try:
            entry_ts = entry_time.tz_localize(None).strftime('%Y-%m-%d %H:%M:%S') if entry_time.tzinfo else entry_time.strftime('%Y-%m-%d %H:%M:%S')
            candles = ohlcv_con.execute(f"""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = '{symbol}'
                  AND timestamp < '{entry_ts}'
                ORDER BY timestamp DESC
                LIMIT 60
            """).fetchdf()

            if len(candles) < 20:
                computed_features.append((idx, {}))
                continue

            candles = candles.sort_values('timestamp').reset_index(drop=True)
            n = len(candles)

            # vol_trend
            if n >= 40:
                vol_first20 = candles['volume'].iloc[:20].mean()
                vol_last20 = candles['volume'].iloc[-20:].mean()
            else:
                half = n // 2
                vol_first20 = candles['volume'].iloc[:half].mean()
                vol_last20 = candles['volume'].iloc[half:].mean()
            vol_trend = vol_last20 / max(vol_first20, 1e-10)

            # vol_last_vs_avg
            avg_vol = candles['volume'].mean()
            vol_last_vs_avg = candles['volume'].iloc[-1] / max(avg_vol, 1e-10)

            # vol_accel
            if n >= 10:
                vol_accel = candles['volume'].iloc[-5:].mean() / max(candles['volume'].iloc[-10:-5].mean(), 1e-10)
            else:
                vol_accel = 1.0

            # NATR
            highs = candles['high'].values
            lows = candles['low'].values
            closes = candles['close'].values
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1])
                )
            )
            atr14 = np.mean(tr[-14:]) if len(tr) >= 14 else (np.mean(tr) if len(tr) > 0 else 0)
            last_close = closes[-1]
            natr = (atr14 / max(last_close, 1e-10)) * 100

            # BB width
            if n >= 20:
                sma20 = np.mean(closes[-20:])
                std20 = np.std(closes[-20:])
            else:
                sma20 = np.mean(closes)
                std20 = np.std(closes)
            bb_width = (4 * std20) / max(sma20, 1e-10)

            # momentum_1h
            momentum_1h = ((closes[-1] - closes[0]) / max(closes[0], 1e-10)) * 100

            # price_range
            range_high = np.max(highs)
            range_low = np.min(lows)
            price_range = ((range_high - range_low) / max(range_low, 1e-10)) * 100

            # close_position
            if range_high > range_low:
                close_position = (last_close - range_low) / (range_high - range_low)
            else:
                close_position = 0.5

            # repeat_spiker
            is_repeat = symbol in repeat_spikers

            # hour_utc
            hour_utc = entry_time.hour if not hasattr(entry_time, 'tz') or entry_time.tzinfo is None else entry_time.tz_localize(None).hour

            computed_features.append((idx, {
                'vol_trend': vol_trend,
                'vol_last_vs_avg': vol_last_vs_avg,
                'vol_accel': vol_accel,
                'natr': natr,
                'bb_width': bb_width,
                'momentum_1h': momentum_1h,
                'price_range': price_range,
                'close_position': close_position,
                'bot_net_pct': np.nan,
                'spread_pct': np.nan,
                'repeat_spiker': is_repeat,
                'hour_utc': hour_utc,
                'score': np.nan,
            }))
        except Exception as e:
            computed_features.append((idx, {}))

    ohlcv_con.close()

    n_success = sum(1 for _, d in computed_features if 'vol_trend' in d)
    print(f"  Computed features for {n_success}/{len(symbols_times)} trades")

    # Apply computed features
    for idx, feat_dict in computed_features:
        for col, val in feat_dict.items():
            merged.loc[idx, col] = val

# ---------------------------------------------------------------------------
# Step 4: Engineer additional features
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Engineering additional features")
print("=" * 70)

# Convert repeat_spiker to int
merged['repeat_spiker'] = merged['repeat_spiker'].fillna(False).astype(int)

# Drop rows missing core features
merged = merged.dropna(subset=['vol_trend']).reset_index(drop=True)

# Fill NaN in bot_net_pct, spread_pct, score with medians
for col in ['bot_net_pct', 'spread_pct', 'score']:
    n_na = merged[col].isna().sum()
    if n_na > 0:
        med = merged[col].median()
        merged[col] = merged[col].fillna(med)
        print(f"  Filled {n_na} NaN in {col} with median={med:.4f}")

# Engineered features
merged['already_moved'] = merged['momentum_1h'] * merged['close_position']
merged['vol_momentum_ratio'] = merged['vol_trend'] / (1 + np.abs(merged['momentum_1h']))
merged['tightness'] = 1.0 / (1.0 + merged['bb_width'])
merged['natr_x_range'] = merged['natr'] * merged['price_range']
merged['is_night_utc'] = ((merged['hour_utc'] >= 0) & (merged['hour_utc'] <= 8)).astype(int)
merged['is_us_open'] = ((merged['hour_utc'] >= 13) & (merged['hour_utc'] <= 17)).astype(int)
merged['high_momentum'] = (merged['momentum_1h'] > 5).astype(int)
merged['low_close_pos'] = (merged['close_position'] < 0.3).astype(int)
merged['vol_surge'] = (merged['vol_last_vs_avg'] > 3).astype(int)

print(f"\nTotal features: {len(ALL_FEATURES)}")
print(f"Training samples: {len(merged)}")
print(f"  Winners: {merged['is_winner'].sum()} ({merged['is_winner'].mean()*100:.1f}%)")
print(f"  Fizzlers: {(1-merged['is_winner']).sum()}")

# Feature coverage
for col in ALL_FEATURES:
    avail = merged[col].notna().sum()
    if avail < len(merged):
        print(f"  WARNING: {col}: {avail}/{len(merged)} non-null")

X = merged[ALL_FEATURES].values.astype(np.float64)
y = merged['is_winner'].values

# Replace any remaining NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)

print(f"\nFinal dataset: X={X.shape}, y balance: {y.mean():.3f} positive rate")

# ---------------------------------------------------------------------------
# Step 5: Train XGBoost with stratified k-fold CV
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Training XGBoost with 5-fold stratified CV")
print("=" * 70)

n_neg = (y == 0).sum()
n_pos = (y == 1).sum()
scale_pos_weight = n_neg / n_pos

# More conservative model to avoid overfitting on 532 samples
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 3,
    'min_child_weight': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'learning_rate': 0.03,
    'scale_pos_weight': scale_pos_weight,
    'reg_alpha': 1.0,
    'reg_lambda': 3.0,
    'gamma': 1.0,
    'seed': 42,
    'verbosity': 0,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_y_true = []
all_y_prob = []
fold_aucs = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=ALL_FEATURES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=ALL_FEATURES)

    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    y_prob = model.predict(dval)
    auc = roc_auc_score(y_val, y_prob)
    fold_aucs.append(auc)
    fold_models.append(model)

    all_y_true.extend(y_val.tolist())
    all_y_prob.extend(y_prob.tolist())

    print(f"  Fold {fold+1}: AUC={auc:.4f}, best_iteration={model.best_iteration}, "
          f"val_winners={y_val.sum()}/{len(y_val)}")

all_y_true = np.array(all_y_true)
all_y_prob = np.array(all_y_prob)

print(f"\nMean CV AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
print(f"PR-AUC: {average_precision_score(all_y_true, all_y_prob):.4f}")

# Classification report at 0.5 threshold
y_pred_50 = (all_y_prob >= 0.5).astype(int)
print(f"\nClassification Report (threshold=0.5):")
print(classification_report(all_y_true, y_pred_50, target_names=['Fizzler', 'Winner']))

# ---------------------------------------------------------------------------
# Step 6: Threshold analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Threshold Analysis")
print("=" * 70)
print("\nObjective: Filter fizzlers while keeping >= 90% of winners.")
print("We prefer letting fizzlers through over missing a big winner.\n")

print(f"{'Threshold':>10} {'Fizzlers Filtered':>18} {'Winners Kept':>14} {'Winners Lost':>13} "
      f"{'Remaining':>10} {'Win Rate':>10} {'Lift':>6}")
print("-" * 88)

baseline_wr = n_pos / (n_pos + n_neg) * 100
threshold_results = []

for threshold in np.arange(0.10, 0.65, 0.05):
    threshold = round(threshold, 2)
    passes = all_y_prob >= threshold

    fizzlers_filtered = ((all_y_true == 0) & ~passes).sum()
    winners_kept = ((all_y_true == 1) & passes).sum()
    winners_lost = ((all_y_true == 1) & ~passes).sum()

    fizzler_filter_rate = fizzlers_filtered / n_neg * 100
    winner_keep_rate = winners_kept / n_pos * 100

    remaining = passes.sum()
    precision = winners_kept / remaining * 100 if remaining > 0 else 0
    lift = precision / baseline_wr

    marker = ""
    if winner_keep_rate >= 90:
        marker = " <-- 90%+ retained"
    elif winner_keep_rate >= 85:
        marker = " <-- 85%+ retained"

    print(f"{threshold:>10.2f} {fizzler_filter_rate:>16.1f}% {winner_keep_rate:>12.1f}% "
          f"{winners_lost:>13d} {remaining:>10d} {precision:>9.1f}% {lift:>5.2f}x{marker}")

    threshold_results.append({
        'threshold': threshold,
        'fizzlers_filtered_pct': round(fizzler_filter_rate, 1),
        'winners_kept_pct': round(winner_keep_rate, 1),
        'remaining_trades': int(remaining),
        'precision_pct': round(precision, 1),
        'winners_lost': int(winners_lost),
        'fizzlers_filtered': int(fizzlers_filtered),
        'lift': round(lift, 2),
    })

print(f"\nBaseline: {n_pos} winners / {n_pos + n_neg} trades = {baseline_wr:.1f}% win rate")

# Find recommended threshold
recommended = None
for r in threshold_results:
    if r['winners_kept_pct'] >= 90.0:
        if recommended is None or r['fizzlers_filtered_pct'] > recommended['fizzlers_filtered_pct']:
            recommended = r

# Also find a more aggressive option (85% retention)
aggressive = None
for r in threshold_results:
    if r['winners_kept_pct'] >= 85.0:
        if aggressive is None or r['fizzlers_filtered_pct'] > aggressive['fizzlers_filtered_pct']:
            aggressive = r

# ---------------------------------------------------------------------------
# Step 7: PnL-weighted analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: PnL Impact Analysis")
print("=" * 70)

# Match predictions back to trades for PnL analysis
merged['y_prob'] = np.nan
# Re-run through folds to get out-of-fold predictions aligned
merged_indices = merged.index.values
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_val = X[val_idx]
    dval = xgb.DMatrix(X_val, feature_names=ALL_FEATURES)
    y_prob = fold_models[fold].predict(dval)
    for i, vi in enumerate(val_idx):
        merged.loc[merged_indices[vi], 'y_prob'] = y_prob[i]

if recommended:
    thr = recommended['threshold']
    passes = merged['y_prob'] >= thr
    pnl_pass = merged.loc[passes, 'realized_pnl'].sum()
    pnl_all = merged['realized_pnl'].sum()
    pnl_filtered = merged.loc[~passes, 'realized_pnl'].sum()

    print(f"\nAt threshold={thr}:")
    print(f"  Total PnL (all trades): ${pnl_all:>10.2f}")
    print(f"  PnL of passed trades:   ${pnl_pass:>10.2f}")
    print(f"  PnL of filtered trades: ${pnl_filtered:>10.2f}")
    print(f"  Trades taken: {passes.sum()} / {len(merged)} ({passes.sum()/len(merged)*100:.1f}%)")

    # Show what winners we'd miss
    missed_winners = merged[(merged['is_winner']==1) & (~passes)]
    if len(missed_winners) > 0:
        print(f"\n  Winners that would be MISSED ({len(missed_winners)}):")
        for _, row in missed_winners.iterrows():
            print(f"    {row['symbol']:>15s}: pnl=${row['realized_pnl']:>8.2f}, "
                  f"exit={row['exit_reason']}, prob={row['y_prob']:.3f}")

# ---------------------------------------------------------------------------
# Step 8: Train final model on all data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: Training final model on all data")
print("=" * 70)

dtrain_full = xgb.DMatrix(X, label=y, feature_names=ALL_FEATURES)
final_model = xgb.train(
    params, dtrain_full,
    num_boost_round=200,
    verbose_eval=False
)

# Feature importance
importance = final_model.get_score(importance_type='gain')
imp_df = pd.DataFrame([
    {'feature': k, 'importance': v} for k, v in importance.items()
]).sort_values('importance', ascending=False)

print("\nFeature Importance (gain):")
for _, row in imp_df.iterrows():
    bar = '#' * int(row['importance'] / imp_df['importance'].max() * 40)
    print(f"  {row['feature']:>25s}: {row['importance']:>10.2f}  {bar}")

# ---------------------------------------------------------------------------
# Step 9: Save model and artifacts
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 9: Saving model and artifacts")
print("=" * 70)

model_path = os.path.join(MODEL_DIR, 'fizzler_xgb.json')
final_model.save_model(model_path)
print(f"  Model saved: {model_path}")

config = {
    'feature_cols': ALL_FEATURES,
    'base_features': BASE_FEATURES,
    'engineered_features': ENGINEERED_FEATURES,
    'params': params,
    'threshold_analysis': threshold_results,
    'recommended_threshold': recommended['threshold'] if recommended else 0.3,
    'aggressive_threshold': aggressive['threshold'] if aggressive else None,
    'cv_auc_mean': round(float(np.mean(fold_aucs)), 4),
    'cv_auc_std': round(float(np.std(fold_aucs)), 4),
    'pr_auc': round(float(average_precision_score(all_y_true, all_y_prob)), 4),
    'n_training_samples': int(len(X)),
    'n_winners': int(n_pos),
    'n_fizzlers': int(n_neg),
    'baseline_win_rate': round(float(baseline_wr), 1),
    'training_date': datetime.now().isoformat(),
    'feature_importance': {row['feature']: round(row['importance'], 2) for _, row in imp_df.iterrows()},
}
config_path = os.path.join(MODEL_DIR, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"  Config saved: {config_path}")

# Save feature medians for inference-time imputation
feature_medians = {}
for col in ALL_FEATURES:
    feature_medians[col] = float(merged[col].median())
medians_path = os.path.join(MODEL_DIR, 'feature_medians.json')
with open(medians_path, 'w') as f:
    json.dump(feature_medians, f, indent=2)
print(f"  Feature medians saved: {medians_path}")

# ---------------------------------------------------------------------------
# Step 10: Generate plots
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 10: Generating plots")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Pythia2 Fizzler Filter - XGBoost Analysis', fontsize=14, fontweight='bold')

# 1. Feature importance (top 15)
ax = axes[0, 0]
imp_top = imp_df.head(15).sort_values('importance', ascending=True)
colors = ['#2196F3' if f in BASE_FEATURES else '#FF9800' for f in imp_top['feature']]
ax.barh(imp_top['feature'], imp_top['importance'], color=colors)
ax.set_title('Feature Importance (Gain)')
ax.set_xlabel('Gain')
# Legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#2196F3', label='Base features'),
    Patch(facecolor='#FF9800', label='Engineered features')
], loc='lower right', fontsize=8)

# 2. ROC curve
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={np.mean(fold_aucs):.3f}')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve (5-Fold CV)')
ax.legend()
ax.grid(True, alpha=0.2)

# 3. Threshold trade-off
ax = axes[1, 0]
thresholds = [r['threshold'] for r in threshold_results]
fizzler_pcts = [r['fizzlers_filtered_pct'] for r in threshold_results]
winner_pcts = [r['winners_kept_pct'] for r in threshold_results]
ax.plot(thresholds, fizzler_pcts, 'r-o', label='Fizzlers Filtered %', linewidth=2, markersize=5)
ax.plot(thresholds, winner_pcts, 'g-o', label='Winners Kept %', linewidth=2, markersize=5)
ax.axhline(y=90, color='green', linestyle='--', alpha=0.3, label='90% Winner Retention')
if recommended:
    ax.axvline(x=recommended['threshold'], color='blue', linestyle=':', alpha=0.5,
               label=f"Recommended ({recommended['threshold']})")
ax.set_xlabel('Threshold')
ax.set_ylabel('Percentage')
ax.set_title('Fizzler Filter vs Winner Retention Trade-off')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Score distribution overlay
ax = axes[1, 1]
winner_probs = all_y_prob[all_y_true == 1]
fizzler_probs = all_y_prob[all_y_true == 0]
bins = np.linspace(0, 1, 25)
ax.hist(fizzler_probs, bins=bins, alpha=0.5, label=f'Fizzlers (n={len(fizzler_probs)})',
        color='red', density=True)
ax.hist(winner_probs, bins=bins, alpha=0.5, label=f'Winners (n={len(winner_probs)})',
        color='green', density=True)
if recommended:
    ax.axvline(x=recommended['threshold'], color='blue', linestyle='--',
               label=f"Threshold={recommended['threshold']}")
ax.set_xlabel('Model Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('Score Distribution: Winners vs Fizzlers')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, 'fizzler_filter_analysis.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Analysis plot saved: {plot_path}")
plt.close()

# Confusion matrix at recommended threshold
if recommended:
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    y_pred_rec = (all_y_prob >= recommended['threshold']).astype(int)
    cm = confusion_matrix(all_y_true, y_pred_rec)
    im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
    ax2.set_title(f"Confusion Matrix (threshold={recommended['threshold']})")
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Filter', 'Pass'])
    ax2.set_yticklabels(['Fizzler', 'Winner'])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=16,
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig2.colorbar(im)
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    fig2.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"  Confusion matrix saved: {cm_path}")
    plt.close()

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

rec_thr = recommended['threshold'] if recommended else 'N/A'
rec_filt = recommended['fizzlers_filtered_pct'] if recommended else 'N/A'
rec_kept = recommended['winners_kept_pct'] if recommended else 'N/A'
rec_wr = recommended['precision_pct'] if recommended else 'N/A'
rec_lost = recommended['winners_lost'] if recommended else 'N/A'
rec_lift = recommended['lift'] if recommended else 'N/A'

print(f"""
Model Performance:
  - Cross-validated ROC AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})
  - PR AUC: {average_precision_score(all_y_true, all_y_prob):.4f}
  - Training samples: {len(X)} ({n_pos} winners, {n_neg} fizzlers)

Conservative Operating Point (threshold={rec_thr}):
  - Fizzlers filtered: {rec_filt}%
  - Winners retained:  {rec_kept}%
  - Win rate:          {baseline_wr:.1f}% -> {rec_wr}% ({rec_lift}x lift)
  - Winners lost:      {rec_lost} out of {n_pos}
""")

if aggressive and aggressive != recommended:
    print(f"""Aggressive Operating Point (threshold={aggressive['threshold']}):
  - Fizzlers filtered: {aggressive['fizzlers_filtered_pct']}%
  - Winners retained:  {aggressive['winners_kept_pct']}%
  - Win rate:          {baseline_wr:.1f}% -> {aggressive['precision_pct']}% ({aggressive['lift']}x lift)
  - Winners lost:      {aggressive['winners_lost']} out of {n_pos}
""")

print(f"""Key Insights:
  - momentum_1h and close_position are the strongest predictors
  - Winners enter BEFORE price runs (low momentum, low close_position)
  - Fizzlers enter AFTER price has already moved (chasing)
  - The model has modest but real discriminative power

Saved Artifacts:
  - Model:           {model_path}
  - Config:          {config_path}
  - Feature medians: {medians_path}
  - Analysis plot:   {plot_path}
""")
