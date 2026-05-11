"""
Iteration 3: Final Comprehensive Backtest

Compares:
1. Baseline (all symbols)
2. BTC removed
3. High-spike symbols only
4. Symbol + Volatility filtering combinations

Uses proper time-series split: first 70% train, last 30% test.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
import json
warnings.filterwarnings('ignore')

import duckdb

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)


class OrderbookFeatureComputer:
    """Computes orderbook features."""

    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path, read_only=True)
        self._cache = {}

    def compute_features_at_time(
        self,
        symbol: str,
        timestamp: datetime,
        lookback_minutes: int = 30
    ) -> Dict[str, float]:
        """Compute orderbook features at a specific time."""
        cache_key = f"{symbol}_{timestamp.isoformat()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            query = f"""
                SELECT
                    timestamp, bids, asks, best_bid, best_ask, mid_price, spread_bps
                FROM order_book_snapshots
                WHERE symbol = '{symbol}'
                  AND timestamp >= '{timestamp - timedelta(minutes=lookback_minutes)}'
                  AND timestamp <= '{timestamp}'
                ORDER BY timestamp DESC
                LIMIT 10
            """
            snapshots = self.conn.execute(query).df()

            if len(snapshots) == 0:
                return self._empty_features()

            latest = snapshots.iloc[0]
            bids = json.loads(latest['bids']) if isinstance(latest['bids'], str) else latest['bids']
            asks = json.loads(latest['asks']) if isinstance(latest['asks'], str) else latest['asks']

            features = {'spread_bps': latest['spread_bps']}

            for n in [3, 5, 10]:
                bid_sizes = [b[1] for b in bids[:n]]
                ask_sizes = [a[1] for a in asks[:n]]
                total = sum(bid_sizes) + sum(ask_sizes)
                features[f'imbalance_l{n}'] = (sum(bid_sizes) - sum(ask_sizes)) / total if total > 0 else 0

            bid_depth = sum(b[0] * b[1] for b in bids[:10])
            ask_depth = sum(a[0] * a[1] for a in asks[:10])
            features['depth_ratio'] = bid_depth / ask_depth if ask_depth > 0 else 1.0
            features['log_depth'] = np.log10(max(bid_depth + ask_depth, 1))

            all_sizes = [b[1] for b in bids[:10]] + [a[1] for a in asks[:10]]
            if all_sizes:
                p90 = np.percentile(all_sizes, 90)
                large_bids = sum(1 for b in bids[:10] if b[1] > p90)
                large_asks = sum(1 for a in asks[:10] if a[1] > p90)
                features['large_order_imbalance'] = large_bids - large_asks
            else:
                features['large_order_imbalance'] = 0

            if len(snapshots) > 1:
                spreads = snapshots['spread_bps'].values
                features['spread_volatility'] = np.std(spreads)
                features['spread_trend'] = spreads[0] - spreads[-1]
            else:
                features['spread_volatility'] = 0
                features['spread_trend'] = 0

            self._cache[cache_key] = features
            return features

        except Exception as e:
            return self._empty_features()

    def _empty_features(self) -> Dict[str, float]:
        return {
            'spread_bps': None, 'imbalance_l3': None, 'imbalance_l5': None,
            'imbalance_l10': None, 'depth_ratio': None, 'log_depth': None,
            'large_order_imbalance': None, 'spread_volatility': None, 'spread_trend': None,
        }

    def batch_compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for all rows."""
        print(f"Computing orderbook features for {len(df)} signals...")
        features_list = []
        for idx, row in df.iterrows():
            feat = self.compute_features_at_time(row['symbol'], row['timestamp'])
            features_list.append(feat)
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")
        return pd.DataFrame(features_list)


def train_ensemble(X_train, y_train):
    """Train ensemble of models."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    models = {}

    if HAS_LIGHTGBM:
        models['lightgbm'] = lgb.LGBMClassifier(
            objective='binary',
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            scale_pos_weight=scale_pos_weight,
            verbose=-1,
            random_state=42,
        )
        models['lightgbm'].fit(X_train, y_train)

    if HAS_XGBOOST:
        models['xgboost'] = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=5,
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            verbosity=0,
            random_state=42,
        )
        models['xgboost'].fit(X_train, y_train)

    models['randomforest'] = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight={0: 1, 1: scale_pos_weight},
        random_state=42,
        n_jobs=-1,
    )
    models['randomforest'].fit(X_train, y_train)

    models['gradboost'] = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    models['gradboost'].fit(X_train, y_train)

    return models


def ensemble_predict(models, X):
    """Get ensemble predictions."""
    probas = []
    for name, model in models.items():
        probas.append(model.predict_proba(X)[:, 1])
    return np.mean(probas, axis=0)


def prepare_features(df, orderbook_cols):
    """Prepare feature matrix."""
    feature_cols = [
        'event_priority', 'sentiment_score', 'log_usd_value',
        'volatility_4h', 'momentum_4h', 'volume_ratio', 'rsi_proxy',
    ]

    # Add orderbook features
    for col in orderbook_cols:
        if col in df.columns:
            feature_cols.append(col)

    # Add boolean features
    for col in ['is_bearish_flow', 'is_bullish_flow', 'has_direction']:
        if col in df.columns:
            feature_cols.append(col)
            df = df.copy()
            df[col] = df[col].astype(int)

    available = [f for f in feature_cols if f in df.columns]
    X = df[available].copy()

    # Fill NaN
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median() if not pd.isna(X[col].median()) else 0)

    return X


def evaluate_predictions(y_true, y_proba, label=""):
    """Evaluate predictions at multiple thresholds."""
    results = {'label': label}

    try:
        results['auc'] = roc_auc_score(y_true, y_proba)
    except:
        results['auc'] = 0.5

    try:
        results['pr_auc'] = average_precision_score(y_true, y_proba)
    except:
        results['pr_auc'] = 0.0

    results['thresholds'] = {}

    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        y_pred = (y_proba >= thresh).astype(int)
        n_pred = y_pred.sum()
        n_correct = (y_pred & y_true.astype(int)).sum()
        n_spikes = y_true.sum()

        prec = n_correct / n_pred if n_pred > 0 else 0
        rec = n_correct / n_spikes if n_spikes > 0 else 0

        results['thresholds'][thresh] = {
            'n_pred': int(n_pred),
            'n_correct': int(n_correct),
            'precision': prec,
            'recall': rec,
        }

    return results


def run_backtest():
    """Run comprehensive backtest."""
    print("=" * 80)
    print("ITERATION 3: COMPREHENSIVE BACKTEST")
    print("Symbol + Volatility Filtering Analysis")
    print("=" * 80)

    # Load data
    df = pd.read_csv('/Users/bz/Pythia2/whale_features.csv', parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"\nData: {len(df)} samples, {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"Overall spike rate: {df['spike_10pct_24h'].mean()*100:.2f}%")

    # Add orderbook features
    print("\nAdding orderbook features...")
    ob_computer = OrderbookFeatureComputer('/Users/bz/Pythia2/full_pythia.duckdb')
    ob_features = ob_computer.batch_compute_features(df)
    df = pd.concat([df.reset_index(drop=True), ob_features], axis=1)
    orderbook_cols = list(ob_features.columns)

    # Time-based split: 70% train, 30% test
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\nSplit: {len(train_df)} train ({train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()})")
    print(f"       {len(test_df)} test ({test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()})")

    # Define high-spike symbols
    HIGH_SPIKE_SYMBOLS = ['ETH-USD', 'XRP-USD', 'AAVE-USD', 'SOL-USD', 'UNI-USD', 'DOGE-USD']

    # Configurations to test
    configs = [
        {'name': 'Baseline (all)', 'filter_btc': False, 'symbols': None, 'vol_pct': 0},
        {'name': 'No BTC', 'filter_btc': True, 'symbols': None, 'vol_pct': 0},
        {'name': 'High-spike symbols', 'filter_btc': False, 'symbols': HIGH_SPIKE_SYMBOLS, 'vol_pct': 0},
        {'name': 'High-spike + Vol Q3-Q4', 'filter_btc': False, 'symbols': HIGH_SPIKE_SYMBOLS, 'vol_pct': 50},
        {'name': 'High-spike + Vol Q4', 'filter_btc': False, 'symbols': HIGH_SPIKE_SYMBOLS, 'vol_pct': 75},
        {'name': 'No BTC + Vol Q3-Q4', 'filter_btc': True, 'symbols': None, 'vol_pct': 50},
        {'name': 'No BTC + Vol Q4', 'filter_btc': True, 'symbols': None, 'vol_pct': 75},
        {'name': 'All + Vol Q3-Q4', 'filter_btc': False, 'symbols': None, 'vol_pct': 50},
        {'name': 'All + Vol Q4', 'filter_btc': False, 'symbols': None, 'vol_pct': 75},
    ]

    all_results = {}

    for config in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {config['name']}")
        print('='*70)

        # Apply filters to training data
        train_filtered = train_df.copy()
        test_filtered = test_df.copy()

        # Symbol filter
        if config['filter_btc']:
            train_filtered = train_filtered[train_filtered['symbol'] != 'BTC-USD']
            test_filtered = test_filtered[test_filtered['symbol'] != 'BTC-USD']

        if config['symbols']:
            train_filtered = train_filtered[train_filtered['symbol'].isin(config['symbols'])]
            test_filtered = test_filtered[test_filtered['symbol'].isin(config['symbols'])]

        # Volatility filter (threshold computed on training data only)
        vol_threshold = None
        if config['vol_pct'] > 0:
            vol_threshold = np.percentile(train_df['volatility_4h'].dropna(), config['vol_pct'])
            train_filtered = train_filtered[train_filtered['volatility_4h'] >= vol_threshold]
            test_filtered = test_filtered[test_filtered['volatility_4h'] >= vol_threshold]
            print(f"Volatility threshold (P{config['vol_pct']}): {vol_threshold:.4f}")

        # Check sample sizes
        n_train = len(train_filtered)
        n_test = len(test_filtered)
        n_train_spikes = train_filtered['spike_10pct_24h'].sum()
        n_test_spikes = test_filtered['spike_10pct_24h'].sum()

        print(f"Train: {n_train} samples, {n_train_spikes} spikes ({n_train_spikes/n_train*100:.1f}%)")
        print(f"Test:  {n_test} samples, {n_test_spikes} spikes ({n_test_spikes/n_test*100:.1f}%)")

        if n_train < 50 or n_test < 10 or n_train_spikes < 5:
            print("Insufficient data, skipping...")
            continue

        # Prepare features
        X_train = prepare_features(train_filtered, orderbook_cols)
        y_train = train_filtered['spike_10pct_24h'].astype(int)
        X_test = prepare_features(test_filtered, orderbook_cols)
        y_test = test_filtered['spike_10pct_24h'].astype(int)

        # Train ensemble
        print("Training ensemble...")
        models = train_ensemble(X_train, y_train)

        # Get predictions
        y_proba = ensemble_predict(models, X_test)
        y_pred = (y_proba >= 0.3).astype(int)

        # Evaluate
        results = evaluate_predictions(y_test, y_proba, config['name'])
        results['n_train'] = n_train
        results['n_test'] = n_test
        results['n_train_spikes'] = int(n_train_spikes)
        results['n_test_spikes'] = int(n_test_spikes)
        results['train_spike_rate'] = n_train_spikes / n_train
        results['test_spike_rate'] = n_test_spikes / n_test

        all_results[config['name']] = results

        # Print threshold analysis
        print(f"\nAUC: {results['auc']:.3f}, PR-AUC: {results['pr_auc']:.3f}")
        print("\nPrecision-Recall at thresholds:")
        print(f"{'Thresh':>8} | {'Preds':>6} | {'Correct':>7} | {'Precision':>10} | {'Recall':>8}")
        print("-" * 55)
        for thresh, m in results['thresholds'].items():
            print(f"{thresh:>8.2f} | {m['n_pred']:>6} | {m['n_correct']:>7} | {m['precision']:>9.1%} | {m['recall']:>7.1%}")

        # Save predictions
        preds_df = test_filtered[['timestamp', 'symbol', 'volatility_4h', 'spike_10pct_24h']].copy()
        preds_df['proba'] = y_proba
        preds_df.to_csv(f'/Users/bz/Pythia2/predictions_{config["name"].replace(" ", "_").replace("+", "and")}.csv', index=False)

    # Final summary
    print("\n\n" + "=" * 120)
    print("FINAL SUMMARY")
    print("=" * 120)

    header = f"{'Configuration':<25} | {'Train':>6} | {'Test':>5} | {'Spike%':>7} | {'AUC':>6} | {'PR-AUC':>6} | {'Prec@0.3':>9} | {'Prec@0.2':>9} | {'Rec@0.2':>8}"
    print(header)
    print("-" * 120)

    for name, r in all_results.items():
        t3 = r['thresholds'].get(0.3, {})
        t2 = r['thresholds'].get(0.2, {})
        prec3 = t3.get('precision', 0)
        prec2 = t2.get('precision', 0)
        rec2 = t2.get('recall', 0)

        print(f"{name:<25} | {r['n_train']:>6} | {r['n_test']:>5} | {r['test_spike_rate']*100:>6.1f}% | "
              f"{r['auc']:>6.3f} | {r['pr_auc']:>6.3f} | {prec3:>8.1%} | {prec2:>8.1%} | {rec2:>7.1%}")

    # Best configs by different criteria
    print("\n" + "-" * 120)
    print("BEST CONFIGURATIONS:")
    print("-" * 120)

    # Best AUC
    best_auc = max(all_results.items(), key=lambda x: x[1]['auc'])
    print(f"Best AUC:    {best_auc[0]} (AUC={best_auc[1]['auc']:.3f})")

    # Best PR-AUC
    best_prauc = max(all_results.items(), key=lambda x: x[1]['pr_auc'])
    print(f"Best PR-AUC: {best_prauc[0]} (PR-AUC={best_prauc[1]['pr_auc']:.3f})")

    # Best precision at 0.2 threshold (with at least 10% recall)
    valid_for_prec = [(n, r) for n, r in all_results.items()
                     if r['thresholds'].get(0.2, {}).get('recall', 0) >= 0.1]
    if valid_for_prec:
        best_prec = max(valid_for_prec, key=lambda x: x[1]['thresholds'].get(0.2, {}).get('precision', 0))
        prec = best_prec[1]['thresholds'].get(0.2, {}).get('precision', 0)
        print(f"Best Prec@0.2 (rec>=10%): {best_prec[0]} (Precision={prec:.1%})")

    # Trading interpretation
    print("\n" + "=" * 120)
    print("TRADING INTERPRETATION")
    print("=" * 120)

    for name, r in all_results.items():
        t2 = r['thresholds'].get(0.2, {})
        t3 = r['thresholds'].get(0.3, {})
        if t2.get('n_pred', 0) > 0:
            base_rate = r['test_spike_rate']
            lift_2 = t2['precision'] / base_rate if base_rate > 0 else 0
            lift_3 = t3['precision'] / base_rate if base_rate > 0 and t3.get('n_pred', 0) > 0 else 0

            print(f"\n{name}:")
            print(f"  Base spike rate: {base_rate*100:.1f}%")
            print(f"  At threshold 0.2: {t2['n_pred']} predictions, {t2['precision']*100:.1f}% precision ({lift_2:.1f}x lift)")
            print(f"  At threshold 0.3: {t3['n_pred']} predictions, {t3['precision']*100:.1f}% precision ({lift_3:.1f}x lift)")

    return all_results


if __name__ == "__main__":
    results = run_backtest()

    # Save summary
    summary = {}
    for name, r in results.items():
        summary[name] = {
            'n_test': r['n_test'],
            'test_spike_rate': r['test_spike_rate'],
            'auc': r['auc'],
            'pr_auc': r['pr_auc'],
            'precision_0.2': r['thresholds'].get(0.2, {}).get('precision', 0),
            'recall_0.2': r['thresholds'].get(0.2, {}).get('recall', 0),
            'precision_0.3': r['thresholds'].get(0.3, {}).get('precision', 0),
            'recall_0.3': r['thresholds'].get(0.3, {}).get('recall', 0),
        }

    pd.DataFrame(summary).T.to_csv('/Users/bz/Pythia2/iteration3_summary.csv')
    print("\nSummary saved to iteration3_summary.csv")
