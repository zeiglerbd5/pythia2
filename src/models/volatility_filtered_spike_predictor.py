"""
Volatility-Filtered Spike Predictor

Key insight: Spikes occur almost exclusively during high-volatility periods.
- Q1 volatility: 0.15% spike rate
- Q4 volatility: 7.74% spike rate (50x higher)

This predictor filters to high-volatility regimes (Q3+Q4) before training,
dramatically improving precision by removing low-signal periods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
import json
warnings.filterwarnings('ignore')

import duckdb

# ML imports
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)


@dataclass
class FoldResult:
    """Result from one walk-forward fold."""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train: int
    n_test: int
    n_filtered_train: int
    n_filtered_test: int
    n_spikes_train: int
    n_spikes_test: int
    base_spike_rate: float
    filtered_spike_rate: float
    model_metrics: Dict[str, Dict[str, float]]
    ensemble_metrics: Dict[str, float]
    predictions: pd.DataFrame
    volatility_threshold: float


class OrderbookFeatureComputer:
    """Computes orderbook features from order_book_snapshots table."""

    def __init__(self, db_path: str = "full_pythia.duckdb"):
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
                    timestamp,
                    bids,
                    asks,
                    best_bid,
                    best_ask,
                    mid_price,
                    spread_bps
                FROM order_book_snapshots
                WHERE symbol = '{symbol}'
                  AND timestamp >= '{timestamp - timedelta(minutes=lookback_minutes)}'
                  AND timestamp <= '{timestamp}'
                ORDER BY timestamp DESC
                LIMIT 20
            """
            snapshots = self.conn.execute(query).df()

            if len(snapshots) == 0:
                return self._empty_orderbook_features()

            latest = snapshots.iloc[0]
            bids = json.loads(latest['bids']) if isinstance(latest['bids'], str) else latest['bids']
            asks = json.loads(latest['asks']) if isinstance(latest['asks'], str) else latest['asks']

            features = {}

            # Basic spread features
            features['spread_bps'] = latest['spread_bps']

            # Volume imbalance at multiple levels
            for n_levels in [3, 5, 10]:
                bid_sizes = [b[1] for b in bids[:n_levels]]
                ask_sizes = [a[1] for a in asks[:n_levels]]
                total_bid = sum(bid_sizes)
                total_ask = sum(ask_sizes)
                total = total_bid + total_ask
                features[f'imbalance_l{n_levels}'] = (total_bid - total_ask) / total if total > 0 else 0

            # Depth in USD
            bid_prices = [b[0] for b in bids[:10]]
            bid_sizes = [b[1] for b in bids[:10]]
            ask_prices = [a[0] for a in asks[:10]]
            ask_sizes = [a[1] for a in asks[:10]]

            bid_depth_usd = sum(p * s for p, s in zip(bid_prices, bid_sizes))
            ask_depth_usd = sum(p * s for p, s in zip(ask_prices, ask_sizes))
            features['depth_ratio'] = bid_depth_usd / ask_depth_usd if ask_depth_usd > 0 else 1.0
            features['log_depth'] = np.log10(max(bid_depth_usd + ask_depth_usd, 1))

            # Large order detection
            if len(bid_sizes) > 0 and len(ask_sizes) > 0:
                all_sizes = bid_sizes + ask_sizes
                p90_size = np.percentile(all_sizes, 90)
                large_bids = sum(1 for s in bid_sizes if s > p90_size)
                large_asks = sum(1 for s in ask_sizes if s > p90_size)
                features['large_order_imbalance'] = large_bids - large_asks

            # Spread dynamics
            if len(snapshots) > 1:
                spreads = snapshots['spread_bps'].values
                features['spread_volatility'] = np.std(spreads) if len(spreads) > 1 else 0
                features['spread_trend'] = spreads[0] - spreads[-1]
            else:
                features['spread_volatility'] = 0
                features['spread_trend'] = 0

            self._cache[cache_key] = features
            return features

        except Exception as e:
            return self._empty_orderbook_features()

    def _empty_orderbook_features(self) -> Dict[str, float]:
        """Return empty orderbook features."""
        return {
            'spread_bps': None,
            'imbalance_l3': None,
            'imbalance_l5': None,
            'imbalance_l10': None,
            'depth_ratio': None,
            'log_depth': None,
            'large_order_imbalance': None,
            'spread_volatility': None,
            'spread_trend': None,
        }

    def batch_compute_features(
        self,
        signals_df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        symbol_col: str = 'symbol'
    ) -> pd.DataFrame:
        """Compute orderbook features for all signals."""
        print(f"Computing orderbook features for {len(signals_df)} signals...")

        features_list = []
        for idx, row in signals_df.iterrows():
            feat = self.compute_features_at_time(row[symbol_col], row[timestamp_col])
            features_list.append(feat)

            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(signals_df)} signals")

        return pd.DataFrame(features_list)


class VolatilityFilteredSpikePredictor:
    """
    Spike predictor that filters to high-volatility regimes.

    Key insight: Q4 volatility has 7.74% spike rate vs 0.15% for Q1.
    By filtering to high-vol periods, we:
    1. Increase base rate from 3.1% to ~7-8%
    2. Remove noise from low-signal periods
    3. Dramatically improve precision
    """

    CATALYST_FEATURES = [
        'event_priority', 'sentiment_score', 'log_usd_value',
        'volatility_4h', 'momentum_4h', 'volume_ratio', 'rsi_proxy',
    ]

    ORDERBOOK_FEATURES = [
        'spread_bps', 'imbalance_l3', 'imbalance_l5', 'imbalance_l10',
        'depth_ratio', 'log_depth', 'large_order_imbalance',
        'spread_volatility', 'spread_trend',
    ]

    BOOL_FEATURES = ['is_bearish_flow', 'is_bullish_flow', 'has_direction']

    def __init__(
        self,
        db_path: str = "full_pythia.duckdb",
        use_orderbook: bool = True,
        volatility_percentile_threshold: float = 50.0,  # Filter to top 50% volatility
    ):
        self.db_path = db_path
        self.use_orderbook = use_orderbook
        self.volatility_percentile_threshold = volatility_percentile_threshold

        self.models = {}
        self.model_weights = {}
        self.feature_importance = {}
        self.volatility_threshold = None  # Computed from training data

        if use_orderbook:
            self.ob_computer = OrderbookFeatureComputer(db_path)

    def load_and_prepare_data(self, whale_csv: str = "whale_features.csv") -> pd.DataFrame:
        """Load and prepare data with orderbook features."""
        df = pd.read_csv(whale_csv, parse_dates=['timestamp'])
        print(f"Loaded {len(df)} whale signals")

        if self.use_orderbook:
            ob_features = self.ob_computer.batch_compute_features(df)
            df = pd.concat([df.reset_index(drop=True), ob_features], axis=1)
            print(f"Added {len(self.ORDERBOOK_FEATURES)} orderbook features")

        return df

    def filter_to_high_volatility(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        volatility_col: str = 'volatility_4h'
    ) -> pd.DataFrame:
        """
        Filter dataframe to high-volatility periods.

        During training (fit=True), compute the threshold from data.
        During testing (fit=False), use the stored threshold.
        """
        if fit:
            # Compute threshold from training data
            self.volatility_threshold = np.percentile(
                df[volatility_col].dropna(),
                self.volatility_percentile_threshold
            )
            print(f"Volatility threshold (P{self.volatility_percentile_threshold}): {self.volatility_threshold:.4f}")

        # Filter
        filtered = df[df[volatility_col] >= self.volatility_threshold].copy()

        return filtered

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'spike_10pct_24h',
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix."""
        data = df.copy()

        feature_cols = self.CATALYST_FEATURES.copy()

        if self.use_orderbook:
            feature_cols.extend(self.ORDERBOOK_FEATURES)

        for col in self.BOOL_FEATURES:
            if col in data.columns:
                feature_cols.append(col)
                data[col] = data[col].astype(int)

        available_features = [f for f in feature_cols if f in data.columns]
        X = data[available_features].copy()
        y = data[target_col].astype(int)

        # Fill NaN with median
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X[col] = X[col].fillna(median_val)

        return X, y

    def _get_models(self, n_pos: int, n_neg: int) -> Dict[str, Any]:
        """Get dictionary of models to train."""
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
        models = {}

        if HAS_LIGHTGBM:
            models['lightgbm'] = lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
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
                use_label_encoder=False,
                verbosity=0,
                random_state=42,
            )

        models['randomforest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight={0: 1, 1: scale_pos_weight},
            random_state=42,
            n_jobs=-1,
        )

        models['gradboost'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

        return models

    def train_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Train all base models."""
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos

        print(f"Training ensemble on {len(X_train)} samples ({n_pos} positive, {n_pos/len(X_train)*100:.1f}%)")

        self.models = self._get_models(n_pos, n_neg)
        results = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            results[name] = {'trained': True}

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_,
                }).sort_values('importance', ascending=False)

        # Compute model weights based on validation performance
        if X_val is not None and y_val is not None:
            self._compute_model_weights(X_val, y_val)
        else:
            self.model_weights = {name: 1.0 / len(self.models) for name in self.models}

        return results

    def _compute_model_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Compute model weights based on validation AUC."""
        aucs = {}
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_val)[:, 1]
                if y_val.sum() > 0:
                    auc = roc_auc_score(y_val, proba)
                else:
                    auc = 0.5
                aucs[name] = auc
            except Exception as e:
                aucs[name] = 0.5

        # Normalize weights
        adjusted = {k: max(v - 0.5, 0.01) for k, v in aucs.items()}
        total = sum(adjusted.values())
        self.model_weights = {k: v / total for k, v in adjusted.items()}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble probability predictions."""
        predictions = []
        weights = []

        for name, model in self.models.items():
            proba = model.predict_proba(X)[:, 1]
            predictions.append(proba)
            weights.append(self.model_weights.get(name, 1.0))

        predictions = np.array(predictions)
        weights = np.array(weights)

        # Weighted average
        ensemble_proba = np.average(predictions, axis=0, weights=weights)
        return ensemble_proba

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'n_predictions': int(y_pred.sum()),
            'n_correct': int((y_pred & y_true).sum()),
        }

        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc'] = 0.5

        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics['pr_auc'] = 0.0

        return metrics

    def walk_forward_validation(
        self,
        df: pd.DataFrame,
        train_days: int = 21,
        test_days: int = 7,
        target_col: str = 'spike_10pct_24h',
    ) -> List[FoldResult]:
        """Walk-forward validation with volatility filtering."""
        print(f"\n{'='*70}")
        print(f"VOLATILITY-FILTERED WALK-FORWARD VALIDATION")
        print(f"Filter: Top {100-self.volatility_percentile_threshold:.0f}% volatility")
        print(f"Train window: {train_days} days, Test window: {test_days} days")
        print(f"{'='*70}")

        df = df.sort_values('timestamp').reset_index(drop=True)
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()

        results = []
        fold = 0
        train_start = min_date

        while True:
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)

            if test_end > max_date:
                break

            # Get raw data for this fold
            train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] < train_end)
            test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)

            train_df_raw = df[train_mask]
            test_df_raw = df[test_mask]

            if len(train_df_raw) < 30 or len(test_df_raw) < 5:
                train_start = test_start
                continue

            fold += 1

            # Filter to high volatility
            train_df = self.filter_to_high_volatility(train_df_raw, fit=True)
            test_df = self.filter_to_high_volatility(test_df_raw, fit=False)

            if len(train_df) < 20 or len(test_df) < 3:
                train_start = test_start
                continue

            # Prepare features
            X_train, y_train = self.prepare_features(train_df, target_col)
            X_test, y_test = self.prepare_features(test_df, target_col)

            # Calculate metrics
            base_spike_rate = train_df_raw[target_col].mean()
            filtered_spike_rate = train_df[target_col].mean()

            print(f"\n--- Fold {fold} ---")
            print(f"Period: {train_start.date()} to {test_end.date()}")
            print(f"Raw samples: {len(train_df_raw)} train, {len(test_df_raw)} test")
            print(f"Filtered:    {len(train_df)} train ({len(train_df)/len(train_df_raw)*100:.0f}%), "
                  f"{len(test_df)} test ({len(test_df)/len(test_df_raw)*100:.0f}%)")
            print(f"Spike rate:  {base_spike_rate*100:.1f}% raw -> {filtered_spike_rate*100:.1f}% filtered")
            print(f"Train spikes: {y_train.sum()}, Test spikes: {y_test.sum()}")

            # Use last 20% of training for validation
            val_split = int(len(X_train) * 0.8)
            X_tr = X_train.iloc[:val_split]
            y_tr = y_train.iloc[:val_split]
            X_val = X_train.iloc[val_split:]
            y_val = y_train.iloc[val_split:]

            # Train ensemble
            self.train_ensemble(X_tr, y_tr, X_val, y_val)

            # Get predictions
            model_metrics = {}
            model_preds = {}

            for name, model in self.models.items():
                proba = model.predict_proba(X_test)[:, 1]
                pred = (proba >= 0.5).astype(int)
                model_metrics[name] = self._compute_metrics(y_test, pred, proba)
                model_preds[name] = proba

            # Ensemble predictions
            ensemble_proba = self.predict_proba(X_test)
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
            ensemble_metrics = self._compute_metrics(y_test, ensemble_pred, ensemble_proba)

            # Log results
            print(f"\n  Model Results:")
            for name, metrics in model_metrics.items():
                print(f"    {name:15}: AUC={metrics['auc']:.3f}, "
                      f"Prec={metrics['precision']:.1%}, Rec={metrics['recall']:.1%}")
            print(f"    {'ENSEMBLE':15}: AUC={ensemble_metrics['auc']:.3f}, "
                  f"Prec={ensemble_metrics['precision']:.1%}, Rec={ensemble_metrics['recall']:.1%}")

            # Store predictions
            predictions = test_df[['timestamp', 'symbol', 'volatility_4h']].copy()
            predictions['y_true'] = y_test.values
            predictions['ensemble_proba'] = ensemble_proba
            predictions['ensemble_pred'] = ensemble_pred
            for name, proba in model_preds.items():
                predictions[f'{name}_proba'] = proba

            result = FoldResult(
                fold=fold,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train=len(train_df_raw),
                n_test=len(test_df_raw),
                n_filtered_train=len(train_df),
                n_filtered_test=len(test_df),
                n_spikes_train=int(y_train.sum()),
                n_spikes_test=int(y_test.sum()),
                base_spike_rate=base_spike_rate,
                filtered_spike_rate=filtered_spike_rate,
                model_metrics=model_metrics,
                ensemble_metrics=ensemble_metrics,
                predictions=predictions,
                volatility_threshold=self.volatility_threshold,
            )
            results.append(result)

            train_start = test_start

        return results

    def print_summary(self, results: List[FoldResult]) -> Dict:
        """Print comprehensive summary."""
        print("\n" + "=" * 80)
        print("VOLATILITY-FILTERED SPIKE PREDICTOR - SUMMARY")
        print("=" * 80)

        # Overall stats
        total_raw = sum(r.n_test for r in results)
        total_filtered = sum(r.n_filtered_test for r in results)
        total_spikes = sum(r.n_spikes_test for r in results)

        print(f"\nFiltering: Top {100-self.volatility_percentile_threshold:.0f}% volatility")
        print(f"Total folds: {len(results)}")
        print(f"Raw test samples: {total_raw}")
        print(f"Filtered test samples: {total_filtered} ({total_filtered/total_raw*100:.1f}%)")
        print(f"Test spikes: {total_spikes} ({total_spikes/total_filtered*100:.1f}% rate in filtered)")

        # Model comparison
        model_names = list(results[0].model_metrics.keys())

        print("\n" + "-" * 70)
        print("AVERAGE METRICS BY MODEL")
        print("-" * 70)
        print(f"{'Model':15} | {'AUC':>7} | {'PR-AUC':>7} | {'Precision':>9} | {'Recall':>7} | {'F1':>7}")
        print("-" * 70)

        summary = {}
        for name in model_names:
            aucs = [r.model_metrics[name]['auc'] for r in results]
            pr_aucs = [r.model_metrics[name].get('pr_auc', 0) for r in results]
            precs = [r.model_metrics[name]['precision'] for r in results]
            recs = [r.model_metrics[name]['recall'] for r in results]
            f1s = [r.model_metrics[name]['f1'] for r in results]

            summary[name] = {
                'auc': np.mean(aucs),
                'pr_auc': np.mean(pr_aucs),
                'precision': np.mean(precs),
                'recall': np.mean(recs),
                'f1': np.mean(f1s),
            }
            print(f"{name:15} | {np.mean(aucs):>7.3f} | {np.mean(pr_aucs):>7.3f} | "
                  f"{np.mean(precs):>9.1%} | {np.mean(recs):>7.1%} | {np.mean(f1s):>7.3f}")

        # Ensemble
        ens_aucs = [r.ensemble_metrics['auc'] for r in results]
        ens_pr_aucs = [r.ensemble_metrics.get('pr_auc', 0) for r in results]
        ens_precs = [r.ensemble_metrics['precision'] for r in results]
        ens_recs = [r.ensemble_metrics['recall'] for r in results]
        ens_f1s = [r.ensemble_metrics['f1'] for r in results]

        print("-" * 70)
        print(f"{'ENSEMBLE':15} | {np.mean(ens_aucs):>7.3f} | {np.mean(ens_pr_aucs):>7.3f} | "
              f"{np.mean(ens_precs):>9.1%} | {np.mean(ens_recs):>7.1%} | {np.mean(ens_f1s):>7.3f}")

        summary['ensemble'] = {
            'auc': np.mean(ens_aucs),
            'pr_auc': np.mean(ens_pr_aucs),
            'precision': np.mean(ens_precs),
            'recall': np.mean(ens_recs),
            'f1': np.mean(ens_f1s),
        }

        # Precision at thresholds
        all_preds = pd.concat([r.predictions for r in results])
        y_true = all_preds['y_true']
        ensemble_proba = all_preds['ensemble_proba']

        print("\n" + "-" * 70)
        print("PRECISION AT THRESHOLDS (Ensemble)")
        print("-" * 70)

        for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            pred = (ensemble_proba >= thresh).astype(int)
            n_pred = pred.sum()
            if n_pred > 0:
                n_correct = (pred & y_true).sum()
                prec = n_correct / n_pred
                rec = n_correct / y_true.sum() if y_true.sum() > 0 else 0
                print(f"  Threshold {thresh:.1f}: {n_pred:>4} predictions, "
                      f"{n_correct:>3} correct -> {prec:.1%} precision, {rec:.1%} recall")

        # Feature importance
        if self.feature_importance:
            print("\n" + "-" * 70)
            print("TOP FEATURES")
            print("-" * 70)

            all_importance = []
            for name, imp_df in self.feature_importance.items():
                all_importance.append(imp_df.set_index('feature')['importance'])

            combined = pd.concat(all_importance, axis=1)
            avg_importance = combined.mean(axis=1).sort_values(ascending=False)

            for feat, imp in avg_importance.head(12).items():
                feat_type = "OB" if feat in self.ORDERBOOK_FEATURES else "CT"
                print(f"  [{feat_type}] {feat:30}: {imp:>8.1f}")

        return summary


def run_comparison():
    """Compare baseline vs volatility-filtered predictor."""
    print("=" * 80)
    print("COMPARISON: BASELINE vs VOLATILITY-FILTERED")
    print("=" * 80)

    configs = [
        {'volatility_threshold': 0, 'name': 'Baseline (no filter)'},
        {'volatility_threshold': 50, 'name': 'Top 50% Volatility'},
        {'volatility_threshold': 75, 'name': 'Top 25% Volatility (Q4)'},
    ]

    all_summaries = {}

    for config in configs:
        print(f"\n\n{'#' * 80}")
        print(f"# CONFIGURATION: {config['name']}")
        print(f"{'#' * 80}")

        predictor = VolatilityFilteredSpikePredictor(
            db_path="/Users/bz/Pythia2/full_pythia.duckdb",
            use_orderbook=True,
            volatility_percentile_threshold=config['volatility_threshold'],
        )

        df = predictor.load_and_prepare_data("/Users/bz/Pythia2/whale_features.csv")

        results = predictor.walk_forward_validation(
            df,
            train_days=21,
            test_days=7,
            target_col='spike_10pct_24h',
        )

        if results:
            summary = predictor.print_summary(results)
            all_summaries[config['name']] = {
                'summary': summary,
                'n_samples': sum(r.n_filtered_test for r in results),
                'n_spikes': sum(r.n_spikes_test for r in results),
            }

            # Save predictions
            all_preds = pd.concat([r.predictions for r in results])
            suffix = f"vol_p{config['volatility_threshold']}"
            pred_file = f"/Users/bz/Pythia2/predictions_{suffix}.csv"
            all_preds.to_csv(pred_file, index=False)
            print(f"\nPredictions saved to: {pred_file}")

    # Final comparison table
    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"{'Configuration':30} | {'Samples':>7} | {'Spikes':>6} | {'Rate':>6} | {'AUC':>6} | {'Prec':>6} | {'Recall':>6}")
    print("-" * 80)

    for name, data in all_summaries.items():
        n_samples = data['n_samples']
        n_spikes = data['n_spikes']
        rate = n_spikes / n_samples * 100 if n_samples > 0 else 0
        ens = data['summary'].get('ensemble', {})
        print(f"{name:30} | {n_samples:>7} | {n_spikes:>6} | {rate:>5.1f}% | "
              f"{ens.get('auc', 0):>6.3f} | {ens.get('precision', 0):>5.1%} | {ens.get('recall', 0):>5.1%}")


def main():
    """Run volatility-filtered spike predictor."""
    run_comparison()


if __name__ == "__main__":
    main()
