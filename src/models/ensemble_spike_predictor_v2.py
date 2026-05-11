"""
Ensemble Spike Predictor V2 with Enhanced Features and Analysis

Improvements over V1:
1. Shorter train/test windows for more folds
2. Threshold optimization per model
3. Feature interaction terms
4. Dynamic threshold selection based on precision targets
5. More detailed analysis of where orderbook helps
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger
import warnings
import json
warnings.filterwarnings('ignore')

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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, classification_report,
    confusion_matrix, precision_recall_curve
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
    n_spikes_train: int
    n_spikes_test: int
    model_aucs: Dict[str, float]
    ensemble_auc: float
    predictions: pd.DataFrame
    optimal_thresholds: Dict[str, float]
    precision_at_thresholds: Dict[str, Dict[float, float]]


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
            features['bid_ask_spread_pct'] = (latest['best_ask'] - latest['best_bid']) / latest['mid_price'] * 100 if latest['mid_price'] > 0 else 0

            # Volume imbalance at multiple levels
            for n_levels in [3, 5, 10]:
                bid_sizes = [b[1] for b in bids[:n_levels]]
                ask_sizes = [a[1] for a in asks[:n_levels]]
                total_bid = sum(bid_sizes)
                total_ask = sum(ask_sizes)
                total = total_bid + total_ask
                features[f'imbalance_l{n_levels}'] = (total_bid - total_ask) / total if total > 0 else 0
                features[f'bid_ask_ratio_l{n_levels}'] = total_bid / total_ask if total_ask > 0 else 1.0

            # Depth in USD
            bid_prices = [b[0] for b in bids[:10]]
            bid_sizes = [b[1] for b in bids[:10]]
            ask_prices = [a[0] for a in asks[:10]]
            ask_sizes = [a[1] for a in asks[:10]]

            bid_depth_usd = sum(p * s for p, s in zip(bid_prices, bid_sizes))
            ask_depth_usd = sum(p * s for p, s in zip(ask_prices, ask_sizes))
            features['depth_ratio'] = bid_depth_usd / ask_depth_usd if ask_depth_usd > 0 else 1.0
            features['total_depth_usd'] = bid_depth_usd + ask_depth_usd
            features['log_depth'] = np.log10(max(features['total_depth_usd'], 1))

            # Large order detection
            if len(bid_sizes) > 0 and len(ask_sizes) > 0:
                all_sizes = bid_sizes + ask_sizes
                median_size = np.median(all_sizes)
                p90_size = np.percentile(all_sizes, 90)

                large_bids = sum(1 for s in bid_sizes if s > p90_size)
                large_asks = sum(1 for s in ask_sizes if s > p90_size)
                features['large_bid_count'] = large_bids
                features['large_ask_count'] = large_asks
                features['large_order_imbalance'] = large_bids - large_asks

                # Volume-weighted large order imbalance
                large_bid_vol = sum(s for s in bid_sizes if s > median_size * 2)
                large_ask_vol = sum(s for s in ask_sizes if s > median_size * 2)
                features['large_vol_imbalance'] = (large_bid_vol - large_ask_vol) / (large_bid_vol + large_ask_vol + 1)

            # Weighted mid price deviation
            total_bid_vol = sum(bid_sizes[:5])
            total_ask_vol = sum(ask_sizes[:5])
            if total_bid_vol > 0 and total_ask_vol > 0:
                weighted_bid = sum(p * s for p, s in zip(bid_prices[:5], bid_sizes[:5])) / total_bid_vol
                weighted_ask = sum(p * s for p, s in zip(ask_prices[:5], ask_sizes[:5])) / total_ask_vol
                weighted_mid = (weighted_bid + weighted_ask) / 2
                features['mid_deviation'] = (latest['mid_price'] - weighted_mid) / latest['mid_price'] * 100
            else:
                features['mid_deviation'] = 0

            # Spread dynamics over time
            if len(snapshots) > 1:
                spreads = snapshots['spread_bps'].values
                features['spread_volatility'] = np.std(spreads) if len(spreads) > 1 else 0
                features['spread_trend'] = spreads[0] - spreads[-1]
                features['spread_range'] = spreads.max() - spreads.min()
            else:
                features['spread_volatility'] = 0
                features['spread_trend'] = 0
                features['spread_range'] = 0

            # Price pressure indicator
            # If bids are clustering near best bid, expect upward pressure
            if len(bid_prices) >= 3:
                bid_spread = (bid_prices[0] - bid_prices[2]) / bid_prices[0] * 100
                features['bid_clustering'] = 1 / (bid_spread + 0.01)
            else:
                features['bid_clustering'] = 1

            self._cache[cache_key] = features
            return features

        except Exception as e:
            logger.debug(f"Orderbook feature error for {symbol}: {e}")
            return self._empty_orderbook_features()

    def _empty_orderbook_features(self) -> Dict[str, float]:
        """Return empty orderbook features."""
        return {
            'spread_bps': None,
            'bid_ask_spread_pct': None,
            'imbalance_l3': None,
            'imbalance_l5': None,
            'imbalance_l10': None,
            'bid_ask_ratio_l3': None,
            'bid_ask_ratio_l5': None,
            'bid_ask_ratio_l10': None,
            'depth_ratio': None,
            'total_depth_usd': None,
            'log_depth': None,
            'large_bid_count': None,
            'large_ask_count': None,
            'large_order_imbalance': None,
            'large_vol_imbalance': None,
            'mid_deviation': None,
            'spread_volatility': None,
            'spread_trend': None,
            'spread_range': None,
            'bid_clustering': None,
        }

    def batch_compute_features(
        self,
        signals_df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        symbol_col: str = 'symbol'
    ) -> pd.DataFrame:
        """Compute orderbook features for all signals."""
        logger.info(f"Computing orderbook features for {len(signals_df)} signals...")

        features_list = []
        for idx, row in signals_df.iterrows():
            feat = self.compute_features_at_time(row[symbol_col], row[timestamp_col])
            features_list.append(feat)

            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{len(signals_df)} signals")

        features_df = pd.DataFrame(features_list)
        logger.info(f"Computed {len(features_df)} orderbook feature rows")

        return features_df


class EnsembleSpikePredictor:
    """Enhanced ensemble spike predictor."""

    CATALYST_FEATURES = [
        'event_priority', 'sentiment_score', 'log_usd_value',
        'volatility_4h', 'momentum_4h', 'volume_ratio', 'rsi_proxy',
    ]

    ORDERBOOK_FEATURES = [
        'spread_bps', 'bid_ask_spread_pct',
        'imbalance_l3', 'imbalance_l5', 'imbalance_l10',
        'bid_ask_ratio_l3', 'bid_ask_ratio_l5',
        'depth_ratio', 'log_depth',
        'large_bid_count', 'large_ask_count', 'large_order_imbalance',
        'large_vol_imbalance', 'mid_deviation',
        'spread_volatility', 'spread_trend', 'spread_range',
        'bid_clustering',
    ]

    BOOL_FEATURES = ['is_bearish_flow', 'is_bullish_flow', 'has_direction']

    def __init__(
        self,
        db_path: str = "full_pythia.duckdb",
        use_orderbook: bool = True,
        target_precision: float = 0.10,  # Target 10% precision minimum
    ):
        self.db_path = db_path
        self.use_orderbook = use_orderbook
        self.target_precision = target_precision
        self.models = {}
        self.optimal_thresholds = {}
        self.feature_importance = {}

        if use_orderbook:
            self.ob_computer = OrderbookFeatureComputer(db_path)

    def load_and_prepare_data(self, whale_csv: str = "whale_features.csv") -> pd.DataFrame:
        """Load and prepare data with orderbook features."""
        df = pd.read_csv(whale_csv, parse_dates=['timestamp'])
        logger.info(f"Loaded {len(df)} whale signals")

        if self.use_orderbook:
            ob_features = self.ob_computer.batch_compute_features(df)
            df = pd.concat([df.reset_index(drop=True), ob_features], axis=1)
            logger.info(f"Added {len(self.ORDERBOOK_FEATURES)} orderbook features")

        # Create interaction features
        df = self._create_interaction_features(df)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between catalyst and orderbook signals."""
        # Volatility * imbalance interaction
        if 'volatility_4h' in df.columns and 'imbalance_l5' in df.columns:
            df['vol_imbalance_interact'] = df['volatility_4h'] * df['imbalance_l5'].fillna(0)

        # Volume ratio * depth interaction
        if 'volume_ratio' in df.columns and 'log_depth' in df.columns:
            df['vol_depth_interact'] = df['volume_ratio'] * df['log_depth'].fillna(0)

        # Direction-specific imbalance
        if 'is_bullish_flow' in df.columns and 'imbalance_l5' in df.columns:
            df['bullish_imbalance'] = df['is_bullish_flow'].astype(float) * df['imbalance_l5'].fillna(0)
            df['bearish_imbalance'] = df['is_bearish_flow'].astype(float) * df['imbalance_l5'].fillna(0)

        return df

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
            # Add interaction features
            interaction_cols = [
                'vol_imbalance_interact', 'vol_depth_interact',
                'bullish_imbalance', 'bearish_imbalance'
            ]
            for col in interaction_cols:
                if col in data.columns:
                    feature_cols.append(col)

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

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict:
        """Train models and evaluate with optimal thresholds."""
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos

        self.models = self._get_models(n_pos, n_neg)

        results = {
            'model_metrics': {},
            'predictions': {},
            'optimal_thresholds': {},
        }

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)

            # Get predictions
            proba = model.predict_proba(X_test)[:, 1]
            results['predictions'][name] = proba

            # Find optimal threshold for target precision
            optimal_thresh = self._find_optimal_threshold(y_test, proba)
            results['optimal_thresholds'][name] = optimal_thresh

            # Compute metrics at optimal threshold
            pred = (proba >= optimal_thresh).astype(int)
            results['model_metrics'][name] = {
                'auc': roc_auc_score(y_test, proba) if y_test.sum() > 0 else 0.5,
                'pr_auc': average_precision_score(y_test, proba) if y_test.sum() > 0 else 0.0,
                'precision': precision_score(y_test, pred, zero_division=0),
                'recall': recall_score(y_test, pred, zero_division=0),
                'f1': f1_score(y_test, pred, zero_division=0),
                'n_predictions': pred.sum(),
                'optimal_threshold': optimal_thresh,
            }

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_,
                }).sort_values('importance', ascending=False)

        # Ensemble prediction (weighted by AUC)
        weights = {k: max(v['auc'] - 0.5, 0.01) for k, v in results['model_metrics'].items()}
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        ensemble_proba = np.zeros(len(X_test))
        for name, proba in results['predictions'].items():
            ensemble_proba += proba * weights[name]

        results['predictions']['ensemble'] = ensemble_proba
        optimal_thresh = self._find_optimal_threshold(y_test, ensemble_proba)
        results['optimal_thresholds']['ensemble'] = optimal_thresh

        pred = (ensemble_proba >= optimal_thresh).astype(int)
        results['model_metrics']['ensemble'] = {
            'auc': roc_auc_score(y_test, ensemble_proba) if y_test.sum() > 0 else 0.5,
            'pr_auc': average_precision_score(y_test, ensemble_proba) if y_test.sum() > 0 else 0.0,
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0),
            'n_predictions': pred.sum(),
            'optimal_threshold': optimal_thresh,
        }

        return results

    def _find_optimal_threshold(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
    ) -> float:
        """Find threshold that maximizes F1 while meeting precision target."""
        if y_true.sum() == 0:
            return 0.5

        best_threshold = 0.5
        best_f1 = 0

        for thresh in np.arange(0.1, 0.9, 0.05):
            pred = (y_proba >= thresh).astype(int)
            if pred.sum() == 0:
                continue

            prec = precision_score(y_true, pred, zero_division=0)
            rec = recall_score(y_true, pred, zero_division=0)
            f1 = f1_score(y_true, pred, zero_division=0)

            # Prioritize F1 but penalize if precision is too low
            if prec >= self.target_precision and f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
            elif prec >= self.target_precision * 0.5 and f1 > best_f1 * 1.2:
                # Accept lower precision if F1 is much better
                best_f1 = f1
                best_threshold = thresh

        return best_threshold

    def walk_forward_validation(
        self,
        df: pd.DataFrame,
        train_days: int = 21,  # 3 weeks
        test_days: int = 7,    # 1 week
        target_col: str = 'spike_10pct_24h',
    ) -> List[FoldResult]:
        """Walk-forward validation with shorter windows for more folds."""
        logger.info(f"Walk-forward: {train_days}d train, {test_days}d test")

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

            train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] < train_end)
            test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)

            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(train_df) < 30 or len(test_df) < 5:
                train_start = test_start
                continue

            fold += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold}: Train {train_start.date()} to {train_end.date()}")
            logger.info(f"         Test {test_start.date()} to {test_end.date()}")

            X_train, y_train = self.prepare_features(train_df, target_col)
            X_test, y_test = self.prepare_features(test_df, target_col)

            logger.info(f"Train: {len(X_train)} samples ({y_train.sum()} spikes, {y_train.mean()*100:.1f}%)")
            logger.info(f"Test:  {len(X_test)} samples ({y_test.sum()} spikes, {y_test.mean()*100:.1f}%)")

            eval_results = self.train_and_evaluate(X_train, y_train, X_test, y_test)

            # Log results
            logger.info("\n--- Results ---")
            for name, metrics in eval_results['model_metrics'].items():
                logger.info(f"{name:15}: AUC={metrics['auc']:.3f}, "
                           f"Prec={metrics['precision']:.3f}, "
                           f"Rec={metrics['recall']:.3f}, "
                           f"F1={metrics['f1']:.3f}, "
                           f"thresh={metrics['optimal_threshold']:.2f}")

            # Store predictions
            predictions = test_df[['timestamp', 'symbol']].copy()
            predictions['y_true'] = y_test.values
            for name, proba in eval_results['predictions'].items():
                predictions[f'{name}_proba'] = proba

            # Precision at various thresholds
            precision_at_thresholds = {}
            for name, proba in eval_results['predictions'].items():
                precision_at_thresholds[name] = {}
                for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    pred = (proba >= thresh).astype(int)
                    if pred.sum() > 0:
                        prec = precision_score(y_test, pred, zero_division=0)
                        precision_at_thresholds[name][thresh] = prec

            result = FoldResult(
                fold=fold,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train=len(train_df),
                n_test=len(test_df),
                n_spikes_train=int(y_train.sum()),
                n_spikes_test=int(y_test.sum()),
                model_aucs={k: v['auc'] for k, v in eval_results['model_metrics'].items()},
                ensemble_auc=eval_results['model_metrics']['ensemble']['auc'],
                predictions=predictions,
                optimal_thresholds=eval_results['optimal_thresholds'],
                precision_at_thresholds=precision_at_thresholds,
            )
            results.append(result)

            train_start = test_start

        return results

    def print_summary(self, results: List[FoldResult]) -> Dict:
        """Print detailed summary."""
        print("\n" + "=" * 80)
        print("ENSEMBLE SPIKE PREDICTOR V2 - RESULTS")
        print("=" * 80)

        print(f"\nTotal folds: {len(results)}")
        print(f"Total test samples: {sum(r.n_test for r in results)}")
        print(f"Total test spikes: {sum(r.n_spikes_test for r in results)}")

        # Average AUC by model
        model_names = list(results[0].model_aucs.keys())

        print("\n" + "-" * 60)
        print("AVERAGE AUC BY MODEL")
        print("-" * 60)

        summary = {}
        for name in model_names:
            aucs = [r.model_aucs[name] for r in results]
            avg_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            summary[name] = {'avg_auc': avg_auc, 'std_auc': std_auc}
            marker = " ***" if name == 'ensemble' else ""
            print(f"{name:15}: {avg_auc:.3f} +/- {std_auc:.3f}{marker}")

        # Per-fold breakdown
        print("\n" + "-" * 60)
        print("PER-FOLD ENSEMBLE RESULTS")
        print("-" * 60)
        print(f"{'Fold':>4} | {'Test Period':^21} | {'N':>5} | {'Spk':>4} | {'AUC':>6}")
        print("-" * 60)

        for r in results:
            print(f"{r.fold:>4} | {r.test_start.date()} to {r.test_end.date()} | "
                  f"{r.n_test:>5} | {r.n_spikes_test:>4} | {r.ensemble_auc:>6.3f}")

        # Feature importance
        if self.feature_importance:
            print("\n" + "-" * 60)
            print("TOP FEATURES (Average across models)")
            print("-" * 60)

            all_importance = []
            for name, imp_df in self.feature_importance.items():
                all_importance.append(imp_df.set_index('feature')['importance'])

            combined = pd.concat(all_importance, axis=1)
            avg_importance = combined.mean(axis=1).sort_values(ascending=False)

            for feat, imp in avg_importance.head(15).items():
                feat_type = "OB" if feat in self.ORDERBOOK_FEATURES else "CT"
                print(f"  [{feat_type}] {feat:35}: {imp:>8.1f}")

        # Analyze precision at thresholds across all folds
        print("\n" + "-" * 60)
        print("PRECISION AT THRESHOLDS (Ensemble, all folds combined)")
        print("-" * 60)

        all_preds = pd.concat([r.predictions for r in results])
        y_true = all_preds['y_true']
        ensemble_proba = all_preds['ensemble_proba']

        for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            pred = (ensemble_proba >= thresh).astype(int)
            n_pred = pred.sum()
            if n_pred > 0:
                n_correct = (pred & y_true).sum()
                prec = n_correct / n_pred
                rec = n_correct / y_true.sum() if y_true.sum() > 0 else 0
                print(f"  Threshold {thresh:.1f}: {n_pred:>4} predictions, "
                      f"{n_correct:>3} correct ({prec:.1%} precision, {rec:.1%} recall)")

        return summary


def main():
    """Run enhanced ensemble predictor."""
    print("=" * 80)
    print("ENSEMBLE SPIKE PREDICTOR V2 - ENHANCED FEATURES")
    print("=" * 80)

    # Compare baseline vs orderbook
    configs = [
        {'use_orderbook': False, 'name': 'Baseline (no orderbook)'},
        {'use_orderbook': True, 'name': 'With Orderbook Features'},
    ]

    all_summaries = {}

    for config in configs:
        print(f"\n\n{'#' * 80}")
        print(f"# CONFIGURATION: {config['name']}")
        print(f"{'#' * 80}")

        predictor = EnsembleSpikePredictor(
            db_path="/Users/bz/Pythia2/full_pythia.duckdb",
            use_orderbook=config['use_orderbook'],
            target_precision=0.08,  # 8% precision target
        )

        df = predictor.load_and_prepare_data("/Users/bz/Pythia2/whale_features.csv")

        results = predictor.walk_forward_validation(
            df,
            train_days=21,  # 3 weeks train
            test_days=7,    # 1 week test
            target_col='spike_10pct_24h',
        )

        if results:
            summary = predictor.print_summary(results)
            all_summaries[config['name']] = summary

            # Save predictions
            all_preds = pd.concat([r.predictions for r in results])
            suffix = 'with_ob' if config['use_orderbook'] else 'no_ob'
            pred_file = f"/Users/bz/Pythia2/ensemble_v2_predictions_{suffix}.csv"
            all_preds.to_csv(pred_file, index=False)
            print(f"\nPredictions saved to: {pred_file}")

    # Final comparison
    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON: BASELINE vs ORDERBOOK")
    print("=" * 80)

    for name, summary in all_summaries.items():
        ens = summary.get('ensemble', {})
        print(f"\n{name}:")
        print(f"  Ensemble AUC: {ens.get('avg_auc', 0):.3f} +/- {ens.get('std_auc', 0):.3f}")

    # Improvement analysis
    if len(all_summaries) == 2:
        baseline = all_summaries['Baseline (no orderbook)']['ensemble']['avg_auc']
        with_ob = all_summaries['With Orderbook Features']['ensemble']['avg_auc']
        improvement = (with_ob - baseline) / baseline * 100
        print(f"\nOrderbook AUC improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
