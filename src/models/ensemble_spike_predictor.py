"""
Ensemble Spike Predictor with Orderbook Features

Combines multiple models (LightGBM, XGBoost, RandomForest) with orderbook
microstructure features to predict price spikes.

Features integrated:
- Catalyst signals (whale moves, direction)
- Price/volume context
- Orderbook features (imbalance, spread, depth, large orders)

Ensemble methods:
- Soft voting (average probabilities)
- Weighted voting (based on validation performance)
- Stacking (meta-learner on base model outputs)
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
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, classification_report,
    confusion_matrix
)


@dataclass
class EnsembleResult:
    """Result from ensemble walk-forward validation."""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train: int
    n_test: int
    n_spikes_train: int
    n_spikes_test: int
    # Per-model metrics
    model_metrics: Dict[str, Dict[str, float]]
    # Ensemble metrics
    ensemble_metrics: Dict[str, float]
    # Predictions
    predictions: pd.DataFrame
    # Feature importance (from best model)
    feature_importance: Optional[pd.DataFrame] = None


class OrderbookFeatureComputer:
    """Computes orderbook features from order_book_snapshots table."""

    def __init__(self, db_path: str = "full_pythia.duckdb"):
        self.conn = duckdb.connect(db_path, read_only=True)

    def compute_features_at_time(
        self,
        symbol: str,
        timestamp: datetime,
        lookback_minutes: int = 30
    ) -> Dict[str, float]:
        """
        Compute orderbook features at a specific time.

        Uses snapshots from the past lookback_minutes to compute:
        - Bid-ask spread metrics
        - Order imbalance (bid vs ask volume)
        - Large order detection
        - Order book depth analysis
        """
        try:
            # Get recent snapshots
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

            # Compute features from most recent snapshot
            latest = snapshots.iloc[0]
            bids = json.loads(latest['bids']) if isinstance(latest['bids'], str) else latest['bids']
            asks = json.loads(latest['asks']) if isinstance(latest['asks'], str) else latest['asks']

            features = {}

            # Spread features
            features['spread_bps'] = latest['spread_bps']
            features['bid_ask_spread_pct'] = (latest['best_ask'] - latest['best_bid']) / latest['mid_price'] * 100 if latest['mid_price'] > 0 else 0

            # Parse bid/ask levels
            bid_prices = [b[0] for b in bids[:10]]
            bid_sizes = [b[1] for b in bids[:10]]
            ask_prices = [a[0] for a in asks[:10]]
            ask_sizes = [a[1] for a in asks[:10]]

            # Volume imbalance at top levels
            total_bid_vol = sum(bid_sizes[:5])
            total_ask_vol = sum(ask_sizes[:5])
            total_vol = total_bid_vol + total_ask_vol

            features['order_book_imbalance_l5'] = (total_bid_vol - total_ask_vol) / total_vol if total_vol > 0 else 0
            features['bid_ask_ratio'] = total_bid_vol / total_ask_vol if total_ask_vol > 0 else 1.0

            # Depth features (USD value at different levels)
            if len(bid_prices) >= 5 and len(ask_prices) >= 5:
                bid_depth_usd = sum(p * s for p, s in zip(bid_prices[:5], bid_sizes[:5]))
                ask_depth_usd = sum(p * s for p, s in zip(ask_prices[:5], ask_sizes[:5]))
                features['order_book_depth_ratio'] = bid_depth_usd / ask_depth_usd if ask_depth_usd > 0 else 1.0
                features['total_depth_usd'] = bid_depth_usd + ask_depth_usd
            else:
                features['order_book_depth_ratio'] = 1.0
                features['total_depth_usd'] = 0

            # Large order detection (orders > 2x median size)
            if len(bid_sizes) > 0 and len(ask_sizes) > 0:
                all_sizes = bid_sizes + ask_sizes
                median_size = np.median(all_sizes)
                large_threshold = median_size * 2

                features['large_bid_orders'] = sum(1 for s in bid_sizes if s > large_threshold)
                features['large_ask_orders'] = sum(1 for s in ask_sizes if s > large_threshold)
                features['large_order_imbalance'] = features['large_bid_orders'] - features['large_ask_orders']
            else:
                features['large_bid_orders'] = 0
                features['large_ask_orders'] = 0
                features['large_order_imbalance'] = 0

            # Weighted mid price (volume-weighted)
            if total_vol > 0:
                weighted_bid = sum(p * s for p, s in zip(bid_prices[:5], bid_sizes[:5])) / total_bid_vol if total_bid_vol > 0 else latest['best_bid']
                weighted_ask = sum(p * s for p, s in zip(ask_prices[:5], ask_sizes[:5])) / total_ask_vol if total_ask_vol > 0 else latest['best_ask']
                features['weighted_mid_price'] = (weighted_bid + weighted_ask) / 2
                features['mid_to_weighted_ratio'] = latest['mid_price'] / features['weighted_mid_price'] if features['weighted_mid_price'] > 0 else 1.0
            else:
                features['weighted_mid_price'] = latest['mid_price']
                features['mid_to_weighted_ratio'] = 1.0

            # Spread dynamics (if we have multiple snapshots)
            if len(snapshots) > 1:
                spreads = snapshots['spread_bps'].values
                features['spread_volatility'] = np.std(spreads) if len(spreads) > 1 else 0
                features['spread_trend'] = spreads[0] - spreads[-1]  # Positive = widening
            else:
                features['spread_volatility'] = 0
                features['spread_trend'] = 0

            return features

        except Exception as e:
            logger.debug(f"Orderbook feature error for {symbol}: {e}")
            return self._empty_orderbook_features()

    def _empty_orderbook_features(self) -> Dict[str, float]:
        """Return empty orderbook features."""
        return {
            'spread_bps': None,
            'bid_ask_spread_pct': None,
            'order_book_imbalance_l5': None,
            'bid_ask_ratio': None,
            'order_book_depth_ratio': None,
            'total_depth_usd': None,
            'large_bid_orders': None,
            'large_ask_orders': None,
            'large_order_imbalance': None,
            'weighted_mid_price': None,
            'mid_to_weighted_ratio': None,
            'spread_volatility': None,
            'spread_trend': None,
        }

    def batch_compute_features(
        self,
        signals_df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        symbol_col: str = 'symbol'
    ) -> pd.DataFrame:
        """
        Compute orderbook features for all signals in a dataframe.
        """
        logger.info(f"Computing orderbook features for {len(signals_df)} signals...")

        features_list = []
        for idx, row in signals_df.iterrows():
            feat = self.compute_features_at_time(row[symbol_col], row[timestamp_col])
            features_list.append(feat)

            if (idx + 1) % 500 == 0:
                logger.info(f"Processed {idx + 1}/{len(signals_df)} signals")

        features_df = pd.DataFrame(features_list)
        logger.info(f"Computed {len(features_df)} orderbook feature rows")

        # Count non-null values
        non_null_counts = features_df.notna().sum()
        logger.info(f"Feature coverage: {non_null_counts.to_dict()}")

        return features_df


class EnsembleSpikePredictor:
    """
    Ensemble spike predictor combining multiple models with orderbook features.
    """

    # Base features from whale signals
    CATALYST_FEATURES = [
        'event_priority',
        'sentiment_score',
        'log_usd_value',
        'volatility_4h',
        'momentum_4h',
        'volume_ratio',
        'rsi_proxy',
    ]

    # Orderbook features
    ORDERBOOK_FEATURES = [
        'spread_bps',
        'bid_ask_spread_pct',
        'order_book_imbalance_l5',
        'bid_ask_ratio',
        'order_book_depth_ratio',
        'large_bid_orders',
        'large_ask_orders',
        'large_order_imbalance',
        'mid_to_weighted_ratio',
        'spread_volatility',
        'spread_trend',
    ]

    # Boolean features
    BOOL_FEATURES = [
        'is_bearish_flow',
        'is_bullish_flow',
        'has_direction',
    ]

    def __init__(
        self,
        db_path: str = "full_pythia.duckdb",
        use_orderbook: bool = True,
        ensemble_method: str = 'weighted_voting'  # 'soft_voting', 'weighted_voting', 'stacking'
    ):
        self.db_path = db_path
        self.use_orderbook = use_orderbook
        self.ensemble_method = ensemble_method

        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.model_weights = {}
        self.meta_learner = None
        self.feature_importance = None

        if use_orderbook:
            self.ob_computer = OrderbookFeatureComputer(db_path)

    def load_and_prepare_data(
        self,
        whale_csv: str = "whale_features.csv"
    ) -> pd.DataFrame:
        """Load whale features and add orderbook features."""
        # Load base features
        df = pd.read_csv(whale_csv, parse_dates=['timestamp'])
        logger.info(f"Loaded {len(df)} whale signals from {whale_csv}")

        if self.use_orderbook:
            # Compute orderbook features
            ob_features = self.ob_computer.batch_compute_features(df)

            # Merge
            df = pd.concat([df.reset_index(drop=True), ob_features], axis=1)
            logger.info(f"Added {len(self.ORDERBOOK_FEATURES)} orderbook features")

        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'spike_10pct_24h',
        fit: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target for training/prediction.
        """
        data = df.copy()

        # Select features based on configuration
        feature_cols = self.CATALYST_FEATURES.copy()

        if self.use_orderbook:
            feature_cols.extend(self.ORDERBOOK_FEATURES)

        # Add boolean features
        for col in self.BOOL_FEATURES:
            if col in data.columns:
                feature_cols.append(col)
                data[col] = data[col].astype(int)

        # Filter to available features
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

    def _get_lgb_model(self, n_pos: int, n_neg: int) -> lgb.LGBMClassifier:
        """Create LightGBM classifier with optimal parameters."""
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
        return lgb.LGBMClassifier(
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

    def _get_xgb_model(self, n_pos: int, n_neg: int) -> 'xgb.XGBClassifier':
        """Create XGBoost classifier."""
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
        return xgb.XGBClassifier(
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

    def _get_rf_model(self, n_pos: int, n_neg: int) -> RandomForestClassifier:
        """Create RandomForest classifier."""
        class_weight = {0: 1, 1: n_neg / n_pos} if n_pos > 0 else 'balanced'
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )

    def _get_gb_model(self, n_pos: int, n_neg: int) -> GradientBoostingClassifier:
        """Create Gradient Boosting classifier (scikit-learn)."""
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    def train_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train all base models and optionally the meta-learner.
        """
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos

        logger.info(f"Training ensemble on {len(X_train)} samples ({n_pos} positive)")

        self.models = {}
        results = {}

        # Train LightGBM
        if HAS_LIGHTGBM:
            logger.info("Training LightGBM...")
            self.models['lightgbm'] = self._get_lgb_model(n_pos, n_neg)
            self.models['lightgbm'].fit(X_train, y_train)
            results['lightgbm'] = {'trained': True}

        # Train XGBoost
        if HAS_XGBOOST:
            logger.info("Training XGBoost...")
            self.models['xgboost'] = self._get_xgb_model(n_pos, n_neg)
            self.models['xgboost'].fit(X_train, y_train)
            results['xgboost'] = {'trained': True}

        # Train RandomForest
        logger.info("Training RandomForest...")
        self.models['randomforest'] = self._get_rf_model(n_pos, n_neg)
        self.models['randomforest'].fit(X_train, y_train)
        results['randomforest'] = {'trained': True}

        # Train GradientBoosting
        logger.info("Training GradientBoosting...")
        self.models['gradientboosting'] = self._get_gb_model(n_pos, n_neg)
        self.models['gradientboosting'].fit(X_train, y_train)
        results['gradientboosting'] = {'trained': True}

        # Compute weights based on validation performance
        if X_val is not None and y_val is not None:
            self._compute_model_weights(X_val, y_val)
        else:
            # Equal weights
            self.model_weights = {name: 1.0 / len(self.models) for name in self.models}

        # Train stacking meta-learner if needed
        if self.ensemble_method == 'stacking' and X_val is not None:
            self._train_meta_learner(X_train, y_train, X_val, y_val)

        # Store feature importance from best model
        self._compute_feature_importance(X_train)

        return results

    def _compute_model_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> None:
        """Compute model weights based on validation AUC."""
        aucs = {}
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, proba)
                aucs[name] = auc
            except Exception as e:
                logger.warning(f"Could not compute AUC for {name}: {e}")
                aucs[name] = 0.5

        # Normalize weights (higher AUC = higher weight)
        # Use (AUC - 0.5) to weight above chance
        adjusted = {k: max(v - 0.5, 0.01) for k, v in aucs.items()}
        total = sum(adjusted.values())
        self.model_weights = {k: v / total for k, v in adjusted.items()}

        logger.info(f"Model weights: {self.model_weights}")

    def _train_meta_learner(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Train a meta-learner on base model predictions."""
        # Get base model predictions on validation set
        meta_features = self._get_meta_features(X_val)

        # Train logistic regression as meta-learner
        self.meta_learner = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
        )
        self.meta_learner.fit(meta_features, y_val)

        logger.info("Trained stacking meta-learner")

    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Get meta-features (base model predictions) for stacking."""
        meta = []
        for name, model in self.models.items():
            proba = model.predict_proba(X)[:, 1]
            meta.append(proba)
        return np.column_stack(meta)

    def _compute_feature_importance(self, X: pd.DataFrame) -> None:
        """Compute aggregate feature importance from models."""
        importances = []

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_,
                    'model': name,
                })
                importances.append(imp)

        if importances:
            all_imp = pd.concat(importances)
            avg_imp = all_imp.groupby('feature')['importance'].mean().reset_index()
            avg_imp = avg_imp.sort_values('importance', ascending=False)
            self.feature_importance = avg_imp

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble probability predictions.
        """
        if self.ensemble_method == 'stacking' and self.meta_learner is not None:
            meta_features = self._get_meta_features(X)
            return self.meta_learner.predict_proba(meta_features)[:, 1]

        # Soft voting or weighted voting
        predictions = []
        weights = []

        for name, model in self.models.items():
            proba = model.predict_proba(X)[:, 1]
            predictions.append(proba)
            weights.append(self.model_weights.get(name, 1.0))

        predictions = np.array(predictions)
        weights = np.array(weights)

        if self.ensemble_method == 'weighted_voting':
            # Weighted average
            ensemble_proba = np.average(predictions, axis=0, weights=weights)
        else:
            # Simple average (soft voting)
            ensemble_proba = np.mean(predictions, axis=0)

        return ensemble_proba

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def walk_forward_validation(
        self,
        df: pd.DataFrame,
        train_months: float = 1.5,
        test_months: float = 0.5,
        target_col: str = 'spike_10pct_24h',
    ) -> List[EnsembleResult]:
        """
        Perform walk-forward validation with ensemble.
        """
        logger.info(f"Walk-forward validation: {train_months}mo train, {test_months}mo test")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()

        results = []
        fold = 0

        train_start = min_date

        while True:
            train_end = train_start + timedelta(days=int(train_months * 30))
            test_start = train_end
            test_end = test_start + timedelta(days=int(test_months * 30))

            if test_end > max_date:
                break

            # Split data
            train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] < train_end)
            test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)

            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(train_df) < 50 or len(test_df) < 10:
                train_start = test_start
                continue

            fold += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold}: Train {train_start.date()} to {train_end.date()}")
            logger.info(f"         Test {test_start.date()} to {test_end.date()}")

            # Prepare features
            X_train, y_train = self.prepare_features(train_df, target_col, fit=True)
            X_test, y_test = self.prepare_features(test_df, target_col, fit=False)

            logger.info(f"Train: {len(X_train)} samples ({y_train.sum()} spikes)")
            logger.info(f"Test:  {len(X_test)} samples ({y_test.sum()} spikes)")

            # Use last 20% of training for validation
            val_split = int(len(X_train) * 0.8)
            X_tr = X_train.iloc[:val_split]
            y_tr = y_train.iloc[:val_split]
            X_val = X_train.iloc[val_split:]
            y_val = y_train.iloc[val_split:]

            # Train ensemble
            self.train_ensemble(X_tr, y_tr, X_val, y_val)

            # Get predictions for each model and ensemble
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
            logger.info(f"\n--- Individual Model Results ---")
            for name, metrics in model_metrics.items():
                logger.info(f"{name:20}: AUC={metrics['auc']:.3f}, Prec={metrics['precision']:.3f}, "
                           f"Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

            logger.info(f"\n--- Ensemble ({self.ensemble_method}) ---")
            logger.info(f"{'ENSEMBLE':20}: AUC={ensemble_metrics['auc']:.3f}, "
                       f"Prec={ensemble_metrics['precision']:.3f}, "
                       f"Recall={ensemble_metrics['recall']:.3f}, F1={ensemble_metrics['f1']:.3f}")

            # Store predictions
            predictions = test_df[['timestamp', 'symbol']].copy()
            predictions['y_true'] = y_test.values
            predictions['ensemble_proba'] = ensemble_proba
            predictions['ensemble_pred'] = ensemble_pred

            for name, proba in model_preds.items():
                predictions[f'{name}_proba'] = proba

            result = EnsembleResult(
                fold=fold,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train=len(train_df),
                n_test=len(test_df),
                n_spikes_train=int(y_train.sum()),
                n_spikes_test=int(y_test.sum()),
                model_metrics=model_metrics,
                ensemble_metrics=ensemble_metrics,
                predictions=predictions,
                feature_importance=self.feature_importance.copy() if self.feature_importance is not None else None,
            )
            results.append(result)

            # Move forward
            train_start = test_start

        return results

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
        }

        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc'] = 0.5

        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics['pr_auc'] = 0.0

        try:
            metrics['brier'] = brier_score_loss(y_true, y_proba)
        except ValueError:
            metrics['brier'] = 1.0

        return metrics

    def print_summary(self, results: List[EnsembleResult]) -> Dict:
        """Print comprehensive summary of results."""
        print("\n" + "=" * 80)
        print("ENSEMBLE SPIKE PREDICTOR - WALK-FORWARD VALIDATION SUMMARY")
        print("=" * 80)

        # Aggregate per-model metrics
        model_names = list(results[0].model_metrics.keys())

        print(f"\nEnsemble method: {self.ensemble_method}")
        print(f"Using orderbook features: {self.use_orderbook}")
        print(f"Total folds: {len(results)}")
        print(f"Total test samples: {sum(r.n_test for r in results)}")

        print("\n" + "-" * 80)
        print("AVERAGE METRICS BY MODEL")
        print("-" * 80)
        print(f"{'Model':20} | {'AUC':>7} | {'PR-AUC':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7}")
        print("-" * 80)

        summary = {}

        for name in model_names:
            aucs = [r.model_metrics[name]['auc'] for r in results]
            pr_aucs = [r.model_metrics[name].get('pr_auc', 0) for r in results]
            precs = [r.model_metrics[name]['precision'] for r in results]
            recalls = [r.model_metrics[name]['recall'] for r in results]
            f1s = [r.model_metrics[name]['f1'] for r in results]

            summary[name] = {
                'auc': np.mean(aucs),
                'pr_auc': np.mean(pr_aucs),
                'precision': np.mean(precs),
                'recall': np.mean(recalls),
                'f1': np.mean(f1s),
            }

            print(f"{name:20} | {np.mean(aucs):>7.3f} | {np.mean(pr_aucs):>7.3f} | "
                  f"{np.mean(precs):>7.3f} | {np.mean(recalls):>7.3f} | {np.mean(f1s):>7.3f}")

        # Ensemble metrics
        ens_aucs = [r.ensemble_metrics['auc'] for r in results]
        ens_pr_aucs = [r.ensemble_metrics.get('pr_auc', 0) for r in results]
        ens_precs = [r.ensemble_metrics['precision'] for r in results]
        ens_recalls = [r.ensemble_metrics['recall'] for r in results]
        ens_f1s = [r.ensemble_metrics['f1'] for r in results]

        print("-" * 80)
        print(f"{'ENSEMBLE':20} | {np.mean(ens_aucs):>7.3f} | {np.mean(ens_pr_aucs):>7.3f} | "
              f"{np.mean(ens_precs):>7.3f} | {np.mean(ens_recalls):>7.3f} | {np.mean(ens_f1s):>7.3f}")

        summary['ensemble'] = {
            'auc': np.mean(ens_aucs),
            'pr_auc': np.mean(ens_pr_aucs),
            'precision': np.mean(ens_precs),
            'recall': np.mean(ens_recalls),
            'f1': np.mean(ens_f1s),
        }

        # Per-fold breakdown
        print("\n" + "-" * 80)
        print("PER-FOLD ENSEMBLE RESULTS")
        print("-" * 80)
        print(f"{'Fold':>4} | {'Test Period':^23} | {'N':>5} | {'Spikes':>6} | {'AUC':>7} | {'Prec':>7} | {'Rec':>7}")
        print("-" * 80)

        for r in results:
            print(f"{r.fold:>4} | {r.test_start.date()} to {r.test_end.date()} | "
                  f"{r.n_test:>5} | {r.n_spikes_test:>6} | "
                  f"{r.ensemble_metrics['auc']:>7.3f} | "
                  f"{r.ensemble_metrics['precision']:>7.3f} | "
                  f"{r.ensemble_metrics['recall']:>7.3f}")

        # Feature importance
        if results[-1].feature_importance is not None:
            print("\n" + "-" * 80)
            print("FEATURE IMPORTANCE (Average across models)")
            print("-" * 80)
            for _, row in results[-1].feature_importance.head(15).iterrows():
                feat_type = "OB" if row['feature'] in self.ORDERBOOK_FEATURES else "CT"
                print(f"  [{feat_type}] {row['feature']:35} {row['importance']:>10.1f}")

        # Precision at thresholds
        all_preds = pd.concat([r.predictions for r in results])

        print("\n" + "-" * 80)
        print("PRECISION AT DIFFERENT THRESHOLDS (Ensemble)")
        print("-" * 80)

        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            pred_pos = all_preds['ensemble_proba'] >= threshold
            n_pred = pred_pos.sum()
            if n_pred > 0:
                n_correct = (pred_pos & all_preds['y_true']).sum()
                prec = n_correct / n_pred
                print(f"  Threshold {threshold:.1f}: {n_pred:>4} predictions, "
                      f"{n_correct:>4} correct ({prec:.1%} precision)")

        return summary


def main():
    """Run ensemble spike predictor evaluation."""
    print("=" * 80)
    print("ENSEMBLE SPIKE PREDICTOR WITH ORDERBOOK FEATURES")
    print("=" * 80)

    # Test different configurations
    configs = [
        {'use_orderbook': False, 'ensemble_method': 'weighted_voting', 'name': 'Baseline (no OB)'},
        {'use_orderbook': True, 'ensemble_method': 'soft_voting', 'name': 'With OB (soft voting)'},
        {'use_orderbook': True, 'ensemble_method': 'weighted_voting', 'name': 'With OB (weighted)'},
        {'use_orderbook': True, 'ensemble_method': 'stacking', 'name': 'With OB (stacking)'},
    ]

    all_summaries = {}

    for config in configs:
        print(f"\n\n{'#' * 80}")
        print(f"# CONFIGURATION: {config['name']}")
        print(f"{'#' * 80}")

        predictor = EnsembleSpikePredictor(
            db_path="/Users/bz/Pythia2/full_pythia.duckdb",
            use_orderbook=config['use_orderbook'],
            ensemble_method=config['ensemble_method'],
        )

        # Load and prepare data
        df = predictor.load_and_prepare_data("/Users/bz/Pythia2/whale_features.csv")

        # Run walk-forward validation
        results = predictor.walk_forward_validation(
            df,
            train_months=1.0,
            test_months=0.5,
            target_col='spike_10pct_24h',
        )

        if results:
            summary = predictor.print_summary(results)
            all_summaries[config['name']] = summary

            # Save predictions
            all_preds = pd.concat([r.predictions for r in results])
            pred_file = f"/Users/bz/Pythia2/ensemble_predictions_{config['ensemble_method']}.csv"
            all_preds.to_csv(pred_file, index=False)
            print(f"\nPredictions saved to: {pred_file}")

    # Final comparison
    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON ACROSS CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Configuration':35} | {'AUC':>7} | {'PR-AUC':>7} | {'Prec':>7} | {'Recall':>7}")
    print("-" * 80)

    for name, summary in all_summaries.items():
        ens = summary.get('ensemble', {})
        print(f"{name:35} | {ens.get('auc', 0):>7.3f} | {ens.get('pr_auc', 0):>7.3f} | "
              f"{ens.get('precision', 0):>7.3f} | {ens.get('recall', 0):>7.3f}")


if __name__ == "__main__":
    main()
