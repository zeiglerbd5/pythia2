"""
Spike Prediction Model

LightGBM-based model for predicting price spikes using catalyst signals.
Uses walk-forward validation for realistic backtesting.

Key features:
- Whale signal direction and magnitude
- Price/volume context at signal time
- Event type encoding
- Walk-forward validation (train on past, test on future)
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. Install with: pip install lightgbm")

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder


@dataclass
class WalkForwardResult:
    """Result from one walk-forward fold."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train: int
    n_test: int
    precision: float
    recall: float
    f1: float
    auc: float
    predictions: pd.DataFrame


class SpikePredictor:
    """
    LightGBM-based spike predictor with walk-forward validation.

    Uses catalyst signals + price context to predict spikes.
    """

    # Features to use for prediction
    NUMERIC_FEATURES = [
        'event_priority',
        'sentiment_score',
        'log_usd_value',
        'volatility_4h',
        'momentum_4h',
        'volume_ratio',
        'rsi_proxy',
    ]

    CATEGORICAL_FEATURES = [
        'direction',
    ]

    def __init__(self, db_path: str = "full_pythia.duckdb"):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM required. Install with: pip install lightgbm")

        self.conn = duckdb.connect(db_path)
        self.model = None
        self.label_encoders = {}
        self.feature_importance = None

    def load_features(self, feature_csv: str = "whale_features.csv") -> pd.DataFrame:
        """Load pre-computed features from CSV."""
        df = pd.read_csv(feature_csv, parse_dates=['timestamp'])
        logger.info(f"Loaded {len(df)} samples from {feature_csv}")
        return df

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'spike_10pct_24h') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.

        Returns:
            X: Feature matrix
            y: Target labels
        """
        # Copy to avoid modifying original
        data = df.copy()

        # Encode categorical features
        for col in self.CATEGORICAL_FEATURES:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data[col].fillna('unknown'))
                else:
                    # Handle unseen categories
                    data[f'{col}_encoded'] = data[col].fillna('unknown').apply(
                        lambda x: self.label_encoders[col].transform([x])[0]
                        if x in self.label_encoders[col].classes_
                        else -1
                    )

        # Select features
        feature_cols = self.NUMERIC_FEATURES.copy()
        for col in self.CATEGORICAL_FEATURES:
            if f'{col}_encoded' in data.columns:
                feature_cols.append(f'{col}_encoded')

        # Add boolean features
        bool_cols = ['is_bearish_flow', 'is_bullish_flow', 'has_direction']
        for col in bool_cols:
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
                X[col] = X[col].fillna(X[col].median())

        return X, y

    def walk_forward_validation(
        self,
        df: pd.DataFrame,
        train_months: float = 2,
        test_months: float = 1,
        target_col: str = 'spike_10pct_24h',
    ) -> List[WalkForwardResult]:
        """
        Perform walk-forward validation.

        Args:
            df: DataFrame with features and target
            train_months: Months of data for training
            test_months: Months of data for testing
            target_col: Target column name

        Returns:
            List of results per fold
        """
        logger.info(f"Starting walk-forward validation: {train_months}mo train, {test_months}mo test")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Get date range
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()

        results = []
        fold = 0

        # Walk forward through time
        train_start = min_date
        while True:
            train_end = train_start + timedelta(days=int(train_months * 30))
            test_start = train_end
            test_end = test_start + timedelta(days=int(test_months * 30))

            # Check if we have enough data
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
            logger.info(f"\nFold {fold}: Train {train_start.date()} to {train_end.date()}, "
                       f"Test {test_start.date()} to {test_end.date()}")

            # Prepare features
            X_train, y_train = self.prepare_features(train_df, target_col)
            X_test, y_test = self.prepare_features(test_df, target_col)

            # Train model
            self.model = self._train_model(X_train, y_train)

            # Predict
            y_pred_proba = self.model.predict(X_test)
            y_pred = (y_pred_proba >= 0.5).astype(int)

            # Metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                auc = 0.5

            # Store predictions
            predictions = test_df[['timestamp', 'symbol']].copy()
            predictions['y_true'] = y_test.values
            predictions['y_pred_proba'] = y_pred_proba
            predictions['y_pred'] = y_pred

            result = WalkForwardResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                n_train=len(train_df),
                n_test=len(test_df),
                precision=precision,
                recall=recall,
                f1=f1,
                auc=auc,
                predictions=predictions,
            )
            results.append(result)

            logger.info(f"  Train: {len(train_df)} samples ({y_train.sum()} spikes)")
            logger.info(f"  Test:  {len(test_df)} samples ({y_test.sum()} spikes)")
            logger.info(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

            # Move forward
            train_start = test_start

        return results

    def _train_model(self, X: pd.DataFrame, y: pd.Series) -> lgb.Booster:
        """Train LightGBM model."""
        # Handle class imbalance with scale_pos_weight
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1,
            'seed': 42,
        }

        train_data = lgb.Dataset(X, label=y)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
        )

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importance(importance_type='gain'),
        }).sort_values('importance', ascending=False)

        return model

    def print_summary(self, results: List[WalkForwardResult]) -> Dict:
        """Print summary of walk-forward validation results."""
        print("\n" + "=" * 70)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("=" * 70)

        # Aggregate metrics
        metrics = {
            'precision': np.mean([r.precision for r in results]),
            'recall': np.mean([r.recall for r in results]),
            'f1': np.mean([r.f1 for r in results]),
            'auc': np.mean([r.auc for r in results]),
            'n_folds': len(results),
            'total_test_samples': sum(r.n_test for r in results),
        }

        print(f"\nFolds: {metrics['n_folds']}")
        print(f"Total test samples: {metrics['total_test_samples']}")
        print(f"\nAverage Metrics:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  AUC:       {metrics['auc']:.3f}")

        # Per-fold breakdown
        print("\n" + "-" * 70)
        print("Per-Fold Results:")
        print("-" * 70)
        print(f"{'Fold':>4} | {'Test Period':^25} | {'N':>5} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'AUC':>6}")
        print("-" * 70)

        for i, r in enumerate(results):
            print(f"{i+1:>4} | {r.test_start.date()} to {r.test_end.date()} | "
                  f"{r.n_test:>5} | {r.precision:>6.3f} | {r.recall:>6.3f} | "
                  f"{r.f1:>6.3f} | {r.auc:>6.3f}")

        # Feature importance
        if self.feature_importance is not None:
            print("\n" + "-" * 70)
            print("Feature Importance (Final Model):")
            print("-" * 70)
            for _, row in self.feature_importance.head(10).iterrows():
                print(f"  {row['feature']:30} {row['importance']:>10.1f}")

        # Combine all predictions
        all_preds = pd.concat([r.predictions for r in results])

        # Precision at different thresholds
        print("\n" + "-" * 70)
        print("Precision at Different Thresholds:")
        print("-" * 70)
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            pred_pos = all_preds['y_pred_proba'] >= threshold
            n_pred = pred_pos.sum()
            if n_pred > 0:
                n_correct = (pred_pos & all_preds['y_true']).sum()
                prec = n_correct / n_pred
                print(f"  Threshold {threshold:.1f}: {n_pred:>4} predictions, {n_correct:>4} correct ({prec:.1%} precision)")

        return metrics


def main():
    """Train and evaluate spike predictor."""
    print("=" * 70)
    print("SPIKE PREDICTION MODEL (LightGBM)")
    print("=" * 70)

    # Load features
    predictor = SpikePredictor()

    try:
        df = predictor.load_features("whale_features.csv")
    except FileNotFoundError:
        print("\nFeature file not found. Running feature engineering first...")
        from src.features.catalyst_features import CatalystFeatureEngineer
        engineer = CatalystFeatureEngineer()
        df = engineer.build_whale_features()
        df.to_csv("whale_features.csv", index=False)

    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"Spike rate: {df['spike_10pct_24h'].mean()*100:.1f}%")

    # Walk-forward validation (shorter windows due to limited data)
    results = predictor.walk_forward_validation(
        df,
        train_months=1,
        test_months=0.5,  # 2 weeks
        target_col='spike_10pct_24h'
    )

    if results:
        metrics = predictor.print_summary(results)

        # Save predictions
        all_preds = pd.concat([r.predictions for r in results])
        all_preds.to_csv("spike_predictions.csv", index=False)
        print(f"\nPredictions saved to: spike_predictions.csv")
    else:
        print("\nNo folds completed - insufficient data for walk-forward validation")


if __name__ == "__main__":
    main()
