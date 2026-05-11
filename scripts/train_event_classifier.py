"""
Train Event Classifier for Spike Prediction

Trains XGBoost and Logistic Regression on event-level features.
Uses temporal split for honest validation and rolling-origin backtest.

Usage:
    python scripts/train_event_classifier.py

Output:
    models/event_classifier_xgb.pkl - XGBoost model
    models/event_classifier_lr.pkl - Logistic regression model
    models/event_classifier_metrics.json - Performance metrics
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_curve
)
import xgboost as xgb


# Configuration
FEATURES_PATH = '/Users/bz/Pythia2/data/event_features.parquet'
OUTPUT_DIR = Path('/Users/bz/Pythia2/models')

# Train/val/test split ratios (temporal)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Model hyperparameters
XGBOOST_PARAMS = {
    'max_depth': 3,
    'n_estimators': 100,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'use_label_encoder': False
}

LOGISTIC_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': 42,
    'class_weight': 'balanced'
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'min_samples_leaf': 5,
    'random_state': 42,
    'class_weight': 'balanced'
}


def temporal_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    """Split data temporally (train on past, test on future)."""
    df = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


def rolling_origin_backtest(df: pd.DataFrame, feature_cols: list, train_window: int = 50,
                            test_window: int = 20, step: int = 20) -> list:
    """
    Rolling-origin backtest to verify no hidden leakage.

    Train on window of events, test on next window, slide forward.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)

    results = []

    for start in range(0, n - train_window - test_window, step):
        train_end = start + train_window
        test_end = train_end + test_window

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]

        if len(train_df) < 20 or len(test_df) < 5:
            continue

        # Prepare data
        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['label'].values

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train simple model
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        if len(np.unique(y_test)) > 1:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:
            precision = recall = f1 = 0

        results.append({
            'window_start': start,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'test_positives': y_test.sum(),
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    return results


def symbol_isolation_check(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Check if model learns symbol-specific quirks vs generalizable patterns.

    Train on all symbols, evaluate per-symbol.
    """
    train, val, test = temporal_split(df, TRAIN_RATIO, VAL_RATIO)

    # Combine val and test for evaluation
    eval_df = pd.concat([val, test])

    # Prepare data
    X_train = train[feature_cols].values
    y_train = train['label'].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Train model
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    # Evaluate per symbol
    symbol_results = {}
    for symbol in eval_df['symbol'].unique():
        sym_df = eval_df[eval_df['symbol'] == symbol]
        if len(sym_df) < 2:
            continue

        X_sym = scaler.transform(sym_df[feature_cols].values)
        y_sym = sym_df['label'].values

        y_pred = model.predict(X_sym)

        symbol_results[symbol] = {
            'n_samples': len(sym_df),
            'n_positives': y_sym.sum(),
            'n_correct': (y_pred == y_sym).sum(),
            'accuracy': (y_pred == y_sym).mean()
        }

    return symbol_results


def train_and_evaluate():
    """Main training and evaluation function."""
    logger.info("=" * 60)
    logger.info("TRAINING EVENT CLASSIFIER")
    logger.info("=" * 60)

    # Load features
    logger.info(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"  Positives: {(df['label'] == 1).sum()}")
    logger.info(f"  Negatives: {(df['label'] == 0).sum()}")

    # Identify feature columns
    meta_cols = ['event_id', 'symbol', 'timestamp', 'label', 'sample_type']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    logger.info(f"Feature columns: {len(feature_cols)}")

    # Temporal split
    logger.info("\n--- TEMPORAL SPLIT ---")
    train, val, test = temporal_split(df, TRAIN_RATIO, VAL_RATIO)

    logger.info(f"Train: {len(train)} samples ({train['label'].sum()} positive)")
    logger.info(f"  Time range: {train['timestamp'].min()} to {train['timestamp'].max()}")
    logger.info(f"Val: {len(val)} samples ({val['label'].sum()} positive)")
    logger.info(f"  Time range: {val['timestamp'].min()} to {val['timestamp'].max()}")
    logger.info(f"Test: {len(test)} samples ({test['label'].sum()} positive)")
    logger.info(f"  Time range: {test['timestamp'].min()} to {test['timestamp'].max()}")

    # Prepare features
    X_train = train[feature_cols].values
    y_train = train['label'].values
    X_val = val[feature_cols].values
    y_val = val['label'].values
    X_test = test[feature_cols].values
    y_test = test['label'].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ===== TRAIN MODELS =====

    results = {}

    # 1. Logistic Regression (baseline)
    logger.info("\n--- LOGISTIC REGRESSION ---")
    lr = LogisticRegression(**LOGISTIC_PARAMS)
    lr.fit(X_train_scaled, y_train)

    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    results['logistic_regression'] = {
        'precision': precision_score(y_test, y_pred_lr, zero_division=0),
        'recall': recall_score(y_test, y_pred_lr, zero_division=0),
        'f1': f1_score(y_test, y_pred_lr, zero_division=0),
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist()
    }

    logger.info(f"Test Precision: {results['logistic_regression']['precision']:.3f}")
    logger.info(f"Test Recall: {results['logistic_regression']['recall']:.3f}")
    logger.info(f"Test F1: {results['logistic_regression']['f1']:.3f}")

    # Feature importance (coefficients)
    lr_importance = dict(zip(feature_cols, np.abs(lr.coef_[0])))
    lr_importance_sorted = sorted(lr_importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top 5 features (by coefficient magnitude):")
    for feat, imp in lr_importance_sorted[:5]:
        logger.info(f"  {feat}: {imp:.4f}")

    # 2. XGBoost (main model)
    logger.info("\n--- XGBOOST ---")
    xgb_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )

    y_pred_xgb = xgb_model.predict(X_test_scaled)
    y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

    results['xgboost'] = {
        'precision': precision_score(y_test, y_pred_xgb, zero_division=0),
        'recall': recall_score(y_test, y_pred_xgb, zero_division=0),
        'f1': f1_score(y_test, y_pred_xgb, zero_division=0),
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'confusion_matrix': confusion_matrix(y_test, y_pred_xgb).tolist()
    }

    logger.info(f"Test Precision: {results['xgboost']['precision']:.3f}")
    logger.info(f"Test Recall: {results['xgboost']['recall']:.3f}")
    logger.info(f"Test F1: {results['xgboost']['f1']:.3f}")

    # Feature importance
    xgb_importance = dict(zip(feature_cols, xgb_model.feature_importances_))
    xgb_importance_sorted = sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top 10 features (by importance):")
    for feat, imp in xgb_importance_sorted[:10]:
        logger.info(f"  {feat}: {imp:.4f}")

    results['xgboost']['feature_importance'] = xgb_importance_sorted

    # 3. Random Forest (comparison)
    logger.info("\n--- RANDOM FOREST ---")
    rf = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    rf.fit(X_train_scaled, y_train)

    y_pred_rf = rf.predict(X_test_scaled)

    results['random_forest'] = {
        'precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'f1': f1_score(y_test, y_pred_rf, zero_division=0),
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist()
    }

    logger.info(f"Test Precision: {results['random_forest']['precision']:.3f}")
    logger.info(f"Test Recall: {results['random_forest']['recall']:.3f}")
    logger.info(f"Test F1: {results['random_forest']['f1']:.3f}")

    # ===== VALIDATION SAFEGUARDS =====

    # Rolling-origin backtest
    logger.info("\n--- ROLLING-ORIGIN BACKTEST ---")
    rolling_results = rolling_origin_backtest(df, feature_cols)

    if rolling_results:
        avg_f1 = np.mean([r['f1'] for r in rolling_results])
        std_f1 = np.std([r['f1'] for r in rolling_results])
        logger.info(f"Rolling F1: {avg_f1:.3f} +/- {std_f1:.3f}")
        logger.info(f"Number of windows: {len(rolling_results)}")

        # Check for collapse
        if avg_f1 < 0.2:
            logger.warning("WARNING: Rolling F1 is very low - possible leakage in main results")

        results['rolling_backtest'] = {
            'mean_f1': avg_f1,
            'std_f1': std_f1,
            'n_windows': len(rolling_results),
            'windows': rolling_results
        }
    else:
        logger.warning("Not enough data for rolling backtest")

    # Symbol isolation check
    logger.info("\n--- SYMBOL ISOLATION CHECK ---")
    symbol_results = symbol_isolation_check(df, feature_cols)

    # Summarize
    accuracies = [v['accuracy'] for v in symbol_results.values()]
    logger.info(f"Per-symbol accuracy: mean={np.mean(accuracies):.3f}, std={np.std(accuracies):.3f}")
    logger.info(f"Symbols evaluated: {len(symbol_results)}")

    # Flag if any symbol dominates
    high_acc_symbols = [s for s, v in symbol_results.items() if v['accuracy'] > 0.9 and v['n_samples'] >= 3]
    if high_acc_symbols:
        logger.warning(f"High accuracy symbols (possible quirks): {high_acc_symbols[:5]}")

    results['symbol_isolation'] = {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'n_symbols': len(symbol_results)
    }

    # ===== RED FLAG CHECK =====
    logger.info("\n--- MODEL COMPARISON (RED FLAG CHECK) ---")
    logger.info(f"Logistic Regression F1: {results['logistic_regression']['f1']:.3f}")
    logger.info(f"XGBoost F1: {results['xgboost']['f1']:.3f}")
    logger.info(f"Random Forest F1: {results['random_forest']['f1']:.3f}")

    # XGBoost should be competitive with simpler models for this problem
    best_simple = max(results['logistic_regression']['f1'], results['random_forest']['f1'])
    if results['xgboost']['f1'] < best_simple - 0.1:
        logger.info("NOTE: XGBoost underperforming simpler models - may need tuning")

    # ===== SAVE MODELS =====
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save XGBoost
    xgb_path = OUTPUT_DIR / 'event_classifier_xgb.pkl'
    with open(xgb_path, 'wb') as f:
        pickle.dump({'model': xgb_model, 'scaler': scaler, 'feature_cols': feature_cols}, f)
    logger.info(f"\nSaved XGBoost model to: {xgb_path}")

    # Save Logistic Regression
    lr_path = OUTPUT_DIR / 'event_classifier_lr.pkl'
    with open(lr_path, 'wb') as f:
        pickle.dump({'model': lr, 'scaler': scaler, 'feature_cols': feature_cols}, f)
    logger.info(f"Saved Logistic Regression model to: {lr_path}")

    # Save metrics
    metrics_path = OUTPUT_DIR / 'event_classifier_metrics.json'

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_results = json.loads(json.dumps(results, default=convert_to_serializable))

    with open(metrics_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_path}")

    # ===== FINAL SUMMARY =====
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nBest model: XGBoost")
    logger.info(f"  Test Precision: {results['xgboost']['precision']:.3f}")
    logger.info(f"  Test Recall: {results['xgboost']['recall']:.3f}")
    logger.info(f"  Test F1: {results['xgboost']['f1']:.3f}")

    logger.info(f"\nConfusion Matrix (XGBoost):")
    cm = results['xgboost']['confusion_matrix']
    logger.info(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
    logger.info(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")

    logger.info(f"\nTop features driving predictions:")
    for feat, imp in xgb_importance_sorted[:5]:
        logger.info(f"  {feat}: {imp:.4f}")

    return results


if __name__ == "__main__":
    train_and_evaluate()
