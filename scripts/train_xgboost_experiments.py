"""
XGBoost Hyperparameter Experiment Suite

Trains 30 XGBoost models (X1-X30) with different parameter configurations
to find optimal settings for spike prediction.

Philosophy:
- Spikes are RARE events - need to balance precision vs recall
- Spikes have PRECURSOR patterns - volatility, volume shifts
- Overfitting is a real risk with ~1200 samples
- Different market regimes may favor different model types

Model Categories:
- X1-X5: Depth exploration (shallow to deep)
- X6-X9: Learning rate & regularization
- X10-X14: Conservative vs aggressive
- X15-X18: L1/L2 regularization & class imbalance
- X19-X22: Precision vs recall tradeoffs
- X23-X26: Sampling strategies
- X27-X30: Ensemble candidates (diverse configs)

Usage:
    python scripts/train_xgboost_experiments.py

Output:
    models/experiments/X{n}/model.pkl
    models/experiments/X{n}/params.json
    models/experiments/summary.csv
    models/experiments/summary.json
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_auc_score,
    average_precision_score
)
import xgboost as xgb

# Configuration
FEATURES_PATH = '/Users/bz/Pythia2/data/event_features.parquet'
OUTPUT_DIR = Path('/Users/bz/Pythia2/models/experiments')
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Base params (shared across all models)
BASE_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'use_label_encoder': False,
    'verbosity': 0
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
# Each model has a name, description, and specific params that override BASE_PARAMS

EXPERIMENTS = [
    # ----- DEPTH EXPLORATION (X1-X5) -----
    {
        'name': 'X1',
        'description': 'Baseline (V5 params)',
        'philosophy': 'Current production config',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X2',
        'description': 'Very shallow trees',
        'philosophy': 'Maximum generalization, stumps only',
        'params': {
            'max_depth': 2,
            'n_estimators': 150,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X3',
        'description': 'Medium depth',
        'philosophy': 'Balance complexity and generalization',
        'params': {
            'max_depth': 5,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X4',
        'description': 'Deep trees',
        'philosophy': 'Capture complex feature interactions',
        'params': {
            'max_depth': 7,
            'n_estimators': 80,
            'learning_rate': 0.08,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X5',
        'description': 'Very deep trees',
        'philosophy': 'Risk overfitting for complex patterns',
        'params': {
            'max_depth': 10,
            'n_estimators': 50,
            'learning_rate': 0.05,
            'min_child_weight': 3,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        }
    },

    # ----- LEARNING RATE & REGULARIZATION (X6-X9) -----
    {
        'name': 'X6',
        'description': 'Slow learner, many trees',
        'philosophy': 'Patient learning, less overfit',
        'params': {
            'max_depth': 3,
            'n_estimators': 500,
            'learning_rate': 0.02,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X7',
        'description': 'Fast learner, few trees',
        'philosophy': 'Quick convergence, early stopping friendly',
        'params': {
            'max_depth': 4,
            'n_estimators': 50,
            'learning_rate': 0.3,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X8',
        'description': 'Heavy regularization',
        'philosophy': 'Prevent overfitting at all costs',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 20,
            'gamma': 1.0,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        }
    },
    {
        'name': 'X9',
        'description': 'Light regularization',
        'philosophy': 'Let the data speak, minimal constraints',
        'params': {
            'max_depth': 4,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
        }
    },

    # ----- CONSERVATIVE VS AGGRESSIVE (X10-X14) -----
    {
        'name': 'X10',
        'description': 'Ultra conservative',
        'philosophy': 'Shallow + slow + heavy reg = stable',
        'params': {
            'max_depth': 2,
            'n_estimators': 300,
            'learning_rate': 0.03,
            'min_child_weight': 15,
            'gamma': 0.5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        }
    },
    {
        'name': 'X11',
        'description': 'Aggressive learner',
        'philosophy': 'Deep + fast + light reg = powerful',
        'params': {
            'max_depth': 6,
            'n_estimators': 80,
            'learning_rate': 0.2,
            'min_child_weight': 3,
            'gamma': 0,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
        }
    },
    {
        'name': 'X12',
        'description': 'Feature sparse',
        'philosophy': 'Force feature selection per tree',
        'params': {
            'max_depth': 4,
            'n_estimators': 150,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.5,
        }
    },
    {
        'name': 'X13',
        'description': 'Data sparse (bagging-like)',
        'philosophy': 'Reduce variance via row sampling',
        'params': {
            'max_depth': 4,
            'n_estimators': 150,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'subsample': 0.6,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X14',
        'description': 'Double sparse',
        'philosophy': 'Maximum diversity per tree',
        'params': {
            'max_depth': 4,
            'n_estimators': 200,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
        }
    },

    # ----- L1/L2 REGULARIZATION & CLASS IMBALANCE (X15-X18) -----
    {
        'name': 'X15',
        'description': 'L1 regularized (sparse weights)',
        'philosophy': 'Encourage feature sparsity',
        'params': {
            'max_depth': 4,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'reg_alpha': 1.0,
            'reg_lambda': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X16',
        'description': 'L2 regularized (shrinkage)',
        'philosophy': 'Smooth weight distribution',
        'params': {
            'max_depth': 4,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'reg_alpha': 0,
            'reg_lambda': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X17',
        'description': 'Elastic net (L1 + L2)',
        'philosophy': 'Best of both regularizations',
        'params': {
            'max_depth': 4,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'reg_alpha': 0.5,
            'reg_lambda': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X18',
        'description': 'Class imbalance aware',
        'philosophy': 'Upweight minority class (spikes)',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'scale_pos_weight': 0.45,  # ratio of neg/pos (371/817)
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },

    # ----- PRECISION VS RECALL TRADEOFFS (X19-X22) -----
    {
        'name': 'X19',
        'description': 'Precision focused',
        'philosophy': 'Shallow + heavy reg = fewer false positives',
        'params': {
            'max_depth': 2,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 15,
            'gamma': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X20',
        'description': 'Recall focused',
        'philosophy': 'Deep + fast + light reg = catch more spikes',
        'params': {
            'max_depth': 5,
            'n_estimators': 100,
            'learning_rate': 0.15,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
        }
    },
    {
        'name': 'X21',
        'description': 'Many weak stumps',
        'philosophy': 'AdaBoost-like: many simple rules',
        'params': {
            'max_depth': 1,
            'n_estimators': 500,
            'learning_rate': 0.05,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X22',
        'description': 'Few powerful trees',
        'philosophy': 'Random forest-like: strong individual trees',
        'params': {
            'max_depth': 8,
            'n_estimators': 30,
            'learning_rate': 0.3,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },

    # ----- SAMPLING STRATEGIES (X23-X26) -----
    {
        'name': 'X23',
        'description': 'Gamma pruning',
        'philosophy': 'Require minimum gain for splits',
        'params': {
            'max_depth': 4,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'gamma': 0.5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'X24',
        'description': 'No sampling (full data)',
        'philosophy': 'Use all data every tree',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
        }
    },
    {
        'name': 'X25',
        'description': 'Per-level column sampling',
        'philosophy': 'Different features at each depth',
        'params': {
            'max_depth': 5,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.7,
        }
    },
    {
        'name': 'X26',
        'description': 'Per-node column sampling',
        'philosophy': 'Maximum feature diversity',
        'params': {
            'max_depth': 5,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bynode': 0.7,
        }
    },

    # ----- ENSEMBLE CANDIDATES (X27-X30) -----
    {
        'name': 'X27',
        'description': 'Ensemble A: Shallow conservative',
        'philosophy': 'Stable base for ensemble',
        'params': {
            'max_depth': 2,
            'n_estimators': 200,
            'learning_rate': 0.05,
            'min_child_weight': 10,
            'subsample': 0.7,
            'colsample_bytree': 0.6,
        }
    },
    {
        'name': 'X28',
        'description': 'Ensemble B: Medium balanced',
        'philosophy': 'All-rounder for ensemble',
        'params': {
            'max_depth': 4,
            'n_estimators': 150,
            'learning_rate': 0.08,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_lambda': 2,
        }
    },
    {
        'name': 'X29',
        'description': 'Ensemble C: Deep specialized',
        'philosophy': 'Complex pattern catcher for ensemble',
        'params': {
            'max_depth': 6,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.2,
        }
    },
    {
        'name': 'X30',
        'description': 'Kitchen sink',
        'philosophy': 'All regularization techniques combined',
        'params': {
            'max_depth': 4,
            'n_estimators': 200,
            'learning_rate': 0.05,
            'min_child_weight': 10,
            'gamma': 0.3,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 5,
        }
    },
]


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


def to_python(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def evaluate_model(model, X, y, threshold=0.5):
    """Evaluate model with multiple metrics."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'precision': to_python(precision_score(y, y_pred, zero_division=0)),
        'recall': to_python(recall_score(y, y_pred, zero_division=0)),
        'f1': to_python(f1_score(y, y_pred, zero_division=0)),
        'accuracy': to_python(accuracy_score(y, y_pred)),
        'roc_auc': to_python(roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0),
        'avg_precision': to_python(average_precision_score(y, y_prob) if len(np.unique(y)) > 1 else 0),
    }

    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])

    return metrics


def evaluate_at_thresholds(model, X, y):
    """Evaluate at multiple probability thresholds."""
    y_prob = model.predict_proba(X)[:, 1]

    thresholds = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    results = {}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        results[f't_{t}'] = {
            'precision': to_python(precision_score(y, y_pred, zero_division=0)),
            'recall': to_python(recall_score(y, y_pred, zero_division=0)),
            'f1': to_python(f1_score(y, y_pred, zero_division=0)),
            'n_predictions': int(y_pred.sum()),
        }

    return results


def train_experiment(exp_config: dict, X_train, y_train, X_val, y_val, X_test, y_test,
                     feature_cols, scaler):
    """Train a single experiment and return results."""
    name = exp_config['name']
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {name}: {exp_config['description']}")
    logger.info(f"Philosophy: {exp_config['philosophy']}")

    # Merge base params with experiment params
    params = {**BASE_PARAMS, **exp_config['params']}

    # Create and train model
    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate on all sets
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    # Threshold analysis on test set
    threshold_metrics = evaluate_at_thresholds(model, X_test, y_test)

    # Feature importance
    importance = {k: to_python(v) for k, v in zip(feature_cols, model.feature_importances_)}
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

    logger.info(f"  Train F1: {train_metrics['f1']:.3f} | Val F1: {val_metrics['f1']:.3f} | Test F1: {test_metrics['f1']:.3f}")
    logger.info(f"  Test Precision: {test_metrics['precision']:.3f} | Recall: {test_metrics['recall']:.3f}")
    logger.info(f"  ROC-AUC: {test_metrics['roc_auc']:.3f} | Avg Precision: {test_metrics['avg_precision']:.3f}")

    # Save model
    model_dir = OUTPUT_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols
        }, f)

    # Save params
    params_path = model_dir / 'params.json'
    with open(params_path, 'w') as f:
        json.dump({
            'name': name,
            'description': exp_config['description'],
            'philosophy': exp_config['philosophy'],
            'params': exp_config['params'],
            'all_params': params
        }, f, indent=2)

    # Save detailed metrics
    metrics_path = model_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'thresholds': threshold_metrics,
            'top_features': top_features
        }, f, indent=2)

    return {
        'name': name,
        'description': exp_config['description'],
        'philosophy': exp_config['philosophy'],
        'params': exp_config['params'],
        'train_f1': train_metrics['f1'],
        'val_f1': val_metrics['f1'],
        'test_f1': test_metrics['f1'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_roc_auc': test_metrics['roc_auc'],
        'test_avg_precision': test_metrics['avg_precision'],
        'test_tp': test_metrics.get('tp', 0),
        'test_fp': test_metrics.get('fp', 0),
        'test_fn': test_metrics.get('fn', 0),
        'test_tn': test_metrics.get('tn', 0),
        'overfit_gap': train_metrics['f1'] - test_metrics['f1'],
        'top_feature': top_features[0][0] if top_features else None,
        'threshold_95_precision': threshold_metrics['t_0.95']['precision'],
        'threshold_95_recall': threshold_metrics['t_0.95']['recall'],
        'threshold_80_f1': threshold_metrics['t_0.8']['f1'],
    }


def main():
    """Run all experiments."""
    logger.info("=" * 60)
    logger.info("XGBOOST HYPERPARAMETER EXPERIMENT SUITE")
    logger.info(f"Running {len(EXPERIMENTS)} experiments")
    logger.info("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"\nLoading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"  Positives (spikes): {(df['label'] == 1).sum()}")
    logger.info(f"  Negatives: {(df['label'] == 0).sum()}")

    # Identify feature columns
    meta_cols = ['event_id', 'symbol', 'timestamp', 'label', 'sample_type']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    logger.info(f"Features: {len(feature_cols)}")

    # Temporal split
    train, val, test = temporal_split(df, TRAIN_RATIO, VAL_RATIO)

    logger.info(f"\nData splits:")
    logger.info(f"  Train: {len(train)} ({train['label'].sum()} pos)")
    logger.info(f"  Val: {len(val)} ({val['label'].sum()} pos)")
    logger.info(f"  Test: {len(test)} ({test['label'].sum()} pos)")

    # Prepare features
    X_train = train[feature_cols].values
    y_train = train['label'].values
    X_val = val[feature_cols].values
    y_val = val['label'].values
    X_test = test[feature_cols].values
    y_test = test['label'].values

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Run all experiments
    results = []
    for exp in EXPERIMENTS:
        try:
            result = train_experiment(
                exp,
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                X_test_scaled, y_test,
                feature_cols, scaler
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error in {exp['name']}: {e}")
            results.append({
                'name': exp['name'],
                'error': str(e)
            })

    # Create summary DataFrame
    summary_df = pd.DataFrame(results)

    # Sort by test F1
    if 'test_f1' in summary_df.columns:
        summary_df = summary_df.sort_values('test_f1', ascending=False)

    # Save summary
    summary_csv = OUTPUT_DIR / 'summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"\nSaved summary to: {summary_csv}")

    summary_json = OUTPUT_DIR / 'summary.json'
    with open(summary_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved detailed summary to: {summary_json}")

    # Print leaderboard
    logger.info("\n" + "=" * 60)
    logger.info("LEADERBOARD (by Test F1)")
    logger.info("=" * 60)

    valid_results = [r for r in results if 'test_f1' in r]
    sorted_results = sorted(valid_results, key=lambda x: x['test_f1'], reverse=True)

    logger.info(f"{'Rank':<5} {'Model':<6} {'Test F1':<10} {'Precision':<10} {'Recall':<10} {'Overfit Gap':<12} Description")
    logger.info("-" * 100)

    for i, r in enumerate(sorted_results[:10], 1):
        logger.info(
            f"{i:<5} {r['name']:<6} {r['test_f1']:.3f}      {r['test_precision']:.3f}      "
            f"{r['test_recall']:.3f}      {r['overfit_gap']:+.3f}        {r['description'][:40]}"
        )

    # Highlight insights
    logger.info("\n" + "=" * 60)
    logger.info("KEY INSIGHTS")
    logger.info("=" * 60)

    if sorted_results:
        best = sorted_results[0]
        logger.info(f"\nBest model: {best['name']} ({best['description']})")
        logger.info(f"  Test F1: {best['test_f1']:.3f}")
        logger.info(f"  Philosophy: {best['philosophy']}")

        # Find least overfit
        least_overfit = min(valid_results, key=lambda x: abs(x['overfit_gap']))
        logger.info(f"\nMost generalizable: {least_overfit['name']} (gap: {least_overfit['overfit_gap']:+.3f})")

        # Find best precision at 95% threshold
        best_precision_95 = max(valid_results, key=lambda x: x.get('threshold_95_precision', 0))
        logger.info(f"Best precision @0.95: {best_precision_95['name']} ({best_precision_95['threshold_95_precision']:.3f})")

        # Find best recall
        best_recall = max(valid_results, key=lambda x: x['test_recall'])
        logger.info(f"Best recall: {best_recall['name']} ({best_recall['test_recall']:.3f})")

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Models saved to: {OUTPUT_DIR}")
    logger.info("=" * 60)

    return summary_df


if __name__ == "__main__":
    main()
