"""
XGBoost Precision-Focused Experiment Suite (P1-P30)

Building on X18's success (scale_pos_weight=0.45), these models
prioritize PRECISION over recall. False positives = stop-loss losses.

Key strategies:
- Lower scale_pos_weight (penalize FPs more heavily)
- Higher min_child_weight (conservative splits)
- Higher gamma (require more gain per split)
- Stronger regularization (L1/L2)
- Shallower trees (less overfitting to noise)

Usage:
    python scripts/train_xgboost_precision.py

Output:
    models/experiments_precision/P{n}/model.pkl
    models/experiments_precision/summary.csv
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
OUTPUT_DIR = Path('/Users/bz/Pythia2/models/experiments_precision')
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Base params
BASE_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'use_label_encoder': False,
    'verbosity': 0
}

# ============================================================================
# PRECISION-FOCUSED MODEL CONFIGURATIONS
# ============================================================================

EXPERIMENTS = [
    # ----- SCALE_POS_WEIGHT EXPLORATION (P1-P6) -----
    # X18 used 0.45, let's go lower to penalize FPs more
    {
        'name': 'P1',
        'description': 'X18 baseline (scale_pos_weight=0.45)',
        'philosophy': 'Reproduce X18 winner as baseline',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'scale_pos_weight': 0.45,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P2',
        'description': 'Lower pos weight (0.35)',
        'philosophy': 'Penalize false positives more',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P3',
        'description': 'Even lower pos weight (0.25)',
        'philosophy': 'Heavy FP penalty',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'scale_pos_weight': 0.25,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P4',
        'description': 'Aggressive FP penalty (0.15)',
        'philosophy': 'Very conservative predictions',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'scale_pos_weight': 0.15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P5',
        'description': 'Extreme FP penalty (0.10)',
        'philosophy': 'Only predict when very confident',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'scale_pos_weight': 0.10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P6',
        'description': 'Moderate pos weight (0.40)',
        'philosophy': 'Slight increase in FP penalty from X18',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'scale_pos_weight': 0.40,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },

    # ----- MIN_CHILD_WEIGHT + SCALE_POS_WEIGHT (P7-P12) -----
    # Higher min_child_weight = more samples needed per leaf = more conservative
    {
        'name': 'P7',
        'description': 'High min_child (10) + low pos_weight',
        'philosophy': 'Conservative splits + FP penalty',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 10,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P8',
        'description': 'Very high min_child (15) + low pos_weight',
        'philosophy': 'Very conservative splits',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 15,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P9',
        'description': 'Extreme min_child (20) + moderate pos_weight',
        'philosophy': 'Only split on very clear patterns',
        'params': {
            'max_depth': 3,
            'n_estimators': 120,
            'learning_rate': 0.1,
            'min_child_weight': 20,
            'scale_pos_weight': 0.40,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P10',
        'description': 'High min_child (12) + aggressive pos_weight',
        'philosophy': 'Double conservative',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 12,
            'scale_pos_weight': 0.25,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P11',
        'description': 'Shallow (depth=2) + high min_child',
        'philosophy': 'Simple rules, conservative',
        'params': {
            'max_depth': 2,
            'n_estimators': 150,
            'learning_rate': 0.1,
            'min_child_weight': 15,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P12',
        'description': 'Very shallow (depth=1) + high min_child',
        'philosophy': 'Stumps only, very simple rules',
        'params': {
            'max_depth': 1,
            'n_estimators': 200,
            'learning_rate': 0.1,
            'min_child_weight': 10,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },

    # ----- GAMMA (MIN SPLIT GAIN) + SCALE_POS_WEIGHT (P13-P18) -----
    # Higher gamma = require more gain to make a split
    {
        'name': 'P13',
        'description': 'Gamma=0.5 + low pos_weight',
        'philosophy': 'Require meaningful gain per split',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'gamma': 0.5,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P14',
        'description': 'Gamma=1.0 + low pos_weight',
        'philosophy': 'High gain threshold',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'gamma': 1.0,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P15',
        'description': 'Gamma=2.0 + low pos_weight',
        'philosophy': 'Very high gain threshold',
        'params': {
            'max_depth': 4,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'gamma': 2.0,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P16',
        'description': 'Gamma + high min_child combo',
        'philosophy': 'Double pruning',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 10,
            'gamma': 0.5,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P17',
        'description': 'Gamma + shallow trees',
        'philosophy': 'Simple but requires clear signal',
        'params': {
            'max_depth': 2,
            'n_estimators': 150,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'gamma': 1.0,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P18',
        'description': 'Triple pruning (gamma + min_child + shallow)',
        'philosophy': 'Maximum conservatism via structure',
        'params': {
            'max_depth': 2,
            'n_estimators': 150,
            'learning_rate': 0.1,
            'min_child_weight': 12,
            'gamma': 0.5,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },

    # ----- L1/L2 REGULARIZATION + PRECISION (P19-P24) -----
    {
        'name': 'P19',
        'description': 'Strong L2 (lambda=10) + low pos_weight',
        'philosophy': 'Shrink weights + FP penalty',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'reg_lambda': 10,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P20',
        'description': 'Very strong L2 (lambda=20) + low pos_weight',
        'philosophy': 'Heavy shrinkage',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'reg_lambda': 20,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P21',
        'description': 'L1 sparsity (alpha=1.0) + low pos_weight',
        'philosophy': 'Feature selection + FP penalty',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'reg_alpha': 1.0,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P22',
        'description': 'Strong L1 (alpha=2.0) + low pos_weight',
        'philosophy': 'Heavy sparsity',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'reg_alpha': 2.0,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P23',
        'description': 'Elastic net (L1+L2) + low pos_weight',
        'philosophy': 'Combined regularization',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'reg_alpha': 0.5,
            'reg_lambda': 10,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P24',
        'description': 'All regularization + low pos_weight',
        'philosophy': 'Maximum regularization',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 10,
            'gamma': 0.5,
            'reg_alpha': 0.5,
            'reg_lambda': 10,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },

    # ----- SLOW LEARNING + PRECISION (P25-P27) -----
    {
        'name': 'P25',
        'description': 'Slow learning + low pos_weight',
        'philosophy': 'Patient + conservative',
        'params': {
            'max_depth': 3,
            'n_estimators': 300,
            'learning_rate': 0.03,
            'min_child_weight': 10,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P26',
        'description': 'Very slow learning + all reg',
        'philosophy': 'Maximum patience + conservatism',
        'params': {
            'max_depth': 3,
            'n_estimators': 500,
            'learning_rate': 0.02,
            'min_child_weight': 10,
            'gamma': 0.3,
            'reg_lambda': 5,
            'scale_pos_weight': 0.35,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P27',
        'description': 'Slow + shallow + conservative',
        'philosophy': 'Simple slow learner',
        'params': {
            'max_depth': 2,
            'n_estimators': 400,
            'learning_rate': 0.02,
            'min_child_weight': 15,
            'scale_pos_weight': 0.30,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },

    # ----- BEST COMBINATIONS (P28-P30) -----
    {
        'name': 'P28',
        'description': 'Precision champion attempt A',
        'philosophy': 'Best of depth=2, gamma, min_child, pos_weight',
        'params': {
            'max_depth': 2,
            'n_estimators': 150,
            'learning_rate': 0.08,
            'min_child_weight': 12,
            'gamma': 0.5,
            'reg_lambda': 5,
            'scale_pos_weight': 0.30,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P29',
        'description': 'Precision champion attempt B',
        'philosophy': 'Best of depth=3, heavy reg, aggressive pos_weight',
        'params': {
            'max_depth': 3,
            'n_estimators': 120,
            'learning_rate': 0.08,
            'min_child_weight': 10,
            'gamma': 0.3,
            'reg_alpha': 0.3,
            'reg_lambda': 8,
            'scale_pos_weight': 0.25,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    {
        'name': 'P30',
        'description': 'Precision champion attempt C',
        'philosophy': 'Balanced precision focus',
        'params': {
            'max_depth': 3,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'min_child_weight': 8,
            'gamma': 0.3,
            'reg_lambda': 10,
            'scale_pos_weight': 0.35,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        }
    },
]


def to_python(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def temporal_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    """Split data with stratification to ensure both classes in all splits."""
    from sklearn.model_selection import train_test_split

    # First split: train vs (val+test)
    test_val_ratio = 1 - train_ratio
    train, val_test = train_test_split(
        df, test_size=test_val_ratio, stratify=df['label'], random_state=42
    )

    # Second split: val vs test (from remaining data)
    val_fraction = val_ratio / test_val_ratio
    val, test = train_test_split(
        val_test, test_size=(1 - val_fraction), stratify=val_test['label'], random_state=42
    )

    return train, val, test


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
    """Evaluate at multiple probability thresholds - focus on high thresholds for precision."""
    y_prob = model.predict_proba(X)[:, 1]

    # Focus on high thresholds since we care about precision
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    results = {}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        fn = ((y_pred == 0) & (y == 1)).sum()

        precision = to_python(precision_score(y, y_pred, zero_division=0))
        recall = to_python(recall_score(y, y_pred, zero_division=0))

        results[f't_{t}'] = {
            'precision': precision,
            'recall': recall,
            'f1': to_python(f1_score(y, y_pred, zero_division=0)),
            'n_predictions': int(y_pred.sum()),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
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

    logger.info(f"  Test Precision: {test_metrics['precision']:.3f} | Recall: {test_metrics['recall']:.3f} | F1: {test_metrics['f1']:.3f}")
    logger.info(f"  @0.8 threshold: Prec={threshold_metrics['t_0.8']['precision']:.3f}, Recall={threshold_metrics['t_0.8']['recall']:.3f}, FP={threshold_metrics['t_0.8']['fp']}")
    logger.info(f"  @0.9 threshold: Prec={threshold_metrics['t_0.9']['precision']:.3f}, Recall={threshold_metrics['t_0.9']['recall']:.3f}, FP={threshold_metrics['t_0.9']['fp']}")

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
        # Standard metrics
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_roc_auc': test_metrics['roc_auc'],
        'test_tp': test_metrics.get('tp', 0),
        'test_fp': test_metrics.get('fp', 0),
        'test_fn': test_metrics.get('fn', 0),
        # Threshold metrics (key for precision focus)
        'prec_at_80': threshold_metrics['t_0.8']['precision'],
        'recall_at_80': threshold_metrics['t_0.8']['recall'],
        'fp_at_80': threshold_metrics['t_0.8']['fp'],
        'prec_at_90': threshold_metrics['t_0.9']['precision'],
        'recall_at_90': threshold_metrics['t_0.9']['recall'],
        'fp_at_90': threshold_metrics['t_0.9']['fp'],
        'prec_at_95': threshold_metrics['t_0.95']['precision'],
        'recall_at_95': threshold_metrics['t_0.95']['recall'],
        'fp_at_95': threshold_metrics['t_0.95']['fp'],
        # Overfit check
        'train_f1': train_metrics['f1'],
        'overfit_gap': train_metrics['f1'] - test_metrics['f1'],
        'top_feature': top_features[0][0] if top_features else None,
    }


def main():
    """Run all experiments."""
    logger.info("=" * 60)
    logger.info("PRECISION-FOCUSED XGBOOST EXPERIMENTS (P1-P30)")
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
    logger.info(f"  Train: {len(train)} ({train['label'].sum()} pos, {len(train) - train['label'].sum()} neg)")
    logger.info(f"  Val: {len(val)} ({val['label'].sum()} pos)")
    logger.info(f"  Test: {len(test)} ({test['label'].sum()} pos, {len(test) - test['label'].sum()} neg)")

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

    # Sort by precision at 0.8 threshold (our key metric)
    if 'prec_at_80' in summary_df.columns:
        summary_df = summary_df.sort_values('prec_at_80', ascending=False)

    # Save summary
    summary_csv = OUTPUT_DIR / 'summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"\nSaved summary to: {summary_csv}")

    summary_json = OUTPUT_DIR / 'summary.json'
    with open(summary_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved detailed summary to: {summary_json}")

    # Print leaderboard - sorted by precision at 0.8 threshold
    logger.info("\n" + "=" * 60)
    logger.info("LEADERBOARD (by Precision @ 0.8 threshold)")
    logger.info("=" * 60)

    valid_results = [r for r in results if 'prec_at_80' in r]
    sorted_by_prec = sorted(valid_results, key=lambda x: x['prec_at_80'], reverse=True)

    logger.info(f"{'Rank':<5} {'Model':<6} {'Prec@0.8':<10} {'Recall@0.8':<12} {'FP@0.8':<8} {'Test F1':<10} Description")
    logger.info("-" * 100)

    for i, r in enumerate(sorted_by_prec[:15], 1):
        logger.info(
            f"{i:<5} {r['name']:<6} {r['prec_at_80']:.3f}      {r['recall_at_80']:.3f}        "
            f"{r['fp_at_80']:<8} {r['test_f1']:.3f}      {r['description'][:35]}"
        )

    # Also show precision at 0.9 threshold
    logger.info("\n" + "=" * 60)
    logger.info("LEADERBOARD (by Precision @ 0.9 threshold)")
    logger.info("=" * 60)

    sorted_by_prec_90 = sorted(valid_results, key=lambda x: x['prec_at_90'], reverse=True)

    logger.info(f"{'Rank':<5} {'Model':<6} {'Prec@0.9':<10} {'Recall@0.9':<12} {'FP@0.9':<8} Description")
    logger.info("-" * 90)

    for i, r in enumerate(sorted_by_prec_90[:10], 1):
        logger.info(
            f"{i:<5} {r['name']:<6} {r['prec_at_90']:.3f}      {r['recall_at_90']:.3f}        "
            f"{r['fp_at_90']:<8} {r['description'][:40]}"
        )

    # Key insights
    logger.info("\n" + "=" * 60)
    logger.info("KEY INSIGHTS")
    logger.info("=" * 60)

    if sorted_by_prec:
        best = sorted_by_prec[0]
        logger.info(f"\nBest precision @0.8: {best['name']} ({best['description']})")
        logger.info(f"  Precision: {best['prec_at_80']:.3f}")
        logger.info(f"  Recall: {best['recall_at_80']:.3f}")
        logger.info(f"  False Positives: {best['fp_at_80']}")

        # Best with minimum recall threshold
        min_recall = 0.5
        good_recall = [r for r in sorted_by_prec if r['recall_at_80'] >= min_recall]
        if good_recall:
            best_balanced = good_recall[0]
            logger.info(f"\nBest precision with recall >= {min_recall}: {best_balanced['name']}")
            logger.info(f"  Precision: {best_balanced['prec_at_80']:.3f}, Recall: {best_balanced['recall_at_80']:.3f}")

        # Lowest FP count
        lowest_fp = min(valid_results, key=lambda x: x['fp_at_80'])
        logger.info(f"\nLowest FP count @0.8: {lowest_fp['name']} ({lowest_fp['fp_at_80']} FPs)")

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Models saved to: {OUTPUT_DIR}")
    logger.info("=" * 60)

    return summary_df


if __name__ == "__main__":
    main()
