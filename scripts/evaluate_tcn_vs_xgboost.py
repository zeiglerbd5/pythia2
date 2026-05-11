#!/usr/bin/env python3
"""
Evaluate XGBoost-Guided TCN vs XGBoost V3

Compares performance on test period (Dec 1-24):
- Precision/Recall at various thresholds
- Detection rates by spike type
- Head-to-head on same spikes

Usage:
    python scripts/evaluate_tcn_vs_xgboost.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import duckdb
import joblib
from loguru import logger
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import RobustScaler

from src.models.xgboost_guided_tcn import XGBoostGuidedTCN, FEATURE_COLUMNS


# Configuration
CONFIG = {
    'db_path': str(PROJECT_ROOT / 'market_data.duckdb'),
    'tcn_model_path': str(PROJECT_ROOT / 'models/xgboost_guided_tcn_v1.pt'),
    'xgb_model_path': str(PROJECT_ROOT / 'models/xgboost_slow_large_v3.pkl'),
    'test_start': '2024-12-01',
    'test_end': '2024-12-24',
    'sequence_length': 30,
    'device': 'mps'
}


def load_tcn_model(model_path: str, device: str) -> Tuple[XGBoostGuidedTCN, dict, RobustScaler]:
    """Load trained TCN model."""
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint['config']
    model = XGBoostGuidedTCN(
        n_features=config['n_features'],
        sequence_length=config['sequence_length'],
        xgboost_importances=checkpoint.get('xgb_importances'),
        num_channels=config['num_channels'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Restore scaler
    scaler = RobustScaler()
    scaler.center_ = np.array(checkpoint['scaler_state']['center_'])
    scaler.scale_ = np.array(checkpoint['scaler_state']['scale_'])

    logger.info(f"Loaded TCN model from {model_path}")
    return model, checkpoint, scaler


def load_xgboost_model(model_path: str) -> object:
    """Load XGBoost model."""
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict):
        return model_data.get('model', model_data)
    return model_data


def load_test_data(
    db_path: str,
    start_date: str,
    end_date: str,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Load test period features."""
    conn = duckdb.connect(db_path, read_only=True)

    try:
        cols_str = ', '.join(['timestamp', 'symbol'] + feature_cols)
        query = f"""
            SELECT {cols_str}
            FROM features
            WHERE timeframe = '1m'
              AND timestamp >= '{start_date}'
              AND timestamp <= '{end_date}'
            ORDER BY symbol, timestamp ASC
        """
        df = conn.execute(query).fetchdf()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"Loaded {len(df)} feature rows for test period")
        return df

    finally:
        conn.close()


def load_test_candles(
    db_path: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Load candles for spike labeling."""
    conn = duckdb.connect(db_path, read_only=True)

    try:
        query = f"""
            SELECT symbol, timestamp, close, volume
            FROM ohlcv
            WHERE timeframe = '1m'
              AND timestamp >= '{start_date}'
              AND timestamp <= '{end_date}'
            ORDER BY symbol, timestamp ASC
        """
        df = conn.execute(query).fetchdf()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    finally:
        conn.close()


def generate_spike_labels(candles_df: pd.DataFrame) -> pd.DataFrame:
    """Generate spike labels for test period."""
    results = []

    for symbol in candles_df['symbol'].unique():
        sym_df = candles_df[candles_df['symbol'] == symbol].copy()
        sym_df = sym_df.sort_values('timestamp')

        prices = sym_df['close'].values
        volumes = sym_df['volume'].values
        timestamps = sym_df['timestamp'].values

        n = len(prices)
        baseline_vol = pd.Series(volumes).rolling(60, min_periods=30).mean().values

        for i in range(n - 75):
            # Future window: 16-76 minutes ahead
            start_idx = i + 16
            price_end = i + 76
            vol_end = i + 46

            max_price = prices[start_idx:price_end].max()
            price_return = (max_price - prices[i]) / (prices[i] + 1e-10)

            avg_vol = volumes[start_idx:vol_end].mean()
            baseline = baseline_vol[i] if not np.isnan(baseline_vol[i]) else volumes[:i+1].mean()
            vol_ratio = avg_vol / (baseline + 1e-10)

            is_spike = 1.0 if (price_return >= 0.15 and vol_ratio >= 2.0) else 0.0

            results.append({
                'symbol': symbol,
                'timestamp': timestamps[i],
                'label': is_spike,
                'max_return': price_return,
                'volume_ratio': vol_ratio
            })

    return pd.DataFrame(results)


def get_tcn_predictions(
    model: XGBoostGuidedTCN,
    features_df: pd.DataFrame,
    scaler: RobustScaler,
    sequence_length: int,
    device: str
) -> pd.DataFrame:
    """Get TCN predictions for test data."""
    results = []
    feature_cols = FEATURE_COLUMNS

    for symbol in features_df['symbol'].unique():
        sym_df = features_df[features_df['symbol'] == symbol].copy()
        sym_df = sym_df.sort_values('timestamp')

        if len(sym_df) < sequence_length + 10:
            continue

        X = sym_df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = scaler.transform(X)

        timestamps = sym_df['timestamp'].values

        for i in range(sequence_length, len(X)):
            seq = X[i - sequence_length:i]
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = model(seq_tensor).cpu().item()

            results.append({
                'symbol': symbol,
                'timestamp': timestamps[i],
                'tcn_prob': prob
            })

    return pd.DataFrame(results)


def get_xgboost_predictions(
    model: object,
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """Get XGBoost predictions for test data."""
    results = []
    feature_cols = FEATURE_COLUMNS

    for symbol in features_df['symbol'].unique():
        sym_df = features_df[features_df['symbol'] == symbol].copy()
        sym_df = sym_df.sort_values('timestamp')

        X = sym_df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        probs = model.predict_proba(X)[:, 1]

        for i, (ts, prob) in enumerate(zip(sym_df['timestamp'].values, probs)):
            results.append({
                'symbol': symbol,
                'timestamp': ts,
                'xgb_prob': prob
            })

    return pd.DataFrame(results)


def evaluate_model(
    predictions: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float] = [0.3, 0.5, 0.7, 0.9]
) -> Dict:
    """Calculate evaluation metrics."""
    metrics = {}

    # AUC metrics
    try:
        metrics['auc_roc'] = roc_auc_score(labels, predictions)
        metrics['auc_pr'] = average_precision_score(labels, predictions)
    except Exception:
        metrics['auc_roc'] = 0.0
        metrics['auc_pr'] = 0.0

    # Threshold-based metrics
    for thresh in thresholds:
        preds_binary = (predictions >= thresh).astype(float)
        metrics[f'precision@{thresh}'] = precision_score(labels, preds_binary, zero_division=0)
        metrics[f'recall@{thresh}'] = recall_score(labels, preds_binary, zero_division=0)
        metrics[f'f1@{thresh}'] = f1_score(labels, preds_binary, zero_division=0)
        metrics[f'signals@{thresh}'] = preds_binary.sum()

    return metrics


def compare_models(merged_df: pd.DataFrame) -> Dict:
    """Compare TCN and XGBoost head-to-head."""
    labels = merged_df['label'].values
    tcn_probs = merged_df['tcn_prob'].values
    xgb_probs = merged_df['xgb_prob'].values

    # Get spikes
    spike_mask = labels == 1
    n_spikes = spike_mask.sum()

    comparison = {
        'n_samples': len(labels),
        'n_spikes': int(n_spikes),
        'spike_ratio': n_spikes / len(labels),
    }

    # Detection rates at threshold 0.7
    for thresh in [0.5, 0.7]:
        tcn_detected = (tcn_probs[spike_mask] >= thresh).sum()
        xgb_detected = (xgb_probs[spike_mask] >= thresh).sum()

        comparison[f'tcn_detection@{thresh}'] = tcn_detected / n_spikes if n_spikes > 0 else 0
        comparison[f'xgb_detection@{thresh}'] = xgb_detected / n_spikes if n_spikes > 0 else 0

        # Unique detections
        tcn_only = ((tcn_probs >= thresh) & (xgb_probs < thresh) & spike_mask).sum()
        xgb_only = ((xgb_probs >= thresh) & (tcn_probs < thresh) & spike_mask).sum()
        both = ((tcn_probs >= thresh) & (xgb_probs >= thresh) & spike_mask).sum()

        comparison[f'tcn_unique@{thresh}'] = int(tcn_only)
        comparison[f'xgb_unique@{thresh}'] = int(xgb_only)
        comparison[f'both@{thresh}'] = int(both)

    return comparison


def analyze_by_spike_magnitude(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze detection rates by spike magnitude."""
    spike_df = merged_df[merged_df['label'] == 1].copy()

    if len(spike_df) == 0:
        return pd.DataFrame()

    # Bin by magnitude
    bins = [0.15, 0.25, 0.40, 0.60, 1.0, 5.0]
    labels = ['15-25%', '25-40%', '40-60%', '60-100%', '100%+']
    spike_df['magnitude_bin'] = pd.cut(spike_df['max_return'], bins=bins, labels=labels)

    results = []
    for bin_label in labels:
        bin_df = spike_df[spike_df['magnitude_bin'] == bin_label]
        if len(bin_df) == 0:
            continue

        for thresh in [0.5, 0.7]:
            results.append({
                'magnitude': bin_label,
                'threshold': thresh,
                'count': len(bin_df),
                'tcn_detection': (bin_df['tcn_prob'] >= thresh).mean(),
                'xgb_detection': (bin_df['xgb_prob'] >= thresh).mean(),
                'tcn_wins': ((bin_df['tcn_prob'] >= thresh) & (bin_df['xgb_prob'] < thresh)).sum(),
                'xgb_wins': ((bin_df['xgb_prob'] >= thresh) & (bin_df['tcn_prob'] < thresh)).sum(),
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate TCN vs XGBoost')
    parser.add_argument('--tcn-path', type=str, default=CONFIG['tcn_model_path'])
    parser.add_argument('--xgb-path', type=str, default=CONFIG['xgb_model_path'])
    parser.add_argument('--device', type=str, default=CONFIG['device'])
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TCN vs XGBoost Evaluation")
    logger.info("=" * 80)

    # Check if TCN model exists
    if not Path(args.tcn_path).exists():
        logger.error(f"TCN model not found at {args.tcn_path}")
        logger.info("Run train_xgboost_guided_tcn.py first")
        return

    # Setup device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    logger.info(f"Using device: {device}")

    # Load models
    logger.info("\n=== Loading Models ===")
    tcn_model, tcn_checkpoint, scaler = load_tcn_model(args.tcn_path, device)
    xgb_model = load_xgboost_model(args.xgb_path)

    # Load test data
    logger.info("\n=== Loading Test Data ===")
    features_df = load_test_data(
        CONFIG['db_path'],
        CONFIG['test_start'],
        CONFIG['test_end'],
        FEATURE_COLUMNS
    )
    candles_df = load_test_candles(
        CONFIG['db_path'],
        CONFIG['test_start'],
        CONFIG['test_end']
    )

    # Generate spike labels
    logger.info("\n=== Generating Spike Labels ===")
    labels_df = generate_spike_labels(candles_df)
    n_spikes = labels_df['label'].sum()
    logger.info(f"Found {n_spikes:.0f} spikes in test period")

    # Get predictions
    logger.info("\n=== Getting Predictions ===")

    logger.info("TCN predictions...")
    tcn_preds = get_tcn_predictions(
        tcn_model, features_df, scaler,
        CONFIG['sequence_length'], device
    )

    logger.info("XGBoost predictions...")
    xgb_preds = get_xgboost_predictions(xgb_model, features_df)

    # Merge predictions with labels
    merged = labels_df.merge(
        tcn_preds, on=['symbol', 'timestamp'], how='inner'
    ).merge(
        xgb_preds, on=['symbol', 'timestamp'], how='inner'
    )

    logger.info(f"Merged: {len(merged)} samples with both predictions")

    # Evaluate each model
    logger.info("\n=== Model Evaluation ===")

    tcn_metrics = evaluate_model(merged['tcn_prob'].values, merged['label'].values)
    xgb_metrics = evaluate_model(merged['xgb_prob'].values, merged['label'].values)

    print("\n" + "=" * 60)
    print("METRICS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'TCN':>15} {'XGBoost':>15}")
    print("-" * 60)

    for key in sorted(tcn_metrics.keys()):
        tcn_val = tcn_metrics[key]
        xgb_val = xgb_metrics[key]

        if 'signals' in key:
            print(f"{key:<25} {tcn_val:>15.0f} {xgb_val:>15.0f}")
        else:
            winner = ""
            if tcn_val > xgb_val * 1.05:
                winner = " ← TCN"
            elif xgb_val > tcn_val * 1.05:
                winner = " ← XGB"
            print(f"{key:<25} {tcn_val:>15.4f} {xgb_val:>15.4f}{winner}")

    # Head-to-head comparison
    comparison = compare_models(merged)

    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 60)
    print(f"Total samples: {comparison['n_samples']:,}")
    print(f"Total spikes: {comparison['n_spikes']}")
    print()

    for thresh in [0.5, 0.7]:
        print(f"At threshold {thresh}:")
        print(f"  TCN detection rate: {comparison[f'tcn_detection@{thresh}']:.1%}")
        print(f"  XGB detection rate: {comparison[f'xgb_detection@{thresh}']:.1%}")
        print(f"  TCN-only detections: {comparison[f'tcn_unique@{thresh}']}")
        print(f"  XGB-only detections: {comparison[f'xgb_unique@{thresh}']}")
        print(f"  Both detected: {comparison[f'both@{thresh}']}")
        print()

    # Analyze by magnitude
    magnitude_df = analyze_by_spike_magnitude(merged)

    if len(magnitude_df) > 0:
        print("=" * 60)
        print("DETECTION BY SPIKE MAGNITUDE")
        print("=" * 60)

        for thresh in [0.7]:
            print(f"\nAt threshold {thresh}:")
            thresh_df = magnitude_df[magnitude_df['threshold'] == thresh]
            for _, row in thresh_df.iterrows():
                winner = "TCN" if row['tcn_detection'] > row['xgb_detection'] else "XGB"
                print(f"  {row['magnitude']}: n={row['count']}, "
                      f"TCN={row['tcn_detection']:.1%}, XGB={row['xgb_detection']:.1%} "
                      f"[{winner}]")

    # Save results
    results = {
        'config': CONFIG,
        'tcn_metrics': tcn_metrics,
        'xgb_metrics': xgb_metrics,
        'comparison': comparison,
        'evaluated_at': datetime.now().isoformat()
    }

    output_path = PROJECT_ROOT / 'tcn_vs_xgboost_evaluation.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
