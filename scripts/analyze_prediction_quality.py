#!/usr/bin/env python3
"""
Analyze the quality of spike predictions from Experiment 3.
Compare true positives vs false positives to find distinguishing characteristics.
"""

import argparse
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

def load_model_predictions(model_dir):
    """Load predictions from model results"""
    import torch

    # Load test predictions
    results_file = f'{model_dir}/results.json'
    with open(results_file, 'r') as f:
        results = json.load(f)

    return results

def get_spike_characteristics(db_path, symbol, timestamp, lookahead_minutes=10):
    """
    Get characteristics of a spike event:
    - Price movement stats
    - Volume profile
    - Order book imbalance
    - Volatility
    """
    conn = sqlite3.connect(db_path)

    # Get order book data for the spike window
    end_time = (datetime.fromisoformat(timestamp) + timedelta(minutes=lookahead_minutes)).isoformat()

    query = """
    SELECT
        timestamp,
        mid_price,
        spread,
        spread_bps,
        bid_depth,
        ask_depth,
        depth_imbalance,
        ofi,
        volume_imbalance,
        trade_volume,
        buy_volume,
        sell_volume,
        price_volatility,
        price_momentum,
        volume_momentum
    FROM order_book_features
    WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
    ORDER BY timestamp
    """

    df = pd.read_sql(query, conn, params=(symbol, timestamp, end_time))
    conn.close()

    if len(df) == 0:
        return None

    # Calculate spike characteristics
    price_change_pct = ((df['mid_price'].iloc[-1] - df['mid_price'].iloc[0]) /
                        df['mid_price'].iloc[0] * 100)

    characteristics = {
        'symbol': symbol,
        'start_time': timestamp,
        'price_change_pct': price_change_pct,
        'max_price_change': ((df['mid_price'].max() - df['mid_price'].iloc[0]) /
                            df['mid_price'].iloc[0] * 100),
        'time_to_peak_seconds': df['mid_price'].idxmax() * 10 if len(df) > 1 else 0,
        'avg_spread_bps': df['spread_bps'].mean(),
        'max_spread_bps': df['spread_bps'].max(),
        'avg_depth_imbalance': df['depth_imbalance'].mean(),
        'max_depth_imbalance': df['depth_imbalance'].abs().max(),
        'avg_ofi': df['ofi'].mean(),
        'max_ofi': df['ofi'].abs().max(),
        'avg_volume_imbalance': df['volume_imbalance'].mean(),
        'total_volume': df['trade_volume'].sum(),
        'buy_sell_ratio': (df['buy_volume'].sum() / df['sell_volume'].sum()
                          if df['sell_volume'].sum() > 0 else 0),
        'avg_volatility': df['price_volatility'].mean(),
        'max_volatility': df['price_volatility'].max(),
        'momentum_acceleration': (df['price_momentum'].iloc[-1] - df['price_momentum'].iloc[0]
                                 if len(df) > 1 else 0),
    }

    return characteristics

def analyze_predictions(db_path, model_dir, output_file):
    """
    Analyze prediction quality:
    - True Positives (correctly predicted spikes)
    - False Positives (predicted spike but no spike occurred)
    - False Negatives (missed spikes)
    - True Negatives (correctly predicted no spike)
    """

    print("Loading model predictions...")
    results = load_model_predictions(model_dir)

    print(f"\nConfusion Matrix:")
    print(f"True Negatives:  {results['confusion_matrix'][0][0]}")
    print(f"False Positives: {results['confusion_matrix'][0][1]}")
    print(f"False Negatives: {results['confusion_matrix'][1][0]}")
    print(f"True Positives:  {results['confusion_matrix'][1][1]}")

    # Load test data to get timestamps and symbols
    data = np.load('data/orderbook_features_120sym_27days.npz')

    # Get test indices
    n_samples = data['X'].shape[0]
    test_size = int(0.15 * n_samples)
    val_size = int(0.15 * n_samples)

    test_start = n_samples - test_size

    X_test = data['X'][test_start:]
    y_test = data['y'][test_start:]
    timestamps = data['timestamps'][test_start:]
    symbols = data['symbols'][test_start:]

    # Get predictions
    predictions_path = f'{model_dir}/test_predictions.npy'
    try:
        y_pred = np.load(predictions_path)
    except:
        print("Warning: Could not load test_predictions.npy")
        return

    # Threshold predictions at 0.5
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Categorize predictions
    true_positives = (y_pred_binary == 1) & (y_test == 1)
    false_positives = (y_pred_binary == 1) & (y_test == 0)
    false_negatives = (y_pred_binary == 0) & (y_test == 1)
    true_negatives = (y_pred_binary == 0) & (y_test == 0)

    print(f"\nAnalyzing {true_positives.sum()} True Positives...")
    tp_chars = []
    for i in np.where(true_positives)[0][:50]:  # Sample first 50
        chars = get_spike_characteristics(db_path, symbols[i], timestamps[i])
        if chars:
            chars['prediction_prob'] = float(y_pred[i])
            tp_chars.append(chars)

    print(f"Analyzing {false_positives.sum()} False Positives...")
    fp_chars = []
    for i in np.where(false_positives)[0][:50]:  # Sample first 50
        chars = get_spike_characteristics(db_path, symbols[i], timestamps[i])
        if chars:
            chars['prediction_prob'] = float(y_pred[i])
            fp_chars.append(chars)

    # Convert to DataFrames for analysis
    tp_df = pd.DataFrame(tp_chars)
    fp_df = pd.DataFrame(fp_chars)

    # Statistical comparison
    print("\n" + "="*80)
    print("TRUE POSITIVES vs FALSE POSITIVES COMPARISON")
    print("="*80)

    numeric_cols = ['price_change_pct', 'max_price_change', 'time_to_peak_seconds',
                   'avg_spread_bps', 'max_spread_bps', 'avg_depth_imbalance',
                   'max_depth_imbalance', 'avg_ofi', 'max_ofi', 'total_volume',
                   'buy_sell_ratio', 'avg_volatility', 'max_volatility',
                   'momentum_acceleration', 'prediction_prob']

    comparison = pd.DataFrame({
        'Feature': numeric_cols,
        'TP_Mean': [tp_df[col].mean() if col in tp_df else np.nan for col in numeric_cols],
        'FP_Mean': [fp_df[col].mean() if col in fp_df else np.nan for col in numeric_cols],
        'TP_Median': [tp_df[col].median() if col in tp_df else np.nan for col in numeric_cols],
        'FP_Median': [fp_df[col].median() if col in fp_df else np.nan for col in numeric_cols],
    })

    comparison['Difference'] = comparison['TP_Mean'] - comparison['FP_Mean']
    comparison['Pct_Difference'] = ((comparison['TP_Mean'] - comparison['FP_Mean']) /
                                    comparison['FP_Mean'] * 100)

    print("\n" + comparison.to_string(index=False))

    # Top distinguishing features
    print("\n" + "="*80)
    print("TOP DISTINGUISHING FEATURES (by % difference)")
    print("="*80)
    top_features = comparison.sort_values('Pct_Difference', key=abs, ascending=False).head(10)
    print(top_features[['Feature', 'TP_Mean', 'FP_Mean', 'Pct_Difference']].to_string(index=False))

    # Save detailed results
    results_dict = {
        'true_positives': tp_chars,
        'false_positives': fp_chars,
        'comparison': comparison.to_dict('records'),
        'summary': {
            'tp_count': len(tp_chars),
            'fp_count': len(fp_chars),
            'tp_avg_price_change': float(tp_df['price_change_pct'].mean() if 'price_change_pct' in tp_df else 0),
            'fp_avg_price_change': float(fp_df['price_change_pct'].mean() if 'price_change_pct' in fp_df else 0),
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    # Potential filtering rules
    print("\n" + "="*80)
    print("POTENTIAL FILTERING RULES TO REDUCE FALSE POSITIVES")
    print("="*80)

    if 'price_change_pct' in tp_df and 'price_change_pct' in fp_df:
        tp_price_10th = tp_df['price_change_pct'].quantile(0.10)
        fp_price_90th = fp_df['price_change_pct'].quantile(0.90)

        print(f"\n1. Price Change Threshold:")
        print(f"   - True Positives 10th percentile: {tp_price_10th:.2f}%")
        print(f"   - False Positives 90th percentile: {fp_price_90th:.2f}%")
        if tp_price_10th > fp_price_90th:
            print(f"   → Require price_change > {fp_price_90th:.2f}% to filter FPs")

    if 'total_volume' in tp_df and 'total_volume' in fp_df:
        tp_vol_10th = tp_df['total_volume'].quantile(0.10)
        fp_vol_90th = fp_df['total_volume'].quantile(0.90)

        print(f"\n2. Volume Threshold:")
        print(f"   - True Positives 10th percentile: {tp_vol_10th:.2f}")
        print(f"   - False Positives 90th percentile: {fp_vol_90th:.2f}")
        if tp_vol_10th > fp_vol_90th:
            print(f"   → Require volume > {fp_vol_90th:.2f} to filter FPs")

    if 'buy_sell_ratio' in tp_df and 'buy_sell_ratio' in fp_df:
        tp_ratio_median = tp_df['buy_sell_ratio'].median()
        fp_ratio_median = fp_df['buy_sell_ratio'].median()

        print(f"\n3. Buy/Sell Ratio:")
        print(f"   - True Positives median: {tp_ratio_median:.2f}")
        print(f"   - False Positives median: {fp_ratio_median:.2f}")
        if tp_ratio_median > fp_ratio_median * 1.2:
            print(f"   → Require buy/sell ratio > {fp_ratio_median * 1.2:.2f} to filter FPs")

def main():
    parser = argparse.ArgumentParser(description='Analyze spike prediction quality')
    parser.add_argument('--db', required=True, help='Path to database')
    parser.add_argument('--model', default='models/orderbook_hf_120sym_27days',
                       help='Path to model directory')
    parser.add_argument('--output', default='data/prediction_analysis.json',
                       help='Output JSON file')

    args = parser.parse_args()

    analyze_predictions(args.db, args.model, args.output)

if __name__ == '__main__':
    main()
