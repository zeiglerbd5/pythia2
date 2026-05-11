#!/usr/bin/env python3
"""
Document individual spike events from model predictions.
Creates a summary of true positives and false negatives with:
- Detection timestamp
- Peak timestamp and time to peak
- Percentage increase
"""

import argparse
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import torch
import pickle
import sys
import os

def load_model_and_data(model_dir, data_file):
    """Load the trained model and test dataset"""

    print("Loading model and data...")

    # Load data
    data = np.load(data_file)

    # Get test split
    n_samples = data['X'].shape[0]
    test_size = int(0.15 * n_samples)
    test_start = n_samples - test_size

    X_test = data['X'][test_start:]
    y_test = data['y'][test_start:]
    timestamps = data['timestamps'][test_start:]
    symbols = data['symbols'][test_start:]

    print(f"Test set: {len(X_test)} samples")
    print(f"Positive samples: {y_test.sum()}")

    # Load model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = f'{model_dir}/best_model.pt'
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Load scaler
    scaler_path = f'{model_dir}/scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Make predictions
    print("Making predictions...")
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        y_pred_prob = model(X_test_tensor).cpu().numpy().flatten()

    y_pred_binary = (y_pred_prob > 0.5).astype(int)

    # Identify true positives and false negatives
    true_positives = (y_pred_binary == 1) & (y_test == 1)
    false_negatives = (y_pred_binary == 0) & (y_test == 1)

    print(f"\nTrue Positives: {true_positives.sum()}")
    print(f"False Negatives: {false_negatives.sum()}")

    return {
        'symbols': symbols,
        'timestamps': timestamps,
        'y_test': y_test,
        'y_pred_prob': y_pred_prob,
        'y_pred_binary': y_pred_binary,
        'true_positives': true_positives,
        'false_negatives': false_negatives
    }

def get_spike_details(db_path, symbol, timestamp, lookahead_minutes=10):
    """
    Get details of a spike event:
    - Peak price and time to peak
    - Percentage increase
    - Start price
    """
    conn = sqlite3.connect(db_path)

    start_time = timestamp
    end_time = (datetime.fromisoformat(timestamp) + timedelta(minutes=lookahead_minutes)).isoformat()

    query = """
    SELECT
        timestamp,
        mid_price
    FROM order_book_features
    WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
    ORDER BY timestamp
    """

    df = pd.read_sql(query, conn, params=(symbol, start_time, end_time))
    conn.close()

    if len(df) == 0:
        return None

    start_price = df['mid_price'].iloc[0]
    peak_price = df['mid_price'].max()
    peak_idx = df['mid_price'].idxmax()
    peak_timestamp = df['timestamp'].iloc[peak_idx]

    # Calculate time to peak in seconds
    start_dt = datetime.fromisoformat(start_time)
    peak_dt = datetime.fromisoformat(peak_timestamp)
    time_to_peak_seconds = (peak_dt - start_dt).total_seconds()

    price_change_pct = ((peak_price - start_price) / start_price) * 100

    return {
        'symbol': symbol,
        'detection_time': start_time,
        'peak_time': peak_timestamp,
        'time_to_peak_seconds': time_to_peak_seconds,
        'time_to_peak_minutes': time_to_peak_seconds / 60,
        'start_price': start_price,
        'peak_price': peak_price,
        'price_change_pct': price_change_pct
    }

def document_spikes(db_path, model_dir, data_file, output_md, output_json):
    """
    Create documentation for true positive and false negative spike events
    """

    # Load model and predictions
    pred_data = load_model_and_data(model_dir, data_file)

    # Document true positives
    print("\n" + "="*80)
    print("DOCUMENTING TRUE POSITIVES")
    print("="*80)

    tp_events = []
    tp_indices = np.where(pred_data['true_positives'])[0]

    for i, idx in enumerate(tp_indices):
        symbol = pred_data['symbols'][idx]
        timestamp = pred_data['timestamps'][idx]
        pred_prob = pred_data['y_pred_prob'][idx]

        details = get_spike_details(db_path, symbol, timestamp)
        if details:
            details['prediction_probability'] = float(pred_prob)
            details['event_type'] = 'TRUE_POSITIVE'
            tp_events.append(details)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(tp_indices)} true positives...")

    print(f"Documented {len(tp_events)} true positive spikes")

    # Document false negatives
    print("\n" + "="*80)
    print("DOCUMENTING FALSE NEGATIVES (MISSED SPIKES)")
    print("="*80)

    fn_events = []
    fn_indices = np.where(pred_data['false_negatives'])[0]

    for i, idx in enumerate(fn_indices):
        symbol = pred_data['symbols'][idx]
        timestamp = pred_data['timestamps'][idx]
        pred_prob = pred_data['y_pred_prob'][idx]

        details = get_spike_details(db_path, symbol, timestamp)
        if details:
            details['prediction_probability'] = float(pred_prob)
            details['event_type'] = 'FALSE_NEGATIVE'
            fn_events.append(details)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(fn_indices)} false negatives...")

    print(f"Documented {len(fn_events)} false negative (missed) spikes")

    # Create markdown document
    print(f"\nCreating markdown documentation: {output_md}")

    with open(output_md, 'w') as f:
        f.write("# Spike Prediction Analysis\n\n")
        f.write("Analysis of individual spike events from neural network predictions.\n\n")

        # Summary statistics
        f.write("## Summary\n\n")
        f.write(f"- **True Positives**: {len(tp_events)} (correctly predicted spikes)\n")
        f.write(f"- **False Negatives**: {len(fn_events)} (missed spikes)\n")

        if len(tp_events) > 0:
            tp_df = pd.DataFrame(tp_events)
            f.write(f"\n### True Positive Statistics\n")
            f.write(f"- Average price increase: {tp_df['price_change_pct'].mean():.2f}%\n")
            f.write(f"- Median price increase: {tp_df['price_change_pct'].median():.2f}%\n")
            f.write(f"- Max price increase: {tp_df['price_change_pct'].max():.2f}%\n")
            f.write(f"- Average time to peak: {tp_df['time_to_peak_minutes'].mean():.2f} minutes\n")
            f.write(f"- Median time to peak: {tp_df['time_to_peak_minutes'].median():.2f} minutes\n")
            f.write(f"- Average prediction probability: {tp_df['prediction_probability'].mean():.3f}\n")

        if len(fn_events) > 0:
            fn_df = pd.DataFrame(fn_events)
            f.write(f"\n### False Negative Statistics\n")
            f.write(f"- Average price increase: {fn_df['price_change_pct'].mean():.2f}%\n")
            f.write(f"- Median price increase: {fn_df['price_change_pct'].median():.2f}%\n")
            f.write(f"- Max price increase: {fn_df['price_change_pct'].max():.2f}%\n")
            f.write(f"- Average time to peak: {fn_df['time_to_peak_minutes'].mean():.2f} minutes\n")
            f.write(f"- Median time to peak: {fn_df['time_to_peak_minutes'].median():.2f} minutes\n")
            f.write(f"- Average prediction probability: {fn_df['prediction_probability'].mean():.3f}\n")

        # True Positives table
        f.write("\n---\n\n")
        f.write("## True Positives (Correctly Predicted Spikes)\n\n")

        if len(tp_events) > 0:
            # Sort by price change descending
            tp_sorted = sorted(tp_events, key=lambda x: x['price_change_pct'], reverse=True)

            f.write("| # | Symbol | Detection Time | Peak Time | Time to Peak (min) | Price Change % | Pred Prob |\n")
            f.write("|---|--------|----------------|-----------|-------------------|---------------|----------|\n")

            for i, event in enumerate(tp_sorted, 1):
                f.write(f"| {i} | {event['symbol']} | {event['detection_time']} | "
                       f"{event['peak_time']} | {event['time_to_peak_minutes']:.1f} | "
                       f"{event['price_change_pct']:.2f}% | {event['prediction_probability']:.3f} |\n")
        else:
            f.write("*No true positive events found*\n")

        # False Negatives table
        f.write("\n---\n\n")
        f.write("## False Negatives (Missed Spikes)\n\n")

        if len(fn_events) > 0:
            # Sort by price change descending
            fn_sorted = sorted(fn_events, key=lambda x: x['price_change_pct'], reverse=True)

            f.write("| # | Symbol | Detection Time | Peak Time | Time to Peak (min) | Price Change % | Pred Prob |\n")
            f.write("|---|--------|----------------|-----------|-------------------|---------------|----------|\n")

            for i, event in enumerate(fn_sorted, 1):
                f.write(f"| {i} | {event['symbol']} | {event['detection_time']} | "
                       f"{event['peak_time']} | {event['time_to_peak_minutes']:.1f} | "
                       f"{event['price_change_pct']:.2f}% | {event['prediction_probability']:.3f} |\n")
        else:
            f.write("*No false negative events found*\n")

    # Save JSON
    print(f"Saving JSON data: {output_json}")

    results = {
        'summary': {
            'true_positives_count': len(tp_events),
            'false_negatives_count': len(fn_events)
        },
        'true_positives': tp_events,
        'false_negatives': fn_events
    }

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("DOCUMENTATION COMPLETE")
    print("="*80)
    print(f"Markdown: {output_md}")
    print(f"JSON: {output_json}")

def main():
    parser = argparse.ArgumentParser(description='Document spike prediction events')
    parser.add_argument('--db', required=True, help='Path to database')
    parser.add_argument('--model', default='models/orderbook_hf_120sym_27days',
                       help='Path to model directory')
    parser.add_argument('--data', default='data/orderbook_features_120sym_27days.npz',
                       help='Path to data file')
    parser.add_argument('--output-md', default='SPIKE_PREDICTIONS.md',
                       help='Output markdown file')
    parser.add_argument('--output-json', default='data/spike_predictions.json',
                       help='Output JSON file')

    args = parser.parse_args()

    document_spikes(args.db, args.model, args.data, args.output_md, args.output_json)

if __name__ == '__main__':
    main()
