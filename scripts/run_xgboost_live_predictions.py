#!/usr/bin/env python3
"""
Real-Time XGBoost Spike Predictions

Monitors the features table in market_data.duckdb and runs XGBoost predictions
in real-time as new features are written by the integrated collector.

Usage:
    python scripts/run_xgboost_live_predictions.py \\
        --model models/xgboost_slow_large_v1.pkl \\
        --threshold 0.7

Features:
- Monitors features table for new entries every 10 seconds
- Runs XGBoost predictions (F1=77.6%, trained on Oct 6-20 data)
- Logs high-confidence spike predictions (>70% probability)
- Saves predictions to JSON file for analysis

Architecture:
    Database (features table) → XGBoost Model → Alert Log
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import argparse
import json
from datetime import datetime, timedelta
import joblib
import numpy as np
import duckdb
from loguru import logger


class XGBoostLivePredictor:
    """
    Live spike prediction using XGBoost model.

    Polls the features table and runs predictions on new feature vectors.
    """

    def __init__(
        self,
        model_path: str,
        db_path: str = "market_data.duckdb",
        threshold: float = 0.5,
        min_alert_probability: float = 0.7,
        alert_log_file: str = "spike_alerts.json",
        poll_interval: int = 10
    ):
        """
        Initialize live predictor.

        Args:
            model_path: Path to XGBoost model (.pkl)
            db_path: Path to DuckDB database
            threshold: Prediction threshold (default: 0.5)
            min_alert_probability: Minimum probability to log alert (default: 0.7)
            alert_log_file: Path to JSON file for logging alerts
            poll_interval: How often to check for new features (seconds)
        """
        self.db_path = db_path
        self.threshold = threshold
        self.min_alert_probability = min_alert_probability
        self.alert_log_file = alert_log_file
        self.poll_interval = poll_interval

        # Load XGBoost model
        logger.info(f"Loading XGBoost model from {model_path}")
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded successfully")

        # Expected feature columns (from training)
        self.feature_columns = [
            'returns', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14', 'NATR',
            'BB_width', 'BB_squeeze', 'VWAP_distance', 'volume_zscore', 'volume_roc',
            'OBV', 'trade_count', 'buy_sell_ratio', 'roll_measure', 'order_flow_imbalance',
            'vpin', 'bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance',
            'returns_5m', 'volume_zscore_5m', 'returns_15m', 'volume_zscore_15m'
        ]

        # Track last processed timestamp per symbol
        self.last_processed = {}

        # Statistics
        self.predictions_made = 0
        self.spikes_predicted = 0
        self.start_time = datetime.now()

        logger.info("="*80)
        logger.info("XGBOOST LIVE SPIKE PREDICTOR")
        logger.info("="*80)
        logger.info(f"Database: {db_path}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Alert threshold: {min_alert_probability}")
        logger.info(f"Poll interval: {poll_interval}s")
        logger.info(f"Alert log: {alert_log_file}")
        logger.info("="*80)
        logger.info("")

    def run(self):
        """Run live prediction loop."""
        logger.info("Starting live prediction loop...")
        logger.info("Monitoring features table for new entries...")
        logger.info("")

        try:
            while True:
                self._poll_and_predict()
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            logger.info("")
            logger.info("Shutting down...")
            self._print_statistics()

    def _poll_and_predict(self):
        """Poll database for new features and run predictions."""
        # Retry logic for database locks (DuckDB doesn't support concurrent readers/writers well)
        max_retries = 3
        retry_delay = 0.5  # seconds

        for attempt in range(max_retries):
            try:
                conn = duckdb.connect(self.db_path, read_only=True)

                # Get latest features for each symbol
                # Only get features from the last 5 minutes to avoid processing old data
                query = """
                    WITH latest AS (
                        SELECT
                            symbol,
                            MAX(timestamp) as max_timestamp
                        FROM features
                        WHERE timestamp >= NOW() - INTERVAL '5 minutes'
                        GROUP BY symbol
                    )
                    SELECT f.*
                    FROM features f
                    INNER JOIN latest l
                        ON f.symbol = l.symbol AND f.timestamp = l.max_timestamp
                    WHERE f.timeframe = '5m'
                    ORDER BY f.timestamp DESC
                """

                result = conn.execute(query).fetchdf()
                conn.close()

                if len(result) == 0:
                    return

                # Filter to only new timestamps we haven't processed
                new_rows = []
                for _, row in result.iterrows():
                    symbol = row['symbol']
                    timestamp = row['timestamp']

                    if symbol not in self.last_processed or timestamp > self.last_processed[symbol]:
                        new_rows.append(row)
                        self.last_processed[symbol] = timestamp

                if len(new_rows) == 0:
                    return

                logger.info(f"Found {len(new_rows)} new feature vectors to process")

                # Run predictions on new rows
                for row in new_rows:
                    self._predict_and_alert(row)

                break  # Success, exit retry loop

            except Exception as e:
                if "Could not set lock" in str(e) and attempt < max_retries - 1:
                    # Database is locked, retry after delay
                    time.sleep(retry_delay)
                    continue
                else:
                    # Other error or max retries reached
                    if attempt == max_retries - 1:
                        logger.error(f"Error polling features after {max_retries} retries: {e}")
                    else:
                        logger.error(f"Error polling features: {e}")
                    break

    def _predict_and_alert(self, row):
        """Run prediction on a single feature vector and alert if spike predicted."""
        try:
            symbol = row['symbol']
            timestamp = row['timestamp']
            timeframe = row['timeframe']

            # Extract features in correct order
            features = []
            for col in self.feature_columns:
                val = row.get(col, 0.0)
                # Handle NaN/None
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                features.append(val)

            features_array = np.array(features).reshape(1, -1)

            # Run prediction
            prob = self.model.predict_proba(features_array)[0][1]
            pred = 1 if prob >= self.threshold else 0

            self.predictions_made += 1

            # Log if spike predicted with high confidence
            if prob >= self.min_alert_probability:
                self.spikes_predicted += 1

                alert = {
                    'timestamp': str(timestamp),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'probability': float(prob),
                    'prediction': int(pred),
                    'alert_time': str(datetime.now())
                }

                logger.warning(
                    f"🔔 SPIKE ALERT: {symbol} | "
                    f"Probability: {prob:.1%} | "
                    f"Time: {timestamp}"
                )

                # Append to alert log
                self._save_alert(alert)

            # Periodic status every 100 predictions
            if self.predictions_made % 100 == 0:
                logger.info(
                    f"Predictions: {self.predictions_made} | "
                    f"Spikes: {self.spikes_predicted} | "
                    f"Rate: {self.spikes_predicted/self.predictions_made*100:.1f}%"
                )

        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")

    def _save_alert(self, alert: dict):
        """Append alert to JSON log file."""
        try:
            # Load existing alerts
            alerts = []
            if Path(self.alert_log_file).exists():
                with open(self.alert_log_file, 'r') as f:
                    alerts = json.load(f)

            # Append new alert
            alerts.append(alert)

            # Save back
            with open(self.alert_log_file, 'w') as f:
                json.dump(alerts, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving alert: {e}")

    def _print_statistics(self):
        """Print prediction statistics."""
        uptime = (datetime.now() - self.start_time).total_seconds()

        logger.info("="*80)
        logger.info("PREDICTION STATISTICS")
        logger.info("="*80)
        logger.info(f"Uptime: {uptime:.0f}s ({uptime/60:.1f} minutes)")
        logger.info(f"Predictions made: {self.predictions_made}")
        logger.info(f"Spikes predicted: {self.spikes_predicted}")
        if self.predictions_made > 0:
            logger.info(f"Spike rate: {self.spikes_predicted/self.predictions_made*100:.1f}%")
        logger.info(f"Symbols tracked: {len(self.last_processed)}")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Real-time XGBoost spike predictions')

    parser.add_argument('--model', default='models/xgboost_slow_large_v1.pkl',
                       help='Path to XGBoost model (.pkl)')
    parser.add_argument('--db', default='market_data.duckdb',
                       help='Path to DuckDB database')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold')
    parser.add_argument('--min-probability', type=float, default=0.7,
                       help='Minimum probability to log alert')
    parser.add_argument('--alert-log', default='spike_alerts.json',
                       help='Path to alert log JSON file')
    parser.add_argument('--poll-interval', type=int, default=10,
                       help='Poll interval in seconds')

    args = parser.parse_args()

    # Validate model exists
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        return

    # Validate database exists
    if not Path(args.db).exists():
        logger.error(f"Database file not found: {args.db}")
        return

    # Create predictor
    predictor = XGBoostLivePredictor(
        model_path=args.model,
        db_path=args.db,
        threshold=args.threshold,
        min_alert_probability=args.min_probability,
        alert_log_file=args.alert_log,
        poll_interval=args.poll_interval
    )

    # Run
    predictor.run()


if __name__ == "__main__":
    main()
