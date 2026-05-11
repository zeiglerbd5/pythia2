#!/usr/bin/env python3
"""
Real-Time Slow & Large Spike Detector

Polls your database for latest features and runs XGBoost predictions.
Alerts when P(spike) exceeds threshold.

Usage:
    # Run continuously
    python spike_detector_live.py
    
    # Test mode (single pass, no loop)
    python spike_detector_live.py --test
    
    # Custom threshold
    python spike_detector_live.py --threshold 0.7
"""

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
import duckdb
import joblib
from loguru import logger

# Optional: Desktop notifications (install with: pip install plyer)
try:
    from plyer import notification
    HAS_NOTIFICATIONS = True
except ImportError:
    HAS_NOTIFICATIONS = False

# Optional: Sound alerts (install with: pip install playsound)
try:
    from playsound import playsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    'db_path': '/Users/brettzeigler/Pythia/market_data.duckdb',
    'model_path': '/Users/brettzeigler/Pythia/models/xgboost_slow_large_v1.pkl',
    'alert_log': '/Users/brettzeigler/Pythia/alerts.jsonl',
    
    # Detection settings
    'threshold': 0.5,           # Probability threshold for alerts
    'poll_interval': 60,        # Seconds between checks (match your candle frequency)
    'lookback_minutes': 15,     # How far back to check for new candles (increased for 15m context)
    
    # Feature columns (must match training order!)
    # These MUST be in the exact same order as training (from feature_columns.txt)
    'feature_cols': [
        'returns',
        'MACD',
        'MACD_signal',
        'MACD_hist',
        'RSI_14',
        'NATR',
        'BB_width',
        'BB_squeeze',
        'VWAP_distance',
        'volume_zscore',
        'volume_roc',
        'OBV',
        'trade_count',
        'buy_sell_ratio',
        'roll_measure',
        'order_flow_imbalance',
        'vpin',
        'bid_ask_spread_pct',
        'order_book_depth_ratio',
        'large_order_imbalance',
        'returns_5m',
        'volume_zscore_5m',
        'returns_15m',
        'volume_zscore_15m',
    ],
    
    # Symbols to monitor (None = all symbols in database)
    'symbols': None,
    
    # Cooldown: Don't alert on same symbol within N minutes
    'alert_cooldown_minutes': 30,
    
    # Alert settings
    'sound_file': None,  # Path to alert sound, or None for system beep
    'desktop_notifications': True,
}


# =============================================================================
# SPIKE DETECTOR CLASS
# =============================================================================

class SpikeDetector:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.recent_alerts = {}  # symbol -> last_alert_time
        
        # Load model
        self._load_model()
        
        # Get feature columns
        self.feature_cols = config['feature_cols']
        
        logger.info(f"SpikeDetector initialized")
        logger.info(f"  Model: {config['model_path']}")
        logger.info(f"  Threshold: {config['threshold']}")
        logger.info(f"  Features: {len(self.feature_cols)}")
    
    def _load_model(self):
        """Load trained XGBoost model."""
        model_path = Path(self.config['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    
    def get_latest_features(self) -> pd.DataFrame:
        """Query database for latest features across all symbols."""
        conn = duckdb.connect(self.config['db_path'], read_only=True)
        
        try:
            # Get the most recent timestamp
            cutoff = datetime.utcnow() - timedelta(minutes=self.config['lookback_minutes'])
            cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')
            
            # Build symbol filter if specified
            symbol_filter = ""
            if self.config['symbols']:
                symbols_str = "', '".join(self.config['symbols'])
                symbol_filter = f"AND symbol IN ('{symbols_str}')"
            
            # Query for latest candle per symbol
            query = f"""
                WITH ranked AS (
                    SELECT 
                        symbol,
                        timestamp,
                        {', '.join(self.feature_cols)},
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                    FROM features
                    WHERE timeframe = '1m'
                    AND timestamp >= '{cutoff_str}'
                    {symbol_filter}
                )
                SELECT symbol, timestamp, {', '.join(self.feature_cols)}
                FROM ranked
                WHERE rn = 1
            """
            
            df = conn.execute(query).fetchdf()
            return df
            
        finally:
            conn.close()
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions on feature dataframe."""
        if len(df) == 0:
            return df
        
        # Extract features in correct order
        X = df[self.feature_cols].values.astype(np.float32)
        
        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Predict probabilities
        probs = self.model.predict_proba(X)[:, 1]
        
        df = df.copy()
        df['spike_probability'] = probs
        df['is_alert'] = probs >= self.config['threshold']
        
        return df
    
    def filter_cooldown(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out alerts that are in cooldown period."""
        if len(df) == 0:
            return df
        
        now = datetime.utcnow()
        cooldown = timedelta(minutes=self.config['alert_cooldown_minutes'])
        
        def check_cooldown(row):
            symbol = row['symbol']
            if symbol in self.recent_alerts:
                if now - self.recent_alerts[symbol] < cooldown:
                    return False
            return True
        
        mask = df.apply(check_cooldown, axis=1)
        return df[mask]
    
    def update_cooldowns(self, alerts: pd.DataFrame):
        """Update cooldown tracking after sending alerts."""
        now = datetime.utcnow()
        for symbol in alerts['symbol'].unique():
            self.recent_alerts[symbol] = now
    
    def send_alert(self, row: pd.Series):
        """Send alert for a detected spike."""
        symbol = row['symbol']
        prob = row['spike_probability']
        timestamp = row['timestamp']
        natr = row.get('NATR', 'N/A')
        
        msg = f"🚀 SPIKE ALERT: {symbol} @ {prob:.1%} probability (NATR={natr:.2f})"
        
        # Console log
        logger.warning(msg)
        
        # Desktop notification
        if HAS_NOTIFICATIONS and self.config['desktop_notifications']:
            try:
                notification.notify(
                    title=f"Spike Alert: {symbol}",
                    message=f"Probability: {prob:.1%}\nNATR: {natr:.2f}",
                    timeout=10
                )
            except Exception as e:
                logger.debug(f"Notification failed: {e}")
        
        # Sound alert
        if HAS_SOUND and self.config['sound_file']:
            try:
                playsound(self.config['sound_file'])
            except Exception as e:
                logger.debug(f"Sound failed: {e}")
        
        # Log to file
        self._log_alert(row)
    
    def _log_alert(self, row: pd.Series):
        """Append alert to JSONL log file."""
        alert_data = {
            'timestamp': str(row['timestamp']),
            'detected_at': datetime.utcnow().isoformat(),
            'symbol': row['symbol'],
            'probability': float(row['spike_probability']),
            'natr': float(row.get('NATR', 0)),
            'returns_5m': float(row.get('returns_5m', 0)),
            'bb_width': float(row.get('BB_width', 0)),
        }
        
        log_path = Path(self.config['alert_log'])
        with open(log_path, 'a') as f:
            f.write(json.dumps(alert_data) + '\n')
    
    def run_once(self) -> int:
        """Run single detection pass. Returns number of alerts sent."""
        # Get latest features
        df = self.get_latest_features()
        
        if len(df) == 0:
            logger.debug("No recent candles found")
            return 0
        
        logger.debug(f"Checking {len(df)} symbols...")
        
        # Run predictions
        df = self.predict(df)
        
        # Filter to alerts only
        alerts = df[df['is_alert']].copy()
        
        if len(alerts) == 0:
            logger.debug("No alerts")
            return 0
        
        # Apply cooldown filter
        alerts = self.filter_cooldown(alerts)
        
        if len(alerts) == 0:
            logger.debug("All alerts filtered by cooldown")
            return 0
        
        # Sort by probability descending
        alerts = alerts.sort_values('spike_probability', ascending=False)
        
        # Send alerts
        for _, row in alerts.iterrows():
            self.send_alert(row)
        
        # Update cooldowns
        self.update_cooldowns(alerts)
        
        return len(alerts)
    
    def run_loop(self):
        """Run continuous detection loop."""
        logger.info("=" * 60)
        logger.info("SPIKE DETECTOR RUNNING")
        logger.info("=" * 60)
        logger.info(f"Polling every {self.config['poll_interval']} seconds")
        logger.info(f"Threshold: {self.config['threshold']}")
        logger.info("Press Ctrl+C to stop")
        logger.info("")
        
        while True:
            try:
                n_alerts = self.run_once()
                
                if n_alerts > 0:
                    logger.info(f"Sent {n_alerts} alert(s)")
                
                time.sleep(self.config['poll_interval'])
                
            except KeyboardInterrupt:
                logger.info("\nStopping detector...")
                break
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(self.config['poll_interval'])


# =============================================================================
# TOP OPPORTUNITIES VIEW
# =============================================================================

def show_top_opportunities(detector: SpikeDetector, top_n: int = 20):
    """Show current top spike opportunities across all symbols."""
    logger.info("=" * 60)
    logger.info("TOP SPIKE OPPORTUNITIES (Current)")
    logger.info("=" * 60)
    logger.info("")
    
    # Get latest features
    df = detector.get_latest_features()
    
    if len(df) == 0:
        logger.warning("No recent candles found")
        return
    
    # Run predictions
    df = detector.predict(df)
    
    # Sort by probability
    df = df.sort_values('spike_probability', ascending=False)
    
    # Show top N
    logger.info(f"{'Symbol':<15} {'Prob':>8} {'NATR':>8} {'Ret5m':>8} {'BBWidth':>8}")
    logger.info("-" * 55)
    
    for _, row in df.head(top_n).iterrows():
        symbol = row['symbol']
        prob = row['spike_probability']
        natr = row.get('NATR', 0)
        ret5m = row.get('returns_5m', 0)
        bb = row.get('BB_width', 0)
        
        flag = "🚀" if prob >= detector.config['threshold'] else "  "
        logger.info(f"{flag} {symbol:<12} {prob:>7.1%} {natr:>8.3f} {ret5m:>8.3f} {bb:>8.3f}")
    
    logger.info("")
    logger.info(f"Showing top {top_n} of {len(df)} symbols")
    logger.info(f"Threshold for alert: {detector.config['threshold']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Real-time Slow & Large spike detector')
    
    parser.add_argument('--test', action='store_true',
                        help='Run once and exit (test mode)')
    parser.add_argument('--top', type=int, default=0,
                        help='Show top N opportunities and exit')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for alerts (default: 0.5)')
    parser.add_argument('--interval', type=int, default=60,
                        help='Poll interval in seconds (default: 60)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to XGBoost model file')
    parser.add_argument('--db', type=str, default=None,
                        help='Path to DuckDB database')
    
    args = parser.parse_args()
    
    # Update config from args
    config = CONFIG.copy()
    config['threshold'] = args.threshold
    config['poll_interval'] = args.interval
    
    if args.model:
        config['model_path'] = args.model
    if args.db:
        config['db_path'] = args.db
    
    # Initialize detector
    detector = SpikeDetector(config)
    
    # Run appropriate mode
    if args.top > 0:
        show_top_opportunities(detector, args.top)
    elif args.test:
        logger.info("Test mode: running single pass...")
        n = detector.run_once()
        logger.info(f"Found {n} alert(s)")
    else:
        detector.run_loop()


if __name__ == '__main__':
    main()
