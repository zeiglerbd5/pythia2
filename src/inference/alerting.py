"""
Alert Manager for Pre-Spike Detections

Handles notifications when high-probability signals are detected.
Supports console output, file logging, and optional external notifications.
"""

import json
from typing import Dict, Optional, Callable
from datetime import datetime
from pathlib import Path
from loguru import logger


class AlertManager:
    """
    Manages alerts for pre-spike detections.

    Features:
    - Console notifications (with color/formatting)
    - JSON log file for signal tracking
    - Optional custom callback for external integrations
    - Alert throttling to avoid spam
    """

    def __init__(
        self,
        min_probability: float = 0.5,
        min_confidence: Optional[float] = None,
        alert_log_file: Optional[str] = None,
        alert_callback: Optional[Callable] = None,
        throttle_seconds: int = 60
    ):
        """
        Initialize alert manager.

        Args:
            min_probability: Minimum probability to trigger alert (0-1)
            min_confidence: Minimum confidence to trigger alert (optional)
            alert_log_file: Path to JSON log file for alerts
            alert_callback: Custom callback function(alert_dict)
            throttle_seconds: Minimum seconds between alerts for same symbol
        """
        self.min_probability = min_probability
        self.min_confidence = min_confidence
        self.alert_log_file = alert_log_file
        self.alert_callback = alert_callback
        self.throttle_seconds = throttle_seconds

        # Track last alert time per symbol (for throttling)
        self.last_alert_time: Dict[str, float] = {}

        # Statistics
        self.alerts_sent = 0
        self.alerts_throttled = 0

        # Initialize log file if specified
        if self.alert_log_file:
            log_path = Path(self.alert_log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Create/initialize log file
            if not log_path.exists():
                with open(log_path, 'w') as f:
                    json.dump([], f)

        logger.info(
            f"AlertManager initialized: "
            f"min_probability={min_probability}, "
            f"throttle={throttle_seconds}s"
        )

    def should_alert(self, symbol: str, probability: float, confidence: float) -> bool:
        """
        Check if alert should be triggered.

        Args:
            symbol: Trading pair symbol
            probability: Model probability (0-1)
            confidence: Confidence score

        Returns:
            True if alert should be sent
        """
        # Check probability threshold
        if probability < self.min_probability:
            return False

        # Check confidence threshold if specified
        if self.min_confidence is not None and confidence < self.min_confidence:
            return False

        # Check throttling
        now = datetime.now().timestamp()

        if symbol in self.last_alert_time:
            time_since_last = now - self.last_alert_time[symbol]

            if time_since_last < self.throttle_seconds:
                self.alerts_throttled += 1
                return False

        return True

    def send_alert(
        self,
        symbol: str,
        probability: float,
        confidence: float,
        timestamp: datetime,
        additional_data: Optional[Dict] = None
    ):
        """
        Send alert for pre-spike detection.

        Args:
            symbol: Trading pair symbol
            probability: Model probability
            confidence: Confidence score
            timestamp: Candle timestamp
            additional_data: Optional extra data to include
        """
        # Check if should alert
        if not self.should_alert(symbol, probability, confidence):
            return

        # Build alert data
        alert = {
            'symbol': symbol,
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
            'probability': round(probability, 4),
            'confidence': round(confidence, 4),
            'alert_time': datetime.now().isoformat(),
        }

        if additional_data:
            alert.update(additional_data)

        # Update throttle tracker
        self.last_alert_time[symbol] = datetime.now().timestamp()

        # Console output (formatted)
        self._log_to_console(alert)

        # Write to log file
        if self.alert_log_file:
            self._log_to_file(alert)

        # Custom callback
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        self.alerts_sent += 1

    def _log_to_console(self, alert: Dict):
        """Log alert to console with formatting."""
        symbol = alert['symbol']
        prob = alert['probability']
        conf = alert['confidence']
        timestamp = alert['timestamp']

        # Color-coded by probability
        if prob >= 0.8:
            level = "CRITICAL"
            emoji = "🔥"
        elif prob >= 0.7:
            level = "HIGH"
            emoji = "⚠️"
        elif prob >= 0.6:
            level = "MEDIUM"
            emoji = "📊"
        else:
            level = "LOW"
            emoji = "📈"

        logger.warning(
            f"\n"
            f"{'='*80}\n"
            f"{emoji} PRE-SPIKE ALERT [{level}] {emoji}\n"
            f"{'='*80}\n"
            f"Symbol:      {symbol}\n"
            f"Probability: {prob:.1%}\n"
            f"Confidence:  {conf:.3f}\n"
            f"Timestamp:   {timestamp}\n"
            f"{'='*80}\n"
        )

    def _log_to_file(self, alert: Dict):
        """Append alert to JSON log file."""
        try:
            # Read existing alerts
            with open(self.alert_log_file, 'r') as f:
                alerts = json.load(f)

            # Append new alert
            alerts.append(alert)

            # Write back
            with open(self.alert_log_file, 'w') as f:
                json.dump(alerts, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to write alert to log file: {e}")

    def get_statistics(self) -> Dict:
        """Get alert statistics."""
        return {
            'alerts_sent': self.alerts_sent,
            'alerts_throttled': self.alerts_throttled,
            'min_probability': self.min_probability,
            'min_confidence': self.min_confidence,
            'throttle_seconds': self.throttle_seconds,
            'symbols_alerted': len(self.last_alert_time),
        }

    def reset_throttle(self, symbol: Optional[str] = None):
        """
        Reset throttle for a symbol or all symbols.

        Args:
            symbol: Symbol to reset, or None for all
        """
        if symbol:
            if symbol in self.last_alert_time:
                del self.last_alert_time[symbol]
                logger.info(f"Throttle reset for {symbol}")
        else:
            self.last_alert_time.clear()
            logger.info("Throttle reset for all symbols")
