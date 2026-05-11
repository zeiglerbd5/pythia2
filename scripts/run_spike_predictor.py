#!/usr/bin/env python3
"""
Spike Predictor Monitor

Runs continuously to detect potential price spikes BEFORE they happen.
Generates alerts based on volume buildup, whale activity, and other precursors.

Usage:
    python scripts/run_spike_predictor.py
    python scripts/run_spike_predictor.py --interval 30  # Check every 30 seconds
"""

import logging
import time
import signal
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies.spike_predictor import SpikePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SpikeMonitor:
    """Continuous monitor for spike detection"""

    def __init__(self, db_path: str = 'data/feature_buffer.db',
                 check_interval: int = 60):
        self.db_path = db_path
        self.check_interval = check_interval
        self.predictor = SpikePredictor(db_path=db_path)
        self.running = True

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received...")
        self.running = False

    def run(self):
        """Main run loop"""
        logger.info("=" * 60)
        logger.info("SPIKE PREDICTOR v1.0 - Early Warning System")
        logger.info("=" * 60)
        logger.info(f"Thresholds:")
        logger.info(f"  Watch:    Vol6h > {self.predictor.VOLUME_RATIO_6H_WATCH}x")
        logger.info(f"  Warning:  Vol6h > {self.predictor.VOLUME_RATIO_6H_WARNING}x")
        logger.info(f"  Critical: Vol6h > {self.predictor.VOLUME_RATIO_6H_CRITICAL}x")
        logger.info(f"Checking every {self.check_interval} seconds...")
        logger.info("")

        while self.running:
            try:
                scan_start = time.time()

                # Scan for new alerts
                new_alerts = self.predictor.scan_all_symbols()

                # Update existing alerts
                resolved = self.predictor.update_alerts()

                # Log summary
                stats = self.predictor.get_stats()
                active = len(self.predictor.alerts)

                if new_alerts or resolved or active > 0:
                    alert_summary = []
                    for level in ['critical', 'warning', 'watch']:
                        count = sum(1 for a in self.predictor.alerts.values()
                                   if a.alert_level == level)
                        if count > 0:
                            alert_summary.append(f"{count} {level}")

                    logger.info(f"Active: {' | '.join(alert_summary) if alert_summary else 'none'} | "
                               f"Confirmed: {stats['alerts_confirmed']} | "
                               f"Symbols: {stats['symbols_tracked']}")

                    # Log critical alerts
                    for symbol, alert in self.predictor.alerts.items():
                        if alert.alert_level == 'critical':
                            logger.info(f"  CRITICAL: {symbol} | "
                                       f"Vol6h: {alert.volume_ratio_6h:.1f}x | "
                                       f"Gain: {alert.max_gain_pct:+.1f}%")

                # Save state for dashboard
                self.predictor.save_state()

                # Wait for next check
                elapsed = time.time() - scan_start
                sleep_time = max(1, self.check_interval - elapsed)

                for _ in range(int(sleep_time)):
                    if not self.running:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in scan loop: {e}")
                time.sleep(5)

        # Final status
        logger.info("\nFinal Status:")
        self.predictor.print_status()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Spike Predictor Monitor')
    parser.add_argument('--db', default='data/feature_buffer.db',
                       help='Path to database')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds')
    args = parser.parse_args()

    monitor = SpikeMonitor(db_path=args.db, check_interval=args.interval)
    monitor.run()


if __name__ == '__main__':
    main()
