"""
Spike Predictor - Early Warning System v1.0

Detects potential price spikes BEFORE they happen by monitoring:
- Volume buildup patterns
- Whale trade activity
- Trade frequency clustering
- Price compression (low volatility before breakout)

Based on analysis of 100+ historical spikes with 50%+ daily moves.

Key Finding: Volume starts building 3-6 hours BEFORE major price moves.
"""

import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class SpikeAlert:
    """Represents an early warning alert for potential spike"""
    symbol: str
    alert_time: datetime
    alert_level: str  # 'watch', 'warning', 'critical'

    # Feature values that triggered alert
    volume_ratio_6h: float
    volume_ratio_1h: float
    trade_frequency_ratio: float
    large_trade_count: int
    price_compression: float
    buy_sell_imbalance: float

    # Price at alert time
    price_at_alert: float

    # Tracking
    status: str = 'active'  # active, confirmed, expired
    peak_price: Optional[float] = None
    peak_time: Optional[datetime] = None
    max_gain_pct: float = 0.0


class SpikePredictor:
    """
    Early warning system for detecting potential price spikes.

    Monitors real-time trade and OHLCV data to identify accumulation
    patterns that precede major price moves.
    """

    # Alert thresholds (tuned from historical analysis)
    VOLUME_RATIO_6H_WATCH = 2.0       # 2x volume buildup = watch
    VOLUME_RATIO_6H_WARNING = 5.0     # 5x = warning
    VOLUME_RATIO_6H_CRITICAL = 10.0   # 10x = critical

    VOLUME_RATIO_1H_TRIGGER = 3.0     # 3x in last hour
    TRADE_FREQ_RATIO_MIN = 1.5        # 1.5x trade frequency
    LARGE_TRADE_COUNT_MIN = 3         # At least 3 whale trades
    PRICE_COMPRESSION_MAX = 0.7       # Volatility < 70% of normal

    # Lookback windows
    LOOKBACK_SHORT = timedelta(hours=1)
    LOOKBACK_MEDIUM = timedelta(hours=6)
    LOOKBACK_LONG = timedelta(hours=24)

    # Alert expiry
    ALERT_EXPIRY_HOURS = 12

    def __init__(self, db_path: str = 'data/feature_buffer.db'):
        """Initialize the spike predictor"""
        self.db_path = db_path

        # State tracking
        self.alerts: Dict[str, SpikeAlert] = {}  # symbol -> active alert
        self.alert_history: List[SpikeAlert] = []

        # Feature caches (updated periodically)
        self.volume_cache: Dict[str, pd.DataFrame] = {}
        self.trade_cache: Dict[str, pd.DataFrame] = {}
        self.feature_cache: Dict[str, dict] = {}

        # Statistics
        self.stats = {
            'scans_completed': 0,
            'alerts_generated': 0,
            'alerts_confirmed': 0,
            'alerts_expired': 0
        }

        logger.info("SpikePredictor initialized")
        logger.info(f"Thresholds: Vol6h>{self.VOLUME_RATIO_6H_WATCH}x, "
                   f"Vol1h>{self.VOLUME_RATIO_1H_TRIGGER}x, "
                   f"TradeFreq>{self.TRADE_FREQ_RATIO_MIN}x")

    def get_ohlcv_data(self, symbol: str, hours: int = 48) -> pd.DataFrame:
        """Get recent OHLCV data for a symbol"""
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ?
            AND timestamp > datetime('now', '-{hours} hours')
            ORDER BY timestamp
        """
        df = pd.read_sql(query, conn, params=[symbol])
        conn.close()

        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def get_trade_data(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Get recent trade data for a symbol"""
        conn = sqlite3.connect(self.db_path)

        # Try trades table first, fall back to ticker if needed
        try:
            query = f"""
                SELECT timestamp, price, size, side
                FROM trades
                WHERE symbol = ?
                AND timestamp > datetime('now', '-{hours} hours')
                ORDER BY timestamp
            """
            df = pd.read_sql(query, conn, params=[symbol])
        except:
            # Fallback: use ticker data
            query = f"""
                SELECT timestamp, price, volume as size
                FROM ticker
                WHERE symbol = ?
                AND timestamp > datetime('now', '-{hours} hours')
                ORDER BY timestamp
            """
            df = pd.read_sql(query, conn, params=[symbol])
            if len(df) > 0:
                df['side'] = 'unknown'

        conn.close()

        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols with recent data"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT DISTINCT symbol
            FROM ohlcv
            WHERE timestamp > datetime('now', '-2 hours')
        """
        symbols = pd.read_sql(query, conn)['symbol'].tolist()
        conn.close()
        return symbols

    def compute_features(self, symbol: str) -> Optional[dict]:
        """
        Compute all spike prediction features for a symbol.

        Returns dict of features or None if insufficient data.
        """
        # Get OHLCV data
        ohlcv = self.get_ohlcv_data(symbol, hours=48)
        if len(ohlcv) < 60:  # Need at least 1 hour of minute data
            return None

        now = ohlcv['timestamp'].max()

        # Aggregate to hourly for some calculations
        ohlcv['hour'] = ohlcv['timestamp'].dt.floor('h')
        hourly = ohlcv.groupby('hour').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()

        if len(hourly) < 12:
            return None

        # Current price
        current_price = ohlcv['close'].iloc[-1]

        # --- Volume Features ---

        # Volume in different windows
        vol_1h = ohlcv[ohlcv['timestamp'] > now - timedelta(hours=1)]['volume'].sum()
        vol_6h = ohlcv[ohlcv['timestamp'] > now - timedelta(hours=6)]['volume'].sum()
        vol_24h = ohlcv[ohlcv['timestamp'] > now - timedelta(hours=24)]['volume'].sum()

        # Previous period volumes for ratios
        vol_prev_1h = ohlcv[
            (ohlcv['timestamp'] > now - timedelta(hours=2)) &
            (ohlcv['timestamp'] <= now - timedelta(hours=1))
        ]['volume'].sum()

        vol_prev_6h = ohlcv[
            (ohlcv['timestamp'] > now - timedelta(hours=12)) &
            (ohlcv['timestamp'] <= now - timedelta(hours=6))
        ]['volume'].sum()

        # Volume ratios
        volume_ratio_1h = vol_1h / vol_prev_1h if vol_prev_1h > 0 else 0
        volume_ratio_6h = vol_6h / vol_prev_6h if vol_prev_6h > 0 else 0

        # Volume acceleration (is it increasing?)
        if len(hourly) >= 6:
            recent_vols = hourly['volume'].tail(6).values
            if len(recent_vols) >= 3:
                vol_trend = np.polyfit(range(len(recent_vols)), recent_vols, 1)[0]
                volume_acceleration = vol_trend / (np.mean(recent_vols) + 1e-10)
            else:
                volume_acceleration = 0
        else:
            volume_acceleration = 0

        # --- Trade Frequency Features ---

        # Count of candles (proxy for trade activity)
        trades_1h = len(ohlcv[ohlcv['timestamp'] > now - timedelta(hours=1)])
        trades_24h_avg = len(ohlcv) / 24 if len(ohlcv) > 0 else 1
        trade_frequency_ratio = trades_1h / trades_24h_avg if trades_24h_avg > 0 else 0

        # --- Large Trade Detection ---

        # Use volume spikes as proxy for large trades
        ohlcv['vol_zscore'] = (ohlcv['volume'] - ohlcv['volume'].mean()) / (ohlcv['volume'].std() + 1e-10)
        large_trade_count = len(ohlcv[
            (ohlcv['timestamp'] > now - timedelta(hours=2)) &
            (ohlcv['vol_zscore'] > 2)
        ])

        # --- Price Compression ---

        # Volatility in recent period vs historical
        price_std_1h = ohlcv[ohlcv['timestamp'] > now - timedelta(hours=1)]['close'].std()
        price_std_24h = ohlcv['close'].std()
        price_compression = price_std_1h / price_std_24h if price_std_24h > 0 else 1

        # --- Price Position ---

        high_24h = ohlcv['high'].max()
        low_24h = ohlcv['low'].min()
        range_24h = high_24h - low_24h
        distance_from_low = (current_price - low_24h) / range_24h if range_24h > 0 else 0.5

        # --- Momentum ---

        price_6h_ago = ohlcv[ohlcv['timestamp'] <= now - timedelta(hours=6)]['close']
        if len(price_6h_ago) > 0:
            momentum_6h = (current_price - price_6h_ago.iloc[-1]) / price_6h_ago.iloc[-1] * 100
        else:
            momentum_6h = 0

        # --- Buy/Sell Imbalance (from price direction) ---

        recent = ohlcv[ohlcv['timestamp'] > now - timedelta(hours=1)]
        if len(recent) > 0:
            up_candles = len(recent[recent['close'] > recent['open']])
            down_candles = len(recent[recent['close'] < recent['open']])
            total = up_candles + down_candles
            buy_sell_imbalance = (up_candles - down_candles) / total if total > 0 else 0
        else:
            buy_sell_imbalance = 0

        # --- Time Features ---

        hour_of_day = now.hour
        day_of_week = now.weekday()
        is_weekend = day_of_week >= 5
        is_peak_hour = hour_of_day in [0, 1, 2, 3, 4, 5, 6, 22, 23]  # Peak spike hours

        features = {
            'symbol': symbol,
            'timestamp': now,
            'price': current_price,

            # Volume features (Tier 1)
            'volume_ratio_1h': volume_ratio_1h,
            'volume_ratio_6h': volume_ratio_6h,
            'volume_acceleration': volume_acceleration,
            'vol_1h': vol_1h,
            'vol_6h': vol_6h,
            'vol_24h': vol_24h,

            # Activity features (Tier 1)
            'trade_frequency_ratio': trade_frequency_ratio,
            'large_trade_count': large_trade_count,

            # Price features (Tier 2)
            'price_compression': price_compression,
            'buy_sell_imbalance': buy_sell_imbalance,
            'distance_from_low': distance_from_low,
            'momentum_6h': momentum_6h,

            # Time features (Tier 2)
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_peak_hour': is_peak_hour,
        }

        return features

    def evaluate_alert_level(self, features: dict) -> Optional[str]:
        """
        Evaluate features and determine alert level.

        Returns: 'watch', 'warning', 'critical', or None
        """
        vol_6h = features['volume_ratio_6h']
        vol_1h = features['volume_ratio_1h']
        trade_freq = features['trade_frequency_ratio']
        large_trades = features['large_trade_count']
        compression = features['price_compression']

        # Critical: Multiple strong signals
        if (vol_6h >= self.VOLUME_RATIO_6H_CRITICAL and
            vol_1h >= self.VOLUME_RATIO_1H_TRIGGER):
            return 'critical'

        # Warning: Strong volume buildup
        if (vol_6h >= self.VOLUME_RATIO_6H_WARNING and
            trade_freq >= self.TRADE_FREQ_RATIO_MIN):
            return 'warning'

        # Watch: Early signs
        if (vol_6h >= self.VOLUME_RATIO_6H_WATCH and
            (trade_freq >= self.TRADE_FREQ_RATIO_MIN or
             large_trades >= self.LARGE_TRADE_COUNT_MIN or
             compression <= self.PRICE_COMPRESSION_MAX)):
            return 'watch'

        return None

    def scan_all_symbols(self) -> List[SpikeAlert]:
        """
        Scan all symbols and generate alerts.

        Returns list of new alerts generated.
        """
        symbols = self.get_all_symbols()
        new_alerts = []

        for symbol in symbols:
            try:
                features = self.compute_features(symbol)
                if features is None:
                    continue

                self.feature_cache[symbol] = features

                alert_level = self.evaluate_alert_level(features)

                if alert_level:
                    # Check if we already have an active alert
                    if symbol in self.alerts:
                        existing = self.alerts[symbol]
                        # Upgrade alert level if needed
                        levels = {'watch': 1, 'warning': 2, 'critical': 3}
                        if levels.get(alert_level, 0) > levels.get(existing.alert_level, 0):
                            existing.alert_level = alert_level
                            logger.info(f"ALERT UPGRADED: {symbol} -> {alert_level}")
                    else:
                        # Create new alert
                        alert = SpikeAlert(
                            symbol=symbol,
                            alert_time=features['timestamp'],
                            alert_level=alert_level,
                            volume_ratio_6h=features['volume_ratio_6h'],
                            volume_ratio_1h=features['volume_ratio_1h'],
                            trade_frequency_ratio=features['trade_frequency_ratio'],
                            large_trade_count=features['large_trade_count'],
                            price_compression=features['price_compression'],
                            buy_sell_imbalance=features['buy_sell_imbalance'],
                            price_at_alert=features['price']
                        )

                        self.alerts[symbol] = alert
                        new_alerts.append(alert)
                        self.stats['alerts_generated'] += 1

                        logger.info(f"NEW ALERT [{alert_level.upper()}]: {symbol} | "
                                   f"Vol6h: {features['volume_ratio_6h']:.1f}x | "
                                   f"Vol1h: {features['volume_ratio_1h']:.1f}x | "
                                   f"TradeFreq: {features['trade_frequency_ratio']:.1f}x | "
                                   f"Price: ${features['price']:.6f}")

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        self.stats['scans_completed'] += 1
        return new_alerts

    def update_alerts(self) -> List[SpikeAlert]:
        """
        Update active alerts with current prices.
        Track peaks and check for confirmation/expiry.

        Returns list of alerts that were confirmed or expired.
        """
        resolved = []
        now = datetime.now()

        for symbol, alert in list(self.alerts.items()):
            try:
                # Get current price
                ohlcv = self.get_ohlcv_data(symbol, hours=1)
                if len(ohlcv) == 0:
                    continue

                current_price = ohlcv['close'].iloc[-1]
                current_time = ohlcv['timestamp'].iloc[-1]

                # Update peak tracking
                if current_price > (alert.peak_price or 0):
                    alert.peak_price = current_price
                    alert.peak_time = current_time

                # Calculate gain from alert price
                gain_pct = (current_price - alert.price_at_alert) / alert.price_at_alert * 100
                alert.max_gain_pct = max(alert.max_gain_pct, gain_pct)

                # Check for confirmation (15%+ gain)
                if alert.max_gain_pct >= 15:
                    alert.status = 'confirmed'
                    resolved.append(alert)
                    self.alert_history.append(alert)
                    del self.alerts[symbol]
                    self.stats['alerts_confirmed'] += 1

                    logger.info(f"ALERT CONFIRMED: {symbol} | "
                               f"Gain: +{alert.max_gain_pct:.1f}% | "
                               f"Peak: ${alert.peak_price:.6f}")

                # Check for expiry
                elif (current_time - alert.alert_time).total_seconds() > self.ALERT_EXPIRY_HOURS * 3600:
                    alert.status = 'expired'
                    resolved.append(alert)
                    self.alert_history.append(alert)
                    del self.alerts[symbol]
                    self.stats['alerts_expired'] += 1

                    logger.info(f"ALERT EXPIRED: {symbol} | "
                               f"Max Gain: {alert.max_gain_pct:+.1f}%")

            except Exception as e:
                logger.debug(f"Error updating alert for {symbol}: {e}")
                continue

        return resolved

    def get_top_alerts(self, n: int = 10) -> List[Tuple[str, dict]]:
        """Get top N symbols by alert potential"""
        scored = []

        for symbol, features in self.feature_cache.items():
            # Composite score
            score = (
                features['volume_ratio_6h'] * 2 +
                features['volume_ratio_1h'] * 3 +
                features['trade_frequency_ratio'] * 1.5 +
                features['large_trade_count'] * 0.5 +
                (1 - features['price_compression']) * 2 +
                (1 if features['is_peak_hour'] else 0) * 1
            )
            scored.append((symbol, features, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return [(s, f) for s, f, _ in scored[:n]]

    def get_stats(self) -> dict:
        """Get predictor statistics"""
        return {
            **self.stats,
            'active_alerts': len(self.alerts),
            'symbols_tracked': len(self.feature_cache),
            'confirmation_rate': (
                self.stats['alerts_confirmed'] / self.stats['alerts_generated'] * 100
                if self.stats['alerts_generated'] > 0 else 0
            )
        }

    def print_status(self):
        """Print current status"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("SPIKE PREDICTOR STATUS")
        print("=" * 60)
        print(f"Scans: {stats['scans_completed']} | "
              f"Alerts: {stats['alerts_generated']} | "
              f"Confirmed: {stats['alerts_confirmed']} | "
              f"Expired: {stats['alerts_expired']}")

        if self.alerts:
            print(f"\nActive Alerts ({len(self.alerts)}):")
            for symbol, alert in sorted(self.alerts.items(),
                                        key=lambda x: x[1].volume_ratio_6h,
                                        reverse=True):
                print(f"  [{alert.alert_level.upper():8}] {symbol:12} | "
                      f"Vol6h: {alert.volume_ratio_6h:.1f}x | "
                      f"Gain: {alert.max_gain_pct:+.1f}%")

    def save_state(self, filepath: str = 'spike_predictor_state.json'):
        """Save current state to JSON"""
        state = {
            'last_update': datetime.now().isoformat(),
            'stats': self.get_stats(),
            'active_alerts': [
                {
                    'symbol': a.symbol,
                    'alert_time': a.alert_time.isoformat(),
                    'alert_level': a.alert_level,
                    'volume_ratio_6h': a.volume_ratio_6h,
                    'volume_ratio_1h': a.volume_ratio_1h,
                    'trade_frequency_ratio': a.trade_frequency_ratio,
                    'large_trade_count': a.large_trade_count,
                    'price_at_alert': a.price_at_alert,
                    'max_gain_pct': a.max_gain_pct,
                }
                for a in self.alerts.values()
            ],
            'top_watchlist': [
                {
                    'symbol': symbol,
                    'volume_ratio_6h': f['volume_ratio_6h'],
                    'volume_ratio_1h': f['volume_ratio_1h'],
                    'trade_frequency_ratio': f['trade_frequency_ratio'],
                    'price': f['price'],
                }
                for symbol, f in self.get_top_alerts(20)
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        return state


def main():
    """Test the spike predictor"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    predictor = SpikePredictor()

    print("Scanning for potential spikes...")
    alerts = predictor.scan_all_symbols()

    print(f"\nGenerated {len(alerts)} new alerts")

    predictor.print_status()

    print("\nTop 10 by spike potential:")
    for symbol, features in predictor.get_top_alerts(10):
        print(f"  {symbol:12} | Vol6h: {features['volume_ratio_6h']:.1f}x | "
              f"Vol1h: {features['volume_ratio_1h']:.1f}x | "
              f"TradeFreq: {features['trade_frequency_ratio']:.1f}x | "
              f"${features['price']:.6f}")


if __name__ == '__main__':
    main()
