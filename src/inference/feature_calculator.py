"""
Real-time Feature Calculator

Calculates all 24 features incrementally as new candles arrive.
Maintains rolling windows for indicators (RSI, MACD, Bollinger Bands, etc.).

Features calculated (same as training):
- 9 price features: returns, MACD(3), RSI, NATR, BB_width, BB_squeeze, VWAP_distance
- 5 volume features: volume_zscore, volume_roc, OBV, trade_count, buy_sell_ratio
- 6 microstructure: roll_measure, order_flow_imbalance, vpin, bid_ask_spread_pct,
                    order_book_depth_ratio, large_order_imbalance
- 4 multi-timeframe: returns_5m, volume_zscore_5m, returns_15m, volume_zscore_15m
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Deque
from collections import deque, defaultdict
from loguru import logger


class RollingWindow:
    """Efficient rolling window for time series calculations."""

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)

    def append(self, value):
        """Add new value to window."""
        self.data.append(value)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.data)

    def is_full(self) -> bool:
        """Check if window has enough data."""
        return len(self.data) == self.maxlen

    def __len__(self):
        return len(self.data)


class EWMACalculator:
    """Efficient exponential weighted moving average calculator."""

    def __init__(self, span: int):
        self.span = span
        self.alpha = 2.0 / (span + 1)
        self.ewma = None

    def update(self, value: float) -> float:
        """Update EWMA with new value."""
        if self.ewma is None:
            self.ewma = value
        else:
            self.ewma = self.alpha * value + (1 - self.alpha) * self.ewma
        return self.ewma

    def get(self) -> Optional[float]:
        """Get current EWMA value."""
        return self.ewma


class SymbolFeatureState:
    """Maintains feature calculation state for a single symbol."""

    def __init__(self, symbol: str):
        self.symbol = symbol

        # Raw candle history (need at least 100 for all indicators)
        self.candle_window = RollingWindow(maxlen=200)

        # Price indicators
        self.ema12 = EWMACalculator(span=12)
        self.ema26 = EWMACalculator(span=26)
        self.macd_signal = EWMACalculator(span=9)

        # RSI components
        self.gain_window = RollingWindow(maxlen=14)
        self.loss_window = RollingWindow(maxlen=14)

        # ATR components
        self.tr_window = RollingWindow(maxlen=14)

        # Bollinger Band components
        self.close_window = RollingWindow(maxlen=20)
        self.bb_width_window = RollingWindow(maxlen=20)

        # Volume indicators
        self.volume_window = RollingWindow(maxlen=20)
        self.volume_5period = RollingWindow(maxlen=6)  # For volume_roc(5)

        # OBV accumulator
        self.obv = 0.0
        self.last_close = None

        # VWAP accumulators
        self.vwap_pv_sum = 0.0  # price * volume cumulative
        self.vwap_v_sum = 0.0   # volume cumulative

        # Microstructure indicators
        self.price_change_window = RollingWindow(maxlen=51)  # For roll measure
        self.order_flow_window = RollingWindow(maxlen=50)    # For VPIN
        self.buy_volume_window = RollingWindow(maxlen=50)
        self.sell_volume_window = RollingWindow(maxlen=50)

        # Multi-timeframe (aggregate 1m → 5m, 15m)
        self.candles_5m = RollingWindow(maxlen=5)   # For 5m aggregation
        self.candles_15m = RollingWindow(maxlen=15)  # For 15m aggregation
        self.last_5m_returns = None
        self.last_5m_volume_zscore = None
        self.last_15m_returns = None
        self.last_15m_volume_zscore = None

        logger.debug(f"SymbolFeatureState initialized for {symbol}")

    def update_candle(self, candle: Dict) -> Optional[Dict]:
        """
        Update state with new candle and calculate features.

        Args:
            candle: Dict with keys: timestamp, open, high, low, close, volume,
                    buy_volume, sell_volume, num_trades

        Returns:
            Dict of 24 features, or None if not enough data yet
        """
        # Add to candle history
        self.candle_window.append(candle)

        # Need at least 50 candles to calculate all features
        if len(self.candle_window) < 50:
            return None

        # Calculate all feature groups
        features = {}

        # 1. Price features (9)
        price_features = self._calculate_price_features(candle)
        features.update(price_features)

        # 2. Volume features (5)
        volume_features = self._calculate_volume_features(candle)
        features.update(volume_features)

        # 3. Microstructure features (6)
        micro_features = self._calculate_microstructure_features(candle)
        features.update(micro_features)

        # 4. Multi-timeframe features (4)
        mtf_features = self._calculate_multitimeframe_features(candle)
        features.update(mtf_features)

        return features

    def _calculate_price_features(self, candle: Dict) -> Dict:
        """Calculate 9 price features."""
        features = {}
        close = candle['close']
        high = candle['high']
        low = candle['low']

        # Get recent candles
        recent_candles = list(self.candle_window.data)

        # 1. Returns (percent change)
        if len(recent_candles) >= 2:
            prev_close = recent_candles[-2]['close']
            features['returns'] = (close - prev_close) / prev_close
        else:
            features['returns'] = 0.0

        # 2-4. MACD components
        ema12_val = self.ema12.update(close)
        ema26_val = self.ema26.update(close)

        if ema12_val is not None and ema26_val is not None:
            macd = ema12_val - ema26_val
            macd_signal_val = self.macd_signal.update(macd)

            features['MACD'] = macd
            features['MACD_signal'] = macd_signal_val if macd_signal_val else 0.0
            features['MACD_hist'] = macd - (macd_signal_val if macd_signal_val else 0.0)
        else:
            features['MACD'] = 0.0
            features['MACD_signal'] = 0.0
            features['MACD_hist'] = 0.0

        # 5. RSI (14-period)
        if len(recent_candles) >= 2:
            prev_close = recent_candles[-2]['close']
            change = close - prev_close
            gain = max(0, change)
            loss = max(0, -change)

            self.gain_window.append(gain)
            self.loss_window.append(loss)

            if self.gain_window.is_full():
                avg_gain = np.mean(self.gain_window.to_array())
                avg_loss = np.mean(self.loss_window.to_array())

                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100.0 if avg_gain > 0 else 50.0

                features['RSI_14'] = rsi
            else:
                features['RSI_14'] = 50.0
        else:
            features['RSI_14'] = 50.0

        # 6. NATR (Normalized ATR)
        if len(recent_candles) >= 2:
            prev_close = recent_candles[-2]['close']
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            self.tr_window.append(tr)

            if self.tr_window.is_full():
                atr = np.mean(self.tr_window.to_array())
                natr = (atr / close) * 100 if close > 0 else 0.0
                features['NATR'] = natr
            else:
                features['NATR'] = 0.0
        else:
            features['NATR'] = 0.0

        # 7-8. Bollinger Bands
        self.close_window.append(close)

        if self.close_window.is_full():
            closes = self.close_window.to_array()
            sma20 = np.mean(closes)
            std20 = np.std(closes)

            bb_width = (std20 * 4) / sma20 if sma20 > 0 else 0.0
            features['BB_width'] = bb_width

            # BB squeeze: width relative to recent average width
            self.bb_width_window.append(bb_width)
            if self.bb_width_window.is_full():
                bb_width_ma = np.mean(self.bb_width_window.to_array())
                bb_squeeze = bb_width / bb_width_ma if bb_width_ma > 0 else 1.0
                features['BB_squeeze'] = bb_squeeze
            else:
                features['BB_squeeze'] = 1.0
        else:
            features['BB_width'] = 0.0
            features['BB_squeeze'] = 1.0

        # 9. VWAP distance
        volume = candle['volume']
        self.vwap_pv_sum += close * volume
        self.vwap_v_sum += volume

        if self.vwap_v_sum > 0:
            vwap = self.vwap_pv_sum / self.vwap_v_sum
            vwap_distance = (close - vwap) / vwap if vwap > 0 else 0.0
            features['VWAP_distance'] = vwap_distance
        else:
            features['VWAP_distance'] = 0.0

        return features

    def _calculate_volume_features(self, candle: Dict) -> Dict:
        """Calculate 5 volume features."""
        features = {}
        volume = candle['volume']

        # 1. Volume z-score
        self.volume_window.append(volume)

        if self.volume_window.is_full():
            volumes = self.volume_window.to_array()
            vol_mean = np.mean(volumes)
            vol_std = np.std(volumes)

            if vol_std > 0:
                volume_zscore = (volume - vol_mean) / vol_std
            else:
                volume_zscore = 0.0

            features['volume_zscore'] = volume_zscore
        else:
            features['volume_zscore'] = 0.0

        # 2. Volume rate of change (5-period)
        self.volume_5period.append(volume)

        if len(self.volume_5period) == 6:  # Need 6 to calculate 5-period change
            vol_5ago = self.volume_5period.data[0]
            if vol_5ago > 0:
                volume_roc = (volume - vol_5ago) / vol_5ago
            else:
                volume_roc = 0.0
            features['volume_roc'] = volume_roc
        else:
            features['volume_roc'] = 0.0

        # 3. OBV (On-Balance Volume)
        if self.last_close is not None:
            close = candle['close']
            if close > self.last_close:
                self.obv += volume
            elif close < self.last_close:
                self.obv -= volume
            # No change if close == last_close

        self.last_close = candle['close']
        features['OBV'] = self.obv

        # 4. Trade count
        features['trade_count'] = candle.get('num_trades', 0)

        # 5. Buy/Sell ratio
        buy_vol = candle.get('buy_volume', 0)
        sell_vol = candle.get('sell_volume', 0)

        if sell_vol > 0:
            buy_sell_ratio = buy_vol / sell_vol
        else:
            buy_sell_ratio = 1.0

        features['buy_sell_ratio'] = buy_sell_ratio

        return features

    def _calculate_microstructure_features(self, candle: Dict) -> Dict:
        """Calculate 6 microstructure features."""
        features = {}

        recent_candles = list(self.candle_window.data)

        # 1. Roll measure (bid-ask spread estimator)
        if len(recent_candles) >= 2:
            prev_close = recent_candles[-2]['close']
            price_change = candle['close'] - prev_close
            self.price_change_window.append(price_change)

            if len(self.price_change_window) == 51:  # Need 51 for 50-period covariance
                changes = self.price_change_window.to_array()
                changes_lag = changes[:-1]
                changes_current = changes[1:]

                # Covariance of consecutive price changes
                cov = np.cov(changes_current, changes_lag)[0, 1]
                roll_measure = 2 * np.sqrt(max(0, -cov))  # Clip to non-negative

                features['roll_measure'] = roll_measure
            else:
                features['roll_measure'] = 0.0
        else:
            features['roll_measure'] = 0.0

        # 2-3. Order flow imbalance and VPIN
        buy_vol = candle.get('buy_volume', 0)
        sell_vol = candle.get('sell_volume', 0)
        total_vol = buy_vol + sell_vol

        if total_vol > 0:
            order_flow_imbalance = (buy_vol - sell_vol) / total_vol
        else:
            order_flow_imbalance = 0.0

        features['order_flow_imbalance'] = order_flow_imbalance

        self.order_flow_window.append(order_flow_imbalance)

        if self.order_flow_window.is_full():
            vpin = np.std(self.order_flow_window.to_array())
            features['vpin'] = vpin
        else:
            features['vpin'] = 0.0

        # 4. Bid-ask spread proxy (high-low range as % of price)
        close = candle['close']
        if close > 0:
            bid_ask_spread_pct = ((candle['high'] - candle['low']) / close) * 100
        else:
            bid_ask_spread_pct = 0.0

        features['bid_ask_spread_pct'] = bid_ask_spread_pct

        # 5. Order book depth ratio (volume relative to typical)
        if self.volume_window.is_full():
            vol_ma = np.mean(self.volume_window.to_array())
            if vol_ma > 0:
                order_book_depth_ratio = candle['volume'] / vol_ma
            else:
                order_book_depth_ratio = 1.0
        else:
            order_book_depth_ratio = 1.0

        features['order_book_depth_ratio'] = order_book_depth_ratio

        # 6. Large order imbalance
        self.buy_volume_window.append(buy_vol)
        self.sell_volume_window.append(sell_vol)

        if self.buy_volume_window.is_full() and self.sell_volume_window.is_full():
            buy_volumes = self.buy_volume_window.to_array()
            sell_volumes = self.sell_volume_window.to_array()

            large_buy_threshold = np.quantile(buy_volumes, 0.9)
            large_sell_threshold = np.quantile(sell_volumes, 0.9)

            large_buy = 1 if buy_vol > large_buy_threshold else 0
            large_sell = 1 if sell_vol > large_sell_threshold else 0

            large_order_imbalance = large_buy - large_sell
        else:
            large_order_imbalance = 0

        features['large_order_imbalance'] = large_order_imbalance

        return features

    def _calculate_multitimeframe_features(self, candle: Dict) -> Dict:
        """Calculate 4 multi-timeframe features (5m, 15m)."""
        features = {}

        # Add candle to multi-timeframe buffers
        self.candles_5m.append(candle)
        self.candles_15m.append(candle)

        # 5-minute features (recalculate every 5 candles)
        if len(self.candles_5m) == 5:
            candles_5m_list = list(self.candles_5m.data)

            # Aggregate to 5m candle
            open_5m = candles_5m_list[0]['open']
            close_5m = candles_5m_list[-1]['close']
            volume_5m = sum(c['volume'] for c in candles_5m_list)

            # Returns 5m
            if open_5m > 0:
                returns_5m = (close_5m - open_5m) / open_5m
            else:
                returns_5m = 0.0

            self.last_5m_returns = returns_5m

            # Volume z-score 5m (would need historical 5m volumes - use approximation)
            # For simplicity, use current volume zscore as proxy
            self.last_5m_volume_zscore = features.get('volume_zscore', 0.0)

        features['returns_5m'] = self.last_5m_returns if self.last_5m_returns is not None else 0.0
        features['volume_zscore_5m'] = self.last_5m_volume_zscore if self.last_5m_volume_zscore is not None else 0.0

        # 15-minute features (recalculate every 15 candles)
        if len(self.candles_15m) == 15:
            candles_15m_list = list(self.candles_15m.data)

            # Aggregate to 15m candle
            open_15m = candles_15m_list[0]['open']
            close_15m = candles_15m_list[-1]['close']
            volume_15m = sum(c['volume'] for c in candles_15m_list)

            # Returns 15m
            if open_15m > 0:
                returns_15m = (close_15m - open_15m) / open_15m
            else:
                returns_15m = 0.0

            self.last_15m_returns = returns_15m

            # Volume z-score 15m (approximation)
            self.last_15m_volume_zscore = features.get('volume_zscore', 0.0)

        features['returns_15m'] = self.last_15m_returns if self.last_15m_returns is not None else 0.0
        features['volume_zscore_15m'] = self.last_15m_volume_zscore if self.last_15m_volume_zscore is not None else 0.0

        return features


class FeatureCalculator:
    """
    Manages feature calculation for all symbols.

    Maintains per-symbol state and calculates 24 features in real-time.
    """

    def __init__(self, symbols: list[str]):
        """
        Initialize feature calculator.

        Args:
            symbols: List of symbols to track
        """
        self.symbols = symbols

        # Per-symbol feature calculation state
        self.symbol_states: Dict[str, SymbolFeatureState] = {}

        for symbol in symbols:
            self.symbol_states[symbol] = SymbolFeatureState(symbol)

        logger.info(f"FeatureCalculator initialized for {len(symbols)} symbols")

    def on_candle_complete(self, symbol: str, candle: Dict) -> Optional[Dict]:
        """
        Calculate features for completed candle.

        Args:
            symbol: Trading pair symbol
            candle: Completed 1-minute candle dict

        Returns:
            Dict of 24 features, or None if not enough history yet
        """
        if symbol not in self.symbol_states:
            logger.warning(f"Unknown symbol: {symbol}")
            return None

        state = self.symbol_states[symbol]
        features = state.update_candle(candle)

        return features

    def get_feature_names(self) -> list[str]:
        """Get ordered list of feature names (same as training)."""
        return [
            # Price features (9)
            'returns', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14', 'NATR',
            'BB_width', 'BB_squeeze', 'VWAP_distance',

            # Volume features (5)
            'volume_zscore', 'volume_roc', 'OBV', 'trade_count', 'buy_sell_ratio',

            # Microstructure features (6)
            'roll_measure', 'order_flow_imbalance', 'vpin', 'bid_ask_spread_pct',
            'order_book_depth_ratio', 'large_order_imbalance',

            # Multi-timeframe features (4)
            'returns_5m', 'volume_zscore_5m', 'returns_15m', 'volume_zscore_15m',
        ]
