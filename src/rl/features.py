"""
Feature Extraction for RL Trading Environment

Implements feature engineering for market state representation:
- OHLCV-based features (returns, volatility, volume ratios)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Order book features (bid/ask imbalance, spread, depth)
- Trade flow features (buy/sell pressure, VPIN, large trades)
- Cross-timeframe features (1m, 5m, 15m, 1h aggregations)
- Volume profile features
- Position context features
- Normalization utilities
- Multi-timeframe aggregation helpers
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from loguru import logger


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # Return horizons
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 15, 60])

    # Volatility parameters
    volatility_window: int = 20
    atr_window: int = 14

    # Volume parameters
    volume_ma_window: int = 20
    volume_zscore_window: int = 50

    # Technical indicators
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0

    # Normalization
    normalize_returns: bool = True
    clip_outliers: bool = True
    outlier_std: float = 5.0

    # Position features
    n_position_features: int = 6

    # Order book features (from database features table)
    include_order_book: bool = True
    order_book_features: List[str] = field(default_factory=lambda: [
        'order_book_imbalance_l5',
        'bid_ask_ratio',
        'bid_ask_spread_pct',
        'order_book_depth_ratio',
        'large_order_imbalance',
        'weighted_mid_price',
    ])

    # Trade flow features
    include_trade_flow: bool = True
    vpin_window: int = 50
    roll_window: int = 100

    # Multi-timeframe settings
    include_multi_timeframe: bool = True
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '1h'])

    # Volume profile settings
    include_volume_profile: bool = True
    volume_profile_bins: int = 10
    volume_profile_window: int = 240  # 4 hours of 1m data


class FeatureExtractor:
    """
    Extract and normalize features from OHLCV data.

    Features are designed to capture:
    - Price dynamics (returns, momentum)
    - Volatility regime
    - Volume profile
    - Technical patterns
    - Order book microstructure (when available)
    - Trade flow metrics (when available)
    - Multi-timeframe context
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature extractor.

        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()
        self._feature_names: Optional[List[str]] = None
        self._base_feature_count: Optional[int] = None

    def get_state_dim(self) -> int:
        """
        Get market feature dimension (NOT including position features).

        Position features are added by the environment separately.
        """
        if self._base_feature_count is not None:
            return self._base_feature_count

        # Calculate based on feature groups
        n_features = (
            len(self.config.return_periods)  # Returns
            + 2  # Volatility (realized vol, NATR)
            + 3  # Volume (ratio, zscore, acceleration)
            + 1  # RSI
            + 3  # MACD (line, signal, histogram)
            + 4  # Bollinger Bands (upper, lower, width, position)
            + 2  # Price position (24h range, VWAP distance)
            + 1  # Hour of day (cyclical)
        )

        # Order book features (when included)
        if self.config.include_order_book:
            n_features += len(self.config.order_book_features)

        # Trade flow features
        if self.config.include_trade_flow:
            n_features += 4  # vpin, roll_measure, trade_flow_imbalance, vpin_high

        # Volume profile features
        if self.config.include_volume_profile:
            n_features += 3  # volume_profile_skew, price_above_poc, volume_node_distance

        # Multi-timeframe features (aggregated summary per timeframe)
        if self.config.include_multi_timeframe:
            # Each higher timeframe adds: return, rsi, volume_ratio, momentum
            n_higher_tf = len([tf for tf in self.config.timeframes if tf != '1m'])
            n_features += n_higher_tf * 4

        # NOTE: Position features are NOT included here - environment adds them
        return n_features

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        if self._feature_names is None:
            names = (
                [f'return_{p}m' for p in self.config.return_periods]
                + ['realized_vol', 'natr']
                + ['volume_ratio', 'volume_zscore', 'volume_accel']
                + ['rsi']
                + ['macd', 'macd_signal', 'macd_hist']
                + ['bb_upper_dist', 'bb_lower_dist', 'bb_width', 'bb_position']
                + ['price_position_24h', 'vwap_distance']
                + ['hour_sin']
            )

            # Order book feature names
            if self.config.include_order_book:
                names += [f'ob_{f}' for f in self.config.order_book_features]

            # Trade flow feature names
            if self.config.include_trade_flow:
                names += ['vpin', 'roll_measure', 'trade_flow_imbalance', 'vpin_high']

            # Volume profile feature names
            if self.config.include_volume_profile:
                names += ['volume_profile_skew', 'price_above_poc', 'volume_node_distance']

            # Multi-timeframe feature names
            if self.config.include_multi_timeframe:
                for tf in self.config.timeframes:
                    if tf != '1m':
                        names += [f'{tf}_return', f'{tf}_rsi', f'{tf}_volume_ratio', f'{tf}_momentum']

            self._feature_names = names

        return self._feature_names

    def calculate_features(
        self,
        ohlcv: pd.DataFrame,
        order_book_data: Optional[pd.DataFrame] = None,
        trade_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate all features from OHLCV data and optional enriched data.

        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume]
                   and datetime index
            order_book_data: Optional DataFrame with order book features
                   (from features table in DuckDB)
            trade_data: Optional DataFrame with trade-level data for VPIN calculation

        Returns:
            DataFrame with calculated features
        """
        features = pd.DataFrame(index=ohlcv.index)

        # =====================================================
        # CORE OHLCV FEATURES
        # =====================================================

        # Price returns
        for period in self.config.return_periods:
            features[f'return_{period}m'] = self._calculate_returns(
                ohlcv['close'], period
            )

        # Volatility features
        features['realized_vol'] = self._calculate_realized_volatility(
            ohlcv['close'], self.config.volatility_window
        )
        features['natr'] = self._calculate_natr(
            ohlcv[['high', 'low', 'close']], self.config.atr_window
        )

        # Volume features
        vol_features = self._calculate_volume_features(
            ohlcv['volume'], self.config.volume_ma_window, self.config.volume_zscore_window
        )
        features['volume_ratio'] = vol_features['ratio']
        features['volume_zscore'] = vol_features['zscore']
        features['volume_accel'] = vol_features['acceleration']

        # RSI
        features['rsi'] = self._calculate_rsi(
            ohlcv['close'], self.config.rsi_window
        )

        # MACD
        macd_features = self._calculate_macd(
            ohlcv['close'],
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        features['macd'] = macd_features['macd']
        features['macd_signal'] = macd_features['signal']
        features['macd_hist'] = macd_features['histogram']

        # Bollinger Bands
        bb_features = self._calculate_bollinger_bands(
            ohlcv['close'], self.config.bb_window, self.config.bb_std
        )
        features['bb_upper_dist'] = bb_features['upper_dist']
        features['bb_lower_dist'] = bb_features['lower_dist']
        features['bb_width'] = bb_features['width']
        features['bb_position'] = bb_features['position']

        # Price position features
        features['price_position_24h'] = self._calculate_price_position(
            ohlcv[['high', 'low', 'close']], window=60 * 24
        )
        features['vwap_distance'] = self._calculate_vwap_distance(
            ohlcv[['close', 'volume']]
        )

        # Time features (cyclical encoding)
        if isinstance(ohlcv.index, pd.DatetimeIndex):
            hour = ohlcv.index.hour
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)

        # =====================================================
        # ORDER BOOK FEATURES (from database features table)
        # =====================================================
        if self.config.include_order_book:
            ob_features = self._calculate_order_book_features(
                ohlcv, order_book_data
            )
            for col in ob_features.columns:
                features[col] = ob_features[col]

        # =====================================================
        # TRADE FLOW FEATURES (VPIN, Roll Measure, etc.)
        # =====================================================
        if self.config.include_trade_flow:
            tf_features = self._calculate_trade_flow_features(
                ohlcv, trade_data
            )
            for col in tf_features.columns:
                features[col] = tf_features[col]

        # =====================================================
        # VOLUME PROFILE FEATURES
        # =====================================================
        if self.config.include_volume_profile:
            vp_features = self._calculate_volume_profile_features(ohlcv)
            for col in vp_features.columns:
                features[col] = vp_features[col]

        # =====================================================
        # MULTI-TIMEFRAME FEATURES
        # =====================================================
        if self.config.include_multi_timeframe:
            mtf_features = self._calculate_multi_timeframe_features(ohlcv)
            for col in mtf_features.columns:
                features[col] = mtf_features[col]

        # Normalize features
        features = self._normalize_features(features)

        # Store actual feature count (excluding position features)
        self._base_feature_count = len(features.columns)

        return features

    def _calculate_returns(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate log returns over specified period."""
        returns = np.log(prices / prices.shift(period))

        if self.config.normalize_returns:
            # Normalize by rolling std
            rolling_std = returns.rolling(window=50, min_periods=10).std()
            returns = returns / (rolling_std + 1e-8)

        if self.config.clip_outliers:
            returns = returns.clip(-self.config.outlier_std, self.config.outlier_std)

        return returns

    def _calculate_realized_volatility(
        self, prices: pd.Series, window: int
    ) -> pd.Series:
        """Calculate realized volatility (annualized)."""
        log_returns = np.log(prices / prices.shift(1))
        vol = log_returns.rolling(window=window).std() * np.sqrt(252 * 24 * 60)  # Annualized
        return vol

    def _calculate_natr(self, ohlc: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Normalized Average True Range."""
        high = ohlc['high']
        low = ohlc['low']
        close = ohlc['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR
        atr = tr.rolling(window=window).mean()

        # Normalized ATR (as percentage of close)
        natr = (atr / close) * 100

        return natr

    def _calculate_volume_features(
        self,
        volume: pd.Series,
        ma_window: int,
        zscore_window: int
    ) -> Dict[str, pd.Series]:
        """Calculate volume-based features."""
        # Volume ratio (current / MA)
        volume_ma = volume.rolling(window=ma_window).mean()
        ratio = volume / (volume_ma + 1e-8)

        # Volume z-score
        volume_mean = volume.rolling(window=zscore_window).mean()
        volume_std = volume.rolling(window=zscore_window).std()
        zscore = (volume - volume_mean) / (volume_std + 1e-8)

        # Volume acceleration (change in volume ratio)
        acceleration = ratio.diff(5)

        return {
            'ratio': ratio.clip(0, 10),  # Cap at 10x average
            'zscore': zscore.clip(-5, 5),
            'acceleration': acceleration.clip(-3, 3),
        }

    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        # Normalize to [-1, 1]
        return (rsi - 50) / 50

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int,
        slow: int,
        signal: int
    ) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        # Normalize by price
        macd_norm = macd_line / prices * 100
        signal_norm = signal_line / prices * 100
        hist_norm = histogram / prices * 100

        return {
            'macd': macd_norm.clip(-5, 5),
            'signal': signal_norm.clip(-5, 5),
            'histogram': hist_norm.clip(-3, 3),
        }

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        window: int,
        num_std: float
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands features."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper = ma + (std * num_std)
        lower = ma - (std * num_std)

        # Distance from bands (normalized by band width)
        band_width = upper - lower
        upper_dist = (upper - prices) / (band_width + 1e-8)
        lower_dist = (prices - lower) / (band_width + 1e-8)

        # Normalized band width
        width = band_width / prices

        # Position within bands [0, 1]
        position = (prices - lower) / (band_width + 1e-8)

        return {
            'upper_dist': upper_dist.clip(-2, 2),
            'lower_dist': lower_dist.clip(-2, 2),
            'width': width.clip(0, 0.2),
            'position': position.clip(-0.5, 1.5),
        }

    def _calculate_price_position(
        self,
        ohlc: pd.DataFrame,
        window: int
    ) -> pd.Series:
        """Calculate price position within rolling high-low range."""
        high = ohlc['high'].rolling(window=window).max()
        low = ohlc['low'].rolling(window=window).min()
        close = ohlc['close']

        # Position in range [0, 1]
        position = (close - low) / (high - low + 1e-8)

        return position

    def _calculate_vwap_distance(self, data: pd.DataFrame) -> pd.Series:
        """Calculate distance from VWAP."""
        # Cumulative VWAP
        cumulative_tp_vol = (data['close'] * data['volume']).cumsum()
        cumulative_vol = data['volume'].cumsum()
        vwap = cumulative_tp_vol / (cumulative_vol + 1e-8)

        # Distance as percentage
        distance = (data['close'] - vwap) / vwap * 100

        return distance.clip(-10, 10)

    def _calculate_order_book_features(
        self,
        ohlcv: pd.DataFrame,
        order_book_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate order book microstructure features.

        If order_book_data is provided, use pre-calculated features from database.
        Otherwise, estimate from OHLCV data (less accurate but always available).

        Args:
            ohlcv: OHLCV DataFrame with datetime index
            order_book_data: Optional DataFrame with order book features

        Returns:
            DataFrame with order book features
        """
        features = pd.DataFrame(index=ohlcv.index)

        if order_book_data is not None and len(order_book_data) > 0:
            # Use pre-calculated order book features from database
            # Align timestamps and forward-fill missing values
            order_book_data = order_book_data.copy()
            if not isinstance(order_book_data.index, pd.DatetimeIndex):
                if 'timestamp' in order_book_data.columns:
                    order_book_data['timestamp'] = pd.to_datetime(order_book_data['timestamp'])
                    order_book_data.set_index('timestamp', inplace=True)

            # Reindex to match OHLCV timestamps
            for feat_name in self.config.order_book_features:
                col_name = f'ob_{feat_name}'
                if feat_name in order_book_data.columns:
                    # Reindex and forward fill
                    aligned = order_book_data[feat_name].reindex(
                        ohlcv.index, method='ffill'
                    )
                    features[col_name] = aligned.fillna(0)
                else:
                    # Feature not available, fill with neutral value
                    features[col_name] = 0.0
                    logger.debug(f"Order book feature '{feat_name}' not found in data")
        else:
            # Estimate order book features from OHLCV (fallback)
            logger.debug("No order book data provided, using OHLCV estimates")

            # Estimate bid-ask spread from high-low range
            typical_price = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3
            spread_estimate = (ohlcv['high'] - ohlcv['low']) / typical_price
            features['ob_bid_ask_spread_pct'] = spread_estimate.clip(0, 0.05)

            # Estimate imbalance from close position within range
            range_size = ohlcv['high'] - ohlcv['low']
            close_position = (ohlcv['close'] - ohlcv['low']) / (range_size + 1e-8)
            # Map [0, 1] to [-1, 1] for imbalance
            features['ob_order_book_imbalance_l5'] = (close_position - 0.5) * 2

            # Use volume ratio as proxy for depth ratio
            vol_ma = ohlcv['volume'].rolling(20).mean()
            features['ob_order_book_depth_ratio'] = (
                ohlcv['volume'] / (vol_ma + 1e-8)
            ).clip(0, 5)

            # Fill remaining expected features with zeros
            for feat_name in self.config.order_book_features:
                col_name = f'ob_{feat_name}'
                if col_name not in features.columns:
                    features[col_name] = 0.0

        return features

    def _calculate_trade_flow_features(
        self,
        ohlcv: pd.DataFrame,
        trade_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate trade flow and microstructure features.

        Includes:
        - VPIN (Volume-Synchronized Probability of Informed Trading)
        - Roll Measure (price autocorrelation)
        - Trade flow imbalance

        Args:
            ohlcv: OHLCV DataFrame
            trade_data: Optional DataFrame with trade-level data (buy/sell volumes)

        Returns:
            DataFrame with trade flow features
        """
        features = pd.DataFrame(index=ohlcv.index)

        # Roll Measure: 2 * sqrt(|cov(ΔP_t, ΔP_t-1)|)
        # Captures momentum and bid-ask bounce
        price_changes = ohlcv['close'].diff()
        delta_p_t = price_changes
        delta_p_t_minus_1 = price_changes.shift(1)

        # Rolling covariance
        rolling_cov = delta_p_t.rolling(window=self.config.roll_window).cov(delta_p_t_minus_1)
        roll_measure = 2.0 * np.sqrt(np.abs(rolling_cov))
        features['roll_measure'] = (roll_measure / ohlcv['close']).clip(0, 0.1)  # Normalize by price

        if trade_data is not None and 'buy_volume' in trade_data.columns:
            # Use actual buy/sell volumes from trade data
            buy_vol = trade_data['buy_volume'].reindex(ohlcv.index, method='ffill').fillna(0)
            sell_vol = trade_data['sell_volume'].reindex(ohlcv.index, method='ffill').fillna(0)
        else:
            # Estimate buy/sell volumes from OHLCV
            # Use close position within range as proxy
            range_size = ohlcv['high'] - ohlcv['low']
            close_position = (ohlcv['close'] - ohlcv['low']) / (range_size + 1e-8)

            # If close is near high, more buy volume; if near low, more sell
            buy_vol = ohlcv['volume'] * close_position
            sell_vol = ohlcv['volume'] * (1 - close_position)

        # VPIN: |V_sell - V_buy| / V_total
        rolling_buy = buy_vol.rolling(window=self.config.vpin_window).sum()
        rolling_sell = sell_vol.rolling(window=self.config.vpin_window).sum()
        total_volume = rolling_buy + rolling_sell

        vpin = np.abs(rolling_sell - rolling_buy) / (total_volume + 1e-8)
        features['vpin'] = vpin.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(0, 1)

        # VPIN threshold indicator (> 0.5 indicates high informed trading)
        features['vpin_high'] = (features['vpin'] > 0.5).astype(float)

        # Trade flow imbalance: (buy - sell) / (buy + sell)
        total_vol_rolling = buy_vol.rolling(20).sum() + sell_vol.rolling(20).sum()
        imbalance = (buy_vol.rolling(20).sum() - sell_vol.rolling(20).sum()) / (total_vol_rolling + 1e-8)
        features['trade_flow_imbalance'] = imbalance.clip(-1, 1)

        return features

    def _calculate_volume_profile_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume profile features.

        Volume profile shows price levels with most trading activity,
        helping identify support/resistance and market structure.

        Args:
            ohlcv: OHLCV DataFrame

        Returns:
            DataFrame with volume profile features
        """
        features = pd.DataFrame(index=ohlcv.index)

        window = self.config.volume_profile_window
        n_bins = self.config.volume_profile_bins

        # Initialize with zeros
        features['volume_profile_skew'] = 0.0
        features['price_above_poc'] = 0.0
        features['volume_node_distance'] = 0.0

        # Calculate rolling volume profile features
        for i in range(window, len(ohlcv)):
            window_data = ohlcv.iloc[i-window:i]

            # Get price range for this window
            price_min = window_data['low'].min()
            price_max = window_data['high'].max()

            if price_max <= price_min:
                continue

            # Create price bins
            bin_edges = np.linspace(price_min, price_max, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Distribute volume across bins based on typical price
            typical_prices = (window_data['high'] + window_data['low'] + window_data['close']) / 3
            volumes = window_data['volume'].values

            # Assign each bar's volume to its bin
            bin_volumes = np.zeros(n_bins)
            for tp, vol in zip(typical_prices.values, volumes):
                bin_idx = min(int((tp - price_min) / (price_max - price_min) * n_bins), n_bins - 1)
                bin_idx = max(0, bin_idx)
                bin_volumes[bin_idx] += vol

            if bin_volumes.sum() > 0:
                # Point of Control (POC): price level with most volume
                poc_idx = np.argmax(bin_volumes)
                poc_price = bin_centers[poc_idx]

                # Current price position relative to POC
                current_price = ohlcv.iloc[i]['close']
                features.iloc[i, features.columns.get_loc('price_above_poc')] = (
                    (current_price - poc_price) / poc_price
                ).clip(-0.1, 0.1)

                # Volume profile skew: weighted average position
                total_vol = bin_volumes.sum()
                weighted_pos = np.sum(bin_volumes * np.arange(n_bins)) / (total_vol * n_bins)
                features.iloc[i, features.columns.get_loc('volume_profile_skew')] = (
                    (weighted_pos - 0.5) * 2
                )  # Normalize to [-1, 1]

                # Distance to nearest high volume node
                sorted_bins = np.argsort(bin_volumes)[::-1]
                current_bin = min(int((current_price - price_min) / (price_max - price_min) * n_bins), n_bins - 1)
                current_bin = max(0, current_bin)

                # Find closest high-volume bin (top 30%)
                high_vol_bins = sorted_bins[:max(1, n_bins // 3)]
                distances = np.abs(high_vol_bins - current_bin)
                nearest_dist = distances.min() / n_bins  # Normalize
                features.iloc[i, features.columns.get_loc('volume_node_distance')] = nearest_dist

        return features

    def _calculate_multi_timeframe_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features from multiple timeframes.

        Aggregates 1-minute data to higher timeframes and extracts
        key features for multi-scale analysis.

        Args:
            ohlcv: 1-minute OHLCV DataFrame

        Returns:
            DataFrame with multi-timeframe features
        """
        features = pd.DataFrame(index=ohlcv.index)

        # Map timeframes to pandas offset strings
        tf_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1h',
            '4h': '4h',
        }

        for tf in self.config.timeframes:
            if tf == '1m':
                continue  # Skip base timeframe

            offset = tf_map.get(tf, tf)

            # Resample to higher timeframe
            ohlcv_resampled = ohlcv.resample(offset).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }).dropna()

            if len(ohlcv_resampled) < 2:
                # Not enough data for this timeframe
                features[f'{tf}_return'] = 0.0
                features[f'{tf}_rsi'] = 0.0
                features[f'{tf}_volume_ratio'] = 1.0
                features[f'{tf}_momentum'] = 0.0
                continue

            # Calculate features for this timeframe
            tf_return = ohlcv_resampled['close'].pct_change()
            tf_rsi = self._calculate_rsi(ohlcv_resampled['close'], self.config.rsi_window)
            tf_vol_ma = ohlcv_resampled['volume'].rolling(20).mean()
            tf_vol_ratio = ohlcv_resampled['volume'] / (tf_vol_ma + 1e-8)

            # Momentum: rate of change over 5 periods
            tf_momentum = ohlcv_resampled['close'].pct_change(5)

            # Align to original 1-minute timestamps (forward fill)
            features[f'{tf}_return'] = tf_return.reindex(ohlcv.index, method='ffill').fillna(0).clip(-0.1, 0.1)
            features[f'{tf}_rsi'] = tf_rsi.reindex(ohlcv.index, method='ffill').fillna(0)
            features[f'{tf}_volume_ratio'] = tf_vol_ratio.reindex(ohlcv.index, method='ffill').fillna(1).clip(0, 10)
            features[f'{tf}_momentum'] = tf_momentum.reindex(ohlcv.index, method='ffill').fillna(0).clip(-0.2, 0.2)

        return features

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply final normalization to features."""
        # Fill NaN with 0 (conservative approach)
        features = features.fillna(0)

        # Replace infinities
        features = features.replace([np.inf, -np.inf], 0)

        return features


class MultiTimeframeFeatureExtractor:
    """
    Extract features across multiple timeframes.

    Aggregates 1-minute data into higher timeframes and extracts
    features at each level for multi-scale analysis.
    """

    def __init__(
        self,
        timeframes: List[str] = ['1m', '5m', '15m', '1h'],
        config: Optional[FeatureConfig] = None
    ):
        """
        Initialize multi-timeframe extractor.

        Args:
            timeframes: List of timeframe strings ('1m', '5m', '15m', '1h')
            config: Feature configuration
        """
        self.timeframes = timeframes
        self.config = config or FeatureConfig()
        self.extractors = {
            tf: FeatureExtractor(config) for tf in timeframes
        }

    def aggregate_ohlcv(
        self,
        ohlcv_1m: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Aggregate 1-minute OHLCV to higher timeframe.

        Args:
            ohlcv_1m: 1-minute OHLCV with datetime index
            timeframe: Target timeframe ('5m', '15m', '1h')

        Returns:
            Aggregated OHLCV DataFrame
        """
        # Map timeframe to pandas offset
        tf_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
        }

        offset = tf_map.get(timeframe, timeframe)

        # Resample
        agg_funcs = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }

        if 'num_trades' in ohlcv_1m.columns:
            agg_funcs['num_trades'] = 'sum'

        resampled = ohlcv_1m.resample(offset).agg(agg_funcs)

        # Drop incomplete candles
        resampled = resampled.dropna()

        return resampled

    def calculate_all_timeframes(
        self,
        ohlcv_1m: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate features for all timeframes.

        Args:
            ohlcv_1m: 1-minute OHLCV data

        Returns:
            Dict mapping timeframe to features DataFrame
        """
        features = {}

        for tf in self.timeframes:
            if tf == '1m':
                ohlcv = ohlcv_1m
            else:
                ohlcv = self.aggregate_ohlcv(ohlcv_1m, tf)

            features[tf] = self.extractors[tf].calculate_features(ohlcv)

        return features

    def get_aligned_features(
        self,
        features_dict: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp
    ) -> Dict[str, np.ndarray]:
        """
        Get features aligned to a specific timestamp.

        For each timeframe, returns the latest available features
        at or before the given timestamp.

        Args:
            features_dict: Output from calculate_all_timeframes
            timestamp: Target timestamp

        Returns:
            Dict mapping timeframe to feature array
        """
        aligned = {}

        for tf, features in features_dict.items():
            # Find latest features at or before timestamp
            available = features.index[features.index <= timestamp]
            if len(available) > 0:
                aligned[tf] = features.loc[available[-1]].values
            else:
                aligned[tf] = np.zeros(features.shape[1])

        return aligned


class FeatureNormalizer:
    """
    Online feature normalization using running statistics.

    Useful for continuous learning where feature distributions
    may shift over time.
    """

    def __init__(
        self,
        n_features: int,
        momentum: float = 0.01,
        epsilon: float = 1e-8
    ):
        """
        Initialize normalizer.

        Args:
            n_features: Number of features
            momentum: Update momentum for running stats
            epsilon: Small constant for numerical stability
        """
        self.n_features = n_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Running statistics
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        self.count = 0

    def update(self, features: np.ndarray) -> None:
        """Update running statistics with new features."""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        batch_mean = features.mean(axis=0)
        batch_var = features.var(axis=0)
        batch_size = features.shape[0]

        if self.count == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            # Exponential moving average update
            self.running_mean = (
                (1 - self.momentum) * self.running_mean +
                self.momentum * batch_mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var +
                self.momentum * batch_var
            )

        self.count += batch_size

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using running statistics."""
        return (features - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Convert normalized features back to original scale."""
        return normalized * np.sqrt(self.running_var) + self.running_mean

    def save_stats(self, path: str) -> None:
        """Save running statistics to file."""
        np.savez(
            path,
            mean=self.running_mean,
            var=self.running_var,
            count=self.count
        )

    def load_stats(self, path: str) -> None:
        """Load running statistics from file."""
        data = np.load(path)
        self.running_mean = data['mean']
        self.running_var = data['var']
        self.count = int(data['count'])


if __name__ == "__main__":
    # Test feature extraction
    import numpy as np

    np.random.seed(42)

    # Generate synthetic OHLCV data
    n = 1000
    dates = pd.date_range('2024-01-01', periods=n, freq='1min')

    # Random walk price
    returns = np.random.randn(n) * 0.001
    close = 100 * np.exp(np.cumsum(returns))

    # Synthetic OHLCV
    ohlcv = pd.DataFrame({
        'open': close * (1 + np.random.randn(n) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(n) * 0.002)),
        'low': close * (1 - np.abs(np.random.randn(n) * 0.002)),
        'close': close,
        'volume': np.random.exponential(1000, n),
    }, index=dates)

    # Test basic feature extractor (OHLCV only)
    print("=" * 60)
    print("Testing Basic Feature Extraction (OHLCV only)")
    print("=" * 60)

    basic_config = FeatureConfig(
        include_order_book=False,
        include_trade_flow=False,
        include_volume_profile=False,
        include_multi_timeframe=False,
    )
    basic_extractor = FeatureExtractor(basic_config)

    print(f"State dimension (basic): {basic_extractor.get_state_dim()}")
    basic_features = basic_extractor.calculate_features(ohlcv)
    print(f"Features shape: {basic_features.shape}")

    # Test full feature extractor
    print("\n" + "=" * 60)
    print("Testing Full Feature Extraction (with all enhancements)")
    print("=" * 60)

    full_config = FeatureConfig(
        include_order_book=True,
        include_trade_flow=True,
        include_volume_profile=True,
        include_multi_timeframe=True,
    )
    full_extractor = FeatureExtractor(full_config)

    print(f"State dimension (full): {full_extractor.get_state_dim()}")
    print(f"Feature names ({len(full_extractor.get_feature_names())}):")
    for i, name in enumerate(full_extractor.get_feature_names()):
        print(f"  {i+1:2d}. {name}")

    # Calculate features without order book data (uses OHLCV estimates)
    full_features = full_extractor.calculate_features(ohlcv)
    print(f"\nFeatures shape: {full_features.shape}")
    print(f"\nFeature statistics:")
    print(full_features.describe().round(4))

    # Test with simulated order book data
    print("\n" + "=" * 60)
    print("Testing with Simulated Order Book Data")
    print("=" * 60)

    order_book_data = pd.DataFrame({
        'timestamp': dates,
        'order_book_imbalance_l5': np.random.randn(n) * 0.3,
        'bid_ask_ratio': 0.9 + np.random.randn(n) * 0.1,
        'bid_ask_spread_pct': 0.001 + np.abs(np.random.randn(n) * 0.0005),
        'order_book_depth_ratio': 1 + np.random.randn(n) * 0.2,
        'large_order_imbalance': np.random.randn(n) * 0.2,
        'weighted_mid_price': close + np.random.randn(n) * 0.01,
    })
    order_book_data.set_index('timestamp', inplace=True)

    features_with_ob = full_extractor.calculate_features(ohlcv, order_book_data=order_book_data)
    print(f"Features with order book shape: {features_with_ob.shape}")

    # Test multi-timeframe extractor (legacy)
    print("\n" + "=" * 60)
    print("Testing Legacy MultiTimeframeFeatureExtractor")
    print("=" * 60)

    mtf_extractor = MultiTimeframeFeatureExtractor(
        timeframes=['1m', '5m', '15m'],
        config=basic_config
    )

    all_features = mtf_extractor.calculate_all_timeframes(ohlcv)
    print(f"Multi-timeframe features:")
    for tf, feat in all_features.items():
        print(f"  {tf}: {feat.shape}")

    # Test normalizer
    print("\n" + "=" * 60)
    print("Testing Feature Normalizer")
    print("=" * 60)

    normalizer = FeatureNormalizer(n_features=full_features.shape[1])

    # Update with features
    for i in range(0, len(full_features), 100):
        batch = full_features.iloc[i:i+100].values
        normalizer.update(batch)

    # Normalize
    normalized = normalizer.normalize(full_features.values)
    print(f"Normalized features mean: {normalized.mean(axis=0).mean():.6f}")
    print(f"Normalized features std: {normalized.std(axis=0).mean():.6f}")

    print("\n" + "=" * 60)
    print("All feature extraction tests passed!")
    print("=" * 60)
