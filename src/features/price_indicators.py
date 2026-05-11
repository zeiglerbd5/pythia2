"""
Price Action Indicators

Implements technical indicators per implementation guide:
- RSI: 14-period Relative Strength Index (overbought/oversold)
- VWAP: Volume-Weighted Average Price with ±1σ, ±2σ, ±3σ bands
- ATR/NATR: Average True Range (volatility measurement)
- Bollinger Bands: Squeeze detection for breakout prediction

Per guide: Focus on momentum and volatility rather than lagging MAs.
VWAP deviations >3-5% signal potential reversals in crypto.
Bollinger Band squeezes provide 1-4 hour advance notice of breakouts.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from loguru import logger


class RSI:
    """
    Relative Strength Index (RSI).

    Per implementation guide:
    - Standard 14-period calculation
    - Divergences more valuable than absolute levels for spike detection
    - Use faster settings (8-10 period) for crypto's rapid movements

    Traditional levels: <30 oversold, >70 overbought
    """

    @staticmethod
    def calculate(
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate RSI using Wilder's smoothing.

        Formula:
        1. Calculate price changes
        2. Separate gains and losses
        3. Calculate exponential MA of gains and losses
        4. RSI = 100 - (100 / (1 + RS))
           where RS = avg_gain / avg_loss

        Args:
            close: Close prices
            period: RSI period (default: 14)

        Returns:
            RSI series (0-100)
        """
        # Calculate price changes
        delta = close.diff()

        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()

        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        # Calculate Wilder's smoothing (exponential MA)
        alpha = 1.0 / period
        avg_gains = gains.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rs = rs.replace([np.inf, -np.inf], 100)  # Handle division by zero

        rsi = 100 - (100 / (1 + rs))

        return pd.Series(rsi, index=close.index, name='rsi')

    @staticmethod
    def detect_divergence(
        close: pd.Series,
        rsi: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Detect RSI divergences from price.

        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high

        Args:
            close: Close prices
            rsi: RSI values
            window: Lookback window

        Returns:
            Divergence indicator (1=bullish, -1=bearish, 0=none)
        """
        # Find local minima and maxima
        price_min = close.rolling(window, center=True).min() == close
        price_max = close.rolling(window, center=True).max() == close

        rsi_min = rsi.rolling(window, center=True).min() == rsi
        rsi_max = rsi.rolling(window, center=True).max() == rsi

        divergence = pd.Series(0, index=close.index)

        # Bullish divergence: price lower low, RSI higher low
        for i in range(window, len(close) - window):
            if price_min.iloc[i]:
                # Look for previous price minimum
                prev_price_mins = close[:i][price_min[:i]]
                if len(prev_price_mins) > 0 and close.iloc[i] < prev_price_mins.iloc[-1]:
                    # Price made lower low, check RSI
                    prev_rsi_mins = rsi[:i][rsi_min[:i]]
                    if len(prev_rsi_mins) > 0 and rsi.iloc[i] > prev_rsi_mins.iloc[-1]:
                        divergence.iloc[i] = 1  # Bullish divergence

        return divergence


class VWAP:
    """
    Volume-Weighted Average Price with Standard Deviation Bands.

    Per implementation guide:
    - VWAP = Σ(Price × Volume) / Σ Volume (reset daily)
    - Deviations >3-5% in crypto signal potential reversals
    - ±1σ, ±2σ, ±3σ bands for scaling positions
    - Functions as dynamic support/resistance

    VWAP is both an early signal and execution reference.
    """

    @staticmethod
    def calculate(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        reset_period: Optional[str] = 'D'
    ) -> pd.DataFrame:
        """
        Calculate VWAP and standard deviation bands.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volumes
            reset_period: Reset period ('D' for daily, None for cumulative)

        Returns:
            DataFrame with vwap, vwap_std, upper/lower bands
        """
        # Calculate typical price
        typical_price = (high + low + close) / 3

        # Calculate cumulative VWAP
        if reset_period:
            # Group by period (e.g., daily)
            groups = typical_price.index.to_period(reset_period)

            # Calculate VWAP per group
            vwap = (typical_price * volume).groupby(groups).cumsum() / volume.groupby(groups).cumsum()

            # Calculate standard deviation
            squared_diff = ((typical_price - vwap) ** 2 * volume).groupby(groups).cumsum()
            variance = squared_diff / volume.groupby(groups).cumsum()
            vwap_std = np.sqrt(variance)

        else:
            # Cumulative VWAP
            vwap = (typical_price * volume).cumsum() / volume.cumsum()

            # Cumulative standard deviation
            squared_diff = ((typical_price - vwap) ** 2 * volume).cumsum()
            variance = squared_diff / volume.cumsum()
            vwap_std = np.sqrt(variance)

        # Create bands
        result = pd.DataFrame({
            'vwap': vwap,
            'vwap_std': vwap_std,
            'vwap_upper_1': vwap + vwap_std,
            'vwap_lower_1': vwap - vwap_std,
            'vwap_upper_2': vwap + 2 * vwap_std,
            'vwap_lower_2': vwap - 2 * vwap_std,
            'vwap_upper_3': vwap + 3 * vwap_std,
            'vwap_lower_3': vwap - 3 * vwap_std,
        }, index=close.index)

        # Calculate distance from VWAP (percentage)
        result['vwap_distance_pct'] = ((close - vwap) / vwap) * 100

        return result

    @staticmethod
    def detect_significant_deviation(
        vwap_distance_pct: pd.Series,
        threshold: float = 5.0
    ) -> pd.Series:
        """
        Detect significant deviations from VWAP.

        Per guide: >3-5% deviation signals potential reversal

        Args:
            vwap_distance_pct: Distance from VWAP in percentage
            threshold: Deviation threshold (default: 5%)

        Returns:
            Boolean series indicating significant deviation
        """
        return abs(vwap_distance_pct) > threshold


class ATR:
    """
    Average True Range (ATR) and Normalized ATR.

    Per implementation guide:
    - ATR normalized by price (NATR) enables cross-asset comparison
    - Essential for adaptive stop-loss placement (2-3x ATR)
    - Captures volatility for position sizing adjustments

    True Range = max(high-low, |high-close_prev|, |low-close_prev|)
    """

    @staticmethod
    def calculate(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period (default: 14)

        Returns:
            ATR series
        """
        # Calculate True Range
        high_low = high - low
        high_close_prev = abs(high - close.shift(1))
        low_close_prev = abs(low - close.shift(1))

        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # Calculate ATR using Wilder's smoothing
        alpha = 1.0 / period
        atr = true_range.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        return pd.Series(atr, index=close.index, name='atr')

    @staticmethod
    def calculate_natr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Normalized ATR (NATR).

        NATR = (ATR / Close) * 100

        Enables comparison across different price levels.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            NATR series (percentage)
        """
        atr = ATR.calculate(high, low, close, period)
        natr = (atr / close) * 100

        return pd.Series(natr, index=close.index, name='natr')


class BollingerBands:
    """
    Bollinger Bands with Squeeze Detection.

    Per implementation guide:
    - 20-period MA with ±2 standard deviations
    - Band squeeze (width below 20-period average) signals low volatility
    - Squeezes precede breakouts by 1-4 hours
    - Use for identifying pre-breakout consolidation

    Bollinger Band width = (upper - lower) / middle
    """

    @staticmethod
    def calculate(
        close: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            close: Close prices
            period: MA period (default: 20)
            num_std: Number of standard deviations (default: 2.0)

        Returns:
            DataFrame with middle, upper, lower, width, %b
        """
        # Calculate middle band (SMA)
        middle = close.rolling(window=period).mean()

        # Calculate standard deviation
        std = close.rolling(window=period).std()

        # Calculate bands
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        # Calculate band width (normalized)
        width = (upper - lower) / middle

        # Calculate %B (position within bands)
        # %B = (close - lower) / (upper - lower)
        percent_b = (close - lower) / (upper - lower)

        result = pd.DataFrame({
            'bb_middle': middle,
            'bb_upper': upper,
            'bb_lower': lower,
            'BB_width': width,
            'bb_percent_b': percent_b
        }, index=close.index)

        return result

    @staticmethod
    def detect_squeeze(
        bb_width: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Detect Bollinger Band squeeze.

        Per guide: Width below 20-period average = squeeze
        Predicts breakouts 1-4 hours ahead

        Args:
            bb_width: Bollinger Band width
            window: Lookback for average width

        Returns:
            Boolean series indicating squeeze
        """
        avg_width = bb_width.rolling(window).mean()
        squeeze = bb_width < avg_width

        return squeeze


def calculate_price_features(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20
) -> pd.DataFrame:
    """
    Calculate all price-based features.

    Per implementation guide: Focus on momentum and volatility,
    not lagging moving averages.

    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volumes
        rsi_period: RSI period (default: 14)
        atr_period: ATR period (default: 14)
        bb_period: Bollinger Band period (default: 20)

    Returns:
        DataFrame with all price features
    """
    features = pd.DataFrame(index=close.index)

    # RSI (14-period per guide)
    features['rsi'] = RSI.calculate(close, period=rsi_period)
    features['rsi_overbought'] = features['rsi'] > 70
    features['rsi_oversold'] = features['rsi'] < 30

    # VWAP with bands (per guide: deviations >3-5%)
    # Using cumulative VWAP (faster, no timezone conversion overhead)
    vwap_data = VWAP.calculate(high, low, close, volume, reset_period=None)
    features = pd.concat([features, vwap_data], axis=1)
    features['vwap_deviation_significant'] = VWAP.detect_significant_deviation(
        features['vwap_distance_pct'], threshold=5.0
    )

    # ATR and NATR (per guide: normalized for cross-asset comparison)
    features['atr'] = ATR.calculate(high, low, close, period=atr_period)
    features['natr'] = ATR.calculate_natr(high, low, close, period=atr_period)

    # Bollinger Bands (squeeze detection per guide)
    bb_data = BollingerBands.calculate(close, period=bb_period, num_std=2.0)
    features = pd.concat([features, bb_data], axis=1)
    features['bb_squeeze'] = BollingerBands.detect_squeeze(
        features['BB_width'], window=20
    )

    logger.debug(
        f"Calculated price features",
        extra={
            "features": list(features.columns),
            "rows": len(features),
            "rsi_mean": features['rsi'].mean(),
            "atr_mean": features['atr'].mean()
        }
    )

    return features


if __name__ == "__main__":
    # Test price indicators
    import numpy as np

    np.random.seed(42)

    # Generate synthetic OHLCV data
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='5min')

    # Random walk price
    close = pd.Series(
        np.cumsum(np.random.randn(n) * 0.5) + 100,
        index=dates
    )

    # Generate OHLC from close
    high = close + np.random.rand(n) * 0.5
    low = close - np.random.rand(n) * 0.5
    open_ = close.shift(1) + np.random.randn(n) * 0.2
    open_.iloc[0] = close.iloc[0]

    # Volume
    volume = pd.Series(
        np.random.exponential(1000, n),
        index=dates
    )

    print("=== Price Indicators Test ===\n")

    # Test RSI
    rsi = RSI.calculate(close, period=14)
    print(f"RSI:")
    print(f"  Current: {rsi.iloc[-1]:.2f}")
    print(f"  Overbought (>70): {(rsi > 70).sum()} bars")
    print(f"  Oversold (<30): {(rsi < 30).sum()} bars")

    # Test VWAP
    vwap_data = VWAP.calculate(high, low, close, volume, reset_period=None)
    print(f"\nVWAP:")
    print(f"  Current Price: {close.iloc[-1]:.2f}")
    print(f"  VWAP: {vwap_data['vwap'].iloc[-1]:.2f}")
    print(f"  Distance: {vwap_data['vwap_distance_pct'].iloc[-1]:.2f}%")
    print(f"  Deviation >5%: {(abs(vwap_data['vwap_distance_pct']) > 5).sum()} bars")

    # Test ATR
    atr = ATR.calculate(high, low, close, period=14)
    natr = ATR.calculate_natr(high, low, close, period=14)
    print(f"\nATR:")
    print(f"  ATR: {atr.iloc[-1]:.2f}")
    print(f"  NATR: {natr.iloc[-1]:.2f}%")
    print(f"  2x ATR Stop: ±{atr.iloc[-1] * 2:.2f}")

    # Test Bollinger Bands
    bb_data = BollingerBands.calculate(close, period=20, num_std=2.0)
    squeeze = BollingerBands.detect_squeeze(bb_data['BB_width'], window=20)
    print(f"\nBollinger Bands:")
    print(f"  Width: {bb_data['BB_width'].iloc[-1]:.4f}")
    print(f"  %B: {bb_data['bb_percent_b'].iloc[-1]:.2f}")
    print(f"  Squeeze Active: {'Yes' if squeeze.iloc[-1] else 'No'}")
    print(f"  Squeezes Detected: {squeeze.sum()} bars")

    # Calculate all features
    features = calculate_price_features(open_, high, low, close, volume)

    print(f"\n=== All Price Features Summary ===")
    print(features.describe())

    print(f"\n=== Recent Values (last 5 bars) ===")
    print(features.tail())
