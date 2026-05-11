"""
Volume Pattern Analysis Indicators

Implements volume-based indicators per implementation guide:
- Volume Spike Ratio: current / 20-period avg > 2.0 (anomaly detection)
- OBV: On-Balance Volume (cumulative volume flow)
- VROC: Volume Rate of Change (acceleration detection)

Per guide: OBV divergences from price provide powerful early signals,
with 25%+ OBV increase while price flat indicating hidden buying pressure.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from loguru import logger


class OnBalanceVolume:
    """
    On-Balance Volume (OBV) - Cumulative Volume Flow Indicator.

    Per implementation guide:
    - Add volume on up-closes, subtract on down-closes
    - OBV divergences from price = early warning signals
    - 25%+ OBV increase while price flat = hidden accumulation
    - Lead time: 1-2 bars before visible price movement

    OBV reveals the balance of buying vs selling pressure over time.
    """

    @staticmethod
    def calculate(
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate On-Balance Volume.

        Algorithm:
        - If close > previous close: OBV += volume
        - If close < previous close: OBV -= volume
        - If close == previous close: OBV unchanged

        Args:
            close: Close prices
            volume: Trading volumes

        Returns:
            OBV series
        """
        # Calculate price direction
        price_change = close.diff()

        # Apply volume based on direction
        obv_change = volume.copy()
        obv_change[price_change < 0] = -volume[price_change < 0]
        obv_change[price_change == 0] = 0

        # Cumulative sum
        obv = obv_change.cumsum()

        return pd.Series(obv, index=close.index, name='obv')

    @staticmethod
    def detect_divergence(
        close: pd.Series,
        obv: pd.Series,
        window: int = 20,
        threshold_pct: float = 0.25
    ) -> pd.Series:
        """
        Detect OBV divergences from price.

        Per guide: OBV rising 25%+ while price flat = bullish signal

        Args:
            close: Close prices
            obv: OBV values
            window: Lookback window for comparison
            threshold_pct: OBV change threshold (default: 25% = 0.25)

        Returns:
            Boolean series indicating divergence
        """
        # Calculate percentage changes over window
        price_pct_change = close.pct_change(window)
        obv_pct_change = obv.pct_change(window)

        # Bullish divergence: OBV rising 25%+, price relatively flat (±5%)
        bullish_divergence = (
            (obv_pct_change > threshold_pct) &
            (abs(price_pct_change) < 0.05)
        )

        # Bearish divergence: OBV falling 25%+, price relatively flat
        bearish_divergence = (
            (obv_pct_change < -threshold_pct) &
            (abs(price_pct_change) < 0.05)
        )

        # Return 1 for bullish, -1 for bearish, 0 for no divergence
        divergence = pd.Series(0, index=close.index)
        divergence[bullish_divergence] = 1
        divergence[bearish_divergence] = -1

        return divergence


class VolumeRateOfChange:
    """
    Volume Rate of Change (VROC).

    Measures percentage change in volume over n periods.
    Acceleration in VROC often precedes price acceleration by 1-2 bars.
    """

    @staticmethod
    def calculate(
        volume: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Volume Rate of Change.

        Formula: ((Volume_t - Volume_t-n) / Volume_t-n) * 100

        Args:
            volume: Trading volumes
            period: Lookback period (default: 14)

        Returns:
            VROC percentage series
        """
        vroc = volume.pct_change(periods=period) * 100

        return pd.Series(vroc, index=volume.index, name='vroc')

    @staticmethod
    def detect_acceleration(
        vroc: pd.Series,
        threshold: float = 50.0
    ) -> pd.Series:
        """
        Detect volume acceleration.

        Args:
            vroc: VROC values
            threshold: Acceleration threshold (default: 50%)

        Returns:
            Boolean series indicating acceleration
        """
        return abs(vroc) > threshold


class VolumeSpikeDetector:
    """
    Volume Spike Detection.

    Per implementation guide:
    - Ratio: current_volume / rolling_avg(20) > 2.0
    - Indicates significant anomaly activity
    - Lead time: 1-2 bars before major price moves
    """

    def __init__(self, window: int = 20, threshold: float = 2.0):
        """
        Initialize volume spike detector.

        Args:
            window: Rolling average window (default: 20 per guide)
            threshold: Spike threshold multiplier (default: 2.0 per guide)
        """
        self.window = window
        self.threshold = threshold

    def calculate_ratio(
        self,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate volume spike ratio.

        Formula: current_volume / rolling_avg(window)

        Args:
            volume: Trading volumes

        Returns:
            Volume spike ratio series
        """
        rolling_avg = volume.rolling(window=self.window).mean()

        # Avoid division by zero
        rolling_avg = rolling_avg.replace(0, np.nan)

        ratio = volume / rolling_avg

        return pd.Series(ratio, index=volume.index, name='volume_spike_ratio')

    def detect_spikes(
        self,
        volume: pd.Series
    ) -> pd.Series:
        """
        Detect volume spikes exceeding threshold.

        Per guide: ratio > 2.0 indicates anomaly

        Args:
            volume: Trading volumes

        Returns:
            Boolean series indicating spikes
        """
        ratio = self.calculate_ratio(volume)
        spikes = ratio > self.threshold

        return spikes

    def calculate_spike_intensity(
        self,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate spike intensity score.

        Returns:
            Spike intensity (0 = normal, >1 = spike)
        """
        ratio = self.calculate_ratio(volume)

        # Normalize: 0 = at average, 1 = at threshold, >1 = above threshold
        intensity = (ratio - 1.0) / (self.threshold - 1.0)

        return intensity.clip(lower=0)


class VolumeProfile:
    """
    Volume Profile Analysis.

    Additional volume metrics:
    - Volume concentration
    - Relative volume
    - Volume trend
    """

    @staticmethod
    def relative_volume(
        volume: pd.Series,
        window: int = 50
    ) -> pd.Series:
        """
        Calculate relative volume vs historical average.

        Args:
            volume: Trading volumes
            window: Historical window (default: 50)

        Returns:
            Relative volume ratio
        """
        avg_volume = volume.rolling(window).mean()
        rel_volume = volume / avg_volume

        return rel_volume

    @staticmethod
    def volume_trend(
        volume: pd.Series,
        short_window: int = 10,
        long_window: int = 50
    ) -> pd.Series:
        """
        Calculate volume trend using moving average crossover.

        Args:
            volume: Trading volumes
            short_window: Short MA window
            long_window: Long MA window

        Returns:
            Trend score (positive = increasing, negative = decreasing)
        """
        short_ma = volume.rolling(short_window).mean()
        long_ma = volume.rolling(long_window).mean()

        trend = (short_ma - long_ma) / long_ma

        return trend

    @staticmethod
    def volume_percentile(
        volume: pd.Series,
        window: int = 100
    ) -> pd.Series:
        """
        Calculate volume percentile rank.

        Shows where current volume sits in historical distribution.

        Args:
            volume: Trading volumes
            window: Historical window

        Returns:
            Percentile rank (0-100)

        Note: Optimized from 34ms to <1ms by using raw=True and avoiding
        Series creation inside the lambda.
        """
        def pct_rank(x):
            # Count values less than or equal to current value
            # x[-1] is current, x[:-1] is history
            return (np.sum(x[:-1] <= x[-1]) / (len(x) - 1)) * 100

        # Use window+1 to have window historical values plus current
        percentile = volume.rolling(window + 1, min_periods=2).apply(pct_rank, raw=True)

        return percentile


def calculate_volume_features(
    close: pd.Series,
    volume: pd.Series,
    spike_window: int = 20,
    spike_threshold: float = 2.0,
    vroc_period: int = 14
) -> pd.DataFrame:
    """
    Calculate all volume-based features.

    Per implementation guide, volume patterns confirm price signals:
    - Volume spikes (>2x average) = 1-2 bar lead time
    - OBV divergences = early accumulation/distribution detection
    - VROC acceleration = momentum confirmation

    Args:
        close: Close prices
        volume: Trading volumes
        spike_window: Window for spike detection (default: 20)
        spike_threshold: Spike threshold (default: 2.0)
        vroc_period: VROC period (default: 14)

    Returns:
        DataFrame with all volume features
    """
    features = pd.DataFrame(index=close.index)

    # On-Balance Volume
    features['obv'] = OnBalanceVolume.calculate(close, volume)

    # OBV divergence detection
    features['obv_divergence'] = OnBalanceVolume.detect_divergence(
        close, features['obv'], window=20, threshold_pct=0.25
    )

    # Volume spike ratio (per guide: > 2.0 = anomaly)
    spike_detector = VolumeSpikeDetector(
        window=spike_window,
        threshold=spike_threshold
    )
    features['volume_spike_ratio'] = spike_detector.calculate_ratio(volume)
    features['volume_spike'] = spike_detector.detect_spikes(volume)
    features['volume_spike_intensity'] = spike_detector.calculate_spike_intensity(volume)

    # Volume Rate of Change
    features['vroc'] = VolumeRateOfChange.calculate(volume, period=vroc_period)
    features['vroc_acceleration'] = VolumeRateOfChange.detect_acceleration(
        features['vroc'], threshold=50.0
    )

    # Relative volume
    features['relative_volume'] = VolumeProfile.relative_volume(volume, window=50)

    # Volume trend
    features['volume_trend'] = VolumeProfile.volume_trend(
        volume, short_window=10, long_window=50
    )

    # Volume percentile
    features['volume_percentile'] = VolumeProfile.volume_percentile(volume, window=100)

    logger.debug(
        f"Calculated volume features",
        extra={
            "features": list(features.columns),
            "rows": len(features),
            "volume_spikes": features['volume_spike'].sum(),
            "obv_divergences": (features['obv_divergence'] != 0).sum()
        }
    )

    return features


if __name__ == "__main__":
    # Test volume indicators
    import numpy as np

    np.random.seed(42)

    # Generate synthetic data
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='5min')

    # Price with trend
    close = pd.Series(
        np.cumsum(np.random.randn(n) * 0.5) + 100,
        index=dates
    )

    # Volume with occasional spikes
    volume = pd.Series(
        np.random.exponential(1000, n),
        index=dates
    )

    # Add some volume spikes
    spike_indices = np.random.choice(range(50, n), size=10, replace=False)
    volume.iloc[spike_indices] *= 3  # 3x volume spikes

    print("=== Volume Indicators Test ===\n")

    # Test OBV
    obv = OnBalanceVolume.calculate(close, volume)
    print(f"OBV Range: {obv.min():.0f} to {obv.max():.0f}")
    print(f"OBV Trend: {'Bullish' if obv.iloc[-1] > obv.iloc[-50] else 'Bearish'}")

    # Test divergence detection
    divergence = OnBalanceVolume.detect_divergence(close, obv)
    bullish = (divergence == 1).sum()
    bearish = (divergence == -1).sum()
    print(f"\nOBV Divergences:")
    print(f"  Bullish: {bullish}")
    print(f"  Bearish: {bearish}")

    # Test volume spike detection
    spike_detector = VolumeSpikeDetector(window=20, threshold=2.0)
    spike_ratio = spike_detector.calculate_ratio(volume)
    spikes = spike_detector.detect_spikes(volume)

    print(f"\nVolume Spikes:")
    print(f"  Detected: {spikes.sum()} / {len(spikes)} bars")
    print(f"  Max Ratio: {spike_ratio.max():.2f}x average")
    print(f"  Threshold: 2.0x")

    # Test VROC
    vroc = VolumeRateOfChange.calculate(volume, period=14)
    print(f"\nVROC:")
    print(f"  Current: {vroc.iloc[-1]:.1f}%")
    print(f"  Mean: {vroc.mean():.1f}%")

    # Calculate all features
    features = calculate_volume_features(close, volume)

    print(f"\n=== All Volume Features Summary ===")
    print(features.describe())

    print(f"\n=== Recent Values (last 5 bars) ===")
    print(features.tail())

    # Show recent spikes
    recent_spikes = features[features['volume_spike']].tail()
    if len(recent_spikes) > 0:
        print(f"\n=== Recent Volume Spikes ===")
        print(recent_spikes[['volume_spike_ratio', 'volume_spike_intensity', 'vroc']])
