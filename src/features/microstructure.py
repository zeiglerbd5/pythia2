"""
Market Microstructure Indicators

Implements advanced order book and trade flow metrics per implementation guide:
- Roll Measure: Top predictor (MDA 0.044-0.058)
- VPIN: Volume-Synchronized Probability of Informed Trading
- Order book metrics (imbalance already in order_book.py)

These indicators provide 5-30 minute early warning before major price movements.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from loguru import logger


class RollMeasure:
    """
    Roll Measure: Captures bid-ask bounce and price autocorrelation.

    Per implementation guide:
    - Formula: 2 * sqrt(|cov(ΔP_t, ΔP_t-1)|)
    - Top predictor with MDA scores of 0.044-0.058
    - Lead time: 15-30 minutes before volatility spikes
    - Window: 50-100 bars optimal for crypto

    This metric captures momentum before it becomes visible in price action.
    """

    def __init__(self, window: int = 100):
        """
        Initialize Roll Measure calculator.

        Args:
            window: Lookback window for covariance calculation (50-100 per guide)
        """
        self.window = window

    def calculate(self, prices: Union[pd.Series, np.ndarray]) -> Optional[float]:
        """
        Calculate Roll Measure from price series.

        Formula: 2 * sqrt(|cov(ΔP_t, ΔP_t-1)|)

        Args:
            prices: Time series of prices

        Returns:
            Roll measure value or None if insufficient data
        """
        if len(prices) < self.window + 2:
            return None

        # Convert to numpy array if pandas Series
        if isinstance(prices, pd.Series):
            prices = prices.values

        # Take last window prices
        recent_prices = prices[-self.window:]

        # Calculate price changes (first differences)
        price_changes = np.diff(recent_prices)

        if len(price_changes) < 2:
            return None

        # Calculate covariance between ΔP_t and ΔP_t-1
        # Create lagged series
        delta_p_t = price_changes[1:]      # ΔP_t
        delta_p_t_minus_1 = price_changes[:-1]  # ΔP_t-1

        # Compute covariance
        covariance = np.cov(delta_p_t, delta_p_t_minus_1)[0, 1]

        # Roll measure: 2 * sqrt(|cov|)
        # Take absolute value before sqrt to handle negative covariance
        roll = 2.0 * np.sqrt(abs(covariance))

        return float(roll)

    def calculate_series(self, prices: pd.Series) -> pd.Series:
        """
        Calculate rolling Roll Measure for entire price series.

        Args:
            prices: Price series with datetime index

        Returns:
            Series of Roll measure values

        Note: Optimized from O(n²) to O(n) using pandas rolling covariance.
        """
        # Calculate price changes
        price_changes = prices.diff()

        # Rolling covariance between ΔP_t and ΔP_t-1
        delta_p_t = price_changes
        delta_p_t_minus_1 = price_changes.shift(1)

        # Use pandas rolling covariance - O(n) instead of O(n²)
        rolling_cov = delta_p_t.rolling(window=self.window).cov(delta_p_t_minus_1)

        # Roll measure: 2 * sqrt(|cov|)
        roll = 2.0 * np.sqrt(np.abs(rolling_cov))

        return roll.rename('roll_measure')


class VPIN:
    """
    Volume-Synchronized Probability of Informed Trading.

    Per implementation guide:
    - Formula: |V_sell - V_buy| / V_total over rolling window
    - Threshold: > 0.5 indicates high informed trading activity
    - Crypto average: 0.45-0.47 vs 0.22-0.23 in equities
    - Lead time: 5-15 minutes before spikes

    VPIN measures order flow toxicity - high values indicate informed traders.
    """

    def __init__(self, window: int = 50):
        """
        Initialize VPIN calculator.

        Args:
            window: Rolling window for volume calculations
        """
        self.window = window

    def calculate(
        self,
        buy_volume: Union[pd.Series, np.ndarray],
        sell_volume: Union[pd.Series, np.ndarray]
    ) -> Optional[float]:
        """
        Calculate VPIN from buy and sell volumes.

        Formula: |V_sell - V_buy| / V_total

        Args:
            buy_volume: Time series of buy volumes
            sell_volume: Time series of sell volumes

        Returns:
            VPIN value or None if insufficient data
        """
        if len(buy_volume) < self.window or len(sell_volume) < self.window:
            return None

        # Convert to numpy if needed
        if isinstance(buy_volume, pd.Series):
            buy_volume = buy_volume.values
        if isinstance(sell_volume, pd.Series):
            sell_volume = sell_volume.values

        # Take last window values
        recent_buy = buy_volume[-self.window:]
        recent_sell = sell_volume[-self.window:]

        # Sum volumes
        total_buy = np.sum(recent_buy)
        total_sell = np.sum(recent_sell)
        total_volume = total_buy + total_sell

        if total_volume == 0:
            return 0.0

        # VPIN = |V_sell - V_buy| / V_total
        vpin = abs(total_sell - total_buy) / total_volume

        return float(vpin)

    def calculate_series(
        self,
        buy_volume: pd.Series,
        sell_volume: pd.Series
    ) -> pd.Series:
        """
        Calculate rolling VPIN for entire volume series.

        Args:
            buy_volume: Buy volume series
            sell_volume: Sell volume series

        Returns:
            Series of VPIN values

        Note: Optimized from O(n²) to O(n) using pandas rolling sums.
        """
        # Use rolling sums - O(n) instead of O(n²)
        rolling_buy = buy_volume.rolling(window=self.window).sum()
        rolling_sell = sell_volume.rolling(window=self.window).sum()
        total_volume = rolling_buy + rolling_sell

        # VPIN = |V_sell - V_buy| / V_total
        vpin = np.abs(rolling_sell - rolling_buy) / total_volume
        vpin = vpin.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return vpin.rename('vpin')


class OrderFlowImbalance:
    """
    Order flow imbalance metrics beyond basic order book imbalance.

    Additional microstructure signals:
    - Trade flow imbalance (buy vs sell initiated trades)
    - Volume-weighted imbalance
    - Persistent imbalance detection
    """

    @staticmethod
    def trade_flow_imbalance(
        buy_volume: pd.Series,
        sell_volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate rolling trade flow imbalance.

        Formula: (Buy_volume - Sell_volume) / (Buy_volume + Sell_volume)

        Args:
            buy_volume: Buy-initiated trade volume
            sell_volume: Sell-initiated trade volume
            window: Rolling window size

        Returns:
            Series of imbalance values [-1, 1]
        """
        buy_sum = buy_volume.rolling(window).sum()
        sell_sum = sell_volume.rolling(window).sum()

        total = buy_sum + sell_sum
        total = total.replace(0, np.nan)  # Avoid division by zero

        imbalance = (buy_sum - sell_sum) / total

        return imbalance

    @staticmethod
    def volume_weighted_imbalance(
        prices: pd.Series,
        volumes: pd.Series,
        sides: pd.Series  # 'BUY' or 'SELL'
    ) -> float:
        """
        Calculate volume-weighted price imbalance.

        Weights price movements by trade volume and direction.

        Args:
            prices: Trade prices
            volumes: Trade volumes
            sides: Trade sides ('BUY' or 'SELL')

        Returns:
            Volume-weighted imbalance
        """
        # Convert sides to numeric: BUY = 1, SELL = -1
        side_numeric = sides.map({'BUY': 1, 'SELL': -1}).fillna(0)

        # Calculate weighted sum
        weighted_sum = (prices * volumes * side_numeric).sum()
        total_volume = volumes.sum()

        if total_volume == 0:
            return 0.0

        return weighted_sum / total_volume

    @staticmethod
    def persistent_imbalance(
        imbalance: pd.Series,
        threshold: float = 0.3,
        min_periods: int = 5
    ) -> pd.Series:
        """
        Detect persistent directional imbalance.

        Per implementation guide: |imbalance| > 0.3 signals directional pressure

        Args:
            imbalance: Order book or trade flow imbalance series
            threshold: Absolute threshold for significant imbalance
            min_periods: Minimum consecutive periods for persistence

        Returns:
            Boolean series indicating persistent imbalance
        """
        # Check if imbalance exceeds threshold
        significant = abs(imbalance) > threshold

        # Count consecutive True values
        consecutive = significant.rolling(min_periods).sum()

        # Persistent if all periods in window are significant
        persistent = consecutive >= min_periods

        return persistent


def calculate_microstructure_features(
    prices: pd.Series,
    buy_volume: pd.Series,
    sell_volume: pd.Series,
    order_book_imbalance: Optional[pd.Series] = None,
    roll_window: int = 100,
    vpin_window: int = 50
) -> pd.DataFrame:
    """
    Calculate all microstructure features for a symbol.

    Per implementation guide, these are the top predictors:
    - Roll Measure: MDA 0.044-0.058 (highest importance)
    - VPIN: Early warning indicator
    - Order book imbalance: 10s-2min lead time

    Args:
        prices: Close prices
        buy_volume: Buy-initiated volume
        sell_volume: Sell-initiated volume
        order_book_imbalance: Order book imbalance from L2 data (optional)
        roll_window: Window for Roll measure (default: 100)
        vpin_window: Window for VPIN (default: 50)

    Returns:
        DataFrame with all microstructure features
    """
    features = pd.DataFrame(index=prices.index)

    # Roll Measure (top predictor per guide)
    roll_calc = RollMeasure(window=roll_window)
    features['roll_measure'] = roll_calc.calculate_series(prices)

    # VPIN (order flow toxicity)
    vpin_calc = VPIN(window=vpin_window)
    features['vpin'] = vpin_calc.calculate_series(buy_volume, sell_volume)

    # Trade flow imbalance
    features['trade_flow_imbalance'] = OrderFlowImbalance.trade_flow_imbalance(
        buy_volume, sell_volume, window=20
    )

    # Order book imbalance (if provided from L2 data)
    if order_book_imbalance is not None:
        features['order_book_imbalance_l5'] = order_book_imbalance

        # Detect persistent imbalance (|ρ| > 0.3 per guide)
        features['persistent_imbalance'] = OrderFlowImbalance.persistent_imbalance(
            order_book_imbalance,
            threshold=0.3,
            min_periods=5
        )

    # VPIN threshold indicator (> 0.5 per guide)
    features['vpin_high'] = features['vpin'] > 0.5

    logger.debug(
        f"Calculated microstructure features",
        extra={
            "features": list(features.columns),
            "rows": len(features),
            "roll_mean": features['roll_measure'].mean(),
            "vpin_mean": features['vpin'].mean()
        }
    )

    return features


if __name__ == "__main__":
    # Test microstructure indicators
    import numpy as np

    np.random.seed(42)

    # Generate synthetic price data with autocorrelation
    n = 200
    prices = pd.Series(
        np.cumsum(np.random.randn(n) * 0.01) + 100,
        index=pd.date_range('2024-01-01', periods=n, freq='5min')
    )

    # Generate synthetic volume data
    buy_volume = pd.Series(
        np.random.exponential(100, n),
        index=prices.index
    )
    sell_volume = pd.Series(
        np.random.exponential(100, n),
        index=prices.index
    )

    # Calculate Roll Measure
    roll = RollMeasure(window=100)
    roll_value = roll.calculate(prices)
    print(f"\nRoll Measure: {roll_value:.6f}")
    print(f"  (Higher values indicate stronger momentum/autocorrelation)")

    # Calculate VPIN
    vpin = VPIN(window=50)
    vpin_value = vpin.calculate(buy_volume, sell_volume)
    print(f"\nVPIN: {vpin_value:.4f}")
    print(f"  Threshold: 0.5 (high informed trading)")
    print(f"  Status: {'⚠️  HIGH' if vpin_value > 0.5 else '✓ Normal'}")

    # Calculate all features
    order_book_imb = pd.Series(
        np.random.randn(n) * 0.2,  # Random imbalance ±0.2
        index=prices.index
    )

    features = calculate_microstructure_features(
        prices=prices,
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        order_book_imbalance=order_book_imb
    )

    print(f"\n=== Microstructure Features Summary ===")
    print(features.describe())

    print(f"\n=== Recent Values (last 5 bars) ===")
    print(features.tail())
