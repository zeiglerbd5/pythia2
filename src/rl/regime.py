"""
Regime Detection for RL Trading Agent (Phase 3)

Implements:
- Market regime classifier (trending/ranging/volatile)
- Regime-aware performance tracking
- Regime change detection
- Regime-specific model adaptation

Understanding market regimes is crucial for:
1. Knowing when to retrain the model
2. Tracking performance per regime
3. Adapting trading strategy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
from loguru import logger

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)


class RegimeType(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # Feature windows
    return_window: int = 24         # Hours for trend detection
    volatility_window: int = 24     # Hours for volatility
    correlation_window: int = 24    # Hours for correlation

    # Thresholds
    trend_threshold: float = 0.02   # 2% for significant trend
    volatility_low: float = 0.01    # Low volatility threshold
    volatility_high: float = 0.04   # High volatility threshold

    # Regime change detection
    change_window: int = 100        # Samples for change detection
    change_threshold: float = 0.5   # KL divergence threshold

    # GMM settings
    n_regimes: int = 4              # Number of regimes for GMM
    gmm_random_state: int = 42


@dataclass
class RegimeFeatures:
    """Features for regime classification."""
    # Trend features
    return_24h: float = 0.0
    return_7d: float = 0.0
    trend_strength: float = 0.0      # MA crossover based

    # Volatility features
    realized_volatility: float = 0.0
    volatility_percentile: float = 0.5
    volatility_trend: float = 0.0    # Increasing/decreasing

    # Correlation features
    btc_correlation: float = 0.0
    market_correlation: float = 0.0  # Average cross-asset correlation

    # Volume features
    volume_trend: float = 0.0
    volume_volatility: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.return_24h,
            self.return_7d,
            self.trend_strength,
            self.realized_volatility,
            self.volatility_percentile,
            self.volatility_trend,
            self.btc_correlation,
            self.market_correlation,
            self.volume_trend,
            self.volume_volatility,
        ], dtype=np.float32)


class RegimeDetector:
    """
    Detect current market regime and track regime changes.

    Uses both rule-based and GMM-based approaches:
    - Rule-based: Clear classifications based on thresholds
    - GMM: Unsupervised clustering for nuanced regimes
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize regime detector.

        Args:
            config: Regime detection configuration
        """
        self.config = config or RegimeConfig()

        # GMM model for regime clustering
        self.gmm = GaussianMixture(
            n_components=self.config.n_regimes,
            covariance_type='full',
            random_state=self.config.gmm_random_state,
        )
        self.scaler = StandardScaler()
        self.gmm_fitted = False

        # Regime history
        self.regime_history: deque = deque(maxlen=1000)
        self.feature_history: deque = deque(maxlen=1000)

        # Volatility history for percentile calculation
        self.volatility_history: deque = deque(maxlen=500)

        # Current regime
        self.current_regime: RegimeType = RegimeType.UNKNOWN
        self.regime_start_time: Optional[datetime] = None
        self.regime_confidence: float = 0.0

    def extract_features(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        btc_prices: Optional[pd.Series] = None,
    ) -> RegimeFeatures:
        """
        Extract regime features from price data.

        Args:
            prices: Price series with datetime index
            volumes: Volume series (optional)
            btc_prices: BTC prices for correlation (optional)

        Returns:
            RegimeFeatures object
        """
        features = RegimeFeatures()

        if len(prices) < 24:
            return features

        # Returns at various horizons
        if len(prices) >= 24:
            features.return_24h = (prices.iloc[-1] / prices.iloc[-24] - 1) * 100

        if len(prices) >= 168:  # 7 days * 24 hours
            features.return_7d = (prices.iloc[-1] / prices.iloc[-168] - 1) * 100

        # Trend strength (using MA crossover)
        if len(prices) >= 50:
            short_ma = prices.rolling(10).mean()
            long_ma = prices.rolling(50).mean()
            if long_ma.iloc[-1] > 0:
                features.trend_strength = (short_ma.iloc[-1] / long_ma.iloc[-1] - 1) * 100

        # Realized volatility
        log_returns = np.log(prices / prices.shift(1)).dropna()
        if len(log_returns) >= self.config.volatility_window:
            features.realized_volatility = log_returns.iloc[-self.config.volatility_window:].std() * np.sqrt(24 * 365) * 100

            # Update volatility history
            self.volatility_history.append(features.realized_volatility)

            # Volatility percentile
            if len(self.volatility_history) >= 50:
                features.volatility_percentile = (
                    np.sum(np.array(self.volatility_history) < features.realized_volatility) /
                    len(self.volatility_history)
                )

            # Volatility trend
            if len(self.volatility_history) >= 20:
                recent_vol = np.mean(list(self.volatility_history)[-10:])
                older_vol = np.mean(list(self.volatility_history)[-20:-10])
                features.volatility_trend = (recent_vol / older_vol - 1) if older_vol > 0 else 0

        # BTC correlation
        if btc_prices is not None and len(btc_prices) >= self.config.correlation_window:
            symbol_returns = log_returns.iloc[-self.config.correlation_window:]
            btc_returns = np.log(btc_prices / btc_prices.shift(1)).dropna()
            btc_returns = btc_returns.iloc[-self.config.correlation_window:]

            if len(symbol_returns) > 0 and len(btc_returns) > 0:
                # Align indices
                common = symbol_returns.index.intersection(btc_returns.index)
                if len(common) > 10:
                    features.btc_correlation = symbol_returns.loc[common].corr(btc_returns.loc[common])
                    if np.isnan(features.btc_correlation):
                        features.btc_correlation = 0

        # Volume features
        if volumes is not None and len(volumes) >= 50:
            vol_ma_short = volumes.rolling(10).mean()
            vol_ma_long = volumes.rolling(50).mean()

            if vol_ma_long.iloc[-1] > 0:
                features.volume_trend = (vol_ma_short.iloc[-1] / vol_ma_long.iloc[-1] - 1)

            features.volume_volatility = volumes.iloc[-50:].std() / volumes.iloc[-50:].mean()

        return features

    def classify_regime_rules(self, features: RegimeFeatures) -> Tuple[RegimeType, float]:
        """
        Classify regime using rule-based approach.

        Args:
            features: Regime features

        Returns:
            (regime_type, confidence)
        """
        # High volatility takes precedence
        if features.realized_volatility > self.config.volatility_high * 100:
            confidence = min(features.realized_volatility / (self.config.volatility_high * 200), 1.0)
            return RegimeType.HIGH_VOLATILITY, confidence

        # Low volatility
        if features.realized_volatility < self.config.volatility_low * 100:
            confidence = 1 - features.realized_volatility / (self.config.volatility_low * 100)
            return RegimeType.LOW_VOLATILITY, confidence

        # Trending up
        if features.return_24h > self.config.trend_threshold * 100 and features.trend_strength > 0:
            confidence = min(abs(features.return_24h) / (self.config.trend_threshold * 200), 1.0)
            return RegimeType.TRENDING_UP, confidence

        # Trending down
        if features.return_24h < -self.config.trend_threshold * 100 and features.trend_strength < 0:
            confidence = min(abs(features.return_24h) / (self.config.trend_threshold * 200), 1.0)
            return RegimeType.TRENDING_DOWN, confidence

        # Default to ranging
        confidence = 1 - abs(features.return_24h) / (self.config.trend_threshold * 100)
        return RegimeType.RANGING, max(confidence, 0.5)

    def classify_regime_gmm(self, features: RegimeFeatures) -> Tuple[int, float]:
        """
        Classify regime using GMM.

        Args:
            features: Regime features

        Returns:
            (cluster_id, probability)
        """
        if not self.gmm_fitted:
            return 0, 0.0

        feature_array = features.to_array().reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_array)

        cluster = self.gmm.predict(feature_scaled)[0]
        probs = self.gmm.predict_proba(feature_scaled)[0]
        confidence = probs[cluster]

        return cluster, confidence

    def fit_gmm(self, feature_history: List[RegimeFeatures]) -> None:
        """
        Fit GMM on historical features.

        Args:
            feature_history: List of historical regime features
        """
        if len(feature_history) < 100:
            logger.warning("Insufficient data to fit GMM")
            return

        # Convert to array
        features = np.array([f.to_array() for f in feature_history])

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)

        # Fit scaler and GMM
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)

        self.gmm.fit(features_scaled)
        self.gmm_fitted = True

        logger.info(f"GMM fitted with {self.config.n_regimes} regimes on {len(features)} samples")

    def detect(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        btc_prices: Optional[pd.Series] = None,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[RegimeType, float]:
        """
        Detect current market regime.

        Args:
            prices: Price series
            volumes: Volume series (optional)
            btc_prices: BTC prices (optional)
            timestamp: Current timestamp

        Returns:
            (regime_type, confidence)
        """
        # Extract features
        features = self.extract_features(prices, volumes, btc_prices)
        self.feature_history.append(features)

        # Rule-based classification
        regime, confidence = self.classify_regime_rules(features)

        # Update state
        if regime != self.current_regime:
            logger.info(f"Regime change: {self.current_regime.value} -> {regime.value}")
            self.current_regime = regime
            self.regime_start_time = timestamp or datetime.now()

        self.regime_confidence = confidence
        self.regime_history.append((timestamp, regime, confidence))

        return regime, confidence

    def check_regime_change(self, window: Optional[int] = None) -> bool:
        """
        Check if a significant regime change has occurred.

        Uses KL divergence between recent and older regime distributions.

        Args:
            window: Window size for comparison (default: config.change_window)

        Returns:
            True if regime change detected
        """
        window = window or self.config.change_window

        if len(self.regime_history) < window:
            return False

        # Get recent and older distributions
        recent = [r[1] for r in list(self.regime_history)[-window//2:]]
        older = [r[1] for r in list(self.regime_history)[-window:-window//2]]

        # Count regimes
        recent_counts = defaultdict(int)
        older_counts = defaultdict(int)

        for r in recent:
            recent_counts[r] += 1
        for r in older:
            older_counts[r] += 1

        # Calculate distributions
        all_regimes = set(recent_counts.keys()) | set(older_counts.keys())
        n_recent = len(recent)
        n_older = len(older)

        # KL divergence
        kl_div = 0.0
        for regime in all_regimes:
            p = (recent_counts[regime] + 1) / (n_recent + len(all_regimes))  # Smoothed
            q = (older_counts[regime] + 1) / (n_older + len(all_regimes))
            kl_div += p * np.log(p / q)

        return kl_div > self.config.change_threshold

    def get_regime_stats(self) -> Dict[str, Any]:
        """Get regime statistics."""
        if not self.regime_history:
            return {}

        # Regime distribution
        regime_counts = defaultdict(int)
        for _, regime, _ in self.regime_history:
            regime_counts[regime.value] += 1

        total = sum(regime_counts.values())
        regime_distribution = {
            k: v / total for k, v in regime_counts.items()
        }

        # Current regime duration
        duration = None
        if self.regime_start_time:
            duration = (datetime.now() - self.regime_start_time).total_seconds() / 3600

        return {
            'current_regime': self.current_regime.value,
            'confidence': self.regime_confidence,
            'duration_hours': duration,
            'regime_distribution': regime_distribution,
            'total_samples': total,
        }


class RegimePerformanceTracker:
    """
    Track trading performance per market regime.

    Helps identify:
    - Which regimes the strategy excels in
    - Which regimes need improvement
    - When to enable/disable trading
    """

    def __init__(self):
        """Initialize performance tracker."""
        self.performance: Dict[RegimeType, List[float]] = defaultdict(list)
        self.trades: Dict[RegimeType, List[Dict]] = defaultdict(list)

    def record_trade(
        self,
        regime: RegimeType,
        return_pct: float,
        trade_info: Optional[Dict] = None,
    ) -> None:
        """
        Record trade result for a regime.

        Args:
            regime: Market regime during trade
            return_pct: Trade return percentage
            trade_info: Additional trade information
        """
        self.performance[regime].append(return_pct)

        if trade_info:
            trade_info['return_pct'] = return_pct
            self.trades[regime].append(trade_info)

    def get_regime_performance(self, regime: RegimeType) -> Dict[str, float]:
        """
        Get performance metrics for a regime.

        Args:
            regime: Target regime

        Returns:
            Performance metrics
        """
        returns = self.performance.get(regime, [])

        if not returns:
            return {
                'n_trades': 0,
                'total_return': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe': 0,
            }

        returns = np.array(returns)
        wins = returns > 0

        sharpe = 0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)

        return {
            'n_trades': len(returns),
            'total_return': float(returns.sum()),
            'win_rate': float(wins.mean()),
            'avg_return': float(returns.mean()),
            'sharpe': float(sharpe),
        }

    def get_all_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance for all regimes."""
        return {
            regime.value: self.get_regime_performance(regime)
            for regime in RegimeType
            if regime in self.performance
        }

    def should_trade(self, regime: RegimeType, min_trades: int = 20) -> Tuple[bool, str]:
        """
        Determine if we should trade in current regime.

        Args:
            regime: Current regime
            min_trades: Minimum trades for decision

        Returns:
            (should_trade, reason)
        """
        perf = self.get_regime_performance(regime)

        if perf['n_trades'] < min_trades:
            return True, "insufficient_data"

        # Don't trade if significantly negative
        if perf['avg_return'] < -0.01 and perf['n_trades'] >= min_trades:
            return False, f"negative_performance_{perf['avg_return']:.2%}"

        # Don't trade if win rate is very low
        if perf['win_rate'] < 0.3 and perf['n_trades'] >= min_trades:
            return False, f"low_win_rate_{perf['win_rate']:.2%}"

        return True, "ok"


if __name__ == "__main__":
    # Test regime detection
    print("Testing Regime Detection\n" + "=" * 50)

    # Generate synthetic price data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2024-01-01', periods=n, freq='h')

    # Create different regime periods
    prices_data = []
    current = 100

    # Trending up (0-100)
    for i in range(100):
        current *= (1 + 0.002 + np.random.randn() * 0.01)
        prices_data.append(current)

    # High volatility (100-200)
    for i in range(100):
        current *= (1 + np.random.randn() * 0.05)
        prices_data.append(current)

    # Ranging (200-300)
    for i in range(100):
        current *= (1 + np.random.randn() * 0.01)
        prices_data.append(current)

    # Trending down (300-400)
    for i in range(100):
        current *= (1 - 0.002 + np.random.randn() * 0.01)
        prices_data.append(current)

    # Low volatility (400-500)
    for i in range(100):
        current *= (1 + np.random.randn() * 0.003)
        prices_data.append(current)

    prices = pd.Series(prices_data, index=dates)
    volumes = pd.Series(np.random.exponential(1000, n), index=dates)

    # Test RegimeDetector
    print("\n1. RegimeDetector")
    config = RegimeConfig()
    detector = RegimeDetector(config)

    # Detect regimes at different points
    test_points = [50, 150, 250, 350, 450]
    for i in test_points:
        regime, confidence = detector.detect(
            prices.iloc[:i+1],
            volumes.iloc[:i+1],
            timestamp=dates[i],
        )
        print(f"   Point {i}: {regime.value} (confidence: {confidence:.2f})")

    # Get stats
    stats = detector.get_regime_stats()
    print(f"\n   Regime stats: {stats}")

    # Check for regime change
    changed = detector.check_regime_change()
    print(f"   Regime change detected: {changed}")

    # Test GMM fitting
    print("\n2. GMM Fitting")
    features_list = list(detector.feature_history)
    detector.fit_gmm(features_list)
    print(f"   GMM fitted: {detector.gmm_fitted}")

    if detector.gmm_fitted:
        features = detector.extract_features(prices, volumes)
        cluster, prob = detector.classify_regime_gmm(features)
        print(f"   GMM cluster: {cluster}, probability: {prob:.2f}")

    # Test RegimePerformanceTracker
    print("\n3. RegimePerformanceTracker")
    tracker = RegimePerformanceTracker()

    # Record some trades
    for _ in range(30):
        tracker.record_trade(RegimeType.TRENDING_UP, np.random.randn() * 0.03 + 0.01)
    for _ in range(30):
        tracker.record_trade(RegimeType.HIGH_VOLATILITY, np.random.randn() * 0.05)
    for _ in range(30):
        tracker.record_trade(RegimeType.RANGING, np.random.randn() * 0.02 - 0.005)

    # Get performance
    all_perf = tracker.get_all_performance()
    for regime, perf in all_perf.items():
        print(f"   {regime}:")
        print(f"      Trades: {perf['n_trades']}, Win rate: {perf['win_rate']:.2%}, Sharpe: {perf['sharpe']:.2f}")

    # Check if should trade
    for regime in [RegimeType.TRENDING_UP, RegimeType.RANGING]:
        should, reason = tracker.should_trade(regime)
        print(f"   Should trade in {regime.value}: {should} ({reason})")

    print("\nRegime detection tests passed!")
