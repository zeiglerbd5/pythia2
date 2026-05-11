"""
Pythia Feature Engineering Module

Phase 2 implementation providing real-time feature calculation.

Per implementation guide:
- 5-minute primary timeframe (optimal for scalping)
- 50-100 bar lookback windows
- ~30-40 features after Boruta selection from 100+ candidates

Modules:
- microstructure: Roll measure (top predictor), VPIN, order flow metrics
- volume_indicators: OBV, VROC, volume spike detection
- price_indicators: RSI, VWAP, ATR, Bollinger Bands
- ohlcv_aggregator: Multi-timeframe candle generation
- feature_engine: Real-time orchestration and rolling window management
- boruta_selector: Feature selection (100+ -> 30-40 features)
"""

from .microstructure import (
    RollMeasure,
    VPIN,
    OrderFlowImbalance,
    calculate_microstructure_features
)

from .volume_indicators import (
    OnBalanceVolume,
    VolumeRateOfChange,
    VolumeSpikeDetector,
    VolumeProfile,
    calculate_volume_features
)

from .price_indicators import (
    RSI,
    VWAP,
    ATR,
    BollingerBands,
    calculate_price_features
)

from .ohlcv_aggregator import (
    OHLCVCandle,
    OHLCVAggregator,
    candles_to_dataframe
)

from .feature_engine import FeatureEngine

from .boruta_selector import BorutaSelector

__all__ = [
    # Microstructure
    'RollMeasure',
    'VPIN',
    'OrderFlowImbalance',
    'calculate_microstructure_features',

    # Volume
    'OnBalanceVolume',
    'VolumeRateOfChange',
    'VolumeSpikeDetector',
    'VolumeProfile',
    'calculate_volume_features',

    # Price
    'RSI',
    'VWAP',
    'ATR',
    'BollingerBands',
    'calculate_price_features',

    # OHLCV
    'OHLCVCandle',
    'OHLCVAggregator',
    'candles_to_dataframe',

    # Engine
    'FeatureEngine',

    # Selection
    'BorutaSelector',
]
