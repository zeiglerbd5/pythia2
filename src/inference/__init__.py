"""
Real-time inference infrastructure for pre-spike detection.

Components:
- CandleAggregator: Aggregate WebSocket data into 1-minute candles
- FeatureCalculator: Calculate 24 features in real-time
- SequenceBuffer: Maintain 60-candle rolling windows
- InferenceEngine: Load model and run predictions
- AlertManager: Handle notifications for high-probability signals
"""

from .candle_aggregator import CandleAggregator
from .feature_calculator import FeatureCalculator
from .sequence_buffer import SequenceBuffer
from .inference_engine import InferenceEngine
from .alerting import AlertManager
from .public_websocket import PublicWebSocketClient

__all__ = [
    'CandleAggregator',
    'FeatureCalculator',
    'SequenceBuffer',
    'InferenceEngine',
    'AlertManager',
    'PublicWebSocketClient',
]
