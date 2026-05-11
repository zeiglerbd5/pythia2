"""
Sequence Buffer for Real-time Inference

Maintains rolling 60-candle feature sequences for each symbol.
Normalizes features using saved scaler and prepares tensors for model inference.
"""

import numpy as np
import torch
import pickle
from typing import Dict, Optional, Deque
from collections import deque
from pathlib import Path
from loguru import logger


class SymbolSequenceBuffer:
    """Maintains feature sequence buffer for a single symbol."""

    def __init__(self, symbol: str, sequence_length: int, n_features: int):
        """
        Initialize symbol sequence buffer.

        Args:
            symbol: Trading pair symbol
            sequence_length: Number of timesteps (e.g., 60 candles)
            n_features: Number of features per timestep (24)
        """
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.n_features = n_features

        # Rolling window of feature vectors
        self.features_buffer = deque(maxlen=sequence_length)

        # Track timestamps for debugging
        self.timestamps_buffer = deque(maxlen=sequence_length)

    def append(self, features: Dict, timestamp):
        """
        Add new feature vector to buffer.

        Args:
            features: Dict of feature values
            timestamp: Candle timestamp
        """
        # Convert features dict to ordered array (must match training order)
        feature_vector = self._dict_to_array(features)

        self.features_buffer.append(feature_vector)
        self.timestamps_buffer.append(timestamp)

    def is_ready(self) -> bool:
        """Check if buffer has enough data for inference."""
        return len(self.features_buffer) == self.sequence_length

    def get_sequence(self) -> np.ndarray:
        """
        Get current sequence as numpy array.

        Returns:
            Array of shape (sequence_length, n_features)
        """
        if not self.is_ready():
            return None

        return np.array(self.features_buffer)

    def get_latest_timestamp(self):
        """Get timestamp of most recent candle."""
        if self.timestamps_buffer:
            return self.timestamps_buffer[-1]
        return None

    def _dict_to_array(self, features: Dict) -> np.ndarray:
        """Convert features dict to ordered array."""
        # Must match training feature order exactly
        feature_order = [
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

        # Extract values in correct order
        values = []
        for feature_name in feature_order:
            value = features.get(feature_name, 0.0)
            # Handle NaN/Inf
            if not np.isfinite(value):
                value = 0.0
            values.append(value)

        return np.array(values, dtype=np.float32)


class SequenceBuffer:
    """
    Manages sequence buffers for all symbols.

    Features:
    - Maintains rolling windows of feature vectors
    - Normalizes using saved scaler
    - Prepares PyTorch tensors for inference
    """

    def __init__(
        self,
        symbols: list[str],
        sequence_length: int = 60,
        n_features: int = 24,
        scaler_path: Optional[str] = None,
        device: str = 'mps'
    ):
        """
        Initialize sequence buffer.

        Args:
            symbols: List of symbols to track
            sequence_length: Number of timesteps (candles)
            n_features: Number of features per timestep
            scaler_path: Path to saved RobustScaler pickle file
            device: PyTorch device ('mps', 'cuda', or 'cpu')
        """
        self.symbols = symbols
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.device = device

        # Per-symbol sequence buffers
        self.buffers: Dict[str, SymbolSequenceBuffer] = {}

        for symbol in symbols:
            self.buffers[symbol] = SymbolSequenceBuffer(
                symbol=symbol,
                sequence_length=sequence_length,
                n_features=n_features
            )

        # Load scaler if provided
        self.scaler = None
        if scaler_path and Path(scaler_path).exists():
            self.load_scaler(scaler_path)
        else:
            logger.warning(f"No scaler loaded - features will not be normalized")

        logger.info(
            f"SequenceBuffer initialized: {len(symbols)} symbols, "
            f"sequence_length={sequence_length}, n_features={n_features}"
        )

    def load_scaler(self, scaler_path: str):
        """Load saved RobustScaler from pickle file."""
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            logger.info(f"Scaler loaded from {scaler_path}")

        except Exception as e:
            logger.error(f"Failed to load scaler from {scaler_path}: {e}")
            self.scaler = None

    def add_features(self, symbol: str, features: Dict, timestamp):
        """
        Add feature vector to symbol's buffer.

        Args:
            symbol: Trading pair symbol
            features: Dict of feature values
            timestamp: Candle timestamp
        """
        if symbol not in self.buffers:
            logger.warning(f"Unknown symbol: {symbol}")
            return

        buffer = self.buffers[symbol]
        buffer.append(features, timestamp)

        # DEBUG: Log first few features for every 100th addition
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 100 == 0:
            feature_sample = {k: features[k] for k in list(features.keys())[:5]}
            logger.debug(f"DEBUG feature sample for {symbol}: {feature_sample}")

    def is_ready(self, symbol: str) -> bool:
        """Check if symbol has enough data for inference."""
        if symbol not in self.buffers:
            return False

        return self.buffers[symbol].is_ready()

    def get_sequence_tensor(self, symbol: str) -> Optional[torch.Tensor]:
        """
        Get normalized sequence as PyTorch tensor ready for inference.

        Args:
            symbol: Trading pair symbol

        Returns:
            Tensor of shape (1, sequence_length, n_features) or None if not ready
        """
        if not self.is_ready(symbol):
            return None

        buffer = self.buffers[symbol]
        sequence = buffer.get_sequence()  # Shape: (sequence_length, n_features)

        # Normalize if scaler available
        if self.scaler:
            # Flatten for scaler (expects 2D: samples x features)
            original_shape = sequence.shape
            sequence_flat = sequence.reshape(-1, self.n_features)

            # Transform
            sequence_normalized = self.scaler.transform(sequence_flat)

            # Reshape back
            sequence = sequence_normalized.reshape(original_shape)

        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Shape: (1, seq_len, n_features)

        # DEBUG: Log tensor statistics every 100th call
        if not hasattr(self, '_tensor_counter'):
            self._tensor_counter = 0
        self._tensor_counter += 1
        if self._tensor_counter % 100 == 0:
            logger.debug(f"DEBUG tensor stats for {symbol}: mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")

        # Move to device
        tensor = tensor.to(self.device)

        return tensor

    def get_latest_timestamp(self, symbol: str):
        """Get timestamp of most recent candle for symbol."""
        if symbol not in self.buffers:
            return None

        return self.buffers[symbol].get_latest_timestamp()

    def get_ready_symbols(self) -> list[str]:
        """Get list of symbols that have enough data for inference."""
        ready = []
        for symbol in self.symbols:
            if self.is_ready(symbol):
                ready.append(symbol)

        return ready

    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        ready_count = len(self.get_ready_symbols())

        buffer_fill_levels = {}
        for symbol, buffer in self.buffers.items():
            fill_pct = (len(buffer.features_buffer) / self.sequence_length) * 100
            buffer_fill_levels[symbol] = fill_pct

        return {
            'symbols_tracked': len(self.symbols),
            'symbols_ready': ready_count,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'scaler_loaded': self.scaler is not None,
            'buffer_fill_levels': buffer_fill_levels,
        }
