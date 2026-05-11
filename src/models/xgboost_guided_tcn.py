"""
XGBoost-Guided Temporal Convolutional Network (TCN)

A TCN model that incorporates knowledge transfer from XGBoost via:
1. Delta Layer - Computes feature acceleration (t - t-1)
2. Feature Attention - Initialized from XGBoost importance weights
3. TCN Backbone - Temporal pattern recognition

Key insight from XGBoost analysis:
- Winners show features ACCELERATING, not just elevated
- volume_zscore_delta > 0.5, NATR_delta > 0.05

Architecture:
    Input (batch, 30, 24) - 30 candles x 24 features
    → Delta Layer (outputs 48 channels: original + deltas)
    → Feature Attention (XGBoost importance-weighted)
    → TCN Backbone (4 blocks, dilations 1,2,4,8)
    → FC Layers (64 → 1 → sigmoid)
    → Output: spike probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from loguru import logger

# Default XGBoost V3 feature importances (from analysis)
# These are used to initialize the attention weights
DEFAULT_XGBOOST_IMPORTANCES = {
    'returns': 0.088,
    'MACD': 0.187,
    'MACD_signal': 0.030,
    'MACD_hist': 0.025,
    'RSI_14': 0.236,
    'NATR': 0.020,
    'BB_width': 0.015,
    'BB_squeeze': 0.012,
    'VWAP_distance': 0.055,
    'volume_zscore': 0.040,
    'volume_roc': 0.025,
    'OBV': 0.020,
    'trade_count': 0.018,
    'buy_sell_ratio': 0.022,
    'roll_measure': 0.030,
    'order_flow_imbalance': 0.028,
    'vpin': 0.025,
    'bid_ask_spread_pct': 0.055,
    'order_book_depth_ratio': 0.020,
    'large_order_imbalance': 0.015,
    'returns_5m': 0.019,
    'volume_zscore_5m': 0.018,
    'returns_15m': 0.012,
    'volume_zscore_15m': 0.010,
}

FEATURE_COLUMNS = list(DEFAULT_XGBOOST_IMPORTANCES.keys())


class TemporalDeltaLayer(nn.Module):
    """
    Computes feature deltas (acceleration) across time steps.

    Key discovery from XGBoost analysis:
    Winners show features INCREASING, not just elevated:
    - volume_zscore_delta > 0.5
    - NATR_delta > 0.05
    - returns_5m > 0

    Outputs concatenation of original features and their deltas.
    """

    def __init__(self):
        super(TemporalDeltaLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature deltas and concatenate with original.

        Args:
            x: Input tensor (batch, seq_len, features)

        Returns:
            Concatenated tensor (batch, seq_len, features * 2)
            First half: original features
            Second half: delta features (t - t-1)
        """
        # Compute deltas: x[t] - x[t-1]
        deltas = x[:, 1:, :] - x[:, :-1, :]

        # Pad first timestep with zeros (no delta for t=0)
        zero_pad = torch.zeros(
            x.size(0), 1, x.size(2),
            device=x.device, dtype=x.dtype
        )
        deltas = torch.cat([zero_pad, deltas], dim=1)

        # Concatenate original and deltas
        return torch.cat([x, deltas], dim=-1)


class XGBoostAttention(nn.Module):
    """
    Feature attention layer initialized from XGBoost importance weights.

    XGBoost V3 top features:
    - RSI_14: 23.6%
    - MACD: 18.7%
    - returns: 8.8%
    - bid_ask_spread_pct: 5.5%
    - VWAP_distance: 5.5%

    Attention weights are learnable but initialized from XGBoost.
    This gives the TCN a "warm start" with known-important features.
    """

    def __init__(
        self,
        n_features: int,
        xgboost_importances: Optional[List[float]] = None,
        temperature: float = 1.0,
        learnable_temp: bool = True
    ):
        """
        Initialize attention with XGBoost importance weights.

        Args:
            n_features: Number of input features (48 after delta layer)
            xgboost_importances: List of importance values for original features
            temperature: Softmax temperature (lower = sharper attention)
            learnable_temp: Whether temperature is a learnable parameter
        """
        super(XGBoostAttention, self).__init__()

        # Initialize attention weights
        if xgboost_importances is not None:
            # Use provided importances for original features
            # For delta features, use same importances (they track same features)
            orig_weights = torch.tensor(xgboost_importances, dtype=torch.float32)
            delta_weights = orig_weights.clone()
            init_weights = torch.cat([orig_weights, delta_weights])
        else:
            # Use default importances
            default_values = list(DEFAULT_XGBOOST_IMPORTANCES.values())
            orig_weights = torch.tensor(default_values, dtype=torch.float32)
            delta_weights = orig_weights.clone()
            init_weights = torch.cat([orig_weights, delta_weights])

        # Normalize to [0, 1] range
        init_weights = init_weights / init_weights.max()

        # Create learnable attention weights
        self.attention_weights = nn.Parameter(init_weights)

        # Temperature for softmax sharpness
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

        self.n_features = n_features

        logger.info(
            f"XGBoostAttention initialized: {n_features} features, "
            f"temp={temperature:.2f}, learnable={learnable_temp}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned attention to features.

        Args:
            x: Input tensor (batch, seq_len, features)

        Returns:
            Attention-weighted tensor (batch, seq_len, features)
        """
        # Compute softmax attention weights
        attention = F.softmax(self.attention_weights / self.temperature, dim=-1)

        # Apply attention (element-wise multiplication)
        # Expand attention to match input shape
        attention = attention.unsqueeze(0).unsqueeze(0)  # (1, 1, features)

        return x * attention

    def get_attention_weights(self) -> torch.Tensor:
        """Return current attention weights (for analysis/visualization)."""
        with torch.no_grad():
            return F.softmax(self.attention_weights / self.temperature, dim=-1)


class Chomp1d(nn.Module):
    """Remove padding from the end to ensure causal convolutions."""

    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Single TCN block with dilated causal convolution.

    Includes:
    - Two dilated causal conv layers
    - Batch normalization
    - ReLU activation
    - Dropout
    - Residual connection
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.3
    ):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation

        # First conv layer
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second conv layer
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class XGBoostGuidedTCN(nn.Module):
    """
    XGBoost-Guided Temporal Convolutional Network.

    Incorporates knowledge transfer from XGBoost:
    1. Delta Layer - Explicitly computes feature acceleration
    2. Feature Attention - Initialized from XGBoost importance
    3. TCN Backbone - Captures temporal patterns

    The delta layer embeds the key insight that "winners show
    features accelerating, not just elevated."
    """

    def __init__(
        self,
        n_features: int = 24,
        sequence_length: int = 30,
        xgboost_importances: Optional[List[float]] = None,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
        attention_temperature: float = 1.0
    ):
        """
        Initialize XGBoost-Guided TCN.

        Args:
            n_features: Number of input features (24 for spike detection)
            sequence_length: Input sequence length (30 candles)
            xgboost_importances: List of XGBoost feature importances
            num_channels: TCN channel sizes (default: [64, 64, 64, 64])
            kernel_size: TCN kernel size (default: 3)
            dropout: Dropout rate (default: 0.3)
            attention_temperature: Attention softmax temperature
        """
        super(XGBoostGuidedTCN, self).__init__()

        self.n_features = n_features
        self.sequence_length = sequence_length

        if num_channels is None:
            num_channels = [64, 64, 64, 64]

        # 1. Delta Layer - computes acceleration
        self.delta_layer = TemporalDeltaLayer()

        # After delta layer: n_features * 2 (original + deltas)
        n_features_with_delta = n_features * 2

        # 2. Feature Attention - XGBoost importance-weighted
        self.attention = XGBoostAttention(
            n_features=n_features_with_delta,
            xgboost_importances=xgboost_importances,
            temperature=attention_temperature
        )

        # 3. TCN Backbone
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8
            in_channels = n_features_with_delta if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    dilation=dilation, dropout=dropout
                )
            )

        self.tcn = nn.Sequential(*layers)

        # 4. Output layers
        self.fc1 = nn.Linear(num_channels[-1], 64)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

        # Store config for serialization
        self.config = {
            'n_features': n_features,
            'sequence_length': sequence_length,
            'num_channels': num_channels,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'attention_temperature': attention_temperature
        }

        logger.info(
            f"XGBoostGuidedTCN initialized: {n_features} features, "
            f"seq_len={sequence_length}, channels={num_channels}, "
            f"dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, sequence_length, n_features)

        Returns:
            Spike probability (batch, 1)
        """
        # 1. Compute deltas (acceleration)
        x = self.delta_layer(x)  # (batch, seq, 48)

        # 2. Apply attention
        x = self.attention(x)  # (batch, seq, 48)

        # 3. Transpose for TCN (expects channels first)
        x = x.transpose(1, 2)  # (batch, 48, seq)

        # 4. TCN backbone
        x = self.tcn(x)  # (batch, 64, seq)

        # 5. Take last timestep output
        x = x[:, :, -1]  # (batch, 64)

        # 6. Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        # 7. Sigmoid for probability
        return torch.sigmoid(x)

    def get_attention_weights(self) -> torch.Tensor:
        """Return current attention weights for analysis."""
        return self.attention.get_attention_weights()

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    xgboost_model_path: Optional[str] = None,
    device: str = 'mps'
) -> XGBoostGuidedTCN:
    """
    Factory function to create XGBoostGuidedTCN with XGBoost importances.

    Args:
        xgboost_model_path: Path to XGBoost V3 model pickle
        device: Target device ('mps', 'cuda', 'cpu')

    Returns:
        XGBoostGuidedTCN model on specified device
    """
    importances = None

    if xgboost_model_path:
        try:
            import joblib
            model_data = joblib.load(xgboost_model_path)
            xgb_model = model_data.get('model', model_data)

            if hasattr(xgb_model, 'feature_importances_'):
                importances = xgb_model.feature_importances_.tolist()
                logger.info(f"Loaded XGBoost importances from {xgboost_model_path}")
        except Exception as e:
            logger.warning(f"Could not load XGBoost model: {e}. Using defaults.")

    model = XGBoostGuidedTCN(
        n_features=24,
        sequence_length=30,
        xgboost_importances=importances
    )

    # Move to device
    if device == 'mps' and torch.backends.mps.is_available():
        model = model.to('mps')
    elif device == 'cuda' and torch.cuda.is_available():
        model = model.to('cuda')
    else:
        model = model.to('cpu')

    logger.info(f"Model created with {model.count_parameters():,} parameters")

    return model


if __name__ == "__main__":
    # Test the model
    print("=== XGBoostGuidedTCN Test ===\n")

    # Create model
    model = XGBoostGuidedTCN(
        n_features=24,
        sequence_length=30,
        num_channels=[64, 64, 64, 64],
        dropout=0.3
    )

    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 30, 24)
    y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"Parameters:   {model.count_parameters():,}")
    print()

    # Show attention weights
    attention = model.get_attention_weights()
    print(f"Attention weights shape: {attention.shape}")
    print(f"Top 5 attended features (original):")
    top_indices = attention[:24].argsort(descending=True)[:5]
    for idx in top_indices:
        print(f"  {FEATURE_COLUMNS[idx]}: {attention[idx]:.4f}")

    print()
    print("Top 5 attended features (deltas):")
    top_delta_indices = attention[24:].argsort(descending=True)[:5]
    for idx in top_delta_indices:
        print(f"  {FEATURE_COLUMNS[idx]}_delta: {attention[24 + idx]:.4f}")

    print()
    print("✓ XGBoostGuidedTCN test passed!")
