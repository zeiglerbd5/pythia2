"""
CNN-LSTM Hybrid Model for Spike Detection

Implements the architecture per implementation guide achieving 82.44% accuracy:
- CNN: 64 filters, kernel size 3, ReLU, batch norm, 0.5 dropout
- LSTM: Two layers (128 → 80 units), TanH, batch norm, 0.5 dropout
- Output: Sigmoid for binary classification

Per guide: This hybrid leverages CNNs for spatial pattern extraction and
LSTMs for temporal dependencies across 30-60 day lookback periods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from loguru import logger


class SpikeCNNLSTM(nn.Module):
    """
    CNN-LSTM Hybrid Neural Network for cryptocurrency spike detection.

    Per implementation guide:
    - Achieves 82.44% accuracy on Bitcoin spike prediction
    - Targets 7-14 day local extrema (not daily movements)
    - Input: (batch, sequence_length, n_features)
    - Output: Binary spike probability [0, 1]

    Architecture:
    1. CNN layer: Extract spatial patterns from feature vectors
    2. LSTM layer 1: 128 units for temporal modeling
    3. LSTM layer 2: 80 units for refined representation
    4. FC layer: Binary classification

    Each layer includes batch normalization and 50% dropout per guide.
    """

    def __init__(
        self,
        n_features: int,
        sequence_length: int = 60,
        cnn_filters: int = 64,
        cnn_kernel_size: int = 3,
        lstm1_units: int = 128,
        lstm2_units: int = 80,
        dropout: float = 0.5
    ):
        """
        Initialize CNN-LSTM model.

        Args:
            n_features: Number of input features (30-40 after Boruta)
            sequence_length: Sequence length (30-60 day lookback per guide)
            cnn_filters: CNN filters (64 per guide)
            cnn_kernel_size: CNN kernel size (3 per guide)
            lstm1_units: First LSTM layer units (128 per guide)
            lstm2_units: Second LSTM layer units (80 per guide)
            dropout: Dropout rate (0.5 per guide)
        """
        super(SpikeCNNLSTM, self).__init__()

        self.n_features = n_features
        self.sequence_length = sequence_length
        self.cnn_filters = cnn_filters
        self.lstm1_units = lstm1_units
        self.lstm2_units = lstm2_units
        self.dropout = dropout

        # CNN layer for spatial feature extraction
        # Input: (batch, seq_len, features) → (batch, features, seq_len) for Conv1d
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2  # Same padding
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.dropout1 = nn.Dropout(dropout)

        # LSTM Layer 1
        self.lstm1 = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm1_units,
            num_layers=1,
            batch_first=True,
            dropout=0  # We handle dropout separately
        )
        self.bn2 = nn.BatchNorm1d(sequence_length)
        self.dropout2 = nn.Dropout(dropout)

        # LSTM Layer 2
        self.lstm2 = nn.LSTM(
            input_size=lstm1_units,
            hidden_size=lstm2_units,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.bn3 = nn.BatchNorm1d(sequence_length)
        self.dropout3 = nn.Dropout(dropout)

        # Fully connected output layer
        self.fc = nn.Linear(lstm2_units, 1)

        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()

        logger.info(
            f"SpikeCNNLSTM initialized",
            extra={
                "n_features": n_features,
                "sequence_length": sequence_length,
                "cnn_filters": cnn_filters,
                "lstm_units": f"{lstm1_units} → {lstm2_units}",
                "dropout": dropout
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, sequence_length, n_features)
            return_logits: If True, return logits instead of probabilities

        Returns:
            Output tensor of shape (batch, 1) with spike probabilities
        """
        batch_size = x.size(0)

        # CNN layer
        # Reshape: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.permute(0, 2, 1)

        # Apply Conv1d + BatchNorm + ReLU + Dropout
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Reshape back: (batch, filters, seq_len) → (batch, seq_len, filters)
        x = x.permute(0, 2, 1)

        # LSTM Layer 1
        x, _ = self.lstm1(x)  # (batch, seq_len, lstm1_units)
        x = self.bn2(x)
        x = self.dropout2(x)

        # LSTM Layer 2
        x, (h_n, c_n) = self.lstm2(x)  # (batch, seq_len, lstm2_units)
        x = self.bn3(x)
        x = self.dropout3(x)

        # Take only the last timestep output
        x = x[:, -1, :]  # (batch, lstm2_units)

        # Fully connected layer
        x = self.fc(x)  # (batch, 1)

        # Apply sigmoid for probability output
        if not return_logits:
            x = self.sigmoid(x)

        return x

    def get_model_info(self) -> dict:
        """
        Get model architecture information.

        Returns:
            Dictionary with model details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "CNN-LSTM Hybrid",
            "n_features": self.n_features,
            "sequence_length": self.sequence_length,
            "cnn_filters": self.cnn_filters,
            "lstm1_units": self.lstm1_units,
            "lstm2_units": self.lstm2_units,
            "dropout": self.dropout,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "size_mb": total_params * 4 / (1024 ** 2)  # Assuming float32
        }


class AlternativeGRU(nn.Module):
    """
    GRU-based alternative architecture.

    Per guide: GRU delivers equivalent performance to LSTM while training
    60% faster and requiring fewer parameters. Ideal for latency-sensitive applications.
    """

    def __init__(
        self,
        n_features: int,
        sequence_length: int = 60,
        cnn_filters: int = 64,
        cnn_kernel_size: int = 3,
        gru1_units: int = 128,
        gru2_units: int = 80,
        dropout: float = 0.5
    ):
        """
        Initialize CNN-GRU model.

        Args:
            n_features: Number of input features
            sequence_length: Sequence length
            cnn_filters: CNN filters
            cnn_kernel_size: CNN kernel size
            gru1_units: First GRU layer units
            gru2_units: Second GRU layer units
            dropout: Dropout rate
        """
        super(AlternativeGRU, self).__init__()

        self.n_features = n_features
        self.sequence_length = sequence_length

        # CNN layer
        self.conv1 = nn.Conv1d(n_features, cnn_filters, cnn_kernel_size, padding=cnn_kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.dropout1 = nn.Dropout(dropout)

        # GRU layers (faster than LSTM per guide)
        self.gru1 = nn.GRU(cnn_filters, gru1_units, batch_first=True)
        self.bn2 = nn.BatchNorm1d(sequence_length)
        self.dropout2 = nn.Dropout(dropout)

        self.gru2 = nn.GRU(gru1_units, gru2_units, batch_first=True)
        self.bn3 = nn.BatchNorm1d(sequence_length)
        self.dropout3 = nn.Dropout(dropout)

        # Output
        self.fc = nn.Linear(gru2_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass."""
        # CNN
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)

        # GRU layers
        x, _ = self.gru1(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x, _ = self.gru2(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        # Output
        x = x[:, -1, :]
        x = self.fc(x)

        if not return_logits:
            x = self.sigmoid(x)

        return x


def create_model(
    model_type: str,
    n_features: int,
    sequence_length: int = 60,
    device: str = "mps"
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: 'cnn_lstm' or 'gru'
        n_features: Number of input features
        sequence_length: Sequence length
        device: Device to place model on ('mps', 'cuda', 'cpu')

    Returns:
        Model instance on specified device
    """
    if model_type == "cnn_lstm":
        model = SpikeCNNLSTM(n_features, sequence_length)
    elif model_type == "gru":
        model = AlternativeGRU(n_features, sequence_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Move to device
    if device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
        logger.info("Model moved to MPS (Metal Performance Shaders)")
    elif device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("Model moved to CUDA")
    else:
        model = model.to("cpu")
        logger.info("Model on CPU")

    return model


if __name__ == "__main__":
    # Test model
    import numpy as np

    print("=== CNN-LSTM Model Test ===\n")

    # Model parameters (per guide)
    n_features = 35  # 30-40 after Boruta selection
    sequence_length = 60  # 30-60 day lookback
    batch_size = 32

    # Create model
    model = SpikeCNNLSTM(
        n_features=n_features,
        sequence_length=sequence_length
    )

    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    x = torch.randn(batch_size, sequence_length, n_features)
    print(f"Input shape: {x.shape}")

    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Sample predictions: {output[:5].squeeze().tolist()}")

    # Test with different devices
    print("\n=== Device Compatibility ===")
    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")

    for device in devices:
        try:
            test_model = create_model("cnn_lstm", n_features, sequence_length, device)
            test_input = torch.randn(2, sequence_length, n_features).to(device)
            test_output = test_model(test_input)
            print(f"  ✓ {device.upper()}: Working (output shape: {test_output.shape})")
        except Exception as e:
            print(f"  ✗ {device.upper()}: Failed - {e}")

    # Test GRU variant
    print("\n=== GRU Variant Test ===")
    gru_model = AlternativeGRU(n_features, sequence_length)
    gru_output = gru_model(x)
    print(f"GRU Output shape: {gru_output.shape}")

    # Compare parameter counts
    lstm_params = sum(p.numel() for p in model.parameters())
    gru_params = sum(p.numel() for p in gru_model.parameters())
    print(f"\nParameter comparison:")
    print(f"  LSTM: {lstm_params:,} parameters")
    print(f"  GRU:  {gru_params:,} parameters ({(1 - gru_params/lstm_params)*100:.1f}% fewer)")
