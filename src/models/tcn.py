"""
Temporal Convolutional Network (TCN) for Spike Detection

TCN is optimized for time series with:
- Parallel processing (faster than LSTM)
- Dilated causal convolutions for long-range dependencies
- No gradient vanishing issues
- Better for short-term predictions (1-3 minute pre-spike detection)

Architecture:
- 4 TCN blocks with increasing dilation (1, 2, 4, 8)
- 64 filters per block
- Residual connections
- Dropout and batch normalization

Advantages over CNN-LSTM for pre-spike detection:
- ~40-60% faster training and inference
- Better at capturing multi-scale temporal patterns
- Ideal for 1-minute prediction windows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from loguru import logger


class Chomp1d(nn.Module):
    """
    Remove padding from the end to ensure causal convolutions.
    Prevents the model from seeing future data.
    """
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
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
        dropout: float = 0.5
    ):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation

        # First conv layer
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second conv layer
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
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


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for pre-spike detection.

    Stacks multiple TemporalBlocks with exponentially increasing dilation
    to capture patterns at different time scales.

    Architecture:
    - Block 1: dilation=1  (captures 1-minute patterns)
    - Block 2: dilation=2  (captures 2-minute patterns)
    - Block 3: dilation=4  (captures 4-minute patterns)
    - Block 4: dilation=8  (captures 8-minute patterns)

    This allows the model to "see" up to 2^n timesteps back efficiently.
    """
    def __init__(
        self,
        n_features: int,
        sequence_length: int = 60,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.5
    ):
        """
        Initialize TCN.

        Args:
            n_features: Number of input features (24 for pre-spike detection)
            sequence_length: Sequence length (not used but kept for compatibility)
            num_channels: List of channel sizes for each block (default: [64, 64, 64, 64])
            kernel_size: Convolution kernel size (default: 3)
            dropout: Dropout rate (default: 0.5)
        """
        super(TemporalConvNet, self).__init__()

        self.n_features = n_features
        self.sequence_length = sequence_length

        if num_channels is None:
            num_channels = [64, 64, 64, 64]  # 4 blocks, 64 channels each

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8
            in_channels = n_features if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

        # Final layers
        self.fc1 = nn.Linear(num_channels[-1], 64)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

        logger.info(
            f"TCN initialized: {num_levels} blocks, {num_channels} channels, "
            f"kernel_size={kernel_size}, dropout={dropout}"
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch, sequence_length, n_features)

        Returns:
            Binary spike probability (batch, 1)
        """
        # TCN expects (batch, channels, sequence)
        # Input is (batch, sequence, features)
        x = x.transpose(1, 2)

        # Pass through TCN blocks
        y = self.network(x)

        # Take the last timestep output
        y = y[:, :, -1]

        # Fully connected layers
        y = self.fc1(y)
        y = F.relu(y)
        y = self.dropout_fc(y)
        y = self.fc2(y)

        # Sigmoid for binary classification
        return torch.sigmoid(y)


# Alias for compatibility with trainer
SpikeTCN = TemporalConvNet


if __name__ == "__main__":
    # Test TCN
    print("=== TCN Model Test ===\n")

    # Create model
    n_features = 24
    sequence_length = 4320  # 3 days @ 1m
    batch_size = 16

    model = TemporalConvNet(
        n_features=n_features,
        sequence_length=sequence_length,
        num_channels=[64, 64, 64, 64],
        kernel_size=3,
        dropout=0.5
    )

    # Test forward pass
    x = torch.randn(batch_size, sequence_length, n_features)
    y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")
    print()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    print("✓ TCN model test passed!")
