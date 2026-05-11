"""
Time-Series Transformer for Big Mover Prediction

Predicts which coins will gain 20%+ in the next 24 hours.

Architecture:
- Learnable positional encoding
- Multi-head self-attention encoder (4 layers, 8 heads)
- Global attention pooling
- Binary classification head

Key features:
- Attention weights reveal which timesteps matter for predictions
- Parallelizable (faster than LSTM)
- Handles variable-length sequences

Usage:
    from src.models.transformer import BigMoverTransformer

    model = BigMoverTransformer(
        n_features=35,
        seq_len=288,  # 24hr of 5-min candles
        d_model=128,
        n_heads=8,
        n_layers=4
    )

    # Forward pass
    output = model(x)  # x: (batch, seq_len, features)

    # With attention weights for interpretability
    output, attention = model(x, return_attention=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for time-series.

    Unlike fixed sinusoidal encoding, learnable encoding can adapt
    to the specific temporal patterns in crypto price data.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class GlobalAttentionPooling(nn.Module):
    """
    Attention-based pooling over time dimension.

    Instead of mean/max pooling, learns a weighted combination
    of all timesteps. The attention weights reveal which timesteps
    are most important for the prediction.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional (batch, seq_len) boolean mask

        Returns:
            pooled: (batch, d_model) - weighted sum over time
            attention_weights: (batch, seq_len) - importance of each timestep
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax over time dimension
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum
        pooled = torch.einsum('bs,bsd->bd', attention_weights, x)  # (batch, d_model)

        return pooled, attention_weights


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block with pre-norm architecture.

    Pre-norm (LayerNorm before attention) is more stable for training
    and works better for time-series tasks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            return_attention: Whether to return attention weights

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: Optional (batch, n_heads, seq_len, seq_len)
        """
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attention(
            x_norm, x_norm, x_norm,
            need_weights=return_attention,
            average_attn_weights=False
        )
        x = x + attn_output

        # Pre-norm feed-forward
        x = x + self.ff(self.norm2(x))

        return x, attn_weights


class BigMoverTransformer(nn.Module):
    """
    Transformer model for predicting 20%+ crypto gains.

    Architecture:
    1. Input projection: features -> d_model
    2. Positional encoding
    3. N transformer encoder blocks
    4. Global attention pooling
    5. Classification head

    The model learns:
    - Which temporal patterns precede big moves
    - Which features are most predictive
    - How far back to look (via attention)
    """

    def __init__(
        self,
        n_features: int = 35,
        seq_len: int = 288,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        classifier_dropout: float = 0.2
    ):
        """
        Args:
            n_features: Number of input features per timestep
            seq_len: Sequence length (e.g., 288 for 24hr of 5-min candles)
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate in encoder
            classifier_dropout: Dropout rate in classifier
        """
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, seq_len, dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Global attention pooling
        self.pooling = GlobalAttentionPooling(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            x: Input features (batch, seq_len, n_features)
            return_attention: Whether to return attention weights

        Returns:
            output: Predicted probability (batch, 1)
            If return_attention:
                encoder_attentions: List of (batch, n_heads, seq_len, seq_len)
                pooling_attention: (batch, seq_len)
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Encoder layers
        encoder_attentions = []
        for layer in self.encoder_layers:
            x, attn = layer(x, return_attention=return_attention)
            if return_attention and attn is not None:
                encoder_attentions.append(attn)

        # Final layer norm
        x = self.final_norm(x)

        # Global attention pooling
        pooled, pooling_attention = self.pooling(x)

        # Classification
        output = self.classifier(pooled)
        output = torch.sigmoid(output)

        if return_attention:
            return output, encoder_attentions, pooling_attention
        return output

    def get_attention_weights(self, x: torch.Tensor) -> dict:
        """
        Get all attention weights for interpretability analysis.

        Args:
            x: Input features (batch, seq_len, n_features)

        Returns:
            Dictionary with:
                'encoder_attention': List of (batch, n_heads, seq_len, seq_len)
                'pooling_attention': (batch, seq_len) - temporal importance
                'time_importance': (seq_len,) - averaged importance per timestep
        """
        with torch.no_grad():
            output, encoder_attentions, pooling_attention = self.forward(x, return_attention=True)

        # Average pooling attention across batch
        time_importance = pooling_attention.mean(dim=0)  # (seq_len,)

        return {
            'encoder_attention': encoder_attentions,
            'pooling_attention': pooling_attention,
            'time_importance': time_importance,
            'prediction': output
        }


class BigMoverDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Big Mover prediction.

    Expects:
        features: (n_samples, seq_len, n_features) numpy array
        labels: (n_samples,) numpy array with 0/1 labels
    """

    def __init__(self, features, labels, oversample_positive: int = 1):
        """
        Args:
            features: Feature sequences
            labels: Binary labels
            oversample_positive: Factor to oversample positive class
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

        # Oversample positive class if requested
        if oversample_positive > 1:
            pos_mask = labels == 1
            pos_features = features[pos_mask]
            pos_labels = labels[pos_mask]

            for _ in range(oversample_positive - 1):
                self.features = torch.cat([
                    self.features,
                    torch.FloatTensor(pos_features)
                ])
                self.labels = torch.cat([
                    self.labels,
                    torch.FloatTensor(pos_labels)
                ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("=== BigMoverTransformer Test ===\n")

    # Create model
    model = BigMoverTransformer(
        n_features=35,
        seq_len=288,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 288, 35)

    # Basic forward
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Forward with attention
    output, encoder_attn, pool_attn = model(x, return_attention=True)
    print(f"\nEncoder attention shapes: {[a.shape for a in encoder_attn]}")
    print(f"Pooling attention shape: {pool_attn.shape}")

    # Attention analysis
    attention_info = model.get_attention_weights(x)
    print(f"\nTime importance shape: {attention_info['time_importance'].shape}")
    print(f"Time importance (first 10): {attention_info['time_importance'][:10].numpy()}")

    # Test with MPS if available
    if torch.backends.mps.is_available():
        print("\n=== Testing on MPS (Apple Silicon) ===")
        device = torch.device("mps")
        model = model.to(device)
        x = x.to(device)
        output = model(x)
        print(f"MPS output: {output.detach().cpu().numpy().flatten()}")

    print("\n=== Test Complete ===")
