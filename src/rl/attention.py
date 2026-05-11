"""
Attention Modules for RL Trading Agent (Phase 2)

Implements multi-scale attention mechanisms:
- TemporalAttention: Attention over time dimension
- MarketAttention: Multi-scale attention (micro/meso/macro)
- Cross-timescale attention
- Feature attention
- Attention weight extraction for interpretability

The attention mechanism allows the agent to learn WHAT historical
information is relevant for the current trading decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class AttentionConfig:
    """Configuration for attention modules."""
    d_model: int = 128            # Model dimension
    n_heads: int = 8              # Number of attention heads
    dropout: float = 0.1         # Dropout rate

    # Input dimensions per timescale
    d_micro: int = 15             # Micro features (1-5 min)
    d_meso: int = 10              # Meso features (1-4 hours)
    d_macro: int = 6              # Macro features (24h+)

    # Sequence lengths per timescale
    seq_micro: int = 60           # 60 minutes of micro history
    seq_meso: int = 24            # 24 hours of meso history
    seq_macro: int = 7            # 7 days of macro history


class TemporalAttention(nn.Module):
    """
    Attention pooling over time dimension.

    Uses a learnable query to attend to historical timesteps,
    producing a single context vector that captures relevant
    temporal patterns.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize temporal attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable query vector (pooling query)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.query)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention pooling.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask (batch, seq_len)

        Returns:
            pooled: Pooled output (batch, d_model)
            attention_weights: Attention weights (batch, seq_len)
        """
        batch_size = x.size(0)

        # Expand query to batch size
        query = self.query.expand(batch_size, -1, -1)

        # Apply multi-head attention
        # query: (batch, 1, d_model)
        # key, value: (batch, seq_len, d_model)
        pooled, attention_weights = self.attention(
            query, x, x,
            key_padding_mask=mask,
            need_weights=True,
            average_attn_weights=True,
        )

        # Remove sequence dimension
        pooled = pooled.squeeze(1)  # (batch, d_model)
        attention_weights = attention_weights.squeeze(1)  # (batch, seq_len)

        # Apply layer norm
        pooled = self.norm(pooled)

        return pooled, attention_weights


class CrossTimescaleAttention(nn.Module):
    """
    Cross-attention between timescales.

    Allows information flow between micro, meso, and macro
    representations to capture cross-scale dependencies.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-timescale attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention forward pass.

        Args:
            query: Query tensor (batch, seq_q, d_model)
            key_value: Key/value tensor (batch, seq_kv, d_model)

        Returns:
            output: Attended output (batch, seq_q, d_model)
            attention_weights: Attention weights (batch, seq_q, seq_kv)
        """
        # Apply cross-attention
        attended, weights = self.attention(
            query, key_value, key_value,
            need_weights=True,
            average_attn_weights=True,
        )

        # Residual connection and norm
        output = self.norm(query + self.dropout(attended))

        return output, weights


class FeatureAttention(nn.Module):
    """
    Attention over feature dimension.

    Learns which features are most important for the current
    market state, providing interpretable feature importance.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        """
        Initialize feature attention.

        Args:
            d_input: Input feature dimension
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()

        # Feature scoring network
        self.score_net = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_input),
        )

        # Feature transformation
        self.transform = nn.Linear(d_input, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply feature attention.

        Args:
            features: Input features (batch, d_input)

        Returns:
            output: Weighted features (batch, d_model)
            attention_weights: Feature importance (batch, d_input)
        """
        # Compute feature scores
        scores = self.score_net(features)  # (batch, d_input)

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Weight features
        weighted = features * attention_weights

        # Transform to model dimension
        output = self.transform(weighted)
        output = self.norm(output)

        return output, attention_weights


class MarketAttention(nn.Module):
    """
    Multi-scale attention for market state processing.

    Learns:
    1. Which timesteps in history matter (temporal attention)
    2. Which features matter (feature attention)
    3. How to combine across timescales (cross attention)

    This is the main attention module for the Phase 2 agent.
    """

    def __init__(self, config: Optional[AttentionConfig] = None):
        """
        Initialize market attention module.

        Args:
            config: Attention configuration
        """
        super().__init__()

        self.config = config or AttentionConfig()
        d_model = self.config.d_model
        n_heads = self.config.n_heads
        dropout = self.config.dropout

        # Project each timescale to common dimension
        self.proj_micro = nn.Linear(self.config.d_micro, d_model)
        self.proj_meso = nn.Linear(self.config.d_meso, d_model)
        self.proj_macro = nn.Linear(self.config.d_macro, d_model)

        # Temporal attention for each timescale
        self.attn_micro = TemporalAttention(d_model, n_heads, dropout)
        self.attn_meso = TemporalAttention(d_model, n_heads, dropout)
        self.attn_macro = TemporalAttention(d_model, n_heads, dropout)

        # Cross-timescale attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feature attention (what features to focus on)
        self.feature_attn = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.LayerNorm(d_model),
        )

        # Positional encoding for sequences
        self.pos_encoding_micro = self._create_positional_encoding(
            self.config.seq_micro, d_model
        )
        self.pos_encoding_meso = self._create_positional_encoding(
            self.config.seq_meso, d_model
        )
        self.pos_encoding_macro = self._create_positional_encoding(
            self.config.seq_macro, d_model
        )

    def _create_positional_encoding(
        self,
        max_len: int,
        d_model: int,
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(
        self,
        micro: torch.Tensor,
        meso: torch.Tensor,
        macro: torch.Tensor,
        current: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with attention weight tracking.

        Args:
            micro: Micro history (batch, seq_micro, d_micro)
            meso: Meso history (batch, seq_meso, d_meso)
            macro: Macro history (batch, seq_macro, d_macro)
            current: Current features (batch, d_current) - optional

        Returns:
            fused: Fused state representation (batch, d_model)
            attention_weights: Dict with all attention patterns
        """
        batch_size = micro.size(0)

        # Project to common space and add positional encoding
        micro_proj = self.proj_micro(micro)
        micro_proj = micro_proj + self.pos_encoding_micro[:, :micro.size(1), :]

        meso_proj = self.proj_meso(meso)
        meso_proj = meso_proj + self.pos_encoding_meso[:, :meso.size(1), :]

        macro_proj = self.proj_macro(macro)
        macro_proj = macro_proj + self.pos_encoding_macro[:, :macro.size(1), :]

        # Temporal attention within each scale
        micro_pooled, attn_micro = self.attn_micro(micro_proj)
        meso_pooled, attn_meso = self.attn_meso(meso_proj)
        macro_pooled, attn_macro = self.attn_macro(macro_proj)

        # Stack scale representations for cross-attention
        multi_scale = torch.stack(
            [micro_pooled, meso_pooled, macro_pooled], dim=1
        )  # (batch, 3, d_model)

        # Cross-scale attention
        fused, attn_cross = self.cross_attn(
            multi_scale, multi_scale, multi_scale,
            need_weights=True,
            average_attn_weights=True,
        )
        fused = fused.mean(dim=1)  # Pool across scales

        # Feature attention
        combined = torch.cat([micro_pooled, meso_pooled, macro_pooled], dim=-1)
        feature_weights = F.softmax(self.feature_attn(combined), dim=-1)
        fused = fused * feature_weights

        # Output projection
        output = self.output(fused)

        # Collect attention weights for interpretability
        attention_weights = {
            'micro': attn_micro,       # Which recent minutes mattered
            'meso': attn_meso,         # Which recent hours mattered
            'macro': attn_macro,       # Which recent days mattered
            'cross': attn_cross.squeeze(1),  # How scales interact
            'feature': feature_weights,      # Which features mattered
        }

        return output, attention_weights

    def get_attention_summary(
        self,
        attention_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Summarize attention patterns for logging.

        Args:
            attention_weights: Output from forward pass

        Returns:
            Summary statistics
        """
        summary = {}

        for name, weights in attention_weights.items():
            if weights is None:
                continue

            # Convert to numpy
            w = weights.detach().cpu().numpy()

            # Compute entropy (higher = more uniform attention)
            entropy = -np.sum(w * np.log(w + 1e-8), axis=-1).mean()
            summary[f'{name}_entropy'] = float(entropy)

            # Peak attention (highest single weight)
            peak = w.max(axis=-1).mean()
            summary[f'{name}_peak'] = float(peak)

        return summary


class AttentionPolicy(nn.Module):
    """
    Custom policy network using attention for trading.

    Combines:
    - MarketAttention for state processing
    - Separate policy and value heads
    - Action masking support
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        attention_config: Optional[AttentionConfig] = None,
        hidden_dim: int = 256,
    ):
        """
        Initialize attention policy.

        Args:
            observation_dim: Observation dimension
            action_dim: Number of actions
            attention_config: Attention configuration
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.attention_config = attention_config or AttentionConfig()
        d_model = self.attention_config.d_model

        # Market attention module
        self.market_attention = MarketAttention(attention_config)

        # Current state projection
        self.current_proj = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.Tanh(),
            nn.LayerNorm(d_model),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        current_obs: torch.Tensor,
        micro_history: torch.Tensor,
        meso_history: torch.Tensor,
        macro_history: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through attention policy.

        Args:
            current_obs: Current observation (batch, obs_dim)
            micro_history: Micro history (batch, seq_micro, d_micro)
            meso_history: Meso history (batch, seq_meso, d_meso)
            macro_history: Macro history (batch, seq_macro, d_macro)
            action_mask: Valid action mask (batch, action_dim)

        Returns:
            action_logits: Action logits (batch, action_dim)
            value: State value (batch, 1)
            attention_weights: Attention patterns
        """
        # Process current observation
        current_features = self.current_proj(current_obs)

        # Apply market attention
        attended_state, attention_weights = self.market_attention(
            micro_history, meso_history, macro_history
        )

        # Fuse current and historical
        fused = self.fusion(
            torch.cat([current_features, attended_state], dim=-1)
        )

        # Policy and value
        action_logits = self.policy_head(fused)
        value = self.value_head(fused)

        # Apply action mask
        if action_mask is not None:
            action_logits = torch.where(
                action_mask,
                action_logits,
                torch.tensor(float('-inf'), device=action_logits.device),
            )

        return action_logits, value, attention_weights


if __name__ == "__main__":
    # Test attention modules
    print("Testing Attention Modules\n" + "=" * 50)

    batch_size = 4
    config = AttentionConfig()

    # Create dummy data
    micro = torch.randn(batch_size, config.seq_micro, config.d_micro)
    meso = torch.randn(batch_size, config.seq_meso, config.d_meso)
    macro = torch.randn(batch_size, config.seq_macro, config.d_macro)

    # Test TemporalAttention
    print("\n1. TemporalAttention")
    temporal_attn = TemporalAttention(config.d_model, config.n_heads)
    projected = torch.randn(batch_size, 60, config.d_model)
    pooled, weights = temporal_attn(projected)
    print(f"   Input: {projected.shape}")
    print(f"   Pooled output: {pooled.shape}")
    print(f"   Attention weights: {weights.shape}")
    print(f"   Weights sum: {weights.sum(dim=-1).mean():.4f}")

    # Test FeatureAttention
    print("\n2. FeatureAttention")
    feature_attn = FeatureAttention(64, config.d_model)
    features = torch.randn(batch_size, 64)
    output, feat_weights = feature_attn(features)
    print(f"   Input: {features.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Feature weights: {feat_weights.shape}")

    # Test MarketAttention
    print("\n3. MarketAttention")
    market_attn = MarketAttention(config)
    fused, all_weights = market_attn(micro, meso, macro)
    print(f"   Micro input: {micro.shape}")
    print(f"   Meso input: {meso.shape}")
    print(f"   Macro input: {macro.shape}")
    print(f"   Fused output: {fused.shape}")
    print(f"   Attention keys: {list(all_weights.keys())}")

    # Get attention summary
    summary = market_attn.get_attention_summary(all_weights)
    print("\n   Attention Summary:")
    for key, value in summary.items():
        print(f"     {key}: {value:.4f}")

    # Test AttentionPolicy
    print("\n4. AttentionPolicy")
    obs_dim = 32
    action_dim = 7
    policy = AttentionPolicy(obs_dim, action_dim, config)

    current = torch.randn(batch_size, obs_dim)
    mask = torch.ones(batch_size, action_dim, dtype=torch.bool)
    mask[:, 3] = False  # Mask one action

    logits, value, attn = policy(current, micro, meso, macro, mask)
    print(f"   Current obs: {current.shape}")
    print(f"   Action logits: {logits.shape}")
    print(f"   Value: {value.shape}")
    print(f"   Masked logit (action 3): {logits[0, 3].item():.2f}")

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\n   Total parameters: {total_params:,}")

    print("\nAttention module tests passed!")
