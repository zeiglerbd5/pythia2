"""
LSTM Feature Extractor for Temporal Pattern Recognition

Provides an LSTM-based feature extractor for stable-baselines3 PPO
that can capture temporal patterns in market data for spike prediction.

Architecture options:
1. Simple LSTM with frame stacking (recommended for initial experiments)
2. Stateful LSTM with hidden state management (advanced)

Usage:
    from src.rl.lstm_features import LSTMFeaturesExtractor, FrameStackWrapper

    # Use with custom policy_kwargs
    policy_kwargs = {
        "features_extractor_class": LSTMFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 128,
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2,
            "sequence_length": 60,
        },
    }
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Optional, List, Tuple, Dict, Any
from loguru import logger


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for temporal pattern recognition.

    Processes sequences of observations through an LSTM to capture
    temporal dependencies in market data. Designed for use with
    stable-baselines3 PPO.

    Architecture:
        Input -> LayerNorm -> LSTM -> LayerNorm -> Linear -> Output

    Key design decisions:
    1. LayerNorm before LSTM for input normalization (critical for stability)
    2. Uses last LSTM output (not hidden state) for features
    3. Dropout between LSTM layers for regularization
    4. Tanh activation on output for bounded features

    Note on PPO Integration:
        During PPO training, observations come in batches from the replay buffer
        and are NOT temporally sequential. This extractor handles both:
        - Single timestep inputs (batch_size, input_dim) - treats as sequence of 1
        - Sequence inputs (batch_size, seq_len, input_dim) - from frame stacking

        For best results, use with FrameStackWrapper to provide true sequences.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        sequence_length: int = 60,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """
        Initialize LSTM feature extractor.

        Args:
            observation_space: Gym observation space. Can be:
                - (input_dim,) for single-timestep observations
                - (seq_len, input_dim) for frame-stacked observations
            features_dim: Output feature dimension for policy/value heads
            lstm_hidden_size: LSTM hidden state size per direction
            lstm_num_layers: Number of stacked LSTM layers
            sequence_length: Expected sequence length (for single-timestep expansion)
            dropout: Dropout rate between LSTM layers (only if num_layers > 1)
            bidirectional: Use bidirectional LSTM (doubles hidden size in output)
        """
        super().__init__(observation_space, features_dim)

        # Determine input dimensions from observation space
        obs_shape = observation_space.shape
        if len(obs_shape) == 1:
            # Single timestep observations
            self.input_dim = obs_shape[0]
            self.has_sequence_input = False
        elif len(obs_shape) == 2:
            # Frame-stacked observations (seq_len, input_dim)
            self.input_dim = obs_shape[1]
            self.has_sequence_input = True
            sequence_length = obs_shape[0]  # Use actual sequence length
        else:
            raise ValueError(f"Unexpected observation shape: {obs_shape}")

        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional

        # Direction multiplier for output size
        self.num_directions = 2 if bidirectional else 1

        # Input normalization - critical for LSTM stability
        self.input_norm = nn.LayerNorm(self.input_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Initialize LSTM weights properly
        self._init_lstm_weights()

        # Output normalization and projection
        lstm_output_size = lstm_hidden_size * self.num_directions
        self.output_norm = nn.LayerNorm(lstm_output_size)

        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_size, features_dim),
            nn.Tanh(),  # Bounded output for stable RL
        )

        logger.info(
            f"LSTMFeaturesExtractor initialized: "
            f"input_dim={self.input_dim}, seq_len={sequence_length}, "
            f"hidden={lstm_hidden_size}, layers={lstm_num_layers}, "
            f"bidir={bidirectional}, features_dim={features_dim}"
        )

    def _init_lstm_weights(self) -> None:
        """Initialize LSTM weights for stable training."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier initialization
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal initialization
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Biases: Zero, except forget gate bias = 1
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                # Forget gate is the second quarter of the bias
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.

        Args:
            observations: Input tensor of shape:
                - (batch_size, input_dim) for single timestep
                - (batch_size, seq_len, input_dim) for frame-stacked

        Returns:
            Features of shape (batch_size, features_dim)
        """
        # Handle different input shapes
        if observations.dim() == 2:
            # Single timestep: (batch_size, input_dim)
            # Expand to sequence by repeating (not ideal but functional)
            batch_size = observations.shape[0]

            # Normalize input
            normalized = self.input_norm(observations)

            # Expand to sequence - all timesteps are the same
            # This is a fallback; prefer frame stacking for real sequences
            sequence = normalized.unsqueeze(1).expand(
                -1, self.sequence_length, -1
            ).contiguous()

        elif observations.dim() == 3:
            # Sequence: (batch_size, seq_len, input_dim)
            batch_size, seq_len, input_dim = observations.shape

            # Normalize each timestep
            flat_obs = observations.reshape(-1, input_dim)
            normalized = self.input_norm(flat_obs)
            sequence = normalized.reshape(batch_size, seq_len, input_dim)

        else:
            raise ValueError(
                f"Expected 2D or 3D input, got shape {observations.shape}"
            )

        # Process through LSTM
        # lstm_out: (batch, seq_len, hidden * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(sequence)

        # Take last timestep output
        # For bidirectional, this concatenates forward and backward final outputs
        last_output = lstm_out[:, -1, :]

        # Normalize and project to feature space
        normalized_out = self.output_norm(last_output)
        features = self.output_proj(normalized_out)

        return features


class FrameStackWrapper(gym.Wrapper):
    """
    Frame stacking wrapper to provide temporal sequences to LSTM.

    Wraps the trading environment to stack observations over time,
    providing the LSTM with actual temporal context instead of
    single-timestep observations.

    Inherits from gym.Wrapper for full compatibility with stable-baselines3
    vectorized environments (DummyVecEnv, SubprocVecEnv).

    Usage:
        env = TradingEnvironment(...)
        env = FrameStackWrapper(env, n_frames=60)  # 60 min context
    """

    def __init__(self, env: gym.Env, n_frames: int = 60):
        """
        Initialize frame stacking wrapper.

        Args:
            env: Base Gym environment
            n_frames: Number of frames to stack (temporal context window)
        """
        super().__init__(env)
        self.n_frames = n_frames
        self._frames: List[np.ndarray] = []

        # Update observation space to reflect stacking
        base_shape = env.observation_space.shape
        if len(base_shape) != 1:
            raise ValueError(
                f"FrameStackWrapper expects 1D observations, got shape {base_shape}"
            )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_frames, base_shape[0]),
            dtype=np.float32
        )

        logger.debug(
            f"FrameStackWrapper: {base_shape} -> {self.observation_space.shape}"
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)

        # Initialize frame stack with copies of first observation
        self._frames = [obs.copy() for _ in range(self.n_frames)]

        return self._get_stacked_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update frame stack (FIFO)
        self._frames.pop(0)
        self._frames.append(obs.copy())

        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _get_stacked_obs(self) -> np.ndarray:
        """Get stacked observations as (n_frames, obs_dim) array."""
        return np.array(self._frames, dtype=np.float32)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over LSTM sequence outputs.

    Instead of taking the last LSTM output, learns to weight
    all timesteps based on their relevance for the current decision.

    This can help the model focus on the most relevant parts of
    the temporal context (e.g., the spike onset).
    """

    def __init__(self, hidden_size: int):
        """
        Args:
            hidden_size: LSTM hidden size (or hidden * 2 for bidirectional)
        """
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling over sequence.

        Args:
            lstm_outputs: (batch, seq_len, hidden_size)

        Returns:
            Pooled output: (batch, hidden_size)
        """
        # Compute attention scores
        # (batch, seq_len, hidden) -> (batch, seq_len, 1)
        scores = self.attention(lstm_outputs)

        # Softmax over sequence dimension
        weights = torch.softmax(scores, dim=1)

        # Weighted sum
        # (batch, seq_len, 1) * (batch, seq_len, hidden) -> (batch, hidden)
        pooled = (weights * lstm_outputs).sum(dim=1)

        return pooled


class LSTMWithAttentionExtractor(BaseFeaturesExtractor):
    """
    LSTM feature extractor with attention pooling.

    Variant that uses attention over all LSTM outputs instead of
    just the last timestep. This can help when the most relevant
    information isn't always at the end of the sequence.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        sequence_length: int = 60,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        obs_shape = observation_space.shape
        if len(obs_shape) == 1:
            self.input_dim = obs_shape[0]
            self.has_sequence_input = False
        else:
            self.input_dim = obs_shape[1]
            self.has_sequence_input = True
            sequence_length = obs_shape[0]

        self.sequence_length = sequence_length

        # Input normalization
        self.input_norm = nn.LayerNorm(self.input_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        # Attention pooling
        self.attention = AttentionPooling(lstm_hidden_size)

        # Output projection
        self.output_norm = nn.LayerNorm(lstm_hidden_size)
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features with attention pooling."""
        if observations.dim() == 2:
            batch_size = observations.shape[0]
            normalized = self.input_norm(observations)
            sequence = normalized.unsqueeze(1).expand(
                -1, self.sequence_length, -1
            ).contiguous()
        else:
            batch_size, seq_len, input_dim = observations.shape
            flat_obs = observations.reshape(-1, input_dim)
            normalized = self.input_norm(flat_obs)
            sequence = normalized.reshape(batch_size, seq_len, input_dim)

        # LSTM forward
        lstm_out, _ = self.lstm(sequence)

        # Attention pooling instead of last output
        pooled = self.attention(lstm_out)

        # Project to features
        normalized_out = self.output_norm(pooled)
        features = self.output_proj(normalized_out)

        return features


# Recommended configurations for different scenarios
LSTM_CONFIGS = {
    "fast": {
        # Quick experiments, lower memory
        "features_dim": 64,
        "lstm_hidden_size": 64,
        "lstm_num_layers": 1,
        "sequence_length": 30,
        "dropout": 0.0,
    },
    "balanced": {
        # Good balance of capacity and speed (RECOMMENDED)
        "features_dim": 128,
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        "sequence_length": 60,
        "dropout": 0.1,
    },
    "large": {
        # Higher capacity for complex patterns
        "features_dim": 256,
        "lstm_hidden_size": 256,
        "lstm_num_layers": 3,
        "sequence_length": 120,
        "dropout": 0.2,
    },
}


if __name__ == "__main__":
    # Test LSTM feature extractor
    import gymnasium as gym

    print("=" * 60)
    print("Testing LSTM Feature Extractors")
    print("=" * 60)

    # Test 1: Single timestep observations
    print("\n--- Test 1: Single timestep observations ---")
    obs_space_1d = spaces.Box(low=-1, high=1, shape=(50,), dtype=np.float32)
    extractor_1d = LSTMFeaturesExtractor(
        obs_space_1d,
        **LSTM_CONFIGS["balanced"]
    )

    batch = torch.randn(16, 50)
    features = extractor_1d(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output range: [{features.min().item():.3f}, {features.max().item():.3f}]")

    # Test 2: Frame-stacked observations
    print("\n--- Test 2: Frame-stacked observations ---")
    obs_space_2d = spaces.Box(low=-1, high=1, shape=(60, 50), dtype=np.float32)
    extractor_2d = LSTMFeaturesExtractor(
        obs_space_2d,
        **LSTM_CONFIGS["balanced"]
    )

    batch = torch.randn(16, 60, 50)
    features = extractor_2d(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output range: [{features.min().item():.3f}, {features.max().item():.3f}]")

    # Test 3: With attention
    print("\n--- Test 3: LSTM with attention pooling ---")
    extractor_attn = LSTMWithAttentionExtractor(
        obs_space_2d,
        **LSTM_CONFIGS["balanced"]
    )

    features = extractor_attn(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output range: [{features.min().item():.3f}, {features.max().item():.3f}]")

    # Test 4: Frame stack wrapper (mock)
    print("\n--- Test 4: FrameStackWrapper ---")

    class MockEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=-1, high=1, shape=(50,), dtype=np.float32)
            self.action_space = spaces.Discrete(7)

        def reset(self, **kwargs):
            return np.random.randn(50).astype(np.float32), {}

        def step(self, action):
            return np.random.randn(50).astype(np.float32), 0.1, False, False, {}

    mock_env = MockEnv()
    wrapped_env = FrameStackWrapper(mock_env, n_frames=60)

    obs, info = wrapped_env.reset()
    print(f"Reset observation shape: {obs.shape}")

    for i in range(5):
        obs, reward, _, _, _ = wrapped_env.step(0)
    print(f"After 5 steps observation shape: {obs.shape}")

    # Parameter count
    print("\n--- Parameter counts ---")
    for name, config in LSTM_CONFIGS.items():
        ext = LSTMFeaturesExtractor(obs_space_2d, **config)
        params = sum(p.numel() for p in ext.parameters())
        print(f"{name}: {params:,} parameters")

    print("\n" + "=" * 60)
    print("All LSTM tests passed!")
    print("=" * 60)
