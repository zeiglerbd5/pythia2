"""
PPO Agent Wrapper for RL Trading

Implements:
- PPO agent using stable-baselines3
- Custom policy network with action masking
- Checkpointing and loading utilities
- Integration with attention-based policies (Phase 2)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Type, Tuple, List, Callable
from pathlib import Path
from loguru import logger

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


@dataclass
class AgentConfig:
    """Configuration for PPO agent."""
    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048           # Steps per update
    batch_size: int = 64
    n_epochs: int = 10            # Epochs per update
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.99      # Higher GAE for sparse rewards
    clip_range: float = 0.2       # PPO clip range
    clip_range_vf: Optional[float] = None  # Value function clip range
    ent_coef: float = 0.00025       # Low entropy for selectivity (was 0.01)
    vf_coef: float = 0.5          # Value function coefficient
    max_grad_norm: float = 0.5    # Max gradient norm

    # Network architecture
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation_fn: str = "tanh"   # "tanh" or "relu"

    # Training settings
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000       # Evaluation frequency
    n_eval_episodes: int = 10     # Episodes per evaluation
    save_freq: int = 50_000       # Checkpoint frequency

    # Device
    device: str = "auto"          # "auto", "cpu", "cuda", "mps"

    # Paths
    model_dir: str = "models/rl"
    log_dir: str = "logs/rl"


class MaskedCategorical(Categorical):
    """Categorical distribution with action masking."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        """
        Initialize masked categorical distribution.

        Args:
            logits: Raw action logits
            mask: Boolean mask (True = valid action)
        """
        # Apply mask by setting invalid action logits to very negative
        masked_logits = torch.where(
            mask,
            logits,
            torch.tensor(float('-inf'), device=logits.device)
        )
        super().__init__(logits=masked_logits)

    def entropy(self) -> torch.Tensor:
        """Calculate entropy, handling masked actions."""
        # Standard entropy, NaN values from -inf are handled
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(
            torch.isfinite(p_log_p),
            p_log_p,
            torch.zeros_like(p_log_p)
        )
        return -p_log_p.sum(-1)


class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for trading observations.

    Processes raw market features through MLPs before policy/value heads.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
    ):
        """
        Initialize feature extractor.

        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            net_arch: Hidden layer sizes
            activation_fn: Activation function
        """
        super().__init__(observation_space, features_dim)

        net_arch = net_arch or [128, 128]
        input_dim = int(np.prod(observation_space.shape))

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in net_arch:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, features_dim))
        layers.append(activation_fn())

        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        return self.mlp(observations)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy with action masking support.

    Modifies the standard policy to:
    1. Accept action masks from the environment
    2. Apply masks during action sampling and evaluation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional action masking.

        Args:
            obs: Observations
            deterministic: Whether to sample deterministically
            action_masks: Boolean mask of valid actions

        Returns:
            (actions, values, log_probs)
        """
        # Get latent features
        latent_pi, latent_vf = self._get_latent(obs)

        # Get action logits
        action_logits = self.action_net(latent_pi)

        # Get values
        values = self.value_net(latent_vf)

        # Create distribution with masking
        if action_masks is not None:
            distribution = MaskedCategorical(action_logits, action_masks)
        else:
            distribution = Categorical(logits=action_logits)

        # Sample or select best action
        if deterministic:
            actions = torch.argmax(distribution.probs, dim=-1)
        else:
            actions = distribution.sample()

        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions for PPO update.

        Args:
            obs: Observations
            actions: Actions taken
            action_masks: Boolean mask of valid actions

        Returns:
            (values, log_probs, entropy)
        """
        latent_pi, latent_vf = self._get_latent(obs)
        action_logits = self.action_net(latent_pi)
        values = self.value_net(latent_vf)

        if action_masks is not None:
            distribution = MaskedCategorical(action_logits, action_masks)
        else:
            distribution = Categorical(logits=action_logits)

        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_probs, entropy

    def _get_latent(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get latent representations for policy and value."""
        features = self.extract_features(obs)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features = self.pi_features_extractor(features)
            vf_features = self.vf_features_extractor(features)
            latent_pi, latent_vf = self.mlp_extractor(pi_features, vf_features)

        return latent_pi, latent_vf


class ActionMaskCallback(BaseCallback):
    """Callback to handle action masks during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._action_masks: Optional[np.ndarray] = None

    def _on_step(self) -> bool:
        """Called at each step."""
        # Get action masks from environments
        action_masks = []
        for env in self.training_env.envs:
            if hasattr(env, 'get_action_mask'):
                action_masks.append(env.get_action_mask())
            else:
                # Default to all valid
                action_masks.append(np.ones(self.training_env.action_space.n, dtype=bool))

        self._action_masks = np.array(action_masks)
        return True


class DegeneratePolicyCallback(BaseCallback):
    """
    Callback to detect and warn about degenerate policies.

    Monitors action distribution and warns/stops if policy collapses
    to always taking one action.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        patience: int = 5,
        stop_on_degenerate: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.threshold = threshold
        self.patience = patience
        self.stop_on_degenerate = stop_on_degenerate
        self.degenerate_count = 0
        self.action_history = []
        self.check_freq = 1000  # Check every N steps

    def _on_step(self) -> bool:
        """Monitor action distribution."""
        # Collect actions
        if hasattr(self.locals, 'actions'):
            actions = self.locals.get('actions', [])
            if len(actions) > 0:
                self.action_history.extend(actions.flatten().tolist())

        # Check periodically
        if self.n_calls % self.check_freq == 0 and len(self.action_history) > 100:
            # Compute action distribution
            actions_array = np.array(self.action_history[-1000:])  # Last 1000 actions
            unique, counts = np.unique(actions_array, return_counts=True)

            if len(unique) > 0:
                max_ratio = counts.max() / len(actions_array)
                dominant_action = unique[counts.argmax()]

                if max_ratio > self.threshold:
                    self.degenerate_count += 1
                    action_name = "WAIT" if dominant_action == 0 else "ENTER"
                    logger.warning(
                        f"DEGENERATE POLICY DETECTED: {action_name} ratio = {max_ratio:.1%} "
                        f"(count: {self.degenerate_count}/{self.patience})"
                    )

                    if self.stop_on_degenerate and self.degenerate_count >= self.patience:
                        logger.error(
                            "STOPPING TRAINING: Policy collapsed. "
                            "Consider: lower LR, higher entropy, entry budget."
                        )
                        return False
                else:
                    self.degenerate_count = 0

        return True

    def _on_rollout_end(self) -> None:
        """Log action stats at end of rollout."""
        if len(self.action_history) > 0:
            actions_array = np.array(self.action_history[-1000:])
            enter_rate = (actions_array == 1).mean()

            # Log to tensorboard
            if self.logger is not None:
                self.logger.record("train/enter_rate", enter_rate)


class EntryRatePenaltyCallback(BaseCallback):
    """
    Callback that adds entry rate penalty to encourage selectivity.

    This implements Quick Win #4: Entry rate regularization.
    """

    def __init__(
        self,
        target_entry_rate: float = 0.10,  # Target 10% entry rate
        penalty_weight: float = 1.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.target_entry_rate = target_entry_rate
        self.penalty_weight = penalty_weight
        self.entry_rates = []

    def _on_rollout_end(self) -> None:
        """Compute entry rate penalty at end of rollout."""
        if hasattr(self, 'model') and self.model is not None:
            rollout_buffer = self.model.rollout_buffer

            if rollout_buffer is not None and hasattr(rollout_buffer, 'actions'):
                actions = rollout_buffer.actions.flatten()
                entry_rate = (actions == 1).mean()
                self.entry_rates.append(entry_rate)

                # Log entry rate
                if self.logger is not None:
                    self.logger.record("train/entry_rate", entry_rate)
                    self.logger.record("train/target_entry_rate", self.target_entry_rate)

                # Apply penalty to rewards if entry rate too high
                if entry_rate > self.target_entry_rate:
                    excess = entry_rate - self.target_entry_rate
                    penalty = -self.penalty_weight * (excess ** 2)

                    # Note: We can't directly modify rewards in SB3
                    # This is logged for awareness, actual penalty is in reward function
                    if self.logger is not None:
                        self.logger.record("train/entry_rate_penalty", penalty)

    def _on_step(self) -> bool:
        """Required method - return True to continue training."""
        return True


class PPOAgent:
    """
    PPO Agent wrapper for trading environment.

    Provides:
    - Easy initialization and training
    - Action masking support
    - Checkpointing and loading
    - Evaluation utilities
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize PPO agent.

        Args:
            env: Trading environment (or env factory)
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        self.env = env

        # Create directories
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

        # Determine device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.config.device

        logger.info(f"Using device: {device}")

        # Get activation function
        activation_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
        }.get(self.config.activation_fn, nn.Tanh)

        # Policy kwargs
        policy_kwargs = {
            "net_arch": dict(
                pi=self.config.net_arch,
                vf=self.config.net_arch,
            ),
            "activation_fn": activation_fn,
            "features_extractor_class": TradingFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": 128,
                "net_arch": [128],
                "activation_fn": activation_fn,
            },
        }

        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            clip_range_vf=self.config.clip_range_vf,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.config.log_dir,
            device=device,
            verbose=1,
        )

        logger.info(
            f"PPO agent initialized with {sum(p.numel() for p in self.model.policy.parameters())} parameters"
        )

    def train(
        self,
        total_timesteps: Optional[int] = None,
        eval_env: Optional[gym.Env] = None,
        callbacks: Optional[List[BaseCallback]] = None,
        tb_log_name: str = "ppo_trading",
    ) -> None:
        """
        Train the agent.

        Args:
            total_timesteps: Total training timesteps
            eval_env: Environment for evaluation
            callbacks: Additional callbacks
            tb_log_name: TensorBoard log name
        """
        total_timesteps = total_timesteps or self.config.total_timesteps
        callbacks = callbacks or []

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=self.config.model_dir,
            name_prefix="ppo_trading",
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(self.config.model_dir, "best"),
                log_path=self.config.log_dir,
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.n_eval_episodes,
                deterministic=True,
            )
            callbacks.append(eval_callback)

        # Action mask callback
        callbacks.append(ActionMaskCallback())

        # Degenerate policy detection callback (Quick Win #5)
        callbacks.append(DegeneratePolicyCallback(
            threshold=0.90,
            patience=10,
            stop_on_degenerate=False,  # Warn but don't stop
        ))

        # Entry rate monitoring callback
        callbacks.append(EntryRatePenaltyCallback(
            target_entry_rate=0.10,
        ))

        logger.info(f"Starting training for {total_timesteps} timesteps")

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=tb_log_name,
            reset_num_timesteps=False,
        )

        logger.info("Training completed")

    def predict(
        self,
        observation: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Predict action for given observation.

        Args:
            observation: Current observation
            action_mask: Boolean mask of valid actions
            deterministic: Whether to select best action

        Returns:
            (action, info)
        """
        # Convert to tensor
        obs_tensor = torch.as_tensor(observation).float().unsqueeze(0)

        if action_mask is not None:
            mask_tensor = torch.as_tensor(action_mask).bool().unsqueeze(0)
        else:
            mask_tensor = None

        # Get action from policy
        with torch.no_grad():
            # Use the model's policy
            obs_tensor = obs_tensor.to(self.model.device)

            # Get features
            features = self.model.policy.extract_features(obs_tensor)
            latent_pi, latent_vf = self.model.policy.mlp_extractor(features)

            # Get logits
            action_logits = self.model.policy.action_net(latent_pi)
            values = self.model.policy.value_net(latent_vf)

            # Apply mask
            if mask_tensor is not None:
                mask_tensor = mask_tensor.to(self.model.device)
                masked_logits = torch.where(
                    mask_tensor,
                    action_logits,
                    torch.tensor(float('-inf'), device=action_logits.device)
                )
            else:
                masked_logits = action_logits

            # Get action
            if deterministic:
                action = torch.argmax(masked_logits, dim=-1)
            else:
                probs = torch.softmax(masked_logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)

            action = action.cpu().numpy()[0]
            value = values.cpu().numpy()[0, 0]

        return action, {"value": value}

    def save(self, path: Optional[str] = None) -> str:
        """
        Save the model.

        Args:
            path: Save path (uses default if None)

        Returns:
            Actual save path
        """
        if path is None:
            path = os.path.join(self.config.model_dir, "ppo_trading_final")

        self.model.save(path)
        logger.info(f"Model saved to {path}")

        return path

    def load(self, path: str) -> None:
        """
        Load a saved model.

        Args:
            path: Path to saved model
        """
        self.model = PPO.load(path, env=self.env, device=self.config.device)
        logger.info(f"Model loaded from {path}")

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        env: gym.Env,
        config: Optional[AgentConfig] = None,
    ) -> "PPOAgent":
        """
        Create agent from checkpoint.

        Args:
            path: Path to checkpoint
            env: Environment
            config: Configuration (for device settings)

        Returns:
            Loaded agent
        """
        agent = cls(env, config)
        agent.load(path)
        return agent


def make_vec_env(
    env_factory: Callable[[], gym.Env],
    n_envs: int = 4,
    use_subproc: bool = False,
) -> DummyVecEnv:
    """
    Create vectorized environment for parallel training.

    Args:
        env_factory: Function that creates an environment
        n_envs: Number of parallel environments
        use_subproc: Use subprocess (for CPU-intensive envs)

    Returns:
        Vectorized environment
    """
    def make_monitored_env():
        env = env_factory()
        return Monitor(env)

    env_fns = [make_monitored_env for _ in range(n_envs)]

    if use_subproc:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


if __name__ == "__main__":
    # Test agent creation
    import gymnasium as gym
    from gymnasium import spaces

    # Create a simple test environment
    class SimpleTestEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.float32)
            self.action_space = spaces.Discrete(7)
            self._step = 0

        def reset(self, **kwargs):
            self._step = 0
            return np.random.randn(20).astype(np.float32), {}

        def step(self, action):
            self._step += 1
            obs = np.random.randn(20).astype(np.float32)
            reward = np.random.randn() * 0.1
            done = self._step >= 100
            return obs, reward, done, False, {}

        def get_action_mask(self):
            # Random mask for testing
            mask = np.ones(7, dtype=bool)
            mask[np.random.randint(7)] = False
            return mask

    # Create environment
    env = SimpleTestEnv()

    # Create agent
    config = AgentConfig(
        total_timesteps=1000,
        n_steps=64,
        batch_size=32,
    )

    agent = PPOAgent(env, config)

    # Test prediction
    obs, _ = env.reset()
    action, info = agent.predict(obs, action_mask=env.get_action_mask())
    print(f"Predicted action: {action}, value: {info['value']:.4f}")

    # Test training (short)
    print("\nTraining for 1000 steps...")
    agent.train(total_timesteps=1000)

    # Test save/load
    save_path = agent.save("/tmp/test_agent")
    agent.load(save_path)

    # Predict again
    action, info = agent.predict(obs, action_mask=env.get_action_mask())
    print(f"Post-load action: {action}, value: {info['value']:.4f}")

    print("\nAgent tests passed!")
