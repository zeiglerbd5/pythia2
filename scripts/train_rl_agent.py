#!/usr/bin/env python3
"""
Training Script for RL Trading Agent

Provides command-line interface for training PPO agents with:
- Argument parsing for hyperparameters
- Training loop with logging
- Periodic evaluation
- Model checkpointing
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import torch

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl.environment import TradingEnvironment, EpisodeConfig
from src.rl.features import FeatureExtractor, FeatureConfig
from src.rl.rewards import (
    RewardCalculator, RewardConfig, RewardType,
    SpikeAwareRewardCalculator, SpikeAwareRewardConfig,
    SpikeQualityBonusCalculator, SpikeQualityBonusConfig,
)
from src.rl.agent import PPOAgent, AgentConfig, make_vec_env
from src.rl.lstm_features import (
    LSTMFeaturesExtractor,
    LSTMWithAttentionExtractor,
    FrameStackWrapper,
    LSTM_CONFIGS,
)
# Entry-only timing imports
from src.rl.entry_timing_env import EntryTimingEnvironment, EntryTimingConfig
from src.rl.entry_timing_rewards import (
    EntryTimingRewardCalculator, EntryTimingRewardConfig,
    AggressiveEntryRewardCalculator,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL Trading Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data settings
    parser.add_argument(
        "--db-path",
        type=str,
        default="/Users/bz/Pythia2/full_pythia.duckdb",
        help="Path to DuckDB database",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Symbols to train on (default: all available)",
    )

    # Episode settings
    parser.add_argument(
        "--episode-length",
        type=int,
        default=480,  # 8 hours - proven stable for v1/v2/v4
        help="Episode length in minutes (480 = 8 hours recommended)",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=10,
        help="Maximum trades per episode",
    )
    parser.add_argument(
        "--initial-stop",
        type=float,
        default=0.02,
        help="Initial stop loss percentage",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0055,
        help="Transaction fee rate",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["random", "event_anchored", "sequential"],
        default="event_anchored",  # Default to event_anchored for spike training
        help="Episode sampling mode (event_anchored recommended for spike training)",
    )

    # Training settings
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Steps per PPO update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Epochs per PPO update",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value function coefficient",
    )

    # Network architecture
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Network architecture (hidden layer sizes)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["tanh", "relu"],
        default="tanh",
        help="Activation function",
    )

    # Reward settings
    parser.add_argument(
        "--reward-type",
        type=str,
        choices=["basic_pnl", "risk_adjusted", "sharpe", "hybrid", "spike_aware", "spike_quality_bonus"],
        default="spike_quality_bonus",  # v4: hybrid + additive bonuses
        help="Reward function type (spike_quality_bonus recommended for v4)",
    )
    parser.add_argument(
        "--reward-scale-win",
        type=float,
        default=15.0,  # Increased for spike_aware
        help="Reward scale for winning trades",
    )
    parser.add_argument(
        "--reward-scale-loss",
        type=float,
        default=12.0,  # More symmetric for spike_aware
        help="Reward scale for losing trades",
    )
    parser.add_argument(
        "--spike-freshness-weight",
        type=float,
        default=1.0,
        help="Weight for spike freshness in reward (spike_aware only)",
    )
    parser.add_argument(
        "--penalty-already-traded",
        type=float,
        default=0.5,
        help="Penalty for trading already-traded spike (spike_aware only)",
    )

    # Evaluation settings
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluation frequency (timesteps)",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50_000,
        help="Checkpoint save frequency (timesteps)",
    )

    # Output settings
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/rl",
        help="Model save directory",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/rl",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (default: timestamp)",
    )

    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level",
    )
    parser.add_argument(
        "--continue-from",
        type=str,
        default=None,
        help="Path to checkpoint to continue training from",
    )

    # LSTM settings (Strategy 3)
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Use LSTM feature extractor for temporal pattern recognition",
    )
    parser.add_argument(
        "--lstm-config",
        type=str,
        choices=["fast", "balanced", "large"],
        default="balanced",
        help="LSTM configuration preset (fast=quick test, balanced=recommended, large=high capacity)",
    )
    parser.add_argument(
        "--lstm-sequence-length",
        type=int,
        default=None,
        help="LSTM sequence length (overrides preset if specified)",
    )
    parser.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=None,
        help="LSTM hidden size (overrides preset if specified)",
    )
    parser.add_argument(
        "--lstm-num-layers",
        type=int,
        default=None,
        help="Number of LSTM layers (overrides preset if specified)",
    )
    parser.add_argument(
        "--lstm-features-dim",
        type=int,
        default=None,
        help="LSTM output features dimension (overrides preset if specified)",
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        help="Use attention pooling over LSTM outputs (instead of last timestep)",
    )

    # =========================================
    # ENTRY-ONLY MODE (Recommended for spike prediction)
    # =========================================
    parser.add_argument(
        "--entry-only",
        action="store_true",
        help="Use entry-only timing environment with rule-based exits (RECOMMENDED)",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.12,
        help="Take profit percentage for entry-only mode (default: 12%%)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.02,
        help="Stop loss percentage for entry-only mode (default: 2%%)",
    )
    parser.add_argument(
        "--max-hold-minutes",
        type=int,
        default=1440,
        help="Maximum hold time in minutes for entry-only mode (default: 1440 = 24hr)",
    )
    parser.add_argument(
        "--cooldown-minutes",
        type=int,
        default=30,
        help="Minimum minutes between entries in entry-only mode (default: 30)",
    )
    parser.add_argument(
        "--aggressive-entries",
        action="store_true",
        help="Use aggressive entry reward calculator (encourages more entries)",
    )

    return parser.parse_args()


def create_env_config(args: argparse.Namespace) -> EpisodeConfig:
    """Create episode configuration from arguments."""
    return EpisodeConfig(
        episode_length=args.episode_length,
        symbols=args.symbols,
        max_trades_per_episode=args.max_trades,
        initial_stop_pct=args.initial_stop,
        fee_rate=args.fee_rate,
        sampling_mode=args.sampling_mode,
    )


def create_reward_config(args: argparse.Namespace):
    """Create reward configuration from arguments.

    Returns:
        RewardConfig, SpikeAwareRewardConfig, or SpikeQualityBonusConfig depending on reward_type
    """
    if args.reward_type == "spike_quality_bonus":
        # v4: Hybrid base + additive bonuses (RECOMMENDED)
        return SpikeQualityBonusConfig()
    elif args.reward_type == "spike_aware":
        # v3: Penalty-heavy approach (DEPRECATED - tends to go negative)
        return SpikeAwareRewardConfig(
            reward_scale_win=args.reward_scale_win,
            reward_scale_loss=args.reward_scale_loss,
            spike_freshness_weight=args.spike_freshness_weight,
            penalty_already_traded_spike=args.penalty_already_traded,
        )
    else:
        # Legacy reward calculators
        reward_type = {
            "basic_pnl": RewardType.BASIC_PNL,
            "risk_adjusted": RewardType.RISK_ADJUSTED,
            "sharpe": RewardType.SHARPE,
            "hybrid": RewardType.HYBRID,
        }[args.reward_type]

        return RewardConfig(
            reward_type=reward_type,
            reward_scale_win=args.reward_scale_win,
            reward_scale_loss=args.reward_scale_loss,
        )


def get_lstm_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Get LSTM configuration from arguments, with preset as base."""
    # Start with preset
    config = LSTM_CONFIGS[args.lstm_config].copy()

    # Override with explicit arguments
    if args.lstm_sequence_length is not None:
        config["sequence_length"] = args.lstm_sequence_length
    if args.lstm_hidden_size is not None:
        config["lstm_hidden_size"] = args.lstm_hidden_size
    if args.lstm_num_layers is not None:
        config["lstm_num_layers"] = args.lstm_num_layers
    if args.lstm_features_dim is not None:
        config["features_dim"] = args.lstm_features_dim

    return config


def create_entry_timing_config(args: argparse.Namespace) -> EntryTimingConfig:
    """Create entry timing configuration from arguments."""
    return EntryTimingConfig(
        episode_length=args.episode_length,
        symbols=args.symbols,
        max_trades_per_episode=args.max_trades,
        take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss,
        max_hold_minutes=args.max_hold_minutes,
        cooldown_minutes=args.cooldown_minutes,
        fee_rate=args.fee_rate,
        sampling_mode=args.sampling_mode,
    )


def create_agent_config(args: argparse.Namespace, experiment_name: str) -> AgentConfig:
    """Create agent configuration from arguments."""
    return AgentConfig(
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        net_arch=args.net_arch,
        activation_fn=args.activation,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_freq=args.save_freq,
        model_dir=os.path.join(args.model_dir, experiment_name),
        log_dir=os.path.join(args.log_dir, experiment_name),
        device=args.device,
    )


def setup_logging(log_dir: str, verbose: int) -> None:
    """Setup logging configuration."""
    # Remove default handler
    logger.remove()

    # Add console handler
    level = "DEBUG" if verbose > 1 else "INFO" if verbose > 0 else "WARNING"
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
    )

    # Add file handler
    log_file = os.path.join(log_dir, "train.log")
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="100 MB",
    )


def set_seeds(seed: Optional[int]) -> None:
    """Set random seeds for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")


def create_lstm_agent(
    env,
    config: AgentConfig,
    lstm_config: Dict[str, Any],
    use_attention: bool = False,
) -> PPOAgent:
    """
    Create PPO agent with LSTM feature extractor.

    This creates a custom PPO model that uses the LSTM feature extractor
    for temporal pattern recognition instead of the default MLP extractor.

    Args:
        env: Training environment (should be wrapped with FrameStackWrapper)
        config: Agent configuration
        lstm_config: LSTM configuration dict from LSTM_CONFIGS
        use_attention: Use attention pooling over LSTM outputs

    Returns:
        PPOAgent with LSTM feature extractor
    """
    from stable_baselines3 import PPO
    import torch.nn as nn

    # Create agent instance but don't use its model
    agent = PPOAgent.__new__(PPOAgent)
    agent.config = config
    agent.env = env

    # Create directories
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # Determine device
    if config.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = config.device

    logger.info(f"Using device: {device}")

    # Get activation function
    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
    }.get(config.activation_fn, nn.Tanh)

    # Choose LSTM extractor class
    if use_attention:
        extractor_class = LSTMWithAttentionExtractor
        logger.info("Using LSTMWithAttentionExtractor (attention pooling)")
    else:
        extractor_class = LSTMFeaturesExtractor
        logger.info("Using LSTMFeaturesExtractor (last timestep)")

    # Policy kwargs with LSTM feature extractor
    policy_kwargs = {
        "net_arch": dict(
            pi=config.net_arch,
            vf=config.net_arch,
        ),
        "activation_fn": activation_fn,
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": {
            "features_dim": lstm_config["features_dim"],
            "lstm_hidden_size": lstm_config["lstm_hidden_size"],
            "lstm_num_layers": lstm_config["lstm_num_layers"],
            "sequence_length": lstm_config["sequence_length"],
            "dropout": lstm_config["dropout"],
        },
    }

    # Create PPO model with LSTM
    agent.model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=config.log_dir,
        device=device,
        verbose=1,
    )

    total_params = sum(p.numel() for p in agent.model.policy.parameters())
    logger.info(f"PPO+LSTM agent initialized with {total_params:,} parameters")

    return agent


def main() -> int:
    """Main training function."""
    args = parse_args()

    # Generate experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        # Add entry_only prefix for clarity
        prefix = "entry_" if args.entry_only else ""
        experiment_name = f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create configurations based on mode
    if args.entry_only:
        # Entry-only mode: simplified environment with rule-based exits
        env_config = create_entry_timing_config(args)
        reward_config = EntryTimingRewardConfig()
    else:
        # Legacy mode: full 7-action environment
        env_config = create_env_config(args)
        reward_config = create_reward_config(args)

    agent_config = create_agent_config(args, experiment_name)

    # Get LSTM config if using LSTM
    lstm_config = get_lstm_config(args) if args.use_lstm else None

    # Create directories
    Path(agent_config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(agent_config.log_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(agent_config.log_dir, args.verbose)

    # Set seeds
    set_seeds(args.seed)

    logger.info("=" * 60)
    if args.entry_only:
        logger.info("ENTRY-ONLY RL Trading Agent Training")
        logger.info("(Binary action space: WAIT or ENTER)")
    else:
        logger.info("RL Trading Agent Training")
    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Total timesteps: {args.total_timesteps:,}")
    logger.info(f"Environments: {args.n_envs}")
    logger.info(f"Device: {args.device}")

    if args.entry_only:
        logger.info("=" * 60)
        logger.info("Entry-Only Configuration:")
        logger.info(f"  Take Profit: {args.take_profit*100:.1f}%")
        logger.info(f"  Stop Loss: {args.stop_loss*100:.1f}%")
        logger.info(f"  Max Hold: {args.max_hold_minutes} minutes")
        logger.info(f"  Cooldown: {args.cooldown_minutes} minutes")
        logger.info(f"  Aggressive entries: {args.aggressive_entries}")

    if args.use_lstm:
        logger.info("=" * 60)
        logger.info("LSTM Configuration (Strategy 3)")
        logger.info(f"  Preset: {args.lstm_config}")
        logger.info(f"  Sequence length: {lstm_config['sequence_length']}")
        logger.info(f"  Hidden size: {lstm_config['lstm_hidden_size']}")
        logger.info(f"  Num layers: {lstm_config['lstm_num_layers']}")
        logger.info(f"  Features dim: {lstm_config['features_dim']}")
        logger.info(f"  Dropout: {lstm_config['dropout']}")
        logger.info(f"  Use attention: {args.use_attention}")
    logger.info("=" * 60)

    # Create environment factory based on mode
    if args.entry_only:
        # Entry-only mode
        def make_base_env():
            feature_extractor = FeatureExtractor(FeatureConfig())

            # Choose reward calculator
            if args.aggressive_entries:
                reward_calculator = AggressiveEntryRewardCalculator(reward_config)
            else:
                reward_calculator = EntryTimingRewardCalculator(reward_config)

            return EntryTimingEnvironment(
                db_path=args.db_path,
                config=env_config,
                reward_calculator=reward_calculator,
                feature_extractor=feature_extractor,
            )
    else:
        # Legacy full-action mode
        def make_base_env():
            feature_extractor = FeatureExtractor(FeatureConfig())

            # Choose reward calculator based on config type
            if isinstance(reward_config, SpikeQualityBonusConfig):
                reward_calculator = SpikeQualityBonusCalculator(bonus_config=reward_config)
            elif isinstance(reward_config, SpikeAwareRewardConfig):
                reward_calculator = SpikeAwareRewardCalculator(reward_config)
            else:
                reward_calculator = RewardCalculator(reward_config)

            return TradingEnvironment(
                db_path=args.db_path,
                config=env_config,
                reward_calculator=reward_calculator,
                feature_extractor=feature_extractor,
            )

    # Create environment factory (with optional frame stacking for LSTM)
    def make_env():
        env = make_base_env()
        if args.use_lstm:
            # Wrap with FrameStackWrapper for temporal sequences
            env = FrameStackWrapper(env, n_frames=lstm_config["sequence_length"])
        return env

    try:
        # Log reward calculator type
        if args.entry_only:
            calc_type = "AggressiveEntryRewardCalculator" if args.aggressive_entries else "EntryTimingRewardCalculator"
            logger.info(f"Using {calc_type} (entry-only mode)")
        elif isinstance(reward_config, SpikeQualityBonusConfig):
            logger.info("Using SpikeQualityBonusCalculator (v4 - hybrid + bonuses)")
        elif isinstance(reward_config, SpikeAwareRewardConfig):
            logger.info("Using SpikeAwareRewardCalculator (v3 - DEPRECATED)")
        else:
            logger.info(f"Using legacy RewardCalculator: {reward_config.reward_type}")

        # Create vectorized training environment
        logger.info("Creating training environments...")

        if args.n_envs > 1:
            train_env = make_vec_env(make_env, n_envs=args.n_envs)
        else:
            train_env = make_env()

        # Create evaluation environment
        logger.info("Creating evaluation environment...")
        eval_env = make_env()

        # Create or load agent
        if args.continue_from:
            logger.info(f"Loading checkpoint from {args.continue_from}")
            agent = PPOAgent.from_checkpoint(
                args.continue_from,
                train_env,
                agent_config,
            )
        elif args.use_lstm:
            # Create PPO with LSTM feature extractor
            logger.info("Creating PPO agent with LSTM feature extractor...")
            agent = create_lstm_agent(train_env, agent_config, lstm_config, args.use_attention)
        else:
            logger.info("Creating new PPO agent...")
            agent = PPOAgent(train_env, agent_config)

        # Log configuration
        logger.info(f"Network architecture: {args.net_arch}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Reward type: {args.reward_type}")
        logger.info(f"Episode length: {args.episode_length} minutes")

        # Train
        logger.info("Starting training...")
        agent.train(
            total_timesteps=args.total_timesteps,
            eval_env=eval_env,
            tb_log_name=experiment_name,
        )

        # Save final model
        final_path = agent.save()
        logger.info(f"Final model saved to {final_path}")

        # Log summary
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {final_path}")
        logger.info(f"TensorBoard logs: {agent_config.log_dir}")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1

    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        return 1

    finally:
        # Cleanup
        if 'train_env' in locals():
            train_env.close()
        if 'eval_env' in locals():
            eval_env.close()


if __name__ == "__main__":
    sys.exit(main())
