"""
Reinforcement Learning Trading System for Pythia2

This module implements a Reinforcement Learning-based crypto trading system
that makes continuous micro-decisions based on current market state rather
than predicting future prices - "navigating like whitewater rapids."

Phases:
    Phase 1 - Minimal Viable Agent:
        - TradingEnvironment: Gym-compatible trading environment
        - FeatureExtractor: OHLCV-based feature extraction
        - RewardCalculator: P&L and risk-adjusted rewards
        - PPOAgent: PPO agent with action masking

    Phase 2 - Attention and Enhanced State:
        - MarketAttention: Multi-scale temporal attention
        - MarketState: Enhanced state representation
        - AttentionPolicy: Custom attention-based policy network

    Phase 3 - Continuous Learning:
        - ContinualLearner: Online learning with EWC
        - RegimeDetector: Market regime classification
        - Evaluator: Walk-forward backtesting and A/B testing

Key Design Principles:
    1. Reactive, not predictive: Respond to market state, don't forecast prices
    2. Continuous decisions: Every timestep is a decision point
    3. Attention-based state: Let the model learn WHAT to focus on
    4. Risk-first rewards: Optimize risk-adjusted returns, not raw P&L
    5. Continuous adaptation: Adapt to regime changes without catastrophic forgetting
"""

from .environment import TradingEnvironment, EpisodeConfig
from .features import FeatureExtractor, FeatureConfig
from .rewards import RewardCalculator, RewardConfig
from .agent import PPOAgent, AgentConfig

# Entry-Only Mode (Recommended for spike prediction)
from .entry_timing_env import EntryTimingEnvironment, EntryTimingConfig
from .entry_timing_rewards import (
    EntryTimingRewardCalculator,
    EntryTimingRewardConfig,
    AggressiveEntryRewardCalculator,
)

# Phase 2
from .attention import TemporalAttention, MarketAttention
from .state import MarketState, StateConfig

# Phase 3
from .continual import ContinualLearner, PrioritizedReplayBuffer
from .regime import RegimeDetector, RegimeType
from .evaluation import Evaluator, WalkForwardValidator

__all__ = [
    # Phase 1
    "TradingEnvironment",
    "EpisodeConfig",
    "FeatureExtractor",
    "FeatureConfig",
    "RewardCalculator",
    "RewardConfig",
    "PPOAgent",
    "AgentConfig",
    # Entry-Only Mode
    "EntryTimingEnvironment",
    "EntryTimingConfig",
    "EntryTimingRewardCalculator",
    "EntryTimingRewardConfig",
    "AggressiveEntryRewardCalculator",
    # Phase 2
    "TemporalAttention",
    "MarketAttention",
    "MarketState",
    "StateConfig",
    # Phase 3
    "ContinualLearner",
    "PrioritizedReplayBuffer",
    "RegimeDetector",
    "RegimeType",
    "Evaluator",
    "WalkForwardValidator",
]

__version__ = "0.1.0"
