#!/usr/bin/env python3
"""
Test Suite for RL Trading System

Verifies:
1. Environment can reset and step
2. Agent can be trained for a few episodes
3. Model can be saved and loaded
4. Basic metrics are computed
"""

import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from loguru import logger

# Configure loguru for tests
logger.remove()
logger.add(sys.stdout, level="INFO")


class MockTradingEnvironment(gym.Env):
    """Mock environment for testing without database."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(26,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)

        self._step = 0
        self._position = None
        self._price = 100.0
        self.episode_trades = []

    def reset(self, **kwargs):
        self._step = 0
        self._position = None
        self._price = 100.0
        self.episode_trades = []
        return self._get_obs(), {'symbol': 'TEST-USD'}

    def step(self, action):
        self._step += 1
        self._price *= 1 + np.random.randn() * 0.01

        # Simulate trading
        reward = 0.0
        info = {
            'symbol': 'TEST-USD',
            'step': self._step,
            'trade_closed': False,
        }

        if action == 1 and self._position is None:  # ENTER_LONG
            self._position = self._price
        elif action == 6 and self._position is not None:  # EXIT_ALL
            reward = (self._price - self._position) / self._position
            self._position = None
            info['trade_closed'] = True

        done = self._step >= 100
        truncated = False

        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        obs = np.random.randn(26).astype(np.float32) * 0.1
        if self._position is not None:
            obs[-6:] = np.array([1, (self._price - self._position) / self._position, 0.5, 0.01, 0.02, 1])
        return obs

    def get_action_mask(self):
        mask = np.ones(7, dtype=bool)
        if self._position is None:
            mask[[2, 3, 4, 5, 6]] = False
        else:
            mask[1] = False
        return mask


class TestFeatureExtractor:
    """Test feature extraction module."""

    def test_feature_extractor_creation(self):
        from src.rl.features import FeatureExtractor, FeatureConfig

        config = FeatureConfig()
        extractor = FeatureExtractor(config)

        assert extractor.get_state_dim() > 0
        assert len(extractor.get_feature_names()) > 0

    def test_feature_calculation(self):
        from src.rl.features import FeatureExtractor, FeatureConfig

        # Create synthetic OHLCV
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='1min')
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.001))

        ohlcv = pd.DataFrame({
            'open': prices * (1 + np.random.randn(n) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(n) * 0.002)),
            'low': prices * (1 - np.abs(np.random.randn(n) * 0.002)),
            'close': prices,
            'volume': np.random.exponential(1000, n),
        }, index=dates)

        extractor = FeatureExtractor(FeatureConfig())
        features = extractor.calculate_features(ohlcv)

        assert len(features) > 0
        assert features.shape[1] > 0
        assert not features.isna().all().all()

    def test_normalizer(self):
        from src.rl.features import FeatureNormalizer

        normalizer = FeatureNormalizer(n_features=10)

        # Update with data
        for _ in range(10):
            normalizer.update(np.random.randn(32, 10))

        # Normalize
        data = np.random.randn(5, 10)
        normalized = normalizer.normalize(data)

        assert normalized.shape == data.shape
        assert np.abs(normalized.mean()) < 2  # Should be roughly centered


class TestRewardCalculator:
    """Test reward calculation module."""

    def test_basic_reward(self):
        from src.rl.rewards import RewardCalculator, RewardConfig, RewardType

        config = RewardConfig(reward_type=RewardType.BASIC_PNL)
        calculator = RewardCalculator(config)

        reward = calculator.calculate(
            action=0,
            prev_price=100,
            current_price=101,
        )

        assert isinstance(reward, float)

    def test_hybrid_reward(self):
        from src.rl.rewards import RewardCalculator, RewardConfig, RewardType

        config = RewardConfig(reward_type=RewardType.HYBRID)
        calculator = RewardCalculator(config)

        # Winning trade
        from dataclasses import dataclass

        @dataclass
        class MockTradeResult:
            return_pct: float = 0.05
            exit_reason: str = "manual"

        reward = calculator.calculate(
            action=6,
            prev_price=100,
            current_price=105,
            trade_result=MockTradeResult(),
        )

        assert reward > 0  # Winning trade should have positive reward


class TestEnvironment:
    """Test trading environment."""

    def test_environment_creation(self):
        env = MockTradingEnvironment()
        assert env.observation_space.shape == (26,)
        assert env.action_space.n == 7

    def test_environment_reset(self):
        env = MockTradingEnvironment()
        obs, info = env.reset()

        assert obs.shape == (26,)
        assert 'symbol' in info

    def test_environment_step(self):
        env = MockTradingEnvironment()
        env.reset()

        for _ in range(10):
            mask = env.get_action_mask()
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)

            obs, reward, done, truncated, info = env.step(action)

            assert obs.shape == (26,)
            assert isinstance(reward, float)
            assert isinstance(done, bool)

    def test_action_masking(self):
        env = MockTradingEnvironment()
        env.reset()

        # No position - can only WAIT (0) or ENTER (1)
        mask = env.get_action_mask()
        assert mask[0] == True  # WAIT
        assert mask[1] == True  # ENTER_LONG
        assert mask[2] == False  # HOLD

        # Enter position
        env.step(1)

        # With position - can't enter again
        mask = env.get_action_mask()
        assert mask[1] == False  # ENTER_LONG disabled


class TestAgent:
    """Test PPO agent wrapper."""

    def test_agent_prediction(self):
        from src.rl.agent import PPOAgent, AgentConfig

        env = MockTradingEnvironment()
        config = AgentConfig(
            n_steps=64,
            batch_size=32,
        )

        agent = PPOAgent(env, config)

        obs, _ = env.reset()
        mask = env.get_action_mask()

        action, info = agent.predict(obs, action_mask=mask)

        assert 0 <= action < 7
        assert 'value' in info

    def test_agent_training(self):
        from src.rl.agent import PPOAgent, AgentConfig

        env = MockTradingEnvironment()
        config = AgentConfig(
            n_steps=64,
            batch_size=32,
            total_timesteps=500,
        )

        agent = PPOAgent(env, config)
        agent.train(total_timesteps=500)

        # Verify model can still predict after training
        obs, _ = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        assert 0 <= action < 7

    def test_agent_save_load(self):
        from src.rl.agent import PPOAgent, AgentConfig

        env = MockTradingEnvironment()
        config = AgentConfig(
            n_steps=64,
            batch_size=32,
        )

        agent = PPOAgent(env, config)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model")
            agent.save(path)

            assert os.path.exists(path + ".zip")

            # Load
            agent.load(path)

            # Verify predictions work
            obs, _ = env.reset()
            action, _ = agent.predict(obs)
            assert 0 <= action < 7


class TestAttention:
    """Test attention modules."""

    def test_temporal_attention(self):
        from src.rl.attention import TemporalAttention

        batch_size = 4
        seq_len = 60
        d_model = 128

        attn = TemporalAttention(d_model, n_heads=8)
        x = torch.randn(batch_size, seq_len, d_model)

        pooled, weights = attn(x)

        assert pooled.shape == (batch_size, d_model)
        assert weights.shape == (batch_size, seq_len)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_market_attention(self):
        from src.rl.attention import MarketAttention, AttentionConfig

        config = AttentionConfig()
        attn = MarketAttention(config)

        batch_size = 4
        micro = torch.randn(batch_size, config.seq_micro, config.d_micro)
        meso = torch.randn(batch_size, config.seq_meso, config.d_meso)
        macro = torch.randn(batch_size, config.seq_macro, config.d_macro)

        fused, weights = attn(micro, meso, macro)

        assert fused.shape == (batch_size, config.d_model)
        assert 'micro' in weights
        assert 'meso' in weights
        assert 'macro' in weights


class TestContinualLearning:
    """Test continuous learning components."""

    def test_prioritized_buffer(self):
        from src.rl.continual import PrioritizedReplayBuffer

        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add experiences
        for i in range(50):
            buffer.add(
                state=np.random.randn(20).astype(np.float32),
                action=np.random.randint(7),
                reward=np.random.randn(),
                next_state=np.random.randn(20).astype(np.float32),
                done=False,
            )

        assert len(buffer) == 50

        # Sample
        experiences, indices, weights = buffer.sample(16)
        assert len(experiences) == 16
        assert len(indices) == 16
        assert len(weights) == 16

    def test_ewc(self):
        from src.rl.continual import ElasticWeightConsolidation

        model = torch.nn.Sequential(
            torch.nn.Linear(20, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 7),
        )

        ewc = ElasticWeightConsolidation(model, ewc_lambda=1000)

        # Initial penalty should be 0
        penalty = ewc.penalty()
        assert penalty.item() == 0.0


class TestRegimeDetection:
    """Test regime detection module."""

    def test_regime_detector(self):
        from src.rl.regime import RegimeDetector, RegimeConfig

        config = RegimeConfig()
        detector = RegimeDetector(config)

        # Generate synthetic data
        n = 200
        dates = pd.date_range('2024-01-01', periods=n, freq='h')
        prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(n) * 0.01 + 0.001)),
            index=dates
        )

        regime, confidence = detector.detect(prices, timestamp=dates[-1])

        assert regime is not None
        assert 0 <= confidence <= 1

    def test_performance_tracker(self):
        from src.rl.regime import RegimePerformanceTracker, RegimeType

        tracker = RegimePerformanceTracker()

        # Record trades
        for _ in range(30):
            tracker.record_trade(
                RegimeType.TRENDING_UP,
                np.random.randn() * 0.03 + 0.01
            )

        perf = tracker.get_regime_performance(RegimeType.TRENDING_UP)
        assert perf['n_trades'] == 30
        assert 'win_rate' in perf


class TestEvaluation:
    """Test evaluation utilities."""

    def test_metrics_calculation(self):
        from src.rl.evaluation import MetricsCalculator, TradeRecord

        calculator = MetricsCalculator()

        trades = []
        t = datetime(2024, 1, 1)

        for _ in range(50):
            duration = np.random.randint(30, 480)
            return_pct = np.random.randn() * 0.03 + 0.005

            trades.append(TradeRecord(
                entry_time=t,
                exit_time=t + timedelta(minutes=duration),
                entry_price=100,
                exit_price=100 * (1 + return_pct),
                return_pct=return_pct,
                size=1.0,
                exit_reason='manual',
            ))
            t += timedelta(hours=12)

        metrics = calculator.calculate(trades, total_duration_days=50)

        assert metrics.n_trades == 50
        assert 0 <= metrics.win_rate <= 1
        assert metrics.profit_factor >= 0

    def test_walk_forward_validator(self):
        from src.rl.evaluation import WalkForwardValidator

        validator = WalkForwardValidator(
            train_period_days=30,
            test_period_days=10,
            step_days=10,
        )

        folds = validator.get_folds(
            datetime(2024, 1, 1),
            datetime(2024, 4, 1),
        )

        assert len(folds) > 0
        for fold in folds:
            assert fold.train_end < fold.test_start


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running RL Trading System Tests")
    print("=" * 60)

    test_classes = [
        TestFeatureExtractor,
        TestRewardCalculator,
        TestEnvironment,
        TestAgent,
        TestAttention,
        TestContinualLearning,
        TestRegimeDetection,
        TestEvaluation,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)

        test_instance = test_class()

        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print(f"  [PASS] {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  [FAIL] {method_name}: {e}")
                    failed_tests.append((test_class.__name__, method_name, str(e)))

    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")

    if failed_tests:
        print("\nFailed tests:")
        for cls, method, error in failed_tests:
            print(f"  {cls}.{method}: {error}")

    print("=" * 60)

    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
