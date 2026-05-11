"""
RL-Based Position Sizing Agent

Uses Deep Q-Network (DQN) to learn optimal position sizing based on:
- Spike probability from LightGBM classifier
- Market context features (volatility, momentum, etc.)
- Historical trade outcomes

The agent learns to output position sizes (0%, 25%, 50%, 75%, 100%)
that maximize risk-adjusted returns.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
import random
from datetime import datetime, timedelta
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. Install with: pip install torch")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RLConfig:
    """RL agent configuration."""
    # State space
    state_dim: int = 8  # Features per observation

    # Action space: discrete position sizes
    # 0 = 0%, 1 = 25%, 2 = 50%, 3 = 75%, 4 = 100%
    n_actions: int = 5
    position_sizes: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    # DQN hyperparameters
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.95  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    # Experience replay
    buffer_size: int = 10000
    batch_size: int = 32
    min_buffer_size: int = 100

    # Target network
    target_update_freq: int = 100
    tau: float = 0.005  # Soft update parameter

    # Training
    n_episodes: int = 500
    max_steps_per_episode: int = 50

    # Reward shaping
    spike_bonus_multiplier: float = 2.0
    missed_spike_penalty: float = -0.5
    transaction_cost_pct: float = 0.2
    drawdown_penalty_factor: float = 0.1
    holding_penalty_per_step: float = 0.001

    # Trading parameters
    base_position_size: float = 1000.0  # Base allocation per trade
    take_profit_pct: float = 10.0
    stop_loss_pct: float = 5.0
    max_hold_steps: int = 24  # Hours to hold


# =============================================================================
# Neural Network Architecture
# =============================================================================

if HAS_TORCH:
    class DQN(nn.Module):
        """
        Dueling DQN architecture for position sizing.

        Separates value and advantage streams for more stable learning.
        """

        def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
            super(DQN, self).__init__()

            # Shared feature extraction
            self.feature_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )

            # Value stream (state value)
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            # Advantage stream (action advantages)
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.feature_layer(x)

            value = self.value_stream(features)
            advantages = self.advantage_stream(features)

            # Combine value and advantages (dueling architecture)
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

            return q_values


# =============================================================================
# Experience Replay Buffer
# =============================================================================

@dataclass
class Experience:
    """Single experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Prioritized experience replay buffer.

    Prioritizes experiences with higher TD errors and spike events.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, experience: Experience, priority: float = 1.0):
        """Add experience with priority."""
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch weighted by priorities."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        return [self.buffer[i] for i in indices]

    def update_priority(self, index: int, priority: float):
        """Update priority for an experience."""
        if index < len(self.priorities):
            self.priorities[index] = priority

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# Trading Environment
# =============================================================================

@dataclass
class TradingState:
    """Current trading state."""
    spike_probability: float
    volatility: float
    momentum: float
    volume_ratio: float
    rsi_proxy: float
    max_return_4h: float  # Forward return (for training only)
    max_return_24h: float  # Forward return (for training only)
    is_spike: bool  # Whether this was actually a spike


class TradingEnvironment:
    """
    Trading environment for RL training.

    Simulates trading based on spike predictions and actual outcomes.
    """

    def __init__(self, data: pd.DataFrame, config: RLConfig):
        self.data = data.reset_index(drop=True)
        self.config = config
        self.current_idx = 0
        self.position = None
        self.entry_price = 0.0
        self.entry_step = 0
        self.total_pnl = 0.0
        self.peak_equity = 0.0
        self.max_drawdown = 0.0

    def reset(self) -> np.ndarray:
        """Reset environment to start of episode."""
        # Random starting point
        max_start = max(0, len(self.data) - self.config.max_steps_per_episode - 1)
        self.current_idx = random.randint(0, max_start) if max_start > 0 else 0
        self.position = None
        self.entry_price = 0.0
        self.entry_step = 0
        self.total_pnl = 0.0
        self.peak_equity = self.config.base_position_size
        self.max_drawdown = 0.0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        if self.current_idx >= len(self.data):
            return np.zeros(self.config.state_dim)

        row = self.data.iloc[self.current_idx]

        # Normalize features
        state = np.array([
            row.get('y_pred_proba', 0.5),  # Spike probability
            min(row.get('volatility_4h', 0.5), 5.0) / 5.0,  # Normalized volatility
            np.clip(row.get('momentum_4h', 0.0), -10, 10) / 10.0,  # Normalized momentum
            min(row.get('volume_ratio', 1.0), 3.0) / 3.0,  # Normalized volume
            row.get('rsi_proxy', 50.0) / 100.0,  # RSI proxy
            float(self.position is not None),  # Currently in position
            (self.current_idx - self.entry_step) / self.config.max_hold_steps if self.position else 0,  # Hold duration
            self.max_drawdown / 10.0,  # Current drawdown impact
        ], dtype=np.float32)

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info.

        Actions:
        0 = No position (or close if holding)
        1-4 = Enter/maintain position with 25%, 50%, 75%, 100% size
        """
        if self.current_idx >= len(self.data) - 1:
            return self._get_state(), 0.0, True, {}

        row = self.data.iloc[self.current_idx]
        position_size_pct = self.config.position_sizes[action]

        reward = 0.0
        info = {}

        # Get actual outcome (for reward computation)
        actual_return_4h = row.get('max_return_4h', 0.0)
        actual_return_24h = row.get('max_return_24h', 0.0)
        is_spike = row.get('y_true', 0) == 1 or actual_return_24h >= 10.0

        # Handle position logic
        if self.position is not None:
            # Check exit conditions
            hold_time = self.current_idx - self.entry_step
            current_return = actual_return_4h  # Simplified: use 4h return as proxy

            should_exit = False
            exit_reason = ""

            if current_return >= self.config.take_profit_pct:
                should_exit = True
                exit_reason = "take_profit"
            elif current_return <= -self.config.stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"
            elif hold_time >= self.config.max_hold_steps:
                should_exit = True
                exit_reason = "time_limit"
            elif action == 0:  # Agent chooses to exit
                should_exit = True
                exit_reason = "agent_exit"

            if should_exit:
                # Calculate PnL
                pnl_pct = current_return - self.config.transaction_cost_pct
                pnl_usd = self.position * (pnl_pct / 100.0)

                # Compute reward
                reward = self._compute_reward(pnl_pct, is_spike, hold_time, exit_reason)

                self.total_pnl += pnl_usd
                self.position = None

                info['exit_reason'] = exit_reason
                info['pnl_pct'] = pnl_pct

        else:
            # Not in position - consider entry
            if action > 0:  # Enter position
                position_value = self.config.base_position_size * position_size_pct
                self.position = position_value
                self.entry_price = row.get('price_at_signal', 1.0)
                self.entry_step = self.current_idx

                # Entry cost
                reward -= self.config.transaction_cost_pct / 100.0

                info['entry'] = True
                info['position_size'] = position_value
            else:
                # No action when not in position
                # Penalize missing spikes
                if is_spike:
                    reward += self.config.missed_spike_penalty
                    info['missed_spike'] = True

        # Update drawdown tracking
        current_equity = self.config.base_position_size + self.total_pnl
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Move to next step
        self.current_idx += 1
        done = self.current_idx >= len(self.data) - 1

        next_state = self._get_state()

        return next_state, reward, done, info

    def _compute_reward(
        self,
        pnl_pct: float,
        is_spike: bool,
        hold_time: int,
        exit_reason: str,
    ) -> float:
        """
        Compute reward with careful shaping.

        Key principles:
        1. Base reward on PnL
        2. Bonus for catching spikes
        3. Penalty for drawdowns
        4. Slight penalty for holding (encourages decisive action)
        """
        # Base reward: scaled PnL
        base_reward = pnl_pct / 10.0  # Scale to [-1, 1] range roughly

        # Spike bonus
        if is_spike and pnl_pct > 0:
            spike_bonus = abs(base_reward) * self.config.spike_bonus_multiplier
        elif is_spike and pnl_pct <= 0:
            spike_bonus = -0.5  # Penalty for losing on spike
        else:
            spike_bonus = 0.0

        # Drawdown penalty
        drawdown_penalty = -self.max_drawdown * self.config.drawdown_penalty_factor

        # Hold time penalty (encourages decisive action)
        hold_penalty = -hold_time * self.config.holding_penalty_per_step

        # Exit reason bonus
        exit_bonus = 0.0
        if exit_reason == "take_profit":
            exit_bonus = 0.5  # Reward for hitting TP
        elif exit_reason == "stop_loss":
            exit_bonus = -0.2  # Small penalty for SL (already penalized by PnL)

        total_reward = base_reward + spike_bonus + drawdown_penalty + hold_penalty + exit_bonus

        return total_reward


# =============================================================================
# DQN Agent
# =============================================================================

class PositionSizingAgent:
    """
    DQN agent for position sizing.

    Learns to select optimal position sizes based on spike probabilities
    and market context.
    """

    def __init__(self, config: RLConfig):
        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = DQN(
            config.state_dim,
            config.n_actions,
            config.hidden_dim
        ).to(self.device)

        self.target_net = DQN(
            config.state_dim,
            config.n_actions,
            config.hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.learning_rate,
        )

        # Experience replay
        self.memory = ReplayBuffer(config.buffer_size)

        # Exploration
        self.epsilon = config.epsilon_start

        # Training stats
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.config.n_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def get_position_size(self, state: np.ndarray) -> float:
        """Get position size (0-1) for given state."""
        action = self.select_action(state, training=False)
        return self.config.position_sizes[action]

    def store_experience(self, experience: Experience, is_spike: bool = False):
        """Store experience in replay buffer."""
        # Higher priority for spike events
        priority = 2.0 if is_spike else 1.0
        self.memory.push(experience, priority)

    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if len(self.memory) < self.config.min_buffer_size:
            return None

        # Sample batch
        batch = self.memory.sample(self.config.batch_size)

        # Prepare tensors
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([float(e.done) for e in batch]).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values (Double DQN)
        with torch.no_grad():
            # Select actions using policy network
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Evaluate using target network
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        # Loss
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        self.training_step += 1
        if self.training_step % self.config.target_update_freq == 0:
            self._soft_update_target()

        return loss.item()

    def _soft_update_target(self):
        """Soft update target network parameters."""
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * policy_param.data +
                (1.0 - self.config.tau) * target_param.data
            )

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_end)
        self.training_step = checkpoint.get('training_step', 0)
        logger.info(f"Model loaded from {path}")


# =============================================================================
# Training Pipeline
# =============================================================================

class RLTrainer:
    """
    Training pipeline for RL position sizing agent.
    """

    def __init__(
        self,
        features_csv: str,
        predictions_csv: str,
        config: Optional[RLConfig] = None,
    ):
        self.config = config or RLConfig()

        # Load data
        features = pd.read_csv(features_csv, parse_dates=['timestamp'])
        predictions = pd.read_csv(predictions_csv, parse_dates=['timestamp'])

        # Merge features and predictions
        self.data = predictions.merge(
            features[['timestamp', 'symbol', 'price_at_signal', 'volatility_4h',
                     'momentum_4h', 'volume_ratio', 'rsi_proxy',
                     'max_return_4h', 'max_return_24h']],
            on=['timestamp', 'symbol'],
            how='left'
        )

        # Fill NaN values
        self.data = self.data.fillna({
            'volatility_4h': 0.5,
            'momentum_4h': 0.0,
            'volume_ratio': 1.0,
            'rsi_proxy': 50.0,
            'max_return_4h': 0.0,
            'max_return_24h': 0.0,
        })

        logger.info(f"Loaded {len(self.data)} samples for training")

        # Split data
        split_idx = int(len(self.data) * 0.7)
        self.train_data = self.data.iloc[:split_idx]
        self.test_data = self.data.iloc[split_idx:]

        logger.info(f"Train: {len(self.train_data)}, Test: {len(self.test_data)}")

        # Initialize agent and environment
        self.agent = PositionSizingAgent(self.config)

    def train(self, verbose: bool = True) -> Dict:
        """
        Train the RL agent.

        Returns:
            Dict with training metrics
        """
        logger.info("Starting RL training...")

        env = TradingEnvironment(self.train_data, self.config)

        episode_rewards = []
        episode_pnls = []
        losses = []

        for episode in range(self.config.n_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_loss = []

            for step in range(self.config.max_steps_per_episode):
                # Select action
                action = self.agent.select_action(state)

                # Take step
                next_state, reward, done, info = env.step(action)

                # Store experience
                is_spike = info.get('missed_spike', False) or \
                          (info.get('pnl_pct', 0) > 5)

                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
                self.agent.store_experience(experience, is_spike)

                # Train
                loss = self.agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)

                episode_reward += reward
                state = next_state

                if done:
                    break

            # Decay exploration
            self.agent.decay_epsilon()

            episode_rewards.append(episode_reward)
            episode_pnls.append(env.total_pnl)
            if episode_loss:
                losses.append(np.mean(episode_loss))

            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_pnl = np.mean(episode_pnls[-50:])
                avg_loss = np.mean(losses[-50:]) if losses else 0
                logger.info(
                    f"Episode {episode+1}/{self.config.n_episodes} | "
                    f"Avg Reward: {avg_reward:.3f} | "
                    f"Avg PnL: ${avg_pnl:.2f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )

        return {
            'episode_rewards': episode_rewards,
            'episode_pnls': episode_pnls,
            'losses': losses,
            'final_epsilon': self.agent.epsilon,
        }

    def evaluate(self, use_test_data: bool = True) -> Dict:
        """
        Evaluate trained agent.

        Returns:
            Dict with evaluation metrics
        """
        data = self.test_data if use_test_data else self.train_data
        env = TradingEnvironment(data, self.config)

        state = env.reset()
        env.current_idx = 0  # Start from beginning for evaluation

        total_reward = 0.0
        trades = []
        position_sizes = []

        while env.current_idx < len(data) - 1:
            action = self.agent.select_action(state, training=False)
            position_size = self.config.position_sizes[action]
            position_sizes.append(position_size)

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if 'exit_reason' in info:
                trades.append({
                    'pnl_pct': info.get('pnl_pct', 0),
                    'exit_reason': info['exit_reason'],
                    'position_size': position_size,
                })

            state = next_state
            if done:
                break

        # Calculate metrics
        results = {
            'total_reward': total_reward,
            'total_pnl': env.total_pnl,
            'n_trades': len(trades),
            'max_drawdown': env.max_drawdown,
            'avg_position_size': np.mean(position_sizes) if position_sizes else 0,
        }

        if trades:
            pnls = [t['pnl_pct'] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses_pnl = [p for p in pnls if p <= 0]

            results['win_rate'] = len(wins) / len(pnls) * 100
            results['avg_win'] = np.mean(wins) if wins else 0
            results['avg_loss'] = np.mean(losses_pnl) if losses_pnl else 0
            results['profit_factor'] = (
                sum(wins) / abs(sum(losses_pnl)) if losses_pnl and sum(losses_pnl) != 0
                else float('inf')
            )

            # Exit reasons
            reasons = [t['exit_reason'] for t in trades]
            results['exits_tp'] = reasons.count('take_profit')
            results['exits_sl'] = reasons.count('stop_loss')
            results['exits_time'] = reasons.count('time_limit')
            results['exits_agent'] = reasons.count('agent_exit')

        return results

    def compare_with_baseline(self) -> Dict:
        """
        Compare RL agent performance with fixed-size baseline.

        Returns:
            Dict with comparison metrics
        """
        # RL agent evaluation
        rl_results = self.evaluate(use_test_data=True)

        # Baseline: fixed 50% position size for all signals above threshold
        baseline_pnl = 0.0
        baseline_trades = 0
        baseline_wins = 0

        for _, row in self.test_data.iterrows():
            if row['y_pred_proba'] >= 0.3:  # Same threshold as current system
                pnl_pct = row['max_return_4h'] - self.config.transaction_cost_pct
                pnl_usd = self.config.base_position_size * 0.5 * (pnl_pct / 100)
                baseline_pnl += pnl_usd
                baseline_trades += 1
                if pnl_pct > 0:
                    baseline_wins += 1

        baseline_results = {
            'total_pnl': baseline_pnl,
            'n_trades': baseline_trades,
            'win_rate': (baseline_wins / baseline_trades * 100) if baseline_trades > 0 else 0,
        }

        return {
            'rl': rl_results,
            'baseline': baseline_results,
            'improvement_pct': (
                (rl_results['total_pnl'] - baseline_results['total_pnl']) /
                abs(baseline_results['total_pnl']) * 100
                if baseline_results['total_pnl'] != 0 else 0
            ),
        }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Train and evaluate RL position sizing agent."""
    print("=" * 70)
    print("RL POSITION SIZING AGENT")
    print("=" * 70)

    if not HAS_TORCH:
        print("\nERROR: PyTorch not installed. Install with: pip install torch")
        return

    # Configuration
    config = RLConfig(
        n_episodes=300,
        max_steps_per_episode=50,
        hidden_dim=64,
        learning_rate=1e-3,
        gamma=0.95,
        epsilon_decay=0.99,
        buffer_size=5000,
        batch_size=32,
    )

    # Initialize trainer
    trainer = RLTrainer(
        features_csv="whale_features.csv",
        predictions_csv="spike_predictions.csv",
        config=config,
    )

    # Train
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)
    train_metrics = trainer.train(verbose=True)

    # Evaluate
    print("\n" + "-" * 70)
    print("EVALUATION")
    print("-" * 70)
    eval_results = trainer.evaluate()

    print(f"\nTest Set Results:")
    print(f"  Total PnL: ${eval_results['total_pnl']:+,.2f}")
    print(f"  Trades: {eval_results['n_trades']}")
    print(f"  Win Rate: {eval_results.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor: {eval_results.get('profit_factor', 0):.2f}")
    print(f"  Max Drawdown: {eval_results['max_drawdown']*100:.1f}%")
    print(f"  Avg Position Size: {eval_results['avg_position_size']*100:.1f}%")

    if 'exits_tp' in eval_results:
        print(f"\n  Exit Breakdown:")
        print(f"    Take Profit: {eval_results['exits_tp']}")
        print(f"    Stop Loss: {eval_results['exits_sl']}")
        print(f"    Time Limit: {eval_results['exits_time']}")
        print(f"    Agent Exit: {eval_results['exits_agent']}")

    # Compare with baseline
    print("\n" + "-" * 70)
    print("COMPARISON WITH BASELINE")
    print("-" * 70)
    comparison = trainer.compare_with_baseline()

    print(f"\nRL Agent:     ${comparison['rl']['total_pnl']:+,.2f} ({comparison['rl']['n_trades']} trades)")
    print(f"Baseline:     ${comparison['baseline']['total_pnl']:+,.2f} ({comparison['baseline']['n_trades']} trades)")
    print(f"Improvement:  {comparison['improvement_pct']:+.1f}%")

    # Save model
    trainer.agent.save("models/rl_position_sizer.pt")

    return trainer, eval_results, comparison


if __name__ == "__main__":
    main()
