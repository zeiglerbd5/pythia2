"""
RL Position Sizing Agent v2

Architectural revisions based on v1 results:
1. Lower take-profit targets (3-5% instead of 10%)
2. Trailing stop implementation
3. Continuous action space for position sizing (PPO-like)
4. Improved reward shaping focusing on actual achievable returns
5. Better state representation with realized PnL tracking
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
    from torch.distributions import Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed.")


# =============================================================================
# Configuration v2
# =============================================================================

@dataclass
class RLConfigV2:
    """RL agent configuration v2 with revised parameters."""
    # State space (expanded)
    state_dim: int = 12

    # Action space: continuous position size [0, 1]
    # Plus discrete exit decision
    use_continuous_actions: bool = True

    # Network architecture
    hidden_dim: int = 128
    n_layers: int = 3

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    n_epochs: int = 10
    n_steps_per_update: int = 256
    batch_size: int = 64
    n_episodes: int = 1000

    # Experience collection
    buffer_size: int = 2048

    # Reward shaping (revised)
    # Key change: Reward smaller, achievable gains
    small_profit_threshold: float = 1.0  # 1% profit
    medium_profit_threshold: float = 3.0  # 3% profit
    large_profit_threshold: float = 5.0  # 5% profit

    small_profit_bonus: float = 0.5
    medium_profit_bonus: float = 1.5
    large_profit_bonus: float = 3.0

    loss_penalty_multiplier: float = 1.5
    transaction_cost_pct: float = 0.2
    hold_cost_per_step: float = 0.001
    opportunity_cost: float = 0.1  # Penalty for not being in profitable position

    # Trading parameters (revised for realistic exits)
    base_position_size: float = 1000.0
    take_profit_pct: float = 5.0  # Lowered from 10%
    stop_loss_pct: float = 3.0  # Tighter stop
    trailing_stop_pct: float = 2.0  # Trailing stop activation
    max_hold_steps: int = 12  # Shorter hold (12 hours instead of 24)


# =============================================================================
# Actor-Critic Network (PPO-style)
# =============================================================================

if HAS_TORCH:
    class ActorCritic(nn.Module):
        """
        Actor-Critic network for PPO.

        Actor: outputs mean and std for position size distribution
        Critic: estimates state value
        """

        def __init__(self, state_dim: int, hidden_dim: int = 128, n_layers: int = 3):
            super(ActorCritic, self).__init__()

            # Shared feature extractor
            layers = []
            in_dim = state_dim
            for i in range(n_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ])
                in_dim = hidden_dim

            self.shared = nn.Sequential(*layers)

            # Actor head (position size)
            self.actor_mean = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),  # Output in [0, 1]
            )

            self.actor_log_std = nn.Parameter(torch.zeros(1))

            # Critic head (value)
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Returns:
                action_mean: Mean position size [0, 1]
                action_std: Standard deviation for exploration
                value: State value estimate
            """
            features = self.shared(x)

            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
            value = self.critic(features)

            return action_mean, action_std, value

        def get_action(self, state: torch.Tensor, deterministic: bool = False):
            """Sample action from policy."""
            action_mean, action_std, value = self.forward(state)

            if deterministic:
                return action_mean, value, None

            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action = torch.clamp(action, 0, 1)  # Ensure valid range
            log_prob = dist.log_prob(action)

            return action, value, log_prob

        def evaluate_action(self, state: torch.Tensor, action: torch.Tensor):
            """Evaluate action for PPO update."""
            action_mean, action_std, value = self.forward(state)

            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            return log_prob, entropy, value


# =============================================================================
# Enhanced Trading Environment
# =============================================================================

@dataclass
class Position:
    """Active trading position."""
    size: float
    entry_price: float
    entry_step: int
    peak_price: float  # For trailing stop
    trailing_stop_triggered: bool = False


class TradingEnvironmentV2:
    """
    Enhanced trading environment with:
    - Trailing stops
    - Multiple take-profit levels
    - Realistic reward shaping
    """

    def __init__(self, data: pd.DataFrame, config: RLConfigV2):
        self.data = data.reset_index(drop=True)
        self.config = config
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment."""
        max_start = max(0, len(self.data) - 100)
        self.current_idx = random.randint(0, max_start) if max_start > 0 else 0
        self.position: Optional[Position] = None
        self.total_pnl = 0.0
        self.realized_pnl = 0.0
        self.peak_equity = self.config.base_position_size
        self.max_drawdown = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.consecutive_losses = 0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Build enhanced state vector.

        Includes:
        - Prediction features
        - Position status
        - Performance metrics
        - Risk indicators
        """
        if self.current_idx >= len(self.data):
            return np.zeros(self.config.state_dim)

        row = self.data.iloc[self.current_idx]

        # Unrealized PnL if in position
        if self.position:
            current_price = row.get('price_at_signal', self.position.entry_price)
            unrealized_pnl = (current_price - self.position.entry_price) / self.position.entry_price
            hold_time = (self.current_idx - self.position.entry_step) / self.config.max_hold_steps
            distance_to_peak = (self.position.peak_price - current_price) / self.position.peak_price
        else:
            unrealized_pnl = 0.0
            hold_time = 0.0
            distance_to_peak = 0.0

        state = np.array([
            # Prediction features
            row.get('y_pred_proba', 0.5),
            min(row.get('volatility_4h', 0.5), 5.0) / 5.0,
            np.clip(row.get('momentum_4h', 0.0), -10, 10) / 10.0,
            min(row.get('volume_ratio', 1.0), 3.0) / 3.0,
            row.get('rsi_proxy', 50.0) / 100.0,

            # Position status
            float(self.position is not None),
            hold_time,
            np.clip(unrealized_pnl, -0.1, 0.1) * 10,  # Scale to [-1, 1]

            # Performance metrics
            self.realized_pnl / self.config.base_position_size,  # Normalized realized PnL
            self.max_drawdown,
            min(self.trade_count / 20.0, 1.0),  # Trade frequency
            min(self.consecutive_losses / 3.0, 1.0),  # Loss streak indicator
        ], dtype=np.float32)

        return state

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action.

        action: Position size [0, 1]
        - 0 means no position / exit
        - >0 means enter/hold with that size
        """
        if self.current_idx >= len(self.data) - 1:
            return self._get_state(), 0.0, True, {}

        row = self.data.iloc[self.current_idx]
        reward = 0.0
        info = {}

        # Get actual forward returns from data
        actual_return_4h = row.get('max_return_4h', 0.0)

        if self.position is not None:
            # === IN POSITION ===
            current_price = row.get('price_at_signal', self.position.entry_price)
            pnl_pct = ((current_price - self.position.entry_price) / self.position.entry_price) * 100

            # Update peak price for trailing stop
            if current_price > self.position.peak_price:
                self.position.peak_price = current_price

            # Check exit conditions
            should_exit = False
            exit_reason = ""

            # Take profit levels
            if pnl_pct >= self.config.take_profit_pct:
                should_exit = True
                exit_reason = "take_profit"
            # Stop loss
            elif pnl_pct <= -self.config.stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"
            # Trailing stop
            elif pnl_pct > self.config.trailing_stop_pct:
                pullback = ((self.position.peak_price - current_price) /
                           self.position.peak_price) * 100
                if pullback >= self.config.trailing_stop_pct:
                    should_exit = True
                    exit_reason = "trailing_stop"
                    self.position.trailing_stop_triggered = True
            # Time limit
            elif (self.current_idx - self.position.entry_step) >= self.config.max_hold_steps:
                should_exit = True
                exit_reason = "time_limit"
            # Agent chooses to exit
            elif action < 0.1:
                should_exit = True
                exit_reason = "agent_exit"

            if should_exit:
                # Calculate final PnL
                net_pnl_pct = pnl_pct - self.config.transaction_cost_pct
                pnl_usd = self.position.size * (net_pnl_pct / 100.0)

                # Compute reward with tiered bonuses
                reward = self._compute_exit_reward(net_pnl_pct, exit_reason)

                # Update stats
                self.realized_pnl += pnl_usd
                self.total_pnl += pnl_usd
                self.trade_count += 1

                if net_pnl_pct > 0:
                    self.win_count += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1

                self.position = None

                info['exit'] = True
                info['exit_reason'] = exit_reason
                info['pnl_pct'] = net_pnl_pct
            else:
                # Still holding - apply hold cost
                reward -= self.config.hold_cost_per_step

        else:
            # === NOT IN POSITION ===
            if action >= 0.2:  # Enter if action > 20%
                position_value = self.config.base_position_size * action
                entry_price = row.get('price_at_signal', 1.0)

                self.position = Position(
                    size=position_value,
                    entry_price=entry_price,
                    entry_step=self.current_idx,
                    peak_price=entry_price,
                )

                # Entry cost
                reward -= self.config.transaction_cost_pct / 100.0
                info['entry'] = True
                info['position_size'] = action
            else:
                # Chose not to enter
                # Check if we missed a good opportunity
                if actual_return_4h >= self.config.medium_profit_threshold:
                    reward -= self.config.opportunity_cost
                    info['missed_opportunity'] = actual_return_4h

        # Update equity tracking
        current_equity = self.config.base_position_size + self.total_pnl
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

        self.current_idx += 1
        done = self.current_idx >= len(self.data) - 1

        return self._get_state(), reward, done, info

    def _compute_exit_reward(self, pnl_pct: float, exit_reason: str) -> float:
        """
        Compute reward for trade exit.

        Key changes from v1:
        - Tiered rewards for different profit levels
        - Bonus for trailing stop exits (captured gains)
        - Reduced penalty for small losses
        """
        # Base reward scaled by PnL
        base_reward = pnl_pct / 5.0  # Scale: 5% gain = 1.0 reward

        bonus = 0.0

        if pnl_pct > 0:
            # Profit bonuses
            if pnl_pct >= self.config.large_profit_threshold:
                bonus = self.config.large_profit_bonus
            elif pnl_pct >= self.config.medium_profit_threshold:
                bonus = self.config.medium_profit_bonus
            elif pnl_pct >= self.config.small_profit_threshold:
                bonus = self.config.small_profit_bonus

            # Extra bonus for trailing stop (locked in gains)
            if exit_reason == "trailing_stop":
                bonus += 0.5

            # Bonus for take profit hit
            if exit_reason == "take_profit":
                bonus += 0.3

        else:
            # Loss penalties (less harsh for small losses)
            if pnl_pct > -1.0:
                # Small loss: minimal penalty
                bonus = -0.2
            elif pnl_pct > -2.0:
                # Medium loss
                bonus = -0.5
            else:
                # Large loss
                bonus = -pnl_pct / 5.0 * self.config.loss_penalty_multiplier

            # Extra penalty for stop loss (didn't manage risk well)
            if exit_reason == "stop_loss":
                bonus -= 0.3

        # Consecutive loss penalty
        if self.consecutive_losses > 2:
            bonus -= 0.2 * (self.consecutive_losses - 2)

        return base_reward + bonus


# =============================================================================
# PPO Agent
# =============================================================================

class PPOAgent:
    """
    Proximal Policy Optimization agent for position sizing.
    """

    def __init__(self, config: RLConfigV2):
        if not HAS_TORCH:
            raise ImportError("PyTorch required.")

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network
        self.network = ActorCritic(
            config.state_dim,
            config.hidden_dim,
            config.n_layers
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )

        # Buffers for PPO update
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, float]:
        """Select action and value."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, value, log_prob = self.network.get_action(state_tensor, deterministic)

        action_val = action.cpu().numpy()[0, 0]
        value_val = value.cpu().numpy()[0, 0]

        if not deterministic and log_prob is not None:
            self.states.append(state)
            self.actions.append(action_val)
            self.log_probs.append(log_prob.cpu().numpy()[0, 0])
            self.values.append(value_val)

        return action_val, value_val

    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)

    def get_position_size(self, state: np.ndarray) -> float:
        """Get position size for inference."""
        action, _ = self.select_action(state, deterministic=True)
        return action

    def update(self) -> Dict[str, float]:
        """Perform PPO update."""
        if len(self.rewards) < self.config.batch_size:
            return {}

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).unsqueeze(1).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0

        for _ in range(self.config.n_epochs):
            # Sample mini-batches
            indices = np.random.permutation(len(self.rewards))

            for start in range(0, len(indices), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, entropy, values = self.network.evaluate_action(
                    batch_states, batch_actions
                )

                # Policy loss (PPO clip)
                ratios = torch.exp(log_probs.squeeze() - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(
                    ratios,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        # Clear buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        n_updates = self.config.n_epochs * (len(indices) // self.config.batch_size + 1)
        return {
            'loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
        }

    def save(self, path: str):
        """Save model."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Model loaded from {path}")


# =============================================================================
# Training Pipeline v2
# =============================================================================

class RLTrainerV2:
    """
    Training pipeline for PPO agent.
    """

    def __init__(
        self,
        features_csv: str,
        predictions_csv: str,
        config: Optional[RLConfigV2] = None,
    ):
        self.config = config or RLConfigV2()

        # Load and merge data
        features = pd.read_csv(features_csv, parse_dates=['timestamp'])
        predictions = pd.read_csv(predictions_csv, parse_dates=['timestamp'])

        self.data = predictions.merge(
            features[['timestamp', 'symbol', 'price_at_signal', 'volatility_4h',
                     'momentum_4h', 'volume_ratio', 'rsi_proxy',
                     'max_return_4h', 'max_return_24h']],
            on=['timestamp', 'symbol'],
            how='left'
        ).fillna({
            'volatility_4h': 0.5,
            'momentum_4h': 0.0,
            'volume_ratio': 1.0,
            'rsi_proxy': 50.0,
            'max_return_4h': 0.0,
            'max_return_24h': 0.0,
        })

        logger.info(f"Loaded {len(self.data)} samples")

        # Split data
        split_idx = int(len(self.data) * 0.7)
        self.train_data = self.data.iloc[:split_idx]
        self.test_data = self.data.iloc[split_idx:]

        logger.info(f"Train: {len(self.train_data)}, Test: {len(self.test_data)}")

        # Initialize agent
        self.agent = PPOAgent(self.config)

    def train(self, verbose: bool = True) -> Dict:
        """Train the PPO agent."""
        logger.info("Starting PPO training...")

        env = TradingEnvironmentV2(self.train_data, self.config)

        episode_rewards = []
        episode_pnls = []

        for episode in range(self.config.n_episodes):
            state = env.reset()
            episode_reward = 0.0
            step = 0

            while step < 100:  # Max steps per episode
                action, value = self.agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                self.agent.store_reward(reward, done)
                episode_reward += reward

                state = next_state
                step += 1

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_pnls.append(env.total_pnl)

            # Update every N steps
            if (episode + 1) % 10 == 0:
                metrics = self.agent.update()

            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_pnl = np.mean(episode_pnls[-100:])
                logger.info(
                    f"Episode {episode+1}/{self.config.n_episodes} | "
                    f"Avg Reward: {avg_reward:.3f} | "
                    f"Avg PnL: ${avg_pnl:.2f}"
                )

        return {
            'episode_rewards': episode_rewards,
            'episode_pnls': episode_pnls,
        }

    def evaluate(self, use_test_data: bool = True) -> Dict:
        """Evaluate trained agent."""
        data = self.test_data if use_test_data else self.train_data
        env = TradingEnvironmentV2(data, self.config)

        state = env.reset()
        env.current_idx = 0

        total_reward = 0.0
        trades = []
        position_sizes = []
        exit_reasons = {'take_profit': 0, 'stop_loss': 0, 'trailing_stop': 0,
                       'time_limit': 0, 'agent_exit': 0}

        while env.current_idx < len(data) - 1:
            action = self.agent.get_position_size(state)

            if action > 0.1:
                position_sizes.append(action)

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if 'exit_reason' in info:
                trades.append({
                    'pnl_pct': info['pnl_pct'],
                    'exit_reason': info['exit_reason'],
                })
                if info['exit_reason'] in exit_reasons:
                    exit_reasons[info['exit_reason']] += 1

            state = next_state
            if done:
                break

        results = {
            'total_reward': total_reward,
            'total_pnl': env.total_pnl,
            'n_trades': len(trades),
            'max_drawdown': env.max_drawdown,
            'avg_position_size': np.mean(position_sizes) if position_sizes else 0,
            'exit_reasons': exit_reasons,
        }

        if trades:
            pnls = [t['pnl_pct'] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses_pnl = [p for p in pnls if p <= 0]

            results['win_rate'] = len(wins) / len(pnls) * 100
            results['avg_win'] = np.mean(wins) if wins else 0
            results['avg_loss'] = np.mean(losses_pnl) if losses_pnl else 0
            results['profit_factor'] = (
                sum(wins) / abs(sum(losses_pnl))
                if losses_pnl and sum(losses_pnl) != 0 else float('inf')
            )

        return results


def main():
    """Train and evaluate PPO position sizing agent."""
    print("=" * 70)
    print("RL POSITION SIZING AGENT v2 (PPO)")
    print("=" * 70)

    if not HAS_TORCH:
        print("\nERROR: PyTorch not installed.")
        return

    import os
    os.makedirs("models", exist_ok=True)

    config = RLConfigV2(
        n_episodes=800,
        hidden_dim=128,
        n_layers=3,
        learning_rate=3e-4,
        take_profit_pct=5.0,  # More achievable
        stop_loss_pct=3.0,
        trailing_stop_pct=2.0,
        max_hold_steps=12,
    )

    trainer = RLTrainerV2(
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
    print("TEST EVALUATION")
    print("-" * 70)
    eval_results = trainer.evaluate(use_test_data=True)

    print(f"\n  Total PnL: ${eval_results['total_pnl']:+,.2f}")
    print(f"  Trades: {eval_results['n_trades']}")
    print(f"  Win Rate: {eval_results.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor: {eval_results.get('profit_factor', 0):.2f}")
    print(f"  Max Drawdown: {eval_results['max_drawdown']*100:.1f}%")
    print(f"  Avg Position Size: {eval_results['avg_position_size']*100:.1f}%")

    print(f"\n  Exit Breakdown:")
    for reason, count in eval_results['exit_reasons'].items():
        print(f"    {reason}: {count}")

    # Save model
    trainer.agent.save("models/rl_position_sizer_v2.pt")

    return trainer, eval_results


if __name__ == "__main__":
    main()
