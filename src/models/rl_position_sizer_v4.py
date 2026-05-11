"""
RL Position Sizing Agent v4

Improvements over v3:
1. Trailing stops - lock in gains when price moves in our favor
2. Momentum-weighted features - momentum is strongest differentiator for winners
3. Better reward shaping - penalize premature exits, reward holding winners
4. Multi-step episodes - don't allow exit after one step
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
import random
from datetime import datetime, timedelta
from loguru import logger
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class RLConfigV4:
    """Configuration v4 with trailing stop parameters."""
    state_dim: int = 12  # Added momentum rank and trailing stop state
    n_actions: int = 5  # 0%, 25%, 50%, 75%, 100%
    position_sizes: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    hidden_dim: int = 128  # Increased capacity
    learning_rate: float = 1e-4
    gamma: float = 0.95  # Higher gamma to consider longer-term rewards
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995

    buffer_size: int = 10000
    batch_size: int = 64
    target_update: int = 50

    n_episodes: int = 800

    # Trading parameters
    base_capital: float = 1000.0
    take_profit_pct: float = 5.0
    stop_loss_pct: float = 3.0
    trailing_stop_activation: float = 2.0  # Activate trailing stop at 2% gain
    trailing_stop_distance: float = 1.0    # Trail 1% behind max
    max_hold_bars: int = 24
    min_hold_bars: int = 3  # Minimum hold time before exit allowed
    fee_pct: float = 0.2


if HAS_TORCH:
    class QNetworkV4(nn.Module):
        """Deeper Q-network with batch normalization."""

        def __init__(self, state_dim: int, n_actions: int, hidden_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.bn1 = nn.LayerNorm(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = nn.LayerNorm(hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.bn3 = nn.LayerNorm(hidden_dim // 2)
            self.out = nn.Linear(hidden_dim // 2, n_actions)

            # Initialize weights
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
            return self.out(x)


class PrioritizedReplayBuffer:
    """Prioritized replay buffer - prioritizes spike events."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done, priority: float = 1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(priority ** self.alpha)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == 0:
            return None

        probs = np.array(self.priorities)
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)),
                                   p=probs, replace=False)

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            weights,
            indices,
        )

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = (p + 1e-6) ** self.alpha

    def __len__(self):
        return len(self.buffer)


class TradingEnvV4:
    """
    Trading environment v4 with trailing stops and momentum features.
    """

    def __init__(self, data: pd.DataFrame, config: RLConfigV4):
        self.data = data.reset_index(drop=True)
        self.config = config
        self.n_samples = len(data)

        # Precompute momentum ranks
        self._compute_momentum_ranks()

    def _compute_momentum_ranks(self):
        """Rank momentum values for better state representation."""
        self.data['momentum_rank'] = self.data['momentum_4h'].rank(pct=True)

    def reset(self):
        """Reset to random starting point."""
        self.idx = random.randint(0, max(0, self.n_samples - 50))
        self.in_position = False
        self.entry_idx = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        self.total_pnl = 0.0
        self.trades = []
        self.max_price_since_entry = 0.0
        self.trailing_stop_active = False
        return self._get_state()

    def _get_state(self):
        """Get state vector with enhanced features."""
        if self.idx >= self.n_samples:
            return np.zeros(self.config.state_dim)

        row = self.data.iloc[self.idx]

        # Position-specific state
        if self.in_position:
            hold_bars = self.idx - self.entry_idx
            hold_frac = min(hold_bars / self.config.max_hold_bars, 1.0)
            # Use momentum as proxy for unrealized PnL direction
            unrealized = row.get('momentum_4h', 0) / 10.0
            trailing_active = float(self.trailing_stop_active)
        else:
            hold_frac = 0.0
            unrealized = 0.0
            trailing_active = 0.0

        state = np.array([
            # Signal features
            row.get('y_pred_proba', 0.5),
            min(row.get('volatility_4h', 0.5), 3.0) / 3.0,
            np.clip(row.get('momentum_4h', 0), -5, 5) / 5.0,
            row.get('momentum_rank', 0.5),  # NEW: momentum rank
            min(row.get('volume_ratio', 1.0), 3.0) / 3.0,
            row.get('rsi_proxy', 50) / 100.0,
            # Position state
            float(self.in_position),
            hold_frac,
            unrealized,
            trailing_active,  # NEW: trailing stop state
            # Account state
            min(len(self.trades), 10) / 10.0,
            np.clip(self.total_pnl / self.config.base_capital, -1, 1),
        ], dtype=np.float32)

        return state

    def step(self, action: int):
        """
        Take action with trailing stop logic.
        """
        if self.idx >= self.n_samples - 1:
            return self._get_state(), 0.0, True, {}

        row = self.data.iloc[self.idx]
        reward = 0.0
        info = {}
        priority = 1.0  # For prioritized replay

        # Get actual forward return
        forward_return = row.get('max_return_4h', 0.0)

        if self.in_position:
            hold_bars = self.idx - self.entry_idx

            # Update max price (simulated by momentum direction)
            current_gain = min(forward_return, self.config.take_profit_pct)
            if current_gain > 0:
                self.max_price_since_entry = max(self.max_price_since_entry, current_gain)

            # Check trailing stop activation
            if self.max_price_since_entry >= self.config.trailing_stop_activation:
                self.trailing_stop_active = True

            # Determine exit conditions
            should_exit = False
            exit_reason = ""
            pnl_pct = 0.0

            # 1. Take profit hit
            if forward_return >= self.config.take_profit_pct:
                should_exit = True
                exit_reason = "tp"
                pnl_pct = self.config.take_profit_pct - self.config.fee_pct
                priority = 3.0  # High priority for TP hits

            # 2. Stop loss hit
            elif forward_return <= -self.config.stop_loss_pct:
                should_exit = True
                exit_reason = "sl"
                pnl_pct = -self.config.stop_loss_pct - self.config.fee_pct
                priority = 2.0

            # 3. Trailing stop hit
            elif self.trailing_stop_active:
                trailing_stop_level = self.max_price_since_entry - self.config.trailing_stop_distance
                if forward_return <= trailing_stop_level:
                    should_exit = True
                    exit_reason = "trail"
                    pnl_pct = trailing_stop_level - self.config.fee_pct
                    priority = 2.5  # Good exit

            # 4. Time limit
            elif hold_bars >= self.config.max_hold_bars:
                should_exit = True
                exit_reason = "time"
                pnl_pct = min(forward_return, self.config.take_profit_pct)
                pnl_pct = max(pnl_pct, -self.config.stop_loss_pct)
                pnl_pct -= self.config.fee_pct

            # 5. Agent manual exit (only after min hold period)
            elif action == 0 and hold_bars >= self.config.min_hold_bars:
                should_exit = True
                exit_reason = "agent"
                pnl_pct = min(forward_return, self.config.take_profit_pct)
                pnl_pct = max(pnl_pct, -self.config.stop_loss_pct)
                pnl_pct -= self.config.fee_pct

                # Penalize premature exits if we're in profit
                if pnl_pct > 1.0 and hold_bars < 6:
                    reward -= 0.3  # Discourage early profit-taking

            if should_exit:
                pnl_usd = self.position_size * (pnl_pct / 100.0)
                self.total_pnl += pnl_usd
                self.trades.append({
                    'pnl_pct': pnl_pct,
                    'exit': exit_reason,
                    'hold_bars': hold_bars,
                    'trailing_active': self.trailing_stop_active,
                })

                # Reset position state
                self.in_position = False
                self.trailing_stop_active = False
                self.max_price_since_entry = 0.0

                # Reward based on PnL with bonuses
                reward = pnl_pct / 3.0  # Base reward

                if pnl_pct > 0:
                    reward += 0.5  # Win bonus
                    if exit_reason == "tp":
                        reward += 1.0  # TP hit bonus
                    elif exit_reason == "trail":
                        reward += 0.7  # Good trailing stop exit
                else:
                    reward -= 0.3  # Loss penalty

                info['exit'] = exit_reason
                info['pnl'] = pnl_pct
                info['hold_bars'] = hold_bars
            else:
                # Still holding - small reward for profitable positions
                if forward_return > 0.5:
                    reward += 0.05  # Encourage holding winners
                elif forward_return < -1.0:
                    reward -= 0.05  # Slight concern signal

        else:
            # Not in position - entry decision
            if action > 0:
                size_pct = self.config.position_sizes[action]
                self.position_size = self.config.base_capital * size_pct
                self.entry_idx = self.idx
                self.entry_price = row.get('price_at_signal', 1.0)
                self.in_position = True
                self.max_price_since_entry = 0.0
                self.trailing_stop_active = False

                # Entry cost and momentum bonus
                reward = -0.05  # Small entry cost

                # Bonus for entering high-momentum signals
                momentum = row.get('momentum_4h', 0)
                if momentum > 0.3:
                    reward += 0.1
                    info['momentum_bonus'] = True

                info['entry'] = size_pct

            else:
                # Skipped signal
                if forward_return >= 3.0:
                    # Missed a good opportunity
                    reward = -0.2
                    info['missed'] = forward_return
                    priority = 2.0

        self.idx += 1
        done = self.idx >= self.n_samples - 1

        return self._get_state(), reward, done, info, priority


class DQNAgentV4:
    """DQN agent v4 with prioritized replay."""

    def __init__(self, config: RLConfigV4):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetworkV4(config.state_dim, config.n_actions, config.hidden_dim).to(self.device)
        self.target_net = QNetworkV4(config.state_dim, config.n_actions, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.buffer = PrioritizedReplayBuffer(config.buffer_size)
        self.epsilon = config.epsilon_start
        self.step_count = 0
        self.beta = 0.4  # For importance sampling

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.config.n_actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax(dim=1).item()

    def get_position_size(self, state: np.ndarray) -> float:
        action = self.select_action(state, training=False)
        return self.config.position_sizes[action]

    def train_step(self):
        if len(self.buffer) < self.config.batch_size:
            return None

        sample = self.buffer.sample(self.config.batch_size, self.beta)
        if sample is None:
            return None

        states, actions, rewards, next_states, dones, weights, indices = sample

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1))

        # Target Q values (Double DQN)
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_net(next_states_t).argmax(1)
            # Evaluate using target network
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards_t + (1 - dones_t) * self.config.gamma * next_q

        # TD error for priority update
        td_error = torch.abs(current_q.squeeze() - target_q).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_error)

        # Weighted loss
        loss = (weights_t * F.smooth_l1_loss(current_q.squeeze(), target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.config.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + 0.001)

    def save(self, path: str):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'beta': self.beta,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_end)
        self.beta = checkpoint.get('beta', 1.0)


def train_and_evaluate():
    """Main training and evaluation with comparison to v3."""
    print("=" * 70)
    print("RL POSITION SIZING v4 - Trailing Stops + Momentum")
    print("=" * 70)

    if not HAS_TORCH:
        print("ERROR: PyTorch required")
        return None

    os.makedirs("models", exist_ok=True)

    # Load data
    features = pd.read_csv("whale_features.csv", parse_dates=['timestamp'])
    predictions = pd.read_csv("spike_predictions.csv", parse_dates=['timestamp'])

    data = predictions.merge(
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

    print(f"\nData: {len(data)} samples")

    # Split
    split = int(len(data) * 0.7)
    train_data = data.iloc[:split].reset_index(drop=True)
    test_data = data.iloc[split:].reset_index(drop=True)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Config
    config = RLConfigV4(
        n_episodes=800,
        hidden_dim=128,
        learning_rate=1e-4,
        gamma=0.95,
        epsilon_decay=0.995,
        take_profit_pct=5.0,
        stop_loss_pct=3.0,
        trailing_stop_activation=2.0,
        trailing_stop_distance=1.0,
        max_hold_bars=24,
        min_hold_bars=3,
    )

    # Agent and env
    agent = DQNAgentV4(config)
    env = TradingEnvV4(train_data, config)

    # Training
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)

    episode_rewards = []
    episode_pnls = []
    best_pnl = float('-inf')

    for ep in range(config.n_episodes):
        state = env.reset()
        ep_reward = 0.0

        for _ in range(150):  # More steps per episode
            action = agent.select_action(state)
            result = env.step(action)

            # Unpack with priority
            next_state, reward, done, info = result[:4]
            priority = result[4] if len(result) > 4 else 1.0

            agent.buffer.push(state, action, reward, next_state, done, priority)
            agent.train_step()

            ep_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(ep_reward)
        episode_pnls.append(env.total_pnl)

        # Save best model
        if env.total_pnl > best_pnl:
            best_pnl = env.total_pnl
            agent.save("models/rl_position_sizer_v4_best.pt")

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            avg_pnl = np.mean(episode_pnls[-100:])
            print(f"  Episode {ep+1}/{config.n_episodes} | "
                  f"Reward: {avg_r:.2f} | PnL: ${avg_pnl:.2f} | "
                  f"Eps: {agent.epsilon:.3f} | Best: ${best_pnl:.2f}")

    # Load best model for evaluation
    agent.load("models/rl_position_sizer_v4_best.pt")

    # Evaluation on test set
    print("\n" + "-" * 70)
    print("TEST EVALUATION")
    print("-" * 70)

    test_env = TradingEnvV4(test_data, config)
    state = test_env.reset()
    test_env.idx = 0  # Start from beginning

    position_sizes = []
    while test_env.idx < len(test_data) - 1:
        action = agent.select_action(state, training=False)
        if action > 0:
            position_sizes.append(config.position_sizes[action])
        result = test_env.step(action)
        next_state, reward, done, info = result[:4]
        state = next_state
        if done:
            break

    # Results
    print(f"\n  Total PnL: ${test_env.total_pnl:+,.2f}")
    print(f"  Trades: {len(test_env.trades)}")

    if test_env.trades:
        pnls = [t['pnl_pct'] for t in test_env.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')

        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Avg Win: {avg_win:.2f}%")
        print(f"  Avg Loss: {avg_loss:.2f}%")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Avg Position: {np.mean(position_sizes)*100:.1f}%" if position_sizes else "")

        # Exit breakdown
        exits = {}
        for t in test_env.trades:
            e = t['exit']
            exits[e] = exits.get(e, 0) + 1
        print(f"\n  Exits: {exits}")

        # Trailing stop stats
        trail_trades = [t for t in test_env.trades if t.get('trailing_active', False)]
        print(f"  Trades with trailing stop active: {len(trail_trades)}")

        # Hold time stats
        hold_times = [t['hold_bars'] for t in test_env.trades]
        print(f"  Avg hold bars: {np.mean(hold_times):.1f}")

    # Compare with baseline
    print("\n" + "-" * 70)
    print("BASELINE COMPARISON")
    print("-" * 70)

    # Baseline: trade all signals above 0.3 with 50% size
    baseline_pnl = 0.0
    baseline_trades = 0
    baseline_wins = 0

    for _, row in test_data.iterrows():
        if row['y_pred_proba'] >= 0.3:
            ret = row.get('max_return_4h', 0)
            # Apply same TP/SL limits
            ret = min(ret, config.take_profit_pct)
            ret = max(ret, -config.stop_loss_pct)
            pnl = (config.base_capital * 0.5 * ret / 100.0) - (config.base_capital * 0.5 * config.fee_pct / 100.0)
            baseline_pnl += pnl
            baseline_trades += 1
            if ret > config.fee_pct:
                baseline_wins += 1

    baseline_wr = (baseline_wins / baseline_trades * 100) if baseline_trades > 0 else 0

    print(f"\n  Baseline PnL: ${baseline_pnl:+,.2f} ({baseline_trades} trades, {baseline_wr:.1f}% win rate)")
    print(f"  RL v4 PnL:    ${test_env.total_pnl:+,.2f} ({len(test_env.trades)} trades)")

    if baseline_pnl != 0:
        improvement = (test_env.total_pnl - baseline_pnl) / abs(baseline_pnl) * 100
        print(f"\n  Improvement over baseline: {improvement:+.1f}%")
    elif test_env.total_pnl > 0:
        print(f"\n  Improvement: RL is profitable while baseline is not")

    # Save final model
    agent.save("models/rl_position_sizer_v4.pt")
    print(f"\nModel saved to models/rl_position_sizer_v4.pt")

    return agent, test_env


if __name__ == "__main__":
    train_and_evaluate()
