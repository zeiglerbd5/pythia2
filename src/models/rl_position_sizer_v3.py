"""
RL Position Sizing Agent v3

Simplified, robust implementation focusing on:
1. Fixed bugs from v2 (environment state leakage)
2. More conservative reward scaling
3. Simple DQN with proper exit handling
4. Realistic market assumptions
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
class RLConfigV3:
    """Configuration v3."""
    state_dim: int = 10
    n_actions: int = 5  # 0%, 25%, 50%, 75%, 100%
    position_sizes: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    hidden_dim: int = 64
    learning_rate: float = 1e-4
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.998

    buffer_size: int = 5000
    batch_size: int = 32
    target_update: int = 100

    n_episodes: int = 500

    # Trading (more realistic)
    base_capital: float = 1000.0
    take_profit_pct: float = 5.0
    stop_loss_pct: float = 3.0
    max_hold_bars: int = 24  # bars, not hours
    fee_pct: float = 0.2


if HAS_TORCH:
    class QNetwork(nn.Module):
        """Simple Q-network."""

        def __init__(self, state_dim: int, n_actions: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            )

        def forward(self, x):
            return self.net(x)


class SimpleReplayBuffer:
    """Simple replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class TradingEnvV3:
    """
    Simplified trading environment.

    Uses actual forward returns from data for realistic simulation.
    """

    def __init__(self, data: pd.DataFrame, config: RLConfigV3):
        self.data = data.reset_index(drop=True)
        self.config = config
        self.n_samples = len(data)

    def reset(self):
        """Reset to random starting point."""
        self.idx = random.randint(0, max(0, self.n_samples - 50))
        self.in_position = False
        self.entry_idx = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        self.total_pnl = 0.0
        self.trades = []
        return self._get_state()

    def _get_state(self):
        """Get state vector."""
        if self.idx >= self.n_samples:
            return np.zeros(self.config.state_dim)

        row = self.data.iloc[self.idx]

        # Unrealized PnL
        if self.in_position:
            # Use max_return_4h as proxy for current price movement
            unrealized = row.get('max_return_4h', 0) / 10.0
            hold_frac = min((self.idx - self.entry_idx) / self.config.max_hold_bars, 1.0)
        else:
            unrealized = 0.0
            hold_frac = 0.0

        state = np.array([
            row.get('y_pred_proba', 0.5),
            min(row.get('volatility_4h', 0.5), 3.0) / 3.0,
            np.clip(row.get('momentum_4h', 0), -5, 5) / 5.0,
            min(row.get('volume_ratio', 1.0), 3.0) / 3.0,
            row.get('rsi_proxy', 50) / 100.0,
            float(self.in_position),
            hold_frac,
            unrealized,
            min(len(self.trades), 10) / 10.0,
            self.total_pnl / self.config.base_capital,
        ], dtype=np.float32)

        return state

    def step(self, action: int):
        """
        Take action.

        action: 0=exit/skip, 1-4=position sizes
        """
        if self.idx >= self.n_samples - 1:
            return self._get_state(), 0.0, True, {}

        row = self.data.iloc[self.idx]
        reward = 0.0
        info = {}

        # Get actual forward return
        forward_return = row.get('max_return_4h', 0.0)

        if self.in_position:
            # Check exit conditions
            hold_time = self.idx - self.entry_idx
            should_exit = False
            exit_reason = ""

            if forward_return >= self.config.take_profit_pct:
                should_exit = True
                exit_reason = "tp"
                pnl_pct = self.config.take_profit_pct - self.config.fee_pct
            elif forward_return <= -self.config.stop_loss_pct:
                should_exit = True
                exit_reason = "sl"
                pnl_pct = -self.config.stop_loss_pct - self.config.fee_pct
            elif hold_time >= self.config.max_hold_bars:
                should_exit = True
                exit_reason = "time"
                pnl_pct = min(forward_return, self.config.take_profit_pct)
                pnl_pct = max(pnl_pct, -self.config.stop_loss_pct)
                pnl_pct -= self.config.fee_pct
            elif action == 0:  # Agent exits
                should_exit = True
                exit_reason = "agent"
                pnl_pct = min(forward_return, self.config.take_profit_pct)
                pnl_pct = max(pnl_pct, -self.config.stop_loss_pct)
                pnl_pct -= self.config.fee_pct

            if should_exit:
                pnl_usd = self.position_size * (pnl_pct / 100.0)
                self.total_pnl += pnl_usd
                self.trades.append({'pnl_pct': pnl_pct, 'exit': exit_reason})
                self.in_position = False

                # Reward based on PnL
                reward = pnl_pct / 5.0  # Scale

                # Bonus for profitable exits
                if pnl_pct > 0:
                    reward += 0.5
                    if exit_reason == "tp":
                        reward += 0.5

                info['exit'] = exit_reason
                info['pnl'] = pnl_pct

        else:
            # Not in position
            if action > 0:
                # Enter position
                size_pct = self.config.position_sizes[action]
                self.position_size = self.config.base_capital * size_pct
                self.entry_idx = self.idx
                self.entry_price = row.get('price_at_signal', 1.0)
                self.in_position = True

                # Small entry cost
                reward = -0.1

                info['entry'] = size_pct
            else:
                # Skipped - penalize if it was a good opportunity
                if forward_return >= 3.0:
                    reward = -0.3
                    info['missed'] = forward_return

        self.idx += 1
        done = self.idx >= self.n_samples - 1

        return self._get_state(), reward, done, info


class DQNAgentV3:
    """Simple DQN agent."""

    def __init__(self, config: RLConfigV3):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(config.state_dim, config.n_actions, config.hidden_dim).to(self.device)
        self.target_net = QNetwork(config.state_dim, config.n_actions, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.buffer = SimpleReplayBuffer(config.buffer_size)
        self.epsilon = config.epsilon_start
        self.step_count = 0

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

        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + (1 - dones_t) * self.config.gamma * next_q

        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

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

    def save(self, path: str):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_end)


def train_and_evaluate():
    """Main training and evaluation."""
    print("=" * 70)
    print("RL POSITION SIZING v3")
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
    config = RLConfigV3(
        n_episodes=500,
        hidden_dim=64,
        learning_rate=5e-4,
        gamma=0.9,
        epsilon_decay=0.995,
        take_profit_pct=5.0,
        stop_loss_pct=3.0,
        max_hold_bars=24,
    )

    # Agent and env
    agent = DQNAgentV3(config)
    env = TradingEnvV3(train_data, config)

    # Training
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)

    episode_rewards = []
    episode_pnls = []

    for ep in range(config.n_episodes):
        state = env.reset()
        ep_reward = 0.0

        for _ in range(100):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.buffer.push(state, action, reward, next_state, done)
            agent.train_step()

            ep_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(ep_reward)
        episode_pnls.append(env.total_pnl)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            avg_pnl = np.mean(episode_pnls[-100:])
            print(f"  Episode {ep+1}/{config.n_episodes} | "
                  f"Reward: {avg_r:.2f} | PnL: ${avg_pnl:.2f} | "
                  f"Eps: {agent.epsilon:.3f}")

    # Evaluation on test set
    print("\n" + "-" * 70)
    print("TEST EVALUATION")
    print("-" * 70)

    test_env = TradingEnvV3(test_data, config)
    state = test_env.reset()
    test_env.idx = 0  # Start from beginning

    position_sizes = []
    while test_env.idx < len(test_data) - 1:
        action = agent.select_action(state, training=False)
        if action > 0:
            position_sizes.append(config.position_sizes[action])
        next_state, reward, done, info = test_env.step(action)
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
    print(f"  RL PnL:       ${test_env.total_pnl:+,.2f} ({len(test_env.trades)} trades)")

    if baseline_pnl != 0:
        improvement = (test_env.total_pnl - baseline_pnl) / abs(baseline_pnl) * 100
        print(f"\n  Improvement: {improvement:+.1f}%")
    elif test_env.total_pnl > 0:
        print(f"\n  Improvement: RL is profitable while baseline is not")

    # Save
    agent.save("models/rl_position_sizer_v3.pt")
    print(f"\nModel saved to models/rl_position_sizer_v3.pt")

    return agent, test_env


if __name__ == "__main__":
    train_and_evaluate()
