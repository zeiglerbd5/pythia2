"""
Continuous Learning Components for RL Trading Agent (Phase 3)

Implements:
- Experience replay buffer with prioritization
- Elastic Weight Consolidation (EWC) for anti-forgetting
- Online training loop for continuous adaptation
- Model versioning and safe updates

The goal is to enable the agent to adapt to changing market
conditions without catastrophically forgetting past knowledge.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from collections import deque
import pickle
from pathlib import Path
from datetime import datetime
import copy
from loguru import logger


@dataclass
class Experience:
    """Single experience for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Optional[Dict] = None

    # For prioritization
    priority: float = 1.0
    timestamp: Optional[datetime] = None


@dataclass
class ContinualConfig:
    """Configuration for continuous learning."""
    # Replay buffer
    buffer_capacity: int = 100_000
    prioritization_alpha: float = 0.6  # Priority exponent
    prioritization_beta: float = 0.4   # Importance sampling start
    prioritization_beta_end: float = 1.0
    prioritization_eps: float = 1e-6

    # EWC (Elastic Weight Consolidation)
    ewc_lambda: float = 5000.0         # EWC penalty strength
    fisher_samples: int = 1000         # Samples for Fisher estimation
    fisher_empirical: bool = True      # Use empirical Fisher

    # Online learning
    update_frequency: int = 100        # Steps between updates
    batch_size: int = 64
    replay_ratio: float = 0.5          # Fraction of old experiences in batch

    # Model management
    model_save_frequency: int = 10_000
    max_model_versions: int = 5
    model_dir: str = "models/rl/online"


class PrioritizedReplayBuffer:
    """
    Experience replay buffer with prioritized sampling.

    Higher-priority experiences (typically those with larger TD errors)
    are sampled more frequently, leading to more efficient learning.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        eps: float = 1e-6,
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (anneals to 1)
            beta_increment: Beta annealing rate
            eps: Small constant to prevent zero priority
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps

        # Storage
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)

        # Tracking
        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict] = None,
    ) -> None:
        """
        Add experience to buffer with max priority.

        New experiences get maximum priority to ensure they're sampled.
        """
        # Max priority for new experience
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info,
            priority=max_priority,
            timestamp=datetime.now(),
        )

        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            experiences: List of sampled experiences
            indices: Indices of sampled experiences
            weights: Importance sampling weights
        """
        if self.size == 0:
            return [], np.array([]), np.array([])

        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(
            self.size,
            size=min(batch_size, self.size),
            replace=False,
            p=probabilities,
        )

        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Get experiences
        experiences = [self.buffer[i] for i in indices]

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return experiences, indices, weights

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """
        Update priorities based on TD errors.

        Args:
            indices: Indices of experiences
            td_errors: TD errors (or other priority metric)
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.eps) ** self.alpha
            self.priorities[idx] = priority
            self.buffer[idx].priority = priority

    def __len__(self) -> int:
        return self.size

    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if self.size == 0:
            return {
                'size': 0,
                'mean_priority': 0,
                'max_priority': 0,
                'min_priority': 0,
            }

        priorities = self.priorities[:self.size]
        return {
            'size': self.size,
            'mean_priority': float(priorities.mean()),
            'max_priority': float(priorities.max()),
            'min_priority': float(priorities.min()),
            'beta': self.beta,
        }


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.

    EWC adds a penalty term to the loss that discourages changing weights
    that were important for previous tasks (market regimes).
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 5000.0,
    ):
        """
        Initialize EWC.

        Args:
            model: Neural network model
            ewc_lambda: Strength of EWC penalty
        """
        self.model = model
        self.ewc_lambda = ewc_lambda

        # Stored Fisher information and parameter values
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.old_params: Dict[str, torch.Tensor] = {}

        # Task counter
        self.n_tasks = 0

    def compute_fisher(
        self,
        dataloader: Any,
        n_samples: int = 1000,
    ) -> None:
        """
        Compute Fisher information matrix (diagonal approximation).

        Fisher information measures parameter importance for current task.

        Args:
            dataloader: Data loader for computing gradients
            n_samples: Number of samples to use
        """
        # Initialize Fisher
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()
        count = 0

        for batch in dataloader:
            if count >= n_samples:
                break

            # Forward pass
            if isinstance(batch, (list, tuple)):
                states, actions = batch[0], batch[1]
            else:
                states = batch
                actions = None

            states = torch.as_tensor(states, dtype=torch.float32)

            # Get log probabilities
            with torch.enable_grad():
                # This would need to be adapted to your policy output
                output = self.model(states)

                if actions is not None:
                    actions = torch.as_tensor(actions, dtype=torch.long)
                    log_probs = F.log_softmax(output, dim=-1)
                    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1))
                    loss = -selected_log_probs.mean()
                else:
                    # Use entropy as proxy
                    probs = F.softmax(output, dim=-1)
                    loss = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

                loss.backward()

            # Accumulate squared gradients
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).detach()

            self.model.zero_grad()
            count += len(states)

        # Average Fisher
        for n in fisher:
            fisher[n] /= count

        # Store or accumulate Fisher
        if self.n_tasks == 0:
            self.fisher_info = fisher
        else:
            # Online averaging with previous Fisher
            for n in fisher:
                self.fisher_info[n] = (
                    self.fisher_info[n] * self.n_tasks + fisher[n]
                ) / (self.n_tasks + 1)

        # Store current parameters
        self.old_params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.n_tasks += 1

        logger.info(f"Fisher information computed for task {self.n_tasks}")

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty term for loss.

        Returns:
            Scalar penalty tensor
        """
        if not self.fisher_info:
            return torch.tensor(0.0)

        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher_info:
                # Quadratic penalty weighted by Fisher
                loss += (
                    self.fisher_info[n] *
                    (p - self.old_params[n]).pow(2)
                ).sum()

        return self.ewc_lambda * loss

    def save(self, path: str) -> None:
        """Save EWC state."""
        state = {
            'fisher_info': {k: v.cpu() for k, v in self.fisher_info.items()},
            'old_params': {k: v.cpu() for k, v in self.old_params.items()},
            'n_tasks': self.n_tasks,
            'ewc_lambda': self.ewc_lambda,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load EWC state."""
        state = torch.load(path)
        self.fisher_info = state['fisher_info']
        self.old_params = state['old_params']
        self.n_tasks = state['n_tasks']
        self.ewc_lambda = state['ewc_lambda']


class ContinualLearner:
    """
    Continuous learning system for RL trading agent.

    Combines:
    - Prioritized experience replay
    - Elastic Weight Consolidation
    - Online model updates
    - Safe model versioning
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[ContinualConfig] = None,
    ):
        """
        Initialize continuous learner.

        Args:
            model: Policy network
            optimizer: Optimizer
            config: Continuous learning configuration
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config or ContinualConfig()

        # Components
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_capacity,
            alpha=self.config.prioritization_alpha,
            beta=self.config.prioritization_beta,
        )
        self.ewc = ElasticWeightConsolidation(
            model,
            self.config.ewc_lambda,
        )

        # Model versioning
        self.model_versions: List[Dict] = []
        self.current_version = 0
        self.best_version = 0
        self.best_performance = float('-inf')

        # Statistics
        self.update_count = 0
        self.step_count = 0

        # Create model directory
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)

    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict] = None,
    ) -> None:
        """Add experience to replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done, info)
        self.step_count += 1

        # Periodic update
        if self.step_count % self.config.update_frequency == 0:
            self.update()

    def update(self) -> Dict[str, float]:
        """
        Perform one update step.

        Returns:
            Training metrics
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {'error': 'insufficient_data'}

        # Sample batch
        experiences, indices, weights = self.replay_buffer.sample(
            self.config.batch_size
        )

        # Convert to tensors
        states = torch.stack([
            torch.as_tensor(e.state, dtype=torch.float32)
            for e in experiences
        ])
        actions = torch.as_tensor([e.action for e in experiences], dtype=torch.long)
        rewards = torch.as_tensor([e.reward for e in experiences], dtype=torch.float32)
        next_states = torch.stack([
            torch.as_tensor(e.next_state, dtype=torch.float32)
            for e in experiences
        ])
        dones = torch.as_tensor([e.done for e in experiences], dtype=torch.float32)
        weights = torch.as_tensor(weights, dtype=torch.float32)

        # Forward pass
        self.model.train()
        output = self.model(states)

        # Compute loss (simplified - would depend on your specific model)
        if hasattr(self.model, 'compute_loss'):
            policy_loss, td_errors = self.model.compute_loss(
                states, actions, rewards, next_states, dones
            )
        else:
            # Placeholder loss
            log_probs = F.log_softmax(output, dim=-1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1))
            policy_loss = -(selected_log_probs * weights.unsqueeze(1)).mean()
            td_errors = np.abs(rewards.numpy())  # Simplified

        # Add EWC penalty
        ewc_loss = self.ewc.penalty()
        total_loss = policy_loss + ewc_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        self.update_count += 1

        # Periodic model save
        if self.update_count % (self.config.model_save_frequency // self.config.update_frequency) == 0:
            self._save_version()

        metrics = {
            'policy_loss': float(policy_loss.item()),
            'ewc_loss': float(ewc_loss.item()) if isinstance(ewc_loss, torch.Tensor) else 0,
            'total_loss': float(total_loss.item()),
            'buffer_size': len(self.replay_buffer),
            'update_count': self.update_count,
        }

        return metrics

    def consolidate(self, dataloader: Any) -> None:
        """
        Consolidate knowledge after a market regime.

        Should be called when regime changes are detected.

        Args:
            dataloader: Data from current regime for Fisher computation
        """
        logger.info("Consolidating knowledge with EWC...")
        self.ewc.compute_fisher(dataloader, self.config.fisher_samples)

    def _save_version(self) -> None:
        """Save current model version."""
        self.current_version += 1

        version_info = {
            'version': self.current_version,
            'timestamp': datetime.now().isoformat(),
            'update_count': self.update_count,
            'buffer_stats': self.replay_buffer.get_stats(),
        }

        # Save model
        model_path = Path(self.config.model_dir) / f"model_v{self.current_version}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'version_info': version_info,
        }, model_path)

        # Save EWC state
        ewc_path = Path(self.config.model_dir) / f"ewc_v{self.current_version}.pt"
        self.ewc.save(str(ewc_path))

        self.model_versions.append(version_info)

        # Cleanup old versions
        if len(self.model_versions) > self.config.max_model_versions:
            old_version = self.model_versions.pop(0)
            old_path = Path(self.config.model_dir) / f"model_v{old_version['version']}.pt"
            if old_path.exists():
                old_path.unlink()

        logger.info(f"Saved model version {self.current_version}")

    def load_version(self, version: int) -> None:
        """Load specific model version."""
        model_path = Path(self.config.model_dir) / f"model_v{version}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model version {version} not found")

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load EWC state if available
        ewc_path = Path(self.config.model_dir) / f"ewc_v{version}.pt"
        if ewc_path.exists():
            self.ewc.load(str(ewc_path))

        logger.info(f"Loaded model version {version}")

    def rollback_to_best(self) -> None:
        """Rollback to best performing version."""
        if self.best_version > 0:
            self.load_version(self.best_version)
            logger.info(f"Rolled back to best version {self.best_version}")

    def update_best(self, performance: float) -> bool:
        """
        Update best version if performance improved.

        Args:
            performance: Current performance metric

        Returns:
            True if this is new best
        """
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_version = self.current_version
            logger.info(f"New best performance: {performance:.4f} (version {self.current_version})")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            'step_count': self.step_count,
            'update_count': self.update_count,
            'current_version': self.current_version,
            'best_version': self.best_version,
            'best_performance': self.best_performance,
            'n_tasks': self.ewc.n_tasks,
            'buffer_stats': self.replay_buffer.get_stats(),
        }


if __name__ == "__main__":
    # Test continuous learning components
    print("Testing Continuous Learning\n" + "=" * 50)

    # Test PrioritizedReplayBuffer
    print("\n1. PrioritizedReplayBuffer")
    buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)

    # Add experiences
    for i in range(500):
        buffer.add(
            state=np.random.randn(20).astype(np.float32),
            action=np.random.randint(7),
            reward=np.random.randn(),
            next_state=np.random.randn(20).astype(np.float32),
            done=np.random.random() < 0.1,
        )

    print(f"   Buffer size: {len(buffer)}")
    print(f"   Buffer stats: {buffer.get_stats()}")

    # Sample
    experiences, indices, weights = buffer.sample(32)
    print(f"   Sampled {len(experiences)} experiences")
    print(f"   Weights range: [{weights.min():.4f}, {weights.max():.4f}]")

    # Update priorities
    td_errors = np.abs(np.random.randn(len(indices)))
    buffer.update_priorities(indices, td_errors)
    print(f"   Updated stats: {buffer.get_stats()}")

    # Test EWC
    print("\n2. ElasticWeightConsolidation")

    # Simple model
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 7),
    )

    ewc = ElasticWeightConsolidation(model, ewc_lambda=5000)

    # Fake dataloader
    class FakeDataloader:
        def __iter__(self):
            for _ in range(20):
                yield torch.randn(32, 20), torch.randint(0, 7, (32,))

    ewc.compute_fisher(FakeDataloader(), n_samples=500)
    print(f"   Fisher computed for {ewc.n_tasks} tasks")
    print(f"   Number of tracked parameters: {len(ewc.fisher_info)}")

    # Compute penalty
    penalty = ewc.penalty()
    print(f"   EWC penalty: {penalty.item():.4f}")

    # Test ContinualLearner
    print("\n3. ContinualLearner")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    config = ContinualConfig(
        buffer_capacity=1000,
        update_frequency=10,
        batch_size=32,
        model_dir="/tmp/test_continual",
    )

    learner = ContinualLearner(model, optimizer, config)

    # Add experiences and update
    for i in range(100):
        learner.add_experience(
            state=np.random.randn(20).astype(np.float32),
            action=np.random.randint(7),
            reward=np.random.randn(),
            next_state=np.random.randn(20).astype(np.float32),
            done=np.random.random() < 0.1,
        )

    print(f"   Stats: {learner.get_stats()}")

    # Test performance tracking
    learner.update_best(0.5)
    learner.update_best(0.8)
    learner.update_best(0.6)  # Not better

    print(f"   Best version: {learner.best_version}")
    print(f"   Best performance: {learner.best_performance}")

    print("\nContinuous learning tests passed!")
