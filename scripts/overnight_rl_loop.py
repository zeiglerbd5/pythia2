#!/usr/bin/env python3 -u
"""
Overnight RL Training Loop

Automatically iterates through training configurations, evaluates results,
and adjusts parameters based on sub-agent recommendations until we get
an impressive system or the user wakes up.

Success criteria:
- Win rate > 35% (above 31.4% break-even)
- Entry rate < 20% (selective)
- Positive total return in backtest
"""

import argparse
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    name: str
    entry_cost: float
    bad_setup_penalty: float
    wait_reward: float
    entry_budget: int
    tp_reward: float
    sl_penalty: float
    ent_coef: float
    gae_lambda: float
    timesteps: int = 500_000  # Shorter runs for faster iteration
    rationale: str = ""


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_trades: int
    win_rate: float
    total_return: float
    entry_rate: float
    profit_factor: float
    mean_reward: float

    @property
    def is_impressive(self) -> bool:
        """Check if results meet success criteria."""
        return (
            self.win_rate > 35.0 and  # Above break-even
            self.entry_rate < 20.0 and  # Selective
            self.total_return > 0 and  # Profitable
            self.total_trades >= 10  # Enough trades to be meaningful
        )

    @property
    def is_promising(self) -> bool:
        """Check if results show promise (worth continuing in this direction)."""
        return (
            self.win_rate > 28.0 and  # Close to break-even
            self.entry_rate < 35.0 and  # Somewhat selective
            self.total_trades >= 5
        )


# Pre-defined strategies based on sub-agent recommendations
STRATEGIES = [
    # Strategy 1: Current v4 params (baseline)
    TrainingConfig(
        name="v4_baseline",
        entry_cost=-0.8,
        bad_setup_penalty=-1.8,
        wait_reward=0.06,
        entry_budget=3,
        tp_reward=12.0,
        sl_penalty=-5.0,
        ent_coef=0.001,
        gae_lambda=0.98,
        rationale="Current v4 parameters as baseline"
    ),

    # Strategy 2: Extreme entry cost
    TrainingConfig(
        name="extreme_entry_cost",
        entry_cost=-1.5,
        bad_setup_penalty=-3.0,
        wait_reward=0.08,
        entry_budget=2,
        tp_reward=15.0,
        sl_penalty=-6.0,
        ent_coef=0.0005,
        gae_lambda=0.98,
        rationale="Much harsher entry penalties to force extreme selectivity"
    ),

    # Strategy 3: High wait rewards
    TrainingConfig(
        name="high_wait_reward",
        entry_cost=-1.0,
        bad_setup_penalty=-2.5,
        wait_reward=0.12,
        entry_budget=3,
        tp_reward=12.0,
        sl_penalty=-5.0,
        ent_coef=0.001,
        gae_lambda=0.98,
        rationale="High wait reward to make patience more attractive"
    ),

    # Strategy 4: Asymmetric with tiny budget
    TrainingConfig(
        name="tiny_budget",
        entry_cost=-0.5,
        bad_setup_penalty=-4.0,
        wait_reward=0.05,
        entry_budget=1,
        tp_reward=20.0,
        sl_penalty=-8.0,
        ent_coef=0.0005,
        gae_lambda=0.99,
        rationale="Single entry per episode forces extreme selectivity"
    ),

    # Strategy 5: Balanced aggressive
    TrainingConfig(
        name="balanced_aggressive",
        entry_cost=-1.2,
        bad_setup_penalty=-2.0,
        wait_reward=0.10,
        entry_budget=2,
        tp_reward=18.0,
        sl_penalty=-4.0,
        ent_coef=0.001,
        gae_lambda=0.98,
        rationale="High TP reward with moderate entry cost for quality over quantity"
    ),

    # Strategy 6: Zero entropy (deterministic)
    TrainingConfig(
        name="zero_entropy",
        entry_cost=-1.0,
        bad_setup_penalty=-2.5,
        wait_reward=0.08,
        entry_budget=2,
        tp_reward=15.0,
        sl_penalty=-5.0,
        ent_coef=0.0001,  # Near-zero entropy
        gae_lambda=0.98,
        rationale="Minimal exploration to converge to deterministic policy"
    ),

    # Strategy 7: Progressive penalty (simulated via high bad setup)
    TrainingConfig(
        name="quality_focus",
        entry_cost=-0.6,
        bad_setup_penalty=-5.0,  # Very harsh for bad setups
        wait_reward=0.06,
        entry_budget=3,
        tp_reward=12.0,
        sl_penalty=-5.0,
        ent_coef=0.001,
        gae_lambda=0.98,
        rationale="Harsh penalty only for low-quality entries, lenient for high-quality"
    ),

    # Strategy 8: Long horizon
    TrainingConfig(
        name="long_horizon",
        entry_cost=-0.8,
        bad_setup_penalty=-2.0,
        wait_reward=0.04,
        entry_budget=3,
        tp_reward=12.0,
        sl_penalty=-5.0,
        ent_coef=0.001,
        gae_lambda=0.995,  # Very long credit assignment
        rationale="Extended horizon for better credit assignment on rare events"
    ),
]


def update_reward_config(config: TrainingConfig) -> None:
    """Update the reward configuration file with new parameters."""
    reward_file = project_root / "src" / "rl" / "entry_timing_rewards.py"

    with open(reward_file, 'r') as f:
        content = f.read()

    # Update parameters using string replacement
    import re

    replacements = [
        (r'entry_cost: float = [-\d.]+', f'entry_cost: float = {config.entry_cost}'),
        (r'entry_cost_bad_setup: float = [-\d.]+', f'entry_cost_bad_setup: float = {config.bad_setup_penalty}'),
        (r'wait_reward_bad_conditions: float = [-\d.]+', f'wait_reward_bad_conditions: float = {config.wait_reward}'),
        (r'entry_budget: int = \d+', f'entry_budget: int = {config.entry_budget}'),
        (r'reward_take_profit: float = [-\d.]+', f'reward_take_profit: float = {config.tp_reward}'),
        (r'penalty_stop_loss: float = [-\d.]+', f'penalty_stop_loss: float = {config.sl_penalty}'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    with open(reward_file, 'w') as f:
        f.write(content)

    # Update agent config
    agent_file = project_root / "src" / "rl" / "agent.py"

    with open(agent_file, 'r') as f:
        agent_content = f.read()

    agent_content = re.sub(
        r'ent_coef: float = [-\d.]+',
        f'ent_coef: float = {config.ent_coef}',
        agent_content
    )
    agent_content = re.sub(
        r'gae_lambda: float = [-\d.]+',
        f'gae_lambda: float = {config.gae_lambda}',
        agent_content
    )

    with open(agent_file, 'w') as f:
        f.write(agent_content)

    print(f"Updated config: entry_cost={config.entry_cost}, bad_setup={config.bad_setup_penalty}, "
          f"wait={config.wait_reward}, budget={config.entry_budget}, ent_coef={config.ent_coef}")


def run_training(config: TrainingConfig, db_path: str) -> Path:
    """Run training with the given configuration."""
    experiment_name = f"overnight_{config.name}_{datetime.now().strftime('%H%M')}"
    log_file = project_root / "logs" / f"rl_overnight_{config.name}.log"

    cmd = [
        sys.executable, str(project_root / "scripts" / "train_rl_agent.py"),
        "--experiment-name", experiment_name,
        "--total-timesteps", str(config.timesteps),
        "--entry-only",
        "--db-path", db_path,
        "--take-profit", "0.12",
        "--stop-loss", "0.02",
        "--max-hold-minutes", "1440",
        "--cooldown-minutes", "30",
    ]

    print(f"\n{'='*60}")
    print(f"Starting training: {config.name}")
    print(f"Rationale: {config.rationale}")
    print(f"{'='*60}")

    with open(log_file, 'w') as f:
        process = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
        )

    if process.returncode != 0:
        print(f"Training failed with return code {process.returncode}")
        return None

    # Find the best model
    model_dir = project_root / "models" / "rl" / experiment_name / "best"
    if model_dir.exists():
        model_path = model_dir / "best_model.zip"
        if model_path.exists():
            return model_path

    # Fallback to checkpoints
    checkpoint_dir = project_root / "models" / "rl" / experiment_name / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        if checkpoints:
            return checkpoints[-1]

    return None


def run_backtest(model_path: Path, db_path: str, num_episodes: int = 100) -> Optional[BacktestResult]:
    """Run backtest on the trained model."""
    print(f"\nRunning backtest on {model_path}...")

    try:
        from stable_baselines3 import PPO
        from src.rl.entry_timing_env import EntryTimingEnvironment, EntryTimingConfig
        from src.rl.entry_timing_rewards import EntryTimingRewardCalculator, EntryTimingRewardConfig
        from src.rl.features import FeatureExtractor
        from collections import defaultdict

        # Load model
        model = PPO.load(str(model_path))

        # Create environment
        config = EntryTimingConfig(
            episode_length=1440,
            take_profit_pct=0.12,
            stop_loss_pct=0.02,
            max_hold_minutes=1440,
            cooldown_minutes=30,
            sampling_mode='event_anchored',
        )

        reward_config = EntryTimingRewardConfig()
        reward_calculator = EntryTimingRewardCalculator(reward_config)
        feature_extractor = FeatureExtractor()

        env = EntryTimingEnvironment(
            db_path=db_path,
            config=config,
            reward_calculator=reward_calculator,
            feature_extractor=feature_extractor,
        )

        # Run episodes
        all_trades = []
        episode_rewards = []
        action_counts = defaultdict(int)

        for ep in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_counts[int(action)] += 1
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                if 'trade_result' in info and info['trade_result'] is not None:
                    all_trades.append(info['trade_result'])

            episode_rewards.append(episode_reward)

        # Calculate statistics
        total_trades = len(all_trades)
        total_actions = sum(action_counts.values())
        entry_rate = (action_counts[1] / total_actions * 100) if total_actions > 0 else 0

        if total_trades > 0:
            wins = sum(1 for t in all_trades if t.return_pct > 0)
            win_rate = wins / total_trades * 100

            returns = [t.return_pct for t in all_trades]
            total_return = sum(returns)

            winning_returns = [r for r in returns if r > 0]
            losing_returns = [r for r in returns if r <= 0]

            if losing_returns and winning_returns:
                profit_factor = abs(sum(winning_returns) / sum(losing_returns))
            else:
                profit_factor = 0.0
        else:
            win_rate = 0.0
            total_return = 0.0
            profit_factor = 0.0

        result = BacktestResult(
            total_trades=total_trades,
            win_rate=win_rate,
            total_return=total_return,
            entry_rate=entry_rate,
            profit_factor=profit_factor,
            mean_reward=np.mean(episode_rewards),
        )

        print(f"\nBacktest Results:")
        print(f"  Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate:.1f}%")
        print(f"  Entry Rate: {result.entry_rate:.1f}%")
        print(f"  Total Return: {result.total_return:.2f}%")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Mean Reward: {result.mean_reward:.2f}")
        print(f"  Is Impressive: {result.is_impressive}")
        print(f"  Is Promising: {result.is_promising}")

        return result

    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_next_config(results_history: List[Dict]) -> Optional[TrainingConfig]:
    """Generate next config based on results history."""
    if not results_history:
        return None

    # Find best result so far
    best_result = max(results_history, key=lambda r: r.get('win_rate', 0))
    best_config = best_result.get('config', {})

    # If we have a promising result, try variations
    if best_result.get('win_rate', 0) > 25:
        # Intensify what's working
        return TrainingConfig(
            name=f"adaptive_{len(results_history)}",
            entry_cost=best_config.get('entry_cost', -1.0) * 1.2,
            bad_setup_penalty=best_config.get('bad_setup_penalty', -2.0) * 1.3,
            wait_reward=best_config.get('wait_reward', 0.08) * 1.1,
            entry_budget=max(1, best_config.get('entry_budget', 2) - 1),
            tp_reward=best_config.get('tp_reward', 15.0) * 1.1,
            sl_penalty=best_config.get('sl_penalty', -5.0) * 1.1,
            ent_coef=best_config.get('ent_coef', 0.001) * 0.5,
            gae_lambda=min(0.995, best_config.get('gae_lambda', 0.98) + 0.005),
            rationale=f"Adaptive config based on best result (win_rate={best_result.get('win_rate', 0):.1f}%)"
        )

    return None


def save_results(results: List[Dict], output_file: Path) -> None:
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def log(msg: str) -> None:
    """Print with flush for immediate output."""
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Overnight RL Training Loop")
    parser.add_argument("--db", type=str, default="full_pythia.duckdb", help="Database path")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum iterations")
    parser.add_argument("--backtest-episodes", type=int, default=100, help="Backtest episodes")
    args = parser.parse_args()

    results_file = project_root / "logs" / "overnight_results.json"
    results_history = []

    log("="*60)
    log("OVERNIGHT RL TRAINING LOOP")
    log("="*60)
    log(f"Started at: {datetime.now()}")
    log(f"Max iterations: {args.max_iterations}")
    log(f"Database: {args.db}")
    log(f"Success criteria: Win rate > 35%, Entry rate < 20%, Positive return")
    log("="*60)

    # Try each predefined strategy
    for i, config in enumerate(STRATEGIES):
        print(f"\n{'#'*60}")
        print(f"ITERATION {i+1}/{len(STRATEGIES)}: {config.name}")
        print(f"{'#'*60}")

        # Update configuration
        update_reward_config(config)

        # Run training
        model_path = run_training(config, args.db)

        if model_path is None:
            print("Training failed, skipping to next config...")
            results_history.append({
                'config_name': config.name,
                'config': asdict(config),
                'status': 'training_failed',
                'timestamp': str(datetime.now()),
            })
            save_results(results_history, results_file)
            continue

        # Run backtest
        result = run_backtest(model_path, args.db, args.backtest_episodes)

        if result is None:
            print("Backtest failed, skipping to next config...")
            results_history.append({
                'config_name': config.name,
                'config': asdict(config),
                'status': 'backtest_failed',
                'model_path': str(model_path),
                'timestamp': str(datetime.now()),
            })
            save_results(results_history, results_file)
            continue

        # Record results
        results_history.append({
            'config_name': config.name,
            'config': asdict(config),
            'status': 'completed',
            'model_path': str(model_path),
            'win_rate': result.win_rate,
            'entry_rate': result.entry_rate,
            'total_return': result.total_return,
            'total_trades': result.total_trades,
            'profit_factor': result.profit_factor,
            'mean_reward': result.mean_reward,
            'is_impressive': result.is_impressive,
            'is_promising': result.is_promising,
            'timestamp': str(datetime.now()),
        })
        save_results(results_history, results_file)

        # Check if we found an impressive result
        if result.is_impressive:
            print("\n" + "="*60)
            print("SUCCESS! FOUND IMPRESSIVE SYSTEM!")
            print("="*60)
            print(f"Config: {config.name}")
            print(f"Win Rate: {result.win_rate:.1f}%")
            print(f"Entry Rate: {result.entry_rate:.1f}%")
            print(f"Total Return: {result.total_return:.2f}%")
            print(f"Model: {model_path}")
            print("="*60)
            return 0

    # Try adaptive configs based on results
    print("\n" + "#"*60)
    print("TRYING ADAPTIVE CONFIGURATIONS")
    print("#"*60)

    for i in range(args.max_iterations - len(STRATEGIES)):
        adaptive_config = generate_next_config(results_history)

        if adaptive_config is None:
            print("No adaptive config generated, stopping...")
            break

        print(f"\nAdaptive iteration {i+1}")

        update_reward_config(adaptive_config)
        model_path = run_training(adaptive_config, args.db)

        if model_path:
            result = run_backtest(model_path, args.db, args.backtest_episodes)

            if result:
                results_history.append({
                    'config_name': adaptive_config.name,
                    'config': asdict(adaptive_config),
                    'status': 'completed',
                    'model_path': str(model_path),
                    'win_rate': result.win_rate,
                    'entry_rate': result.entry_rate,
                    'total_return': result.total_return,
                    'total_trades': result.total_trades,
                    'profit_factor': result.profit_factor,
                    'mean_reward': result.mean_reward,
                    'is_impressive': result.is_impressive,
                    'is_promising': result.is_promising,
                    'timestamp': str(datetime.now()),
                })
                save_results(results_history, results_file)

                if result.is_impressive:
                    print("\nSUCCESS! FOUND IMPRESSIVE SYSTEM!")
                    return 0

    # Print summary
    print("\n" + "="*60)
    print("OVERNIGHT LOOP COMPLETED")
    print("="*60)
    print(f"Total iterations: {len(results_history)}")

    # Find best result
    completed = [r for r in results_history if r.get('status') == 'completed']
    if completed:
        best = max(completed, key=lambda r: r.get('win_rate', 0))
        print(f"\nBest Result:")
        print(f"  Config: {best['config_name']}")
        print(f"  Win Rate: {best.get('win_rate', 0):.1f}%")
        print(f"  Entry Rate: {best.get('entry_rate', 0):.1f}%")
        print(f"  Total Return: {best.get('total_return', 0):.2f}%")
        print(f"  Model: {best.get('model_path', 'N/A')}")

    print(f"\nResults saved to: {results_file}")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
