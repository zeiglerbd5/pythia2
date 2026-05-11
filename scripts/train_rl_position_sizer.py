#!/usr/bin/env python3
"""
Train RL Position Sizing Agent

This script trains the DQN agent for position sizing and evaluates
its performance against the baseline fixed-size strategy.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.rl_position_sizer import (
    RLTrainer,
    RLConfig,
    PositionSizingAgent,
    HAS_TORCH
)


def main():
    """Train and evaluate RL position sizing agent."""
    print("=" * 70)
    print("RL POSITION SIZING TRAINING")
    print("=" * 70)

    if not HAS_TORCH:
        print("\nERROR: PyTorch not installed.")
        print("Install with: pip install torch")
        return

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Configuration optimized for small dataset
    config = RLConfig(
        # Network
        state_dim=8,
        n_actions=5,
        hidden_dim=64,

        # Training
        n_episodes=500,
        max_steps_per_episode=40,
        learning_rate=5e-4,
        gamma=0.95,

        # Exploration
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,

        # Experience replay
        buffer_size=5000,
        batch_size=32,
        min_buffer_size=64,

        # Target network
        target_update_freq=50,
        tau=0.01,

        # Reward shaping
        spike_bonus_multiplier=2.0,
        missed_spike_penalty=-0.3,
        transaction_cost_pct=0.2,
        drawdown_penalty_factor=0.05,
        holding_penalty_per_step=0.002,

        # Trading
        base_position_size=1000.0,
        take_profit_pct=10.0,
        stop_loss_pct=5.0,
        max_hold_steps=24,
    )

    # Check for data files
    features_path = "whale_features.csv"
    predictions_path = "spike_predictions.csv"

    if not os.path.exists(features_path):
        print(f"\nERROR: {features_path} not found.")
        print("Run spike_predictor.py first to generate predictions.")
        return

    if not os.path.exists(predictions_path):
        print(f"\nERROR: {predictions_path} not found.")
        print("Run spike_predictor.py first to generate predictions.")
        return

    # Initialize trainer
    print("\nLoading data...")
    trainer = RLTrainer(
        features_csv=features_path,
        predictions_csv=predictions_path,
        config=config,
    )

    # Train
    print("\n" + "-" * 70)
    print("TRAINING PHASE")
    print("-" * 70)
    print(f"\nConfiguration:")
    print(f"  Episodes: {config.n_episodes}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Epsilon decay: {config.epsilon_decay}")

    train_metrics = trainer.train(verbose=True)

    # Evaluate on training set
    print("\n" + "-" * 70)
    print("TRAINING SET EVALUATION")
    print("-" * 70)
    train_eval = trainer.evaluate(use_test_data=False)
    print(f"\n  Total PnL: ${train_eval['total_pnl']:+,.2f}")
    print(f"  Trades: {train_eval['n_trades']}")
    print(f"  Win Rate: {train_eval.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor: {train_eval.get('profit_factor', 0):.2f}")

    # Evaluate on test set
    print("\n" + "-" * 70)
    print("TEST SET EVALUATION")
    print("-" * 70)
    test_eval = trainer.evaluate(use_test_data=True)
    print(f"\n  Total PnL: ${test_eval['total_pnl']:+,.2f}")
    print(f"  Trades: {test_eval['n_trades']}")
    print(f"  Win Rate: {test_eval.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor: {test_eval.get('profit_factor', 0):.2f}")
    print(f"  Max Drawdown: {test_eval['max_drawdown']*100:.1f}%")
    print(f"  Avg Position Size: {test_eval['avg_position_size']*100:.1f}%")

    if 'exits_tp' in test_eval:
        print(f"\n  Exit Breakdown:")
        print(f"    Take Profit: {test_eval['exits_tp']}")
        print(f"    Stop Loss: {test_eval['exits_sl']}")
        print(f"    Time Limit: {test_eval['exits_time']}")
        print(f"    Agent Exit: {test_eval['exits_agent']}")

    # Compare with baseline
    print("\n" + "-" * 70)
    print("BASELINE COMPARISON")
    print("-" * 70)
    comparison = trainer.compare_with_baseline()

    print(f"\n  RL Agent:     ${comparison['rl']['total_pnl']:+,.2f} "
          f"({comparison['rl']['n_trades']} trades, "
          f"{comparison['rl'].get('win_rate', 0):.1f}% win rate)")

    print(f"  Baseline:     ${comparison['baseline']['total_pnl']:+,.2f} "
          f"({comparison['baseline']['n_trades']} trades, "
          f"{comparison['baseline'].get('win_rate', 0):.1f}% win rate)")

    print(f"\n  Improvement:  {comparison['improvement_pct']:+.1f}%")

    # Save model
    model_path = "models/rl_position_sizer.pt"
    trainer.agent.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Training curve summary
    print("\n" + "-" * 70)
    print("TRAINING SUMMARY")
    print("-" * 70)

    rewards = train_metrics['episode_rewards']
    pnls = train_metrics['episode_pnls']

    # First half vs second half performance
    mid = len(rewards) // 2
    first_half_reward = sum(rewards[:mid]) / mid if mid > 0 else 0
    second_half_reward = sum(rewards[mid:]) / (len(rewards) - mid) if len(rewards) > mid else 0

    print(f"\n  First half avg reward:  {first_half_reward:.3f}")
    print(f"  Second half avg reward: {second_half_reward:.3f}")
    print(f"  Improvement: {((second_half_reward - first_half_reward) / abs(first_half_reward) * 100) if first_half_reward != 0 else 0:+.1f}%")

    first_half_pnl = sum(pnls[:mid]) / mid if mid > 0 else 0
    second_half_pnl = sum(pnls[mid:]) / (len(pnls) - mid) if len(pnls) > mid else 0

    print(f"\n  First half avg PnL:  ${first_half_pnl:+,.2f}")
    print(f"  Second half avg PnL: ${second_half_pnl:+,.2f}")

    print(f"\n  Final epsilon: {train_metrics['final_epsilon']:.3f}")

    # Decision
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if comparison['improvement_pct'] > 10:
        print("\n  RL agent shows SIGNIFICANT improvement over baseline.")
        print("  Recommend using RL-based position sizing.")
    elif comparison['improvement_pct'] > 0:
        print("\n  RL agent shows MARGINAL improvement over baseline.")
        print("  Consider additional training or hyperparameter tuning.")
    else:
        print("\n  RL agent does NOT improve over baseline.")
        print("  See architectural revisions below.")
        print_revision_suggestions(test_eval, comparison)


def print_revision_suggestions(eval_results: dict, comparison: dict):
    """Print suggestions for architectural revisions."""
    print("\n" + "-" * 70)
    print("SUGGESTED REVISIONS")
    print("-" * 70)

    suggestions = []

    # Check for issues
    if eval_results.get('n_trades', 0) < 5:
        suggestions.append(
            "1. SPARSE TRADING: Agent is too conservative.\n"
            "   - Reduce missed_spike_penalty\n"
            "   - Increase exploration (slower epsilon decay)\n"
            "   - Add reward for taking positions"
        )

    if eval_results.get('win_rate', 0) < 40:
        suggestions.append(
            "2. LOW WIN RATE: Agent entry timing is poor.\n"
            "   - Add more features (order book imbalance, funding rates)\n"
            "   - Use LSTM/GRU for temporal patterns\n"
            "   - Implement hindsight experience replay"
        )

    if eval_results.get('exits_sl', 0) > eval_results.get('exits_tp', 0) * 2:
        suggestions.append(
            "3. TOO MANY STOP LOSSES: Risk management is failing.\n"
            "   - Increase stop_loss penalty in reward\n"
            "   - Add volatility-adjusted position sizing\n"
            "   - Consider tighter stop losses with larger positions"
        )

    if eval_results.get('avg_position_size', 0) < 0.3:
        suggestions.append(
            "4. SMALL POSITIONS: Agent is not confident.\n"
            "   - Scale rewards by position size taken\n"
            "   - Add bonus for larger positions on correct predictions"
        )

    if comparison.get('improvement_pct', 0) < -20:
        suggestions.append(
            "5. SEVERE UNDERPERFORMANCE: Consider architectural changes.\n"
            "   - Switch to PPO (more stable than DQN)\n"
            "   - Use continuous action space for position sizing\n"
            "   - Implement distributional RL (QR-DQN) for uncertainty"
        )

    if not suggestions:
        suggestions.append(
            "No specific issues identified. Consider:\n"
            "   - More training episodes\n"
            "   - Larger network capacity\n"
            "   - Feature engineering improvements"
        )

    for suggestion in suggestions:
        print(f"\n  {suggestion}")


if __name__ == "__main__":
    main()
