#!/usr/bin/env python3
"""Backtest the entry-only RL model."""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from src.rl.entry_timing_env import EntryTimingEnvironment, EntryTimingConfig
from src.rl.entry_timing_rewards import EntryTimingRewardCalculator, EntryTimingRewardConfig
from src.rl.features import FeatureExtractor
from src.rl.lstm_features import FrameStackWrapper


def run_backtest(
    model_path: str,
    db_path: str,
    num_episodes: int = 50,
    take_profit: float = 0.12,
    stop_loss: float = 0.02,
    max_hold_minutes: int = 1440,
    cooldown_minutes: int = 30,
    use_lstm: bool = False,
    lstm_sequence_length: int = 60,
    verbose: bool = True,
):
    """Run backtest on the entry-only model."""

    print("=" * 60)
    print("Entry-Only Model Backtest")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Exit Rules: TP={take_profit*100:.1f}%, SL={stop_loss*100:.1f}%, MaxHold={max_hold_minutes}min")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path)

    # Create config
    config = EntryTimingConfig(
        episode_length=1440,
        take_profit_pct=take_profit,
        stop_loss_pct=stop_loss,
        max_hold_minutes=max_hold_minutes,
        cooldown_minutes=cooldown_minutes,
        sampling_mode='event_anchored',
    )

    # Create reward calculator
    reward_config = EntryTimingRewardConfig()
    reward_calculator = EntryTimingRewardCalculator(reward_config)

    # Create feature extractor
    feature_extractor = FeatureExtractor()

    # Create environment
    print("Creating environment...")
    env = EntryTimingEnvironment(
        db_path=db_path,
        config=config,
        reward_calculator=reward_calculator,
        feature_extractor=feature_extractor,
    )

    # Wrap with frame stacking for LSTM models
    if use_lstm:
        print(f"Wrapping environment with FrameStackWrapper (n_frames={lstm_sequence_length})")
        env = FrameStackWrapper(env, n_frames=lstm_sequence_length)

    # Track results
    all_trades = []
    episode_rewards = []
    episode_returns = []

    action_counts = defaultdict(int)
    exit_reason_counts = defaultdict(int)

    print(f"\nRunning {num_episodes} episodes...")
    print("-" * 60)

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_trades = []
        done = False
        step = 0

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action)] += 1

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

            # Track trades
            if 'trade_result' in info and info['trade_result'] is not None:
                trade = info['trade_result']
                episode_trades.append(trade)
                all_trades.append(trade)
                exit_reason_counts[str(trade.exit_reason)] += 1

        # Calculate episode return
        episode_return = sum(t.return_pct for t in episode_trades) if episode_trades else 0
        episode_rewards.append(episode_reward)
        episode_returns.append(episode_return)

        if verbose and (ep + 1) % 10 == 0:
            wins = sum(1 for t in episode_trades if t.return_pct > 0)
            losses = len(episode_trades) - wins
            print(f"Episode {ep+1:3d}: Reward={episode_reward:7.2f}, "
                  f"Trades={len(episode_trades):2d}, W/L={wins}/{losses}, "
                  f"Return={episode_return:6.2f}%")

    # Calculate statistics
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    total_trades = len(all_trades)
    if total_trades > 0:
        wins = sum(1 for t in all_trades if t.return_pct > 0)
        losses = total_trades - wins
        win_rate = wins / total_trades * 100

        returns = [t.return_pct for t in all_trades]
        avg_return = np.mean(returns)
        total_return = sum(returns)

        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r <= 0]

        avg_win = np.mean(winning_returns) if winning_returns else 0
        avg_loss = np.mean(losing_returns) if losing_returns else 0

        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Wins: {wins} ({win_rate:.1f}%)")
        print(f"  Losses: {losses} ({100-win_rate:.1f}%)")
        print(f"  Average Return: {avg_return:.2f}%")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Average Win: {avg_win:.2f}%")
        print(f"  Average Loss: {avg_loss:.2f}%")

        if losing_returns:
            profit_factor = abs(sum(winning_returns) / sum(losing_returns)) if winning_returns else 0
            print(f"  Profit Factor: {profit_factor:.2f}")

        print(f"\nExit Reasons:")
        for reason, count in sorted(exit_reason_counts.items()):
            pct = count / total_trades * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
    else:
        print("\nNo trades executed!")
        win_rate = 0
        total_return = 0

    print(f"\nAction Distribution:")
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items()):
        action_name = "WAIT" if action == 0 else "ENTER"
        pct = count / total_actions * 100
        print(f"  {action_name}: {count} ({pct:.1f}%)")

    print(f"\nEpisode Statistics:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"  Mean Return: {np.mean(episode_returns):.2f}% +/- {np.std(episode_returns):.2f}%")
    print(f"  Trades/Episode: {total_trades / num_episodes:.1f}")

    # Profitability assessment
    print("\n" + "=" * 60)
    print("PROFITABILITY ASSESSMENT")
    print("=" * 60)

    # With 12% TP and 2% SL (after ~1.2% fees each way = ~2.4% total fees)
    # Net TP ≈ 9.6%, Net SL ≈ 4.4%
    # Break-even win rate = 4.4 / (9.6 + 4.4) ≈ 31.4%
    effective_tp = take_profit - 0.024  # After fees
    effective_sl = stop_loss + 0.024    # After fees
    breakeven_wr = effective_sl / (effective_tp + effective_sl) * 100

    if total_trades > 0:
        if win_rate > breakeven_wr:
            print(f"  Status: PROFITABLE")
            print(f"  Win Rate ({win_rate:.1f}%) > Break-even ({breakeven_wr:.1f}%)")
        else:
            print(f"  Status: NOT PROFITABLE")
            print(f"  Win Rate ({win_rate:.1f}%) < Break-even ({breakeven_wr:.1f}%)")
            print(f"  Need {breakeven_wr - win_rate:.1f}% higher win rate")
    else:
        print("  Status: NO TRADES - Cannot assess profitability")

    print("=" * 60)

    return {
        'total_trades': total_trades,
        'win_rate': win_rate if total_trades > 0 else 0,
        'total_return': total_return if total_trades > 0 else 0,
        'mean_reward': np.mean(episode_rewards),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest entry-only RL model")
    parser.add_argument("--model", type=str, default="models/rl/entry_v1/best/best_model.zip",
                        help="Path to model")
    parser.add_argument("--db", type=str, default="rl_training_data.db",
                        help="Path to database")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of episodes to run")
    parser.add_argument("--take-profit", type=float, default=0.12,
                        help="Take profit percentage")
    parser.add_argument("--stop-loss", type=float, default=0.02,
                        help="Stop loss percentage")
    parser.add_argument("--max-hold", type=int, default=1440,
                        help="Max hold time in minutes")
    parser.add_argument("--cooldown", type=int, default=30,
                        help="Cooldown between trades in minutes")
    parser.add_argument("--use-lstm", action="store_true",
                        help="Use LSTM frame stacking for the environment")
    parser.add_argument("--lstm-seq-len", type=int, default=60,
                        help="LSTM sequence length (default: 60)")

    args = parser.parse_args()

    run_backtest(
        model_path=args.model,
        db_path=args.db,
        num_episodes=args.episodes,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        max_hold_minutes=args.max_hold,
        cooldown_minutes=args.cooldown,
        use_lstm=args.use_lstm,
        lstm_sequence_length=args.lstm_seq_len,
    )
