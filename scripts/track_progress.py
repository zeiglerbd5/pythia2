#!/usr/bin/env python3
"""
Track Training Progress Over Time

Monitors model performance as you collect more data and retrain weekly.

Usage:
    # After each training run, log the results:
    python scripts/track_progress.py \
      --symbol BTC-USD \
      --days 10 \
      --evaluation models/prototype_BTC-USD_20251006_123456/evaluation.json

    # View progress:
    python scripts/track_progress.py --show

    # Plot improvement curve:
    python scripts/track_progress.py --plot
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger


PROGRESS_FILE = Path("data/training_progress.json")


def log_training_run(
    symbol: str,
    days: int,
    evaluation_file: str,
    notes: str = ""
):
    """
    Log a training run to progress tracker.

    Args:
        symbol: Trading pair
        days: Days of data used
        evaluation_file: Path to evaluation.json
        notes: Optional notes about this run
    """
    # Load evaluation
    eval_path = Path(evaluation_file)
    if not eval_path.exists():
        logger.error(f"Evaluation file not found: {eval_path}")
        return

    with open(eval_path, 'r') as f:
        evaluation = json.load(f)

    # Create progress entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "days_of_data": days,
        "accuracy": evaluation["classification"]["accuracy"],
        "precision": evaluation["classification"]["precision"],
        "recall": evaluation["classification"]["recall"],
        "f1": evaluation["classification"]["f1"],
        "roc_auc": evaluation["classification"]["roc_auc"],
        "tp": evaluation["classification"]["tp"],
        "fp": evaluation["classification"]["fp"],
        "tn": evaluation["classification"]["tn"],
        "fn": evaluation["classification"]["fn"],
        "meets_targets": evaluation["meets_classification_targets"],
        "model_path": str(eval_path.parent),
        "notes": notes
    }

    # Load existing progress
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
    else:
        progress = []

    # Add entry
    progress.append(entry)

    # Save
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

    logger.success(f"Logged training run: {symbol} with {days} days")
    logger.info(f"  Accuracy:  {entry['accuracy']:.4f}")
    logger.info(f"  Precision: {entry['precision']:.4f}")
    logger.info(f"  F1:        {entry['f1']:.4f}")


def show_progress(symbol: str = None):
    """
    Display progress over time.

    Args:
        symbol: Filter by symbol (optional)
    """
    if not PROGRESS_FILE.exists():
        logger.warning("No progress data yet. Log your first training run!")
        return

    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)

    # Filter by symbol if specified
    if symbol:
        progress = [p for p in progress if p["symbol"] == symbol]

    if not progress:
        logger.warning(f"No progress data for {symbol}")
        return

    # Convert to DataFrame
    df = pd.DataFrame(progress)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("days_of_data")

    logger.info("=" * 100)
    logger.info("TRAINING PROGRESS OVER TIME")
    logger.info("=" * 100)

    # Print table
    print()
    print(f"{'Week':<6} {'Days':<6} {'Date':<12} {'Accuracy':<10} {'Precision':<10} {'F1':<10} {'Targets':<8}")
    print("-" * 100)

    for idx, row in df.iterrows():
        week = (row["days_of_data"] - 10) // 7 + 1  # Assuming started at 10 days
        date = row["timestamp"].strftime("%Y-%m-%d")
        targets = "✓" if row["meets_targets"] else "✗"

        print(
            f"{week:<6} "
            f"{row['days_of_data']:<6} "
            f"{date:<12} "
            f"{row['accuracy']:<10.4f} "
            f"{row['precision']:<10.4f} "
            f"{row['f1']:<10.4f} "
            f"{targets:<8}"
        )

    print()

    # Summary statistics
    first = df.iloc[0]
    last = df.iloc[-1]

    improvement = {
        "accuracy": (last["accuracy"] - first["accuracy"]) * 100,
        "precision": (last["precision"] - first["precision"]) * 100,
        "f1": (last["f1"] - first["f1"]) * 100
    }

    logger.info("IMPROVEMENT SUMMARY")
    logger.info(f"  Data: {first['days_of_data']} → {last['days_of_data']} days (+{last['days_of_data'] - first['days_of_data']})")
    logger.info(f"  Accuracy:  {first['accuracy']:.4f} → {last['accuracy']:.4f} (+{improvement['accuracy']:.2f}%)")
    logger.info(f"  Precision: {first['precision']:.4f} → {last['precision']:.4f} (+{improvement['precision']:.2f}%)")
    logger.info(f"  F1:        {first['f1']:.4f} → {last['f1']:.4f} (+{improvement['f1']:.2f}%)")

    # Targets
    logger.info("")
    logger.info("TARGETS (Implementation Guide)")
    logger.info(f"  Accuracy:  {last['accuracy']:.4f} / 0.8244 {'✓' if last['accuracy'] >= 0.8244 else '✗'}")
    logger.info(f"  Precision: {last['precision']:.4f} / 0.9000 {'✓' if last['precision'] >= 0.90 else '✗'}")

    # Estimate days to target
    if not last["meets_targets"] and len(df) > 1:
        # Linear extrapolation (rough estimate)
        days_per_week = 7
        weeks = len(df)
        days_collected = last["days_of_data"] - first["days_of_data"]

        acc_per_day = (last["accuracy"] - first["accuracy"]) / days_collected if days_collected > 0 else 0
        prec_per_day = (last["precision"] - first["precision"]) / days_collected if days_collected > 0 else 0

        days_to_acc_target = (0.8244 - last["accuracy"]) / acc_per_day if acc_per_day > 0 else 999
        days_to_prec_target = (0.90 - last["precision"]) / prec_per_day if prec_per_day > 0 else 999

        logger.info("")
        logger.info("ESTIMATED TIME TO TARGETS (Linear Extrapolation)")
        if days_to_acc_target < 999:
            logger.info(f"  Accuracy target:  ~{int(days_to_acc_target)} more days")
        if days_to_prec_target < 999:
            logger.info(f"  Precision target: ~{int(days_to_prec_target)} more days")

    logger.info("=" * 100)


def plot_progress(symbol: str = None):
    """
    Plot progress over time.

    Args:
        symbol: Filter by symbol (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed. Install with: pip install matplotlib")
        return

    if not PROGRESS_FILE.exists():
        logger.warning("No progress data yet. Log your first training run!")
        return

    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)

    # Filter by symbol if specified
    if symbol:
        progress = [p for p in progress if p["symbol"] == symbol]

    if not progress:
        logger.warning(f"No progress data for {symbol}")
        return

    # Convert to DataFrame
    df = pd.DataFrame(progress)
    df = df.sort_values("days_of_data")

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Training Progress Over Time - {symbol or 'All Symbols'}", fontsize=14)

    # Accuracy
    axes[0, 0].plot(df["days_of_data"], df["accuracy"], marker='o', linewidth=2)
    axes[0, 0].axhline(y=0.8244, color='r', linestyle='--', label='Target (82.44%)')
    axes[0, 0].set_xlabel("Days of Data")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Accuracy Over Time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Precision
    axes[0, 1].plot(df["days_of_data"], df["precision"], marker='o', linewidth=2, color='green')
    axes[0, 1].axhline(y=0.90, color='r', linestyle='--', label='Target (90%)')
    axes[0, 1].set_xlabel("Days of Data")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision Over Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 0].plot(df["days_of_data"], df["f1"], marker='o', linewidth=2, color='orange')
    axes[1, 0].set_xlabel("Days of Data")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_title("F1 Score Over Time")
    axes[1, 0].grid(True, alpha=0.3)

    # ROC AUC
    axes[1, 1].plot(df["days_of_data"], df["roc_auc"], marker='o', linewidth=2, color='purple')
    axes[1, 1].set_xlabel("Days of Data")
    axes[1, 1].set_ylabel("ROC AUC")
    axes[1, 1].set_title("ROC AUC Over Time")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/training_progress.png", dpi=150)
    logger.success("Plot saved to: data/training_progress.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Track training progress')

    # Action
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--log', action='store_true', help='Log a training run')
    group.add_argument('--show', action='store_true', help='Show progress')
    group.add_argument('--plot', action='store_true', help='Plot progress')

    # Logging parameters
    parser.add_argument('--symbol', help='Trading pair symbol')
    parser.add_argument('--days', type=int, help='Days of data used')
    parser.add_argument('--evaluation', help='Path to evaluation.json')
    parser.add_argument('--notes', default='', help='Notes about this run')

    args = parser.parse_args()

    if args.log:
        if not all([args.symbol, args.days, args.evaluation]):
            logger.error("--log requires: --symbol, --days, --evaluation")
            sys.exit(1)

        log_training_run(
            symbol=args.symbol,
            days=args.days,
            evaluation_file=args.evaluation,
            notes=args.notes
        )

    elif args.show:
        show_progress(symbol=args.symbol)

    elif args.plot:
        plot_progress(symbol=args.symbol)

    else:
        logger.error("Specify an action: --log, --show, or --plot")
        parser.print_help()


if __name__ == "__main__":
    main()
