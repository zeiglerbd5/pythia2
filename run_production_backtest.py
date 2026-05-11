#!/usr/bin/env python
"""
Production Spike Prediction System - Backtest Runner

This script runs a full comparison between:
1. Baseline system (single LightGBM + fixed position sizing)
2. Integrated system (Ensemble + Volatility Filter + RL Position Sizing)

Usage:
    python run_production_backtest.py

Output:
    - Console: Comparison metrics
    - integrated_backtest_trades.csv: Detailed trade log
    - equity_curves.csv: Equity curves for both systems
"""

import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.integrated_production_system import run_full_comparison


def main():
    """Run the production backtest comparison."""
    print("Starting Production Backtest Comparison...")
    print("This will take a few minutes to train the ensemble and run backtests.\n")

    try:
        baseline_result, integrated_result = run_full_comparison()

        print("\n" + "=" * 80)
        print("BACKTEST COMPLETE")
        print("=" * 80)

        # Summary
        improvement_pct = (
            (integrated_result.total_pnl - baseline_result.total_pnl)
            / abs(baseline_result.total_pnl) * 100
            if baseline_result.total_pnl != 0 else float('inf')
        )

        print(f"\nIntegrated system PnL improvement: {improvement_pct:+.1f}%")
        print(f"Integrated system used {integrated_result.n_trades} trades vs {baseline_result.n_trades} baseline")
        print(f"Trailing stops captured {integrated_result.exits_trail} exits ({integrated_result.exits_trail/integrated_result.n_trades*100:.1f}% of trades)")

        print("\nOutput files:")
        print("  - integrated_backtest_trades.csv")
        print("  - equity_curves.csv")
        print("  - PRODUCTION_CONFIG.md (documentation)")

        return 0

    except Exception as e:
        print(f"\nError during backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
