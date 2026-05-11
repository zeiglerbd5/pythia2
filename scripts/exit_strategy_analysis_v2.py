#!/usr/bin/env python3
"""
Exit Strategy Analysis and Optimization - V2

More accurate simulation that matches the actual backtest behavior.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import duckdb
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "/Users/bz/Pythia2/full_pythia.duckdb"
TRADES_CSV = "/Users/bz/Pythia2/integrated_backtest_trades.csv"


def load_trades() -> pd.DataFrame:
    """Load backtest trades."""
    df = pd.read_csv(TRADES_CSV, parse_dates=['entry_time', 'exit_time'])
    return df


def get_ohlcv_for_trade(conn, symbol: str, entry_time: datetime, max_hours: int = 24) -> pd.DataFrame:
    """Get OHLCV data for a trade period."""
    end_time = entry_time + timedelta(hours=max_hours)
    query = f"""
        SELECT timestamp, open, high, low, close
        FROM ohlcv
        WHERE symbol = '{symbol}' AND timeframe = '5m'
          AND timestamp >= '{entry_time}'
          AND timestamp <= '{end_time}'
        ORDER BY timestamp
    """
    return conn.execute(query).df()


def simulate_exit(
    candles: pd.DataFrame,
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    trail_activation: float,
    trail_distance: float,
    trail_tighten_levels: Optional[List[Tuple[float, float]]] = None,
    fee_pct: float = 0.1,
) -> Dict:
    """
    Simulate exit strategy on candle data.
    Returns exit info dict.
    """
    if len(candles) == 0:
        return None

    max_gain_pct = 0
    trailing_active = False
    current_trail_distance = trail_distance

    for _, candle in candles.iterrows():
        high = candle['high']
        low = candle['low']
        close = candle['close']
        ts = candle['timestamp']

        # Calculate returns as percentages
        high_pct = ((high - entry_price) / entry_price) * 100
        low_pct = ((low - entry_price) / entry_price) * 100
        close_pct = ((close - entry_price) / entry_price) * 100

        # Update max gain seen
        max_gain_pct = max(max_gain_pct, high_pct)

        # Check trailing activation
        if max_gain_pct >= trail_activation:
            trailing_active = True

        # Update trail distance if using stepped tightening (levels are in decimal, trail_distance in pct)
        if trail_tighten_levels and trailing_active:
            for level, tighter_trail in trail_tighten_levels:
                if max_gain_pct >= level:  # level is in percentage
                    current_trail_distance = tighter_trail

        # Check exits in priority order

        # 1. Stop Loss - check if low went below SL level
        if low_pct <= -sl_pct:
            gross_pnl = -sl_pct
            net_pnl = gross_pnl - (2 * fee_pct)
            return {
                'exit_reason': 'sl',
                'exit_time': ts,
                'pnl_pct': net_pnl,
                'max_gain_pct': max_gain_pct,
                'trailing_activated': trailing_active
            }

        # 2. Take Profit - check if high went above TP level
        if high_pct >= tp_pct:
            gross_pnl = tp_pct
            net_pnl = gross_pnl - (2 * fee_pct)
            return {
                'exit_reason': 'tp',
                'exit_time': ts,
                'pnl_pct': net_pnl,
                'max_gain_pct': max_gain_pct,
                'trailing_activated': True  # Must have been activated to hit TP
            }

        # 3. Trailing Stop - check if close dropped below trail level
        if trailing_active:
            trailing_level = max_gain_pct - current_trail_distance
            if close_pct <= trailing_level:
                gross_pnl = trailing_level
                net_pnl = gross_pnl - (2 * fee_pct)
                return {
                    'exit_reason': 'trail',
                    'exit_time': ts,
                    'pnl_pct': net_pnl,
                    'max_gain_pct': max_gain_pct,
                    'trailing_activated': True
                }

    # Time exit - use last close
    last_close = candles['close'].iloc[-1]
    close_pct = ((last_close - entry_price) / entry_price) * 100
    net_pnl = close_pct - (2 * fee_pct)

    return {
        'exit_reason': 'time',
        'exit_time': candles['timestamp'].iloc[-1],
        'pnl_pct': net_pnl,
        'max_gain_pct': max_gain_pct,
        'trailing_activated': trailing_active
    }


def run_simulation(
    trades: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
    config: Dict
) -> pd.DataFrame:
    """Run exit strategy simulation with given config."""
    results = []

    for idx, trade in trades.iterrows():
        symbol = trade['symbol']
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        position_size_usd = trade['position_size_usd']

        candles = get_ohlcv_for_trade(conn, symbol, entry_time)

        if len(candles) == 0:
            continue

        exit_info = simulate_exit(
            candles,
            entry_price,
            tp_pct=config['tp_pct'],
            sl_pct=config['sl_pct'],
            trail_activation=config['trail_activation'],
            trail_distance=config['trail_distance'],
            trail_tighten_levels=config.get('trail_tighten_levels'),
            fee_pct=config.get('fee_pct', 0.1)
        )

        if exit_info:
            pnl_usd = position_size_usd * (exit_info['pnl_pct'] / 100)
            results.append({
                'symbol': symbol,
                'entry_time': entry_time,
                'exit_time': exit_info['exit_time'],
                'entry_price': entry_price,
                'position_size_usd': position_size_usd,
                'exit_reason': exit_info['exit_reason'],
                'pnl_pct': exit_info['pnl_pct'],
                'pnl_usd': pnl_usd,
                'max_gain_pct': exit_info['max_gain_pct'],
                'trailing_activated': exit_info['trailing_activated']
            })

    return pd.DataFrame(results)


def summarize_results(result_df: pd.DataFrame, name: str) -> Dict:
    """Summarize simulation results."""
    if len(result_df) == 0:
        return {}

    total_pnl = result_df['pnl_usd'].sum()
    avg_pnl = result_df['pnl_pct'].mean()
    win_rate = (result_df['pnl_pct'] > 0).mean() * 100
    n_trades = len(result_df)

    # By exit reason
    exit_counts = result_df['exit_reason'].value_counts().to_dict()

    # PnL by exit reason
    pnl_by_exit = {}
    for reason in ['tp', 'sl', 'trail', 'time']:
        subset = result_df[result_df['exit_reason'] == reason]
        if len(subset) > 0:
            pnl_by_exit[reason] = {
                'count': len(subset),
                'avg_pnl': subset['pnl_pct'].mean(),
                'total_pnl': subset['pnl_usd'].sum()
            }

    return {
        'name': name,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'win_rate': win_rate,
        'n_trades': n_trades,
        'n_tp': exit_counts.get('tp', 0),
        'n_sl': exit_counts.get('sl', 0),
        'n_trail': exit_counts.get('trail', 0),
        'n_time': exit_counts.get('time', 0),
        'pnl_by_exit': pnl_by_exit
    }


def main():
    """Main analysis."""
    trades = load_trades()
    conn = duckdb.connect(DB_PATH, read_only=True)

    print("=" * 100)
    print("EXIT STRATEGY OPTIMIZATION ANALYSIS")
    print("=" * 100)

    # Define configurations to test
    configs = [
        # Current config (baseline)
        {
            'name': 'CURRENT: TP=5%, SL=3%, Trail@2%/1%',
            'tp_pct': 5.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
        },
        # Higher TP tests
        {
            'name': 'TP=6%, SL=3%, Trail@2%/1%',
            'tp_pct': 6.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
        },
        {
            'name': 'TP=7%, SL=3%, Trail@2%/1%',
            'tp_pct': 7.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
        },
        {
            'name': 'TP=8%, SL=3%, Trail@2%/1%',
            'tp_pct': 8.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
        },
        # Earlier trail activation
        {
            'name': 'TP=5%, SL=3%, Trail@1.5%/1%',
            'tp_pct': 5.0,
            'sl_pct': 3.0,
            'trail_activation': 1.5,
            'trail_distance': 1.0,
        },
        {
            'name': 'TP=5%, SL=3%, Trail@1.5%/0.8%',
            'tp_pct': 5.0,
            'sl_pct': 3.0,
            'trail_activation': 1.5,
            'trail_distance': 0.8,
        },
        # Tighter trail distance
        {
            'name': 'TP=5%, SL=3%, Trail@2%/0.8%',
            'tp_pct': 5.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 0.8,
        },
        {
            'name': 'TP=5%, SL=3%, Trail@2%/0.6%',
            'tp_pct': 5.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 0.6,
        },
        # Stepped trailing from STRATEGY_RESEARCH.md
        {
            'name': 'TP=5%, Stepped v1 (3%->0.8%, 4%->0.6%)',
            'tp_pct': 5.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
            'trail_tighten_levels': [
                (3.0, 0.8),   # +3% gain -> 0.8% trail
                (4.0, 0.6),   # +4% gain -> 0.6% trail
            ]
        },
        # More aggressive stepped
        {
            'name': 'TP=6%, Stepped v2 (2.5%->0.7%, 3.5%->0.5%)',
            'tp_pct': 6.0,
            'sl_pct': 3.0,
            'trail_activation': 1.5,
            'trail_distance': 0.8,
            'trail_tighten_levels': [
                (2.5, 0.7),
                (3.5, 0.5),
            ]
        },
        # From research: 30%->8%, 50%->6%, 70%->5% (scaled down for our 5% TP)
        # Let's try proportional: 3%->0.8%, 5%->0.6%
        {
            'name': 'TP=6%, Stepped v3 (research-based)',
            'tp_pct': 6.0,
            'sl_pct': 3.0,
            'trail_activation': 1.5,
            'trail_distance': 1.0,
            'trail_tighten_levels': [
                (2.0, 0.8),   # +2% -> 0.8% trail
                (3.0, 0.6),   # +3% -> 0.6% trail
                (4.0, 0.5),   # +4% -> 0.5% trail
            ]
        },
        # No TP - let trailing capture everything
        {
            'name': 'NO TP (10%), Stepped Trail',
            'tp_pct': 10.0,
            'sl_pct': 3.0,
            'trail_activation': 1.5,
            'trail_distance': 0.8,
            'trail_tighten_levels': [
                (2.5, 0.6),
                (3.5, 0.5),
                (5.0, 0.4),
            ]
        },
        # Test wider SL
        {
            'name': 'TP=5%, SL=3.5%, Trail@2%/1%',
            'tp_pct': 5.0,
            'sl_pct': 3.5,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
        },
        {
            'name': 'TP=5%, SL=4%, Trail@2%/1%',
            'tp_pct': 5.0,
            'sl_pct': 4.0,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
        },
        # Combined optimizations
        {
            'name': 'OPT1: TP=6%, SL=3%, Trail@1.5%/0.8%',
            'tp_pct': 6.0,
            'sl_pct': 3.0,
            'trail_activation': 1.5,
            'trail_distance': 0.8,
        },
        {
            'name': 'OPT2: TP=7%, SL=3%, Trail@2%/0.6%',
            'tp_pct': 7.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 0.6,
        },
        {
            'name': 'OPT3: TP=6%, SL=3%, Stepped+Early',
            'tp_pct': 6.0,
            'sl_pct': 3.0,
            'trail_activation': 1.5,
            'trail_distance': 0.8,
            'trail_tighten_levels': [
                (2.5, 0.6),
                (3.5, 0.5),
                (4.5, 0.4),
            ]
        },
    ]

    # Run all simulations
    results = []
    for config in configs:
        result_df = run_simulation(trades, conn, config)
        summary = summarize_results(result_df, config['name'])
        results.append(summary)

    # Print results table
    print("\n" + "-" * 120)
    print("CONFIGURATION COMPARISON (sorted by Total PnL)")
    print("-" * 120)

    # Sort by total PnL
    results = sorted(results, key=lambda x: x.get('total_pnl', 0), reverse=True)

    print(f"\n{'Config':<50} {'Total PnL':>12} {'Avg PnL':>10} {'Win%':>8} {'TP':>5} {'SL':>5} {'Trail':>6} {'Time':>5}")
    print("-" * 120)

    baseline_pnl = None
    for r in results:
        if 'CURRENT' in r['name']:
            baseline_pnl = r['total_pnl']
            marker = "  <<< BASELINE"
        elif baseline_pnl is not None:
            diff = r['total_pnl'] - baseline_pnl
            marker = f"  ({diff:+.2f})"
        else:
            marker = ""

        print(f"{r['name']:<50} ${r['total_pnl']:>10.2f} {r['avg_pnl']:>9.2f}% {r['win_rate']:>7.1f}% "
              f"{r['n_tp']:>5} {r['n_sl']:>5} {r['n_trail']:>6} {r['n_time']:>5}{marker}")

    # Find best config
    best = results[0]
    print("\n" + "=" * 100)
    print("BEST CONFIGURATION")
    print("=" * 100)
    print(f"\n{best['name']}")
    print(f"  Total PnL: ${best['total_pnl']:.2f}")
    print(f"  Avg PnL/Trade: {best['avg_pnl']:.2f}%")
    print(f"  Win Rate: {best['win_rate']:.1f}%")

    # Compare best vs current
    current = next(r for r in results if 'CURRENT' in r['name'])

    print("\n" + "-" * 60)
    print("IMPROVEMENT OVER CURRENT")
    print("-" * 60)
    print(f"  PnL Improvement: ${best['total_pnl'] - current['total_pnl']:+.2f}")
    print(f"  Win Rate Change: {best['win_rate'] - current['win_rate']:+.1f}%")
    print(f"  SL Exits: {current['n_sl']} -> {best['n_sl']}")
    print(f"  Trail Exits: {current['n_trail']} -> {best['n_trail']}")

    # Detailed analysis of top 3 configs
    print("\n" + "=" * 100)
    print("TOP 3 CONFIGURATIONS - DETAILED BREAKDOWN")
    print("=" * 100)

    for i, r in enumerate(results[:3]):
        print(f"\n{'='*60}")
        print(f"#{i+1}: {r['name']}")
        print(f"{'='*60}")
        print(f"\nTotal PnL: ${r['total_pnl']:.2f}")
        print(f"Avg PnL: {r['avg_pnl']:.2f}%")
        print(f"Win Rate: {r['win_rate']:.1f}%")
        print(f"\nExit Breakdown:")
        for reason in ['tp', 'sl', 'trail', 'time']:
            if reason in r.get('pnl_by_exit', {}):
                data = r['pnl_by_exit'][reason]
                print(f"  {reason.upper():>6}: {data['count']:>3} trades, "
                      f"avg {data['avg_pnl']:+.2f}%, "
                      f"total ${data['total_pnl']:+.2f}")

    # Generate recommended config
    print("\n" + "=" * 100)
    print("RECOMMENDED CONFIGURATION FOR PRODUCTION")
    print("=" * 100)

    # Based on analysis, recommend the best config
    print(f"""
Based on the analysis, the recommended exit configuration is:

{best['name']}

Key parameters to update in integrated_production_system.py:

    take_profit_pct: {next(c['tp_pct'] for c in configs if c['name'] == best['name'])}
    stop_loss_pct: {next(c['sl_pct'] for c in configs if c['name'] == best['name'])}
    trailing_stop_activation: {next(c['trail_activation'] for c in configs if c['name'] == best['name'])}
    trailing_stop_distance: {next(c['trail_distance'] for c in configs if c['name'] == best['name'])}

Expected Results:
- Total PnL: ${best['total_pnl']:.2f} (vs ${current['total_pnl']:.2f} current)
- Improvement: ${best['total_pnl'] - current['total_pnl']:+.2f}
- Win Rate: {best['win_rate']:.1f}% (vs {current['win_rate']:.1f}% current)
""")

    conn.close()


if __name__ == "__main__":
    main()
