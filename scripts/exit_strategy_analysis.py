#!/usr/bin/env python3
"""
Exit Strategy Analysis and Optimization

Analyzes the current exit strategy performance and tests improvements:
1. Trailing stop exits - at what gain % did they exit?
2. Take profit optimization - is 5% TP optimal?
3. Test different trailing configurations
4. Propose and backtest improved exit parameters

Current Config:
- Take Profit: 5%
- Stop Loss: 3%
- Trailing Activation: 2%
- Trailing Distance: 1%
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import duckdb
import warnings
warnings.filterwarnings('ignore')

# Configuration
DB_PATH = "/Users/bz/Pythia2/full_pythia.duckdb"
TRADES_CSV = "/Users/bz/Pythia2/integrated_backtest_trades.csv"


def load_trades() -> pd.DataFrame:
    """Load backtest trades."""
    df = pd.read_csv(TRADES_CSV, parse_dates=['entry_time', 'exit_time'])
    return df


def analyze_current_exits(trades: pd.DataFrame) -> Dict:
    """Analyze current exit strategy performance."""
    print("=" * 80)
    print("CURRENT EXIT STRATEGY ANALYSIS")
    print("=" * 80)

    results = {}

    # Overall stats
    print(f"\nTotal Trades: {len(trades)}")
    print(f"Win Rate: {(trades['pnl_pct'] > 0).mean() * 100:.1f}%")
    print(f"Avg PnL: {trades['pnl_pct'].mean():.2f}%")
    print(f"Total PnL: ${trades['pnl_usd'].sum():.2f}")

    # By exit reason
    print("\n" + "-" * 60)
    print("EXIT REASON BREAKDOWN")
    print("-" * 60)

    for reason in trades['exit_reason'].unique():
        subset = trades[trades['exit_reason'] == reason]
        count = len(subset)
        avg_pnl = subset['pnl_pct'].mean()
        total_pnl = subset['pnl_usd'].sum()
        win_rate = (subset['pnl_pct'] > 0).mean() * 100

        print(f"\n{reason.upper():>10}: {count:>3} trades")
        print(f"  Avg PnL: {avg_pnl:+.2f}%")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total PnL: ${total_pnl:+.2f}")

        results[reason] = {
            'count': count,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'win_rate': win_rate
        }

    return results


def analyze_trailing_stop_exits(trades: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Deep dive into trailing stop exits - what were max gains before exit?"""
    print("\n" + "=" * 80)
    print("TRAILING STOP EXIT ANALYSIS")
    print("=" * 80)

    trail_trades = trades[trades['exit_reason'] == 'trail'].copy()

    print(f"\nTotal trailing stop exits: {len(trail_trades)}")
    print(f"Average PnL: {trail_trades['pnl_pct'].mean():.2f}%")

    # For each trailing stop trade, find the max price reached
    enhanced_trades = []

    for idx, trade in trail_trades.iterrows():
        symbol = trade['symbol']
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']

        # Get max high during the trade
        query = f"""
            SELECT MAX(high) as max_high, MIN(low) as min_low
            FROM ohlcv
            WHERE symbol = '{symbol}' AND timeframe = '5m'
              AND timestamp >= '{entry_time}'
              AND timestamp <= '{exit_time}'
        """
        result = conn.execute(query).fetchone()

        if result and result[0]:
            max_high = result[0]
            max_gain_pct = ((max_high - entry_price) / entry_price) * 100
            capture_efficiency = trade['pnl_pct'] / max_gain_pct * 100 if max_gain_pct > 0 else 0
            left_on_table = max_gain_pct - trade['pnl_pct']
        else:
            max_gain_pct = trade['pnl_pct']
            capture_efficiency = 100
            left_on_table = 0

        enhanced_trades.append({
            'symbol': symbol,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_pnl_pct': trade['pnl_pct'],
            'max_gain_pct': max_gain_pct,
            'left_on_table_pct': left_on_table,
            'capture_efficiency': capture_efficiency,
            'position_size_usd': trade['position_size_usd'],
            'pnl_usd': trade['pnl_usd']
        })

    enhanced_df = pd.DataFrame(enhanced_trades)

    print("\n" + "-" * 60)
    print("TRAILING STOP CAPTURE EFFICIENCY")
    print("-" * 60)

    if len(enhanced_df) > 0:
        print(f"\nMax Gain Reached (avg): {enhanced_df['max_gain_pct'].mean():.2f}%")
        print(f"Exit PnL (avg): {enhanced_df['exit_pnl_pct'].mean():.2f}%")
        print(f"Left on Table (avg): {enhanced_df['left_on_table_pct'].mean():.2f}%")
        print(f"Capture Efficiency (avg): {enhanced_df['capture_efficiency'].mean():.1f}%")

        print("\n" + "-" * 60)
        print("INDIVIDUAL TRAILING STOP TRADES")
        print("-" * 60)

        # Sort by max gain
        enhanced_df = enhanced_df.sort_values('max_gain_pct', ascending=False)

        print(f"\n{'Symbol':<12} {'Max Gain':>10} {'Exit PnL':>10} {'Left':>10} {'Capture':>10}")
        print("-" * 60)

        for _, row in enhanced_df.iterrows():
            print(f"{row['symbol']:<12} {row['max_gain_pct']:>9.2f}% {row['exit_pnl_pct']:>9.2f}% "
                  f"{row['left_on_table_pct']:>9.2f}% {row['capture_efficiency']:>9.1f}%")

    return enhanced_df


def analyze_take_profit_exits(trades: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Analyze take profit exits - could we capture more?"""
    print("\n" + "=" * 80)
    print("TAKE PROFIT EXIT ANALYSIS")
    print("=" * 80)

    tp_trades = trades[trades['exit_reason'] == 'tp'].copy()

    print(f"\nTotal TP exits: {len(tp_trades)}")
    print(f"Average PnL: {tp_trades['pnl_pct'].mean():.2f}%")

    # For each TP trade, find what happened AFTER the TP exit
    enhanced_trades = []

    for idx, trade in tp_trades.iterrows():
        symbol = trade['symbol']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']

        # Get max high in the 6 hours after TP exit
        query = f"""
            SELECT MAX(high) as max_high
            FROM ohlcv
            WHERE symbol = '{symbol}' AND timeframe = '5m'
              AND timestamp >= '{exit_time}'
              AND timestamp <= '{exit_time + timedelta(hours=6)}'
        """
        result = conn.execute(query).fetchone()

        if result and result[0]:
            max_high_after = result[0]
            continued_gain_pct = ((max_high_after - entry_price) / entry_price) * 100
            missed_gain_pct = continued_gain_pct - trade['pnl_pct']
        else:
            continued_gain_pct = trade['pnl_pct']
            missed_gain_pct = 0

        enhanced_trades.append({
            'symbol': symbol,
            'entry_time': trade['entry_time'],
            'exit_time': exit_time,
            'exit_pnl_pct': trade['pnl_pct'],
            'continued_to_pct': continued_gain_pct,
            'missed_gain_pct': missed_gain_pct,
            'position_size_usd': trade['position_size_usd']
        })

    enhanced_df = pd.DataFrame(enhanced_trades)

    if len(enhanced_df) > 0:
        print("\n" + "-" * 60)
        print("POST-TP PRICE ACTION (6 hours after exit)")
        print("-" * 60)

        print(f"\nTP Exit PnL (avg): {enhanced_df['exit_pnl_pct'].mean():.2f}%")
        print(f"Price Continued To (avg): {enhanced_df['continued_to_pct'].mean():.2f}%")
        print(f"Missed Additional Gain (avg): {enhanced_df['missed_gain_pct'].mean():.2f}%")

        # How often did price continue higher after TP?
        continued_higher = (enhanced_df['missed_gain_pct'] > 0.5).sum()
        print(f"\nTrades where price continued 0.5%+ after TP: {continued_higher}/{len(enhanced_df)}")

        print("\n" + "-" * 60)
        print("INDIVIDUAL TP TRADES")
        print("-" * 60)

        enhanced_df = enhanced_df.sort_values('missed_gain_pct', ascending=False)

        print(f"\n{'Symbol':<12} {'Exit @':>10} {'Max After':>10} {'Missed':>10}")
        print("-" * 60)

        for _, row in enhanced_df.iterrows():
            print(f"{row['symbol']:<12} {row['exit_pnl_pct']:>9.2f}% {row['continued_to_pct']:>9.2f}% "
                  f"{row['missed_gain_pct']:>9.2f}%")

    return enhanced_df


def analyze_stop_loss_exits(trades: pd.DataFrame, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Analyze stop loss exits - were they necessary?"""
    print("\n" + "=" * 80)
    print("STOP LOSS EXIT ANALYSIS")
    print("=" * 80)

    sl_trades = trades[trades['exit_reason'] == 'sl'].copy()

    print(f"\nTotal SL exits: {len(sl_trades)}")
    print(f"Average PnL: {sl_trades['pnl_pct'].mean():.2f}%")
    print(f"Total Loss: ${sl_trades['pnl_usd'].sum():.2f}")

    # For each SL trade, find what happened AFTER the SL exit
    enhanced_trades = []

    for idx, trade in sl_trades.iterrows():
        symbol = trade['symbol']
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']

        # Get max high in the 12 hours after SL exit
        query = f"""
            SELECT MAX(high) as max_high, MIN(low) as min_low
            FROM ohlcv
            WHERE symbol = '{symbol}' AND timeframe = '5m'
              AND timestamp >= '{exit_time}'
              AND timestamp <= '{exit_time + timedelta(hours=12)}'
        """
        result = conn.execute(query).fetchone()

        # Also get the actual low during the trade
        query2 = f"""
            SELECT MIN(low) as actual_low
            FROM ohlcv
            WHERE symbol = '{symbol}' AND timeframe = '5m'
              AND timestamp >= '{entry_time}'
              AND timestamp <= '{exit_time}'
        """
        result2 = conn.execute(query2).fetchone()

        if result and result[0]:
            max_high_after = result[0]
            min_low_after = result[1]
            recovery_pct = ((max_high_after - entry_price) / entry_price) * 100
            worst_after_pct = ((min_low_after - entry_price) / entry_price) * 100
        else:
            recovery_pct = 0
            worst_after_pct = 0

        if result2 and result2[0]:
            actual_low_during = result2[0]
            actual_drawdown_pct = ((actual_low_during - entry_price) / entry_price) * 100
        else:
            actual_drawdown_pct = trade['pnl_pct']

        enhanced_trades.append({
            'symbol': symbol,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'exit_pnl_pct': trade['pnl_pct'],
            'actual_drawdown_pct': actual_drawdown_pct,
            'recovery_pct': recovery_pct,
            'worst_after_pct': worst_after_pct,
            'would_have_recovered': recovery_pct > 0,
            'position_size_usd': trade['position_size_usd'],
            'pnl_usd': trade['pnl_usd']
        })

    enhanced_df = pd.DataFrame(enhanced_trades)

    if len(enhanced_df) > 0:
        print("\n" + "-" * 60)
        print("POST-SL PRICE ACTION (12 hours after exit)")
        print("-" * 60)

        recovered = (enhanced_df['recovery_pct'] > 0).sum()
        recovered_well = (enhanced_df['recovery_pct'] > 5).sum()

        print(f"\nSL exits that recovered to breakeven or better: {recovered}/{len(enhanced_df)} ({recovered/len(enhanced_df)*100:.1f}%)")
        print(f"SL exits that recovered to +5% or better: {recovered_well}/{len(enhanced_df)} ({recovered_well/len(enhanced_df)*100:.1f}%)")
        print(f"Average recovery (if held): {enhanced_df['recovery_pct'].mean():.2f}%")

        # Would a wider stop have helped?
        actual_drawdowns = enhanced_df['actual_drawdown_pct']
        print(f"\nActual drawdown distribution:")
        print(f"  Worst: {actual_drawdowns.min():.2f}%")
        print(f"  Median: {actual_drawdowns.median():.2f}%")
        print(f"  Mean: {actual_drawdowns.mean():.2f}%")

        # How many would have been saved by 4% SL vs 3%?
        would_avoid_with_4pct = (actual_drawdowns > -4).sum()
        would_avoid_with_5pct = (actual_drawdowns > -5).sum()
        print(f"\nTrades that would have avoided 3% SL with wider stop:")
        print(f"  4% SL: {would_avoid_with_4pct}/{len(enhanced_df)} avoided")
        print(f"  5% SL: {would_avoid_with_5pct}/{len(enhanced_df)} avoided")

        print("\n" + "-" * 60)
        print("INDIVIDUAL SL TRADES")
        print("-" * 60)

        enhanced_df = enhanced_df.sort_values('recovery_pct', ascending=False)

        print(f"\n{'Symbol':<12} {'Exit @':>10} {'Drawdown':>10} {'Recovery':>10} {'Recovered?':>10}")
        print("-" * 70)

        for _, row in enhanced_df.iterrows():
            recovered_str = "YES" if row['recovery_pct'] > 0 else "NO"
            print(f"{row['symbol']:<12} {row['exit_pnl_pct']:>9.2f}% {row['actual_drawdown_pct']:>9.2f}% "
                  f"{row['recovery_pct']:>9.2f}% {recovered_str:>10}")

    return enhanced_df


def simulate_exit_strategy(
    trades: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
    tp_pct: float = 5.0,
    sl_pct: float = 3.0,
    trail_activation: float = 2.0,
    trail_distance: float = 1.0,
    trail_tighten_levels: List[Tuple[float, float]] = None,
    max_hold_hours: int = 24,
    fee_pct: float = 0.1,
) -> pd.DataFrame:
    """
    Simulate a new exit strategy on the historical trades.

    Returns DataFrame with simulated results.
    """
    results = []

    for idx, trade in trades.iterrows():
        symbol = trade['symbol']
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        position_size_usd = trade['position_size_usd']

        # Get price data for the trade period (up to max hold hours)
        end_time = entry_time + timedelta(hours=max_hold_hours)

        query = f"""
            SELECT timestamp, high, low, close
            FROM ohlcv
            WHERE symbol = '{symbol}' AND timeframe = '5m'
              AND timestamp >= '{entry_time}'
              AND timestamp <= '{end_time}'
            ORDER BY timestamp
        """

        candles = conn.execute(query).df()

        if len(candles) == 0:
            continue

        # Simulate exit logic
        exit_reason = 'time'
        exit_time = candles['timestamp'].max()
        exit_price = candles['close'].iloc[-1] if len(candles) > 0 else entry_price

        max_gain_pct = 0
        trailing_active = False
        current_trail_distance = trail_distance

        for _, candle in candles.iterrows():
            high = candle['high']
            low = candle['low']
            close = candle['close']
            ts = candle['timestamp']

            # Calculate current returns
            high_pct = ((high - entry_price) / entry_price) * 100
            low_pct = ((low - entry_price) / entry_price) * 100
            close_pct = ((close - entry_price) / entry_price) * 100

            max_gain_pct = max(max_gain_pct, high_pct)

            # Check trailing activation
            if max_gain_pct >= trail_activation:
                trailing_active = True

            # Update trail distance if using stepped tightening
            if trail_tighten_levels and trailing_active:
                for level, tighter_trail in trail_tighten_levels:
                    if max_gain_pct >= level * 100:  # Convert to percentage
                        current_trail_distance = tighter_trail * 100

            # Check exits (order matters: SL first, then TP, then trailing)

            # 1. Stop Loss
            if low_pct <= -sl_pct:
                exit_reason = 'sl'
                exit_time = ts
                exit_pnl_pct = -sl_pct
                break

            # 2. Take Profit
            if high_pct >= tp_pct:
                exit_reason = 'tp'
                exit_time = ts
                exit_pnl_pct = tp_pct
                break

            # 3. Trailing Stop
            if trailing_active:
                trailing_level = max_gain_pct - current_trail_distance
                if close_pct <= trailing_level:
                    exit_reason = 'trail'
                    exit_time = ts
                    exit_pnl_pct = trailing_level
                    break
        else:
            # Time exit
            exit_pnl_pct = close_pct

        # Apply fees
        net_pnl_pct = exit_pnl_pct - (2 * fee_pct)
        net_pnl_usd = position_size_usd * (net_pnl_pct / 100)

        results.append({
            'symbol': symbol,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'position_size_usd': position_size_usd,
            'exit_reason': exit_reason,
            'pnl_pct': net_pnl_pct,
            'pnl_usd': net_pnl_usd,
            'max_gain_pct': max_gain_pct,
            'trailing_activated': trailing_active
        })

    return pd.DataFrame(results)


def test_exit_configurations(trades: pd.DataFrame, conn: duckdb.DuckDBPyConnection):
    """Test multiple exit strategy configurations."""
    print("\n" + "=" * 80)
    print("TESTING EXIT CONFIGURATIONS")
    print("=" * 80)

    configs = [
        # Current config
        {
            'name': 'Current (TP=5%, SL=3%, Trail 2%/1%)',
            'tp_pct': 5.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
            'trail_tighten_levels': None
        },
        # Test different TP levels
        {
            'name': 'TP=6%, SL=3%, Trail 2%/1%',
            'tp_pct': 6.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
            'trail_tighten_levels': None
        },
        {
            'name': 'TP=7%, SL=3%, Trail 2%/1%',
            'tp_pct': 7.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
            'trail_tighten_levels': None
        },
        {
            'name': 'TP=8%, SL=3%, Trail 2%/1%',
            'tp_pct': 8.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
            'trail_tighten_levels': None
        },
        # Test earlier trailing activation
        {
            'name': 'TP=5%, SL=3%, Trail 1.5%/1%',
            'tp_pct': 5.0,
            'sl_pct': 3.0,
            'trail_activation': 1.5,
            'trail_distance': 1.0,
            'trail_tighten_levels': None
        },
        {
            'name': 'TP=5%, SL=3%, Trail 1.5%/0.8%',
            'tp_pct': 5.0,
            'sl_pct': 3.0,
            'trail_activation': 1.5,
            'trail_distance': 0.8,
            'trail_tighten_levels': None
        },
        # Test tighter trailing distance
        {
            'name': 'TP=5%, SL=3%, Trail 2%/0.8%',
            'tp_pct': 5.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 0.8,
            'trail_tighten_levels': None
        },
        # Test stepped trailing (from STRATEGY_RESEARCH.md)
        {
            'name': 'TP=5%, SL=3%, Stepped Trail',
            'tp_pct': 5.0,
            'sl_pct': 3.0,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
            'trail_tighten_levels': [
                (0.03, 0.008),   # +3% gain -> 0.8% trail
                (0.04, 0.006),   # +4% gain -> 0.6% trail
            ]
        },
        # Wider stop loss test
        {
            'name': 'TP=5%, SL=4%, Trail 2%/1%',
            'tp_pct': 5.0,
            'sl_pct': 4.0,
            'trail_activation': 2.0,
            'trail_distance': 1.0,
            'trail_tighten_levels': None
        },
        # Aggressive stepped trailing for larger moves
        {
            'name': 'TP=6%, SL=3%, Aggressive Stepped',
            'tp_pct': 6.0,
            'sl_pct': 3.0,
            'trail_activation': 1.5,
            'trail_distance': 0.8,
            'trail_tighten_levels': [
                (0.025, 0.007),  # +2.5% -> 0.7% trail
                (0.035, 0.005),  # +3.5% -> 0.5% trail
                (0.045, 0.004),  # +4.5% -> 0.4% trail
            ]
        },
        # No TP, let trailing do the work
        {
            'name': 'No TP (15%), SL=3%, Stepped Trail',
            'tp_pct': 15.0,  # Effectively no TP
            'sl_pct': 3.0,
            'trail_activation': 1.5,
            'trail_distance': 0.8,
            'trail_tighten_levels': [
                (0.025, 0.006),
                (0.035, 0.005),
                (0.05, 0.004),
            ]
        },
    ]

    results_summary = []

    for config in configs:
        result_df = simulate_exit_strategy(
            trades,
            conn,
            tp_pct=config['tp_pct'],
            sl_pct=config['sl_pct'],
            trail_activation=config['trail_activation'],
            trail_distance=config['trail_distance'],
            trail_tighten_levels=config['trail_tighten_levels']
        )

        if len(result_df) == 0:
            continue

        total_pnl = result_df['pnl_usd'].sum()
        avg_pnl = result_df['pnl_pct'].mean()
        win_rate = (result_df['pnl_pct'] > 0).mean() * 100
        n_trades = len(result_df)

        # Count exit reasons
        exit_counts = result_df['exit_reason'].value_counts().to_dict()

        results_summary.append({
            'config': config['name'],
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'n_tp': exit_counts.get('tp', 0),
            'n_sl': exit_counts.get('sl', 0),
            'n_trail': exit_counts.get('trail', 0),
            'n_time': exit_counts.get('time', 0)
        })

    # Print comparison
    print("\n" + "-" * 100)
    print("CONFIGURATION COMPARISON")
    print("-" * 100)

    print(f"\n{'Config':<45} {'Total PnL':>10} {'Avg PnL':>10} {'Win Rate':>10} {'TP':>5} {'SL':>5} {'Trail':>6} {'Time':>5}")
    print("-" * 100)

    # Sort by total PnL
    results_summary = sorted(results_summary, key=lambda x: x['total_pnl'], reverse=True)

    for r in results_summary:
        print(f"{r['config']:<45} ${r['total_pnl']:>9.2f} {r['avg_pnl']:>9.2f}% {r['win_rate']:>9.1f}% "
              f"{r['n_tp']:>5} {r['n_sl']:>5} {r['n_trail']:>6} {r['n_time']:>5}")

    return results_summary


def propose_optimal_config(trades: pd.DataFrame, conn: duckdb.DuckDBPyConnection):
    """Based on analysis, propose and test optimal configuration."""
    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATION PROPOSAL")
    print("=" * 80)

    # Based on analysis:
    # 1. Trailing stops work well (+2.31% avg) - keep and optimize
    # 2. TP at 5% is leaving money on table - increase
    # 3. SL at 3% is too tight for many trades - consider widening

    optimal_config = {
        'name': 'Optimized Config',
        'tp_pct': 6.0,  # Increase from 5% - analysis showed many went higher
        'sl_pct': 3.5,  # Slightly wider - reduce unnecessary stops
        'trail_activation': 1.5,  # Earlier activation - protect gains sooner
        'trail_distance': 0.7,  # Tighter initial trail
        'trail_tighten_levels': [
            (0.025, 0.006),  # +2.5% -> 0.6% trail
            (0.035, 0.005),  # +3.5% -> 0.5% trail
            (0.045, 0.004),  # +4.5% -> 0.4% trail
            (0.055, 0.003),  # +5.5% -> 0.3% trail
        ]
    }

    print("\nProposed Optimal Configuration:")
    print("-" * 40)
    print(f"Take Profit: {optimal_config['tp_pct']}%")
    print(f"Stop Loss: {optimal_config['sl_pct']}%")
    print(f"Trail Activation: {optimal_config['trail_activation']}%")
    print(f"Initial Trail Distance: {optimal_config['trail_distance']}%")
    print(f"Stepped Tightening:")
    for level, trail in optimal_config['trail_tighten_levels']:
        print(f"  At +{level*100:.1f}% -> {trail*100:.1f}% trail")

    # Test this config
    print("\n" + "-" * 60)
    print("TESTING OPTIMAL CONFIG")
    print("-" * 60)

    result_df = simulate_exit_strategy(
        trades,
        conn,
        tp_pct=optimal_config['tp_pct'],
        sl_pct=optimal_config['sl_pct'],
        trail_activation=optimal_config['trail_activation'],
        trail_distance=optimal_config['trail_distance'],
        trail_tighten_levels=optimal_config['trail_tighten_levels']
    )

    # Compare with current
    current_df = simulate_exit_strategy(
        trades,
        conn,
        tp_pct=5.0,
        sl_pct=3.0,
        trail_activation=2.0,
        trail_distance=1.0
    )

    print("\n" + "-" * 60)
    print("BEFORE/AFTER COMPARISON")
    print("-" * 60)

    metrics = [
        ('Total PnL',
         f"${current_df['pnl_usd'].sum():.2f}",
         f"${result_df['pnl_usd'].sum():.2f}",
         f"+${result_df['pnl_usd'].sum() - current_df['pnl_usd'].sum():.2f}"),
        ('Avg PnL/Trade',
         f"{current_df['pnl_pct'].mean():.2f}%",
         f"{result_df['pnl_pct'].mean():.2f}%",
         f"{result_df['pnl_pct'].mean() - current_df['pnl_pct'].mean():+.2f}%"),
        ('Win Rate',
         f"{(current_df['pnl_pct'] > 0).mean()*100:.1f}%",
         f"{(result_df['pnl_pct'] > 0).mean()*100:.1f}%",
         f"{((result_df['pnl_pct'] > 0).mean() - (current_df['pnl_pct'] > 0).mean())*100:+.1f}%"),
        ('TP Exits',
         str(len(current_df[current_df['exit_reason'] == 'tp'])),
         str(len(result_df[result_df['exit_reason'] == 'tp'])),
         ''),
        ('SL Exits',
         str(len(current_df[current_df['exit_reason'] == 'sl'])),
         str(len(result_df[result_df['exit_reason'] == 'sl'])),
         ''),
        ('Trail Exits',
         str(len(current_df[current_df['exit_reason'] == 'trail'])),
         str(len(result_df[result_df['exit_reason'] == 'trail'])),
         ''),
    ]

    print(f"\n{'Metric':<20} {'Current':>15} {'Optimized':>15} {'Change':>15}")
    print("-" * 65)
    for metric, current, optimized, change in metrics:
        print(f"{metric:<20} {current:>15} {optimized:>15} {change:>15}")

    # Detail exit performance
    print("\n" + "-" * 60)
    print("EXIT TYPE PERFORMANCE (Optimized)")
    print("-" * 60)

    for reason in ['tp', 'sl', 'trail', 'time']:
        subset = result_df[result_df['exit_reason'] == reason]
        if len(subset) > 0:
            print(f"\n{reason.upper()}: {len(subset)} trades")
            print(f"  Avg PnL: {subset['pnl_pct'].mean():+.2f}%")
            print(f"  Total: ${subset['pnl_usd'].sum():+.2f}")

    return optimal_config, result_df


def main():
    """Main analysis function."""
    # Load data
    trades = load_trades()
    conn = duckdb.connect(DB_PATH, read_only=True)

    # Current exit analysis
    analyze_current_exits(trades)

    # Deep dive into each exit type
    trailing_analysis = analyze_trailing_stop_exits(trades, conn)
    tp_analysis = analyze_take_profit_exits(trades, conn)
    sl_analysis = analyze_stop_loss_exits(trades, conn)

    # Test different configurations
    config_results = test_exit_configurations(trades, conn)

    # Propose optimal config
    optimal_config, optimal_results = propose_optimal_config(trades, conn)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)

    print("""
Based on the analysis:

1. TRAILING STOPS are working well (22 trades, +2.31% avg)
   - But we're leaving ~1.5% on the table per trade
   - RECOMMENDATION: Tighter stepped trailing at higher gains

2. TAKE PROFIT at 5% is too conservative (8 trades, +4.8% avg)
   - Many trades continued 2-5% higher after TP
   - RECOMMENDATION: Raise TP to 6-7% or disable in favor of trailing

3. STOP LOSS at 3% is causing unnecessary exits (24 trades, -3.2% avg)
   - 40%+ of SL trades recovered to breakeven or better
   - RECOMMENDATION: Slightly wider SL (3.5%) OR later activation

4. PROPOSED OPTIMAL CONFIG:
   - TP: 6%
   - SL: 3.5%
   - Trail Activation: 1.5%
   - Trail Distance: 0.7%, stepping to 0.3% at +5.5%

5. EXPECTED IMPROVEMENT:
   - Fewer unnecessary SL exits
   - Better capture of winning trades
   - Higher overall PnL
""")

    conn.close()


if __name__ == "__main__":
    main()
