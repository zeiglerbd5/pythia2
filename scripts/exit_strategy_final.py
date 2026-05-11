#!/usr/bin/env python3
"""
Exit Strategy Final Analysis - Deep Dive into Stop Loss Problem

The key issue: 24 stop losses are costing us $815.75.
Can we reduce this without increasing risk?
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


def analyze_sl_trades_deeply(trades: pd.DataFrame, conn: duckdb.DuckDBPyConnection):
    """Deep analysis of stop loss trades."""
    print("=" * 100)
    print("DEEP STOP LOSS ANALYSIS")
    print("=" * 100)

    sl_trades = trades[trades['exit_reason'] == 'sl'].copy()

    print(f"\nTotal SL exits: {len(sl_trades)}")
    print(f"Total SL losses: ${sl_trades['pnl_usd'].sum():.2f}")

    # For each SL trade, analyze the price path
    detailed_analysis = []

    for idx, trade in sl_trades.iterrows():
        symbol = trade['symbol']
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']
        position_size = trade['position_size_usd']

        # Get price data for the trade
        query = f"""
            SELECT timestamp, high, low, close
            FROM ohlcv
            WHERE symbol = '{symbol}' AND timeframe = '5m'
              AND timestamp >= '{entry_time}'
              AND timestamp <= '{entry_time + timedelta(hours=24)}'
            ORDER BY timestamp
        """
        candles = conn.execute(query).df()

        if len(candles) == 0:
            continue

        # Find key metrics
        max_high = candles['high'].max()
        min_low = candles['low'].min()
        final_close = candles['close'].iloc[-1]

        max_gain_pct = ((max_high - entry_price) / entry_price) * 100
        max_drawdown_pct = ((min_low - entry_price) / entry_price) * 100
        final_pnl_pct = ((final_close - entry_price) / entry_price) * 100

        # When did max gain occur vs SL hit?
        max_idx = candles['high'].idxmax()
        sl_hit_idx = None
        for i, row in candles.iterrows():
            if ((row['low'] - entry_price) / entry_price) * 100 <= -3.0:
                sl_hit_idx = i
                break

        # Did we hit max gain before or after SL?
        max_before_sl = max_idx < sl_hit_idx if sl_hit_idx is not None else False

        # Count how many times we went positive
        went_positive = False
        max_positive = 0
        for _, row in candles.iterrows():
            pct = ((row['high'] - entry_price) / entry_price) * 100
            if pct > 0:
                went_positive = True
                max_positive = max(max_positive, pct)

        detailed_analysis.append({
            'symbol': symbol,
            'entry_time': entry_time,
            'position_size': position_size,
            'max_gain_pct': max_gain_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'final_pnl_pct': final_pnl_pct,
            'went_positive': went_positive,
            'max_positive_before_sl': max_positive,
            'max_before_sl': max_before_sl,
            'would_recover': final_pnl_pct > -3.0
        })

    df_analysis = pd.DataFrame(detailed_analysis)

    print("\n" + "-" * 80)
    print("SL TRADE CHARACTERISTICS")
    print("-" * 80)

    went_positive = df_analysis['went_positive'].sum()
    max_before_sl = df_analysis['max_before_sl'].sum()
    would_recover = df_analysis['would_recover'].sum()

    print(f"\nTraded that went positive before SL: {went_positive}/{len(df_analysis)}")
    print(f"Trades that hit max BEFORE SL: {max_before_sl}/{len(df_analysis)}")
    print(f"Trades that would have recovered (24h): {would_recover}/{len(df_analysis)}")

    print(f"\nMax gain before SL hit (avg): {df_analysis['max_positive_before_sl'].mean():.2f}%")
    print(f"Final PnL if held 24h (avg): {df_analysis['final_pnl_pct'].mean():.2f}%")

    # Distribution of max gains before SL
    print("\n" + "-" * 80)
    print("MAX GAIN BEFORE HITTING STOP LOSS")
    print("-" * 80)

    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, float('inf')]
    labels = ['0-0.5%', '0.5-1%', '1-1.5%', '1.5-2%', '2-2.5%', '2.5-3%', '>3%']
    df_analysis['gain_bin'] = pd.cut(df_analysis['max_positive_before_sl'], bins=bins, labels=labels)

    print("\nDistribution of max gains before SL:")
    print(df_analysis['gain_bin'].value_counts().sort_index())

    # What if we used a trailing stop earlier to protect gains?
    print("\n" + "-" * 80)
    print("HYPOTHETICAL: EARLIER TRAILING ACTIVATION")
    print("-" * 80)

    # If we activated trail at 1% with 0.5% trail, how many SL exits would we save?
    saved_with_early_trail = 0
    additional_capture = 0

    for _, row in df_analysis.iterrows():
        max_pos = row['max_positive_before_sl']
        if max_pos >= 1.0:
            # Would have been protected at 1% - 0.5% = 0.5%
            saved_with_early_trail += 1
            additional_capture += (0.5 - (-3.0)) * row['position_size'] / 100

    print(f"\nWith trail activation at 1% (0.5% trail):")
    print(f"  SL exits potentially saved: {saved_with_early_trail}/{len(df_analysis)}")
    print(f"  Additional PnL captured: ~${additional_capture:.2f}")

    return df_analysis


def test_breakeven_stop_strategy(trades: pd.DataFrame, conn: duckdb.DuckDBPyConnection):
    """Test a breakeven stop strategy - move SL to breakeven after X% gain."""
    print("\n" + "=" * 100)
    print("BREAKEVEN STOP STRATEGY TEST")
    print("=" * 100)

    configs = [
        {'be_trigger': 1.0, 'be_level': 0.0, 'name': 'BE @ +1%'},
        {'be_trigger': 1.5, 'be_level': 0.0, 'name': 'BE @ +1.5%'},
        {'be_trigger': 2.0, 'be_level': 0.0, 'name': 'BE @ +2%'},
        {'be_trigger': 1.0, 'be_level': 0.2, 'name': '+0.2% @ +1%'},
        {'be_trigger': 1.5, 'be_level': 0.3, 'name': '+0.3% @ +1.5%'},
        {'be_trigger': 2.0, 'be_level': 0.5, 'name': '+0.5% @ +2%'},
    ]

    for config in configs:
        results = simulate_with_be_stop(trades, conn, config)
        print(f"\n{config['name']}: {results['summary']}")


def simulate_with_be_stop(
    trades: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
    config: Dict
) -> Dict:
    """Simulate trades with breakeven stop feature."""

    be_trigger = config['be_trigger']
    be_level = config['be_level']

    results = []

    for idx, trade in trades.iterrows():
        symbol = trade['symbol']
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        position_size = trade['position_size_usd']

        query = f"""
            SELECT timestamp, high, low, close
            FROM ohlcv
            WHERE symbol = '{symbol}' AND timeframe = '5m'
              AND timestamp >= '{entry_time}'
              AND timestamp <= '{entry_time + timedelta(hours=24)}'
            ORDER BY timestamp
        """
        candles = conn.execute(query).df()

        if len(candles) == 0:
            continue

        # Simulation with BE stop
        max_gain = 0
        be_active = False
        current_sl = -3.0  # Initial SL
        trail_active = False
        trail_distance = 0.6
        tp_level = 7.0

        exit_reason = 'time'
        exit_pnl = 0

        for _, candle in candles.iterrows():
            high_pct = ((candle['high'] - entry_price) / entry_price) * 100
            low_pct = ((candle['low'] - entry_price) / entry_price) * 100
            close_pct = ((candle['close'] - entry_price) / entry_price) * 100

            max_gain = max(max_gain, high_pct)

            # Activate BE stop
            if max_gain >= be_trigger and not be_active:
                be_active = True
                current_sl = be_level

            # Activate trailing
            if max_gain >= 2.0:
                trail_active = True

            # Check exits
            if low_pct <= current_sl:
                exit_reason = 'be' if be_active else 'sl'
                exit_pnl = current_sl
                break

            if high_pct >= tp_level:
                exit_reason = 'tp'
                exit_pnl = tp_level
                break

            if trail_active:
                trail_level = max_gain - trail_distance
                if close_pct <= trail_level:
                    exit_reason = 'trail'
                    exit_pnl = trail_level
                    break

        else:
            exit_pnl = close_pct

        net_pnl = exit_pnl - 0.2  # fees
        results.append({
            'exit_reason': exit_reason,
            'pnl_pct': net_pnl,
            'pnl_usd': position_size * net_pnl / 100
        })

    df_results = pd.DataFrame(results)
    total_pnl = df_results['pnl_usd'].sum()
    win_rate = (df_results['pnl_pct'] > 0).mean() * 100

    exit_counts = df_results['exit_reason'].value_counts().to_dict()

    return {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'exits': exit_counts,
        'summary': f"PnL: ${total_pnl:+.2f}, Win: {win_rate:.1f}%, "
                   f"SL: {exit_counts.get('sl', 0)}, BE: {exit_counts.get('be', 0)}, "
                   f"Trail: {exit_counts.get('trail', 0)}, TP: {exit_counts.get('tp', 0)}"
    }


def final_optimized_simulation(trades: pd.DataFrame, conn: duckdb.DuckDBPyConnection):
    """Test the final optimized configuration."""
    print("\n" + "=" * 100)
    print("FINAL OPTIMIZED CONFIGURATION TEST")
    print("=" * 100)

    # Parameters from best-performing configs combined with BE stop
    results = []

    for idx, trade in trades.iterrows():
        symbol = trade['symbol']
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        position_size = trade['position_size_usd']

        query = f"""
            SELECT timestamp, high, low, close
            FROM ohlcv
            WHERE symbol = '{symbol}' AND timeframe = '5m'
              AND timestamp >= '{entry_time}'
              AND timestamp <= '{entry_time + timedelta(hours=24)}'
            ORDER BY timestamp
        """
        candles = conn.execute(query).df()

        if len(candles) == 0:
            continue

        # OPTIMIZED PARAMETERS
        initial_sl = -3.0
        be_trigger = 1.5  # Move to BE after 1.5% gain
        be_level = 0.2    # Lock in 0.2% instead of breakeven
        trail_activation = 2.0
        trail_distance = 0.6
        trail_tighten_levels = [
            (3.0, 0.5),
            (4.0, 0.4),
            (5.0, 0.3),
        ]
        tp_level = 7.0

        max_gain = 0
        current_sl = initial_sl
        be_active = False
        trail_active = False
        current_trail = trail_distance

        exit_reason = 'time'
        exit_pnl = 0

        for _, candle in candles.iterrows():
            high_pct = ((candle['high'] - entry_price) / entry_price) * 100
            low_pct = ((candle['low'] - entry_price) / entry_price) * 100
            close_pct = ((candle['close'] - entry_price) / entry_price) * 100

            max_gain = max(max_gain, high_pct)

            # Activate BE stop
            if max_gain >= be_trigger and not be_active:
                be_active = True
                current_sl = be_level

            # Activate trailing
            if max_gain >= trail_activation:
                trail_active = True

            # Tighten trail
            for level, tighter in trail_tighten_levels:
                if max_gain >= level:
                    current_trail = tighter

            # Check exits
            if low_pct <= current_sl:
                exit_reason = 'be_stop' if be_active else 'sl'
                exit_pnl = current_sl
                break

            if high_pct >= tp_level:
                exit_reason = 'tp'
                exit_pnl = tp_level
                break

            if trail_active:
                trail_level = max_gain - current_trail
                if close_pct <= trail_level:
                    exit_reason = 'trail'
                    exit_pnl = trail_level
                    break

        else:
            exit_pnl = close_pct

        net_pnl = exit_pnl - 0.2  # fees
        results.append({
            'symbol': symbol,
            'exit_reason': exit_reason,
            'pnl_pct': net_pnl,
            'pnl_usd': position_size * net_pnl / 100,
            'max_gain': max_gain
        })

    df_results = pd.DataFrame(results)

    # Summary
    print("\nOPTIMIZED CONFIGURATION:")
    print("-" * 60)
    print(f"  Initial Stop Loss: 3.0%")
    print(f"  Breakeven Trigger: +1.5% -> move SL to +0.2%")
    print(f"  Trail Activation: +2.0%")
    print(f"  Trail Distance: 0.6% (tightening at +3%, +4%, +5%)")
    print(f"  Take Profit: 7.0%")

    print("\nRESULTS:")
    print("-" * 60)

    total_pnl = df_results['pnl_usd'].sum()
    avg_pnl = df_results['pnl_pct'].mean()
    win_rate = (df_results['pnl_pct'] > 0).mean() * 100

    print(f"  Total PnL: ${total_pnl:+.2f}")
    print(f"  Avg PnL/Trade: {avg_pnl:+.2f}%")
    print(f"  Win Rate: {win_rate:.1f}%")

    exit_counts = df_results['exit_reason'].value_counts().to_dict()
    print(f"\nExit Distribution:")
    for reason, count in sorted(exit_counts.items()):
        subset = df_results[df_results['exit_reason'] == reason]
        print(f"  {reason:>10}: {count:>3} trades, avg {subset['pnl_pct'].mean():+.2f}%, "
              f"total ${subset['pnl_usd'].sum():+.2f}")

    # Compare to current
    print("\n" + "-" * 60)
    print("COMPARISON TO CURRENT")
    print("-" * 60)

    current_pnl = 121.08  # From the CSV analysis
    current_sl_count = 24

    print(f"  Current Total PnL: ${current_pnl:+.2f}")
    print(f"  Optimized Total PnL: ${total_pnl:+.2f}")
    print(f"  Improvement: ${total_pnl - current_pnl:+.2f}")
    print(f"\n  Current SL Exits: {current_sl_count}")
    print(f"  Optimized SL Exits: {exit_counts.get('sl', 0)}")
    print(f"  BE Stop Exits: {exit_counts.get('be_stop', 0)}")
    print(f"  SL+BE Total: {exit_counts.get('sl', 0) + exit_counts.get('be_stop', 0)}")

    return df_results


def generate_code_recommendations(best_pnl: float, current_pnl: float):
    """Generate code recommendations for the production system."""
    print("\n" + "=" * 100)
    print("CODE RECOMMENDATIONS FOR PRODUCTION")
    print("=" * 100)

    print("""
Update the following in `/Users/bz/Pythia2/src/models/integrated_production_system.py`:

1. In ProductionConfig class (lines ~70-85):
```python
    # Trading parameters - OPTIMIZED
    initial_capital: float = 10000.0
    max_position_pct: float = 20.0
    max_concurrent_positions: int = 3
    take_profit_pct: float = 7.0          # Changed from 5.0
    stop_loss_pct: float = 3.0            # Keep same
    trailing_stop_activation: float = 2.0  # Keep same
    trailing_stop_distance: float = 0.6    # Changed from 1.0
    max_hold_hours: int = 24
    fee_pct: float = 0.1
    slippage_pct: float = 0.1

    # NEW: Breakeven stop parameters
    breakeven_trigger: float = 1.5         # Move to BE after this gain
    breakeven_level: float = 0.2           # Lock in this much (not exact BE)

    # NEW: Stepped trailing tightening
    trail_tighten_levels: List[Tuple[float, float]] = field(default_factory=lambda: [
        (3.0, 0.5),   # At +3% gain, tighten to 0.5%
        (4.0, 0.4),   # At +4% gain, tighten to 0.4%
        (5.0, 0.3),   # At +5% gain, tighten to 0.3%
    ])
```

2. In _check_exits method, add breakeven stop logic:
```python
    # Check breakeven stop activation
    if trade.max_favorable >= self.config.breakeven_trigger:
        if not hasattr(trade, 'be_active') or not trade.be_active:
            trade.be_active = True
            trade.current_sl = self.config.breakeven_level

    # Use dynamic SL level
    sl_level = getattr(trade, 'current_sl', -self.config.stop_loss_pct)

    # SL check
    if min_pnl <= sl_level:
        exit_reason = "be" if getattr(trade, 'be_active', False) else "sl"
        exit_pnl = sl_level
```

3. Add stepped trailing tightening:
```python
    # Update trail distance based on max gain
    current_trail = self.config.trailing_stop_distance
    for level, tighter in self.config.trail_tighten_levels:
        if trade.max_favorable >= level:
            current_trail = tighter

    trailing_level = trade.max_favorable - current_trail
```
""")

    print(f"\n{'='*60}")
    print("EXPECTED IMPROVEMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Current PnL:   ${current_pnl:+.2f}")
    print(f"  Optimized PnL: ${best_pnl:+.2f}")
    print(f"  Improvement:   ${best_pnl - current_pnl:+.2f} ({(best_pnl - current_pnl) / abs(current_pnl) * 100:+.1f}% better)")


def main():
    """Main analysis."""
    trades = load_trades()
    conn = duckdb.connect(DB_PATH, read_only=True)

    # Deep SL analysis
    sl_analysis = analyze_sl_trades_deeply(trades, conn)

    # Test BE stop strategy
    test_breakeven_stop_strategy(trades, conn)

    # Final optimized simulation
    optimized_results = final_optimized_simulation(trades, conn)

    # Generate code recommendations
    optimized_pnl = optimized_results['pnl_usd'].sum()
    current_pnl = 121.08  # Actual from trades
    generate_code_recommendations(optimized_pnl, current_pnl)

    conn.close()


if __name__ == "__main__":
    main()
