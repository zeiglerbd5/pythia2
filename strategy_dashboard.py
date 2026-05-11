#!/usr/bin/env python3
"""
Strategy Dashboard - Breakout Hunter v5.1 + Accumulation Hunter v6.0

Displays real-time status of both trading strategies.
Reads from state JSON files and auto-refreshes.

Usage:
    python strategy_dashboard.py
    python strategy_dashboard.py --refresh 5  # Custom refresh rate
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def load_state(state_file: str) -> dict:
    """Load strategy state from JSON file"""
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def format_currency(value: float, width: int = 0) -> str:
    """Format currency with color indicators"""
    if value > 0:
        s = f"\033[92m${value:+,.2f}\033[0m"  # Green
    elif value < 0:
        s = f"\033[91m${value:+,.2f}\033[0m"  # Red
    else:
        s = f"${value:,.2f}"
    return s


def format_pct(value: float) -> str:
    """Format percentage with color indicators"""
    if value > 0:
        return f"\033[92m{value:+.1f}%\033[0m"  # Green
    elif value < 0:
        return f"\033[91m{value:+.1f}%\033[0m"  # Red
    else:
        return f"{value:.1f}%"


def get_update_str(last_update: str) -> str:
    """Format update timestamp with staleness indicator"""
    try:
        update_dt = datetime.fromisoformat(last_update)
        update_str = update_dt.strftime('%H:%M:%S')
        age_seconds = (datetime.now() - update_dt).total_seconds()
        if age_seconds > 120:
            update_str += f" \033[93m({int(age_seconds)}s ago)\033[0m"
    except:
        update_str = last_update
    return update_str


def render_breakout_hunter(state: dict, width: int = 65):
    """Render the Breakout Hunter v5.1 section"""
    params = state.get('params', {})
    stats = state.get('stats', {})
    pending = state.get('pending_signals', [])
    active = state.get('active_trades', [])
    completed = state.get('completed_trades', [])
    last_update = state.get('last_update', 'Unknown')

    update_str = get_update_str(last_update)

    # Header
    print("\033[1m" + "=" * width + "\033[0m")
    print(f"\033[1m{'BREAKOUT HUNTER v5.1 - Triple Confirmation':^{width}}\033[0m")
    print(f"{'Updated: ' + update_str:^{width}}")
    print("\033[1m" + "-" * width + "\033[0m")

    # Performance summary
    total_pnl = stats.get('total_pnl', 0)
    win_rate = stats.get('win_rate', 0)
    total_trades = stats.get('total_trades', 0)
    wins = stats.get('wins', 0)
    losses = stats.get('losses', 0)

    print(f"  P&L: {format_currency(total_pnl):>15} | "
          f"Win Rate: {format_pct(win_rate):>10} ({wins}W/{losses}L) | "
          f"Trades: {total_trades}")

    # Pending signals
    if pending:
        print(f"\n  \033[93mPENDING ({len(pending)}):\033[0m")
        for sig in pending[:3]:  # Show max 3
            symbol = sig.get('symbol', '???')
            move = sig.get('initial_move_pct', 0)
            vol = sig.get('volume_ratio', 0)
            status = sig.get('status', '?')
            status_str = "T+1" if status == 'pending_t1' else "T+2"
            print(f"    {symbol:12} | +{move:.1f}% | {vol:.1f}x vol | waiting {status_str}")

    # Active trades
    if active:
        print(f"\n  \033[92mACTIVE ({len(active)}):\033[0m")
        for trade in active:
            symbol = trade.get('symbol', '???')
            entry_price = trade.get('entry_price', 0)
            highest = trade.get('highest_price', entry_price)
            trailing = trade.get('trailing_active', False)
            if entry_price > 0:
                max_return = (highest - entry_price) / entry_price * 100
            else:
                max_return = 0
            trail_str = "\033[92mTRAIL\033[0m" if trailing else ""
            print(f"    {symbol:12} | Entry: ${entry_price:.4f} | Peak: +{max_return:.1f}% {trail_str}")

    if not pending and not active:
        print(f"\n  \033[90m(watching for breakouts...)\033[0m")


def render_accumulation_hunter(state: dict, width: int = 65):
    """Render the Accumulation Hunter v6.0 section"""
    params = state.get('params', {})
    stats = state.get('stats', {})
    watch_list = state.get('watch_list', [])
    active = state.get('active_trades', [])
    completed = state.get('completed_trades', [])
    last_update = state.get('last_update', 'Unknown')

    update_str = get_update_str(last_update)

    # Header
    print("\033[1m" + "=" * width + "\033[0m")
    print(f"\033[1m{'ACCUMULATION HUNTER v6.0 - Order Book Detection':^{width}}\033[0m")
    print(f"{'Updated: ' + update_str:^{width}}")
    print("\033[1m" + "-" * width + "\033[0m")

    # Performance summary
    total_pnl = stats.get('total_pnl', 0)
    win_rate = stats.get('win_rate', 0)
    total_trades = stats.get('total_trades', 0)
    wins = stats.get('wins', 0)
    losses = stats.get('losses', 0)

    print(f"  P&L: {format_currency(total_pnl):>15} | "
          f"Win Rate: {format_pct(win_rate):>10} ({wins}W/{losses}L) | "
          f"Trades: {total_trades}")

    # Watch list - sort by signal strength
    if watch_list:
        sorted_watch = sorted(watch_list, key=lambda x: x.get('signal_strength', 0), reverse=True)

        # Count by status
        accumulating = [w for w in watch_list if w.get('status') in ('accumulating', 'ready')]
        watching = [w for w in watch_list if w.get('status') == 'watch']

        print(f"\n  \033[93mWATCH LIST ({len(watch_list)}):\033[0m "
              f"[{len(accumulating)} accumulating, {len(watching)} watching]")

        for sig in sorted_watch[:5]:  # Show top 5
            symbol = sig.get('symbol', '???')
            score = sig.get('signal_strength', 0)
            status = sig.get('status', '?')
            bar = sig.get('bar_multiple', 0)
            collapse = sig.get('ask_collapse_pct', 0) * 100
            hours = sig.get('accumulation_hours', 0)

            # Status color
            if status == 'ready':
                status_str = f"\033[92m{status:12}\033[0m"
            elif status == 'accumulating':
                status_str = f"\033[93m{status:12}\033[0m"
            else:
                status_str = f"\033[90m{status:12}\033[0m"

            # Score color
            if score >= 70:
                score_str = f"\033[92m{score:>3.0f}\033[0m"
            elif score >= 50:
                score_str = f"\033[93m{score:>3.0f}\033[0m"
            else:
                score_str = f"\033[90m{score:>3.0f}\033[0m"

            print(f"    {symbol:12} | Score:{score_str} | BAR:{bar:>4.1f}x | "
                  f"Collapse:{collapse:>3.0f}% | {status_str}")

    # Active trades
    if active:
        print(f"\n  \033[92mACTIVE ({len(active)}):\033[0m")
        for trade in active:
            symbol = trade.get('symbol', '???')
            entry_price = trade.get('entry_price', 0)
            highest = trade.get('highest_price', entry_price)
            accum_hours = trade.get('accumulation_hours', 0)
            if entry_price > 0:
                max_return = (highest - entry_price) / entry_price * 100
            else:
                max_return = 0
            print(f"    {symbol:12} | Entry: ${entry_price:.4f} | Peak: +{max_return:.1f}% | "
                  f"Accum: {accum_hours:.1f}h")

    if not watch_list and not active:
        print(f"\n  \033[90m(scanning order books...)\033[0m")


def render_combined_dashboard(breakout_state: dict, accum_state: dict):
    """Render combined dashboard for both strategies"""
    width = 70

    # Title
    print("\033[1;36m" + "=" * width + "\033[0m")
    print(f"\033[1;36m{'PYTHIA TRADING SYSTEM':^{width}}\033[0m")
    print(f"\033[1;36m{'Breakout Hunter v5.1 + Accumulation Hunter v6.0':^{width}}\033[0m")
    print("\033[1;36m" + "=" * width + "\033[0m")

    # Combined stats
    b_pnl = breakout_state.get('stats', {}).get('total_pnl', 0) if breakout_state else 0
    a_pnl = accum_state.get('stats', {}).get('total_pnl', 0) if accum_state else 0
    total_pnl = b_pnl + a_pnl

    b_trades = breakout_state.get('stats', {}).get('total_trades', 0) if breakout_state else 0
    a_trades = accum_state.get('stats', {}).get('total_trades', 0) if accum_state else 0

    b_active = len(breakout_state.get('active_trades', [])) if breakout_state else 0
    a_active = len(accum_state.get('active_trades', [])) if accum_state else 0

    print(f"\n  \033[1mCOMBINED:\033[0m P&L: {format_currency(total_pnl)} | "
          f"Trades: {b_trades + a_trades} | Active: {b_active + a_active}")
    print()

    # Breakout Hunter section
    if breakout_state:
        render_breakout_hunter(breakout_state, width)
    else:
        print("\033[1m" + "=" * width + "\033[0m")
        print(f"\033[1m{'BREAKOUT HUNTER v5.1':^{width}}\033[0m")
        print(f"\033[93m{'(not running - start with: python scripts/run_breakout_hunter.py)':^{width}}\033[0m")

    print()

    # Accumulation Hunter section
    if accum_state:
        render_accumulation_hunter(accum_state, width)
    else:
        print("\033[1m" + "=" * width + "\033[0m")
        print(f"\033[1m{'ACCUMULATION HUNTER v6.0':^{width}}\033[0m")
        print(f"\033[93m{'(not running - start with: python scripts/run_accumulation_hunter.py)':^{width}}\033[0m")

    print("\n" + "=" * width)


def main():
    parser = argparse.ArgumentParser(description='Strategy Dashboard')
    parser.add_argument('--refresh', '-r', type=int, default=10, help='Refresh rate in seconds')
    parser.add_argument('--once', '-1', action='store_true', help='Run once and exit')
    parser.add_argument('--breakout-state', default='breakout_hunter_state.json',
                        help='Breakout hunter state file')
    parser.add_argument('--accum-state', default='accumulation_hunter_state.json',
                        help='Accumulation hunter state file')
    args = parser.parse_args()

    print("\033[1mPythia Strategy Dashboard\033[0m")
    print(f"Breakout Hunter: {args.breakout_state}")
    print(f"Accumulation Hunter: {args.accum_state}")
    print(f"Refresh rate: {args.refresh}s")
    print("Press Ctrl+C to exit\n")
    time.sleep(1)

    try:
        while True:
            clear_screen()

            breakout_state = load_state(args.breakout_state)
            accum_state = load_state(args.accum_state)

            render_combined_dashboard(breakout_state, accum_state)

            if args.once:
                break

            # Countdown
            for i in range(args.refresh, 0, -1):
                print(f"\rNext refresh in {i}s... (Ctrl+C to exit)", end='', flush=True)
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
