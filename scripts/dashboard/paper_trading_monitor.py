#!/usr/bin/env python3
"""
Loading Strategy Monitor - Terminal Dashboard

Displays real-time status of the two-phase loading strategy:
- Portfolio stats (cash, P&L, win/loss)
- Open positions (Phase 1 and Phase 2)
- Top loading scores across all symbols
- Recent closed trades
- Active scanner alerts (loading → triggered → confirmed)

Updates every 30 seconds. Run in a separate terminal.

Usage:
    python scripts/dashboard/paper_trading_monitor.py
"""

import os
import sys
import time
import json
import re
import subprocess
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOADING_STATE = PROJECT_ROOT / 'data' / 'loading_trader_state.json'
FEATURE_BUFFER = PROJECT_ROOT / 'data' / 'feature_buffer.db'
LIVE_DB = PROJECT_ROOT / 'data' / 'pythia.duckdb'
LOG_DIR = PROJECT_ROOT / 'logs'


class C:
    """ANSI colors."""
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BG_GREEN = '\033[42m'
    BG_RED = '\033[41m'
    BG_YELLOW = '\033[43m'
    WHITE = '\033[97m'


def clear():
    os.system('clear')


def pnl_color(val):
    if val > 0:
        return f"{C.GREEN}+${val:,.2f}{C.RESET}"
    elif val < 0:
        return f"{C.RED}-${abs(val):,.2f}{C.RESET}"
    return f"${val:,.2f}"


def pct_color(val):
    if val > 0:
        return f"{C.GREEN}+{val:.1f}%{C.RESET}"
    elif val < 0:
        return f"{C.RED}{val:.1f}%{C.RESET}"
    return f"{val:.1f}%"


def phase_color(phase):
    if phase == 'phase2':
        return f"{C.BOLD}{C.YELLOW}PHASE2{C.RESET}"
    return f"{C.DIM}phase1{C.RESET}"


def collector_running():
    try:
        result = subprocess.run(['pgrep', '-f', 'integrated_collector'], capture_output=True)
        return result.returncode == 0
    except:
        return False


def buffer_age_minutes():
    """How many minutes of data in the FeatureBuffer."""
    try:
        conn = sqlite3.connect(str(FEATURE_BUFFER), timeout=5)
        row = conn.execute("SELECT MIN(timestamp) FROM ohlcv LIMIT 1").fetchone()
        conn.close()
        if row and row[0]:
            from dateutil import parser
            oldest = parser.parse(row[0])
            if oldest.tzinfo is None:
                oldest = oldest.replace(tzinfo=timezone.utc)
            return int((datetime.now(timezone.utc) - oldest).total_seconds() / 60)
    except:
        pass
    return 0


def load_state():
    if LOADING_STATE.exists():
        try:
            with open(LOADING_STATE) as f:
                return json.load(f)
        except:
            pass
    return None


def get_recent_log_events(n=20):
    """Get recent loading scanner events from collector log."""
    events = []
    ansi = re.compile(r'\x1b\[[0-9;]*m')

    log_files = sorted(LOG_DIR.glob('collector_*.log'), reverse=True)
    if not log_files:
        return events

    try:
        result = subprocess.run(
            ['grep', '-E', r'LOADING_TRADER|LOADING\].*score=|TRIGGER\]|CONFIRMED\]',
             str(log_files[0])],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')[-n:]
        for line in lines:
            line = ansi.sub('', line).strip()
            if line:
                # Extract timestamp
                ts = line[:19] if len(line) > 19 else ''
                events.append((ts, line))
    except:
        pass

    return events


def get_top_scores_from_log():
    """Get latest loading scores from log."""
    scores = {}
    ansi = re.compile(r'\x1b\[[0-9;]*m')

    log_files = sorted(LOG_DIR.glob('collector_*.log'), reverse=True)
    if not log_files:
        return []

    try:
        result = subprocess.run(
            ['grep', r'\[LOADING\].*score=', str(log_files[0])],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')[-100:]

        for line in reversed(lines):
            line = ansi.sub('', line)
            if 'score=' in line and '[LOADING]' in line:
                try:
                    sym = line.split('[LOADING]')[1].strip().split()[0]
                    score = float(line.split('score=')[1].split()[0])
                    if sym not in scores:
                        scores[sym] = score
                except:
                    continue
    except:
        pass

    return sorted(scores.items(), key=lambda x: -x[1])[:15]


def display():
    clear()
    now = datetime.now()
    state = load_state()

    # ── HEADER ────────────────────────────────────────────
    print(f"{C.BOLD}{C.CYAN}{'═' * 80}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  PYTHIA LOADING STRATEGY MONITOR{C.RESET}")
    print(f"{C.DIM}  {now.strftime('%Y-%m-%d %H:%M:%S')}  |  Refreshes every 30s{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 80}{C.RESET}")

    # ── STATUS BAR ────────────────────────────────────────
    running = collector_running()
    status = f"{C.BG_GREEN}{C.WHITE} RUNNING {C.RESET}" if running else f"{C.BG_RED}{C.WHITE} STOPPED {C.RESET}"
    buf_age = buffer_age_minutes()
    warmup_left = max(0, 360 - buf_age)

    disk_free = shutil.disk_usage("/").free / (1024**3)
    db_size = LIVE_DB.stat().st_size / (1024**3) if LIVE_DB.exists() else 0

    warmup_str = f"  {C.BG_YELLOW}{C.WHITE} WARMUP {warmup_left}min {C.RESET}" if warmup_left > 0 else f"  {C.GREEN}LIVE{C.RESET}"

    print(f"\n  Collector: {status}{warmup_str}  |  Buffer: {buf_age}min  |  DB: {db_size:.2f}GB  |  Disk: {disk_free:.0f}GB free")

    # ── PORTFOLIO ─────────────────────────────────────────
    print(f"\n{C.BOLD}{C.YELLOW}  PORTFOLIO{C.RESET}")
    print(f"  {C.DIM}{'─' * 76}{C.RESET}")

    if state:
        stats = state.get('stats', {})
        cash = state.get('cash', 0)
        starting = state.get('starting_capital', 5000)
        total_pnl = stats.get('total_pnl', 0)
        p1_entries = stats.get('phase1_entries', 0)
        p2_scaleins = stats.get('phase2_scaleins', 0)
        p1_exits = stats.get('phase1_exits', 0)
        p2_exits = stats.get('phase2_exits', 0)
        n_open = len(state.get('positions', {}))
        n_closed = p1_exits + p2_exits

        # Win/loss from closed trades
        closed = state.get('last_closed', [])
        wins = sum(1 for t in closed if t.get('realized_pnl', 0) > 0)
        losses = sum(1 for t in closed if t.get('realized_pnl', 0) <= 0)
        win_pct = (wins / len(closed) * 100) if closed else 0

        open_value = sum(p.get('total_size', 0) for p in state.get('positions', {}).values())
        total_equity = cash + open_value

        print(f"  Cash: ${cash:,.0f}  |  Open: ${open_value:,.0f}  |  Equity: ${total_equity:,.0f}  |  P&L: {pnl_color(total_pnl)}")
        print(f"  Phase1: {p1_entries} entries, {p1_exits} exits  |  Phase2: {p2_scaleins} scale-ins, {p2_exits} exits  |  W/L: {wins}/{losses} ({win_pct:.0f}%)")

        # ── OPEN POSITIONS ────────────────────────────────
        positions = state.get('positions', {})
        if positions:
            print(f"\n{C.BOLD}{C.YELLOW}  OPEN POSITIONS ({len(positions)}){C.RESET}")
            print(f"  {C.DIM}{'─' * 76}{C.RESET}")
            print(f"  {C.DIM}{'Symbol':<14} {'Phase':<8} {'Size':>6} {'Entry':>10} {'Current':>10} {'P&L':>10} {'Peak':>10} {'Off Peak':>9} {'Held':>6}{C.RESET}")

            sorted_pos = sorted(positions.items(),
                                key=lambda x: (0 if x[1].get('phase') == 'phase2' else 1, -x[1].get('pnl_pct', 0)))

            for sym, pos in sorted_pos:
                phase = pos.get('phase', '?')
                size = pos.get('total_size', 0)
                entry = pos.get('entry_price', 0)
                current = pos.get('current_price', 0)
                pnl_p = pos.get('pnl_pct', 0)
                peak = pos.get('peak_price', 0)
                off_peak = ((current - peak) / peak * 100) if peak > 0 else 0

                # Hold time
                p1t = pos.get('phase1_time')
                if p1t:
                    try:
                        from dateutil import parser
                        dt = parser.parse(p1t)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        held = f"{(datetime.now(timezone.utc) - dt).total_seconds() / 3600:.1f}h"
                    except:
                        held = "?"
                else:
                    held = "?"

                pnl_str = pct_color(pnl_p)
                off_str = pct_color(off_peak) if off_peak < -3 else f"{C.DIM}{off_peak:.1f}%{C.RESET}"
                phase_str = phase_color(phase)

                print(f"  {sym:<14} {phase_str:<16} ${size:>5.0f} ${entry:>9.4f} ${current:>9.4f} {pnl_str:>18} ${peak:>9.4f} {off_str:>17} {held:>5}")

    else:
        print(f"  {C.DIM}No state file found{C.RESET}")

    # ── TOP LOADING SCORES ────────────────────────────────
    print(f"\n{C.BOLD}{C.YELLOW}  TOP LOADING SCORES{C.RESET}")
    print(f"  {C.DIM}{'─' * 76}{C.RESET}")

    top_scores = get_top_scores_from_log()
    if top_scores:
        for sym, score in top_scores:
            bar_len = int(score * 4)
            bar = '█' * bar_len
            if score >= 7.0:
                color = C.GREEN
            elif score >= 5.0:
                color = C.YELLOW
            else:
                color = C.DIM
            threshold_marker = " ◀ THRESHOLD" if score >= 7.0 else ""
            print(f"  {sym:<14} {color}{score:5.1f}  {bar}{C.RESET}{threshold_marker}")
    else:
        print(f"  {C.DIM}No scores available{C.RESET}")

    # ── RECENT CLOSED TRADES ──────────────────────────────
    if state:
        closed = state.get('last_closed', [])
        if closed:
            print(f"\n{C.BOLD}{C.YELLOW}  RECENT CLOSED TRADES{C.RESET}")
            print(f"  {C.DIM}{'─' * 76}{C.RESET}")
            print(f"  {C.DIM}{'Symbol':<14} {'Phase':<8} {'Exit Reason':<20} {'P&L':>12}{C.RESET}")

            for trade in reversed(closed[-10:]):
                sym = trade.get('symbol', '?')
                phase = trade.get('phase', '?')
                reason = trade.get('exit_reason', '?')
                pnl = trade.get('realized_pnl', 0)

                phase_str = phase_color(phase)
                pnl_str = pnl_color(pnl)

                print(f"  {sym:<14} {phase_str:<16} {reason:<20} {pnl_str:>20}")

    # ── RECENT LOG EVENTS ─────────────────────────────────
    print(f"\n{C.BOLD}{C.YELLOW}  RECENT EVENTS{C.RESET}")
    print(f"  {C.DIM}{'─' * 76}{C.RESET}")

    events = get_recent_log_events(8)
    for ts, line in events[-8:]:
        # Highlight key events
        if 'PHASE 2 SCALE' in line:
            icon = f"{C.YELLOW}▲{C.RESET}"
        elif 'PHASE 1 ENTRY' in line:
            icon = f"{C.CYAN}●{C.RESET}"
        elif 'EXIT:' in line:
            icon = f"{C.RED}✕{C.RESET}"
        elif 'CONFIRMED' in line:
            icon = f"{C.GREEN}✓{C.RESET}"
        elif 'TRIGGER' in line:
            icon = f"{C.YELLOW}⚡{C.RESET}"
        else:
            icon = f"{C.DIM}·{C.RESET}"

        # Truncate long lines
        display_line = line[20:96] if len(line) > 96 else line[20:]
        print(f"  {icon} {C.DIM}{ts[11:]}{C.RESET} {display_line}")

    # ── FOOTER ────────────────────────────────────────────
    print(f"\n{C.BOLD}{C.CYAN}{'═' * 80}{C.RESET}")
    print(f"{C.DIM}  Ctrl+C to exit  |  Strategy: 2-phase loading (Phase1=20%@3%stop, Phase2=100%@8%stop, 10%trail@15%){C.RESET}")


def main():
    print("Starting Loading Strategy Monitor...")
    try:
        while True:
            display()
            time.sleep(30)
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Monitor stopped.{C.RESET}")


if __name__ == '__main__':
    main()
