#!/usr/bin/env python3
"""
Web Dashboard for Pythia Loading Strategy

Shows live paper trading status for the two-phase loading detector.

Access from any browser on your network:
    http://brett-zeiglers-mac-mini.local:5001

Run with:
    python scripts/dashboard/web_dashboard.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import subprocess
import re
import shutil

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template_string

app = Flask(__name__)

LOADING_STATE_FILE = PROJECT_ROOT / 'data' / 'loading_trader_state.json'
FEATURE_BUFFER_DB = PROJECT_ROOT / 'data' / 'feature_buffer.db'
LIVE_DB = PROJECT_ROOT / 'data' / 'pythia.duckdb'

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Pythia Loading Strategy</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {
            font-family: 'SF Mono', 'Monaco', 'Menlo', monospace;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 { color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 10px; }
        h2 { color: #e3b341; margin-top: 30px; }
        .status-bar {
            display: flex;
            gap: 15px;
            align-items: center;
            margin: 15px 0;
            flex-wrap: wrap;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .running { background: #238636; color: #fff; }
        .stopped { background: #da3633; color: #fff; }
        .warmup { background: #d29922; color: #000; }
        .refresh { color: #484f58; font-size: 12px; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin: 15px 0;
        }
        .stat {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .stat-value { font-size: 22px; color: #58a6ff; }
        .stat-label { font-size: 11px; color: #484f58; margin-top: 4px; }
        .positive { color: #3fb950; }
        .negative { color: #f85149; }
        .neutral { color: #8b949e; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        th, td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #21262d;
        }
        th { color: #58a6ff; font-size: 12px; text-transform: uppercase; }
        .phase1 { color: #d29922; }
        .phase2 { color: #f0883e; font-weight: bold; }
        .confirmed { color: #3fb950; font-weight: bold; }
        .score-bar {
            display: inline-block;
            height: 8px;
            border-radius: 4px;
            background: #58a6ff;
            margin-left: 8px;
            vertical-align: middle;
        }
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .alert-row { border-left: 3px solid #d29922; padding-left: 12px; margin: 8px 0; }
        .alert-row.triggered { border-left-color: #f0883e; }
        .alert-row.confirmed { border-left-color: #3fb950; }
    </style>
</head>
<body>
    <h1>Pythia Loading Strategy</h1>
    <div class="status-bar">
        <span class="badge {{ 'running' if collector_running else 'stopped' }}">
            Collector {{ 'RUNNING' if collector_running else 'STOPPED' }}
        </span>
        {% if warmup_remaining > 0 %}
        <span class="badge warmup">WARMUP {{ warmup_remaining }}min remaining</span>
        {% endif %}
        <span style="color: #484f58">DB: {{ db_size }} | Disk: {{ disk_free }} free</span>
        <span class="refresh">Updated: {{ timestamp }}</span>
    </div>

    <div class="stats-grid">
        <div class="stat">
            <div class="stat-value {{ 'positive' if total_pnl >= 0 else 'negative' }}">
                {{ "+" if total_pnl >= 0 else "" }}${{ "%.2f"|format(total_pnl) }}
            </div>
            <div class="stat-label">Realized P&L</div>
        </div>
        <div class="stat">
            <div class="stat-value">${{ "%.0f"|format(cash) }}</div>
            <div class="stat-label">Cash</div>
        </div>
        <div class="stat">
            <div class="stat-value">{{ n_open }}</div>
            <div class="stat-label">Open Positions</div>
        </div>
        <div class="stat">
            <div class="stat-value">{{ phase1_entries }}</div>
            <div class="stat-label">Phase 1 Entries</div>
        </div>
        <div class="stat">
            <div class="stat-value">{{ phase2_scaleins }}</div>
            <div class="stat-label">Phase 2 Scale-ins</div>
        </div>
        <div class="stat">
            <div class="stat-value">{{ n_closed }}</div>
            <div class="stat-label">Closed Trades</div>
        </div>
    </div>

    {% if open_positions %}
    <h2>Open Positions</h2>
    <table>
        <tr>
            <th>Symbol</th>
            <th>Phase</th>
            <th>Size</th>
            <th>Entry</th>
            <th>Current</th>
            <th>P&L</th>
            <th>Peak</th>
            <th>From Peak</th>
            <th>Held</th>
        </tr>
        {% for pos in open_positions %}
        <tr>
            <td><strong>{{ pos.symbol }}</strong></td>
            <td class="{{ 'phase2' if pos.phase == 'phase2' else 'phase1' }}">{{ pos.phase }}</td>
            <td>${{ "%.0f"|format(pos.size) }}</td>
            <td>${{ "%.4f"|format(pos.entry) }}</td>
            <td>${{ "%.4f"|format(pos.current) }}</td>
            <td class="{{ 'positive' if pos.pnl_pct >= 0 else 'negative' }}">
                {{ "+" if pos.pnl_pct >= 0 else "" }}{{ "%.1f"|format(pos.pnl_pct) }}%
                (${{ "+" if pos.pnl_usd >= 0 else "" }}{{ "%.2f"|format(pos.pnl_usd) }})
            </td>
            <td>${{ "%.4f"|format(pos.peak) }}</td>
            <td class="{{ 'negative' if pos.from_peak < -3 else 'neutral' }}">
                {{ "%.1f"|format(pos.from_peak) }}%
            </td>
            <td>{{ pos.held }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    {% if scanner_alerts %}
    <h2>Active Scanner Alerts</h2>
    {% for alert in scanner_alerts %}
    <div class="alert-row {{ alert.phase }}">
        <strong>{{ alert.symbol }}</strong>
        <span class="{{ alert.phase }}">{{ alert.phase|upper }}</span>
        score={{ "%.1f"|format(alert.score) }}
        <span class="score-bar" style="width: {{ (alert.score * 8)|int }}px"></span>
        {% if alert.move %}| move: {{ "+" if alert.move >= 0 else "" }}{{ "%.1f"|format(alert.move) }}%{% endif %}
    </div>
    {% endfor %}
    {% endif %}

    {% if top_scores %}
    <h2>Top Loading Scores</h2>
    <table>
        <tr>
            <th>Symbol</th>
            <th>Score</th>
            <th>Visual</th>
        </tr>
        {% for item in top_scores %}
        <tr>
            <td>{{ item.symbol }}</td>
            <td>{{ "%.1f"|format(item.score) }}</td>
            <td><span class="score-bar" style="width: {{ (item.score * 12)|int }}px"></span></td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}

    {% if closed_trades %}
    <h2>Recent Closed Trades</h2>
    <table>
        <tr>
            <th>Symbol</th>
            <th>Phase</th>
            <th>Exit Reason</th>
            <th>P&L</th>
        </tr>
        {% for trade in closed_trades %}
        <tr>
            <td>{{ trade.symbol }}</td>
            <td class="{{ 'phase2' if trade.phase == 'phase2' else 'phase1' }}">{{ trade.phase }}</td>
            <td>{{ trade.reason }}</td>
            <td class="{{ 'positive' if trade.pnl >= 0 else 'negative' }}">
                {{ "+" if trade.pnl >= 0 else "" }}${{ "%.2f"|format(trade.pnl) }}
            </td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}
</body>
</html>
"""


def check_collector_running():
    try:
        result = subprocess.run(['pgrep', '-f', 'integrated_collector'], capture_output=True)
        return result.returncode == 0
    except:
        return False


def get_db_size():
    if LIVE_DB.exists():
        size = LIVE_DB.stat().st_size
        if size > 1e9:
            return f"{size/1e9:.1f}GB"
        return f"{size/1e6:.0f}MB"
    return "N/A"


def get_disk_free():
    try:
        free = shutil.disk_usage("/").free / (1024**3)
        return f"{free:.0f}GB"
    except:
        return "N/A"


def get_warmup_remaining():
    """Check how much warmup time is left by looking at FeatureBuffer data age."""
    try:
        conn = sqlite3.connect(str(FEATURE_BUFFER_DB), timeout=5)
        row = conn.execute("SELECT MIN(timestamp) FROM ohlcv LIMIT 1").fetchone()
        conn.close()
        if row and row[0]:
            from dateutil import parser
            oldest = parser.parse(row[0])
            if oldest.tzinfo is None:
                oldest = oldest.replace(tzinfo=timezone.utc)
            age_minutes = (datetime.now(timezone.utc) - oldest).total_seconds() / 60
            remaining = max(0, 360 - age_minutes)
            return int(remaining)
    except:
        pass
    return 0


def load_state():
    if LOADING_STATE_FILE.exists():
        try:
            with open(LOADING_STATE_FILE) as f:
                return json.load(f)
        except:
            pass
    return None


def get_scanner_alerts():
    """Parse recent loading scanner alerts from collector log."""
    alerts = []
    ansi = re.compile(r'\x1b\[[0-9;]*m')

    # Find latest log file
    log_dir = PROJECT_ROOT / 'logs'
    log_files = sorted(log_dir.glob('collector_*.log'), reverse=True)
    if not log_files:
        return alerts

    try:
        result = subprocess.run(
            ['grep', '-E', r'LOADING\]|TRIGGER\]|CONFIRMED\]', str(log_files[0])],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')[-30:]

        seen = set()
        for line in reversed(lines):
            line = ansi.sub('', line)
            if '[LOADING]' in line and 'score=' in line:
                try:
                    sym = line.split('[LOADING]')[1].strip().split()[0]
                    score = float(line.split('score=')[1].split()[0])
                    if sym not in seen:
                        seen.add(sym)
                        alerts.append({'symbol': sym, 'score': score, 'phase': 'loading', 'move': None})
                except:
                    continue
            elif '[TRIGGER]' in line:
                try:
                    sym = line.split('[TRIGGER]')[1].strip().split()[0]
                    move = float(line.split('+')[1].split('%')[0])
                    if sym not in seen:
                        seen.add(sym)
                        alerts.append({'symbol': sym, 'score': 0, 'phase': 'triggered', 'move': move})
                except:
                    continue
            elif '[CONFIRMED]' in line:
                try:
                    sym = line.split('[CONFIRMED]')[1].strip().split()[0]
                    move = float(line.split('+')[1].split('%')[0])
                    if sym not in seen:
                        seen.add(sym)
                        alerts.append({'symbol': sym, 'score': 0, 'phase': 'confirmed', 'move': move})
                except:
                    continue

    except:
        pass

    return alerts[:15]


def get_top_scores():
    """Get top loading scores from the latest scanner log line."""
    scores = []
    ansi = re.compile(r'\x1b\[[0-9;]*m')

    log_dir = PROJECT_ROOT / 'logs'
    log_files = sorted(log_dir.glob('collector_*.log'), reverse=True)
    if not log_files:
        return scores

    try:
        result = subprocess.run(
            ['grep', r'\[LOADING\].*score=', str(log_files[0])],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')[-50:]

        seen = set()
        for line in reversed(lines):
            line = ansi.sub('', line)
            if 'score=' in line and '[LOADING]' in line:
                try:
                    sym = line.split('[LOADING]')[1].strip().split()[0]
                    score = float(line.split('score=')[1].split()[0])
                    if sym not in seen and score >= 4.0:
                        seen.add(sym)
                        scores.append({'symbol': sym, 'score': score})
                except:
                    continue
    except:
        pass

    return sorted(scores, key=lambda x: -x['score'])[:15]


@app.route('/')
def dashboard():
    state = load_state()

    # Defaults
    cash = 5000
    total_pnl = 0
    n_open = 0
    n_closed = 0
    phase1_entries = 0
    phase2_scaleins = 0
    open_positions = []
    closed_trades = []

    if state:
        cash = state.get('cash', 5000)
        stats = state.get('stats', {})
        total_pnl = stats.get('total_pnl', 0)
        phase1_entries = stats.get('phase1_entries', 0)
        phase2_scaleins = stats.get('phase2_scaleins', 0)
        n_closed = stats.get('phase1_exits', 0) + stats.get('phase2_exits', 0)

        # Open positions
        for sym, pos in state.get('positions', {}).items():
            entry = pos.get('entry_price', 0)
            current = pos.get('current_price', entry)
            peak = pos.get('peak_price', entry)
            size = pos.get('total_size', 0)
            pnl_pct = pos.get('pnl_pct', 0)
            pnl_usd = size * (pnl_pct / 100) if size else 0
            from_peak = ((current - peak) / peak * 100) if peak > 0 else 0

            # Calculate hold time
            p1_time = pos.get('phase1_time')
            if p1_time:
                try:
                    from dateutil import parser
                    entry_dt = parser.parse(p1_time)
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                    held_hours = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
                    held = f"{held_hours:.1f}h"
                except:
                    held = "?"
            else:
                held = "?"

            open_positions.append({
                'symbol': sym,
                'phase': pos.get('phase', '?'),
                'size': size,
                'entry': entry,
                'current': current,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'peak': peak,
                'from_peak': from_peak,
                'held': held,
            })

        n_open = len(open_positions)

        # Sort: phase2 first, then by pnl
        open_positions.sort(key=lambda x: (0 if x['phase'] == 'phase2' else 1, -x['pnl_pct']))

        # Closed trades
        for trade in state.get('last_closed', [])[-20:]:
            closed_trades.append({
                'symbol': trade.get('symbol', '?'),
                'phase': trade.get('phase', '?'),
                'reason': trade.get('exit_reason', '?'),
                'pnl': trade.get('realized_pnl', 0),
            })
        closed_trades.reverse()

    return render_template_string(
        HTML_TEMPLATE,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        collector_running=check_collector_running(),
        warmup_remaining=get_warmup_remaining(),
        db_size=get_db_size(),
        disk_free=get_disk_free(),
        cash=cash,
        total_pnl=total_pnl,
        n_open=n_open,
        n_closed=n_closed,
        phase1_entries=phase1_entries,
        phase2_scaleins=phase2_scaleins,
        open_positions=open_positions,
        scanner_alerts=get_scanner_alerts(),
        top_scores=get_top_scores(),
        closed_trades=closed_trades,
    )


if __name__ == '__main__':
    print("=" * 50)
    print("  Pythia Loading Strategy Dashboard")
    print("=" * 50)
    print()
    print("Access from any browser:")
    print("  http://localhost:5001")
    print("  http://brett-zeiglers-mac-mini.local:5001")
    print()

    app.run(host='0.0.0.0', port=5001, debug=False)
