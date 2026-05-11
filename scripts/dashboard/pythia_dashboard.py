#!/usr/bin/env python3
"""
Pythia Live Dashboard

Real-time PyQt6 dashboard that visualizes collector predictions,
showing statistics, model performance, and near-miss HOME RUN alerts.

Usage:
    python scripts/dashboard/pythia_dashboard.py [log_file]

Default log file: logs/collector_bb_fix.log
"""

import sys
import re
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QPushButton, QFrame, QScrollArea,
    QSizePolicy
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPalette, QColor


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Prediction:
    """Single prediction entry from log"""
    timestamp: datetime
    symbol: str
    v3: float
    v1: float
    va: float
    vb: float
    vc: float
    bb: float

    @property
    def home_run_score(self) -> float:
        """Calculate how close to HOME RUN thresholds (0-1, 1=passes all)"""
        v3_score = min(self.v3 / 93.0, 1.0)
        v1_score = min(self.v1 / 92.0, 1.0)
        bb_score = min(self.bb / 0.07, 1.0)
        return (v3_score + v1_score + bb_score) / 3.0

    @property
    def is_home_run(self) -> bool:
        """Check if this prediction passes all HOME RUN criteria"""
        return self.v3 >= 93.0 and self.v1 >= 92.0 and self.bb > 0.07

    @property
    def v3_passes(self) -> bool:
        return self.v3 >= 93.0

    @property
    def v1_passes(self) -> bool:
        return self.v1 >= 92.0

    @property
    def bb_passes(self) -> bool:
        return self.bb > 0.07


# ============================================================================
# Log Parser
# ============================================================================

class LogParser:
    """Parses collector log files for prediction entries"""

    # Pattern to match prediction lines
    # [PRED] SYMBOL: V3=XX.X% V1=XX.X% vA=XX.X% vB=XX.X% vC=XX.X% BB=X.XXXX
    PRED_PATTERN = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*'
        r'\[PRED\] (\S+): '
        r'V3=(\d+\.?\d*)% '
        r'V1=(\d+\.?\d*)% '
        r'vA=(\d+\.?\d*)% '
        r'vB=(\d+\.?\d*)% '
        r'vC=(\d+\.?\d*)% '
        r'BB=(\d+\.?\d*)'
    )

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.last_position = 0
        self.predictions: List[Prediction] = []
        self.first_timestamp: Optional[datetime] = None
        self.home_runs: List[Prediction] = []

    def parse_line(self, line: str) -> Optional[Prediction]:
        """Parse a single log line, return Prediction if it matches"""
        # Strip ANSI color codes
        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)

        match = self.PRED_PATTERN.search(clean_line)
        if match:
            try:
                timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                return Prediction(
                    timestamp=timestamp,
                    symbol=match.group(2).rstrip(':'),
                    v3=float(match.group(3)),
                    v1=float(match.group(4)),
                    va=float(match.group(5)),
                    vb=float(match.group(6)),
                    vc=float(match.group(7)),
                    bb=float(match.group(8))
                )
            except (ValueError, IndexError):
                pass
        return None

    def read_new_lines(self) -> List[Prediction]:
        """Read new lines from log file since last read"""
        new_predictions = []

        if not self.log_path.exists():
            return new_predictions

        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.last_position)
                for line in f:
                    pred = self.parse_line(line)
                    if pred:
                        new_predictions.append(pred)
                        self.predictions.append(pred)

                        if self.first_timestamp is None:
                            self.first_timestamp = pred.timestamp

                        if pred.is_home_run:
                            self.home_runs.append(pred)

                self.last_position = f.tell()
        except Exception as e:
            print(f"Error reading log: {e}")

        return new_predictions

    def get_top_signals(self, n: int = 10) -> List[Prediction]:
        """Get top N predictions by home_run_score"""
        # Get unique best prediction per symbol
        best_by_symbol = {}
        for pred in self.predictions:
            if pred.symbol not in best_by_symbol or pred.home_run_score > best_by_symbol[pred.symbol].home_run_score:
                best_by_symbol[pred.symbol] = pred

        sorted_preds = sorted(best_by_symbol.values(), key=lambda p: p.home_run_score, reverse=True)
        return sorted_preds[:n]

    @property
    def runtime(self) -> timedelta:
        """Get runtime since first prediction"""
        if not self.first_timestamp:
            return timedelta(0)
        return datetime.now() - self.first_timestamp

    @property
    def predictions_per_minute(self) -> float:
        """Calculate predictions per minute"""
        runtime_minutes = self.runtime.total_seconds() / 60
        if runtime_minutes < 0.1:
            return 0.0
        return len(self.predictions) / runtime_minutes


# ============================================================================
# Dark Theme Stylesheet
# ============================================================================

DARK_STYLESHEET = """
QMainWindow {
    background-color: #1a1a2e;
}
QWidget {
    background-color: #1a1a2e;
    color: #eee;
    font-family: 'Consolas', 'Monaco', monospace;
}
QFrame {
    background-color: #16213e;
    border-radius: 8px;
    border: 1px solid #0f3460;
}
QLabel {
    background-color: transparent;
    border: none;
}
QProgressBar {
    border: 1px solid #0f3460;
    border-radius: 4px;
    background-color: #1a1a2e;
    text-align: center;
    color: #eee;
}
QProgressBar::chunk {
    background-color: #e94560;
    border-radius: 3px;
}
QPushButton {
    background-color: #0f3460;
    border: 1px solid #e94560;
    border-radius: 4px;
    padding: 8px 16px;
    color: #eee;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #e94560;
}
QPushButton:pressed {
    background-color: #c13b52;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
"""


# ============================================================================
# Custom Widgets
# ============================================================================

class StatsPanel(QFrame):
    """Header panel showing overall statistics"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)

        # Title
        title = QLabel("PYTHIA DASHBOARD")
        title.setFont(QFont('Consolas', 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #e94560;")
        layout.addWidget(title)

        layout.addStretch()

        # Stats labels
        self.predictions_label = QLabel("Predictions: 0")
        self.predictions_label.setFont(QFont('Consolas', 12))
        layout.addWidget(self.predictions_label)

        layout.addWidget(self._separator())

        self.rate_label = QLabel("Rate: 0/min")
        self.rate_label.setFont(QFont('Consolas', 12))
        layout.addWidget(self.rate_label)

        layout.addWidget(self._separator())

        self.runtime_label = QLabel("Runtime: 0:00:00")
        self.runtime_label.setFont(QFont('Consolas', 12))
        layout.addWidget(self.runtime_label)

        layout.addWidget(self._separator())

        self.home_runs_label = QLabel("HOME RUNS: 0")
        self.home_runs_label.setFont(QFont('Consolas', 12, QFont.Weight.Bold))
        self.home_runs_label.setStyleSheet("color: #4ecca3;")
        layout.addWidget(self.home_runs_label)

    def _separator(self) -> QLabel:
        sep = QLabel("|")
        sep.setStyleSheet("color: #0f3460;")
        return sep

    def update_stats(self, total: int, rate: float, runtime: timedelta, home_runs: int):
        self.predictions_label.setText(f"Predictions: {total:,}")
        self.rate_label.setText(f"Rate: {rate:.1f}/min")

        hours, remainder = divmod(int(runtime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        self.runtime_label.setText(f"Runtime: {hours}:{minutes:02d}:{seconds:02d}")

        self.home_runs_label.setText(f"HOME RUNS: {home_runs}")
        if home_runs > 0:
            self.home_runs_label.setStyleSheet("color: #ffd700; background-color: #2d5a27;")


class TopSignalPanel(QFrame):
    """Main panel showing the top signal"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        # Header
        header = QLabel("TOP SIGNAL")
        header.setFont(QFont('Consolas', 14, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #ffd700;")
        layout.addWidget(header)

        # Symbol
        self.symbol_label = QLabel("---")
        self.symbol_label.setFont(QFont('Consolas', 36, QFont.Weight.Bold))
        self.symbol_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.symbol_label.setStyleSheet("color: #4ecca3;")
        layout.addWidget(self.symbol_label)

        # Timestamp
        self.timestamp_label = QLabel("")
        self.timestamp_label.setFont(QFont('Consolas', 10))
        self.timestamp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timestamp_label.setStyleSheet("color: #888;")
        layout.addWidget(self.timestamp_label)

        layout.addSpacing(10)

        # Progress bars for V3, V1, BB
        self.v3_bar = self._create_progress_row("V3", 93.0, "%")
        self.v1_bar = self._create_progress_row("V1", 92.0, "%")
        self.bb_bar = self._create_progress_row("BB", 0.07, "", scale=100)

        layout.addLayout(self.v3_bar[0])
        layout.addLayout(self.v1_bar[0])
        layout.addLayout(self.bb_bar[0])

        layout.addSpacing(10)

        # Secondary models
        self.secondary_label = QLabel("vA: --  |  vB: --  |  vC: --")
        self.secondary_label.setFont(QFont('Consolas', 11))
        self.secondary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.secondary_label.setStyleSheet("color: #888;")
        layout.addWidget(self.secondary_label)

        # Score
        self.score_label = QLabel("Score: 0%")
        self.score_label.setFont(QFont('Consolas', 12, QFont.Weight.Bold))
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.score_label)

    def _create_progress_row(self, name: str, threshold: float, suffix: str, scale: float = 1.0):
        layout = QHBoxLayout()

        label = QLabel(f"{name}:")
        label.setFont(QFont('Consolas', 12, QFont.Weight.Bold))
        label.setFixedWidth(40)
        layout.addWidget(label)

        bar = QProgressBar()
        bar.setMinimum(0)
        bar.setMaximum(100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(20)
        layout.addWidget(bar)

        value_label = QLabel(f"0{suffix}")
        value_label.setFont(QFont('Consolas', 11))
        value_label.setFixedWidth(80)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(value_label)

        target_label = QLabel(f"(need {threshold}{suffix})")
        target_label.setFont(QFont('Consolas', 9))
        target_label.setStyleSheet("color: #888;")
        target_label.setFixedWidth(100)
        layout.addWidget(target_label)

        check_label = QLabel("")
        check_label.setFixedWidth(20)
        layout.addWidget(check_label)

        return (layout, bar, value_label, check_label, threshold, suffix, scale)

    def update_signal(self, pred: Optional[Prediction]):
        if pred is None:
            self.symbol_label.setText("---")
            self.timestamp_label.setText("")
            self.secondary_label.setText("vA: --  |  vB: --  |  vC: --")
            self.score_label.setText("Score: 0%")
            return

        self.symbol_label.setText(pred.symbol)
        self.timestamp_label.setText(pred.timestamp.strftime("%H:%M:%S UTC"))

        # Update V3 bar
        self._update_bar(self.v3_bar, pred.v3, pred.v3_passes)

        # Update V1 bar
        self._update_bar(self.v1_bar, pred.v1, pred.v1_passes)

        # Update BB bar (scale by 100 for display)
        self._update_bar(self.bb_bar, pred.bb * 100, pred.bb_passes, is_bb=True)

        # Secondary models
        self.secondary_label.setText(f"vA: {pred.va:.1f}%  |  vB: {pred.vb:.1f}%  |  vC: {pred.vc:.1f}%")

        # Score
        score = pred.home_run_score * 100
        self.score_label.setText(f"Score: {score:.1f}%")

        if pred.is_home_run:
            self.score_label.setStyleSheet("color: #ffd700; font-size: 14px;")
            self.symbol_label.setStyleSheet("color: #ffd700; background-color: #2d5a27;")
        else:
            self.score_label.setStyleSheet("color: #4ecca3;")
            self.symbol_label.setStyleSheet("color: #4ecca3;")

    def _update_bar(self, bar_tuple, value: float, passes: bool, is_bb: bool = False):
        _, bar, value_label, check_label, threshold, suffix, scale = bar_tuple

        if is_bb:
            # BB is already scaled by 100
            bar_value = min(int(value / (threshold * 100) * 100), 100)
            value_label.setText(f"{value/100:.4f}")
        else:
            bar_value = min(int(value / threshold * 100), 100)
            value_label.setText(f"{value:.1f}{suffix}")

        bar.setValue(bar_value)

        if passes:
            check_label.setText("✓")
            check_label.setStyleSheet("color: #4ecca3; font-weight: bold;")
            bar.setStyleSheet("""
                QProgressBar::chunk { background-color: #4ecca3; }
            """)
        else:
            check_label.setText("")
            bar.setStyleSheet("""
                QProgressBar::chunk { background-color: #e94560; }
            """)


class NearMissesPanel(QFrame):
    """Panel showing top 10 near-miss signals"""

    signal_selected = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.predictions: List[Prediction] = []
        self.current_page = 0
        self.items_per_page = 4
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)

        # Header with navigation
        header_layout = QHBoxLayout()

        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(40)
        self.prev_btn.clicked.connect(self.prev_page)
        header_layout.addWidget(self.prev_btn)

        header = QLabel("TOP 10 NEAR MISSES")
        header.setFont(QFont('Consolas', 12, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(header)

        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(40)
        self.next_btn.clicked.connect(self.next_page)
        header_layout.addWidget(self.next_btn)

        self.page_label = QLabel("[1/1]")
        self.page_label.setFixedWidth(50)
        header_layout.addWidget(self.page_label)

        layout.addLayout(header_layout)

        # List items
        self.item_labels: List[QLabel] = []
        for i in range(self.items_per_page):
            item = QLabel("")
            item.setFont(QFont('Consolas', 10))
            item.setStyleSheet("""
                padding: 8px;
                background-color: #1a1a2e;
                border-radius: 4px;
                margin: 2px;
            """)
            item.setCursor(Qt.CursorShape.PointingHandCursor)
            item.mousePressEvent = lambda e, idx=i: self.item_clicked(idx)
            self.item_labels.append(item)
            layout.addWidget(item)

    def item_clicked(self, idx: int):
        actual_idx = self.current_page * self.items_per_page + idx
        if actual_idx < len(self.predictions):
            self.signal_selected.emit(self.predictions[actual_idx])

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_display()

    def next_page(self):
        max_pages = (len(self.predictions) - 1) // self.items_per_page + 1
        if self.current_page < max_pages - 1:
            self.current_page += 1
            self.update_display()

    def update_predictions(self, predictions: List[Prediction]):
        self.predictions = predictions
        self.update_display()

    def update_display(self):
        start_idx = self.current_page * self.items_per_page
        max_pages = max(1, (len(self.predictions) - 1) // self.items_per_page + 1)

        self.page_label.setText(f"[{self.current_page + 1}/{max_pages}]")

        for i, label in enumerate(self.item_labels):
            idx = start_idx + i
            if idx < len(self.predictions):
                pred = self.predictions[idx]
                score = pred.home_run_score * 100

                # Format: rank. SYMBOL   V3=XX.X%  V1=XX.X%  BB=X.XXXX  HH:MM
                text = (f"{idx+1:2d}. {pred.symbol:<12} "
                       f"V3={pred.v3:5.1f}%  V1={pred.v1:5.1f}%  "
                       f"BB={pred.bb:.4f}  {pred.timestamp.strftime('%H:%M')}")
                label.setText(text)

                if pred.is_home_run:
                    label.setStyleSheet("""
                        padding: 8px;
                        background-color: #2d5a27;
                        border-radius: 4px;
                        margin: 2px;
                        color: #ffd700;
                    """)
                else:
                    label.setStyleSheet("""
                        padding: 8px;
                        background-color: #1a1a2e;
                        border-radius: 4px;
                        margin: 2px;
                    """)
            else:
                label.setText("")
                label.setStyleSheet("""
                    padding: 8px;
                    background-color: #1a1a2e;
                    border-radius: 4px;
                    margin: 2px;
                """)


# ============================================================================
# Main Window
# ============================================================================

class PythiaDashboard(QMainWindow):
    """Main dashboard window"""

    def __init__(self, log_path: str):
        super().__init__()
        self.parser = LogParser(log_path)
        self.setup_ui()
        self.setup_timer()

    def setup_ui(self):
        self.setWindowTitle("Pythia Dashboard")
        self.setMinimumSize(700, 700)
        self.resize(800, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Stats panel
        self.stats_panel = StatsPanel()
        layout.addWidget(self.stats_panel)

        # Top signal panel
        self.top_signal_panel = TopSignalPanel()
        layout.addWidget(self.top_signal_panel, stretch=2)

        # Near misses panel
        self.near_misses_panel = NearMissesPanel()
        self.near_misses_panel.signal_selected.connect(self.on_signal_selected)
        layout.addWidget(self.near_misses_panel, stretch=1)

        # Apply stylesheet
        self.setStyleSheet(DARK_STYLESHEET)

    def setup_timer(self):
        """Set up timer for periodic log reading"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)  # Update every second

        # Initial read
        self.update_data()

    def update_data(self):
        """Read new log data and update UI"""
        new_preds = self.parser.read_new_lines()

        # Update stats
        self.stats_panel.update_stats(
            total=len(self.parser.predictions),
            rate=self.parser.predictions_per_minute,
            runtime=self.parser.runtime,
            home_runs=len(self.parser.home_runs)
        )

        # Update top signals
        top_signals = self.parser.get_top_signals(10)

        if top_signals:
            self.top_signal_panel.update_signal(top_signals[0])
        else:
            self.top_signal_panel.update_signal(None)

        self.near_misses_panel.update_predictions(top_signals)

    def on_signal_selected(self, pred: Prediction):
        """Handle click on near-miss item"""
        self.top_signal_panel.update_signal(pred)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    # Default log path
    default_log = "logs/collector_bb_fix.log"

    # Get log path from command line or use default
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        # Try to find Pythia root
        script_dir = Path(__file__).parent
        pythia_root = script_dir.parent.parent
        log_path = pythia_root / default_log

        if not log_path.exists():
            # Try current directory
            log_path = Path(default_log)

    print(f"Pythia Dashboard")
    print(f"Log file: {log_path}")
    print("-" * 40)

    if not Path(log_path).exists():
        print(f"Warning: Log file not found: {log_path}")
        print("Dashboard will start but show no data until log file appears.")

    app = QApplication(sys.argv)

    # Set application-wide dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 46))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(238, 238, 238))
    palette.setColor(QPalette.ColorRole.Base, QColor(22, 33, 62))
    palette.setColor(QPalette.ColorRole.Text, QColor(238, 238, 238))
    app.setPalette(palette)

    window = PythiaDashboard(str(log_path))
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
