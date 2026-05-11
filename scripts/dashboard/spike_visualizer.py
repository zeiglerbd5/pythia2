#!/usr/bin/env python3
"""
Spike Visualizer GUI

PyQt6 GUI for browsing historical 25%+ price spikes with feature visualization.
Helps identify patterns that distinguish predictable/profitable spikes.

Usage:
    python scripts/dashboard/spike_visualizer.py [db_path]
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QListWidget, QListWidgetItem,
    QSplitter, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Handle import based on how script is run
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from spike_finder import SpikeFinder, Spike


# ============================================================================
# Feature Categories and Colors
# ============================================================================

FEATURE_CATEGORIES = {
    'Momentum': {
        'features': ['returns', 'returns_5m', 'returns_15m', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14'],
        'color': '#3498db'  # Blue
    },
    'Volatility': {
        'features': ['NATR', 'BB_width', 'BB_squeeze'],
        'color': '#e67e22'  # Orange
    },
    'Volume': {
        'features': ['volume_zscore', 'volume_zscore_5m', 'volume_zscore_15m', 'volume_roc', 'OBV'],
        'color': '#27ae60'  # Green
    },
    'Microstructure': {
        'features': ['trade_count', 'buy_sell_ratio', 'roll_measure', 'order_flow_imbalance', 'vpin'],
        'color': '#9b59b6'  # Purple
    },
    'Order Book': {
        'features': ['bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance'],
        'color': '#e74c3c'  # Red
    },
    'VWAP': {
        'features': ['VWAP_distance'],
        'color': '#1abc9c'  # Teal
    }
}

# Build reverse mapping: feature -> category
FEATURE_TO_CATEGORY = {}
FEATURE_TO_COLOR = {}
for cat_name, cat_info in FEATURE_CATEGORIES.items():
    for feat in cat_info['features']:
        FEATURE_TO_CATEGORY[feat] = cat_name
        FEATURE_TO_COLOR[feat] = cat_info['color']

ALL_FEATURES = list(FEATURE_TO_COLOR.keys())


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
QPushButton {
    background-color: #0f3460;
    color: #eee;
    border: 1px solid #1a5276;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #1a5276;
}
QPushButton:pressed {
    background-color: #0a2540;
}
QListWidget {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 4px;
    outline: none;
}
QListWidget::item {
    padding: 4px 8px;
    border-radius: 2px;
}
QListWidget::item:selected {
    background-color: #0f3460;
    color: #fff;
}
QListWidget::item:hover {
    background-color: #1a3a5c;
}
QSplitter::handle {
    background-color: #0f3460;
}
"""


# ============================================================================
# Display Mode Enum
# ============================================================================

class DisplayMode:
    TIMELINE = "timeline"      # 60 bars per feature across time (original)
    FEATURE_BARS = "feature"   # 24 fixed bars for selected candle


# ============================================================================
# Chart Widget
# ============================================================================

class SpikeChart(FigureCanvas):
    """Matplotlib canvas for spike visualization"""

    feature_clicked = pyqtSignal(str)  # Emitted when a feature is clicked
    cursor_changed = pyqtSignal(int)   # Emitted when cursor position changes

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 8), facecolor='#1a1a2e')
        super().__init__(self.fig)
        self.setParent(parent)

        # Chart state
        self.candles: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.spike: Optional[Spike] = None
        self.selected_feature: Optional[str] = None
        self.cursor_x: Optional[int] = None  # Index of cursor position

        # Display mode
        self.display_mode = DisplayMode.FEATURE_BARS  # Default to new mode

        # Z-score normalized features
        self.normalized_features: Optional[pd.DataFrame] = None

        # Setup axes
        self.ax = self.fig.add_subplot(111)
        self._setup_axes()

        # Connect mouse events
        self.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.mpl_connect('button_press_event', self._on_mouse_click)

    def _setup_axes(self):
        """Configure axes styling"""
        self.ax.set_facecolor('#1a1a2e')
        self.ax.tick_params(colors='#888')
        self.ax.spines['bottom'].set_color('#333')
        self.ax.spines['top'].set_color('#333')
        self.ax.spines['left'].set_color('#333')
        self.ax.spines['right'].set_color('#333')
        self.ax.grid(True, alpha=0.2, color='#444')

    def set_data(self, candles: pd.DataFrame, features: pd.DataFrame, spike: Spike):
        """Set candle and feature data for visualization"""
        self.candles = candles
        self.features = features
        self.spike = spike
        self.cursor_x = None

        # Normalize features to z-scores
        self._normalize_features()

        # Default to first available feature
        if self.selected_feature is None and not features.empty:
            available = [f for f in ALL_FEATURES if f in features.columns]
            if available:
                self.selected_feature = available[0]

        self._draw()

    def _normalize_features(self):
        """Z-score normalize features for consistent bar heights"""
        if self.features is None or self.features.empty:
            self.normalized_features = None
            return

        self.normalized_features = pd.DataFrame(index=self.features.index)

        for feat in ALL_FEATURES:
            if feat in self.features.columns:
                values = self.features[feat].values
                mean = np.nanmean(values)
                std = np.nanstd(values)
                if std > 0:
                    self.normalized_features[feat] = (values - mean) / std
                else:
                    self.normalized_features[feat] = 0.0

    def set_selected_feature(self, feature: str):
        """Select a feature to highlight"""
        self.selected_feature = feature
        self._draw()

    def set_display_mode(self, mode: str):
        """Switch between display modes"""
        self.display_mode = mode
        self._draw()

    def toggle_display_mode(self):
        """Toggle between Timeline and Feature Bars modes"""
        if self.display_mode == DisplayMode.TIMELINE:
            self.display_mode = DisplayMode.FEATURE_BARS
        else:
            self.display_mode = DisplayMode.TIMELINE
        self._draw()
        return self.display_mode

    def _draw(self):
        """Draw the chart based on display mode"""
        # Clear any secondary axes from feature bars mode
        if hasattr(self, '_ax_bars') and self._ax_bars is not None:
            try:
                self._ax_bars.remove()
            except:
                pass
            self._ax_bars = None

        # Clear all axes and recreate
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self._setup_axes()

        if self.candles is None or self.candles.empty:
            self.ax.text(0.5, 0.5, 'No data loaded', ha='center', va='center',
                        color='#888', fontsize=14, transform=self.ax.transAxes)
            self.draw()
            return

        if self.display_mode == DisplayMode.FEATURE_BARS:
            self._draw_feature_bars_mode()
        else:
            self._draw_timeline_mode()

        self.fig.tight_layout()
        self.draw()

    def _draw_timeline_mode(self):
        """Draw timeline mode - 60 bars per feature across time (original mode)"""
        n_candles = len(self.candles)
        x = np.arange(n_candles)
        prices = self.candles['close'].values
        price_normalized = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

        # Draw feature bars FIRST (so price line is on top)
        if self.normalized_features is not None and not self.normalized_features.empty:
            n_features = len([f for f in ALL_FEATURES if f in self.normalized_features.columns])
            if n_features > 0:
                bar_width = 0.8 / max(n_features, 1)

                for i, feat in enumerate(ALL_FEATURES):
                    if feat not in self.normalized_features.columns:
                        continue

                    z_scores = self.normalized_features[feat].values
                    z_clipped = np.clip(z_scores, -3, 3)
                    bar_heights = (z_clipped + 3) / 6

                    color = FEATURE_TO_COLOR.get(feat, '#888')
                    alpha = 0.8 if feat == self.selected_feature else 0.3

                    offset = (i - n_features/2) * bar_width
                    self.ax.bar(x + offset, bar_heights, width=bar_width, color=color,
                               alpha=alpha, zorder=1)  # Low zorder so price line is on top

        # Draw price line ON TOP of bars (white, slightly transparent)
        self.ax.plot(x, price_normalized * 0.9 + 0.05, color='#ffffff', alpha=0.7,
                    linewidth=3, zorder=10)

        # Draw spike start marker
        if n_candles > 30:
            self.ax.axvline(x=30, color='#e74c3c', linestyle='--', alpha=0.7, zorder=11)

        # Draw cursor line
        if self.cursor_x is not None and 0 <= self.cursor_x < n_candles:
            self.ax.axvline(x=self.cursor_x, color='#fff', linestyle='-', alpha=0.5, zorder=11)

        # Configure axes
        self.ax.set_xlim(-1, n_candles)
        self.ax.set_ylim(-0.1, 1.1)

        tick_positions = [0, 15, 30, 45, 60] if n_candles >= 60 else list(range(0, n_candles, 10))
        tick_labels = [f'{p-30}m' for p in tick_positions]
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(tick_labels)
        self.ax.set_xlabel('Time relative to spike', color='#888')

        if self.selected_feature and self.features is not None:
            if self.selected_feature in self.features.columns:
                feat_values = self.features[self.selected_feature].values
                feat_min, feat_max = np.nanmin(feat_values), np.nanmax(feat_values)
                self.ax.set_ylabel(f'{self.selected_feature} ({feat_min:.2f} - {feat_max:.2f})',
                                  color=FEATURE_TO_COLOR.get(self.selected_feature, '#888'))

    def _draw_feature_bars_mode(self):
        """Draw feature bars mode - 24 fixed bars for selected candle with price line on top"""
        n_candles = len(self.candles)
        prices = self.candles['close'].values
        price_normalized = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

        # Get cursor position
        cursor_idx = self.cursor_x if self.cursor_x is not None else 30
        cursor_idx = max(0, min(cursor_idx, n_candles - 1))

        # Get available features
        available_features = [f for f in ALL_FEATURES if f in self.normalized_features.columns]
        n_features = len(available_features)

        if n_features == 0 or self.normalized_features is None:
            self.ax.text(0.5, 0.5, 'No features available', ha='center', va='center',
                        color='#888', transform=self.ax.transAxes)
            return

        # Use single axes for both price line and feature bars
        self.ax.set_position([0.1, 0.15, 0.85, 0.75])

        # Get feature values at cursor position
        bar_values = []
        bar_colors = []
        bar_labels = []

        for feat in available_features:
            if cursor_idx < len(self.normalized_features):
                z_score = self.normalized_features.iloc[cursor_idx][feat]
                z_clipped = np.clip(z_score, -3, 3)
                bar_height = (z_clipped + 3) / 6  # Map [-3, 3] to [0, 1]
                bar_values.append(bar_height)
            else:
                bar_values.append(0.5)

            bar_colors.append(FEATURE_TO_COLOR.get(feat, '#888'))
            bar_labels.append(feat)

        # Draw feature bars FIRST (low zorder, so price line is on top)
        x_bars = np.arange(n_features)
        bars = self.ax.bar(x_bars, bar_values, color=bar_colors, alpha=0.7, width=0.8, zorder=1)

        # Highlight selected feature
        if self.selected_feature and self.selected_feature in available_features:
            idx = available_features.index(self.selected_feature)
            bars[idx].set_alpha(1.0)
            bars[idx].set_edgecolor('#fff')
            bars[idx].set_linewidth(2)

        # Draw price line ON TOP of bars (map time to feature bar positions)
        # Scale x from [0, n_candles] to [0, n_features]
        x_price = np.linspace(0, n_features - 1, n_candles)
        self.ax.plot(x_price, price_normalized, color='#00ff88', alpha=0.95,
                    linewidth=4, zorder=10)  # Bright green, ON TOP

        # Draw cursor position on the price line
        cursor_x_mapped = cursor_idx * (n_features - 1) / (n_candles - 1) if n_candles > 1 else 0
        cursor_y = price_normalized[cursor_idx] if cursor_idx < len(price_normalized) else 0.5
        self.ax.scatter([cursor_x_mapped], [cursor_y], color='#ffffff', s=150, zorder=15,
                       edgecolors='#000', linewidths=2)

        # Draw vertical cursor line
        self.ax.axvline(x=cursor_x_mapped, color='#ffffff', linestyle='-', alpha=0.5,
                       linewidth=1, zorder=5)

        # Draw spike start marker (at t=0, which is index 30)
        if n_candles > 30:
            spike_x = 30 * (n_features - 1) / (n_candles - 1) if n_candles > 1 else 0
            self.ax.axvline(x=spike_x, color='#e74c3c', linestyle='--', alpha=0.7,
                           linewidth=2, zorder=5, label='Spike Start (t=0)')

        # Add horizontal line at z=0 (middle)
        self.ax.axhline(y=0.5, color='#666', linestyle='--', alpha=0.3, zorder=0)

        # Configure axes
        self.ax.set_xlim(-0.5, n_features - 0.5)
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_xticks(x_bars)
        self.ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)
        self.ax.set_ylabel('Normalized (0-1)', color='#888', fontsize=9)

        # Add title with time and price info
        time_offset = cursor_idx - 30
        current_price = prices[cursor_idx] if cursor_idx < len(prices) else 0
        self.ax.set_title(f't = {time_offset:+d} min  |  Price: ${current_price:.4f}  |  Move mouse left/right to scrub',
                         color='#ffffff', fontsize=11, pad=10)

        # Store reference for cleanup
        self._ax_bars = None

    def _on_mouse_move(self, event):
        """Handle mouse movement - update cursor position"""
        if event.inaxes != self.ax:
            return

        if self.candles is None:
            return

        n_candles = len(self.candles)

        if self.display_mode == DisplayMode.FEATURE_BARS:
            # In Feature Bars mode, x-axis is feature positions (0 to n_features-1)
            # Need to map back to candle index (0 to n_candles-1)
            available_features = [f for f in ALL_FEATURES if f in self.normalized_features.columns]
            n_features = len(available_features)

            if n_features > 0 and event.xdata is not None:
                # Map feature position to candle index
                # x_price = np.linspace(0, n_features - 1, n_candles), so reverse it
                candle_idx = event.xdata * (n_candles - 1) / (n_features - 1) if n_features > 1 else 0
                self.cursor_x = int(round(candle_idx))
            else:
                self.cursor_x = None
        else:
            # In Timeline mode, x-axis is already candle indices
            self.cursor_x = int(round(event.xdata)) if event.xdata else None

        if self.cursor_x is not None:
            self.cursor_x = max(0, min(self.cursor_x, n_candles - 1))

        self._draw()

    def _on_mouse_click(self, event):
        """Handle mouse click - select feature from bar"""
        if event.inaxes != self.ax:
            return

        # For now, just emit that a click happened
        # Could add bar hit detection later
        if self.selected_feature:
            self.feature_clicked.emit(self.selected_feature)

    def get_cursor_info(self) -> Optional[Dict]:
        """Get info about the current cursor position"""
        if self.cursor_x is None or self.candles is None or self.features is None:
            return None

        if self.cursor_x >= len(self.candles):
            return None

        result = {
            'index': self.cursor_x,
            'time_offset': self.cursor_x - 30,  # Minutes from spike start
        }

        # Get candle data
        if self.cursor_x < len(self.candles):
            result['price'] = self.candles.iloc[self.cursor_x]['close']
            result['timestamp'] = self.candles.iloc[self.cursor_x]['timestamp']

        # Get feature value
        if self.selected_feature and self.cursor_x < len(self.features):
            if self.selected_feature in self.features.columns:
                result['feature_value'] = self.features.iloc[self.cursor_x][self.selected_feature]
            if self.normalized_features is not None and self.selected_feature in self.normalized_features.columns:
                result['z_score'] = self.normalized_features.iloc[self.cursor_x][self.selected_feature]

        return result


# ============================================================================
# Main Window
# ============================================================================

class SpikeVisualizer(QMainWindow):
    """Main window for spike visualization"""

    def __init__(self, db_path: str = None):
        super().__init__()
        self.setWindowTitle("Pythia Spike Visualizer")
        self.setGeometry(100, 100, 1400, 900)

        # Data
        self.spike_finder = SpikeFinder(db_path)  # SpikeFinder handles default path
        self.spikes: List[Spike] = []
        self.current_index = 0

        # Setup UI
        self._setup_ui()
        self.setStyleSheet(DARK_STYLESHEET)

        # Load spikes
        self._load_spikes()

    def _setup_ui(self):
        """Setup the user interface"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Header
        header = self._create_header()
        main_layout.addWidget(header)

        # Main content area (chart + feature list)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Chart area
        chart_frame = QFrame()
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        self.chart = SpikeChart()
        self.chart.feature_clicked.connect(self._on_feature_clicked)
        chart_layout.addWidget(self.chart)

        splitter.addWidget(chart_frame)

        # Feature list panel
        feature_panel = self._create_feature_panel()
        splitter.addWidget(feature_panel)

        splitter.setSizes([1000, 300])
        main_layout.addWidget(splitter, stretch=1)

        # Info panel at bottom
        info_panel = self._create_info_panel()
        main_layout.addWidget(info_panel)

        # Connect chart mouse movement to update info
        self.chart.mpl_connect('motion_notify_event', self._update_info_panel)

    def _create_header(self) -> QFrame:
        """Create header with spike info and navigation"""
        frame = QFrame()
        frame.setFixedHeight(80)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(15, 10, 15, 10)

        # Prev button
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self._prev_spike)
        layout.addWidget(self.prev_btn)

        # Spike info
        info_layout = QVBoxLayout()

        self.symbol_label = QLabel("---")
        self.symbol_label.setFont(QFont("Consolas", 20, QFont.Weight.Bold))
        self.symbol_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.symbol_label)

        self.details_label = QLabel("Loading spikes...")
        self.details_label.setFont(QFont("Consolas", 12))
        self.details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.details_label.setStyleSheet("color: #888;")
        info_layout.addWidget(self.details_label)

        layout.addLayout(info_layout, stretch=1)

        # Mode toggle button
        self.mode_btn = QPushButton("Mode: Feature Bars")
        self.mode_btn.setFixedWidth(150)
        self.mode_btn.clicked.connect(self._toggle_mode)
        layout.addWidget(self.mode_btn)

        # Next button
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self._next_spike)
        layout.addWidget(self.next_btn)

        return frame

    def _create_feature_panel(self) -> QFrame:
        """Create feature list panel"""
        frame = QFrame()
        frame.setMinimumWidth(200)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel("Features")
        title.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        self.feature_list = QListWidget()
        self.feature_list.itemClicked.connect(self._on_feature_list_click)

        # Populate feature list by category
        for cat_name, cat_info in FEATURE_CATEGORIES.items():
            # Category header
            header_item = QListWidgetItem(f"-- {cat_name} --")
            header_item.setFlags(Qt.ItemFlag.NoItemFlags)
            header_item.setForeground(QColor(cat_info['color']))
            self.feature_list.addItem(header_item)

            # Features in category
            for feat in cat_info['features']:
                item = QListWidgetItem(f"  {feat}")
                item.setData(Qt.ItemDataRole.UserRole, feat)
                item.setForeground(QColor(cat_info['color']))
                self.feature_list.addItem(item)

        layout.addWidget(self.feature_list)
        return frame

    def _create_info_panel(self) -> QFrame:
        """Create info panel at bottom"""
        frame = QFrame()
        frame.setFixedHeight(50)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(15, 5, 15, 5)

        self.info_label = QLabel("Hover over chart to see feature values")
        self.info_label.setFont(QFont("Consolas", 11))
        self.info_label.setStyleSheet("color: #888;")
        layout.addWidget(self.info_label)

        return frame

    def _load_spikes(self):
        """Load spikes from database"""
        self.details_label.setText("Loading spikes...")
        QApplication.processEvents()

        self.spikes = self.spike_finder.find_spikes(min_gain_pct=25.0, limit=500)

        if self.spikes:
            self.details_label.setText(f"Spike 1 of {len(self.spikes)}")
            self._show_spike(0)
        else:
            self.details_label.setText("No spikes found")
            self.symbol_label.setText("No Data")

    def _show_spike(self, index: int):
        """Show spike at given index"""
        if not self.spikes or index < 0 or index >= len(self.spikes):
            return

        self.current_index = index
        spike = self.spikes[index]

        # Update header
        self.symbol_label.setText(spike.symbol)
        ts_str = spike.timestamp.strftime('%Y-%m-%d %H:%M')
        self.details_label.setText(
            f"{ts_str}  |  +{spike.gain_pct:.1f}%  |  Spike {index + 1} of {len(self.spikes)}"
        )

        # Load candle and feature data
        candles = self.spike_finder.get_spike_window(spike)
        features = self.spike_finder.get_spike_features(spike)

        # Update chart
        self.chart.set_data(candles, features, spike)

    def _prev_spike(self):
        """Show previous spike"""
        if self.current_index > 0:
            self._show_spike(self.current_index - 1)

    def _next_spike(self):
        """Show next spike"""
        if self.current_index < len(self.spikes) - 1:
            self._show_spike(self.current_index + 1)

    def _toggle_mode(self):
        """Toggle between display modes"""
        new_mode = self.chart.toggle_display_mode()
        if new_mode == DisplayMode.TIMELINE:
            self.mode_btn.setText("Mode: Timeline")
        else:
            self.mode_btn.setText("Mode: Feature Bars")

    def _on_feature_list_click(self, item: QListWidgetItem):
        """Handle feature list click"""
        feat = item.data(Qt.ItemDataRole.UserRole)
        if feat:
            self.chart.set_selected_feature(feat)

    def _on_feature_clicked(self, feature: str):
        """Handle feature click on chart"""
        # Find and select in list
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == feature:
                self.feature_list.setCurrentItem(item)
                break

    def _update_info_panel(self, event):
        """Update info panel based on cursor position"""
        info = self.chart.get_cursor_info()

        if info is None:
            self.info_label.setText("Hover over chart to see feature values")
            return

        parts = []

        # Time relative to spike
        parts.append(f"t = {info['time_offset']:+d} min")

        # Price
        if 'price' in info:
            parts.append(f"Price: ${info['price']:.4f}")

        # Selected feature value
        if 'feature_value' in info and self.chart.selected_feature:
            feat = self.chart.selected_feature
            val = info['feature_value']
            z = info.get('z_score', 0)
            parts.append(f"{feat}: {val:.4f} (z={z:.2f})")

        self.info_label.setText("  |  ".join(parts))


# ============================================================================
# Main
# ============================================================================

def main():
    # Use command line arg if provided, else default
    db_path = sys.argv[1] if len(sys.argv) > 1 else None

    app = QApplication(sys.argv)
    window = SpikeVisualizer(db_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
