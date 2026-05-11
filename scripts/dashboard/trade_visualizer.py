#!/usr/bin/env python3
"""
Trade Visualizer GUI

PyQt6 GUI for browsing backtest trades with price and feature visualization.
Shows winners vs losers to help identify patterns for ML filtering.

Usage:
    python scripts/dashboard/trade_visualizer.py [backtest_trades.duckdb]
"""

import sys
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QListWidget, QListWidgetItem,
    QSplitter, QSizePolicy, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# Feature categories (matching spike_visualizer structure)
# These match market_data.duckdb columns
FEATURE_CONFIG = {
    # Column name in DataFrame -> (Label for display, Color, Category)
    # Momentum (blue) - 5 features
    'returns_norm': ('Returns', '#3498db', 'Momentum'),
    'returns_5m_norm': ('Ret 5m', '#3498db', 'Momentum'),
    'macd_norm': ('MACD', '#3498db', 'Momentum'),
    'macd_hist_norm': ('MACD Hist', '#3498db', 'Momentum'),
    'rsi_norm': ('RSI', '#3498db', 'Momentum'),
    # Volatility (orange) - 3 features
    'natr_norm': ('NATR', '#e67e22', 'Volatility'),
    'bb_width_norm': ('BB Width', '#e67e22', 'Volatility'),
    'bb_squeeze_norm': ('BB Squeeze', '#e67e22', 'Volatility'),
    # Volume (green) - 4 features
    'vol_zscore_norm': ('Vol Z', '#27ae60', 'Volume'),
    'vol_roc_norm': ('Vol ROC', '#27ae60', 'Volume'),
    'obv_norm': ('OBV', '#27ae60', 'Volume'),
    'vwap_dist_norm': ('VWAP', '#27ae60', 'Volume'),
    # Microstructure (purple) - 4 features
    'trade_cnt_norm': ('Trades', '#9b59b6', 'Micro'),
    'buy_sell_norm': ('Buy/Sell', '#9b59b6', 'Micro'),
    'flow_imbal_norm': ('Flow Imb', '#9b59b6', 'Micro'),
    'vpin_norm': ('VPIN', '#9b59b6', 'Micro'),
    # Order Book (red) - 3 features (matches market_data.duckdb features table)
    'spread_pct_norm': ('Spread %', '#e74c3c', 'OrderBook'),
    'depth_ratio_norm': ('Depth Ratio', '#e74c3c', 'OrderBook'),
    'lg_imbal_norm': ('Lg Imbal', '#e74c3c', 'OrderBook'),
}

# Ordered list of features for display
FEATURE_ORDER = list(FEATURE_CONFIG.keys())
N_FEATURES = len(FEATURE_ORDER)

# Dark Theme Stylesheet
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
}
QLabel {
    color: #eee;
}
QPushButton {
    background-color: #0f3460;
    color: #eee;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #1a4a7a;
}
QPushButton:pressed {
    background-color: #0a2840;
}
QListWidget {
    background-color: #16213e;
    border: none;
    border-radius: 8px;
    padding: 5px;
}
QListWidget::item {
    padding: 8px;
    margin: 2px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #0f3460;
}
QListWidget::item:hover {
    background-color: #1a3a5c;
}
QComboBox {
    background-color: #0f3460;
    color: #eee;
    border: none;
    padding: 8px;
    border-radius: 6px;
}
QComboBox::drop-down {
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #16213e;
    color: #eee;
    selection-background-color: #0f3460;
}
"""


class TradeListItem(QListWidgetItem):
    """Custom list item for trades."""

    def __init__(self, trade: dict):
        self.trade = trade

        # Format display text
        outcome_emoji = "+" if trade['outcome'] == 'winner' else "-"
        pnl_color = "#27ae60" if trade['final_pnl_pct'] > 0 else "#e74c3c"

        text = f"{outcome_emoji} {trade['symbol']:<10} {trade['final_pnl_pct']:>+6.1f}%  {trade['exit_reason']:<6}  {trade['ob_trigger']}"
        super().__init__(text)

        # Color based on outcome
        if trade['outcome'] == 'winner':
            self.setForeground(QColor("#27ae60"))
        else:
            self.setForeground(QColor("#e74c3c"))


class DisplayMode:
    """Chart display modes."""
    LINE = "line"          # Separate price and feature subplots (original)
    BARS = "bars"          # Feature bars with price line overlay


class PriceChartWidget(FigureCanvas):
    """Matplotlib widget for price charts with entry/exit markers."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 6), facecolor='#1a1a2e')
        super().__init__(self.fig)
        self.setParent(parent)

        # Display mode
        self.display_mode = DisplayMode.LINE

        # Cursor position for bars mode (index into price data)
        self.cursor_idx = 0
        self.n_candles = 0

        # Create subplots (will be reconfigured based on mode)
        self.ax_price = self.fig.add_subplot(211)
        self.ax_features = self.fig.add_subplot(212)

        self._style_axes()

        # Connect mouse events for scrubbing in bars mode
        self.mpl_connect('motion_notify_event', self._on_mouse_move)

        # Store data for mouse scrubbing
        self._feature_data = None
        self._price_data = None
        self._price_normalized = None
        self._price_min = 0
        self._price_range = 1

    def _on_mouse_move(self, event):
        """Handle mouse movement for scrubbing in bars mode."""
        if self.display_mode != DisplayMode.BARS:
            return
        if event.inaxes != self.ax_price:
            return
        if self.n_candles <= 0:
            return

        x = event.xdata
        if x is None:
            return

        # In bars mode, x-axis is feature positions (0 to n_features-1)
        # Need to map back to candle index (0 to n_candles-1)
        n_features = N_FEATURES
        if n_features > 1 and self.n_candles > 1:
            # Reverse mapping: x_feature -> candle_idx
            candle_idx = x * (self.n_candles - 1) / (n_features - 1)
            new_idx = int(round(candle_idx))
        else:
            new_idx = 0

        new_idx = max(0, min(new_idx, self.n_candles - 1))

        if new_idx != self.cursor_idx:
            self.cursor_idx = new_idx
            self._update_bars_at_cursor()

    def _update_bars_at_cursor(self):
        """Redraw bars mode with updated cursor position (bars change height)."""
        if self._feature_data is None or self._feature_data.empty:
            return
        if not hasattr(self, '_current_trade'):
            return

        # Full redraw to update bar heights at new cursor position
        self._plot_bars_mode(self._current_trade, self._current_price_data, self._current_feature_data)
        self.draw_idle()

    def _style_axes(self, axes_list=None):
        """Apply dark theme to axes."""
        if axes_list is None:
            axes_list = [self.ax_price, self.ax_features]
        for ax in axes_list:
            if ax is not None:
                ax.set_facecolor('#16213e')
                ax.tick_params(colors='#aaa')
                ax.spines['bottom'].set_color('#444')
                ax.spines['top'].set_color('#444')
                ax.spines['left'].set_color('#444')
                ax.spines['right'].set_color('#444')
                ax.xaxis.label.set_color('#aaa')
                ax.yaxis.label.set_color('#aaa')
                ax.title.set_color('#eee')

    def toggle_mode(self):
        """Toggle between LINE and BARS display modes."""
        if self.display_mode == DisplayMode.LINE:
            self.display_mode = DisplayMode.BARS
        else:
            self.display_mode = DisplayMode.LINE
        return self.display_mode

    def plot_trade(self, trade: dict, price_data: pd.DataFrame, feature_data: pd.DataFrame):
        """Plot price chart with entry/exit markers and features."""
        # Store data for replotting on mode change
        self._current_trade = trade
        self._current_price_data = price_data
        self._current_feature_data = feature_data

        # Reset cursor for new trade
        self.cursor_idx = 0

        # Clear figure and recreate axes based on mode
        self.fig.clear()

        if self.display_mode == DisplayMode.BARS:
            self._plot_bars_mode(trade, price_data, feature_data)
        else:
            self._plot_line_mode(trade, price_data, feature_data)

        self.fig.tight_layout()
        self.draw()

    def _plot_line_mode(self, trade: dict, price_data: pd.DataFrame, feature_data: pd.DataFrame):
        """Original mode: separate price and feature subplots."""
        self.ax_price = self.fig.add_subplot(211)
        self.ax_features = self.fig.add_subplot(212)
        self._style_axes([self.ax_price, self.ax_features])

        if price_data.empty:
            self.ax_price.text(0.5, 0.5, 'No price data available',
                              transform=self.ax_price.transAxes, ha='center', color='#aaa')
            return

        # Plot price
        self.ax_price.plot(price_data['ts'], price_data['close'],
                          color='#3498db', linewidth=1.5, label='Price')

        # Entry marker
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        self.ax_price.axvline(entry_time, color='#f1c40f', linestyle='--', alpha=0.7, label='Entry')
        self.ax_price.scatter([entry_time], [entry_price], color='#f1c40f', s=100, zorder=5, marker='^')

        # Exit marker
        exit_time = trade['exit_time']
        exit_price = trade['exit_price']
        exit_color = '#27ae60' if trade['final_pnl_pct'] > 0 else '#e74c3c'
        self.ax_price.axvline(exit_time, color=exit_color, linestyle='--', alpha=0.7, label='Exit')
        self.ax_price.scatter([exit_time], [exit_price], color=exit_color, s=100, zorder=5, marker='v')

        # Max/min markers
        if trade['max_gain_pct'] > 0:
            self.ax_price.axhline(trade['max_price'], color='#27ae60', linestyle=':', alpha=0.5)
        if trade['max_loss_pct'] < 0:
            self.ax_price.axhline(trade['min_price'], color='#e74c3c', linestyle=':', alpha=0.5)

        # Stop loss line at -1%
        stop_price = entry_price * 0.99
        self.ax_price.axhline(stop_price, color='#e74c3c', linestyle='-', alpha=0.3, label='-1% Stop')

        # Title
        outcome = "WINNER" if trade['outcome'] == 'winner' else "LOSER"
        self.ax_price.set_title(
            f"{trade['symbol']} | {outcome} | {trade['final_pnl_pct']:+.1f}% | {trade['exit_reason']} | trigger: {trade['ob_trigger']}",
            fontsize=12, fontweight='bold', color='#eee'
        )
        self.ax_price.legend(loc='upper left', fontsize=8)
        self.ax_price.set_ylabel('Price', color='#aaa')

        # Plot features
        if not feature_data.empty:
            features_to_plot = ['natr_norm', 'spread_norm', 'depth_norm', 'imbalance_norm']
            colors = ['#e67e22', '#e74c3c', '#9b59b6', '#3498db']
            labels = ['NATR', 'Spread', 'Depth', 'Imbalance']

            for feat, color, label in zip(features_to_plot, colors, labels):
                if feat in feature_data.columns:
                    self.ax_features.plot(feature_data['timestamp'], feature_data[feat],
                                         color=color, linewidth=1, alpha=0.8, label=label)

            # 0.8 threshold line
            self.ax_features.axhline(0.8, color='#f1c40f', linestyle='--', alpha=0.5, label='Threshold')

            # Entry time
            self.ax_features.axvline(entry_time, color='#f1c40f', linestyle='--', alpha=0.7)

            self.ax_features.set_ylabel('Normalized (0-1)', color='#aaa')
            self.ax_features.set_ylim(0, 1.1)
            self.ax_features.legend(loc='upper left', fontsize=8, ncol=5)

    def _plot_bars_mode(self, trade: dict, price_data: pd.DataFrame, feature_data: pd.DataFrame):
        """Bar mode: N feature bars at cursor position with price line overlaid (like spike_visualizer)."""
        self.fig.clear()
        self.ax_price = self.fig.add_subplot(111)
        self.ax_features = None
        self._style_axes([self.ax_price])

        if price_data.empty:
            self.ax_price.text(0.5, 0.5, 'No price data available',
                              transform=self.ax_price.transAxes, ha='center', color='#aaa')
            return

        # Store for mouse scrubbing
        self._feature_data = feature_data
        self._price_data = price_data

        # Features configuration from global config
        feature_names = FEATURE_ORDER
        feature_labels = [FEATURE_CONFIG[f][0] for f in feature_names]
        feature_colors = [FEATURE_CONFIG[f][1] for f in feature_names]

        n_features = N_FEATURES
        n_candles = len(price_data)
        self.n_candles = n_candles

        # Get price normalized to 0-1
        prices = price_data['close'].values
        price_min, price_max = prices.min(), prices.max()
        price_range = price_max - price_min if price_max > price_min else 1
        self._price_min = price_min
        self._price_range = price_range
        price_normalized = (prices - price_min) / price_range
        self._price_normalized = price_normalized

        # Find entry/exit indices
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_idx = 0
        exit_idx = n_candles - 1

        if 'ts' in price_data.columns:
            for idx, ts in enumerate(price_data['ts']):
                if ts >= entry_time and entry_idx == 0:
                    entry_idx = idx
                if ts >= exit_time:
                    exit_idx = idx
                    break

        # Initialize cursor to entry if not set
        if self.cursor_idx == 0:
            self.cursor_idx = entry_idx

        # Clamp cursor
        cursor_idx = max(0, min(self.cursor_idx, n_candles - 1))

        # === GET FEATURE VALUES AT CURSOR POSITION ===
        bar_values = []
        bar_colors = []
        bar_labels = []

        for feat, color, label in zip(feature_names, feature_colors, feature_labels):
            if feat in feature_data.columns and cursor_idx < len(feature_data):
                val = feature_data[feat].iloc[cursor_idx]
                if np.isnan(val):
                    val = 0.0
            else:
                val = 0.0
            bar_values.append(val)
            bar_colors.append(color)
            bar_labels.append(label)

        # === DRAW FEATURE BARS (one per feature, heights = values at cursor) ===
        x_bars = np.arange(n_features)
        bars = self.ax_price.bar(x_bars, bar_values, color=bar_colors, alpha=0.8, width=0.8,
                                zorder=1, edgecolor='#000', linewidth=1)

        # === DRAW PRICE LINE ON TOP (mapped across the bars) ===
        # Scale x from [0, n_candles-1] to [0, n_features-1]
        x_price = np.linspace(0, n_features - 1, n_candles)
        self.ax_price.plot(x_price, price_normalized, color='#00ff88', alpha=0.95,
                          linewidth=4, zorder=10)

        # === CURSOR DOT ON PRICE LINE ===
        cursor_x_mapped = cursor_idx * (n_features - 1) / (n_candles - 1) if n_candles > 1 else 0
        cursor_y = price_normalized[cursor_idx] if cursor_idx < len(price_normalized) else 0.5
        self.ax_price.scatter([cursor_x_mapped], [cursor_y], color='#ffffff', s=150, zorder=15,
                             edgecolors='#000', linewidths=2)

        # === VERTICAL CURSOR LINE ===
        self.ax_price.axvline(x=cursor_x_mapped, color='#ffffff', linestyle='-', alpha=0.5,
                             linewidth=1, zorder=5)

        # === ENTRY MARKER ===
        entry_x_mapped = entry_idx * (n_features - 1) / (n_candles - 1) if n_candles > 1 else 0
        self.ax_price.axvline(x=entry_x_mapped, color='#f1c40f', linestyle='--', alpha=0.7,
                             linewidth=2, zorder=5)

        # === EXIT MARKER ===
        exit_x_mapped = exit_idx * (n_features - 1) / (n_candles - 1) if n_candles > 1 else 0
        exit_color = '#27ae60' if trade['final_pnl_pct'] > 0 else '#e74c3c'
        self.ax_price.axvline(x=exit_x_mapped, color=exit_color, linestyle='--', alpha=0.7,
                             linewidth=2, zorder=5)

        # === 0.8 THRESHOLD LINE ===
        self.ax_price.axhline(y=0.8, color='#f1c40f', linestyle='--', alpha=0.5, zorder=0)

        # === CONFIGURE AXES ===
        self.ax_price.set_xlim(-0.5, n_features - 0.5)
        self.ax_price.set_ylim(-0.05, 1.15)
        self.ax_price.set_xticks(x_bars)
        self.ax_price.set_xticklabels(bar_labels, fontsize=10, fontweight='bold')
        self.ax_price.set_ylabel('Normalized (0-1)', color='#aaa')

        # Color the x-axis labels
        for i, (label, color) in enumerate(zip(self.ax_price.get_xticklabels(), bar_colors)):
            label.set_color(color)

        # Get cursor timestamp
        cursor_time_str = ""
        if 'ts' in price_data.columns and cursor_idx < len(price_data):
            ts = price_data['ts'].iloc[cursor_idx]
            if hasattr(ts, 'strftime'):
                cursor_time_str = ts.strftime('%H:%M:%S')

        # Current price at cursor
        current_price = prices[cursor_idx] if cursor_idx < len(prices) else 0

        # Title with trade info and cursor position
        outcome = "WINNER" if trade['outcome'] == 'winner' else "LOSER"
        time_offset = cursor_idx - entry_idx
        self.ax_price.set_title(
            f"{cursor_time_str} | {trade['symbol']} | {outcome} | {trade['final_pnl_pct']:+.1f}% | "
            f"t={time_offset:+d}min | ${current_price:.4f}",
            fontsize=11, fontweight='bold', color='#eee'
        )

    def replot(self):
        """Replot with current data (called after mode toggle)."""
        if hasattr(self, '_current_trade') and self._current_trade is not None:
            self.plot_trade(self._current_trade, self._current_price_data, self._current_feature_data)


class TradeInfoPanel(QFrame):
    """Panel showing trade details."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(250)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        self.title_label = QLabel("Trade Details")
        self.title_label.setFont(QFont('Consolas', 14, QFont.Weight.Bold))
        layout.addWidget(self.title_label)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setFont(QFont('Consolas', 10))
        layout.addWidget(self.info_label)

        layout.addStretch()

    def update_trade(self, trade: dict):
        """Update panel with trade info."""
        outcome_color = "#27ae60" if trade['outcome'] == 'winner' else "#e74c3c"

        info = f"""
<b style='color: {outcome_color}; font-size: 16px;'>{trade['outcome'].upper()}</b>

<b>Symbol:</b> {trade['symbol']}
<b>Entry:</b> {trade['entry_time']}
<b>Exit:</b> {trade['exit_time']}

<b>Entry Price:</b> ${trade['entry_price']:.4f}
<b>Exit Price:</b> ${trade['exit_price']:.4f}
<b>Max Price:</b> ${trade['max_price']:.4f}
<b>Min Price:</b> ${trade['min_price']:.4f}

<b style='color: {outcome_color};'>P&L: {trade['final_pnl_pct']:+.2f}%</b>
<b>Max Gain:</b> {trade['max_gain_pct']:+.2f}%
<b>Max Loss:</b> {trade['max_loss_pct']:+.2f}%

<b>Exit Reason:</b> {trade['exit_reason']}
<b>OB Trigger:</b> {trade['ob_trigger']}

<hr>
<b>Entry Features:</b>
  NATR: {trade['natr_norm']:.3f}
  Spread: {trade['spread_norm']:.3f}
  Depth: {trade['depth_norm']:.3f}
  Imbalance: {trade['imbalance_norm']:.3f}
  RSI: {trade.get('rsi', 0):.1f}
  Vol Z: {trade.get('volume_zscore', 0):.2f}
"""
        self.info_label.setText(info)


class TradeVisualizerWindow(QMainWindow):
    """Main window for trade visualization."""

    def __init__(self, trades_db_path: str, market_db_path: str = None):
        super().__init__()

        self.trades_db_path = trades_db_path
        if market_db_path is None:
            market_db_path = Path(__file__).parent.parent.parent / "market_data.duckdb"
        self.market_db_path = market_db_path
        self.trades = []
        self.current_index = 0

        self._setup_ui()
        self._load_trades()

    def _setup_ui(self):
        """Setup the UI layout."""
        self.setWindowTitle("Trade Visualizer - Winners vs Losers")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(DARK_STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel - trade list + details stacked
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(280)
        left_panel.setMinimumWidth(250)

        # Filter controls
        filter_layout = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Trades", "Winners Only", "Losers Only", "Stop Losses", "Holds"])
        self.filter_combo.currentTextChanged.connect(self._filter_trades)
        filter_layout.addWidget(self.filter_combo)
        left_layout.addLayout(filter_layout)

        # Stats label
        self.stats_label = QLabel("")
        self.stats_label.setFont(QFont('Consolas', 9))
        left_layout.addWidget(self.stats_label)

        # Trade list (about 15-18 items visible)
        self.trade_list = QListWidget()
        self.trade_list.setMaximumHeight(400)
        self.trade_list.currentRowChanged.connect(self._on_trade_selected)
        left_layout.addWidget(self.trade_list)

        # Trade details panel (moved from right side)
        self.info_panel = TradeInfoPanel()
        self.info_panel.setMaximumWidth(280)
        left_layout.addWidget(self.info_panel)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # Right panel - chart (extended to full right)
        chart_panel = QFrame()
        chart_layout = QVBoxLayout(chart_panel)

        # Navigation bar
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self._prev_trade)
        nav_layout.addWidget(self.prev_btn)

        self.trade_counter = QLabel("0 / 0")
        self.trade_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trade_counter.setFont(QFont('Consolas', 12, QFont.Weight.Bold))
        nav_layout.addWidget(self.trade_counter)

        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self._next_trade)
        nav_layout.addWidget(self.next_btn)

        nav_layout.addStretch()

        # Mode toggle button
        self.mode_btn = QPushButton("Bars Mode")
        self.mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #a569bd;
            }
        """)
        self.mode_btn.clicked.connect(self._toggle_mode)
        nav_layout.addWidget(self.mode_btn)

        chart_layout.addLayout(nav_layout)

        # Chart (now takes full width)
        self.chart = PriceChartWidget()
        chart_layout.addWidget(self.chart)

        main_layout.addWidget(chart_panel, stretch=3)

    def _load_trades(self):
        """Load trades from database."""
        conn = duckdb.connect(self.trades_db_path, read_only=True)
        df = conn.execute("SELECT * FROM trades ORDER BY entry_time").fetchdf()
        conn.close()

        self.all_trades = df.to_dict('records')
        self.trades = self.all_trades.copy()

        self._update_list()
        self._update_stats()

        if self.trades:
            self.trade_list.setCurrentRow(0)

    def _filter_trades(self, filter_text: str):
        """Filter trades based on selection."""
        if filter_text == "All Trades":
            self.trades = self.all_trades.copy()
        elif filter_text == "Winners Only":
            self.trades = [t for t in self.all_trades if t['outcome'] == 'winner']
        elif filter_text == "Losers Only":
            self.trades = [t for t in self.all_trades if t['outcome'] == 'loser']
        elif filter_text == "Stop Losses":
            self.trades = [t for t in self.all_trades if t['exit_reason'] == 'stop']
        elif filter_text == "Holds":
            self.trades = [t for t in self.all_trades if t['exit_reason'] == 'hold']

        self._update_list()
        self._update_stats()

        if self.trades:
            self.trade_list.setCurrentRow(0)

    def _update_list(self):
        """Update the trade list widget."""
        self.trade_list.clear()
        for trade in self.trades:
            item = TradeListItem(trade)
            self.trade_list.addItem(item)

    def _update_stats(self):
        """Update stats label."""
        if not self.trades:
            self.stats_label.setText("No trades")
            return

        winners = sum(1 for t in self.trades if t['outcome'] == 'winner')
        losers = len(self.trades) - winners
        avg_pnl = sum(t['final_pnl_pct'] for t in self.trades) / len(self.trades)
        total_pnl = sum(t['final_pnl_pct'] for t in self.trades)

        self.stats_label.setText(
            f"<b>{len(self.trades)}</b> trades | "
            f"<span style='color: #27ae60;'>{winners} W</span> / "
            f"<span style='color: #e74c3c;'>{losers} L</span> | "
            f"Avg: {avg_pnl:+.2f}% | Total: {total_pnl:+.1f}%"
        )

    def _on_trade_selected(self, row: int):
        """Handle trade selection."""
        if row < 0 or row >= len(self.trades):
            return

        self.current_index = row
        trade = self.trades[row]

        # Update counter
        self.trade_counter.setText(f"{row + 1} / {len(self.trades)}")

        # Update info panel
        self.info_panel.update_trade(trade)

        # Load price and feature data
        price_data, feature_data = self._load_trade_data(trade)

        # Update chart
        self.chart.plot_trade(trade, price_data, feature_data)

    def _load_trade_data(self, trade: dict) -> tuple:
        """Load price and feature data for a trade."""
        conn = duckdb.connect(self.market_db_path, read_only=True)

        symbol = trade['symbol']
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']

        # Window: 20 min before entry to 20 min after exit
        start_time = entry_time - timedelta(minutes=20)
        end_time = exit_time + timedelta(minutes=20)

        # Load price data from trades table
        price_df = conn.execute(f"""
            SELECT
                DATE_TRUNC('minute', timestamp) as ts,
                LAST(price) as close
            FROM trades
            WHERE symbol = '{symbol}'
              AND timestamp >= '{start_time}'
              AND timestamp <= '{end_time}'
            GROUP BY DATE_TRUNC('minute', timestamp)
            ORDER BY ts
        """).fetchdf()

        # Load feature data with 60 extra minutes for rolling normalization
        # This matches the paper trading's 60-minute rolling z-score window
        lookback_start = start_time - timedelta(minutes=60)

        feature_df = conn.execute(f"""
            SELECT
                timestamp,
                -- Momentum (blue)
                returns as returns_norm,
                returns_5m as returns_5m_norm,
                MACD as macd_norm,
                MACD_hist as macd_hist_norm,
                RSI_14 as rsi_norm,
                -- Volatility (orange)
                NATR as natr_norm,
                BB_width as bb_width_norm,
                BB_squeeze as bb_squeeze_norm,
                -- Volume (green)
                volume_zscore as vol_zscore_norm,
                volume_roc as vol_roc_norm,
                OBV as obv_norm,
                VWAP_distance as vwap_dist_norm,
                -- Microstructure (purple)
                trade_count as trade_cnt_norm,
                buy_sell_ratio as buy_sell_norm,
                order_flow_imbalance as flow_imbal_norm,
                vpin as vpin_norm,
                -- Order Book (red) - directly from features table
                bid_ask_spread_pct as spread_pct_norm,
                order_book_depth_ratio as depth_ratio_norm,
                large_order_imbalance as lg_imbal_norm
            FROM features
            WHERE symbol = '{symbol}'
              AND timestamp >= '{lookback_start}'
              AND timestamp <= '{end_time}'
            ORDER BY timestamp
        """).fetchdf()

        # Fill missing columns with defaults
        for col in FEATURE_ORDER:
            if col not in feature_df.columns:
                feature_df[col] = 0.5

        # Apply rolling 60-minute z-score normalization (matches paper trading)
        # This computes z-score at each point using only the previous 60 minutes of data
        if len(feature_df) > 0 and 'timestamp' in feature_df.columns:
            feature_df = feature_df.set_index('timestamp')

            # Features requiring rolling z-score normalization
            zscore_cols = ['natr_norm', 'bb_width_norm', 'bb_squeeze_norm', 'vol_zscore_norm',
                           'vol_roc_norm', 'obv_norm', 'vwap_dist_norm', 'spread_pct_norm',
                           'trade_cnt_norm', 'flow_imbal_norm', 'vpin_norm',
                           'returns_norm', 'returns_5m_norm', 'macd_norm', 'macd_hist_norm',
                           'depth_ratio_norm']

            for col in zscore_cols:
                if col in feature_df.columns:
                    vals = feature_df[col].fillna(0)
                    # Rolling 60-minute window mean and std
                    rolling_mean = vals.rolling('60min', min_periods=10).mean()
                    rolling_std = vals.rolling('60min', min_periods=10).std()
                    rolling_std = rolling_std.replace(0, 1)  # Avoid division by zero
                    # Z-score, clip to [-3, +3], map to [0, 1]
                    z = np.clip((vals - rolling_mean) / rolling_std, -3, 3)
                    feature_df[col] = (z + 3) / 6
                    feature_df[col] = feature_df[col].fillna(0.5)

            feature_df = feature_df.reset_index()

            # Filter to display window (remove the 60-min lookback data)
            feature_df = feature_df[feature_df['timestamp'] >= start_time]

        # RSI is 0-100, normalize to 0-1 (no rolling needed)
        if 'rsi_norm' in feature_df.columns:
            feature_df['rsi_norm'] = feature_df['rsi_norm'].fillna(50) / 100

        # buy_sell_ratio: 0-2 range (0=all sells, 1=balanced, 2=all buys) -> 0-1
        if 'buy_sell_norm' in feature_df.columns:
            feature_df['buy_sell_norm'] = feature_df['buy_sell_norm'].fillna(1).clip(0, 2) / 2

        # lg_imbal_norm (large_order_imbalance): -1 to 1 range -> 0-1
        if 'lg_imbal_norm' in feature_df.columns:
            feature_df['lg_imbal_norm'] = (feature_df['lg_imbal_norm'].fillna(0).clip(-1, 1) + 1) / 2

        conn.close()

        return price_df, feature_df

    def _prev_trade(self):
        """Go to previous trade."""
        if self.current_index > 0:
            self.trade_list.setCurrentRow(self.current_index - 1)

    def _next_trade(self):
        """Go to next trade."""
        if self.current_index < len(self.trades) - 1:
            self.trade_list.setCurrentRow(self.current_index + 1)

    def _toggle_mode(self):
        """Toggle between line and bars chart modes."""
        new_mode = self.chart.toggle_mode()
        if new_mode == DisplayMode.BARS:
            self.mode_btn.setText("Line Mode")
            self.mode_btn.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    padding: 8px 20px;
                }
                QPushButton:hover {
                    background-color: #2ecc71;
                }
            """)
        else:
            self.mode_btn.setText("Bars Mode")
            self.mode_btn.setStyleSheet("""
                QPushButton {
                    background-color: #9b59b6;
                    padding: 8px 20px;
                }
                QPushButton:hover {
                    background-color: #a569bd;
                }
            """)
        self.chart.replot()


def main():
    # Get database paths
    if len(sys.argv) > 1:
        trades_db = sys.argv[1]
    else:
        trades_db = "backtest_trades.duckdb"

    if not Path(trades_db).exists():
        print(f"Error: {trades_db} not found")
        print("Run the backtest first to create the trades database")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = TradeVisualizerWindow(trades_db)  # market_db defaults to pythia.duckdb
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
