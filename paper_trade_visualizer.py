#!/usr/bin/env python3
"""
Paper Trade Visualizer GUI

PyQt6 GUI for browsing paper trades with price visualization.
Shows all trades from the multi-strategy paper trading system.

Usage:
    python paper_trade_visualizer.py
    python paper_trade_visualizer.py --state paper_trading_state.json
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict

import paper_trades

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


# Feature Categories and Colors (from spike_visualizer)
FEATURE_CATEGORIES = {
    'Momentum': {
        'features': ['returns', 'returns_5m', 'returns_15m', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI_14'],
        'color': '#3498db'  # Blue
    },
    'Volatility': {
        'features': ['NATR', 'NATR_delta', 'BB_width', 'BB_squeeze'],
        'color': '#e67e22'  # Orange
    },
    'Volume': {
        'features': ['volume_zscore', 'volume_zscore_5m', 'volume_zscore_15m', 'volume_zscore_delta', 'volume_roc', 'OBV'],
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
    },
    'Model': {
        'features': ['V3_prob'],
        'color': '#f1c40f'  # Yellow
    }
}

# Build reverse mapping
FEATURE_TO_COLOR = {}
for cat_name, cat_info in FEATURE_CATEGORIES.items():
    for feat in cat_info['features']:
        FEATURE_TO_COLOR[feat] = cat_info['color']

ALL_FEATURES = list(FEATURE_TO_COLOR.keys())


# Dark Theme Stylesheet
DARK_STYLESHEET = """
QMainWindow {
    background-color: #1a1a2e;
}
QWidget {
    background-color: #1a1a2e;
    color: #eee;
    font-family: 'Monaco', 'Menlo', monospace;
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

# Strategy colors
STRATEGY_COLORS = {
    'A_no_natr': '#3498db',      # Blue
    'B_no_ret5m': '#e67e22',     # Orange
    'C_v3_vol_only': '#9b59b6',  # Purple
}


class TradeListItem(QListWidgetItem):
    """Custom list item for paper trades."""

    def __init__(self, trade: dict, strategy: str):
        self.trade = trade
        self.strategy = strategy

        # Format display text
        symbol = trade.get('symbol', '???')
        pnl = trade.get('realized_pnl', 0)
        exit_reason = trade.get('exit_reason', '?')
        is_open = trade.get('exit_time') is None

        if is_open:
            text = f"[OPEN] {symbol:<12} {strategy[:1]}"
            super().__init__(text)
            self.setForeground(QColor("#f1c40f"))  # Yellow for open
        else:
            pnl_str = f"${pnl:+.0f}"
            text = f"{symbol:<12} {pnl_str:>8} | {exit_reason:<15} | {strategy[:1]}"
            super().__init__(text)

            # Color based on P&L
            if pnl > 0:
                self.setForeground(QColor("#27ae60"))  # Green
            else:
                self.setForeground(QColor("#e74c3c"))  # Red


class PriceChartWidget(FigureCanvas):
    """Matplotlib widget for price charts with entry/exit markers."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 6), facecolor='#1a1a2e')
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self._style_axes()

    def _style_axes(self):
        """Apply dark theme to axes."""
        self.ax.set_facecolor('#16213e')
        self.ax.tick_params(colors='#aaa')
        self.ax.spines['bottom'].set_color('#444')
        self.ax.spines['top'].set_color('#444')
        self.ax.spines['left'].set_color('#444')
        self.ax.spines['right'].set_color('#444')
        self.ax.xaxis.label.set_color('#aaa')
        self.ax.yaxis.label.set_color('#aaa')
        self.ax.title.set_color('#eee')

    def plot_trade(self, trade: dict, strategy: str, price_data: pd.DataFrame):
        """Plot price chart with entry/exit markers."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        if price_data.empty:
            self.ax.text(0.5, 0.5, 'No price data available',
                        transform=self.ax.transAxes, ha='center', color='#aaa', fontsize=14)
            self.fig.tight_layout()
            self.draw()
            return

        symbol = trade.get('symbol', '???')
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        entry_time_str = trade.get('entry_time', '')
        exit_time_str = trade.get('exit_time', '')
        exit_reason = trade.get('exit_reason', '')
        pnl = trade.get('realized_pnl', 0)
        is_open = exit_time_str is None or exit_time_str == ''

        # Parse times
        try:
            entry_time = datetime.fromisoformat(entry_time_str)
            if entry_time.tzinfo:
                entry_time = entry_time.replace(tzinfo=None)
        except:
            entry_time = None

        try:
            if exit_time_str:
                exit_time = datetime.fromisoformat(exit_time_str)
                if exit_time.tzinfo:
                    exit_time = exit_time.replace(tzinfo=None)
            else:
                exit_time = None
        except:
            exit_time = None

        # Plot price line
        strategy_color = STRATEGY_COLORS.get(strategy, '#3498db')
        self.ax.plot(price_data['timestamp'], price_data['close'],
                    color=strategy_color, linewidth=2, label='Price')

        # Entry marker
        if entry_time:
            self.ax.axvline(entry_time, color='#f1c40f', linestyle='--', alpha=0.8, linewidth=2)
            self.ax.scatter([entry_time], [entry_price], color='#f1c40f', s=150, zorder=5,
                           marker='^', edgecolors='white', linewidths=2)

        # Exit marker (if closed)
        if exit_time and not is_open:
            exit_color = '#27ae60' if pnl > 0 else '#e74c3c'
            self.ax.axvline(exit_time, color=exit_color, linestyle='--', alpha=0.8, linewidth=2)
            self.ax.scatter([exit_time], [exit_price], color=exit_color, s=150, zorder=5,
                           marker='v', edgecolors='white', linewidths=2)

        # Stop loss line at -1%
        stop_price = entry_price * 0.99
        self.ax.axhline(stop_price, color='#e74c3c', linestyle=':', alpha=0.5, label='-1% Stop')

        # Take profit levels
        tp1_price = entry_price * 1.20
        tp2_price = entry_price * 1.30
        self.ax.axhline(tp1_price, color='#27ae60', linestyle=':', alpha=0.3, label='+20% TP')
        self.ax.axhline(tp2_price, color='#27ae60', linestyle=':', alpha=0.3)

        # Entry price line
        self.ax.axhline(entry_price, color='#f1c40f', linestyle='-', alpha=0.3)

        # Title
        if is_open:
            status = "OPEN"
            title_color = '#f1c40f'
        elif pnl > 0:
            status = "WINNER"
            title_color = '#27ae60'
        else:
            status = "LOSER"
            title_color = '#e74c3c'

        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if exit_price and entry_price else 0

        title = f"{symbol} | [{strategy}] | {status}"
        if not is_open:
            title += f" | ${pnl:+.2f} ({pnl_pct:+.1f}%) | {exit_reason}"

        self.ax.set_title(title, fontsize=12, fontweight='bold', color=title_color)
        self.ax.set_ylabel('Price ($)', color='#aaa')
        self.ax.set_xlabel('Time', color='#aaa')

        # Format x-axis
        self.ax.tick_params(axis='x', rotation=45)

        self.ax.legend(loc='upper left', fontsize=8)
        self.ax.grid(True, alpha=0.2, color='#444')

        self.fig.tight_layout()
        self.draw()

    def plot_feature_bars(self, trade: dict, strategy: str, feature_data: List[Dict], price_data: pd.DataFrame):
        """Plot feature bars mode - colored bars for each feature with price line overlay."""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        symbol = trade.get('symbol', '???')
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        pnl = trade.get('realized_pnl', 0)
        is_open = trade.get('exit_time') is None

        if not feature_data:
            self.ax.text(0.5, 0.5, 'No feature data available\n(waiting for collector to record features)',
                        transform=self.ax.transAxes, ha='center', color='#aaa', fontsize=12)
            self.fig.tight_layout()
            self.draw()
            return

        # Find features that exist in the data
        sample = feature_data[0] if feature_data else {}
        available_features = [f for f in ALL_FEATURES if f in sample]

        if not available_features:
            # Try to find any numeric features
            available_features = [k for k, v in sample.items()
                                if isinstance(v, (int, float)) and k != 'timestamp'][:20]

        if not available_features:
            self.ax.text(0.5, 0.5, 'No features found in data',
                        transform=self.ax.transAxes, ha='center', color='#aaa', fontsize=12)
            self.fig.tight_layout()
            self.draw()
            return

        n_features = len(available_features)
        n_timepoints = len(feature_data)

        # Build feature matrix (timepoints x features)
        feature_matrix = np.zeros((n_timepoints, n_features))
        for t, snapshot in enumerate(feature_data):
            for f_idx, feat in enumerate(available_features):
                if feat in snapshot:
                    feature_matrix[t, f_idx] = snapshot[feat]

        # Normalize each feature to z-scores, then map to [0, 1]
        for f_idx in range(n_features):
            col = feature_matrix[:, f_idx]
            mean = np.nanmean(col)
            std = np.nanstd(col)
            if std > 0:
                feature_matrix[:, f_idx] = (col - mean) / std
            else:
                feature_matrix[:, f_idx] = 0

        # Clip to [-3, 3] and map to [0, 1]
        feature_matrix = np.clip(feature_matrix, -3, 3)
        feature_matrix = (feature_matrix + 3) / 6

        # Draw feature bars
        bar_width = 0.8 / n_features if n_features > 0 else 0.8
        x = np.arange(n_timepoints)

        for f_idx, feat in enumerate(available_features):
            color = FEATURE_TO_COLOR.get(feat, '#888888')
            offset = (f_idx - n_features/2) * bar_width
            self.ax.bar(x + offset, feature_matrix[:, f_idx],
                       width=bar_width, color=color, alpha=0.6, zorder=1)

        # Overlay price line if we have price data
        if not price_data.empty and len(price_data) > 0:
            prices = price_data['close'].values
            price_norm = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

            # Scale x to match feature data timepoints
            x_price = np.linspace(0, n_timepoints - 1, len(prices))
            self.ax.plot(x_price, price_norm, color='#00ff88', linewidth=3, alpha=0.9, zorder=10)

            # Mark entry point
            self.ax.axhline(y=0.5, color='#f1c40f', linestyle='--', alpha=0.5, zorder=5)
            self.ax.scatter([0], [(entry_price - prices.min()) / (prices.max() - prices.min() + 1e-10)],
                          color='#f1c40f', s=200, marker='^', zorder=15, edgecolors='white', linewidths=2)

        # Configure axes
        self.ax.set_xlim(-0.5, n_timepoints - 0.5)
        self.ax.set_ylim(-0.05, 1.05)

        # Title
        if is_open:
            status = "OPEN"
            title_color = '#f1c40f'
        elif pnl > 0:
            status = "WINNER"
            title_color = '#27ae60'
        else:
            status = "LOSER"
            title_color = '#e74c3c'

        self.ax.set_title(f"{symbol} | [{strategy}] | {status} | {n_features} features × {n_timepoints} timepoints",
                         fontsize=11, fontweight='bold', color=title_color)

        # Add feature legend at bottom
        legend_text = " | ".join([f for f in available_features[:8]])  # First 8 features
        if len(available_features) > 8:
            legend_text += f" | +{len(available_features) - 8} more"
        self.ax.set_xlabel(legend_text, fontsize=8, color='#888')

        self.ax.set_ylabel('Normalized (0-1)', color='#aaa')
        self.ax.grid(True, alpha=0.2, color='#444')

        self.fig.tight_layout()
        self.draw()


class TradeInfoPanel(QFrame):
    """Panel showing trade details."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        self.title_label = QLabel("Trade Details")
        self.title_label.setFont(QFont('Monaco', 14, QFont.Weight.Bold))
        layout.addWidget(self.title_label)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setFont(QFont('Monaco', 10))
        layout.addWidget(self.info_label)

        layout.addStretch()

    def update_trade(self, trade: dict, strategy: str):
        """Update panel with trade info."""
        symbol = trade.get('symbol', '???')
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        entry_time = trade.get('entry_time', '')
        exit_time = trade.get('exit_time', '')
        exit_reason = trade.get('exit_reason', '')
        pnl = trade.get('realized_pnl', 0)
        quantity = trade.get('quantity', 0)
        position_size = trade.get('position_size', 0)
        is_open = exit_time is None or exit_time == ''

        # Format times for display
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            if entry_dt.tzinfo:
                entry_dt = entry_dt.astimezone()
            entry_str = entry_dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            entry_str = entry_time

        if is_open:
            exit_str = "(still open)"
            pnl_pct = 0
            outcome_color = "#f1c40f"
            outcome = "OPEN"
        else:
            try:
                exit_dt = datetime.fromisoformat(exit_time)
                if exit_dt.tzinfo:
                    exit_dt = exit_dt.astimezone()
                exit_str = exit_dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                exit_str = exit_time
            pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0
            outcome_color = "#27ae60" if pnl > 0 else "#e74c3c"
            outcome = "WINNER" if pnl > 0 else "LOSER"

        strategy_color = STRATEGY_COLORS.get(strategy, '#3498db')

        info = f"""
<b style='color: {outcome_color}; font-size: 16px;'>{outcome}</b>

<b>Symbol:</b> {symbol}
<b style='color: {strategy_color};'>Strategy:</b> {strategy}

<b>Entry Time:</b> {entry_str}
<b>Exit Time:</b> {exit_str}

<b>Entry Price:</b> ${entry_price:.6f}
<b>Exit Price:</b> {f'${exit_price:.6f}' if exit_price else 'N/A'}

<b>Position Size:</b> ${position_size:,.2f}
<b>Quantity:</b> {quantity:,.2f}

<b style='color: {outcome_color};'>P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)</b>

<b>Exit Reason:</b> {exit_reason if exit_reason else 'N/A'}
"""
        self.info_label.setText(info)


class PaperTradeVisualizer(QMainWindow):
    """Main window for paper trade visualization."""

    def __init__(self):
        super().__init__()

        self.all_trades = []
        self.trades = []
        self.current_index = 0
        self.display_mode = "price"  # "price" or "features"

        self._setup_ui()
        self._load_trades()

    def _setup_ui(self):
        """Setup the UI layout."""
        self.setWindowTitle("Paper Trade Visualizer")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(DARK_STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel - trade list + filters
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(320)
        left_panel.setMinimumWidth(280)

        # Filter controls
        filter_layout = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "All Trades",
            "Open Only",
            "Closed Only",
            "Winners Only",
            "Losers Only",
            "A_no_natr",
            "B_no_ret5m",
            "C_v3_vol_only"
        ])
        self.filter_combo.currentTextChanged.connect(self._filter_trades)
        filter_layout.addWidget(self.filter_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._load_trades)
        filter_layout.addWidget(self.refresh_btn)

        left_layout.addLayout(filter_layout)

        # Stats label
        self.stats_label = QLabel("")
        self.stats_label.setFont(QFont('Monaco', 9))
        left_layout.addWidget(self.stats_label)

        # Trade list
        self.trade_list = QListWidget()
        self.trade_list.currentRowChanged.connect(self._on_trade_selected)
        left_layout.addWidget(self.trade_list)

        # Trade details panel
        self.info_panel = TradeInfoPanel()
        left_layout.addWidget(self.info_panel)

        main_layout.addWidget(left_panel)

        # Right panel - chart
        chart_panel = QFrame()
        chart_layout = QVBoxLayout(chart_panel)

        # Navigation bar
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self._prev_trade)
        nav_layout.addWidget(self.prev_btn)

        self.trade_counter = QLabel("0 / 0")
        self.trade_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trade_counter.setFont(QFont('Monaco', 12, QFont.Weight.Bold))
        nav_layout.addWidget(self.trade_counter)

        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self._next_trade)
        nav_layout.addWidget(self.next_btn)

        nav_layout.addStretch()

        # Mode toggle button
        self.mode_btn = QPushButton("Mode: Price")
        self.mode_btn.setFixedWidth(140)
        self.mode_btn.clicked.connect(self._toggle_mode)
        nav_layout.addWidget(self.mode_btn)

        chart_layout.addLayout(nav_layout)

        # Chart
        self.chart = PriceChartWidget()
        chart_layout.addWidget(self.chart)

        main_layout.addWidget(chart_panel, stretch=3)

    def _load_trades(self):
        """Load trades from SQLite database."""
        try:
            trades = paper_trades.get_all_trades()
        except Exception as e:
            self.stats_label.setText(f"Error loading trades: {e}")
            return

        self.all_trades = []

        for trade in trades:
            # Convert SQLite row to dict format expected by UI
            t = {
                'symbol': trade['symbol'],
                'entry_price': trade['entry_price'],
                'entry_time': trade['entry_time'],
                'exit_price': trade['exit_price'],
                'exit_time': trade['exit_time'],
                'exit_reason': trade['exit_reason'] or '',
                'position_size': trade['position_size'],
                'quantity': trade['quantity'],
                'realized_pnl': trade['realized_pnl'] or 0,
                '_strategy': trade['strategy'],
                '_is_open': trade['is_open'] == 1,
            }
            self.all_trades.append(t)

        self.trades = self.all_trades.copy()
        self._update_list()
        self._update_stats()

        if self.trades:
            self.trade_list.setCurrentRow(0)

    def _filter_trades(self, filter_text: str):
        """Filter trades based on selection."""
        if filter_text == "All Trades":
            self.trades = self.all_trades.copy()
        elif filter_text == "Open Only":
            self.trades = [t for t in self.all_trades if t.get('_is_open', False)]
        elif filter_text == "Closed Only":
            self.trades = [t for t in self.all_trades if not t.get('_is_open', False)]
        elif filter_text == "Winners Only":
            self.trades = [t for t in self.all_trades if t.get('realized_pnl', 0) > 0]
        elif filter_text == "Losers Only":
            self.trades = [t for t in self.all_trades if t.get('realized_pnl', 0) <= 0 and not t.get('_is_open', False)]
        elif filter_text in STRATEGY_COLORS:
            self.trades = [t for t in self.all_trades if t.get('_strategy') == filter_text]
        else:
            self.trades = self.all_trades.copy()

        self._update_list()
        self._update_stats()

        if self.trades:
            self.trade_list.setCurrentRow(0)

    def _update_list(self):
        """Update the trade list widget."""
        self.trade_list.clear()
        for trade in self.trades:
            strategy = trade.get('_strategy', '?')
            item = TradeListItem(trade, strategy)
            self.trade_list.addItem(item)

    def _update_stats(self):
        """Update stats label."""
        if not self.trades:
            self.stats_label.setText("No trades")
            return

        closed = [t for t in self.trades if not t.get('_is_open', False)]
        open_count = len(self.trades) - len(closed)
        winners = sum(1 for t in closed if t.get('realized_pnl', 0) > 0)
        losers = len(closed) - winners
        total_pnl = sum(t.get('realized_pnl', 0) for t in closed)

        self.stats_label.setText(
            f"<b>{len(self.trades)}</b> trades | "
            f"<span style='color: #f1c40f;'>{open_count} open</span> | "
            f"<span style='color: #27ae60;'>{winners} W</span> / "
            f"<span style='color: #e74c3c;'>{losers} L</span> | "
            f"Total: <b>${total_pnl:+.2f}</b>"
        )

    def _on_trade_selected(self, row: int):
        """Handle trade selection."""
        if row < 0 or row >= len(self.trades):
            return

        self.current_index = row
        trade = self.trades[row]
        strategy = trade.get('_strategy', '?')

        # Update counter
        self.trade_counter.setText(f"{row + 1} / {len(self.trades)}")

        # Update info panel
        self.info_panel.update_trade(trade, strategy)

        # Load data and update chart based on mode
        price_data = self._load_price_data(trade)

        if self.display_mode == "features":
            feature_data = self._load_feature_data(trade)
            self.chart.plot_feature_bars(trade, strategy, feature_data, price_data)
        else:
            self.chart.plot_trade(trade, strategy, price_data)

    def _toggle_mode(self):
        """Toggle between price and feature bar display modes."""
        if self.display_mode == "price":
            self.display_mode = "features"
            self.mode_btn.setText("Mode: Features")
        else:
            self.display_mode = "price"
            self.mode_btn.setText("Mode: Price")

        # Refresh current trade view
        if self.trades and self.current_index >= 0:
            self._on_trade_selected(self.current_index)

    def _load_feature_data(self, trade: dict) -> List[Dict]:
        """Load feature data for a trade from SQLite."""
        symbol = trade.get('symbol', '')
        entry_time_str = trade.get('entry_time', '')
        exit_time_str = trade.get('exit_time', '')

        try:
            entry_time = datetime.fromisoformat(entry_time_str)
            if entry_time.tzinfo:
                entry_time = entry_time.replace(tzinfo=None)
        except Exception:
            return []

        # Window: entry to exit (or now if open)
        start_time = entry_time - timedelta(minutes=5)

        if exit_time_str:
            try:
                exit_time = datetime.fromisoformat(exit_time_str)
                if exit_time.tzinfo:
                    exit_time = exit_time.replace(tzinfo=None)
                end_time = exit_time + timedelta(minutes=5)
            except:
                end_time = datetime.now() + timedelta(minutes=5)
        else:
            end_time = datetime.now() + timedelta(minutes=5)

        try:
            features = paper_trades.get_feature_data(symbol, start_time, end_time)
            print(f"[features] {symbol}: {len(features)} snapshots")
            return features
        except Exception as e:
            print(f"Error loading feature data: {e}")
            return []

    def _load_price_data(self, trade: dict) -> pd.DataFrame:
        """Load price data for a trade from SQLite."""
        symbol = trade.get('symbol', '')
        entry_time_str = trade.get('entry_time', '')
        exit_time_str = trade.get('exit_time', '')

        try:
            entry_time = datetime.fromisoformat(entry_time_str)
            if entry_time.tzinfo:
                entry_time = entry_time.replace(tzinfo=None)
        except Exception as e:
            print(f"Error parsing entry time: {e}")
            return pd.DataFrame()

        # Window: 10 min before entry to exit (or now if open) + 10 min
        start_time = entry_time - timedelta(minutes=10)

        if exit_time_str:
            try:
                exit_time = datetime.fromisoformat(exit_time_str)
                if exit_time.tzinfo:
                    exit_time = exit_time.replace(tzinfo=None)
                end_time = exit_time + timedelta(minutes=10)
            except:
                end_time = datetime.now() + timedelta(minutes=10)
        else:
            end_time = datetime.now() + timedelta(minutes=10)

        try:
            prices = paper_trades.get_price_data(symbol, start_time, end_time)
            if prices:
                df = pd.DataFrame(prices)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"[sqlite] {symbol}: {len(df)} rows ({start_time} to {end_time})")
                return df
            else:
                print(f"[sqlite] {symbol}: No price data found")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading price data: {e}")
            return pd.DataFrame()

    def _prev_trade(self):
        """Go to previous trade."""
        if self.current_index > 0:
            self.trade_list.setCurrentRow(self.current_index - 1)

    def _next_trade(self):
        """Go to next trade."""
        if self.current_index < len(self.trades) - 1:
            self.trade_list.setCurrentRow(self.current_index + 1)


def main():
    # Check SQLite database exists
    if not paper_trades.DB_PATH.exists():
        print(f"Note: {paper_trades.DB_PATH} will be created")
        print("Make sure the paper trading collector is running to populate trades")

    app = QApplication(sys.argv)
    window = PaperTradeVisualizer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
