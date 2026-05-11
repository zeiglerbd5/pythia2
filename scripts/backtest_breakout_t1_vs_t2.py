#!/usr/bin/env python3
"""
Backtest Breakout Hunter: T+1 Entry vs T+2 Entry

Compares:
1. T+1 entry: Enter at T+1 close when T+1 > 0%
2. T+2 entry: Enter at T+2 open when T+2 resumes >5%

Also analyzes max T+1 gain filter (don't enter if T+1 already >25%)

Strategy parameters:
- T+0: Detect breakout >5% with volume >3x average
- T+1: Must be positive (>0%)
- Stop loss: 8%
- Profit locks: +5%->-2%, +10%->0%, +15%->5%, +25%->12%, +40%->25%
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import sqlite3
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Strategy parameters
VOLUME_THRESHOLD = 3.0          # Volume must be 3x 24h average
INITIAL_MOVE_PCT = 5.0          # T+0 breakout must be >5%
T1_MIN_RETURN_PCT = 0.0         # T+1 must be positive (>0%)
T2_RESUMPTION_PCT = 5.0         # T+2 return from T+1 close must be >5%

INITIAL_STOP_LOSS_PCT = 0.08    # 8% initial stop loss

PROFIT_LOCK_LEVELS = [
    (0.05, -0.02),  # Once up 5%, lock -2%
    (0.10, 0.00),   # Once up 10%, lock breakeven
    (0.15, 0.05),   # Once up 15%, lock 5% profit
    (0.25, 0.12),   # Once up 25%, lock 12% profit
    (0.40, 0.25),   # Once up 40%, lock 25% profit
]

TRAIL_TRIGGER_PCT = 0.20        # Activate trailing after +20%
TRAIL_STOP_PCT = 0.10           # Trail at 10% below high
TRAIL_TIGHTEN_LEVELS = [
    (0.30, 0.08),
    (0.50, 0.06),
    (0.70, 0.05),
]

MAX_TAKE_PROFIT_PCT = 0.80
MAX_HOLD_HOURS = 48
FEE_RATE = 0.0055


@dataclass
class Signal:
    """Breakout signal"""
    symbol: str
    signal_time: datetime
    signal_price: float  # T+0 close
    initial_move_pct: float
    volume_ratio: float
    pre_signal_volume: float = 0.0
    t1_close: Optional[float] = None
    t1_return: Optional[float] = None
    t1_time: Optional[datetime] = None
    t1_volume: Optional[float] = None
    t2_open: Optional[float] = None
    t2_return: Optional[float] = None


@dataclass
class Trade:
    """Trade record"""
    symbol: str
    entry_time: datetime
    entry_price: float
    signal: Signal
    status: str = 'open'
    highest_price: float = 0.0
    current_stop: float = 0.0
    locked_profit_pct: float = -1.0
    trailing_active: bool = False
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None


def load_hourly_data():
    """Load and aggregate OHLCV data to hourly"""
    print("Loading data from feature_buffer.db...")
    conn = sqlite3.connect('/Users/bz/Pythia2/data/feature_buffer.db')

    df = pd.read_sql('''
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM ohlcv WHERE volume > 0 AND open > 0
        ORDER BY symbol, timestamp
    ''', conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df['hour'] = df['timestamp'].dt.floor('h')

    # Aggregate to hourly
    hourly = df.groupby(['symbol', 'hour']).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    hourly = hourly.rename(columns={'hour': 'timestamp'})

    # Add duckdb data if available for more history
    try:
        conn2 = duckdb.connect('/Users/bz/Pythia2/data/pythia.duckdb', read_only=True)
        df2 = conn2.execute('''
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM ohlcv WHERE volume > 0 AND open > 0
            ORDER BY symbol, timestamp
        ''').df()
        conn2.close()

        df2['timestamp'] = pd.to_datetime(df2['timestamp'])
        df2['hour'] = df2['timestamp'].dt.floor('h')

        hourly2 = df2.groupby(['symbol', 'hour']).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        hourly2 = hourly2.rename(columns={'hour': 'timestamp'})

        # Combine, preferring newer data
        combined = pd.concat([hourly2, hourly])
        combined = combined.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
        combined = combined.sort_values(['symbol', 'timestamp'])
        hourly = combined

        print(f"Combined with pythia.duckdb data")
    except Exception as e:
        print(f"Could not load pythia.duckdb: {e}")

    print(f"Loaded {len(hourly)} hourly candles for {hourly['symbol'].nunique()} symbols")
    print(f"Date range: {hourly['timestamp'].min()} to {hourly['timestamp'].max()}")

    return hourly


def calculate_volume_ratio(symbol: str, current_volume: float,
                          hourly_by_symbol: Dict[str, pd.DataFrame],
                          current_idx: int) -> float:
    """Calculate volume ratio vs 24h average"""
    if symbol not in hourly_by_symbol:
        return 0.0

    df = hourly_by_symbol[symbol]
    if current_idx < 24:
        return 0.0

    # Average of last 24 hours (excluding current)
    avg_vol = df['volume'].iloc[max(0, current_idx-24):current_idx].mean()

    if avg_vol <= 0:
        return 0.0

    return current_volume / avg_vol


def get_pre_signal_volume(symbol: str, hourly_by_symbol: Dict[str, pd.DataFrame],
                         current_idx: int, hours: int = 6) -> float:
    """Get total volume in the N hours before current"""
    if symbol not in hourly_by_symbol:
        return 0.0

    df = hourly_by_symbol[symbol]
    if current_idx < hours + 1:
        return 0.0

    return df['volume'].iloc[current_idx-hours:current_idx].sum()


def check_exit(trade: Trade, current_price: float, current_time: datetime) -> Optional[str]:
    """Check exit conditions with profit locks and trailing stop"""

    # Update highest price
    if current_price > trade.highest_price:
        trade.highest_price = current_price

    # Calculate returns
    peak_return = (trade.highest_price - trade.entry_price) / trade.entry_price
    current_return = (current_price - trade.entry_price) / trade.entry_price

    # Update profit locks
    for threshold, lock_pct in PROFIT_LOCK_LEVELS:
        if peak_return >= threshold and trade.locked_profit_pct < lock_pct:
            trade.locked_profit_pct = lock_pct

    # Check trailing activation
    if not trade.trailing_active and peak_return >= TRAIL_TRIGGER_PCT:
        trade.trailing_active = True

    # Calculate stop levels
    stop_candidates = [trade.entry_price * (1 - INITIAL_STOP_LOSS_PCT)]

    if trade.locked_profit_pct >= -1:  # Has a lock
        if trade.locked_profit_pct >= -0.02:  # Meaningful lock
            lock_stop = trade.entry_price * (1 + trade.locked_profit_pct)
            stop_candidates.append(lock_stop)

    if trade.trailing_active:
        active_trail_pct = TRAIL_STOP_PCT
        for threshold, tighter_pct in TRAIL_TIGHTEN_LEVELS:
            if peak_return >= threshold:
                active_trail_pct = tighter_pct
        trail_stop = trade.highest_price * (1 - active_trail_pct)
        stop_candidates.append(trail_stop)

    trade.current_stop = max(stop_candidates)

    # Check stop
    if current_price <= trade.current_stop:
        if trade.trailing_active:
            return 'trail_stop'
        elif trade.locked_profit_pct >= 0:
            return 'profit_lock'
        else:
            return 'stop_loss'

    # Check take profit
    if current_price >= trade.entry_price * (1 + MAX_TAKE_PROFIT_PCT):
        return 'take_profit'

    # Check max hold
    if current_time >= trade.entry_time + timedelta(hours=MAX_HOLD_HOURS):
        return 'max_hold'

    return None


def run_backtest(hourly: pd.DataFrame, entry_mode: str = 't1',
                 max_t1_filter: Optional[float] = None) -> List[Trade]:
    """
    Run backtest with specified entry mode.

    Args:
        hourly: Hourly OHLCV data
        entry_mode: 't1' for T+1 entry, 't2' for T+2 entry
        max_t1_filter: If set, skip entries where T+1 gain > this value (e.g., 0.25 = 25%)

    Returns:
        List of completed trades
    """
    completed_trades = []
    pending_signals = {}  # symbol -> Signal
    active_trades = {}    # symbol -> Trade

    # Group by symbol for efficient lookup
    hourly_by_symbol = {sym: df.reset_index(drop=True)
                       for sym, df in hourly.groupby('symbol')}

    # Create global timeline of all hours
    all_hours = sorted(hourly['timestamp'].unique())

    # Create hour index lookup for each symbol
    symbol_hour_idx = {}
    for symbol, df in hourly_by_symbol.items():
        symbol_hour_idx[symbol] = {ts: i for i, ts in enumerate(df['timestamp'])}

    for hour_ts in all_hours:
        # Get all candles for this hour
        hour_candles = hourly[hourly['timestamp'] == hour_ts]

        # 1. Check exits for active trades
        for symbol in list(active_trades.keys()):
            sym_candle = hour_candles[hour_candles['symbol'] == symbol]
            if len(sym_candle) == 0:
                continue

            row = sym_candle.iloc[0]
            trade = active_trades[symbol]

            exit_reason = check_exit(trade, row['close'], hour_ts)

            if exit_reason:
                # Exit the trade
                exit_price = row['close']

                # If stopped, assume fill at stop with possible slippage
                if exit_reason in ('stop_loss', 'profit_lock', 'trail_stop'):
                    if exit_price < trade.current_stop:
                        exit_price = trade.current_stop * 0.98

                gross_return = (exit_price - trade.entry_price) / trade.entry_price
                net_return = gross_return - (FEE_RATE * 2)

                trade.status = 'closed'
                trade.exit_time = hour_ts
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.pnl_pct = net_return * 100

                completed_trades.append(trade)
                del active_trades[symbol]

        # 2. Update T+1 data for pending signals
        for symbol in list(pending_signals.keys()):
            signal = pending_signals[symbol]

            # Check if this is T+1 hour (1 hour after signal)
            hours_since = (hour_ts - signal.signal_time).total_seconds() / 3600

            if hours_since >= 1 and signal.t1_close is None:
                sym_candle = hour_candles[hour_candles['symbol'] == symbol]
                if len(sym_candle) > 0:
                    row = sym_candle.iloc[0]
                    signal.t1_close = row['close']
                    signal.t1_time = hour_ts
                    signal.t1_volume = row['volume']
                    signal.t1_return = (signal.t1_close - signal.signal_price) / signal.signal_price * 100

                    # T+1 must be positive
                    if signal.t1_return <= T1_MIN_RETURN_PCT:
                        # Signal failed
                        del pending_signals[symbol]
                        continue

                    # T+1 entry mode - enter now
                    if entry_mode == 't1':
                        # Check max T+1 filter
                        if max_t1_filter is not None and signal.t1_return > max_t1_filter * 100:
                            del pending_signals[symbol]
                            continue

                        # Skip if max positions
                        if len(active_trades) >= 3:
                            del pending_signals[symbol]
                            continue

                        # Enter trade
                        trade = Trade(
                            symbol=symbol,
                            entry_time=hour_ts,
                            entry_price=signal.t1_close,
                            signal=signal,
                            highest_price=signal.t1_close,
                            current_stop=signal.t1_close * (1 - INITIAL_STOP_LOSS_PCT)
                        )
                        active_trades[symbol] = trade
                        del pending_signals[symbol]

        # 3. Check T+2 confirmations (for T+2 entry mode)
        if entry_mode == 't2':
            for symbol in list(pending_signals.keys()):
                signal = pending_signals[symbol]

                if signal.t1_close is None:
                    continue

                hours_since = (hour_ts - signal.signal_time).total_seconds() / 3600

                if hours_since >= 2:
                    sym_candle = hour_candles[hour_candles['symbol'] == symbol]
                    if len(sym_candle) > 0:
                        row = sym_candle.iloc[0]

                        # Calculate T+2 return from T+1 close
                        t2_return = (row['open'] - signal.t1_close) / signal.t1_close * 100

                        # T+2 must resume > threshold
                        if t2_return < T2_RESUMPTION_PCT:
                            del pending_signals[symbol]
                            continue

                        # Skip if max positions
                        if len(active_trades) >= 3:
                            del pending_signals[symbol]
                            continue

                        signal.t2_open = row['open']
                        signal.t2_return = t2_return

                        # Enter at T+2 open
                        trade = Trade(
                            symbol=symbol,
                            entry_time=hour_ts,
                            entry_price=signal.t2_open,
                            signal=signal,
                            highest_price=signal.t2_open,
                            current_stop=signal.t2_open * (1 - INITIAL_STOP_LOSS_PCT)
                        )
                        active_trades[symbol] = trade
                        del pending_signals[symbol]

        # 4. Expire old signals
        for symbol in list(pending_signals.keys()):
            signal = pending_signals[symbol]
            hours_since = (hour_ts - signal.signal_time).total_seconds() / 3600
            if hours_since > 3:
                del pending_signals[symbol]

        # 5. Check for new T+0 breakouts
        for _, row in hour_candles.iterrows():
            symbol = row['symbol']

            # Skip if already have signal or trade
            if symbol in pending_signals or symbol in active_trades:
                continue

            # Skip if max positions
            if len(active_trades) >= 3:
                continue

            # Calculate hourly return
            if row['open'] <= 0:
                continue
            hour_return = (row['close'] - row['open']) / row['open'] * 100

            # Check minimum move
            if hour_return < INITIAL_MOVE_PCT:
                continue

            # Calculate volume ratio
            if symbol not in symbol_hour_idx or hour_ts not in symbol_hour_idx[symbol]:
                continue

            idx = symbol_hour_idx[symbol][hour_ts]
            vol_ratio = calculate_volume_ratio(symbol, row['volume'], hourly_by_symbol, idx)

            if vol_ratio < VOLUME_THRESHOLD:
                continue

            # Get pre-signal volume
            pre_vol = get_pre_signal_volume(symbol, hourly_by_symbol, idx)

            # Create signal
            signal = Signal(
                symbol=symbol,
                signal_time=hour_ts,
                signal_price=row['close'],
                initial_move_pct=hour_return,
                volume_ratio=vol_ratio,
                pre_signal_volume=pre_vol
            )
            pending_signals[symbol] = signal

    # Close any remaining trades at last price
    for symbol, trade in active_trades.items():
        last_hour = hourly[hourly['symbol'] == symbol]['timestamp'].max()
        last_price = hourly[(hourly['symbol'] == symbol) &
                           (hourly['timestamp'] == last_hour)]['close'].iloc[0]

        gross_return = (last_price - trade.entry_price) / trade.entry_price
        net_return = gross_return - (FEE_RATE * 2)

        trade.status = 'closed'
        trade.exit_time = last_hour
        trade.exit_price = last_price
        trade.exit_reason = 'backtest_end'
        trade.pnl_pct = net_return * 100

        completed_trades.append(trade)

    return completed_trades


def analyze_trades(trades: List[Trade], label: str):
    """Analyze and print trade statistics"""
    if not trades:
        print(f"\n{label}: No trades")
        return

    pnls = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Total trades: {len(trades)}")
    print(f"Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
    print(f"Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
    print(f"")
    print(f"Average gain/loss: {np.mean(pnls):+.2f}%")
    print(f"Average win: {np.mean(wins):+.2f}%" if wins else "Average win: N/A")
    print(f"Average loss: {np.mean(losses):+.2f}%" if losses else "Average loss: N/A")
    print(f"")
    print(f"Best trade: {max(pnls):+.2f}%")
    print(f"Worst trade: {min(pnls):+.2f}%")
    print(f"Total P&L: {sum(pnls):+.2f}%")

    # Exit reason breakdown
    exit_reasons = defaultdict(list)
    for t in trades:
        exit_reasons[t.exit_reason].append(t.pnl_pct)

    print(f"\nExit Reasons:")
    for reason, pnls_for_reason in sorted(exit_reasons.items(), key=lambda x: len(x[1]), reverse=True):
        avg = np.mean(pnls_for_reason)
        print(f"  {reason}: {len(pnls_for_reason)} trades, avg {avg:+.2f}%")

    return {
        'total_trades': len(trades),
        'win_rate': len(wins)/len(trades)*100,
        'avg_pnl': np.mean(pnls),
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'total_pnl': sum(pnls)
    }


def run_backtest_t2_relaxed(hourly: pd.DataFrame, t2_threshold: float = 0.0) -> List[Trade]:
    """
    Run T+2 entry backtest with relaxed T+2 threshold.

    Enter at T+2 open if T+1 > 0% and T+2 open >= t2_threshold from T+1 close.
    """
    completed_trades = []
    pending_signals = {}
    active_trades = {}

    hourly_by_symbol = {sym: df.reset_index(drop=True)
                       for sym, df in hourly.groupby('symbol')}
    all_hours = sorted(hourly['timestamp'].unique())
    symbol_hour_idx = {}
    for symbol, df in hourly_by_symbol.items():
        symbol_hour_idx[symbol] = {ts: i for i, ts in enumerate(df['timestamp'])}

    for hour_ts in all_hours:
        hour_candles = hourly[hourly['timestamp'] == hour_ts]

        # 1. Check exits
        for symbol in list(active_trades.keys()):
            sym_candle = hour_candles[hour_candles['symbol'] == symbol]
            if len(sym_candle) == 0:
                continue
            row = sym_candle.iloc[0]
            trade = active_trades[symbol]
            exit_reason = check_exit(trade, row['close'], hour_ts)
            if exit_reason:
                exit_price = row['close']
                if exit_reason in ('stop_loss', 'profit_lock', 'trail_stop'):
                    if exit_price < trade.current_stop:
                        exit_price = trade.current_stop * 0.98
                gross_return = (exit_price - trade.entry_price) / trade.entry_price
                net_return = gross_return - (FEE_RATE * 2)
                trade.status = 'closed'
                trade.exit_time = hour_ts
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.pnl_pct = net_return * 100
                completed_trades.append(trade)
                del active_trades[symbol]

        # 2. Update T+1 data
        for symbol in list(pending_signals.keys()):
            signal = pending_signals[symbol]
            hours_since = (hour_ts - signal.signal_time).total_seconds() / 3600
            if hours_since >= 1 and signal.t1_close is None:
                sym_candle = hour_candles[hour_candles['symbol'] == symbol]
                if len(sym_candle) > 0:
                    row = sym_candle.iloc[0]
                    signal.t1_close = row['close']
                    signal.t1_time = hour_ts
                    signal.t1_return = (signal.t1_close - signal.signal_price) / signal.signal_price * 100
                    if signal.t1_return <= T1_MIN_RETURN_PCT:
                        del pending_signals[symbol]

        # 3. Check T+2 entry
        for symbol in list(pending_signals.keys()):
            signal = pending_signals[symbol]
            if signal.t1_close is None:
                continue
            hours_since = (hour_ts - signal.signal_time).total_seconds() / 3600
            if hours_since >= 2:
                sym_candle = hour_candles[hour_candles['symbol'] == symbol]
                if len(sym_candle) > 0:
                    row = sym_candle.iloc[0]
                    t2_return = (row['open'] - signal.t1_close) / signal.t1_close * 100

                    # Relaxed T+2 threshold
                    if t2_return < t2_threshold:
                        del pending_signals[symbol]
                        continue
                    if len(active_trades) >= 3:
                        del pending_signals[symbol]
                        continue
                    signal.t2_open = row['open']
                    signal.t2_return = t2_return
                    trade = Trade(
                        symbol=symbol,
                        entry_time=hour_ts,
                        entry_price=signal.t2_open,
                        signal=signal,
                        highest_price=signal.t2_open,
                        current_stop=signal.t2_open * (1 - INITIAL_STOP_LOSS_PCT)
                    )
                    active_trades[symbol] = trade
                    del pending_signals[symbol]

        # 4. Expire old signals
        for symbol in list(pending_signals.keys()):
            signal = pending_signals[symbol]
            hours_since = (hour_ts - signal.signal_time).total_seconds() / 3600
            if hours_since > 3:
                del pending_signals[symbol]

        # 5. New T+0 breakouts
        for _, row in hour_candles.iterrows():
            symbol = row['symbol']
            if symbol in pending_signals or symbol in active_trades:
                continue
            if len(active_trades) >= 3:
                continue
            if row['open'] <= 0:
                continue
            hour_return = (row['close'] - row['open']) / row['open'] * 100
            if hour_return < INITIAL_MOVE_PCT:
                continue
            if symbol not in symbol_hour_idx or hour_ts not in symbol_hour_idx[symbol]:
                continue
            idx = symbol_hour_idx[symbol][hour_ts]
            vol_ratio = calculate_volume_ratio(symbol, row['volume'], hourly_by_symbol, idx)
            if vol_ratio < VOLUME_THRESHOLD:
                continue
            pre_vol = get_pre_signal_volume(symbol, hourly_by_symbol, idx)
            signal = Signal(
                symbol=symbol,
                signal_time=hour_ts,
                signal_price=row['close'],
                initial_move_pct=hour_return,
                volume_ratio=vol_ratio,
                pre_signal_volume=pre_vol
            )
            pending_signals[symbol] = signal

    # Close remaining
    for symbol, trade in active_trades.items():
        last_hour = hourly[hourly['symbol'] == symbol]['timestamp'].max()
        last_price = hourly[(hourly['symbol'] == symbol) &
                           (hourly['timestamp'] == last_hour)]['close'].iloc[0]
        gross_return = (last_price - trade.entry_price) / trade.entry_price
        net_return = gross_return - (FEE_RATE * 2)
        trade.status = 'closed'
        trade.exit_time = last_hour
        trade.exit_price = last_price
        trade.exit_reason = 'backtest_end'
        trade.pnl_pct = net_return * 100
        completed_trades.append(trade)

    return completed_trades


def main():
    print("="*70)
    print("BREAKOUT HUNTER BACKTEST: T+1 Entry vs T+2 Entry")
    print("="*70)

    # Load data
    hourly = load_hourly_data()

    # Run T+1 entry backtest
    print("\n\nRunning T+1 Entry backtest...")
    trades_t1 = run_backtest(hourly, entry_mode='t1')
    stats_t1 = analyze_trades(trades_t1, "T+1 ENTRY (Enter at T+1 close when T+1 > 0%)")

    # Run T+2 entry backtest (original strict version)
    print("\n\nRunning T+2 Entry backtest (strict: T+2 > 5%)...")
    trades_t2 = run_backtest(hourly, entry_mode='t2')
    stats_t2 = analyze_trades(trades_t2, "T+2 ENTRY STRICT (Enter at T+2 open when T+2 > 5%)")

    # Run T+2 entry with relaxed thresholds
    print("\n\nRunning T+2 Entry backtest (relaxed: T+2 > 0%)...")
    trades_t2_relaxed = run_backtest_t2_relaxed(hourly, t2_threshold=0.0)
    stats_t2_relaxed = analyze_trades(trades_t2_relaxed, "T+2 ENTRY RELAXED (Enter at T+2 open when T+2 > 0%)")

    # Run T+1 entry with max filter
    print("\n\nRunning T+1 Entry with max T+1 gain filter (25%)...")
    trades_t1_filtered = run_backtest(hourly, entry_mode='t1', max_t1_filter=0.25)
    stats_t1_filtered = analyze_trades(trades_t1_filtered,
                                       "T+1 ENTRY with MAX FILTER (Skip if T+1 > 25%)")

    # Run T+1 entry with max filter at 15%
    print("\n\nRunning T+1 Entry with max T+1 gain filter (15%)...")
    trades_t1_filtered_15 = run_backtest(hourly, entry_mode='t1', max_t1_filter=0.15)
    stats_t1_filtered_15 = analyze_trades(trades_t1_filtered_15,
                                          "T+1 ENTRY with MAX FILTER (Skip if T+1 > 15%)")

    # Run T+1 entry with max filter at 20%
    print("\n\nRunning T+1 Entry with max T+1 gain filter (20%)...")
    trades_t1_filtered_20 = run_backtest(hourly, entry_mode='t1', max_t1_filter=0.20)
    stats_t1_filtered_20 = analyze_trades(trades_t1_filtered_20,
                                          "T+1 ENTRY with MAX FILTER (Skip if T+1 > 20%)")

    # Run T+1 entry with max filter at 10%
    print("\n\nRunning T+1 Entry with max T+1 gain filter (10%)...")
    trades_t1_filtered_10 = run_backtest(hourly, entry_mode='t1', max_t1_filter=0.10)
    stats_t1_filtered_10 = analyze_trades(trades_t1_filtered_10,
                                          "T+1 ENTRY with MAX FILTER (Skip if T+1 > 10%)")

    # Summary comparison
    print("\n\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Strategy':<45} {'Trades':>8} {'Win%':>8} {'Avg P&L':>10} {'Total P&L':>12}")
    print("-"*70)

    results = [
        ("T+1 Entry (no filter)", stats_t1),
        ("T+2 Entry STRICT (T+2 > 5%)", stats_t2),
        ("T+2 Entry RELAXED (T+2 > 0%)", stats_t2_relaxed),
        ("T+1 Entry (max 25% filter)", stats_t1_filtered),
        ("T+1 Entry (max 20% filter)", stats_t1_filtered_20),
        ("T+1 Entry (max 15% filter)", stats_t1_filtered_15),
        ("T+1 Entry (max 10% filter)", stats_t1_filtered_10),
    ]

    for name, stats in results:
        if stats:
            print(f"{name:<45} {stats['total_trades']:>8} {stats['win_rate']:>7.1f}% "
                  f"{stats['avg_pnl']:>+9.2f}% {stats['total_pnl']:>+11.2f}%")

    # Analyze T+1 gain distribution for filtered-out trades
    print("\n\n" + "="*70)
    print("T+1 GAIN DISTRIBUTION ANALYSIS")
    print("="*70)

    t1_gains = [t.signal.t1_return for t in trades_t1 if t.signal.t1_return is not None]

    print(f"T+1 gains for all T+1 entry trades:")
    print(f"  Min: {min(t1_gains):.1f}%")
    print(f"  Max: {max(t1_gains):.1f}%")
    print(f"  Mean: {np.mean(t1_gains):.1f}%")
    print(f"  Median: {np.median(t1_gains):.1f}%")

    # Breakdown by T+1 gain buckets
    print(f"\nPerformance by T+1 gain bucket:")
    buckets = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 50), (50, 100)]

    for low, high in buckets:
        bucket_trades = [t for t in trades_t1
                        if t.signal.t1_return is not None
                        and low <= t.signal.t1_return < high]
        if bucket_trades:
            bucket_pnls = [t.pnl_pct for t in bucket_trades]
            wins = sum(1 for p in bucket_pnls if p > 0)
            print(f"  T+1 {low:>2}-{high:>2}%: {len(bucket_trades):>3} trades, "
                  f"win rate {wins/len(bucket_trades)*100:>5.1f}%, "
                  f"avg P&L {np.mean(bucket_pnls):>+6.2f}%")

    # Sample trades
    print("\n\n" + "="*70)
    print("SAMPLE T+1 ENTRY TRADES")
    print("="*70)

    sorted_trades = sorted(trades_t1, key=lambda t: t.pnl_pct, reverse=True)

    print("\nTop 5 Winning Trades:")
    for t in sorted_trades[:5]:
        print(f"  {t.symbol}: T+0={t.signal.initial_move_pct:+.1f}%, T+1={t.signal.t1_return:+.1f}% "
              f"-> P&L={t.pnl_pct:+.1f}% ({t.exit_reason})")

    print("\nTop 5 Losing Trades:")
    for t in sorted_trades[-5:]:
        print(f"  {t.symbol}: T+0={t.signal.initial_move_pct:+.1f}%, T+1={t.signal.t1_return:+.1f}% "
              f"-> P&L={t.pnl_pct:+.1f}% ({t.exit_reason})")


if __name__ == '__main__':
    main()
