#!/usr/bin/env python3
"""
Backtest Rule-Based Spike Trading Strategy

Tests different parameter configurations on historical data to find optimal thresholds.
Uses the same logic as RB_paper_trader.py but replays historical candles/trades.

Usage:
    python scripts/backtest_rb_strategy.py --db "/Users/brettzeigler/Pythia/market_data copy_86.db" --days 7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import argparse
from loguru import logger


@dataclass
class BacktestConfig:
    """Backtest configuration parameters"""
    # Portfolio
    initial_balance: float = 10000.0
    position_size_pct: float = 0.25
    max_positions: int = 4
    stop_loss_pct: float = 0.0075

    # Detection thresholds
    volume_spike_threshold: float = 1.5
    price_spike_threshold: float = 0.06
    rsi_overbought: float = 80.0
    rsi_exemption_threshold: float = 15.0

    # OFI thresholds
    ofi_threshold: float = 0.3
    ofi_min_trades: int = 10

    # Exit thresholds
    checkpoint_time_min: int = 10
    grace_period_duration_min: int = 10
    grace_period_min_gain: float = 0.03
    fast_steep_gain_threshold: float = 0.15
    slow_large_gain_min: float = 0.08
    fast_steep_time_limit_min: int = 30
    fast_steep_drawdown_pct: float = 0.02
    slow_large_time_limit_min: int = 1440
    slow_large_drawdown_pct: float = 0.05


@dataclass
class Trade:
    """Single trade record"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    symbol: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: float = 0.0
    spike_type: Optional[str] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    peak_price: float = 0.0
    entry_rsi: float = 0.0
    entry_volume_ratio: float = 0.0
    entry_price_gain: float = 0.0
    entry_ofi: float = 0.0


@dataclass
class Position:
    """Active position"""
    symbol: str
    entry_time: datetime
    entry_price: float
    shares: float
    spike_type: Optional[str] = None
    peak_price: float = 0.0
    peak_time: Optional[datetime] = None
    entry_rsi: float = 0.0
    entry_volume_ratio: float = 0.0
    entry_price_gain: float = 0.0
    entry_ofi: float = 0.0


class BacktestEngine:
    """Backtest engine for rule-based strategy"""

    def __init__(self, db_path: str, config: BacktestConfig):
        self.db_path = db_path
        self.config = config
        self.conn = sqlite3.connect(db_path)

        # Portfolio state
        self.cash = config.initial_balance
        self.initial_balance = config.initial_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI from price series"""
        if len(prices) < period + 1:
            return 50.0

        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0.0)
        losses = -deltas.where(deltas < 0, 0.0)

        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

    def calculate_ofi(self, symbol: str, timestamp: datetime) -> tuple:
        """Calculate OFI from trades in 1-minute window before timestamp"""
        # Query trades in 60-second window
        query = """
            SELECT side, size
            FROM trades
            WHERE symbol = ?
              AND timestamp >= datetime(?, '-60 seconds')
              AND timestamp <= ?
            ORDER BY timestamp
        """

        df = pd.read_sql_query(
            query,
            self.conn,
            params=(symbol, timestamp.isoformat(), timestamp.isoformat())
        )

        if len(df) < self.config.ofi_min_trades:
            return 0.0, len(df)

        buy_volume = df[df['side'] == 'buy']['size'].sum()
        sell_volume = df[df['side'] == 'sell']['size'].sum()
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0.0, len(df)

        ofi = (buy_volume - sell_volume) / total_volume
        return ofi, len(df)

    def detect_spike(self, symbol: str, candles: pd.DataFrame, current_time: datetime) -> Optional[dict]:
        """Detect spike using same logic as RB_paper_trader"""
        if len(candles) < 15:
            return None

        # Get last 10 candles for volume baseline
        last_10 = candles.tail(10)
        last_3 = candles.tail(3)
        current = candles.iloc[-1]

        # Calculate volume spike (3-min cumulative vs baseline)
        recent_3min_volume = last_3['volume'].sum()

        if len(last_10) >= 6:
            baseline_periods = [
                last_10.iloc[0:3]['volume'].sum(),
                last_10.iloc[3:6]['volume'].sum(),
                last_10.iloc[6:9]['volume'].sum() if len(last_10) >= 9 else 0,
            ]
            avg_3min_volume = np.mean([p for p in baseline_periods if p > 0])
        else:
            avg_3min_volume = last_10['volume'].mean() * 3

        volume_ratio = recent_3min_volume / avg_3min_volume if avg_3min_volume > 0 else 0

        # Calculate price spike (3-min and 10-min windows)
        first_close_3min = last_3.iloc[0]['close']
        max_high_3min = last_3['high'].max()
        price_gain_3min = ((max_high_3min / first_close_3min) - 1) * 100 if first_close_3min > 0 else 0

        price_gain_10min = 0.0
        if len(candles) >= 11:
            first_close_10min = candles.iloc[-11]['close']
            current_close = current['close']
            price_gain_10min = ((current_close / first_close_10min) - 1) * 100 if first_close_10min > 0 else 0

        price_gain_pct = max(price_gain_3min, price_gain_10min)

        # Calculate RSI
        rsi = self.calculate_rsi(candles['close'].tail(20))

        # Check spike criteria
        volume_spike = volume_ratio >= self.config.volume_spike_threshold
        price_spike = (price_gain_pct / 100) >= self.config.price_spike_threshold

        large_price_move = price_gain_pct >= self.config.rsi_exemption_threshold
        rsi_check_passed = (rsi < self.config.rsi_overbought) or large_price_move

        if not (volume_spike and price_spike and rsi_check_passed):
            return None

        # Calculate OFI (final gate)
        ofi, trade_count = self.calculate_ofi(symbol, current_time)
        ofi_passed = ofi >= self.config.ofi_threshold

        if not ofi_passed:
            return None

        return {
            'volume_ratio': volume_ratio,
            'price_gain_pct': price_gain_pct,
            'rsi': rsi,
            'ofi': ofi,
            'trade_count': trade_count,
        }

    def check_exit(self, position: Position, current_price: float, current_time: datetime) -> Optional[str]:
        """Check if position should exit"""
        hold_time_min = (current_time - position.entry_time).total_seconds() / 60
        current_gain_pct = ((current_price / position.entry_price) - 1)

        # Update peak
        if current_price > position.peak_price:
            position.peak_price = current_price
            position.peak_time = current_time

        # Stop loss (always active)
        if current_price <= position.entry_price * (1 - self.config.stop_loss_pct):
            return 'stop_loss'

        # Before 10-min checkpoint, only stop loss active
        if hold_time_min < self.config.checkpoint_time_min:
            return None

        # 10-minute checkpoint (classify spike type)
        if position.spike_type is None:
            if current_gain_pct >= self.config.fast_steep_gain_threshold:
                position.spike_type = 'fast_steep'
            elif current_gain_pct >= self.config.slow_large_gain_min:
                position.spike_type = 'slow_large'
            elif current_gain_pct >= self.config.grace_period_min_gain:
                position.spike_type = 'grace_period'
            else:
                return '10min_checkpoint_underperform'

        # Grace period re-check at 20 minutes
        if position.spike_type == 'grace_period' and hold_time_min >= 20:
            if current_gain_pct >= self.config.slow_large_gain_min:
                position.spike_type = 'slow_large'
            else:
                return '20min_grace_period_underperform'

        # Fast & Steep exits
        if position.spike_type == 'fast_steep':
            # Time exit
            if hold_time_min >= self.config.fast_steep_time_limit_min:
                return 'fast_steep_time_limit'

            # Drawdown exit
            if position.peak_price > 0:
                drawdown = (position.peak_price - current_price) / position.peak_price
                if drawdown >= self.config.fast_steep_drawdown_pct:
                    return 'fast_steep_drawdown'

        # Slow & Large exits
        if position.spike_type == 'slow_large':
            # Time exit
            if hold_time_min >= self.config.slow_large_time_limit_min:
                return 'slow_large_time_limit'

            # Drawdown exit
            if position.peak_price > 0:
                drawdown = (position.peak_price - current_price) / position.peak_price
                if drawdown >= self.config.slow_large_drawdown_pct:
                    return 'slow_large_drawdown'

        return None

    def run_backtest(self, start_date: str, end_date: str, symbols: Optional[List[str]] = None):
        """Run backtest on historical data"""
        logger.info(f"Running backtest: {start_date} to {end_date}")
        logger.info(f"Config: volume={self.config.volume_spike_threshold}x, price={self.config.price_spike_threshold*100}%, ofi={self.config.ofi_threshold}")

        # Get all symbols if not specified
        if symbols is None:
            symbols_df = pd.read_sql_query(
                f"SELECT DISTINCT symbol FROM candles WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'",
                self.conn
            )
            symbols = symbols_df['symbol'].tolist()

        logger.info(f"Testing {len(symbols)} symbols")

        # Get all candles in date range
        query = f"""
            SELECT symbol, timestamp, open, high, low, close, volume, buy_volume, sell_volume, num_trades
            FROM candles
            WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'
            ORDER BY timestamp
        """

        all_candles = pd.read_sql_query(query, self.conn)
        all_candles['timestamp'] = pd.to_datetime(all_candles['timestamp'])

        # Group by symbol
        candles_by_symbol = {symbol: df for symbol, df in all_candles.groupby('symbol')}

        # Process minute by minute
        unique_times = sorted(all_candles['timestamp'].unique())

        for i, current_time in enumerate(unique_times):
            if i % 1440 == 0:  # Log daily
                logger.info(f"Processing {current_time} ({i}/{len(unique_times)} candles)")

            # Check exits first
            positions_to_close = []
            for symbol, position in self.positions.items():
                # Get current price from candles
                symbol_candles = candles_by_symbol.get(symbol)
                if symbol_candles is None:
                    continue

                current_candle = symbol_candles[symbol_candles['timestamp'] == current_time]
                if current_candle.empty:
                    continue

                current_price = current_candle.iloc[0]['close']
                exit_reason = self.check_exit(position, current_price, current_time)

                if exit_reason:
                    positions_to_close.append((symbol, current_price, exit_reason))

            # Close positions
            for symbol, exit_price, exit_reason in positions_to_close:
                position = self.positions.pop(symbol)
                value = position.shares * exit_price
                self.cash += value

                pnl = value - (position.shares * position.entry_price)
                pnl_pct = pnl / (position.shares * position.entry_price) * 100

                trade = Trade(
                    entry_time=position.entry_time,
                    exit_time=current_time,
                    symbol=symbol,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    shares=position.shares,
                    spike_type=position.spike_type,
                    exit_reason=exit_reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    peak_price=position.peak_price,
                    entry_rsi=position.entry_rsi,
                    entry_volume_ratio=position.entry_volume_ratio,
                    entry_price_gain=position.entry_price_gain,
                    entry_ofi=position.entry_ofi,
                )
                self.trades.append(trade)

            # Check for new entries (if room available)
            if len(self.positions) < self.config.max_positions:
                for symbol in symbols:
                    if symbol in self.positions:
                        continue

                    # Get candle history for this symbol up to current_time
                    symbol_candles = candles_by_symbol.get(symbol)
                    if symbol_candles is None:
                        continue

                    history = symbol_candles[symbol_candles['timestamp'] <= current_time]
                    if len(history) < 15:
                        continue

                    # Detect spike
                    spike_data = self.detect_spike(symbol, history, current_time)
                    if spike_data is None:
                        continue

                    # Enter position
                    current_candle = history.iloc[-1]
                    entry_price = current_candle['close']
                    position_value = self.cash * self.config.position_size_pct
                    shares = position_value / entry_price

                    if position_value > self.cash:
                        continue

                    self.cash -= position_value

                    position = Position(
                        symbol=symbol,
                        entry_time=current_time,
                        entry_price=entry_price,
                        shares=shares,
                        peak_price=entry_price,
                        peak_time=current_time,
                        entry_rsi=spike_data['rsi'],
                        entry_volume_ratio=spike_data['volume_ratio'],
                        entry_price_gain=spike_data['price_gain_pct'],
                        entry_ofi=spike_data['ofi'],
                    )
                    self.positions[symbol] = position

                    # Only 1 entry per time step
                    break

        # Close any remaining positions at end
        if self.positions:
            logger.info(f"Closing {len(self.positions)} remaining positions at end")
            for symbol, position in self.positions.items():
                symbol_candles = candles_by_symbol.get(symbol)
                if symbol_candles is None:
                    continue
                final_price = symbol_candles.iloc[-1]['close']
                value = position.shares * final_price
                self.cash += value

                pnl = value - (position.shares * position.entry_price)
                pnl_pct = pnl / (position.shares * position.entry_price) * 100

                trade = Trade(
                    entry_time=position.entry_time,
                    exit_time=symbol_candles.iloc[-1]['timestamp'],
                    symbol=symbol,
                    entry_price=position.entry_price,
                    exit_price=final_price,
                    shares=position.shares,
                    spike_type=position.spike_type,
                    exit_reason='backtest_end',
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    peak_price=position.peak_price,
                    entry_rsi=position.entry_rsi,
                    entry_volume_ratio=position.entry_volume_ratio,
                    entry_price_gain=position.entry_price_gain,
                    entry_ofi=position.entry_ofi,
                )
                self.trades.append(trade)

        self.positions.clear()

    def print_results(self):
        """Print backtest results"""
        final_value = self.cash
        total_return = ((final_value / self.initial_balance) - 1) * 100

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance:   ${final_value:,.2f}")
        print(f"Total Return:    {total_return:+.2f}%")
        print()
        print(f"Total Trades:    {len(self.trades)}")
        print(f"Wins:            {len(wins)} ({len(wins)/len(self.trades)*100:.1f}%)" if self.trades else "Wins: 0")
        print(f"Losses:          {len(losses)} ({len(losses)/len(self.trades)*100:.1f}%)" if self.trades else "Losses: 0")

        if self.trades:
            avg_return = np.mean([t.pnl_pct for t in self.trades])
            avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
            avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0

            print()
            print(f"Avg Return:      {avg_return:+.2f}%")
            print(f"Avg Win:         {avg_win:+.2f}%")
            print(f"Avg Loss:        {avg_loss:+.2f}%")

            # Spike type breakdown
            fast_steep = [t for t in self.trades if t.spike_type == 'fast_steep']
            slow_large = [t for t in self.trades if t.spike_type == 'slow_large']

            if fast_steep:
                print()
                print(f"Fast & Steep:    {len(fast_steep)} trades, avg {np.mean([t.pnl_pct for t in fast_steep]):+.2f}%")
            if slow_large:
                print(f"Slow & Large:    {len(slow_large)} trades, avg {np.mean([t.pnl_pct for t in slow_large]):+.2f}%")

        print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest RB spike trading strategy')
    parser.add_argument('--db', required=True, help='Path to database')
    parser.add_argument('--days', type=int, default=7, help='Number of days to backtest')
    parser.add_argument('--volume', type=float, default=1.5, help='Volume spike threshold')
    parser.add_argument('--price', type=float, default=6.0, help='Price spike threshold (%)')
    parser.add_argument('--ofi', type=float, default=0.3, help='OFI threshold')

    args = parser.parse_args()

    # Get date range from database
    conn = sqlite3.connect(args.db)
    max_date = pd.read_sql_query('SELECT MAX(timestamp) as max_date FROM candles', conn).iloc[0]['max_date']
    conn.close()

    end_date = pd.to_datetime(max_date)
    start_date = end_date - timedelta(days=args.days)

    # Create config
    config = BacktestConfig(
        volume_spike_threshold=args.volume,
        price_spike_threshold=args.price / 100,
        ofi_threshold=args.ofi,
    )

    # Run backtest
    engine = BacktestEngine(args.db, config)
    engine.run_backtest(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    engine.print_results()

    # Save trades to CSV
    if engine.trades:
        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'symbol': t.symbol,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl_pct': t.pnl_pct,
                'spike_type': t.spike_type,
                'exit_reason': t.exit_reason,
                'entry_volume_ratio': t.entry_volume_ratio,
                'entry_price_gain': t.entry_price_gain,
                'entry_rsi': t.entry_rsi,
                'entry_ofi': t.entry_ofi,
            }
            for t in engine.trades
        ])
        output_file = f"backtest_results_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        trades_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(engine.trades)} trades to {output_file}")
