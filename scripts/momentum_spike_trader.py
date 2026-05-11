#!/usr/bin/env python3
"""
Momentum-Based Spike Trader

Based on backtest analysis of March 19-23, 2026 data showing:
- Volume-based entry (3x+) generated 0 trades (too restrictive)
- Pure momentum (6%+ in 1h) with trailing stops: 56% win rate, +$817/4 days

Key insights from analysis:
1. Big movers (SYND +120%, BOBA +66%) had NORMAL volume during spikes
2. Volume explosion often comes AFTER the move, not before
3. Short-term momentum (1h) is a better predictor than 24h price change
4. Trailing stops capture more profit than fixed take-profit

Strategy:
- Entry: 6%+ price gain in the last 1 hour (detected via collector)
- Stop Loss: 2% (tight, to manage the high-volatility trades)
- Exit: Ratcheting trailing stops (1.2% at 2-6%, 1.8% at 6-10%, etc.)
- Max Hold: 4 hours (most continuation happens quickly)

Usage:
    python scripts/momentum_spike_trader.py
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import sqlite3
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

try:
    import paper_trades as pt
    HAS_PAPER_TRADES = True
except ImportError:
    HAS_PAPER_TRADES = False


@dataclass
class MomentumPosition:
    """Represents an open position."""
    symbol: str
    entry_time: datetime
    entry_price: float
    position_size_usd: float
    signal_momentum: float  # The 1h momentum that triggered entry
    highest_price: float = 0.0
    trailing_activated: bool = False

    def __post_init__(self):
        self.highest_price = self.entry_price


@dataclass
class MomentumTrade:
    """Completed trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size_usd: float
    pnl_pct: float
    pnl_usd: float
    exit_reason: str
    signal_momentum: float


@dataclass
class MomentumTraderConfig:
    """Trader configuration based on backtest optimization."""
    # Capital
    initial_capital: float = 10000.0
    position_size_pct: float = 10.0
    max_positions: int = 3

    # Entry: Momentum-based (replaces volume filter)
    min_momentum_1h: float = 6.0  # 6%+ in 1 hour triggers entry

    # FOMO filter: Don't chase extremely extended moves
    max_momentum_1h: float = 25.0  # Skip if already up 25%+ (likely to dump)

    # Exits
    stop_loss_pct: float = 2.0  # 2% stop (tight for volatile plays)
    max_hold_hours: int = 4    # 4 hours max (most continuation is fast)

    # Ratcheting trailing stop (proven effective in backtest)
    trail_activation_pct: float = 2.0  # Trail activates at +2%
    ratchet_levels: List[tuple] = field(default_factory=lambda: [
        (2.0, 1.2),   # 2-6%:   1.2% trail
        (6.0, 1.8),   # 6-10%:  1.8% trail (give room to run)
        (10.0, 1.2),  # 10-15%: 1.2% trail (tightening)
        (15.0, 4.0),  # 15%+:   4.0% trail (lock in big gains)
    ])


class MomentumSpikeTrader:
    """
    Momentum-based spike trading system.

    Monitors 1-hour momentum and enters on strong moves.
    Uses trailing stops to capture continuation while limiting drawdown.
    """

    def __init__(self, config: MomentumTraderConfig = None, db_path: str = "data/feature_buffer.db"):
        self.config = config or MomentumTraderConfig()
        self.db_path = db_path

        # Use SQLite for price data (supports concurrent access via WAL)
        self.sqlite_conn = sqlite3.connect(db_path, timeout=30)
        self.sqlite_conn.execute("PRAGMA journal_mode=WAL")
        self.sqlite_conn.row_factory = sqlite3.Row

        # State
        self.capital = self.config.initial_capital
        self.positions: Dict[str, MomentumPosition] = {}
        self.trades: List[MomentumTrade] = []
        self.running = False

        # Stats
        self.signals_seen = 0
        self.signals_filtered = 0
        self.signals_blocked_fomo = 0

        # Cooldown: Don't re-enter same symbol within 2 hours
        self.cooldowns: Dict[str, datetime] = {}
        self.cooldown_hours = 2

        logger.info(f"Momentum trader initialized with ${self.config.initial_capital:,.0f}")
        logger.info(f"Entry: {self.config.min_momentum_1h}%+ 1h momentum")
        logger.info(f"Stop: {self.config.stop_loss_pct}%, Max hold: {self.config.max_hold_hours}h")

        # Persistence
        self.state_file = Path("logs/momentum_trader_state.json")
        self._load_state()

    def _load_state(self):
        """Load persisted state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                self.capital = state.get('capital', self.config.initial_capital)
                self.signals_seen = state.get('signals_seen', 0)
                self.signals_filtered = state.get('signals_filtered', 0)
                self.signals_blocked_fomo = state.get('signals_blocked_fomo', 0)

                # Restore trades
                for t in state.get('trades', []):
                    self.trades.append(MomentumTrade(
                        symbol=t['symbol'],
                        entry_time=datetime.fromisoformat(t['entry_time']),
                        exit_time=datetime.fromisoformat(t['exit_time']),
                        entry_price=t['entry_price'],
                        exit_price=t['exit_price'],
                        position_size_usd=t['position_size_usd'],
                        pnl_pct=t['pnl_pct'],
                        pnl_usd=t['pnl_usd'],
                        exit_reason=t['exit_reason'],
                        signal_momentum=t.get('signal_momentum', 0),
                    ))

                # Restore open positions
                for sym, p in state.get('positions', {}).items():
                    self.positions[sym] = MomentumPosition(
                        symbol=p['symbol'],
                        entry_time=datetime.fromisoformat(p['entry_time']),
                        entry_price=p['entry_price'],
                        position_size_usd=p['position_size_usd'],
                        signal_momentum=p.get('signal_momentum', 0),
                        highest_price=p.get('highest_price', p['entry_price']),
                        trailing_activated=p.get('trailing_activated', False),
                    )

                logger.info(f"Restored state: ${self.capital:,.2f}, {len(self.trades)} trades, {len(self.positions)} positions")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Persist state to disk."""
        state = {
            'capital': self.capital,
            'signals_seen': self.signals_seen,
            'signals_filtered': self.signals_filtered,
            'signals_blocked_fomo': self.signals_blocked_fomo,
            'trades': [{
                'symbol': t.symbol,
                'entry_time': t.entry_time.isoformat(),
                'exit_time': t.exit_time.isoformat(),
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'position_size_usd': t.position_size_usd,
                'pnl_pct': t.pnl_pct,
                'pnl_usd': t.pnl_usd,
                'exit_reason': t.exit_reason,
                'signal_momentum': t.signal_momentum,
            } for t in self.trades],
            'positions': {sym: {
                'symbol': p.symbol,
                'entry_time': p.entry_time.isoformat(),
                'entry_price': p.entry_price,
                'position_size_usd': p.position_size_usd,
                'signal_momentum': p.signal_momentum,
                'highest_price': p.highest_price,
                'trailing_activated': p.trailing_activated,
            } for sym, p in self.positions.items()},
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from SQLite database."""
        try:
            cursor = self.sqlite_conn.execute("""
                SELECT close FROM ohlcv
                WHERE symbol = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (symbol,))
            result = cursor.fetchone()
            return result[0] if result else None
        except:
            return None

    def get_1h_momentum(self, symbol: str) -> Optional[float]:
        """
        Calculate 1-hour price momentum.

        Returns the percentage change over the last hour.
        """
        try:
            # Get price from 1 hour ago and current
            cursor = self.sqlite_conn.execute("""
                SELECT
                    (SELECT close FROM ohlcv
                     WHERE symbol = ?
                     ORDER BY timestamp DESC LIMIT 1) as current_price,
                    (SELECT close FROM ohlcv
                     WHERE symbol = ?
                       AND timestamp <= datetime('now', '-1 hour')
                     ORDER BY timestamp DESC LIMIT 1) as price_1h_ago
            """, (symbol, symbol))
            result = cursor.fetchone()

            if result and result[0] and result[1] and result[1] > 0:
                momentum = ((result[0] / result[1]) - 1) * 100
                return momentum
            return None
        except Exception as e:
            logger.debug(f"Error getting momentum for {symbol}: {e}")
            return None

    def scan_for_momentum_signals(self) -> List[Dict]:
        """
        Scan all symbols for momentum breakouts.

        Returns list of symbols with 6%+ 1h momentum.
        """
        signals = []

        try:
            # Get all active symbols (traded in last 2 hours)
            cursor = self.sqlite_conn.execute("""
                SELECT DISTINCT symbol FROM ohlcv
                WHERE timestamp >= datetime('now', '-2 hours')
            """)
            symbols = [row[0] for row in cursor.fetchall()]

            for symbol in symbols:
                momentum = self.get_1h_momentum(symbol)
                if momentum is None:
                    continue

                # Check if momentum exceeds threshold
                if momentum >= self.config.min_momentum_1h:
                    current_price = self.get_current_price(symbol)
                    if current_price:
                        signals.append({
                            'symbol': symbol,
                            'momentum_1h': momentum,
                            'price': current_price,
                            'timestamp': datetime.now(timezone.utc),
                        })

            return signals

        except Exception as e:
            logger.error(f"Error scanning for signals: {e}")
            return []

    def should_block_entry(self, momentum_1h: float) -> tuple:
        """
        Check if entry should be blocked (FOMO filter).

        Blocks extremely extended moves that are likely to dump.
        """
        if momentum_1h > self.config.max_momentum_1h:
            return True, f"Extreme momentum: {momentum_1h:.1f}% > {self.config.max_momentum_1h}%"

        return False, ""

    def is_on_cooldown(self, symbol: str) -> bool:
        """Check if symbol is on cooldown from recent exit."""
        if symbol in self.cooldowns:
            cooldown_until = self.cooldowns[symbol]
            if datetime.now(timezone.utc) < cooldown_until:
                return True
            else:
                del self.cooldowns[symbol]
        return False

    def get_trail_distance(self, gain_pct: float) -> float:
        """Get trailing stop distance based on current gain level."""
        for level_pct, trail_dist in reversed(self.config.ratchet_levels):
            if gain_pct >= level_pct:
                return trail_dist
        return 1.2  # Default

    def check_exits(self) -> List[MomentumTrade]:
        """Check all positions for exit conditions."""
        closed = []
        now = datetime.now(timezone.utc)

        for symbol, pos in list(self.positions.items()):
            current_price = self.get_current_price(symbol)
            if current_price is None:
                continue

            pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100

            # Update highest price for trailing
            if current_price > pos.highest_price:
                pos.highest_price = current_price

            exit_reason = None
            exit_price = current_price

            # Check stop loss first
            if pnl_pct <= -self.config.stop_loss_pct:
                exit_reason = "sl"

            # Check ratcheting trailing stop
            elif pnl_pct >= self.config.trail_activation_pct or pos.trailing_activated:
                pos.trailing_activated = True

                # Get trail distance based on highest gain reached
                high_pnl_pct = ((pos.highest_price - pos.entry_price) / pos.entry_price) * 100
                trail_pct = self.get_trail_distance(high_pnl_pct)

                drawdown_from_high = ((pos.highest_price - current_price) / pos.highest_price) * 100
                if drawdown_from_high >= trail_pct:
                    exit_reason = "trail"

            # Check max hold time
            hold_hours = (now - pos.entry_time).total_seconds() / 3600
            if hold_hours >= self.config.max_hold_hours:
                exit_reason = "time"

            if exit_reason:
                # Close position
                pnl_usd = pos.position_size_usd * (pnl_pct / 100)
                self.capital += pos.position_size_usd + pnl_usd

                trade = MomentumTrade(
                    symbol=symbol,
                    entry_time=pos.entry_time,
                    exit_time=now,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    position_size_usd=pos.position_size_usd,
                    pnl_pct=pnl_pct,
                    pnl_usd=pnl_usd,
                    exit_reason=exit_reason,
                    signal_momentum=pos.signal_momentum,
                )
                self.trades.append(trade)
                closed.append(trade)
                del self.positions[symbol]

                # Set cooldown
                self.cooldowns[symbol] = now + timedelta(hours=self.cooldown_hours)

                logger.warning(
                    f"EXIT {symbol}: {exit_reason.upper()} | "
                    f"PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f}) | "
                    f"Entry momentum: {pos.signal_momentum:.1f}%"
                )

                # Record to SQLite for visualizer
                if HAS_PAPER_TRADES:
                    pt.record_exit(
                        strategy="momentum_spike",
                        symbol=symbol,
                        exit_price=exit_price,
                        exit_time=now,
                        exit_reason=exit_reason,
                        realized_pnl=pnl_usd,
                    )
                self._save_state()

        return closed

    def try_entry(self, signal: Dict) -> bool:
        """Try to enter a new position based on momentum signal."""
        symbol = signal['symbol']
        momentum = signal['momentum_1h']

        # Skip if already in position
        if symbol in self.positions:
            return False

        # Skip if at max positions
        if len(self.positions) >= self.config.max_positions:
            return False

        # Skip if on cooldown
        if self.is_on_cooldown(symbol):
            logger.debug(f"Skipping {symbol}: on cooldown")
            return False

        # Check FOMO filter
        blocked, reason = self.should_block_entry(momentum)
        if blocked:
            self.signals_blocked_fomo += 1
            logger.info(f"BLOCKED {symbol}: {reason}")
            return False

        # Get entry price
        entry_price = signal.get('price') or self.get_current_price(symbol)
        if entry_price is None:
            return False

        # Calculate position size
        position_size = min(
            self.capital * (self.config.position_size_pct / 100),
            self.capital * 0.95
        )

        if position_size < 10:
            return False

        # Enter position
        self.capital -= position_size
        self.positions[symbol] = MomentumPosition(
            symbol=symbol,
            entry_time=datetime.now(timezone.utc),
            entry_price=entry_price,
            position_size_usd=position_size,
            signal_momentum=momentum,
        )

        logger.warning(
            f"ENTRY {symbol} @ ${entry_price:.6f} | "
            f"Size: ${position_size:.2f} | Momentum: {momentum:.1f}%"
        )

        # Record to SQLite for visualizer
        if HAS_PAPER_TRADES:
            pt.record_entry(
                strategy="momentum_spike",
                symbol=symbol,
                entry_price=entry_price,
                entry_time=datetime.now(timezone.utc),
                position_size=position_size,
                quantity=position_size / entry_price,
            )
        self._save_state()

        return True

    def print_status(self):
        """Print current status."""
        total_position_value = sum(p.position_size_usd for p in self.positions.values())
        total_equity = self.capital + total_position_value

        total_pnl = sum(t.pnl_usd for t in self.trades)
        win_count = sum(1 for t in self.trades if t.pnl_usd > 0)

        print("\n" + "=" * 60)
        print("MOMENTUM SPIKE TRADER STATUS")
        print("=" * 60)
        print(f"Equity:        ${total_equity:,.2f}")
        print(f"Cash:          ${self.capital:,.2f}")
        print(f"Positions:     {len(self.positions)}/{self.config.max_positions}")
        print(f"Total Trades:  {len(self.trades)}")
        print(f"Win Rate:      {win_count}/{len(self.trades)} ({win_count/max(len(self.trades),1)*100:.1f}%)")
        print(f"Total PnL:     ${total_pnl:+,.2f}")
        print(f"Signals Seen:  {self.signals_seen}")
        print(f"FOMO Blocked:  {self.signals_blocked_fomo}")

        if self.positions:
            print("\nOpen Positions:")
            for sym, pos in self.positions.items():
                current = self.get_current_price(sym) or pos.entry_price
                pnl = ((current - pos.entry_price) / pos.entry_price) * 100
                print(f"  {sym}: ${pos.position_size_usd:.0f} @ ${pos.entry_price:.6f} -> ${current:.6f} ({pnl:+.2f}%) | Entry mom: {pos.signal_momentum:.1f}%")

        if self.trades:
            print("\nRecent Trades:")
            for trade in self.trades[-5:]:
                print(f"  {trade.symbol}: {trade.exit_reason.upper()} {trade.pnl_pct:+.2f}% (${trade.pnl_usd:+.2f})")

        print("=" * 60)

    async def run(self, interval_seconds: int = 60):
        """Run the trader loop."""
        self.running = True
        logger.info(f"Starting momentum trader (scanning every {interval_seconds}s)")

        iteration = 0
        while self.running:
            try:
                iteration += 1

                # Check exits first
                closed = self.check_exits()

                # Scan for momentum breakouts
                signals = self.scan_for_momentum_signals()
                self.signals_seen += len(signals)

                # Log high-momentum signals
                for sig in signals:
                    logger.info(f"SIGNAL {sig['symbol']}: {sig['momentum_1h']:.1f}% 1h momentum")

                # Try entries on all signals
                for signal in signals:
                    self.try_entry(signal)

                # Print status every 5 iterations
                if iteration % 5 == 0:
                    self.print_status()
                    self._save_state()

                await asyncio.sleep(interval_seconds)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Trader error: {e}")
                await asyncio.sleep(interval_seconds)

        self.print_status()
        logger.info("Momentum trader stopped")


async def main():
    """Run the momentum spike trader."""
    import argparse

    parser = argparse.ArgumentParser(description="Momentum-Based Spike Trader")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval (seconds)")
    parser.add_argument("--db", type=str, default="data/feature_buffer.db", help="SQLite database path")
    parser.add_argument("--momentum", type=float, default=6.0, help="Minimum 1h momentum threshold")
    parser.add_argument("--stop-loss", type=float, default=2.0, help="Stop loss percentage")
    parser.add_argument("--max-hold", type=int, default=4, help="Max hold time in hours")
    args = parser.parse_args()

    config = MomentumTraderConfig(
        initial_capital=args.capital,
        min_momentum_1h=args.momentum,
        stop_loss_pct=args.stop_loss,
        max_hold_hours=args.max_hold,
    )
    trader = MomentumSpikeTrader(config=config, db_path=args.db)

    try:
        await trader.run(interval_seconds=args.interval)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    asyncio.run(main())
