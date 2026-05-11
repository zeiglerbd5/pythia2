#!/usr/bin/env python3
"""
Spike Prediction Paper Trader

Runs the integrated ML spike prediction system in paper trading mode:
- Ensemble model (LightGBM + XGBoost + RF)
- Volatility filter (P50+)
- Bad entry filter (FOMO trap detection)
- RL position sizing
- Optimized exits (stepped trailing, 6% TP, 3% SL)

Usage:
    python scripts/run_spike_paper_trader.py
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.data_ingestion.database import DuckDBManager
import paper_trades as pt


@dataclass
class PaperPosition:
    """Represents an open paper position."""
    symbol: str
    entry_time: datetime
    entry_price: float
    position_size_usd: float
    pred_probability: float
    highest_price: float = 0.0
    trailing_activated: bool = False

    def __post_init__(self):
        self.highest_price = self.entry_price


@dataclass
class PaperTrade:
    """Completed paper trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size_usd: float
    pnl_pct: float
    pnl_usd: float
    exit_reason: str
    pred_probability: float


@dataclass
class PaperTraderConfig:
    """Paper trader configuration."""
    # Capital
    initial_capital: float = 10000.0
    position_size_pct: float = 10.0
    max_positions: int = 3

    # Entry filters
    min_pred_probability: float = 0.3
    min_volatility_4h: float = 0.205  # P50 threshold

    # Bad entry filter (FOMO trap)
    max_momentum_4h: float = 5.0
    max_volatility_4h: float = 0.60

    # Exits (no fixed TP - using ratcheting trail instead)
    stop_loss_pct: float = 3.0
    max_hold_hours: int = 24

    # Ratcheting trailing stop (replaces fixed TP)
    # Format: (activation_pct, trail_distance_pct)
    trail_activation_pct: float = 2.0  # Trail activates at +2%
    ratchet_levels: List[tuple] = field(default_factory=lambda: [
        (2.0, 1.2),   # 2-6%:   1.2% trail (activation zone)
        (6.0, 1.8),   # 6-10%:  1.8% trail (former TP zone - liberal)
        (10.0, 1.2),  # 10-15%: 1.2% trail (tightening)
        (15.0, 4.0),  # 15%+:   4.0% trail (lock in big gains)
    ])


class SpikePaperTrader:
    """
    Paper trading system for spike prediction.

    Monitors signals and simulates trades without real execution.
    """

    def __init__(self, config: PaperTraderConfig = None, db_path: str = "data/feature_buffer.db"):
        self.config = config or PaperTraderConfig()
        self.db_path = db_path

        # Use SQLite for price data (supports concurrent access via WAL)
        import sqlite3
        self.sqlite_conn = sqlite3.connect(db_path, timeout=30)
        self.sqlite_conn.execute("PRAGMA journal_mode=WAL")
        self.sqlite_conn.row_factory = sqlite3.Row

        # State
        self.capital = self.config.initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.trades: List[PaperTrade] = []
        self.running = False

        # Stats
        self.signals_seen = 0
        self.signals_filtered = 0
        self.entries_blocked = 0

        logger.info(f"Paper trader initialized with ${self.config.initial_capital:,.0f}")

        # Persistence
        self.state_file = Path("logs/paper_trader_state.json")
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
                self.entries_blocked = state.get('entries_blocked', 0)

                # Restore trades
                for t in state.get('trades', []):
                    self.trades.append(PaperTrade(
                        symbol=t['symbol'],
                        entry_time=datetime.fromisoformat(t['entry_time']),
                        exit_time=datetime.fromisoformat(t['exit_time']),
                        entry_price=t['entry_price'],
                        exit_price=t['exit_price'],
                        position_size_usd=t['position_size_usd'],
                        pnl_pct=t['pnl_pct'],
                        pnl_usd=t['pnl_usd'],
                        exit_reason=t['exit_reason'],
                        pred_probability=t['pred_probability'],
                    ))

                # Restore open positions
                for sym, p in state.get('positions', {}).items():
                    self.positions[sym] = PaperPosition(
                        symbol=p['symbol'],
                        entry_time=datetime.fromisoformat(p['entry_time']),
                        entry_price=p['entry_price'],
                        position_size_usd=p['position_size_usd'],
                        pred_probability=p['pred_probability'],
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
            'entries_blocked': self.entries_blocked,
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
                'pred_probability': t.pred_probability,
            } for t in self.trades],
            'positions': {sym: {
                'symbol': p.symbol,
                'entry_time': p.entry_time.isoformat(),
                'entry_price': p.entry_price,
                'position_size_usd': p.position_size_usd,
                'pred_probability': p.pred_probability,
                'highest_price': p.highest_price,
                'trailing_activated': p.trailing_activated,
            } for sym, p in self.positions.items()},
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def should_block_entry(self, volatility_4h: float, momentum_4h: float) -> tuple:
        """
        Check if entry should be blocked (FOMO trap filter).

        Returns:
            (blocked: bool, reason: str)
        """
        # FOMO trap: high volatility AND high momentum
        if volatility_4h > 0.50 and momentum_4h > 3.0:
            return True, f"FOMO trap: vol={volatility_4h:.2f}, mom={momentum_4h:.1f}"

        # Extreme momentum
        if momentum_4h > self.config.max_momentum_4h:
            return True, f"Extreme momentum: {momentum_4h:.1f} > {self.config.max_momentum_4h}"

        # Extreme volatility
        if volatility_4h > self.config.max_volatility_4h:
            return True, f"Extreme volatility: {volatility_4h:.2f} > {self.config.max_volatility_4h}"

        return False, ""

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

    def get_market_context(self, symbol: str) -> Dict:
        """Get volatility and momentum for entry filter."""
        try:
            # Get last 240 minutes (4 hours) of 1-min data
            cursor = self.sqlite_conn.execute("""
                SELECT close FROM ohlcv
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 240
            """, (symbol,))
            rows = cursor.fetchall()

            if len(rows) < 10:
                return {'volatility_4h': 0, 'momentum_4h': 0}

            closes = [r[0] for r in rows]
            returns = [(closes[i] / closes[i+1] - 1) for i in range(len(closes)-1)]

            volatility = np.std(returns) * 100 if returns else 0
            momentum = ((closes[0] - closes[-1]) / closes[-1] * 100) if closes[-1] > 0 else 0

            return {
                'volatility_4h': volatility,
                'momentum_4h': momentum,
            }
        except:
            return {'volatility_4h': 0, 'momentum_4h': 0}

    def check_exits(self) -> List[PaperTrade]:
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

            # Check ratcheting trailing stop (replaces fixed TP)
            elif pos.trailing_activated or pnl_pct >= self.config.trail_activation_pct:
                pos.trailing_activated = True

                # Get trail distance based on current PnL level (ratcheting)
                # Find the highest level we've reached
                high_pnl_pct = ((pos.highest_price - pos.entry_price) / pos.entry_price) * 100

                trail_pct = 1.2  # Default for activation zone (2-6%)
                for level_pct, trail_dist in self.config.ratchet_levels:
                    if high_pnl_pct >= level_pct:
                        trail_pct = trail_dist

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

                trade = PaperTrade(
                    symbol=symbol,
                    entry_time=pos.entry_time,
                    exit_time=now,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    position_size_usd=pos.position_size_usd,
                    pnl_pct=pnl_pct,
                    pnl_usd=pnl_usd,
                    exit_reason=exit_reason,
                    pred_probability=pos.pred_probability,
                )
                self.trades.append(trade)
                closed.append(trade)
                del self.positions[symbol]

                logger.warning(
                    f"📤 CLOSED {symbol}: {exit_reason.upper()} | "
                    f"PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})"
                )

                # Record to SQLite for visualizer
                pt.record_exit(
                    strategy="spike_predictor",
                    symbol=symbol,
                    exit_price=exit_price,
                    exit_time=now,
                    exit_reason=exit_reason,
                    realized_pnl=pnl_usd,
                )
                self._save_state()

        return closed

    def try_entry(self, signal: Dict) -> bool:
        """Try to enter a new position based on signal."""
        symbol = signal['symbol']

        # Skip if already in position
        if symbol in self.positions:
            return False

        # Skip if at max positions
        if len(self.positions) >= self.config.max_positions:
            return False

        # Get market context
        context = self.get_market_context(symbol)

        # Check volatility filter
        if context['volatility_4h'] < self.config.min_volatility_4h:
            self.signals_filtered += 1
            return False

        # Check bad entry filter
        blocked, reason = self.should_block_entry(
            context['volatility_4h'],
            context['momentum_4h']
        )
        if blocked:
            self.entries_blocked += 1
            logger.info(f"🚫 BLOCKED {symbol}: {reason}")
            return False

        # Get entry price
        entry_price = self.get_current_price(symbol)
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
        self.positions[symbol] = PaperPosition(
            symbol=symbol,
            entry_time=datetime.now(timezone.utc),
            entry_price=entry_price,
            position_size_usd=position_size,
            pred_probability=signal.get('pred_probability', 0.5),
        )

        logger.warning(
            f"📥 ENTERED {symbol} @ ${entry_price:.4f} | "
            f"Size: ${position_size:.2f} | Prob: {signal.get('pred_probability', 0):.2f}"
        )

        # Record to SQLite for visualizer
        pt.record_entry(
            strategy="spike_predictor",
            symbol=symbol,
            entry_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            position_size=position_size,
            quantity=position_size / entry_price,
        )
        self._save_state()

        return True

    def get_signals(self) -> List[Dict]:
        """
        Get new signals from signals file.

        The collector writes signals to data/live_signals.json.
        """
        signals_file = Path("/Users/bz/Pythia2/data/live_signals.json")
        try:
            if not signals_file.exists():
                return []

            with open(signals_file) as f:
                all_signals = json.load(f)

            # Filter to recent signals (last hour)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            recent = []
            for sig in all_signals:
                sig_time = datetime.fromisoformat(sig['timestamp'].replace('Z', '+00:00'))
                if sig_time > cutoff:
                    recent.append(sig)

            return recent
        except Exception as e:
            logger.debug(f"Signal fetch: {e}")
            return []

    def get_signals_from_db(self) -> List[Dict]:
        """
        Legacy: Get signals from DuckDB news_signals table.
        Not used when collector has DB lock.
        """
        try:
            import duckdb
            conn = duckdb.connect(self.db_path, config={'access_mode': 'READ_ONLY'})
            signals = conn.execute("""
                SELECT
                    symbol,
                    timestamp,
                    event_type,
                    event_priority as pred_probability
                FROM news_signals
                WHERE timestamp > NOW() - INTERVAL 1 HOUR
                  AND event_type = 'whale_move'
                  AND symbol NOT LIKE '%USDT%'
                  AND symbol NOT LIKE '%USDC%'
                ORDER BY timestamp DESC
                LIMIT 10
            """).df()

            return signals.to_dict('records')
        except Exception as e:
            logger.debug(f"Signal fetch error: {e}")
            return []

    def print_status(self):
        """Print current status."""
        total_position_value = sum(p.position_size_usd for p in self.positions.values())
        total_equity = self.capital + total_position_value

        total_pnl = sum(t.pnl_usd for t in self.trades)
        win_count = sum(1 for t in self.trades if t.pnl_usd > 0)

        print("\n" + "=" * 60)
        print("SPIKE PAPER TRADER STATUS")
        print("=" * 60)
        print(f"Equity:        ${total_equity:,.2f}")
        print(f"Cash:          ${self.capital:,.2f}")
        print(f"Positions:     {len(self.positions)}/{self.config.max_positions}")
        print(f"Total Trades:  {len(self.trades)}")
        print(f"Win Rate:      {win_count}/{len(self.trades)} ({win_count/max(len(self.trades),1)*100:.1f}%)")
        print(f"Total PnL:     ${total_pnl:+,.2f}")
        print(f"Signals Seen:  {self.signals_seen}")
        print(f"Filtered Out:  {self.signals_filtered}")
        print(f"Blocked (FOMO): {self.entries_blocked}")

        if self.positions:
            print("\nOpen Positions:")
            for sym, pos in self.positions.items():
                current = self.get_current_price(sym) or pos.entry_price
                pnl = ((current - pos.entry_price) / pos.entry_price) * 100
                print(f"  {sym}: ${pos.position_size_usd:.0f} @ ${pos.entry_price:.4f} → ${current:.4f} ({pnl:+.2f}%)")

        if self.trades:
            print("\nRecent Trades:")
            for trade in self.trades[-5:]:
                print(f"  {trade.symbol}: {trade.exit_reason.upper()} {trade.pnl_pct:+.2f}% (${trade.pnl_usd:+.2f})")

        print("=" * 60)

    async def run(self, interval_seconds: int = 60):
        """Run the paper trader loop."""
        self.running = True
        logger.info(f"Starting paper trader (checking every {interval_seconds}s)")

        iteration = 0
        while self.running:
            try:
                iteration += 1

                # Check exits first
                closed = self.check_exits()

                # Get new signals
                signals = self.get_signals()
                self.signals_seen += len(signals)

                # Try entries
                for signal in signals:
                    # Accept pred_probability, event_priority, or confidence as probability
                    prob = signal.get('pred_probability') or signal.get('event_priority') or signal.get('confidence', 0)
                    if prob >= self.config.min_pred_probability:
                        self.try_entry(signal)

                # Print status and save every 5 iterations
                if iteration % 5 == 0:
                    self.print_status()
                    self._save_state()

                await asyncio.sleep(interval_seconds)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Paper trader error: {e}")
                await asyncio.sleep(interval_seconds)

        self.print_status()
        logger.info("Paper trader stopped")


async def main():
    """Run the spike paper trader."""
    import argparse

    parser = argparse.ArgumentParser(description="Spike Prediction Paper Trader")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
    parser.add_argument("--db", type=str, default="data/feature_buffer.db", help="SQLite database path")
    args = parser.parse_args()

    config = PaperTraderConfig(initial_capital=args.capital)
    trader = SpikePaperTrader(config=config, db_path=args.db)

    try:
        await trader.run(interval_seconds=args.interval)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    asyncio.run(main())
