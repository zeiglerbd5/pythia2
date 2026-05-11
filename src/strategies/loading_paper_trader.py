"""
Loading Strategy Paper Trader

Two-phase entry system:
  Phase 1 (LOADING): 20% position when loading score crosses threshold
  Phase 2 (CONFIRMED): Scale to 100% when trigger + 1h confirmation

Exits:
  Phase 1: -3% stop or loading score fades for 1h
  Phase 2: -8% stop, 10% trail from high after +15%, 48h time stop

Uses the same Position/Portfolio classes as the existing paper trading system
for compatibility with the dashboard and visualizer.
"""

import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

try:
    import paper_trades
except ImportError:
    paper_trades = None


@dataclass
class LoadingPosition:
    """A position managed by the loading strategy."""
    symbol: str
    phase: str  # "phase1" or "phase2"

    # Phase 1 entry
    phase1_time: datetime = None
    phase1_price: float = 0.0
    phase1_size: float = 0.0  # Dollar amount (20% of full size)
    phase1_quantity: float = 0.0

    # Phase 2 scale-in
    phase2_time: Optional[datetime] = None
    phase2_price: float = 0.0
    phase2_size: float = 0.0  # Dollar amount (80% of full size)
    phase2_quantity: float = 0.0

    # Blended
    entry_price: float = 0.0  # Weighted average entry
    total_size: float = 0.0
    total_quantity: float = 0.0

    # Tracking
    peak_price: float = 0.0
    current_price: float = 0.0
    loading_score: float = 0.0
    score_below_threshold_since: Optional[datetime] = None
    trailing_stop_active: bool = False  # Set True once gain exceeds trail_activate

    # Exit
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    realized_pnl: float = 0.0

    def pnl_pct(self, price: float = None) -> float:
        p = price or self.current_price
        if self.entry_price <= 0:
            return 0.0
        return (p - self.entry_price) / self.entry_price * 100

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'phase': self.phase,
            'phase1_time': self.phase1_time.isoformat() if self.phase1_time else None,
            'phase1_price': self.phase1_price,
            'phase2_time': self.phase2_time.isoformat() if self.phase2_time else None,
            'phase2_price': self.phase2_price,
            'entry_price': self.entry_price,
            'total_size': self.total_size,
            'peak_price': self.peak_price,
            'current_price': self.current_price,
            'pnl_pct': self.pnl_pct(),
            'trailing_stop_active': self.trailing_stop_active,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason,
            'realized_pnl': self.realized_pnl,
        }


class LoadingPaperTrader:
    """
    Paper trader for the two-phase loading strategy.
    """

    def __init__(
        self,
        full_position_size: float = 1000.0,
        max_positions: int = 5,
        phase1_pct: float = 0.20,     # 20% of full size for Phase 1
        phase1_stop: float = 0.03,    # 3% stop loss for Phase 1
        phase2_stop: float = 0.08,    # 8% stop loss for Phase 2
        trail_pct: float = 0.10,      # 10% trailing stop
        trail_activate: float = 0.15, # Activate trailing after 15% gain
        time_stop_hours: float = 48,  # Max hold time
        score_fade_minutes: int = 60, # Exit Phase 1 if score fades for this long
        state_file: str = "data/loading_trader_state.json",
    ):
        self.full_position_size = full_position_size
        self.max_positions = max_positions
        self.phase1_pct = phase1_pct
        self.phase1_stop = phase1_stop
        self.phase2_stop = phase2_stop
        self.trail_pct = trail_pct
        self.trail_activate = trail_activate
        self.time_stop_hours = time_stop_hours
        self.score_fade_minutes = score_fade_minutes
        self.state_file = Path(state_file)

        # State
        self.positions: Dict[str, LoadingPosition] = {}
        self.closed_positions: List[LoadingPosition] = []
        self.cash = full_position_size * max_positions  # Start with enough for max positions
        self.starting_capital = self.cash

        # Stats
        self.stats = {
            'phase1_entries': 0,
            'phase2_scaleins': 0,
            'phase1_exits': 0,
            'phase2_exits': 0,
            'total_pnl': 0.0,
        }

        # Load existing state
        self._load_state()

        logger.info(
            f"LoadingPaperTrader: ${full_position_size}/pos, max {max_positions}, "
            f"Phase1={phase1_pct*100:.0f}% @{phase1_stop*100:.0f}%stop, "
            f"Phase2={phase2_stop*100:.0f}%stop, trail={trail_pct*100:.0f}%@{trail_activate*100:.0f}%"
        )

    def on_loading_alert(self, symbol: str, score: float, price: float, timestamp: datetime) -> bool:
        """
        Called when the loading scanner fires an alert.
        Opens a Phase 1 position (small, speculative).
        Returns True if position was opened, False if blocked.
        """
        if symbol in self.positions:
            # Update score on existing position
            self.positions[symbol].loading_score = score
            self.positions[symbol].score_below_threshold_since = None
            return True  # Already in, counts as success

        if len(self.positions) >= self.max_positions:
            logger.debug(f"[LOADING_TRADER] Max positions reached, skipping {symbol}")
            return False

        phase1_size = self.full_position_size * self.phase1_pct
        if self.cash < phase1_size:
            logger.debug(f"[LOADING_TRADER] Insufficient cash for {symbol}")
            return False

        quantity = phase1_size / price if price > 0 else 0

        pos = LoadingPosition(
            symbol=symbol,
            phase="phase1",
            phase1_time=timestamp,
            phase1_price=price,
            phase1_size=phase1_size,
            phase1_quantity=quantity,
            entry_price=price,
            total_size=phase1_size,
            total_quantity=quantity,
            peak_price=price,
            current_price=price,
            loading_score=score,
        )

        self.positions[symbol] = pos
        self.cash -= phase1_size
        self.stats['phase1_entries'] += 1

        logger.warning(
            f"[LOADING_TRADER] PHASE 1 ENTRY: {symbol} "
            f"${phase1_size:.0f} @ ${price:.6f} (score={score:.1f})"
        )

        self._record_entry(pos)
        self._save_state()
        return True

    def on_trigger_confirmed(self, symbol: str, price: float, timestamp: datetime):
        """
        Called when a loading alert coin triggers (5%+) and confirms (still positive at T+1h).
        Scales Phase 1 to full Phase 2 position.
        """
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        if pos.phase != "phase1":
            return

        phase2_size = self.full_position_size * (1.0 - self.phase1_pct)
        if self.cash < phase2_size:
            logger.warning(f"[LOADING_TRADER] Insufficient cash for Phase 2 scale-in on {symbol}")
            return

        phase2_quantity = phase2_size / price if price > 0 else 0

        pos.phase = "phase2"
        pos.phase2_time = timestamp
        pos.phase2_price = price
        pos.phase2_size = phase2_size
        pos.phase2_quantity = phase2_quantity
        pos.total_size += phase2_size
        pos.total_quantity += phase2_quantity

        # Blended entry price
        pos.entry_price = pos.total_size / pos.total_quantity if pos.total_quantity > 0 else price
        pos.peak_price = max(pos.peak_price, price)

        self.cash -= phase2_size
        self.stats['phase2_scaleins'] += 1

        gain_from_p1 = (price - pos.phase1_price) / pos.phase1_price * 100

        logger.warning(
            f"[LOADING_TRADER] PHASE 2 SCALE-IN: {symbol} "
            f"+${phase2_size:.0f} @ ${price:.6f} (total ${pos.total_size:.0f}, "
            f"blended entry ${pos.entry_price:.6f}, Phase1 +{gain_from_p1:.1f}%)"
        )

        self._record_entry(pos)
        self._save_state()

    def update_prices(self, prices: Dict[str, float], timestamp: datetime, scores: Dict[str, float] = None):
        """
        Update prices and check exit conditions for all open positions.
        Called periodically (every 1 min from loading scanner).
        """
        positions_to_close = []

        for symbol, pos in self.positions.items():
            if symbol not in prices:
                continue

            price = prices[symbol]
            pos.current_price = price
            pos.peak_price = max(pos.peak_price, price)

            # Update loading score
            if scores and symbol in scores:
                new_score = scores[symbol]
                pos.loading_score = new_score

            pnl = pos.pnl_pct(price)

            # ── EXIT CHECKS ──────────────────────────────────

            # Phase 1 exits
            if pos.phase == "phase1":
                # Stop loss
                if pnl <= -self.phase1_stop * 100:
                    positions_to_close.append((symbol, "phase1_stop", price, timestamp))
                    continue

                # Score fade: if loading score has been below threshold for too long
                if scores and symbol in scores:
                    if scores[symbol] < 5.0:  # Below original threshold
                        if pos.score_below_threshold_since is None:
                            pos.score_below_threshold_since = timestamp
                        elif (timestamp - pos.score_below_threshold_since).total_seconds() > self.score_fade_minutes * 60:
                            positions_to_close.append((symbol, "score_fade", price, timestamp))
                            continue
                    else:
                        pos.score_below_threshold_since = None

                # Time stop for Phase 1 (don't hold speculative positions forever)
                if pos.phase1_time and (timestamp - pos.phase1_time).total_seconds() > 6 * 3600:
                    positions_to_close.append((symbol, "phase1_timeout", price, timestamp))
                    continue

            # Phase 2 exits
            elif pos.phase == "phase2":
                entry_time = pos.phase2_time or pos.phase1_time

                # Hard stop loss
                if pnl <= -self.phase2_stop * 100:
                    positions_to_close.append((symbol, "phase2_stop", price, timestamp))
                    continue

                # Trailing stop: activate permanently once gain exceeds threshold
                if pnl >= self.trail_activate * 100:
                    pos.trailing_stop_active = True

                if pos.trailing_stop_active:
                    trail_price = pos.peak_price * (1 - self.trail_pct)
                    if price <= trail_price:
                        positions_to_close.append((symbol, "trailing_stop", price, timestamp))
                        continue

                # Time stop
                if entry_time and (timestamp - entry_time).total_seconds() > self.time_stop_hours * 3600:
                    positions_to_close.append((symbol, "time_stop", price, timestamp))
                    continue

        # Execute closes
        for symbol, reason, price, ts in positions_to_close:
            self._close_position(symbol, reason, price, ts)

    def _close_position(self, symbol: str, reason: str, price: float, timestamp: datetime):
        """Close a position and record the result."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        pos.exit_price = price
        pos.exit_time = timestamp
        pos.exit_reason = reason

        # Calculate P&L
        pnl_pct = pos.pnl_pct(price)
        pos.realized_pnl = pos.total_size * (pnl_pct / 100)

        # Return capital
        self.cash += pos.total_size + pos.realized_pnl
        self.stats['total_pnl'] += pos.realized_pnl

        if pos.phase == "phase1":
            self.stats['phase1_exits'] += 1
        else:
            self.stats['phase2_exits'] += 1

        hold_time = (timestamp - (pos.phase1_time or timestamp)).total_seconds() / 3600

        logger.warning(
            f"[LOADING_TRADER] EXIT: {symbol} ({reason}) "
            f"pnl={pnl_pct:+.1f}% (${pos.realized_pnl:+.2f}) "
            f"phase={pos.phase} hold={hold_time:.1f}h"
        )

        # Record to SQLite for visualizer
        if paper_trades:
            try:
                paper_trades.record_exit(
                    strategy="loading_v1",
                    symbol=symbol,
                    exit_price=price,
                    exit_time=timestamp,
                    exit_reason=reason,
                    realized_pnl=pos.realized_pnl,
                )
            except Exception as e:
                logger.debug(f"Failed to record exit to SQLite: {e}")

        self.closed_positions.append(pos)
        del self.positions[symbol]
        self._save_state()

    def _record_entry(self, pos: LoadingPosition):
        """Record entry to SQLite for visualizer."""
        if paper_trades:
            try:
                paper_trades.record_entry(
                    strategy="loading_v1",
                    symbol=pos.symbol,
                    entry_price=pos.entry_price,
                    entry_time=pos.phase1_time,
                    position_size=pos.total_size,
                    quantity=pos.total_quantity,
                )
            except Exception as e:
                logger.debug(f"Failed to record entry to SQLite: {e}")

    def get_summary(self) -> str:
        """Get a one-line summary of current state."""
        open_p1 = sum(1 for p in self.positions.values() if p.phase == "phase1")
        open_p2 = sum(1 for p in self.positions.values() if p.phase == "phase2")
        total_pnl = self.stats['total_pnl']
        n_closed = len(self.closed_positions)
        return (
            f"Loading Trader: {open_p1} Phase1 + {open_p2} Phase2 open | "
            f"{n_closed} closed | PnL: ${total_pnl:+.2f}"
        )

    def _save_state(self):
        """Save state to JSON for persistence across restarts."""
        state = {
            'cash': self.cash,
            'starting_capital': self.starting_capital,
            'stats': self.stats,
            'positions': {s: p.to_dict() for s, p in self.positions.items()},
            'closed_count': len(self.closed_positions),
            'last_closed': [p.to_dict() for p in self.closed_positions[-20:]],  # Keep last 20
        }
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"Failed to save loading trader state: {e}")

    def _load_state(self):
        """Load state from JSON."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file) as f:
                state = json.load(f)

            self.cash = state.get('cash', self.cash)
            self.starting_capital = state.get('starting_capital', self.starting_capital)
            self.stats = state.get('stats', self.stats)

            # Restore open positions
            for sym, pdict in state.get('positions', {}).items():
                pos = LoadingPosition(
                    symbol=sym,
                    phase=pdict.get('phase', 'phase1'),
                )
                pos.phase1_price = pdict.get('phase1_price', 0)
                pos.phase2_price = pdict.get('phase2_price', 0)
                pos.entry_price = pdict.get('entry_price', 0)
                pos.total_size = pdict.get('total_size', 0)
                pos.peak_price = pdict.get('peak_price', 0)
                pos.current_price = pdict.get('current_price', 0)
                pos.trailing_stop_active = pdict.get('trailing_stop_active', False)

                if pdict.get('phase1_time'):
                    pos.phase1_time = datetime.fromisoformat(pdict['phase1_time'])
                if pdict.get('phase2_time'):
                    pos.phase2_time = datetime.fromisoformat(pdict['phase2_time'])

                self.positions[sym] = pos

            n_open = len(self.positions)
            if n_open > 0:
                logger.info(f"[LOADING_TRADER] Restored {n_open} open positions from state")

        except Exception as e:
            logger.warning(f"Failed to load loading trader state: {e}")

        # Reconcile: close orphaned DB records not tracked in memory
        if paper_trades:
            active = set(self.positions.keys())
            closed_count = paper_trades.close_orphaned_positions(active)
            if closed_count:
                logger.warning(f"[LOADING_TRADER] Closed {closed_count} orphaned DB positions on startup")
