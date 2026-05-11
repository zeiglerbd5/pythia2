"""
Breakout Hunter Strategy v5.3 - Triple Confirmation + Enhanced Exit Structure

Detects breakouts and waits for THREE confirmations before entry:
1. T+1 must be positive (continuation, not reversal) - FILTER ONLY
2. T+2 must show strong resumption (>5%)
3. Volume must continue (next 3h > previous 6h)

This combination achieved 94.6% win rate over 143 days of backtesting.

ENTRY CRITERIA (Triple Confirmation):
1. T+0 (Breakout Hour):
   - Price move >5%
   - Volume >3x 24h average
2. T+1 (Filter - NOT entry point):
   - Return from T+0 close > 0% (MUST be positive)
   - This filters out immediate reversals
   - NOTE: T+1 entry was tested and failed - see STRATEGY_RESEARCH.md
3. T+2 (Resumption - ENTRY POINT):
   - Return from T+1 close >= 5%
   - Volume continuation > 1x (next 3h vol > prev 6h vol)
   - Entry at T+2 OPEN price

EXIT CRITERIA (Enhanced v5.3):
- Initial stop loss: 8% from entry
- Stepped profit locks:
  - Once up +5%, lock -2% (small loss ok)
  - Once up +10%, lock breakeven (0%)
  - Once up +15%, lock +5% profit
  - Once up +25%, lock +12% profit
  - Once up +40%, lock +25% profit
- Tiered trailing stop:
  - +20%: Activate 10% trail
  - +30%: Tighten to 8% trail
  - +50%: Tighten to 6% trail
  - +70%: Tighten to 5% trail
- Max hold: 48 hours

BACKTESTED RESULTS (Oct 2025 - Mar 2026, 143 days):
- 56 trades, 94.6% win rate
- Expected return: +61.9% per trade
- ~1 trade every 2.5 days

KEY INSIGHTS:
- T+1 MUST be positive - negative T+1 = likely reversal
- T+1 is a FILTER, not an entry point (T+1 entry has 33% win rate - regression)
- Volume continuation confirms buying pressure hasn't exhausted
- This filters out 97% of signals, but the ones that pass are gold
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BreakoutSignal:
    """Represents a detected breakout signal (Triple Confirmation Pattern)"""
    symbol: str
    signal_time: datetime          # T+0 timestamp
    initial_move_pct: float        # T+0 price move
    volume_ratio: float            # T+0 volume ratio
    signal_price: float            # T+0 close price
    pre_signal_volume: float = 0.0 # Volume in 6h before signal (for continuation calc)
    status: str = 'pending_t1'     # pending_t1, pending_t2, confirmed, expired
    t1_close: Optional[float] = None      # T+1 close price
    t1_return: Optional[float] = None     # T+1 return from T+0 close
    t1_time: Optional[datetime] = None    # T+1 timestamp
    t2_open: Optional[float] = None       # T+2 open price (entry price)
    t2_return: Optional[float] = None     # T+2 return from T+1 close
    vol_continuation: Optional[float] = None  # Volume continuation ratio
    confirmation_time: Optional[datetime] = None


@dataclass
class BreakoutTrade:
    """Represents an active or completed trade"""
    symbol: str
    entry_time: datetime
    entry_price: float
    position_size: float
    quantity: float
    initial_stop_loss: float
    take_profit_price: float
    max_hold_time: datetime
    status: str = 'open'  # open, closed
    highest_price: float = 0.0  # Track highest for trailing stop
    trailing_active: bool = False  # Whether trailing stop is active
    current_stop: float = 0.0  # Current stop loss level
    locked_profit_pct: float = -1.0  # Current locked profit level (-1 = none, 0 = breakeven, etc)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None


class BreakoutHunter:
    """
    Breakout Hunter Strategy

    Monitors for volume + price breakouts, waits for confirmation,
    then enters with wide stops and high profit targets.
    """

    # Strategy parameters v5.3 - T+2 entry with improved exit structure
    VOLUME_THRESHOLD = 3.0          # Volume must be 3x 24h average
    INITIAL_MOVE_PCT = 5.0          # T+0 breakout must be >5%
    T1_MIN_RETURN_PCT = 0.0         # T+1 must be positive (>0%) - FILTER only
    ENTER_ON_T1 = False             # v5.3: Reverted to T+2 entry (T+1 entry was a regression)

    # T+2 entry parameters
    T2_RESUMPTION_PCT = 5.0         # T+2 return from T+1 close must be >5%
    VOL_CONTINUATION_MIN = 1.0      # Volume continuation must be >1x

    # Exit parameters with profit locks and trailing stop
    INITIAL_STOP_LOSS_PCT = 0.08    # 8% initial stop loss

    # Stepped profit lock levels - never let winners become losers
    PROFIT_LOCK_LEVELS = [
        (0.05, -0.02),  # Once up 5%, lock -2% (small loss ok)
        (0.10, 0.00),   # Once up 10%, lock breakeven
        (0.15, 0.05),   # Once up 15%, lock 5% profit
        (0.25, 0.12),   # Once up 25%, lock 12% profit
        (0.40, 0.25),   # Once up 40%, lock 25% profit
    ]

    TRAIL_TRIGGER_PCT = 0.20        # Activate trailing after +20%
    TRAIL_STOP_PCT = 0.10           # Trail at 10% below high (default)

    # Tiered trailing - tighten trail at higher gains to lock in more profit
    TRAIL_TIGHTEN_LEVELS = [
        (0.30, 0.08),   # Once up 30%, tighten to 8% trail
        (0.50, 0.06),   # Once up 50%, tighten to 6% trail
        (0.70, 0.05),   # Once up 70%, tighten to 5% trail
    ]

    MAX_TAKE_PROFIT_PCT = 0.80      # Cap at 80% profit
    MAX_HOLD_HOURS = 48             # Max 48 hour hold

    POSITION_SIZE_USD = 2500    # Position size per trade
    MAX_POSITIONS = 3           # Max concurrent positions
    FEE_RATE = 0.0055           # Coinbase fee rate

    def __init__(self, paper_trading: bool = True):
        """Initialize the breakout hunter strategy"""
        self.paper_trading = paper_trading

        # State tracking
        self.pending_signals: Dict[str, BreakoutSignal] = {}  # symbol -> signal
        self.active_trades: Dict[str, BreakoutTrade] = {}     # symbol -> trade
        self.completed_trades: List[BreakoutTrade] = []

        # Hourly data cache for volume ratio calculation
        self.hourly_data: Dict[str, pd.DataFrame] = {}  # symbol -> hourly OHLCV

        version = "5.3"  # T+2 entry with enhanced exit structure
        entry_mode = "T+1" if self.ENTER_ON_T1 else "T+2"
        logger.info(f"BreakoutHunter v{version} initialized (paper_trading={paper_trading})")
        logger.info(f"Entry: T+0>{self.INITIAL_MOVE_PCT}%, Vol>{self.VOLUME_THRESHOLD}x, "
                   f"T+1>{self.T1_MIN_RETURN_PCT}% → ENTER on {entry_mode}")
        lock_str = ", ".join([f"+{t*100:.0f}%→{l*100:.1f}%" for t, l in self.PROFIT_LOCK_LEVELS])
        trail_str = f"{self.TRAIL_STOP_PCT*100:.0f}%@+{self.TRAIL_TRIGGER_PCT*100:.0f}%"
        for thresh, pct in self.TRAIL_TIGHTEN_LEVELS:
            trail_str += f", {pct*100:.0f}%@+{thresh*100:.0f}%"
        logger.info(f"Exit: SL={self.INITIAL_STOP_LOSS_PCT*100:.0f}%, "
                   f"Locks=[{lock_str}], "
                   f"Trail=[{trail_str}]")

    def update_hourly_data(self, symbol: str, hourly_df: pd.DataFrame):
        """Update cached hourly data for a symbol"""
        self.hourly_data[symbol] = hourly_df.sort_values('timestamp').tail(48)  # Keep 48 hours

    def calculate_volume_ratio(self, symbol: str, current_volume: float) -> float:
        """Calculate volume ratio vs 24h average"""
        if symbol not in self.hourly_data:
            return 0.0

        df = self.hourly_data[symbol]
        if len(df) < 12:
            return 0.0

        # Average of last 24 hours (excluding current)
        avg_vol = df['volume'].iloc[-25:-1].mean() if len(df) > 24 else df['volume'].iloc[:-1].mean()

        if avg_vol <= 0:
            return 0.0

        return current_volume / avg_vol

    def get_pre_signal_volume(self, symbol: str, hours: int = 6) -> float:
        """Get total volume in the N hours before the most recent candle"""
        if symbol not in self.hourly_data:
            return 0.0

        df = self.hourly_data[symbol]
        if len(df) < hours + 1:
            return 0.0

        # Sum of volume in the hours before the last candle
        return df['volume'].iloc[-(hours+1):-1].sum()

    def check_for_breakout(self, symbol: str, hourly_candle: dict) -> Optional[BreakoutSignal]:
        """
        Check if an hourly candle represents a T+0 breakout signal.

        Args:
            symbol: Trading pair symbol
            hourly_candle: Dict with 'open', 'high', 'low', 'close', 'volume', 'timestamp'

        Returns:
            BreakoutSignal if T+0 criteria met, None otherwise
        """
        # Skip if we already have a pending signal or active trade
        if symbol in self.pending_signals or symbol in self.active_trades:
            return None

        # Skip if max positions reached
        if len(self.active_trades) >= self.MAX_POSITIONS:
            return None

        open_price = hourly_candle['open']
        close_price = hourly_candle['close']
        volume = hourly_candle['volume']
        timestamp = hourly_candle['timestamp']

        if open_price <= 0:
            return None

        # Calculate hourly return
        hour_return = (close_price - open_price) / open_price * 100

        # Check minimum initial move (T+0 must be > threshold)
        if hour_return < self.INITIAL_MOVE_PCT:
            return None

        # Calculate volume ratio
        vol_ratio = self.calculate_volume_ratio(symbol, volume)

        # Check volume threshold
        if vol_ratio < self.VOLUME_THRESHOLD:
            return None

        # Get pre-signal volume for later continuation calculation
        pre_vol = self.get_pre_signal_volume(symbol, hours=6)

        # T+0 Breakout detected - create pending signal
        signal = BreakoutSignal(
            symbol=symbol,
            signal_time=timestamp,
            initial_move_pct=hour_return,
            volume_ratio=vol_ratio,
            signal_price=close_price,
            pre_signal_volume=pre_vol,
            status='pending_t1'
        )

        self.pending_signals[symbol] = signal

        logger.info(f"T+0 BREAKOUT: {symbol} | Move: +{hour_return:.1f}% | "
                   f"Vol: {vol_ratio:.1f}x | Price: ${close_price:.6f}")
        if self.ENTER_ON_T1:
            logger.info(f"  Waiting for T+1 confirmation (must be positive)...")
        else:
            logger.info(f"  Waiting for T+1 then T+2 resumption >={self.T2_RESUMPTION_PCT}%...")

        return signal

    def update_t1_data(self, symbol: str, t1_close: float, t1_time: datetime, t1_volume: float = 0) -> Optional[BreakoutTrade]:
        """
        Update T+1 close data for a pending signal.
        Called when T+1 hour completes.

        v5.2: If ENTER_ON_T1=True, enter trade immediately when T+1 is positive.
        v5.0/5.1: T+1 MUST be positive (>0%) or signal is rejected.

        Returns:
            BreakoutTrade if entering on T+1, None otherwise
        """
        if symbol not in self.pending_signals:
            return None

        signal = self.pending_signals[symbol]
        if signal.status != 'pending_t1':
            return None

        # Calculate T+1 return
        t1_return = (t1_close - signal.signal_price) / signal.signal_price * 100
        signal.t1_return = t1_return
        signal.t1_close = t1_close
        signal.t1_time = t1_time

        # T+1 MUST be positive - this is the key filter
        if t1_return <= self.T1_MIN_RETURN_PCT:
            logger.info(f"T+1 FAILED: {symbol} | T+1 return: {t1_return:+.1f}% (need >{self.T1_MIN_RETURN_PCT}%)")
            logger.info(f"  Signal rejected - immediate reversal detected")
            signal.status = 'expired'
            del self.pending_signals[symbol]
            return None

        logger.info(f"T+1 PASSED: {symbol} | T+1 return: {t1_return:+.1f}% | Close: ${t1_close:.6f}")

        # v5.2: Enter on T+1 close instead of waiting for T+2
        if self.ENTER_ON_T1:
            return self._enter_trade(symbol, signal, t1_close, t1_time)
        else:
            signal.status = 'pending_t2'
            logger.info(f"  Waiting for T+2 resumption with vol continuation...")
            return None

    def _enter_trade(self, symbol: str, signal: BreakoutSignal, entry_price: float, entry_time: datetime) -> BreakoutTrade:
        """
        Create and register a new trade entry.
        """
        if len(self.active_trades) >= self.MAX_POSITIONS:
            logger.warning(f"MAX POSITIONS ({self.MAX_POSITIONS}) reached - skipping {symbol}")
            signal.status = 'expired'
            del self.pending_signals[symbol]
            return None

        signal.status = 'confirmed'
        signal.confirmation_time = entry_time

        quantity = self.POSITION_SIZE_USD / entry_price
        initial_stop = entry_price * (1 - self.INITIAL_STOP_LOSS_PCT)
        take_profit = entry_price * (1 + self.MAX_TAKE_PROFIT_PCT)
        max_hold = entry_time + timedelta(hours=self.MAX_HOLD_HOURS)

        trade = BreakoutTrade(
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            position_size=self.POSITION_SIZE_USD,
            quantity=quantity,
            initial_stop_loss=initial_stop,
            take_profit_price=take_profit,
            max_hold_time=max_hold,
            highest_price=entry_price,
            current_stop=initial_stop
        )

        self.active_trades[symbol] = trade
        del self.pending_signals[symbol]

        logger.info(f"ENTRY: {symbol} @ ${entry_price:.6f}")
        logger.info(f"  T+0: +{signal.initial_move_pct:.1f}% | Vol: {signal.volume_ratio:.1f}x | T+1: +{signal.t1_return:.1f}%")
        logger.info(f"  Stop: ${initial_stop:.6f} ({self.INITIAL_STOP_LOSS_PCT*100:.0f}%) | Target: ${take_profit:.6f}")

        return trade

    def check_confirmation(self, symbol: str, current_price: float,
                          current_time: datetime, t2_open: float = None,
                          post_signal_volume: float = 0) -> Optional[BreakoutTrade]:
        """
        Check if a pending signal meets all v5.0 Triple Confirmation criteria.

        Args:
            symbol: Trading pair symbol
            current_price: Current price (T+2 close)
            current_time: Current timestamp
            t2_open: T+2 open price (entry price if confirmed)
            post_signal_volume: Volume in 3h after signal (for continuation calc)

        Returns:
            BreakoutTrade if all criteria met, None otherwise
        """
        if symbol not in self.pending_signals:
            return None

        signal = self.pending_signals[symbol]

        # Check if signal has expired (more than 3 hours old - need T+0, T+1, T+2)
        if current_time - signal.signal_time > timedelta(hours=3):
            logger.info(f"Signal expired: {symbol} (no confirmation within window)")
            signal.status = 'expired'
            del self.pending_signals[symbol]
            return None

        # Need T+1 data first
        if signal.status == 'pending_t1':
            return None  # Still waiting for T+1 to complete

        if signal.status != 'pending_t2':
            return None

        # Calculate T+2 return from T+1 close
        if signal.t1_close is None or signal.t1_close <= 0:
            return None

        t2_return = (current_price - signal.t1_close) / signal.t1_close * 100

        # Calculate volume continuation
        vol_continuation = 0.0
        if signal.pre_signal_volume > 0 and post_signal_volume > 0:
            vol_continuation = post_signal_volume / signal.pre_signal_volume

        # Check T+2 resumption threshold
        if t2_return < self.T2_RESUMPTION_PCT:
            if t2_return < -5:
                # Strong reversal - fail fast
                logger.info(f"T+2 FAILED: {symbol} | T+2: {t2_return:+.1f}% (weak resumption)")
                signal.status = 'expired'
                del self.pending_signals[symbol]
            return None

        # Check volume continuation
        if vol_continuation < self.VOL_CONTINUATION_MIN:
            logger.info(f"T+2 VOL FAILED: {symbol} | T+2: {t2_return:+.1f}% but VolCont: {vol_continuation:.2f}x (need >{self.VOL_CONTINUATION_MIN}x)")
            signal.status = 'expired'
            del self.pending_signals[symbol]
            return None

        # ALL THREE CRITERIA MET - Enter trade at T+2 OPEN
        signal.status = 'confirmed'
        signal.t2_return = t2_return
        signal.vol_continuation = vol_continuation
        signal.t2_open = t2_open if t2_open else current_price
        signal.confirmation_time = current_time

        # Entry at T+2 OPEN price
        entry_price = signal.t2_open
        quantity = self.POSITION_SIZE_USD / entry_price
        initial_stop = entry_price * (1 - self.INITIAL_STOP_LOSS_PCT)
        take_profit = entry_price * (1 + self.MAX_TAKE_PROFIT_PCT)
        max_hold = current_time + timedelta(hours=self.MAX_HOLD_HOURS)

        trade = BreakoutTrade(
            symbol=symbol,
            entry_time=current_time,
            entry_price=entry_price,
            position_size=self.POSITION_SIZE_USD,
            quantity=quantity,
            initial_stop_loss=initial_stop,
            take_profit_price=take_profit,
            max_hold_time=max_hold,
            highest_price=entry_price,
            trailing_active=False,
            current_stop=initial_stop
        )

        self.active_trades[symbol] = trade
        del self.pending_signals[symbol]

        logger.info(f"TRIPLE CONFIRMED: {symbol}")
        logger.info(f"  T+0: +{signal.initial_move_pct:.1f}% | T+1: +{signal.t1_return:.1f}% | T+2: +{t2_return:.1f}%")
        logger.info(f"  VolCont: {vol_continuation:.1f}x | Entry: ${entry_price:.6f}")
        logger.info(f"  Stop: ${initial_stop:.6f} ({self.INITIAL_STOP_LOSS_PCT*100}%)")

        return trade

    def check_exit_conditions(self, symbol: str, current_price: float,
                             current_time: datetime) -> Optional[str]:
        """
        Check if an active trade should be exited.
        Implements profit locks and trailing stop logic.

        Args:
            symbol: Trading pair symbol
            current_price: Current price
            current_time: Current timestamp

        Returns:
            Exit reason string if should exit, None otherwise
        """
        if symbol not in self.active_trades:
            return None

        trade = self.active_trades[symbol]

        # Update highest price
        if current_price > trade.highest_price:
            trade.highest_price = current_price

        # Calculate current return from peak
        peak_return = (trade.highest_price - trade.entry_price) / trade.entry_price

        # Check and update profit locks
        for threshold, lock_pct in self.PROFIT_LOCK_LEVELS:
            if peak_return >= threshold and trade.locked_profit_pct < lock_pct:
                trade.locked_profit_pct = lock_pct
                lock_price = trade.entry_price * (1 + lock_pct)
                logger.info(f"PROFIT LOCKED: {symbol} at {lock_pct*100:.0f}% (${lock_price:.6f}) | Peak: +{peak_return*100:.1f}%")

        # Check if trailing stop should activate
        trigger_price = trade.entry_price * (1 + self.TRAIL_TRIGGER_PCT)
        if not trade.trailing_active and trade.highest_price >= trigger_price:
            trade.trailing_active = True
            logger.info(f"TRAILING STOP ACTIVATED: {symbol} at ${trade.highest_price:.6f} (+{peak_return*100:.1f}%)")

        # Determine current stop level (highest of: initial stop, profit lock, trailing stop)
        stop_candidates = [trade.initial_stop_loss]

        # Add profit lock stop if active
        if trade.locked_profit_pct >= 0:
            lock_stop = trade.entry_price * (1 + trade.locked_profit_pct)
            stop_candidates.append(lock_stop)

        # Add trailing stop if active (with tiered tightening)
        if trade.trailing_active:
            # Start with default trail percentage
            active_trail_pct = self.TRAIL_STOP_PCT

            # Check if we should use a tighter trail based on peak return
            for threshold, tighter_pct in self.TRAIL_TIGHTEN_LEVELS:
                if peak_return >= threshold:
                    active_trail_pct = tighter_pct

            trail_stop = trade.highest_price * (1 - active_trail_pct)
            stop_candidates.append(trail_stop)

        trade.current_stop = max(stop_candidates)

        # Check stop loss (initial, profit lock, or trailing)
        if current_price <= trade.current_stop:
            if trade.trailing_active:
                return 'trail_stop'
            elif trade.locked_profit_pct >= 0:
                return 'profit_lock'
            else:
                return 'stop_loss'

        # Check take profit cap
        if current_price >= trade.take_profit_price:
            return 'take_profit'

        # Check max hold time
        if current_time >= trade.max_hold_time:
            return 'max_hold'

        return None

    def exit_trade(self, symbol: str, exit_price: float, exit_time: datetime,
                  exit_reason: str) -> BreakoutTrade:
        """
        Exit an active trade.

        Args:
            symbol: Trading pair symbol
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: Reason for exit

        Returns:
            Completed trade object
        """
        trade = self.active_trades[symbol]

        # If we're exiting due to stop/lock but price gapped past our stop,
        # simulate fill at stop price (real stop order would have filled there)
        actual_exit = exit_price
        if exit_reason in ('stop_loss', 'profit_lock', 'trail_stop'):
            if exit_price < trade.current_stop:
                # Price gapped past our stop - assume fill at stop with 2% slippage
                actual_exit = trade.current_stop * 0.98
                logger.info(f"GAP DETECTED: Price ${exit_price:.6f} below stop ${trade.current_stop:.6f}, "
                           f"simulating fill at ${actual_exit:.6f}")

        # Calculate P&L using actual exit price
        gross_return = (actual_exit - trade.entry_price) / trade.entry_price
        fees = self.FEE_RATE * 2  # Entry + exit
        net_return = gross_return - fees
        pnl = trade.position_size * net_return

        # Update trade
        trade.status = 'closed'
        trade.exit_time = exit_time
        trade.exit_price = actual_exit
        trade.exit_reason = exit_reason
        trade.pnl = pnl

        # Move to completed
        self.completed_trades.append(trade)
        del self.active_trades[symbol]

        result = "WIN" if pnl > 0 else "LOSS"
        max_return = (trade.highest_price - trade.entry_price) / trade.entry_price * 100
        captured_pct = (net_return * 100 / max_return * 100) if max_return > 0 else 0

        logger.info(f"TRADE CLOSED: {symbol} | {exit_reason}")
        logger.info(f"  Exit: ${exit_price:.6f} | Net: {net_return*100:+.1f}% | Max: {max_return:+.1f}%")
        logger.info(f"  Captured: {captured_pct:.0f}% of max move | P&L: ${pnl:+.0f} | {result}")

        return trade

    def get_stats(self) -> dict:
        """Get strategy statistics"""
        if not self.completed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_pnl': 0
            }

        pnls = [t.pnl for t in self.completed_trades]
        returns = [(t.exit_price - t.entry_price) / t.entry_price * 100
                  for t in self.completed_trades]
        wins = sum(1 for p in pnls if p > 0)

        return {
            'total_trades': len(self.completed_trades),
            'wins': wins,
            'losses': len(self.completed_trades) - wins,
            'win_rate': wins / len(self.completed_trades) * 100,
            'avg_return': np.mean(returns),
            'total_pnl': sum(pnls),
            'best_trade': max(pnls),
            'worst_trade': min(pnls),
            'active_positions': len(self.active_trades),
            'pending_signals': len(self.pending_signals)
        }

    def print_status(self):
        """Print current strategy status"""
        stats = self.get_stats()

        print("\n" + "=" * 50)
        print("BREAKOUT HUNTER STATUS")
        print("=" * 50)
        print(f"Completed: {stats['total_trades']} trades | "
              f"Win Rate: {stats['win_rate']:.0f}% | "
              f"P&L: ${stats['total_pnl']:+,.0f}")

        if self.pending_signals:
            print(f"\nPending signals ({len(self.pending_signals)}):")
            for sym, sig in self.pending_signals.items():
                print(f"  {sym}: +{sig.initial_move_pct:.1f}% at ${sig.signal_price:.6f}")

        if self.active_trades:
            print(f"\nActive trades ({len(self.active_trades)}):")
            for sym, trade in self.active_trades.items():
                print(f"  {sym}: Entry ${trade.entry_price:.6f} | "
                      f"SL ${trade.current_stop:.6f} | TP ${trade.take_profit_price:.6f}")

    def save_state(self, filepath: str = 'breakout_hunter_state.json'):
        """Save current state to JSON file for dashboard"""
        import json

        state = {
            'last_update': datetime.now().isoformat(),
            'paper_trading': self.paper_trading,
            'params': {
                'version': '5.3',
                'volume_threshold': self.VOLUME_THRESHOLD,
                'initial_move_pct': self.INITIAL_MOVE_PCT,
                't1_min_return_pct': self.T1_MIN_RETURN_PCT,
                't2_resumption_pct': self.T2_RESUMPTION_PCT,
                'vol_continuation_min': self.VOL_CONTINUATION_MIN,
                'initial_stop_loss_pct': self.INITIAL_STOP_LOSS_PCT,
                'profit_lock_levels': self.PROFIT_LOCK_LEVELS,
                'trail_trigger_pct': self.TRAIL_TRIGGER_PCT,
                'trail_stop_pct': self.TRAIL_STOP_PCT,
                'position_size_usd': self.POSITION_SIZE_USD,
            },
            'pending_signals': [
                {
                    'symbol': sig.symbol,
                    'signal_time': sig.signal_time.isoformat(),
                    'initial_move_pct': sig.initial_move_pct,
                    'volume_ratio': sig.volume_ratio,
                    'signal_price': sig.signal_price,
                    'status': sig.status,
                }
                for sig in self.pending_signals.values()
            ],
            'active_trades': [
                {
                    'symbol': trade.symbol,
                    'entry_time': trade.entry_time.isoformat(),
                    'entry_price': trade.entry_price,
                    'position_size': trade.position_size,
                    'quantity': trade.quantity,
                    'initial_stop_loss': trade.initial_stop_loss,
                    'current_stop': trade.current_stop,
                    'take_profit_price': trade.take_profit_price,
                    'highest_price': trade.highest_price,
                    'trailing_active': trade.trailing_active,
                    'locked_profit_pct': trade.locked_profit_pct,
                    'max_hold_time': trade.max_hold_time.isoformat(),
                }
                for trade in self.active_trades.values()
            ],
            'completed_trades': [
                {
                    'symbol': trade.symbol,
                    'entry_time': trade.entry_time.isoformat(),
                    'entry_price': trade.entry_price,
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'exit_price': trade.exit_price,
                    'exit_reason': trade.exit_reason,
                    'pnl': trade.pnl,
                    'highest_price': trade.highest_price,
                    'position_size': trade.position_size,
                }
                for trade in self.completed_trades
            ],
            'stats': self.get_stats(),
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        return state


def main():
    """Test the strategy with sample data"""
    import sqlite3

    print("Loading test data...")
    conn = sqlite3.connect('/Users/bz/Pythia2/data/feature_buffer.db')

    # Get minute data
    df = pd.read_sql('''
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM ohlcv WHERE volume > 0 AND open > 0
        ORDER BY symbol, timestamp
    ''', conn)

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df['hour'] = df['timestamp'].dt.floor('h')

    # Aggregate to hourly
    hourly = df.groupby(['symbol', 'hour']).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).reset_index()
    hourly = hourly.rename(columns={'hour': 'timestamp'})

    # Initialize strategy
    hunter = BreakoutHunter(paper_trading=True)

    # Simulate
    print("Running simulation...")

    for symbol in hourly['symbol'].unique():
        sym_data = hourly[hourly['symbol'] == symbol].sort_values('timestamp')
        hunter.update_hourly_data(symbol, sym_data)

    # Process each hour chronologically
    all_hours = sorted(hourly['timestamp'].unique())

    for hour_ts in all_hours:
        hour_data = hourly[hourly['timestamp'] == hour_ts]

        # Check for new T+0 breakouts
        for _, row in hour_data.iterrows():
            candle = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'timestamp': row['timestamp']
            }
            hunter.check_for_breakout(row['symbol'], candle)

        # Update T+1 data for pending signals (signals from 1 hour ago)
        for symbol in list(hunter.pending_signals.keys()):
            signal = hunter.pending_signals[symbol]
            if signal.status == 'pending_t1':
                hours_since = (hour_ts - signal.signal_time).total_seconds() / 3600
                if hours_since >= 1:
                    sym_row = hour_data[hour_data['symbol'] == symbol]
                    if len(sym_row) > 0:
                        hunter.update_t1_data(symbol, sym_row.iloc[0]['close'], hour_ts)

        # Check T+2 confirmations
        for symbol in list(hunter.pending_signals.keys()):
            signal = hunter.pending_signals[symbol]
            if signal.status == 'pending_t2':
                hours_since = (hour_ts - signal.signal_time).total_seconds() / 3600
                if hours_since >= 2:
                    sym_row = hour_data[hour_data['symbol'] == symbol]
                    if len(sym_row) > 0:
                        row = sym_row.iloc[0]
                        hunter.check_confirmation(symbol, row['close'], hour_ts, t2_open=row['open'])

        # Check exit conditions for active trades
        for symbol in list(hunter.active_trades.keys()):
            sym_row = hour_data[hour_data['symbol'] == symbol]
            if len(sym_row) > 0:
                row = sym_row.iloc[0]
                exit_reason = hunter.check_exit_conditions(symbol, row['close'], hour_ts)
                if exit_reason:
                    hunter.exit_trade(symbol, row['close'], hour_ts, exit_reason)

    # Print results
    hunter.print_status()

    print("\nCompleted trades:")
    for trade in hunter.completed_trades:
        print(f"  {trade.symbol}: Entry ${trade.entry_price:.6f} -> "
              f"Exit ${trade.exit_price:.6f} ({trade.exit_reason}) | "
              f"P&L: ${trade.pnl:+.0f}")

    conn.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
