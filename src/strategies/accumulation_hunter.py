"""
Accumulation Hunter Strategy v6.0 - Stealth Accumulation Detection

Detects stealth accumulation patterns 12-48 hours BEFORE major price spikes.
Based on CFG-USD analysis showing detectable signals 40+ hours before a 60% spike.

DETECTION CRITERIA (CFG Example):
| Signal              | Threshold                    | CFG Example    |
|---------------------|------------------------------|----------------|
| Volume anomaly      | >2x baseline, price flat     | 3.7x volume    |
| Order book imbalance| Bid/ask ratio >3x (watch)    | 8.5x ratio     |
| Ask depth collapse  | >50% reduction from baseline | 92% collapse   |
| Sustained pattern   | Persists 2+ hours            | 40+ hours      |

ENTRY MODE: Confirmed Breakout
1. Detect accumulation pattern early (watch list)
2. Track as it builds over hours (accumulating state)
3. Wait for actual breakout to enter:
   - Price moves >3% from accumulation range
   - Volume surges >2x baseline
4. Enter at breakout confirmation

ADVANTAGE: We're watching BEFORE the breakout, ready to enter immediately.

EXIT CRITERIA (tighter than v5.1):
- Initial stop loss: 6% from entry (vs 8%)
- Profit locks: +5%→0%, +10%→3%, +20%→8%
- Trailing stop: 6% trail after +15% gain
- Max hold: 48 hours
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AccumulationSignal:
    """Represents a detected accumulation signal"""
    symbol: str
    detection_time: datetime
    volume_ratio: float           # Current vol / baseline
    price_change_pct: float       # Should be <2% (flat)
    bid_ask_ratio: float          # Current bid depth / ask depth
    bar_multiple: float           # Current BAR / baseline BAR
    ask_collapse_pct: float       # % reduction from baseline ask depth
    status: str = 'watch'         # watch -> accumulating -> ready -> triggered -> expired
    accumulation_hours: float = 0.0
    signal_strength: float = 0.0  # 0-100 score
    price_at_detection: float = 0.0
    price_range_low: float = 0.0  # Accumulation range tracking
    price_range_high: float = 0.0
    baseline_bid_depth: float = 0.0  # For collapse tracking
    baseline_ask_depth: float = 0.0
    last_update: Optional[datetime] = None


@dataclass
class AccumulationTrade:
    """Represents an active or completed accumulation trade"""
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
    locked_profit_pct: float = -1.0  # Current locked profit level
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    # Track accumulation signal that led to this trade
    accumulation_hours: float = 0.0
    signal_strength: float = 0.0


class AccumulationHunter:
    """
    Accumulation Hunter Strategy v6.0

    Monitors for stealth accumulation patterns using order book data,
    then enters on confirmed breakout.
    """

    # Detection Parameters
    VOLUME_ANOMALY_THRESHOLD = 2.0    # 2x baseline volume
    PRICE_FLAT_MAX = 0.02             # <2% price move (flat)
    BAR_WATCH = 3.0                   # 3x BAR = watch
    BAR_STRONG = 5.0                  # 5x BAR = strong signal
    ASK_COLLAPSE_MIN = 0.50           # >50% collapse from baseline

    # Confirmation
    MIN_ACCUMULATION_HOURS = 2.0      # Must persist 2+ hours

    # Entry (confirmed breakout mode)
    BREAKOUT_PRICE_PCT = 0.03         # 3% move triggers entry
    BREAKOUT_VOLUME_RATIO = 2.0       # 2x volume surge required

    # Exit parameters (tighter than v5.1)
    INITIAL_STOP_LOSS_PCT = 0.06      # 6% initial stop loss (vs 8%)

    # Profit lock levels
    PROFIT_LOCK_LEVELS = [
        (0.05, 0.00),   # Once up 5%, lock breakeven (0%)
        (0.10, 0.03),   # Once up 10%, lock 3% profit
        (0.20, 0.08),   # Once up 20%, lock 8% profit
    ]

    TRAIL_TRIGGER_PCT = 0.15          # Activate trailing after +15%
    TRAIL_STOP_PCT = 0.06             # 6% trail (vs 8%)

    MAX_TAKE_PROFIT_PCT = 0.80        # Cap at 80% profit
    MAX_HOLD_HOURS = 48               # Max 48 hour hold

    # Position sizing (conservative)
    POSITION_SIZE_USD = 2000          # $2000 per trade (vs $2500)
    MAX_POSITIONS = 2                 # Max 2 concurrent (vs 3)
    FEE_RATE = 0.0055                 # Coinbase fee rate

    # Baseline calculation
    BASELINE_HOURS = 24               # Hours for baseline calculation

    def __init__(self, paper_trading: bool = True):
        """Initialize the accumulation hunter strategy"""
        self.paper_trading = paper_trading

        # State tracking
        self.watch_list: Dict[str, AccumulationSignal] = {}  # symbol -> signal
        self.active_trades: Dict[str, AccumulationTrade] = {}  # symbol -> trade
        self.completed_trades: List[AccumulationTrade] = []

        # Baseline caches (computed from historical data)
        self.baseline_data: Dict[str, dict] = {}  # symbol -> {bid_depth, ask_depth, bar, volume}

        logger.info(f"AccumulationHunter v6.0 initialized (paper_trading={paper_trading})")
        logger.info(f"Detection: Vol>{self.VOLUME_ANOMALY_THRESHOLD}x, Price<{self.PRICE_FLAT_MAX*100}%, "
                   f"BAR>{self.BAR_WATCH}x (watch), >{self.BAR_STRONG}x (strong)")
        logger.info(f"Entry: Breakout>{self.BREAKOUT_PRICE_PCT*100}%, Vol>{self.BREAKOUT_VOLUME_RATIO}x")
        lock_str = ", ".join([f"+{t*100:.0f}%→{l*100:.0f}%" for t, l in self.PROFIT_LOCK_LEVELS])
        logger.info(f"Exit: SL={self.INITIAL_STOP_LOSS_PCT*100:.0f}%, Locks=[{lock_str}], "
                   f"Trail={self.TRAIL_STOP_PCT*100:.0f}%@+{self.TRAIL_TRIGGER_PCT*100:.0f}%")

    def update_baseline(self, symbol: str, baseline_data: dict):
        """
        Update baseline metrics for a symbol.

        Args:
            symbol: Trading pair symbol
            baseline_data: Dict with keys:
                - avg_bid_depth: Average top-10 bid depth over baseline period
                - avg_ask_depth: Average top-10 ask depth over baseline period
                - avg_bar: Average bid/ask ratio
                - avg_volume: Average hourly volume
        """
        self.baseline_data[symbol] = baseline_data

    def compute_accumulation_metrics(self, symbol: str,
                                      current_bid_depth: float,
                                      current_ask_depth: float,
                                      current_volume: float,
                                      current_price: float,
                                      price_24h_ago: float) -> Optional[dict]:
        """
        Compute accumulation metrics for a symbol.

        Args:
            symbol: Trading pair symbol
            current_bid_depth: Sum of top-10 bid sizes
            current_ask_depth: Sum of top-10 ask sizes
            current_volume: Recent volume
            current_price: Current price
            price_24h_ago: Price 24 hours ago

        Returns:
            Dict with metrics or None if baseline not available
        """
        if symbol not in self.baseline_data:
            return None

        baseline = self.baseline_data[symbol]

        # Calculate current metrics
        current_bar = current_bid_depth / current_ask_depth if current_ask_depth > 0 else 0

        # Volume ratio vs baseline
        volume_ratio = current_volume / baseline['avg_volume'] if baseline.get('avg_volume', 0) > 0 else 0

        # Price change over 24h
        price_change_pct = (current_price - price_24h_ago) / price_24h_ago if price_24h_ago > 0 else 0

        # BAR multiple vs baseline
        bar_multiple = current_bar / baseline['avg_bar'] if baseline.get('avg_bar', 0) > 0 else 0

        # Ask depth collapse (% reduction from baseline)
        ask_collapse_pct = 1 - (current_ask_depth / baseline['avg_ask_depth']) if baseline.get('avg_ask_depth', 0) > 0 else 0

        return {
            'symbol': symbol,
            'current_bid_depth': current_bid_depth,
            'current_ask_depth': current_ask_depth,
            'current_bar': current_bar,
            'volume_ratio': volume_ratio,
            'price_change_pct': price_change_pct,
            'bar_multiple': bar_multiple,
            'ask_collapse_pct': ask_collapse_pct,
            'current_price': current_price,
        }

    def evaluate_signal_strength(self, metrics: dict) -> Tuple[float, str]:
        """
        Evaluate signal strength based on accumulation metrics.

        Returns:
            Tuple of (score 0-100, alert_level: 'none'|'watch'|'warning'|'critical')
        """
        score = 0.0

        # Volume anomaly (max 25 points)
        # 15 @ 2x, 25 @ 3.5x
        vol_ratio = metrics['volume_ratio']
        if vol_ratio >= 3.5:
            score += 25
        elif vol_ratio >= 2.0:
            score += 15 + (vol_ratio - 2.0) / 1.5 * 10
        elif vol_ratio >= 1.5:
            score += (vol_ratio - 1.5) / 0.5 * 15

        # Price flatness (max 15 points)
        # 15 @ <1%, 10 @ <2%
        price_abs = abs(metrics['price_change_pct'])
        if price_abs < 0.01:
            score += 15
        elif price_abs < 0.02:
            score += 10
        elif price_abs < 0.03:
            score += 5

        # BAR multiple (max 30 points)
        # 10 @ 3x, 20 @ 5x, 30 @ 8x
        bar_mult = metrics['bar_multiple']
        if bar_mult >= 8.0:
            score += 30
        elif bar_mult >= 5.0:
            score += 20 + (bar_mult - 5.0) / 3.0 * 10
        elif bar_mult >= 3.0:
            score += 10 + (bar_mult - 3.0) / 2.0 * 10
        elif bar_mult >= 2.0:
            score += (bar_mult - 2.0) / 1.0 * 10

        # Ask collapse (max 20 points)
        # 10 @ 30%, 15 @ 50%, 20 @ 80%
        collapse = metrics['ask_collapse_pct']
        if collapse >= 0.80:
            score += 20
        elif collapse >= 0.50:
            score += 15 + (collapse - 0.50) / 0.30 * 5
        elif collapse >= 0.30:
            score += 10 + (collapse - 0.30) / 0.20 * 5
        elif collapse >= 0.15:
            score += (collapse - 0.15) / 0.15 * 10

        # Sell absorption proxy (max 10 points)
        # High BAR + flat price suggests sells are being absorbed
        if bar_mult >= 3.0 and price_abs < 0.02:
            score += min(10, bar_mult)

        # Determine alert level
        if score >= 70:
            alert_level = 'critical'
        elif score >= 50:
            alert_level = 'warning'
        elif score >= 30:
            alert_level = 'watch'
        else:
            alert_level = 'none'

        return min(100, score), alert_level

    def check_for_accumulation(self, symbol: str, metrics: dict,
                                current_time: datetime) -> Optional[AccumulationSignal]:
        """
        Check if metrics indicate a new accumulation pattern.

        Args:
            symbol: Trading pair symbol
            metrics: Output from compute_accumulation_metrics()
            current_time: Current timestamp

        Returns:
            AccumulationSignal if pattern detected, None otherwise
        """
        # Skip if we already have this symbol or max positions
        if symbol in self.watch_list or symbol in self.active_trades:
            return None
        if len(self.active_trades) >= self.MAX_POSITIONS:
            return None

        # Evaluate signal
        score, alert_level = self.evaluate_signal_strength(metrics)

        if alert_level == 'none':
            return None

        # Check minimum criteria
        vol_ok = metrics['volume_ratio'] >= self.VOLUME_ANOMALY_THRESHOLD
        price_flat = abs(metrics['price_change_pct']) < self.PRICE_FLAT_MAX
        bar_ok = metrics['bar_multiple'] >= self.BAR_WATCH

        if not (vol_ok and price_flat and bar_ok):
            return None

        # Create signal
        signal = AccumulationSignal(
            symbol=symbol,
            detection_time=current_time,
            volume_ratio=metrics['volume_ratio'],
            price_change_pct=metrics['price_change_pct'],
            bid_ask_ratio=metrics['current_bar'],
            bar_multiple=metrics['bar_multiple'],
            ask_collapse_pct=metrics['ask_collapse_pct'],
            status='watch',
            signal_strength=score,
            price_at_detection=metrics['current_price'],
            price_range_low=metrics['current_price'] * 0.98,
            price_range_high=metrics['current_price'] * 1.02,
            baseline_bid_depth=self.baseline_data[symbol].get('avg_bid_depth', 0),
            baseline_ask_depth=self.baseline_data[symbol].get('avg_ask_depth', 0),
            last_update=current_time,
        )

        self.watch_list[symbol] = signal

        logger.info(f"ACCUMULATION DETECTED: {symbol} | Score: {score:.0f} ({alert_level})")
        logger.info(f"  Vol: {metrics['volume_ratio']:.1f}x | BAR: {metrics['bar_multiple']:.1f}x | "
                   f"AskCollapse: {metrics['ask_collapse_pct']*100:.0f}% | Price: ${metrics['current_price']:.6f}")

        return signal

    def update_accumulation_state(self, symbol: str, metrics: dict,
                                   current_time: datetime) -> Optional[str]:
        """
        Update accumulation state for a watched symbol.

        Args:
            symbol: Trading pair symbol
            metrics: Current metrics
            current_time: Current timestamp

        Returns:
            New status if changed, None otherwise
        """
        if symbol not in self.watch_list:
            return None

        signal = self.watch_list[symbol]

        # Update tracking
        hours_elapsed = (current_time - signal.detection_time).total_seconds() / 3600
        signal.accumulation_hours = hours_elapsed
        signal.last_update = current_time

        # Re-evaluate signal strength
        new_score, alert_level = self.evaluate_signal_strength(metrics)
        signal.signal_strength = new_score

        # Update metrics
        signal.volume_ratio = metrics['volume_ratio']
        signal.bar_multiple = metrics['bar_multiple']
        signal.ask_collapse_pct = metrics['ask_collapse_pct']
        signal.price_change_pct = metrics['price_change_pct']

        # Track price range
        current_price = metrics['current_price']
        signal.price_range_low = min(signal.price_range_low, current_price)
        signal.price_range_high = max(signal.price_range_high, current_price)

        old_status = signal.status

        # Check if pattern is breaking down
        if alert_level == 'none':
            # Pattern weakened - check how long
            if signal.status in ('watch', 'accumulating'):
                logger.info(f"PATTERN WEAKENED: {symbol} | Score dropped to {new_score:.0f}")
                signal.status = 'expired'
                del self.watch_list[symbol]
                return 'expired'

        # Check for status upgrade
        if signal.status == 'watch':
            if hours_elapsed >= self.MIN_ACCUMULATION_HOURS and new_score >= 50:
                signal.status = 'accumulating'
                logger.info(f"ACCUMULATION CONFIRMED: {symbol} | Hours: {hours_elapsed:.1f} | Score: {new_score:.0f}")
                return 'accumulating'

        elif signal.status == 'accumulating':
            if new_score >= 70:
                signal.status = 'ready'
                logger.info(f"BREAKOUT READY: {symbol} | Hours: {hours_elapsed:.1f} | Score: {new_score:.0f}")
                return 'ready'

        # Expire if accumulating too long without breakout (>72 hours)
        if hours_elapsed > 72 and signal.status in ('watch', 'accumulating', 'ready'):
            logger.info(f"SIGNAL EXPIRED: {symbol} | No breakout after {hours_elapsed:.0f} hours")
            signal.status = 'expired'
            del self.watch_list[symbol]
            return 'expired'

        return None

    def check_entry_conditions(self, symbol: str, current_price: float,
                                current_volume_ratio: float,
                                current_time: datetime) -> Optional[AccumulationTrade]:
        """
        Check if a watched symbol should be entered (breakout confirmed).

        Args:
            symbol: Trading pair symbol
            current_price: Current price
            current_volume_ratio: Current volume vs baseline
            current_time: Current timestamp

        Returns:
            AccumulationTrade if entry triggered, None otherwise
        """
        if symbol not in self.watch_list:
            return None

        signal = self.watch_list[symbol]

        # Need at least 'accumulating' status for entry consideration
        if signal.status not in ('accumulating', 'ready'):
            return None

        # Check breakout conditions
        price_move = (current_price - signal.price_at_detection) / signal.price_at_detection

        # Must be an upside breakout
        if price_move < self.BREAKOUT_PRICE_PCT:
            return None

        # Volume must surge
        if current_volume_ratio < self.BREAKOUT_VOLUME_RATIO:
            return None

        # BREAKOUT CONFIRMED - Enter trade
        signal.status = 'triggered'

        entry_price = current_price
        quantity = self.POSITION_SIZE_USD / entry_price
        initial_stop = entry_price * (1 - self.INITIAL_STOP_LOSS_PCT)
        take_profit = entry_price * (1 + self.MAX_TAKE_PROFIT_PCT)
        max_hold = current_time + timedelta(hours=self.MAX_HOLD_HOURS)

        trade = AccumulationTrade(
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
            current_stop=initial_stop,
            accumulation_hours=signal.accumulation_hours,
            signal_strength=signal.signal_strength,
        )

        self.active_trades[symbol] = trade
        del self.watch_list[symbol]

        logger.info(f"ENTRY TRIGGERED: {symbol}")
        logger.info(f"  Accumulated for: {signal.accumulation_hours:.1f}h | Score: {signal.signal_strength:.0f}")
        logger.info(f"  Entry: ${entry_price:.6f} | Breakout: +{price_move*100:.1f}%")
        logger.info(f"  Stop: ${initial_stop:.6f} ({self.INITIAL_STOP_LOSS_PCT*100:.0f}%)")

        return trade

    def check_exit_conditions(self, symbol: str, current_price: float,
                               current_time: datetime) -> Optional[str]:
        """
        Check if an active trade should be exited.

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

        # Determine current stop level
        stop_candidates = [trade.initial_stop_loss]

        # Add profit lock stop if active
        if trade.locked_profit_pct >= 0:
            lock_stop = trade.entry_price * (1 + trade.locked_profit_pct)
            stop_candidates.append(lock_stop)

        # Add trailing stop if active
        if trade.trailing_active:
            trail_stop = trade.highest_price * (1 - self.TRAIL_STOP_PCT)
            stop_candidates.append(trail_stop)

        trade.current_stop = max(stop_candidates)

        # Check stop loss
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
                   exit_reason: str) -> AccumulationTrade:
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

        # Handle gap past stop
        actual_exit = exit_price
        if exit_reason in ('stop_loss', 'profit_lock', 'trail_stop'):
            if exit_price < trade.current_stop:
                actual_exit = trade.current_stop * 0.98
                logger.info(f"GAP DETECTED: Price ${exit_price:.6f} below stop ${trade.current_stop:.6f}, "
                           f"simulating fill at ${actual_exit:.6f}")

        # Calculate P&L
        gross_return = (actual_exit - trade.entry_price) / trade.entry_price
        fees = self.FEE_RATE * 2
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
        logger.info(f"  Accumulated: {trade.accumulation_hours:.1f}h | P&L: ${pnl:+.0f} | {result}")

        return trade

    def get_stats(self) -> dict:
        """Get strategy statistics"""
        if not self.completed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_pnl': 0,
                'watch_list_count': len(self.watch_list),
                'active_positions': len(self.active_trades),
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
            'avg_accumulation_hours': np.mean([t.accumulation_hours for t in self.completed_trades]),
            'watch_list_count': len(self.watch_list),
            'active_positions': len(self.active_trades),
        }

    def print_status(self):
        """Print current strategy status"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("ACCUMULATION HUNTER v6.0 STATUS")
        print("=" * 60)
        print(f"Completed: {stats['total_trades']} trades | "
              f"Win Rate: {stats['win_rate']:.0f}% | "
              f"P&L: ${stats['total_pnl']:+,.0f}")

        if self.watch_list:
            print(f"\nWatch List ({len(self.watch_list)}):")
            for sym, sig in sorted(self.watch_list.items(),
                                   key=lambda x: x[1].signal_strength, reverse=True):
                print(f"  {sym}: {sig.status} | Score: {sig.signal_strength:.0f} | "
                      f"Hours: {sig.accumulation_hours:.1f} | "
                      f"BAR: {sig.bar_multiple:.1f}x")

        if self.active_trades:
            print(f"\nActive Trades ({len(self.active_trades)}):")
            for sym, trade in self.active_trades.items():
                pct = (trade.highest_price - trade.entry_price) / trade.entry_price * 100
                print(f"  {sym}: Entry ${trade.entry_price:.6f} | "
                      f"High: +{pct:.1f}% | Stop: ${trade.current_stop:.6f}")

    def save_state(self, filepath: str = 'accumulation_hunter_state.json'):
        """Save current state to JSON file for dashboard"""
        state = {
            'last_update': datetime.now().isoformat(),
            'paper_trading': self.paper_trading,
            'params': {
                'version': '6.0',
                'volume_anomaly_threshold': self.VOLUME_ANOMALY_THRESHOLD,
                'price_flat_max': self.PRICE_FLAT_MAX,
                'bar_watch': self.BAR_WATCH,
                'bar_strong': self.BAR_STRONG,
                'ask_collapse_min': self.ASK_COLLAPSE_MIN,
                'min_accumulation_hours': self.MIN_ACCUMULATION_HOURS,
                'breakout_price_pct': self.BREAKOUT_PRICE_PCT,
                'breakout_volume_ratio': self.BREAKOUT_VOLUME_RATIO,
                'initial_stop_loss_pct': self.INITIAL_STOP_LOSS_PCT,
                'profit_lock_levels': self.PROFIT_LOCK_LEVELS,
                'trail_trigger_pct': self.TRAIL_TRIGGER_PCT,
                'trail_stop_pct': self.TRAIL_STOP_PCT,
                'position_size_usd': self.POSITION_SIZE_USD,
                'max_positions': self.MAX_POSITIONS,
            },
            'watch_list': [
                {
                    'symbol': sig.symbol,
                    'detection_time': sig.detection_time.isoformat(),
                    'volume_ratio': sig.volume_ratio,
                    'bar_multiple': sig.bar_multiple,
                    'ask_collapse_pct': sig.ask_collapse_pct,
                    'status': sig.status,
                    'accumulation_hours': sig.accumulation_hours,
                    'signal_strength': sig.signal_strength,
                    'price_at_detection': sig.price_at_detection,
                    'last_update': sig.last_update.isoformat() if sig.last_update else None,
                }
                for sig in self.watch_list.values()
            ],
            'accumulating': [
                sig for sig in state.get('watch_list', [])
                if sig.get('status') in ('accumulating', 'ready')
            ] if 'watch_list' in locals() else [],
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
                    'accumulation_hours': trade.accumulation_hours,
                    'signal_strength': trade.signal_strength,
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
                    'accumulation_hours': trade.accumulation_hours,
                }
                for trade in self.completed_trades
            ],
            'stats': self.get_stats(),
        }

        # Fix accumulating list
        state['accumulating'] = [
            sig for sig in state['watch_list']
            if sig.get('status') in ('accumulating', 'ready')
        ]

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        return state


def main():
    """Test the strategy with sample data"""
    import duckdb

    print("Loading test data from DuckDB...")
    conn = duckdb.connect('/Users/bz/Pythia2/data/pythia.duckdb', read_only=True)

    # Get unique symbols
    symbols = conn.execute("SELECT DISTINCT symbol FROM order_book_snapshots").fetchdf()['symbol'].tolist()
    print(f"Found {len(symbols)} symbols")

    # Initialize strategy
    hunter = AccumulationHunter(paper_trading=True)

    # Process each symbol
    for symbol in symbols[:10]:  # Test with first 10
        # Get order book snapshots
        snapshots = conn.execute(f"""
            SELECT timestamp, bids, asks, mid_price
            FROM order_book_snapshots
            WHERE symbol = '{symbol}'
            ORDER BY timestamp DESC
            LIMIT 100
        """).fetchdf()

        if len(snapshots) < 10:
            continue

        # Calculate baseline (older data)
        import json as json_lib
        baseline_depths = []
        for _, row in snapshots.tail(50).iterrows():
            try:
                bids = json_lib.loads(row['bids']) if row['bids'] else []
                asks = json_lib.loads(row['asks']) if row['asks'] else []
                bid_depth = sum(float(b[1]) for b in bids[:10])
                ask_depth = sum(float(a[1]) for a in asks[:10])
                baseline_depths.append({
                    'bid_depth': bid_depth,
                    'ask_depth': ask_depth,
                    'bar': bid_depth / ask_depth if ask_depth > 0 else 0
                })
            except:
                continue

        if not baseline_depths:
            continue

        # Set baseline
        avg_bid = np.mean([d['bid_depth'] for d in baseline_depths])
        avg_ask = np.mean([d['ask_depth'] for d in baseline_depths])
        avg_bar = np.mean([d['bar'] for d in baseline_depths])

        hunter.update_baseline(symbol, {
            'avg_bid_depth': avg_bid,
            'avg_ask_depth': avg_ask,
            'avg_bar': avg_bar,
            'avg_volume': 1000,  # Placeholder
        })

        # Check latest data
        latest = snapshots.iloc[0]
        try:
            bids = json_lib.loads(latest['bids']) if latest['bids'] else []
            asks = json_lib.loads(latest['asks']) if latest['asks'] else []
            bid_depth = sum(float(b[1]) for b in bids[:10])
            ask_depth = sum(float(a[1]) for a in asks[:10])

            metrics = hunter.compute_accumulation_metrics(
                symbol=symbol,
                current_bid_depth=bid_depth,
                current_ask_depth=ask_depth,
                current_volume=2500,  # Simulated 2.5x volume
                current_price=latest['mid_price'],
                price_24h_ago=latest['mid_price'] * 0.99,  # Simulated flat
            )

            if metrics:
                score, level = hunter.evaluate_signal_strength(metrics)
                if level != 'none':
                    print(f"{symbol}: Score={score:.0f} ({level}) | BAR={metrics['bar_multiple']:.1f}x | Collapse={metrics['ask_collapse_pct']*100:.0f}%")

        except Exception as e:
            continue

    conn.close()
    print("\nTest complete.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
