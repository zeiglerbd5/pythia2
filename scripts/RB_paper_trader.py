#!/usr/bin/env python3
"""
Rule-Based Spike Paper Trader
Uses 5x volume + 6% price criteria to detect spikes in real-time.
Implements unified entry with adaptive exit strategy (Fast & Steep vs Slow & Large).

Strategy:
- Entry: Immediate on spike detection (RSI < 80 filter for Small Movers)
- 10-minute checkpoint: Classify spike type based on gain
  - >=15%: Fast & Steep
  - 8-15%: Slow & Large
  - 3-8%: Grace period (re-check at 20 min for slow starters)
  - <3%: Exit as underperform
- Exit: Adaptive based on spike type (Fast & Steep: 30min/2% drawdown, Slow & Large: 24hr/5% drawdown)
"""

import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import csv
import os
import numpy as np
import json
import threading
from collections import deque
try:
    import websocket
except ImportError:
    websocket = None
    logger.warning("websocket-client not installed. Order flow tracking will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('RB_paper_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure separate debug logger
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler('RB_paper_trader_debug.log')
debug_handler.setFormatter(logging.Formatter('%(asctime)s [DEBUG] %(message)s'))
debug_logger.addHandler(debug_handler)
debug_logger.propagate = False  # Don't propagate to root logger

# Configuration
CONFIG = {
    'initial_balance': 10000.0,
    'position_size_pct': 0.25,  # 25% per trade
    'max_positions': 4,
    'stop_loss_pct': 0.0075,  # 0.75%

    # Spike detection thresholds
    # Multi-layer defense: Volume + Price + RSI + OFI + 10-min checkpoint
    # Lower thresholds allow more candidates to reach OFI filter (final gatekeeper)
    'volume_spike_threshold': 1.5,  # 1.5x (catch spikes earlier, before volume peaks - OFI is final gate)
    'price_spike_threshold': 0.06,  # 6% in 3-min window (or 10-min window if higher)
    'rsi_overbought': 80,  # Filter Small Movers (bypassed for price moves >15%)
    'rsi_exemption_threshold': 15.0,  # If price gain >15%, ignore RSI filter (legitimate spike, not pump-and-dump)

    # 10-minute checkpoint thresholds
    'fast_steep_gain_threshold': 15.0,  # >15% = Fast & Steep
    'slow_large_gain_min': 8.0,  # 8-15% = Slow & Large
    'grace_period_min_gain': 3.0,  # Minimum gain to qualify for grace period (avoid true losers)
    'grace_period_duration_min': 10,  # Additional time to give slow starters

    # Exit thresholds
    'fast_steep_time_limit_min': 30,
    'fast_steep_drawdown_pct': 0.02,  # 2%
    'slow_large_time_limit_min': 1440,  # 24 hours
    'slow_large_drawdown_pct': 0.05,  # 5%

    # Rate limiting
    'api_delay_seconds': 0.04,  # Tuned for ~59 second cycles
    'cycle_duration_seconds': 60,

    # Symbol filtering
    'max_price_per_coin': 500.0,  # Skip expensive coins (BTC, ETH) - low % gains
    'min_volume_usd': 50000,  # Skip low-volume symbols
    'excluded_symbols': [
        # Stablecoins
        'USDT-USD', 'USDC-USD', 'DAI-USD', 'PYUSD-USD', 'TUSD-USD', 'GUSD-USD',
        'USDP-USD', 'BUSD-USD', 'FRAX-USD', 'LUSD-USD', 'USDD-USD',
        # Wrapped/derivative tokens (won't spike independently)
        'WBTC-USD', 'WETH-USD', 'STETH-USD'
    ],

    # Order Flow Imbalance (OFI) filtering
    'ofi_enabled': True,  # Enable order flow tracking
    'ofi_threshold': 0.3,  # Minimum buy pressure (-1 to 1, positive = net buying)
    'ofi_min_trades': 10,  # Minimum trades in window for reliable OFI
    'ofi_window_seconds': 60,  # Rolling window size (1 minute)

    # Debug logging
    'debug_mode': True,  # Enable detailed debug logging
    'log_near_misses': True,  # Log symbols that almost triggered (4/5 criteria)
    'log_all_scans': False,  # Log every symbol scanned (VERY VERBOSE - use sparingly)
    'near_miss_volume_threshold': 3.0,  # Log if volume is at least 3x (even if < 5x)
    'near_miss_price_threshold': 0.04,  # Log if price is at least 4% (even if < 6%)
}


class CoinbaseAPI:
    """Coinbase Public API client for fetching market data."""

    BASE_URL = "https://api.exchange.coinbase.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RB-Spike-Trader/1.0'
        })

    def get_all_products(self) -> List[Dict]:
        """Fetch all USD trading pairs."""
        try:
            response = self.session.get(f"{self.BASE_URL}/products")
            response.raise_for_status()
            products = response.json()

            # Filter to USD pairs only
            usd_products = [
                p for p in products
                if p.get('quote_currency') == 'USD' and p.get('status') == 'online'
            ]

            logger.info(f"Fetched {len(usd_products)} USD trading pairs")
            return usd_products

        except Exception as e:
            logger.error(f"Error fetching products: {e}")
            return []

    def get_ticker(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            response = self.session.get(f"{self.BASE_URL}/products/{symbol}/ticker")
            response.raise_for_status()
            data = response.json()
            return float(data.get('price', 0))

        except Exception as e:
            logger.debug(f"Error fetching ticker for {symbol}: {e}")
            return None

    def get_candles(self, symbol: str, granularity: int = 60, limit: int = 15) -> List[Dict]:
        """
        Get historical candles for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            granularity: Candle size in seconds (60 = 1 minute)
            limit: Number of candles to fetch

        Returns:
            List of candles with keys: timestamp, low, high, open, close, volume
        """
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(seconds=granularity * limit)

            params = {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'granularity': granularity
            }

            response = self.session.get(f"{self.BASE_URL}/products/{symbol}/candles", params=params)
            response.raise_for_status()
            data = response.json()

            # Convert to dict format: [timestamp, low, high, open, close, volume]
            candles = []
            for candle in data:
                candles.append({
                    'timestamp': candle[0],
                    'low': float(candle[1]),
                    'high': float(candle[2]),
                    'open': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })

            # Sort by timestamp (oldest first)
            candles.sort(key=lambda x: x['timestamp'])
            return candles

        except Exception as e:
            logger.debug(f"Error fetching candles for {symbol}: {e}")
            return []

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get recent trades for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            limit: Number of trades to fetch (max 100)

        Returns:
            List of trades with keys: time, trade_id, price, size, side
        """
        try:
            params = {'limit': min(limit, 100)}
            response = self.session.get(f"{self.BASE_URL}/products/{symbol}/trades", params=params)
            response.raise_for_status()
            trades = response.json()

            # Convert to our format
            result = []
            for trade in trades:
                result.append({
                    'time': trade['time'],
                    'trade_id': trade['trade_id'],
                    'price': float(trade['price']),
                    'size': float(trade['size']),
                    'side': trade['side']  # 'buy' or 'sell'
                })

            return result

        except Exception as e:
            logger.debug(f"Error fetching trades for {symbol}: {e}")
            return []


@dataclass
class Trade:
    """Represents a single trade for order flow tracking."""
    timestamp: datetime
    size: float
    side: str  # 'buy' or 'sell'


class OrderFlowTracker:
    """
    Tracks real-time order flow imbalance from trade data.

    Maintains a rolling 1-minute window of trades to calculate:
    - Buy/sell volume split
    - Order flow imbalance (OFI): (buy_vol - sell_vol) / total_vol
    - Trade count for reliability assessment
    """

    def __init__(self, symbol: str, window_seconds: int = 60):
        """
        Initialize order flow tracker for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            window_seconds: Rolling window size in seconds (default: 60)
        """
        self.symbol = symbol
        self.window_seconds = window_seconds
        self.trades: deque = deque()  # Rolling window of trades
        self.lock = threading.Lock()  # Thread-safe for WebSocket updates

    def add_trade(self, size: float, side: str, timestamp: Optional[datetime] = None):
        """
        Add a trade to the tracker.

        Args:
            size: Trade size (volume)
            side: 'buy' or 'sell' (aggressor side)
            timestamp: Trade timestamp (defaults to now)
        """
        if timestamp is None:
            from datetime import timezone
            timestamp = datetime.now(timezone.utc)

        trade = Trade(timestamp=timestamp, size=size, side=side.lower())

        with self.lock:
            self.trades.append(trade)
            self._expire_old_trades()

    def _expire_old_trades(self):
        """Remove trades older than the window size."""
        from datetime import timezone
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.window_seconds)

        while self.trades and self.trades[0].timestamp < cutoff:
            self.trades.popleft()

    def get_order_flow_imbalance(self) -> Tuple[float, int, float, float]:
        """
        Calculate current order flow imbalance.

        Returns:
            Tuple of (ofi, trade_count, buy_volume, sell_volume)
            - ofi: Order flow imbalance [-1, 1] (positive = net buying)
            - trade_count: Number of trades in window
            - buy_volume: Total buy volume
            - sell_volume: Total sell volume
        """
        with self.lock:
            self._expire_old_trades()

            if not self.trades:
                return 0.0, 0, 0.0, 0.0

            buy_volume = sum(t.size for t in self.trades if t.side == 'buy')
            sell_volume = sum(t.size for t in self.trades if t.side == 'sell')
            total_volume = buy_volume + sell_volume
            trade_count = len(self.trades)

            if total_volume == 0:
                return 0.0, trade_count, buy_volume, sell_volume

            # OFI = (buy - sell) / total
            # Range: [-1, 1]  where +1 = all buying, -1 = all selling
            ofi = (buy_volume - sell_volume) / total_volume

            return ofi, trade_count, buy_volume, sell_volume

    def reset(self):
        """Clear all tracked trades."""
        with self.lock:
            self.trades.clear()


class RuleBasedSpikeDetector:
    """Detects spikes using rule-based criteria: 5x volume + 6% price."""

    def __init__(self, config: Dict):
        self.config = config

    def calculate_rsi(self, candles: List[Dict], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)."""
        if len(candles) < period + 1:
            return 50.0  # Neutral default

        closes = [c['close'] for c in candles[-(period+1):]]

        gains = []
        losses = []

        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def detect_spike(
        self,
        candles: List[Dict],
        order_flow_tracker: Optional[OrderFlowTracker] = None,
        symbol: str = None
    ) -> Tuple[bool, float, float, float, float, int]:
        """
        Detect if current candle shows a spike.

        Args:
            candles: List of recent candles
            order_flow_tracker: Optional OrderFlowTracker for OFI filtering
            symbol: Symbol name for logging (optional)

        Returns:
            (is_spike, volume_ratio, price_gain_pct, rsi, ofi, trade_count)
        """
        if len(candles) < 15:
            return (False, 0.0, 0.0, 50.0, 0.0, 0)

        # Get current and recent candles
        current_candle = candles[-1]
        last_3_candles = candles[-3:]
        last_10_candles = candles[-11:-1]  # Exclude current for average

        # Calculate volume spike (3-min cumulative vs 3-min rolling baseline)
        # Use sum of last 3 minutes vs average of older 3-min periods
        # Better detects sustained volume build-up vs single-candle spikes
        recent_3min_volume = sum(c['volume'] for c in last_3_candles)

        # Calculate rolling 3-min baseline from older candles (excludes last 3)
        if len(last_10_candles) >= 6:
            # Average of three 3-minute periods from candles 4-12 (older data)
            baseline_periods = [
                sum(c['volume'] for c in last_10_candles[0:3]),  # Oldest 3 min
                sum(c['volume'] for c in last_10_candles[3:6]),  # Middle 3 min
                sum(c['volume'] for c in last_10_candles[6:9]) if len(last_10_candles) >= 9 else 0,  # Newer 3 min
            ]
            avg_3min_volume = np.mean([p for p in baseline_periods if p > 0])
        else:
            # Fallback: use mean of individual candles * 3
            avg_3min_volume = np.mean([c['volume'] for c in last_10_candles]) * 3

        if avg_3min_volume == 0:
            volume_ratio = 0.0
        else:
            volume_ratio = recent_3min_volume / avg_3min_volume

        # Calculate price spike - check both 3-min and 10-min windows, use higher
        # 3-minute window (fast spikes)
        first_close_3min = last_3_candles[0]['close']
        max_high_3min = max(c['high'] for c in last_3_candles)

        if first_close_3min == 0:
            price_gain_3min = 0.0
        else:
            price_gain_3min = ((max_high_3min / first_close_3min) - 1) * 100

        # 10-minute window (gradual spikes)
        price_gain_10min = 0.0
        if len(candles) >= 11:
            first_close_10min = candles[-11]['close']
            current_close = current_candle['close']
            if first_close_10min > 0:
                price_gain_10min = ((current_close / first_close_10min) - 1) * 100

        # Use whichever window shows larger gain
        price_gain_pct = max(price_gain_3min, price_gain_10min)

        # Calculate RSI
        rsi = self.calculate_rsi(candles)

        # Get order flow imbalance (if tracker available)
        ofi = 0.0
        trade_count = 0
        ofi_passed = True  # Default to passing if OFI disabled

        if order_flow_tracker is not None and self.config['ofi_enabled']:
            ofi, trade_count, buy_vol, sell_vol = order_flow_tracker.get_order_flow_imbalance()

            # Only apply OFI filter if we have sufficient data
            if trade_count >= self.config['ofi_min_trades']:
                ofi_passed = ofi >= self.config['ofi_threshold']
            else:
                # Insufficient data - skip OFI filter for this symbol
                ofi_passed = True

        # Check spike criteria
        volume_spike = volume_ratio >= self.config['volume_spike_threshold']
        price_spike = (price_gain_pct / 100) >= self.config['price_spike_threshold']
        not_overbought = rsi < self.config['rsi_overbought']

        # RSI exemption: Large price moves (>15%) bypass RSI filter
        # Rationale: RSI filter is for "small movers" (8-10% pumps that dump quickly).
        # A genuine 20-28% spike is not a pump-and-dump, even if RSI is high.
        large_price_move = price_gain_pct >= self.config.get('rsi_exemption_threshold', 15.0)
        rsi_check_passed = not_overbought or large_price_move

        is_spike = volume_spike and price_spike and rsi_check_passed and ofi_passed

        # Debug logging
        if self.config.get('debug_mode', False) and symbol:
            # Log all scans if enabled
            if self.config.get('log_all_scans', False):
                debug_logger.debug(
                    f"{symbol}: Vol={volume_ratio:.1f}x Price={price_gain_pct:.1f}% "
                    f"RSI={rsi:.1f} OFI={ofi:+.2f}({trade_count}t) "
                    f"[{'PASS' if is_spike else 'FAIL'}]"
                )

            # Log near misses (symbols that came close)
            if self.config.get('log_near_misses', False):
                criteria_met = sum([volume_spike, price_spike, rsi_check_passed, ofi_passed])
                is_near_miss = (criteria_met >= 3 or
                               volume_ratio >= self.config.get('near_miss_volume_threshold', 3.0) or
                               (price_gain_pct / 100) >= self.config.get('near_miss_price_threshold', 0.04))

                if is_near_miss and not is_spike:
                    failed_criteria = []
                    if not volume_spike:
                        failed_criteria.append(f"Vol:{volume_ratio:.1f}x<{self.config['volume_spike_threshold']}x")
                    if not price_spike:
                        failed_criteria.append(f"Price:{price_gain_pct:.1f}%<{self.config['price_spike_threshold']*100}%")
                    if not rsi_check_passed:
                        if large_price_move:
                            failed_criteria.append(f"RSI:{rsi:.1f}>{self.config['rsi_overbought']} (exempt at {price_gain_pct:.1f}%)")
                        else:
                            failed_criteria.append(f"RSI:{rsi:.1f}>{self.config['rsi_overbought']}")
                    if not ofi_passed:
                        failed_criteria.append(f"OFI:{ofi:+.2f}<{self.config['ofi_threshold']}")

                    rsi_status = f"RSI={rsi:.1f}" + (" [EXEMPT]" if large_price_move else "")
                    debug_logger.debug(
                        f"NEAR MISS - {symbol}: {criteria_met}/4 criteria met | "
                        f"Failed: {', '.join(failed_criteria)} | "
                        f"Vol={volume_ratio:.1f}x Price={price_gain_pct:.1f}% {rsi_status} OFI={ofi:+.2f}({trade_count}t)"
                    )

        if volume_spike and price_spike and rsi_check_passed:
            rsi_note = f"RSI: {rsi:.1f}" + (" [EXEMPT]" if large_price_move else "")
            if ofi_passed:
                logger.info(
                    f"SPIKE DETECTED - Volume: {volume_ratio:.1f}x, Price: {price_gain_pct:.1f}%, "
                    f"{rsi_note}, OFI: {ofi:+.2f} ({trade_count} trades)"
                )
            else:
                logger.info(
                    f"SPIKE FILTERED (Low OFI) - Volume: {volume_ratio:.1f}x, Price: {price_gain_pct:.1f}%, "
                    f"{rsi_note}, OFI: {ofi:+.2f} ({trade_count} trades) - Need >{self.config['ofi_threshold']}"
                )

        return (is_spike, volume_ratio, price_gain_pct, rsi, ofi, trade_count)


@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    entry_price: float
    entry_time: datetime
    shares: float
    spike_data: Dict

    spike_type: Optional[str] = None  # 'fast_steep', 'slow_large', or 'underperform'
    peak_price: float = 0.0
    checkpoint_passed: bool = False
    grace_period_granted: bool = False  # Track if we gave slow starter more time

    def __post_init__(self):
        self.peak_price = self.entry_price
        self.stop_loss_price = self.entry_price * (1 - CONFIG['stop_loss_pct'])

    def get_hold_time_minutes(self) -> float:
        """Get minutes since entry."""
        return (datetime.now() - self.entry_time).total_seconds() / 60

    def classify_spike_type(self, current_price: float) -> str:
        """Classify spike type at 10-minute checkpoint based on gain."""
        current_gain_pct = ((current_price / self.entry_price) - 1) * 100

        if current_gain_pct > CONFIG['fast_steep_gain_threshold']:
            return 'fast_steep'
        elif current_gain_pct >= CONFIG['slow_large_gain_min']:
            return 'slow_large'
        else:
            return 'underperform'

    def should_exit(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Returns:
            (should_exit, reason)
        """
        hold_time = self.get_hold_time_minutes()

        # Update peak price
        if current_price > self.peak_price:
            self.peak_price = current_price

        # Stop loss (always active)
        if current_price <= self.stop_loss_price:
            return (True, 'stop_loss')

        # Phase 1: Before 10-minute checkpoint
        if hold_time < 10:
            return (False, '')

        # Phase 2: At 10-minute checkpoint (classify spike type)
        if not self.checkpoint_passed:
            self.checkpoint_passed = True
            current_gain_pct = ((current_price / self.entry_price) - 1) * 100

            # Classify spike type
            if current_gain_pct >= CONFIG['fast_steep_gain_threshold']:
                self.spike_type = 'fast_steep'
                logger.info(f"{self.symbol} 10-min checkpoint: Fast & Steep (gain: {current_gain_pct:.1f}%)")

            elif current_gain_pct >= CONFIG['slow_large_gain_min']:
                self.spike_type = 'slow_large'
                logger.info(f"{self.symbol} 10-min checkpoint: Slow & Large (gain: {current_gain_pct:.1f}%)")

            elif current_gain_pct >= CONFIG['grace_period_min_gain']:
                # Slow starter - grant grace period
                self.grace_period_granted = True
                self.spike_type = 'slow_large'  # Tentatively classify as slow_large
                logger.info(f"{self.symbol} 10-min checkpoint: GRACE PERIOD granted (gain: {current_gain_pct:.1f}%, "
                          f"will re-check at 20 min)")

            else:
                # True underperformer
                self.spike_type = 'underperform'
                logger.info(f"{self.symbol} 10-min checkpoint: Underperform (gain: {current_gain_pct:.1f}%)")
                return (True, '10min_checkpoint_underperform')

        # Phase 2.5: Grace period final checkpoint at 20 minutes
        if self.grace_period_granted and hold_time >= 20:
            # Re-evaluate - this only runs once
            self.grace_period_granted = False  # Don't check again
            current_gain_pct = ((current_price / self.entry_price) - 1) * 100

            if current_gain_pct < CONFIG['slow_large_gain_min']:
                # Still underperforming after grace period
                logger.info(f"{self.symbol} 20-min grace period expired: Still underperform (gain: {current_gain_pct:.1f}%)")
                return (True, '20min_grace_period_underperform')
            else:
                # Accelerated! Continue as Slow & Large
                logger.info(f"{self.symbol} 20-min grace period: Accelerated to {current_gain_pct:.1f}%, continuing as Slow & Large")

        # Phase 3: Post-checkpoint adaptive exits
        if self.spike_type == 'fast_steep':
            # Time limit: 30 minutes
            if hold_time > CONFIG['fast_steep_time_limit_min']:
                return (True, 'fast_steep_time_limit')

            # Drawdown from peak: 2%
            drawdown_pct = (self.peak_price - current_price) / self.peak_price
            if drawdown_pct > CONFIG['fast_steep_drawdown_pct']:
                return (True, 'fast_steep_drawdown')

        elif self.spike_type == 'slow_large':
            # Time limit: 24 hours
            if hold_time > CONFIG['slow_large_time_limit_min']:
                return (True, 'slow_large_time_limit')

            # Drawdown from peak: 5%
            drawdown_pct = (self.peak_price - current_price) / self.peak_price
            if drawdown_pct > CONFIG['slow_large_drawdown_pct']:
                return (True, 'slow_large_drawdown')

        return (False, '')


class PaperPortfolio:
    """Manages paper trading portfolio with $10k starting capital."""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict] = []
        self.position_size_pct = CONFIG['position_size_pct']

        # Load existing trades if file exists
        self.trade_log_path = 'RB_paper_trades.csv'
        if os.path.exists(self.trade_log_path):
            logger.info(f"Loading existing trades from {self.trade_log_path}")
            self._load_trades()

    def _load_trades(self):
        """Load existing trades from CSV."""
        try:
            with open(self.trade_log_path, 'r') as f:
                reader = csv.DictReader(f)
                self.trades = list(reader)
            logger.info(f"Loaded {len(self.trades)} existing trades")
        except Exception as e:
            logger.warning(f"Could not load existing trades: {e}")

    def _save_trade(self, trade: Dict):
        """Append trade to CSV log."""
        file_exists = os.path.exists(self.trade_log_path)

        with open(self.trade_log_path, 'a', newline='') as f:
            fieldnames = [
                'timestamp', 'symbol', 'side', 'shares', 'price', 'value',
                'spike_type', 'reason', 'pnl', 'pnl_pct', 'hold_time_min',
                'rsi', 'volume_ratio', 'price_gain', 'ofi', 'trade_count_1m',
                'cash', 'portfolio_value'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(trade)

    def can_enter_position(self) -> bool:
        """Check if we can enter a new position."""
        return len(self.positions) < CONFIG['max_positions'] and self.cash > 100

    def enter_position(self, symbol: str, price: float, spike_data: Dict) -> bool:
        """
        Enter a new position.

        Args:
            symbol: Trading pair
            price: Entry price
            spike_data: Dict with volume_ratio, price_gain, rsi

        Returns:
            True if position entered successfully
        """
        if not self.can_enter_position():
            logger.warning(f"Cannot enter {symbol}: max positions or insufficient cash")
            return False

        # Calculate position size (25% of cash)
        position_value = self.cash * self.position_size_pct
        shares = position_value / price

        # Create position
        position = Position(
            symbol=symbol,
            entry_price=price,
            entry_time=datetime.now(),
            shares=shares,
            spike_data=spike_data
        )

        self.positions[symbol] = position
        self.cash -= position_value

        # Log trade
        trade = {
            'timestamp': position.entry_time.isoformat(),
            'symbol': symbol,
            'side': 'BUY',
            'shares': shares,
            'price': price,
            'value': position_value,
            'spike_type': None,
            'reason': 'entry',
            'pnl': 0,
            'pnl_pct': 0,
            'hold_time_min': 0,
            'rsi': spike_data.get('rsi', 0),
            'volume_ratio': spike_data.get('volume_ratio', 0),
            'price_gain': spike_data.get('price_gain', 0),
            'ofi': spike_data.get('ofi', 0),
            'trade_count_1m': spike_data.get('trade_count_1m', 0),
            'cash': self.cash,
            'portfolio_value': self.get_portfolio_value()
        }
        self.trades.append(trade)
        self._save_trade(trade)

        logger.info(f"ENTERED {symbol} @ ${price:.2f} | Shares: {shares:.4f} | "
                   f"Value: ${position_value:.2f} | Portfolio: ${self.get_portfolio_value():.2f}")

        return True

    def exit_position(self, symbol: str, price: float, reason: str) -> bool:
        """
        Exit a position.

        Args:
            symbol: Trading pair
            price: Exit price
            reason: Exit reason

        Returns:
            True if position exited successfully
        """
        if symbol not in self.positions:
            logger.warning(f"Cannot exit {symbol}: position not found")
            return False

        position = self.positions[symbol]

        # Calculate P&L
        exit_value = position.shares * price
        entry_value = position.shares * position.entry_price
        pnl = exit_value - entry_value
        pnl_pct = (pnl / entry_value) * 100
        hold_time = position.get_hold_time_minutes()

        # Update cash
        self.cash += exit_value

        # Log trade
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': 'SELL',
            'shares': position.shares,
            'price': price,
            'value': exit_value,
            'spike_type': position.spike_type,
            'reason': reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hold_time_min': hold_time,
            'rsi': position.spike_data.get('rsi', 0),
            'volume_ratio': position.spike_data.get('volume_ratio', 0),
            'price_gain': position.spike_data.get('price_gain', 0),
            'ofi': position.spike_data.get('ofi', 0),
            'trade_count_1m': position.spike_data.get('trade_count_1m', 0),
            'cash': self.cash,
            'portfolio_value': self.get_portfolio_value()
        }
        self.trades.append(trade)
        self._save_trade(trade)

        # Remove position
        del self.positions[symbol]

        logger.info(f"EXITED {symbol} @ ${price:.2f} | Reason: {reason} | "
                   f"P&L: ${pnl:.2f} ({pnl_pct:+.1f}%) | Hold: {hold_time:.1f}min | "
                   f"Portfolio: ${self.get_portfolio_value():.2f}")

        return True

    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get current value of a position."""
        if symbol not in self.positions:
            return 0.0
        return self.positions[symbol].shares * current_price

    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions at last known prices)."""
        return self.cash + sum(
            pos.shares * pos.peak_price for pos in self.positions.values()
        )

    def get_summary(self) -> Dict:
        """Get portfolio summary statistics."""
        total_value = self.get_portfolio_value()
        total_return = ((total_value / self.initial_balance) - 1) * 100

        # Calculate trade statistics
        closed_trades = [t for t in self.trades if t.get('side') == 'SELL']
        if closed_trades:
            wins = [t for t in closed_trades if float(t.get('pnl', 0)) > 0]
            win_rate = (len(wins) / len(closed_trades)) * 100
            avg_pnl_pct = np.mean([float(t.get('pnl_pct', 0)) for t in closed_trades])
        else:
            win_rate = 0
            avg_pnl_pct = 0

        return {
            'total_value': total_value,
            'cash': self.cash,
            'total_return_pct': total_return,
            'position_count': len(self.positions),
            'trade_count': len(closed_trades),
            'win_rate': win_rate,
            'avg_return_pct': avg_pnl_pct
        }


class RBPaperTrader:
    """Main paper trading bot."""

    def __init__(self):
        self.api = CoinbaseAPI()
        self.detector = RuleBasedSpikeDetector(CONFIG)
        self.portfolio = PaperPortfolio(CONFIG['initial_balance'])
        self.symbol_list = []
        self.cycle_count = 0
        self.symbol_batch_size = 50  # Scan 50 symbols per cycle
        self.symbol_batch_index = 0  # Current batch starting index

        # Debug tracking
        self.scan_stats = {
            'symbols_scanned': 0,
            'spike_candidates': 0,
            'near_misses': 0,
            'entries': 0
        }

    def initialize(self):
        """Initialize trader by fetching available symbols."""
        logger.info("Initializing RB Paper Trader...")
        logger.info(f"Starting balance: ${CONFIG['initial_balance']:.2f}")
        logger.info(f"Position sizing: {CONFIG['position_size_pct']*100:.0f}% per trade")
        logger.info(f"Max positions: {CONFIG['max_positions']}")

        # Fetch all products
        products = self.api.get_all_products()
        logger.info(f"Fetched {len(products)} total products")

        # Filter symbols by exclusion list
        filtered_by_exclusion = [
            p for p in products
            if p['id'] not in CONFIG['excluded_symbols']
        ]
        logger.info(f"After exclusion filter: {len(filtered_by_exclusion)} symbols")

        # Filter by price (fetch current prices and remove >$500 coins)
        symbols_with_prices = []
        logger.info("Filtering by price (removing coins >$500)...")

        for product in filtered_by_exclusion:
            symbol = product['id']
            time.sleep(0.02)  # Small delay to avoid rate limit during init

            price = self.api.get_ticker(symbol)
            if price and price <= CONFIG['max_price_per_coin']:
                symbols_with_prices.append(symbol)

        self.symbol_list = symbols_with_prices

        logger.info(f"After price filter: {len(self.symbol_list)} symbols")
        logger.info(f"Monitoring {len(self.symbol_list)} symbols")
        logger.info(f"Detection criteria: {CONFIG['volume_spike_threshold']}x volume + "
                   f"{CONFIG['price_spike_threshold']*100:.0f}% price + RSI < {CONFIG['rsi_overbought']}")
        logger.info(f"Price filter: coins <${CONFIG['max_price_per_coin']:.0f}")

    def check_position_exits(self):
        """Check all open positions for exit conditions."""
        for symbol in list(self.portfolio.positions.keys()):
            position = self.portfolio.positions[symbol]

            # Get current price
            current_price = self.api.get_ticker(symbol)
            if current_price is None:
                logger.warning(f"Could not get price for {symbol}, skipping exit check")
                continue

            # Check exit conditions
            should_exit, reason = position.should_exit(current_price)

            if should_exit:
                self.portfolio.exit_position(symbol, current_price, reason)

            time.sleep(CONFIG['api_delay_seconds'])

    def scan_for_entries(self):
        """Scan symbols for new entry opportunities."""
        if not self.portfolio.can_enter_position():
            return

        # Debug: Log scan cycle start
        if CONFIG.get('debug_mode', False):
            scan_start_time = datetime.now()
            cycle_scan_count = 0
            debug_logger.debug(f"=" * 80)
            debug_logger.debug(f"SCAN CYCLE START - Cycle {self.cycle_count}")
            debug_logger.debug(f"Scanning {len(self.symbol_list)} symbols for entry opportunities")
            debug_logger.debug(f"=" * 80)

        for symbol in self.symbol_list:
            # Skip if already in position
            if symbol in self.portfolio.positions:
                continue

            # Rate limiting
            time.sleep(CONFIG['api_delay_seconds'])

            # Fetch candles
            candles = self.api.get_candles(symbol, granularity=60, limit=15)
            if len(candles) < 15:
                continue

            # Debug: Track symbol scanned
            if CONFIG.get('debug_mode', False):
                cycle_scan_count += 1
                self.scan_stats['symbols_scanned'] += 1

            # First pass: Check spike without OFI (fast)
            is_spike_candidate, volume_ratio, price_gain, rsi, _, _ = self.detector.detect_spike(
                candles,
                order_flow_tracker=None,  # Skip OFI on first pass
                symbol=symbol
            )

            # Debug: Track spike candidates
            if CONFIG.get('debug_mode', False) and is_spike_candidate:
                self.scan_stats['spike_candidates'] += 1

            # Second pass: Only fetch trades if spike looks promising
            order_flow_tracker = None
            ofi = 0.0
            trade_count = 0

            if is_spike_candidate and CONFIG['ofi_enabled']:
                # Rate limit before trade fetch
                time.sleep(CONFIG['api_delay_seconds'])

                recent_trades = self.api.get_recent_trades(symbol, limit=100)
                if recent_trades:
                    # Create temporary tracker and populate with recent trades
                    order_flow_tracker = OrderFlowTracker(
                        symbol=symbol,
                        window_seconds=CONFIG['ofi_window_seconds']
                    )

                    # Add trades from last minute only
                    from datetime import timezone
                    cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=CONFIG['ofi_window_seconds'])
                    for trade in recent_trades:
                        # Parse ISO timestamp
                        trade_time = datetime.fromisoformat(trade['time'].replace('Z', '+00:00'))
                        if trade_time >= cutoff_time:
                            order_flow_tracker.add_trade(
                                size=trade['size'],
                                side=trade['side'],
                                timestamp=trade_time
                            )

                    # Get OFI values
                    ofi, trade_count, _, _ = order_flow_tracker.get_order_flow_imbalance()

            # Final spike decision with OFI filter
            is_spike, _, _, _, ofi, trade_count = self.detector.detect_spike(
                candles,
                order_flow_tracker=order_flow_tracker,
                symbol=symbol
            )

            if is_spike:
                # Get current price for entry
                current_price = self.api.get_ticker(symbol)
                if current_price is None:
                    logger.warning(f"Could not get price for {symbol}, skipping entry")
                    continue

                # MOMENTUM CHECK: Ensure spike is still active
                last_candle = candles[-1]
                last_close = last_candle['close']

                # Check 1: Current price should be near or above last close (within 2%)
                price_vs_close = ((current_price / last_close) - 1) * 100

                # Check 2: Last candle should be green (bullish)
                last_candle_green = last_candle['close'] > last_candle['open']

                # Check 3: Current price should be above the spike detection threshold
                # (not already fallen back below the initial spike level)
                spike_base_price = candles[-3]['close']  # 3 candles ago
                current_gain_from_base = ((current_price / spike_base_price) - 1) * 100

                if price_vs_close < -2.0:
                    logger.info(f"Skipping {symbol} - momentum fading (price {price_vs_close:+.1f}% vs last close)")
                    continue

                if not last_candle_green:
                    logger.info(f"Skipping {symbol} - last candle bearish (reversal detected)")
                    continue

                if current_gain_from_base < 3.0:  # Should still be at least 3% up from base
                    logger.info(f"Skipping {symbol} - retraced too much (only {current_gain_from_base:.1f}% from base)")
                    continue

                # Passed all momentum checks - enter position
                logger.info(f"Momentum confirmed for {symbol}: price vs close {price_vs_close:+.1f}%, "
                           f"gain from base {current_gain_from_base:.1f}%, last candle green")

                spike_data = {
                    'volume_ratio': volume_ratio,
                    'price_gain': price_gain,
                    'rsi': rsi,
                    'ofi': ofi,
                    'trade_count_1m': trade_count
                }

                success = self.portfolio.enter_position(symbol, current_price, spike_data)

                if success:
                    # Debug: Track entry
                    if CONFIG.get('debug_mode', False):
                        self.scan_stats['entries'] += 1

                    # Only enter one position per cycle
                    break

        # Debug: Log scan cycle end
        if CONFIG.get('debug_mode', False):
            scan_duration = (datetime.now() - scan_start_time).total_seconds()
            debug_logger.debug(f"=" * 80)
            debug_logger.debug(f"SCAN CYCLE END - Cycle {self.cycle_count}")
            debug_logger.debug(f"Duration: {scan_duration:.1f}s | Symbols scanned: {cycle_scan_count}")
            debug_logger.debug(f"Spike candidates: {self.scan_stats['spike_candidates']} | "
                             f"Entries: {self.scan_stats['entries']}")
            debug_logger.debug(f"=" * 80)

    def log_portfolio_status(self):
        """Log current portfolio status."""
        summary = self.portfolio.get_summary()

        logger.info("=" * 80)
        logger.info(f"PORTFOLIO STATUS - Cycle {self.cycle_count}")
        logger.info(f"Total Value: ${summary['total_value']:.2f} ({summary['total_return_pct']:+.2f}%)")
        logger.info(f"Cash: ${summary['cash']:.2f}")
        logger.info(f"Open Positions: {summary['position_count']}/{CONFIG['max_positions']}")
        logger.info(f"Closed Trades: {summary['trade_count']} | "
                   f"Win Rate: {summary['win_rate']:.1f}% | "
                   f"Avg Return: {summary['avg_return_pct']:+.2f}%")

        # Log open positions
        for symbol, position in self.portfolio.positions.items():
            hold_time = position.get_hold_time_minutes()
            unrealized_pnl_pct = ((position.peak_price / position.entry_price) - 1) * 100
            logger.info(f"  {symbol}: ${position.entry_price:.2f} -> ${position.peak_price:.2f} "
                       f"({unrealized_pnl_pct:+.1f}%) | Hold: {hold_time:.1f}min | "
                       f"Type: {position.spike_type or 'pending'}")

        logger.info("=" * 80)

    def log_debug_summary(self):
        """Log debug scan statistics summary."""
        if not CONFIG.get('debug_mode', False):
            return

        debug_logger.debug("")
        debug_logger.debug("=" * 80)
        debug_logger.debug(f"DEBUG SUMMARY - Cycles 1-{self.cycle_count}")
        debug_logger.debug("=" * 80)
        debug_logger.debug(f"Total symbols scanned: {self.scan_stats['symbols_scanned']:,}")
        debug_logger.debug(f"Spike candidates found: {self.scan_stats['spike_candidates']}")
        debug_logger.debug(f"Positions entered: {self.scan_stats['entries']}")

        if self.scan_stats['symbols_scanned'] > 0:
            spike_rate = (self.scan_stats['spike_candidates'] / self.scan_stats['symbols_scanned']) * 100
            debug_logger.debug(f"Spike detection rate: {spike_rate:.3f}%")

        if self.scan_stats['spike_candidates'] > 0:
            entry_rate = (self.scan_stats['entries'] / self.scan_stats['spike_candidates']) * 100
            debug_logger.debug(f"Entry rate (from candidates): {entry_rate:.1f}%")

        debug_logger.debug("=" * 80)
        debug_logger.debug("")

    def start_trading(self):
        """Start the main trading loop."""
        self.initialize()

        logger.info("Starting trading loop...")
        logger.info(f"Cycle duration: {CONFIG['cycle_duration_seconds']}s")

        try:
            while True:
                self.cycle_count += 1
                cycle_start = time.time()

                logger.info(f"\n--- Cycle {self.cycle_count} ---")

                # Step 1: Check exits (highest priority)
                self.check_position_exits()

                # Step 2: Scan for entries
                self.scan_for_entries()

                # Step 3: Log status (every 10 cycles = ~10 minutes)
                if self.cycle_count % 10 == 0:
                    self.log_portfolio_status()
                    self.log_debug_summary()

                # Step 4: Sleep until next cycle
                elapsed = time.time() - cycle_start
                sleep_time = max(0, CONFIG['cycle_duration_seconds'] - elapsed)

                logger.info(f"Cycle completed in {elapsed:.1f}s, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("\nShutdown requested by user")
            self.log_portfolio_status()
            logger.info("Trading stopped")


if __name__ == "__main__":
    trader = RBPaperTrader()
    trader.start_trading()
