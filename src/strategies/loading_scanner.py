"""
Loading Scanner — Stage 1 of the spike detection system.

Runs every 60 seconds. For each symbol, reads the last 6 hours of 1m candles
from the FeatureBuffer (SQLite) and computes rolling loading indicators.

When a coin enters a "loading" state (volume building, volatility expanding,
BB widening, VPIN low), it flags it for Stage 2 monitoring.

Loading signals (from pre-trigger analysis of 50%+ spikes):
  - Volume trend (late vs early):          1.51x before spikes vs 1.08 normal
  - Volume acceleration (last 30m):        1.50x vs 0.94
  - Volume last 15m vs avg:                2.57x vs 0.40
  - NATR (from raw candles):               2.54x vs normal
  - BB width:                              2.24x vs normal
  - Price range (6h):                      2.58x vs normal
  - Close near top of range:               0.64 vs 0.43
  - Volume spike ratio:                    56x vs normal
"""

import sqlite3
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from .loading_paper_trader import LoadingPaperTrader
from .fizzler_filter import FizzlerFilter


@dataclass
class LoadingAlert:
    """A loading detection alert."""
    symbol: str
    timestamp: datetime
    score: float
    components: Dict[str, float]
    price_at_alert: float
    phase: str = "loading"  # loading, triggered, confirmed, expired


@dataclass
class TrackedPosition:
    """A tracked position from loading alert through confirmation."""
    symbol: str
    alert_time: datetime
    alert_price: float
    alert_score: float
    phase: str = "loading"  # loading → triggered → confirmed → closed
    trigger_time: Optional[datetime] = None
    trigger_price: Optional[float] = None
    confirm_time: Optional[datetime] = None
    confirm_price: Optional[float] = None
    entry_price: Optional[float] = None  # Blended entry
    high_since_entry: float = 0.0
    current_price: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: Optional[str] = None


class LoadingScanner:
    """
    Scans all symbols for pre-spike loading patterns.

    Reads 1m candles from the FeatureBuffer SQLite and computes
    loading indicators every scan_interval seconds.
    """

    # Minimum candles needed for reliable signals
    MIN_CANDLES_6H = 200  # ~3.3h minimum (out of 6h = 360 candles)
    MIN_CANDLES_1H = 40

    def __init__(
        self,
        feature_buffer_path: str = None,
        scan_interval: int = 60,
        loading_threshold: float = 5.0,
        trigger_pct: float = 5.0,
        confirm_minutes: int = 60,
        callback=None,
    ):
        """
        Args:
            feature_buffer_path: Path to FeatureBuffer SQLite DB
            scan_interval: Seconds between scans (default 60)
            loading_threshold: Score threshold to flag as loading (default 5.0)
            trigger_pct: Price move % to trigger Stage 2 (default 5.0)
            confirm_minutes: Minutes to wait for confirmation after trigger (default 60)
            callback: Async callback for alerts: callback(alert: LoadingAlert)
        """
        if feature_buffer_path is None:
            feature_buffer_path = str(Path(__file__).parent.parent.parent / "data" / "feature_buffer.db")

        self.db_path = feature_buffer_path
        self.scan_interval = scan_interval
        self.loading_threshold = loading_threshold
        self.trigger_pct = trigger_pct
        self.confirm_minutes = confirm_minutes
        self.callback = callback

        # State
        self.running = False
        self.scores: Dict[str, float] = {}  # Latest score per symbol
        self.alerts: Dict[str, LoadingAlert] = {}  # Active alerts
        self.tracked: Dict[str, TrackedPosition] = {}  # Tracked positions
        self.scan_count = 0
        self._latest_prices: Dict[str, float] = {}  # Latest close price per symbol
        self._start_time = datetime.now(timezone.utc)
        self._warmup_minutes = 360  # Don't trade until 6h of data in buffer

        # Paper trader
        self.paper_trader = LoadingPaperTrader(
            full_position_size=1000.0,
            max_positions=10,
            phase1_pct=0.20,
            phase1_stop=0.03,
            phase2_stop=0.08,
            trail_pct=0.08,         # 8% trail (optimal from backtest: best median capture)
            trail_activate=0.15,    # Activate after 15% gain
            time_stop_hours=48,
            score_fade_minutes=60,
        )

        # ML fizzler filter (v2 — XGBoost + CNN sequence features)
        self.fizzler_filter = FizzlerFilter(threshold=0.30)

        # Load repeat spikers from elite_movers.duckdb
        self._repeat_spikers: set = set()
        self._load_repeat_spikers()

        # Stats
        self.stats = {
            'scans': 0,
            'alerts_fired': 0,
            'triggers_detected': 0,
            'confirmations': 0,
            'expired': 0,
        }

        logger.info(
            f"LoadingScanner initialized: interval={scan_interval}s, "
            f"threshold={loading_threshold}, trigger={trigger_pct}%"
        )

    def _load_repeat_spikers(self):
        """Load repeat spiker symbols from elite_movers.duckdb."""
        try:
            import duckdb
            elite_path = str(Path(__file__).parent.parent.parent / "data" / "elite_movers.duckdb")
            conn = duckdb.connect(elite_path, read_only=True)
            rows = conn.execute(
                "SELECT symbol FROM symbol_stats WHERE is_repeat_spiker"
            ).fetchall()
            self._repeat_spikers = {r[0] for r in rows}
            conn.close()
            logger.info(f"[LOADING] Loaded {len(self._repeat_spikers)} repeat spikers from elite_movers.duckdb")
        except Exception as e:
            logger.debug(f"[LOADING] Could not load repeat spikers: {e}")
            self._repeat_spikers = set()

    def _record_trade(self, symbol: str, score: float, components: dict,
                       price: float, timestamp: datetime, event_type: str = "entry",
                       exit_reason: str = None, pnl_pct: float = None,
                       peak_price: float = None, hold_hours: float = None,
                       phase: str = "phase1"):
        """Record a trade entry/exit to elite_movers.duckdb for fizzler analysis."""
        try:
            import duckdb
            elite_path = str(Path(__file__).parent.parent.parent / "data" / "elite_movers.duckdb")
            conn = duckdb.connect(elite_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    timestamp TIMESTAMP,
                    symbol VARCHAR,
                    event_type VARCHAR,
                    phase VARCHAR,
                    score DOUBLE,
                    price DOUBLE,
                    vol_trend DOUBLE,
                    vol_last_vs_avg DOUBLE,
                    vol_accel DOUBLE,
                    natr DOUBLE,
                    bb_width DOUBLE,
                    momentum_1h DOUBLE,
                    price_range DOUBLE,
                    close_position DOUBLE,
                    bot_net_pct DOUBLE,
                    spread_pct DOUBLE,
                    repeat_spiker BOOLEAN,
                    hour_utc INTEGER,
                    exit_reason VARCHAR,
                    pnl_pct DOUBLE,
                    peak_price DOUBLE,
                    hold_hours DOUBLE
                )
            """)
            conn.execute("""
                INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                timestamp.isoformat(),
                symbol,
                event_type,
                phase,
                score,
                price,
                components.get('vol_trend'),
                components.get('vol_last_vs_avg'),
                components.get('vol_accel'),
                components.get('natr'),
                components.get('bb_width'),
                components.get('momentum_1h'),
                components.get('price_range'),
                components.get('close_position'),
                components.get('bot_net_pct'),
                components.get('spread_pct'),
                components.get('repeat_spiker', False),
                components.get('hour_utc'),
                exit_reason,
                pnl_pct,
                peak_price,
                hold_hours,
            ])
            conn.close()
        except Exception as e:
            logger.debug(f"[LOADING] Trade record error: {e}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get SQLite connection to FeatureBuffer."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_candles(self, conn, symbol: str, minutes: int = 360) -> Optional[pd.DataFrame]:
        """Load last N minutes of 1m candles for a symbol."""
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()

        rows = conn.execute(
            "SELECT timestamp, open, high, low, close, volume "
            "FROM ohlcv WHERE symbol = ? AND timestamp > ? "
            "ORDER BY timestamp",
            (symbol, cutoff)
        ).fetchall()

        if not rows or len(rows) < self.MIN_CANDLES_1H:
            return None

        df = pd.DataFrame([dict(r) for r in rows])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Drop exact duplicates
        df = df.drop_duplicates(subset='timestamp', keep='last')

        return df

    def _compute_bot_accumulation(self, conn, symbol: str, minutes: int = 360) -> Dict[str, float]:
        """
        Detect algorithmic accumulation from raw trades.

        Bot signature: 5+ trades in the same second with <=3 unique sizes.
        Returns net bot buy pressure as % of total volume.

        Validated: 24.7x higher before big movers vs control periods.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()

        rows = conn.execute(
            "SELECT timestamp, price, size, side "
            "FROM trades WHERE symbol = ? AND timestamp > ? "
            "ORDER BY timestamp",
            (symbol, cutoff)
        ).fetchall()

        if not rows or len(rows) < 50:
            return {'bot_net_pct': 0, 'bot_buy_ratio': 0.5, 'bot_bursts': 0}

        # Group by second
        from collections import defaultdict
        by_second = defaultdict(list)
        total_usd = 0

        for r in rows:
            price = float(r['price'])
            size = float(r['size'])
            usd = price * size
            total_usd += usd
            # Floor to second
            ts_str = str(r['timestamp'])[:19]
            by_second[ts_str].append((size, r['side'], usd))

        if total_usd <= 0:
            return {'bot_net_pct': 0, 'bot_buy_ratio': 0.5, 'bot_bursts': 0}

        buy_bot_usd = 0
        sell_bot_usd = 0
        bot_bursts = 0

        for ts, trades in by_second.items():
            if len(trades) < 5:
                continue

            sizes = [round(t[0], 2) for t in trades]
            if len(set(sizes)) <= 3:
                bot_bursts += 1
                for size, side, usd in trades:
                    if side == 'BUY':
                        buy_bot_usd += usd
                    else:
                        sell_bot_usd += usd

        bot_total = buy_bot_usd + sell_bot_usd
        return {
            'bot_net_pct': (buy_bot_usd - sell_bot_usd) / total_usd * 100,
            'bot_buy_ratio': buy_bot_usd / bot_total if bot_total > 0 else 0.5,
            'bot_bursts': bot_bursts,
        }

    def _get_latest_spreads(self, conn) -> Dict[str, float]:
        """
        Get latest bid-ask spread for all symbols from order_book_snapshots.
        Returns {symbol: spread_pct}. Uses a lightweight approach: extracts
        just the first bid/ask price from JSON via string slicing to avoid
        full JSON parsing of large blobs (which caused heap corruption under
        concurrent writes).
        """
        spreads = {}
        try:
            # Use SQLite json_extract to pull just the top-of-book prices
            # without loading full JSON blobs into Python
            rows = conn.execute(
                """SELECT symbol,
                          json_extract(bids, '$[0][0]') as best_bid,
                          json_extract(asks, '$[0][0]') as best_ask
                   FROM order_book_snapshots
                   WHERE id IN (
                       SELECT MAX(id) FROM order_book_snapshots GROUP BY symbol
                   )
                   AND bids IS NOT NULL AND asks IS NOT NULL"""
            ).fetchall()
            for r in rows:
                try:
                    best_bid = float(r['best_bid'])
                    best_ask = float(r['best_ask'])
                    mid = (best_bid + best_ask) / 2
                    if mid > 0 and best_ask > best_bid:
                        spreads[r['symbol']] = (best_ask - best_bid) / mid * 100
                except (TypeError, ValueError):
                    pass
        except Exception as e:
            logger.debug(f"[LOADING] Spread query error: {e}")
        return spreads

    def compute_loading_score(self, df: pd.DataFrame, bot_data: Dict[str, float] = None, spread_pct: float = None, symbol: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Compute loading score from 1m candles.

        Returns (score, component_dict).
        Score >= threshold means the coin is in a loading state.
        """
        close = df['close'].values
        volume = df['volume'].values
        high = df['high'].values
        low = df['low'].values
        n = len(df)

        components = {}

        # ── Volume features ──────────────────────────────────────

        # Volume trend: last 1h vs first 1h of window
        if n >= 120:
            first_1h = volume[:60]
            last_1h = volume[-60:]
        else:
            half = n // 2
            first_1h = volume[:half]
            last_1h = volume[half:]

        first_mean = first_1h.mean()
        last_mean = last_1h.mean()

        vol_trend = last_mean / first_mean if first_mean > 0 else 1.0
        components['vol_trend'] = vol_trend

        # Volume acceleration: last 15m vs previous 15m
        if n >= 30:
            prev_15m = volume[-30:-15]
            last_15m = volume[-15:]
            prev_mean = prev_15m.mean()
            vol_accel = last_15m.mean() / prev_mean if prev_mean > 0 else 1.0
        else:
            vol_accel = 1.0
        components['vol_accel'] = vol_accel

        # Volume last 15m vs full window average
        avg_vol = volume.mean()
        last_15m_mean = volume[-15:].mean() if n >= 15 else avg_vol
        vol_last_vs_avg = last_15m_mean / avg_vol if avg_vol > 0 else 1.0
        components['vol_last_vs_avg'] = vol_last_vs_avg

        # Volume max spike ratio
        vol_max_ratio = volume.max() / avg_vol if avg_vol > 0 else 1.0
        components['vol_max_ratio'] = vol_max_ratio

        # ── Volatility / Price features ──────────────────────────

        returns = np.diff(close) / close[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) < 20:
            return 0.0, components

        # NATR equivalent: ATR / close (from raw candles)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr_14 = pd.Series(tr).rolling(14).mean().iloc[-1]
        natr = atr_14 / close[-1] * 100 if close[-1] > 0 else 0
        components['natr'] = natr

        # BB width equivalent: (upper - lower) / middle from 20-period
        close_series = pd.Series(close)
        sma_20 = close_series.rolling(20).mean().iloc[-1]
        std_20 = close_series.rolling(20).std().iloc[-1]
        if sma_20 > 0 and np.isfinite(std_20):
            bb_width = (2 * 2 * std_20) / sma_20  # 2σ bands
        else:
            bb_width = 0
        components['bb_width'] = bb_width

        # Price compression: last 1h volatility vs full window volatility
        if len(returns) >= 60:
            vol_1h = returns[-60:].std()
            vol_full = returns.std()
            compression = vol_1h / vol_full if vol_full > 0 else 1.0
        else:
            compression = 1.0
        components['compression'] = compression

        # Price range over window
        price_range = (close.max() - close.min()) / close.min() * 100 if close.min() > 0 else 0
        components['price_range'] = price_range

        # Close position within range (0 = at low, 1 = at high)
        if close.max() > close.min():
            close_position = (close[-1] - close.min()) / (close.max() - close.min())
        else:
            close_position = 0.5
        components['close_position'] = close_position

        # Momentum: last 1h
        if n >= 60:
            momentum_1h = (close[-1] / close[-60] - 1) * 100
        else:
            momentum_1h = (close[-1] / close[0] - 1) * 100
        components['momentum_1h'] = momentum_1h

        # ── Compute score ────────────────────────────────────────
        # Calibrated from elite movers (204 events, 50%+) vs fizzlers analysis.
        # Key findings: vol_trend is #1 discriminator (5.0x elite vs 1.6x fizzlers),
        # NATR is #2 (0.60 vs 0.40), bot_net ANTI-correlates (fizzlers have MORE
        # bot buying than winners).
        score = 0.0

        # ── Hard gate: NATR must be >= 0.5 ──
        # Blocks 65% of fizzlers, keeps 53% of elite movers.
        # Most fizzlers have natr 0.3-0.5, elite movers are typically 0.6+.
        if natr < 0.5:
            components['score'] = 0.0
            components['gate_failed'] = 'natr'
            return 0.0, components

        # Volume signals — raised thresholds (elite median vol_trend=5.0x)
        if vol_last_vs_avg > 2.0:
            score += 2.0
        elif vol_last_vs_avg > 1.5:
            score += 1.0

        if vol_trend > 3.0:
            score += 2.5
        elif vol_trend > 2.0:
            score += 1.5
        elif vol_trend > 1.3:
            score += 0.5

        if vol_accel > 1.3:
            score += 1.0
        elif vol_accel > 1.1:
            score += 0.5

        # Volatility signals (NATR already passed gate >= 0.5)
        if natr > 1.0:
            score += 2.0
        elif natr > 0.7:
            score += 1.5
        else:
            score += 1.0  # 0.5-0.7 range (passed gate)

        if bb_width > 0.05:
            score += 1.5
        elif bb_width > 0.03:
            score += 1.0
        elif bb_width > 0.02:
            score += 0.5

        # Price action signals
        if close_position > 0.6:
            score += 0.5

        if price_range > 8:
            score += 1.0
        elif price_range > 5:
            score += 0.5

        # Small momentum bonus (not too much — big momentum means we're late)
        if 1.0 < momentum_1h < 5.0:
            score += 0.5

        # ── Bot accumulation: RECORD ONLY, no score impact ──
        # Fizzler analysis showed bot_net ANTI-correlates with winners:
        # fizzlers median +7.15% bot_net vs elite movers +0.53%.
        # 70% of fizzlers have positive bot_net vs only 48% of elite.
        # Bot buy boost was actively selecting worse trades — removed entirely.
        if bot_data:
            components['bot_net_pct'] = bot_data.get('bot_net_pct', 0)
            components['bot_buy_ratio'] = bot_data.get('bot_buy_ratio', 0.5)
            components['bot_bursts'] = bot_data.get('bot_bursts', 0)
            # No score adjustment — bot data is recorded for analysis only

        # ── Repeat spiker bonus (56% of elite events vs 15% of fizzlers) ──
        if symbol and symbol in self._repeat_spikers:
            score += 2.0
            components['repeat_spiker'] = True
        else:
            components['repeat_spiker'] = False

        # ── Spread signal (1.58x lift for illiquid coins) ──
        if spread_pct is not None:
            components['spread_pct'] = spread_pct
            if spread_pct >= 0.5:
                score += 1.5
            elif spread_pct >= 0.2:
                score += 0.5

        # ── Time-of-day signal (1.7x lift at 02-04 UTC, 0.6x at 10-12) ──
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour
        components['hour_utc'] = hour
        if 2 <= hour < 4:
            score += 1.0
        elif 0 <= hour < 2 or 20 <= hour < 22:
            score += 0.5
        elif 10 <= hour < 12:
            score -= 0.5

        components['score'] = score
        return score, components

    def scan_all(self) -> List[LoadingAlert]:
        """
        Run one scan across all symbols.

        Returns list of new alerts (symbols crossing threshold).
        """
        conn = self._get_conn()
        new_alerts = []

        try:
            # Get all symbols in the buffer
            symbols = [r[0] for r in conn.execute(
                "SELECT DISTINCT symbol FROM ohlcv"
            ).fetchall()]

            # Pre-fetch latest spread for all symbols (one query)
            spread_map = self._get_latest_spreads(conn)

            for symbol in symbols:
                df = self._load_candles(conn, symbol, minutes=360)
                if df is None:
                    continue

                current_price = df['close'].iloc[-1]

                # Skip coins below $0.0001 — tick granularity makes stops unreliable
                # (only 6 coins affected: MOG, PEPE, SHIB, BONK, FLOKI, NOICE)
                # Only 2 of 190 fifty-percent movers were below this threshold
                if current_price < 0.0001:
                    continue

                spread_pct = spread_map.get(symbol)

                # First pass: compute base score without bot data (fast)
                score, components = self.compute_loading_score(df, spread_pct=spread_pct, symbol=symbol)

                # Second pass: if base score is promising, compute bot accumulation
                # (expensive — only for candidates, not all 300+ symbols)
                # Bot data is recorded for analysis but no longer affects score
                if score >= self.loading_threshold - 2.0:
                    try:
                        bot_data = self._compute_bot_accumulation(conn, symbol, minutes=360)
                        score, components = self.compute_loading_score(df, bot_data, spread_pct=spread_pct, symbol=symbol)
                    except Exception as e:
                        logger.debug(f"[LOADING] Bot analysis error for {symbol}: {e}")

                self.scores[symbol] = score

                # Track latest prices for paper trader
                self._latest_prices[symbol] = current_price

                # Check for new loading alert
                if score >= self.loading_threshold and symbol not in self.alerts:
                    # ML fizzler filter — disabled, logging only for data collection
                    # Re-enable once we have 1000+ labeled trades to train on
                    fizzle_prob, _ = self.fizzler_filter.predict(components, df)
                    # Log but don't block
                    if fizzle_prob >= 0.30:
                        logger.debug(
                            f"[FIZZLER] {symbol} score={score:.1f} "
                            f"fizzle_prob={fizzle_prob:.2f} (log-only)"
                        )

                    # Paper trader: try Phase 1 entry first
                    now = datetime.now(timezone.utc)
                    entered = self.paper_trader.on_loading_alert(symbol, score, current_price, now)

                    if entered:
                        # Only track alert if we actually entered (or already have position)
                        alert = LoadingAlert(
                            symbol=symbol,
                            timestamp=datetime.now(timezone.utc),
                            score=score,
                            components=components,
                            price_at_alert=current_price,
                        )
                        self.alerts[symbol] = alert
                        new_alerts.append(alert)
                        self.stats['alerts_fired'] += 1
                    # If entry failed (max positions, no cash), don't add to self.alerts
                    # so the symbol can re-alert on the next scan when a slot opens

                        bot_str = f" bot_net={components.get('bot_net_pct', 0):+.1f}%" if 'bot_net_pct' in components else ""
                        spread_str = f" spread={components.get('spread_pct', 0):.2f}%" if 'spread_pct' in components else ""
                        logger.info(
                            f"[LOADING] {symbol} score={score:.1f} price=${current_price:.4f} "
                            f"vol_trend={components.get('vol_trend', 0):.2f} "
                            f"natr={components.get('natr', 0):.2f} "
                            f"bb={components.get('bb_width', 0):.4f}"
                            f"{bot_str}{spread_str}"
                        )

                        # Record entry to elite_movers.duckdb trades table
                        self._record_trade(symbol, score, components, current_price, now)

                # Update existing alerts
                elif symbol in self.alerts:
                    alert = self.alerts[symbol]

                    # Check for trigger (5%+ move from alert price)
                    if alert.phase == "loading":
                        move_pct = (current_price - alert.price_at_alert) / alert.price_at_alert * 100
                        if move_pct >= self.trigger_pct:
                            alert.phase = "triggered"
                            alert.timestamp = datetime.now(timezone.utc)  # Reset for confirmation timer
                            self.stats['triggers_detected'] += 1
                            logger.info(
                                f"[TRIGGER] {symbol} +{move_pct:.1f}% from alert "
                                f"(${alert.price_at_alert:.4f} → ${current_price:.4f})"
                            )

                        # Expire if score drops and no trigger after 6h
                        elif (datetime.now(timezone.utc) - alert.timestamp).total_seconds() > 21600:
                            if score < self.loading_threshold * 0.7:
                                del self.alerts[symbol]
                                self.stats['expired'] += 1
                                logger.debug(f"[EXPIRED] {symbol} loading faded (score={score:.1f})")

                    # Check for confirmation (still positive 1h after trigger)
                    elif alert.phase == "triggered":
                        time_since_trigger = (datetime.now(timezone.utc) - alert.timestamp).total_seconds()
                        move_from_alert = (current_price - alert.price_at_alert) / alert.price_at_alert * 100

                        if time_since_trigger >= self.confirm_minutes * 60:
                            if move_from_alert > 0:
                                alert.phase = "confirmed"
                                self.stats['confirmations'] += 1
                                logger.success(
                                    f"[CONFIRMED] {symbol} +{move_from_alert:.1f}% "
                                    f"still positive after {self.confirm_minutes}min"
                                )

                                # Paper trader: Phase 2 scale-in
                                self.paper_trader.on_trigger_confirmed(
                                    symbol, current_price, datetime.now(timezone.utc)
                                )
                            else:
                                alert.phase = "expired"
                                del self.alerts[symbol]
                                self.stats['expired'] += 1
                                logger.info(
                                    f"[FAILED] {symbol} {move_from_alert:+.1f}% "
                                    f"reversed after trigger"
                                )

                # Clear old alerts for symbols that dropped below threshold
                elif symbol in self.alerts and score < self.loading_threshold * 0.5:
                    alert = self.alerts[symbol]
                    if alert.phase == "loading":
                        del self.alerts[symbol]

        except Exception as e:
            logger.error(f"[LOADING] Scan error: {e}")
        finally:
            conn.close()

        # Update paper trader with latest prices and scores
        now = datetime.now(timezone.utc)
        n_closed_before = len(self.paper_trader.closed_positions)
        self.paper_trader.update_prices(self._latest_prices, now, self.scores)

        # Record any new exits to elite_movers.duckdb trades table
        new_closes = self.paper_trader.closed_positions[n_closed_before:]
        for pos in new_closes:
            hold_hours = (pos.exit_time - pos.phase1_time).total_seconds() / 3600 if pos.exit_time and pos.phase1_time else None
            self._record_trade(
                symbol=pos.symbol,
                score=pos.loading_score,
                components={},  # Components not stored on position, but we have the key fields
                price=pos.exit_price,
                timestamp=pos.exit_time or now,
                event_type="exit",
                exit_reason=pos.exit_reason,
                pnl_pct=pos.pnl_pct(pos.exit_price),
                peak_price=pos.peak_price,
                hold_hours=hold_hours,
                phase=pos.phase,
            )

        self.scan_count += 1
        self.stats['scans'] += 1

        return new_alerts

    def get_active_alerts(self) -> Dict[str, LoadingAlert]:
        """Get all active alerts."""
        return dict(self.alerts)

    def get_top_scores(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N symbols by loading score."""
        sorted_scores = sorted(self.scores.items(), key=lambda x: -x[1])
        return sorted_scores[:n]

    def get_stats(self) -> Dict:
        """Get scanner statistics."""
        return {
            **self.stats,
            'active_alerts': len(self.alerts),
            'symbols_monitored': len(self.scores),
            'alerts_by_phase': {
                phase: sum(1 for a in self.alerts.values() if a.phase == phase)
                for phase in ['loading', 'triggered', 'confirmed']
            }
        }

    async def start(self):
        """Start the scanner loop (async, for integration with collector)."""
        import asyncio
        self.running = True
        logger.info(f"[LOADING] Scanner started (interval={self.scan_interval}s)")

        while self.running:
            try:
                t0 = time.time()
                new_alerts = self.scan_all()

                # Fire callback for new alerts
                if new_alerts and self.callback:
                    for alert in new_alerts:
                        try:
                            await self.callback(alert)
                        except Exception as e:
                            logger.error(f"[LOADING] Callback error: {e}")

                # Reload repeat spikers and log summary periodically
                if self.scan_count % 60 == 0:  # Every ~60 minutes
                    self._load_repeat_spikers()
                    stats = self.get_stats()
                    top = self.get_top_scores(5)
                    top_str = ", ".join(f"{s}={sc:.1f}" for s, sc in top)
                    trader_summary = self.paper_trader.get_summary()
                    logger.info(
                        f"[LOADING] Scan #{self.scan_count}: "
                        f"{stats['symbols_monitored']} symbols, "
                        f"{stats['active_alerts']} active alerts, "
                        f"top: {top_str}"
                    )
                    logger.info(f"[LOADING] {trader_summary}")

                elapsed = time.time() - t0
                sleep_time = max(0, self.scan_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[LOADING] Loop error: {e}")
                await asyncio.sleep(self.scan_interval)

        logger.info("[LOADING] Scanner stopped")

    def stop(self):
        """Stop the scanner loop."""
        self.running = False
