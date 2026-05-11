"""
Elite Mover Tracker — auto-detects 50%+ price moves and records them in elite_movers.duckdb.

Runs every hour. Scans the feature buffer for coins that have moved 50%+ from their
24h low to their 24h high. New events are inserted with:
  - Full 1m candle data (4h pre-spike through 1h post-peak)
  - Bot accumulation metrics from L1 trades
  - Feature snapshots (NATR, BB width, RSI, VPIN, spread, orderbook)
  - Post-spike behavior (price at +15m, +30m, +60m after peak)

Can run standalone or as an async task inside the collector.
"""

import sqlite3
import json
import duckdb
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from loguru import logger
from collections import defaultdict


ELITE_DB = str(Path(__file__).parent.parent.parent / "data" / "elite_movers.duckdb")
FEATURE_BUFFER = str(Path(__file__).parent.parent.parent / "data" / "feature_buffer.db")


def _get_buffer_conn(buffer_path: str):
    conn = sqlite3.connect(buffer_path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def detect_spikes(buffer_path: str = FEATURE_BUFFER, min_gain_pct: float = 50.0) -> list:
    """
    Scan feature buffer for coins that moved 50%+ in the last 24h.
    Returns list of dicts with event details.
    """
    conn = _get_buffer_conn(buffer_path)
    cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

    rows = conn.execute("""
        SELECT symbol,
               MIN(low) as low_price,
               MAX(high) as peak_price,
               MIN(timestamp) as first_ts,
               MAX(timestamp) as last_ts,
               AVG(volume) as avg_vol,
               COUNT(*) as candle_count
        FROM ohlcv
        WHERE timestamp > ?
        GROUP BY symbol
        HAVING low_price > 0
    """, (cutoff_24h,)).fetchall()

    spikes = []
    for r in rows:
        low = r['low_price']
        peak = r['peak_price']
        gain_pct = (peak - low) / low * 100

        if gain_pct >= min_gain_pct:
            # Find approximate start time (when was the low?)
            low_row = conn.execute("""
                SELECT timestamp FROM ohlcv
                WHERE symbol = ? AND timestamp > ? AND low = ?
                ORDER BY timestamp LIMIT 1
            """, (r['symbol'], cutoff_24h, low)).fetchone()

            # Find peak time
            peak_row = conn.execute("""
                SELECT timestamp FROM ohlcv
                WHERE symbol = ? AND timestamp > ? AND high = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (r['symbol'], cutoff_24h, peak)).fetchone()

            low_ts = low_row['timestamp'] if low_row else r['first_ts']
            peak_ts = peak_row['timestamp'] if peak_row else r['last_ts']
            duration_hours = (
                datetime.fromisoformat(peak_ts) - datetime.fromisoformat(low_ts)
            ).total_seconds() / 3600

            spikes.append({
                'symbol': r['symbol'],
                'start': low_ts,
                'end': peak_ts,
                'peak_gain_pct': gain_pct,
                'low_price': low,
                'peak_price': peak,
                'duration_hours': max(duration_hours, 0.01),
            })

    conn.close()
    return spikes


def collect_event_data(buffer_path: str, symbol: str, start_ts: str, peak_ts: str,
                       low_price: float, peak_price: float) -> dict:
    """
    Collect rich data for a spike event:
    - 1m candles: 4h before spike start through 1h after peak
    - Feature snapshot at spike start
    - Bot accumulation from 4h of trades before spike
    - Post-spike prices (15m, 30m, 60m after peak)
    - Orderbook spread at spike start
    """
    conn = _get_buffer_conn(buffer_path)
    start_dt = datetime.fromisoformat(start_ts)
    peak_dt = datetime.fromisoformat(peak_ts)

    # ── 1m candles: 4h pre through 1h post-peak ──
    candle_start = (start_dt - timedelta(hours=4)).isoformat()
    candle_end = (peak_dt + timedelta(hours=1)).isoformat()

    candle_rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
    """, (symbol, candle_start, candle_end)).fetchall()

    candles = [
        {
            'timestamp': r['timestamp'],
            'open': r['open'],
            'high': r['high'],
            'low': r['low'],
            'close': r['close'],
            'volume': r['volume'],
        }
        for r in candle_rows
    ]

    # ── Pre-spike metrics (1h before start) ──
    pre_start = (start_dt - timedelta(hours=1)).isoformat()
    pre_candles = conn.execute("""
        SELECT open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
        ORDER BY timestamp
    """, (symbol, pre_start, start_ts)).fetchall()

    pre_avg_price = None
    pre_avg_vol = None
    pre_price_cv = None
    pre_natr = None
    pre_bb_width = None
    if pre_candles and len(pre_candles) >= 10:
        closes = [r['close'] for r in pre_candles]
        volumes = [r['volume'] for r in pre_candles]
        highs = [r['high'] for r in pre_candles]
        lows = [r['low'] for r in pre_candles]
        pre_avg_price = np.mean(closes)
        pre_avg_vol = np.mean(volumes)
        pre_price_cv = np.std(closes) / np.mean(closes) if np.mean(closes) > 0 else 0

        # NATR
        tr = [max(h - l, abs(h - closes[i-1]), abs(l - closes[i-1]))
              for i, (h, l) in enumerate(zip(highs[1:], lows[1:]), 1)]
        if tr and closes[-1] > 0:
            atr14 = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            pre_natr = atr14 / closes[-1] * 100

        # BB width
        if len(closes) >= 20:
            sma20 = np.mean(closes[-20:])
            std20 = np.std(closes[-20:])
            if sma20 > 0:
                pre_bb_width = (4 * std20) / sma20

    # ── During-spike metrics ──
    during_candles = conn.execute("""
        SELECT volume FROM ohlcv
        WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
    """, (symbol, start_ts, peak_ts)).fetchall()

    during_avg_vol = None
    during_total_vol = None
    vol_multiplier = None
    if during_candles:
        vols = [r['volume'] for r in during_candles]
        during_avg_vol = np.mean(vols)
        during_total_vol = sum(vols)
        if pre_avg_vol and pre_avg_vol > 0:
            vol_multiplier = during_avg_vol / pre_avg_vol

    # ── Feature snapshot at spike start ──
    feat_row = conn.execute("""
        SELECT * FROM features
        WHERE symbol = ? AND timestamp <= ?
        ORDER BY timestamp DESC LIMIT 1
    """, (symbol, start_ts)).fetchone()

    features = {}
    if feat_row:
        for key in feat_row.keys():
            if key not in ('id', 'timestamp', 'symbol'):
                features[key] = feat_row[key]

    # ── Orderbook spread at spike start ──
    spread_pct = None
    ob_row = conn.execute("""
        SELECT bids, asks FROM order_book_snapshots
        WHERE symbol = ? AND timestamp <= ?
        ORDER BY timestamp DESC LIMIT 1
    """, (symbol, start_ts)).fetchone()

    if ob_row and ob_row['bids'] and ob_row['asks']:
        try:
            bids = json.loads(ob_row['bids'])
            asks = json.loads(ob_row['asks'])
            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid = (best_bid + best_ask) / 2
                if mid > 0 and best_ask > best_bid:
                    spread_pct = (best_ask - best_bid) / mid * 100
        except (ValueError, IndexError, json.JSONDecodeError):
            pass

    # ── Post-spike behavior ──
    post_prices = {}
    for minutes in [15, 30, 60]:
        post_ts = (peak_dt + timedelta(minutes=minutes)).isoformat()
        post_row = conn.execute("""
            SELECT close FROM ohlcv
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp LIMIT 1
        """, (symbol, post_ts, (peak_dt + timedelta(minutes=minutes+2)).isoformat())).fetchone()

        if post_row:
            post_prices[f'price_post_{minutes}m'] = post_row['close']
            post_prices[f'pct_post_{minutes}m'] = (post_row['close'] - peak_price) / peak_price * 100
        else:
            post_prices[f'price_post_{minutes}m'] = None
            post_prices[f'pct_post_{minutes}m'] = None

    # ── Bot accumulation (4h before spike) ──
    bot = _compute_bot_metrics(conn, symbol, start_ts)

    conn.close()

    return {
        'candles': candles,
        'pre_avg_price': pre_avg_price,
        'pre_avg_vol': pre_avg_vol,
        'pre_price_cv': pre_price_cv,
        'pre_natr': pre_natr,
        'pre_bb_width': pre_bb_width,
        'during_avg_vol': during_avg_vol,
        'during_total_vol': during_total_vol,
        'vol_multiplier': vol_multiplier,
        'features': features,
        'spread_pct': spread_pct,
        'post_prices': post_prices,
        'bot': bot,
    }


def _compute_bot_metrics(conn, symbol: str, start_ts: str) -> dict:
    """Compute bot accumulation from trades in 4h before spike start."""
    start_dt = datetime.fromisoformat(start_ts)
    window_start = (start_dt - timedelta(hours=4)).isoformat()

    rows = conn.execute(
        "SELECT timestamp, price, size, side "
        "FROM trades WHERE symbol = ? AND timestamp > ? AND timestamp <= ? "
        "ORDER BY timestamp",
        (symbol, window_start, start_ts)
    ).fetchall()

    if not rows or len(rows) < 50:
        return None

    by_second = defaultdict(list)
    total_usd = 0

    for r in rows:
        price = float(r['price'])
        size = float(r['size'])
        usd = price * size
        total_usd += usd
        ts_str = str(r['timestamp'])[:19]
        by_second[ts_str].append((size, r['side'], usd))

    if total_usd <= 0:
        return None

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
        'bot_buy_volume_usd': buy_bot_usd,
        'bot_sell_volume_usd': sell_bot_usd,
        'bot_net_usd': buy_bot_usd - sell_bot_usd,
        'total_volume_usd': total_usd,
        'bot_net_pct': (buy_bot_usd - sell_bot_usd) / total_usd * 100,
        'bot_buy_ratio': buy_bot_usd / bot_total if bot_total > 0 else 0.5,
        'bot_burst_count': bot_bursts,
        'total_trade_count': len(rows),
    }


def _ensure_schema(conn):
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS elite_events (
            event_id BIGINT,
            symbol VARCHAR,
            start TIMESTAMP,
            "end" TIMESTAMP,
            peak_gain_pct DOUBLE,
            low_price DOUBLE,
            peak_price DOUBLE,
            duration_hours DOUBLE,
            -- Pre-spike metrics
            pre_avg_vol DOUBLE,
            pre_price_cv DOUBLE,
            pre_avg_price DOUBLE,
            pre_natr DOUBLE,
            pre_bb_width DOUBLE,
            -- During-spike metrics
            during_avg_vol DOUBLE,
            during_total_vol DOUBLE,
            vol_multiplier DOUBLE,
            -- Feature snapshot at spike start
            feat_rsi_14 DOUBLE,
            feat_vpin DOUBLE,
            feat_natr DOUBLE,
            feat_bb_width DOUBLE,
            feat_volume_spike_ratio DOUBLE,
            feat_bid_ask_spread_pct DOUBLE,
            feat_order_book_depth_ratio DOUBLE,
            feat_large_order_imbalance DOUBLE,
            ob_spread_bps DOUBLE,
            -- Bot accumulation (4h pre-spike)
            bot_buy_volume_usd DOUBLE,
            bot_sell_volume_usd DOUBLE,
            bot_net_usd DOUBLE,
            total_volume_usd DOUBLE,
            bot_net_pct DOUBLE,
            bot_buy_ratio DOUBLE,
            bot_burst_count INTEGER,
            total_trade_count INTEGER,
            -- Post-spike behavior
            price_post_15m DOUBLE,
            price_post_30m DOUBLE,
            price_post_60m DOUBLE,
            pct_post_15m DOUBLE,
            pct_post_30m DOUBLE,
            pct_post_60m DOUBLE,
            -- Candle data (JSON blob of 1m candles: 4h pre through 1h post)
            candles_json VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS symbol_stats (
            symbol VARCHAR PRIMARY KEY,
            spike_count INTEGER,
            avg_peak_gain DOUBLE,
            max_peak_gain DOUBLE,
            avg_bot_net_pct DOUBLE,
            avg_pre_price DOUBLE,
            last_spike_date DATE,
            is_repeat_spiker BOOLEAN
        )
    """)


def update_elite_movers(buffer_path: str = FEATURE_BUFFER, elite_path: str = ELITE_DB):
    """
    Detect new 50%+ movers and insert them into elite_movers.duckdb.
    Returns count of new events added.
    """
    spikes = detect_spikes(buffer_path)
    if not spikes:
        return 0

    conn = duckdb.connect(elite_path)
    _ensure_schema(conn)

    # Get next event_id
    max_id = conn.execute("SELECT COALESCE(MAX(event_id), -1) FROM elite_events").fetchone()[0]
    next_id = max_id + 1

    added = 0
    for spike in spikes:
        # Check for existing event: same symbol within 12h
        existing = conn.execute("""
            SELECT event_id, peak_gain_pct FROM elite_events
            WHERE symbol = $1
              AND ABS(EPOCH(start - $2::TIMESTAMP)) < 43200
            ORDER BY peak_gain_pct DESC LIMIT 1
        """, [spike['symbol'], spike['start']]).fetchall()

        if existing:
            old_id, old_gain = existing[0]
            if spike['peak_gain_pct'] > old_gain * 1.05:
                # Spike grew by 5%+ — update the existing entry with fresh data
                conn.execute("DELETE FROM elite_events WHERE event_id = ?", [old_id])
                logger.info(
                    f"[ELITE_MOVERS] Updating {spike['symbol']}: "
                    f"+{old_gain:.1f}% → +{spike['peak_gain_pct']:.1f}%"
                )
                use_id = old_id  # Reuse the old event_id
            else:
                continue
        else:
            use_id = next_id
            next_id += 1

        # Collect rich event data
        data = collect_event_data(
            buffer_path, spike['symbol'], spike['start'], spike['end'],
            spike['low_price'], spike['peak_price']
        )

        bot = data['bot']
        feat = data['features']
        post = data['post_prices']
        candles_json = json.dumps(data['candles']) if data['candles'] else None

        conn.execute("""
            INSERT INTO elite_events (
                event_id, symbol, start, "end", peak_gain_pct, low_price, peak_price,
                duration_hours,
                pre_avg_vol, pre_price_cv, pre_avg_price, pre_natr, pre_bb_width,
                during_avg_vol, during_total_vol, vol_multiplier,
                feat_rsi_14, feat_vpin, feat_natr, feat_bb_width,
                feat_volume_spike_ratio, feat_bid_ask_spread_pct,
                feat_order_book_depth_ratio, feat_large_order_imbalance,
                ob_spread_bps,
                bot_buy_volume_usd, bot_sell_volume_usd, bot_net_usd,
                total_volume_usd, bot_net_pct, bot_buy_ratio,
                bot_burst_count, total_trade_count,
                price_post_15m, price_post_30m, price_post_60m,
                pct_post_15m, pct_post_30m, pct_post_60m,
                candles_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            use_id,
            spike['symbol'],
            spike['start'],
            spike['end'],
            spike['peak_gain_pct'],
            spike['low_price'],
            spike['peak_price'],
            spike['duration_hours'],
            # Pre-spike
            data['pre_avg_price'],
            data['pre_price_cv'],
            data['pre_avg_price'],
            data['pre_natr'],
            data['pre_bb_width'],
            # During
            data['during_avg_vol'],
            data['during_total_vol'],
            data['vol_multiplier'],
            # Features
            feat.get('RSI_14'),
            feat.get('vpin'),
            feat.get('natr'),
            feat.get('BB_width'),
            feat.get('volume_zscore'),
            feat.get('bid_ask_spread_pct'),
            feat.get('order_book_depth_ratio'),
            feat.get('large_order_imbalance'),
            data['spread_pct'] * 100 if data['spread_pct'] else None,  # to bps
            # Bot
            bot['bot_buy_volume_usd'] if bot else None,
            bot['bot_sell_volume_usd'] if bot else None,
            bot['bot_net_usd'] if bot else None,
            bot['total_volume_usd'] if bot else None,
            bot['bot_net_pct'] if bot else None,
            bot['bot_buy_ratio'] if bot else None,
            bot['bot_burst_count'] if bot else None,
            bot['total_trade_count'] if bot else None,
            # Post-spike
            post.get('price_post_15m'),
            post.get('price_post_30m'),
            post.get('price_post_60m'),
            post.get('pct_post_15m'),
            post.get('pct_post_30m'),
            post.get('pct_post_60m'),
            # Candles
            candles_json,
        ])

        added += 1

        candle_count = len(data['candles']) if data['candles'] else 0
        bot_str = f" bot_net={bot['bot_net_pct']:+.1f}%" if bot else " (no trades)"
        post_str = ""
        if post.get('pct_post_60m') is not None:
            post_str = f" post_60m={post['pct_post_60m']:+.1f}%"
        logger.info(
            f"[ELITE_MOVERS] New 50%+ event: {spike['symbol']} "
            f"+{spike['peak_gain_pct']:.1f}% "
            f"(${spike['low_price']:.6f} → ${spike['peak_price']:.6f})"
            f"{bot_str}{post_str} [{candle_count} candles]"
        )

    # Rebuild symbol_stats
    if added > 0:
        conn.execute("DELETE FROM symbol_stats")
        conn.execute("""
            INSERT INTO symbol_stats
            SELECT
                symbol,
                COUNT(*) as spike_count,
                AVG(peak_gain_pct) as avg_peak_gain,
                MAX(peak_gain_pct) as max_peak_gain,
                AVG(bot_net_pct) as avg_bot_net_pct,
                AVG(pre_avg_price) as avg_pre_price,
                MAX(start::DATE) as last_spike_date,
                COUNT(*) >= 2 as is_repeat_spiker
            FROM elite_events
            GROUP BY symbol
        """)

    conn.close()
    return added


async def elite_mover_loop(buffer_path: str = FEATURE_BUFFER, elite_path: str = ELITE_DB, interval_hours: float = 1.0):
    """Async loop for running inside the collector."""
    import asyncio

    await asyncio.sleep(120)  # Wait 2 min after startup

    while True:
        try:
            added = update_elite_movers(buffer_path, elite_path)
            if added > 0:
                total = duckdb.connect(elite_path, read_only=True).execute(
                    "SELECT COUNT(*) FROM elite_events"
                ).fetchone()[0]
                logger.info(f"[ELITE_MOVERS] Added {added} new events (total: {total})")
        except Exception as e:
            logger.error(f"[ELITE_MOVERS] Update error: {e}")

        await asyncio.sleep(int(interval_hours * 3600))


if __name__ == "__main__":
    added = update_elite_movers()
    if added:
        print(f"Added {added} new elite mover events")
    else:
        print("No new 50%+ movers detected")

    conn = duckdb.connect(ELITE_DB, read_only=True)
    total = conn.execute("SELECT COUNT(*) FROM elite_events").fetchone()[0]
    latest = conn.execute("SELECT MAX(start)::DATE FROM elite_events").fetchone()[0]

    # Show a rich entry
    r = conn.execute("""
        SELECT symbol, peak_gain_pct, candles_json IS NOT NULL as has_candles,
               pct_post_15m, pct_post_30m, pct_post_60m,
               pre_natr, feat_natr, bot_net_pct, ob_spread_bps
        FROM elite_events WHERE candles_json IS NOT NULL
        ORDER BY start DESC LIMIT 5
    """).fetchall()
    print(f"\nTotal events: {total}, latest: {latest}")
    print(f"\nRecent rich events:")
    for row in r:
        print(f"  {row[0]:<15} +{row[1]:.0f}%  candles={row[2]}  "
              f"post: 15m={row[3]}  30m={row[4]}  60m={row[5]}  "
              f"natr={row[6]}  bot={row[8]}")
    conn.close()
