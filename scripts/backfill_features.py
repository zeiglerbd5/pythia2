#!/usr/bin/env python3
"""
Backfill missing features in research.duckdb by recomputing from OHLCV data.

Reads OHLCV candles for days with missing features, runs the feature calculation
pipeline (price, volume, microstructure indicators), and writes results back.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from src.features.price_indicators import calculate_price_features
from src.features.volume_indicators import calculate_volume_features
from src.features.microstructure import calculate_microstructure_features

RESEARCH_DB = "data/research.duckdb"

# Map from calculation output column names to DB schema column names
COLUMN_MAP = {
    "rsi": "rsi_14",
    "BB_width": "bb_width",
    "VWAP_distance": "vwap_distance_pct",
}

# Columns in the features table we can populate
DB_FEATURE_COLS = [
    "order_book_imbalance_l5", "roll_measure", "vpin",
    "volume", "volume_spike_ratio", "obv", "vroc",
    "rsi_14", "vwap", "vwap_std", "vwap_distance_pct", "atr", "natr",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "bid_ask_ratio", "weighted_mid_price", "large_bid_orders", "large_ask_orders",
    "bid_ask_spread_pct", "order_book_depth_ratio", "large_order_imbalance",
]


def compute_features_for_symbol(ohlcv_df: pd.DataFrame, ob_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute all features from OHLCV data for a single symbol.

    Args:
        ohlcv_df: DataFrame with DatetimeIndex and open/high/low/close/volume columns.
                  Should have enough history (100+ bars) for lookback indicators.
        ob_df: Optional orderbook stats DataFrame with best_bid, best_ask, spread, spread_bps columns.

    Returns:
        DataFrame with computed features, one row per OHLCV bar.
    """
    if len(ohlcv_df) < 20:
        return pd.DataFrame()

    open_ = ohlcv_df["open"]
    high = ohlcv_df["high"]
    low = ohlcv_df["low"]
    close = ohlcv_df["close"]
    volume = ohlcv_df["volume"]

    # No buy/sell volume split available historically — use 50/50
    buy_volume = volume * 0.5
    sell_volume = volume * 0.5

    # Price indicators (RSI, VWAP, ATR, Bollinger Bands)
    price_feats = calculate_price_features(open_, high, low, close, volume)

    # Volume indicators (OBV, volume spike, VROC)
    volume_feats = calculate_volume_features(close, volume)

    # Microstructure (Roll measure, VPIN, trade flow)
    roll_window = min(100, len(close) - 1)
    vpin_window = min(50, len(close) - 1)
    micro_feats = calculate_microstructure_features(
        close, buy_volume, sell_volume,
        order_book_imbalance=None,
        roll_window=max(roll_window, 2),
        vpin_window=max(vpin_window, 2),
    )

    all_feats = pd.concat([price_feats, volume_feats, micro_feats], axis=1)

    # Add volume from OHLCV
    all_feats["volume"] = volume

    # Add orderbook-derived features if available
    if ob_df is not None and not ob_df.empty:
        # Resample orderbook to match OHLCV timestamps (nearest)
        ob_resampled = ob_df.reindex(all_feats.index, method="nearest", tolerance=pd.Timedelta("2min"))
        if "best_bid" in ob_resampled.columns and "best_ask" in ob_resampled.columns:
            mid = (ob_resampled["best_bid"] + ob_resampled["best_ask"]) / 2
            all_feats["bid_ask_spread_pct"] = (ob_resampled["best_ask"] - ob_resampled["best_bid"]) / mid
            all_feats["weighted_mid_price"] = mid
        if "spread_bps" in ob_resampled.columns:
            all_feats["spread_bps"] = ob_resampled["spread_bps"]

    # Rename columns to match DB schema
    all_feats.rename(columns=COLUMN_MAP, inplace=True)

    return all_feats


def backfill_features():
    db = duckdb.connect(RESEARCH_DB)
    db.execute("SET memory_limit = '2GB'")
    db.execute("SET preserve_insertion_order = false")

    # Find days that have OHLCV but no features
    gap_days = db.execute("""
        WITH ohlcv_days AS (
            SELECT DATE_TRUNC('day', timestamp)::DATE as day, COUNT(*) as cnt
            FROM ohlcv
            WHERE timeframe = '5m'
            GROUP BY 1 HAVING cnt >= 100
        ),
        feat_days AS (
            SELECT DATE_TRUNC('day', timestamp)::DATE as day, COUNT(*) as cnt
            FROM features
            GROUP BY 1
        )
        SELECT o.day, o.cnt as ohlcv_cnt, COALESCE(f.cnt, 0) as feat_cnt
        FROM ohlcv_days o
        LEFT JOIN feat_days f ON o.day = f.day
        WHERE COALESCE(f.cnt, 0) < 100
        ORDER BY o.day
    """).fetchall()

    if not gap_days:
        print("No feature gaps found!")
        db.close()
        return

    print(f"Found {len(gap_days)} days with OHLCV but missing features")
    for d in gap_days:
        print(f"  {d[0]}: {d[1]:,} ohlcv candles, {d[2]:,} features")

    # Get all symbols
    symbols = [r[0] for r in db.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol").fetchall()]
    print(f"\nProcessing {len(symbols)} symbols...")

    total_written = 0

    for day_row in gap_days:
        day = day_row[0]
        # Load a wider window: 2 days before for lookback warmup
        window_start = day - timedelta(days=2)
        window_end = day + timedelta(days=1)

        print(f"\n--- {day} ---")
        day_written = 0

        batch_records = []

        for sym_idx, symbol in enumerate(symbols):
            # Load 5m OHLCV for this symbol with warmup window
            ohlcv = db.execute("""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND timeframe = '5m'
                  AND timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
            """, [symbol, window_start, window_end]).df()

            if len(ohlcv) < 30:
                continue

            ohlcv.set_index("timestamp", inplace=True)
            if ohlcv.index.tz is not None:
                ohlcv.index = ohlcv.index.tz_localize(None)

            # Load orderbook stats for enrichment
            ob = db.execute("""
                SELECT timestamp, best_bid, best_ask, spread, spread_bps
                FROM orderbook_stats
                WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
            """, [symbol, window_start, window_end]).df()

            if not ob.empty:
                ob.set_index("timestamp", inplace=True)
                if ob.index.tz is not None:
                    ob.index = ob.index.tz_localize(None)
            else:
                ob = None

            # Compute features
            try:
                feats = compute_features_for_symbol(ohlcv, ob)
            except Exception as e:
                logger.debug(f"Error computing features for {symbol} on {day}: {e}")
                continue

            if feats.empty:
                continue

            # Filter to only the target day (drop warmup rows)
            day_start = pd.Timestamp(day)
            day_end = pd.Timestamp(day + timedelta(days=1))
            feats = feats[(feats.index >= day_start) & (feats.index < day_end)]

            if feats.empty:
                continue

            # Build records for insertion
            for ts, row in feats.iterrows():
                record = {"symbol": symbol, "timestamp": ts, "timeframe": "5m"}
                for col in DB_FEATURE_COLS:
                    if col in row.index and pd.notna(row[col]):
                        record[col] = float(row[col])
                batch_records.append(record)

            day_written += len(feats)

            if (sym_idx + 1) % 50 == 0:
                print(f"  {sym_idx + 1}/{len(symbols)} symbols...", flush=True)

        # Batch insert all records for this day
        if batch_records:
            insert_df = pd.DataFrame(batch_records)

            # Ensure all DB columns exist in the dataframe
            for col in DB_FEATURE_COLS:
                if col not in insert_df.columns:
                    insert_df[col] = None

            cols = ["symbol", "timestamp", "timeframe"] + DB_FEATURE_COLS
            insert_df = insert_df[cols]

            db.register("batch_df", insert_df)

            update_cols = [c for c in DB_FEATURE_COLS]
            update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
            cols_str = ", ".join(cols)

            db.execute(f"""
                INSERT INTO features ({cols_str})
                SELECT {cols_str} FROM batch_df
                ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET {update_clause}
            """)
            db.unregister("batch_df")

            total_written += day_written
            print(f"  Wrote {day_written:,} features for {day}")

    print(f"\n{'='*50}")
    print(f"Total features backfilled: {total_written:,}")

    # Show updated coverage
    print("\nUpdated coverage (days with 100+ news):")
    coverage = db.execute("""
        WITH news_days AS (
            SELECT DATE_TRUNC('day', timestamp)::DATE as day
            FROM news_signals GROUP BY 1 HAVING COUNT(*) >= 100
        )
        SELECT n.day,
            (SELECT COUNT(*) FROM features WHERE DATE_TRUNC('day', timestamp)::DATE = n.day) as feat_cnt
        FROM news_days n
        ORDER BY n.day
    """).fetchall()
    for row in coverage:
        status = "OK" if row[1] > 1000 else f"LOW ({row[1]})"
        print(f"  {row[0]}: {row[1]:>8,} features  {status}")

    db.close()
    print("\nDone!")


if __name__ == "__main__":
    backfill_features()
