#!/usr/bin/env python3
"""
Build a consolidated research database from all available Pythia data sources.

Stitches together:
- /Volumes/LaCie/full_pythia.duckdb (Dec 2025 - Mar 17 2026)
- /Volumes/LaCie/Pythia_Archives/offload_20260407/pythia_march2026.duckdb (Mar 29 - Apr 6 2026)

Output: data/research.duckdb (~2-4GB)

Tables included:
- ohlcv: OHLCV candles (all timeframes)
- features: Computed indicators (RSI, VWAP, BB, OBV, VPIN, etc.)
- news_signals: News events with sentiment scores
- whale_transactions: Large whale movements
- orderbook_stats: Extracted summary stats from order book snapshots (no raw JSON)
"""

import duckdb
import time
import sys
from pathlib import Path

SOURCES = [
    ("/Volumes/LaCie/full_pythia.duckdb", "full_pythia"),
    ("/Volumes/LaCie/Pythia_Archives/offload_20260407/pythia_march2026.duckdb", "march2026"),
]

OUTPUT_DB = "data/research.duckdb"


def fmt(n):
    return f"{n:,}"


def build_research_db():
    # Remove old output if exists
    output_path = Path(OUTPUT_DB)
    if output_path.exists():
        print(f"Removing existing {OUTPUT_DB}...")
        output_path.unlink()
    for suffix in [".wal", "-shm"]:
        p = output_path.with_suffix(output_path.suffix + suffix)
        if p.exists():
            p.unlink()

    out = duckdb.connect(str(output_path))

    # Tune for large operations on constrained machine
    out.execute("SET memory_limit = '2GB'")
    out.execute("SET threads = 2")
    out.execute("SET preserve_insertion_order = false")

    # ── 1. OHLCV ──────────────────────────────────────────────────────
    print("\n═══ OHLCV ═══")
    out.execute("""
        CREATE TABLE ohlcv (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            timeframe VARCHAR NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume DOUBLE NOT NULL,
            num_trades INTEGER,
            PRIMARY KEY (symbol, timestamp, timeframe)
        )
    """)

    for db_path, label in SOURCES:
        print(f"  Loading from {label}...")
        t0 = time.time()
        out.execute(f"ATTACH '{db_path}' AS src (READ_ONLY)")
        # Get distinct symbols and batch to avoid OOM
        symbols = [r[0] for r in out.execute("SELECT DISTINCT symbol FROM src.ohlcv ORDER BY symbol").fetchall()]
        batch_size = 20
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            placeholders = ", ".join(f"'{s}'" for s in batch)
            out.execute(f"""
                INSERT INTO ohlcv
                SELECT symbol, timestamp, timeframe, open, high, low, close, volume, num_trades
                FROM src.ohlcv
                WHERE symbol IN ({placeholders})
                ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high,
                    low = EXCLUDED.low, close = EXCLUDED.close,
                    volume = EXCLUDED.volume, num_trades = EXCLUDED.num_trades
            """)
            if (i // batch_size) % 5 == 0:
                print(f"    {i+len(batch)}/{len(symbols)} symbols...", flush=True)
        out.execute("DETACH src")
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

    count = out.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
    minmax = out.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv").fetchone()
    print(f"  Total: {fmt(count)} rows | {minmax[0]} → {minmax[1]}")

    # ── 2. Features ───────────────────────────────────────────────────
    print("\n═══ Features ═══")
    out.execute("""
        CREATE TABLE features (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            timeframe VARCHAR NOT NULL DEFAULT '5m',
            order_book_imbalance_l5 DOUBLE,
            roll_measure DOUBLE,
            vpin DOUBLE,
            volume DOUBLE,
            volume_spike_ratio DOUBLE,
            obv DOUBLE,
            vroc DOUBLE,
            rsi_14 DOUBLE,
            vwap DOUBLE,
            vwap_std DOUBLE,
            vwap_distance_pct DOUBLE,
            atr DOUBLE,
            natr DOUBLE,
            bb_upper DOUBLE,
            bb_middle DOUBLE,
            bb_lower DOUBLE,
            bb_width DOUBLE,
            bid_ask_ratio DOUBLE,
            weighted_mid_price DOUBLE,
            large_bid_orders INTEGER,
            large_ask_orders INTEGER,
            bid_ask_spread_pct DOUBLE,
            order_book_depth_ratio DOUBLE,
            large_order_imbalance DOUBLE,
            PRIMARY KEY (symbol, timestamp, timeframe)
        )
    """)

    feature_cols = [
        "symbol", "timestamp", "timeframe",
        "order_book_imbalance_l5", "roll_measure", "vpin",
        "volume", "volume_spike_ratio", "obv", "vroc",
        "rsi_14", "vwap", "vwap_std", "vwap_distance_pct", "atr", "natr",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "bid_ask_ratio", "weighted_mid_price", "large_bid_orders", "large_ask_orders",
        "bid_ask_spread_pct", "order_book_depth_ratio", "large_order_imbalance",
    ]
    update_cols = [c for c in feature_cols if c not in ("symbol", "timestamp", "timeframe")]
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    cols_str = ", ".join(feature_cols)

    for db_path, label in SOURCES:
        print(f"  Loading from {label}...")
        t0 = time.time()
        out.execute(f"ATTACH '{db_path}' AS src (READ_ONLY)")

        # Check which columns exist in source
        src_cols = set(r[0] for r in out.execute("SELECT column_name FROM information_schema.columns WHERE table_catalog='src' AND table_name='features'").fetchall())
        select_parts = []
        for c in feature_cols:
            if c in src_cols:
                select_parts.append(c)
            else:
                select_parts.append(f"NULL AS {c}")
        select_str = ", ".join(select_parts)

        out.execute(f"""
            INSERT INTO features ({cols_str})
            SELECT {select_str}
            FROM src.features
            ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET {update_clause}
        """)
        out.execute("DETACH src")
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

    count = out.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    minmax = out.execute("SELECT MIN(timestamp), MAX(timestamp) FROM features").fetchone()
    print(f"  Total: {fmt(count)} rows | {minmax[0]} → {minmax[1]}")

    # ── 3. News Signals ───────────────────────────────────────────────
    print("\n═══ News Signals ═══")
    out.execute("""
        CREATE TABLE news_signals (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            source VARCHAR NOT NULL,
            event_type VARCHAR NOT NULL,
            confidence DOUBLE NOT NULL,
            title TEXT,
            url TEXT,
            signal_hash VARCHAR,
            source_credibility DOUBLE,
            entity_certainty DOUBLE,
            event_priority DOUBLE,
            recency_score DOUBLE,
            engagement_score DOUBLE,
            sentiment_score DOUBLE
        )
    """)

    for db_path, label in SOURCES:
        print(f"  Loading from {label}...")
        t0 = time.time()
        out.execute(f"ATTACH '{db_path}' AS src (READ_ONLY)")

        # Check which columns exist in source
        src_cols = set(r[0] for r in out.execute("SELECT column_name FROM information_schema.columns WHERE table_catalog='src' AND table_name='news_signals'").fetchall())

        news_cols = [
            "symbol", "timestamp", "source", "event_type", "confidence",
            "title", "url", "signal_hash", "source_credibility",
            "entity_certainty", "event_priority", "recency_score",
            "engagement_score", "sentiment_score",
        ]
        select_parts = []
        for c in news_cols:
            if c in src_cols:
                select_parts.append(c)
            else:
                select_parts.append(f"NULL AS {c}")

        out.execute(f"""
            INSERT INTO news_signals
            SELECT {', '.join(select_parts)}
            FROM src.news_signals
        """)
        out.execute("DETACH src")
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

    # Deduplicate news_signals by signal_hash
    before = out.execute("SELECT COUNT(*) FROM news_signals").fetchone()[0]
    out.execute("""
        CREATE TABLE news_signals_deduped AS
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY COALESCE(signal_hash, symbol || timestamp::VARCHAR || title)
                ORDER BY timestamp
            ) AS rn
            FROM news_signals
        ) WHERE rn = 1
    """)
    out.execute("DROP TABLE news_signals")
    out.execute("ALTER TABLE news_signals_deduped RENAME TO news_signals")
    out.execute("ALTER TABLE news_signals DROP COLUMN rn")
    after = out.execute("SELECT COUNT(*) FROM news_signals").fetchone()[0]
    minmax = out.execute("SELECT MIN(timestamp), MAX(timestamp) FROM news_signals").fetchone()
    print(f"  Total: {fmt(after)} rows (deduped from {fmt(before)}) | {minmax[0]} → {minmax[1]}")

    # ── 4. Whale Transactions ─────────────────────────────────────────
    print("\n═══ Whale Transactions ═══")
    out.execute("""
        CREATE TABLE whale_transactions (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            amount_usd DOUBLE NOT NULL,
            subtype VARCHAR NOT NULL,
            from_name VARCHAR,
            to_name VARCHAR,
            blockchain VARCHAR,
            tx_hash VARCHAR
        )
    """)

    for db_path, label in SOURCES:
        print(f"  Loading from {label}...")
        t0 = time.time()
        out.execute(f"ATTACH '{db_path}' AS src (READ_ONLY)")
        out.execute("""
            INSERT INTO whale_transactions
            SELECT symbol, timestamp, amount_usd, subtype, from_name, to_name, blockchain, tx_hash
            FROM src.whale_transactions
        """)
        out.execute("DETACH src")
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

    # Deduplicate by tx_hash
    before = out.execute("SELECT COUNT(*) FROM whale_transactions").fetchone()[0]
    out.execute("""
        CREATE TABLE whale_deduped AS
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY COALESCE(tx_hash, symbol || timestamp::VARCHAR || amount_usd::VARCHAR)
                ORDER BY timestamp
            ) AS rn
            FROM whale_transactions
        ) WHERE rn = 1
    """)
    out.execute("DROP TABLE whale_transactions")
    out.execute("ALTER TABLE whale_deduped RENAME TO whale_transactions")
    out.execute("ALTER TABLE whale_transactions DROP COLUMN rn")
    after = out.execute("SELECT COUNT(*) FROM whale_transactions").fetchone()[0]
    minmax = out.execute("SELECT MIN(timestamp), MAX(timestamp) FROM whale_transactions").fetchone()
    print(f"  Total: {fmt(after)} rows (deduped from {fmt(before)}) | {minmax[0]} → {minmax[1]}")

    # ── 5. Order Book Stats (extracted from snapshots) ────────────────
    print("\n═══ Order Book Stats ═══")
    out.execute("""
        CREATE TABLE orderbook_stats (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            best_bid DOUBLE,
            best_ask DOUBLE,
            mid_price DOUBLE,
            spread DOUBLE,
            spread_bps DOUBLE,
            PRIMARY KEY (symbol, timestamp)
        )
    """)

    for db_path, label in SOURCES:
        print(f"  Loading from {label} (extracting numeric stats, skipping JSON)...")
        t0 = time.time()
        out.execute(f"ATTACH '{db_path}' AS src (READ_ONLY)")
        symbols = [r[0] for r in out.execute("SELECT DISTINCT symbol FROM src.order_book_snapshots ORDER BY symbol").fetchall()]
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            placeholders = ", ".join(f"'{s}'" for s in batch)
            out.execute(f"""
                INSERT INTO orderbook_stats
                SELECT symbol, timestamp, best_bid, best_ask, mid_price, spread, spread_bps
                FROM src.order_book_snapshots
                WHERE symbol IN ({placeholders})
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    best_bid = EXCLUDED.best_bid, best_ask = EXCLUDED.best_ask,
                    mid_price = EXCLUDED.mid_price, spread = EXCLUDED.spread,
                    spread_bps = EXCLUDED.spread_bps
            """)
            if (i // batch_size) % 5 == 0:
                print(f"    {i+len(batch)}/{len(symbols)} symbols...", flush=True)
        out.execute("DETACH src")
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s")

    count = out.execute("SELECT COUNT(*) FROM orderbook_stats").fetchone()[0]
    minmax = out.execute("SELECT MIN(timestamp), MAX(timestamp) FROM orderbook_stats").fetchone()
    print(f"  Total: {fmt(count)} rows | {minmax[0]} → {minmax[1]}")

    # ── 6. Create useful indexes ──────────────────────────────────────
    print("\n═══ Creating Indexes ═══")
    out.execute("CREATE INDEX idx_ohlcv_symbol ON ohlcv(symbol)")
    out.execute("CREATE INDEX idx_features_symbol ON features(symbol)")
    out.execute("CREATE INDEX idx_news_symbol_ts ON news_signals(symbol, timestamp)")
    out.execute("CREATE INDEX idx_whale_symbol_ts ON whale_transactions(symbol, timestamp)")
    out.execute("CREATE INDEX idx_whale_subtype ON whale_transactions(subtype)")
    out.execute("CREATE INDEX idx_ob_symbol ON orderbook_stats(symbol)")
    print("  Done")

    # ── 7. Summary ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RESEARCH DATABASE SUMMARY")
    print("═" * 60)
    tables = out.execute("SHOW TABLES").fetchall()
    for t in tables:
        name = t[0]
        count = out.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        minmax = out.execute(f'SELECT MIN(timestamp), MAX(timestamp) FROM "{name}"').fetchone()
        syms = out.execute(f'SELECT COUNT(DISTINCT symbol) FROM "{name}"').fetchone()[0]
        print(f"  {name:25s} {fmt(count):>15s} rows | {syms:>4d} symbols | {minmax[0]} → {minmax[1]}")

    out.close()

    # Final file size
    size_gb = output_path.stat().st_size / (1024**3)
    print(f"\n  Output: {OUTPUT_DB} ({size_gb:.2f} GB)")
    print("  Done!")


if __name__ == "__main__":
    build_research_db()
