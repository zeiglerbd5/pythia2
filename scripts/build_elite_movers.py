#!/usr/bin/env python3
"""
Build elite_movers.duckdb - tracking 50%+ crypto price moves with bot accumulation metrics.

Strategy: Per-symbol queries with persistent connection + SQL-side aggregation
to minimize data transfer from the huge trade DBs.
"""

import duckdb
import datetime
import sys
import os
import time
from collections import defaultdict

# Paths
BIG_MOVERS_DB = "/Users/bz/Pythia2/data/big_movers.duckdb"
ELITE_DB = "/Users/bz/Pythia2/data/elite_movers.duckdb"
FULL_PYTHIA_DB = "/Volumes/LaCie/full_pythia.duckdb"
MARCH_DB = "/Volumes/LaCie/Pythia_Archives/offload_20260407/pythia_march2026.duckdb"

FULL_PYTHIA_CUTOFF = datetime.datetime(2026, 3, 17, 20, 0, 0)

# SQL to compute bot metrics entirely within DuckDB (no raw trade transfer)
BOT_METRICS_SQL = """
WITH windowed AS (
    SELECT
        timestamp,
        price,
        size,
        side,
        price * size as notional,
        time_bucket(INTERVAL '1 second', timestamp) as sec_bucket
    FROM trades
    WHERE symbol = $1
      AND timestamp >= $2
      AND timestamp < $3
),
second_stats AS (
    SELECT
        sec_bucket,
        count(*) as trade_count,
        count(DISTINCT round(price * size, 2)) as unique_sizes
    FROM windowed
    GROUP BY sec_bucket
),
bot_seconds AS (
    SELECT sec_bucket
    FROM second_stats
    WHERE trade_count >= 5 AND unique_sizes <= 3
),
bot_trades AS (
    SELECT
        w.notional,
        w.side
    FROM windowed w
    INNER JOIN bot_seconds b ON w.sec_bucket = b.sec_bucket
),
agg AS (
    SELECT
        coalesce(sum(CASE WHEN upper(side) = 'BUY' THEN notional ELSE 0 END), 0) as bot_buy_vol,
        coalesce(sum(CASE WHEN upper(side) != 'BUY' THEN notional ELSE 0 END), 0) as bot_sell_vol,
        count(*) as bot_trade_count
    FROM bot_trades
)
SELECT
    agg.bot_buy_vol,
    agg.bot_sell_vol,
    agg.bot_buy_vol - agg.bot_sell_vol as bot_net,
    (SELECT coalesce(sum(notional), 0) FROM windowed) as total_volume_usd,
    CASE WHEN (SELECT coalesce(sum(notional), 0) FROM windowed) > 0
         THEN (agg.bot_buy_vol - agg.bot_sell_vol) / (SELECT sum(notional) FROM windowed) * 100
         ELSE 0 END as bot_net_pct,
    CASE WHEN (agg.bot_buy_vol + agg.bot_sell_vol) > 0
         THEN agg.bot_buy_vol / (agg.bot_buy_vol + agg.bot_sell_vol)
         ELSE 0 END as bot_buy_ratio,
    (SELECT count(*) FROM bot_seconds) as bot_burst_count,
    (SELECT count(*) FROM windowed) as total_trade_count
FROM agg
"""


def main():
    print("=== Building elite_movers.duckdb ===\n")

    if os.path.exists(ELITE_DB):
        os.remove(ELITE_DB)
        print(f"Removed existing {ELITE_DB}")

    # Load elite events
    src = duckdb.connect(BIG_MOVERS_DB, read_only=True)
    events = src.sql("SELECT * FROM events WHERE peak_gain_pct >= 50 ORDER BY start").fetchall()
    col_names = [d[0] for d in src.sql("SELECT * FROM events LIMIT 0").description]
    schema = src.sql("DESCRIBE events").fetchall()
    src.close()
    print(f"Loaded {len(events)} elite events (50%+ gain)\n")

    eid_idx = col_names.index('event_id')
    symbol_idx = col_names.index('symbol')
    start_idx = col_names.index('start')

    # Group events by (db, symbol) to batch queries for same symbol
    # Key: (db_path, symbol) -> [(event_idx, event_start), ...]
    symbol_groups = defaultdict(list)
    for i, event in enumerate(events):
        symbol = event[symbol_idx]
        event_start = event[start_idx]
        window_start = event_start - datetime.timedelta(hours=4)
        db_path = FULL_PYTHIA_DB if window_start < FULL_PYTHIA_CUTOFF else MARCH_DB
        symbol_groups[(db_path, symbol)].append((i, event_start))

    n_fp = sum(1 for (db, _) in symbol_groups if db == FULL_PYTHIA_DB)
    n_m = sum(1 for (db, _) in symbol_groups if db == MARCH_DB)
    print(f"Unique (db, symbol) groups: {n_fp} in full_pythia, {n_m} in march2026")
    print(f"Total groups to query: {len(symbol_groups)}\n")

    # Create output DB
    out = duckdb.connect(ELITE_DB)
    type_map = {row[0]: row[1] for row in schema}
    create_cols = []
    for col in col_names:
        dtype = type_map.get(col, 'VARCHAR')
        if 'TIMESTAMP' in dtype:
            dtype = 'TIMESTAMP'
        create_cols.append(f'    "{col}" {dtype}')
    create_cols.extend([
        "    bot_buy_volume_usd DOUBLE",
        "    bot_sell_volume_usd DOUBLE",
        "    bot_net_usd DOUBLE",
        "    total_volume_usd DOUBLE",
        "    bot_net_pct DOUBLE",
        "    bot_buy_ratio DOUBLE",
        "    bot_burst_count INTEGER",
        "    total_trade_count INTEGER",
    ])
    out.sql(f"CREATE TABLE elite_events (\n{',\n'.join(create_cols)}\n)")

    # Process each (db, symbol) group
    # Keep connections open per-DB
    metrics_by_event_idx = {}
    connections = {}

    def get_conn(db_path):
        if db_path not in connections:
            connections[db_path] = duckdb.connect(db_path, read_only=True)
        return connections[db_path]

    total_groups = len(symbol_groups)
    t_start = time.time()

    # Sort groups to do all full_pythia first, then march (minimize connection switches)
    sorted_groups = sorted(symbol_groups.items(), key=lambda x: (0 if x[0][0] == FULL_PYTHIA_DB else 1, x[0][1]))

    for g_idx, ((db_path, symbol), event_list) in enumerate(sorted_groups):
        db_label = "FP" if db_path == FULL_PYTHIA_DB else "M26"

        # For this symbol, find the overall time range needed
        all_window_starts = [es - datetime.timedelta(hours=4) for _, es in event_list]
        all_window_ends = [es for _, es in event_list]
        min_start = min(all_window_starts)
        max_end = max(all_window_ends)

        t0 = time.time()
        elapsed_total = t0 - t_start
        rate = (g_idx / elapsed_total * 60) if elapsed_total > 0 and g_idx > 0 else 0
        eta = ((total_groups - g_idx) / rate) if rate > 0 else 0

        print(f"[{g_idx+1:3d}/{total_groups}] [{db_label}] {symbol:20s} "
              f"({len(event_list)} events) ", end="", flush=True)

        con = get_conn(db_path)

        # If multiple events for same symbol, we query the full range once
        # then run bot metrics for each event's specific window
        if len(event_list) == 1:
            # Single event - direct SQL aggregation
            event_idx, event_start = event_list[0]
            window_start = event_start - datetime.timedelta(hours=4)
            try:
                result = con.sql(BOT_METRICS_SQL, params=[symbol, window_start, event_start]).fetchone()
                if result and result[7] > 0:  # total_trade_count > 0
                    metrics_by_event_idx[event_idx] = {
                        'bot_buy_volume_usd': result[0],
                        'bot_sell_volume_usd': result[1],
                        'bot_net_usd': result[2],
                        'total_volume_usd': result[3],
                        'bot_net_pct': result[4],
                        'bot_buy_ratio': result[5],
                        'bot_burst_count': result[6],
                        'total_trade_count': result[7],
                    }
                    dt = time.time() - t0
                    m = metrics_by_event_idx[event_idx]
                    print(f"trades={m['total_trade_count']:,} bursts={m['bot_burst_count']} "
                          f"net={m['bot_net_pct']:+.1f}% ({dt:.0f}s) ETA:{eta:.0f}m")
                else:
                    dt = time.time() - t0
                    print(f"NO TRADES ({dt:.0f}s)")
            except Exception as e:
                dt = time.time() - t0
                print(f"ERROR: {e} ({dt:.0f}s)")
        else:
            # Multiple events - fetch raw trades for the range, process in Python
            try:
                rows = con.sql("""
                    SELECT timestamp, price, size, side
                    FROM trades
                    WHERE symbol = $1 AND timestamp >= $2 AND timestamp < $3
                    ORDER BY timestamp
                """, params=[symbol, min_start, max_end]).fetchall()

                dt = time.time() - t0
                print(f"fetched {len(rows):,} trades ({dt:.0f}s) -> ", end="", flush=True)

                # Assign to each event
                for event_idx, event_start in event_list:
                    ws = event_start - datetime.timedelta(hours=4)
                    we = event_start
                    event_trades = [(ts, p, s, sd) for ts, p, s, sd in rows if ws <= ts < we]

                    if event_trades:
                        total_trade_count = len(event_trades)
                        total_volume_usd = sum(t[1] * t[2] for t in event_trades)

                        second_buckets = defaultdict(list)
                        for ts, price, size, side in event_trades:
                            sec_key = ts.replace(microsecond=0)
                            second_buckets[sec_key].append((price, size, side))

                        bot_buy_vol = 0.0
                        bot_sell_vol = 0.0
                        bot_burst_count = 0
                        for bucket in second_buckets.values():
                            if len(bucket) < 5:
                                continue
                            unique_sizes = set(round(p * s, 2) for p, s, _ in bucket)
                            if len(unique_sizes) <= 3:
                                bot_burst_count += 1
                                for p, s, sd in bucket:
                                    n = p * s
                                    if sd and sd.upper() == 'BUY':
                                        bot_buy_vol += n
                                    else:
                                        bot_sell_vol += n

                        bot_net = bot_buy_vol - bot_sell_vol
                        metrics_by_event_idx[event_idx] = {
                            'bot_buy_volume_usd': bot_buy_vol,
                            'bot_sell_volume_usd': bot_sell_vol,
                            'bot_net_usd': bot_net,
                            'total_volume_usd': total_volume_usd,
                            'bot_net_pct': (bot_net / total_volume_usd * 100) if total_volume_usd > 0 else 0,
                            'bot_buy_ratio': (bot_buy_vol / (bot_buy_vol + bot_sell_vol)) if (bot_buy_vol + bot_sell_vol) > 0 else 0,
                            'bot_burst_count': bot_burst_count,
                            'total_trade_count': total_trade_count,
                        }

                print(f"{len(event_list)} events processed ETA:{eta:.0f}m")
                del rows

            except Exception as e:
                dt = time.time() - t0
                print(f"ERROR: {e} ({dt:.0f}s)")

    # Close trade DB connections
    for c in connections.values():
        c.close()

    total_extract_time = time.time() - t_start
    print(f"\nExtraction complete in {total_extract_time:.0f}s ({total_extract_time/60:.1f}m)")

    # Insert all events
    print("\nInserting events...")
    n_with_data = 0
    n_no_data = 0

    for i, event in enumerate(events):
        metrics = metrics_by_event_idx.get(i)
        values = list(event)
        if metrics:
            n_with_data += 1
            values.extend([
                metrics['bot_buy_volume_usd'],
                metrics['bot_sell_volume_usd'],
                metrics['bot_net_usd'],
                metrics['total_volume_usd'],
                metrics['bot_net_pct'],
                metrics['bot_buy_ratio'],
                metrics['bot_burst_count'],
                metrics['total_trade_count'],
            ])
        else:
            n_no_data += 1
            values.extend([None] * 8)

        placeholders = ', '.join(['$' + str(j+1) for j in range(len(values))])
        out.sql(f"INSERT INTO elite_events VALUES ({placeholders})", params=values)

    print(f"  {n_with_data} with trade data, {n_no_data} without")

    # symbol_stats
    print("\nBuilding symbol_stats table...")
    out.sql("""
        CREATE TABLE symbol_stats AS
        SELECT
            symbol,
            count(*) as spike_count,
            avg(peak_gain_pct) as avg_peak_gain,
            max(peak_gain_pct) as max_peak_gain,
            avg(bot_net_pct) as avg_bot_net_pct,
            avg(pre_avg_price) as avg_pre_price,
            max(cast("start" as date)) as last_spike_date,
            (count(*) >= 2) as is_repeat_spiker
        FROM elite_events
        GROUP BY symbol
        ORDER BY spike_count DESC
    """)
    n_symbols = out.sql("SELECT count(*) FROM symbol_stats").fetchone()[0]
    n_repeat = out.sql("SELECT count(*) FROM symbol_stats WHERE is_repeat_spiker").fetchone()[0]
    print(f"  {n_symbols} unique symbols, {n_repeat} repeat spikers")

    # daily_summary view
    out.sql("""
        CREATE VIEW daily_summary AS
        SELECT
            cast("start" as date) as event_date,
            count(*) as event_count,
            list(symbol ORDER BY "start") as symbols
        FROM elite_events
        GROUP BY 1
        ORDER BY 1
    """)

    # Summary
    print("\n=== FINAL SUMMARY ===")
    summary = out.sql("""
        SELECT
            count(*),
            count(bot_burst_count),
            round(avg(peak_gain_pct), 1),
            round(avg(bot_net_pct), 1),
            round(avg(bot_buy_ratio), 3),
            round(avg(bot_burst_count), 0),
            round(median(bot_burst_count), 0)
        FROM elite_events
    """).fetchone()
    print(f"  Total events:        {summary[0]}")
    print(f"  With trade data:     {summary[1]}")
    print(f"  Avg peak gain:       {summary[2]}%")
    print(f"  Avg bot net pct:     {summary[3]}%")
    print(f"  Avg bot buy ratio:   {summary[4]}")
    print(f"  Avg bot bursts:      {summary[5]}")
    print(f"  Median bot bursts:   {summary[6]}")

    print("\nTop repeat spikers:")
    rows = out.sql("""
        SELECT symbol, spike_count, round(avg_peak_gain, 1),
               round(avg_bot_net_pct, 1)
        FROM symbol_stats WHERE is_repeat_spiker
        ORDER BY spike_count DESC LIMIT 10
    """).fetchall()
    for r in rows:
        print(f"  {r[0]:20s} spikes={r[1]} avg_gain={r[2]}% bot_net={r[3]}%")

    n_days = out.sql('SELECT count(distinct cast("start" as date)) FROM elite_events').fetchone()[0]
    print(f"\nDaily event rate: {len(events) / n_days:.1f} events/day")

    out.close()
    total_time = time.time() - t_start
    print(f"\nDone! {ELITE_DB}")
    print(f"Size: {os.path.getsize(ELITE_DB) / 1024 / 1024:.1f} MB")
    print(f"Total: {total_time:.0f}s ({total_time/60:.1f}m)")


if __name__ == "__main__":
    main()
