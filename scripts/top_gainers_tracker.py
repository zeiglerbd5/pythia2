#!/usr/bin/env python3
"""
Top Gainers Tracker - Tracks top 50 gaining coins on Coinbase.
"""

import sqlite3
import requests
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "top_gainers.db"

def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            rank INTEGER NOT NULL,
            change_24h REAL NOT NULL,
            price REAL,
            volume_24h REAL,
            open_24h REAL,
            high_24h REAL,
            low_24h REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON snapshots(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_symbol ON snapshots(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_rank ON snapshots(rank)")
    conn.commit()

def fetch_all_gainers():
    try:
        r = requests.get("https://api.exchange.coinbase.com/products", timeout=30)
        r.raise_for_status()
        products = r.json()
    except Exception as e:
        print(f"Error fetching products: {e}")
        return []

    usd_pairs = [p['id'] for p in products if p.get('quote_currency') == 'USD' and p.get('status') == 'online']
    print(f"Fetching stats for {len(usd_pairs)} USD pairs...")

    gainers = []
    for i, symbol in enumerate(usd_pairs):
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(usd_pairs)}...")
        try:
            r = requests.get(f"https://api.exchange.coinbase.com/products/{symbol}/stats", timeout=5)
            if r.status_code != 200:
                continue
            stats = r.json()
            open_price = float(stats.get('open', 0))
            last_price = float(stats.get('last', 0))
            if open_price > 0:
                change_24h = (last_price - open_price) / open_price * 100
                gainers.append({
                    'symbol': symbol, 'change_24h': change_24h, 'price': last_price,
                    'volume_24h': float(stats.get('volume', 0)), 'open_24h': open_price,
                    'high_24h': float(stats.get('high', 0)), 'low_24h': float(stats.get('low', 0)),
                })
        except Exception:
            continue

    gainers.sort(key=lambda x: x['change_24h'], reverse=True)
    return gainers

def save_snapshot(conn, gainers, top_n=50):
    timestamp = datetime.now(timezone.utc).isoformat()
    for rank, gainer in enumerate(gainers[:top_n], 1):
        conn.execute("""
            INSERT INTO snapshots (timestamp, symbol, rank, change_24h, price, volume_24h, open_24h, high_24h, low_24h)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, gainer['symbol'], rank, gainer['change_24h'], gainer['price'],
              gainer['volume_24h'], gainer['open_24h'], gainer['high_24h'], gainer['low_24h']))
    conn.commit()
    return timestamp

def view_latest(conn, top_n=50):
    result = conn.execute("SELECT MAX(timestamp) FROM snapshots").fetchone()
    if not result or not result[0]:
        print("No snapshots found.")
        return
    latest_ts = result[0]
    print(f"\n{'='*60}\nTOP {top_n} GAINERS (24h)\nSnapshot: {latest_ts}\n{'='*60}\n")
    rows = conn.execute("SELECT rank, symbol, change_24h, price, volume_24h FROM snapshots WHERE timestamp = ? ORDER BY rank LIMIT ?", (latest_ts, top_n)).fetchall()
    print(f"{'Rank':<6} {'Symbol':<15} {'Change':<10} {'Price':<12} {'Volume 24h':<15}")
    print("-" * 60)
    for rank, symbol, change, price, volume in rows:
        print(f"{rank:<6} {symbol:<15} {change:>+7.1f}%   ${price:<10.4f} ${volume:>12,.0f}")

def view_history(conn, symbol):
    rows = conn.execute("SELECT timestamp, rank, change_24h, price FROM snapshots WHERE symbol = ? ORDER BY timestamp DESC LIMIT 50", (symbol,)).fetchall()
    if not rows:
        print(f"No history found for {symbol}")
        return
    print(f"\n{'='*60}\nHISTORY: {symbol}\n{'='*60}\n")
    print(f"{'Timestamp':<25} {'Rank':<6} {'Change':<10} {'Price':<12}")
    print("-" * 60)
    for ts, rank, change, price in rows:
        print(f"{ts[:19].replace('T', ' '):<25} {rank:<6} {change:>+7.1f}%   ${price:<10.4f}")

def view_movers(conn):
    result = conn.execute("SELECT DISTINCT timestamp FROM snapshots ORDER BY timestamp DESC LIMIT 2").fetchall()
    if len(result) < 2:
        print("Need at least 2 snapshots to show movers.")
        return
    latest_ts, prev_ts = result[0][0], result[1][0]
    latest = {row[0]: row[1] for row in conn.execute("SELECT symbol, rank FROM snapshots WHERE timestamp = ?", (latest_ts,)).fetchall()}
    prev = {row[0]: row[1] for row in conn.execute("SELECT symbol, rank FROM snapshots WHERE timestamp = ?", (prev_ts,)).fetchall()}
    movers = [(symbol, new_rank, prev.get(symbol, 999), prev.get(symbol, 999) - new_rank) for symbol, new_rank in latest.items() if prev.get(symbol, 999) != new_rank]
    movers.sort(key=lambda x: x[3], reverse=True)
    print(f"\n{'='*60}\nRANK MOVERS\nFrom: {prev_ts[:19]}\nTo:   {latest_ts[:19]}\n{'='*60}\n")
    print("UP:")
    for symbol, new_rank, old_rank, movement in movers[:10]:
        if movement > 0:
            print(f"  {symbol:<15} #{new_rank} (was {old_rank if old_rank < 999 else 'NEW'}, +{movement})")
    print("\nDOWN:")
    for symbol, new_rank, old_rank, movement in reversed(movers[-10:]):
        if movement < 0:
            print(f"  {symbol:<15} #{new_rank} (was #{old_rank}, {movement})")

def run_daemon(interval_minutes=5):
    print(f"Starting daemon mode (updating every {interval_minutes} minutes)\nPress Ctrl+C to stop\n")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    try:
        while True:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching top gainers...")
            gainers = fetch_all_gainers()
            if gainers:
                timestamp = save_snapshot(conn, gainers)
                print(f"Saved snapshot at {timestamp}\n\nTop 10:")
                for i, g in enumerate(gainers[:10], 1):
                    print(f"  {i}. {g['symbol']:<15} {g['change_24h']:>+7.1f}%")
            print(f"\nSleeping {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        print("\nStopping daemon...")
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Track top gaining coins on Coinbase")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in minutes")
    parser.add_argument("--view", action="store_true", help="View latest top gainers")
    parser.add_argument("--history", type=str, help="View history for a specific symbol")
    parser.add_argument("--movers", action="store_true", help="Show coins moving up/down")
    parser.add_argument("--top", type=int, default=50, help="Number of top gainers to track")
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.daemon:
        run_daemon(args.interval)
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    try:
        if args.view:
            view_latest(conn, args.top)
        elif args.history:
            view_history(conn, args.history)
        elif args.movers:
            view_movers(conn)
        else:
            print("Fetching current top gainers...")
            gainers = fetch_all_gainers()
            if gainers:
                timestamp = save_snapshot(conn, gainers, args.top)
                print(f"\nSaved {args.top} top gainers at {timestamp}\n\n{'='*60}\nTOP {min(20, args.top)} GAINERS (24h)\n{'='*60}\n")
                for i, g in enumerate(gainers[:20], 1):
                    print(f"{i:>3}. {g['symbol']:<15} {g['change_24h']:>+7.1f}%  @ ${g['price']:.4f}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
