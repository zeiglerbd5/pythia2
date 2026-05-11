#!/usr/bin/env python3
"""
Find all 10%+ price spikes in research.duckdb and analyze precursor patterns.

Approach:
1. Scan 5m OHLCV for all 10%+ moves within rolling 6h windows
2. Deduplicate overlapping events (keep largest per symbol per 12h)
3. For each spike, pull the 6h pre-spike context: features, orderbook, news, whale
4. Categorize spikes by profile (fast/slow, volume-led/momentum-led, etc.)
5. Output analysis + CSV for further modeling

Targets the 43-day clean window in research.duckdb.
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import duckdb
import pandas as pd
import numpy as np
from datetime import timedelta
from collections import defaultdict

DB_PATH = "data/research.duckdb"
SPIKE_THRESHOLD = 0.10  # 10% minimum move
LOOKBACK_BARS = 72      # 6 hours of 5m candles
DEDUP_HOURS = 12        # Merge spikes within 12h for same symbol


def find_spikes(db):
    """Find all 10%+ moves using rolling 6h low-to-high."""
    print("Scanning for 10%+ spikes in 5m OHLCV data...")

    # For each symbol, compute rolling 6h low and check if any subsequent
    # price within that window is 10%+ higher
    spikes_df = db.execute(f"""
        WITH candles AS (
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE timeframe = '5m'
            ORDER BY symbol, timestamp
        ),
        with_rolling AS (
            SELECT *,
                MIN(low) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN {LOOKBACK_BARS} PRECEDING AND CURRENT ROW
                ) as rolling_low,
                MAX(high) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN CURRENT ROW AND {LOOKBACK_BARS} FOLLOWING
                ) as forward_high
            FROM candles
        ),
        spike_candidates AS (
            SELECT *,
                (high - rolling_low) / NULLIF(rolling_low, 0) as gain_from_low,
                (forward_high - low) / NULLIF(low, 0) as forward_gain
            FROM with_rolling
            WHERE rolling_low > 0
        )
        SELECT symbol, timestamp, open, high, low, close, volume,
               rolling_low, gain_from_low
        FROM spike_candidates
        WHERE gain_from_low >= {SPIKE_THRESHOLD}
        ORDER BY symbol, timestamp
    """).df()

    print(f"  Raw spike candles: {len(spikes_df):,}")
    return spikes_df


def deduplicate_spikes(spikes_df):
    """
    Group overlapping spike candles into discrete events.
    For each symbol, merge candles within DEDUP_HOURS into one event,
    keeping the highest gain.
    """
    events = []
    for symbol, group in spikes_df.groupby('symbol'):
        group = group.sort_values('timestamp')
        current_event = None

        for _, row in group.iterrows():
            ts = row['timestamp']

            if current_event is None:
                current_event = {
                    'symbol': symbol,
                    'start': ts,
                    'end': ts,
                    'peak_gain': row['gain_from_low'],
                    'peak_ts': ts,
                    'rolling_low': row['rolling_low'],
                    'peak_high': row['high'],
                    'candle_count': 1,
                }
            elif (ts - current_event['end']) <= timedelta(hours=DEDUP_HOURS):
                # Extend current event
                current_event['end'] = ts
                current_event['candle_count'] += 1
                if row['gain_from_low'] > current_event['peak_gain']:
                    current_event['peak_gain'] = row['gain_from_low']
                    current_event['peak_ts'] = ts
                    current_event['peak_high'] = row['high']
            else:
                # New event
                events.append(current_event)
                current_event = {
                    'symbol': symbol,
                    'start': ts,
                    'end': ts,
                    'peak_gain': row['gain_from_low'],
                    'peak_ts': ts,
                    'rolling_low': row['rolling_low'],
                    'peak_high': row['high'],
                    'candle_count': 1,
                }

        if current_event:
            events.append(current_event)

    events_df = pd.DataFrame(events)
    events_df['duration_hours'] = (events_df['end'] - events_df['start']).dt.total_seconds() / 3600
    events_df = events_df.sort_values('peak_gain', ascending=False).reset_index(drop=True)
    events_df['event_id'] = range(len(events_df))
    print(f"  Deduplicated to {len(events_df)} independent spike events")
    return events_df


def enrich_with_context(db, events_df):
    """
    For each spike, pull 6h pre-spike context from all tables.
    """
    print("Enriching spikes with pre-spike context...")
    enriched = []

    for idx, event in events_df.iterrows():
        symbol = event['symbol']
        spike_start = event['start']
        pre_start = spike_start - timedelta(hours=6)

        # --- Pre-spike OHLCV (6h before) ---
        pre_ohlcv = db.execute("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = '5m'
              AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp
        """, [symbol, pre_start, spike_start]).df()

        # --- Pre-spike features (closest to spike start) ---
        pre_features = db.execute("""
            SELECT *
            FROM features
            WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp DESC LIMIT 1
        """, [symbol, pre_start, spike_start]).df()

        # --- Pre-spike orderbook stats (last 6h) ---
        pre_ob = db.execute("""
            SELECT AVG(spread_bps) as avg_spread_bps,
                   MIN(spread_bps) as min_spread_bps,
                   MAX(spread_bps) as max_spread_bps,
                   COUNT(*) as ob_snapshots
            FROM orderbook_stats
            WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
        """, [symbol, pre_start, spike_start]).df()

        # --- News signals in 24h before spike ---
        news = db.execute("""
            SELECT COUNT(*) as news_count,
                   AVG(sentiment_score) as avg_sentiment,
                   AVG(confidence) as avg_confidence,
                   MAX(confidence) as max_confidence,
                   STRING_AGG(DISTINCT source, ', ') as news_sources
            FROM news_signals
            WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
        """, [symbol, spike_start - timedelta(hours=24), spike_start]).df()

        # --- Whale transactions in 24h before spike ---
        whales = db.execute("""
            SELECT COUNT(*) as whale_count,
                   SUM(amount_usd) as whale_total_usd,
                   SUM(CASE WHEN subtype = 'exchange_inflow' THEN amount_usd ELSE 0 END) as whale_inflow_usd,
                   SUM(CASE WHEN subtype = 'exchange_outflow' THEN amount_usd ELSE 0 END) as whale_outflow_usd,
                   SUM(CASE WHEN subtype = 'wallet_transfer' THEN amount_usd ELSE 0 END) as whale_wallet_usd
            FROM whale_transactions
            WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
        """, [symbol, spike_start - timedelta(hours=24), spike_start]).df()

        # --- Compute derived pre-spike metrics from OHLCV ---
        ctx = {
            'event_id': event['event_id'],
            'symbol': symbol,
            'spike_start': spike_start,
            'peak_ts': event['peak_ts'],
            'peak_gain_pct': event['peak_gain'] * 100,
            'duration_hours': event['duration_hours'],
            'candle_count': event['candle_count'],
        }

        if len(pre_ohlcv) >= 5:
            vol = pre_ohlcv['volume']
            close = pre_ohlcv['close']
            high = pre_ohlcv['high']
            low = pre_ohlcv['low']

            # Volume profile
            ctx['pre_avg_volume'] = vol.mean()
            ctx['pre_max_volume'] = vol.max()
            ctx['pre_volume_trend'] = vol.iloc[-12:].mean() / vol.iloc[:12].mean() if vol.iloc[:12].mean() > 0 else 1.0
            ctx['pre_volume_accel'] = vol.iloc[-6:].mean() / vol.iloc[-12:-6].mean() if len(vol) >= 12 and vol.iloc[-12:-6].mean() > 0 else 1.0

            # Price profile
            ctx['pre_price_range_pct'] = (close.max() - close.min()) / close.min() * 100 if close.min() > 0 else 0
            returns = close.pct_change().dropna()
            ctx['pre_volatility'] = returns.std() * 100  # pct
            ctx['pre_momentum_1h'] = (close.iloc[-1] / close.iloc[-12] - 1) * 100 if len(close) >= 12 else 0
            ctx['pre_momentum_3h'] = (close.iloc[-1] / close.iloc[-36] - 1) * 100 if len(close) >= 36 else 0

            # Price compression (vol last 1h / vol last 6h)
            if len(returns) >= 12:
                vol_1h = returns.iloc[-12:].std()
                vol_6h = returns.std()
                ctx['pre_price_compression'] = vol_1h / vol_6h if vol_6h > 0 else 1.0
            else:
                ctx['pre_price_compression'] = 1.0

            # Time of day
            ctx['spike_hour_utc'] = spike_start.hour if hasattr(spike_start, 'hour') else pd.Timestamp(spike_start).hour
            ctx['spike_dow'] = pd.Timestamp(spike_start).dayofweek  # 0=Mon, 6=Sun
        else:
            for k in ['pre_avg_volume', 'pre_max_volume', 'pre_volume_trend',
                       'pre_volume_accel', 'pre_price_range_pct', 'pre_volatility',
                       'pre_momentum_1h', 'pre_momentum_3h', 'pre_price_compression',
                       'spike_hour_utc', 'spike_dow']:
                ctx[k] = None

        # Features at spike start
        if not pre_features.empty:
            row = pre_features.iloc[0]
            for col in ['rsi_14', 'vpin', 'roll_measure', 'natr', 'bb_width',
                        'volume_spike_ratio', 'obv', 'order_book_imbalance_l5',
                        'bid_ask_spread_pct', 'order_book_depth_ratio', 'large_order_imbalance',
                        'vwap_distance_pct', 'atr']:
                ctx[f'feat_{col}'] = row[col] if col in row.index and pd.notna(row[col]) else None

        # Orderbook context
        if not pre_ob.empty:
            for col in pre_ob.columns:
                ctx[f'ob_{col}'] = pre_ob.iloc[0][col]

        # News context
        if not news.empty:
            for col in news.columns:
                ctx[f'news_{col}'] = news.iloc[0][col]

        # Whale context
        if not whales.empty:
            for col in whales.columns:
                ctx[f'whale_{col}'] = whales.iloc[0][col]

        enriched.append(ctx)

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(events_df)} events enriched...", flush=True)

    return pd.DataFrame(enriched)


def categorize_spikes(df):
    """
    Categorize spikes into types based on their precursor profile.
    """
    categories = []

    for _, row in df.iterrows():
        gain = row['peak_gain_pct']
        duration = row['duration_hours']
        vol_trend = row.get('pre_volume_trend', 1.0)
        compression = row.get('pre_price_compression', 1.0)
        momentum = row.get('pre_momentum_1h', 0)

        if duration <= 1 and gain >= 10:
            spike_type = 'fast_steep'
        elif duration > 6 and gain >= 20:
            spike_type = 'slow_large'
        elif vol_trend and vol_trend > 2.0:
            spike_type = 'volume_buildup'
        elif compression and compression < 0.5:
            spike_type = 'compression_breakout'
        elif momentum and momentum > 3:
            spike_type = 'momentum_continuation'
        elif gain < 15:
            spike_type = 'minor_spike'
        else:
            spike_type = 'other'

        categories.append(spike_type)

    df['spike_type'] = categories
    return df


def print_analysis(df):
    """Print comprehensive analysis of spike patterns."""
    print("\n" + "=" * 70)
    print("  SPIKE PATTERN ANALYSIS")
    print("=" * 70)

    print(f"\nTotal spike events: {len(df)}")
    print(f"Unique symbols: {df['symbol'].nunique()}")
    print(f"Date range: {df['spike_start'].min()} → {df['spike_start'].max()}")

    # Gain distribution
    print(f"\n--- Gain Distribution ---")
    for lo, hi, label in [(10, 15, '10-15%'), (15, 25, '15-25%'), (25, 50, '25-50%'),
                           (50, 100, '50-100%'), (100, 500, '100%+')]:
        n = len(df[(df['peak_gain_pct'] >= lo) & (df['peak_gain_pct'] < hi)])
        print(f"  {label:10s}: {n:4d} events")

    # Spike types
    print(f"\n--- Spike Types ---")
    type_stats = df.groupby('spike_type').agg(
        count=('event_id', 'count'),
        avg_gain=('peak_gain_pct', 'mean'),
        med_gain=('peak_gain_pct', 'median'),
        avg_duration=('duration_hours', 'mean'),
    ).sort_values('count', ascending=False)
    print(type_stats.to_string())

    # Pre-spike feature patterns by type
    print(f"\n--- Pre-Spike Feature Averages by Type ---")
    feature_cols = [c for c in df.columns if c.startswith('pre_') or c.startswith('feat_')]
    key_features = ['pre_volume_trend', 'pre_volume_accel', 'pre_price_compression',
                    'pre_volatility', 'pre_momentum_1h', 'feat_rsi_14', 'feat_vpin',
                    'feat_bb_width', 'feat_natr', 'feat_volume_spike_ratio']
    available = [f for f in key_features if f in df.columns]

    if available:
        by_type = df.groupby('spike_type')[available].mean()
        print(by_type.round(3).to_string())

    # Time of day
    print(f"\n--- Hour of Day (UTC) ---")
    if 'spike_hour_utc' in df.columns:
        hour_counts = df['spike_hour_utc'].value_counts().sort_index()
        for hour, count in hour_counts.items():
            bar = '#' * count
            print(f"  {int(hour):02d}:00  {count:3d}  {bar}")

    # Day of week
    print(f"\n--- Day of Week ---")
    if 'spike_dow' in df.columns:
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_counts = df['spike_dow'].value_counts().sort_index()
        for dow, count in dow_counts.items():
            bar = '#' * count
            print(f"  {dow_names[int(dow)]:3s}  {count:3d}  {bar}")

    # News/whale context
    print(f"\n--- News & Whale Context ---")
    has_news = df['news_news_count'].fillna(0) > 0
    has_whale = df['whale_whale_count'].fillna(0) > 0
    print(f"  Spikes with pre-event news: {has_news.sum()} ({has_news.mean()*100:.1f}%)")
    print(f"  Spikes with pre-event whale txns: {has_whale.sum()} ({has_whale.mean()*100:.1f}%)")

    if has_whale.sum() > 0:
        whale_spikes = df[has_whale]
        print(f"  Avg whale USD (when present): ${whale_spikes['whale_whale_total_usd'].mean():,.0f}")
        print(f"  Avg whale inflow: ${whale_spikes['whale_whale_inflow_usd'].mean():,.0f}")
        print(f"  Avg whale outflow: ${whale_spikes['whale_whale_outflow_usd'].mean():,.0f}")

    # Top spikes
    print(f"\n--- Top 20 Spikes ---")
    top = df.nlargest(20, 'peak_gain_pct')[
        ['symbol', 'spike_start', 'peak_gain_pct', 'duration_hours', 'spike_type',
         'pre_volume_trend', 'pre_price_compression']
    ]
    print(top.to_string(index=False))


def main():
    db = duckdb.connect(DB_PATH, read_only=True)

    # 1. Find raw spike candles
    spikes_raw = find_spikes(db)

    if spikes_raw.empty:
        print("No spikes found!")
        db.close()
        return

    # 2. Deduplicate into events
    events = deduplicate_spikes(spikes_raw)
    print(f"  Gain range: {events['peak_gain'].min()*100:.1f}% - {events['peak_gain'].max()*100:.1f}%")

    # 3. Enrich with context
    enriched = enrich_with_context(db, events)
    db.close()

    # 4. Categorize
    enriched = categorize_spikes(enriched)

    # 5. Analyze
    print_analysis(enriched)

    # 6. Save
    output_path = "data/spike_analysis.csv"
    enriched.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return enriched


if __name__ == "__main__":
    main()
