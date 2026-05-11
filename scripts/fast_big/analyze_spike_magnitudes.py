#!/usr/bin/env python3
"""
Analyze Spike Magnitudes - Whale vs Guppy Feature Analysis (FAST VERSION)

Uses SQL joins instead of Python loops for speed.

Usage:
    python scripts/fast_big/analyze_spike_magnitudes.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import duckdb
from loguru import logger
from scipy import stats


def main():
    logger.info("=" * 80)
    logger.info("SPIKE MAGNITUDE ANALYSIS - WHALE vs GUPPY (FAST SQL VERSION)")
    logger.info("=" * 80)

    DB_PATH = 'market_data_analysis.duckdb'
    START_DATE = '2025-10-06'
    END_DATE = '2025-10-20'

    conn = duckdb.connect(DB_PATH, read_only=True)

    # === STEP 1: Find spikes using SQL with window functions ===
    logger.info("Finding spikes using SQL window functions...")

    spike_query = f"""
    WITH price_data AS (
        SELECT
            symbol,
            timestamp,
            close,
            volume,
            -- Forward looking 30-min max price
            MAX(close) OVER (
                PARTITION BY symbol
                ORDER BY timestamp
                ROWS BETWEEN 1 FOLLOWING AND 30 FOLLOWING
            ) as future_max_price,
            -- Trailing 10-min avg volume
            AVG(volume) OVER (
                PARTITION BY symbol
                ORDER BY timestamp
                ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
            ) as pre_volume_avg
        FROM candles
        WHERE timestamp >= '{START_DATE}'
        AND timestamp <= '{END_DATE}'
    ),
    spikes AS (
        SELECT
            symbol,
            timestamp as spike_start,
            close as start_price,
            future_max_price,
            (future_max_price - close) / NULLIF(close, 0) as magnitude,
            volume,
            pre_volume_avg,
            volume / NULLIF(pre_volume_avg, 0) as volume_ratio,
            -- Row number to filter overlapping spikes
            ROW_NUMBER() OVER (
                PARTITION BY symbol,
                -- Group by 30-minute windows to avoid overlap
                DATE_TRUNC('hour', timestamp) + INTERVAL (EXTRACT(MINUTE FROM timestamp)::INT / 30 * 30) MINUTE
                ORDER BY (future_max_price - close) / NULLIF(close, 0) DESC
            ) as rn
        FROM price_data
        WHERE close > 0
        AND future_max_price > close * 1.03  -- At least 3% gain
    )
    SELECT
        symbol,
        spike_start,
        start_price,
        future_max_price,
        magnitude,
        volume_ratio,
        CASE
            WHEN magnitude >= 0.20 THEN 'moby'
            WHEN magnitude >= 0.10 THEN 'whale'
            WHEN magnitude >= 0.05 THEN 'marlin'
            ELSE 'guppy'
        END as category
    FROM spikes
    WHERE rn = 1  -- Keep only best spike per 30-min window
    ORDER BY symbol, spike_start
    """

    spikes_df = conn.execute(spike_query).fetchdf()
    logger.info(f"Found {len(spikes_df):,} spikes")

    # Category distribution
    category_counts = spikes_df['category'].value_counts()
    logger.info("")
    logger.info("Spike distribution:")
    for cat in ['guppy', 'marlin', 'whale', 'moby']:
        count = category_counts.get(cat, 0)
        pct = count / len(spikes_df) * 100
        logger.info(f"  {cat.upper():8s}: {count:5d} ({pct:5.1f}%)")

    # === STEP 2: Get features at T-offset for whale vs guppy comparison ===
    logger.info("")
    logger.info("=" * 80)
    logger.info("FEATURE ANALYSIS: SPARK ZONE (T-1 to T-10 before spike)")
    logger.info("=" * 80)

    whale_moby = spikes_df[spikes_df['category'].isin(['whale', 'moby'])]
    guppy = spikes_df[spikes_df['category'] == 'guppy']

    logger.info(f"Comparing {len(whale_moby)} whale/moby vs {len(guppy)} guppy spikes")

    # Sample guppies to match whale count (for balanced comparison)
    guppy_sample = guppy.sample(min(len(guppy), len(whale_moby) * 2), random_state=42)

    feature_cols = ['returns', 'RSI_14', 'BB_width', 'volume_zscore', 'volume_roc',
                    'VWAP_distance', 'NATR', 'order_flow_imbalance', 'buy_sell_ratio',
                    'vpin', 'bid_ask_spread_pct', 'order_book_depth_ratio']

    for offset in [1, 2, 3, 5, 10]:
        logger.info(f"\n=== T-{offset} (minutes before spike) ===")

        # Build query to get features at spike_start - offset minutes
        # For whale/moby spikes
        whale_timestamps = whale_moby[['symbol', 'spike_start']].copy()
        whale_timestamps['target_time'] = whale_timestamps['spike_start'] - pd.Timedelta(minutes=offset)

        # Create temp tables and query
        conn.execute("DROP TABLE IF EXISTS temp_whale_times")
        conn.execute("CREATE TEMP TABLE temp_whale_times AS SELECT * FROM whale_timestamps")

        whale_features_query = f"""
        SELECT f.*
        FROM features f
        JOIN temp_whale_times t
        ON f.symbol = t.symbol
        AND f.timestamp = t.target_time
        """

        try:
            whale_feat_df = conn.execute(whale_features_query).fetchdf()
        except:
            whale_feat_df = pd.DataFrame()

        # For guppy spikes
        guppy_timestamps = guppy_sample[['symbol', 'spike_start']].copy()
        guppy_timestamps['target_time'] = guppy_timestamps['spike_start'] - pd.Timedelta(minutes=offset)

        conn.execute("DROP TABLE IF EXISTS temp_guppy_times")
        conn.execute("CREATE TEMP TABLE temp_guppy_times AS SELECT * FROM guppy_timestamps")

        guppy_features_query = f"""
        SELECT f.*
        FROM features f
        JOIN temp_guppy_times t
        ON f.symbol = t.symbol
        AND f.timestamp = t.target_time
        """

        try:
            guppy_feat_df = conn.execute(guppy_features_query).fetchdf()
        except:
            guppy_feat_df = pd.DataFrame()

        logger.info(f"Got {len(whale_feat_df)} whale features, {len(guppy_feat_df)} guppy features")

        if len(whale_feat_df) < 10 or len(guppy_feat_df) < 10:
            logger.warning("Not enough data for comparison")
            continue

        # Statistical comparison
        results = []
        for col in feature_cols:
            if col not in whale_feat_df.columns or col not in guppy_feat_df.columns:
                continue

            whale_vals = whale_feat_df[col].dropna().values
            guppy_vals = guppy_feat_df[col].dropna().values

            if len(whale_vals) > 10 and len(guppy_vals) > 10:
                t_stat, p_value = stats.ttest_ind(whale_vals, guppy_vals)
                whale_mean = np.mean(whale_vals)
                guppy_mean = np.mean(guppy_vals)
                whale_std = np.std(whale_vals)
                guppy_std = np.std(guppy_vals)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt((whale_std**2 + guppy_std**2) / 2)
                cohens_d = (whale_mean - guppy_mean) / pooled_std if pooled_std > 0 else 0

                diff_pct = (whale_mean - guppy_mean) / abs(guppy_mean) * 100 if guppy_mean != 0 else 0

                results.append({
                    'feature': col,
                    'whale_mean': whale_mean,
                    'guppy_mean': guppy_mean,
                    'diff_pct': diff_pct,
                    'p_value': p_value,
                    'cohens_d': cohens_d
                })

        results.sort(key=lambda x: x['p_value'])

        logger.info(f"{'Feature':<22s} {'Whale':>10s} {'Guppy':>10s} {'Diff%':>8s} {'p-val':>8s} {'d':>6s} {'Sig':>4s}")
        logger.info("-" * 75)

        for r in results:
            sig = "***" if r['p_value'] < 0.001 else ("**" if r['p_value'] < 0.01 else ("*" if r['p_value'] < 0.05 else ""))
            d_str = f"{r['cohens_d']:>5.2f}" if abs(r['cohens_d']) < 10 else f"{r['cohens_d']:>5.1f}"
            logger.info(
                f"{r['feature']:<22s} "
                f"{r['whale_mean']:>10.3f} "
                f"{r['guppy_mean']:>10.3f} "
                f"{r['diff_pct']:>7.1f}% "
                f"{r['p_value']:>8.4f} "
                f"{d_str} "
                f"{sig:>4s}"
            )

    # === STEP 3: Top whale/moby examples ===
    logger.info("")
    logger.info("=" * 80)
    logger.info("TOP 20 WHALE/MOBY SPIKES (Sorted by Magnitude)")
    logger.info("=" * 80)

    top_spikes = whale_moby.nlargest(20, 'magnitude')

    logger.info(f"{'Symbol':<15s} {'Start Time':<22s} {'Mag':>8s} {'Vol Ratio':>10s}")
    logger.info("-" * 60)

    for _, spike in top_spikes.iterrows():
        vol_str = f"{spike['volume_ratio']:.1f}x" if pd.notna(spike['volume_ratio']) else "N/A"
        logger.info(
            f"{spike['symbol']:<15s} "
            f"{str(spike['spike_start']):<22s} "
            f"{spike['magnitude']*100:>7.1f}% "
            f"{vol_str:>10s}"
        )

    # === Summary ===
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY & RECOMMENDATIONS")
    logger.info("=" * 80)
    logger.info(f"Total spikes analyzed: {len(spikes_df):,}")
    logger.info(f"Whales (10-20%): {category_counts.get('whale', 0)}")
    logger.info(f"Mobys (20%+): {category_counts.get('moby', 0)}")
    logger.info("")
    logger.info("Features with *** (p < 0.001) are STRONG discriminators")
    logger.info("Features with |Cohen's d| > 0.5 have MEANINGFUL effect size")
    logger.info("")
    logger.info("Use these insights to build WhaleSparkTargetGenerator:")
    logger.info("1. Label T-10 to T-1 candles before whale/moby spikes")
    logger.info("2. Weight features by their discriminative power")
    logger.info("3. Train model to detect 'whale spark' pattern")

    conn.close()


if __name__ == "__main__":
    main()
