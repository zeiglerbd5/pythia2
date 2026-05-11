"""
Engineer Features for Event Classifier

Computes ~30 summary features for each event (positive spike or negative sample).
Features focus on volatility, volume, momentum, and market structure.

Usage:
    python scripts/engineer_event_features.py

Output:
    data/event_features.parquet - Feature matrix for training
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger


# Configuration
EVENTS_PATH = '/Users/bz/Pythia2/data/spike_events.parquet'
NEGATIVES_PATH = '/Users/bz/Pythia2/data/negative_samples_v2.parquet'
DB_PATH = '/Users/bz/Pythia2/data/pythia.duckdb'
OUTPUT_PATH = '/Users/bz/Pythia2/data/event_features.parquet'

# Feature computation lookback (need enough history)
MIN_LOOKBACK_MINUTES = 1500  # ~25 hours of 1-min data


def compute_all_features(ohlcv_df: pd.DataFrame, target_idx: int) -> dict:
    """
    Compute all features at a specific index in the OHLCV dataframe.

    Returns dict of feature name -> value
    """
    if target_idx < MIN_LOOKBACK_MINUTES or target_idx >= len(ohlcv_df):
        return None

    # Get data up to target point (no lookahead)
    df = ohlcv_df.iloc[:target_idx + 1].copy()

    # Basic price data at target
    close = df['close'].iloc[-1]
    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    volume = df['volume'].iloc[-1]

    features = {}

    # ===== VOLATILITY FEATURES =====

    # Returns
    df['returns'] = df['close'].pct_change()

    # NATR (Normalized ATR) - KEY FEATURE
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean().iloc[-1]
    features['natr_14'] = (atr_14 / close * 100) if close > 0 else 0

    # Bollinger Band width
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_width = ((bb_middle.iloc[-1] + 2 * bb_std.iloc[-1]) - (bb_middle.iloc[-1] - 2 * bb_std.iloc[-1])) / bb_middle.iloc[-1]
    features['bb_width_20'] = bb_width if pd.notna(bb_width) else 0

    # Volatility ratios (short vs long term)
    vol_20 = df['returns'].rolling(20).std().iloc[-1]
    vol_60 = df['returns'].rolling(60).std().iloc[-1]
    vol_240 = df['returns'].rolling(240).std().iloc[-1]  # 4 hours

    features['vol_ratio_20_60'] = (vol_20 / vol_60) if pd.notna(vol_60) and vol_60 > 0 else 1
    features['vol_ratio_60_240'] = (vol_60 / vol_240) if pd.notna(vol_240) and vol_240 > 0 else 1

    # VOLATILITY ACCELERATION - KEY FEATURE
    # Rate of change of volatility (is vol increasing?)
    vol_20_series = df['returns'].rolling(20).std()
    vol_acceleration = (vol_20_series.iloc[-1] - vol_20_series.iloc[-60]) / (vol_20_series.iloc[-60] + 1e-10) if len(vol_20_series) > 60 else 0
    features['vol_acceleration'] = vol_acceleration if pd.notna(vol_acceleration) else 0

    # ===== VOLUME FEATURES =====

    # Volume vs moving averages
    vol_ma_20 = df['volume'].rolling(20).mean().iloc[-1]
    vol_ma_60 = df['volume'].rolling(60).mean().iloc[-1]

    features['volume_vs_ma20'] = (volume / vol_ma_20) if pd.notna(vol_ma_20) and vol_ma_20 > 0 else 1

    # Volume trend (is volume increasing?)
    vol_ma_6hr = df['volume'].rolling(360).mean()
    vol_trend = (vol_ma_20 - vol_ma_6hr.iloc[-1]) / (vol_ma_6hr.iloc[-1] + 1e-10) if len(vol_ma_6hr) > 0 and pd.notna(vol_ma_6hr.iloc[-1]) else 0
    features['volume_trend_6hr'] = vol_trend if pd.notna(vol_trend) else 0

    # OBV slope
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    obv_slope = (obv.iloc[-1] - obv.iloc[-60]) / 60 if len(obv) > 60 else 0
    features['obv_slope_1hr'] = obv_slope if pd.notna(obv_slope) else 0

    # VROC (Volume Rate of Change)
    vroc = (volume - df['volume'].iloc[-12]) / (df['volume'].iloc[-12] + 1e-10) if len(df) > 12 else 0
    features['vroc_12'] = vroc if pd.notna(vroc) else 0

    # VOLUME-PRICE DIVERGENCE - KEY FEATURE
    # Volume rising while price is range-bound
    price_range_60 = (df['close'].iloc[-60:].max() - df['close'].iloc[-60:].min()) / close if len(df) >= 60 else 0
    vol_change_60 = (vol_ma_20 - df['volume'].iloc[-60:-40].mean()) / (df['volume'].iloc[-60:-40].mean() + 1e-10) if len(df) >= 60 else 0

    # If price range is small (<2%) but volume is up (>30%), this is divergence
    vol_price_divergence = vol_change_60 if (price_range_60 < 0.02 and vol_change_60 > 0.3) else 0
    features['vol_price_divergence'] = vol_price_divergence if pd.notna(vol_price_divergence) else 0

    # ===== MOMENTUM FEATURES =====

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
    rs = gain / (loss + 1e-10)
    features['rsi_14'] = (100 - (100 / (1 + rs))) if pd.notna(rs) else 50

    # Returns at various horizons
    features['returns_1hr'] = (close / df['close'].iloc[-60] - 1) if len(df) > 60 else 0
    features['returns_6hr'] = (close / df['close'].iloc[-360] - 1) if len(df) > 360 else 0
    features['returns_12hr'] = (close / df['close'].iloc[-720] - 1) if len(df) > 720 else 0

    # Momentum (rate of change)
    features['momentum_5'] = (close / df['close'].iloc[-5] - 1) if len(df) > 5 else 0
    features['momentum_20'] = (close / df['close'].iloc[-20] - 1) if len(df) > 20 else 0

    # ===== PRICE POSITION FEATURES =====

    # Distance from 24hr high/low
    high_24hr = df['high'].iloc[-1440:].max() if len(df) >= 1440 else df['high'].max()
    low_24hr = df['low'].iloc[-1440:].min() if len(df) >= 1440 else df['low'].min()

    features['dist_from_24hr_high'] = (close - high_24hr) / high_24hr if high_24hr > 0 else 0
    features['dist_from_24hr_low'] = (close - low_24hr) / low_24hr if low_24hr > 0 else 0

    # Bollinger Band position (0 = at lower band, 1 = at upper band)
    bb_lower = bb_middle.iloc[-1] - 2 * bb_std.iloc[-1]
    bb_upper = bb_middle.iloc[-1] + 2 * bb_std.iloc[-1]
    bb_range = bb_upper - bb_lower
    features['bb_position'] = (close - bb_lower) / bb_range if pd.notna(bb_range) and bb_range > 0 else 0.5

    # ===== PRICE STRUCTURE FEATURES =====

    # High-low range (normalized)
    features['hl_range'] = (high - low) / close if close > 0 else 0

    # Body ratio average (how much of candle is body vs wick)
    body = (df['close'] - df['open']).abs()
    wick = df['high'] - df['low']
    body_ratio = body / (wick + 1e-10)
    features['body_ratio_avg_1hr'] = body_ratio.iloc[-60:].mean() if len(body_ratio) >= 60 else 0.5

    # Range compression (is price consolidating?)
    range_20 = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / close
    range_60 = (df['high'].rolling(60).max() - df['low'].rolling(60).min()) / close
    features['range_compression'] = (range_20.iloc[-1] / range_60.iloc[-1]) if pd.notna(range_60.iloc[-1]) and range_60.iloc[-1] > 0 else 1

    # ===== TEMPORAL FEATURES =====
    # Hour of day and day of week (cyclical encoding)
    ts = df['timestamp'].iloc[-1]
    if isinstance(ts, pd.Timestamp):
        hour = ts.hour
        dow = ts.dayofweek

        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    else:
        features['hour_sin'] = 0
        features['hour_cos'] = 1
        features['dow_sin'] = 0
        features['dow_cos'] = 1

    # Clean up any remaining NaN/inf
    for k, v in features.items():
        if pd.isna(v) or np.isinf(v):
            features[k] = 0

    return features


def main():
    logger.info("=" * 60)
    logger.info("ENGINEERING EVENT FEATURES")
    logger.info("=" * 60)

    # Load events and negatives
    logger.info("Loading spike events and negative samples...")
    events_df = pd.read_parquet(EVENTS_PATH)
    negatives_df = pd.read_parquet(NEGATIVES_PATH)

    logger.info(f"Spike events (positives): {len(events_df)}")
    logger.info(f"Negative samples: {len(negatives_df)}")

    # Prepare combined sample list
    samples = []

    # Add positives
    for _, row in events_df.iterrows():
        samples.append({
            'event_id': row['event_id'],
            'symbol': row['symbol'],
            'timestamp': row['entry_start'],
            'label': 1,
            'sample_type': 'spike'
        })

    # Add negatives
    for _, row in negatives_df.iterrows():
        samples.append({
            'event_id': row['event_id'],
            'symbol': row['symbol'],
            'timestamp': row['timestamp'],
            'label': 0,
            'sample_type': row['negative_type']
        })

    samples_df = pd.DataFrame(samples)
    logger.info(f"Total samples to featurize: {len(samples_df)}")

    # Connect to DB
    conn = duckdb.connect(DB_PATH, read_only=True)

    # Process each symbol
    symbols = samples_df['symbol'].unique()
    logger.info(f"Symbols to process: {len(symbols)}")

    all_features = []
    errors = 0

    for sym_idx, symbol in enumerate(symbols):
        try:
            # Load OHLCV for this symbol
            ohlcv_query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = '{symbol}' AND timeframe = '1m'
                ORDER BY timestamp
            """
            ohlcv_df = conn.execute(ohlcv_query).fetchdf()

            if len(ohlcv_df) < MIN_LOOKBACK_MINUTES:
                logger.warning(f"{symbol}: Not enough data ({len(ohlcv_df)} candles)")
                continue

            # Create timestamp index
            ts_to_idx = dict(zip(ohlcv_df['timestamp'], range(len(ohlcv_df))))

            # Get samples for this symbol
            sym_samples = samples_df[samples_df['symbol'] == symbol]

            for _, sample in sym_samples.iterrows():
                ts = sample['timestamp']
                idx = ts_to_idx.get(ts)

                if idx is None:
                    # Find closest timestamp
                    closest_idx = ohlcv_df['timestamp'].searchsorted(ts)
                    idx = min(closest_idx, len(ohlcv_df) - 1)

                if idx < MIN_LOOKBACK_MINUTES:
                    continue

                features = compute_all_features(ohlcv_df, idx)

                if features:
                    features['event_id'] = sample['event_id']
                    features['symbol'] = sample['symbol']
                    features['timestamp'] = sample['timestamp']
                    features['label'] = sample['label']
                    features['sample_type'] = sample['sample_type']
                    all_features.append(features)

            if (sym_idx + 1) % 25 == 0:
                logger.info(f"Processed {sym_idx + 1}/{len(symbols)} symbols, {len(all_features)} samples featurized")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            errors += 1

    conn.close()

    # Create DataFrame
    features_df = pd.DataFrame(all_features)

    # Reorder columns: metadata first, then features
    meta_cols = ['event_id', 'symbol', 'timestamp', 'label', 'sample_type']
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    features_df = features_df[meta_cols + feature_cols]

    # Sort by timestamp
    features_df = features_df.sort_values('timestamp').reset_index(drop=True)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples with features: {len(features_df)}")
    logger.info(f"  Positives (spikes): {(features_df['label'] == 1).sum()}")
    logger.info(f"  Negatives: {(features_df['label'] == 0).sum()}")
    logger.info(f"Features computed: {len(feature_cols)}")
    logger.info(f"Errors: {errors}")

    # Label distribution
    logger.info(f"\nSample type distribution:")
    print(features_df['sample_type'].value_counts())

    # Feature statistics
    logger.info(f"\nFeature statistics (mean):")
    for col in feature_cols[:10]:  # Show first 10
        logger.info(f"  {col}: {features_df[col].mean():.4f}")

    # Check for NaN/inf
    nan_counts = features_df[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"\nFeatures with NaN values:")
        print(nan_counts[nan_counts > 0])

    # Save
    features_df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"\nSaved to: {OUTPUT_PATH}")

    # Show sample
    logger.info(f"\nSample rows:")
    print(features_df[['event_id', 'symbol', 'label', 'natr_14', 'vol_acceleration', 'vol_price_divergence']].head(10).to_string())

    return features_df


if __name__ == "__main__":
    main()
