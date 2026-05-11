"""
Preprocess Data into Shards for Memory-Efficient Training

Creates parquet shards that can be streamed during training,
avoiding the need to load entire dataset into RAM.

Usage:
    python scripts/preprocess_shards.py

Output:
    data/shards/shard_XXX.parquet - Sequence data shards
    data/shards/metadata.json - Normalization stats and config
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2')

import gc
import json
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime
from loguru import logger

# Configuration
CONFIG = {
    'labels_path': '/Users/bz/Pythia2/data/big_mover_labels.parquet',
    'db_path': '/Users/bz/Pythia2/data/pythia.duckdb',
    'output_dir': '/Users/bz/Pythia2/data/shards',
    'seq_len': 720,  # 12 hours of 1-min candles
    'symbols_per_shard': 10,  # Process 10 symbols at a time
    'min_positive_samples': 1,  # Minimum positives to include symbol
    'feature_cols': [
        'returns', 'log_returns', 'volatility_20', 'volatility_60',
        'rsi_14', 'natr', 'atr',
        'momentum_5', 'momentum_20', 'momentum_60',
        'bb_width', 'bb_position',
        'volume_ratio', 'volume_zscore', 'obv_slope', 'vroc',
        'vwap_distance',
        'hl_range', 'body_ratio',
    ],
}


def compute_features_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators from OHLCV data."""
    df = df.copy()

    # Price returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_60'] = df['returns'].rolling(60).std()

    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # NATR (Normalized ATR)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    df['natr'] = atr_14 / df['close'] * 100
    df['atr'] = atr_14

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # Volume features
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
    df['volume_zscore'] = (df['volume'] - df['volume_ma_20']) / (df['volume'].rolling(20).std() + 1e-10)

    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv_slope'] = df['obv'].diff(5)

    # VROC (Volume Rate of Change)
    df['vroc'] = df['volume'].pct_change(12)

    # VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-10)
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap'] * 100

    # Price momentum
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_20'] = df['close'].pct_change(20)
    df['momentum_60'] = df['close'].pct_change(60)

    # High-low range
    df['hl_range'] = (df['high'] - df['low']) / df['close']

    # Candle body ratio
    df['body_ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-10)

    return df


def create_sequences(features: np.ndarray, labels: np.ndarray, seq_len: int) -> tuple:
    """Create sliding window sequences."""
    n_samples = len(features) - seq_len
    if n_samples <= 0:
        return np.array([]), np.array([])

    X = np.zeros((n_samples, seq_len, features.shape[1]), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        X[i] = features[i:i+seq_len]
        y[i] = labels[i+seq_len]

    return X, y


def process_symbol_batch(symbols: list, all_ohlcv: pd.DataFrame, labels_df: pd.DataFrame,
                         feature_cols: list, seq_len: int) -> tuple:
    """Process a batch of symbols and return sequences."""
    all_X = []
    all_y = []

    for symbol in symbols:
        try:
            # Get OHLCV for this symbol
            sym_ohlcv = all_ohlcv[all_ohlcv['symbol'] == symbol].copy()
            if len(sym_ohlcv) < seq_len + 100:
                continue

            # Compute features
            sym_ohlcv = compute_features_from_ohlcv(sym_ohlcv)

            # Get labels for this symbol
            sym_labels = labels_df[labels_df['symbol'] == symbol].copy()
            sym_labels = sym_labels.set_index('timestamp')

            # Merge features with labels
            sym_ohlcv = sym_ohlcv.set_index('timestamp')
            merged = sym_ohlcv.join(sym_labels[['label']], how='inner')

            if len(merged) < seq_len + 10:
                continue

            # Fill NaN with 0
            merged = merged.fillna(0)

            # Check all feature columns exist
            missing = [c for c in feature_cols if c not in merged.columns]
            if missing:
                logger.warning(f"{symbol}: Missing columns {missing}")
                continue

            # Extract features and labels
            feature_values = merged[feature_cols].values.astype(np.float32)
            label_values = merged['label'].values.astype(np.float32)

            # Create sequences
            X, y = create_sequences(feature_values, label_values, seq_len)

            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    if not all_X:
        return None, None

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def main():
    logger.info("=" * 60)
    logger.info("PREPROCESSING DATA INTO SHARDS")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing shards
    for f in output_dir.glob('shard_*.parquet'):
        f.unlink()
    logger.info(f"Output directory: {output_dir}")

    # Load labels
    logger.info("Loading labels...")
    labels_df = pd.read_parquet(CONFIG['labels_path'])
    logger.info(f"Total label samples: {len(labels_df):,}")

    # Find symbols with positive samples
    pos_counts = labels_df.groupby('symbol')['label'].sum()
    useful_symbols = pos_counts[pos_counts >= CONFIG['min_positive_samples']].index.tolist()
    logger.info(f"Symbols with >= {CONFIG['min_positive_samples']} positive samples: {len(useful_symbols)}")

    # Filter labels to useful symbols only
    labels_df = labels_df[labels_df['symbol'].isin(useful_symbols)]
    logger.info(f"Filtered labels: {len(labels_df):,} samples")

    # Load OHLCV data for useful symbols
    logger.info("Loading OHLCV data...")
    conn = duckdb.connect(CONFIG['db_path'], read_only=True)
    symbols_str = "', '".join(useful_symbols)
    all_ohlcv = conn.execute(f"""
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol IN ('{symbols_str}') AND timeframe = '1m'
        ORDER BY symbol, timestamp
    """).fetchdf()
    conn.close()
    logger.info(f"Loaded {len(all_ohlcv):,} OHLCV rows")

    # Process symbols in batches
    n_symbols = len(useful_symbols)
    batch_size = CONFIG['symbols_per_shard']
    n_batches = (n_symbols + batch_size - 1) // batch_size

    logger.info(f"Processing {n_symbols} symbols in {n_batches} batches of {batch_size}")

    # Track statistics for normalization
    all_means = []
    all_stds = []
    total_sequences = 0
    total_positives = 0
    shard_info = []

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_symbols)
        batch_symbols = useful_symbols[start:end]

        logger.info(f"Processing batch {batch_idx + 1}/{n_batches}: {len(batch_symbols)} symbols")

        # Process this batch
        X, y = process_symbol_batch(
            batch_symbols, all_ohlcv, labels_df,
            CONFIG['feature_cols'], CONFIG['seq_len']
        )

        if X is None or len(X) == 0:
            logger.warning(f"Batch {batch_idx + 1} produced no sequences")
            continue

        # Track stats for normalization (before normalizing)
        batch_mean = X.mean(axis=(0, 1))
        batch_std = X.std(axis=(0, 1))
        all_means.append((batch_mean, len(X)))
        all_stds.append((batch_std, len(X)))

        n_pos = int(y.sum())
        total_sequences += len(X)
        total_positives += n_pos

        # Save shard (store raw data, normalize during training)
        shard_path = output_dir / f'shard_{batch_idx:03d}.parquet'

        # Flatten X for parquet storage
        # Shape: (n_samples, seq_len * n_features)
        X_flat = X.reshape(len(X), -1)

        shard_df = pd.DataFrame({
            'features': list(X_flat),  # Store as list of arrays
            'label': y
        })
        shard_df.to_parquet(shard_path)

        shard_info.append({
            'path': str(shard_path),
            'n_samples': len(X),
            'n_positives': n_pos,
            'symbols': batch_symbols
        })

        logger.info(f"  Saved {shard_path.name}: {len(X):,} sequences, {n_pos:,} positives ({100*n_pos/len(X):.1f}%)")

        # Free memory
        del X, y, shard_df
        gc.collect()

    # Compute global normalization stats (weighted average)
    logger.info("Computing global normalization stats...")
    total_weight = sum(w for _, w in all_means)
    global_mean = sum(m * w for m, w in all_means) / total_weight

    # For std, we need to combine properly
    # Var(combined) ≈ weighted average of variances (approximation)
    global_var = sum((s ** 2) * w for s, w in all_stds) / total_weight
    global_std = np.sqrt(global_var) + 1e-8

    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'config': CONFIG,
        'n_shards': len(shard_info),
        'total_sequences': total_sequences,
        'total_positives': total_positives,
        'positive_rate': total_positives / total_sequences if total_sequences > 0 else 0,
        'n_symbols': len(useful_symbols),
        'symbols': useful_symbols,
        'feature_cols': CONFIG['feature_cols'],
        'seq_len': CONFIG['seq_len'],
        'norm_mean': global_mean.tolist(),
        'norm_std': global_std.tolist(),
        'shards': shard_info
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Shards created: {len(shard_info)}")
    logger.info(f"Total sequences: {total_sequences:,}")
    logger.info(f"Total positives: {total_positives:,} ({100*total_positives/total_sequences:.2f}%)")
    logger.info(f"Symbols included: {len(useful_symbols)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Metadata saved: {metadata_path}")

    return metadata


if __name__ == "__main__":
    main()
