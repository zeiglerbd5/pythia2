"""
Dataset Preparation and Data Loaders for Spike Detection

Implements data pipeline per implementation guide:
- Time-series sequences: 30-60 day lookback (60 per guide)
- Features: 30-40 after Boruta selection
- Targets: 7-14 day local extrema (not daily movements)
- Class imbalance: 1-5% positive class typical
- Walk-forward validation: Temporal train/val/test splits

Handles loading from DuckDB, normalization, sequence generation,
and PyTorch DataLoader creation with batch balancing.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Optional, List, Dict, TYPE_CHECKING
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
from loguru import logger
import duckdb

if TYPE_CHECKING:
    from .spike_filter import SpikeCategoryFilter


class SpikeDataset(Dataset):
    """
    PyTorch Dataset for cryptocurrency spike detection.

    Per implementation guide:
    - Input: (sequence_length, n_features) time-series sequences
    - Output: Binary spike label (0/1) for 7-14 day forward window
    - Handles extreme class imbalance (typically 1-5% positive)
    - Preserves temporal ordering for walk-forward validation

    Each sample is a 60-day sequence predicting if a spike occurs
    in the next 7-14 days.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
        symbols: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize spike detection dataset.

        Args:
            sequences: Input sequences of shape (n_samples, sequence_length, n_features)
            labels: Binary labels of shape (n_samples,)
            timestamps: Timestamp for each sequence of shape (n_samples,)
            symbols: Symbol for each sequence of shape (n_samples,)
            feature_names: List of feature names
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.timestamps = timestamps
        self.symbols = symbols
        self.feature_names = feature_names or []

        # Statistics
        self.n_samples = len(self.sequences)
        self.sequence_length = self.sequences.shape[1]
        self.n_features = self.sequences.shape[2]
        self.pos_ratio = self.labels.mean().item()

        logger.info(
            f"SpikeDataset initialized",
            extra={
                "n_samples": self.n_samples,
                "sequence_length": self.sequence_length,
                "n_features": self.n_features,
                "positive_ratio": f"{self.pos_ratio*100:.2f}%",
                "date_range": f"{timestamps.min()} to {timestamps.max()}"
            }
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (sequence, label)
            - sequence: (sequence_length, n_features)
            - label: scalar (0 or 1)
        """
        return self.sequences[idx], self.labels[idx]

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training.

        Per guide: Use inverse frequency weighting for imbalanced data.

        Returns:
            Tensor of shape (2,) with weights for [negative, positive] classes
        """
        n_neg = (self.labels == 0).sum().item()
        n_pos = (self.labels == 1).sum().item()

        total = n_neg + n_pos

        # Inverse frequency
        weight_neg = total / (2.0 * n_neg)
        weight_pos = total / (2.0 * n_pos)

        return torch.tensor([weight_neg, weight_pos])

    def get_sample_weights(self) -> torch.Tensor:
        """
        Get per-sample weights for WeightedRandomSampler.

        Returns:
            Tensor of shape (n_samples,) with weight for each sample
        """
        class_weights = self.get_class_weights()
        return class_weights[self.labels.long()]

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        return {
            "n_samples": self.n_samples,
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "positive_samples": (self.labels == 1).sum().item(),
            "negative_samples": (self.labels == 0).sum().item(),
            "positive_ratio": self.pos_ratio,
            "date_range": (str(self.timestamps.min()), str(self.timestamps.max())),
            "symbols": list(np.unique(self.symbols))
        }


class SpikeDatasetWithSoftLabels(SpikeDataset):
    """
    Extended SpikeDataset that includes XGBoost soft labels for knowledge distillation.

    For knowledge transfer from XGBoost to TCN:
    - hard_labels: True spike labels (0/1)
    - soft_labels: XGBoost probability predictions (0-1)

    The soft labels allow the TCN to learn XGBoost's "intuition" about
    which samples are confidently spikes vs edge cases.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        hard_labels: np.ndarray,
        soft_labels: np.ndarray,
        timestamps: np.ndarray,
        symbols: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize dataset with soft labels.

        Args:
            sequences: Input sequences (n_samples, sequence_length, n_features)
            hard_labels: True spike labels (n_samples,) - 0 or 1
            soft_labels: XGBoost probabilities (n_samples,) - 0.0 to 1.0
            timestamps: Timestamps (n_samples,)
            symbols: Symbol names (n_samples,)
            feature_names: Feature column names
        """
        # Initialize parent class
        super().__init__(
            sequences=sequences,
            labels=hard_labels,
            timestamps=timestamps,
            symbols=symbols,
            feature_names=feature_names
        )

        # Add soft labels
        self.soft_labels = torch.FloatTensor(soft_labels)

        # Validate soft labels
        assert len(self.soft_labels) == self.n_samples, \
            f"Soft labels length {len(self.soft_labels)} != samples {self.n_samples}"

        # Log soft label statistics
        mean_soft = self.soft_labels.mean().item()
        max_soft = self.soft_labels.max().item()

        logger.info(
            f"SpikeDatasetWithSoftLabels: {self.n_samples} samples, "
            f"soft_label mean={mean_soft:.4f}, max={max_soft:.4f}"
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample with soft label.

        Returns:
            Tuple of (sequence, hard_label, soft_label)
        """
        return self.sequences[idx], self.labels[idx], self.soft_labels[idx]

    def get_statistics(self) -> dict:
        """Get dataset statistics including soft labels."""
        stats = super().get_statistics()
        stats.update({
            "soft_label_mean": self.soft_labels.mean().item(),
            "soft_label_std": self.soft_labels.std().item(),
            "soft_label_min": self.soft_labels.min().item(),
            "soft_label_max": self.soft_labels.max().item(),
        })
        return stats


class SpikeTargetGenerator:
    """
    Generate pre-spike targets for early detection.

    **Updated Strategy (October 2025):**
    - Target: Detect accumulation phase 1-3 minutes BEFORE spike
    - Price spike: 6%+ move in next 3 minutes
    - Volume explosion: 5x increase in next 1-2 minutes
    - Combines volume + price signals for high-precision entries
    - Optimized for 0.75% stop loss (8:1 reward-risk ratio)
    """

    def __init__(
        self,
        price_window: int = 3,  # Minutes to look ahead for price spike
        volume_window: int = 2,  # Minutes to look ahead for volume explosion
        min_price_spike: float = 0.06,  # 6% minimum price move
        min_volume_spike: float = 5.0,   # 5x volume increase
        prediction_offset: int = 2  # NEW: Minutes to offset prediction (for true forecasting)
    ):
        """
        Initialize pre-spike target generator.

        Args:
            price_window: Minutes to look forward for price spike (default: 3)
            min_price_spike: Minimum price increase to qualify (default: 6%)
            volume_window: Minutes to look forward for volume explosion (default: 2)
            min_volume_spike: Minimum volume multiplier (default: 5x)
            prediction_offset: Minutes to offset prediction window (default: 2)
                             - offset=0: Label candles during spike (DESCRIPTIVE - current v1)
                             - offset=2: Label candles 2min before spike (PREDICTIVE - new v2)
        """
        self.price_window = int(price_window)
        self.volume_window = int(volume_window)
        self.min_price_spike = min_price_spike
        self.min_volume_spike = min_volume_spike
        self.prediction_offset = int(prediction_offset)

        logger.info(
            f"SpikeTargetGenerator initialized (pre-spike detection)",
            extra={
                "price_window": f"{self.price_window} minutes",
                "volume_window": f"{self.volume_window} minutes",
                "min_price_spike": f"{min_price_spike*100:.1f}%",
                "min_volume_spike": f"{min_volume_spike:.1f}x",
                "prediction_offset": f"{prediction_offset} minutes ({'PREDICTIVE' if prediction_offset > 0 else 'DESCRIPTIVE'})"
            }
        )

    def generate_targets(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        timestamps: pd.Series = None
    ) -> pd.Series:
        """
        Generate binary pre-spike targets (VECTORIZED - 10-100x faster).

        Detects accumulation patterns 1-3 minutes before spike:
        - Volume explosion: 5x increase in next 1-2 minutes
        - Price spike: 6%+ move in next 3 minutes

        Args:
            prices: Close prices (1-minute candles)
            volumes: Trade volumes (1-minute candles)
            timestamps: Corresponding timestamps (optional, for logging)

        Returns:
            Binary series (0/1) indicating if pre-spike pattern detected
        """
        n = len(prices)
        max_window = max(self.price_window, self.volume_window)

        # Convert to numpy for faster operations
        price_arr = prices.values
        volume_arr = volumes.values

        # Initialize targets
        targets = np.zeros(n, dtype=np.float32)

        # Vectorized calculation of max future volumes (rolling max)
        # Apply prediction_offset to shift window forward for true forecasting
        max_future_volumes = np.zeros(n, dtype=np.float32)
        for i in range(n - max_window - self.prediction_offset):
            # OLD (DESCRIPTIVE): volume_arr[i+1:i+1+self.volume_window]
            # NEW (PREDICTIVE): volume_arr[i+1+offset:i+1+offset+window]
            start_idx = i + 1 + self.prediction_offset
            end_idx = start_idx + self.volume_window
            max_future_volumes[i] = volume_arr[start_idx:end_idx].max()

        # Vectorized calculation of max future prices (rolling max)
        max_future_prices = np.zeros(n, dtype=np.float32)
        for i in range(n - max_window - self.prediction_offset):
            start_idx = i + 1 + self.prediction_offset
            end_idx = start_idx + self.price_window
            max_future_prices[i] = price_arr[start_idx:end_idx].max()

        # Vectorized ratio calculations (avoid division by zero)
        volume_ratios = max_future_volumes / (volume_arr + 1e-10)
        price_returns = (max_future_prices - price_arr) / (price_arr + 1e-10)

        # Vectorized condition checks
        has_volume_explosion = volume_ratios >= self.min_volume_spike
        has_price_spike = price_returns >= self.min_price_spike
        volume_not_zero = volume_arr > 0  # Skip gap-filled candles

        # Combine conditions
        targets = (has_volume_explosion & has_price_spike & volume_not_zero).astype(np.float32)

        return pd.Series(targets, index=prices.index)

    def analyze_targets(
        self,
        targets: pd.Series,
        prices: pd.Series
    ) -> dict:
        """
        Analyze generated pre-spike targets for validation.

        For pre-spike detection, targets are binary (0/1) indicating whether
        the accumulation pattern (volume explosion + price spike) was detected
        in the next 1-3 minutes. We analyze pattern statistics, not returns.

        Returns:
            Dictionary with target statistics
        """
        pos_indices = targets[targets == 1].index
        n_total = len(targets)
        n_spikes = len(pos_indices)

        if n_spikes == 0:
            return {
                "n_spikes": 0,
                "spike_ratio": 0.0,
                "n_total_candles": n_total,
                "expected_signals_per_day": 0.0
            }

        # Calculate metrics for pre-spike detection
        spike_ratio = n_spikes / n_total

        # Estimate signals per day (assuming 1-minute candles)
        # 1,440 candles per day = 24 hours * 60 minutes
        candles_per_day = 1440
        expected_signals_per_day = spike_ratio * candles_per_day

        return {
            "n_spikes": n_spikes,
            "spike_ratio": spike_ratio,
            "n_total_candles": n_total,
            "expected_signals_per_day": expected_signals_per_day
        }


class DatasetBuilder:
    """
    Build datasets from DuckDB feature storage.

    Handles:
    - Loading features and prices from database
    - Target generation (7-14 day spikes)
    - Sequence generation (60-day lookback)
    - Normalization and preprocessing
    - Train/val/test splitting (temporal)
    - Feature selection integration
    """

    def __init__(
        self,
        db_path: str,
        sequence_length: int = 60,
        forward_window: int = 14,
        min_spike_threshold: float = 0.15,
        scaler_type: str = 'robust'  # 'standard' or 'robust'
    ):
        """
        Initialize dataset builder.

        Args:
            db_path: Path to DuckDB database
            sequence_length: Sequence length for LSTM (30-60 per guide)
            forward_window: Forward window for target generation (7-14)
            min_spike_threshold: Minimum return for spike (15%)
            scaler_type: Type of scaler ('standard' or 'robust')
        """
        self.db_path = db_path
        self.sequence_length = sequence_length
        self.forward_window = forward_window
        self.min_spike_threshold = min_spike_threshold

        # Target generator (uses pre-spike detection parameters)
        # forward_window and min_spike_threshold are legacy params, not used
        self.target_generator = SpikeTargetGenerator(
            price_window=3,  # 3 minutes ahead for price spike
            volume_window=2,  # 2 minutes ahead for volume explosion
            min_price_spike=0.06,  # 6% price move
            min_volume_spike=5.0  # 5x volume increase
        )

        # Scaler (fit on training data only)
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()  # Better for outliers per guide

        self.scaler_fitted = False

        logger.info(
            f"DatasetBuilder initialized",
            extra={
                "db_path": db_path,
                "sequence_length": sequence_length,
                "forward_window": forward_window,
                "min_spike_threshold": f"{min_spike_threshold*100:.1f}%",
                "scaler_type": scaler_type
            }
        )

    def load_features(
        self,
        symbol: str,
        timeframe: str = '5m',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load features from DuckDB.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe ('1m', '5m', '15m')
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            feature_columns: Specific features to load (None = all)

        Returns:
            DataFrame with features indexed by timestamp
        """
        conn = duckdb.connect(self.db_path, read_only=True)

        try:
            # Build query
            query = f"""
                SELECT *
                FROM features
                WHERE symbol = '{symbol}'
                AND timeframe = '{timeframe}'
            """

            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"

            query += " ORDER BY timestamp ASC"

            # Load data
            df = conn.execute(query).df()

            if df.empty:
                logger.warning(f"No features found for {symbol} {timeframe}")
                return pd.DataFrame()

            # Set index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Select feature columns if specified
            if feature_columns:
                # Keep metadata columns
                meta_cols = ['symbol', 'timeframe']
                cols_to_keep = meta_cols + [c for c in feature_columns if c in df.columns]
                df = df[cols_to_keep]

            logger.info(f"Loaded {len(df)} feature rows for {symbol} {timeframe}")

            return df

        finally:
            conn.close()

    def load_prices_and_volumes(
        self,
        symbol: str,
        timeframe: str = '1m',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Load close prices AND volumes from database for pre-spike target generation.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (default: 1m for minute-by-minute data)
            start_date: Start date
            end_date: End date

        Returns:
            Tuple of (prices, volumes) - both Series indexed by timestamp
        """
        conn = duckdb.connect(self.db_path, read_only=True)

        try:
            # Prioritize forward-filled data: candles_filled > candles > ohlcv
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [t[0] for t in tables]

            if 'candles_filled' in table_names:
                table_name = 'candles_filled'
                logger.debug(f"Using 'candles_filled' table (forward-filled data)")
            elif 'candles' in table_names:
                table_name = 'candles'
                logger.debug(f"Using 'candles' table")
            else:
                table_name = 'ohlcv'
                logger.debug(f"Using 'ohlcv' table")

            query = f"""
                SELECT timestamp, close, volume
                FROM {table_name}
                WHERE symbol = '{symbol}'
            """

            # For candles table, no timeframe column
            if table_name == 'ohlcv':
                query += f" AND timeframe = '{timeframe}'"

            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"

            query += " ORDER BY timestamp ASC"

            df = conn.execute(query).df()

            if df.empty:
                logger.warning(f"No data found for {symbol} in {table_name}")
                return pd.Series(dtype=float), pd.Series(dtype=float)

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            return df['close'], df['volume']

        finally:
            conn.close()

    # Keep old method for backward compatibility
    def load_prices(self, symbol: str, timeframe: str = '1m', start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.Series:
        """Legacy method - loads only prices. Use load_prices_and_volumes for pre-spike detection."""
        prices, _ = self.load_prices_and_volumes(symbol, timeframe, start_date, end_date)
        return prices

    def load_targets(
        self,
        symbol: str,
        timeframe: str = '1m',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Load pre-computed targets from database.

        This loads targets that were pre-computed by scripts/precompute_targets.py.
        Much faster than computing targets on-the-fly during training.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (default: 1m)
            start_date: Start date
            end_date: End date

        Returns:
            Series of binary targets (0/1) indexed by timestamp
        """
        conn = duckdb.connect(self.db_path, read_only=True)

        try:
            query = f"""
                SELECT timestamp, target
                FROM targets
                WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
            """

            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"

            query += " ORDER BY timestamp ASC"

            df = conn.execute(query).df()

            if df.empty:
                logger.warning(f"No pre-computed targets found for {symbol} {timeframe}")
                return pd.Series(dtype=float)

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            logger.debug(f"Loaded {len(df)} pre-computed targets for {symbol} {timeframe}")

            return df['target']

        finally:
            conn.close()

    def create_sequences(
        self,
        features: pd.DataFrame,
        targets: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create time-series sequences from features and targets.

        Args:
            features: Feature DataFrame
            targets: Target series

        Returns:
            Tuple of (sequences, labels, timestamps, symbols)
            - sequences: (n_samples, sequence_length, n_features)
            - labels: (n_samples,)
            - timestamps: (n_samples,) - timestamp of prediction point
            - symbols: (n_samples,) - symbol for each sequence
        """
        # Align features and targets
        common_index = features.index.intersection(targets.index)
        features = features.loc[common_index]
        targets = targets.loc[common_index]

        # Get feature columns (exclude metadata and non-numeric)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in ['symbol', 'timeframe']]
        feature_matrix = features[feature_cols].values

        # Clean NaN/inf values from forward-filled data (volume=0 causes division issues)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Generate sequences using vectorized sliding window (100x faster than loop)
        n = len(feature_matrix)

        # Check if we have enough data
        if n < self.sequence_length + self.forward_window:
            logger.warning(f"Not enough data: {n} < {self.sequence_length + self.forward_window}")
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([]), np.array([])

        # Use NumPy's sliding_window_view for vectorized sequence creation
        from numpy.lib.stride_tricks import sliding_window_view

        # Create sliding windows: shape (n_windows, sequence_length, n_features)
        # sliding_window_view creates (n - seq_len + 1) windows
        sequences = sliding_window_view(
            feature_matrix,
            window_shape=(self.sequence_length, feature_matrix.shape[1])
        ).squeeze(axis=1)

        # Calculate valid range considering forward_window
        # We need to exclude forward_window samples at the end
        max_sequences = n - self.sequence_length - self.forward_window
        sequences = sequences[:max_sequences]

        # Vectorized label/timestamp/symbol extraction
        # Labels correspond to the timestep AFTER each sequence
        label_start_idx = self.sequence_length
        label_end_idx = self.sequence_length + max_sequences

        labels = targets.iloc[label_start_idx:label_end_idx].values.astype(np.float32)
        timestamps = features.index[label_start_idx:label_end_idx].values

        # Handle symbols - vectorized
        if 'symbol' in features.columns:
            symbols = features['symbol'].iloc[label_start_idx:label_end_idx].values
        else:
            symbols = np.array(['UNKNOWN'] * len(labels))

        # Convert to proper dtype
        sequences = sequences.astype(np.float32)

        logger.info(
            f"Created {len(sequences)} sequences",
            extra={
                "sequence_shape": sequences.shape,
                "positive_ratio": f"{labels.mean()*100:.2f}%"
            }
        )

        return sequences, labels, timestamps, symbols

    def normalize_sequences(
        self,
        sequences: np.ndarray,
        fit: bool = False
    ) -> np.ndarray:
        """
        Normalize sequences using fitted scaler.

        Args:
            sequences: Input sequences (n_samples, sequence_length, n_features)
            fit: If True, fit scaler on this data (training set only)

        Returns:
            Normalized sequences
        """
        original_shape = sequences.shape

        # Reshape to (n_samples * sequence_length, n_features)
        sequences_2d = sequences.reshape(-1, sequences.shape[-1])

        # Handle inf/NaN values from forward-filled data (volume=0 causes division issues)
        # Replace inf with NaN, then fill NaN with 0
        sequences_2d = np.nan_to_num(sequences_2d, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            sequences_2d = self.scaler.fit_transform(sequences_2d)
            self.scaler_fitted = True
            logger.info("Scaler fitted on training data")
        else:
            if not self.scaler_fitted:
                raise ValueError("Scaler not fitted. Call with fit=True on training data first.")
            sequences_2d = self.scaler.transform(sequences_2d)

        # Reshape back
        sequences = sequences_2d.reshape(original_shape)

        return sequences

    def build_dataset(
        self,
        symbol: str,
        timeframe: str = '5m',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        normalize: bool = True,
        fit_scaler: bool = False
    ) -> Optional[SpikeDataset]:
        """
        Build complete dataset for a symbol.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            feature_columns: Features to use
            normalize: Whether to normalize
            fit_scaler: Whether to fit scaler (training set only)

        Returns:
            SpikeDataset instance or None if insufficient data
        """
        # Load features
        features = self.load_features(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            feature_columns=feature_columns
        )

        if features.empty:
            return None

        # Load pre-computed targets (INSTANT - computed once by precompute_targets.py)
        targets = self.load_targets(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if targets.empty:
            logger.warning(f"No pre-computed targets for {symbol} - run precompute_targets.py first")
            return None

        # Quick stats logging
        n_positives = int(targets.sum())
        positive_pct = (n_positives / len(targets)) * 100 if len(targets) > 0 else 0
        logger.info(f"{symbol}: {len(targets):,} targets, {n_positives} pre-spike patterns ({positive_pct:.2f}%)")

        # Create sequences
        sequences, labels, timestamps, symbols = self.create_sequences(features, targets)

        if len(sequences) == 0:
            logger.warning(f"No valid sequences created for {symbol}")
            return None

        # Normalize
        if normalize:
            sequences = self.normalize_sequences(sequences, fit=fit_scaler)

        # Create dataset
        dataset = SpikeDataset(
            sequences=sequences,
            labels=labels,
            timestamps=timestamps,
            symbols=symbols,
            feature_names=feature_columns
        )

        return dataset

    def build_dataset_multi_symbol(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = '1m',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        min_candles: int = 100,
        max_symbols: Optional[int] = None,
        skip_symbols: int = 0,
        normalize: bool = True,
        fit_scaler: bool = False,
        spike_filter: Optional['SpikeCategoryFilter'] = None
    ) -> Optional[SpikeDataset]:
        """
        Build dataset from multiple symbols simultaneously.

        Loads all symbols, creates sequences for each, and concatenates
        into one unified dataset. Enables training on entire market.

        Args:
            symbols: List of symbols (None = all symbols in database)
            timeframe: Timeframe ('1m', '5m', '15m')
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            feature_columns: Features to use (None = all)
            min_candles: Minimum candles required per symbol
            max_symbols: Maximum number of symbols to process (None = all)
            skip_symbols: Number of symbols to skip (for batch processing)
            normalize: Whether to normalize features
            fit_scaler: Whether to fit scaler (training only)
            spike_filter: Optional SpikeCategoryFilter to filter targets to specific spike types

        Returns:
            SpikeDataset with all symbols combined, or None if no data
        """
        logger.info("Building multi-symbol dataset...")

        # Get list of symbols if not provided
        if symbols is None:
            symbols = self._get_all_symbols(timeframe)
            logger.info(f"Discovered {len(symbols)} symbols in database")

        # Filter by minimum candles
        if min_candles > 0:
            symbol_counts = self._count_candles_per_symbol(symbols, timeframe)
            symbols = [s for s in symbols if symbol_counts.get(s, 0) >= min_candles]
            logger.info(f"Filtered to {len(symbols)} symbols with >= {min_candles} candles")

        # Limit to max_symbols (take top N by candle count for consistency)
        if max_symbols or skip_symbols:
            symbol_counts = self._count_candles_per_symbol(symbols, timeframe)
            symbols = sorted(symbols, key=lambda s: symbol_counts.get(s, 0), reverse=True)

            if skip_symbols:
                symbols = symbols[skip_symbols:]
                logger.info(f"Skipped first {skip_symbols} symbols by data volume")

            if max_symbols and len(symbols) > max_symbols:
                symbols = symbols[:max_symbols]
                logger.info(f"Limited to {max_symbols} symbols (ranks {skip_symbols+1}-{skip_symbols+max_symbols})")

        if not symbols:
            logger.error("No symbols found matching criteria")
            return None

        # Build dataset for each symbol
        all_sequences = []
        all_labels = []
        all_timestamps = []
        all_symbols = []

        for i, symbol in enumerate(symbols):
            logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})...")

            # Load features (try from features table first, fallback to candles)
            features = self.load_features(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                feature_columns=feature_columns
            )

            # If no features table, we need prices at minimum
            if features.empty:
                logger.warning(f"No features for {symbol}, skipping")
                continue

            # Load pre-computed targets (INSTANT - computed once by precompute_targets.py)
            targets = self.load_targets(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if targets.empty:
                logger.warning(f"No pre-computed targets for {symbol} - run precompute_targets.py first, skipping")
                continue

            # Quick stats logging
            n_positives = int(targets.sum())
            positive_pct = (n_positives / len(targets)) * 100 if len(targets) > 0 else 0
            if n_positives > 0:
                logger.info(f"{symbol}: {len(targets):,} targets, {n_positives} pre-spike patterns ({positive_pct:.2f}%)")

            # Create sequences for this symbol
            sequences, labels, timestamps, symbol_array = self.create_sequences(features, targets)

            if len(sequences) == 0:
                logger.warning(f"No valid sequences for {symbol}")
                continue

            # Apply spike category filter if provided
            if spike_filter is not None:
                original_positives = int(labels.sum())
                labels = spike_filter.filter_targets(
                    targets=labels,
                    timestamps=timestamps,
                    symbols=symbol_array,
                    categories=['Slow & Large'],
                    exclude_unlabeled=True
                )
                filtered_positives = int(labels.sum())
                if original_positives > 0:
                    logger.info(
                        f"{symbol}: Filtered {original_positives} → {filtered_positives} spikes "
                        f"({filtered_positives/original_positives*100:.1f}% kept)"
                    )

            # Accumulate
            all_sequences.append(sequences)
            all_labels.append(labels)
            all_timestamps.append(timestamps)
            all_symbols.append(symbol_array)

            logger.info(f"{symbol}: Created {len(sequences)} sequences")

        if not all_sequences:
            logger.error("No sequences created for any symbol")
            return None

        # Concatenate all symbols
        logger.info("Concatenating all sequences...")

        combined_sequences = np.concatenate(all_sequences, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        combined_timestamps = np.concatenate(all_timestamps, axis=0)
        combined_symbols = np.concatenate(all_symbols, axis=0)

        logger.info(
            f"Combined dataset: {len(combined_sequences)} sequences "
            f"from {len(symbols)} symbols"
        )

        # Normalize (fit on all training data)
        if normalize:
            combined_sequences = self.normalize_sequences(
                combined_sequences,
                fit=fit_scaler
            )

        # Create unified dataset
        dataset = SpikeDataset(
            sequences=combined_sequences,
            labels=combined_labels,
            timestamps=combined_timestamps,
            symbols=combined_symbols,
            feature_names=feature_columns
        )

        # Log per-symbol statistics
        unique_symbols = np.unique(combined_symbols)
        logger.info(f"\nPer-symbol distribution:")
        for sym in unique_symbols[:10]:  # Show first 10
            count = (combined_symbols == sym).sum()
            logger.info(f"  {sym}: {count} sequences")

        if len(unique_symbols) > 10:
            logger.info(f"  ... and {len(unique_symbols) - 10} more symbols")

        return dataset

    def _get_all_symbols(self, timeframe: str = '1m') -> List[str]:
        """
        Get all unique symbols from database.

        Args:
            timeframe: Timeframe to check

        Returns:
            List of symbol strings
        """
        conn = duckdb.connect(self.db_path, read_only=True)

        try:
            # Prioritize forward-filled data: candles_filled > candles > ohlcv
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [t[0] for t in tables]

            if 'candles_filled' in table_names:
                table_name = 'candles_filled'
            elif 'candles' in table_names:
                table_name = 'candles'
            else:
                table_name = 'ohlcv'

            query = f"SELECT DISTINCT symbol FROM {table_name}"

            # OHLCV table has timeframe column
            if table_name == 'ohlcv':
                query += f" WHERE timeframe = '{timeframe}'"

            query += " ORDER BY symbol"

            result = conn.execute(query).fetchall()
            symbols = [r[0] for r in result]

            return symbols

        finally:
            conn.close()

    def _count_candles_per_symbol(
        self,
        symbols: List[str],
        timeframe: str = '1m'
    ) -> Dict[str, int]:
        """
        Count candles per symbol.

        Args:
            symbols: List of symbols to check
            timeframe: Timeframe

        Returns:
            Dictionary mapping symbol to candle count
        """
        conn = duckdb.connect(self.db_path, read_only=True)

        try:
            # Prioritize forward-filled data: candles_filled > candles > ohlcv
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [t[0] for t in tables]

            if 'candles_filled' in table_names:
                table_name = 'candles_filled'
            elif 'candles' in table_names:
                table_name = 'candles'
            else:
                table_name = 'ohlcv'

            counts = {}

            for symbol in symbols:
                query = f"SELECT COUNT(*) FROM {table_name} WHERE symbol = '{symbol}'"

                if table_name == 'ohlcv':
                    query += f" AND timeframe = '{timeframe}'"

                result = conn.execute(query).fetchone()
                counts[symbol] = result[0] if result else 0

            return counts

        finally:
            conn.close()


def create_dataloaders(
    train_dataset: SpikeDataset,
    val_dataset: Optional[SpikeDataset] = None,
    test_dataset: Optional[SpikeDataset] = None,
    batch_size: int = 32,
    use_weighted_sampler: bool = True,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train/val/test sets.

    Per guide: Use weighted sampling for class imbalance during training.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        use_weighted_sampler: Use WeightedRandomSampler for training
        num_workers: Number of worker processes

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    loaders = {}

    # Training loader (with weighted sampling for imbalanced data)
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        logger.info(f"Training DataLoader created with WeightedRandomSampler (batch_size={batch_size})")
    else:
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        logger.info(f"Training DataLoader created with shuffle (batch_size={batch_size})")

    # Validation loader (no sampling)
    if val_dataset:
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"Validation DataLoader created (batch_size={batch_size})")

    # Test loader (no sampling)
    if test_dataset:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"Test DataLoader created (batch_size={batch_size})")

    return loaders


if __name__ == "__main__":
    # Test dataset building
    import sys
    from pathlib import Path

    # Add parent directory
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("=== Spike Dataset Test ===\n")

    # Simulate feature data
    n_samples = 1000
    sequence_length = 60
    n_features = 35

    print(f"Generating synthetic data:")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence Length: {sequence_length}")
    print(f"  Features: {n_features}\n")

    # Create synthetic sequences
    np.random.seed(42)
    sequences = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)

    # Create synthetic labels (5% positive per guide)
    labels = np.zeros(n_samples, dtype=np.float32)
    n_positive = int(n_samples * 0.05)
    positive_indices = np.random.choice(n_samples, n_positive, replace=False)
    labels[positive_indices] = 1.0

    # Create timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='5min')
    symbols = np.array(['BTC-USD'] * n_samples)

    # Create dataset
    print("=== Creating SpikeDataset ===")
    dataset = SpikeDataset(
        sequences=sequences,
        labels=labels,
        timestamps=timestamps.values,
        symbols=symbols,
        feature_names=[f"feature_{i}" for i in range(n_features)]
    )

    print(f"\nDataset Statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test class weights
    print("\n=== Class Weights ===")
    class_weights = dataset.get_class_weights()
    print(f"  Negative class weight: {class_weights[0]:.4f}")
    print(f"  Positive class weight: {class_weights[1]:.4f}")

    # Test DataLoader
    print("\n=== Testing DataLoader ===")

    # Split dataset (simple split for testing)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    train_sequences = sequences[:train_size]
    train_labels = labels[:train_size]
    train_timestamps = timestamps.values[:train_size]
    train_symbols = symbols[:train_size]

    val_sequences = sequences[train_size:train_size+val_size]
    val_labels = labels[train_size:train_size+val_size]
    val_timestamps = timestamps.values[train_size:train_size+val_size]
    val_symbols = symbols[train_size:train_size+val_size]

    train_dataset = SpikeDataset(train_sequences, train_labels, train_timestamps, train_symbols)
    val_dataset = SpikeDataset(val_sequences, val_labels, val_timestamps, val_symbols)

    # Create loaders
    loaders = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=32,
        use_weighted_sampler=True
    )

    print(f"\nDataLoaders created:")
    print(f"  Training batches: {len(loaders['train'])}")
    print(f"  Validation batches: {len(loaders['val'])}")

    # Test batch
    print("\n=== Testing Batch Sampling ===")
    batch_X, batch_y = next(iter(loaders['train']))
    print(f"Batch shapes:")
    print(f"  X: {batch_X.shape}")
    print(f"  y: {batch_y.shape}")
    print(f"Batch positive ratio: {batch_y.mean():.4f} (expected ~0.5 with weighted sampling)")

    # Test target generation
    print("\n=== Testing Target Generation ===")

    # Create synthetic price data with spike
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = pd.Series(100.0, index=dates)

    # Add a spike at day 50 (30% increase)
    prices.iloc[50:55] = [105, 110, 130, 125, 120]
    prices.iloc[55:] = 110

    target_gen = SpikeTargetGenerator(forward_window=14, min_spike_threshold=0.15)
    targets = target_gen.generate_targets(prices, dates)

    print(f"Generated {targets.sum():.0f} spike targets")

    # Find where spike was detected
    spike_indices = targets[targets == 1].index
    if len(spike_indices) > 0:
        print(f"Spike detected at: {spike_indices[0]}")
        print(f"Actual spike occurred at: {dates[50]}")
        print(f"Detection window: {(spike_indices[0] - dates[50]).days} days before")

    # Analyze targets
    stats = target_gen.analyze_targets(targets, prices)
    print(f"\nTarget Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            if 'return' in key or 'ratio' in key:
                print(f"  {key}: {value*100:.2f}%")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n✓ Dataset module test complete!")
