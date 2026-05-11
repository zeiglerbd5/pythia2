"""
Fizzler Filter v2 — ML gate for loading strategy entries.

Uses XGBoost + CNN sequence features to predict whether a loading alert
will fizzle (stop-loss/timeout) or succeed (trailing stop).

Conservative threshold: filters ~18% of fizzlers while keeping ~92% of winners.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple
from loguru import logger

try:
    import xgboost as xgb
    import torch
    import torch.nn as nn
    HAS_ML = True
except ImportError:
    HAS_ML = False

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "fizzler_filter_v2"

# CNN architecture (must match training)
SEQ_LEN = 60
SEQ_CHANNELS = 5
EMBEDDING_DIM = 8


class CandleCNN(nn.Module):
    """Small 1D CNN — must match training architecture exactly."""
    def __init__(self, embedding_dim=8, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(SEQ_CHANNELS, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc_embed = nn.Linear(32, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        embed = self.relu(self.fc_embed(x))
        out = self.fc_out(self.dropout(embed))
        return out, embed

    def get_embedding(self, x):
        with torch.no_grad():
            _, embed = self.forward(x)
        return embed


class FizzlerFilter:
    """
    Predicts whether a loading alert will fizzle.

    Usage:
        ff = FizzlerFilter()
        should_enter, prob = ff.should_enter(components, candles_df)
        if should_enter:
            # proceed with entry
    """

    def __init__(self, threshold: float = 0.30):
        self.threshold = threshold
        self.enabled = False
        self.xgb_model = None
        self.cnn_model = None
        self.feature_cols = None
        self.tabular_features = None
        self.feature_medians = None
        self.stats = {'checked': 0, 'passed': 0, 'filtered': 0}

        if not HAS_ML:
            logger.warning("[FIZZLER] ML libraries not available, filter disabled")
            return

        try:
            self._load_models()
            self.enabled = True
            logger.info(f"[FIZZLER] Filter loaded (threshold={self.threshold}, "
                        f"{len(self.feature_cols)} features)")
        except Exception as e:
            logger.warning(f"[FIZZLER] Failed to load models, filter disabled: {e}")

    def _load_models(self):
        """Load XGBoost model, CNN encoder, and config."""
        config_path = MODEL_DIR / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        self.feature_cols = config['feature_cols']
        self.tabular_features = config['tabular_features']

        with open(MODEL_DIR / "feature_medians.json") as f:
            self.feature_medians = json.load(f)

        # Load XGBoost
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(str(MODEL_DIR / "fizzler_xgb_v2.json"))

        # Load CNN encoder
        self.cnn_model = CandleCNN(embedding_dim=EMBEDDING_DIM, dropout=0.0)
        state = torch.load(str(MODEL_DIR / "cnn_encoder.pt"), map_location='cpu')
        self.cnn_model.load_state_dict(state)
        self.cnn_model.eval()

    def _engineer_tabular_features(self, components: Dict) -> Dict:
        """Compute engineered features from raw score components."""
        features = {}
        for key in ['vol_trend', 'vol_last_vs_avg', 'vol_accel', 'natr',
                     'bb_width', 'momentum_1h', 'price_range', 'close_position',
                     'bot_net_pct', 'spread_pct', 'repeat_spiker', 'hour_utc', 'score']:
            val = components.get(key)
            if val is None and key in self.feature_medians:
                val = self.feature_medians[key]
            elif val is None:
                val = 0.0
            features[key] = float(val) if val is not None else 0.0

        # Engineered features (must match training)
        features['already_moved'] = features['momentum_1h'] * features['close_position']
        features['vol_momentum_ratio'] = features['vol_trend'] / (1 + abs(features['momentum_1h']))
        features['tightness'] = 1.0 / (1.0 + features['bb_width'])
        features['natr_x_range'] = features['natr'] * features['price_range']
        features['is_night_utc'] = int(0 <= features['hour_utc'] <= 8)
        features['is_us_open'] = int(13 <= features['hour_utc'] <= 17)
        features['high_momentum'] = int(features['momentum_1h'] > 5)
        features['low_close_pos'] = int(features['close_position'] < 0.3)
        features['vol_surge'] = int(features['vol_last_vs_avg'] > 3)

        return features

    def _extract_sequence(self, candles) -> Optional[np.ndarray]:
        """
        Extract normalized candle sequence from a list/array of candles.

        candles: list of dicts with keys [open, high, low, close, volume]
                 OR a pandas DataFrame with those columns, ordered oldest-first.
        Returns: (5, 60) numpy array or None if insufficient data.
        """
        try:
            import pandas as pd
            if isinstance(candles, pd.DataFrame):
                if len(candles) < 30:
                    return None
                df = candles.tail(SEQ_LEN)
                opens = df['open'].values.astype(float)
                highs = df['high'].values.astype(float)
                lows = df['low'].values.astype(float)
                closes = df['close'].values.astype(float)
                volumes = df['volume'].values.astype(float)
            elif isinstance(candles, (list, np.ndarray)):
                if len(candles) < 30:
                    return None
                candles = candles[-SEQ_LEN:]
                opens = np.array([c.get('open', c.get('o', 0)) for c in candles], dtype=float)
                highs = np.array([c.get('high', c.get('h', 0)) for c in candles], dtype=float)
                lows = np.array([c.get('low', c.get('l', 0)) for c in candles], dtype=float)
                closes = np.array([c.get('close', c.get('c', 0)) for c in candles], dtype=float)
                volumes = np.array([c.get('volume', c.get('v', 0)) for c in candles], dtype=float)
            else:
                return None

            # Normalize: prices as % change from first close
            base_price = closes[0] if closes[0] > 0 else 1.0
            norm_open = (opens / base_price - 1) * 100
            norm_high = (highs / base_price - 1) * 100
            norm_low = (lows / base_price - 1) * 100
            norm_close = (closes / base_price - 1) * 100

            # Volume as ratio to mean
            mean_vol = volumes.mean() if volumes.mean() > 0 else 1.0
            norm_vol = volumes / mean_vol

            # Stack and pad to 60 if needed
            seq = np.stack([norm_open, norm_high, norm_low, norm_close, norm_vol])  # (5, N)
            if seq.shape[1] < SEQ_LEN:
                pad = np.zeros((5, SEQ_LEN - seq.shape[1]))
                seq = np.concatenate([pad, seq], axis=1)  # left-pad

            return seq.astype(np.float32)

        except Exception as e:
            logger.debug(f"[FIZZLER] Sequence extraction error: {e}")
            return None

    def predict(self, components: Dict, candles=None) -> Tuple[float, bool]:
        """
        Predict fizzle probability.

        Args:
            components: score components dict from compute_loading_score()
            candles: optional 60 pre-entry 1m candles (DataFrame or list of dicts)

        Returns:
            (fizzle_probability, should_enter)
        """
        if not self.enabled:
            return 0.0, True

        self.stats['checked'] += 1

        try:
            # Tabular features
            features = self._engineer_tabular_features(components)

            # Sequence features from CNN
            seq_features = np.zeros(EMBEDDING_DIM)
            if candles is not None:
                seq = self._extract_sequence(candles)
                if seq is not None:
                    tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, 5, 60)
                    embedding = self.cnn_model.get_embedding(tensor)
                    seq_features = embedding.squeeze(0).numpy()

            for i in range(EMBEDDING_DIM):
                features[f'seq_feat_{i}'] = float(seq_features[i])

            # Build feature vector in correct order
            feature_vec = [features.get(col, self.feature_medians.get(col, 0.0))
                           for col in self.feature_cols]

            # Predict
            dmatrix = xgb.DMatrix(np.array([feature_vec]), feature_names=self.feature_cols)
            fizzle_prob = float(self.xgb_model.predict(dmatrix)[0])

            should_enter = fizzle_prob < self.threshold

            if should_enter:
                self.stats['passed'] += 1
            else:
                self.stats['filtered'] += 1

            return fizzle_prob, should_enter

        except Exception as e:
            logger.debug(f"[FIZZLER] Prediction error: {e}")
            self.stats['passed'] += 1
            return 0.0, True  # fail open — don't block on errors
