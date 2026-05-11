#!/usr/bin/env python3
"""
Fizzler Filter v2: Tabular + Sequence Features

Extends the v1 XGBoost fizzler filter by extracting learned features from
the 60 1-minute candles immediately before each trade entry using a small 1D CNN.

The CNN produces an 8-dimensional embedding per trade. These "sequence features"
are combined with the original 22 tabular features and fed into a new XGBoost model.

Architecture:
  CNN: Conv1d(5,16,k=5) -> ReLU -> Conv1d(16,32,k=5) -> ReLU
       -> AdaptiveAvgPool1d(1) -> Linear(32,8) -> Dropout
  The 8-dim penultimate layer is the sequence embedding.

Out-of-fold generation:
  5-fold CV is used to produce sequence features for every sample without leakage.
  Each fold trains a separate CNN and generates embeddings for the held-out samples.

Data sources:
  - paper_trades.db: labeled loading_v1 trades
  - elite_movers.duckdb: entry-time tabular features (Apr 17+)
  - backtest_ohlcv.duckdb: 1-min candles for sequence extraction + older feature computation
"""

import sys
sys.path.insert(0, '/Users/bz/Pythia2/venv/lib/python3.9/site-packages')

import os
import json
import warnings
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

MODEL_DIR_V1 = '/Users/bz/Pythia2/models/fizzler_filter'
MODEL_DIR_V2 = '/Users/bz/Pythia2/models/fizzler_filter_v2'
os.makedirs(MODEL_DIR_V2, exist_ok=True)

DEVICE = 'cpu'  # Tiny model, CPU is fine for 532 samples

# ── Feature lists (same as v1) ─────────────────────────────────────────────
BASE_FEATURES = [
    'vol_trend', 'vol_last_vs_avg', 'vol_accel', 'natr', 'bb_width',
    'momentum_1h', 'price_range', 'close_position', 'bot_net_pct',
    'spread_pct', 'repeat_spiker', 'hour_utc', 'score'
]

ENGINEERED_FEATURES = [
    'already_moved', 'vol_momentum_ratio', 'tightness', 'natr_x_range',
    'is_night_utc', 'is_us_open', 'high_momentum', 'low_close_pos', 'vol_surge',
]

TABULAR_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
SEQ_FEATURES = [f'seq_feat_{i}' for i in range(8)]
ALL_FEATURES_V2 = TABULAR_FEATURES + SEQ_FEATURES

SEQ_LEN = 60        # 60 one-minute candles before entry
MIN_CANDLES = 30    # Minimum candles required
SEQ_CHANNELS = 5    # open, high, low, close, volume (normalized)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Load trades and build tabular features (reused from v1)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Loading trades and building tabular features")
print("=" * 70)

# Load paper trades
sqlite_con = sqlite3.connect('/Users/bz/Pythia2/paper_trades.db')
trades_df = pd.read_sql_query("""
    SELECT id, symbol, entry_price, entry_time, exit_price, exit_time,
           exit_reason, position_size, realized_pnl, is_open
    FROM trades
    WHERE strategy = 'loading_v1' AND is_open = 0
    ORDER BY entry_time
""", sqlite_con)
sqlite_con.close()

trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

# Deduplicate: keep the phase1 ($200) entry per (symbol, entry_time rounded to minute)
trades_df['entry_minute'] = trades_df['entry_time'].dt.tz_localize(None).dt.floor('min')
trades_df = trades_df.sort_values('position_size').drop_duplicates(
    subset=['symbol', 'entry_minute'], keep='first'
).sort_values('entry_time').reset_index(drop=True)

# Label
trades_df['is_winner'] = (
    (trades_df['exit_reason'] == 'trailing_stop') |
    (trades_df['realized_pnl'] > 0)
).astype(int)
trades_df = trades_df[trades_df['exit_reason'] != 'orphaned_restart'].reset_index(drop=True)

print(f"Total trades: {len(trades_df)}")
print(f"  Winners: {trades_df['is_winner'].sum()} ({trades_df['is_winner'].mean()*100:.1f}%)")
print(f"  Fizzlers: {(1-trades_df['is_winner']).sum()}")

# ── Join elite_movers tabular features ──────────────────────────────────
elite_con = duckdb.connect('/Users/bz/Pythia2/data/elite_movers.duckdb', read_only=True)
elite_entries = elite_con.execute("""
    SELECT timestamp, symbol, score, vol_trend, vol_last_vs_avg, vol_accel,
           natr, bb_width, momentum_1h, price_range, close_position,
           bot_net_pct, spread_pct, repeat_spiker, hour_utc
    FROM trades WHERE event_type = 'entry' ORDER BY timestamp
""").fetchdf()
symbol_stats = elite_con.execute("SELECT symbol, is_repeat_spiker FROM symbol_stats").fetchdf()
repeat_spikers = set(symbol_stats[symbol_stats['is_repeat_spiker']]['symbol'].tolist())
elite_con.close()

elite_entries['timestamp'] = pd.to_datetime(elite_entries['timestamp']).dt.tz_localize(None)
elite_entries['entry_minute'] = elite_entries['timestamp'].dt.floor('min')

merged = trades_df.merge(
    elite_entries.drop(columns=['timestamp']),
    on=['symbol', 'entry_minute'], how='left', suffixes=('', '_elite')
)
has_elite = merged['vol_trend'].notna()
print(f"Trades with elite features: {has_elite.sum()}, needing OHLCV computation: {(~has_elite).sum()}")

# ── Compute tabular features from OHLCV for older trades ────────────────
needs_features = merged[~has_elite].copy()
ohlcv_con = duckdb.connect('/Users/bz/Pythia2/data/backtest_ohlcv.duckdb', read_only=True)

if len(needs_features) > 0:
    computed_features = []
    for i, (idx, row) in enumerate(needs_features.iterrows()):
        symbol = row['symbol']
        entry_time = row['entry_time']
        if i % 100 == 0:
            print(f"  Computing tabular features: {i}/{len(needs_features)}...")
        try:
            entry_ts = entry_time.tz_localize(None).strftime('%Y-%m-%d %H:%M:%S') if entry_time.tzinfo else entry_time.strftime('%Y-%m-%d %H:%M:%S')
            candles = ohlcv_con.execute(f"""
                SELECT timestamp, open, high, low, close, volume FROM ohlcv
                WHERE symbol = '{symbol}' AND timestamp < '{entry_ts}'
                ORDER BY timestamp DESC LIMIT 60
            """).fetchdf()

            if len(candles) < 20:
                computed_features.append((idx, {}))
                continue

            candles = candles.sort_values('timestamp').reset_index(drop=True)
            n = len(candles)

            if n >= 40:
                vol_trend = candles['volume'].iloc[-20:].mean() / max(candles['volume'].iloc[:20].mean(), 1e-10)
            else:
                half = n // 2
                vol_trend = candles['volume'].iloc[half:].mean() / max(candles['volume'].iloc[:half].mean(), 1e-10)

            avg_vol = candles['volume'].mean()
            vol_last_vs_avg = candles['volume'].iloc[-1] / max(avg_vol, 1e-10)
            vol_accel = candles['volume'].iloc[-5:].mean() / max(candles['volume'].iloc[-10:-5].mean(), 1e-10) if n >= 10 else 1.0

            highs, lows, closes = candles['high'].values, candles['low'].values, candles['close'].values
            tr = np.maximum(highs[1:]-lows[1:], np.maximum(np.abs(highs[1:]-closes[:-1]), np.abs(lows[1:]-closes[:-1])))
            atr14 = np.mean(tr[-14:]) if len(tr) >= 14 else (np.mean(tr) if len(tr) > 0 else 0)
            natr = (atr14 / max(closes[-1], 1e-10)) * 100

            sma20 = np.mean(closes[-20:]) if n >= 20 else np.mean(closes)
            std20 = np.std(closes[-20:]) if n >= 20 else np.std(closes)
            bb_width = (4 * std20) / max(sma20, 1e-10)

            momentum_1h = ((closes[-1] - closes[0]) / max(closes[0], 1e-10)) * 100
            range_high, range_low = np.max(highs), np.min(lows)
            price_range = ((range_high - range_low) / max(range_low, 1e-10)) * 100
            close_position = (closes[-1] - range_low) / (range_high - range_low) if range_high > range_low else 0.5
            hour_utc = entry_time.hour if entry_time.tzinfo is None else entry_time.tz_localize(None).hour

            computed_features.append((idx, {
                'vol_trend': vol_trend, 'vol_last_vs_avg': vol_last_vs_avg,
                'vol_accel': vol_accel, 'natr': natr, 'bb_width': bb_width,
                'momentum_1h': momentum_1h, 'price_range': price_range,
                'close_position': close_position, 'bot_net_pct': np.nan,
                'spread_pct': np.nan, 'repeat_spiker': symbol in repeat_spikers,
                'hour_utc': hour_utc, 'score': np.nan,
            }))
        except Exception:
            computed_features.append((idx, {}))

    n_success = sum(1 for _, d in computed_features if 'vol_trend' in d)
    print(f"  Computed features for {n_success}/{len(needs_features)} trades")
    for idx, feat_dict in computed_features:
        for col, val in feat_dict.items():
            merged.loc[idx, col] = val

# ── Engineer features ───────────────────────────────────────────────────
merged['repeat_spiker'] = merged['repeat_spiker'].fillna(False).astype(int)
merged = merged.dropna(subset=['vol_trend']).reset_index(drop=True)

for col in ['bot_net_pct', 'spread_pct', 'score']:
    n_na = merged[col].isna().sum()
    if n_na > 0:
        merged[col] = merged[col].fillna(merged[col].median())

merged['already_moved'] = merged['momentum_1h'] * merged['close_position']
merged['vol_momentum_ratio'] = merged['vol_trend'] / (1 + np.abs(merged['momentum_1h']))
merged['tightness'] = 1.0 / (1.0 + merged['bb_width'])
merged['natr_x_range'] = merged['natr'] * merged['price_range']
merged['is_night_utc'] = ((merged['hour_utc'] >= 0) & (merged['hour_utc'] <= 8)).astype(int)
merged['is_us_open'] = ((merged['hour_utc'] >= 13) & (merged['hour_utc'] <= 17)).astype(int)
merged['high_momentum'] = (merged['momentum_1h'] > 5).astype(int)
merged['low_close_pos'] = (merged['close_position'] < 0.3).astype(int)
merged['vol_surge'] = (merged['vol_last_vs_avg'] > 3).astype(int)

X_tab = merged[TABULAR_FEATURES].values.astype(np.float64)
X_tab = np.nan_to_num(X_tab, nan=0.0, posinf=100.0, neginf=-100.0)
y = merged['is_winner'].values

print(f"\nTabular features ready: X={X_tab.shape}, positive rate={y.mean():.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Extract 60-candle sequences before each trade entry
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Extracting 60-candle sequences from backtest OHLCV")
print("=" * 70)

sequences = []  # Will hold (60, 5) arrays or None
valid_mask = []  # True if sequence is usable

for i, row in merged.iterrows():
    symbol = row['symbol']
    entry_time = row['entry_time']
    entry_ts = entry_time.tz_localize(None).strftime('%Y-%m-%d %H:%M:%S') if entry_time.tzinfo else entry_time.strftime('%Y-%m-%d %H:%M:%S')

    if i % 100 == 0:
        print(f"  Extracting sequences: {i}/{len(merged)}...")

    try:
        candles = ohlcv_con.execute(f"""
            SELECT open, high, low, close, volume FROM ohlcv
            WHERE symbol = '{symbol}' AND timestamp < '{entry_ts}'
            ORDER BY timestamp DESC LIMIT {SEQ_LEN}
        """).fetchdf()

        n = len(candles)

        if n < MIN_CANDLES:
            sequences.append(np.zeros((SEQ_LEN, SEQ_CHANNELS), dtype=np.float32))
            valid_mask.append(False)
            continue

        # Reverse to chronological order
        candles = candles.iloc[::-1].reset_index(drop=True)

        # Normalize prices: % change from first candle's close
        base_close = candles['close'].iloc[0]
        if base_close < 1e-10:
            sequences.append(np.zeros((SEQ_LEN, SEQ_CHANNELS), dtype=np.float32))
            valid_mask.append(False)
            continue

        norm_open = ((candles['open'].values / base_close) - 1.0) * 100
        norm_high = ((candles['high'].values / base_close) - 1.0) * 100
        norm_low = ((candles['low'].values / base_close) - 1.0) * 100
        norm_close = ((candles['close'].values / base_close) - 1.0) * 100

        # Normalize volume: ratio to mean volume of the sequence
        mean_vol = candles['volume'].mean()
        norm_vol = candles['volume'].values / max(mean_vol, 1e-10)

        # Stack into (n, 5) array
        seq = np.stack([norm_open, norm_high, norm_low, norm_close, norm_vol], axis=1).astype(np.float32)

        # Left-pad with zeros if fewer than 60 candles
        if n < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - n, SEQ_CHANNELS), dtype=np.float32)
            seq = np.vstack([pad, seq])

        # Clip extreme values
        seq = np.clip(seq, -50, 50)

        sequences.append(seq)
        valid_mask.append(True)

    except Exception:
        sequences.append(np.zeros((SEQ_LEN, SEQ_CHANNELS), dtype=np.float32))
        valid_mask.append(False)

ohlcv_con.close()

valid_mask = np.array(valid_mask)
sequences = np.array(sequences)  # (N, 60, 5)

print(f"\nSequences extracted: {valid_mask.sum()}/{len(valid_mask)} valid")
print(f"Excluded (< {MIN_CANDLES} candles): {(~valid_mask).sum()}")

# For invalid sequences, we'll still include them with zero sequences
# but flag them. The CNN will produce near-zero embeddings for these.
# This is better than dropping samples from an already small dataset.
print(f"Sequence array shape: {sequences.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Define and train CNN feature extractor with out-of-fold embeddings
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Training CNN feature extractor (5-fold out-of-fold)")
print("=" * 70)


class CandleCNN(nn.Module):
    """Small 1D CNN to learn sequence embeddings from candle data.

    Input: (batch, 5, 60)  -- 5 channels, 60 timesteps
    Output: (batch, 1) for classification, but we extract the 8-dim embedding.
    """
    def __init__(self, embedding_dim=8, dropout=0.3):
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
        """x: (batch, channels=5, seq_len=60)"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)      # (batch, 32)
        x = self.dropout(x)
        embed = self.relu(self.fc_embed(x))  # (batch, 8)
        out = self.fc_out(self.dropout(embed))  # (batch, 1)
        return out, embed

    def get_embedding(self, x):
        """Extract just the embedding (inference mode)."""
        with torch.no_grad():
            _, embed = self.forward(x)
        return embed


def train_cnn_fold(X_seq_train, y_train, X_seq_val, y_val,
                   n_epochs=50, lr=1e-3, weight_decay=1e-3, patience=10):
    """Train one CNN fold and return (model, val_embeddings, train_loss_history)."""

    # Class weights for imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)

    model = CandleCNN(embedding_dim=8, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Transpose sequences: (N, 60, 5) -> (N, 5, 60) for Conv1d
    X_train_t = torch.tensor(X_seq_train.transpose(0, 2, 1), dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    X_val_t = torch.tensor(X_seq_val.transpose(0, 2, 1), dtype=torch.float32, device=DEVICE)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(X_train_t)

        # Validation loss for early stopping
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val_t)
            y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE).unsqueeze(1)
            val_loss = criterion(val_logits, y_val_t).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Load best model and extract embeddings
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_embeddings = model.get_embedding(X_val_t).cpu().numpy()

    return model, val_embeddings, epoch + 1


# ── Run 5-fold out-of-fold CNN embedding generation ─────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_embeddings = np.zeros((len(y), 8), dtype=np.float32)
cnn_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tab, y)):
    X_seq_train = sequences[train_idx]
    X_seq_val = sequences[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    model, val_embeds, n_epochs_used = train_cnn_fold(
        X_seq_train, y_train, X_seq_val, y_val,
        n_epochs=50, lr=1e-3, weight_decay=1e-3, patience=10
    )

    oof_embeddings[val_idx] = val_embeds
    cnn_models.append(model)

    # Quick check: CNN-only AUC on this fold
    cnn_probs = torch.sigmoid(torch.tensor(
        model(torch.tensor(X_seq_val.transpose(0, 2, 1), dtype=torch.float32, device=DEVICE))[0].detach().cpu().numpy()
    )).numpy().flatten()
    try:
        cnn_auc = roc_auc_score(y_val, cnn_probs)
    except ValueError:
        cnn_auc = 0.5
    print(f"  Fold {fold+1}: CNN trained {n_epochs_used} epochs, CNN-only AUC={cnn_auc:.4f}")

# Overall CNN-only AUC (out-of-fold)
# We need to compute CNN logits for OOF -- let's do it per fold
oof_cnn_probs = np.zeros(len(y), dtype=np.float32)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_tab, y)):
    model = cnn_models[fold]
    model.eval()
    X_val_t = torch.tensor(sequences[val_idx].transpose(0, 2, 1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits, _ = model(X_val_t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    oof_cnn_probs[val_idx] = probs

cnn_only_auc = roc_auc_score(y, oof_cnn_probs)
print(f"\nCNN-only OOF AUC: {cnn_only_auc:.4f}")
print(f"Sequence embedding stats:")
print(f"  Mean: {oof_embeddings.mean(axis=0).round(3)}")
print(f"  Std:  {oof_embeddings.std(axis=0).round(3)}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Train XGBoost on tabular-only (v1 baseline reproduction)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: XGBoost baseline (tabular only) -- reproducing v1")
print("=" * 70)

n_neg = (y == 0).sum()
n_pos = (y == 1).sum()
scale_pos_weight = n_neg / n_pos

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 3,
    'min_child_weight': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'learning_rate': 0.03,
    'scale_pos_weight': scale_pos_weight,
    'reg_alpha': 1.0,
    'reg_lambda': 3.0,
    'gamma': 1.0,
    'seed': 42,
    'verbosity': 0,
}

# ── V1: Tabular only ───────────────────────────────────────────────────
v1_y_true, v1_y_prob = [], []
v1_fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tab, y)):
    dtrain = xgb.DMatrix(X_tab[train_idx], label=y[train_idx], feature_names=TABULAR_FEATURES)
    dval = xgb.DMatrix(X_tab[val_idx], label=y[val_idx], feature_names=TABULAR_FEATURES)
    model = xgb.train(xgb_params, dtrain, num_boost_round=500,
                       evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
    y_prob = model.predict(dval)
    auc = roc_auc_score(y[val_idx], y_prob)
    v1_fold_aucs.append(auc)
    v1_y_true.extend(y[val_idx].tolist())
    v1_y_prob.extend(y_prob.tolist())

v1_y_true = np.array(v1_y_true)
v1_y_prob = np.array(v1_y_prob)
v1_auc = np.mean(v1_fold_aucs)
v1_prauc = average_precision_score(v1_y_true, v1_y_prob)
print(f"V1 (tabular only): AUC={v1_auc:.4f} +/- {np.std(v1_fold_aucs):.4f}, PR-AUC={v1_prauc:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Train XGBoost on tabular + sequence features (v2)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: XGBoost v2 (tabular + sequence features)")
print("=" * 70)

X_combined = np.hstack([X_tab, oof_embeddings])
print(f"Combined feature matrix: {X_combined.shape} ({len(TABULAR_FEATURES)} tabular + 8 sequence)")

v2_y_true, v2_y_prob = [], []
v2_fold_aucs = []
v2_fold_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y)):
    dtrain = xgb.DMatrix(X_combined[train_idx], label=y[train_idx], feature_names=ALL_FEATURES_V2)
    dval = xgb.DMatrix(X_combined[val_idx], label=y[val_idx], feature_names=ALL_FEATURES_V2)
    model = xgb.train(xgb_params, dtrain, num_boost_round=500,
                       evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
    y_prob = model.predict(dval)
    auc = roc_auc_score(y[val_idx], y_prob)
    v2_fold_aucs.append(auc)
    v2_fold_models.append(model)
    v2_y_true.extend(y[val_idx].tolist())
    v2_y_prob.extend(y_prob.tolist())
    print(f"  Fold {fold+1}: AUC={auc:.4f}, best_iteration={model.best_iteration}")

v2_y_true = np.array(v2_y_true)
v2_y_prob = np.array(v2_y_prob)
v2_auc = np.mean(v2_fold_aucs)
v2_prauc = average_precision_score(v2_y_true, v2_y_prob)
print(f"\nV2 (tabular + sequence): AUC={v2_auc:.4f} +/- {np.std(v2_fold_aucs):.4f}, PR-AUC={v2_prauc:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Head-to-head comparison
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Head-to-Head Comparison")
print("=" * 70)

delta_auc = v2_auc - v1_auc
delta_prauc = v2_prauc - v1_prauc

print(f"\n{'Metric':<25} {'V1 (Tabular)':>14} {'V2 (Tab+Seq)':>14} {'Delta':>10}")
print("-" * 65)
print(f"{'ROC AUC':<25} {v1_auc:>14.4f} {v2_auc:>14.4f} {delta_auc:>+10.4f}")
print(f"{'PR AUC':<25} {v1_prauc:>14.4f} {v2_prauc:>14.4f} {delta_prauc:>+10.4f}")
print(f"{'CNN-only AUC':<25} {cnn_only_auc:>14.4f} {'---':>14} {'---':>10}")

improved = delta_auc > 0.005  # Meaningful improvement threshold
if improved:
    print(f"\n  >> V2 IMPROVES over V1 by {delta_auc:+.4f} AUC. Sequence features add value.")
elif delta_auc > -0.005:
    print(f"\n  >> V2 is COMPARABLE to V1 (delta={delta_auc:+.4f}). Sequence features are neutral.")
else:
    print(f"\n  >> V2 is WORSE than V1 by {delta_auc:+.4f} AUC. Sequence features may be adding noise.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Threshold analysis for v2
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Threshold Analysis (V2)")
print("=" * 70)

baseline_wr = n_pos / (n_pos + n_neg) * 100

print(f"\n{'Threshold':>10} {'Fizzlers Filtered':>18} {'Winners Kept':>14} {'Winners Lost':>13} "
      f"{'Remaining':>10} {'Win Rate':>10} {'Lift':>6}")
print("-" * 88)

threshold_results_v2 = []
for threshold in np.arange(0.10, 0.65, 0.05):
    threshold = round(threshold, 2)
    passes = v2_y_prob >= threshold
    fizzlers_filtered = ((v2_y_true == 0) & ~passes).sum()
    winners_kept = ((v2_y_true == 1) & passes).sum()
    winners_lost = ((v2_y_true == 1) & ~passes).sum()
    fizzler_filter_rate = fizzlers_filtered / n_neg * 100
    winner_keep_rate = winners_kept / n_pos * 100
    remaining = passes.sum()
    precision = winners_kept / remaining * 100 if remaining > 0 else 0
    lift = precision / baseline_wr

    marker = ""
    if winner_keep_rate >= 90:
        marker = " <-- 90%+ retained"
    elif winner_keep_rate >= 85:
        marker = " <-- 85%+ retained"

    print(f"{threshold:>10.2f} {fizzler_filter_rate:>16.1f}% {winner_keep_rate:>12.1f}% "
          f"{winners_lost:>13d} {remaining:>10d} {precision:>9.1f}% {lift:>5.2f}x{marker}")

    threshold_results_v2.append({
        'threshold': threshold,
        'fizzlers_filtered_pct': round(fizzler_filter_rate, 1),
        'winners_kept_pct': round(winner_keep_rate, 1),
        'remaining_trades': int(remaining),
        'precision_pct': round(precision, 1),
        'winners_lost': int(winners_lost),
        'lift': round(lift, 2),
    })

# Find recommended thresholds
recommended_v2 = None
for r in threshold_results_v2:
    if r['winners_kept_pct'] >= 90.0:
        if recommended_v2 is None or r['fizzlers_filtered_pct'] > recommended_v2['fizzlers_filtered_pct']:
            recommended_v2 = r

aggressive_v2 = None
for r in threshold_results_v2:
    if r['winners_kept_pct'] >= 85.0:
        if aggressive_v2 is None or r['fizzlers_filtered_pct'] > aggressive_v2['fizzlers_filtered_pct']:
            aggressive_v2 = r

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Train final model on all data and save
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: Training final v2 model on all data")
print("=" * 70)

# Train final CNN on all data (for inference on new trades)
print("  Training final CNN on all data...")
final_cnn = CandleCNN(embedding_dim=8, dropout=0.3).to(DEVICE)
X_all_t = torch.tensor(sequences.transpose(0, 2, 1), dtype=torch.float32, device=DEVICE)
y_all_t = torch.tensor(y, dtype=torch.float32, device=DEVICE).unsqueeze(1)
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(final_cnn.parameters(), lr=1e-3, weight_decay=1e-3)

train_ds = TensorDataset(X_all_t, y_all_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

for epoch in range(30):  # Conservative: no validation set, so fewer epochs
    final_cnn.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits, _ = final_cnn(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

final_cnn.eval()
print("  CNN training complete.")

# Get full-data embeddings from final CNN for final XGBoost
with torch.no_grad():
    all_embeddings = final_cnn.get_embedding(X_all_t).cpu().numpy()

X_combined_final = np.hstack([X_tab, all_embeddings])

# Train final XGBoost
dtrain_full = xgb.DMatrix(X_combined_final, label=y, feature_names=ALL_FEATURES_V2)
final_xgb = xgb.train(xgb_params, dtrain_full, num_boost_round=200, verbose_eval=False)

# Feature importance
importance = final_xgb.get_score(importance_type='gain')
imp_df = pd.DataFrame([
    {'feature': k, 'importance': v} for k, v in importance.items()
]).sort_values('importance', ascending=False)

print("\nFeature Importance (gain) -- V2 combined model:")
for _, row in imp_df.iterrows():
    bar = '#' * int(row['importance'] / imp_df['importance'].max() * 40)
    is_seq = row['feature'].startswith('seq_feat_')
    tag = " [SEQ]" if is_seq else ""
    print(f"  {row['feature']:>25s}: {row['importance']:>10.2f}  {bar}{tag}")

# Show which sequence features made it into the model
seq_in_model = [f for f in SEQ_FEATURES if f in importance]
seq_not_in_model = [f for f in SEQ_FEATURES if f not in importance]
print(f"\nSequence features used by XGBoost: {len(seq_in_model)}/8")
if seq_not_in_model:
    print(f"Sequence features NOT used (zero importance): {seq_not_in_model}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: Save model artifacts
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 9: Saving model artifacts")
print("=" * 70)

# Save XGBoost model
xgb_path = os.path.join(MODEL_DIR_V2, 'fizzler_xgb_v2.json')
final_xgb.save_model(xgb_path)
print(f"  XGBoost model: {xgb_path}")

# Save CNN encoder
cnn_path = os.path.join(MODEL_DIR_V2, 'cnn_encoder.pt')
torch.save(final_cnn.state_dict(), cnn_path)
print(f"  CNN encoder: {cnn_path}")

# Save per-fold CNN models (for generating OOF embeddings on retraining)
for fold_i, cnn_model in enumerate(cnn_models):
    fold_path = os.path.join(MODEL_DIR_V2, f'cnn_encoder_fold{fold_i}.pt')
    torch.save(cnn_model.state_dict(), fold_path)
print(f"  Per-fold CNN models: {MODEL_DIR_V2}/cnn_encoder_fold*.pt")

# Save config
config_v2 = {
    'feature_cols': ALL_FEATURES_V2,
    'tabular_features': TABULAR_FEATURES,
    'sequence_features': SEQ_FEATURES,
    'seq_len': SEQ_LEN,
    'min_candles': MIN_CANDLES,
    'cnn_embedding_dim': 8,
    'xgb_params': xgb_params,
    'threshold_analysis': threshold_results_v2,
    'recommended_threshold': recommended_v2['threshold'] if recommended_v2 else 0.3,
    'aggressive_threshold': aggressive_v2['threshold'] if aggressive_v2 else None,
    'v1_cv_auc': round(float(v1_auc), 4),
    'v2_cv_auc': round(float(v2_auc), 4),
    'v2_cv_auc_std': round(float(np.std(v2_fold_aucs)), 4),
    'v2_pr_auc': round(float(v2_prauc), 4),
    'cnn_only_auc': round(float(cnn_only_auc), 4),
    'delta_auc': round(float(delta_auc), 4),
    'n_training_samples': int(len(X_tab)),
    'n_winners': int(n_pos),
    'n_fizzlers': int(n_neg),
    'baseline_win_rate': round(float(baseline_wr), 1),
    'sequence_features_used_by_xgb': seq_in_model,
    'training_date': datetime.now().isoformat(),
    'feature_importance': {row['feature']: round(row['importance'], 2) for _, row in imp_df.iterrows()},
}
config_path = os.path.join(MODEL_DIR_V2, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config_v2, f, indent=2)
print(f"  Config: {config_path}")

# Save feature medians (for inference-time imputation)
feature_medians = {}
for i, col in enumerate(TABULAR_FEATURES):
    feature_medians[col] = float(np.median(X_tab[:, i]))
for i, col in enumerate(SEQ_FEATURES):
    feature_medians[col] = 0.0  # Sequence features default to 0 if CNN not available
medians_path = os.path.join(MODEL_DIR_V2, 'feature_medians.json')
with open(medians_path, 'w') as f:
    json.dump(feature_medians, f, indent=2)
print(f"  Feature medians: {medians_path}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 10: Generate comparison plots
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 10: Generating comparison plots")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Pythia2 Fizzler Filter V2 -- Tabular + Sequence Features', fontsize=14, fontweight='bold')

# 1. ROC curves: v1 vs v2
ax = axes[0, 0]
fpr1, tpr1, _ = roc_curve(v1_y_true, v1_y_prob)
fpr2, tpr2, _ = roc_curve(v2_y_true, v2_y_prob)
ax.plot(fpr1, tpr1, 'b-', linewidth=2, label=f'V1 tabular AUC={v1_auc:.3f}')
ax.plot(fpr2, tpr2, 'r-', linewidth=2, label=f'V2 tab+seq AUC={v2_auc:.3f}')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve: V1 vs V2')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# 2. PR curves: v1 vs v2
ax = axes[0, 1]
p1, r1, _ = precision_recall_curve(v1_y_true, v1_y_prob)
p2, r2, _ = precision_recall_curve(v2_y_true, v2_y_prob)
ax.plot(r1, p1, 'b-', linewidth=2, label=f'V1 PR-AUC={v1_prauc:.3f}')
ax.plot(r2, p2, 'r-', linewidth=2, label=f'V2 PR-AUC={v2_prauc:.3f}')
ax.axhline(y=baseline_wr/100, color='gray', linestyle='--', alpha=0.3, label=f'Baseline={baseline_wr:.1f}%')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall: V1 vs V2')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# 3. Feature importance (top 15) highlighting sequence features
ax = axes[0, 2]
imp_top = imp_df.head(15).sort_values('importance', ascending=True)
colors = ['#E91E63' if f.startswith('seq_feat_') else '#2196F3' if f in BASE_FEATURES else '#FF9800'
          for f in imp_top['feature']]
ax.barh(imp_top['feature'], imp_top['importance'], color=colors)
ax.set_title('Feature Importance (V2)')
ax.set_xlabel('Gain')
ax.legend(handles=[
    Patch(facecolor='#2196F3', label='Base tabular'),
    Patch(facecolor='#FF9800', label='Engineered'),
    Patch(facecolor='#E91E63', label='Sequence (CNN)'),
], loc='lower right', fontsize=8)

# 4. Threshold trade-off for v2
ax = axes[1, 0]
thresholds = [r['threshold'] for r in threshold_results_v2]
fizzler_pcts = [r['fizzlers_filtered_pct'] for r in threshold_results_v2]
winner_pcts = [r['winners_kept_pct'] for r in threshold_results_v2]
ax.plot(thresholds, fizzler_pcts, 'r-o', label='Fizzlers Filtered %', linewidth=2, markersize=5)
ax.plot(thresholds, winner_pcts, 'g-o', label='Winners Kept %', linewidth=2, markersize=5)
ax.axhline(y=90, color='green', linestyle='--', alpha=0.3, label='90% Winner Retention')
if recommended_v2:
    ax.axvline(x=recommended_v2['threshold'], color='blue', linestyle=':', alpha=0.5,
               label=f"Recommended ({recommended_v2['threshold']})")
ax.set_xlabel('Threshold')
ax.set_ylabel('Percentage')
ax.set_title('V2 Threshold Trade-off')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 5. Score distributions: v1 vs v2
ax = axes[1, 1]
bins = np.linspace(0, 1, 25)
ax.hist(v1_y_prob[v1_y_true == 1], bins=bins, alpha=0.4, label='V1 Winners', color='green', density=True)
ax.hist(v1_y_prob[v1_y_true == 0], bins=bins, alpha=0.4, label='V1 Fizzlers', color='red', density=True)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('V1 Score Distribution')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

ax = axes[1, 2]
ax.hist(v2_y_prob[v2_y_true == 1], bins=bins, alpha=0.4, label='V2 Winners', color='green', density=True)
ax.hist(v2_y_prob[v2_y_true == 0], bins=bins, alpha=0.4, label='V2 Fizzlers', color='red', density=True)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('V2 Score Distribution')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR_V2, 'v1_vs_v2_comparison.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Comparison plot: {plot_path}")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

rec_v2 = recommended_v2 if recommended_v2 else {}

print(f"""
Model Comparison:
  V1 (tabular only):     ROC AUC = {v1_auc:.4f}, PR-AUC = {v1_prauc:.4f}
  V2 (tabular + seq):    ROC AUC = {v2_auc:.4f}, PR-AUC = {v2_prauc:.4f}
  CNN standalone:         ROC AUC = {cnn_only_auc:.4f}
  Delta (V2 - V1):       AUC {delta_auc:+.4f}, PR-AUC {delta_prauc:+.4f}

Training Data:
  Samples: {len(y)} ({n_pos} winners, {n_neg} fizzlers)
  Sequences valid: {valid_mask.sum()}/{len(valid_mask)}

Sequence Features Used by XGBoost: {len(seq_in_model)}/8
  {seq_in_model}
""")

if rec_v2:
    print(f"""V2 Recommended Threshold = {rec_v2.get('threshold', 'N/A')}:
  Fizzlers filtered: {rec_v2.get('fizzlers_filtered_pct', 'N/A')}%
  Winners retained:  {rec_v2.get('winners_kept_pct', 'N/A')}%
  Win rate lift:     {rec_v2.get('lift', 'N/A')}x
""")

verdict = "IMPROVED" if improved else "NO IMPROVEMENT"
print(f"Verdict: {verdict}")
if not improved:
    print("  The 60-candle pre-entry sequences did not meaningfully improve discrimination")
    print("  beyond what the tabular features already capture. This is common with small")
    print("  datasets (532 samples) -- the CNN may not have enough data to learn robust")
    print("  patterns. Consider: (a) more training data, (b) hand-crafted sequence stats,")
    print("  (c) pre-training the CNN on a self-supervised task.")

print(f"""
Saved Artifacts:
  XGBoost model:    {xgb_path}
  CNN encoder:      {cnn_path}
  Fold CNNs:        {MODEL_DIR_V2}/cnn_encoder_fold*.pt
  Config:           {config_path}
  Feature medians:  {medians_path}
  Comparison plot:  {plot_path}
""")
