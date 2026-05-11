#!/usr/bin/env python3
"""
High-Frequency Order Book Model Training

Trains a CNN-LSTM model on 10-second resolution order book features.

Usage:
    python scripts/train_orderbook_model_hf.py \
        --data data/orderbook_features_5pct_10min.npz \
        --epochs 100 \
        --batch-size 32 \
        --learning-rate 0.0001 \
        --output models/orderbook_hf_model
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, auc
from datetime import datetime
from pathlib import Path
import json
import pickle


class OrderBookDataset(Dataset):
    """PyTorch Dataset for order book sequences."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for high-frequency order book spike prediction.

    Architecture:
    1. 1D Conv layers to extract temporal patterns
    2. LSTM layers to capture sequence dependencies
    3. Fully connected layers for classification
    """

    def __init__(self, input_dim, sequence_length, hidden_dim=64, num_lstm_layers=2, dropout=0.3):
        super(CNNLSTMModel, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(dropout)

        # LSTM layers
        # After 2 pooling layers: sequence_length // 4
        self.lstm_input_dim = 128
        self.lstm_sequence_length = sequence_length // 4

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.bn_fc = nn.BatchNorm1d(32)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, sequence_length, features)

        # Transpose for Conv1d: (batch, features, sequence_length)
        x = x.transpose(1, 2)

        # CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_cnn(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_cnn(x)

        # Transpose back for LSTM: (batch, sequence_length, features)
        x = x.transpose(1, 2)

        # LSTM layers
        lstm_out, _ = self.lstm(x)

        # Take last output
        x = lstm_out[:, -1, :]

        # Fully connected layers
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.sigmoid(self.fc2(x))

        return x.squeeze()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Focuses training on hard examples (minority class).
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        all_preds.extend((outputs > 0.5).cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return avg_loss, f1


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return avg_loss, f1, all_probs, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train high-frequency order book model')
    parser.add_argument('--data', required=True, help='Path to extracted features (.npz)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--hidden-dim', type=int, default=64, help='LSTM hidden dimension (default: 64)')
    parser.add_argument('--num-lstm-layers', type=int, default=2, help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal loss alpha (default: 0.25)')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma (default: 2.0)')
    parser.add_argument('--output', required=True, help='Output directory for model')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.2, help='Validation set size from train (default: 0.2)')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()

    # Device setup
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    data = np.load(args.data, allow_pickle=True)
    X = data['X']
    y = data['y']
    metadata = data['metadata']
    feature_names = data['feature_names']

    print(f"Dataset shape: {X.shape}")
    print(f"Sequence length: {X.shape[1]}")
    print(f"Features: {X.shape[2]}")
    print(f"Positive samples: {y.sum()} ({y.mean()*100:.2f}%)")
    print()

    # Normalize features
    print("Normalizing features...")
    n_samples, n_timesteps, n_features = X.shape
    X_flat = X.reshape(-1, n_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    X = X_scaled.reshape(n_samples, n_timesteps, n_features)
    print(f"Features normalized (mean=0, std=1)")
    print()

    # Train/val/test split
    print("Splitting data...")
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=args.val_size, random_state=42, stratify=y_train_val
    )

    print(f"Train: {len(y_train)} samples ({y_train.sum()} positive, {y_train.mean()*100:.2f}%)")
    print(f"Val:   {len(y_val)} samples ({y_val.sum()} positive, {y_val.mean()*100:.2f}%)")
    print(f"Test:  {len(y_test)} samples ({y_test.sum()} positive, {y_test.mean()*100:.2f}%)")
    print()

    # Create datasets
    train_dataset = OrderBookDataset(X_train, y_train)
    val_dataset = OrderBookDataset(X_val, y_val)
    test_dataset = OrderBookDataset(X_test, y_test)

    # Weighted sampling for imbalanced data
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    print("=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    model = CNNLSTMModel(
        input_dim=n_features,
        sequence_length=n_timesteps,
        hidden_dim=args.hidden_dim,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout
    ).to(device)

    print(model)
    print()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()

    # Loss and optimizer
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    # Training loop
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_val_f1 = 0
    best_epoch = 0
    patience = 20
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_f1': [],
        'val_loss': [],
        'val_f1': []
    }

    for epoch in range(args.epochs):
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_probs, val_preds, val_labels = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        scheduler.step(val_f1)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0

            # Save checkpoint
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'config': {
                    'input_dim': n_features,
                    'sequence_length': n_timesteps,
                    'hidden_dim': args.hidden_dim,
                    'num_lstm_layers': args.num_lstm_layers,
                    'dropout': args.dropout
                }
            }, output_dir / 'best_model.pt')

            print(f"  → New best model saved (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered (patience={patience})")
            break

    print()
    print(f"Best validation F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print()

    # Load best model for final evaluation
    print("=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_f1, test_probs, test_preds, test_labels = validate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print()

    print("Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['No Spike', 'Spike'], zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    print()

    # Calculate precision-recall AUC
    precision, recall, _ = precision_recall_curve(test_labels, test_probs)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    print()

    # Save scaler and metadata
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names.tolist(), f, indent=2)

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'best_epoch': best_epoch,
            'best_val_f1': float(best_val_f1),
            'test_loss': float(test_loss),
            'test_f1': float(test_f1),
            'test_pr_auc': float(pr_auc),
            'confusion_matrix': cm.tolist(),
            'config': {
                'data_file': args.data,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'hidden_dim': args.hidden_dim,
                'num_lstm_layers': args.num_lstm_layers,
                'dropout': args.dropout,
                'focal_alpha': args.focal_alpha,
                'focal_gamma': args.focal_gamma
            }
        }, f, indent=2)

    print(f"Model saved to {output_dir}")
    print("\nDONE!")


if __name__ == '__main__':
    main()
