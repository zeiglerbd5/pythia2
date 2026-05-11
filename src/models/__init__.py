"""
Pythia ML Pipeline Module (Phase 3)

Complete machine learning pipeline for cryptocurrency spike detection.

Per implementation guide:
- CNN-LSTM hybrid architecture (82.44% accuracy target)
- Focal Loss for imbalanced classification (alpha=0.25, gamma=2.0)
- SMOTE oversampling (1:5 to 1:10 ratio)
- Walk-forward validation (temporal splits)
- Early stopping and checkpointing
- Ensemble voting with Sharpe weighting (90%+ precision target)
- Comprehensive metrics tracking

Modules:
- cnn_lstm: CNN-LSTM and GRU architectures
- focal_loss: Focal Loss and Weighted BCE
- dataset: PyTorch datasets, data loaders, target generation
- smote: SMOTE oversampling for time-series
- validation: Walk-forward validation framework
- trainer: Training pipeline with early stopping
- ensemble: Ensemble manager with Sharpe-weighted voting
- metrics: Evaluation metrics and reporting
"""

# Models
from .cnn_lstm import (
    SpikeCNNLSTM,
    AlternativeGRU,
    create_model
)

# Loss functions
from .focal_loss import (
    FocalLoss,
    WeightedBCELoss,
    create_loss_function
)

# Dataset and data loading
from .dataset import (
    SpikeDataset,
    SpikeTargetGenerator,
    DatasetBuilder,
    create_dataloaders
)

# SMOTE oversampling
from .smote import (
    TimeSeriesSMOTE,
    SMOTEDatasetWrapper,
    apply_smote_to_dataset
)

# Validation
from .validation import (
    ValidationFold,
    WalkForwardValidator,
    TimeSeriesSplitter
)

# Training
from .trainer import (
    EarlyStopping,
    ModelTrainer
)

# Ensemble
from .ensemble import (
    EnsembleMember,
    EnsembleManager
)

# Metrics
from .metrics import (
    ClassificationMetrics,
    TradingMetrics,
    MetricsCalculator,
    ModelEvaluator
)

__all__ = [
    # Models
    'SpikeCNNLSTM',
    'AlternativeGRU',
    'create_model',

    # Loss
    'FocalLoss',
    'WeightedBCELoss',
    'create_loss_function',

    # Dataset
    'SpikeDataset',
    'SpikeTargetGenerator',
    'DatasetBuilder',
    'create_dataloaders',

    # SMOTE
    'TimeSeriesSMOTE',
    'SMOTEDatasetWrapper',
    'apply_smote_to_dataset',

    # Validation
    'ValidationFold',
    'WalkForwardValidator',
    'TimeSeriesSplitter',

    # Training
    'EarlyStopping',
    'ModelTrainer',

    # Ensemble
    'EnsembleMember',
    'EnsembleManager',

    # Metrics
    'ClassificationMetrics',
    'TradingMetrics',
    'MetricsCalculator',
    'ModelEvaluator',
]
