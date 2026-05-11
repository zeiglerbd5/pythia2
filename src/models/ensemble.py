"""
Ensemble Model Manager with Sharpe-Weighted Voting

Implements ensemble methods per implementation guide:
- 3-5 models per ensemble for robust predictions
- Sharpe ratio weighting: Higher Sharpe = higher voting weight
- Agreement threshold: 75-90% consensus required for signal
- Confidence scoring based on vote strength and model performance
- Individual model performance tracking

Per guide: Ensemble outperforms single models, achieving 90%+ precision
on spike detection when combined with strict agreement thresholds.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
from loguru import logger

from .cnn_lstm import SpikeCNNLSTM, AlternativeGRU


class EnsembleMember:
    """
    Single member of an ensemble with performance tracking.

    Tracks:
    - Model instance
    - Performance metrics (accuracy, precision, Sharpe ratio)
    - Voting weight based on Sharpe ratio
    """

    def __init__(
        self,
        model: nn.Module,
        model_id: str,
        device: torch.device,
        sharpe_ratio: float = 1.0,
        accuracy: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        f1: float = 0.0
    ):
        """
        Initialize ensemble member.

        Args:
            model: PyTorch model instance
            model_id: Unique identifier for this model
            device: Computation device
            sharpe_ratio: Sharpe ratio on validation set (for weighting)
            accuracy: Validation accuracy
            precision: Validation precision
            recall: Validation recall
            f1: Validation F1 score
        """
        self.model = model.to(device)
        self.model_id = model_id
        self.device = device
        self.sharpe_ratio = sharpe_ratio
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

        # Voting weight (based on Sharpe ratio)
        self.weight = max(0.0, sharpe_ratio)  # Clip to non-negative

        logger.info(
            f"EnsembleMember {model_id} initialized",
            extra={
                "sharpe": f"{sharpe_ratio:.3f}",
                "accuracy": f"{accuracy:.3f}",
                "precision": f"{precision:.3f}",
                "weight": f"{self.weight:.3f}"
            }
        )

    def predict(self, sequences: torch.Tensor) -> np.ndarray:
        """
        Generate predictions.

        Args:
            sequences: Input sequences (batch, seq_len, features)

        Returns:
            Predictions array (batch,) with values in [0, 1]
        """
        self.model.eval()

        with torch.no_grad():
            sequences = sequences.to(self.device)
            outputs = self.model(sequences)
            predictions = outputs.cpu().numpy().flatten()

        return predictions

    def to_dict(self) -> dict:
        """Convert member info to dictionary."""
        return {
            "model_id": self.model_id,
            "sharpe_ratio": self.sharpe_ratio,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "weight": self.weight
        }


class EnsembleManager:
    """
    Manage ensemble of models with Sharpe-weighted voting.

    Per implementation guide:
    - Combines 3-5 models trained on different data splits or hyperparameters
    - Weights votes by Sharpe ratio (risk-adjusted returns on validation)
    - Requires 75-90% weighted agreement for positive signal
    - Provides confidence scores based on vote strength

    Achieves 90%+ precision through conservative voting thresholds.
    """

    def __init__(
        self,
        device: str = 'mps',
        agreement_threshold: float = 0.80,  # 80% weighted agreement
        min_confidence: float = 0.70  # Minimum confidence for signals
    ):
        """
        Initialize ensemble manager.

        Args:
            device: Computation device ('mps', 'cuda', 'cpu')
            agreement_threshold: Required weighted agreement (0.75-0.90 per guide)
            min_confidence: Minimum confidence to emit signal
        """
        self.device = self._setup_device(device)
        self.agreement_threshold = agreement_threshold
        self.min_confidence = min_confidence

        self.members: List[EnsembleMember] = []
        self.total_weight = 0.0

        if agreement_threshold < 0.75 or agreement_threshold > 0.90:
            logger.warning(
                f"agreement_threshold={agreement_threshold:.2f} outside recommended range [0.75, 0.90]. "
                "Per guide: 75-90% agreement achieves best precision."
            )

        logger.info(
            f"EnsembleManager initialized",
            extra={
                "device": str(self.device),
                "agreement_threshold": f"{agreement_threshold*100:.0f}%",
                "min_confidence": f"{min_confidence*100:.0f}%"
            }
        )

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        elif device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def add_member(
        self,
        model: nn.Module,
        model_id: str,
        sharpe_ratio: float = 1.0,
        accuracy: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        f1: float = 0.0
    ):
        """
        Add model to ensemble.

        Args:
            model: PyTorch model instance
            model_id: Unique identifier
            sharpe_ratio: Sharpe ratio for weighting
            accuracy: Validation accuracy
            precision: Validation precision
            recall: Validation recall
            f1: Validation F1 score
        """
        member = EnsembleMember(
            model=model,
            model_id=model_id,
            device=self.device,
            sharpe_ratio=sharpe_ratio,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1
        )

        self.members.append(member)
        self.total_weight = sum(m.weight for m in self.members)

        logger.info(
            f"Added ensemble member {model_id}",
            extra={
                "total_members": len(self.members),
                "total_weight": f"{self.total_weight:.3f}"
            }
        )

    def predict(
        self,
        sequences: torch.Tensor,
        return_confidence: bool = True,
        return_votes: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate ensemble predictions with confidence scores.

        Args:
            sequences: Input sequences (batch, seq_len, features)
            return_confidence: Return confidence scores
            return_votes: Return individual model votes

        Returns:
            Tuple of (predictions, confidences, votes)
            - predictions: Binary predictions (batch,)
            - confidences: Confidence scores (batch,) if return_confidence
            - votes: Individual votes (batch, n_models) if return_votes
        """
        if len(self.members) == 0:
            raise ValueError("No ensemble members added")

        batch_size = sequences.shape[0]

        # Collect predictions from all members
        all_predictions = []
        all_weights = []

        for member in self.members:
            preds = member.predict(sequences)
            all_predictions.append(preds)
            all_weights.append(member.weight)

        # Stack predictions (batch, n_models)
        all_predictions = np.stack(all_predictions, axis=1)
        all_weights = np.array(all_weights)

        # Normalize weights
        if self.total_weight > 0:
            normalized_weights = all_weights / self.total_weight
        else:
            normalized_weights = np.ones(len(all_weights)) / len(all_weights)

        # Weighted voting
        # For each sample: weighted_vote = sum(prediction_i * weight_i)
        weighted_votes = np.sum(all_predictions * normalized_weights[None, :], axis=1)

        # Binary predictions based on agreement threshold
        ensemble_predictions = (weighted_votes >= self.agreement_threshold).astype(float)

        # Confidence scores
        if return_confidence:
            # Confidence = distance from threshold
            # High confidence when far from threshold (either very high or very low)
            confidence_scores = np.abs(weighted_votes - 0.5) * 2  # Scale to [0, 1]
            confidence_scores = np.clip(confidence_scores, 0, 1)
        else:
            confidence_scores = None

        # Individual votes
        if return_votes:
            votes = all_predictions
        else:
            votes = None

        return ensemble_predictions, confidence_scores, votes

    def predict_with_metadata(
        self,
        sequences: torch.Tensor
    ) -> List[Dict]:
        """
        Generate predictions with full metadata.

        Args:
            sequences: Input sequences

        Returns:
            List of dictionaries with prediction details for each sample
        """
        predictions, confidences, votes = self.predict(
            sequences,
            return_confidence=True,
            return_votes=True
        )

        results = []

        for i in range(len(predictions)):
            sample_votes = votes[i]  # Individual model votes

            # Calculate agreement statistics
            vote_mean = sample_votes.mean()
            vote_std = sample_votes.std()

            # Count models voting positive
            n_positive = (sample_votes > 0.5).sum()
            n_total = len(sample_votes)

            result = {
                "prediction": int(predictions[i]),
                "confidence": float(confidences[i]),
                "weighted_vote": float(np.sum(sample_votes * (np.array([m.weight for m in self.members]) / self.total_weight))),
                "vote_mean": float(vote_mean),
                "vote_std": float(vote_std),
                "models_positive": int(n_positive),
                "models_total": int(n_total),
                "agreement_ratio": float(n_positive / n_total),
                "individual_votes": sample_votes.tolist(),
                "model_ids": [m.model_id for m in self.members]
            }

            results.append(result)

        return results

    def filter_by_confidence(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray
    ) -> np.ndarray:
        """
        Filter predictions by minimum confidence.

        Args:
            predictions: Binary predictions
            confidences: Confidence scores

        Returns:
            Filtered predictions (low confidence → 0)
        """
        filtered = predictions.copy()
        filtered[confidences < self.min_confidence] = 0

        n_filtered = (predictions != filtered).sum()

        if n_filtered > 0:
            logger.info(f"Filtered {n_filtered} low-confidence predictions")

        return filtered

    def get_ensemble_statistics(self) -> dict:
        """
        Get ensemble statistics.

        Returns:
            Dictionary with ensemble info
        """
        if len(self.members) == 0:
            return {
                "n_members": 0,
                "total_weight": 0.0
            }

        # Collect metrics from all members
        sharpe_ratios = [m.sharpe_ratio for m in self.members]
        accuracies = [m.accuracy for m in self.members]
        precisions = [m.precision for m in self.members]
        weights = [m.weight for m in self.members]

        return {
            "n_members": len(self.members),
            "total_weight": self.total_weight,
            "avg_sharpe": np.mean(sharpe_ratios),
            "max_sharpe": np.max(sharpe_ratios),
            "min_sharpe": np.min(sharpe_ratios),
            "avg_accuracy": np.mean(accuracies),
            "avg_precision": np.mean(precisions),
            "weight_distribution": {
                "mean": np.mean(weights),
                "std": np.std(weights),
                "min": np.min(weights),
                "max": np.max(weights)
            },
            "members": [m.to_dict() for m in self.members]
        }

    def save_ensemble(self, save_dir: str):
        """
        Save ensemble to directory.

        Args:
            save_dir: Directory to save ensemble
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save ensemble metadata
        metadata = {
            "agreement_threshold": self.agreement_threshold,
            "min_confidence": self.min_confidence,
            "n_members": len(self.members),
            "members": [m.to_dict() for m in self.members],
            "created_at": datetime.now().isoformat()
        }

        metadata_path = save_path / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save individual models
        for member in self.members:
            model_path = save_path / f"{member.model_id}.pt"
            torch.save(member.model.state_dict(), model_path)

        logger.info(f"Ensemble saved to {save_path}")

    def load_ensemble(
        self,
        load_dir: str,
        model_class: nn.Module,
        model_kwargs: dict
    ):
        """
        Load ensemble from directory.

        Args:
            load_dir: Directory containing ensemble
            model_class: Model class (SpikeCNNLSTM or AlternativeGRU)
            model_kwargs: Kwargs for model initialization
        """
        load_path = Path(load_dir)

        if not load_path.exists():
            raise FileNotFoundError(f"Ensemble directory not found: {load_path}")

        # Load metadata
        metadata_path = load_path / "ensemble_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.agreement_threshold = metadata['agreement_threshold']
        self.min_confidence = metadata['min_confidence']

        # Load individual models
        for member_info in metadata['members']:
            # Initialize model
            model = model_class(**model_kwargs)

            # Load weights
            model_path = load_path / f"{member_info['model_id']}.pt"
            model.load_state_dict(torch.load(model_path, map_location=self.device))

            # Add to ensemble
            self.add_member(
                model=model,
                model_id=member_info['model_id'],
                sharpe_ratio=member_info['sharpe_ratio'],
                accuracy=member_info['accuracy'],
                precision=member_info['precision'],
                recall=member_info['recall'],
                f1=member_info['f1']
            )

        logger.info(f"Ensemble loaded from {load_path}")
        logger.info(f"  Members: {len(self.members)}")
        logger.info(f"  Total weight: {self.total_weight:.3f}")


if __name__ == "__main__":
    # Test ensemble
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from .cnn_lstm import SpikeCNNLSTM

    print("=== Ensemble Manager Test ===\n")

    # Create synthetic test data
    n_samples = 100
    sequence_length = 60
    n_features = 35

    sequences = torch.randn(n_samples, sequence_length, n_features)

    print(f"Test data: {sequences.shape}\n")

    # Initialize ensemble
    print("=== Creating Ensemble ===\n")

    ensemble = EnsembleManager(
        device='cpu',
        agreement_threshold=0.80,
        min_confidence=0.70
    )

    # Add 5 models with different Sharpe ratios
    print("Adding ensemble members:\n")

    sharpe_ratios = [1.5, 1.3, 1.2, 1.0, 0.8]
    accuracies = [0.85, 0.83, 0.82, 0.80, 0.78]
    precisions = [0.92, 0.90, 0.88, 0.85, 0.82]

    for i, (sharpe, acc, prec) in enumerate(zip(sharpe_ratios, accuracies, precisions)):
        model = SpikeCNNLSTM(n_features=n_features, sequence_length=sequence_length)

        ensemble.add_member(
            model=model,
            model_id=f"model_{i+1}",
            sharpe_ratio=sharpe,
            accuracy=acc,
            precision=prec,
            recall=0.75,
            f1=0.80
        )

        print(f"Model {i+1}: Sharpe={sharpe:.2f}, Acc={acc:.2f}, Prec={prec:.2f}")

    print()

    # Get ensemble statistics
    print("=== Ensemble Statistics ===\n")

    stats = ensemble.get_ensemble_statistics()
    print(f"Members: {stats['n_members']}")
    print(f"Total weight: {stats['total_weight']:.3f}")
    print(f"Avg Sharpe: {stats['avg_sharpe']:.3f}")
    print(f"Avg Accuracy: {stats['avg_accuracy']:.3f}")
    print(f"Avg Precision: {stats['avg_precision']:.3f}")

    # Test predictions
    print("\n=== Testing Predictions ===\n")

    predictions, confidences, votes = ensemble.predict(
        sequences[:10],
        return_confidence=True,
        return_votes=True
    )

    print(f"Predictions shape: {predictions.shape}")
    print(f"Confidences shape: {confidences.shape}")
    print(f"Votes shape: {votes.shape}")

    print(f"\nSample predictions:")
    for i in range(5):
        print(f"  Sample {i}: pred={predictions[i]:.0f}, conf={confidences[i]:.3f}, votes={votes[i]}")

    # Test metadata predictions
    print("\n=== Predictions with Metadata ===\n")

    results = ensemble.predict_with_metadata(sequences[:3])

    for i, result in enumerate(results):
        print(f"Sample {i}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Weighted vote: {result['weighted_vote']:.3f}")
        print(f"  Agreement ratio: {result['agreement_ratio']:.1%}")
        print(f"  Models positive: {result['models_positive']}/{result['models_total']}")
        print()

    # Test confidence filtering
    print("=== Testing Confidence Filtering ===\n")

    filtered_predictions = ensemble.filter_by_confidence(predictions, confidences)

    n_filtered = (predictions != filtered_predictions).sum()
    print(f"Filtered predictions: {n_filtered}/{len(predictions)}")
    print(f"Original positive: {predictions.sum():.0f}")
    print(f"Filtered positive: {filtered_predictions.sum():.0f}")

    # Test save/load
    print("\n=== Testing Save/Load ===\n")

    save_dir = './test_ensemble'
    ensemble.save_ensemble(save_dir)
    print(f"Ensemble saved to {save_dir}")

    # Load ensemble
    ensemble2 = EnsembleManager(device='cpu')
    ensemble2.load_ensemble(
        load_dir=save_dir,
        model_class=SpikeCNNLSTM,
        model_kwargs={'n_features': n_features, 'sequence_length': sequence_length}
    )

    print(f"Ensemble loaded:")
    print(f"  Members: {len(ensemble2.members)}")
    print(f"  Total weight: {ensemble2.total_weight:.3f}")

    # Verify predictions match
    predictions2, _, _ = ensemble2.predict(sequences[:10], return_votes=True)
    predictions_match = np.allclose(predictions, predictions2)
    print(f"  Predictions match: {predictions_match}")

    # Clean up
    import shutil
    shutil.rmtree(save_dir)

    print("\n✓ Ensemble module test complete!")
