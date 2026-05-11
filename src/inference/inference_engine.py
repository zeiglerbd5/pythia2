"""
Inference Engine for Real-time Pre-Spike Detection

Loads trained models and runs predictions on feature sequences.
Supports hot-swapping models without restarting the system.
"""

import torch
import time
from typing import Optional, Dict
from pathlib import Path
from loguru import logger

# Import model architectures
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.cnn_lstm import SpikeCNNLSTM
from src.models.tcn import TemporalConvNet


class InferenceEngine:
    """
    Real-time model inference engine.

    Features:
    - Load CNN-LSTM or TCN models
    - Hot-swap models without restart
    - Batch and single inference
    - Probability and binary predictions
    - Performance monitoring
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = 'cnn_lstm',
        n_features: int = 24,
        sequence_length: int = 60,
        device: str = 'mps',
        threshold: float = 0.5
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to model checkpoint (.pt file)
            model_type: 'cnn_lstm' or 'tcn'
            n_features: Number of features per timestep (24)
            sequence_length: Number of timesteps (60)
            device: PyTorch device ('mps', 'cuda', or 'cpu')
            threshold: Prediction threshold for binary classification
        """
        self.model_path = model_path
        self.model_type = model_type
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.device = device
        self.threshold = threshold

        # Model state
        self.model = None
        self.model_loaded_at = None

        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0.0

        # Load initial model
        self.load_model()

        logger.info(
            f"InferenceEngine initialized: {model_type}, "
            f"threshold={threshold}, device={device}"
        )

    def load_model(self, model_path: Optional[str] = None):
        """
        Load model from checkpoint.

        Args:
            model_path: Optional new model path (for hot-swapping)
        """
        if model_path:
            self.model_path = model_path

        if not Path(self.model_path).exists():
            logger.error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            # Create model architecture
            if self.model_type == 'cnn_lstm':
                model = SpikeCNNLSTM(
                    n_features=self.n_features,
                    sequence_length=self.sequence_length
                )
            elif self.model_type == 'tcn':
                model = TemporalConvNet(
                    n_features=self.n_features,
                    num_channels=[64, 64, 64, 64],  # 4 blocks of 64 channels
                    kernel_size=3,
                    dropout=0.5
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Checkpoint is just the state dict
                    model.load_state_dict(checkpoint)
            else:
                # Checkpoint is the full model
                model = checkpoint

            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()

            self.model = model
            self.model_loaded_at = time.time()

            logger.info(f"Model loaded successfully from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

    def reload_model(self, model_path: str):
        """
        Hot-swap model without restarting inference.

        Args:
            model_path: Path to new model checkpoint
        """
        logger.info(f"Hot-swapping model: {model_path}")

        old_path = self.model_path
        try:
            self.load_model(model_path)
            logger.info(f"Model hot-swapped successfully: {old_path} → {model_path}")

        except Exception as e:
            logger.error(f"Hot-swap failed, keeping old model: {e}")
            raise

    def predict(self, sequence_tensor: torch.Tensor) -> Dict:
        """
        Run inference on a single sequence.

        Args:
            sequence_tensor: Tensor of shape (1, sequence_length, n_features)

        Returns:
            Dict with:
                - probability: float (0-1)
                - prediction: int (0 or 1)
                - confidence: float (distance from threshold)
                - inference_time_ms: float
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        with torch.no_grad():
            # Forward pass
            output = self.model(sequence_tensor)

            # Get probability (sigmoid already applied in model)
            if output.dim() > 1:
                probability = output.item()
            else:
                probability = output.item()

            # Binary prediction
            prediction = 1 if probability >= self.threshold else 0

            # Confidence (distance from threshold)
            confidence = abs(probability - self.threshold)

        inference_time = (time.time() - start_time) * 1000  # milliseconds

        # Update statistics
        self.inference_count += 1
        self.total_inference_time += inference_time

        return {
            'probability': probability,
            'prediction': prediction,
            'confidence': confidence,
            'inference_time_ms': inference_time,
        }

    def predict_batch(self, sequence_tensors: list[torch.Tensor]) -> list[Dict]:
        """
        Run inference on multiple sequences (batch processing).

        Args:
            sequence_tensors: List of tensors, each (1, sequence_length, n_features)

        Returns:
            List of prediction dicts
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if not sequence_tensors:
            return []

        # Concatenate into batch
        batch_tensor = torch.cat(sequence_tensors, dim=0)  # (batch_size, seq_len, n_features)

        start_time = time.time()

        with torch.no_grad():
            # Forward pass
            outputs = self.model(batch_tensor)

            # Process each output
            results = []
            for i, output in enumerate(outputs):
                probability = output.item()
                prediction = 1 if probability >= self.threshold else 0
                confidence = abs(probability - self.threshold)

                results.append({
                    'probability': probability,
                    'prediction': prediction,
                    'confidence': confidence,
                })

        inference_time = (time.time() - start_time) * 1000  # milliseconds
        per_sample_time = inference_time / len(sequence_tensors)

        # Update statistics
        self.inference_count += len(sequence_tensors)
        self.total_inference_time += inference_time

        # Add timing to each result
        for result in results:
            result['inference_time_ms'] = per_sample_time

        return results

    def set_threshold(self, threshold: float):
        """
        Update prediction threshold.

        Args:
            threshold: New threshold value (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        old_threshold = self.threshold
        self.threshold = threshold

        logger.info(f"Prediction threshold updated: {old_threshold:.3f} → {threshold:.3f}")

    def get_statistics(self) -> Dict:
        """Get inference statistics."""
        avg_inference_time = (
            self.total_inference_time / self.inference_count
            if self.inference_count > 0
            else 0.0
        )

        return {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'model_loaded': self.model is not None,
            'model_loaded_at': self.model_loaded_at,
            'inference_count': self.inference_count,
            'total_inference_time_ms': self.total_inference_time,
            'avg_inference_time_ms': avg_inference_time,
            'threshold': self.threshold,
            'device': self.device,
        }

    @property
    def is_ready(self) -> bool:
        """Check if engine is ready for inference."""
        return self.model is not None
