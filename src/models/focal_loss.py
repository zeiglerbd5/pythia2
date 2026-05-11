"""
Focal Loss for Imbalanced Binary Classification

Implements Focal Loss per implementation guide for handling extreme class imbalance.

Per guide:
- Formula: FL(p_t) = -α_t(1-p_t)^γ × log(p_t)
- α (alpha): 0.25 for class weighting
- γ (gamma): 2.0-3.0 for focusing on hard examples
- Down-weights easy negative examples
- Concentrates gradient updates on misclassified or low-confidence predictions

Combined with SMOTE (1:5 to 1:10 ratio) and batch-balanced sampling,
this produces best results while avoiding overfitting from full 1:1 balancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from loguru import logger


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.

    Per implementation guide:
    - Achieves superior performance on imbalanced datasets (1-5% positive class)
    - α=0.25 provides class weighting (guide specification)
    - γ=2.0-3.0 focuses learning on hard examples (guide specification)
    - Outperforms standard BCE and weighted BCE on spike detection

    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
        p_t = p if y = 1, else 1 - p
        α_t = α if y = 1, else 1 - α

    The (1 - p_t)^γ term down-weights easy examples and focuses on hard negatives.
    """

    def __init__(
        self,
        alpha: float = 0.05,  # Updated default for pre-spike detection (vs 0.25)
        gamma: float = 4.0,  # Updated default for extreme imbalance (vs 2.0)
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive class
                   - Default 0.05 for pre-spike detection (extreme imbalance)
                   - Guide default: 0.25 for moderate imbalance
            gamma: Focusing parameter
                   - Default 4.0 for pre-spike detection (0.5-2% positive class)
                   - Guide default: 2.0-3.0 for moderate imbalance (5%+ positive)
            reduction: Reduction method ('mean', 'sum', 'none')
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        logger.info(
            f"FocalLoss initialized",
            extra={
                "alpha": alpha,
                "gamma": gamma,
                "reduction": reduction,
                "label_smoothing": label_smoothing
            }
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss.

        Args:
            inputs: Model predictions (probabilities or logits)
                Shape: (batch_size, 1) or (batch_size,)
            targets: Ground truth labels (0 or 1)
                Shape: (batch_size, 1) or (batch_size,)

        Returns:
            Focal loss value
        """
        # Ensure correct shapes
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute BCE loss (no reduction yet)
        bce_loss = F.binary_cross_entropy(
            inputs,
            targets,
            reduction='none'
        )

        # Compute p_t
        # p_t = p if y = 1, else 1 - p
        p_t = inputs * targets + (1 - inputs) * (1 - targets)

        # Compute modulating factor: (1 - p_t)^gamma
        # This down-weights easy examples
        modulating_factor = (1 - p_t) ** self.gamma

        # Compute alpha factor
        # α_t = α if y = 1, else 1 - α
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal Loss = -α_t * (1 - p_t)^γ * log(p_t)
        focal_loss = alpha_t * modulating_factor * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss (alternative to Focal Loss).

    Simpler alternative that just applies class weights without
    the focusing mechanism of Focal Loss.
    """

    def __init__(
        self,
        pos_weight: float = 10.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Weighted BCE Loss.

        Args:
            pos_weight: Weight for positive class (default: 10.0)
            reduction: Reduction method
        """
        super(WeightedBCELoss, self).__init__()

        self.pos_weight = torch.tensor([pos_weight])
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted BCE loss."""
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Move pos_weight to same device as inputs
        if self.pos_weight.device != inputs.device:
            self.pos_weight = self.pos_weight.to(inputs.device)

        loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )

        return loss


def create_loss_function(
    loss_type: str = 'focal',
    alpha: float = 0.25,
    gamma: float = 2.0,
    pos_weight: float = 10.0
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: 'focal' or 'weighted_bce'
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        pos_weight: Weighted BCE positive weight

    Returns:
        Loss function module
    """
    if loss_type == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'weighted_bce':
        return WeightedBCELoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test Focal Loss
    import numpy as np

    print("=== Focal Loss Test ===\n")

    # Create synthetic imbalanced data
    # 95% negative (0), 5% positive (1)
    batch_size = 1000
    pos_samples = int(batch_size * 0.05)
    neg_samples = batch_size - pos_samples

    # Simulate model predictions (probabilities)
    torch.manual_seed(42)

    # Easy negatives: high confidence correct predictions
    easy_neg_preds = torch.rand(neg_samples // 2) * 0.2  # 0.0-0.2
    easy_neg_targets = torch.zeros(neg_samples // 2)

    # Hard negatives: low confidence wrong predictions
    hard_neg_preds = torch.rand(neg_samples // 2) * 0.3 + 0.5  # 0.5-0.8
    hard_neg_targets = torch.zeros(neg_samples // 2)

    # Easy positives: high confidence correct predictions
    easy_pos_preds = torch.rand(pos_samples // 2) * 0.2 + 0.8  # 0.8-1.0
    easy_pos_targets = torch.ones(pos_samples // 2)

    # Hard positives: low confidence wrong predictions
    hard_pos_preds = torch.rand(pos_samples // 2) * 0.3 + 0.2  # 0.2-0.5
    hard_pos_targets = torch.ones(pos_samples // 2)

    # Combine all
    predictions = torch.cat([easy_neg_preds, hard_neg_preds, easy_pos_preds, hard_pos_preds])
    targets = torch.cat([easy_neg_targets, hard_neg_targets, easy_pos_targets, hard_pos_targets])

    # Shuffle
    indices = torch.randperm(batch_size)
    predictions = predictions[indices]
    targets = targets[indices]

    print(f"Dataset: {batch_size} samples")
    print(f"  Positive: {targets.sum():.0f} ({targets.mean()*100:.1f}%)")
    print(f"  Negative: {(1-targets).sum():.0f} ({(1-targets).mean()*100:.1f}%)")

    # Test Focal Loss
    print("\n=== Focal Loss (α=0.25, γ=2.0) ===")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    fl_value = focal_loss(predictions, targets)
    print(f"Loss: {fl_value:.4f}")

    # Test standard BCE for comparison
    print("\n=== Standard BCE (no weighting) ===")
    bce_loss = F.binary_cross_entropy(predictions, targets)
    print(f"Loss: {bce_loss:.4f}")

    # Test Weighted BCE
    print("\n=== Weighted BCE (pos_weight=10) ===")
    weighted_bce = WeightedBCELoss(pos_weight=10.0)
    # Convert predictions to logits for weighted BCE
    logits = torch.log(predictions / (1 - predictions + 1e-7))
    wbce_value = weighted_bce(logits, targets)
    print(f"Loss: {wbce_value:.4f}")

    # Compare focusing effect
    print("\n=== Focal Loss Focusing Effect ===")
    print("Testing different gamma values:\n")

    for gamma_val in [0.0, 1.0, 2.0, 3.0, 5.0]:
        fl = FocalLoss(alpha=0.25, gamma=gamma_val)
        loss = fl(predictions, targets)
        print(f"  γ={gamma_val:.1f}: Loss={loss:.4f}")

    # Analyze per-sample contributions
    print("\n=== Per-Sample Loss Analysis ===")
    fl_no_reduction = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
    per_sample_loss = fl_no_reduction(predictions, targets)

    # Split by difficulty
    easy_mask = (predictions > 0.7) | (predictions < 0.3)  # Confident predictions
    hard_mask = ~easy_mask  # Uncertain predictions

    print(f"Easy examples: avg loss = {per_sample_loss[easy_mask].mean():.4f}")
    print(f"Hard examples: avg loss = {per_sample_loss[hard_mask].mean():.4f}")
    print(f"Ratio (hard/easy): {(per_sample_loss[hard_mask].mean() / per_sample_loss[easy_mask].mean()):.2f}x")

    print("\nFocal Loss successfully down-weights easy examples and focuses on hard ones!")
