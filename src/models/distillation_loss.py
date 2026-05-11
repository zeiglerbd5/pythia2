"""
Knowledge Distillation Loss for XGBoost-Guided TCN Training

Implements combined loss for transferring knowledge from XGBoost to TCN:
    L_total = alpha * L_hard + (1 - alpha) * L_soft

Where:
- L_hard: FocalLoss vs true spike labels (0/1)
- L_soft: MSE vs XGBoost probability predictions (soft labels)

The soft label component allows the TCN to learn XGBoost's "intuition"
about which samples are more confidently spikes vs edge cases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from loguru import logger

from .focal_loss import FocalLoss


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining hard and soft labels.

    Uses a weighted combination of:
    1. Hard label loss (FocalLoss vs true spike labels)
    2. Soft label loss (MSE vs XGBoost predictions)

    This allows the TCN to:
    - Learn the actual spike prediction task (hard labels)
    - Inherit XGBoost's learned feature importance (soft labels)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        hard_loss_fn: Optional[nn.Module] = None,
        focal_alpha: float = 0.05,
        focal_gamma: float = 4.0,
        temperature: float = 1.0,
        soft_loss_type: str = 'mse'
    ):
        """
        Initialize DistillationLoss.

        Args:
            alpha: Weight for hard label loss (0-1)
                   - alpha=0.5: Equal weight to hard and soft labels
                   - alpha=0.7: More emphasis on true labels
                   - alpha=0.3: More emphasis on XGBoost guidance
            hard_loss_fn: Loss function for hard labels (default: FocalLoss)
            focal_alpha: FocalLoss alpha parameter
            focal_gamma: FocalLoss gamma parameter
            temperature: Temperature for softening predictions
            soft_loss_type: Type of soft label loss ('mse' or 'kl')
        """
        super(DistillationLoss, self).__init__()

        self.alpha = alpha
        self.temperature = temperature
        self.soft_loss_type = soft_loss_type

        # Hard label loss (vs true spike labels)
        if hard_loss_fn is not None:
            self.hard_loss = hard_loss_fn
        else:
            self.hard_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Soft label loss (vs XGBoost predictions)
        if soft_loss_type == 'mse':
            self.soft_loss = nn.MSELoss()
        elif soft_loss_type == 'kl':
            self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError(f"Unknown soft_loss_type: {soft_loss_type}")

        logger.info(
            f"DistillationLoss initialized: alpha={alpha}, "
            f"temp={temperature}, soft_loss={soft_loss_type}"
        )

    def forward(
        self,
        predictions: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute distillation loss.

        Args:
            predictions: TCN model predictions (batch, 1) or (batch,)
            hard_labels: True spike labels 0/1 (batch, 1) or (batch,)
            soft_labels: XGBoost probability predictions (batch, 1) or (batch,)

        Returns:
            Tuple of (total_loss, hard_loss, soft_loss)
        """
        # Ensure correct shapes
        predictions = predictions.view(-1)
        hard_labels = hard_labels.view(-1)
        soft_labels = soft_labels.view(-1)

        # Compute hard label loss (vs true labels)
        L_hard = self.hard_loss(predictions, hard_labels)

        # Compute soft label loss (vs XGBoost predictions)
        if self.soft_loss_type == 'kl':
            # For KL divergence, apply temperature scaling
            pred_log = F.log_softmax(
                torch.stack([1 - predictions, predictions], dim=1) / self.temperature,
                dim=1
            )
            soft_target = F.softmax(
                torch.stack([1 - soft_labels, soft_labels], dim=1) / self.temperature,
                dim=1
            )
            L_soft = self.soft_loss(pred_log, soft_target) * (self.temperature ** 2)
        else:
            # MSE loss: simple distance between predictions
            L_soft = self.soft_loss(predictions, soft_labels)

        # Combined loss
        L_total = self.alpha * L_hard + (1 - self.alpha) * L_soft

        return L_total, L_hard, L_soft


class AdaptiveDistillationLoss(nn.Module):
    """
    Adaptive distillation loss that adjusts alpha during training.

    Strategy: Start with high soft label influence (learn from XGBoost),
    then gradually shift to hard labels (learn true task).

    This prevents the model from being overly constrained by XGBoost
    in later training stages.
    """

    def __init__(
        self,
        initial_alpha: float = 0.3,
        final_alpha: float = 0.7,
        warmup_epochs: int = 10,
        focal_alpha: float = 0.05,
        focal_gamma: float = 4.0
    ):
        """
        Initialize AdaptiveDistillationLoss.

        Args:
            initial_alpha: Starting alpha (low = more XGBoost influence)
            final_alpha: Ending alpha (high = more true label influence)
            warmup_epochs: Epochs to transition from initial to final
            focal_alpha: FocalLoss alpha parameter
            focal_gamma: FocalLoss gamma parameter
        """
        super(AdaptiveDistillationLoss, self).__init__()

        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        self.hard_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.soft_loss = nn.MSELoss()

        logger.info(
            f"AdaptiveDistillationLoss initialized: "
            f"alpha {initial_alpha} → {final_alpha} over {warmup_epochs} epochs"
        )

    def set_epoch(self, epoch: int):
        """Update current epoch for alpha calculation."""
        self.current_epoch = epoch

    def get_alpha(self) -> float:
        """Calculate current alpha based on epoch."""
        if self.current_epoch >= self.warmup_epochs:
            return self.final_alpha

        # Linear interpolation
        progress = self.current_epoch / self.warmup_epochs
        return self.initial_alpha + progress * (self.final_alpha - self.initial_alpha)

    def forward(
        self,
        predictions: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute adaptive distillation loss."""
        predictions = predictions.view(-1)
        hard_labels = hard_labels.view(-1)
        soft_labels = soft_labels.view(-1)

        L_hard = self.hard_loss(predictions, hard_labels)
        L_soft = self.soft_loss(predictions, soft_labels)

        alpha = self.get_alpha()
        L_total = alpha * L_hard + (1 - alpha) * L_soft

        return L_total, L_hard, L_soft


class FocalDistillationLoss(nn.Module):
    """
    Focal-weighted distillation loss.

    Applies focal weighting to both hard and soft losses,
    focusing on samples where model disagrees with both
    true labels AND XGBoost predictions.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        focal_alpha: float = 0.05
    ):
        """
        Initialize FocalDistillationLoss.

        Args:
            alpha: Weight for hard label loss
            gamma: Focal gamma for down-weighting easy examples
            focal_alpha: Class balance alpha for FocalLoss
        """
        super(FocalDistillationLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.hard_loss = FocalLoss(alpha=focal_alpha, gamma=gamma)
        self.soft_loss = nn.MSELoss(reduction='none')

        logger.info(
            f"FocalDistillationLoss initialized: "
            f"alpha={alpha}, gamma={gamma}"
        )

    def forward(
        self,
        predictions: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute focal-weighted distillation loss."""
        predictions = predictions.view(-1)
        hard_labels = hard_labels.view(-1)
        soft_labels = soft_labels.view(-1)

        # Hard loss (already focal-weighted)
        L_hard = self.hard_loss(predictions, hard_labels)

        # Soft loss with focal weighting
        # Down-weight samples where prediction matches soft label
        soft_per_sample = self.soft_loss(predictions, soft_labels)

        # Focal weight: (1 - agreement)^gamma
        # agreement = 1 - |pred - soft| (how close we are)
        agreement = 1 - torch.abs(predictions - soft_labels)
        focal_weight = (1 - agreement) ** self.gamma

        L_soft = (focal_weight * soft_per_sample).mean()

        L_total = self.alpha * L_hard + (1 - self.alpha) * L_soft

        return L_total, L_hard, L_soft


def create_distillation_loss(
    loss_type: str = 'standard',
    alpha: float = 0.5,
    **kwargs
) -> nn.Module:
    """
    Factory function to create distillation loss.

    Args:
        loss_type: 'standard', 'adaptive', or 'focal'
        alpha: Weight for hard label loss
        **kwargs: Additional arguments for specific loss types

    Returns:
        Distillation loss module
    """
    if loss_type == 'standard':
        return DistillationLoss(alpha=alpha, **kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveDistillationLoss(initial_alpha=alpha, **kwargs)
    elif loss_type == 'focal':
        return FocalDistillationLoss(alpha=alpha, **kwargs)
    else:
        raise ValueError(f"Unknown distillation loss type: {loss_type}")


if __name__ == "__main__":
    # Test distillation losses
    print("=== Distillation Loss Test ===\n")

    # Create synthetic data
    batch_size = 100
    torch.manual_seed(42)

    # Predictions (model output)
    predictions = torch.sigmoid(torch.randn(batch_size))

    # Hard labels (true spike labels)
    hard_labels = torch.zeros(batch_size)
    hard_labels[torch.randperm(batch_size)[:5]] = 1.0  # 5% positive

    # Soft labels (XGBoost predictions)
    # Simulate XGBoost being more confident on some samples
    soft_labels = torch.sigmoid(torch.randn(batch_size) * 0.5)
    # Make soft labels correlate with hard labels somewhat
    soft_labels[hard_labels == 1] = soft_labels[hard_labels == 1] * 0.5 + 0.5

    print(f"Batch size: {batch_size}")
    print(f"Positive ratio: {hard_labels.mean():.2%}")
    print(f"Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"Soft labels range: [{soft_labels.min():.3f}, {soft_labels.max():.3f}]")
    print()

    # Test standard distillation
    print("=== Standard Distillation Loss ===")
    for alpha in [0.3, 0.5, 0.7]:
        loss = DistillationLoss(alpha=alpha)
        total, hard, soft = loss(predictions, hard_labels, soft_labels)
        print(f"  α={alpha}: total={total:.4f}, hard={hard:.4f}, soft={soft:.4f}")

    print()

    # Test adaptive distillation
    print("=== Adaptive Distillation Loss ===")
    adaptive_loss = AdaptiveDistillationLoss(
        initial_alpha=0.3,
        final_alpha=0.7,
        warmup_epochs=10
    )
    for epoch in [0, 5, 10, 15]:
        adaptive_loss.set_epoch(epoch)
        alpha = adaptive_loss.get_alpha()
        total, hard, soft = adaptive_loss(predictions, hard_labels, soft_labels)
        print(f"  Epoch {epoch:2d}: α={alpha:.2f}, total={total:.4f}")

    print()

    # Test focal distillation
    print("=== Focal Distillation Loss ===")
    focal_distill = FocalDistillationLoss(alpha=0.5, gamma=2.0)
    total, hard, soft = focal_distill(predictions, hard_labels, soft_labels)
    print(f"  total={total:.4f}, hard={hard:.4f}, soft={soft:.4f}")

    print()
    print("✓ Distillation loss tests passed!")
