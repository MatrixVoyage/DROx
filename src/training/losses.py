"""Loss functions for deepfake detection: FocalLoss, SupervisedContrastiveLoss, CompositeDetectionLoss."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Numerically stable binary Focal Loss with alpha balancing and gamma focusing.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where:
        p = sigmoid(logits)
        p_t = p if y == 1 else 1 - p
        alpha_t = alpha if y == 1 else (1 - alpha)

    Args:
        alpha: Weighting factor for the positive (fake) class in (0, 1). Default: 0.25.
        gamma: Focusing parameter for modulating easy/hard samples (>= 0). Default: 2.0.
        reduction: "mean", "sum", or "none". Default: "mean".
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary focal loss.

        Args:
            logits: Predicted raw logits of shape (B, 1) or (B,).
            targets: Binary ground truth labels of shape (B, 1) or (B,), values in {0, 1}.

        Returns:
            Computed focal loss scalar or tensor.
        """
        if logits.shape != targets.shape:
            raise ValueError(
                f"Logits shape {tuple(logits.shape)} and targets shape {tuple(targets.shape)} must match."
            )

        targets = targets.float()
        # Numerically stable unreduced BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t calculation
        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)

        # Alpha factor
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)

        focal_loss = alpha_t * modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss (SupCon) for video representations.

    Pulls representations of the same class (intra-class) together in latent space
    while pushing representations of different classes (inter-class) apart.

    Args:
        temperature: Scaling temperature for cosine similarities (> 0). Default: 0.1.
        contrast_mode: "all" or "one". Default: "all".
    """

    def __init__(self, temperature: float = 0.1, contrast_mode: str = "all"):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)
        self.contrast_mode = contrast_mode

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Supervised Contrastive Loss.

        Args:
            features: Tensor of embeddings of shape (B, D) or (B, n_views, D).
            labels: 1D or 2D tensor of class labels of shape (B,) or (B, 1).

        Returns:
            Scalar contrastive loss.
        """
        device = features.device
        if labels.dim() == 2:
            labels = labels.view(-1)

        # If batch contains only one class, there are no negatives to contrast against
        if len(torch.unique(labels)) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if features.dim() == 2:
            # Shape: (B, 1, D)
            features = features.unsqueeze(1)
        elif features.dim() != 3:
            raise ValueError(f"Expected features of dim 2 or 3, got dim {features.dim()}")

        batch_size = features.shape[0]
        n_views = features.shape[1]

        if labels.dim() == 2:
            labels = labels.view(-1)
        if labels.shape[0] != batch_size:
            raise ValueError(f"Label batch size {labels.shape[0]} does not match features {batch_size}")

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Flatten views: (B * n_views, D)
        contrast_count = n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # Normalize features
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

        # Tile mask for multiple views
        mask = mask.repeat(contrast_count, contrast_count)
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        # Compute dot-product similarity
        anchor_dot_contrast = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature

        # For numerical stability, subtract max
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Compute log-probabilities over all pairs (excluding self)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive pairs
        pos_per_anchor = mask.sum(1)
        # Avoid division by zero for anchors with no other positives in batch
        has_positives = pos_per_anchor > 0
        if not has_positives.any():
            # No positive pairs in the batch
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[has_positives] / pos_per_anchor[has_positives]

        loss = -mean_log_prob_pos.mean()
        return loss


class CompositeDetectionLoss(nn.Module):
    """Composite loss combining classification (Focal/BCE) and embedding contrastive loss.

    Total Loss = L_cls + lambda_contrastive * L_supcon
    """

    def __init__(
        self,
        cls_loss_type: str = "focal",
        alpha: float = 0.25,
        gamma: float = 2.0,
        contrastive_weight: float = 0.1,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.contrastive_weight = float(contrastive_weight)
        if cls_loss_type == "focal":
            self.cls_loss: nn.Module = FocalLoss(alpha=alpha, gamma=gamma)
        elif cls_loss_type == "bce":
            self.cls_loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown cls_loss_type: '{cls_loss_type}'")

        if self.contrastive_weight > 0.0:
            self.contrastive_loss: Optional[nn.Module] = SupervisedContrastiveLoss(temperature=temperature)
        else:
            self.contrastive_loss = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute composite detection loss.

        Args:
            logits: Classifier logits (B, 1).
            targets: Target labels (B, 1).
            features: Optional latent representations (B, D) for contrastive loss.

        Returns:
            Tuple of (total_loss_tensor, loss_dict_with_floats).
        """
        cls_l = self.cls_loss(logits, targets)
        loss_dict: Dict[str, float] = {"cls_loss": float(cls_l.detach().item())}

        if self.contrastive_loss is not None and features is not None and self.contrastive_weight > 0.0:
            # Verify batch has both classes to form contrastive pairs
            unique_labels = torch.unique(targets)
            if len(unique_labels) >= 2:
                con_l = self.contrastive_loss(features, targets)
                total_loss = cls_l + self.contrastive_weight * con_l
                loss_dict["con_loss"] = float(con_l.detach().item())
            else:
                total_loss = cls_l
                loss_dict["con_loss"] = 0.0
        else:
            total_loss = cls_l
            loss_dict["con_loss"] = 0.0

        loss_dict["total_loss"] = float(total_loss.detach().item())
        return total_loss, loss_dict
