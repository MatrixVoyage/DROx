"""Identity/Manipulation Disentanglement Modules for Milestone 13.2c."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubspaceProjectionHead(nn.Module):
    """Projection head mapping penultimate representations into a normalized latent subspace.

    Architecture:
        Linear(in_features -> hidden_dim)
        -> BatchNorm1d(hidden_dim)
        -> ReLU(inplace=True)
        -> Dropout(p=dropout)
        -> Linear(hidden_dim -> out_dim)
        -> L2 Normalization
    """

    def __init__(
        self,
        in_features: int = 1280,
        hidden_dim: int = 512,
        out_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Project input into subspace.

        Args:
            x: Input tensor of shape (B, in_features).
            normalize: Whether to apply L2 normalization to output.

        Returns:
            Projected tensor of shape (B, out_dim).
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input (B, D), got {x.dim()}D tensor with shape {tuple(x.shape)}")

        b, d = x.shape
        if d != self.in_features:
            raise ValueError(f"Expected input dimension {self.in_features}, got {d}")

        h = self.net[0](x)
        if b > 1 or not self.training:
            h = self.net[1](h)
        h = self.net[2](h)
        h = self.net[3](h)
        out = self.net[4](h)

        if normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


class SubspaceOrthogonalityLoss(nn.Module):
    """Loss function penalizing correlation and alignment between two subspaces.

    Loss comprises two terms:
        1. Pairwise Cosine Alignment: (1/B) * sum_i (z_id[i]^T * z_manip[i])^2
        2. Cross-Covariance Frobenius Penalty: (1/d^2) * ||Z_id^T * Z_manip||_F^2
    """

    def __init__(self, cov_weight: float = 1.0):
        super().__init__()
        self.cov_weight = cov_weight

    def forward(self, z_id: torch.Tensor, z_manip: torch.Tensor) -> torch.Tensor:
        """Compute orthogonality penalty.

        Args:
            z_id: L2-normalized identity tensor of shape (B, d).
            z_manip: L2-normalized manipulation tensor of shape (B, d).

        Returns:
            Scalar tensor representing orthogonality loss.
        """
        if z_id.shape != z_manip.shape:
            raise ValueError(f"Shape mismatch: z_id {tuple(z_id.shape)} vs z_manip {tuple(z_manip.shape)}")

        b, d = z_id.shape
        if b <= 1:
            return torch.tensor(0.0, device=z_id.device, requires_grad=True)

        # 1. Individual sample alignment: dot-product squared
        dot_products = (z_id * z_manip).sum(dim=-1)  # (B,)
        pairwise_loss = (dot_products ** 2).mean()

        # 2. Batch cross-covariance penalty
        # Center embeddings
        z_id_c = z_id - z_id.mean(dim=0, keepdim=True)
        z_manip_c = z_manip - z_manip.mean(dim=0, keepdim=True)

        # Cross-covariance matrix: (d, d)
        cov_matrix = torch.mm(z_id_c.t(), z_manip_c) / (b - 1)
        cov_loss = (cov_matrix ** 2).sum() / (d * d)

        total_loss = pairwise_loss + self.cov_weight * cov_loss
        return total_loss


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss (SupCon) handling multi-instance labels.

    Supports batches where some samples may have zero positive pairs (gracefully skipped).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCon loss across the batch.

        Args:
            features: L2-normalized feature tensor of shape (B, d).
            labels: Integer or categorical label tensor of shape (B,) or (B, 1).

        Returns:
            Scalar contrastive loss.
        """
        labels = labels.view(-1)
        b = features.size(0)
        if b <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Compute cosine similarity matrix: (B, B)
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)

        # For numerical stability, subtract max
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # Mask of positive pairs: (B, B)
        labels_eq = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(features.device)
        # Exclude self-similarity
        self_mask = torch.eye(b, device=features.device)
        pos_mask = labels_eq - self_mask

        # Number of positives per sample
        pos_counts = pos_mask.sum(dim=1)

        # Only consider samples that have at least one positive pair
        valid_samples = pos_counts > 0
        if not valid_samples.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Denominator: sum over all other samples (excluding self)
        denom_mask = 1.0 - self_mask
        exp_sim = torch.exp(sim_matrix) * denom_mask
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Numerator: log(exp(sim)) - log_denom = sim - log_denom
        log_prob = (sim_matrix - log_denom) * pos_mask

        # Mean over positive pairs per sample
        mean_log_prob_pos = log_prob[valid_samples].sum(dim=1) / pos_counts[valid_samples]
        loss = -mean_log_prob_pos.mean()
        return loss
