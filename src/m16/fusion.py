"""Step 5: Evidence-Gated Fusion Architecture.

Builds modular branches selected through empirical incremental-value testing on the M16
Development/Validation split (Step 4 -> Step 5).
Learned gating parameters are initialized directly from diagnostic discriminability scores,
preventing uninformative handcrafted branches from eroding clean-real specificity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class EvidenceGatedFusion(nn.Module):
    """Evidence-gated multi-branch classifier supporting incremental modality testing.

    Architecture Invariants (M16):
    - Univariate AUROC is a diagnostic statistic, NOT an architectural gate.
    - Modalities are selected through empirical incremental-value testing on the
      M16 Development/Validation split (e.g. M13.5 + Temporal vs M13.5).
    - Anchor backbone representations (M13.5) are permanently preserved as the anchor trunk.
    - Active candidate modalities project to 64-D vectors with learned sigmoid gates.
    """

    def __init__(
        self,
        candidate_modalities: List[str] | Dict[str, int],
        signal_priors: Optional[Dict[str, float]] = None,
        anchor_dim: int = 1280,  # EfficientNet-B0 penultimate feature dim (or 256 for compressed trunk)
    ):
        """Initialize the EvidenceGatedFusion module.

        Args:
            candidate_modalities: List of modality names or dict mapping name to input dimension.
            signal_priors: Optional dictionary mapping modality names to diagnostic scores (e.g. discovery AUROC)
                           used strictly to initialize gate logit priors.
            anchor_dim: Dimensionality of M13.5 anchor representation.
        """
        super().__init__()
        self.anchor_dim = anchor_dim

        # Resolve candidate modalities and their input dimensions
        if isinstance(candidate_modalities, dict):
            self.feature_dims = candidate_modalities
        else:
            self.feature_dims = {name: 1 for name in candidate_modalities}

        self.active_branches: List[str] = list(self.feature_dims.keys())
        priors = signal_priors or {}

        # Project each active forensic branch to a unified 64-dimensional space
        self.projectors = nn.ModuleDict()
        self.gates = nn.ParameterDict()

        for name, in_dim in self.feature_dims.items():
            prior = priors.get(name, 0.50)  # Default neutral prior (0.50 -> 0.0 logit)
            self.projectors[name] = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
            )
            # Gate logit initialized from prior: 0.5 -> 0.0, 0.8 -> 0.6
            init_val = float(prior * 2.0 - 1.0)
            self.gates[name] = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))

        # Total fused representation size = anchor_dim + (64 * number of active branches)
        total_fused_dim = self.anchor_dim + (64 * len(self.active_branches))

        self.classifier = nn.Sequential(
            nn.Linear(total_fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )


    def forward(
        self,
        m13_5_features: torch.Tensor,
        signal_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass through evidence-gated fusion network.

        Args:
            m13_5_features: Backbone temporal anchor features of shape (B, anchor_dim).
            signal_features: Optional dictionary of forensic branch tensors (B, dim_i).

        Returns:
            Scalar logit tensor of shape (B, 1).
        """
        # Ensure 2D tensor
        if m13_5_features.dim() == 1:
            m13_5_features = m13_5_features.unsqueeze(0)

        branch_outputs = [m13_5_features]
        sig_feats = signal_features or {}

        b_size = m13_5_features.shape[0]
        device = m13_5_features.device

        for name, projector in self.projectors.items():
            if name in sig_feats:
                feat = sig_feats[name]
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                feat = feat.to(device, dtype=torch.float32)
                # Reshape to (B, in_dim) if needed
                if feat.shape[0] != b_size:
                    feat = feat.expand(b_size, -1)
                projected = projector(feat)
                gate = torch.sigmoid(self.gates[name])
                branch_outputs.append(projected * gate)
            else:
                # Zero imputation if branch missing at inference
                zero_feat = torch.zeros(b_size, 64, device=device, dtype=torch.float32)
                branch_outputs.append(zero_feat)

        combined = torch.cat(branch_outputs, dim=-1)
        return self.classifier(combined)

    def gate_report(self) -> Dict[str, float]:
        """Print and return active branch contributions and learned sigmoid gate values."""
        print("\n" + "=" * 60)
        print("M16 EVIDENCE-GATED FUSION REPORT")
        print("=" * 60)
        print(f"Anchor Backbone Dim: {self.anchor_dim}")
        print(f"Active Candidate Branches ({len(self.active_branches)}):")

        gate_vals = {}
        for name, gate in self.gates.items():
            g = float(torch.sigmoid(gate).item())
            gate_vals[name] = round(g, 4)
            bar_len = int(g * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {name:<35} |{bar}| {g:.4f}")

        print("=" * 60)
        return gate_vals


def evaluate_incremental_modality_value(
    y_true: List[int] | np.ndarray,
    baseline_probs: List[float] | np.ndarray,
    candidate_probs: List[float] | np.ndarray,
    modality_name: str = "candidate_modality",
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Measure the empirical incremental value of adding a candidate forensic modality to M13.5.

    Args:
        y_true: Ground truth binary labels (1=Fake, 0=Real).
        baseline_probs: M13.5 baseline predicted probabilities.
        candidate_probs: M13.5 + Modality predicted probabilities.
        modality_name: Name of the candidate forensic modality.
        threshold: Decision threshold.

    Returns:
        Dictionary reporting Delta AUC, Delta Detection, and Delta Specificity.
    """
    from sklearn.metrics import roc_auc_score, confusion_matrix

    y_t = np.array(y_true)
    p_base = np.array(baseline_probs)
    p_cand = np.array(candidate_probs)

    auc_base = float(roc_auc_score(y_t, p_base)) if len(np.unique(y_t)) > 1 else 0.5
    auc_cand = float(roc_auc_score(y_t, p_cand)) if len(np.unique(y_t)) > 1 else 0.5

    pred_base = (p_base >= threshold).astype(int)
    pred_cand = (p_cand >= threshold).astype(int)

    cm_b = confusion_matrix(y_t, pred_base, labels=[0, 1])
    cm_c = confusion_matrix(y_t, pred_cand, labels=[0, 1])

    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
    tn_c, fp_c, fn_c, tp_c = cm_c.ravel()

    rec_b = float(tp_b / (tp_b + fn_b)) if (tp_b + fn_b) > 0 else 0.0
    rec_c = float(tp_c / (tp_c + fn_c)) if (tp_c + fn_c) > 0 else 0.0

    spec_b = float(tn_b / (tn_b + fp_b)) if (tn_b + fp_b) > 0 else 1.0
    spec_c = float(tn_c / (tn_c + fp_c)) if (tn_c + fp_c) > 0 else 1.0

    delta_auc = auc_cand - auc_base
    delta_rec = rec_c - rec_b
    delta_spec = spec_c - spec_b

    # A modality adds true incremental value if it increases AUC or recall without eroding specificity
    adds_value = bool((delta_auc > 0.005 or delta_rec > 0.02) and delta_spec >= -0.02)

    return {
        "modality": modality_name,
        "baseline_auc": round(auc_base, 4),
        "candidate_auc": round(auc_cand, 4),
        "delta_auc": round(delta_auc, 4),
        "baseline_detection_rate": round(rec_b, 4),
        "candidate_detection_rate": round(rec_c, 4),
        "delta_detection_rate": round(delta_rec, 4),
        "baseline_specificity": round(spec_b, 4),
        "candidate_specificity": round(spec_c, 4),
        "delta_specificity": round(delta_spec, 4),
        "adds_incremental_value": adds_value,
    }

