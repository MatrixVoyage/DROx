"""Unit tests for Milestone 13.2c Disentangled Deepfake Detector and Losses."""

import hashlib
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.models.detector import DeepfakeDetector
from src.models.disentangled_detector import DisentangledDeepfakeDetector, DisentangledOutput
from src.models.heads.disentangled import (
    SubspaceOrthogonalityLoss,
    SubspaceProjectionHead,
    SupervisedContrastiveLoss,
)
from src.utils.config import ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
M12_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
FREEZE_MANIFEST_PATH = PROJECT_ROOT / "reports" / "m12_baseline_freeze.json"


def test_m12_baseline_invariance_m13_2c():
    """Verify that M12 baseline best.pth is completely untouched and bitwise immutable."""
    assert M12_CHECKPOINT_PATH.is_file(), f"M12 checkpoint missing at {M12_CHECKPOINT_PATH}"
    assert FREEZE_MANIFEST_PATH.is_file(), f"Freeze manifest missing at {FREEZE_MANIFEST_PATH}"

    with open(FREEZE_MANIFEST_PATH, "r") as f:
        freeze_data = json.load(f)

    expected_hash = freeze_data["fingerprints"]["m12_checkpoint"]["sha256"]

    hasher = hashlib.sha256()
    with open(M12_CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    current_hash = hasher.hexdigest()

    assert current_hash == expected_hash, (
        f"CRITICAL INVARIANT VIOLATION: M12 best.pth modified! "
        f"Expected {expected_hash}, got {current_hash}"
    )


def test_subspace_projection_and_normalization():
    """Verify SubspaceProjectionHead outputs expected shape and L2 normalized vectors."""
    b, in_dim, out_dim = 4, 1280, 128
    head = SubspaceProjectionHead(in_features=in_dim, hidden_dim=256, out_dim=out_dim)
    head.eval()

    x = torch.randn(b, in_dim)
    z = head(x, normalize=True)

    assert z.shape == (b, out_dim)
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(b), atol=1e-5)


def test_subspace_orthogonality_loss():
    """Verify orthogonality loss behavior on orthogonal vs collinear vectors."""
    orth_loss_fn = SubspaceOrthogonalityLoss(cov_weight=1.0)
    b, d = 8, 128

    # Case 1: Identical / collinear vectors (worst case)
    v = torch.randn(b, d)
    z1 = torch.nn.functional.normalize(v, p=2, dim=-1)
    z2 = z1.clone()
    high_loss = orth_loss_fn(z1, z2)
    assert high_loss.item() > 0.5, f"Expected high loss for collinear vectors, got {high_loss.item()}"

    # Case 2: Perfectly orthogonal vectors
    # Construct orthogonal bases: first d/2 non-zero for z_a, second d/2 for z_b
    z_a = torch.zeros(b, d)
    z_b = torch.zeros(b, d)
    z_a[:, : d // 2] = torch.nn.functional.normalize(torch.randn(b, d // 2), p=2, dim=-1)
    z_b[:, d // 2 :] = torch.nn.functional.normalize(torch.randn(b, d // 2), p=2, dim=-1)

    low_loss = orth_loss_fn(z_a, z_b)
    # Cosine alignment is exactly 0; cross-covariance is 0
    assert low_loss.item() < 1e-4, f"Expected near-zero loss for orthogonal vectors, got {low_loss.item()}"


def test_supervised_contrastive_loss():
    """Verify SupCon loss computes correctly and handles missing positive pairs gracefully."""
    supcon_fn = SupervisedContrastiveLoss(temperature=0.1)
    b, d = 6, 128

    # 3 groups with 2 samples each: [0, 0, 1, 1, 2, 2]
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    feats = torch.randn(b, d)
    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)

    loss = supcon_fn(feats, labels)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() > 0.0

    # Test edge case: all unique labels (zero positive pairs)
    unique_labels = torch.tensor([0, 1, 2, 3, 4, 5])
    zero_loss = supcon_fn(feats, unique_labels)
    assert zero_loss.item() == 0.0


def test_disentangled_detector_forward_and_isolation():
    """Verify DisentangledDeepfakeDetector forward contracts and strict gradient isolation."""
    model = DisentangledDeepfakeDetector(
        num_families=5,
        subspace_dim=128,
        projection_hidden_dim=256,
        enable_mid_head=True,
    )
    model.eval()

    b, t, c, h, w = 2, 16, 3, 112, 112
    x = torch.randn(b, t, c, h, w)

    out = model(x, return_dict=True)
    assert isinstance(out, DisentangledOutput)
    assert out.primary_logits.shape == (b, 1)
    assert out.aux_family_logits.shape == (b, 5)
    assert out.mid_logits.shape == (b, 1)
    assert out.z_id.shape == (b, 128)
    assert out.z_manip.shape == (b, 128)
    assert out.p_fused.shape == (b, 1)
    assert (out.p_fused >= 0.0).all() and (out.p_fused <= 1.0).all()

    # Gradient isolation test:
    # A backward pass from primary_logits MUST NOT backpropagate into id_head
    model.train()
    model.unfreeze_temporal_and_heads()
    model.zero_grad()

    x_train = torch.randn(b, t, c, h, w, requires_grad=True)
    out_train = model(x_train, return_dict=True)

    loss_prim = out_train.primary_logits.sum()
    loss_prim.backward()

    # Check gradients
    for name, p in model.manip_head.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, f"Expected grad in manip_head {name}"

    for name, p in model.primary_classifier.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, f"Expected grad in primary_classifier {name}"

    # CRITICAL: id_head must have NONE or ZERO gradients!
    for name, p in model.id_head.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum() == 0.0, f"STRICT ISOLATION VIOLATED: id_head {name} received non-zero grad!"
