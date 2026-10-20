"""Unit and integration tests for Milestone 13.2 Auxiliary Heads and Multi-Head Detector."""

import hashlib
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.models.heads.auxiliary import (
    AuxiliaryManipulationHead,
    MidLevelSpatialHead,
    MANIPULATION_FAMILY_MAPPING,
)
from src.models.multi_head_detector import MultiHeadDeepfakeDetector, MultiHeadOutput
from src.utils.config import ModelConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
M12_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
FREEZE_MANIFEST_PATH = PROJECT_ROOT / "reports" / "m12_baseline_freeze.json"


def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_m12_frozen_integrity_remains_intact():
    """Verify that M12 baseline best.pth has not been modified."""
    if FREEZE_MANIFEST_PATH.exists() and M12_CHECKPOINT_PATH.exists():
        with open(FREEZE_MANIFEST_PATH, "r") as f:
            freeze_data = json.load(f)
        expected_hash = freeze_data["fingerprints"]["m12_checkpoint"]["sha256"]
        current_hash = sha256_file(M12_CHECKPOINT_PATH)
        assert current_hash == expected_hash, "M12 checkpoint has been altered!"


def test_auxiliary_manipulation_head_contract():
    """Test shape contracts, batch sizes, and probability conversion of AuxiliaryManipulationHead."""
    head = AuxiliaryManipulationHead(in_features=1280, hidden_dim=256, num_families=5, dropout=0.3)
    
    # Eval mode
    head.eval()
    feat = torch.randn(4, 1280)
    logits = head(feat)
    assert logits.shape == (4, 5), f"Expected shape (4, 5), got {logits.shape}"
    
    p_fake = head.compute_fake_probability(logits)
    assert p_fake.shape == (4, 1), f"Expected shape (4, 1), got {p_fake.shape}"
    assert (p_fake >= 0.0).all() and (p_fake <= 1.0).all(), "Probabilities must be in [0, 1]"

    # Single-sample inference guard
    single_feat = torch.randn(1, 1280)
    single_logits = head(single_feat)
    assert single_logits.shape == (1, 5)

    # Invalid dimension guards
    with pytest.raises(ValueError):
        head(torch.randn(4, 512))  # Wrong feature dim
    with pytest.raises(ValueError):
        head(torch.randn(4, 16, 1280))  # 3D tensor


def test_mid_level_spatial_head_contract():
    """Test 5D, 3D, and 2D tensor handling in MidLevelSpatialHead."""
    head = MidLevelSpatialHead(in_channels=80, hidden_dim=64, num_classes=1, dropout=0.2)
    head.eval()

    # 5D: (B=2, T=4, C=80, H=14, W=14)
    x_5d = torch.randn(2, 4, 80, 14, 14)
    out_5d = head(x_5d)
    assert out_5d.shape == (2, 1)

    # 3D: (B=3, T=4, C=80)
    x_3d = torch.randn(3, 4, 80)
    out_3d = head(x_3d)
    assert out_3d.shape == (3, 1)

    # 2D: (B=5, C=80)
    x_2d = torch.randn(5, 80)
    out_2d = head(x_2d)
    assert out_2d.shape == (5, 1)

    # 4D: (B*T=8, C=80, H=7, W=7) with batch_size=2
    x_4d = torch.randn(8, 80, 7, 7)
    out_4d = head(x_4d, batch_size=2)
    assert out_4d.shape == (2, 1)


def test_multi_head_detector_forward_and_freezing():
    """Test MultiHeadDeepfakeDetector with frozen base trunk."""
    cfg = ModelConfig()
    cfg.backbone.name = "efficientnet_b0"
    cfg.backbone.input_size = [112, 112]
    cfg.backbone.output_dim = 1280
    cfg.temporal.hidden_dim = 1280
    cfg.temporal.input_dim = 1280
    cfg.classifier.in_features = 1280

    model = MultiHeadDeepfakeDetector(config=cfg, num_families=5, enable_mid_head=True)
    
    # Test freezing
    model.freeze_trunk()
    for name, p in model.base_detector.named_parameters():
        assert not p.requires_grad, f"Parameter {name} in base detector should be frozen"
    for name, p in model.aux_head.named_parameters():
        assert p.requires_grad, f"Parameter {name} in aux_head should be trainable"

    # Fast forward from features
    feats = torch.randn(2, 1280)
    prim_logits, aux_logits = model.forward_from_features(feats)
    assert prim_logits.shape == (2, 1)
    assert aux_logits.shape == (2, 5)

    # Test multi-task loss computation
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    target_bin = torch.tensor([[1.0], [0.0]])
    target_fam = torch.tensor([2, 0], dtype=torch.long)

    loss_prim = bce(prim_logits, target_bin)
    loss_aux = ce(aux_logits, target_fam)
    total_loss = loss_prim + 0.5 * loss_aux
    assert total_loss.item() > 0.0

    # Test backward pass updates only aux_head
    total_loss.backward()
    for p in model.base_detector.parameters():
        assert p.grad is None, "Frozen base trunk parameters must not receive gradients"
    assert model.aux_head.fc1.weight.grad is not None, "Auxiliary head must receive gradients"
