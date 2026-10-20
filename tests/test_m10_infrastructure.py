"""Comprehensive unit tests for Milestone 10 infrastructure components.

Covers:
    - Loss numerical stability (FocalLoss, SupervisedContrastiveLoss, CompositeDetectionLoss)
    - Exact 1:1 sampler balance (BalancedBatchSampler)
    - Temporal-consistency of augmentations (SequenceAugmentor)
    - Temporal Transformer shapes, pooling modes, and gradients
    - EfficientNet backbone output shapes, freezing, and factory
    - Frozen-checkpoint bitwise integrity
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.samplers import BalancedBatchSampler, SequenceAugmentor
from src.models.backbones.factory import build_backbone
from src.models.backbones.efficientnet import EfficientNetBackbone
from src.models.temporal.factory import build_temporal_model
from src.models.temporal.transformer import TemporalTransformer
from src.training.losses import CompositeDetectionLoss, FocalLoss, SupervisedContrastiveLoss
from src.utils.config import BackboneConfig, TemporalConfig


# ==============================================================================
# 1. LOSS NUMERICAL STABILITY & FORMULATION TESTS
# ==============================================================================

def test_focal_loss_numerical_stability():
    """Verify FocalLoss numerical stability with extreme logits and gradients."""
    fl = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

    # Extreme logits (+50, -50)
    extreme_logits = torch.tensor([[50.0], [-50.0], [0.0], [10.0]], requires_grad=True)
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    loss = fl(extreme_logits, targets)
    assert not torch.isnan(loss), "FocalLoss produced NaN on extreme logits"
    assert not torch.isinf(loss), "FocalLoss produced Inf on extreme logits"
    assert loss.item() >= 0.0, "Loss must be non-negative"

    # Test gradient backpropagation
    loss.backward()
    assert extreme_logits.grad is not None
    assert not torch.isnan(extreme_logits.grad).any()
    assert not torch.isinf(extreme_logits.grad).any()


def test_focal_loss_matches_bce_when_gamma_zero():
    """Verify FocalLoss with gamma=0 and alpha=0.5 equals 0.5 * BCEWithLogitsLoss."""
    fl = FocalLoss(alpha=0.5, gamma=0.0, reduction="mean")
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    logits = torch.randn(10, 1)
    targets = torch.randint(0, 2, (10, 1)).float()

    fl_val = fl(logits, targets)
    bce_val = bce(logits, targets) * 0.5

    assert torch.allclose(fl_val, bce_val, atol=1e-5), f"FL={fl_val.item()}, 0.5*BCE={bce_val.item()}"


def test_focal_loss_parameter_validation():
    """Verify FocalLoss rejects invalid parameters."""
    with pytest.raises(ValueError):
        FocalLoss(alpha=-0.1)
    with pytest.raises(ValueError):
        FocalLoss(alpha=1.5)
    with pytest.raises(ValueError):
        FocalLoss(gamma=-1.0)
    with pytest.raises(ValueError):
        FocalLoss(reduction="invalid")


def test_supervised_contrastive_loss():
    """Verify SupervisedContrastiveLoss pulls same-class embeddings together."""
    con_loss = SupervisedContrastiveLoss(temperature=0.1)

    # 4 samples: 2 real (class 0), 2 fake (class 1)
    # Give class 0 identical features, class 1 identical features orthogonal to class 0
    f_c0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    f_c1 = torch.tensor([0.0, 1.0, 0.0, 0.0])

    features = torch.stack([f_c0, f_c0, f_c1, f_c1]).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1])

    loss = con_loss(features, labels)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0

    loss.backward()
    assert features.grad is not None
    assert not torch.isnan(features.grad).any()


def test_supervised_contrastive_loss_single_class_guard():
    """Verify SupCon gracefully handles batches with only one class."""
    con_loss = SupervisedContrastiveLoss(temperature=0.1)
    features = torch.randn(4, 16, requires_grad=True)
    labels = torch.tensor([1, 1, 1, 1])  # Only one class

    loss = con_loss(features, labels)
    assert loss.item() == 0.0
    assert not torch.isnan(loss)


def test_composite_detection_loss():
    """Verify CompositeDetectionLoss computes combined loss and components."""
    comp = CompositeDetectionLoss(
        cls_loss_type="focal",
        alpha=0.25,
        gamma=2.0,
        contrastive_weight=0.2,
        temperature=0.1,
    )

    logits = torch.randn(4, 1, requires_grad=True)
    targets = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    features = torch.randn(4, 64, requires_grad=True)

    total_loss, loss_dict = comp(logits, targets, features=features)
    assert not torch.isnan(total_loss)
    assert "cls_loss" in loss_dict
    assert "con_loss" in loss_dict
    assert "total_loss" in loss_dict
    assert total_loss.item() > 0.0

    total_loss.backward()
    assert logits.grad is not None
    assert features.grad is not None


# ==============================================================================
# 2. EXACT 1:1 BALANCED BATCH SAMPLER TESTS
# ==============================================================================

def test_balanced_batch_sampler_exact_balance():
    """Verify BalancedBatchSampler yields exactly 1:1 balance in every batch."""
    # 20 real (0) and 80 fake (1) - 4:1 imbalance
    labels = [0] * 20 + [1] * 80
    batch_size = 4  # Each batch must have exactly 2 real, 2 fake

    sampler = BalancedBatchSampler(labels=labels, batch_size=batch_size, shuffle=True, seed=42)

    assert len(sampler) == 40  # 80 fake // 2 = 40 batches

    for batch_indices in sampler:
        assert len(batch_indices) == batch_size
        batch_labels = [labels[i] for i in batch_indices]
        assert batch_labels.count(0) == batch_size // 2, f"Expected 2 real, got {batch_labels.count(0)}"
        assert batch_labels.count(1) == batch_size // 2, f"Expected 2 fake, got {batch_labels.count(1)}"


def test_balanced_batch_sampler_validation():
    """Verify BalancedBatchSampler parameter checks."""
    # Odd batch size must be rejected
    with pytest.raises(ValueError):
        BalancedBatchSampler(labels=[0, 1], batch_size=3)

    # Zero positive samples must be rejected
    with pytest.raises(ValueError):
        BalancedBatchSampler(labels=[0, 0, 0], batch_size=2)

    # Zero negative samples must be rejected
    with pytest.raises(ValueError):
        BalancedBatchSampler(labels=[1, 1, 1], batch_size=2)


# ==============================================================================
# 3. TEMPORAL-CONSISTENCY OF AUGMENTATIONS
# ==============================================================================

def test_sequence_augmentor_temporal_consistency():
    """Verify SequenceAugmentor applies identical transform parameters across all frames."""
    t, c, h, w = 16, 3, 112, 112
    # Create non-symmetric sequence where left half is 1.0, right half is 0.0
    seq = torch.zeros(t, c, h, w)
    seq[:, :, :, : w // 2] = 1.0

    # Force flip
    augmentor = SequenceAugmentor(
        horizontal_flip_p=1.0,
        color_jitter_p=0.0,
        is_training=True,
    )
    aug_seq = augmentor(seq)

    # Check shape
    assert aug_seq.shape == (t, c, h, w)

    # Verify that EVERY frame was flipped identically: right half should now be 1.0
    for frame_idx in range(t):
        frame = aug_seq[frame_idx]
        assert torch.allclose(frame[:, :, w // 2 :], torch.ones(c, h, w // 2)), (
            f"Frame {frame_idx} was not flipped consistently!"
        )
        assert torch.allclose(frame[:, :, : w // 2], torch.zeros(c, h, w // 2)), (
            f"Frame {frame_idx} was not flipped consistently!"
        )


def test_sequence_augmentor_eval_identity():
    """Verify SequenceAugmentor is strict identity when is_training=False."""
    seq = torch.randn(16, 3, 112, 112)
    augmentor = SequenceAugmentor(is_training=False)
    out_seq = augmentor(seq)
    assert torch.equal(seq, out_seq), "Eval augmentor must return identical tensor."


# ==============================================================================
# 4. TEMPORAL TRANSFORMER TESTS
# ==============================================================================

def test_temporal_transformer_shapes_and_gradients():
    """Verify TemporalTransformer forward shapes, pooling modes, and gradients."""
    # Test D=2048 (ResNeXt dim)
    cfg_mean = TemporalConfig(
        type="transformer",
        input_dim=2048,
        nhead=4,
        transformer_layers=2,
        dropout=0.1,
        pooling="mean",
    )
    model_mean = TemporalTransformer(cfg_mean)

    x = torch.randn(2, 16, 2048, requires_grad=True)
    out_mean = model_mean(x)
    assert out_mean.shape == (2, 2048), f"Expected (2, 2048), got {out_mean.shape}"

    # Gradient check
    out_mean.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Test D=1280 (EfficientNet dim) with cls_token pooling
    cfg_cls = TemporalConfig(
        type="transformer",
        input_dim=1280,
        nhead=4,
        transformer_layers=2,
        dropout=0.1,
        pooling="cls_token",
    )
    model_cls = TemporalTransformer(cfg_cls)
    x_eff = torch.randn(3, 16, 1280)
    out_cls = model_cls(x_eff)
    assert out_cls.shape == (3, 1280), f"Expected (3, 1280), got {out_cls.shape}"


def test_temporal_factory_with_transformer():
    """Verify temporal factory instantiates TemporalTransformer."""
    cfg = TemporalConfig(type="transformer", input_dim=2048)
    module = build_temporal_model(cfg)
    assert isinstance(module, TemporalTransformer)
    assert cfg.output_dim == 2048


# ==============================================================================
# 5. EFFICIENTNET BACKBONE & FACTORY TESTS
# ==============================================================================

def test_efficientnet_backbone_shapes():
    """Verify EfficientNetBackbone output dimensions for both 4D and 5D inputs."""
    cfg = BackboneConfig(name="efficientnet_b0", output_dim=1280, weights=None)
    backbone = EfficientNetBackbone(cfg)

    # 4D input: (B, 3, 112, 112)
    x4 = torch.randn(2, 3, 112, 112)
    pooled4, spatial4 = backbone(x4, return_spatial_map=True)
    assert pooled4.shape == (2, 1280)
    assert spatial4.shape == (2, 1280, 4, 4)

    # 5D input: (B, T, 3, 112, 112)
    x5 = torch.randn(2, 16, 3, 112, 112)
    pooled5, spatial5 = backbone(x5, return_spatial_map=True)
    assert pooled5.shape == (2, 16, 1280)
    assert spatial5.shape == (2, 16, 1280, 4, 4)

    # Provenance
    prov = backbone.get_provenance()
    assert prov["name"] == "efficientnet_b0"
    assert prov["output_dim"] == 1280
    assert prov["total_parameters"] > 0


def test_backbone_factory():
    """Verify backbone factory instantiates both ResNeXt and EfficientNet."""
    resnext_cfg = BackboneConfig(name="resnext50_32x4d", weights=None)
    bb_resnext = build_backbone(resnext_cfg)
    assert bb_resnext.__class__.__name__ == "ResNeXt50Backbone"

    eff_cfg = BackboneConfig(name="efficientnet_b0", output_dim=1280, weights=None)
    bb_eff = build_backbone(eff_cfg)
    assert bb_eff.__class__.__name__ == "EfficientNetBackbone"


# ==============================================================================
# 6. FROZEN-CHECKPOINT INTEGRITY TEST
# ==============================================================================

def test_frozen_baseline_checkpoint_sha256():
    """CRITICAL INVARIANT: Verify best.pth remains 100% bitwise identical."""
    ckpt_path = Path("checkpoints/ffpp_c23_baseline_soft_lstm/best.pth")
    assert ckpt_path.exists(), "Reference checkpoint does not exist!"

    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    expected_sha256 = "9d22df92bdab20a35e1311d627d9fa487c757b7afb32a249ebbf7f95bf3b7e6a"
    actual_sha256 = hasher.hexdigest()

    assert actual_sha256 == expected_sha256, (
        f"INVARIANT VIOLATION: best.pth was modified!\n"
        f"Expected: {expected_sha256}\n"
        f"Actual:   {actual_sha256}"
    )


# ==============================================================================
# 7. ADDITIONAL FACTORY, LOSS & FULL DETECTOR INTEGRATION TESTS
# ==============================================================================

def test_backbone_factory_invalid_name():
    """Verify backbone factory rejects unknown backbone names."""
    with pytest.raises(ValueError, match="Unknown backbone name"):
        build_backbone(BackboneConfig(name="unknown_backbone"))


def test_temporal_factory_invalid_type():
    """Verify temporal factory rejects unknown temporal types."""
    with pytest.raises(ValueError, match="Invalid temporal.type"):
        build_temporal_model(TemporalConfig(type="unknown_temporal"))


def test_efficientnet_freezing_modes():
    """Verify parameter freezing in EfficientNetBackbone."""
    for strat in ("early", "mid", "late", "all"):
        cfg = BackboneConfig(name="efficientnet_b0", output_dim=1280, freeze_strategy=strat, weights=None)
        m = EfficientNetBackbone(cfg)
        assert len(m.frozen_stages) > 0


def test_composite_loss_weight_zero():
    """Verify CompositeDetectionLoss with contrastive_weight=0."""
    comp = CompositeDetectionLoss(cls_loss_type="focal", contrastive_weight=0.0)
    logits = torch.randn(4, 1)
    targets = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    features = torch.randn(4, 64)
    total_loss, loss_dict = comp(logits, targets, features=features)
    assert loss_dict["con_loss"] == 0.0
    assert total_loss.item() == pytest.approx(loss_dict["cls_loss"], rel=1e-5)


def test_full_detector_resnext_transformer_forward():
    """Verify DeepfakeDetector with ResNeXt-50 + TemporalTransformer."""
    from src.models.detector import DeepfakeDetector
    from src.utils.config import AttentionConfig, ClassifierConfig, ModelConfig

    cfg = ModelConfig(
        backbone=BackboneConfig(name="resnext50_32x4d", weights=None, freeze_strategy="none"),
        attention=AttentionConfig(type="none", in_channels=2048),
        temporal=TemporalConfig(type="transformer", input_dim=2048, nhead=4, transformer_layers=1),
        classifier=ClassifierConfig(in_features=2048),
    )
    detector = DeepfakeDetector(cfg)
    detector.eval()

    x = torch.randn(2, 16, 3, 112, 112)
    with torch.no_grad():
        logits, embeddings = detector(x, return_embeddings=True)
    assert logits.shape == (2, 1)
    assert embeddings.shape == (2, 2048)


def test_full_detector_efficientnet_transformer_forward():
    """Verify DeepfakeDetector with EfficientNet-B0 + TemporalTransformer."""
    from src.models.detector import DeepfakeDetector
    from src.utils.config import AttentionConfig, ClassifierConfig, ModelConfig

    cfg = ModelConfig(
        backbone=BackboneConfig(name="efficientnet_b0", output_dim=1280, weights=None, freeze_strategy="none"),
        attention=AttentionConfig(type="none", in_channels=1280),
        temporal=TemporalConfig(type="transformer", input_dim=1280, nhead=4, transformer_layers=1),
        classifier=ClassifierConfig(in_features=1280),
    )
    detector = DeepfakeDetector(cfg)
    detector.eval()

    x = torch.randn(2, 16, 3, 112, 112)
    with torch.no_grad():
        logits, embeddings = detector(x, return_embeddings=True)
    assert logits.shape == (2, 1)
    assert embeddings.shape == (2, 1280)

