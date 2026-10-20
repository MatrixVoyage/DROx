"""Unit and integration tests for ClassifierHead and DeepfakeDetector in Milestone 4."""

import io
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.models.classifier import ClassifierHead
from src.models.detector import DeepfakeDetector, DetectorOutput
from src.utils.config import (
    AppConfig,
    AttentionConfig,
    BackboneConfig,
    ClassifierConfig,
    ModelConfig,
    TemporalConfig,
    load_config,
    save_config,
)


# ============================================================================
# ClassifierHead Tests
# ============================================================================


def test_classifier_output_shape():
    """Verify ClassifierHead maps (B, 2048) temporal representations to (B, 1) scalar logits."""
    head = ClassifierHead()
    head.eval()

    for batch_size in (1, 4):
        x = torch.randn(batch_size, 2048)
        logits = head(x)
        assert logits.shape == (batch_size, 1)
        assert logits.dtype == torch.float32


def test_classifier_validation():
    """Verify input dimension and config validation for ClassifierHead."""
    head = ClassifierHead()

    # Rejects 1D or 3D tensors
    with pytest.raises(ValueError, match="Expected 2D input tensor"):
        head(torch.randn(2048))

    with pytest.raises(ValueError, match="Expected 2D input tensor"):
        head(torch.randn(2, 4, 2048))

    # Rejects B=0
    with pytest.raises(ValueError, match="Batch size must be > 0"):
        head(torch.randn(0, 2048))

    # Rejects D != 2048
    with pytest.raises(ValueError, match="Expected input feature dimension D=2048"):
        head(torch.randn(2, 512))

    # Rejects invalid config parameters
    with pytest.raises(ValueError, match="classifier.in_features must be 2048"):
        ClassifierConfig(in_features=1024)

    with pytest.raises(ValueError, match="classifier.hidden_dim must be > 0"):
        ClassifierConfig(hidden_dim=0)

    with pytest.raises(ValueError, match="classifier.num_classes must be 1"):
        ClassifierConfig(num_classes=2)

    with pytest.raises(ValueError, match=r"classifier.dropout must be in \[0.0, 1.0\)"):
        ClassifierConfig(dropout=1.0)


def test_classifier_bce_loss_compatibility():
    """Verify ClassifierHead raw scalar logits interface directly with nn.BCEWithLogitsLoss."""
    head = ClassifierHead()
    head.train()
    criterion = nn.BCEWithLogitsLoss()

    x = torch.randn(4, 2048)
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

    logits = head(x)
    loss = criterion(logits, targets)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None
        assert p.grad.abs().sum() > 0.0


def test_classifier_eval_determinism():
    """Verify ClassifierHead is deterministic in eval mode and stochastic in train mode."""
    head = ClassifierHead()
    x = torch.randn(4, 2048)

    head.eval()
    with torch.no_grad():
        out1 = head(x)
        out2 = head(x)
        assert torch.equal(out1, out2)

    head.train()
    out_train1 = head(x)
    out_train2 = head(x)
    assert not torch.equal(out_train1, out_train2)


# ============================================================================
# DeepfakeDetector Integration Tests
# ============================================================================


def test_deepfake_detector_construction_frozen_backbone():
    """Verify DeepfakeDetector enforces cumulative freezing of stem/layer1/layer2."""
    cfg = ModelConfig(
        backbone=BackboneConfig(weights=None, freeze_strategy="layer2"),
        attention=AttentionConfig(type="soft"),
        temporal=TemporalConfig(type="lstm"),
        classifier=ClassifierConfig(),
    )
    detector = DeepfakeDetector(cfg)

    # Frozen stages
    for stage in (detector.backbone.stem, detector.backbone.layer1, detector.backbone.layer2):
        for p in stage.parameters():
            assert not p.requires_grad

    # Unfrozen stages
    for stage in (detector.backbone.layer3, detector.backbone.layer4):
        for p in stage.parameters():
            assert p.requires_grad

    # Attention, temporal, classifier must be trainable
    for p in detector.attention.parameters():
        assert p.requires_grad
    for p in detector.temporal.parameters():
        assert p.requires_grad
    for p in detector.classifier.parameters():
        assert p.requires_grad


def test_deepfake_detector_input_validation():
    """Verify detector rejects malformed input tensors."""
    cfg = ModelConfig(backbone=BackboneConfig(weights=None))
    detector = DeepfakeDetector(cfg)

    # Rejects 4D tensor
    with pytest.raises(ValueError, match="Expected 5D sequence tensor"):
        detector(torch.randn(2, 3, 112, 112))

    # Rejects B=0 or T=0
    with pytest.raises(ValueError, match="Batch size B and sequence length T must be > 0"):
        detector(torch.randn(0, 4, 3, 112, 112))

    with pytest.raises(ValueError, match="Batch size B and sequence length T must be > 0"):
        detector(torch.randn(1, 0, 3, 112, 112))

    # Rejects incorrect frame resolution or channels
    with pytest.raises(ValueError, match=r"Expected frame shape \(3, 112, 112\)"):
        detector(torch.randn(1, 4, 1, 112, 112))

    with pytest.raises(ValueError, match=r"Expected frame shape \(3, 112, 112\)"):
        detector(torch.randn(1, 4, 3, 224, 224))


def test_deepfake_detector_end_to_end_forward():
    """Verify complete end-to-end forward pass produces raw logits of shape (B, 1)."""
    cfg = ModelConfig(backbone=BackboneConfig(weights=None))
    detector = DeepfakeDetector(cfg)
    detector.eval()

    # Use sequence length T=4 for efficient testing
    x = torch.randn(2, 4, 3, 112, 112)
    with torch.no_grad():
        logits = detector(x)

    assert logits.shape == (2, 1)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_deepfake_detector_intermediates():
    """Verify return_intermediates=True produces populated DetectorOutput container."""
    cfg = ModelConfig(backbone=BackboneConfig(weights=None))
    detector = DeepfakeDetector(cfg)
    detector.eval()

    x = torch.randn(1, 4, 3, 112, 112)
    with torch.no_grad():
        output = detector(x, return_intermediates=True)

    assert isinstance(output, DetectorOutput)
    assert output.logits.shape == (1, 1)
    assert output.probabilities.shape == (1, 1)
    # Probabilities must match torch.sigmoid(logits)
    assert torch.allclose(output.probabilities, torch.sigmoid(output.logits), atol=1e-6)
    assert (0.0 <= output.probabilities <= 1.0).all()

    # Intermediate shapes
    assert output.frame_features.shape == (1, 4, 2048)
    assert output.temporal_features.shape == (1, 2048)
    assert isinstance(output.attention_maps, dict)
    assert "soft_map" in output.attention_maps
    assert output.attention_maps["soft_map"].shape == (1, 4, 1, 4, 4)


def test_deepfake_detector_attention_map_propagation():
    """Verify attention map extraction across all 5 attention configurations."""
    attention_types = ["soft", "none", "channel", "spatial", "cbam"]

    for attn_type in attention_types:
        cfg = ModelConfig(
            backbone=BackboneConfig(weights=None),
            attention=AttentionConfig(type=attn_type),
        )
        detector = DeepfakeDetector(cfg)
        detector.eval()

        x = torch.randn(1, 2, 3, 112, 112)
        with torch.no_grad():
            output = detector(x, return_intermediates=True)

        assert isinstance(output.attention_maps, dict)
        if attn_type == "soft":
            assert "soft_map" in output.attention_maps
            assert output.attention_maps["soft_map"].shape == (1, 2, 1, 4, 4)
        elif attn_type == "none":
            assert len(output.attention_maps) == 0
        elif attn_type == "channel":
            assert "channel_map" in output.attention_maps
            assert output.attention_maps["channel_map"].shape == (1, 2, 2048, 1, 1)
        elif attn_type == "spatial":
            assert "spatial_map" in output.attention_maps
            assert output.attention_maps["spatial_map"].shape == (1, 2, 1, 4, 4)
        elif attn_type == "cbam":
            assert "channel_map" in output.attention_maps
            assert "spatial_map" in output.attention_maps
            assert output.attention_maps["channel_map"].shape == (1, 2, 2048, 1, 1)
            assert output.attention_maps["spatial_map"].shape == (1, 2, 1, 4, 4)



def test_deepfake_detector_eval_determinism():
    """Verify DeepfakeDetector forward pass is bitwise identical across multiple eval passes."""
    cfg = ModelConfig(backbone=BackboneConfig(weights=None))
    detector = DeepfakeDetector(cfg)
    detector.eval()

    x = torch.randn(1, 4, 3, 112, 112)
    with torch.no_grad():
        out1 = detector(x)
        out2 = detector(x)

    assert torch.equal(out1, out2)


def test_deepfake_detector_state_dict_checkpoint_compatibility():
    """Verify model state dict can be serialized, reloaded, and produces exact outputs."""
    cfg = ModelConfig(backbone=BackboneConfig(weights=None))
    model1 = DeepfakeDetector(cfg)
    model1.eval()

    buffer = io.BytesIO()
    torch.save(model1.state_dict(), buffer)
    buffer.seek(0)

    model2 = DeepfakeDetector(cfg)
    model2.load_state_dict(torch.load(buffer, weights_only=True))
    model2.eval()

    x = torch.randn(1, 4, 3, 112, 112)
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    assert torch.equal(out1, out2)


def test_deepfake_detector_end_to_end_gradient_flow():
    """Verify gradients propagate end-to-end to unfrozen stages and respect frozen backbone stages."""
    cfg = ModelConfig(
        backbone=BackboneConfig(weights=None, freeze_strategy="layer2"),
        attention=AttentionConfig(type="soft"),
        temporal=TemporalConfig(type="lstm"),
        classifier=ClassifierConfig(),
    )
    detector = DeepfakeDetector(cfg)
    detector.train()

    x = torch.randn(1, 2, 3, 112, 112)
    target = torch.tensor([[1.0]], dtype=torch.float32)

    logits = detector(x)
    loss = nn.BCEWithLogitsLoss()(logits, target)
    loss.backward()

    # 1. Frozen backbone parameters must have None grad
    for stage in (detector.backbone.stem, detector.backbone.layer1, detector.backbone.layer2):
        for name, p in stage.named_parameters():
            assert p.grad is None, f"Frozen parameter {name} received gradient"

    # 2. Unfrozen backbone parameters must have non-zero grad
    for stage in (detector.backbone.layer3, detector.backbone.layer4):
        for name, p in stage.named_parameters():
            assert p.grad is not None, f"Trainable backbone param {name} missing gradient"
            assert p.grad.abs().sum() > 0.0

    # 3. Attention module parameters must have non-zero grad
    for name, p in detector.attention.named_parameters():
        assert p.grad is not None, f"Attention param {name} missing gradient"
        assert p.grad.abs().sum() > 0.0

    # 4. Temporal module parameters must have non-zero grad
    for name, p in detector.temporal.lstm.named_parameters():
        assert p.grad is not None, f"Temporal param {name} missing gradient"
        assert p.grad.abs().sum() > 0.0

    # 5. Classifier head parameters must have non-zero grad
    for name, p in detector.classifier.named_parameters():
        assert p.grad is not None, f"Classifier param {name} missing gradient"
        assert p.grad.abs().sum() > 0.0


def test_deepfake_detector_cuda_fp16_amp():
    """Verify detector forward and backward pass on CUDA under FP16 Automatic Mixed Precision."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")

    cfg = ModelConfig(backbone=BackboneConfig(weights=None, freeze_strategy="layer2"))
    detector = DeepfakeDetector(cfg).cuda()
    detector.train()

    # Standard training sequence: B=1, T=16, 3, 112, 112
    x = torch.randn(1, 16, 3, 112, 112, device="cuda")
    target = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")

    scaler = torch.amp.GradScaler("cuda")
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = detector(x)
        loss = nn.BCEWithLogitsLoss()(logits, target)

    assert logits.shape == (1, 1)
    assert torch.isfinite(loss)

    scaler.scale(loss).backward()
    scaler.step(torch.optim.Adam(filter(lambda p: p.requires_grad, detector.parameters()), lr=1e-4))
    scaler.update()

    # Peak VRAM must be well within RTX 3050 6GB budget
    max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    assert max_memory_mb < 6000, f"Peak memory {max_memory_mb:.2f} MB exceeded 6 GB budget"
