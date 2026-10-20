"""Unit and integration tests for Attention Mechanism modules (Milestone 3)."""

import pytest
import torch
import torch.nn as nn

from src.models.attention import (
    CBAMAttention,
    ChannelAttention,
    IdentityPooling,
    SoftAttention,
    SpatialAttention,
    build_attention_module,
)
from src.models.backbones.resnext import ResNeXt50Backbone
from src.utils.config import AppConfig, AttentionConfig, BackboneConfig, ModelConfig


ALL_ATTN_TYPES = ["soft", "cbam", "channel", "spatial", "none"]


@pytest.mark.parametrize("attn_type", ALL_ATTN_TYPES)
def test_attention_factory_all_modes(attn_type):
    """Verify factory builds the exact expected class and callable interface for every mode."""
    cfg = AttentionConfig(type=attn_type)
    module = build_attention_module(cfg)

    expected_classes = {
        "soft": SoftAttention,
        "cbam": CBAMAttention,
        "channel": ChannelAttention,
        "spatial": SpatialAttention,
        "none": IdentityPooling,
    }
    assert isinstance(module, expected_classes[attn_type])
    assert callable(module)


def test_attention_factory_invalid_type():
    """Verify factory raises ValueError on unknown type."""
    with pytest.raises(ValueError, match="Invalid attention.type"):
        AttentionConfig(type="invalid_type")


def test_attention_parameter_counts():
    """Verify exact trainable parameter counts for each module."""
    soft = SoftAttention(2048)
    cam = ChannelAttention(2048, reduction_ratio=16)
    sam = SpatialAttention(2048, kernel_size=3)
    cbam = CBAMAttention(2048, reduction_ratio=16, kernel_size=3)
    identity = IdentityPooling(2048)

    # Soft attention: Conv2d(2048 -> 1, 1x1, bias=True) => 2048 * 1 + 1 = 2,049
    assert sum(p.numel() for p in soft.parameters()) == 2049

    # CAM MLP: Linear(2048, 128, bias=False) + Linear(128, 2048, bias=False) => 2048*128 + 128*2048 = 524,288
    assert sum(p.numel() for p in cam.parameters()) == 524288

    # SAM Conv: Conv2d(2 -> 1, 3x3, bias=True) => 2 * 1 * 3 * 3 + 1 = 19
    assert sum(p.numel() for p in sam.parameters()) == 19

    # CBAM: CAM + SAM => 524,288 + 19 = 524,307
    assert sum(p.numel() for p in cbam.parameters()) == 524307

    # Identity: 0 parameters
    assert sum(p.numel() for p in identity.parameters()) == 0


@pytest.mark.parametrize("attn_type", ALL_ATTN_TYPES)
def test_attention_input_validation(attn_type):
    """Verify strict input shape contract: asserts (C=2048, H=4, W=4)."""
    module = build_attention_module(AttentionConfig(type=attn_type))

    # 3D input
    with pytest.raises(ValueError, match="Expected 4D or 5D input tensor"):
        module(torch.randn(2, 2048, 4))

    # Wrong channels (e.g. 1024 != 2048)
    with pytest.raises(ValueError, match="Expected input with \\(C=2048, H=4, W=4\\)"):
        module(torch.randn(2, 1024, 4, 4))

    # Wrong height / width (e.g. 7x7 != 4x4)
    with pytest.raises(ValueError, match="Expected input with \\(C=2048, H=4, W=4\\)"):
        module(torch.randn(2, 2048, 7, 7))

    # 5D wrong spatial dims
    with pytest.raises(ValueError, match="Expected input with \\(C=2048, H=4, W=4\\)"):
        module(torch.randn(2, 4, 2048, 5, 5))


def test_soft_attention_uniform_logit_sanity():
    """Verify that uniform logits collapse SoftAttention to exact Global Average Pooling."""
    model = SoftAttention(in_channels=2048)
    # Zero out projection weights and bias so logits are identically 0 everywhere
    nn.init.zeros_(model.projector.weight)
    nn.init.zeros_(model.projector.bias)
    model.eval()

    x = torch.randn(2, 2048, 4, 4)
    with torch.no_grad():
        out, maps = model(x, return_attention_maps=True)

    expected = nn.AdaptiveAvgPool2d((1, 1))(x).flatten(1)
    assert torch.allclose(out, expected, atol=1e-6)

    # Weights must be exactly 1/16 = 0.0625 everywhere
    expected_weights = torch.full((2, 1, 4, 4), 1.0 / 16.0)
    assert torch.allclose(maps["soft_map"], expected_weights, atol=1e-6)


def test_soft_attention_softmax_contract():
    """Verify that soft attention weights sum to 1.0 and are all non-negative."""
    model = SoftAttention(in_channels=2048)
    model.eval()

    x = torch.randn(4, 2048, 4, 4)
    with torch.no_grad():
        _, maps = model(x, return_attention_maps=True)

    soft_map = maps["soft_map"]
    # Sum over H, W dimensions must equal 1.0 for every sample
    sums = soft_map.sum(dim=(-1, -2))
    assert torch.allclose(sums, torch.ones(4, 1), atol=1e-6)
    assert (soft_map >= 0.0).all()


@pytest.mark.parametrize("attn_type", ALL_ATTN_TYPES)
def test_attention_dtype_device_preservation(attn_type):
    """Verify that output tensor preserves input dtype and device."""
    module = build_attention_module(AttentionConfig(type=attn_type))
    module.eval()

    # Float32
    x_f32 = torch.randn(2, 2048, 4, 4, dtype=torch.float32)
    with torch.no_grad():
        out_f32 = module(x_f32)
    assert out_f32.dtype == torch.float32
    assert out_f32.device == x_f32.device

    # Float64
    module_f64 = build_attention_module(AttentionConfig(type=attn_type)).double()
    x_f64 = torch.randn(2, 2048, 4, 4, dtype=torch.float64)
    with torch.no_grad():
        out_f64 = module_f64(x_f64)
    assert out_f64.dtype == torch.float64
    assert out_f64.device == x_f64.device


@pytest.mark.parametrize("attn_type", ALL_ATTN_TYPES)
def test_attention_output_finite(attn_type):
    """Verify that outputs and attention maps contain no NaNs or Infs."""
    module = build_attention_module(AttentionConfig(type=attn_type))
    module.eval()

    x = torch.randn(2, 2048, 4, 4)
    with torch.no_grad():
        out, maps = module(x, return_attention_maps=True)

    assert torch.isfinite(out).all()
    for k, v in maps.items():
        assert torch.isfinite(v).all(), f"Attention map '{k}' contains non-finite values"


@pytest.mark.parametrize("attn_type", ALL_ATTN_TYPES)
def test_attention_sequence_equivalence(attn_type):
    """Verify that 5D sequence processing matches 4D flattened frame processing in eval() mode.

    Demonstrates strict per-frame invariant (zero temporal mixing).
    """
    torch.manual_seed(42)
    module = build_attention_module(AttentionConfig(type=attn_type))
    module.eval()

    b, t = 2, 4
    x_5d = torch.randn(b, t, 2048, 4, 4)
    x_4d = x_5d.view(b * t, 2048, 4, 4)

    with torch.no_grad():
        out_5d, maps_5d = module(x_5d, return_attention_maps=True)
        out_4d, maps_4d = module(x_4d, return_attention_maps=True)

    assert out_5d.shape == (b, t, 2048)
    assert torch.allclose(out_5d, out_4d.view(b, t, 2048), atol=1e-5)

    for k in maps_5d:
        assert torch.allclose(maps_5d[k], maps_4d[k].view(b, t, *maps_4d[k].shape[1:]), atol=1e-5)


def test_attention_map_contracts():
    """Verify structured dictionary outputs for all attention modes."""
    x = torch.randn(2, 2048, 4, 4)

    # Soft Attention
    soft = SoftAttention(2048)
    _, m_soft = soft(x, return_attention_maps=True)
    assert "soft_map" in m_soft
    assert m_soft["soft_map"].shape == (2, 1, 4, 4)

    # CBAM Attention
    cbam = CBAMAttention(2048)
    _, m_cbam = cbam(x, return_attention_maps=True)
    assert "channel_map" in m_cbam and "spatial_map" in m_cbam
    assert m_cbam["channel_map"].shape == (2, 2048, 1, 1)
    assert m_cbam["spatial_map"].shape == (2, 1, 4, 4)

    # Channel Attention
    cam = ChannelAttention(2048)
    _, m_cam = cam(x, return_attention_maps=True)
    assert "channel_map" in m_cam
    assert m_cam["channel_map"].shape == (2, 2048, 1, 1)

    # Spatial Attention
    sam = SpatialAttention(2048)
    _, m_sam = sam(x, return_attention_maps=True)
    assert "spatial_map" in m_sam
    assert m_sam["spatial_map"].shape == (2, 1, 4, 4)

    # Identity Pooling
    identity = IdentityPooling(2048)
    _, m_id = identity(x, return_attention_maps=True)
    assert m_id == {}


def test_identity_pooling_equivalence():
    """Verify IdentityPooling is mathematically identical to Global Average Pooling."""
    identity = IdentityPooling(2048)
    x = torch.randn(3, 2048, 4, 4)
    out = identity(x)
    expected = nn.AdaptiveAvgPool2d((1, 1))(x).flatten(1)
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("attn_type", ["soft", "cbam", "channel", "spatial"])
def test_attention_gradient_flow(attn_type):
    """Verify that gradients flow back to all trainable parameters and input tensor."""
    torch.manual_seed(42)
    module = build_attention_module(AttentionConfig(type=attn_type))
    module.train()

    x = torch.randn(2, 2048, 4, 4, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0

    for name, p in module.named_parameters():
        assert p.grad is not None, f"Parameter {name} has None gradient"
        assert p.grad.abs().sum() > 0, f"Parameter {name} has zero gradient"


def test_attention_config_cross_validation():
    """Verify cross-validation between attention.in_channels and backbone.output_dim."""
    # Mismatched channels should raise ValueError in validate()
    config = AppConfig(
        model=ModelConfig(
            backbone=BackboneConfig(weights=None),
            attention=AttentionConfig(type="soft", in_channels=2048),
        )
    )
    # Valid config validates fine
    config.validate()

    # Manually mismatch
    config.model.attention.in_channels = 1024
    with pytest.raises(ValueError, match="model.attention.in_channels .* must match"):
        config.validate()


def test_backbone_to_attention_composition():
    """Verify end-to-end integration: ResNeXt-50 -> SoftAttention produces (B, T, 2048)."""
    backbone = ResNeXt50Backbone(BackboneConfig(weights=None))
    attention = SoftAttention(in_channels=2048)
    backbone.eval()
    attention.eval()

    frames = torch.randn(1, 16, 3, 112, 112)
    with torch.no_grad():
        _, spatial_maps = backbone(frames, return_spatial_map=True)
        assert spatial_maps.shape == (1, 16, 2048, 4, 4)

        context_vector, maps = attention(spatial_maps, return_attention_maps=True)

    assert context_vector.shape == (1, 16, 2048)
    assert maps["soft_map"].shape == (1, 16, 1, 4, 4)


def test_attention_cuda_fp16_amp():
    """Verify attention forward and backward passes under CUDA FP16 AMP."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for FP16 AMP attention test")

    module = SoftAttention(2048).cuda()
    module.train()

    x = torch.randn(2, 4, 2048, 4, 4, device="cuda", requires_grad=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = module(x)
        loss = out.sum()

    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(out).all()
