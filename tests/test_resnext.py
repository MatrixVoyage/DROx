"""Unit and integration tests for ResNeXt-50 spatial backbone."""

import io
import socket
import urllib.error
import pytest
import torch
import torch.nn as nn

from src.models.backbones.resnext import ResNeXt50Backbone
from src.utils.config import BackboneConfig


def test_resnext_spatial_4x4_contract():
    """Verify that (B, 3, 112, 112) -> (B, 2048) and (B, 2048, 4, 4) under weights=None."""
    config = BackboneConfig(weights=None)
    model = ResNeXt50Backbone(config=config)
    model.eval()

    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        pooled = model(x)
        pooled_ret, spatial = model(x, return_spatial_map=True)

    assert pooled.shape == (2, 2048)
    assert pooled_ret.shape == (2, 2048)
    assert spatial.shape == (2, 2048, 4, 4)
    assert torch.allclose(pooled, pooled_ret)


def test_resnext_batch_size_1():
    """Verify inference on a single frame in eval() mode."""
    config = BackboneConfig(weights=None)
    model = ResNeXt50Backbone(config=config)
    model.eval()

    x = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        pooled = model(x)

    assert pooled.shape == (1, 2048)


def test_resnext_sequence_equivalence():
    """Verify that 5D sequence (B, T, 3, 112, 112) matches 4D flattened (B*T, 3, 112, 112)."""
    torch.manual_seed(42)
    config = BackboneConfig(weights=None)
    model = ResNeXt50Backbone(config=config)
    model.eval()

    b, t = 2, 4
    x_5d = torch.randn(b, t, 3, 112, 112)
    x_4d = x_5d.view(b * t, 3, 112, 112)

    with torch.no_grad():
        pooled_5d, spatial_5d = model(x_5d, return_spatial_map=True)
        pooled_4d, spatial_4d = model(x_4d, return_spatial_map=True)

    assert pooled_5d.shape == (b, t, 2048)
    assert spatial_5d.shape == (b, t, 2048, 4, 4)
    assert torch.allclose(pooled_5d, pooled_4d.view(b, t, 2048), atol=1e-5)
    assert torch.allclose(spatial_5d, spatial_4d.view(b, t, 2048, 4, 4), atol=1e-5)


@pytest.mark.parametrize(
    "strategy,frozen_stages",
    [
        ("none", []),
        ("stem", ["stem"]),
        ("layer1", ["stem", "layer1"]),
        ("layer2", ["stem", "layer1", "layer2"]),
        ("layer3", ["stem", "layer1", "layer2", "layer3"]),
        ("all", ["stem", "layer1", "layer2", "layer3", "layer4"]),
    ],
)
def test_resnext_cumulative_freezing(strategy, frozen_stages):
    """Verify that cumulative freezing strictly freezes expected stages and leaves remaining stages trainable."""
    config = BackboneConfig(weights=None, freeze_strategy=strategy)
    model = ResNeXt50Backbone(config=config)

    stage_map = {
        "stem": model.stem,
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
    }

    for name, stage in stage_map.items():
        expected_requires_grad = name not in frozen_stages
        for p in stage.parameters():
            assert p.requires_grad == expected_requires_grad, (
                f"Stage {name} param requires_grad={p.requires_grad}, expected {expected_requires_grad} "
                f"under strategy '{strategy}'"
            )


def test_resnext_freeze_all_zero_trainable():
    """Verify that freeze_strategy='all' results in zero trainable parameters."""
    config = BackboneConfig(weights=None, freeze_strategy="all")
    model = ResNeXt50Backbone(config=config)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_count == 0


def test_resnext_decoupled_batchnorm_freezing():
    """Verify decoupled BatchNorm freezing: frozen stages stay in eval() mode while others train."""
    # Test case 1: freeze_batchnorm=True
    config_frozen_bn = BackboneConfig(weights=None, freeze_strategy="layer2", freeze_batchnorm=True)
    model_frozen = ResNeXt50Backbone(config=config_frozen_bn)
    model_frozen.train()

    # Stem, layer1, layer2 BNs must be eval
    for stage in [model_frozen.stem, model_frozen.layer1, model_frozen.layer2]:
        for m in stage.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                assert not m.training, "Frozen stage BatchNorm should remain in eval() mode"

    # Layer3, layer4 BNs must be training
    for stage in [model_frozen.layer3, model_frozen.layer4]:
        for m in stage.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                assert m.training, "Unfrozen stage BatchNorm should be in training mode"

    # Test case 2: freeze_batchnorm=False
    config_active_bn = BackboneConfig(weights=None, freeze_strategy="layer2", freeze_batchnorm=False)
    model_active = ResNeXt50Backbone(config=config_active_bn)
    model_active.train()

    # When freeze_batchnorm=False, all BNs enter train mode when model.train() is called
    for stage in [model_active.stem, model_active.layer1, model_active.layer2]:
        for m in stage.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                assert m.training, "BatchNorm should enter training mode when freeze_batchnorm=False"


def test_resnext_gradient_flow_granular():
    """Verify that backward pass computes gradients exclusively for unfrozen stages."""
    torch.manual_seed(42)
    config = BackboneConfig(weights=None, freeze_strategy="layer2", freeze_batchnorm=True)
    model = ResNeXt50Backbone(config=config)
    model.train()

    x = torch.randn(2, 3, 112, 112)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # stem, layer1, layer2 must have NO gradients
    for stage in [model.stem, model.layer1, model.layer2]:
        for p in stage.parameters():
            assert p.grad is None, "Frozen parameter unexpectedly received gradients"

    # layer3, layer4 must have non-zero gradients
    layer3_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.layer3.parameters())
    layer4_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.layer4.parameters())
    assert layer3_has_grad, "Layer3 should receive gradients during backward pass"
    assert layer4_has_grad, "Layer4 should receive gradients during backward pass"


def test_resnext_state_dict_compatibility():
    """Verify that model state_dict can be saved, reloaded with strict=True, and produces identical outputs."""
    torch.manual_seed(42)
    config = BackboneConfig(weights=None, freeze_strategy="layer1")
    model1 = ResNeXt50Backbone(config=config)
    model1.eval()

    buffer = io.BytesIO()
    torch.save(model1.state_dict(), buffer)
    buffer.seek(0)

    model2 = ResNeXt50Backbone(config=config)
    model2.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    model2.eval()

    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    assert torch.allclose(out1, out2)


def test_resnext_invalid_input_dim():
    """Verify that invalid tensor dimensions and configurations raise appropriate exceptions."""
    config = BackboneConfig(weights=None)
    model = ResNeXt50Backbone(config=config)

    # 3D input
    with pytest.raises(ValueError, match="Expected 4D or 5D input tensor"):
        model(torch.randn(3, 112, 112))

    # Wrong spatial size (e.g. 224x224 instead of 112x112)
    with pytest.raises(ValueError, match="Expected frame shape"):
        model(torch.randn(1, 3, 224, 224))

    # Wrong channel count (e.g. 1 instead of 3)
    with pytest.raises(ValueError, match="Expected frame shape"):
        model(torch.randn(1, 1, 112, 112))

    # 5D with wrong channel
    with pytest.raises(ValueError, match="Expected frame shape"):
        model(torch.randn(1, 4, 1, 112, 112))

    # Unsupported weights string in config
    with pytest.raises(ValueError, match="weights must be 'IMAGENET1K_V1' or None"):
        BackboneConfig(weights="UNSUPPORTED_WEIGHTS")


def test_resnext_spatial_contract_violation_diagnostic():
    """Verify that RuntimeError is raised with exact diagnostic when spatial output violates contract."""
    # Create a config with non-matching spatial_dim expectation
    config = BackboneConfig(weights=None)
    model = ResNeXt50Backbone(config=config)
    # Deliberately modify config expectation to simulate architecture drift / contract check
    model.config.spatial_dim = 7

    x = torch.randn(1, 3, 112, 112)
    with pytest.raises(RuntimeError) as exc_info:
        model(x)

    msg = str(exc_info.value)
    assert "ResNeXt-50 spatial shape contract violated" in msg
    assert "expected (7, 7)" in msg
    assert "got (4, 4)" in msg


def test_resnext_provenance_metadata():
    """Verify that provenance metadata correctly documents model configuration."""
    config = BackboneConfig(weights=None, freeze_strategy="layer2", freeze_batchnorm=True)
    model = ResNeXt50Backbone(config=config)
    prov = model.get_provenance()

    assert prov["backbone_name"] == "resnext50_32x4d"
    assert prov["weights_name"] is None
    assert prov["pretrained"] is False
    assert prov["freeze_strategy"] == "layer2"
    assert prov["freeze_batchnorm"] is True
    assert prov["input_size"] == (112, 112)
    assert prov["output_dim"] == 2048
    assert prov["spatial_dim"] == 4
    assert "torchvision_version" in prov


def test_resnext_fp16_autocast():
    """Verify forward pass under FP16 autocast on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for FP16 autocast test")

    config = BackboneConfig(weights=None)
    model = ResNeXt50Backbone(config=config).cuda()
    model.eval()

    x = torch.randn(2, 3, 112, 112, device="cuda")
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(x)

    assert out.shape == (2, 2048)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_resnext_pretrained_weights_offline_safe():
    """Verify pretrained weights load successfully and differ from uninitialized weights.

    Skips exclusively on network/connection errors without masking bugs.
    """
    network_errors = (urllib.error.URLError, socket.timeout, ConnectionError, OSError)
    try:
        import requests
        network_errors = network_errors + (requests.exceptions.RequestException,)
    except ImportError:
        pass

    try:
        config = BackboneConfig(weights="IMAGENET1K_V1")
        pretrained_model = ResNeXt50Backbone(config=config)
    except network_errors as e:
        pytest.skip(f"Network unavailable to download/verify IMAGENET1K_V1 weights: {e}")

    # Compare conv1 weight to random weights
    uninit_model = ResNeXt50Backbone(config=BackboneConfig(weights=None))
    p_weight = pretrained_model.stem[0].weight.data
    u_weight = uninit_model.stem[0].weight.data

    assert not torch.allclose(p_weight, u_weight), (
        "Pretrained weights should not match uninitialized weights"
    )
    assert pretrained_model.config.pretrained is True
    assert pretrained_model.config.weights == "IMAGENET1K_V1"
