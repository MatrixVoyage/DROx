"""Unit and Integration Tests for Milestone 13.3.

Tests:
1. BlockDCTConv2d: Orthogonal 2D-DCT basis filter bank and shapes.
2. BayarConv2d: Constrained high-pass filter and zero-response on constant images.
3. FrequencyResidualBranch: 5D/4D forward pass, shapes, and gradients.
4. LocalTextureROIBranch: Multi-region ROI patch extraction and Sobel gradient texture.
5. FamilyInvariantDeepfakeDetector: Multi-domain fusion, heads, freeze policies, and backprop.
6. LOFO Manifests: Zero group leakage and proper exclusion across all 4 folds.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import pandas as pd

from src.models.branches.frequency_residual import (
    BlockDCTConv2d,
    BayarConv2d,
    FrequencyResidualBranch,
    build_dct_basis_filters,
)
from src.models.branches.local_texture import LocalTextureROIBranch
from src.models.family_invariant_detector import (
    FamilyInvariantDeepfakeDetector,
    FamilyInvariantOutput,
)
from src.utils.config import ModelConfig


class TestFrequencyResidualBranch:
    """Tests for Block DCT and Bayar constrained frequency residual components."""

    def test_dct_basis_orthogonality(self):
        """Verify that the generated 8x8 DCT basis filters are mutually orthogonal."""
        filters = build_dct_basis_filters(block_size=8)  # (64, 1, 8, 8)
        assert filters.shape == (64, 1, 8, 8)

        # Flatten each 8x8 basis kernel to a 64-D vector: (64, 64)
        basis_matrix = filters.view(64, 64)
        gram = torch.matmul(basis_matrix, basis_matrix.t())
        identity = torch.eye(64)

        # Check orthogonality: Gram matrix should equal Identity within float tolerance
        max_diff = (gram - identity).abs().max().item()
        assert max_diff < 1e-5, f"DCT basis is not orthogonal, max deviation: {max_diff}"

    def test_block_dct_conv2d_forward(self):
        """Verify BlockDCTConv2d spatial downsampling and channel transformation."""
        conv = BlockDCTConv2d(block_size=8)

        # Test single-channel input
        x_1c = torch.randn(2, 1, 112, 112)
        out_1c = conv(x_1c)
        assert out_1c.shape == (2, 64, 14, 14)

        # Test 3-channel RGB input (automatically converts to luminance)
        x_3c = torch.randn(2, 3, 112, 112)
        out_3c = conv(x_3c)
        assert out_3c.shape == (2, 64, 14, 14)

    def test_bayar_conv2d_constraint(self):
        """Verify Bayar high-pass constraint: center=-1, surround sum=+1, zero response on flat skin."""
        bayar = BayarConv2d(in_channels=1, out_channels=4, kernel_size=5)

        # Test constant image: constant skin tone should produce strictly 0.0 response
        const_img = torch.full((2, 1, 112, 112), 0.75, dtype=torch.float32)
        out = bayar(const_img)

        # Center interior pixels should be exactly 0.0 (excluding boundary padding margin)
        interior = out[:, :, 5:107, 5:107]
        max_val = interior.abs().max().item()
        assert max_val < 1e-5, f"Bayar filter did not suppress constant image: max={max_val}"

    def test_bayar_conv2d_gradients(self):
        """Verify learnable Bayar parameters receive valid gradients."""
        bayar = BayarConv2d(in_channels=1, out_channels=4, kernel_size=5)
        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        out = bayar(x)
        loss = out.sum()
        loss.backward()

        assert bayar.raw_weights.grad is not None
        assert not torch.isnan(bayar.raw_weights.grad).any()
        assert not torch.isinf(bayar.raw_weights.grad).any()

    def test_frequency_residual_branch_forward(self):
        """Verify full FrequencyResidualBranch forward pass and output dimensions."""
        branch = FrequencyResidualBranch(
            block_size=8,
            bayar_channels=8,
            hidden_dim=64,
            out_features=64,
            dropout=0.1,
        )

        # 5D video input: (B=2, T=4, C=3, H=112, W=112)
        x_5d = torch.randn(2, 4, 3, 112, 112)
        z_freq, spatial_map = branch(x_5d)

        assert z_freq.shape == (2, 64)
        assert spatial_map.shape == (8, 64, 14, 14)

        # 4D frame input: (B=4, C=3, H=112, W=112)
        x_4d = torch.randn(4, 3, 112, 112)
        z_freq_4d, spatial_map_4d = branch(x_4d)

        assert z_freq_4d.shape == (4, 64)
        assert spatial_map_4d.shape == (4, 64, 14, 14)


class TestLocalTextureROIBranch:
    """Tests for Multi-Region Semantic ROIs and spatial gradient texture branch."""

    def test_roi_extraction(self):
        """Verify extraction and resizing of mouth, eye, and perimeter patches."""
        branch = LocalTextureROIBranch(in_channels=3, roi_feature_dim=32, out_features=64)
        x = torch.randn(2, 3, 112, 112)

        mouth, eyes, boundary = branch.extract_rois(x)
        assert mouth.shape == (2, 3, 32, 32)
        assert eyes.shape == (2, 3, 32, 32)
        assert boundary.shape == (2, 3, 32, 32)

    def test_gradient_magnitude(self):
        """Verify Sobel spatial gradient magnitude output shape and values."""
        branch = LocalTextureROIBranch(in_channels=3, roi_feature_dim=32, out_features=64)
        x = torch.randn(2, 3, 112, 112)

        mag = branch.compute_gradient_magnitude(x)
        assert mag.shape == (2, 1, 112, 112)
        assert (mag >= 0.0).all(), "Gradient magnitude must be non-negative"

    def test_local_texture_branch_forward_and_backward(self):
        """Verify forward and backward pass through LocalTextureROIBranch."""
        branch = LocalTextureROIBranch(in_channels=3, roi_feature_dim=32, out_features=64)

        # 5D video input: (B=2, T=4, C=3, 112, 112)
        x = torch.randn(2, 4, 3, 112, 112, requires_grad=True)
        z_tex = branch(x)

        assert z_tex.shape == (2, 64)
        loss = z_tex.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestFamilyInvariantDeepfakeDetector:
    """Integration tests for the complete Family-Invariant Detector."""

    @pytest.fixture
    def small_detector(self):
        """Instantiate a lightweight detector for CPU testing."""
        cfg = ModelConfig()
        cfg.backbone.name = "efficientnet_b0"
        cfg.backbone.output_dim = 1280
        cfg.backbone.freeze_strategy = "early"
        cfg.attention.in_channels = 1280
        cfg.temporal.type = "transformer"
        cfg.temporal.input_dim = 1280
        cfg.temporal.hidden_dim = 1280
        cfg.temporal.num_layers = 1
        cfg.classifier.in_features = 1280
        cfg.classifier.hidden_dim = 64

        model = FamilyInvariantDeepfakeDetector(
            config=cfg,
            num_families=5,
            aux_hidden_dim=64,
            freq_dim=32,
            texture_dim=32,
            fusion_dim=64,
            enable_mid_head=False,  # test without hook first
        )
        return model

    def test_forward_output_structure(self, small_detector):
        """Verify FamilyInvariantOutput dataclass structure and tensor dimensions."""
        model = small_detector
        model.eval()

        # Batch of 2 videos with 4 frames each
        x = torch.randn(2, 4, 3, 112, 112)
        with torch.no_grad():
            out = model(x, return_dict=True)

        assert isinstance(out, FamilyInvariantOutput)
        assert out.primary_logits.shape == (2, 1)
        assert out.aux_family_logits.shape == (2, 5)
        assert out.freq_logits.shape == (2, 1)
        assert out.texture_logits.shape == (2, 1)
        assert out.temporal_features.shape == (2, 1280)
        assert out.freq_features.shape == (2, 32)
        assert out.texture_features.shape == (2, 32)

        # Probabilities bounded in [0, 1]
        assert (out.p_fused >= 0.0).all() and (out.p_fused <= 1.0).all()
        assert (out.p_composite >= 0.0).all() and (out.p_composite <= 1.0).all()
        assert (out.p_aux_fake >= 0.0).all() and (out.p_aux_fake <= 1.0).all()

    def test_unfreeze_policies(self, small_detector):
        """Verify freeze and unfreeze policies."""
        model = small_detector

        # Test unfreeze_branches_only
        model.unfreeze_branches_only()
        assert not any(p.requires_grad for p in model.base_detector.backbone.parameters())
        assert not any(p.requires_grad for p in model.base_detector.temporal.parameters())
        assert all(p.requires_grad for p in model.freq_branch.parameters())
        assert all(p.requires_grad for p in model.texture_branch.parameters())
        assert all(p.requires_grad for p in model.fusion_head.parameters())

        # Test unfreeze_temporal_and_heads
        model.unfreeze_temporal_and_heads()
        assert not any(p.requires_grad for p in model.base_detector.backbone.parameters())
        assert all(p.requires_grad for p in model.base_detector.temporal.parameters())
        assert all(p.requires_grad for p in model.fusion_head.parameters())

    def test_training_step(self, small_detector):
        """Verify backpropagation with multi-task loss."""
        model = small_detector
        model.train()
        model.unfreeze_temporal_and_heads()

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )
        x = torch.randn(2, 4, 3, 112, 112)
        y_binary = torch.tensor([[1.0], [0.0]])
        y_family = torch.tensor([1, 0])

        out = model(x, return_dict=True)

        loss_prim = nn.functional.binary_cross_entropy_with_logits(out.primary_logits, y_binary)
        loss_freq = nn.functional.binary_cross_entropy_with_logits(out.freq_logits, y_binary)
        loss_tex = nn.functional.binary_cross_entropy_with_logits(out.texture_logits, y_binary)
        loss_aux = nn.functional.cross_entropy(out.aux_family_logits, y_family)

        total_loss = loss_prim + 0.5 * loss_aux + 0.2 * loss_freq + 0.2 * loss_tex

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        assert not torch.isnan(total_loss)
        assert total_loss.item() > 0.0


class TestLOFOManifests:
    """Verifies that all 4 LOFO manifests adhere to strict leakage and hold-out requirements."""

    @pytest.mark.parametrize("fold_key,held_out", [
        ("f2f", "Face2Face"),
        ("nt", "NeuralTextures"),
        ("fs", "FaceSwap"),
        ("df", "Deepfakes"),
    ])
    def test_lofo_manifest_integrity(self, fold_key: str, held_out: str):
        """Check zero group leakage and proper exclusion in manifests/lofo_{fold_key}.csv."""
        manifest_path = Path("manifests") / f"lofo_{fold_key}.csv"
        assert manifest_path.exists(), f"Manifest {manifest_path} must exist"

        df = pd.read_csv(manifest_path)
        assert len(df) > 0

        # Check zero group leakage between splits
        for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
            g1 = set(df[df["split"] == s1]["group_id"].unique())
            g2 = set(df[df["split"] == s2]["group_id"].unique())
            overlap = g1 & g2
            assert len(overlap) == 0, f"[{fold_key}] Leakage between {s1} and {s2}: {overlap}"

        # Check that held-out manipulation is absent from train and val
        train_val_manips = df[df["split"].isin(["train", "val"])]["manipulation_type"].unique()
        assert held_out not in train_val_manips, (
            f"[{fold_key}] Held-out manipulation '{held_out}' must NOT be in train or val!"
        )

        # Check that held-out manipulation is present in test
        test_manips = df[df["split"] == "test"]["manipulation_type"].unique()
        assert held_out in test_manips, (
            f"[{fold_key}] Held-out manipulation '{held_out}' must be present in test split!"
        )
