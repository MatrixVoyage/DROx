"""Unit and physical artifact tests for Model Export (Gate 2)."""

from pathlib import Path
import pytest
import torch

from src.inference.export import (
    export_checkpoint,
    export_onnx,
    export_torchscript,
    verify_export,
)
from src.models.detector import DeepfakeDetector
from src.utils.config import load_config

CHECKPOINT_PATH = Path("checkpoints/ffpp_c23_baseline_soft_lstm/best.pth")


@pytest.fixture
def mock_detector():
    """Create lightweight model for fast unit testing."""
    config = load_config("configs/default.yaml")
    model = DeepfakeDetector(config.model)
    model.eval()
    return model


def test_torchscript_export_and_parity(mock_detector, tmp_path):
    """Verify TorchScript tracing and numerical parity."""
    out_file = tmp_path / "model.pt"
    x = torch.randn(1, 16, 3, 112, 112, dtype=torch.float32)

    export_torchscript(mock_detector, out_file, example_input=x)
    assert out_file.exists()

    res = verify_export(mock_detector, out_file, format_type="torchscript", example_input=x, atol=1e-5)
    assert res["parity_verified"] is True
    assert res["max_abs_diff"] < 1e-5


def test_onnx_export_and_parity(mock_detector, tmp_path):
    """Verify ONNX export with fixed T=16 and ONNX Runtime parity."""
    out_file = tmp_path / "model.onnx"
    x = torch.randn(1, 16, 3, 112, 112, dtype=torch.float32)

    export_onnx(mock_detector, out_file, example_input=x, dynamic_batch=True)
    assert out_file.exists()

    res = verify_export(mock_detector, out_file, format_type="onnx", example_input=x, atol=1e-4)
    assert res["parity_verified"] is True
    assert res["max_abs_diff"] < 1e-4

    # Test dynamic batch size B=2 with fixed T=16 in ONNX Runtime
    import onnxruntime as ort
    session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
    x_b2 = torch.randn(2, 16, 3, 112, 112, dtype=torch.float32)
    with torch.no_grad():
        eager_b2 = mock_detector(x_b2).numpy()
    onnx_b2 = session.run(["logits"], {"video_frames": x_b2.numpy()})[0]
    assert onnx_b2.shape == (2, 1)
    diff_b2 = float(abs(eager_b2 - onnx_b2).max())
    assert diff_b2 < 1e-4


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="best.pth not found")
def test_gate2_export_physical_best_checkpoint(tmp_path):
    """Gate 2 physical verification: Export real frozen best.pth and verify both TorchScript and ONNX parity."""
    export_dir = tmp_path / "exports"
    results = export_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        output_dir=export_dir,
        export_format="all",
        device="cpu",
    )

    assert "torchscript" in results["exports"]
    assert "onnx" in results["exports"]

    ts_res = results["exports"]["torchscript"]
    assert ts_res["parity_verified"] is True
    assert ts_res["max_abs_diff"] < 1e-5

    onnx_res = results["exports"]["onnx"]
    assert onnx_res["parity_verified"] is True
    assert onnx_res["max_abs_diff"] < 1e-4
