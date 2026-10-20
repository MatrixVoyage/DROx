"""Gate 2 Standalone Export Verification Runner.

Exports frozen best.pth to:
1. TorchScript: checkpoints/export/deepfake_detector_baseline.pt
2. ONNX: checkpoints/export/deepfake_detector_baseline.onnx
Validates:
- Eager PyTorch vs TorchScript max absolute difference (atol=1e-5)
- Eager PyTorch vs ONNX Runtime max absolute difference (atol=1e-4)
- Dynamic batch dimension (B=1, B=2) with fixed T=16 sequence length.
Saves audit evidence to reports/gate2_export_audit.json.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import onnxruntime as ort
from src.inference.export import export_checkpoint, verify_export
from src.models.detector import DeepfakeDetector
from src.utils.config import load_config, _dict_to_config


def main():
    print("=" * 70)
    print("GATE 2 — MODEL EXPORT PHYSICAL AUDIT")
    print("=" * 70)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "ffpp_c23_baseline_soft_lstm" / "best.pth"
    export_dir = PROJECT_ROOT / "checkpoints" / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"
    print(f"Loading frozen checkpoint: {ckpt_path.name}")

    # Load in-memory model (frozen checkpoint remains untouched)
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = _dict_to_config(ckpt_data["config"]) if "config" in ckpt_data else load_config("configs/default.yaml")
    config.device = "cpu"
    model = DeepfakeDetector(config.model)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    # Target export paths
    ts_path = export_dir / "deepfake_detector_baseline.pt"
    onnx_path = export_dir / "deepfake_detector_baseline.onnx"

    # 1. TorchScript Export & Parity
    print(f"\n1. Exporting TorchScript to: {ts_path}")
    from src.inference.export import export_torchscript, export_onnx
    x1 = torch.randn(1, 16, 3, 112, 112, dtype=torch.float32)
    export_torchscript(model, ts_path, example_input=x1)
    ts_res = verify_export(model, ts_path, format_type="torchscript", example_input=x1, atol=1e-5)
    print(f"TorchScript Max Difference: {ts_res['max_abs_diff']:.10f}")
    print(f"TorchScript Parity:         {'PASS' if ts_res['parity_verified'] else 'FAIL'}")

    # 2. ONNX Export & Parity (Fixed T=16, dynamic batch B)
    print(f"\n2. Exporting ONNX (fixed T=16, dynamic batch) to: {onnx_path}")
    export_onnx(model, onnx_path, example_input=x1, dynamic_batch=True)
    onnx_res = verify_export(model, onnx_path, format_type="onnx", example_input=x1, atol=1e-4)
    print(f"ONNX Runtime Max Diff (B=1): {onnx_res['max_abs_diff']:.10f}")
    print(f"ONNX Runtime Parity (B=1):   {'PASS' if onnx_res['parity_verified'] else 'FAIL'}")

    # 3. Test ONNX Dynamic Batch (B=2, T=16)
    print("\n3. Testing ONNX Runtime Dynamic Batch (B=2, T=16)...")
    x2 = torch.randn(2, 16, 3, 112, 112, dtype=torch.float32)
    with torch.no_grad():
        eager_b2 = model(x2).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_b2 = session.run(["logits"], {"video_frames": x2.numpy()})[0]
    b2_diff = float(abs(eager_b2 - onnx_b2).max())
    b2_passed = b2_diff < 1e-4
    print(f"ONNX Runtime Max Diff (B=2): {b2_diff:.10f}")
    print(f"ONNX Dynamic Batch Parity:   {'PASS' if b2_passed else 'FAIL'}")

    gate2_passed = ts_res["parity_verified"] and onnx_res["parity_verified"] and b2_passed

    audit_payload = {
        "gate": "Gate 2 — Model Export",
        "status": "PASS" if gate2_passed else "FAILED",
        "checkpoint": str(ckpt_path),
        "artifacts": {
            "torchscript": {
                "path": str(ts_path.resolve()),
                "file_size_bytes": ts_path.stat().st_size if ts_path.exists() else 0,
                "max_abs_diff": ts_res["max_abs_diff"],
                "parity_verified": ts_res["parity_verified"],
            },
            "onnx": {
                "path": str(onnx_path.resolve()),
                "file_size_bytes": onnx_path.stat().st_size if onnx_path.exists() else 0,
                "max_abs_diff_b1": onnx_res["max_abs_diff"],
                "max_abs_diff_b2": b2_diff,
                "fixed_sequence_length": 16,
                "dynamic_batch": True,
                "parity_verified": onnx_res["parity_verified"] and b2_passed,
            }
        },
        "parity_verified": gate2_passed,
    }

    out_json = PROJECT_ROOT / "reports" / "gate2_export_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    print(f"\nGate 2 Audit Artifact saved to: {out_json}")
    print("=" * 70)
    print(f"GATE 2 AUDIT RESULT: {'PASS' if gate2_passed else 'FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
