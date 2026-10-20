"""Model export utilities for TorchScript and ONNX serving formats."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from src.models.detector import DeepfakeDetector
from src.utils.config import AppConfig, _dict_to_config, load_config
from src.utils.logger import setup_logger

logger = setup_logger("model_exporter")


def export_torchscript(
    model: nn.Module,
    output_path: Union[str, Path],
    example_input: Optional[torch.Tensor] = None,
) -> Path:
    """Trace PyTorch model into an optimized TorchScript artifact.

    Args:
        model: DeepfakeDetector model instance.
        output_path: Destination path (.pt).
        example_input: Optional sample input tensor. Defaults to (1, 16, 3, 112, 112) on CPU.

    Returns:
        Path to the saved TorchScript artifact.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    if example_input is None:
        device = next(model.parameters()).device
        example_input = torch.randn(1, 16, 3, 112, 112, dtype=torch.float32, device=device)

    logger.info(f"Tracing TorchScript model with input shape {tuple(example_input.shape)}...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(str(out_p))

    if was_training:
        model.train()

    logger.info(f"Successfully saved TorchScript model to: {out_p}")
    return out_p


def export_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    example_input: Optional[torch.Tensor] = None,
    dynamic_batch: bool = True,
    opset_version: int = 17,
) -> Path:
    """Export PyTorch model to ONNX graph format.

    In accordance with research specifications:
    - Sequence dimension remains fixed at T=16 (as designed and trained).
    - Batch dimension is dynamic (B >= 1) when dynamic_batch is True.

    Args:
        model: DeepfakeDetector model instance.
        output_path: Destination path (.onnx).
        example_input: Optional sample input tensor. Defaults to (1, 16, 3, 112, 112) on CPU.
        dynamic_batch: Whether to allow dynamic batch dimension.
        opset_version: Target ONNX opset version (default: 17).

    Returns:
        Path to the saved ONNX artifact.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    if example_input is None:
        device = next(model.parameters()).device
        example_input = torch.randn(1, 16, 3, 112, 112, dtype=torch.float32, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "video_frames": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    logger.info(
        f"Exporting ONNX model (opset={opset_version}, dynamic_batch={dynamic_batch}, fixed T=16) to {out_p}..."
    )

    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            str(out_p),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["video_frames"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )

    # Validate ONNX schema
    try:
        import onnx
        onnx_model = onnx.load(str(out_p))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX schema check passed.")
    except ImportError:
        logger.warning("onnx package not installed; skipping onnx.checker validation.")

    if was_training:
        model.train()

    logger.info(f"Successfully saved ONNX model to: {out_p}")
    return out_p


def verify_export(
    model: nn.Module,
    exported_path: Union[str, Path],
    format_type: str = "torchscript",
    example_input: Optional[torch.Tensor] = None,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """Verify numerical parity between eager PyTorch model and exported artifact.

    Args:
        model: Original PyTorch eager model.
        exported_path: Path to exported .pt or .onnx file.
        format_type: 'torchscript' or 'onnx'.
        example_input: Tensor to evaluate on. Defaults to (1, 16, 3, 112, 112).
        atol: Maximum allowable absolute numerical tolerance.

    Returns:
        Dictionary containing max difference, parity boolean, and output shapes.
    """
    exp_p = Path(exported_path)
    if not exp_p.is_file():
        raise FileNotFoundError(f"Exported artifact not found: {exp_p}")

    model.eval()
    device = next(model.parameters()).device

    if example_input is None:
        example_input = torch.randn(1, 16, 3, 112, 112, dtype=torch.float32, device=device)

    # 1. Eager PyTorch output
    with torch.no_grad():
        eager_out = model(example_input).detach().cpu()

    # 2. Exported output
    if format_type.lower() in ("torchscript", "ts", "pt"):
        ts_model = torch.jit.load(str(exp_p), map_location=device)
        ts_model.eval()
        with torch.no_grad():
            exported_out = ts_model(example_input).detach().cpu()
    elif format_type.lower() == "onnx":
        import onnxruntime as ort
        session = ort.InferenceSession(str(exp_p), providers=["CPUExecutionProvider"])
        inp_np = example_input.detach().cpu().numpy()
        onnx_res = session.run(["logits"], {"video_frames": inp_np})
        exported_out = torch.from_numpy(onnx_res[0])
    else:
        raise ValueError(f"Unsupported format_type: {format_type}. Must be 'torchscript' or 'onnx'.")

    # 3. Numerical comparison
    abs_diff = torch.abs(eager_out - exported_out)
    max_diff = float(abs_diff.max().item())
    is_parity = max_diff <= atol

    result = {
        "format": format_type,
        "exported_path": str(exp_p.resolve()),
        "max_abs_diff": max_diff,
        "atol": atol,
        "parity_verified": is_parity,
        "eager_shape": list(eager_out.shape),
        "exported_shape": list(exported_out.shape),
    }

    logger.info(
        f"Export Parity [{format_type}]: Max diff = {max_diff:.8f} (tolerance = {atol}) -> "
        f"{'PASSED' if is_parity else 'FAILED'}"
    )

    return result


def export_checkpoint(
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    export_format: str = "all",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Export a trained checkpoint to TorchScript, ONNX, or both without modifying the checkpoint.

    Args:
        checkpoint_path: Path to frozen .pth checkpoint.
        output_dir: Destination directory.
        export_format: 'torchscript', 'onnx', or 'all'.
        device: Device to load in-memory model onto.

    Returns:
        Dictionary summarizing exported artifacts and verification results.
    """
    ckpt_p = Path(checkpoint_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_data = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    if "config" in ckpt_data and ckpt_data["config"] is not None:
        config = _dict_to_config(ckpt_data["config"])
    else:
        config = load_config("configs/default.yaml")

    config.device = device
    model = DeepfakeDetector(config.model)
    state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    example_input = torch.randn(1, 16, 3, 112, 112, dtype=torch.float32, device=device)
    stem = ckpt_p.stem
    results: Dict[str, Any] = {"checkpoint": str(ckpt_p), "exports": {}}

    if export_format in ("torchscript", "all"):
        ts_path = out_dir / f"{stem}.pt"
        export_torchscript(model, ts_path, example_input=example_input)
        ts_verify = verify_export(model, ts_path, format_type="torchscript", example_input=example_input)
        results["exports"]["torchscript"] = ts_verify

    if export_format in ("onnx", "all"):
        onnx_path = out_dir / f"{stem}.onnx"
        export_onnx(model, onnx_path, example_input=example_input, dynamic_batch=True)
        onnx_verify = verify_export(model, onnx_path, format_type="onnx", example_input=example_input)
        results["exports"]["onnx"] = onnx_verify

    return results
