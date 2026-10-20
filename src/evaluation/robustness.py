"""Video and frame perturbation utilities for robustness evaluation under controlled degradations."""

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PerturbationConfig:
    """Configuration for a specific robustness perturbation."""

    name: str
    category: str  # "compression", "spatial", "temporal", "photometric"
    parameter_name: str
    parameter_value: Any
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_spatial_downsampling(
    tensor: torch.Tensor,
    low_res_size: Tuple[int, int] = (56, 56),
    final_size: Tuple[int, int] = (112, 112),
) -> torch.Tensor:
    """Downsample face crops to a lower resolution and upsample back to 112x112.

    Simulates low-resolution or distant face camera capture.

    Args:
        tensor: Shape (T, C, H, W) or (B, T, C, H, W).
        low_res_size: Target intermediate resolution (e.g. 56x56 or 28x28).
        final_size: Final model input resolution (112x112).

    Returns:
        Perturbed tensor of identical input shape.
    """
    is_5d = tensor.dim() == 5
    orig_shape = tensor.shape

    if is_5d:
        b, t, c, h, w = orig_shape
        x = tensor.view(b * t, c, h, w)
    else:
        t, c, h, w = orig_shape
        x = tensor

    # Bilinear downsampling then upsampling
    down = F.interpolate(x, size=low_res_size, mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=final_size, mode="bilinear", align_corners=False)

    return up.view(orig_shape)


def apply_gaussian_blur(
    tensor: torch.Tensor,
    sigma: float = 1.0,
    kernel_size: int = 7,
) -> torch.Tensor:
    """Apply 2D Gaussian blur across all frames in a tensor.

    Args:
        tensor: Shape (T, C, H, W) or (B, T, C, H, W).
        sigma: Gaussian standard deviation (e.g. 0.5, 1.0, 1.5).
        kernel_size: Odd integer kernel size.

    Returns:
        Blurred tensor.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
    g1d = g1d / g1d.sum()

    # Create 2D separable depthwise kernel: (C, 1, K, K)
    g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)
    c = tensor.shape[-3]
    kernel = g2d.repeat(c, 1, 1, 1).to(tensor.device, dtype=tensor.dtype)

    is_5d = tensor.dim() == 5
    orig_shape = tensor.shape
    if is_5d:
        b, t, c, h, w = orig_shape
        x = tensor.view(b * t, c, h, w)
    else:
        t, c, h, w = orig_shape
        x = tensor

    pad = kernel_size // 2
    blurred = F.conv2d(x, kernel, padding=pad, groups=c)
    return blurred.view(orig_shape)


def apply_additive_noise(
    tensor: torch.Tensor,
    std: float = 0.05,
    seed: Optional[int] = 42,
) -> torch.Tensor:
    """Add zero-mean Gaussian sensor noise to normalized tensor.

    Args:
        tensor: Input tensor.
        std: Noise standard deviation.
        seed: Deterministic random seed.

    Returns:
        Tensor with additive noise.
    """
    if seed is not None:
        generator = torch.Generator(device=tensor.device).manual_seed(seed)
        noise = torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype) * std
    else:
        noise = torch.randn_like(tensor) * std
    return tensor + noise


def apply_contrast_reduction(
    tensor: torch.Tensor,
    factor: float = 0.5,
) -> torch.Tensor:
    """Reduce contrast around per-frame channel mean.

    Args:
        tensor: Input tensor.
        factor: Contrast factor in [0.0, 1.0] (1.0 = unchanged, 0.5 = 50% contrast).

    Returns:
        Contrast-reduced tensor.
    """
    # Compute mean across spatial dimensions (H, W)
    mean = tensor.mean(dim=(-1, -2), keepdim=True)
    return mean + factor * (tensor - mean)


def apply_jpeg_compression_simulation(
    tensor: torch.Tensor,
    quality: int = 30,
) -> torch.Tensor:
    """Simulate lossy JPEG/H.264 block compression via OpenCV encoding/decoding.

    Args:
        tensor: Shape (T, 3, 112, 112) or (B, T, 3, 112, 112).
        quality: JPEG compression quality parameter in [1, 100].

    Returns:
        Compressed tensor.
    """
    is_5d = tensor.dim() == 5
    orig_shape = tensor.shape
    x = tensor.view(-1, 3, 112, 112).detach().cpu().numpy()

    # ImageNet stats for un-normalizing
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    # Unnormalize to [0, 255]
    unnorm = (x * std + mean) * 255.0
    unnorm = np.clip(unnorm, 0, 255).astype(np.uint8)

    compressed_list = []
    for i in range(unnorm.shape[0]):
        # Convert (3, H, W) RGB -> (H, W, 3) BGR for OpenCV
        frame_rgb = np.transpose(unnorm[i], (1, 2, 0))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Compress to JPEG memory buffer
        success, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            dec_bgr = frame_bgr
        else:
            dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        dec_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        dec_norm = (np.transpose(dec_rgb, (2, 0, 1)) - mean.squeeze(0)) / std.squeeze(0)
        compressed_list.append(dec_norm)

    out_arr = np.stack(compressed_list, axis=0)
    out_tensor = torch.from_numpy(out_arr).to(device=tensor.device, dtype=tensor.dtype)
    return out_tensor.view(orig_shape)
