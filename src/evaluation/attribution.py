"""Attention attribution and spatial saliency extraction for DeepfakeDetector.

Extracts internal 4x4 soft spatial attention weights from the frozen baseline model,
computes spatial concentration metrics (entropy, top-k mass, peak coordinates),
and produces normalized heatmap overlays on face crops without modifying model internals.

Scientific Note: Attention weights visualize intermediate spatial weighting in the
feature aggregation bottleneck; they do NOT prove causal pixel-level manipulation masks.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FrameAttentionStats:
    """Concentration and spatial distribution statistics for a single frame."""

    frame_idx: int
    entropy: float  # Shannon entropy in nats (uniform = ln(16) ~ 2.773)
    max_weight: float  # Max activation in [0.0625, 1.0]
    min_weight: float
    top4_mass: float  # Sum of top 4 weights (top 25% spatial cells)
    peak_y: int  # 0..3 grid coordinate of max activation
    peak_x: int  # 0..3 grid coordinate of max activation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceAttentionSummary:
    """Summary of spatial attention behavior across an entire video sequence."""

    sequence_length: int
    mean_entropy: float
    mean_max_weight: float
    mean_top4_mass: float
    most_frequent_peak: Tuple[int, int]
    frame_stats: List[FrameAttentionStats]
    attention_shape: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_attention_weights(
    model: torch.nn.Module,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Extract soft spatial attention weights from a frozen DeepfakeDetector.

    Args:
        model: DeepfakeDetector instance (in eval mode).
        tensor: Video frame tensor of shape (1, 16, 3, 112, 112) or (16, 3, 112, 112).

    Returns:
        Tensor of shape (16, 4, 4) with softmax spatial weights summing to 1.0 per frame.
    """
    if tensor.dim() == 4:
        # Add batch dimension: (1, 16, 3, 112, 112)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() != 5 or tensor.shape[1] != 16 or tensor.shape[2] != 3:
        raise ValueError(
            f"Expected input tensor of shape (1, 16, 3, 112, 112), got {tuple(tensor.shape)}"
        )

    model.eval()
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor, return_intermediates=True)

    if not hasattr(output, "attention_maps") or "soft_map" not in output.attention_maps:
        raise RuntimeError("Model did not return 'soft_map' in attention_maps dictionary.")

    # Shape is (1, 16, 1, 4, 4) -> squeeze batch and channel -> (16, 4, 4)
    attn_map = output.attention_maps["soft_map"].squeeze(0).squeeze(1).detach().cpu()
    return attn_map


def compute_sequence_attention_stats(
    attn_weights: Union[torch.Tensor, np.ndarray],
) -> SequenceAttentionSummary:
    """Compute concentration and entropy statistics for (T, 4, 4) attention weights.

    Args:
        attn_weights: Tensor or ndarray of shape (T, 4, 4).

    Returns:
        SequenceAttentionSummary dataclass.
    """
    if isinstance(attn_weights, torch.Tensor):
        w = attn_weights.detach().cpu().numpy()
    else:
        w = np.asarray(attn_weights, dtype=np.float32)

    if w.ndim != 3 or w.shape[1:] != (4, 4):
        raise ValueError(f"Expected attention shape (T, 4, 4), got {w.shape}")

    t = w.shape[0]
    frame_stats: List[FrameAttentionStats] = []
    peak_counts: Dict[Tuple[int, int], int] = {}

    for i in range(t):
        frame_w = w[i]  # (4, 4)
        flat_w = frame_w.flatten()

        # Shannon entropy: -sum(p * log(p))
        clipped = np.clip(flat_w, 1e-12, 1.0)
        entropy = float(-np.sum(clipped * np.log(clipped)))

        max_w = float(np.max(flat_w))
        min_w = float(np.min(flat_w))

        # Top 4 cells mass (25% of 16 cells)
        sorted_w = np.sort(flat_w)[::-1]
        top4_mass = float(np.sum(sorted_w[:4]))

        # Peak coordinate
        peak_idx = int(np.argmax(flat_w))
        py, px = int(peak_idx // 4), int(peak_idx % 4)
        peak_counts[(py, px)] = peak_counts.get((py, px), 0) + 1

        frame_stats.append(
            FrameAttentionStats(
                frame_idx=i,
                entropy=round(entropy, 4),
                max_weight=round(max_w, 4),
                min_weight=round(min_w, 4),
                top4_mass=round(top4_mass, 4),
                peak_y=py,
                peak_x=px,
            )
        )

    most_freq_peak = max(peak_counts.items(), key=lambda item: item[1])[0] if peak_counts else (0, 0)
    mean_entropy = float(np.mean([f.entropy for f in frame_stats]))
    mean_max_w = float(np.mean([f.max_weight for f in frame_stats]))
    mean_top4 = float(np.mean([f.top4_mass for f in frame_stats]))

    return SequenceAttentionSummary(
        sequence_length=t,
        mean_entropy=round(mean_entropy, 4),
        mean_max_weight=round(mean_max_w, 4),
        mean_top4_mass=round(mean_top4, 4),
        most_frequent_peak=most_freq_peak,
        frame_stats=frame_stats,
        attention_shape=list(w.shape),
    )


def interpolate_attention_heatmap(
    attn_weights: Union[torch.Tensor, np.ndarray],
    target_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Bilinearly interpolate (T, 4, 4) attention weights to target spatial resolution.

    Args:
        attn_weights: Shape (T, 4, 4).
        target_size: Target (H, W) resolution (default: (112, 112)).

    Returns:
        ndarray of shape (T, target_size[0], target_size[1]) normalized to [0.0, 1.0] per frame.
    """
    if isinstance(attn_weights, np.ndarray):
        t_w = torch.from_numpy(attn_weights).float()
    else:
        t_w = attn_weights.float()

    if t_w.ndim != 3:
        raise ValueError(f"Expected 3D tensor (T, 4, 4), got {t_w.shape}")

    # (T, 1, 4, 4)
    t_in = t_w.unsqueeze(1)
    upsampled = F.interpolate(t_in, size=target_size, mode="bilinear", align_corners=False)
    # (T, H, W)
    heatmaps = upsampled.squeeze(1).numpy()

    # Normalize each frame to [0.0, 1.0] for visualization contrast
    normalized = np.zeros_like(heatmaps)
    for i in range(heatmaps.shape[0]):
        h_min = heatmaps[i].min()
        h_max = heatmaps[i].max()
        if h_max > h_min:
            normalized[i] = (heatmaps[i] - h_min) / (h_max - h_min)
        else:
            normalized[i] = np.zeros_like(heatmaps[i])

    return normalized


def create_attention_overlay(
    frame_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a single RGB face crop with a single 2D heatmap.

    Args:
        frame_rgb: uint8 array of shape (H, W, 3) in [0, 255] RGB.
        heatmap: float array of shape (H, W) in [0.0, 1.0].
        alpha: Blending weight for heatmap (1 - alpha for original crop).
        colormap: OpenCV colormap enum.

    Returns:
        uint8 RGB blended image of shape (H, W, 3).
    """
    if frame_rgb.shape[:2] != heatmap.shape:
        raise ValueError(f"Shape mismatch: frame {frame_rgb.shape[:2]} vs heatmap {heatmap.shape}")

    # Heatmap to uint8 grayscale
    h_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    # Apply colormap -> produces BGR
    colored_bgr = cv2.applyColorMap(h_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(colored_rgb, alpha, frame_rgb, 1.0 - alpha, 0)
    return blended


def create_attribution_panel(
    frame_rgb: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
) -> np.ndarray:
    """Concatenate original face crop, raw heatmap, and overlay horizontally."""
    # Heatmap RGB representation
    h_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
    h_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    panel = np.hstack([frame_rgb, h_rgb, overlay])
    return panel
