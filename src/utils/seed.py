"""Reproducibility and deterministic seed utilities."""

import hashlib
import os
import random
from typing import Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Set random seeds across Python, NumPy, and PyTorch for reproducible runs.

    Args:
        seed: The integer seed to apply.
        deterministic: If True, configures PyTorch cuDNN to deterministic algorithms.

    Returns:
        The seed integer used.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except AttributeError:
                pass

    return seed


def derive_video_seed(seed: int, video_id: str) -> int:
    """Derive a stable, deterministic 32-bit integer seed for a specific video.

    Uses SHA-256 to guarantee stable seed derivation across Python processes
    and operating systems (avoiding Python's randomized hash()).

    Args:
        seed: Base experiment seed.
        video_id: Unique string identifier of the video.

    Returns:
        A deterministic positive 32-bit integer seed.
    """
    key = f"{seed}:{video_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], byteorder="big") & 0x7FFFFFFF
