"""Tensor preprocessing: resizing, scaling, and normalization."""

from __future__ import annotations

from typing import List
import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TensorPreprocessor:
    """Resizes cropped face frames to target dimensions and constructs normalized PyTorch tensors."""

    def __init__(self, image_size: int = 112, normalization: str = "imagenet"):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TensorPreprocessor.")

        self.image_size = image_size
        self.normalization = normalization

        # ImageNet channel statistics
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def preprocess_crops(self, crop_images: List[np.ndarray]) -> torch.Tensor:
        """Convert a sequence of RGB face crops into a normalized tensor.

        Args:
            crop_images: List of HxWxC uint8 RGB ndarrays.

        Returns:
            torch.Tensor of shape (T, 3, image_size, image_size) and dtype float32.

        Raises:
            ValueError: If crop_images is empty or contains None values.
        """
        if not crop_images:
            raise ValueError("crop_images cannot be empty")

        resized_list: List[np.ndarray] = []
        for i, img in enumerate(crop_images):
            if img is None:
                raise ValueError(f"Crop at index {i} is None; cannot construct tensor.")
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                resized = cv2.resize(
                    img,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                resized = img
            resized_list.append(resized)

        # Stack into (T, H, W, 3)
        stacked = np.stack(resized_list, axis=0)

        # Permute to (T, 3, H, W) and scale to [0.0, 1.0]
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0

        if self.normalization == "imagenet":
            tensor = (tensor - self.imagenet_mean) / self.imagenet_std
        elif self.normalization == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalization: {self.normalization}")

        return tensor
