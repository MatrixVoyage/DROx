"""Atomic persistence of processed tensors, metadata JSON, and experiment configuration."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, Tuple, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.utils.config import AppConfig, save_config
from src.data.types import PreprocessResult


class ProcessedDataStorage:
    """Manages atomic saving and loading of preprocessed face-sequence tensors and audit metadata."""

    def __init__(self, base_dir: Union[str, Path] = "processed"):
        self.base_dir = Path(base_dir)

    def _atomic_write_bytes(self, destination: Path, write_fn) -> None:
        """Write file atomically using a temporary file in the destination folder."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            dir=str(destination.parent),
            prefix=f".tmp_{destination.stem}_",
            suffix=".tmp",
        )
        os.close(fd)
        try:
            write_fn(temp_path)
            os.replace(temp_path, str(destination))
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def save(
        self,
        result: PreprocessResult,
        split: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Path], Path]:
        """Atomically persist preprocessing outputs.

        Invariant: Rejected or failed sequences are never written as successful tensors.

        Args:
            result: PreprocessResult from the pipeline.
            split: Optional split override ("train", "val", "test").
            config_snapshot: Optional configuration dictionary snapshot.

        Returns:
            Tuple of (Optional[tensor_path], metadata_path).
        """
        target_split = split or result.split or "unassigned"

        if result.status != "success":
            # Persist rejected metadata only; do not save tensor
            meta_dir = self.base_dir / target_split / "rejected_metadata"
            meta_path = meta_dir / f"{result.video_id}.json"

            meta_data = result.to_metadata_dict(config_snapshot=config_snapshot)

            def write_meta(path_str: str) -> None:
                with open(path_str, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f, indent=2)

            self._atomic_write_bytes(meta_path, write_meta)
            return None, meta_path

        # Successful result
        tensor_dir = self.base_dir / target_split / "tensors"
        meta_dir = self.base_dir / target_split / "metadata"

        tensor_path = tensor_dir / f"{result.video_id}.pt"
        meta_path = meta_dir / f"{result.video_id}.json"

        # 1. Save tensor atomically
        def write_tensor(path_str: str) -> None:
            torch.save(result.tensor, path_str)

        self._atomic_write_bytes(tensor_path, write_tensor)

        # 2. Save metadata JSON atomically
        meta_data = result.to_metadata_dict(config_snapshot=config_snapshot)

        def write_meta(path_str: str) -> None:
            with open(path_str, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)

        self._atomic_write_bytes(meta_path, write_meta)

        return tensor_path, meta_path

    def save_run_config(self, config: AppConfig, filename: str = "config.yaml") -> Path:
        """Atomically save the run-level configuration file in the base storage directory."""
        dest_path = self.base_dir / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, dest_path)
        return dest_path

    def load_tensor(self, tensor_path: Union[str, Path]) -> "torch.Tensor":
        """Load a saved tensor from disk with weights_only=True."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to load tensors.")
        return torch.load(str(tensor_path), weights_only=True)

    def load_metadata(self, metadata_path: Union[str, Path]) -> Dict[str, Any]:
        """Load metadata dictionary from JSON file."""
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
