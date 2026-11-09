"""FaceForensics++ (FF++) dataset adapter for group_id extraction and manifest indexing."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from src.data.manifest import split_manifest_by_group


def extract_ffpp_group_id(filename_or_stem: Union[str, Path]) -> str:
    """Extract the original source identity group_id from a FaceForensics++ video path or name.

    In FaceForensics++:
    - Pristine (original) videos are named: '<source_id>.mp4' (e.g. '000.mp4' -> group_id '000').
    - Manipulated videos are named: '<source_id>_<target_id>.mp4' (e.g. '000_003.mp4' -> group_id '000').

    Mapping both pristine and manipulated derivatives to '<source_id>' ensures
    that videos sharing the same original identity never appear across multiple splits.

    Args:
        filename_or_stem: Filename, stem, or Path of the video.

    Returns:
        The extracted source group_id as a string.
    """
    stem = Path(filename_or_stem).stem
    parts = stem.split("_")
    # First component represents the original source video id
    group_id = parts[0].strip()
    if not group_id:
        raise ValueError(f"Unable to extract FF++ group_id from: '{filename_or_stem}'")
    return group_id


def build_ffpp_manifest_from_paths(
    video_paths: List[Union[str, Path]],
    default_label_is_fake: Optional[bool] = None,
) -> pd.DataFrame:
    """Construct a manifest DataFrame from a list of FaceForensics++ video paths.

    Args:
        video_paths: List of paths to video files.
        default_label_is_fake: If True or False, force all labels. If None,
                               infers 0 if stem has no underscore, 1 if stem has underscore.

    Returns:
        DataFrame conforming to the manifest schema (video_id, group_id, video_path, label, dataset_name).
    """
    records = []
    for p in video_paths:
        path_obj = Path(p)
        stem = path_obj.stem
        group_id = extract_ffpp_group_id(stem)

        if default_label_is_fake is not None:
            label = 1 if default_label_is_fake else 0
        else:
            # Typical convention: single number like '000' is real; '000_003' is fake
            label = 1 if "_" in stem else 0

        manipulation = "pristine" if label == 0 else "manipulated"

        records.append({
            "video_id": stem,
            "group_id": group_id,
            "video_path": str(path_obj.resolve()),
            "label": label,
            "dataset_name": "FaceForensics++",
            "manipulation_type": manipulation,
            "split": "",
        })

    return pd.DataFrame(records)


def build_ffpp_manifest(
    dataset_root: Union[str, Path],
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    include_all_methods: bool = False,
) -> pd.DataFrame:
    """Construct a complete FaceForensics++ manifest from a dataset root directory.

    Scans pristine ('original') and manipulation directories ('Deepfakes', 'Face2Face',
    'FaceSwap', 'NeuralTextures', and optionally 'FaceShifter', 'DeepFakeDetection').

    Establishes mutual pair mappings to assign both interacting actors in any manipulation
    pair to the same group_id, strictly guaranteeing zero identity/group leakage across
    train, val, and test splits.

    Args:
        dataset_root: Path to the root FaceForensics++ folder.
        split_ratios: (train, val, test) split fractions.
        seed: Random seed for deterministic group stratification.
        include_all_methods: If True, include FaceShifter and DeepFakeDetection.

    Returns:
        DataFrame conforming to the manifest schema with allocated splits.
    """
    root_path = Path(dataset_root)
    if not root_path.exists():
        raise FileNotFoundError(f"FaceForensics++ dataset root not found: {dataset_root}")

    # Build mutual pair mapping from Deepfakes or other standard manipulation directory
    pair_map: Dict[str, str] = {}
    pair_dir = root_path / "Deepfakes"
    if pair_dir.exists():
        for f in pair_dir.glob("*.mp4"):
            parts = f.stem.split("_")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                s, t = parts[0], parts[1]
                grp = f"{min(int(s), int(t)):03d}"
                pair_map[s] = grp
                pair_map[t] = grp

    records: List[Dict[str, Any]] = []

    # 1. Pristine / original videos
    orig_dir = root_path / "original"
    if orig_dir.exists():
        for f in sorted(orig_dir.glob("*.mp4")):
            stem = f.stem
            grp = pair_map.get(stem, stem)
            records.append({
                "video_id": f"orig_{stem}",
                "group_id": grp,
                "video_path": str(f.resolve()),
                "label": 0,
                "dataset_name": "FaceForensics++",
                "manipulation_type": "original",
            })

    # 2. Canonical manipulation methods
    method_prefix = {
        "Deepfakes": "df",
        "Face2Face": "f2f",
        "FaceSwap": "fs",
        "NeuralTextures": "nt",
    }
    if include_all_methods:
        method_prefix["FaceShifter"] = "fsh"
        method_prefix["DeepFakeDetection"] = "dfd"

    for method, pfx in method_prefix.items():
        m_dir = root_path / method
        if not m_dir.exists():
            continue
        for f in sorted(m_dir.glob("*.mp4")):
            stem = f.stem
            parts = stem.split("_")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                grp = f"{min(int(parts[0]), int(parts[1])):03d}"
            else:
                grp = parts[0]

            records.append({
                "video_id": f"{pfx}_{stem}",
                "group_id": grp,
                "video_path": str(f.resolve()),
                "label": 1,
                "dataset_name": "FaceForensics++",
                "manipulation_type": method,
            })

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise ValueError(f"No video files found under FaceForensics++ root: {dataset_root}")

    # Split manifest using indivisible group stratification
    split_df = split_manifest_by_group(df, ratios=split_ratios, seed=seed, check_paths=False)
    return split_df

