"""Step 1: Failure Case Collection Infrastructure.

Provides structured registration, hashing, and storage for failure cases (false negatives and false alarms)
in an append-only JSONL database.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class HumanVerdict(str, Enum):
    DEFINITELY_FAKE = "definitely_fake"
    PROBABLY_FAKE = "probably_fake"
    UNCERTAIN = "uncertain"
    DEFINITELY_REAL = "definitely_real"
    PROBABLY_REAL = "probably_real"


class ManipulationFamily(str, Enum):
    FACE_SWAP = "face_swap"  # FaceSwap, FaceFusion, Roop
    REENACTMENT = "reenactment"  # Face2Face, LivePortrait
    NEURAL_TEXTURE = "neural_texture"  # NeuralTextures
    FULL_SYNTHESIS = "full_synthesis"  # Sora, Runway Gen-3, Kling
    LIP_SYNC = "lip_sync"  # Wav2Lip, SadTalker
    AUDIO_CLONING = "audio_cloning"  # ElevenLabs, Tortoise
    UNKNOWN = "unknown"


class HumanConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HumanReason(str, Enum):
    TEMPORAL_MOTION = "temporal_motion"
    LIP_SYNC = "lip_sync"
    EYE_BEHAVIOR = "eye_behavior"
    MOUTH_GEOMETRY = "mouth_geometry"
    SKIN_TEXTURE = "skin_texture"
    IDENTITY = "identity"
    LIGHTING = "lighting"
    ANATOMY = "anatomy"
    AUDIO = "audio"
    CONTEXTUAL = "contextual"
    UNKNOWN = "unknown"


class DatasetSplit(str, Enum):
    DISCOVERY = "discovery"  # Signal exploration and feature engineering
    DEVELOPMENT = "development"  # Model fusion, hyperparameters, threshold tuning
    HOLDOUT = "holdout"  # Locked final comparison with M13.5
    EXTERNAL_OOD = "external_ood"  # Unseen manipulation/generator families


@dataclass
class FailureCase:
    """Strongly typed container for an annotated failure case with human forensic observations."""

    video_id: str
    video_path: str
    sha256: str
    true_label: str  # FAKE / REAL
    dataset_split: str  # discovery / development / holdout / external_ood
    m13_5_verdict: str  # REAL / DEEPFAKE / REJECT
    m13_5_logit: float
    m13_5_prob: float
    # Human evidence annotations: answering "What are humans seeing that M13.5 isn't?"
    human_verdict: str  # HumanVerdict string
    human_confidence: str  # HumanConfidence string (high, medium, low)
    human_reason: str  # HumanReason string
    evidence_region: str  # "mouth", "eyes", "jawline", "neck", "full_face", "background"
    evidence_time_range: List[float]  # [start_sec, end_sec]
    manipulation_family: str  # ManipulationFamily string
    source_tool: str  # e.g. "FaceFusion 2.6", "Kling v1.5"
    source_url: str  # if downloaded
    compression_chain: List[str]  # e.g. ["H264-CRF18", "TikTok-VP9"]
    notes: str
    collected_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def register_failure(
    video_path: str | Path,
    m13_5_verdict: str,
    m13_5_logit: float,
    human_verdict: HumanVerdict | str,
    manipulation_family: ManipulationFamily | str,
    true_label: str = "FAKE",
    dataset_split: DatasetSplit | str = DatasetSplit.DISCOVERY,
    human_confidence: HumanConfidence | str = HumanConfidence.HIGH,
    human_reason: HumanReason | str = HumanReason.UNKNOWN,
    evidence_region: str = "full_face",
    evidence_time_range: Optional[List[float]] = None,
    source_tool: str = "",
    source_url: str = "",
    compression_chain: Optional[List[str]] = None,
    notes: str = "",
    db_path: str | Path = "data/failure_cases.jsonl",
) -> FailureCase:
    """Register a new failure case into the failure corpus JSONL database with human observations.

    Args:
        video_path: Path to the physical video file.
        m13_5_verdict: Model verdict (REAL, DEEPFAKE, REJECT).
        m13_5_logit: Uncalibrated model logit.
        human_verdict: Human expert or ground-truth assessment.
        manipulation_family: Category of manipulation.
        true_label: Ground truth label ("FAKE" or "REAL").
        dataset_split: Split allocation (discovery, development, holdout, external_ood).
        human_confidence: Human confidence (high, medium, low).
        human_reason: What specific visual/temporal cue humans observed.
        evidence_region: Anatomical/spatial area of the anomaly (e.g. mouth, eyes, jawline).
        evidence_time_range: [start_sec, end_sec] where the anomaly occurs.
        source_tool: Tool or algorithm name used to generate.
        source_url: Source URL if available.
        compression_chain: List of encoding/transcoding stages.
        notes: Contextual observations.
        db_path: Destination JSONL path.

    Returns:
        Populated FailureCase instance.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    # Compute SHA-256
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha = hasher.hexdigest()

    h_verdict_str = human_verdict.value if isinstance(human_verdict, HumanVerdict) else str(human_verdict)
    fam_str = manipulation_family.value if isinstance(manipulation_family, ManipulationFamily) else str(manipulation_family)
    split_str = dataset_split.value if isinstance(dataset_split, DatasetSplit) else str(dataset_split)
    conf_str = human_confidence.value if isinstance(human_confidence, HumanConfidence) else str(human_confidence)
    reason_str = human_reason.value if isinstance(human_reason, HumanReason) else str(human_reason)

    # Temperature-scaled probability (T=2.0502 from M13.5 calibration)
    calibrated_prob = float(1.0 / (1.0 + 2.718281828459045 ** (-m13_5_logit / 2.0502)))

    case = FailureCase(
        video_id=sha[:12],
        video_path=str(path.resolve()),
        sha256=sha,
        true_label=true_label.upper(),
        dataset_split=split_str.lower(),
        m13_5_verdict=m13_5_verdict.upper(),
        m13_5_logit=round(float(m13_5_logit), 4),
        m13_5_prob=round(calibrated_prob, 4),
        human_verdict=h_verdict_str,
        human_confidence=conf_str,
        human_reason=reason_str,
        evidence_region=evidence_region,
        evidence_time_range=evidence_time_range or [0.0, 0.0],
        manipulation_family=fam_str,
        source_tool=source_tool,
        source_url=source_url,
        compression_chain=compression_chain or [],
        notes=notes,
        collected_at=datetime.now(timezone.utc).isoformat(),
    )

    db_p = Path(db_path)
    db_p.parent.mkdir(parents=True, exist_ok=True)

    with open(db_p, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(case)) + "\n")

    print(
        f"[+] Registered failure: {case.video_id} ({case.dataset_split}) | "
        f"logit={case.m13_5_logit:.3f} | human_reason={case.human_reason} | region={case.evidence_region}"
    )
    return case



def load_failure_cases(db_path: str | Path = "data/failure_cases.jsonl") -> List[FailureCase]:
    """Load all failure cases from JSONL database."""
    db_p = Path(db_path)
    if not db_p.is_file():
        return []

    cases: List[FailureCase] = []
    with open(db_p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(FailureCase(**data))
    return cases


def main() -> None:
    """CLI interface for manual failure case registration."""
    parser = argparse.ArgumentParser(description="Register a deepfake detection failure case.")
    parser.add_argument("--video-path", type=str, required=True, help="Path to video file")
    parser.add_argument("--verdict", type=str, default="REAL", help="Model verdict (REAL/DEEPFAKE/REJECT)")
    parser.add_argument("--logit", type=float, default=0.0, help="Model raw logit")
    parser.add_argument(
        "--human-verdict",
        type=str,
        default="definitely_fake",
        choices=[v.value for v in HumanVerdict],
        help="Human verdict",
    )
    parser.add_argument(
        "--family",
        type=str,
        default="unknown",
        choices=[f.value for f in ManipulationFamily],
        help="Manipulation family",
    )
    parser.add_argument("--true-label", type=str, default="FAKE", choices=["FAKE", "REAL"], help="Ground truth")
    parser.add_argument("--source-tool", type=str, default="", help="Source generator tool name")
    parser.add_argument("--source-url", type=str, default="", help="Source URL")
    parser.add_argument("--notes", type=str, default="", help="Observational notes")
    parser.add_argument("--db-path", type=str, default="data/failure_cases.jsonl", help="Database file path")

    args = parser.parse_args()
    register_failure(
        video_path=args.video_path,
        m13_5_verdict=args.verdict,
        m13_5_logit=args.logit,
        human_verdict=HumanVerdict(args.human_verdict),
        manipulation_family=ManipulationFamily(args.family),
        true_label=args.true_label,
        source_tool=args.source_tool,
        source_url=args.source_url,
        notes=args.notes,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
