"""Configuration schemas, YAML loading, saving, and strict cross-validation."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class HardwareConfig:
    max_vram_gb: float = 6.0


@dataclass
class SamplingConfig:
    method: str = "random"  # "random" or "uniform"
    num_frames: int = 16
    short_video_policy: str = "reject"  # strictly "reject"


@dataclass
class DetectionConfig:
    selection_strategy: str = "largest"  # "largest" or "highest_confidence"
    confidence_threshold: float = 0.90


@dataclass
class FallbackConfig:
    use_previous_bbox: bool = True
    allow_center_crop: bool = True


@dataclass
class CropConfig:
    image_size: int = 112
    margin_ratio: float = 0.20
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    min_valid_frames: int = 12
    max_fallback_ratio: float = 0.50

    def __post_init__(self):
        if isinstance(self.fallback, dict):
            self.fallback = FallbackConfig(**self.fallback)


@dataclass
class AugmentationConfig:
    enabled: bool = False
    jpeg_p: float = 0.4
    blur_p: float = 0.3
    noise_p: float = 0.3
    resize_p: float = 0.2
    reencode_p: float = 0.15
    frame_drop_p: float = 0.1
    max_drop_frames: int = 2
    severity_mild_p: float = 0.75
    severity_moderate_p: float = 0.20
    severity_severe_p: float = 0.05


@dataclass
class DataConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    normalization: str = "imagenet"  # "imagenet" or "none"


@dataclass
class StorageConfig:
    processed_dir: str = "processed"


@dataclass
class BackboneConfig:
    name: str = "resnext50_32x4d"
    weights: Optional[str] = "IMAGENET1K_V1"  # "IMAGENET1K_V1", "DEFAULT", or None
    freeze_strategy: str = "layer2"  # "none", "stem", "layer1", "layer2", "layer3", "early", "mid", "late", "all"
    freeze_batchnorm: bool = True    # keep frozen stages' BatchNorm in eval() mode
    input_size: list[int] = field(default_factory=lambda: [112, 112])
    output_dim: int = 2048
    spatial_dim: int = 4

    def __post_init__(self):
        if self.weights not in ("IMAGENET1K_V1", "DEFAULT", None):
            raise ValueError(f"weights must be 'IMAGENET1K_V1' or None, got '{self.weights}'")
        valid_freeze = ("none", "stem", "layer1", "layer2", "layer3", "early", "mid", "late", "all")
        if self.freeze_strategy not in valid_freeze:
            raise ValueError(f"freeze_strategy must be one of {valid_freeze}, got '{self.freeze_strategy}'")

    @property
    def pretrained(self) -> bool:
        return self.weights is not None


@dataclass
class AttentionConfig:
    type: str = "soft"  # "soft", "cbam", "channel", "spatial", "none"
    in_channels: int = 2048
    reduction_ratio: int = 16
    kernel_size: int = 3

    def __post_init__(self):
        valid_types = ("soft", "cbam", "channel", "spatial", "none")
        if self.type not in valid_types:
            raise ValueError(f"Invalid attention.type: '{self.type}'. Must be one of {valid_types}.")
        if self.in_channels not in (2048, 1280):
            raise ValueError(f"attention.in_channels must be 2048 or 1280, got {self.in_channels}")
        if self.reduction_ratio <= 0:
            raise ValueError(f"attention.reduction_ratio must be > 0, got {self.reduction_ratio}")
        if self.in_channels % self.reduction_ratio != 0:
            raise ValueError(
                f"attention.in_channels ({self.in_channels}) must be divisible by reduction_ratio ({self.reduction_ratio})"
            )
        if self.kernel_size not in (3, 7):
            raise ValueError(f"attention.kernel_size must be 3 or 7, got {self.kernel_size}")


@dataclass
class TemporalConfig:
    type: str = "lstm"          # "lstm" (paper default), "bilstm", or "transformer"
    input_dim: int = 2048
    hidden_dim: int = 2048      # 2048 for lstm; 1024 for bilstm; ignored for transformer
    num_layers: int = 1         # strictly 1 for M4 baseline LSTM
    bidirectional: bool = False # False for lstm; True for bilstm
    dropout: float = 0.5        # Applied explicitly post-pooling
    pooling: str = "last"       # "last", "mean", or "cls_token"
    nhead: int = 4              # Transformer attention heads
    transformer_layers: int = 2 # Transformer encoder layers
    dim_feedforward: int = 2048 # Transformer feedforward dim

    def __post_init__(self):
        if self.type not in ("lstm", "bilstm", "transformer"):
            raise ValueError(f"Invalid temporal.type: '{self.type}'. Must be 'lstm', 'bilstm', or 'transformer'.")
        if self.type == "lstm":
            if self.bidirectional:
                raise ValueError("temporal.type='lstm' requires bidirectional=False")
            if self.hidden_dim != 2048:
                raise ValueError(f"temporal.type='lstm' requires hidden_dim=2048, got {self.hidden_dim}")
            if self.num_layers != 1:
                raise ValueError(f"Milestone 4 strictly requires num_layers=1, got {self.num_layers}")
        elif self.type == "bilstm":
            if not self.bidirectional:
                raise ValueError("temporal.type='bilstm' requires bidirectional=True")
            if self.hidden_dim != 1024:
                raise ValueError(
                    f"temporal.type='bilstm' requires hidden_dim=1024 for controlled 2048-D output, got {self.hidden_dim}"
                )
            if self.num_layers != 1:
                raise ValueError(f"Milestone 4 strictly requires num_layers=1, got {self.num_layers}")
        elif self.type == "transformer":
            if self.input_dim not in (2048, 1280):
                raise ValueError(f"temporal.input_dim must be 2048 or 1280, got {self.input_dim}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"temporal.dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.pooling not in ("last", "mean", "cls_token"):
            raise ValueError(f"Invalid temporal.pooling: '{self.pooling}'. Must be 'last', 'mean', or 'cls_token'.")

    @property
    def output_dim(self) -> int:
        if self.type == "transformer":
            return self.input_dim
        return self.hidden_dim * (2 if self.bidirectional else 1)


@dataclass
class ClassifierConfig:
    in_features: int = 2048
    hidden_dim: int = 512
    num_classes: int = 1
    dropout: float = 0.5

    def __post_init__(self):
        if self.in_features not in (2048, 1280):
            raise ValueError(f"classifier.in_features must be 2048 or 1280, got {self.in_features}")
        if self.hidden_dim <= 0:
            raise ValueError(f"classifier.hidden_dim must be > 0, got {self.hidden_dim}")
        if self.num_classes != 1:
            raise ValueError(f"classifier.num_classes must be 1 for binary classification, got {self.num_classes}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"classifier.dropout must be in [0.0, 1.0), got {self.dropout}")


@dataclass
class AuxiliaryConfig:
    enabled: bool = False
    num_families: int = 5
    hidden_dim: int = 512
    dropout: float = 0.5
    loss_weight: float = 0.5


@dataclass
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    auxiliary: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)


@dataclass
class OptimizerConfig:
    type: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])

    def __post_init__(self):
        if self.type != "adam":
            raise ValueError(f"Invalid optimizer.type: '{self.type}'. Expected 'adam'.")
        if self.learning_rate <= 0:
            raise ValueError(f"optimizer.learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"optimizer.weight_decay must be >= 0, got {self.weight_decay}")
        if len(self.betas) != 2 or not (0.0 <= self.betas[0] < 1.0) or not (0.0 <= self.betas[1] < 1.0):
            raise ValueError(f"optimizer.betas must be 2 floats in [0, 1), got {self.betas}")


@dataclass
class SchedulerConfig:
    type: str = "plateau"  # "plateau", "cosine", "none"
    factor: float = 0.1
    patience: int = 5
    min_lr: float = 1e-6
    mode: str = "max"  # "min" or "max"

    def __post_init__(self):
        valid_types = ("plateau", "cosine", "none")
        if self.type not in valid_types:
            raise ValueError(f"Invalid scheduler.type: '{self.type}'. Must be one of {valid_types}.")
        if self.factor <= 0 or self.factor >= 1.0:
            raise ValueError(f"scheduler.factor must be in (0, 1), got {self.factor}")
        if self.patience < 0:
            raise ValueError(f"scheduler.patience must be >= 0, got {self.patience}")
        if self.min_lr < 0:
            raise ValueError(f"scheduler.min_lr must be >= 0, got {self.min_lr}")
        if self.mode not in ("min", "max"):
            raise ValueError(f"scheduler.mode must be 'min' or 'max', got '{self.mode}'")


@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_epochs: int = 50
    grad_accum_steps: int = 2
    amp: bool = True
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    monitor_metric: str = "val_auc"  # "val_auc", "val_loss", "val_acc", "val_f1"
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 0
    pin_memory: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss_type: str = "bce"  # "bce", "focal", "composite"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.1
    balanced_sampling: bool = False
    sequence_augmentation: bool = False

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"training.batch_size must be > 0, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"training.num_epochs must be > 0, got {self.num_epochs}")
        if self.grad_accum_steps <= 0:
            raise ValueError(f"training.grad_accum_steps must be > 0, got {self.grad_accum_steps}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"training.max_grad_norm must be > 0, got {self.max_grad_norm}")
        if self.early_stopping_patience < 0:
            raise ValueError(f"training.early_stopping_patience must be >= 0, got {self.early_stopping_patience}")
        valid_monitors = ("val_auc", "val_loss", "val_acc", "val_f1")
        if self.monitor_metric not in valid_monitors:
            raise ValueError(
                f"Invalid training.monitor_metric: '{self.monitor_metric}'. Must be one of {valid_monitors}."
            )
        if self.num_workers < 0:
            raise ValueError(f"training.num_workers must be >= 0, got {self.num_workers}")
        if not self.checkpoint_dir:
            raise ValueError("training.checkpoint_dir cannot be empty")
        valid_losses = ("bce", "focal", "composite")
        if self.loss_type not in valid_losses:
            raise ValueError(f"Invalid training.loss_type: '{self.loss_type}'. Must be one of {valid_losses}.")


@dataclass
class EvaluationConfig:
    batch_size: int = 4
    threshold: float = 0.5

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"evaluation.batch_size must be > 0, got {self.batch_size}")
        if not (0.0 < self.threshold < 1.0):
            raise ValueError(f"evaluation.threshold must be in (0, 1), got {self.threshold}")


@dataclass
class AppConfig:
    seed: int = 42
    device: str = "cuda"
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        """Perform cross-validation and assert validity of all hyperparameter combinations."""
        # Hardware validation
        if self.hardware.max_vram_gb <= 0:
            raise ValueError(f"hardware.max_vram_gb must be positive, got {self.hardware.max_vram_gb}")

        # Sampling validation
        if self.data.sampling.method not in ("random", "uniform"):
            raise ValueError(f"Invalid sampling.method: '{self.data.sampling.method}'. Must be 'random' or 'uniform'.")
        if self.data.sampling.num_frames <= 0:
            raise ValueError(f"sampling.num_frames must be > 0, got {self.data.sampling.num_frames}")
        if self.data.sampling.short_video_policy not in ("reject",):
            raise ValueError(
                f"Invalid short_video_policy: '{self.data.sampling.short_video_policy}'. "
                "Must be 'reject' (no silent frame duplication)."
            )

        # Detection validation
        if self.data.detection.selection_strategy not in ("largest", "highest_confidence"):
            raise ValueError(
                f"Invalid detection.selection_strategy: '{self.data.detection.selection_strategy}'. "
                "Must be 'largest' or 'highest_confidence'."
            )
        if not (0.0 <= self.data.detection.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {self.data.detection.confidence_threshold}"
            )

        # Crop validation
        if self.data.crop.image_size <= 0:
            raise ValueError(f"crop.image_size must be > 0, got {self.data.crop.image_size}")
        if self.data.crop.margin_ratio < 0.0:
            raise ValueError(f"crop.margin_ratio must be >= 0.0, got {self.data.crop.margin_ratio}")
        if not (0.0 <= self.data.crop.max_fallback_ratio <= 1.0):
            raise ValueError(
                f"crop.max_fallback_ratio must be in [0.0, 1.0], got {self.data.crop.max_fallback_ratio}"
            )
        if self.data.crop.min_valid_frames > self.data.sampling.num_frames:
            raise ValueError(
                f"crop.min_valid_frames ({self.data.crop.min_valid_frames}) cannot exceed "
                f"sampling.num_frames ({self.data.sampling.num_frames})"
            )
        if self.data.crop.min_valid_frames < 0:
            raise ValueError(f"crop.min_valid_frames must be >= 0, got {self.data.crop.min_valid_frames}")

        # Normalization validation
        if self.data.normalization not in ("imagenet", "none"):
            raise ValueError(
                f"Invalid data.normalization: '{self.data.normalization}'. Must be 'imagenet' or 'none'."
            )

        # Model & Backbone validation
        bb = self.model.backbone
        valid_backbone_names = ("resnext50_32x4d", "efficientnet_b0")
        if bb.name not in valid_backbone_names:
            raise ValueError(f"Invalid model.backbone.name: '{bb.name}'. Expected one of {valid_backbone_names}.")
        if bb.weights not in (None, "IMAGENET1K_V1", "DEFAULT"):
            raise ValueError(
                f"Invalid model.backbone.weights: '{bb.weights}'. Must be 'IMAGENET1K_V1', 'DEFAULT', or None."
            )
        valid_freeze = ("none", "stem", "layer1", "layer2", "layer3", "early", "mid", "late", "all")
        if bb.freeze_strategy not in valid_freeze:
            raise ValueError(
                f"Invalid model.backbone.freeze_strategy: '{bb.freeze_strategy}'. Must be one of {valid_freeze}."
            )
        if bb.name == "resnext50_32x4d":
            if bb.output_dim != 2048:
                raise ValueError(f"Invalid model.backbone.output_dim: {bb.output_dim}. Expected 2048.")
            if bb.spatial_dim != 4:
                raise ValueError(f"Invalid model.backbone.spatial_dim: {bb.spatial_dim}. Expected 4.")
            if list(bb.input_size) != [112, 112]:
                raise ValueError(f"Invalid model.backbone.input_size: {bb.input_size}. Expected [112, 112].")
        elif bb.name == "efficientnet_b0":
            if bb.output_dim != 1280:
                raise ValueError(f"Invalid model.backbone.output_dim: {bb.output_dim}. Expected 1280.")
            if bb.spatial_dim not in (4, 7):
                raise ValueError(f"Invalid model.backbone.spatial_dim: {bb.spatial_dim}. Expected 4 or 7.")

        # Attention validation
        attn = self.model.attention
        if attn.in_channels != bb.output_dim:
            raise ValueError(
                f"model.attention.in_channels ({attn.in_channels}) must match "
                f"model.backbone.output_dim ({bb.output_dim})"
            )

        # Temporal & Classifier validation
        temp = self.model.temporal
        if temp.input_dim != attn.in_channels:
            raise ValueError(
                f"model.temporal.input_dim ({temp.input_dim}) must match "
                f"model.attention.in_channels ({attn.in_channels})"
            )
        clf = self.model.classifier
        if clf.in_features != temp.output_dim:
            raise ValueError(
                f"model.classifier.in_features ({clf.in_features}) must match "
                f"model.temporal.output_dim ({temp.output_dim})"
            )

        # Training & Scheduler cross-validation
        tr = self.training
        expected_mode = "min" if tr.monitor_metric == "val_loss" else "max"
        if tr.scheduler.type == "plateau" and tr.scheduler.mode != expected_mode:
            raise ValueError(
                f"scheduler.mode '{tr.scheduler.mode}' conflicts with monitor_metric '{tr.monitor_metric}' "
                f"(expected '{expected_mode}')."
            )

        # Device validation & auto-fallback
        valid_devices = ("cuda", "cpu", "mps")
        if self.device not in valid_devices:
            raise ValueError(f"Invalid device: '{self.device}'. Must be one of {valid_devices}.")

    def get_effective_device(self) -> str:
        """Return the effective device, falling back to 'cpu' if 'cuda' is requested but unavailable."""
        if self.device == "cuda":
            if HAS_TORCH and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.device

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


def _dict_to_config(d: dict[str, Any]) -> AppConfig:
    """Recursively reconstruct AppConfig from dictionary."""
    hardware_dict = d.get("hardware", {})
    hardware = HardwareConfig(**hardware_dict)

    data_dict = d.get("data", {})
    sampling_dict = data_dict.get("sampling", {})
    sampling = SamplingConfig(**sampling_dict)

    detection_dict = data_dict.get("detection", {})
    detection = DetectionConfig(**detection_dict)

    crop_dict = data_dict.get("crop", {})
    crop = CropConfig(**crop_dict)

    normalization = data_dict.get("normalization", "imagenet")

    data = DataConfig(
        sampling=sampling,
        detection=detection,
        crop=crop,
        normalization=normalization,
    )

    model_dict = d.get("model", {})
    backbone_dict = model_dict.get("backbone", {})
    backbone = BackboneConfig(**backbone_dict)
    attention_dict = model_dict.get("attention", {})
    attention = AttentionConfig(**attention_dict)
    temporal_dict = model_dict.get("temporal", {})
    temporal = TemporalConfig(**temporal_dict)
    classifier_dict = model_dict.get("classifier", {})
    classifier = ClassifierConfig(**classifier_dict)
    auxiliary_dict = model_dict.get("auxiliary", {})
    auxiliary = AuxiliaryConfig(**auxiliary_dict)
    model = ModelConfig(
        backbone=backbone,
        attention=attention,
        temporal=temporal,
        classifier=classifier,
        auxiliary=auxiliary,
    )

    storage_dict = d.get("storage", {})
    storage = StorageConfig(**storage_dict)

    training_dict = d.get("training", {})
    opt_dict = training_dict.get("optimizer", {})
    optimizer = OptimizerConfig(**opt_dict)
    sched_dict = training_dict.get("scheduler", {})
    scheduler = SchedulerConfig(**sched_dict)
    training_kwargs = {k: v for k, v in training_dict.items() if k not in ("optimizer", "scheduler")}
    training = TrainingConfig(
        optimizer=optimizer,
        scheduler=scheduler,
        **training_kwargs,
    )

    eval_dict = d.get("evaluation", {})
    evaluation = EvaluationConfig(**eval_dict)

    config = AppConfig(
        seed=d.get("seed", 42),
        device=d.get("device", "cuda"),
        hardware=hardware,
        data=data,
        model=model,
        storage=storage,
        training=training,
        evaluation=evaluation,
    )
    config.validate()
    return config


def load_config(path: Union[str, Path]) -> AppConfig:
    """Load configuration from a YAML file with cross-validation.

    Args:
        path: Path to the YAML file.

    Returns:
        Validated AppConfig instance.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    return _dict_to_config(raw_dict)


def save_config(config: AppConfig, path: Union[str, Path]) -> None:
    """Serialize AppConfig instance to a YAML file.

    Args:
        config: The AppConfig object.
        path: Destination YAML path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
