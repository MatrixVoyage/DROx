"""Model architectures, backbones, attention modules, temporal models, and detectors."""

from src.models.backbones.resnext import ResNeXt50Backbone
from src.models.attention import (
    CBAMAttention,
    ChannelAttention,
    IdentityPooling,
    SoftAttention,
    SpatialAttention,
    build_attention_module,
)
from src.models.temporal import LSTMTemporalModel, build_temporal_model
from src.models.classifier import ClassifierHead
from src.models.detector import DeepfakeDetector, DetectorOutput
from src.models.m15_cross_attention_detector import (
    CUDABlockDCTCompression,
    CrossAttentionTemporalSpatialFusion,
    M15CrossAttentionDetector,
    M15DetectorOutput,
    M15Loss,
    M15RobustLoss,
)
from src.models.m15_ablation_detector import (
    LinearTemporalSpatialFusion,
    M15AblationNoGateDetector,
)

__all__ = [
    "ResNeXt50Backbone",
    "SoftAttention",
    "ChannelAttention",
    "SpatialAttention",
    "CBAMAttention",
    "IdentityPooling",
    "build_attention_module",
    "LSTMTemporalModel",
    "build_temporal_model",
    "ClassifierHead",
    "DeepfakeDetector",
    "DetectorOutput",
    "CrossAttentionTemporalSpatialFusion",
    "M15CrossAttentionDetector",
    "M15DetectorOutput",
    "M15Loss",
    "M15RobustLoss",
    "CUDABlockDCTCompression",
    "LinearTemporalSpatialFusion",
    "M15AblationNoGateDetector",
]

