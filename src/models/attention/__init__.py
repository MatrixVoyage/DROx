"""Attention modules and factory for spatial-temporal refinement."""

from src.models.attention.cbam import CBAMAttention
from src.models.attention.channel import ChannelAttention
from src.models.attention.factory import build_attention_module
from src.models.attention.identity import IdentityPooling
from src.models.attention.soft import SoftAttention
from src.models.attention.spatial import SpatialAttention

__all__ = [
    "SoftAttention",
    "ChannelAttention",
    "SpatialAttention",
    "CBAMAttention",
    "IdentityPooling",
    "build_attention_module",
]
