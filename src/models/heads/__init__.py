"""Auxiliary and specialized classification heads for deepfake detection."""

from src.models.heads.auxiliary import (
    AuxiliaryManipulationHead,
    MidLevelSpatialHead,
    MANIPULATION_FAMILY_MAPPING,
    FAMILY_NAMES,
)

__all__ = [
    "AuxiliaryManipulationHead",
    "MidLevelSpatialHead",
    "MANIPULATION_FAMILY_MAPPING",
    "FAMILY_NAMES",
]
