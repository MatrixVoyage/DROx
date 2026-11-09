"""Dataset-specific adapters for manifest construction and metadata extraction."""

from src.data.adapters.celeb_df import (
    build_celebdf_manifest,
    extract_celebdf_group_id,
    parse_celebdf_test_list,
)
from src.data.adapters.dfdc import (
    build_dfdc_manifest_from_metadata,
    extract_dfdc_group_id,
)
from src.data.adapters.ffpp import (
    build_ffpp_manifest_from_paths,
    extract_ffpp_group_id,
)

__all__ = [
    "extract_ffpp_group_id",
    "build_ffpp_manifest_from_paths",
    "extract_celebdf_group_id",
    "parse_celebdf_test_list",
    "build_celebdf_manifest",
    "extract_dfdc_group_id",
    "build_dfdc_manifest_from_metadata",
]

