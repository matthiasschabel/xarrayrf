"""Import elastix registration results as transforms between named frames.

elastix writes a result as parameter maps, one per stage, applied initial stage first and
mapping fixed-image points to moving-image points. This module reads the maps or files with
NumPy alone, so no ITK runtime is needed, and wraps each stage as a core transform between
frames the caller names: identity is never inferred from a file.
"""

from __future__ import annotations

from ._import import ElastixImport, Report, transform_from_parameter_maps
from ._parameter_files import (
    NO_INITIAL_TRANSFORM,
    ParameterMap,
    parse_parameter_text,
    read_parameter_file,
    read_parameter_files,
)

__all__ = [
    "NO_INITIAL_TRANSFORM",
    "ElastixImport",
    "ParameterMap",
    "Report",
    "parse_parameter_text",
    "read_parameter_file",
    "read_parameter_files",
    "transform_from_parameter_maps",
]
