"""Import ITK transform files as transforms between named frames.

ITK writes transforms as text (``.tfm``), MATLAB (``.mat``) and HDF5 (``.h5``) files, and
displacement fields as NIfTI vector images. This module reads them in float64 with SciPy, h5py
and nibabel, without an ITK runtime, and wraps them as core transforms between frames the caller
names: identity is never inferred from a file. Install ``xarrayrf[itk]``.
"""

from __future__ import annotations

from ._import import SUPPORTED_TYPES, ItkImport, Report, read_transform

__all__ = ["SUPPORTED_TYPES", "ItkImport", "Report", "read_transform"]
