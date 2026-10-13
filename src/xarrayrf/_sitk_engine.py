"""SimpleITK evaluation of dense displacement fields on target lattices, where ITK agrees.

ITK's ``DisplacementFieldTransform`` interpolates a vector image linearly, holds the edge
sample in the outer half cells and returns zero displacement outside ``[-0.5, n - 0.5)`` along
each axis: exactly a :class:`~xarrayrf.DisplacementFieldTransform` whose grid has centred
cells (``sample_offset`` 0.5 on every axis) under ``outside="identity"``. Its C++ loop
evaluates a target slab about ten times faster than three SciPy interpolations.

Everything happens in the field's index space, so no frame coordinate or origin reaches ITK:
the field image has unit spacing at the origin, holds displacements in index units, and the
reference image is the target lattice as seen in field indices. ITK orders axes x-fastest, the
reverse of a C-ordered array, so index axes are reversed on the way in and out.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

ENABLED = True
"""Tests turn the engine off to compare against the NumPy path."""

ORTHOGONALITY_TOLERANCE = 1e-12
"""Relative departure from orthogonal columns that an ITK direction matrix may absorb."""


def simpleitk() -> ModuleType | None:
    """SimpleITK when enabled and importable, else None."""
    if not ENABLED:
        return None
    try:
        import SimpleITK
    except ImportError:
        return None
    module: ModuleType = SimpleITK
    return module


def direction_and_spacing(
    linear: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Split a square map into orthonormal columns times positive spacings, or None."""
    if linear.shape[0] != linear.shape[1]:
        return None
    spacing = np.linalg.norm(linear, axis=0)
    if not np.all(spacing > 0):
        return None
    direction = linear / spacing
    if np.max(np.abs(direction.T @ direction - np.eye(len(spacing)))) > ORTHOGONALITY_TOLERANCE:
        return None
    return direction, spacing


def index_transform(sitk: ModuleType, index_field: npt.NDArray[np.float64]) -> Any:
    """ITK's displacement field transform of displacements in index units ``(..., D)``.

    ITK's physical axis a is field index axis ``D - 1 - a``, so components reverse too. The
    image is a float64 copy, ``8 D`` bytes per field sample, kept by the caller for reuse.
    """
    image = sitk.GetImageFromArray(np.ascontiguousarray(index_field[..., ::-1]), isVector=True)
    return sitk.DisplacementFieldTransform(image)


class FieldEngine:
    """One field's ITK transform in index space and a target lattice's slab geometry.

    ``index_displacement`` maps target lattice rows ``[first, stop)`` (whole rows of the first
    target dimension) to displacements in field index units, ``(D, n)``.
    """

    def __init__(
        self,
        sitk: ModuleType,
        transform: Any,
        linear: npt.NDArray[np.float64],
        offset: npt.NDArray[np.float64],
        target_shape: tuple[int, ...],
        direction: npt.NDArray[np.float64],
        spacing: npt.NDArray[np.float64],
    ) -> None:
        self._sitk = sitk
        dims = len(target_shape)
        self._transform = transform
        self._linear = linear
        self._offset = offset
        self._shape = target_shape
        reverse = np.arange(dims)[::-1]
        # Reference image axis a is target dimension dims - 1 - a, in reversed field indices.
        self._direction = direction[reverse][:, reverse]
        self._spacing = spacing[reverse]

    def index_displacement(self, first: int, stop: int, threads: int) -> npt.NDArray[np.float64]:
        sitk = self._sitk
        reverse = slice(None, None, -1)
        origin = self._offset + self._linear[:, 0] * first
        reference = sitk.Image([*self._shape[1:][::-1], stop - first], sitk.sitkUInt8)
        reference.SetOrigin(origin[reverse].tolist())
        reference.SetSpacing(self._spacing.tolist())
        reference.SetDirection(self._direction.ravel().tolist())
        sampler = sitk.TransformToDisplacementFieldFilter()
        sampler.SetReferenceImage(reference)
        sampler.SetOutputPixelType(sitk.sitkVectorFloat64)
        # The pool threader ignores a thread count alone; one work unit keeps ITK on the
        # calling thread, whose parallelism is xarrayrf's.
        sampler.SetNumberOfThreads(threads)
        sampler.SetNumberOfWorkUnits(threads)
        output = sampler.Execute(self._transform)
        # The array view is C-ordered as the target (first dimension slowest); components come
        # back in ITK's reversed order.
        values = sitk.GetArrayViewFromImage(output).reshape(-1, len(self._shape))
        rows: npt.NDArray[np.float64] = np.ascontiguousarray(values[:, reverse].T)
        return rows
