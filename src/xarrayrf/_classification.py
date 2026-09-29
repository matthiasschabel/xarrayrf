"""Classify the geometry of constant affine maps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._lattice import Lattice
from ._transform import SupportsAffine, check_endpoint
from ._validation import real_float_array

type AffineClassName = Literal[
    "identity",
    "translation",
    "rotation",
    "rigid",
    "similarity",
    "scaled_rigid",
    "affine",
    "singular",
    "rectangular",
]

_CONTAINMENT: dict[str, frozenset[str]] = {
    "identity": frozenset(
        {"identity", "translation", "rotation", "rigid", "similarity", "scaled_rigid", "affine"}
    ),
    "translation": frozenset({"translation", "rigid", "similarity", "scaled_rigid", "affine"}),
    "rotation": frozenset({"rotation", "rigid", "similarity", "scaled_rigid", "affine"}),
    "rigid": frozenset({"rigid", "similarity", "scaled_rigid", "affine"}),
    "similarity": frozenset({"similarity", "scaled_rigid", "affine"}),
    "scaled_rigid": frozenset({"scaled_rigid", "affine"}),
    "affine": frozenset({"affine"}),
    "singular": frozenset({"singular"}),
    "rectangular": frozenset({"rectangular"}),
}


@dataclass(frozen=True, slots=True)
class AffineClass:
    """The narrowest geometric class satisfied by an affine map.

    Attributes:
        name: Geometric class of the matrix and translation.
        proper: Whether a square nonsingular matrix preserves handedness; otherwise None.
        scales: Column lengths for classes up to scaled rigid; otherwise None.
        residual: Largest dimensionless defining residual for the reported constrained class.
        same_units: Whether all source and target axes have one identical unit string.
    """

    name: AffineClassName
    proper: bool | None
    scales: tuple[float, ...] | None
    residual: float | None
    same_units: bool

    def at_most(self, name: AffineClassName) -> bool:
        """Test whether this class is contained in ``name``.

        Translation and rotation are incomparable. Singular and rectangular maps contain
        only themselves, and are outside the invertible affine class.

        Raises:
            ValueError: If ``name`` is not a known affine class.
        """
        if not isinstance(name, str) or name not in _CONTAINMENT:
            raise ValueError(f"unknown affine class {name!r}")
        return name in _CONTAINMENT[self.name]


def _nonnegative_tolerance(value: float, *, field: str) -> float:
    """Require a finite nonnegative threshold at the public boundary."""
    if not isinstance(value, int | float | np.integer | np.floating) or isinstance(value, bool):
        raise TypeError(f"{field} must be a real number, got {type(value).__name__}")
    try:
        threshold = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite and >= 0, got {value!r}") from error
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"{field} must be finite and >= 0, got {value!r}")
    return threshold


def affine_class(
    affine: SupportsAffine | Lattice,
    *,
    tolerance: float = 1e-6,
    offset_tolerance: float = 1e-9,
) -> AffineClass:
    """Classify a constant affine transform or lattice by its resulting geometry.

    The class describes the matrix alone when source and target units differ. For a lattice,
    positions have dimensionless units; its column lengths are voxel spacings when the frame
    has a common physical unit. Reflections are represented by ``proper=False``.

    Args:
        affine: Affine transform or lattice to classify.
        tolerance: Maximum dimensionless matrix residual for a constrained class.
        offset_tolerance: Maximum absolute translation for identity or rotation.

    Returns:
        The narrowest class meeting every containing class's matrix constraints.

    Raises:
        TypeError: If ``affine`` is neither a lattice nor a valid affine transform, or a
            tolerance is not a real number.
        ValueError: If a tolerance is negative or nonfinite, or a transform's coefficients
            have an invalid shape or nonfinite values.
    """
    tolerance = _nonnegative_tolerance(tolerance, field="tolerance")
    offset_tolerance = _nonnegative_tolerance(offset_tolerance, field="offset_tolerance")
    if isinstance(affine, Lattice):
        # A Lattice validated its shapes and finiteness at construction.
        matrix = affine.matrix
        translation = affine.origin
        same_units = all(unit == "1" for unit in affine.frame.units)
        expected = (len(affine.frame.axes), len(affine.dims))
    elif isinstance(affine, SupportsAffine):
        source = check_endpoint(affine.source, field="affine.source")
        target = check_endpoint(affine.target, field="affine.target")
        matrix = real_float_array(affine.matrix, field="affine.matrix")
        translation = real_float_array(affine.translation, field="affine.translation")
        units = (*source.units, *target.units)
        same_units = bool(units) and units[0] is not None and len(set(units)) == 1
        expected = (len(target.axes), len(source.axes))
    else:
        raise TypeError(
            f"affine must implement SupportsAffine or be a Lattice, got {type(affine).__name__}"
        )
    if matrix.shape != expected:
        raise ValueError(f"affine.matrix must have shape {expected}, got {matrix.shape}")
    if translation.shape != (expected[0],):
        raise ValueError(
            f"affine translation must have shape {(expected[0],)}, got {translation.shape}"
        )

    rows, columns = matrix.shape
    if rows != columns:
        return AffineClass("rectangular", None, None, None, same_units)
    scales_array = np.linalg.norm(matrix, axis=0)
    if np.any(scales_array == 0):
        return AffineClass("singular", None, None, None, same_units)
    normalized = matrix / scales_array
    if np.linalg.matrix_rank(normalized) < columns:
        return AffineClass("singular", None, None, None, same_units)

    proper = bool(np.linalg.det(matrix) > 0)
    orthogonality = float(np.max(np.abs(normalized.T @ normalized - np.eye(columns))))
    if orthogonality > tolerance:
        return AffineClass("affine", proper, None, None, same_units)
    scales = tuple(float(value) for value in scales_array)
    uniformity = float(np.max(np.abs(scales_array / np.mean(scales_array) - 1)))
    if uniformity > tolerance:
        return AffineClass("scaled_rigid", proper, scales, orthogonality, same_units)

    gram = float(np.max(np.abs(matrix.T @ matrix - np.eye(columns))))
    determinant = abs(abs(float(np.linalg.det(matrix))) - 1)
    if gram > tolerance or determinant > tolerance:
        return AffineClass("similarity", proper, scales, max(orthogonality, uniformity), same_units)

    identity = float(np.max(np.abs(matrix - np.eye(columns))))
    offset = float(np.max(np.abs(translation)))
    if identity <= tolerance:
        name: AffineClassName = "identity" if offset <= offset_tolerance else "translation"
        residual = max(orthogonality, uniformity, gram, determinant, identity)
    else:
        name = "rotation" if offset <= offset_tolerance else "rigid"
        residual = max(orthogonality, uniformity, gram, determinant)
    return AffineClass(name, proper, scales, residual, same_units)
