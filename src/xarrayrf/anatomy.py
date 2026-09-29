"""Canonical anatomical directions shared by format adapters."""

from __future__ import annotations

from collections.abc import Sequence

from ._coordinate_system import CoordinateSystem
from ._vocabulary import DirectionVocabulary

VOCABULARY = DirectionVocabulary(
    "ome-ngff:rfc-4:anatomical",
    (
        "left-to-right",
        "anterior-to-posterior",
        "inferior-to-superior",
        "dorsal-to-ventral",
        "proximal-to-distal",
        "dorsal-to-palmar",
        "dorsal-to-plantar",
        "rostral-to-caudal",
        "cranial-to-caudal",
        "superficial-to-deep",
        "apical-to-basal",
        "apex-to-base",
    ),
)
"""OME-NGFF RFC-4 anatomical direction vocabulary."""

RAS = ("left-to-right", "posterior-to-anterior", "inferior-to-superior")
"""Directions in which the three RAS coordinate values increase."""

LPS = ("right-to-left", "anterior-to-posterior", "inferior-to-superior")
"""Directions in which the three LPS coordinate values increase."""


def patient_coordinate_system(
    orientation: Sequence[str], unit: str, *, axes: Sequence[str] = ("x", "y", "z")
) -> CoordinateSystem:
    """Construct three oriented patient spatial axes.

    Args:
        orientation: Three RFC-4 direction tokens, one per axis.
        unit: Unit shared by all three axes.
        axes: Names of the three axes, defaulting to ``x``, ``y``, ``z``.

    Returns:
        An oriented Cartesian coordinate system with spatial axis types.

    Raises:
        TypeError: If an argument has the wrong Python type.
        ValueError: If the axis declarations are invalid.
    """
    return CoordinateSystem(
        axes,
        (unit,) * 3,
        axis_types=("space",) * 3,
        vocabulary=VOCABULARY,
        orientation=orientation,
    )
