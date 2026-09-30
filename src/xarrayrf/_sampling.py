"""NumPy descriptions and per-axis sampling arithmetic shared by Grid and Geometry.

Geometry supplies coordinate values lazily when a query needs them; Grid supplies frozen values.
Neither the description nor the math needs xarray or pixels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from ._frame import ReferenceFrame
from ._transform import SupportsPoints

LATTICE_TOLERANCE = 1e-6
"""Default allowed deviation from uniform spacing, as a fraction of one step."""

type Domain = Literal["samples", "cells"]
"""Where values are defined: between the outer samples, or out to the edges of their cells."""

DOMAINS: tuple[Domain, ...] = ("samples", "cells")

POSITION_SLACK = 1e-9
"""Positions within this many steps outside the domain count as on its edge, not outside.

Round-trip arithmetic through a frame puts an edge sample a few ulps beyond its exact position;
this absorbs that without accepting any genuinely outside point.
"""

SINGLE_SAMPLE_TOLERANCE = 1e-9
"""Relative tolerance for matching a coordinate to a dimension's single sample.

With one sample there is no step to measure a position slack against, so the match is on the
coordinate value itself, relative to its magnitude (and absolute below 1).
"""

EXACT_STEP_TOLERANCE = 1e-12
"""Largest deviation from uniform spacing, in steps, for the exact-arithmetic inversion.

This is a numerical fast path, not a policy: below it, ``(coordinate - first) / step`` and the
piecewise-linear inversion agree to rounding, so the cheaper one is used. It is deliberately
independent of the tolerance a caller passes to ``Geometry.lattice``, which decides whether the
samples are *declared* a lattice.
"""


@dataclass(frozen=True)
class AxisSampling:
    """One source axis: a retained scalar, or one-dimensional coordinate values along a dim.

    ``step`` is the exact step when an xarray ``RangeIndex`` defines the coordinate, so uniform
    spacing is known rather than tested numerically.
    """

    axis: str
    dim: str | None
    values: npt.NDArray[np.float64]
    step: float | None = None


@dataclass(frozen=True)
class Sampling:
    """NumPy sampling description, prepared only when a query needs coordinate values."""

    transform: SupportsPoints
    dims: tuple[str, ...]
    sizes: Mapping[str, int]
    _axes: tuple[AxisSampling, ...] | Callable[[], tuple[AxisSampling, ...]]

    @cached_property
    def axes(self) -> tuple[AxisSampling, ...]:
        """Read coordinates once per prepared query, after metadata-only refusals."""
        return self._axes() if callable(self._axes) else self._axes

    @property
    def frame(self) -> ReferenceFrame:
        """The transform's target frame."""
        assert isinstance(self.transform.target, ReferenceFrame)
        return self.transform.target


def uniform_step(values: npt.NDArray[np.float64], tolerance: float) -> float | None:
    """Return the constant step of ``values``, or ``None`` if they are not uniformly spaced.

    Uniform means every value lies within ``tolerance`` steps of ``values[0] + step * i``, with
    ``step`` taken end to end. A single value has no step.
    """
    if values.size < 2:
        return None
    step = float(values[-1] - values[0]) / (values.size - 1)
    if step == 0.0:
        return None
    ideal = values[0] + step * np.arange(values.size)
    if float(np.max(np.abs(values - ideal))) > tolerance * abs(step):
        return None
    return step


def cell_extent(sampling: AxisSampling, offset: float | None) -> tuple[float, float]:
    """Return how far one axis's cells reach beyond its outer samples, in positions.

    The result is ``(before, after)``: the reach below position 0 and above position ``n - 1``.
    ``offset`` is measured toward higher coordinate values, so it is read backwards along a
    dimension whose coordinates descend.

    An axis declaring point samples (``offset`` None) has no cells, so it reaches no further
    than its samples.

    Raises:
        ValueError: If the axis declares cells but has a single sample, whose cell width no
            neighbour determines.
    """
    if offset is None:
        return 0.0, 0.0
    if sampling.values.size < 2:
        raise ValueError(
            f"source axis {sampling.axis!r} declares cells but has a single sample, so no "
            "neighbour determines its cell width, and physical thickness is not described; use "
            "the samples domain",
        )
    if sampling.values[-1] > sampling.values[0]:
        return offset, 1.0 - offset
    return 1.0 - offset, offset


def coordinate_to_position(
    values: npt.NDArray[np.float64],
    coordinates: npt.NDArray[np.float64],
    step: float | None = None,
    extent: tuple[float, float] = (0.0, 0.0),
    *,
    extrapolate: bool = False,
) -> npt.NDArray[np.float64]:
    """Invert one dimension's coordinate values: fractional positions, NaN outside.

    Uniformly spaced values use exact arithmetic, with ``step`` taken as given when an index
    defines it; nonuniform, strictly monotonic values use piecewise-linear inversion, extended
    beyond the outer samples by the outer steps. The domain is ``[-before, n - 1 + after]`` for
    ``extent = (before, after)``, from :func:`cell_extent`; the default is the samples alone.
    A single sample then admits only its own coordinate.

    Raises:
        ValueError: If the values are not strictly monotonic.
    """
    size = values.size
    if size == 0:
        raise ValueError("an empty coordinate has no positions to locate")
    before, after = extent
    if size == 1:
        tolerance = SINGLE_SAMPLE_TOLERANCE * max(1.0, abs(float(values[0])))
        return np.where(np.abs(coordinates - values[0]) <= tolerance, 0.0, np.nan)
    if step is None:
        step = uniform_step(values, tolerance=EXACT_STEP_TOLERANCE)
    if step is not None:
        positions = (coordinates - values[0]) / step
    else:
        differences = np.diff(values)
        indices = np.arange(size, dtype=np.float64)
        if np.all(differences > 0):
            ascending, labels = values, indices
        elif np.all(differences < 0):
            ascending, labels = values[::-1], indices[::-1]
        else:
            raise ValueError(
                "coordinate values are not strictly monotonic, so a coordinate does not determine "
                "a unique position",
            )
        positions = np.interp(coordinates, ascending, labels)
        low = coordinates < ascending[0]
        high = coordinates > ascending[-1]
        positions = np.where(
            low,
            labels[0]
            + (coordinates - ascending[0])
            * (labels[1] - labels[0])
            / (ascending[1] - ascending[0]),
            positions,
        )
        positions = np.where(
            high,
            labels[-1]
            + (coordinates - ascending[-1])
            * (labels[-1] - labels[-2])
            / (ascending[-1] - ascending[-2]),
            positions,
        )
    if extrapolate:
        return cast(npt.NDArray[np.float64], positions)
    lowest, highest = 0.0 - before, size - 1 + after
    outside = (positions < lowest - POSITION_SLACK) | (positions > highest + POSITION_SLACK)
    return np.where(outside, np.nan, np.clip(positions, lowest, highest))


def position_to_coordinate(
    values: npt.NDArray[np.float64], positions: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Interpolate positions piecewise linearly, extending by the outer steps."""
    if values.size == 0:
        raise ValueError("an empty coordinate has no positions to locate")
    if values.size == 1:
        if np.any(np.abs(positions) > POSITION_SLACK):
            raise ValueError(
                "a single sample has no step for fractional positions or extrapolation"
            )
        return np.full(positions.shape, values[0])
    indices = np.clip(np.floor(positions), 0, values.size - 2).astype(np.intp)
    return values[indices] + (positions - indices) * (values[indices + 1] - values[indices])
