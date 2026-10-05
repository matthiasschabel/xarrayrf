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

from ._array_coordinates import ArrayCoordinates
from ._frame import ReferenceFrame
from ._transform import SupportsPoints
from ._validation import frozen_float_array

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

SINGLE_SAMPLE_TOLERANCE = 8 * float(np.finfo(np.float64).eps)
"""Rounding allowance for singleton matching, in multiples of coordinate magnitude.

Eight epsilons cover the observed few-ulp affine round trips. Inverse evaluation also supplies
an allowance based on its intermediate terms, so cancellation does not require a broad
relative support window.
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
    intervals: npt.NDArray[np.float64] | None = None


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


INTERVAL_TOLERANCE = 1e-9
"""Maximum sample/interval disagreement as a fraction of declared cell width."""

INTERVAL_ROUNDOFF_FACTOR = 8
"""Float64 epsilon multiples allowed at the bounds' magnitude for offset agreement.

Large origins, such as epoch seconds, can round more than the width-relative tolerance.
The agreement check uses the larger of the width tolerance and this roundoff allowance.
"""


def freeze_intervals(
    source: ArrayCoordinates,
    values: Mapping[str, npt.ArrayLike],
    intervals: Mapping[str, npt.ArrayLike] | None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Validate declared support against samples and freeze its float64 rows."""
    if intervals is None:
        return {}
    if not isinstance(intervals, Mapping):
        raise TypeError("intervals must be a mapping from source axis names to rows")
    unknown = set(intervals) - set(source.axes)
    if unknown:
        raise ValueError(f"intervals name unknown source axes {sorted(unknown)}")
    result = {}
    for name, rows in intervals.items():
        offset = source.sample_offset[source.axes.index(name)]
        if offset is None:
            raise ValueError(f"source axis {name!r} is point-sampled and cannot declare intervals")
        samples = np.asarray(values[name], dtype=np.float64)
        bounds = frozen_float_array(rows, field=f"intervals for source axis {name!r}")
        expected = (*samples.shape, 2)
        if samples.ndim > 1 or bounds.shape != expected:
            raise ValueError(f"intervals for source axis {name!r} must have shape {expected}")
        with np.errstate(over="ignore", invalid="ignore"):
            width = bounds[..., 1] - bounds[..., 0]
        if np.any(width <= 0) or not np.all(np.isfinite(width)):
            raise ValueError(f"intervals for source axis {name!r} need finite widths and lo < hi")
        with np.errstate(over="ignore", invalid="ignore"):
            disagreement = np.abs(samples - (bounds[..., 0] + offset * width))
        tolerance = np.maximum(
            INTERVAL_TOLERANCE * width,
            INTERVAL_ROUNDOFF_FACTOR * np.finfo(np.float64).eps * np.max(np.abs(bounds), axis=-1),
        )
        if not np.all(np.isfinite(samples)) or np.any(disagreement > tolerance):
            raise ValueError(f"intervals for source axis {name!r} disagree with sample_offset")
        if np.any((samples < bounds[..., 0]) | (samples > bounds[..., 1])):
            raise ValueError(f"source axis {name!r} sample lies outside its declared interval")
        result[name] = bounds
    return result


def singleton_step(sampling: AxisSampling, domain: Domain) -> float | None:
    """Use a declared singleton width as the fractional-position step in the cells domain."""
    if domain == "cells" and sampling.values.size == 1 and sampling.intervals is not None:
        return float(np.diff(sampling.intervals.reshape(1, 2), axis=-1)[0, 0])
    return sampling.step


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

    Declared intervals supply the outer coordinate bounds, converted to positions using the
    same coordinate inverse as sampling queries. A singleton interval supplies its width.

    An axis declaring point samples (``offset`` None) has no cells, so it reaches no further
    than its samples.

    Raises:
        ValueError: If the axis is empty, or declares cells but has a single sample without
            an interval, whose cell width no neighbour determines.
    """
    if sampling.intervals is not None:
        if sampling.values.size == 0:
            raise ValueError("an empty coordinate has no cells to locate")
        bounds = np.array([sampling.intervals[..., 0].min(), sampling.intervals[..., 1].max()])
        positions = coordinate_to_position(
            sampling.values, bounds, singleton_step(sampling, "cells"), extrapolate=True
        )
        return -float(positions.min()), float(positions.max()) - (sampling.values.size - 1)
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


def affine_roundoff(
    matrix: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Estimate affine evaluation roundoff from absolute terms before cancellation."""
    scale = np.abs(points) @ np.abs(matrix).T + np.abs(translation)
    roundoff: npt.NDArray[np.float64] = SINGLE_SAMPLE_TOLERANCE * (matrix.shape[1] + 1) * scale
    return roundoff


def coordinate_to_position(
    values: npt.NDArray[np.float64],
    coordinates: npt.NDArray[np.float64],
    step: float | None = None,
    extent: tuple[float, float] = (0.0, 0.0),
    *,
    extrapolate: bool = False,
    roundoff: npt.NDArray[np.float64] | float = 0.0,
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
    if size == 1 and step is None:
        scale = np.maximum(1.0, np.maximum(abs(float(values[0])), np.abs(coordinates)))
        tolerance = np.maximum(SINGLE_SAMPLE_TOLERANCE * scale, roundoff)
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
    values: npt.NDArray[np.float64],
    positions: npt.NDArray[np.float64],
    step: float | None = None,
) -> npt.NDArray[np.float64]:
    """Interpolate positions piecewise linearly, extending by the outer steps."""
    if values.size == 0:
        raise ValueError("an empty coordinate has no positions to locate")
    if values.size == 1:
        if step is not None:
            return np.asarray(values[0] + positions * step, dtype=np.float64)
        if np.any(np.abs(positions) > POSITION_SLACK):
            raise ValueError(
                "a single sample has no step for fractional positions or extrapolation"
            )
        return np.full(positions.shape, values[0])
    indices = np.clip(np.floor(positions), 0, values.size - 2).astype(np.intp)
    fraction = positions - indices
    low, high = values[indices], values[indices + 1]
    # Inside a cell the weighted form reproduces stored samples exactly (a difference form
    # loses them to cancellation across large magnitudes); outside, extend from the nearer
    # endpoint, where the weighted form's separate products could overflow.
    with np.errstate(over="ignore", invalid="ignore"):
        inside = (1 - fraction) * low + fraction * high
    local_step = high - low
    beyond = np.where(fraction < 0, low + fraction * local_step, high + (fraction - 1) * local_step)
    return np.asarray(np.where((fraction >= 0) & (fraction <= 1), inside, beyond))
