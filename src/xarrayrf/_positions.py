"""Positions, frame points and lattices from a neutral sampling description.

Forward and inverse queries share piecewise-linear coordinate mapping and samples/cells bounds.
Grid, the live geometry view, resampling and coincidence all use this NumPy math.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from ._array_coordinates import ArrayCoordinates
from ._lattice import Lattice
from ._sampling import (
    DOMAINS,
    POSITION_SLACK,
    AxisSampling,
    Domain,
    Sampling,
    cell_extent,
    coordinate_to_position,
    position_to_coordinate,
    uniform_step,
)
from ._transform import SupportsAffine, SupportsInverse, check_transform
from ._validation import _check_str_sequence, real_float_array

type Outside = Literal["raise", "nan", "extrapolate"]

type LocatedAxis = tuple[AxisSampling, int, float | None]
"""A geometry dimension's source axis sampling, its index among the source axes, its offset."""

type Locator = Callable[[npt.ArrayLike], npt.NDArray[np.float64]]
"""Frame points ``(..., M)`` to positions ``(..., D)``, NaN for a point outside the domain."""


def check_domain(domain: object) -> None:
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")


def located_axes(geometry: Sampling, *, scalars: bool = False) -> list[LocatedAxis]:
    """Return each geometry dimension's source axis, its index and its sample offset.

    Raises:
        ValueError: If a source axis is a retained scalar, or two share a dimension.
    """
    source = geometry.transform.source
    assert isinstance(source, ArrayCoordinates)
    samplings = geometry.axes
    by_dim: dict[str, int] = {}
    for index, sampling in enumerate(samplings):
        if sampling.dim is None:
            if scalars:
                continue
            raise ValueError(
                f"source axis {sampling.axis!r} is a retained scalar: locating a point on a "
                "selected plane needs a projection policy, which is a separate operation",
            )
        if sampling.dim in by_dim:
            raise ValueError(
                f"dimension {sampling.dim!r} carries more than one source axis, so a "
                "position is not determined by a single coordinate",
            )
        by_dim[sampling.dim] = index
    return [
        (samplings[by_dim[dim]], by_dim[dim], source.sample_offset[by_dim[dim]])
        for dim in geometry.dims
    ]


def reach(located: Sequence[LocatedAxis], domain: Domain) -> list[tuple[float, float]]:
    """Per located axis, how far the domain reaches beyond its outer samples, in positions."""
    if domain == "samples":
        return [(0.0, 0.0)] * len(located)
    return [cell_extent(sampling, offset) for sampling, _, offset in located]


def extents(geometry: Sampling, domain: Domain) -> list[tuple[float, float]]:
    """Return how far the domain reaches beyond the outer samples, per geometry dimension.

    Raises:
        ValueError: If ``domain`` is unknown, or a cells-domain axis has a single sample.
    """
    check_domain(domain)
    return reach(located_axes(geometry), domain)


def locator(
    geometry: Sampling,
    domain: Domain = "samples",
    slack: float = 0.0,
    *,
    extrapolate: bool = False,
) -> Locator:
    """Validate once and return frame points -> positions ``(..., D)``, NaN outside.

    The inverse and the coordinate values are prepared here, so repeated calls, such as one
    per resampling block, pay only for the arithmetic. ``slack`` widens the domain by that
    many steps on every side, for comparisons within a tolerance.

    Raises:
        TypeError: If the transform has no inverse.
        ValueError: As :func:`located_axes` and :func:`extents`.
    """
    check_domain(domain)
    transform = geometry.transform
    if not isinstance(transform, SupportsInverse):
        raise TypeError(f"{type(transform).__name__} has no inverse, so points cannot be located")
    located = located_axes(geometry)
    order = [
        (sampling, axis, (before + slack, after + slack))
        for (sampling, axis, _), (before, after) in zip(
            located, reach(located, domain), strict=True
        )
    ]
    inverse = check_transform(transform.inverse())

    def locate(points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        coordinates = inverse.transform_point(points)
        positions = np.stack(
            [
                coordinate_to_position(
                    sampling.values,
                    coordinates[..., axis],
                    sampling.step,
                    extent,
                    extrapolate=extrapolate,
                )
                for sampling, axis, extent in order
            ],
            axis=-1,
        )
        # A point outside along any dimension is outside as a whole.
        positions[np.isnan(positions).any(axis=-1)] = np.nan
        return positions

    return locate


def sample_columns(
    samplings: Sequence[AxisSampling], indices: npt.NDArray[np.intp], dims: Sequence[str]
) -> npt.NDArray[np.float64]:
    """Gather source-axis inputs for samples given as integer positions.

    ``indices`` is shaped ``(len(dims), n)``, one row of positions per dimension in ``dims``.
    Returns ``(n, len(samplings))``: a retained scalar repeats its value, an axis along a
    dimension reads its coordinate at that dimension's positions.
    """
    count = indices.shape[1]
    columns = [
        np.full(count, float(sampling.values))
        if sampling.dim is None
        else sampling.values[indices[dims.index(sampling.dim)]]
        for sampling in samplings
    ]
    return np.stack(columns, axis=-1)


def lattice_parts(
    geometry: Sampling, order: tuple[str, ...], tolerance: float, *, single_samples: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the lattice origin and columns for an affine geometry.

    A single-sample dimension gets a zero column when ``single_samples`` allows it: a target
    that never moves along such a dimension needs no step there, so resampling allows it; a
    public lattice does not, because its spacing would be undefined.

    Raises:
        ValueError: If a coordinate is a multidimensional field, a dimension has a single
            sample (unless allowed), or a coordinate is not uniformly spaced within
            ``tolerance``.
    """
    transform = geometry.transform
    assert isinstance(transform, SupportsAffine)
    matrix = transform.matrix
    columns = np.zeros((matrix.shape[0], len(order)))
    base = np.zeros(matrix.shape[1])
    for index, sampling in enumerate(geometry.axes):
        if sampling.dim is None:
            base[index] = float(sampling.values)
            continue
        if sampling.values.size < 2:
            if single_samples:
                base[index] = sampling.values[0]
                continue
            raise ValueError(
                f"dimension {sampling.dim!r} has a single sample, so its step is not "
                "determined; a lattice needs at least two samples along each dimension",
            )
        step = (
            sampling.step if sampling.step is not None else uniform_step(sampling.values, tolerance)
        )
        if step is None:
            raise ValueError(
                f"coordinate {sampling.axis!r} is not uniformly spaced within {tolerance} of "
                f"a step along {sampling.dim!r}; nonuniform samples form no lattice",
            )
        base[index] = sampling.values[0]
        columns[:, order.index(sampling.dim)] += matrix[:, index] * step
    return matrix @ base + transform.translation, columns


def check_outside(outside: object) -> None:
    """Validate the policy for queries outside the sampling domain."""
    if outside not in ("raise", "nan", "extrapolate"):
        raise ValueError(f"outside must be 'raise', 'nan' or 'extrapolate', got {outside!r}")


def positions_at(
    sampling: Sampling, points: npt.ArrayLike, *, domain: Domain, outside: Outside
) -> npt.NDArray[np.float64]:
    """Locate frame points with the requested outside policy."""
    check_outside(outside)
    positions = locator(sampling, domain, extrapolate=outside == "extrapolate")(points)
    if outside == "raise" and np.isnan(positions).any():
        raise ValueError(
            f"points lie outside the {domain} domain; pass outside='nan' to mark them instead"
        )
    return positions


def points_at(
    sampling: Sampling, positions: npt.ArrayLike, *, domain: Domain, outside: Outside
) -> npt.NDArray[np.float64]:
    """Map fractional positions into the frame using each axis's actual coordinates."""
    check_domain(domain)
    check_outside(outside)
    values = real_float_array(positions, field="positions")
    if values.ndim == 0 or values.shape[-1] != len(sampling.dims):
        raise ValueError(
            f"positions must have shape (..., {len(sampling.dims)}), got {values.shape}"
        )
    located = located_axes(sampling, scalars=True)
    mask = np.zeros(values.shape[:-1], dtype=bool)
    bounded = values.copy()
    for column, ((axis, _, _), (before, after)) in enumerate(
        zip(located, reach(located, domain), strict=True)
    ):
        low, high = -before, axis.values.size - 1 + after
        mask |= (values[..., column] < low - POSITION_SLACK) | (
            values[..., column] > high + POSITION_SLACK
        )
        if outside != "extrapolate":
            bounded[..., column] = np.clip(values[..., column], low, high)
    if outside == "raise" and mask.any():
        raise ValueError(
            f"positions lie outside the {domain} domain; pass outside='nan' to mark them instead"
        )
    # Invalid rows are evaluated at a sample so they do not demand a nonexistent singleton step.
    if outside == "nan":
        bounded[mask] = 0.0
    coordinates = np.stack(
        [
            np.full(values.shape[:-1], float(axis.values))
            if axis.dim is None
            else position_to_coordinate(axis.values, bounded[..., sampling.dims.index(axis.dim)])
            for axis in sampling.axes
        ],
        axis=-1,
    )
    points = sampling.transform.transform_point(coordinates)
    if outside == "nan":
        points = np.where(mask[..., None], np.nan, points)
    return points


def check_position(value: object, *, dim: str, size: int) -> int:
    """Validate one zero-based, non-negative, in-bounds sample position."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(
            f"index for dimension {dim!r} must be a Python or NumPy integer, got "
            f"{type(value).__name__}; a boolean is not read as 0 or 1",
        )
    position = int(value)
    if not 0 <= position < size:
        raise IndexError(
            f"index {position} is out of range for dimension {dim!r} of current size {size}; "
            "positions are zero-based, and a negative position is not read from the end",
        )
    return position


def check_tolerance(tolerance: float) -> float:
    """Validate a coincidence tolerance, in fractions of a step."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, int | float | np.floating):
        raise TypeError(f"tolerance must be a real number, got {type(tolerance).__name__}")
    if not (np.isfinite(tolerance) and 0.0 <= tolerance < 0.5):
        raise ValueError(f"tolerance must lie in [0, 0.5) steps, got {tolerance!r}")
    return float(tolerance)


def lattice(sampling: Sampling, dims: Sequence[str] | None, *, tolerance: float) -> Lattice:
    """Construct a lattice from an affine, uniformly sampled description."""
    if not isinstance(sampling.transform, SupportsAffine):
        raise TypeError(
            f"a lattice needs an affine transform; {type(sampling.transform).__name__} is not one"
        )
    order = sampling.dims if dims is None else _check_str_sequence(dims, field="dims")
    if sorted(order) != sorted(sampling.dims) or len(set(order)) != len(order):
        raise ValueError(
            f"dims must name each geometry dimension {sampling.dims} once, got {order}"
        )
    origin, columns = lattice_parts(sampling, order, tolerance, single_samples=False)
    degenerate = [order[j] for j in range(len(order)) if not np.any(columns[:, j])]
    if degenerate:
        raise ValueError(f"dimensions {degenerate} map to no displacement in the frame")
    return Lattice(frame=sampling.frame, dims=order, origin=origin, matrix=columns)
