"""Array positions and frame points for a :class:`~xarrayrf.Geometry`.

Locating frame points among an array's samples (the exact inverse of the transform followed by
a per-axis coordinate-to-position inversion), the domain a location may reach into, and the
lattice a uniformly sampled geometry forms. Shared by the geometry view, resampling, coincidence
and lazy frame coordinates, which all read the same public geometry attributes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._array_coordinates import ArrayCoordinates
from ._sampling import (
    DOMAINS,
    AxisSampling,
    Domain,
    cell_extent,
    coordinate_to_position,
    sample_axes,
    uniform_step,
)
from ._transform import SupportsAffine, SupportsInverse, check_transform

if TYPE_CHECKING:
    from ._geometry import Geometry

type LocatedAxis = tuple[AxisSampling, int, float | None]
"""A geometry dimension's source axis sampling, its index among the source axes, its offset."""

type Locator = Callable[[npt.ArrayLike], npt.NDArray[np.float64]]
"""Frame points ``(..., M)`` to positions ``(..., D)``, NaN for a point outside the domain."""


def check_domain(domain: object) -> None:
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")


def located_axes(geometry: Geometry) -> list[LocatedAxis]:
    """Return each geometry dimension's source axis, its index and its sample offset.

    Raises:
        ValueError: If a source axis is a retained scalar, or two share a dimension.
    """
    source = geometry.transform.source
    assert isinstance(source, ArrayCoordinates)
    samplings = sample_axes(geometry.array, source.axes)
    by_dim: dict[str, int] = {}
    for index, sampling in enumerate(samplings):
        if sampling.dim is None:
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


def extents(geometry: Geometry, domain: Domain) -> list[tuple[float, float]]:
    """Return how far the domain reaches beyond the outer samples, per geometry dimension.

    Raises:
        ValueError: If ``domain`` is unknown, or a cells-domain axis has a single sample.
    """
    check_domain(domain)
    geometry.coordinate_dependencies  # noqa: B018  # revalidates the current coordinates
    return reach(located_axes(geometry), domain)


def locator(geometry: Geometry, domain: Domain = "samples", slack: float = 0.0) -> Locator:
    """Validate once and return frame points -> positions ``(..., D)``, NaN outside.

    The inverse and the coordinate values are prepared here, so repeated calls, such as one
    per resampling block, pay only for the arithmetic. ``slack`` widens the domain by that
    many steps on every side, for comparisons within a tolerance.

    Raises:
        TypeError: If the transform has no inverse.
        ValueError: As :func:`located_axes` and :func:`extents`.
    """
    check_domain(domain)
    geometry.coordinate_dependencies  # noqa: B018  # revalidates the current coordinates
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
                    sampling.values, coordinates[..., axis], sampling.step, extent
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
    geometry: Geometry, order: tuple[str, ...], tolerance: float, *, single_samples: bool
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
    for index, sampling in enumerate(sample_axes(geometry.array, transform.source.axes)):
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
