"""Sampling declarations with frozen coordinates, independent of pixels and xarray."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from ._array_coordinates import ArrayCoordinates
from ._coincidence import is_coincident
from ._frame import ReferenceFrame
from ._lattice import Lattice
from ._positions import (
    Outside,
    check_position,
    check_tolerance,
    lattice,
    points_at,
    positions_at,
    transform_points,
)
from ._sampling import LATTICE_TOLERANCE, AxisSampling, Domain, Sampling, freeze_intervals
from ._transform import SupportsPoints, check_transform
from ._validation import check_names, check_str, frozen_coordinate_array

type Coordinate = npt.ArrayLike | tuple[str, npt.ArrayLike]


@dataclass(frozen=True)
class _AxisCoordinate:
    axis: str
    dim: str | None
    values: npt.NDArray[np.int64] | npt.NDArray[np.float64]


class Grid:
    """A transform and frozen 0-D/1-D source coordinates describing sampling without pixels.

    Coordinate entries are ``name: (dim, values)`` for varying axes and ``name: value``
    for retained scalars. Integers are stored as int64, floats as float64. Units come
    from the transform's source. Optional intervals map source axis names to finite
    [lo, hi] rows agreeing with each sample's declared offset.

    The transform is retained by reference. Its endpoints, behavior and scalar-boolean
    equality must stay stable for this Grid's lifetime. Equality reflects the transform's
    chosen declaration identity, which may be object identity. Built-in affine transforms
    are immutable; composite stability depends on their members. Hashing a Grid additionally
    requires a hashable transform with a stable, equality-consistent hash. Unhashable
    transforms support queries and equality, but ``hash(grid)`` raises TypeError.
    """

    __slots__ = ("_axes", "_dims", "_intervals", "_transform")

    _transform: SupportsPoints
    _axes: tuple[_AxisCoordinate, ...]
    _dims: tuple[str, ...]

    def __init__(
        self,
        transform: SupportsPoints,
        coordinates: Mapping[str, Coordinate],
        *,
        intervals: Mapping[str, npt.ArrayLike] | None = None,
    ) -> None:
        """Validate and copy the coordinates into immutable storage.

        Raises:
            TypeError: If coordinates is not a mapping or its values are not real arrays,
                a dimension is not a string, or transform does not implement SupportsPoints.
            ValueError: If endpoints are not ArrayCoordinates -> ReferenceFrame, coordinates
                do not name exactly the source axes, values have the wrong rank or are not
                finite, dimensions repeat, integers exceed int64, or intervals violate shape,
                bounds or sample-offset agreement.
        """
        check_transform(transform)
        if not isinstance(transform.source, ArrayCoordinates):
            raise ValueError("Grid needs a transform from ArrayCoordinates")
        if not isinstance(transform.target, ReferenceFrame):
            raise ValueError("Grid needs a transform into a ReferenceFrame")
        if not isinstance(coordinates, Mapping):
            raise TypeError(f"coordinates must be a mapping, got {type(coordinates).__name__}")
        if set(coordinates) != set(transform.source.axes):
            raise ValueError(
                f"coordinates must name exactly the source axes {transform.source.axes}"
            )
        axes: dict[str, _AxisCoordinate] = {}
        dims = []
        for name, entry in coordinates.items():
            if isinstance(entry, tuple) and len(entry) == 2:
                dim = check_str(entry[0], field=f"coordinate {name!r} dimension")
                values = frozen_coordinate_array(entry[1], field=f"coordinate {name!r}")
                if values.ndim != 1:
                    raise ValueError(f"coordinate {name!r} values must be one-dimensional")
                if dim in dims:
                    raise ValueError(
                        f"dimension {dim!r} carries more than one source axis; a grid has at "
                        "most one varying coordinate per dimension"
                    )
                dims.append(dim)
            else:
                if (
                    isinstance(entry, Sequence)
                    and not isinstance(entry, str)
                    and len(entry) == 2
                    and isinstance(entry[0], str)
                ):
                    raise TypeError(
                        f"coordinate {name!r} looks like a (dim, values) pair; pass it as a tuple"
                    )
                dim = None
                values = frozen_coordinate_array(entry, field=f"coordinate {name!r}")
                if values.ndim != 0:
                    raise ValueError(f"coordinate {name!r} must be a scalar or (dim, values)")
            axes[name] = _AxisCoordinate(name, dim, values)
        self._dims = check_names(dims, field="dims", allow_empty=True)
        self._transform = transform
        self._axes = tuple(axes[name] for name in transform.source.axes)
        self._intervals = freeze_intervals(
            transform.source, {name: axis.values for name, axis in axes.items()}, intervals
        )

    @property
    def transform(self) -> SupportsPoints:
        """The coordinate transform from array coordinates to the frame."""
        return self._transform

    @property
    def frame(self) -> ReferenceFrame:
        """The frame the samples are located in."""
        assert isinstance(self._transform.target, ReferenceFrame)
        return self._transform.target

    @property
    def dims(self) -> tuple[str, ...]:
        """Varying dimensions in coordinate order."""
        return self._dims

    @property
    def sizes(self) -> Mapping[str, int]:
        """The read-only size of each varying dimension."""
        by_dim = {axis.dim: axis.values.size for axis in self._axes if axis.dim is not None}
        return MappingProxyType({dim: by_dim[dim] for dim in self._dims})

    @property
    def coordinates(self) -> Mapping[str, Coordinate]:
        """Read-only coordinate views over immutable buffers; scalar axes are Python values."""
        by_dim = {axis.dim: axis for axis in self._axes if axis.dim is not None}
        ordered = [by_dim[dim] for dim in self._dims]
        ordered += [axis for axis in self._axes if axis.dim is None]
        return MappingProxyType(
            {
                axis.axis: axis.values.item()
                if axis.dim is None
                else (axis.dim, axis.values.view())
                for axis in ordered
            }
        )

    @property
    def intervals(self) -> Mapping[str, npt.NDArray[np.float64]]:
        """Read-only declared interval views, keyed by source axis name."""
        return MappingProxyType({name: rows.view() for name, rows in self._intervals.items()})

    def _sampling(self) -> Sampling:
        return Sampling(
            self._transform,
            self._dims,
            self.sizes,
            tuple(
                AxisSampling(
                    axis.axis,
                    axis.dim,
                    np.asarray(axis.values, dtype=np.float64),
                    intervals=self._intervals.get(axis.axis),
                )
                for axis in self._axes
            ),
        )

    def point_at(self, /, **positions: int | np.integer[Any]) -> npt.NDArray[np.float64]:
        """Locate one sample using a nonnegative integer position per dimension.

        Raises:
            TypeError: If a position is not an integer, is a boolean, or the transform
                returns non-real points.
            ValueError: If positions do not name exactly the varying dimensions, or the
                transform returns points of the wrong shape or non-finite values.
            IndexError: If a position is outside its dimension.
        """
        if set(positions) != set(self._dims):
            raise ValueError(f"positions must name each geometry dimension {self._dims} once")
        indices = [
            check_position(positions[dim], dim=dim, size=self.sizes[dim]) for dim in self._dims
        ]
        coordinates = np.array(
            [
                float(axis.values)
                if axis.dim is None
                else axis.values[indices[self._dims.index(axis.dim)]]
                for axis in self._axes
            ],
            dtype=np.float64,
        )
        return transform_points(self._transform, coordinates)

    def points(self) -> npt.NDArray[np.float64]:
        """Return all sample points shaped (*sizes, number of frame axes).

        Raises:
            TypeError: If the transform returns non-real points.
            ValueError: If the transform returns points of the wrong shape or non-finite values.
        """
        shape = tuple(self.sizes.values())
        columns = []
        for axis in self._axes:
            if axis.dim is None:
                columns.append(np.broadcast_to(axis.values, shape))
            else:
                axis_shape = [1] * len(shape)
                axis_shape[self._dims.index(axis.dim)] = axis.values.size
                columns.append(np.broadcast_to(axis.values.reshape(axis_shape), shape))
        return transform_points(self._transform, np.stack(columns, axis=-1).astype(np.float64))

    def points_at(
        self,
        positions: npt.ArrayLike,
        *,
        domain: Domain = "samples",
        outside: Outside = "raise",
    ) -> npt.NDArray[np.float64]:
        """Map fractional positions (..., D), in dims order, to frame points (..., M).

        Coordinates interpolate piecewise linearly. Retained scalars need no position.
        The samples/cells domain bounds accepted positions; outside selects refusal,
        all-NaN rows, or outer-step extrapolation ("raise", "nan", "extrapolate").

        Raises:
            TypeError: If positions or transformed points have a non-real dtype.
            ValueError: If positions have the wrong shape, domain or outside is unknown,
                an empty or single-sample axis has no step, positions are outside the domain,
                or transformed points have the wrong shape or non-finite values.
        """
        return points_at(self._sampling(), positions, domain=domain, outside=outside)

    def positions_at(
        self,
        points: npt.ArrayLike,
        *,
        domain: Domain = "samples",
        outside: Outside = "raise",
    ) -> npt.NDArray[np.float64]:
        """Locate frame points (..., M) as fractional positions (..., D) in dims order.

        The exact transform inverse and piecewise-linear coordinate inverse are used.
        The cells domain extends the outer sample bounds to declared interval edges, or to
        sample-offset edges when no interval is declared. Interior gaps between declared
        cells remain interpolated in both domains; outer-cell positions are returned unclipped.
        Outside points raise, become all-NaN rows, or extrapolate by the outer steps,
        according to outside ("raise", "nan", "extrapolate").

        Raises:
            TypeError: If the transform has no inverse or points have a non-real dtype.
            ValueError: If the inverse is undetermined, an axis is retained or empty,
                coordinates are not strictly monotonic, domain or outside is unknown,
                a cells-domain axis has no width, or points are outside the domain.
        """
        return positions_at(self._sampling(), points, domain=domain, outside=outside)

    def lattice(
        self, dims: Sequence[str] | None = None, *, tolerance: float = LATTICE_TOLERANCE
    ) -> Lattice:
        """Return the regular lattice, with columns in dims order (default: grid order).

        Tolerance bounds deviations from uniform coordinates in fractions of a step.

        Raises:
            TypeError: If the transform is not affine or dims is not a sequence of strings.
            ValueError: If dims does not name every varying dimension exactly once,
                coordinates are nonuniform, an axis has fewer than two samples, or a
                dimension maps to no displacement in the frame.
        """
        return lattice(self._sampling(), dims, tolerance=tolerance)

    def is_coincident(self, other: Grid, *, tolerance: float = LATTICE_TOLERANCE) -> bool:
        """Compare sample locations within a fraction of a step; see Geometry.is_coincident.

        Raises:
            TypeError: If other is not a Grid or tolerance is not a real number.
            ValueError: If tolerance is outside [0, 0.5) or sampling cannot be inverted.
        """
        if not isinstance(other, Grid):
            raise TypeError(f"other must be a Grid, got {type(other).__name__}")
        return is_coincident(self._sampling(), other._sampling(), check_tolerance(tolerance))

    def isel(self, **indexers: Any) -> Grid:
        """Select samples by dimension using xarray's positional indexing semantics.

        Integers retain scalar axes; lists, integer arrays, boolean masks and slices
        retain varying axes. Negative integer indices count from the end.

        Raises:
            ImportError: If xarray is unavailable.
            ValueError: If dimensions are unknown or indexing changes geometry dimensions.
            IndexError: If an indexer is invalid or outside its dimension.
        """
        try:
            import xarray as xr
        except ImportError as error:
            raise ImportError(
                "Grid.isel requires xarray; install xarray to select samples"
            ) from error
        from ._binding import grid_from_binding
        from .native import grid_coordinates

        selected = xr.Dataset(coords=grid_coordinates(self)).isel(indexers)
        return grid_from_binding(selected.coords)

    def sel(self, **indexers: Any) -> Grid:
        """Select samples by source-coordinate label using xarray's selection semantics.

        Scalar labels retain scalar axes; lists and label slices retain varying axes.

        Raises:
            ImportError: If xarray is unavailable.
            KeyError: If a coordinate or label is absent, or the source axis is fixed.
            ValueError: If indexing changes geometry dimensions.
        """
        try:
            import xarray as xr
        except ImportError as error:
            raise ImportError(
                "Grid.sel requires xarray; install xarray to select samples"
            ) from error
        from ._binding import grid_from_binding
        from .native import grid_coordinates

        selected = xr.Dataset(coords=grid_coordinates(self)).sel(indexers)
        return grid_from_binding(selected.coords)

    def transpose(self, *dims: str) -> Grid:
        """Reorder every varying dimension; with no arguments, reverse dimension order.

        Raises:
            TypeError: If a dimension is not a string.
            ValueError: If dims does not name every varying dimension exactly once.
        """
        order = check_names(dims, field="dims", allow_empty=True) if dims else self._dims[::-1]
        if set(order) != set(self._dims):
            raise ValueError(
                f"dims must name each geometry dimension {self._dims} once, got {order}"
            )
        by_dim = {axis.dim: axis.axis for axis in self._axes if axis.dim is not None}
        names = [by_dim[dim] for dim in order]
        names += [axis.axis for axis in self._axes if axis.dim is None]
        return Grid(
            self._transform,
            {name: self.coordinates[name] for name in names},
            intervals=self._intervals,
        )

    def __eq__(self, other: object) -> bool:
        """Compare transforms, dimension order, coordinates and declared support exactly."""
        if not isinstance(other, Grid):
            return NotImplemented
        return (
            self._transform == other._transform
            and self._dims == other._dims
            and self._intervals.keys() == other._intervals.keys()
            and all(
                np.array_equal(rows, other._intervals[name])
                for name, rows in self._intervals.items()
            )
            and all(
                a.dim == b.dim
                and a.values.dtype.kind == b.values.dtype.kind
                and np.array_equal(a.values, b.values)
                for a, b in zip(self._axes, other._axes, strict=True)
            )
        )

    def __hash__(self) -> int:
        """Hash the same declaration exact equality compares."""
        return hash(
            (
                self._transform,
                self._dims,
                tuple(
                    (name, self._intervals[name].tobytes())
                    for name in self._transform.source.axes
                    if name in self._intervals
                ),
                tuple(
                    (axis.dim, axis.values.dtype.kind, axis.values.tobytes()) for axis in self._axes
                ),
            )
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Reconstruct copies and pickles through the validated immutable constructor."""
        return _restore_grid, (self._transform, dict(self.coordinates), dict(self.intervals))

    def __repr__(self) -> str:
        """Name the source axes, target frame, dimensions and sizes."""
        state = ", anonymous=True" if self.frame.is_anonymous else ""
        return (
            f"Grid(source={self._transform.source.axes!r}, target={self.frame.identifier!r}, "
            f"dims={self._dims!r}, sizes={dict(self.sizes)!r}, intervals={tuple(self._intervals)!r}{state})"
        )


def _restore_grid(
    transform: SupportsPoints,
    coordinates: Mapping[str, Coordinate],
    intervals: Mapping[str, npt.ArrayLike],
) -> Grid:
    return Grid(transform, coordinates, intervals=intervals)
