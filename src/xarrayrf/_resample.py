"""xarray I/O for resampling onto another array's samples."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import partial

import numpy as np
import numpy.typing as npt
import xarray as xr

from ._affine import AffineTransform
from ._binding import grid_variables
from ._geometry import Geometry
from ._grid import Grid
from ._resampling import BLOCK_POINTS, Support, check_options, execute
from ._resampling import Method as Method
from ._resampling import plan as _plan
from ._sampling import Domain, Sampling
from ._transform import SupportsPoints


class _GeometrySizes(Mapping[str, int]):
    """Revalidate geometry when the planner first queries a dimension's size.

    The array cannot change during planning, so later reads reuse the first validated sizes.
    """

    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry
        self._sizes: Mapping[str, int] | None = None

    def __getitem__(self, dim: str) -> int:
        if self._sizes is None:
            self._sizes = self.geometry.sizes
        return self._sizes[dim]

    def __iter__(self) -> Iterator[str]:
        return iter(self.geometry.dims)

    def __len__(self) -> int:
        return len(self.geometry.dims)


def _sampling_for_plan(geometry: Geometry | Grid) -> Sampling:
    """Defer revalidation to the planner's size and coordinate reads."""
    if isinstance(geometry, Grid):
        return geometry._sampling()
    return Sampling(
        geometry.transform,
        geometry.dims,
        _GeometrySizes(geometry),
        lambda: geometry._sampling().axes,
    )


def resample(
    source: Geometry,
    target: Geometry | Grid,
    *,
    transform: SupportsPoints | None = None,
    method: Method = "linear",
    fill_value: float = np.nan,
    domain: Domain = "samples",
    support: Support = "point",
    min_coverage: float = 0.5,
    block_points: int = BLOCK_POINTS,
) -> xr.DataArray:
    """Resample the source array's values onto a target Geometry or Grid.

    For each target sample, its point in the target frame is mapped into the source frame and
    located among the source samples, and the source values are interpolated there.
    Non-geometry dimensions of the source (time, echo, channel) are carried through unchanged.
    A fully scalar target represents one point and adds no dimensions to the result; its
    scalar geometry coordinates are retained. Retained scalar source axes remain unsupported.

    Frames: equal frames need nothing; equivalent frames in different coordinate systems (LPS
    and RAS) use the derived coordinate-system change; different frames need ``transform``, from
    the target's frame to the source's, such as a registration result. Numerically identical
    declarations never stand in for identity.

    Box methods (``step`` and ``overlap_mean``) treat each slab value as a mean over its
    declared interval and reconstruct the mean of the slabs covering each point. ``step``
    refuses overlapping source slabs. ``support="average"`` averages this reconstruction over
    covered target support; ``support="point"`` evaluates it at target coordinates. Uncovered
    targets and coverage below ``min_coverage`` receive ``fill_value``. Box methods require
    floating or complex source values, a separable affine map, source intervals on slab axes,
    and target intervals when averaging those axes. They refuse non-default ``domain``.
    Average support for nearest, linear and cubic is not yet supported. Defaults are unchanged.
    Unlike point interpolation, box methods assume nothing between slabs: targets in gaps
    are fill rather than bridged, and values reach the outer slab edges rather than
    stopping at the outer slab centres.
    Box methods apply sparse per-axis weights without cropping; Dask geometry dimensions are
    core dimensions and may be rechunked into memory together.

    Domain: by default values are defined between the outer source samples, as for xarray's
    ``interp``, and ``fill_value`` marks everything else. ``domain="cells"`` also answers in the
    outer samples' cells, holding the edge sample's value there for point interpolation methods
    (ITK's domain is the same; its nearest and linear interpolators also hold the edge).
    Each source axis's :attr:`~xarrayrf.ArrayCoordinates.sample_offset` says where its cells
    lie; a point-sampled axis has none and reaches no further than its samples. Interpolation
    between samples is the same in both domains.

    Linear interpolation ignores NaN components whose interpolation weight is at most
    ``1e-9`` (measured position roundoff), and propagates larger missing weights. Real and
    imaginary components are handled independently, without renormalizing the finite
    weights. The same allowance classifies integral signed-permutation maps for exact
    gathering. It covers measured large-origin oblique grids, not arbitrary ill-scaled maps.
    Cubic prefiltering can spread NaNs through spline coefficients; only same-grid gathering
    retains original samples. Infinities retain SciPy's existing interpolation behavior.

    Empty targets return empty values without locating or interpolating samples. Empty sources
    with non-empty targets raise ValueError, regardless of the domain or ``fill_value``.

    Performance: when both arrays form regular lattices and every transform is affine, target
    positions map to source positions through one composed affine, and each non-geometry slice
    is interpolated by ``scipy.ndimage.affine_transform`` with a
    separable outside mask; ``block_points`` does not apply. Otherwise target coordinates are
    mapped through the transforms and the source's exact inverse in blocks of ``block_points``
    samples, and ``scipy.ndimage.map_coordinates`` interpolates; for nearest and linear each
    block's positions are reused for every slice when no linear NaN handling is needed.
    Linear NaN buffers and cubic spline coefficients are prepared for one source slice at a
    time; positions are cached within a bounded budget or recomputed per slice. A Dask-backed
    source is processed lazily, one task per chunk of its non-geometry dimensions; positions
    and masks are shared within a task, not across tasks.
    Install the ``resample`` extra for scipy.

    Args:
        source: The array whose values are resampled, with its geometry.
        target: Geometry or Grid whose samples receive values; target pixels are ignored.
        transform: The transform from the target's frame to the source's frame, required when
            they are different frames.
        method: ``"nearest"``, ``"linear"``, ``"cubic"``, ``"step"`` or ``"overlap_mean"``.
        fill_value: Value for target samples outside the domain.
        domain: ``"samples"`` or ``"cells"``, as above.
        support: ``"point"`` (default) or ``"average"`` over declared target intervals.
        min_coverage: Minimum covered fraction in ``(0, 1]``; default 0.5. Validated for
            every method, but used only by box methods; coverage multiplies across axes.
        block_points: Target samples located per block.

    Returns:
        The resampled values over the source's non-geometry dimensions and the target's geometry
        dimensions, with the target's geometry coordinates and the source's non-geometry ones.
        Complex128 for complex sources and float64 otherwise, except that nearest-neighbour
        resampling of an integer or boolean array keeps its dtype when ``fill_value`` is
        representable in it.

    Raises:
        TypeError: If an argument has the wrong type, ``transform`` is not a point transform,
            or a box method receives non-floating source values.
        ValueError: If the frames are different and no transform is given, ``transform`` has the
            wrong endpoints, the source is empty and the target is not, ``method`` or ``domain``
            is unknown, a source axis declaring cells
            has a single sample in the cells domain, the
            source's samples cannot be located (no
            exact inverse, a retained scalar or multidimensional source coordinate, or
            non-monotonic coordinates), or a target geometry dimension name is used by a
            non-geometry dimension of the source, or a target geometry coordinate name
            collides with a source non-geometry coordinate.
        ImportError: If scipy is not installed, after input and frame validation.
    """
    result, _, _ = _resample_with_intervals(
        source,
        target,
        transform=transform,
        method=method,
        fill_value=fill_value,
        domain=domain,
        support=support,
        min_coverage=min_coverage,
        block_points=block_points,
    )
    return result


def _resample_with_intervals(
    source: Geometry,
    target: Geometry | Grid,
    *,
    transform: SupportsPoints | None = None,
    method: Method = "linear",
    fill_value: float = np.nan,
    domain: Domain = "samples",
    support: Support = "point",
    min_coverage: float = 0.5,
    block_points: int = BLOCK_POINTS,
) -> tuple[xr.DataArray, dict[str, npt.NDArray[np.float64]], xr.DataArray | None]:
    """Return unframed values, interval claims and optional box coverage from one plan."""
    if not isinstance(source, Geometry) or not isinstance(target, Geometry | Grid):
        raise TypeError("source must be Geometry and target must be Geometry or Grid")
    check_options(method, block_points, support, min_coverage)
    target_order = (
        target.dims
        if isinstance(target, Grid)
        else tuple(str(dim) for dim in target.array.dims if dim in target.dims)
    )
    other_dims = [str(dim) for dim in source.array.dims if dim not in source.dims]

    def adoption_suffices(adopted: AffineTransform) -> bool:
        source_grid = source.grid()
        target_grid = target if isinstance(target, Grid) else target.grid()
        return (
            Grid(adopted, source_grid.coordinates, intervals=source_grid.intervals) == target_grid
        )

    plan = _plan(
        _sampling_for_plan(source),
        _sampling_for_plan(target),
        source_order=source.dims,
        target_order=target_order,
        dtype=source.array.dtype,
        other_dims=tuple(other_dims),
        transform=transform,
        method=method,
        fill_value=fill_value,
        domain=domain,
        support=support,
        min_coverage=min_coverage,
        block_points=block_points,
        adoption_suffices=adoption_suffices,
    )
    source_values = source.array
    if plan.window is not None:
        source_values = source_values.isel(dict(zip(plan.source_dims, plan.window, strict=True)))
    source_context = {
        name
        for name, coordinate in source.array.coords.items()
        if name not in source.transform.source.axes and set(coordinate.dims) <= set(other_dims)
    }
    collisions = set(target.transform.source.axes) & source_context
    if collisions:
        raise ValueError(
            f"target geometry coordinate {sorted(collisions)[0]!r} collides with a "
            "source non-geometry coordinate"
        )
    renamed = {dim: f"__xarrayrf_target_{dim}" for dim in plan.target_dims}
    ordered = source_values.transpose(*other_dims, *plan.source_dims)
    if plan.total == 0 and ordered.chunks is not None:
        # Dask's automatic gufunc rechunking divides by zero on empty core dimensions.
        ordered = ordered.chunk({dim: -1 for dim in plan.source_dims})
    result: xr.DataArray = xr.apply_ufunc(
        partial(execute, plan),
        ordered,
        input_core_dims=[list(plan.source_dims)],
        output_core_dims=[[renamed[dim] for dim in plan.target_dims]],
        exclude_dims=set(plan.source_dims),
        dask="parallelized",
        output_dtypes=[plan.output_dtype],
        dask_gufunc_kwargs={
            "output_sizes": {
                renamed[dim]: size
                for dim, size in zip(plan.target_dims, plan.target_shape, strict=True)
            },
            "allow_rechunk": plan.total != 0,
        },
        keep_attrs=False,
    )
    result = result.rename({value: key for key, value in renamed.items()})
    coordinates = grid_variables(target) if isinstance(target, Grid) else target.array.coords
    target_coordinates = {
        name: coordinate
        for name, coordinate in coordinates.items()
        if name in target.transform.source.axes
    }
    source_coordinates = {
        name: coordinate.variable
        for name, coordinate in source.array.coords.items()
        if name in source_context and name not in result.xindexes
    }
    for name in result.xindexes:
        if name in source.array.coords and set(source.array.coords[name].dims) <= set(other_dims):
            result.coords[name].attrs = dict(source.array.coords[name].attrs)
    result = result.drop_vars(
        [name for name in result.coords if set(result.coords[name].dims) & set(plan.source_dims)],
        errors="ignore",
    ).assign_coords({**source_coordinates, **target_coordinates})
    coverage = (
        None
        if plan.box is None
        else xr.DataArray(
            plan.box.coverage,
            dims=plan.target_dims,
            coords=target_coordinates,
        )
    )
    return result, plan.intervals, coverage
