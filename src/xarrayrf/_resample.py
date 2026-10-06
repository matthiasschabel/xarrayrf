"""Resampling an array's values onto another array's samples."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from itertools import product
from types import ModuleType
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from ._affine import equilibrated_inverse
from ._binding import grid_variables
from ._frame import ReferenceFrame
from ._frame_compatibility import anonymous_frame_difference
from ._geometry import Geometry, adopt_frame
from ._grid import Grid
from ._orientation import coordinate_system_change
from ._positions import check_domain, extents, lattice_parts, locator, sample_columns
from ._sampling import LATTICE_TOLERANCE, POSITION_SLACK, Domain
from ._transform import SupportsAffine, SupportsPoints, check_transform

type Method = Literal["nearest", "linear", "cubic"]

_SPLINE_ORDER: Final = {"nearest": 0, "linear": 1, "cubic": 3}

_SLAB_SAMPLES: Final = 1 << 22
"""Target samples per slab when computing the outside mask of the lattice path."""

POSITION_CACHE_BYTES: Final = 1 << 28
"""Largest size of general-path positions kept across slices for cubic resampling (256 MiB)."""

BLOCK_POINTS: Final = 1 << 20
"""Target samples located per block: bounds the position arrays to a few tens of megabytes."""

CROP_MARGIN: Final = {0: 1, 1: 1, 3: 24}
"""Source samples kept beyond the target's footprint when cropping, per spline order.

Nearest and linear read at most one neighbour past a position. The cubic prefilter is not
local: its influence decays by the spline pole |z| = 0.268 per sample, so 24 samples leave
0.268**24 ~ 2e-14 of a boundary change, below float64 resolution of the result.
"""

type PositionMap = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
"""Target positions ``(Dt, n)`` to source positions ``(Ds, n)``, NaN where outside."""


class _LatticeMap:
    """Target positions to source positions as one affine: ``source = linear @ target + offset``."""

    __slots__ = ("highest", "linear", "lowest", "offset")

    def __init__(
        self,
        linear: npt.NDArray[np.float64],
        offset: npt.NDArray[np.float64],
        lowest: npt.NDArray[np.float64],
        highest: npt.NDArray[np.float64],
    ) -> None:
        self.linear = linear
        self.offset = offset
        self.lowest = lowest
        self.highest = highest

    def outside(
        self,
        target_shape: tuple[int, ...],
        lowest: npt.NDArray[np.float64] | None = None,
        highest: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Mark target samples whose source position lies outside, one slab at a time.

        The bounds default to the domain's. Each source position is separable in the target
        positions, so it is accumulated by broadcasting one-dimensional terms; only one slab of
        the first target axis is held.
        """
        mask = np.zeros(target_shape, dtype=bool)
        axes = [np.arange(size, dtype=np.float64) for size in target_shape]
        lower = (self.lowest if lowest is None else lowest) - POSITION_SLACK
        upper = (self.highest if highest is None else highest) + POSITION_SLACK
        dims = len(target_shape)
        rest = int(np.prod(target_shape[1:], dtype=np.int64))
        slab_rows = max(1, _SLAB_SAMPLES // max(1, rest))
        for start in range(0, target_shape[0], slab_rows):
            stop = min(target_shape[0], start + slab_rows)
            terms = [axes[0][start:stop], *axes[1:]]
            for axis in range(self.linear.shape[0]):
                position = np.full((stop - start, *target_shape[1:]), self.offset[axis])
                for column, term in enumerate(terms):
                    shape = [1] * dims
                    shape[column] = -1
                    position += self.linear[axis, column] * term.reshape(shape)
                mask[start:stop] |= (position < lower[axis]) | (position > upper[axis])
        return mask


def _endpoint_name(endpoint: object) -> str:
    identifier = getattr(endpoint, "identifier", None)
    return ":".join(identifier) if identifier else repr(endpoint)


def _frame_map(
    target: ReferenceFrame, source: ReferenceFrame, transform: SupportsPoints | None
) -> SupportsPoints | None:
    """Return the transform from the target frame to the source frame; ``None`` is identity."""
    if transform is not None:
        check_transform(transform)
        if transform.source != target or transform.target != source:
            given = (_endpoint_name(transform.source), _endpoint_name(transform.target))
            needed = (_endpoint_name(target), _endpoint_name(source))
            detail = " (same frames, different coordinate systems)" if given == needed else ""
            raise ValueError(
                f"the transform maps {given[0]} to {given[1]}, but this resampling needs "
                f"{needed[0]} (target) to {needed[1]} (source){detail}; a transform applies "
                "only to the frames it was computed between",
            )
        return transform
    if target == source:
        return None
    if target.is_equivalent_frame(source):
        return coordinate_system_change(target, source)
    raise ValueError(
        f"the target frame {target.identifier} and the source frame {source.identifier} are "
        "different frames; supply the transform between them, such as a registration result",
    )


def _homogeneous(
    matrix: npt.NDArray[np.float64], offset: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    rows, columns = matrix.shape
    result = np.zeros((rows + 1, columns + 1))
    result[:rows, :columns] = matrix
    result[:rows, columns] = offset
    result[rows, columns] = 1.0
    return result


def _crop_window(
    lattice_map: _LatticeMap, target_shape: tuple[int, ...], sizes: tuple[int, ...], order: int
) -> tuple[slice, ...] | None:
    """Source index ranges the target can read, or ``None`` when that is the whole source.

    The map is affine, so the extreme source positions over the target index box occur at its
    corners; adding the interpolation margin bounds every sample the target touches.
    """
    corners = np.array(list(product(*((0, n - 1) for n in target_shape))), dtype=np.float64).T
    positions = lattice_map.linear @ corners + lattice_map.offset[:, None]
    margin = CROP_MARGIN[order]
    start = np.clip(np.floor(positions.min(axis=1)) - margin, 0, np.array(sizes) - 1)
    stop = np.clip(np.ceil(positions.max(axis=1)) + margin + 1, 1, np.array(sizes))
    stop = np.maximum(stop, start + 1)  # a target entirely outside keeps one (masked) sample
    window = tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))
    if all(w.start == 0 and w.stop == n for w, n in zip(window, sizes, strict=True)):
        return None
    return window


def _lattice_map(
    source: Geometry,
    target: Geometry | Grid,
    target_dims: tuple[str, ...],
    frame_map: SupportsPoints | None,
    extents: list[tuple[float, float]],
) -> _LatticeMap | None:
    """Collapse target positions -> frame -> source positions into one affine, if possible."""
    if frame_map is not None and not isinstance(frame_map, SupportsAffine):
        return None
    if not isinstance(target.transform, SupportsAffine):
        return None
    try:
        # A single-sample target dimension is never stepped along, so it needs no step.
        target_origin, target_columns = lattice_parts(
            target._sampling(), target_dims, LATTICE_TOLERANCE, single_samples=True
        )
        source_lattice = source.lattice(source.dims)
    except (TypeError, ValueError):
        return None
    matrix = source_lattice.matrix
    if matrix.shape[0] != matrix.shape[1]:
        return None
    try:
        source_inverse = equilibrated_inverse(matrix)
    except ValueError:
        return None
    frame = np.eye(matrix.shape[0] + 1)
    if frame_map is not None:
        assert isinstance(frame_map, SupportsAffine)
        frame = _homogeneous(frame_map.matrix, frame_map.translation)
    combined = (
        _homogeneous(source_inverse, -(source_inverse @ source_lattice.origin))
        @ frame
        @ _homogeneous(target_columns, target_origin)
    )
    sizes = np.array([source.array.sizes[dim] for dim in source.dims], dtype=np.float64)
    reach = np.array(extents, dtype=np.float64).reshape(-1, 2)
    return _LatticeMap(
        combined[: matrix.shape[0], : len(target_dims)],
        combined[: matrix.shape[0], len(target_dims)],
        -reach[:, 0],
        sizes - 1.0 + reach[:, 1],
    )


def _general_map(
    source: Geometry,
    target: Geometry | Grid,
    target_dims: tuple[str, ...],
    frame_map: SupportsPoints | None,
    domain: Domain,
) -> PositionMap:
    """Map through target coordinates, the frames and the source's exact inverse."""
    samplings = target._sampling().axes
    transform = target.transform
    locate = locator(source._sampling(), domain)

    def positions(target_positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        indices = target_positions.astype(np.intp)
        points = transform.transform_point(sample_columns(samplings, indices, target_dims))
        if frame_map is not None:
            points = frame_map.transform_point(points)
        located: npt.NDArray[np.float64] = locate(points).T
        return located

    return positions


def _output_dtype(
    dtype: np.dtype[np.generic], method: Method, fill_value: float
) -> np.dtype[np.generic]:
    if dtype.kind == "c":
        return np.dtype(np.complex128)
    representable = (
        method == "nearest"
        and dtype.kind in "biu"
        and not np.isnan(fill_value)
        and float(np.asarray(fill_value).astype(dtype)) == fill_value
    )
    return dtype if representable else np.dtype(np.float64)


def _same_grid_indices(
    lattice_map: _LatticeMap, target_shape: tuple[int, ...], source_shape: tuple[int, ...]
) -> tuple[tuple[npt.NDArray[np.intp], ...], tuple[int, ...]] | None:
    """Return clipped source indices and target axis order for an integral grid map."""
    linear = lattice_map.linear
    if linear.shape != (len(source_shape), len(target_shape)) or len(source_shape) != len(
        target_shape
    ):
        return None
    rounded = np.rint(linear).astype(np.intp)
    if not (
        np.all(np.isin(rounded, (-1, 0, 1)))
        and np.all(np.count_nonzero(rounded, axis=0) == 1)
        and np.all(np.count_nonzero(rounded, axis=1) == 1)
    ):
        return None
    offset = np.rint(lattice_map.offset).astype(np.intp)
    for corner in product(*((0, size - 1) for size in target_shape)):
        position = np.asarray(corner, dtype=np.float64)
        error = (linear - rounded) @ position + lattice_map.offset - offset
        if np.max(np.abs(error)) > 1e-6:
            return None
    source_to_target = tuple(int(np.flatnonzero(row)[0]) for row in rounded)
    indices = tuple(
        np.clip(
            offset[axis] + rounded[axis, target_axis] * np.arange(target_shape[target_axis]),
            0,
            source_shape[axis] - 1,
        ).astype(np.intp)
        for axis, target_axis in enumerate(source_to_target)
    )
    return indices, tuple(
        int(np.argsort(source_to_target)[axis]) for axis in range(len(target_shape))
    )


@dataclass(frozen=True)
class _Plan:
    """Validated sampling choices and geometry shared by every block."""

    order: int
    output_dtype: np.dtype[np.generic]
    fill_value: float
    source_dims: tuple[str, ...]
    target_dims: tuple[str, ...]
    target_shape: tuple[int, ...]
    total: int
    block_points: int
    lattice_map: _LatticeMap | None
    position_map: PositionMap | None
    same_grid: tuple[tuple[npt.NDArray[np.intp], ...], tuple[int, ...]] | None
    source_last: npt.NDArray[np.float64]
    ndimage: ModuleType


def _plan(
    source: Geometry,
    target: Geometry | Grid,
    *,
    transform: SupportsPoints | None,
    method: Method,
    fill_value: float,
    domain: Domain,
    block_points: int,
) -> tuple[_Plan, xr.DataArray]:
    """Validate resampling and prepare the source region and block inputs."""
    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError("resample needs scipy; install xarrayrf[resample]") from error

    if not isinstance(source, Geometry) or not isinstance(target, Geometry | Grid):
        raise TypeError("source must be Geometry and target must be Geometry or Grid")
    if method not in _SPLINE_ORDER:
        raise ValueError(f"method must be one of {tuple(_SPLINE_ORDER)}, got {method!r}")
    if isinstance(block_points, bool) or not isinstance(block_points, int) or block_points < 1:
        raise ValueError(f"block_points must be a positive integer, got {block_points!r}")
    order = _SPLINE_ORDER[method]
    if (
        transform is None
        and not target.frame.is_equivalent_frame(source.frame)
        and (source.frame.is_anonymous or target.frame.is_anonymous)
    ):
        try:
            adopted = adopt_frame(source.transform, target.frame)
        except ValueError:
            suffices = False
        else:
            source_grid = source.grid()
            target_grid = target if isinstance(target, Grid) else target.grid()
            suffices = (
                Grid(adopted, source_grid.coordinates, intervals=source_grid.intervals)
                == target_grid
            )
        raise ValueError(
            anonymous_frame_difference(
                source._sampling(),
                target._sampling(),
                labels=("source operand", "target operand"),
                adoption_suffices=suffices,
            )
        )
    frame_map = _frame_map(target.frame, source.frame, transform)
    target_dims = (
        target.dims
        if isinstance(target, Grid)
        else tuple(str(dim) for dim in target.array.dims if dim in target.dims)
    )
    source_dims = source.dims
    other_dims = [str(dim) for dim in source.array.dims if dim not in source_dims]
    clashes = sorted(set(target_dims) & set(other_dims))
    if clashes:
        raise ValueError(
            f"target geometry dimensions {clashes} are non-geometry dimensions of the source",
        )
    target_shape = tuple(target.sizes[dim] for dim in target_dims)
    total = int(np.prod(target_shape, dtype=np.int64))
    check_domain(domain)
    lattice_map = None
    position_map = None
    if total:
        if any(source.sizes[dim] == 0 for dim in source_dims):
            raise ValueError("cannot resample an empty source: nothing to sample from")
        reach = extents(source._sampling(), domain)
        lattice_map = _lattice_map(source, target, target_dims, frame_map, reach)
        if lattice_map is None:
            position_map = _general_map(source, target, target_dims, frame_map, domain)
    output_dtype = _output_dtype(source.array.dtype, method, fill_value)
    source_values = source.array
    if lattice_map is not None:
        # Read only the source region the target can touch: for a lazily loaded source (a
        # remote COG or OME-Zarr) this is the difference between a crop and the whole file.
        window = _crop_window(
            lattice_map,
            target_shape,
            tuple(source.array.sizes[dim] for dim in source_dims),
            order,
        )
        if window is not None:
            source_values = source.array.isel(dict(zip(source_dims, window, strict=True)))
            shift = np.array([w.start for w in window], dtype=np.float64)
            lattice_map = _LatticeMap(
                lattice_map.linear,
                lattice_map.offset - shift,
                lattice_map.lowest - shift,
                lattice_map.highest - shift,
            )
    same_grid = (
        _same_grid_indices(
            lattice_map, target_shape, tuple(source_values.sizes[dim] for dim in source_dims)
        )
        if lattice_map is not None
        else None
    )

    source_last = np.array(
        [source_values.sizes[dim] - 1 for dim in source_dims], dtype=np.float64
    ).reshape(-1, 1)

    return (
        _Plan(
            order=order,
            output_dtype=output_dtype,
            fill_value=fill_value,
            source_dims=source_dims,
            target_dims=target_dims,
            target_shape=target_shape,
            total=total,
            block_points=block_points,
            lattice_map=lattice_map,
            position_map=position_map,
            same_grid=same_grid,
            source_last=source_last,
            ndimage=ndimage,
        ),
        source_values,
    )


def _coefficients(plan: _Plan, volume: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Cubic spline coefficients for one slice; nearest and linear use values directly."""
    if plan.order <= 1:
        return volume
    coefficient_dtype = np.complex128 if plan.output_dtype.kind == "c" else np.float64
    filtered: npt.NDArray[np.generic] = plan.ndimage.spline_filter(
        volume, order=plan.order, mode="nearest", output=coefficient_dtype
    )
    return filtered


def _interpolate(
    plan: _Plan, volume: npt.NDArray[np.generic], positions: npt.NDArray[np.float64], size: int
) -> npt.NDArray[np.generic]:
    """Interpolate one position block, holding edge values within outer cells."""
    inside = ~np.isnan(positions[0])
    block = np.full(size, plan.fill_value, dtype=plan.output_dtype)
    # Beyond the outer samples, in their cells, the edge value holds; a cubic spline would
    # otherwise extrapolate there.
    block[inside] = plan.ndimage.map_coordinates(
        volume,
        np.clip(positions[:, inside], 0.0, plan.source_last),
        order=plan.order,
        mode="nearest",
        prefilter=False,
        output=plan.output_dtype,
    )
    return block


def _block_positions(plan: _Plan, start: int, stop: int) -> npt.NDArray[np.float64]:
    """Locate a flattened block of target samples in the source."""
    assert plan.position_map is not None
    target_positions = (
        np.array(np.unravel_index(np.arange(start, stop), plan.target_shape), dtype=np.float64)
        if plan.target_shape
        else np.empty((0, stop - start), dtype=np.float64)
    )
    return plan.position_map(target_positions)


def _lattice_block(
    plan: _Plan, slices: list[npt.NDArray[np.generic]], leading: tuple[int, ...]
) -> npt.NDArray[np.generic]:
    """Resample slices through the composed affine or integral-grid gather."""
    lattice_map = plan.lattice_map
    assert lattice_map is not None
    target_shape = plan.target_shape
    if not target_shape:
        # SciPy requires positive output rank; the dummy column never varies.
        target_shape = (1,)
        lattice_map = _LatticeMap(
            np.zeros((len(plan.source_dims), 1)),
            lattice_map.offset,
            lattice_map.lowest,
            lattice_map.highest,
        )
    # One compiled pass per slice, prefiltering only that slice; edges extend so
    # rounding at an exact edge sample cannot turn it into fill, and the separable
    # mask applies the true boundary. Nearest and linear already hold the edge value
    # beyond the outer samples; a cubic spline would extrapolate, so target samples in
    # the outer cells are evaluated again at clamped positions.
    outside = lattice_map.outside(target_shape)
    if plan.same_grid is not None:
        indices, axis_order = plan.same_grid
        output = np.empty((len(slices), *target_shape), dtype=plan.output_dtype)
        for index, volume in enumerate(slices):
            output[index] = volume[np.ix_(*indices)].transpose(axis_order)
            output[index][outside] = plan.fill_value
        return output.reshape((*leading, *plan.target_shape))
    last = plan.source_last[:, 0]
    shell = None
    if plan.order > 1 and ((lattice_map.lowest < 0.0).any() or (lattice_map.highest > last).any()):
        beyond = lattice_map.outside(target_shape, np.zeros_like(last), last)
        shell = np.nonzero(beyond & ~outside)
        shell_positions = np.clip(
            lattice_map.linear @ np.array(shell, dtype=np.float64) + lattice_map.offset[:, None],
            0.0,
            plan.source_last,
        )
    output = np.empty((len(slices), *target_shape), dtype=plan.output_dtype)
    for index, volume in enumerate(slices):
        filtered = _coefficients(plan, volume)
        plan.ndimage.affine_transform(
            filtered,
            lattice_map.linear,
            offset=lattice_map.offset,
            output_shape=target_shape,
            output=output[index],
            order=plan.order,
            mode="nearest",
            prefilter=False,
        )
        output[index][outside] = plan.fill_value
        if shell is not None:
            output[index][shell] = plan.ndimage.map_coordinates(
                filtered,
                shell_positions,
                order=plan.order,
                mode="nearest",
                prefilter=False,
                output=plan.output_dtype,
            )
    return output.reshape((*leading, *plan.target_shape))


def _general_block(
    plan: _Plan, slices: list[npt.NDArray[np.generic]], leading: tuple[int, ...]
) -> npt.NDArray[np.generic]:
    """Resample slices in position blocks, sharing positions within the cache budget."""
    flat = np.empty((len(slices), plan.total), dtype=plan.output_dtype)
    blocks = [
        (start, min(start + plan.block_points, plan.total))
        for start in range(0, plan.total, plan.block_points)
    ]
    if plan.order <= 1:
        # No coefficients to hold: locate each block once and reuse it for every slice.
        for start, stop in blocks:
            positions = _block_positions(plan, start, stop)
            for index, volume in enumerate(slices):
                flat[index, start:stop] = _interpolate(plan, volume, positions, stop - start)
    else:
        # Hold one slice's coefficients at a time; keep positions across slices only
        # when they fit the cache budget, otherwise locate again per slice.
        cache = plan.total * len(plan.source_dims) * 8 <= POSITION_CACHE_BYTES
        cached = [_block_positions(plan, start, stop) for start, stop in blocks] if cache else None
        for index, volume in enumerate(slices):
            filtered = _coefficients(plan, volume)
            for number, (start, stop) in enumerate(blocks):
                positions = (
                    cached[number] if cached is not None else _block_positions(plan, start, stop)
                )
                flat[index, start:stop] = _interpolate(plan, filtered, positions, stop - start)
    return flat.reshape((*leading, *plan.target_shape))


def _resample_block(plan: _Plan, values: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Split non-geometry axes into slices and dispatch the planned sampling path."""
    leading = values.shape[: values.ndim - len(plan.source_dims)]
    if plan.total == 0:
        return np.empty((*leading, *plan.target_shape), dtype=plan.output_dtype)
    slices = [values[index] for index in np.ndindex(*leading)]
    if plan.lattice_map is None:
        return _general_block(plan, slices, leading)
    return _lattice_block(plan, slices, leading)


def resample(
    source: Geometry,
    target: Geometry | Grid,
    *,
    transform: SupportsPoints | None = None,
    method: Method = "linear",
    fill_value: float = np.nan,
    domain: Domain = "samples",
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

    Domain: by default values are defined between the outer source samples, as for xarray's
    ``interp``, and ``fill_value`` marks everything else. ``domain="cells"`` also answers in the
    outer samples' cells, holding the edge sample's value there for every method
    (ITK's domain is the same; its nearest and linear interpolators also hold the edge).
    Each source axis's :attr:`~xarrayrf.ArrayCoordinates.sample_offset` says where its cells
    lie; a point-sampled axis has none and reaches no further than its samples. Interpolation
    between samples is the same in both domains.

    Empty targets return empty values without locating or interpolating samples. Empty sources
    with non-empty targets raise ValueError, regardless of the domain or ``fill_value``.

    Performance: when both arrays form regular lattices and every transform is affine, target
    positions map to source positions through one composed affine, and each non-geometry slice
    is interpolated in a single compiled ``scipy.ndimage.affine_transform`` pass with a
    separable outside mask; ``block_points`` does not apply. Otherwise target coordinates are
    mapped through the transforms and the source's exact inverse in blocks of ``block_points``
    samples, and ``scipy.ndimage.map_coordinates`` interpolates; for nearest and linear each
    block's positions are reused for every slice. Cubic spline coefficients are computed for one
    slice at a time. A Dask-backed source is processed lazily, one task per chunk of its
    non-geometry dimensions; positions and masks are shared within a task, not across tasks.
    Install the ``resample`` extra for scipy.

    Args:
        source: The array whose values are resampled, with its geometry.
        target: Geometry or Grid whose samples receive values; target pixels are ignored.
        transform: The transform from the target's frame to the source's frame, required when
            they are different frames.
        method: ``"nearest"``, ``"linear"`` or ``"cubic"`` (spline) interpolation.
        fill_value: Value for target samples outside the domain.
        domain: ``"samples"`` or ``"cells"``, as above.
        block_points: Target samples located per block.

    Returns:
        The resampled values over the source's non-geometry dimensions and the target's geometry
        dimensions, with the target's geometry coordinates and the source's non-geometry ones.
        Complex128 for complex sources and float64 otherwise, except that nearest-neighbour
        resampling of an integer or boolean array keeps its dtype when ``fill_value`` is
        representable in it.

    Raises:
        TypeError: If an argument has the wrong type, or ``transform`` is not a point transform.
        ValueError: If the frames are different and no transform is given, ``transform`` has the
            wrong endpoints, the source is empty and the target is not, ``method`` or ``domain``
            is unknown, a source axis declaring cells
            has a single sample in the cells domain, the
            source's samples cannot be located (no
            exact inverse, a retained scalar or multidimensional source coordinate, or
            non-monotonic coordinates), or a target geometry dimension name is used by a
            non-geometry dimension of the source, or a target geometry coordinate name
            collides with a source non-geometry coordinate.
        ImportError: If scipy is not installed.
    """
    plan, source_values = _plan(
        source,
        target,
        transform=transform,
        method=method,
        fill_value=fill_value,
        domain=domain,
        block_points=block_points,
    )
    other_dims = [str(dim) for dim in source.array.dims if dim not in plan.source_dims]
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
        partial(_resample_block, plan),
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
    return result.drop_vars(
        [name for name in result.coords if set(result.coords[name].dims) & set(plan.source_dims)],
        errors="ignore",
    ).assign_coords({**source_coordinates, **target_coordinates})
