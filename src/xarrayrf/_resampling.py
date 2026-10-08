"""Private planning and execution of resampling on NumPy sampling descriptions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from numbers import Real
from types import ModuleType
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt

from ._affine import AffineTransform, equilibrated_inverse
from ._array_coordinates import ArrayCoordinates
from ._box import BoxOperator, as_operator, interval_tolerance, weights
from ._frame import ReferenceFrame
from ._frame_adoption import adopt_frame
from ._frame_compatibility import anonymous_frame_difference
from ._orientation import coordinate_system_change
from ._positions import check_domain, extents, lattice, lattice_parts, locator, sample_columns
from ._sampling import (
    LATTICE_TOLERANCE,
    POSITION_SLACK,
    AxisSampling,
    Domain,
    Sampling,
    coordinate_to_position,
    freeze_intervals,
)
from ._transform import SupportsAffine, SupportsPoints, check_transform

type Method = Literal["nearest", "linear", "cubic", "step", "overlap_mean"]
type Support = Literal["point", "average"]
BOX_METHODS = ("step", "overlap_mean")

_SPLINE_ORDER: Final = {"nearest": 0, "linear": 1, "cubic": 3}

ROUNDING_ALLOWANCE: Final = 1e-9
"""Index/gather and missing-weight roundoff bound, measured on large-origin oblique grids."""

_SLAB_SAMPLES: Final = 1 << 22
"""Target samples per slab when computing the outside mask of the lattice path."""

POSITION_CACHE_BYTES: Final = 1 << 28
"""Largest size of general-path positions kept across source slices (256 MiB)."""

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
    source: Sampling,
    target: Sampling,
    source_dims: tuple[str, ...],
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
            target, target_dims, LATTICE_TOLERANCE, single_samples=True
        )
        source_lattice = lattice(source, source_dims, tolerance=LATTICE_TOLERANCE)
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
    sizes = np.array([source.sizes[dim] for dim in source_dims], dtype=np.float64)
    reach = np.array(extents, dtype=np.float64).reshape(-1, 2)
    return _LatticeMap(
        combined[: matrix.shape[0], : len(target_dims)],
        combined[: matrix.shape[0], len(target_dims)],
        -reach[:, 0],
        sizes - 1.0 + reach[:, 1],
    )


def _general_map(
    source: Sampling,
    target: Sampling,
    source_dims: tuple[str, ...],
    target_dims: tuple[str, ...],
    frame_map: SupportsPoints | None,
    domain: Domain,
) -> PositionMap:
    """Map through target coordinates, the frames and the source's exact inverse."""
    samplings = target.axes
    transform = target.transform
    locate = locator(source, domain)
    source_axes = [source.dims.index(dim) for dim in source_dims]
    identity_order = source_dims == source.dims

    def positions(target_positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        indices = target_positions.astype(np.intp)
        points = transform.transform_point(sample_columns(samplings, indices, target_dims))
        if frame_map is not None:
            points = frame_map.transform_point(points)
        located = locate(points).T
        return located if identity_order else located[source_axes]

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
        if np.max(np.abs(error)) > ROUNDING_ALLOWANCE:
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
class _AxisClass:
    """One target axis mapped by ``c_s = a * c_t + b``; indices mark pass-through."""

    source_axis: AxisSampling
    a: float
    b: float
    indices: npt.NDArray[np.intp] | None

    @property
    def pass_through(self) -> bool:
        return self.indices is not None


def _classify_axes(
    source: Sampling,
    target: Sampling,
    frame_map: SupportsPoints | None,
    support: Support = "point",
) -> tuple[_AxisClass, ...] | None:
    """Classify in target coordinate-axis order, including retained scalar axes."""
    if not isinstance(source.transform, SupportsAffine) or not isinstance(
        target.transform, SupportsAffine
    ):
        return None
    if frame_map is not None and not isinstance(frame_map, SupportsAffine):
        return None
    matrix = source.transform.matrix
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if any(axis.dim is None or axis.values.size == 0 for axis in source.axes):
        return None
    try:
        inverse = equilibrated_inverse(matrix)
    except ValueError:
        # Claims are metadata: an uninvertible source leaves location errors to the planner.
        return None
    linear = target.transform.matrix
    offset = target.transform.translation
    if frame_map is not None:
        linear = frame_map.matrix @ linear
        offset = frame_map.matrix @ offset + frame_map.translation
    linear = inverse @ linear
    offset = inverse @ (offset - source.transform.translation)
    if linear.shape[0] != linear.shape[1]:
        return None
    # Use the gather allowance relative to each row, so coordinate units do not
    # impose an absolute threshold on a scaled permutation.
    significant = np.abs(linear) > ROUNDING_ALLOWANCE * np.max(
        np.abs(linear), axis=1, keepdims=True
    )
    if not (
        np.all(np.count_nonzero(significant, axis=0) == 1)
        and np.all(np.count_nonzero(significant, axis=1) == 1)
    ):
        return None
    classes = []
    for j, target_axis in enumerate(target.axes):
        i = int(np.flatnonzero(significant[:, j])[0])
        axis = source.axes[i]
        a, b = float(linear[i, j]), float(offset[i])
        # Fractional positions scale the gather allowance by the local spacing;
        # singletons reuse coordinate_to_position's SINGLE_SAMPLE_TOLERANCE rule.
        positions = coordinate_to_position(
            axis.values, a * target_axis.values + b, extrapolate=True
        )
        rounded = np.rint(positions)
        matches = (
            np.isfinite(positions)
            & (np.abs(positions - rounded) <= ROUNDING_ALLOWANCE)
            & (rounded >= 0)
            & (rounded < axis.values.size)
        )
        indices = rounded.astype(np.intp) if np.all(matches) else None
        if support == "average" and indices is not None and target_axis.intervals is not None:
            mapped = np.sort(a * target_axis.intervals + b, axis=-1)
            if axis.intervals is None or np.any(
                np.abs(mapped - axis.intervals[indices]) > interval_tolerance(mapped)[..., None]
            ):
                indices = None
        classes.append(_AxisClass(axis, a, b, indices))
    return tuple(classes)


def _interval_claims(
    source: Sampling,
    target: Sampling,
    frame_map: SupportsPoints | None,
    support: Support = "point",
) -> dict[str, npt.NDArray[np.float64]]:
    classes = _classify_axes(source, target, frame_map, support)
    claims: dict[str, npt.NDArray[np.float64]] = {}
    if classes is None:
        return claims
    declaration = target.transform.source
    assert isinstance(declaration, ArrayCoordinates)
    for axis, classification in zip(target.axes, classes, strict=True):
        if support == "average" and not classification.pass_through:
            if axis.intervals is not None:
                claims[axis.axis] = axis.intervals
            continue
        rows = classification.source_axis.intervals
        if rows is None or classification.indices is None:
            continue
        mapped = (rows[classification.indices] - classification.b) / classification.a
        if classification.a < 0:
            mapped = mapped[..., ::-1]
        try:
            claims.update(
                freeze_intervals(declaration, {axis.axis: axis.values}, {axis.axis: mapped})
            )
        except ValueError:
            # A claim may be incompatible with the target's unchanged sample_offset.
            continue
    return claims


def _box_operator(
    source: Sampling,
    target: Sampling,
    frame_map: SupportsPoints | None,
    source_dims: tuple[str, ...],
    target_dims: tuple[str, ...],
    method: Method,
    support: Support,
    min_coverage: float,
) -> BoxOperator:
    if any(
        not isinstance(t, SupportsAffine)
        for t in (source.transform, target.transform, frame_map)
        if t is not None
    ):
        raise ValueError("box methods require affine transforms; a non-affine transform was given")
    classes = _classify_axes(source, target, frame_map, support)
    if classes is None:
        raise ValueError(
            "box methods require a separable composed map; non-separable map or unsupported source axes"
        )
    coverage = np.ones(tuple(target.sizes[d] for d in target_dims))
    operators = []
    for axis, cls in zip(target.axes, classes, strict=True):
        if cls.indices is not None:
            operator = cls.indices.ravel()
            fraction = np.ones(axis.values.size)
        else:
            slabs = cls.source_axis.intervals
            if slabs is None:
                raise ValueError(
                    f"source axis {cls.source_axis.axis!r} needs intervals; resample that axis first with a point method or declare intervals"
                )
            if support == "average" and axis.intervals is None:
                raise ValueError(f"average target axis {axis.axis!r} needs declared intervals")
            slabs = slabs.reshape(-1, 2)
            if method == "step":
                sorted_slabs = slabs[np.argsort(slabs[:, 0])]
                tolerance = np.maximum(
                    interval_tolerance(sorted_slabs[:-1]), interval_tolerance(sorted_slabs[1:])
                )
                if np.any(sorted_slabs[:-1, 1] - sorted_slabs[1:, 0] > tolerance):
                    raise ValueError(
                        f"step refuses overlapping source intervals on axis {cls.source_axis.axis!r}"
                    )
            coordinates = axis.values
            if support == "average":
                assert axis.intervals is not None
                coordinates = axis.intervals
            mapped = cls.a * coordinates + cls.b
            if support == "average":
                mapped = np.sort(mapped, axis=-1)
                query = slabs
            else:
                # A point within roundoff of an edge (after a unit change or reversal) is on it.
                query = slabs + interval_tolerance(slabs)[:, None] * [-1, 1]
            rows, fraction = weights(query, mapped, average=support == "average")
            operator = as_operator(rows, len(slabs))
        shape = [1] * len(target_dims)
        if axis.dim is not None:
            shape[target_dims.index(axis.dim)] = -1
        coverage *= fraction.reshape(shape)
        assert cls.source_axis.dim is not None
        operators.append((source_dims.index(cls.source_axis.dim), axis.dim, operator))
    return BoxOperator(operators, coverage, min_coverage)


@dataclass(frozen=True)
class _Plan:
    """Validated sampling choices and geometry shared by every block."""

    box: BoxOperator | None
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
    window: tuple[slice, ...] | None
    intervals: dict[str, npt.NDArray[np.float64]]


def check_options(
    method: Method, block_points: int, support: Support = "point", min_coverage: float = 0.5
) -> int:
    """Validate interpolation options before a binding revalidates its geometry."""
    if method not in (*_SPLINE_ORDER, *BOX_METHODS):
        raise ValueError(f"method must be one of {(*_SPLINE_ORDER, *BOX_METHODS)}, got {method!r}")
    if isinstance(block_points, bool) or not isinstance(block_points, int) or block_points < 1:
        raise ValueError(f"block_points must be a positive integer, got {block_points!r}")
    check_support(method, support, min_coverage)
    return _SPLINE_ORDER.get(method, 0)


def check_support(method: Method, support: Support, min_coverage: float) -> None:
    """Validate new options without changing legacy method-error ordering."""
    if support not in ("point", "average"):
        raise ValueError("support must be 'point' or 'average'")
    if support == "average" and method not in BOX_METHODS:
        raise ValueError(
            "average support is not yet supported for this method; use step or overlap_mean"
        )
    if (
        isinstance(min_coverage, bool)
        or not isinstance(min_coverage, Real)
        or not 0 < min_coverage <= 1
    ):
        raise ValueError("min_coverage must be a real number in (0, 1]")


def check_box_values(method: Method, dtype: np.dtype[np.generic], domain: Domain) -> None:
    """Refuse unsupported value and domain declarations before coordinate reads."""
    if method in BOX_METHODS:
        if dtype.kind not in "fc":
            raise TypeError("box methods require floating source values; convert explicitly")
        if domain != "samples":
            raise ValueError(
                "box methods refuse non-default domain; declared support is the domain"
            )


def plan(
    source: Sampling,
    target: Sampling,
    *,
    source_order: tuple[str, ...],
    target_order: tuple[str, ...],
    dtype: np.dtype[np.generic],
    transform: SupportsPoints | None,
    method: Method,
    fill_value: float,
    domain: Domain,
    block_points: int,
    other_dims: tuple[str, ...],
    support: Support = "point",
    min_coverage: float = 0.5,
    adoption_suffices: Callable[[AffineTransform], bool] | None = None,
) -> _Plan:
    """Plan interpolation for trailing source axes and output axes in explicit storage order.

    ``other_dims`` names source context dimensions for collision validation. ``window`` must
    be applied to the values before execution; lattice offsets and bounds are relative to it.
    ``source_order`` and ``target_order`` must name exactly their sampling's dims, once each.
    Coordinate reads are deferred until option and frame checks have passed.
    ``adoption_suffices`` is called only for the anonymous-frame diagnostic; without it,
    adoption is treated as insufficient.

    Raises:
        ValueError: If either storage order does not name exactly its sampling's dims.
    """
    order = check_options(method, block_points, support, min_coverage)
    check_box_values(method, dtype, domain)
    for name, storage_order, sampling in (
        ("source_order", source_order, source),
        ("target_order", target_order, target),
    ):
        if len(storage_order) != len(sampling.dims) or set(storage_order) != set(sampling.dims):
            raise ValueError(f"{name} must name exactly the sampling dims {sampling.dims} once")
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
            suffices = adoption_suffices(adopted) if adoption_suffices is not None else False
        raise ValueError(
            anonymous_frame_difference(
                source,
                target,
                labels=("source operand", "target operand"),
                adoption_suffices=suffices,
            )
        )
    frame_map = _frame_map(target.frame, source.frame, transform)
    target_dims = target_order
    source_dims = source_order
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
    box = None
    if total and any(source.sizes[dim] == 0 for dim in source_dims):
        raise ValueError("cannot resample an empty source: nothing to sample from")
    if method in BOX_METHODS:
        box = _box_operator(
            source, target, frame_map, source_dims, target_dims, method, support, min_coverage
        )
    elif total:
        reach = extents(source, domain)
        reach = [reach[source.dims.index(dim)] for dim in source_dims]
        lattice_map = _lattice_map(source, target, source_dims, target_dims, frame_map, reach)
        if lattice_map is None:
            position_map = _general_map(source, target, source_dims, target_dims, frame_map, domain)
    output_dtype = _output_dtype(dtype, method, fill_value)
    source_shape = tuple(source.sizes[dim] for dim in source_dims) if total else ()
    window = None
    if lattice_map is not None:
        # Read only the source region the target can touch: for a lazily loaded source (a
        # remote COG or OME-Zarr) this is the difference between a crop and the whole file.
        window = _crop_window(
            lattice_map,
            target_shape,
            source_shape,
            order,
        )
        if window is not None:
            source_shape = tuple(w.stop - w.start for w in window)
            shift = np.array([w.start for w in window], dtype=np.float64)
            lattice_map = _LatticeMap(
                lattice_map.linear,
                lattice_map.offset - shift,
                lattice_map.lowest - shift,
                lattice_map.highest - shift,
            )
    same_grid = (
        _same_grid_indices(lattice_map, target_shape, source_shape)
        if lattice_map is not None
        else None
    )

    source_last = np.array([size - 1 for size in source_shape], dtype=np.float64).reshape(-1, 1)

    try:
        from scipy import ndimage
    except ImportError as error:
        raise ImportError("resample needs scipy; install xarrayrf[resample]") from error

    return _Plan(
        box=box,
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
        window=window,
        intervals=_interval_claims(source, target, frame_map, support) if total else {},
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


def _linear_inputs(
    volume: npt.NDArray[np.generic],
) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic] | None]:
    """Zero missing components and encode their weights once for one source slice."""
    if not np.isnan(volume).any():
        return volume, None
    values = cast(npt.NDArray[np.number[Any]], volume.copy())
    missing = np.zeros(volume.shape, dtype=np.complex128 if np.iscomplexobj(volume) else np.float64)
    if np.iscomplexobj(volume):
        for component, mask in ((values.real, missing.real), (values.imag, missing.imag)):
            absent = np.isnan(component)
            mask[absent] = 1.0
            component[absent] = 0.0
    else:
        absent = np.isnan(values)
        missing[absent] = 1.0
        values[absent] = 0.0
    return values, missing


def _restore_missing(values: npt.NDArray[np.generic], weights: npt.NDArray[np.generic]) -> None:
    """Propagate positive missing weight, allowing measured position roundoff."""
    numeric_values = cast(npt.NDArray[np.number[Any]], values)
    numeric_weights = cast(npt.NDArray[np.number[Any]], weights)
    numeric_values.real[numeric_weights.real > ROUNDING_ALLOWANCE] = np.nan
    if np.iscomplexobj(values):
        numeric_values.imag[numeric_weights.imag > ROUNDING_ALLOWANCE] = np.nan


def _interpolate(
    plan: _Plan,
    volume: npt.NDArray[np.generic],
    positions: npt.NDArray[np.float64],
    size: int,
    missing: npt.NDArray[np.generic] | None = None,
) -> npt.NDArray[np.generic]:
    """Interpolate one position block, holding edge values within outer cells."""
    inside = ~np.isnan(positions[0])
    block = np.full(size, plan.fill_value, dtype=plan.output_dtype)
    # Beyond the outer samples, in their cells, the edge value holds; a cubic spline would
    # otherwise extrapolate there.
    clipped = np.clip(positions[:, inside], 0.0, plan.source_last)
    sampled = plan.ndimage.map_coordinates(
        volume,
        clipped,
        order=plan.order,
        mode="nearest",
        prefilter=False,
        output=plan.output_dtype,
    )
    if missing is not None:
        weights = plan.ndimage.map_coordinates(
            missing,
            clipped,
            order=plan.order,
            mode="nearest",
            prefilter=False,
            output=plan.output_dtype,
        )
        _restore_missing(sampled, weights)
    block[inside] = sampled
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
    # Compiled passes per slice, prefiltering only that slice; edges extend so
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
        filtered, missing = (
            _linear_inputs(volume) if plan.order == 1 else (_coefficients(plan, volume), None)
        )
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
        if missing is not None:
            weights = plan.ndimage.affine_transform(
                missing,
                lattice_map.linear,
                offset=lattice_map.offset,
                output_shape=target_shape,
                output=plan.output_dtype,
                order=plan.order,
                mode="nearest",
                prefilter=False,
            )
            _restore_missing(output[index], weights)
            del weights
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
        del filtered, missing
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
    needs_masks = plan.order == 1 and any(np.isnan(volume).any() for volume in slices)
    if plan.order <= 1 and not needs_masks:
        # No coefficients to hold: locate each block once and reuse it for every slice.
        for start, stop in blocks:
            positions = _block_positions(plan, start, stop)
            for index, volume in enumerate(slices):
                flat[index, start:stop] = _interpolate(plan, volume, positions, stop - start)
    else:
        # Hold one slice's coefficients or NaN buffers at a time; keep positions only
        # when they fit the cache budget, otherwise locate again per slice.
        cache = plan.total * len(plan.source_dims) * 8 <= POSITION_CACHE_BYTES
        cached = [_block_positions(plan, start, stop) for start, stop in blocks] if cache else None
        for index, volume in enumerate(slices):
            filtered, missing = (
                _linear_inputs(volume) if plan.order == 1 else (_coefficients(plan, volume), None)
            )
            for number, (start, stop) in enumerate(blocks):
                positions = (
                    cached[number] if cached is not None else _block_positions(plan, start, stop)
                )
                flat[index, start:stop] = _interpolate(
                    plan, filtered, positions, stop - start, missing
                )
            del filtered, missing
    return flat.reshape((*leading, *plan.target_shape))


def execute(plan: _Plan, values: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Split non-geometry axes into slices and dispatch the planned sampling path."""
    leading = values.shape[: values.ndim - len(plan.source_dims)]
    if plan.total == 0:
        return np.empty((*leading, *plan.target_shape), dtype=plan.output_dtype)
    if plan.box is not None:
        return plan.box.apply(values, len(plan.source_dims), plan.target_dims, plan.fill_value)
    slices = [values[index] for index in np.ndindex(*leading)]
    if plan.lattice_map is None:
        return _general_block(plan, slices, leading)
    return _lattice_block(plan, slices, leading)
