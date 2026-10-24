"""Private planning and execution of resampling on NumPy sampling descriptions."""

from __future__ import annotations

import math
import os
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import islice, product
from numbers import Real
from types import ModuleType
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt

from ._affine import AffineTransform, equilibrated_inverse
from ._array_coordinates import ArrayCoordinates
from ._box import BoxOperator, as_operator, interval_tolerance, weights
from ._field_transform import BSplineTransform, DisplacementFieldTransform, PositionFieldTransform
from ._frame import ReferenceFrame
from ._frame_adoption import adopt_frame
from ._frame_compatibility import anonymous_frame_difference
from ._orientation import coordinate_system_change
from ._periodic import CUBIC_HALO, PeriodicAxes, periodic_axes
from ._positions import (
    check_domain,
    extents,
    lattice,
    lattice_parts,
    located_axes,
    locator,
    reach,
    sample_columns,
)
from ._sampling import (
    LATTICE_TOLERANCE,
    POSITION_SLACK,
    AxisSampling,
    Domain,
    Sampling,
    coordinate_to_position,
    freeze_intervals,
)
from ._sitk_engine import FieldEngine, direction_and_spacing, index_transform, simpleitk
from ._transform import SupportsAffine, SupportsPoints, check_transform
from ._validity import chain, has_raising_field, masked_map

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

THREAD_BYTES_PER_POINT: Final = 160
"""Bytes a worker holds per in-flight target point: indices, positions, field rows and their
temporaries. In-flight blocks share ``POSITION_CACHE_BYTES`` across workers."""

MIN_THREAD_POINTS: Final = 1 << 16
"""Target points per worker below which threads cost more than they save."""
"""Target samples located per block: bounds the position arrays to a few tens of megabytes."""

CROP_MARGIN: Final = {0: 1, 1: 1, 3: 24}
"""Source samples kept beyond the target's footprint when cropping, per spline order.

Nearest and linear read at most one neighbour past a position. The cubic prefilter is not
local: its influence decays by the spline pole |z| = 0.268 per sample, so 24 samples leave
0.268**24 ~ 2e-14 of a boundary change, below float64 resolution of the result.
"""

type PositionMap = Callable[[npt.NDArray[np.intp]], npt.NDArray[np.float64]]
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
    periodic: PeriodicAxes | None = None,
) -> PositionMap:
    """Map through target coordinates, the frames and the source's exact inverse."""
    samplings = target.axes
    transform = target.transform
    locate = locator(source, domain, periodic=periodic)
    source_axes = [source.dims.index(dim) for dim in source_dims]
    identity_order = source_dims == source.dims

    masked = has_raising_field(frame_map)

    def positions(target_positions: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
        points = transform.transform_point(sample_columns(samplings, target_positions, target_dims))
        if masked:
            # Targets a "raise" field does not map have no source position: they are filled.
            assert frame_map is not None
            points, _, valid = masked_map(frame_map, np.asarray(points, dtype=np.float64))
            located = np.full((len(source.dims), points.shape[0]), np.nan)
            if valid.any():
                located[:, valid] = locate(points[valid]).T
        else:
            if frame_map is not None:
                points = frame_map.transform_point(points)
            located = locate(points).T
        return located if identity_order else located[source_axes]

    return positions


def _affine_parts(
    members: list[SupportsPoints], size: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Matrix and translation of affine members applied in order, or None for a non-affine."""
    matrix, translation = np.eye(size), np.zeros(size)
    for member in members:
        if not isinstance(member, SupportsAffine):
            return None
        linear = np.asarray(member.matrix, dtype=np.float64)
        if linear.shape != (size, size):
            return None
        matrix = linear @ matrix
        translation = linear @ translation + np.asarray(member.translation, dtype=np.float64)
    return matrix, translation


ENGINE_MARGIN: Final = 1e-9
"""Index units by which every target-box corner must clear a raising or clamping field's
support for SimpleITK, whose exterior is the identity, to stand in for it."""


def _field_engine(
    field: DisplacementFieldTransform,
    field_linear: npt.NDArray[np.float64],
    field_offset: npt.NDArray[np.float64],
    target_shape: tuple[int, ...],
) -> FieldEngine | None:
    """A SimpleITK engine for this field and target lattice, when ITK's semantics are ours.

    That needs SimpleITK; two or three dimensions, the only ones its transforms have; centred
    cells on every axis (support ``[-0.5, n - 0.5)``, ITK's);
    the identity exterior, or every target-box corner inside the support by ``ENGINE_MARGIN``
    (the box is convex, so then no point reaches the exterior); and a target lattice whose
    field-index columns are orthogonal, as an ITK image's direction and spacing are.
    """
    sitk = simpleitk()
    lookup = field._lookup
    if sitk is None or len(lookup.shape) not in (2, 3) or 0 in target_shape:
        return None
    if len(target_shape) != len(lookup.shape):
        return None
    if not (np.all(lookup.lower == -0.5) and np.all(lookup.upper == np.array(lookup.shape) - 0.5)):
        return None
    if field.outside != "identity":
        corners = np.array(list(product(*((0, n - 1) for n in target_shape))), dtype=np.float64)
        reached = corners @ field_linear.T + field_offset
        if np.any(reached < lookup.lower + ENGINE_MARGIN) or np.any(
            reached >= lookup.upper - ENGINE_MARGIN
        ):
            return None
    split = direction_and_spacing(field_linear)
    if split is None:
        return None
    if field._engine_cache is None:
        assert lookup.index_map is not None
        _, linear = lookup.index_map
        # Displacements in index units: the field's own map from frame vectors to indices.
        index_field = np.einsum("kj,...j->...k", linear, np.asarray(field.field))
        field._engine_cache = index_transform(sitk, index_field)
    return FieldEngine(sitk, field._engine_cache, field_linear, field_offset, target_shape, *split)


def _field_lattice_map(
    source: Sampling,
    target: Sampling,
    source_dims: tuple[str, ...],
    target_dims: tuple[str, ...],
    frame_map: SupportsPoints | None,
    domain: Domain,
) -> tuple[PositionMap, int] | None:
    """Fuse lattice target -> affines -> one field -> affines -> lattice source, if possible.

    With ``x(i)`` the target lattice, ``A`` and ``B`` the affines before and after the field
    ``F``, and the source lattice ``w = S s + o``, a target index ``i`` maps to field index
    ``p = P i + c`` and source position ``s = Q i + K u(p) + q``. Every constant cancels its
    origins before scaling (the target point of index zero against the field's anchor and the
    source origin), so a large origin never multiplies an index. Boundary decisions may differ
    from the general path's by roundoff (within about 1e-9 index units). Returns None, for the
    general path, unless the field is exactly a built-in field type on a uniform grid, composites
    and pairs are exactly the built-in types, every other member is affine and square, and both
    samplings are lattices (to ``LATTICE_TOLERANCE``) without single-sample source dims.

    With the map comes the block alignment, in target samples: whole rows of the first target
    dimension when SimpleITK evaluates the field (``_field_engine``), else 1. A block that is
    not whole rows falls back to the NumPy evaluation.
    """
    if frame_map is None:
        return None
    members = chain(frame_map)
    fields = [
        index for index, member in enumerate(members) if not isinstance(member, SupportsAffine)
    ]
    if len(fields) != 1:
        return None
    field = members[fields[0]]
    # A position field is not a displacement, which the index formula above assumes.
    if type(field) not in (DisplacementFieldTransform, BSplineTransform):
        return None
    assert isinstance(field, DisplacementFieldTransform | BSplineTransform)
    if field._lookup.index_map is None:
        return None
    size = len(field.source.axes)
    before = _affine_parts(members[: fields[0]], size)
    after = _affine_parts(members[fields[0] + 1 :], size)
    if before is None or after is None or not isinstance(target.transform, SupportsAffine):
        return None
    try:
        # As the affine lattice path: an axis uniform to LATTICE_TOLERANCE (1e-6 of a step)
        # is a lattice; positions, and the domain's edges, then differ from the general path's
        # exact inversion by at most about that fraction of a sample (the agreed precision).
        target_origin, target_columns = lattice_parts(
            target, target_dims, LATTICE_TOLERANCE, single_samples=True
        )
        source_lattice = lattice(source, source_dims, tolerance=LATTICE_TOLERANCE)
        source_inverse = equilibrated_inverse(source_lattice.matrix)
    except (TypeError, ValueError):
        return None
    if source_inverse.shape != (size, size) or target_origin.shape != (size,):
        return None
    anchor, linear = field._lookup.index_map
    (pre, pre_offset), (post, post_offset) = before, after
    origin = pre @ target_origin + pre_offset
    field_linear = linear @ pre @ target_columns
    field_offset = linear @ (origin - anchor)
    push = source_inverse @ post
    source_linear = push @ pre @ target_columns
    source_offset = source_inverse @ ((post @ origin + post_offset) - source_lattice.origin)
    reach = extents(source, domain)
    reach = [reach[source.dims.index(dim)] for dim in source_dims]
    sizes = np.array([source.sizes[dim] for dim in source_dims], dtype=np.float64)
    lowest = -np.array([low for low, _ in reach])[:, None]
    highest = (sizes - 1.0 + np.array([high for _, high in reach]))[:, None]

    shape = tuple(target.sizes[dim] for dim in target_dims)
    engine = (
        _field_engine(field, field_linear, field_offset, shape)
        if type(field) is DisplacementFieldTransform
        else None
    )
    row = math.prod(shape[1:])
    # Index-unit displacements reach the source through the field's inverse index map.
    index_push = push @ equilibrated_inverse(field._lookup.index_map[1]) if engine else push

    def whole_rows(indices: npt.NDArray[np.intp]) -> tuple[int, int] | None:
        count = indices.shape[1]
        if not count or count % row or np.any(indices[1:, 0]):
            return None
        first = int(indices[0, 0])
        return first, first + count // row

    def positions(indices: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
        steps = indices.astype(np.float64)
        rows = whole_rows(indices) if engine is not None else None
        if rows is not None:
            assert engine is not None
            displacement = engine.index_displacement(*rows, threads=1)
            linear_push = index_push
        else:
            columns: npt.NDArray[np.float64] = np.einsum("ij,jn->in", field_linear, steps)
            columns += field_offset[:, None]
            # Under "raise" a column outside the support has a NaN displacement, so a NaN
            # source position, and is filled.
            displacement, _ = field._displacement_rows(columns, masked=True)
            linear_push = push
        located: npt.NDArray[np.float64] = np.einsum("ij,jn->in", source_linear, steps)
        located += np.einsum("ij,jn->in", linear_push, displacement)
        located += source_offset[:, None]
        # As coordinate_to_position: outside the domain beyond the slack is NaN, else clamped.
        outside = np.any(
            (located < lowest - POSITION_SLACK) | (located > highest + POSITION_SLACK), axis=0
        )
        np.clip(located, lowest, highest, out=located)
        located[:, outside] = np.nan
        return located

    return positions, row if engine is not None else 1


def _target_frame_points(
    target: Sampling, target_dims: tuple[str, ...], target_shape: tuple[int, ...]
) -> Callable[[int, int], npt.NDArray[np.float64]]:
    """Target-frame points ``(stop - start, k)`` of a flattened block of target samples."""

    def points(start: int, stop: int) -> npt.NDArray[np.float64]:
        # A fully scalar target is one point with no dimension to index, as in _block_positions.
        indices = (
            np.array(np.unravel_index(np.arange(start, stop), target_shape), dtype=np.intp)
            if target_shape
            else np.empty((0, stop - start), dtype=np.intp)
        )
        return target.transform.transform_point(sample_columns(target.axes, indices, target_dims))

    return points


def _output_dtype(
    dtype: np.dtype[np.generic], method: Method, fill_value: float
) -> np.dtype[np.generic]:
    """The result dtype: floating data keeps its precision, everything else becomes float64.

    Data stored as float32 (or complex64) declares that precision sufficient, so the result,
    the largest array a resampling allocates, keeps it; positions and weights stay float64.
    float16 becomes float32, which SciPy interpolates, and wider types become float64.
    """
    if dtype.kind == "c":
        output = np.dtype(np.complex64 if dtype.itemsize <= 8 else np.complex128)
    elif dtype.kind == "f":
        output = np.dtype(np.float32 if dtype.itemsize <= 4 else np.float64)
    elif (
        method == "nearest"
        and dtype.kind in "biu"
        and not np.isnan(fill_value)
        and float(np.asarray(fill_value).astype(dtype)) == fill_value
    ):
        return dtype
    else:
        output = np.dtype(np.float64)
    if np.isfinite(fill_value):
        with np.errstate(over="ignore"):
            overflows = not np.isfinite(np.asarray(fill_value, dtype=output))
        if overflows:
            raise ValueError(
                f"fill_value {fill_value!r} overflows the {output} result of {dtype} values; "
                "use a smaller fill_value, or cast the values to float64 first"
            )
    return output


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
    _refuse_latitude_slabs(source, classes)
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


def _refuse_latitude_slabs(source: Sampling, classes: tuple[_AxisClass, ...]) -> None:
    """Refuse box averaging along latitude: a mean over degrees of latitude is not an area mean.

    Longitude slabs stay allowed; at fixed latitude a mean over longitude degrees is an
    arc-length mean. Pass-through axes are exact index matches and average nothing.
    """
    roles = source.frame.coordinate_system.axis_roles
    if "latitude" not in roles:
        return
    assert isinstance(source.transform, SupportsAffine)  # box methods require affine maps
    latitude = source.transform.matrix[roles.index("latitude")]
    for cls in classes:
        if cls.pass_through:
            continue
        # Identity, not equality: AxisSampling holds arrays. Columns follow source.axes.
        column = next(i for i, axis in enumerate(source.axes) if axis is cls.source_axis)
        if abs(latitude[column]) > ROUNDING_ALLOWANCE * np.max(np.abs(latitude)):
            raise ValueError(
                f"box methods refuse source axis {cls.source_axis.axis!r}, which runs along "
                "latitude: a mean over degrees of latitude is not an area mean. Use a point "
                "method, or a conservative regridding tool",
            )


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
    frame_map: SupportsPoints | None
    target_frame_points: Callable[[int, int], npt.NDArray[np.float64]]
    extend: Callable[[npt.NDArray[np.generic], float], npt.NDArray[np.generic]] | None = None
    """Halo extension of each source block for a wrapping or pole-crossing source; the float
    multiplies the rows mirrored across a pole."""
    pole_halo: bool = False
    """Whether source rows are mirrored across a pole, so component signs matter."""
    workers: int = 1
    """Threads over blocks or slabs; 1 runs inline."""


def check_workers(workers: object) -> int | None:
    """Validate ``workers``: None for automatic, or a positive integer."""
    if workers is None:
        return None
    if isinstance(workers, bool) or not isinstance(workers, int | np.integer):
        raise TypeError(f"workers must be None or a positive integer, got {workers!r}")
    if workers < 1:
        raise ValueError(f"workers must be None or a positive integer, got {workers!r}")
    return int(workers)


_THREAD_SAFE_TYPES: Final = (
    AffineTransform,
    DisplacementFieldTransform,
    BSplineTransform,
    PositionFieldTransform,
)
"""Point transforms whose evaluation keeps no mutable state; anything else runs inline."""


def _thread_safe(*transforms: SupportsPoints | None) -> bool:
    """Whether every member, after opening built-in composites and pairs, is a known type."""
    return all(
        transform is None or all(type(member) in _THREAD_SAFE_TYPES for member in chain(transform))
        for transform in transforms
    )


def _thread_count(workers: int | None, total: int, safe: bool) -> int:
    """The worker count: as requested, or automatic (every core, at least
    ``MIN_THREAD_POINTS`` each); one when the transforms are not known to be thread-safe."""
    if not safe:
        return 1
    if workers is not None:
        return workers
    return max(1, min(os.cpu_count() or 1, total // MIN_THREAD_POINTS))


def _thread_block(block_points: int, total: int, workers: int) -> int:
    """Points per block: ``block_points`` inline; threaded, small enough for every worker's
    in-flight block to share the position budget and for at least two blocks per worker."""
    if workers == 1:
        return block_points
    budget = POSITION_CACHE_BYTES // (THREAD_BYTES_PER_POINT * workers)
    return max(1, min(block_points, budget, math.ceil(total / (2 * workers))))


def _run[T](plan: _Plan, function: Callable[[T], object], items: Iterable[T]) -> None:
    """Apply ``function`` to each item, on ``plan.workers`` threads when there are several.

    At most two items per worker are submitted at a time, so queued work and the buffers of
    failed items stay bounded however many items there are. Results are consumed in item
    order, so the first failing item's exception is raised whatever the threads' timing;
    queued items are then cancelled.
    """
    if plan.workers == 1:
        for item in items:
            function(item)
        return
    remaining = iter(items)
    pool = ThreadPoolExecutor(max_workers=plan.workers)
    pending = deque(pool.submit(function, item) for item in islice(remaining, 2 * plan.workers))
    try:
        while pending:
            pending.popleft().result()
            pending.extend(pool.submit(function, item) for item in islice(remaining, 1))
    except BaseException:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)


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
    pole_halos: bool = True,
    workers: int | None = None,
    fuse_raising_fields: bool = True,
) -> _Plan:
    """Plan interpolation for trailing source axes and output axes in explicit storage order.

    ``other_dims`` names source context dimensions for collision validation. ``window`` must
    be applied to the values before execution; lattice offsets and bounds are relative to it.
    ``source_order`` and ``target_order`` must name exactly their sampling's dims, once each.
    Coordinate reads are deferred until option and frame checks have passed.
    ``adoption_suffices`` is called only for the anonymous-frame diagnostic; without it,
    adoption is treated as insufficient. ``pole_halos`` allows interpolation across a pole of
    a global longitude/latitude source; it is off for declared components in a basis with no
    pole rule, and ``execute`` takes the signs of those that have one. ``workers`` is None for
    automatic threads, or a count; transforms not known to be thread-safe always run inline.
    ``fuse_raising_fields`` False keeps a field declared ``"raise"`` off the fused lattice path,
    so the targets it fills are exactly those that pointwise evaluation of the same target
    points classifies as invalid (the component stage evaluates them that way).

    Raises:
        ValueError: If either storage order does not name exactly its sampling's dims.
    """
    order = check_options(method, block_points, support, min_coverage)
    workers = check_workers(workers)
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
    fused = False
    alignment = 1
    box = None
    periodic: PeriodicAxes | None = None
    if total and any(source.sizes[dim] == 0 for dim in source_dims):
        raise ValueError("cannot resample an empty source: nothing to sample from")
    if method in BOX_METHODS:
        box = _box_operator(
            source, target, frame_map, source_dims, target_dims, method, support, min_coverage
        )
    elif total:
        located = located_axes(source)
        periodic = periodic_axes(
            source,
            halo=CUBIC_HALO if order == 3 else 1,
            pole_halos=pole_halos,
            domain=domain,
            extents={
                axis: extent
                for (_, axis, _), extent in zip(located, reach(located, domain), strict=True)
            },
        )
        if periodic is None:
            source_reach = extents(source, domain)
            source_reach = [source_reach[source.dims.index(dim)] for dim in source_dims]
            lattice_map = _lattice_map(
                source, target, source_dims, target_dims, frame_map, source_reach
            )
        if lattice_map is None:
            # A modulo or a mirror is not affine, and a crop could drop samples across the
            # seam, so a periodic source takes the general path.
            mapped = (
                _field_lattice_map(source, target, source_dims, target_dims, frame_map, domain)
                if periodic is None and (fuse_raising_fields or not has_raising_field(frame_map))
                else None
            )
            # The fused map evaluates only the built-in field; the general one evaluates the
            # target grid, the frame map and the source's inverse.
            fused = mapped is not None
            if mapped is not None:
                position_map, alignment = mapped
            else:
                position_map = _general_map(
                    source, target, source_dims, target_dims, frame_map, domain, periodic
                )
    output_dtype = _output_dtype(dtype, method, fill_value)
    safe = (
        box is not None
        or lattice_map is not None
        or fused
        or _thread_safe(frame_map, target.transform, source.transform)
    )
    threads = 1 if box is not None else _thread_count(workers, total, safe)
    block_points = _thread_block(block_points, total, threads)
    # SimpleITK evaluates whole rows of the first target dimension; when a row exceeds the
    # bounded block, blocks keep their bound and NumPy evaluates them.
    if alignment <= block_points:
        block_points = block_points // alignment * alignment
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

    extend = None
    if periodic is not None and total:
        axis_of = {sampling.dim: index for index, sampling in enumerate(source.axes)}
        source_shape = tuple(
            periodic.extended_size(axis_of[dim], size)
            for dim, size in zip(source_dims, source_shape, strict=True)
        )
        storage = {axis_of[dim]: k for k, dim in enumerate(source_dims)}
        lon_dim = storage[periodic.longitude.axis]
        lat_dim = None if periodic.latitude is None else storage[periodic.latitude.axis]

        def extend(volume: npt.NDArray[np.generic], sign: float) -> npt.NDArray[np.generic]:
            return periodic.extend(volume, lon_dim, lat_dim, sign)

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
        frame_map=frame_map,
        target_frame_points=_target_frame_points(target, target_dims, target_shape),
        extend=extend,
        pole_halo=periodic is not None and periodic.latitude is not None,
        workers=threads,
    )


def _coefficients(plan: _Plan, volume: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Cubic spline coefficients for one slice; nearest and linear use values directly."""
    if plan.order <= 1:
        return volume
    # SciPy filters each line in double precision and stores the coefficients in the result
    # dtype between axes. The cubic prefilter gains at most 3 per axis (at the Nyquist
    # frequency), so values within 3**ndim of the dtype's maximum keep double coefficients
    # rather than overflow where the interpolated result would fit.
    dtype = plan.output_dtype
    if dtype in _COMPACT_DTYPES and volume.size:
        limit = float(np.finfo(cast(Any, dtype)).max) / 3.0**volume.ndim
        if float(np.abs(volume).max()) > limit:
            dtype = np.dtype(np.complex128 if dtype.kind == "c" else np.float64)
    filtered: npt.NDArray[np.generic] = plan.ndimage.spline_filter(
        volume, order=plan.order, mode="nearest", output=dtype
    )
    return filtered


_COMPACT_DTYPES: Final = (np.dtype(np.float32), np.dtype(np.complex64))


def _weight_dtype(plan: _Plan) -> np.dtype[np.generic]:
    """Missing-value weights stay double: they are compared with ``ROUNDING_ALLOWANCE``."""
    return np.dtype(np.complex128 if plan.output_dtype.kind == "c" else np.float64)


def _linear_inputs(
    volume: npt.NDArray[np.generic],
) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic] | None]:
    """Zero missing components and encode their weights once for one source slice."""
    if not np.isnan(volume).any():
        return volume, None
    values = cast(npt.NDArray[np.number[Any]], volume.copy())
    # Zero and one are exact in any floating dtype, so the mask matches the values' dtype.
    missing = cast(npt.NDArray[np.number[Any]], np.zeros(volume.shape, dtype=volume.dtype))
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
    out: npt.NDArray[np.generic],
    missing: npt.NDArray[np.generic] | None = None,
) -> None:
    """Interpolate one position block into ``out``, holding edge values within outer cells.

    A block wholly inside the domain (the common case) is interpolated straight into ``out``,
    with no intermediate array, mask or scatter.
    """
    inside = ~np.isnan(positions[0])
    every = bool(inside.all())
    clipped = positions if every else positions[:, inside]
    # Beyond the outer samples, in their cells, the edge value holds; a cubic spline would
    # otherwise extrapolate there. Most blocks lie within the samples and need no clip.
    if clipped.size and (
        clipped.min() < 0.0 or np.any(clipped.max(axis=1, keepdims=True) > plan.source_last)
    ):
        clipped = np.clip(clipped, 0.0, plan.source_last)
    sampled: npt.NDArray[np.generic] = plan.ndimage.map_coordinates(
        volume,
        clipped,
        order=plan.order,
        mode="nearest",
        prefilter=False,
        output=out if every else plan.output_dtype,
    )
    if missing is not None:
        weights = plan.ndimage.map_coordinates(
            missing,
            clipped,
            order=plan.order,
            mode="nearest",
            prefilter=False,
            output=_weight_dtype(plan),
        )
        _restore_missing(sampled, weights)
    if not every:
        out[...] = plan.fill_value
        out[inside] = sampled


def _block_positions(plan: _Plan, start: int, stop: int) -> npt.NDArray[np.float64]:
    """Locate a flattened block of target samples in the source."""
    assert plan.position_map is not None
    target_positions = (
        np.array(np.unravel_index(np.arange(start, stop), plan.target_shape), dtype=np.intp)
        if plan.target_shape
        else np.empty((0, stop - start), dtype=np.intp)
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
        _affine_slabs(plan, lattice_map, filtered, output[index])
        if missing is not None:
            weights = np.empty(target_shape, dtype=_weight_dtype(plan))
            _affine_slabs(plan, lattice_map, missing, weights)
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


def _affine_slabs(
    plan: _Plan,
    lattice_map: _LatticeMap,
    values: npt.NDArray[np.generic],
    output: npt.NDArray[np.generic],
) -> None:
    """``affine_transform`` into ``output``, in slabs along its first axis on several workers.

    A slab starting at target row ``a`` is the same map with offset ``offset + linear[:, 0] a``;
    positions agree with the whole pass to roundoff.
    """
    rows = output.shape[0]
    count = min(rows, 2 * plan.workers) if plan.workers > 1 else 1
    edges = np.linspace(0, rows, count + 1).astype(int)

    def slab(bounds: tuple[int, int]) -> None:
        start, stop = bounds
        if start == stop:
            return
        plan.ndimage.affine_transform(
            values,
            lattice_map.linear,
            offset=lattice_map.offset + lattice_map.linear[:, 0] * start,
            output_shape=(stop - start, *output.shape[1:]),
            output=output[start:stop],
            order=plan.order,
            mode="nearest",
            prefilter=False,
        )

    _run(plan, slab, list(zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True)))


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
        def every_slice(block: tuple[int, int]) -> None:
            start, stop = block
            positions = _block_positions(plan, start, stop)
            for index, volume in enumerate(slices):
                _interpolate(plan, volume, positions, flat[index, start:stop])

        _run(plan, every_slice, blocks)
        return flat.reshape((*leading, *plan.target_shape))
    # Hold one slice's coefficients or NaN buffers at a time; keep positions only when they
    # fit the cache budget, otherwise locate again per slice.
    cached: list[npt.NDArray[np.float64]] | None = None
    if plan.total * len(plan.source_dims) * 8 <= POSITION_CACHE_BYTES:
        cached = [np.empty(0)] * len(blocks)

        def locate(number: int) -> None:
            assert cached is not None
            cached[number] = _block_positions(plan, *blocks[number])

        _run(plan, locate, range(len(blocks)))

    def one_slice(
        index: int,
        filtered: npt.NDArray[np.generic],
        missing: npt.NDArray[np.generic] | None,
        number: int,
    ) -> None:
        start, stop = blocks[number]
        positions = cached[number] if cached is not None else _block_positions(plan, start, stop)
        _interpolate(plan, filtered, positions, flat[index, start:stop], missing)

    for index, volume in enumerate(slices):
        filtered, missing = (
            _linear_inputs(volume) if plan.order == 1 else (_coefficients(plan, volume), None)
        )
        _run(plan, partial(one_slice, index, filtered, missing), range(len(blocks)))
        del filtered, missing
    return flat.reshape((*leading, *plan.target_shape))


def execute(
    plan: _Plan,
    values: npt.NDArray[np.generic],
    signs: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.generic]:
    """Split non-geometry axes into slices and dispatch the planned sampling path.

    ``signs``, broadcastable to the non-geometry shape, multiplies the rows a pole halo mirrors
    (declared components that reverse across a pole); ``None`` leaves them as they are.
    """
    leading = values.shape[: values.ndim - len(plan.source_dims)]
    if plan.total == 0:
        return np.empty((*leading, *plan.target_shape), dtype=plan.output_dtype)
    if plan.box is not None:
        # Box sums accumulate in float64; only the result takes the output dtype.
        return plan.box.apply(
            values, len(plan.source_dims), plan.target_dims, plan.fill_value
        ).astype(plan.output_dtype, copy=False)
    if plan.order == 3 and np.isnan(values).any():
        # The spline prefilter is global along each line, so one NaN would empty its line,
        # except where the same-grid gather bypasses the prefilter: refuse both alike.
        raise ValueError(
            "cubic resampling refuses missing (NaN) source values, which the spline prefilter "
            "spreads along whole lines; fill or mask them first, or use method='linear'"
        )
    if values.dtype.kind == "f" and values.dtype.itemsize < 4:
        # SciPy does not interpolate float16; the float32 result dtype holds it exactly.
        values = values.astype(plan.output_dtype)
    indices = list(np.ndindex(*leading))
    slices = [values[index] for index in indices]
    if plan.extend is not None:
        if signs is None:
            slices = [plan.extend(volume, 1.0) for volume in slices]
        else:
            per_slice = np.broadcast_to(np.asarray(signs), leading)
            slices = [
                plan.extend(volume, float(per_slice[index]))
                for volume, index in zip(slices, indices, strict=True)
            ]
    if plan.lattice_map is None:
        return _general_block(plan, slices, leading)
    return _lattice_block(plan, slices, leading)
