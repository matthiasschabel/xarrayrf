"""Canonical anatomical directions and sampling operations on patient grids."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, permutations, product

import numpy as np
import numpy.typing as npt

from ._affine import AffineTransform
from ._array_coordinates import ArrayCoordinates
from ._coordinate_system import CoordinateSystem
from ._grid import Grid
from ._sampling import (
    LATTICE_TOLERANCE,
    AxisSampling,
    cell_extent,
    position_to_coordinate,
    uniform_step,
)
from ._transform import SupportsAffine
from ._validation import check_names, real_float_array
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

ASSIGNMENT_TOLERANCE = 1e-6
"""Maximum difference in summed absolute cosines considered an ambiguous assignment."""

COVERAGE_TOLERANCE = 1e-9
"""Slack in output steps subtracted before rounding a coverage count upward."""

CROSS_ROW_ROUNDOFF = 64 * float(np.finfo(np.float64).eps)
"""Roundoff allowed on a non-anatomical frame axis, relative to covered column contributions.

The check only has to absorb floating-point noise from composing transforms, which scales with
each column's largest coefficient times its own coordinate span.
"""

_LETTER_TOKENS = {
    "R": "left-to-right",
    "L": "right-to-left",
    "A": "posterior-to-anterior",
    "P": "anterior-to-posterior",
    "S": "inferior-to-superior",
    "I": "superior-to-inferior",
}
_TOKEN_LETTERS = {token: letter for letter, token in _LETTER_TOKENS.items()}
_PLANES = {"axial": "SPL", "transverse": "SPL", "coronal": "PIL", "sagittal": "RIP"}


def _orientation(orientation: str | tuple[str, ...], count: int) -> tuple[str, ...]:
    if isinstance(orientation, str):
        letters = _PLANES.get(orientation, orientation)
        if any(letter not in _LETTER_TOKENS for letter in letters):
            raise ValueError("orientation must use case-sensitive patient letters or a named plane")
        tokens = tuple(_LETTER_TOKENS[letter] for letter in letters)
    elif isinstance(orientation, tuple):
        tokens = tuple(VOCABULARY.check(token) for token in orientation)
    else:
        raise TypeError(
            "orientation must be a patient-letter string or a tuple of direction tokens"
        )
    if len(tokens) != count:
        raise ValueError(f"orientation must give {count} directions, got {len(tokens)}")
    if len({VOCABULARY.pair(token) for token in tokens}) != count:
        raise ValueError("orientation directions must use distinct anatomical axes")
    return tokens


def _codes(tokens: tuple[str, ...]) -> str | tuple[str, ...]:
    if all(token in _TOKEN_LETTERS for token in tokens):
        return "".join(_TOKEN_LETTERS[token] for token in tokens)
    return tokens


def _anatomical_grid(
    grid: Grid,
) -> tuple[SupportsAffine, tuple[AxisSampling, ...], tuple[int, ...], npt.NDArray[np.float64]]:
    if not isinstance(grid, Grid):
        raise TypeError("grid must be a Grid")
    transform = grid.transform
    if not isinstance(transform, SupportsAffine):
        raise TypeError("anatomical grid operations require an affine transform")
    system = grid.frame.coordinate_system
    oriented = tuple(i for i, token in enumerate(system.orientation) if token is not None)
    if (
        system.vocabulary != VOCABULARY
        or len(oriented) != 3
        or any(system.axis_types[i] != "space" for i in oriented)
    ):
        raise ValueError("the frame needs three spatial axes oriented in anatomy.VOCABULARY")
    if len({system.units[i] for i in oriented}) != 1:
        raise ValueError("anatomically oriented axes must share one unit for angles and spacing")
    by_dim = {axis.dim: axis for axis in grid._sampling().axes if axis.dim is not None}
    axes = tuple(by_dim[dim] for dim in grid.dims)
    if len(axes) > 3:
        raise ValueError("at most three varying dimensions can be assigned to anatomical axes")
    cosines = np.empty((len(axes), 3))
    for j, axis in enumerate(axes):
        if axis.values.size == 0:
            raise ValueError(f"dimension {axis.dim!r} is empty and has no step direction")
        sign = 1.0
        if axis.values.size > 1:
            differences = np.diff(axis.values)
            if not (np.all(differences > 0) or np.all(differences < 0)):
                raise ValueError(f"dimension {axis.dim!r} needs strictly monotonic coordinates")
            sign = 1.0 if differences[0] > 0 else -1.0
        column = transform.source.axes.index(axis.axis)
        vector = transform.matrix[list(oriented), column] * sign
        with np.errstate(over="ignore", under="ignore"):
            norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"dimension {axis.dim!r} has no finite nonzero anatomical direction")
        cosines[j] = vector / norm
    for j, k in combinations(range(len(axes)), 2):
        if np.linalg.matrix_rank(cosines[[j, k]]) < 2:
            raise ValueError(
                f"dimensions {axes[j].dim!r} and {axes[k].dim!r} point along the same direction"
            )
    return transform, axes, oriented, cosines


def _axis_assignment(
    cosines: npt.NDArray[np.float64], *, ambiguity_message: str
) -> tuple[int, ...]:
    candidates = sorted(
        (
            (sum(abs(float(cosines[j, i])) for j, i in enumerate(order)), order)
            for order in permutations(range(3), len(cosines))
        ),
        reverse=True,
    )
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] <= ASSIGNMENT_TOLERANCE:
        raise ValueError(ambiguity_message)
    return candidates[0][1]


def _assignment(
    grid: Grid,
) -> tuple[SupportsAffine, tuple[AxisSampling, ...], tuple[int, ...], tuple[str, ...]]:
    transform, axes, oriented, cosines = _anatomical_grid(grid)
    order = _axis_assignment(
        cosines,
        ambiguity_message="anatomical axis assignment is ambiguous; resample to a cardinal grid first",
    )
    assigned = tuple(oriented[i] for i in order)
    system = grid.frame.coordinate_system
    tokens = []
    for j, i in enumerate(order):
        token = system.orientation[oriented[i]]
        assert token is not None
        tokens.append(token if cosines[j, i] >= 0 else VOCABULARY.opposite(token))
    return transform, axes, assigned, tuple(tokens)


def orientation_codes(grid: Grid) -> str | tuple[str, ...]:
    """Return index-increase directions in ``grid.dims`` order, ignoring retained scalars.

    Return patient letters when every direction has a letter, otherwise RFC-4 tokens.
    Assignment maximizes summed absolute unit-step cosines one-to-one; score differences
    at most ``ASSIGNMENT_TOLERANCE`` (1e-6) refuse as an exact tie, not an obliquity bound.
    These are nearest-axis labels in the manner of nibabel's ``aff2axcodes``: an oblique
    rotation's labels flip across 45 degrees. Callers needing an angular bound must read
    the residual angle from ``CoordinateSystem.axis_codes``. Descending coordinates
    reverse the column's direction; a singleton uses its affine column alone.

    Raises:
        TypeError: If grid is not a Grid or its transform is not affine.
        ValueError: If the frame lacks three anatomical spatial axes with a common unit,
            there are more than three varying dims, a dim is empty, nonmonotonic or has
            no anatomical displacement, anatomical directions are parallel or antiparallel,
            or the assignment is ambiguous.
    """
    return _codes(_assignment(grid)[3])


def reoriented(grid: Grid, orientation: str | tuple[str, ...]) -> Grid:
    """Permute and reverse samples to the requested index-increase directions.

    Accept patient letters, RFC-4 token tuples, or DICOM display planes in
    (slice, row, column) order: axial/transverse=SPL, coronal=PIL, sagittal=RIP.
    Corresponding points and cells stay unchanged. Ordinary reversals use ``isel``;
    singleton reversals negate the affine column and coordinate, mirror any interval,
    and complement the sample offset only when an interval is declared. Without an
    interval a singleton has no cells, so its offset stays unchanged and two reflections
    restore the grid exactly. With an interval, two reflections restore points,
    coordinates and intervals exactly; offset subtraction can round within the absolute
    bound ``np.spacing(1.0)``. This is the only transform change: the source affine matrix
    is rebuilt as an ``AffineTransform``, replacing any other ``SupportsAffine`` implementation.

    Raises:
        TypeError: As on orientation_codes, or if orientation has the wrong type.
        ValueError: As on orientation_codes, or if directions are invalid, repeat axes,
            have the wrong count, or are not a signed permutation of the available axes.
        ImportError: If a nonsingleton reversal needs xarray and it is unavailable.
    """
    transform, axes, _, available = _assignment(grid)
    requested = _orientation(orientation, len(axes))
    pairs = [VOCABULARY.pair(token) for token in available]
    if {VOCABULARY.pair(token) for token in requested} != set(pairs):
        raise ValueError(
            f"requested axes are unavailable; available orientation is {_codes(available)!r}"
        )
    order = [pairs.index(VOCABULARY.pair(token)) for token in requested]
    reversed_dims = {}
    singleton_axes = []
    for token, j in zip(requested, order, strict=True):
        if token != available[j]:
            if axes[j].values.size == 1:
                singleton_axes.append(axes[j].axis)
            else:
                reversed_dims[grid.dims[j]] = slice(None, None, -1)
    result = grid.isel(**reversed_dims) if reversed_dims else grid
    if singleton_axes:
        source = transform.source
        assert isinstance(source, ArrayCoordinates)
        matrix = transform.matrix.copy()
        offsets = list(source.sample_offset)
        coordinates = dict(result.coordinates)
        intervals = dict(result.intervals)
        for name in singleton_axes:
            j = source.axes.index(name)
            matrix[:, j] *= -1
            entry = coordinates[name]
            assert isinstance(entry, tuple)
            dim, values = entry
            # Object integers avoid overflow when reflecting int64's minimum value.
            values = np.asarray(values)
            mirrored = -values.astype(object) if values.dtype.kind == "i" else -values
            if values.dtype.kind == "i" and int(mirrored[0]) > np.iinfo(np.int64).max:
                mirrored = -values.astype(np.float64)
            coordinates[name] = (dim, mirrored.tolist())
            if name in intervals:
                intervals[name] = -intervals[name][..., ::-1]
                offset = offsets[j]
                assert offset is not None
                offsets[j] = 1.0 - offset
        reflected = AffineTransform(
            source=ArrayCoordinates(
                source.axes, source.units, axis_types=source.axis_types, sample_offset=offsets
            ),
            target=grid.frame,
            matrix=matrix,
            translation=transform.translation,
        )
        result = Grid(reflected, coordinates, intervals=intervals)
    return result.transpose(*(grid.dims[j] for j in order))


def _source_corners(
    grid: Grid, axes: tuple[AxisSampling, ...], cover: str
) -> npt.NDArray[np.float64]:
    source = grid.transform.source
    assert isinstance(source, ArrayCoordinates)
    bounds = {}
    for axis in axes:
        if cover == "samples":
            bounds[axis.axis] = (float(axis.values.min()), float(axis.values.max()))
        elif axis.intervals is not None:
            bounds[axis.axis] = (
                float(axis.intervals[:, 0].min()),
                float(axis.intervals[:, 1].max()),
            )
        else:
            offset = source.sample_offset[source.axes.index(axis.axis)]
            if offset is None:
                raise ValueError(f"dimension {axis.dim!r} has no cells; use cover='samples'")
            before, after = cell_extent(axis, offset)
            edges = position_to_coordinate(
                axis.values, np.array([-before, axis.values.size - 1 + after])
            )
            bounds[axis.axis] = (float(edges.min()), float(edges.max()))
    corners = np.array(list(product(*(bounds[name] for name in source.axes))))
    return corners


def cardinal_grid(
    grid: Grid,
    orientation: str | tuple[str, ...],
    *,
    spacing: npt.ArrayLike | None = None,
    dims: Sequence[str] | None = None,
    cover: str = "cells",
) -> Grid:
    """Build a cardinal target grid in the source frame, covering cells or sample hull.

    Orientation uses the notation accepted by reoriented. Three varying dims and no
    retained axes are required. Output dims inherit names from source dims assigned to
    the same frame axes unless overridden. Spacing is a positive scalar or three values
    in output order; omitted, use assigned source step lengths (uniform within
    ``LATTICE_TOLERANCE``), or a singleton's declared interval width in frame units.
    An ambiguous assignment requires both explicit spacing and dims. Output directions
    and coverage depend only on the frame and source corners, not on that assignment.

    Exact source corners are projected onto output axes. Counts round upward after
    subtracting ``COVERAGE_TOLERANCE`` (1e-9 output steps); spacing is never shrunk.
    Cells start at the projected minimum edge; samples start at the minimum sample.
    Coverage overshoots only at the far side, by less than one step. In cells mode,
    a singleton with positive covered extent declares that exact interval and is centred
    on it. Samples mode is point support and declares no intervals.
    The affine maps int64 indices from dimensionless ArrayCoordinates.
    Non-anatomical cross-row variation must be roundoff only: within ``CROSS_ROW_ROUNDOFF``
    times the largest matched contribution (each column's largest absolute coefficient
    times its covered source coordinate span), independent of source coordinate scaling.
    The output takes the midpoint of any tolerated variation on such an axis.

    Raises:
        TypeError: As on orientation_codes, or for invalid orientation, spacing or dims types.
        ValueError: As on orientation_codes, or if the requested directions are unavailable,
            there are not three varying dims without retained axes, cover or dims are invalid,
            spacing is not finite and positive, default spacing is undetermined, cells have no
            support, anatomical directions lack rank three, or the source varies along a
            non-anatomical frame axis. Assignment ambiguity refuses only when spacing or
            dims is omitted.
    """
    transform, axes, oriented, cosines = _anatomical_grid(grid)
    if len(axes) != 3 or len(transform.source.axes) != 3:
        raise ValueError("cardinal_grid requires three varying dims and no retained scalar axes")
    if np.linalg.matrix_rank(cosines) < 3:
        raise ValueError("cardinal_grid requires anatomical step directions of rank three")
    requested = _orientation(orientation, 3)
    system = grid.frame.coordinate_system
    frame_pairs = {}
    for i in oriented:
        token = system.orientation[i]
        assert token is not None
        frame_pairs[VOCABULARY.pair(token)] = i
    if any(VOCABULARY.pair(token) not in frame_pairs for token in requested):
        raise ValueError(
            f"requested axes are unavailable; available frame directions are {system.orientation!r}"
        )
    output_axes = [frame_pairs[VOCABULARY.pair(token)] for token in requested]
    source_order = []
    if spacing is None or dims is None:
        order = _axis_assignment(
            cosines,
            ambiguity_message="anatomical axis assignment is ambiguous; pass explicit spacing and dims",
        )
        assigned = tuple(oriented[i] for i in order)
        source_order = [assigned.index(i) for i in output_axes]
    names = (
        tuple(grid.dims[j] for j in source_order)
        if dims is None
        else check_names(dims, field="dims")
    )
    if len(names) != 3:
        raise ValueError("dims must give three unique names")
    if cover not in ("cells", "samples"):
        raise ValueError("cover must be 'cells' or 'samples'")
    if spacing is None:
        lengths = []
        for j in source_order:
            axis = axes[j]
            step = uniform_step(axis.values, LATTICE_TOLERANCE)
            if axis.values.size == 1 and axis.intervals is not None:
                step = float(axis.intervals[0, 1] - axis.intervals[0, 0])
            if step is None:
                raise ValueError(
                    f"dimension {axis.dim!r} has nonuniform or undetermined spacing; supply explicit spacing"
                )
            column = transform.source.axes.index(axis.axis)
            lengths.append(
                abs(step) * float(np.linalg.norm(transform.matrix[list(oriented), column]))
            )
        steps = np.asarray(lengths)
    else:
        steps = real_float_array(spacing, field="spacing")
        if steps.ndim == 0:
            steps = np.full(3, float(steps))
        if steps.shape != (3,):
            raise ValueError("spacing must be a positive scalar or three values in output order")
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("spacing must be finite and positive")
    direction = np.zeros((len(grid.frame.axes), 3))
    for j, i in enumerate(output_axes):
        direction[i, j] = 1.0 if requested[j] == system.orientation[i] else -1.0
    source_corners = _source_corners(grid, axes, cover)
    corners = transform.transform_point(source_corners)
    coordinate_extent = np.ptp(source_corners, axis=0)
    roundoff = CROSS_ROW_ROUNDOFF * float(
        np.max(np.max(np.abs(transform.matrix), axis=0) * coordinate_extent)
    )
    unoriented = set(range(len(grid.frame.axes))) - set(oriented)
    for i in unoriented:
        cross_extent = float(np.abs(transform.matrix[i]) @ coordinate_extent)
        if cross_extent > roundoff:
            raise ValueError(
                "source varies along a non-anatomical frame axis; a cardinal grid cannot cover it"
            )
    projected = corners @ direction
    low, high = projected.min(axis=0), projected.max(axis=0)
    extent = high - low
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        counts = np.maximum(1, np.ceil(extent / steps - COVERAGE_TOLERANCE) + (cover == "samples"))
    if not np.all(np.isfinite(counts)) or np.any(counts >= np.iinfo(np.int64).max):
        raise ValueError("covered extent and spacing require too many int64 samples")
    sizes = counts.astype(np.int64)
    first = low + (0.5 * steps if cover == "cells" else 0)
    intervals = {}
    for j, name in enumerate(names):
        if cover == "cells" and sizes[j] == 1 and extent[j] > 0:
            first[j] = low[j] + extent[j] / 2
            width = extent[j] / steps[j]
            intervals[name] = np.array([[-width / 2, width / 2]])
    origin = direction @ first
    for i in unoriented:
        origin[i] = (corners[:, i].min() + corners[:, i].max()) / 2
    return Grid(
        AffineTransform(
            source=ArrayCoordinates(
                names, ("1",) * 3, sample_offset=(0.5,) * 3 if cover == "cells" else (None,) * 3
            ),
            target=grid.frame,
            matrix=direction * steps,
            translation=origin,
        ),
        {
            name: (name, np.arange(size, dtype=np.int64))
            for name, size in zip(names, sizes, strict=True)
        },
        intervals=intervals,
    )


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
