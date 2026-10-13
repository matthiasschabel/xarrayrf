"""Transforms between Cartesian frames whose displacement comes from a sampled field.

A registration maps one frame to another. Deformable registration tools write that map as a
displacement ``y = x + u(x)``, either sampled on a grid (ITK and ANTs displacement fields) or
parameterized by cubic B-spline coefficients on a control grid (ITK, elastix, NiftyReg). Both
classes here hold the producer's representation as given and evaluate it in float64.

The field's grid is a :class:`~xarrayrf.Grid` in the source frame. It is the field's declared
domain, not a constraint on what the transform applies to: points anywhere in the source frame
are mapped, and the ``outside`` declaration says what happens beyond the support. ITK's
conventions define the support: a dense field covers its cells, half-open at the upper index
edge (continuous index ``[-0.5, n - 0.5)`` for centred cells), and a cubic B-spline covers
control index ``[1, n - 2]``, closed. Edges are compared exactly, as ITK compares them.

Evaluation uses SciPy, loaded on first use as resampling loads it: values through
``ndimage.map_coordinates`` (linear, or cubic without prefiltering), dense Jacobians through the
multilinear derivative here, whose tie rule on cell faces is part of the contract, and B-spline
Jacobians through ``interpolate.NdBSpline``. Construction and inspection need only NumPy.

Neither class has an exact inverse, so neither implements ``inverse()``.
"""

from __future__ import annotations

import itertools
from typing import Any, Final, Literal, get_args

import numpy as np
import numpy.typing as npt

from ._affine import equilibrated_inverse
from ._array_coordinates import ArrayCoordinates
from ._coordinate_system import CARTESIAN
from ._frame import ReferenceFrame
from ._grid import Grid
from ._positions import transform_points
from ._sampling import (
    EXACT_STEP_TOLERANCE,
    LATTICE_TOLERANCE,
    AxisSampling,
    cell_extent,
    coordinate_to_position,
    singleton_step,
    uniform_step,
)
from ._transform import SupportsAffine, SupportsPoints, check_transform
from ._validation import REAL_KINDS, real_float_array

type Outside = Literal["raise", "identity", "nearest"]
"""What a field transform does beyond its support."""

Support = Literal["closed", "half-open"]
"""Whether a B-spline's upper support face is inside: ITK's ``"closed"`` or elastix's
``"half-open"``."""

OUTSIDE: Final[tuple[Outside, ...]] = get_args(Outside.__value__)

_BLOCK: Final = 2**16
"""Points evaluated at once, which bounds the gathered-coefficient memory (64 taps per point
for a 3-D B-spline)."""

_STORED_DTYPES: Final = (np.dtype(np.float32), np.dtype(np.float64))


def _check_outside(outside: object) -> Outside:
    if outside not in OUTSIDE:
        raise ValueError(f"outside must be one of {OUTSIDE}, got {outside!r}")
    return outside


def _check_endpoints(source: object, target: object, *, what: str) -> None:
    """Require Cartesian frames with equal coordinate systems: a displacement is added."""
    for name, endpoint in (("source", source), ("target", target)):
        if not isinstance(endpoint, ReferenceFrame):
            raise TypeError(
                f"{what} {name} must be a ReferenceFrame, got {type(endpoint).__name__}"
            )
        representation = endpoint.coordinate_system.representation
        if representation != CARTESIAN:
            raise ValueError(
                f"{what} {name} has a {representation} coordinate system; field transforms "
                "interpolate components and need Cartesian axes (an angular seam needs a "
                "seam-aware interpolant, which is not implemented)",
            )
    assert isinstance(source, ReferenceFrame) and isinstance(target, ReferenceFrame)
    if source.coordinate_system != target.coordinate_system:
        raise ValueError(
            f"{what} adds a displacement to the point, so source and target need equal coordinate "
            f"systems (axes, units and types); got {source.coordinate_system!r} and "
            f"{target.coordinate_system!r}",
        )


def _frozen_values(
    value: object, *, field: str, shape: tuple[int, ...]
) -> npt.NDArray[np.floating]:
    """Copy values into a bytes-backed buffer, keeping float32 or float64 as given."""
    if isinstance(value, np.ma.MaskedArray):
        raise TypeError(f"{field} must not be a masked array")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a rectangular numeric array") from error
    if array.dtype.kind not in REAL_KINDS:
        raise TypeError(
            f"{field} must hold real integer or floating values, got dtype {array.dtype}"
        )
    dtype = array.dtype if array.dtype in _STORED_DTYPES else np.dtype(np.float64)
    array = np.asarray(array, dtype=dtype)
    if array.shape != shape:
        raise ValueError(
            f"{field} must have shape {shape} (grid sizes, then components), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must contain only finite values")
    # Owning bytes, rather than a read-only flag, keeps the published views from being made
    # writeable again (see frozen_float_array).
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=dtype).reshape(shape)


def _tiles(sampling: AxisSampling) -> bool:
    """Whether declared intervals tile their axis, compared in ascending coordinate order."""
    rows = sampling.intervals
    assert rows is not None
    if rows.shape[0] < 2:
        return True
    ordered = rows[np.argsort(rows[:, 0])]
    widths = ordered[:, 1] - ordered[:, 0]
    mismatch = np.abs(ordered[1:, 0] - ordered[:-1, 1])
    return bool(np.all(mismatch <= LATTICE_TOLERANCE * np.minimum(widths[1:], widths[:-1])))


class _FieldGrid:
    """A field's grid prepared once: frame point -> continuous index, support and its derivative.

    Dimension ``k`` follows ``grid.dims[k]``, which is also axis ``k`` of the stored array.
    """

    __slots__ = (
        "affine_inverse",
        "axis_of",
        "lower",
        "shape",
        "steps",
        "translation",
        "upper",
        "values",
    )

    def __init__(self, grid: object, source: ReferenceFrame, *, spline: bool, what: str) -> None:
        if not isinstance(grid, Grid):
            raise TypeError(f"{what} grid must be a Grid, got {type(grid).__name__}")
        if grid.frame != source:
            raise ValueError(
                f"the {what} grid is located in {grid.frame.identifier}, but the transform's source "
                f"is {source.identifier}; a field is sampled in its source frame",
            )
        transform = grid.transform
        if not isinstance(transform, SupportsAffine):
            raise ValueError(
                f"the {what} grid's transform must be affine (use compose() to collapse a chain "
                f"of affines), got {type(transform).__name__}",
            )
        coordinates = transform.source
        assert isinstance(coordinates, ArrayCoordinates)
        sampling = grid._sampling()
        scalars = [axis.axis for axis in sampling.axes if axis.dim is None]
        if scalars:
            raise ValueError(
                f"the {what} grid retains scalar axes {scalars}; every source axis must vary "
                "along its own dimension",
            )
        matrix = np.asarray(transform.matrix, dtype=np.float64)
        if matrix.shape != (len(source.axes), len(source.axes)):
            raise ValueError(
                f"the {what} grid's transform must be square, one dimension per frame axis; its "
                f"matrix has shape {matrix.shape}",
            )
        self.affine_inverse = equilibrated_inverse(matrix)
        self.translation = np.asarray(transform.translation, dtype=np.float64)
        by_dim = {axis.dim: index for index, axis in enumerate(sampling.axes)}
        self.axis_of = np.array([by_dim[dim] for dim in grid.dims], dtype=np.intp)
        self.values: list[npt.NDArray[np.float64]] = []
        self.steps: list[float | None] = []
        lower, upper = [], []
        for dim, axis_index in zip(grid.dims, self.axis_of, strict=True):
            axis = sampling.axes[axis_index]
            values = axis.values
            size = values.size
            differences = np.diff(values)
            if size > 1 and not (np.all(differences > 0) or np.all(differences < 0)):
                raise ValueError(
                    f"the {what} grid's coordinate along {dim!r} is not strictly monotonic"
                )
            if spline:
                if size < 4:
                    raise ValueError(
                        f"a cubic B-spline needs at least four control points per axis; {dim!r} has {size}",
                    )
                step = uniform_step(values, EXACT_STEP_TOLERANCE)
                if step is None:
                    raise ValueError(
                        f"the B-spline control grid's coordinate along {dim!r} is not uniform to "
                        f"{EXACT_STEP_TOLERANCE:g} steps; a cubic B-spline needs one constant step",
                    )
                lower.append(1.0)
                upper.append(size - 2.0)
            else:
                offset = coordinates.sample_offset[axis_index]
                if offset is None and axis.intervals is None:
                    raise ValueError(
                        f"the displacement field's grid declares point samples along {dim!r}; a "
                        "dense field's support is its cells, so declare a sample_offset or intervals",
                    )
                if size == 1 and axis.intervals is None:
                    raise ValueError(
                        f"the displacement field's grid has a single sample along {dim!r}; declare "
                        "its interval, which gives the slab its thickness",
                    )
                if axis.intervals is not None and not _tiles(axis):
                    raise ValueError(
                        f"the displacement field's intervals along {dim!r} leave gaps or overlap; "
                        "a field's cells must tile its support",
                    )
                step = (
                    singleton_step(axis, "cells")
                    if size == 1
                    else uniform_step(values, EXACT_STEP_TOLERANCE)
                )
                before, after = cell_extent(axis, offset)
                lower.append(-before)
                upper.append(size - 1 + after)
            self.values.append(values)
            self.steps.append(step)
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.shape = tuple(grid.sizes[dim] for dim in grid.dims)

    def positions(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Continuous indices ``(N, D)`` of frame points ``(N, D)``, extended beyond the samples."""
        coordinates = (points - self.translation) @ self.affine_inverse.T
        return np.stack(
            [
                coordinate_to_position(values, coordinates[:, axis], step, extrapolate=True)
                for values, step, axis in zip(self.values, self.steps, self.axis_of, strict=True)
            ],
            axis=-1,
        )

    def position_jacobian(self, positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """``∂p/∂x`` shaped ``(N, D, D)``: per-dimension ``1 / local step`` times the affine inverse.

        The local step is that of the cell ``floor(p)`` clipped into the samples, the same cell
        the dense interpolant uses, so a sample face takes the upper cell's step.
        """
        rows = self.affine_inverse[self.axis_of]
        scales = np.empty_like(positions)
        for k, (values, step) in enumerate(zip(self.values, self.steps, strict=True)):
            if step is not None:
                scales[:, k] = 1.0 / step
            else:
                cell = np.clip(np.floor(positions[:, k]), 0, values.size - 2).astype(np.intp)
                scales[:, k] = 1.0 / (values[cell + 1] - values[cell])
        return scales[:, :, None] * rows[None, :, :]


def _multilinear_derivative(
    values: npt.NDArray[np.floating], positions: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """``∂u/∂p`` ``(N, C, D)`` of the multilinear interpolant at positions clamped to the samples.

    The cell is ``floor(p)`` clipped to ``[0, n - 2]``: an interior sample face takes the upper
    cell's derivative and ``n - 1`` the lower cell's. A single-sample axis has no derivative.
    """
    shape = values.shape[:-1]
    count, dims = positions.shape
    base = np.stack(
        [np.clip(np.floor(positions[:, k]), 0, max(shape[k] - 2, 0)) for k in range(dims)], axis=-1
    ).astype(np.intp)
    fraction = positions - base
    derivative = np.zeros((count, values.shape[-1], dims))
    corners = [(0, 1) if shape[k] > 1 else (0,) for k in range(dims)]
    for corner in itertools.product(*corners):
        taps = values[tuple(base[:, k] + corner[k] for k in range(dims))].astype(np.float64)
        factors = [fraction[:, k] if corner[k] else 1.0 - fraction[:, k] for k in range(dims)]
        for k in range(dims):
            if shape[k] == 1:
                continue
            others = (
                np.prod([factors[j] for j in range(dims) if j != k], axis=0)
                if dims > 1
                else np.ones(count)
            )
            sign = 1.0 if corner[k] else -1.0
            derivative[:, :, k] += (sign * others)[:, None] * taps
    return derivative


def _scipy_interpolation() -> Any:
    """SciPy's interpolation modules, loaded on first evaluation as resampling loads them."""
    try:
        from scipy import interpolate, ndimage
    except ImportError as error:
        raise ImportError(
            "evaluating a field transform needs SciPy; install xarrayrf[resample]"
        ) from error
    return ndimage, interpolate


def _sampled_values(
    stored: npt.NDArray[np.floating], positions: npt.NDArray[np.float64], *, order: int, mode: str
) -> npt.NDArray[np.float64]:
    """``(N, C)`` values of the order-``order`` B-spline of ``stored``, without prefiltering."""
    ndimage, _ = _scipy_interpolation()
    coordinates = positions.T
    result = np.empty((len(positions), stored.shape[-1]))
    for channel in range(stored.shape[-1]):
        # float64 output keeps the arithmetic in double for float32 storage.
        result[:, channel] = ndimage.map_coordinates(
            stored[..., channel],
            coordinates,
            order=order,
            prefilter=False,
            mode=mode,
            output=np.float64,
        )
    return result


class _FieldTransform:
    """Shared evaluation: locate, classify against the support, apply the exterior, interpolate."""

    __slots__ = (
        "_derivative_cache",
        "_grid",
        "_lookup",
        "_outside",
        "_source",
        "_stored",
        "_target",
        "_upper_closed",
    )

    _upper_closed: bool
    _clamp_to_support: bool
    _what: str

    def _setup(
        self,
        source: object,
        target: object,
        values: object,
        grid: object,
        outside: object,
        *,
        field: str,
        upper_closed: bool,
    ) -> None:
        _check_endpoints(source, target, what=self._what)
        self._upper_closed = upper_closed
        self._derivative_cache: Any = None
        assert isinstance(source, ReferenceFrame) and isinstance(target, ReferenceFrame)
        self._source = source
        self._target = target
        self._outside = _check_outside(outside)
        self._lookup = _FieldGrid(grid, source, spline=self._clamp_to_support, what=self._what)
        assert isinstance(grid, Grid)
        self._grid = grid
        self._stored = _frozen_values(
            values, field=field, shape=(*self._lookup.shape, len(target.axes))
        )

    @property
    def source(self) -> ReferenceFrame:
        """The frame points are mapped from; the grid is located in it."""
        return self._source

    @property
    def target(self) -> ReferenceFrame:
        """The frame points are mapped to."""
        return self._target

    @property
    def grid(self) -> Grid:
        """The sampling or control grid, in the source frame."""
        return self._grid

    @property
    def outside(self) -> Outside:
        """``"raise"``, ``"identity"`` or ``"nearest"`` beyond the support."""
        return self._outside

    def _points(self, values: npt.ArrayLike, *, field: str) -> npt.NDArray[np.float64]:
        points = real_float_array(values, field=field)
        width = len(self._source.axes)
        if points.ndim == 0 or points.shape[-1] != width:
            raise ValueError(
                f"{field} must have a trailing axis of length {width} ordered as "
                f"{self._source.axes}, got shape {points.shape}",
            )
        return points

    def _evaluate(
        self, points: npt.NDArray[np.float64], *, jacobian: bool
    ) -> npt.NDArray[np.float64]:
        """Mapped points, or Jacobians, for points ``(..., D)``."""
        lookup = self._lookup
        flat = points.reshape(-1, points.shape[-1])
        dims = flat.shape[-1]
        if jacobian:
            result = np.broadcast_to(np.eye(dims), (flat.shape[0], dims, dims)).copy()
        else:
            result = flat.copy()
        for start in range(0, flat.shape[0], _BLOCK):
            block = flat[start : start + _BLOCK]
            positions = lookup.positions(block)
            above = (
                positions >= lookup.upper if not self._upper_closed else positions > lookup.upper
            )
            outside_rows = np.any((positions < lookup.lower) | above, axis=-1)
            if outside_rows.any() and self._outside == "raise":
                first = start + int(np.argmax(outside_rows))
                where = np.unravel_index(first, points.shape[:-1]) if points.ndim > 1 else ()
                row = first - start
                edge = "]" if self._upper_closed else ")"
                support = ", ".join(
                    f"{dim}: [{lo:g}, {hi:g}{edge}"
                    for dim, lo, hi in zip(self._grid.dims, lookup.lower, lookup.upper, strict=True)
                )
                raise ValueError(
                    f"point {tuple(int(i) for i in where)} lies outside the {self._what}'s support "
                    f"at continuous index {tuple(round(float(p), 6) for p in positions[row])}; the "
                    f"support is {support}. Declare outside='identity' or 'nearest' to map such "
                    "points",
                )
            rows = ~outside_rows if self._outside == "identity" else np.ones(len(block), dtype=bool)
            if not rows.any():
                continue
            wanted = positions[rows]
            if self._clamp_to_support:
                clamped = np.clip(wanted, lookup.lower, lookup.upper)
            else:
                clamped = np.clip(wanted, 0.0, np.array(lookup.shape, dtype=np.float64) - 1)
            target = np.flatnonzero(rows) + start
            if not jacobian:
                result[target] += self._values(clamped)
                continue
            derivative = self._derivative(clamped)
            # The constant extension beyond the samples, or beyond the support under "nearest",
            # has no derivative along the clamped axis.
            derivative[np.broadcast_to((clamped != wanted)[:, None, :], derivative.shape)] = 0.0
            result[target] += derivative @ lookup.position_jacobian(clamped)
        shape = (*points.shape[:-1], dims, dims) if jacobian else points.shape
        return result.reshape(shape)

    def _values(self, positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def _derivative(self, positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map points ``(..., D)`` ordered as the source axes to ``x + u(x)`` in the target.

        Raises:
            TypeError: If the values are not real.
            ValueError: If they are not finite or have the wrong trailing length, or a point lies
                outside the support while ``outside="raise"``.
        """
        return self._evaluate(self._points(points, field="points"), jacobian=False)

    def jacobian(self, at: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        """Return ``I + ∂u/∂x`` at points ``(..., D)``, shaped ``(..., D, D)``.

        The derivative is that of the interpolant, chained with the world-to-index map. It is the
        identity at an ``"identity"`` exterior point, and has no component along an axis that is
        clamped (beyond a dense field's outer samples, or by ``"nearest"``).

        Raises:
            ValueError: If ``at`` is omitted (a field is not affine), or as :meth:`transform_point`.
        """
        if at is None:
            raise ValueError(
                f"at is required: a {self._what} is not affine, so its Jacobian varies"
            )
        return self._evaluate(self._points(at, field="at"), jacobian=True)

    def __repr__(self) -> str:
        """Endpoints, grid sizes, dtype and exterior; never the array."""
        sizes = dict(self._grid.sizes)
        return (
            f"{type(self).__name__}(source={self._source.identifier!r}, "
            f"target={self._target.identifier!r}, grid={sizes}, dtype={self._stored.dtype}, "
            f"outside={self._outside!r})"
        )


class DisplacementFieldTransform(_FieldTransform):
    """``y = x + u(x)`` with ``u`` sampled on a grid and interpolated linearly.

    This is the representation of ITK's and ANTs's displacement fields and NGFF's
    ``displacements``. The support is the union of the grid's cells, half-open at the upper index
    edge as in ITK: ``[-0.5, n - 0.5)`` in continuous index for centred cells. Inside it, beyond
    the outer samples, the edge samples extend as constants. Equality is identity.
    """

    __slots__ = ()

    _clamp_to_support = False
    _what = "displacement field"

    def __init__(
        self,
        *,
        source: ReferenceFrame,
        target: ReferenceFrame,
        field: npt.ArrayLike,
        grid: Grid,
        outside: Outside = "raise",
    ) -> None:
        """Validate the endpoints and grid, and copy the field into immutable storage.

        Args:
            source: The frame points are mapped from; the grid's frame.
            target: The frame points are mapped to, with the source's coordinate system.
            field: Displacements shaped ``(*grid sizes in grid.dims order, D)``, components
                ordered as the target axes; float32 and float64 are kept, integers are stored as
                float64.
            grid: The sampling grid: an affine, square, invertible transform from array
                coordinates to ``source``, with strictly monotonic coordinates and cells declared
                on every axis (a ``sample_offset`` or tiling intervals; a single-sample axis
                needs an interval).
            outside: ``"raise"``, ``"identity"`` (ITK's semantics) or ``"nearest"`` (the
                nearest edge displacement).

        Raises:
            TypeError: If an endpoint is not a ReferenceFrame, ``grid`` is not a Grid, or the
                field is masked or not real.
            ValueError: If the endpoints are not Cartesian with equal coordinate systems, the
                grid is not admissible, the field has the wrong shape or is not finite, or
                ``outside`` is unknown.
        """
        self._setup(source, target, field, grid, outside, field="field", upper_closed=False)

    def _values(self, positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Positions are clamped to the samples, so "nearest" padding is never reached.
        return _sampled_values(self._stored, positions, order=1, mode="nearest")

    def _derivative(self, positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return _multilinear_derivative(self._stored, positions)

    @property
    def field(self) -> npt.NDArray[np.floating]:
        """The stored displacements, a read-only view."""
        return self._stored.view()

    def __reduce__(self) -> tuple[object, ...]:
        """Rebuild through the constructor; the copy is a new, unequal transform."""
        return (
            _rebuild,
            (
                DisplacementFieldTransform,
                self._source,
                self._target,
                "field",
                np.array(self._stored),
                self._grid,
                self._outside,
            ),
        )


class BSplineTransform(_FieldTransform):
    """``y = x + Σ c_k β(x - k)``: a cubic B-spline displacement from control-point coefficients.

    The representation of ITK's ``BSplineTransform``, elastix and NiftyReg. Coefficients are not
    samples of the displacement and are evaluated without prefiltering. The support is control
    index ``[1, n - 2]``, closed at both ends as in ITK, or ``[1, n - 2)`` as in elastix when
    declared ``support="half-open"``. Equality is identity.
    """

    __slots__ = ()

    _clamp_to_support = True
    _what = "B-spline transform"

    def __init__(
        self,
        *,
        source: ReferenceFrame,
        target: ReferenceFrame,
        coefficients: npt.ArrayLike,
        grid: Grid,
        order: int = 3,
        outside: Outside = "raise",
        support: Support = "closed",
    ) -> None:
        """Validate the endpoints and control grid, and copy the coefficients.

        Args:
            source: The frame points are mapped from; the control grid's frame.
            target: The frame points are mapped to, with the source's coordinate system.
            coefficients: Shaped ``(*grid sizes in grid.dims order, D)``, components ordered as
                the target axes; float32 and float64 are kept.
            grid: The control grid: an affine, square, invertible transform from array
                coordinates to ``source``, with at least four uniformly spaced coordinates per
                axis (to 1e-12 steps).
            order: The spline order; only 3 is implemented.
            outside: ``"raise"``, ``"identity"`` (ITK's semantics) or ``"nearest"`` (evaluated
                at the nearest point of the support).
            support: The upper support edge: ``"closed"`` (``[1, n - 2]``, ITK's) or
                ``"half-open"`` (``[1, n - 2)``, elastix's). Only points on the upper face are
                classified differently.

        Raises:
            NotImplementedError: If ``order`` is not 3.
            TypeError: As :class:`DisplacementFieldTransform`.
            ValueError: As :class:`DisplacementFieldTransform`, if an axis has fewer than four
                control points or is not uniform, or ``support`` is unknown.
        """
        if isinstance(order, bool) or not isinstance(order, int | np.integer):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order != 3:
            raise NotImplementedError(
                f"only cubic B-splines (order 3) are implemented, got order {order}"
            )
        if support not in ("closed", "half-open"):
            raise ValueError(f"support must be 'closed' or 'half-open', got {support!r}")
        self._setup(
            source,
            target,
            coefficients,
            grid,
            outside,
            field="coefficients",
            upper_closed=support == "closed",
        )

    def _values(self, positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Taps floor(p) - 1 .. floor(p) + 2 stay inside the grid on [1, n - 2], except the
        # fourth at p = n - 2, whose weight is exactly zero, so the padding never contributes.
        return _sampled_values(self._stored, positions, order=3, mode="grid-constant")

    def _derivative(self, positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """``∂u/∂p`` through NdBSpline with cardinal knots, built once on first use."""
        if self._derivative_cache is None:
            _, interpolate = _scipy_interpolation()
            knots = tuple(np.arange(n + 4, dtype=np.float64) - 2.0 for n in self._lookup.shape)
            self._derivative_cache = interpolate.NdBSpline(knots, np.asarray(self._stored), 3)
        spline = self._derivative_cache
        dims = positions.shape[-1]
        return np.stack(
            [
                np.asarray(spline(positions, nu=tuple(int(a == b) for b in range(dims))))
                for a in range(dims)
            ],
            axis=-1,
        )

    @property
    def coefficients(self) -> npt.NDArray[np.floating]:
        """The stored control-point coefficients, a read-only view."""
        return self._stored.view()

    @property
    def order(self) -> int:
        """The spline order, 3."""
        return 3

    @property
    def support(self) -> Support:
        """``"closed"`` (ITK) or ``"half-open"`` (elastix): whether the upper face is inside."""
        return "closed" if self._upper_closed else "half-open"

    def __repr__(self) -> str:
        """As the base, naming a non-default support."""
        text = super().__repr__()
        return text if self._upper_closed else f"{text[:-1]}, support='half-open')"

    def __reduce__(self) -> tuple[object, ...]:
        """Rebuild through the constructor; the copy is a new, unequal transform."""
        return (
            _rebuild,
            (
                BSplineTransform,
                self._source,
                self._target,
                "coefficients",
                np.array(self._stored),
                self._grid,
                self._outside,
                {"support": self.support},
            ),
        )


def _rebuild(
    cls: type[DisplacementFieldTransform] | type[BSplineTransform],
    source: ReferenceFrame,
    target: ReferenceFrame,
    name: str,
    values: npt.NDArray[np.floating],
    grid: Grid,
    outside: Outside,
    options: dict[str, object] | None = None,
) -> DisplacementFieldTransform | BSplineTransform:
    keywords: dict[str, Any] = {name: values, **(options or {})}
    return cls(source=source, target=target, grid=grid, outside=outside, **keywords)


def sample_displacement_field(
    transform: SupportsPoints, grid: Grid, *, outside: Outside = "raise"
) -> DisplacementFieldTransform:
    """Sample any transform's displacement at a grid's points, as a dense field.

    This is the explicit conversion of a B-spline (or an affine, or a chain) into the dense
    representation: ``u = T(x) - x`` at each sample, interpolated linearly afterwards, so the
    result agrees with ``transform`` at the samples only.

    Args:
        transform: A point transform between Cartesian frames with equal coordinate systems.
        grid: Where to sample, in ``transform.source``; it must satisfy the
            :class:`DisplacementFieldTransform` grid rules, including declared cells.
        outside: The returned field's exterior behaviour.

    Returns:
        A float64 :class:`DisplacementFieldTransform` with the transform's endpoints.

    Raises:
        TypeError: If ``transform`` is not a point transform or ``grid`` is not a Grid.
        ValueError: As :class:`DisplacementFieldTransform`, or if ``transform`` refuses a sample.
    """
    check_transform(transform)
    _check_endpoints(transform.source, transform.target, what="sample_displacement_field")
    assert isinstance(transform.source, ReferenceFrame) and isinstance(
        transform.target, ReferenceFrame
    )
    if not isinstance(grid, Grid):
        raise TypeError(f"grid must be a Grid, got {type(grid).__name__}")
    if grid.frame != transform.source:
        raise ValueError(
            f"the grid is located in {grid.frame.identifier}, but the transform maps from "
            f"{transform.source.identifier}; sample in the transform's source frame",
        )
    # The grid's admissibility is checked before the transform is evaluated on it.
    _FieldGrid(grid, transform.source, spline=False, what="displacement field")
    points = grid.points()
    displacement = transform_points(transform, points) - points
    return DisplacementFieldTransform(
        source=transform.source,
        target=transform.target,
        field=displacement,
        grid=grid,
        outside=outside,
    )
