"""The transform protocol, its capability protocols, and evaluation by axis name.

A transform maps points from a source endpoint to a target endpoint. Each endpoint is a
:class:`~xarrayrf.ReferenceFrame` or :class:`~xarrayrf.ArrayCoordinates`; its axes and units
are the axes and units points are mapped from or to. What else a transform can do is expressed by
separate capability protocols rather than flags, so a representation cannot claim a capability
its methods contradict. User-defined transforms satisfy these protocols structurally; nothing is
registered or discovered.

``isinstance`` against these protocols is a structural check only: it confirms that the
attributes exist, not their signatures, return types or values. Code at a trust boundary
validates what a transform declares with :func:`check_transform` and validates its results.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ._array_coordinates import ArrayCoordinates
from ._frame import ReferenceFrame
from ._validation import real_float_array

type Endpoint = ReferenceFrame | ArrayCoordinates
"""The source or target of a transform."""


@runtime_checkable
class Transform(Protocol):
    """Declared endpoints shared by every transform.

    A transform also implements point mapping and any of the other capabilities protocol. Points are mapped from
    ``source.axes`` in ``source.units`` to ``target.axes`` in ``target.units``.

    When retained by a Grid or binding, endpoints, coefficients/behavior and equality must
    remain stable. Equality returns a scalar boolean and reflects the chosen declaration
    identity; identity equality is valid. Hashing is optional for binding and queries. A
    hashable transform must have a stable hash consistent with equality; hashing a Grid
    containing an unhashable transform raises TypeError. Composite stability and hashability
    depend on its members. These are caller contracts, not runtime immutability checks.
    """

    @property
    def source(self) -> Endpoint:
        """The endpoint points are mapped from."""
        ...

    @property
    def target(self) -> Endpoint:
        """The endpoint points are mapped to."""
        ...


@runtime_checkable
class SupportsPoints(Transform, Protocol):
    """A transform that maps points; translation applies."""

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map points whose trailing axis is ordered as the source axes.

        Returns an array of the same leading shape whose trailing axis is ordered as the
        target axes. Raises outside the transform's valid domain rather than extrapolating.
        """
        ...


@runtime_checkable
class SupportsJacobian(Transform, Protocol):
    """A transform with a local linear part (its Jacobian)."""

    def jacobian(self, at: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        """Return the Jacobian, target axes by source axes.

        ``at`` holds points shaped like the argument of :meth:`SupportsPoints.transform_point` and is
        required unless the transform is :class:`SupportsAffine`.
        """
        ...


@runtime_checkable
class SupportsAffine(SupportsPoints, SupportsJacobian, Protocol):
    """A transform ``x -> matrix @ x + translation`` with constant matrix and translation."""

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """The constant linear part, target axes by source axes."""
        ...

    @property
    def translation(self) -> npt.NDArray[np.float64]:
        """The constant offset, one value per target axis."""
        ...


@runtime_checkable
class SupportsInverse(Transform, Protocol):
    """A transform with an exact inverse on its declared domain."""

    def inverse(self) -> Transform:
        """Return the inverse, with source and target exchanged."""
        ...


def check_endpoint(value: object, *, field: str) -> Endpoint:
    """Require a :class:`~xarrayrf.ReferenceFrame` or :class:`~xarrayrf.ArrayCoordinates`."""
    if not isinstance(value, ReferenceFrame | ArrayCoordinates):
        raise TypeError(
            f"{field} must be a ReferenceFrame or ArrayCoordinates, got {type(value).__name__}",
        )
    return value


def check_transform(transform: object) -> SupportsPoints:
    """Validate what a point transform declares, before using it.

    Args:
        transform: Candidate transform, possibly user-defined.

    Returns:
        The same object, typed as a point transform.

    Raises:
        TypeError: If it does not implement :class:`SupportsPoints`, or an endpoint is not a
            :class:`~xarrayrf.ReferenceFrame` or :class:`~xarrayrf.ArrayCoordinates`.
    """
    if not isinstance(transform, SupportsPoints):
        raise TypeError(
            f"transform must implement SupportsPoints, got {type(transform).__name__}",
        )
    name = type(transform).__name__
    check_endpoint(transform.source, field=f"{name}.source")
    check_endpoint(transform.target, field=f"{name}.target")
    return transform


def _is_builtin_transform(transform: SupportsPoints) -> bool:
    """Whether ``transform`` is exactly AffineTransform or CompositeTransform (not a subclass)."""
    from ._affine import AffineTransform
    from ._composite import CompositeTransform

    return type(transform) in (AffineTransform, CompositeTransform)


def transform_named(
    transform: SupportsPoints,
    coordinates: Mapping[str, npt.ArrayLike],
) -> dict[str, npt.NDArray[np.float64]]:
    """Evaluate a point transform on coordinates given by source axis name.

    Names are matched against ``transform.source.axes``, so column order can never be wrong.
    Values broadcast against each other: a selected plane passes a scalar for its fixed
    coordinate alongside arrays for the varying ones.

    Args:
        transform: The transform to evaluate.
        coordinates: One finite real scalar or array per source axis. A mixed Python sequence
            follows ordinary NumPy promotion, so its inferred dtype is what must be real
            integer or floating; a masked array is refused rather than unmasked.

    Returns:
        One array per target axis, keyed by axis name, each of the common broadcast shape.
        Scalar coordinates produce zero-dimensional arrays.

    Raises:
        TypeError: If ``transform`` fails :func:`check_transform`, ``coordinates`` is not a
            mapping, a value's inferred dtype is not real integer or floating, a value is a
            masked array, or the transform returns a non-real result.
        ValueError: If a source axis is missing or an unexpected name is given, a value is not
            a rectangular numeric array, a value is not finite, the values do not broadcast,
            or the transform returns a result of the wrong shape or a non-finite result.
    """
    check_transform(transform)
    if not isinstance(coordinates, Mapping):
        raise TypeError(
            "coordinates must be a mapping of source axis name to values, got "
            f"{type(coordinates).__name__}",
        )
    axes = transform.source.axes
    missing = tuple(axis for axis in axes if axis not in coordinates)
    if missing:
        raise ValueError(
            f"missing values for source axes {missing}; every source axis {axes} must be supplied",
        )
    unexpected = tuple(name for name in coordinates if name not in axes)
    if unexpected:
        raise ValueError(
            f"unexpected coordinates {unexpected}; this transform maps only {axes}",
        )
    values = [real_float_array(coordinates[axis], field=f"coordinate {axis!r}") for axis in axes]
    try:
        broadcast = np.broadcast_arrays(*values)
    except ValueError as error:
        shapes = tuple(value.shape for value in values)
        raise ValueError(
            f"coordinate values for {axes} do not broadcast; got shapes {shapes}",
        ) from error
    shape = broadcast[0].shape
    name = type(transform).__name__
    mapped = transform.transform_point(np.stack(broadcast, axis=-1))
    # The exact built-in classes return finite float arrays by contract; a subclass or a
    # user-defined transform has its result validated here.
    result = (
        np.asarray(mapped, dtype=np.float64)
        if _is_builtin_transform(transform)
        else real_float_array(mapped, field=f"{name}.transform_point result")
    )
    target_axes = transform.target.axes
    if result.shape != (*shape, len(target_axes)):
        raise ValueError(
            f"{name}.transform_point returned shape {result.shape}; expected "
            f"{(*shape, len(target_axes))} for target axes {target_axes}",
        )
    # Copy each axis so the result never shares memory with a transform's own storage.
    return {axis: result[..., index].copy() for index, axis in enumerate(target_axes)}
