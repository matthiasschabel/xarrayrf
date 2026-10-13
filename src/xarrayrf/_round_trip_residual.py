"""How far a transform and its inverse are from undoing each other on chosen points.

A declared inverse (an ANTs ``1InverseWarp``, a chain's ``declared_inverse()``) is the
producer's assertion, not a guarantee; this report is how it is checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._composite import exact_or_declared_inverse, has_exact_or_declared_inverse
from ._coordinate_system import CARTESIAN
from ._frame import ReferenceFrame
from ._grid import Grid
from ._transform import SupportsPoints, check_transform
from ._validation import frozen_float_array, real_float_array


def _checked(
    transform: SupportsPoints, points: npt.NDArray[np.float64], *, field: str
) -> npt.NDArray[np.float64]:
    """Map ``points`` and require one finite real point per input point.

    A user transform's result is otherwise unchecked, and a wrongly shaped one would broadcast
    into a plausible residual.
    """
    result = real_float_array(transform.transform_point(points), field=field)
    expected = (*points.shape[:-1], len(transform.target.axes))
    if result.shape != expected:
        raise ValueError(f"{field} has shape {result.shape}; expected {expected}")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class RoundTripResidualReport:
    """How far ``inverse(transform(x))`` lands from ``x``.

    Attributes:
        residual: Read-only float64 ``inverse(transform(x)) - x``, shaped like the points, or
            like a grid's sizes in ``grid.dims`` order followed by the axis.
        distance: Read-only Euclidean length of each residual, in ``unit``.
        maximum: The largest distance.
        root_mean_square: The root mean square of the distances.
        unit: The source frame's unit, shared by all its axes (``None`` if undeclared).
    """

    residual: npt.NDArray[np.float64]
    distance: npt.NDArray[np.float64]
    maximum: float
    root_mean_square: float
    unit: str | None


def round_trip_residual_report(
    transform: SupportsPoints, at: npt.ArrayLike | Grid
) -> RoundTripResidualReport:
    """Report ``inverse(transform(x)) - x`` at points or at a grid's samples.

    The inverse is chosen as :meth:`CompositeTransform.declared_inverse` chooses a member's: a
    chain's ``declared_inverse()``, else an exact ``inverse()``, else a declared one. Pass the
    other direction (``pair.backward``) to check the opposite round trip.

    Args:
        transform: A point transform with an exact or declared inverse, whose source is a
            Cartesian :class:`~xarrayrf.ReferenceFrame` with one unit on every axis.
        at: Points shaped ``(..., D)`` ordered as the source axes, or a :class:`~xarrayrf.Grid`
            located in the source frame, evaluated at its sample points.

    Returns:
        The residuals, their lengths, maximum and root mean square.

    Raises:
        TypeError: If ``transform`` has neither an exact nor a declared inverse, or ``at`` is
            not real.
        ValueError: If the source is not a Cartesian frame with one unit, the inverse's
            endpoints are not the transform's exchanged, the grid is in another frame, there
            are no points, or either direction refuses a point.
    """
    forward = check_transform(transform)
    if not has_exact_or_declared_inverse(forward):
        raise TypeError(
            f"{type(forward).__name__} has neither an exact nor a declared inverse; pair a field "
            "transform with its producer's inverse through DeclaredInversePair",
        )
    source = forward.source
    if not isinstance(source, ReferenceFrame):
        raise ValueError(
            f"the source is array coordinates {source.axes}; a round-trip distance needs a "
            "Cartesian reference frame",
        )
    representation = source.coordinate_system.representation
    # Axes that all leave the unit undeclared (None) share it; a mix is refused.
    units = set(source.coordinate_system.units)
    if representation != CARTESIAN or len(units) != 1:
        raise ValueError(
            f"the source frame has a {representation} coordinate system with units "
            f"{source.coordinate_system.units}; a Euclidean distance needs Cartesian axes "
            "sharing one unit",
        )
    inverse = check_transform(exact_or_declared_inverse(forward))
    if inverse.source != forward.target or inverse.target != source:
        raise ValueError(
            f"the inverse maps {inverse.source!r} to {inverse.target!r}; expected "
            f"{forward.target!r} to {source!r}",
        )
    if isinstance(at, Grid):
        if at.frame != source:
            raise ValueError(
                f"the grid is located in {at.frame.identifier}, but the transform maps from "
                f"{source!r}; evaluate it in the transform's source frame",
            )
        points = at.points()
    else:
        points = real_float_array(at, field="at")
    width = len(source.axes)
    if points.ndim == 0 or points.shape[-1] != width:
        raise ValueError(
            f"at must have a trailing axis of length {width} ordered as {source.axes}, got "
            f"shape {points.shape}",
        )
    if points.size == 0:
        raise ValueError("at holds no points")
    mapped = _checked(forward, points, field="the transform's result")
    returned = _checked(inverse, mapped, field="the inverse's result")
    # Bytes-backed, so the published values cannot be made writeable and drift from the summary.
    residual = frozen_float_array(returned - points, field="residual")
    distance = frozen_float_array(np.linalg.norm(residual, axis=-1), field="distance")
    return RoundTripResidualReport(
        residual=residual,
        distance=distance,
        maximum=float(distance.max()),
        root_mean_square=float(np.sqrt(np.mean(distance**2))),
        unit=next(iter(units)),
    )
