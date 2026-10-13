"""The Jacobian determinant of a transform: where it folds, as a model-independent check.

A deformable registration is invertible only where its Jacobian determinant is positive.
Regularized and physics-informed models guarantee that only up to discretization, so this is
the check of a result, whatever produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._grid import Grid
from ._transform import SupportsJacobian, check_endpoint
from ._validation import frozen_float_array, real_float_array


@dataclass(frozen=True, slots=True, eq=False)
class JacobianDeterminantReport:
    """The determinant of a transform's Jacobian at a set of points.

    Attributes:
        determinant: Read-only float64 values, shaped like the points' leading axes, or like a
            grid's sizes in ``grid.dims`` order.
        minimum: The smallest determinant.
        folding_fraction: The fraction of points where the determinant is zero or negative,
            where the map folds and has no local inverse.
    """

    determinant: npt.NDArray[np.float64]
    minimum: float
    folding_fraction: float


def jacobian_determinant_report(
    transform: SupportsJacobian, at: npt.ArrayLike | Grid
) -> JacobianDeterminantReport:
    """Report the Jacobian determinant of ``transform`` at points or at a grid's samples.

    Args:
        transform: A transform with a square Jacobian (as many target axes as source axes).
        at: Points shaped ``(..., D)`` ordered as the source axes, or a :class:`~xarrayrf.Grid`
            located in the transform's source frame, evaluated at its sample points.

    Returns:
        The determinants, their minimum and the fraction of folded points.

    Raises:
        TypeError: If ``transform`` has no Jacobian or ``at`` is not real.
        ValueError: If the Jacobian is not square, the grid is in another frame, there are no
            points, or the transform refuses a point.
    """
    if not isinstance(transform, SupportsJacobian):
        raise TypeError(f"{type(transform).__name__} provides no Jacobian")
    source = check_endpoint(transform.source, field="transform.source")
    target = check_endpoint(transform.target, field="transform.target")
    if len(source.axes) != len(target.axes):
        raise ValueError(
            f"the Jacobian maps {len(source.axes)} source axes to {len(target.axes)} target axes; "
            "a determinant needs a square Jacobian",
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
    if points.size == 0:
        raise ValueError("at holds no points")
    jacobian = real_float_array(transform.jacobian(points), field="Jacobian")
    expected = (*points.shape[:-1], len(target.axes), len(source.axes))
    if jacobian.shape != expected:
        raise ValueError(f"the Jacobian has shape {jacobian.shape}; expected {expected}")
    # Bytes-backed, so the published values cannot be made writeable and drift from the summary.
    determinant = frozen_float_array(np.linalg.det(jacobian), field="determinant")
    return JacobianDeterminantReport(
        determinant=determinant,
        minimum=float(determinant.min()),
        folding_fraction=float(np.mean(determinant <= 0.0)),
    )
