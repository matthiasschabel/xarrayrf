"""Per-point validity of a frame map, so ``resample`` fills targets a field does not map.

A field transform declared ``outside="raise"`` refuses a point beyond its support when called
directly. ``resample`` instead evaluates the frame map here: each member of the chain sees only
the rows still valid, so no invalid row reaches a later member or the source locator, and the
caller fills the rest. Only the built-in field types report validity, recognized by exact type
(a subclass may override ``transform_point``); every other member is called as usual and its
exceptions propagate, never reinterpreted as a domain exclusion.
"""

from __future__ import annotations

from typing import Final, cast

import numpy as np
import numpy.typing as npt

from ._affine import AffineTransform
from ._composite import CompositeTransform
from ._declared_inverse import PairedTransform, _PairedTransformWithJacobian
from ._field_transform import BSplineTransform, DisplacementFieldTransform, PositionFieldTransform
from ._transform import SupportsJacobian, SupportsPoints
from ._validation import real_float_array

PAIRED_TYPES: Final = (PairedTransform, _PairedTransformWithJacobian)
FIELD_TYPES: Final = (DisplacementFieldTransform, BSplineTransform, PositionFieldTransform)


def chain(transform: SupportsPoints) -> list[SupportsPoints]:
    """The members of a frame map in application order, composites and declared pairs opened.

    Only the exact built-in types are opened: a subclass may override ``transform_point``, and
    evaluating its members directly would skip the override.
    """
    if type(transform) is CompositeTransform:
        return [member for inner in transform.transforms for member in chain(inner)]
    if type(transform) in PAIRED_TYPES:
        return chain(cast(PairedTransform, transform).transform)
    return [transform]


def has_raising_field(transform: SupportsPoints | None) -> bool:
    """Whether a built-in field declared ``"raise"`` takes part in the frame map."""
    return transform is not None and any(
        type(member) in FIELD_TYPES and cast(DisplacementFieldTransform, member).outside == "raise"
        for member in chain(transform)
    )


def masked_map(
    transform: SupportsPoints, points: npt.NDArray[np.float64], *, jacobian: bool = False
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None, npt.NDArray[np.bool_]]:
    """Mapped points, optionally the chain-rule Jacobian, and validity, for points ``(N, K)``.

    Invalid rows are NaN in both results. Each member's Jacobian is taken where the earlier
    members place the point, as :meth:`CompositeTransform.jacobian` does.

    Raises:
        TypeError: If ``jacobian`` is requested and a member provides none. Anything a member
            other than a built-in field raises propagates unchanged.
    """
    members = chain(transform)
    if jacobian:
        missing = [
            index
            for index, member in enumerate(members)
            if not isinstance(member, SupportsJacobian)
        ]
        if missing:
            raise TypeError(
                f"transforms {missing} provide no Jacobian, so the chain has none; no finite "
                "difference is taken implicitly",
            )
    count = points.shape[0]
    valid = np.ones(count, dtype=bool)
    current = points
    product: npt.NDArray[np.float64] | None = None
    for member in members:
        rows = np.flatnonzero(valid)
        if not rows.size:
            break
        subset = current if rows.size == count else current[rows]
        local: npt.NDArray[np.float64] | None = None
        if type(member) in FIELD_TYPES:
            field = cast(DisplacementFieldTransform, member)
            if jacobian:
                local, _ = field._masked(subset, jacobian=True)
            mapped, kept = field._masked(subset, jacobian=False)
        elif type(member) is AffineTransform:
            if jacobian:
                local = np.asarray(member.jacobian(subset), dtype=np.float64)
            mapped = np.asarray(member.transform_point(subset), dtype=np.float64)
            kept = None
        else:
            # Validated as CompositeTransform validates an untrusted member's result: a mask, a
            # complex part or a string is refused, not discarded by a float conversion.
            if jacobian:
                local = real_float_array(
                    cast(SupportsJacobian, member).jacobian(subset), field="composite Jacobian"
                )
            mapped = real_float_array(member.transform_point(subset), field="composite result")
            kept = None
        out = np.full((count, mapped.shape[-1]), np.nan)
        out[rows] = mapped
        if local is not None:
            # Affine members return one matrix without ``at``; with it, one per row.
            local = np.broadcast_to(local, (rows.size, *local.shape[-2:]))
            step = np.full((count, *local.shape[-2:]), np.nan)
            step[rows] = local
            product = step if product is None else np.matmul(step, product)
        if kept is not None:
            valid[rows[~kept]] = False
        current = out
    width = len(transform.target.axes)
    if current.shape[-1] != width:
        # Every row became invalid before the last member could set the target width.
        current = np.full((count, width), np.nan)
    if jacobian and (product is None or product.shape[-2] != width):
        product = np.full((count, width, points.shape[-1]), np.nan)
    current[~valid] = np.nan
    if product is not None:
        product[~valid] = np.nan
    return current, product, valid
