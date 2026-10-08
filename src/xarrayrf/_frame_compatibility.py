"""Frame diagnostics; numerical agreement never establishes identity."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ._affine import AffineTransform
from ._frame import ReferenceFrame
from ._frame_adoption import adopt_frame
from ._orientation import coordinate_system_change
from ._sampling import Sampling
from ._transform import SupportsAffine, SupportsPoints


def binding_difference(
    mine: SupportsPoints,
    theirs: SupportsPoints,
    *,
    samplings: Callable[[], tuple[Sampling, Sampling]],
    adoption_suffices: Callable[[AffineTransform], bool],
) -> str:
    """Explain incompatible bindings, using the adapter's exact adoption equality.

    Sampling snapshots are deferred until after adoption and equality, and are needed only
    for anonymous-frame diagnostics.
    """
    left = getattr(mine, "target", None)
    right = getattr(theirs, "target", None)
    if (
        isinstance(left, ReferenceFrame)
        and isinstance(right, ReferenceFrame)
        and not left.is_equivalent_frame(right)
    ):
        if left.is_anonymous or right.is_anonymous:
            try:
                adopted = adopt_frame(mine, right)
            except ValueError:
                suffices = False
            else:
                suffices = adoption_suffices(adopted)
            a, b = samplings()
            return anonymous_frame_difference(
                a, b, labels=("left operand", "right operand"), adoption_suffices=suffices
            )
        return (
            f"the operands are in different reference frames ({left.identifier[0]}:"
            f"{left.identifier[1]} and {right.identifier[0]}:{right.identifier[1]}); "
            "resample one onto the other with a transform between the frames, or use "
            "rf.assume_frame if they are the same space"
        )
    return (
        "the operands sample the same frame on different grids; resample one onto the "
        "other with rf.resample_to"
    )


def _numerically_matching(mine: Sampling, theirs: Sampling, change: AffineTransform) -> bool:
    """Compare affine sample points, ignoring identity, in time linear in axis lengths."""
    if dict(mine.sizes) != dict(theirs.sizes):
        return False
    a, b = mine.transform, theirs.transform
    if not isinstance(a, SupportsAffine) or not isinstance(b, SupportsAffine):
        return False
    if any(size == 0 for size in mine.sizes.values()):
        return False
    base = a.translation - change.transform_point(b.translation)
    scale = np.abs(a.translation) + np.abs(change.transform_point(b.translation))
    terms: dict[str, npt.NDArray[np.float64]] = {}
    for sampling, matrix, sign in ((mine, a.matrix, 1), (theirs, change.matrix @ b.matrix, -1)):
        for column, axis in enumerate(sampling.axes):
            term = sign * matrix[:, column, None] * axis.values
            scale = scale + np.max(np.abs(term), axis=1)
            if axis.dim is None:
                base = base + term.reshape(-1)
            else:
                terms[axis.dim] = terms.get(axis.dim, np.zeros_like(term)) + term
    low, high = base.copy(), base.copy()
    for term in terms.values():
        low += np.min(term, axis=1)
        high += np.max(term, axis=1)
    # Only accumulated float64 roundoff is allowed, rather than a fraction of grid spacing.
    slack = 8 * np.finfo(np.float64).eps * np.maximum(1, scale)
    return bool(np.all(np.maximum(np.abs(low), np.abs(high)) <= slack))


def anonymous_frame_difference(
    mine: Sampling,
    theirs: Sampling,
    *,
    labels: tuple[str, str],
    adoption_suffices: bool,
) -> str:
    """Name the anonymous operand and only offer remedies its mappings support."""
    anonymous = [
        label
        for label, frame in zip(labels, (mine.frame, theirs.frame), strict=True)
        if frame.is_anonymous
    ]
    state = " and ".join(anonymous) + (" are anonymous" if len(anonymous) == 2 else " is anonymous")
    prefix = f"different reference frames; {state}; "
    try:
        change = coordinate_system_change(
            mine.frame.with_coordinate_system(theirs.frame.coordinate_system), mine.frame
        )
    except ValueError as error:
        return (
            prefix
            + f"the coordinate-system change is not derivable: {error}; supply an explicit transform between the frames"
        )
    if not isinstance(mine.transform, SupportsAffine) or not isinstance(
        theirs.transform, SupportsAffine
    ):
        return (
            prefix
            + "adopting a frame requires an affine coordinate transform; supply an explicit transform between the frames"
        )
    remedy = "assert the shared world with frame= or rf.assume_frame"
    if adoption_suffices:
        detail = (
            "the grids' points match numerically"
            if _numerically_matching(mine, theirs, change)
            else "adopting the frame makes the bindings equal"
        )
        return prefix + detail + "; " + remedy + "; rf.assume_frame alone suffices"
    matching = _numerically_matching(mine, theirs, change)
    detail = (
        "the grids' points match numerically but their bindings differ"
        if matching
        else "the grids differ"
    )
    return prefix + detail + "; " + remedy + ", then use rf.resample_to"
