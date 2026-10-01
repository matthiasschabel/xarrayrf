"""Diagnostics for anonymous worlds; numerical agreement never establishes identity."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._affine import AffineTransform
from ._orientation import coordinate_system_change
from ._sampling import Sampling
from ._transform import SupportsAffine


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
    matching = _numerically_matching(mine, theirs, change)
    remedy = "assert the shared world with frame= or rf.assume_frame"
    if matching and adoption_suffices:
        return (
            prefix
            + "the grids' points match numerically; "
            + remedy
            + "; rf.assume_frame alone suffices"
        )
    detail = (
        "the grids' points match numerically but their bindings differ"
        if matching
        else "the grids differ"
    )
    return prefix + detail + "; " + remedy + ", then use rf.resample_to"
