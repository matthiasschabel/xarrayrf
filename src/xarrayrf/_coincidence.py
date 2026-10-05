"""Whether two geometries sample the same points, element for element."""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from ._orientation import coordinate_system_change
from ._positions import located_axes, locator, sample_columns
from ._sampling import POSITION_SLACK, Sampling, affine_roundoff, coordinate_to_position
from ._transform import SupportsAffine, SupportsInverse, SupportsPoints

COINCIDENCE_BLOCK_POINTS: Final = 1 << 20
"""Samples located per block when comparing geometries that do not form lattices."""


def is_coincident(geometry: Sampling, other: Sampling, tolerance: float) -> bool:
    """Implement :meth:`~xarrayrf.Geometry.is_coincident`; arguments are already validated."""
    if geometry.frame != other.frame and not geometry.frame.is_equivalent_frame(other.frame):
        return False
    if dict(geometry.sizes) != dict(other.sizes):
        return False
    # Both must locate points, whichever direction fails first, so errors do not depend on
    # the order of the operands.
    locator(geometry)
    locator(other)
    return _locates_within(geometry, other, tolerance) and _locates_within(
        other, geometry, tolerance
    )


def _locates_within(geometry: Sampling, other: Sampling, tolerance: float) -> bool:
    """Whether every sample of ``other`` lies within ``tolerance`` steps of its own position."""
    change = (
        None
        if other.frame == geometry.frame
        else coordinate_system_change(other.frame, geometry.frame)
    )
    exact = _affine_within(geometry, other, change, tolerance)
    if exact is not None:
        return exact
    locate = locator(geometry, slack=tolerance)
    order = list(other.dims)
    shape = tuple(other.sizes[dim] for dim in order)
    columns = [order.index(dim) for dim in geometry.dims]
    samplings = other.axes
    total = int(np.prod(shape, dtype=np.int64))
    for start in range(0, total, COINCIDENCE_BLOCK_POINTS):
        stop = min(total, start + COINCIDENCE_BLOCK_POINTS)
        indices = np.array(np.unravel_index(np.arange(start, stop), shape)).reshape(
            len(shape), stop - start
        )
        points = other.transform.transform_point(sample_columns(samplings, indices, order))
        if change is not None:
            points = change.transform_point(points)
        positions = locate(points)
        if np.isnan(positions).any():
            return False
        deviation = np.abs(positions - indices[columns].T)
        if float(deviation.max(initial=0.0)) > tolerance + POSITION_SLACK:
            return False
    return True


def _affine_within(
    geometry: Sampling, other: Sampling, change: SupportsPoints | None, tolerance: float
) -> bool | None:
    """Exact check when every map is affine, in time linear in the samples per dimension.

    ``geometry``'s source coordinates of ``other``'s samples are then affine in ``other``'s
    coordinate values, so each is a sum of one term per dimension. Along a dimension, the
    terms from the other dimensions vary independently of the position there, and the
    coordinate-to-position map is monotonic, so the largest deviation at each position is
    attained with those terms at their minimum or maximum. Returns ``None`` when a map is
    not affine.
    """
    mine, theirs = geometry.transform, other.transform
    if not (
        isinstance(mine, SupportsAffine)
        and isinstance(mine, SupportsInverse)
        and isinstance(theirs, SupportsAffine)
        and (change is None or isinstance(change, SupportsAffine))
    ):
        return None
    inverse = mine.inverse()
    if not isinstance(inverse, SupportsAffine):
        return None
    matrix, translation = theirs.matrix, theirs.translation
    if change is not None:
        assert isinstance(change, SupportsAffine)
        matrix, translation = change.matrix @ matrix, change.matrix @ translation
        translation = translation + change.translation
    weights = inverse.matrix @ matrix
    base = inverse.matrix @ translation + inverse.translation
    samplings = other.axes
    source_scale = np.array([np.abs(axis.values).max(initial=0.0) for axis in samplings])
    frame_scale = np.abs(matrix) @ source_scale + np.abs(translation)
    roundoff = affine_roundoff(inverse.matrix, inverse.translation, frame_scale)
    for sampling, row, _ in located_axes(geometry):
        dim = sampling.dim
        assert dim is not None
        size = geometry.sizes[dim]
        along = np.zeros(size)
        others = 0.0, 0.0
        constant = float(base[row])
        terms: dict[str, npt.NDArray[np.float64]] = {}
        for column, source in enumerate(samplings):
            term = weights[row, column] * source.values
            if source.dim is None:
                constant += float(term)
            else:
                terms[source.dim] = terms.get(source.dim, 0.0) + term
        for source_dim, term in terms.items():
            if source_dim == dim:
                along = along + term
            else:
                others = others[0] + float(term.min()), others[1] + float(term.max())
        expected = np.arange(size, dtype=np.float64)
        for extreme in others:
            positions = coordinate_to_position(
                sampling.values,
                constant + extreme + along,
                sampling.step,
                (tolerance, tolerance),
                roundoff=float(roundoff[row]),
            )
            if np.isnan(positions).any():
                return False
            if float(np.abs(positions - expected).max()) > tolerance + POSITION_SLACK:
                return False
    return True
