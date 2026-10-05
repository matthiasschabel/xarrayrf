"""Explicit composition of transforms, applied first to last."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from ._affine import AffineTransform
from ._array_coordinates import ArrayCoordinates
from ._transform import (
    Endpoint,
    SupportsAffine,
    SupportsInverse,
    SupportsJacobian,
    SupportsPoints,
    check_transform,
)
from ._validation import real_float_array


def _check_chain(transforms: Sequence[object]) -> tuple[SupportsPoints, ...]:
    """Validate members and their endpoints; return them typed as point transforms."""
    members = tuple(check_transform(transform) for transform in transforms)
    if not members:
        raise ValueError("a composition needs at least one transform")
    for index, (first, second) in enumerate(pairwise(members)):
        if isinstance(first.target, ArrayCoordinates):
            raise ValueError(
                f"transform {index} ends at array coordinates {first.target.axes}, which may "
                "only begin or end a chain: array coordinates have no identity, so equal names "
                "cannot prove that two transforms describe the same array",
            )
        if first.target != second.source:
            raise ValueError(
                f"transform {index} ends at {first.target!r} but transform {index + 1} starts at "
                f"{second.source!r}; each target must equal the next source exactly",
            )
    return members


class CompositeTransform:
    """A chain of transforms applied first to last.

    ``CompositeTransform(a, b)`` maps a point by ``a`` and then by ``b``, as Astropy's ``a | b``
    and NGFF's ``sequence`` do. ITK's ``CompositeTransform``, SimpleITK and VTK's default
    PreMultiply mode apply the most recently added transform first, so a chain ported from them
    must be reversed. Each member's target must equal the next member's source exactly, and
    array coordinates may only begin or end the chain.

    Capabilities follow the members. Points are always mapped. The Jacobian is available when
    every member provides one and follows the chain rule; ``at`` is required unless every member
    is affine. The inverse is available when every member's is, and applies their inverses in
    reverse order. Where a member lacks a capability, the corresponding method raises rather
    than approximating.

    Use :func:`compose` to obtain a single collapsed :class:`~xarrayrf.AffineTransform` when
    every member is affine.
    """

    __slots__ = ("_members",)

    _members: tuple[SupportsPoints, ...]

    def __init__(self, *transforms: SupportsPoints) -> None:
        """Validate and freeze a chain.

        Args:
            *transforms: Point transforms in the order they are applied.

        Raises:
            TypeError: If a member is not a point transform with valid endpoints.
            ValueError: If no transform is given, a target differs from the next source, or an
                intermediate endpoint is array coordinates.
        """
        self._members = _check_chain(transforms)

    @property
    def transforms(self) -> tuple[SupportsPoints, ...]:
        """The members, in the order they are applied."""
        return self._members

    @property
    def source(self) -> Endpoint:
        """The first member's source."""
        return self._members[0].source

    @property
    def target(self) -> Endpoint:
        """The last member's target."""
        return self._members[-1].target

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map points through every member in order.

        Args:
            points: Values shaped ``(..., K)`` in the source's axis order.

        Returns:
            Points shaped ``(..., M)`` in the target's axis order.

        Raises:
            TypeError: If a member refuses the values' type.
            ValueError: If a member refuses the values or its domain excludes them.
        """
        # The first member validates the caller's values itself, so a masked array reaches it
        # intact instead of being unmasked here. Built-in members return finite float arrays
        # by contract, so only a user-defined last member's result is validated again.
        mapped: npt.ArrayLike = points
        for member in self._members:
            mapped = member.transform_point(mapped)
        result = (
            np.asarray(mapped, dtype=np.float64)
            if _trusted(self._members[-1])
            else real_float_array(mapped, field="composite result")
        )
        width = len(self.target.axes)
        if result.ndim == 0 or result.shape[-1] != width:
            raise ValueError(
                f"the chain returned shape {result.shape}; expected a trailing axis of length "
                f"{width} for target axes {self.target.axes}",
            )
        return result

    def jacobian(self, at: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        """Return the chain-rule Jacobian ``J_n(x_{n-1}) ... J_1(x_0)``.

        Args:
            at: Points shaped ``(..., K)`` in the source's axis order. Required unless every
                member is affine, because each later Jacobian is evaluated where the earlier
                members place the point.

        Returns:
            The Jacobian, target axes by source axes, shaped ``(M, K)`` without ``at`` or
            ``(..., M, K)`` with it.

        Raises:
            TypeError: If a member provides no Jacobian, or ``at`` has a non-real dtype.
            ValueError: If ``at`` is omitted while a member is not affine, or a member refuses
                the points.
        """
        missing = [
            index
            for index, member in enumerate(self._members)
            if not isinstance(member, SupportsJacobian)
        ]
        if missing:
            raise TypeError(
                f"transforms {missing} provide no Jacobian, so the chain has none; no finite "
                "difference is taken implicitly",
            )
        if at is None:
            if not all(isinstance(member, SupportsAffine) for member in self._members):
                raise ValueError("at is required unless every member of the chain is affine")
            product = np.eye(len(self.source.axes))
            for member in self._members:
                assert isinstance(member, SupportsJacobian)
                product = member.jacobian() @ product
            return product
        points = real_float_array(at, field="at")
        width = len(self.source.axes)
        if points.ndim == 0 or points.shape[-1] != width:
            raise ValueError(
                f"at must have a trailing axis of length {width} ordered as {self.source.axes}, "
                f"got shape {points.shape}",
            )
        product = np.broadcast_to(np.eye(width), (*points.shape[:-1], width, width))
        last = len(self._members) - 1
        for index, member in enumerate(self._members):
            assert isinstance(member, SupportsJacobian)
            product = member.jacobian(points) @ product
            if index < last:
                points = member.transform_point(points)
        result = real_float_array(product, field="composite Jacobian")
        expected = (*points.shape[:-1], len(self.target.axes), width)
        if result.shape != expected:
            raise ValueError(f"the chain's Jacobian has shape {result.shape}; expected {expected}")
        return result

    def inverse(self) -> CompositeTransform:
        """Return the chain of inverses in reverse order.

        Raises:
            TypeError: If a member provides no inverse.
            ValueError: If a member's inverse is not determined, such as a non-square affine.
        """
        missing = [
            index
            for index, member in enumerate(self._members)
            if not isinstance(member, SupportsInverse)
        ]
        if missing:
            raise TypeError(f"transforms {missing} provide no inverse, so the chain has none")
        inverses = []
        for member in reversed(self._members):
            assert isinstance(member, SupportsInverse)
            inverses.append(check_transform(member.inverse()))
        return CompositeTransform(*inverses)

    def __eq__(self, other: object) -> bool:
        """Compare members in order."""
        if not isinstance(other, CompositeTransform):
            return NotImplemented
        return self._members == other._members

    def __hash__(self) -> int:
        """Hash the members; raises ``TypeError`` if a member is unhashable."""
        return hash((CompositeTransform, self._members))

    def __repr__(self) -> str:
        """Return a representation listing the members."""
        return f"CompositeTransform({', '.join(repr(member) for member in self._members)})"


def _trusted(member: SupportsPoints) -> bool:
    """Whether a member is exactly a built-in transform, whose results need no revalidation.

    A subclass may override ``transform_point``, so only the exact classes are trusted.
    """
    return type(member) in (AffineTransform, CompositeTransform)


def _flattened(chain: CompositeTransform) -> tuple[SupportsPoints, ...]:
    """Return the chain's members with nested composites expanded, in application order."""
    members: list[SupportsPoints] = []
    for member in chain.transforms:
        if isinstance(member, CompositeTransform):
            members.extend(_flattened(member))
        else:
            members.append(member)
    return tuple(members)


def compose(*transforms: SupportsPoints) -> AffineTransform | CompositeTransform:
    """Compose transforms applied first to last, collapsing affine chains.

    When every member is affine, the result is a single :class:`~xarrayrf.AffineTransform`
    with ``matrix = M_n ... M_1`` and the correspondingly accumulated translation, so resampling
    and export can treat a chain of affines as one affine. Otherwise it is a
    :class:`CompositeTransform`. Endpoint rules are those of :class:`CompositeTransform`.

    Args:
        *transforms: Point transforms in the order they are applied.

    Returns:
        The collapsed affine, or the chain.

    Raises:
        TypeError: If a member is not a point transform with valid endpoints.
        ValueError: If no transform is given, a target differs from the next source, or an
            intermediate endpoint is array coordinates.
    """
    chain = CompositeTransform(*transforms)
    members = _flattened(chain)
    if not all(isinstance(member, SupportsAffine) for member in members):
        return chain
    # Accumulated over the flattened members, since a nested composite has no matrix of its own.
    matrix = np.eye(len(chain.source.axes))
    translation = np.zeros(len(chain.source.axes))
    for member in members:
        assert isinstance(member, SupportsAffine)
        matrix = member.matrix @ matrix
        translation = member.matrix @ translation + member.translation
    return AffineTransform.from_matrix(
        source=chain.source, target=chain.target, matrix=matrix, translation=translation
    )
