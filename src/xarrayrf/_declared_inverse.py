"""Inverse pairs declared by a producer, such as an ANTs ``1Warp`` and ``1InverseWarp``.

A field transform has no exact inverse, but registration tools often write one alongside it.
The pair owns both directions so neither transform refers to the other, which would make
equality, pickling and encoding cyclic. Nothing here verifies that the two directions invert
each other; :func:`~xarrayrf.round_trip_residual_report` measures it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from ._transform import (
    Endpoint,
    SupportsAffine,
    SupportsJacobian,
    SupportsPoints,
    check_transform,
)

Direction = Literal["forward", "backward"]


class DeclaredInversePair:
    """Two transforms that a producer declares to be each other's inverse.

    ``pair.forward`` maps like ``forward`` and ``pair.backward`` like ``backward``; each
    returns the other from :meth:`PairedTransform.declared_inverse`. The declaration is not
    verified: check it with :func:`~xarrayrf.round_trip_residual_report`.

    Equality compares the members in order with their own equality; field transforms compare
    by identity, so pairs built from the same field objects are equal.
    """

    __slots__ = ("_backward", "_forward", "_members")

    _members: tuple[SupportsPoints, SupportsPoints]
    _forward: PairedTransform
    _backward: PairedTransform

    def __init__(self, forward: SupportsPoints, backward: SupportsPoints) -> None:
        """Declare ``backward`` the inverse of ``forward``.

        Args:
            forward: The transform from ``forward.source`` to ``forward.target``.
            backward: Its declared inverse, from ``forward.target`` to ``forward.source``.

        Raises:
            TypeError: If a member is not a point transform with valid endpoints, is already a
                direction of another pair, or is affine (its inverse is exact).
            ValueError: If ``backward`` does not run from ``forward.target`` to
                ``forward.source`` exactly.
        """
        members = (check_transform(forward), check_transform(backward))
        for name, member in zip(("forward", "backward"), members, strict=True):
            if isinstance(member, PairedTransform):
                raise TypeError(
                    f"{name} is already a direction of a DeclaredInversePair and has a declared "
                    "inverse; pair its underlying transform (.transform) instead",
                )
            if isinstance(member, SupportsAffine):
                raise TypeError(
                    f"{name} is affine and has an exact inverse(); pairing it would hide its "
                    "affine and exact-inverse capabilities",
                )
        if backward.source != forward.target or backward.target != forward.source:
            raise ValueError(
                f"backward must map {forward.target!r} to {forward.source!r}, the reverse of "
                f"forward; got {backward.source!r} to {backward.target!r}",
            )
        self._members = members
        self._forward = PairedTransform._direction_of(self, "forward")
        self._backward = PairedTransform._direction_of(self, "backward")

    @property
    def forward(self) -> PairedTransform:
        """The forward direction, carrying ``backward`` as its declared inverse."""
        return self._forward

    @property
    def backward(self) -> PairedTransform:
        """The backward direction, carrying ``forward`` as its declared inverse."""
        return self._backward

    def __eq__(self, other: object) -> bool:
        """Compare both members in order."""
        if not isinstance(other, DeclaredInversePair):
            return NotImplemented
        return self._members == other._members

    def __hash__(self) -> int:
        """Hash the members; raises ``TypeError`` if a member is unhashable."""
        return hash((DeclaredInversePair, self._members))

    def __reduce__(self) -> tuple[object, ...]:
        """Rebuild through the constructor from the two members."""
        return (DeclaredInversePair, self._members)

    def __repr__(self) -> str:
        """Return a representation naming both members."""
        forward, backward = self._members
        return f"DeclaredInversePair(forward={forward!r}, backward={backward!r})"


class PairedTransform:
    """One direction of a :class:`DeclaredInversePair`; obtain it as ``pair.forward``.

    It maps points as its underlying transform does and provides a Jacobian exactly when that
    transform does. It has no exact ``inverse()``, so code that needs one keeps refusing it;
    :meth:`declared_inverse` returns the pair's other direction, unverified.

    Two directions are equal when they are the same direction of equal pairs, so equal
    directions have equal declared inverses.
    """

    __slots__ = ("_direction", "_pair")

    _pair: DeclaredInversePair
    _direction: Direction

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Refuse direct construction: a direction only exists as part of its pair."""
        raise TypeError(
            "PairedTransform is created by DeclaredInversePair(forward, backward); use "
            "pair.forward or pair.backward",
        )

    @staticmethod
    def _direction_of(pair: DeclaredInversePair, direction: Direction) -> PairedTransform:
        """Build ``direction`` of ``pair``, with a Jacobian exactly when its member has one."""
        member = pair._members[0 if direction == "forward" else 1]
        cls = (
            _PairedTransformWithJacobian
            if isinstance(member, SupportsJacobian)
            else (PairedTransform)
        )
        half = object.__new__(cls)
        half._pair = pair
        half._direction = direction
        return half

    @property
    def pair(self) -> DeclaredInversePair:
        """The pair this direction belongs to."""
        return self._pair

    @property
    def direction(self) -> Direction:
        """``"forward"`` or ``"backward"``."""
        return self._direction

    @property
    def transform(self) -> SupportsPoints:
        """The underlying transform this direction maps with."""
        return self._pair._members[0 if self._direction == "forward" else 1]

    @property
    def source(self) -> Endpoint:
        """The underlying transform's source."""
        return self.transform.source

    @property
    def target(self) -> Endpoint:
        """The underlying transform's target."""
        return self.transform.target

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map points with the underlying transform."""
        return self.transform.transform_point(points)

    def declared_inverse(self) -> PairedTransform:
        """Return the pair's other direction, which the producer declares inverts this one.

        The declaration is not verified; :func:`~xarrayrf.round_trip_residual_report` measures
        how well it holds on chosen points.
        """
        return self._pair.backward if self._direction == "forward" else self._pair.forward

    def __eq__(self, other: object) -> bool:
        """Equal when it is the same direction of an equal pair."""
        if not isinstance(other, PairedTransform):
            return NotImplemented
        return self._direction == other._direction and self._pair == other._pair

    def __hash__(self) -> int:
        """Hash the pair and direction."""
        return hash((PairedTransform, self._direction, self._pair))

    def __reduce__(self) -> tuple[object, ...]:
        """Pickle as the attribute of its pair, so both directions share one pair."""
        return (getattr, (self._pair, self._direction))

    def __repr__(self) -> str:
        """Return the pair's representation and the direction."""
        return f"{self._pair!r}.{self._direction}"


class _PairedTransformWithJacobian(PairedTransform):
    """A direction whose underlying transform provides a Jacobian.

    A separate class keeps the runtime ``SupportsJacobian`` check truthful: the core dispatches
    on it, so a direction must not claim a Jacobian its transform lacks.
    """

    __slots__ = ()

    def jacobian(self, at: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        """Return the underlying transform's Jacobian."""
        member = self.transform
        assert isinstance(member, SupportsJacobian)
        return member.jacobian(at)
