"""Coordinate-system changes derived from axis orientation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._affine import AffineTransform
from ._coordinate_system import CARTESIAN, CoordinateSystem
from ._frame import ReferenceFrame


def _check_frame(value: object, *, field: str) -> ReferenceFrame:
    if not isinstance(value, ReferenceFrame):
        raise TypeError(f"{field} must be a ReferenceFrame, got {type(value).__name__}")
    return value


def coordinate_system_change(source: ReferenceFrame, target: ReferenceFrame) -> AffineTransform:
    """Derive the transform between two coordinate systems of one reference frame.

    The result is a signed permutation, such as diag(-1, -1, 1) from LPS to RAS, that may
    also rename axes. Oriented axes are matched by their direction pair, never by name, so
    ``x`` in one system can be ``L`` in the other. Unoriented axes pass through only when
    they are identical in both systems. Units are never rescaled; supply an explicit scaling
    :class:`~xarrayrf.AffineTransform` for a unit change.

    Args:
        source: The frame expressed in the coordinate system to convert from.
        target: The same frame expressed in the coordinate system to convert to.

    Returns:
        An affine transform from ``source`` to ``target`` with zero translation.

    Raises:
        TypeError: If either argument is not a reference frame.
        ValueError: If the frames are not equivalent, or the change is not derivable: an
            axis is oriented in one system only, the systems use
            different vocabularies or direction pairs, matched axes have different units,
            unoriented axes differ in name, unit or position, or either system is angular and
            the two differ in representation or set of axis roles. Angular axes are matched by
            role, so reordering longitude and latitude is derivable.
    """
    _check_frame(source, field="source")
    _check_frame(target, field="target")
    if not source.is_equivalent_frame(target):
        raise ValueError(
            f"{source.identifier} and {target.identifier} are not the same frame (identifier, "
            "definition and context must match); relate different frames with an explicit "
            "transform",
        )
    a = source.coordinate_system
    b = target.coordinate_system
    if a.representation != CARTESIAN or b.representation != CARTESIAN:
        return AffineTransform.from_matrix(
            source=source,
            target=target,
            matrix=_angular_permutation(a, b),
            translation=np.zeros(len(b.axes)),
        )
    oriented_a = {i: token for i, token in enumerate(a.orientation) if token is not None}
    oriented_b = {i: token for i, token in enumerate(b.orientation) if token is not None}
    if (oriented_a or oriented_b) and a.vocabulary != b.vocabulary:
        raise ValueError(
            "the coordinate systems are oriented in different vocabularies, or only one is "
            "oriented; the change is not derivable from orientation",
        )
    vocabulary = a.vocabulary
    matrix = np.zeros((len(b.axes), len(a.axes)))
    if oriented_a:
        assert vocabulary is not None  # both systems share a vocabulary when oriented
        pairs_a = {vocabulary.pair(token): i for i, token in oriented_a.items()}
        pairs_b = {vocabulary.pair(token): i for i, token in oriented_b.items()}
        if set(pairs_a) != set(pairs_b):
            raise ValueError(
                "the coordinate systems orient their axes along different direction pairs; "
                "the change is not derivable from orientation",
            )
        for pair, i in pairs_b.items():
            j = pairs_a[pair]
            if a.units[j] != b.units[i]:
                raise ValueError(
                    f"axis {a.axes[j]!r} in {a.units[j]!r} matches axis {b.axes[i]!r} in "
                    f"{b.units[i]!r}; units are never rescaled here, so supply an explicit "
                    "scaling transform",
                )
            if a.axis_types[j] != b.axis_types[i]:
                raise ValueError(
                    f"axis {a.axes[j]!r} of type {a.axis_types[j]!r} matches axis {b.axes[i]!r} "
                    f"of type {b.axis_types[i]!r}; an axis keeps its type across coordinate "
                    "systems of one frame",
                )
            matrix[i, j] = 1.0 if oriented_a[j] == oriented_b[i] else -1.0
    unoriented_a = [i for i, token in enumerate(a.orientation) if token is None]
    unoriented_b = [i for i, token in enumerate(b.orientation) if token is None]
    for i in unoriented_a:
        if i not in unoriented_b or (a.axes[i], a.units[i], a.axis_types[i]) != (
            b.axes[i],
            b.units[i],
            b.axis_types[i],
        ):
            raise ValueError(
                f"unoriented axis {a.axes[i]!r} in {a.units[i]!r} does not appear unchanged at "
                "the same position in the target; unoriented axes pass through only when "
                "identical",
            )
        matrix[i, i] = 1.0
    if len(unoriented_a) != len(unoriented_b):
        raise ValueError(
            "the coordinate systems have different numbers of unoriented axes; the change is "
            "not derivable",
        )
    return AffineTransform.from_matrix(
        source=source,
        target=target,
        matrix=matrix,
        translation=np.zeros(len(b.axes)),
    )


def _angular_permutation(a: CoordinateSystem, b: CoordinateSystem) -> npt.NDArray[np.float64]:
    """Match angular axes by role and pass other axes through unchanged.

    Unoriented axes are otherwise matched by name, unit and position only, so two angular
    declarations with the same axes, such as spherical and geodetic latitude, would derive an
    identity; representations and role sets must therefore agree first.
    """
    if a.representation != b.representation or sorted(map(str, a.axis_roles)) != sorted(
        map(str, b.axis_roles)
    ):
        raise ValueError(
            f"a {a.representation} system with roles {a.axis_roles} and a {b.representation} "
            f"system with roles {b.axis_roles} are not related by an axis permutation; use an "
            "explicit chart transition such as GeodeticToCartesian or SphericalToCartesian",
        )
    matrix = np.zeros((len(b.axes), len(a.axes)))
    for i, role in enumerate(b.axis_roles):
        if role is None:
            if (a.axes[i], a.units[i], a.axis_types[i], a.axis_roles[i]) != (
                b.axes[i],
                b.units[i],
                b.axis_types[i],
                None,
            ):
                raise ValueError(
                    f"axis {b.axes[i]!r} has no role and does not appear unchanged at the same "
                    "position in the source; such axes pass through only when identical",
                )
            matrix[i, i] = 1.0
            continue
        j = a.axis_roles.index(role)
        if a.units[j] != b.units[i]:
            raise ValueError(
                f"{role} axis {a.axes[j]!r} in {a.units[j]!r} matches {b.axes[i]!r} in "
                f"{b.units[i]!r}; units are never rescaled here, so supply an explicit "
                "scaling transform",
            )
        if a.axis_types[j] != b.axis_types[i]:
            raise ValueError(
                f"{role} axis {a.axes[j]!r} of type {a.axis_types[j]!r} matches {b.axes[i]!r} "
                f"of type {b.axis_types[i]!r}; an axis keeps its type across coordinate "
                "systems of one frame",
            )
        matrix[i, j] = 1.0
    return matrix
