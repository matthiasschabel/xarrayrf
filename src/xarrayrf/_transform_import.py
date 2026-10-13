"""Frames for an imported registration result: the rules every engine importer shares.

An engine (elastix, ITK, ANTs) computes in one physical coordinate system, ``space``, and
records no frame identity. The caller names the fixed and moving frames; the stages run between
those frames expressed in ``space``, through one intermediate frame per internal boundary; and
where the caller's frames use another coordinate system, ``coordinate_system_change`` joins them
at the ends. Defaults that need ``xarrayrf.anatomy`` (LPS) are the importers' to supply.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._composite import compose
from ._coordinate_system import CoordinateSystem
from ._frame import ReferenceFrame
from ._orientation import coordinate_system_change
from ._transform import SupportsPoints

type ReportEntries = list[tuple[str, str]]


def check_space(space: object, dimension: int, *, engine: str) -> CoordinateSystem:
    """Require a coordinate system with one axis per engine dimension."""
    if not isinstance(space, CoordinateSystem):
        raise TypeError(f"space must be a CoordinateSystem, got {type(space).__name__}")
    if len(space.axes) != dimension:
        raise ValueError(
            f"space has {len(space.axes)} axes but the {engine} transform is {dimension}-D"
        )
    return space


def stage_frames(
    stages: int,
    *,
    fixed: object,
    moving: object,
    space: CoordinateSystem,
    intermediate_frames: Sequence[ReferenceFrame] | None,
) -> list[ReferenceFrame]:
    """The frames the stages map between, in ``space``: fixed, intermediates, moving.

    Raises:
        TypeError: If ``fixed`` or ``moving`` is not a ReferenceFrame.
        ValueError: If they are the same frame, or ``intermediate_frames`` does not hold one
            frame in ``space`` per boundary between stages.
    """
    for name, frame in (("fixed", fixed), ("moving", moving)):
        if not isinstance(frame, ReferenceFrame):
            raise TypeError(f"{name} must be a ReferenceFrame, got {type(frame).__name__}")
    assert isinstance(fixed, ReferenceFrame) and isinstance(moving, ReferenceFrame)
    if fixed.is_equivalent_frame(moving):
        raise ValueError(
            "fixed and moving are the same frame; a non-identity map from a frame to itself "
            "contradicts frame identity, so give the moving image its own frame"
        )
    boundaries = stages - 1
    if intermediate_frames is None:
        between = [ReferenceFrame.local(space) for _ in range(boundaries)]
    else:
        between = list(intermediate_frames)
        if len(between) != boundaries:
            raise ValueError(
                f"intermediate_frames needs one frame per boundary between stages ({boundaries}), "
                f"got {len(between)}"
            )
        for position, frame in enumerate(between):
            if not isinstance(frame, ReferenceFrame) or frame.coordinate_system != space:
                raise ValueError(
                    f"intermediate frame {position} must be a ReferenceFrame with coordinate "
                    "system space"
                )
    return [fixed.with_coordinate_system(space), *between, moving.with_coordinate_system(space)]


def joined(
    members: Sequence[SupportsPoints],
    *,
    fixed: ReferenceFrame,
    moving: ReferenceFrame,
    space: CoordinateSystem,
    engine: str,
) -> tuple[SupportsPoints, ReportEntries]:
    """Join stages in ``space`` to the caller's frames, and compose them.

    Returns one affine when every member is affine, the member itself when there is one, else
    the chain; and a report entry for each coordinate change.

    Raises:
        ValueError: If a caller frame's coordinate system cannot be converted to ``space``.
    """
    chain: list[SupportsPoints] = list(members)
    report: ReportEntries = []
    if fixed.coordinate_system != space:
        chain.insert(0, coordinate_system_change(fixed, fixed.with_coordinate_system(space)))
        report.append(
            (
                "coordinate-system-change",
                f"the fixed frame {fixed.identifier} is converted to {engine}'s {space.axes} first",
            )
        )
    if moving.coordinate_system != space:
        chain.append(coordinate_system_change(moving.with_coordinate_system(space), moving))
        report.append(
            (
                "coordinate-system-change",
                f"the result is converted from {engine}'s {space.axes} to the moving frame "
                f"{moving.identifier} last",
            )
        )
    return (chain[0] if len(chain) == 1 else compose(*chain)), report
