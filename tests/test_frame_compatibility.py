"""Binding diagnostics without array-representation dependencies."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from xarrayrf import AffineTransform, ArrayCoordinates, CoordinateSystem, Grid, ReferenceFrame
from xarrayrf._frame_compatibility import binding_difference
from xarrayrf._sampling import Sampling


def transform(frame: ReferenceFrame) -> AffineTransform:
    return AffineTransform.from_matrix(
        source=ArrayCoordinates(("u",), ("mm",)), target=frame, matrix=[[1]], translation=[0]
    )


def unread_samplings() -> tuple[Sampling, Sampling]:
    raise AssertionError("sampling must not be read for this diagnostic")


def unused_adoption(adopted: AffineTransform) -> bool:
    raise AssertionError("adoption must not be attempted for this diagnostic")


def test_binding_difference_named_frames_does_not_read_sampling() -> None:
    system = CoordinateSystem(("x",), ("mm",))
    mine = ReferenceFrame.declared(("test", "left"), system)
    theirs = ReferenceFrame.declared(("test", "right"), system)
    assert binding_difference(
        transform(mine),
        transform(theirs),
        samplings=unread_samplings,
        adoption_suffices=unused_adoption,
    ) == (
        "the operands are in different reference frames (test:left and test:right); "
        "resample one onto the other with a transform between the frames, or use "
        "rf.assume_frame if they are the same space"
    )


@pytest.mark.parametrize("array_target", [False, True])
def test_binding_difference_equivalent_or_non_frame_targets(array_target: bool) -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("x",), ("mm",)))
    mine = transform(frame)
    theirs = mine.with_endpoints(
        target=ArrayCoordinates(("x",), ("mm",)) if array_target else frame
    )
    assert binding_difference(
        mine, theirs, samplings=unread_samplings, adoption_suffices=unused_adoption
    ) == (
        "the operands sample the same frame on different grids; resample one onto the "
        "other with rf.resample_to"
    )


@pytest.mark.parametrize("anonymous", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("suffices", [True, False, None])
@pytest.mark.parametrize("matching", [True, False])
def test_binding_difference_anonymous_frames(
    anonymous: tuple[bool, bool], suffices: bool | None, matching: bool
) -> None:
    system = CoordinateSystem(("x",), ("mm",))
    frames = [
        ReferenceFrame.anonymous(system) if flag else ReferenceFrame.local(system)
        for flag in anonymous
    ]
    mine, theirs = (transform(frame) for frame in frames)
    left = Grid(mine, {"u": ("i", [0, 1])})._sampling()
    right = Grid(theirs, {"u": ("i", [0, 1] if matching else [2, 3])})._sampling()
    events = []

    def adoption(adopted: AffineTransform) -> bool:
        events.append("adoption")
        assert adopted == mine.with_endpoints(target=frames[1])
        return bool(suffices)

    def samplings() -> tuple[Sampling, Sampling]:
        events.append("sampling")
        return left, right

    result = binding_difference(
        mine,
        theirs,
        samplings=samplings,
        adoption_suffices=(lambda _: False) if suffices is None else adoption,
    )
    labels = [
        label
        for label, flag in zip(("left operand", "right operand"), anonymous, strict=True)
        if flag
    ]
    state = " and ".join(labels) + (" are anonymous" if len(labels) == 2 else " is anonymous")
    if suffices:
        detail = (
            "the grids' points match numerically"
            if matching
            else "adopting the frame makes the bindings equal"
        )
        remedy = "; rf.assume_frame alone suffices"
    else:
        detail = (
            "the grids' points match numerically but their bindings differ"
            if matching
            else "the grids differ"
        )
        remedy = ", then use rf.resample_to"
    assert result == (
        f"different reference frames; {state}; {detail}; "
        f"assert the shared world with frame= or rf.assume_frame{remedy}"
    )
    assert events == (["sampling"] if suffices is None else ["adoption", "sampling"])


def test_binding_difference_underivable_adoption_skips_equality() -> None:
    mine = transform(ReferenceFrame.local(CoordinateSystem(("x",), ("mm",))))
    theirs = transform(ReferenceFrame.anonymous(CoordinateSystem(("q",), ("mm",))))
    result = binding_difference(
        mine,
        theirs,
        samplings=lambda: (
            Grid(mine, {"u": ("i", [0, 1])})._sampling(),
            Grid(theirs, {"u": ("i", [0, 1])})._sampling(),
        ),
        adoption_suffices=unused_adoption,
    )
    assert "different reference frames; right operand is anonymous; " in result
    assert "the coordinate-system change is not derivable: unoriented axis" in result
    assert result.endswith("; supply an explicit transform between the frames")
    assert "rf.assume_frame" not in result


def test_binding_difference_non_affine_adoption_skips_equality() -> None:
    class PointOnly:
        def __init__(self, affine: AffineTransform) -> None:
            self.source, self.target = affine.source, affine.target
            self.affine = affine

        def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return self.affine.transform_point(points)

    system = CoordinateSystem(("x",), ("mm",))
    mine = PointOnly(transform(ReferenceFrame.anonymous(system)))
    theirs = transform(ReferenceFrame.local(system))
    assert binding_difference(
        mine,
        theirs,
        samplings=lambda: (
            Grid(mine, {"u": ("i", [0, 1])})._sampling(),
            Grid(theirs, {"u": ("i", [0, 1])})._sampling(),
        ),
        adoption_suffices=unused_adoption,
    ) == (
        "different reference frames; left operand is anonymous; adopting a frame requires "
        "an affine coordinate transform; supply an explicit transform between the frames"
    )


@pytest.mark.parametrize("error_type", [ValueError, TypeError])
def test_binding_difference_propagates_equality_errors_before_sampling(
    error_type: type[Exception],
) -> None:
    system = CoordinateSystem(("x",), ("mm",))
    mine = transform(ReferenceFrame.anonymous(system))
    theirs = transform(ReferenceFrame.local(system))

    def failing_adoption(adopted: AffineTransform) -> bool:
        raise error_type("binding equality failed")

    with pytest.raises(error_type, match="binding equality failed"):
        binding_difference(
            mine, theirs, samplings=unread_samplings, adoption_suffices=failing_adoption
        )
