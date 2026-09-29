"""Coordinate-system changes and axis codes: acceptance cases 1-7 of the frame design."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from xarrayrf import (
    AxisCode,
    CoordinateSystem,
    DirectionVocabulary,
    ReferenceFrame,
    coordinate_system_change,
)

ATOL = 1e-12
"""The derived matrices are exact signed permutations; this only absorbs trigonometry."""

UID = ("test-frame", "patient-1")
ANATOMY = DirectionVocabulary(
    "test-anatomy", ("right-to-left", "anterior-to-posterior", "inferior-to-superior")
)
GEOGRAPHY = DirectionVocabulary("test-geography", ("west-to-east", "south-to-north", "down-to-up"))
LPS = ("right-to-left", "anterior-to-posterior", "inferior-to-superior")
RAS = ("left-to-right", "posterior-to-anterior", "inferior-to-superior")


def oriented(
    orientation: tuple[str | None, ...],
    *,
    axes: tuple[str, ...] = ("x", "y", "z"),
    units: tuple[str, ...] = ("mm", "mm", "mm"),
    vocabulary: DirectionVocabulary = ANATOMY,
) -> CoordinateSystem:
    return CoordinateSystem(axes, units, vocabulary=vocabulary, orientation=orientation)


def patient(system: CoordinateSystem, **context: str) -> ReferenceFrame:
    return ReferenceFrame.declared(UID, system, context=context or None)


def test_lps_and_ras_are_one_frame_related_by_a_derived_flip() -> None:
    """Case 1: equivalent, not equal, and diag(-1, -1, 1) between them."""
    lps = patient(oriented(LPS))
    ras = lps.with_coordinate_system(oriented(RAS))
    assert lps.is_equivalent_frame(ras)
    assert lps != ras
    assert not lps.conflicts_with(ras)
    change = coordinate_system_change(lps, ras)
    assert change.source == lps
    assert change.target == ras
    assert_allclose(change.matrix, np.diag([-1.0, -1.0, 1.0]), rtol=0, atol=ATOL)
    assert_allclose(change.translation, 0.0, rtol=0, atol=ATOL)
    assert_allclose(change.transform_point([10.0, 20.0, 30.0]), [-10.0, -20.0, 30.0], atol=ATOL)


def test_same_identifier_with_different_context_conflicts_and_has_no_change() -> None:
    """Case 1 (continued); also case 8's generic rule for differing defining context."""
    supine = patient(oriented(LPS), position="supine")
    prone = patient(oriented(LPS), position="prone")
    assert supine.conflicts_with(prone)
    with pytest.raises(ValueError, match="not the same frame"):
        coordinate_system_change(supine, prone)


def test_renamed_axes_are_matched_by_direction_not_by_name() -> None:
    """Case 2: x,y,z in LPS order and a permuted, renamed S,R,A system."""
    source = patient(oriented(LPS))
    target = source.with_coordinate_system(
        oriented(
            ("inferior-to-superior", "left-to-right", "posterior-to-anterior"),
            axes=("S", "R", "A"),
        )
    )
    change = coordinate_system_change(source, target)
    expected = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    assert_allclose(change.matrix, expected, rtol=0, atol=ATOL)
    assert change.source.axes == ("x", "y", "z")
    assert change.target.axes == ("S", "R", "A")


def test_unoriented_axes_pass_through_when_identical() -> None:
    source = patient(oriented(("right-to-left", None), axes=("x", "t"), units=("mm", "s")))
    target = source.with_coordinate_system(
        oriented(("left-to-right", None), axes=("x", "t"), units=("mm", "s"))
    )
    assert_allclose(
        coordinate_system_change(source, target).matrix, np.diag([-1.0, 1.0]), rtol=0, atol=ATOL
    )


def build_underivable(case: str) -> tuple[ReferenceFrame, ReferenceFrame]:
    source = patient(oriented(LPS))
    targets: dict[str, Callable[[], CoordinateSystem]] = {
        "partly-oriented": lambda: oriented((*RAS[:2], None)),
        "unoriented-target": lambda: CoordinateSystem(("x", "y", "z"), ("mm",) * 3),
        "other-vocabulary": lambda: oriented(
            ("west-to-east", "south-to-north", "down-to-up"), vocabulary=GEOGRAPHY
        ),
        "units": lambda: oriented(RAS, units=("m", "m", "m")),
    }
    if case == "unoriented-renamed":
        plain = patient(CoordinateSystem(("x", "y"), ("mm", "mm")))
        return plain, plain.with_coordinate_system(CoordinateSystem(("y", "x"), ("mm", "mm")))
    if case == "oriented-versus-unoriented-name":
        mixed = patient(oriented(("right-to-left", None), axes=("x", "y"), units=("mm", "mm")))
        other = oriented((None, "left-to-right"), axes=("x", "y"), units=("mm", "mm"))
        return mixed, mixed.with_coordinate_system(other)
    return source, source.with_coordinate_system(targets[case]())


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("partly-oriented", "different direction pairs"),
        ("unoriented-target", "only one is oriented"),
        ("other-vocabulary", "different vocabularies"),
        ("units", "never rescaled"),
        ("unoriented-renamed", "pass through only when identical"),
        ("oriented-versus-unoriented-name", "pass through only when identical"),
    ],
)
def test_underivable_changes_raise_but_frames_stay_equivalent(case: str, message: str) -> None:
    """Case 3: none of these is a contradiction; the change simply is not derivable."""
    source, target = build_underivable(case)
    assert source.is_equivalent_frame(target)
    assert not source.conflicts_with(target)
    with pytest.raises(ValueError, match=message):
        coordinate_system_change(source, target)


def test_only_the_target_being_oriented_is_not_derivable() -> None:
    source = patient(CoordinateSystem(("x", "y", "z"), ("mm",) * 3))
    with pytest.raises(ValueError, match="only one is oriented"):
        coordinate_system_change(source, source.with_coordinate_system(oriented(LPS)))


def test_a_different_count_of_unoriented_axes_is_not_derivable() -> None:
    source = patient(oriented(("right-to-left",), axes=("x",), units=("mm",)))
    target = source.with_coordinate_system(
        oriented(("left-to-right", None), axes=("x", "t"), units=("mm", "s"))
    )
    with pytest.raises(ValueError, match="different numbers of unoriented axes"):
        coordinate_system_change(source, target)


def test_change_requires_frames() -> None:
    with pytest.raises(TypeError, match="source must be a ReferenceFrame"):
        coordinate_system_change(cast(ReferenceFrame, "lps"), patient(oriented(LPS)))


def test_regional_names_live_outside_the_derived_change() -> None:
    """Case 5: naming is an adapter layer over constant canonical tokens.

    Modelled on DICOM quadrupeds, where +y reads dorsal on the trunk and cranial on proximal
    limbs. The adapter's naming table varies by region; the core sees only the canonical
    token, so one derived change serves every region.
    """
    quadruped = DirectionVocabulary(
        "test-quadruped", ("right-to-left", "minus-y-to-plus-y", "minus-z-to-plus-z")
    )
    names = {
        ("minus-y-to-plus-y", "trunk"): "dorsal",
        ("minus-y-to-plus-y", "proximal limb"): "cranial",
    }
    source = patient(
        oriented(("right-to-left", "minus-y-to-plus-y", "minus-z-to-plus-z"), vocabulary=quadruped)
    )
    target = source.with_coordinate_system(
        oriented(("left-to-right", "plus-y-to-minus-y", "minus-z-to-plus-z"), vocabulary=quadruped)
    )
    assert names[("minus-y-to-plus-y", "trunk")] != names[("minus-y-to-plus-y", "proximal limb")]
    change = coordinate_system_change(source, target)
    assert_allclose(change.matrix, np.diag([-1.0, -1.0, 1.0]), rtol=0, atol=ATOL)


def test_axis_codes_rank_an_oblique_vector() -> None:
    """Case 7: nearest signed axes with the residual angle a single label would hide."""
    system = oriented(LPS)
    vector = [-math.sin(math.radians(20.0)), 0.0, math.cos(math.radians(20.0))]
    codes = system.axis_codes(vector)
    expected = [
        ("z", "inferior-to-superior", math.radians(20.0)),
        ("x", "left-to-right", math.radians(70.0)),
        ("y", "anterior-to-posterior", math.pi / 2),
    ]
    assert all(isinstance(code, AxisCode) for code in codes)
    assert [(code.axis, code.direction) for code in codes] == [e[:2] for e in expected]
    assert_allclose([code.angle for code in codes], [e[2] for e in expected], rtol=0, atol=ATOL)


def test_axis_codes_ignore_unoriented_axes() -> None:
    """A time coordinate in seconds must not tilt an angle measured in millimetres."""
    system = oriented(("right-to-left", None), axes=("x", "t"), units=("mm", "s"))
    codes = system.axis_codes([1.0, 1000.0])
    assert [code.axis for code in codes] == ["x"]
    assert codes[0].direction == "right-to-left"
    assert codes[0].angle == pytest.approx(0.0, abs=ATOL)
    with pytest.raises(ValueError, match="zero along every oriented axis"):
        system.axis_codes([0.0, 5.0])


@pytest.mark.parametrize(
    ("system", "vector", "message"),
    [
        pytest.param(CoordinateSystem(("x",), ("mm",)), [1.0], "no oriented axis", id="none"),
        pytest.param(
            oriented(LPS, units=("mm", "mm", "m")), [1.0, 0.0, 0.0], "different units", id="units"
        ),
        pytest.param(oriented(LPS), [0.0, 0.0, 0.0], "zero along every oriented axis", id="zero"),
        pytest.param(oriented(LPS), [1.0, 0.0], "one value per axis", id="length"),
        pytest.param(oriented(LPS), [float("nan"), 0.0, 0.0], "finite", id="nan"),
    ],
)
def test_axis_codes_refuse_meaningless_questions(
    system: CoordinateSystem, vector: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        system.axis_codes(vector)


def test_matched_axes_must_keep_their_type() -> None:
    typed = CoordinateSystem(
        ("x", "y", "z", "t"),
        ("mm", "mm", "mm", "s"),
        axis_types=("space", "space", "space", "time"),
        vocabulary=ANATOMY,
        orientation=(*LPS, None),
    )
    frame = ReferenceFrame.declared(UID, typed)
    ras = CoordinateSystem(
        ("x", "y", "z", "t"),
        ("mm", "mm", "mm", "s"),
        axis_types=("space", "space", "space", "time"),
        vocabulary=ANATOMY,
        orientation=(*RAS, None),
    )
    change = coordinate_system_change(frame, frame.with_coordinate_system(ras))
    assert_allclose(change.matrix, np.diag([-1.0, -1.0, 1.0, 1.0]), atol=ATOL)
    for types in (("space", "space", "space", "echo"), ("space", "space", None, "time")):
        other = CoordinateSystem(
            ("x", "y", "z", "t"),
            ("mm", "mm", "mm", "s"),
            axis_types=types,
            vocabulary=ANATOMY,
            orientation=(*RAS, None),
        )
        with pytest.raises(ValueError, match=r"type|unchanged"):
            coordinate_system_change(frame, frame.with_coordinate_system(other))


def test_an_unoriented_axis_without_a_unit_passes_through() -> None:
    system = CoordinateSystem(
        ("c", "x", "y", "z"), (None, "mm", "mm", "mm"), vocabulary=ANATOMY, orientation=(None, *LPS)
    )
    frame = ReferenceFrame.declared(UID, system)
    ras = CoordinateSystem(
        ("c", "x", "y", "z"), (None, "mm", "mm", "mm"), vocabulary=ANATOMY, orientation=(None, *RAS)
    )
    change = coordinate_system_change(frame, frame.with_coordinate_system(ras))
    assert_allclose(change.matrix, np.diag([1.0, -1.0, -1.0, 1.0]), atol=ATOL)
