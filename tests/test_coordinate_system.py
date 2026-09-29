"""Public behavior of :class:`xarrayrf.CoordinateSystem` and :class:`xarrayrf.DirectionVocabulary`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

from xarrayrf import CoordinateSystem, DirectionVocabulary

ANATOMY = DirectionVocabulary(
    "test-anatomy", ("right-to-left", "anterior-to-posterior", "inferior-to-superior")
)
LPS = ("right-to-left", "anterior-to-posterior", "inferior-to-superior")


def test_vocabulary_implies_each_opposite() -> None:
    assert "left-to-right" in ANATOMY.directions
    assert len(ANATOMY.directions) == 6
    assert ANATOMY.opposite("posterior-to-anterior") == "anterior-to-posterior"
    assert ANATOMY.pair("left-to-right") == ANATOMY.pair("right-to-left")


@pytest.mark.parametrize(
    "token",
    ["RAI", "LPS", "left", "left-to-", "-to-right", "left-to-left", "a-to-b-to-c"],
)
def test_bare_or_malformed_direction_codes_are_refused(token: str) -> None:
    """'RAI' names 'from' directions in ITK's legacy codes and 'to' directions elsewhere."""
    with pytest.raises(ValueError, match="explicit direction"):
        DirectionVocabulary("codes", (token,))


def test_vocabulary_lists_each_pair_once() -> None:
    with pytest.raises(ValueError, match="repeats a pair"):
        DirectionVocabulary("twice", ("left-to-right", "right-to-left"))


def test_vocabulary_needs_at_least_one_direction() -> None:
    with pytest.raises(ValueError, match="at least one direction"):
        DirectionVocabulary("empty", ())


def test_vocabulary_refuses_a_bare_string_of_directions() -> None:
    with pytest.raises(TypeError, match="iterable of tokens"):
        DirectionVocabulary("bare", "left-to-right")


def test_vocabulary_check_refuses_foreign_tokens() -> None:
    with pytest.raises(ValueError, match="not in vocabulary"):
        ANATOMY.check("west-to-east")


def test_vocabularies_compare_by_identifier_and_directions() -> None:
    same = DirectionVocabulary(
        "test-anatomy", ("left-to-right", "posterior-to-anterior", "superior-to-inferior")
    )
    renamed = DirectionVocabulary("other", LPS)
    assert same == ANATOMY
    assert hash(same) == hash(ANATOMY)
    assert renamed != ANATOMY


def test_oriented_system_records_one_direction_per_axis() -> None:
    system = CoordinateSystem(
        ("L", "P", "S"), ("mm", "mm", "mm"), vocabulary=ANATOMY, orientation=LPS
    )
    assert system.orientation == LPS
    assert system.vocabulary is ANATOMY
    assert CoordinateSystem(("x",), ("mm",)).orientation == (None,)


def test_equality_covers_orientation() -> None:
    lps = CoordinateSystem(("x", "y", "z"), ("mm",) * 3, vocabulary=ANATOMY, orientation=LPS)
    ras = CoordinateSystem(
        ("x", "y", "z"),
        ("mm",) * 3,
        vocabulary=ANATOMY,
        orientation=("left-to-right", "posterior-to-anterior", "inferior-to-superior"),
    )
    unoriented = CoordinateSystem(("x", "y", "z"), ("mm",) * 3)
    assert lps != ras
    assert lps != unoriented
    assert len({lps, ras, unoriented}) == 3


def test_two_axes_cannot_share_a_direction_pair() -> None:
    with pytest.raises(ValueError, match="each axis needs its own direction pair"):
        CoordinateSystem(
            ("a", "b"),
            ("mm", "mm"),
            vocabulary=ANATOMY,
            orientation=("left-to-right", "right-to-left"),
        )


def test_orientation_requires_its_vocabulary() -> None:
    with pytest.raises(ValueError, match="require the vocabulary"):
        CoordinateSystem(("x",), ("mm",), orientation=("left-to-right",))


def test_orientation_tokens_must_come_from_the_vocabulary() -> None:
    with pytest.raises(ValueError, match="not in vocabulary"):
        CoordinateSystem(("x",), ("mm",), vocabulary=ANATOMY, orientation=("west-to-east",))


def test_orientation_needs_one_entry_per_axis() -> None:
    with pytest.raises(ValueError, match="one entry per axis"):
        CoordinateSystem(("x", "y"), ("mm", "mm"), vocabulary=ANATOMY, orientation=(None,))


def test_orientation_must_be_a_sequence() -> None:
    with pytest.raises(TypeError, match="orientation must be a sequence"):
        CoordinateSystem(("x",), ("mm",), vocabulary=ANATOMY, orientation="left-to-right")


def test_vocabulary_must_be_a_vocabulary() -> None:
    with pytest.raises(TypeError, match="DirectionVocabulary"):
        CoordinateSystem(("x",), ("mm",), vocabulary=cast(DirectionVocabulary, "anatomy"))


@pytest.mark.parametrize("unit", ["micrometer", "um", "mm", "1", "metre"])
def test_units_are_open_strings(unit: str) -> None:
    assert CoordinateSystem(("x",), (unit,)).units == (unit,)


def test_units_compare_exactly() -> None:
    assert CoordinateSystem(("x",), ("mm",)) != CoordinateSystem(("x",), ("millimeter",))


@pytest.mark.parametrize("unit", ["", " mm", "mm "])
def test_malformed_units_are_refused(unit: str) -> None:
    with pytest.raises(ValueError, match=r"whitespace|empty"):
        CoordinateSystem(("x",), (unit,))


@pytest.mark.parametrize(
    "unit",
    ["rad", "deg", "degrees", "arcsec", "degrees_east", "degrees_north", "degree_N", "arc_degree"],
)
def test_cartesian_axes_refuse_known_angular_units(unit: str) -> None:
    with pytest.raises(ValueError, match="angular"):
        CoordinateSystem(("lon", "lat"), (unit, unit))


def test_axis_and_unit_counts_must_agree() -> None:
    with pytest.raises(ValueError, match="one unit per axis"):
        CoordinateSystem(("x", "y", "z"), ("mm", "mm"))


def test_axis_names_must_be_unique_and_present() -> None:
    with pytest.raises(ValueError, match="unique"):
        CoordinateSystem(("x", "x"), ("mm", "mm"))
    with pytest.raises(ValueError, match="at least one name"):
        CoordinateSystem((), ())


def test_unsupported_representation_is_rejected() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        CoordinateSystem(("x",), ("mm",), representation="spherical")


@pytest.mark.parametrize(
    ("axes", "message"),
    [
        pytest.param(("x", 2), "entries must be strings", id="mixed"),
        pytest.param("xy", "sequence of strings", id="bare-string"),
        pytest.param((("x",),), "entries must be strings", id="nested"),
    ],
)
def test_axis_types_are_validated(axes: object, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        CoordinateSystem(cast(Sequence[str], axes), ("mm",))


def test_coordinate_system_is_immutable() -> None:
    system = CoordinateSystem(("x",), ("mm",))
    with pytest.raises(AttributeError):
        setattr(system, "axes", ("y",))  # noqa: B010


@pytest.mark.parametrize("token", [" left-to-right", "left-to-right ", "left -to-right"])
def test_padded_direction_tokens_are_refused(token: str) -> None:
    with pytest.raises(ValueError, match="explicit direction"):
        DirectionVocabulary("padded", (token,))


def test_a_vocabulary_without_orientation_is_discarded() -> None:
    with_vocabulary = CoordinateSystem(("x",), ("mm",), vocabulary=ANATOMY)
    plain = CoordinateSystem(("x",), ("mm",))
    assert with_vocabulary.vocabulary is None
    assert with_vocabulary == plain
    assert hash(with_vocabulary) == hash(plain)


def test_units_may_be_undeclared_which_is_not_dimensionless() -> None:
    system = CoordinateSystem(("x", "c"), ("mm", None))
    assert system.units == ("mm", None)
    assert system != CoordinateSystem(("x", "c"), ("mm", "1"))
    with pytest.raises(ValueError, match="use None for an undeclared entry"):
        CoordinateSystem(("x", "c"), ("mm", ""))


def test_axis_types_are_open_declarations_compared_exactly() -> None:
    system = CoordinateSystem(
        ("t", "x", "c"), ("s", "m", None), axis_types=("time", "space", "channel")
    )
    assert system.axis_types == ("time", "space", "channel")
    assert CoordinateSystem(("x",), ("m",)).axis_types == (None,)
    assert system != CoordinateSystem(("t", "x", "c"), ("s", "m", None))
    same = CoordinateSystem(
        ("t", "x", "c"), ("s", "m", None), axis_types=("time", "space", "channel")
    )
    assert system == same and hash(system) == hash(same)
    assert "axis_types" in repr(system)
    assert CoordinateSystem(("f",), ("Hz",), axis_types=("em.freq",)).axis_types == ("em.freq",)


@pytest.mark.parametrize(
    ("axis_types", "error", "message"),
    [
        pytest.param(("space",), ValueError, "one type per axis", id="count"),
        pytest.param(("space", ""), ValueError, "empty", id="empty"),
        pytest.param(("space", " time"), ValueError, "whitespace", id="padded"),
        pytest.param(("space", 1), TypeError, "strings or None", id="entry"),
        pytest.param("space", TypeError, "sequence", id="string"),
    ],
)
def test_axis_type_declarations_are_validated(
    axis_types: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        CoordinateSystem(("x", "t"), ("m", "s"), axis_types=cast(Sequence[str], axis_types))


def test_an_oriented_axis_needs_a_unit() -> None:
    with pytest.raises(ValueError, match="oriented but declares no unit"):
        CoordinateSystem(("x", "y", "z"), ("mm", None, "mm"), vocabulary=ANATOMY, orientation=LPS)
