"""Public behavior of :class:`xarrayrf.ArrayCoordinates`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from xarrayrf import ArrayCoordinates


def test_array_coordinates_record_axes_and_units() -> None:
    coordinates = ArrayCoordinates(("slice_offset", "row", "column"), ("mm", "1", "1"))
    assert coordinates.axes == ("slice_offset", "row", "column")
    assert coordinates.units == ("mm", "1", "1")


def test_array_coordinates_have_no_identity() -> None:
    """Equal names and units compare equal, whichever array they describe."""
    first = ArrayCoordinates(("i", "j"), ("1", "1"))
    second = ArrayCoordinates(("i", "j"), ("1", "1"))
    assert first == second
    assert hash(first) == hash(second)
    assert first != ArrayCoordinates(("j", "i"), ("1", "1"))
    assert first != ArrayCoordinates(("i", "j"), ("1", "mm"))


@pytest.mark.parametrize(
    ("axes", "units", "message"),
    [
        pytest.param(("row", "row"), ("1", "1"), "unique", id="duplicate"),
        pytest.param(("row", "column"), ("1",), "one unit per axis", id="unit-count"),
        pytest.param(("row", "column"), ("1", " pixel"), "whitespace", id="padded-unit"),
        pytest.param((), (), "at least one name", id="empty"),
    ],
)
def test_declarations_are_validated(
    axes: Sequence[str], units: Sequence[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ArrayCoordinates(axes, units)


def test_axis_names_must_be_strings() -> None:
    with pytest.raises(TypeError, match="sequence of strings"):
        ArrayCoordinates(cast(Sequence[str], "ij"), ("1", "1"))


def test_array_coordinates_are_immutable() -> None:
    coordinates = ArrayCoordinates(("i",), ("1",))
    with pytest.raises(AttributeError):
        setattr(coordinates, "axes", ("j",))  # noqa: B010


def test_sample_offsets_default_to_point_samples() -> None:
    assert ArrayCoordinates(("i", "j"), ("1", "1")).sample_offset == (None, None)


def test_sample_offsets_are_recorded_per_axis_and_compared() -> None:
    coordinates = ArrayCoordinates(
        ("k", "j", "i", "echo_time"), ("mm", "1", "1", "ms"), sample_offset=(0.5, 0.5, 0, None)
    )
    assert coordinates.sample_offset == (0.5, 0.5, 0.0, None)
    assert coordinates != ArrayCoordinates(("k", "j", "i", "echo_time"), ("mm", "1", "1", "ms"))
    assert "sample_offset" in repr(coordinates)
    same = ArrayCoordinates(
        ("k", "j", "i", "echo_time"), ("mm", "1", "1", "ms"), sample_offset=(0.5, 0.5, 0.0, None)
    )
    assert coordinates == same
    assert hash(coordinates) == hash(same)


def test_a_sample_offset_may_label_a_cell_at_its_end() -> None:
    """A coordinate is its sample's location, so 1 is a cell labelled at its upper edge."""
    assert ArrayCoordinates(("t",), ("s",), sample_offset=(1.0,)).sample_offset == (1.0,)


@pytest.mark.parametrize(
    ("offsets", "error", "message"),
    [
        pytest.param((0.5,), ValueError, "one offset per axis", id="count"),
        pytest.param((0.5, 1.5), ValueError, r"\[0, 1\]", id="above"),
        pytest.param((-0.1, 0.5), ValueError, r"\[0, 1\]", id="below"),
        pytest.param((float("nan"), 0.5), ValueError, r"\[0, 1\]", id="nan"),
        pytest.param((True, 0.5), TypeError, "real number", id="boolean"),
        pytest.param(("0.5", 0.5), TypeError, "real number", id="string-entry"),
        pytest.param("05", TypeError, "sequence", id="string"),
    ],
)
def test_sample_offsets_are_validated(
    offsets: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        ArrayCoordinates(("i", "j"), ("1", "1"), sample_offset=cast(Sequence[float], offsets))


def test_array_coordinates_accept_undeclared_units_and_axis_types() -> None:
    coordinates = ArrayCoordinates(
        ("c", "y", "x"), (None, "1", "1"), axis_types=("channel", None, None)
    )
    assert coordinates.units == (None, "1", "1")
    assert coordinates.axis_types == ("channel", None, None)
    assert coordinates != ArrayCoordinates(("c", "y", "x"), (None, "1", "1"))
    assert "axis_types" in repr(coordinates)


def test_sample_offsets_accept_a_numpy_array() -> None:
    coordinates = ArrayCoordinates(("i", "j"), ("1", "1"), sample_offset=np.array([0.5, 0.0]))
    assert coordinates.sample_offset == (0.5, 0.0)
    with pytest.raises(ValueError, match="one offset per axis"):
        ArrayCoordinates(("i", "j"), ("1", "1"), sample_offset=np.array([0.5]))
    masked_array = cast(Callable[..., npt.NDArray[np.float64]], np.ma.array)
    masked = masked_array([0.5, 0.25], mask=[False, True])
    with pytest.raises(TypeError, match="masked"):
        ArrayCoordinates(("i", "j"), ("1", "1"), sample_offset=masked)
    assert ArrayCoordinates(("i",), ("1",), sample_offset=np.array([1])).sample_offset == (1.0,)
