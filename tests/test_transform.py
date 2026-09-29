"""Public behavior of the transform protocols and :func:`xarrayrf.transform_named`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import xarrayrf as xrf

ATOL = 1e-12
"""Rounding allowance for a few multiply-adds on exact synthetic values."""

UV = xrf.ArrayCoordinates(("u", "v"), ("1", "1"))


class Shift:
    """A user-defined point transform that scales and translates each coordinate.

    It implements only ``transform_point``, so it has no linearization capability. The
    keyword arguments let a test declare something malformed.
    """

    def __init__(
        self,
        target: object,
        *,
        source: object = UV,
        shape_error: bool = False,
        result: float | None = None,
    ) -> None:
        self._source = source
        self._target = target
        self._shape_error = shape_error
        self._result = result

    @property
    def source(self) -> xrf.Endpoint:
        return cast(xrf.Endpoint, self._source)

    @property
    def target(self) -> xrf.Endpoint:
        return cast(xrf.Endpoint, self._target)

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        values: npt.NDArray[np.float64] = np.asarray(points, dtype=np.float64)
        if self._shape_error:
            return values[..., :1]
        if self._result is not None:
            return np.full(values.shape, self._result)
        shifted: npt.NDArray[np.float64] = values * np.array([2.0, 3.0]) + np.array([10.0, 20.0])
        return shifted


@pytest.fixture
def frame() -> xrf.ReferenceFrame:
    return xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("mm", "mm")))


@pytest.fixture
def affine(frame: xrf.ReferenceFrame) -> xrf.AffineTransform:
    return xrf.AffineTransform(
        source=xrf.ArrayCoordinates(("row", "column"), ("1", "1")),
        target=frame,
        matrix=((0.0, 1.0), (1.0, 0.0)),
        translation=(0.0, 0.0),
    )


def test_affine_satisfies_its_capability_protocols(affine: xrf.AffineTransform) -> None:
    assert isinstance(affine, xrf.Transform)
    assert isinstance(affine, xrf.SupportsPoints)
    assert isinstance(affine, xrf.SupportsJacobian)
    assert isinstance(affine, xrf.SupportsAffine)
    assert isinstance(affine, xrf.SupportsInverse)


def test_a_user_defined_transform_is_recognized_structurally(frame: xrf.ReferenceFrame) -> None:
    shift = Shift(frame)
    assert isinstance(shift, xrf.SupportsPoints)
    assert not isinstance(shift, xrf.SupportsJacobian)
    assert not isinstance(shift, xrf.SupportsAffine)


def test_transform_named_evaluates_a_user_defined_transform(frame: xrf.ReferenceFrame) -> None:
    mapped = xrf.transform_named(Shift(frame), {"v": [1.0, 2.0], "u": 5.0})
    assert_allclose(mapped["x"], [20.0, 20.0], rtol=0, atol=ATOL)
    assert_allclose(mapped["y"], [23.0, 26.0], rtol=0, atol=ATOL)


def test_transform_named_matches_by_name_not_position(affine: xrf.AffineTransform) -> None:
    forward = xrf.transform_named(affine, {"row": 1.0, "column": 2.0})
    reversed_order = xrf.transform_named(affine, {"column": 2.0, "row": 1.0})
    assert float(forward["x"]) == pytest.approx(2.0, abs=ATOL)
    assert float(reversed_order["x"]) == pytest.approx(2.0, abs=ATOL)
    assert forward["x"].shape == ()


def test_transform_named_requires_every_source_axis(affine: xrf.AffineTransform) -> None:
    with pytest.raises(ValueError, match="missing values for source axes"):
        xrf.transform_named(affine, {"row": 1.0})
    with pytest.raises(ValueError, match="unexpected coordinates"):
        xrf.transform_named(affine, {"row": 1.0, "column": 2.0, "slice": 0.0})


def test_transform_named_refuses_a_non_transform() -> None:
    with pytest.raises(TypeError, match="must implement SupportsPoints"):
        xrf.transform_named(cast(xrf.SupportsPoints, object()), {})


def test_transform_named_refuses_a_malformed_result(frame: xrf.ReferenceFrame) -> None:
    with pytest.raises(ValueError, match="returned shape"):
        xrf.transform_named(Shift(frame, shape_error=True), {"u": [1.0], "v": [2.0]})


def test_a_non_finite_result_is_refused_rather_than_returned(frame: xrf.ReferenceFrame) -> None:
    with pytest.raises(ValueError, match="transform_point result must contain only finite"):
        xrf.transform_named(Shift(frame, result=float("nan")), {"u": 1.0, "v": 2.0})


def test_transform_named_rejects_a_non_mapping(affine: xrf.AffineTransform) -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        xrf.transform_named(affine, cast(Mapping[str, npt.ArrayLike], [1.0, 2.0]))


@pytest.mark.parametrize("field", ["source", "target"])
@pytest.mark.parametrize("value", [None, "patient", ("u", "v")])
def test_endpoints_must_be_frames_or_array_coordinates(
    frame: xrf.ReferenceFrame, field: str, value: object
) -> None:
    transform = Shift(frame, source=value) if field == "source" else Shift(value)
    with pytest.raises(TypeError, match=f"{field} must be a ReferenceFrame or ArrayCoordinates"):
        xrf.check_transform(transform)


def test_check_transform_returns_a_valid_transform(affine: xrf.AffineTransform) -> None:
    assert xrf.check_transform(affine) is affine


def test_a_transform_may_end_at_array_coordinates(frame: xrf.ReferenceFrame) -> None:
    """The exact part of a frame-to-sample lookup maps a frame into array coordinates."""
    to_array = Shift(xrf.ArrayCoordinates(("i", "j"), ("1", "1")), source=frame)
    mapped = xrf.transform_named(to_array, {"x": 1.0, "y": 1.0})
    assert set(mapped) == {"i", "j"}


def test_geometry_evaluates_a_user_defined_transform(frame: xrf.ReferenceFrame) -> None:
    array = xr.DataArray(
        np.zeros((2, 3)), dims=("v", "u"), coords={"v": [0.0, 1.0], "u": [5.0, 6.0, 7.0]}
    )
    geometry = xrf.Geometry(array, Shift(frame), dims=("v", "u"))
    point = geometry.point_at(v=1, u=2)
    assert list(point.axis.values) == ["x", "y"]
    assert list(point.units.values) == ["mm", "mm"]
    assert_allclose(point.values, [24.0, 23.0], rtol=0, atol=ATOL)
    assert geometry.frame is frame


def test_geometry_refuses_a_transform_between_frames(frame: xrf.ReferenceFrame) -> None:
    stage = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("u", "v"), ("1", "1")))
    array = xr.DataArray(np.zeros(1), dims=("u",), coords={"u": [0.0], "v": 0.0})
    with pytest.raises(ValueError, match="needs a transform from ArrayCoordinates"):
        xrf.Geometry(array, Shift(frame, source=stage), dims=("u",))


def test_geometry_refuses_a_transform_into_array_coordinates() -> None:
    array = xr.DataArray(np.zeros(1), dims=("u",), coords={"u": [0.0], "v": 0.0})
    other = xrf.ArrayCoordinates(("i", "j"), ("1", "1"))
    with pytest.raises(ValueError, match="needs a transform into a ReferenceFrame"):
        xrf.Geometry(array, Shift(other), dims=("u",))


def test_axes_do_not_share_memory_with_each_other(affine: xrf.AffineTransform) -> None:
    mapped = xrf.transform_named(affine, {"row": [1.0, 2.0], "column": [3.0, 4.0]})
    assert not np.shares_memory(mapped["x"], mapped["y"])


if TYPE_CHECKING:
    # Static conformance: mypy fails here if AffineTransform drops a member or changes a
    # parameter list or type in a way the protocols reject, which a runtime isinstance check
    # cannot detect. mypy does not compare parameter names, so a pure rename is not caught.
    _point: type[xrf.SupportsPoints] = xrf.AffineTransform
    _affine: type[xrf.SupportsAffine] = xrf.AffineTransform
