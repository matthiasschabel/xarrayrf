"""Public behavior of :meth:`xarrayrf.Geometry.lattice` and :class:`xarrayrf.Lattice`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import xarrayrf as xrf

ATOL = 1e-12
"""Rounding allowance for products of small exact matrices."""

FRAME = xrf.ReferenceFrame.declared(
    ("test", "patient"), xrf.CoordinateSystem(("x", "y", "z"), ("mm",) * 3)
)


def geometry(
    k: npt.ArrayLike = (0.0, 1.0, 2.0),
    matrix: npt.ArrayLike = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    translation: npt.ArrayLike = (10.0, 20.0, 30.0),
) -> xrf.Geometry:
    k_values = np.asarray(k, dtype=np.float64)
    array = xr.DataArray(
        np.zeros((k_values.size, 4, 5)),
        dims=("k", "j", "i"),
        coords={"k": k_values, "j": np.arange(4), "i": np.arange(5)},
    )
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("i", "j", "k"), ("1", "1", "1")),
        target=FRAME,
        matrix=matrix,
        translation=translation,
    )
    return xrf.Geometry(array, transform, dims=("k", "j", "i"))


def test_a_uniform_array_forms_a_lattice_in_array_order() -> None:
    lattice = geometry(k=(4.0, 6.0, 8.0), matrix=np.diag([0.5, 0.25, 3.0])).lattice()
    assert lattice.dims == ("k", "j", "i")
    assert lattice.frame is FRAME
    # Position 0 has k = 4, so z = 30 + 3 * 4; one k step is 2 coordinate units = 6 mm.
    assert_allclose(lattice.origin, [10.0, 20.0, 42.0], atol=ATOL)
    assert_allclose(lattice.matrix, [[0, 0, 0.5], [0, 0.25, 0], [6.0, 0, 0]], atol=ATOL)
    assert_allclose(lattice.spacing, [6.0, 0.25, 0.5], atol=ATOL)
    assert_allclose(lattice.direction, [[0, 0, 1], [0, 1, 0], [1, 0, 0]], atol=ATOL)


def test_dims_choose_the_column_order_like_itk_and_nifti() -> None:
    lattice = geometry(matrix=np.diag([0.5, 0.25, 3.0])).lattice(("i", "j", "k"))
    expected = np.array([[0.5, 0, 0, 10.0], [0, 0.25, 0, 20.0], [0, 0, 3.0, 30.0], [0, 0, 0, 1.0]])
    assert_allclose(lattice.affine, expected, atol=ATOL)
    assert_allclose(lattice.transform_point([2.0, 1.0, 1.0]), [11.0, 20.25, 33.0], atol=ATOL)


def test_lattice_points_agree_with_geometry_points() -> None:
    angle = np.radians(25.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]]
    )
    source = geometry(k=(1.0, 3.0, 5.0), matrix=rotation @ np.diag([0.5, 0.5, 2.0]))
    lattice = source.lattice()
    positions = np.moveaxis(np.indices((3, 4, 5), dtype=np.float64), 0, -1)
    assert_allclose(lattice.transform_point(positions), source.points().values, atol=ATOL)


def test_the_lattice_does_not_share_its_arrays() -> None:
    lattice = geometry().lattice()
    lattice.origin[0] = 99.0
    lattice.matrix[0, 0] = 99.0
    assert lattice.origin[0] == pytest.approx(10.0)
    assert lattice.matrix[0, 0] == pytest.approx(0.0)
    assert "Lattice(" in repr(lattice)


@pytest.mark.parametrize(
    ("k", "message"),
    [
        pytest.param((0.0, 1.0, 3.0), "not uniformly spaced", id="nonuniform"),
        pytest.param((0.0,), "single sample", id="single"),
    ],
)
def test_irregular_samples_form_no_lattice(k: tuple[float, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        geometry(k=k).lattice()


def test_a_looser_tolerance_admits_scanner_jitter() -> None:
    jittered = geometry(k=(0.0, 1.0 + 1e-4, 2.0))
    with pytest.raises(ValueError, match="not uniformly spaced"):
        jittered.lattice()
    assert jittered.lattice(tolerance=1e-3).spacing[0] == pytest.approx(1.0)


def test_dims_must_name_every_geometry_dimension_once() -> None:
    with pytest.raises(ValueError, match="must name each geometry dimension"):
        geometry().lattice(("k", "j"))
    with pytest.raises(ValueError, match="must name each geometry dimension"):
        geometry().lattice(("k", "k", "j"))


def test_a_dimension_mapping_to_no_displacement_is_refused() -> None:
    with pytest.raises(ValueError, match="map to no displacement"):
        geometry(matrix=np.diag([1.0, 1.0, 0.0])).lattice()


def test_a_non_affine_transform_forms_no_lattice() -> None:
    class Points:
        source = xrf.ArrayCoordinates(("i", "j", "k"), ("1", "1", "1"))
        target = FRAME

        def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return np.asarray(points, dtype=np.float64)

    source = geometry()
    with pytest.raises(TypeError, match="needs an affine transform"):
        xrf.Geometry(source.array, Points(), dims=("k", "j", "i")).lattice()


def test_retained_scalar_coordinates_join_the_origin() -> None:
    plane = geometry(k=(0.0, 2.0, 4.0), matrix=np.diag([1.0, 1.0, 3.0]))
    selected = xrf.Geometry(plane.array.isel(k=2), plane.transform, dims=("j", "i"))
    lattice = selected.lattice()
    assert lattice.dims == ("j", "i")
    assert_allclose(lattice.origin, [10.0, 20.0, 42.0], atol=ATOL)
    assert lattice.matrix.shape == (3, 2)


def test_a_coordinate_field_forms_no_lattice() -> None:
    array = xr.DataArray(
        np.zeros((2, 2)),
        dims=("j", "i"),
        coords={"u": (("j", "i"), [[0.0, 1.0], [2.0, 3.0]]), "i": [0, 1]},
    )
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("u", "i"), ("1", "1")),
        target=xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("mm", "mm"))),
        matrix=np.eye(2),
        translation=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="multidimensional coordinate field"):
        xrf.Geometry(array, transform, dims=("j", "i")).lattice()


def test_lattice_positions_need_one_value_per_dim() -> None:
    with pytest.raises(ValueError, match="trailing axis of length 3"):
        geometry().lattice().transform_point([1.0, 2.0])


@pytest.mark.parametrize(
    "units",
    [
        pytest.param(("s", "m", "m"), id="time-and-space"),
        pytest.param(("1/mm", "1/mm", "rad/s"), id="k-and-omega"),
    ],
)
def test_spacing_needs_one_unit_across_the_frame(units: tuple[str, str, str]) -> None:
    frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("a", "b", "c"), units))
    array = xr.DataArray(
        np.zeros((2, 2, 2)), dims=("k", "j", "i"), coords={"k": [0, 1], "j": [0, 1], "i": [0, 1]}
    )
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("i", "j", "k"), ("1", "1", "1")),
        target=frame,
        matrix=np.eye(3),
        translation=(0.0, 0.0, 0.0),
    )
    lattice = xrf.Geometry(array, transform, dims=("k", "j", "i")).lattice()
    assert lattice.matrix.shape == (3, 3)
    with pytest.raises(ValueError, match="use different units"):
        _ = lattice.spacing
    with pytest.raises(ValueError, match="use different units"):
        _ = lattice.direction


def test_reciprocal_space_has_a_spacing_in_its_common_unit() -> None:
    frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("kx", "ky"), ("1/mm", "1/mm")))
    array = xr.DataArray(np.zeros((3, 3)), dims=("v", "u"), coords={"v": [0, 1, 2], "u": [0, 1, 2]})
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("u", "v"), ("1", "1")),
        target=frame,
        matrix=np.diag([0.25, 0.5]),
        translation=(-0.25, -0.5),
    )
    lattice = xrf.Geometry(array, transform, dims=("v", "u")).lattice()
    assert_allclose(lattice.spacing, [0.5, 0.25], atol=ATOL)


def test_range_index_coordinates_give_their_exact_step() -> None:
    """A RangeIndex step of 0.1 is used as declared, not recomputed end to end from floats."""
    from xarray.indexes import RangeIndex

    index = RangeIndex.arange(0.0, 3.0, 0.1, coord_name="i", dim="i")
    array = xr.DataArray(np.zeros(30), dims="i", coords=xr.Coordinates.from_xindex(index))
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("i",), ("mm",)),
        target=xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x",), ("mm",))),
        matrix=((1.0,),),
        translation=(0.0,),
    )
    geometry = xrf.Geometry(array, transform, dims=("i",))
    assert geometry.lattice().matrix[0, 0] == index.step
    cropped = xrf.Geometry(array.isel(i=slice(3, None, 4)), transform, dims=("i",))
    cropped_index = cropped.array.xindexes["i"]
    assert isinstance(cropped_index, RangeIndex)
    assert cropped.lattice().matrix[0, 0] == cropped_index.step
    assert_allclose(cropped.positions_at([[0.7]]), [[1.0]], rtol=0, atol=ATOL)


@pytest.mark.parametrize("units", [(None, None), ("mm", None)], ids=["all-undeclared", "mixed"])
def test_spacing_and_direction_need_one_declared_unit(units: tuple[str | None, str | None]) -> None:
    frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("a", "b"), units))
    lattice = xrf.Lattice(frame=frame, dims=("i", "j"), origin=np.zeros(2), matrix=np.eye(2))
    for name in ("spacing", "direction"):
        with pytest.raises(ValueError, match="declare no unit"):
            getattr(lattice, name)


def test_lattice_validates_its_declaration_and_compares_by_value() -> None:
    frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("a", "b", "c"), ("mm",) * 3))
    with pytest.raises(TypeError, match="frame must be a ReferenceFrame"):
        xrf.Lattice(frame="patient", dims=("i",), origin=np.zeros(3), matrix=np.ones((3, 1)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"matrix must have shape \(3, 1\)"):
        xrf.Lattice(frame=frame, dims=("i",), origin=np.zeros(3), matrix=np.ones((5, 7)))
    with pytest.raises(ValueError, match=r"origin must have shape \(3,\)"):
        xrf.Lattice(frame=frame, dims=("i",), origin=np.zeros(2), matrix=np.ones((3, 1)))
    with pytest.raises(ValueError, match="finite"):
        xrf.Lattice(frame=frame, dims=("i",), origin=[np.nan, 0, 0], matrix=np.ones((3, 1)))
    with pytest.raises(ValueError, match="dims"):
        xrf.Lattice(frame=frame, dims=("i", "i"), origin=np.zeros(3), matrix=np.ones((3, 2)))
    first = xrf.Lattice(frame=frame, dims=("i",), origin=np.zeros(3), matrix=np.ones((3, 1)))
    same = xrf.Lattice(frame=frame, dims=("i",), origin=[0, 0, 0], matrix=[[1.0], [1.0], [1.0]])
    other = xrf.Lattice(frame=frame, dims=("i",), origin=np.ones(3), matrix=np.ones((3, 1)))
    assert first == same and hash(first) == hash(same)
    assert first != other
    assert first != "lattice"


def test_signed_zero_does_not_split_equal_lattices() -> None:
    frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("a",), ("mm",)))
    plus = xrf.Lattice(frame=frame, dims=("i",), origin=[0.0], matrix=[[1.0]])
    minus = xrf.Lattice(frame=frame, dims=("i",), origin=[-0.0], matrix=[[1.0]])
    assert plus == minus and hash(plus) == hash(minus)


def test_a_fully_selected_point_forms_a_zero_dimensional_lattice() -> None:
    frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("a",), ("mm",)))
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("i",), ("1",)),
        target=frame,
        matrix=[[2.0]],
        translation=[1.0],
    )
    array = xr.DataArray(np.zeros(3), dims="i", coords={"i": ("i", np.arange(3), {"units": "1"})})
    lattice = xrf.Geometry(array.isel(i=2), transform, dims=()).lattice()
    assert lattice.dims == ()
    assert lattice.matrix.shape == (1, 0)
    assert_allclose(lattice.origin, [5.0], rtol=0, atol=ATOL)
    assert_allclose(lattice.transform_point(np.zeros((4, 0))), np.full((4, 1), 5.0), atol=ATOL)
