"""Public behavior of :meth:`xarrayrf.Geometry.frame_coordinates`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from xarray.indexes import CoordinateTransformIndex, RangeIndex

import xarrayrf as xrf
from xarrayrf._frame_coordinates import FrameCoordinateTransform

ATOL = 1e-12
"""Rounding allowance for products of small exact matrices."""

FRAME = xrf.ReferenceFrame.declared(
    ("test", "patient"), xrf.CoordinateSystem(("x", "y", "z"), ("mm",) * 3)
)


def geometry(
    translation: tuple[float, float, float] = (10.0, 20.0, 30.0),
    *,
    sample_offset: tuple[float, float, float] | None = None,
) -> xrf.Geometry:
    angle = np.radians(30.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]]
    )
    array = xr.DataArray(
        np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5),
        dims=("k", "j", "i"),
        coords={"k": np.arange(3), "j": np.arange(4), "i": np.arange(5)},
    )
    transform = xrf.AffineTransform(
        source=xrf.ArrayCoordinates(("i", "j", "k"), ("1", "1", "1"), sample_offset=sample_offset),
        target=FRAME,
        matrix=rotation @ np.diag([0.5, 0.5, 2.0]),
        translation=translation,
    )
    return xrf.Geometry(transform=transform, array=array, dims=("k", "j", "i"))


def test_frame_coordinates_match_points_and_carry_units() -> None:
    source = geometry()
    framed = source.array.assign_coords(source.frame_coordinates())
    points = source.points()
    for index, name in enumerate(("x", "y", "z")):
        assert framed[name].dims == ("k", "j", "i")
        assert framed[name].attrs["units"] == "mm"
        assert_allclose(framed[name].values, points.values[..., index], rtol=0, atol=ATOL)


def test_frame_coordinates_are_computed_on_access_not_stored() -> None:
    source = geometry()
    framed = source.array.assign_coords(source.frame_coordinates())
    assert isinstance(framed.xindexes["x"], CoordinateTransformIndex)


def test_nearest_selection_by_frame_point() -> None:
    source = geometry()
    framed = source.array.assign_coords(source.frame_coordinates())
    target = source.point_at(k=1, j=1, i=3).values + 0.1  # inside the samples along every dim
    labels = {
        name: xr.DataArray([value], dims="point") for name, value in zip("xyz", target, strict=True)
    }
    selected = framed.sel(labels, method="nearest")
    assert float(selected.values[0]) == float(source.array.isel(k=1, j=1, i=3))


def test_nearest_selection_uses_the_geometry_domain() -> None:
    """A point 0.3 steps past the last sample is outside the samples but inside its cell."""
    source = geometry(sample_offset=(0.5, 0.5, 0.5))
    point = source.lattice().transform_point(np.array([[2.3, 1.0, 3.0]]))[0]
    labels = {
        name: xr.DataArray([value], dims="point") for name, value in zip("xyz", point, strict=True)
    }
    assert np.isnan(source.positions_at(point[None, :], outside="nan")).all()
    with pytest.raises(ValueError, match="outside the samples"):
        source.array.assign_coords(source.frame_coordinates()).sel(labels, method="nearest")
    cells = source.array.assign_coords(source.frame_coordinates(domain="cells"))
    assert float(cells.sel(labels, method="nearest").values[0]) == float(
        source.array.isel(k=2, j=1, i=3)
    )
    assert_allclose(
        source.positions_at(point[None, :], domain="cells"), [[2.3, 1.0, 3.0]], atol=ATOL
    )


def test_slicing_keeps_lazy_frame_coordinates() -> None:
    source = geometry()
    framed = source.array.assign_coords(source.frame_coordinates())
    cropped = framed.isel(i=slice(1, None, 2), k=slice(1, 3))
    assert type(cropped.xindexes["x"]).__name__ == "FrameCoordinateIndex"
    assert_allclose(cropped.x.values, framed.x.values[1:3, :, 1::2], rtol=0, atol=ATOL)
    assert_allclose(cropped.z.values, framed.z.values[1:3, :, 1::2], rtol=0, atol=ATOL)


def test_other_selections_drop_the_index_but_keep_correct_values() -> None:
    source = geometry()
    framed = source.array.assign_coords(source.frame_coordinates())
    gathered = framed.isel(i=[4, 0])
    assert "x" not in gathered.xindexes
    assert_allclose(gathered.x.values, framed.x.values[..., [4, 0]], rtol=0, atol=ATOL)


def test_different_frame_coordinates_do_not_align_silently() -> None:
    first = geometry()
    second = geometry(translation=(11.0, 20.0, 30.0))
    a = first.array.assign_coords(first.frame_coordinates())
    b = second.array.assign_coords(second.frame_coordinates())
    assert_allclose((a + a).values, 2 * a.values)
    with pytest.raises(NotImplementedError):
        _ = a + b


def test_names_avoid_collisions_with_the_array() -> None:
    source = geometry()
    with pytest.raises(ValueError, match="already coordinates or dimensions"):
        source.frame_coordinates(("i", "y_mm", "z_mm"))
    renamed = source.frame_coordinates(("x_mm", "y_mm", "z_mm"))
    assert set(renamed) == {"x_mm", "y_mm", "z_mm"}
    with pytest.raises(ValueError, match="one distinct name per frame axis"):
        source.frame_coordinates(("a", "b"))


def test_irregular_samples_have_frame_coordinates() -> None:
    source = geometry()
    irregular = xrf.Geometry(
        source.array.assign_coords(k=[0.0, 1.0, 5.0]), source.transform, dims=source.dims
    )
    framed = irregular.array.assign_coords(irregular.frame_coordinates())
    for index, name in enumerate("xyz"):
        assert_allclose(
            framed[name].values, irregular.points().values[..., index], rtol=0, atol=ATOL
        )
    point = (
        0.4 * irregular.point_at(k=1, j=1, i=2).values
        + 0.6 * irregular.point_at(k=2, j=1, i=2).values
    )
    nearest = dict(
        zip(irregular.dims, np.rint(irregular.positions_at(point)).astype(int), strict=True)
    )
    assert_allclose(
        framed.sel(_labels(point), method="nearest").values[0],
        irregular.array.isel(nearest).values,
        rtol=0,
        atol=ATOL,
    )


def test_a_plane_in_a_volume_has_coordinates_but_no_reverse_selection() -> None:
    plane = xr.DataArray(np.zeros((2, 3)), dims=("j", "i"), coords={"j": [0, 1], "i": [0, 1, 2]})
    transform = xrf.AffineTransform(
        source=xrf.ArrayCoordinates(("i", "j"), ("1", "1")),
        target=FRAME,
        matrix=[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        translation=(0.0, 0.0, 5.0),
    )
    source = xrf.Geometry(plane, transform, dims=("j", "i"))
    framed = plane.assign_coords(source.frame_coordinates())
    assert_allclose(framed.z.values, 5.0, rtol=0, atol=ATOL)
    labels = {name: xr.DataArray([0.0], dims="point") for name in "xyz"}
    with pytest.raises(ValueError, match="no reverse mapping"):
        framed.sel(labels, method="nearest")


@pytest.mark.parametrize(
    "offset",
    [
        pytest.param((0.0, 0.0, -1.0), id="one-step-before-i"),
        pytest.param((0.0, 0.0, 9.0), id="far"),
    ],
)
def test_nearest_selection_refuses_points_outside_the_samples(
    offset: tuple[float, float, float],
) -> None:
    """Rounding position -1 would otherwise wrap to the last sample."""
    source = geometry()
    framed = source.array.assign_coords(source.frame_coordinates())
    positions = np.array([[2.0, 1.0, 0.0]]) + np.array(offset)
    point = source.lattice().transform_point(positions)[0]
    labels = {
        name: xr.DataArray([value], dims="point") for name, value in zip("xyz", point, strict=True)
    }
    with pytest.raises(ValueError, match="outside the samples"):
        framed.sel(labels, method="nearest")


def test_nearest_selection_refuses_nan() -> None:
    source = geometry()
    framed = source.array.assign_coords(source.frame_coordinates())
    labels = {name: xr.DataArray([np.nan], dims="point") for name in "xyz"}
    with pytest.raises(ValueError, match="not finite"):
        framed.sel(labels, method="nearest")


def test_reversed_and_empty_slices_keep_correct_coordinates() -> None:
    source = geometry()
    framed = source.array.assign_coords(source.frame_coordinates())
    reversed_i = framed.isel(i=slice(None, None, -1))
    assert_allclose(reversed_i.x.values, framed.x.values[..., ::-1], rtol=0, atol=ATOL)
    empty = framed.isel(i=slice(3, 3))
    assert empty.x.shape == (3, 4, 0)


def test_renamed_dimensions_keep_the_frame_coordinates() -> None:
    source = geometry()
    framed = source.array.assign_coords(source.frame_coordinates())
    renamed = framed.rename({"i": "column"}).isel(column=slice(1, 3))
    assert_allclose(renamed.x.values, framed.x.values[..., 1:3], rtol=0, atol=ATOL)


def test_the_same_lattice_in_another_dimension_order_does_not_align() -> None:
    """Swapped dims give different frame coordinates even with equal sizes and matrix."""
    array = xr.DataArray(np.zeros((3, 3)), dims=("a", "b"), coords={"a": [0, 1, 2], "b": [0, 1, 2]})
    transform = xrf.AffineTransform(
        source=xrf.ArrayCoordinates(("a", "b"), ("1", "1")),
        target=xrf.ReferenceFrame.local(xrf.CoordinateSystem(("u", "v"), ("mm", "mm"))),
        matrix=[[1.0, 0.0], [0.0, 2.0]],
        translation=(0.0, 0.0),
    )
    first = xrf.Geometry(array, transform, dims=("a", "b")).frame_coordinates()
    second = xrf.Geometry(array.transpose("b", "a"), transform, dims=("a", "b")).frame_coordinates()
    index_first = first.xindexes["u"]
    index_second = second.xindexes["u"]
    assert not index_first.equals(index_second)


@pytest.mark.parametrize(
    ("names", "error", "message"),
    [
        pytest.param("xyz", TypeError, "sequence of strings", id="bare-string"),
        pytest.param(("x_mm", 2, "z_mm"), TypeError, "entries must be strings", id="non-string"),
        pytest.param(("a", "a", "b"), ValueError, "one distinct name", id="repeated"),
    ],
)
def test_names_are_validated(names: object, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        geometry().frame_coordinates(names)  # type: ignore[arg-type]


def test_a_non_affine_transform_has_no_frame_coordinates() -> None:
    class Points:
        source = xrf.ArrayCoordinates(("i", "j", "k"), ("1", "1", "1"))
        target = FRAME

        def transform_point(self, points: object) -> npt.NDArray[np.float64]:
            return np.asarray(points, dtype=np.float64)

    source = geometry()
    with pytest.raises(TypeError, match="needs an affine transform"):
        xrf.Geometry(source.array, Points(), dims=source.dims).frame_coordinates()


def _labels(point: npt.NDArray[np.float64]) -> dict[str, xr.DataArray]:
    return {name: xr.DataArray([v], dims="point") for name, v in zip("xyz", point, strict=True)}


def test_outer_cells_select_the_edge_sample_even_for_whole_step_reach() -> None:
    """Position -0.75 must not round to -1 and wrap to the last sample."""
    source = geometry(sample_offset=(1.0, 1.0, 1.0))  # cells reach one full step below
    cells = source.array.assign_coords(source.frame_coordinates(domain="cells"))
    point = source.lattice().transform_point(np.array([[1.0, 1.0, -0.75]]))[0]
    assert float(cells.sel(_labels(point), method="nearest").values[0]) == float(
        source.array.isel(k=1, j=1, i=0)
    )
    with pytest.raises(ValueError, match="outside the cells domain"):
        cells.sel(_labels(point - 100.0), method="nearest")


def test_reversed_slice_swaps_the_cell_reach() -> None:
    source = geometry(sample_offset=(0.25, 0.25, 0.25))  # reach (0.25 before, 0.75 after)
    cells = source.array.assign_coords(source.frame_coordinates(domain="cells"))
    reversed_i = cells.isel(i=slice(None, None, -1))
    index = reversed_i.xindexes["x"]
    assert isinstance(index, CoordinateTransformIndex)
    transform = index.transform
    assert isinstance(transform, FrameCoordinateTransform)
    assert transform.reach[transform.dims.index("i")] == (0.75, 0.25)
    # The point positions_at admits on the reversed geometry is admitted by selection too.
    reversed_geometry = xrf.Geometry(reversed_i, source.transform, dims=source.dims)
    point = source.lattice().transform_point(np.array([[1.0, 1.0, 4.6]]))[0]  # 0.6 past i=4
    assert not np.isnan(reversed_geometry.positions_at(point[None, :], domain="cells")).any()
    assert float(reversed_i.sel(_labels(point), method="nearest").values[0]) == float(
        source.array.isel(k=1, j=1, i=4)
    )


def test_nearest_selection_uses_actual_coordinates_near_a_midpoint() -> None:
    source = geometry()
    source = xrf.Geometry(
        source.array.assign_coords(k=[0.0, 1.00000075, 2.0]), source.transform, dims=source.dims
    )
    framed = source.array.assign_coords(source.frame_coordinates())
    point = source.transform.transform_point(np.array([2.0, 1.0, 0.5000002]))
    positions = source.positions_at(point)
    assert round(float(positions[0])) == 0
    nearest = dict(zip(source.dims, np.rint(positions).astype(int), strict=True))
    assert_allclose(
        framed.sel(_labels(point), method="nearest").values[0],
        source.array.isel(nearest).values,
        rtol=0,
        atol=ATOL,
    )


def test_non_monotonic_coordinates_are_refused_at_construction() -> None:
    source = geometry()
    source = xrf.Geometry(
        source.array.assign_coords(k=[0.0, 2.0, 1.0]), source.transform, dims=source.dims
    )
    with pytest.raises(ValueError, match="not strictly monotonic"):
        source.frame_coordinates()


def test_single_sample_dimension_supports_forward_and_reverse_mapping() -> None:
    source = geometry()
    source = xrf.Geometry(source.array.isel(k=slice(1, 2)), source.transform, dims=source.dims)
    framed = source.array.assign_coords(source.frame_coordinates())
    for index, name in enumerate("xyz"):
        assert_allclose(framed[name].values, source.points().values[..., index], rtol=0, atol=ATOL)
    point = source.point_at(k=0, j=1, i=2).values
    assert_allclose(
        framed.sel(_labels(point), method="nearest").values[0],
        source.array.isel(k=0, j=1, i=2).values,
        rtol=0,
        atol=ATOL,
    )
    with pytest.raises(ValueError, match="outside the samples"):
        framed.sel(_labels(point + np.array([0.0, 0.0, 0.1])), method="nearest")


def test_retained_scalar_supports_forward_but_no_reverse_mapping() -> None:
    source = geometry()
    source = xrf.Geometry(source.array.isel(k=1), source.transform, dims=("j", "i"))
    framed = source.array.assign_coords(source.frame_coordinates())
    for index, name in enumerate("xyz"):
        assert_allclose(framed[name].values, source.points().values[..., index], rtol=0, atol=ATOL)
    with pytest.raises(ValueError, match="no reverse mapping"):
        framed.sel(_labels(source.point_at(j=1, i=2).values), method="nearest")


def test_fractional_forward_mapping_interpolates_actual_coordinates() -> None:
    source = geometry()
    source = xrf.Geometry(
        source.array.assign_coords(k=[0.0, 1.0, 5.0]), source.transform, dims=source.dims
    )
    index = source.frame_coordinates().xindexes["x"]
    assert isinstance(index, CoordinateTransformIndex)
    actual = index.transform.forward(
        {"k": np.array([1.25]), "j": np.array([1]), "i": np.array([2])}
    )
    expected = source.transform.transform_point(np.array([[2.0, 1.0, 2.0]]))
    assert_allclose(
        np.stack([actual[name] for name in "xyz"], axis=-1), expected, rtol=0, atol=ATOL
    )


def test_nonuniform_outer_cell_uses_the_actual_edge_step() -> None:
    source = geometry(sample_offset=(0.5, 0.5, 0.5))
    source = xrf.Geometry(
        source.array.assign_coords(k=[0.0, 1.00000075, 2.0]), source.transform, dims=source.dims
    )
    framed = source.array.assign_coords(source.frame_coordinates(domain="cells"))
    point = source.transform.transform_point(np.array([2.0, 1.0, -0.5000002]))
    assert np.isfinite(source.positions_at(point, domain="cells")).all()
    assert_allclose(
        framed.sel(_labels(point), method="nearest").values[0],
        source.array.isel(k=0, j=1, i=2).values,
        rtol=0,
        atol=ATOL,
    )


def test_nonuniform_reversed_strided_slice_preserves_selection() -> None:
    source = geometry(sample_offset=(0.25, 0.25, 0.25))
    source = xrf.Geometry(
        source.array.assign_coords(i=[0.0, 1.0, 3.0, 8.0, 12.0]), source.transform, dims=source.dims
    )
    framed = source.array.assign_coords(source.frame_coordinates(domain="cells"))
    sliced = framed.rename({"i": "column"}).isel(column=slice(None, None, -2))
    sampled = xrf.Geometry(
        source.array.isel(i=slice(None, None, -2)), source.transform, dims=source.dims
    )
    point = source.transform.transform_point(np.array([13.0, 1.0, 1.0]))
    assert np.isfinite(sampled.positions_at(point, domain="cells")).all()
    assert_allclose(
        sliced.sel(_labels(point), method="nearest").values[0],
        source.array.isel(k=1, j=1, i=4).values,
        rtol=0,
        atol=ATOL,
    )
    for index, name in enumerate("xyz"):
        assert_allclose(sliced[name].values, sampled.points().values[..., index], rtol=0, atol=ATOL)


def test_equal_frame_coordinates_require_the_same_domain_and_actual_values() -> None:
    source = geometry()
    original = source.frame_coordinates().xindexes["x"]
    assert not original.equals(source.frame_coordinates(domain="cells").xindexes["x"])
    shifted = xrf.Geometry(
        source.array.assign_coords(k=[0.0, 1.00000075, 2.0]), source.transform, dims=source.dims
    )
    assert not original.equals(shifted.frame_coordinates().xindexes["x"])


def test_frame_coordinates_snapshot_source_values() -> None:
    source = geometry()
    array = xr.DataArray(
        source.array.data,
        dims=source.array.dims,
        coords=xr.Coordinates(
            {"k": [0.0, 1.0, 5.0], "j": np.arange(4), "i": np.arange(5)}, indexes={}
        ),
    )
    source = xrf.Geometry(array, source.transform, dims=source.dims)
    coordinates = source.frame_coordinates()
    expected = source.points().values.copy()
    array.coords["k"].values[1] = 2.0
    for index, name in enumerate("xyz"):
        assert_allclose(coordinates[name].values, expected[..., index], rtol=0, atol=ATOL)


def test_range_index_stride_updates_reverse_spacing() -> None:
    source = geometry()
    coordinates = xr.Coordinates.from_xindex(RangeIndex.arange(5, dim="i"))
    source = xrf.Geometry(
        source.array.drop_indexes("i").assign_coords(coordinates),
        source.transform,
        dims=source.dims,
    )
    sliced = source.array.assign_coords(source.frame_coordinates()).isel(i=slice(None, None, -2))
    point = source.point_at(k=1, j=1, i=2).values
    assert_allclose(
        sliced.sel(_labels(point), method="nearest").values[0],
        source.array.isel(k=1, j=1, i=2).values,
        rtol=0,
        atol=ATOL,
    )
