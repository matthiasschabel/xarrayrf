"""Public behavior of :class:`xarrayrf.Geometry`.

Every test goes through the exported public API with a standalone ``Geometry``.
Native accessor and binding behavior is covered in ``test_native.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from dask.callbacks import Callback
from numpy.testing import assert_allclose, assert_array_equal

from xarrayrf import AffineTransform, ArrayCoordinates, CoordinateSystem, Geometry, ReferenceFrame

ATOL = 1e-12
"""Floating-point arithmetic tolerance for these small synthetic affines.

A rounding allowance for a handful of multiply-add operations, not a physical equivalence
tolerance; the reference values below are exact rationals computed by hand.
"""


@pytest.fixture
def patient_frame() -> ReferenceFrame:
    """A minted 3-D Cartesian frame in millimetres."""
    return ReferenceFrame.local(CoordinateSystem(("L", "P", "S"), ("mm", "mm", "mm")))


@pytest.fixture
def volume_transform(patient_frame: ReferenceFrame) -> AffineTransform:
    """L = 10 + 0.5*column, P = 20 + 0.25*row, S = -5 + slice_offset."""
    return AffineTransform.from_matrix(
        target=patient_frame,
        source=ArrayCoordinates(("slice_offset", "row", "column"), ("mm", "1", "1")),
        matrix=((0.0, 0.0, 0.5), (0.0, 0.25, 0.0), (1.0, 0.0, 0.0)),
        translation=(10.0, 20.0, -5.0),
    )


def _volume(offsets: npt.ArrayLike = (0.0, 2.0, 5.0)) -> xr.DataArray:
    """A 3-by-4-by-5 stack whose slices sit at nonuniform physical offsets."""
    return xr.DataArray(
        np.zeros((3, 4, 5)),
        dims=("slice", "row", "column"),
        coords={
            "slice": np.arange(3),
            "slice_offset": ("slice", np.asarray(offsets, dtype=float), {"units": "mm"}),
            "row": ("row", np.arange(4), {"units": "1"}),
            "column": ("column", np.arange(5), {"units": "1"}),
        },
    )


@pytest.fixture
def volume() -> xr.DataArray:
    return _volume()


@contextmanager
def _recorded_tasks() -> Iterator[list[object]]:
    """Record every Dask task key executed inside the block."""
    tasks: list[object] = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        yield tasks


def test_nonuniform_offsets_are_read_as_given(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    """The third slice sits at its declared 5 mm offset; no spacing is inferred from 0, 2."""
    geometry = Geometry(volume, volume_transform, dims=("slice", "row", "column"))
    point = geometry.point_at(slice=2, row=1, column=3)
    assert tuple(point.axis.values) == ("L", "P", "S")
    assert_allclose(point.sel(axis="L"), 11.5, rtol=0, atol=ATOL)
    assert_allclose(point.sel(axis="P"), 20.25, rtol=0, atol=ATOL)
    assert_allclose(point.sel(axis="S"), 0.0, rtol=0, atol=ATOL)
    # The second slice is at 2 mm, so an inferred 2.5 mm spacing would show up here.
    assert_allclose(geometry.point_at(slice=1, row=3, column=4).sel(axis="S"), -3.0, atol=ATOL)


def test_declared_properties_are_derived_afresh_and_read_only(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    geometry = Geometry(volume, volume_transform, dims=("slice", "row", "column"))
    assert geometry.transform is volume_transform
    assert geometry.dims == ("slice", "row", "column")
    assert dict(geometry.sizes) == {"slice": 3, "row": 4, "column": 5}
    assert dict(geometry.coordinate_dependencies) == {
        "slice_offset": ("slice",),
        "row": ("row",),
        "column": ("column",),
    }
    assert geometry.sizes is not geometry.sizes
    with pytest.raises(TypeError):
        geometry.sizes["row"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        geometry.coordinate_dependencies["row"] = ()  # type: ignore[index]


def test_embedded_plane_maps_two_source_axes_into_three_axes() -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("L", "P", "S"), ("mm", "mm", "mm")))
    transform = AffineTransform.from_matrix(
        target=frame,
        source=ArrayCoordinates(("row", "column"), ("1", "1")),
        matrix=((0.6, 0.0), (0.8, 0.0), (0.0, -1.0)),
        translation=(1.0, 2.0, 3.0),
    )
    image = xr.DataArray(
        np.zeros((3, 5)),
        dims=("row", "column"),
        coords={"row": np.arange(3), "column": np.arange(5)},
    )
    geometry = Geometry(image, transform, dims=("row", "column"))
    point = geometry.point_at(row=2, column=4)
    # L = 1 + 0.6*2 = 2.2; P = 2 + 0.8*2 = 3.6; S = 3 - 1*4 = -1. No third axis is invented.
    assert_allclose(point.sel(axis="L"), 2.2, rtol=0, atol=ATOL)
    assert_allclose(point.sel(axis="P"), 3.6, rtol=0, atol=ATOL)
    assert_allclose(point.sel(axis="S"), -1.0, rtol=0, atol=ATOL)


def test_one_based_labels_are_absorbed_by_the_affine_offset() -> None:
    """Position 0 carries label 1; the transform's translation, not the view, handles the origin."""
    frame = ReferenceFrame.local(CoordinateSystem(("x", "y"), ("mm", "mm")))
    transform = AffineTransform.from_matrix(
        target=frame,
        source=ArrayCoordinates(("row", "column"), ("1", "1")),
        matrix=((2.0, 0.0), (0.0, 3.0)),
        translation=(-2.0, -3.0),
    )
    image = xr.DataArray(
        np.zeros((3, 4)),
        dims=("row", "column"),
        coords={"row": np.arange(1, 4), "column": np.arange(1, 5)},
    )
    geometry = Geometry(image, transform, dims=("row", "column"))
    point = geometry.point_at(row=1, column=2)
    # Labels are 2 and 3: x = 2*2 - 2 = 2; y = 3*3 - 3 = 6.
    assert_allclose(point.sel(axis="x"), 2.0, rtol=0, atol=ATOL)
    assert_allclose(point.sel(axis="y"), 6.0, rtol=0, atol=ATOL)
    # Position 0 is label 1, which the offset places at the origin.
    assert_allclose(geometry.point_at(row=0, column=0).sel(axis="x"), 0.0, rtol=0, atol=ATOL)


def test_a_selected_plane_keeps_its_fixed_offset_as_a_scalar_source_axis(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    plane = volume.isel(slice=1)
    geometry = Geometry(plane, volume_transform, dims=("row", "column"))
    assert geometry.coordinate_dependencies["slice_offset"] == ()
    point = geometry.point_at(row=1, column=3)
    assert_allclose(point.sel(axis="S"), -3.0, rtol=0, atol=ATOL)
    assert_allclose(point.sel(axis="L"), 11.5, rtol=0, atol=ATOL)
    assert_allclose(point.sel(axis="P"), 20.25, rtol=0, atol=ATOL)
    with pytest.raises(ValueError, match="unexpected positions"):
        geometry.point_at(slice=1, row=1, column=3)


def test_a_fully_selected_point_has_no_geometry_dimensions(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    point = volume.isel(slice=1, row=0, column=2)
    geometry = Geometry(point, volume_transform, dims=())
    assert dict(geometry.sizes) == {}
    assert set(geometry.coordinate_dependencies.values()) == {()}
    point = geometry.point_at()
    assert_allclose(point.sel(axis="L"), 11.0, rtol=0, atol=ATOL)
    assert_allclose(point.sel(axis="P"), 20.0, rtol=0, atol=ATOL)
    assert_allclose(point.sel(axis="S"), -3.0, rtol=0, atol=ATOL)


def test_dimension_order_does_not_change_placement(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    """Storage order and the declared order of dims are both immaterial."""
    reordered = volume.transpose("column", "slice", "row")
    geometry = Geometry(reordered, volume_transform, dims=("row", "column", "slice"))
    point = geometry.point_at(column=3, slice=2, row=1)
    assert_allclose(point.sel(axis="L"), 11.5, rtol=0, atol=ATOL)
    assert_allclose(point.sel(axis="P"), 20.25, rtol=0, atol=ATOL)
    assert_allclose(point.sel(axis="S"), 0.0, rtol=0, atol=ATOL)


def test_a_multidimensional_coordinate_field_is_read_at_the_requested_sample() -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("x",), ("mm",)))
    transform = AffineTransform.from_matrix(
        target=frame,
        source=ArrayCoordinates(("u", "v"), ("mm", "mm")),
        matrix=((1.0, 2.0),),
        translation=(0.5,),
    )
    field = np.array([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]])
    image = xr.DataArray(
        np.zeros((3, 4)),
        dims=("j", "i"),
        coords={"u": (("j", "i"), field), "v": ("i", np.array([0.0, 0.5, 1.0, 1.5]))},
    )
    geometry = Geometry(image, transform, dims=("j", "i"))
    assert geometry.coordinate_dependencies == {"u": ("j", "i"), "v": ("i",)}
    # u(1,2) = 12, v(2) = 1.0; x = 0.5 + 12 + 2*1 = 14.5.
    assert_allclose(geometry.point_at(j=1, i=2).sel(axis="x"), 14.5, rtol=0, atol=ATOL)


def test_a_non_geometry_dimension_is_ignored_rather_than_indexed(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    """An echo axis changes the array's rank, not where its samples sit."""
    stack = volume.expand_dims(echo=2)
    geometry = Geometry(stack, volume_transform, dims=("slice", "row", "column"))
    assert "echo" not in geometry.sizes
    assert_allclose(
        geometry.point_at(slice=2, row=1, column=3).sel(axis="L"), 11.5, rtol=0, atol=ATOL
    )


def test_construction_rejects_wrong_parameter_types(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    with pytest.raises(TypeError, match=r"array must be an xarray\.DataArray"):
        Geometry(volume.values, volume_transform, dims=("slice",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="transform must implement SupportsPoints"):
        Geometry(volume, volume_transform.target, dims=("slice",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dims must be a sequence"):
        Geometry(volume, volume_transform, dims="slice")


def test_construction_rejects_bad_geometry_dimension_sets(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    with pytest.raises(ValueError, match="must be unique"):
        Geometry(volume, volume_transform, dims=("slice", "row", "column", "row"))
    with pytest.raises(ValueError, match="not dimensions of the array"):
        Geometry(volume, volume_transform, dims=("slice", "row", "column", "echo"))
    with pytest.raises(ValueError, match="outside the declared geometry dimensions"):
        Geometry(volume, volume_transform, dims=("row", "column"))
    with pytest.raises(ValueError, match="carry no source axis"):
        Geometry(
            volume.expand_dims(echo=2),
            volume_transform,
            dims=("slice", "row", "column", "echo"),
        )


def test_construction_requires_every_mapping_source_axis_as_a_coordinate(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    without_offsets = volume.drop_vars("slice_offset")
    with pytest.raises(ValueError, match="'slice_offset' is not a coordinate"):
        Geometry(without_offsets, volume_transform, dims=("slice", "row", "column"))


def test_a_declared_coordinate_unit_must_match_the_transform(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    offsets = volume.slice_offset.values
    wrong = volume.assign_coords(slice_offset=("slice", offsets, {"units": "m"}))
    with pytest.raises(ValueError, match=r"declares attrs\['units'\] = 'm'"):
        Geometry(wrong, volume_transform, dims=("slice", "row", "column"))
    non_string = volume.assign_coords(slice_offset=("slice", offsets, {"units": 1}))
    with pytest.raises(TypeError, match=r"attrs\['units'\] of type int"):
        Geometry(non_string, volume_transform, dims=("slice", "row", "column"))


def test_an_absent_unit_attribute_leaves_the_transform_declaration_in_charge(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    bare = volume.assign_coords(slice_offset=("slice", volume.slice_offset.values))
    assert "units" not in bare.slice_offset.attrs
    geometry = Geometry(bare, volume_transform, dims=("slice", "row", "column"))
    assert_allclose(geometry.point_at(slice=2, row=1, column=3).sel(axis="S"), 0.0, atol=ATOL)


@pytest.mark.parametrize(
    "values",
    [
        np.array([True, False, True]),
        np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j]),
        np.array(["a", "b", "c"]),
        np.array([object(), object(), object()], dtype=object),
        np.array(["2026-01-01"] * 3, dtype="datetime64[D]"),
    ],
    ids=["bool", "complex", "string", "object", "datetime"],
)
def test_non_real_coordinate_dtypes_are_refused(
    values: npt.NDArray[Any], volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    labelled = volume.assign_coords(slice_offset=("slice", values))
    with pytest.raises(TypeError, match="must hold real integer or floating values"):
        Geometry(labelled, volume_transform, dims=("slice", "row", "column"))


def test_a_query_requires_exactly_the_varying_geometry_dimensions(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    geometry = Geometry(volume, volume_transform, dims=("slice", "row", "column"))
    with pytest.raises(ValueError, match="missing positions"):
        geometry.point_at(slice=1, row=0)
    with pytest.raises(ValueError, match="unexpected positions"):
        geometry.point_at(slice=1, row=0, column=0, echo=0)


@pytest.mark.parametrize(
    "position",
    [1.0, "1", True, np.float64(1.0), None],
    ids=["float", "string", "bool", "numpy float", "none"],
)
def test_a_position_must_be_a_non_boolean_integer(
    position: object, volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    geometry = Geometry(volume, volume_transform, dims=("slice", "row", "column"))
    indexers: dict[str, Any] = {"slice": position, "row": 0, "column": 0}
    with pytest.raises(TypeError, match="must be a Python or NumPy integer"):
        geometry.point_at(**indexers)


def test_numpy_integers_are_accepted(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    geometry = Geometry(volume, volume_transform, dims=("slice", "row", "column"))
    indexers: dict[str, Any] = {
        "slice": np.int64(2),
        "row": np.uint8(1),
        "column": np.int32(3),
    }
    assert_allclose(geometry.point_at(**indexers).sel(axis="S"), 0.0, rtol=0, atol=ATOL)


@pytest.mark.parametrize("position", [-1, 3, 100])
def test_positions_are_zero_based_and_bounded_by_the_current_size(
    position: int, volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    geometry = Geometry(volume, volume_transform, dims=("slice", "row", "column"))
    with pytest.raises(IndexError, match="out of range for dimension 'slice'"):
        geometry.point_at(slice=position, row=0, column=0)


def test_only_the_selected_value_has_to_be_finite(volume_transform: AffineTransform) -> None:
    """Construction does no whole-field scan, so a NaN elsewhere is found at the query."""
    ragged = _volume(offsets=(0.0, np.nan, 5.0))
    geometry = Geometry(ragged, volume_transform, dims=("slice", "row", "column"))
    assert_allclose(geometry.point_at(slice=2, row=1, column=3).sel(axis="S"), 0.0, atol=ATOL)
    with pytest.raises(ValueError, match="must contain only finite values"):
        geometry.point_at(slice=1, row=1, column=3)


def test_the_view_follows_the_caller_s_array_rather_than_a_snapshot(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    """Coordinates and unit attributes are re-read; validity is never cached."""
    geometry = Geometry(volume, volume_transform, dims=("slice", "row", "column"))
    assert_allclose(geometry.point_at(slice=2, row=1, column=3).sel(axis="S"), 0.0, atol=ATOL)

    volume.coords["slice_offset"] = ("slice", np.array([0.0, 2.0, 9.0]))
    assert_allclose(geometry.point_at(slice=2, row=1, column=3).sel(axis="S"), 4.0, atol=ATOL)

    volume.coords["slice_offset"].attrs["units"] = "m"
    with pytest.raises(ValueError, match=r"declares attrs\['units'\] = 'm'"):
        geometry.point_at(slice=2, row=1, column=3)
    with pytest.raises(ValueError, match=r"declares attrs\['units'\] = 'm'"):
        _ = geometry.sizes

    del volume.coords["slice_offset"]
    with pytest.raises(ValueError, match="'slice_offset' is not a coordinate"):
        _ = geometry.coordinate_dependencies


def test_chunked_pixels_are_never_computed(volume_transform: AffineTransform) -> None:
    eager = _volume()
    lazy = eager.copy(data=da.from_array(eager.values, chunks=(1, 2, 2)))  # type: ignore[no-untyped-call]
    with _recorded_tasks() as tasks:
        geometry = Geometry(lazy, volume_transform, dims=("slice", "row", "column"))
        sizes = dict(geometry.sizes)
        dependencies = dict(geometry.coordinate_dependencies)
        point = geometry.point_at(slice=2, row=1, column=3)
    # Nothing executed at all, so in particular no pixel task did.
    assert tasks == []
    assert sizes == {"slice": 3, "row": 4, "column": 5}
    assert dependencies["slice_offset"] == ("slice",)
    assert_allclose(point.sel(axis="S"), 0.0, rtol=0, atol=ATOL)
    assert isinstance(lazy.data, da.Array)


def test_a_lazy_coordinate_query_computes_only_the_requested_chunks(
    volume_transform: AffineTransform,
) -> None:
    """64 single-sample offset chunks; one point query must not walk the field."""
    offsets = np.arange(64, dtype=float) * 2.0
    image = xr.DataArray(
        da.from_array(np.zeros((64, 4, 5)), chunks=(1, 4, 5)),  # type: ignore[no-untyped-call]
        dims=("slice", "row", "column"),
        coords={
            "slice_offset": ("slice", da.from_array(offsets, chunks=1)),  # type: ignore[no-untyped-call]
            "row": ("row", np.arange(4)),
            "column": ("column", np.arange(5)),
        },
    )
    geometry = Geometry(image, volume_transform, dims=("slice", "row", "column"))
    with _recorded_tasks() as tasks:
        point = geometry.point_at(slice=40, row=1, column=3)
    # S = -5 + 80; the requested chunk is read and the other 63 are not.
    assert_allclose(point.sel(axis="S"), 75.0, rtol=0, atol=ATOL)
    assert_allclose(point.sel(axis="L"), 11.5, rtol=0, atol=ATOL)
    assert 0 < len(tasks) < 10
    assert isinstance(image.data, da.Array)


def test_repr_names_the_target_and_geometry_dimensions(
    volume: xr.DataArray, volume_transform: AffineTransform, patient_frame: ReferenceFrame
) -> None:
    text = repr(Geometry(volume, volume_transform, dims=("slice", "row", "column")))
    assert text.startswith("Geometry(")
    assert "slice_offset" in text
    assert repr(patient_frame.identifier) in text
    assert "dims=('slice', 'row', 'column')" in text


def test_a_dimension_named_self_can_be_queried() -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("x",), ("mm",)))
    transform = AffineTransform.from_matrix(
        target=frame,
        source=ArrayCoordinates(("self",), ("1",)),
        matrix=((2.0,),),
        translation=(10.0,),
    )
    array = xr.DataArray(np.zeros(2), dims=("self",), coords={"self": [0, 1]})
    geometry = Geometry(array, transform, dims=("self",))
    assert_allclose(geometry.point_at(self=1).sel(axis="x"), 12.0, rtol=0, atol=ATOL)


def test_geometry_dimension_entries_must_be_strings(
    volume: xr.DataArray, volume_transform: AffineTransform
) -> None:
    with pytest.raises(TypeError, match="dims"):
        Geometry(volume, volume_transform, dims=(1,))  # type: ignore[arg-type]


def _queries_geometry(k: list[float]) -> Geometry:
    frame = ReferenceFrame.declared(
        ("test", "queries"), CoordinateSystem(("x", "y", "z"), ("mm",) * 3)
    )
    array = xr.DataArray(
        np.zeros((len(k), 3, 4)),
        dims=("k", "j", "i"),
        coords={"k": ("k", k), "j": np.arange(3), "i": np.arange(4)},
    )
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("i", "j", "k"), ("1", "1", "mm")),
        target=frame,
        matrix=np.diag([0.5, 0.5, 1.0]),
        translation=(1.0, 2.0, 3.0),
    )
    return Geometry(array, transform, dims=("k", "j", "i"))


def test_points_cover_every_sample_with_labelled_axes() -> None:
    geometry = _queries_geometry([0.0, 2.0, 5.0])
    points = geometry.points()
    assert points.dims == ("k", "j", "i", "axis")
    assert list(points.axis.values) == ["x", "y", "z"]
    assert list(points.units.values) == ["mm", "mm", "mm"]
    assert_allclose(points.sel(k=5.0, j=1, i=3).values, [2.5, 2.5, 8.0], rtol=0, atol=ATOL)
    assert_allclose(
        points.isel(k=2, j=1, i=3).values,
        geometry.point_at(k=2, j=1, i=3).values,
        rtol=0,
        atol=ATOL,
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_points_custom_names_support_a_fully_selected_point(
    volume: xr.DataArray, volume_transform: AffineTransform, lazy: bool
) -> None:
    if lazy:
        volume = volume.chunk({"slice": 1})
    with _recorded_tasks() as tasks:
        selected = volume.isel(slice=1, row=1, column=3)
        geometry = Geometry(selected, volume_transform, dims=())
        default = geometry.points()
        result = geometry.points(axis_dim="component", units_coord="component_units")
    assert not tasks
    assert default.dims == ("axis",)
    assert result.dims == ("component",)
    assert_array_equal(result.component, ["L", "P", "S"])
    assert_array_equal(result.component_units, ["mm", "mm", "mm"])
    assert_allclose(result.compute(), [11.5, 20.25, -3.0], rtol=0, atol=ATOL)
    assert_allclose(default.compute(), [11.5, 20.25, -3.0], rtol=0, atol=ATOL)
    if lazy:
        assert isinstance(result.data, da.Array)


def test_points_follow_the_array_chunks_lazily() -> None:
    geometry = _queries_geometry([0.0, 2.0, 5.0])
    lazy = Geometry(geometry.array.chunk({"k": 1}), geometry.transform, dims=("k", "j", "i"))
    points = lazy.points()
    assert isinstance(points.data, da.Array)
    assert points.chunks is not None and points.chunks[0] == (1, 1, 1)
    assert_allclose(points.values, geometry.points().values, rtol=0, atol=ATOL)


@pytest.mark.parametrize("dim", ["axis", "units", "__xarrayrf_source_axis__"])
@pytest.mark.parametrize("lazy", [False, True])
def test_points_output_names_preserve_geometry_labels(dim: str, lazy: bool) -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("position",), ("mm",)))
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("x",), ("mm",)),
        target=frame,
        matrix=[[1.0]],
        translation=[0.0],
    )
    array = xr.DataArray(
        np.zeros(3), dims=dim, coords={dim: [10, 20, 30], "x": (dim, [0.0, 1.0, 2.0])}
    )
    if lazy:
        array = array.chunk({dim: 2})
        array = array.assign_coords(x=(dim, da.from_array([0.0, 1.0, 2.0], chunks=2)))  # type: ignore[no-untyped-call]
    geometry = Geometry(array, transform, dims=(dim,))
    with _recorded_tasks() as tasks:
        if dim in {"axis", "units"}:
            with pytest.raises(ValueError, match=r"collides.*axis_dim.*units_coord"):
                geometry.points()
        else:
            default = geometry.points()
            assert default.dims == (dim, "axis")
        result = geometry.points(axis_dim="component", units_coord="component_units")
    assert not tasks
    assert result.dims == (dim, "component")
    assert_array_equal(result.coords[dim], [10, 20, 30])
    assert_array_equal(result.component, ["position"])
    assert_array_equal(result.component_units, ["mm"])
    assert_allclose(result.compute(), [[0.0], [1.0], [2.0]], rtol=0, atol=ATOL)
    if lazy:
        assert result.chunks == ((2, 1), (1,))


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"axis_dim": 1}, TypeError, "axis_dim must be a string"),
        ({"units_coord": None}, TypeError, "units_coord must be a string"),
        ({"axis_dim": ""}, ValueError, "axis_dim must be nonempty"),
        ({"units_coord": ""}, ValueError, "units_coord must be nonempty"),
        ({"axis_dim": "units"}, ValueError, "must be distinct"),
        ({"axis_dim": "k"}, ValueError, "collides.*axis_dim.*units_coord"),
        ({"units_coord": "i"}, ValueError, "collides.*axis_dim.*units_coord"),
    ],
)
def test_points_reject_invalid_output_names_before_evaluation(
    options: dict[str, Any], error: type[Exception], message: str
) -> None:
    geometry = _queries_geometry([0.0, 2.0, 5.0])
    lazy = Geometry(geometry.array.chunk({"k": 1}), geometry.transform, dims=geometry.dims)
    with _recorded_tasks() as tasks, pytest.raises(error, match=message):
        lazy.points(**options)
    assert not tasks


def test_points_custom_names_support_nonseparable_lazy_coordinates() -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("second", "first"), ("mm", "mm")))
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("u", "v"), ("mm", "mm")),
        target=frame,
        matrix=[[0.0, 2.0], [1.0, 0.0]],
        translation=[0.0, 0.0],
    )

    # Pixel evaluation must remain impossible even when the coordinate result is computed.
    def refuse_pixels(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise AssertionError("pixels evaluated")

    pixels = da.zeros((2, 3), chunks=(1, 2)).map_blocks(
        refuse_pixels, dtype=float, meta=np.empty((0, 0))
    )
    u = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    array = xr.DataArray(
        pixels,
        dims=("row", "column"),
        coords={
            "u": (("row", "column"), da.from_array(u, chunks=(1, 2))),  # type: ignore[no-untyped-call]
            "v": ("column", [0.0, 0.5, 1.0]),
            "row": [10, 20],
            "column": [3, 4, 5],
            "axis": "omitted",
            "units": "omitted",
        },
    )
    geometry = Geometry(array, transform, dims=("column", "row"))
    with _recorded_tasks() as tasks:
        default = geometry.points()
        custom = geometry.points(axis_dim="component", units_coord="component_units")
    assert not tasks
    assert custom.dims == ("row", "column", "component")
    assert custom.chunks == ((1, 1), (2, 1), (2,))
    expected = np.stack([np.broadcast_to([0.0, 1.0, 2.0], u.shape), u], axis=-1)
    assert_allclose(custom.compute(), expected, rtol=0, atol=ATOL)
    xr.testing.assert_identical(
        default.compute(), custom.rename(component="axis", component_units="units").compute()
    )
    point = geometry.point_at(row=1, column=2).rename(axis="component", units="component_units")
    assert_allclose(custom.isel(row=1, column=2), point, rtol=0, atol=ATOL)


def test_positions_at_inverts_nonuniform_offsets() -> None:
    """z = 3 + k, so z = 6.5 lies halfway between the samples at k = 2 and k = 5 mm."""
    geometry = _queries_geometry([0.0, 2.0, 5.0])
    positions = geometry.positions_at([[2.0, 2.5, 6.5], [1.0, 2.0, 3.0]])
    assert_allclose(positions, [[1.5, 1.0, 2.0], [0.0, 0.0, 0.0]], rtol=0, atol=ATOL)


def test_positions_at_marks_or_refuses_outside_points() -> None:
    geometry = _queries_geometry([0.0, 2.0, 5.0])
    with pytest.raises(ValueError, match="outside the samples"):
        geometry.positions_at([[1.0, 2.0, 9.0]])
    assert np.isnan(geometry.positions_at([[1.0, 2.0, 9.0]], outside="nan")).all()
    with pytest.raises(ValueError, match="outside must be"):
        geometry.positions_at([[1.0, 2.0, 3.0]], outside="clip")  # type: ignore[arg-type]


def test_positions_at_refuses_a_selected_plane() -> None:
    geometry = _queries_geometry([0.0, 2.0, 5.0])
    plane = Geometry(geometry.array.isel(k=1), geometry.transform, dims=("j", "i"))
    with pytest.raises(ValueError, match="needs a projection policy"):
        plane.positions_at([[1.0, 2.0, 5.0]])


def test_positions_at_refuses_non_monotonic_coordinates() -> None:
    with pytest.raises(ValueError, match="not strictly monotonic"):
        _queries_geometry([0.0, 5.0, 2.0]).positions_at([[1.0, 2.0, 4.0]])


def test_positions_at_refuses_two_axes_along_one_dimension() -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("x", "y"), ("mm", "mm")))
    array = xr.DataArray(
        np.zeros(3), dims=("s",), coords={"a": ("s", [0.0, 1.0, 2.0]), "b": ("s", [0.0, 2.0, 4.0])}
    )
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("a", "b"), ("1", "1")),
        target=frame,
        matrix=np.eye(2),
        translation=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="more than one source axis"):
        Geometry(array, transform, dims=("s",)).positions_at([[1.0, 2.0]])


def test_positions_at_needs_an_inverse() -> None:
    class Forward:
        source = ArrayCoordinates(("i", "j", "k"), ("1", "1", "mm"))
        target = ReferenceFrame.local(CoordinateSystem(("x", "y", "z"), ("mm",) * 3))

        def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return np.asarray(points, dtype=np.float64)

    geometry = _queries_geometry([0.0, 2.0, 5.0])
    with pytest.raises(TypeError, match="has no inverse"):
        Geometry(geometry.array, Forward(), dims=("k", "j", "i")).positions_at([[0.0, 0.0, 0.0]])


def test_points_refuse_a_transform_returning_the_wrong_shape() -> None:
    class Narrow:
        source = ArrayCoordinates(("i", "j", "k"), ("1", "1", "mm"))
        target = ReferenceFrame.local(CoordinateSystem(("x", "y", "z"), ("mm",) * 3))

        def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return np.asarray(points, dtype=np.float64)[..., :1]

    geometry = _queries_geometry([0.0, 2.0, 5.0])
    with pytest.raises(ValueError, match="returned shape"):
        Geometry(geometry.array, Narrow(), dims=("k", "j", "i")).points()


def test_positions_at_admits_a_single_sample_only_at_its_coordinate() -> None:
    geometry = _queries_geometry([2.0])
    assert_allclose(geometry.positions_at([[1.0, 2.0, 5.0]]), [[0.0, 0.0, 0.0]], atol=ATOL)
    assert np.isnan(geometry.positions_at([[1.0, 2.0, 5.5]], outside="nan")).all()


def _cells_geometry(
    k: list[float], offset: float | None = 0.5, j_offset: float | None = 0.5
) -> Geometry:
    """Unit-step lattice in j and i, with slice offsets ``k``; z = k, y = j, x = i."""
    frame = ReferenceFrame.declared(
        ("test", "cells"), CoordinateSystem(("x", "y", "z"), ("mm",) * 3)
    )
    array = xr.DataArray(
        np.zeros((len(k), 3, 4)),
        dims=("k", "j", "i"),
        coords={"k": ("k", k), "j": np.arange(3), "i": np.arange(4)},
    )
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(
            ("i", "j", "k"), ("1", "1", "mm"), sample_offset=(0.5, j_offset, offset)
        ),
        target=frame,
        matrix=np.eye(3),
        translation=(0.0, 0.0, 0.0),
    )
    return Geometry(array, transform, dims=("k", "j", "i"))


def test_the_cells_domain_reaches_half_a_step_beyond_centred_samples() -> None:
    geometry = _cells_geometry([0.0, 1.0, 2.0])
    corner = [[-0.5, -0.5, -0.5], [3.5, 2.5, 2.5]]
    with pytest.raises(ValueError, match="outside the samples domain"):
        geometry.positions_at(corner)
    positions = geometry.positions_at(corner, domain="cells")
    assert_allclose(positions, [[-0.5, -0.5, -0.5], [2.5, 2.5, 3.5]], rtol=0, atol=ATOL)
    assert np.isnan(geometry.positions_at([[-0.6, 0.0, 0.0]], outside="nan", domain="cells")).all()


@pytest.mark.parametrize(
    ("offset", "low", "high"),
    [(0.0, 0.0, 1.0), (1.0, -1.0, 0.0), (0.25, -0.25, 0.75)],
)
def test_the_sample_offset_places_the_cells(offset: float, low: float, high: float) -> None:
    """Cells reach ``offset`` steps below the first sample and ``1 - offset`` above the last."""
    geometry = _cells_geometry([0.0, 1.0, 2.0], offset=offset)
    inside = [[0.0, 0.0, low], [0.0, 0.0, 2.0 + high]]
    assert_allclose(
        geometry.positions_at(inside, domain="cells")[:, 0], [low, 2.0 + high], atol=ATOL
    )
    beyond = [[0.0, 0.0, low - 0.01], [0.0, 0.0, 2.0 + high + 0.01]]
    assert np.isnan(geometry.positions_at(beyond, outside="nan", domain="cells")).all()


def test_the_sample_offset_is_measured_by_coordinate_value_so_reversal_keeps_it() -> None:
    """A corner offset of 0 lies below each sample's coordinate, whichever way the array runs."""
    geometry = _cells_geometry([0.0, 1.0, 2.0], offset=0.0)
    reversed_ = Geometry(
        geometry.array.isel(k=slice(None, None, -1)), geometry.transform, dims=geometry.dims
    )
    for z, forward, backward in [(-0.001, np.nan, np.nan), (0.0, 0.0, 2.0), (2.9, 2.9, -0.9)]:
        point = [[1.0, 1.0, z]]
        assert_allclose(
            geometry.positions_at(point, outside="nan", domain="cells")[0, 0], forward, atol=ATOL
        )
        assert_allclose(
            reversed_.positions_at(point, outside="nan", domain="cells")[0, 0], backward, atol=ATOL
        )


def test_nonuniform_cells_reach_by_the_outer_steps() -> None:
    """Offsets 0, 2, 5: the first cell reaches 1 below 0 and the last 1.5 above 5."""
    geometry = _cells_geometry([0.0, 2.0, 5.0])
    positions = geometry.positions_at([[0.0, 0.0, -1.0], [0.0, 0.0, 6.5]], domain="cells")
    assert_allclose(positions[:, 0], [-0.5, 2.5], rtol=0, atol=ATOL)
    assert np.isnan(geometry.positions_at([[0.0, 0.0, 6.6]], outside="nan", domain="cells")).all()


def test_the_cells_domain_follows_a_crop() -> None:
    geometry = _cells_geometry([0.0, 1.0, 2.0, 3.0])
    cropped = Geometry(geometry.array.isel(k=slice(1, 3)), geometry.transform, dims=geometry.dims)
    positions = cropped.positions_at([[0.0, 0.0, 0.5], [0.0, 0.0, 2.5]], domain="cells")
    assert_allclose(positions[:, 0], [-0.5, 1.5], rtol=0, atol=ATOL)


def test_a_point_sampled_axis_reaches_no_further_than_its_samples() -> None:
    geometry = _cells_geometry([0.0, 1.0], j_offset=None)
    positions = geometry.positions_at([[-0.5, 0.0, -0.5], [3.5, 2.0, 1.5]], domain="cells")
    assert_allclose(positions, [[-0.5, 0.0, -0.5], [1.5, 2.0, 3.5]], rtol=0, atol=ATOL)
    beyond = [[0.0, -0.01, 0.0], [0.0, 2.01, 0.0]]
    assert np.isnan(geometry.positions_at(beyond, outside="nan", domain="cells")).all()


def test_the_cells_domain_refuses_a_single_sample_with_cells_and_unknown_domains() -> None:
    with pytest.raises(ValueError, match="declares cells but has a single sample"):
        _cells_geometry([2.0]).positions_at([[0.0, 0.0, 2.0]], domain="cells")
    single = _cells_geometry([2.0], offset=None)
    assert_allclose(single.positions_at([[0.0, 0.0, 2.0]], domain="cells"), [[0, 0, 0]], atol=ATOL)
    with pytest.raises(ValueError, match="domain must be"):
        _cells_geometry([0.0, 1.0]).positions_at(
            [[0.0, 0.0, 0.0]],
            domain="voxels",  # type: ignore[arg-type]
        )


def test_nonuniform_coordinates_admit_rounding_slack_at_the_outer_samples() -> None:
    """As uniform ones do: a point a few ulps outside lands on the edge sample, not NaN."""
    positions = _queries_geometry([0.0, 2.0, 5.0]).positions_at(
        [[1.0, 2.0, 3.0 - 1e-12], [1.0, 2.0, 8.0 + 1e-12]]
    )
    assert_array_equal(positions[:, 0], [0.0, 2.0])
    assert not np.signbit(positions).any(), "an edge position is +0.0, not -0.0"


_ORIENTED = ("right-to-left", "anterior-to-posterior", "inferior-to-superior")


def _grid(
    *,
    k: list[float] | None = None,
    shape: tuple[int, int, int] = (4, 5, 6),
    matrix: npt.ArrayLike = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 2.0)),
    translation: npt.ArrayLike = (1.0, 2.0, 3.0),
    frame: ReferenceFrame | None = None,
    affine: bool = True,
    lazy: bool = False,
) -> Geometry:
    """x = 0.5 i, y = 0.5 j, z = 2 k (k may be nonuniform offsets), in an LPS-oriented frame."""
    from xarrayrf import DirectionVocabulary

    if frame is None:
        vocabulary = DirectionVocabulary("test-anatomy", _ORIENTED)
        frame = ReferenceFrame.declared(
            ("test", "coincidence"),
            CoordinateSystem(
                ("x", "y", "z"), ("mm",) * 3, vocabulary=vocabulary, orientation=_ORIENTED
            ),
        )
    offsets = list(range(shape[0])) if k is None else k
    array = xr.DataArray(
        (da.zeros if lazy else np.zeros)((len(offsets), shape[1], shape[2])),
        dims=("k", "j", "i"),
        coords={
            "k": ("k", np.asarray(offsets, dtype=float)),
            "j": np.arange(shape[1]),
            "i": np.arange(shape[2]),
        },
    )
    transform: Any = AffineTransform.from_matrix(
        source=ArrayCoordinates(("i", "j", "k"), ("1", "1", "1")),
        target=frame,
        matrix=matrix,
        translation=translation,
    )
    if not affine:
        transform = _PointsOnly(transform)
    return Geometry(array, transform, dims=("k", "j", "i"))


class _PointsOnly:
    """An affine hidden behind the point and inverse protocols, forcing the general path."""

    def __init__(self, affine: AffineTransform) -> None:
        self._affine = affine

    @property
    def source(self) -> Any:
        return self._affine.source

    @property
    def target(self) -> Any:
        return self._affine.target

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return self._affine.transform_point(points)

    def inverse(self) -> AffineTransform:
        return self._affine.inverse()


def test_a_geometry_coincides_with_itself_and_a_jittered_copy() -> None:
    grid = _grid()
    assert grid.is_coincident(grid)
    assert grid.is_coincident(_grid(translation=(1.0 + 1e-7, 2.0, 3.0)))
    assert not grid.is_coincident(_grid(translation=(1.0 + 1e-5, 2.0, 3.0)))


def test_the_tolerance_is_in_steps_so_it_scales_with_spacing() -> None:
    """A 2e-6 mm shift is 1e-6 of the 2 mm slice step but 4e-6 of the 0.5 mm row step."""
    grid = _grid()
    assert grid.is_coincident(_grid(translation=(1.0, 2.0, 3.0 + 1.9e-6)))
    assert not grid.is_coincident(_grid(translation=(1.0, 2.0 + 1.9e-6, 3.0)))
    assert grid.is_coincident(_grid(translation=(1.0, 2.0 + 1.9e-6, 3.0)), tolerance=1e-5)


def test_a_rotation_small_near_the_origin_is_caught_at_the_far_corner() -> None:
    angle = 1e-6
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    big = (4, 400, 400)
    grid = _grid(shape=big)
    rotated = _grid(shape=big, matrix=rotation @ np.diag([0.5, 0.5, 2.0]))
    assert not grid.is_coincident(rotated)
    assert grid.is_coincident(rotated, tolerance=1e-3)


def test_nonuniform_offsets_compare_by_local_gap() -> None:
    grid = _grid(k=[0.0, 1.0, 3.0, 7.0])
    assert grid.is_coincident(_grid(k=[0.0, 1.0, 3.0 + 1e-6, 7.0]))
    assert not grid.is_coincident(_grid(k=[0.0, 1.0, 3.0 + 1e-4, 7.0]))


def test_the_general_path_agrees_with_the_corner_check() -> None:
    grid = _grid()
    for shift, expected in ((1e-7, True), (1e-5, False)):
        other = _grid(translation=(1.0 + shift, 2.0, 3.0), affine=False)
        assert grid.is_coincident(other) is expected
        assert other.is_coincident(grid) is expected


def test_coordinate_systems_of_one_frame_compare_through_the_derived_change() -> None:
    lps = _grid()
    ras_frame = lps.frame.with_coordinate_system(
        CoordinateSystem(
            ("x", "y", "z"),
            ("mm",) * 3,
            vocabulary=lps.frame.coordinate_system.vocabulary,
            orientation=("left-to-right", "posterior-to-anterior", "inferior-to-superior"),
        )
    )
    flip = np.diag([-1.0, -1.0, 1.0])
    ras = _grid(
        matrix=flip @ np.diag([0.5, 0.5, 2.0]), translation=(-1.0, -2.0, 3.0), frame=ras_frame
    )
    assert lps.is_coincident(ras)
    assert ras.is_coincident(lps)


def test_different_frames_never_coincide() -> None:
    other = ReferenceFrame.declared(("test", "elsewhere"), _grid().frame.coordinate_system)
    assert not _grid().is_coincident(_grid(frame=other))


def test_dimensions_pair_by_name_and_size() -> None:
    grid = _grid()
    assert not grid.is_coincident(_grid(shape=(4, 5, 7)))
    stored = Geometry(grid.array.transpose("i", "k", "j"), grid.transform, dims=("i", "j", "k"))
    assert grid.is_coincident(stored)
    renamed = Geometry(
        grid.array.rename(k="s"),
        AffineTransform.from_matrix(
            source=ArrayCoordinates(("i", "j", "s"), ("1", "1", "1")),
            target=grid.frame,
            matrix=np.diag([0.5, 0.5, 2.0]),
            translation=(1.0, 2.0, 3.0),
        ),
        dims=("s", "j", "i"),
    )
    assert not grid.is_coincident(renamed)


def test_a_crop_coincides_only_with_the_same_crop() -> None:
    grid = _grid()
    crop = Geometry(grid.array.isel(k=slice(1, 3)), grid.transform, dims=grid.dims)
    assert crop.is_coincident(
        Geometry(grid.array.isel(k=slice(1, 3)), grid.transform, dims=grid.dims)
    )
    assert not crop.is_coincident(
        Geometry(grid.array.isel(k=slice(0, 2)), grid.transform, dims=grid.dims)
    )


@pytest.mark.parametrize(
    ("tolerance", "error", "message"),
    [
        (-1e-6, ValueError, r"\[0, 0.5\)"),
        (0.5, ValueError, r"\[0, 0.5\)"),
        (float("nan"), ValueError, r"\[0, 0.5\)"),
        (True, TypeError, "real number"),
        ("1e-6", TypeError, "real number"),
    ],
)
def test_the_tolerance_is_validated(
    tolerance: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _grid().is_coincident(_grid(), tolerance=tolerance)  # type: ignore[arg-type]


def test_is_coincident_needs_a_geometry() -> None:
    with pytest.raises(TypeError, match="must be a Geometry"):
        _grid().is_coincident(_grid().array)  # type: ignore[arg-type]


def test_equivalent_frames_without_a_derivable_change_raise() -> None:
    grid = _grid()
    metres = grid.frame.with_coordinate_system(CoordinateSystem(("x", "y", "z"), ("m",) * 3))
    with pytest.raises(ValueError):
        grid.is_coincident(_grid(frame=metres))


def test_interior_deviations_of_nearly_uniform_offsets_are_not_missed() -> None:
    """Both stacks pass as lattices within 1e-6 steps and share their corners, yet differ inside."""
    first = _grid(k=[0.0, 1.0 + 0.75e-6, 2.0, 3.0])
    second = _grid(k=[0.0, 1.0 - 0.75e-6, 2.0, 3.0])
    assert first.lattice().spacing is not None and second.lattice().spacing is not None
    assert not first.is_coincident(second)
    assert first.is_coincident(second, tolerance=2e-6)


def test_a_single_sample_dimension_must_agree_to_rounding() -> None:
    plane = _grid(k=[2.0])
    assert plane.is_coincident(_grid(k=[2.0]))
    assert not plane.is_coincident(_grid(k=[2.0 + 1e-6]))


def test_the_general_path_agrees_on_nonuniform_offsets() -> None:
    grid = _grid(k=[0.0, 1.0, 3.0, 7.0])
    for k, expected in (([0.0, 1.0, 3.0 + 1e-6, 7.0], True), ([0.0, 1.0, 3.0 + 1e-4, 7.0], False)):
        assert grid.is_coincident(_grid(k=k, affine=False)) is expected


def test_the_affine_check_does_not_visit_every_sample() -> None:
    """512^3 samples: linear in the samples per dimension, so this is immediate."""
    import time

    shape = (512, 512, 512)
    grid = _grid(shape=shape, lazy=True)
    other = _grid(shape=shape, translation=(1.0 + 1e-7, 2.0, 3.0), lazy=True)
    assert isinstance(grid.array.data, da.Array)
    started = time.perf_counter()
    assert grid.is_coincident(other)
    assert time.perf_counter() - started < 1.0


def test_a_geometry_without_a_determined_inverse_raises_in_either_order() -> None:
    singular = _grid(matrix=((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.0)))
    with pytest.raises(ValueError):
        _grid().is_coincident(singular)
    with pytest.raises(ValueError):
        singular.is_coincident(_grid())


def test_a_unitless_axis_refuses_a_units_attribute_and_writes_none() -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("c", "x"), (None, "mm")))
    array = xr.DataArray(
        np.zeros((2, 3)), dims=("c", "i"), coords={"c": ("c", [0.0, 1.0]), "i": np.arange(3)}
    )
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("c", "i"), (None, "1")),
        target=frame,
        matrix=np.diag([1.0, 0.5]),
        translation=(0.0, 0.0),
    )
    geometry = Geometry(array, transform, dims=("c", "i"))
    units = geometry.point_at(c=1, i=2).units
    assert units.isnull().values.tolist() == [True, False] and units.values[1] == "mm"
    coordinates = geometry.frame_coordinates(names=("cc", "xx"))
    assert "units" not in coordinates["cc"].attrs and coordinates["xx"].attrs["units"] == "mm"
    labelled = array.assign_coords(c=array.c.assign_attrs(units="1"))
    with pytest.raises(ValueError, match="declares no unit"):
        Geometry(labelled, transform, dims=("c", "i"))
