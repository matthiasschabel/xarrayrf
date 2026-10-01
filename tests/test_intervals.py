"""Declared support through public sampling, binding and persistence APIs."""

from __future__ import annotations

import copy
import json
import pickle
import warnings
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from xarrayrf import (
    AffineTransform,
    ArrayCoordinates,
    CoordinateSystem,
    Geometry,
    Grid,
    MalformedDataError,
    ReferenceFrame,
    decode,
    encode,
)
from xarrayrf.native import frame_array, grid_coordinates

ATOL = 1e-12  # Roundoff for synthetic coordinate arithmetic in millimetres.


def mapping(offset: float | None = 0.5) -> AffineTransform:
    return AffineTransform(
        source=ArrayCoordinates(("z",), ("mm",), sample_offset=(offset,)),
        target=ReferenceFrame.declared(
            ("synthetic", "intervals"), CoordinateSystem(("Z",), ("mm",))
        ),
        matrix=[[1]],
        translation=[0],
    )


def grid(values: Any = (0, 2, 5), width: float = 1.0) -> Grid:
    centers = np.asarray(values, dtype=np.float64)
    rows = np.stack((centers - width / 2, centers + width / 2), axis=-1)
    return Grid(mapping(), {"z": ("slice", values)}, intervals={"z": rows})


@pytest.mark.parametrize(
    "intervals,error,match",
    [
        ([], TypeError, "mapping"),
        ({"slice": [[-1, 1]]}, ValueError, "unknown source"),
        ({"z": [-1, 1]}, ValueError, "shape"),
        ({"z": [[-1, 1], [1, 3]]}, ValueError, "shape"),
        ({"z": [[0, 0]]}, ValueError, "lo < hi"),
        ({"z": [[1, -1]]}, ValueError, "lo < hi"),
        ({"z": [[-np.inf, 1]]}, ValueError, "finite"),
        ({"z": [[-1, np.nan]]}, ValueError, "finite"),
        ({"z": [[-1, 2]]}, ValueError, "sample_offset"),
        ({"z": [["-1", "1"]]}, TypeError, "real"),
    ],
)
def test_interval_construction_refusals(intervals: Any, error: type[Exception], match: str) -> None:
    for construct in (
        lambda: Grid(mapping(), {"z": ("slice", [0])}, intervals=intervals),
        lambda: Geometry(
            xr.DataArray([0], dims="slice", coords={"z": ("slice", [0])}),
            mapping(),
            dims=("slice",),
            intervals=intervals,
        ),
    ):
        with pytest.raises(error, match=match):
            construct()


def test_point_samples_refuse_intervals() -> None:
    with pytest.raises(ValueError, match=r"'z'.*point-sampled"):
        Grid(mapping(None), {"z": ("slice", [0])}, intervals={"z": [[-1, 1]]})


def test_scalar_interval_shape_and_immutable_value_semantics() -> None:
    scalar = Grid(mapping(), {"z": 0}, intervals={"z": [-1, 1]})
    with pytest.raises(ValueError, match="shape"):
        Grid(mapping(), {"z": 0}, intervals={"z": [[-1, 1]]})
    original = grid()
    assert original != Grid(original.transform, original.coordinates)
    assert original != grid(width=2)
    assert "intervals=('z',)" in repr(original)
    assert not Grid(mapping(), {"z": 0}).intervals
    for value in (original, scalar):
        for restored in (
            copy.copy(value),
            copy.deepcopy(value),
            pickle.loads(pickle.dumps(value)),
            decode(json.loads(json.dumps(encode(value)))),
        ):
            assert restored == value
            assert hash(restored) == hash(value)
            rows = restored.intervals["z"]
            assert rows.dtype == np.float64
            with pytest.raises(ValueError):
                rows.setflags(write=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                rows.shape = (rows.size,)
            assert restored == value
        with pytest.raises(TypeError):
            value.intervals["z"] = np.zeros(2)  # type: ignore[index]
    assert original.transpose() == original


@pytest.mark.parametrize("width", [1.0, 8.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_gapped_and_overlapping_cells_grid_geometry_agree(width: float, reverse: bool) -> None:
    value = grid(width=width)
    if reverse:
        value = value.isel(slice=slice(None, None, -1))
    array = frame_array(np.arange(3), value)
    plain_view = Geometry(
        array.rf.unframe(), value.transform, dims=value.dims, intervals=value.intervals
    )
    lo = value.intervals["z"][:, 0].min()
    hi = value.intervals["z"][:, 1].max()
    points = np.array([[lo], [1.25], [hi]])
    expected = value.positions_at(points, domain="cells")
    for view in (array.rf.geometry, plain_view):
        assert view.grid() == value
        assert_allclose(view.positions_at(points, domain="cells"), expected, rtol=0, atol=ATOL)
        assert_allclose(view.points_at(expected, domain="cells"), points, rtol=0, atol=ATOL)
    with pytest.raises(ValueError, match="outside"):
        value.positions_at([[lo - 0.1]], domain="cells")
    assert np.isnan(value.positions_at([[hi + 0.1]], domain="cells", outside="nan")).all()


@pytest.mark.parametrize("offset", [0.0, 0.25, 0.5, 1.0])
def test_singleton_slab_round_trip(offset: float) -> None:
    value = Grid(mapping(offset), {"z": ("slice", [10 + offset * 4])}, intervals={"z": [[10, 14]]})
    array = frame_array(np.array([7]), value)
    positions = np.array([[-offset], [0], [1 - offset]])
    for query in (value, array.rf.geometry):
        points = query.points_at(positions, domain="cells")
        assert_allclose(points[[0, 2]], [[10], [14]], rtol=0, atol=ATOL)
        assert_allclose(query.positions_at(points, domain="cells"), positions, rtol=0, atol=ATOL)
        with pytest.raises(ValueError, match="outside"):
            query.points_at([[-offset - 0.1]], domain="cells")


@pytest.mark.parametrize("declared", [False, True])
def test_singleton_samples_domain_keeps_coordinate_matching_tolerance(declared: bool) -> None:
    slab = grid([1000], width=1)
    value = slab if declared else Grid(slab.transform, slab.coordinates)
    array = frame_array(np.array([7]), value)
    point = [[1000 + 5e-7]]  # Within 1e-9 of the sample's magnitude, beyond 1e-9 of its width.
    for query in (value, array.rf.geometry):
        assert_allclose(query.positions_at(point), [[0]], rtol=0, atol=ATOL)
        with pytest.raises(ValueError, match="single sample"):
            query.points_at([[0.25]], outside="extrapolate")
    coordinates = array.rf.unframe().assign_coords(array.rf.geometry.frame_coordinates())
    result = coordinates.sel(
        Z=xr.DataArray(np.asarray(point)[:, 0], dims="point"), method="nearest"
    )
    assert_array_equal(result.values, [7])


@pytest.mark.parametrize(
    "indexer",
    [
        slice(1, None),
        slice(None, None, 2),
        slice(None, None, -1),
        [2, 0],
        np.array([True, False, True]),
    ],
)
def test_isel_rows_follow_labels(indexer: Any) -> None:
    value = grid()
    result = frame_array(np.arange(3), value).isel(slice=indexer)
    assert_array_equal(result.rf.grid.intervals["z"], value.intervals["z"][indexer])
    assert result.rf.grid == value.isel(slice=indexer)


def test_sel_rows_follow_labels_and_scalar_selection_retains_support() -> None:
    value = grid()
    array = frame_array(np.arange(3), value)
    result = array.sel(z=[5, 0])
    assert_array_equal(result.rf.grid.intervals["z"], value.intervals["z"][[2, 0]])
    assert result.rf.grid == value.sel(z=[5, 0])
    for scalar in (array.sel(z=2), array.isel(slice=1)):
        assert scalar.rf.grid.dims == ()
        assert_array_equal(scalar.rf.grid.intervals["z"], value.intervals["z"][1])
        assert scalar.rf.encode().rf.decode().rf.grid == scalar.rf.grid


def test_roll_support_moves_with_coordinate_labels() -> None:
    value = grid()
    array = frame_array(np.arange(3), value)
    result = array.roll(slice=1, roll_coords=True)
    assert_array_equal(result.rf.grid.intervals["z"], value.intervals["z"][[2, 0, 1]])
    assert array.roll(slice=1, roll_coords=False).rf.grid == value


def test_rename_carries_support() -> None:
    value = grid()
    array = frame_array(np.arange(3), value)
    renamed = array.rename({"z": "depth", "slice": "plane"})
    assert_array_equal(renamed.rf.grid.intervals["depth"], value.intervals["z"])
    assert renamed.rf.grid.dims == ("plane",)


@pytest.mark.xfail(
    not hasattr(xr.Index, "swap_dims"), strict=True, reason="stock xarray lacks Index.swap_dims"
)
def test_swap_dims_carries_support() -> None:
    value = grid()
    array = frame_array(np.arange(3), value)
    swapped = array.assign_coords(plane=("slice", [10, 11, 12])).swap_dims({"slice": "plane"})
    assert swapped.rf.grid.dims == ("plane",)
    assert_array_equal(swapped.rf.grid.intervals["z"], value.intervals["z"])


def test_binding_alignment_merges_support_from_both_operands() -> None:
    left = frame_array(np.arange(2), grid([0, 2]))
    right = frame_array(np.arange(2), grid([2, 5]))
    a, b = xr.align(left, right, join="outer")
    assert a.rf.grid == b.rf.grid == grid()
    a, b = xr.align(left, right, join="inner")
    assert a.rf.grid == b.rf.grid == grid([2])
    assert (left + left).rf.grid == left.rf.grid


def test_matched_labels_with_conflicting_or_missing_support_refuse() -> None:
    left = frame_array(np.arange(3), grid())
    right = frame_array(np.arange(3), grid(width=2))
    for operation in (
        lambda: xr.align(left, right),
        lambda: left + right,
    ):
        with pytest.raises(ValueError, match=r"source axis 'z'.*intervals"):
            operation()
    plain_support = frame_array(
        np.arange(3), Grid(left.rf.grid.transform, left.rf.grid.coordinates)
    )
    assert left.xindexes["z"].equals(plain_support.xindexes["z"]) is False
    with pytest.raises(ValueError, match=r"source axis 'z'.*intervals"):
        xr.align(left, plain_support)


def test_binding_interval_equality_is_bool_and_honours_excluded_dimensions() -> None:
    left = two_axis_array()
    value = left.rf.grid
    right = frame_array(
        left.data,
        Grid(value.transform, value.coordinates, intervals={"z": grid(width=2).intervals["z"]}),
    )
    assert left.xindexes["z"].equals(right.xindexes["z"]) is False
    assert left.xindexes["z"].equals(right.xindexes["z"], exclude=frozenset({"z"})) is True
    a, b = xr.align(left, right, exclude=["z"])
    assert a.rf.grid == left.rf.grid
    assert b.rf.grid == right.rf.grid
    with pytest.raises(ValueError, match=r"source axis 'z'.*intervals"):
        xr.align(left, right, exclude=["x"])


def test_merge_of_conflicting_intervals_raises_xarray_merge_error() -> None:
    left = frame_array(np.arange(3), grid())
    right = frame_array(np.arange(3), grid(width=2))
    with pytest.raises(xr.MergeError):
        left.coords.merge(right.coords)


@pytest.mark.parametrize("reverse", [False, True])
def test_binding_equality_with_repeated_labels_returns_false(reverse: bool) -> None:
    left = frame_array(np.arange(3), grid())
    right = left.isel(slice=[0, 0, 1])
    if reverse:
        left, right = right, left
    assert left.xindexes["z"].equals(right.xindexes["z"]) is False


@pytest.mark.parametrize("reverse", [False, True])
def test_coordinate_merge_with_repeated_labels_raises_xarray_merge_error(reverse: bool) -> None:
    left = frame_array(np.arange(3), grid())
    right = left.isel(slice=[0, 0, 1])
    if reverse:
        left, right = right, left
    with pytest.raises(xr.MergeError):
        left.coords.merge(right.coords)


def two_axis_array() -> xr.DataArray:
    transform = AffineTransform(
        source=ArrayCoordinates(("z", "x"), ("mm", "mm"), sample_offset=(0.5, 0.5)),
        target=ReferenceFrame.declared(
            ("synthetic", "plane"), CoordinateSystem(("Z", "X"), ("mm", "mm"))
        ),
        matrix=np.eye(2),
        translation=[0, 0],
    )
    value = Grid(
        transform,
        {"z": ("z", [0, 2, 5]), "x": ("x", [0, 1])},
        intervals={"z": grid().intervals["z"]},
    )
    return frame_array(np.zeros((3, 2)), value)


@pytest.mark.xfail(
    not hasattr(xr.Index, "join_overlapping"),
    strict=True,
    raises=xr.AlignmentError,
    reason="stock xarray lacks mixed-index alignment hooks",
)
@pytest.mark.parametrize("how", ["inner", "left", "exact"])
def test_plain_index_alignment_subsets_keep_known_support(how: Any) -> None:
    array = two_axis_array()
    plain = xr.DataArray([1, 2, 3], dims="z", coords={"z": [0, 2, 5]})
    plain.coords["z"].attrs["units"] = "mm"
    result, _ = xr.align(array, plain, join=how)
    assert result.rf.grid == array.rf.grid
    subset, _ = xr.align(array, plain.isel(z=slice(1, None)), join="inner")
    assert_array_equal(subset.rf.grid.intervals["z"], array.rf.grid.intervals["z"][1:])


def test_plain_index_reindex_new_labels_refused_by_xarray() -> None:
    value = Grid(mapping(), {"z": ("z", [0, 2, 5])}, intervals={"z": grid().intervals["z"]})
    array = frame_array(np.arange(3), value)
    plain = xr.DataArray([1, 2], dims="z", coords={"z": [0, 8]})
    with pytest.raises(xr.AlignmentError, match="coordinate 'z'"):
        array.reindex(z=[0, 8])
    with pytest.raises(xr.AlignmentError, match="coordinate 'z'"):
        array.reindex_like(plain)


def test_binding_reindex_subset_keeps_support() -> None:
    array = frame_array(np.arange(3), grid())
    subset = array.isel(slice=[2, 0])
    result = array.reindex_like(subset)
    assert result.rf.grid == subset.rf.grid


@pytest.mark.xfail(
    not hasattr(xr.Index, "join_overlapping"),
    strict=True,
    reason="stock xarray lacks mixed-index alignment hooks",
)
def test_plain_index_new_labels_refuse_with_axis_message() -> None:
    array = two_axis_array()
    plain = xr.DataArray([1, 2], dims="z", coords={"z": [0, 8]})
    plain.coords["z"].attrs["units"] = "mm"
    with pytest.raises(ValueError, match=r"source axis 'z'.*intervals"):
        xr.align(array, plain, join="outer")


def test_concat_along_a_geometry_dimension_refuses() -> None:
    array = frame_array(np.arange(3), grid())
    with pytest.raises(ValueError, match="concatenation"):
        xr.concat([array, array], dim="slice")


@pytest.mark.parametrize("dim", ["time", "echo"])
def test_concat_along_other_dimensions_keeps_binding_and_intervals(dim: str) -> None:
    def series(start: int) -> xr.DataArray:
        return frame_array(
            np.zeros((2, 3)),
            grid(),
            dims=("time", "slice"),
            coords={"time": ("time", np.arange(start, start + 2), {})},
        )

    second = series(2) if dim == "time" else series(0)
    result = xr.concat([series(0), second], dim=dim)
    assert result.rf.grid == grid()
    assert result.sizes[dim] == (4 if dim == "time" else 2)


def test_concat_along_time_with_differing_grids_aligns_or_refuses_exactly() -> None:
    def series(values: Any, start: int) -> xr.DataArray:
        return frame_array(
            np.zeros((1, 3)),
            grid(values),
            dims=("time", "slice"),
            coords={"time": ("time", np.array([start]), {})},
        )

    first, second = series((0, 2, 5), 0), series((1, 3, 6), 1)
    joined = xr.concat([first, second], dim="time", join="outer")
    assert joined.sizes["slice"] == 6
    assert joined.rf.is_framed
    with pytest.raises(ValueError):
        xr.concat([first, second], dim="time", join="exact")


def test_every_framing_door_and_encoding_preserves_intervals() -> None:
    value = grid()
    plain = xr.DataArray(np.arange(3), dims="slice", coords=dict(value.coordinates))
    for array in (
        frame_array(plain.data, value),
        plain.rf.frame(value),
        plain.rf.frame(value.transform, dims=value.dims, intervals=value.intervals),
        plain.assign_coords(grid_coordinates(value)),
    ):
        assert array.rf.grid == value
        assert array.rf.geometry.grid() == value
        assert array.rf.encode().rf.decode().rf.grid == value
        assert array.copy(deep=True).rf.grid == value
        assert array.rf.assume_frame(value.frame).rf.grid == value


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {"z": []},
        {"z": [[False, 1]]},
        {"z": [["-1", "1"]]},
        {"z": [[-1, float("inf")]]},
        {"z": [[0, 0]]},
        {"missing": [[-1, 1]]},
        {"z": [[-1, 2]]},
    ],
)
def test_grid_and_native_decode_strict_interval_validation(bad: Any) -> None:
    value = grid([0])
    payload = encode(value)
    payload["value"]["intervals"] = bad
    with pytest.raises(MalformedDataError):
        decode(payload)
    array = frame_array(np.array([1]), value).rf.encode()
    payload = json.loads(array.attrs["xarrayrf_binding"])
    payload["intervals"] = bad
    array.attrs["xarrayrf_binding"] = json.dumps(payload)
    with pytest.raises(MalformedDataError):
        array.rf.decode()


def test_grid_and_native_decode_refuse_empty_intervals_for_scalar_coordinate() -> None:
    value = Grid(mapping(), {"z": 0}, intervals={"z": [-1, 1]})
    payload = encode(value)
    payload["value"]["intervals"] = {"z": []}
    with pytest.raises(MalformedDataError):
        decode(payload)
    array = frame_array(np.array(1), value).rf.encode()
    payload = json.loads(array.attrs["xarrayrf_binding"])
    payload["intervals"] = {"z": []}
    array.attrs["xarrayrf_binding"] = json.dumps(payload)
    with pytest.raises(MalformedDataError):
        array.rf.decode()


@pytest.mark.parametrize("origin", ["direct", "selection", "inner_join"])
@pytest.mark.parametrize("native", [False, True])
def test_empty_intervals_round_trip(origin: str, native: bool) -> None:
    array = frame_array(np.empty(0), grid([]))
    if origin == "selection":
        array = frame_array(np.arange(3), grid()).isel(slice=[])
    elif origin == "inner_join":
        array, _ = xr.align(
            frame_array(np.arange(3), grid()),
            frame_array(np.arange(2), grid([10, 12])),
            join="inner",
        )
    value = array.rf.grid
    restored = (
        array.rf.encode().rf.decode().rf.grid
        if native
        else decode(json.loads(json.dumps(encode(value))))
    )
    assert restored == value
    assert restored.intervals["z"].shape == (0, 2)


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_resample_slab_uses_source_support_and_retains_only_target_intervals(method: Any) -> None:
    source = frame_array(np.array([7.0]), grid([0], width=4))
    target = grid([-2, 0, 2, 3], width=0.5)
    for operand in (
        target,
        frame_array(np.zeros(4), target),
        frame_array(np.zeros(4), target).rf.geometry,
    ):
        result = source.rf.resample_to(operand, domain="cells", method=method)
        assert_allclose(result.values, [7, 7, 7, np.nan], rtol=0, atol=ATOL)
        assert result.rf.grid == target
    without = Grid(target.transform, target.coordinates)
    assert not source.rf.resample_to(without, domain="cells").rf.grid.intervals


def test_public_binding_index_reindex_like_refuses_plain_index() -> None:
    array = two_axis_array()
    known = xr.DataArray([0, 0], dims="z", coords={"z": [5, 0]})
    unknown = known.assign_coords(z=[5, 8])
    for plain in (known, unknown):
        with pytest.raises(ValueError, match="incompatible framed coordinate mappings"):
            array.xindexes["z"].reindex_like(plain.xindexes["z"])


def test_interval_snapshot_is_detached_and_transpose_keeps_axis_associations() -> None:
    array = two_axis_array()
    value = array.rf.grid
    assert value.transpose().transpose() == value
    assert_array_equal(value.transpose().intervals["z"], value.intervals["z"])
    original = np.array([[-1.0, 1.0]])
    frozen = Grid(mapping(), {"z": ("slice", [0])}, intervals={"z": original})
    original[:] = 0
    assert_array_equal(frozen.intervals["z"], [[-1, 1]])
    for copied in (array.copy(deep=True), pickle.loads(pickle.dumps(array))):
        assert copied.rf.grid == value
        with pytest.raises(ValueError):
            cast(Any, copied.xindexes["z"]).intervals["z"].setflags(write=True)


def test_interval_agreement_tolerance_is_relative_to_width() -> None:
    value = Grid(mapping(), {"z": ("slice", [5e-10])}, intervals={"z": [[-0.5, 0.5]]})
    assert value.sizes == {"slice": 1}
    with pytest.raises(ValueError, match="sample_offset"):
        Grid(mapping(), {"z": ("slice", [2e-9])}, intervals={"z": [[-0.5, 0.5]]})


@pytest.mark.parametrize("accepted", [True, False])
def test_epoch_scale_interval_agreement_accepts_roundoff_but_refuses_disagreement(
    accepted: bool,
) -> None:
    transform = AffineTransform(
        source=ArrayCoordinates(("t",), ("s",), sample_offset=(0.5,)),
        target=ReferenceFrame.declared(("synthetic", "epoch"), CoordinateSystem(("T",), ("s",))),
        matrix=[[1]],
        translation=[0],
    )
    lo = 1.7e9 + 0.1
    intervals = {"t": [[lo, lo + 1]]}
    sample = np.nextafter(lo + 0.5, np.inf) if accepted else lo + 0.5 + 1e-4
    coordinates = {"t": ("time", [sample])}
    array = xr.DataArray([0], dims="time", coords=coordinates)
    for construct in (
        lambda: Grid(transform, coordinates, intervals=intervals),
        lambda: Geometry(array, transform, dims=("time",), intervals=intervals),
    ):
        if accepted:
            assert construct().sizes == {"time": 1}
        else:
            with pytest.raises(ValueError, match="sample_offset"):
                construct()


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("above", [False, True])
def test_epoch_scale_narrow_interval_refuses_sample_outside(scalar: bool, above: bool) -> None:
    lo = 1.7e9
    hi = np.nextafter(lo, np.inf)
    sample = lo + 2e-6 if above else lo - 2e-6
    coordinates = {"z": sample if scalar else ("slice", [sample])}
    intervals = {"z": [lo, hi] if scalar else [[lo, hi]]}
    dims = () if scalar else ("slice",)
    array = xr.DataArray(np.array(0) if scalar else [0], dims=dims, coords=coordinates)
    for construct in (
        lambda: Grid(mapping(), coordinates, intervals=intervals),
        lambda: Geometry(array, mapping(), dims=dims, intervals=intervals),
    ):
        with pytest.raises(ValueError, match=r"source axis 'z'.*sample lies outside.*interval"):
            construct()


def test_geometry_rechecks_interval_agreement_after_coordinate_edit() -> None:
    array = frame_array(np.zeros(3), grid()).rf.unframe()
    view = Geometry(array, mapping(), dims=("slice",), intervals=grid().intervals)
    array.coords["z"] = ("slice", [0, 2, 6])
    with pytest.raises(ValueError, match="sample_offset"):
        view.positions_at([[0]], domain="cells")


def test_empty_declared_axis_refuses_nonempty_cells_queries() -> None:
    value = grid([])
    with pytest.raises(ValueError, match="empty coordinate"):
        value.points_at([[0]], domain="cells")


@pytest.mark.parametrize("values", [[0], [0, 2, 5]])
def test_frame_coordinate_selection_uses_declared_outer_bounds(values: list[int]) -> None:
    value = grid(values, width=4)
    source = frame_array(np.arange(len(values)), value).rf.unframe()
    view = Geometry(source, value.transform, dims=value.dims, intervals=value.intervals)
    coordinates = source.assign_coords(view.frame_coordinates(domain="cells"))
    for selection in (slice(None), slice(None, None, -1), slice(0, 1)):
        selected = coordinates.isel(slice=selection)
        bounds = value.intervals["z"][selection]
        labels = xr.DataArray([bounds[:, 0].min(), bounds[:, 1].max()], dims="point")
        result = selected.sel(Z=labels, method="nearest")
        expected = [0, 0] if selection.stop == 1 else [0, len(values) - 1]
        assert_array_equal(result.values, expected)
        with pytest.raises(ValueError, match="outside the cells"):
            selected.sel(Z=labels - 100, method="nearest")


def test_repeated_integer_selection_keeps_rows_and_aligns_with_itself() -> None:
    array = frame_array(np.arange(3), grid()).isel(slice=[1, 1, 0])
    assert_array_equal(array.rf.grid.intervals["z"], grid().intervals["z"][[1, 1, 0]])
    assert (array + array.copy()).rf.grid == array.rf.grid
