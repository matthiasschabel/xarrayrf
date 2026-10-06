"""Public sampling queries on Grid and Geometry agree on synthetic coordinates."""

from __future__ import annotations

import copy
import json
import pickle
import warnings
from typing import Any

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from numpy.testing import assert_allclose
from xarray.indexes import RangeIndex

import xarrayrf.native  # noqa: F401  (registers the .rf accessor)
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

ATOL = 1e-12
"""Roundoff allowance for small synthetic transforms and piecewise interpolation."""


def transform(axes: tuple[str, ...] = ("offset",)) -> AffineTransform:
    frame = ReferenceFrame.declared(
        ("synthetic", "grid"),
        CoordinateSystem(tuple(f"x{i}" for i in range(len(axes))), ("mm",) * len(axes)),
    )
    return AffineTransform.from_matrix(
        source=ArrayCoordinates(axes, ("mm",) * len(axes), sample_offset=(0.5,) * len(axes)),
        target=frame,
        matrix=np.eye(len(axes)),
        translation=np.zeros(len(axes)),
    )


def geometry(grid: Grid) -> Geometry:
    array = xr.DataArray(
        np.zeros(tuple(grid.sizes.values())), dims=grid.dims, coords=dict(grid.coordinates)
    )
    return Geometry(array, grid.transform, dims=grid.dims)


@pytest.mark.parametrize("values", [[0, 2, 4, 6], [0, 2, 5, 9], [9, 5, 2, 0]])
def test_one_dimensional_queries_agree(values: list[int]) -> None:
    grid = Grid(transform(), {"offset": ("i", values)})
    view = geometry(grid)
    assert grid.frame == view.frame
    assert grid.dims == view.dims
    assert grid.sizes == view.sizes
    assert view.grid() == grid
    assert_allclose(grid.points(), view.points(), atol=ATOL, rtol=0)
    assert_allclose(grid.point_at(i=2), view.point_at(i=2), atol=ATOL, rtol=0)
    positions = np.array([[0], [1.5], [3]])
    assert_allclose(grid.points_at(positions), view.points_at(positions), atol=ATOL, rtol=0)
    assert_allclose(
        grid.positions_at(grid.points()), view.positions_at(view.points()), atol=ATOL, rtol=0
    )
    assert grid.is_coincident(view.grid()) == view.is_coincident(geometry(grid))
    if values == [0, 2, 4, 6]:
        assert grid.lattice() == view.lattice()
    else:
        for query in (grid.lattice, view.lattice):
            with pytest.raises(ValueError, match="not uniformly spaced"):
                query()


def test_two_dimensional_structural_operations_commute_with_points() -> None:
    grid = Grid(transform(("u", "v")), {"v": ("j", [1, 3, 8]), "u": ("i", [0, 2, 5, 9])})
    view = geometry(grid)
    assert grid.dims == ("j", "i")
    assert view.grid() == grid
    assert_allclose(grid.points(), view.points(), atol=ATOL, rtol=0)
    assert_allclose(grid.point_at(i=2, j=1), view.point_at(i=2, j=1), atol=ATOL, rtol=0)
    assert grid.is_coincident(grid.transpose("i", "j"))
    assert view.is_coincident(geometry(grid.transpose("i", "j")))
    assert_allclose(grid.transpose().points(), grid.points().transpose(1, 0, 2), atol=ATOL, rtol=0)
    assert_allclose(
        grid.isel(i=slice(None, None, -2), j=slice(1, None)).points(),
        grid.points()[1:, ::-2],
        atol=ATOL,
        rtol=0,
    )
    for index in (1, -1, np.int64(2)):
        selected = grid.isel(i=index)
        assert selected.dims == ("j",)
        assert_allclose(selected.points(), grid.points()[:, index], atol=ATOL, rtol=0)
        assert_allclose(
            selected.points_at([[1.5]]), geometry(selected).points_at([[1.5]]), atol=ATOL, rtol=0
        )
    assert grid.isel(i=0, j=0).points().shape == (2,)
    assert_allclose(
        grid.isel(i=0, j=0).points_at(np.empty((3, 0))),
        np.repeat(grid.points()[0, 0][None], 3, axis=0),
        atol=ATOL,
        rtol=0,
    )
    assert grid.isel(i=slice(0, 0)).points().shape == (3, 0, 2)


@pytest.mark.parametrize("reverse", [False, True])
def test_nonuniform_forward_inverse_extrapolation_round_trip(reverse: bool) -> None:
    values = [0, 2, 5, 9]
    if reverse:
        values.reverse()
    grid = Grid(transform(), {"offset": ("i", values)})
    positions = np.array([[-1], [0], [1.5], [4]])
    expected = np.array([[13], [9], [3.5], [-2]] if reverse else [[-2], [0], [3.5], [13]])
    for query in (grid, geometry(grid)):
        points = query.points_at(positions, outside="extrapolate")
        assert_allclose(points, expected, atol=ATOL, rtol=0)
        assert_allclose(
            query.positions_at(points, outside="extrapolate"), positions, atol=ATOL, rtol=0
        )


def test_grid_is_an_immutable_exact_snapshot() -> None:
    values = np.array([0.0, 2.0, 4.0])
    grid = Grid(transform(), {"offset": ("i", values)})
    same = Grid(grid.transform, {"offset": ("i", values)})
    values[0] = 100
    assert grid == same
    assert hash(grid) == hash(same)
    assert grid != Grid(grid.transform, {"offset": ("i", [0, 2, 4.0000001])})
    assert grid != Grid(transform(), {"offset": ("j", [0, 2, 4])})
    assert grid != object()
    assert "Grid(" in repr(grid) and "'i': 3" in repr(grid)
    with pytest.raises(AttributeError):
        grid.transform = transform()  # type: ignore[misc]
    entry = grid.coordinates["offset"]
    assert isinstance(entry, tuple)
    with pytest.raises(ValueError):
        np.asarray(entry[1]).flags.writeable = True
    view = geometry(grid)
    snapshot = view.grid()
    view.array.coords["offset"] = ("i", [10, 12, 14])
    assert view.grid() != snapshot
    assert_allclose(snapshot.points(), [[0], [2], [4]], atol=ATOL, rtol=0)


def test_grid_snapshot_and_fractional_queries_read_coordinates_but_never_pixels() -> None:
    tasks: list[object] = []
    grid = Grid(transform(), {"offset": ("i", [0, 2, 5, 9])})
    array = xr.DataArray(
        da.zeros(4, chunks=2),
        dims="i",
        coords={"offset": ("i", da.from_array(np.array([0, 2, 5, 9]), chunks=2))},  # type: ignore[no-untyped-call]
    )
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        view = Geometry(array, grid.transform, dims=("i",))
        assert not tasks
        lazy = view.points()
        assert not tasks
        assert isinstance(lazy.data, da.Array)
        assert view.grid() == grid
        assert tasks
        assert not any("zeros" in str(task) for task in tasks)
        assert_allclose(view.points_at([[1.5]]), [[3.5]], atol=ATOL, rtol=0)
    array.coords["offset"].attrs["units"] = "m"
    for query in (view.grid, lambda: view.points_at([[1.5]])):
        with pytest.raises(ValueError, match="declares"):
            query()


def test_geometry_grid_refuses_coordinate_fields_and_shared_dimensions() -> None:
    mapping = transform(("u", "v"))
    field = xr.DataArray(
        np.zeros((2, 2)), dims=("i", "j"), coords={"u": (("i", "j"), np.ones((2, 2))), "v": 0}
    )
    with pytest.raises(ValueError, match="multidimensional"):
        Geometry(field, mapping, dims=("i", "j")).grid()
    shared = xr.DataArray(np.zeros(2), dims="i", coords={"u": ("i", [0, 1]), "v": ("i", [0, 1])})
    with pytest.raises(ValueError, match="more than one source axis"):
        Geometry(shared, mapping, dims=("i",)).grid()


@pytest.mark.parametrize(
    ("coordinates", "error", "match"),
    [
        (None, TypeError, "mapping"),
        ({}, ValueError, "exactly"),
        ({"offset": [1, 2, 3]}, ValueError, "scalar"),
        ({"offset": ("i", [[1, 2]])}, ValueError, "one-dimensional"),
        ({"offset": ("", [1, 2])}, ValueError, "empty"),
        ({"offset": (1, [1, 2])}, TypeError, "string"),
        ({"offset": ("i", [True, False])}, TypeError, "real"),
        ({"offset": ("i", np.ma.array([0, 1], mask=[False, True]))}, TypeError, "masked"),
        ({"offset": ("i", [[0, 1], [2]])}, ValueError, "rectangular"),
        ({"offset": ("i", [0, np.nan])}, ValueError, "finite"),
    ],
)
def test_grid_constructor_refusals(coordinates: Any, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        Grid(transform(), coordinates)


def test_grid_constructor_endpoint_refusals() -> None:
    mapping = transform()
    with pytest.raises(ValueError, match="ArrayCoordinates"):
        Grid(mapping.inverse(), {"x0": 0})
    with pytest.raises(ValueError, match="ReferenceFrame"):
        Grid(
            AffineTransform.from_matrix(
                source=mapping.source, target=mapping.source, matrix=[[1]], translation=[0]
            ),
            {"offset": 0},
        )
    with pytest.raises(ValueError, match="more than one source axis"):
        Grid(transform(("u", "v")), {"u": ("i", [0, 1]), "v": ("i", [2, 3])})


@pytest.mark.parametrize(
    ("method", "args", "kwargs", "error", "match"),
    [
        ("point_at", (), {}, ValueError, "each geometry"),
        ("point_at", (), {"i": True}, TypeError, "integer"),
        ("point_at", (), {"i": -1}, IndexError, "out of range"),
        ("isel", (), {"unknown": 0}, ValueError, "unknown"),
        ("isel", (), {"i": True}, ValueError, "Multi-dimensional indexing"),
        ("isel", (), {"i": 10}, IndexError, "out of bounds"),
        ("isel", (), {"i": slice(None, None, 0)}, ValueError, "zero"),
        ("transpose", ("other",), {}, ValueError, "each geometry"),
        ("transpose", ("i", "i"), {}, ValueError, "unique"),
        ("points_at", ([0],), {"outside": "bad"}, ValueError, "outside"),
        ("points_at", ([0],), {"domain": "bad"}, ValueError, "domain"),
        ("points_at", (0,), {}, ValueError, "shape"),
        ("points_at", ([-1],), {}, ValueError, "outside"),
        ("positions_at", ([-1],), {}, ValueError, "outside"),
        ("positions_at", ([0],), {"outside": "bad"}, ValueError, "outside"),
        ("is_coincident", (None,), {}, TypeError, "Grid"),
        ("is_coincident", (), {"tolerance": True}, TypeError, "real number"),
        ("is_coincident", (), {"tolerance": 0.5}, ValueError, "steps"),
    ],
)
def test_grid_query_refusals(
    method: str, args: tuple[Any, ...], kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    grid = Grid(transform(), {"offset": ("i", [0, 2, 5, 9])})
    if method == "is_coincident" and not args:
        args = (grid,)
    with pytest.raises(error, match=match):
        getattr(grid, method)(*args, **kwargs)


@pytest.mark.parametrize("kind", ["grid", "geometry"])
def test_fractional_domain_policies_and_single_sample_refusals(kind: str) -> None:
    grid = Grid(transform(), {"offset": ("i", [0, 2, 5, 9])})
    query = grid if kind == "grid" else geometry(grid)
    assert_allclose(
        query.points_at(np.array([[-0.5], [3.5]]), domain="cells"), [[-1], [11]], atol=ATOL, rtol=0
    )
    result = query.points_at(np.array([[-1], [1.5]]), outside="nan")
    assert np.isnan(result[0]).all()
    assert_allclose(result[1], [3.5], atol=ATOL, rtol=0)
    assert np.isnan(query.positions_at([[-2]], outside="nan")).all()
    singleton = Grid(transform(), {"offset": ("i", [2])})
    query = singleton if kind == "grid" else geometry(singleton)
    with pytest.raises(ValueError, match="single sample"):
        query.points_at([[1]], outside="extrapolate")
    with pytest.raises(ValueError, match="single sample"):
        query.points_at([[0]], domain="cells")
    empty = Grid(transform(), {"offset": ("i", [])})
    query = empty if kind == "grid" else geometry(empty)
    with pytest.raises(ValueError, match="empty coordinate"):
        query.points_at([[0]], outside="extrapolate")
    with pytest.raises(ValueError, match="empty coordinate"):
        query.positions_at([[0]])
    retained = Grid(transform(), {"offset": 2})
    query = retained if kind == "grid" else geometry(retained)
    assert_allclose(query.point_at(), [2], atol=ATOL, rtol=0)
    assert_allclose(query.points_at(np.empty((0,))), [2], atol=ATOL, rtol=0)
    with pytest.raises(ValueError, match="retained scalar"):
        query.positions_at([[2]])


@pytest.mark.parametrize("operation", ["transpose", "isel", "scalar"])
def test_grid_encoding_round_trip(operation: str) -> None:
    grid = Grid(transform(("u", "v")), {"u": ("i", [0, 2, 5]), "v": ("j", [1, 3])})
    if operation == "transpose":
        grid = grid.transpose()
    elif operation == "isel":
        grid = grid.isel(i=slice(None, None, -1))
    else:
        grid = grid.isel(i=1, j=0)
    data = encode(grid)
    assert data["value"]["kind"] == "grid"
    result = decode(json.loads(json.dumps(data, sort_keys=True)))
    assert result == grid
    assert hash(result) == hash(grid)
    assert_allclose(result.points(), grid.points(), atol=ATOL, rtol=0)


@pytest.mark.parametrize(
    "record",
    [
        {"dim": "i", "values": [0, True], "dtype": "int64"},
        {"dim": "i", "values": [0, "2"], "dtype": "float64"},
        {"dim": "i", "values": [[0, 2]], "dtype": "int64"},
        {"dim": "i", "values": [0, 2], "dtype": "int64", "unknown": 1},
        {"value": True, "dtype": "int64"},
        {"value": [1], "dtype": "float64"},
        {"value": 1, "dtype": "int64", "dim": "i"},
        {},
    ],
)
def test_grid_decode_refuses_malformed_coordinates(record: dict[str, Any]) -> None:
    data = encode(Grid(transform(), {"offset": ("i", [0, 2])}))
    data["value"]["coordinates"]["offset"] = record
    with pytest.raises(MalformedDataError):
        decode(data)


@pytest.mark.parametrize("dims", [["i"], ["j", "j"], ["i", 1], "ij"])
def test_grid_decode_refuses_dims_that_disagree_with_coordinates(dims: object) -> None:
    data = encode(Grid(transform(("u", "v")), {"u": ("i", [0, 1]), "v": ("j", [0, 1])}))
    data["value"]["dims"] = dims
    with pytest.raises(MalformedDataError):
        decode(data)


@pytest.mark.parametrize(
    "duplicate", [copy.copy, copy.deepcopy, lambda g: pickle.loads(pickle.dumps(g))]
)
def test_grid_copies_and_pickles(duplicate: Any) -> None:
    grid = Grid(transform(("u", "v")), {"u": ("i", [0, 2, 5]), "v": 1.5})
    result = duplicate(grid)
    assert result == grid
    assert hash(result) == hash(grid)


@pytest.mark.parametrize(
    "duplicate", [lambda g: g, copy.copy, copy.deepcopy, lambda g: pickle.loads(pickle.dumps(g))]
)
def test_grid_coordinate_storage_stays_immutable(duplicate: Any) -> None:
    original = Grid(
        transform(("u", "v", "w")),
        {"u": ("i", [0, 2, 5]), "v": ("j", [1.0, 3.0]), "w": 1.5},
    ).transpose()
    grid = duplicate(original)
    expected = grid.points().copy()
    expected_hash = hash(grid)
    assert grid == original
    for entry in grid.coordinates.values():
        if not isinstance(entry, tuple):
            continue
        values = np.asarray(entry[1])
        with pytest.raises(ValueError):
            values[0] = 99
        with pytest.raises(ValueError):
            values.flags.writeable = True
        # Re-typing a view in place is deprecated in NumPy 2.5, which is fine: the point is
        # that a caller who still does it cannot reach the grid's storage.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            values.dtype = np.uint8  # type: ignore[misc]
            values.shape = (values.size, 1)
        np.testing.assert_array_equal(grid.points(), expected)
        assert hash(grid) == expected_hash
        assert grid == original


@pytest.mark.parametrize("kind", ["grid", "geometry"])
@pytest.mark.parametrize("method", ["point_at", "points", "points_at"])
@pytest.mark.parametrize("invalid", ["shape", "nonfinite", "dtype"])
def test_sampling_queries_validate_transform_results(kind: str, method: str, invalid: str) -> None:
    class Invalid:
        source = transform().source
        target = transform().target

        def transform_point(self, points: Any) -> Any:
            if invalid == "shape":
                return np.zeros(3)
            shape = np.asarray(points).shape
            return np.full(shape, np.nan if invalid == "nonfinite" else "invalid")

    grid = Grid(Invalid(), {"offset": ("i", [0, 2])})
    query = grid if kind == "grid" else geometry(grid)
    error = TypeError if invalid == "dtype" else ValueError
    match = {"shape": "returned shape", "nonfinite": "finite", "dtype": "real"}[invalid]
    with pytest.raises(error, match=match):
        if method == "point_at":
            query.point_at(i=0)
        elif method == "points":
            query.points()
        else:
            query.points_at([[3]], outside="nan")


@pytest.mark.parametrize("kind", ["grid", "geometry"])
@pytest.mark.parametrize("values", [[1e16, 1.0], [-1e16, 1.0], [1e16, 1.0, -1e16]])
def test_integer_positions_preserve_stored_endpoints(kind: str, values: list[float]) -> None:
    grid = Grid(transform(), {"offset": ("i", values)})
    query = grid if kind == "grid" else geometry(grid)
    positions = np.arange(len(values), dtype=np.float64)[:, None]
    expected = np.stack([np.asarray(query.point_at(i=i)) for i in range(len(values))])
    # Integer positions must reproduce the stored endpoints without any rounding allowance.
    np.testing.assert_array_equal(query.points_at(positions), expected)


@pytest.mark.parametrize("kind", ["grid", "geometry"])
def test_extrapolation_near_the_float_limit_stays_finite(kind: str) -> None:
    grid = Grid(transform(), {"offset": ("i", [1e308, 1.1e308])})
    query = grid if kind == "grid" else geometry(grid)
    actual = query.points_at([[-1.0], [0.5]], outside="extrapolate")
    np.testing.assert_allclose(actual, [[9e307], [1.05e308]], rtol=1e-14, atol=0)


@pytest.mark.parametrize("indexer", [[2, 0], np.array([2, -1]), np.array([True, False, True])])
def test_grid_isel_matches_xarray_indexers(indexer: Any) -> None:
    grid = Grid(transform(), {"offset": ("i", [0, 2, 5])})
    selected = grid.isel(i=indexer)
    expected = geometry(grid).array.isel(i=indexer)
    assert selected == Geometry(expected, grid.transform, dims=("i",)).grid()


@pytest.mark.parametrize("label", [2, [5, 0], slice(0, 2)])
def test_grid_sel_matches_xarray_labels(label: Any) -> None:
    from xarrayrf.native import frame_array

    grid = Grid(transform(), {"offset": ("i", [0, 2, 5])})
    expected = frame_array(np.zeros(3), grid).sel(offset=label).rf.grid
    assert grid.sel(offset=label) == expected


@pytest.mark.parametrize("kind", ["grid", "geometry"])
@pytest.mark.parametrize("domain", ["samples", "cells"])
@pytest.mark.parametrize("outside", ["raise", "nan", "extrapolate"])
def test_empty_queries_on_empty_sampling(kind: str, domain: Any, outside: Any) -> None:
    grid = Grid(transform(("u", "v")), {"u": ("i", []), "v": ("j", [0, 1])})
    query = grid if kind == "grid" else geometry(grid)
    assert query.points_at(np.empty((0, 2)), domain=domain, outside=outside).shape == (0, 2)
    assert query.positions_at(np.empty((0, 2)), domain=domain, outside=outside).shape == (0, 2)


def test_grid_snapshot_lattice_checks_range_index_values() -> None:
    index = RangeIndex.arange(1e9, 1e9 + 3, 0.1, coord_name="offset", dim="i")
    array = xr.DataArray(np.zeros(index.size), dims="i", coords=xr.Coordinates.from_xindex(index))
    view = Geometry(array, transform(), dims=("i",))
    assert_allclose(view.lattice().matrix, [[0.1]], atol=ATOL, rtol=0)
    with pytest.raises(ValueError, match="not uniformly spaced"):
        view.grid().lattice()


def test_integer_only_sequences_avoid_float_promotion() -> None:
    values = [np.int64(2**53 + 1), np.uint64(2**53 + 3)]
    grid = Grid(transform(), {"offset": ("i", values)})
    entry = grid.coordinates["offset"]
    assert isinstance(entry, tuple)
    assert np.asarray(entry[1]).dtype == np.int64
    np.testing.assert_array_equal(entry[1], np.array([2**53 + 1, 2**53 + 3], dtype=np.int64))


@pytest.mark.parametrize("values", [[-1, 2**63], [np.int64(2**63 - 1), np.uint64(2**63)]])
def test_mixed_signed_integer_sequence_overflow_is_refused(values: Any) -> None:
    with pytest.raises(ValueError, match="fit in int64"):
        Grid(transform(), {"offset": ("i", values)})


def test_grid_refuses_two_coordinates_on_one_dimension() -> None:
    with pytest.raises(ValueError, match="more than one source axis"):
        Grid(transform(("u", "v")), {"u": ("i", [0, 1]), "v": ("i", [0, 1])})


def test_grid_names_the_tuple_form_for_a_listed_pair() -> None:
    with pytest.raises(TypeError, match="pass it as a tuple"):
        Grid(transform(), {"offset": ["i", [0, 1]]})  # type: ignore[dict-item]


def test_grid_decode_wraps_constructor_errors_and_refuses_unknown_fields() -> None:
    data = encode(Grid(transform(), {"offset": 2}))
    data["value"]["coordinates"] = {}
    with pytest.raises(MalformedDataError, match="exactly") as caught:
        decode(data)
    assert isinstance(caught.value.__cause__, ValueError)
    data = encode(Grid(transform(), {"offset": 2}))
    data["value"]["extra"] = None
    with pytest.raises(MalformedDataError, match="extra"):
        decode(data)


def test_retained_scalar_lattice_and_snapshot_dimension_order() -> None:
    grid = Grid(transform(("u", "v")), {"u": ("i", [0, 2, 4]), "v": ("j", [1, 3, 5])})
    retained = grid.isel(i=1)
    view = geometry(retained)
    assert retained.lattice() == view.lattice()
    assert retained.isel(j=1).lattice() == geometry(retained.isel(j=1)).lattice()
    array = geometry(grid).array
    view = Geometry(array, grid.transform, dims=("j", "i"))
    snapshot = view.grid()
    assert snapshot.dims == view.dims
    assert_allclose(snapshot.points(), view.points().transpose("j", "i", "axis"), atol=ATOL, rtol=0)
    positions = [[1.5, 0.5]]
    assert_allclose(snapshot.points_at(positions), view.points_at(positions), atol=ATOL, rtol=0)


def test_fractional_queries_reject_coordinate_fields_and_shared_dimensions() -> None:
    mapping = transform(("u", "v"))
    field = xr.DataArray(
        np.zeros((2, 2)), dims=("i", "j"), coords={"u": (("i", "j"), np.ones((2, 2))), "v": 0}
    )
    with pytest.raises(ValueError, match="multidimensional"):
        Geometry(field, mapping, dims=("i", "j")).points_at([[0, 0]])
    shared = xr.DataArray(np.zeros(2), dims="i", coords={"u": ("i", [0, 1]), "v": ("i", [0, 1])})
    with pytest.raises(ValueError, match="more than one"):
        Geometry(shared, mapping, dims=("i",)).points_at([[0]])


def test_grid_equality_and_hash_normalize_signed_zero() -> None:
    mapping = transform()
    positive = Grid(mapping, {"offset": ("i", [0.0, 1.0])})
    negative = Grid(mapping, {"offset": ("i", [-0.0, 1.0])})
    assert positive == negative
    assert hash(positive) == hash(negative)
    assert isinstance(mapping.target, ReferenceFrame)
    other_frame = ReferenceFrame.local(mapping.target.coordinate_system)
    other = Grid(
        AffineTransform.from_matrix(
            source=mapping.source, target=other_frame, matrix=[[1]], translation=[0]
        ),
        {"offset": ("i", [0.0, 1.0])},
    )
    assert positive != other
    assert not positive.is_coincident(other)


def test_coincidence_frame_refusal_does_not_compute_chunked_coordinates() -> None:
    mapping = transform()
    array = xr.DataArray(
        da.zeros(4, chunks=2), dims="i", coords={"offset": ("i", da.arange(4, chunks=2))}
    )
    assert isinstance(mapping.target, ReferenceFrame)
    other_mapping = AffineTransform.from_matrix(
        source=mapping.source,
        target=ReferenceFrame.local(mapping.target.coordinate_system),
        matrix=[[1]],
        translation=[0],
    )
    tasks: list[object] = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        assert not Geometry(array, mapping, dims=("i",)).is_coincident(
            Geometry(array, other_mapping, dims=("i",))
        )
    assert not tasks


@pytest.mark.parametrize("dtype", [np.int8, np.uint16, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("empty", [False, True])
def test_coordinate_kinds_round_trip_exactly(dtype: Any, empty: bool) -> None:
    values = np.array([] if empty else [0, 2, 4], dtype=dtype)
    grid = Grid(transform(), {"offset": ("i", values)})
    entry = grid.coordinates["offset"]
    assert isinstance(entry, tuple)
    expected_dtype = np.int64 if values.dtype.kind in "iu" else np.float64
    assert np.asarray(entry[1]).dtype == expected_dtype
    with pytest.raises(ValueError):
        np.asarray(entry[1]).flags.writeable = True
    encoded = json.loads(json.dumps(encode(grid)))
    result = decode(encoded)
    assert result == grid
    assert hash(result) == hash(grid)
    if not empty:
        scalar = grid.isel(i=1)
        assert type(scalar.coordinates["offset"]) is (int if expected_dtype is np.int64 else float)
        assert decode(json.loads(json.dumps(encode(scalar)))) == scalar


def test_integer_coordinates_above_float_precision_stay_exact() -> None:
    values = np.array([2**53 + 1, 2**53 + 3], dtype=np.int64)
    grid = Grid(transform(), {"offset": ("i", values)})
    encoded = encode(grid)
    assert encoded["value"]["coordinates"]["offset"]["values"] == values.tolist()
    assert decode(json.loads(json.dumps(encoded))) == grid
    assert grid != Grid(grid.transform, {"offset": ("i", values.astype(np.float64))})
    floating = Grid(grid.transform, {"offset": ("i", [0.0, 2.0])})
    integer = Grid(grid.transform, {"offset": ("i", [0, 2])})
    assert integer != floating
    assert len({integer, floating}) == 2


def test_grid_integer_overflow_is_refused() -> None:
    with pytest.raises(ValueError, match="fit in int64"):
        Grid(transform(), {"offset": ("i", np.array([2**63], dtype=np.uint64))})
    encoded = encode(Grid(transform(), {"offset": ("i", [0, 2])}))
    encoded["value"]["coordinates"]["offset"]["values"] = [2**63]
    with pytest.raises(MalformedDataError, match="fit in int64"):
        decode(encoded)


@pytest.mark.parametrize(
    "record",
    [
        {"dim": "i", "values": [0.0], "dtype": "int64"},
        {"dim": "i", "values": [2**63], "dtype": "int64"},
        {"dim": "i", "values": [-(2**63) - 1], "dtype": "int64"},
        {"dim": "i", "values": [], "dtype": "int32"},
        {"dim": "i", "values": []},
        {"value": 0.0, "dtype": "int64"},
        {"value": 2**63, "dtype": "int64"},
        {"value": 0, "dtype": "int32"},
        {"value": 0},
        {"value": float("inf"), "dtype": "float64"},
        {"dim": "i", "values": [float("nan")], "dtype": "float64"},
        {"value": 10**400, "dtype": "float64"},
    ],
)
def test_coordinate_dtype_metadata_is_required_and_validated(record: dict[str, Any]) -> None:
    encoded = encode(Grid(transform(), {"offset": ("i", [])}))
    encoded["value"]["coordinates"]["offset"] = record
    with pytest.raises(MalformedDataError):
        decode(encoded)


@pytest.mark.parametrize("scalar", [False, True])
def test_float_grid_survives_integral_json_number_rewriting(scalar: bool) -> None:
    grid = Grid(transform(), {"offset": ("i", [0.0, 2.0])})
    if scalar:
        grid = grid.isel(i=0)
    encoded = encode(grid)
    assert encoded["value"]["coordinates"]["offset"]["dtype"] == "float64"
    rewritten = json.dumps(encoded).replace("0.0", "0").replace("2.0", "2")
    result = decode(json.loads(rewritten))
    assert result == grid
    assert hash(result) == hash(grid)


@pytest.mark.parametrize("selection", ["full", "retained", "scalar", "empty", "transposed"])
def test_every_grid_door_has_the_same_sampling(selection: str) -> None:
    from xarrayrf.native import frame_array, grid_coordinates

    grid = Grid(transform(("u", "v")), {"u": ("i", [0, 2, 5]), "v": ("j", [1.0, 3.0])})
    if selection == "retained":
        grid = grid.isel(i=1)
    elif selection == "scalar":
        grid = grid.isel(i=1, j=0)
    elif selection == "empty":
        grid = grid.isel(i=slice(0, 0))
    elif selection == "transposed":
        grid = grid.transpose()
    data = np.zeros(tuple(grid.sizes.values()))
    plain = xr.DataArray(data, dims=grid.dims)
    arrays = [
        frame_array(data, grid),
        plain.rf.frame(grid),
        plain.assign_coords(grid_coordinates(grid)),
    ]
    for array in arrays:
        assert array.data is data
        assert array.rf.grid == grid
        assert_allclose(array.rf.geometry.points(), grid.points(), atol=ATOL, rtol=0)
        assert array.rf.coordinate_transform == grid.transform
        assert array.rf.geometry_dims == grid.dims


@pytest.mark.parametrize("door", ["transform", "grid", "frame_array", "grid_coordinates"])
def test_framing_doors_order_bound_coordinates_canonically(door: str) -> None:
    from xarrayrf.native import CoordinateSpec, frame_array, grid_coordinates

    grid = Grid(
        transform(("v", "u", "fixed_b", "fixed_a")),
        {"fixed_a": 5, "v": ("j", [1, 3]), "u": ("i", [0, 2, 5]), "fixed_b": 7},
    ).transpose("i", "j")
    extras: dict[str, CoordinateSpec] = {
        "context_b": ((), 7, {"description": "first context"}),
        "context_a": ((), 9, {"description": "second context"}),
        "labels": ("j", np.array([10, 20]), {"description": "sample labels"}),
    }
    plain = xr.DataArray(
        np.zeros((3, 2)),
        dims=grid.dims,
        coords={
            "context_b": extras["context_b"],
            "v": grid.coordinates["v"],
            "context_a": extras["context_a"],
            "fixed_a": grid.coordinates["fixed_a"],
            "u": grid.coordinates["u"],
            "fixed_b": grid.coordinates["fixed_b"],
            "labels": extras["labels"],
        },
        attrs={"note": "pixels"},
        name="signal",
    )
    plain.encoding["dtype"] = "float32"
    if door == "transform":
        result = plain.rf.frame(grid.transform, dims=grid.dims)
    elif door == "grid":
        result = plain.rf.frame(grid)
    elif door == "frame_array":
        result = frame_array(plain.data, grid, coords=extras)
    else:
        coordinates = grid_coordinates(grid)
        assert list(coordinates) == ["u", "v", "fixed_b", "fixed_a"]
        result = plain.drop_vars(list(grid.coordinates)).assign_coords(coordinates)
    assert [name for name in result.coords if name in grid.coordinates] == [
        "u",
        "v",
        "fixed_b",
        "fixed_a",
    ]
    assert [name for name in result.coords if name not in grid.coordinates] == list(extras)
    for name in extras:
        assert result.coords[name].variable.identical(plain.coords[name].variable)
    assert result.data is plain.data
    assert result.rf.grid == grid
    if door in ("transform", "grid"):
        assert result.name == plain.name
        assert result.attrs == plain.attrs
        assert result.encoding == plain.encoding


def test_integer_binding_snapshot_materializes_dtype_units_and_existing_attrs() -> None:
    from xarrayrf.native import frame_array, index_coordinate

    mapping = transform(("i",)).with_endpoints(source=ArrayCoordinates(("i",), ("1",)))
    coord = index_coordinate("i", 3, start=2**53 + 1)
    original = xr.DataArray(np.zeros(3), dims="i", coords={"i": coord}).rf.frame(
        mapping, dims=("i",)
    )
    snapshot = original.rf.grid
    restored = frame_array(original.data, snapshot)
    assert restored.rf.grid == snapshot
    assert restored.coords["i"].dtype == np.int64
    np.testing.assert_array_equal(restored.coords["i"], original.coords["i"])
    assert restored.coords["i"].attrs == {"units": "1"}
    plain = original.rf.unframe()
    plain.coords["i"].attrs["description"] = "sample labels"
    reframed = plain.rf.frame(snapshot)
    assert reframed.coords["i"].variable.identical(plain.coords["i"].variable)
    assert reframed.rf.grid == snapshot


@pytest.mark.parametrize("existing", [[9, 10, 11], [0.0, 2.0, 5.0]])
def test_grid_frame_replaces_different_values_or_dtype_kind(existing: list[float]) -> None:
    grid = Grid(transform(), {"offset": ("i", [0, 2, 5])})
    plain = xr.DataArray(np.zeros(3), dims="i", coords={"offset": ("i", existing)})
    plain.coords["offset"].attrs["description"] = "old labels"
    before = plain.copy(deep=True)
    with pytest.raises(ValueError, match=r"offset.*replace_coordinates=True"):
        plain.rf.frame(grid)
    xr.testing.assert_identical(plain, before)
    result = plain.rf.frame(grid, replace_coordinates=True)
    assert result.data is plain.data
    assert result.rf.grid == grid
    assert result.coords["offset"].dtype == np.int64
    assert result.coords["offset"].attrs == {"units": "mm"}


@pytest.mark.parametrize("multichannel", [False, True])
def test_grid_doors_keep_lazy_nongeometry_dimensions_and_coordinates(multichannel: bool) -> None:
    from xarrayrf.native import CoordinateSpec, frame_array, grid_coordinates

    axes = ("y", "x") if multichannel else ("i", "j", "k")
    grid = Grid(transform(axes), {axis: (axis, np.arange(3)) for axis in axes})
    context_dim = "channel" if multichannel else "time"
    dims = (context_dim, *axes) if multichannel else (*axes, context_dim)
    other_coords: dict[str, CoordinateSpec] = {
        context_dim: (context_dim, np.array([0.25, 0.75]), {"units": "s"}),
        "context": ((), 7, {"description": "retained context"}),
    }
    shape = tuple(3 if dim in axes else 2 for dim in dims)
    data = da.zeros(shape, chunks=1)
    plain = xr.DataArray(data, dims=dims, coords=other_coords, attrs={"description": "pixels"})
    tasks: list[object] = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        arrays = [
            frame_array(data, grid, dims=dims, coords=other_coords, attrs=plain.attrs),
            plain.rf.frame(grid),
            plain.assign_coords(grid_coordinates(grid)),
        ]
        for array in arrays:
            assert array.data is data
            assert array.dims == dims
            assert array.rf.grid == grid
            assert array.attrs == plain.attrs
            for name in other_coords:
                xr.testing.assert_identical(array.coords[name], plain.coords[name])
        assert not tasks


def test_undeclared_grid_units_do_not_create_coordinate_attrs() -> None:
    from xarrayrf.native import frame_array, grid_coordinates

    mapping = transform().with_endpoints(source=ArrayCoordinates(("offset",), (None,)))
    grid = Grid(mapping, {"offset": ("i", [0, 1])})
    assert grid_coordinates(grid)["offset"].attrs == {}
    assert frame_array(np.zeros(2), grid).coords["offset"].attrs == {}


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_resampling_to_grid_equals_framed_array_target(nonuniform: bool, method: Any) -> None:
    from xarrayrf import resample
    from xarrayrf.native import frame_array

    mapping = transform(("y", "x"))
    source_grid = Grid(
        mapping,
        {"y": ("y", [0, 2, 5, 9] if nonuniform else [0, 2, 4, 6]), "x": ("x", [0, 1, 2, 3])},
    )
    source = frame_array(
        np.arange(32).reshape(2, 4, 4),
        source_grid,
        dims=("channel", "y", "x"),
        coords={"channel": ("channel", np.array([2, 4]), {"description": "channels"})},
        attrs={"description": "source"},
    )
    source.name = "signal"
    target_grid = Grid(
        mapping, {"y": ("row", [1.0, 3.0, 5.0]), "x": ("col", [0.5, 1.5])}
    ).transpose()
    target = frame_array(np.zeros((2, 3)), target_grid)
    by_grid = source.rf.resample_to(target_grid, method=method)
    by_array = source.rf.resample_to(target, method=method)
    xr.testing.assert_identical(by_grid, by_array)
    assert by_grid.rf.grid == target_grid
    xr.testing.assert_identical(by_grid.coords["channel"], source.coords["channel"])
    core_grid = resample(source.rf.geometry, target_grid, method=method, block_points=3)
    core_array = resample(source.rf.geometry, target.rf.geometry, method=method, block_points=3)
    xr.testing.assert_identical(core_grid, core_array)
    assert not core_grid.rf.is_framed


@pytest.mark.parametrize("dims", [("i",), None])
def test_grid_frame_refuses_dims_argument(dims: Any) -> None:
    grid = Grid(transform(), {"offset": ("i", [0, 1])})
    with pytest.raises(TypeError, match="dims must not be supplied"):
        xr.DataArray(np.zeros(2), dims="i").rf.frame(grid, dims=dims)


@pytest.mark.parametrize("dims,shape", [("j", (2,)), ("i", (3,))])
def test_grid_frame_refuses_missing_or_wrong_sized_dimensions(dims: str, shape: tuple[int]) -> None:
    grid = Grid(transform(), {"offset": ("i", [0, 1])})
    with pytest.raises(ValueError, match="must be an array dimension of size"):
        xr.DataArray(np.zeros(shape), dims=dims).rf.frame(grid)


def test_grid_frame_refuses_existing_or_encoded_binding() -> None:
    from xarrayrf.native import frame_array

    grid = Grid(transform(), {"offset": ("i", [0, 1])})
    array = frame_array(np.zeros(2), grid)
    with pytest.raises(ValueError, match="already framed"):
        array.rf.frame(grid)
    with pytest.raises(ValueError, match="encoded binding"):
        array.rf.encode().rf.frame(grid)
    with pytest.raises(ValueError, match="encoded binding"):
        frame_array(np.zeros(2), grid, attrs={"xarrayrf_binding": "stale"})


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"coords": {"offset": ("i", np.array([0, 1]), {})}}, ValueError, "redefine grid"),
        ({"coords": []}, TypeError, "coords must be a mapping"),
        ({"dims": ("j",)}, ValueError, "include grid dimensions"),
        ({"dims": ("i", "time")}, ValueError, "no coordinate declares"),
        ({"dims": ("i", "i")}, ValueError, "must be unique"),
        ({"dims": "i"}, TypeError, "sequence"),
    ],
)
def test_frame_array_refusals(kwargs: dict[str, Any], error: type[Exception], match: str) -> None:
    from xarrayrf.native import frame_array

    grid = Grid(transform(), {"offset": ("i", [0, 1])})
    with pytest.raises(error, match=match):
        frame_array(np.zeros(2), grid, **kwargs)


def test_frame_array_refuses_pixel_shape_mismatch() -> None:
    from xarrayrf.native import frame_array

    grid = Grid(transform(), {"offset": ("i", [0, 1])})
    with pytest.raises(ValueError, match=r"data shape.*geometry shape"):
        frame_array(np.zeros(3), grid)


def test_frame_array_refuses_wrong_sized_nongeometry_coordinate_on_grid_dim() -> None:
    from xarrayrf.native import frame_array

    grid = Grid(transform(), {"offset": ("i", [0, 1])})
    with pytest.raises(
        ValueError, match="coordinate 'label' along grid dimension 'i' has length 3; expected 2"
    ):
        frame_array(np.zeros(2), grid, coords={"label": ("i", np.arange(3), {})})


def test_grid_coordinates_requires_grid() -> None:
    from xarrayrf.native import frame_array, grid_coordinates

    for call in (lambda: grid_coordinates(3), lambda: frame_array(np.zeros(2), 3)):  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="grid must be a Grid"):
            call()


def test_resampling_to_grid_keeps_pixels_lazy_and_context_coords_exact() -> None:
    from xarrayrf.native import frame_array

    grid = Grid(transform(), {"offset": ("i", [0, 2, 4, 6])})
    source = frame_array(
        da.zeros((4, 2), chunks=(2, 1)),
        grid,
        dims=("i", "time"),
        coords={"time": ("time", np.array([0.25, 0.75]), {"units": "s"})},
    )
    target_grid = Grid(grid.transform, {"offset": ("j", [1.0, 3.0, 5.0])})
    target = frame_array(da.zeros(3, chunks=1), target_grid)
    tasks: list[object] = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        result = source.rf.resample_to(target_grid)
        equivalent = source.rf.resample_to(target)
        assert isinstance(result.data, da.Array)
        assert result.rf.grid == target_grid
        xr.testing.assert_identical(result.coords["time"], source.coords["time"])
        assert not tasks
    xr.testing.assert_identical(result.compute(), equivalent.compute())


def test_grid_property_refuses_unframed_array() -> None:

    with pytest.raises(ValueError, match="array is unframed"):
        _ = xr.DataArray(np.zeros(2), dims="i").rf.grid


def test_grid_frame_preserves_retained_scalar_attrs() -> None:
    from xarrayrf.native import frame_array

    grid = Grid(transform(("u", "v")), {"u": ("i", [0, 2]), "v": 3})
    plain = frame_array(np.zeros(2), grid).rf.unframe()
    plain.coords["v"].attrs["description"] = "selected slice"
    result = plain.rf.frame(grid)
    assert result.coords["v"].variable.identical(plain.coords["v"].variable)
    assert result.rf.grid == grid


def test_grid_frame_validates_units_on_preserved_coordinates() -> None:
    grid = Grid(transform(), {"offset": ("i", [0, 2])})
    plain = xr.DataArray(np.zeros(2), dims="i", coords=dict(grid.coordinates))
    plain.coords["offset"].attrs["units"] = "m"
    with pytest.raises(ValueError, match=r"offset.*declares.*'m'.*'mm'.*replace_coordinates=True"):
        plain.rf.frame(grid)


@pytest.mark.parametrize("unit", ["m", 123])
def test_grid_frame_unit_conflicts_require_explicit_replacement(unit: object) -> None:
    grid = Grid(transform(), {"offset": ("i", [0, 2])})
    plain = xr.DataArray(np.zeros(2), dims="i", coords=dict(grid.coordinates))
    plain.coords["offset"].attrs = {"units": unit, "description": "stale"}
    before = plain.copy(deep=True)
    error = TypeError if isinstance(unit, int) else ValueError
    with pytest.raises(error, match=r"offset.*replace_coordinates=True") as caught:
        plain.rf.frame(grid)
    if error is TypeError:
        assert isinstance(caught.value.__cause__, TypeError)
    xr.testing.assert_identical(plain, before)
    result = plain.rf.frame(grid, replace_coordinates=True)
    assert result.rf.grid == grid
    assert result.coords["offset"].attrs == {"units": "mm"}
    assert result.data is plain.data


@pytest.mark.parametrize(
    ("dims", "values", "reason"),
    [
        ("time", [0, 2], r"dims.*time.*i"),
        ("i", [0.0, 2.0], r"dtype kind 'f'.*'i'"),
        ("i", [0, 3], "values differ"),
    ],
)
def test_grid_frame_reports_coordinate_conflict_reason(
    dims: str, values: list[int] | list[float], reason: str
) -> None:
    grid = Grid(transform(), {"offset": ("i", [0, 2])})
    plain = xr.DataArray(np.zeros((2, 2)), dims=("i", "time"), coords={"offset": (dims, values)})
    with pytest.raises(ValueError, match=rf"offset.*{reason}.*replace_coordinates=True"):
        plain.rf.frame(grid)


def test_grid_frame_replaces_conflicting_indexed_dimension_coordinate() -> None:
    grid = Grid(transform(("i",)), {"i": ("i", [0, 2])})
    plain = xr.DataArray(np.zeros(2), dims="i", coords={"i": [0, 3]})
    before = plain.copy(deep=True)
    original_index = plain.xindexes["i"]
    with pytest.raises(ValueError, match=r"i.*values differ.*replace_coordinates=True"):
        plain.rf.frame(grid)
    xr.testing.assert_identical(plain, before)
    assert plain.xindexes["i"] is original_index

    result = plain.rf.frame(grid, replace_coordinates=True)
    assert result.rf.grid == grid
    assert result.xindexes["i"] is not original_index
    assert result.data is plain.data
    xr.testing.assert_identical(plain, before)
    assert plain.xindexes["i"] is original_index


def test_grid_frame_replaces_scalar_and_dimension_conflicts_without_computing_pixels() -> None:
    grid = Grid(
        transform(("u", "v")), {"u": ("i", [0, 2]), "v": 3}, intervals={"u": [[-1, 1], [1, 3]]}
    )
    pixels = da.zeros((2, 2), chunks=1)
    plain = xr.DataArray(
        pixels,
        dims=("i", "time"),
        coords={"u": ("time", [0, 2]), "v": 99, "time": [4, 5], "label": "untouched"},
        attrs={"description": "image"},
    )
    before = plain.copy(deep=True)
    indexes = dict(plain.xindexes)
    tasks: list[object] = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        with pytest.raises(ValueError, match=r"u.*v.*replace_coordinates=True"):
            plain.rf.frame(grid)
        result = plain.rf.frame(grid, replace_coordinates=True)
    assert not tasks
    xr.testing.assert_identical(plain, before)
    assert dict(plain.xindexes) == indexes
    assert result.data is pixels
    assert result.data.__dask_graph__() is pixels.__dask_graph__()
    assert result.rf.grid == grid
    assert result.time.variable.identical(plain.time.variable)
    assert result.label.item() == "untouched"
    assert result.attrs == plain.attrs


@pytest.mark.parametrize("replace", [False, True])
def test_grid_frame_supplies_missing_coordinates_and_preserves_compatible_attrs(
    replace: bool,
) -> None:
    grid = Grid(transform(("u", "v")), {"u": ("i", [0, 2]), "v": 3})
    plain = xr.DataArray(np.zeros(2), dims="i", coords={"u": ("i", [0, 2])})
    plain.u.attrs["description"] = "no explicit unit"
    result = plain.rf.frame(grid, replace_coordinates=replace)
    assert result.u.attrs == plain.u.attrs
    assert result.v.item() == 3
    assert result.rf.grid == grid


@pytest.mark.parametrize("flag", [None, 0, 1, "yes", np.bool_(True)])
def test_grid_frame_requires_boolean_replacement_flag(flag: Any) -> None:
    grid = Grid(transform(), {"offset": ("i", [0, 2])})
    with pytest.raises(TypeError, match="replace_coordinates must be a bool"):
        xr.DataArray(np.zeros(2), dims="i").rf.frame(grid, replace_coordinates=flag)


def test_transform_frame_refuses_coordinate_replacement() -> None:
    grid = Grid(transform(), {"offset": ("i", [0, 2])})
    plain = xr.DataArray(np.zeros(2), dims="i", coords=dict(grid.coordinates))
    with pytest.raises(ValueError, match="requires a Grid"):
        plain.rf.frame(grid.transform, dims=grid.dims, replace_coordinates=True)
