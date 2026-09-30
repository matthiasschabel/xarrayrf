"""Public sampling queries on Grid and Geometry agree on synthetic coordinates."""

from __future__ import annotations

import copy
import json
import pickle
from typing import Any

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from numpy.testing import assert_allclose

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
    return AffineTransform(
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
        ({"offset": ("i", [0, np.nan])}, ValueError, "finite"),
    ],
)
def test_grid_constructor_refusals(coordinates: Any, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        Grid(transform(), coordinates)


def test_grid_constructor_endpoint_and_interval_refusals() -> None:
    mapping = transform()
    with pytest.raises(ValueError, match="stage 3"):
        Grid(mapping, {"offset": 0}, intervals={})
    with pytest.raises(ValueError, match="ArrayCoordinates"):
        Grid(mapping.inverse(), {"x0": 0})
    with pytest.raises(ValueError, match="ReferenceFrame"):
        Grid(
            AffineTransform(
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
        ("isel", (), {"i": True}, TypeError, "integer or slice"),
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
        {"dim": "i", "values": [0, True]},
        {"dim": "i", "values": [0, "2"]},
        {"dim": "i", "values": [[0, 2]]},
        {"dim": "i", "values": [0, 2], "unknown": 1},
        {"value": True},
        {"value": [1]},
        {"value": 1, "dim": "i"},
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
    data["value"]["intervals"] = None
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
        AffineTransform(source=mapping.source, target=other_frame, matrix=[[1]], translation=[0]),
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
    other_mapping = AffineTransform(
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
