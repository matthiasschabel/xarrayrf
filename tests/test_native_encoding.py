"""Persistence of a native DataArray binding through explicit encoding."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from numpy.testing import assert_allclose

import xarrayrf.native  # noqa: F401  # register the DataArray accessor
from xarrayrf import (
    AffineTransform,
    ArrayCoordinates,
    CoordinateSystem,
    MalformedDataError,
    MissingDecoderError,
    ReferenceFrame,
    encode,
)


@pytest.fixture
def framed() -> xr.DataArray:
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("y", "x"), ("1", "1")),
        target=ReferenceFrame.local(CoordinateSystem(("a", "b"), ("mm", "mm"))),
        matrix=np.array([[2.0, 0.5], [0.0, 3.0]]),
        translation=np.array([4.0, -1.0]),
    )
    array = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("y", "x"),
        coords={"y": [0, 2, 4], "x": [0, 3, 6, 9]},
        attrs={"note": "retained"},
    )
    framed = cast(xr.DataArray, array.rf.frame(transform, dims=("y", "x")))
    # A stale value written after framing; encode() must replace it.
    framed.attrs["xarrayrf_binding"] = "old declaration"
    return framed


@contextmanager
def pixel_tasks() -> Iterator[list[object]]:
    tasks: list[object] = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        yield tasks


def check_restored(restored: xr.DataArray, original: xr.DataArray) -> None:
    assert restored.rf.is_framed
    assert restored.rf.coordinate_transform == original.rf.coordinate_transform
    assert restored.rf.geometry_dims == original.rf.geometry_dims
    assert restored.attrs == {"note": "retained"}
    assert_allclose(
        restored.rf.geometry.points(), original.rf.geometry.points(), rtol=0, atol=1e-12
    )


def test_encoded_copy_is_canonical_and_shares_pixels(framed: xr.DataArray) -> None:
    encoded = framed.rf.encode()
    assert not encoded.rf.is_framed
    assert encoded.data is framed.data
    assert framed.attrs["xarrayrf_binding"] == "old declaration"
    payload = {
        "dims": ["y", "x"],
        "transform": encode(framed.rf.coordinate_transform),
        "intervals": {},
    }
    assert encoded.attrs == {
        "note": "retained",
        "xarrayrf_binding": json.dumps(payload, sort_keys=True, separators=(",", ":")),
    }
    restored = encoded.rf.decode()
    assert restored.data is encoded.data
    assert "xarrayrf_binding" in encoded.attrs
    check_restored(restored, framed)


def test_encoding_requires_a_binding(framed: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="unframed"):
        framed.rf.unframe().rf.encode()


def test_decode_requires_attribute_and_unframed_array(framed: xr.DataArray) -> None:
    plain = framed.rf.unframe().copy(deep=False)
    plain.attrs = {"note": "retained"}
    with pytest.raises(ValueError, match="no 'xarrayrf_binding'"):
        plain.rf.decode()
    with pytest.raises(ValueError, match="already framed"):
        framed.rf.decode()


def test_frame_refuses_an_encoded_binding_left_in_attrs(framed: xr.DataArray) -> None:
    encoded = framed.rf.encode()
    with pytest.raises(ValueError, match=r"encoded binding.*rf\.decode\(\)"):
        encoded.rf.frame(framed.rf.coordinate_transform, dims=framed.rf.geometry_dims)


def test_unframe_drops_a_stale_encoded_binding(framed: xr.DataArray) -> None:
    old = framed.rf.encode().attrs["xarrayrf_binding"]
    other = AffineTransform.from_matrix(
        source=framed.rf.coordinate_transform.source,
        target=framed.rf.reference_frame,
        matrix=np.eye(2),
        translation=np.zeros(2),
    )
    plain = framed.rf.unframe()
    plain.attrs = {}
    rebound = plain.rf.frame(other, dims=("y", "x"))
    rebound.attrs["xarrayrf_binding"] = old  # a valid but outdated declaration
    assert "xarrayrf_binding" not in rebound.rf.unframe().attrs
    with pytest.raises(ValueError, match="no 'xarrayrf_binding'"):
        rebound.rf.unframe().rf.decode()
    assert rebound.rf.assume_frame(framed.rf.reference_frame).rf.coordinate_transform == other


def test_resample_to_does_not_carry_a_stale_encoded_binding(framed: xr.DataArray) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    other = AffineTransform.from_matrix(
        source=framed.rf.coordinate_transform.source,
        target=framed.rf.reference_frame,
        matrix=np.eye(2),
        translation=np.zeros(2),
    )
    plain = framed.rf.unframe()
    plain.attrs = {}
    target = plain.rf.frame(other, dims=("y", "x"))
    result = framed.rf.resample_to(target)  # framed still has the stale attribute
    assert "xarrayrf_binding" not in result.attrs
    assert result.rf.encode().rf.decode().rf.coordinate_transform == other


@pytest.mark.parametrize(
    "payload",
    [
        42,
        "{",
        "[]",
        "{}",
        '{"dims":["y","x"],"transform":{},"extra":1}',
        '{"dims":"y","transform":{}}',
        '{"dims":["y",1],"transform":{}}',
        '{"dims":["y","x"],"transform":{}}',
    ],
)
def test_malformed_binding_is_reported(framed: xr.DataArray, payload: Any) -> None:
    array = framed.rf.unframe().assign_attrs(xarrayrf_binding=payload)
    with pytest.raises(MalformedDataError):
        array.rf.decode()


def test_non_transform_declaration_is_malformed(framed: xr.DataArray) -> None:
    payload = {"dims": ["y", "x"], "transform": encode(framed.rf.reference_frame), "intervals": {}}
    array = framed.rf.unframe().assign_attrs(xarrayrf_binding=json.dumps(payload))
    with pytest.raises(MalformedDataError, match="point transform"):
        array.rf.decode()


def test_extension_decoders_are_passed_through(framed: xr.DataArray) -> None:
    encoded = framed.rf.encode()
    payload = json.loads(encoded.attrs["xarrayrf_binding"])
    affine = payload["transform"]["value"]
    payload["transform"]["value"] = {
        "kind": "example:alias",
        "version": 1,
        "source": affine["source"],
        "target": affine["target"],
        "data": {},
    }
    encoded.attrs["xarrayrf_binding"] = json.dumps(payload)
    with pytest.raises(MissingDecoderError, match="example:alias"):
        encoded.rf.decode()
    restored = encoded.rf.decode(
        decoders={"example:alias": lambda data, **kwargs: framed.rf.coordinate_transform},
    )
    check_restored(restored, framed)


def test_decoding_revalidates_current_coordinates(framed: xr.DataArray) -> None:
    derived = framed.rf.encode().drop_vars("x")
    with pytest.raises(ValueError, match=r"coordinate|axis"):
        derived.rf.decode()


def test_netcdf_round_trip(tmp_path: Path, framed: xr.DataArray) -> None:
    pytest.importorskip("scipy")
    path = tmp_path / "binding.nc"
    framed.rf.encode().to_netcdf(path, engine="scipy")
    with xr.open_dataarray(path, engine="scipy") as raw:
        assert not raw.rf.is_framed
        check_restored(raw.rf.decode(), framed)


@pytest.mark.parametrize("anonymous", [True, False])
def test_netcdf_round_trip_combines_with_original(anonymous: bool) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    from xarrayrf import nifti

    nib = pytest.importorskip("nibabel")
    header = nib.Nifti1Header()
    header.set_data_shape((2, 3, 4))
    header.set_sform(np.eye(4), code=1)
    header.set_xyzt_units("mm")
    grid = nifti.from_header(header)
    if not anonymous:
        grid = nifti.from_header(header, frame=ReferenceFrame.local(grid.frame.coordinate_system))
    original = nifti.to_dataarray(grid, np.ones((2, 3, 4)))
    encoded = original.rf.encode().to_netcdf(engine="scipy")
    with xr.open_dataarray(io.BytesIO(encoded), engine="scipy") as raw:
        restored = raw.rf.decode()
        assert list(restored.coords) == ["i", "j", "k"]
        assert restored.rf.grid == original.rf.grid
        result = original + restored
        assert result.rf.grid == original.rf.grid
        assert_allclose(result.values, 2 * original.values, rtol=0, atol=1e-12)


def test_zarr_round_trip(tmp_path: Path, framed: xr.DataArray) -> None:
    path = tmp_path / "binding.zarr"
    framed.rf.encode().to_zarr(path)
    with xr.open_zarr(path) as dataset:
        raw = dataset["__xarray_dataarray_variable__"]
        assert not raw.rf.is_framed
        check_restored(raw.rf.decode(), framed)


def test_dask_zarr_round_trip_stays_lazy(tmp_path: Path, framed: xr.DataArray) -> None:
    path = tmp_path / "lazy.zarr"
    lazy = framed.copy(data=da.from_array(framed.data, chunks=(2, 2)))  # type: ignore[no-untyped-call]
    with pixel_tasks() as tasks:
        encoded = lazy.rf.encode()
    assert not tasks
    encoded.to_zarr(path)
    with pixel_tasks() as tasks, xr.open_zarr(path, chunks="auto") as dataset:
        raw = dataset["__xarray_dataarray_variable__"]
        assert not raw.rf.is_framed
        restored = raw.rf.decode()
        assert isinstance(restored.data, da.Array)
        assert not tasks
        assert restored.rf.coordinate_transform == lazy.rf.coordinate_transform
        assert restored.rf.geometry_dims == lazy.rf.geometry_dims
        restored.compute()
        assert tasks
        check_restored(restored, lazy)


# --- Datasets holding the arrays a binding or value references --------------------------------

FIELD_SYSTEM = CoordinateSystem(("a", "b"), ("mm", "mm"))
FIXED = ReferenceFrame.declared(("test", "fixed"), FIELD_SYSTEM)
MOVING = ReferenceFrame.declared(("test", "moving"), FIELD_SYSTEM)
FIELD_POINTS = np.random.default_rng(5).uniform(0.0, 4.0, size=(16, 2))


def displacement(source: ReferenceFrame, target: ReferenceFrame, dtype: type, seed: int) -> Any:
    from xarrayrf import DisplacementFieldTransform, Grid

    to_frame = AffineTransform.from_matrix(
        source=ArrayCoordinates(("p", "q"), ("1", "1"), sample_offset=(0.5, 0.5)),
        target=source,
        matrix=np.diag([1.0, 1.5]),
        translation=np.zeros(2),
    )
    grid = Grid(to_frame, {"p": ("p", np.arange(6)), "q": ("q", np.arange(5))})
    values: Any = np.random.default_rng(seed).normal(scale=0.3, size=(6, 5, 2)).astype(dtype)
    return DisplacementFieldTransform(
        source=source, target=target, field=values, grid=grid, outside="identity"
    )


def positions(source: ReferenceFrame, target: ReferenceFrame, seed: int) -> Any:
    from xarrayrf import Grid, PositionFieldTransform

    to_frame = AffineTransform.from_matrix(
        source=ArrayCoordinates(("p", "q"), ("1", "1")),
        target=source,
        matrix=np.diag([1.0, 1.5]),
        translation=np.zeros(2),
    )
    grid = Grid(to_frame, {"p": ("p", np.arange(6)), "q": ("q", np.arange(5))})
    jitter = np.random.default_rng(seed).normal(scale=0.1, size=(6, 5, 2))
    return PositionFieldTransform(
        source=source, target=target, field=grid.points() + jitter, grid=grid, outside="identity"
    )


@pytest.fixture
def warped() -> xr.DataArray:
    """An image bound to the moving frame through a dense field."""
    from xarrayrf import CompositeTransform

    to_fixed = AffineTransform.from_matrix(
        source=ArrayCoordinates(("y", "x"), ("1", "1")),
        target=FIXED,
        matrix=np.array([[0.5, 0.0], [0.1, 0.75]]),
        translation=np.array([0.2, 0.0]),
    )
    image = xr.DataArray(
        np.arange(20.0).reshape(4, 5),
        dims=("y", "x"),
        coords={"y": np.arange(4.0), "x": np.arange(5.0)},
        name="image",
        attrs={"note": "retained"},
    )
    transform = CompositeTransform(to_fixed, displacement(FIXED, MOVING, np.float32, 1))
    return cast(xr.DataArray, image.rf.frame(transform, dims=("y", "x")))


def assert_same_mapping(restored: Any, original: Any) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    assert_allclose(
        restored.transform_point(FIELD_POINTS),
        original.transform_point(FIELD_POINTS),
        rtol=0,
        atol=0,
    )


def write_and_open(dataset: xr.Dataset, path: Path, backend: str) -> xr.Dataset:
    if backend == "netcdf":
        pytest.importorskip("scipy")
        dataset.to_netcdf(path, engine="scipy")
        return xr.open_dataset(path, engine="scipy")
    pytest.importorskip("zarr")
    dataset.to_zarr(path, consolidated=False)
    return cast(xr.Dataset, xr.open_zarr(path, consolidated=False))


@pytest.mark.parametrize("backend", ["netcdf", "zarr"])
def test_a_field_bound_array_round_trips_through_a_dataset(
    tmp_path: Path, warped: xr.DataArray, backend: str
) -> None:
    from xarrayrf.native import array_from_dataset

    dataset = warped.rf.to_dataset()
    (reserved,) = (name for name in dataset.data_vars if name != "image")
    assert str(reserved).startswith("xarrayrf_array_")
    assert dataset[reserved].attrs["xarrayrf_reference"].startswith("sha256:")
    with write_and_open(dataset, tmp_path / f"warped.{backend}", backend) as opened:
        restored = array_from_dataset(opened)
        assert restored.rf.is_framed and restored.name == "image"
        assert restored.attrs == {"note": "retained"}
        assert "xarrayrf_array_" not in "".join(map(str, restored.coords))
        field = restored.rf.coordinate_transform.transforms[1]
        assert field.field.dtype == np.float32
        assert_same_mapping(restored.rf.coordinate_transform, warped.rf.coordinate_transform)
        assert_allclose(restored.values, warped.values, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["netcdf", "zarr"])
def test_a_declared_pair_of_position_fields_round_trips(tmp_path: Path, backend: str) -> None:
    from xarrayrf import DeclaredInversePair
    from xarrayrf.native import value_from_dataset, value_to_dataset

    pair = DeclaredInversePair(positions(FIXED, MOVING, 1), positions(MOVING, FIXED, 2))
    with write_and_open(
        value_to_dataset(pair.forward), tmp_path / f"y.{backend}", backend
    ) as opened:
        restored = value_from_dataset(opened)
        assert restored.direction == "forward"
        assert_same_mapping(restored, pair.forward)
        assert_same_mapping(restored.declared_inverse(), pair.backward)


@pytest.mark.parametrize("backend", ["netcdf", "zarr"])
def test_a_declared_pair_round_trips_through_a_dataset(tmp_path: Path, backend: str) -> None:
    from xarrayrf import DeclaredInversePair
    from xarrayrf.native import value_from_dataset, value_to_dataset

    pair = DeclaredInversePair(
        displacement(FIXED, MOVING, np.float32, 1), displacement(MOVING, FIXED, np.float64, 2)
    )
    dataset = value_to_dataset(pair.backward)
    assert len(dataset.data_vars) == 2
    with write_and_open(dataset, tmp_path / f"pair.{backend}", backend) as opened:
        restored = value_from_dataset(opened)
        assert restored.direction == "backward"
        assert (restored.source, restored.target) == (MOVING, FIXED)
        assert restored.transform.field.dtype == np.float64
        assert restored.declared_inverse().transform.field.dtype == np.float32
        assert_same_mapping(restored, pair.backward)
        assert_same_mapping(restored.declared_inverse(), pair.forward)


def test_affine_bindings_and_values_use_the_same_path(framed: xr.DataArray) -> None:
    from xarrayrf.native import array_from_dataset, value_from_dataset, value_to_dataset

    dataset = framed.rf.to_dataset(name="pixels")
    assert list(dataset.data_vars) == ["pixels"]
    check_restored(array_from_dataset(dataset), framed)
    transform = framed.rf.coordinate_transform
    assert not value_to_dataset(transform).data_vars
    assert value_from_dataset(value_to_dataset(transform)) == transform


def test_datasets_holding_one_field_merge(warped: xr.DataArray) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    from xarrayrf.native import array_from_dataset, value_from_dataset, value_to_dataset

    field = warped.rf.coordinate_transform.transforms[1]
    merged = xr.merge(
        [warped.rf.to_dataset(), value_to_dataset(field)],
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
    )
    assert len(merged.data_vars) == 2
    assert_same_mapping(value_from_dataset(merged), field)
    assert_same_mapping(
        array_from_dataset(merged).rf.coordinate_transform, warped.rf.coordinate_transform
    )


def poisoned(shape: tuple[int, ...]) -> Any:
    """A lazy array that fails if any of its chunks is computed."""

    def refuse(block: Any) -> Any:
        raise AssertionError("an unreferenced array was computed")

    return da.zeros(shape, chunks=1).map_blocks(refuse, dtype=np.float64)


def test_reading_a_dataset_loads_only_the_referenced_arrays(warped: xr.DataArray) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    from xarrayrf.native import array_from_dataset

    dataset = warped.rf.to_dataset()
    dataset["image"] = dataset["image"].copy(data=poisoned(warped.shape))
    dataset = dataset.assign(companion=xr.DataArray(poisoned((3,)), dims="u")).chunk()
    with pixel_tasks() as tasks:
        restored = array_from_dataset(dataset, name="image")
    assert tasks, "the referenced field is loaded"
    assert isinstance(restored.data, da.Array)
    assert_same_mapping(restored.rf.coordinate_transform, warped.rf.coordinate_transform)


def test_any_hashable_array_name_round_trips(warped: xr.DataArray) -> None:
    from xarrayrf.native import array_from_dataset

    dataset = warped.rename(7).rf.to_dataset()
    assert array_from_dataset(dataset).name == 7
    assert array_from_dataset(dataset, name=7).rf.is_framed


def test_a_field_binding_needs_a_store_or_a_dataset(warped: xr.DataArray) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    with pytest.raises(TypeError, match=r"rf\.to_dataset\(\)"):
        warped.rf.encode()
    arrays: dict[str, Any] = {}
    encoded = warped.rf.encode(arrays=arrays)
    assert len(arrays) == 1
    from xarrayrf import MissingArrayError

    with pytest.raises(MissingArrayError):
        encoded.rf.decode()
    assert_same_mapping(
        encoded.rf.decode(arrays=arrays).rf.coordinate_transform, warped.rf.coordinate_transform
    )


def test_to_dataset_needs_a_free_unreserved_name(warped: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="pass name="):
        warped.rename(None).rf.to_dataset()
    reference = warped.rf.to_dataset()
    (reserved,) = (str(name) for name in reference.data_vars if name != "image")
    with pytest.raises(ValueError, match="reserved"):
        warped.rf.to_dataset(name=reserved)
    clash = warped.assign_coords({reserved + "_dim0": ("x", np.arange(5))})
    with pytest.raises(ValueError, match="reserved"):
        clash.rf.to_dataset()


def test_array_from_dataset_chooses_one_unreserved_variable(warped: xr.DataArray) -> None:
    from xarrayrf.native import array_from_dataset

    dataset = warped.rf.to_dataset()
    (reserved,) = (str(name) for name in dataset.data_vars if name != "image")
    two = dataset.assign(other=xr.DataArray([1.0], dims="o"))
    with pytest.raises(ValueError, match="pass name="):
        array_from_dataset(two)
    assert array_from_dataset(two, name="image").rf.is_framed
    for name in (reserved, "missing"):
        with pytest.raises(ValueError, match="not an unreserved data variable"):
            array_from_dataset(dataset, name=name)
    with pytest.raises(TypeError, match="Dataset"):
        array_from_dataset(warped)  # type: ignore[arg-type]


def test_reserved_variables_must_declare_one_reference_each(warped: xr.DataArray) -> None:
    from xarrayrf.native import array_from_dataset, value_from_dataset, value_to_dataset

    dataset = warped.rf.to_dataset()
    (reserved,) = (str(name) for name in dataset.data_vars if name != "image")
    unlabelled = dataset.copy()
    unlabelled[reserved].attrs = {}
    with pytest.raises(MalformedDataError, match="string 'xarrayrf_reference'"):
        array_from_dataset(unlabelled)
    twice = dataset.assign({"xarrayrf_array_copy": dataset[reserved]})
    with pytest.raises(MalformedDataError, match="two reserved variables"):
        array_from_dataset(twice)
    for attrs in ({}, {"xarrayrf_value": "{"}):
        with pytest.raises(MalformedDataError):
            value_from_dataset(xr.Dataset(attrs=attrs))
    with pytest.raises(TypeError, match="Dataset"):
        value_from_dataset({})  # type: ignore[arg-type]
    assert value_to_dataset(FIXED).attrs["xarrayrf_value"]
