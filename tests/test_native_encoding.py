"""Persistence of a native DataArray binding through explicit encoding."""

from __future__ import annotations

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
    transform = AffineTransform(
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
    other = AffineTransform(
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
    other = AffineTransform(
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
