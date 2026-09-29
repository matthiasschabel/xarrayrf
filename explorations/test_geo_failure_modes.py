"""Strict xarrayrf expectations and pinned observations from other geospatial libraries."""

from __future__ import annotations

import importlib.util

import pyproj
import pytest
import rasterio
import xarray as xr
from geo_failure_modes import _AFFINE, _PIXELS, observations
from numpy.testing import assert_allclose
from rasterio.io import MemoryFile

from xarrayrf import geotiff


def _framed(crs: str) -> xr.DataArray:
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=4, height=3, count=1, dtype="float64", crs=crs, transform=_AFFINE
        ) as dataset:
            dataset.write(_PIXELS)
        return geotiff.open(memory.name, chunks=None)


def test_xarrayrf_strided_pixel_location() -> None:
    array = _framed("EPSG:32633")
    stride = array.isel(column=slice(None, None, 2))
    expected = [rasterio.transform.xy(_AFFINE, 0, col) for col in (0, 2)]
    assert_allclose(stride.rf.geometry.points().data[0], expected, rtol=0, atol=1e-10)


def test_xarrayrf_conflicting_projected_arithmetic_refused() -> None:
    with pytest.raises(ValueError, match=r"frame|reference|conflict"):
        _ = _framed("EPSG:32633") + _framed("EPSG:32634")


def test_xarrayrf_join_keeps_frame_and_concat_refused() -> None:
    array = _framed("EPSG:32633")
    joined = xr.align(array, array.isel(column=slice(1, None)), join="inner")[0]
    assert joined.rf.reference_frame == array.rf.reference_frame
    with pytest.raises(ValueError, match="concatenation of framed arrays"):
        xr.concat([array, array], dim="column")


def test_xarrayrf_angular_override_rejected_at_import() -> None:
    for crs in ("OGC:CRS84", "EPSG:4326"):
        with pytest.raises(ValueError, match="angular coordinates are not representable yet"):
            geotiff.crs_frame(pyproj.CRS.from_user_input(crs))


@pytest.mark.xfail(
    not hasattr(xr.Index, "check_override"),
    strict=True,
    raises=pytest.fail.Exception,
    reason="this xarray lane lacks the check_override Index hook (known align_override hole)",
)
def test_xarrayrf_projected_override_refused() -> None:
    first = _framed("EPSG:32633")
    second = _framed("EPSG:32634")
    with pytest.raises(ValueError, match=r"frame|reference|conflict"):
        xr.align(first, second, join="override")


def test_other_library_observations_remain_recorded() -> None:
    expected = {
        "strided isel pixel location": "reproduced",
        "different projected CRS arithmetic": "reproduced",
        "join/concat identity": "reproduced",
        "CRS84/EPSG:4326 override": "reproduced",
        "projected CRS override": "reproduced",
    }
    for item in observations():
        if importlib.util.find_spec(item.library) is None:
            assert item.observed == "could-not-run"
        else:
            assert item.observed == expected[item.case], (
                f"{item.library} {item.version} changed {item.case}: {item.observed}"
            )
