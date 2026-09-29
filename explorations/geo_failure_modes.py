"""Reproduce selected geospatial identity and pixel-location failure modes.

This is a development comparison, not a claim that xarrayrf replaces these engines.
A changed other-library result is reported as ``fixed`` and causes the paired test
expecting the pinned observation to fail.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyproj
import rasterio
import xarray as xr
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from xarrayrf import geotiff


@dataclass(frozen=True)
class Observation:
    """One measured comparison result."""

    case: str
    library: str
    version: str
    observed: str
    xarrayrf: str


_AFFINE = Affine(2, 0, 100, 0, -3, 200)
_PIXELS = np.arange(12, dtype=np.float64).reshape(1, 3, 4)


def _framed(crs: str = "EPSG:32633") -> xr.DataArray:
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=4, height=3, count=1, dtype="float64", crs=crs, transform=_AFFINE
        ) as dataset:
            dataset.write(_PIXELS)
        return geotiff.open(memory.name, chunks=None)


def _rioxarray_array(crs: str = "EPSG:32633") -> xr.DataArray:
    import rioxarray  # noqa: F401 - registers the rio accessor

    x = np.array([rasterio.transform.xy(_AFFINE, 0, col)[0] for col in range(4)])
    y = np.array([rasterio.transform.xy(_AFFINE, row, 0)[1] for row in range(3)])
    return (
        xr.DataArray(_PIXELS.copy(), dims=("band", "y", "x"), coords={"band": [1], "y": y, "x": x})
        .rio.write_crs(crs)
        .rio.write_transform(_AFFINE)
    )


def _rioxarray_stride() -> bool:
    array = _rioxarray_array()
    stride = array.isel(x=slice(None, None, 2))
    # The second retained sample is source column 2. A stored transform still maps
    # it as column 1, whereas the coordinate-derived transform follows the stride.
    stored_x = rasterio.transform.xy(stride.rio.transform(recalc=False), 0, 1)[0]
    truth_x = rasterio.transform.xy(_AFFINE, 0, 2)[0]
    return not np.isclose(stored_x, truth_x, rtol=0, atol=1e-10)


def _rioxarray_mismatch() -> bool:
    first = _rioxarray_array("EPSG:32633")
    second = _rioxarray_array("EPSG:32634")
    result = first + second
    return result.rio.crs == first.rio.crs


def _rasterix_identity() -> bool:
    from rasterix import RasterIndex

    first = RasterIndex.from_transform(_AFFINE, width=4, height=3, crs="EPSG:32633")
    second = RasterIndex.from_transform(_AFFINE, width=4, height=3, crs="EPSG:32633")
    joined = first.join(second, how="inner")
    adjacent = RasterIndex.from_transform(
        _AFFINE @ Affine.translation(4, 0), width=4, height=3, crs="EPSG:32633"
    )
    concatenated = RasterIndex.concat([first, adjacent], dim="x")
    other = RasterIndex.from_transform(_AFFINE, width=4, height=3, crs="EPSG:32634")
    # A CRS-less index compares equal to a differently framed one.
    missing_matches = joined.equals(other)
    return joined.crs is None and concatenated.crs is None and missing_matches


def _xproj_override(first_crs: str, second_crs: str) -> bool:
    import xproj  # noqa: F401 - registers the proj accessor

    first = xr.DataArray(np.array([1.0]), dims=("x",), coords={"x": [0]}).proj.assign_crs(
        spatial_ref=first_crs
    )
    second = xr.DataArray(np.array([2.0]), dims=("x",), coords={"x": [0]}).proj.assign_crs(
        spatial_ref=second_crs
    )
    aligned = xr.align(first, second, join="override")[1]
    return aligned.proj.crs == pyproj.CRS.from_user_input(first_crs)


def _xarrayrf_projected_override() -> str:
    first = _framed("EPSG:32633")
    second = _framed("EPSG:32634")
    try:
        aligned = xr.align(first, second, join="override")[1]
    except ValueError:
        return "refused"
    if aligned.rf.reference_frame == first.rf.reference_frame:
        return "incorrectly rebound EPSG:32634 to EPSG:32633"
    return "accepted; binding unchanged"


def _observe(
    case: str, library: str, check: Callable[[], bool], xarrayrf_outcome: str
) -> Observation:
    try:
        module = importlib.import_module(library)
    except ImportError:
        return Observation(case, library, "absent", "could-not-run", xarrayrf_outcome)
    reproduced = check()
    return Observation(
        case,
        library,
        str(module.__version__),
        "reproduced" if reproduced else "fixed",
        xarrayrf_outcome,
    )


def observations() -> tuple[Observation, ...]:
    """Measure pinned library outcomes and state the corresponding xarrayrf behavior."""
    return (
        _observe(
            "strided isel pixel location", "rioxarray", _rioxarray_stride, "pixel centre retained"
        ),
        _observe("different projected CRS arithmetic", "rioxarray", _rioxarray_mismatch, "refused"),
        _observe(
            "join/concat identity",
            "rasterix",
            _rasterix_identity,
            "join keeps frame; framed concat refused",
        ),
        _observe(
            "CRS84/EPSG:4326 override",
            "xproj",
            lambda: _xproj_override("OGC:CRS84", "EPSG:4326"),
            "rejected at import (angular CRS)",
        ),
        _observe(
            "projected CRS override",
            "xproj",
            lambda: _xproj_override("EPSG:32633", "EPSG:32634"),
            _xarrayrf_projected_override(),
        ),
    )


def main() -> None:
    """Print a compact comparison table."""
    lane = "patched (Index hooks)" if hasattr(xr.Index, "check_override") else "stock/upstream"
    print(f"xarray {xr.__version__}, lane: {lane}")
    print("| case | library + version | observed outcome | xarrayrf outcome |")
    print("|---|---|---|---|")
    for result in observations():
        print(
            f"| {result.case} | {result.library} {result.version} | "
            f"{result.observed} | {result.xarrayrf} |"
        )


if __name__ == "__main__":
    main()
