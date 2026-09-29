"""Projected GeoTIFF metadata and window-reader behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import pytest
import rasterio  # type: ignore[import-untyped]
import xarray as xr
from dask.callbacks import Callback
from numpy.testing import assert_allclose, assert_array_equal
from rasterio.io import MemoryFile  # type: ignore[import-untyped]
from rasterio.transform import Affine  # type: ignore[import-untyped]

from xarrayrf import ArrayCoordinates, geotiff


@pytest.mark.parametrize("area_or_point", ["Area", "Point"])
@pytest.mark.parametrize(
    "affine", [Affine(2, 0, 100, 0, -3, 200), Affine(2, 0.4, 100, 0.2, -3, 200)]
)
def test_pixel_centres_match_rasterio_xy(area_or_point: str, affine: Affine) -> None:
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=3,
            height=2,
            count=1,
            dtype="uint8",
            crs="EPSG:32633",
            transform=affine,
        ) as dataset:
            dataset.write(np.arange(6, dtype=np.uint8).reshape(1, 2, 3))
            dataset.update_tags(AREA_OR_POINT=area_or_point)
        with rasterio.open(memory.name) as dataset:
            geometry = geotiff.from_profile(
                dataset.profile, area_or_point=dataset.tags()["AREA_OR_POINT"]
            )
            array = geotiff.to_dataarray(geometry, dataset.read())
            expected = [
                dataset.xy(row, column, offset="center") for row in range(2) for column in range(3)
            ]
            assert_allclose(
                array.rf.geometry.points().data.reshape(-1, 2), expected, rtol=0, atol=1e-10
            )
            offset = 0.5 if area_or_point == "Area" else None
            assert isinstance(geometry.transform.source, ArrayCoordinates)
            assert geometry.transform.source.sample_offset == (offset, offset)
            assert array.dims == ("band", "row", "column")
            assert_array_equal(array.band, [1])


def test_shared_epsg_identity_and_conflicting_crs_refusal() -> None:
    arrays = []
    for crs in ("EPSG:32633", "EPSG:32633", "EPSG:32634"):
        profile = dict(
            crs=rasterio.crs.CRS.from_string(crs),
            transform=Affine(1, 0, 100, 0, -1, 200),
            width=2,
            height=2,
            count=1,
        )
        geometry = geotiff.from_profile(profile)
        arrays.append(geotiff.to_dataarray(geometry, np.ones((1, 2, 2), dtype=np.float64)))
    assert arrays[0].rf.reference_frame == arrays[1].rf.reference_frame
    assert arrays[0].rf.reference_frame == geotiff.crs_frame(pyproj.CRS.from_epsg(32633))
    assert (arrays[0] + arrays[1]).rf.reference_frame == arrays[0].rf.reference_frame
    with pytest.raises(ValueError, match=r"frame|reference|conflict"):
        _ = arrays[0] + arrays[2]


def test_local_wkt_and_axis_units() -> None:
    crs = pyproj.CRS.from_proj4(
        "+proj=tmerc +lat_0=0 +lon_0=17 +k=0.9996 +x_0=500000 +y_0=0 "
        "+datum=WGS84 +units=us-ft +type=crs"
    )
    assert crs.to_authority(min_confidence=100) is None
    first = geotiff.crs_frame(crs)
    second = geotiff.crs_frame(crs)
    assert first != second
    assert first.definition["crs"] == crs.to_wkt()
    assert first.coordinate_system.units == ("US_survey_foot", "US_survey_foot")
    foot = pyproj.CRS.from_proj4("+proj=utm +zone=33 +datum=WGS84 +units=ft +type=crs")
    assert geotiff.crs_frame(foot).coordinate_system.units == ("ft", "ft")
    metre = geotiff.crs_frame(pyproj.CRS.from_epsg(32633))
    assert metre.coordinate_system.axes == ("Easting", "Northing")
    assert metre.coordinate_system.units == ("m", "m")


def test_crs_refusals() -> None:
    with pytest.raises(TypeError, match="crs"):
        geotiff.crs_frame("EPSG:32633")
    with pytest.raises(ValueError, match="angular coordinates are not representable yet"):
        geotiff.crs_frame(pyproj.CRS.from_epsg(4326))
    with pytest.raises(ValueError, match="compound and vertical"):
        geotiff.crs_frame(pyproj.CRS.from_epsg(7415))
    with pytest.raises(ValueError, match="kilometre"):
        geotiff.crs_frame(
            pyproj.CRS.from_proj4("+proj=utm +zone=33 +datum=WGS84 +units=km +type=crs")
        )


def test_profile_and_data_errors() -> None:
    profile: dict[str, Any] = dict(
        crs=rasterio.crs.CRS.from_epsg(32633),
        transform=Affine.identity(),
        width=2,
        height=2,
        count=1,
    )
    with pytest.raises(TypeError, match="profile"):
        geotiff.from_profile(3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'crs'"):
        geotiff.from_profile({key: value for key, value in profile.items() if key != "crs"})
    with pytest.raises(TypeError, match="transform"):
        geotiff.from_profile(profile | {"transform": 3})
    with pytest.raises(TypeError, match="integers"):
        geotiff.from_profile(profile | {"count": 1.5})
    with pytest.raises(TypeError, match="area_or_point"):
        geotiff.from_profile(profile, area_or_point=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="area_or_point"):
        geotiff.from_profile(profile, area_or_point="Neither")
    with pytest.raises(ValueError, match="positive"):
        geotiff.from_profile(profile | {"width": 0})
    geometry = geotiff.from_profile(profile)
    with pytest.raises(TypeError, match="geometry"):
        geotiff.to_dataarray(3, np.empty((1, 2, 2)))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="duck array"):
        geotiff.to_dataarray(geometry, [[[1]]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        geotiff.to_dataarray(geometry, np.empty((1, 1, 2)))


def test_open_lazy_windows_bands_nodata_and_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    from xarrayrf.geotiff import _reader

    pixels = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    reads: list[tuple[int, ...]] = []
    original = _reader._read_window

    def counted(path: str, bands: tuple[int, ...], window: Any) -> np.ndarray[Any, Any]:
        reads.append(bands)
        return original(path, bands, window)

    monkeypatch.setattr(_reader, "_read_window", counted)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=4,
            height=3,
            count=2,
            dtype="uint16",
            crs="EPSG:32633",
            transform=Affine(2, 0, 100, 0, -2, 200),
            nodata=65535,
        ) as dataset:
            dataset.write(pixels)
        tasks: list[Any] = []
        with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
            array = geotiff.open(memory.name, bands=[2], chunks=(1, 2, 2))
        assert not reads and not tasks
        assert array.attrs["nodata"] == 65535
        assert_array_equal(array.band, [2])
        assert array.rf.reference_frame == geotiff.crs_frame(pyproj.CRS.from_epsg(32633))
        assert_array_equal(array.compute().data, pixels[1:2])
        assert len(reads) == 4
        eager = geotiff.open(memory.name, bands=[2], chunks=None)
        assert isinstance(eager.data, np.ndarray)
        assert_array_equal(eager.data, pixels[1:2])
        full = geotiff.open(memory.name, chunks=None)
        assert_array_equal(full.band, [1, 2])
        assert_array_equal(full.data, pixels)


def test_open_input_errors(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="path"):
        geotiff.open(3)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        geotiff.open(tmp_path / "missing.tif")
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint8",
            crs="EPSG:32633",
            transform=Affine(1, 0, 100, 0, -1, 200),
        ) as dataset:
            dataset.write(np.zeros((1, 2, 2), dtype=np.uint8))
        with pytest.raises(TypeError, match="bands"):
            geotiff.open(memory.name, bands="1")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="bands"):
            geotiff.open(memory.name, bands=[2])


def test_automatic_chunks_follow_internal_tiles(tmp_path: Path) -> None:
    path = tmp_path / "tiled.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=600,
        height=500,
        count=2,
        dtype="uint8",
        crs="EPSG:32633",
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_600_000.0),
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dataset:
        dataset.write(np.zeros((2, 500, 600), dtype=np.uint8))
    array = geotiff.open(path)
    assert array.data.chunks == ((2,), (500,), (600,))  # smaller than one 2048-pixel chunk
    with rasterio.open(path) as dataset:
        assert geotiff._reader._tile_aligned_chunks(dataset, 2) == (2, 500, 600)
    big = type("Dataset", (), {"block_shapes": [(256, 512)], "height": 10980, "width": 10980})()
    assert geotiff._reader._tile_aligned_chunks(big, 3) == (3, 2048, 2048)


def test_northing_first_crs_orders_frame_axes_by_authority() -> None:
    """EPSG:3035 declares northing before easting; rasterio's affine is always x then y."""
    affine = Affine(10.0, 2.0, 3200000.0, 1.5, -10.0, 4330000.0)  # sheared, so rows mix
    profile = {"transform": affine, "width": 4, "height": 3, "count": 1}
    lambert = geotiff.from_profile({"crs": pyproj.CRS.from_epsg(3035), **profile})
    utm = geotiff.from_profile({"crs": pyproj.CRS.from_epsg(32633), **profile})
    assert lambert.frame.coordinate_system.axes == ("Northing", "Easting")
    assert utm.frame.coordinate_system.axes == ("Easting", "Northing")
    pixels = np.array([[2.0, 1.0], [0.0, 3.0]])  # (row, column)
    expected = np.array(
        [rasterio.transform.xy(affine, row, col, offset="center") for row, col in pixels]
    )
    assert_allclose(utm.transform.transform_point(pixels), expected, rtol=0, atol=1e-6)
    assert_allclose(lambert.transform.transform_point(pixels), expected[:, ::-1], rtol=0, atol=1e-6)


def test_crs_frame_accepts_westing_southing_but_import_refuses_it() -> None:
    crs = pyproj.CRS.from_epsg(2046)  # Hartebeesthoek94 / Lo15: westing, southing
    assert geotiff.crs_frame(crs).coordinate_system.axes == ("Westing", "Southing")
    with pytest.raises(ValueError, match="axis direction"):
        geotiff.from_profile(
            {"crs": crs, "transform": Affine.identity(), "width": 1, "height": 1, "count": 1}
        )


def test_source_axes_are_typed_as_space() -> None:
    profile = {
        "crs": pyproj.CRS.from_epsg(32633),
        "transform": rasterio.transform.from_origin(0.0, 0.0, 1.0, 1.0),
        "width": 2,
        "height": 2,
        "count": 1,
    }
    source = geotiff.from_profile(profile).transform.source
    assert isinstance(source, ArrayCoordinates)
    assert source.axis_types == ("space", "space")
    assert source == ArrayCoordinates(
        ("row", "column"), ("1", "1"), axis_types=("space", "space"), sample_offset=(0.5, 0.5)
    )


_MEMORY_AFFINE = Affine(2, 0, 100, 0, -3, 200)


def _open_in_memory(crs: str) -> xr.DataArray:
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=4,
            height=3,
            count=1,
            dtype="float64",
            crs=crs,
            transform=_MEMORY_AFFINE,
        ) as dataset:
            dataset.write(np.arange(12, dtype=np.float64).reshape(1, 3, 4))
        return geotiff.open(memory.name, chunks=None)


def test_strided_selection_keeps_pixel_locations() -> None:
    stride = _open_in_memory("EPSG:32633").isel(column=slice(None, None, 2))
    expected = [rasterio.transform.xy(_MEMORY_AFFINE, 0, column) for column in (0, 2)]
    assert_allclose(stride.rf.geometry.points().data[0], expected, rtol=0, atol=1e-10)


def test_join_keeps_the_frame_and_concatenation_is_refused() -> None:
    array = _open_in_memory("EPSG:32633")
    joined = xr.align(array, array.isel(column=slice(1, None)), join="inner")[0]
    assert joined.rf.reference_frame == array.rf.reference_frame
    with pytest.raises(ValueError, match="concatenation of framed arrays"):
        xr.concat([array, array], dim="column")


@pytest.mark.xfail(
    not hasattr(xr.Index, "check_override"),
    strict=True,
    raises=pytest.fail.Exception,
    reason="this xarray lane lacks the check_override Index hook (known align_override hole)",
)
def test_override_alignment_across_crs_is_refused() -> None:
    with pytest.raises(ValueError, match=r"frame|reference|conflict"):
        xr.align(_open_in_memory("EPSG:32633"), _open_in_memory("EPSG:32634"), join="override")
