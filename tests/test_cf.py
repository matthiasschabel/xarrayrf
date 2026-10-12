"""Public behaviour of the CF longitude/latitude reader, ``xarrayrf.cf``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import xarrayrf as xrf
import xarrayrf.cf as rfcf

LON = np.arange(0.0, 360.0, 2.5)
LAT = np.arange(-88.75, 90.0, 2.5)  # cell-centred: the poles lie half a cell beyond the rows
SPHERE = {"grid_mapping_name": "latitude_longitude", "earth_radius": 6371229.0}
WGS84 = {
    "grid_mapping_name": "latitude_longitude",
    "semi_major_axis": 6378137.0,
    "inverse_flattening": 298.257223563,
}


def sphere_field(lon: npt.ArrayLike, lat: npt.ArrayLike) -> npt.NDArray[np.float64]:
    lam, phi = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    phi, lam = np.meshgrid(phi, lam, indexing="ij")
    field: npt.NDArray[np.float64] = (
        np.cos(phi) * np.cos(lam) + 2 * np.cos(phi) * np.sin(lam) + 3 * np.sin(phi)
    )
    return field


def dataset(
    mapping: dict[str, Any] | None = SPHERE,
    *,
    lon: npt.NDArray[np.float64] = LON,
    lat: npt.NDArray[np.float64] = LAT,
    lon_attrs: dict[str, Any] | None = None,
    lat_attrs: dict[str, Any] | None = None,
    bounds: bool = True,
    lazy: bool = False,
) -> xr.Dataset:
    """A CMIP-style monthly field: time by lat by lon, with bounds and a grid mapping."""
    values = np.stack([sphere_field(lon, lat)] * 2)
    data: Any = da.from_array(values, chunks=(1, -1, -1)) if lazy else values  # type: ignore[no-untyped-call]
    variable_attrs = {"units": "K"} if mapping is None else {"units": "K", "grid_mapping": "crs"}
    variables: dict[str, Any] = {"tas": (("time", "lat", "lon"), data, variable_attrs)}
    lon_meta = {"units": "degrees_east", "standard_name": "longitude", **(lon_attrs or {})}
    lat_meta = {"units": "degrees_north", "standard_name": "latitude", **(lat_attrs or {})}
    if bounds:
        variables["lon_bnds"] = (("lon", "bnds"), cell_bounds(lon))
        variables["lat_bnds"] = (("lat", "bnds"), cell_bounds(lat))
        lon_meta.setdefault("bounds", "lon_bnds")
        lat_meta.setdefault("bounds", "lat_bnds")
    if mapping is not None:
        variables["crs"] = ((), np.int32(0), mapping)
    return xr.Dataset(
        variables,
        coords={"time": [0, 1], "lon": ("lon", lon, lon_meta), "lat": ("lat", lat, lat_meta)},
    )


def cell_bounds(labels: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Bounds halfway to each neighbour, in the coordinate's own order, as CF requires."""
    step = labels[1] - labels[0]
    return np.stack([labels - step / 2, labels + step / 2], axis=-1)


def codes(geometry: rfcf.CfGeometry) -> list[str]:
    return [code for code, _ in geometry.report]


def test_a_cmip_style_field_is_framed_and_resamples_across_the_seam_and_poles() -> None:
    tas = rfcf.frame_variable(dataset(lazy=True), "tas")
    frame = tas.rf.reference_frame
    assert frame.coordinate_system.representation == "geodetic"
    assert frame.coordinate_system.axis_roles == ("longitude", "latitude")
    assert dict(frame.definition) == {
        "figure": "sphere",
        "semi_major_axis": 6371229.0,
        "length_unit": "m",
    }
    assert isinstance(tas.data, da.Array)
    assert list(tas.time.values) == [0, 1]
    assert tas.attrs["units"] == "K"
    target_lon, target_lat = np.arange(-180.0, 180.0, 2.5) + 0.3, np.array([0.0, 89.5, -89.9])
    target = xrf.Grid(
        tas.rf.grid.transform, {"lon": ("lon", target_lon), "lat": ("lat", target_lat)}
    )
    result = xrf.resample(tas.rf.geometry, target)
    assert isinstance(result.data, da.Array)
    values = result.transpose("time", "lat", "lon").values
    # Linear interpolation over 2.5° cells: (2.5°)² / 8 · max |f''| ≈ 1e-3.
    assert_allclose(values[0], sphere_field(target_lon, target_lat), atol=2e-3, rtol=0)


@pytest.mark.parametrize(
    "mapping, definition",
    [
        (SPHERE, {"figure": "sphere", "semi_major_axis": 6371229.0}),
        (
            WGS84,
            {
                "figure": "ellipsoid",
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257223563,
            },
        ),
        (
            {
                "grid_mapping_name": "latitude_longitude",
                "semi_major_axis": 6371000.0,
                "inverse_flattening": 0.0,
            },
            {"figure": "sphere", "semi_major_axis": 6371000.0},
        ),
        (
            {
                "grid_mapping_name": "latitude_longitude",
                "semi_major_axis": 2.0,
                "semi_minor_axis": 1.5,
            },
            {"figure": "ellipsoid", "semi_major_axis": 2.0, "inverse_flattening": 4.0},
        ),
        (
            {
                "grid_mapping_name": "latitude_longitude",
                "semi_major_axis": 6371000.0,
                "semi_minor_axis": 6371000.0,
            },
            {"figure": "sphere", "semi_major_axis": 6371000.0},
        ),
        (
            {
                "grid_mapping_name": "latitude_longitude",
                "semi_major_axis": 6371000.0,
                "semi_minor_axis": 6371000.0,
                "inverse_flattening": 0.0,
            },
            {"figure": "sphere", "semi_major_axis": 6371000.0},
        ),
        (
            {**WGS84, "semi_minor_axis": 6356752.314245179},
            {
                "figure": "ellipsoid",
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257223563,
            },
        ),
        (
            {"grid_mapping_name": "latitude_longitude", "semi_major_axis": 6371000.0},
            {"figure": "sphere", "semi_major_axis": 6371000.0},
        ),
        (
            {**WGS84, "horizontal_datum_name": "WGS_1984", "reference_ellipsoid_name": "WGS 84"},
            {
                "figure": "ellipsoid",
                "semi_major_axis": 6378137.0,
                "inverse_flattening": 298.257223563,
                "datum_name": "WGS_1984",
                "ellipsoid_name": "WGS 84",
            },
        ),
    ],
    ids=[
        "earth-radius",
        "wgs84",
        "zero-flattening",
        "semi-axes",
        "equal-semi-axes",
        "three-parameter-sphere",
        "three-parameter-ellipsoid",
        "semi-major-only",
        "datum-and-ellipsoid-names",
    ],
)
def test_figures(mapping: dict[str, Any], definition: dict[str, Any]) -> None:
    geometry = rfcf.from_dataset(dataset(mapping), "tas")
    assert dict(geometry.frame.definition) == {**definition, "length_unit": "m"}


@pytest.mark.parametrize(
    "mapping, message",
    [
        ({**WGS84, "semi_minor_axis": 6356000.0}, "semi_minor_axis implies inverse_flattening"),
        ({**SPHERE, "semi_major_axis": 6371229.0}, "earth_radius together with ellipsoid"),
        (
            {"grid_mapping_name": "latitude_longitude", "inverse_flattening": 298.0},
            "lacks semi_major_axis",
        ),
        ({**SPHERE, "longitude_of_prime_meridian": 2.337}, "longitude_of_prime_meridian=2.337"),
        ({"grid_mapping_name": "rotated_latitude_longitude"}, "rotated pole"),
        ({"grid_mapping_name": "lambert_conformal_conic"}, "'lambert_conformal_conic'"),
        ({**SPHERE, "earth_radius": "big"}, "earth_radius='big' is not a number"),
    ],
)
def test_figure_refusals(mapping: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rfcf.from_dataset(dataset(mapping), "tas")


def test_grid_mapping_references_are_checked() -> None:
    extended = dataset()
    extended["tas"].attrs["grid_mapping"] = "crs: lat lon"
    with pytest.raises(ValueError, match="extended grid_mapping syntax"):
        rfcf.from_dataset(extended, "tas")
    missing = dataset()
    missing["tas"].attrs["grid_mapping"] = "nowhere"
    with pytest.raises(ValueError, match="grid_mapping 'nowhere' of 'tas' is not in the dataset"):
        rfcf.from_dataset(missing, "tas")


@pytest.mark.parametrize(
    "mapping", [None, {"grid_mapping_name": "latitude_longitude"}], ids=["none", "no-figure"]
)
def test_an_unspecified_figure_needs_a_frame(mapping: dict[str, Any] | None) -> None:
    with pytest.raises(ValueError, match=r"declares no earth figure.*pass frame="):
        rfcf.from_dataset(dataset(mapping), "tas")
    model = rfcf.frame_variable(dataset(), "tas")
    adopted = rfcf.from_dataset(dataset(mapping), "tas", frame=model)
    assert adopted.frame == model.rf.reference_frame
    assert "figure-ignored" not in codes(adopted)


def test_a_supplied_frame_is_shared_and_replaces_a_declared_figure() -> None:
    first = rfcf.frame_variable(dataset(), "tas")
    second = rfcf.frame_variable(dataset(), "tas")
    # The same radius does not make the same frame.
    assert first.rf.reference_frame != second.rf.reference_frame
    shared = rfcf.from_dataset(dataset(WGS84), "tas", frame=first)
    assert shared.frame == first.rf.reference_frame
    assert "figure-ignored" in codes(shared)
    joined = rfcf.frame_variable(dataset(), "tas", frame=first)
    result = xrf.resample(joined.rf.geometry, first.rf.grid)  # no transform needed
    assert_allclose(result.transpose(*first.dims).values, first.values, atol=1e-12, rtol=0)


def test_a_frame_with_other_unit_spellings_is_not_adopted() -> None:
    frame = xrf.ReferenceFrame.local(
        xrf.CoordinateSystem(
            ("lon", "lat"),
            ("deg", "deg"),
            representation="geodetic",
            axis_roles=("longitude", "latitude"),
        )
    )
    with pytest.raises(ValueError, match=r"cannot adopt frame.*never rescaled"):
        rfcf.from_dataset(dataset(), "tas", frame=frame)


def test_west_and_south_units_keep_their_labels() -> None:
    west = rfcf.frame_variable(
        dataset(
            lon=-LON,
            lat=-LAT,
            lon_attrs={"units": "degrees_west"},
            lat_attrs={"units": "degrees_south"},
            bounds=False,
        ),
        "tas",
    )
    geometry = rfcf.from_dataset(
        dataset(
            lon=-LON,
            lat=-LAT,
            lon_attrs={"units": "degrees_west"},
            lat_attrs={"units": "degrees_south"},
            bounds=False,
        ),
        "tas",
    )
    assert {"west-longitude", "south-latitude"} <= set(codes(geometry))
    assert_allclose(west.lon.values, -LON)
    assert west.rf.reference_frame.units == ("degrees_east", "degrees_north")
    points = xrf.transform_named(west.rf.grid.transform, {"lon": -10.0, "lat": -20.0})
    assert_allclose([points["lon"], points["lat"]], [10.0, 20.0])


def test_descending_latitude_with_descending_bounds() -> None:
    geometry = rfcf.from_dataset(dataset(lat=LAT[::-1]), "tas")
    source = geometry.grid.transform.source
    assert isinstance(source, xrf.ArrayCoordinates)
    assert source.sample_offset == (0.5, 0.5)
    rows = geometry.grid.intervals["lat"]
    assert np.all(rows[:, 0] < rows[:, 1])
    assert "no-bounds" not in codes(geometry)


def test_off_centre_bounds_are_dropped_and_reported() -> None:
    """Gaussian-grid latitudes sit at varying positions within their cells."""
    ds = dataset()
    shifted = cell_bounds(LAT)
    shifted[::2] += [0.3, 0.3]
    ds["lat_bnds"] = (("lat", "bnds"), shifted)
    geometry = rfcf.from_dataset(ds, "tas")
    assert "bounds-off-centre" in codes(geometry)
    assert "lat" not in geometry.grid.intervals
    assert "lon" in geometry.grid.intervals


def test_missing_bounds_are_reported_not_inferred() -> None:
    geometry = rfcf.from_dataset(dataset(bounds=False), "tas")
    assert codes(geometry).count("no-bounds") == 2
    assert geometry.grid.intervals == {}


def test_coordinates_are_found_by_standard_name_or_units_under_any_name() -> None:
    ds = dataset(lon_attrs={"units": "degrees"}, lat_attrs={"standard_name": "latitude"})
    assert rfcf.from_dataset(ds, "tas").longitude == "lon"
    # Auxiliary coordinates named differently from their dimensions.
    renamed = dataset(bounds=False).rename({"lon": "x", "lat": "y"})
    renamed = renamed.assign_coords(
        longitude=("x", renamed.x.values, dict(renamed.x.attrs)),
        latitude=("y", renamed.y.values, dict(renamed.y.attrs)),
    )
    for name in ("x", "y"):
        renamed[name].attrs = {}
    geometry = rfcf.from_dataset(renamed, "tas")
    assert (geometry.longitude, geometry.latitude) == ("longitude", "latitude")
    assert geometry.grid.dims == ("x", "y")
    framed = rfcf.frame_variable(renamed, "tas")
    assert framed.rf.reference_frame.axes == ("longitude", "latitude")


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda ds: ds.assign_coords(lon2=("lon", ds.lon.values, {"units": "degrees_east"})),
            r"exactly one one-dimensional longitude coordinate, found \['lon', 'lon2'\]",
        ),
        (
            lambda ds: ds.assign(
                tas=ds.tas.assign_coords(lon=ds.lon.assign_attrs(units="m", standard_name="x"))
            ),
            "exactly one one-dimensional longitude coordinate, found \\[\\]",
        ),
        (lambda ds: ds.assign_coords(lon=ds.lon.assign_attrs(units="km")), "has units 'km'"),
        (
            lambda ds: ds.assign_coords(lon=ds.lon.assign_attrs(standard_name="grid_longitude")),
            "rotated-pole grid_longitude",
        ),
        (
            lambda ds: ds.assign_coords(
                lat2d=(
                    ("lat", "lon"),
                    np.zeros((LAT.size, LON.size)),
                    {"standard_name": "latitude", "units": "degrees_north"},
                )
            ),
            "curvilinear",
        ),
    ],
    ids=["two-longitudes", "no-longitude", "bad-units", "rotated", "two-dimensional"],
)
def test_coordinate_refusals(change: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rfcf.from_dataset(change(dataset()), "tas")


def test_argument_refusals() -> None:
    with pytest.raises(TypeError, match="dataset must be an xarray Dataset"):
        rfcf.from_dataset(dataset()["tas"], "tas")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="variable must be a string"):
        rfcf.from_dataset(dataset(), 3)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="'pr' is not a data variable"):
        rfcf.from_dataset(dataset(), "pr")


def test_a_netcdf_round_trip_reads_back_identically(tmp_path: Path) -> None:
    path = tmp_path / "tas.nc"
    dataset().to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as reopened:
        framed = rfcf.frame_variable(reopened, "tas")
        direct = rfcf.frame_variable(dataset(), "tas")
        assert dict(framed.rf.reference_frame.definition) == dict(
            direct.rf.reference_frame.definition
        )
        assert framed.rf.grid.intervals.keys() == direct.rf.grid.intervals.keys()
        assert_allclose(framed.values, direct.values)


@pytest.mark.parametrize(
    "mapping, message",
    [
        ({"grid_mapping_name": "latitude_longitude", "earth_radius": 0.0}, "earth_radius=0"),
        ({**WGS84, "semi_major_axis": 0.0}, "semi_major_axis=0"),
        ({**WGS84, "inverse_flattening": 0.5}, "inverse_flattening=0.5; it must be 0"),
        (
            {
                "grid_mapping_name": "latitude_longitude",
                "semi_major_axis": 1.0,
                "semi_minor_axis": 2.0,
            },
            "prolate",
        ),
        ({**SPHERE, "earth_radius": float("inf")}, "must be finite and non-negative"),
        ({**SPHERE, "earth_radius": -1.0}, "must be finite and non-negative"),
    ],
    ids=["zero-radius", "zero-major", "flattening-below-one", "prolate", "infinite", "negative"],
)
def test_figures_outside_the_transitions_range_are_refused(
    mapping: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        rfcf.from_dataset(dataset(mapping), "tas")


def test_bounds_must_exist_and_fit() -> None:
    broken = dataset()
    broken["lon"].attrs["bounds"] = "missing"
    with pytest.raises(
        ValueError, match="'lon' declares bounds 'missing', which is not a variable"
    ):
        rfcf.from_dataset(broken, "tas")
    short = dataset()
    short["lon_bnds"] = (("lon", "three"), np.zeros((LON.size, 3)))
    with pytest.raises(ValueError, match=r"have shape \(144, 3\); expected \(144, 2\)"):
        rfcf.from_dataset(short, "tas")
    empty = dataset()
    empty["lat_bnds"] = (("lat", "bnds"), np.stack([LAT, LAT], axis=-1))
    with pytest.raises(ValueError, match="need finite, positive widths"):
        rfcf.from_dataset(empty, "tas")


def test_longitude_and_latitude_need_separate_dimensions() -> None:
    ds = dataset(bounds=False)
    ds["lat"].attrs = {}  # the dimension coordinate is no longer latitude…
    along_lon = np.linspace(-80.0, 80.0, LON.size)
    ds = ds.assign_coords(lat2=("lon", along_lon, {"units": "degrees_north"}))  # …but this is
    with pytest.raises(ValueError, match="share dimension"):
        rfcf.from_dataset(ds, "tas")
