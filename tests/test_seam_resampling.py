"""Resampling across the longitude seam and the poles of a longitude/latitude source."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import xarrayrf as xrf
from xarrayrf._sampling import POSITION_SLACK
from xarrayrf.native import frame_array

FRAME = xrf.ReferenceFrame.local(
    xrf.CoordinateSystem(
        ("lon", "lat"),
        ("degrees_east", "degrees_north"),
        representation="spherical",
        axis_roles=("longitude", "latitude"),
    )
)
ROUNDOFF = 1e-10
"""Seam handling against an equivalent interior evaluation: only float64 roundoff differs."""


def transform(
    frame: xrf.ReferenceFrame = FRAME,
    matrix: npt.ArrayLike = ((1.0, 0.0), (0.0, 1.0)),
    translation: npt.ArrayLike = (0.0, 0.0),
    sample_offset: tuple[float | None, float | None] = (None, None),
) -> xrf.AffineTransform:
    units = frame.units
    return xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("lon", "lat"), units, sample_offset=sample_offset),
        target=frame,
        matrix=matrix,
        translation=translation,
    )


def grid(
    lon: npt.ArrayLike,
    lat: npt.ArrayLike,
    *,
    mapping: xrf.AffineTransform | None = None,
    intervals: dict[str, npt.ArrayLike] | None = None,
) -> xrf.Grid:
    return xrf.Grid(
        transform() if mapping is None else mapping,
        {"lon": ("lon", np.asarray(lon, float)), "lat": ("lat", np.asarray(lat, float))},
        intervals=intervals,
    )


def sphere_field(lon: npt.ArrayLike, lat: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """x + 2y + 3z of the unit vector: smooth on the sphere, single-valued at the poles."""
    lam, phi = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    lam, phi = np.meshgrid(lam, phi, indexing="ij")
    field: npt.NDArray[np.float64] = (
        np.cos(phi) * np.cos(lam) + 2 * np.cos(phi) * np.sin(lam) + 3 * np.sin(phi)
    )
    return field


def source(
    lon: npt.ArrayLike, lat: npt.ArrayLike, values: npt.ArrayLike | None = None, **kwargs: object
) -> xr.DataArray:
    data = sphere_field(lon, lat) if values is None else np.asarray(values, float)
    return frame_array(data, grid(lon, lat, **kwargs))  # type: ignore[arg-type]


def resample(src: xr.DataArray, target: xrf.Grid, **kwargs: object) -> npt.NDArray[np.float64]:
    result = xrf.resample(src.rf.geometry, target, **kwargs)  # type: ignore[arg-type]
    return np.asarray(result.transpose("lon", "lat").values)


GLOBAL_LON = np.arange(0.0, 360.0, 2.5)
LAT = np.array([-30.0, -10.0, 15.0, 40.0])
UNIFORM_LAT = np.array([-30.0, -10.0, 10.0, 30.0])
SEAM_TARGETS = np.array([356.0, 358.75, 359.9, 0.0, 0.5, 1.25, -1.25, 718.75])


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_a_global_grid_interpolates_across_the_seam(method: xrf.Method) -> None:
    """Rotated by half a turn, the same samples put the seam far from the targets."""
    pytest.importorskip("scipy", minversion="1.18")
    wrapped = resample(source(GLOBAL_LON, LAT), grid(SEAM_TARGETS, [0.0, 20.0]), method=method)
    rotated_lon = GLOBAL_LON - 180.0
    interior = resample(
        source(rotated_lon, LAT),
        grid(np.remainder(SEAM_TARGETS + 180.0, 360.0) - 180.0, [0.0, 20.0]),
        method=method,
    )
    assert not np.isnan(wrapped).any()
    assert_allclose(wrapped, interior, atol=ROUNDOFF, rtol=0)


def test_descending_and_closing_duplicate_grids_match_the_ascending_one() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    target = grid(SEAM_TARGETS, [0.0, 20.0])
    expected = resample(source(GLOBAL_LON, LAT), target)
    descending = resample(source(GLOBAL_LON[::-1], LAT), target)
    closed_lon = np.append(GLOBAL_LON, 360.0)
    values = sphere_field(closed_lon, LAT)
    values[-1] = 1e6  # never read: the closing label repeats the first one turn later
    closed = resample(source(closed_lon, LAT, values), target)
    closed_descending = resample(source(closed_lon[::-1], LAT, values[::-1]), target)
    for result in (descending, closed, closed_descending):
        assert_allclose(result, expected, atol=ROUNDOFF, rtol=0)


def test_nonuniform_global_longitude_locates_exactly() -> None:
    # 0, 2, 5, 7, …, 355, 357: steps alternate 2° and 3°, and the seam step 357 → 360 is 3°.
    pytest.importorskip("scipy", minversion="1.18")
    lon = np.concatenate(([0.0], np.cumsum(np.tile([2.0, 3.0], 72))[:-1]))
    assert lon[-1] == 357.0
    values = np.repeat(lon[:, None], LAT.size, axis=1)
    result = resample(source(lon, LAT, values), grid([6.0, 356.5], LAT))
    assert_allclose(result[:, 0], [6.0, 356.5], atol=ROUNDOFF, rtol=0)
    cosine = np.repeat(np.cos(np.radians(lon))[:, None], LAT.size, axis=1)
    seam = resample(source(lon, LAT, cosine), grid([359.0], LAT))
    # Two thirds of the way from 357° to 360° (the first sample, one turn on).
    expected = np.cos(np.radians(357.0)) / 3 + 2 / 3
    assert_allclose(seam[0, 0], expected, atol=ROUNDOFF, rtol=0)


@pytest.mark.parametrize(
    "lon, is_global",
    [
        pytest.param((np.arange(3600) / 10).astype(np.float32), True, id="float32-0.1"),
        pytest.param((np.arange(36000) / 100).astype(np.float32), True, id="float32-0.01"),
        pytest.param(np.arange(0.0, 355.1, 2.5), False, id="one-cell-short"),
        pytest.param(np.cumsum(np.tile([2.0, 3.0], 72))[:-1], False, id="missing-2-degree-sample"),
    ],
)
def test_coverage_from_labels(lon: npt.NDArray[np.floating[Any]], is_global: bool) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    lon64 = lon.astype(np.float64)
    result = resample(
        source(lon64, LAT), grid([float(lon64[-1]) + 0.5 * float(lon64[-1] - lon64[-2])], LAT)
    )
    assert np.isfinite(result).all() == is_global


def test_a_fine_float64_global_grid_keeps_its_last_sample() -> None:
    """A float32 rounding allowance would call this 0.0001° seam a duplicate endpoint."""
    pytest.importorskip("scipy", minversion="1.18")
    lon = np.arange(3_600_000) / 10_000
    values = np.repeat(np.arange(lon.size, dtype=float)[:, None], 2, axis=1)
    result = resample(source(lon, [0.0, 1.0], values), grid([359.99995], [0.0]))
    # Halfway between the last sample and the first: (3_599_999 + 0) / 2.
    assert_allclose(result[0, 0], 1_799_999.5, atol=1e-6, rtol=0)


def test_declared_intervals_decide_coverage() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    target = grid([358.75], LAT)
    offset = transform(sample_offset=(0.5, 0.5))
    lat_bounds = LAT[:, None] + [-1.0, 1.0]
    tiling = {"lon": GLOBAL_LON[:, None] + [-1.25, 1.25], "lat": lat_bounds}
    gapped = {"lon": GLOBAL_LON[:, None] + [-1.0, 1.0], "lat": lat_bounds}
    tiled = resample(source(GLOBAL_LON, LAT, mapping=offset, intervals=tiling), target)
    assert np.isfinite(tiled).all()
    regional = resample(source(GLOBAL_LON, LAT, mapping=offset, intervals=gapped), target)
    assert np.isnan(regional).all()


@pytest.mark.parametrize(
    "labels, targets, equivalent",
    [
        (
            np.arange(170.0, 190.1, 1.0),
            [-185.0, -175.5, 175.0, 185.25],
            [175.0, 184.5, 175.0, 185.25],
        ),
        (np.arange(-10.0, 10.1, 1.0), [355.0, 365.5, -5.0], [-5.0, 5.5, -5.0]),
    ],
)
def test_a_regional_grid_is_found_from_another_branch(
    labels: npt.NDArray[np.float64], targets: list[float], equivalent: list[float]
) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    values = np.repeat(labels[:, None], LAT.size, axis=1)
    result = resample(source(labels, LAT, values), grid(targets, LAT))
    assert_allclose(result[:, 0], equivalent, atol=ROUNDOFF, rtol=0)
    beyond = resample(source(labels, LAT, values), grid([labels[-1] + 10.0], LAT))
    assert np.isnan(beyond).all()


def test_cells_reach_into_the_shifted_branch() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    labels = np.arange(170.0, 190.1, 1.0)
    values = np.repeat(labels[:, None], LAT.size, axis=1)
    src = source(labels, LAT, values, mapping=transform(sample_offset=(0.5, 0.5)))
    # 169.6 lies in the first sample's cell; -190.4 is the same longitude.
    result = resample(src, grid([169.6, -190.4], LAT), domain="cells")
    assert_allclose(result[:, 0], [170.0, 170.0], atol=ROUNDOFF, rtol=0)


def test_a_coupled_source_mapping_does_not_wrap() -> None:
    """lon = u + 0.1 v: shifting u by a period would also move the sample along v."""
    pytest.importorskip("scipy", minversion="1.18")
    coupled = transform(matrix=((1.0, 0.1), (0.0, 1.0)))
    src = source(GLOBAL_LON, LAT, mapping=coupled)
    result = xrf.resample(src.rf.geometry, grid([359.0 + 0.1 * 15.0], [15.0]))
    assert np.isnan(result.values).all()


def test_transposed_storage_wraps_the_longitude_axis() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    target = grid(SEAM_TARGETS, [0.0, 20.0])
    expected = resample(source(GLOBAL_LON, LAT), target, method="cubic")
    transposed = source(GLOBAL_LON, LAT).transpose("lat", "lon")
    assert_allclose(resample(transposed, target, method="cubic"), expected, atol=ROUNDOFF, rtol=0)


def test_cubic_wraps_a_grid_smaller_than_its_halo() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    lon = np.arange(0.0, 360.0, 45.0)
    targets = np.array([340.0, 10.0, 357.0])
    wrapped = resample(source(lon, LAT), grid(targets, [15.0]), method="cubic")
    rotated = resample(
        source(lon - 180.0, LAT),
        grid(np.remainder(targets + 180.0, 360.0) - 180.0, [15.0]),
        method="cubic",
    )
    assert_allclose(wrapped, rotated, atol=1e-9, rtol=0)


def test_missing_values_near_the_seam_spread_only_to_their_neighbours() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    values = sphere_field(GLOBAL_LON, LAT)
    values[0, :] = np.nan
    result = resample(source(GLOBAL_LON, LAT, values), grid([358.75, 1.25, 3.75], LAT))
    assert np.isnan(result[:2]).all()
    assert np.isfinite(result[2]).all()


# Poles

CENTRED_LAT = np.arange(-89.0, 89.1, 2.0)
POLE_LAT = np.arange(-90.0, 90.1, 2.0)
POLAR_LON = np.arange(0.0, 360.0, 2.0)
CAP_TARGETS = (np.array([0.0, 37.0, 181.0, 300.0]), np.array([-90.0, -89.5, 89.2, 89.9, 90.0]))
# Linear interpolation of this field over 2° cells is accurate to (2°)² / 8 · max |f''| ≈ 6e-4.
LINEAR_ERROR = 1e-3


@pytest.mark.parametrize("lat", [CENTRED_LAT, POLE_LAT], ids=["cell-centred", "pole-sample"])
def test_the_polar_cap_interpolates_across_the_pole(lat: npt.NDArray[np.float64]) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    lon_t, lat_t = CAP_TARGETS
    result = resample(source(POLAR_LON, lat), grid(lon_t, lat_t))
    assert_allclose(result, sphere_field(lon_t, lat_t), atol=LINEAR_ERROR, rtol=0)
    # The pole is one point: every longitude label there gets one value.
    for column in (0, -1):
        assert np.ptp(result[:, column]) < 1e-12


def test_cubic_across_the_pole_is_accurate() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    lon_t, lat_t = CAP_TARGETS
    result = resample(source(POLAR_LON, CENTRED_LAT), grid(lon_t, lat_t), method="cubic")
    # Cubic splines over 2° cells: error of order (2°)⁴, far below the linear bound.
    assert_allclose(result, sphere_field(lon_t, lat_t), atol=1e-6, rtol=0)


def test_latitude_beyond_a_pole_is_outside() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    result = resample(source(POLAR_LON, CENTRED_LAT), grid([10.0], [90.5, -90.5, 89.5]))
    assert np.isnan(result[0, :2]).all()
    assert np.isfinite(result[0, 2])


@pytest.mark.parametrize("lat", [CENTRED_LAT, POLE_LAT], ids=["cell-centred", "pole-sample"])
def test_roundoff_past_a_pole_lands_on_it(lat: npt.NDArray[np.float64]) -> None:
    # The band is POSITION_SLACK (1e-9) of the 2° step: 2e-9°. 1e-6° is past it.
    pytest.importorskip("scipy", minversion="1.18")
    past = [90.0 + 1e-12, np.nextafter(90.0, 100.0), -90.0 - 1e-12, 90.0 + 1e-6]
    result = resample(source(POLAR_LON, lat), grid([10.0], [90.0, -90.0, *past]))
    assert_allclose(result[0, 2:5], result[0, [0, 0, 1]], atol=1e-12, rtol=0)
    assert np.isfinite(result[0, :2]).all()
    assert np.isnan(result[0, 5])


@pytest.mark.parametrize("step", [2.0, 0.5])
@pytest.mark.parametrize("descending", [False, True], ids=["ascending", "descending"])
def test_the_pole_band_scales_with_the_latitude_step(step: float, descending: bool) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    lat = np.arange(-90.0 + step / 2, 90.0, step)
    inside, outside = 0.5 * POSITION_SLACK * step, 2.0 * POSITION_SLACK * step
    targets = [90.0, 90.0 + inside, 90.0 + outside, -90.0, -90.0 - inside, -90.0 - outside]
    result = resample(source(POLAR_LON, lat[::-1] if descending else lat), grid([10.0], targets))
    assert np.isfinite(result[0, [0, 1, 3, 4]]).all()
    assert_allclose(result[0, [1, 4]], result[0, [0, 3]], atol=1e-12, rtol=0)
    assert np.isnan(result[0, [2, 5]]).all()


@pytest.mark.parametrize(
    "lon, lat",
    [
        pytest.param(np.arange(0.0, 360.0, 360.0 / 179), CENTRED_LAT, id="odd-count"),
        pytest.param(
            np.concatenate(([0.0, 2.5], np.arange(4.0, 360.0, 2.0))),  # 2.5 has no antipode
            CENTRED_LAT,
            id="not-antipodal",
        ),
        pytest.param(POLAR_LON, np.arange(-59.0, 59.1, 2.0), id="latitude-stops-at-60"),
        pytest.param(np.arange(0.0, 340.0, 2.0), CENTRED_LAT, id="regional-longitude"),
    ],
)
def test_no_pole_crossing_without_antipodal_global_coverage(
    lon: npt.NDArray[np.float64], lat: npt.NDArray[np.float64]
) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    beyond = float(lat[-1]) + 0.5
    result = resample(source(lon, lat), grid([10.0], [beyond]))
    assert np.isnan(result).all()


def test_poles_follow_the_latitude_unit_and_mapping() -> None:
    """Degrees of longitude with radians of latitude, and latitude = 2c + 10 (in radians)."""
    pytest.importorskip("scipy", minversion="1.18")
    frame = xrf.ReferenceFrame.local(
        xrf.CoordinateSystem(
            ("lon", "lat"),
            ("degrees_east", "rad"),
            representation="spherical",
            axis_roles=("longitude", "latitude"),
        )
    )
    mapping = transform(frame, matrix=((1.0, 0.0), (0.0, 2.0)), translation=(0.0, 10.0))
    lat_rad = np.radians(CENTRED_LAT)
    coordinates = (lat_rad - 10.0) / 2.0
    src = frame_array(
        sphere_field(POLAR_LON, CENTRED_LAT),
        xrf.Grid(mapping, {"lon": ("lon", POLAR_LON), "lat": ("lat", coordinates)}),
    )
    targets = np.radians([89.5, 90.0])
    target = xrf.Grid(mapping, {"lon": ("lon", [37.0]), "lat": ("lat", (targets - 10.0) / 2.0)})
    result = xrf.resample(src.rf.geometry, target).transpose("lon", "lat").values
    assert_allclose(result, sphere_field([37.0], [89.5, 90.0]), atol=LINEAR_ERROR, rtol=0)


def test_declared_components_wrap_at_the_seam_but_not_across_the_pole() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    values = np.stack([sphere_field(POLAR_LON, CENTRED_LAT)] * 2, axis=-1)
    plain = xr.DataArray(
        values, dims=("lon", "lat", "component"), coords={"component": list(FRAME.axes)}
    )
    declared = cast(xr.DataArray, plain.rf.frame(grid(POLAR_LON, CENTRED_LAT)))
    declared = declared.rf.declare_tensor_components(
        xrf.VECTOR, component_dims=("component",), bases=FRAME
    )
    result = xrf.resample(
        declared.rf.geometry,
        grid([359.0, 10.0], [0.0, 89.5]),
        output_component_basis=FRAME,
        component_reorientation="jacobian",
    ).transpose("lon", "lat", "component")
    assert np.isfinite(result.values[:, 0]).all()  # across the seam
    assert np.isnan(result.values[:, 1]).all()  # in the polar cap


def test_a_single_longitude_cell_keeps_its_declared_extent() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    src = source(
        [0.0],
        LAT,
        mapping=transform(sample_offset=(0.5, 0.5)),
        intervals={"lon": [[-0.5, 0.5]], "lat": LAT[:, None] + [-1.0, 1.0]},
    )
    # Both halves of the cell, on two branches; 1.0 and -1.0 lie outside it.
    targets = [0.25, -0.25, 360.25, 359.75, 1.0, -1.0]
    result = resample(src, grid(targets, [-30.0]), domain="cells")
    assert np.isfinite(result[:4]).all()
    assert np.isnan(result[4:]).all()


def test_unordered_longitude_labels_are_refused() -> None:
    with pytest.raises(ValueError, match="not strictly monotonic"):
        resample(source([0.0, 180.0, 90.0, 270.0], LAT), grid([100.0], [-30.0]))


def _paths(monkeypatch: pytest.MonkeyPatch, array: xr.DataArray) -> list[str]:
    """Which execution path resampling a framed array onto its own grid takes."""
    import xarrayrf._resampling as resampling

    taken: list[str] = []
    with monkeypatch.context() as patch:
        for name in ("_lattice_block", "_general_block"):
            real = getattr(resampling, name)

            def spy(
                *args: object, _real: object = real, _name: str = name, **kwargs: object
            ) -> object:
                taken.append(_name)
                return _real(*args, **kwargs)  # type: ignore[operator]

            patch.setattr(resampling, name, spy)
        xrf.resample(array.rf.geometry, array.rf.grid)
    return taken


def test_only_wrapping_sources_leave_the_lattice_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A performance contract invisible in results: unaffected sources keep their path.

    Uniform latitudes, since a lattice needs uniform samples on every axis.
    """
    pytest.importorskip("scipy", minversion="1.18")
    assert _paths(monkeypatch, source(GLOBAL_LON, UNIFORM_LAT)) == ["_general_block"]
    cartesian = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("lon", "lat"), ("m", "m")))
    for matrix in (((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.1), (0.0, 1.0))):
        angular = source(GLOBAL_LON, UNIFORM_LAT, mapping=transform(matrix=matrix))
        plain = frame_array(
            np.zeros((GLOBAL_LON.size, UNIFORM_LAT.size)),
            grid(GLOBAL_LON, UNIFORM_LAT, mapping=transform(cartesian, matrix=matrix)),
        )
        expected = _paths(monkeypatch, plain)
        if matrix[0][1]:
            # A coupled (rotated) mapping does not wrap, so it keeps the Cartesian path.
            assert _paths(monkeypatch, angular) == expected
        else:
            assert expected == ["_lattice_block"]
