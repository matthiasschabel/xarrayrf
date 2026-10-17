"""Public behaviour of the chart transitions between angular and Cartesian charts."""

from __future__ import annotations

import copy
import json
import math
import pickle
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import xarrayrf as xrf
from xarrayrf.native import frame_array


class Figure(TypedDict):
    semi_major_axis: float
    inverse_flattening: float


WGS84: Figure = {"semi_major_axis": 6378137.0, "inverse_flattening": 298.257223563}
A = WGS84["semi_major_axis"]
E2 = (1 / WGS84["inverse_flattening"]) * (2 - 1 / WGS84["inverse_flattening"])
EVOLUTE = E2 * A / math.sqrt(1 - E2)  # e² a² / b, about 42.84 km
ECEF = xrf.CoordinateSystem(("x", "y", "z"), ("m", "m", "m"))

# Measured agreement with pyproj (PROJ 9) was 0 m forward and about 1e-9 m and 1e-14 degrees
# inverse; the tolerances leave headroom for other PROJ builds and platforms.
FORWARD_ATOL_M = 1e-6
HEIGHT_ATOL_M = 1e-6
ANGLE_ATOL_DEG = 1e-10


def geodetic(
    axes: tuple[str, ...] = ("lon", "lat", "h"),
    units: tuple[str, ...] = ("deg", "deg", "m"),
    roles: tuple[xrf.AxisRole, ...] = ("longitude", "latitude", "height"),
) -> xrf.CoordinateSystem:
    return xrf.CoordinateSystem(axes, units, representation="geodetic", axis_roles=roles)


def spherical(units: tuple[str, ...] = ("deg", "deg", "pc")) -> xrf.CoordinateSystem:
    return xrf.CoordinateSystem(
        ("ra", "dec", "distance"),
        units,
        representation="spherical",
        axis_roles=("longitude", "latitude", "radius"),
    )


def charts(
    angular: xrf.CoordinateSystem,
    cartesian: xrf.CoordinateSystem = ECEF,
    definition: dict[str, Any] | None = None,
) -> tuple[xrf.ReferenceFrame, xrf.ReferenceFrame]:
    frame = xrf.ReferenceFrame.declared(("test", "body"), angular, definition=definition)
    return frame, frame.with_coordinate_system(cartesian)


def wgs84(**kwargs: Any) -> xrf.GeodeticToCartesian:
    return xrf.GeodeticToCartesian(*charts(geodetic()), **WGS84, **kwargs)


def oracle_points() -> npt.NDArray[np.float64]:
    latitudes = [-90.0, -90 + math.degrees(1e-9), -60.0, -1e-7, 0.0, 30.0, 89.99999, 90.0]
    longitudes = [-179.9, -90.0, 0.0, 45.0, 179.9]
    heights = [-10e3, 0.0, 1e6]
    grid = np.meshgrid(longitudes, latitudes, heights, indexing="ij")
    return np.stack(grid, axis=-1).reshape(-1, 3)


def test_geodetic_transition_matches_pyproj_both_ways() -> None:
    pyproj = pytest.importorskip("pyproj")
    points = oracle_points()
    expected = np.stack(
        pyproj.Transformer.from_crs(4979, 4978, always_xy=True).transform(*points.T), axis=-1
    )
    transition = wgs84()
    assert_allclose(transition.transform_point(points), expected, atol=FORWARD_ATOL_M, rtol=0)
    # PROJ's pole images lie a few 1e-10 m off the axis, so their longitude survives too.
    back = transition.inverse().transform_point(expected)
    assert_allclose(back[:, :2], points[:, :2], atol=ANGLE_ATOL_DEG, rtol=0)
    assert_allclose(back[:, 2], points[:, 2], atol=HEIGHT_ATOL_M, rtol=0)


def test_spherical_transition_matches_astropy() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "astropy_spherical.json").read_text()
    )
    points, expected = np.array(fixture["spherical"]), np.array(fixture["cartesian"])
    transition = xrf.SphericalToCartesian(
        *charts(spherical(), xrf.CoordinateSystem(("x", "y", "z"), ("pc",) * 3)),
        longitude_start=0.0,
    )
    # Distances span 1e-3 to 1e6 pc, so the allowance scales with each row's distance. At exact
    # poles Astropy leaves cos(π/2) ≈ 6e-17 of residue where the transition gives exact zeros.
    allowance = 1e-15 * points[:, 2:]
    assert np.all(np.abs(transition.transform_point(points) - expected) <= allowance)
    # Astropy's pole images lie just off the axis, so their longitude survives the inverse.
    back = transition.inverse().transform_point(expected)
    assert_allclose(back, points, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("start", [-180.0, 0.0, 180.0])
@pytest.mark.parametrize(
    "units, roles, axes",
    [
        (("deg", "deg", "m"), ("longitude", "latitude", "height"), ("lon", "lat", "h")),
        (("m", "rad", "rad"), ("height", "latitude", "longitude"), ("h", "lat", "lon")),
    ],
)
def test_round_trips_in_any_axis_order_and_unit(
    start: float, units: tuple[str, ...], roles: tuple[xrf.AxisRole, ...], axes: tuple[str, ...]
) -> None:
    radians = units.count("rad") == 2
    scale = math.pi / 180.0 if radians else 1.0
    rng = np.random.default_rng(1)
    angles = {
        "longitude": rng.uniform(start, start + 360.0, 50) * scale,
        "latitude": rng.uniform(-89.0, 89.0, 50) * scale,
        "height": rng.uniform(-5e3, 5e5, 50),
    }
    points = np.stack([angles[role] for role in roles], axis=-1)
    transition = xrf.GeodeticToCartesian(
        *charts(geodetic(axes, units, roles)), **WGS84, longitude_start=start * scale
    )
    back = transition.inverse().transform_point(transition.transform_point(points))
    assert_allclose(back, points, atol=1e-6, rtol=0)  # metres, and angles far below that


def test_inverse_longitudes_lie_in_the_declared_interval() -> None:
    frame, cartesian = charts(geodetic())
    inverse = xrf.CartesianToGeodetic(cartesian, frame, **WGS84, longitude_start=0.0)
    xyz = wgs84().transform_point([[-90.0, 0.0, 0.0], [-1e-12, 10.0, 0.0], [180.0, 0.0, 0.0]])
    longitudes = inverse.transform_point(xyz)[:, 0]
    assert np.all((longitudes >= 0.0) & (longitudes < 360.0))
    assert_allclose(longitudes, [270.0, 360.0 - 1e-12, 180.0], atol=1e-9, rtol=0)


@pytest.mark.parametrize(
    "units, pole", [(("deg", "deg", "m"), 90.0), (("rad", "rad", "m"), math.pi / 2)]
)
def test_exact_poles_land_on_the_axis(units: tuple[str, ...], pole: float) -> None:
    transition = xrf.GeodeticToCartesian(*charts(geodetic(units=units)), **WGS84)
    xyz = transition.transform_point([[0.7, pole, 0.0], [0.7, -pole, 10.0]])
    assert np.all(xyz[:, :2] == 0.0)
    back = transition.inverse().transform_point(xyz)
    assert np.all(back[:, 0] == transition.longitude_start)
    assert_allclose(back[:, 1:], [[pole, 0.0], [-pole, 10.0]], atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match=r"point \(1,\) is on the polar axis"):
        transition.inverse().jacobian(at=np.array([[A, 0.0, 0.0], xyz[0]]))


def finite_difference(
    transition: xrf.SupportsPoints, point: npt.NDArray[np.float64], steps: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    columns = []
    for k, step in enumerate(np.asarray(steps, dtype=float)):
        offset = np.zeros(3)
        offset[k] = step
        ahead = transition.transform_point(point + offset)
        behind = transition.transform_point(point - offset)
        columns.append((ahead - behind) / (2 * step))
    return np.stack(columns, axis=-1)


def test_jacobians_match_finite_differences_and_invert_each_other() -> None:
    transition = wgs84()
    points = np.array([[10.0, 45.0, 100.0], [-170.0, -60.0, -5e3], [95.0, 1.0, 2e5]])
    jacobians = transition.jacobian(at=points)
    assert jacobians.shape == (3, 3, 3)
    for point, jacobian in zip(points, jacobians, strict=True):
        # Steps of 1e-6 degree and 1 mm; second-order error is far below the rtol.
        estimate = finite_difference(transition, point, [1e-6, 1e-6, 1e-3])
        assert_allclose(estimate, jacobian, rtol=1e-6, atol=1e-6)
    inverse = transition.inverse().jacobian(at=transition.transform_point(points))
    assert_allclose(inverse @ jacobians, np.broadcast_to(np.eye(3), (3, 3, 3)), atol=1e-10, rtol=0)
    spherical_transition = xrf.SphericalToCartesian(
        *charts(spherical(("rad", "rad", "pc")), xrf.CoordinateSystem(("x", "y", "z"), ("pc",) * 3))
    )
    point = np.array([1.0, 0.3, 2.0])
    assert_allclose(
        finite_difference(spherical_transition, point, [1e-7, 1e-7, 1e-7]),
        spherical_transition.jacobian(at=point),
        rtol=1e-7,
        atol=1e-9,
    )


def test_the_inverse_jacobian_stays_finite_just_off_the_axis() -> None:
    """Here the computed latitude rounds to exactly 90°, but the distance from the axis is not 0."""
    inverse = wgs84().inverse()
    point = np.array([[1e-9, 0.0, 6356752.314245]])
    assert inverse.transform_point(point)[0, 1] == 90.0
    jacobian = inverse.jacobian(at=point)[0]
    assert np.all(np.isfinite(jacobian))
    # d(longitude)/dy is 1 / (distance from the axis), in degrees.
    assert_allclose(jacobian[0, 1], math.degrees(1.0 / 1e-9), rtol=1e-12)


def test_the_forward_jacobian_is_singular_at_the_poles() -> None:
    jacobian = wgs84().jacobian(at=[[30.0, 90.0, 0.0]])[0]
    assert np.all(np.isfinite(jacobian))
    assert np.all(jacobian[:, 0] == 0.0)  # longitude moves nothing on the axis


def test_the_jacobian_needs_points() -> None:
    with pytest.raises(TypeError, match="pass at="):
        wgs84().jacobian()


@pytest.mark.parametrize("inverse_flattening", [0.0, WGS84["inverse_flattening"]])
@pytest.mark.parametrize("latitude", [0.0, 45.0, -80.0])
def test_the_forward_domain_is_the_inverse_image(
    inverse_flattening: float, latitude: float
) -> None:
    transition = xrf.GeodeticToCartesian(
        *charts(geodetic()), semi_major_axis=A, inverse_flattening=inverse_flattening
    )
    e2 = 0.0 if inverse_flattening == 0.0 else E2
    sine = math.sin(math.radians(latitude))
    limit = -A / math.sqrt(1 - e2 * sine**2) * (1 - e2)
    # Exactly at the limit, rounding decides; a metre below it is on the far side.
    with pytest.raises(ValueError, match=r"point \(0,\) reaches the far side of the centre"):
        transition.transform_point([[10.0, latitude, limit - 1.0]])
    deepest = np.array([[10.0, latitude, limit + 1.0]])
    if e2:
        # The normal meets the equatorial plane N e² cos φ from the centre, never more than
        # e² a² / b, so on an ellipsoid the evolute bound is reached before the height limit.
        with pytest.raises(ValueError, match=r"maps within 42841\.\d+ m of the centre"):
            transition.transform_point(deepest)
    else:
        back = transition.inverse().transform_point(transition.transform_point(deepest))
        assert_allclose(back, deepest, atol=1e-6, rtol=0)


def test_points_just_outside_the_evolute_round_trip() -> None:
    """On the equator the evolute bound is crossed at x = e² a² / b, deep inside the Earth."""
    transition = wgs84()
    height = EVOLUTE * 1.001 - A
    point = np.array([[25.0, 0.0, height]])
    assert np.linalg.norm(transition.transform_point(point)) > EVOLUTE
    back = transition.inverse().transform_point(transition.transform_point(point))
    assert_allclose(back, point, atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match=r"point \(0,\) is at or within 42841"):
        transition.inverse().transform_point([[EVOLUTE * 0.999, 0.0, 0.0]])


def test_the_spherical_centre_is_outside_both_directions() -> None:
    frame, cartesian = charts(spherical(), xrf.CoordinateSystem(("x", "y", "z"), ("pc",) * 3))
    forward = xrf.SphericalToCartesian(frame, cartesian)
    with pytest.raises(ValueError, match="far side"):
        forward.transform_point([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="at or within the centre"):
        forward.inverse().transform_point([[0.0, 0.0, 0.0]])


def test_latitude_beyond_the_poles_is_refused() -> None:
    with pytest.raises(ValueError, match=r"latitude outside ±90 at point \(1,\)"):
        wgs84().transform_point([[0.0, 0.0, 0.0], [0.0, 90.000001, 0.0]])
    # The roundoff band is POLE_ROUNDOFF (1e-12) of the pole, 9e-11°.
    with pytest.raises(ValueError, match=r"latitude outside ±90 at point \(0,\)"):
        wgs84().transform_point([[0.0, -90.0 - 1e-9, 0.0]])


@pytest.mark.parametrize(
    "units, pole", [(("deg", "deg", "m"), 90.0), (("rad", "rad", "m"), math.pi / 2)]
)
def test_roundoff_past_a_pole_lands_on_it(units: tuple[str, ...], pole: float) -> None:
    transition = xrf.GeodeticToCartesian(*charts(geodetic(units=units)), **WGS84)
    exact = transition.transform_point([[0.7, pole, 5.0], [0.7, -pole, 5.0]])
    past = [pole * (1 + 1e-13), np.nextafter(pole, 2 * pole)]
    xyz = transition.transform_point(
        [[0.7, past[0], 5.0], [0.7, past[1], 5.0], [0.7, -past[0], 5.0]]
    )
    assert np.all(xyz[:, :2] == 0.0)
    # Snapped onto the pole, the points are the pole's own (1e-9 m on a 6e6 m radius).
    assert_allclose(xyz, exact[[0, 0, 1]], atol=1e-9, rtol=0)


@pytest.mark.parametrize(
    "build, error, message",
    [
        pytest.param(
            lambda f, c: xrf.GeodeticToCartesian(f, "frame", **WGS84),  # type: ignore[arg-type]
            TypeError,
            "target must be a ReferenceFrame",
            id="endpoint-type",
        ),
        pytest.param(
            lambda f, c: xrf.GeodeticToCartesian(f, xrf.ReferenceFrame.local(ECEF), **WGS84),
            ValueError,
            "are not the same frame",
            id="different-frames",
        ),
        pytest.param(
            lambda f, c: xrf.GeodeticToCartesian(c, f, **WGS84),
            ValueError,
            "source must be a geodetic chart",
            id="reversed",
        ),
        pytest.param(
            lambda f, c: xrf.SphericalToCartesian(f, c),
            ValueError,
            "source must be a spherical chart",
            id="wrong-representation",
        ),
        pytest.param(
            lambda f, c: xrf.GeodeticToCartesian(
                f,
                c.with_coordinate_system(xrf.CoordinateSystem(("x", "y", "z"), ("km",) * 3)),
                **WGS84,
            ),
            ValueError,
            "must all be in the radial unit 'm'",
            id="unit",
        ),
        pytest.param(
            lambda f, c: xrf.GeodeticToCartesian(
                f.with_coordinate_system(
                    geodetic(("lon", "lat"), ("deg", "deg"), ("longitude", "latitude"))
                ),
                c,
                **WGS84,
            ),
            ValueError,
            "exactly three axes",
            id="no-height",
        ),
        pytest.param(
            lambda f, c: xrf.GeodeticToCartesian(
                f, c.with_coordinate_system(xrf.CoordinateSystem(("x", "y"), ("m", "m"))), **WGS84
            ),
            ValueError,
            "Cartesian chart with exactly three axes",
            id="two-cartesian-axes",
        ),
        pytest.param(
            lambda f, c: xrf.GeodeticToCartesian(f, c, semi_major_axis=0.0, inverse_flattening=0),
            ValueError,
            "semi_major_axis must be positive",
            id="radius",
        ),
        pytest.param(
            lambda f, c: xrf.GeodeticToCartesian(f, c, semi_major_axis=A, inverse_flattening=1.0),
            ValueError,
            "0 \\(a sphere\\) or greater than 1",
            id="flattening",
        ),
        pytest.param(
            lambda f, c: xrf.GeodeticToCartesian(
                f, c, semi_major_axis=A, inverse_flattening=math.inf
            ),
            ValueError,
            "inverse_flattening must be finite",
            id="infinite-flattening",
        ),
        pytest.param(
            lambda f, c: xrf.GeodeticToCartesian(
                f,
                c,
                semi_major_axis="6378137",  # type: ignore[arg-type]
                inverse_flattening=0,
            ),
            TypeError,
            "semi_major_axis must be a real number",
            id="string-radius",
        ),
        pytest.param(
            lambda f, c: xrf.GeodeticToCartesian(f, c, **WGS84, longitude_start=True),
            TypeError,
            "longitude_start must be a real number",
            id="bool-start",
        ),
    ],
)
def test_constructor_refusals(build: Any, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        build(*charts(geodetic()))


SPHERE_DEFINITION = {"figure": "sphere", "semi_major_axis": 6371229.0, "length_unit": "m"}
WGS84_DEFINITION = {
    "figure": "ellipsoid",
    "semi_major_axis": A,
    "inverse_flattening": WGS84["inverse_flattening"],
    "length_unit": "m",
}


@pytest.mark.parametrize(
    "definition, figure",
    [
        (SPHERE_DEFINITION, {"semi_major_axis": 6371229.0, "inverse_flattening": 0.0}),
        (WGS84_DEFINITION, WGS84),
        ({"datum_name": "anything"}, WGS84),
    ],
    ids=["sphere", "ellipsoid", "no-figure"],
)
def test_a_figure_agreeing_with_the_definition_is_accepted(
    definition: dict[str, Any], figure: Figure
) -> None:
    xrf.GeodeticToCartesian(*charts(geodetic(), definition=definition), **figure)


@pytest.mark.parametrize(
    "definition, figure, key",
    [
        ({**WGS84_DEFINITION, "figure": "sphere"}, WGS84, "figure"),
        ({**WGS84_DEFINITION, "semi_major_axis": 6378136.0}, WGS84, "semi_major_axis"),
        ({**WGS84_DEFINITION, "inverse_flattening": 298.0}, WGS84, "inverse_flattening"),
        ({**WGS84_DEFINITION, "length_unit": "km"}, WGS84, "length_unit"),
        ({k: v for k, v in WGS84_DEFINITION.items() if k != "length_unit"}, WGS84, "length_unit"),
        (
            {**SPHERE_DEFINITION, "inverse_flattening": 0.0},
            {"semi_major_axis": 6371229.0, "inverse_flattening": 0.0},
            "inverse_flattening",
        ),
    ],
    ids=["figure", "radius", "flattening", "unit", "missing-unit", "sphere-flattening"],
)
def test_a_figure_disagreeing_with_the_definition_is_refused(
    definition: dict[str, Any], figure: Figure, key: str
) -> None:
    with pytest.raises(ValueError, match=f"declares {key}="):
        xrf.GeodeticToCartesian(*charts(geodetic(), definition=definition), **figure)


def test_comparison_hashing_repr_pickling_and_inverse_identity() -> None:
    transition = wgs84(longitude_start=0.0)
    assert transition == wgs84(longitude_start=0.0)
    assert hash(transition) == hash(wgs84(longitude_start=0.0))
    assert transition != wgs84()
    assert transition.inverse().inverse() == transition
    assert transition != transition.inverse()
    assert pickle.loads(pickle.dumps(transition)) == transition
    assert copy.deepcopy(transition.inverse()) == transition.inverse()
    assert "semi_major_axis=6378137.0" in repr(transition)
    assert repr(transition).startswith("GeodeticToCartesian(source=ReferenceFrame(")
    assert isinstance(transition, xrf.SupportsPoints | xrf.SupportsJacobian | xrf.SupportsInverse)


def test_transitions_round_trip_through_encoding() -> None:
    frame, cartesian = charts(spherical(), xrf.CoordinateSystem(("x", "y", "z"), ("pc",) * 3))
    values = [
        wgs84(longitude_start=0.0),
        wgs84().inverse(),
        xrf.SphericalToCartesian(frame, cartesian, longitude_start=-math.pi),
        xrf.CartesianToSpherical(cartesian, frame),
    ]
    for value in values:
        decoded = xrf.decode(json.loads(json.dumps(xrf.encode(value))))
        assert decoded == value and type(decoded) is type(value)
    to_ecef = wgs84()
    shift = xrf.AffineTransform.from_matrix(
        source=to_ecef.target, target=to_ecef.target, matrix=np.eye(3), translation=[1.0, 0, 0]
    )
    composite = xrf.CompositeTransform(to_ecef, shift)
    assert xrf.decode(xrf.encode(composite)) == composite


def test_a_malformed_transition_record_is_refused() -> None:
    data = xrf.encode(wgs84())
    data["value"]["semi_major_axis"] = "big"
    with pytest.raises(xrf.MalformedDataError, match="semi_major_axis must be a real number"):
        xrf.decode(data)


def test_resampling_a_geodetic_grid_onto_earth_centred_points() -> None:
    """A field linear in longitude, latitude and height is reproduced by linear resampling."""
    frame, cartesian = charts(geodetic())
    to_geodetic = xrf.CartesianToGeodetic(cartesian, frame, **WGS84)
    lon, lat, h = np.arange(20.0, 41.0), np.arange(30.0, 51.0), np.array([-2e3, 0.0, 2e3])
    source_transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("lon", "lat", "h"), ("deg", "deg", "m")),
        target=frame,
        matrix=np.eye(3),
        translation=(0.0, 0.0, 0.0),
    )
    grid = np.meshgrid(lon, lat, h, indexing="ij")
    source = frame_array(
        grid[0] + 10.0 * grid[1] + grid[2] / 1000.0,
        xrf.Grid(source_transform, {"lon": ("lon", lon), "lat": ("lat", lat), "h": ("h", h)}),
    )
    centre = wgs84().transform_point([30.0, 40.0, 0.0])
    axes = {
        name: (name, centre[k] + np.array([-500.0, 0.0, 700.0])) for k, name in enumerate("xyz")
    }
    target_transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("x", "y", "z"), ("m", "m", "m")),
        target=cartesian,
        matrix=np.eye(3),
        translation=(0.0, 0.0, 0.0),
    )
    target = xrf.Grid(target_transform, axes)
    result = xrf.resample(source.rf.geometry, target, transform=to_geodetic)
    xyz = np.stack(np.meshgrid(*(axes[n][1] for n in "xyz"), indexing="ij"), axis=-1)
    expected = to_geodetic.transform_point(xyz) @ [1.0, 10.0, 1e-3]
    assert isinstance(result, xr.DataArray)
    # Multilinear interpolation of a linear field is exact up to float64 roundoff of degrees.
    assert_allclose(result.transpose("x", "y", "z").values, expected, atol=1e-9, rtol=0)


def test_the_inverse_is_compared_with_projs_own_inverse() -> None:
    """PROJ's geocentric inverse is an approximation that drifts at altitude.

    Measured against PROJ 9.8: 7.5 mm in height and 3.4e-8 degrees in latitude at 1000 km,
    where this inverse recovers PROJ's forward inputs to about 1e-9 m. The tolerances bound
    PROJ's error, not this implementation's.
    """
    pyproj = pytest.importorskip("pyproj")
    transition = wgs84()
    xyz = transition.transform_point(oracle_points())
    expected = np.stack(
        pyproj.Transformer.from_crs(4978, 4979, always_xy=True).transform(*xyz.T), axis=-1
    )
    ours = transition.inverse().transform_point(xyz)
    # On the polar axis PROJ answers longitude 0 and this inverse its longitude_start.
    off_axis = np.hypot(xyz[:, 0], xyz[:, 1]) > 0.0
    assert_allclose(ours[off_axis, 0], expected[off_axis, 0], atol=1e-7, rtol=0)
    assert_allclose(ours[:, 1], expected[:, 1], atol=1e-7, rtol=0)
    assert_allclose(ours[:, 2], expected[:, 2], atol=0.02, rtol=0)


def test_jacobians_that_overflow_float64_raise() -> None:
    frame, cartesian = charts(spherical(("deg", "deg", "m")))
    inverse = xrf.CartesianToSpherical(cartesian, frame)
    with pytest.raises(ValueError, match=r"Jacobian at point \(1,\) is not finite"):
        inverse.jacobian(at=[[1.0, 1.0, 1.0], [1e-307, 0.0, 1.0]])


def test_a_flattening_near_one_keeps_one_minus_e_squared() -> None:
    """e² rounds to 1 here; 1 - e² is carried as (1 - f)² instead of cancelling to zero."""
    frame, cartesian = charts(geodetic())
    transition = xrf.GeodeticToCartesian(
        frame, cartesian, semi_major_axis=1.0, inverse_flattening=1.000000001
    )
    # The equator, and the pole, where 1 - e² sin²φ would also cancel if subtracted.
    point = np.array([[0.0, 0.0, 1e12], [-180.0, 90.0, 1e12]])
    assert_allclose(
        transition.inverse().transform_point(transition.transform_point(point)),
        point,
        rtol=1e-12,
    )
    assert np.all(np.isfinite(transition.jacobian(at=point)))


def test_points_need_three_finite_coordinates() -> None:
    with pytest.raises(ValueError, match="trailing axis of length 3"):
        wgs84().transform_point([[0.0, 0.0]])
    with pytest.raises(ValueError, match="trailing axis of length 3"):
        wgs84().jacobian(at=5.0)
    assert wgs84().transform_point(np.empty((0, 3))).shape == (0, 3)
    assert wgs84().jacobian(at=np.empty((2, 0, 3))).shape == (2, 0, 3, 3)


def test_an_overflowing_cartesian_point_raises() -> None:
    with pytest.raises(ValueError, match="non-finite coordinates"):
        wgs84().inverse().transform_point([[1e308, 1e308, 0.0]])


def test_mixed_degree_and_radian_angles_round_trip() -> None:
    transition = xrf.GeodeticToCartesian(*charts(geodetic(units=("deg", "rad", "m"))), **WGS84)
    points = np.array([[[10.0, 0.5, 100.0], [-120.0, -1.2, -50.0]]])
    xyz = transition.transform_point(points)
    assert xyz.shape == (1, 2, 3)
    assert_allclose(transition.inverse().transform_point(xyz), points, atol=1e-6, rtol=0)
    expected = xrf.GeodeticToCartesian(*charts(geodetic()), **WGS84).transform_point(
        [[10.0, math.degrees(0.5), 100.0], [-120.0, math.degrees(-1.2), -50.0]]
    )
    assert_allclose(xyz[0], expected, atol=1e-6, rtol=0)


def test_a_chart_without_its_radial_axis_is_refused() -> None:
    system = xrf.CoordinateSystem(
        ("lon", "lat", "t"),
        ("deg", "deg", "s"),
        representation="spherical",
        axis_roles=("longitude", "latitude", None),
    )
    with pytest.raises(ValueError, match="source chart needs a radius axis"):
        xrf.SphericalToCartesian(*charts(system))


def test_a_transition_record_with_array_endpoints_is_malformed() -> None:
    data = xrf.encode(wgs84())
    data["value"]["source"] = xrf.encode(xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3))["value"]
    with pytest.raises(xrf.MalformedDataError, match="endpoints must be reference frames"):
        xrf.decode(data)


def polar_source(components: bool) -> xr.DataArray:
    """A geodetic grid whose last latitude row is the north pole."""
    frame, _ = charts(geodetic())
    lon, lat, h = np.arange(0.0, 360.0, 30.0), np.array([80.0, 85.0, 90.0]), np.array([0.0, 1.0])
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("lon", "lat", "h"), ("deg", "deg", "m")),
        target=frame,
        matrix=np.eye(3),
        translation=(0.0, 0.0, 0.0),
    )
    grid = xrf.Grid(transform, {"lon": ("lon", lon), "lat": ("lat", lat), "h": ("h", h)})
    shape = (lon.size, lat.size, h.size)
    if not components:
        return frame_array(np.full(shape, 7.0), grid)
    plain = xr.DataArray(
        np.ones((*shape, 3)),
        dims=("lon", "lat", "h", "component"),
        coords={"component": list(frame.axes)},
    )
    array = cast(xr.DataArray, plain.rf.frame(grid))
    declared: xr.DataArray = array.rf.declare_tensor_components(
        xrf.VECTOR, component_dims=("component",), bases=frame
    )
    return declared


def polar_target() -> xrf.Grid:
    """One Earth-centred point on the polar axis, half a metre above the pole."""
    _, cartesian = charts(geodetic())
    b = A * (1 - 1 / WGS84["inverse_flattening"])
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("x", "y", "z"), ("m", "m", "m")),
        target=cartesian,
        matrix=np.eye(3),
        translation=(0.0, 0.0, 0.0),
    )
    return xrf.Grid(transform, {"x": ("x", [0.0]), "y": ("y", [0.0]), "z": ("z", [b + 0.5])})


def test_scalar_resampling_reaches_the_pole_through_a_transition() -> None:
    frame, cartesian = charts(geodetic())
    result = xrf.resample(
        polar_source(components=False).rf.geometry,
        polar_target(),
        # The axis inverts to longitude_start, which must lie among the 0…330 labels: the
        # seam does not wrap yet.
        transform=xrf.CartesianToGeodetic(cartesian, frame, **WGS84, longitude_start=0.0),
    )
    assert_allclose(result.values.ravel(), [7.0], atol=1e-12, rtol=0)


def test_component_resampling_through_the_pole_names_the_target_sample() -> None:
    frame, cartesian = charts(geodetic())
    with pytest.raises(ValueError, match=r"undefined at target sample \{'x': 0, 'y': 0, 'z': 0\}"):
        xrf.resample(
            polar_source(components=True).rf.geometry,
            polar_target(),
            transform=xrf.CartesianToGeodetic(cartesian, frame, **WGS84, longitude_start=0.0),
            output_component_basis=cartesian,
            component_reorientation="jacobian",
            units="m",
        )
