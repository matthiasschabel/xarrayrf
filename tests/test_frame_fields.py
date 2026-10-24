"""Pointwise component bases: ``FrameField``, ``EastNorthUp`` and the operations that use them."""

from __future__ import annotations

import json
import math
from typing import Any, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import xarrayrf as xrf
import xarrayrf.native  # registers the .rf accessor

A = 6378137.0
INVERSE_F = 298.257223563


class Figure(TypedDict):
    semi_major_axis: float
    inverse_flattening: float


FIGURE: Figure = {"semi_major_axis": A, "inverse_flattening": INVERSE_F}
E2 = (1 / INVERSE_F) * (2 - 1 / INVERSE_F)
OMEGA = (
    7.292115e-5  # Earth's rotation rate, rad/s: the solid-body wind is OMEGA x r (cross product)
)


def geodetic(
    axes: tuple[str, ...] = ("lon", "lat", "h"),
    units: tuple[str, ...] = ("degrees_east", "degrees_north", "m"),
    roles: tuple[xrf.AxisRole, ...] = ("longitude", "latitude", "height"),
    definition: dict[str, Any] | None = None,
) -> xrf.ReferenceFrame:
    system = xrf.CoordinateSystem(axes, units, representation="geodetic", axis_roles=roles)
    return xrf.ReferenceFrame.declared(("test", "wgs84"), system, definition=definition)


EARTH = geodetic()
ECEF = EARTH.with_coordinate_system(xrf.CoordinateSystem(("x", "y", "z"), ("m", "m", "m")))
TO_ECEF = xrf.GeodeticToCartesian(EARTH, ECEF, **FIGURE)
ENU = xrf.EastNorthUp(EARTH, **FIGURE)


def random_points(count: int = 40, seed: int = 3) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(-180, 180, count),
            rng.uniform(-89.5, 89.5, count),
            rng.uniform(-1e3, 1e5, count),
        ],
        axis=-1,
    )


# The field


def test_east_north_up_columns_are_orthonormal_and_point_east_north_up() -> None:
    points = random_points()
    columns = TO_ECEF.jacobian(at=points) @ ENU.matrix(points)
    assert_allclose(columns, ENU.directions(points), atol=1e-12, rtol=0)
    gram = np.swapaxes(columns, -1, -2) @ columns
    assert_allclose(gram, np.broadcast_to(np.eye(3), gram.shape), atol=1e-12, rtol=0)
    lam, phi = np.radians(points[:, 0]), np.radians(points[:, 1])
    east = np.stack([-np.sin(lam), np.cos(lam), np.zeros_like(lam)], axis=-1)
    assert_allclose(columns[..., 0], east, atol=1e-12, rtol=0)
    assert_allclose(columns[..., 2, 1], np.cos(phi), atol=1e-12, rtol=0)  # north rises in z
    inverse = ENU.inverse_matrix(points) @ ENU.matrix(points)
    assert_allclose(inverse, np.broadcast_to(np.eye(3), inverse.shape), atol=1e-12, rtol=0)


def test_components_follow_roles_whatever_the_chart_axis_order() -> None:
    reordered = geodetic(
        ("lat", "h", "lon"), ("rad", "m", "degrees_east"), ("latitude", "height", "longitude")
    )
    field = xrf.EastNorthUp(reordered, **FIGURE)
    transition = xrf.GeodeticToCartesian(
        reordered,
        reordered.with_coordinate_system(ECEF.coordinate_system),
        **FIGURE,
    )
    points = random_points()
    shuffled = np.stack([np.radians(points[:, 1]), points[:, 2], points[:, 0]], axis=-1)
    columns = transition.jacobian(at=shuffled) @ field.matrix(shuffled)
    assert field.axes == ("east", "north", "up")
    assert_allclose(columns, ENU.directions(points), atol=1e-12, rtol=0)


def test_a_two_dimensional_field_lies_on_the_surface() -> None:
    surface = geodetic(("lon", "lat"), ("degrees_east", "degrees_north"), ("longitude", "latitude"))
    field = xrf.EastNorthUp(surface, **FIGURE, length_unit="m")
    points = random_points()[:, :2]
    on_surface = np.concatenate([points, np.zeros((len(points), 1))], axis=-1)
    tangent = TO_ECEF.jacobian(at=on_surface)[..., :, :2]
    assert field.axes == ("east", "north")
    assert field.units == ("m", "m")
    assert_allclose(tangent @ field.matrix(points), ENU.directions(on_surface)[..., :2], atol=1e-12)


def test_a_spherical_chart_with_a_radius_needs_no_figure() -> None:
    system = xrf.CoordinateSystem(
        ("ra", "dec", "r"),
        ("rad", "rad", "pc"),
        representation="spherical",
        axis_roles=("longitude", "latitude", "radius"),
    )
    sky = xrf.ReferenceFrame.local(system)
    field = xrf.EastNorthUp(sky)
    to_cartesian = xrf.SphericalToCartesian(
        sky, sky.with_coordinate_system(xrf.CoordinateSystem(("x", "y", "z"), ("pc",) * 3))
    )
    points = np.array([[0.3, -0.4, 2.0], [4.0, 1.2, 0.5]])
    columns = to_cartesian.jacobian(at=points) @ field.matrix(points)
    assert_allclose(
        np.swapaxes(columns, -1, -2) @ columns, np.broadcast_to(np.eye(3), (2, 3, 3)), atol=1e-12
    )
    assert field.matches(to_cartesian)


def test_the_chart_basis_does_not_exist_at_a_pole_but_the_directions_do() -> None:
    with pytest.raises(
        ValueError, match=r"at point \(1,\) a chart coordinate vector has no length"
    ):
        ENU.matrix([[0.0, 0.0, 0.0], [30.0, 90.0, 0.0]])
    with pytest.raises(ValueError, match="no length"):
        ENU.inverse_matrix([[30.0, -90.0, 0.0]])
    directions = ENU.directions([[30.0, 90.0, 0.0]])[0]
    lam = math.radians(30.0)
    # The meridian limit: east as on the meridian, north pointing back along it.
    assert_allclose(directions[:, 0], [-math.sin(lam), math.cos(lam), 0.0], atol=1e-15)
    assert_allclose(directions[:, 1], [-math.cos(lam), -math.sin(lam), 0.0], atol=1e-15)


@pytest.mark.parametrize(
    "build, error, message",
    [
        (lambda: xrf.EastNorthUp("frame"), TypeError, "base must be a ReferenceFrame"),  # type: ignore[arg-type]
        (lambda: xrf.EastNorthUp(ECEF), ValueError, "needs a geodetic or spherical chart"),
        (
            lambda: xrf.EastNorthUp(EARTH),
            ValueError,
            "needs semi_major_axis and inverse_flattening",
        ),
        (
            lambda: xrf.EastNorthUp(EARTH, semi_major_axis=-1.0, inverse_flattening=0.0),
            ValueError,
            "must be positive",
        ),
        (
            lambda: xrf.EastNorthUp(EARTH, semi_major_axis=A, inverse_flattening=0.5),
            ValueError,
            "0 \\(a sphere\\)",
        ),
        (
            lambda: xrf.EastNorthUp(EARTH, **FIGURE, length_unit="km"),
            ValueError,
            "disagrees with the chart's height unit",
        ),
        (
            lambda: xrf.EastNorthUp(EARTH, **FIGURE, axes=("e", "n")),
            ValueError,
            "must give 3 labels",
        ),
        (
            lambda: xrf.EastNorthUp(
                geodetic(("lon", "lat"), ("deg", "deg"), ("longitude", "latitude")), **FIGURE
            ),
            ValueError,
            "pass length_unit=",
        ),
        (
            lambda: xrf.EastNorthUp(
                xrf.ReferenceFrame.local(
                    xrf.CoordinateSystem(
                        ("ra", "dec"),
                        ("rad", "rad"),
                        representation="spherical",
                        axis_roles=("longitude", "latitude"),
                    )
                )
            ),
            ValueError,
            "spherical chart without a radius axis",
        ),
        (
            lambda: xrf.EastNorthUp(
                xrf.ReferenceFrame.local(
                    xrf.CoordinateSystem(
                        ("lon", "lat", "r"),
                        ("rad", "rad", "m"),
                        representation="spherical",
                        axis_roles=("longitude", "latitude", "radius"),
                    )
                ),
                **FIGURE,
            ),
            ValueError,
            "apply to geodetic charts",
        ),
        (
            lambda: xrf.EastNorthUp(
                geodetic(
                    definition={
                        "figure": "sphere",
                        "semi_major_axis": 6371000.0,
                        "length_unit": "m",
                    }
                ),
                **FIGURE,
            ),
            ValueError,
            "the field's figure has",
        ),
        (
            lambda: xrf.EastNorthUp(
                geodetic(("t", "lon", "lat"), ("s", "deg", "deg"), (None, "longitude", "latitude")),  # type: ignore[arg-type]
                **FIGURE,
            ),
            ValueError,
            "axes without a role",
        ),
    ],
)
def test_field_refusals(build: Any, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        build()


def test_fields_compare_hash_and_pickle_by_declaration() -> None:
    import pickle

    same = xrf.EastNorthUp(EARTH, **FIGURE)
    assert same == ENU and hash(same) == hash(ENU)
    assert xrf.EastNorthUp(EARTH, **FIGURE, axes=("u", "v", "w")) != ENU
    assert pickle.loads(pickle.dumps(ENU)) == ENU
    assert isinstance(ENU, xrf.FrameField)
    assert not isinstance(EARTH, xrf.FrameField)
    assert "EastNorthUp(base=" in repr(ENU)


# Arrays

LON = np.arange(0.0, 360.0, 30.0)
LAT = np.array([-90.0, -60.0, -20.0, 0.0, 30.0, 70.0, 90.0])
H = np.array([0.0, 500.0])


def grid(
    lon: npt.NDArray[np.float64] = LON,
    lat: npt.NDArray[np.float64] = LAT,
    h: npt.NDArray[np.float64] = H,
) -> xrf.Grid:
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("lon", "lat", "h"), EARTH.units),
        target=EARTH,
        matrix=np.eye(3),
        translation=(0.0, 0.0, 0.0),
    )
    return xrf.Grid(transform, {"lon": ("lon", lon), "lat": ("lat", lat), "h": ("h", h)})


def points_of(target: xrf.Grid) -> npt.NDArray[np.float64]:
    values = [np.asarray(target.coordinates[name][1], float) for name in ("lon", "lat", "h")]  # type: ignore[index]
    return np.stack(np.meshgrid(*values, indexing="ij"), axis=-1)


def solid_body_enu(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """East = OMEGA (N + h) cos φ, north = up = 0: the rotation of the Earth seen in place."""
    phi = np.radians(points[..., 1])
    n = A / np.sqrt(1 - E2 * np.sin(phi) ** 2)
    east = OMEGA * (n + points[..., 2]) * np.cos(phi)
    return np.stack([east, np.zeros_like(east), np.zeros_like(east)], axis=-1)


def declared(
    values: npt.NDArray[np.number[Any]],
    target: xrf.Grid,
    basis: xrf.ReferenceFrame | xrf.FrameField,
) -> xr.DataArray:
    plain = xr.DataArray(
        values, dims=("lon", "lat", "h", "component"), coords={"component": list(basis.axes)}
    )
    framed = cast(xr.DataArray, plain.rf.frame(target))
    result: xr.DataArray = framed.rf.declare_tensor_components(
        xrf.VECTOR, component_dims=("component",), bases=basis
    )
    return result


def ordered(array: xr.DataArray) -> npt.NDArray[np.float64]:
    return np.asarray(array.transpose("lon", "lat", "h", "component").values)


def test_east_north_up_converts_to_earth_centred_components_on_pole_rows_too() -> None:
    wind = declared(solid_body_enu(points_of(grid())), grid(), ENU)
    earth_centred = wind.rf.change_component_basis(ECEF, transform=TO_ECEF)
    expected = np.cross([0.0, 0.0, OMEGA], TO_ECEF.transform_point(points_of(grid())))
    assert_allclose(ordered(earth_centred), expected, atol=1e-9, rtol=0)
    assert earth_centred.rf.tensor_component_declaration.bases == (ECEF,)
    back = earth_centred.rf.change_component_basis(ENU, transform=TO_ECEF.inverse())
    assert_allclose(ordered(back), ordered(wind), atol=1e-9, rtol=0)


def test_the_chart_basis_round_trip_and_its_pole_refusal() -> None:
    inner = grid(lat=np.array([-60.0, 0.0, 45.0]))
    wind = declared(solid_body_enu(points_of(inner)), inner, ENU)
    chart = wind.rf.change_component_basis(EARTH, units="mixed")
    # East velocity over (N + h) cos φ is the longitude rate, here OMEGA in degrees per second.
    assert_allclose(ordered(chart)[..., 0], math.degrees(OMEGA), rtol=1e-12)
    back = chart.rf.change_component_basis(ENU, units="m s-1")
    assert_allclose(ordered(back), ordered(wind), atol=1e-9, rtol=0)
    with pytest.raises(ValueError, match=r"at sample \{'lon': 0, 'lat': 0, 'h': 0\}"):
        declared(solid_body_enu(points_of(grid())), grid(), ENU).rf.change_component_basis(
            EARTH, units="mixed"
        )


def test_units_are_required_exactly_when_they_change() -> None:
    wind = declared(solid_body_enu(points_of(grid())), grid(), ENU)
    with pytest.raises(ValueError, match="units= is only accepted"):
        wind.rf.change_component_basis(ECEF, transform=TO_ECEF, units="m s-1")
    inner = grid(lat=np.array([0.0]))
    with pytest.raises(ValueError, match="pass units="):
        declared(solid_body_enu(points_of(inner)), inner, ENU).rf.change_component_basis(EARTH)


def test_a_field_over_another_frame_is_refused() -> None:
    other = xrf.EastNorthUp(
        geodetic(
            ("lat", "lon", "h"),
            ("degrees_north", "degrees_east", "m"),
            ("latitude", "longitude", "height"),
        ),
        **FIGURE,
    )
    with pytest.raises(ValueError, match="frame field basis must be over this array's frame"):
        declared(np.zeros((*points_of(grid()).shape[:-1], 3)), grid(), other)


def test_a_pointwise_transform_must_touch_the_arrays_frame() -> None:
    elsewhere = xrf.ReferenceFrame.local(geodetic().coordinate_system)
    stray = xrf.GeodeticToCartesian(
        elsewhere, elsewhere.with_coordinate_system(ECEF.coordinate_system), **FIGURE
    )
    wind = declared(solid_body_enu(points_of(grid())), grid(), ENU)
    with pytest.raises(ValueError, match="start or end at the array's frame"):
        wind.rf.change_component_basis(stray.target, transform=stray)


# Resampling


def test_east_north_components_regrid_within_one_frame_across_the_seam() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    source = declared(solid_body_enu(points_of(grid())), grid(), ENU)
    target = grid(lon=np.array([355.0, 5.0, 190.0]), lat=np.array([-20.0, 30.0]), h=np.array([0.0]))
    with pytest.raises(ValueError, match="cannot declare components in EastNorthUp"):
        xrf.resample(
            source.rf.geometry,
            target,
            output_component_basis=ENU,
            component_reorientation="jacobian",
        )
    result = source.rf.resample_to(
        target, output_component_basis=ENU, component_reorientation="jacobian"
    )
    assert result.rf.tensor_component_declaration.bases == (ENU,)
    expected = solid_body_enu(points_of(target))
    assert_allclose(ordered(result), expected, atol=1e-9, rtol=0)  # east depends on latitude only


def test_east_north_components_resample_onto_earth_centred_points() -> None:
    # Heights span the target box, whose corners reach 350 m above and below its centre; 2°
    # latitude cells keep the interpolation error of cos φ near 1e-4.
    pytest.importorskip("scipy", minversion="1.18")
    deep = grid(lat=np.arange(0.0, 21.0, 2.0), h=np.array([-1000.0, 1000.0]))
    source = declared(solid_body_enu(points_of(deep)), deep, ENU)
    centre = TO_ECEF.transform_point([40.0, 10.0, 250.0])
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("x", "y", "z"), ("m", "m", "m")),
        target=ECEF,
        matrix=np.eye(3),
        translation=(0.0, 0.0, 0.0),
    )
    offsets = np.array([-200.0, 0.0, 200.0])
    target = xrf.Grid(
        transform, {name: (name, centre[k] + offsets) for k, name in enumerate("xyz")}
    )
    result = xrf.resample(
        source.rf.geometry,
        target,
        transform=TO_ECEF.inverse(),
        output_component_basis=ECEF,
        component_reorientation="jacobian",
    )
    xyz = np.stack(np.meshgrid(*(centre[k] + offsets for k in range(3)), indexing="ij"), axis=-1)
    expected = np.cross([0.0, 0.0, OMEGA], xyz)
    values = np.asarray(result.transpose("x", "y", "z", "component").values)
    # Interpolating east in latitude over 2° cells: (2°)² / 8 relative to cos φ, about 2e-4.
    assert_allclose(values, expected, rtol=0, atol=5e-4 * np.abs(expected).max())


def test_earth_centred_components_cross_the_poles() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    lon, lat = np.arange(0.0, 360.0, 10.0), np.arange(-85.0, 86.0, 10.0)
    source_grid = grid(lon=lon, lat=lat, h=np.array([0.0, 1.0]))
    points = points_of(source_grid)
    earth_centred = declared(
        np.cross([0.0, 0.0, OMEGA], TO_ECEF.transform_point(points)), source_grid, ECEF
    )
    cap = grid(lon=np.array([20.0, 200.0]), lat=np.array([88.0, -89.0]), h=np.array([0.0]))
    result = xrf.resample(
        earth_centred.rf.geometry,
        cap,
        output_component_basis=ECEF,
        component_reorientation="jacobian",
    )
    expected = np.cross([0.0, 0.0, OMEGA], TO_ECEF.transform_point(points_of(cap)))
    assert_allclose(ordered(result), expected, atol=0.05 * np.abs(expected).max(), rtol=0)


# East/north across a pole: halo rows from the antipodal meridian, east and north negated.

# A constant Earth-centred vector seen in east/north/up: smooth everywhere, its east and north
# vary like sin λ and cos λ near a pole, and its up component is about 0.5 there, so a missing
# flip on any component (or on an east-up or north-up tensor entry) shows.
CONSTANT = np.array([1.0, 0.3, 0.5])
CELL_CENTRED = np.arange(-85.0, 86.0, 10.0)  # each pole 5° beyond the outer rows
POLE_ROWS = np.arange(-90.0, 91.0, 10.0)
LONGITUDES = np.arange(0.0, 360.0, 10.0)
# Linear interpolation of a function with second derivatives up to |V| per rad² over 10° steps
# in both angles: |V| (Δλ² + Δφ²) / 8.
CAP_TOLERANCE = float(np.linalg.norm(CONSTANT)) * 2 * np.radians(10.0) ** 2 / 8
NEGATE = np.diag([-1.0, -1.0, 1.0])


def constant_enu(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """D(x)ᵀ V at geodetic points: the components of CONSTANT along east, north and up."""
    lam, phi = np.radians(points[..., 0]), np.radians(points[..., 1])
    east = np.stack([-np.sin(lam), np.cos(lam), np.zeros_like(lam)], axis=-1)
    north = np.stack([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)], -1)
    up = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)
    return np.stack([d @ CONSTANT for d in (east, north, up)], axis=-1)


def east_north(lat: npt.NDArray[np.float64]) -> tuple[xr.DataArray, xrf.Grid]:
    source_grid = grid(lon=LONGITUDES, lat=lat, h=np.array([0.0, 1.0]))
    return declared(constant_enu(points_of(source_grid)), source_grid, ENU), source_grid


def resample_enu(source: xr.DataArray, target: xrf.Grid, **options: Any) -> xr.DataArray:
    result: xr.DataArray = source.rf.resample_to(
        target, output_component_basis=ENU, component_reorientation="jacobian", **options
    )
    return result


def test_east_north_components_cross_a_pole_with_east_and_north_reversed() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    source, _ = east_north(CELL_CENTRED)
    # Linear at the pole on meridian 20°: the mean of the outer row and its mirror, which is
    # the row on meridian 200° with east and north negated.
    at_pole = grid(lon=np.array([20.0]), lat=np.array([90.0, -90.0]), h=np.array([0.0]))
    result = ordered(resample_enu(source, at_pole, method="linear"))[0, :, 0]
    for k, row in enumerate((85.0, -85.0)):
        near = constant_enu(np.array([20.0, row, 0.0]))
        far = NEGATE @ constant_enu(np.array([200.0, row, 0.0]))
        assert_allclose(result[k], (near + far) / 2, rtol=1e-12, atol=1e-12)
        # Without the flip east and north would average to about zero; here they do not.
        assert np.abs(result[k][:2]).max() > 0.25
    assert_allclose(result, constant_enu(points_of(at_pole))[0, :, 0], atol=CAP_TOLERANCE, rtol=0)


@pytest.mark.parametrize(
    ("lat", "method"),
    [(CELL_CENTRED, "linear"), (CELL_CENTRED, "cubic"), (POLE_ROWS, "cubic")],
    ids=["cell-centred-linear", "cell-centred-cubic", "pole-rows-cubic"],
)
def test_east_north_components_are_accurate_in_both_polar_caps(
    lat: npt.NDArray[np.float64], method: str
) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    source, _ = east_north(lat)
    cap = grid(
        lon=np.array([3.0, 127.0, 251.0]),
        lat=np.array([-89.5, -86.0, 86.0, 88.0, 89.5]),
        h=np.array([0.0]),
    )
    result = resample_enu(source, cap, method=method)
    assert not np.isnan(result.values).any()
    assert_allclose(ordered(result), constant_enu(points_of(cap)), atol=CAP_TOLERANCE, rtol=0)


def test_a_field_smooth_in_east_north_is_as_accurate_in_the_cap_as_inside() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    source_grid = grid(lon=LONGITUDES, lat=CELL_CENTRED, h=np.array([0.0, 1.0]))
    wind = declared(solid_body_enu(points_of(source_grid)), source_grid, ENU)
    cap = grid(lon=np.array([20.0, 200.0]), lat=np.array([88.0, -89.0]), h=np.array([0.0]))
    expected = solid_body_enu(points_of(cap))
    # East is Ω (N + h) cos φ, smooth through the pole once its sign flips with cos φ.
    result = resample_enu(wind, cap, method="linear")
    scale = OMEGA * A
    assert_allclose(ordered(result), expected, atol=CAP_TOLERANCE * scale, rtol=0)


def outer(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return values[..., :, None] * values[..., None, :]


def test_second_order_east_north_tensors_cross_a_pole_unpacked_and_packed() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    source_grid = grid(lon=LONGITUDES, lat=CELL_CENTRED, h=np.array([0.0, 1.0]))
    tensor = outer(constant_enu(points_of(source_grid)))
    labels = list(ENU.axes)
    plain = xr.DataArray(
        tensor, dims=("lon", "lat", "h", "i", "j"), coords={"i": labels, "j": labels}
    ).rf.frame(source_grid)
    unpacked = plain.rf.declare_tensor_components(
        xrf.SYMMETRIC_SECOND_ORDER_CONTRAVARIANT_TENSOR, component_dims=("i", "j"), bases=ENU
    )
    # Voigt order (east-east, north-north, up-up, north-up, east-up, east-north) with a
    # non-unit off-diagonal scale: the entry order differs from the unpacked one.
    packing = xrf.SymmetricPacking.voigt(3, off_diagonal_scale=2.0)
    packed = xr.DataArray(
        packing.pack(tensor),
        dims=("lon", "lat", "h", "voigt"),
        coords={"voigt": list(packing.labels(ENU.axes))},
    ).rf.frame(source_grid)
    packed = packed.rf.declare_tensor_components(
        xrf.SYMMETRIC_SECOND_ORDER_CONTRAVARIANT_TENSOR,
        component_dims=("voigt",),
        bases=ENU,
        packing=packing,
    )
    cap = grid(lon=np.array([20.0, 200.0]), lat=np.array([88.0, 90.0]), h=np.array([0.0]))
    expected = outer(constant_enu(points_of(cap)))
    # East-up and north-up are about ±0.5 · 0.3 and ±0.5 near the pole: their sign is visible.
    bound = 3 * float(np.linalg.norm(CONSTANT)) * CAP_TOLERANCE
    result = resample_enu(unpacked, cap, method="linear").transpose("lon", "lat", "h", "i", "j")
    assert_allclose(result.values, expected, atol=bound, rtol=0)
    result = resample_enu(packed, cap, method="linear").transpose("lon", "lat", "h", "voigt")
    assert_allclose(packing.unpack(result.values), expected, atol=bound, rtol=0)


def test_a_pseudovector_crosses_a_pole_with_no_determinant_factor() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    source_grid = grid(lon=LONGITUDES, lat=CELL_CENTRED, h=np.array([0.0, 1.0]))
    plain = xr.DataArray(
        constant_enu(points_of(source_grid)),
        dims=("lon", "lat", "h", "component"),
        coords={"component": list(ENU.axes)},
    ).rf.frame(source_grid)
    axial = plain.rf.declare_tensor_components(
        xrf.CROSS_PRODUCT_PSEUDOVECTOR, component_dims=("component",), bases=ENU
    )
    cap = grid(lon=np.array([20.0]), lat=np.array([88.0, -89.0]), h=np.array([0.0]))
    result = resample_enu(axial, cap, method="linear")
    assert_allclose(ordered(result), constant_enu(points_of(cap)), atol=CAP_TOLERANCE, rtol=0)


def test_a_two_dimensional_east_north_field_crosses_a_pole() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    surface = geodetic(("lon", "lat"), ("degrees_east", "degrees_north"), ("longitude", "latitude"))
    field = xrf.EastNorthUp(surface, **FIGURE, length_unit="m")
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("lon", "lat"), surface.units),
        target=surface,
        matrix=np.eye(2),
        translation=(0.0, 0.0),
    )

    def surface_grid(lon: npt.NDArray[np.float64], lat: npt.NDArray[np.float64]) -> xrf.Grid:
        return xrf.Grid(transform, {"lon": ("lon", lon), "lat": ("lat", lat)})

    def components(target: xrf.Grid) -> npt.NDArray[np.float64]:
        lon, lat = (np.asarray(target.coordinates[n][1], float) for n in ("lon", "lat"))  # type: ignore[index]
        mesh = np.stack(np.meshgrid(lon, lat, np.zeros(1), indexing="ij"), axis=-1)[:, :, 0]
        return constant_enu(mesh)[..., :2]

    source_grid = surface_grid(LONGITUDES, CELL_CENTRED)
    wind = (
        xr.DataArray(
            components(source_grid),
            dims=("lon", "lat", "component"),
            coords={"component": list(field.axes)},
        )
        .rf.frame(source_grid)
        .rf.declare_tensor_components(xrf.VECTOR, component_dims=("component",), bases=field)
    )
    at_pole = surface_grid(np.array([20.0]), np.array([90.0]))
    result = wind.rf.resample_to(
        at_pole, output_component_basis=field, component_reorientation="jacobian", method="linear"
    )
    near = components(surface_grid(np.array([20.0]), np.array([85.0])))[0, 0]
    far = -components(surface_grid(np.array([200.0]), np.array([85.0])))[0, 0]
    values = np.asarray(result.transpose("lon", "lat", "component").values)[0, 0]
    assert_allclose(values, (near + far) / 2, rtol=1e-12, atol=1e-12)


def test_east_north_pole_halos_follow_dask_chunks() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    source, _ = east_north(CELL_CENTRED)
    timed = xr.concat([source, 2.0 * source], dim="time")
    cap = grid(lon=np.array([20.0, 200.0]), lat=np.array([88.0, -89.0]), h=np.array([0.0]))
    eager = resample_enu(timed, cap, method="linear")
    lazy = resample_enu(timed.chunk({"component": 1, "time": 1}), cap, method="linear")
    assert lazy.chunks is not None
    assert_allclose(lazy.compute().values, eager.values, rtol=0, atol=0)
    assert_allclose(
        ordered(eager.isel(time=1)), 2.0 * constant_enu(points_of(cap)), atol=2 * CAP_TOLERANCE
    )
    empty = grid(lon=np.array([]), lat=np.array([88.0]), h=np.array([0.0]))
    nothing = resample_enu(source.chunk({"component": 1}), empty, method="linear")
    assert nothing.compute().size == 0


def test_bases_without_a_pole_rule_leave_the_cap_outside() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    source_grid = grid(lon=LONGITUDES, lat=CELL_CENTRED, h=np.array([0.0, 1.0]))
    # The chart basis: its longitude component diverges like 1 / cos φ at a pole.
    chart = declared(np.ones((*points_of(source_grid).shape[:-1], 3)), source_grid, EARTH)
    cap = grid(lon=np.array([20.0]), lat=np.array([88.0]), h=np.array([0.0]))
    outside = chart.rf.resample_to(
        cap, output_component_basis=EARTH, component_reorientation="jacobian", method="linear"
    )
    assert np.isnan(outside.values).all()


def test_components_that_cannot_be_negated_are_refused() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    source_grid = grid(lon=LONGITUDES, lat=CELL_CENTRED, h=np.array([0.0, 1.0]))
    shape = (*points_of(source_grid).shape[:-1], 3)
    cap = grid(lon=np.array([20.0]), lat=np.array([88.0]), h=np.array([0.0]))
    unsigned = declared(np.ones(shape, dtype=np.uint16), source_grid, ENU)
    with pytest.raises(ValueError, match="uint16 cannot be negated"):
        resample_enu(unsigned, cap, method="nearest", fill_value=0)
    lowest = np.ones(shape, dtype=np.int16)
    lowest[:, -1, :, 0] = np.iinfo(np.int16).min
    signed = declared(lowest, source_grid, ENU)
    with pytest.raises(ValueError, match=r"int16 cannot represent -\(-32768\)"):
        resample_enu(signed, cap, method="nearest", fill_value=0)
    # A signed integer away from its minimum negates exactly.
    small = declared(np.ones(shape, dtype=np.int16), source_grid, ENU)
    assert resample_enu(small, cap, method="nearest", fill_value=0).dtype.kind in "if"


class Rotation:
    """A user frame field: the axes of a Cartesian plane turned by a fixed angle."""

    def __init__(self, base: xrf.ReferenceFrame, angle: float) -> None:
        self.base, self.angle = base, angle
        self.axes = ("along", "across")
        self.units = base.units

    def _rotation(self, at: npt.ArrayLike) -> npt.NDArray[np.float64]:
        shape = np.asarray(at, dtype=float).shape[:-1]
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.broadcast_to(np.array([[c, -s], [s, c]]), (*shape, 2, 2)).copy()

    def matrix(self, at: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return self._rotation(at)

    def inverse_matrix(self, at: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.swapaxes(self._rotation(at), -1, -2)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Rotation) and (self.base, self.angle) == (other.base, other.angle)

    def __hash__(self) -> int:
        return hash((Rotation, self.base, self.angle))


def test_a_user_frame_field_works_through_both_operations() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    plane = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("m", "m")))
    field = Rotation(plane, math.radians(30.0))
    assert isinstance(field, xrf.FrameField)
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("x", "y"), ("m", "m")),
        target=plane,
        matrix=np.eye(2),
        translation=(0.0, 0.0),
    )
    source_grid = xrf.Grid(transform, {"x": ("x", np.arange(4.0)), "y": ("y", np.arange(3.0))})
    plain = xr.DataArray(
        np.broadcast_to([1.0, 0.0], (4, 3, 2)).copy(),
        dims=("x", "y", "component"),
        coords={"component": ["along", "across"]},
    )
    framed = cast(xr.DataArray, plain.rf.frame(source_grid))
    along = framed.rf.declare_tensor_components(
        xrf.VECTOR, component_dims=("component",), bases=field
    )
    in_plane = along.rf.change_component_basis(plane)
    assert_allclose(
        in_plane.isel(x=0, y=0).values, [math.cos(math.radians(30)), math.sin(math.radians(30))]
    )
    target = xrf.Grid(transform, {"x": ("x", [0.5, 2.5]), "y": ("y", [1.5])})
    result = xrf.resample(
        along.rf.geometry, target, output_component_basis=plane, component_reorientation="jacobian"
    )
    assert_allclose(
        result.transpose("x", "y", "component").values[0, 0], in_plane.isel(x=0, y=0).values
    )
    with pytest.raises(TypeError, match="Rotation is a user frame field"):
        xrf.encode(along.rf.tensor_component_declaration)


def test_east_north_up_declarations_persist() -> None:
    wind = declared(solid_body_enu(points_of(grid())), grid(), ENU)
    declaration = wind.rf.tensor_component_declaration
    decoded = xrf.decode(json.loads(json.dumps(xrf.encode(declaration))))
    assert decoded == declaration
    surface = geodetic(("lon", "lat"), ("degrees_east", "degrees_north"), ("longitude", "latitude"))
    for field in (xrf.EastNorthUp(surface, **FIGURE, length_unit="m"), ENU):
        assert xrf.decode(xrf.encode(field)) == field
    restored = wind.rf.encode().rf.decode()
    assert restored.rf.tensor_component_declaration == declaration


def test_a_callable_reorientation_is_applied_even_within_one_frame() -> None:
    """The identity shortcut holds only for the named reorientations."""
    pytest.importorskip("scipy", minversion="1.18")
    source = declared(solid_body_enu(points_of(grid())), grid(), ENU)
    inner = grid(lat=np.array([0.0, 30.0]), h=np.array([0.0]))
    doubled = source.rf.resample_to(
        inner,
        output_component_basis=ENU,
        component_reorientation=lambda push: 2.0 * np.asarray(push),
    )
    assert_allclose(ordered(doubled), 2.0 * solid_body_enu(points_of(inner)), atol=1e-9, rtol=0)
    # A multiple of the identity commutes with the basis maps, so pole rows stay defined.
    poles = grid(lat=np.array([-90.0, 90.0]), h=np.array([0.0]))
    at_poles = source.rf.resample_to(
        poles,
        output_component_basis=ENU,
        component_reorientation=lambda push: 2.0 * np.asarray(push),
    )
    assert_allclose(ordered(at_poles), 2.0 * solid_body_enu(points_of(poles)), atol=1e-9, rtol=0)
    # A general callable result is applied in the chart basis, which does not exist at a pole.
    with pytest.raises(ValueError, match="undefined at target sample"):
        source.rf.resample_to(
            poles,
            output_component_basis=ENU,
            component_reorientation=lambda push: np.asarray(push) @ np.diag([1.0, 2.0, 1.0]),
        )


class Collapse(Rotation):
    """A user field whose basis is singular: both vectors along x."""

    def _rotation(self, at: npt.ArrayLike) -> npt.NDArray[np.float64]:
        shape = np.asarray(at, dtype=float).shape[:-1]
        return np.broadcast_to(np.array([[1.0, 1.0], [0.0, 0.0]]), (*shape, 2, 2)).copy()


def test_a_singular_user_field_is_refused_by_both_operations() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    plane = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("m", "m")))
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("x", "y"), ("m", "m")),
        target=plane,
        matrix=np.eye(2),
        translation=(0.0, 0.0),
    )
    source_grid = xrf.Grid(transform, {"x": ("x", np.arange(3.0)), "y": ("y", np.arange(2.0))})
    plain = xr.DataArray(
        np.ones((3, 2, 2)), dims=("x", "y", "component"), coords={"component": ["along", "across"]}
    )
    framed = cast(xr.DataArray, plain.rf.frame(source_grid))
    collapsed = framed.rf.declare_tensor_components(
        xrf.VECTOR, component_dims=("component",), bases=Collapse(plane, 0.0)
    )
    with pytest.raises(ValueError, match=r"at sample \{'x': 0, 'y': 0\}.*singular"):
        collapsed.rf.change_component_basis(plane)
    with pytest.raises(ValueError, match=r"undefined at target sample \{'x': 0, 'y': 0\}"):
        collapsed.rf.resample_to(
            xrf.Grid(transform, {"x": ("x", [0.5]), "y": ("y", [0.5])}),
            output_component_basis=plane,
            component_reorientation="jacobian",
        )


def test_a_zero_callable_reorientation_is_refused() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    source = declared(solid_body_enu(points_of(grid())), grid(), ENU)
    with pytest.raises(
        ValueError, match=r"0\.0 times the identity; a component map must be invertible"
    ):
        source.rf.resample_to(
            grid(lat=np.array([0.0]), h=np.array([0.0])),
            output_component_basis=ENU,
            component_reorientation=lambda push: 0.0 * np.asarray(push),
        )


def test_pointwise_changes_keep_dask_chunks_along_geometry_dims() -> None:
    wind = declared(solid_body_enu(points_of(grid())), grid(), ENU)
    lazy = wind.chunk({"lon": 4, "lat": 3})
    converted = lazy.rf.change_component_basis(ECEF, transform=TO_ECEF)
    assert converted.chunks is not None and len(converted.chunksizes["lon"]) == 3
    eager = wind.rf.change_component_basis(ECEF, transform=TO_ECEF)
    assert_allclose(ordered(converted), ordered(eager), atol=0, rtol=0)


def test_undeclared_scalars_change_only_between_constant_bases() -> None:
    wind = declared(solid_body_enu(points_of(grid())), grid(), ENU)
    scalar = wind.isel(component=0, drop=True)
    with pytest.raises(ValueError, match="frame-field basis varies by position"):
        scalar.rf.change_component_basis(
            EARTH,
            tensor_type=xrf.TensorType((), determinant_exponent=-1.0),
            from_basis=ENU,
            units="x",
        )


def test_a_field_basis_goes_stale_when_the_frame_changes() -> None:
    wind = declared(solid_body_enu(points_of(grid())), grid(), ENU)
    moved = wind.rf.assume_frame(xrf.ReferenceFrame.local(EARTH.coordinate_system))
    with pytest.raises(ValueError, match="frame field that no longer belongs"):
        moved.rf.encode()
