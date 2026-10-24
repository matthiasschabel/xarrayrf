"""Transitions between an angular chart and a Cartesian chart of one reference frame.

A lon/lat/height chart and an Earth-centred Cartesian chart express the same frame, so the
conversion between them keeps identity: it is a chart transition, not a transform between
frames, and both endpoints must be equivalent frames. Nothing is registered or discovered; a
caller constructs the transition it needs and passes it where a transform is accepted.

Conventions for the Cartesian chart, read in its declared axis order ``(x, y, z)``: x points to
longitude 0 on the equator, z to the north pole, and the axes are right-handed. Longitude is
measured eastward from x. Spherical latitude is geocentric and the radius is the distance from
the centre; geodetic latitude is the angle of the ellipsoid normal and the height is measured
along it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import ClassVar, Final

import numpy as np
import numpy.typing as npt

from ._coordinate_system import CARTESIAN, GEODETIC, SPHERICAL
from ._frame import ReferenceFrame
from ._validation import angular_period, real_float_array

POLE_ROUNDOFF: Final = 1e-12
"""Relative band beyond a pole that counts as on it: 9e-11° (about 6000 ulp of 90°, 10 µm on Earth).

Unit conversion and affine round trips put an exact pole some ulp past it; there is no sample
step here to scale the band by, as resampling does.
"""

FIGURE_KEYS: Final = ("figure", "semi_major_axis", "inverse_flattening", "length_unit")
"""Reserved frame-definition keys that declare an Earth figure, checked against a transition."""


def _check_real(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"{field} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {number!r}")
    return number


def latitude_trig(
    latitude: npt.NDArray[np.float64], scale: float, pole: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """cos φ and sin φ of latitudes in a declared unit, exact at the poles.

    A latitude beyond a pole by at most ``POLE_ROUNDOFF`` of the pole is roundoff and lies on it.

    Raises:
        ValueError: If a latitude lies further beyond a pole, naming the first such point.
    """
    past = np.abs(latitude) - pole
    outside = past > POLE_ROUNDOFF * pole
    if outside.any():
        index = tuple(int(i) for i in np.argwhere(outside)[0])
        raise ValueError(
            f"latitude outside ±{pole:g} at point {index}; it is bounded by the poles",
        )
    # Exact poles land exactly on the axis rather than cos(π/2) ≈ 6e-17 beside it.
    at_pole = past >= 0.0
    radians = latitude * scale
    return (
        np.where(at_pole, 0.0, np.cos(radians)),
        np.where(at_pole, np.sign(latitude), np.sin(radians)),
    )


def local_frame(
    longitude: npt.NDArray[np.float64],
    cos_lat: npt.NDArray[np.float64],
    sin_lat: npt.NDArray[np.float64],
    height: npt.NDArray[np.float64] | float,
    a: float,
    b2: float,
) -> tuple[npt.NDArray[np.float64], ...]:
    """East, north and up unit vectors in Earth-centred axes, and the east and north scales.

    ``longitude`` is in radians; ``a`` is the semi-major axis and ``b2`` is ``1 - e²`` (both 0
    and 1 for the spherical chart, whose radius is then ``height``). The scales are the lengths
    of the longitude and latitude coordinate vectors per radian: ``(N + h) cos φ`` and ``M + h``.
    The directions are defined at the poles too (the limit along the meridian).
    """
    denominator = cos_lat**2 + b2 * sin_lat**2  # 1 - e² sin²φ without cancellation
    n = a / np.sqrt(denominator)
    m = a * b2 / denominator**1.5
    cos_lon, sin_lon = np.cos(longitude), np.sin(longitude)
    zero = np.zeros_like(cos_lon)
    east = np.stack((-sin_lon, cos_lon, zero), axis=-1)
    north = np.stack((-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat), axis=-1)
    up = np.stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat), axis=-1)
    return east, north, up, (n + height) * cos_lat, m + height


def check_figure_definition(
    definition: Mapping[str, object],
    semi_major_axis: float,
    inverse_flattening: float,
    length_unit: str | None,
    *,
    what: str,
) -> None:
    """Refuse a figure that disagrees with the reserved keys of a frame definition."""
    if "figure" not in definition:
        return
    sphere = inverse_flattening == 0.0
    expected: dict[str, object] = {
        "figure": "sphere" if sphere else "ellipsoid",
        "semi_major_axis": semi_major_axis,
        "length_unit": length_unit,
    }
    if not sphere:
        expected["inverse_flattening"] = inverse_flattening
    for key in FIGURE_KEYS:
        declared = definition.get(key)
        if key not in expected:
            if key in definition:
                raise ValueError(
                    f"the frame definition declares {key}={declared!r}, but a sphere has none; "
                    f"the {what}'s figure disagrees with the frame",
                )
            continue
        if key not in definition or declared != expected[key]:
            raise ValueError(
                f"the frame definition declares {key}={declared!r}, but the {what}'s figure has "
                f"{expected[key]!r}; a {what} must use the frame's figure",
            )


class _AngularChart:
    """Where an angular chart keeps its three spatial axes, and its angle scales."""

    __slots__ = (
        "height",
        "latitude",
        "latitude_scale",
        "length_unit",
        "longitude",
        "period",
        "pole",
    )

    def __init__(self, frame: ReferenceFrame, representation: str, radial: str, field: str):
        system = frame.coordinate_system
        if system.representation != representation or len(system.axes) != 3:
            raise ValueError(
                f"{field} must be a {representation} chart with exactly three axes "
                f"(longitude, latitude and {radial}), got {system!r}",
            )
        roles = system.axis_roles
        if radial not in roles:
            raise ValueError(f"{field} chart needs a {radial} axis, got roles {roles}")
        self.longitude = roles.index("longitude")
        self.latitude = roles.index("latitude")
        self.height = roles.index(radial)
        units = system.units
        longitude_unit, latitude_unit = units[self.longitude], units[self.latitude]
        assert longitude_unit is not None and latitude_unit is not None  # stage 1 requires them
        self.period = angular_period(longitude_unit)
        # Radians per declared unit; a quarter turn is exact in both units (90, or π/2 as
        # math.pi / 2), so comparing against it detects exact poles.
        self.latitude_scale = 2.0 * math.pi / angular_period(latitude_unit)
        self.pole = angular_period(latitude_unit) / 4.0
        self.length_unit = units[self.height]

    @property
    def longitude_scale(self) -> float:
        return 2.0 * math.pi / self.period


def _cartesian_unit(frame: ReferenceFrame, length_unit: str | None, field: str) -> None:
    system = frame.coordinate_system
    if system.representation != CARTESIAN or len(system.axes) != 3:
        raise ValueError(
            f"{field} must be a Cartesian chart with exactly three axes (x, y, z), got {system!r}"
        )
    if length_unit is None or any(unit != length_unit for unit in system.units):
        raise ValueError(
            f"{field} axes must all be in the radial unit {length_unit!r}, got {system.units}; "
            "units are never rescaled here",
        )


class _ChartTransition:
    """Shared validation, evaluation and comparison; subclasses fix direction and chart kind."""

    __slots__ = (
        "_cartesian",
        "_chart",
        "_inverse_flattening",
        "_longitude_start",
        "_semi_major_axis",
        "_source",
        "_target",
    )

    _REPRESENTATION: ClassVar[str]
    _RADIAL: ClassVar[str]
    _TO_CARTESIAN: ClassVar[bool]

    _source: ReferenceFrame
    _target: ReferenceFrame
    _chart: _AngularChart
    _semi_major_axis: float
    _inverse_flattening: float
    _longitude_start: float

    def _setup(
        self,
        source: object,
        target: object,
        semi_major_axis: float,
        inverse_flattening: float,
        longitude_start: object,
    ) -> None:
        for name, frame in (("source", source), ("target", target)):
            if not isinstance(frame, ReferenceFrame):
                raise TypeError(f"{name} must be a ReferenceFrame, got {type(frame).__name__}")
        assert isinstance(source, ReferenceFrame) and isinstance(target, ReferenceFrame)
        if not source.is_equivalent_frame(target):
            raise ValueError(
                f"{source.identifier} and {target.identifier} are not the same frame; a chart "
                "transition re-expresses one frame, so relate different frames with a transform",
            )
        angular, cartesian = (source, target) if self._TO_CARTESIAN else (target, source)
        angular_field, cartesian_field = (
            ("source", "target") if self._TO_CARTESIAN else ("target", "source")
        )
        self._chart = _AngularChart(angular, self._REPRESENTATION, self._RADIAL, angular_field)
        _cartesian_unit(cartesian, self._chart.length_unit, cartesian_field)
        self._source, self._target = source, target
        self._cartesian = cartesian
        self._semi_major_axis = semi_major_axis
        self._inverse_flattening = inverse_flattening
        self._longitude_start = (
            -0.5 * self._chart.period
            if longitude_start is None
            else _check_real(longitude_start, field="longitude_start")
        )

    @property
    def source(self) -> ReferenceFrame:
        """The frame points are mapped from."""
        return self._source

    @property
    def target(self) -> ReferenceFrame:
        """The frame points are mapped to."""
        return self._target

    @property
    def longitude_start(self) -> float:
        """Start of the longitude interval ``[start, start + period)`` that inverses produce."""
        return self._longitude_start

    # The figure: a sphere of radius a when e² = 0; the spherical chart is the degenerate
    # a = 0, e² = 0, for which the geodetic formulas reduce to radius = height.
    # 1 - e² is carried as (1 - f)², not computed by subtraction: for a flattening near 1, e²
    # rounds to 1 and 1 - e² would cancel to 0.
    def _figure(self) -> tuple[float, float, float]:
        f = 0.0 if self._inverse_flattening == 0.0 else 1.0 / self._inverse_flattening
        return self._semi_major_axis, f * (2.0 - f), (1.0 - f) ** 2

    def _evolute_radius(self) -> float:
        """Distance from the centre within which the inverse is not unique (e² a² / b)."""
        a, e2, b2 = self._figure()
        return 0.0 if e2 == 0.0 else e2 * a / math.sqrt(b2)

    def _points(self, points: npt.ArrayLike, *, field: str) -> npt.NDArray[np.float64]:
        values = real_float_array(points, field=field)
        if values.ndim == 0 or values.shape[-1] != 3:
            raise ValueError(
                f"{field} must have a trailing axis of length 3 ordered as {self._source.axes}, "
                f"got shape {values.shape}",
            )
        return values

    @staticmethod
    def _first(mask: npt.NDArray[np.bool_]) -> tuple[int, ...]:
        return tuple(int(i) for i in np.argwhere(mask)[0])

    def _angles(self, values: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], ...]:
        """Validate angular-chart points; return λ, cos φ, sin φ (radians) and the height."""
        chart = self._chart
        longitude = values[..., chart.longitude] * chart.longitude_scale
        cos_lat, sin_lat = latitude_trig(
            values[..., chart.latitude], chart.latitude_scale, chart.pole
        )
        return longitude, cos_lat, sin_lat, values[..., chart.height]

    def _forward(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a, _, b2 = self._figure()
        longitude, cos_lat, sin_lat, height = self._angles(values)
        # 1 - e² sin²φ, written without cancellation (see _figure).
        n = a / np.sqrt(cos_lat**2 + b2 * sin_lat**2)
        # The forward domain is the inverse's image: the point must stay on its surface foot's
        # side of the equator and the axis, and outside the evolute, or another foot is nearer.
        shallow = height > -n * b2
        if not shallow.all():
            raise ValueError(
                f"{self._RADIAL} at point {self._first(~shallow)} reaches the far side of the "
                "centre, where the point has a nearer surface foot and no unique inverse",
            )
        across = (n + height) * cos_lat
        xyz: npt.NDArray[np.float64] = np.stack(
            (
                across * np.cos(longitude),
                across * np.sin(longitude),
                (n * b2 + height) * sin_lat,
            ),
            axis=-1,
        )
        inside = np.linalg.norm(xyz, axis=-1) <= self._evolute_radius()
        if inside.any():
            raise ValueError(
                f"point {self._first(inside)} maps within {self._evolute_radius():g} "
                f"{self._chart.length_unit} of the centre, where the inverse is not unique",
            )
        return xyz

    def _backward(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a, e2, b2 = self._figure()
        x, y, z = values[..., 0], values[..., 1], values[..., 2]
        radius = np.sqrt(x * x + y * y + z * z)
        bound = self._evolute_radius()
        inside = radius <= bound
        if inside.any():
            where = "the centre" if bound == 0.0 else f"{bound:g} {self._chart.length_unit} of it"
            raise ValueError(
                f"point {self._first(inside)} is at or within {where}, where latitude has no "
                "unique value",
            )
        rho = np.hypot(x, y)
        if e2 == 0.0:
            latitude = np.arctan2(z, rho)
            height = radius - a
        else:
            # Vermeille, "Direct transformation from geocentric coordinates to geodetic
            # coordinates", J. Geodesy 76 (2002): closed form, exact outside the evolute.
            p = rho**2 / a**2
            q = b2 * z**2 / a**2
            r = (p + q - e2**2) / 6.0
            s = e2**2 * p * q / (4.0 * r**3)
            t = np.cbrt(1.0 + s + np.sqrt(s * (2.0 + s)))
            u = r * (1.0 + t + 1.0 / t)
            v = np.sqrt(u**2 + e2**2 * q)
            w = e2 * (u + v - q) / (2.0 * v)
            k = np.sqrt(u + v + w**2) - w
            d = k * rho / (k + e2)
            latitude = 2.0 * np.arctan2(z, d + np.hypot(d, z))
            height = (k - b2) / k * np.hypot(d, z)
        chart = self._chart
        start, period = self._longitude_start, chart.period
        longitude = np.arctan2(y, x) / chart.longitude_scale
        longitude = start + np.mod(longitude - start, period)
        # Rounding can land exactly on the excluded end; the polar axis has no longitude.
        longitude = np.where((longitude >= start + period) | (rho == 0.0), start, longitude)
        result = np.empty_like(values)
        result[..., chart.longitude] = longitude
        result[..., chart.latitude] = latitude / chart.latitude_scale
        result[..., chart.height] = height
        return result

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map points whose trailing axis is ordered as the source axes.

        Args:
            points: Finite real values shaped ``(..., 3)``.

        Returns:
            A new array shaped ``(..., 3)``, ordered as the target axes.

        Raises:
            TypeError: If the input is not real or is a masked array.
            ValueError: If the input has the wrong shape or is not finite, or a point is
                outside the domain: a latitude beyond the poles, a radial coordinate on the
                far side of the centre, or a point at or within the centre's
                non-uniqueness radius (the evolute bound ``e² a² / b``; the centre itself for a
                sphere). The error names the first offending point index.
        """
        values = self._points(points, field="points")
        # Overflow is reported below as an error naming the problem, not as a warning.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mapped = self._forward(values) if self._TO_CARTESIAN else self._backward(values)
        if not np.isfinite(mapped).all():
            raise ValueError("chart transition produced non-finite coordinates")
        return mapped

    def _angular_jacobian(
        self, angular: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Orthonormal directions (..., 3, 3) of λ, φ, h and their scale factors (..., 3).

        Columns of the forward Jacobian are ``directions[..., :, j] * scales[..., j]`` in chart
        order, since the λ, φ and h directions are orthogonal.
        """
        a, _, b2 = self._figure()
        chart = self._chart
        longitude, cos_lat, sin_lat, height = self._angles(angular)
        east, north, up, east_scale, north_scale = local_frame(
            longitude, cos_lat, sin_lat, height, a, b2
        )
        scales = np.empty((*longitude.shape, 3))
        directions = np.empty((*longitude.shape, 3, 3))
        for column, vector, scale in (
            (chart.longitude, east, east_scale * chart.longitude_scale),
            (chart.latitude, north, north_scale * chart.latitude_scale),
            (chart.height, up, np.ones_like(east_scale)),
        ):
            directions[..., :, column] = vector
            scales[..., column] = scale
        return directions, scales

    def jacobian(self, at: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        """Return the Jacobian at points, target axes by source axes, shaped ``(..., 3, 3)``.

        Angles are differentiated in their declared units. The angular-to-Cartesian Jacobian
        is finite everywhere in the domain and singular at the poles, where it is returned as
        is. Its inverse does not exist on the polar axis, so the Cartesian-to-angular Jacobian
        raises there.

        Args:
            at: Points shaped ``(..., 3)``, ordered as the source axes. Required.

        Raises:
            TypeError: If ``at`` is missing, not real, or a masked array.
            ValueError: If ``at`` has the wrong shape, is not finite or is outside the domain,
                or, for a Cartesian source, lies on the polar axis, or the Jacobian overflows
                float64 (at a point within about 1e-306 of the polar axis, say). The error names the
                first offending point index.
        """
        if at is None:
            raise TypeError("a chart transition's Jacobian varies by point; pass at=")
        values = self._points(at, field="at")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            jacobian = self._jacobian(values)
        finite = np.isfinite(jacobian).all(axis=(-2, -1))
        if not finite.all():
            raise ValueError(
                f"the Jacobian at point {self._first(~finite)} is not finite in float64, "
                "so it cannot be represented there",
            )
        return jacobian

    def _jacobian(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._TO_CARTESIAN:
            self.transform_point(values)  # domain and finiteness checks only
            directions, scales = self._angular_jacobian(values)
            return directions * scales[..., None, :]
        on_axis = (values[..., 0] == 0.0) & (values[..., 1] == 0.0)
        if on_axis.any():
            raise ValueError(
                f"point {self._first(on_axis)} is on the polar axis, where longitude has no "
                "derivative, so the Jacobian does not exist there",
            )
        directions, scales = self._angular_jacobian(self.transform_point(values))
        # (N + h) cos φ is the distance from the axis. Near the axis the computed latitude can
        # round to exactly ±90°, so take that distance from the Cartesian point instead.
        rho = np.hypot(values[..., 0], values[..., 1])
        scales[..., self._chart.longitude] = rho * self._chart.longitude_scale
        # Rows of the inverse are the directions over their scales: J = D S with D orthonormal.
        return np.swapaxes(directions, -1, -2) / scales[..., :, None]

    def _parameters(self) -> tuple[float, ...]:
        return (self._semi_major_axis, self._inverse_flattening, self._longitude_start)

    def __eq__(self, other: object) -> bool:
        """Compare class, endpoints and parameters exactly."""
        if not isinstance(other, _ChartTransition):
            return NotImplemented
        return type(self) is type(other) and (
            self._source,
            self._target,
            self._parameters(),
        ) == (other._source, other._target, other._parameters())

    def __hash__(self) -> int:
        """Hash consistently with :meth:`__eq__`."""
        return hash((type(self), self._source, self._target, self._parameters()))

    def __reduce__(self) -> tuple[object, ...]:
        return (_rebuild, (type(self), self._source, self._target, self._parameters()))


def _rebuild(
    cls: type[_ChartTransition],
    source: ReferenceFrame,
    target: ReferenceFrame,
    parameters: tuple[float, float, float],
) -> _ChartTransition:
    semi_major_axis, inverse_flattening, longitude_start = parameters
    if issubclass(cls, _Spherical):
        return cls(source, target, longitude_start=longitude_start)
    return cls(  # type: ignore[call-arg]
        source,
        target,
        semi_major_axis=semi_major_axis,
        inverse_flattening=inverse_flattening,
        longitude_start=longitude_start,
    )


class _Spherical(_ChartTransition):
    __slots__ = ()
    _REPRESENTATION = SPHERICAL
    _RADIAL = "radius"

    def __init__(
        self,
        source: ReferenceFrame,
        target: ReferenceFrame,
        *,
        longitude_start: float | None = None,
    ) -> None:
        self._setup(source, target, 0.0, 0.0, longitude_start)

    def __repr__(self) -> str:
        """Return a representation naming the endpoints and the longitude start."""
        return (
            f"{type(self).__name__}(source={self._source!r}, target={self._target!r}, "
            f"longitude_start={self._longitude_start!r})"
        )


class _Geodetic(_ChartTransition):
    __slots__ = ()
    _REPRESENTATION = GEODETIC
    _RADIAL = "height"

    def __init__(
        self,
        source: ReferenceFrame,
        target: ReferenceFrame,
        *,
        semi_major_axis: float,
        inverse_flattening: float,
        longitude_start: float | None = None,
    ) -> None:
        a = _check_real(semi_major_axis, field="semi_major_axis")
        if a <= 0.0:
            raise ValueError(f"semi_major_axis must be positive, got {a!r}")
        inverse_f = _check_real(inverse_flattening, field="inverse_flattening")
        if inverse_f != 0.0 and inverse_f <= 1.0:
            raise ValueError(
                f"inverse_flattening must be 0 (a sphere) or greater than 1, got {inverse_f!r}"
            )
        self._setup(source, target, a, inverse_f, longitude_start)
        self._check_definition()

    def _check_definition(self) -> None:
        check_figure_definition(
            self._source.definition,
            self._semi_major_axis,
            self._inverse_flattening,
            self._chart.length_unit,
            what="transition",
        )

    @property
    def semi_major_axis(self) -> float:
        """Equatorial radius, in the radial unit."""
        return self._semi_major_axis

    @property
    def inverse_flattening(self) -> float:
        """``a / (a - b)``; ``0.0`` for a sphere."""
        return self._inverse_flattening

    def __repr__(self) -> str:
        """Return a representation naming the endpoints and the figure."""
        return (
            f"{type(self).__name__}(source={self._source!r}, target={self._target!r}, "
            f"semi_major_axis={self._semi_major_axis!r}, "
            f"inverse_flattening={self._inverse_flattening!r}, "
            f"longitude_start={self._longitude_start!r})"
        )


class SphericalToCartesian(_Spherical):
    """Spherical (longitude, latitude, radius) to Cartesian (x, y, z) within one frame.

    Latitude is geocentric and the radius is the distance from the centre, as for the
    celestial sphere with a distance or a spherical-polar physics chart. The forward domain
    excludes the centre, which has no latitude. ``longitude_start`` only affects
    :meth:`inverse`, so that ``t.inverse().inverse() == t``.
    """

    __slots__ = ()
    _TO_CARTESIAN = True

    def inverse(self) -> CartesianToSpherical:
        """Return the Cartesian-to-spherical transition with the endpoints exchanged."""
        return CartesianToSpherical(
            self._target, self._source, longitude_start=self._longitude_start
        )


class CartesianToSpherical(_Spherical):
    """Cartesian (x, y, z) to spherical (longitude, latitude, radius) within one frame.

    Longitudes lie in ``[longitude_start, longitude_start + period)``; the default start is
    -180° (-π). On the polar axis every longitude names the same point, and the result is
    ``longitude_start``. The centre raises.
    """

    __slots__ = ()
    _TO_CARTESIAN = False

    def inverse(self) -> SphericalToCartesian:
        """Return the spherical-to-Cartesian transition with the endpoints exchanged."""
        return SphericalToCartesian(
            self._target, self._source, longitude_start=self._longitude_start
        )


class GeodeticToCartesian(_Geodetic):
    """Geodetic (longitude, latitude, height) to Earth-centred Cartesian within one frame.

    The figure is an ellipsoid with ``semi_major_axis`` in the height unit and
    ``inverse_flattening`` (``0.0`` for a sphere, as in pyproj). When the frame definition
    declares a figure (``figure``, ``semi_major_axis``, ``inverse_flattening``,
    ``length_unit``), it must agree. Heights may be negative down to the point where the
    normal reaches the equatorial plane or the polar axis; the image must also lie outside
    the evolute bound ``e² a² / b`` (about 42.8 km for WGS 84), so every accepted point maps
    back through :meth:`inverse` to itself, with longitude modulo the period and
    ``longitude_start`` on the polar axis.
    """

    __slots__ = ()
    _TO_CARTESIAN = True

    def inverse(self) -> CartesianToGeodetic:
        """Return the Cartesian-to-geodetic transition with the endpoints exchanged."""
        return CartesianToGeodetic(
            self._target,
            self._source,
            semi_major_axis=self._semi_major_axis,
            inverse_flattening=self._inverse_flattening,
            longitude_start=self._longitude_start,
        )


class CartesianToGeodetic(_Geodetic):
    """Earth-centred Cartesian to geodetic (longitude, latitude, height) within one frame.

    Uses Vermeille's closed form, exact outside the evolute bound ``e² a² / b``; points at or
    within it raise. Longitudes lie in ``[longitude_start, longitude_start + period)``, and on
    the polar axis the result is ``longitude_start``.
    """

    __slots__ = ()
    _TO_CARTESIAN = False

    def inverse(self) -> GeodeticToCartesian:
        """Return the geodetic-to-Cartesian transition with the endpoints exchanged."""
        return GeodeticToCartesian(
            self._target,
            self._source,
            semi_major_axis=self._semi_major_axis,
            inverse_flattening=self._inverse_flattening,
            longitude_start=self._longitude_start,
        )
