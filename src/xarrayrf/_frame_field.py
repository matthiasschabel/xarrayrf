"""Component bases that vary by position: frame fields, and east/north/up on the Earth.

A frame field assigns every point of a chart a basis of its tangent space. Components declared
in it (eastward and northward wind, a tetrad's components) mean different directions at
different points, so converting them needs the field evaluated at each sample. A frame field
maps tangent spaces, not points, so it is not a transform.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ._chart_transitions import (
    CartesianToGeodetic,
    CartesianToSpherical,
    GeodeticToCartesian,
    SphericalToCartesian,
    check_figure_definition,
    latitude_trig,
    local_frame,
)
from ._coordinate_system import GEODETIC, SPHERICAL
from ._frame import ReferenceFrame
from ._validation import angular_period, check_names, real_float_array


@runtime_checkable
class FrameField(Protocol):
    """A basis of the tangent space at every point of a chart.

    Components ``u`` in the field at a point ``x`` are ``matrix(x) @ u`` in the coordinate basis
    of ``base`` (one column per basis vector, in ``base``'s axis order). Implement this protocol
    for other fields, such as a tetrad; nothing is registered. Equality and hashing must follow
    the declaration, since component declarations compare their bases.
    """

    @property
    def base(self) -> ReferenceFrame:
        """The chart the field is defined over; points are in its coordinates."""
        ...

    @property
    def axes(self) -> tuple[str, ...]:
        """Component labels, one per basis vector."""
        ...

    @property
    def units(self) -> tuple[str | None, ...]:
        """The unit each basis vector is measured in, one per basis vector."""
        ...

    def matrix(self, at: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """The basis vectors in ``base``'s coordinate basis at points shaped ``(..., k)``.

        Returns ``(..., k, n)``; raises ``ValueError`` naming the first point where the field
        has no such matrix.
        """
        ...

    def inverse_matrix(self, at: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """The inverse of :meth:`matrix` at the same points, ``(..., n, k)``."""
        ...


_DEFAULT_AXES = ("east", "north", "up")


class EastNorthUp:
    """The local east/north/up frame of a spherical or geodetic chart.

    East points toward increasing longitude, north toward the north pole along the meridian,
    and up along the normal (the ellipsoid normal for geodetic charts, radially for spherical
    ones); the three are orthonormal, measured in the chart's length unit. Components are ordered
    by role (east, north, up) whatever the chart's axis order. On a geodetic chart without a
    height axis the field has east and north only, on the surface.

    At a pole the directions are the limit along the sample's meridian, which is how gridded
    data store pole rows. The chart's own coordinate basis does not exist there (a longitude
    step has no length), so :meth:`matrix` and :meth:`inverse_matrix` raise at the poles;
    :meth:`directions`, the vectors in Earth-centred axes, does not.

    Args:
        base: A geodetic chart (longitude, latitude, optionally height) or a spherical chart
            (longitude, latitude, radius), with no other axes.
        semi_major_axis: The figure's equatorial radius in the length unit (geodetic only).
        inverse_flattening: ``0.0`` for a sphere, otherwise greater than 1 (geodetic only).
        length_unit: The length unit, for a geodetic chart without a height axis (taken from
            the frame definition's ``length_unit`` when omitted); must agree with the radial
            axis's unit when one exists.
        axes: Component labels; default ``("east", "north", "up")`` or its first two.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the chart is not a supported angular chart, parameters are missing,
            invalid or not applicable, the figure disagrees with the frame definition, or the
            labels do not fit.
    """

    __slots__ = (
        "_axes",
        "_base",
        "_figure",
        "_inverse_flattening",
        "_latitude",
        "_latitude_scale",
        "_length_unit",
        "_longitude",
        "_longitude_scale",
        "_pole",
        "_radial",
        "_semi_major_axis",
    )

    def __init__(
        self,
        base: ReferenceFrame,
        *,
        semi_major_axis: float | None = None,
        inverse_flattening: float | None = None,
        length_unit: str | None = None,
        axes: tuple[str, ...] | None = None,
    ) -> None:
        if not isinstance(base, ReferenceFrame):
            raise TypeError(f"base must be a ReferenceFrame, got {type(base).__name__}")
        system = base.coordinate_system
        roles = system.axis_roles
        if system.representation not in (GEODETIC, SPHERICAL):
            raise ValueError(
                f"east/north/up needs a geodetic or spherical chart, got {system.representation}"
            )
        if None in roles:
            raise ValueError(
                f"the chart has axes without a role {system.axes}; east/north/up spans the "
                "chart's spatial axes only"
            )
        radial = "height" if system.representation == GEODETIC else "radius"
        if system.representation == SPHERICAL and radial not in roles:
            raise ValueError(
                "a spherical chart without a radius axis has no length scale for east and "
                "north; declare a radius axis or use a geodetic chart with a figure"
            )
        self._base = base
        self._longitude = roles.index("longitude")
        self._latitude = roles.index("latitude")
        self._radial = roles.index(radial) if radial in roles else None
        units = system.units
        longitude_unit, latitude_unit = units[self._longitude], units[self._latitude]
        assert longitude_unit is not None and latitude_unit is not None  # roles declare units
        self._longitude_scale = 2.0 * math.pi / angular_period(longitude_unit)
        self._latitude_scale = 2.0 * math.pi / angular_period(latitude_unit)
        self._pole = angular_period(latitude_unit) / 4.0
        radial_unit = None if self._radial is None else units[self._radial]
        if system.representation == SPHERICAL:
            if semi_major_axis is not None or inverse_flattening is not None:
                raise ValueError(
                    "a spherical chart measures its radius directly; semi_major_axis and "
                    "inverse_flattening apply to geodetic charts"
                )
            self._semi_major_axis, self._inverse_flattening = 0.0, 0.0
            self._figure = (0.0, 1.0)
        else:
            if semi_major_axis is None or inverse_flattening is None:
                raise ValueError(
                    "a geodetic chart needs semi_major_axis and inverse_flattening for its figure"
                )
            a = _real(semi_major_axis, "semi_major_axis")
            inverse_f = _real(inverse_flattening, "inverse_flattening")
            if a <= 0.0:
                raise ValueError(f"semi_major_axis must be positive, got {a!r}")
            if inverse_f != 0.0 and inverse_f <= 1.0:
                raise ValueError(
                    f"inverse_flattening must be 0 (a sphere) or greater than 1, got {inverse_f!r}"
                )
            self._semi_major_axis, self._inverse_flattening = a, inverse_f
            flattening = 0.0 if inverse_f == 0.0 else 1.0 / inverse_f
            self._figure = (a, (1.0 - flattening) ** 2)
        declared = base.definition.get("length_unit")
        resolved = length_unit if length_unit is not None else radial_unit
        if resolved is None and isinstance(declared, str):
            resolved = declared
        if resolved is None:
            raise ValueError(
                "the chart has no length axis; pass length_unit= the unit of semi_major_axis"
            )
        if not isinstance(resolved, str):
            raise TypeError(f"length_unit must be a string, got {type(resolved).__name__}")
        if radial_unit is not None and resolved != radial_unit:
            raise ValueError(
                f"length_unit {resolved!r} disagrees with the chart's {radial} unit {radial_unit!r}"
            )
        self._length_unit = resolved
        if system.representation == GEODETIC:
            check_figure_definition(
                base.definition,
                self._semi_major_axis,
                self._inverse_flattening,
                resolved,
                what="field",
            )
        count = len(system.axes)
        labels = _DEFAULT_AXES[:count] if axes is None else check_names(axes, field="axes")
        if len(labels) != count:
            raise ValueError(f"axes must give {count} labels for this chart, got {labels}")
        self._axes = labels

    @property
    def base(self) -> ReferenceFrame:
        """The chart the field is defined over."""
        return self._base

    @property
    def axes(self) -> tuple[str, ...]:
        """Component labels, in role order: east, north and (for 3-D charts) up."""
        return self._axes

    @property
    def units(self) -> tuple[str | None, ...]:
        """The length unit, for every basis vector."""
        return (self._length_unit,) * len(self._axes)

    def _pole_signs(self) -> tuple[float, ...]:
        """Per-component signs that carry a row across a pole onto the antipodal meridian.

        Continuing a meridian over a pole, the frame that varies smoothly along the path is
        (east, north, up) on the near side and (-east, -north, up) of the far side: a half-turn
        about up, determinant +1 in 2-D and 3-D alike.
        """
        return (-1.0, -1.0, 1.0)[: len(self._axes)]

    @property
    def semi_major_axis(self) -> float:
        """The figure's equatorial radius (0 for a spherical chart)."""
        return self._semi_major_axis

    @property
    def inverse_flattening(self) -> float:
        """The figure's inverse flattening (0 for a sphere or a spherical chart)."""
        return self._inverse_flattening

    @property
    def length_unit(self) -> str:
        """The unit lengths are measured in."""
        return self._length_unit

    def _frame(self, at: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], ...]:
        values = real_float_array(at, field="at")
        count = len(self._axes)
        if values.ndim == 0 or values.shape[-1] != count:
            raise ValueError(
                f"at must have a trailing axis of length {count} ordered as {self._base.axes}, "
                f"got shape {values.shape}"
            )
        longitude = values[..., self._longitude] * self._longitude_scale
        cos_lat, sin_lat = latitude_trig(
            values[..., self._latitude], self._latitude_scale, self._pole
        )
        height = 0.0 if self._radial is None else values[..., self._radial]
        a, b2 = self._figure
        return local_frame(longitude, cos_lat, sin_lat, height, a, b2)

    def _scales(self, at: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], tuple[int, ...]]:
        """Coordinate-vector lengths per declared unit, by role; refuses where one vanishes."""
        _, _, _, east, north = self._frame(at)
        scales = [east * self._longitude_scale, north * self._latitude_scale]
        if self._radial is not None:
            scales.append(np.ones_like(east))
        stacked = np.stack(scales, axis=-1)
        bad = ~np.isfinite(stacked).all(axis=-1) | (stacked == 0.0).any(axis=-1)
        if bad.any():
            index = tuple(int(i) for i in np.argwhere(bad)[0])
            raise ValueError(
                f"at point {index} a chart coordinate vector has no length (a pole, or the "
                "centre), so the chart basis and east/north/up do not convert there"
            )
        chart_axes = (self._longitude, self._latitude) + (
            () if self._radial is None else (self._radial,)
        )
        return stacked, chart_axes

    def matrix(self, at: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """East/north/up in the chart's coordinate basis, ``(..., k, k)``; raises at poles."""
        scales, chart_axes = self._scales(at)
        result = np.zeros((*scales.shape, scales.shape[-1]))
        for role, axis in enumerate(chart_axes):
            result[..., axis, role] = 1.0 / scales[..., role]
        return result

    def inverse_matrix(self, at: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Chart-basis components to east/north/up, ``(..., k, k)``; raises at poles."""
        scales, chart_axes = self._scales(at)
        result = np.zeros((*scales.shape, scales.shape[-1]))
        for role, axis in enumerate(chart_axes):
            result[..., role, axis] = scales[..., role]
        return result

    def directions(self, at: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """East, north (and up) as unit vectors in Earth-centred axes, ``(..., 3, n)``.

        The Earth-centred axes are ``(x, y, z)`` as the chart transitions define them: x toward
        longitude 0 on the equator, z toward the north pole. Defined at the poles too.
        """
        east, north, up, _, _ = self._frame(at)
        vectors = (east, north) if self._radial is None else (east, north, up)
        return np.stack(vectors, axis=-1)

    def matches(self, transition: object) -> bool:
        """Whether ``transition`` maps this field's chart to its own Earth-centred chart."""
        if self._base.coordinate_system.representation == SPHERICAL:
            if isinstance(transition, SphericalToCartesian):
                return transition.source == self._base
            if isinstance(transition, CartesianToSpherical):
                return transition.target == self._base
            return False
        figure = (self._semi_major_axis, self._inverse_flattening)
        if isinstance(transition, GeodeticToCartesian):
            return transition.source == self._base and figure == (
                transition.semi_major_axis,
                transition.inverse_flattening,
            )
        if isinstance(transition, CartesianToGeodetic):
            return transition.target == self._base and figure == (
                transition.semi_major_axis,
                transition.inverse_flattening,
            )
        return False

    def _key(self) -> tuple[object, ...]:
        return (
            self._base,
            self._semi_major_axis,
            self._inverse_flattening,
            self._length_unit,
            self._axes,
        )

    def __eq__(self, other: object) -> bool:
        """Compare the chart, figure, length unit and labels."""
        if not isinstance(other, EastNorthUp):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        """Hash consistently with :meth:`__eq__`."""
        return hash((EastNorthUp, self._key()))

    def __reduce__(self) -> tuple[object, ...]:
        geodetic = self._base.coordinate_system.representation == GEODETIC
        return (
            _rebuild,
            (
                self._base,
                self._semi_major_axis if geodetic else None,
                self._inverse_flattening if geodetic else None,
                self._length_unit,
                self._axes,
            ),
        )

    def __repr__(self) -> str:
        """Return a representation naming the chart, the figure and the labels."""
        return (
            f"EastNorthUp(base={self._base!r}, semi_major_axis={self._semi_major_axis!r}, "
            f"inverse_flattening={self._inverse_flattening!r}, "
            f"length_unit={self._length_unit!r}, axes={self._axes!r})"
        )


def _rebuild(
    base: ReferenceFrame,
    semi_major_axis: float | None,
    inverse_flattening: float | None,
    length_unit: str,
    axes: tuple[str, ...],
) -> EastNorthUp:
    return EastNorthUp(
        base,
        semi_major_axis=semi_major_axis,
        inverse_flattening=inverse_flattening,
        length_unit=length_unit,
        axes=axes,
    )


def _real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"{field} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {number!r}")
    return number
