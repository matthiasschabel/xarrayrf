"""Frame CF longitude/latitude variables of an xarray Dataset.

Reads what the CF conventions declare on a Dataset already opened by xarray: one-dimensional
longitude and latitude (dimension or auxiliary coordinates, found by ``standard_name`` or
``units``), the earth figure from a ``latitude_longitude`` grid mapping, and cell bounds. Returns
a geodetic frame (a sphere is the zero-flattening case) and the ``Grid`` placing the variable on
it. Projected grid mappings, rotated poles and two-dimensional coordinates are refused.

The frame is anonymous: a shared earth radius or datum never implies a shared frame, so two
files of one model share a frame only when the caller passes ``frame=``.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import xarray as xr

from xarrayrf import (
    AffineTransform,
    ArrayCoordinates,
    CoordinateSystem,
    Grid,
    ReferenceFrame,
)
from xarrayrf._geometry import adopt_frame
from xarrayrf._sampling import INTERVAL_ROUNDOFF_FACTOR, INTERVAL_TOLERANCE
from xarrayrf.native import Report

_RADIANS = frozenset({"rad", "radian", "radians"})
_PLAIN_DEGREES = frozenset({"deg", "degree", "degrees"})


def _directional(full: str, letter: str) -> frozenset[str]:
    return frozenset(
        spelling
        for stem in ("degree", "degrees")
        for spelling in (f"{stem}_{full}", f"{stem}_{letter}", f"{stem}{letter}")
    )


_EAST, _WEST = _directional("east", "E"), _directional("west", "W")
_NORTH, _SOUTH = _directional("north", "N"), _directional("south", "S")
_ROTATED = frozenset({"grid_longitude", "grid_latitude"})
_FIGURE_RELATIVE_TOLERANCE = 1e-9
"""Agreement required between semi_minor_axis and inverse_flattening when both are given."""


@dataclass(frozen=True)
class CfGeometry:
    """A CF variable's longitude/latitude grid, frame and import report.

    Attributes:
        grid: The longitude and latitude coordinates mapped into ``frame``.
        frame: Geodetic frame with longitude and latitude roles.
        longitude: Name of the longitude coordinate.
        latitude: Name of the latitude coordinate.
        report: Metadata decisions made during import.
    """

    grid: Grid
    frame: ReferenceFrame
    longitude: str
    latitude: str
    report: Report


@dataclass(frozen=True)
class _Axis:
    name: str
    dim: str
    units: str
    frame_unit: str
    sign: float


def _units(coordinate: xr.DataArray) -> str | None:
    units = coordinate.attrs.get("units")
    return units if isinstance(units, str) else None


def _classify(name: str, coordinate: xr.DataArray) -> str | None:
    standard_name = coordinate.attrs.get("standard_name")
    if standard_name in _ROTATED:
        raise ValueError(
            f"coordinate {name!r} is a rotated-pole {standard_name}; rotated poles are not "
            "supported yet"
        )
    units = _units(coordinate)
    if standard_name == "longitude" or units in _EAST | _WEST:
        return "longitude"
    if standard_name == "latitude" or units in _NORTH | _SOUTH:
        return "latitude"
    return None


def _axis(name: str, coordinate: xr.DataArray, role: str) -> tuple[_Axis, list[tuple[str, str]]]:
    units = _units(coordinate)
    forward, backward = (_EAST, _WEST) if role == "longitude" else (_NORTH, _SOUTH)
    frame_unit = "degrees_east" if role == "longitude" else "degrees_north"
    report: list[tuple[str, str]] = []
    if units in _RADIANS:
        return _Axis(name, str(coordinate.dims[0]), units, "rad", 1.0), report
    if units in forward or units in _PLAIN_DEGREES:
        return _Axis(name, str(coordinate.dims[0]), units, frame_unit, 1.0), report
    if units in backward:
        code = "west-longitude" if role == "longitude" else "south-latitude"
        report.append(
            (
                code,
                f"{role} {name!r} is in {units!r}; mapped to {frame_unit} by negation, labels kept",
            )
        )
        return _Axis(name, str(coordinate.dims[0]), units, frame_unit, -1.0), report
    raise ValueError(
        f"{role} {name!r} has units {units!r}; expected degrees (east/north or west/south "
        "spellings) or radians"
    )


def _find_axes(variable: xr.DataArray, name: str) -> dict[str, tuple[str, xr.DataArray]]:
    found: dict[str, list[tuple[str, xr.DataArray]]] = {"longitude": [], "latitude": []}
    for coordinate_name, coordinate in variable.coords.items():
        role = _classify(str(coordinate_name), coordinate)
        if role is None:
            continue
        if coordinate.ndim != 1:
            raise ValueError(
                f"{role} {coordinate_name!r} of {name!r} spans {coordinate.dims}; curvilinear "
                "(two-dimensional) longitude/latitude is not supported yet"
            )
        found[role].append((str(coordinate_name), coordinate))
    result = {}
    for role, candidates in found.items():
        if len(candidates) != 1:
            names = [candidate for candidate, _ in candidates]
            raise ValueError(
                f"{name!r} needs exactly one one-dimensional {role} coordinate, found {names}"
            )
        result[role] = candidates[0]
    if result["longitude"][1].dims == result["latitude"][1].dims:
        raise ValueError(
            f"longitude and latitude of {name!r} share dimension {result['longitude'][1].dims}; "
            "a rectilinear grid needs one dimension each"
        )
    return result


def _number(attrs: Mapping[Hashable, Any], key: str) -> float | None:
    if key not in attrs:
        return None
    value = attrs[key]
    try:
        number = float(np.asarray(value).item())
    except (TypeError, ValueError) as error:
        raise ValueError(f"grid mapping attribute {key}={value!r} is not a number") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"grid mapping attribute {key}={value!r} must be finite and non-negative")
    return number


def _figure(
    dataset: xr.Dataset, variable: xr.DataArray, name: str
) -> tuple[dict[str, str | float] | None, list[tuple[str, str]]]:
    """The figure definition from the grid mapping, or ``None`` when none is declared."""
    mapping_name = variable.attrs.get("grid_mapping", variable.encoding.get("grid_mapping"))
    if mapping_name is None:
        return None, [("figure", f"{name!r} declares no grid_mapping, so no earth figure")]
    if not isinstance(mapping_name, str) or ":" in mapping_name or " " in mapping_name.strip():
        raise ValueError(
            f"{name!r} uses the extended grid_mapping syntax {mapping_name!r}; only a single "
            "grid-mapping variable name is supported"
        )
    if mapping_name not in dataset.variables:
        raise ValueError(f"grid_mapping {mapping_name!r} of {name!r} is not in the dataset")
    attrs = dataset.variables[mapping_name].attrs
    kind = attrs.get("grid_mapping_name")
    if kind == "rotated_latitude_longitude":
        raise ValueError(f"grid_mapping {mapping_name!r} is a rotated pole; not supported yet")
    if kind != "latitude_longitude":
        raise ValueError(
            f"grid_mapping {mapping_name!r} is {kind!r}; only 'latitude_longitude' is supported "
            "(projected mappings need pyproj and are not read here)"
        )
    prime = _number(attrs, "longitude_of_prime_meridian")
    if prime not in (None, 0.0):
        raise ValueError(
            f"grid_mapping {mapping_name!r} has longitude_of_prime_meridian={prime}; only 0 is "
            "supported"
        )
    radius = _number(attrs, "earth_radius")
    a = _number(attrs, "semi_major_axis")
    b = _number(attrs, "semi_minor_axis")
    inverse_f = _number(attrs, "inverse_flattening")
    # Ranges the chart transitions require, checked here so that no frame declares a figure
    # its transitions would refuse.
    for key, value in (("earth_radius", radius), ("semi_major_axis", a), ("semi_minor_axis", b)):
        if value == 0.0:
            raise ValueError(f"grid_mapping {mapping_name!r} has {key}=0; it must be positive")
    if inverse_f is not None and 0.0 < inverse_f <= 1.0:
        raise ValueError(
            f"grid_mapping {mapping_name!r} has inverse_flattening={inverse_f}; it must be 0 (a "
            "sphere) or greater than 1"
        )
    if a is not None and b is not None and b > a:
        raise ValueError(
            f"grid_mapping {mapping_name!r} has semi_minor_axis {b} greater than semi_major_axis "
            f"{a}; a prolate figure is not an earth figure"
        )
    if radius is not None and (a, b, inverse_f) != (None, None, None):
        raise ValueError(
            f"grid_mapping {mapping_name!r} declares earth_radius together with ellipsoid "
            "parameters; CF allows one or the other"
        )
    definition: dict[str, str | float]
    if radius is not None:
        definition = {"figure": "sphere", "semi_major_axis": radius}
    elif a is None:
        if (b, inverse_f) != (None, None):
            raise ValueError(f"grid_mapping {mapping_name!r} lacks semi_major_axis")
        return None, [("figure", f"grid_mapping {mapping_name!r} declares no earth figure")]
    else:
        from_minor = None if b is None else (0.0 if b == a else a / (a - b))
        if from_minor is not None and inverse_f is not None:
            scale = max(abs(from_minor), abs(inverse_f), 1.0)
            if abs(from_minor - inverse_f) > _FIGURE_RELATIVE_TOLERANCE * scale:
                raise ValueError(
                    f"grid_mapping {mapping_name!r}: semi_minor_axis implies inverse_flattening "
                    f"{from_minor}, but {inverse_f} is declared"
                )
        flattening = inverse_f if inverse_f is not None else from_minor
        if flattening in (None, 0.0):
            definition = {"figure": "sphere", "semi_major_axis": a}
        else:
            assert flattening is not None
            definition = {
                "figure": "ellipsoid",
                "semi_major_axis": a,
                "inverse_flattening": flattening,
            }
    definition["length_unit"] = "m"
    for source_key, target_key in (
        ("horizontal_datum_name", "datum_name"),
        ("reference_ellipsoid_name", "ellipsoid_name"),
    ):
        value = attrs.get(source_key)
        if isinstance(value, str) and value:
            definition[target_key] = value
    return definition, [("figure", f"earth figure from grid_mapping {mapping_name!r}")]


def _intervals(
    dataset: xr.Dataset, axis: _Axis, coordinate: xr.DataArray
) -> tuple[npt.NDArray[np.float64] | None, float | None, list[tuple[str, str]]]:
    """Sorted bounds and the one sample offset they imply, or nothing with a reason."""
    bounds_name = coordinate.attrs.get("bounds", coordinate.encoding.get("bounds"))
    if bounds_name is None:
        return None, None, [("no-bounds", f"{axis.name!r} declares no bounds")]
    # A declared reference that leads nowhere is an error, not an absence: bounds decide coverage.
    if not isinstance(bounds_name, str) or bounds_name not in dataset.variables:
        raise ValueError(
            f"{axis.name!r} declares bounds {bounds_name!r}, which is not a variable of the dataset"
        )
    bounds = np.sort(np.asarray(dataset.variables[bounds_name].values, dtype=np.float64), axis=-1)
    labels = np.asarray(coordinate.values, dtype=np.float64)
    if bounds.shape != (labels.size, 2):
        raise ValueError(
            f"bounds {bounds_name!r} of {axis.name!r} have shape {bounds.shape}; expected "
            f"{(labels.size, 2)}"
        )
    width = bounds[:, 1] - bounds[:, 0]
    if np.any(width <= 0) or not np.all(np.isfinite(bounds)):
        raise ValueError(f"bounds {bounds_name!r} of {axis.name!r} need finite, positive widths")
    offsets = (labels - bounds[:, 0]) / width
    offset = float(np.mean(offsets))
    # The tolerance freeze_intervals applies, so the Grid accepts what is kept here.
    tolerance = np.maximum(
        INTERVAL_TOLERANCE * width,
        INTERVAL_ROUNDOFF_FACTOR * np.finfo(np.float64).eps * np.max(np.abs(bounds), axis=-1),
    )
    if np.any(np.abs(labels - (bounds[:, 0] + offset * width)) > tolerance) or not 0 <= offset <= 1:
        return (
            None,
            None,
            [
                (
                    "bounds-off-centre",
                    f"bounds {bounds_name!r} of {axis.name!r} place labels at varying positions "
                    "within their cells, which one sample offset cannot express; not declared",
                )
            ],
        )
    return bounds, offset, []


def from_dataset(
    dataset: xr.Dataset,
    variable: str,
    *,
    frame: ReferenceFrame | xr.DataArray | None = None,
) -> CfGeometry:
    """Import a CF variable's longitude/latitude geometry without reading its data.

    Args:
        dataset: Dataset opened by xarray, with CF attributes.
        variable: Name of the data variable.
        frame: Frame or framed array to adopt, overriding the imported identity. Required when
            the dataset declares no earth figure.

    Returns:
        The grid, its geodetic frame and a report of every normalization.

    Raises:
        TypeError: If ``dataset`` is not a Dataset or ``variable`` not a string.
        KeyError: If ``variable`` is not a data variable of ``dataset``.
        ValueError: If longitude/latitude cannot be identified as rectilinear and one-dimensional,
            the grid mapping is not a supported ``latitude_longitude`` mapping, its parameters
            disagree, no figure is declared and ``frame`` is not given, or adoption fails.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"dataset must be an xarray Dataset, got {type(dataset).__name__}")
    if not isinstance(variable, str):
        raise TypeError(f"variable must be a string, got {type(variable).__name__}")
    if variable not in dataset.data_vars:
        raise KeyError(f"{variable!r} is not a data variable of the dataset")
    array = dataset[variable]
    found = _find_axes(array, variable)
    report: list[tuple[str, str]] = []
    axes: list[tuple[_Axis, xr.DataArray]] = []
    for role in ("longitude", "latitude"):
        axis, entries = _axis(*found[role], role)
        axes.append((axis, found[role][1]))
        report += entries
    (lon, _), (lat, _) = axes
    definition, entries = _figure(dataset, array, variable)
    report += entries
    if definition is None and frame is None:
        raise ValueError(
            f"{variable!r} declares no earth figure (CF leaves it unspecified without a "
            "latitude_longitude grid mapping that gives one); pass frame= to state which frame "
            "it is in"
        )
    if definition is not None and frame is not None:
        report.append(("figure-ignored", "frame= replaces the imported frame and its figure"))
    intervals: dict[str, npt.NDArray[np.float64]] = {}
    offsets: list[float | None] = []
    for axis, coordinate in axes:
        bounds, offset, entries = _intervals(dataset, axis, coordinate)
        report += entries
        offsets.append(offset)
        if bounds is not None:
            intervals[axis.name] = bounds
    system = CoordinateSystem(
        (lon.name, lat.name),
        (lon.frame_unit, lat.frame_unit),
        representation="geodetic",
        axis_roles=("longitude", "latitude"),
    )
    imported = ReferenceFrame.anonymous(system, definition=definition)
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(
            (lon.name, lat.name), (lon.units, lat.units), sample_offset=tuple(offsets)
        ),
        target=imported,
        matrix=((lon.sign, 0.0), (0.0, lat.sign)),
        translation=(0.0, 0.0),
    )
    if frame is not None:
        transform = adopt_frame(transform, frame, name="frame")
    assert isinstance(transform.target, ReferenceFrame)
    grid = Grid(
        transform,
        {axis.name: (axis.dim, np.asarray(coordinate.values)) for axis, coordinate in axes},
        intervals=intervals or None,
    )
    return CfGeometry(grid, transform.target, lon.name, lat.name, tuple(report))


def frame_variable(
    dataset: xr.Dataset,
    variable: str,
    *,
    frame: ReferenceFrame | xr.DataArray | None = None,
) -> xr.DataArray:
    """Return a CF variable framed on its longitude/latitude grid, data untouched.

    Other coordinates (time, pressure level), attributes and lazy (dask) data are kept: the
    existing longitude and latitude coordinates must agree with the imported grid, which they
    do by construction.

    Args:
        dataset: Dataset opened by xarray, with CF attributes.
        variable: Name of the data variable.
        frame: As for :func:`from_dataset`.

    Raises:
        As :func:`from_dataset`.
    """
    geometry = from_dataset(dataset, variable, frame=frame)
    return cast(xr.DataArray, dataset[variable].rf.frame(geometry.grid))


__all__ = ["CfGeometry", "frame_variable", "from_dataset"]
