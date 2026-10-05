"""Translate projected GeoTIFF metadata into xarrayrf geometry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pyproj
import rasterio  # type: ignore[import-untyped]
import xarray as xr

from xarrayrf import AffineTransform, ArrayCoordinates, CoordinateSystem, ReferenceFrame
from xarrayrf._geometry import adopt_frame
from xarrayrf.native import (
    CoordinateSpec,
    DuckArray,
    Report,
    _grid_and_coords,
    frame_array,
    index_coordinate,
)

from ._reader import open as open

_UNITS = {"metre": "m", "US survey foot": "US_survey_foot", "foot": "ft"}


@dataclass(frozen=True)
class GeoTiffGeometry:
    """Projected raster dimensions, coordinates, affine, frame and import report.

    Attributes:
        dims: Pixel dimension order, including the non-geometry band dimension.
        coords: Xarray-compatible dimension coordinate declarations.
        transform: Pixel-centre coordinates to projected CRS coordinates.
        frame: The CRS reference frame.
        report: Metadata decisions made during import.
        nodata: Stored nodata sentinel, recorded but not applied to pixels.
    """

    dims: tuple[str, ...]
    coords: Mapping[str, CoordinateSpec]
    transform: AffineTransform
    frame: ReferenceFrame
    report: Report
    nodata: int | float | None


def crs_frame(crs: pyproj.CRS | rasterio.crs.CRS) -> ReferenceFrame:
    """Return the shared projected frame declared by a pyproj or rasterio CRS.

    Args:
        crs: Projected CRS with two linear axes.

    Returns:
        Shared declared frame for an exact authority code, otherwise a new anonymous frame.

    Raises:
        TypeError: If crs is neither a pyproj nor rasterio CRS.
        ValueError: If the CRS is angular, vertical, compound, or has unsupported units.
    """
    if not isinstance(crs, pyproj.CRS | rasterio.crs.CRS):
        raise TypeError(f"crs must be a pyproj or rasterio CRS, got {type(crs).__name__}")
    parsed = pyproj.CRS.from_user_input(crs)
    if parsed.is_geographic or any(
        axis.unit_name in ("degree", "radian") for axis in parsed.axis_info
    ):
        raise ValueError("angular coordinates are not representable yet")
    if (
        parsed.is_compound
        or parsed.is_vertical
        or not parsed.is_projected
        or len(parsed.axis_info) != 2
    ):
        raise ValueError(
            "CRS must be a two-axis projected CRS; compound and vertical CRSs are unsupported"
        )
    axes = tuple(axis.name for axis in parsed.axis_info)
    units: list[str] = []
    for axis in parsed.axis_info:
        if axis.unit_name not in _UNITS:
            raise ValueError(f"unsupported CRS axis unit {axis.unit_name!r}")
        units.append(_UNITS[axis.unit_name])
    system = CoordinateSystem(axes, units, axis_types=("space", "space"))
    authority = parsed.to_authority(min_confidence=100)
    if authority is None:
        return ReferenceFrame.anonymous(system, definition={"crs": parsed.to_wkt()})
    name, code = authority
    return ReferenceFrame.declared(
        (name.lower(), code), system, definition={"crs": f"{name}:{code}"}
    )


def _easting_northing_rows(crs: pyproj.CRS) -> tuple[int, int]:
    """Return which rasterio (x, y) component feeds each CRS axis, in CRS axis order.

    Rasterio and GDAL always express an affine in easting/northing order, whatever order the
    CRS authority declares (EPSG:3035 lists northing first). The frame follows the authority.
    Only pixel import needs this; ``crs_frame`` accepts any two-axis projected CRS.
    """
    rows: list[int] = []
    for axis in crs.axis_info:
        if axis.direction == "east":
            rows.append(0)
        elif axis.direction == "north":
            rows.append(1)
        else:
            raise ValueError(
                f"unsupported CRS axis direction {axis.direction!r} for {axis.name!r}; only "
                "east and north are representable"
            )
    if sorted(rows) != [0, 1]:
        raise ValueError("CRS must have one easting and one northing axis")
    return rows[0], rows[1]


def from_profile(
    profile: Mapping[str, Any],
    *,
    frame: ReferenceFrame | xr.DataArray | None = None,
    area_or_point: str = "Area",
) -> GeoTiffGeometry:
    """Import projected GeoTIFF metadata without reading pixels.

    Args:
        profile: Rasterio profile with CRS, affine, width, height and band count.
        frame: Frame or framed array to adopt, overriding the imported CRS identity.
        area_or_point: GDAL AREA_OR_POINT tag, ``Area`` or ``Point``.

    Returns:
        Geometry whose affine locates pixel centres for either tag.

    Raises:
        TypeError: If the profile or tag has an invalid type.
        ValueError: If required metadata, CRS or tag cannot be represented.
    """
    if not isinstance(profile, Mapping):
        raise TypeError(f"profile must be a mapping, got {type(profile).__name__}")
    if not isinstance(area_or_point, str):
        raise TypeError(f"area_or_point must be a string, got {type(area_or_point).__name__}")
    if area_or_point not in ("Area", "Point"):
        raise ValueError(f"area_or_point must be 'Area' or 'Point', got {area_or_point!r}")
    for key in ("crs", "transform", "width", "height", "count"):
        if key not in profile:
            raise ValueError(f"profile must contain {key!r}")
    affine = profile["transform"]
    if not isinstance(affine, rasterio.Affine):
        raise TypeError(f"profile transform must be an Affine, got {type(affine).__name__}")
    shape = tuple(profile[key] for key in ("count", "height", "width"))
    if any(isinstance(size, bool) or not isinstance(size, int) for size in shape):
        raise TypeError("profile count, height and width must be integers")
    if any(size <= 0 for size in shape):
        raise ValueError(f"profile count, height and width must be positive, got {shape}")
    imported_frame = crs_frame(profile["crs"])
    x00, y00 = rasterio.transform.xy(affine, 0, 0, offset="center")
    source = ArrayCoordinates(
        ("row", "column"),
        ("1", "1"),
        axis_types=("space", "space"),
        sample_offset=(0.5, 0.5) if area_or_point == "Area" else None,
    )
    # Rows in rasterio's (x, y) order, then permuted into the CRS authority's axis order.
    xy_rows = ((affine.b, affine.a), (affine.e, affine.d))
    xy_translation = (x00, y00)
    order = _easting_northing_rows(pyproj.CRS.from_user_input(profile["crs"]))
    transform = AffineTransform.from_matrix(
        source=source,
        target=imported_frame,
        matrix=tuple(xy_rows[i] for i in order),
        translation=tuple(xy_translation[i] for i in order),
    )
    if frame is not None:
        transform = adopt_frame(transform, frame, name="frame")
    assert isinstance(transform.target, ReferenceFrame)
    frame = transform.target
    coords = {
        dim: index_coordinate(dim, size, start=1 if dim == "band" else 0)
        for dim, size in zip(("band", "row", "column"), shape, strict=True)
    }
    nodata = profile.get("nodata")
    report: Report = (("pixel-convention", f"{area_or_point} pixels at rasterio centres"),)
    return GeoTiffGeometry(
        ("band", "row", "column"), MappingProxyType(coords), transform, frame, report, nodata
    )


def to_dataarray(geometry: GeoTiffGeometry, data: DuckArray) -> xr.DataArray:
    """Bind caller-supplied pixels to GeoTIFF geometry without evaluating them.

    Args:
        geometry: Geometry returned by ``from_profile``.
        data: NumPy or dask array in band, row, column order.

    Returns:
        Framed DataArray with the stored nodata sentinel in attrs.

    Raises:
        TypeError: If geometry or data has an invalid type.
        ValueError: If the pixel shape differs from the profile.
    """
    if not isinstance(geometry, GeoTiffGeometry):
        raise TypeError(f"geometry must be a GeoTiffGeometry, got {type(geometry).__name__}")
    grid, other_coords = _grid_and_coords(geometry.transform, geometry.coords)
    return frame_array(
        data,
        grid,
        dims=geometry.dims,
        coords=other_coords,
        attrs={"nodata": geometry.nodata},
    )


__all__ = ["GeoTiffGeometry", "crs_frame", "from_profile", "open", "to_dataarray"]
