"""Translate OME-Zarr 0.6 coordinate metadata without opening a store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np
import xarray as xr
from ome_zarr_models.v06 import coordinate_transforms as ct
from ome_zarr_models.v06.multiscales import Multiscale
from ome_zarr_models.v06.scene import SceneAttrs

from xarrayrf import AffineTransform, ReferenceFrame
from xarrayrf.native import CoordinateSpec, DuckArray, Report, _grid_and_coords, frame_array

from ._export import to_multiscale_level as to_multiscale_level
from ._export import to_transform as to_transform
from ._metadata import NAMESPACE as NAMESPACE
from ._metadata import (
    FrameMap,
    JsonObject,
    _frame,
    _import_transform,
    _join,
    _location,
    _reject_rotation_path,
    _subset,
    _system,
    _transform,
)
from ._reader import open as open


@dataclass(frozen=True)
class NgffLevel:
    """One array level's dimensions, index coordinates and intrinsic transform."""

    dims: tuple[str, ...]
    coords: Mapping[str, CoordinateSpec]
    transform: AffineTransform


@dataclass(frozen=True)
class NgffMultiscale:
    """A multiscale's shared intrinsic frame, levels and additional transforms."""

    frame: ReferenceFrame
    levels: Mapping[str, NgffLevel]
    transforms: tuple[AffineTransform, ...]
    report: Report


@dataclass(frozen=True)
class NgffScene:
    """A scene's frame-to-frame transforms and import report."""

    transforms: tuple[AffineTransform, ...]
    report: Report


def to_dataarray(level: NgffLevel, data: DuckArray) -> xr.DataArray:
    """Bind caller-supplied pixels to an imported NGFF level without evaluating them.

    Args:
        level: Level returned by ``from_multiscale``.
        data: NumPy or dask array in the level's dimension order.

    Returns:
        DataArray carrying the level transform and sharing the supplied pixels.

    Raises:
        TypeError: If ``level`` has the wrong type or ``data`` is not a duck array.
        ValueError: If the pixel shape differs from the declared level shape.
    """
    if not isinstance(level, NgffLevel):
        raise TypeError(f"level must be a NgffLevel, got {type(level).__name__}")
    grid, other_coords = _grid_and_coords(level.transform, level.coords)
    return frame_array(data, grid, dims=level.dims, coords=other_coords)


def coordinate_system(
    cs: ct.CoordinateSystem | JsonObject, *, store: str | None = None, group: str = ""
) -> tuple[ReferenceFrame, Report]:
    """Import a named v06 coordinate system as a local or store-resolved frame.

    Args:
        cs: v06 model or equivalent JSON attributes.
        store: Resolved URI without a trailing slash; omitted identities are local.
        group: Relative group path containing the system.

    Returns:
        The named reference frame and an import report.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a path or axis cannot be represented.
    """
    store, group = _location(store, group)
    report: list[tuple[str, str]] = []
    frame = _frame(_system(cs), store, group, None, {}, report)
    return frame, tuple(report)


def transform(
    t: ct.Transform | JsonObject,
    systems: Sequence[ct.CoordinateSystem | JsonObject],
    *,
    store: str | None = None,
    group: str = "",
    dims: Mapping[str, Sequence[str]] | None = None,
    frames: FrameMap | None = None,
) -> tuple[AffineTransform, Report]:
    """Import a v06 coordinate transform as one affine between resolved endpoints.

    Args:
        t: v06 transform model or equivalent JSON attributes.
        systems: Named coordinate systems in the current group.
        store: Optional resolved store URI.
        group: Relative group path.
        dims: Dimension names indexed by array path for path-only endpoints.
        frames: Previously resolved local frames indexed by ``(group path, name)``.

    Returns:
        An affine transform and an import report; discrete identity axes are omitted.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If endpoints or transform parameters cannot be represented.
    """
    store, group = _location(store, group)
    if isinstance(systems, str | bytes) or not isinstance(systems, Sequence):
        raise TypeError("systems must be a sequence of v06 coordinate systems")
    if dims is not None and not isinstance(dims, Mapping):
        raise TypeError("dims must map array paths to dimension names")
    if frames is not None and not isinstance(frames, Mapping):
        raise TypeError("frames must map (group path, name) to ReferenceFrame")
    checked = [_system(cs) for cs in systems]
    if len({cs.name for cs in checked}) != len(checked):
        raise ValueError("systems must have unique names")
    scope = {(group, cs.name): cs for cs in checked}
    report: list[tuple[str, str]] = []
    imported = _import_transform(_transform(t), scope, dims, store, group, frames, {}, report)
    return imported, tuple(report)


def from_multiscale(
    ms: Multiscale | JsonObject,
    *,
    shapes: Mapping[str, Sequence[int]],
    dims: Mapping[str, Sequence[str]] | None = None,
    store: str | None = None,
    group: str = "",
    frames: FrameMap | None = None,
) -> NgffMultiscale:
    """Import array levels and additional transforms from v06 multiscale metadata.

    Args:
        ms: v06 multiscale model or equivalent JSON attributes.
        shapes: Array shapes keyed by dataset path; no array is read.
        dims: Optional dimension names keyed by dataset path.
        store: Optional resolved store URI.
        group: Relative group path.
        frames: Previously resolved local frames.

    Returns:
        The intrinsic frame, levels, other transforms and an import report.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If metadata, shapes or endpoints cannot be represented.
    """
    store, group = _location(store, group)
    if not isinstance(shapes, Mapping):
        raise TypeError("shapes must map dataset paths to array shapes")
    if dims is not None and not isinstance(dims, Mapping):
        raise TypeError("dims must map dataset paths to dimension names")
    if frames is not None and not isinstance(frames, Mapping):
        raise TypeError("frames must map (group path, name) to ReferenceFrame")
    if not isinstance(ms, Multiscale):
        _reject_rotation_path(ms)
    model = ms if isinstance(ms, Multiscale) else Multiscale.model_validate(ms)
    systems = {(group, cs.name): cs for cs in model.coordinateSystems}
    report: list[tuple[str, str]] = []
    cache: dict[tuple[str, str], ReferenceFrame] = {}
    intrinsic_name: str | None = None
    for dataset in model.datasets:
        for item in dataset.coordinateTransformations:
            if item.output is None or item.output.name is None or item.output.path is not None:
                raise ValueError(
                    f"dataset {dataset.path!r} must output to a named intrinsic system"
                )
            if intrinsic_name is None:
                intrinsic_name = item.output.name
            elif item.output.name != intrinsic_name:
                raise ValueError(
                    f"dataset {dataset.path!r} must output to intrinsic system {intrinsic_name!r}"
                )
    if intrinsic_name is None:
        raise ValueError("multiscale needs at least one dataset with an intrinsic system")
    intrinsic_system = systems[(group, intrinsic_name)]
    full_frame = _frame(intrinsic_system, store, group, frames, cache, report)
    kept = [
        i
        for i, axis in enumerate(intrinsic_system.axes)
        if not (axis.discrete is True or axis.type == "channel")
    ]
    frame = (
        cast(ReferenceFrame, _subset(full_frame, kept))
        if len(kept) != len(full_frame.axes)
        else full_frame
    )
    result: dict[str, NgffLevel] = {}
    for dataset in model.datasets:
        path = dataset.path
        _join(group, path)
        shape = shapes.get(path)
        if (
            not isinstance(shape, Sequence)
            or isinstance(shape, str | bytes)
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape)
        ):
            raise ValueError(f"dataset {path!r} needs a shape of nonnegative integers")
        if (
            dims is not None
            and path in dims
            and (isinstance(dims[path], str | bytes) or not isinstance(dims[path], Sequence))
        ):
            raise TypeError(f"dataset {path!r} dims must be a sequence of dimension names")
        names = tuple(dims[path]) if dims is not None and path in dims else full_frame.axes
        if len(names) != len(shape) or len(set(names)) != len(names):
            raise ValueError(f"dataset {path!r} dims must uniquely name every shape axis")
        spec = {
            name: (name, np.arange(length, dtype=np.int64), MappingProxyType({"units": "1"}))
            for name, length in zip(names, shape, strict=True)
        }
        level_transform = _import_transform(
            dataset.coordinateTransformations[0],
            systems,
            {path: names},
            store,
            group,
            frames,
            cache,
            report,
        )
        assert level_transform.target == frame
        if path in result:
            raise ValueError(f"duplicate dataset path {path!r}")
        result[path] = NgffLevel(names, MappingProxyType(spec), level_transform)
    extra = tuple(
        _import_transform(t, systems, None, store, group, frames, cache, report)
        for t in model.coordinateTransformations or ()
    )
    for additional in extra:
        if additional.source != frame:
            raise ValueError("multiscale additional transforms must start at the intrinsic frame")
    return NgffMultiscale(frame, MappingProxyType(result), extra, tuple(report))


def from_scene(
    scene: SceneAttrs | JsonObject,
    *,
    systems: Mapping[str, Sequence[ct.CoordinateSystem | JsonObject]],
    store: str | None = None,
    group: str = "",
    frames: FrameMap | None = None,
) -> NgffScene:
    """Import scene frame-to-frame transforms using referenced image systems.

    Args:
        scene: v06 scene attributes or equivalent JSON attributes.
        systems: Coordinate systems of referenced images, keyed by image path.
        store: Optional resolved store URI.
        group: Relative scene group path.
        frames: Previously resolved local image frames.

    Returns:
        Scene transforms in metadata order and an import report.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an endpoint or transform cannot be represented.
    """
    store, group = _location(store, group)
    if not isinstance(systems, Mapping):
        raise TypeError("systems must map image paths to coordinate systems")
    if frames is not None and not isinstance(frames, Mapping):
        raise TypeError("frames must map (group path, name) to ReferenceFrame")
    if not isinstance(scene, SceneAttrs):
        _reject_rotation_path(scene)
    model = scene if isinstance(scene, SceneAttrs) else SceneAttrs.model_validate(scene)
    scope = {(group, cs.name): cs for cs in model.coordinateSystems or ()}
    for path, declarations in systems.items():
        location = _join(group, path)
        for declaration in declarations:
            cs = _system(declaration)
            scope[(location, cs.name)] = cs
    cache: dict[tuple[str, str], ReferenceFrame] = {}
    report: list[tuple[str, str]] = []
    transforms = tuple(
        _import_transform(t, scope, None, store, group, frames, cache, report)
        for t in model.coordinateTransformations
    )
    return NgffScene(transforms, tuple(report))


__all__ = [
    "NAMESPACE",
    "NgffLevel",
    "NgffMultiscale",
    "NgffScene",
    "coordinate_system",
    "from_multiscale",
    "from_scene",
    "open",
    "to_dataarray",
    "to_multiscale_level",
    "to_transform",
    "transform",
]
