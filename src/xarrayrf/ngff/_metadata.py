"""Private OME-Zarr 0.6 metadata conversion helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast
from urllib.parse import quote

import numpy as np
import numpy.typing as npt
from ome_zarr_models.v06 import coordinate_transforms as ct
from pydantic import TypeAdapter

from xarrayrf import AffineTransform, ArrayCoordinates, CoordinateSystem, ReferenceFrame
from xarrayrf.anatomy import VOCABULARY
from xarrayrf.units import canonical

NAMESPACE = "ome-zarr"
"""Namespace for store-resolved OME-Zarr coordinate system identities."""

type ResolvedFrames = Mapping[tuple[str, str], ReferenceFrame]
type JsonObject = Mapping[str, object]

_TRANSFORM_ADAPTER: TypeAdapter[ct.AnyTransform] = TypeAdapter(ct.AnyTransform)


def _location(store: str | None, group: str) -> tuple[str | None, str]:
    if store is not None and not isinstance(store, str):
        raise TypeError("store must be a resolved URI string or None")
    if store is not None and (not store or store.endswith("/")):
        raise ValueError("store must be a resolved URI without a trailing '/' or None")
    if not isinstance(group, str):
        raise TypeError("group must be a path string")
    if (
        group.startswith("/")
        or group.endswith("/")
        or any(part in ("", ".", "..") for part in group.split("/") if group)
    ):
        raise ValueError("group must have no leading/trailing '/' or '.' or '..' segments")
    return store, group


def _join(group: str, path: str) -> str:
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or path.endswith("/")
        or any(part in ("", ".", "..") for part in path.split("/"))
    ):
        raise ValueError(f"path {path!r} must be relative with no empty, '.' or '..' segments")
    return "/".join(part for part in (group, path) if part)


def _system(value: ct.CoordinateSystem | JsonObject) -> ct.CoordinateSystem:
    if isinstance(value, ct.CoordinateSystem):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("coordinate system must be a v06 CoordinateSystem or JSON object")
    return ct.CoordinateSystem.model_validate(value)


def _checked_axes(system: ct.CoordinateSystem) -> CoordinateSystem:
    orientation: list[str | None] = []
    for axis in system.axes:
        if axis.name is None:
            raise ValueError(f"coordinate system {system.name!r} has an axis without a name")
        if axis.unit is not None and not isinstance(axis.unit, str):
            raise ValueError(f"axis {axis.name!r} has a non-string unit; supply a unit string")
        token = (axis.model_extra or {}).get("orientation")
        if token is not None:
            if axis.type != "space":
                raise ValueError(f"axis {axis.name!r}: orientation requires type 'space'")
            try:
                token = VOCABULARY.check(token)
            except ValueError as error:
                raise ValueError(
                    f"orientation {token!r} is not in vocabulary {VOCABULARY.identifier!r}"
                ) from error
        orientation.append(token)
    return CoordinateSystem(
        tuple(cast(str, axis.name) for axis in system.axes),
        tuple(canonical(cast(str | None, axis.unit)) for axis in system.axes),
        axis_types=tuple(axis.type for axis in system.axes),
        vocabulary=VOCABULARY,
        orientation=tuple(orientation),
    )


def _frame(
    system: ct.CoordinateSystem,
    store: str | None,
    group: str,
    resolved_frames: ResolvedFrames | None,
    cache: dict[tuple[str, str], ReferenceFrame],
    report: list[tuple[str, str]],
) -> ReferenceFrame:
    key = (group, system.name)
    coordinate_system = _checked_axes(system)
    if key not in cache:
        for axis in system.axes:
            if axis.discrete is not None:
                report.append(
                    ("discrete-deferred", f"axis {axis.name!r}: discrete is not retained")
                )
            if axis.longName is not None:
                report.append(
                    ("long-name-deferred", f"axis {axis.name!r}: longName is not retained")
                )
    existing = cache.get(key) or (resolved_frames.get(key) if resolved_frames is not None else None)
    if existing is not None:
        if not isinstance(existing, ReferenceFrame):
            raise TypeError(f"frame for {key!r} must be a ReferenceFrame")
        if store is not None:
            location = "/".join(part for part in (store, group) if part)
            expected_identifier = (NAMESPACE, f"{location}#{quote(system.name, safe='')}")
            if existing.identifier != expected_identifier:
                raise ValueError(f"frame for {key!r} has an identity inconsistent with store")
        if existing.coordinate_system != coordinate_system:
            kept = [
                i
                for i, axis in enumerate(system.axes)
                if not (axis.discrete is True or axis.type == "channel")
            ]
            if not kept or len(kept) == len(system.axes):
                raise ValueError(f"frame for {key!r} disagrees with coordinate system axes")
            geometry_system = CoordinateSystem(
                tuple(coordinate_system.axes[i] for i in kept),
                tuple(coordinate_system.units[i] for i in kept),
                axis_types=tuple(coordinate_system.axis_types[i] for i in kept),
                vocabulary=coordinate_system.vocabulary,
                orientation=tuple(coordinate_system.orientation[i] for i in kept),
            )
            if existing.coordinate_system != geometry_system:
                raise ValueError(f"frame for {key!r} disagrees with coordinate system axes")
            existing = existing.with_coordinate_system(coordinate_system)
        cache[key] = existing
        return existing
    if store is None:
        result = ReferenceFrame.anonymous(coordinate_system)
    else:
        location = "/".join(part for part in (store, group) if part)
        result = ReferenceFrame.declared(
            (NAMESPACE, f"{location}#{quote(system.name, safe='')}"), coordinate_system
        )
    cache[key] = result
    return result


def _transform(value: ct.Transform | JsonObject) -> ct.Transform:
    if isinstance(value, ct.Transform):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("transform must be a v06 transform or JSON object")
    _reject_rotation_path(value)
    return _TRANSFORM_ADAPTER.validate_python(value)


def _reject_rotation_path(value: object) -> None:
    # ome-zarr-models calls rotation_matrix during validation and raises
    # NotImplementedError for path-only rotations before returning a model.
    if isinstance(value, Mapping):
        if value.get("type") == "rotation" and value.get("path") is not None:
            raise ValueError("rotation array-backed path is unsupported; supply an inline rotation")
        for key in (
            "coordinateTransformations",
            "datasets",
            "transformations",
            "transformation",
            "forward",
            "inverse",
        ):
            if key in value:
                _reject_rotation_path(value[key])
    elif isinstance(value, list | tuple):
        for child in value:
            _reject_rotation_path(child)


def _output_count(t: ct.Transform, inputs: int) -> int:
    if isinstance(t, ct.Affine):
        return len(t.affine or ())
    if isinstance(t, ct.Rotation):
        return len(t.rotation or ())
    if isinstance(t, ct.MapAxis):
        return len(t.mapAxis)
    if isinstance(t, ct.ProjectAxis):
        return inputs - len(t.droppedInputs or ()) + len(t.createdOutputs or ())
    if isinstance(t, ct.ByDimension):
        return max((i for child in t.transformations for i in child.outputAxes), default=-1) + 1
    if isinstance(t, ct.Bijection):
        return _output_count(t.forward, inputs)
    if isinstance(t, ct.Sequence):
        count = inputs
        for child in t.transformations:
            count = _output_count(child, count)
        return count
    return inputs


def _report_declared_inverse(t: ct.Transform, report: list[tuple[str, str]]) -> None:
    if isinstance(t, ct.Bijection):
        report.append(
            ("declared-inverse-deferred", "bijection inverse was checked but not retained")
        )
        _report_declared_inverse(t.forward, report)
        _report_declared_inverse(t.inverse, report)
    elif isinstance(t, ct.Sequence):
        for child in t.transformations:
            _report_declared_inverse(child, report)
    elif isinstance(t, ct.ByDimension):
        for member in t.transformations:
            _report_declared_inverse(member.transformation, report)


def _affine(
    t: ct.Transform, inputs: int, outputs: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    kind = t.type
    if isinstance(t, ct.Identity):
        if inputs != outputs:
            raise ValueError("identity requires equal input and output axis counts")
        return np.eye(inputs), np.zeros(outputs)
    if isinstance(t, ct.Scale):
        if len(t.scale) != inputs or inputs != outputs:
            raise ValueError("scale needs one factor per input and output axis")
        return np.diag(t.scale), np.zeros(outputs)
    if isinstance(t, ct.Translation):
        if len(t.translation) != outputs or inputs != outputs:
            raise ValueError("translation needs one value per input and output axis")
        return np.eye(inputs), np.asarray(t.translation, dtype=np.float64)
    if isinstance(t, ct.Affine):
        if t.path is not None:
            raise ValueError("affine array-backed path is unsupported; supply an inline affine")
        matrix = np.asarray(t.affine, dtype=np.float64)
        if matrix.shape != (outputs, inputs + 1):
            raise ValueError(f"affine must have shape {(outputs, inputs + 1)}, got {matrix.shape}")
        return matrix[:, :-1], matrix[:, -1]
    if isinstance(t, ct.Rotation):
        if t.path is not None:
            raise ValueError("rotation array-backed path is unsupported; supply an inline rotation")
        matrix = np.asarray(t.rotation, dtype=np.float64)
        if matrix.shape != (outputs, inputs):
            raise ValueError(f"rotation must have shape {(outputs, inputs)}, got {matrix.shape}")
        return matrix, np.zeros(outputs)
    if isinstance(t, ct.MapAxis):
        if len(t.mapAxis) != outputs or any(i < 0 or i >= inputs for i in t.mapAxis):
            raise ValueError("mapAxis indices must name the input axes")
        matrix = np.zeros((outputs, inputs))
        matrix[np.arange(outputs), t.mapAxis] = 1
        return matrix, np.zeros(outputs)
    if isinstance(t, ct.ProjectAxis):
        dropped = set(t.droppedInputs or ())
        created = set(t.createdOutputs or ())
        if any(i < 0 or i >= inputs for i in dropped) or any(
            i < 0 or i >= outputs for i in created
        ):
            raise ValueError("projectAxis indices must be within input and output axes")
        kept = [i for i in range(inputs) if i not in dropped]
        target = [i for i in range(outputs) if i not in created]
        if len(kept) != len(target):
            raise ValueError("projectAxis retained input and output axis counts must agree")
        matrix = np.zeros((outputs, inputs))
        matrix[target, kept] = 1
        return matrix, np.zeros(outputs)
    if isinstance(t, ct.Sequence):
        matrix = np.eye(inputs)
        translation = np.zeros(inputs)
        count = inputs
        for child in t.transformations:
            target_count = _output_count(child, count)
            child_matrix, child_translation = _affine(child, count, target_count)
            matrix = child_matrix @ matrix
            translation = child_matrix @ translation + child_translation
            count = target_count
        if count != outputs:
            raise ValueError("sequence output axis count disagrees with its endpoint")
        return matrix, translation
    if isinstance(t, ct.ByDimension):
        matrix = np.zeros((outputs, inputs))
        translation = np.zeros(outputs)
        covered: set[int] = set()
        for member in t.transformations:
            ia, oa = member.inputAxes, member.outputAxes
            if any(i < 0 or i >= inputs for i in ia) or any(i < 0 or i >= outputs for i in oa):
                raise ValueError("byDimension inputAxes/outputAxes indices are out of range")
            if len(set(oa)) != len(oa) or covered.intersection(oa):
                raise ValueError("byDimension must cover every output axis exactly once")
            block, offset = _affine(member.transformation, len(ia), len(oa))
            matrix[np.ix_(oa, ia)] = block
            translation[list(oa)] = offset
            covered.update(oa)
        if covered != set(range(outputs)):
            raise ValueError("byDimension must cover every output axis exactly once")
        return matrix, translation
    if isinstance(t, ct.Bijection):
        matrix, translation = _affine(t.forward, inputs, outputs)
        back, back_translation = _affine(t.inverse, outputs, inputs)
        if (
            inputs != outputs
            or not np.allclose(back @ matrix, np.eye(inputs), rtol=1e-12, atol=1e-12)
            or not np.allclose(
                back @ translation + back_translation, np.zeros(inputs), rtol=1e-12, atol=1e-12
            )
        ):
            raise ValueError("bijection inverse disagrees with the forward affine within 1e-12")
        return matrix, translation
    if isinstance(t, ct.Displacements | ct.Coordinates):
        raise ValueError(f"{kind} is nonlinear and requires an external transform engine")
    raise ValueError(f"unsupported NGFF transform {kind!r}")


def _resolve(
    endpoint: ct.CoordinateSystemIdentifier | None,
    systems: Mapping[tuple[str, str], ct.CoordinateSystem],
    dims: Mapping[str, Sequence[str]] | None,
    store: str | None,
    group: str,
    resolved_frames: ResolvedFrames | None,
    cache: dict[tuple[str, str], ReferenceFrame],
    report: list[tuple[str, str]],
) -> tuple[ReferenceFrame | ArrayCoordinates, tuple[bool, ...]]:
    if endpoint is None:
        raise ValueError("transform requires input and output endpoints")
    if endpoint.name is None:
        if endpoint.path is None:
            raise ValueError("transform endpoint requires a name or array path")
        path = _join(group, endpoint.path)
        names = dims.get(endpoint.path) if dims is not None else None
        if names is None:
            raise ValueError(f"array path {endpoint.path!r} needs dims")
        if isinstance(names, str | bytes) or not isinstance(names, Sequence):
            raise TypeError(f"array path {endpoint.path!r} dims must be a sequence of names")
        axes = tuple(names)
        return ArrayCoordinates(axes, ("1",) * len(axes), sample_offset=(0.5,) * len(axes)), (
            False,
        ) * len(axes)
    path = _join(group, endpoint.path) if endpoint.path is not None else group
    system = systems.get((path, endpoint.name))
    if system is None:
        raise ValueError(f"unresolved coordinate system {endpoint.name!r} at path {path!r}")
    frame = _frame(system, store, path, resolved_frames, cache, report)
    return frame, tuple(axis.discrete is True or axis.type == "channel" for axis in system.axes)


def _import_transform(
    t: ct.Transform,
    systems: Mapping[tuple[str, str], ct.CoordinateSystem],
    dims: Mapping[str, Sequence[str]] | None,
    store: str | None,
    group: str,
    resolved_frames: ResolvedFrames | None,
    cache: dict[tuple[str, str], ReferenceFrame],
    report: list[tuple[str, str]],
) -> AffineTransform:
    source, source_discrete = _resolve(
        t.input, systems, dims, store, group, resolved_frames, cache, report
    )
    target, target_discrete = _resolve(
        t.output, systems, dims, store, group, resolved_frames, cache, report
    )
    if isinstance(source, ArrayCoordinates) and len(source_discrete) == len(target_discrete):
        source_discrete = target_discrete
        source = ArrayCoordinates(
            source.axes,
            source.units,
            axis_types=source.axis_types,
            sample_offset=tuple(None if flag else 0.5 for flag in source_discrete),
        )
    if isinstance(target, ArrayCoordinates) and len(target_discrete) == len(source_discrete):
        target_discrete = source_discrete
        target = ArrayCoordinates(
            target.axes,
            target.units,
            axis_types=target.axis_types,
            sample_offset=tuple(None if flag else 0.5 for flag in target_discrete),
        )
    matrix, translation = _affine(t, len(source.axes), len(target.axes))
    _report_declared_inverse(t, report)
    source_keep = [i for i, flag in enumerate(source_discrete) if not flag]
    target_keep = [i for i, flag in enumerate(target_discrete) if not flag]
    if sum(source_discrete) != sum(target_discrete):
        raise ValueError("discrete axes must be identity-mapped on both endpoints")
    if not source_keep or not target_keep:
        raise ValueError("transform has no continuous geometry axes after dropping discrete axes")
    for i, flag in enumerate(source_discrete):
        if flag:
            matching = [
                j
                for j, other in enumerate(target_discrete)
                if other
                and matrix[j, i] == 1
                and translation[j] == 0
                and np.count_nonzero(matrix[j]) == 1
                and np.count_nonzero(matrix[:, i]) == 1
            ]
            if len(matching) != 1:
                raise ValueError(f"discrete axis {source.axes[i]!r} mixes with other axes")
            report.append(("discrete-axis-dropped", f"identity-mapped axis {source.axes[i]!r}"))
    if len(source_keep) != len(source.axes):
        matrix = matrix[np.ix_(target_keep, source_keep)]
        translation = translation[target_keep]
        source = _subset(source, source_keep)
        target = _subset(target, target_keep)
    return AffineTransform(source=source, target=target, matrix=matrix, translation=translation)


def _subset(
    endpoint: ReferenceFrame | ArrayCoordinates, indices: list[int]
) -> ReferenceFrame | ArrayCoordinates:
    axes = tuple(endpoint.axes[i] for i in indices)
    units = tuple(endpoint.units[i] for i in indices)
    kinds = tuple(
        endpoint.coordinate_system.axis_types[i]
        if isinstance(endpoint, ReferenceFrame)
        else endpoint.axis_types[i]
        for i in indices
    )
    if isinstance(endpoint, ReferenceFrame):
        system = endpoint.coordinate_system
        return endpoint.with_coordinate_system(
            CoordinateSystem(
                axes,
                units,
                axis_types=kinds,
                vocabulary=system.vocabulary,
                orientation=tuple(system.orientation[i] for i in indices),
            )
        )
    return ArrayCoordinates(
        axes,
        units,
        axis_types=kinds,
        sample_offset=tuple(endpoint.sample_offset[i] for i in indices),
    )
