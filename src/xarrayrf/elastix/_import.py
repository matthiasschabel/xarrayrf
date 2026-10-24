"""elastix parameter maps to core transforms between the caller's frames."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from xarrayrf._affine import AffineTransform
from xarrayrf._array_coordinates import ArrayCoordinates
from xarrayrf._coordinate_system import CoordinateSystem
from xarrayrf._field_transform import BSplineTransform
from xarrayrf._frame import ReferenceFrame
from xarrayrf._grid import Grid
from xarrayrf._transform import SupportsPoints
from xarrayrf._transform_import import check_space, joined, stage_frames
from xarrayrf.anatomy import LPS, patient_coordinate_system

from ._parameter_files import NO_INITIAL_TRANSFORM

type Report = tuple[tuple[str, str], ...]

_LINEAR = ("TranslationTransform", "EulerTransform", "AffineTransform")
_SPLINES = ("BSplineTransform", "RecursiveBSplineTransform")
_INDEX_AXES = ("i", "j", "k", "l", "m", "n")


@dataclass(frozen=True, slots=True)
class ElastixImport:
    """An imported elastix result.

    Attributes:
        transform: Fixed-frame points to moving-frame points, as elastix's result maps them;
            this is the transform that resamples the moving image onto the fixed grid. One
            :class:`~xarrayrf.AffineTransform` when every stage is linear, the B-spline itself for a
            single B-spline stage, else a :class:`~xarrayrf.CompositeTransform` keeping each
            stage.
        report: ``(code, message)`` for every normalization made.
    """

    transform: SupportsPoints
    report: Report


def _stage_maps(maps: object) -> list[Mapping[str, Sequence[str]]]:
    """Maps in application order, from a sequence or an itk-elastix ``ParameterObject``."""
    if hasattr(maps, "GetNumberOfParameterMaps") and hasattr(maps, "GetParameterMap"):
        source: Any = maps
        return [source.GetParameterMap(i) for i in range(int(source.GetNumberOfParameterMaps()))]
    if isinstance(maps, Mapping | str | bytes):
        raise TypeError(
            "maps must be a sequence of parameter maps in application order, or a "
            "ParameterObject; wrap a single map in a list"
        )
    if not isinstance(maps, Sequence):
        raise TypeError(f"maps must be a sequence of parameter maps, got {type(maps).__name__}")
    for position, entry in enumerate(maps):
        if not isinstance(entry, Mapping):
            raise TypeError(f"stage {position} is a {type(entry).__name__}, not a parameter map")
    return list(maps)


class _Stage:
    """One parameter map's values, read with the stage named in every message."""

    def __init__(self, entry: Mapping[str, Sequence[str]], position: int) -> None:
        self.entry = entry
        self.position = position

    def fail(self, message: str) -> ValueError:
        return ValueError(f"stage {self.position}: {message}")

    def words(self, key: str, default: tuple[str, ...] | None = None) -> tuple[str, ...]:
        value = self.entry.get(key)
        if value is None:
            if default is None:
                raise self.fail(f"{key} is missing")
            return default
        if isinstance(value, str):
            return (value,)
        return tuple(str(v) for v in value)

    def word(self, key: str, default: str | None = None) -> str:
        values = self.words(key, None if default is None else (default,))
        if len(values) != 1:
            raise self.fail(f"{key} must hold one value, got {values!r}")
        return values[0]

    def flag(self, key: str, default: str) -> bool:
        value = self.word(key, default)
        if value not in ("true", "false"):
            raise self.fail(f"{key} must be 'true' or 'false', got {value!r}")
        return value == "true"

    def numbers(self, key: str, count: int | None = None) -> npt.NDArray[np.float64]:
        words = self.words(key)
        try:
            values = np.array([float(word) for word in words], dtype=np.float64)
        except ValueError:
            raise self.fail(f"{key} must hold numbers, got {words!r}") from None
        if count is not None and values.shape != (count,):
            raise self.fail(f"{key} must hold {count} values, got {values.size}")
        if not np.all(np.isfinite(values)):
            raise self.fail(f"{key} must be finite")
        return values

    def integers(self, key: str, count: int) -> tuple[int, ...]:
        values = self.numbers(key, count)
        if np.any(values != np.round(values)):
            raise self.fail(f"{key} must hold integers, got {self.words(key)!r}")
        return tuple(int(v) for v in values)

    def integer(self, key: str) -> int:
        return self.integers(key, 1)[0]


def _rotation(angle: float) -> npt.NDArray[np.float64]:
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, -sin], [sin, cos]])


def _euler_matrix(stage: _Stage, angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if angles.size == 1:
        return _rotation(float(angles[0]))
    cx, cy, cz = (math.cos(float(angle)) for angle in angles)
    sx, sy, sz = (math.sin(float(angle)) for angle in angles)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    # elastix's Euler3DTransform: Z·Y·X when ComputeZYX is true, else Z·X·Y (ITK's default).
    order = (rz, ry, rx) if stage.flag("ComputeZYX", "false") else (rz, rx, ry)
    rotation: npt.NDArray[np.float64] = np.linalg.multi_dot(order)
    return rotation


def _linear(
    stage: _Stage, name: str, dimension: int, source: ReferenceFrame, target: ReferenceFrame
) -> AffineTransform:
    if name == "TranslationTransform":
        matrix, offset = np.eye(dimension), stage.numbers("TransformParameters", dimension)
        center = np.zeros(dimension)
    else:
        if name == "EulerTransform":
            if dimension not in (2, 3):
                raise stage.fail(f"EulerTransform is defined in 2 or 3 dimensions, not {dimension}")
            angles = 1 if dimension == 2 else 3
            values = stage.numbers("TransformParameters", angles + dimension)
            matrix = _euler_matrix(stage, values[:angles])
            offset = values[angles:]
        else:
            values = stage.numbers("TransformParameters", dimension * dimension + dimension)
            matrix = values[: dimension * dimension].reshape(dimension, dimension)
            offset = values[dimension * dimension :]
        center = stage.numbers("CenterOfRotationPoint", dimension)
    # y = A (x - c) + c + t
    return AffineTransform.from_matrix(
        source=source, target=target, matrix=matrix, translation=center + offset - matrix @ center
    )


def _bspline(
    stage: _Stage, name: str, dimension: int, source: ReferenceFrame, target: ReferenceFrame
) -> BSplineTransform:
    order = (
        stage.integer("BSplineTransformSplineOrder")
        if "BSplineTransformSplineOrder" in (stage.entry)
        else 3
    )
    if order != 3:
        raise NotImplementedError(
            f"stage {stage.position}: BSplineTransformSplineOrder {order} is not implemented; "
            "the core B-spline is cubic"
        )
    if stage.flag("UseCyclicTransform", "false"):
        raise stage.fail("UseCyclicTransform must be false; cyclic B-splines are not implemented")
    if dimension > len(_INDEX_AXES):
        raise stage.fail(f"B-splines in {dimension} dimensions are not supported")
    size = stage.integers("GridSize", dimension)
    index = np.asarray(stage.integers("GridIndex", dimension), dtype=np.float64)
    if name == "RecursiveBSplineTransform" and np.any(index != 0):
        # Measured with itk-elastix 0.25.4: its support follows GridIndex but its values are
        # those of a zero index, which no single control grid reproduces.
        raise NotImplementedError(
            f"stage {stage.position}: a RecursiveBSplineTransform with GridIndex "
            f"{tuple(int(i) for i in index)} is not supported: elastix evaluates it inconsistently "
            "(support from GridIndex, values as if it were zero); elastix itself writes zero"
        )
    spacing = stage.numbers("GridSpacing", dimension)
    origin = stage.numbers("GridOrigin", dimension)
    direction = stage.numbers("GridDirection", dimension * dimension).reshape(
        dimension, dimension, order="F"
    )
    if min(size) < 4:
        raise stage.fail(f"GridSize must be at least 4 per axis, got {size}")
    if np.any(spacing <= 0):
        raise stage.fail(f"GridSpacing must be positive, got {spacing.tolist()}")
    parameters = stage.numbers("TransformParameters", dimension * math.prod(size))
    # Component-major, each component's control grid with the first index fastest.
    blocks = parameters.reshape(dimension, *reversed(size))
    coefficients = np.moveaxis(blocks.transpose(0, *range(dimension, 0, -1)), 0, -1)
    axes = _INDEX_AXES[:dimension]
    scaled = direction * spacing
    grid = Grid(
        AffineTransform.from_matrix(
            source=ArrayCoordinates(axes, ("1",) * dimension),
            target=source,
            matrix=scaled,
            translation=origin + scaled @ index,
        ),
        {axis: (axis, np.arange(n)) for axis, n in zip(axes, size, strict=True)},
    )
    return BSplineTransform(
        source=source,
        target=target,
        coefficients=coefficients,
        grid=grid,
        outside="identity",
        support="half-open",
    )


def transform_from_parameter_maps(
    maps: Sequence[Mapping[str, Sequence[str]]] | object,
    *,
    fixed: ReferenceFrame,
    moving: ReferenceFrame,
    space: CoordinateSystem | None = None,
    intermediate_frames: Sequence[ReferenceFrame] | None = None,
) -> ElastixImport:
    """Import an elastix result: its stages, from the fixed frame to the moving frame.

    Supported stages are ``TranslationTransform``, ``EulerTransform`` (2-D and 3-D, with
    ``ComputeZYX``), ``AffineTransform`` and cubic ``BSplineTransform`` or
    ``RecursiveBSplineTransform``, combined by ``Compose``. B-spline stages declare elastix's
    support, ``[1, n - 2)`` in control index, and its identity outside it.

    Args:
        maps: Parameter maps in application order (the initial transform first), as
            itk-elastix's ``ParameterObject.GetParameterMap`` or :func:`read_parameter_files`
            return them, or a ``ParameterObject``.
        fixed: The frame of the fixed image; the result maps from it.
        moving: The frame of the moving image; the result maps to it.
        space: The coordinate system elastix's physical points were in. Defaults to the LPS
            patient system in millimetres for three dimensions, since ITK reads medical images
            into LPS; required otherwise.
        intermediate_frames: One frame per boundary between stages, each with coordinate system
            ``space``. Defaults to a fresh local frame per boundary.

    Returns:
        The transform and a report of every normalization.

    Raises:
        TypeError: If ``maps`` or a frame has the wrong type.
        ValueError: If there are no maps, ``fixed`` and ``moving`` are the same frame, ``space``
            is missing or has the wrong dimension, a frame's coordinate system cannot be
            converted to ``space``, or a stage is malformed or uses an unsupported mode.
        NotImplementedError: If a stage's transform or spline order is not supported.
    """
    stages = _stage_maps(maps)
    if not stages:
        raise ValueError("maps holds no stages")
    first = _Stage(stages[0], 0)
    initial = first.word("InitialTransformParameterFileName", NO_INITIAL_TRANSFORM)
    if initial != NO_INITIAL_TRANSFORM:
        raise ValueError(
            f"stage 0 names the initial transform {initial!r}, which is not among the maps, so "
            "the chain would be incomplete; read files with read_parameter_files, which follows "
            "it, or pass every stage in application order"
        )
    dimension = first.integer("FixedImageDimension")
    if space is None:
        if dimension != 3:
            raise ValueError(
                f"space is required for {dimension}-D maps: name the coordinate system elastix's "
                "physical points were in"
            )
        space = patient_coordinate_system(LPS, "mm")
    space = check_space(space, dimension, engine="elastix")
    frames = stage_frames(
        len(stages),
        fixed=fixed,
        moving=moving,
        space=space,
        intermediate_frames=intermediate_frames,
    )
    report: list[tuple[str, str]] = []
    members: list[SupportsPoints] = []
    splines = False
    for position, entry in enumerate(stages):
        stage = _Stage(entry, position)
        name = stage.word("Transform")
        for key in ("FixedImageDimension", "MovingImageDimension"):
            if stage.integer(key) != dimension:
                raise stage.fail(f"{key} is {stage.word(key)}, but stage 0 is {dimension}-D")
        if stage.word("HowToCombineTransforms", "Compose") != "Compose":
            raise stage.fail("HowToCombineTransforms must be 'Compose'; 'Add' is not supported")
        if not stage.flag("UseDirectionCosines", "true"):
            raise stage.fail(
                "UseDirectionCosines must be true: without it elastix ignores image directions, "
                "so its physical space is not the images'"
            )
        if "NumberOfParameters" in entry:
            count = stage.integer("NumberOfParameters")
            if count != len(stage.words("TransformParameters")):
                raise stage.fail(
                    f"NumberOfParameters is {count} but TransformParameters holds "
                    f"{len(stage.words('TransformParameters'))} values"
                )
        source, target = frames[position], frames[position + 1]
        member: SupportsPoints
        if name in _LINEAR:
            member = _linear(stage, name, dimension, source, target)
        elif name in _SPLINES:
            member = _bspline(stage, name, dimension, source, target)
            splines = True
        else:
            raise NotImplementedError(
                f"stage {position}: elastix transform {name!r} is not supported; supported are "
                f"{', '.join(_LINEAR + _SPLINES)}"
            )
        members.append(member)
        report.append(
            ("stage", f"stage {position}: {name} from {source.identifier} to {target.identifier}")
        )
    transform, changes = joined(members, fixed=fixed, moving=moving, space=space, engine="elastix")
    report.extend(changes)
    if splines:
        report.append(
            (
                "bspline-support",
                "B-spline stages declare elastix's support, control index [1, n - 2), and the "
                "identity outside it",
            )
        )
    return ElastixImport(transform=transform, report=tuple(report))
