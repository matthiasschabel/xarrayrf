"""ITK transform files to core transforms between the caller's frames."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from xarrayrf._affine import AffineTransform
from xarrayrf._array_coordinates import ArrayCoordinates
from xarrayrf._coordinate_system import CoordinateSystem
from xarrayrf._field_transform import DisplacementFieldTransform
from xarrayrf._frame import ReferenceFrame
from xarrayrf._grid import Grid
from xarrayrf._transform import SupportsPoints
from xarrayrf._transform_import import check_space, joined, stage_frames
from xarrayrf.anatomy import LPS, patient_coordinate_system

from ._files import FieldStage, LinearStage, Stage, read_stages

type Report = tuple[tuple[str, str], ...]

_INDEX_AXES = ("i", "j", "k")

SUPPORTED_TYPES = (
    "AffineTransform",
    "MatrixOffsetTransformBase",
    "TranslationTransform",
    "IdentityTransform",
    "Euler2DTransform",
    "Rigid2DTransform",
    "Euler3DTransform",
    "VersorRigid3DTransform",
    "Similarity3DTransform",
    "DisplacementFieldTransform",
)
"""ITK transform types this module reads."""


@dataclass(frozen=True, slots=True)
class ItkImport:
    """An imported ITK transform file.

    Attributes:
        transform: Fixed-frame points to moving-frame points, as ITK's transform maps them (the
            transform that resamples the moving image onto the fixed grid). One
            :class:`~xarrayrf.AffineTransform` when every member is linear, the field itself for
            a single displacement field, else a :class:`~xarrayrf.CompositeTransform`.
        report: ``(code, message)`` for every normalization made.
    """

    transform: SupportsPoints
    report: Report


def _rotation_2d(angle: float) -> npt.NDArray[np.float64]:
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, -sin], [sin, cos]])


def _euler_3d(angles: npt.NDArray[np.float64], zyx: bool) -> npt.NDArray[np.float64]:
    cx, cy, cz = (math.cos(float(a)) for a in angles)
    sx, sy, sz = (math.sin(float(a)) for a in angles)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    # ITK's Euler3DTransform: Z·Y·X when ComputeZYX, else Z·X·Y.
    rotation: npt.NDArray[np.float64] = np.linalg.multi_dot((rz, ry, rx) if zyx else (rz, rx, ry))
    return rotation


def _versor(vector: npt.NDArray[np.float64], *, where: str) -> npt.NDArray[np.float64]:
    """The rotation of the unit quaternion with vector part ``vector`` (ITK's versor)."""
    norm = float(vector @ vector)
    if norm > 1.0:
        raise ValueError(f"{where}: a versor's vector part must have length at most 1")
    x, y, z = (float(v) for v in vector)
    w = math.sqrt(1.0 - norm)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _expect(stage: LinearStage, parameters: int, fixed: Sequence[int], *, where: str) -> None:
    if stage.parameters.size != parameters:
        raise ValueError(
            f"{where}: {stage.name} in {stage.dimension}-D has {parameters} parameters, got "
            f"{stage.parameters.size}"
        )
    if stage.fixed_parameters.size not in fixed:
        raise ValueError(
            f"{where}: {stage.name} has {' or '.join(map(str, fixed))} fixed parameters, got "
            f"{stage.fixed_parameters.size}"
        )


def _linear(
    stage: LinearStage, source: ReferenceFrame, target: ReferenceFrame, *, where: str
) -> AffineTransform:
    """``y = A (x - c) + c + t`` for each supported parametrization."""
    name, dimension, values = stage.name, stage.dimension, stage.parameters
    center = stage.fixed_parameters[:dimension]
    if name in ("AffineTransform", "MatrixOffsetTransformBase"):
        _expect(stage, dimension * dimension + dimension, (dimension,), where=where)
        matrix = values[: dimension * dimension].reshape(dimension, dimension)
        offset = values[dimension * dimension :]
    elif name == "TranslationTransform":
        _expect(stage, dimension, (0,), where=where)
        matrix, offset, center = np.eye(dimension), values, np.zeros(dimension)
    elif name == "IdentityTransform":
        _expect(stage, 0, (0,), where=where)
        matrix, offset, center = np.eye(dimension), np.zeros(dimension), np.zeros(dimension)
    elif name in ("Euler2DTransform", "Rigid2DTransform") and dimension == 2:
        _expect(stage, 3, (2,), where=where)
        matrix, offset = _rotation_2d(float(values[0])), values[1:]
    elif name == "Euler3DTransform" and dimension == 3:
        _expect(stage, 6, (3, 4), where=where)
        flags = stage.fixed_parameters[3:]
        if flags.size and flags[0] not in (0.0, 1.0):
            raise ValueError(f"{where}: the ComputeZYX flag must be 0 or 1, got {flags[0]:g}")
        matrix, offset = _euler_3d(values[:3], bool(flags.size and flags[0])), values[3:]
    elif name == "VersorRigid3DTransform" and dimension == 3:
        _expect(stage, 6, (3,), where=where)
        matrix, offset = _versor(values[:3], where=where), values[3:]
    elif name == "Similarity3DTransform" and dimension == 3:
        _expect(stage, 7, (3,), where=where)
        if values[6] <= 0:
            raise ValueError(f"{where}: the similarity scale must be positive, got {values[6]:g}")
        matrix, offset = values[6] * _versor(values[:3], where=where), values[3:6]
    else:
        raise NotImplementedError(
            f"{where}: ITK transform {name} in {dimension}-D is not supported; supported are "
            f"{', '.join(SUPPORTED_TYPES)}"
        )
    return AffineTransform.from_matrix(
        source=source, target=target, matrix=matrix, translation=center + offset - matrix @ center
    )


def _field(
    stage: FieldStage, source: ReferenceFrame, target: ReferenceFrame, *, where: str
) -> DisplacementFieldTransform:
    dimension = stage.dimension
    if np.any(stage.spacing <= 0):
        raise ValueError(
            f"{where}: the field spacing must be positive, got {stage.spacing.tolist()}"
        )
    axes = _INDEX_AXES[:dimension]
    grid = Grid(
        AffineTransform.from_matrix(
            source=ArrayCoordinates(axes, ("1",) * dimension, sample_offset=(0.5,) * dimension),
            target=source,
            matrix=stage.direction * stage.spacing,
            translation=stage.origin,
        ),
        {axis: (axis, np.arange(n)) for axis, n in zip(axes, stage.vectors.shape, strict=False)},
    )
    return DisplacementFieldTransform(
        source=source, target=target, field=stage.vectors, grid=grid, outside="identity"
    )


def read_transform(
    path: str | os.PathLike[str],
    *,
    fixed: ReferenceFrame,
    moving: ReferenceFrame,
    space: CoordinateSystem | None = None,
    intermediate_frames: Sequence[ReferenceFrame] | None = None,
) -> ItkImport:
    """Import an ITK transform file, from the fixed frame to the moving frame.

    Reads ``.tfm``/``.txt`` (Insight Transform File), ``.mat`` (MATLAB, as ITK and ANTs write
    affines), ``.h5`` (ITK HDF5, including composites) and ``.nii``/``.nii.gz`` displacement
    fields, in float64. A composite applies its last-added member first, as ITK does.
    Displacement fields are the identity outside their cells, as in ITK.

    Args:
        path: The file.
        fixed: The frame the transform maps from (the fixed image's).
        moving: The frame it maps to (the moving image's).
        space: The coordinate system of ITK's physical points. Defaults to the LPS patient system
            in millimetres for 3-D files; required for 2-D ones.
        intermediate_frames: One frame per boundary between a composite's members, each with
            coordinate system ``space``. Defaults to a fresh local frame per boundary.

    Returns:
        The transform and a report of every normalization.

    Raises:
        OSError: If the file cannot be read.
        TypeError: If a frame or ``space`` has the wrong type.
        ValueError: If the file is malformed, its geometry is ambiguous, ``space`` is missing or
            has the wrong dimension, the frames are the same or cannot be converted to ``space``.
        NotImplementedError: If a transform type is not supported.
    """
    file = Path(path)
    stages: list[Stage] = read_stages(file)
    dimensions = {stage.dimension for stage in stages}
    if len(dimensions) != 1:
        raise ValueError(f"{file}: its transforms mix dimensions {sorted(dimensions)}")
    (dimension,) = dimensions
    if space is None:
        if dimension != 3:
            raise ValueError(
                f"space is required for a {dimension}-D transform: name the coordinate system "
                "ITK's physical points were in"
            )
        space = patient_coordinate_system(LPS, "mm")
    space = check_space(space, dimension, engine="ITK")
    frames = stage_frames(
        len(stages),
        fixed=fixed,
        moving=moving,
        space=space,
        intermediate_frames=intermediate_frames,
    )
    members: list[SupportsPoints] = []
    report: list[tuple[str, str]] = []
    for position, stage in enumerate(stages):
        where = f"{file.name} stage {position}"
        source, target = frames[position], frames[position + 1]
        if isinstance(stage, FieldStage):
            members.append(_field(stage, source, target, where=where))
            report.extend(stage.notes)
            name = "DisplacementFieldTransform"
        else:
            members.append(_linear(stage, source, target, where=where))
            name = stage.name
        report.append(
            ("stage", f"stage {position}: {name} from {source.identifier} to {target.identifier}")
        )
    transform, changes = joined(members, fixed=fixed, moving=moving, space=space, engine="ITK")
    return ItkImport(transform=transform, report=(*report, *changes))
