"""Export core affine geometry as OME-Zarr 0.6 metadata."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import unquote

import numpy as np
import numpy.typing as npt
from ome_zarr_models.v06 import coordinate_transforms as ct
from ome_zarr_models.v06.multiscales import Dataset, Multiscale

from xarrayrf import AffineTransform, ArrayCoordinates, Geometry, ReferenceFrame
from xarrayrf.anatomy import VOCABULARY
from xarrayrf.native import Report
from xarrayrf.units import udunits_name

from ._metadata import NAMESPACE, _join, _location


def _name(frame: ReferenceFrame, names: Mapping[ReferenceFrame, str] | None) -> str:
    if names is not None and frame in names:
        name = names[frame]
        if not isinstance(name, str):
            raise TypeError(f"name for frame {frame.identifier!r} must be a string")
        if not name:
            raise ValueError(f"name for frame {frame.identifier!r} must be nonempty")
        return name
    namespace, value = frame.identifier
    if namespace == NAMESPACE and "#" in value:
        return unquote(value.rsplit("#", 1)[1])
    raise ValueError(f"frame {frame.identifier!r} needs a name in names for NGFF export")


def _system(frame: ReferenceFrame, name: str) -> ct.CoordinateSystem:
    system = frame.coordinate_system
    axes = [
        ct.Axis(  # type: ignore[call-arg]
            name=axis, unit=udunits_name(unit), type=kind, orientation=token
        )
        if system.vocabulary == VOCABULARY and token is not None and kind == "space"
        else ct.Axis(name=axis, unit=udunits_name(unit), type=kind)
        for axis, unit, kind, token in zip(
            frame.axes, frame.units, system.axis_types, system.orientation, strict=True
        )
    ]
    return ct.CoordinateSystem(name=name, axes=tuple(axes))


def _losses(frame: ReferenceFrame, name: str) -> list[tuple[str, str]]:
    report = [("frame-identity", f"frame {name!r}: identity {frame.identifier!r} is not retained")]
    for field in ("definition", "context", "role", "display"):
        if getattr(frame, field):
            report.append((field, f"frame {name!r}: {field} is not represented"))
    system = frame.coordinate_system
    if system.vocabulary is not None and system.vocabulary != VOCABULARY:
        report.append(("vocabulary", f"frame {name!r}: direction vocabulary is not represented"))
        if any(value is not None for value in system.orientation):
            report.append(("orientation", f"frame {name!r}: orientation is not represented"))
    elif system.vocabulary == VOCABULARY:
        # RFC-4 orients space axes only; an oriented axis of another type cannot be written.
        for axis, kind, token in zip(
            system.axes, system.axis_types, system.orientation, strict=True
        ):
            if token is not None and kind != "space":
                report.append(
                    (
                        "orientation",
                        f"frame {name!r}: orientation of non-space axis {axis!r} is not represented",
                    )
                )
    return report


def _affine_member(
    matrix: npt.NDArray[np.float64], translation: npt.NDArray[np.float64]
) -> ct.Affine:
    rows = np.column_stack((matrix, translation)).tolist()
    return ct.Affine(affine=tuple(tuple(float(value) for value in row) for row in rows))


def to_transform(
    t: AffineTransform, *, names: Mapping[ReferenceFrame, str] | None = None
) -> tuple[ct.Affine, Report]:
    """Export a frame-to-frame affine and report declarations v06 cannot retain.

    Args:
        t: Frame-to-frame affine transform.
        names: NGFF coordinate-system names for frames without an OME-Zarr identity.

    Raises:
        TypeError: If ``t`` is not an affine transform.
        ValueError: If either endpoint is not a reference frame or has no NGFF name.
    """
    if not isinstance(t, AffineTransform):
        raise TypeError(f"t must be an AffineTransform, got {type(t).__name__}")
    if not isinstance(t.source, ReferenceFrame) or not isinstance(t.target, ReferenceFrame):
        raise ValueError("to_transform requires ReferenceFrame source and target")
    if names is not None and not isinstance(names, Mapping):
        raise TypeError("names must map ReferenceFrame instances to NGFF names")
    source_name, target_name = _name(t.source, names), _name(t.target, names)
    result = ct.Affine(
        affine=_affine_member(t.matrix, t.translation).affine,
        input=ct.CoordinateSystemIdentifier(name=source_name),
        output=ct.CoordinateSystemIdentifier(name=target_name),
    )
    report = [*_losses(t.source, source_name), *_losses(t.target, target_name)]
    return result, tuple(report)


def to_multiscale_level(
    geometry: Geometry,
    *,
    path: str,
    name: str = "intrinsic",
    frame_name: str = "physical",
    store: str | None = None,
) -> tuple[Multiscale, Report]:
    """Export one regular affine level as validated OME-Zarr 0.6 multiscale metadata.

    Args:
        geometry: Array samples mapped into a reference frame.
        path: Relative path of the level array.
        name: Name for the intrinsic coordinate system.
        frame_name: Name for the physical coordinate system of a non-diagonal lattice.
        store: Optional resolved store URI, used to validate location syntax.

    Returns:
        A one-dataset multiscale model and a report of unrepresented declarations.

    Raises:
        TypeError: If ``geometry`` or a public argument has the wrong type.
        ValueError: If the path, name, geometry, or v06 multiscale is invalid.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError(f"geometry must be a Geometry, got {type(geometry).__name__}")
    _location(store, "")
    if not isinstance(path, str):
        raise TypeError(f"path must be a relative string, got {type(path).__name__}")
    _join("", path)
    if not isinstance(name, str):
        raise TypeError(f"name must be a string, got {type(name).__name__}")
    if not name:
        raise ValueError("name must be nonempty")
    if not isinstance(frame_name, str):
        raise TypeError(f"frame_name must be a string, got {type(frame_name).__name__}")
    if not frame_name:
        raise ValueError("frame_name must be nonempty")
    try:
        lattice = geometry.lattice()
    except (TypeError, ValueError) as error:
        raise ValueError(f"NGFF export requires a regular affine lattice: {error}") from error
    if len(lattice.dims) != len(geometry.array.dims):
        raise ValueError("NGFF level requires every array dimension to be a geometry dimension")
    frame = lattice.frame
    matrix, origin = lattice.matrix, lattice.origin
    if matrix.shape[1] != len(frame.axes):
        raise ValueError("NGFF level needs one array dimension per intrinsic axis")
    source = geometry.transform.source
    assert isinstance(source, ArrayCoordinates)
    diagonal = np.array_equal(matrix, np.diag(np.diag(matrix)))
    report = _losses(frame, name if diagonal else frame_name)
    for axis, offset in zip(source.axes, source.sample_offset, strict=True):
        if offset != 0.5:
            report.append(
                ("sample-offset", f"axis {axis!r}: sample offset {offset!r} is not represented")
            )
    for axis, unit in zip(source.axes, source.units, strict=True):
        if unit != "1":
            report.append(("array-unit", f"axis {axis!r}: array unit {unit!r} is not represented"))
    for axis, kind in zip(source.axes, source.axis_types, strict=True):
        if kind is not None:
            report.append(
                ("array-axis-type", f"axis {axis!r}: array axis type {kind!r} is not represented")
            )
    input_id = ct.CoordinateSystemIdentifier(path=path)
    output_id = ct.CoordinateSystemIdentifier(name=name)
    additional: tuple[ct.Affine, ...] | None = None
    systems: tuple[ct.CoordinateSystem, ...] = (_system(frame, name),)
    if diagonal:
        members: tuple[ct.AnyTransform, ...] = (
            ct.Scale(scale=tuple(float(value) for value in np.diag(matrix))),
            ct.Translation(translation=tuple(float(value) for value in origin)),
        )
    else:
        if frame_name == name:
            raise ValueError("frame_name must differ from name for a non-diagonal lattice")
        spacing = np.linalg.norm(matrix, axis=0)
        members = (
            ct.Scale(scale=tuple(float(value) for value in spacing)),
            ct.Translation(translation=(0.0,) * matrix.shape[1]),
        )
        systems = (
            ct.CoordinateSystem(
                name=name,
                axes=tuple(
                    ct.Axis(name=dim, unit=udunits_name(unit), type="space")
                    for dim, unit in zip(lattice.dims, frame.units, strict=True)
                ),
            ),
            _system(frame, frame_name),
        )
        additional = (
            ct.Affine(
                affine=_affine_member(matrix / spacing, origin).affine,
                input=ct.CoordinateSystemIdentifier(name=name),
                output=ct.CoordinateSystemIdentifier(name=frame_name),
            ),
        )
        report.append(
            (
                "intrinsic-synthesized",
                f"intrinsic system {name!r} uses array axes; frame axes are in {frame_name!r}",
            )
        )
    level = ct.Sequence(input=input_id, output=output_id, transformations=members)
    dataset = Dataset(path=path, coordinateTransformations=(level,))
    metadata = Multiscale(
        coordinateSystems=systems, datasets=(dataset,), coordinateTransformations=additional
    )
    return metadata, tuple(report)
