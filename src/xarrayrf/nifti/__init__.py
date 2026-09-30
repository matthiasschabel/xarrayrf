"""Translate NIfTI header geometry without reading image data."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
import xarray as xr

from xarrayrf import (
    AffineTransform,
    ArrayCoordinates,
    CoordinateSystem,
    Geometry,
    Lattice,
    ReferenceFrame,
    affine_class,
    compose,
    coordinate_system_change,
)
from xarrayrf.anatomy import RAS, patient_coordinate_system
from xarrayrf.native import (
    CoordinateSpec,
    DuckArray,
    Report,
    _grid_and_coords,
    frame_array,
    index_coordinate,
)

from ._reader import open as open

_DIM_NAMES = ("i", "j", "k", "t", "u", "v", "w")


def _ras_system(spatial_unit: str, time_unit: str | None) -> CoordinateSystem:
    """The RAS patient system, with a fourth ``t`` axis when a time unit is given."""
    system = patient_coordinate_system(RAS, spatial_unit)
    if time_unit is None:
        return system
    return CoordinateSystem(
        (*system.axes, "t"),
        (*system.units, time_unit),
        axis_types=("space", "space", "space", "time"),
        vocabulary=system.vocabulary,
        orientation=(*RAS, None),
    )


_SPATIAL_UNITS = {1: "m", 2: "mm", 3: "um"}
_TIME_UNITS = {8: "s", 16: "ms", 24: "us"}
_SPECTRAL_UNITS = {32: "Hz", 40: "ppm", 48: "rad/s"}
_XFORMS = frozenset({"best", "sform", "qform"})
_AFFINE_TOLERANCE = 1e-6
_QFORM_ORTHOGONALITY = 1e-6
_TEMPLATE_NAME = re.compile(r"[A-Za-z0-9]+\Z")


@dataclass(frozen=True)
class NiftiGeometry:
    """Header dimensions, array coordinates, affine, frame and import report.

    ``coords`` values are xarray compatible ``(dimension, values, attrs)`` declarations.
    The retained ``k`` coordinate of a 2-D header is scalar. ``report`` records selected
    xform metadata and every normalization or discrepancy detected during import.
    """

    dims: tuple[str, ...]
    coords: Mapping[str, CoordinateSpec]
    transform: AffineTransform
    frame: ReferenceFrame
    report: Report


def _select_xform(
    header: nib.Nifti1Header | nib.Nifti2Header, choice: str
) -> tuple[str, int, npt.NDArray[np.float64], int, int]:
    s_code = int(header["sform_code"])
    q_code = int(header["qform_code"])
    selected = choice
    if choice == "best":
        selected = "sform" if s_code > 0 else "qform"
    code = s_code if selected == "sform" else q_code
    if code <= 0:
        raise ValueError(
            f"{choice!r} requires a coded sform or qform; got sform_code={s_code}, "
            f"qform_code={q_code}"
        )
    affine = header.get_sform() if selected == "sform" else header.get_qform()  # type: ignore[no-untyped-call]
    assert affine is not None  # the selected code is positive
    return selected, code, np.asarray(affine, dtype=np.float64), s_code, q_code


def from_header(
    header: nib.Nifti1Header | nib.Nifti2Header,
    *,
    frame: ReferenceFrame | xr.DataArray | None = None,
    template: str | None = None,
    xform: Literal["best", "sform", "qform"] = "best",
    spatial_unit: str | None = "mm",
    time: bool = False,
) -> NiftiGeometry:
    """Translate a NIfTI-1 or NIfTI-2 header into array and frame geometry.

    Args:
        header: Header with a coded sform or qform; no image data is read.
        frame: Optional known identity or framed array, expressed in a derivable patient system.
        template: TemplateFlow/BIDS space label for a coded aligned xform (codes 2-5).
        xform: ``best`` prefers a coded sform, then a coded qform. Either can be
            selected explicitly when its code is positive.
        spatial_unit: Unit to use when the header's spatial unit is unknown. ``None``
            refuses an unknown unit.
        time: Include a fourth, unoriented time axis in the frame and transform.
            Otherwise time remains a non-geometry array coordinate.

    Returns:
        Header dimensions, coordinates, affine transform, target frame and report.

    Raises:
        TypeError: If a public argument has the wrong Python type.
        ValueError: If dimensions, xform, units, time axis or supplied frame cannot be
            represented by this adapter, or ``template`` is malformed, combined with ``frame``
            or ``time=True``, or the selected xform code is not 2-5.
    """
    if not isinstance(header, nib.Nifti1Header | nib.Nifti2Header):
        raise TypeError(f"header must be a NIfTI-1 or NIfTI-2 header, got {type(header).__name__}")
    if isinstance(frame, xr.DataArray):
        frame = frame.rf.reference_frame
    if frame is not None and not isinstance(frame, ReferenceFrame):
        raise TypeError(
            f"frame must be a ReferenceFrame, framed DataArray or None, got {type(frame).__name__}"
        )
    if template is not None and not isinstance(template, str):
        raise TypeError("template must be a BIDS space label or None")
    if template is not None and _TEMPLATE_NAME.fullmatch(template) is None:
        raise ValueError("template must be a nonempty alphanumeric BIDS space label")
    if template is not None and frame is not None:
        raise ValueError("template and frame cannot both be supplied")
    if not isinstance(xform, str) or xform not in _XFORMS:
        raise ValueError(f"xform must be 'best', 'sform' or 'qform', got {xform!r}")
    if spatial_unit is not None and not isinstance(spatial_unit, str):
        raise TypeError(
            f"spatial_unit must be a unit string or None, got {type(spatial_unit).__name__}"
        )
    if spatial_unit is not None and not spatial_unit.strip():
        raise ValueError(
            f"spatial_unit must be a nonempty unit string or None, got {spatial_unit!r}"
        )
    if not isinstance(time, bool):
        raise TypeError(f"time must be a bool, got {type(time).__name__}")

    shape = header.get_data_shape()  # type: ignore[no-untyped-call]
    ndim = int(header["dim"][0])
    if ndim < 2 or ndim > 7:
        raise ValueError(f"NIfTI header must have 2 to 7 dimensions, got dim[0]={ndim}")
    dims = _DIM_NAMES[:ndim]
    selected, code, affine, s_code, q_code = _select_xform(header, xform)
    if template is not None and code not in (2, 3, 4, 5):
        raise ValueError("template requires selected xform code 2-5")
    if template is not None and time:
        raise ValueError(
            "template names a spatial space; a time=True frame also carries this acquisition's "
            "clock, so pass frame= to share a spacetime frame"
        )
    qfac = float(header["pixdim"][0])
    unit_code = int(header["xyzt_units"])
    spatial_code = unit_code & 0x07
    temporal_code = unit_code & 0x38
    report: list[tuple[str, str]] = []
    if spatial_code == 0:
        if spatial_unit is None:
            raise ValueError("xyzt_units has unknown spatial unit; supply spatial_unit")
        unit = spatial_unit
        report.append(("unknown-spatial-unit", f"xyzt_units spatial code 0; assumed {unit!r}"))
    elif spatial_code in _SPATIAL_UNITS:
        unit = _SPATIAL_UNITS[spatial_code]
    else:
        raise ValueError(f"xyzt_units has unsupported spatial code {spatial_code}")
    temporal_unit = (_TIME_UNITS | _SPECTRAL_UNITS).get(temporal_code)
    if temporal_code in _SPECTRAL_UNITS:
        report.append(
            ("spectral-time-unit", f"xyzt_units code {temporal_code} means {temporal_unit}")
        )
    elif temporal_code and temporal_unit is None:
        raise ValueError(f"xyzt_units has unsupported temporal code {temporal_code}")
    if time and (ndim < 4 or temporal_code not in _TIME_UNITS):
        raise ValueError("time=True requires a fourth dimension and a time unit (s, ms or us)")
    if frame is not None and (len(frame.axes) == 4) != time:
        raise ValueError("supplied frame's time axis must agree with time=True or time=False")

    code_name = str(nib.nifti1.xform_codes.label[code])
    report.append(
        (
            "xform",
            f"selected {selected} ({code_name}, code {code}); sform_code={s_code}, "
            f"qform_code={q_code}, qfac={qfac:g}",
        )
    )
    if s_code > 0 and q_code > 0:
        sform = header.get_sform()  # type: ignore[no-untyped-call]
        qform = header.get_qform()  # type: ignore[no-untyped-call]
        assert sform is not None and qform is not None
        difference = float(np.max(np.abs(sform - qform)))
        voxel_size = float(np.max(np.linalg.norm(affine[:3, :3], axis=0)))
        if difference > _AFFINE_TOLERANCE * voxel_size:
            report.append(
                ("sform-qform-disagree", f"maximum affine difference {difference:g} {unit}")
            )
    spacing = np.asarray(header["pixdim"][1:4], dtype=np.float64)
    lengths = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(np.abs(lengths - spacing) > _AFFINE_TOLERANCE * np.maximum(lengths, spacing)):
        report.append(
            ("pixdim-spacing-mismatch", "pixdim[1:4] differs from selected xform spacing")
        )

    ras_system = _ras_system(unit, temporal_unit if time else None)
    if frame is None:
        local_definition: dict[str, str | int | float] = {
            "xform": selected,
            "xform_code": code,
            "xform_code_name": code_name,
            "sform_code": s_code,
            "qform_code": q_code,
            "qfac": qfac,
        }
        if template is not None:
            frame = ReferenceFrame.declared(
                ("templateflow", template), ras_system, definition={"space": template}
            )
        elif not time and code in (3, 4):
            space = "Talairach" if code == 3 else "MNI152"
            frame = ReferenceFrame.declared(
                ("nifti-template", space),
                ras_system,
                definition={"space": space, "variant": "unspecified"},
            )
        else:
            frame = ReferenceFrame.local(ras_system, definition=local_definition)
    ras_view = (
        frame if frame.coordinate_system == ras_system else frame.with_coordinate_system(ras_system)
    )

    matrix = affine[:3, :3]
    translation = affine[:3, 3]
    source_axes: tuple[str, ...] = ("i", "j", "k")
    if time:
        step = float(header["pixdim"][4])
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f"time=True requires positive finite pixdim[4], got {step!r}")
        matrix = np.zeros((4, 4), dtype=np.float64)
        matrix[:3, :3] = affine[:3, :3]
        matrix[3, 3] = step
        translation = np.array([*translation, float(header["toffset"])])
        source_axes = (*source_axes, "t")
    source = ArrayCoordinates(
        source_axes,
        ("1",) * len(source_axes),
        axis_types=("space",) * 3 + (("time",) if time else ()),
        sample_offset=(0.5,) * 3 + ((None,) if time else ()),
    )
    ras_affine = AffineTransform(
        source=source, target=ras_view, matrix=matrix, translation=translation
    )
    transform: AffineTransform
    if frame == ras_view:
        transform = ras_affine
    else:
        try:
            composed = compose(ras_affine, coordinate_system_change(ras_view, frame))
        except ValueError as error:
            raise ValueError(f"supplied frame cannot be derived from NIfTI RAS: {error}") from error
        assert isinstance(composed, AffineTransform)
        transform = composed

    coords: dict[str, CoordinateSpec] = {}
    for index, dim in enumerate(dims):
        if index < 3 or (index == 3 and time):
            coords[dim] = index_coordinate(dim, shape[index])
        elif index == 3:
            step = float(header["pixdim"][4])
            if np.isfinite(step) and step > 0:
                values = float(header["toffset"]) + step * np.arange(shape[index])
                attrs = {"units": temporal_unit} if temporal_unit is not None else {}
                coords[dim] = (dim, values, MappingProxyType(attrs))
            else:
                # A zero or missing step would give every sample the same time value.
                report.append(
                    ("invalid-time-step", f"pixdim[4]={step!r}; t is an index coordinate")
                )
                coords[dim] = index_coordinate(dim, shape[index])
        else:
            coords[dim] = index_coordinate(dim, shape[index])
    if ndim == 2:
        coords["k"] = ((), 0, MappingProxyType({"units": "1"}))
    return NiftiGeometry(dims, MappingProxyType(coords), transform, frame, tuple(report))


def to_dataarray(geometry: NiftiGeometry, data: DuckArray) -> xr.DataArray:
    """Bind caller-supplied pixels to imported NIfTI geometry without evaluating them.

    Args:
        geometry: Geometry returned by ``from_header``.
        data: NumPy or dask array in header dimension order.

    Returns:
        DataArray carrying the imported transform and sharing the supplied pixels.

    Raises:
        TypeError: If ``geometry`` has the wrong type or ``data`` is not a duck array.
        ValueError: If the pixel shape differs from the header shape.
    """
    if not isinstance(geometry, NiftiGeometry):
        raise TypeError(f"geometry must be a NiftiGeometry, got {type(geometry).__name__}")
    grid, other_coords = _grid_and_coords(geometry.transform, geometry.coords)
    return frame_array(data, grid, dims=geometry.dims, coords=other_coords)


def to_header(
    geometry: Geometry,
    *,
    dims: Sequence[str],
    xform_code: str = "scanner",
    qform: bool = True,
    header: nib.Nifti1Header | nib.Nifti2Header | None = None,
) -> tuple[nib.Nifti1Header | nib.Nifti2Header, Report]:
    """Export a three- or four-dimensional lattice into a NIfTI header.

    Args:
        geometry: Current array samples and their transform into a patient frame.
        dims: Geometry dimensions in NIfTI voxel-axis order, independent of array order.
        xform_code: Coded NIfTI space for the sform and representable qform.
        qform: Write a qform when the spatial columns are orthogonal.
        header: Optional NIfTI-1 or NIfTI-2 header to copy before updating geometry.
            A three-dimensional export preserves its time fields.

    Returns:
        A new header and a report of geometry that could not be represented.

    Raises:
        TypeError: If a public argument has the wrong Python type.
        ValueError: If the geometry is not a three- or four-dimensional regular lattice,
            its frame cannot be converted to RAS, its units are unsupported, or time is
            coupled to space.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError(f"geometry must be a Geometry, got {type(geometry).__name__}")
    if isinstance(dims, str) or not isinstance(dims, Sequence):
        raise TypeError("dims must be a sequence of three or four dimension names")
    if len(dims) not in (3, 4):
        raise ValueError(f"dims must name three or four geometry dimensions, got {tuple(dims)!r}")
    if not isinstance(qform, bool):
        raise TypeError(f"qform must be a bool, got {type(qform).__name__}")
    if header is not None and not isinstance(header, nib.Nifti1Header | nib.Nifti2Header):
        raise TypeError(
            f"header must be a NIfTI-1 or NIfTI-2 header or None, got {type(header).__name__}"
        )
    if not isinstance(xform_code, str):
        raise TypeError(
            f"xform_code must be a coded NIfTI space name, got {type(xform_code).__name__}"
        )
    try:
        code = int(cast(int, nib.nifti1.xform_codes.code[xform_code]))
    except KeyError as error:
        raise ValueError(f"xform_code must name a known NIfTI space, got {xform_code!r}") from error
    if code <= 0:
        raise ValueError("xform_code must name a coded NIfTI space, got 'unknown'")

    order = tuple(dims)
    try:
        lattice = geometry.lattice(dims=order)
    except (TypeError, ValueError) as error:
        raise ValueError(f"NIfTI export requires a regular affine lattice: {error}") from error

    frame = lattice.frame
    if len(frame.axes) != len(order):
        raise ValueError(
            f"NIfTI export requires {len(order)} frame axes for {len(order)} dims, "
            f"got {len(frame.axes)}"
        )
    spatial_units = frame.units[:3]
    if len(set(spatial_units)) != 1 or spatial_units[0] not in _SPATIAL_UNITS.values():
        raise ValueError(f"NIfTI export requires spatial units m, mm or um, got {spatial_units!r}")
    spatial_unit = spatial_units[0]
    assert spatial_unit is not None
    time_unit: str | None = None
    if len(order) == 4:
        time_unit = frame.units[3]
        if time_unit not in _TIME_UNITS.values() or frame.coordinate_system.axis_types[3] != "time":
            raise ValueError(
                f"NIfTI export requires a fourth time axis with unit s, ms or us, got "
                f"{frame.coordinate_system.axis_types[3]!r} in {time_unit!r}"
            )
    ras_system = _ras_system(spatial_unit, time_unit)
    ras_view = frame.with_coordinate_system(ras_system)
    try:
        change = coordinate_system_change(frame, ras_view)
    except ValueError as error:
        raise ValueError(f"NIfTI export requires a frame convertible to RAS: {error}") from error
    matrix = change.matrix @ lattice.matrix
    origin = change.matrix @ lattice.origin + change.translation
    if len(order) == 4 and (np.any(matrix[:3, 3] != 0) or np.any(matrix[3, :3] != 0)):
        raise ValueError("NIfTI export requires zero space-time cross terms")
    if len(order) == 4 and (not np.isfinite(matrix[3, 3]) or matrix[3, 3] <= 0):
        raise ValueError(f"NIfTI export requires a positive finite time step, got {matrix[3, 3]!r}")

    affine = np.eye(4)
    affine[:3, :3] = matrix[:3, :3]
    affine[:3, 3] = origin[:3]
    result = nib.Nifti1Header() if header is None else header.copy()  # type: ignore[no-untyped-call]
    preserved_time_step = float(result["pixdim"][4])
    result.set_data_shape(tuple(geometry.array.sizes[dim] for dim in order))
    result.set_sform(affine, code=code)
    spatial_code = next(code for code, unit in _SPATIAL_UNITS.items() if unit == spatial_unit)
    if time_unit is not None:
        temporal_code = next(code for code, unit in _TIME_UNITS.items() if unit == time_unit)
        result["pixdim"][4] = matrix[3, 3]
        result["toffset"] = origin[3]
    elif header is not None:
        temporal_code = int(result["xyzt_units"]) & 0x38
        result["pixdim"][4] = preserved_time_step
    else:
        temporal_code = 0
    result["xyzt_units"] = spatial_code | temporal_code

    report: list[tuple[str, str]] = []
    columns = affine[:3, :3]
    scales = np.linalg.norm(columns, axis=0)
    result["pixdim"][1:4] = scales
    spatial = Lattice(
        frame=ReferenceFrame.local(patient_coordinate_system(RAS, spatial_unit)),
        dims=order[:3],
        origin=origin[:3],
        matrix=columns,
    )
    # NIfTI-1 stores the sform as float32, so a rotation read from a file is orthogonal only to
    # about 1e-7; nibabel strips that residual shear when it fits the quaternion.
    if qform and affine_class(spatial, tolerance=_QFORM_ORTHOGONALITY).at_most("scaled_rigid"):
        result.set_qform(affine, code=code)
    else:
        result.set_qform(None, code=0)
        if qform:
            report.append(("qform-unrepresentable", "spatial affine has shear or a zero step"))
    return result, tuple(report)


__all__ = [
    "NiftiGeometry",
    "from_header",
    "open",
    "to_dataarray",
    "to_header",
]
