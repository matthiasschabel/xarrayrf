"""Translate DICOM image geometry without reading pixel data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from pydicom.dataset import Dataset
from pydicom.uid import DeformableSpatialRegistrationStorage

from xarrayrf import (
    AffineTransform,
    ArrayCoordinates,
    CoordinateSystem,
    ReferenceFrame,
    affine_class,
)
from xarrayrf._sampling import uniform_step
from xarrayrf.anatomy import LPS, patient_coordinate_system
from xarrayrf.native import (
    CoordinateSpec,
    DuckArray,
    Report,
    _grid_and_coords,
    _require_duck_array,
    frame_array,
    index_coordinate,
)

from ._reader import open as open

FRAME_OF_REFERENCE_NAMESPACE = "dicom-frame-of-reference"
"""Identifier namespace of a declared DICOM frame: ``(namespace, FrameOfReferenceUID)``.

Callers declaring the same frame for data from another format, such as a NIfTI file converted
from a DICOM series, use this namespace so the frames compare equivalent.
"""


@dataclass(frozen=True)
class DicomGeometry:
    """Array coordinates, patient-space transform and import decisions.

    ``order`` maps sorted slices to input dataset or original frame indices.
    ``slice_intervals`` uses the slice coordinate's units and is absent when any
    slice has no positive thickness.
    """

    dims: tuple[str, str, str]
    coords: Mapping[str, CoordinateSpec]
    transform: AffineTransform
    frame: ReferenceFrame
    order: tuple[int, ...]
    patient_position: str | None
    slice_intervals: npt.NDArray[np.float64] | None
    report: Report


def to_dataarray(geometry: DicomGeometry, data: DuckArray) -> xr.DataArray:
    """Bind source-order pixels to sorted DICOM geometry without evaluating them.

    The slice axis is axis 0 (``k``). For ``from_datasets``, provide one slice per
    dataset in input order. For ``from_enhanced``, provide the full multiframe pixel
    array in original frame order, including unselected frames. ``geometry.order``
    selects and sorts slices. Declared ``slice_intervals`` supply slice thickness
    in the cells domain. A noncontiguous NumPy order requires an indexed gather; dask
    gathers remain lazy.

    Args:
        geometry: Geometry returned by ``from_datasets`` or ``from_enhanced``.
        data: NumPy or dask source pixel stack, with slice axis first.

    Returns:
        DataArray carrying the imported transform and sorted pixel slices.

    Raises:
        TypeError: If ``geometry`` has the wrong type or ``data`` is not a duck array.
        ValueError: If the in-plane shape differs or the source lacks a selected slice.
    """
    if not isinstance(geometry, DicomGeometry):
        raise TypeError(f"geometry must be a DicomGeometry, got {type(geometry).__name__}")
    shape = (len(geometry.order), *(np.size(geometry.coords[dim][1]) for dim in geometry.dims[1:]))
    data = _require_duck_array(data)
    source_shape = data.shape
    if len(source_shape) != len(shape) or source_shape[1:] != shape[1:]:
        raise ValueError(f"data shape {source_shape} does not match geometry shape {shape}")
    if source_shape[0] <= max(geometry.order):
        raise ValueError(
            f"data shape {source_shape} cannot supply geometry shape {shape}: "
            f"slice axis 0 requires index {max(geometry.order)}"
        )
    first = geometry.order[0]
    if geometry.order == tuple(range(first, first + len(geometry.order))):
        sorted_data = data[first : first + len(geometry.order), :, :]
    else:
        sorted_data = cast(Any, data)[list(geometry.order), :, :]
    grid, other_coords = _grid_and_coords(
        geometry.transform,
        geometry.coords,
        intervals={geometry.transform.source.axes[0]: geometry.slice_intervals}
        if geometry.slice_intervals is not None
        else None,
    )
    return frame_array(sorted_data, grid, dims=geometry.dims, coords=other_coords)


@dataclass(frozen=True)
class _Slice:
    index: int
    position: npt.NDArray[np.float64]
    orientation: npt.NDArray[np.float64]
    spacing: npt.NDArray[np.float64]
    rows: int
    columns: int
    uid: str | None
    patient_position: str | None
    thickness: float | None


def _positive_tolerance(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _vector(dataset: Dataset, name: str, length: int, index: int) -> npt.NDArray[np.float64]:
    value = getattr(dataset, name, None)
    if value is None:
        raise ValueError(f"dataset {index} requires {name}")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"dataset {index} needs numeric {name}") from error
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"dataset {index} needs {length} finite values for {name}")
    return result


def _dimension(dataset: Dataset, name: str, index: int) -> int:
    value = getattr(dataset, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"dataset {index} requires positive integer {name}")
    return int(value)


def _optional_text(dataset: Dataset, name: str) -> str | None:
    value = getattr(dataset, name, None)
    return str(value) if value not in (None, "") else None


def patient_frame(frame_of_reference_uid: str | None) -> ReferenceFrame:
    """Return the LPS patient frame declared by a DICOM Frame of Reference UID.

    An absent UID gives a fresh local frame, since no shared identity was declared.

    Args:
        frame_of_reference_uid: DICOM Frame of Reference UID, or None.

    Returns:
        The declared patient frame, or a new local frame.

    Raises:
        TypeError: If the UID is neither a string nor None.
        ValueError: If the UID is empty.
    """
    if frame_of_reference_uid is not None and not isinstance(frame_of_reference_uid, str):
        raise TypeError("frame_of_reference_uid must be a string or None")
    if frame_of_reference_uid == "":
        raise ValueError("frame_of_reference_uid must be nonempty")
    system = patient_coordinate_system(LPS, "mm")
    return (
        ReferenceFrame.declared((FRAME_OF_REFERENCE_NAMESPACE, frame_of_reference_uid), system)
        if frame_of_reference_uid is not None
        else ReferenceFrame.local(system)
    )


def _read_slice(dataset: Dataset, index: int, metadata: Dataset) -> _Slice:
    if not isinstance(dataset, Dataset):
        raise TypeError(f"dataset {index} must be a pydicom Dataset")
    spacing = _vector(dataset, "PixelSpacing", 2, index)
    if np.any(spacing <= 0):
        raise ValueError(f"dataset {index} requires positive PixelSpacing")
    thickness_value = getattr(dataset, "SliceThickness", None)
    thickness: float | None = None
    if thickness_value is not None:
        try:
            parsed = float(thickness_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"dataset {index} needs numeric SliceThickness") from error
        if np.isfinite(parsed) and parsed > 0:
            thickness = parsed
    anatomical_type = _optional_text(metadata, "AnatomicalOrientationType")
    if anatomical_type not in (None, "BIPED"):
        raise ValueError(
            f"dataset {index} has unsupported AnatomicalOrientationType {anatomical_type!r}"
        )
    return _Slice(
        index,
        _vector(dataset, "ImagePositionPatient", 3, index),
        _vector(dataset, "ImageOrientationPatient", 6, index),
        spacing,
        _dimension(metadata, "Rows", index),
        _dimension(metadata, "Columns", index),
        _optional_text(metadata, "FrameOfReferenceUID"),
        _optional_text(metadata, "PatientPosition"),
        thickness,
    )


def _assemble(
    slices: Sequence[_Slice], orientation_tolerance: float, slice_tolerance: float
) -> DicomGeometry:
    if not slices:
        raise ValueError("datasets must contain at least one image")
    first = slices[0]
    raw_r, raw_c = first.orientation[:3], first.orientation[3:]
    corrections: list[float] = []
    for item in slices:
        if np.max(np.abs(item.orientation - first.orientation)) > orientation_tolerance:
            raise ValueError(
                f"dataset {item.index} ImageOrientationPatient disagrees with dataset {first.index}"
            )
        r, c = item.orientation[:3], item.orientation[3:]
        if (
            abs(np.linalg.norm(r) - 1) > orientation_tolerance
            or abs(np.linalg.norm(c) - 1) > orientation_tolerance
            or abs(float(np.dot(r, c))) > orientation_tolerance
        ):
            raise ValueError(f"dataset {item.index} ImageOrientationPatient is not orthonormal")
        if not np.array_equal(item.spacing, first.spacing):
            raise ValueError(
                f"dataset {item.index} PixelSpacing disagrees with dataset {first.index}"
            )
        for name in ("rows", "columns", "uid"):
            if getattr(item, name) != getattr(first, name):
                raise ValueError(
                    f"dataset {item.index} {name} disagrees with dataset {first.index}"
                )
        r_unit = r / np.linalg.norm(r)
        c_unit = c - np.dot(c, r_unit) * r_unit
        c_unit /= np.linalg.norm(c_unit)
        corrections.append(float(max(np.max(np.abs(r_unit - r)), np.max(np.abs(c_unit - c)))))

    r = raw_r / np.linalg.norm(raw_r)
    c = raw_c - np.dot(raw_c, r) * r
    c /= np.linalg.norm(c)
    normal = np.cross(r, c)
    sorted_slices = sorted(slices, key=lambda item: float(np.dot(normal, item.position)))
    positions = np.asarray([float(np.dot(normal, item.position)) for item in sorted_slices])
    origin = sorted_slices[0].position
    plane = origin - np.dot(origin, normal) * normal
    for item in sorted_slices[1:]:
        item_plane = item.position - np.dot(item.position, normal) * normal
        displacement = item_plane - plane
        in_plane_pixels = np.array(
            [np.dot(displacement, r) / first.spacing[1], np.dot(displacement, c) / first.spacing[0]]
        )
        if np.linalg.norm(in_plane_pixels) > slice_tolerance:
            raise ValueError(f"dataset {item.index} ImagePositionPatient has an in-plane shift")
    if len(positions) > 1:
        mean_step = float((positions[-1] - positions[0]) / (len(positions) - 1))
        gaps = np.diff(positions)
        duplicates = np.flatnonzero(gaps <= slice_tolerance * mean_step)
        if duplicates.size:
            a = sorted_slices[int(duplicates[0])].index
            b = sorted_slices[int(duplicates[0]) + 1].index
            raise ValueError(
                f"duplicate slice positions at input indices {a} and {b}; several images share "
                "a slice position (e.g. echoes, b-values or time points); assemble such series "
                "with an application-level reader and frame each spatial stack"
            )

    report: list[tuple[str, str]] = []
    correction = max(corrections)
    report.append(
        (
            "orientation-orthonormalized",
            f"maximum cosine correction {correction:g}; tolerance {orientation_tolerance:g}",
        )
    )
    offsets = positions - positions[0]
    step = uniform_step(offsets, slice_tolerance)
    if step is None:
        axis = "slice_offset"
        values = offsets
        unit = "mm"
        report.append(("slice-axis", "slice_offset in mm; nonuniform or single-slice stack"))
    else:
        axis = "k"
        values = np.arange(len(slices), dtype=np.int64)
        unit = "1"
        report.append(("slice-axis", f"k index; uniform step {step:g} mm"))
    uid = first.uid
    frame = patient_frame(uid)
    if uid is None:
        report.append(("local-frame", "FrameOfReferenceUID absent; created a local frame"))
    patient_positions = {item.patient_position for item in slices}
    patient_position = first.patient_position if len(patient_positions) == 1 else None
    if len(patient_positions) > 1:
        report.append(("patient-position-disagree", "PatientPosition differs across slices"))
    elif patient_position is None:
        report.append(("patient-position-missing", "PatientPosition absent on all slices"))
    thicknesses = [item.thickness for item in sorted_slices]
    intervals = None
    if all(value is not None for value in thicknesses):
        widths = np.asarray(thicknesses, dtype=np.float64) / (step if step is not None else 1.0)
        centers = np.asarray(values, dtype=np.float64)
        intervals = np.column_stack((centers - widths / 2, centers + widths / 2))
        intervals.setflags(write=False)
    else:
        report.append(
            ("slice-thickness-missing", "a slice lacks positive SliceThickness; no intervals")
        )
    source = ArrayCoordinates(
        (axis, "j", "i"),
        (unit, "1", "1"),
        axis_types=("space",) * 3,
        sample_offset=(0.5,) * 3,
    )
    transform = AffineTransform(
        source=source,
        target=frame,
        matrix=np.column_stack(
            (
                normal * (step if step is not None else 1.0),
                c * first.spacing[0],
                r * first.spacing[1],
            )
        ),
        translation=origin,
    )
    coords: dict[str, CoordinateSpec] = {
        axis: ("k", values, MappingProxyType({"units": unit})),
        "j": index_coordinate("j", first.rows),
        "i": index_coordinate("i", first.columns),
    }
    return DicomGeometry(
        ("k", "j", "i"),
        MappingProxyType(coords),
        transform,
        frame,
        tuple(item.index for item in sorted_slices),
        patient_position,
        intervals,
        tuple(report),
    )


def from_datasets(
    datasets: Sequence[Dataset],
    *,
    orientation_tolerance: float = 1e-4,
    slice_tolerance: float = 0.01,
) -> DicomGeometry:
    """Import a classic single-frame stack from metadata-only pydicom datasets.

    Args:
        datasets: One dataset per slice; input order may be arbitrary.
        orientation_tolerance: Maximum cosine error or disagreement.
        slice_tolerance: Slice residual in steps and in-plane shift in pixel spacings.

    Returns:
        Geometry declaration with ``order`` mapping slices to input indices.

    Raises:
        TypeError: If an argument or dataset has the wrong type.
        ValueError: If required geometry is missing, inconsistent or unsupported.
    """
    orientation_tolerance = _positive_tolerance(orientation_tolerance, "orientation_tolerance")
    slice_tolerance = _positive_tolerance(slice_tolerance, "slice_tolerance")
    if not isinstance(datasets, Sequence) or isinstance(datasets, str | bytes):
        raise TypeError("datasets must be a sequence of pydicom Datasets")
    return _assemble(
        [_read_slice(dataset, index, dataset) for index, dataset in enumerate(datasets)],
        orientation_tolerance,
        slice_tolerance,
    )


def _group(shared: Dataset, frame: Dataset, name: str, index: int) -> Dataset:
    shared_items = getattr(shared, name, None)
    frame_items = getattr(frame, name, None)
    if shared_items is not None and frame_items is not None:
        raise ValueError(f"frame {index} has {name} in both shared and per-frame groups")
    items = frame_items if frame_items is not None else shared_items
    if items is None or len(items) != 1:
        raise ValueError(f"frame {index} requires one {name} in shared or per-frame groups")
    return cast(Dataset, items[0])


def from_enhanced(
    dataset: Dataset,
    *,
    frames: Sequence[int] | None = None,
    orientation_tolerance: float = 1e-4,
    slice_tolerance: float = 0.01,
) -> DicomGeometry:
    """Import selected frames of one enhanced multiframe DICOM object.

    Args:
        dataset: Enhanced object; pixels are never accessed.
        frames: Optional frame indices selecting one spatial stack.
        orientation_tolerance: Maximum cosine error or disagreement.
        slice_tolerance: Slice residual in steps and in-plane shift in pixel spacings.

    Returns:
        Geometry declaration with ``order`` mapping slices to frame indices.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If functional groups or selected geometry are invalid.
    """
    orientation_tolerance = _positive_tolerance(orientation_tolerance, "orientation_tolerance")
    slice_tolerance = _positive_tolerance(slice_tolerance, "slice_tolerance")
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a pydicom Dataset")
    count = _dimension(dataset, "NumberOfFrames", 0)
    shared_items = getattr(dataset, "SharedFunctionalGroupsSequence", None)
    frame_items = getattr(dataset, "PerFrameFunctionalGroupsSequence", None)
    if shared_items is None:
        shared = Dataset()
    elif len(shared_items) != 1:
        raise ValueError("SharedFunctionalGroupsSequence requires one item")
    else:
        shared = shared_items[0]
    if frame_items is None or len(frame_items) != count:
        raise ValueError("PerFrameFunctionalGroupsSequence length must equal NumberOfFrames")
    for index, frame in enumerate(frame_items):
        for name in ("PlanePositionSequence", "PlaneOrientationSequence", "PixelMeasuresSequence"):
            if getattr(shared, name, None) is not None and getattr(frame, name, None) is not None:
                raise ValueError(f"frame {index} has {name} in both shared and per-frame groups")
    if frames is None:
        selected = tuple(range(count))
    else:
        if not isinstance(frames, Sequence) or isinstance(frames, str | bytes):
            raise TypeError("frames must be a sequence of frame indices")
        selected = tuple(frames)
        if (
            not selected
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= count
                for index in selected
            )
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("frames must contain unique indices within NumberOfFrames")
    slices: list[_Slice] = []
    for index in selected:
        frame = frame_items[index]
        plane = _group(shared, frame, "PlanePositionSequence", index)
        orientation = _group(shared, frame, "PlaneOrientationSequence", index)
        measures = _group(shared, frame, "PixelMeasuresSequence", index)
        synthetic = Dataset()
        if hasattr(plane, "ImagePositionPatient"):
            synthetic.ImagePositionPatient = plane.ImagePositionPatient
        if hasattr(orientation, "ImageOrientationPatient"):
            synthetic.ImageOrientationPatient = orientation.ImageOrientationPatient
        if hasattr(measures, "PixelSpacing"):
            synthetic.PixelSpacing = measures.PixelSpacing
        if hasattr(measures, "SliceThickness"):
            synthetic.SliceThickness = measures.SliceThickness
        slices.append(_read_slice(synthetic, index, dataset))
    return _assemble(slices, orientation_tolerance, slice_tolerance)


def _homogeneous_matrix(
    value: object, name: str, orientation_tolerance: float
) -> npt.NDArray[np.float64]:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain 16 finite numbers") from error
    if matrix.size != 16 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain 16 finite numbers")
    matrix = matrix.reshape(4, 4)
    if not np.allclose(matrix[3], (0, 0, 0, 1), rtol=0, atol=orientation_tolerance):
        raise ValueError(f"{name} must have last row (0, 0, 0, 1)")
    # Small permitted row errors must not turn a product of affine matrices projective.
    matrix[3] = (0, 0, 0, 1)
    return matrix


def _check_matrix_type(
    matrix: npt.NDArray[np.float64], matrix_type: str, name: str, tolerance: float
) -> None:
    if matrix_type == "AFFINE":
        return
    if matrix_type not in {"RIGID", "RIGID_SCALE"}:
        raise ValueError(
            f"{name} has unsupported FrameOfReferenceTransformationMatrixType {matrix_type!r}"
        )
    axes = ArrayCoordinates(("x", "y", "z"), ("mm",) * 3)
    transform = AffineTransform(
        source=axes, target=axes, matrix=matrix[:3, :3], translation=matrix[:3, 3]
    )
    classification = affine_class(transform, tolerance=tolerance)
    if matrix_type == "RIGID":
        if not classification.at_most("rigid") or not classification.proper:
            raise ValueError(f"{name} RIGID matrix must be orthonormal with determinant +1")
    else:
        if not classification.at_most("scaled_rigid"):
            raise ValueError(f"{name} RIGID_SCALE matrix must have orthogonal nonzero columns")


def equipment_transform(
    dataset: Dataset, *, orientation_tolerance: float = 1e-4
) -> AffineTransform | None:
    """Import an Image to Equipment Mapping Matrix as a patient-to-equipment transform.

    The equipment frame is local to this dataset even when another dataset names the same
    equipment. Only the defined ``ISOCENTER`` equipment identifier is accepted.

    Args:
        dataset: Metadata-only pydicom dataset.
        orientation_tolerance: Maximum rigid-matrix and homogeneous-row error.

    Returns:
        A transform to a local equipment frame, or ``None`` when the matrix is absent.

    Raises:
        TypeError: If the dataset or tolerance has the wrong type.
        ValueError: If the matrix or equipment identifier is invalid.
    """
    orientation_tolerance = _positive_tolerance(orientation_tolerance, "orientation_tolerance")
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a pydicom Dataset")
    value = getattr(dataset, "ImageToEquipmentMappingMatrix", None)
    if value is None:
        return None
    identifier = _optional_text(dataset, "EquipmentCoordinateSystemIdentification")
    if identifier != "ISOCENTER":
        raise ValueError(
            f"EquipmentCoordinateSystemIdentification must be 'ISOCENTER', got {identifier!r}"
        )
    matrix = _homogeneous_matrix(value, "ImageToEquipmentMappingMatrix", orientation_tolerance)
    _check_matrix_type(matrix, "RIGID", "ImageToEquipmentMappingMatrix", orientation_tolerance)
    definition = {"EquipmentCoordinateSystemIdentification": identifier}
    for name in ("Manufacturer", "DeviceSerialNumber"):
        text = _optional_text(dataset, name)
        if text is not None:
            definition[name] = text
    equipment = ReferenceFrame.local(
        CoordinateSystem(("x", "y", "z"), ("mm",) * 3, axis_types=("space",) * 3),
        definition=definition,
    )
    return AffineTransform(
        source=patient_frame(_optional_text(dataset, "FrameOfReferenceUID")),
        target=equipment,
        matrix=matrix[:3, :3],
        translation=matrix[:3, 3],
    )


def registrations(
    dataset: Dataset, *, orientation_tolerance: float = 1e-4
) -> tuple[AffineTransform, ...]:
    """Import frame-to-frame transforms from a Spatial Registration object.

    Matrix Sequence items are multiplied in listed order, so the last matrix acts first
    on a column-vector point. Referenced images cannot resolve a missing item frame here.

    Args:
        dataset: Spatial Registration pydicom dataset.
        orientation_tolerance: Maximum type and homogeneous-row error.

    Returns:
        One affine transform per Registration Sequence item, in item order.

    Raises:
        TypeError: If the dataset or tolerance has the wrong type.
        ValueError: If a frame, registration item or matrix is missing or invalid.
    """
    orientation_tolerance = _positive_tolerance(orientation_tolerance, "orientation_tolerance")
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a pydicom Dataset")
    if _optional_text(dataset, "SOPClassUID") == DeformableSpatialRegistrationStorage:
        raise ValueError("Deformable Spatial Registration objects are unsupported")
    own_uid = _optional_text(dataset, "FrameOfReferenceUID")
    if own_uid is None:
        raise ValueError("dataset requires its own FrameOfReferenceUID")
    items = getattr(dataset, "RegistrationSequence", None)
    if items is None:
        raise ValueError("dataset requires RegistrationSequence")
    target = patient_frame(own_uid)
    result: list[AffineTransform] = []
    for index, item in enumerate(items):
        uid = _optional_text(item, "FrameOfReferenceUID")
        if uid is None:
            raise ValueError(
                f"RegistrationSequence item {index} requires FrameOfReferenceUID; "
                "ReferencedImageSequence cannot resolve it"
            )
        matrix_registrations = getattr(item, "MatrixRegistrationSequence", None)
        if matrix_registrations is None or len(matrix_registrations) != 1:
            raise ValueError(
                f"RegistrationSequence item {index} requires one MatrixRegistrationSequence item"
            )
        matrices = getattr(matrix_registrations[0], "MatrixSequence", None)
        if matrices is None or len(matrices) == 0:
            raise ValueError(f"RegistrationSequence item {index} requires MatrixSequence")
        combined = np.eye(4)
        for matrix_index, entry in enumerate(matrices):
            name = f"RegistrationSequence item {index} MatrixSequence item {matrix_index}"
            value = getattr(entry, "FrameOfReferenceTransformationMatrix", None)
            if value is None:
                raise ValueError(f"{name} requires FrameOfReferenceTransformationMatrix")
            matrix = _homogeneous_matrix(value, name, orientation_tolerance)
            matrix_type = _optional_text(entry, "FrameOfReferenceTransformationMatrixType")
            _check_matrix_type(matrix, matrix_type or "", name, orientation_tolerance)
            combined = combined @ matrix
        result.append(
            AffineTransform(
                source=patient_frame(uid),
                target=target,
                matrix=combined[:3, :3],
                translation=combined[:3, 3],
            )
        )
    return tuple(result)


__all__ = [
    "FRAME_OF_REFERENCE_NAMESPACE",
    "DicomGeometry",
    "equipment_transform",
    "from_datasets",
    "from_enhanced",
    "open",
    "patient_frame",
    "registrations",
    "to_dataarray",
]
