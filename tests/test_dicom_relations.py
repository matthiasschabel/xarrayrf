"""Frame relations imported from synthetic DICOM metadata."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence

from xarrayrf import ReferenceFrame
from xarrayrf.dicom import (
    FRAME_OF_REFERENCE_NAMESPACE,
    equipment_transform,
    registrations,
)

ATOL = 1e-12  # Matrices here contain only exact small integers.
OWN_UID = "1.2.3.4"
SOURCE_UID = "1.2.3.5"


def matrix_values(matrix: npt.NDArray[np.float64 | np.int64]) -> list[float | int]:
    return cast(list[float | int], matrix.ravel().tolist())


def equipment(matrix: npt.NDArray[np.float64 | np.int64] | None = None) -> Dataset:
    result = Dataset()
    result.FrameOfReferenceUID = OWN_UID
    if matrix is not None:
        result.ImageToEquipmentMappingMatrix = matrix_values(matrix)
        result.EquipmentCoordinateSystemIdentification = "ISOCENTER"
    return result


def matrix_item(matrix: npt.NDArray[np.float64 | np.int64], matrix_type: str = "RIGID") -> Dataset:
    result = Dataset()
    result.FrameOfReferenceTransformationMatrix = matrix_values(matrix)
    result.FrameOfReferenceTransformationMatrixType = matrix_type
    return result


def registration_item(
    uid: str = SOURCE_UID, matrices: tuple[Dataset, ...] | None = None
) -> Dataset:
    result = Dataset()
    result.FrameOfReferenceUID = uid
    group = Dataset()
    group.MatrixSequence = DicomSequence(list(matrices or (matrix_item(np.eye(4)),)))
    result.MatrixRegistrationSequence = DicomSequence([group])
    return result


def registration(*items: Dataset) -> Dataset:
    result = Dataset()
    result.FrameOfReferenceUID = OWN_UID
    result.RegistrationSequence = DicomSequence(list(items))
    return result


def test_equipment_present_and_absent() -> None:
    assert equipment_transform(equipment()) is None
    matrix = np.eye(4)
    matrix[:3, 3] = [10, 20, 30]
    source = equipment(matrix)
    source.Manufacturer = "Acme"
    source.DeviceSerialNumber = "42"
    transform = equipment_transform(source)
    assert transform is not None
    assert isinstance(transform.source, ReferenceFrame)
    assert isinstance(transform.target, ReferenceFrame)
    assert transform.source.identifier == (FRAME_OF_REFERENCE_NAMESPACE, OWN_UID)
    assert transform.source.coordinate_system.orientation == (
        "right-to-left",
        "anterior-to-posterior",
        "inferior-to-superior",
    )
    assert transform.target.is_anonymous
    assert not transform.source.is_anonymous
    assert transform.target.coordinate_system.axes == ("x", "y", "z")
    assert transform.target.coordinate_system.units == ("mm", "mm", "mm")
    assert transform.target.coordinate_system.orientation == (None, None, None)
    assert dict(transform.target.definition) == {
        "EquipmentCoordinateSystemIdentification": "ISOCENTER",
        "Manufacturer": "Acme",
        "DeviceSerialNumber": "42",
    }
    assert_allclose(transform.transform_point([1, 2, 3]), [11, 22, 33], rtol=0, atol=ATOL)
    second = equipment_transform(source)
    assert second is not None
    assert isinstance(second.target, ReferenceFrame)
    assert second.target.is_anonymous
    assert not transform.target.is_equivalent_frame(second.target)


def test_equipment_without_frame_uid_mints_anonymous_patient_frame() -> None:
    source = equipment(np.eye(4))
    del source.FrameOfReferenceUID
    transform = equipment_transform(source)
    assert transform is not None
    assert isinstance(transform.source, ReferenceFrame)
    assert transform.source.identifier[0] == "xarrayrf.anonymous"
    assert transform.source.is_anonymous
    assert isinstance(transform.target, ReferenceFrame)
    assert transform.target.is_anonymous


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.diag([2.0, 1.0, 1.0, 1.0]), "RIGID matrix must be orthonormal"),
        (np.diag([-1.0, 1.0, 1.0, 1.0]), r"determinant \+1"),
        (np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]]), "last row"),
        (np.diag([float("nan"), 1.0, 1.0, 1.0]), "16 finite numbers"),
    ],
)
def test_equipment_invalid_matrices(
    matrix: npt.NDArray[np.float64 | np.int64], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        equipment_transform(equipment(matrix))


def test_equipment_invalid_identifier_and_matrix_values() -> None:
    source = equipment(np.eye(4))
    source.EquipmentCoordinateSystemIdentification = "OTHER"
    with pytest.raises(ValueError, match="EquipmentCoordinateSystemIdentification must be"):
        equipment_transform(source)
    del source.EquipmentCoordinateSystemIdentification
    with pytest.raises(ValueError, match="EquipmentCoordinateSystemIdentification must be"):
        equipment_transform(source)
    source = equipment(np.eye(4))
    source.ImageToEquipmentMappingMatrix = [1.0] * 15
    with pytest.raises(ValueError, match="ImageToEquipmentMappingMatrix must contain 16"):
        equipment_transform(source)
    del source.ImageToEquipmentMappingMatrix
    source.add_new(0x00289520, "LO", ["bad"] * 16)
    with pytest.raises(ValueError, match="ImageToEquipmentMappingMatrix must contain 16"):
        equipment_transform(source)


def test_registration_two_items_and_noncommuting_matrix_order() -> None:
    translation = np.eye(4)
    translation[:3, 3] = [1, 0, 0]
    rotation = np.eye(4)
    rotation[:2, :2] = [[0, -1], [1, 0]]
    source = registration(
        registration_item(matrices=(matrix_item(translation), matrix_item(rotation))),
        registration_item(uid="1.2.3.6"),
    )
    first, second = registrations(source)
    assert isinstance(first.source, ReferenceFrame)
    assert isinstance(first.target, ReferenceFrame)
    assert isinstance(second.source, ReferenceFrame)
    assert isinstance(second.target, ReferenceFrame)
    assert first.source.identifier == (FRAME_OF_REFERENCE_NAMESPACE, SOURCE_UID)
    assert second.source.identifier == (FRAME_OF_REFERENCE_NAMESPACE, "1.2.3.6")
    assert (
        first.target.identifier
        == second.target.identifier
        == (
            FRAME_OF_REFERENCE_NAMESPACE,
            OWN_UID,
        )
    )
    assert_allclose(first.transform_point([1, 0, 0]), [1, 1, 0], rtol=0, atol=ATOL)
    assert_allclose(first.matrix, (translation @ rotation)[:3, :3], rtol=0, atol=ATOL)
    assert_allclose(second.transform_point([1, 0, 0]), [1, 0, 0], rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    ("matrix_type", "matrix", "accepted", "message"),
    [
        ("RIGID", np.eye(4), True, ""),
        ("RIGID", np.diag([-1.0, 1.0, 1.0, 1.0]), False, r"determinant \+1"),
        ("RIGID", np.diag([2.0, 1.0, 1.0, 1.0]), False, "orthonormal"),
        ("RIGID_SCALE", np.diag([2.0, 3.0, 4.0, 1.0]), True, ""),
        (
            "RIGID_SCALE",
            np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            False,
            "orthogonal nonzero columns",
        ),
        ("RIGID_SCALE", np.diag([0.0, 1.0, 1.0, 1.0]), False, "orthogonal nonzero columns"),
        (
            "AFFINE",
            np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            True,
            "",
        ),
        (
            "AFFINE",
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]),
            False,
            "last row",
        ),
        ("OTHER", np.eye(4), False, "unsupported FrameOfReferenceTransformationMatrixType"),
    ],
)
def test_registration_matrix_types(
    matrix_type: str, matrix: npt.NDArray[np.float64 | np.int64], accepted: bool, message: str
) -> None:
    source = registration(registration_item(matrices=(matrix_item(matrix, matrix_type),)))
    if accepted:
        (transform,) = registrations(source)
        assert_allclose(transform.matrix, matrix[:3, :3], rtol=0, atol=ATOL)
    else:
        with pytest.raises(ValueError, match=message):
            registrations(source)


@pytest.mark.parametrize("step", [1e-8, 1e-16])
def test_registration_rigid_scale_accepts_tiny_nonzero_step(step: float) -> None:
    matrix = np.diag([step, 1.0, 1.0, 1.0])
    source = registration(registration_item(matrices=(matrix_item(matrix, "RIGID_SCALE"),)))
    (result,) = registrations(source)
    assert_allclose(result.matrix, matrix[:3, :3], rtol=1e-12, atol=0)


def test_registration_rigid_rejects_small_scale_with_excess_determinant() -> None:
    tolerance = 1e-4
    matrix = np.diag([np.sqrt(1 + 0.8 * tolerance)] * 3 + [1.0])
    source = registration(registration_item(matrices=(matrix_item(matrix, "RIGID"),)))
    with pytest.raises(ValueError, match=r"RIGID matrix must be orthonormal with determinant \+1"):
        registrations(source, orientation_tolerance=tolerance)


def test_registration_rigid_requires_normalized_orthogonality_within_tolerance() -> None:
    # Gram and determinant errors are within 1e-4, but the normalized columns miss orthogonality
    # by 1.0001e-4. A rigid map must meet every containing class's constraints, so this is
    # refused; before the shared classifier it was accepted (an O(tolerance**2) boundary).
    tolerance = 1e-4
    length = np.sqrt(1 - tolerance)
    cosine = tolerance / (1 - tolerance)
    first = length * np.array([1.0, 0.0, 0.0])
    second = length * np.array([cosine, np.sqrt(1 - cosine**2), 0.0])
    third = np.sqrt(1 + tolerance) * np.array([0.0, 0.0, 1.0])
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([first, second, third])
    gram = matrix[:3, :3].T @ matrix[:3, :3]
    assert np.max(np.abs(gram - np.eye(3))) <= tolerance + 1e-15
    source = registration(registration_item(matrices=(matrix_item(matrix, "RIGID"),)))
    with pytest.raises(ValueError, match=r"RIGID matrix must be orthonormal with determinant \+1"):
        registrations(source, orientation_tolerance=tolerance)


def test_registration_nonfinite_matrix() -> None:
    matrix = np.eye(4)
    matrix[0, 0] = float("inf")
    with pytest.raises(ValueError, match="item 0 MatrixSequence item 0 must contain 16 finite"):
        registrations(registration(registration_item(matrices=(matrix_item(matrix),))))


def test_registration_nearly_homogeneous_row_is_normalized_before_composition() -> None:
    first = np.eye(4)
    first[0, 3] = 10
    second = np.eye(4)
    second[3, 0] = 1e-6
    (transform,) = registrations(
        registration(registration_item(matrices=(matrix_item(first), matrix_item(second))))
    )
    assert_allclose(transform.transform_point([1, 0, 0]), [11, 0, 0], rtol=0, atol=ATOL)


def test_registration_referenced_image_only_and_missing_own_uid() -> None:
    image = Dataset()
    image.ReferencedImageSequence = DicomSequence([Dataset()])
    source = registration(image)
    with pytest.raises(ValueError, match=r"item 0 requires FrameOfReferenceUID.*ReferencedImage"):
        registrations(source)
    del source.FrameOfReferenceUID
    with pytest.raises(ValueError, match="dataset requires its own FrameOfReferenceUID"):
        registrations(source)


def test_registration_missing_sequences_and_matrix_fields() -> None:
    source = registration(registration_item())
    del source.RegistrationSequence
    with pytest.raises(ValueError, match="dataset requires RegistrationSequence"):
        registrations(source)
    item = registration_item()
    del item.MatrixRegistrationSequence
    with pytest.raises(ValueError, match="item 0 requires one MatrixRegistrationSequence item"):
        registrations(registration(item))
    item = registration_item()
    item.MatrixRegistrationSequence[0].MatrixSequence = DicomSequence([])
    with pytest.raises(ValueError, match="item 0 requires MatrixSequence"):
        registrations(registration(item))
    item = registration_item()
    del item.MatrixRegistrationSequence[0].MatrixSequence[0].FrameOfReferenceTransformationMatrix
    with pytest.raises(ValueError, match="item 0 MatrixSequence item 0 requires"):
        registrations(registration(item))


def test_deformable_registration_refused() -> None:
    source = registration(registration_item())
    source.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.3"
    with pytest.raises(ValueError, match="Deformable Spatial Registration"):
        registrations(source)


@pytest.mark.parametrize("function", [equipment_transform, registrations])
def test_relation_public_validation(function: object) -> None:
    with pytest.raises(TypeError, match="orientation_tolerance must be a positive finite number"):
        function(Dataset(), orientation_tolerance="bad")  # type: ignore[operator]
    with pytest.raises(ValueError, match="orientation_tolerance must be a positive finite number"):
        function(Dataset(), orientation_tolerance=0)  # type: ignore[operator]
    with pytest.raises(TypeError, match="dataset must be a pydicom Dataset"):
        function(object())  # type: ignore[operator]
