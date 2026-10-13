"""The ITK transform-file importer against SimpleITK, and its refusals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

import xarrayrf as xrf
from xarrayrf.anatomy import LPS, RAS, patient_coordinate_system

pytest.importorskip("nibabel")
pytest.importorskip("scipy")
pytest.importorskip("h5py")
from xarrayrf import itk as rfi

DIRECTORY = Path(__file__).parent / "fixtures" / "itk"
ORACLE = json.loads((DIRECTORY / "oracle.json").read_text(encoding="utf-8"))

LINEAR_ATOL = 1e-12
"""Double-precision files evaluated in a different operation order than ITK's."""

NIFTI_ATOL = 1e-6
"""NIfTI stores geometry in float32 and ITK re-orthonormalizes the sform's direction, which
moves points by about 1e-8 of their distance from the origin (measured: under 1e-6 mm here)."""

LPS_MM = patient_coordinate_system(LPS, "mm")
RAS_MM = patient_coordinate_system(RAS, "mm")
PLANE = xrf.CoordinateSystem(("x", "y"), ("mm", "mm"))


def frames(system: xrf.CoordinateSystem) -> tuple[xrf.ReferenceFrame, xrf.ReferenceFrame]:
    return (
        xrf.ReferenceFrame.declared(("test", "fixed"), system),
        xrf.ReferenceFrame.declared(("test", "moving"), system),
    )


def read(name: str, **options: Any) -> rfi.ItkImport:
    dimension = len(ORACLE["files"][name]["points"][0]) if name in ORACLE["files"] else 3
    space = options.pop("space", None if dimension == 3 else PLANE)
    fixed, moving = frames(space or LPS_MM)
    return rfi.read_transform(DIRECTORY / name, fixed=fixed, moving=moving, space=space, **options)


@pytest.mark.parametrize("name", sorted(ORACLE["files"]))
def test_each_file_matches_simpleitk(name: str) -> None:
    case = ORACLE["files"][name]
    result = read(name)
    mapped = result.transform.transform_point(np.asarray(case["points"]))
    atol = NIFTI_ATOL if name.endswith(".nii.gz") else LINEAR_ATOL
    assert_allclose(mapped, case["mapped"], atol=atol, rtol=0)


def test_types_follow_the_members() -> None:
    assert isinstance(read("affine_3d.mat").transform, xrf.AffineTransform)
    field = read("field_3d.nii.gz").transform
    assert isinstance(field, xrf.DisplacementFieldTransform)
    assert field.outside == "identity"
    assert field.field.dtype == np.float64
    composite = read("composite_3d.h5").transform
    assert isinstance(composite, xrf.CompositeTransform)
    first, second = composite.transforms
    # The field was added last, so it applies first.
    assert isinstance(first, xrf.DisplacementFieldTransform)
    assert isinstance(second, xrf.AffineTransform)
    # A text composite of linear members collapses to one affine.
    assert isinstance(read("composite_3d.tfm").transform, xrf.AffineTransform)


def test_a_float_mat_is_read_exactly() -> None:
    from scipy.io import loadmat

    contents = loadmat(str(DIRECTORY / "ants_0GenericAffine.mat"))
    stored = np.asarray(contents["AffineTransform_float_3_3"]).ravel()
    center = np.asarray(contents["fixed"]).ravel().astype(np.float64)
    affine = read("ants_0GenericAffine.mat").transform
    assert isinstance(affine, xrf.AffineTransform)
    matrix = stored[:9].astype(np.float64).reshape(3, 3)
    assert_allclose(affine.matrix, matrix, rtol=0, atol=0)
    expected = center + stored[9:].astype(np.float64) - matrix @ center
    assert_allclose(affine.translation, expected, rtol=0, atol=1e-13)


def test_ras_frames_are_joined_by_coordinate_changes() -> None:
    case = ORACLE["files"]["composite_3d.h5"]
    fixed, moving = frames(RAS_MM)
    result = rfi.read_transform(DIRECTORY / "composite_3d.h5", fixed=fixed, moving=moving)
    flip = np.array([-1.0, -1.0, 1.0])
    mapped = result.transform.transform_point(np.asarray(case["points"]) * flip)
    assert_allclose(mapped, np.asarray(case["mapped"]) * flip, atol=LINEAR_ATOL, rtol=0)
    assert [code for code, _ in result.report].count("coordinate-system-change") == 2


def test_2d_needs_space_and_frames_must_differ() -> None:
    fixed, moving = frames(PLANE)
    with pytest.raises(ValueError, match="space is required for a 2-D transform"):
        rfi.read_transform(DIRECTORY / "affine_2d.tfm", fixed=fixed, moving=moving)
    with pytest.raises(ValueError, match="space has 3 axes but the ITK transform is 2-D"):
        rfi.read_transform(DIRECTORY / "affine_2d.tfm", fixed=fixed, moving=moving, space=LPS_MM)
    lps_fixed, _ = frames(LPS_MM)
    with pytest.raises(ValueError, match="same frame"):
        rfi.read_transform(DIRECTORY / "affine_3d.tfm", fixed=lps_fixed, moving=lps_fixed)


def test_intermediate_frames_can_be_named() -> None:
    fixed, moving = frames(LPS_MM)
    between = xrf.ReferenceFrame.declared(("test", "between"), LPS_MM)
    result = rfi.read_transform(
        DIRECTORY / "composite_3d.h5", fixed=fixed, moving=moving, intermediate_frames=[between]
    )
    assert isinstance(result.transform, xrf.CompositeTransform)
    assert result.transform.transforms[0].target == between


# --- Refusals ---------------------------------------------------------------------------------


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


HEADER = "#Insight Transform File V1.0\n"


@pytest.mark.parametrize(
    ("text", "error", "match"),
    [
        ("not a transform\n", ValueError, "not an Insight Transform File"),
        (HEADER, ValueError, "holds no transform"),
        (
            HEADER + "#Transform 0\nTransform: BSplineTransform_double_3_3\nParameters: 1\n"
            "FixedParameters: 1\n",
            NotImplementedError,
            "BSplineTransform in 3-D is not supported",
        ),
        (
            HEADER + "#Transform 0\nTransform: AffineDTI3DTransform_double_3_3\n"
            "Parameters: " + "0 " * 12 + "\nFixedParameters: 0 0 0\n",
            NotImplementedError,
            "AffineDTI3DTransform",
        ),
        (
            HEADER + "#Transform 0\nTransform: AffineTransform_double_3_3\nParameters: 1 0 0\n"
            "FixedParameters: 0 0 0\n",
            ValueError,
            "has 12 parameters, got 3",
        ),
        (
            HEADER + "#Transform 0\nTransform: TranslationTransform_double_3_3\n"
            "Parameters: 1 nan 0\nFixedParameters:\n",
            ValueError,
            "must be finite",
        ),
        (
            HEADER + "#Transform 0\nTransform: TranslationTransform_double_3_3\n"
            "Parameters: 1 x 0\nFixedParameters:\n",
            ValueError,
            "expected numbers",
        ),
        (
            HEADER + "#Transform 0\nTransform: AffineTransform_double_3_2\nParameters: 1\n"
            "FixedParameters: 0\n",
            ValueError,
            "different dimensions",
        ),
        (
            HEADER + "#Transform 0\nTransform: Affine\nParameters: 1\nFixedParameters: 0\n",
            ValueError,
            "not an ITK transform type",
        ),
        (
            HEADER + "#Transform 0\nTransform: TranslationTransform_double_3_3\n"
            "Parameters: 1 0 0\nFixedParameters:\n#Transform 1\n"
            "Transform: TranslationTransform_double_3_3\nParameters: 1 0 0\nFixedParameters:\n",
            ValueError,
            "without a CompositeTransform header",
        ),
        (
            HEADER + "#Transform 0\nTransform: CompositeTransform_double_3_3\n",
            ValueError,
            "composite holds no transforms",
        ),
        (
            HEADER + "#Transform 0\nTransform: TranslationTransform_double_3_3\nFixedParameters:\n",
            ValueError,
            "no 'Parameters:' line",
        ),
        (
            HEADER + "#Transform 0\nTransform: Euler3DTransform_double_3_3\n"
            "Parameters: 0 0 0 0 0 0\nFixedParameters: 0 0 0 2\n",
            ValueError,
            "ComputeZYX flag must be 0 or 1",
        ),
        (
            HEADER + "#Transform 0\nTransform: VersorRigid3DTransform_double_3_3\n"
            "Parameters: 1 1 0 0 0 0\nFixedParameters: 0 0 0\n",
            ValueError,
            "length at most 1",
        ),
        (
            HEADER + "#Transform 0\nTransform: Similarity3DTransform_double_3_3\n"
            "Parameters: 0 0 0 0 0 0 -1\nFixedParameters: 0 0 0\n",
            ValueError,
            "scale must be positive",
        ),
        (
            "#Insight Transform File V1.0\nParameters: 1\n",
            ValueError,
            "inside a '#Transform' block",
        ),
    ],
)
def test_malformed_or_unsupported_text_is_refused(
    tmp_path: Path, text: str, error: type[Exception], match: str
) -> None:
    fixed, moving = frames(LPS_MM)
    with pytest.raises(error, match=match):
        rfi.read_transform(write(tmp_path, "t.tfm", text), fixed=fixed, moving=moving)


def test_other_files_are_refused(tmp_path: Path) -> None:
    from scipy.io import savemat

    fixed, moving = frames(LPS_MM)
    with pytest.raises(ValueError, match="unknown suffix"):
        rfi.read_transform(write(tmp_path, "t.json", "{}"), fixed=fixed, moving=moving)
    savemat(
        str(tmp_path / "bad.mat"), {"AffineTransform_double_3_3": np.zeros((12, 1))}, format="4"
    )
    with pytest.raises(ValueError, match="one transform variable and 'fixed'"):
        rfi.read_transform(tmp_path / "bad.mat", fixed=fixed, moving=moving)
    with pytest.raises(FileNotFoundError):
        rfi.read_transform(tmp_path / "missing.tfm", fixed=fixed, moving=moving)


def test_hdf5_structure_is_checked(tmp_path: Path) -> None:
    import h5py

    fixed, moving = frames(LPS_MM)
    with h5py.File(tmp_path / "empty.h5", "w") as handle:
        handle.create_group("Other")
    with pytest.raises(ValueError, match="no TransformGroup"):
        rfi.read_transform(tmp_path / "empty.h5", fixed=fixed, moving=moving)
    with h5py.File(tmp_path / "field.h5", "w") as handle:
        member = handle.create_group("TransformGroup/0")
        member["TransformType"] = [b"DisplacementFieldTransform_double_3_3"]
        member["TransformParameters"] = np.zeros(10)
        member["TransformFixedParameters"] = np.array(
            [2, 2, 2, 0, 0, 0, 1, 1, 1, *np.eye(3).ravel()]
        )
    with pytest.raises(ValueError, match="needs 24 parameters, got 10"):
        rfi.read_transform(tmp_path / "field.h5", fixed=fixed, moving=moving)


# --- NIfTI fields -----------------------------------------------------------------------------


def nifti_field(tmp_path: Path, **header: Any) -> Path:
    import nibabel as nib

    image: Any = nib.load(str(DIRECTORY / "field_3d.nii.gz"))
    data = np.asarray(image.dataobj)
    copy: Any = nib.Nifti1Image(data, image.affine, image.header.copy())  # type: ignore[no-untyped-call]
    shape = header.pop("shape", None)
    if shape is not None:
        copy = nib.Nifti1Image(np.zeros(shape), image.affine, image.header.copy())  # type: ignore[no-untyped-call]
    for key, value in header.items():
        if key == "units":
            copy.header.set_xyzt_units(value)
        elif key == "qform":
            copy.set_qform(value, code=1)
        elif key == "codes":
            copy.set_sform(None, code=0)
            copy.set_qform(None, code=0)
        elif key == "intent":
            copy.header.set_intent(value)
    path = tmp_path / "field.nii.gz"
    nib.save(copy, str(path))
    return path


def test_nifti_geometry_rules(tmp_path: Path) -> None:
    fixed, moving = frames(LPS_MM)
    with pytest.raises(ValueError, match="the spatial unit is 'meter'"):
        rfi.read_transform(nifti_field(tmp_path, units="meter"), fixed=fixed, moving=moving)
    result = rfi.read_transform(nifti_field(tmp_path, units="unknown"), fixed=fixed, moving=moving)
    assert "nifti-units" in [code for code, _ in result.report]
    with pytest.raises(ValueError, match="neither sform nor qform is coded"):
        rfi.read_transform(nifti_field(tmp_path, codes=True), fixed=fixed, moving=moving)
    shifted = np.diag([1.5, 1.2, 2.0, 1.0])
    with pytest.raises(ValueError, match=r"sform .* and qform .* disagree"):
        rfi.read_transform(nifti_field(tmp_path, qform=shifted), fixed=fixed, moving=moving)
    with pytest.raises(ValueError, match="five-dimensional"):
        rfi.read_transform(nifti_field(tmp_path, shape=(6, 7, 5, 3)), fixed=fixed, moving=moving)
    result = rfi.read_transform(nifti_field(tmp_path, intent="none"), fixed=fixed, moving=moving)
    assert "nifti-intent" in [code for code, _ in result.report]


def test_a_2d_field_must_not_mix_z(tmp_path: Path) -> None:
    import nibabel as nib

    image: Any = nib.load(str(DIRECTORY / "field_2d.nii.gz"))
    affine = np.asarray(image.affine).copy()
    affine[0, 2] = 0.3
    nib.save(
        nib.Nifti1Image(np.asarray(image.dataobj), affine, image.header.copy()),  # type: ignore[no-untyped-call]
        str(tmp_path / "f.nii.gz"),
    )
    fixed, moving = frames(PLANE)
    with pytest.raises(ValueError, match="z axis must not mix"):
        rfi.read_transform(tmp_path / "f.nii.gz", fixed=fixed, moving=moving, space=PLANE)


def test_space_must_be_a_coordinate_system() -> None:
    fixed, moving = frames(LPS_MM)
    with pytest.raises(TypeError, match="space must be a CoordinateSystem"):
        rfi.read_transform(
            DIRECTORY / "affine_3d.tfm", fixed=fixed, moving=moving, space=cast(Any, "LPS")
        )
