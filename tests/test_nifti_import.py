"""NIfTI header import through the public adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import dask.array as da
import nibabel as nib
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from numpy.testing import assert_allclose

from xarrayrf import (
    ArrayCoordinates,
    CoordinateSystem,
    DirectionVocabulary,
    Geometry,
    ReferenceFrame,
    coordinate_system_change,
)
from xarrayrf.anatomy import LPS, RAS, patient_coordinate_system
from xarrayrf.nifti import from_header, to_dataarray

ATOL = 1e-6  # NIfTI-1 stores sform coefficients as float32.


def test_to_dataarray_binds_header_geometry() -> None:
    geometry = from_header(header())
    pixels = np.arange(120).reshape(4, 5, 6)
    array = to_dataarray(geometry, pixels)
    assert array.data is pixels
    assert array.rf.coordinate_transform == geometry.transform
    assert array.rf.geometry_dims == geometry.dims
    for i, j, k in ((0, 0, 0), (1, 2, 3), (3, 4, 5)):
        assert_allclose(
            np.asarray(array.rf.geometry.point_at(i=i, j=j, k=k)),
            geometry.transform.transform_point([i, j, k]),
            rtol=0,
            atol=ATOL,
        )
    with pytest.raises(ValueError, match=r"data shape \(4, 5, 5\).*geometry shape \(4, 5, 6\)"):
        to_dataarray(geometry, np.zeros((4, 5, 5)))


def test_to_dataarray_keeps_dask_pixels_lazy() -> None:
    geometry = from_header(header())
    pixels = da.ones((4, 5, 6), chunks=(2, 2, 3))
    tasks: list[object] = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        array = to_dataarray(geometry, pixels)
    assert not tasks
    assert array.data is pixels


@pytest.mark.parametrize("pixels", [np.zeros((4, 5, 6)).tolist(), 1])
def test_to_dataarray_refuses_non_duck_pixels(pixels: object) -> None:
    geometry = from_header(header())
    with pytest.raises(TypeError, match=r"duck array.*np.asarray"):
        to_dataarray(geometry, pixels)  # type: ignore[arg-type]


def header(
    *,
    shape: tuple[int, ...] = (4, 5, 6),
    kind: type[nib.Nifti1Header] = nib.Nifti1Header,
    sform_code: int = 2,
    qform_code: int = 0,
    unit_code: int = 2,
) -> nib.Nifti1Header:
    result = kind()
    result.set_data_shape(shape)  # type: ignore[no-untyped-call]
    result["xyzt_units"] = unit_code
    result.set_sform(np.diag([2.0, 3.0, 4.0, 1.0]), code=sform_code)  # type: ignore[no-untyped-call]
    result.set_qform(np.diag([5.0, 6.0, 7.0, 1.0]), code=qform_code)  # type: ignore[no-untyped-call]
    return result


@pytest.mark.parametrize(
    ("s_code", "q_code", "choice", "expected", "scale"),
    [
        (2, 1, "best", "sform", 2.0),
        (0, 1, "best", "qform", 5.0),
        (2, 0, "best", "sform", 2.0),
        (2, 1, "sform", "sform", 2.0),
        (2, 1, "qform", "qform", 5.0),
    ],
)
def test_xform_precedence_and_selection(
    s_code: int, q_code: int, choice: str, expected: str, scale: float
) -> None:
    result = from_header(header(sform_code=s_code, qform_code=q_code), xform=choice)  # type: ignore[arg-type]
    assert result.frame.definition["xform"] == expected
    assert any(
        code == "xform" and f"selected {expected}" in message for code, message in result.report
    )
    assert_allclose(result.transform.matrix[0, 0], scale, rtol=0, atol=ATOL)


@pytest.mark.parametrize("code,space", [(3, "Talairach"), (4, "MNI152")])
def test_coded_templates_share_canonical_spatial_identity(code: int, space: str) -> None:
    first = header(sform_code=code, qform_code=0)
    second = header(sform_code=code, qform_code=1)
    a = from_header(first)
    b = from_header(second)
    assert a.frame == b.frame
    assert a.frame.identifier == ("nifti-template", space)
    assert a.frame.definition == {"space": space, "variant": "unspecified"}
    assert a.frame.context == {}
    assert any("qform_code=1" in text for kind, text in b.report if kind == "xform")


def test_template_names_and_frame_override() -> None:
    source = header(sform_code=4)
    named = from_header(source, template="MNI152NLin2009cAsym")
    other = from_header(source, template="MNI152NLin6Asym")
    assert named.frame.identifier == ("templateflow", "MNI152NLin2009cAsym")
    assert named.frame.definition == {"space": "MNI152NLin2009cAsym"}
    assert named.frame != other.frame
    assert named.frame != from_header(source).frame
    framed = to_dataarray(named, np.zeros((4, 5, 6)))
    other_array = to_dataarray(other, np.zeros((4, 5, 6)))
    with pytest.raises(ValueError, match=r"conflicting indexes|incompatible"):
        _ = framed + other_array
    assert from_header(source, frame=framed).frame == named.frame
    with pytest.raises(ValueError, match="cannot both"):
        from_header(source, frame=named.frame, template="NMT31Sym")
    with pytest.raises(ValueError, match="code 2"):
        from_header(header(sform_code=1), template="NMT31Sym")
    with pytest.raises(ValueError, match="BIDS space label"):
        from_header(source, template="not-valid")


def test_template_coded_time_series_remain_acquisition_local() -> None:
    source = header(shape=(4, 5, 6, 2), sform_code=4, unit_code=2 | 8)
    first = from_header(source, time=True)
    second = from_header(source, time=True)
    assert first.frame != second.frame
    assert first.frame.identifier[0] == "xarrayrf.local"
    first_array = to_dataarray(first, np.zeros((4, 5, 6, 2)))
    second_array = to_dataarray(second, np.zeros((4, 5, 6, 2)))
    with pytest.raises(ValueError, match=r"conflicting indexes|incompatible"):
        _ = first_array + second_array
    with pytest.raises(ValueError, match="spacetime frame"):
        from_header(source, time=True, template="MNI152NLin2009cAsym")


@pytest.mark.parametrize(
    ("s_code", "q_code", "choice"),
    [
        (0, 0, "best"),
        (0, 1, "sform"),
        (2, 0, "qform"),
    ],
)
def test_uncoded_xform_refused(s_code: int, q_code: int, choice: str) -> None:
    with pytest.raises(ValueError, match="requires a coded sform or qform"):
        from_header(header(sform_code=s_code, qform_code=q_code), xform=choice)  # type: ignore[arg-type]


def test_qfac_negative_is_applied_by_nibabel() -> None:
    source = header(sform_code=0, qform_code=1)
    source.set_qform(np.diag([2.0, 3.0, -4.0, 1.0]), code=1)  # type: ignore[no-untyped-call]
    result = from_header(source)
    assert_allclose(result.transform.matrix, np.diag([2.0, 3.0, -4.0]), rtol=0, atol=ATOL)
    assert result.frame.definition["qfac"] == pytest.approx(-1.0, abs=ATOL)


def test_nifti2_header_uses_the_same_import_contract() -> None:
    result = from_header(header(kind=nib.Nifti2Header))
    assert result.dims == ("i", "j", "k")
    assert_allclose(result.transform.matrix, np.diag([2.0, 3.0, 4.0]), rtol=0, atol=1e-12)


@pytest.mark.parametrize(("code", "unit"), [(1, "m"), (2, "mm"), (3, "um")])
def test_spatial_unit_codes(code: int, unit: str) -> None:
    result = from_header(header(unit_code=code))
    assert result.frame.units == (unit,) * 3


@pytest.mark.parametrize(
    ("code", "unit"),
    [
        (8, "s"),
        (16, "ms"),
        (24, "us"),
        (32, "Hz"),
        (40, "ppm"),
        (48, "rad/s"),
    ],
)
def test_temporal_unit_codes_as_non_geometry_coordinates(code: int, unit: str) -> None:
    source = header(shape=(2, 3, 4, 5), unit_code=2 | code)
    source["pixdim"][4] = 2
    source["toffset"] = 7
    result = from_header(source)
    assert result.frame.axes == ("x", "y", "z")
    assert result.coords["t"][2]["units"] == unit
    assert_allclose(result.coords["t"][1], [7, 9, 11, 13, 15], rtol=0, atol=ATOL)
    assert ("spectral-time-unit" in {entry[0] for entry in result.report}) == (code >= 32)


def test_unknown_spatial_unit_uses_policy_or_refuses() -> None:
    source = header(unit_code=0)
    result = from_header(source, spatial_unit="um")
    assert result.frame.units == ("um",) * 3
    assert "unknown-spatial-unit" in {code for code, _ in result.report}
    assert from_header(source).frame.units == ("mm",) * 3
    with pytest.raises(ValueError, match="supply spatial_unit"):
        from_header(source, spatial_unit=None)


def test_supplied_lps_frame_keeps_identity_and_maps_same_points() -> None:
    source = header()
    ras = from_header(source)
    lps_frame = ReferenceFrame.declared(
        ("dicom-frame-of-reference", "1.2.3.4"), patient_coordinate_system(LPS, "mm")
    )
    lps = from_header(source, frame=lps_frame)
    change = coordinate_system_change(
        lps_frame, lps_frame.with_coordinate_system(patient_coordinate_system(RAS, "mm"))
    )
    point = np.array([1.0, 2.0, 3.0])
    assert lps.frame is lps_frame
    assert lps.transform.target is lps_frame
    assert_allclose(
        change.transform_point(lps.transform.transform_point(point)),
        ras.transform.transform_point(point),
        rtol=0,
        atol=ATOL,
    )
    assert "xform" in {code for code, _ in lps.report}


def test_supplied_ras_frame_is_the_transform_target_object() -> None:
    frame = ReferenceFrame.local(patient_coordinate_system(RAS, "mm"))
    result = from_header(header(), frame=frame)
    assert result.frame is frame
    assert result.transform.target is frame


def test_time_geometry_and_five_dimensional_header() -> None:
    source = header(shape=(2, 3, 4, 5, 6), unit_code=2 | 8)
    source["pixdim"][4] = 2
    source["toffset"] = 7
    result = from_header(source, time=True)
    assert result.dims == ("i", "j", "k", "t", "u")
    assert result.frame.axes == ("x", "y", "z", "t")
    assert result.frame.coordinate_system.axis_types == ("space", "space", "space", "time")
    assert isinstance(result.transform.source, ArrayCoordinates)
    assert result.transform.source.sample_offset == (0.5, 0.5, 0.5, None)
    assert_allclose(
        result.transform.transform_point([1, 2, 3, 4]), [2, 6, 12, 15], rtol=0, atol=ATOL
    )
    assert result.coords["u"][2]["units"] == "1"


def test_two_dimensional_header_retains_scalar_k() -> None:
    result = from_header(header(shape=(4, 5)))
    assert result.dims == ("i", "j")
    assert result.coords["k"][0] == ()
    assert result.coords["k"][1] == 0
    assert result.transform.source.axes == ("i", "j", "k")
    array = xr.DataArray(np.zeros((4, 5)), dims=result.dims, coords=result.coords)
    point = Geometry(array, result.transform, dims=result.dims).point_at(i=1, j=2)
    assert_allclose(np.asarray(point), [2, 6, 0], rtol=0, atol=ATOL)


def test_geometry_maps_voxel_through_selected_affine() -> None:
    result = from_header(header())
    array = xr.DataArray(np.zeros((4, 5, 6)), dims=result.dims, coords=result.coords)
    point = Geometry(array, result.transform, dims=result.dims).point_at(i=1, j=2, k=3)
    assert_allclose(np.asarray(point), [2, 6, 12], rtol=0, atol=ATOL)


@pytest.mark.parametrize("step", [0.0, float("nan")])
def test_invalid_time_step_gives_an_index_coordinate(step: float) -> None:
    source = header(shape=(2, 3, 4, 5), unit_code=2 | 8)
    source["pixdim"][4] = step
    result = from_header(source)
    np.testing.assert_array_equal(result.coords["t"][1], np.arange(5))
    assert result.coords["t"][2]["units"] == "1"
    assert "invalid-time-step" in {code for code, _ in result.report}


@pytest.mark.parametrize("xform", ["sform", "qform"])
def test_oblique_affine_with_translation_maps_voxels(xform: str) -> None:
    rotation = np.array([[0.0, -1.0, 0.0], [0.6, 0.0, 0.8], [-0.8, 0.0, 0.6]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([1.5, 2.0, 2.5])
    affine[:3, 3] = [10.0, -20.0, 30.0]
    source = nib.Nifti1Header()  # type: ignore[no-untyped-call]
    source.set_data_shape((4, 5, 6))  # type: ignore[no-untyped-call]
    source["xyzt_units"] = 2
    source.set_sform(affine, code=1 if xform == "sform" else 0)  # type: ignore[no-untyped-call]
    source.set_qform(affine, code=1 if xform == "qform" else 0)  # type: ignore[no-untyped-call]
    result = from_header(source)
    array = xr.DataArray(np.zeros((4, 5, 6)), dims=result.dims, coords=result.coords)
    point = Geometry(array, result.transform, dims=result.dims).point_at(i=1, j=2, k=3)
    # float32 header storage: about 1e-7 relative on coefficients of order 30.
    assert_allclose(np.asarray(point), (affine @ [1, 2, 3, 1])[:3], rtol=0, atol=1e-5)


def test_disagreement_and_spacing_mismatch_are_reported() -> None:
    result = from_header(header(sform_code=2, qform_code=1))
    codes = {code for code, _ in result.report}
    assert "sform-qform-disagree" in codes
    assert "pixdim-spacing-mismatch" in codes


def test_xform_disagreement_uses_voxel_relative_tolerance_below_one_unit() -> None:
    source = header(sform_code=2, qform_code=1)
    source.set_sform(np.diag([0.001, 0.001, 0.001, 1.0]), code=2)  # type: ignore[no-untyped-call]
    source.set_qform(np.diag([0.00100001, 0.001, 0.001, 1.0]), code=1)  # type: ignore[no-untyped-call]
    result = from_header(source)
    assert "sform-qform-disagree" in {code for code, _ in result.report}


@pytest.mark.parametrize(
    ("modifier", "error", "message"),
    [
        (lambda h: h.set_data_shape((4,)), ValueError, "2 to 7 dimensions"),
        (lambda h: h.__setitem__("xyzt_units", 7), ValueError, "unsupported spatial code"),
        (lambda h: h.__setitem__("xyzt_units", 2 | 56), ValueError, "unsupported temporal code"),
    ],
)
def test_invalid_header_fields_raise(
    modifier: Callable[[nib.Nifti1Header], object], error: type[Exception], message: str
) -> None:
    source = header()
    modifier(source)
    with pytest.raises(error, match=message):
        from_header(source)


def test_public_argument_errors() -> None:
    source = header()
    with pytest.raises(TypeError, match="header must be a NIfTI-1 or NIfTI-2 header"):
        from_header(cast(nib.Nifti1Header, object()))
    with pytest.raises(TypeError, match="frame must be a ReferenceFrame"):
        from_header(source, frame=cast(ReferenceFrame, "bad"))
    with pytest.raises(ValueError, match="xform must be"):
        from_header(source, xform=cast(Literal["best", "sform", "qform"], "bad"))
    with pytest.raises(ValueError, match="spatial_unit must be"):
        from_header(source, spatial_unit="")
    with pytest.raises(TypeError, match="spatial_unit must be a unit string or None"):
        from_header(source, spatial_unit=cast(str, 3))
    with pytest.raises(TypeError, match="time must be a bool"):
        from_header(source, time=cast(bool, 1))


def test_time_and_supplied_frame_errors() -> None:
    source = header(shape=(2, 3, 4, 5), unit_code=2 | 8)
    with pytest.raises(ValueError, match="time=True requires a fourth dimension and a time unit"):
        from_header(header(), time=True)
    with pytest.raises(ValueError, match="time=True requires a fourth dimension and a time unit"):
        from_header(header(shape=(2, 3, 4, 5), unit_code=2 | 32), time=True)
    with pytest.raises(ValueError, match="positive finite pixdim\\[4\\]"):
        source["pixdim"][4] = 0
        from_header(source, time=True)
    source["pixdim"][4] = 1
    lps = ReferenceFrame.local(patient_coordinate_system(LPS, "mm"))
    with pytest.raises(ValueError, match="time axis must agree"):
        from_header(source, frame=lps, time=True)
    timed = ReferenceFrame.local(
        CoordinateSystem(
            ("x", "y", "z", "t"),
            ("mm", "mm", "mm", "s"),
            axis_types=("space", "space", "space", "time"),
            vocabulary=patient_coordinate_system(RAS, "mm").vocabulary,
            orientation=(*RAS, None),
        )
    )
    with pytest.raises(ValueError, match="time axis must agree"):
        from_header(source, frame=timed)
    unknown = ReferenceFrame.local(CoordinateSystem(("x", "y", "z"), ("mm",) * 3))
    with pytest.raises(ValueError, match="cannot be derived from NIfTI RAS"):
        from_header(source, frame=unknown)
    wrong_unit = ReferenceFrame.local(patient_coordinate_system(LPS, "m"))
    with pytest.raises(ValueError, match="cannot be derived from NIfTI RAS"):
        from_header(source, frame=wrong_unit)
    other_vocabulary = ReferenceFrame.local(
        CoordinateSystem(
            ("x", "y", "z"),
            ("mm",) * 3,
            axis_types=("space",) * 3,
            vocabulary=DirectionVocabulary("other", LPS),
            orientation=LPS,
        )
    )
    with pytest.raises(ValueError, match="cannot be derived from NIfTI RAS"):
        from_header(source, frame=other_vocabulary)
