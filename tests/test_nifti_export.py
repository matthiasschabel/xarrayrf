"""NIfTI header export through the public adapter."""

from __future__ import annotations

from typing import cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from xarrayrf import AffineTransform, ArrayCoordinates, CoordinateSystem, Geometry, ReferenceFrame
from xarrayrf.anatomy import LPS, RAS, patient_coordinate_system
from xarrayrf.nifti import from_header, to_header


def geometry(
    affine: npt.NDArray[np.float64] | None = None,
    *,
    orientation: tuple[str, str, str] = RAS,
    unit: str = "mm",
    shape: tuple[int, ...] = (4, 5, 6),
    time_unit: str | None = None,
    time_coupling: bool = False,
    time_step: float = 2.0,
) -> Geometry:
    spatial = patient_coordinate_system(orientation, unit)
    if time_unit is None:
        frame = ReferenceFrame.local(spatial)
    else:
        frame = ReferenceFrame.local(
            CoordinateSystem(
                (*spatial.axes, "t"),
                (*spatial.units, time_unit),
                axis_types=("space", "space", "space", "time"),
                vocabulary=spatial.vocabulary,
                orientation=(*orientation, None),
            )
        )
    dims = ("i", "j", "k", "t")[: len(shape)]
    array = xr.DataArray(
        np.zeros(shape),
        dims=dims,
        coords={
            dim: (dim, np.arange(size), {"units": "1"})
            for dim, size in zip(dims, shape, strict=True)
        },
    )
    spatial_affine = np.diag([1.5, 2.5, 3.5, 1.0]) if affine is None else affine
    matrix = np.zeros((len(shape), len(shape)))
    matrix[:3, :3] = spatial_affine[:3, :3]
    translation = np.zeros(len(shape))
    translation[:3] = spatial_affine[:3, 3]
    if len(shape) == 4:
        matrix[3, 3] = time_step
        translation[3] = 7.0
        if time_coupling:
            matrix[0, 3] = 1.0
    transform = AffineTransform(
        source=ArrayCoordinates(
            dims,
            ("1",) * len(dims),
            axis_types=("space", "space", "space") + (("time",) if len(dims) == 4 else ()),
            sample_offset=(0.5, 0.5, 0.5) + ((None,) if len(dims) == 4 else ()),
        ),
        target=frame,
        matrix=matrix,
        translation=translation,
    )
    return Geometry(array, transform, dims=dims)


def test_oblique_nifti1_round_trip_and_dyadic_exactness() -> None:
    affine = np.array(
        [[1.2, -0.3, 0.0, 10.2], [0.2, 1.7, -0.4, -5.1], [0.0, 0.1, 2.4, 3.3], [0, 0, 0, 1]]
    )
    source = geometry(affine)
    header, report = to_header(source, dims=("i", "j", "k"))
    assert report[0][0] == "qform-unrepresentable"
    assert int(header["sform_code"]) == 1
    assert int(header["qform_code"]) == 0
    assert_allclose(header.get_sform(), affine, rtol=0, atol=1e-6 * np.max(np.abs(affine)))  # type: ignore[no-untyped-call]
    imported = from_header(header)
    assert_allclose(
        imported.transform.matrix, affine[:3, :3], rtol=0, atol=1e-6 * np.max(np.abs(affine))
    )
    dyadic = np.diag([1.5, 2.5, 3.5, 1.0])
    dyadic[:3, 3] = [2.0, -4.0, 8.0]
    dyadic_header, report = to_header(geometry(dyadic), dims=("i", "j", "k"))
    assert not report
    assert int(dyadic_header["qform_code"]) == 1
    np.testing.assert_array_equal(dyadic_header.get_sform(), dyadic)  # type: ignore[no-untyped-call]
    np.testing.assert_array_equal(from_header(dyadic_header).transform.matrix, dyadic[:3, :3])


def test_nifti2_round_trip_uses_float64_precision() -> None:
    affine = np.array(
        [
            [1.123456789, 0, 0, 0.123456789],
            [0, 2.234567891, 0, -4.876543219],
            [0, 0, 3.345678912, 8.765432198],
            [0, 0, 0, 1],
        ]
    )
    header, _ = to_header(geometry(affine), dims=("i", "j", "k"), header=nib.Nifti2Header())  # type: ignore[no-untyped-call]
    assert isinstance(header, nib.Nifti2Header)
    assert_allclose(from_header(header).transform.matrix, affine[:3, :3], rtol=1e-12, atol=0)
    assert_allclose(header.get_sform(), affine, rtol=1e-12, atol=0)  # type: ignore[no-untyped-call]


@pytest.mark.parametrize("unit", ["m", "mm", "um"])
def test_lps_and_spatial_units_export_to_ras(unit: str) -> None:
    source = geometry(orientation=LPS, unit=unit)
    header, _ = to_header(source, dims=("k", "j", "i"), xform_code="aligned")
    expected = np.array([[0, 0, -1.5, 0], [0, -2.5, 0, 0], [3.5, 0, 0, 0], [0, 0, 0, 1]])
    assert_allclose(header.get_sform(), expected, rtol=0, atol=1e-6)  # type: ignore[no-untyped-call]
    assert int(header["xyzt_units"]) & 7 == {"m": 1, "mm": 2, "um": 3}[unit]
    assert int(header["qform_code"]) == 2


@pytest.mark.parametrize(("unit", "code"), [("s", 8), ("ms", 16), ("us", 24)])
def test_four_dimensional_time_and_header_preservation(unit: str, code: int) -> None:
    source = geometry(shape=(4, 5, 6, 7), time_unit=unit)
    header, _ = to_header(source, dims=("i", "j", "k", "t"))
    assert int(header["xyzt_units"]) == 2 | code
    assert_allclose([header["pixdim"][4], header["toffset"]], [2, 7], rtol=0, atol=1e-6)
    imported = from_header(header, time=True)
    assert_allclose(imported.transform.matrix[3, 3], 2, rtol=0, atol=1e-6)
    supplied = nib.Nifti1Header()  # type: ignore[no-untyped-call]
    supplied["pixdim"][4] = 3.0
    supplied["toffset"] = 9.0
    supplied["xyzt_units"] = 16
    preserved, _ = to_header(geometry(), dims=("i", "j", "k"), header=supplied)
    assert_allclose([preserved["pixdim"][4], preserved["toffset"]], [3, 9], rtol=0, atol=1e-6)
    assert int(preserved["xyzt_units"]) == 18
    assert_allclose(supplied["pixdim"][4], 3, rtol=0, atol=0)


def test_lattice_failures_are_chained() -> None:
    source = geometry()
    source.array.coords["k"] = [0, 1, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="NIfTI export requires a regular affine lattice") as exc:
        to_header(source, dims=("i", "j", "k"))
    assert isinstance(exc.value.__cause__, ValueError)
    assert "not uniformly spaced" in str(exc.value.__cause__)
    with pytest.raises(ValueError, match="NIfTI export requires a regular affine lattice"):
        to_header(geometry(shape=(4, 5, 1)), dims=("i", "j", "k"))


def test_qform_can_be_explicitly_omitted() -> None:
    header, report = to_header(geometry(), dims=("i", "j", "k"), qform=False)
    assert int(header["sform_code"]) == 1
    assert int(header["qform_code"]) == 0
    assert not report


@pytest.mark.parametrize("step", [1e-8, 1e-16])
def test_qform_accepts_tiny_nonzero_spatial_step(step: float) -> None:
    affine = np.diag([step, 1.0, 1.0, 1.0])
    header, report = to_header(geometry(affine), dims=("i", "j", "k"))
    assert not report
    assert int(header["qform_code"]) == 1


def test_non_affine_transform_is_refused_with_lattice_cause() -> None:
    source = geometry()

    class PointsOnly:
        def __init__(self, affine: AffineTransform) -> None:
            self.source = affine.source
            self.target = affine.target
            self.affine = affine

        def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return self.affine.transform_point(points)

    non_affine = Geometry(
        source.array, PointsOnly(cast(AffineTransform, source.transform)), dims=source.dims
    )
    with pytest.raises(ValueError, match="NIfTI export requires a regular affine lattice") as exc:
        to_header(non_affine, dims=("i", "j", "k"))
    assert isinstance(exc.value.__cause__, TypeError)
    assert "needs an affine transform" in str(exc.value.__cause__)


def test_coordinate_field_is_refused_with_lattice_cause() -> None:
    source = geometry()
    source_affine = cast(AffineTransform, source.transform)
    field = np.broadcast_to(np.arange(4)[:, None], (4, 5))
    array = source.array.assign_coords(field=(("i", "j"), field, {"units": "1"}))
    transformed = AffineTransform(
        source=ArrayCoordinates(
            ("field", "j", "k"),
            ("1", "1", "1"),
            sample_offset=(0.5, 0.5, 0.5),
        ),
        target=source.frame,
        matrix=source_affine.matrix,
        translation=source_affine.translation,
    )
    with pytest.raises(ValueError, match="NIfTI export requires a regular affine lattice") as exc:
        to_header(Geometry(array, transformed, dims=source.dims), dims=("i", "j", "k"))
    assert isinstance(exc.value.__cause__, ValueError)
    assert "multidimensional" in str(exc.value.__cause__)


def test_invalid_arguments_and_geometry_refused() -> None:
    source = geometry()
    source_affine = cast(AffineTransform, source.transform)
    with pytest.raises(TypeError, match="geometry must be a Geometry"):
        to_header(cast(Geometry, object()), dims=("i", "j", "k"))
    with pytest.raises(TypeError, match="dims must be a sequence"):
        to_header(source, dims=cast(tuple[str, ...], "ijk"))
    with pytest.raises(ValueError, match="dims must name three or four"):
        to_header(source, dims=("i", "j"))
    with pytest.raises(TypeError, match="qform must be a bool"):
        to_header(source, dims=("i", "j", "k"), qform=cast(bool, 1))
    with pytest.raises(TypeError, match="header must be a NIfTI-1 or NIfTI-2"):
        to_header(source, dims=("i", "j", "k"), header=cast(nib.Nifti1Header, object()))
    with pytest.raises(TypeError, match="xform_code must be a coded NIfTI space name"):
        to_header(source, dims=("i", "j", "k"), xform_code=cast(str, 1))
    with pytest.raises(ValueError, match="xform_code must name a known NIfTI space"):
        to_header(source, dims=("i", "j", "k"), xform_code="bogus")
    with pytest.raises(ValueError, match="xform_code must name a coded NIfTI space"):
        to_header(source, dims=("i", "j", "k"), xform_code="unknown")
    with pytest.raises(ValueError, match="regular affine lattice"):
        to_header(source, dims=("i", "j", "k", "t"))
    with pytest.raises(ValueError, match="requires spatial units m, mm or um"):
        to_header(geometry(unit="cm"), dims=("i", "j", "k"))
    with pytest.raises(ValueError, match="requires a fourth time axis"):
        to_header(geometry(shape=(4, 5, 6, 7), time_unit="Hz"), dims=("i", "j", "k", "t"))
    with pytest.raises(ValueError, match="requires zero space-time cross terms"):
        to_header(
            geometry(shape=(4, 5, 6, 7), time_unit="s", time_coupling=True),
            dims=("i", "j", "k", "t"),
        )
    with pytest.raises(ValueError, match="requires a positive finite time step"):
        to_header(
            geometry(shape=(4, 5, 6, 7), time_unit="s", time_step=-2),
            dims=("i", "j", "k", "t"),
        )
    timed = geometry(shape=(4, 5, 6, 7), time_unit="s")
    extra_axis = AffineTransform(
        source=source_affine.source,
        target=timed.frame,
        matrix=np.vstack((source_affine.matrix, np.zeros(3))),
        translation=np.zeros(4),
    )
    with pytest.raises(ValueError, match="requires 3 frame axes for 3 dims"):
        to_header(Geometry(source.array, extra_axis, dims=source.dims), dims=("i", "j", "k"))
    four_axis = AffineTransform(
        source=timed.transform.source,
        target=timed.frame,
        matrix=np.asarray(cast(AffineTransform, timed.transform).matrix),
        translation=cast(AffineTransform, timed.transform).translation,
    )
    spatial_time = np.array(four_axis.matrix, copy=True)
    spatial_time[3, 0] = 0.5
    coupled = AffineTransform(
        source=four_axis.source,
        target=four_axis.target,
        matrix=spatial_time,
        translation=four_axis.translation,
    )
    with pytest.raises(ValueError, match="requires zero space-time cross terms"):
        to_header(Geometry(timed.array, coupled, dims=timed.dims), dims=("i", "j", "k", "t"))
    with pytest.raises(ValueError, match="requires a frame convertible to RAS"):
        unoriented = ReferenceFrame.local(CoordinateSystem(("x", "y", "z"), ("mm",) * 3))
        changed = AffineTransform(
            source=source.transform.source,
            target=unoriented,
            matrix=source_affine.matrix,
            translation=source_affine.translation,
        )
        to_header(Geometry(source.array, changed, dims=source.dims), dims=("i", "j", "k"))


def test_oblique_nifti1_round_trip_keeps_the_qform() -> None:
    """A rotation read back from float32 storage is still written as a qform."""
    angle = 0.3
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([0.7, 0.9, 3.1])
    affine[:3, 3] = [11.3, -7.7, 40.1]
    source = nib.Nifti1Header()  # type: ignore[no-untyped-call]
    source.set_data_shape((4, 5, 6))  # type: ignore[no-untyped-call]
    source["xyzt_units"] = 2
    source.set_sform(affine, code=1)  # type: ignore[no-untyped-call]
    imported = from_header(source)
    array = xr.DataArray(np.zeros((4, 5, 6)), dims=imported.dims, coords=imported.coords)
    exported, report = to_header(
        Geometry(array, imported.transform, dims=imported.dims), dims=imported.dims
    )
    assert report == ()
    assert int(exported["qform_code"]) == 1
    # float32 header storage: about 1e-7 relative on coefficients of order 40.
    qform = np.asarray(exported.get_qform(), dtype=np.float64)  # type: ignore[no-untyped-call]
    assert_allclose(qform, affine, rtol=0, atol=1e-5)
