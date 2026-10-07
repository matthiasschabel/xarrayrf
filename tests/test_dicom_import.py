"""DICOM import from synthetic metadata-only pydicom datasets."""

from __future__ import annotations

from dataclasses import replace

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from dask.callbacks import Callback
from numpy.testing import assert_allclose
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence

from xarrayrf import AffineTransform, ArrayCoordinates, Geometry
from xarrayrf.anatomy import cardinal_grid
from xarrayrf.dicom import DicomGeometry, from_datasets, from_enhanced, patient_frame, to_dataarray

ATOL = 1e-9  # Synthetic Decimal Strings use exact short decimal inputs.


def test_patient_frame_shares_declared_uid_and_mints_missing_uid() -> None:
    assert patient_frame("1.2.3") == patient_frame("1.2.3")
    assert patient_frame("1.2.3").identifier == ("dicom-frame-of-reference", "1.2.3")
    assert patient_frame(None) != patient_frame(None)
    with pytest.raises(TypeError, match="string or None"):
        patient_frame(3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonempty"):
        patient_frame("")


def test_duplicate_slice_message_directs_multidimensional_assembly() -> None:
    with pytest.raises(
        ValueError, match=r"several images share a slice position.*application-level"
    ):
        from_datasets([image(0), image(0)])


def test_patient_grid_resamples_exactly_between_index_and_mm_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    from scipy import ndimage

    geometry = from_datasets([image(0), image(2), image(4)])
    values = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
    indexed = to_dataarray(geometry, values)
    steps = np.array([2.0, 2.0, 3.0])
    millimetres = xr.DataArray(
        values.copy(),
        dims=("z", "y", "x"),
        coords={
            "z": np.arange(3) * steps[0],
            "y": np.arange(4) * steps[1],
            "x": np.arange(5) * steps[2],
        },
    )
    offset_transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("z", "y", "x"), ("mm",) * 3),
        target=patient_frame("1.2.3.4"),
        matrix=geometry.transform.matrix / steps,
        translation=geometry.transform.translation,
    )
    millimetres = millimetres.rf.frame(offset_transform, dims=("z", "y", "x"))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("same-grid resampling interpolated")

    monkeypatch.setattr(ndimage, "affine_transform", forbidden)
    onto_index = millimetres.rf.resample_to(indexed)
    onto_mm = indexed.rf.resample_to(millimetres)
    assert onto_index.rf.reference_frame == indexed.rf.reference_frame
    assert onto_mm.rf.reference_frame == millimetres.rf.reference_frame
    assert_allclose(onto_index.values, values, rtol=0, atol=1e-12)
    assert_allclose(onto_mm.values, values, rtol=0, atol=1e-12)


R = np.array([0.8, 0.6, 0.0])
C = np.array([-0.36, 0.48, 0.8])
N = np.cross(R, C)
ORIGIN = np.array([10.0, 20.0, 30.0])


def test_oblique_dicom_cardinal_grid_resamples_linear_data_and_masks_outside() -> None:
    geometry = from_datasets([image(0), image(2), image(4)])
    k, j, i = np.indices((3, 4, 5), dtype=float)
    points = ORIGIN + 2 * k[..., None] * N + 2 * j[..., None] * C + 3 * i[..., None] * R
    coefficients = np.array([2.0, -3.0, 0.5])
    source = to_dataarray(geometry, 7 + points @ coefficients)
    target = cardinal_grid(source.rf.grid, "axial", spacing=1)
    pytest.importorskip("scipy", minversion="1.18")
    result = source.rf.resample_to(target)
    target_points = target.points()
    displacement = target_points - ORIGIN
    positions = np.stack(
        (displacement @ N / 2, displacement @ C / 2, displacement @ R / 3), axis=-1
    )
    inside = np.all((positions >= -ATOL) & (positions <= np.array([2, 3, 4]) + ATOL), axis=-1)
    assert inside.any() and (~inside).any()
    np.testing.assert_array_equal(np.isnan(result.values), ~inside)
    assert_allclose(
        result.values[inside], (7 + target_points @ coefficients)[inside], rtol=0, atol=ATOL
    )
    assert not target.intervals
    assert result.rf.grid == target


def test_thick_dicom_slice_cardinal_resampling_keeps_intervals_and_cell_mask() -> None:
    dataset = image(7, thickness=4)
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [10, 20, 37]
    geometry = from_datasets([dataset])
    _, j, i = np.indices((1, 4, 5), dtype=float)
    values = 7 + 2 * (10 + 3 * i) - 3 * (20 + 2 * j)
    source = to_dataarray(geometry, values)
    target = cardinal_grid(source.rf.grid, "axial", spacing=(10, 1.3, 1.3))
    pytest.importorskip("scipy", minversion="1.18")
    result = source.rf.resample_to(target, domain="cells")
    points = target.points()
    positions = (points - [10, 20, 37]) / [3, 2, 1]
    inside = np.all(
        (positions >= np.array([-0.5, -0.5, -2]) - ATOL)
        & (positions <= np.array([4.5, 3.5, 2]) + ATOL),
        axis=-1,
    )
    assert inside.any() and (~inside).any()
    np.testing.assert_array_equal(np.isnan(result.values), ~inside)
    # Cells outside the sample hull hold the edge sample's value.
    expected = (
        7
        + 2 * (10 + 3 * np.clip(positions[..., 0], 0, 4))
        - 3 * (20 + 2 * np.clip(positions[..., 1], 0, 3))
    )
    assert_allclose(result.values[inside], expected[inside], rtol=0, atol=ATOL)
    assert target.sizes["k"] == 1
    assert set(target.intervals) == {"k"}
    assert_allclose(target.intervals["k"], [[-0.2, 0.2]], rtol=0, atol=ATOL)
    assert_allclose(points[..., 2], 37, rtol=0, atol=ATOL)
    assert result.rf.grid == target


def test_to_dataarray_orders_classic_pixels_with_positions() -> None:
    geometry = from_datasets([image(4), image(0), image(2)])
    source = np.stack([np.full((4, 5), value) for value in (4, 0, 2)])
    array = to_dataarray(geometry, source)
    assert array.rf.coordinate_transform == geometry.transform
    assert array.rf.geometry_dims == geometry.dims
    for k, offset in enumerate((0, 2, 4)):
        np.testing.assert_array_equal(array.isel(k=k).data, offset)
        assert_allclose(
            np.asarray(array.rf.geometry.point_at(k=k, j=0, i=0)),
            ORIGIN + offset * N,
            rtol=0,
            atol=ATOL,
        )
    assert "slice_intervals" not in array.coords


@pytest.mark.parametrize("count", [1, 3])
def test_to_dataarray_requires_exact_classic_source_count(count: int) -> None:
    geometry = from_datasets([image(0), image(2)])
    with pytest.raises(ValueError, match="requires exactly 2 source slices"):
        to_dataarray(geometry, np.zeros((count, 4, 5)))


@pytest.mark.parametrize("count", [2, 4])
def test_enhanced_selection_requires_full_original_stack(count: int) -> None:
    geometry = from_enhanced(enhanced([0, 2, 4]), frame_indices=[0, 1])
    with pytest.raises(ValueError, match="requires exactly 3 source slices"):
        to_dataarray(geometry, np.zeros((count, 4, 5)))


def test_source_count_validation_does_not_evaluate_lazy_pixels() -> None:
    geometry = from_datasets([image(0), image(2)])
    pixels = da.zeros((3, 4, 5), chunks=(1, 4, 5))
    tasks: list[object] = []
    with (
        Callback(pretask=lambda key, *args: tasks.append(key)),  # type: ignore[no-untyped-call]
        pytest.raises(ValueError, match="requires exactly 2 source slices"),
    ):
        to_dataarray(geometry, pixels)
    assert not tasks


def test_manual_geometry_without_source_count_keeps_index_validation() -> None:
    imported = from_datasets([image(0), image(2)])
    manual = DicomGeometry(
        imported.dims,
        imported.coords,
        imported.transform,
        imported.frame,
        imported.order,
        imported.patient_position,
        imported.slice_intervals,
        imported.report,
    )
    assert manual.source_count is None
    assert to_dataarray(manual, np.zeros((3, 4, 5))).shape == (2, 4, 5)
    with pytest.raises(ValueError, match="slice axis 0 requires index 1"):
        to_dataarray(manual, np.zeros((1, 4, 5)))


@pytest.mark.parametrize("count,error", [(True, TypeError), ("2", TypeError), (0, ValueError)])
def test_source_count_requires_a_positive_integer(count: object, error: type[Exception]) -> None:
    imported = from_datasets([image(0), image(2)])
    with pytest.raises(error, match="source_count must be a positive integer or None"):
        replace(imported, source_count=count)  # type: ignore[arg-type]


@pytest.mark.parametrize("order", [(-1, 1), (0, 2)])
def test_explicit_source_count_must_contain_selected_indices(order: tuple[int, int]) -> None:
    imported = from_datasets([image(0), image(2)])
    with pytest.raises(ValueError, match="order must contain source indices within source_count"):
        replace(imported, order=order)


def test_to_dataarray_selects_original_enhanced_frames() -> None:
    geometry = from_enhanced(enhanced([0, 0, 2]), frame_indices=[2, 1])
    source = np.stack([np.full((4, 5), value) for value in (10, 20, 30)])
    array = to_dataarray(geometry, source)
    assert geometry.order == (1, 2)
    assert array.rf.coordinate_transform == geometry.transform
    np.testing.assert_array_equal(array[:, 0, 0].data, [20, 30])
    assert_allclose(
        np.asarray(array.rf.geometry.point_at(k=1, j=0, i=0)),
        ORIGIN + 2 * N,
        rtol=0,
        atol=ATOL,
    )
    with pytest.raises(ValueError, match=r"data shape \(2, 4, 5\).*geometry shape \(2, 4, 5\)"):
        to_dataarray(geometry, source[:2])
    with pytest.raises(ValueError, match=r"data shape \(3, 3, 5\).*geometry shape \(2, 4, 5\)"):
        to_dataarray(geometry, np.zeros((3, 3, 5)))


def test_to_dataarray_reorders_dask_stack_lazily() -> None:
    geometry = from_datasets([image(4), image(0), image(2)])
    source = da.from_array(  # type: ignore[no-untyped-call]
        np.arange(3 * 4 * 5).reshape(3, 4, 5), chunks=(1, 4, 5)
    )
    tasks: list[object] = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        array = to_dataarray(geometry, source)
    assert not tasks
    assert isinstance(array.data, da.Array)
    np.testing.assert_array_equal(array[:, 0, 0].compute().data, [20, 40, 0])


@pytest.mark.parametrize("pixels", [np.zeros((1, 4, 5)).tolist(), 1])
def test_to_dataarray_refuses_non_duck_pixels(pixels: object) -> None:
    geometry = from_datasets([image(0)])
    with pytest.raises(TypeError, match=r"duck array.*np.asarray"):
        to_dataarray(geometry, pixels)  # type: ignore[arg-type]


def image(offset: float, *, thickness: float | None = 2.0) -> Dataset:
    result = Dataset()
    result.ImagePositionPatient = [float(v) for v in ORIGIN + offset * N]
    result.ImageOrientationPatient = [*R, *C]
    result.PixelSpacing = [2.0, 3.0]
    result.Rows = 4
    result.Columns = 5
    result.FrameOfReferenceUID = "1.2.3.4"
    result.PatientPosition = "HFS"
    if thickness is not None:
        result.SliceThickness = thickness
    return result


def point(result: DicomGeometry, k: int = 0, j: int = 0, i: int = 0) -> npt.NDArray[np.float64]:
    array = xr.DataArray(
        np.zeros((len(result.order), 4, 5)), dims=result.dims, coords=result.coords
    )
    return np.asarray(Geometry(array, result.transform, dims=result.dims).point_at(k=k, j=j, i=i))


def enhanced(offsets: list[float], *, shared: bool = False) -> Dataset:
    result = Dataset()
    result.NumberOfFrames = len(offsets)
    result.Rows = 4
    result.Columns = 5
    result.FrameOfReferenceUID = "1.2.3.4"
    result.PatientPosition = "HFS"
    result.SharedFunctionalGroupsSequence = DicomSequence([Dataset()])
    frames = []
    for offset in offsets:
        groups = Dataset()
        source = image(offset)
        position = Dataset()
        position.ImagePositionPatient = source.ImagePositionPatient
        orientation = Dataset()
        orientation.ImageOrientationPatient = source.ImageOrientationPatient
        measures = Dataset()
        measures.PixelSpacing = source.PixelSpacing
        measures.SliceThickness = source.SliceThickness
        for name, item in (
            ("PlanePositionSequence", position),
            ("PlaneOrientationSequence", orientation),
            ("PixelMeasuresSequence", measures),
        ):
            if shared:
                setattr(result.SharedFunctionalGroupsSequence[0], name, DicomSequence([item]))
            else:
                setattr(groups, name, DicomSequence([item]))
        frames.append(groups)
    result.PerFrameFunctionalGroupsSequence = DicomSequence(frames)
    return result


def test_shuffled_oblique_stack_and_spacing_order() -> None:
    result = from_datasets([image(4), image(0), image(2)])
    assert result.dims == ("k", "j", "i")
    assert result.order == (1, 2, 0)
    assert result.transform.source.axes == ("k", "j", "i")
    assert result.frame.identifier == ("dicom-frame-of-reference", "1.2.3.4")
    assert dict(result.frame.context) == {}
    assert result.patient_position == "HFS"
    assert_allclose(point(result, 2, 1, 1), ORIGIN + 4 * N + 2 * C + 3 * R, rtol=0, atol=ATOL)
    array = xr.DataArray(np.zeros((3, 4, 5)), dims=result.dims, coords=result.coords)
    assert_allclose(
        Geometry(array, result.transform, dims=result.dims).lattice().matrix[:, 0],
        2 * N,
        rtol=0,
        atol=ATOL,
    )
    assert result.slice_intervals is not None
    assert_allclose(
        result.slice_intervals, [[-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]], rtol=0, atol=ATOL
    )


def test_nonuniform_and_single_slice_use_offsets_and_thickness() -> None:
    stack = [image(0, thickness=1), image(2, thickness=3), image(5, thickness=2)]
    result = from_datasets(stack)
    assert "slice_offset" in result.coords and "k" not in result.coords
    assert result.transform.source.axes[0] == "slice_offset"
    assert_allclose(result.coords["slice_offset"][1], [0, 2, 5], rtol=0, atol=ATOL)
    assert result.slice_intervals is not None
    assert_allclose(
        result.slice_intervals, np.array([[-0.5, 0.5], [0.5, 3.5], [4, 6]]), rtol=0, atol=ATOL
    )
    assert_allclose(point(result, 2), ORIGIN + 5 * N, rtol=0, atol=ATOL)
    single = from_datasets([image(7, thickness=4)])
    assert "slice_offset" in single.coords
    assert single.slice_intervals is not None
    assert_allclose(single.slice_intervals, [[-2, 2]], rtol=0, atol=ATOL)
    assert_allclose(point(single), ORIGIN + 7 * N, rtol=0, atol=ATOL)


@pytest.mark.parametrize(
    ("offsets", "message"),
    [([0, 0, 0, 1], "duplicate slice positions"), ([0, 2, 2], "duplicate slice positions")],
)
def test_duplicate_positions_refused(offsets: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        from_datasets([image(value) for value in offsets])


def test_in_plane_shift_and_orientation_errors() -> None:
    shifted = image(2)
    shifted.ImagePositionPatient = [float(v) for v in ORIGIN + 2 * N + 0.1 * R]
    with pytest.raises(ValueError, match=r"dataset 1.*in-plane shift"):
        from_datasets([image(0), shifted])
    rotated = image(2)
    rotated.ImageOrientationPatient = [0, 1, 0, *C]
    with pytest.raises(ValueError, match="dataset 1 ImageOrientationPatient disagrees"):
        from_datasets([image(0), rotated])
    bad = image(0)
    bad.ImageOrientationPatient = [1, 0, 0, 0.1, 1, 0]
    with pytest.raises(ValueError, match="not orthonormal"):
        from_datasets([bad])
    small = image(0)
    small.ImageOrientationPatient = [*R, *(C + 1e-5 * R)]
    corrected = from_datasets([small])
    assert "orientation-orthonormalized" in {code for code, _ in corrected.report}
    assert_allclose(point(corrected), ORIGIN, rtol=0, atol=ATOL)


def test_anonymous_frame_patient_position_and_missing_thickness() -> None:
    first, second = image(0), image(2, thickness=None)
    del first.FrameOfReferenceUID
    del second.FrameOfReferenceUID
    second.PatientPosition = "FFS"
    result = from_datasets([first, second])
    assert result.frame.identifier[0] == "xarrayrf.anonymous"
    assert result.patient_position is None
    assert result.slice_intervals is None
    assert {code for code, _ in result.report} >= {
        "anonymous-frame",
        "patient-position-disagree",
        "slice-thickness-missing",
    }
    del first.PatientPosition
    del second.PatientPosition
    missing = from_datasets([first, second])
    assert missing.patient_position is None
    assert "patient-position-missing" in {code for code, _ in missing.report}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda ds: setattr(ds, "PixelSpacing", [2, 4]), "PixelSpacing disagrees"),
        (lambda ds: setattr(ds, "Rows", 6), "rows disagrees"),
        (lambda ds: setattr(ds, "Columns", 6), "columns disagrees"),
        (lambda ds: setattr(ds, "FrameOfReferenceUID", "1.2.3.5"), "uid disagrees"),
        (
            lambda ds: setattr(ds, "AnatomicalOrientationType", "QUADRUPED"),
            "AnatomicalOrientationType",
        ),
    ],
)
def test_inconsistent_metadata_refused(change: object, message: str) -> None:
    a, b = image(0), image(2)
    change(b)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        from_datasets([a, b])


def test_enhanced_shared_and_per_frame_groups() -> None:
    shared = from_enhanced(enhanced([0], shared=True))
    assert shared.order == (0,)
    assert_allclose(point(shared), ORIGIN, rtol=0, atol=ATOL)
    per_frame = from_enhanced(enhanced([4, 0, 2]))
    assert per_frame.order == (1, 2, 0)
    assert_allclose(point(per_frame, 2), ORIGIN + 4 * N, rtol=0, atol=ATOL)


def test_enhanced_selection_and_duplicate_positions() -> None:
    source = enhanced([0, 0, 2])
    with pytest.raises(ValueError, match=r"duplicate slice positions.*0 and 1"):
        from_enhanced(source)
    selected = from_enhanced(source, frame_indices=[2, 1])
    assert selected.order == (1, 2)
    assert_allclose(point(selected, 1), ORIGIN + 2 * N, rtol=0, atol=ATOL)


def test_enhanced_group_and_count_errors() -> None:
    source = enhanced([0])
    source.SharedFunctionalGroupsSequence[0].PlanePositionSequence = DicomSequence([Dataset()])
    with pytest.raises(ValueError, match="PlanePositionSequence in both"):
        from_enhanced(source)
    source = enhanced([0, 2])
    del source.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence
    source.SharedFunctionalGroupsSequence[0].PlanePositionSequence = DicomSequence([Dataset()])
    with pytest.raises(ValueError, match="frame 1 has PlanePositionSequence in both"):
        from_enhanced(source, frame_indices=[0])
    source = enhanced([0, 2])
    source.PerFrameFunctionalGroupsSequence.pop()
    with pytest.raises(ValueError, match="length must equal NumberOfFrames"):
        from_enhanced(source)
    source = enhanced([0])
    del source.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence
    with pytest.raises(ValueError, match="requires one PixelMeasuresSequence"):
        from_enhanced(source)


@pytest.mark.parametrize(
    "field", ["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"]
)
def test_missing_required_vectors(field: str) -> None:
    source = image(0)
    delattr(source, field)
    with pytest.raises(ValueError, match=f"dataset 0 requires {field}"):
        from_datasets([source])


def test_invalid_numeric_and_nonfinite_vectors() -> None:
    source = image(0)
    source.ImagePositionPatient = ["", 0, 0]
    with pytest.raises(ValueError, match="numeric ImagePositionPatient"):
        from_datasets([source])
    source = image(0)
    source.ImageOrientationPatient = [1, 0, 0, 0, 1, float("nan")]
    with pytest.raises(ValueError, match="6 finite values for ImageOrientationPatient"):
        from_datasets([source])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ImagePositionPatient", [1, 2], "3 finite values for ImagePositionPatient"),
        ("ImageOrientationPatient", [0, 0, 0, 1, 0, 0], "not orthonormal"),
        ("PixelSpacing", [-2, 3], "positive PixelSpacing"),
        ("Rows", 0, "positive integer Rows"),
        ("Columns", 0, "positive integer Columns"),
        ("SliceThickness", "", "numeric SliceThickness"),
    ],
)
def test_invalid_classic_fields(field: str, value: object, message: str) -> None:
    source = image(0)
    setattr(source, field, value)
    with pytest.raises(ValueError, match=message):
        from_datasets([source])


@pytest.mark.parametrize("name", ["orientation_tolerance", "slice_tolerance"])
def test_tolerance_validation(name: str) -> None:
    with pytest.raises(TypeError, match=f"{name} must be a positive finite number"):
        from_datasets([image(0)], **{name: "bad"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=f"{name} must be a positive finite number"):
        from_enhanced(enhanced([0]), **{name: float("nan")})  # type: ignore[arg-type]


def test_enhanced_public_validation() -> None:
    with pytest.raises(TypeError, match="dataset must be a pydicom Dataset"):
        from_enhanced(object())  # type: ignore[arg-type]
    source = enhanced([0])
    source.SharedFunctionalGroupsSequence = DicomSequence([])
    with pytest.raises(ValueError, match="SharedFunctionalGroupsSequence requires one item"):
        from_enhanced(source)
    source = enhanced([0])
    with pytest.raises(TypeError, match="frame_indices must be a sequence"):
        from_enhanced(source, frame_indices=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="frame_indices must contain unique indices"):
        from_enhanced(source, frame_indices=[0, 0])
    source = enhanced([0])
    source.NumberOfFrames = 0
    with pytest.raises(ValueError, match="positive integer NumberOfFrames"):
        from_enhanced(source)


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: from_datasets([]), ValueError, "at least one image"),
        (lambda: from_datasets(7), TypeError, "datasets must be a sequence"),  # type: ignore[arg-type]
        (lambda: from_datasets([object()]), TypeError, "dataset 0 must be"),  # type: ignore[list-item]
        (lambda: from_datasets([Dataset()]), ValueError, "requires PixelSpacing"),
        (lambda: from_datasets([image(0)], slice_tolerance=0), ValueError, "slice_tolerance"),
        (lambda: from_enhanced(Dataset()), ValueError, "NumberOfFrames"),
        (
            lambda: from_enhanced(enhanced([0]), frame_indices=[1]),
            ValueError,
            "frame_indices must contain",
        ),
    ],
)
def test_public_input_errors(call: object, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        call()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("offsets", "expected"),
    [((0.0, 2.0, 4.0), ["k", "j", "i"]), ((0.0, 2.0, 5.0), ["slice_offset", "j", "i"])],
)
def test_to_dataarray_orders_coordinates_by_dimension(
    offsets: tuple[float, ...], expected: list[str]
) -> None:
    geometry = from_datasets([image(offset) for offset in offsets])
    assert list(to_dataarray(geometry, np.zeros((3, 4, 5))).coords) == expected


@pytest.mark.parametrize("thickness", [None, 4.0])
def test_single_slice_declared_thickness_cells(thickness: float | None) -> None:
    geometry = from_datasets([image(0, thickness=thickness)])
    array = to_dataarray(geometry, np.zeros((1, 4, 5)))
    if thickness is None:
        assert not array.rf.grid.intervals
        with pytest.raises(ValueError, match="single sample"):
            array.rf.geometry.points_at([[0, 0, 0]], domain="cells")
    else:
        assert_allclose(array.rf.grid.intervals["slice_offset"], [[-2, 2]], rtol=0, atol=ATOL)
        positions = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
        points = array.rf.geometry.points_at(positions, domain="cells")
        assert_allclose(points, [ORIGIN - 2 * N, ORIGIN + 2 * N], rtol=0, atol=ATOL)
        assert_allclose(
            array.rf.geometry.positions_at(points, domain="cells"), positions, rtol=0, atol=ATOL
        )
        assert array.rf.grid == array.rf.encode().rf.decode().rf.grid


@pytest.mark.parametrize(
    "offsets,axis,expected",
    [
        ([0, 2, 4], "k", [[-1, 1], [0, 2], [1, 3]]),
        ([0, 2, 5], "slice_offset", [[-2, 2], [0, 4], [3, 7]]),
    ],
)
def test_slice_thickness_intervals_attach_in_slice_coordinate_units(
    offsets: list[int], axis: str, expected: list[list[int]]
) -> None:
    geometry = from_datasets([image(offset, thickness=4) for offset in offsets])
    array = to_dataarray(geometry, np.zeros((3, 4, 5)))
    assert_allclose(array.rf.grid.intervals[axis], expected, rtol=0, atol=ATOL)
    assert geometry.slice_intervals is not None
    assert_allclose(array.rf.grid.intervals[axis], geometry.slice_intervals, rtol=0, atol=ATOL)


@pytest.mark.parametrize("axis_aligned", [False, True])
def test_imported_grid_anatomical_orientation(axis_aligned: bool) -> None:
    from xarrayrf.anatomy import orientation_codes

    datasets = [image(0), image(2), image(4)]
    if axis_aligned:
        for k, dataset in enumerate(datasets):
            dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            dataset.ImagePositionPatient = [10, 20, 30 + 2 * k]
    geometry = from_datasets(datasets)
    array = to_dataarray(geometry, np.zeros((3, 4, 5)))
    assert orientation_codes(array.rf.grid) == ("SPL" if axis_aligned else "ASL")


@pytest.mark.parametrize("enhanced_input", [False, True])
@pytest.mark.parametrize("uid", [None, "1.2.3.4"])
@pytest.mark.parametrize("array_target", [False, True])
def test_dicom_import_frame_override_adopts_complete_ras_declaration(
    enhanced_input: bool,
    uid: str | None,
    array_target: bool,
) -> None:
    from xarrayrf import ReferenceFrame
    from xarrayrf.anatomy import RAS, patient_coordinate_system

    target = ReferenceFrame.declared(
        ("synthetic", "shared"),
        patient_coordinate_system(RAS, "mm"),
        definition={"space": "asserted"},
        context={"epoch": 1},
    )
    partner = to_dataarray(from_datasets([image(0), image(2)], frame=target), np.ones((2, 4, 5)))
    other = partner if array_target else target
    if enhanced_input:
        source = enhanced([0.0, 2.0])
        if uid is None:
            del source.FrameOfReferenceUID
        else:
            source.FrameOfReferenceUID = uid
        original = from_enhanced(source)
        imported = from_enhanced(source, frame=other)
    else:
        datasets = [image(0), image(2)]
        for dataset in datasets:
            if uid is None:
                del dataset.FrameOfReferenceUID
            else:
                dataset.FrameOfReferenceUID = uid
        original = from_datasets(datasets)
        imported = from_datasets(datasets, frame=other)
    assert original.frame.is_anonymous == (uid is None)
    assert imported.frame == target
    assert imported.frame.definition == target.definition
    assert imported.frame.context == target.context
    assert_allclose(
        imported.transform.matrix,
        np.diag([-1.0, -1.0, 1.0]) @ original.transform.matrix,
        rtol=0,
        atol=ATOL,
    )
    assert_allclose(
        imported.transform.translation,
        original.transform.translation * [-1, -1, 1],
        rtol=0,
        atol=ATOL,
    )
    pixels = np.ones((2, 4, 5))
    assumed = to_dataarray(original, pixels).rf.assume_frame(other)
    assert imported.transform == assumed.rf.coordinate_transform


@pytest.mark.parametrize("enhanced_input", [False, True])
def test_dicom_import_rejects_underivable_override_and_unframed_array(enhanced_input: bool) -> None:
    from xarrayrf import CoordinateSystem, ReferenceFrame

    target = ReferenceFrame.local(CoordinateSystem(("x", "y", "z"), ("mm",) * 3))
    error: type[Exception]
    for other in (target, xr.DataArray([1]), "bad"):
        if isinstance(other, ReferenceFrame):
            error, message = ValueError, "cannot adopt frame.*oriented"
        elif isinstance(other, xr.DataArray):
            error, message = ValueError, "unframed"
        else:
            error, message = TypeError, "frame must be"
        with pytest.raises(error, match=message):
            if enhanced_input:
                from_enhanced(enhanced([0.0, 2.0]), frame=other)  # type: ignore[arg-type]
            else:
                from_datasets([image(0), image(2)], frame=other)  # type: ignore[arg-type]


def test_supplied_frame_is_reported_and_replaces_any_uid() -> None:
    target = patient_frame("1.2")
    anonymous = [image(0), image(2)]
    for dataset in anonymous:
        del dataset.FrameOfReferenceUID
    for datasets, replaced in (
        (anonymous, "the anonymous frame"),
        ([image(0), image(2)], "FrameOfReferenceUID 1.2.3.4"),
    ):
        result = from_datasets(datasets, frame=target)
        codes = dict(result.report)
        assert result.frame == target
        assert "anonymous-frame" not in codes
        assert codes["frame-override"] == f"supplied frame replaces {replaced}"
