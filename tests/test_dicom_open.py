"""DICOM file reader behavior using small explicit-VR files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import pytest
from dask.callbacks import Callback
from numpy.testing import assert_allclose
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from xarrayrf.dicom import open


def write_slice(
    path: Path, offset: int, uid: str, *, slope: int = 2, intercept: int = 5, rgb: bool = False
) -> None:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = uid
    ds.SeriesDescription = "synthetic"
    ds.FrameOfReferenceUID = "1.2.3.4"
    ds.ImagePositionPatient = [0, 0, offset]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [2, 3]
    ds.Rows = 2
    ds.Columns = 3
    ds.SamplesPerPixel = 3 if rgb else 1
    ds.PhotometricInterpretation = "RGB" if rgb else "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    pixels = np.full((2, 3, 3) if rgb else (2, 3), offset + 1, dtype=np.uint16)
    ds.PixelData = pixels.tobytes()
    pydicom.dcmwrite(path, ds, enforce_file_format=True)


def write_enhanced(path: Path) -> None:
    write_slice(path, 0, "1.2.3.5")
    ds = pydicom.dcmread(path)
    ds.NumberOfFrames = 2
    ds.PixelData = np.stack(
        [np.ones((2, 3), dtype=np.uint16), np.full((2, 3), 2, dtype=np.uint16)]
    ).tobytes()
    ds.SharedFunctionalGroupsSequence = DicomSequence([Dataset()])
    frames = []
    for offset, slope in ((0, 2), (1, 3)):
        group = Dataset()
        position = Dataset()
        position.ImagePositionPatient = [0, 0, offset]
        orientation = Dataset()
        orientation.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        measure = Dataset()
        measure.PixelSpacing = [2, 3]
        transform = Dataset()
        transform.RescaleSlope = slope
        transform.RescaleIntercept = 4
        group.PlanePositionSequence = DicomSequence([position])
        group.PlaneOrientationSequence = DicomSequence([orientation])
        group.PixelMeasuresSequence = DicomSequence([measure])
        group.PixelValueTransformationSequence = DicomSequence([transform])
        frames.append(group)
    ds.PerFrameFunctionalGroupsSequence = DicomSequence(frames)
    pydicom.dcmwrite(path, ds, enforce_file_format=True)


def test_dicom_classic_source_order_lut_and_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uid = "1.2.3.5"
    paths = [tmp_path / f"slice{i}.dcm" for i in (2, 0, 1)]
    for path, offset in zip(paths, (2, 0, 1), strict=True):
        write_slice(path, offset, uid)
    reads = []
    original = pydicom.dcmread

    def counted(*args: Any, **kwargs: Any) -> Any:
        if not kwargs.get("stop_before_pixels", False):
            reads.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(pydicom, "dcmread", counted)
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        image = open(paths)
    assert not reads and not tasks
    assert image.dtype == np.float64
    assert_allclose(image[:, 0, 0].compute().data, [7, 9, 11], rtol=0, atol=1e-12)
    assert len(reads) == 3
    assert_allclose(
        image.rf.geometry.points().isel(k=2, j=0, i=0).data, [0, 0, 2], rtol=0, atol=1e-12
    )
    eager = open(paths, chunks=None, modality_lut=False)
    assert isinstance(eager.data, np.ndarray)
    assert_allclose(eager[:, 0, 0].data, [1, 2, 3], rtol=0, atol=1e-12)


def test_dicom_directory_selection_and_bad_files(tmp_path: Path) -> None:
    write_slice(tmp_path / "a.dcm", 0, "1.2.3.5")
    write_slice(tmp_path / "b.dcm", 0, "1.2.3.6")
    (tmp_path / "junk").write_text("not dicom")
    (tmp_path / "DICOMDIR").write_text("directory entry")
    with (
        pytest.warns(UserWarning, match=r"DICOMDIR.*junk"),
        pytest.raises(ValueError, match=r"1.2.3.5.*1.2.3.6"),
    ):
        open(tmp_path)
    with pytest.warns(UserWarning, match="junk"):
        assert open(tmp_path, series_uid="1.2.3.5").shape == (1, 2, 3)
    with pytest.raises(ValueError, match="invalid DICOM"):
        open([tmp_path / "junk"])
    with pytest.raises(ValueError, match="unknown series_uid"):
        open([tmp_path / "a.dcm"], series_uid="bad")
    with pytest.raises(ValueError, match="frames requires"):
        open([tmp_path / "a.dcm"], frames=[0])
    with pytest.raises(FileNotFoundError):
        open(tmp_path / "missing.dcm")


def test_dicom_enhanced_rescale_and_rgb_rejection(tmp_path: Path) -> None:
    enhanced = tmp_path / "enhanced.dcm"
    write_enhanced(enhanced)
    image = open(enhanced)
    assert_allclose(image[:, 0, 0].compute().data, [6, 10], rtol=0, atol=1e-12)
    rgb = tmp_path / "rgb.dcm"
    write_slice(rgb, 0, "1.2.3.7", rgb=True)
    with pytest.raises(ValueError, match="monochrome"):
        open(rgb)
    with pytest.raises(TypeError, match="paths"):
        open(3)  # type: ignore[arg-type]


def test_dicom_pixel_declaration_and_enhanced_lut_errors(tmp_path: Path) -> None:
    path = tmp_path / "image.dcm"
    write_slice(path, 0, "1.2.3.5")
    ds = pydicom.dcmread(path)
    ds.BitsAllocated = 12
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    with pytest.raises(ValueError, match="BitsAllocated"):
        open(path)
    ds.BitsAllocated = 16
    ds.PixelRepresentation = 2
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    with pytest.raises(ValueError, match="PixelRepresentation"):
        open(path)
    ds.BitsAllocated = 1
    ds.PixelRepresentation = 1
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    with pytest.raises(ValueError, match="1-bit pixels"):
        open(path)

    write_enhanced(path)
    ds = pydicom.dcmread(path)
    ds.ModalityLUTSequence = DicomSequence([Dataset()])
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    with pytest.raises(ValueError, match="ModalityLUTSequence"):
        open(path, modality_lut=False)
    del ds.ModalityLUTSequence
    ds.PerFrameFunctionalGroupsSequence[0].PixelValueTransformationSequence = DicomSequence([])
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    with pytest.raises(ValueError, match="PixelValueTransformationSequence"):
        open(path)


def test_dicom_empty_and_argument_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        open([])
    with pytest.raises(TypeError, match="paths"):
        open([3])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="no DICOM"):
        open(tmp_path)
    path = tmp_path / "image.dcm"
    write_slice(path, 0, "1.2.3.5")
    with pytest.raises(TypeError, match="series_uid"):
        open(path, series_uid=3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="modality_lut"):
        open(path, modality_lut="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="frames"):
        open(path, frames=3)  # type: ignore[arg-type]
