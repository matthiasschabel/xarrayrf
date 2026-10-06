from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian, MRImageStorage

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "deidentify_dicom.py"


@pytest.fixture
def tool(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("deidentify_dicom", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def series_factory(tmp_path: Path) -> Callable[[str, int], Path]:
    series_number = 0

    def make(name: str, count: int) -> Path:
        nonlocal series_number
        series_number += 1
        source = tmp_path / "sources" / name
        source.mkdir(parents=True)
        for index in range(count):
            uid = f"1.2.826.0.1.3680043.8.100.{series_number}.{index + 1}"
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = UID(uid)
            dataset = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
            dataset.SOPClassUID = MRImageStorage
            dataset.SOPInstanceUID = uid
            dataset.StudyInstanceUID = "1.2.826.0.1.3680043.8.101"
            dataset.SeriesInstanceUID = f"1.2.826.0.1.3680043.8.102.{series_number}"
            dataset.FrameOfReferenceUID = "1.2.826.0.1.3680043.8.103"
            dataset.PatientName = "Private^Patient"
            dataset.PatientID = "private-id"
            dataset.InstitutionName = "Private hospital"
            dataset.StudyDate = "20250101"
            dataset.Modality = "MR"
            dataset.ImagePositionPatient = [0.0, 0.0, 4.0 * index]
            dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            dataset.PixelSpacing = [0.7, 0.8]
            dataset.SliceThickness = 1.0
            dataset.Rows = 2
            dataset.Columns = 2
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 0
            dataset.PixelData = np.arange(4, dtype="<u2").tobytes()
            dataset.add_new((0x0011, 0x0010), "LO", "Private creator")
            dataset.add_new((0x0011, 0x1001), "LO", "Private value")
            dataset.save_as(source / f"{index:04d}.dcm", enforce_file_format=True)
        return source

    return make


def snapshot(path: Path) -> dict[str, bytes | None]:
    return {
        str(item.relative_to(path)): item.read_bytes() if item.is_file() else None
        for item in path.rglob("*")
    }


def cli(output: Path, *sources: Path, execute: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(output), *map(str, sources)]
        + (["--execute"] if execute else []),
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_snapshots_files_without_creating_output(
    tmp_path: Path, series_factory: Callable[[str, int], Path]
) -> None:
    source = series_factory("001-first", 2)
    (source / "0001.dcm").rename(source / ".hidden.dcm")
    (source / "ignored-directory").mkdir()
    before = snapshot(tmp_path)
    output = tmp_path / "absent" / "output"
    result = cli(output, source, execute=False)
    assert result.returncode == 0, result.stderr
    assert snapshot(tmp_path) == before
    manifest = json.loads(result.stdout)
    assert manifest["dry_run"] is True
    assert manifest["counts"] == {
        "planned_series": 1,
        "completed_series": 0,
        "failed_series": 0,
        "not_attempted_series": 1,
        "planned_files": 2,
        "published_files": 0,
    }
    assert manifest["series"][0]["target"] == str(output / "first")
    assert "planned files: 2" in result.stderr
    assert "log: stderr" in result.stderr


@pytest.mark.parametrize(
    "case",
    [
        "duplicate-name",
        "casefold-name",
        "duplicate-input",
        "symlink-input",
        "existing",
        "dangling-target",
        "file-parent",
        "dangling-parent",
        "missing",
        "file-input",
        "empty",
        "empty-name",
        "dot-name",
        "dotdot-name",
        "output-inside-input",
        "input-inside-output",
        "same-root",
    ],
)
def test_preflight_rejects_batch_before_writing(
    case: str, tmp_path: Path, series_factory: Callable[[str, int], Path]
) -> None:
    first = series_factory("001-first", 1)
    second = series_factory("002-second", 1)
    output = tmp_path / "output"
    if case == "duplicate-name":
        second = series_factory("003-first", 1)
    elif case == "casefold-name":
        second = series_factory("003-FIRST", 1)
    elif case == "duplicate-input":
        second = first
    elif case == "symlink-input":
        second = tmp_path / "004-alias"
        second.symlink_to(first, target_is_directory=True)
    elif case in {"existing", "dangling-target"}:
        output.mkdir()
        if case == "existing":
            (output / "second").mkdir()
            (output / "second" / "keep").write_bytes(b"keep")
        else:
            (output / "second").symlink_to(tmp_path / "missing-target")
    elif case == "file-parent":
        output.write_bytes(b"keep")
        output = output / "child"
    elif case == "dangling-parent":
        output.symlink_to(tmp_path / "missing-parent")
        output = output / "child"
    elif case == "missing":
        second = tmp_path / "missing"
    elif case == "file-input":
        second = first / "0000.dcm"
    elif case == "empty":
        second = series_factory("003-empty", 0)
    elif case in {"empty-name", "dot-name", "dotdot-name"}:
        suffix = {"empty-name": "", "dot-name": ".", "dotdot-name": ".."}[case]
        second = series_factory("003-" + suffix, 1)
    elif case == "output-inside-input":
        output = second / "output"
    elif case == "input-inside-output":
        output = second.parent
    elif case == "same-root":
        output = second
    # rglob does not follow directory symlinks; retain link targets explicitly as well.
    before = snapshot(tmp_path)
    links = {str(p): p.readlink() for p in tmp_path.rglob("*") if p.is_symlink()}
    result = cli(output, first, second)
    assert result.returncode != 0
    assert snapshot(tmp_path) == before
    assert {str(p): p.readlink() for p in tmp_path.rglob("*") if p.is_symlink()} == links
    manifest = json.loads(result.stdout)
    assert manifest["counts"]["completed_series"] == 0
    assert manifest["counts"]["published_files"] == 0
    assert manifest["failures"][0]["category"] == "preflight"
    assert "Summary" in result.stderr


def test_failed_middle_series_keeps_only_complete_series(
    tmp_path: Path, series_factory: Callable[[str, int], Path]
) -> None:
    first = series_factory("001-first", 1)
    broken = series_factory("002-broken", 2)
    last = series_factory("003-last", 1)
    (broken / "0001.dcm").write_bytes(b"not DICOM")
    before = snapshot(first.parent)
    output = tmp_path / "output"
    result = cli(output, first, broken, last)
    assert result.returncode != 0
    assert sorted(p.name for p in output.iterdir()) == ["first", "last"]
    assert snapshot(first.parent) == before
    manifest = json.loads(result.stdout)
    assert manifest["counts"] == {
        "planned_series": 3,
        "completed_series": 2,
        "failed_series": 1,
        "not_attempted_series": 0,
        "planned_files": 4,
        "published_files": 2,
    }
    assert manifest["failures"][0]["category"] == "read"
    assert manifest["failures"][0]["operation"] == "read"
    assert manifest["failures"][0]["input"] == str(broken)
    assert manifest["failures"][0]["file"] == str(broken / "0001.dcm")
    assert str(broken / "0001.dcm") in result.stderr
    assert "InvalidDicomError" in result.stderr


@pytest.mark.parametrize("missing", ["SOPInstanceUID", "SOPClassUID", "TransferSyntaxUID"])
def test_missing_required_metadata_is_reported_as_series_failure(
    missing: str, tmp_path: Path, series_factory: Callable[[str, int], Path]
) -> None:
    source = series_factory("001-broken", 1)
    good = series_factory("002-good", 1)
    path = source / "0000.dcm"
    dataset = pydicom.dcmread(path)
    delattr(dataset.file_meta if missing == "TransferSyntaxUID" else dataset, missing)
    dataset.save_as(path, enforce_file_format=False)
    before = snapshot(source.parent)
    result = cli(tmp_path / "output", source, good)
    assert result.returncode != 0
    assert sorted(p.name for p in (tmp_path / "output").iterdir()) == ["good"]
    assert snapshot(source.parent) == before
    manifest = json.loads(result.stdout)
    assert manifest["failures"][0]["category"] == "validate"
    assert missing in result.stderr


@pytest.mark.parametrize("failure", ["write", "publish", "unexpected", "interrupt"])
def test_injected_failure_cleans_staging_and_always_summarizes(
    failure: str,
    tmp_path: Path,
    series_factory: Callable[[str, int], Path],
    tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = series_factory("001-broken", 2)
    later = series_factory("002-later", 1)
    before = snapshot(source.parent)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), str(output), str(source), str(later), "--execute"]
    )
    original_save = pydicom.dataset.Dataset.save_as
    original_rename = Path.rename

    def save(dataset: pydicom.Dataset, filename: Any, *args: Any, **kwargs: Any) -> None:
        if Path(filename).name == "0001.dcm":
            Path(filename).write_bytes(b"partial")
            if failure == "unexpected":
                raise RuntimeError("injected programming error")
            if failure == "interrupt":
                raise KeyboardInterrupt("injected interruption")
            if failure == "write":
                raise OSError("injected write error")
        original_save(dataset, filename, *args, **kwargs)

    def rename(path: Path, target: Path) -> Path:
        if target.name == "broken":
            raise OSError("injected publication error")
        return original_rename(path, target)

    monkeypatch.setattr(pydicom.dataset.Dataset, "save_as", save)
    if failure == "publish":
        monkeypatch.setattr(Path, "rename", rename)
    if failure in {"unexpected", "interrupt"}:
        with pytest.raises(RuntimeError if failure == "unexpected" else KeyboardInterrupt):
            tool.main()
    else:
        assert tool.main() != 0
    captured = capsys.readouterr()
    assert sorted(p.name for p in output.iterdir()) == (
        [] if failure in {"unexpected", "interrupt"} else ["later"]
    )
    assert snapshot(source.parent) == before
    manifest = json.loads(captured.out)
    recorded_failure = manifest["failures"][0]
    assert recorded_failure["category"] == ("interrupted" if failure == "interrupt" else failure)
    assert recorded_failure["operation"] == ("publish" if failure == "publish" else "write")
    assert recorded_failure["input"] == str(source)
    assert recorded_failure["file"] == str(source / "0001.dcm")
    assert recorded_failure["target"] == str(output / "broken")
    assert manifest["counts"]["failed_series"] == 1
    assert manifest["counts"]["published_files"] == (
        0 if failure in {"unexpected", "interrupt"} else 1
    )
    assert manifest["counts"]["not_attempted_series"] == (
        1 if failure in {"unexpected", "interrupt"} else 0
    )
    assert "Summary" in captured.err
    assert str(source / "0001.dcm") in captured.err or failure == "publish"


def test_publication_rechecks_existing_destination(
    tmp_path: Path,
    series_factory: Callable[[str, int], Path],
    tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = series_factory("001-first", 1)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(output), str(source), "--execute"])
    original_save = pydicom.dataset.Dataset.save_as

    def save(dataset: pydicom.Dataset, filename: Any, *args: Any, **kwargs: Any) -> None:
        original_save(dataset, filename, *args, **kwargs)
        (output / "first").mkdir(exist_ok=True)
        (output / "first" / "keep").write_bytes(b"keep")

    monkeypatch.setattr(pydicom.dataset.Dataset, "save_as", save)
    assert tool.main() != 0
    assert snapshot(output) == {"first": None, "first/keep": b"keep"}
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["counts"]["published_files"] == 0
    assert manifest["failures"][0]["category"] == "publish"


def test_saved_files_preserve_pixels_geometry_uid_links_and_tag_policy(
    tmp_path: Path, series_factory: Callable[[str, int], Path]
) -> None:
    sources = [series_factory("001-first", 2), series_factory("002-second", 1)]
    before = snapshot(sources[0].parent)
    output = tmp_path / "output"
    result = cli(output, *sources)
    assert result.returncode == 0, result.stderr
    assert snapshot(sources[0].parent) == before
    saved = []
    for source, target in zip(sources, [output / "first", output / "second"], strict=True):
        assert len(list(target.iterdir())) == len(list(source.iterdir()))
        for path in source.iterdir():
            original = pydicom.dcmread(path)
            actual = pydicom.dcmread(target / path.name)
            saved.append(actual)
            assert actual.PixelData == original.PixelData
            np.testing.assert_array_equal(actual.pixel_array, original.pixel_array)
            for keyword in [
                "ImagePositionPatient",
                "ImageOrientationPatient",
                "PixelSpacing",
                "SliceThickness",
            ]:
                # Serialization preserves decimal-string geometry to far below a nanometre.
                np.testing.assert_allclose(
                    getattr(actual, keyword), getattr(original, keyword), rtol=0, atol=1e-12
                )
            assert actual.SOPClassUID == actual.file_meta.MediaStorageSOPClassUID == MRImageStorage
            assert actual.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
            assert actual.SOPInstanceUID == actual.file_meta.MediaStorageSOPInstanceUID
            assert actual.SOPInstanceUID != original.SOPInstanceUID
            assert actual.PatientName == "xarrayrf^demo"
            assert actual.PatientID == "xarrayrf-demo"
            assert actual.InstitutionName == actual.StudyDate == ""
            assert actual.PatientIdentityRemoved == "YES"
            assert not any(element.tag.is_private for element in actual.iterall())
    assert len({dataset.StudyInstanceUID for dataset in saved}) == 1
    assert len({dataset.FrameOfReferenceUID for dataset in saved}) == 1
    assert saved[0].StudyInstanceUID != pydicom.dcmread(sources[0] / "0000.dcm").StudyInstanceUID
    retry = tmp_path / "retry"
    assert cli(retry, sources[1]).returncode == 0
    assert (
        pydicom.dcmread(retry / "second" / "0000.dcm").StudyInstanceUID == saved[0].StudyInstanceUID
    )
    manifest = json.loads(result.stdout)
    assert manifest["counts"]["published_files"] == 3
    assert manifest["counts"]["completed_series"] == 2
    assert manifest["failures"] == []
    assert "published files: 3" in result.stderr
