"""De-identify DICOM series for publication as example data.

Writes de-identified copies of the given series directories into OUTPUT; the sources are never
modified. Every UID is remapped consistently across all inputs (series from one study keep a
shared Frame of Reference), private tags are dropped, and patient, staff, site, device and date
attributes are emptied or replaced. Acquisition geometry and pixel data are kept.

This covers classic MR images like the tour's HASTE series. It is not a general PS3.15
de-identification profile: review a new kind of input before publishing it.

Usage: python tools/deidentify_dicom.py OUTPUT SERIES_DIR [SERIES_DIR ...] [--execute]
Without --execute the script validates the batch paths, reports the planned work and writes
nothing. Every immediate regular file, including dotfiles, is selected; a non-DICOM file fails
its series. Input directories must be nonempty, distinct and disjoint from the output root.
Output names are the suffix after the first hyphen and must be distinct even ignoring case.
Existing targets (including dangling links) are refused before any output is written.

Each completed series is published by a same-parent directory rename. Expected input or I/O
failures discard that series's staging directory and leave independent later series running;
any failure returns nonzero. Interruptions and unexpected errors clean staging and propagate.
The output tree requires a single writer: the absence recheck is not a concurrent no-replace
guarantee. A hard kill may leave staging directories; there is no crash-durability guarantee.
An interruption immediately after rename can leave a complete target before completion status
is recorded, so the output report may require reconciliation with that target.

Human diagnostics, tracebacks and a final summary go to stderr (the log destination). After
successful argument parsing, stdout contains one JSON run manifest with planned and published
counts, paths and failures, including on preflight failure or interruption. Staging files are
never counted as published.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import UID, generate_uid

UID_PREFIX = "1.2.826.0.1.3680043.10.1234."
"""Prefix of remapped UIDs (a demonstration root; UIDs are derived deterministically)."""

REPLACE = {
    "PatientName": "xarrayrf^demo",
    "PatientID": "xarrayrf-demo",
    "StudyDescription": "xarrayrf tour",
    "PerformedProcedureStepDescription": "xarrayrf tour",
    "RequestedProcedureDescription": "xarrayrf tour",
    "StudyID": "1",
}
"""Identifying attributes replaced by neutral values."""

BLANK = (
    "PatientBirthDate",
    "PatientAge",
    "PatientWeight",
    "PatientSize",
    "OtherPatientIDs",
    "OtherPatientNames",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "PhysiciansOfRecord",
    "NameOfPhysiciansReadingStudy",
    "RequestingPhysician",
    "StationName",
    "DeviceSerialNumber",
    "AccessionNumber",
    "ImageComments",
    "PatientComments",
    "AdditionalPatientHistory",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "InstanceCreationDate",
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ContentTime",
    "InstanceCreationTime",
    "AcquisitionDateTime",
    "PerformedProcedureStepStartDate",
    "PerformedProcedureStepStartTime",
    "PerformedProcedureStepID",
    "RequestAttributesSequence",
    "ReferencedStudySequence",
    "ReferencedPatientSequence",
    "ReferencedPerformedProcedureStepSequence",
)
"""Attributes emptied (kept present so Type 2 requirements still hold)."""


def _remap(uid: str, mapping: dict[str, str]) -> str:
    if uid not in mapping:
        mapping[uid] = generate_uid(prefix=UID_PREFIX, entropy_srcs=[uid])
    return mapping[uid]


def _deidentify(dataset: pydicom.Dataset, mapping: dict[str, str]) -> None:
    dataset.remove_private_tags()
    for keyword in BLANK:
        if keyword in dataset:
            # Blank rather than delete: several are Type 2 (present, possibly empty).
            element = dataset[keyword]
            element.value = [] if element.VR == "SQ" else ""
    for keyword, value in REPLACE.items():
        if keyword in dataset:
            setattr(dataset, keyword, value)
    for element in dataset.iterall():
        if element.VR == "UI" and element.value:
            values = element.value if element.VM > 1 else [element.value]
            # Registered UIDs (SOP classes, transfer syntaxes) are public; instance UIDs are
            # remapped. An unregistered UID's name is the UID itself.
            remapped = [
                str(v) if UID(str(v)).name != str(v) else _remap(str(v), mapping) for v in values
            ]
            element.value = remapped if element.VM > 1 else remapped[0]
    dataset.PatientIdentityRemoved = "YES"
    dataset.DeidentificationMethod = "xarrayrf tools/deidentify_dicom.py"
    meta = dataset.file_meta
    meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID


class InvalidDatasetError(ValueError):
    """Required DICOM instance or file metadata is missing or inconsistent."""


class PreflightError(ValueError):
    """The requested batch paths cannot safely produce distinct output series."""


@dataclass
class SeriesRun:
    source: Path
    target: Path
    files: tuple[Path, ...] = ()
    outcome: str = "not-attempted"
    operation: str = "preflight"
    current_file: Path | None = None


Failure = dict[str, str | None]


def _failure(
    failures: list[Failure], error: BaseException, run: SeriesRun | None, category: str
) -> None:
    if run is not None:
        run.outcome = "failed"
    failures.append(
        {
            "category": category,
            "operation": run.operation if run is not None else "preflight",
            "error": type(error).__name__,
            "message": str(error),
            "input": str(run.source) if run is not None else None,
            "file": str(run.current_file) if run is not None and run.current_file else None,
            "target": str(run.target) if run is not None else None,
        }
    )


def _preflight(output: Path, runs: list[SeriesRun], failures: list[Failure]) -> None:
    # Snapshot all selections even when another input fails; no later enumeration changes counts.
    for run in runs:
        try:
            if not run.source.is_dir():
                raise PreflightError(f"input is not a directory: {run.source}")
            run.files = tuple(sorted(path for path in run.source.iterdir() if path.is_file()))
            if not run.files:
                raise PreflightError(f"input contains no regular files: {run.source}")
        except (OSError, PreflightError) as error:
            _failure(failures, error, run, "preflight")
            traceback.print_exc(file=sys.stderr)

    try:
        for parent in (output, *output.parents):
            if os.path.lexists(parent) and not parent.is_dir():
                raise PreflightError(f"output ancestor is not a directory: {parent}")
        resolved_output = output.resolve()
    except (OSError, PreflightError) as error:
        _failure(failures, error, None, "preflight")
        traceback.print_exc(file=sys.stderr)
        return

    inputs: set[Path] = set()
    targets: set[Path] = set()
    names: set[str] = set()
    for run in runs:
        if run.outcome == "failed":
            continue
        try:
            name = run.source.name.split("-", 1)[-1]
            if name in {"", ".", ".."}:
                raise PreflightError(f"invalid output name {name!r} for {run.source}")
            source = run.source.resolve(strict=True)
            target = run.target.resolve()
            if source in inputs:
                raise PreflightError(f"duplicate input directory: {run.source}")
            inputs.add(source)
            if source.is_relative_to(resolved_output) or resolved_output.is_relative_to(source):
                raise PreflightError(
                    f"input and output directories overlap: {run.source}, {output}"
                )
            if target in targets or name.casefold() in names:
                raise PreflightError(f"duplicate output name (case-insensitive): {run.target}")
            targets.add(target)
            names.add(name.casefold())
            if os.path.lexists(run.target):
                raise PreflightError(f"output target already exists: {run.target}")
        except (OSError, PreflightError) as error:
            _failure(failures, error, run, "preflight")
            traceback.print_exc(file=sys.stderr)


def _validate_dataset(dataset: pydicom.Dataset) -> None:
    meta = getattr(dataset, "file_meta", None)
    if not isinstance(meta, pydicom.Dataset):
        raise InvalidDatasetError("missing file_meta")
    for container, keywords in (
        (dataset, ("SOPClassUID", "SOPInstanceUID")),
        (meta, ("MediaStorageSOPClassUID", "MediaStorageSOPInstanceUID", "TransferSyntaxUID")),
    ):
        for keyword in keywords:
            value = getattr(container, keyword, None)
            if not isinstance(value, str) or not value or not UID(value).is_valid:
                raise InvalidDatasetError(f"missing or invalid {keyword}")
    if not UID(meta.TransferSyntaxUID).is_transfer_syntax:
        raise InvalidDatasetError("unrecognized TransferSyntaxUID")
    if meta.MediaStorageSOPClassUID != dataset.SOPClassUID:
        raise InvalidDatasetError("MediaStorageSOPClassUID does not match SOPClassUID")
    if meta.MediaStorageSOPInstanceUID != dataset.SOPInstanceUID:
        raise InvalidDatasetError("MediaStorageSOPInstanceUID does not match SOPInstanceUID")


def _publish_series(run: SeriesRun, mapping: dict[str, str]) -> None:
    run.operation = "stage"
    run.target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{run.target.name}-", dir=run.target.parent) as temp:
        staging = Path(temp)
        for index, path in enumerate(run.files):
            run.current_file = path
            run.operation = "read"
            dataset = pydicom.dcmread(path)
            run.operation = "validate"
            _validate_dataset(dataset)
            run.operation = "deidentify"
            _deidentify(dataset, mapping)
            run.operation = "write"
            dataset.save_as(staging / f"{index:04d}.dcm", enforce_file_format=True)
        run.operation = "publish"
        if os.path.lexists(run.target):
            raise FileExistsError(f"output target already exists: {run.target}")
        staging.rename(run.target)
        run.outcome = "completed"


def _report(
    output: Path, runs: list[SeriesRun], failures: list[Failure], dry_run: bool, started: float
) -> None:
    counts = {
        "planned_series": len(runs),
        "completed_series": sum(run.outcome == "completed" for run in runs),
        "failed_series": sum(run.outcome == "failed" for run in runs),
        "not_attempted_series": sum(run.outcome == "not-attempted" for run in runs),
        "planned_files": sum(len(run.files) for run in runs),
        "published_files": sum(len(run.files) for run in runs if run.outcome == "completed"),
    }
    elapsed = time.perf_counter() - started
    print(f"Summary ({'dry run; pass --execute' if dry_run else 'execute'}):", file=sys.stderr)
    print(f"output: {output}", file=sys.stderr)
    for run in runs:
        print(
            f"{run.source} -> {run.target} ({len(run.files)} files; {run.outcome})",
            file=sys.stderr,
        )
    for key, value in counts.items():
        print(f"{key.replace('_', ' ')}: {value}", file=sys.stderr)
    for failure in failures:
        print(
            f"{failure['category']} ({failure['operation']}): "
            f"{failure['file'] or failure['input']} -> "
            f"{failure['target']}: {failure['error']}: {failure['message']}",
            file=sys.stderr,
        )
    print(f"elapsed: {elapsed:.3f}s; log: stderr", file=sys.stderr)
    print(
        json.dumps(
            {
                "output": str(output),
                "dry_run": dry_run,
                "counts": counts,
                "series": [
                    {
                        "input": str(run.source),
                        "target": str(run.target),
                        "outcome": run.outcome,
                        "planned_files": len(run.files),
                        "published_files": len(run.files) if run.outcome == "completed" else 0,
                    }
                    for run in runs
                ],
                "failures": failures,
                "elapsed_seconds": elapsed,
                "log": "stderr",
            },
            separators=(",", ":"),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("series", type=Path, nargs="+")
    parser.add_argument("--execute", action="store_true", help="write the de-identified copies")
    args = parser.parse_args()
    started = time.perf_counter()
    runs = [
        SeriesRun(source, args.output / source.name.split("-", 1)[-1]) for source in args.series
    ]
    failures: list[Failure] = []
    current: SeriesRun | None = None
    finished = False
    try:
        _preflight(args.output, runs, failures)
        if not failures and args.execute:
            mapping: dict[str, str] = {}
            for current in runs:
                try:
                    _publish_series(current, mapping)
                except (OSError, InvalidDicomError, InvalidDatasetError) as error:
                    _failure(failures, error, current, current.operation)
                    traceback.print_exc(file=sys.stderr)
                current = None
        finished = True
        return 1 if failures else 0
    finally:
        active_error = sys.exception()
        if not finished and active_error is not None:
            category = (
                "interrupted" if isinstance(active_error, KeyboardInterrupt) else "unexpected"
            )
            _failure(failures, active_error, current, category)
        _report(args.output, runs, failures, not args.execute, started)


if __name__ == "__main__":
    raise SystemExit(main())
