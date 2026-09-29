"""De-identify DICOM series for publication as example data.

Writes de-identified copies of the given series directories into OUTPUT; the sources are never
modified. Every UID is remapped consistently across all inputs (series from one study keep a
shared Frame of Reference), private tags are dropped, and patient, staff, site, device and date
attributes are emptied or replaced. Acquisition geometry and pixel data are kept.

This covers classic MR images like the tour's HASTE series. It is not a general PS3.15
de-identification profile: review a new kind of input before publishing it.

Usage: python tools/deidentify_dicom.py OUTPUT SERIES_DIR [SERIES_DIR ...] [--execute]
Without --execute the script reports what it would write and writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pydicom
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("series", type=Path, nargs="+")
    parser.add_argument("--execute", action="store_true", help="write the de-identified copies")
    args = parser.parse_args()
    mapping: dict[str, str] = {}
    written = 0
    for series in args.series:
        files = sorted(path for path in series.iterdir() if path.is_file())
        target = args.output / series.name.split("-", 1)[-1]
        print(f"{series} -> {target} ({len(files)} files)", file=sys.stderr)
        if not args.execute:
            continue
        target.mkdir(parents=True, exist_ok=False)
        for index, path in enumerate(files):
            dataset = pydicom.dcmread(path)
            _deidentify(dataset, mapping)
            dataset.save_as(target / f"{index:04d}.dcm", enforce_file_format=True)
            written += 1
    total = sum(1 for s in args.series for p in s.iterdir() if p.is_file())
    if args.execute:
        print(f"wrote {written} files; {len(mapping)} UIDs remapped", file=sys.stderr)
    else:
        print(f"would write {total} files (dry run; pass --execute)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
