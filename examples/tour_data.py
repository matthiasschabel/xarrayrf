"""Data for the xarrayrf tour notebook.

The MRI study (de-identified DICOM series and their dcm2niix NIfTI conversions, CC BY 4.0) is
downloaded once and cached, as are the public brain images and atlas. The microscopy and satellite examples read public data directly from the web.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

TOUR_DATA_URL = (
    "https://github.com/matthiasschabel/xarrayrf/releases/download/"
    "tour-data-1/xarrayrf-tour-data-1.zip"
)
TOUR_DATA_SHA256 = "499ec8be464b93f599dcda707ac7a8255944f9f8d5b06d21079b19b9874999fd"


def _download(archive: Path) -> None:
    """Fetch the bundle anonymously, or with the GitHub CLI when the repository needs access."""
    try:
        urllib.request.urlretrieve(TOUR_DATA_URL, archive)
    except urllib.error.HTTPError as error:
        if error.code != 404 or shutil.which("gh") is None:
            raise
        # A private repository answers anonymous requests with 404; an authenticated
        # collaborator can still fetch the release asset.
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                "tour-data-1",
                "--repo",
                "matthiasschabel/xarrayrf",
                "--pattern",
                "xarrayrf-tour-data-1.zip",
                "--output",
                str(archive),
                "--clobber",
            ],
            check=True,
        )


def mri_study() -> Path:
    """Directory holding ``dicom/`` and ``nifti/`` of the tour's MRI study, downloading it once.

    The cache is ``~/.cache/xarrayrf/tour-data-1``; set ``XARRAYRF_TOUR_DATA`` to use another
    directory that already contains the extracted files.
    """
    configured = os.environ.get("XARRAYRF_TOUR_DATA")
    if configured:
        return Path(configured)
    cache = Path.home() / ".cache" / "xarrayrf" / "tour-data-1"
    if (cache / "README.txt").exists():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    archive = cache.with_suffix(".zip.part")
    _download(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != TOUR_DATA_SHA256:
        archive.unlink()
        raise OSError(f"tour data checksum mismatch: {digest}")
    staging = cache.with_suffix(".part")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(staging)
    staging.rename(cache)
    archive.unlink()
    return cache


BRAIN_FILES = {
    "sub-01_T1w.nii.gz": (
        "https://s3.amazonaws.com/openneuro.org/ds000001/sub-01/anat/sub-01_T1w.nii.gz",
        "bdb7022ae229c5b8edd16425928c9243c562f84082b9b8e6f97cdba8b9354a98",
    ),
    "sub-02_T1w.nii.gz": (
        "https://s3.amazonaws.com/openneuro.org/ds000001/sub-02/anat/sub-02_T1w.nii.gz",
        "02ed06fb4e3421c620998af9232739be1c3c0526dd7bcf6652bf8522f5bf2944",
    ),
    "MNI152NLin2009cAsym_T1w.nii.gz": (
        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
        "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz",
        "1f27aabea9f7183dc0c69dafa71e1787b921ca778d655d9c1bf302b273d5627a",
    ),
    "Schaefer2018_100Parcels7Networks.nii.gz": (
        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
        "tpl-MNI152NLin2009cAsym_res-01_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.nii.gz",
        "6451253870267d695bce91b292feb60d4805eb3817fb1bd4f8dc626f901f3def",
    ),
    "Schaefer2018_100Parcels7Networks.tsv": (
        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
        "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.tsv",
        "a81a63f67fdaffd7b2b4b6c7df05f75b16899cba70100f464bbccba0a257cc6c",
    ),
}
"""Public brain images: two OpenNeuro ds000001 subjects (CC0), the MNI152NLin2009cAsym template
and the Schaefer 2018 100-parcel atlas from TemplateFlow."""

MNI_TO_SUBJECT_MATRIX = [
    [0.920692, 0.029702, -0.033015],
    [-0.029144, 0.870898, -0.225135],
    [0.010626, 0.25032, 0.824056],
]
MNI_TO_SUBJECT_TRANSLATION = [-1.9677, -0.1347, -2.7548]
"""Affine from MNI152NLin2009cAsym to sub-01's scanner space (RAS mm), from
tools/register_tour_subject.py (SimpleITK, Mattes mutual information, three levels)."""


def brain_images() -> Path:
    """Directory holding the tour's brain images, downloading each file once and verifying it."""
    cache = Path.home() / ".cache" / "xarrayrf" / "brain-1"
    cache.mkdir(parents=True, exist_ok=True)
    for name, (url, sha256) in BRAIN_FILES.items():
        path = cache / name
        if path.exists():
            continue
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        digest = hashlib.sha256(partial.read_bytes()).hexdigest()
        if digest != sha256:
            partial.unlink()
            raise OSError(f"{name}: checksum mismatch {digest}")
        partial.rename(path)
    return cache


TISSUE_URL = (
    "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.26.0/src/skimage/data/ihc.png"
)
TISSUE_SHA256 = "f8dd1aa387ddd1f49d8ad13b50921b237df8e9b262606d258770687b0ef93cef"


def tissue_section() -> Path:
    """scikit-image's immunohistochemistry sample (512 x 512 RGB), downloaded once."""
    path = Path.home() / ".cache" / "xarrayrf" / "ihc.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".png.part")
        urllib.request.urlretrieve(TISSUE_URL, partial)
        digest = hashlib.sha256(partial.read_bytes()).hexdigest()
        if digest != TISSUE_SHA256:
            partial.unlink()
            raise OSError(f"tissue section: checksum mismatch {digest}")
        partial.rename(path)
    return path
