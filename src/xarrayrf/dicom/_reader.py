"""Read DICOM pixel stacks using the existing metadata geometry importer."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import numpy.typing as npt
import pydicom
import xarray as xr
from dask import delayed  # type: ignore[attr-defined]
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.pixels import apply_modality_lut  # type: ignore[attr-defined]


def _stored_dtype(ds: Dataset) -> np.dtype[Any]:
    bits = int(ds.BitsAllocated)
    if bits not in (1, 8, 16, 32, 64):
        raise ValueError(f"unsupported BitsAllocated {bits}")
    if int(ds.PixelRepresentation) not in (0, 1):
        raise ValueError("PixelRepresentation must be 0 or 1")
    if bits == 1:
        if int(ds.PixelRepresentation):
            raise ValueError("1-bit pixels must be unsigned")
        return np.dtype("uint8")
    kind = "i" if int(ds.PixelRepresentation) else "u"
    return np.dtype(f"{kind}{bits // 8}")


def _pixel_transform(ds: Dataset, index: int) -> tuple[float, float]:
    groups = ds.PerFrameFunctionalGroupsSequence
    shared = getattr(ds, "SharedFunctionalGroupsSequence", ())
    candidates = (groups[index], shared[0] if shared else ds, ds)
    for group in candidates:
        if hasattr(group, "ModalityLUTSequence"):
            raise ValueError("enhanced ModalityLUTSequence is unsupported")
        values = getattr(group, "PixelValueTransformationSequence", None)
        if values is not None and any(hasattr(item, "ModalityLUTSequence") for item in values):
            raise ValueError("enhanced ModalityLUTSequence is unsupported")
    for group in candidates:
        values = getattr(group, "PixelValueTransformationSequence", None)
        if values is not None:
            if len(values) != 1:
                raise ValueError("PixelValueTransformationSequence requires one item")
            item = values[0]
            return float(getattr(item, "RescaleSlope", 1)), float(
                getattr(item, "RescaleIntercept", 0)
            )
    return float(getattr(ds, "RescaleSlope", 1)), float(getattr(ds, "RescaleIntercept", 0))


def _read_pixels(
    path: Path, metadata: Dataset, modality_lut: bool, enhanced: bool
) -> npt.NDArray[np.integer[Any] | np.floating[Any]]:
    pixels = np.asarray(pydicom.dcmread(path).pixel_array)
    if not modality_lut:
        return pixels
    if not enhanced:
        return np.asarray(apply_modality_lut(pixels, metadata), dtype=np.float64)
    result = np.empty(pixels.shape, dtype=np.float64)
    for index in range(pixels.shape[0]):
        slope, intercept = _pixel_transform(metadata, index)
        result[index] = pixels[index].astype(np.float64) * slope + intercept
    return result


def open(
    paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    series_uid: str | None = None,
    frames: Sequence[int] | None = None,
    modality_lut: bool = True,
    orientation_tolerance: float = 1e-4,
    slice_tolerance: float = 0.01,
    chunks: Any = "auto",
) -> xr.DataArray:
    """Open one monochrome DICOM series as a framed array.

    Stored MONOCHROME1 values are retained without display inversion. Modality LUT
    output is always float64. Compressed JPEG/JPEG2000 pixels may require optional
    ``pylibjpeg``, ``pylibjpeg-libjpeg``, ``pylibjpeg-openjpeg`` or ``python-gdcm``;
    decoder errors arise when lazy pixels are computed.

    Args:
        paths: Directory, file, or explicit sequence of files.
        series_uid: SeriesInstanceUID to select when more than one exists.
        frames: Enhanced multiframe indices forming one stack.
        modality_lut: Apply rescale or modality LUT to pixels.
        orientation_tolerance: Geometry importer orientation tolerance.
        slice_tolerance: Geometry importer slice tolerance.
        chunks: Dask chunks, ``auto`` by default; ``None`` reads eagerly.

    Returns:
        Framed pixel array with slices sorted by geometry.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If files, series, pixels or metadata cannot be represented.
        FileNotFoundError: If an explicit file or directory does not exist.
    """
    from . import DicomGeometry, from_datasets, from_enhanced, to_dataarray

    if isinstance(paths, str | os.PathLike):
        source = Path(paths)
        if not source.exists():
            raise FileNotFoundError(source)
        directory = source.is_dir()
        files = (
            sorted(path for path in source.iterdir() if path.is_file()) if directory else [source]
        )
    elif isinstance(paths, Sequence):
        directory = False
        if not paths:
            raise ValueError("paths must contain at least one file")
        if any(not isinstance(path, str | os.PathLike) for path in paths):
            raise TypeError("paths must contain filesystem paths")
        files = [Path(path) for path in paths]
    else:
        raise TypeError("paths must be a path or sequence of paths")
    if series_uid is not None and not isinstance(series_uid, str):
        raise TypeError("series_uid must be a string or None")
    if frames is not None and (not isinstance(frames, Sequence) or isinstance(frames, str | bytes)):
        raise TypeError("frames must be a sequence of frame indices or None")
    if not isinstance(modality_lut, bool):
        raise TypeError("modality_lut must be a bool")
    series: dict[str, list[tuple[Path, Dataset]]] = {}
    skipped: list[str] = []
    for path in files:
        if directory and path.name == "DICOMDIR":
            skipped.append(path.name)
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except InvalidDicomError as error:
            if directory:
                skipped.append(path.name)
                continue
            raise ValueError(f"invalid DICOM file {path}") from error
        uid = str(getattr(ds, "SeriesInstanceUID", ""))
        series.setdefault(uid, []).append((path, ds))
    if skipped:
        warnings.warn(f"skipped non-DICOM files: {', '.join(skipped)}", UserWarning, stacklevel=2)
    if not series:
        raise ValueError("no DICOM image files found")
    if series_uid is None:
        if len(series) != 1:
            listing = ", ".join(
                f"{uid!r} ({getattr(items[0][1], 'SeriesDescription', '')!s}, {len(items)} files)"
                for uid, items in series.items()
            )
            raise ValueError(f"multiple DICOM series: {listing}; supply series_uid")
        series_uid = next(iter(series))
    if series_uid not in series:
        raise ValueError(f"unknown series_uid {series_uid!r}; available: {', '.join(series)}")
    selected = series[series_uid]
    enhanced = len(selected) == 1 and hasattr(selected[0][1], "PerFrameFunctionalGroupsSequence")
    if enhanced:
        geometry: DicomGeometry = from_enhanced(
            selected[0][1],
            frames=frames,
            orientation_tolerance=orientation_tolerance,
            slice_tolerance=slice_tolerance,
        )
    else:
        if frames is not None:
            raise ValueError("frames requires one enhanced multiframe file")
        geometry = from_datasets(
            [ds for _, ds in selected],
            orientation_tolerance=orientation_tolerance,
            slice_tolerance=slice_tolerance,
        )
    blocks = []
    for path, metadata in selected:
        if int(getattr(metadata, "SamplesPerPixel", 0)) != 1 or getattr(
            metadata, "PhotometricInterpretation", None
        ) not in ("MONOCHROME1", "MONOCHROME2"):
            raise ValueError(f"unsupported pixel format in {path}: monochrome samples only")
        stored_dtype = _stored_dtype(metadata)
        dtype = np.dtype("float64") if modality_lut else stored_dtype
        if enhanced:
            for index in range(int(metadata.NumberOfFrames)):
                _pixel_transform(metadata, index)
        shape = (
            (int(metadata.NumberOfFrames), int(metadata.Rows), int(metadata.Columns))
            if enhanced
            else (int(metadata.Rows), int(metadata.Columns))
        )
        if chunks is None:
            block = _read_pixels(path, metadata, modality_lut, enhanced)
        else:
            block = da.from_delayed(  # type: ignore[no-untyped-call]
                delayed(_read_pixels)(path, metadata, modality_lut, enhanced),
                shape=shape,
                dtype=dtype,
            )
        blocks.append(block)
    pixels = (
        blocks[0]
        if enhanced
        else (
            np.stack(blocks) if chunks is None else da.stack(blocks)  # type: ignore[no-untyped-call]
        )
    )
    if chunks is not None and chunks != "auto":
        pixels = pixels.rechunk(chunks)  # type: ignore[union-attr]
    return to_dataarray(geometry, pixels)
