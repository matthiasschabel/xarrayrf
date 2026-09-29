"""Read real adapter data and print geometry samples without writing files."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from xarrayrf import dicom, ngff, nifti


def read(path: Path) -> xr.DataArray:
    """Choose an adapter from a local path suffix or directory."""
    name = path.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        return nifti.open(path)
    if name.endswith(".zarr"):
        return ngff.open(path)
    return dicom.open(path)


def show(path: Path, image: xr.DataArray) -> None:
    """Print array metadata and first/last geometry sample positions."""
    print(f"{path}: dims={image.dims}, shape={image.shape}, dtype={image.dtype}")
    print(f"  frame={image.rf.reference_frame.identifier}")
    points = image.rf.geometry.points()
    dimensions = image.rf.geometry_dims
    for corner in (0, 1):
        indices = {dim: (0 if corner == 0 else image.sizes[dim] - 1) for dim in dimensions}
        print(f"  corner {indices}: {points.isel(indices).values.tolist()}")


def main() -> None:
    """Inspect each supplied path, optionally pairing a NIfTI label image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--label", type=Path, help="NIfTI label to pair with one image")
    args = parser.parse_args()
    if args.label is not None and len(args.paths) != 1:
        parser.error("--label requires exactly one image path")
    for path in args.paths:
        image = read(path)
        show(path, image)
        if args.label is not None:
            if not path.name.lower().endswith((".nii", ".nii.gz")):
                parser.error("--label requires a NIfTI image")
            label = nifti.open(args.label, frame=image.rf.reference_frame)
            show(args.label, label)
            dataset = xr.Dataset({"image": image, "label": label})
            print(f"  dataset image framed={dataset['image'].rf.is_framed}")
            print(f"  dataset label framed={dataset['label'].rf.is_framed}")


if __name__ == "__main__":
    main()
