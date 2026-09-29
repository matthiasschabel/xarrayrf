# xarrayrf

Reference frames for [xarray](https://xarray.dev): record where an array's samples sit in a
(potentially but not necessarily) physical space, and keep that information correct as the array 
is sliced, combined, saved and resampled.

> **Status: pre-alpha.** The API and the on-disk format may change, and no release has been
> published. Full support for native xarray operations currently requires a patched xarray;
> see [Status](#status).

## The problem

An xarray coordinate tells you the value of each sample along a dimension: row 12, slice offset
5 mm, time 3 s. It does not tell you what physical space those values refer to, how the array's
axes map into it, or whether two arrays refer to the same space at all. That information usually
lives in format metadata (a DICOM header, a NIfTI affine, OME-Zarr transforms, a GeoTIFF CRS) and
is carried alongside the array by hand. It is easy to lose on a crop, easy to get wrong on a
transpose, and hard to reconcile between formats that disagree about axis order and direction
(DICOM's LPS against NIfTI's RAS, for example).

xarrayrf attaches that information to the array once and lets ordinary xarray operations carry
it. An array that has been cropped, transposed or combined with others still knows where its
samples are, and the information can be saved with the array and restored. Operations that would
make the geometry wrong raise an error rather than returning a plausible but incorrect result.
(This last guarantee is complete only with the patched xarray described under
[Status](#status).)

It is intended for any field that works with sampled data in a physical or reference space:
medical imaging, microscopy, brain atlases, remote sensing and physics simulations.

## Example

```python
import numpy as np
import xarray as xr
import xarrayrf as xrf
import xarrayrf.native  # registers the .rf accessor

# A 2-D frame in millimetres, and the affine placing each (row, column) sample in it.
stage = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("mm", "mm")))
pixel_to_stage = xrf.AffineTransform(
    source=xrf.ArrayCoordinates(("row", "column"), ("1", "1")),
    target=stage,
    matrix=((0.0, 0.5), (0.5, 0.0)),  # 0.5 mm pixels; rows run along y
    translation=(10.0, 20.0),
)
image = xr.DataArray(
    np.arange(12.0).reshape(3, 4),
    dims=("row", "column"),
    coords={"row": np.arange(3), "column": np.arange(4)},
)
framed = image.rf.frame(pixel_to_stage, dims=("row", "column"))

crop = (framed * 2).isel(column=slice(1, 3))  # ordinary xarray operations
crop.rf.geometry.point_at(row=0, column=0)  # x=10.5, y=20.0: the crop's first sample
crop.rf.resample_to(framed)  # back onto the full grid, still framed
```

Adding `framed` to an array framed in a different reference frame raises a `ValueError` that
names both frames and suggests resampling with an explicit transform, or `rf.assume_frame` if
the two are known to be the same space.

For a longer introduction on real data from microscopy, medical imaging, brain atlases and
satellite imagery, see [the xarrayrf tour](https://github.com/matthiasschabel/xarrayrf/blob/main/examples/xarrayrf_tour.ipynb).

## Concepts

- **Reference frame** (`ReferenceFrame`): a named physical space, such as a patient, a
  microscope stage or a map projection. Two arrays are comparable only if they are in the same
  frame. A frame either has a private identity (`ReferenceFrame.local`) or adopts an external
  one, such as a DICOM Frame of Reference UID (`ReferenceFrame.declared`).
- **Coordinate system** (`CoordinateSystem`): how points in a frame are written down, with ordered
  axes, units and, optionally, the direction each axis increases toward. One frame can be
  written in several coordinate systems; LPS and RAS are two coordinate systems for the same
  patient, and `coordinate_system_change` derives the conversion between them.
- **Transform** (`AffineTransform`, `CompositeTransform`, or your own class implementing the
  `Transform` protocols): a mapping from an array's own coordinates (`ArrayCoordinates`) into a
  frame, or from one frame to another.
- **Geometry** (`Geometry`, or `array.rf.geometry`): where a particular array's current samples
  sit. It reads the array's coordinate values as stored, so irregular slice positions or
  one-based labels are used exactly, with no spacing inferred.

The coordinate classes depend only on NumPy and can be used without xarray. The full
vocabulary and API contract are in
[the core interface](https://github.com/matthiasschabel/xarrayrf/blob/main/docs/core_interface.md).

## What it does

- **Framed arrays.** `array.rf.frame(transform, dims=...)` binds a transform to an array's
  coordinates. Selection, arithmetic, `where`, alignment, transposition, renaming and Datasets
  carry the binding. Operations whose geometry cannot be determined, such as `coarsen`, `stack`
  and `pad`, raise an error naming `rf.unframe()`, which removes the binding explicitly.
- **Mixed operands.** An array without a frame can be combined with a framed one; it supplies
  values, and the result keeps the framed operand's geometry.
- **Geometry queries.** The frame point of any sample (`point_at`, `points`), the regular
  lattice when there is one (origin, spacing, direction and the homogeneous affine used by NIfTI
  and ITK, 4×4 for a 3-D image in a 3-D frame), the location of given frame points among the samples (`positions_at`), and frame
  coordinates usable for nearest-point selection.
- **Resampling.** `resample` and `rf.resample_to` move values onto another array's samples.
  Conversions between coordinate systems of one frame, such as LPS to RAS, are applied
  automatically; moving between different frames requires an explicit transform.
- **Orientation.** Axis directions are written as explicit `from-to` tokens following
  [OME-NGFF RFC-4](https://ngff.openmicroscopy.org/rfc/4/). Letter codes such as `RAI` are
  refused, because libraries disagree about whether each letter names where an axis points from
  or to. `xarrayrf.anatomy` provides the anatomical vocabulary.
- **Persistence.** `rf.encode()` stores the binding as ordinary attributes before writing to
  netCDF or Zarr, and `rf.decode()` restores it after reading. An array read without
  `rf.decode()` is unframed.
- **Format readers.** Optional adapters read frames, transforms and lazy (dask) pixel data:

  ```python
  from xarrayrf import dicom, geotiff, ngff, nifti

  image = nifti.open("sub-01_T1w.nii.gz")
  label = nifti.open("sub-01_dseg.nii.gz", frame=image)  # declare a shared scanner frame
  series = dicom.open("study/002-t2_haste")  # one spatial stack, classic or enhanced
  level0 = ngff.open("image.zarr")  # OME-Zarr 0.4, 0.5 or 0.6
  scene = geotiff.open("scene.tif")  # projected CRSs only
  ```

  All adapters spell units the same way (`xarrayrf.units`), so frames read from different
  formats can match.

## What it does not do

- **No unit conversion.** Units are CF/UDUNITS strings compared exactly; `"mm"` and
  `"millimeter"` differ.
- **No registration.** xarrayrf applies transforms; estimating them is left to tools such as
  ITK or ANTs.
- **No geodetic coordinates.** Geographic (latitude/longitude), compound and vertical CRSs are
  refused; for general geospatial work see [rioxarray](https://corteva.github.io/rioxarray/) or
  [xproj](https://xproj.readthedocs.io).
- **No multi-image DICOM assembly.** `dicom.open` reads one spatial stack. Series with several
  images per position (echoes, b-values, time points) should be assembled by the application,
  which can then bind each array with `rf.frame`.
- **No array wrapper and no viewer.** Framed arrays are ordinary `xarray.DataArray` objects.
  Plotting and viewing libraries can read the frame coordinates, but xarrayrf does not display
  anything itself.

## Status

xarrayrf is pre-alpha. The following are planned but not implemented:

- Declared sample cells beyond a point or centred offset, such as slice thickness and intervals.
- Nonlinear transforms. The `Transform` protocols allow them, but no implementation or adapter
  provides one yet.
- Approximate or user-declared inverses for transforms without an exact inverse.
- Geodetic and spherical coordinate representations.

**xarray version.** Some operations need changes to xarray itself. These are maintained as a
small patch series on a [public xarray fork](https://github.com/matthiasschabel/xarray) and are
being submitted upstream, where some have already merged. With the patched xarray, every
operation in
[the operation inventory](https://github.com/matthiasschabel/xarrayrf/blob/main/docs/dev/binding/binding_operation_inventory.md)
behaves as described above. With released xarray, the main gaps are:

- combining a framed array with an unframed one (arithmetic, `where`, `align`) raises
  `AlignmentError`;
- `swap_dims` and Dataset reductions leave the binding incomplete, and every `.rf` member except
  `rf.unframe()` then raises;
- `coarsen`, `stack` and `pad` return an unframed result instead of raising;
- `isel`/`sel` with `drop=True` and `align(..., join="override")` return a result where they
  should raise, because they discard or overwrite coordinates the binding depends on.

The test suite marks these as expected failures on released xarray.

**Compatibility.** Persistence schema 1 and the adapter APIs are provisional and may change
before the first release.

## Installation

Install from a source checkout:

```sh
git clone https://github.com/matthiasschabel/xarrayrf && cd xarrayrf
python -m pip install .                 # core, with released xarray
python -m pip install '.[resample]'     # adds SciPy for resampling
python -m pip install '.[nifti,dicom,ngff,geotiff]'  # format readers
```

Python 3.12 or newer is required. The `resample` extra needs SciPy 1.18 or newer, and therefore
NumPy 2; the core supports NumPy 1.26.

To use the patched xarray:

```sh
python -m pip install "xarray @ git+https://github.com/matthiasschabel/xarray@xarrayrf-patches-4"
```

## Contributing

Development setup, tests and the xarray patch workflow are described in
[CONTRIBUTING.md](https://github.com/matthiasschabel/xarrayrf/blob/main/CONTRIBUTING.md).
The design rationale is recorded in
[docs/design.md](https://github.com/matthiasschabel/xarrayrf/blob/main/docs/design.md).

## License

MIT; see [LICENSE](https://github.com/matthiasschabel/xarrayrf/blob/main/LICENSE).
