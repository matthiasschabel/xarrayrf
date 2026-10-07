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

## A first example

Install the core from a [source checkout](#installation) to run this example. Resampling
requires the optional `resample` extra; complete native-operation support requires the
[patched xarray](#status).

```python
import numpy as np
import xarray as xr
import xarrayrf as xrf
import xarrayrf.native  # registers the .rf accessor

# A 2-D frame in millimetres, and the affine placing each (j, i) sample in it.
stage = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("mm", "mm")))
pixel_to_stage = xrf.AffineTransform(
    source=xrf.ArrayCoordinates(("j", "i"), ("1", "1")),
    target=stage,
    target_axes=("x", "y"),
    basis_vectors={"j": (0.0, 0.5), "i": (0.5, 0.0)},  # 0.5 mm pixels
    translation=(10.0, 20.0),
)
image = xr.DataArray(
    np.arange(12.0).reshape(3, 4),
    dims=("j", "i"),
    coords={"j": np.arange(3), "i": np.arange(4)},
)
framed = image.rf.frame(pixel_to_stage, dims=("j", "i"))

crop = (framed * 2).isel(i=slice(1, 3))  # ordinary xarray operations
crop.rf.geometry.point_at(j=0, i=0)  # x=10.5, y=20.0: the crop's first sample
```

With `python -m pip install '.[resample]'`, resample the crop back onto the full grid:

```python
crop.rf.resample_to(framed)  # still framed
```

Each named vector says how the frame coordinates change for a unit increase in that source
coordinate. Its components and the translation follow the explicitly asserted `target_axes`
order, which must match the frame. Index names do not imply physical directions. If you
already have coefficients, use `AffineTransform.from_matrix(source=..., target=...,
matrix=..., translation=...)`; matrix rows follow target axes and columns follow source axes.

Adding `framed` to an array framed in a different reference frame raises a `ValueError` that
names both frames and suggests resampling with an explicit transform, or `rf.assume_frame` if
the two are known to be the same space.

For a longer introduction on real data from microscopy, medical imaging, brain atlases and
satellite imagery, see [the xarrayrf tour](https://github.com/matthiasschabel/xarrayrf/blob/main/examples/xarrayrf_tour.ipynb).

The [interface specification](https://github.com/matthiasschabel/xarrayrf/blob/main/docs/core_interface.md)
describes explicit Grid coordinate replacement, declared geometry order after transposition,
and indexed assignment's destination-frame and coordinate checks.

## Architecture

xarrayrf separates *which space* data lives in, *how* an array's coordinates map into it, and
*which samples* an array has. Built-in affine declarations and Grid coordinates are immutable.
A Grid retains its transform by reference: extension transforms must keep endpoints, behavior
and scalar-boolean equality stable. Hashing a Grid requires a hashable transform with a stable,
equality-consistent hash; unhashable transforms still support queries, equality and binding.
Composite stability depends on its members. The binding on a `DataArray` ties the declaration
to actual coordinates, and xarray's own operations carry the binding.

```text
 format adapters                  value objects (NumPy only)
 dicom · nifti · ngff · geotiff   ReferenceFrame ── CoordinateSystem ── DirectionVocabulary
          │                            ▲
          │ build                      │ target
          ▼                            │
       Grid  ── transform: ArrayCoordinates ──► ReferenceFrame   (AffineTransform,
  (sampling, no pixels)                                          CompositeTransform, ...)
          │ rf.frame(grid) / frame_array(data, grid)      ▲ rf.grid (snapshot)
          ▼                                                │
  framed DataArray ── .rf accessor: grid · resample_to · assume_frame · encode/decode
          │       └── rf.geometry: Geometry, a live view of the array's current samples
          │
  ordinary xarray operations (isel, sel, arithmetic, align, concat along time, Datasets, ...)
```

| Component | What it is | Use it to |
|---|---|---|
| `ReferenceFrame` | The identity of a space: a patient, a slide, a map projection. **Complete** frames have an identity: `declared` (an external name such as a DICOM Frame of Reference UID or a template space) or `local` (created on purpose and shared by passing the object). **Anonymous** frames (`ReferenceFrame.anonymous`, `is_anonymous`) have geometry but no known identity. | Decide whether two arrays can be compared. |
| `CoordinateSystem`, `DirectionVocabulary` | How points in a frame are written: ordered axes, units, and optionally the direction each axis increases toward. One frame can be written in several systems (LPS and RAS for one patient). | Convert between systems with `coordinate_system_change`. |
| `ArrayCoordinates` | An array's own coordinate axes as a transform endpoint, with units and where each sample sits in its cell (`sample_offset`). | Be the source of the transform that places samples. |
| `AffineTransform`, `CompositeTransform`, the `Transform` protocols | Point mappings from array coordinates into a frame, or between frames (a registration result). | Place samples; relate frames explicitly. |
| `Grid` | A sampling declaration with frozen coordinates and no pixels: the transform plus the coordinate values of each geometry dimension and, optionally, declared cell intervals (slice thickness). | Describe a target for resampling, a region of interest, or any geometry you need without an array; persist it with `encode`. |
| `Geometry` (`array.rf.geometry`) | A live view of a framed array's current samples, re-read on every query. | Ask where samples are (`point_at`, `points_at`) and which samples are at given points (`positions_at`). |
| `Lattice` | The regular special case: origin, spacing, direction and the homogeneous affine used by NIfTI and ITK. | Interoperate with affine-based libraries. |
| The binding (`array.rf`) | A private xarray index that owns the geometry coordinates of a framed `DataArray`, so selection, arithmetic and alignment carry or refuse it. | Frame, query, resample, re-identify and persist arrays. |
| `xarrayrf.anatomy` | Anatomical vocabulary and grid operations: `orientation_codes`, `reoriented`, `cardinal_grid`. | Name orientations and reformat to axial, coronal or sagittal planes. |
| Adapters | `dicom`, `nifti`, `ngff`, `geotiff`: read headers into grids and frames, and pixels lazily. | Load data with its geometry attached. |

Typical uses, and where they live:

- **Load and combine data from different formats**: the adapters plus `rf.resample_to`, once
  the data share a frame (a declared identity, or one asserted with `frame=` or
  `rf.assume_frame`); conversions between coordinate systems of one frame (LPS and RAS) are
  then applied automatically.
- **Process without losing geometry**: ordinary xarray code on framed arrays.
- **Resample or reformat**: `rf.resample_to(target)` onto a framed array, a `Geometry` or a
  `Grid`; `anatomy.cardinal_grid` builds axial, coronal or sagittal targets.
- **Work with geometry but no pixels**: `Grid` for regions of interest, viewer planes or
  registration domains; `array.rf.grid` and `frame_array` convert both ways.
- **Describe slices with thickness or gaps**: declared intervals on a `Grid` or binding; the
  `domain="cells"` queries and resampling use them.
- **Relate data whose source names no space**: anonymous frames, completed explicitly with
  `frame=` when loading (every reader) or `rf.assume_frame` afterwards (any framed array).
- **Save and restore**: `rf.encode`/`rf.decode` for arrays, `encode`/`decode` for value objects.

The value objects depend only on NumPy and can be used without xarray. The full vocabulary and
API contract are in
[the core interface](https://github.com/matthiasschabel/xarrayrf/blob/main/docs/core_interface.md).

## More examples

### Grids, slice thickness and reformatting

```python
from xarrayrf import anatomy
from xarrayrf.native import frame_array

# Four 2 mm thick axial slices, 3 mm apart, of 5 x 6 one-millimetre voxels, in LPS millimetres.
patient = xrf.ReferenceFrame.local(anatomy.patient_coordinate_system(anatomy.LPS, "mm"))
index_to_patient = xrf.AffineTransform(
    source=xrf.ArrayCoordinates(("k", "j", "i"), ("1", "1", "1"), sample_offset=(0.5, 0.5, 0.5)),
    target=patient,
    target_axes=("x", "y", "z"),
    basis_vectors={"k": (0.0, 0.0, 3.0), "j": (0.0, 1.0, 0.0), "i": (1.0, 0.0, 0.0)},
    translation=[-2.5, -2.0, 0.0],
)
k = np.arange(4)
grid = xrf.Grid(
    index_to_patient,
    {"k": ("k", k), "j": ("j", np.arange(5)), "i": ("i", np.arange(6))},
    # Declared cells in the k coordinate's own units: 2 mm thick on a 3 mm step is +-1/3.
    intervals={"k": np.stack([k - 1 / 3, k + 1 / 3], axis=-1)},
)

grid.point_at(k=1, j=0, i=0)  # [-2.5, -2.0, 3.0] mm: a sample's position
grid.points_at([[0.5, 0.0, 0.0]])  # fractional positions in, points out
grid.positions_at([[0.0, 0.0, 4.5]])  # [[1.5, 2.0, 2.5]]: and back again
anatomy.orientation_codes(grid)  # "SPL": index increases toward S, P and L

# The same geometry on pixels, without copying or reading them.
volume = frame_array(np.arange(120.0).reshape(4, 5, 6), grid)
assert volume.rf.grid == grid
assert volume.isel(k=slice(1, 3)).rf.grid == grid.isel(k=slice(1, 3))

# Reformat to a sagittal grid ("RIP") covering every source cell at 1 mm, then resample.
# The target covers cells, so resample in the cells domain to fill its edges.
sagittal = anatomy.cardinal_grid(grid, "sagittal", spacing=1.0)
resliced = volume.rf.resample_to(sagittal, domain="cells", method="nearest")
anatomy.orientation_codes(resliced.rf.grid)  # "RIP"

# Grids are values: compare them, select from them, save them.
assert xrf.decode(xrf.encode(grid)) == grid
```

`orientation_codes` names, for each dimension, the direction its index increases toward,
written either as patient letters (`R` means toward the right, as in nibabel's
`aff2axcodes`) or as explicit RFC-4 tokens such as `"inferior-to-superior"`. Codes are
nearest-axis labels, so a strongly oblique grid's codes change as it rotates through 45°; read
`CoordinateSystem.axis_codes` for the remaining angle. `reoriented` permutes and reverses
dimensions without moving any sample.

### Data whose source does not name its space

A scanner-space NIfTI file, or a DICOM series without a Frame of Reference UID, has geometry
but no identity. Its frame is **anonymous**: fully usable on its own, but never assumed to be
the same space as anything else.

```python
def acquisition() -> xr.DataArray:
    """Stands in for nifti.open(...) on a file that names no space."""
    frame = xrf.ReferenceFrame.anonymous(anatomy.patient_coordinate_system(anatomy.RAS, "mm"))
    to_frame = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("i", "j", "k"), ("1", "1", "1")),
        target=frame,
        matrix=np.eye(3),
        translation=np.zeros(3),
    )
    coords = {name: (name, np.arange(n)) for name, n in zip("ijk", (3, 4, 2))}
    return frame_array(np.ones((3, 4, 2)), xrf.Grid(to_frame, coords))


t1, t2 = acquisition(), acquisition()
assert t1.rf.reference_frame.is_anonymous
try:
    t1 + t2
except ValueError as error:
    refusal = str(error)  # names the anonymous operands and says rf.assume_frame alone suffices

combined = t1 + t2.rf.assume_frame(t1)  # your statement that both share one space
```

With files, say so at load time instead: `nifti.open("t2.nii.gz", frame=t1)`; every reader
takes `frame=`. Both `frame=` and `rf.assume_frame` follow one contract: they adopt the other
frame's identity and apply any derivable coordinate-system change, such as RAS to LPS.
Asserting a shared space never makes two different grids compatible; resample one onto the
other with `rf.resample_to`.

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
- **Grids.** `Grid` describes sampling without pixels and converts both ways with framed arrays
  (`rf.grid`, `rf.frame(grid)`, `frame_array`). `Grid.isel` and `Grid.sel` use xarray's own
  selection rules.
- **Declared cells.** Samples can declare their extent as intervals (slice thickness, gaps,
  overlap). They follow selection, must agree when arrays align, and extend the outer bounds
  of `domain="cells"` queries and resampling; interior gaps remain interpolated. For samples
  at 0 and 4 mm with intervals `[-0.5, 0.5]` and `[3.5, 4.5]` and values 0 and 8, querying
  2 mm gives position 0.5 and linear value 4 in either domain. At -0.25 mm, only the cells
  domain admits the query and resampling holds the edge value 0 (the samples domain fills
  outside points). DICOM import declares intervals from `SliceThickness`.
- **Resampling.** `resample` and `rf.resample_to` move values onto another array's samples or
  onto a `Grid`. Conversions between coordinate systems of one frame, such as LPS to RAS, are
  applied automatically; moving between different frames requires an explicit transform.
- **Orientation.** Coordinate-system axis directions are written as explicit `from-to` tokens
  following [OME-NGFF RFC-4](https://ngff.openmicroscopy.org/rfc/4/); `CoordinateSystem` refuses
  letter codes such as `RAI`, because libraries disagree about whether a letter names where an
  axis points from or to. The `xarrayrf.anatomy` functions alone accept patient letters, with
  one stated convention: each letter names the direction an index increases toward (`"RAS"`,
  as nibabel's `aff2axcodes`). Named planes keep DICOM display order in-plane (rows, then
  columns), with the slice axis toward S, P and R: axial `SPL`, coronal `PIL`, sagittal `RIP`.
- **Frame identity.** Sources that name their space (a Frame of Reference UID, a template space,
  a CRS authority code, an NGFF store) give declared frames that match across files. Sources that
  do not give anonymous frames; relating them is the caller's explicit act (`frame=` on every
  reader, or `rf.assume_frame`), and refusals say which remedy applies.
- **Persistence.** `rf.encode()` stores the binding, including declared intervals, as ordinary
  attributes before writing to netCDF or Zarr, and `rf.decode()` restores it after reading. An
  array read without `rf.decode()` is unframed.
- **Format readers.** Optional adapters read frames, transforms and lazy (dask) pixel data:

  ```python
  # Needs the data files named below.
  from xarrayrf import dicom, geotiff, ngff, nifti

  image = nifti.open("sub-01_T1w.nii.gz")  # scanner space: an anonymous frame
  label = nifti.open("sub-01_dseg.nii.gz", frame=image)  # declare the shared scanner frame
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

- Concatenating framed arrays along a geometry dimension (stitching slabs). Concatenation along
  other dimensions, such as time, keeps the binding when the grids are identical; differing
  grids align like any join, so pass `join="exact"` to require identical ones.
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
