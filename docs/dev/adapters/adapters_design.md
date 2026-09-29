# Format adapters

**Status:** Implemented
**Last updated:** 2026-09-28
**Scope:** `xarrayrf.nifti`, `xarrayrf.dicom`, `xarrayrf.ngff`, `xarrayrf.geotiff` and
`xarrayrf.anatomy`. The normative contract is the Adapters section of
[core_interface.md](../../core_interface.md); this note records why the adapters look as they do.

## Context

The core has frames, `ArrayCoordinates`, `AffineTransform`, `coordinate_system_change`,
`Geometry.lattice` and `resample`, but knows no file format. Adapters translate format metadata
into those objects and back. NIfTI came first (one affine per xform, no identity), then DICOM
(per-slice geometry, a Frame of Reference UID, declared slice thickness), OME-NGFF 0.6 (named
coordinate systems and transform graphs), and GeoTIFF (projected rasters with a CRS authority).

## Current Decision

### Shared contract

- **Import boundary.** Each adapter is an optional subpackage with its own extra. The core never
  imports an adapter; an adapter imports only the core (including `xarrayrf.native`),
  `xarrayrf.anatomy` and its format library. `tests/test_import_boundary.py` scans every module of
  each package and is proven able to fail.
- **Metadata functions** (`nifti.from_header`, `dicom.from_datasets`/`from_enhanced`,
  `ngff.from_multiscale`, `geotiff.from_profile`) consume format-library objects and never read
  pixels. They return a frozen declaration with `dims`, xarray-compatible
  `coords` (`(dim, values, {"units": ...})`, integer index values), a `transform` from
  `ArrayCoordinates` to the frame, the `frame`, and a `report` of `(code, message)` entries for
  every normalization, loss and assumed default.
- **Framing.** `to_dataarray(geometry, data)` checks a duck array's type and shape, builds the
  `DataArray` and calls `array.rf.frame` (shared helper `xarrayrf.native.frame_dataarray`). Lists
  and scalars are refused; dask stays lazy.
- **Readers.** One `open` per adapter, lazy by default like `xr.open_dataset`, returning exactly
  one framed `DataArray`. Each composes the metadata function and `to_dataarray`; no reader adds
  geometry logic. `chunks="auto"` binds dask arrays; `chunks=None` reads eagerly. Import reports
  are available only through the metadata functions; `open` does not change its return type for
  them. Each reader's extra therefore includes `dask[array]`.
- **Export** writes native metadata from a frame, a transform and current coordinates, refuses
  what the format cannot represent, and never resamples. After import xarrayrf never calls back
  into the format.
- **Units.** Adapters emit CF symbols (`m`, `mm`, `um`, `s`, `ft`, `US_survey_foot`) so frames
  from different formats compare equal. NGFF maps UDUNITS names through
  `xarrayrf.units.canonical` on import and `udunits_name` on export; GeoTIFF maps pyproj unit
  names through its own table. The core never relabels a unit.
- **Anatomy.** `xarrayrf.anatomy` (dependency-free) holds `VOCABULARY`, the 12 OME-NGFF RFC-4
  anatomical direction pairs, the `RAS`/`LPS` orientation tuples, and one constructor
  `patient_coordinate_system(orientation, unit)`, so NIfTI, DICOM and NGFF frames built from it
  convert through `coordinate_system_change`. A conformance test pins the token list to RFC-4.
- **Channel axes** stay non-geometry dimensions; interpolating across channels is meaningless.

### Frame identity

- A frame's identity decides whether framed arrays may combine; combination then also requires
  compatible mappings, so matching identity never merges mismatched grids.
- **Declared** frames (`ReferenceFrame.declared((namespace, value), system)`) are shared, external
  identities: `("dicom-frame-of-reference", uid)`, `("nifti-template", ...)`,
  `("templateflow", name)`, `("ome-zarr", "store/group#name")`, `("epsg", code)`.
- **Local** frames (`ReferenceFrame.local`) are minted fresh per import when the format declares no
  identity. Two separately opened NIfTI scanner-space files get distinct frames even with
  identical sforms, because identical headers from different subjects are common.
- **`frame=`** on `nifti.open`/`from_header` (a `ReferenceFrame` or a framed `DataArray`) and
  `frames=` on NGFF functions let callers declare sharing, for example an image and its label map.
- **`array.rf.assume_frame(other)`** re-targets an array's mapping to `other`'s complete
  declaration (identifier, definition, context). It refuses differing coordinate systems, since
  that is a conversion. It is the explicit escape hatch for cross-study, cross-modality or atlas
  comparison; genuinely different spaces keep distinct frames and are related by a registration
  transform plus `resample_to`.
- Template frames and `template=` are described under NIfTI below.

### NIfTI (`xarrayrf[nifti]`, nibabel)

Reads NIfTI-1 and NIfTI-2 headers; writes NIfTI-1 headers (or a copy of a supplied NIfTI-2).

- **Xform choice** follows NIfTI-1 and nibabel: `"best"` is a coded sform, else a coded qform,
  else refused (method 1 has no orientation; ANALYZE is out of scope). `"sform"`/`"qform"` refuse
  a zero code. Matrices come from `get_sform()`/`get_qform()`, so nibabel applies qfac. When both
  are coded and differ beyond `1e-6` of the voxel size, `sform-qform-disagree` is reported and
  neither is refused.
- **Array coordinates** `i, j, k` with `sample_offset=0.5`. A 2-D header keeps a scalar `k=0` so
  the transform retains the sform's third column.
- **Units** from raw `xyzt_units` codes: 1/2/3 are `m`/`mm`/`um`, 8/16/24 are `s`/`ms`/`us`;
  `Hz`, `ppm`, `rad/s` never become geometry. Unknown spatial units use `spatial_unit` (default
  `"mm"`, what nibabel users, ITK and FSL assume) and are reported; `spatial_unit=None` refuses.
- **Time.** Default `time=False`: `t` is a non-geometry coordinate `toffset + pixdim[4] * l`, so
  3-D and 4-D images of one subject share a 3-D frame. A non-positive `pixdim[4]` gives an integer
  index and `invalid-time-step`. `time=True` adds an unoriented time axis when a time unit is
  declared. Dims 5-7 are non-geometry index dims.
- **Supplied frame.** The affine is built against a RAS view of the supplied frame
  (`with_coordinate_system`, same identity) and composed with `coordinate_system_change`, so the
  transform targets the supplied frame itself. Frames whose change is not derivable are refused.
- **Identity** follows the *selected* xform code. Code 4 declares `("nifti-template", "MNI152")`,
  code 3 `("nifti-template", "Talairach")`, each with definition
  `{"space": ..., "variant": "unspecified"}` and empty context. `template=name` (codes 2-5,
  nonempty alphanumeric BIDS space label, not checked against a registry) declares
  `("templateflow", name)` with `{"space": name}`. `template=` excludes `frame=` and `time=True`.
  Codes 1, 2 and unnamed 5 mint local frames whose definition records the xform, codes and qfac.
  Header values never enter a declared definition; they go to the report, so files differing
  only in unused qform metadata share one frame.
- **Trust policy for templates: share.** Unnamed code-4 files share `MNI152`, matching FSL, ANTs
  and nilearn practice: they are meant to be in one world, if not to high precision. The
  imprecision is documented in the definition, not enforced. Named and unnamed variants are
  distinct identities; relate them with `assume_frame` or the same `template=`.
- **Time frames stay local.** `time=True` mints a local frame unless `frame=` is given: spatial
  normalization asserts a shared space, not a shared acquisition clock.
- **Export** (`to_header`) uses `Geometry.lattice(dims=...)` with `dims` in NIfTI voxel order,
  converts the frame to RAS (LPS is converted), writes a coded sform and a qform only when the
  spatial columns are orthogonal (tolerance `1e-6`, since float32 storage leaves residual shear
  on oblique images; nibabel strips it). Shear keeps the sform and reports
  `qform-unrepresentable`. 4-D export needs a time axis in `s`/`ms`/`us` with exactly zero
  space-time cross terms; step and origin become `pixdim[4]` and `toffset`. A supplied header
  keeps its time fields on 3-D export. Nonuniform, single-sample, field and non-affine geometry
  is refused.
- **Precision.** NIfTI-1 stores the sform as float32: general affines round-trip to
  `atol = 1e-6 * max|affine|`, dyadic affines exactly, NIfTI-2 to `1e-12` relative.
- **Reader.** `img.dataobj` applies `scl_slope`/`scl_inter`; the lazy dtype follows nibabel's
  scaled proxy dtype, which keeps the stored dtype under identity scaling.

### DICOM (`xarrayrf[dicom]`, pydicom)

Reads classic single-frame series and enhanced multiframe objects as one spatial stack. No
export.

- **Classic import** (`from_datasets`) needs `ImagePositionPatient`, `ImageOrientationPatient`,
  `PixelSpacing`, `Rows`, `Columns`. All datasets must agree on orientation (within
  `orientation_tolerance`), spacing, rows, columns and `FrameOfReferenceUID`; the first
  disagreeing index is named.
- **Orientation.** Row cosine `r = IOP[0:3]`, column cosine `c = IOP[3:6]`, orthonormalized
  (Gram-Schmidt on `c`) with the largest correction reported; normal `n = r x c`.
- **Pixel spacing** is `[row spacing, column spacing]` (PS3.3 C.7.6.2): `j` steps along `c` by
  `PixelSpacing[0]`, `i` along `r` by `PixelSpacing[1]`. Dims are `(k, j, i)`.
- **Slice ordering** sorts by `n . IPP` ascending; `order` maps sorted slices to input indices.
  Duplicates are refused when any adjacent gap is at most `slice_tolerance` times the end-to-end
  mean step (a median step is zero for `[0, 0, 0, 1]` and would miss them). In-plane IPP
  components must agree within `slice_tolerance` of the in-plane spacing (tilted or shifted
  stacks are not one grid).
- **Slice axis.** Uniform stacks (the `uniform_step` rule `Geometry.lattice` uses, at least two
  slices) get an index `k` and column `n * step`, so lattice and NIfTI export work. Nonuniform or
  single-slice stacks carry a `slice_offset` coordinate in mm from slice 0, with column `n`.
- **Thickness is not spacing.** `slice_intervals` holds each slice's own `SliceThickness` as an
  interval in the slice coordinate's values, or `None` when any slice lacks a positive thickness.
  It is returned as data and not attached until declared cells exist.
- **Frame** is declared `(FRAME_OF_REFERENCE_NAMESPACE, uid)` in LPS mm, or local without a UID.
  `patient_frame(uid)` returns the same frame so application readers can frame their own stacks
  in the same patient world.
- **Patient Position is not frame context.** Context is part of identity, and DICOM patient
  coordinates are defined relative to the patient, not the table. As context, two series sharing
  a UID but differing in whether they record Patient Position would stop being one frame. It is
  a result field (`patient_position`, `None` when datasets disagree).
- **Quadruped** `AnatomicalOrientationType` is refused; absent or `BIPED` is LPS.
- **Enhanced import** (`from_enhanced`) reads plane position, orientation and pixel measures from
  the shared or per-frame functional groups; a group in both is refused (C.7.6.16), as is a
  per-frame count differing from `NumberOfFrames`. Frames go through the same assembly; duplicate
  positions are refused unless `frames=` selects one stack.
- **Equipment mapping** (`equipment_transform`): patient frame to a fresh local equipment frame
  (unoriented `x, y, z` mm). Only `ISOCENTER` and a finite rigid matrix with homogeneous last row
  are accepted (C.7.6.21). Equipment frames never share identity across datasets.
- **Spatial registration** (`registrations`): one transform per `RegistrationSequence` item, from
  its UID's frame to the dataset's. Matrix order follows C.20.2.1.1: `M1 @ M2 @ ...` on column
  vectors, so the last matrix acts first (pinned by a noncommuting test). `RIGID`, `RIGID_SCALE`
  and `AFFINE` constraints are checked, not trusted. Image-only items, missing UIDs and
  deformable registration are refused; resolving referenced images is the caller's job.
- **Reader.** Directories skip non-DICOM files and `DICOMDIR` with one `UserWarning`; explicit
  file lists raise. Several series without `series_uid` raise, listing them. Only
  `SamplesPerPixel == 1` MONOCHROME1/2; stored MONOCHROME1 values are not display inverted.
  `modality_lut` uses `pydicom.pixels.apply_modality_lut` for classic files and each frame's
  `PixelValueTransformationSequence` (per-frame, shared, then top level) for enhanced files,
  refusing a nonlinear LUT there. One delayed `dcmread(...).pixel_array` per slice. Missing
  compressed-syntax decoders surface pydicom's error at compute time, chained.
- **Single-stack limit.** Several images per slice position (multi-echo, diffusion b-values,
  dynamics) and mixed-orientation localizers are refused. Grouping them needs vendor rules and
  private tags, which is application policy: an application reader assembles the extra
  dimension and frames each spatial stack with `patient_frame`.

### OME-NGFF (`xarrayrf[ngff]`, ome-zarr-models, zarr)

Imports 0.4, 0.5 and 0.6 metadata; exports 0.6 only. Input is ome-zarr-models v06 objects or the
equivalent JSON attributes, validated through the same models.

- **Normative 0.6 points relied on:** `affine` is `M x (N+1)` with translation last;
  `mapAxis[i]` names the input axis that becomes output `i`; `byDimension` covers each output
  axis exactly once; a `path`-only endpoint is the array's implicit coordinate system; the centre
  of element 0 is coordinate 0 (so `sample_offset=0.5`, unit `"1"`).
- **Identity.** With `store`, a named system is declared
  `("ome-zarr", f"{store}/{group}#{quote(name)}")` (no trailing slash, normalized group, the slash
  dropped at the root). Without `store`, frames are local and reused across calls via `frames`,
  since a name alone is not an identity across stores.
- **Transforms.** `identity`, `scale`, `translation`, inline (rectangular) `affine`, `rotation`,
  `mapAxis`, `projectAxis`, `sequence`, `byDimension` and `bijection` collapse into one
  `AffineTransform`. `sequence` applies members in listed order (`M = Mn @ ... @ M1`, the reverse
  of DICOM's product; pinned by `sequence.json`). `bijection` uses `forward`; the declared
  inverse is checked within `1e-12` relative and reported. Array-backed parameters,
  `displacements` and `coordinates` are refused.
- **Axes.** Missing unit becomes `None`; an unnamed axis or non-string unit is refused.
  Identity-mapped channel and other discrete axes are dropped from geometry and reported; mixed
  discrete axes are refused. `discrete` and `longName` are reported as lost. RFC-4 `orientation`
  is read and written through `VOCABULARY`; unknown tokens or orientation on non-space axes are
  refused.
- **Multiscales and scenes.** `from_multiscale` needs `shapes` (it reads no arrays) and returns a
  shared intrinsic frame, levels keyed by path, and additional transforms. `from_scene` needs
  `systems` for referenced images' coordinate systems, because scenes name them without declaring
  them; unresolved endpoints are refused.
- **0.4/0.5.** Parsed with `ome_zarr_models.v04`/`v05`, converted through `to_version("0.6")`,
  then the multiscale-level scale and translation are folded into each level's single intrinsic
  transform. 0.4/0.5 define one physical space, and `to_version` would move it into a separate
  output system that `open` must not bind. Time keeps its calibration; path-based transforms are
  refused. Most public OME-Zarr (IDR) is 0.4, which is why it is supported.
- **Reader.** `open` binds the first (full-resolution) level unless `level` is given, selects
  among several multiscales by index or name (ambiguity raises with a listing), and uses
  `dask.array.from_zarr`. For 0.6 it binds the intrinsic system; other systems stay reachable
  through `from_multiscale(...).transforms`.
- **Export.** `to_multiscale_level` writes one regular affine level. v06 restricts dataset
  transforms to scale then translation, so a diagonal lattice uses the frame's axes directly,
  while a non-diagonal lattice gets an array-aligned intrinsic system (column norms, zero
  translation) plus a multiscale-level `affine` to `frame_name`, reported as
  `intrinsic-synthesized`. Rectangular levels are refused (full-model validation cannot hold
  them); `to_transform` exports frame-to-frame affines, including rectangular ones, and needs
  `names` unless a frame has an `ome-zarr` identity. Every xarrayrf field NGFF cannot hold
  (identity beyond the name, definition, context, role, display, non-centred offsets, other
  vocabularies) is reported.
- **Persistence gate.** `tests/test_ngff_persistence_gate.py` round-trips a named lossless subset
  of the vendored 0.6 examples (`tests/fixtures/ngff_0_6/`) through `encode`/`decode` and checks
  that the rest are refused or reported. Schema 1 stays provisional; see
  [persistence_design.md](../architecture/persistence_design.md).

### GeoTIFF (`xarrayrf[geotiff]`, rasterio, pyproj)

Reads projected GeoTIFF/COG. No export.

- **Import** (`from_profile`) takes a rasterio profile plus the `AREA_OR_POINT` tag and returns
  dims `(band, row, column)`; only `(row, column)` is geometry, and `band` has 1-based
  coordinates.
- **Area and Point.** GDAL keeps an area-based, corner-referenced GeoTransform for both tags
  (RFC 33), so every sample sits at rasterio's `xy(row, col, offset="center")`. `Area` declares
  centred cells (`sample_offset=0.5`); `Point` declares point samples (`None`). Rotated and
  sheared affines are kept exactly.
- **Identity.** A CRS with an exact authority code (`to_authority(min_confidence=100)`) is
  declared `("epsg", code)` (other authorities lowercased); otherwise a local frame holding the
  WKT. Axis names, order and units come from `pyproj.CRS.axis_info`; rasterio's `(x, y)` rows are
  permuted by direction into the authority's order (EPSG:3035 lists northing first). Units:
  `metre`, `foot`, `US survey foot` only.
- **Refused CRSs.** Geographic and other angular CRSs ("angular coordinates are not
  representable yet": Cartesian systems refuse angles), compound, vertical, and anything without
  exactly one easting and one northing axis.
- **Reader.** One delayed windowed read per dask chunk, each task opening the file (rasterio
  handles are not thread-safe). `bands` selects 1-based bands. `nodata` goes to `attrs`; masks
  and scale/offset are not applied.
- **Failure-mode comparison.** `explorations/geo_failure_modes.py` runs known rioxarray, xproj
  and rasterix lifecycle failures (stale transform after strided `isel`, silent CRS mixing, CRS
  lost in joins, `join="override"` rewriting a CRS) beside xarrayrf, recording each other-library
  outcome as `reproduced`, `fixed` or `could-not-run`. On stock xarray the override case rebinds
  the CRS; it is closed only by the `check_override` hook in the patched xarray lane, and the
  xarrayrf assertion is a strict xfail elsewhere.

## Alternatives Considered

- **Suffixed reader names** (`open_image`, `open_series`, `open_multiscale`) or `open_dataset`:
  each adapter has one natural unit and returns a `DataArray`.
- **xarray backend entrypoints** (`engine="nifti"`): more machinery for the same result; revisit
  on user demand.
- **Identity-less frames compatible by value:** fails open for different subjects scanned with
  one protocol. A global identity-ignore switch would silently affect every operation;
  `assume_frame` is explicit and per array.
- **Local frames for NIfTI codes 3/4:** first chosen, then reversed; standard template codes exist
  to declare a shared space, and FSL, ANTs and nilearn treat them so.
- **Returning both sform and qform frames:** nothing consumes it.
- **Time as NIfTI geometry by default:** 3-D and 4-D files of one subject would stop sharing a
  frame.
- **Always a DICOM `slice_offset` coordinate:** uniform stacks keep an index `k` so lattice and
  NIfTI export apply.
- **Series grouping (`stack_by`) in the DICOM adapter:** grouping echoes, b-values and dynamics
  relies on vendor-specific tags and heuristics; it belongs to applications.
- **ngff-zarr as the NGFF dependency** (itkwasm, rich) or a hand-written 0.6 parser:
  ome-zarr-models already models and validates every version.
- **Keeping NGFF sequences as `CompositeTransform`:** unnecessary while every member is affine;
  revisit with nonlinear members.
- **rioxarray as the GeoTIFF reader:** pulls in its accessor and CRS model; rasterio is the engine
  underneath anyway. The module is named for the format tested, not "GDAL raster".

## Deferred Work

- Declared cells consuming DICOM `slice_intervals` (see
  [geometry_and_resampling_design.md](../architecture/geometry_and_resampling_design.md)).
- NIfTI: dual-xform import, 2-D and single-slice export, intent codes for vector and displacement
  fields, `slice_duration`/`slice_code`, file writers beyond `to_header`.
- NIfTI product spacetime frames (shared template space with a private acquisition clock).
  Trigger: combining template-registered time series across runs.
- DICOM: geometry export, mixed-orientation splitting, quadruped orientation, Real World Value
  Mapping, deformable registration, decoder extras.
- NGFF: nonlinear members through an engine, array-backed parameters, `discrete`/`longName`,
  declared `bijection` inverses, 0.4 `omero` metadata and labels.
- GeoTIFF: export, nodata/mask and scale/offset, CF `grid_mapping` import, geographic and
  celestial coordinates (angular representations in the core, a core decision).
- Full native lifecycle support for `rf.frame` bindings (release blocker in the core contract).

## Next Steps

None beyond the deferred items. `explorations/real_data_check.py` (read-only, takes paths) is the
manual check against real NIfTI, DICOM and OME-Zarr data after adapter changes.
