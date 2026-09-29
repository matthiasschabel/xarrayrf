# Nonlinear geometry: staging and the displacement-field transform

**Status:** Active (draft plan; not started)
**Last updated:** 2026-09-28
**Scope:** The order in which xarrayrf takes on nonlinear geometry, and a detailed plan for the
first stage, field-backed transforms between Cartesian frames (deformable registration). Later
stages are sketched with their acceptance cases and open decisions.

## Context

[The design](../../design.md) §7 allows `world = F(c)` to be nonlinear, bounded and partially
invertible, and names three non-affine representations: sampled coordinate fields, field-backed
transforms and external providers. Only affine transforms exist today. The capability protocols
(`SupportsPoints`, `SupportsJacobian`, `SupportsInverse`) and `CompositeTransform` already admit
nonlinear members, and `resample`'s general path already maps target points through any
`SupportsPoints` frame transform (`_general_map` in `src/xarrayrf/_resample.py`).

Three candidate directions were considered: deformable image registration, geographic (GIS)
frames and Astropy celestial frames. Earth-system model output (GCMs, ocean and atmosphere
reanalyses) is a large share of xarray's use and also needs spherical coordinates, so it is
treated as a first-class target of the later stages rather than as part of "GIS".

Three facts shape the staging:

1. **Geographic and celestial data need angular coordinate systems first.** Longitude and
   latitude, or right ascension and declination, are periodic and singular at the poles.
   `CoordinateSystem` has a `representation` parameter but only `"cartesian"` exists, and the
   [relativity notes](relativity_notes.md) list the open decisions for angular charts.
2. **Deformable registration needs no change to the core value objects.** Source and target
   frames are Cartesian with the same axes and units; a field is a transform between them.
3. **Registration and curvilinear grids share one mechanism.** A displacement field and a 2-D
   latitude/longitude grid are both a regular grid of stored values with a declared
   interpolant and a bounded domain. Stage 1 builds that mechanism on Cartesian data; stage 3
   reuses it with array coordinates as the source.

### A registration is a transform between frames

A registration maps a source frame to a target frame. It is not tied to the two images used to
estimate it: it applies to anything located in its source frame (an image, a label map, a mesh,
landmarks, another transform) and produces a result in its target frame. Registration tools'
"fixed" and "moving" images only fix the direction:

- **Images are resampled by pulling.** Each output sample in frame B needs its location in frame
  A, so resampling an image from A into B uses the B → A transform. `resample(source, target,
  transform=...)` already takes the transform from the target's frame to the source's.
- **Points and meshes are pushed.** Each point in A is mapped forward through A → B.

One field therefore serves one of the two uses cheaply; the other use needs its inverse. The
field's own sampling grid is its declared domain in the source frame, not a constraint on what
it can be applied to.

## Current Decision

### Staging

| Stage | Capability | Unlocks | Core change |
|---|---|---|---|
| 1 | Field-backed transforms between Cartesian frames | Deformable registration (ITK/ANTs warps, NGFF 0.6 `displacements`, `coordinates`, `bijection`) | New transform classes; per-point validity in `resample`; array-valued persistence |
| 2 | Angular coordinate systems | Rectilinear longitude/latitude grids (most GCM and reanalysis output), celestial coordinates | New `representation` values, periodic and bounded axes |
| 3 | Curvilinear coordinate fields | 2-D longitude/latitude grids (ocean models, cubed-sphere tiles, satellite swaths), Cartesian curvilinear grids | Array-coordinate transforms backed by coordinate arrays; binding ownership of 2-D coordinates; inverse lookup by search |
| 4 | Provider adapters | Reprojection between CRSs (pyproj), celestial WCS (Astropy, APE 14), CF `grid_mapping` | Adapters only |

Curvilinear coordinate fields come before the WCS provider adapters. They are far more common in
xarray workloads than FITS WCS, and a provider adapter is thin once angular systems exist,
because the provider does the mathematics. For earth-system data, however, curvilinear grids
are almost always longitude/latitude, so stage 3 depends on stage 2; a Cartesian-only stage 3
would serve swaths in projected coordinates and distortion-corrected imaging but not GCM grids.

A CF adapter (`grid_mapping`, 1-D and 2-D `lat`/`lon`, `bounds`) grows with stages 2 to 4
rather than being a stage of its own: CF is how most earth-system data declares its geometry.

### Stage 1: field-backed transforms

**Representations.** Two transform classes, both implementing `SupportsPoints` and
`SupportsJacobian`, with frame endpoints:

- `DisplacementFieldTransform(*, source, target, field, grid)`: `y = x + u(x)`. Requires source
  and target to have equal coordinate systems (axes, units, types), since adding a displacement
  is meaningless otherwise. This is the ITK and ANTs representation and NGFF's `displacements`.
- `PositionFieldTransform(*, source, target, field, grid)`: `y = f(x)`, the stored value is the
  target position itself. No coordinate-system restriction. This is NGFF's `coordinates`.

`grid` is the affine mapping from field sample positions into the source frame (a `Lattice`-like
declaration: origin, spacing, direction), so a field is a regular grid in its source frame. Its
extent is the transform's domain. `field` has shape `grid shape + (n_target_axes,)` and is stored
as float32 or float64, keeping the producer's dtype (registration tools commonly write float32);
evaluation is always in float64. It is copied on construction into immutable storage, following
`frozen_float_array` in `_validation.py`, and exposed only as views that cannot be made writeable
back into that storage.

**Evaluation.** Linear interpolation of the field, ITK's default for displacement fields. The
interpolant is part of the transform's declaration; NGFF also permits nearest and cubic, and an
importer refuses an interpolant the transform does not implement rather than substituting
linear. Points outside the grid raise, as `transform_point` requires. The Jacobian is
the derivative of the interpolant (`I + ∂u/∂x` for displacements), with a stated tie rule on cell
faces.

**Equality and hashing.** Exact and structural, as bindings require: endpoints, grid,
interpolant, dtype, shape and field bytes, with signed zeros normalized as the existing value
objects do. A SHA-256 digest computed once at construction serves as the hash and as a fast
inequality check; equal digests are confirmed by comparing bytes, so equality never rests on the
digest alone.

**Inverse.** `inverse()` stays exact, as the core interface requires, and a field transform does
not provide it. A producer-declared pair (an NGFF `bijection`; an ANTs `1Warp` and
`1InverseWarp`, which are inverses between the fixed frame and the intermediate frame after the
affine) is exposed through a separately named capability, `declared_inverse()`, which returns the
declared inverse without verifying it. A helper reports the round-trip residual of a pair on
caller-chosen points, so a user can check a pair before trusting it.
`CompositeTransform.inverse()` keeps its current rule and therefore refuses a chain containing a
field. Numerical inversion is
the separate approximate-inverse capability that the core interface already defers.

**Resampling.** `resample` already accepts a non-affine frame transform on its general path
(`_general_map`). It currently fails the whole block if any target point is outside the
transform's domain, which makes warping unusable at field edges. Stage 1 adds per-point validity
for this use only: a transform may implement an optional
`transform_point_masked(points) -> (values, valid)`, and `resample` uses it when present, filling
invalid samples with `fill_value`. `CompositeTransform` implements it by propagating the mask
through its members, evaluating each member only at points still valid, and treating a member
without the method as valid wherever its `transform_point` succeeds on the remaining points.
Direct `transform_point` calls still raise. This resolves open decision 4 of the
[relativity notes](relativity_notes.md) in favour of "exceptions by default, validity only when
explicitly requested", with `resample` as the requester.

**Composition.** Affine and field members compose through the existing `CompositeTransform`;
`compose` still collapses only affine runs. A registration tool's output is usually such a chain,
and each member gets explicit frame endpoints, including the intermediate frame between them.

**Persistence.** Schema 1 stores parameters inline as JSON, which is unsuitable for arrays, and
`rf.encode()` returns a single DataArray, which cannot carry a companion field. The proposal:

- Standalone `encode`/`decode` stay defined. A field transform's record holds its declaration
  (endpoints, grid, interpolant, dtype, shape, digest) and a reference to where its array lives,
  and decoding it requires a caller-supplied resolver for that reference; without one it raises
  `MissingDecoderError`-style, never guesses.
- A container-level API writes and reads a framed array together with its field arrays: a
  Dataset (netCDF) or a Zarr group. Field variables use reserved, collision-checked names; a
  name already present is refused rather than overwritten. Loading copies field data into the
  transform's immutable storage and verifies the digest.
- Records containing references are incompatible with schema-1 readers, so the schema version
  is bumped.

NGFF 0.6 stores field transforms differently (a reference to a multiscale group) and is handled
by the NGFF adapter, not by this schema.

**Adapters.**

- `xarrayrf.ngff`: import `displacements`, `coordinates` and `bijection` (as a declared inverse),
  including inside `sequence`. NGFF names a multiscale group holding the field, with its own
  coordinate transformations from array index to the field's input space and a vector axis.
  The supported subset is defined against official fixtures, pinned to a specification
  version; unsupported interpolants, axis layouts or levels are refused with a report. Export
  follows once import is tested.
- ITK/ANTs warps stored as NIfTI (5-D, intent vector, displacements in LPS millimetres), read with
  nibabel so no ITK dependency is added. File metadata carries no frame identity, so the reader
  requires explicit frames, consistent with the rule that identity is never inferred.
- ANTs direction, per its transform documentation: the forward output (`0GenericAffine.mat`
  plus `1Warp.nii.gz`) maps points from the fixed frame to the moving frame, which is also the
  transform used to resample the moving image onto the fixed grid; the inverse applies the
  inverted affine and `1InverseWarp.nii.gz` in the opposite order. The importer builds the
  whole chain with explicit fixed, intermediate and moving frames, not isolated fields, so the
  affine's position in each direction is fixed by construction.
- FSL warps (relative or absolute, FSL's scaled-millimetre coordinates) are deferred: their
  conventions differ enough to need their own golden tests.

**Acceptance tests.**

1. A constant displacement field equals the corresponding `AffineTransform` translation on
   random in-domain points; a field sampled from an affine map reproduces it exactly under
   linear interpolation.
2. Points outside the grid raise; `resample` fills them.
3. A declared pair's round-trip residual is reported correctly for a smooth analytic pair.
4. An image and a label map (nearest) warped through a field match SimpleITK's resampling of the
   same field within a stated tolerance. SimpleITK would become a dev-only test dependency; today
   only `tools/register_tour_subject.py` uses it, through `uv run --with`.
5. An ANTs registration fixture, checked in both directions: fixed-frame points mapped by the
   forward chain match `antsApplyTransformsToPoints`, and the moving image resampled onto the
   fixed grid matches `antsApplyTransforms`, with the affine included.
6. NGFF 0.6 fixtures for `displacements`, `coordinates` and `bijection` import and resample.
7. Encoding round trip through a Zarr group and through netCDF.
8. A benchmark: a 256³ image through a 128³ field, compared with SimpleITK's `Resample`.

### Stage 2: angular coordinate systems (outline)

- New `representation` values for longitude/latitude charts (spherical and geodetic) and the
  celestial sphere, with a declared periodic interval for longitude and bounded latitude.
- Label alignment stays literal (0 and 360 never align implicitly). Resampling across the
  longitude seam is explicit and supported, because a global grid is unusable without it;
  evaluation at the poles is refused or reported per point.
- No metric is inferred from angles. Area weights, great-circle distances and conservative
  regridding stay with dedicated tools (xESMF, xarray-regrid, pyresample).
- Frame identity matters here: a model's spherical Earth and the WGS 84 ellipsoid are different
  frames, and treating model latitude as geodetic latitude is a known source of errors of
  tens of kilometres. Declared identity (datum or sphere radius in the frame definition) keeps
  them apart.
- A minimal CF reader belongs in this stage, not stage 4: rectilinear `lat`/`lon` with CF
  units, and the earth figure from `grid_mapping` (`latitude_longitude` with
  `earth_radius`, `semi_major_axis` and related attributes) in the frame definition. Without
  it the GCM acceptance case has no import path.
- Acceptance cases: a rectilinear global model grid with a declared spherical earth read
  through that reader and resampled across the longitude seam, a regional grid crossing the
  antimeridian, and a celestial example checked against Astropy.

### Stage 3: curvilinear coordinate fields (outline)

- A transform from `ArrayCoordinates` to a frame backed by coordinate arrays (the 2-D `lat` and
  `lon` of an ocean or swath grid), reusing stage 1's grid, interpolation and validity
  machinery.
- Forward evaluation is exact at nodes. The inverse (frame point to array position) is a search,
  exposed as the approximate or per-point lookup rather than `inverse()`.
- `resample` currently refuses multidimensional coordinate fields; this stage lifts that for
  declared coordinate-field transforms.
- **Binding redesign is a prerequisite.** The binding owns one index per source axis and
  `rf.frame` refuses coordinates that share a dimension, so owning 2-D `lat`/`lon` needs a new
  binding design, measured with the operation inventory across the xarray lanes. Given the
  size of the GCM audience, whether to start that design earlier (alongside stage 2) is a
  maintainer decision.
- Cubed-sphere tiles are curvilinear per tile; tile connectivity is deferred.

### Stage 4: provider adapters (outline)

- pyproj: a transform between two CRS frames (projected or, after stage 2, geographic), wrapping
  an immutable snapshot of the `Transformer` configuration.
- Astropy: an APE 14 WCS as a transform from array coordinates to a celestial frame.
- CF `grid_mapping` beyond stage 2: projected mappings through pyproj, and rotated-pole grids.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Geographic frames first | Needs angular systems before any value; rioxarray, odc-geo, xproj and rasterix already serve projected rasters, and xarrayrf sits beside them |
| Astropy WCS first | A thin adapter once angular systems exist; a small xarray audience on its own |
| Curvilinear grids before angular systems | Serves Cartesian curvilinear data only; GCM grids are longitude/latitude |
| Field inverse through `inverse()` | `inverse()` is exact by contract; a declared pair is an unverified assertion and must be named differently |
| Pairing ANTs fields without an intermediate frame | Each field sits in a chain with an affine whose order differs by direction; the pair is only well defined between the fixed frame and the intermediate frame |
| Numerical field inversion in stage 1 | Approximate by nature; the approximate-inverse capability is deferred and needs its own failure contract |
| Keep field transforms as DataArrays | Mutable; a binding's transform must be immutable and structurally hashable |
| Inline JSON arrays in the encoding | Unworkable at field sizes; NGFF already references arrays by path |
| Mandatory ITK or pyproj dependency | Imposes engine conventions; adapters and dev-only oracles suffice |

## Deferred Work

- Lazy (dask-backed) fields; stage 1 holds the field in memory (a 256³ three-component field
  is about 200 MB as float32, 400 MB as float64).
- Nearest and cubic field interpolation; FSL warp conventions; field export.
- Unstructured meshes (UGRID, MPAS, ICON) and cubed-sphere connectivity.
- Parametric vertical coordinates (CF `formula_terms`, hybrid sigma-pressure), which make the
  vertical coordinate depend on other variables such as surface pressure.
- Time-dependent transforms.

## Open Decisions

1. Names: `DisplacementFieldTransform`, `PositionFieldTransform`, `declared_inverse()` and
   `transform_point_masked` (proposed above).
2. The container-level persistence API: its name, and whether it lives in `xarrayrf.native`.
3. Whether stage 2 includes seam-aware resampling (proposed: yes, for global grids) or only
   declarations and evaluation, as the relativity notes originally recommended.
4. Whether the stage-3 binding redesign starts alongside stage 2.

Settled in review: float32 and float64 storage with float64 evaluation; a schema version bump
for records with array references.

## Next Steps

1. Review this plan; settle the open decisions.
2. Stage 1, in order: the two transform classes with analytic tests; per-point validity in
   `resample`; persistence by reference; NGFF import; ITK/ANTs NIfTI import with SimpleITK and
   ANTs golden tests; benchmark. Update [the core interface](../../core_interface.md) with each
   public name as it lands.
3. Before stage 3, probe binding ownership of 2-D coordinates with the operation inventory.
