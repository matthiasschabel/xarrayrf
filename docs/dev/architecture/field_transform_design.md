# Field-backed transforms

**Status:** Active (design; not started)
**Last updated:** 2026-10-09
**Scope:** Stage 1 of the [nonlinear geometry plan](nonlinear_geometry_plan.md): transforms
between Cartesian frames whose values come from a sampled field or a B-spline control grid
(deformable registration), per-point validity in `resample`, declared inverse pairs,
persistence by reference, and importers built on nitransforms. The implementations start from
Pirana's existing deformable transforms, which move here; Pirana adopts them later.

## Context

Only affine transforms exist today. The capability protocols (`SupportsPoints`,
`SupportsJacobian`, `SupportsInverse`) and `CompositeTransform` already admit nonlinear members,
and `resample`'s general path already maps target points through any `SupportsPoints` frame
transform (`_general_map` in `src/xarrayrf/_resampling.py`). Deformable registration needs no
change to the core value objects: source and target frames are Cartesian, and a field is a
transform between them.

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

### Existing implementations

**Pirana** (`pirana.spatial._transforms`) already has `BSplineTransform` and `DisplacementField`,
with elastix and ANTs importers in `pirana.registration`, all tested. They are the starting
point: deformable transforms are general geometry and belong in xarrayrf, not in an
application. What they already settle:

- `BSplineTransform` stores coefficients, evaluates with `map_coordinates(order=3)` and no
  prefilter, and derives its domain as control-grid indices 1 through `shape - 2`, outside which
  it is the identity, matching ITK. elastix's component-major coefficient layout is converted on
  import.
- `DisplacementField` interpolates linearly, with an explicit exterior policy and nonuniform
  slice positions.

What changes in the move: frame endpoints instead of Pirana's `ReferenceFrame` geometry, the
validity and declared-pair contracts below, and structural equality. Pirana's frame classes are
not moved. The parsers that turn elastix parameter maps and ANTs outputs into transforms move
with the classes; running elastix or ANTs stays in Pirana. nitransforms covers the formats Pirana
does not read.

**[nitransforms](https://github.com/nipy/nitransforms)** (NiPy, MIT, about 5,000 lines; depends on
NumPy ≥ 2, SciPy, nibabel and h5py) reads ITK and ANTs NIfTI warps (with the LPS sign flip),
ITK composite `.h5`, FSL, AFNI, FreeSurfer LTA and X5. Its readers are the valuable part and
are reused here. Its transform classes are not, because they break this design's contracts
(read from 25.1.0, `nonlinear.py`):

- `DenseFieldTransform.map` casts points to float32 and always interpolates cubically with the
  prefilter recomputed per call; ITK and ANTs default to linear interpolation of displacement
  fields, so it cannot reproduce them.
- Points outside the field are silently returned unchanged, where this design raises or masks.
- `BSplineFieldTransform.map` loops over points in Python, which is unusable for a volume
  resample.
- Neither has a Jacobian or an inverse; both are mutable (`reference` is a settable attribute)
  and RAS-only.

## Current Decision

### Scope and constraints from later stages

Stage 1 is Cartesian only. The [nonlinear geometry plan](nonlinear_geometry_plan.md) settles
the structural questions for angular charts, metrics and tensors; stage 1 implements none of
them but must not preclude them:

1. Field transforms require Cartesian source and target coordinate systems. "No coordinate
   system restriction" on position fields means only that axes and units may differ.
   Componentwise interpolation is wrong across an angular seam (179° and −179° interpolate to
   0°), so stage 2 reuses the storage and grid evaluation but not the interpolant.
2. The validity mask means "this point maps", nothing more: not that the Jacobian is invertible
   and not that an inverse search would converge.
3. `affine_class` stays a numeric classifier of the matrix; no metric enters stage 1.
4. The persistence record admits typed array resources that are not point transforms (a future
   metric field or frame field), rather than assuming every array belongs to a transform.

### Split

- **1a, in memory:** the three transform classes, declared inverse pairs and chain inversion
  through them, masked evaluation in `resample`, and the nitransforms-based importers. This is
  the complete deformable-registration capability, independent of any consumer.
- **1b, persistence:** array references in `encode`/`decode`, the container-level API and the
  schema bump. Separate because a schema bump is the hardest decision here to undo.

### Representations

Three classes, each implementing `SupportsPoints` and `SupportsJacobian`, with frame endpoints:

- `DisplacementFieldTransform(*, source, target, field, grid)`: `y = x + u(x)`. Requires source
  and target to have equal coordinate systems (axes, units, types), since adding a displacement
  is meaningless otherwise. This is the ITK and ANTs representation and NGFF's `displacements`.
- `PositionFieldTransform(*, source, target, field, grid)`: `y = f(x)`, the stored value is the
  target position itself. This is NGFF's `coordinates`.
- `BSplineTransform(*, source, target, coefficients, grid, order=3)`: `y = x + Σ c_k β(x)`, a
  displacement parameterized by control-point coefficients (ITK `BSplineTransform`, nitransforms
  `BSplineFieldTransform`). Coefficients are not samples of the displacement, so this is not a
  `DisplacementFieldTransform` with a cubic interpolant. Evaluation is
  `scipy.ndimage.map_coordinates(coefficients, ..., order=3, prefilter=False)`, which evaluates a
  spline from its coefficients, vectorized. Rasterizing it into a dense field is an explicit
  conversion with a caller-chosen grid.

`grid` is the affine mapping from sample (or control-point) positions into the source frame (a
`Lattice`-like declaration: origin, spacing, direction). Its extent is the transform's domain;
for a B-spline it is control-grid indices 1 through `shape - 2`, as in Pirana and ITK. `field` and `coefficients` have shape
`grid shape + (n_target_axes,)`, stored as float32 or float64, keeping the producer's dtype;
evaluation is always in float64. They are copied on construction into immutable storage,
following `frozen_float_array` in `_validation.py`, and exposed only as views that cannot be
made writeable back into that storage.

### Representations by producer

Keep the producer's native representation; never fit coefficients to a dense field or rasterize
coefficients on import. Parametric forms are compact, smooth and have analytic Jacobians, but
the most common diffeomorphic tools write dense fields, and both directions where they can.

| Representation | Producers | Inverse | Stage |
|---|---|---|---|
| Cubic B-spline free-form deformation (coefficients) | ITK `BSplineTransform`, elastix, NiftyReg `reg_f3d`, FSL FNIRT `--cout` | None in closed form | 1a |
| Dense displacement or position field pair | ANTs SyN (`Warp`/`InverseWarp`), SPM DARTEL and Shoot (`y_`/`iy_`), NGFF `bijection` | Declared pair | 1a |
| Stationary velocity field, dense or B-spline-parameterized; `φ = exp(v)` by scaling and squaring | NiftyReg `-vel`, log-domain Demons, SPM DARTEL flow fields, diffeomorphic VoxelMorph/SynthMorph | Exact up to integration error: `exp(-v)` | After 1a, the first invertible parametric family |
| Thin-plate spline from landmarks | BigWarp (Fiji), ITK, histology and morphometrics tools | None in closed form | After 1a, if a microscopy consumer asks |

Time-varying velocity fields (LDDMM) are deferred. Polynomial and SIP distortion terms arrive
through the stage-4 provider adapters (GDAL, Astropy WCS).

### Evaluation

Linear interpolation of dense fields, ITK's default. The interpolant is part of the transform's
declaration; NGFF also permits nearest and cubic, and an importer refuses an interpolant the
transform does not implement rather than substituting linear. Exterior behaviour is part of the
declaration: `"raise"` (the default for dense fields, as `transform_point` requires) or
`"identity"` (ITK's B-spline semantics, and Pirana's existing option), so warps reproduce their
producer outside the support. The Jacobian is the derivative of the interpolant
(`I + ∂u/∂x` for displacements and B-splines), with a stated tie rule on cell faces.

### Validity

A transform may implement an optional `transform_point_masked(points) -> (values, valid)`, and
`resample` uses it when present, filling invalid samples with `fill_value`. The contract:

- `valid.shape == points.shape[:-1]`; values are finite wherever valid.
- The mask covers the whole mapping pipeline in `_general_map`: target geometry points, the
  frame transform and the source locator. Invalid rows never reach a later step.
- `CompositeTransform` evaluates each member only at rows still valid. A member without the
  method is called through its ordinary batch `transform_point`, and its exceptions propagate;
  an arbitrary failure is never reinterpreted as a domain exclusion.
- Direct `transform_point` calls still raise.

This resolves open decision 4 of the [relativity notes](relativity_notes.md) in favour of
"exceptions by default, validity only when explicitly requested", with `resample` as the
requester. Fields are supported as explicit frame maps between locatable image geometries; a
field embedded in the source geometry itself needs a locator and belongs to stage 3.

### Inverse

`inverse()` stays exact, as the core interface requires, and a field transform does not provide
it. A producer-declared pair (an NGFF `bijection`; an ANTs `1Warp` and `1InverseWarp`, which are
inverses between the fixed frame and the intermediate frame after the affine) is one immutable
pair object that owns both directions. Each direction exposes the other through a separately
named capability, `declared_inverse()`, which returns it without verifying it. Owning both in
one object avoids mutually referencing transforms, which would make equality and encoding
cyclic.

`CompositeTransform.inverse()` keeps its current rule and refuses a chain containing a field.
`CompositeTransform.declared_inverse()` is offered when every member is exactly invertible or
has a declared inverse, and returns the reversed chain. Warping in the opposite direction is the
most common registration task, so it must be one call; the method name carries the "declared,
not verified" status. A helper reports the round-trip residual of a pair, or of a chain and its
declared inverse, on caller-chosen points. Numerical inversion is the separate
approximate-inverse capability that the core interface already defers; point validity is never
used to signal an inverse's existence (a folded field maps every point but has no local
inverse).

### Equality and hashing

Exact and structural: endpoints, grid, interpolant or order, dtype, shape and array bytes, with
signed zeros normalized as the existing value objects do. A SHA-256 digest computed once at
construction serves as the hash and as a fast inequality check; equal digests are confirmed by
comparing bytes, so equality never rests on the digest alone. Structural hashing is chosen, not
required: `Transform` permits identity equality and optional hashing (`_transform.py`).

### Composition

Affine and field members compose through the existing `CompositeTransform`; `compose` still
collapses only affine runs. A registration tool's output is usually such a chain, and each
member gets explicit frame endpoints, including the intermediate frame between them.

### Importers

Behind an optional extra (name open; for example `warps = ["nitransforms>=25.1",
"nibabel>=5.4"]`, which brings h5py and a NumPy 2 floor within the extra only):

- ITK/ANTs NIfTI warps, ITK composite `.h5`, ITK B-spline transforms and FSL warps are read
  through `nitransforms.io` for arrays, affines and sign conventions, then wrapped in the
  classes above. nitransforms works in RAS millimetres; the importer declares that coordinate
  system explicitly. File metadata carries no frame identity, so the caller supplies frames,
  consistent with the rule that identity is never inferred. FSL support, previously deferred
  for its conventions, comes from the same layer.
- ANTs direction, per its transform documentation: the forward output (`0GenericAffine.mat`
  plus `1Warp.nii.gz`) maps points from the fixed frame to the moving frame, which is also the
  transform used to resample the moving image onto the fixed grid; the inverse applies the
  inverted affine and `1InverseWarp.nii.gz` in the opposite order. The importer builds the whole
  chain with explicit fixed, intermediate and moving frames, not isolated fields, so the
  affine's position in each direction is fixed by construction.
- `xarrayrf.ngff`: import `displacements`, `coordinates` and `bijection` (as a declared pair),
  including inside `sequence`. NGFF names a multiscale group holding the field, with its own
  coordinate transformations from array index to the field's input space and a vector axis.
  The supported subset is defined against official fixtures pinned to a specification version;
  unsupported interpolants, axis layouts or levels are refused with a report. Export follows
  once import is tested.

### Persistence (1b)

Schema 1 stores parameters inline as JSON, which is unsuitable for arrays, and `rf.encode()`
returns a single DataArray, which cannot carry a companion array.

- Standalone `encode`/`decode` stay defined. A field record holds its declaration (endpoints,
  grid, interpolant or order, dtype, shape, digest) and a reference to where its array lives;
  decoding requires a caller-supplied resolver for that reference and raises without one,
  never guesses. The resolver is storage-neutral; Dataset and Zarr assembly live in integration
  code, per the [core layering](core_layering_design.md).
- A declared pair is encoded as one record owning both directions.
- A container-level API writes and reads a framed array together with its arrays, as a Dataset
  (netCDF) or a Zarr group, and also handles the affine-only case, so users have one save path
  whatever the transform. Array variables use reserved, collision-checked names; a name already
  present is refused. Loading copies data into immutable storage and verifies the digest.
- Records with references are incompatible with schema-1 readers, so the schema version is
  bumped; the new reader continues to accept schema 1.

NGFF 0.6 stores field transforms as references to multiscale groups and is handled by the NGFF
adapter, not by this schema.

### Acceptance tests

1. A constant displacement field equals the corresponding `AffineTransform` translation on
   random in-domain points; a field sampled from an affine map reproduces it exactly under
   linear interpolation.
2. Points outside the domain raise; `resample` fills them. A composite mixing valid and invalid
   rows never passes an invalid row to the source locator.
3. A declared pair's round-trip residual is reported correctly for a smooth analytic pair, and a
   chain built from an ANTs pair inverts through `declared_inverse()`.
4. An image and a label map (nearest) warped through a field match SimpleITK's resampling of the
   same field within a stated tolerance. SimpleITK becomes a dev-only test dependency.
5. An ANTs registration fixture, checked in both directions: fixed-frame points mapped by the
   forward chain match `antsApplyTransformsToPoints`, and the moving image resampled onto the
   fixed grid matches `antsApplyTransforms`, with the affine included.
6. `BSplineTransform` matches nitransforms' `BSplineFieldTransform.map` and ITK on the same
   coefficients, which pins the control-grid placement and boundary convention.
7. NGFF 0.6 fixtures for `displacements`, `coordinates` and `bijection` import and resample.
8. (1b) Encoding round trip through a Zarr group and through netCDF, including a declared pair.
9. A benchmark: a 256³ image through a 128³ field, compared with SimpleITK's `Resample`.

nitransforms is an oracle for file conventions and B-spline evaluation only; its dense-field
evaluation is cubic and cannot check the linear interpolant.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| nitransforms transform classes as the engine | float32 points, fixed cubic interpolation, silent identity outside the domain, per-point Python loop for B-splines, mutable, no Jacobian (see Context) |
| Hand-written NIfTI/ANTs/FSL readers | nitransforms already handles the conventions and is light; reimplementing them buys only maintenance |
| A B-spline as `DisplacementFieldTransform(interpolant="cubic")` | Coefficients are not displacement samples; the two differ everywhere except for an interpolating spline |
| Field inverse through `inverse()` | `inverse()` is exact by contract; a declared pair is an unverified assertion and must be named differently |
| Each direction referencing the other as its inverse | Cyclic equality and encoding; one pair object owns both |
| Pairing ANTs fields without an intermediate frame | Each field sits in a chain with an affine whose order differs by direction; the pair is only well defined between the fixed frame and the intermediate frame |
| Numerical field inversion in stage 1 | Approximate by nature; the approximate-inverse capability is deferred and needs its own failure contract |
| Keep field transforms as DataArrays | Mutable; a binding's transform must be immutable |
| Inline JSON arrays in the encoding | Unworkable at field sizes; NGFF already references arrays by path |
| Mandatory ITK dependency | Imposes engine conventions; nitransforms and dev-only oracles suffice |

## Deferred Work

- Lazy (dask-backed) fields; stage 1 holds the field in memory (a 256³ three-component field is
  about 200 MB as float32, 400 MB as float64).
- Nearest and cubic dense-field interpolation; field export; AFNI and LTA importers until a
  consumer asks.
- Possible upstream pull requests to nitransforms (float64 points, a vectorized B-spline map),
  drafted for the maintainer to submit, never posted by an agent.

## Open Decisions

1. Names: `DisplacementFieldTransform`, `PositionFieldTransform`, `BSplineTransform`, the pair
   class, `declared_inverse()` and `transform_point_masked`.
2. The optional extra's name, and whether the container-level persistence API lives in
   `xarrayrf.native`.
3. Whether stationary velocity fields follow 1a directly. They are the only family here whose
   inverse is the same object with a sign flipped.

## Next Steps

1. Settle the open decisions with the plan's.
2. 1a, in order: the transform classes, moved from Pirana with its tests and adapted to frame
   endpoints, plus analytic tests; the pair and chain inversion;
   masked evaluation in `resample`; the elastix and ANTs importers from Pirana, then nitransforms
   importers, with SimpleITK, ANTs and nitransforms golden tests; NGFF import; benchmark. Update [the core interface](../../core_interface.md)
   with each public name as it lands.
3. 1b once 1a is complete. Pirana adopts the classes only after xarrayrf's release gate (see the
   [roadmap](../roadmap.md)).
