# Field-backed transforms

**Status:** Active (design; not started)
**Last updated:** 2026-10-09
**Scope:** Stage 1 of the [nonlinear geometry plan](nonlinear_geometry_plan.md): transforms
between Cartesian frames whose values come from a sampled field or a B-spline control grid
(deformable registration), declared inverse pairs, a Jacobian-determinant diagnostic, importers,
persistence by reference and per-point validity in `resample`. The implementations start from
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
  import. It requires a regular grid with at least four coefficients per axis.
- `DisplacementField` interpolates linearly, supports nonuniform slice positions and a
  single-slice slab domain, and offers `"identity"`, `"raise"` and `"nearest"` exterior
  behaviour. Its domain is closed, unlike ITK's half-open one (below), so Pirana is not an exact
  boundary oracle for ITK.
- Pirana has no analytic Jacobians: `jacobian_measures` uses central differences.

What moves: the two classes (adapted to frame endpoints and the contracts below) and the elastix
parameter-map parser. Pirana's frame classes do not move, and neither does its ANTs reader,
which reads files through antspyx (`ants.read_transform`, `ants.image_read`), a dependency too
heavy for xarrayrf. Running elastix or ANTs stays in Pirana. Pirana's tests depend on its frame
classes, so they are adapted rather than moved.

**[nitransforms](https://github.com/nipy/nitransforms)** (NiPy, MIT, about 5,000 lines; depends on
NumPy ≥ 2, SciPy, nibabel and h5py) reads ITK and ANTs affine `.mat` files and NIfTI
displacement fields (with the LPS sign flip), ITK composite `.h5` containing affine and
displacement-field transforms, FSL, AFNI, FreeSurfer LTA and X5. It does not decode ITK
B-spline transforms: its `.h5` reader raises for any other transform type. Its readers are
reused here for the formats they cover. Its transform classes are not, because they break this
design's contracts (read from 25.1.0, `nonlinear.py`):

- `DenseFieldTransform.map` casts points to float32 and always interpolates cubically with the
  prefilter recomputed per call; ITK and ANTs interpolate displacement fields linearly, so it
  cannot reproduce them.
- Points outside the field are silently returned unchanged, where this design declares the
  exterior behaviour.
- `BSplineFieldTransform.map` loops over points in Python, which is unusable for a volume
  resample.
- Neither has a Jacobian or an inverse; both are mutable (`reference` is a settable attribute)
  and RAS-only.

### ITK boundary conventions

Measured with SimpleITK on a 4-sample axis: a `DisplacementFieldTransform` applies its
displacement on continuous indices `[-0.5, n - 0.5)`, half-open, clamping interpolation to the
edge samples inside that interval, and is the identity outside it (at −0.500001 and at 3.5 the
displacement is zero). ITK's cubic `BSplineTransform` uses the narrower `[1, n - 2]` with an
upper-edge ULP adjustment. Importers reproduce these conventions; they are what "identity
exterior" means for ITK-family producers.

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

- **1a, in memory:** `DisplacementFieldTransform` and `BSplineTransform` on a `Grid`, with
  `"raise"`, `"identity"` and `"nearest"` exterior behaviour; analytic Jacobians and the
  Jacobian-determinant diagnostic; declared inverse pairs and
  `CompositeTransform.declared_inverse()`; the elastix parser and nitransforms-based ITK/ANTs
  readers. This preserves Pirana's capabilities and adds inversion. A field declared `"raise"`
  makes `resample` fail when any queried point is outside it, until 1c; ITK-family warps are
  identity outside and unaffected.
- **1b, persistence:** array references in `encode`/`decode`, the container-level API and the
  schema bump. Separate because a schema bump is the hardest decision here to undo.
- **1c, partial domains:** `PositionFieldTransform`, per-point validity in `resample` and
  `CompositeTransform`, and NGFF import. Their producers (NGFF `coordinates`, SPM `y_`) have no
  consumer yet.

### Representations

`DisplacementFieldTransform`, `BSplineTransform` and, in 1c, `PositionFieldTransform` each
implement `SupportsPoints` and `SupportsJacobian`, with frame endpoints:

- `DisplacementFieldTransform(*, source, target, field, grid, outside)`: `y = x + u(x)`.
  Requires source and target to have equal coordinate systems (axes, units, types), since
  adding a displacement is meaningless otherwise. This is the ITK and ANTs representation and
  NGFF's `displacements`.
- `BSplineTransform(*, source, target, coefficients, grid, order=3)`: `y = x + Σ c_k β(x)`, a
  displacement parameterized by control-point coefficients (ITK `BSplineTransform`, elastix,
  nitransforms `BSplineFieldTransform`). Coefficients are not samples of the displacement, so
  this is not a `DisplacementFieldTransform` with a cubic interpolant. Rasterizing it into a
  dense field is an explicit conversion with a caller-chosen grid.
- `PositionFieldTransform(*, source, target, field, grid, outside)` (1c): `y = f(x)`, the stored
  value is the target position itself. This is NGFF's `coordinates`.

**The sampling grid is an xarrayrf `Grid`** in the source frame, not a new lattice declaration:
`Grid` already holds frozen, possibly nonuniform 1-D coordinates and a transform into its frame,
which covers Pirana's nonuniform slice positions and single-slice slabs. An arbitrary `Grid`
does not guarantee evaluation, so a field's grid must satisfy:

- `grid.frame == source`, exactly;
- a square, invertible affine grid transform, with no retained scalar axes;
- nonempty, strictly monotonic coordinates for dense fields; regular coordinates and at least
  four coefficients per axis for B-splines;
- explicit intervals on a single-slice axis, which defines its slab extent.

The world-to-index locator is prepared once at construction, not per call.

`field` and `coefficients` have shape `grid shape + (n_target_axes,)`, stored as float32 or
float64, keeping the producer's dtype; evaluation is always in float64 (`map_coordinates` is
called with `output=np.float64`, since float32 input otherwise yields float32). They are copied
on construction into immutable storage, following `frozen_float_array` in `_validation.py`,
and exposed only as views that cannot be made writeable back into that storage.

### Representations by producer

Keep the producer's native representation; never fit coefficients to a dense field or rasterize
coefficients on import. Parametric forms are compact, smooth and have analytic Jacobians, but
the most common diffeomorphic tools write dense fields, and both directions where they can.
Physics-informed models (linear elasticity, viscous fluid, hyperelastic energies, biomechanical
FEM) constrain estimation, not representation: their output is one of the forms below. Material
parameters stay with the optimizer; xarrayrf does not evaluate mechanics.

| Representation | Producers | Inverse | Stage |
|---|---|---|---|
| Cubic B-spline free-form deformation (coefficients) | ITK `BSplineTransform`, elastix, NiftyReg `reg_f3d`, FSL FNIRT `--cout` | None in closed form | 1a (elastix; ITK files per open decision 2) |
| Dense displacement field, or a declared pair of them | ANTs SyN (`Warp`/`InverseWarp`), elastically regularized and viscous-fluid registration | Declared pair where the producer writes both | 1a |
| Dense position field pair | SPM DARTEL and Shoot (`y_`/`iy_`), NGFF `coordinates` and `bijection` | Declared pair | 1c |
| Stationary velocity field, dense or B-spline-parameterized; `φ = exp(v)` by scaling and squaring | NiftyReg `-vel`, log-domain Demons, SPM DARTEL flow fields, diffeomorphic VoxelMorph/SynthMorph; log-Euclidean polyaffine models | Exact up to integration error: `exp(-v)` | After 1a, the first invertible parametric family |
| Piecewise-affine simplicial mesh (node positions on a tetrahedral or triangular mesh) | Biomechanical FEM (NiftySim, FEBio, SOFA), hyperelastic registration on meshes, CASTalign `Triangulation` | Exact where every element keeps positive orientation: point location in the deformed mesh | After 1a; shared with the CASTalign adapter |
| Kernel landmark transform (landmarks, kernel weights and an affine part, with the kernel declared) | Thin-plate splines (BigWarp, ITK, histology), elastic body splines (Navier-equation kernels), Gaussian and Wendland kernels | None in closed form | After 1a, if a consumer asks |

Time-varying velocity fields and LDDMM initial momenta (Deformetrica, SPM Shoot) are deferred.
Polynomial and SIP distortion terms arrive through the stage-4 provider adapters (GDAL, Astropy
WCS).

### Evaluation

Three things are specified separately: locating a point on the grid, the transform's support,
and how interpolation treats the edge.

- **Dense fields:** linear interpolation, ITK's default. Support is declared per producer; for
  ITK-family fields it is `[-0.5, n - 0.5)` in continuous index with edge samples clamped inside
  it. An importer refuses an interpolant the transform does not implement rather than
  substituting linear.
- **B-splines:** cardinal cubic B-spline evaluation from coefficients, without prefiltering,
  supported on `[1, n - 2]` as in ITK and Pirana. Two backends are candidates:
  `map_coordinates(coefficients, ..., order=3, prefilter=False)` for values, and
  `scipy.interpolate.NdBSpline` with knots `arange(n + 4) - 2` per axis, which also evaluates
  derivatives but copies coefficients to float64. One backend serving both values and Jacobians
  is preferred; choose by the benchmark (acceptance test 8).
- **Exterior:** declared on the transform: `"raise"` (the default when constructing directly, as
  `transform_point` requires), `"identity"` (ITK's semantics for both dense fields and
  B-splines) or `"nearest"` (the nearest edge displacement, Pirana's existing option). These are
  transform words, not `Grid`'s `outside`; `"extrapolate"` is neither identity nor nearest.
- **Jacobian:** the derivative of the interpolant in index space, chained with the derivative of
  the world-to-index map (local spacing for nonuniform coordinates): `I + ∂u/∂x` for
  displacements and B-splines, with a stated tie rule on cell faces, and the identity in an
  identity exterior.

### Jacobian-determinant diagnostic

A helper reports, for any `SupportsJacobian` transform on caller-chosen points or a `Grid`, the
determinant of the Jacobian, its minimum and the fraction of points where it is ≤ 0 (folding).
Physics-informed models guarantee a positive determinant only up to discretization, and a
declared inverse is implausible where the field folds, so this is the model-independent check
of invertibility. It replaces Pirana's finite-difference `jacobian_measures`.

### Inverse

`inverse()` stays exact, as the core interface requires, and a field transform does not provide
it. A producer-declared pair (an NGFF `bijection`; an ANTs `1Warp` and `1InverseWarp`) is one
immutable pair object that owns both directions. Each direction exposes the other through a
separately named capability, `declared_inverse()`, which returns it without verifying it.
Owning both in one object avoids mutually referencing transforms, which would make equality and
encoding cyclic. A half's equality is its own identity, independent of its partner.

`CompositeTransform.inverse()` keeps its current rule and refuses a chain containing a field.
`CompositeTransform.declared_inverse()` is offered when every member is exactly invertible or
has a declared inverse, and returns the reversed chain. Warping in the opposite direction is the
most common registration task, so it must be one call; the method name carries the "declared,
not verified" status. Affine-only elastix results invert exactly; a chain containing an unpaired
B-spline stage (elastix writes no inverse) has no `declared_inverse()`. A helper reports the
round-trip residual of a pair, or of a chain and its declared inverse, on caller-chosen points.
Numerical inversion is the separate approximate-inverse capability that the core interface
already defers; point validity is never used to signal an inverse's existence (a folded field
maps every point but has no local inverse).

### Equality and hashing

Identity equality in 1a: `Transform` permits it (`_transform.py`), `Grid` documents equality
that may be object identity, and bindings and `CompositeTransform` only delegate to their
members' declared equality (`_binding.py`, `_composite.py`). Independently constructed identical
fields compare unequal, which is permitted. The array digest arrives with 1b, where persistence
verifies loaded data against it.

### Composition

Affine and field members compose through the existing `CompositeTransform`; `compose` still
collapses only affine runs. A registration tool's output is usually such a chain, and each
member gets explicit frame endpoints, including the intermediate frame between them.

### Importers

Under the [transform adapter](../adapters/transform_adapters_design.md) contract: engine
subpackages with their own extras, core-only imports, caller-named endpoint frames and a report
of every normalization. File metadata carries no frame identity, so the caller supplies frames,
consistent with the rule that identity is never inferred.

- **elastix:** Pirana's parameter-map parser moves as code: translation, Euler, affine and
  B-spline transforms, including the component-major coefficient layout. It needs no ITK
  runtime.
- **ITK and ANTs files:** read through `nitransforms.io` (the ITK-family extra depends on
  `nitransforms>=25.1` and `nibabel>=5.4`, which brings h5py and a NumPy 2 floor within that
  extra only): affine `.mat`, NIfTI displacement fields and composite `.h5` with affine and
  displacement-field members. nitransforms works in RAS millimetres; the importer declares that
  coordinate system explicitly. ITK B-spline files are not covered (open decision 2). Pirana's
  evaluation-based handling of ITK linear types (Euler, Similarity, Affine with any center) is
  checked against nitransforms' decoding by fixtures before it is relied on.
- **ANTs direction:** ITK composites apply the last-added stage first, so the forward output
  (`1Warp.nii.gz` plus `0GenericAffine.mat`) applies the warp to the fixed point, then the
  affine, mapping fixed to moving; that is also the transform that resamples the moving image
  onto the fixed grid. The inverse applies the inverted affine, then `1InverseWarp.nii.gz`. The
  declared pair sits between the fixed frame and the intermediate frame before the affine. The
  importer builds the whole chain with explicit fixed, intermediate and moving frames, so the
  affine's position in each direction is fixed by construction.
- **FSL** warps come from the same nitransforms layer, after golden tests for their conventions.
- **NGFF (1c):** `xarrayrf.ngff` imports `displacements`, `coordinates` and `bijection` (as a
  declared pair), including inside `sequence`. NGFF names a multiscale group holding the field,
  with its own coordinate transformations from array index to the field's input space and a
  vector axis. The supported subset is defined against official fixtures pinned to a
  specification version; unsupported interpolants, axis layouts or levels are refused with a
  report. Export follows once import is tested.

### Persistence (1b)

Schema 1 stores parameters inline as JSON, which is unsuitable for arrays, and `rf.encode()`
returns a single DataArray, which cannot carry a companion array.

- Standalone `encode`/`decode` stay defined. A field record holds its declaration (endpoints,
  grid, interpolant or order, exterior, dtype, shape, digest) and a reference to where its array
  lives; decoding requires a caller-supplied resolver for that reference and raises without one,
  never guesses. The resolver is storage-neutral; Dataset and Zarr assembly live in integration
  code, per the [core layering](core_layering_design.md).
- A declared pair is encoded as one record owning both directions, plus which direction was
  encoded, so decoding a half returns that half with its original source and target.
- A container-level API writes and reads a framed array together with its arrays, as a Dataset
  (netCDF) or a Zarr group, and also handles the affine-only case, so users have one save path
  whatever the transform. Array variables use reserved, collision-checked names; a name already
  present is refused. Loading copies data into immutable storage and verifies the digest.
- Records with references are incompatible with schema-1 readers, so the schema version is
  bumped; the new reader continues to accept schema 1.

NGFF 0.6 stores field transforms as references to multiscale groups and is handled by the NGFF
adapter, not by this schema.

### Validity (1c)

Per-point validity lets `resample` fill points outside a `"raise"` field instead of failing. It
cannot be confined to `resample`: `CompositeTransform.transform_point` passes every row to the
next member, and affine members and the source locator reject non-finite values, so an invalid
row from a field fails at the next member before `resample` sees it. The contract:

- An optional masked evaluation returns `(values, valid)` with `valid.shape ==
  points.shape[:-1]`; values are finite wherever valid.
- `CompositeTransform` filters rows between members: each member is evaluated only at rows still
  valid. A member without masked evaluation is called through its ordinary batch
  `transform_point`, and its exceptions propagate; an arbitrary failure is never reinterpreted
  as a domain exclusion.
- `resample` locates only valid rows and fills the rest with `fill_value`.
- Direct `transform_point` calls still raise; NaN is never returned as an ordinary result.

This resolves open decision 4 of the [relativity notes](relativity_notes.md) in favour of
"exceptions by default, validity only when explicitly requested", with `resample` as the
requester. Fields are supported as explicit frame maps between locatable image geometries; a
field embedded in the source geometry itself needs a locator and belongs to stage 3.

### Acceptance tests

Oracle results are precomputed fixtures, generated by a script in `tools/` that records the
generating tool and version, so CI needs no ANTs or ITK binaries; SimpleITK is a dev-only test
dependency for the fixtures it can regenerate.

1. A constant displacement field equals the corresponding `AffineTransform` translation on
   random in-domain points; a field sampled from an affine map reproduces it exactly under
   linear interpolation.
2. Boundary behaviour matches the pinned ITK version exactly at the support edges and at
   adjacent points, for dense fields (`-0.5`, `n - 0.5`) and B-splines (`1`, `n - 2`), including
   a single-slice slab; `"raise"` refuses and `"nearest"` matches Pirana.
3. A declared pair's round-trip residual is reported correctly for a smooth analytic pair, and a
   chain built from an ANTs pair inverts through `declared_inverse()`.
4. An image and a label map (nearest) warped through a field match SimpleITK's resampling of the
   same field within a stated tolerance.
5. An ANTs registration fixture, checked in both directions: fixed-frame points mapped by the
   forward chain match `antsApplyTransformsToPoints`, and the moving image resampled onto the
   fixed grid matches `antsApplyTransforms`, with the affine included.
6. `BSplineTransform` from an elastix parameter map matches elastix's transformix and
   nitransforms' `BSplineFieldTransform.map` on the same coefficients.
7. Analytic Jacobians match finite differences, and the determinant diagnostic flags a folded
   field.
8. A benchmark: a 256³ image through a 128³ field (16.7M points), measuring prepared grid lookup,
   values alone and values plus Jacobians for each B-spline backend, with peak memory, compared
   with SimpleITK's `Resample`.
9. (1b) Encoding round trip through a Zarr group and through netCDF, including each half of a
   declared pair.
10. (1c) A composite mixing valid and invalid rows never passes an invalid row to a later member
    or the locator; NGFF 0.6 fixtures for `displacements`, `coordinates` and `bijection` import
    and resample.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| nitransforms transform classes as the engine | float32 points, fixed cubic interpolation, silent identity outside the domain, per-point Python loop for B-splines, mutable, no Jacobian (see Context) |
| Hand-written NIfTI/ANTs/FSL readers | nitransforms already handles the conventions and is light; reimplementing them buys only maintenance |
| Moving Pirana's ANTs reader | It reads files through antspyx, far too heavy a dependency for xarrayrf |
| A new lattice declaration for the sampling grid | Drops nonuniform slices and slab domains; `Grid` already provides them |
| `Grid`'s `outside` words as the exterior vocabulary | They are lookup policies; ITK's half-open support, edge clamping and identity exterior are not expressible with them |
| A B-spline as `DisplacementFieldTransform(interpolant="cubic")` | Coefficients are not displacement samples; the two differ everywhere except for an interpolating spline |
| Field inverse through `inverse()` | `inverse()` is exact by contract; a declared pair is an unverified assertion and must be named differently |
| Each direction referencing the other as its inverse | Cyclic equality and encoding; one pair object owns both |
| Pairing ANTs fields without an intermediate frame | Each field sits in a chain with an affine whose order differs by direction; the pair is only well defined between the fixed frame and the intermediate frame |
| Numerical field inversion in stage 1 | Approximate by nature; the approximate-inverse capability is deferred and needs its own failure contract |
| Masking inside `resample` only | Invalid rows fail at the next composite member first |
| Structural equality with a digest in 1a | Hashes up to 400 MB per construction with no 1a consumer; the digest serves 1b persistence |
| Keep field transforms as DataArrays | Mutable; a binding's transform must be immutable |
| Inline JSON arrays in the encoding | Unworkable at field sizes; NGFF already references arrays by path |
| Mechanics (elasticity tensors, FEM solves) in xarrayrf | The adapter contract translates results and never runs an optimizer; material parameters are provenance at most |
| Mandatory ITK dependency | Imposes engine conventions; nitransforms and dev-only oracles suffice |

## Deferred Work

- Lazy (dask-backed) fields; stage 1 holds the field in memory (a 256³ three-component field is
  about 200 MB as float32, 400 MB as float64).
- Stationary velocity fields, piecewise-affine simplicial meshes (shared with the CASTalign
  `Triangulation` adapter) and kernel landmark transforms, in the order of the table above.
- Time-varying velocity fields and LDDMM initial momenta.
- Material parameters as provenance metadata on imported transforms, if a biomechanical consumer
  needs them.
- Nearest and cubic dense-field interpolation; field export; AFNI and LTA importers until a
  consumer asks.
- Possible upstream pull requests to nitransforms (float64 points, a vectorized B-spline map, a
  B-spline `.h5` reader), drafted for the maintainer to submit, never posted by an agent.

## Open Decisions

1. Names: `DisplacementFieldTransform`, `PositionFieldTransform`, `BSplineTransform`, the pair
   class, `declared_inverse()`, the masked-evaluation method and the determinant helper.
2. ITK `.h5`/`.tfm` B-spline files: defer until a producer needs them (recommended), or write a
   small decoder in 1a, since nitransforms has none.
3. The ITK-family subpackage layout (one `xarrayrf.itk` for ITK, ANTs and elastix, or one per
   engine) and its extra's name; whether the container-level persistence API lives in
   `xarrayrf.native`.
4. Whether stationary velocity fields or simplicial meshes follow 1a first.

## Refinement Record (2026-10-09)

A Claude review of this design (P, S and C items) was checked by Codex (gpt-6-astra, high). The
three high-confidence factual points were verified independently: ITK's half-open dense support
by SimpleITK, nitransforms' missing B-spline decoding in its source, and unfiltered composite
evaluation in `_composite.py`.

| ID | Finding | Disposition |
|---|---|---|
| P1 | Use `Grid` as the sampling grid | Accepted with an admissibility contract and a prepared locator |
| P2 | Pirana's ANTs reader needs antspyx | Accepted; nitransforms covers affine and displacement files only |
| P3 | Analytic Jacobians are new work | Accepted; `NdBSpline` viable with cardinal knots, backend chosen by benchmark |
| P4 | ITK dense fields are identity outside | Accepted; half-open support and edge clamping specified |
| S1 | Narrow 1a | Revised: `"nearest"` exterior kept; masking, position fields and NGFF moved to 1c |
| S2 | Masking local to `resample` | Rejected by the review; filtering between composite members in 1c |
| S3 | Identity equality in 1a | Accepted |
| S4 | Precomputed fixtures; adapted tests | Accepted, with generator provenance |
| C1 | ANTs order wording | Accepted |
| C2 | Pair equality and encoding | Revised: identity equality; the encoded direction is recorded |
| C3 | Grid frame equals source | Accepted |
| C4 | float64 output | Accepted |
| C5 | No elastix inverse | Revised: only chains with an unpaired B-spline stage |

No disagreements remain open; the 1a scope and open decision 2 are maintainer decisions.

## Next Steps

1. Settle the open decisions with the plan's.
2. 1a, in order: the two transform classes on `Grid`, adapted from Pirana with its tests, plus
   analytic tests and the ITK boundary fixtures; Jacobians and the determinant diagnostic; the
   pair and chain inversion; the elastix parser, then nitransforms readers, with SimpleITK, ANTs,
   elastix and nitransforms fixtures; the benchmark. Update
   [the core interface](../../core_interface.md) with each public name as it lands.
3. 1b once 1a is complete; 1c when a consumer of partial-domain fields appears. Pirana adopts the
   classes only after xarrayrf's release gate (see the [roadmap](../roadmap.md)).
