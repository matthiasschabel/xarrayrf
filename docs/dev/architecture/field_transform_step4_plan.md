# Field transforms, step 4: ITK and ANTs importers

**Status:** Implemented (2026-10-13; plan and implementation reviewed by Codex gpt-6-sol)
**Last updated:** 2026-10-13
**Scope:** `xarrayrf.itk` (ITK transform files and displacement-field images) and `xarrayrf.ants`
(an ANTs registration's affine, warp and inverse warp as one chain with a declared inverse
pair). The fourth of the five steps of stage 1a in the
[field transform design](field_transform_design.md) (step table in the
[step 1 plan](field_transform_step1_plan.md)), under the
[transform adapter](../adapters/transform_adapters_design.md) contract.

## Context

The design reads ITK and ANTs files through `nitransforms.io` (`nitransforms>=25.1`). Reading its
`io/itk.py` (25.1.0) for this plan found:

- `.tfm` text parameters are parsed as float32 (`np.genfromtxt(..., dtype="f4")`), and every
  centre of rotation is stored as float32 (`("offset", "f4", 3)`), including from double `.mat`
  and `.h5` files.
- The transform type of a text file is not read: any twelve parameters are taken as a matrix and
  translation, so an `AffineDTI3DTransform` (angles, shears, scales, translation) would be
  misread silently. Euler, versor and similarity types are refused only by reshape errors.
- Everything is 3-D; 2-D ITK transforms are unsupported.
- What it does is small: `scipy.io.loadmat` for `.mat`, `h5py` for `.h5`, a sign flip of x and
  y between LPS and RAS, and a five-dimensional NIfTI vector image.

**Decision change:** read the formats directly with `scipy.io`, `h5py` and `nibabel`, in float64,
with the transform type checked, and keep nitransforms out of the dependencies. The design's
reason for adopting it ("reimplementing buys only maintenance") does not hold against
precision loss and silent misreads; the readers are a few hundred lines with an independent
oracle (SimpleITK) in the fixture generator.

**Measured with SimpleITK 2.x and antspyx in Pirana's environment:**

- An ANTs SyN registration's forward transforms `[1Warp.nii.gz, 0GenericAffine.mat]` map a
  fixed point `p` to `Affine(Warp(p))`: `ants.apply_transforms` on coordinate ramps agrees with
  SimpleITK's `aff.TransformPoint(warp.TransformPoint(p))` to float32 precision, and not with
  the other order. SimpleITK's `CompositeTransform([warp, aff])` applies the last-added first.
- ANTs writes `AffineTransform_float_3_3` with a centre (`fixed`) in `.mat` (MATLAB v4), and the
  warp as a five-dimensional NIfTI vector image whose vectors are LPS while its header is RAS.

## Current Decision

### Packages and extras

- `xarrayrf.itk` and `xarrayrf.ants`, each with an extra:
  `itk = ["nibabel>=5.4", "scipy>=1.18", "h5py>=3.11"]` and `ants = ["xarrayrf[itk]"]`.
  `xarrayrf.ants` imports only `xarrayrf.itk` and the core. h5py is imported lazily, only for
  `.h5` files.
- Both work in ITK's physical space, `space` (default the LPS patient system in mm for 3-D,
  required for 2-D), joined to the caller's frames by `coordinate_system_change` exactly as in
  `xarrayrf.elastix` (step 3). The frame joining (fresh or given intermediate frames, the
  boundary coordinate changes, their report entries, the same-frame refusal) moves from
  `xarrayrf.elastix` into a private core module, `_transform_import.py`, used by all three
  importers; it takes `space` explicitly, and each importer supplies the LPS default from
  `xarrayrf.anatomy`, since the core may not import it.

### `xarrayrf.itk`

`read_transform(path, *, fixed, moving, space=None, intermediate_frames=None) -> ItkImport`
(`transform`, `report`), mapping fixed points to moving points as ITK's transform does. The
format follows the suffix:

| Suffix | Content |
|---|---|
| `.tfm`, `.txt` | Insight Transform File V1.0: one transform, or a `CompositeTransform` header followed by its members |
| `.mat` | MATLAB v4: one `AffineTransform_{double,float}_D_D` (or `MatrixOffsetTransformBase`) with `fixed` |
| `.h5` | ITK HDF5: `TransformGroup/0` a single transform or a `CompositeTransform`, members in `1, 2, ...` |
| `.nii`, `.nii.gz` | A displacement field: five-dimensional, `(x, y, z, 1, D)` or `(x, y, 1, 1, 2)`, vectors in LPS |

Members of a composite are listed in the order they were added; ITK applies the last-added first,
so the chain is the reversed list. A text file with several transforms and no composite header is
refused (ITK reads it as a list whose meaning the caller must state).

Linear types, double or float, D = 2 or 3, read in float64 and checked against SimpleITK:

| Type | Parameters | Fixed parameters |
|---|---|---|
| `AffineTransform`, `MatrixOffsetTransformBase` | `A` (D², row-major), `t` | centre `c` |
| `TranslationTransform` | `t` | none |
| `Euler2DTransform`, `Rigid2DTransform` | `θ, t` | `c` |
| `Euler3DTransform` | `θx, θy, θz, t` | `c`, optionally a fourth `ComputeZYX` flag |
| `VersorRigid3DTransform` | versor `v` (3), `t` | `c` |
| `Similarity3DTransform` | `v`, `t`, scale `s` | `c` |

All are `y = A(x − c) + c + t`; the versor's rotation is that of the unit quaternion with vector
part `v`. `IdentityTransform` maps to the identity affine. Any other type, including
`BSplineTransform` (deferred by open decision 2), is refused with `NotImplementedError` naming it.

Displacement fields become `DisplacementFieldTransform(..., outside="identity")` (ITK's exterior;
support `[-0.5, n − 0.5)` as measured in step 1) on a grid with `ArrayCoordinates` `i, j, k`,
`sample_offset` 0.5 and unit coordinates:

- In `.h5`: fixed parameters `size, origin, spacing, direction` (row-major), parameters
  interleaved with the first index fastest.
- In NIfTI: a coded sform or qform is required (neither coded is refused). When both are coded
  they must agree within 1e-6 of the voxel size, else the import is refused: unlike an image,
  whose misplacement shows, a warp placed by the other form displaces every point silently, and
  ITK and nibabel prefer different forms. ANTs writes both, equal. The spatial unit must be
  millimetres; an unknown unit is read as millimetres, as ITK does, and reported
  (`nifti-units`); metres or micrometres are refused, since ITK would rescale them. The RAS
  affine is converted to LPS by negating its first two rows, and the vectors are kept as LPS. For
  a 2-D field the affine's z column and row must not mix with x and y (else refused), and the
  grid is its upper-left 2×2 block and first two translation entries. A field whose `intent`
  is not `vector` is reported (`nifti-intent`), not refused, as nibabel and ITK both read it.

### `xarrayrf.ants`

`transform_from_registration(*, fixed, moving, affine=None, warp=None, inverse_warp=None,
intermediate=None, space=None) -> AntsImport` (`transform`, `report`):

- `affine` (`0GenericAffine.mat`) maps `intermediate` to `moving`; `warp` (`1Warp.nii.gz`) maps
  `fixed` to `intermediate`; the forward chain is `CompositeTransform(warp, affine)`. Without
  `warp`, the affine maps fixed to moving; without `affine`, the warp does. At least one is
  required.
- `inverse_warp` (`1InverseWarp.nii.gz`) requires `warp`; the two become a
  `DeclaredInversePair`, whose forward direction stands in the chain, so
  `transform.declared_inverse()` is ANTs' inverse (`affine⁻¹`, then the inverse warp) and
  `round_trip_residual_report` measures it.
- `intermediate` defaults to a fresh local frame in `space`.

### Fixtures

`tools/build_itk_transform_fixture.py` (dry run by default, `--apply` writes) runs in an
environment with SimpleITK and antspyx (Pirana's). It writes `tests/fixtures/itk/`, each file
with SimpleITK point oracles:

| Reader | Files |
|---|---|
| `.tfm` | affine 2-D and 3-D, translation, Euler 2-D, Euler 3-D with and without `ComputeZYX`, versor rigid, similarity; a composite of an Euler and an affine (order) |
| `.mat` | affine 2-D and 3-D (double, SimpleITK); affine 3-D (float, ANTs) |
| `.h5` | a single affine; a single Euler 3-D; a composite of an affine and a displacement field (order) |
| `.nii.gz` | fields 3-D (oblique) and 2-D (rotated); ANTs warp and inverse warp |

and a small ANTs SyN registration on 16³ synthetic images (affine, warp, inverse warp). A JSON file records query
points and SimpleITK's double-precision `TransformPoint` results for each, and for the ANTs
forward and inverse chains. ANTs registration is not bit-reproducible across machines; the
fixture is generated once and its provenance recorded.

### Tests (`tests/test_itk.py`, `tests/test_ants.py`)

Each file agrees with SimpleITK to 1e-9 mm (float files: 1e-6); RAS caller frames; composite
order; 2-D; refusals (unknown type, B-spline, list without composite header, malformed text,
mismatched parameter counts, non-finite values, sform and qform disagreement, wrong field shape,
`inverse_warp` without `warp`, missing files); the ANTs chain forward against SimpleITK,
`declared_inverse()` against SimpleITK's inverse chain, the round-trip residual of the pair
small; import-boundary rules for both packages.

### Documentation

`docs/core_interface.md` Adapters gains ITK and ANTs paragraphs; README row; changelogs; the
design's Importers section and Alternatives record the nitransforms decision change.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| nitransforms `io` as the reader (the design's choice) | float32 parameters and centres, unchecked transform types, 3-D only (see Context) |
| nitransforms for NIfTI fields only | It adds scipy, h5py and nitransforms to save a sign flip and a reshape |
| SimpleITK or ITK as a runtime dependency | Heavy; the design keeps engines as oracles only |
| One `xarrayrf.itk` holding ANTs too | Settled with the maintainer: one package per engine |

## Refinement Record

Plan review (Codex gpt-6-sol, high, 2026-10-13): Revise.

| Finding | Disposition |
|---|---|
| B1: NIfTI units and uncoded xforms | Accepted: coded xform required; mm, or unknown read as mm and reported; others refused |
| B2: fixture coverage per reader | Accepted: the fixture matrix above, including text and HDF5 composite order |
| N1: the 2-D NIfTI projection | Accepted: stated, with a rotated 2-D fixture |
| N2: refusing sform/qform disagreement | Accepted and justified above; the ANTs fixture has both forms coded |

Implementation review (Codex gpt-6-sol, high, 2026-10-13): prior findings resolved; one new
finding, B3 (the ANTs `affine` slot accepted a displacement field), accepted and fixed with a
refusal test.

## Deferred Work

- ITK B-spline files (open decision 2), elastix-style kernel and other ITK transform types.
- FSL and AFNI warps.
- Writing ITK files.

## Next Steps

1. Done: plan review, implementation, implementation review.
2. Next: step 5 (benchmark and backend choice).
