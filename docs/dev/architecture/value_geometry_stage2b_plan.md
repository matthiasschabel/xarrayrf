# Value geometry stage 2b: packed symmetric tensor components

**Status:** Implemented
**Last updated:** 2026-10-12
**Scope:** Core: a `SymmetricPacking` value object with NumPy pack and unpack; xarray binding
layer: packed declarations on `TensorComponentDeclaration` and `TensorComponentIndex`,
`rf.change_component_basis` through unpack, transform, repack, explicit pack and unpack on the
accessor, persistence. Implements the "Symmetric packing" item of
[the value geometry design](value_geometry_design.md) (section 2) and the stage 2b entry in
[the stage 2 plan](value_geometry_stage2_plan.md)'s Deferred Work.

## Context

Symmetric order-2 fields are stored packed far more often than as full matrices: diffusion
tensors (6 components), stress and strain (Voigt or Mandel), covariance. Stage 2 declares only
one component dimension per slot, so a packed array cannot be declared, and transforming its
six components as if they were a vector is silently wrong. The design fixes the rule: unpack,
transform, repack; never transform packed components directly.

Conventions disagree on two independent things, and both are real across domains:

| Convention | Component order (3-D) | Off-diagonal scale |
|---|---|---|
| Voigt, stress-like | `11, 22, 33, 23, 13, 12` | 1 |
| Voigt, engineering strain | `11, 22, 33, 23, 13, 12` | 2 |
| Mandel (Kelvin) | `11, 22, 33, 23, 13, 12` | sqrt 2 |
| Upper triangle, row major (FSL `dtifit`, `numpy.triu_indices`) | `11, 12, 13, 22, 23, 33` | 1 |
| Lower triangle, row major (NIfTI `SYM_MATRIX`) | `11, 21, 22, 31, 32, 33` | 1 |
| Diagonal first (MRtrix) | `11, 22, 33, 12, 13, 23` | 1 |

Software varies within a name, too: Abaqus documents its contracted ("Voigt") stress and strain
in the order `11, 22, 33, 12, 13, 23`, the design's `xx, yy, zz, xy, xz, yz`, with engineering
shear strains. The design named a convention string (`"voigt"`, `"mandel"`) that conflates order
and scale and leaves the order ambiguous. `SymmetricPacking.voigt` means the order in the table
above (2-D: `11, 22, 12`), stated in its docstring; Abaqus and the others use explicit pairs or
the triangular constructors. This plan replaces it
with an explicit value.

## Current Decision

### `SymmetricPacking` (core, NumPy only)

New `src/xarrayrf/_symmetric_packing.py`, core layer (no xarray import; added to the core side
of `tests/test_import_boundary.py`).

```python
SymmetricPacking(index_pairs: Sequence[tuple[int, int]], *, off_diagonal_scale: float = 1.0)
```

- `index_pairs`: the matrix entry each packed component holds, in packed order. Each
  unordered pair `{i, j}` over `0..n-1` appears exactly once, in either orientation; `n`
  (`dimension`) is inferred, and the count must be `n(n+1)/2`. The orientation is kept, since it
  names the label (`(2, 1)` labels `zy`).
- `off_diagonal_scale`: packed off-diagonal = scale x matrix entry; finite and positive.
  Diagonal entries are never scaled.
- Constructors: `voigt(dimension, *, off_diagonal_scale=1.0)` (dimension 2 or 3; standard
  Voigt order), `mandel(dimension)` (Voigt order, scale `sqrt(2)`),
  `upper_triangular(dimension, *, off_diagonal_scale=1.0)` and
  `lower_triangular(dimension, *, off_diagonal_scale=1.0)` (row major, any dimension >= 1).
  Others (MRtrix) via explicit pairs.
- `unpack(components)`: last axis of length `n(n+1)/2` to trailing `(n, n)`, dividing
  off-diagonals by the scale and filling both triangles. Dtype `result_type(values, float64)`;
  masked, bool and object values refused, as in the kernel.
- `pack(matrices)`: trailing `(n, n)` to last axis, reading the upper triangle (with
  orientation from the pairs). Packing an asymmetric matrix would lose data silently, so per
  matrix it refuses when mirrored entries `T[i, j]`, `T[j, i]`:
  - differ in missingness (one NaN, one not): packing would drop a known value or a gap;
  - are infinite and not identical (`inf` against a finite value or `-inf`);
  - are both finite and differ by more than `SYMMETRY_TOLERANCE * max|finite T|` (`1e-6`, the
    `ORTHOGONALITY_TOLERANCE` precedent). The scale uses finite entries only, so a NaN or an
    infinite diagonal cannot disable the check; a matrix with no finite entry needs only the
    first two conditions.
- `labels(axes)`: `axes[i] + axes[j]` per pair; refuses when two labels coincide (multi-letter
  axis names can collide, as `ab + a` and `a + ba` over
  axes `("a", "ab", "ba")`).
- Exact equality and hashing over `(index_pairs, off_diagonal_scale)`.

Name: "symmetric packing" is the descriptive term; "Voigt notation" names one instance, and
"Mandel notation" another. The scale is named for what it multiplies, not "weight" or
"factor".

### New tensor type constants

`SYMMETRIC_SECOND_ORDER_CONTRAVARIANT_TENSOR` (stress, diffusion, covariance) and
`SYMMETRIC_SECOND_ORDER_COVARIANT_TENSOR` (metric, small strain as a bilinear form), both
`index_symmetry="symmetric"`, even, `w = 0`. No packable type exists without them.

### Packed declarations

`TensorComponentDeclaration(tensor_type, component_dims, bases, *, packing=None)`:

- With `packing`: the type must be order 2 with `index_symmetry="symmetric"` (equal variance is
  already enforced by `TensorType`); `component_dims` names exactly one dimension; `bases` is one
  basis (or two equal ones), stored per slot as now; `packing.dimension` must equal the basis
  axis count; `packing.labels(basis.axes)` must be distinct.
- `component_dims` becomes "one dimension per component index, or one packed dimension for
  both indices of a symmetric pair". `labels(position)` takes a position in `component_dims`
  (identical to the slot for unpacked declarations), returning the packed labels when packed.
  This is a semantic change to a public method, recorded in the changelog.
- Equality and hashing include `packing`.
- `TensorComponentIndex.isel` compares against `labels(position)`, so a subset or permutation of
  packed components drops the declaration as for unpacked ones; `rename` and the refusals carry
  over unchanged.

### Accessor

- `rf.declare_tensor_components(..., packing=None)`: passes it through; the packed dimension's
  length must be `n(n+1)/2`.
- `rf.change_component_basis`: a mapping target keyed by the packed dimension sets both slots.
  The two slots share one map by the declaration's equal-bases rule, so the apply function is
  `pack_symmetric_part(transform_tensor_components(unpack(values), J, type))`, where the internal
  repack averages each off-diagonal pair instead of checking it (symmetric by construction, up
  to rounding). Labels become `packing.labels(target.axes)`; the packing is kept.
- `rf.unpack_symmetric_components(component_dims=(row, column))`: replaces the packed dimension
  with two new dimensions at its position, declared unpacked in the same bases and type. The
  names are required and must not already be dimensions or coordinates (including a scalar
  coordinate).
- `rf.pack_symmetric_components(packing, *, component_dim)`: the inverse, from an unpacked
  symmetric declaration; the packed dimension takes the first slot's position, and its name
  must not be an existing dimension or coordinate other than the two it replaces. Refuses an
  asymmetric matrix (computed lazily under Dask, so the refusal surfaces at compute).

Both are explicit conversions of a declared array; neither accepts an undeclared one (declare
first). Both keep coordinates on other dimensions, attrs and the binding.

### Persistence

Per the schema evolution rule (persistence design), the existing `tensor_component_declaration`
kind is unchanged: an unpacked declaration encodes exactly as before. Two new kinds:

| Kind | Fields |
|---|---|
| `symmetric_packing` | `index_pairs` (list of `[i, j]`), `off_diagonal_scale` (JSON number) |
| `packed_tensor_component_declaration` | `tensor_type`, `component_dim`, `basis`, `packing` |

An older release refuses a packed record with `UnknownKindError`, never half-reading it as an
unpacked declaration. `sqrt(2)` round-trips exactly through `repr`-based JSON floats.

### Probe and tests

- Probe: packed rows for `isel_identity`/`isel_subset`/`isel_permutation`, `rename`,
  `add_equal`, `concat_component` and `encode_decode`, on all three lanes.
- Tests (`tests/test_symmetric_packing.py`, additions to `tests/test_tensor_components.py` and
  `tests/test_encoding.py`): every constructor's pairs against the table above; pack/unpack
  round trip per convention including dimension 1 and 2; Mandel preserves the Frobenius norm
  (`|c|_2 == |T|_F`) and Voigt does not; asymmetric `pack` refuses, a rounding-level asymmetry
  passes; the non-finite cases (mismatched NaN, `inf` against finite or `-inf`, NaN and
  infinite diagonals with finite asymmetric off-diagonals, a matrix with no finite entry) in
  `pack` and lazily in the accessor; label collision refuses; packed `change_component_basis`
  equals unpack, transform, repack against independent `einsum` formulas, `J T J^T` for the
  contravariant type and `J^-T T J^-1` for the covariant one, times `|det J|^w` for a
  determinant-exponent type, all under a shear with anisotropic scaling so a wrong variance
  fails; Mandel and Voigt give the same unpacked
  result; mapping targets on the packed dimension; pack/unpack accessor round trip keeping
  attrs, coordinates and binding, and refusals (undeclared, unsymmetric type, existing dims,
  asymmetric values); every new error path; encode/decode round trip for each convention and
  the `UnknownKindError` of an unknown kind; Dask laziness for packed changes.

## Alternatives Considered

- **A `packing_convention` string (`"voigt"`, `"mandel"`), as the design sketched.** Rejected:
  it fixes one order and one scale per name, while the table shows independent order and scale
  choices in common use (Voigt strain's factor 2 is the classic trap), and software disagrees
  on the order even under the one name. An explicit value is exact and persists without a
  vocabulary.
- **A dimension-free named order rule (packing valid for any `n`).** Rejected: Voigt order is
  defined for 2 and 3 only, and named rules invite the same disputes; explicit pairs cost one
  validation against the basis.
- **Adding a `packing` field to the `tensor_component_declaration` kind.** Rejected by rule 2 of
  the schema evolution rule (a changed kind bumps the version); a new kind leaves every existing
  record and reader unchanged.
- **A separate `PackedTensorComponentDeclaration` class.** Rejected: every consumer
  (`isel`, `change_basis`, stale-basis checks, the resampling guard) would branch on two types
  where one optional field suffices; the split exists only in the encoding.
- **Symmetrising silently in public `pack`.** Rejected: hard failure over permissive coercion;
  the internal repack symmetrises only output it produced from a symmetric input.
- **Antisymmetric packing** (`n(n-1)/2` components). Deferred: its sign and label conventions
  differ again, and its common use (two-forms in 3-D) is already served by
  `TWO_FORM_DUAL_PSEUDOVECTOR`.

## Deferred Work

- Antisymmetric packing.
- Packing with mixed or per-slot bases (excluded by the equal-bases rule for symmetric types).
- Accepting undeclared arrays in `pack_symmetric_components` (declare first, for now).

## Review

Codex (gpt-6-astra, reasoning effort high), first pass: B1 (non-finite entries could hide or
silently resolve asymmetry in `pack`), N1 (Voigt order is not universal; Abaqus differs), N2
(the covariant oracle is `J^-T T J^-1`), N3 (new dimension names colliding with retained
coordinates). All accepted and folded in above.

Implementation review, same reviewer: B1 (packing could not reuse a replaced dimension name;
`apply_ufunc` needed `exclude_dims`) and N1 (the off-diagonal mean overflowed for entries near
the float64 maximum); both fixed with regression tests, and the second pass recommended Accept.

## Outcome

`src/xarrayrf/_symmetric_packing.py`, packed declarations through `_tensor_component_declaration.py`,
`_tensor_components.py` and the accessor, kinds in `_encoding.py`; tests in
`tests/test_symmetric_packing.py`, `tests/test_tensor_components.py` and `tests/test_encoding.py`.
Suites: stock 2244 passed (47 xfailed), patched 2291 passed. The probe's nine packed rows are all
`ok`; totals 48 ok / 14 limitations (stock), 50 / 12 (patched), 47 / 15 (upstream), no holes.

## Next Steps

1. Stage 3, resampling declared components (design section 3), which must unpack, resample
   or reorient, and repack the same way.
