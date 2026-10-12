# Value geometry stage 1: quantity type, transform law, NumPy kernel

**Status:** Active (plan reviewed; not started)
**Last updated:** 2026-10-11
**Scope:** Geometry core only: a new `_quantity` module, a batched form of the equilibrated
inverse, tests, the core interface section on quantity rules, the changelog. Implements stage 1
of [the value geometry design](value_geometry_design.md) (sections 1 and 7).

## Context

The design fixes one law,

    T' = sign(det J)^[parity odd] * |det J|^w * (J on each up slot)(J^-T on each down slot) T

with `J[a, b] = d target_a / d source_b`, and stages its delivery. Stage 1 is the law alone: a
value object declaring a quantity's type and a NumPy function applying the law to arrays. It
changes no binding, `resample`, persistence or format adapter, and must stay importable and
runnable without xarray.

## Current Decision

### Module and layer

New `src/xarrayrf/_quantity.py`, geometry-core layer (NumPy only). `tests/test_import_boundary.py`
already treats every module outside its integration list as core, so the import rule applies
automatically. Add a `transform_components` call to its `CORE_WITHOUT_XARRAY` subprocess so the
guarantee covers execution, not only imports.

### `QuantityType`

Frozen, slotted dataclass with exact structural equality and hashing, like the other value
objects:

| Field | Type | Validation |
|---|---|---|
| `slots` | `tuple[Literal["up", "down"], ...]` | any sequence of those strings, normalised to a tuple; empty means scalar |
| `parity` | `Literal["even", "odd"]` | default `"even"` |
| `det_power` | `float` | finite real, not `bool`; default `0.0`; stored as `float` so `-1` and `-1.0` are equal and hash equal |
| `symmetry` | `Literal["symmetric", "antisymmetric"] \| None` | default `None`; only with exactly two slots of equal variance, else `ValueError` naming the shear counterexample |

Properties: `rank` (`len(slots)`). No `basis` field: bases belong to the stage-2 component
index, never to the type.

Shortcuts, as module constants of plain `QuantityType` values (not persisted categories):

| Constant | slots | parity | `det_power` |
|---|---|---|---|
| `SCALAR` | `()` | even | 0 |
| `PSEUDOSCALAR` | `()` | odd | 0 |
| `VECTOR` | `("up",)` | even | 0 |
| `COVARIANT_VECTOR` | `("down",)` | even | 0 |
| `SECOND_RANK_TENSOR` | `("up", "up")` | even | 0 |
| `CROSS_PRODUCT_VECTOR` | `("down",)` | odd | 1 |
| `TWO_FORM_DUAL` | `("up",)` | odd | -1 |
| `PSEUDOVECTOR` | `("up",)` | odd | 0, orthogonal maps only |

A scalar density is spelled `QuantityType((), det_power=-1)`; no constant, since the exponent
depends on what is counted per volume. `PSEUDOVECTOR` needs a refusal the others do not, so
`QuantityType` carries one more field, `orthogonal_only: bool = False`, set only by this constant
(and by any caller declaring the same restriction). It participates in equality.

### `transform_components`

    transform_components(values, linear_map, quantity_type, *, component_axes=None)
        -> NDArray

- `values`: array-like of real integer, floating or complex dtype. A masked array is refused
  before conversion (`TypeError`), as `real_float_array` does, because `np.asarray` drops the
  mask silently; boolean, object and other dtypes raise `TypeError`. Non-finite values are
  allowed and propagate (NaN marks missing samples in stage 3). Converted with `np.asarray`, so a
  Dask array is materialised; per-block use belongs to stage 2.
- `linear_map`: converted with `real_float_array` (float64, finite, real; `TypeError` or
  `ValueError` as there), shape `(*batch, n, n)`, square only. Let `S` be the shape of `values`
  with the component axes removed, in their original order. `batch` must broadcast **to** `S`
  under NumPy's right-aligned rule without enlarging it (`np.broadcast_shapes(batch, S) == S`);
  otherwise `ValueError`. So `(3, 3)` applies one map everywhere and `(4, 3, 3)` against values
  `(4, 3)` applies one map per point, while values `(1, 3)` or `(3,)` against maps `(4, 3, 3)`
  are refused: a caller wanting expansion broadcasts the values first. A rectangular map is
  refused even for an up-only type: an embedding has no inverse or determinant, and no stage-1
  consumer needs it.
- `component_axes`: one integer per slot (`TypeError` for non-integers, including `bool`;
  negative allowed; `ValueError` if out of range or repeated), naming which axes of `values`
  hold the components, in slot order; each must have length `n`. Defaults to the trailing `rank` axes
  in slot order, matching the existing calling convention ("a trailing component axis in
  source-axis order"). Required to be empty for rank 0; the scalar law needs only `det J`.
- Returns a new array with exactly the shape and axis order of `values`; dtype is
  `np.result_type(values_array.dtype, np.float64)` (float32 and integers become float64, complex
  stays complex at least complex128).

Algorithm, all batched, no Python loop over points:

1. Validate the type and map (finite, real, square, `n` consistent with every component axis).
2. Move component axes to the end. For each up slot apply `J`, for each down slot `J^-T`, by
   `matmul` along that axis with the map broadcast over the other component axes.
3. If parity is odd or `det_power != 0`, compute `sign, logabsdet = np.linalg.slogdet(J)`
   batched. The scalar factor is `exp(w * logabsdet)`, times `sign` **only for odd parity** (an
   even density is never negated by a reflection). A direct `det` is not used: `1e-120 * I_3` and
   `1e120 * I_3` pass the conditioning test with condition number 1, yet `det` (`1e-360`,
   `1e360`) gives 0 and inf, while `w = 1/3` gives the representable factors `1e-120` and
   `1e120`. If the factor itself is
   not finite or is zero for a nonzero `logabsdet * w` (over- or underflow of the factor), raise
   `ValueError` naming the batch index. Overflow in the subsequent products of finite factors
   and values follows NumPy (inf with a warning), as for any array arithmetic.
4. Move axes back.

Refusal of singular maps: when the type has a down slot, odd parity or `det_power != 0`, the map
must pass the equilibrated conditioning test already used by `AffineTransform.inverse`
(`INVERSE_CONDITION_LIMIT`, unit-independent). `equilibrated_inverse` in `_affine.py` is 2-D
only; generalise it to `(..., n, n)` with per-matrix row and column equilibration, keeping the
2-D behaviour and messages, and report the worst offending index for a batch. An up-only even
type with `w = 0` accepts a singular map (projection is meaningful for vectors).

`orthogonal_only` (`bool` only, `TypeError` otherwise): refuses unless `max |J^T J - I| <= ORTHOGONALITY_TOLERANCE` per matrix, with
`ORTHOGONALITY_TOLERANCE: Final = 1e-6` matching `affine_class`'s default constrained-class
tolerance. The message names `CROSS_PRODUCT_VECTOR` and `TWO_FORM_DUAL`. Units are invisible to
the kernel, so "orthogonal" is about the numbers; a caller mixing units is responsible, as with
the rest of the kernel.

Distinct bases per slot (stage 2, e.g. an up slot in a frame and a down slot in array
coordinates) need no per-slot map argument: the caller first converts each slot to a common
basis with a rank-one call on that axis (`QuantityType(("up",))` or `(("down",))` with
`component_axes=(axis,)`), then applies the common map. Stage 2 documents this route.

Symmetry: the kernel does not check that symmetric input is symmetric (that is a carrier
invariant from stage 2) and does not re-symmetrise output.

### `polar_rotation`

    polar_rotation(linear_map) -> NDArray

The orthogonal factor `U` of the polar decomposition `J = U P`, batched over `(..., n, n)`,
computed from the SVD (`U = W V^T` for `J = W S V^T`). Refuses singular maps (not unique) with the
same conditioning test. For an improper `J` it returns the improper orthogonal factor; it is
documented as "the orthogonal factor", the name kept for continuity with the specified rule.
Tested on an improper map. It is the `"rotation"` reorientation of stage 3, offered now because it is a pure linear-map
helper and is needed for the oracle.

### Public API

Export from `xarrayrf`: `QuantityType`, the eight constants, `transform_components`,
`polar_rotation`, `ORTHOGONALITY_TOLERANCE`. A NumPy-only user (meshes, point clouds, tables)
needs the kernel directly; that is the core's stated audience. The specified-but-unimplemented
`rule(transform, values, *, at)` callables in `core_interface.md` are replaced, not implemented:
evaluating `transform.jacobian(at)` is the caller's one line, and choosing between `J` and
`polar_rotation(J)` is exactly the policy decision the design moved out of the type. `point`
was never a value law and stays `transform_point`.

No encoding: persisting `QuantityType` is the stage-2 component-index kind. `QuantityType` does
not implement `SupportsEncoding` in stage 1.

### Tests (`tests/test_quantity.py`)

Oracles, `assert_allclose` with `rtol=1e-12, atol=1e-12` (a few float64 operations on O(1)
numbers):

1. Shear `J = [[1,1,0],[0,1,0],[0,0,1]]`: `COVARIANT_VECTOR (1,0,0) -> (1,-1,0)` (matches ITK);
   `VECTOR (0,1,0) -> (1,1,0)`; `SECOND_RANK_TENSOR diag(3,2,1) -> [[5,2,0],[2,2,0],[0,0,1]]`
   (deliberately not ITK's `diag(3,2,1)`); `polar_rotation` of the same shear applied as a
   second-rank tensor to `diag(2,3,1)` gives upper triangle `[2.2, 0.4, 0, 2.8, 0, 1]`
   (prior-art oracle 4; these values are exact: `R = [[2,1,0],[-1,2,0],[0,0,√5]]/√5`).
2. Rotation: `SECOND_RANK_TENSOR` under a rotation equals `R T R^T`. Mixed tensor
   `QuantityType(("up", "down"))` under the shear equals `J T J^-1` (ITK's symmetric-tensor
   formula, reachable only by declaring mixed variance). Rank 3 `("up", "down", "down")` on a
   seeded random map equals an explicit `einsum("ai,bj,ck,ibc->ajk", J, Jinv, Jinv, T)`
   (written per index, independent of the kernel's matmul route).
3. Reflection `diag(-1,1,1)`: `VECTOR (1,1,1) -> (-1,1,1)`; `PSEUDOVECTOR (1,1,1) -> (1,-1,-1)`;
   `PSEUDOSCALAR 2 -> -2`.
4. A seeded random map with `det J != ±1`, and the scaled reflection `diag(-2, 1, 1)`, so the
   positive and negative determinant powers are distinguished: `cross(J u, J v)` equals
   `CROSS_PRODUCT_VECTOR` of `cross(u, v)`; the axial dual of `J^-T F J^-1` for antisymmetric
   `F` equals `TWO_FORM_DUAL` of the dual of `F`; the two results differ; `PSEUDOVECTOR` refuses
   both maps with the message naming both types.
5. Scalar density `w = -1` scales by `1 / |det J|`; `PSEUDOSCALAR` gives `sign(det J)` for a
   non-orthogonal improper map.
6. Lorentz boost, `boost_matrix(0.6)` from `tests/test_relativity.py` (c = 1): with
   `F^{0i} = E_i`, `F^{ij} = eps_ijk B_k`, `E = (1, 2, 3)`, `B = (4, 5, 6)`, the
   `("up", "up")` antisymmetric type gives `E' = (1, -2, 7.5)`, `B' = (4, 8.5, 6)`, which match
   the textbook `E'_y = gamma (E_y - beta B_z)` and its companions.
7. Pointwise maps: a `(4, 3, 3)` stack against `(4, 3)` values equals four separate calls;
   broadcasting a single map over `(5, 4, 3)`.
8. `component_axes`: a tensor stored `(3, 7, 3)` with `component_axes=(0, 2)` equals the
   transposed trailing-axis call; output axis order preserved.
9. Complex values: real map applied to complex vectors equals separate real and imaginary calls.
10. Rank 0 with `w = 0`, even: unchanged values, singular map accepted.
11. Determinant range: `1e-120 * I_3` and `1e120 * I_3` with `w = 1/3` give factors
    `1e-120` and `1e120` (`rtol=1e-12, atol=0`, so zero cannot pass); a factor that
    overflows (`1e120 * I_3`, `w = 1`, factor `1e360`) is refused; an
    even density under `diag(-1, 1, 1)` is not negated.
12. Broadcasting: one map over `(5, 4, 3)` values; a `(4, 3, 3)` stack over `(4, 3)`; refusal of
    expansion (`(1, 3)` and `(3,)` values against `(4, 3, 3)` maps); refusal of incompatible
    batches; a scalar batch `(3, 3)`; non-trailing and negative `component_axes` with a batched
    map; an empty batch `(0, 3)` returns `(0, 3)`.
13. Batched equilibrated inverse: equals the per-matrix 2-D result for a stack, keeps the 2-D
    messages, handles an empty batch, and names the offending index.

Refusals, one test per raised error, asserting type and message fragment: masked values;
boolean and object values; non-integer and `bool` `component_axes`; non-`bool`
`orthogonal_only`; invalid slot string;
non-sequence slots; bad parity; non-finite or `bool` `det_power`; symmetry on rank != 2;
symmetry on mixed variance; non-square map; mismatched `n`; wrong number of `component_axes`;
duplicate or out-of-range axes; non-finite map; singular map where required (single and batched,
naming the index); `orthogonal_only` with a shear; `polar_rotation` of a singular map.

Import boundary: the xarray-blocked subprocess calls `transform_components` and
`polar_rotation`.

Equality and hashing: `QuantityType(("up",)) == VECTOR`; `det_power=-1` vs `-1.0` hash equal;
`orthogonal_only` distinguishes `PSEUDOVECTOR` from `QuantityType(("up",), parity="odd")`.

### Documentation

- `docs/core_interface.md` "Quantity rules": replace the specified callables with the law, the
  type, the kernel signature and the two-pseudovector warning, with a short executed example.
- `CHANGELOG.md` "Added".
- Docstrings with a worked example each (reflection of a vector and a pseudovector).
- Value geometry design: status to "stage 1 in progress", then "stage 1 implemented". Amended
  now (with this plan): the component index declaration (section 2) and its persistence kind
  (section 5) carry `orthogonal_only`, and it takes part in index equality, so a persisted
  `PSEUDOVECTOR` does not decode as an unrestricted odd vector.
- `core_model_design.md` quantity table: point to the constants.

## Alternatives Considered

- **Old rule callables `rule(transform, values, *, at)`.** Rejected: they bind the type to a
  transform and hide the choice of map (`J` versus its polar factor), which the design moved to
  the caller. Wrappers can be added later if a consumer wants them.
- **Classmethod constructors (`QuantityType.vector()`).** Rejected for constants: the shortcuts
  have no parameters, and constants read like `CARTESIAN`.
- **A parity-only pseudovector valid for all maps.** Rejected by the design: the two physical
  origins differ under shear.
- **Validating symmetry of input in the kernel.** Deferred to the carrier, which owns the
  declaration; the kernel stays a law.
- **Accepting rectangular maps for up-only types.** Deferred until a consumer (slice embedding)
  needs it; the refusal is easy to lift.

## Deferred Work

- Voigt/Mandel packing helpers: stage 2, where the packed component coordinate lives.
- Dask-aware application (`map_blocks`): stage 2 with the binding.
- Persistence of `QuantityType`: stage 2 schema kind.

## Settled in review

Codex (gpt-6-astra, high), two passes: two blocking findings (broadcasting that could enlarge
the output; direct determinants that are 0 or inf for well-conditioned maps) and three
non-blocking ones, then two oracle corrections, all accepted; no open disagreements.
`orthogonal_only` stays on the type (a kernel argument is easier to
lose) and persists in stage 2; `polar_rotation` keeps its name, documented as the orthogonal
factor including reflections; the private inverse helper is generalised rather than duplicated.

## Next Steps

1. Implement `_quantity.py` and the batched inverse with tests, narrowest first
   (`pytest tests/test_quantity.py tests/test_affine.py tests/test_import_boundary.py`), then the
   full suite, ruff and mypy.
2. Documentation as above; mark stage 1 implemented in the design.
