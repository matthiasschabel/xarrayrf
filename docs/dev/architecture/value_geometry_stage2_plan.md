# Value geometry stage 2: tensor components on xarray variables

**Status:** Implemented
**Last updated:** 2026-10-11
**Scope:** The xarray binding layer: a private `TensorComponentIndex`, declaration and inspection
on the `.rf` accessor, `rf.change_component_basis`, the resampling guard, persistence of the
declaration, an operation probe. Implements stage 2 of
[the value geometry design](value_geometry_design.md) (section 2, section 5, stage-2 acceptance
cases). Builds on stage 1 (`TensorType`, `transform_tensor_components`, implemented).

## Context

Stage 1 gives the law on NumPy arrays. Nothing yet records, on an xarray variable, which
dimensions hold components, in which basis and under which law, so a typed array cannot exist
and no operation can refuse to mistreat one. The design fixes the carrier: a custom xarray index
over the component coordinates, by analogy with the frame binding's `BindingIndex`
([binding design](../binding/binding_design.md)), never attrs. This plan turns that into code.

## Current Decision

### Module and layer

New `src/xarrayrf/_tensor_components.py`, xarray binding layer (added to `INTEGRATION_MODULES`
in `tests/test_import_boundary.py`). Accessor members in `src/xarrayrf/native.py`. The kernel
stays in `_tensor_type.py`; this layer only resolves bases, derives numeric maps and calls it.

### Contract (maintainer decision, 2026-10-11)

The declaration is **advisory with respect to values**. It is enforced on structural operations
between typed operands (selection, alignment, concat, rename, transpose, the patched-lane
preflights) and on every xarrayrf operation that reads it (`change_component_basis`, the
resampling guard, persistence). Native element-wise and value-moving operations treat the
components as plain numbers and can carry the declaration across a change that invalidates it.
This is narrower than the frame binding's invariant (no valid-looking wrong binding) and is a
deliberate choice, not a release hole: an xarray index lives with the coordinates, and xarray
lets the values change independently of them (`.data =`, `copy(data=)`, `apply_ufunc`,
`map_blocks`, NumPy on `.data`), so no index can watch a value operation. Only a carrier that
moves with the data can (Deferred Work).

Measured on stock 2026.7.0 and patched series 4 with a single-coordinate guard index (the first
review's verification probe); the stage-2 probe re-measures every row with the real index:

| Path | Stock | Patched |
|---|---|---|
| Non-linear element-wise operations (`v ** 2`, `abs`, component-by-component `v * w`, `max`) | kept | kept |
| A mask or fill that varies along a component dimension (`where`, `fillna`), and NaN-skipping reductions over other dims when components are missing at different samples | kept | kept |
| `roll` (default `roll_coords=False`) and `shift` along a component dimension: values move, the index is not called | kept | kept |
| Payload replacement (`.data =`, `copy(data=)`) | kept | kept |
| An operand with the component dimension and no coordinate, either order | adopted | adopted |
| An operand with component labels and no index, either order | adopted | refused |
| An operand with a default index on the component coordinate | refused | adopted with the typed operand first; dropped (untyped result) with it second |
| `align(..., join="override")` against a different declaration | adopted | refused |

Each row is listed in the `declare_tensor_components` docstring and the core interface, and
measured by the probe as a documented limitation. Correct by construction: sums and differences
of equal declarations, scaling by a scalar or by a field without component dimensions, `mean`
without missing components, a `where` mask constant along the component dimensions.

### Declaration value object

`TensorComponentDeclaration(tensor_type, component_dims, bases)`, a frozen core value object
(NumPy-only module beside `_tensor_type.py`, exported): the `TensorType`, one dimension name per
slot and one basis per slot (`ReferenceFrame` or `ArrayCoordinates`). It validates what it can
alone: order at least 1, one dim and one basis per slot, distinct dims, and
`index_symmetry` only when both slots have the same basis (the design's equal-variance,
same-basis rule; the kernel cannot check bases). The index holds one; persistence encodes it;
a future data-borne carrier would carry the same object, so stage 2's API and records stay valid
if one is added.

### `TensorComponentIndex` (private)

An `xr.Index` jointly owning one 1-D label coordinate per component index (slot):

- **State:** a `TensorComponentDeclaration` and one `PandasIndex` per component coordinate for
  label work.
- **Bases:** a `ReferenceFrame` (labels equal its coordinate system's axis names, in order), or
  `ArrayCoordinates` equal to the `source` of the same array's frame binding (labels equal the
  source axis names, components per unit of each coordinate value). A component dimension must
  not be a geometry dimension of the binding.
- **Labels are fixed by the basis.** At declaration the labels must equal the basis axes in
  order; if the dimension has no coordinate, the labels are created from the basis.
- **Order 0 has no index.** Densities and pseudoscalars are declared per call (below).
- **Construction** only through `rf.declare_tensor_components`; `from_variables` raises, as
  `BindingIndex` does, so `set_xindex` cannot build a half-checked declaration.

Index behaviour, mirroring `BindingIndex` where the situation is the same:

| Method | Behaviour |
|---|---|
| `sel` | Delegates per component coordinate to its `PandasIndex`. |
| `isel` | Keeps the index only when every component dimension keeps exactly its labels in basis order (identity slices, `[0, 1, 2]`). Anything else (a scalar, a subset, a permutation) returns `None`: xarray then drops the whole index and keeps the labels as plain, unindexed coordinates (verified on 2026.7.0), so a selected component is an untyped value, never a mislabelled tensor. Label `sel` on the remaining coordinate needs `set_xindex` first. |
| `equals` | Same `TensorType`, dims, bases and labels; no tolerance. |
| `join`, `reindex_like` | Require `equals`-level agreement; otherwise `ValueError` naming the difference (law, basis or labels) and the corrective action (`rf.change_component_basis`, or `rf.clear_tensor_components`). Components are never outer-joined. |
| `concat` | Along a component dimension: refuses. Along other dimensions xarray aligns the operands' indexes through `equals`/`join`, so time stacking keeps the type. |
| `rename` | Component dims and coordinate names follow. Labels and bases do not change. |
| `roll` | Called only for `roll_coords=True`; along a component dimension it refuses. The default `roll_coords=False` and `shift` never call the index (contract table). |
| `check_override`, `check_stack`, `check_pad`, `check_coarsen` | Refuse when they touch a component dimension (patched-lane hooks, as for the binding). |
| `join_overlapping`, `check_unindexed_coord_conflicts` | Refuse where xarray calls them: a plain coordinate with component labels carries no transformation law, so adopting this index's type for it would be a guess (unlike the binding's shared-grid case, where shared labels do assert the same sample). Operands those hooks never reach are in the contract table. |

### Accessor

On `DataArray.rf`; none of these needs a frame binding unless a basis is `ArrayCoordinates`:

- `rf.declare_tensor_components(tensor_type, *, component_dims, bases)` → typed array; it builds
  the `TensorComponentDeclaration`. `bases` is one basis per slot, or a single basis for every
  slot. Refuses order 0 (naming the
  per-call route), an already typed array, a stale `xarrayrf_tensor_components` attr, a
  component dimension that is a geometry dimension or absent, wrong labels, a length different
  from the basis dimension, and an `ArrayCoordinates` basis that is not this array's binding
  source. Never computes or copies data.
- `rf.has_tensor_components` and `rf.tensor_component_declaration` (the value object). Like
  `is_framed`, reads verify the index still owns all its coordinates and raise on a split
  index.
- `rf.clear_tensor_components()` → drops the index, restores default indexes on the component
  coordinates, strips a stale encoded attr.

### `rf.change_component_basis(basis, *, transform=None, units=None, tensor_type=None, from_basis=None)`

Passive change of basis at the same sample locations. One target basis for every slot (a mapping
from component dimension to basis is accepted for mixed outputs). Per slot, the map from the
slot's basis `S` to the target `O` is derived:

With the binding `B: A → F` (array coordinates to the bound frame) and a frame map `C: F → O`:

| `S` → `O` | Map |
|---|---|
| unchanged | identity |
| frame `F` → frame `O` | `transform` when supplied (its source must be `F`'s frame and its target `O`'s, else refused); otherwise `coordinate_system_change(F, O)` for the same frame in another coordinate system, which refuses unit rescaling, so a unit change needs `transform` |
| `A` → `F` | `B` |
| `F` → `A` | `B^-1` |
| `A` → `O` | `C @ B` |
| `O` → `A` | `B^-1 @ C^-1` |
| anything else | refused, naming `transform` |

Every resolved map must be square and pass the equilibrated invertibility test, **whatever the
law**: a passive change of basis is a bijection, and a singular map would destroy components
while the rebuilt declaration claimed the new basis. (The kernel itself still accepts a singular
map for contravariant laws, where a projection is meaningful.)

Only constant (affine) maps in stage 2. A non-affine binding or `transform` refuses, naming the
deferral: pointwise maps arrive with field transforms and angular charts (design section 4), and
the kernel already accepts them.

- Slots with different maps are transformed one slot at a time (an order-one call on that
  component axis with that slot's variance and the declaration's `requires_orthogonal_map`). Odd
  reflection parity or a nonzero determinant exponent needs one common map for `det J`, so it
  requires every slot to share the map; mixed maps with those refuse. A symmetric or
  antisymmetric declaration requires the slots to end in one basis; a change that would leave
  them in different bases refuses (a shear on one slot of the identity gives a non-symmetric
  result).
- Labels are rewritten to `O`'s axis names and the index rebuilt with the new bases.
- **Units.** If any slot's source and target axes declare different units, `units=` (the output
  `attrs["units"]`) is required, otherwise the call refuses; with equal units the attribute is
  kept. Unit strings are never parsed.
- **Order 0.** Without an index, `tensor_type=` (order 0) and `from_basis=` are required, and the
  result stays untyped; with an index they are refused.
- **Laziness.** Applied with `xr.apply_ufunc(..., dask="parallelized")` over the component dims as
  core dims; Dask inputs stay lazy. Component dimensions split across chunks raise xarray's own
  core-dimension error, which the docstring names.

### Resampling guard

`resample(source, ...)` and `rf.resample_to` refuse a source carrying a `TensorComponentIndex`:
the stage-3 law (output basis, reorientation) is not defined yet, and resampling components as
independent scalars is the silent wrong answer the design exists to prevent. The message names
`rf.clear_tensor_components()` for a caller who deliberately wants scalar resampling.

### Persistence

- Core `_encoding`: two new kinds, added to the schema evolution table: `tensor_type` (fields
  `index_variance`, `reflection_parity`, `determinant_exponent`, `index_symmetry`,
  `requires_orthogonal_map`) and `tensor_component_declaration` (`tensor_type`,
  `component_dims`, `bases`, each basis a `reference_frame` or `array_coordinates`).
- Native: a separate reserved attr `xarrayrf_tensor_components` holding the canonical
  `encode()` envelope of the declaration. The binding attr and its exact fields are unchanged.
- `rf.encode()` encodes whichever of the binding and the declaration are present and requires at
  least one; `rf.decode()` restores both, binding first, re-validating through the constructors.
  netCDF and Zarr round trips, including string component labels, as in
  `tests/test_native_encoding.py`.

### Operation probe

`tools/tensor_component_probe.py` (`make probe-tensor`), the same lanes and classification as the
binding probe; measured below. Cases: isel/sel identity, scalar, subset, permutation; transpose (slots keep their
names); rename of component dims; arithmetic with an equal declaration, a different law, a
different basis, a different label order, a scalar; untyped operands with a default index,
with unindexed labels and with a bare component dimension, each in both orders; every row of
the contract table as a measured limitation (non-linear ufuncs, component-varying masks and
fills, NaN-skipping reductions, default `roll`, `shift`, payload replacement); `where` with a
constant mask; `align` inner/outer/exact/override;
`concat` along component and other dims; reductions over a component dim (drops) and over others
(keeps); `stack`, `pad`, `coarsen`, `roll` on component dims; Dataset extraction of a variable
sharing component dims and of one with its own; `copy`, `astype`, `chunk`, `compute`, pickle;
`encode`/`decode`; `resample_to` refusal; renaming a binding source axis under an
`ArrayCoordinates` basis (the stale basis must refuse at `change_component_basis`).

### Measured inventory

`tools/tensor_component_probe.py` (`make probe-tensor`), 2026-10-11, on stock xarray 2026.7.0,
upstream `main` at `8de862c29` and patched series 4. Requirements: `keep`, `drop`, `refuse`;
`safe` (raising or dropping are fine, keeping is a documented limitation); `limitation` (a
documented path where the declaration survives a value change). Totals: stock 39 ok,
14 limitations; upstream 38 ok, 15 limitations; patched 41 ok,
12 limitations; no holes on any lane.

| Case | Requirement | Stock | Upstream | Patched |
|---|---|---|---|---|
| isel_identity | keep | kept (ok) | kept (ok) | kept (ok) |
| isel_scalar | drop | dropped (ok) | dropped (ok) | dropped (ok) |
| isel_subset | drop | dropped (ok) | dropped (ok) | dropped (ok) |
| isel_permutation | drop | dropped (ok) | dropped (ok) | dropped (ok) |
| isel_other_dim | keep | kept (ok) | kept (ok) | kept (ok) |
| sel_label | drop | dropped (ok) | dropped (ok) | dropped (ok) |
| sel_all_labels | keep | kept (ok) | kept (ok) | kept (ok) |
| transpose | keep | kept (ok) | kept (ok) | kept (ok) |
| rename_component_dim | keep | kept (ok) | kept (ok) | kept (ok) |
| add_equal | keep | kept (ok) | kept (ok) | kept (ok) |
| add_scalar | keep | kept (ok) | kept (ok) | kept (ok) |
| scale | keep | kept (ok) | kept (ok) | kept (ok) |
| add_different_law | refuse | raises ValueError (ok) | raises ValueError (ok) | raises ValueError (ok) |
| add_different_basis | refuse | raises ValueError (ok) | raises ValueError (ok) | raises ValueError (ok) |
| add_default_index | safe | raises AlignmentError (ok) | kept (limitation) | kept (limitation) |
| add_default_index_reverse | safe | raises AlignmentError (ok) | dropped (ok) | dropped (ok) |
| add_unindexed_labels | safe | kept (limitation) | kept (limitation) | raises ValueError (ok) |
| add_unindexed_labels_reverse | safe | kept (limitation) | kept (limitation) | raises ValueError (ok) |
| add_bare_dimension | safe | kept (limitation) | kept (limitation) | kept (limitation) |
| add_bare_dimension_reverse | safe | kept (limitation) | kept (limitation) | kept (limitation) |
| square | limitation | kept (limitation) | kept (limitation) | kept (limitation) |
| abs | limitation | kept (limitation) | kept (limitation) | kept (limitation) |
| component_product | limitation | kept (limitation) | kept (limitation) | kept (limitation) |
| where_component_mask | limitation | kept (limitation) | kept (limitation) | kept (limitation) |
| where_constant_mask | keep | kept (ok) | kept (ok) | kept (ok) |
| fillna | limitation | kept (limitation) | kept (limitation) | kept (limitation) |
| nanmean_missing | limitation | kept (limitation) | kept (limitation) | kept (limitation) |
| roll_values | limitation | kept (limitation) | kept (limitation) | kept (limitation) |
| roll_coords | refuse | raises ValueError (ok) | raises ValueError (ok) | raises ValueError (ok) |
| shift | limitation | kept (limitation) | kept (limitation) | kept (limitation) |
| payload_replacement | limitation | kept (limitation) | kept (limitation) | kept (limitation) |
| mean_time | keep | kept (ok) | kept (ok) | kept (ok) |
| sum_component | drop | dropped (ok) | dropped (ok) | dropped (ok) |
| align_inner | keep | kept (ok) | kept (ok) | kept (ok) |
| align_outer | keep | kept (ok) | kept (ok) | kept (ok) |
| align_override | safe | kept (limitation) | kept (limitation) | raises ValueError (ok) |
| concat_time | keep | kept (ok) | kept (ok) | kept (ok) |
| concat_component | refuse | raises ValueError (ok) | raises ValueError (ok) | raises ValueError (ok) |
| stack_component | safe | dropped (ok) | dropped (ok) | raises ValueError (ok) |
| pad_component | safe | dropped (ok) | dropped (ok) | raises ValueError (ok) |
| coarsen_component | safe | raises TypeError (ok) | raises TypeError (ok) | raises ValueError (ok) |
| expand_dims | keep | kept (ok) | kept (ok) | kept (ok) |
| dataset_sibling | keep | kept (ok) | kept (ok) | kept (ok) |
| copy | keep | kept (ok) | kept (ok) | kept (ok) |
| astype | keep | kept (ok) | kept (ok) | kept (ok) |
| chunk | keep | kept (ok) | kept (ok) | kept (ok) |
| compute | keep | kept (ok) | kept (ok) | kept (ok) |
| dask_add | keep | kept (ok) | kept (ok) | kept (ok) |
| pickle | keep | kept (ok) | kept (ok) | kept (ok) |
| encode_decode | keep | kept (ok) | kept (ok) | kept (ok) |
| resample_to | refuse | raises ValueError (ok) | raises ValueError (ok) | raises ValueError (ok) |
| stale_array_basis | refuse | raises ValueError (ok) | raises ValueError (ok) | raises ValueError (ok) |
| dask_from_array | keep | kept (ok) | kept (ok) | kept (ok) |

Every limitation is a row of the contract table above; the lanes differ only in which untyped
operands are adopted or refused and in the patched preflights for override, stack and pad.

### Tests

`tests/test_tensor_components.py`, through the public accessor only:

- Declaration happy paths: vector in a frame; covector and order-2 tensor in array coordinates
  on a framed array; labels created when absent; Dask stays lazy.
- Every refusal of `declare_tensor_components`, the inspection members and
  `change_component_basis`, one test per error, asserting type and message.
- `change_component_basis` refusals specific to review: a singular map for `VECTOR`; a
  symmetric declaration left in mixed bases; `requires_orthogonal_map` kept slot by slot; a
  supplied `transform` with the wrong endpoints; a same-frame unit change without `transform`.
- `change_component_basis`: a non-commuting round trip `A → O → A` through `C @ B` and
  `B^-1 @ C^-1`; a same-frame unit rescaling through `transform`; a proper two-axis flip (LPS to RAS through
  `coordinate_system_change`) for a vector and a pseudovector, which agree; an improper one-axis
  flip (RAS to LAS), under which the pseudovector picks up the extra sign; array coordinates with anisotropic spacing to the frame for a vector and a
  covector (spacing multiplies one, divides the other); a mixed array/frame order-2 tensor
  slot by slot; a density through the order-0 route; units required and rewritten; round trip
  back to the original basis within `1e-12`.
- Stage-2 acceptance cases from the design: alignment refuses mixed declarations; transpose keeps
  slots; `resample` refuses typed arrays; encode/decode round trip.
- Dataset sharing; the split-index refusal.

### Documentation

Accessor docstrings with executed examples; `docs/core_interface.md` tensor section gains the
xarray layer; the README or tour gains a short typed-vector example only if it can run in
`tests/test_readme.py`; changelog; design status "stage 2 implemented"; the binding design's
Deferred Work points here for the second index.

## Alternatives Considered

- **Attrs carrier.** Rejected by the design (arithmetic keeps the first operand's attrs;
  `resample` copies attrs).
- **Extending `BindingIndex` to own component coordinates.** Rejected: an unframed table of
  vectors in a frame is legitimate, and tying the type's lifetime to the binding's would drop
  or split it on every binding change.
- **Keeping the index through partial or permuted component selection.** Rejected: a subset is
  not a tensor, and a permutation would make label order and slot order disagree. Dropping the
  type leaves plain labels, which claim nothing.
- **Adopting the type for an untyped operand with matching labels** (the binding's shared-grid
  analogue). Rejected: shared sample labels assert the same place; shared component labels do
  not assert the same law (a covector stored without a declaration has the same labels).
- **A duck-array carrier now** for value semantics, as pint-xarray does for units. Deferred by
  the maintainer (2026-10-11): it is the only route to the strict invariant (it sees every NumPy
  operation, and payload replacement drops it, the safe direction), but it means defining the
  NumPy function surface, tracking component axes through transposes and selections, and
  checking labels against bases at use. Simple path first.
- **Upstream xarray hooks instead.** Partial: opt-in preflights could close `roll`/`shift` and
  adoption by coordinate-free operands, like the patch series' existing `check_*` hooks, but no
  hook can cover payload replacement, which xarray allows by design.
- **Pointwise maps now.** Deferred: no non-affine transform ships yet, so nothing could test them.

## Deferred Work

- Symmetric packing (`packing_convention`, Voigt and Mandel): a stage 2b, before resampling,
  with unpack, transform, repack; the index then owns one packed coordinate for two slots.
- Pointwise maps for `change_component_basis` (non-affine bindings and transforms).
- **A data-borne carrier** (duck array wrapping the components, with the same
  `TensorComponentDeclaration`), giving the strict invariant; its own design. The door is kept
  open by the declaration object, the separate persistence record and accessor members that do
  not expose the index.
- Patched-lane hooks `check_roll`/`check_shift` and a preflight when alignment installs an index
  on an operand without the coordinate; and the patched gap where a default-indexed operand is
  adopted in one order.
- A Dataset accessor.

## Settled in review

Codex (gpt-6-astra, high), two passes. The first raised four blocking findings (an
under-specified safety contract, overstated refusal of typed plus untyped operands, symmetry
surviving mixed slot maps, singular maps accepted as passive changes) and one non-blocking
(resolver precedence and the reverse composite route). The contract question was escalated and
decided by the maintainer (advisory declaration, data-borne carrier deferred); the rest were
accepted. The second pass resolved all five with no new findings.

## Next Steps

Done on 2026-10-11: `src/xarrayrf/_tensor_component_declaration.py`,
`src/xarrayrf/_tensor_components.py`, the accessor members, the resampling guard, the two
persistence kinds, `tests/test_tensor_components.py` and `tools/tensor_component_probe.py`
(measured above). The implementation review (Codex, gpt-6-astra, high, two passes) added three
refusals the plan implied but the first cut missed: a supplied transform that does not fit its
hop, framing a declared component dimension as geometry, and unframing or encoding a stale
array-coordinate basis. Suites pass on stock, upstream and patched xarray.

Next, before stage 3: symmetric packing (stage 2b, Deferred Work), then resampling of
declared components (design section 3).
