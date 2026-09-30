# Native `.rf` binding design

**Status:** Implemented
**Last updated:** 2026-09-28
**Scope:** The runtime binding behind the `DataArray.rf` accessor (`src/xarrayrf/_binding.py`,
`src/xarrayrf/native.py`): its carrier, ownership rules, Dataset rule, native-operation outcomes,
operand checks and the local xarray hooks it relies on. Per-operation results are in the
[operation inventory](binding_operation_inventory.md).

## Context

A framed array must behave as an ordinary xarray `DataArray` whose reference-frame information
survives the operations that do not change its sampling, while operations that do change it
either derive a correct new binding or refuse. The release invariant (see
[design.md](../../design.md)) is that no reachable path yields a *valid-looking wrong* binding.
Whole-binding loss is acceptable only where the result has no geometry in the source frame
(reductions over a geometry dimension); loss on a geometry-keeping operation (selection,
arithmetic, alignment, transpose, rename) is a release-blocking hole.

`DataArray` has no private state. Data, coordinates and indexes are each publicly replaceable,
and no stock xarray hook sees every replacement. So on stock xarray no carrier inside a native
`DataArray` meets a strict "never half-complete" requirement. The design therefore chose a
carrier that makes the common paths correct, treats raw setters as documented exits, and closes
the remaining paths with small, opt-in `Index`-API hooks in a locally patched xarray.

## Current Decision

### The binding is a private joint index

`BindingIndex` (private, `xr.Index` subclass) jointly owns every coordinate named by the
transform's source axes and holds the transform and the geometry dimensions itself:

- 1-D source coordinates are *axes*, each delegating label work to a `PandasIndex`;
  0-D source coordinates are *fixed terms* (for example the retained scalar of a selected plane).
- There is no separate declaration or label coordinate. Removing the index removes the binding
  completely, so no guardless declaration can exist.
- `rf.frame(transform, dims=...)` is the only constructor (`set_xindex` raises). It validates by
  building a `Geometry` first, replaces existing indexes on the owned coordinates, and never
  copies or computes pixel data (dask stays lazy).
- The index class and runtime layout are not public API, and the on-disk encoding differs from
  the runtime carrier, so a later carrier can replace it without a breaking change.

The accessor is registered by importing `xarrayrf.native` (not `xarrayrf`), on DataArray only:
`is_framed`, `reference_frame`, `coordinate_transform`, `geometry_dims`, `geometry`,
`frame`, `unframe`, `encode`, `decode`, `resample_to`, `assume_frame`. Adapters build bound
arrays through `frame_array` in `xarrayrf.native`.

### Invariants enforced in code

- `rf.frame` refuses an already framed array, and an array whose attrs still carry the reserved
  `xarrayrf_binding` key (the encoded declaration would otherwise be decoded later as current).
- `rf.frame` refuses source coordinates of 2 or more dimensions, and several 1-D source
  coordinates varying along one dimension: per-axis indexers would each write that dimension,
  so a selection could satisfy one label and not the other.
- Every accessor read verifies the index still owns every coordinate it claims. A *split*
  binding (for example `expand_dims` of a retained scalar, or a Dataset reduction on unpatched
  xarray) raises "neither framed nor unframed" from every member, including `is_framed`, except
  `unframe()`, which clears it. Returning `is_framed == False` was rejected: the array would
  look safely unframed while a `BindingIndex` still sits on some coordinates.
- `unframe()` drops the index, restores default `PandasIndex`es on 1-D dimension coordinates,
  and strips a stale `xarrayrf_binding` attr. A raw `drop_indexes` also unframes but leaves
  unindexed coordinates; it is not the supported route.
- Payload replacement (`.data =`, `copy(data=)`) keeps the binding, since the
  coordinates it describes are unchanged. Deliberate whole-index removal and conversion to raw
  values are the documented exits. Third-party code that rebuilds arrays from values and a
  coordinate dict produces unframed arrays.
- Partial removal raises through xarray's existing protection of multi-coordinate indexes
  (`drop_vars("x")`, `assign_coords(x=...)`).
- Index equality is the value objects' `==` on transform, dims and fixed terms; there is no
  numerical tolerance. Unit checks at bind time and in alignment share
  `Geometry`'s `check_coordinate_unit`, so a non-string `units` attr and a unit on an axis
  declaring none are refused the same way everywhere.

### Index behaviour

| Method | Behaviour |
|---|---|
| `sel` | Delegates per axis to `PandasIndex`; a fixed term cannot be selected by label. |
| `isel` | Slices and integer arrays rebuild the axis; a scalar turns the axis into a fixed term and drops its geometry dimension; an indexer introducing a new dimension raises `ValueError` naming `rf.unframe()`. |
| `equals`, `join`, `reindex_like` | Require the same transform, dims and fixed terms; otherwise `ValueError` that says whether the frames differ (resample with a transform, or `rf.assume_frame`) or only the grids (use `rf.resample_to`). |
| `rename` | Source axis names follow; the affine transform is rebuilt with renamed `ArrayCoordinates` (non-affine transforms refuse a source-axis rename). |
| `roll` | Rolls owned labels with their geometry axes. |
| `concat` | Raises `ValueError` (concatenation of framed arrays is unsupported in v1). |
| `swap_dims`, `join_overlapping`, `check_unindexed_coord_conflicts`, `check_override`, `check_stack`, `check_pad`, `check_coarsen` | Implementations of opt-in hooks that exist only in the patched xarray lane (below). |

### Test lanes and xarray patches

- **stock**: released xarray (2026.7.0 at the time of measurement). Cases needing an xarray fix
  are strict xfails naming it.
- **upstream-main**: unmodified upstream `main` via `PYTHONPATH`.
- **patched**: the published local series (currently series 4, tag `xarrayrf-patches-4`), pinned
  through the `patched` dependency group. Manifest and upstream status:
  [xarray-upstream notes](../xarray-upstream/).

A strict xfail is a milestone, never a release pass. Release needs a released xarray containing
the fix, a validated patch, or a recorded contract decision. Each xarray change is either a
generic bug fix or an opt-in hook with a no-op default, so xarray behaviour is unchanged for
indexes that do not implement it; no domain rule lives in xarray.

| Patch | Change | Upstream |
|---|---|---|
| 1 | Broadcast tracks every dimension of a multidimensional index | merged (#11615) |
| 2 | `isel`/`sel(drop=True)` refuses dropping only some coordinates of one index | PR #11617 |
| 3, 4 | Opt-in `join_overlapping`, `check_unindexed_coord_conflicts`, `check_override`; joined index propagated to every aligned object; custom indexes kept through broadcast | local; 4 must be split before submission |
| 5 | `swap_dims` decides per index and asks an opt-in `Index.swap_dims` | local |
| 6 | Dataset reductions drop a multi-coordinate index whole | PR #11616 |
| 7, 7b | Dataset `update` no longer silently replaces the Dataset's index: 7 raises on an index-type mismatch; 7b asks the index (`join_overlapping`) and keeps it for equal labels | PR #11621; 7b local |
| 8 | Zero-length indexed `roll` no longer divides by zero | merged (#11613) |
| 9 | Coarsen keeps unaffected indexes; opt-in `Index.check_coarsen` | local |
| 10 | Opt-in `Index.check_stack`, `Index.check_pad` | local; split before submission |

### Coordinate ownership: why a joint index

Before adoption, bare, scalar-guard and joint-index carriers were measured on released xarray
and upstream `main` with development-only prototypes, since removed:

| Operation | Bare / scalar guard | Joint index |
|---|---|---|
| Label selection (exact, nearest, slice) | Ordinary | Delegated, passes |
| Crop, stride, gather | Coordinates kept | Coordinates and index kept |
| Scalar selection | Scalar kept | Fixed term kept and compared by the index |
| Different selected planes combined | Conflicting scalar silently dropped | `ValueError` |
| Rename of an input coordinate | Declaration keeps the stale name | Name rewritten |
| Same-dimension Dataset sibling | Inherits the declaration | Same (resolved by the shared-grid rule) |

A joint index was the only carrier that can see selection and renames. Its remaining failures
(mixed alignment, `drop=True`, override, sibling association) became the stage 3b work.

A `sys.setprofile` trace of the call chains showed the following. Binary
arithmetic, NumPy ufuncs and `where` converge on alignment then `merge_collected`, where
operand identity is already lost; `align(join="override")` goes through
`Aligner.override_indexes` without any merge; array-plus-scalar bypasses both alignment and
merge. So no single merge-time callback can check operands: checks are needed in alignment and
override separately, and scalar operands have nothing to conflict with.

### Operand-conflict checks

An xarray-only measurement (`tools/upstream_reproducers/issue_11607_operand_check.py`) used a ~60-line scalar custom index whose `equals`,
`join`, `reindex_like` and `concat` raise on a value mismatch. It ran 135 cells per lane
(11 pair operations x 6 cases x 2 orders, plus 3 scalar operations); released xarray and
upstream `main` gave identical results.

| Case | Outcome across `+`, `np.add`, `apply_ufunc`, `concat`, `merge`, `where`, `align` |
|---|---|
| Both indexed, different values | Refused (by the index or by xarray's exact-alignment check), except `align(join="override")`, which silently copies the first operand's index |
| Indexed vs plain same-name coordinate, different values | Silent everywhere: the indexed value wins in either order; `align(inner/exact)` rewrites the plain operand |
| Plain vs plain, different values | `merge` raises `MergeError`; others drop or concatenate the tag |
| Scalar operand | Index kept, never consulted (correct: nothing to conflict with) |

The indexed-vs-plain asymmetry (an error between two plain coordinates becomes a silent pick
once one side is indexed) is a generic xarray defect, reported upstream as
[pydata/xarray#11607](https://github.com/pydata/xarray/issues/11607) (related to #2996). A local
prototype comparing values in `Aligner._get_indexes_and_vars` and `merge_collected` failed 2 of
about 19,700 xarray tests when restricted to same-dimension variables (one lazy store read,
one error precedence) and 7 without the restriction. A default-on fix waits for maintainer
feedback. `join="override"` replacing indexes is its documented purpose, so it is not an
upstream defect; the binding refuses it through `check_override` instead.

xarrayrf does not wait for #11607. On the patched lane, `join_overlapping` joins plain indexes
on source axes while keeping the binding (and checks their units), and
`check_unindexed_coord_conflicts` refuses a discarded plain coordinate that contradicts a bound
axis or fixed term, in both operand orders.

### Dataset: shared-grid rule

A binding belongs to its geometry dimensions, not to one variable. In a Dataset every data
variable spanning all geometry dimensions is framed (extra dimensions are nonspatial);
variables lacking a geometry dimension, scalars and coordinates such as `ds.x` are unframed;
distinct grids need distinct dimension names, each with its own binding; a spatial variable on
the same dimensions with a conflicting frame is refused. `rf.frame` stays DataArray-only;
Datasets are assembled from framed arrays.

Rationale: xarray's shared labels already assert that `sibling[y, x]` and `owner[y, x]` are the
same sample, and `framed + plain` is already framed. Per-variable ownership would have needed a
new xarray extraction hook plus ownership tracking through rename, merge, `map`,
`to_dataarray` and persistence, and would strip geometry that is physically correct. Metadata
that is not frame-compatible lives beside the framed array in an application object or in
nonspatial variables.

Measured on the patched lane, extraction, `to_dataarray`, merge, renames, selection,
arithmetic, `where`, transpose, nonspatial reductions, `concat` along a nonspatial dimension,
`copy` and pickle stay framed. Two xarray causes needed fixes: `Dataset.reduce`, `quantile` and
`_integrate_one` filtered indexes per coordinate name and split the index (patch 6 drops it
whole, as the DataArray path already did); `Dataset.update` let the incoming `PandasIndex`es
replace the Dataset's index for the same coordinates (patches 7 and 7b). On stock and
upstream, mixed Dataset construction still fails before #11532 or splits the index.

### `swap_dims`

Unpatched `Dataset.swap_dims` keeps each coordinate's index by name, so a multi-coordinate
index is split: the binding stays on `y` with stale dimensions while `x` becomes plain.
The same loop causes generic defects (#11099; a promoted coordinate keeping an index on the old
dimension). Patch 5 decides per index: an index intersecting the swap is asked through
`Index.swap_dims(dims_dict) -> Self | None`; `None` (the default) drops it whole, a returned
index is kept for all its coordinates with renamed dimensions. `BindingIndex.swap_dims` returns
`self.rename({}, dims_dict)`, so `f.swap_dims({"x": "column"})` stays framed with
`geometry_dims == ("y", "column")` and `x` bound along `column`. On stock the split is now
detected and refused on inspection; `raw.swap_dims(...).rf.frame(...)` reaches the same state.

### Coarsen, stack, pad, roll

- **Coarsen**: coarsening only nongeometry dimensions, including `construct`, keeps the exact
  binding (patch 9 carries unaffected indexes through DataArray and Dataset results). Coarsening
  a geometry dimension raises `ValueError` from `check_coarsen`: a mean coordinate alone does
  not define sample or cell support.
- **Stack**: geometry stacking raises from `check_stack`. Two geometry axes on one dimension
  violate the one-coordinate-per-dimension rule; a geometry axis stacked with a nongeometry axis
  is representable but its duplicate-label alignment and selection contract is unproved.
  Nongeometry stack and unstack preserve the binding. The hook also fires for internal stacking
  (for example `groupby`), which now receives the refusal instead of losing the binding.
- **Pad**: geometry padding raises from `check_pad`, because most modes declare no positions
  for new samples and edge, reflect and wrap repeat coordinates. Nongeometry padding preserves
  the binding.
- **Roll**: `BindingIndex.roll` rolls labels with data; the empty-axis crash was an xarray bug
  (patch 8, merged).

### Persistence

`rf.encode()` returns an unframed, pixel-sharing copy with the binding as canonical schema-1
JSON under the reserved `xarrayrf_binding` attr; `rf.decode()` rebuilds it through `rf.frame`,
so validation re-runs against the current coordinates. A raw read is unframed and the encoded
attr is inert until decoded; `decode` raises on a malformed payload. netCDF and Zarr round
trips are in `tests/test_native_encoding.py`. Schema 1 is provisional. There is no Dataset
encoding.

## Alternatives Considered

- **Attrs carrier:** carries metadata well where coordinates
  carry geometry (crops, strides, gathers, coarsening and netCDF stay correct), but enforces
  nothing. Cross-frame arithmetic silently unframes or keeps the first operand's attrs
  depending on path (`np.add`, `xr.where`, `concat`); integer selection and renames of a
  geometry dimension break the declaration. Not a carrier.
- **Declaration coordinate plus scalar guard index:**
  `drop_indexes`, `reset_index`, `set_xindex`, save/load, vectorized selection and `coarsen`
  leave the label without its guard, and a same-dimension Dataset sibling inherits it. The
  label outliving the guard is exactly the half-state to avoid.
- **Duck-array carrier** (binding travels with the data, as pint-xarray does for units): fails
  as a standalone carrier. Payload replacement (`.data =`, `copy(data=)`) bypasses it and can
  transplant another array's binding, and coordinate-only operations (`assign_coords`,
  `drop_vars`, coordinate merging, scalar selection with or without `drop=True`) never reach
  the data, so fixed terms and conflicting plain coordinates are invisible. xarray-internal
  `np.asarray` also cannot be told from deliberate extraction. Running it beside a coordinate
  carrier would add a way for the two to disagree.
- **`DataArray` subclass**: functions such as `xr.where`, `apply_ufunc`, `concat` and `merge`
  build base `DataArray`s, and subclass state must still live in coordinates, attrs or data. It
  adds a type, not an enforcement point.
- **Wrapper container**: full encapsulation, but not a `DataArray`; every xarray function,
  plotting and third-party library needs unwrapping, and delegated methods silently return bare
  arrays unless each is re-wrapped.
- **Per-variable binding with validation hooks at every operand, payload and reconstruction
  boundary in patched xarray**: the only option that meets the strict requirement for a native
  `DataArray`; deferred as a cross-extension proposal.
- **Patch-first, stock xarray refusing framed native operations**: correct under the strict
  contract but a maximal downgrade on stock xarray, committed to patches before the hole list
  existed.
- **Refusing `drop_indexes` on framed arrays**: needs a veto hook, covers one removal route of
  several, and blocks deliberate unframing.
- **Import-time monkey patches; accessor-only arithmetic**: forbidden by project rules, and a
  product change, respectively.
- **Matching-key shims or relabelling bare operands**: hide the mixed-index failure rather than
  resolve it.
- **Per-variable Dataset ownership; refusing Datasets with framed arrays; a Dataset accessor
  now**: see the shared-grid rationale; refusal would block multi-map workflows (parameter maps
  with uncertainties and masks).
- **`swap_dims` alternatives**: drop-only generic fix (whole-binding loss on a geometry-keeping
  operation violates the release invariant); refusing through the hook (allowed, but
  preservation is equally cheap); reusing `Index.rename` (not opt-in: `PandasIndex.rename`
  would keep the demoted coordinate indexed); an accessor-only check (xarray keeps carrying the
  corrupt index). Open PR #11100 alone leaves the split and promoted-index defects.
- **Reattaching after geometry coarsen or pad; refusing every framed coarsen**: reattachment
  assigns unproved sample semantics; blanket refusal rejects valid nongeometry coarsening.
- **A blanket `to_pandas_index()` on the joint index**: would give several coordinates the same
  pandas index and corrupt index introspection; point queries read coordinate variables instead.

## Deferred Work

- A general xarray proposal for a limitation shared across extensions (metadata carried by an
  index or coordinate silently lost, overridden or misattributed by operations that never consult
  its owner). Candidate sources to verify: rioxarray, xproj `CRSIndex`, pint-xarray, cf-xarray,
  rasterix, MetPy, xvec, xdggs. Upstream precedent: CRS demand (#2288) was met by custom indexes,
  and per-variable coordinates (#9152) were closed in favour of cf-xarray, so a new core
  "binding" concept is unlikely to land.
- A default-on fix for #11607: lazy comparison without store reads, error precedence, and
  `merge(compat="override")`.
- Geometry coarsening with declared output cells, point-set bindings (several source coordinates
  along one dimension), mixed geometry/nongeometry stack propagation, geometry-aware padding with
  caller-supplied coordinates, and 2-D or higher source coordinates. Each needs a consumer.
  Treating no-op geometry coarsen and zero-width pad as identities is a smaller follow-up.
- Dataset `rf` accessor and Dataset persistence; consulting `should_add_coord_to_array` in
  Dataset reductions (#11215 follow-up).
- Framed `concat`, refused in v1.

## Next Steps

1. Keep stock, upstream and patched claims distinct; re-run `make probe` and update the
   [inventory](binding_operation_inventory.md) when the patch series or the index changes.
2. Split patches 4, 9 and 10 as listed in the
   [patch manifest](../xarray-upstream/xarray_patches.md) before any upstream submission; follow
   maintainer feedback on #11607 and the open PRs.
3. Continue the release-invariant audit: the 93-case probe certifies its measured paths only.
