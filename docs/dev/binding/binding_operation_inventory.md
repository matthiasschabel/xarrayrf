# Native binding operation inventory

**Status:** Active
**Last updated:** 2026-10-25
**Scope:** Measured outcome of 132 native xarray operations on framed arrays, on stock xarray,
upstream `main` and the patched series; the release gate for the private `BindingIndex`.

## Context

The probe is `tools/binding_operation_probe.py`; `tools/check_binding_probe.py` gates its JSON
in CI (`make probe-pinned` locally). The first 93 cases use one 2-D affine fixture; N01-N35 are
the [audit matrix](native_operation_audit_matrix.md), whose expected outcomes were written and
reviewed before any run; N36-N39 were added after the
[extension survey](../xarray-upstream/extension_hook_survey.md) found the same split in rasterix. Design and rationale: [binding_design.md](binding_design.md).

Lanes at measurement (2026-10-24): stock xarray 2026.7.0 wheel; upstream `main` at `b2acc77a`;
patched series 5 (tag `xarrayrf-patches-5`, `a6576179`); the unpublished series-6 candidate
(`b13b0a6b`). The reported xarray version can be stale
editable-install metadata under `PYTHONPATH`; the probe's `xarray_module` identifies the lane.

Every case states one outcome per output. **preserve**: the binding survives and world points,
evaluated with the *original* transform at the result's source labels, match within `1e-12`
(the synthetic grids are integer-spaced, so this covers affine rounding without accepting a
moved sample), together with structural checks of identity, axis order, fixed terms and index
ownership. **unframe**: no binding remains and the expected labels do. **refuse**: the exact
exception class and a message fragment, with inputs unchanged. Setup, operation, inspection
and compute failures are reported separately. A case the contract does not decide is
`pending_review` and never counts as `ok`; none remain.

## Current Decision

Totals: stock 74/132 `ok` (58 holes), upstream 85/132 (47), series 5 128/132 (N36-N39),
series-6 candidate 132/132. Patched-lane
evidence is a release precondition, not a release pass; stock and upstream holes are the
reason the patched pin exists.

Owners of the stock and upstream holes, all fixed on the series-6 candidate
([manifest](../xarray-upstream/xarray_patches.md)):

| Holes | Owner on series 5 |
|---|---|
| scalar `drop=True` (`isel_drop_true`, `sel_drop_true`, N34) | #11617 |
| `Dataset.update` replacing the binding (`dataset_setitem_labelled`, N07-N09, N30) | #11621 and `join_overlapping` update acceptance |
| mixed framed/plain alignment, operand order, `where`, `ufunc`, partial alignment and merge (N01-N06, N24) | `join_overlapping` joins and validation; stock also lacks merged #11532 |
| `join="override"` | `join_overlapping(how="override")` |
| unindexed coordinate and unit conflicts | `join_overlapping(how=None)` |
| `broadcast`, N33 (excluded dimension) | broadcast common-index fix and label retention |
| `stack`, `pad`, `coarsen`, `interp` | `Index.check_drop` |
| coarsen of nongeometry dimensions | coarsen index-retention fix |
| `swap_dims`, field-backed swap (N32) | per-index `swap_dims` |
| equal-label 1-D operand first: arithmetic, `where`, Dataset, merge (N36-N39; also series 5) | series 6: shadowed subset index dropped on the no-reindex path |
| Dataset geometry reductions (stock only; N14) | merged #11616 |
| empty roll (stock only; N17) | merged #11613 |

Review decisions recorded in the probe: `broadcast_plane_volume` and `groupby` accept xarray's
own refusals (an alignment conflict, a corrupt-index error), since no binding results;
`rolling` must preserve; `interp` must refuse at the operation (maintainer decision
2026-10-24).

| Case | Expected | Stock | Upstream | Series 5 | Series 6 candidate |
|---|---|---|---|---|---|
| isel_scalar | preserve | ok | ok | ok | ok |
| isel_slice | preserve | ok | ok | ok | ok |
| isel_array | preserve | ok | ok | ok | ok |
| isel_vectorized | refuse `ValueError`: vectorized indexing changes geometry dimensions; call rf.unframe() | ok | ok | ok | ok |
| isel_drop_true | refuse `ValueError`: would corrupt the following index | hole (preserved) | hole (preserved) | ok | ok |
| sel_scalar | preserve | ok | ok | ok | ok |
| sel_slice | preserve | ok | ok | ok | ok |
| sel_nearest | preserve | ok | ok | ok | ok |
| sel_array | preserve | ok | ok | ok | ok |
| sel_vectorized | refuse `ValueError`: vectorized indexing changes geometry dimensions; call rf.unframe() | ok | ok | ok | ok |
| sel_drop_true | refuse `ValueError`: would corrupt the following index | hole (preserved) | hole (preserved) | ok | ok |
| head | preserve | ok | ok | ok | ok |
| tail | preserve | ok | ok | ok | ok |
| thin | preserve | ok | ok | ok | ok |
| transpose | preserve | ok | ok | ok | ok |
| rename | preserve | ok | ok | ok | ok |
| swap_dims | preserve | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | ok | ok |
| expand_dims | preserve | ok | ok | ok | ok |
| squeeze | preserve | ok | ok | ok | ok |
| squeeze_geometry | preserve | ok | ok | ok | ok |
| add_framed | preserve | ok | ok | ok | ok |
| add_different_transform | refuse `ValueError`: incompatible framed coordinate mappings | ok | ok | ok | ok |
| add_unframed | preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | ok | ok | ok |
| add_unframed_reverse | preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (AssertionError: expected preserved binding) | ok | ok |
| add_unframed_shifted_labels | preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | ok | ok |
| add_unframed_conflict | refuse `ValueError`: conflicting unindexed coordinate 'y' | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (preserved) | ok | ok |
| add_unframed_conflict_reverse | refuse `ValueError`: conflicting unindexed coordinate 'y' | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (preserved) | ok | ok |
| add_unframed_unit_conflict | refuse `ValueError`: coordinate 'x' declares attrs['units'] = 'cm' | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (preserved) | ok | ok |
| add_unframed_unit_conflict_reverse | refuse `ValueError`: coordinate 'x' declares attrs['units'] = 'cm' | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (unframed) | ok | ok |
| add_unindexed_unit_conflict | refuse `ValueError`: coordinate 'x' declares attrs['units'] = 'cm' | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (preserved) | ok | ok |
| add_unindexed_unit_conflict_reverse | refuse `ValueError`: coordinate 'x' declares attrs['units'] = 'cm' | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (preserved) | ok | ok |
| add_scalar | preserve | ok | ok | ok | ok |
| add_scalar_reverse | preserve | ok | ok | ok | ok |
| ufunc_framed | preserve | ok | ok | ok | ok |
| ufunc_unframed | preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | ok | ok | ok |
| ufunc_unframed_reverse | preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (AssertionError: expected preserved binding) | ok | ok |
| ufunc_scalar | preserve | ok | ok | ok | ok |
| ufunc_scalar_reverse | preserve | ok | ok | ok | ok |
| ufunc_unary | preserve | ok | ok | ok | ok |
| ufunc_unframed_conflict | refuse `ValueError`: conflicting unindexed coordinate 'y' | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (preserved) | ok | ok |
| ufunc_unframed_conflict_reverse | refuse `ValueError`: conflicting unindexed coordinate 'y' | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (preserved) | ok | ok |
| where_method | preserve | ok | ok | ok | ok |
| where_function | preserve | ok | ok | ok | ok |
| where_unframed_x | preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | ok | ok | ok |
| where_unframed_y | preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | ok | ok | ok |
| where_unframed_condition | preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | ok | ok | ok |
| where_framed_x | preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (AssertionError: expected preserved binding) | ok | ok |
| where_framed_y | preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (AssertionError: expected preserved binding) | ok | ok |
| where_conflicting_coordinate | refuse `ValueError`: conflicting unindexed coordinate 'y' | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | ok | ok |
| align_inner | preserve; preserve | ok | ok | ok | ok |
| align_outer | preserve; preserve | ok | ok | ok | ok |
| align_exact | refuse `AlignmentError`: cannot align objects with join='exact' | ok | ok | ok | ok |
| align_override | refuse `ValueError`: join='override' is unsupported for framed coordinates | hole (preserved) | hole (preserved) | ok | ok |
| align_unframed | preserve; preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (AssertionError: expected preserved binding) | ok | ok |
| align_unframed_reverse | preserve; preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (AssertionError: expected preserved binding) | ok | ok |
| align_unframed_shifted_right | preserve; preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | ok | ok |
| align_override_framed_first | refuse `ValueError`: join='override' is unsupported for framed coordinates | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (preserved) | ok | ok |
| align_override_framed_second | refuse `ValueError`: join='override' is unsupported for framed coordinates | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (unframed) | ok | ok |
| broadcast | preserve | hole (ValueError: new dimensions {'y': 3, 'channel': 2} must be a superset of existing dimensions ('y', 'x')) | hole (AssertionError: expected preserved binding) | ok | ok |
| broadcast_plane_volume | refuse `AlignmentError`: cannot align objects on coordinate 'y' because of conflicting indexes | ok | ok | ok | ok |
| sum_geometry | unframe | ok | ok | ok | ok |
| sum_nongeometry | preserve | ok | ok | ok | ok |
| coarsen | refuse `ValueError`: cannot coarsen geometry dimensions; call rf.unframe() | hole (unframed) | hole (unframed) | ok | ok |
| rolling | preserve | ok | ok | ok | ok |
| groupby | refuse `ValueError`: would corrupt the following index | ok | ok | ok | ok |
| resample | unframe | ok | ok | ok | ok |
| stack | refuse `ValueError`: cannot stack geometry dimensions; call rf.unframe() | hole (unframed) | hole (unframed) | ok | ok |
| unstack | preserve | ok | ok | ok | ok |
| interp | refuse `ValueError`: cannot interpolate geometry dimensions; use rf.resample_to or call rf.unframe() | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | ok | ok |
| reindex | refuse `AlignmentError`: cannot align objects on coordinate 'x' because of conflicting indexes | ok | ok | ok | ok |
| reindex_like | preserve | ok | ok | ok | ok |
| shift | preserve | ok | ok | ok | ok |
| roll_values | preserve | ok | ok | ok | ok |
| roll_coords | preserve | ok | ok | ok | ok |
| roll_empty | preserve | hole (ZeroDivisionError: integer modulo by zero) | ok | ok | ok |
| pad | refuse `ValueError`: cannot pad geometry dimensions; call rf.unframe() | hole (unframed) | hole (unframed) | ok | ok |
| concat_geometry | refuse `ValueError`: concatenation of framed arrays is unsupported; call rf.unframe() | ok | ok | ok | ok |
| concat_nongeometry | preserve | ok | ok | ok | ok |
| merge | preserve; preserve | ok | ok | ok | ok |
| to_dataset_extract_owner | preserve | ok | ok | ok | ok |
| dataset_extract_sibling | preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | ok | ok | ok |
| dataset_mean_geometry | unframe | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | ok | ok | ok |
| dataset_quantile_geometry | unframe | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | ok | ok | ok |
| dataset_setitem_labelled | preserve; preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (AssertionError: expected preserved binding) | ok | ok |
| assign_coords | refuse `ValueError`: would corrupt the following index | ok | ok | ok | ok |
| drop_vars | refuse `ValueError`: would corrupt the following index | ok | ok | ok | ok |
| copy | preserve | ok | ok | ok | ok |
| astype | preserve | ok | ok | ok | ok |
| chunk | preserve | ok | ok | ok | ok |
| compute | preserve | ok | ok | ok | ok |
| dask_isel | preserve | ok | ok | ok | ok |
| dask_add | preserve | ok | ok | ok | ok |
| pickle | preserve | ok | ok | ok | ok |
| N01 | preserve | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | ok | ok |
| N02 | preserve | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | ok | ok |
| N03 | preserve | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | ok | ok |
| N04 | refuse `ValueError`: coordinate 'x' differs for join='exact' | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | ok | ok |
| N05 | unframe; preserve | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | ok | ok |
| N06 | refuse `ValueError`: conflicting unindexed coordinate 'x' | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (preserved) | ok | ok |
| N07 | preserve; preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (AssertionError: expected preserved binding) | ok | ok |
| N08 | refuse `ValueError`: conflicting unindexed coordinate 'x' | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (ValueError: the binding no longer owns its source coordinate(s) ['y']; the array is neither framed nor) | ok | ok |
| N09 | refuse `AlignmentError`: cannot update Dataset: coordinate | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | ok | ok |
| N10 | preserve | ok | ok | ok | ok |
| N11 | unframe | ok | ok | ok | ok |
| N12 | unframe | ok | ok | ok | ok |
| N13 | preserve | ok | ok | ok | ok |
| N14 | unframe | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | ok | ok | ok |
| N15 | preserve | ok | ok | ok | ok |
| N16 | refuse `KeyError`: source axis 'y' is fixed and cannot be selected by label | ok | ok | ok | ok |
| N17 | preserve | hole (ZeroDivisionError: integer modulo by zero) | ok | ok | ok |
| N18 | preserve | ok | ok | ok | ok |
| N19 | refuse `ValueError`: would corrupt the following index | ok | ok | ok | ok |
| N20 | refuse `ValueError`: would corrupt the following index | ok | ok | ok | ok |
| N21 | preserve | ok | ok | ok | ok |
| N22 | unframe | ok | ok | ok | ok |
| N23 | preserve | ok | ok | ok | ok |
| N24 | preserve | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | ok | ok |
| N25 | refuse `ValueError`: incompatible framed coordinate mappings | ok | ok | ok | ok |
| N26 | preserve | ok | ok | ok | ok |
| N27 | refuse `ValueError`: incompatible framed coordinate mappings | ok | ok | ok | ok |
| N28 | refuse `ValueError`: left operand is anonymous | ok | ok | ok | ok |
| N29 | preserve | ok | ok | ok | ok |
| N30 | preserve; preserve | hole (AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes) | hole (AssertionError: expected preserved binding) | ok | ok |
| N31 | refuse `ValueError`: this transform cannot rename source axes | ok | ok | ok | ok |
| N32 | preserve | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | ok | ok |
| N33 | preserve; unframe | hole (CoordinateValidationError: coordinate x has dimensions ('x',), but these are not a subset of the DataArray dimensions) | hole (CoordinateValidationError: coordinate x has dimensions ('x',), but these are not a subset of the DataArray dimensions) | ok | ok |
| N34 | refuse `ValueError`: would corrupt the following index | hole (preserved) | hole (preserved) | ok | ok |
| N35 | refuse `ValueError`: vectorized indexing changes geometry dimensions; call rf.unframe() | ok | ok | ok | ok |
| N36 | preserve | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | ok |
| N37 | preserve | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | ok |
| N38 | unframe; preserve | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | ok |
| N39 | unframe; preserve | hole (AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes) | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | hole (ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor) | ok |

## Alternatives Considered

None; the probe design is recorded in [binding_design.md](binding_design.md) and the matrix
note.

## Deferred Work

- Geometry-aware coarsen, stack, pad and interpolation need explicit output-domain semantics
  before native support can replace the refusals.
- Matrix Q1: the transform-index update candidate in
  [upstream PR notes](../xarray-upstream/upstream_prs.md#deferred-work) still lacks a
  reproducer.

## Next Steps

Re-run all three lanes and regenerate this table when the patch series, `BindingIndex` or the
probe changes. The inventory measures 132 paths; it certifies neither unprobed operations nor
a release.
