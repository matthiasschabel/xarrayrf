# Native binding operation inventory

**Status:** Active
**Last updated:** 2026-09-28
**Scope:** Measured outcome of 93 native xarray operations on a framed DataArray, on stock
xarray, upstream `main` and the patched series; the release gate for the private `BindingIndex`.

## Context

The probe is `tools/binding_operation_probe.py` (`make probe`; it prints JSON, and this
table is transcribed from it). Lanes at measurement: stock xarray 2026.7.0; upstream `main` at
`dfd25c72`; patched series 4 (tag `xarrayrf-patches-4`, `5db75d53`). The reported xarray
version can be stale editable-install metadata under `PYTHONPATH`; the probe's import paths
identify the lanes. Design and rationale: [binding_design.md](binding_design.md).

Every preserved result is checked against the original transform and against the original
world points at matching source-coordinate labels, with absolute tolerance `1e-12` world units
(the synthetic grid is integer-spaced, so this covers affine rounding without accepting a moved
sample).

Classification: `ok` means preserved with matching world points, a permitted refusal, or an
unframed result with no geometry in the source frame. `hole` blocks the native-operation
contract. Required refusals and absence of a sibling binding are checked separately; a partial
tuple result is `unframed`. The point oracle checks placement, not conflict history, so
operand-conflict and override cases require refusal even when points look valid. The probe
accepts any exception for may-refuse cases; focused tests in `tests/test_native.py` assert the
exact error type and message.

## Current Decision

One row per probe case. "Remaining hole / owner" names the fixing patch (series 1/2 commit ids
or patch numbers; see the patch table in [binding_design.md](binding_design.md)). "Fault path"
and the reproducer were established on upstream `8de862c`. Totals: stock 67/93 `ok`
(26 holes), upstream 64/93 (29), patched 93/93 (0). The patched lane is an experiment, not a
release pass.

Fixture used by the reproducers:

```python
import numpy as np
import xarray as xr
from xarrayrf import AffineTransform, ArrayCoordinates, CoordinateSystem, ReferenceFrame
import xarrayrf.native

frame = ReferenceFrame.local(CoordinateSystem(("a", "b"), ("mm", "mm")))
transform = AffineTransform.from_matrix(
    source=ArrayCoordinates(("y", "x"), ("1", "1")),
    target=frame,
    matrix=np.eye(2),
    translation=np.zeros(2),
)
raw = xr.DataArray(
    np.arange(12).reshape(3, 4),
    dims=("y", "x"),
    coords={"y": [0, 2, 4], "x": [0, 3, 6, 9]},
)
raw = raw.assign_coords(y=raw.y.assign_attrs(units="1"), x=raw.x.assign_attrs(units="1"))
f = raw.rf.frame(transform, dims=("y", "x"))
plane = f.isel(y=1)
shifted = raw.assign_coords(x=[1, 4, 7, 10]).rf.frame(transform, dims=("y", "x"))
conflict = xr.DataArray(np.ones(4), dims="x", coords={"x": [0, 3, 6, 9], "y": 4})
wrong_units = raw.assign_coords(x=raw.x.assign_attrs(units="cm"))
```

Cells marked "was: preserved, incorrect" were re-measured after split-binding detection was
added: a binding that lost a coordinate to another index now raises on inspection instead of
reading as preserved.

| Case | Stock outcome | Stock | Upstream outcome | Upstream | Patched outcome | Patched | Remaining hole / owner | Fault path on 8de862c | Minimal reproducer |
|---|---|---|---|---|---|---|---|---|---|
| isel_scalar | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| isel_slice | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| isel_array | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| isel_vectorized | raises ValueError: vectorized indexing changes geometry dimensions; call rf.unframe() | ok | raises ValueError: vectorized indexing changes geometry dimensions; call rf.unframe() | ok | raises ValueError: vectorized indexing changes geometry dimensions; call rf.unframe() | ok | — | — | — |
| isel_drop_true | preserved, correct | hole | preserved, correct | hole | raises ValueError: cannot remove coordinate(s) 'y', which would corrupt the following index built from coordinates 'y', 'x': | ok | — | core/dataarray.py:DataArray.isel | `f.isel(y=1, drop=True)` |
| sel_scalar | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| sel_slice | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| sel_nearest | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| sel_array | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| sel_vectorized | raises ValueError: vectorized indexing changes geometry dimensions; call rf.unframe() | ok | raises ValueError: vectorized indexing changes geometry dimensions; call rf.unframe() | ok | raises ValueError: vectorized indexing changes geometry dimensions; call rf.unframe() | ok | — | — | — |
| sel_drop_true | preserved, correct | hole | preserved, correct | hole | raises ValueError: cannot remove coordinate(s) 'y', which would corrupt the following index built from coordinates 'y', 'x': | ok | — | core/dataarray.py:DataArray.sel → DataArray.isel | `f.sel(y=2, drop=True)` |
| head | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| tail | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| thin | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| transpose | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| rename | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| swap_dims | raises ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor unframed. Call rf.u (was: preserved, incorrect; the split binding is now detected on inspection) | hole | raises ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor unframed. Call rf.u (was: preserved, incorrect; the split binding is now detected on inspection) | hole | preserved, correct | ok | xarray `Index.swap_dims` hook (`20036f16`) | core/dataset.py:Dataset.swap_dims | `f.swap_dims({"x": "column"})` |
| expand_dims | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| squeeze | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| squeeze_geometry | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| add_framed | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| add_different_transform | raises ValueError: incompatible framed coordinate mappings: the operands sample the same frame on different grids; resample one onto the other with rf.resample_to | ok | raises ValueError: incompatible framed coordinate mappings: the operands sample the same frame on different grids; resample one onto the other with rf.resample_to | ok | raises ValueError: incompatible framed coordinate mappings: the operands sample the same frame on different grids; resample one onto the other with rf.resample_to | ok | — | — | — |
| add_unframed | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | preserved, correct | ok | preserved, correct | ok | — | structure/alignment.py:Aligner.align_indexes | `f + raw` |
| add_unframed_reverse | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | unframed | hole | preserved, correct | ok | — | structure/alignment.py:Aligner.align_indexes | `raw + f` |
| add_unframed_shifted_labels | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | raises AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes | hole | preserved, correct | ok | — | structure/alignment.py:Aligner.align_indexes | `f + raw.assign_coords(x=[3, 6, 9, 12])` |
| add_unframed_conflict | raises AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes | ok | preserved, correct | hole | raises ValueError: conflicting unindexed coordinate 'y' | ok | — | structure/merge.py:merge_collected | `plane + conflict` |
| add_unframed_conflict_reverse | raises AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes | ok | preserved, correct | hole | raises ValueError: conflicting unindexed coordinate 'y' | ok | — | — | — |
| add_unframed_unit_conflict | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | ok | preserved, correct | hole | raises ValueError: coordinate 'x' declares attrs['units'] = 'cm', but the transform's source declares that axis in '1'; convert a | ok | — | structure/merge.py:merge_collected | `f + wrong_units` |
| add_unframed_unit_conflict_reverse | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | ok | unframed | hole | raises ValueError: coordinate 'x' declares attrs['units'] = 'cm', but the transform's source declares that axis in '1'; convert a | ok | — | — | — |
| add_unindexed_unit_conflict | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | ok | preserved, correct | hole | raises ValueError: coordinate 'x' declares attrs['units'] = 'cm', but the transform's source declares that axis in '1'; convert a | ok | — | — | — |
| add_unindexed_unit_conflict_reverse | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | ok | preserved, correct | hole | raises ValueError: coordinate 'x' declares attrs['units'] = 'cm', but the transform's source declares that axis in '1'; convert a | ok | — | — | — |
| add_scalar | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| add_scalar_reverse | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| ufunc_framed | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| ufunc_unframed | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | preserved, correct | ok | preserved, correct | ok | — | structure/alignment.py:Aligner.align_indexes | `np.add(f, raw)` |
| ufunc_unframed_reverse | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | unframed | hole | preserved, correct | ok | — | structure/alignment.py:Aligner.align_indexes | `np.add(raw, f)` |
| ufunc_scalar | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| ufunc_scalar_reverse | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| ufunc_unary | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| ufunc_unframed_conflict | raises AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes | ok | preserved, correct | hole | raises ValueError: conflicting unindexed coordinate 'y' | ok | — | structure/merge.py:merge_collected | `np.add(plane, conflict)` |
| ufunc_unframed_conflict_reverse | raises AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes | ok | preserved, correct | hole | raises ValueError: conflicting unindexed coordinate 'y' | ok | — | — | — |
| where_method | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| where_function | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| where_unframed_x | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | preserved, correct | ok | preserved, correct | ok | — | structure/alignment.py:Aligner.align_indexes | `xr.where(f > 2, raw, f)` |
| where_unframed_y | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | preserved, correct | ok | preserved, correct | ok | — | structure/alignment.py:Aligner.align_indexes | `xr.where(f > 2, f, raw)` |
| where_unframed_condition | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | preserved, correct | ok | preserved, correct | ok | — | — | — |
| where_framed_x | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | unframed | hole | preserved, correct | ok | — | — | — |
| where_framed_y | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | unframed | hole | preserved, correct | ok | — | — | — |
| where_conflicting_coordinate | raises AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes | ok | raises ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor unframed. Call rf.u | ok | raises ValueError: conflicting unindexed coordinate 'y' | ok | — | structure/merge.py:merge_collected | `xr.where(conflict > 0, plane, 0)` |
| align_inner | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| align_outer | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| align_exact | raises AlignmentError: cannot align objects with join='exact' where index/labels/sizes are not equal along these coordinates (dimensions): 'y' ('y',), 'x' ('x',) | ok | raises AlignmentError: cannot align objects with join='exact' where index/labels/sizes are not equal along these coordinates (dimensions): 'y' ('y',), 'x' ('x',) | ok | raises AlignmentError: cannot align objects with join='exact' where index/labels/sizes are not equal along these coordinates (dimensions): 'y' ('y',), 'x' ('x',) | ok | — | — | — |
| align_override | preserved, correct | hole | preserved, correct | hole | raises ValueError: join='override' is unsupported for framed coordinates | ok | — | structure/alignment.py:Aligner.override_indexes | `xr.align(f, shifted, join="override")` |
| align_unframed | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | unframed | hole | preserved, correct | ok | — | structure/alignment.py:Aligner.align_indexes | `xr.align(f, raw)` |
| align_unframed_reverse | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | unframed | hole | preserved, correct | ok | — | — | — |
| align_unframed_shifted_right | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | raises AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes | hole | preserved, correct | ok | — | structure/alignment.py:Aligner._reindex_one | `xr.align(f, raw.assign_coords(x=[3, 6, 9, 12]), join="right")` |
| align_override_framed_first | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | ok | unframed | hole | raises ValueError: join='override' is unsupported for framed coordinates | ok | — | — | — |
| align_override_framed_second | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | ok | unframed | hole | raises ValueError: join='override' is unsupported for framed coordinates | ok | — | — | — |
| broadcast | raises ValueError: new dimensions {'y': 3, 'channel': 2} must be a superset of existing dimensions ('y', 'x') | hole | unframed | hole | preserved, correct | ok | — | structure/alignment.py:broadcast | `xr.broadcast(f, xr.DataArray([1, 2], dims="channel"))` |
| broadcast_plane_volume | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | ok | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | ok | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | ok | — | — | — |
| sum_geometry | unframed | ok | unframed | ok | unframed | ok | — | — | — |
| sum_nongeometry | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| coarsen | unframed | hole | unframed | hole | raises ValueError: cannot coarsen geometry dimensions; call rf.unframe() | ok | local patch 9 and binding preflight | core/dataset.py:Dataset.coarsen | `f.coarsen(x=2).mean()` |
| rolling | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| groupby | raises ValueError: cannot drop or update coordinate(s) 'y', which would corrupt the following index built from coordinates 'y', 'x': | ok | raises ValueError: cannot drop or update coordinate(s) 'y', which would corrupt the following index built from coordinates 'y', 'x': | ok | raises ValueError: cannot drop or update coordinate(s) 'y', which would corrupt the following index built from coordinates 'y', 'x': | ok | — | — | — |
| resample | unframed | ok | unframed | ok | unframed | ok | — | — | — |
| stack | unframed | hole | unframed | hole | raises ValueError: cannot stack geometry dimensions; call rf.unframe() | ok | local patch 10 and binding preflight | core/dataset.py:Dataset._stack_once | `f.stack(pixel=("y", "x"))` |
| unstack | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | Stack and unstack nongeometry dimensions; the binding remains on its geometry dimensions. |
| interp | raises ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor unframed. Call rf.u | ok | raises ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor unframed. Call rf.u | ok | raises ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor unframed. Call rf.u | ok | — | — | — |
| reindex | raises AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes | ok | raises AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes | ok | raises AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes | ok | — | — | — |
| reindex_like | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| shift | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| roll_values | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| roll_coords | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| roll_empty | raises ZeroDivisionError: integer modulo by zero | hole | preserved, correct | ok | preserved, correct | ok | merged upstream #11613 (was local patch 8) | core/variable.py:Variable._roll_one_dim | `f.isel(x=slice(0, 0)).roll(x=1, roll_coords=True)` |
| pad | unframed | hole | unframed | hole | raises ValueError: cannot pad geometry dimensions; call rf.unframe() | ok | local patch 10 and binding preflight | core/dataset.py:Dataset.pad | `f.pad(x=(1, 1))` |
| concat_geometry | raises ValueError: concatenation of framed arrays is unsupported; call rf.unframe() | ok | raises ValueError: concatenation of framed arrays is unsupported; call rf.unframe() | ok | raises ValueError: concatenation of framed arrays is unsupported; call rf.unframe() | ok | — | — | — |
| concat_nongeometry | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| merge | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| to_dataset_extract_owner | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| dataset_extract_sibling | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | preserved, correct | ok | preserved, correct | ok | — (shared grid, [binding_design.md](binding_design.md)); stock needs #11532 | — | `xr.Dataset({"owner": f, "sibling": raw})["sibling"]` |
| dataset_mean_geometry | raises ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor unframed. Call rf.u (was: preserved, incorrect; the split binding is now detected on inspection) | hole | raises ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor unframed. Call rf.u (was: preserved, incorrect; the split binding is now detected on inspection) | hole | unframed | ok | xarray `4b250c41` | core/dataset.py:Dataset.reduce | `xr.Dataset({"owner": f}).mean("x")` |
| dataset_quantile_geometry | raises ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor unframed. Call rf.u (was: preserved, incorrect; the split binding is now detected on inspection) | hole | raises ValueError: the binding no longer owns its source coordinate(s) ['x']; the array is neither framed nor unframed. Call rf.u (was: preserved, incorrect; the split binding is now detected on inspection) | hole | unframed | ok | xarray `4b250c41` | core/dataset.py:Dataset.quantile | `xr.Dataset({"owner": f}).quantile(0.5, dim="x")` |
| dataset_setitem_labelled | raises AlignmentError: cannot align objects on coordinate 'y' because of conflicting indexes | hole | unframed | hole | preserved, correct | ok | xarray `35f13b39` | structure/merge.py:dataset_update_method | `xr.Dataset({"owner": f}).assign(sibling=raw * 2)` |
| assign_coords | raises ValueError: cannot drop or update coordinate(s) 'x', which would corrupt the following index built from coordinates 'y', 'x': | ok | raises ValueError: cannot drop or update coordinate(s) 'x', which would corrupt the following index built from coordinates 'y', 'x': | ok | raises ValueError: cannot drop or update coordinate(s) 'x', which would corrupt the following index built from coordinates 'y', 'x': | ok | — | — | — |
| drop_vars | raises ValueError: cannot remove coordinate(s) 'x', which would corrupt the following index built from coordinates 'y', 'x': | ok | raises ValueError: cannot remove coordinate(s) 'x', which would corrupt the following index built from coordinates 'y', 'x': | ok | raises ValueError: cannot remove coordinate(s) 'x', which would corrupt the following index built from coordinates 'y', 'x': | ok | — | — | — |
| copy | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| astype | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| chunk | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| compute | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| dask_isel | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| dask_add | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |
| pickle | preserved, correct | ok | preserved, correct | ok | preserved, correct | ok | — | — | — |

### Interpretation

- Stock mixed labelled operations fail alignment before #11532. On upstream `main` (with
  #11532) the forward order works, but reverse order and `align(f, raw)` lose the frame, and
  indexed-vs-plain conflicts resolve silently (#11607). The patched lane preserves the joined
  binding in both orders and refuses conflicts.
- `sum_geometry` and this probe's `resample` aggregate over a geometry dimension; their results
  have no geometry in the source frame, so whole-binding loss is allowed. `coarsen(x=2)` would
  keep representative positions, so it must refuse rather than unframe.
- Unpatched geometry `stack`, `pad` and `coarsen` drop the index without calling it; the
  patched preflights refuse first. Nongeometry stack, unstack, pad and coarsen preserve the
  binding. The earlier `unstack` case stacked geometry first (losing the binding in `stack`);
  it was replaced by a nongeometry unstack check.
- Unpatched `swap_dims` and Dataset reductions split the index; this is now detected and
  refused on inspection.
- The original `broadcast` failure is generic: `_get_broadcast_dims_map_common_coords` skipped
  a dimension once any index claimed a coordinate of that name, so any index spanning several
  dimensions hit it. Merged #11615 lets the operation run; upstream then unframes, while the
  patched series keeps the custom index.
- Stock `roll` of a zero-length indexed dimension raised `ZeroDivisionError`; fixed by merged
  #11613, now on upstream `main`.
- Persistence is not an operation row: `rf.encode()` / `rf.decode()` round trips through netCDF
  and Zarr are tested in `tests/test_native_encoding.py`.

## Alternatives Considered

None; the probe design is recorded in [binding_design.md](binding_design.md).

## Deferred Work

Geometry-aware coarsen, stack and pad need explicit output-domain semantics before native
support can replace the refusals.

## Next Steps

Re-run all three lanes and update this table when the patch series or `BindingIndex` changes.
Keep stock and patched claims distinct: the inventory measures 93 paths and certifies neither
unprobed operations nor a release.
