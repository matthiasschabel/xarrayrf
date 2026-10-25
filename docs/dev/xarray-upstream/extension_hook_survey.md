# Extension hook survey

**Status:** Active
**Last updated:** 2026-10-25
**Scope:** Evidence that xarray extensions other than xarrayrf need the two `Index` hooks of
patch series 5 ([hook design](index_hook_design.md)), for a future upstream design issue.

## Context

Maintainers judge a new `Index` method by whether extensions they know need it. This note
records measured failures in four extensions and whether a hook resolves each one. The tool is
`tools/extension_hook_survey.py` (JSON output with the imported xarray path and commit); run it
in the `priorart` environment on each lane. Versions measured on 2026-10-25: pint-xarray 0.6.1
(pint 0.26.2), rasterix 0.2.2, xproj 0.2.1, cf-xarray 0.11.3, pyproj 3.8.0. Lanes: stock xarray
2026.7.0; upstream `main` (`b2acc77a`); series 5 (`a6576179`); the series-6 candidate (`b13b0a6b`, series 5 plus the
alignment fix below).

A hook is credited only by a same-tree control: the operation runs on the extension's own index
and on a subclass implementing only that hook. The hook resolves a case when the original index
fails, the subclass's hook is called, and the subclass then refuses or gives the correct result.
The subclasses are survey stubs, not proposals for the extensions: a refusal shows that the
extension could stop silent loss, while the better remedy (keeping units or the affine) stays
the extension's choice. Stock xarray never calls the hooks. There the override subclasses
fail with an `AlignmentError` instead, because their type differs from the other operand's
index; that is a side effect of the control, not evidence.

## Current Decision

| Case | Library | Operation | Hook | Original index: stock / main / series 5 / series 6 | Hook subclass: stock / main / series 5 / series 6 |
|---|---|---|---|---|---|
| `pint_stack` | pint-xarray | stack | A (check_drop) | units of the indexed coordinate dropped (index now PandasIndex) | not reached: units of the indexed coordinate dropped (index now PandasIndex) / not reached: units of the indexed coordinate dropped (index now PandasIndex) / hook resolves / hook resolves |
| `pint_pad` | pint-xarray | pad | A (check_drop) | units of the indexed coordinate dropped (index now PandasIndex) | not reached: units of the indexed coordinate dropped (index now PandasIndex) / not reached: units of the indexed coordinate dropped (index now PandasIndex) / hook resolves / hook resolves |
| `pint_coarsen` | pint-xarray | coarsen | A (check_drop) | units of the indexed coordinate dropped (index now PandasIndex) | not reached: units of the indexed coordinate dropped (index now PandasIndex) / not reached: units of the indexed coordinate dropped (index now PandasIndex) / hook resolves / hook resolves |
| `pint_interp` | pint-xarray | interp | A (check_drop) | units of the indexed coordinate dropped (index now PandasIndex) | not reached: units of the indexed coordinate dropped (index now PandasIndex) / not reached: units of the indexed coordinate dropped (index now PandasIndex) / hook resolves / hook resolves |
| `pint_override_units` | pint-xarray | align(km, m, join='override') | B (join_overlapping, how='override') | metre labels relabelled as kilometer | not reached: AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes / not reached: AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes / hook resolves / hook resolves |
| `pint_unindexed_conflict` | pint-xarray | quantified + plain array whose unindexed x differs | B (join_overlapping, how=None) | unindexed x with different values silently discarded | not reached: unindexed x with different values silently discarded / not reached: unindexed x with different values silently discarded / hook resolves / hook resolves |
| `pint_broadcast` | pint-xarray | broadcast against a coordinate-free array | broadcast fix (no hook) | units of the indexed coordinate dropped (index now PandasIndex) / units of the indexed coordinate dropped (index now PandasIndex) / correct / correct | — |
| `rasterix_stack` | rasterix | stack | A (check_drop) | RasterIndex (affine and CRS) replaced by ['PandasMultiIndex'] | not reached: RasterIndex (affine and CRS) replaced by ['PandasMultiIndex'] / not reached: RasterIndex (affine and CRS) replaced by ['PandasMultiIndex'] / hook resolves / hook resolves |
| `rasterix_pad` | rasterix | pad | A (check_drop) | RasterIndex (affine and CRS) replaced by ['PandasIndex'] | not reached: RasterIndex (affine and CRS) replaced by ['PandasIndex'] / not reached: RasterIndex (affine and CRS) replaced by ['PandasIndex'] / hook resolves / hook resolves |
| `rasterix_coarsen` | rasterix | coarsen | A (check_drop) | RasterIndex (affine and CRS) replaced by ['PandasIndex'] | not reached: RasterIndex (affine and CRS) replaced by ['PandasIndex'] / not reached: RasterIndex (affine and CRS) replaced by ['PandasIndex'] / hook resolves / hook resolves |
| `rasterix_interp` | rasterix | interp | A (check_drop) | RasterIndex (affine and CRS) replaced by ['PandasIndex'] | not reached: RasterIndex (affine and CRS) replaced by ['PandasIndex'] / not reached: RasterIndex (affine and CRS) replaced by ['PandasIndex'] / hook resolves / hook resolves |
| `rasterix_broadcast` | rasterix | broadcast against a coordinate-free array | broadcast fix (no hook) | ValueError: new dimensions {'y': 3, 'c': 2} must be a superset of existing dimensions ('y', 'x') / RasterIndex (affine and CRS) replaced by ['PandasIndex'] / correct / correct | — |
| `rasterix_profile_first` | rasterix | 1-D array on x with equal labels + raster | B (join_overlapping, join) | AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes / RasterIndex split into ['PandasIndex', 'RasterIndex'] / RasterIndex split into ['PandasIndex', 'RasterIndex'] / RasterIndex split into ['PandasIndex', 'RasterIndex'] | not reached: AlignmentError: cannot align objects on coordinate 'x' because of conflicting indexes / not reached: RasterIndex split into ['HookedRasterIndex', 'PandasIndex'] / hook reached but not sufficient / hook resolves |
| `rasterix_join_crs` | rasterix | RasterIndex.join of two EPSG:32633 indexes | not a hook: the extension's own join | RasterIndex.join returns crs=None | — |
| `xproj_override_crs` | xproj | align(EPSG:4326, OGC:CRS84, join='override') | B (join_overlapping, how='override') | OGC:CRS84 operand relabelled EPSG:4326 | not reached: AlignmentError: cannot align objects on coordinate 'spatial_ref' because of conflicting indexes / not reached: correct / hook resolves / hook resolves |
| `xproj_two_crs_names` | xproj | EPSG:4326 on spatial_ref + OGC:CRS84 on crs | not a hook: different coordinate names never overlap | two CRS coordinates coexist: ['crs', 'spatial_ref'] | — |
| `cf_coarsen_bounds` | cf-xarray | coarsen(x=2).mean() with x_bounds | not a hook today: bounds are attribute-associated; A would apply to an index owning x and x_bounds | bounds averaged to [0.0, 1.0] instead of [-0.5, 1.5] | — |

Per hook:

- **`check_drop` (hook A): two extensions.** pint-xarray loses the units of an indexed
  coordinate, and rasterix loses its affine and CRS, through `stack`, `pad`, `coarsen` and
  `interp`, silently on every lane. A subclass refuses each one on series 5.
- **`join_overlapping(how="override")`: two extensions.** xproj relabels an OGC:CRS84 operand
  as EPSG:4326, and pint-xarray replaces metre coordinate labels with kilometre labels, under `join="override"`.
- **`join_overlapping(how=None)`: one extension.** pint-xarray discards an unindexed `x` with
  different values (the same root cause as #11607). The rioxarray first-CRS-wins case involves no
  index on either side, so this hook does not reach it.
- **`join_overlapping` joins: one extension, after a fix.** A 1-D array on `x` with equal labels
  followed by a raster splits `RasterIndex` into a `PandasIndex` on `x` beside it. On series 5 the
  hook is called but the split remains; xarrayrf's binding split the same way
  ([inventory](../binding/binding_operation_inventory.md) N36-N39). The no-reindex alignment path
  kept the lower-dimensional operand's own index for a coordinate the join had taken over, so the
  following merge chose it. The series-6 candidate drops that index as the reindexing path
  already does, and the case resolves.
- **Broadcast fix (no hook):** pint-xarray and rasterix both keep their index through
  `broadcast` on series 5; upstream `main` loses it, and 2026.7.0 loses it or raises.
- **Outside the hooks:** rasterix's own `RasterIndex.join` returns `crs=None` (an extension bug);
  xproj lets CRS coordinates with different names coexist (no shared name, so no overlap);
  cf-xarray's bounds are attribute-associated, and `coarsen` averages them (hook A would apply
  only if an index owned `x` and `x_bounds`).

Tracker searches (2026-10-25, issues and PRs, open and closed) by symptom in pydata/xarray and
each extension's repository found no report of these failures in pint-xarray, rasterix or xproj.
Related threads: xarray #11607 and #11716 (indexed versus unindexed values), #11215
(multi-coordinate indexes dropped), #11615 (broadcast), xproj #25 (CRSIndex and CRS-aware
indexes), cf-xarray #10 (cell-boundary-aware operations). Terms: custom index dropped stack/pad,
coarsen custom index, broadcast custom index, join override custom index, interp custom index
dropped, index unindexed coordinate silently; pint-xarray stack units lost, coarsen,
PintIndex dropped, override, pad; rasterix stack, coarsen, pad, interp, concat crs; xproj
override, align crs; cf-xarray coarsen bounds, bounds index.

## Alternatives Considered

- Counting the stock-to-series-5 difference as hook evidence: series 5 also carries
  independent fixes, so only same-tree controls isolate a hook.
- Stub indexes written for the survey instead of the extensions' own classes: they would show
  the hook works, not that an extension needs it.

## Deferred Work

- MetPy, xvec and xdggs were not surveyed.
- The validation and join modes each have one extension case besides xarrayrf; a design issue
  should say so rather than imply broader demand.

## Next Steps

1. Publish series 6 (the alignment fix) once approved, re-pin, and land N36-N39.
2. Use this table in the design-issue draft; re-run the survey when an extension releases.
