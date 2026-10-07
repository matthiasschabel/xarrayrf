# Upstream xarray bug-fix pull requests

**Status:** Active
**Last updated:** 2026-10-06
**Scope:** The five independent bug-fix PRs split from the local patch series: #11613 and #11615
and #11616 merged, #11617 and #11621 open. No hook API is proposed here.

## Context

Five local patches were generic xarray bug fixes with xarray-only reproducers, so they went
upstream without waiting for the hook design. Waiting costs rebase drift and divergent upstream
fixes (#11286 already fixed the DataArray half of patch 6). Each merge retires a local patch;
see [xarray_patches.md](xarray_patches.md).

Each PR is one commit from a `pr/<name>` branch on the fork, in its own worktree
`~/GitHub/xarray-pr-<name>` cut from upstream `main`. PR bodies on GitHub are the record of the
posted text. Reproducers are in `tools/upstream_reproducers/pr_<number>.py`, built on the
`RasterIndex` from xarray's custom-index guide; run them with `tools/xrpr PATH_TO_XARRAY`
([docs/upstream.md](../../upstream.md)). Historical bug-fix base: `dfd25c72`.

PR status and remote heads below were checked through the GitHub API on 2026-10-06.
Local worktrees can lag those heads. The descriptions below explain the local fixes; consult
the posted PR for its newest implementation and review threads. No containing released-xarray
version has been verified;
merged fixes must not be presented as available in stock 2026.7.0.

xarray's AI policy (`doc/contribute/ai-policy.md`): the submitter reviews every line, and PR
descriptions and replies are in the submitter's own words. Nothing is posted or pushed without
the maintainer's authorization for that specific action.

## Current Decision

### #11613 Roll on an empty dimension (merged)

https://github.com/pydata/xarray/pull/11613. `roll` on a zero-length dimension raised
`ZeroDivisionError` from `Variable._roll_one_dim` and, with `roll_coords=True`, from
`PandasIndex.roll`; both now take the shift modulo `size or 1`.

```python
xr.DataArray([], dims="x").roll(x=1)  # ZeroDivisionError before the fix
```

Merged 2026-09-28 as `b335cea0`. Local patch 8 retired in series 3; branch, worktree and fork
branch removed. Reproducer runs against an upstream checkout containing the merge.

### #11615 Broadcast with an index spanning several dimensions (merged)

https://github.com/pydata/xarray/pull/11615. `_get_broadcast_dims_map_common_coords` skipped a
dimension once an index had contributed a same-named coordinate, so a 2-D index left `x` out of
the dimension map and `xr.broadcast(raster, band)` raised "new dimensions ... must be a superset
of existing dimensions". The fix collects dimensions and index coordinates separately.

Merged as `dfd25c7252e71a47e789aa24b8a06dd911a56461` (final head
`bb201f750a3a127293bde0d75857a8423bb4a772`). Local patch 1 retired in series 4; branches and
worktree pruned. Stock 2026.7.0 still lacks the fix; verify on a release before claiming the
stock contract passes. Preserving the custom index itself through `broadcast` is a separate
change (local patch 4).

### #11616 Drop a multi-coordinate index whole in Dataset reductions (merged)

https://github.com/pydata/xarray/pull/11616 merged 2026-10-01 as
`23c9dd138605ef46e59a7341a1c8e07821bf416f`, final head
`2c14a3595f731370668be4cc7bfcec81a9ea2726`. Series 4 still contains local patch 6 because its
immutable base predates this merge; retire it when a new series rebases onto the merged fix.

`Dataset.reduce`, `Dataset.quantile` and `Dataset._integrate_one` filter indexes by coordinate
name, so reducing away one dimension of a multi-coordinate index leaves it attached to the
surviving coordinates with stale dimensions. The fix uses `filter_indexes_from_coords`, as
DataArray reductions have since #11286. `PandasIndex` and `PandasMultiIndex` are unaffected.

```python
ds = raster.to_dataset(name="red")  # RasterIndex over x and y
ds.mean("x").xindexes  # main: RasterIndex on y, still spanning x; PR: dropped
```

Review notes:

- Relationship to #11215: the body says it is related but does not close it. #11215 asks to
  keep a multi-coordinate index on whichever coordinates survive; this PR makes Dataset match
  DataArray's all-or-nothing rule now. Expect a maintainer question if #11215 moves toward
  partial keep.
- `Dataset.interp` was checked and does not have the bug.
- The current-main refresh retained the original lambda tests; focused run 592 passed,
  7 skipped, 1 xfailed, 1 xpassed.

### #11617 Respect drop=True for scalar coordinates an index keeps (open)

https://github.com/pydata/xarray/pull/11617, head `ee1b7f7023b950e592ccffa7713d78cd1b34b318` on
`dfd25c72`. Local patch 2.

An index whose `isel` rebuilds the selected position as a 0-d coordinate returns it in
`index_variables`, and the `isel` paths copy it into the result before `drop` is considered, so
`drop=True` silently keeps it. The fix adds `drop_scalar_index_coords`, called from
`DataArray.isel`, `Dataset.isel` and `Dataset._isel_fancy`, dropping those coordinates as
`drop_vars` would and raising the existing corrupt-index error when only some of an index's
coordinates would go. Built-in indexes return `None` from a scalar `isel` and are unaffected.

```python
r.isel(x=1, y=0, drop=True).coords  # main: 0-d x and y kept; PR: empty
r.isel(y=0, drop=True)  # PR: ValueError (partial drop of a shared index)
```

Review notes:

- Expected question: raise on a partial drop versus keep-and-warn. Position: the old result
  broke the documented `drop=True` contract and `drop_vars` already raises the same error, so it
  is listed under Bug Fixes; the body offers to move it to Breaking Changes or warn instead.
- An earlier revision crashed when the object had an index the indexers did not touch; fixed
  and tested with an extra untouched `PandasIndex` dimension.

### #11621 Raise when Dataset.update would replace an index with another type (open)

https://github.com/pydata/xarray/pull/11621, remote head
`78074ee2f9d776ffc77df9fac635dedd7aeafcae`. Series 4 patch 7 reflects an earlier revision;
refresh from the posted diff in the next series. Local patch 7b stays after merge.

`Dataset.update`, `ds[name] = value` and `assign` align the incoming object and then give it
merge priority, so when labels are equal its default `PandasIndex` silently replaces the
Dataset's custom index. Now `_check_update_index_types` raises `AlignmentError` when an incoming
index on a coordinate the Dataset already indexes has a different type or covers different
coordinates, and the message suggests `.drop_indexes([...])` over whole incoming index groups.
A mapping key naming the coordinate still replaces it explicitly. pandas values are coerced as
`merge_core` does before the check. whats-new lists it under Breaking Changes: a Dataset with a
`RangeIndex` now refuses `ds["v"] = da` when `da` carries its own labels.

```python
ds = raster.to_dataset(name="red")
ds["nir"] = nir  # main: RasterIndex silently replaced; PR: AlignmentError
ds["nir"] = nir.drop_indexes(["x", "y"])  # keeps the RasterIndex
```

Review state: dcherian requested changes on 2026-09-28 (raise instead of keeping the Dataset's
index; resolve with `drop_indexes`). The rework and a follow-up (coerced pandas values, whole
group hint, explicit-replacement test, single `xindexes` binding, coordinate-set mismatch test)
were pushed and the author replied on the PR. Review remains open.

Possibly outstanding, check against the posted PR before responding:

- The posted AI Disclosure line predates the rework and does not say which tool wrote it. The
  wording is the author's call.
- The body sentence "plain `PandasIndex` datasets are unaffected" should be qualified for the
  reverse direction (incoming custom index) and the coordinate-set mismatch.

## Alternatives Considered

- One PR for all five fixes: rejected; independent review, and xarray asks for focused diffs.
- Opening PRs from the patch-series branch: rejected; the diff would carry the hook series.
- #11621 first design (internal `priority_overrides` on `merge_core`, keep the Dataset's index
  when incoming coordinates are equal, applied per index group): replaced at the maintainer's
  request by a raise. The keep behavior survives locally as patch 7b behind the opt-in hook.

## Deferred Work

Upstream bugs seen during review, out of scope, not filed (candidates for separate issues):

- With a shared custom index and incoming `a` equal, `b` unequal and unindexed, `update`
  silently drops `b`.
- On a transform-backed Dataset, an unequal incoming coordinate is reindexed away silently by
  alignment.
- `broadcast(..., exclude=<one dim of a multi-dim index>)` raises `CoordinateValidationError`
  on `main`.

The fix halves of local patches 4 (custom index preserved through `broadcast`) and 9 (coarsen
retains unaffected indexes) join this stream once split; see
[index_hook_design.md](index_hook_design.md). Patch 5 (`swap_dims`) is API and is not here.

## Next Steps

1. Respond to maintainer review on #11617 and #11621, in the author's own words.
2. On each merge, record it in [xarray_patches.md](xarray_patches.md) and retire the local patch
   in the next numbered series. Raise the stock floor only after a release is verified.
3. Open PRs for the split fix halves of patches 4 and 9, one per fix, with a reproducer script.
