# Index hook consolidation design

**Status:** Active
**Last updated:** 2026-09-28
**Scope:** Local xarray patches 3, 4, 5, 9 and 10 and the matching `BindingIndex` methods in
`src/xarrayrf/_binding.py`. Goal: the same native-operation contract with two new `Index`
methods instead of seven, on a future series 5, plus the cross-extension evidence a design
issue upstream would need.

## Context

The local series adds seven optional `Index` methods. They express two concepts: "this
operation is about to drop your index; may it?" and "another operand touches a subset of your
coordinates; how do we join?". Seven new methods is a different review conversation from two,
and maintainers would likely force the consolidation during review; doing it first keeps the
design in this project's hands.

| Hook | Patch | Called from | Semantics |
|---|---|---|---|
| `check_stack(dims)` | 10 | `Dataset._stack_once` | veto; xarray then drops the index |
| `check_pad(pad_width)` | 10 | `Dataset.pad` | veto |
| `check_coarsen(windows)` | 9 | `Coarsen.__init__` | veto |
| `check_override(other)` | 3 | `Aligner.override_indexes` | veto for `join="override"` |
| `check_unindexed_coord_conflicts(variables)` | 3 | alignment, `merge_collected` | veto when same-name unindexed coordinates would be replaced |
| `join_overlapping(other_indexes, *, other_variables, how) -> (Index, targets) \| None` | 3, 4 | `Aligner.align_indexes` (two sites) | join with plain indexes covering a subset of this index's coordinates |
| `swap_dims(dims_dict) -> Self \| None` | 5 | `Dataset.swap_dims` | return renamed index or drop |

Line numbers in the series 2 diff (`git -C ~/GitHub/xarray-upstream diff 8de862c2..2af8cfea --
xarray/core/indexes.py`) are stale; locate call sites on series 4. `BindingIndex` implements all
seven; the three vetoes are one-line raises with the same message shape. Local patch 7b also
calls `join_overlapping` from `Dataset.update`; the consolidated method must keep that working.

## Current Decision

### Hook A: one preflight veto

Replace `check_stack`, `check_pad` and `check_coarsen` with one optional method, default no-op,
called once per affected index before xarray drops or replaces it. Working shape (name open):

```python
def check_drop(self, operation: str, dims: Sequence[Hashable]) -> None:
    """Called before `operation` ("stack", "pad", "coarsen", ...) drops this index from `dims`.
    Raise ValueError to refuse; the default allows the drop."""
```

A veto does not need the per-operation payload (`pad_width`, `windows`): `BindingIndex` refuses
any geometry-dimension drop regardless. If a maintainer wants the payload, pass it as
keyword-only `**context`. The three call sites stay. Precedent: `Index.isel` returning `None`
already lets an index decide its own fate.

### Hook B: one overlapping join

Fold `check_override` and `check_unindexed_coord_conflicts` into `join_overlapping`, so
alignment and merge consult one method whenever another operand touches a subset of this
index's coordinates:

- plain `PandasIndex` overlaps: unchanged;
- `how="override"`: the index raises (today's `check_override`) or returns the override target;
- unindexed same-name coordinates: `other_indexes` empty, `other_variables` carries the plain
  coordinates; the index compares values and units and raises or returns itself unchanged.

Unverified: the unindexed case also fires from `merge_collected`, so the merge path must call a
join-shaped method with `how` set to the merge's compat semantics, or with an explicit
`how=None` meaning "validate only, no reindex". If that cannot be expressed cleanly, fall back
to three methods (`join_overlapping` plus one conflict check) and record why two was impossible.
Do not reach three by default.

### Outside the count: `swap_dims`

Patch 5 is not part of this proposal. benbovy's open refactor #8911 records that maintainers do
not want custom indexes recreated; patch 5 is compatible with an index keeping itself. Offer it
as a contribution to #8911 (beside open PR #11100), and keep `BindingIndex.swap_dims` as is.

### Split the bundled patches first

| Patch | Fix half (to [upstream_prs.md](upstream_prs.md)) | Hook half (stays here) |
|---|---|---|
| 4 | propagate joined indexes to every aligned object including no-reindex paths; preserve custom indexes through `broadcast` (`_broadcast_helper`) | pass indexed plain-coordinate variables to `join_overlapping`; check override for every overlapping pair in both orders |
| 9 | coarsen retains unaffected indexes across DataArray/Dataset reduction and `construct` | `check_coarsen` call in `Coarsen.__init__` |

Whether the patch 4 fix half stands alone without `join_overlapping` must be tested: binding
retention through broadcast historically required the hook refinements too.

### Rebuild plan (future series 5)

On a new branch and worktree cut from series 4 (never modify `index-hooks-4`), or from a newer
upstream `main` if more fixes have merged:

1. Carry the PR-derived patches unchanged (they leave when merged upstream).
2. Add the split fix halves of 4 and 9 as their own commits.
3. Replace patches 3, 4-hook, 9-hook and 10 with two commits: hook A with its three call sites,
   hook B with the alignment and merge call sites. Rewrite `xarray/tests/test_index_hooks.py`
   around the two methods, one behavior per test, same behaviors covered.
4. Keep patch 5 last, labelled as the #8911 contribution; keep 7b working.
5. Update `BindingIndex` to implement `check_drop` and the widened `join_overlapping`; delete
   the five replaced methods. No public xarrayrf API changes (`docs/design.md` §5 names no hook).

Validation, recorded in [xarray_patches.md](xarray_patches.md):

```sh
# xarray, in the new worktree
PYTHONPATH=$PWD ~/GitHub/xarray-upstream/.venv/bin/pytest xarray/tests/test_index_hooks.py \
  xarray/tests/test_dataset.py xarray/tests/test_indexes.py xarray/tests/test_coarsen.py -q
PYTHONPATH=$PWD ~/GitHub/xarray-upstream/.venv/bin/pytest xarray/tests -n 6 -q
# xarrayrf against the new tree
PYTHONPATH=$HOME/GitHub/<new-worktree> uv run --no-sync pytest
PYTHONPATH=$HOME/GitHub/<new-worktree> uv run --no-sync \
  python explorations/binding_operation_probe.py
make check
```

Gate: 93/93 `ok` on the patched lane; stock and upstream strict xfails still xfail. Publishing
(push, tag `xarrayrf-patches-5`, re-pin, pretend version, refresh `.venv-patched`) needs the
maintainer's authorization.

### Cross-extension evidence for the design issue

Maintainers judge a new `Index` method by whether extensions they know need it. CRS demand
(#2288) was met by custom indexes (xproj `CRSIndex`); per-variable coordinates (#9152) was
closed in favour of cf-xarray. A proposal backed only by xarrayrf is expected to stall on "who
else uses this". Target: at least two extension-backed reproducers per hook, or a note stating
that fewer exist and what that means.

Observations already verified in
[prior_art_coverage_study.md](../architecture/prior_art_coverage_study.md) and pinned by
`explorations/test_geo_failure_modes.py`. The hook column is a hypothesis per row:

| Observed failure | Library | Candidate hook |
|---|---|---|
| `join="override"` silently replaced CRS EPSG:4326 with CRS84 | xproj | B (override) |
| Conflicting CRSs combine silently; first operand's CRS wins | rioxarray (`spatial_ref` + `grid_mapping`) | B (unindexed conflict) |
| Two CRS coordinates under different names coexist after arithmetic | xproj | B (conflict) |
| Joins and `concat` return CRS `None`, which then matches any CRS | rasterix | B, or A (refuse) |
| `RasterIndex` dropped by coupled and scalar selection | rasterix (`7899c7a2`) | not a hook; related to #11617 |
| Stale `GeoTransform` after strided `isel` | rioxarray | not a hook; attribute-carried metadata |

Unchecked candidates: pint-xarray (units stripped and restored around `interp`, `ffill`,
`bfill`, `chunk`; units on indexed coordinates), cf-xarray (per-variable `grid_mapping`, bounds,
ancillary association), MetPy, xvec, xdggs. Treat these as hypotheses until verified.

Method per candidate: reproduce on stock xarray 2026.7.0 with pinned versions, preferably also
on the patched lane with a stub `Index` implementing the hook; locate the xarray code path;
search xarray's and the extension's trackers by symptom including closed items (start from
#2996, #11607, #6481, #8911, #8914, #11099, #11100, #11215, #11286, #9152, #2288) and record
terms and dates; classify as already reported, same symptom different cause, or unreported.
Time-box about an hour per extension; record "none found" with the operations tried.

Installed in the `priorart` group: rasterix 0.2.2, rioxarray 0.23.0, xproj 0.2.1. Install
others in a separate environment (`UV_PROJECT_ENVIRONMENT`) so `.venv` stays the stock lane.

## Alternatives Considered

- Keep seven hooks and let review shrink them: costs a review round and hands the design to the
  reviewer.
- Widen `Index.join(other, how)` to accept a partial cover: `join` takes a same-coordinate
  `Index` and returns one index, while the overlapping case needs per-coordinate targets back.
  Reusing the name overloads known semantics. Revisit only if the merge-path analysis shows
  `join` fits naturally.
- Payload-specific vetoes (`check_pad(pad_width)` and so on): unnecessary API surface.
- A general "notify index of any lifecycle event" hook: rejected (`docs/design.md` §5 and
  [binding_design.md](../binding/binding_design.md)); no measured hole needs it.
- Persuading maintainers by polishing xarrayrf (adapters, notebooks): wrong audience.
- Filing the hook proposal now with xarrayrf-only evidence: expected to stall; #11607 has had no
  response, and #6481 and #8914 have none.
- Claiming xarrayrf subsumes rioxarray, xproj or rasterix: position it as one identity and
  lifecycle user beside them.

## Deferred Work

- Proposing hooks A and B upstream: gated on the evidence above and on the bug-fix PRs.
- Retiring the stock-lane `join_overlapping` xfails once a released xarray carries the hook.
- Any lifecycle-notification hook the survey surfaces: record demand, do not propose.

## Next Steps

1. Analyse the merge path for hook B; record "two methods, verified at all alignment and merge
   call sites" or "three, because ..." in *Current Decision*.
2. Split patches 4 and 9; test the patch 4 fix half standalone; hand fix halves to
   [upstream_prs.md](upstream_prs.md).
3. Build series 5 from series 4, update `BindingIndex`, run the validation above, update the
   manifest.
4. Turn the six verified observations into reproducers with code paths and hook
   classification; add pint-xarray and cf-xarray reproducers and tracker searches.
5. Write an evidence note (`docs/dev/extension_hook_survey.md`, one row per reproducer with
   paired pinned tests) and a design-issue draft for the author to rewrite and post: the two
   methods and defaults, a table of extensions and the failure each hook fixes, the xarray code
   paths, related threads, xarrayrf named once. Under 500 words, at most one dash.
6. List extension maintainers or threads where a co-sponsor might be sought, with evidence. The
   maintainer decides whether to approach anyone, and whether patch 5 goes to #8911 as a comment
   (recommended first) or a dependent PR.
