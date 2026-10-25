# Index hook consolidation design

**Status:** Active
**Last updated:** 2026-10-24
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
| `check_override(other)` | 3, 4 | `Aligner.check_override_indexes` | veto for `join="override"`, both operand orders |
| `check_unindexed_coord_conflicts(variables)` | 3 | alignment, `merge_collected` | veto when same-name unindexed coordinates would be replaced |
| `join_overlapping(other_indexes, *, other_variables, how) -> (Index, targets) \| None` | 3, 4 | `Aligner.align_indexes` (two sites) | join with plain indexes covering a subset of this index's coordinates |
| `swap_dims(dims_dict) -> Self \| None` | 5 | `Dataset.swap_dims` | return renamed index or drop |

Evidence below is from `~/GitHub/xarray-patched-5`, `series5-port` at `dbe12c36`, based on
upstream `b2acc77a`: patches 2 (`8f7d0159`), 3 (`249f30ca`), 4 (`02e262bd`), 5 (`b019d23b`),
7 (`071648c8`), 9 (`b338e0bd`), 10 (`0a42dd98`), 7b (`dbe12c36`). Paths beginning `xarray/`
refer to that tree; line numbers refer to its tip, not the historical patch offsets.
This is R2a's read-only design analysis for [release gate R2](../release_plan.md); the
rebuild and its runtime validation remain future work. Patch 6 is already in the base.

## Current Decision

### Hook A: one preflight veto

Verified: one optional, default-no-op preflight covers all three vetoes. Working shape
(name subject to maintainer review):

```python
def check_drop(self, operation: str, dims: Sequence[Hashable]) -> None:
    """Check whether an operation may drop or replace this index."""
```

| Current call site | Consolidated call and timing |
|---|---|
| `xarray/core/dataset.py:5393`, `Dataset._stack_once` | `check_drop("stack", dims)` after expanding ellipsis, before transforming variables; once per affected index per stacking step, including `create_index=False` and internal stacking |
| `xarray/core/dataset.py:9398`, `Dataset.pad` | `check_drop("pad", tuple(index_pad_width))` before padding; only this index's requested dimensions, including explicit zero widths |
| `xarray/computation/rolling.py:1177`, `Coarsen.__init__` | `check_drop("coarsen", tuple(windows))` after missing-dimension validation, before reduction/construct; not called again during execution |

Keep the existing affected-index filters and payload dimension scope: stack/coarsen include
all operation dimensions; pad includes only affected dimensions. Each groups by index, so a
shared multi-coordinate index is consulted once; untouched indexes are skipped. These paths
also serve DataArray operations. `BindingIndex`'s three unconditional raises
(`src/xarrayrf/_binding.py:357`, `:365`, `:373`) need neither widths nor window sizes.
Whether other extensions need those payloads is a maintainer/evidence question, not a reason
to add speculative `**context` now.

### Hook B: one overlapping join

**Two methods, verified at all call sites** (A plus B, excluding `swap_dims`): the source audit
finds six production calls. Widen `join_overlapping` to accept `how: JoinOptions | None`.
`None` means validation only, never merge compat or an implicit join. The default returns
`None`; it declines a join in joining contexts and permits existing behavior in validation
contexts. Raising vetoes in every context. For validation, an implementing index may return
`(self, {})`; callers ignore the return. This preserves the current no-op defaults.

| Current call site | Semantics in force and consolidated call |
|---|---|
| `xarray/structure/alignment.py:530`, `Aligner.align_indexes` | Indexed strict subsets only; disabled for `join="override"` and explicit `indexes=` (`:494`). Keep `other_variables` with attrs. Pass inner/outer/exact unchanged; left/right are relative to the superset receiver, reversing according to subset position and the first/last superset pivot (`:516–529`). `None` declines; a tuple supplies the joined superset and subset reindex targets. |
| `xarray/structure/alignment.py:563`, same method | After multiple subsets, call the final joined index again with `how="left"` to refresh targets for final labels, regardless of the original join. Keep target coverage/group checks and the exact/no-reindex constraint (`:566–607`). |
| `xarray/structure/alignment.py:664`, `Aligner.check_override_indexes` | Alignment `join="override"`, not merge `compat="override"`. Every pair of participating indexes on different operands sharing any name is checked in both directions, including equal coordinate sets and partial intersections. Call each receiver with `how="override"`, maps keyed by shared names and the peer's variables/index objects. This is a veto-only call: ignore its return and retain xarray's first-index-per-matching-group replacement, size checks, and no reindexing (`:420`, `:666–681`). Do not install a returned target or require a strict subset here. |
| `xarray/structure/alignment.py:761`, `Aligner._get_indexes_and_vars` | Before installing an aligned index, check same-name, same-dimension coordinates that have no index (`:752–758`). Call `join_overlapping({}, other_variables=unindexed_vars, how=None)`, independently of the alignment join; ignore the return. Values/units may be refused, not reindexed. |
| `xarray/structure/merge.py:303`, `merge_collected` | Non-prioritized indexed coordinate wins over same-name, same-dimension unindexed candidates (`:279–303`). The check runs for every valid compat, including override; compat only affects other merge checks (e.g. indexed attrs at `:313`). No alignment join is available here. Use the same empty-index-map, `how=None` validation call; leave winner selection, indexed equality and attrs handling intact. Priority entries bypass this branch. |
| `xarray/structure/merge.py:1237`, `_check_update_index_types` (7b) | Keep `how="left"` with the Dataset's index as receiver. Only after incoming index groups fit inside its coordinate set and labels compare equal (`:1230–1235`), a non-`None` result accepts dropping incoming indexes; returned targets are not installed. Mapping assignment/update and Dataset update drop those indexes at `:1292–1300`. Subsequent `merge_core` uses **join outer, compat broadcast_equals, priority_arg=1 and explicit Dataset indexes** (`:1302–1308`), not a left merge. Decline retains patch 7's raise; exceptions propagate. |

Override must dispatch before the plain-PandasIndex join checks, since its peers need not be
plain or subsets. Validation-only unindexed calls compare values and units even under merge
compat override; weakening this would change the current opt-in contract. No call site needs
a third method, provided the maintainer accepts these explicit validation modes.

Existing coverage limits, not resolved by consolidation: exact/no-copy alignment with no
shadowed keys skips `_get_indexes_and_vars` (`xarray/structure/alignment.py:817`); override
also bypasses it. `merge_collected`'s map at `xarray/structure/merge.py:297` collapses multiple
unindexed candidates of one name to the last. The maintainer must decide whether to expand
coverage in a separate fix (including interaction with #11607); do not quietly change those
semantics while renaming hooks. A direct `Coordinates.merge` regression is needed to isolate
the merge hook from alignment (`xarray/core/coordinates.py:591`); the existing `xr.merge` test at
`xarray/tests/test_index_hooks.py:429` can raise earlier in alignment.

### Outside the count: `swap_dims`

Patch 5 is not part of this proposal. benbovy's open refactor #8911 records that maintainers do
not want custom indexes recreated; patch 5 is compatible with an index keeping itself. Offer it
as a contribution to #8911 (beside open PR #11100), and keep `BindingIndex.swap_dims` as is.

### Split the bundled patches first

| Patch | Exact generic fix hunks | Exact hook-dependent hunks |
|---|---|---|
| 4 (`02e262bd`) | All `DataArray.broadcast_like` changes (`xarray/core/dataarray.py:2072`); only the `Dataset.broadcast_like` hunk in `xarray/core/dataset.py:3454`; all changes to `_get_broadcast_dims_map_common_coords`, `_broadcast_helper` and `broadcast` in `xarray/structure/alignment.py:1281`, `:1299`, `:1474`. Carry coordinate variables **and** their indexes into reconstruction. | The other Dataset hunk, `_reindex_callback` (`:3494`); all `xarray/core/indexes.py` changes (override docstring and overlap signature/docs); all other alignment hunks: variables passed at `:524`/`:562`, expanded override checker at `:653`, removal of the old check inside `override_indexes`, and its preflight invocation at `:810`. |
| 9 (`b338e0bd`) | In `xarray/computation/rolling.py`, the three imports (`:14`, `:15`, `:23`), all of `_construct_coords` (`:1188`), all `construct` reconstruction changes (`:1335–1356`), `v.variable` and coordinate assembly in DataArray reduction (`:1401–1405`), and Dataset reduction assembly (`:1499`). | Only the `Coarsen.__init__` preflight loop (`:1175–1177`) in rolling; the entire added `Index.check_coarsen` method in `xarray/core/indexes.py:244`. |

Test hunk ownership is also explicit:

- Patch 4: the two added index assertions in `test_broadcast_with_two_dimension_index` and
  `test_broadcast_preserves_common_indexes` belong to the generic fix. All signature edits,
  `AttributeCheckingIndex`, equal-label replacement/operations, right-join replacement,
  coordinate-attribute and override tests belong to B. Split the combined added-test hunk
  after `test_broadcast_preserves_common_indexes` (`xarray/tests/test_index_hooks.py:195`).
- Patch 9: `JointIndex`, `custom_index_obj`, `test_coarsen_custom_index`,
  `test_coarsen_unindexed_coordinate`, `test_coarsen_untouched_multiindex` are generic.
  `CoarsenGuardIndex` and both `test_coarsen_custom_index_preflight*` tests belong to A
  (`xarray/tests/test_coarsen.py:19–138`).

The earlier proposed patch-4 "fix half" was too broad: **joined-index propagation through
alignment needs B**. `_reindex_callback` reads `aligner.shadowed_keys`, introduced by patch 3
and populated only after an accepted overlapping join (`xarray/structure/alignment.py:584`).
That hunk alone on `b2acc77a` accesses a missing attribute on the no-new-name fast path;
without overlap dispatch, conflicting subset/superset groups already fail alignment. Keep it
in B. Broadcast without competing indexed subsets is independently fixable.

Standalone xarray-only regressions to implement and execute on both `b2acc77a` and each fix
alone (source-derived expectations below, **not runtime results from this design pass**):

- **4-fix:** use a small two-coordinate `xr.Index` with ordinary index methods and no new
  hooks, on `(y, x)`; broadcast against a coordinate-free `channel` array. Assert both outputs
  share one custom index for `x`/`y`, with unchanged labels/data; cover DataArray and Dataset
  `broadcast_like` too. Base reconstruction loses that ownership; the broadcast hunks pass
  indexes explicitly. Adapt `test_broadcast_preserves_common_indexes`; do not use
  `NoOverlapHookIndex`, whose class body references a hook absent on the base. This does not
  claim mixed binding/plain-index broadcast works without B.
- **9-fix:** use the hook-free `JointIndex` over `x` and `label(y)` on `(time, x, y)` data;
  coarsen `time=2` with mean/reduce and construct on both DataArray and Dataset. Assert the
  exact original shared index survives, coordinates remain equal, and reduced/reshaped data
  are correct. Base reconstructs coordinates without retaining that joint index; the fix
  carries it through `Coordinates`. Also retain affected-index whole-drop, unindexed-leftover
  and untouched-MultiIndex checks from the generic test hunks. No veto is needed for this fix.

### Rebuild plan (future series 5)

Build `index-hooks-5` on **`b2acc77a`**, not on series 4 or atop `series5-port`. Preserve both
existing trees. The following are planned commits, not changes made during R2a. In the table,
`P` expands to `PYTHONPATH=$PWD ~/GitHub/xarray-upstream/.venv/bin/pytest`; run it from the
new xarray tree using Python 3.12+.

| Order / commit | Files and behavior | Tests and validation command |
|---|---|---|
| 1. Patch 2 unchanged (`8f7d0159`) | `xarray/core/{dataarray,dataset,indexes}.py`, `xarray/tests/test_dataset.py`; scalar `drop=True` ownership and partial-drop refusal | Retain its tests; no hook-file changes yet. `P xarray/tests/test_dataset.py -k 'isel or sel' -q` |
| 2. Patch 7 unchanged (`071648c8`) | `xarray/structure/merge.py`, `xarray/tests/test_dataset.py`; reject incoming index type/coordinate-set replacement in update | Retain its tests; no hook-file changes yet. `P xarray/tests/test_dataset.py -k update -q` |
| 3. Patch 4 broadcast fix | `xarray/core/{dataarray,dataset}.py`, `xarray/structure/alignment.py`, `xarray/tests/test_index_hooks.py`; only the generic hunks above | Introduce hook-free fixture and broadcast regressions above. `P xarray/tests/test_index_hooks.py -k broadcast -q`; also run the same tests on base, requiring failure. |
| 4. Patch 9 coarsen fix | `xarray/computation/rolling.py`, `xarray/tests/test_coarsen.py`; only generic hunks/tests above | Keep preservation tests in `test_coarsen.py`; no new hook-file tests in this commit. `P xarray/tests/test_coarsen.py -k 'custom_index or unindexed_coordinate or untouched_multiindex' -q`; repeat on base, requiring regression failure. |
| 5. Hook A | `xarray/core/{indexes,dataset}.py`, `xarray/computation/rolling.py`, `xarray/tests/{test_index_hooks,test_dataset,test_coarsen}.py`, `doc/api/indexes.rst`; one no-op `check_drop` and three preflights | Move/rewrite old stack/pad/coarsen veto tests around `check_drop` in the hook file. Check operation/dim payload, once-per-shared-index timing, untouched indexes, default acceptance, stack ellipsis/`create_index=False`, zero pad width, and coarsen creation before mean/construct. `P xarray/tests/test_index_hooks.py xarray/tests/test_dataset.py xarray/tests/test_coarsen.py -k 'stack or pad or coarsen or check_drop' -q` |
| 6. Hook B including 7b | `xarray/core/{indexes,dataset}.py`, `xarray/structure/{alignment,merge}.py`, `xarray/tests/test_index_hooks.py`, `doc/api/indexes.rst`; patch 3 plus 4's dependent hunks and 7b, expressed by one method | Rewrite `GuardedIndex` checks as override/validation-only branches. Retain all overlap joins, operand orders, multi-object target refresh, exact failures, equal-label no-reindex replacement, attrs, declined/refused hooks and target-contract errors. Add direct `Coordinates.merge` compat cases (including override) that bypass alignment, unindexed value/unit validation, unchanged default behavior, full/partial overlap override checks, explicit-index exclusion, and 7b equal-label acceptance vs unequal-label/default/refusal for assignment and Dataset update. `P xarray/tests/test_index_hooks.py xarray/tests/test_alignment.py xarray/tests/test_merge.py -q`; `P xarray/tests/test_dataset.py -k update -q` |
| 7. Patch 5 last (`b019d23b`) | `xarray/core/{dataset,indexes}.py`, `xarray/tests/test_dataset.py`, `doc/api/indexes.rst`, `doc/whats-new.rst`; per-index `swap_dims`, separate from A/B | Retain swap tests; no new hook-file tests needed. `P xarray/tests/test_dataset.py -k swap_dims -q` |

Preserve PR-derived code in commits 1–2; recheck local patch IDs against available posted-PR
snapshots before rebuilding. Current remote PR heads were not checked in this offline task.
For new A/B documentation, list the two methods and their defaults in the existing Index API
page; do not retain documentation for the five removed methods.

Port conventions: retain PEP 695 class parameters (`xarray/structure/alignment.py:145`,
`xarray/computation/rolling.py:1114`); do not restore old `Generic[...]` class declarations.
The broadcast helper still uses legacy `T_Alignable`/`T_DataArray`/`T_Dataset` annotations
(`xarray/structure/alignment.py:1299–1324`), also present on `b2acc77a`; this is not a new port
regression or grounds for a broad typing cleanup. No new PEP 695 conflict was found in the
inspected hunks. New/rewritten methods and tests need types.
The read-only Ruff check of the seven inspected files found four errors: B035 at
`xarray/structure/merge.py:298` (the last-candidate map), D205 at `:1206` (7b's multiline
summary lacks a blank line), I001 at `xarray/tests/test_index_hooks.py:3`, and SIM114 at
`:98`. Fix docstring/import/branch style in B; resolve the B035 behavior decision with the
maintainer. The configured NumPy docstrings and D212/D402 rules are at `pyproject.toml:270`.
Validate each changed Python file with `ruff check --no-cache <files>` and
`ruff format --check --no-cache <files>` in addition to its tests.

Matching xarrayrf work, after the xarray commits: in `src/xarrayrf/_binding.py`, remove
`check_stack`, `check_pad`, `check_coarsen`, `check_override`, and
`check_unindexed_coord_conflicts`; add `check_drop` with the existing operation-specific
refusals. Widen `join_overlapping` (`:288`) to handle override first (same unconditional
refusal as `:349`), then `how=None` (the value/unit checks from `:333`, return `(self, {})`),
then its existing indexed joins. Keep `swap_dims` and public xarrayrf APIs unchanged.
Update capability detection and strict-xfail reasons in `tests/test_native.py:928`, `:950`,
`:969`, `:982` and `tests/test_geotiff.py:300` together; method presence for the old vetoes
must not misclassify the consolidated lane. Existing overlap detection can stay.

Validation, recorded in [xarray_patches.md](xarray_patches.md):

```sh
# xarray, in the new worktree
PYTHONPATH=$PWD ~/GitHub/xarray-upstream/.venv/bin/pytest xarray/tests/test_index_hooks.py \
  xarray/tests/test_dataset.py xarray/tests/test_indexes.py xarray/tests/test_coarsen.py -q
PYTHONPATH=$PWD ~/GitHub/xarray-upstream/.venv/bin/pytest xarray/tests -n 6 -q
# xarrayrf against the new tree
PYTHONPATH=$HOME/GitHub/<new-worktree> uv run --no-sync pytest
PYTHONPATH=$HOME/GitHub/<new-worktree> uv run --no-sync \
  python tools/binding_operation_probe.py
make check
```

Gate: 93/93 `ok` on the patched lane; stock and upstream strict xfails still xfail. Publishing
(push, tag `xarrayrf-patches-5`, re-pin, pretend version, refresh `.venv-patched`) needs the
maintainer's authorization.

Compare skips/xfails/xpasses with series 4 as R2 requires. No pytest suites or standalone
split experiments were run in R2a: only this note may change, and xarray's autouse fixture
creates temporary directories (`conftest.py:147`). The tests above are rebuild gates, not
claimed passes. Source/hunk inspection and a no-cache Ruff check were read-only.

### Cross-extension evidence for the design issue

Maintainers judge a new `Index` method by whether extensions they know need it. CRS demand
(#2288) was met by custom indexes (xproj `CRSIndex`); per-variable coordinates (#9152) was
closed in favour of cf-xarray. A proposal backed only by xarrayrf is expected to stall on "who
else uses this". Target: at least two extension-backed reproducers per hook, or a note stating
that fewer exist and what that means.

Observations already verified in
[prior_art_coverage_study.md](../architecture/prior_art_coverage_study.md) and reproduced by
`tools/geo_failure_modes.py`. The hook column is a hypothesis per row:

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

Done 2026-10-24: series 5 (`xarrayrf-patches-5`) was built in this order, with two
additions from the R1 audit: `interp` is a fourth `check_drop` call site (maintainer
decision), and broadcast keeps an index off outputs that lack an excluded dimension while
keeping their surviving labels (commit 8). The hook-free regressions fail on the unfixed
tree; `BindingIndex` implements only the two hooks. See the
[manifest](xarray_patches.md#series-5-current-published).

1. Maintainer settles the public names and the validation-only contract (`how=None`,
   `how="override"` with ignored returns), and whether the exact/no-copy fast path and the
   multiple-unindexed-candidate map get a separate fix; keep #11607's warning policy separate.
2. Hand the broadcast fix (commit 3 with 8 squashed in) and the coarsen fix (commit 4) to
   [upstream_prs.md](upstream_prs.md). Joined-index propagation stays with hook B.
3. Turn the six verified observations into reproducers with code paths and hook
   classification; add pint-xarray and cf-xarray reproducers and tracker searches.
4. Write an evidence note (`docs/dev/extension_hook_survey.md`, one row per reproducer with
   paired pinned tests) and a design-issue draft for the author to rewrite and post: the two
   methods and defaults, a table of extensions and the failure each hook fixes, the xarray code
   paths, related threads, xarrayrf named once. Under 500 words, at most one dash.
5. List extension maintainers or threads where a co-sponsor might be sought, with evidence. The
   maintainer decides whether to approach anyone, and whether patch 5 goes to #8911 as a comment
   (recommended first) or a dependent PR.
