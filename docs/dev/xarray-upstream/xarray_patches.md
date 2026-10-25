# Maintained xarray patches

**Status:** Active
**Last updated:** 2026-10-24
**Scope:** The local xarray patch series that implements xarrayrf's native-operation contract,
its pin in `pyproject.toml`, and its validation.

## Context

Some lifecycle behavior xarrayrf needs is missing or wrong in xarray. The project keeps those
fixes in a separate xarray checkout, organized for upstream submission; see
[the workflow](../../upstream.md). Upstream pull request status is in
[upstream_prs.md](upstream_prs.md); the planned hook redesign is in
[index_hook_design.md](index_hook_design.md).

Repositories: upstream `https://github.com/pydata/xarray`; public fork
`https://github.com/matthiasschabel/xarray` (remote `fork`). In `~/GitHub/xarray-upstream` the
`origin` push URL is `DISABLED`; push only to `fork`, and only with authorization.

## Current Decision

### Series 5 (current, published)

| Item | Value |
|---|---|
| Base | upstream `main` `b2acc77ab2ca4a0caedce527c8cce2e0d9737c2c` (includes merged #11613, #11615, #11616) |
| Checkout | `~/GitHub/xarray-hooks-5`, branch `index-hooks-5` (immutable; build changes on a new branch) |
| Tag | annotated `xarrayrf-patches-5` -> `a657617942f5c6f6b6bba88d2aeb2f10691af386` on the fork |
| `git describe --tags --match 'v*'` | `v2026.09.0-146-ga6576179` |
| Pin | `patched` dependency group, `tag = "xarrayrf-patches-5"`; `SETUPTOOLS_SCM_PRETEND_VERSION = "2026.9.1.dev146+xarrayrf.patches.5"` |
| Default dependency | stock `xarray>=2026.7.0`; `.venv` stays stock, `.venv-patched` holds the pin |

Series 5 replaces series 4's seven optional `Index` hooks with two (`check_drop`,
`join_overlapping`), following the [hook design](index_hook_design.md), and splits the generic
fixes out of the old patches 4 and 9. Patch 6 retired (#11616 is in the base). Ordered
manifest:

| # | Commit | Purpose | Upstream reference |
|---|---|---|---|
| 1 | `26d1e1159ca6697dc0df5205a1497c6a8419d0b4` | `isel`/`sel(drop=True)` drops index-owned scalar coordinates; raises on a partial drop (old patch 2) | [PR #11617](upstream_prs.md); code identical apart from a merged import line, since `main` moved |
| 2 | `9b1bf4ea236ddf4364a9bfebbede56f894570099` | `Dataset.update` raises when an incoming index of another type would replace the Dataset's (old patch 7) | [PR #11621](upstream_prs.md); same patch ID as head `e52d7fa4` |
| 3 | `67b0647b905063f3ad3e1fda25fc2abee4d3dea6` | broadcast and `broadcast_like` keep common indexes; an index on an excluded dimension stays off outputs lacking it (generic half of old patch 4) | local; candidate fix PR |
| 4 | `49c93abdedfdb29681e2cbf51cd68206de84f2aa` | coarsen retains unaffected indexes (generic half of old patch 9) | local; candidate fix PR |
| 5 | `87732447217dd480b18d9873420fcafd76fc1a29` | `Index.check_drop(operation, dims)`, consulted before stack, pad, coarsen and interp drop an index (replaces `check_stack`, `check_pad`, `check_coarsen`) | local; hook A |
| 6 | `23874f092c53e147cbffb529f785b4ce81d8624e` | `Index.join_overlapping(..., how)` for overlapping joins, `how="override"` vetoes and `how=None` unindexed validation, plus update acceptance (replaces patch 3, the hook half of 4, and 7b) | local; hook B |
| 7 | `36959c6a218a14a69a5fab299a6e51c15247cfad` | `Dataset.swap_dims` decides per index via optional `Index.swap_dims` (old patch 5) | local; relates to #8911, #11100 |
| 8 | `a657617942f5c6f6b6bba88d2aeb2f10691af386` | broadcast keeps a joint index's surviving labels, with default indexes, when an excluded dimension stops the index itself | local; squash into 3 before any upstream PR |

#### Series 5 validation (2026-10-24)

| Lane | Result |
|---|---|
| xarray full suite, base `b2acc77a` | 20,929 passed, 1,111 skipped, 187 xfailed, 31 xpassed |
| xarray full suite, series 5 | 21,273 passed, same skip/xfail/xpass, 0 failed |
| every series-5 commit | ruff check clean; `test_index_hooks.py` and `test_coarsen.py` pass |
| new commit-3/4 regressions on the unfixed tree | fail, as required |
| xarrayrf on series 5 (path) | 3,108 passed; probe 128/128 `ok` |
| xarrayrf on upstream `b2acc77a` | 3,046 passed, 62 xfailed; probe 85 ok / 43 holes |
| xarrayrf on stock 2026.7.0 | 3,039 passed, 69 xfailed; probe 74 ok / 54 holes |
| `make check` | pass |

Series 5 drops series 4's hook names, so series 4 is no longer a supported lane for
`BindingIndex`. Remaining maintainer decisions on the hook contract are in the
[hook design](index_hook_design.md) Next Steps.

Commands:

```sh
# xarray, from inside the tree (PYTHONPATH=$PWD is required)
PYTHONPATH=$PWD ~/GitHub/xarray-upstream/.venv/bin/pytest xarray/tests -n 6 -q
# xarrayrf lanes
make test            # stock .venv
make test-upstream   # ~/GitHub/xarray-upstream (set XARRAY_UPSTREAM to a main checkout)
make test-patched    # ~/GitHub/xarray-hooks-5 by path
make test-pinned     # published tag in .venv-patched
make probe-pinned    # probe plus gate against the published tag
make check
```

Pipe-redirected output needs an explicit exit-code check.

### Earlier series

Published tags retain reproducible history; old ordered manifests and validation transcripts
remain in Git history rather than competing with the current manifest.

| Series | Tag target | Historical base |
|---|---|---|
| `xarrayrf-patches-4` | `5db75d53f5515fabb3ca5ffc8419eb9c982cf4a2` | `dfd25c72` |
| `xarrayrf-patches-3` | `9ca40562c9f292d482ee653f1dc18e0ed11b9a4c` | `56a5d370` |
| `xarrayrf-patches-2` | `2af8cfea3d530b2e7f0c4d6c805bea8c5fbfc509` | `8de862c29544c2fb1957ce20985745952bc81de7` |
| `xarrayrf-patches-1` | `35f13b39` | See the tag's history |

The original patch 7 kept Dataset indexes through internal `merge_core` priority overrides.
Upstream review replaced that with a raise; series 4's patch 7b, now part of series 5's
`join_overlapping` commit, retains opt-in keep behavior through the hook. This distinction
matters when refreshing the update patches.

### Tracker searches recorded for local patches

- Hooks and broadcast (patches 1, 3, 4): #11532 (merged, in base) covers basic mixed alignment;
  #11607 records indexed-versus-unindexed coordinate loss; #6481 (broadcast refactor) related.
- `swap_dims` (patch 5, 2026-09-25): #8914, #11099, open PR #11100 (subsumed), draft #8911
  (maintainers do not want custom indexes recreated; compatible with an index keeping itself).
- Reductions and `update` (patches 6, 7, 2026-09-25): #11215, merged #11286 (DataArray half).
- Roll, stack, pad, coarsen (2026-09-25): searches for `roll empty`, `modulo by zero`,
  `stack custom index`, `stack index dropped`, `pad custom index`, `pad index dropped`,
  `coarsen custom index`, `coarsen index lost` found no report of the same root cause.

### Measurement prototype: indexed vs unindexed coordinate value check

Not a patch xarrayrf uses. Archived fork branches measuring how much of xarray relies on an
indexed coordinate silently winning over a same-named unindexed one (#2996, same root cause).
Symptom: custom scalar index `tag=10` vs plain `tag=20` keeps 10 silently through `a + b`,
`b + a`, `merge` and `align(exact)`, while merging two plain conflicting coordinates raises
`MergeError`.

- `prototype/index-value-check` (`b7fc47f8`): check in `merge_collected`.
- `prototype/index-value-check-v2` (`c73ff48d`): adds the check in
  `Aligner._get_indexes_and_vars`, both restricted to same-dimension variables.
- Switch `XARRAY_PROTOTYPE_MERGE_CHECK=raise|warn`; baseline 19,689 passed of 21,291 collected.

| Variant | Existing tests failing | What they rely on |
|---|---|---|
| `merge_collected` only | 5 | misses the scalar conflict: alignment already overwrote the plain coordinate |
| merge + alignment, any dims | 7 | selection leftovers (`orig + orig[0, 0]`), `Coordinates.merge` with scalar `x=nan`, dask `dot`, stacked round trip, zarr `test_region_write`, two error-precedence tests |
| merge + alignment, same dims only | 2 | zarr `test_region_write` (lazy coordinate read: 6 store reads, 5 allowed); `test_merge_multiindex_level` (error precedence) |

Upstream status: issue #11607 (links #2996). A maintainer asked for a PR (2026-10-12) and
suggested a deprecation first. Branch `pr/index-coord-conflict-warning` in `xarray-upstream`
(not yet pushed) emits a `FutureWarning` in `Aligner._get_indexes_and_vars`, where the index is
assigned. It does not cover the `join="exact", copy=False` fast path (`apply_ufunc`, `where`,
`merge` under the new combine defaults), which is left as an open question for maintainers.

## Alternatives Considered

- Runtime monkey patches or edits inside site-packages: rejected; provenance and review become
  unreliable. Fixes live in their owning xarray tree with regression tests.
- Mutating a published series: rejected; each series revision gets a new branch, immutable tag,
  matching pin and pretend version.

## Deferred Work

- #11607: the warning PR leaves the exact/no-copy path (`merge_collected`'s `TODO(shoyer)`
  branch) and the eventual raise for follow-ups, pending maintainer review. The opt-in hook catches
  unindexed conflicts for `BindingIndex`.
- The `patched` CI job validates the pinned series with the probe gate on every push.
- Retirement: re-evaluate each patch once a released xarray contains the fix and xarrayrf's
  contract tests pass on that release; only then raise the stock floor.

## Next Steps

1. When an open PR changes in review, refresh the matching patch from the PR head in the next
   numbered series.
2. When fixes merge upstream, build the next series on a base containing them, retire the
   duplicate commits (keep the update acceptance in `join_overlapping`), tag, re-pin and
   revalidate. Series 5 stays unchanged.
3. Before a fix PR for commit 3, squash commit 8 into it; commits 3 and 4 are the
   standalone fix candidates (see [index_hook_design.md](index_hook_design.md)).
4. Search the tracker before any further xarray change.
