# Maintained xarray patches

**Status:** Active
**Last updated:** 2026-10-06
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

### Series 4 (current, published)

| Item | Value |
|---|---|
| Base | upstream `main` `dfd25c7252e71a47e789aa24b8a06dd911a56461` (includes merged #11613 `b335cea0` and #11615 `dfd25c72`) |
| Checkout | `~/GitHub/xarray-patched-4`, branch `index-hooks-4` (immutable; build changes on a new branch) |
| Tag | annotated `xarrayrf-patches-4` -> `5db75d53f5515fabb3ca5ffc8419eb9c982cf4a2` on the fork |
| `git describe --tags --match 'v*'` | `v2026.07.0-80-g5db75d53` |
| Pin | `patched` dependency group, `tag = "xarrayrf-patches-4"`; `SETUPTOOLS_SCM_PRETEND_VERSION = "2026.7.1.dev80+xarrayrf.patches.4"` (uv's git checkout lacks release tags) |
| Default dependency | stock `xarray>=2026.7.0`; `.venv` stays stock, `.venv-patched` holds the pin |

Patch numbers are stable across series. Patch 1 retired (#11615 is in the base); patch 8 retired
in series 3 (#11613 merged). Ordered manifest:

| # | Commit | Purpose | Upstream reference |
|---|---|---|---|
| 2 | `f3d6dd3f28b490c2a9375153495b26c2e5828737` | `isel`/`sel(drop=True)` drops index-owned scalar coordinates; raises on a partial drop | [PR #11617](upstream_prs.md) |
| 3 | `e84aa71bdaaf62a20147f8d50747484d3c839972` | optional `Index.join_overlapping`, `check_unindexed_coord_conflicts`, `check_override` hooks | local; see hook design |
| 4 | `edd934dc3e3d16fba05072bd55e8f08e17dc416c` | alignment hook refinements plus broadcast custom-index preservation; depends on 3 | local; fix half to be split out |
| 5 | `3634c8748161b7f1d2637b54b33da92753006b11` | `Dataset.swap_dims` decides per index via optional `Index.swap_dims` | local; relates to #8911, #11100 |
| 6 | `ee3dd5592d9ed1399142b00338ead87b31f1d356` | Dataset reductions drop a multi-coordinate index whole | [PR #11616](upstream_prs.md) |
| 7 | `4b628738757b5e08445fce485952efbbd75a64f1` | `Dataset.update` raises when an incoming index of another type would replace the Dataset's | [PR #11621](upstream_prs.md) |
| 9 | `f0d55c1c7be753627a32710fbe84c80f49906d3e` | coarsen retains unaffected indexes; `Index.check_coarsen` veto | local; fix half to be split out |
| 10 | `c6fa89ce66256391327cc4c4c6e2c8732a046f4e` | `Index.check_stack` and `Index.check_pad` vetoes | local |
| 7b | `5db75d53f5515fabb3ca5ffc8419eb9c982cf4a2` | before #11621's raise, `update` asks the Dataset's index through `join_overlapping(how="left")`; if it accepts and labels are equal, the incoming indexes are dropped. Default hook returns `None`, so behavior equals #11621. Depends on 3 and 7 | local; stays after #11621 merges |

Retained patches other than refreshed patch 6 have the same patch IDs as series 3 (a negative
control confirmed different patches do not share IDs). PR-derived patches match the revisions
used to publish this immutable series, without their whats-new bullets. Current merge/review
state and newer remote heads live in [upstream_prs.md](upstream_prs.md); refresh changed PR
patches in a new series rather than modifying the published tag.

#### Series 4 validation (2026-09-28)

| Lane | Result |
|---|---|
| xarray full suite, base `dfd25c72` | 19,879 passed, 1,097 skipped, 187 xfailed, 31 xpassed |
| xarray full suite, series 4 | 20,077 passed, same skip/xfail/xpass, 0 failed |
| xarray focused patched tests | 971 passed, 7 skipped, 1 xfailed, 1 xpassed |
| xarrayrf on upstream `main` | 1,044 passed, 1 skipped, 49 xfailed; probe 64 ok / 29 holes |
| xarrayrf on series 4 (path and installed tag) | 1,083 passed, 1 skipped, 10 xfailed; probe 93 ok / 0 holes |
| xarrayrf on stock 2026.7.0 | 1,040 passed, 54 xfailed; probe 67 ok / 26 holes |
| `make check`, `make build` | pass |

Upstream `broadcast` now returns an unframed result (a contract hole); series 4 preserves the
binding. When `PYTHONPATH` overrides an editable install, xarray's reported version can be stale;
identify the tree by `xarray.__file__`.

Commands:

```sh
# xarray, from inside the tree (PYTHONPATH=$PWD is required)
PYTHONPATH=$PWD ~/GitHub/xarray-upstream/.venv/bin/pytest xarray/tests -n 6 -q
# xarrayrf lanes
make test            # stock .venv
make test-upstream   # ~/GitHub/xarray-upstream main
make test-patched    # ~/GitHub/xarray-patched-4 by path
make test-pinned     # published tag in .venv-patched
make probe           # 93-case native-operation probe
make check
```

Pipe-redirected output needs an explicit exit-code check.

### Earlier series

Published tags retain reproducible history; old ordered manifests and validation transcripts
remain in Git history rather than competing with the current manifest.

| Series | Tag target | Historical base |
|---|---|---|
| `xarrayrf-patches-3` | `9ca40562c9f292d482ee653f1dc18e0ed11b9a4c` | `56a5d370` |
| `xarrayrf-patches-2` | `2af8cfea3d530b2e7f0c4d6c805bea8c5fbfc509` | `8de862c29544c2fb1957ce20985745952bc81de7` |
| `xarrayrf-patches-1` | `35f13b39` | See the tag's history |

The original patch 7 kept Dataset indexes through internal `merge_core` priority overrides.
Upstream review replaced that with a raise; patch 7b retains opt-in keep behavior through the
hook. This distinction matters when refreshing the update patches.

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

Upstream status: issue #11607 (links #2996), no PR, pending maintainer feedback.

## Alternatives Considered

- Runtime monkey patches or edits inside site-packages: rejected; provenance and review become
  unreliable. Fixes live in their owning xarray tree with regression tests.
- Mutating a published series: rejected; each series revision gets a new branch, immutable tag,
  matching pin and pretend version.

## Deferred Work

- A default-on fix for #11607 (lazy comparison without store reads, `merge(compat="override")`
  aligning first, error precedence) waits for maintainer feedback. The opt-in hook catches
  unindexed conflicts for `BindingIndex`.
- No hosted CI lane runs the patched pin yet.
- Retirement: re-evaluate each patch once a released xarray contains the fix and xarrayrf's
  contract tests pass on that release; only then raise the stock floor.

## Next Steps

1. When an open PR changes in review, refresh the matching patch from the PR head in the next
   numbered series.
2. Rebase a new series onto merged fixes recorded in [upstream_prs.md](upstream_prs.md),
   retire duplicate patches (keep 7b), tag, re-pin and revalidate. The current series remains
   unchanged until that separate upstream-development task is undertaken.
3. Split the fix halves out of patches 4 and 9, and split patch 10 into stack and pad, before any
   upstream PR (see [index_hook_design.md](index_hook_design.md)).
4. Search the tracker before any further xarray change.
