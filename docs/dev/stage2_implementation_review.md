# Stage 2 implementation and refinement record

**Status:** Implemented
**Last updated:** 2026-10-05
**Scope:** F6 scalar resampling targets, F7 labelled point-output naming and M1 Geometry docs

## Context

The user requested Claude Opus 5.5 plan agreement, implementation by gpt-6-astra, then
Opus implementation review through collaborative-refinement. The base is
`f98299f40ee63d483c8dbf259138715f0286d7a0`. Reviewer commands explicitly select
`claude-opus-5-5`; the implementation handoff explicitly selects `gpt-6-astra` and uses a
dedicated detached worktree. The implementation-review phase made no commits or remote
publication. The subsequent QAEngineer invocation authorized committing clean stage 2 work
to main. The unrelated user documentation edits are excluded and preserved.

The CLI reviewers retain fixed read-only controls. Runner escalation enables network and
macOS Keychain access, so read-only review relies on application controls rather than the
enclosing OS sandbox. Implementation uses Codex's workspace-write sandbox in its worktree.

## Current Decision

Stage 2 is implemented, reviewed, independently QA-verified and committed to main.
Opus accepted both the plan and implementation after two passes each. All implementation
findings are resolved, with no new findings or open disagreements.

The fixes are independently committed as `74e0021` (scalar resampling) and `f4a7d6a`
(point output naming and Geometry documentation). Shared public documentation, QA and review
records accompany them in a separate commit.

Opus accepted the plan after two passes. Preserve affine crop-limited reads, promote scalar
output only within the affine kernel, construct explicit zero-row position arrays in the
general path, and reshape with tuples. Add `axis_dim` and `units_coord` to dense `points`,
remove its temporary stacking dimension, and preserve `point_at`'s keyword indexers.

### Plan finding dispositions

| ID | Disposition | Decision |
|---|---|---|
| N1 | Accepted | Name the scalar outside-mask crash and cover cells/cubic masking under promotion. |
| N2 | Accepted | Use a one-sample affine computational shape; direct 3-D SciPy probe succeeds. |
| N3 | Accepted | Stack scalar-core input columns inside the evaluation callback; remove private dimension. |
| N4 | Revised | Keep `point_at` fixed names; new keywords would take valid indexer names. Document xarray rename for custom-name alignment. |
| N5 | Accepted; factual claim withdrawn | Check only carried dimensions/index coordinates. xarray permits the unindexed units layout the first report said it refused. |
| N6 | Rejected as unnecessary bookkeeping | Removing the corner reshape needs no scalar branch; planner promotion would require separate public/computational shapes for strict output-size construction. Reviewer leaves the simpler choice to implementer. |
| N7 | Accepted | Assert scalar Dask chunks and parity with eager values. |

## Alternatives Considered

The original dedicated scalar point-interpolation strategy was replaced by local affine
promotion, reusing existing masking and cubic shell logic. Global planner promotion would
thread a second shape through output construction. `point_at` API expansion and automatic
public-name suffixes were rejected. Output naming remains explicit and defaults stay stable.

## Deferred Work

Audit stages 3–5 remain pending. Scalar-source inversion/projection and interpolation NaN
policies remain outside stage 2.

## Next Steps

Stage 3 is the next independent audit task. Stage 2 commits preserve implementation attribution
to Codex gpt-6-astra and the required `Reviewed-By: Claude (claude-opus-5-5)` trailer.
No remote publication is authorized or performed.
The isolated worktree was removed after byte-for-byte integration and successful final checks.

## Open Disagreements

No blocking disagreement. N6 is an optional implementation simplification; the reviewer
accepted either choice based on the resulting complexity.

## Plan Review Reports

The following are verbatim reviewer reports, retained as quotations.

### Initial plan review

> ## Artifact Summary
> The plan covers three fixes: F6 makes scalar (rank-0) resampling targets work, F7 adds `axis_dim`/`units_coord` options so `Geometry.points` output names stop colliding with input names, and M1 corrects a stale docstring. I checked the failure claims against the source and they hold:
>
> - **`_crop_window` failure:** `np.array([()]).T` has shape (0,1), and `reshape(0,-1)` on it is ambiguous (`_resample.py:151-153`).
> - **Block reshapes:** `reshape(*(), *())` is called with no arguments (`:484,518,548`).
> - **General-map position:** `_block_positions` calls `unravel_index` with shape `()` (`:461`).
> - **F7 fixed names:** `AXIS_DIM`, `_SOURCE_AXIS` and the literal `"units"` are hard-coded (`_geometry.py:35,38,456-477`).
> - **M1 stale text:** the docstring at `_geometry.py:168-169` still calls `.rf.geometry` "future" work.
>
> I found nothing blocking, but the scalar path has a crash site the plan doesn't name (N1), and there are two simplifications worth weighing (N2, N3).
>
> ## Blocking Findings
> None.
>
> ## Non-Blocking Findings
>
> ### N1: `_LatticeMap.outside(())` is a third scalar crash site the plan doesn't name
> - Confidence: High
> - Evidence: `_resample.py:86` runs `range(0, target_shape[0], …)`, which raises IndexError when `target_shape == ()`. `_lattice_block` calls `outside()` unconditionally (`:477`), and calls it again for the cubic shell (`:488`). The plan says scalar affine handling must "obey the same domain bounds", but it doesn't say whether the new point strategy bypasses `outside()` or fixes it.
> - Suggested change: Name it explicitly. Either the point strategy masks with `lowest`/`highest` directly on `offset` (the single source position), or `outside()` handles `()`. The regression test should cover a scalar point inside the cell but outside the samples, with cubic interpolation.
> - Would change my mind: If the planned point strategy never reaches `_lattice_block` or `outside()`, the only gap is that the plan doesn't say so.
>
> ### N2: Consider promoting the scalar target to a singleton dimension internally instead of adding a new strategy
> - Confidence: Medium. This is a design alternative; I haven't prototyped it.
> - Evidence: `lattice_parts(..., single_samples=True)` already accepts single-sample dimensions (`:179-181`). With an internal target shape of (1,) plus one zero column in `linear`, every existing path works unchanged:
>   - the crop corners,
>   - `outside()`,
>   - `affine_transform` with `output_shape=(1,)`,
>   - the cubic shell.
>
>   `_same_grid_indices` would correctly return None, because the zero column fails `count_nonzero == 1` (`:260`). Only the final reshape and the `apply_ufunc` output dimensions would use `()`. The cost is the general map: `sample_columns` indexes by `target_dims` (`_positions.py:194`), so its position rows would need the phantom row dropped, or `(0,n)` built as the plan already proposes.
> - Suggested change: Prefer whichever approach adds fewer branches, and record why in the plan. Promotion probably wins for the lattice path; the plan's explicit `(0,n)` construction is fine for the general path.
> - Would change my mind: A prototype showing promotion needs more special cases than a dedicated point path.
>
> ### N3: F7 can remove the private stacking dimension instead of choosing a collision-proof name
> - Confidence: Medium-High
> - Evidence: `_SOURCE_AXIS` exists only to stack the coordinates with `xr.concat` and then rechunk to -1 (`_geometry.py:456-458`). If each broadcast coordinate is passed to `apply_ufunc` as its own argument with `input_core_dims=[[]]*n`, `evaluate(*cols)` can `np.stack(cols, -1)` itself. That deletes the private dimension, removes one class of collision entirely, and drops the rechunk step. Dask laziness is kept because the call is still `dask="parallelized"`.
> - Suggested change: Replace "choose a private stacking dimension that cannot collide" with "eliminate the stacking dimension".
> - Would change my mind: If `apply_ufunc` with n scalar-core inputs measurably changes the Dask graph or chunk alignment, for example broadcasting coordinates differently from what `xr.broadcast` produces today.
>
> ### N4: The F7 naming contract leaves `Geometry.point_at` asymmetric
> - Confidence: High (fact); the right resolution is a judgment call
> - Evidence: `point_at` returns a DataArray with the same hard-coded `"axis"`/`"units"` names (`_geometry.py:418-422`). It can't collide, because its output has no geometry dimensions. But after F7, `points(axis_dim=…)` and `point_at()` would label the same quantity differently with no way to align them.
> - Suggested change: Either accept the same two keywords on `point_at` (cheap, and it keeps the mental model uniform), or state in the docs that only `points` takes them and why. This is for the user to decide. On the question asked: yes, `axis_dim` + `units_coord` is the simplest predictable contract. A single `prefix=` argument would be less explicit, and auto-renaming would hide the problem.
> - Would change my mind: The user preferring that `point_at` stay fixed.
>
> ### N5: Narrow the "units" collision rule
> - Confidence: Medium
> - Evidence: A geometry dimension named `"units"` without an index still produces dimension `units` alongside a coordinate `units` on dimension `axis`. xarray rejects that combination, so the check must cover dimension names as well as index coordinates. The plan says "dimensions/index coordinates", which covers this. Non-index, non-geometry coordinates are not carried (`:472-474`), so there's no need to refuse those names. Keep the check that narrow so it doesn't over-reject.
> - Would change my mind: If carried coordinates are planned to expand beyond index coordinates.
>
> ## Verification Commands
> None (no blockers). For N1, the crash can be confirmed with:
> `python -c "from xarrayrf._resample import _LatticeMap; import numpy as np; _LatticeMap(np.zeros((1,0)),np.zeros(1),np.zeros(1),np.ones(1)).outside(())"` (expect IndexError).
>
> ## Recommendation
> Accept, after small plan edits: name the `outside()` site (N1), decide between promotion and a dedicated point path (N2), and adopt N3 unless there's a reason against it. N4 is a user decision.

### Plan convergence

> ## Convergence Assessment
> | ID | Status | Rationale |
> |---|---|---|
> | N1 | Resolved | The plan now names the `outside(())` crash at `_resample.py:86` and calls `outside()` only with the promoted `(1,)` shape. That covers both calls in `_lattice_block`: the fill mask (`:477`) and the cubic shell (`:488`). The acceptance tests include cubic edge holding in the cells domain. |
> | N2 | Resolved | The plan adopts promotion to `(1,)` and backs it with a SciPy prototype. I checked that the integral-grid shortcut stays off for scalar targets. With the unpromoted map (shape `(n,0)`), the shape check at `_resample.py:253-256` fails. With the promoted map, the zero column fails `count_nonzero(...) == 1` (`:260`). Either way `same_grid` is None, so no new special case appears there. |
> | N3 | Resolved | The plan removes the private stacking dimension and passes each broadcast coordinate to `apply_ufunc` as its own scalar-core argument. Chunk alignment survives because the per-coordinate `.chunk(chunks)` step at `_geometry.py:453-455` already runs before stacking. Only the concat at `:456` and the rechunk at `:457-458` go away. |
> | N4 | Resolved | The designer's rationale holds against the source. `Geometry.point_at(self, /, **indexers)` (`_geometry.py:360`) accepts any dimension name as a keyword. Adding `axis_dim=`/`units_coord=` would take those two names away from users whose geometry dimensions have them, which is a real change to the existing contract. Documenting the asymmetry and pointing to `.rename` is the cheaper and safer choice. |
> | N5 | Resolved (partly withdrawn) | I withdraw my claim that xarray rejects an unindexed `units` dimension alongside a `units` coordinate on `axis`. I never ran it, and the designer reports the opposite. The planned contract does not depend on who is right: it still refuses collisions with carried dimensions and index coordinates, and it no longer over-rejects omitted coordinates. To settle the fact: `python -c "import xarray as xr; xr.DataArray([[0]], dims=('units','axis'), coords={'axis':['x'],'units':('axis',['mm'])})"` |
>
> ## New Findings
>
> ### N6: The scalar case is handled in two places, which could be one (non-blocking, simplification)
> - Confidence: Medium. This is a design judgment; I have not prototyped it.
> - Evidence: The plan keeps the public shape `()` for crop planning and gives it a special `(0,1)` corner construction (`_resample.py:151-153`). It then promotes to `(1,)` inside the kernel only. That makes two separate places that each have to know about rank-0 targets.
> - Alternative: Build the promoted lattice map, with `linear` of shape `(n,1)` and an internal target shape of `(1,)`, once when the plan is made. Keep the public `()` only for the output reshape and the `apply_ufunc` dimensions. The existing `_crop_window` corner code then works unchanged: product over `(0, 0)` gives one corner, and `linear @ [[0]] + offset` gives the single source position. The crop-limited read is preserved with no `(0,1)` branch.
> - Would change my mind: If `plan.target_shape` is read by code outside the kernel that must see `()` (for example, output chunk or dimension construction), separate internal and public shape fields would be needed. That may cost about as much as the current split.
>
> ### N7: Specify what a scalar target means for Dask chunking (minor, missing requirement)
> - Confidence: Low to medium. I did not verify this against the source.
> - Evidence: The acceptance list says Dask scalar targets should be tested and that no pixel tasks should run while the graph is built. It does not say what output chunking is expected when the target has no dimensions. If `output_sizes`/chunk construction loops over target dimensions, an empty loop needs to produce a single chunk.
> - Suggested change: Add one assertion that a Dask scalar result has the expected `chunks` and computes to the eager value. This is one line in the existing test.
> - Would confirm: The planned Dask scalar test itself.
>
> ## Open Disagreements
> | ID | Designer / Implementor Position | Critic / Reviewer Position | Human Decision Needed? |
> |---|---|---|---|
> | N6 | Promote to `(1,)` inside the kernel only; special-case the scalar crop corners | Promote once at plan time so the crop code needs no special case | No. The implementor can choose whichever has fewer branches. |
>
> ## Recommendation
> Accept. Every prior finding is resolved or withdrawn, and N6/N7 are optional refinements. Nothing high-impact remains, so no human escalation is needed.


## Implementation Handoff and First Review

The CLI header confirms `gpt-6-astra` with high reasoning effort. It changed seven files in the
isolated worktree. Opus accepted the source-grounded implementation review with no blockers;
the orchestrator accepted all three optional follow-ups for the implementer to apply.

| Implementation finding | Disposition | Change |
|---|---|---|
| N1 | Accepted | Remove the vacuous crop-read bounds assertion; retain one-chunk/no-read guards. |
| N2 | Accepted | Use AXIS_DIM as the default; no new naming abstraction. |
| N3 | Accepted | Add direct eager/Dask scalar points coverage with custom names. |

The independent orchestrator focused suite passed 295 tests with 41 expected stock failures.
Six additional probes passed: descending coordinates/reflection, integral nearest dtype,
complex linear/cubic values, affine/general maps and framed scalar output.

### gpt-6-astra implementation report

> ## Implementation Handoff
>
> - Implementer: codex
> - Worktree: /private/tmp/xarrayrf-stage2-astra-20261005
> - Base: f98299f40ee63d483c8dbf259138715f0286d7a0
> - Local review command: `git -C /private/tmp/xarrayrf-stage2-astra-20261005 diff f98299f40ee63d483c8dbf259138715f0286d7a0`
>
> ## Actual Changed Files
>
> ```text
> CHANGELOG.md
> docs/core_interface.md
> src/xarrayrf/_geometry.py
> src/xarrayrf/_resample.py
> tests/test_geometry.py
> tests/test_native.py
> tests/test_resample.py
> ```
>
> ## Diff Stat
>
> ```text
> CHANGELOG.md              |   9 ++++
>  docs/core_interface.md    |  22 ++++++++--
>  src/xarrayrf/_geometry.py |  63 +++++++++++++++++----------
>  src/xarrayrf/_resample.py |  40 +++++++++++------
>  tests/test_geometry.py    | 107 ++++++++++++++++++++++++++++++++++++++++++++--
>  tests/test_native.py      |  66 +++++++++++++++++++++++++++-
>  tests/test_resample.py    |  63 +++++++++++++++++++++++++++
>  7 files changed, 326 insertions(+), 44 deletions(-)
> ```
>
> ## Implementer Report
>
> ## Implementation Summary
>
> Implemented F6, F7 and M1 in the dedicated worktree.
>
> Scalar targets retain public shape `()`, using kernel-local promotion for affine interpolation. Cropping, cubic edge holding, Dask laziness, metadata and binding behavior are covered. `Geometry.points()` now accepts validated `axis_dim` and `units_coord` names and no longer uses a private stacking dimension.
>
> No commits, remote actions or source-checkout edits. Protected documents remain untouched; audit records remain with the orchestrator.
>
> ## Files Changed
>
> - `src/xarrayrf/_resample.py`: scalar crop planning, position mapping and kernel output handling.
> - `src/xarrayrf/_geometry.py`: configurable point-result names, collision validation and corrected lifecycle documentation.
> - `tests/test_resample.py`: scalar interpolation, Dask, cropped reads and scalar-source refusal regressions.
> - `tests/test_geometry.py`: naming collisions, validation and lazy nonseparable-coordinate regressions.
> - `tests/test_native.py`: scalar target binding, context, metadata and chunk preservation.
> - `docs/core_interface.md`: public behavior and naming documentation.
> - `CHANGELOG.md`: release notes.
>
> ## Tests Run
>
> All executables came from `/Users/matthiasschabel/GitHub/xarrayrf/.venv/bin`. Python commands used `PYTHONDONTWRITEBYTECODE=1` and worktree `src` first in `PYTHONPATH`; the patched lane additionally included `/Users/matthiasschabel/GitHub/xarray-patched-4`. Verified `xarrayrf.__file__` resolves inside this worktree.
>
> For the regression commands below, `K` was:
> `points_output_names or points_reject_invalid_output_names or points_custom_names or scalar_volume_target or scalar_affine_target or resample_to_scalar_target`
>
> - `pytest -q tests/test_geometry.py tests/test_resample.py tests/test_native.py -k "$K" --tb=line`: run twice before implementation; **54 failed each time**. The first run also exposed test-construction mistakes, corrected before the second.
> - Same selection with `--tb=short`, after implementation: **50 passed, 4 failed**. Remaining failures were test assumptions about cell declarations and Dask read boundaries; corrected.
> - `pytest -q tests/test_geometry.py tests/test_resample.py tests/test_native.py --tb=short`: **295 passed, 41 xfailed**.
> - Stock `pytest -q --tb=short`: **1,694 passed, 47 xfailed**, three dependency warnings.
> - Patched `pytest -q --tb=short -o cache_dir=.pytest_cache-patched`: **1,741 passed**, three dependency warnings.
> - `pytest -q tests/test_native.py -k test_resample_to_empty_target --tb=short`: **24 passed** in each lane, after restoring an accidentally altered existing assertion during final diff review. Patched command also used its separate cache directory.
> - `ruff format src/xarrayrf/_geometry.py src/xarrayrf/_resample.py tests/test_geometry.py tests/test_resample.py tests/test_native.py`: four files formatted.
> - `ruff check .`: initially two regex-literal findings; **passed after correction**.
> - `ruff format --check .`: **passed both runs**, 125 files.
> - `mypy`: initially one unused type-ignore; **passed after correction**, 81 files.
> - `git diff --check`: **passed**.
>
> All requested checks were run.
>
> ## Unresolved Decisions
>
> None.
>
> ## Suggested Review Focus
>
> - Scalar affine promotion through outside masking and cubic cells-shell handling.
> - General mapping and zero-length leading dimensions.
> - Collision validation restricted to carried geometry names.
> - Coordinate-only Dask evaluation and retained output chunking.

### Opus implementation review

> ## Artifact Summary
>
> I recommend accepting this. There are no blocking findings, only three small optional ones. I checked the diff against the code around it in `_resample.py` (`_LatticeMap.outside`, `_crop_window`, `_same_grid_indices`, `_plan`, `resample`), `_positions.py` (`sample_columns`, `transform_points`), `Geometry.__init__`/`_dependencies` and the native `rf.geometry`. I read the code but did not run the tests or SciPy myself.
>
> What checks out:
> - **Scalar crop** (`_resample.py:151`): with an empty `target_shape`, `product()` gives one empty corner. The resulting arrays have shapes (1,0), then (0,1), then `linear (n,0) @ (0,1)` gives (n,1) plus the offset. The crop window comes out correctly, and no `reshape(0,-1)` is needed.
> - **Same-grid path** (`:251`): for a scalar target, `linear.shape` is (n,0) and `len(source_shape) != 0`, so it returns `None` before the promotion. The scalar case can't reach the same-grid fast path with a mismatched shape.
> - **Promotion** (`:473-481`): the (n,1) zero matrix with output rank 1 matches SciPy's row/column rules. It also can't be mistaken for a homogeneous matrix, because that would need 1 == output rank + 1. `outside`, `beyond` and `shell` all use the promoted (1,) shape. `shell_positions = linear @ (1,k) + offset` gives the edge-held cubic positions. Every return reshapes to `plan.target_shape` with an explicit tuple.
> - **General path** (`:458-462`): a (0,k) index array works with `sample_columns` (`_positions.py:191-198`). It reads `count` from `indices.shape[1]` and repeats retained scalars.
> - **Dask wrapper** (`:673-688`): `total=1`, so `allow_rechunk` is True and `output_sizes` is empty. Context, name and attribute handling is unchanged.
> - **`points()` transpose** (`_geometry.py:470`): `transpose(*order)` is safe because `_dependencies` (`:309-314`) refuses any geometry dimension that has no source axis. Every broadcast coordinate therefore spans every dimension in `order`, and the old `if dim in broadcast[0].dims` filter is no longer needed.
> - **Name collisions**: carried index coordinates are only those named after a dimension (`:489-491`), so checking `name in order` is exact. Unrelated coordinates are correctly ignored.
> - **Lazy coordinate fields**: removing the private stacking dimension also helps an unchunked array whose coordinates are Dask arrays. The old concat could split the core dimension, which `dask="parallelized"` refuses. Apply_ufunc dimensions with no core axes have no such constraint.
> - **M1 documentation**: matches the code. `native.py:462-465` builds a fresh `Geometry` from the binding's transform, dimensions and intervals.
>
> ## Blocking Findings
> None.
>
> ## Non-Blocking Findings
> ### N1: The crop test's slice-bounds assertion checks nothing
> - Confidence: High
> - Evidence: in `tests/test_resample.py` (new test), `all(part.start >= 0 and part.stop <= limit ...)` always holds, because Dask chunk reads never cross chunk boundaries. The real guard is `len(recording.reads) == 1` against 16 chunks (`_RecordingArray.shape = (8,64,64)` with chunks (8,16,16), `:624,638`), and that one is meaningful.
> - Suggested change: delete the `all(...)` assertion, or assert the expected chunk's start offsets instead.
> - Would change my mind: if dask's `from_array` can issue partial reads that don't line up with chunk boundaries here.
>
> ### N2: The `points()` default repeats the `AXIS_DIM` constant as a literal
> - Confidence: High
> - Evidence: `_geometry.py:425` uses `axis_dim: str = "axis"`, while `AXIS_DIM` (`:35`) is still the name `point_at` uses (`:418-419`). The two can drift apart.
> - Suggested change: write `axis_dim: str = AXIS_DIM`. Optionally add a matching `UNITS_COORD` constant.
> - Would change my mind: if keeping the literal in the signature is a deliberate choice for readable rendered docs.
>
> ### N3: Fully scalar `Geometry.points()` has no direct test after the rewrite
> - Confidence: Medium. This is a coverage gap, not a known bug.
> - Evidence: with `order=()`, `evaluate` gets 0-d arrays, and `np.stack` produces shape (n_src,). This matches the old behaviour, but none of the new tests cover a scalar-selected Geometry with custom names or Dask, and the plan's acceptance list mentions scalar Geometry targets.
> - Suggested change: add one case, a selected scalar view with `points(axis_dim="component", ...)`, that checks dims `("component",)` and the values.
> - Would change my mind: an existing test that already calls `points()` on a fully selected Geometry.
>
> ## Verification Commands
> - N3: `pytest tests/test_geometry.py -k "points and (scalar or select)" -q`. If nothing is collected, the gap is confirmed.
> - General sanity: `pytest tests/test_resample.py tests/test_geometry.py tests/test_native.py -q`
>
> ## Recommendation
> Accept. N1–N3 are optional cleanups and need no human decision.

### Opus implementation convergence

> ## Convergence Assessment
> | ID | Status | Rationale |
> |---|---|---|
> | N1 | Resolved | The empty `all(part.start >= 0 ...)` check is gone from `test_scalar_affine_target_reads_only_nearby_source_chunks`. The two meaningful guards are still there: no reads before compute (`tests/test_resample.py:685`) and exactly one read out of 16 chunks (`:688`). The `all(part.stop ...)` at `:643` is in an older, unrelated test, where it filters out probe reads and does real work. |
> | N2 | Resolved | `src/xarrayrf/_geometry.py:425` now uses `axis_dim: str = AXIS_DIM`. The `"units"` literal is unchanged, which is fine because I offered a `UNITS_COORD` constant only as an option. |
> | N3 | Resolved | `test_points_custom_names_support_a_fully_selected_point` covers both eager and Dask. It builds `Geometry(..., dims=())` on `isel(slice=1, row=1, column=3)` and checks that construction and `points()` run no tasks. It checks the default `("axis",)` and custom `("component",)` dims, the frame labels and units, and the values. The expected values match the fixture's transform (`tests/test_geometry.py:39-45`): L = 10 + 0.5·3 = 11.5, P = 20 + 0.25·1 = 20.25, S = −5 + 2.0 = −3.0. This assumes the default integer index coordinates for row and column; I did not read the lines that build them. |
>
> Notes on the lazy scalar case. I traced this by reading the code; I did not run it:
> - A 0-d Dask DataArray has `chunks == ()`, not `None`, so `points()` goes down the chunk branch with an empty dict.
> - `coordinate.chunk({})` turns the NumPy coordinates into Dask arrays, so the test's `isinstance(result.data, da.Array)` is consistent with the code.
> - If xarray ever returns `None` here for a 0-d array, that assertion would fail visibly rather than pass silently.
>
> To confirm:
> `pytest tests/test_geometry.py -k "fully_selected_point" -q`
>
> ## New Findings
> None. The revisions only delete an assertion, swap a literal for a constant, and add a test. None of them changes production behaviour beyond what the first pass already reviewed.
>
> ## Open Disagreements
> | ID | Designer / Implementor Position | Critic / Reviewer Position | Human Decision Needed? |
> |---|---|---|---|
> | — | — | — | No |
>
> ## Recommendation
> Accept. The orchestrator still needs to run the full suite after integration, on both stock and patched xarray:
> `pytest tests/test_resample.py tests/test_geometry.py tests/test_native.py -q`, then the full run.

## Final Independent Validation

| Check | Result |
|---|---|
| Stock xarray full suite after integration | 1,696 passed, 47 expected failures, 3 existing dependency warnings |
| Patched xarray full suite after integration | 1,743 passed, 3 existing dependency warnings |
| Ruff lint / formatting | Passed |
| Mypy | Passed, 81 source files |
| Whitespace | Passed |
| Integration | Seven implementation files match the final reviewed worktree byte-for-byte |
| Unrelated user files / branch / index | Preserved |

The gpt-6-astra follow-up also passed 21 narrow tests, formatting, lint and whitespace checks.
Model selection for the initial handoff is confirmed by the CLI header (`gpt-6-astra`, high
reasoning effort). The follow-up used an explicitly selected gpt-6-astra subagent with a
minimal task payload and no surrounding conversation history. Claude's CLI did not separately
announce its model; attribution uses the recorded `--model claude-opus-5-5` selections.

Review response caps were 900/650 words for the plan and 1,000/500 words for implementation.
Source files were referenced by path; only the short plan, briefs and complete prior reports
were embedded. No duplicate in-progress reviewer was launched. The four reviewer runs
completed successfully and their temporary sessions were cleaned up by the runner.

## Independent QA Before Commit

The subsequent QAEngineer pass found no additional defect and required no source correction.
An archived snapshot of the original source at the stage 2 base fails all 40 selected scalar
resampling regressions. The current focused suite passes 297 tests with 41 expected stock
failures. Four independent eager/Dask nonlinear-registration probes verify scalar output
against a ramp oracle, outside fill and target-frame binding. Coordinate-only Dask fields
also work with matching and differing chunk layouts.

Both full suites and static checks were repeated with the same final results above. The
[internal changelog](changelog.md) records QA approval. The commit scope excludes the user's
README index edits, relativity-note edit and curved-spacetime review.
