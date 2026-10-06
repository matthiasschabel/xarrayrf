# Stage 3 implementation and refinement record

**Status:** Implemented
**Last updated:** 2026-10-05
**Scope:** F3 NGFF intrinsic-axis semantics and F4 linear resampling near NaNs

## Context

The user authorized stage 3 using the established workflow and explicitly allowed clean
correctness changes during development. The base is
`7724d4134c87ea5d32b6b59b8da491ec5acc1ee3`. Claude Code plan and implementation reviews
explicitly selected `claude-opus-5-5`. Codex `gpt-6-astra` implemented the fixes in a detached
worktree, then handled targeted review followups. The initial implementation CLI used high
reasoning effort. Codex GPT-6 orchestrated, independently checked the defects and performed
QA. The user's three unrelated documentation changes were preserved and excluded.

The review runner retained fixed read-only application controls; escalation allowed macOS
Keychain/network access, so reviewer isolation relied on those application controls.
Implementation edits were isolated from the source checkout. No remote publication occurred.

## Current Decision

Stage 3 is implemented, cross-reviewed and independently QA-verified. The fixes and shared
records are committed to local main in separate coherent units.
NGFF metadata is committed as `e302e92` and linear resampling as `b1fd1b3`.
Shared public contracts, QA and review records accompany them in a separate commit.

Non-diagonal NGFF exports derive each intrinsic axis from its exact contributing physical
rows. Unmixed columns inherit type and unit, including unknown values. Multirow columns
require explicitly spatial rows with the same canonical declared unit. Numeric unit
conversion and implicit coefficient snapping are not introduced. Existing intrinsic/physical
systems, affine mapping, orientation reports and lazy pixels are preserved. Upstream v06
layout failures become contextual ValueErrors with the original ValidationError cause;
guidance distinguishes intrinsic transposition from physical frame declaration ordering.

Linear interpolation runs values and component-wise NaN weights through the same kernel.
Weights above `1e-9` propagate missingness; zero and roundoff-only weights do not. The same
bound classifies integral gathers, replacing the old `1e-6` bound. There is no weight
renormalization. Buffer preparation occurs once per slice, with at most one slice's extra
buffers retained. Cropped lazy reads, scalar targets, cells boundaries and fill stay covered.

### Plan dispositions and measured rounding allowance

Both plan passes recommended **Revise**. The first exposed partially integral contamination
and v06 axis-layout constraints; the second resolved both blockers and requested a measured
allowance before implementation. The final plan adopted the reviewer's fixed-bound alternative,
then the implementation review explicitly accepted that rule. This is evidence-based resolution,
not a claim that the second plan report said Accept.

| Plan ID | Disposition | Decision |
|---|---|---|
| B1 | Accepted | Use two-pass masks for fully integral, partially integral and fractional queries; handle complex components separately. |
| B2 | Revised | Wrap the upstream validator instead of duplicating all axis-order/count rules. Preserve the cause and caller actions. |
| N1/N6 | Accepted with measurements | Use shared `1e-9` gather/missing-weight bound and cover large-origin oblique grids. |
| N2 | Deferred, scope accepted | Keep infinity behavior and document SciPy's existing interpolation contract. |
| N3 | Accepted | Exact nonzero support means near-permutations with residual cross-type coefficients remain unsupported. |
| N4 | Accepted | Compare canonical unit spelling aliases without numeric conversion. |
| N5 | Accepted | Test cubic prefilter propagation separately from same-grid gather. |
| N7 | Accepted | Document that tiny zero-fill deficits below the bound are not renormalized. |

Public `Grid.points` to `positions_at` round trips on 13×17 grids rotated 31 degrees measured:

| Origin | Spacing | Maximum index error |
|---|---|---|
| (-250, 300) | (0.5, 0.7) | 1.14e-13 |
| (-250, 300) | (0.037, 0.081) | 1.59e-12 |
| (-13700, 8456) | (0.5, 0.7) | 3.64e-12 |
| (-13700, 8456) | (0.037, 0.081) | 8.73e-11 |
| (100000, -120000) | (0.5, 0.7) | 2.91e-11 |
| (100000, -120000) | (0.037, 0.081) | 4.66e-10 |

The fixed bound covers these measured cases, not arbitrary ill-scaled custom transforms.

### Implementation review dispositions

The first implementation pass recommended **Accept**, with no blockers. gpt-6-astra handled
four inexpensive followups, and the final convergence pass confirmed acceptance.

| ID | Disposition | Decision |
|---|---|---|
| N1 | Accepted | Remove misleading per-failure advice parameter; test the intentionally combined guidance. |
| N2 | Accepted | Confirm every v06 system is validated and add physical-frame ordering guidance and a public rejection/repaired-declaration test. |
| N3 | Deferred | Preserve the current bounded per-slice loop/cache; profile large multi-context tasks before adding routing complexity. |
| N4 | Accepted | Extend six large-origin cases with a `1e-8` pixel affine offset, forcing the mask kernel instead of gather. |
| N5 | Accepted | Wrap touched prose; qualify earlier timing claims as finite-data measurements. |

### Independent QA

QA ran the new public tests against the original source before integration: **33 failed,
5 passed**, confirming the old behavior rather than relying solely on green new tests.
Four additional eager/Dask affine/general partial-integral probes independently recovered
`[1.5, 2.5]` beside a NaN in the neighbouring row. A separate physical-frame ordering probe
confirmed rejection and preserved upstream error context. Semantic metadata and physical
sample mappings, complex components, positive missing weights, buffer lifetime, lazy crop
reads and boundary behavior were checked through public interfaces.

After byte-for-byte integration of all six reviewed implementation files, both full suites
passed on main: **stock xarray 1,739 passed, 47 expected failures; patched xarray 1,786 passed**.
Both retain the three existing dependency warnings. Ruff lint, formatting (127 files), mypy
(81 source files) and whitespace checks pass. Only scoped implementation and documentation
files enter the commits; the user's original file hashes and empty index were checked before
integration and before committing.

## Alternatives Considered

Gather-only restoration cannot fix partially integral queries. Duplicating the NGFF validator
would create an additional format-rule implementation to maintain. Renormalized interpolation,
NaN-aware spline fitting and numeric unit conversion would change the requested scope.
The measured fixed bound avoids a configurable tolerance/error-propagation subsystem.

## Deferred Work

Stages 4–5 remain pending. Cubic prefiltering can spread NaNs through coefficients; same-grid
gathers preserve original samples, so cubic still has the explicitly documented consistency
limitation. Infinity behavior remains SciPy's existing behavior. Optional N3 performance
regrouping belongs to maintainers after profiling large context stacks above the position
cache budget; it is not claimed negligible or solved.

## Next Steps

Stage 4 requires deliberate public-contract decisions before interface changes. Stage 3 commits
retain `Co-Authored-By: Codex (gpt-6-astra)` and
`Reviewed-By: Claude (claude-opus-5-5)` attribution. No remote push is authorized or performed.
The task-owned isolated worktree was removed after integration and successful final checks.

## Open Disagreements

None blocking. The optional performance optimization is explicitly deferred with its cost and
followup evidence stated.

## Reviewer Reports

The following are complete verbatim reviewer reports, retained as quotations. Their line
references describe the reviewed worktree versions, not necessarily final line numbering.

### Initial plan review

> Reviewing the stage 3 plan against the source: F4 has a gap with partially integral queries, and F3 can now hit NGFF axis-order rules. Report follows.
>
> ## Artifact Summary
>
> Stage 3 makes two fixes:
> - **F3:** synthetic NGFF intrinsic axes take their type and unit from the physical row(s) each lattice column maps to, instead of hard-coded `type="space"` and units zipped by position.
> - **F4:** linear resampling returns the actual source value at an exact sample position, even when a neighbour with zero weight is NaN.
>
> Both diagnoses check out against the source:
> - `_export.py:196-199` hard-codes `"space"` and zips `lattice.dims` with `frame.units` by position.
> - `_resample.py:444` and `:508` send linear queries through SciPy unmasked. Same-grid gather (`:488-494`) bypasses SciPy for every order.
>
> The F3 design is mostly sound. The F4 design leaves a real inconsistency.
>
> ## Blocking Findings
>
> ### B1: The F4 exact-sample fix leaves partially integral queries broken, and those are the common case
> - Confidence: High
> - Evidence: SciPy order-1 weights per axis are `1-t, t`. At an integral coordinate `t=0`, so the next neighbour is still read with weight 0, and `0*NaN = NaN`. This happens per axis, not per point. Take a 2-D resample that keeps axis 0 and upsamples axis 1 (a common slice-spacing change). Every target point is integral on axis 0, so a NaN in row `r+1` poisons row `r` everywhere. The plan's gather only fires when all coordinates are integral, so this stays wrong. "Nonintegral queries retain SciPy behaviour" also lumps together two different cases: a NaN neighbour with positive weight (propagating is correct) and one with zero weight (the bug).
> - Suggested change: use the plan's alternative, a mask pass. Only when `np.isnan(volume).any()`:
>   - Interpolate `where(isnan, 0, volume)` with the existing kernel.
>   - Interpolate `isnan(volume).astype(float64)` with the same kernel and the same positions/matrix: `affine_transform` at `:508`, `map_coordinates` at `:444`.
>   - Set NaN wherever the interpolated mask exceeds a named tolerance.
>
>   Why this is simpler:
>   - Linear weights are non-negative, so the mask value is exactly the total weight on missing samples.
>   - It uses the same code in the affine and general paths.
>   - The affine path needs no position computation or mesh.
>   - It covers full, partial and nonintegral queries.
>   - It works for complex values (`np.isnan` on complex is true if either part is NaN; zero-fill is `0+0j`).
>   - Cost is one extra slice copy plus one float64 mask per affected slice.
>   - Restrict it to `order == 1`; nearest needs nothing, and cubic stays documented.
> - Would change my mind: a stated requirement that partially integral queries keep propagating NaN, or a benchmark showing the second compiled pass is unacceptable on NaN-bearing slices.
>
> ### B2: F3 inherited types can violate the NGFF v0.6 axis-order validator
> - Confidence: High (validator read; the failure itself is not run)
> - Evidence: `.venv/.../ome_zarr_models/v06/multiscales.py:346-393` requires, for every coordinate system:
>   - 2–3 space axes, and space axes last;
>   - at most one time axis, and it must be first;
>   - at most one "custom" axis, and `type=None` counts as custom.
>
>   The intrinsic system's axis order is fixed by the array dims, because the Scale applies per array axis. Its types now come from whichever physical row each column hits. Examples that fail:
>   - Array dims `(j, i, t)` with the time column last.
>   - Two columns inheriting `None`.
>
>   Both raise a raw pydantic `ValidationError` from `Multiscale(...)` at `_export.py:218`. Today all-space intrinsics never trip the order rule (only the count rule). One change runs the other way: a 4-D `(t, z, y, x)` rotation currently fails with four space axes and would start passing.
> - Suggested change: check inherited intrinsic types against the v06 rules before building the model. On violation, raise a `ValueError` that names the array dims and the required order (time first, space last) and tells the user to transpose the array. Add acceptance tests for one transposed time case and one double-`None` case. Synthetic reordering is not possible without redesigning the format, which the plan rules out.
> - Would change my mind: evidence that the plan already intends to rely on the validator's message and the user accepts the raw pydantic error.
>
> ## Non-Blocking Findings
>
> ### N1: The integrality tolerance must match the gather tolerance
> - Evidence: `_same_grid_indices` treats maps within `1e-6` index units as integral (`_resample.py:266`) and gathers exact samples. A non-gather path with a "tight" roundoff tolerance (e.g. `1e-12`) would treat a `1e-8`-off composite map differently. That is the path dependence this fix is meant to remove.
> - Suggested change: one named module constant, used both by `_same_grid_indices` and by the F4 mask threshold (or the integrality test, if B1 is rejected).
> - Would change my mind: a reason the gather tolerance should be broader than the interpolation tolerance.
>
> ### N2: ±Inf has the same zero-weight failure
> - Evidence (speculation, not run): `0*inf = NaN` in SciPy's weighted sum, so a finite exact sample next to an Inf also comes back NaN. The plan covers NaN only.
> - Suggested change: either extend the mask to `~np.isfinite` (then the Inf value needs restoring, which adds policy), or explicitly scope Inf out and document it. I'd scope it out and document it.
> - Would change my mind: the command in Verification Commands returning `[1, 0.5]`.
>
> ### N3: F3 "contributing rows" needs exact-zero semantics stated
> - Evidence: the plan says not to drop small coefficients. A rotation stored in floating point (e.g. `cos(π/2) ≈ 6e-17`) then becomes a "mixing" column. Between a time row and a space row it would refuse export of a geometrically pure permutation.
> - Suggested change: state that this refusal is intended, and add a test using a floating-point-generated 90° rotation so the behaviour is visible.
> - Would change my mind: the lattice constructor already snapping such entries to exact zero.
>
> ### N4: Unit agreement check
> - Evidence: `canonical()` (`units.py:56`) maps `"millimeter"` to `"mm"`, so comparing `canonical(a) == canonical(b)` is enough. This is correct as planned. Keep `None == None` (undeclared on both rows) as a refusal, which the plan already says.
>
> ### N5: F4 cubic documentation test
> - Evidence: gather applies to cubic too (`_resample.py:488`, not gated by order). The plan's "distinguish gather from a true spline query" test is the right one. Make sure it asserts the current output, not an aspiration.
>
> ## Verification Commands
>
> - **B1:** `.venv/bin/python -c "import numpy as np;from scipy import ndimage as n;a=np.array([[1.,2,3],[np.nan,5,6]]);print(n.map_coordinates(a,[[0,0],[0.5,1.5]],order=1,mode='nearest'))"`. I expect `[nan nan]`; correct is `[1.5 2.5]`.
> - **B2:** build a `Geometry` whose array dims are `(j, i, t)` and whose lattice maps the last column to a time row with an off-diagonal spatial pair, then call `to_multiscale_level`. Expect a pydantic error saying "Time axis must be at the beginning".
> - **N2:** `.venv/bin/python -c "import numpy as np;from scipy import ndimage as n;print(n.map_coordinates(np.array([1.,np.inf]),[[0,0.5]],order=1))"`. My speculation predicts `[nan inf]`; a result of `[1, …]` would refute N2's premise.
>
> ## Recommendation
>
> **Revise.** B1 changes the F4 design: I recommend the mask pass over exact-sample gather, which answers the plan's question for Opus. B2 adds a required pre-validation step to F3.
>
> Two decisions belong to the human:
> - Whether to scope Inf in or out (N2).
> - Whether refusing export of floating-point near-permutations is acceptable (N3).

### Plan convergence

> ## Convergence Assessment
> | ID | Status | Rationale |
> |---|---|---|
> | B1 | Resolved | The two-pass order-1 mask covers fully integral, partially integral and fractional queries in both the affine (`_resample.py:508`) and general (`:444`) kernels. Splitting complex masks by component is stricter than I asked for, and correct. The plan prepares buffers once per slice and holds only one slice's buffers at a time, which addresses the memory concern. |
> | B2 | Resolved | Wrapping the upstream `ValidationError` from `Multiscale(...)` (`ngff/_export.py:218`) keeps the cause, adds array dims and types, and avoids copying v06 rules. That matches my "would change my mind" condition. The tests cover time-last ordering and multiple unknown axes. |
> | N1 | Maintained, partly: the revision added a regression (see N6) | Sharing one allowance between gather and mask resolves the path-dependence I raised. But the proposed value, an absolute ~1e-12 in index units, is too tight for real geometry. |
> | N2 | Resolved | Inf is out of scope and documented. The SciPy result `[NaN, Inf]` confirms the premise. |
> | N3 | Resolved | Exact structural zeros are now documented and tested. Same-unit spatial near-rotations stay valid. |
> | N4 | Resolved | Unchanged; already correct. |
> | N5 | Resolved | Cubic prefilter propagation is tested separately from gather. |
>
> ## New Findings
>
> ### N6: An absolute 1e-12 allowance pushes ordinary same-grid maps off the gather path
> - Confidence: medium-high. The arithmetic below is reasoned, not run.
> - Evidence: `_same_grid_indices` measures error at the target corners, in absolute index units (`_resample.py:263-267`). The lattice map comes from composing physical affines, so roundoff grows with coordinate size: roughly eps × |origin| / spacing.
>   - A typical CT origin of −250 mm with 0.5 mm spacing gives about 2e-16 × 500 ≈ 1e-13. That is fine.
>   - Composites going through inverses, or origins around 1e3–1e4 with sub-mm spacing, easily reach 1e-12 to 1e-10.
> - At 1e-12 those cases fall off the gather path:
>   - **Linear:** values pick up ~1e-12 relative errors instead of exact samples. A NaN neighbour now carries a "positive" weight of ~1e-11, which is above the allowance, so the exact sample becomes NaN. That is F4's own bug coming back through roundoff.
>   - **Cubic:** the query goes through the spline. With NaN anywhere in the slice, the prefilter spreads NaN across it, where gather used to return the original samples. This makes the cubic limitation the plan documents worse.
> - The plan's 1e-8 perturbation test can't catch this, because it only checks that positive-weight NaNs propagate. Nothing in the acceptance criteria asks whether realistic round trips still take the gather path.
> - Suggested change:
>   - Keep one shared allowance, but scale it to the size of the map: e.g. `k·eps·max(1, |offset| + Σ|linear|·(size−1))` per source axis, with a small k such as 64. Otherwise justify a fixed bound of about 1e-9 from measured round trips.
>   - Add an acceptance case that runs a large-origin, sub-mm-spacing composite round trip (forward then inverse) and asserts both the gather path and exact NaN-neighbour samples.
>   - "Calibrate around 1e-12" is a placeholder, not a decision. The implementer needs a concrete value or rule.
> - Would change my mind: measured `max|error|` across the existing public round-trip tests and a large-origin case, all below 1e-12. The orchestrator can check this by temporarily logging `np.max(np.abs(error))` in `_same_grid_indices` during the focused resample test run.
>
> ### N7 (minor): The zero-filled value isn't renormalized below the allowance
> - When the missing weight w is positive but at or below the allowance, the output is `(1−w)·value` rather than `value`. At 1e-12 that doesn't matter. If N6 moves the allowance up to ~1e-9 it still doesn't matter, but either state that it's ignored or divide by `(1−w)`. No action is needed beyond one sentence in the plan.
>
> ## Open Disagreements
> | ID | Designer / Implementor Position | Critic / Reviewer Position | Human Decision Needed? |
> |---|---|---|---|
> | N1/N6 | Shared absolute allowance of ~1e-12, calibrated during implementation | Shared allowance must scale with the map's magnitude (or be justified at ~1e-9); otherwise roundoff brings back F4's bug and the cubic NaN spread on realistic geometry | No. This can be settled with data from the command above. Escalate only if the designer keeps an absolute 1e-12 without measurements. |
>
> ## Recommendation
> **Revise, minor.** B1, B2 and N2–N5 have converged. Before implementation, replace the "around 1e-12" candidate with a magnitude-scaled rule (or a measured fixed bound), and add the large-origin round-trip acceptance case (N6). No human escalation is needed.

### Initial implementation review

> ## Artifact Summary
> F3 and F4 are both correct as implemented, and the 1e-9 disposition holds up. I found no blockers, only five non-blocking issues.
>
> **F3 (NGFF intrinsic axes).** `_intrinsic_axes` (`src/xarrayrf/ngff/_export.py:112-141`) uses exact nonzero support. An unmixed column inherits its row's type and unit. A mixed column is accepted only when every row is declared `space` and the rows share one canonical unit. `udunits_name(canonical(u))` gives the same spellings as before. A zero column is already refused upstream at `_positions.py:376` ("map to no displacement"), so the `assert` is a real invariant and not validation.
>
> **F4 (linear NaN handling).** `_linear_inputs` and `_restore_missing` run the same kernel, `mode="nearest"` and clipping on the zero-filled values and on the mask. Outside fill is applied after restore. The cubic-only shell (`_resample.py:541`) never touches the zero-filled linear buffers, so NaN edge cells stay NaN. Complex real and imaginary parts are handled separately through views on the copy. Integer and bool sources get float64 output (`_output_dtype`), and `np.isnan` is safe on them.
>
> **1e-9 allowance.** For linear interpolation, the missing weight at a node is at most the index error. Multi-axis neighbours only make it smaller. So the measured maximum of 4.66e-10 sits about 2× under the bound. That is a defensible fixed bound, and the docs scope it honestly. Tightening the gather threshold from 1e-6 only sends more maps through interpolation; it cannot gather wrongly.
>
> ## Blocking Findings
> None.
>
> ## Non-Blocking Findings
>
> ### N1: The ordering and count advice is always both, so the test parameter can't tell them apart
> - Confidence: High
> - Evidence: the error message at `_export.py:262-266` always contains both "transpose" and "explicit". So `test_invalid_ngff_axis_layout_has_actionable_error`'s `advice` parameter (`tests/test_ngff_export.py` ~470) passes for every case. The plan asked for advice matched to the failure type.
> - Suggested change: either accept the combined advice and drop the `advice` parameter, or choose the hint from `error.errors()` content and assert the other hint is absent.
> - Would change my mind: a decision that combined advice is acceptable. If so, only the test's false precision needs fixing.
>
> ### N2: "Transpose the array" is wrong advice when the frame system is the one that fails
> - Confidence: Medium (speculation: I couldn't confirm that v06 order-validates every coordinate system)
> - Evidence: the `try` wraps validation of both `systems`, including `_system(frame, ...)`. In the diagonal branch the intrinsic system *is* the frame. If the frame axes are, say, `(x, t)`, transposing the array can't fix it; the frame has to be reordered.
> - Suggested change: when the failing system is the frame or `frame_name` system, say "reorder frame axes" instead.
> - Would change my mind: v06 validating ordering only on the dataset's intrinsic system. Check with the first verification command below.
>
> ### N3: One NaN slice makes every slice in the task take the slow path
> - Confidence: High
> - Evidence: `needs_masks` (`_resample.py:602`) is evaluated per task. If any slice has a NaN, finite slices also go through the per-slice loop. Above `POSITION_CACHE_BYTES` that means positions are recomputed for every slice. Each slice is also scanned twice for NaNs (the `any(...)` check, then `_linear_inputs`).
> - Suggested change: optional. Precompute per-slice NaN flags once and route finite slices through `_interpolate` without masks. The positions question remains for the uncached case, and the docs already disclose it.
> - Would change my mind: a profile showing it's negligible for realistic echo/time counts.
>
> ### N4: The large-origin oblique test only exercises the mask under roundoff on the Composite path
> - Confidence: High
> - Evidence: in `test_linear_nan_oblique_round_trip_at_large_origins` the plain-affine self-mapping composes to the identity and hits the gather path. That is still a valid regression test for gather, but it never tests the affine mask under roundoff. The 1e-8 perturbation is only exercised on the 1-D unit grid, not at large origins.
> - Suggested change: add one oblique case with a non-integral target offset, or a 1e-8 shift, so `affine_transform` mask weights are tested at origins of 1e5.
> - Would change my mind: the plan's owner considering 1-D perturbation coverage sufficient.
>
> ### N5: Over-long lines in prose
> - Confidence: High
> - Evidence: `_resample.py:691` (146 characters) and `:38`, plus `docs/core_interface.md` around the "Dask-backed" bullet, exceed 100 columns. E501 is ignored, so lint passes, but surrounding prose is wrapped at 100.
> - Suggested change: rewrap these lines.
> - Would change my mind: nothing; this is a cosmetic housekeeping nit.
>
> ## Verification Commands
> None of these are needed for a blocker.
> - N2: `python -c "from ome_zarr_models.v06 import multiscales, inspect; print(inspect.getsource(multiscales.Multiscale))"` (run in the project venv). Shows which systems get ordering validation.
> - Confirm the suite is green: `pytest tests/test_ngff_export.py tests/test_resample.py -q`
>
> ## Recommendation
> **Accept.** N1 and N4 are cheap test fixes worth doing before commit. N2 needs a human decision only if v06 turns out to validate the frame system's ordering.

### Implementation convergence

> ## Convergence Assessment
> | ID | Status | Rationale |
> |---|---|---|
> | N1 | Resolved | The `advice` parameter is gone. `test_invalid_ngff_axis_layout_has_actionable_error` now asserts the combined message (`intrinsic.*d0.*transpose.*explicit`) and checks that the cause is a `ValidationError`. The test no longer claims to tell the two kinds of advice apart. |
> | N2 | Resolved | The message now gives separate advice for physical-frame ordering (`_export.py:262-268`). In `test_invalid_physical_frame_order_has_actionable_error`, the permutation gives intrinsic types `(time, space, space)`, which are valid, and frame types `(space, time, space)`, which are not. So it really tests the frame-system failure, and the reordered case shows the fix works. That also settles my earlier guess: v06 does validate the ordering of the frame system. The test also matches the pydantic message text `['space', 'time', 'space']`, which could break on an `ome_zarr_models` upgrade. Minor; not raised as a finding. |
> | N3 | Maintained (deferral accepted) | The risk is still in the code: `needs_masks` at `_resample.py:602` is computed per task. The deferral rationale is sound, though: buffers stay bounded, nothing has been measured as a bottleneck, and the docs say "positions are reused … when no linear NaN buffers are needed". One small gap: the docs don't say that a single NaN slice moves the *whole task* off the reuse path. Non-blocking. |
> | N4 | Resolved | The new `shifted` case moves `j` by 1e-8, which is above the 1e-9 gather bound, at origins up to 1.2e5. This forces the plain-affine path through `affine_transform`. It checks both effects: the positive missing weight shows up as NaN at `expected[4, 7]`, and the cross-axis roundoff (≤4.66e-10) on neighbours such as `[5, 6]` stays finite. The check uses `assert_array_equal` on the NaN pattern plus `atol=1e-9` on values. |
> | N5 | Resolved (cosmetic residue) | The over-long lines are fixed, and the old timings are now described as "For finite data". A few wraps are still ragged: `_export.py:266-268` (`"axis declarations "` / `"supported by v06. ..."`), `docs/core_interface.md:719-720` (`adds about 0.01 s` on its own line) and `:991-993`. The new line `tests/test_ngff_export.py:539` is 102 columns, from a `type: ignore`. All harmless. |
>
> ## New Findings
> None. I checked the revised paths:
> - **Lattice path:** the order-1 branch builds the mask buffer and restores NaNs before outside fill and before the cubic-only shell.
> - **General path:** buffers are dropped (`del`) per slice, and the cached and uncached position branches are both covered by `test_linear_nan_buffers_are_reused_per_slice_and_released` (`cache_bytes` 0 and 1 MiB).
> - **Intrinsic axes:** the `systems` assignment moved into each branch without changing diagonal behaviour.
>
> ## Open Disagreements
> | ID | Designer / Implementor Position | Critic / Reviewer Position | Human Decision Needed? |
> |---|---|---|---|
> | N3 | Defer until a many-slice, over-budget task is profiled. | Agree. Optionally say per task in the docs. | No |
>
> ## Recommendation
> **Accept.** One optional tidy-up: rewrap the three ragged spots listed under N5. To confirm, run `pytest tests/test_ngff_export.py tests/test_resample.py -q`. I didn't run it because this was a read-only pass.
