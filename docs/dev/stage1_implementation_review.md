# Stage 1 implementation and refinement record

**Status:** Implemented
**Last updated:** 2026-10-05
**Scope:** Audit findings F1 (NGFF identity), F2 (singleton support), and F8 (DICOM cardinality)

## Context

The user authorized implementation of stage 1 of the [audit plan](codebase_audit_plan.md),
followed by Claude Opus 5.5 review through collaborative-refinement. Codex implemented the
changes in the shared working tree at base commit de48d8ab0097bf6196fdcbced0779de0ae155fea.
The implementation-review phase made no commits or upstream submissions. The subsequent
QAEngineer invocation authorized committing the clean stage 1 changes to main. The user's
unrelated documentation changes remain outside that scope.

Two report-only implementation-review passes explicitly selected `--model claude-opus-5-5`
and fixed `--permission-mode plan --output-format text`. Both succeeded; the CLI did not
separately announce its model, so attribution uses the recorded selection flags. Runner
escalation supplied network and macOS Keychain access while leaving Claude's application-level
read-only review controls in place. Claude checked source but did not re-run tests; Codex ran
the verification below. Logs are `/tmp/xarrayrf-stage1-claude-log.md` and
`/tmp/xarrayrf-stage1-convergence-log.md`; both full reports are preserved below.

## Current Decision

Stage 1 is complete, has passed QA and is committed to main. Claude's final recommendation
is **Accept**, with all findings resolved and no new findings or open disagreements.

The fixes are independently committed as `299eda4` (NGFF identity), `b490720` (singleton
matching) and `3d97352` (DICOM cardinality). Shared documentation and review records accompany
them in a separate commit. No remote publication was requested or performed.

- NGFF local readers resolve filesystem paths and symlinks to the same bare absolute path
  for opening and frame identity. Metadata-only store identities remain caller-resolved;
  URLs retain prior handling. Existing persisted raw-path identifiers are not rewritten.
- Singleton support uses a float64 rounding allowance rather than a broad origin-relative
  physical window. Affine inverse terms account for cancellation; built-in affine chains
  propagate their allowances. Lookup, coincidence, and resampling share the matching rule.
- DICOM imports retain exact original `source_count` before selection/sorting. Missing or
  surplus pixel slices fail on shape metadata before evaluation. The final defaulted field
  preserves legacy constructor call shapes; unknown manual counts retain index validation.
  Explicit counts are positive integers and must contain every selected index.

Public documentation and `CHANGELOG.md` describe these contract changes. Persistence schema
and later-stage public defaults were not changed.

### Validation

| Check | Final result |
|---|---|
| Stock xarray full suite | 1,638 passed, 47 expected failures, 3 existing dependency warnings |
| Patched xarray full suite | 1,685 passed, 3 existing dependency warnings |
| Ruff lint | Passed |
| Ruff formatting | Passed, 125 files |
| Mypy | Passed, 81 source files |
| Git diff whitespace | Passed |

The first focused regression run before implementation had 12 failures and 2 passing
protective cases, reproducing the original defects and changed contracts. Subsequent focused
runs and both final full suites passed. Coverage includes canonical paths and metadata frame
reuse, epoch-scale false matches, large-translation affine/composite round trips, point-only
providers, composite subclasses, original enhanced stack lengths, count/order contradictions,
legacy manual construction, and lazy shape validation.

The subsequent QA pass independently reproduced eight failing cases against an archived
snapshot of the original source without changing the shared checkout. Six protective cases
passed on that snapshot. All three focused modules pass on the current source (178 passed,
five expected stock-xarray failures). Four mixed-axis probes also pass for affine/composite
maps, including sign changes, empty queries, singleton round trips, coincidence, resampling
and outside fill. Both full suites and static checks were repeated with the results above.
No source correction was needed. See the [internal changelog](changelog.md) for the QA record.

The roundoff calibration used public forward/inverse calls. An epoch-scale round trip differed
by one float64 ulp. With scale 0.1234567 and translation 1e9, final source coordinates near 0.1
and 1000 differed by roughly 5.7e-7 and 9.5e-7, respectively. These can be millions of ulps of
the small final coordinate, explaining why intermediate inverse terms must enter the allowance.
The factor of eight machine epsilons plus dimension-dependent term scale is a conservative
allowance, not an error certificate for arbitrary providers.

## Alternatives Considered

Rejected tightening the singleton relative constant without accounting for affine cancellation.
Rejected resolving metadata-only identity strings through the working directory or migrating
local identities wholesale to file URIs. Rejected inferring enhanced source count from selected
indices. Rejected carrying an error allowance unchanged through an unknown non-affine mapping:
its axis count, units, or scale may differ.

## Deferred Work

A non-affine inverse member resets accumulated allowances. A mixed chain containing large
translations may therefore fail singleton self-lookup even if an affine-only chain succeeds.
This limitation is explicitly documented and the coordinate-only fallback is tested. Affine
providers should expose `SupportsAffine`; no implicit numerical derivatives or general error
propagation framework was added. Stages 2–5 of the audit remain pending.

## Next Steps

Stage 2 is the next independent implementation task. Stage 1 commits carry the reviewer
attribution required by collaborative-refinement:

`Reviewed-By: Claude (claude-opus-5-5)`

## Artifact Summary

The reviewed diff covers six source modules, three test modules, `docs/core_interface.md`,
and `CHANGELOG.md`. Plan/status records accompany the implementation. Review constraints
excluded the user's unrelated edits and untracked files from the implementation artifact.

## Disposition of Findings

| ID | Finding | Disposition | Rationale / Change |
|---|---|---|---|
| N1 | Non-affine member loses earlier affine allowance | revised | Verified the behavior. Documented the reset and mixed-chain singleton limitation rather than assume scale preservation. The already approved non-affine coordinate-only fallback remains. Claude marked resolved. |
| N2 | Explicit DICOM count can contradict order | accepted | Constructor validation now rejects negative or out-of-range selected indices when the count is known; public tests cover both bounds. Unknown-count legacy behavior is retained. Claude marked resolved. |
| N3 | Non-affine fallback lacks coverage | accepted | Added a point-only invertible identity provider test exercising singleton lookup, coincidence, resampling, roundoff acceptance, and one-second rejection. Claude marked resolved. |

A further local correction makes inverse-chain traversal apply only to the exact built-in
CompositeTransform class. Subclass point mappings remain authoritative, matching the existing
built-in recognition policy. A public subclass regression verifies this; Claude confirmed it
in the convergence pass.

## Open Disagreements

None. Both the design response and final critic report accept the documented non-affine limit.
No blocking issues remain. Acceptance is supported by the source review and the final local
checks, not by model agreement alone.

## Recommendation

**Accept.** Stage 1 is implemented and verified. Stop the refinement cycle after two passes.

## First-pass critic report

## Artifact Summary
I recommend accepting. F1, F2 and F8 match the brief, and the claims I checked against the source held up. I found no blocking defects, only three non-blocking issues. I did not re-run the test suite.

What I checked:
- **F1 (ngff paths):** The URL branch still runs before the local-path branch (`src/xarrayrf/ngff/_reader.py:114-117`). The local branch now passes one resolved absolute path both to zarr and to the frame identity (`:122-123`).
- **F2, coincidence:** Coincidence evaluates the map analytically (`base = Minv@t + tinv`), not by running the forward and then the inverse. The allowance it uses covers the cancellation in `base`: `affine_roundoff(Minv, tinv, |M|@s + |t|)` (`src/xarrayrf/_coincidence.py:98-100`) is at least `eps·(|Minv||t| + |tinv|)`. It also covers `|weights|·|s|`, because `|Minv||M||s|` is at least that.
- **F2, lookup:** Lookup and resampling both go through `locator` (`src/xarrayrf/_resample.py:221`, `src/xarrayrf/_positions.py:264`). The extra roundoff work only runs when a dimension has one sample and no step in the domain (`_positions.py:124-127`), so other paths behave as before.
- **F2, infinite values:** Including `|query|` in the scale could match an infinite query to a single sample (`coordinate_to_position(np.array([1000.]), np.array([np.inf]), None)` returns 0). It can't happen through the public API: `transform_points` rejects non-finite output (`_positions.py:327`, and `AffineTransform` raises on overflow). Not a finding.
- **F8 (DICOM):**
  - `_dimension` returns a plain `int`, so `source_count=count` passes the new check (`dicom/__init__.py:164-166`).
  - `from_datasets` requires a `Sequence`, so `len(datasets)` is safe (`:409`).
  - The exact-count check runs before any data is selected or read.
  - Old pickles without the field still load, because the dataclass default supplies `None`.

## Blocking Findings
None.

## Non-Blocking Findings
### N1: A non-affine member in a composite inverse throws away the allowance built up by earlier affine members
- Confidence: Medium. The code path is verified; that it causes failures is my inference, and I did not run it.
- Evidence: in `_transform_with_roundoff` (`src/xarrayrf/_positions.py:172-178`), a member that is not `SupportsAffine` returns `np.zeros_like(coordinates)` and drops `incoming`.
  - Take a forward chain `[N, A]`, where `A` is the large-translation affine from your own measurements (scale 0.1234567, translation 1e9, sample 0.1).
  - Its inverse runs `[A⁻¹, N⁻¹]`. The ~5.7e-7 error that `A⁻¹` introduces is still present after `N⁻¹`, but the allowance falls back to `8·eps·max(1,|coord|)`, about 1.8e-15.
  - The single sample would then fail to locate its own points. The old `1e-9` window accepted this case.
  - The docs say non-affine providers use the coordinate allowance. What they don't say is that a composite loses the allowance its affine members already earned. No built-in non-affine inverse exists, so only user-supplied providers are affected.
- Suggested change: carry the earlier allowance through, e.g. `roundoff = incoming` instead of zeros (this assumes the member roughly preserves scale). Or state in `docs/core_interface.md` that a non-affine member resets the allowance.
- Would change my mind: a design decision that user-supplied non-affine chains containing large translations are out of scope, or a test showing such a chain still locates its own sample.

### N2: An explicit `source_count` is not checked against `order`
- Confidence: High.
- Evidence: `__post_init__` (`dicom/__init__.py:66-71`) checks only the type and that the count is positive. A manual `DicomGeometry(..., order=(0, 5), source_count=2)` builds without error.
  - Given 2 slices, it passes the exact-count check and only fails later at the index check (`:107`).
  - The error is still caught, so this is not a correctness bug. It does mean the object can hold a contradiction.
- Suggested change: in `__post_init__`, raise `ValueError` when `source_count is not None and max(order) >= source_count`.
- Would change my mind: a deliberate decision to keep the manual constructor permissive.

### N3: No test pins the documented non-affine fallback
- Confidence: Medium.
- Evidence: the new tests in `tests/test_intervals.py` cover affine transforms and affine-only composites. None drives a single-sample dimension through a non-affine provider, so the fallback the docs promise isn't checked.
- Suggested change: add one regression test with a small user-defined non-affine `SupportsInverse` transform. It should exercise whichever behavior N1 settles on.
- Would change my mind: an existing test that already does this.

## Verification Commands
None needed (no blockers). To check N1: build a `CompositeTransform(N, A)`, where `N` is a trivial user-defined non-affine identity with an inverse and `A` is the large-translation affine from `test_singleton_round_trip_survives_large_translation`. Then run `Grid(...).positions_at(value.points())` under `.venv/bin/pytest`. I expect it to raise `ValueError` matching "outside".

## Recommendation
Accept. Deciding whether to fix N1 or document it is a scope call for a person.


## Second-pass convergence report

## Convergence Assessment
| ID | Status | Rationale |
|---|---|---|
| N1 | Resolved | The designer chose the "document it" option I offered. The docstring at `src/xarrayrf/_positions.py:163-167` and the singleton paragraph in `docs/core_interface.md` now say three things: a non-affine member resets the allowance, a mixed chain with large translations may fail to locate its own single sample, and affine providers should expose `SupportsAffine`. Their reason for not passing `incoming` through unchanged holds: a non-affine member can change axis count or units, so carrying the earlier allowance could produce the wrong shape or accept too much. |
| N2 | Resolved | `__post_init__` (`src/xarrayrf/dicom/__init__.py:66-73`) now rejects indices below 0 or at/above `source_count` when the count is given explicitly. `dataclasses.replace` runs `__post_init__` again, so the parametrized test checks both bounds. When the count is `None`, behavior is unchanged. |
| N3 | Resolved | `test_singleton_non_affine_provider_uses_coordinate_roundoff` runs a user-defined provider that only maps points and has an inverse. It goes through lookup, coincidence and resampling, and checks that the next float above the sample matches while +1 s does not. |

## New Findings
None.
- **Composite subclass check:** the new exact-type check, `type(transform) is CompositeTransform` (`_positions.py:168`), matches how `_is_builtin_transform` already treats built-in transforms (`src/xarrayrf/_transform.py:132-137`).
  - A subclass that overrides `transform_point` is evaluated as one transform, so its own mapping is used rather than its members'.
  - `test_singleton_lookup_respects_composite_subclass_point_mapping` locks this in.
- **DICOM checks, cosmetic only:** when `source_count` is set, the new index check overlaps with the `max(order)` guard in `to_dataarray`. It's harmless and not a finding.

## Open Disagreements
None.

## Recommendation
Accept. I did not re-run the tests. Both full xarray runs were still in progress when the brief was written, so confirm they pass before committing:
- `.venv/bin/pytest -q -p no:cacheprovider`
- `PYTHONPATH=/Users/matthiasschabel/GitHub/xarray-patched-4 .venv/bin/pytest -q -p no:cacheprovider`
