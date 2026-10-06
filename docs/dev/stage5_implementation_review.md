# Stage 5 implementation and refinement record

**Status:** Implemented
**Last updated:** 2026-10-06
**Scope:** M2 DICOM tool output reliability, A3 cells-domain explanation and singleton documentation

## Context

The user authorized Stage 5 work as in the earlier audit stages and explicitly requested
Opus 5.5 review through collaborative-refinement at logical points. Base revision:
`df385d3d762f3ef6cb11b83343b3e9db41c5ffb9`. Implementation is isolated on `audit/stage5`
while the shared main checkout retains the user's unrelated documentation edits and handoff.
Codex gpt-6-astra implements; Claude `claude-opus-5-5` reviews the plan and implementation;
root Codex GPT-6 independently checks the actual diff and applies QAEngineer guidance.
No remote publication is part of this work.

## Current Decision

The DICOM tool and its CLI regressions are committed as `febbe89`. Public sampling
documentation and completion/QA records accompany them in a separate local commit.

The concrete plan covers complete batch preflight, sibling staging and per-series atomic
publication, exception-safe cleanup, explicit partial-failure reporting and documentation.
A4 adapter Grid/report conveniences and A6 construction/mixed coincidence remain deferred
with usage triggers in the audit plan. No numerical policy or persistence schema changes.

### Plan review

Opus raised no blockers and recommended revisions. Source/output overlap and explicit
malformed-dataset validation are accepted. The designer preserves strict file selection
and the already-authorized keep-going policy, and retains a compact JSON stdout manifest
under the supplied multi-step audit convention. A full report and dispositions follow below.

### Verification

| Check | Result |
|---|---|
| New CLI regressions against original tool | 28 failed in 3.02 seconds; includes new reporting contracts and reproduced output defects. |
| Implementer focused tool + intervals | 112 passed, five existing stock expected failures. |
| Final tool tests after review refinements | 28 passed in 2.58 seconds. |
| Independent root CLI probes | 12 passed: dry run, case collisions, duplicate symlink input, dangling target, overlaps in both directions, invalid output ancestor, empty input, failed middle read with later success, missing metadata, strict dotfile failure and saved/retry UID invariants. Source directory entries, bytes and symlink targets stay unchanged. |
| Independent root documentation probe | Gap position 0.5; linear values `[4, 0]` in cells domain and `[4, -9]` in samples domain with fill -9. Absolute tolerance `1e-12` for small synthetic affine/interpolation roundoff. |
| Independent root invocation probe | A dry-run `main()` call inside an unrelated active exception reports success and no failures; no output is created. |
| Geometry/Grid runtime scope | Executable AST is identical to base after stripping docstrings; numerical behavior is unchanged. |
| Full stock suite | 1,793 passed, 47 existing expected failures, three existing dependency warnings. |
| Full patched suite | 1,840 passed, three existing dependency warnings; dependency SHA `5db75d53f5515fabb3ca5ffc8419eb9c982cf4a2`. |
| Static checks | Repository Ruff lint/format, mypy (82 files) and whitespace pass. Final documentation/test-only refinements pass the relevant narrow checks. |
| Main integration verification | 114 tool/interval/README tests pass plus five existing stock expected failures; repository static checks pass again. Reviewed implementation files are integrated byte-for-byte and protected user-file hashes remain unchanged. |
| Distribution | Wheel and sdist rebuilt; payload checker passes, excluding tools/tests/maintainer docs. |

Full suites ran on the reviewed implementation. Subsequent refinements changed only documentation
and strengthened test attribution/fixture identities, with all 28 tool tests passing afterward;
production runtime code is identical. No full-suite rerun was needed for those refinements.
The original overlap probe independently confirms that the base tool returned success while
creating output inside its source tree. The implemented preflight refuses it before writes.

Stage 4 hosted run `37493729018` at the base passed locked/current
lanes; minimum-core failed 60 tests requiring absent SciPy and minimum-resample mypy failed
eight NumPy-typing checks in three existing files. These failures precede Stage 5 and are
separate from its locked local stock/patched verification.

## Alternatives Considered

A whole-batch transaction would complicate retries and discard successful independent series.
Skipping dotfiles changes input selection; preserve strict immediate-regular-file selection.
A schema API, sidecar logs or general confidentiality framework would overextend this tool.
The output tree is single-writer; directory rename alone is not a concurrent no-replace guarantee.

## Deferred Work

A4/A6 retain the audit plan's explicit reopening criteria. Viewer/upstream/release work and
minimum-dependency CI repair remain separate. No compatibility/deprecation scaffolding.

## Next Steps

Stage 5 implementation, Opus review and independent QA are complete. No further audit stage
is queued. A4/A6, viewer, upstream, minimum-dependency CI and release work retain their own
usage triggers and scope. User documentation and the historical handoff remain outside
these commits. No push or release is part of this work.

## Open Disagreements

None. The implementation reviewer withdrew the earlier JSON-complexity concern after
checking the compact implementation. Strict file selection, keep-going and single-writer
publication retain the reviewed contract. The small post-rename interruption reporting window
is explicitly documented rather than introducing signal/rollback machinery.

## Reviewer Reports

### Plan review

> ## Artifact Summary
>
> The plan is sound. Nothing in it blocks implementation, but it is over-scoped against the stated "minimal implementation" constraint. I confirmed its M2 diagnosis against the source: `tools/deidentify_dicom.py:120-131` calls `mkdir`, then reads and writes files one by one with no handling or cleanup. It prints the summary only after success (`:133-136`). It also lists every source directory a second time (`:132`). The A3 claims check out as well:
> - `_geometry.py:727-729` still says "1e-9 relative … designed but not implemented".
> - `core_interface.md:151-152` still gives `1e-9`.
> - The actual value is `8 * eps` (`_sampling.py:37`).
>
> The A4/A6 deferrals are well argued and include reopening criteria.
>
> ## Blocking Findings
>
> None.
>
> ## Non-Blocking Findings
>
> ### N1: The JSON stdout manifest and the per-category counts are more than this tool needs
> - Confidence: Medium
> - Evidence: This is a maintainer script under `tools/` (`CONTRIBUTING.md:44`). It never modifies its sources and refuses existing targets. In core conventions, audit manifests are for "destructive or multi-step workflows", and this tool is neither. The plan's step 4 still adds a JSON schema, five series-count categories, elapsed time and a "diagnostic destination". Each of those becomes something to test. Together they contradict "no speculative API".
> - Suggested change: Keep a single stderr summary that prints even on failure: series planned/published/failed/not attempted, files published, a reason per failed series, and the exit status. Drop the JSON manifest and its parse test. Drop "diagnostic destination", since stderr is the only log.
> - Would change my mind: A named consumer that parses this output, such as CI or a release script.
>
> ### N2: "Malformed-dataset failures" can't be told apart from programming errors without explicit checks
> - Confidence: High
> - Evidence: Missing data surfaces as `AttributeError`. A file without `SOPInstanceUID` fails at `:109`, and pydicom can also raise `KeyError`/`ValueError` while saving. Catching those errors to keep going also hides real bugs, which defeats the plan's rule that "unexpected programming errors still fail with their traceback".
> - Suggested change: Catch only `OSError` and `pydicom.errors.InvalidDicomError`. Check the required attributes (`SOPInstanceUID`, `file_meta`) explicitly and raise a single tool-local error for them. Everything else propagates.
> - Would change my mind: A pydicom-documented exception set for malformed-but-parseable datasets that excludes generic Python errors.
>
> ### N3: Preflight doesn't refuse an output that overlaps an input
> - Confidence: Medium (reasoning; not probed)
> - Evidence: If OUTPUT is inside a series directory, or a series is inside OUTPUT, staging and publishing write into a source tree. Snapshotting the file list once prevents reading the new files back, but the source directory still changes. That breaks the plan's source-preservation goal.
> - Suggested change: Add a preflight refusal when the resolved output root and any resolved input are equal or one contains the other, plus one boundary test.
> - Would change my mind: A decision that only file bytes count as "source preservation", not directory contents.
>
> ### N4: File selection is a decision point the plan doesn't state
> - Confidence: Medium
> - Evidence: `:121` takes every regular file. On macOS a `.DS_Store` makes `dcmread` raise `InvalidDicomError`, so under the new policy the whole series fails. That matches current behaviour, but the change makes it a reported series failure.
> - Suggested change: State the rule explicitly. Recommendation: skip dotfiles in the snapshot and treat every other file as DICOM. Ask the user if unsure.
> - Would change my mind: The user prefers strict behaviour on inputs that are already curated.
>
> ### N5: Duplicate destinations differing only in letter case get past the preflight
> - Confidence: Medium (reasoning)
> - Evidence: Comparing `resolve()` paths treats `T2` and `t2` as different. On the default case-insensitive APFS they are the same directory. The pre-publication `exists()` recheck catches the clash, but as a mid-batch failure after earlier series have published.
> - Suggested change: Compare `casefold()`ed names in the duplicate check. Don't add a test for it, since it isn't portable.
> - Would change my mind: Agreement that a refusal at publication time is acceptable here.
>
> ### N6: The plan doesn't say how the in-process test loads the script, or where the DICOM fixtures come from
> - Confidence: High
> - Evidence: `tools/` is not a package. pytest runs without `src` on the path or a `tests/__init__.py` (`pyproject.toml:104-108`). `tests/synthetic.py` has no DICOM writer.
> - Suggested change: Load the script with `importlib.util.spec_from_file_location` inside the test module. Build a tiny two-slice MR series with pydicom in a local fixture, and say in the plan that this fixture is new test-only code.
> - Would change my mind: An existing DICOM writer helper elsewhere in `tests/`.
>
> ### N7: Keep-going vs. fail-fast is a judgment call for the user
> - Confidence: Low
> - Evidence: UIDs are derived deterministically (`generate_uid(..., entropy_srcs=[uid])`, `:83`). A rerun on only the failed inputs therefore reproduces the same study and frame UIDs, so fail-fast loses nothing for consistency and is simpler. The plan's keep-going is also defensible. Separately, the plan's own reason for "one batch mapping" overstates the need.
> - Suggested change: Present it as a choice. Fix the rationale for the mapping either way.
> - Would change my mind: N/A. This is a judgment call for the user.
>
> The A3 documentation plan and the A4/A6 deferrals need no changes. The concrete gap example matches the planning check (`[4, 0]` / `[4, -9]`), and avoiding a new notebook is the right call.
>
> ## Verification Commands
>
> None needed for blocking findings. For N3:
>
> `python tools/deidentify_dicom.py SERIES/out SERIES --execute`
>
> Run this on a scratch copy and check whether the output lands inside the source directory.
>
> ## Recommendation
>
> Revise, only for N1–N3. Escalate N4 and N7 to the user as policy choices. Proceed with implementation once those are settled.

### Plan finding dispositions (before implementation)

Stage 5: DICOM per-series atomic output, complete preflight and failure reporting; cell-gap/singleton documentation. A4/A6 remain deferred.

#### Disposition of Findings
| ID | Disposition | Decision |
|---|---|---|
| N1 | Revised | Keep one compact JSON stdout manifest because supplied core conventions require a structured audit for multi-step workflows. Avoid a schema/version API or sidecar/logging framework; stderr reports the same minimal outcomes and elapsed time. Stderr is the log destination. |
| N2 | Accepted | Catch only OSError, InvalidDicomError and an explicit invalid-dataset error from required metadata checks. Generic programming errors propagate after cleanup/final summary. |
| N3 | Accepted | Resolve sources/output and refuse equal or ancestor/descendant overlap before output writes. |
| N4 | Rejected | Preserve existing strict selection of all immediate regular files, including dotfiles. Skipping files changes data selection without a requested policy change. Document that malformed/non-DICOM input fails its series. |
| N5 | Accepted | Conservatively reject casefolded destination-name collisions on every platform; document/test this deterministic policy. |
| N6 | Accepted | File-path import of script and a minimal local pydicom fixture; no fixtures/source path changes in existing adapter tests. |
| N7 | Retained agreed plan | The user authorized the concrete plan's continue-independent-series behavior. No fresh policy question is needed. Retain deterministic batch cache and correct its rationale: retry subsets get identical UID remapping. |

#### Plan-phase disagreements

Reviewer recommends dropping JSON without a parser consumer. Designer retains only a compact stdout manifest under the explicit supplied core convention, avoiding persistence/schema machinery. This is nonblocking and bounded; implementation review will assess resulting complexity. Strict file selection and keep-going were already grounded in the existing/approved contract.

#### Plan-phase recommendation

Proceed with revised implementation; no unresolved blocking concern.


## Implementation Review Dispositions

Opus recommends **Accept** with no blockers. One implementation pass was sufficient; accepted
follow-ups change documentation and test strength, not runtime behavior.

| ID | Disposition | Resolution |
|---|---|---|
| N1 | Accepted | Update audit/review status and record final QA results. |
| N2 | Accepted documentation option | A post-rename interruption can leave a complete target before completion status is recorded; the report may need reconciliation. No signal or rollback machinery. |
| N3 | Accepted | Injected-failure tests assert category, operation and source/file/target attribution. |
| N4 | Accepted | Per-fixture series counter replaces name-length UIDs, preserving shared study/frame while making instances/series distinct. |
| N5 | Accepted | Remove two new blank list separators in public changelog. |
| N6 | Accepted | Qualify manifest emission as after successful argument parsing; ordinary argparse help/usage semantics remain intact. |

### Implementation review

> ## Artifact Summary
>
> The M2 implementation does what the plan promised, and I found nothing blocking. I checked these points against the source:
>
> - **Preflight before writes.** Preflight finishes before any output write (`main` gates execution on `not failures`).
> - **Per-series staging.** Each series is staged in a sibling `TemporaryDirectory` and published with a single same-parent `rename`.
> - **Cleanup on exit.** The context manager cleans up staging on every exit path. On the success path the staging directory is gone after the rename. CPython's `TemporaryDirectory._rmtree` ignores `FileNotFoundError`, so cleanup still exits cleanly there.
> - **Narrow catch set.** Only `OSError`, `InvalidDicomError` and `InvalidDatasetError` are caught.
> - **Failure reporting.** Unexpected errors and interruptions are recorded in `finally` through `sys.exception()`; `requires-python >=3.12` at `pyproject.toml:10` makes that safe.
> - **File counts.** Published counts come only from series whose outcome is `completed`. Staging files are never counted.
> - **Duplicate validation.** `_validate_dataset` runs before `_deidentify`, so a missing `SOPInstanceUID` can no longer surface as a generic `AttributeError`.
>
> The A3 docstring claim also holds. "Declared interval bounds when present" matches `_sampling.py:184-187`, which takes the outer bounds from the intervals' min and max, so interior gaps are not excluded.
>
> The tests cover preflight boundaries well: 17 cases, with symlink-preserving snapshots. They also cover injected write, publish, unexpected and interrupt failures.
>
> ## Blocking Findings
>
> None.
>
> ## Non-Blocking Findings
>
> ### N1: The audit status docs still say Stage 5 has not started
> - Confidence: High
> - Evidence: `docs/dev/codebase_audit_plan.md` (diff, line ~13) says "implementation has not started". Deferred Work says "Stage 5 implementation remains pending". Next Steps says "The next implementation task can start with M2's CLI regressions…". `stage5_implementation_review.md` §Verification says "Pending implementation". All of these are now false. This is the stale-note problem the dev-note convention warns about.
> - Suggested change: Update the status lines in both notes to "implemented; QA pending". The orchestrator could fold this into the planned post-QA record update rather than a separate edit.
> - Would change my mind: If the records are deliberately frozen until full QA lands and that update is already scheduled, this is moot.
>
> ### N2: An interrupt between rename and `outcome = "completed"` misreports a published series
> - Confidence: Medium. This is reasoning; the window is one bytecode statement.
> - Evidence: `_publish_series` calls `staging.rename(run.target)` and only then sets `run.outcome = "completed"`. A `KeyboardInterrupt` that lands between those two statements is recorded as `failed`, with `published_files: 0`, even though `run.target` exists complete on disk. The changelog says counts are faithful "even after … interruption", and here they are not. A retry would then refuse the existing target, so the error at least surfaces.
> - Suggested change: Accept the window and document it in the one existing "no crash-durability" sentence. Alternatively, in `finally`, mark a failed run `completed` when `run.operation == "publish"` and the target exists and the staging path does not. I would take the documentation route.
> - Would change my mind: A judgement that a one-statement window is below the bar for a maintainer tool.
>
> ### N3: The unexpected/interrupt tests don't assert the category or which series is attributed
> - Confidence: High
> - Evidence: `test_injected_failure_cleans_staging_and_always_summarizes` asserts `failed_series == 1`. It does not check that `failures[0]["category"]` is `"unexpected"` or `"interrupted"`, or that `failures[0]["input"]` names the broken source. A regression that recorded the error as preflight (`run=None`) while some other path marked the run failed could still pass.
> - Suggested change: Add one assertion on `manifest["failures"][-1]["category"]` and its `input` field.
> - Would change my mind: Evidence that another test already covers category attribution for these two paths. I found none.
>
> ### N4: Fixture UIDs depend on `len(name)` and silently collide across series
> - Confidence: High
> - Evidence: In `series_factory`, `SOPInstanceUID = f"…100.{len(name)}.{index+1}"` and `SeriesInstanceUID = f"…102.{len(name)}"`.
>   - `001-first` and `003-first` would share UIDs.
>   - `001-broken` and `002-second` are both 10 characters, so they would too.
>   - The current tests are safe: those pairs either fail in preflight or never appear together.
>   - A future success test that pairs two same-length names would build sources whose series share UIDs. Its UID-consistency assertions would then prove less than they appear to.
> - Suggested change: Use a per-call counter, or pass the series index explicitly.
> - Would change my mind: None. This is a low-cost robustness fix, not a current failure.
>
> ### N5: Loose-list blank lines in CHANGELOG
> - Confidence: High
> - Evidence: Both new entries are followed by a blank line before the existing bullets ("…unchanged.\n\n- Clarified indexed…", "…single writer.\n\n- NGFF…"). Neighbouring entries are tight, so each section now renders as a loose list.
> - Suggested change: Remove the two blank lines.
> - Would change my mind: If the repository deliberately separates newer entries this way. Nearby history doesn't show that.
>
> ### N6: The docstring promises a manifest on every run, but usage errors produce none
> - Confidence: High
> - Evidence: The module docstring says stdout "contains one JSON run manifest … including on preflight failure or interruption". `parser.parse_args()` runs before the `try`, so a bad CLI exits with code 2, prints argparse usage, and writes no stdout.
> - Suggested change: Add "after argument parsing" to the docstring sentence. Don't add code.
> - Would change my mind: If argparse errors are understood as outside "a run".
>
> **Policy points I am not reopening:** strict selection of dotfiles, keep-going, the JSON manifest and single-writer output are all settled. On the earlier complexity concern, the implementation stays reasonably compact (`_report` is about 50 lines). I withdraw it.
>
> ## Verification Commands
>
> None for blocking findings.
> - **N2:** inspect only; the window is not reliably testable without injecting a failure at the statement after `rename`.
> - **N3:** `pytest tests/tools -k injected -q` after adding the category assertion.
>
> ## Recommendation
>
> Accept. Before the final record, fix the documentation and test nits N1, N3 and N5, and choose the documentation wording for N2 and N6. None of these needs human escalation.
