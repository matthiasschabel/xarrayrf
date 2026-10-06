# Codebase audit and staged remediation plan

**Status:** Implemented
**Last updated:** 2026-10-06
**Scope:** xarrayrf correctness, public API consistency, adapters, resampling, and supporting tools

## Context

The user requested a thorough audit emphasizing material correctness, comprehensible APIs,
and the principle of least surprise, followed by critique from Claude Opus 5.5. This plan
covers the checkout at `de48d8ab0097bf6196fdcbced0779de0ae155fea`. Stage 1 has been implemented, cross-reviewed, verified by QA and committed to main;
stages 2–4 are also implemented, cross-reviewed, verified by QA and committed to main.
Stage 5 is implemented, accepted by Opus 5.5 and independently verified; A4/A6 are deferred
with concrete usage triggers. The audit remediation is complete within its agreed scope.

The audit inspected the shipped modules, native xarray binding, sampling and resampling,
format adapters and readers, public contracts, test coverage, and supporting tooling.
Two collaborative-refinement passes used Claude Code with explicit model selection
`--model claude-opus-5-5` and read-only plan mode. Claude checked source rather than running
the reproductions. Codex independently re-ran the public reproductions and verified disputed
claims. The final reviewer recommendation was **Accept**, with no blocking disagreements.
See [the refinement record](codebase_audit_refinement.md) for findings and dispositions.

Baseline verification from the audit:

| Check | Result |
|---|---|
| Stock xarray pytest lane | 1,617 passed, 47 xfailed, 3 warnings |
| Patched xarray pytest lane | 1,664 passed, 3 warnings |
| Ruff lint and formatting | Passed |
| Mypy | Passed, 81 source files |

These checks establish the baseline, not correctness of the unimplemented fixes. Synthetic
public reproductions exposed the defects below despite the passing suite. The review did
not repeat every real-data example or notebook.

## Current Decision

**Stage 1 completed, 2026-10-05:** F1, F2, and F8 are implemented with public regressions,
updated contracts, and changelog entries. Two Claude Opus 5.5 implementation-review passes
ended in **Accept**, with all findings resolved. Final verification: stock xarray 1,638 passed
and 47 expected failures; patched xarray 1,685 passed; Ruff lint/format and mypy passed.
Both lanes retain the three existing dependency warnings. See
[the stage 1 implementation record](stage1_implementation_review.md) for scope, evidence,
compatibility choices, and the documented non-affine inverse limitation.
The subsequent QAEngineer pass reproduced the original failures independently, exercised
mixed-axis boundaries and repeated both full suites and static checks without finding an
additional defect. The [internal changelog](changelog.md) records that verification.

**Stage 2 completed, 2026-10-05:** Opus 5.5 accepted the expanded F6/F7/M1 plan, gpt-6-astra
implemented it in an isolated worktree, and Opus accepted the implementation after its three
optional follow-ups were resolved. The reviewed diff is integrated, verified by QA and committed to main. Final
independent verification: stock xarray 1,696 passed and 47 expected failures; patched xarray
1,743 passed; Ruff lint/format, mypy and whitespace passed. Both lanes retain three existing
dependency warnings. See [the stage 2 implementation record](stage2_implementation_review.md)
for decisions, complete reviewer reports and verification.
The subsequent QAEngineer pass reproduced 40 original scalar-target failures, checked
nonlinear registration and coordinate-only Dask fields, and repeated both full suites and
static checks without finding an additional defect.

**Stage 3 completed, 2026-10-05:** F3/F4 are implemented by gpt-6-astra, reviewed by Opus 5.5
and independently QA-verified. The reviewed files were integrated byte-for-byte; the fixes
and shared records are committed to main separately. The two plan passes resolved the blockers;
measured index errors settled the remaining rounding-bound choice. Both implementation passes
ended in **Accept**, with one optional performance optimization explicitly deferred.
Independent verification: stock xarray 1,739 passed and 47 expected failures; patched xarray
1,786 passed. Ruff lint/format, mypy and whitespace pass; three existing dependency warnings
remain in each lane. Public regressions fail 33 selected cases against the original source.
See [the stage 3 implementation record](stage3_implementation_review.md) for complete reviewer
reports, decisions, measured bounds and QA evidence.
The subsequent QAEngineer pass reproduces 34 failures against the original source, verifies
32 independent interpolation-oracle cases, scalar complex accessor behavior and NGFF semantic
round trips, and repeats both full suites and static checks. It finds no additional defect;
the [internal changelog](changelog.md) and stage 3 review record retain the fresh evidence.

**Stage 4 completed, 2026-10-06:** The user explicitly authorized direct clean API changes
without compatibility/deprecation scaffolding during development. A1 chooses declared geometry
dimension order across queries and snapshots, with explicit storage order for NGFF export.
A2 adds `replace_coordinates=False`: conflicting Grid coordinate declarations refuse unless
replacement is explicit, with per-coordinate diagnostics. A5 documents stable extension
transforms and conditional hashing; F5 documents destination-frame payload assignment under
xarray's label checks. Opus accepted the plan and both implementation passes; gpt-6-astra
implemented and resolved the useful followups. Independent QA reproduces 15 failures against
the original source and checks eager/Dask queries, native round trips, NGFF physical mappings,
replacement pixel identity and metadata. Final suites: stock xarray 1,764 passed and 47 expected
failures; patched xarray 1,811 passed, with three existing dependency warnings in both lanes.
Ruff lint/format, mypy and whitespace pass. See
[the stage 4 implementation record](stage4_implementation_review.md) and
[internal changelog](changelog.md) for full review/verification evidence.
The subsequent QAEngineer pass repeats both full lanes and static checks, passes 72 independent
ordering/export probes and boundary replacement checks, and clarifies the retained-scalar
assignment boundary with a public test. Final totals are stock 1,765 passed plus 47 expected
failures and patched 1,812 passed. No production source correction was needed.


**Stage 5 completed, 2026-10-06:** M2 preflights the complete DICOM batch, stages each series
beside its final destination, cleans unpublished staging and reports partial failure without
counting discarded files. Explicit metadata checks preserve narrow exception handling. A3
and singleton documentation now agree with existing sampling behavior; A4/A6 retain reasoned
deferrals. Opus reviewed the plan and accepted the implementation; all useful findings are
resolved, with the publication-interruption reporting window explicitly documented.
Independent QA passes 12 CLI scenarios, the gap/edge example, active-exception invocation and
executable-AST checks confirming no geometry runtime changes. Full suites: stock 1,793 passed
plus 47 existing expected failures; patched 1,840 passed; three existing warnings each. Ruff,
formatting, mypy (82 files), whitespace and distribution payload checks pass. See
[the Stage 5 refinement and QA record](stage5_implementation_review.md).

Keep the architecture: frame identity, coordinate systems, transforms, sampling, and array
binding form a coherent model. Preserve explicit registration between distinct frames,
separation of identity from numerical coincidence, and coordinate values that survive
cropping. Fix the concentrated defects and clarify public contracts. A wrapper hierarchy,
global frame registry, or replacement of native xarray binding is not justified.

### Findings and classification

| ID | Finding | Final classification |
|---|---|---|
| F1 | Relative NGFF paths can identify different stores as the same frame, while absolute and relative openings of one store disagree. | Confirmed identity defect; first priority. |
| F2 | A singleton at 1,800,000,000 seconds accepts a query one second later as the same sample. | Confirmed faulty tolerance contract; first priority. |
| F8 | A classic DICOM geometry for two slices accepts three pixel slices and silently discards the third. | Confirmed cardinality defect; preserve enhanced subset selection. |
| F6 | Resampling onto a valid scalar Grid or selected scalar DataArray fails in reshape logic. | Confirmed valid-input failure; fixed in stage 2. |
| F7 | Geometry.points collides with axis/units output names and its private stacking dimension. | Confirmed name collisions and units-metadata loss; fixed in stage 2. |
| F3 | Non-diagonal NGFF export labels an unmixed time axis as spatial and pairs column axes with row units by index. | Fixed in stage 3 with conservative representability rules and contextual v06 layout errors. |
| F4 | Equivalent resampling paths disagree on NaNs at exact sample locations. | Linear zero-weight NaN contamination fixed in stage 3; cubic prefilter limitation explicitly documented. |
| F5 | Indexed assignment accepts another frame's labelled payload while preserving the destination frame. | Intentional payload replacement; documented and tested in stage 4, with destination frame retained and xarray label checks enforced. |
| A1 | Positional queries and dense point/lattice outputs used different dimension orders. | Unified on declared sampling order in stage 4; exporters request storage order explicitly. |
| A2 | Grid framing silently replaced conflicting coordinates. | Stage 4 requires explicit replacement and reports what disagrees. |
| A5 | Grid/binding docs overstated immutability and unconditional hashing for extensions. | Stage 4 states stable transform value contracts and conditional hashing. |

The order below is the recommended default. Each fix is independently reviewable. A public
interface decision in one item must not hold up an unrelated, straightforward fix.

### Stage 1: Stop silent identity and data-association errors

**F1: Canonicalize local NGFF reader identities.** Resolve filesystem paths to a bare absolute
path, including symlink resolution. Prefer this spelling over introducing `file://` because
it retains the existing identity form for canonical absolute-path callers. Use that exact
location when constructing frames and passing the reader's store identity to metadata import.
URLs keep their supplied identity spelling.

Metadata-only `store=` remains a caller-supplied resolved location, accepting documented
resolved URIs or absolute filesystem locations. It must not acquire implicit filesystem
access or working-directory dependence. Clarify the distinction between a reader input path
and a resolved metadata identity. Do not silently reinterpret identities already persisted
using relative paths; document the change and explicit frame adoption for workflows that
need to retain an existing identity.

Acceptance checks:

- Different directories containing the same basename produce different frames and refuse
  incompatible arithmetic.
- Relative, absolute, and symlink openings of one physical store agree.
- Explicit reader `frame=` adoption still works across path spellings.
- Metadata-only import using the documented canonical location can reuse a reader's frame
  through `resolved_frames`.
- Existing URL handling and group/system-name distinctions remain covered.

**F2: Replace the singleton support window with a measured roundoff bound.** The current
`1e-9 * max(1, abs(sample))` creates a 1.8-second window at Unix-scale coordinates. Use a small,
justified multiple of float64 rounding error, considering inverse-affine intermediate values
and translations where available. A bound based only on the final coordinate can reject
valid round trips after cancellation. Do not choose a smaller constant without measuring
representative public forward/inverse and composite-transform cases.

Acceptance checks: reject the reproduced one-second mismatch in position lookup, coincidence,
and resampling; preserve measured round trips; keep singleton samples distinct from declared
cell support. Replace the existing test permitting `1000 + 5e-7` and update the documented
contract. Do not require rejection of a one-microsecond difference at epoch scale before
establishing the bound: it is only about four float64 ulps.

**F8: Validate DICOM source cardinality before sorting or selection.** Preserve the original
source count from the classic dataset sequence or enhanced `NumberOfFrames`. Recommend a
final, defaulted `source_count` field on `DicomGeometry`, preserving existing constructor call
shapes. Importer-produced objects always carry the exact count. For legacy manually built
objects with unknown count, retain the existing minimum-index validation initially and
state that limitation; do not infer enhanced original count from selected indices.

Acceptance checks: reject surplus and missing slices for imported classic geometries; require
the complete original enhanced stack and preserve selected/sorted subsets; validate lazy
array shapes without computing pixels. Document the dataclass field and equality implications.

### Stage 2: Repair failures on valid inputs

**F6: Support scalar resampling targets throughout the pipeline.** Handle zero varying target
dimensions in crop-window construction, outside masks, block positions, and result reshaping.
The output contains one sampled value, retaining unrelated source dimensions. Do not patch
only the first reshape error or conflate scalar targets with unsupported scalar-source inversion.
The implementation preserves affine crop planning and promotes scalar output to one internal
sample only within the numerical kernel, reusing outside/cubic-shell handling. The general
path constructs zero-row position arrays; public output keeps shape `()`.

Acceptance checks: scalar Grid and scalar framed DataArray targets, eager and Dask outputs,
representative affine and general mappings, outside fill, and unrelated leading dimensions.

**F7: Disambiguate labelled point-result dimensions and units.** Keep valid input dimensions
such as `axis` and `units`. Dense output now accepts
`points(*, axis_dim="axis", units_coord="units")` through the existing geometry accessor,
with actionable conflict errors. The private stacking dimension is removed; broadcast inputs
are stacked within the evaluation callback. Preserve ordinary default output, labels, units
and lazy field support. `point_at` retains its fixed names and unrestricted dimension indexers;
use xarray `rename` to align a selected point with customized dense output.
`Grid.points` returns a NumPy array and has no reusable labelled-dimension naming mechanism.
Do not convert through Grid eagerly and lose Dask or nonseparable coordinate support.

Acceptance checks: a geometry varying over a dimension named `axis` can produce correctly
labelled points using the documented disambiguation; ordinary outputs retain their labels,
units, ordering, and lazy evaluation.

**M1** is completed in stage 2: Geometry documentation now distinguishes a standalone query
view from the existing native accessor and binding lifecycle.

### Stage 3: Correct export semantics and narrow interpolation behavior (completed)

**F3: Preserve representable NGFF axis semantics; refuse the rest.** Non-diagonal intrinsic
columns inherit the type and unit of their contributing physical rows. Mixing is supported
only for explicitly spatial rows with the same canonical declared unit; numeric scale
conversion is not introduced. Exact structural zeros define support. Unsupported semantics
and upstream v06 axis-order/count failures report actionable context and preserve the cause.
Array transposition and physical frame reordering are distinguished; export does not reorder
pixels. Tests verify metadata types/units, physical sample mappings, canonical spelling aliases,
invalid mixtures/layouts, orientation/loss reports and lazy pixels.

**F4: Correct linear zero-weight NaN behavior; document cubic separately.** Linear resampling
interpolates zero-filled values and component-wise NaN masks with the same kernel/boundaries,
fixing fully integral and partially integral queries across affine and general paths. Buffers
are prepared once per slice and held for one slice at a time. The shared `1e-9` gather and
missing-weight allowance replaces `1e-6`, justified by measured oblique round trips up to
4.66e-10 index error. Missing weights above the bound propagate; smaller deficits are not
renormalized. Public tests cover complex components, large-origin affine mask queries,
positive missing weights, scalar targets, eager/Dask output, cropped reads, cells and fill.

Cubic prefiltering can spread a NaN through coefficients, while same-grid gather retains
original samples. That consistency limitation and SciPy infinity behavior remain explicitly
documented and tested. General-path NaN tasks retain bounded position caching or per-slice
recomputation; optional routing optimization awaits profiling.

### Stage 4: Make API contracts consistent (completed)

- **A1, dimension ordering:** declared geometry dimensions govern positional query input/output,
  dense points, default lattice columns, frame-coordinate fields and Grid snapshots. Pixel
  transpose preserves binding order. Use named xarray transpose and explicit lattice dimensions
  for a consumer's storage order; NGFF export requests actual pixel-axis order itself.
- **A2, coordinate replacement:** `rf.frame(grid, replace_coordinates=False)` supplies missing
  coordinates and retains compatible metadata. Conflicting dimensions, values, dtype kind or
  explicit unit attrs refuse with coordinate-specific reasons. Explicit replacement adopts the
  Grid declarations, discarding stale metadata and repairing malformed units while sharing
  pixels. The flag is boolean; True with a transform refuses. No resampling/conversion occurs.
- **A5, extension transforms:** endpoints, behavior and scalar equality must stay stable when
  retained by Grid or binding. Hashing is conditional and equality-consistent. Grid freezes its
  own coordinates/intervals and retains transforms by reference; composites depend on members.
  Unhashable extensions remain valid for querying, equality and binding. No generic freezer.
- **F5, indexed assignment:** replaces destination-frame values when xarray's dimension-label
  checks permit it; it neither adopts nor resamples the right operand's frame. Retained scalar
  coordinates are ignored by those checks, even when their values differ. Documentation and
  public tests clarify the existing runtime behavior; no upstream guard was added.

The direct public contracts and concise usage examples replace the earlier ambiguity.
Focused public tests and full QA verify these deliberate API choices. No deprecation wrappers,
legacy aliases, migration modes or encoding schema bump were introduced.

### Stage 5: Improve usability and tool reliability

**Implemented scope, 2026-10-06:** M2 and A3 plus remaining singleton documentation are
complete. A4/A6 are deferred with the evidence and reopening criteria below. This is the final
audit stage; viewer, upstream and release work remain separate. The plan below was refined
and implemented under the user's subsequent authorization; final evidence is in the Stage 5
review record. No external publication is part of this work.

#### Reconciled starting state

The local untracked continuation handoff was read after the supplied core conventions,
`CONTRIBUTING.md` and the project's design/interface guidance. It remains preserved outside
the implementation commits. Root
`AGENTS.md` remains absent. Read-only checks found:

- Main remains `df385d3d762f3ef6cb11b83343b3e9db41c5ffb9`. Initially it was 16 commits ahead
  of live `origin/main` at `de48d8ab0097bf6196fdcbced0779de0ae155fea`; a concurrent push during
  planning advanced both live origin and the tracking ref to `df385d3`, confirmed on final
  recheck. This agent performed no push. The index is empty; no stashes, conflicts or
  additional xarrayrf worktrees are present.
- The same user edits remain in `docs/dev/README.md` and
  `docs/dev/architecture/relativity_notes.md`; the curved-spacetime review and handoff remain
  untracked. Preserve them and leave the handoff unchanged.
- All six xarray dependency/PR worktrees retain their recorded revisions and clean working
  trees. Patched xarray remains `5db75d53f5515fabb3ca5ffc8419eb9c982cf4a2`; local PR #11621
  remains behind its remote head. No shared xarray stashes are present.
- GitHub reports no open xarrayrf PRs. Initially no Actions were queued/in progress; final
  recheck after the concurrent push found checks run `37493729018` queued at `df385d3`.
  Its result is pending, not local validation evidence. Upstream #11616 remains merged,
  #11617 open, and #11621 open with changes requested at the recorded remote head.
  These dependencies do not block the proposed Stage 5 scope.
- Stock `.venv` imports xarray 2026.7.0 and pydicom 3.0.2; `.venv-patched` imports
  2026.7.1.dev80+xarrayrf.patches.4. Stage 4's recorded full verification covers the unchanged
  production revision: 1,765 stock passes plus 47 expected failures, and 1,812 patched passes.
  Those full suites were not rerun for planning. Hosted checks on the older remote main
  failed; they are not validation of the local audit commits.
- Arbitrary OS jobs remain unknown because `ps` is sandbox-denied. No duplicate review,
  external runner or long-running job was launched.

#### M2: DICOM output reliability

Temporary synthetic CLI probes established the failure before implementation:

| Case | Original result |
|---|---|
| Dry run | Exit 0, no output created, summary present. |
| Invalid second DICOM file | Exit 1 after publishing the first file in the final series directory; no batch summary. |
| Injected second-file write failure | Uncaught `OSError`; final directory contains one completed file and one partial file; no batch summary. |
| Two input names mapping to `series` | First series published before the second destination creation fails; no batch summary. |
| A later destination already exists | Earlier series published before refusal; no batch summary. |

All original probes left source bytes unchanged. The implementation now has 28 CLI regressions.

Implemented in `tools/deidentify_dicom.py` with CLI tests in
`tests/tools/test_deidentify_dicom.py`. Keep the existing UID remapping and tag policy.

1. **Preflight the complete batch before any output writes.** Snapshot each input's sorted
   immediate file list once; validate readable directory inputs and refuse empty series.
   Keep the current output-name rule (suffix after the first hyphen), but refuse empty,
   `.` or `..` names. Compare resolved destinations and casefolded output names; refuse duplicate
   inputs/destinations, every existing target including dangling symlinks, and invalid
   output-parent paths. Refuse overlap between the resolved output root and an input
   directory in either direction. Preserve strict selection of every immediate regular
   file, including dotfiles: a non-DICOM file causes a reported series failure.
   A preflight failure rejects the whole batch before publishing any series. Dry run uses
   the same plan and validations, prints every mapping/count, and creates no files or directories.
   It validates paths and enumeration; reading/validating all DICOM payloads remains execution work.
2. **Stage one series at a time.** Create a unique temporary directory next to its final
   directory, write all numbered DICOM copies there, then rename only the completed directory
   into place. Recheck destination absence immediately before publication. Use exception-safe
   cleanup for every unpublished staging directory, including failed writes and interruptions.
   Files must close before publication. Same-parent rename supplies per-series atomic visibility;
   no whole-batch transaction or crash-durability guarantee is promised. A hard kill can leave
   an unpublished temporary directory; a subsequent run must not mistake it for final output.
   The output tree has one writer per run; concurrent writers require a separately scoped
   no-replace publication mechanism rather than assuming directory rename cannot overwrite.
3. **Make batch failure explicit.** After successful preflight, continue independent later
   series after an expected read, de-identification, write or publication failure. Retain
   earlier completed series. Catch `OSError`, `pydicom.errors.InvalidDicomError` and an explicit tool-local
   invalid-dataset error after checking required metadata. Do not catch generic
   `AttributeError`, `KeyError`, `TypeError` or `ValueError` around the whole operation;
   unexpected programming errors still fail with their traceback.
   Identify the source file, target and operation in diagnostics. Keep the existing batch UID mapping as a cache. Deterministic UID derivation
   preserves shared identities even when only failed series are retried.
4. **Report attempted and published work separately.** Emit from finalization, including
   unexpected errors and interruptions before re-raising, a
   stderr summary: input/output paths, planned/completed/failed/not-attempted series,
   planned/published file counts, categorized failures and elapsed time. Stderr is the log
   destination. Keep the JSON manifest limited to these outcomes, without a new schema API.
   Dry-run counts mean planned work; staging files discarded on failure are never counted as
   published. Return nonzero for any rejected or failed series. Emit a compact JSON run
   manifest on stdout for redirection, with the same per-series outcomes and counts; keep
   human diagnostics and tracebacks on stderr, identified as the log destination. This
   satisfies batch auditing without creating sidecar files during dry run or adding a logging
   subsystem. Do not include the original-to-remapped UID mapping in the manifest.

Acceptance tests exercise the script entry point, using subprocesses for ordinary CLI behavior
and the entry point with dependency failure injection for deterministic write/publication errors.
Load the script by file path with `importlib.util.spec_from_file_location`; build small
explicit-VR MR files in a local test fixture, without changing existing adapter fixtures:

- Dry run leaves even an absent output root absent, with matching planned paths and counts.
- Duplicate destination names and a later existing target refuse before any series publishes;
  cover dangling links, invalid/empty output names, casefolded name collisions and
  source/output overlap as boundary checks.
- Failure on the second read or a write that creates partial bytes leaves no final failed
  series and no staging directory; pre-existing output and all source bytes remain unchanged.
- A successful series, failed series and later successful series produce two complete finals,
  accurate published counts, contextual failure diagnostics and nonzero exit status.
- Publication failure cleans staging without deleting an existing destination. Unexpected
  errors/interruptions also clean staging and never report success.
- A successful two-series run preserves geometry and pixel bytes, consistent shared study/frame
  UID remapping, registered class/transfer-syntax UIDs and SOP/file-meta UID agreement. Check
  representative tag replacement/private-tag removal through saved files, without broadening
  the confidentiality policy. JSON stdout parses, stderr summaries agree and success exits 0.

#### A3 and singleton documentation reconciliation

The normative query/resampling sections already describe interpolation across cell gaps, and
`test_gapped_and_overlapping_cells_grid_geometry_agree` already exercises gap lookup. The remaining
discrepancies were documentation, corrected with existing behavior as the oracle:

- Clarify the README's declared-cells paragraph and `Geometry.positions_at`/`Grid.positions_at`
  help: cells extend the outer domain; interior gaps remain interpolated.
- Add one concrete example: samples at 0 and 4 mm, intervals `[-0.5, 0.5]` and `[3.5, 4.5]`,
  values 0 and 8. Querying 2 mm gives position 0.5 and linear value 4 despite the acquisition
  gap; -0.25 mm is admitted only by the cells domain and holds the edge value 0. Prefer a
  compact README explanation over another notebook or a new support mode.
- Correct `Geometry.is_coincident`'s obsolete `1e-9 relative`/unimplemented-cell-width language.
  Singleton sample coincidence uses float64 roundoff and is not widened by declared cells.
- Also correct `docs/core_interface.md`'s **Exported names and constants** paragraph: it still
  gives `SINGLE_SAMPLE_TOLERANCE` as `1e-9`, although the implementation and detailed singleton
  section use `8 * eps` plus the implemented affine intermediate-term allowance. This is an
  additional stale passage discovered on pickup, not a numerical-policy change.

Validate the concrete example against queries and resampling and run existing interval and
singleton regressions. Add a behavior test only if the example exposes an uncovered contract;
do not add tests that merely repeat documentation wording. Completed review transcripts remain
historical records, not another status queue.

#### A4/A6: deferrals and reopening criteria

**A4, adapter grids and reader reports:** metadata results already expose `dims`, `coords`,
`transform` and reports; DICOM adds ordering/intervals, NGFF separates levels from multiscale
reports/transforms, and GeoTIFF adds nodata. Ordinary readers discard those import result
objects and return framed arrays whose Grid is already available as `array.rf.grid`.
Inventory of README/examples/tour/real-data tooling found no consumer of `.report` needing a
new channel. A common metadata Grid property would need to preserve format-specific nongeometry
coordinates, DICOM source ordering and intervals, and NGFF level choice; it cannot replace
those results indiscriminately. Defer new accessors/report storage until a concrete caller
needs a metadata-only Grid or a report from an ordinary reader. At that point propose the
smallest consistent contract preserving provenance, DataArray returns and pixel laziness;
do not put an adapter report in core geometry merely to make it visible.

**A6, construction and mixed coincidence:** a Dask `(channel, i)` array with no channel labels
currently fails `frame_array(..., dims=("channel", "i"))` because no coordinate declares the
channel size. The existing `xr.DataArray(data, dims=("channel", "i")).rf.frame(grid)` works,
shares the original pixel object and computes no Dask tasks. Likewise
`grid.is_coincident(array.rf.geometry)` refuses the concrete type, while
`grid.is_coincident(array.rf.grid)` succeeds without computing pixels in the small separable
probe. These are demonstrated convenience restrictions, not blocked correctness cases.
Defer both changes until a caller needs the shorter construction door or comparison without
snapshot materialization. A Geometry snapshot can materialize coordinates, so the latter is
not a blanket laziness guarantee for field coordinates. Reopening requires a reviewed public
example and focused validation of shapes, laziness, explicit failures, frame identity and
existing step/singleton tolerances; no sampling-protocol hierarchy is implied.

Planning verification passed 17 existing interval tests selected with
`-k 'gapped_and_overlapping or singleton'` (72 deselected). The native
`source.rf.resample_to(...)` example returned `[4, 0]` in the cells domain and `[4, -9]` in
the samples domain with fill -9, checked at absolute tolerance `1e-12` for small synthetic
affine/interpolation roundoff. Local document links and whitespace checks passed. These
checks validate the proposed documentation against current behavior, not the unimplemented M2 fix.

#### Completed execution order and acceptance gates

1. Isolate implementation from the preserved user edits on a feature branch/worktree; add
   CLI regressions and demonstrate the M2 failures against `df385d3` before fixing them.
2. Implement preflight, staging, cleanup and outcome reporting; pass the focused CLI tests.
3. Reconcile public docs, execute the gapped example and pass existing interval/singleton
   coverage. Update `CHANGELOG.md`, the audit status and the internal QA changelog with actual
   results; record A4/A6 as deferred rather than implemented.
4. Run Ruff lint/format, mypy, whitespace checks and both full xarrayrf lanes once the final
   diff is settled. Recheck the patched dependency SHA first. Use the handoff's commands;
   add the new focused test path to the narrow run. Preserve all 47 existing stock expected
   failures and require no new failures or unexplained warnings. No notebook/build rerun is
   needed unless the resulting changes touch their behavior or inputs.

Stage 5 satisfies these failure/success checks; public documentation agrees with sampling
policy, A4/A6 have explicit dispositions, and final review/QA evidence is recorded. Independent model review/QA workflows run only when invoked; consult
their current skills and available model selectors then. No schema bump, upstream worktree
mutation or release publication follows from this plan.

### Verification and release discipline

For each implemented fix, first establish a failing public regression, then make it pass.
Use focused eager/lazy cases where behavior actually differs, not a combinatorial test matrix.
Once shared changes settle, run both stock and patched xarray suites, Ruff lint/format, and
mypy. Do not weaken existing coverage. Update public documentation and `CHANGELOG.md` for
identity, tolerance, dataclass, signature, support, and refusal changes. No encoding schema
bump follows automatically from this plan.

## Alternatives Considered

A broad architectural cleanup would increase migration risk without addressing the confirmed
failures. A generic NaN-aware interpolator or unit-analysis layer would make narrow fixes too
large. Treating documented policies as implementation bugs would also obscure the actual
product choices. Conservative refusal and explicit contracts are preferable where semantics
cannot be represented safely.

## Deferred Work

Stage 5 is implemented and independently verified. A4/A6 remain deferred unless their stated
usage triggers arise. Stage 4 public contracts
are settled and implemented without backward-compatibility scaffolding. Cubic path dependence with NaNs remains an explicit
limitation unless a later interpolation policy change is justified. Stage 3's optional
many-context-slice routing optimization awaits profiling above the position cache budget.
Stages 1–4 are committed to main and now present on origin after the concurrent push observed
during planning. This agent made no remote changes; the user's unrelated uncommitted work is preserved.

## Next Steps

Stages 1–5 are complete within the agreed audit scope. Retain the acceptance checks and
review records as evidence; no further audit stage is queued. Resume A4/A6 only when their
usage triggers arise. Viewer, upstream, minimum-dependency CI and release work need their
own scope. Preserve the unrelated user documentation and the historical handoff.
