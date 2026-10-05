# Codebase audit and staged remediation plan

**Status:** Active
**Last updated:** 2026-10-05
**Scope:** xarrayrf correctness, public API consistency, adapters, resampling, and supporting tools

## Context

The user requested a thorough audit emphasizing material correctness, comprehensible APIs,
and the principle of least surprise, followed by critique from Claude Opus 5.5. This plan
covers the checkout at `de48d8ab0097bf6196fdcbced0779de0ae155fea`. Stage 1 has been implemented, cross-reviewed, verified by QA and committed to main;
stages 2–5 remain recommendations.

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
| F6 | Resampling onto a valid scalar Grid or selected scalar DataArray fails in reshape logic. | Confirmed valid-input failure. |
| F7 | Geometry.points fails when a varying input dimension is named axis. | Confirmed dimension-name collision. |
| F3 | Non-diagonal NGFF export labels an unmixed time axis as spatial and pairs column axes with row units by index. | Confirmed metadata defect; use conservative representability rules. |
| F4 | Equivalent resampling paths disagree on NaNs at exact sample locations. | Linear exact-sample behavior needs correction; cubic prefilter behavior needs an explicit contract. |
| F5 | Indexed assignment accepts another frame's labelled payload while preserving the destination frame. | Intentional payload replacement; documentation, not a runtime bug fix. |

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

Acceptance checks: scalar Grid and scalar framed DataArray targets, eager and Dask outputs,
representative affine and general mappings, outside fill, and unrelated leading dimensions.

**F7: Disambiguate labelled point-result dimensions.** Keep valid input dimensions such as
`axis`. Use collision-safe internal dimensions and a deliberate output naming rule. Recommend
an explicit `axis_dim="axis"` keyword for dense labelled point output, exposed through the
accessor, with an actionable conflict error suggesting another name. Preserve the existing
result for callers without a conflict. Confirm this small signature choice before coding.
`Grid.points` returns a NumPy array and has no reusable labelled-dimension naming mechanism.
Do not convert through Grid eagerly and lose Dask or nonseparable coordinate support.

Acceptance checks: a geometry varying over a dimension named `axis` can produce correctly
labelled points using the documented disambiguation; ordinary outputs retain their labels,
units, ordering, and lazy evaluation.

Schedule **M1**, the stale Geometry/accessor documentation, as a small documentation follow-up
in these early stages. This planning task does not authorize an unsolicited code edit or commit.

### Stage 3: Correct export semantics and narrow interpolation behavior

**F3: Preserve representable NGFF axis semantics; refuse the rest.** Retain unmixed axes with
their type and unit. Allow columns mixing rows only when they describe compatible spatial
axes with a shared unit; derive the column's unit from its contributing rows, not its index.
Refuse mixed time/space or incompatible-unit sums clearly. If implementing this limited rule
requires disproportionate machinery, initially refuse unsupported non-diagonal mixed-type or
mixed-unit cases. Do not invent dimensional meaning for their Euclidean norms.

Acceptance checks: inspect emitted axis types and units for an unmixed time axis plus spatial
permutation, preserve same-unit spatial rotations and their numeric geometry, and test explicit
refusal of unsupported mixing. A successful numeric round trip alone is insufficient.

**F4: Fix linear exact-sample NaN behavior; document cubic separately.** Reproduced identity
resampling of `[1, NaN, 3, 4]` preserves the first sample on the gather path but turns it into
NaN on the general linear path. Integer sample queries should preserve the actual sample
under the declared mapping, regardless of a built-in equivalent transform representation.
Use a bounded fix and preserve lazy execution.

Cubic prefiltering can spread a NaN through the coefficients. Document that behavior and the
existing same-grid gather exception initially. This leaves an explicit cubic consistency
limitation; it does not establish path independence for every interpolation method. Do not
add NaN-aware spline fitting or silently substitute nearest interpolation.

Acceptance checks: equivalent linear paths preserve exact samples with missing neighbours;
finite-data interpolation equivalence remains valid; nonintegral missing-data behavior and
cubic propagation are documented and covered by representative public tests.

### Stage 4: Decide the public paradigm before changing interfaces

- **A1, dimension ordering:** make positional queries and dense point/lattice outputs agree
  by default, or expose their order explicitly. Compare declaration order with current
  DataArray order using a transposed, anisotropic example. Choose one public contract and
  migration story before changing defaults; do not rewrite binding internals speculatively.
- **A2, framing and coordinate replacement:** decide whether `rf.frame(grid)` should require
  explicit permission to replace conflicting coordinates. My recommendation is an explicit
  replacement option rather than silently changing labels. The current behavior is documented,
  so this is an API policy revision.
- **A5, extension transforms:** document immutability, equality, and conditional hashability
  requirements for transforms retained by Grid. Avoid pretending arbitrary third-party objects
  can be made immutable with a generic deepcopy or freezer.
- **F5, indexed assignment:** document that assignment replaces payload in the destination
  frame. Keep current runtime behavior and leave an upstream assignment guard out of scope.
  This clarification can land earlier with related binding documentation.

Acceptance: concise examples demonstrate the chosen contracts; intentional policies are
identified; any changed signature/default has a migration note and focused public coverage.
These are design decisions, not automatic correctness patches.

### Stage 5: Improve usability and tool reliability

- **A3:** explain that the cells domain uses extended outer bounds, including interpolation
  across internal gaps; it does not mean the union of isolated cell intervals. Start with
  documentation rather than new modes or silent support changes.
- **A4:** consider additive common grid accessors for adapter results and visible import
  reports in ordinary reader workflows. Preserve provenance and existing return types;
  require concrete examples of friction before adding API surface.
- **A6:** consider reducing frame_array's nongeometric-coordinate requirements and supporting
  cross-type Grid/Geometry coincidence checks. Keep these optional unless real usage justifies them.
- **M2:** stage DICOM tool output in a sibling temporary directory and publish a completed
  series atomically. Preflight destination collisions and report partial batch failures.
  Distinguish per-series atomicity from a transaction across the entire batch; do not expand
  this work into a general DICOM confidentiality framework.

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

Stages 2–5 remain pending. Stage 4 needs deliberate public-contract choices;
Stage 5 contains optional usability work. Cubic path dependence with NaNs remains an explicit
limitation unless a later interpolation policy change is justified. No upstream submissions or commits have been made, and the user's unrelated uncommitted work
was preserved.

## Next Steps

Stage 1 is complete. Stage 2 begins with F6 scalar resampling targets and F7 point-output
dimension naming. Follow their acceptance checks when implementation is requested; revisit
this plan if a fix reveals a materially different cause or requires wider API changes.
