# Changelog

Maintainer record of fixes found in review. User-facing changes are in `CHANGELOG.md`.

## [2026-10-06] — Independent QA of stage 4 and scalar assignment clarification

- **Problem**: Stage 4 needed fresh independent QA. The assignment documentation did not
  explicitly distinguish shared dimension-coordinate checks from retained scalar coordinates.
- **Resolution**: QA confirms declared ordering, conservative coordinate replacement and
  extension-transform contracts. Clarified that xarray ignores retained scalar coordinates
  during slice assignment, even when their values differ, while destination binding remains
  intact. Added a public contract test; no production source correction was needed.
- **Verification**: Original-source snapshot `a721688` fails all 15 selected regressions.
  Initial focused suite: 490 passed, 41 expected stock failures. Independent probes pass
  72 three-dimensional declaration/storage/eager-Dask permutations, checking all query views,
  native encode/decode and NGFF/NIfTI physical mappings; eight empty/scalar/eager-Dask
  replacement cases preserve caller indexes, pixels, context and intervals. Additional probes
  verify time/heterogeneous-unit NGFF semantics, reverse frame-coordinate selection and
  unhashable custom/composite querying and binding. The new scalar-assignment contract test
  and existing varying-dimension rejection test pass. Final full suites: stock xarray 1,765
  passed and 47 expected failures; patched xarray 1,812 passed, each with three existing
  dependency warnings. Ruff lint/format, mypy (81 source files), links and whitespace pass.
- **Files affected**: `tests/test_native.py`, `README.md`, `docs/core_interface.md`,
  `CHANGELOG.md`, this changelog, the [audit plan](codebase_audit_plan.md) and
  [stage 4 review record](stage4_implementation_review.md). Production source is unchanged;
  the unrelated user documentation edits remain intact and outside these commits.
- **Reviewed by**: Codex GPT-6 (exact serving model ID not exposed); prior implementation
  reviews by Claude Opus 5.5 (`claude-opus-5-5`). Stage 4 implemented by Codex gpt-6-astra.

## [2026-10-06] — Audit stage 4 API contracts and completed QA

- **Problem**: Geometry queries disagreed on dimension order, Grid framing silently replaced
  conflicting coordinates, and extension-transform value requirements were overstated.
  Indexed assignment semantics needed an explicit destination-frame contract.
- **Resolution**: Declared sampling order is canonical across queries; NGFF extracts storage
  columns explicitly. Grid framing requires `replace_coordinates=True` for conflicts and gives
  per-coordinate diagnostic reasons. Compatible metadata/pixels remain intact; opt-in adopts
  Grid declarations, including malformed unit replacement. Documented stable extension
  transforms, conditional hashing and assignment under xarray's label checks. No compatibility
  aliases, deprecation machinery, generic freezer or new assignment API was added.
- **Verification**: Opus accepted the plan and two implementation passes. gpt-6-astra implemented
  the work and diagnostic/index-coverage followups. Independent QA reproduces 15 failures against
  original source, passes the 486-test initial focused suite with 41 expected stock failures,
  and checks eager/Dask query parity, native encode/decode, NGFF storage-axis mapping, replacement
  pixel identity and metadata, custom values and distinct-frame assignment. Followup Grid tests:
  199 passed on patched xarray. Final independent full suites on main: stock xarray 1,764 passed
  and 47 expected failures; patched xarray 1,811 passed. Both retain three existing dependency
  warnings. Ruff lint/format, mypy (81 source files), links and whitespace pass. Reviewed source
  bytes and unrelated user edits were preserved through integration and commit.
- **Files affected**: `src/xarrayrf/_geometry.py`, `src/xarrayrf/native.py`,
  `src/xarrayrf/ngff/_export.py`, `src/xarrayrf/_grid.py`, `src/xarrayrf/_transform.py`;
  their public tests in `tests/test_geometry.py`, `tests/test_grid.py`, `tests/test_native.py`,
  `tests/test_frame_coordinates.py`, `tests/test_ngff_export.py`; `README.md`,
  `docs/core_interface.md`, `CHANGELOG.md`, the [audit plan](codebase_audit_plan.md) and
  [stage 4 refinement record](stage4_implementation_review.md).
- **Reviewed by**: Claude Opus 5.5 (`claude-opus-5-5`); independent QA and orchestration by
  Codex GPT-6 (exact serving model ID not exposed). Implemented by Codex gpt-6-astra
  (initial CLI used high reasoning effort).

## [2026-10-05] — Independent QA of audit stage 3

- **Problem**: The committed NGFF semantics and linear NaN fixes needed a fresh QAEngineer
  review independent of the implementation and earlier cross-review.
- **Resolution**: QA confirms that contributing physical rows determine intrinsic metadata,
  unsupported mixtures fail explicitly, and matching value/missing-weight kernels exclude
  zero-weight NaN neighbours. Complex components remain independent; outside fill, cells,
  source context, lazy output and cropped reads retain their contracts. No additional source
  correction was needed. Updated the audit status and stage 3 verification record.
- **Verification**: Original source `7724d41`, extracted into a separate snapshot without
  changing the checkout, fails 34 of 39 selected new regressions. The current focused suite
  passes 340 tests with 41 expected stock failures. An independent corner-weight oracle
  passes 32 cases across float32/float64/complex64/complex128, eager/Dask, affine/general,
  samples/cells, finite/missing slices and finite outside fill (seed 731). Two scalar complex
  accessor probes preserve binding, source name/attributes and context. Three independent
  NGFF cases preserve axis semantics and physical corners for a negative spatial permutation,
  a canonical-unit shear and a four-dimensional time/spatial map. Invalid physical ordering
  retains its upstream cause and corrective guidance. Full suites: stock xarray 1,739 passed
  and 47 expected failures; patched xarray 1,786 passed. Both retain three existing dependency
  warnings. Ruff lint/format, mypy (81 source files), documentation links and whitespace pass.
- **Files affected**: Review covers `src/xarrayrf/ngff/_export.py`, `src/xarrayrf/_resample.py`,
  their public tests, `docs/core_interface.md` and `CHANGELOG.md`. This QA commit updates only
  this changelog, the [audit plan](codebase_audit_plan.md) and
  [stage 3 review record](stage3_implementation_review.md).
- **Reviewed by**: Codex GPT-6 (exact serving model ID not exposed); prior implementation
  reviews by Claude Opus 5.5 (`claude-opus-5-5`). Implemented by Codex gpt-6-astra.

## [2026-10-05] — Audit stage 3 export semantics and linear missing weights

- **Problem**: Non-diagonal NGFF export assigned every intrinsic axis a spatial type and
  paired units with row indices rather than contributing quantities. Linear interpolation
  let zero-weight NaN neighbours contaminate exact and partially integral samples.
- **Resolution**: Intrinsic axes inherit contributing-row semantics; mixed rows require
  explicitly spatial quantities with one canonical declared unit. Upstream layout errors
  retain their cause and name caller actions for array and physical-frame ordering. Linear
  resampling uses component-wise missing-weight masks and a shared measured `1e-9` allowance,
  with one slice's buffers retained at a time. Public contracts and release notes are updated.
- **Verification**: gpt-6-astra implemented the fixes; Claude Opus 5.5 accepted both
  implementation review passes. Independent QA reproduces 33 failures against the original
  source, checks four additional eager/Dask affine/general partial-integral queries and
  physical-frame ordering, and verifies byte-for-byte integration. Full suites: stock xarray
  1,739 passed and 47 expected failures; patched xarray 1,786 passed. Both retain three existing
  dependency warnings. Ruff lint/format, mypy (81 source files) and whitespace pass. Original
  user-owned documentation edits remain intact and outside the commits.
- **Files affected**: `src/xarrayrf/ngff/_export.py`, `src/xarrayrf/_resample.py`,
  `tests/test_ngff_export.py`, `tests/test_resample.py`, `docs/core_interface.md`,
  `CHANGELOG.md`, the [audit plan](codebase_audit_plan.md) and
  [stage 3 refinement record](stage3_implementation_review.md).
- **Reviewed by**: Claude Opus 5.5 (`claude-opus-5-5`); independent QA and orchestration by
  Codex GPT-6 (exact serving model ID not exposed). Implemented by Codex gpt-6-astra
  (initial CLI used high reasoning effort).
- **Deferred**: Cubic coefficient NaN spreading and existing SciPy infinity behavior remain
  documented limitations; optional per-task slice routing awaits measured performance need.

## [2026-10-05] — QA of audit stage 2

- **Problem**: Scalar-target resampling and dense point-output naming needed independent
  verification before committing the stage 2 implementation.
- **Resolution**: QA confirms the fixes address rank-zero planning/interpolation and output
  name collisions directly, preserve cropped lazy reads and metadata, and reject invalid
  output names early. No additional source correction was needed.
- **Verification**: An isolated original-source snapshot fails all 40 selected scalar-target
  regressions. The current focused suite passes 297 tests with 41 expected stock failures.
  Four independent probes pass for eager/Dask nonlinear registration, ramp values, outside
  fill and target binding. Coordinate-only Dask fields also work with matching and differing
  chunk layouts. Full suites: stock xarray 1,696 passed and 47 expected failures; patched
  xarray 1,743 passed. Both retain three existing dependency warnings. Ruff lint/format,
  mypy (81 source files), documentation links and whitespace pass.
- **Files affected**: `src/xarrayrf/_resample.py`, `src/xarrayrf/_geometry.py`, their public
  tests in `tests/test_resample.py`, `tests/test_geometry.py`, `tests/test_native.py`, and
  the public docs/changelog and [stage 2 review record](stage2_implementation_review.md).
- **Reviewed by**: Codex GPT-6 (exact serving model ID not exposed); prior implementation
  reviews by Claude Opus 5.5 (`claude-opus-5-5`). Implemented by Codex gpt-6-astra.

## [2026-10-05] — Scalar resampling and labelled point-output collisions

- **Problem**: Valid scalar targets failed in crop planning, general indexing and block
  reshaping. Dense point output failed on an `axis` dimension or its private stacking name,
  and an indexed `units` dimension silently replaced component-unit metadata.
- **Resolution**: Scalar output retains shape `()` while the affine kernel uses one internal
  sample, preserving cropped reads, domain masks and cubic edge holding. General mapping
  supports zero varying target dimensions. `Geometry.points` accepts explicit `axis_dim`
  and `units_coord`, validates carried-name collisions and stacks broadcast coordinates only
  within the evaluation callback. Updated Geometry/accessor documentation. No source
  projection or later-stage interpolation policy was introduced.
- **Verification**: gpt-6-astra implemented the agreed plan; Opus 5.5 accepted the plan and
  final implementation with all findings resolved. After byte-for-byte integration, independent
  full suites pass: stock xarray 1,696 passed and 47 expected failures; patched xarray
  1,743 passed. Both retain three existing dependency warnings. Ruff lint/format, mypy
  (81 source files) and whitespace pass. Public regressions cover scalar binding/context,
  eager/Dask and empty extra dimensions, volumetric interpolation, crop-limited reads,
  output naming, scalar points and lazy nonseparable fields.
- **Files affected**: `src/xarrayrf/_resample.py`, `src/xarrayrf/_geometry.py`;
  `tests/test_resample.py`, `tests/test_geometry.py`, `tests/test_native.py`;
  `docs/core_interface.md`, `CHANGELOG.md` and the
  [audit plan](codebase_audit_plan.md). Complete decisions and reports are in the
  [stage 2 refinement record](stage2_implementation_review.md).
- **Reviewed by**: Claude Opus 5.5 (`claude-opus-5-5`); independent checks and orchestration
  by Codex GPT-6 (exact serving model ID not exposed). Implemented by Codex gpt-6-astra
  (initial CLI invocation used high reasoning effort).

## [2026-10-05] — QA of audit stage 1 identity, singleton support and DICOM cardinality

- **Problem**: Local NGFF reader identities depended on path spelling, allowing different
  stores to share a frame. Singleton matching admitted a one-second mismatch at epoch-scale
  coordinates. DICOM binding silently discarded surplus source slices and could accept an
  incomplete original enhanced stack when all selected indices were present.
- **Resolution**: Readers use resolved absolute local paths, including symlinks; metadata-only
  identities remain caller-resolved. Singleton matching uses float64 rounding allowances,
  with affine cancellation and built-in composite propagation accounted for. DICOM importers
  preserve the original source count and validate pixel-stack shape before gathering slices.
  Manual geometries with unknown count retain their existing index-based contract. QA found
  no additional functional defect. The non-affine inverse allowance limitation is documented;
  audit stages 2–5 remain pending.
- **Verification**: The original source snapshot fails eight selected regression cases
  (including the tightened DICOM shape-error contract), while six protective cases pass.
  The current focused suite has 178 passing tests and five expected stock-xarray failures.
  Four independent mixed-axis probes cover affine/composite singleton round trips, sign
  changes, empty lookup, coincidence, resampling and outside fill. Full suites: stock xarray
  1,638 passed and 47 expected failures; patched xarray 1,685 passed. Both retain three existing
  dependency warnings. Ruff lint/format, mypy (81 source files) and whitespace checks pass.
- **Files affected**: `src/xarrayrf/ngff/_reader.py`, `src/xarrayrf/ngff/__init__.py`,
  `src/xarrayrf/_sampling.py`, `src/xarrayrf/_positions.py`, `src/xarrayrf/_coincidence.py`,
  `src/xarrayrf/dicom/__init__.py`; `tests/test_ngff_open.py`, `tests/test_intervals.py`,
  `tests/test_dicom_import.py`; `CHANGELOG.md`, `docs/core_interface.md`, and the
  [audit plan](codebase_audit_plan.md), [plan refinement](codebase_audit_refinement.md) and
  [implementation review](stage1_implementation_review.md) records.
- **Reviewed by**: Codex GPT-6 (exact serving model ID not exposed); prior implementation
  reviews by Claude Opus 5.5 (`claude-opus-5-5`).

## [2026-10-04] — QA of named-basis affine construction

- **Problem**: Matrix-only construction left source-column and target-component ordering
  implicit. The implementation review also caught two accidentally migrated SimpleITK
  constructors; the QA pass found one remaining old-signature example in the relativity notes.
- **Resolution**: The default constructor uses source-named `basis_vectors` and requires
  `target_axes` to match the target exactly. Each vector is validated before assembly; both
  construction paths share immutable coefficients and retain schema 1. Matrix callers use
  `from_matrix`. The SimpleITK helper is byte-for-byte unchanged from the baseline, and the
  stale documentation example is corrected. QA found no remaining functional defect.
- **Verification**: 150 focused tests pass; stock xarray has 1617 passing tests and 47 expected
  failures for existing hook gaps; patched xarray has 1664 passing tests. Lint, formatting,
  mypy and `git diff --check` pass. An AST audit confirms 35 migrated Python files changed
  only the constructor spelling. The implementation pass also executed all 10 relativity
  notebook cells and checked the tour's placement against an independent point calculation.
  The full network-backed tour and optional SimpleITK registration helper were not executed.
- **Files affected**: `src/xarrayrf/_affine.py`; matrix callers in the core, anatomy and
  DICOM, NIfTI, NGFF and GeoTIFF adapters; affine, encoding, README and integration tests;
  `README.md`, `CHANGELOG.md`, `docs/core_interface.md`, the core-model and relativity notes,
  binding-operation inventory; notebook examples, their builders, the binding probe and
  resampling benchmark.
- **Reviewed by**: Codex GPT-6 (exact serving model ID not exposed); implementation reviews
  by Claude Opus 5.5 (`claude-opus-5-5`).

## [2026-10-01] — QA fixes for anonymous frames (grid plan stage 5)

- **Problem**: A QA pass over stage 5 found that a framed array written to NetCDF and read back
  could no longer combine with its original on stock xarray: the reload reordered the bound
  coordinates, and stock xarray keys alignment by coordinate order, so two equal bindings
  conflicted (this affected complete frames as well and predated stage 5; patched xarray was
  unaffected). It also found that two identically framed anonymous arrays sliced to zero length
  were told to resample although `assume_frame` alone sufficed, and following that advice
  raised `IndexError` inside resampling.
- **Resolution**: Every framing door and `rf.decode` produce bound coordinates and indexes in one
  canonical order, keeping the adapter orders already pinned by tests and leaving other
  coordinates and metadata untouched. Equal adopted bindings recommend `assume_frame` alone
  regardless of the numerical point check. Resampling onto an empty target returns an empty
  framed result, and an empty source with a non-empty target raises a documented `ValueError`.
- **Files affected**: `src/xarrayrf/native.py`, `_frame_compatibility.py`, `_resample.py`;
  `tests/test_grid.py`, `tests/test_native.py`, `tests/test_native_encoding.py`;
  `docs/core_interface.md`, `CHANGELOG.md`.
- **Reviewed by**: Codex gpt-6-astra (reasoning effort high), with fixes implemented by Codex
  gpt-6.1-sol and verified by Claude Opus 5.5 (claude-opus-5-5)

## [2026-10-01] — QA fixes for anatomy on grids (grid plan stage 4)

- **Problem**: A QA pass over `xarrayrf.anatomy` found that `cardinal_grid` always assigned
  source dimensions to frame axes, so a source whose assignment is ambiguous (a 45° volume)
  was refused even with explicit `spacing` and `dims`, although the ambiguity error tells
  callers to resample to a cardinal grid; and that the roundoff bound on non-anatomical frame
  axes multiplied the largest coefficient by the largest source span independently, so
  rescaling one source coordinate loosened it enough to accept a real one-second time
  variation. Committed tests also did not resample onto a cardinal grid, and the reflection
  offset bound was worded inconsistently.
- **Resolution**: Output directions come from the requested orientation and the frame;
  the source assignment is used only for default spacing and dimension names, which refuse an
  ambiguous source with advice to pass them explicitly. The roundoff bound uses each source
  column's own physical scale (coefficient times span), so equivalent representations decide
  alike. Integration tests resample an oblique DICOM-like volume and a thick single slice onto
  cardinal grids against a linear oracle. The offset bound is stated as `np.spacing(1.0)`.
- **Files affected**: `src/xarrayrf/anatomy.py`; `tests/test_anatomy.py`,
  `tests/test_dicom_import.py`; `docs/core_interface.md`, `docs/dev/architecture/grid_plan.md`,
  `CHANGELOG.md`.
- **Reviewed by**: Codex gpt-6-astra (reasoning effort high), with fixes implemented by Codex
  gpt-6.1-sol and verified by Claude Opus 5.5 (claude-opus-5-5)

## [2026-09-30] — QA fixes for declared intervals (grid plan stage 3)

- **Problem**: A QA pass over declared intervals found that an empty varying axis with declared
  intervals could not round-trip through either persistence API (its `(0, 2)` rows encoded as
  `[]` and decoded with the wrong shape); that the magnitude-aware offset/interval agreement
  tolerance could exceed a narrow interval's width, admitting a sample outside its own cell, which
  the cells domain then refused to locate; and that binding-index equality raised pandas'
  `InvalidIndexError` in one operand order when labels repeated, while the reverse returned
  `False`.
- **Resolution**: Empty interval lists decode as `(0, 2)` rows for empty varying axes, with strict
  rejection kept elsewhere. Construction checks exact containment `lo <= sample <= hi` in addition
  to the approximate offset agreement. `BindingIndex.equals` compares coordinate indexes before
  any label lookup, so it returns a bool in both orders and merges raise xarray's `MergeError`.
- **Files affected**: `src/xarrayrf/_encoding.py`, `_sampling.py`, `_binding.py`;
  `tests/test_intervals.py`; `docs/core_interface.md`,
  `docs/dev/architecture/geometry_and_resampling_design.md`, `CHANGELOG.md`.
- **Reviewed by**: Codex gpt-6-astra (reasoning effort high), with fixes implemented by Codex
  gpt-6.1-sol and verified by Claude Opus 5.5 (claude-opus-5-5)

## [2026-09-30] — QA fixes for freestanding grids (grid plan stages 1 and 2)

- **Problem**: A QA pass over the `Grid` value and its doors found:
  - copied and unpickled grids exposed writable coordinates, and returned arrays let callers
    re-type the stored bytes;
  - `Grid` query paths accepted malformed custom-transform output that `Geometry.points`
    already refused;
  - fractional interpolation lost stored endpoints to cancellation across large magnitudes
    (the first fix then overflowed when extrapolating near the float limit);
  - integer-only input was promoted to float64 and rounded before validation;
  - a `Geometry` resample target overwrote the source's non-geometry scalar coordinates,
    unlike a `Grid` target;
  - `Grid.isel` duplicated xarray's indexing and the binding's slicing and lacked list,
    array and mask indexers; zero-row queries on empty grids raised.
- **Resolution**: Copies and pickles rebuild through the constructor and coordinates are
  returned as read-only views. Transform-result validation is shared by every `Grid` and
  `Geometry` query. Interpolation uses a weighted form inside a cell (exact at samples) and
  extends from the nearer endpoint outside it. Integer input is detected and range-checked
  before promotion. Resampling takes only geometry coordinates from the target and refuses
  name collisions with source context. `Grid.isel` and the new `Grid.sel` run xarray's
  selection on a coordinate-only Dataset carrying the binding index. Zero-row queries return
  empty results. The lost `RangeIndex` step on snapshots is documented as a parity difference.
- **Files affected**: `src/xarrayrf/_grid.py`, `_positions.py`, `_sampling.py`, `_validation.py`,
  `_resample.py`, `_binding.py`, `_geometry.py`, `native.py`; `tests/test_grid.py`,
  `tests/test_resample.py`, `tests/test_import_boundary.py`; `docs/core_interface.md`,
  `docs/dev/architecture/geometry_and_resampling_design.md`, `CHANGELOG.md`.
- **Reviewed by**: Codex gpt-6-astra (reasoning effort high), with fixes implemented by Codex
  gpt-6.1-sol and verified by Claude Opus 5.5 (claude-opus-5-5)
