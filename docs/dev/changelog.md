# Changelog

Maintainer record of fixes found in review. User-facing changes are in `CHANGELOG.md`.

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
