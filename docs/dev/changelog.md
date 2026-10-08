# Changelog

## [2026-10-07] — Independent QA of the core layering refactor

- **Problem**: Verify that moving resampling planning, frame adoption, `Grid.isel`, binding
  diagnostics and interval arithmetic out of the xarray modules preserves public behavior,
  performance and the binding lifecycle, and that the stated import boundary holds.
- **Resolution**: One defect fixed. The deferred-revalidation size view in `_resample` revalidated
  the source Geometry on every planner size read (13 coordinate checks per call versus 9 at the
  base), making small resamples about 18% slower. It now revalidates on the first read and
  reuses those sizes; small resamples are now at or below base time, large ones unchanged.
  Stale design text claiming Grid selection and BindingIndex policy were still pending was
  corrected; the layering is recorded in
  [core_layering_design.md](architecture/core_layering_design.md). The `Grid.isel` error-type
  change for invalid indexers is recorded in `CHANGELOG.md`.
- **Verification**: Full suites: stock 1,904 plus 47 xfailed; patched 1,951; minimum-core (no
  SciPy) 1,648 plus 88 skipped/47 xfailed; minimum-resample 1,904 plus 47 xfailed; current
  Python 3.13/xarray 2026.9.0 1,908 plus 43 xfailed. Ruff, mypy (stock, floor, current) pass.
  `tools/binding_operation_probe.py` output is byte-identical to `da5f627`. Anonymous-frame
  diagnostic messages match `da5f627` for RangeIndex-step and int64/float cases. Implementation
  by Codex (gpt-6-astra) in three rounds, each reviewed by Claude Opus 5.5.
- **Files affected**: `src/xarrayrf/_resample.py` (size caching); design notes and changelogs.
- **Reviewed by**: Claude Opus 5.5 (`claude-opus-5-5`) applying QAEngineer.

## [2026-10-07] — Independent QA before committing repository cleanup

- **Problem**: Verify that the dependency-floor corrections, publishing smoke and documentation
  consolidation preserve public behavior, optional-dependency coverage and unique audit evidence.
- **Resolution**: No functional correction was needed. Clarified the metadata comment about
  lazy SciPy loading and rewrapped the touched Grid rationale. Kept the separate curved-spacetime
  assessment and its links outside the commit scope.
- **Verification**: Fresh full suites pass: stock 1,796 plus 47 xfailed; patched 1,843;
  minimum-core 1,559 plus 86 skipped/47 xfailed; minimum-resample 1,796 plus 47 xfailed;
  current Python 3.13/xarray 2026.9.0 1,800 plus 43 xfailed. Original source reproduces all
  three selected regressions and eight NumPy-floor mypy errors. Nine independent missing-extra
  probes verify validation, exception causes and zero Dask pixel tasks; interpolation kernels
  are unchanged after removing typing-only casts/aliases. Stock/floor/current mypy, Ruff,
  dependency consistency, helper/quoted-path checks and scoped Markdown paths/anchors pass.
  Distribution build/payload, strict metadata, sdist wheel rebuild and exact clean-wheel
  publishing smoke pass without SciPy. The original publishing block fails on its removed
  constructor keyword. Hosted CI and upload are separate from this local QA.
- **Files affected**: Cleanup changes in core resampling, tests, maintainer commands, publishing
  workflow and documentation; QA wording in `pyproject.toml` and Grid rationale.
- **Reviewed by**: Codex GPT-6 applying QAEngineer (exact serving model ID not exposed).
  Implementation review attribution remains Claude Opus 5.5 (`claude-opus-5-5`), as recorded
  in the preceding entry; no additional external review was required for wording-only changes.

Maintainer QA and validation provenance. User-visible behavior belongs in `CHANGELOG.md`;
completed audit decisions and per-stage evidence are in [the audit record](codebase_audit_review.md).

## [2026-10-06] — Repository documentation, dependency floors and publishing smoke

- **Problem**: The publishing smoke used a removed affine constructor; minimum-core tests
  loaded an absent optional dependency before validation; NumPy-floor stubs exposed eight
  typing errors. Completed audit transcripts and status snapshots contradicted current code.
- **Resolution**: Use the named-basis smoke declaration, load SciPy after resampling validation,
  guard actual sampling/NetCDF tests, and retain core-only frame/adoption assertions. Narrow
  numeric annotations without changing interpolation; consolidate completed audit history,
  refresh source-of-truth links and current/deferred status, and make local source paths explicit.
  `xrpr` defaults to the repository environment with an `XARRAY_PYTHON` override.
- **Verification**: The missing-extra validation regression fails before the import-order fix
  and passes after it. Exact floor environments reproduce the original 60 test failures/eight
  mypy errors and now pass: core 1,559 passed/86 skipped/47 xfailed; resample 1,796 passed/47 xfailed.
  Stock 1,796 passed/47 xfailed; patched 1,843 passed. Stock and floor mypy pass 82 files; Ruff
  lint/format and whitespace pass. Current Python 3.13/xarray 2026.9.0 passes 1,800 tests with
  43 xfailed and current mypy passes 82 files. Distribution build/payload and the exact
  clean-wheel publishing smoke pass on Python 3.12 without SciPy. Local helper and Markdown
  link/anchor checks pass. Opus accepted correctness convergence and documentation, with useful
  test/wording follow-ups resolved. Hosted CI has not rerun these unpushed changes; TestPyPI
  was not dispatched. Supported-floor warning totals remain in the test logs; none failed checks.
- **Files affected**: `src/xarrayrf/_resample.py`; optional-dependency/README/grid tests;
  `.github/workflows/publish.yml`; Makefile, `tools/xrpr`, binding probe; contributor/user docs
  and the maintainer documentation tree.
- **Reviewed by**: Codex GPT-6 applying QAEngineer; Claude Opus 5.5 (`claude-opus-5-5`) through
  collaborative-refinement. Model selection is recorded by invocation; the CLI did not announce
  its serving model separately.

## [2026-10-05–06] — Completed five-stage audit

- **Problem**: Incorrect identity/support/cardinality, valid-input failures, export/interpolation
  semantics, ambiguous public declarations and partial DICOM output were confirmed through
  public interfaces.
- **Resolution**: Five independently reviewable stages, including fresh Unicode-destination QA,
  are summarized in [codebase_audit_review.md](codebase_audit_review.md). That record preserves
  numerical calibration, commit provenance, original-source failures, independent probes and
  the uncommitted Stage 5 correction evidence. Original committed transcripts remain in Git.
- **Files affected**: Core geometry/sampling/native code, adapters, DICOM tool, public regression
  tests and normative docs; see the audit stage/commit table for scope.
- **Reviewed by**: Codex GPT-6 applying QAEngineer; Claude Opus 5.5 (`claude-opus-5-5`).

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
