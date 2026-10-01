# Changelog

Maintainer record of fixes found in review. User-facing changes are in `CHANGELOG.md`.

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
