# Changelog

Maintainer record of fixes found in review. User-facing changes are in `CHANGELOG.md`.

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
