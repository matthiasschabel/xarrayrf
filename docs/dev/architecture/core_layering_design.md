# Core layering: frames, sampling and resampling independent of xarray

**Status:** Implemented
**Last updated:** 2026-10-07
**Scope:** Internal module boundaries between the geometry core and the xarray binding:
resampling, frame adoption, `Grid` selection, binding diagnostics and interval arithmetic.
No public API or packaging change.

## Context

Handling reference frames and resampling does not depend on the array representation. xarray is
the first binding, but NumPy, Dask, Zarr or a viewer such as napari could bind the same geometry.
Before this change the core was NumPy-only for frames, transforms, `Grid` storage and queries, but
several geometry decisions still lived in, or went through, the xarray integration modules:

- the resampling planner took `Geometry` and cropped with `DataArray.isel`;
- `adopt_frame` lived in `_geometry` with a `DataArray` branch;
- `Grid.isel`/`sel` built an `xr.Dataset` through the native binding;
- `BindingIndex` explained incompatibilities and compared interval rows itself.

## Current Decision

xarray remains a required dependency. The layers are:

| Layer | Modules | May import |
|---|---|---|
| Geometry core | frames, coordinate systems, transforms, `Lattice`, `Grid`, `_sampling`, `_positions`, `_coincidence`, `_frame_adoption`, `_frame_compatibility`, `_intervals`, `_selection`, encoding | NumPy only |
| Value kernels | `_resampling` (`plan`, `execute`) | NumPy; SciPy lazily |
| xarray binding | `_geometry`, `_binding`, `_frame_coordinates`, `_resample`, `_grid_selection`, `native` | anything |
| Format adapters | `nifti`, `dicom`, `ngff`, `geotiff` | anything |

`tests/test_import_boundary.py` enforces the first two rows at any import depth (function-level
imports included) for `xarray`, `pandas` and every integration module. Its only exemption is the
lazy integration import in `Grid.sel`. A subprocess with xarray blocked exercises core `Grid.isel`
and the diagnostics; a separate SciPy-gated subprocess runs `plan`/`execute`.

Contracts a second binding relies on (all private):

- **Resampling.** `plan(source: Sampling, target: Sampling, *, source_order, target_order, ...)`
  validates options and frames, and returns a `_Plan` with the source `window` to crop. The binding
  slices its storage with the window **before** reading values, so lazy sources read only the
  footprint; `execute(plan, block)` interpolates a NumPy block whose trailing axes follow
  `source_order`. `Sampling` keeps exact `RangeIndex` steps and reads coordinates only when needed.
  The anonymous-frame "adoption suffices" hint takes a binding-supplied predicate so it keeps the
  binding's exact equality (the xarray wrapper uses `Grid` equality, which compares dtype kind and
  ignores steps).
- **Frame adoption.** `_frame_adoption.adopt_frame(transform, frame)`; the `_geometry` wrapper only
  resolves a framed `DataArray`.
- **Selection.** `_selection` selects dtype-preserving coordinate arrays and interval rows
  (int64 labels above 2**53 stay exact) and multiplies exact steps by slice stride.
  `FrameCoordinateTransform.sliced` uses it and keeps its own cell-reach bookkeeping.
- **Binding diagnostics.** `_frame_compatibility.binding_difference` takes the two transforms, a
  lazy samplings callback and the binding's exact adoption predicate, preserving evaluation order.
  `_intervals` compares and assembles interval rows from `get_indexer`-style positions (`-1` for
  missing). Label matching, `equals` and declaration compatibility stay in `BindingIndex`: they
  are different contracts (a disjoint-label inner join is valid).

Adapter policy that deliberately stays in the binding: `FrameCoordinateTransform.reverse`
clipping and restrictions, `Geometry.frame_coordinates` structural validation, label joins and
xarray index hooks.

## Alternatives Considered

| Alternative | Rejected because |
|---|---|
| A separate frame/resampling package | The core already ran without xarray; a split adds release coupling for no consumer |
| Making xarray an optional extra | Maintainer decision: xarray stays required; the goal is internal decoupling |
| A public NumPy-level resampling API | No non-xarray consumer yet; entry points stay private |
| Planner input as `Grid` | `Grid` snapshots lose exact `RangeIndex` steps and read coordinates eagerly |
| Rebuilding `BindingIndex` from core `Sampling` | `Sampling` is float64 and cannot reproduce exact labels; equality and compatibility are distinct |
| A generic framed-array wrapper | Lifecycle preservation depends on each library's operation model |

## Deferred Work

- A public NumPy-level resampling entry point, when a non-xarray consumer (napari) needs it.
- Neutral format readers returning `(values, Grid)`.
- Rebuilding binding state through core, if a second binding needs it.

## Next Steps

None. Keep the boundary test's integration set current when adding modules.
