# Freestanding grids

**Status:** Active
**Last updated:** 2026-09-30
**Scope:** A NumPy-only `Grid` value describing an array's sampling without pixels; the doors that
bind, snapshot, resample onto and persist it; declared intervals carried by grids and bindings;
anatomical grid operations; and the frame-identity rule adapters follow.

## Context

Applications use sampling geometry without pixels: regions of interest drawn on an image, a
viewer's target plane, a registration's fixed domain, a file format's grid block. Today xarrayrf
has a freestanding `ReferenceFrame` (a world identity), but sampling exists only as a binding
on a `DataArray` (`Geometry` re-reads the array; `Lattice` is regular-only and not encodable).
An application therefore keeps its own grid type beside xarrayrf, which duplicates the model and
invites drift. The viewer plan's "pixel-free target domain" (item 3) is the same need.

Constraints: the core stays NumPy-only; identity is explicit (no value-based frame matching);
the binding stays the only authority on a framed array; no valid-looking incorrect geometry.

## Current Decision (proposed)

### 1. `Grid`: an immutable sampling value, NumPy-only core

`Grid(transform, coordinates, *, intervals=None)`:

- `transform`: a point transform from `ArrayCoordinates` into a `ReferenceFrame`, exactly what a
  binding holds.
- `coordinates`: one entry per source axis, `name -> (dim, values)` with `values` 1-D for a
  varying axis, or `name -> value` for a retained scalar (0-D), matching the binding's rule that
  bound coordinates are 0-D or 1-D and at most one per dimension. Values are copied, frozen, and
  checked finite and of real dtype; units come from the transform's source, not from attrs.
- `dims` (in coordinate order) and `sizes` are derived. A grid describes the geometry
  dimensions only: no pixels, no dtype, no non-geometry dimensions (time, channel, echo).
- `==` is exact (frame identity, transform, coordinate values, intervals); hashable.

`Geometry` stays the live, lazy view of an array; `Grid` is the immutable snapshot. They share
one private sampling module, so neither can disagree with the other. `Geometry.grid()` returns
the snapshot of a bindable geometry.

Queries, the same names and semantics on both:

- `frame`, `dims`, `sizes`, `point_at(**positions)`, `points()`, `lattice()` (when uniform),
  `is_coincident(other, tolerance=)`.
- **`points_at(positions, *, domain="samples", outside="raise")`: new.** Batched fractional
  positions to frame points, the forward partner of `positions_at`. Positions map to coordinate
  values linearly on uniform axes and piecewise linearly on nonuniform ones; beyond the outer
  samples they extend by the outer step, the rule default cells already use. `domain` bounds
  what is accepted (`"samples"` hull, `"cells"` extent); `outside` is `"raise"`, `"nan"` or
  `"extrapolate"`. Retained scalar axes are read from the grid, not supplied.
- `positions_at(points, *, domain=, outside=)`: the inverse with the same rules, including
  `"extrapolate"`, so the two round-trip on nonuniform grids.

Structural operations, named as xarray names them: `isel(**indexers)` (crop, stride, reverse,
scalar selection retaining the scalar axis), `transpose(*dims)`. Each returns a `Grid` whose
points equal the corresponding selection of the original's points.

### 2. Doors (no new binding path)

| From | Call | Notes |
|---|---|---|
| framed `DataArray` | `da.rf.grid` | Snapshot of the binding; property, like `rf.geometry` |
| existing `DataArray` | `da.rf.frame(grid)` | Overload: the grid's dims must be dims of the array with equal sizes; assigns the grid's coordinate values, frames, and leaves other dims and coords untouched. The `(transform, dims=)` form stays. |
| raw pixels (NumPy, Dask, duck array) | `xarrayrf.native.frame_array(data, grid, *, dims=None, coords=None, attrs=None)` | Public form of today's `frame_dataarray`. `dims` names every array dimension (default: the grid's); `coords` supplies non-geometry coordinates (a NIfTI time axis, channels). Never evaluates `data`. Adapters use it. |
| framed source | `da.rf.resample_to(grid)`, core `resample(source, grid)` | `Grid` joins `DataArray | Geometry` as a target; non-geometry dims are carried as today |
| xarray coordinates | `xarrayrf.native.grid_coordinates(grid) -> xr.Coordinates` | Carries the binding index, so `assign_coords` binds natively |

`Grid` has no `bind` method: arrays are framed through the `.rf` accessor or `frame_array`.

### 3. Declared intervals

Per-axis optional `(n, 2)` `[lo, hi]` in coordinate values (thickness, gaps, overlap, single
slices), per `geometry_and_resampling_design.md`. On grids, the `intervals` field. On arrays,
carried by the existing `BindingIndex`, which already owns the axis coordinates and retained
scalars, so no second index is needed. `domain="cells"` uses declared intervals when present and
the sample-offset default otherwise; an axis declaring both must agree. DICOM import declares
intervals from `SliceThickness`, which fixes the single-slice case.

Lifecycle, to be approved before stage 3 starts:

| Path | Intervals |
|---|---|
| `isel` / `sel` slice, stride, reverse | Rows sliced with the labels; `[lo, hi]` stay in coordinate values, so reversal reorders rows but never swaps `lo` and `hi` |
| Scalar selection | Kept with the retained scalar in `fixed` |
| `roll` | Rows rolled with the labels |
| `rename`, `swap_dims` | Carried |
| Alignment and join, matched labels | Intervals must be equal; conflicting support refuses |
| Join or reindex introducing labels the binding has no interval for | Refuses (no invented support) |
| Operand with a plain index and no intervals | Refuses when the binding declares intervals |
| `concat` | Concatenated when every operand declares them; mixed refuses |
| `rf.grid`, `frame_array`, `rf.frame(grid)`, encoding | Preserved exactly |
| `resample_to(target)` | The result carries the target's intervals, never the source's |

### 4. Anatomy on grids (`xarrayrf.anatomy`)

Affine transforms into anatomically oriented frames only; nonlinear transforms refuse.

- `orientation_codes(grid)`: the anatomical direction of each dim. Dims are assigned to frame
  axes one-to-one by the permutation maximizing the summed absolute cosines of the unit step
  directions (at most 3! candidates); if the best and second-best permutations are within
  tolerance, it refuses as ambiguous. Strongly sheared grids may therefore refuse; resample them
  to a cardinal grid first.
- `reoriented(grid, orientation)`: the same samples with dims permuted and reversed to match a
  requested orientation (pure `transpose` + reversing `isel`; points unchanged).
- `cardinal_grid(grid, orientation, *, spacing=None, cover="cells")`: a new axis-aligned grid in
  the same frame, for `resample_to`: the reformat-to-axial/sagittal/coronal case. It covers the
  source's cell extent (declared intervals, else sample-offset cells) or, with `cover="samples"`,
  its sample hull. Default spacing per output axis is the step of the source axis assigned to
  it; a nonuniform source axis needs explicit `spacing`. A single thick slice keeps its
  declared interval as the output slice's interval.

### 5. Frame identity (decision gate, stage 5)

Rule: **the same source yields the same frame; different sources share a frame only when an
identity is declared or the caller asserts it with `rf.assume_frame`.** No value matching. A
false "same" is a valid-looking incorrect geometry; a false "different" is a refusal with a
remedy, so identity policies must never equate sources that differ in content.

- DICOM: `("dicom-frame-of-reference", FrameOfReferenceUID)` (unchanged); without that UID,
  `("dicom-series", SeriesInstanceUID)` instead of a fresh local frame.
- NIfTI without a template identity (today each open mints a new local frame). Options, for
  maintainer decision:
  - (a) Full-content SHA-256 of the file: equal content, equal frame, wherever it lives. Costs a
    full read at open, which conflicts with lazy opening of large series.
  - (b) Resolved path plus size, modification time and header digest: reopening the unchanged
    file yields the same frame; a copy or move yields a different one (refusal with remedy).
    Cheap, but not a content guarantee: replacing the voxel data in place while preserving size,
    header and modification time keeps the identity.
  - (c) Keep local frames; callers reuse one through `frame=`.
  - Partial-content fingerprints are rejected: files differing only in their interiors would
    share a frame.
  - Open: (b) weakens the rule to "the same unmodified file"; the reviewer recommends (c) by
    default with (a) opt-in. Unresolved until the maintainer decides.
- `ReferenceFrame.local` stays random for programmatic use.
- Refusal diagnostics: when two frames differ in identity but their coordinates match
  numerically, the error says so and names `rf.assume_frame`. This is a separate private check,
  not `is_coincident`, which by definition requires the same frame.
- Unframed arrays (including wrapped NumPy) are index space. Nothing frames them implicitly.

### 6. Persistence

Schema 1 (not frozen) gains a `grid` kind: transform, coordinates (dim, values) and intervals,
plus intervals in the native binding encoding. Decoding rebuilds through constructors.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| `xr.Coordinates` carrying the binding index as the grid type | No home for grid methods (no accessor on `Coordinates`) and pulls xarray into the core; kept as the `grid_coordinates` conversion |
| Generalize `Lattice` | Regular-only by definition; nonuniform slice positions and intervals do not fit |
| A dataless `DataArray` as the grid | Needs a dummy payload; operations would re-enter the binding lifecycle for no pixels |
| Merge `Geometry` into `Grid` | Loses the live, lazy view over chunked coordinates and multidimensional fields |
| Non-geometry dims inside `Grid` | A grid would have to describe time and channels it cannot locate |
| `grid.bind(data)` | A third framing door beside `rf.frame` and `frame_array` |
| Value-based frame matching for unidentified sources | Two subjects with identical headers would silently share a world |
| Partial-content NIfTI fingerprints | Files differing only in their interiors would share a frame |
| Auto-framing plain arrays as local frames | Each mint is a new identity, so two same-shape arrays refuse to combine |

## Deferred Work

- Multidimensional coordinate fields in grids (bindings refuse them today).
- Grids for nonlinear transforms beyond what `SupportsPoints` already allows.
- Dataset-level grids.

## Next Steps

Each stage updates `core_interface.md` (normative) and its design note in the same change, and is
reviewed separately.

1. `Grid` value, shared sampling module with `Geometry`, `Geometry.grid()`, `points_at` and
   extrapolating `positions_at`, `isel`/`transpose`, `is_coincident`, `grid` encoding. Accept:
   Grid and Geometry answers agree; `isel` commutes with points; nonuniform forward/inverse
   round trip including extrapolation; encode/decode round trip; refusals.
2. Doors: `rf.grid`, `rf.frame(grid)`, `frame_array` (adapters migrated), `grid_coordinates`,
   `resample`/`resample_to` onto a grid. Accept: every door yields the same points; a lazy 4-D
   NIfTI and a multichannel array keep their non-geometry dims and coords; pixels unread.
3. Declared intervals, after the lifecycle table is approved: `Grid`, `BindingIndex`, cells
   domain, DICOM `SliceThickness`. Accept: one test per lifecycle row.
4. Anatomy functions. Accept: an oblique volume, a single thick slice, and a sheared grid whose
   per-dim largest cosines collide.
5. Identity rule in the adapters, after the maintainer decision; refusal diagnostics.
6. Design, viewer plan and roadmap updates.
