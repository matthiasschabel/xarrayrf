# Freestanding grids

**Status:** Active
**Last updated:** 2026-09-30
**Scope:** A NumPy-only `Grid` value describing an array's sampling without pixels; the doors that
bind, snapshot, resample onto and persist it; declared intervals carried by grids and bindings;
anatomical grid operations; and complete versus anonymous frames.

## Context

Applications use sampling geometry without pixels: regions of interest drawn on an image, a
viewer's target plane, a registration's fixed domain, a file format's grid block. Today xarrayrf
has a freestanding `ReferenceFrame` (a world identity), but sampling exists only as a binding
on a `DataArray` (`Geometry` re-reads the array; `Lattice` is regular-only and not encodable).
An application therefore keeps its own grid type beside xarrayrf, which duplicates the model and
invites drift. The viewer plan's "pixel-free target domain" (item 3) is the same need.

Constraints: the core stays NumPy-only; identity is explicit (no value-based frame matching);
the binding stays the only authority on a framed array; no valid-looking incorrect geometry.

## Current Decision

### 1. `Grid`: an immutable sampling value, NumPy-only core

`Grid(transform, coordinates, *, intervals=None)`:

- `transform`: a point transform from `ArrayCoordinates` into a `ReferenceFrame`, exactly what a
  binding holds.
- `coordinates`: one entry per source axis, `name -> (dim, values)` with `values` 1-D for a
  varying axis, or `name -> value` for a retained scalar (0-D), matching the binding's rule that
  bound coordinates are 0-D or 1-D and at most one per dimension. Values are copied, frozen, and
  checked finite and of real dtype (integers stored as int64, floats as float64; booleans
  refused); units come from the transform's source, not from attrs.
- `dims` (in coordinate order) and `sizes` are derived. A grid describes the geometry
  dimensions only: no pixels, no pixel dtype, no non-geometry dimensions (time, channel, echo).
- `==` is exact (frame identity, transform, coordinate values and dtype kind, intervals);
  hashable. Integer and float declarations differ.

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
| raw pixels (NumPy, Dask, duck array) | `xarrayrf.native.frame_array(data, grid, *, dims=None, coords=None, attrs=None)` | Replaces `frame_dataarray` (removed). `dims` names every array dimension (default: the grid's); `coords` supplies non-geometry coordinates (a NIfTI time axis, channels). Never evaluates `data`. Adapters use it. |
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

Approved lifecycle (implemented in stage 3):

| Path | Intervals |
|---|---|
| `isel` / `sel` slice, stride, reverse | Rows sliced with the labels; `[lo, hi]` stay in coordinate values, so reversal reorders rows but never swaps `lo` and `hi` |
| Scalar selection | Kept with the retained scalar in `fixed` |
| `roll` | Rows rolled with the labels |
| `rename`, `swap_dims` | Carried |
| Alignment and join, matched labels | Intervals must be equal; conflicting support refuses |
| Join or reindex introducing labels the binding has no interval for | Refuses (no invented support) |
| Operand with a plain index and no intervals | Known-label subsets retain support; introducing labels without intervals refuses |
| `concat` | Refused along a geometry dimension, as before; along other dimensions bindings align as in any join (`join="exact"` requires identical grids) |
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

### 5. Complete and anonymous frames

A **complete** frame has geometry and an identity: a declared name (a DICOM Frame of Reference
UID, a NIfTI template space) or a local frame someone created deliberately and shares by
passing the object around. Consistency between complete frames is enforced, and nothing is
ambiguous. An **anonymous** frame has geometry but no identity, because its source did not say
which world it belongs to: a scanner-space NIfTI affine, a DICOM series without a Frame of
Reference UID, an NGFF coordinate system read without a store. An unframed array has neither and
is index space.

The library never guesses an identity for an anonymous source: no content hashing, no path
fingerprints, no shared default world, no series-UID fallback. Completing a frame is the user's
explicit act, made at the point where it matters.

- **Construction.** `ReferenceFrame.anonymous(coordinate_system, *, definition=None, ...)` mints a
  distinct identity in its own namespace, and `frame.is_anonymous` reports it. Adapters use it
  wherever their source declares no identity; `ReferenceFrame.local` stays the constructor for
  frames created on purpose. Each adapter call site is classified in stage 5.
- **Alone, an anonymous frame is fully usable.** Viewing, geometry queries, `rf.grid`, resampling
  onto its own grids and every array derived from it work, because nothing is being related to
  anything else. Derived arrays share its identity through the binding.
- **Completing it: asserting a shared world.** At load, pass `frame=` a frame or a framed array
  (`t2 = nifti.open(p2, frame=t1)`); afterwards, `t2.rf.assume_frame(t1)`, which already accepts a
  frame or a framed array. Both state "these share a world", and the statement is the user's. The
  two remedies have one contract, implemented once: adopt the other frame's identity, and when
  the coordinate systems differ apply the exact derivable change between them
  (`coordinate_system_change`, e.g. RAS to LPS), as NIfTI's `frame=` already does. `assume_frame`
  is extended accordingly; it refuses only an underivable change (different units or axis
  meanings) or a non-affine mapping, as `frame=` does.
- **Asserting a world is not matching a grid.** Adopting an identity never bypasses the binding's
  grid checks. Arrays sharing a frame align and combine in arithmetic only when their bindings
  are compatible (same transform, dims and retained coordinates); otherwise the remedy is
  `resample_to`, which needs only the shared frame.
- **Refusals name the remedy that applies.** When arrays with different frames meet (alignment,
  arithmetic, `resample_to` without a transform) and either frame is anonymous, the error says
  which array is anonymous and gives the applicable steps: `frame=`/`assume_frame` to assert the
  shared world, followed by `resample_to` when the grids differ; when the coordinates also match
  numerically, it says so and `assume_frame` alone suffices. An underivable coordinate-system
  difference is named instead of offering a remedy that would fail. This is a separate private
  check, not `is_coincident`, which requires the same frame.
- **Visible state.** `repr` of a frame, a binding and a grid marks an anonymous frame, so the
  state is visible before any error.
- **Persistence.** Encoding keeps the anonymous namespace and identifier, so a saved array
  reloads with the same anonymous identity and still combines with arrays that shared it.
- Unframed arrays (including wrapped NumPy) are index space. Nothing frames them implicitly.

### 6. Persistence

Schema 1 (not frozen) gains a `grid` kind: transform, an explicit `dims` order, coordinates
(dim, values) and intervals,
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
| Identity derived from the source (content hash, path and modification time, series UID) | Cleverness the user cannot predict: a full hash reads the whole file, a path breaks on copy and can miss in-place edits, a partial fingerprint equates files differing only in their interiors |
| One shared default world for all anonymous frames, with opt-in uniqueness | Different subjects would resample onto each other silently; safety on request protects only users who already know the danger |
| Requiring `frame=` at load for anonymous sources | Ceremony for the common one-array case with no added safety, since an anonymous frame already cannot combine with another |
| Auto-framing plain arrays as local frames | Each mint is a new identity, so two same-shape arrays refuse to combine |

## Deferred Work

- Multidimensional coordinate fields in grids (bindings refuse them today).
- Grids for nonlinear transforms beyond what `SupportsPoints` already allows.
- Dataset-level grids.

## Next Steps

Each stage updates `core_interface.md` (normative) and its design note in the same change, and is
reviewed separately.

1. **Done.** `Grid` value, shared sampling module with `Geometry`, `Geometry.grid()`,
   `points_at` and extrapolating `positions_at`, `isel`/`transpose`, `is_coincident`, `grid`
   encoding. Accept: Grid and Geometry answers agree; `isel` commutes with points; nonuniform
   forward/inverse round trip including extrapolation; encode/decode round trip; refusals.
2. **Done.** Doors: `rf.grid`, `rf.frame(grid)`, `frame_array` (adapters migrated), `grid_coordinates`,
   `resample`/`resample_to` onto a grid. Accept: every door yields the same points; a lazy 4-D
   NIfTI and a multichannel array keep their non-geometry dims and coords; pixels unread.
   Stage 1 amendment: integer snapshots and materialized coordinates stay int64, including
   exact JSON round trips; floating coordinates stay float64. All adapter outputs remain unchanged.
3. **Done.** Declared intervals: `Grid`, `BindingIndex`, shared cells domain, all doors,
   native/grid persistence and DICOM `SliceThickness`. Public lifecycle tests cover selection,
   scalars, roll, rename, `swap_dims`, support conflicts, joins, missing-label refusals, concat
   refusal and target support. Existing xarray hook limitations remain on the stock lane;
   mixed-index subsets and `swap_dims` are exercised against the local patched lane.
4. Anatomy functions. Accept: an oblique volume, a single thick slice, and a sheared grid whose
   per-dim largest cosines collide.
5. Anonymous frames: `ReferenceFrame.anonymous` and `is_anonymous`; adapter call sites
   classified (anonymous versus deliberate local); `frame=` accepting a framed array where it
   does not yet; `assume_frame` applying derivable coordinate-system changes through the same
   implementation as `frame=`; refusal messages naming the applicable remedy; `repr` marking;
   persistence round trip. Accept: two anonymous opens of equal grids refuse, then combine after
   either `frame=t1` or `assume_frame(t1)`; an anonymous RAS array adopts an LPS frame through
   both remedies with identical points; shared-identity arrays with different grids still refuse
   arithmetic and resample onto each other; an underivable change refuses with its reason; a
   reloaded anonymous array still combines with its partners.
6. Design, viewer plan and roadmap updates.
