# Geometry, cells and resampling

**Status:** Implemented
**Last updated:** 2026-10-07
**Scope:** `Grid` (the immutable sampling value), `Geometry` (the read-through view),
`ArrayCoordinates.sample_offset` and the
`"samples" | "cells"` domain, `Geometry.is_coincident`, core `resample` including the same-grid
gather, and the accessor's `rf.resample_to` and `rf.assume_frame`. Declared intervals are
implemented. [The core interface](../../core_interface.md) is normative.

## Context

A transform says what coordinate values mean; it carries no array. Answering "where is sample
`(2, 1, 3)`?" needs only the array's own coordinate values, so it can ship without a binding.
Three questions that all produce frame points must stay apart:

- **Where is this sample?** Read coordinates at a position and apply the transform. `Geometry`.
- **What is the value at this point?** Needs an interpolant, a domain and a policy. `resample`.
- **How do two frames relate?** Needs an explicit registration or domain transform, never
  inferred from identity plus matching labels.

Half-voxel errors are a recurring source of silent incorrectness, so where a sample sits in its
cell, and how far values extend beyond the outer samples, must be declared rather than assumed.
Ecosystems differ at the edge: xarray `interp`, scipy's constant mode and nibabel answer only
between outer samples; ITK answers out to the voxel extent. CF describes explicit extents with
bounds variables; VTK distinguishes point from cell data.

## Current Decision

### Geometry is a view that re-reads the array

`Geometry(array, transform, dims=..., intervals=None)` holds a reference to the array, which stays the authority
for its sample domain.

- **Validity is not cached.** Every geometry-dependent property and query re-runs the structural
  check against the array as it is now. Validating once would make the view a stale snapshot
  after the caller edits `attrs["units"]` or assigns a coordinate. The cost is proportional to
  the number of source axes, not samples.
- **`dims` must equal the dependency dimensions.** A coordinate depending on a dimension outside
  `dims` is refused (nonspatial context may not silently parameterize geometry), and so is a
  declared dimension no source axis depends on. Empty `dims` is meaningful: a fully selected point.
- **Only selected values are evaluated.** Construction checks names, presence, dtype kind and
  units; values are read in the query that needs them, which keeps chunked coordinates lazy. A
  non-finite value surfaces at the query it affects.
- **Units:** an explicit `attrs["units"]` must equal the declared unit exactly; nothing converts.
- **Positions** are zero-based and bounded; negative positions are refused rather than read from
  the end, so an off-by-one is an error, not a sample at the far edge.

Queries: `point_at`, `points()` (lazy), `lattice()`, `frame_coordinates()` (lazy xarray
coordinates for interoperability), `points_at`, `positions_at` (inverse lookup, refused for retained scalars,
fields and embedded planes) and `is_coincident`. The `.rf.geometry` accessor delegates to this
view; the binding adds snapshot, association and enforcement, which the view never provides.

### Grid is a frozen sampling value

Stages 1 through 3 of [the grid plan](grid_plan.md) are implemented. `Grid(transform, coordinates,
intervals=None)` copies finite real 0-D/1-D coordinates into immutable NumPy buffers:
integer inputs become int64 without passing through float, floating inputs become float64.
Out-of-range integers refuse. Equality and hashing include the coordinate dtype kind; sampling
math uses float64, so integer declarations stay exact without changing interpolation. It has
no pixels or non-geometry dimensions. Construction and sampling have no xarray dependency.
Coordinate arrays are fresh read-only views, so editing their headers cannot change the stored
declaration; copying and unpickling reconstruct through the constructor to freeze the buffers.
Equality and hashing
compare the exact declaration, including dimension order; `is_coincident` is the separate
step-tolerant query. `isel` selects dtype-preserving NumPy coordinates and interval rows in
core, retaining scalar selections. Positional lists, arrays and masks follow the existing
xarray indexing contract. `sel` remains an xarray convenience, importing an integration helper
that selects a coordinate-only Dataset carrying the native binding. `transpose` stays native.
All commute with `points()`. Core per-axis selection also updates exact steps for sliced frame
coordinates; their reach bookkeeping remains in the frame-coordinate integration module.

`Geometry.grid()` revalidates and snapshots the current coordinates in its declared dimension
order. It refuses multidimensional fields and shared dimensions. It never reads pixels, but
snapshotting necessarily evaluates the coordinate values. Geometry construction, selected
`point_at` reads, lazy `points()` and revalidation on every query retain their existing behavior.
The snapshot stores coordinate values, not `RangeIndex` step metadata. Its lattice checks
uniformity from those values within tolerance, so large-coordinate rounding can make the
snapshot refuse a lattice that the live geometry accepts using the exact index step.

The private NumPy sampling description carries transform, dims, sizes, source coordinate values
and exact index steps when available; offsets come from the source. Geometry supplies coordinate
values only when a query needs them. The positions, lattice and coincidence math consumes this
description, shared by Grid and Geometry. Resampling accepts Grid targets directly through
this description, without constructing dummy pixels or a target array.

`points_at(positions, domain="samples", outside="raise")` maps fractional positions piecewise
linearly and reads retained scalars from the sampling description. It and `positions_at` accept
`outside="extrapolate"`, extending by the outer steps, and round-trip even on nonuniform axes.
The existing `"raise"` and `"nan"` policies enforce the chosen samples/cells domain. A singleton
has no step for forward fractional extrapolation; inverse lookup still admits only its coordinate.
Retained scalars still refuse inverse lookup and coincidence pending a projection policy.
Empty query batches return empty points or positions even on empty axes; non-empty queries on
empty axes refuse. Interpolation preserves stored endpoints at integer positions. Shared result
validation checks transform output shape, real dtype and finiteness before the NaN outside mask.

### Grid doors share the native binding

`rf.grid`, `Grid.isel` and `Grid.sel` share one private conversion from native binding
coordinates to a Grid in binding dimension order, preserving coordinate dtype and validating
units before freezing values. `rf.frame(grid)` checks geometry dimensions
and sizes, then uses `native.grid_coordinates(grid)`. Matching existing coordinates keep their
attrs; differing values or dtype kinds are replaced by the grid's coordinates with declared
source units. Geometry validation still checks preserved coordinate attrs against the transform.
`dims=` refuses with a grid. Non-geometry dimensions and coordinates stay untouched.

`native.frame_array(data, grid, dims=None, coords=None, attrs=None)` validates duck storage and
the shape declared by geometry and non-geometry coordinates, then frames through the accessor.
It never evaluates pixels. The four format adapters construct a Grid from their existing
geometry coordinates and pass non-geometry coordinates separately; their output declarations
remain the same. `frame_dataarray` is removed.

`native.grid_coordinates(grid)` builds 0-D/1-D variables with declared source unit attrs and
uses the same private BindingIndex factory as transform framing. Assigning these coordinates
to an array with matching dimensions and sizes is the native xarray framing door.

Intervals, anatomy and adapter identity changes remain later stages.

### Sample offsets and cells

- **Coordinate:** where its sample is. Crop offsets and one-based labels live in coordinate
  values, not in a separate index offset.
- **Cell:** the region an element stands for (voxel, pixel, time bin); nominal support, not the
  point-spread function.
- **Sample offset:** per axis, where the sample sits in its cell, a fraction in `[0, 1]` measured
  from the cell edge at lower coordinate values; `None` for point samples (echo times, time
  points), which have no cells. Part of `ArrayCoordinates` equality.

The range is closed because a coordinate is the sample's own location, so `1` (a cell ending at
its sample, as for accumulations) differs from `0`. Measuring toward higher coordinate values,
not higher index, keeps the offset true under reversal, cropping and striding. `coarsen`
recomputes coordinates and keeps only a centred offset true; this is documented, not detected.

**Default cells are dense and defined in positions.** Position `p` spans `p - s` to `p + 1 - s`,
mapped to coordinates as positions are (linear for uniform axes, piecewise linear for nonuniform
ones, extended by the outer steps). On a nonuniform axis the fraction holds in positions, not
distance: offsets `[0, 2, 5]` with `s = 0.5` give a middle cell `[1, 3.5]`. Default cells describe
the current sampling, so a stride widens them (`[-0.5, 0.5]` becomes `[-1, 1]` after selecting
every other sample). This is a declared default applied only when the producer declares an
offset, never inference of support from spacing.

**Domains.** `"samples"` (default, as xarray) is defined between the outer samples;
`"cells"` extends the hull to the outer cells' edges, where every method holds the edge value
(ITK's domain is the same; its nearest and linear hold the edge, its B-spline mirrors). Cubic
re-evaluates the shell at clamped positions; nearest and linear cost nothing extra. A
point-sampled axis adds no reach. An axis declaring cells with a single sample and no interval is refused: no
neighbour fixes its width, and calling it point-sampled would misdescribe a slice with thickness.
`positions_at(..., domain="cells")` returns true fractional positions such as `-0.5`.

### Declared intervals (implemented)

Declared intervals override the default cells extent: one `[lo, hi]` per sample, keyed by
source axis name, in that coordinate's values and units. They allow cells that do not tile,
such as multislice MRI with gaps or overlap, and single slices. Spacing never stands in for
thickness; the DICOM adapter attaches intervals from `SliceThickness`, in index units for a
uniform stack and millimetres for `slice_offset`.

- Varying axes use `(n, 2)` rows; retained scalars use `(2,)`. Storage is frozen float64 and
  public mappings return read-only views. Bounds and widths must be finite, with `lo < hi`.
- The sample must agree with `lo + s * (hi - lo)` within
  `max(1e-9 * (hi - lo), 8 * eps * max(|lo|, |hi|))`, where `s` is the source axis's sample
  offset and `eps` is float64 machine epsilon. `INTERVAL_ROUNDOFF_FACTOR = 8` allows arithmetic
  roundoff at large origins such as epoch seconds. Point-sampled axes (`s=None`) refuse intervals.
- Independently of offset agreement, every sample must lie inside its declared interval exactly:
  `lo <= sample <= hi`, for both varying axes and retained scalars. The magnitude allowance
  cannot admit samples outside narrow cells.
- The cells domain reaches the outer declared bounds; interior gaps remain interpolated.
  Multi-sample positions retain their piecewise-linear mapping and outer-step extrapolation.
  Only in the cells domain does a single sample use interval width as its step: positions `-s`
  and `1-s` map to its bounds. In the samples domain its matching tolerance remains
  `SINGLE_SAMPLE_TOLERANCE * max(1, |v|)`, returning position zero for a match regardless of
  interval declarations; its interval supplies no fractional extrapolation step.
- Grid and Geometry use one sampling implementation. A plain Geometry has no intervals unless
  explicitly supplied; the bound accessor passes its BindingIndex's declaration.
- BindingIndex carries rows through selection, retained scalar selection, roll, rename and
  `swap_dims`. Reversal reorders rows without reversing bounds. Matched labels must have exact
  support agreement; new labels must have support in a binding operand. Plain-index subsets
  in mixed alignment keep known support; introducing unknown labels refuses. The public
  `reindex_like` hook refuses non-binding operands. Equality returns `False` for interval
  mismatches and skips axes on excluded dimensions; coordinate merging raises xarray's
  `MergeError`. Joins and compatibility checks still raise `ValueError` naming the axis.
  Concat refuses along a geometry dimension; along other dimensions bindings only align.
- All grid doors and native encoding preserve intervals. Native resampling carries only the
  target's support; core resampling continues to return an unframed array. Existing xarray
  hook requirements still apply to mixed-index operations and `swap_dims`.

In a 2-D frame a pixel has in-plane extent only; placing it in 3-D needs a third source axis
whose offset says where the sample sits along the normal and whose interval gives thickness.
`cell_corners`, curvilinear cells and a strict domain only inside individual declared cells
remain deferred.

### Tolerant comparison

`Geometry.is_coincident(other, *, tolerance=1e-6)` asks whether both arrays sample the same
points of the same (or equivalent) frame element for element. The tolerance is in steps, as ITK's
coordinate tolerance is a fraction of spacing, so it is unit-independent. The samples-domain
locator is widened by the tolerance so edge jitter passes. For affine maps the check is exact and
separable, linear in samples per dimension; corner-only checks were rejected because they miss
interior deviations of approximate lattices. A single-sample dimension must agree to rounding
until declared cells give it a width. Cells, offsets and values are not compared. `==` stays
exact.

### Core `resample`

The private `_resampling.plan` consumes `Sampling` descriptions and explicit source/output
axis orders; `_resampling.execute` interpolates NumPy blocks. Neither imports xarray, pandas
or an integration module. `_resample.resample` translates Geometry/Grid inputs, applies the
planned source window before transpose and Dask rechunking, and assembles xarray coordinates.
Exact RangeIndex steps stay in `AxisSampling.step` through lattice construction. Cropped
sizes determine gather indices and interpolation bounds; offsets and domain bounds shift
together. SciPy loads only after input and frame checks. These entry points remain private.

Frame adoption lives in `_frame_adoption`, separate from derivation of coordinate-system
changes in `_orientation`: adoption asserts shared identity before composing that change.
The `_geometry.adopt_frame` wrapper resolves a framed DataArray and retains binding-facing
errors. The layering and the contracts a second binding relies on are in
[core_layering_design.md](core_layering_design.md).

`resample(source, target, *, transform=None, ...)` computes each target sample's frame point,
maps it into the source frame (an exact coordinate-system change between equivalent frames is
derived automatically; different frames need a caller transform), inverts the source transform
and interpolates. It is lazy over Dask chunks of non-geometry dimensions, carries non-geometry
dimensions through, resamples complex sources as complex128, and returns an unframed array. When
both sides form lattices with affine transforms it composes one affine and calls
`scipy.ndimage.affine_transform` per slice; otherwise it works in blocks of `block_points`
(default 2^20) with `map_coordinates`, computing cubic coefficients one slice at a time and
caching positions across slices within a 256 MiB budget.

**Same-grid gather.** On the lattice path the composed target-to-source index map is rounded to
the nearest signed permutation with integer offset. If the rounded and actual maps differ by at
most `1e-6` source samples at every corner of the target index box (the difference is affine, so
its maximum is at a corner, which bounds accumulated coefficient error for any shape), values are
gathered by transposing, reversing and slicing, with the domain's fill or edge rule outside the
source. Dask sources stay lazy. This makes relabelling between index and millimetre-offset
conventions, crops, flips and permutations exact and cheap, through one entry point; a
half-sample shift or a coefficient off by `5e-10` over a large range takes the interpolating path.

### `rf.resample_to` and `rf.assume_frame`

`source.rf.resample_to(target, *, transform=None, method="linear", fill_value=np.nan,
domain="samples")` takes a Grid, framed DataArray or Geometry (target values are ignored), calls
core `resample`, and frames the result with the target's coordinate transform and
geometry dims. Core resampling takes Geometry or Grid targets, using their shared sampling
math and restoring the source's non-geometry coordinate variables, including attrs that xarray
would otherwise drop. Grid targets have the same values as equivalent framed-array targets.
Only the target's source-axis coordinates contribute to the result. Unrelated target scalar
context is ignored; source context is preserved, and a target geometry coordinate name that
collides with source non-geometry context raises `ValueError` naming it.
The result keeps the source's non-geometry dims and coords, name and attributes
(restored explicitly because core `resample` drops attrs); the reserved `xarrayrf_binding`
attribute is not carried. Unframed source or target raise `ValueError`; other target types raise
`TypeError`. It was motivated by a round trip between a DICOM reader's index `(k, j, i)` array
and the same data framed with millimetre-offset `(z, y, x)` coordinates in one patient frame.

`array.rf.assume_frame(other)` re-targets an affine array transform to another frame's complete
identity, definition and context without moving samples. It requires full `CoordinateSystem`
equality and is implemented as unframe plus frame; mapping compatibility is still checked when
arrays combine. The NIfTI template identities and `dicom.patient_frame` that accompanied it are
adapter decisions, recorded with the adapters.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Copy coordinates into the view, or cache validation | A second authority that goes stale when the caller edits the array |
| Infer `dims` from coordinate dependencies | A coordinate on `time` would silently make time spatial |
| Validate all coordinate values at construction | Materializes chunked fields for a guarantee no query needs |
| Accept a `units` attribute that merely converts | No unit engine; no silent conversion |
| Derive support from spacing (`Cells(anchor)`) | Misdescribes thickness; kept only as a producer-declared default |
| CF bounds as a DataArray coordinate | xarray refuses a coordinate with an extra dimension |
| Parallel `*_lower`/`*_upper` coordinates | Not CF, unlinked, corrupted by `coarsen` |
| `pandas.IntervalIndex` coordinate | Cannot be written to netCDF |
| Categorical offsets (`center`, `left`; xgcm style) | A fraction is simpler; xgcm solves staggered-grid relations |
| Half-open `[0, 1)` offset | Here `1` is a distinct location, not the next cell's `0` |
| Offset measured by index order | Reversal would silently turn `s` into `1 - s` |
| Single-slice width from spacing | Spacing wearing thickness's name |
| Core `resample` returning framed arrays | Breaks layering; the accessor composes instead |
| A separate `relabel` API | Callers would need to know when it applies |

## Deferred Work

- CF bounds encoding, `cell_corners`, and a strict domain defined only inside declared cells.
- Curvilinear and irregular cells: dense cells share a vertex grid (VTK structured grids, UGRID);
  sparse cells are per-element shapes (CF 2-D bounds, footprints), possibly not boxes.
- Conversions for edge-counting conventions (plot extents, GDAL geotransforms), refused when no
  offset or interval is declared.
- Framed results for `coarsen` and `pad`; non-affine `assume_frame`; Dataset-level `resample_to`;
  `block_points` on `rf.resample_to`.
- Masked coordinates: only a mask that survives into a selected value is observable; whether
  xarray should keep masks on coordinates is an upstream question.

## Next Steps

1. Revisit per-access validation cost only if a workload shows it matters.
