# Changelog

## [Unreleased]

### Changed

- Clarified that the cells domain extends the outer bounds while interpolating interior gaps,
  with a concrete gap/edge example. Singleton coincidence and constant documentation now
  describe the implemented float64 roundoff allowances; numerical behavior is unchanged.
- Clarified indexed assignment checks: retained scalar coordinates on selected slices are
  ignored, while shared dimension coordinates must agree. Destination binding is preserved.
- Geometry dense points, default lattice columns and frame-coordinate fields now follow declared
  `dims` order, agreeing with positional queries and Grid snapshots after pixel transposition.
  Use `points().transpose(*order, "axis")` and `lattice(dims=order)` for pixel storage order.
  NGFF exports explicitly use storage-axis columns.
- `array.rf.frame(grid, replace_coordinates=False)` now refuses conflicting existing source
  coordinates. Set `replace_coordinates=True` to use the Grid declarations without changing
  pixels. Compatible coordinates retain attrs; replacements discard stale metadata.
- Clarified extension-transform stability and conditional Grid hashing: unhashable transforms
  remain valid for queries, equality and binding. Indexed framed-array assignment replaces
  destination values while retaining its binding, subject to xarray's coordinate agreement.

### Fixed

- The DICOM example-data tool preflights the complete batch before writing, refusing duplicate
  inputs/output names (including equivalent Unicode spellings and case variants), invalid
  paths, existing targets and source/output overlap. Each series publishes only after all files
  are saved in sibling staging; expected
  failures discard staging and allow independent later series to finish. Final stderr summaries
  and JSON stdout manifests report planned and published work even after failure or interruption.
  Dry runs create nothing; output requires a single writer.
- NGFF non-diagonal exports derive intrinsic axis types, units and spacings from contributing
  physical rows. Unmixed time and differently unitized spatial permutations retain their
  semantics; mixed columns require explicit spatial axes with the same unit. Unsupported
  v06 axis layouts now report actionable errors without reordering pixels.
- Linear resampling preserves exact samples and zero-weight neighbors near NaNs across
  affine and general paths, including independent complex components. Missing weights above
  the shared `1e-9` rounding allowance propagate NaN; integral gather classification now uses
  that bound instead of `1e-6`. Cubic prefilter NaN spreading and SciPy infinity behavior remain.

- Resampling accepts fully scalar Grid and selected Geometry/DataArray targets, preserving
  scalar coordinates, source context and lazy Dask output. Affine queries retain cropped
  source reads, including for cubic interpolation and cells-domain edge holding.
- `Geometry.points` accepts `axis_dim` and `units_coord` to avoid collisions with carried
  geometry names, and rejects conflicting names before evaluation. Geometry dimensions named
  `__xarrayrf_source_axis__` now work without overrides. Defaults remain `axis` and `units`;
  `point_at` and NumPy-returning `Grid.points` are unchanged.
- Geometry documentation now distinguishes the standalone query view from the existing
  native accessor and binding lifecycle.
- Local `ngff.open` paths now derive frame identity from the resolved absolute path,
  including symlinks. Relative and absolute openings of one store agree, and identical
  basenames in different directories no longer share a frame. Metadata-only `store=`
  remains caller-resolved. Existing persisted raw-path identities are unchanged; an
  explicit reader `frame=` can retain them when reopening.
- Singleton sample matching now uses float64 roundoff rather than a `1e-9` relative
  support window, preventing second-scale false matches at large coordinate origins.
  Affine and built-in composite inverse allowances account for cancellation; lookup,
  coincidence, and resampling share the tightened contract.
- DICOM imports retain the original pixel-stack length in a defaulted `source_count`
  field and reject missing or surplus source slices before selection or sorting.
  Enhanced selections still consume the full original stack. Manually constructed
  geometries with the default `None` retain selected-index validation; an explicit
  positive integer count must contain every selected source index and participates in
  dataclass equality.

### Changed

- `AffineTransform` now takes `basis_vectors` keyed by source axis and a required
  `target_axes` assertion for vector and translation component order. Vectors include scale
  and support general rectangular, sheared and singular maps. Existing matrix inputs move
  to `AffineTransform.from_matrix`; the old constructor is replaced without a deprecation
  shim during active pre-release development. Matrix storage and persistence schema 1 are
  unchanged.

### Added

- `ngff.open` accepts `frame=` as a frame or framed DataArray, matching every other reader.
  The override replaces anonymous or store-derived identities and composes derivable
  coordinate-system changes for OME-Zarr 0.4, 0.5 and 0.6.

- Anonymous reference frames with distinct persisted identities and visible repr markers.
  Sources without a declared world now import as anonymous; deliberate local frames and
  declared template, DICOM UID, NGFF store and CRS identities keep their existing meaning.
- Shared frame adoption for `rf.assume_frame` and adapter `frame=`, including exact derivable
  coordinate-system changes such as RAS to LPS. DICOM metadata/readers and GeoTIFF metadata/readers
  now accept a frame or framed array override. Different-frame refusals identify anonymous
  operands and name the applicable assumption/resampling remedy; adopting a world preserves
  grid and declared-support checks. DICOM reports missing UIDs as `anonymous-frame`.

- Anatomical grid operations: `anatomy.orientation_codes`, `reoriented` and `cardinal_grid`,
  with patient letters, RFC-4 tokens and DICOM display planes. Reorientation preserves samples
  and cells, including singleton reflections; cardinal targets cover exact source cell or
  sample corners with requested spacing and retain singleton slab support. Ambiguous anatomical
  assignments and undetermined default spacing refuse explicitly.

- Declared per-source-axis intervals on grids, geometry views and native bindings, including
  retained scalars, selection and roll propagation, exact support checks during alignment,
  all grid doors, target support on native resampling and provisional schema-1 persistence.
  Cells-domain queries use declared outer bounds, including singleton slabs. DICOM attaches
  slice intervals from `SliceThickness` in the slice coordinate's units.

- `Grid`: an immutable sampling value with no pixels, `points_at` for batched fractional
  positions, and `positions_at(outside="extrapolate")` for positions beyond the sample domain.
  `Geometry.grid()` snapshots sampling, and the provisional `grid` encoding kind persists it.
- Native grid binding through `rf.grid`, `rf.frame(grid)`, `xarrayrf.native.frame_array` and
  `grid_coordinates`, plus `Grid` targets for core and native resampling. Integer coordinates
  retain int64 values, including values above 2**53; grid equality includes numeric dtype kind.
  Every encoded grid coordinate declares `int64` or `float64` explicitly.
- Pre-alpha package metadata and a manual TestPyPI rehearsal workflow. This is release
  preparation; no package release or native-operation release pass is claimed.
- Independent repository scaffold, migrated architecture/review records, and local xarray
  patch guidance.
- Development-only scalar-binding feasibility probes.
- Value objects for the standalone coordinate contract, immutable and NumPy-only:
  `ReferenceFrame`, `CoordinateSystem`, `DirectionVocabulary`, `ArrayCoordinates` and
  `AffineTransform`. Terms and names follow `docs/core_interface.md`. Equality is exact and
  structural; there is no approximate geometric comparison.
  - `ReferenceFrame.local()` mints a private identity and `ReferenceFrame.declared()` adopts an
    external one such as a DICOM Frame of Reference UID; any namespace may be adopted, so a
    persisted minted identity can be re-adopted. Each frame is expressed in a
    `CoordinateSystem` and exposes its `axes` and `units`.
  - `definition`, `context` and `display` entries compare by type as well as value, so `True`
    and `1` are distinct and a boolean entry cannot hide a numeric conflict.
  - Immutability is documented as covering supported public use: reinitializing an instance or
    assigning to a private slot is outside the supported API.
- Frame comparison is split into separate questions. `is_equivalent_frame()` asks whether two
  declarations describe the same frame, comparing identifier, definition and context and
  ignoring the coordinate system, as Astropy does. `conflicts_with()` reports the same
  identifier with a contradictory definition or context. `==` also requires the same
  coordinate system, because transform endpoints need exact axes. The optional `role`
  (`"world"` or `"object"`) and `display` metadata are excluded from all of them.
  `with_coordinate_system()` re-expresses a frame without converting anything.
- Axis orientation. A `CoordinateSystem` may orient each axis with a direction token from a
  `DirectionVocabulary`: data an adapter supplies, written in the explicit `from-to` form of
  OME-NGFF RFC-4, with each opposite implied. Bare codes such as `RAI` are refused.
  `coordinate_system_change()` derives the signed permutation between two coordinate systems of
  one frame, matching axes by direction rather than name, or raises when it cannot.
  `CoordinateSystem.axis_codes()` ranks oriented axes by how closely a vector follows them and
  reports the residual angle.
- Units are open CF/UDUNITS strings compared exactly and never converted; empty or padded strings
  are refused, and Cartesian axes refuse known angular spellings, including CF geographic degrees
  such as `degrees_east`. Array coordinates may be angular. This replaces the earlier closed unit
  vocabulary.
- Transforms have a required `source` and `target` endpoint, each a `ReferenceFrame` or
  `ArrayCoordinates` (an array's own coordinate values: axis names and units, no identity). A
  transform from array coordinates locates an array's samples; a transform between frames is a
  registration or a coordinate-system change. Points are mapped from the source axes to the
  target axes.
  - The `Transform` protocol has separate capability protocols (`SupportsPoints`,
    `SupportsJacobian`, `SupportsAffine`, `SupportsInverse`), satisfied structurally by
    user-defined transforms; there is no registry.
  - `AffineTransform.from_matrix(source=..., target=..., matrix=..., translation=...)` implements
    `SupportsAffine`. `matrix` and `translation` return fresh read-only snapshots over their own
    buffers, so a caller can neither write to the transform's storage nor change it by editing an
    array header.
  - `transform_point(points)` maps an `(..., K)` array whose trailing axis follows the source
    axes, the ITK/VTK layout, and returns `(..., M)` in target-axis order.
  - `transform_named(transform, {axis: values})` evaluates any point transform by source axis
    name, broadcasting values (a retained scalar plane term stays cheap) and checking the shape
    and finiteness of what a user-defined transform returns.
  - `check_transform(transform)` validates that both endpoints are a `ReferenceFrame` or
    `ArrayCoordinates`; `transform_named` and `Geometry` call it.
  - `AffineTransform.inverse()` swaps the endpoints and inverts the matrix, only for a square
    matrix within `INVERSE_CONDITION_LIMIT`; a rectangular, singular or ill-conditioned affine
    raises `ValueError`. The inverse of a transform from array coordinates maps its frame back
    into those array coordinates.
  - `CompositeTransform(first, second, ...)` applies transforms first to last (unlike ITK's
    composite and VTK's default PreMultiply), requires each target to equal the next source
    exactly, and refuses array coordinates inside a chain. Its Jacobian follows the chain rule
    and its inverse reverses the members' inverses. `compose()` collapses a chain of affines into
    one `AffineTransform` and returns a `CompositeTransform` otherwise.
  - `jacobian(at=None)` returns the constant matrix, broadcast to `(..., M, K)` when points are
    given, so affine and location-dependent transforms share one layout.
  - Coefficients and coordinate values must have a real integer or floating dtype *after*
    ordinary NumPy inference. String, complex, boolean-dtype and object values raise `TypeError`
    instead of being parsed, losing an imaginary part or being read as 0 and 1. A masked array
    raises `TypeError` naming the mask, because `numpy.asarray` would discard it silently.
- `Geometry`: a read-through view of where one array's current samples sit in a reference frame.
  `Geometry(array, transform, dims=...)` requires a transform from `ArrayCoordinates`
  into a `ReferenceFrame` and validates that the source axes are real-valued coordinates of the
  array depending only on the declared geometry dimensions. `point_at(**positions)` locates a
  single sample through those coordinates' actual values and returns its point as a
  `DataArray` over the dimension `axis`, labelled by the frame's axis names and carrying their
  units. `frame`, `transform`, `dims`, `sizes` and `coordinate_dependencies`
  report the declaration and the current sample structure.
  - The array is the authoritative sample domain. The view keeps a reference to it, never a copy
    of its shape, coordinate values or validity, and every geometry-dependent property and query
    revalidates the current coordinate structure.
  - A coordinate's `attrs["units"]`, when present, must be a string exactly equal to the unit the
    transform's source declares for that axis; an absent key leaves the declaration in charge.
  - `dims` must be exactly the dimensions some source axis depends on: a declared
    dimension no source axis uses is refused, and a source axis depending on an undeclared
    dimension such as time or channel is refused. An empty `dims` is valid for a fully
    selected point.
  - Positions are zero-based, non-negative Python or NumPy integers bounded by the array's
    current sizes; a boolean is refused and a negative position raises `IndexError`.
  - Pixels are never read and no dense mesh of points is built; only the selected coordinate
    values are evaluated.
  - This is a query object, not an attachment: it binds nothing, registers no accessor, and
    gives the array no geometry that an xarray operation could propagate.
- `Geometry.points()` returns every sample's point, lazily when the array is Dask-backed.
  `Geometry.lattice()` returns a `Lattice` (origin, spacing, direction, matrix and homogeneous
  matrix, with a chosen column order) when the samples are regular. `Geometry.positions_at()`
  locates frame points as fractional positions, inverting nonuniform coordinates such as DICOM
  slice offsets. `Geometry.array` exposes the paired array.
- `resample(source, target, ...)` resamples values onto another array's samples with nearest,
  linear or cubic interpolation, carrying non-geometry dimensions through. Equivalent frames in
  different coordinate systems convert automatically; different frames need an explicit
  transform. Regular affine cases run as one composed affine through
  `scipy.ndimage.affine_transform`; others are processed in bounded-memory blocks. scipy is the
  optional `resample` extra. `benchmarks/resample_benchmark.py` times both paths against scipy
  alone.
- Data-only persistence: `encode` and `decode` convert the core value objects and user-defined
  transforms (`SupportsEncoding`, decoded only through caller-supplied `decoders`) to and from
  versioned JSON data, rebuilding through the constructors, with distinct `EncodingError`
  subclasses for each failure. The schema is provisional until NGFF fixtures pass.
- Per-axis types and undeclared units. `CoordinateSystem` and `ArrayCoordinates` take
  `axis_types` (open strings such as NGFF's `"space"`, `"time"`, `"channel"`; declarative, part
  of equality) and accept `None` as a unit where none is declared, distinct from `"1"`. An
  oriented axis must declare a unit; `coordinate_system_change` requires matched axes to keep
  their type; `Lattice.spacing` and `direction` refuse axes without a unit.
- `Geometry.is_coincident(other, *, tolerance=1e-6)`: the explicit tolerant geometry
  comparison, asking whether two arrays sample the same points of the same frame element for
  element. The tolerance is a fraction of the local step, so it is unit-independent; affine
  cases are checked exactly in time linear in the samples per dimension.
- Sample offsets and the cells domain. `ArrayCoordinates(..., sample_offset=...)` declares per
  axis whether samples stand for cells and where each sample sits in its cell: `None` for point
  samples (the default), or a fraction in `[0, 1]` measured toward higher coordinate values,
  `0.5` for centred voxels. `resample` and `Geometry.positions_at` take
  `domain="samples" | "cells"`: the default answers between the outer samples, as xarray's
  `interp` does; `"cells"` also answers in the outer samples' cells, holding the edge value as
  ITK's nearest and linear interpolators do; a point-sampled axis reaches no further than its
  samples, and an axis declaring cells with a single sample is refused. Nonuniform coordinates now admit points within
  rounding slack of the outer samples, as uniform ones already did.
- `Geometry.frame_coordinates(names=None, *, domain="samples")` returns lazy xarray coordinates
  giving each sample's frame coordinates, backed by a `CoordinateTransformIndex` subclass that
  maps through the array's actual source coordinates (uniform or not), survives slicing, aligns
  exactly and supports point-wise `sel(..., method="nearest")` by frame point with the same
  domain rule as `positions_at`. Coordinates defined by an xarray `RangeIndex` contribute their
  exact step to lattices and lookups.
- `AffineTransform.inverse()` measures conditioning after row and column equilibration, so the
  choice of units (seconds and metres, millimetres and nanometres) no longer causes a valid
  inverse to be refused. `Lattice.spacing` and `.direction` are refused unless every frame axis
  shares one unit. `Geometry`'s `spatial_dims` and `spatial_sizes` are named `dims` and `sizes`:
  geometry dimensions may include time, as for spacetime or fMRI arrays.
- The core modules never import xarray. `import xarrayrf` works without xarray installed, and
  `xarrayrf.Geometry` then raises `ImportError` naming xarray; a test enforces the boundary.
- Development-only `index_hook_probe.py`, which records the public xarray `Index` hooks
  stock xarray calls at operation boundaries.

- The native binding. `import xarrayrf.native` registers the DataArray `.rf` accessor:
  `rf.frame(transform, dims=...)` attaches a transform through a private index that owns the
  array's source coordinates, `rf.is_framed`, `rf.reference_frame`, `rf.coordinate_transform`,
  `rf.geometry_dims` and `rf.geometry` inspect it, `rf.unframe()` removes it, `rf.assume_frame()`
  adopts another frame's identity, `rf.resample_to()` resamples onto another framed array, and
  `rf.encode()`/`rf.decode()` persist it through the reserved `xarrayrf_binding` attribute.
  Ordinary xarray operations carry the binding, refuse (geometry `stack`, `pad`, `coarsen` and
  concatenation raise `ValueError` naming `rf.unframe()`; `join="override"` and incompatible
  framed grids raise `ValueError`; a plain index conflicting with a bound coordinate raises
  xarray's `AlignmentError`), or unframe when the result has no geometry in the source frame; a
  Dataset binding belongs to its geometry dimensions. `rf.frame` refuses an
  array still carrying an encoded binding, `rf.unframe` drops one, and an index that lost a
  coordinate to another index is neither framed nor unframed until unframed.
- Format adapters, each an extra: `xarrayrf.nifti` (NIfTI-1 import and export, template frames,
  `open`), `xarrayrf.dicom` (classic and enhanced series, registrations, `patient_frame`, `open`),
  `xarrayrf.ngff` (OME-Zarr 0.4, 0.5 and 0.6 import, 0.6 export with a loss report, RFC-4
  `orientation` read and written through `xarrayrf.anatomy.VOCABULARY`, `open`) and
  `xarrayrf.geotiff` (projected CRSs, frame axes in the CRS authority's order, `open`).
  `xarrayrf.anatomy` holds the RFC-4 anatomical direction vocabulary and
  `patient_coordinate_system`. `xarrayrf.units` maps UDUNITS names to the CF symbols every
  adapter emits (`canonical`) and back (`udunits_name`), so unit spelling never keeps frames
  from different formats from comparing equal (identity and declarations still must match). `xarrayrf.native.frame_array` and `index_coordinate` are the adapters' shared binding
  step and index-coordinate declaration.
- `affine_class(transform_or_lattice, *, tolerance=..., offset_tolerance=...)` classifies an
  affine by its matrix geometry (identity, translation, rigid, scaled, general, rectangular).
- `AffineTransform.with_endpoints(source=..., target=...)` keeps the coefficients between replaced
  endpoints. `compose` collapses nested all-affine chains. `Lattice` validates its frame, dims,
  origin and matrix at construction and compares by value.
- Readers return lazily loaded, framed DataArrays through dask (`chunks=`), reading only the
  source region a resampling target touches.

### Changed

- Adapter parameters no longer share the name `frames` with the reference-frame parameter
  `frame=`: DICOM's multiframe selection (`dicom.open`, `dicom.from_enhanced`) is now
  `frame_indices=`, and NGFF's mapping of frames resolved by an earlier import
  (`ngff.transform`, `ngff.from_multiscale`, `ngff.from_scene`) is now `resolved_frames=`. The old
  names are removed without aliases (pre-alpha, no users).
- `Grid.isel` uses xarray's positional indexing through a coordinate-only native binding,
  including lists, integer arrays and boolean masks. `Grid.sel` adds source-coordinate label
  selection. Both require xarray only when called; `transpose` stays NumPy-only.
- Grid lattice checks use stored coordinate values within tolerance. Unlike live Geometry
  queries, snapshots do not retain a `RangeIndex`'s exact step; this parity difference is
  documented.

### Fixed

- Canonical bound-coordinate and index order across native framing and decoding lets framed
  arrays combine with their scipy NetCDF round trips on stock xarray, including anonymous frames.
- Equal adopted anonymous bindings recommend `rf.assume_frame` alone even on empty grids.
- Resampling onto empty targets returns empty framed values; empty sources with non-empty
  targets raise a clear `ValueError` instead of failing during position calculation.

- `anatomy.cardinal_grid` accepts ambiguous source orientations when both `spacing` and `dims`
  are explicit; ambiguous defaults now request both arguments. Frame and rank validation,
  output directions and coverage do not require a source-axis assignment.
- Cardinal-grid non-anatomical roundoff bounds match each affine column to its own covered
  coordinate span, so source-coordinate rescaling cannot hide real variation.
- Singleton reflection's documented offset error uses the absolute bound `np.spacing(1.0)`;
  sample points and declared cells remain exact.

- Empty varying axes with declared intervals round-trip through core and native persistence,
  including empty selections and disjoint inner joins.
- Declared intervals require exact sample containment for varying axes and retained scalars,
  even when the offset agreement tolerance exceeds a narrow cell's width.
- Binding index equality returns `False` for differing coordinate indexes with repeated labels
  in either operand order; coordinate merges raise xarray's `MergeError`.
- Binding index equality returns a bool for interval mismatches and honours excluded
  dimensions; coordinate merges report conflicting intervals through xarray's `MergeError`.
  Direct index reindexing again refuses non-binding operands.
- Declared singleton intervals supply a step only in the cells domain, preserving the
  samples-domain coordinate matching tolerance. Interval validation allows float64 roundoff
  at large origins such as epoch seconds while still rejecting offset disagreements.
- `rf.frame(grid)` replaces differing existing coordinates on patched xarray by dropping
  them before binding, while preserving matching coordinates and their attrs.
- Grid coordinates remain immutable through copies and pickle round trips; editing a returned
  array's dtype or shape leaves its points and hash unchanged.
- Grid and Geometry sampling queries reject custom transform results with invalid shapes,
  non-real dtypes or non-finite values before applying the NaN outside mask.
- Fractional interpolation preserves stored endpoints exactly at integer positions, including
  ascending and descending coordinates spanning large magnitudes.
- Empty point and position batches return empty results on grids with size-zero dimensions.
- Integer-only coordinate input is checked before NumPy promotion, preserving mixed signed and
  unsigned integers exactly and refusing values outside int64.
- Resampling ignores unrelated target context and preserves source non-geometry coordinates;
  collisions with target geometry coordinate names raise a named `ValueError`.
- Core resampling retains attrs and custom indexes on non-geometry coordinates for both
  `Geometry` and `Grid` targets.
- Cubic resampling at array edges requires SciPy 1.18 or newer, which includes the upstream
  [spline boundary fix](https://github.com/scipy/scipy/pull/24615). The `resample` and `dev`
  extras now enforce that floor; SciPy 1.18 requires NumPy 2, while the core NumPy floor remains
  1.26.
- GeoTIFF frames for a northing-first CRS (EPSG:3035) had their axes labelled in authority order
  but evaluated in rasterio's easting/northing order; the affine rows are now permuted by axis
  direction, and directions other than east and north are refused at import.
- NGFF frames never compared equal to DICOM or NIfTI frames because unit spellings differed;
  adapters now share one spelling table.
- With the local xarray native-operation patches, framed geometry stacking, padding and
  coarsening now raise `ValueError` directing callers to `rf.unframe()`. Nongeometry operations
  preserve the binding, including coarsen reductions and `construct` for DataArray and Dataset;
  indexed empty rolls preserve geometry with either `roll_coords` setting. These guarantees
  require the pinned `xarrayrf-patches-3` series recorded in
  `docs/dev/xarray-upstream/xarray_patches.md`;
  stock xarray still has the corresponding gaps.

### Removed

- `xarrayrf.native.frame_dataarray`; use `xarrayrf.native.frame_array(data, grid, ...)` instead.
