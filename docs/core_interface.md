# Core interface and nomenclature

**Status:** Active (normative)
**Last updated:** 2026-10-08
**Scope:** The public vocabulary, value objects, transform protocol, calling conventions and
adapter contract of xarrayrf. Every public name follows this document. Where
[the core model design](dev/architecture/core_model_design.md) or [the architecture](design.md) disagree with it,
this document wins.

## Overview

The public surface has five layers. Each section below is normative for its layer.

| Layer | Names | Section |
|---|---|---|
| Value objects (NumPy only) | `ReferenceFrame` (declared, local, anonymous), `CoordinateSystem`, `DirectionVocabulary`, `ArrayCoordinates` | [Value objects](#value-objects), [Glossary](#glossary) |
| Transforms | `AffineTransform`, `CompositeTransform`, `compose`, `coordinate_system_change`, the `Transform` protocols | [Transforms](#transforms) |
| Sampling | `Grid` (frozen coordinates, no pixels), `Geometry` (live view of an array), `Lattice` (regular case), declared intervals, `xarrayrf.anatomy` grid operations | [Grid values](#grid-values), [Anatomical grids](#anatomical-grids), [Geometry queries](#geometry-queries) |
| Binding and operations | the `.rf` accessor (`frame`, `grid`, `geometry`, `resample_to`, `assume_frame`, `encode`/`decode`, `unframe`), `native.frame_array`, `native.grid_coordinates`, `resample` | [Native binding](#native-binding), [Native grid doors](#native-grid-doors), [Resampling](#resampling) |
| Persistence and adapters | `encode`/`decode`, `dicom`, `nifti`, `ngff`, `geotiff` | [Persistence](#persistence), [Adapters](#adapters) |

A framed array's identity question ("is this the same space?") is answered by its
`ReferenceFrame`; its sampling question ("which samples, where?") by its `Grid` or `Geometry`.
Combining arrays requires both to agree; `rf.assume_frame` and adapter `frame=` change only the
first, and `rf.resample_to` changes only the second.

## Native binding

Import `xarrayrf.native` to register the DataArray `.rf` accessor; importing `xarrayrf` alone
does not register it. `array.rf.frame(coordinate_transform, *, dims, intervals=None)` validates the pairing
with `Geometry` and attaches a private index owning the coordinate transform's source
coordinates. Its `ArrayCoordinates` source maps the array's coordinates into the target
`ReferenceFrame`. The returned array is **framed**. `array.rf.is_framed` reports the state, and
raises, as does every other member except `rf.unframe()`, when an operation left the binding
index on only some of its coordinates (a Dataset reduction on stock xarray, `expand_dims` of a
retained scalar); `rf.unframe()` clears that;
`array.rf.reference_frame`, `array.rf.coordinate_transform`, and `array.rf.geometry_dims`
expose the binding declaration.
`array.rf.geometry` builds a fresh `Geometry` from the current array; `array.rf.grid`
returns its `Grid` coordinate snapshot (reading geometry coordinates, never pixels).
`array.rf.unframe()` removes the binding while retaining ordinary coordinate values and
indexes. The binding has no declaration coordinate.

Indexed assignment from another framed array replaces destination values. It does not register
or resample the right operand and does not adopt its frame. Shared dimension coordinates must
agree with the indexed destination or xarray raises `IndexError`. Retained scalar coordinates
on a selected slice are ignored by this check, even when they differ from the destination's.
For two arrays bound to distinct frames with matching dimension labels:

```python
destination[{"y": slice(0, 2)}] = source.isel(y=slice(0, 2))
# destination keeps its original binding; the selected pixel values come from source.
```

With the [pinned native-operation patches](dev/xarray-upstream/xarray_patches.md#series-4-current-published),
stacking, padding (including zero width), or coarsening a geometry dimension raises
`ValueError` before losing its binding; the message directs callers to `rf.unframe()`.
This includes stacking a geometry dimension together with a nongeometry dimension and
constructing coarsen windows along geometry dimensions. Stack/unstack and pad of only
nongeometry dimensions preserve the binding. Nongeometry coarsen reductions and `construct`
preserve the exact binding on both DataArray and Dataset. Indexed empty rolls preserve the
empty geometry with either `roll_coords` setting. These fixes are included in the
`xarrayrf-patches-4` tag (the empty-roll and broadcast fixes are also on upstream xarray
`main`); strict conditional xfails record absent capabilities on other lanes.

`array.rf.encode()` returns an unframed copy sharing pixel data, with the reserved
`attrs["xarrayrf_binding"]` set to canonical schema-1 JSON text containing the encoded
transform, ordered geometry dimensions and an always-present `intervals` object, keyed by source
axis name (empty when undeclared). It replaces any earlier value under that key.
`array.rf.decode(*, decoders=None)` removes the attribute and calls `rf.frame` with
the decoded coordinate transform, geometry dimensions and intervals. This validates the binding against
the array's current coordinates. A raw netCDF or Zarr read remains unframed until explicitly
decoded. A missing attribute,
malformed payload or already framed input raises; custom transform decoders are caller supplied.
Schema 1 remains provisional.

The test suite marks known xarray gaps with strict xfails. A test for mixed labelled operands
requiring xarray PR #11532 is marked only when xarray is imported from the locked 2026.7.0
environment; it must pass against upstream main. The other strict xfails mark operations that
need the xarray index hooks of the patched lane.

Declared intervals live in the binding index, keyed by source axis name. Positional and label
selection (slices, strides, reversals, integer arrays and boolean masks) take rows with labels;
scalar selection retains the row. Coordinate rolls roll rows; pixel-only rolls leave them
unchanged. Rename and `swap_dims` carry support. Matched labels between bindings must have
exactly equal interval rows; missing or conflicting support refuses and names the source axis.
Index equality returns `False` for interval mismatches and ignores axes whose dimensions are
excluded from alignment. Coordinate merging reports conflicting intervals as xarray's
`MergeError`; joins and binding compatibility checks raise `ValueError` naming the axis.
Joins introducing labels take their rows from the binding operand that declares them. A plain
index in mixed alignment or a binding reindex target may select known labels, but cannot
introduce labels without declared support. Mixed-index alignment and `swap_dims` require the
existing optional xarray hooks; stock xarray's known refusals remain. Plain `DataArray.reindex`
/ `reindex_like` can be refused by xarray before dispatch, even for subsets on the current
patched lane; use label selection for such subsets. The public Index `reindex_like` hook refuses
non-binding operands. Concatenation along a geometry dimension refuses. Along any other
dimension (time, echo, a new stacking dimension) xarray only aligns the operands' bindings:
identical grids pass through, and differing grids join like any alignment, so pass
`join="exact"` to require identical ones (xarray is moving its `concat` default to `"exact"`).
Native resampling claims mapped source intervals on pass-through axes only when valid under the
unchanged target's `sample_offset`; average slab axes claim target intervals, while point slab
axes claim none. See the
[interval-claim design](dev/architecture/support_aware_resampling_design.md#first-slices-interval-claims-and-the-box-methods).
Core `resample` retains its ordinary unframed result contract.

## Glossary

Each term has one meaning. Public names, docstrings, errors and docs use these meanings only.

| Term | Definition |
|---|---|
| **Reference frame** | An identity: what coordinates are relative to, such as one patient's DICOM Frame of Reference or one microscope stage. `ReferenceFrame`. |
| **Coordinate system** | Ordered axes with units, optional axis types, a representation and optional orientation, in which a frame's coordinates are expressed. Carries no identity. `CoordinateSystem`. |
| **Array coordinates** | An array's own coordinate values along named dimensions, with units and no identity: the NGFF array coordinate system, napari's data coordinates. `ArrayCoordinates`. |
| **Anonymous frame** | Geometry whose source names no shared world. `ReferenceFrame.anonymous()` mints a distinct identity; `is_anonymous` reports it. |
| **Complete frame** | A declared external identity or a local identity deliberately created and shared by the caller. |
| **Local frame** | A frame whose identity the caller deliberately created (`ReferenceFrame.local()`), with no external identifier. It says nothing about how the frame relates to other frames; relations are always explicit transforms. |
| **Endpoint** | The source or target of a transform: a reference frame or array coordinates. |
| **Axis** | One named, ordered entry of a coordinate system or of array coordinates, with a unit and an optional type. ("Component" is not used.) |
| **Unit** | An open CF/UDUNITS string compared exactly, or `None` where no unit is declared. `None` is not `"1"` (dimensionless) and implies nothing else about the axis. |
| **Axis type** | An optional open string naming what kind of axis it is, such as NGFF's `"space"`, `"time"` and `"channel"`, Astropy's `SPECTRAL`, or a UCD. Declarative: nothing in the core depends on it, so no axis names or kinds are blessed. |
| **Position** | A zero-based index along an array or grid dimension, restarting at 0 after a crop. Sample positions are integers; interpolated positions may be fractional. |
| **Coordinate** | A value along an axis, as xarray uses the word. At import an adapter usually sets coordinates equal to positions; after a crop xarray keeps the original values. A coordinate is where its sample is. |
| **Cell** | The region an element stands for, such as a voxel, a pixel or a time bin: its nominal support, not its point-spread function or slice profile. By default cells are dense, tiling each axis from the current samples and the sample offset. Declared cells may leave gaps or overlap, as slices of a multislice 2D MRI stack do; a cell width derived from spacing never stands in for slice thickness. |
| **Sample offset** | Where a sample sits in its cell along one axis, in position units: a fraction in `[0, 1]` measured from the cell's edge at lower coordinate values, `0.5` for a centred voxel; `None` for point samples, which have no cells. The cell of position p spans positions p − s to p + 1 − s, mapped to coordinates as positions are (piecewise linearly for nonuniform values), so on a nonuniform axis the fraction holds in positions, not in coordinate distance. |
| **Domain** | Where `resample` and `positions_at` define values: `"samples"`, between the outer samples; `"cells"`, the sample hull extended to the outer edges of the outer samples' cells. Gaps between declared cells inside the hull are interpolated across in both. |
| **Point** | A tuple of coordinates, one per axis of an endpoint. |
| **Vector** | A displacement between points; transformed by the Jacobian J. |
| **Covariant vector** | A gradient or normal; transformed by J⁻ᵀ, never normalized. |
| **Transform** | A map from points of a source endpoint to points of a target endpoint. |
| **Coordinate transform** | A framed array's map from its `ArrayCoordinates` source to its target `ReferenceFrame`; exposed as `rf.coordinate_transform`. |
| **Affine** | A transform x ↦ Mx + t with constant M and t. "Linear" means t = 0 and is used only in that sense. |
| **Inverse** | The exact inverse where it is mathematically determined. Approximate or declared inverses are not inverses in this vocabulary (deferred). |
| **Direction** | A token `<from>-to-<to>` naming the way an axis increases, from a direction vocabulary. |
| **Role** | The descriptive label `"world"` or `"object"` on a frame. The word "world" is used for nothing else. |
| **Geometry dimensions** | The array dimensions a `Geometry`'s source axes vary along (`Geometry.dims`), which may include time, as for a spacetime or fMRI array. Every other dimension, such as echo or channel, is a non-geometry dimension, carried through unchanged. |
| **Grid** | A sampling declaration with frozen coordinates and no pixels: a coordinate transform plus the values of each 0-D or 1-D source coordinate and optional declared intervals. `Grid`; a framed array's snapshot is `rf.grid`. |
| **Declared interval** | A sample's declared cell, `[lo, hi]` in its source axis's own coordinate values, keyed by source axis name. Overrides the sample-offset default for the `"cells"` domain; may leave gaps or overlap. |
| **Orientation code** | For each varying dimension, the direction its index increases toward, as an RFC-4 token or a patient letter (`R` = toward the right). `xarrayrf.anatomy.orientation_codes`. |
| **Binding** | The association of a transform from array coordinates with a particular array, held by a private index that owns the source coordinates (`array.rf.frame`). |
| **Unframed** | An array with no binding. |

### Exported names and constants

Besides the classes and functions above, `xarrayrf` exports: `Endpoint` (the union type
`ReferenceFrame | ArrayCoordinates`), `Role` (the literal `"world" | "object"`), `AxisCode` (the
`axis`/`direction`/`angle` record `CoordinateSystem.axis_codes` returns), `Domain` (`"samples" | "cells"`),
`Method` (`"nearest" | "linear" | "cubic"`), `CARTESIAN` (the only representation), `LOCAL_NAMESPACE`
(the identifier namespace `ReferenceFrame.local` mints in), `SCHEMA_VERSION` (the persistence
schema, 1) and `INVERSE_CONDITION_LIMIT`. `xarrayrf.units.SYMBOLS` is the read-only spelling table;
`xarrayrf.ngff.NAMESPACE` (`"ome-zarr"`) is the identifier namespace of frames read from a store.
The numerical thresholds are module constants, not parameters: `LATTICE_TOLERANCE` (1e-6 of a step,
the default for `lattice`), `POSITION_SLACK` (1e-9 of a step, the edge slack when locating points),
`SINGLE_SAMPLE_TOLERANCE` (`8 * eps`, where `eps` is float64 machine epsilon; matching a
coordinate to a lone sample uses this times `max(1, |sample|, |query|)`, or the affine
inverse's allowance from absolute evaluation terms before cancellation, whichever is larger;
declared cells do not widen sample coincidence),
`EXACT_STEP_TOLERANCE` (1e-12 of a step, the exact-arithmetic inversion fast path) and
`BLOCK_POINTS` (the default `resample(block_points=)`, 2**20 target samples per block). Each adapter
package defines `__all__`; star imports expose only its public API.

## Value objects

**`CoordinateSystem(axes, units, *, axis_types=None, representation="cartesian",
vocabulary=None, orientation=None)`**. Units may be `None` (undeclared); an oriented axis must
declare a unit, since a direction needs the axis's quantity. Axis types join strict equality, and
`coordinate_system_change` requires matched axes to keep their type. Format-specific rules, such
as NGFF RFC-4's orientation only on spatial axes, are checked by adapters.

**`ReferenceFrame`**: identifier, `coordinate_system`, optional `role`, `definition`, `context`,
`display`; constructed by `ReferenceFrame.local(coordinate_system, ...)`,
`ReferenceFrame.anonymous(coordinate_system, *, role=None, definition=None, context=None,
display=None)` or `ReferenceFrame.declared(identifier, coordinate_system, ...)`. Comparisons:
`is_equivalent_frame`, `conflicts_with`, strict `==`; role and display never count.
`with_coordinate_system(coordinate_system)` re-expresses the same frame (same identity) in
another coordinate system, which adapters use to present a stored frame in their own axes.

Anonymous frames mint in `"xarrayrf.anonymous"`, separately from `"xarrayrf.local"`.
Every mint is distinct; identical metadata, files or paths never establish sharing. Local
frames remain deliberate caller-created worlds. `frame.is_anonymous` follows the identifier
namespace, including after `with_coordinate_system` and encoding/decoding. Equality, hashing,
definition and context follow the same rules as local frames. Frame, binding-index and grid
representations mark this state with `anonymous=True`. An anonymous array works alone and
with its derived arrays or any array explicitly sharing its identity. Unframed arrays remain
index space; no world is inferred.

**`ArrayCoordinates(axes, units, *, axis_types=None, sample_offset=None)`**: axis names equal to
the array coordinate names a transform reads, one unit (or `None`) and optional type per axis,
and one sample offset per axis
(`None` or a fraction in `[0, 1]`; omitted, all `None`). Immutable, structurally compared and
hashed, sample offsets included. No orientation, representation or identity.

Sample offsets are oriented by coordinate value, not by position order, so reversing, cropping or
striding an array leaves them true, while the cells are recomputed from the current samples
(striding widens them); operations that recompute coordinates, such as `coarsen`,
keep only a centred offset true. The offset never changes where a sample is: the transform
already maps each coordinate to its sample. It says where the sample's cell lies, which matters
only beyond the outer samples and when converting to conventions that count voxel edges.
Adapters set it from the format's specification (`0.5` for DICOM, NIfTI, ITK and NGFF), and
convert foreign coordinate conventions, such as corner-referenced ROI vertices, once on import.

**`DirectionVocabulary(identifier, directions)`**. Vocabularies are
data. The canonical anatomical vocabulary is RFC-4's pairs under one published identifier, defined
as `xarrayrf.anatomy.VOCABULARY` and checked by a conformance test. That module also exposes
the `RAS` and `LPS` orientation tuples and
`patient_coordinate_system(orientation, unit, *, axes=("x", "y", "z"))` for a three-axis
oriented spatial `CoordinateSystem`.

## Transforms

### Protocol

Every transform declares two endpoints and implements `SupportsPoints`, plus any of the other
capabilities; `check_transform` requires point mapping, so a Jacobian-only object is not a
transform here:

| Member | Meaning |
|---|---|
| `source` | A `ReferenceFrame` or `ArrayCoordinates`. Required; there is no default. |
| `target` | A `ReferenceFrame` or `ArrayCoordinates`. |

The input axes and units are the source endpoint's axes and units (a frame's coordinate system,
or the array coordinates'). There are no separate `inputs` or `input_units` members, so they
cannot disagree with the source.

| Capability protocol | Members | Meaning |
|---|---|---|
| `SupportsPoints` | `transform_point(points)` | Maps points; raises outside a valid domain, never extrapolates |
| `SupportsJacobian` | `jacobian(at=None)` | Local linear part, target axes by source axes; `at` required unless the transform is affine |
| `SupportsAffine` | `matrix`, `translation`, plus the two above | Constant M and t |
| `SupportsInverse` | `inverse()` | Exact inverse with endpoints swapped |

**`AffineTransform(*, source, target, target_axes, basis_vectors, translation)`** constructs
an affine from named vectors. `basis_vectors` is a mapping with exactly the source axis names;
its insertion order does not count. A vector gives the target displacement for a unit increase
in that source coordinate, including scale and unit conversion. Its components and the
translation follow `target_axes`, which must match the target's axis names and order exactly.
Every vector must be finite, real and 1-D with one component per target axis, validated before
assembly. Missing and extra names, differing target order, non-real values and wrong shapes
raise explicit errors. Mixed Python sequences follow NumPy promotion within each vector;
masked vectors are refused.

The vectors need not be orthogonal, independent or a complete target-space basis. Names such as
`k`, `j`, `i` do not imply directions such as `z`, `y`, `x`; the vectors establish that relation.
They describe coordinate-value increments, not array-position increments when the coordinates
are nonuniform or physical distances. Translation locates source coordinate zero, not
necessarily the first retained sample.

For a frame-to-frame map with source `moving` and target `fixed`, both using `(x, y, z)`, the
keys name **source** axes and the vector components name **target** axes:

```python
registration = AffineTransform(
    source=moving,
    target=fixed,
    target_axes=("x", "y", "z"),
    basis_vectors={"x": (0, 1, 0), "y": (-1, 0, 0), "z": (0, 0, 1)},
    translation=(10, 20, 30),
)
```

**`AffineTransform.from_matrix(*, source, target, matrix, translation)`** constructs the same
value from existing coefficients. Use this public constructor for matrices from registration
tools, file formats or numerical operations. Rows follow target axes; columns follow source
axes. Both constructors share coefficient validation, immutable storage, equality and hashing;
schema 1 continues to encode matrix and translation, independently of the construction path.
The old `AffineTransform(..., matrix=...)` call is replaced directly by `from_matrix` during
pre-release development.

`AffineTransform` implements `SupportsAffine` and `SupportsInverse`.
Its `inverse()` is computed only for a square matrix whose condition number,
measured after row and column equilibration so the choice of units does not matter, is within
`INVERSE_CONDITION_LIMIT` (1/(10·eps)), and raises `ValueError` otherwise:
a rectangular affine has no inverse, and projecting onto its image is a separate operation.
`with_endpoints(*, source=None, target=None)` returns the same coefficients between replaced
endpoints, for renaming source axes or adopting another frame's identity.

`affine_class(affine, *, tolerance=1e-6, offset_tolerance=1e-9) -> AffineClass` classifies a
`SupportsAffine` transform or `Lattice` by the resulting matrix, independent of how it was
estimated. The frozen result names the narrowest class: `identity`, `translation`, `rotation`,
`rigid`, `similarity`, `scaled_rigid`, `affine`, `singular` or `rectangular`. It also gives
`proper` (`det > 0` for a nonsingular square matrix), per-column `scales` and a dimensionless
`residual` for constrained classes, and `same_units`. A lattice's source positions are
dimensionless, so its class describes the matrix rather than a physical rigid motion unless
the frame axes are also dimensionless. `AffineClass.at_most(name)` tests containment:
`identity` belongs to both the translation and rotation branches, which meet at `rigid`;
`rigid ⊂ similarity ⊂ scaled_rigid ⊂ affine`. Singular and rectangular maps contain only
themselves. Unknown names raise `ValueError`.

A capability protocol states that the method exists. Where a particular instance cannot provide
it (a non-square affine's inverse, a chain whose member has no Jacobian) the method raises; it
never approximates.

User-defined transforms satisfy the protocols structurally; there is no registry. A transform
used in a binding or Grid must keep its endpoints, coefficients/behavior and equality stable
for their lifetime. Equality must return a scalar boolean reflecting its chosen declaration
identity; object-identity equality is valid. Hashing is optional, including for binding.
A hashable transform must provide an equality-consistent stable hash. Persistence requires
a versioned, data-only encoding.

### Endpoint rules

- `Geometry` and bindings require `source` to be `ArrayCoordinates` and `target` a
  `ReferenceFrame`.
- A transform between frames has two `ReferenceFrame` endpoints (registration, patient to
  equipment, coordinate-system change).
- The inverse of an array-coordinates transform maps a frame to `ArrayCoordinates`: the exact part
  of a world-to-sample lookup, followed by a coordinate-to-position lookup in the array.
- **Array coordinates may only begin or end a chain.** `ArrayCoordinates` has no identity, so two
  images with equally named coordinates compare equal. Composing through an `ArrayCoordinates`
  intermediate would connect unrelated arrays and is refused.

### Composition

`CompositeTransform(first, second, ...)` applies transforms first to last, requires each
`target == next.source` exactly, and refuses `ArrayCoordinates` intermediates. Its Jacobian
follows the chain rule and needs `at` unless every member is affine; its inverse is the members'
inverses in reverse order. ITK, SimpleITK and VTK's default PreMultiply apply the last added
transform first; the docstring says so.

`compose(first, second, ...)` returns a single collapsed `AffineTransform` when every member is
affine, so resampling and export can ask `isinstance(t, SupportsAffine)` of a chain, and a
`CompositeTransform` otherwise.

### Quantity rules

Specified, not yet implemented: `point`, `vector`, `covariant_vector`, `second_rank_tensor` (J·T·Jᵀ) and
`polar_rotation`, as plain callables over a trailing component axis in source-axis order. Domain
rules (diffusion PPD, ITK's J·T·J⁻¹) live downstream.

## Naming conventions

- `Supports…` names a capability protocol (`SupportsPoints`, `SupportsJacobian`,
  `SupportsAffine`, `SupportsInverse`), following Python's `typing.SupportsInt` convention.
- `…Transform` names the base protocol (`Transform`) or a concrete class (`AffineTransform`,
  later `CompositeTransform`).
- Python names are snake_case per PEP 8; ITK/VTK concepts are mirrored, not their CamelCase.

## Calling conventions

1. **Points are positional arrays** at the transform and coordinate-system level: a trailing axis
   in the endpoint's declared axis order.
2. **Names belong to the xarray layer and to functions named for them.** `Geometry` returns
   `DataArray`s whose axis dimension is labelled by axis names; `transform_named` evaluates any
   point transform from a mapping of axis name to values.
3. **Free functions for protocol-generic operations** that must work for user-defined transforms:
   `transform_named`, `check_transform`, `coordinate_system_change`, `compose`. **Methods for
   queries on a concrete value object**: `CoordinateSystem.axis_codes(vector)` replaces the free
   function `nearest_axis_codes(vector, coordinate_system)`.
4. **Positions enter through sampling queries on `Geometry` and `Grid`**, which read
   coordinates at positions and apply the transform. `point_at(**positions)` locates a sample;
   `points_at(positions)` locates fractional positions.

## Grid values

`Grid(transform, coordinates, *, intervals=None)` is a NumPy-only sampling
value with frozen coordinates, exported from `xarrayrf`. It describes geometry without pixels. The transform must map
`ArrayCoordinates` into a `ReferenceFrame`. Coordinates name exactly its source axes, with one
entry `name: (dim, values)` per varying axis (1-D values), or `name: value` per retained scalar
(0-D). At most one axis varies along each dimension. Values are copied into immutable storage,
must have real integer or floating dtype and must be finite; booleans are refused. Integer
input is stored as int64 (values outside its range refuse), floating input as float64. Sampling
math converts to float64; the declaration keeps integer values exact. Units come from the source.
Integer-only sequences, including mixed signed and unsigned NumPy integers, are checked before
dtype promotion so values cannot be rounded through float64.
`intervals` is an optional mapping keyed by **source axis name**, with one `[lo, hi]` row per
sample: shape `(n, 2)` for a varying coordinate, `(2,)` for a retained scalar. Missing axes
use the sample-offset default. Rows must be finite with finite positive widths (`lo < hi`), in
the source coordinate's own values and units. They are copied into immutable float64 buffers.
Each sample must equal `lo + s * (hi - lo)` for its declared `sample_offset=s`, within
`max(1e-9 * (hi - lo), 8 * eps * max(|lo|, |hi|))`, where `eps` is float64 machine epsilon.
The magnitude term admits rounding at large origins such as epoch seconds without relaxing
the width-relative check near zero. Independently of that tolerance, each sample must lie inside
its declared interval exactly: `lo <= sample <= hi`, for both varying axes and retained scalars.
Point-sampled axes (`s=None`) refuse intervals. Gaps and
overlaps are allowed; tiling is not required. `Grid.intervals` returns a read-only mapping (empty when
undeclared) of fresh read-only views, with the same header-isolation guarantee as coordinates.

The transform is retained by reference, without copying or runtime freezing. Built-in affine
transforms are immutable; composite stability and hashability depend on their members. Extension
transforms must honor the stable endpoint, behavior and scalar-boolean equality contract above.
`hash(grid)` requires a hashable transform with a stable, equality-consistent hash; an unhashable
transform remains valid for queries and equality, and `hash(grid)` raises `TypeError`.

`transform`, `frame`, `coordinates`, `dims` and read-only `sizes` expose the declaration. `dims`
follows the varying coordinates' insertion order. Empty dimensions and fully scalar grids are
valid. Equality compares the transform (including frame identity), dimension order and coordinate
values, dtype kind and declared intervals exactly; integer and float coordinates declare different grids even
when their numeric values match. Equality never uses coincidence tolerance. The representation names source axes,
target identity, dimensions, sizes and axes with declared intervals.
Coordinate arrays are fresh read-only views over immutable buffers; editing a returned array's
dtype or shape cannot change the stored declaration. Copies, deep copies and pickle round trips
reconstruct through the constructor and retain this guarantee.

`point_at`, `points_at`, `positions_at`, `lattice` and `is_coincident` use the same sampling
implementation and semantics as `Geometry`. `point_at` requires a nonnegative integer per
varying dimension and returns a NumPy point. `points()` returns a NumPy array shaped
`(*sizes, number_of_frame_axes)` in grid dimension order. `is_coincident` accepts another `Grid`;
`Geometry.is_coincident` continues to accept only `Geometry`.
Transform results must be finite real arrays of the declared target shape, including before
applying `outside="nan"`. Integer positions in `points_at` reproduce stored sample endpoints
exactly. An empty query batch returns an empty result with its batch shape and the appropriate
trailing point or position axis, even when a grid dimension has size zero. Non-empty queries
on empty axes are refused.

Grid lattices test uniformity from stored coordinate values within the requested tolerance;
grids carry no exact index-step metadata. `Geometry.lattice()` can use an xarray `RangeIndex`'s
exact step, so it can succeed when `Geometry.grid().lattice()` refuses because materialized
values at large magnitudes exceed that tolerance. This is a known parity difference.

`isel(**indexers)` delegates positional selection by dimension to xarray, accepting integers,
lists, integer arrays, boolean masks and slices, including negative indices and steps.
`sel(**indexers)` delegates label selection by source-coordinate name to xarray, including
lists and inclusive label slices. Scalar selections retain that source axis as a scalar.
Both operate on a coordinate-only Dataset carrying the native binding index; vectorized
indexers that change geometry dimensions are refused by that index. These methods import
xarray lazily and raise a clear `ImportError` when it is unavailable; Grid construction and
NumPy sampling queries still work without xarray.
`transpose(*dims)` names every varying dimension exactly once; with no arguments it reverses
their order and remains NumPy-only. All return grids whose points equal the corresponding
selection or permutation, preserving the coordinate transform, sample offsets and intervals.
Selection takes interval rows with the labels; reversing an axis never swaps `lo` and `hi`.
Scalar selection retains its `(2,)` interval with the fixed source coordinate.

### Anatomical grids

`xarrayrf.anatomy` supplies three operations on `Grid` values. They require an affine
transform (`SupportsAffine`) into a frame with three spatial (`axis_types="space"`) axes
oriented in `anatomy.VOCABULARY`, sharing one unit. Non-affine transforms raise `TypeError`;
an unsuitable frame raises `ValueError`.

An **orientation** names, per varying dimension in `grid.dims` order, the direction in which
its index increases. Accept a case-sensitive string of patient letters (the nibabel
`aff2axcodes` convention) or a tuple of RFC-4 direction tokens:

| Letter | Direction token |
|---|---|
| `R` | `left-to-right` |
| `L` | `right-to-left` |
| `A` | `posterior-to-anterior` |
| `P` | `anterior-to-posterior` |
| `S` | `inferior-to-superior` |
| `I` | `superior-to-inferior` |

Named planes use the DICOM display convention, with dimensions ordered `(slice, row, column)`:
`"axial"` and its alias `"transverse"` mean `"SPL"`, `"coronal"` means `"PIL"`, and
`"sagittal"` means `"RIP"`. They are accepted wherever an orientation is accepted. Direction
pairs must be distinct and the direction count must match the output dimensions. Invalid
directions, counts, or repeated pairs raise `ValueError`; unsupported argument types raise
`TypeError`. Codes are returned as a letter string when every direction has a patient letter,
otherwise as a tuple of direction tokens.

**`orientation_codes(grid)`** assigns varying dimensions to anatomical frame axes one-to-one
by maximizing the sum of absolute cosines between their unit step directions and the frame
axes, considering all permutations (including partial assignments for fewer than three dims).
Each sign comes from its assigned cosine. A step direction is the affine matrix column times
the sign of the coordinate step; descending coordinates point the other way. A size-1 dim
uses its column alone. Retained scalar axes are ignored; a fully scalar grid returns `""`.
The best and second-best scores differing by at most `ASSIGNMENT_TOLERANCE=1e-6` are ambiguous
and raise `ValueError`, asking the caller to resample to a cardinal grid first. This tolerance
detects exact ties; it does not bound obliquity. These are nearest-axis labels in the manner of
nibabel's `aff2axcodes`, so an oblique rotation's codes flip as it crosses 45°. Callers needing
an angular bound must read and bound the residual angle from `CoordinateSystem.axis_codes`,
consistent with that method's policy. Parallel or antiparallel anatomical step directions
instead refuse with "dimensions ... point along the same direction". More than three
varying dims, empty dims, nonmonotonic coordinates, and zero or non-finite anatomical step
lengths also refuse. Nonuniform but strictly monotonic coordinates can have an orientation.
Components along unoriented frame axes are ignored for anatomical direction and length.

**`reoriented(grid, orientation)`** returns the same samples with varying dims permuted by
`transpose` and reversed by negative-step `isel`. Corresponding sample points and cells are
unchanged; declared intervals follow selection. Requests must be a signed permutation of
the grid's assigned axes; otherwise `ValueError` names the available orientation (a plane
assigned to `S` and `P` cannot supply `L`). For a singleton reversal, selection has no step
to reverse: negate its affine matrix column and coordinate, mirror its declared interval
`[lo, hi]` to `[-hi, -lo]`, and complement its sample offset `s` to `1-s` only when it has a
declared interval. Offsets are measured toward higher coordinate values. A singleton without
an interval has no cells (the cells domain refuses it); reflection leaves its offset unchanged,
and a double reflection is an exact grid identity. With an interval, a double reflection restores
points, coordinates and intervals exactly, but `1-(1-s)` can round. The absolute offset error
is bounded by `np.spacing(1.0)`, one ulp at unit scale, rather than an offset-relative ulp.
This is the only case that changes the transform: the reflected transform is rebuilt as an
`AffineTransform` from the source's affine matrix, replacing any other `SupportsAffine` type.
It preserves the sample and cell exactly, including non-centred cells.
Negating the int64 minimum yields its exactly representable float64
opposite because the positive value does not fit int64. Other integer coordinates stay int64.
Nonsingleton reversals use `Grid.isel` and therefore require xarray.

**`cardinal_grid(grid, orientation, *, spacing=None, dims=None, cover="cells")`** builds a
target grid in the same frame for `rf.resample_to`. It requires three varying dims (size 1
is allowed) and refuses retained scalar axes. Output dims take the names of the source dims
assigned to the same frame axes; `dims=` overrides them with three unique, non-empty names.
Frame, affine-transform and direction-rank validation are independent of source-dim assignment:
the anatomical step directions must have rank three. Output directions come from the requested
orientation and the frame; coverage comes from projected source corners. Assignment is needed
only for default `spacing` or default `dims`. If either is omitted and assignment is ambiguous,
the error asks callers to pass explicit `spacing` and `dims`. Supplying both permits a 45°
volume to produce a cardinal target; `orientation_codes` still refuses the tie.
The output transform is an `AffineTransform` from dimensionless `ArrayCoordinates`, using
int64 coordinates `0..n-1`. The sample offsets are `(0.5,)*3` for cells mode and `(None,)*3`
for samples mode (point support).
Its columns are exactly `spacing[d] * direction[d]` along the requested signed frame axes.
Spacing never shrinks to land on the far bound.

`spacing=None` uses the assigned source dim's uniform coordinate step length in frame units,
with uniformity tested within `LATTICE_TOLERANCE=1e-6` coordinate steps. A size-1 source dim
uses its declared interval width times its column length. Nonuniform or undetermined spacing
refuses with a request for explicit spacing. A positive scalar specifies isotropic spacing;
three finite positive values specify spacing in output dim order. Invalid spacing or dim
counts raise `ValueError`.

Coverage projects the exact source corners onto the requested output directions:

- `cover="cells"` uses declared interval bounds where present, otherwise the sample-offset
  cells (outer steps extrapolate nonuniform coordinates). Every source dim must describe
  cells; point support or a singleton without an interval refuses. The first output cell's
  lower edge sits at the projected minimum. There are
  `max(1, ceil(extent / spacing - COVERAGE_TOLERANCE))` cells.
- `cover="samples"` uses the sample hull. The first sample sits at its projected minimum and
  there are `max(1, ceil(extent / spacing - COVERAGE_TOLERANCE) + 1)` samples. The last sample
  reaches or passes the maximum, within the tolerance.

`COVERAGE_TOLERANCE=1e-9` is in output steps and absorbs rounding near integer counts.
Coverage starts at the near edge; any overshoot (under one step) is on the far side.
In cells mode, a size-1 output dim with positive covered extent declares an interval of that
exact extent, expressed in its index coordinate units. Its sample is centred on the covered
extent, preserving a slab even when explicit spacing is wider than the slab. This singleton
support takes precedence over using a full spacing-wide cell. Samples mode declares no
intervals, including for singleton outputs; other output dims also declare no intervals.
Unknown cover modes, counts exceeding int64, or source variation along an unoriented frame
axis that a three-axis cardinal box cannot cover raise `ValueError`. The unoriented-axis
check uses the affine cross rows weighted by covered source coordinate spans and accepts
only floating-point roundoff: `CROSS_ROW_ROUNDOFF` (64 machine epsilons) times
`max_j(max_k(abs(M[k, j])) * span_j)`, where `span_j` is the covered source coordinate span
for column `j`. Each column's coefficient is matched to its own span, so rescaling a source
coordinate and inversely scaling its column leaves the bound unchanged. The check uses the
frame's unoriented axes, independently of assignment. It does not depend on the axes' units,
and a large translation cannot hide real variation; the output takes the midpoint of tolerated noise.

### Native grid doors

`array.rf.frame(grid, *, replace_coordinates=False)` requires every grid dimension to be an
array dimension with the same size. Missing source coordinates are supplied. Existing coordinates
are retained with their attrs when dimensions, values and dtype kind agree and any explicit
unit token matches the transform's source. An absent unit attr is compatible; descriptive attrs
do not cause conflicts. Conflicts raise `ValueError`, naming the coordinates and
`replace_coordinates=True`; malformed unit attrs raise `TypeError`.

`replace_coordinates=True` replaces conflicting declarations with the Grid's dimensions, values,
dtype and unit attrs, dropping stale descriptive attrs. It also replaces malformed unit attrs.
Compatible coordinates retain their metadata even with opt-in. This relabels samples without
resampling, unit conversion or pixel evaluation. Non-geometry coordinates, pixels and intervals
are preserved. The flag must be a bool; True with a transform raises `ValueError`.
A `dims=` argument with a Grid raises `TypeError`, even when `None`; the transform form requires
`dims`. Already framed arrays and arrays carrying an encoded binding refuse as in the transform
form. Passing `intervals=` with a Grid is refused; the grid supplies them.

All framing doors (`rf.frame` with a transform or Grid, `frame_array`, `grid_coordinates`,
and `rf.decode`) order bound coordinates canonically: varying coordinates in geometry dimension
order, then retained scalars in transform source-axis order. Coordinate and index mappings use
the same order, so equal bindings align on stock xarray after a serialization round trip.
Non-bound coordinates retain their relative order and metadata. The adapter orders remain
NIfTI `i, j, k[, t]` and DICOM `k, j, i` or `slice_offset, j, i`.

`xarrayrf.native.grid_coordinates(grid)` returns `xr.Coordinates` carrying the same binding
index; `array.assign_coords(grid_coordinates(grid))` frames an array with matching dimensions
and sizes. Coordinates carry `attrs={"units": unit}` when the transform's source unit is declared,
and no units attribute otherwise. Varying integer coordinates and retained integer scalars stay
int64. All native grid doors use this conversion and the existing binding index.

`xarrayrf.native.frame_array(data, grid, *, dims=None, coords=None, attrs=None)` builds a framed
DataArray sharing NumPy, Dask or other duck-array pixels without evaluating them. `dims` uniquely
names every array dimension, defaulting to the grid's geometry dimensions. `coords` maps names
to non-geometry `CoordinateSpec` declarations; redefining a grid coordinate refuses. Every
non-geometry dimension needs a coordinate declaring its size, and the complete declared shape
must equal `data.shape`. Geometry coordinates, dtypes and unit attrs come from the grid.
Non-geometry coordinates along a grid dimension must have that dimension's grid size.

## Geometry queries

`Geometry(array, transform, dims=..., intervals=None)` pairs an array with its transform from array
coordinates. A plain view has no intervals by default; explicit intervals follow the Grid
validation rules, and `Geometry.intervals` exposes their read-only views. Declared coordinates
are read to validate offset agreement; sampling queries recheck agreement against current values.
Constructing a standalone `Geometry` does not attach a binding or enforce xarray operation
lifecycles. Native `array.rf.geometry` exposes this query view over the attached binding and
passes its intervals. `grid()` snapshots them.

Declared sampling order governs `Geometry.dims`, native `geometry_dims`, Grid snapshots,
`points_at` inputs, `positions_at` outputs, dense points, default lattice columns and frame
coordinate fields. Transposing pixels retains the binding's order. Construct another Geometry
with reordered `dims`, or use `Grid.transpose`, to declare a different sampling order.
Dense Geometry points no longer positionally follow transposed pixels. To match pixel layout:

```python
order = tuple(d for d in geometry.array.dims if d in geometry.dims)
points = geometry.points().transpose(*order, "axis")
lattice = geometry.lattice(dims=order)
```

NGFF export explicitly uses pixel storage order for lattice columns after checking that every
array dimension is a geometry dimension. Named `point_at` queries are independent of order.
Beyond `point_at`:

- `grid()`: a `Grid` coordinate snapshot of the current coordinates, in `Geometry.dims` order.
  Coordinate values are read, pixels are never read. Multidimensional coordinates and multiple
  source axes along one dimension are refused because they cannot form a grid.
- `points(*, axis_dim="axis", units_coord="units")`: every sample's point as a `DataArray`
  over the geometry dimensions in declared `dims` order and the component dimension `axis_dim`,
  labelled in the frame's axis order with per-component `units_coord`; lazy and chunked like
  the array when it is Dask-backed. Both names must be nonempty strings, distinct and absent
  from the carried geometry dimensions/index coordinates. For a dimension named `axis` or
  `units`, use `points(axis_dim="component", units_coord="component_units")`. Unrelated
  coordinates omitted from the result do not restrict these names. `point_at` keeps its
  fixed `axis` and `units` names; use `.rename(axis="component", units="component_units")`
  to align an individual point with customized dense output. `Grid.points()` stays NumPy-only.
- `lattice(dims=None, *, tolerance=...)`: a `Lattice` (origin, spacing, direction, matrix and
  `affine`, the homogeneous index-to-frame matrix such as a NIfTI 4x4) when the transform is affine and every source axis is a retained
  scalar or a uniformly spaced one-dimensional coordinate. Columns default to declared `dims`
  order. Explicit `dims` orders the columns, for
  example `("i", "j", "k")` for ITK and NIfTI. Nonuniform, single-sample or field coordinates
  form no lattice. Coordinates defined by an xarray `RangeIndex` contribute their exact step.
  A `Lattice` maps positions to points with `transform_point(positions)` and is a value object
  (validated, compared and hashed by frame, dims, origin and matrix).
- `frame_coordinates(names=None, *, domain="samples")`: exact lazy xarray coordinates of
  each sample's frame coordinates in declared `dims` order, backed by a `CoordinateTransformIndex` that survives slicing;
  an interoperability aid for xarray, plotting and viewers, not an attachment. Selection is
  point-wise, with `DataArray` or `Variable` labels and `method="nearest"` (xarray's transform
  indexes accept no scalar labels), and alignment is exact only: equal frame coordinates align,
  and any other join raises xarray's `NotImplementedError`. Nonuniform strictly monotonic source coordinates are supported; no lattice is
  required. Reverse mapping through the actual source coordinates agrees exactly with
  `positions_at`, including its `domain` rule. Admitted outer-cell positions are clipped to
  the edge sample before rounding for nearest selection. Retained scalars support forward
  mapping only; a single-sample dimension admits only its own coordinate in reverse unless its declared
  interval supplies a cells-domain width.
- `positions_at(points, *, outside="raise", domain="samples")`: frame points to fractional
  positions, through the exact inverse and a per-axis coordinate-to-position inversion
  (arithmetic for uniform coordinates, monotonic interpolation for nonuniform ones, extended by
  the outer steps). A retained scalar axis, a field, or two axes along one dimension are refused.
  With `domain="cells"`, positions in the outer cells lie outside `[0, n - 1]`, such as `-0.5`,
  and are returned as they are.
- `points_at(positions, *, domain="samples", outside="raise")`: the forward partner, mapping
  positions shaped `(..., D)` in `dims` order to NumPy frame points shaped `(..., M)`. Uniform
  coordinates map linearly; nonuniform coordinates map piecewise linearly, extending by the
  outer steps. Retained scalar axes are read from the array or grid, not supplied in positions.
  Fields and multiple axes along one dimension are refused. A single-sample dimension accepts
  position zero but has no step for fractional positions or extrapolation without a declared
  interval in the cells domain. Only in that domain does an interval supply its width as the
  position step: `p=0` is the sample and positions `-s` and `1-s` map to `lo` and `hi`. Empty axes cannot
  locate non-empty position batches; empty batches return empty points. Likewise, `positions_at`
  accepts empty point batches on empty axes.
- `is_coincident(other, *, tolerance=1e-6)`: whether both arrays sample the same points of the
  same frame, element for element, so one's values stand for the other's without resampling.
  The explicit tolerant comparison; `==` on transforms and frames stays exact. Frames must be
  equal or equivalent (compared through `coordinate_system_change`); different frames are never
  coincident. Geometry dimensions pair by name and must have equal sizes. The tolerance is in
  steps (local spacing), as ITK's coordinate tolerance is a fraction of spacing, so it is
  unit-independent. With affine transforms the check is exact and linear in the samples per
  dimension; otherwise every sample is checked in blocks. A single-sample dimension must agree
  to rounding; declared cells do not widen sample coincidence. Cells, offsets and values are not compared.

Both fractional queries accept `outside="raise"` (refuse points outside the domain), `"nan"`
(mark every component of an outside row as NaN) and `"extrapolate"` (extend beyond the domain
using each axis's outer step). `domain="samples"` bounds positions to `[0, n - 1]`;
`domain="cells"` extends to the outer declared interval bounds when present, and otherwise to
the sample-offset edges. Interior gaps remain interpolated; overlaps are allowed. Single-sample
cell-width refusal applies only without an interval. Extrapolation still validates the domain declaration. Forward and
inverse queries round-trip on strictly monotonic axes with a numerically accurate transform inverse,
including extrapolated nonuniform positions: offsets `0, 2, 5, 9` map positions
`-1, 0, 1.5, 4` to coordinates `-2, 0, 3.5, 13`. Retained scalars continue to refuse inverse
lookup and coincidence because projection is a separate policy.
In the samples domain, a singleton returns position zero only for a match within float64
roundoff, with or without declared intervals. The coordinate allowance is
`8 * eps * max(1, |sample|, |query|)`. Affine inverses also account for the absolute terms before
cancellation: `8 * eps * (K + 1) * (|M| @ |point| + |translation|)`, where `K` is the number of
input axes and the coefficients are those of the inverse. For built-in composite inverses,
these allowances propagate through affine members. A non-affine inverse member resets any
accumulated allowance because its error-propagation scale is not declared. Such a mixed chain
with large translations may fail to locate its own singleton sample; no scale preservation
is guessed. Non-affine providers use the coordinate allowance and are responsible for their
inverse's numerical accuracy; affine providers should expose `SupportsAffine`. This is a rounding
allowance, not an origin-relative physical support window. An interval supplies no step for
samples-domain fractional extrapolation.

## Resampling

`resample` is explicit and lives in the xarray layer. For each target sample it computes the point
in the target frame, maps it into the source frame, then through the inverse of the source's
transform into source array coordinates; converts each coordinate value to a fractional position
(directly for index coordinates, by monotonic inversion for one-dimensional auxiliary coordinates;
multidimensional coordinate fields are refused); and interpolates values at those positions,
carrying non-geometry dimensions through. It is format-neutral and lazy.

`source.rf.resample_to(target, *, transform=None, method="linear", fill_value=np.nan,
domain="samples", support="point", min_coverage=0.5, return_coverage=False)` accepts a `Grid`,
framed DataArray or `Geometry`, calls the core `resample`, and binds
the result to the target's coordinate transform and geometry dimensions. Target pixels are ignored.
Core `resample(source, target, ...)` takes a source `Geometry`
and a target `Geometry` or `Grid`; a grid produces the same values as its equivalent framed
array geometry. Source non-geometry coordinates retain their dtypes, attrs and custom indexes.
The target contributes only coordinates named by its transform's source axes. Unrelated target
coordinates, including scalar context, are ignored; non-geometry coordinates come from the
source. If a target geometry coordinate name collides with a source non-geometry coordinate,
resampling raises `ValueError` naming that coordinate instead of replacing it.

`method="step"` and `method="overlap_mean"` reconstruct the pointwise mean of covering source
slabs, whose values represent means over their declared intervals. `step` additionally refuses
source overlap on slab axes; touching intervals are allowed. A point on a shared edge takes
the mean of both slabs. `support="point"` evaluates this reconstruction, while
`support="average"` averages it over the covered part of each declared target interval.
Coverage is the product of per-axis covered fractions (pass-through axes contribute 1).
Gaps give fill, as do fractions below `min_coverage` (default 0.5, valid range `(0, 1]`).
Partial results are means over covered support. Unlike point interpolation, box methods do not
bridge gaps, and they answer out to the outer slab edges rather than the outer slab centres.
`min_coverage` is validated but ignored by
other methods. The default linear point interpolation is unchanged; average support with
nearest, linear or cubic is not yet supported.

Box methods require real or complex floating values (convert integers explicitly), separable
affine coordinate maps, and declared source intervals on every slab axis; averaging also
requires target intervals there. Axes selecting coincident source samples pass through;
for average support, target intervals must be absent or match mapped source intervals to
qualify. Resample other axes first with a point method or declare their intervals. Non-default
`domain` refuses because declared support defines the box domain. NaN components propagate
only from positive weights, independently for real and imaginary parts. Output is float64 or
complex128. This path uses sparse tensor weights without cropping or same-grid gathering;
Dask remains lazy, with geometry dimensions rechunked as core dimensions.

Native `return_coverage=True` returns `(values, coverage)` for box methods only. Coverage is a
framed float64 DataArray with the target's geometry dimensions and transform, no context
dimensions and no claimed intervals. Point slab axes contribute 0 or 1. Core `resample` keeps
its unframed DataArray return type and accepts `support` and `min_coverage` only.

Input, frame and empty-source validation runs before loading optional SciPy. Valid sampling
requests require the `resample` extra, including empty-target requests; without it they raise
`ImportError` with an installation hint.
The result retains the source's name, ordinary attributes and non-geometry coordinates; a
reserved `xarrayrf_binding` attribute is not carried, and `rf.frame` refuses an array that still
has one (decode it or drop it first). `rf.unframe()` drops it as stale. The core
`resample` continues to return an unframed array. `array.rf.assume_frame(other)` accepts a
`ReferenceFrame` or framed DataArray and adopts its complete identity, definition and context
through the same private adoption implementation as adapter `frame=`. Both accept a frame or a
framed DataArray and retarget an affine to that full declaration, including its coordinate
system, role and display. If systems differ, adoption composes the exact
`coordinate_system_change` (for example RAS to LPS); sample values and source coordinates stay
unchanged. Underivable changes refuse with the reason, including differing units, direction
vocabularies or unoriented axes, and non-affine mappings refuse explicitly. No unit conversion
or registration is guessed. Passing an anonymous partner retains its anonymous namespace while
explicitly sharing that partner's identity. Subsequent combination still checks transform,
dimensions, retained coordinates and declared intervals; adopting a world never matches a grid.

When different frames meet in alignment, arithmetic, binding joins or resampling without an
explicit transform, and either is anonymous, the refusal identifies the left/right operand
(or source/target for resampling). For derivable affine systems it names `frame=` or
`rf.assume_frame` to assert the shared world, followed by `rf.resample_to` for different grids.
A separate private comparison, independent of `is_coincident`, reports numerically matching
sample points after the derivable system change. It allows only accumulated float64 roundoff,
uses affine extrema in time linear in axis lengths, and never establishes identity. When the
adopted bindings also agree, the message says `rf.assume_frame alone suffices`, even for empty
grids whose numerical point comparison is undefined. Numerically
matching points with different bindings or support still require resampling. Underivable
systems and non-affine mappings name the reason and request an explicit transform, without
suggesting an assumption that would fail. Existing refusals for two complete frames remain
unchanged.

A fully scalar Grid or selected Geometry/DataArray target has shape `()` and represents one
point. Resampling retains its scalar geometry coordinates and the source's non-geometry
dimensions and context; `rf.resample_to` also preserves the source name and attributes and
binds the target geometry. Dask output stays lazy, with no dummy dimension. Scalar affine
queries still crop source reads to the interpolation neighborhood. Retained scalar source
axes remain unsupported: this does not introduce projection or scalar-source inversion.

An empty target returns an empty result with the target geometry and the source's non-geometry
dimensions and coordinates; `rf.resample_to` binds it as usual. This also works with an empty
source and remains lazy for Dask pixels. An empty source with a non-empty target raises
`ValueError` because there is nothing to sample from, regardless of `fill_value` or domain.
`fill_value` applies to points outside an existing source domain. Empty targets require no
point location or interpolation, but frame identity, transform endpoints, method, domain and
coordinate-name compatibility are still validated.

Linear interpolation preserves exact samples near NaNs. For affected source slices, it
interpolates zero-filled values and a missing-weight mask using the same kernel and boundary
rules. Missing weights above `ROUNDING_ALLOWANCE=1e-9` propagate NaN; weights at or below it
are treated as roundoff, without renormalizing the tiny finite-value deficit. Complex real
and imaginary components are handled independently. Outside fill is applied afterward, and
cells still hold edge values. The same `1e-9` allowance classifies signed-permutation gathers.
It covers measured index round trips on 31-degree oblique grids with origins up to 120000
and spacings down to 0.037 (maximum error 4.66e-10), not arbitrary ill-scaled custom transforms.
Cubic spline prefiltering may spread NaNs throughout coefficients; same-grid gathers retain
the original samples. Infinity behavior remains SciPy's existing behavior for interpolated
queries; this is not normalized masked interpolation.

Performance is part of the contract:

- When both arrays form lattices and every transform is affine, target positions map to source
  positions through one composed affine. Signed-permutation maps with integral offset, bounded
  within `1e-9` source samples at every target-box corner, gather source samples without
  interpolation; samples outside the source use the selected domain's fill or edge rule.
  Other maps use `scipy.ndimage.affine_transform` in a compiled pass per non-geometry
  slice, with a separable outside mask. For finite data at 128³ with a rotation, linear runs
  within about 15% and cubic within about 5% of scipy applying the same composed map; the mask
  adds about 0.01 s
  (`benchmarks/resample_benchmark.py`).
- Otherwise target positions are processed in blocks of `block_points` samples (bounded
  memory), positions are reused for every non-geometry slice when no linear NaN buffers are
  needed, and values are interpolated by `scipy.ndimage.map_coordinates`.
- Linear NaN buffers and cubic spline coefficients are prepared once per source slice and held
  for only one slice at a time. General-path positions are cached up to `POSITION_CACHE_BYTES`
  (256 MiB), or recomputed per slice beyond that budget. A Dask-backed source is processed
  lazily, one task per chunk of its non-geometry dimensions; positions and masks are shared
  within a task, not across tasks.
- A single-sample target dimension (a one-slice target) keeps the fast path. Complex sources
  are resampled as complex128.
- scipy is the optional `resample` extra; the core does not import it.

- Equivalent frames in the same coordinate system need no frame transform.
- Equivalent frames in different coordinate systems (LPS and RAS) use the derived
  `coordinate_system_change` automatically: it is exact, and the call is already explicit.
- Frames that are not equivalent require a caller-supplied frame-to-frame transform; numerically
  identical declarations never substitute for identity.

The domain is explicit. By default values are defined between the outer source samples and
`fill_value` marks the rest, as xarray's `interp` and scipy's constant mode do. With
`domain="cells"` values are also defined in the outer samples' cells, holding the edge sample's
value for every method. ITK's domain is the same, and its nearest and linear
interpolators also hold the edge, while its B-spline mirrors. An axis declaring point samples
has no cells and reaches no further than its samples, as for echo times or time points alongside
voxel axes. Declared intervals override sample-offset bounds; gaps inside the sample hull
remain interpolated. An axis declaring cells with a single sample is refused only when it has
no interval: no neighbour determines its cell width. A singleton with a declared interval
maps linearly across that slab and holds its sole sample value. Interpolation between samples
is the same in both domains.
The cells domain adds nothing for nearest and linear; cubic re-evaluates the shell samples at
clamped positions, at a cost proportional to the shell.

## Persistence

`encode(value)` and `decode(data, *, decoders=None)` convert vocabularies, coordinate systems,
frames, array coordinates, grids, affine and composite transforms, and user-defined transforms to and from
versioned, plain JSON data. Decoding rebuilds through the public constructors; frame identity,
metadata types and float64 coefficients survive exactly. User-defined transforms implement
`SupportsEncoding` (namespaced `kind`, own `version`, `to_data()`) and decode only through
caller-supplied `decoders`; nothing in the data selects code. Failures are `EncodingError`
subclasses: `UnsupportedVersionError`, `UnknownKindError`, `MissingDecoderError`,
`MalformedDataError`, `DecoderResultError`. The schema is provisional until the NGFF boundary
fixtures pass (`docs/dev/architecture/persistence_design.md`).

Provisional schema 1 includes a `grid` kind with `transform`, `dims` and `coordinates`
fields. Each coordinate is encoded as `{"dim": ..., "values": [...], "dtype": ...}` or
`{"value": ..., "dtype": ...}` for a scalar. Every record requires `"dtype": "int64"` or
`"dtype": "float64"`, including empty coordinates. `dims` fixes the dimension order, so tools
that reorder JSON object keys cannot change it. int64 records accept only JSON integers fitting
in int64; float64 records accept finite numbers, including integral JSON numbers. The explicit
dtype preserves numeric kind when tools rewrite `2.0` as `2`. Integer values never pass through
float64. Missing or unknown dtypes and unknown fields at either level are refused; numbers are
checked strictly, and constructor errors become
chained `MalformedDataError`.
Intervals are not encoded in stages 1 or 2. Schema 1 is not frozen.

## Units

Open CF/UDUNITS strings compared exactly; no conversion; Cartesian axes refuse known angular
spellings including CF geographic degrees. Pint-backed conversion is an optional later helper.
`xarrayrf.units.canonical(unit)` maps the UDUNITS-2 names and plurals that OME-NGFF prefers to
the CF symbols every adapter emits (`"micrometer"` to `"um"`), and `udunits_name(unit)` maps
back for export; unknown spellings pass through. Adapters apply it at their boundary so frames
read from different formats compare equal; the core itself never relabels a unit.

## Adapters

Format adapters are optional subpackages of the xarrayrf distribution, each with an extra
(`xarrayrf[dicom]`, `xarrayrf[nifti]`, `xarrayrf[ngff]`, `xarrayrf[geotiff]`). They share
`xarrayrf.native.frame_array` (duck-array and shape check, then framing on the imported
`Grid`), `index_coordinate`, and the `CoordinateSpec`,
`Report` and `DuckArray` aliases.
`xarrayrf.anatomy` holds the canonical anatomical vocabulary and grid operations; it depends
only on the NumPy core. Reorientation delegates nonsingleton reversal to `Grid.isel`, which
imports xarray lazily.

- The core never imports an adapter; an adapter imports only the core (including
  `xarrayrf.native`), `xarrayrf.anatomy` and its own format library. The import-boundary test
  enforces both directions.
- Metadata functions consume format-library objects without reading pixels. Each adapter's
  `open` reads metadata and binds pixels through its existing `to_dataarray`; by default,
  pixel reads are deferred through dask until compute. `chunks=None` reads eagerly. Adapters
  do not depend on application objects.
- **Import** returns frames, the coordinates to place on the array (with `units` attributes), the
  transform from those array coordinates into the frame, any transforms between frames, and a
  report of every normalization applied and its tolerance.
- **Framing** takes an import result and caller-supplied duck-array pixels, checks their type and
  shape, constructs a `DataArray` from its `dims` and `coords`, and calls `array.rf.frame` with its
  coordinate transform. Dask pixels remain lazy. DICOM sorts or selects source-order slices using `order`.
- **Export** writes native metadata from a frame, a transform and the array's current
  coordinates, and refuses what the format cannot represent. It never resamples.
- After import, xarrayrf never calls back into the format.

Every reader accepts `frame=None` or a frame/framed DataArray, as do NIfTI `from_header`,
DICOM `from_datasets`/`from_enhanced`, and GeoTIFF `from_profile`. A supplied frame
explicitly overrides the imported identity through the shared adoption contract above,
even a declared DICOM Frame of Reference UID, CRS authority or NGFF store identity. Metadata
consistency checks within a source still apply; DICOM reports the override as `frame-override`,
naming any discarded Frame of Reference UID. `frame=` is always the identity to adopt; DICOM's
multiframe selection is `frame_indices=`, and NGFF's reuse of earlier imports is `resolved_frames=`.

**File readers:** `xarrayrf.nifti.open(path, *, frame=None, template=None, xform="best",
spatial_unit="mm", time=False, chunks="auto")`, `xarrayrf.dicom.open(paths, *,
frame=None, series_uid=None, frame_indices=None, modality_lut=True, orientation_tolerance=1e-4,
slice_tolerance=0.01, chunks="auto")`, and `xarrayrf.ngff.open(store, *, group="",
multiscale=None, level=None, frame=None, chunks="auto")` each return one framed `DataArray`. Default
pixels are lazy dask arrays. NIfTI uses nibabel's scaled proxy dtype and values. DICOM accepts
a directory, file, or explicit file sequence; one enhanced object or one classic series is
selected. Modality LUT output is float64 when requested, and MONOCHROME1 values are not display
inverted. Compressed DICOM decoding may require `pylibjpeg`, `pylibjpeg-libjpeg`,
`pylibjpeg-openjpeg`, or `python-gdcm`; decoder errors arise at compute time. NGFF selects the
first level unless a dataset path is supplied. If several multiscales exist, select one by index
or name.

NGFF 0.4 and 0.5 multiscales are parsed with `ome_zarr_models`, evaluated through its 0.6
conversion, and imported through `from_multiscale` after folding multiscale-level scale and
translation into each level's single intrinsic transform. This preserves time calibration and
omits channel axes from geometry. Path-based transforms are unsupported. For 0.6, `open` binds
the intrinsic system; additional transforms remain available through `from_multiscale`.

**DICOM import:** `xarrayrf.dicom.from_datasets(datasets, *, frame=None, orientation_tolerance=1e-4,
slice_tolerance=0.01) -> DicomGeometry` imports a classic single-frame stack, and
`xarrayrf.dicom.from_enhanced(dataset, *, frame_indices=None, frame=None,
orientation_tolerance=1e-4, slice_tolerance=0.01) -> DicomGeometry` imports one enhanced
multiframe object. `frame_indices` selects the zero-based indices of the frames forming one
stack. Both consume metadata-only pydicom datasets and refuse
duplicate positions. `DicomGeometry` is a frozen declaration with `dims`, `coords`, `transform`,
`frame`, `order`, `patient_position`, `slice_intervals`, `report` and `source_count`. `order` maps sorted slices to
input dataset or frame indices. The frame is declared as
`("dicom-frame-of-reference", FrameOfReferenceUID)` (`xarrayrf.dicom.FRAME_OF_REFERENCE_NAMESPACE`)
when the UID is present and is otherwise anonymous; Patient Position is a result field, not defining frame context.

`xarrayrf.dicom.to_dataarray(geometry, data) -> DataArray` takes a source pixel stack with slice
axis 0 (`k`). For `from_datasets`, it has one slice per dataset in the supplied order. For
`from_enhanced`, it is the full pixel array in original frame order, including unselected frames.
The function applies `order` before binding and attaches `slice_intervals` to the slice source
axis when present, including single-slice thickness in the cells domain. Imported geometry
retains the exact original stack length as `source_count` and rejects both missing and surplus
source slices before applying `order`. Enhanced selections still require the full original
stack, not just the selected frames. The new field defaults to `None` for compatibility with
manually constructed `DicomGeometry` objects; those retain minimum selected-index validation.
An explicit count must be a positive integer, contain every selected source index, and
participates in dataclass equality.

Dimensions are `(k, j, i)`. Uniform stacks have an index `k`; nonuniform and single-slice
stacks carry a `slice_offset` coordinate in mm on dimension `k`. `slice_intervals` contains each
slice's thickness interval in the slice coordinate's units, or `None` when any positive
`SliceThickness` is unavailable. In-plane columns follow DICOM Pixel Spacing's row, column
order. The report records orientation correction, slice-axis choice and missing or differing
metadata. Quadruped orientation is unsupported.

`xarrayrf.dicom.patient_frame(frame_of_reference_uid)` returns the shared LPS patient frame for
a UID, or a fresh anonymous frame when the UID is absent. Application readers can use it to frame
their own spatial stacks in the same patient world. `dicom.open` remains a single-stack reader;
when several images occupy one slice position, an application-level reader must assemble the
extra echo, diffusion or time dimension and frame each spatial stack.

`xarrayrf.dicom.equipment_transform(dataset, *, orientation_tolerance=1e-4) ->
AffineTransform | None` imports an Image to Equipment Mapping Matrix, or returns `None` when it is
absent. Its source is the dataset's LPS mm patient frame, declared by Frame of Reference UID when
present and otherwise anonymous. Its target is a fresh anonymous equipment frame with unoriented `x`, `y`,
`z` axes in mm; its definition records `EquipmentCoordinateSystemIdentification` and available
`Manufacturer` and `DeviceSerialNumber`. Only `ISOCENTER` and a finite rigid matrix with proper
rotation and homogeneous last row are accepted. Equipment frames from separate datasets do not
share an identity: the equipment identifier names a type, not a unique instance.

`xarrayrf.dicom.registrations(dataset, *, orientation_tolerance=1e-4) ->
tuple[AffineTransform, ...]` imports one transform per Spatial Registration
`RegistrationSequence` item, from that item's Frame of Reference UID to the dataset's own Frame
of Reference UID. Each item's Matrix Sequence is multiplied in listed order (`M1 @ M2 @ ...` on
column vectors), so the last matrix acts first. An item must contain one Matrix Registration
Sequence item with a nonempty Matrix Sequence. `RIGID` requires a proper orthonormal rotation;
`RIGID_SCALE` requires orthogonal nonzero columns; `AFFINE` allows any finite linear part. All
require a homogeneous last row within `orientation_tolerance`. Missing own or item Frame of
Reference UIDs, referenced-image-only items, unsupported matrix types and Deformable Spatial
Registration objects are refused; resolving referenced images is the caller's responsibility.

**NIfTI import:** `xarrayrf.nifti.from_header(header, *, frame=None, template=None, xform="best",
spatial_unit="mm", time=False) -> NiftiGeometry` accepts a NIfTI-1 or NIfTI-2 header without
reading image data. `NiftiGeometry` is a frozen declaration with `dims`, `coords`, `transform`,
`frame` and `report`. `coords` maps coordinate names to xarray-compatible
`(dimension, values, {"units": unit})` tuples; a 2-D header retains a scalar `k=0`. Source
axes `i`, `j`, `k` are integer index coordinates with centred sample offsets.
`xarrayrf.nifti.to_dataarray(geometry, data) -> DataArray` binds caller-supplied pixels in header
dimension order.

`"best"` selects a coded sform before a coded qform; either can be selected explicitly. Import
is refused when neither xform is coded. Spatial units map from `xyzt_units` codes to `m`, `mm` or
`um`; an
unknown unit uses the caller's `spatial_unit` and is reported, or is refused when that argument
is `None`. Selected xform code 4 declares the shared `("nifti-template", "MNI152")` frame with
definition `{"space": "MNI152", "variant": "unspecified"}`; code 3 similarly declares
`("nifti-template", "Talairach")`. `template=name` declares `("templateflow", name)` with
definition `{"space": name}` for a nonempty alphanumeric BIDS space label and selected code 2–5.
Named and unnamed variants have distinct identities. `template=` cannot be combined with
`frame=` or `time=True`; `frame=` also accepts a framed DataArray. Other codes mint anonymous frames. `time=True`
always mints an anonymous frame unless `frame=` is supplied, since a spatial template does not declare
a shared clock. The selected xform, codes and qfac appear in the report and in anonymous frame
definitions only. A supplied frame must be reachable through `coordinate_system_change` from a
RAS view of that same identity. With `time=False`, a time dimension is a non-geometry coordinate derived from
`toffset` and `pixdim[4]` (an integer index, reported, when `pixdim[4]` is not positive);
`time=True` adds an unoriented time axis when the header declares a time unit. Frequency units
remain non-geometry.

**NIfTI export:** `xarrayrf.nifti.to_header(geometry, *, dims, xform_code="scanner",
qform=True, header=None) -> (header, report)` returns a new NIfTI-1 header, or a copy of a supplied
NIfTI-1 or NIfTI-2 header. `dims` gives the three or four geometry dimensions in NIfTI voxel-axis
order, regardless of array storage order. Export uses `Geometry.lattice`, converts the frame to
RAS, writes a coded sform, and writes a qform only when the spatial columns are orthogonal. A
sheared sform is retained, with a `qform-unrepresentable` report entry when qform was requested.
Spatial units must be `m`, `mm` or `um`. Four-dimensional export requires a time axis in `s`,
`ms` or `us`, with zero space-time cross terms; its step and origin become `pixdim[4]` and
`toffset`. Three-dimensional export preserves a supplied header's time fields. Export refuses
nonuniform, single-sample, field and non-affine geometry, unsupported units, frames that cannot
convert to RAS, and coupled space-time mappings. Two-dimensional export is deferred.

**NGFF (OME-Zarr 0.6 import).** `xarrayrf.ngff.coordinate_system(cs, *, store=None,
group="") -> tuple[ReferenceFrame, Report]` imports a named v06 system.
`transform(t, systems, *, store=None, group="", dims=None, resolved_frames=None) ->
tuple[AffineTransform, Report]` imports a transform between named systems or a path-only array
endpoint. `from_multiscale(ms, *, shapes, dims=None, store=None, group="",
resolved_frames=None) -> NgffMultiscale` returns a shared intrinsic frame, levels keyed by path (each with
`dims`, xarray-compatible integer-index `coords`, and `transform`), additional transforms, and
an import `report`. `from_scene(scene, *, systems, store=None, group="", resolved_frames=None) ->
NgffScene`
returns frame-to-frame `transforms` and an import `report`. All four accept v06 models or
equivalent JSON attributes; the adapter reads metadata only. `shapes` supplies each dataset's
array shape, `dims` supplies dimension names by path, and a scene's `systems` supplies systems
of referenced images. Axis
names, units, and types become the core coordinate system; a missing unit becomes `None`, while
an unnamed axis or non-string unit is refused.
`xarrayrf.ngff.to_dataarray(level, data) -> DataArray` binds caller-supplied pixels to one
`NgffLevel` in its declared dimension order.

With `store`, a named system's identity is `("ome-zarr", "store/group#quoted-name")`, omitting
the group slash at the root. Metadata-only `store=` is a caller-resolved URI or canonical
absolute filesystem path; it is used verbatim, without filesystem access or working-directory
resolution. `ngff.open` resolves local paths and symlinks to a bare absolute filesystem path
before deriving identity. Different spellings of one local store therefore share a frame;
the same relative basename in different directories does not. URLs retain their supplied
identity spelling apart from the trailing slash. To reuse a reader frame through
`resolved_frames`, supply that same resolved absolute path as `store`. Previously persisted
raw-path identities are not rewritten: reopen with an explicit `frame=` if their identity must
be retained. `store` has no trailing slash; `group` is relative and has no
empty, `.` or `..` segments. Without `store`, frames are anonymous; pass a mapping from
`(group path, name)` to `ReferenceFrame` as `resolved_frames` to reuse them across imports. A path-only
array endpoint
uses centred cells (`sample_offset=0.5`) and unit `"1"`. All affine members, including rectangular
`affine`, `rotation`, `mapAxis`, `projectAxis`, `sequence`, `byDimension` and an inverse-checked
`bijection`, become one `AffineTransform`. Sequence members apply in listed order. Identity-mapped
channel and other discrete axes are omitted from geometry and reported by multiscale import;
mixed discrete axes are refused. Array-backed parameters and nonlinear transforms are refused.
The NGFF adapter reads and writes RFC-4 `orientation` through `xarrayrf.anatomy.VOCABULARY`.
`discrete` and `longName` declarations and checked `bijection` inverses are deferred; multiscale
reports their loss.

**NGFF export:** `xarrayrf.ngff.to_multiscale_level(geometry, *, path, name="intrinsic",
frame_name="physical", store=None) -> tuple[Multiscale, Report]` exports one regular affine level
as a validated v06 multiscale model with one dataset. A diagonal lattice uses the frame's axes
as its intrinsic system and a dataset sequence of scale then translation. For a non-diagonal
lattice, intrinsic axes use array dimension names. A column with one nonzero frame row inherits
that row's axis type and unit (including `None`), and its spacing is the coefficient magnitude.
A column mixing rows is supported only when all contributing rows are explicitly spatial and
have the same declared unit, using the Euclidean column norm. Adapter unit spelling aliases
agree (`mm` and `millimeter`), but different scales (`mm` and `um`) do not; no numbers are
converted. Mixed time/space, multiple time rows, unknown types and missing or incompatible
units in mixed columns are refused with the array axis and contributing frame declarations.
Support uses exact structural zeros: a near-90-degree space/time rotation with a tiny nonzero
cross term is still mixed. Supply actual zeros for an intended permutation; same-unit spatial
rotations remain valid. Zero columns are refused before normalization. The dataset sequence
scales by these spacings and translates by zero. One additional
multiscale-level affine maps that intrinsic system to `frame_name`, whose axes are the frame's,
using the normalized lattice matrix and origin. This synthesis is reported as
`intrinsic-synthesized`. Both systems are validated against v06 axis-order/count constraints;
errors name intrinsic dimensions/types and advise transposing array geometry for ordering or
providing explicit supported declarations for type counts. Export never reorders pixels.
A level needs one array dimension per intrinsic axis; rectangular
mappings remain available through `to_transform`. `xarrayrf.ngff.to_transform(t, *, names=None)`
exports a frame-to-frame affine and returns `tuple[Affine, Report]`. `names` maps reference frames
to coordinate-system names declared alongside the export; without a mapping entry, only a frame
declared in the `ome-zarr` namespace supplies its own NGFF name. Export refuses non-lattice
geometry, non-affine transforms and invalid v06 multiscale declarations. Reports name frame
identity beyond the NGFF name, `definition`, `context`, `role`, `display`, direction vocabulary and
orientation outside the anatomical vocabulary, non-centred sample offsets, array-coordinate units
other than `"1"`, and array-coordinate axis types. The v06 path-only array endpoint cannot retain
those declarations.

**GeoTIFF import:** `xarrayrf.geotiff.from_profile(profile, *, frame=None, area_or_point="Area") ->
GeoTiffGeometry` imports rasterio metadata without reading pixels. `GeoTiffGeometry` holds
`dims`, `coords`, `transform`, `frame`, `report` and `nodata`. `to_dataarray(geometry, data)`
binds pixels in `(band, row, column)` order; band is a non-geometry dimension with 1-based
coordinates. `geotiff.open(path, *, frame=None, bands=None, chunks="auto")` reads projected GeoTIFFs,
with one independent windowed read per dask chunk, or eagerly with `chunks=None`.
`bands` selects 1-based band indices. `nodata` is recorded in attrs but masks and scale/offset
are not applied. `crs_frame(crs)` returns the frame for a pyproj or rasterio CRS. Exact
authority codes share identity (`("epsg", code)` for EPSG); CRSs without an exact authority
mint anonymous frames containing WKT. Axis names and linear units come from pyproj. The affine
locates rasterio pixel centres for both `AREA_OR_POINT=Area` and `Point`; Area declares centred
cells, while Point declares point samples. Angular, compound and vertical CRSs are refused.

Application policy stays in the application: multi-image series assembly, pixel handling,
regions of interest, display conventions and the choice of resampling targets.

## Removed and deferred

- **Full native lifecycle support is deferred.** `array.rf.frame` attaches an experimental
  binding, while known native-operation gaps remain release blockers until the inventory and
  xarray hook stages complete.

- **Orientation-only poses are removed.** A transform whose translation is unknown cannot map
  points honestly, and no mainstream library models one. Relating a moved subject to a scanner is
  an explicit affine the caller builds (choosing or registering the translation). Direction
  queries need no special type: the vector rule uses only an affine's matrix.
- **Approximate and declared inverses are deferred** to a separately named, opt-in capability.
- Geodetic and spherical representations, product transforms and declared domains are recorded
  in the [prior-art study](dev/architecture/prior_art_coverage_study.md). Per-axis type and undeclared units
  are implemented; NGFF's `discrete` and `longName` are deferred.
- **Channel axes stay non-geometry dimensions.** Interpolating across channels is meaningless and
  nothing marks an axis discrete yet, so adapters must keep channel axes out of the geometry, and
  `resample` carries them through. The NGFF adapter drops identity-mapped channel axes from transforms.
