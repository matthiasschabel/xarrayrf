# Reference frames as an xarray extension

**Status:** Active
**Last updated:** 2026-10-06
**Scope:** Independent reference-frame semantics for xarray; DICOM and other scientific-image
producers; geometry-aware downstream algorithms. The value objects, the `Grid` sampling value and
the `Geometry` view, declared cell intervals, anatomical grid operations, complete and anonymous
frames, the native `.rf` binding (a private index that jointly owns an array's source
coordinates), persistence and the DICOM, NIfTI, NGFF and GeoTIFF adapters are implemented. The native-operation lifecycle is
measured per xarray lane in [the operation inventory](dev/binding/binding_operation_inventory.md): complete on
the pinned patched lane, with the stock-lane gaps tracked by the upstream pull requests in
[the patch manifest](dev/xarray-upstream/xarray_patches.md).


> **Terminology.** This document predates [the core interface](core_interface.md), which is
> normative. Here "world" coordinates means coordinates in the target reference frame, "mapping
> input" means a source axis of a transform from array coordinates, and "component" means an axis.

## Context

### Problem Statement

Labelled arrays should carry enough information to locate their samples in a reference space,
so downstream code does not repeatedly pass a separate reference frame. The extension should
fit xarray's named dimensions, coordinates, alignment, broadcasting and lazy arrays, support
2-D and higher-dimensional data, and let producers attach geometry once.

The design is driven by concrete use cases across medical imaging (DICOM, NIfTI), microscopy
(OME-NGFF), astronomy (Astropy WCS) and remote sensing (GIS rasters), recorded in the
[cross-domain cases](dev/architecture/cross_domain_cases.md). No
application's class hierarchy, serialization format or compatibility policy is preserved; format
producers such as a DICOM reader are ordinary clients that attach geometry once.

Mixed operations follow one rule: an unframed operand supplies values under ordinary array
semantics, with no independent world-space or identity claim. Its index labels still
assert correspondence under the framed operand's ordinary xarray indexes, which may differ from
the auxiliary coordinate values read by the world mapping. It may participate in
an operation with framed operands without first acquiring its own frame. The result retains
geometry when the operation supplies a valid relationship between output coordinates and
physical locations. Equal array size alone is neither necessary nor sufficient.

The rationale behind the core model and the acceptance cases are summarised in
[the core model design](dev/architecture/core_model_design.md) and
[the cross-domain cases](dev/architecture/cross_domain_cases.md).

## Current Decision

The public vocabulary, endpoint model, calling conventions and adapter contract are specified
normatively in [the core interface](core_interface.md); it takes precedence over this document.

### Recommended Architecture

Use native DataArray coordinates to describe the current sample domain and attach an
independent reference-space declaration plus a mapping from coordinates to world locations.
A thin accessor constructs and inspects that binding. Xarray operations supply the coordinate
transformations; an integration layer validates compatibility and propagates or invalidates
the binding. There is no companion mutable frame shape or hidden parent image.

```text
DICOM producer       generic image producer       optional WCS adapter
        \                    |                         /
         +--- construct space + coordinate mapping + Grid ---+
                                   |
                                   v      (Grid: the same sampling without pixels,
           native DataArray: values + coordinates + binding    for targets and ROIs)
                                   |
                 xarray operations and alignment
                 + reference-binding validation
                                   |
                                   v
                native DataArray with current geometry
                         /                      \
                segmentation                 registration

Generic persistence stores geometry. Domain export owns image provenance and encoding.
```

Responsibilities:

- **Reference space:** define the meaning, units and identity of world coordinates.
- **Coordinate mapping:** map declared input coordinates to that space, independently of array size.
- **Array binding:** associate the mapping with actual coordinates on an individual variable.
- **Xarray integration:** apply operation semantics and validate all declared bindings.
- **Producer adapters:** construct valid bindings from domain metadata, without participating in
  subsequent generic arithmetic or selection.

- **Sampling values:** describe which samples a geometry has, with or without pixels
  (`Grid`, its live counterpart `Geometry`, declared intervals), so targets, regions of interest
  and file grid blocks need no second grid type in applications.

These are conceptual responsibilities, not a requirement for five public classes. Start with
one immutable declaration and an accessor-derived geometry view. Keep construction, validation
and representation handling in cohesive modules; no class hierarchy or provider registry.

### Reuse and the core/adapter boundary

Reuse, adapt or repurpose existing implementations where their semantics fit this design.
A partial fit can justify reusing a component or improving its upstream implementation without
adopting the surrounding framework. Evaluate dependency cost, maintenance, licensing and the
amount of adaptation together; neither importing an entire ecosystem nor rewriting useful code
is the default. Preserve attribution and applicable licenses when adapting source.

The core owns the common contract: reference-space declarations, coordinate mappings, binding
to xarray variables, compatibility and operation lifecycle. Domain adapters own interpretation
and normalization of source metadata, authority/name resolution, domain conventions, specialized
transform engines, and format-specific import/export. A provider may evaluate a specialized
mapping through a narrow generic capability contract; ordinary propagation must not rediscover
geometry through the original producer object.

Keep extension points sufficient for concrete DICOM, NGFF, astronomy and GIS cases, including
named systems, 2-D/ND geometry and future nonlinear mappings. Supporting these cases does not
require their domain rules or dependencies in the core. When an adapter cannot express a valid
case cleanly, identify the missing general capability and test it across relevant examples
before extending the core. Do not accumulate domain-specific branches or build a speculative
universal framework. These use cases test the architecture; they are not a commitment to ship
every adapter in the first release.

### 1. A frame describes coordinates; xarray owns the sampled domain

Let `c1, ..., ck` be coordinates carried by a DataArray. Its binding declares:

`world = F(c1, ..., ck)` in reference space `S`.

Each coordinate has named dimension dependencies. The array's current coordinates and sizes
define which samples exist. A geometry view reads them now; it does not cache an old shape or
selection history. Neither the immutable mapping nor space owns pixel values.

The first representation is an affine map of coordinate values:

`world = translation + A @ c`.

`A` has M rows and K columns; no requirement for M or K to be three, or for A to be square,
orthonormal or invertible. Input units and world-component units are explicit; A includes any
needed scale conversion. Input coordinates may be nonuniform, scalar after selection, or
multidimensional after vectorized indexing. They need not be dimension-index coordinates.
Affine in these coordinate values does not imply uniformly sampled array indices.

Do not silently assume named x/y/z coordinates mean world coordinates. A producer explicitly
binds existing coordinates or creates them. For an oblique image, local row/column coordinates
remain useful indexes while F maps them into patient space. An already-cropped array uses its
actual coordinate values when the mapping is constructed; no legacy origin convention is
part of this contract.

Keep distinct:

- array rank, which may include time, echo, channel, parameter or ensemble dimensions;
- dimension dependencies of the spatial sample coordinates;
- number of mapping inputs K and world dimensions M;
- geometric capability: a regular grid, nonuniform grid, embedded plane, point set or constant
  point. A point list is not automatically a line suitable for interpolation.

A 2-D array may describe a 2-D space or a plane in 3-D. Selecting a plane keeps the fixed
coordinate as a scalar term, so its physical position survives. Transposition changes storage
order, not the meaning of named coordinate dependencies. Genuine 2-D data needs no synthetic
third axis or slice thickness. Algorithm-specific lattice/rank requirements are checked when
an algorithm requires them, not imposed on every array.

#### Sample anchors and index origins

Array positions are zero-based; mapping inputs are coordinate values and may instead use
one-based labels or physical distances. The binding locates the sample reference point, usually
a voxel centre, without inferring voxel boundaries or physical support. Adapters encode index
origin and centre/corner offsets once in their coordinates and mapping. For first sample centre
C and spacing s, zero-based i uses C+s*i, one-based j uses C+s*(j-1); a lower cell boundary B
instead gives B+s*(i+0.5). The latter assumes a uniformly sampled cell-centred grid. No global
half-voxel correction is applied. Crop/stride retains original coordinate values; cell bounds,
thickness and support require separate geometry when needed.

The sample's place in its cell is declared, not inferred.
`ArrayCoordinates.sample_offset` records it per axis (`None` for point samples, a fraction in
`[0, 1]` measured toward higher coordinate values, `0.5` for centred voxels). It never moves a sample, since coordinates already locate
samples; it defines default cells from neighbouring sample spacing. Resampling uses
it only when asked for `domain="cells"`; the default domain stays the samples, as in xarray.
Declared `Grid.intervals` give each sample's bounds in its source coordinate values, describing
physical thickness, gaps and overlap; the binding carries them through supported operations.
DICOM import declares intervals from `SliceThickness`. The spacing-derived default is a declared
tiling convention, not acquisition support, and a single slice needs declared bounds to have
cell width. The cells domain reaches the outer bounds and interpolates interior gaps; it is
not a union of isolated supports. See the normative
[sampling contract](core_interface.md#geometry-queries) and the
[geometry rationale](dev/architecture/geometry_and_resampling_design.md).

**What a binding is, and what it is not.** The binding is exactly one thing: a mapping from an
array's own coordinate values to points in one declared reference frame, `world = F(c)`. Crop,
stride and selection change which coordinate values exist, not `F`. Four adjacent concepts stay
outside it, and conflating any of them with the binding is the failure mode to avoid:

- **A transform between two reference frames** is a separately declared relation. Applying it
  changes which frame a sample's point is attributed to; it never moves, resamples or
  reinterprets the samples, is never applied implicitly during arithmetic, and two frames are
  related only when a caller supplies the relation.
- **Choosing an output sample domain** is an explicit operation with its own declared geometry,
  never implied by a relation, and array size is never evidence about it.
- **Interpolation** evaluates values where nothing was sampled; it needs a declared interpolant
  and a representation capable of it. Interpolating a coordinate field and resampling an image
  are separate operations with separate validity rules.
- **Registration**, estimating a relation from data, is outside the core, as is any rule that
  treats two independently constructed frames as the same because their names, numbers or
  extents agree. A downstream consumer supplies a validated relation and owns its optimizer.

### 2. Named reference systems, coordinate representation and identity

A DICOM FrameOfReferenceUID is already a name for a specific frame, as is an authority-qualified
identifier in another domain. Distinguish the identifier from the definition that gives
coordinates meaning. A readable name alone is a label, not evidence that two frames coincide.

Two immutable value objects keep these roles separate: a `ReferenceFrame` holds identity,
definition, context and display metadata, and is expressed in a `CoordinateSystem` holding the
coordinate representation. The structure, frame roles, orientation and transforms between
frames are specified in [the core model design](dev/architecture/core_model_design.md).

| Role | Meaning and ownership |
|---|---|
| Identifier | Opaque namespaced identity for a particular frame; globally identified, scoped to a container, or minted locally. DICOM FoR UIDs belong here. |
| Definition | Optional versioned domain declaration needed to interpret an externally defined system. An adapter resolves and validates it at construction; the core does not look up names. |
| Coordinate representation | The `CoordinateSystem`: ordered axes, axis units, representation kind and optional axis orientation from an adapter-supplied direction vocabulary; these describe the output of F. First implementation: Cartesian. |
| Defining context | Only parameters needed to interpret that system or transformation, such as an applicable epoch, observer or realization. Acquisition timestamps are not automatically defining context. |
| Display metadata | Human-readable labels and provenance, excluded from geometric compatibility. |

One axis-unit declaration is authoritative. Mapping coefficients carry conversion from
declared input units into those output units; they do not maintain a competing world-unit list.
LPS and RAS describe the same identified frame in different coordinate systems:
`is_equivalent_frame` is true, `==` is false, and `coordinate_system_change` derives the signed
permutation between them from axis orientation. Neither an identifier nor an axis name
prescribes storage order.

**Unit representation:** open unit strings following the CF/UDUNITS convention used by xarray
`attrs["units"]` and OME-NGFF, compared exactly and case-sensitively. Empty or padded strings
are refused. Adapters normalize spellings (`mm` and `millimeter` compare as different) and
perform explicit conversions before construction; the core never parses, relabels or silently
converts units and does not require Pint. An optional Pint-backed conversion helper may be added;
unit objects are never stored. Cartesian axes refuse known angular spellings, a best-effort
check; recognizing an angle does not supply an angular representation. Affine inputs may be
angular, because inputs are coordinate values.

**Compatibility is a sequence of checks, not overloaded approximate equality:**

1. Validate both declarations and all coordinate dependencies. A matching identifier cannot
   override a contradictory definition or defining context (`conflicts_with`).
2. Establish a shared reference system from a shared local/global identifier or an explicitly
   validated external definition. Independently constructed unknown spaces are unrelated.
   Creating a local space once and sharing it is sufficient; persistence retains its identifier.
3. Require identical coordinate interpretation, including defining context and coordinate
   system, for direct arithmetic. The same frame in another coordinate system is converted
   explicitly, by a derived or supplied transform, never during arithmetic.
4. Check sample placement on the aligned output domain. A shared frame does not imply a shared
   grid. Mapping and input-coordinate agreement must hold wherever samples are combined.

V1 accepts matching validated declarations and mappings conservatively, using canonical
structural identity; it does not attempt numerical proof that two arbitrary functions are equal.
For v1 affines this means structural identity of the space/context/representation, input names
and units, matrix and translation, plus agreement of protected input coordinates and topology
on the combined domain. Two distinct affine formulas that happen to agree on an overlap are
not accepted without explicit normalization. Future sampled-field union rules do not relax
this affine rule. Approximate geometric comparison is a separate explicit query with caller-specified, unit-aware
tolerances. It is not index equality and never creates a transitive equivalence relation.

An authority adapter can construct a shared descriptor from a complete validated system
specification. A name/code may be sufficient for some definitions but not for every frame family.
Defining context is normalized by that adapter; the core never guesses omitted values or equates
unknown with known. Provider/library versions belong in provenance unless changing them changes
the actual definition or transformation. Record resolved numerical models/resources when needed
for reproducibility; do not resolve a remote catalog during arithmetic.

For NGFF, a coordinate-system name is scoped to its container and group. Independent stores with
the same name are not thereby the same frame. Import shares descriptors within one explicitly
scoped import context; cross-open or cross-store identity requires preserved stable identity or
an explicit caller-provided relation. A storage path is a locator, not a permanent global frame
identifier. Export must report if its target cannot retain identity or defining context.

Registration or a known domain transformation supplies an explicit relation. Applying that
relation to coordinates differs from resampling values onto a target grid. Neither occurs
implicitly during `+`. No core frame graph, global name registry or universal CRS parser is needed.

### 3. Mixed operations follow xarray's value semantics

**The contract, enforced by the binding index:** a supported operation gathers bindings from
**all** inputs, including conditions and both branches of `where`, regardless of operand order.
Section 5 records how it is enforced and the lane on which each row is measured:

| Declared bindings | Result rule |
|---|---|
| None | Ordinary xarray/NumPy behavior; do not invent geometry |
| One | Adopt that binding on the result if output coordinates support it |
| Several, compatible | Reconcile their mappings and retain valid output geometry |
| Several, incompatible or unresolved | Reject; require explicit normalization, alignment or transformation |
| Any present but invalid binding | Reject; never treat malformed framed data as plain values |

An unframed array makes no independent world-space or identity assertion. Its index labels are
interpreted through the framed operand's matching xarray indexes. **Alignment keys and mapping
inputs are separate roles:** F may read auxiliary coordinates; attaching F never turns those
coordinates into alignment keys. Combining arrays uses ordinary index-label alignment (or
xarray's size/positional rules for unindexed dimensions), then evaluates F on retained input
coordinates. Labels do not acquire physical meaning merely because a mapping depends on them.
This is intentional and cannot detect a bare mask that actually came from another patient.
A caller who needs that check must preserve or attach the mask's own binding. NumPy arrays
broadcast positionally; DataArrays align by named dimensions and labels. No silent coercion
between those two contracts, and no implicit frame construction on the input operand.

For example, `slice=[0,1,2]` with auxiliary `slice_offset(slice)=[0,2,5]` mm aligns a bare
`slice=[0,2]` operand to slice labels 0 and 2, whose offsets are 0 and 5 mm. That is intentional
ordinal-label alignment, not a request for offsets 0 and 2 mm. Physical-offset alignment requires
both arrays to expose those offsets as the matching dimension index (for example `slice=[0,2,5]`
mm on the image). V1 allows both representations, and tests them separately. No value-based
inference guesses which coordinate system a bare operand intended.

Non-index coordinates referenced by the binding are also protected inputs. If an unframed
operand carries a conflicting scalar plane coordinate, reject that conflict rather than let
xarray drop the fixed position. Likewise reject an explicit incompatible input-unit declaration;
missing units on an unframed operand mean interpretation in the bound input units, not automatic
conversion. The explicit unit channel is each relevant coordinate's `attrs["units"]`, not the
signal variable's units. At attachment, mapping-input attributes must agree with the frame's
input units; on combination, provided protected-input units and matching index-coordinate units
must agree with their framed counterparts when a unit is known. Unknown tokens are `ValueError`.
A missing key inherits only an established matching coordinate's unit, never the unit of an
auxiliary coordinate merely because it depends on that dimension. If an unframed operand supplies
a unit but the matching non-mapping index has no established unit, reject rather than infer one.
Clearing an incidental coordinate is an explicit caller choice. This deliberately
strengthens xarray's ordinary non-index-coordinate conflict behavior for bound coordinates only.

Consequences:

- `image > threshold`, `image * weights`, `image + ndarray`, and their reflected forms can
  preserve geometry without attaching a frame to constants or weights.
- A factor along `echo` may add or broadcast a nonspatial dimension; spatial locations stay
  the same even though total array size changes. Newly introduced dimensions do not become
  spatial just because of their name or length.
- A labelled unframed operand may select a coordinate intersection under xarray's normal
  arithmetic join. The result has the retained samples' locations. This is allowed, not a
  spatial error. Users requiring identical extents can request xarray's exact alignment.
- With several framed operands, matching label strings alone is insufficient. On samples
  which are combined, the mappings must agree under section 2's representation-specific
  compatibility rule; v1 affines require structural identity. For an outer join, the retained declaration
  must also locate samples contributed by either input. This is a capability requirement:
  do not synthesize missing curvilinear locations or extrapolate a bounded mapping. A sampled
  union is possible only when every required output location is supplied and conflicts can be
  checked; Cartesian-product holes make an otherwise plausible union invalid.
- Disjoint framed spatial dimensions are not permission to produce the Cartesian product of
  unrelated images. Spatial roles and dependencies must be compatible; adding nonspatial axes
  differs from implicitly broadcasting a framed plane through a volume. The latter needs an
  explicit operation which declares that interpretation.

This supersedes the previous proposal to reject every framed/unframed combination. It also
supersedes treating xarray's normal coordinate intersection as inherently unsafe.

### 4. Propagation depends on coordinate meaning, not array size

The decisive invariant is: **the output binding correctly locates the output samples under
that operation's documented semantics.** Preserve the immutable mapping when possible and let
xarray carry the changing coordinates. Derive shape from the result; do not mutate frame.shape.

The table gives each operation family's geometry rule and its measured state on the pinned
development lane (the local patch series; the inventory itself notes that this lane is not a
release pass, which additionally needs hosted CI and released xarray fixes). Stock-lane
differences are the holes listed in
[the operation inventory](dev/binding/binding_operation_inventory.md) and tracked by the upstream pull
requests. **Native** means the binding is carried, or the operation refuses (a `ValueError` from
the binding, or xarray's own `AlignmentError` where alignment itself fails); **Unframes** means the
result has no geometry in the source frame and the binding is dropped whole; **Accessor** means
the checked path is an explicit `.rf` method. Rows marked *(not probed)* state the contract; the
inventory has no case for them yet.

| Operation | Geometry rule | State on the patched lane |
|---|---|---|
| Pointwise arithmetic, comparison, ufuncs, `where` | Reconcile all bindings and preserve placement | Native; incompatible bindings raise, both operand orders |
| Broadcasting and nonspatial reduction | Preserve the mapping on the result domain | Native; a framed plane against a framed volume raises |
| Crop, reverse, stride, gather, transpose | Keep F and the current selected coordinates | Native |
| Scalar selection / squeeze | Retain fixed coordinate terms | Native; `drop=True` of an index-owned scalar raises rather than corrupting the index (stock keeps it, pending pydata/xarray#11617) |
| Rename, `swap_dims` | Rewrite binding references consistently | Native (`BindingIndex.rename`, `swap_dims` on the patched lane) |
| Alignment / reindex | Locate output samples with F, including missing values | Native for ordinary joins with plain operands; `join="override"` and an incompatible framed grid raise `ValueError` from the binding; a plain index that conflicts with a bound coordinate raises xarray's `AlignmentError` |
| Interpolation in a coordinate domain | Evaluate F only where continuous evaluation is defined | Native `interp` along a geometry dimension splits the binding, which then raises on inspection (fail-closed, not preserved; refusing at the operation needs a hook); `resample` is the explicit path with a declared domain |
| Resampling to a new mapping or frame | Attach an explicitly constructed target binding | Accessor: `rf.resample_to` |
| Concatenation | Validate a shared F on the union domain | Refuses along a geometry dimension (`rf.unframe()` first); nongeometry concatenation is native |
| Spatial reduction, contraction, FFT | Explicitly unframe or derive new geometry | Unframes: reducing over a geometry dimension drops the whole binding, for DataArray and Dataset; contraction and FFT *(not probed)* |
| Coarsening / binning | Derive representative locations explicitly | Refuses geometry coarsen; nongeometry coarsen and `construct` are native |
| Stack, pad | Explicitly redefine sample locations | Refuses on geometry dimensions; native on nongeometry dimensions |
| Arbitrary `apply_ufunc` / `map_blocks` | Caller declares the output domain relationship | Caller's responsibility; output size is not evidence *(not probed)* |
| Coordinate replacement | Explicitly redefine sample locations | Partial replacement of a bound coordinate raises the corrupt-index error; re-framing is explicit |
| Raw reshape / array conversion | Require explicit reattachment | Unframes: a new array carries nothing *(not probed)* |
| Dataset / coordinate extraction | Shared grid: variables spanning all geometry dimensions carry the binding; coordinates and partial-dimension variables are unframed | Native on the patched lane |

For a crop starting at index 10 with step 2, a positional-index mapping would need composition
with `old_index = 10 + 2 * new_index`. Retaining the original sample coordinates avoids that
extra state. Merely changing shape would lose both the offset and stride. Conversely, a
resampled array may have the original shape but a different orientation and origin.

Value rearrangement is not always geometry rearrangement: shifting or rolling values while
keeping xarray coordinates fixed produces new values on the same grid. Rolling the coordinates
too changes their sample ordering. Neither case warrants guessing a physical registration.

Coordinate replacement redefines sample locations by construction. Validation can check
units, shapes, finite values and mapping requirements, but cannot distinguish intentional
recalibration from a caller's mistake. Equal lengths add no protection.

Spatial averaging is not plane selection. Removing a mapped dimension via `mean` cannot
silently become `isel` at the center. A projection API may deliberately supply a new 2-D
geometry, but that is an explicit interpretation. A binding with missing required coordinate
terms is invalid, even if a scalar declaration survived the operation.

### 5. Integration with xarray: lifecycle ownership is the target

The recommended production model is **lifecycle ownership**: the extension manages a binding
and its coordinate dependencies together across supported operations. It must validate inputs
before conflicts can be discarded, then construct a valid result or explicitly remove geometry.
This is implemented by `BindingIndex` (`src/xarrayrf/_binding.py`), an xarray `Index` owning every
source coordinate of the transform, whose `isel`, `sel`, `rename`, `roll`, `equals`, `join`,
`reindex_like` and `concat` methods carry or refuse each native operation, and whose optional
hooks (`join_overlapping`, `check_unindexed_coord_conflicts`, `check_override`, `check_stack`,
`check_pad`, `check_coarsen`) are called by the local xarray patches on the paths stock xarray
decides without asking the index.

There are two runtime states: no binding and a valid binding (route (ii), decided 2026-09-25;
see [the binding design](dev/binding/binding_design.md)). The binding lives only in a
private index that jointly owns its input coordinates, with no separate declaration coordinate,
so removing the index removes the binding completely and no guardless declaration exists.
Deliberate removal of the whole index (`rf.unframe()`, `drop_indexes`) and conversion to raw
values are the documented exits that unframe; payload replacement (`.data =`, `copy(data=)`)
keeps the binding, since the coordinates it describes are unchanged. An encoded declaration
(persistence) is inert metadata until `decode`, which raises on a malformed payload rather than
returning an unframed array.
A binding that lost a coordinate to another index (a Dataset reduction on stock xarray, or
`expand_dims` of a retained scalar) is neither state: every accessor member except
`rf.unframe()` raises and names it, and `rf.unframe()` clears it. The superseded route (i), a declaration coordinate plus a guard
index with late validation, is recorded in
[the binding design](dev/binding/binding_design.md); it could produce valid-looking
results after losing conflicting evidence, which is why it was rejected.

Prefer public xarray extension points and composition. Keep native DataArrays, ordinary
label alignment and broadcasting, and lazy pixels. Do not subclass DataArray, monkey-patch
operators, change global alignment options or build a full replacement array container.
An accessor supplies construction, inspection, world evaluation, checked explicit operations,
unframing and persistence. Explicit `unframe()` removes the binding and its guard together
while retaining ordinary data coordinates; `reset_coords(drop=True)` is not equivalent.

External domain metadata can coexist only under an explicit ownership rule. On import, an
adapter consumes the authoritative source once and attaches a validated snapshot. It must then
remove, invalidate, or regenerate redundant transform metadata when geometry changes; a stale
GeoTransform, WCS header or NGFF declaration cannot remain a second writable geometry authority.
Calling a domain accessor that changes geometry requires an explicit adapter round trip unless
its native path is covered by the integration contract. Unsupported paths remain subject to the
same release invariant below; documentation alone cannot certify them.

Persistence is separate from the runtime carrier: `rf.encode()` writes the declaration into the
reserved `attrs["xarrayrf_binding"]` key of an unframed copy and `rf.decode()` restores and
revalidates it; a live binding and that key never coexist (`rf.frame` refuses one, `rf.unframe`
drops it). Joint ownership of input coordinates is measured against ordinary `PandasIndex`
operands in the inventory, including mixed arithmetic and nearest selection; label alignment
keeps xarray's ordinary behaviour rather than exact-only transform matching. xarray's
`RangeIndex` compares with a tolerance by default (see the
[prior-art study](dev/architecture/prior_art_coverage_study.md)); `rf.frame` rebuilds every source axis as a
`PandasIndex`, so a binding's alignment is exact-label.

The accessor derives geometry from current coordinates, including fixed terms, and does not
cache validity on a mutable DataArray. Coordinate buffers are owned or truly immutable when
shared in snapshots. Attachment never copies or computes pixel arrays. Encoding is canonical
(sorted fields, normalized scalar representation and units); equality compares validated
structure rather than arbitrary JSON formatting. Numerical tolerance stays out of index equality.

**Measured record.** [The operation inventory](dev/binding/binding_operation_inventory.md) runs 93
cases on the stock, upstream-main and patched lanes and classifies each outcome; the patched lane
has no hole. The failures that motivated the local patches (conflict evidence lost in
`_binary_op`/`_merge_raw`, override bypasses, index splits on reduction, geometry coarsen, stack
and pad) are listed there with their fault paths and reproducers, and the fixes are recorded in
[the patch manifest](dev/xarray-upstream/xarray_patches.md). Chunked arrays are covered: framed arithmetic,
alignment and selection on Dask-backed data execute no pixel task.

A Dataset binding belongs to its geometry dimensions (shared grid, user decision 2026-09-25;
[binding design](dev/binding/binding_design.md)). Every data variable whose dimensions include all
geometry dimensions carries it; extra dimensions are nonspatial. Variables lacking a geometry
dimension, scalars and coordinates (`ds.x`) are unframed. Distinct grids use distinct dimension
names, each with its own binding; a spatial variable on the same dimensions with a conflicting
frame is refused. xarray's shared labels already assert that same-labelled samples coincide, and
`framed + plain` is framed, so a same-grid sibling has the owner's geometry. Metadata that is not
frame-compatible lives beside the framed array in an application object or in nonspatial
variables.

**Minimum native subset required for transparency:** pointwise arithmetic and comparisons
(including NumPy ufuncs and both mixed operand orders), `where` with all inputs checked,
`isel` crop/stride/gather and scalar selection with retained fixed coordinates, transpose,
and inner-join alignment with ordinary labelled unframed operands. Framed-plane conflicts,
framed plane/framed volume broadcasting, incompatible declarations and explicit input-unit
conflicts are negative cases of that same subset, not optional features. Each case's supporting
lane (stock release, upstream main, or patched xarray) is recorded in
[the binding design](dev/binding/binding_design.md); a strict xfail on a lane is a
milestone, not a release pass.
Default scalar selection must work natively; `drop=True` must reject with `ValueError` when preserving location is impossible. Moving any member of this minimum to an
accessor-only API is a product change requiring the maintainer's decision, not a way to pass the gate.

**Required outcomes for minimum-subset negative cases:**

| Inputs / operation | Required outcome |
|---|---|
| Two valid framed planes with different fixed positions | Raise `ValueError` before coordinate conflict evidence is lost. |
| A framed plane and a framed volume, even in the same reference space | Raise `ValueError`; broadcasting values does not supply missing volume placement. |
| A framed plane and an unframed array introducing a new dimension | Valid broadcast with binding on the original spatial subset, provided no protected coordinate or unit conflicts exist. The added dimension is nonspatial regardless of its name. |
| Unframed operand explicitly contradicts a protected coordinate or input unit | Raise `ValueError`; no silent dropping or implicit conversion. |
| Malformed encoded declaration | `decode` raises (`MalformedDataError` or another encoding error); never returns an unframed array. An encoded array that has not been decoded is an unframed operand: its attribute is inert metadata. |
| An operand whose binding was deliberately removed | Combines as an unframed operand (route (ii)); the rows above still apply to its coordinates. |
| Valid but incompatible framed identities, representations, context or mappings | Raise `ValueError`. |
| Framed alignment using `join='override'` | Raise `ValueError` in v1, including otherwise compatible operands; do not replace checks with caller override. |

Errors include the affected coordinate/operand and a corrective action. The future tests assert
these types and diagnostic categories; no new exception hierarchy is required yet. Late invalidity
is permitted only for separately enumerated unsupported operations, never these mandatory cases.
For unsupported `drop=True` removal of a required fixed coordinate, v1 requires `ValueError`;
other unsupported paths must acquire an explicit expected outcome before entering the test suite.

**Release invariant:** native paths operating on decorated arrays must either enforce their
stated geometry semantics, reject, or unframe where the inventory allows it (below). An unsupported native path that produces a valid-looking incorrect
binding blocks release regardless of its documentation label. Loss of the entire binding counts as
fail-closed only for operations whose result has no geometry in the source frame, such as a
reduction over a geometry dimension, each listed in the
[operation inventory](dev/binding/binding_operation_inventory.md). For operations whose result keeps
geometry (selection, arithmetic, alignment, transpose, rename) it is a release-blocking hole. Deliberate exits such as `unframe()` and conversion to raw values are separately
specified; after that exit the values have no geometry guarantee.

**Gate, met for the measured cases on the pinned development lane (2026-09-25):** lifecycle
enforcement for the minimum subset and fail-closed behaviour for the other probed native paths,
including reduction, extraction, override and concatenation, are measured in the inventory with
no remaining hole. A release pass additionally needs hosted CI on that lane and released xarray
versions carrying the fixes; the patched lane has no hosted CI and the needed xarray fixes
are not yet released. The ladder that was followed
was public extension hooks first, then focused local xarray fixes as separately reviewable
commits (the `index-hooks-4` series, see `docs/dev/xarray-upstream/xarray_patches.md`); five
generic bug fixes were proposed upstream, of which the empty-roll and broadcast fixes are
merged. Stock xarray keeps the holes those fixes close; a strict expected failure on a lane is
a milestone, not a release pass. Reducing native coverage or introducing a controlled type would
change the product contract and remains the maintainer's decision, not a fallback.

#### Existing extension mechanisms and dependency decisions

| Mechanism | Evidence and decision |
|---|---|
| xarray CoordinateTransform / CoordinateTransformIndex | Reused for `Geometry.frame_coordinates()`: a subclass maps positions through the array's actual source coordinates and the affine, survives slicing, aligns exactly and answers point-wise nearest selection. It is an interoperability view, not the binding: the authoritative binding is `BindingIndex`, and nothing reads frame coordinates back. |
| xarray NDPointIndex | Candidate for explicit nearest lookup over sampled world coordinates, not a geometry carrier or compatibility rule. Installed 2026.7.0 inherits Index.join, which raises NotImplementedError; nearest lookup does not establish general alignment or a domain-appropriate metric. Evaluate only when a concrete lookup use case needs it. |
| xproj CRSIndex | Existing scalar CRS-index precedent, leaving ordinary coordinate indexes in place. Useful for carrier/alignment design and interoperability; documented CRS comparison requires matching reference-coordinate names. It describes geospatial CRS via Pyproj, not a patient-space identity plus general sample mapping. Do not add as a mandatory dependency to solve a different problem. |
| rasterix RasterIndex | Existing affine raster lifecycle work built on transform indexes, including slicing and inner/outer alignment. Its documented use is 2-D raster/GeoTIFF geometry. Source inspection at revision 7899c7a2 confirms rectilinear slice preservation, but coupled x/y and scalar/fancy selection drop the index. Its joins use raster bounds and numerical tolerances. Study/reuse suitable pieces; direct adoption for 2->3, generic ND and strict compatibility is not established. Its design documents the conflict when two indexes own the same CRS variable. |

These are reuse candidates and upstream collaborators, not maturity rankings based on project
size. The spike must compare real behavior before selecting a runtime implementation. In
particular, do not generalize the base transform index's limitations to every subclass: Rasterix
already implements behavior the base class omits. The xarray-independent coordinate mapping
contract stays the same whichever index strategy succeeds.

### 6. DICOM supplies bindings once; it does not drive the architecture

A DICOM producer constructs world geometry from orientation, pixel spacing and per-frame position,
declares the coordinate system and its orientation, and attaches any known frame identity. All DICOM
interpretation lives in the DICOM adapter; the core contains no DICOM concept. Thereafter signal,
rescaled signal, magnitude, phase, real, imaginary and complex arrays use exactly the same generic
rules as non-DICOM arrays.

Component algebra is pointwise when inputs share sampling: forming complex data from real
and imaginary arrays preserves the binding automatically. Acquisitions with different geometry
must be reconciled before component construction. Per-frame intensity calibration and padding
handling transform values, not patient locations. No geometry propagation callback should
reach back into the producer to rediscover metadata after each operation.

Acceptance cases cover a single 2-D image embedded in patient space, an oblique stack, selected
planes and points, nonuniform positions where representable, and nonspatial time/echo axes.
Do not inherit any application's frame-class constraints, such as a required determinant sign,
regularity or singleton shape. Not all DICOM objects describe a patient-space lattice: absent
geometry remains absent; tiled/slide coordinates or varying per-frame poses need an appropriate
mapping or partition into separately framed arrays. Never fit one affine map to incompatible
poses merely to offer a uniform API. Those richer producers can be added without changing
pointwise-operation semantics.

Frame of Reference identity is not Study/Series/SOP provenance. The generic result can retain
its world geometry without retaining exact source-image identity, pixel encoding or eligibility
for DICOM export. An export adapter separately validates source references, derivation, value
units, encoding and target information. Sample spacing, finite sample support and physical
slice thickness are distinct concepts; a retained plane does not imply a slab volume.
The bridge promises geometry-transparent array use, not lossless reconstruction of a producer's higher-level data objects.

### 7. Nonlinear mappings and bounded capabilities

`world = F(c)` remains the general contract, but it does not imply that F is total, invertible,
Euclidean or representable by a matrix. Capabilities follow from the representation and its
parameters; a mutable bag of capability flags must not contradict the mapping.

| Representation | Evaluation domain | Inverse / metric / interpolation |
|---|---|---|
| Affine on coordinate values | Finite real inputs; numerical overflow or nonfinite results fail | Inverse only when mathematically determined on the requested domain; a rectangular embedding is not a bijection onto the whole output space. Physical spacing/metric requires appropriate units and geometry. |
| Sampled coordinate fields | Stored nodes with declared spatial topology | No continuous model or inverse inferred. Selection is valid; off-node queries need a separately declared interpolant/model. |
| Field-backed transform | Field grid plus specified interpolation and validity rules | Preserve field interpolation separately from image resampling. Inversion is an additional capability, never inferred from matching rank. |
| External WCS / nonlinear provider | Provider-declared valid domain and required context | Delegate computation to the provider. Inverses may be absent, partial, numerical or ambiguous; preserve that distinction. |

First ship the affine representation and its operation contract. Sampled-field and provider
rows specify extension boundaries and acceptance requirements, not v1 implementation promises.
No mandatory Astropy, GWCS, PROJ, GDAL or nonlinear solver. Adapters can use existing engines,
including an internally composite model, without a core transform graph or discovery registry.

World coordinates need not be Cartesian. Future representations can declare angular components,
periodicity and singularities. Native xarray label alignment remains literal: longitude labels
0 and 360 do not automatically align, and no angular wrapping or branch selection happens in
arithmetic. Explicit domain operations may normalize representations. Coordinate components do
not by themselves define a Euclidean distance, area, spacing or physical voxel support.

Out-of-domain evaluation must fail explicitly or return an explicitly requested per-point validity
result; never extrapolate silently. A world-to-coordinate operation requires a declared inverse
policy. A point off an embedded plane is not silently projected onto it. Nearest-sample lookup
is a different operation and requires a metric and tie policy. None is inferred from `inv(A)`.

**Pixel-based engine boundary:** input coordinates can explicitly be the original pixel
coordinates expected by a WCS. Crop/stride retain those values; the immutable engine is not
silently rebased. For a current array position i, a derived positional adapter evaluates
`F(C(i))`, where C reads the array's current coordinates. Its shape is derived and nonauthoritative.
External mutable engines must be snapshotted through a supported immutable representation or
reconstructed from a validated payload; retaining an externally mutable object as authority is
not acceptable. No callbacks into a DICOM producer are required after attachment.

Coordinate fields may depend on multiple spatial dimensions. Nonspatial context such as time or
channel must have an explicit role before it is allowed to parameterize geometry. V1 refuses
such context-dependent mappings rather than treating all coordinate dependencies as spatial.
Channel identity in an NGFF transform is compatible with ordinary channel broadcasting; a
transform that actually couples channel and spatial positions requires a supported contextual
model, not silently stripping its channel rows.

### 8. Persistence and OME interoperability

Separate the semantic contract from a storage carrier. `xarrayrf.ngff` maps OME-Zarr 0.4, 0.5 and
0.6 coordinate-system and transform metadata to validated runtime bindings on `ome-zarr-models`,
exports 0.6 multiscale levels and frame-to-frame affines with a loss report, and reads and writes
RFC-4 axis orientation through the anatomical vocabulary; the persistence gate test round-trips
the NGFF-representable cases against schema 1.
Keep format parsing and numerical engines in optional adapters, with producer dependencies
pointing toward the core.

The [cross-domain cases](dev/architecture/cross_domain_cases.md) record the cases
and sources; `ome-zarr-models` 1.8 was selected after comparing ngff-zarr, xarray-ome and
ome-zarr-py by supported behaviour and dependency cost ([the adapters
design](dev/adapters/adapters_design.md)). NGFF's image-container restrictions must not become the
generic core's dimensionality restrictions.

For each multiscale level preserve its own array-to-intrinsic mapping and the selected onward
mapping. Do not derive geometry from shape ratios. At export, compose current coordinate values
with F: crop/stride offsets must survive even though a new stored array starts at index zero.
If this composition cannot be represented in the requested format/version, fail with a concrete
loss report. No implicit resampling, origin reset, dimension padding, or loss of reference identity.

If an existing backend already exposes physical xarray coordinates, bind those coordinates
without applying the source scale/translation again. Choose the reference system explicitly
when several output systems exist. Dataset bindings belong to grids (geometry dimensions); a multiscale hierarchy
may use DataTree without imposing one mutable frame on every level.

Generic persistence must retain the declaration, its required coordinate arrays, defining
context, identity and the grid's geometry dimensions. A supported decoder validates these and restores
runtime enforcement. Unknown schema/kind, malformed declaration, missing referenced buffers,
and unavailable optional provider are distinct failures. Passive metadata may be inspected or
copied without a provider, but cannot be silently treated as an active frame or plain unframed
values. Geometry-dependent evaluation/combination requires activation; any structurally safe
operations allowed on inactive payloads must be explicitly specified and tested.

A domain definition or provider payload must be versioned and data-only. No pickled live frame,
arbitrary import path, executable callback or automatic network lookup on decode. Unknown
payloads may be preserved opaquely without activation. Plain xarray IO does not promise restored
runtime guards. Backend behavior must be proven by real round trips, not inferred from surviving
attrs. Export to DICOM additionally requires domain provenance/encoding checks outside this core.

The public surface is normative in [the core interface](core_interface.md): immutable frame and
transform declarations, the accessor-derived `Geometry` view and the `.rf` binding. No constructor
requires an array shape. Runtime indexes may maintain derived positional state but not a
competing domain.

### 9. Sampling values, declared cells, anatomy and frame completeness

The [grid plan](dev/architecture/grid_plan.md) records the decisions; this section states the
architecture they produce.

- **`Grid` is the pixel-free sampling value.** It holds what a binding holds (the coordinate
  transform, the 0-D and 1-D source coordinate values, declared intervals) and nothing else: no
  pixels, no non-geometry dimensions. `Geometry` stays the live view of an array and shares one
  sampling implementation with `Grid`, so the two cannot disagree. Arrays and grids convert both
  ways through one internal path (`rf.grid`, `rf.frame(grid)`, `native.frame_array`,
  `native.grid_coordinates`); `Grid.isel`/`sel` run xarray's own selection on a coordinate-only
  Dataset, so there is one slicing implementation. A grid is a resampling target like a framed
  array.
- **Cells may be declared.** A declared interval per sample, in its own coordinate values,
  describes thickness, gaps and overlap. The binding index carries intervals through selection,
  roll and rename and requires them to agree on alignment; `domain="cells"` uses them. Spacing
  never stands in for thickness: DICOM import declares intervals from `SliceThickness`.
- **Anatomy is grid algebra.** `xarrayrf.anatomy` names orientations (`orientation_codes`),
  reorders without moving samples (`reoriented`) and builds axis-aligned targets
  (`cardinal_grid`); reformatting is `rf.resample_to(cardinal_grid(...))`, not a separate
  resampler. Labels are nearest-axis, as nibabel's `aff2axcodes`.
- **Identity is explicit, and absence of identity is visible.** A frame is complete when it
  has an identity (declared, or a local frame created and shared on purpose) and anonymous when
  its source names no space. Nothing guesses an identity: no hashing, path fingerprints or shared
  default world. `rf.assume_frame` and every reader's `frame=` share one adoption contract,
  including derivable coordinate-system changes, and never bypass grid checks. Refusals name the remedy
  that applies.

### Local xarray fixes are an intended development route

Prefer a clean change in xarray itself whenever that is the appropriate abstraction boundary.
Maintain local fixes rather than inventing fragile downstream workarounds or reducing the
required semantics. Keep changes in a separate xarray checkout, split into small independently
reviewable commits with upstream-only regression tests where possible. Exact source revisions,
patch ordering and stock/patched validation belong in the [patch record](dev/xarray-upstream/xarray_patches.md).

Upstream acceptance is not a prerequisite for local development. Prepare issue/PR-sized pieces
as work proceeds and prepare submissions when reference-frame behavior stabilizes. Search
existing reports before building; seek maintainer input before submitting substantial new APIs.
No external posting is implied. See [the maintained-fix workflow](upstream.md).

### Decision Classification and ranked risks

| Priority / decision | Forcing trigger | Response / cost to change |
|---|---|---|
| 1. Native lifecycle enforcement | Minimum operation gate fails or patch needs the prohibited coupling above | Stop and present measured patch/fallback options; high behavioral cost after adoption. |
| 2. Identity, context and unit contract | Cross-domain fixtures disagree about equality or normalization | Revise descriptor before schema freeze; conservative refusal is available, implicit equivalence is not. |
| 3. Persistence and provider snapshots | Codec cannot round-trip identity, context, units or mapping | Refuse that export/provider; do not freeze schema until boundary fixtures pass. |
| 4. Dataset association | A Dataset path splits or replaces a grid's binding | Shared-grid rule; fix the xarray path (patches 6 and 7) or refuse. |
| 5. Runtime carrier, names and optional codec dependency | Reuse study or spike demonstrates a simpler mechanism | Change before publication; low migration cost while no public API exists. |

The chosen unit policy is open CF/UDUNITS strings compared exactly, with adapter normalization.
It replaced a closed token vocabulary, which could not follow CF or NGFF spellings and never
prevented mislabelling. The rejected alternative is a mandatory general unit engine (unneeded
core dependency). The current value objects plus a future accessor are a provisional API, not a class-count
constraint. The canonical alternatives and decision record live here; examples do not duplicate it.

## Alternatives Considered

| Approach | Simplicity | Extensibility | Testability | Migration Cost |
|---|---|---|---|---|
| Store today's ReferenceFrame in attrs | Initially small, stale-state risk | Tied to old classes | Easy to falsify | Low initially |
| Copy a frame and update its shape | Small but geometrically insufficient | Wrong abstraction for selection | Fails crop/resampling cases | Corrective work later |
| Reject every mixed unframed operand | Extra caller ceremony | Restricts ordinary array algebra | Simple but unnecessarily strict | High for existing algorithms |
| Controlled wrapper or DataArray subclass | Own dispatch, large API burden | Constrained interoperability | Local rules testable | High |
| Mandatory WCS/domain ecosystem | More dependencies | Broad geometry, still needs xarray rules | Integration-heavy | Medium/high |
| xarray transform index / Rasterix lifecycle reuse | Existing machinery; behavior must fit | Strong starting point, ND fit unproven | Compare existing public APIs | Depends on supported subset |
| xproj scalar CRS-index pattern alone | Small carrier | Space semantics, incomplete lifecycle | Counterexamples already observed | Low but insufficient alone |
| Independent coordinate binding plus bounded xarray extension | Small conceptual model; propagation needs proof | 2-D, ND and future mapping representations | Synthetic operation tests | Moderate |

Choose the last option. Its value is adopting xarray's semantics rather than replacing them.
The difficult part is operation propagation, so that is the first evidence gate.

## Deferred Work

### Scope Boundaries

No universal interception of arbitrary Python/NumPy code, implicit registration, automatic
DICOM export, general transform graph, mandatory nonlinear solver, per-voxel provenance, or
full xarray API wrapper. Repository setup and local implementation are authorized. Upstream submissions and remote
publication still require explicit authorization. Existing adapters can be revised after the
independent contract is proven.

### Risks and Unknowns

- Required native operations may exceed public xarray extension hooks. The probe must expose
  this before downstream adoption; documentation cannot turn a missing guard into enforcement.
- Serialization and Dataset paths may lose a grid's binding.
- Equivalent mappings may be conservatively refused until explicitly normalized.
- Coordinate evaluation/comparison must remain compact or lazy for large curvilinear data.
- Explicit coordinate reassignment can change physical meaning without changing shape.
- Arbitrary user functions cannot be classified from their output size; declarations of domain
  preservation are caller assertions, not facts inferred from numerical values.

## Next Steps

1. Land the five upstream bug-fix pull requests and retire the matching local patches as
   releases ship them ([manifest](dev/xarray-upstream/xarray_patches.md)); consolidate the remaining hook patches
   to two methods before proposing them ([index hook design](dev/xarray-upstream/index_hook_design.md)).
2. A bounded nonlinear provider through the capability protocols (section 7); no adapter yet
   exercises one.
3. Validation with downstream consumers that bind their own arrays through `rf.frame`, or hold
   array-free geometry as `Grid` values, such as an application-level DICOM series assembler.
4. Core support needed by interactive viewers, in priority order: selection overhead, backend
   neutrality, source footprints for `Grid` targets, checked on-plane inversion
   ([viewer boundary plan](dev/architecture/viewer_boundary_plan.md)).

## Sources and Evidence

Primary documentation consulted for the extension boundaries:

- [Xarray computation and alignment](https://docs.xarray.dev/en/stable/user-guide/computation.html).
- [Xarray accessors and composition](https://docs.xarray.dev/en/stable/internals/extending-xarray.html).
- [Xarray custom indexes](https://docs.xarray.dev/en/stable/internals/how-to-create-custom-index.html).
- [Xarray transform-index examples](https://xarray-indexes.readthedocs.io/blocks/transform.html).
- [Xproj CRS handling and alignment](https://xproj.readthedocs.io/en/latest/usage.html).
- [Rasterix alignment](https://rasterix.readthedocs.io/en/latest/raster_index/aligning.html) and
  [index ownership design](https://rasterix.readthedocs.io/en/latest/raster_index/design_choices.html).
- [Astropy shared WCS interface](https://docs.astropy.org/en/stable/wcs/wcsapi.html).

Local evidence: xarray 2026.7.0 stock, upstream `main` and the pinned patch series, measured by
`tools/binding_operation_probe.py` and recorded in the operation inventory.
