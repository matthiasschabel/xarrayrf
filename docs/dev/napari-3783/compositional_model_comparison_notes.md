# The napari compositional model compared with xarrayrf

**Status:** Active
**Last updated:** 2026-10-11
**Scope:** xarrayrf core (transforms, axis vocabulary, value geometry) and its napari-facing
design; the napari fork's experimental compositional data model
(`src/napari/experimental/_data_model/` on branch `explore/compositional-data-model`).

## Context

The napari fork carries an experimental description of what a dataset is to a viewer: data
objects, domains, fields, axis roles and an embedding into world space. About 2.5k lines of it
(`_axis`, `_mapping`, `_graph`, `_basis_conversion`, `_level_geometry`, `_display_sampling` and
the xarray/NGFF adapters) handle coordinates, which is xarrayrf's job. On 2026-10-11 both code
bases were compared by reading the code, to decide whether xarrayrf should be folded into the
model, and whether anything in the model's coordinate layer is better than xarrayrf's.

Conclusion: xarrayrf should replace the model's coordinate layer, not be folded into it, and
four ideas from the model are worth carrying over (P1 to P4 below). The napari fork now records
its coordinate layer as frozen pending that replacement.

xarrayrf is domain-neutral. Every proposal below is justified by need across domains
(geospatial, microscopy, astronomy, physics, medical imaging), and domain vocabularies stay in
adapters or as open strings, never as core enums.

## Current Decision

Decided by the maintainer on 2026-10-11. None of P1 to P3 is a release gate.

**P1. Transform edge semantics: adopt as a core value object.** A wrapper around any
`SupportsPoints` transform, alongside `CompositeTransform`, carrying:

- `role`: `correction` (same object or calibration; quantitative comparison valid),
  `correspondence` (different but alike objects: homology, atlas, survey-to-survey; comparison
  needs caveats) or `composition` (placement for display only);
- `accuracy`: a non-negative scalar in target units, or `None` when unknown;
- `validity_region`: target-frame bounds where the transform is defined, `None` for everywhere;
- `provenance`: free text;
- the reason for the correspondence as an **open string**, with adapters reserving values
  (`"cross_species_homology"` in the medical adapters, `"datum_transformation"` in a geospatial
  one). The model's closed `CorrespondenceBasis` enum is half biology and is not copied.

Composition yields the weakest role and the root-sum-square accuracy (`None` if any member
lacks one). `resample` and `rf.resample_to` accept the wrapper and take an explicit policy
(refuse, warn or allow) for chains containing a `composition` member; plain transforms are
unaffected. PROJ's per-operation accuracy and area of use is the geospatial precedent; the
DICOM Spatial Registration importer is the first natural producer. Persistence needs a new
record kind: the schema-1 freeze (release criterion 3) must record whether additive kinds are
allowed after the freeze, so P1 can land on either side of it.

**P2. Value geometry and basis conversion: belongs in xarrayrf.** Values with direction
(wind and current vectors, electromagnetic fields, stress and diffusion tensors, flow
velocities) must be re-expressed when the frame they are written in changes; xarrayrf today
transforms positions only, so resampling such a field into a rotated frame returns
wrong-direction values without error. The model's `FieldGeometry` (value kind, named
components, basis frame, orthonormal-basis capability, packing for symmetric tensors) is the
metadata half; its `convert_field_basis` (lazy, orthogonality-checked, unit-checked) is the
operation. Both depend on `ReferenceFrame` as the basis frame and are not domain-specific. They are
the narrow passive, orthonormal special case of the general law in
[the value geometry design](../architecture/value_geometry_design.md), which also covers
registrations (general `J`), covariant slots, pseudovectors and densities.

**P3. Axis roles: reserved `axis_types` values.** Document `"coordinate"` (a shared physical
axis: one slider drives every dataset's `time`), `"local"` (an object's own axis: channel 0 of
two images is unrelated) and `"component"` (the index into a vector or tensor value; ties to
P2) as reserved values in [the core interface](../../core_interface.md). Other strings stay
free; no enum.

**P4. Typed non-results: no decision needed now.** If a container of many transforms between
many frames is ever added, "no path" returns `Unplaced(source, target, reason)` and the set of
connected worlds is derived from current edges. The model's implicit path resolution (prefer
the direct edge, then fewest hops, then rank by role and accuracy, then frame-name tie-breaks)
is not copied.

**The viewer-facing model** (DataObject, Domain, Field, change notification) is not an
xarrayrf concern. The recommendation is a separate, domain-neutral napari integration package
over stock napari and xarrayrf; no consumer exists yet, so there is no urgency.

### Spike: the model's view bridge driven by a framed Dataset

Run 2026-10-11 against xarrayrf `71e23d9` and the napari fork at `471b412f9`, in the
`scipy-dev` environment (Python 3.14, stock xarray 2026.7.0). The model's six-axis synthetic
MRI (`_mri.py`: time, echo time, flip angle, z, y, x; complex signal with derived magnitude and
phase) was rebuilt as an `xr.Dataset` framed by one `Grid` (anisotropic affine into a local
scanner frame), with magnitude and phase as lazy Dask views. Its DataObject was built by the
model's own `data_object_from_xarray` with **no embedding**, and the bridge's
`_layer_coordinates` was replaced by one reading `Geometry.lattice()` from the binding. The
rest of `_view_bridge.add_to_viewer` and `_image_binding.ImageBinding` ran unchanged in a
headless `ViewerModel`.

| Check | Result |
|---|---|
| World placement of 20 random samples vs the model's own embedding | identical (max difference 0) |
| Axis labels, units, interpolation, sliced pixels | identical |
| Oblique (30° tilt) embedding through the model | refused (`NotImplementedError`, axis-aligned only) |
| Oblique through xarrayrf, passed to napari as a layer `affine` | equals `Lattice.transform_point` and the model's own matrix arithmetic (max difference 0) |
| `ImageBinding` construct, refresh, and a placement change through `DataObject.update` | works |

Replacing the coordinate layer is therefore mechanical: one function (`_layer_coordinates`) is
the whole seam, and the oblique case the model cannot display works for free. napari itself
warns that non-orthogonal slicing ignores the out-of-slice rotation, a napari limitation
independent of either model.

Concepts the bridge reads that have no xarrayrf home:

- **Change notification.** xarray objects are values; the model's identity-bearing
  `DataObject.update` / `subscribe` belongs to the viewer-integration package, not xarrayrf.
- **Per-variable interpolation policy** (nearest for categorical, linear for continuous).
  xarrayrf's `resample` takes `method=` per call. A declared default per variable is a small
  candidate if a second consumer needs it.
- **Axis roles**: covered by P3.
- **Component axes and value geometry**: covered by P2.
- **In-memory multiscale container.** xarrayrf's NGFF reader returns one framed array per level
  with its own transform; grouping levels is the viewer package's job.

### Where xarrayrf is already ahead (do not regress)

| Concern | Model | xarrayrf |
|---|---|---|
| Inverse conditioning | raw `np.linalg.cond` vs 1e12; unit-dependent | row/column equilibration (`equilibrated_inverse`) |
| Non-invertible | `NotImplementedError` | `ValueError` explaining a plane has no preimage |
| Frame identity | object identity, `FrameRegistry` needed to round-trip | `(namespace, id)` value identity; local, anonymous, declared; persistable |
| Frame vs coordinate system | one `CoordinateFrame`; cannot express one frame in LPS and RAS | `ReferenceFrame` + `CoordinateSystem` + `DirectionVocabulary` |
| Oblique geometry | view path axis-aligned only | full affines; `affine_class` |
| Sample placement | none | `ArrayCoordinates.sample_offset`, `Grid.intervals` |
| Coordinate values | `CoordinateAxis` reimplements xarray coordinates | native xarray |
| Multiscale | diagonal `LevelGeometry` per level | full transform per NGFF level, never inferred from shapes |
| Naming | `offset`, `map_points(points, frame=)`, `CoordinateEmbedding` | `translation`, `transform_point`, no redundant subclass |

Units are a wash: the model validates spellings with pint; xarrayrf compares canonical strings
and never converts. The dimensionality check in basis conversion is the one unit feature
xarrayrf lacks, and comes with P2.

## Alternatives Considered

- **Fold xarrayrf into the napari fork's experimental package.** Rejected: the fork cannot
  depend on patched xarray, and a viewer model should depend on the frame library, not host it.
- **Adopt the model's `CoordinateGraph` resolution.** Rejected: it is implicit resolution;
  only the edge semantics (P1) and typed non-results (P4) carry over.
- **Replace `axis_types` with an `AxisRole` enum.** Rejected for reserved values (P3).
- **Copy `CorrespondenceBasis` as a core enum.** Rejected: domain vocabulary in the core.
- **Home value geometry in an application or viewer package.** Rejected: it depends only on
  frames and is needed in every domain with directional values.

## Deferred Work

- Whether a per-variable default interpolation policy belongs in xarrayrf.
- A transform container (P4), still open in the
  [viewer boundary plan](../architecture/viewer_boundary_plan.md).

## Next Steps

1. P1 design note under `docs/dev/architecture/`, with the `compose` and `resample` policy and
   the persistence kind specified before code; settle the additive-kind rule in the schema-1
   freeze decision.
2. P3: document the reserved values in the core interface.
3. P2: designed in [the value geometry design](../architecture/value_geometry_design.md);
   stage 1 (the law and NumPy kernel) is next.
4. Viewer boundary plan items 1 and 2 (selection overhead, backend neutrality) remain
   prerequisites for any napari integration.
