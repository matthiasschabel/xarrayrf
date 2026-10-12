# What xarrayrf offers a viewer data model, and what it does not own

**Status:** Active
**Last updated:** 2026-10-11
**Scope:** xarrayrf's boundary toward a viewer-facing data model (identity, fields, domains,
sources, napari adapters). Builds on [the comparison notes](compositional_model_comparison_notes.md)
and [the value geometry design](../architecture/value_geometry_design.md); changes neither.
Reviewed as a plan by Codex (gpt-6-astra, high) on 2026-10-11; findings folded in.

## Context

The napari compositional model's coordinate layer is being replaced by xarrayrf (comparison
notes, spike). What remains above xarrayrf is a viewer model: object identity with revisions and
subscriptions, source-backed fields with a value kind and interpolation policy, domains (grid,
mesh, polygon set, ragged readout), a lazy exact-versus-preview source contract, and napari
adapters. This note fixes which of those concepts xarrayrf serves, through which interface, and
which it deliberately leaves above itself. The first draft proposed a new package below xarrayrf
for fields and the value algebra; review showed that layer already exists as xarrayrf's enforced
NumPy-only geometry core, and that a field's basis cannot be named structurally because
coordinate systems and array coordinates carry no identity.

## Current Decision

xarrayrf owns the geometry of regular sample sets and the transformation law of values. It does
not own identity, change notification, storage, element topology or any viewer adapter.

### Provided upward

| Concept | xarrayrf API | Status |
|---|---|---|
| Frame identity, coordinate systems, directions | `ReferenceFrame`, `CoordinateSystem`, `DirectionVocabulary` | implemented |
| Placement of a grid of samples | `Grid`, `Geometry`, the `.rf` binding on `DataArray` | implemented; `Dataset`-level `.rf` is new work |
| Transforms between frames, with meaning | `AffineTransform`, `CompositeTransform`; the edge wrapper with role, accuracy, validity region, provenance (P1) | P1 designed, not implemented |
| Resampling onto a target grid | `resample_to`, `resample` | implemented for scalars; component variables in value geometry stage 3 |
| Value transformation law | `QuantityType` and `transform_components(values, linear_map, quantity_type)` in the geometry core; the component index and `rf.express_in` (P2) | staged in the value geometry design |
| Axis roles | reserved `axis_types` values `coordinate`, `local`, `component` (P3) | to document in the core interface |
| Placement of a point set (mesh vertices, landmarks) | value-positioned point sets, viewer boundary plan item 6 | planned; ragged geometry excluded |
| Format geometry | `dicom`, `nifti`, `ngff`, `geotiff` readers | implemented |

### Not owned, by design

- **Identity and change notification.** xarray objects are values. A model that keeps one
  identity across updates and notifies consumers wraps them; xarrayrf has no callback surface
  and wants none.
- **Storage and lazy reads.** xarrayrf reads what the array backend gives it. Exact-versus-preview
  reads, multiscale grouping and revision announcements are a source contract above it.
- **Element topology.** Which elements exist and how a field associates with them (vertex
  versus face values on a mesh, polygon membership, ragged readouts) is not geometry. xarrayrf
  places samples; it does not say what they are samples of.
- **Rank-0 value declarations.** A scalar's kind (density, pseudoscalar, categorical) and its
  interpolation policy have no dimension to carry them, so they are per-call arguments; the
  model stays authoritative and passes them explicitly.
- **Viewer adapters.** Nothing in xarrayrf imports a viewer.

### Where domains are scoped

A domain splits into a placement half and a topology half. xarrayrf owns placement for regular
sample sets and nothing else:

| Domain | Placement (xarrayrf) | Topology and association (above xarrayrf) |
|---|---|---|
| Structured grid | `Grid` plus xarray dims; complete today | none needed beyond the dims |
| Mesh | vertex positions as a point set (item 6, planned) | connectivity; vertex versus face association |
| Polygon set | vertex positions in a frame (point set, planned) | which vertices form which polygon; per-polygon fields; ragged, so never a binding |
| Ragged readout | none; not geometry | entirely above |

### Interface contract

Two calls cross the boundary downward into the geometry core, and the viewer model never
implements them: xarrayrf resolves which bases are involved and calls
`transform_components` with a numeric map; and `resample` consults the component index for
rank >= 1 and takes `quantity=` for rank 0. Upward, the model uses only public xarrayrf API. The
dependency rule is no upward concrete dependency; the model may register callbacks on its own
objects and implement xarrayrf-owned protocols.

## Alternatives Considered

- A new package below xarrayrf holding fields and the value algebra: rejected; the geometry core
  is that layer, and no non-xarray consumer justifies a split
  ([layering note](../architecture/core_layering_design.md)).
- A structural `Basis` protocol satisfied by `CoordinateSystem`: rejected; neither it nor
  `ArrayCoordinates` carries identity, so axis names, units and handedness cannot identify a
  basis. Basis resolution stays in xarrayrf.
- Identity and subscriptions as the upper half of xarrayrf: rejected; xarrayrf stays geometry
  only for the wider xarray community.
- A component index carrying rank-0 declarations: rejected in the value geometry design; no
  dimension to hold it.

## Deferred Work

- `Dataset`-level `.rf` and the per-variable round trip (selection, extraction, alignment,
  serialization) that a model built on a Dataset needs before it is more than a projection.
- A per-variable default interpolation policy, if a second consumer needs it.
- Point-set placement (item 6); ragged stays excluded.

## Next Steps

Value geometry stage 1, the P1 design note, P3 in the core interface; viewer boundary plan
items 1 and 2 before any viewer integration.
