# Development notes

Maintainer notes, grouped by area. Each carries a status header: `Active` and `Deferred` notes
are reviewed whenever their area changes; `Implemented` notes record decisions and their
reasons. The stable documents are [the design](../design.md),
[the core interface](../core_interface.md) (normative) and [the upstream workflow](../upstream.md).

- [roadmap.md](roadmap.md): release criteria, planned and parked work, and the decisions most
  likely to be revisited. Start here.
- [changelog.md](changelog.md): maintainer record of fixes found in review.
- [codebase_audit_review.md](codebase_audit_review.md): completed audit decisions, validation
  provenance and the deferred A4/A6 convenience questions. No further audit stage is queued.

## architecture/

- [geometry_tolerance_integration_notes.md](architecture/geometry_tolerance_integration_notes.md):
  existing configurable sampling queries, tolerance ownership, and deferred pirana integration.

- [core_model_design.md](architecture/core_model_design.md): frames and coordinate systems,
  identity and equivalence, roles, orientation, the `Transform` protocol, units, affine classes.
- [geometry_and_resampling_design.md](architecture/geometry_and_resampling_design.md): the
  `Geometry` view, sample offsets and cells, declared intervals, resampling.
- [support_aware_resampling_design.md](architecture/support_aware_resampling_design.md): what a
  value on a declared interval means, and the resampling operators (step, PCHIP on the integral,
  overlap mean, smoothest consistent) that respect it; interval claims and the box methods are
  implemented, PCHIP and smooth are next.
- [core_layering_design.md](architecture/core_layering_design.md): which modules form the
  xarray-free geometry core, and the private contracts another array binding would use.
- [grid_plan.md](architecture/grid_plan.md): implemented freestanding `Grid` rationale, its doors,
  declared intervals, anatomy on grids, and complete versus anonymous frames.
- [nonlinear_geometry_plan.md](architecture/nonlinear_geometry_plan.md): staging for nonlinear
  geometry and the structural rules (chart identity, metrics, angular axes, tensors) every stage
  respects.
- [field_transform_design.md](architecture/field_transform_design.md): stage 1, displacement,
  position and B-spline transforms on `Grid`, declared inverse pairs, a Jacobian-determinant
  diagnostic, importers, and the deformation models to support after it.
- [persistence_design.md](architecture/persistence_design.md): schema 1 of `encode`/`decode`,
  provisional until frozen.
- [cross_domain_cases.md](architecture/cross_domain_cases.md): the DICOM, OME-NGFF, astronomy
  and GIS acceptance cases, with the worked examples the point-oracle tests reproduce.
- [prior_art_coverage_study.md](architecture/prior_art_coverage_study.md): spatialdata/NGFF,
  geospatial xarray libraries, Astropy/GWCS and ITK/VTK/nibabel compared with xarrayrf.
- [viewer_boundary_plan.md](architecture/viewer_boundary_plan.md): what the core owes an
  interactive viewer such as napari, and where the viewer's responsibility begins.
- [relativity_notes.md](architecture/relativity_notes.md): spacetime arrays as a test of the
  design's generality, and the deferred relativistic-viewing stages.
- [curved_spacetime_review.md](architecture/curved_spacetime_review.md): whether the model
  carries curved, pseudo-Riemannian spacetime; the layers that fit, the five gaps, and the
  smallest changes that would close them.
- [release_and_repository_notes.md](architecture/release_and_repository_notes.md): package
  metadata, CI lanes, dependency floors and the TestPyPI rehearsal.

## binding/

- [binding_design.md](binding/binding_design.md): the native `.rf` binding, why a joint index
  was chosen over other carriers, the Dataset rule, and how operations carry or refuse it.
- [binding_operation_inventory.md](binding/binding_operation_inventory.md): the 93-case probe
  matrix across the stock, upstream and patched xarray lanes.

## xarray-upstream/

- [xarray_patches.md](xarray-upstream/xarray_patches.md): the patch manifest (base, series,
  validation and pin).
- [upstream_prs.md](xarray-upstream/upstream_prs.md): the independent bug-fix pull requests and
  what remains open on each.
- [index_hook_design.md](xarray-upstream/index_hook_design.md): reducing the lifecycle hook
  patches to two `Index` methods, and the cross-extension reproducers that motivate them.

## napari-3783/

- [transformations_wishlist_review.md](napari-3783/transformations_wishlist_review.md): the
  model-side coverage of napari/napari#3783's transform wish list and thread requests.

## castalign/

- [castalign_comparison_review.md](castalign/castalign_comparison_review.md): CASTalign's
  transforms, graph and GUI compared with xarrayrf; overlaps, model-layer problems, synergies.

## adapters/

- [adapters_design.md](adapters/adapters_design.md): the shared adapter contract, frame
  identity, and the NIfTI, DICOM, OME-NGFF and GeoTIFF adapters.
- [transform_adapters_design.md](adapters/transform_adapters_design.md): adapters that wrap
  registration results from CASTalign, ITK, ANTs and elastix as xarrayrf transforms, without
  running an optimizer; the ITK-family codec is scheduled with the field-backed transforms.
