# Development notes

Maintainer notes, grouped by area. Each carries a status header: `Active` and `Deferred` notes
are reviewed whenever their area changes; `Implemented` notes record decisions and their
reasons. The stable documents are [the design](../design.md),
[the core interface](../core_interface.md) (normative) and [the upstream workflow](../upstream.md).

- [roadmap.md](roadmap.md): release criteria, planned and parked work, and the decisions most
  likely to be revisited. Start here.

## architecture/

- [core_model_design.md](architecture/core_model_design.md): frames and coordinate systems,
  identity and equivalence, roles, orientation, the `Transform` protocol, units, affine classes.
- [geometry_and_resampling_design.md](architecture/geometry_and_resampling_design.md): the
  `Geometry` view, sample offsets and cells, declared intervals (open), resampling.
- [grid_plan.md](architecture/grid_plan.md): the freestanding `Grid` sampling value, its doors,
  declared intervals, anatomy on grids, and the frame-identity rule.
- [nonlinear_geometry_plan.md](architecture/nonlinear_geometry_plan.md): draft staging for
  nonlinear geometry and the plan for displacement-field transforms.
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
- [release_and_repository_notes.md](architecture/release_and_repository_notes.md): package
  metadata, CI lanes, dependency floors and the TestPyPI rehearsal.

## binding/

- [binding_design.md](binding/binding_design.md): the native `.rf` binding, why a joint index
  was chosen over other carriers, the Dataset rule, and how operations carry or refuse it.
- [binding_operation_inventory.md](binding/binding_operation_inventory.md): the 93-case probe
  matrix across the stock, upstream and patched xarray lanes.

## xarray-upstream/

- [xarray_patches.md](xarray-upstream/xarray_patches.md): the patch manifest (base, series,
  validation, pin, upstream status).
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
- [transform_adapters_design.md](adapters/transform_adapters_design.md): proposed adapters that
  wrap registration results from CASTalign, ITK, ANTs and elastix as xarrayrf transforms,
  without running an optimizer.
