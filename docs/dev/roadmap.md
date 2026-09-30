# Roadmap

**Status:** Active
**Last updated:** 2026-09-28
**Scope:** Project-wide: release criteria, planned work, parked work and the decisions a new
maintainer is most likely to revisit. Area-specific detail lives in the linked notes.

## Context

xarrayrf provides reference frames for native xarray objects: frames and transforms bound to
`DataArray`s that survive frame-preserving xarray operations and refuse or unframe otherwise.
The core, the `.rf` binding, persistence schema 1 and the NIfTI, DICOM, OME-NGFF and GeoTIFF
adapters are implemented. No package has been published.

A release requires the invariant in [the design](../design.md): no native operation may return
a valid-looking incorrect binding. On the patched xarray lane all 93 probed operations are
correct ([inventory](binding/binding_operation_inventory.md)); on released xarray several are
not, pending the upstream pull requests in [upstream_prs.md](xarray-upstream/upstream_prs.md).

## Current Decision

### Release criteria

1. The release invariant holds on released xarray, which needs the open upstream fixes to
   merge and ship ([xarray_patches.md](xarray-upstream/xarray_patches.md)).
2. A native-operation audit beyond the 93-case probe, concentrating on paths that could return
   a valid-looking incorrect binding.
3. Persistence schema 1 frozen ([persistence design](architecture/persistence_design.md)).
   Freezing is an explicit maintainer decision.
4. The TestPyPI rehearsal completed ([release notes](architecture/release_and_repository_notes.md),
   [releasing.md](../releasing.md)).

### Planned work, in rough priority order

- **xarray upstream.** Respond to review on the open bug-fix PRs, retire each local patch as a
  release ships it, and reduce the lifecycle hook patches to two `Index` methods before
  proposing them ([index hook design](xarray-upstream/index_hook_design.md)).
- **Freestanding grids.** A NumPy-only `Grid` value, declared intervals and anatomy on grids
  ([grid plan](architecture/grid_plan.md)); it also covers the viewer plan's pixel-free target domain.
- **Declared cells.** Slice thickness and intervals beyond the sample offset
  ([geometry and resampling](architecture/geometry_and_resampling_design.md)).
- **Viewer support.** Selection overhead, backend neutrality, pixel-free target domains and
  checked on-plane inversion ([viewer boundary plan](architecture/viewer_boundary_plan.md)).
- **Nonlinear geometry**, staged in the
  [draft plan](architecture/nonlinear_geometry_plan.md): field-backed transforms for deformable
  registration, then angular coordinate systems with a minimal CF reader (rectilinear GCM and
  celestial grids), then curvilinear coordinate fields, then provider adapters (pyproj, Astropy).
  Angular systems change core value objects and need their own reviewed plan.
- **Small follow-ups.** `xarrayrf.native` imports the private
  `xarray.namedarray._typing.duckarray` for its `DuckArray` alias; DICOM localizer
  (mixed-orientation) splitting; a NIfTI writer on top of `nifti.to_header`.

### Parked, with what would bring each item back

| Item | Returns when |
|---|---|
| Dataset `.rf` accessor and Dataset persistence | A consumer needs Dataset save and load |
| Product spacetime frames (shared spatial template with a private clock) | A consumer combines template-registered time series across runs |
| GeoTIFF export; NGFF `omero` metadata and labels; NGFF RFC-3 relaxed axes | A consumer needs them, or RFC-3 is adopted |
| Geographic and celestial data | Angular coordinate systems exist |
| Relativistic viewing (light-cone sampling, aberration, Doppler) | Angular systems and nonlinear transforms exist |
| Publishing a patched-xarray build | Upstream fixes are not released in time for a needed release |

### Decisions most likely to be revisited

- **Multi-image DICOM series** (echoes, b-values, time points) are assembled by applications,
  not by `xarrayrf.dicom`, which reads one spatial stack. Applications bind their arrays with
  `rf.frame`.
- **Datasets follow the shared-grid rule**: a binding belongs to its geometry dimensions, not
  to individual variables ([binding design](binding/binding_design.md)).
- **Identity is explicit.** `rf.assume_frame` is the only override; there is no global switch
  and no value-based frame matching. Unnamed NIfTI MNI152 and Talairach files share one frame;
  time-bearing NIfTI frames stay local ([adapters design](adapters/adapters_design.md)).
- **Geospatial scope.** xarrayrf does not replace rioxarray, xproj or rasterix; it provides
  frame identity and operation lifecycle beside them.
- **Names.** `nifti.open`, `dicom.open`, `ngff.open` and `geotiff.open` read lazily by default;
  `rf.resample_to` and `Lattice.affine` are settled
  ([core model design](architecture/core_model_design.md)).
- **Licences.** Code is MIT; the tour data release is CC BY 4.0.

## Alternatives Considered

Rejected alternatives are recorded in the area notes; the ones most often re-proposed are
per-variable Dataset ownership, a DICOM multi-image assembler in xarrayrf, value-compatible
frames without identity, an array wrapper or subclass, and attribute- or scalar-based carriers
([binding design](binding/binding_design.md)).

## Deferred Work

See *Parked* above.

## Next Steps

1. Continue the upstream PR work and the hook consolidation.
2. Start the native-operation audit.
3. Keep this roadmap current when an item starts, finishes or is parked.
