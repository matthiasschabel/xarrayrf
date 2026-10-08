# Roadmap

**Status:** Active
**Last updated:** 2026-10-07
**Scope:** Project-wide: release criteria, planned work, parked work and the decisions a new
maintainer is most likely to revisit. Area-specific detail lives in the linked notes.

## Context

xarrayrf provides reference frames for native xarray objects: frames and transforms bound to
`DataArray`s that survive frame-preserving xarray operations and refuse or unframe otherwise.
The core, the `.rf` binding, freestanding `Grid` values with declared cell intervals,
anatomical reformatting, complete and anonymous frames, persistence schema 1 and the NIfTI,
DICOM, OME-NGFF and GeoTIFF adapters are implemented. No package has been published.
The [five-stage codebase audit](codebase_audit_review.md) is complete; no further audit stage
is queued. Deferred conveniences and release/viewer work below are separate.

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
4. All supported dependency-floor CI lanes pass. This criterion is met on `da5f627`:
   [hosted run 37668735524](https://github.com/matthiasschabel/xarrayrf/actions/runs/37668735524)
   passes locked, minimum-core, minimum-resample and current. Hosted validation of the patched
   pin remains separate work; passing stock CI with known xfails does not meet criterion 1.
5. The TestPyPI rehearsal completed ([release notes](architecture/release_and_repository_notes.md),
   [releasing.md](../releasing.md)).

### Planned work, in rough priority order

- **Release preparation.** Extend the native-operation audit, add hosted validation of the
  immutable patched pin, prepare the schema-freeze decision and complete the TestPyPI rehearsal
  when publishing is authorized. Dependency-floor hosted verification is complete.
- **xarray upstream.** Respond to review on the open bug-fix PRs, retire duplicate patches when
  a new series baseline includes their merged fixes, verify released-xarray support when fixes
  ship, and reduce the lifecycle hook patches to two `Index` methods before
  proposing them ([index hook design](xarray-upstream/index_hook_design.md)).
- **Viewer support.** Selection overhead, backend neutrality, source footprints for `Grid`
  targets and checked on-plane inversion
  ([viewer boundary plan](architecture/viewer_boundary_plan.md)). The pixel-free target domain
  itself is done (`Grid`, [grid plan](architecture/grid_plan.md)).
- **Framed concat along a geometry dimension** (stitching slabs that share a transform), if a
  consumer needs it; overlapping labels would stack samples at the same place.
- **Nonlinear geometry**, staged in the
  [draft plan](architecture/nonlinear_geometry_plan.md): field-backed transforms for deformable
  registration, then angular coordinate systems with a minimal CF reader (rectilinear GCM and
  celestial grids), then curvilinear coordinate fields, then provider adapters (pyproj, Astropy).
  Angular systems change core value objects and need their own reviewed plan.
- **Small follow-ups.** DICOM localizer (mixed-orientation) splitting and a NIfTI writer on top
  of `nifti.to_header`, when a consumer needs them. `DuckArray` already uses a local protocol.

### Parked, with what would bring each item back

| Item | Returns when |
|---|---|
| A4: common metadata Grid/report convenience | A concrete metadata-only Grid or ordinary-reader report consumer ([audit rationale](codebase_audit_review.md#deferred-work)) |
| A6: unlabeled nongeometry construction and mixed Grid/Geometry coincidence | Existing DataArray framing or Grid snapshot comparison blocks a real workflow ([audit rationale](codebase_audit_review.md#deferred-work)) |
| Grid snapshot RangeIndex step parity | Rounded coordinate snapshots block a measured workflow |
| Cubic missing-data policy; general-path many-context routing optimization | A concrete NaN case or profiling shows a material limitation |
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
- **Identity is explicit.** `rf.assume_frame` and every reader's `frame=` (one adoption
  contract) are the only overrides; there is no global switch and no value-based frame
  matching. Sources that
  name no space give anonymous frames, never a guessed identity (content hashes, path
  fingerprints and a shared default world were rejected; see the
  [anonymous-frame rationale](architecture/grid_plan.md#5-complete-and-anonymous-frames-implemented)).
  Unnamed NIfTI MNI152 and Talairach files share one frame; every other NIfTI frame
  (scanner, aligned, other-template, time-bearing) is anonymous ([adapters design](adapters/adapters_design.md)).
- **Orientation letters** are accepted only by `xarrayrf.anatomy`, with one convention (the
  direction an index increases toward, as nibabel's `aff2axcodes`); coordinate systems keep
  explicit RFC-4 tokens. Named planes keep DICOM display order in-plane; the slice axis
  increases toward S, P and R (an xarrayrf choice).
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
