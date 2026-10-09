# Nonlinear geometry: staging and structural decisions

**Status:** Active (roadmap; stage 1 designed, later stages outlined)
**Last updated:** 2026-10-09
**Scope:** The order in which xarrayrf takes on nonlinear geometry, and the structural rules
(chart identity, metrics, angular axes, tensors) that every stage must respect. Stage 1 has its
own design, [field-backed transforms](field_transform_design.md); later stages are outlined with
their acceptance cases and open decisions. The [curved spacetime review](curved_spacetime_review.md)
is the stress test these rules answer.

## Context

[The design](../../design.md) §7 allows `world = F(c)` to be nonlinear, bounded and partially
invertible, and names three non-affine representations: sampled coordinate fields, field-backed
transforms and external providers. Only affine transforms exist today.

Three candidate directions were considered: deformable image registration, geographic (GIS)
frames and Astropy celestial frames. Earth-system model output (GCMs, ocean and atmosphere
reanalyses) is a large share of xarray's use and also needs spherical coordinates, so it is
treated as a first-class target of the later stages rather than as part of "GIS".

Three facts shape the staging:

1. **Geographic and celestial data need angular coordinate systems first.** Longitude and
   latitude, or right ascension and declination, are periodic and singular at the poles.
   `CoordinateSystem` has a `representation` parameter but only `"cartesian"` exists, and the
   [relativity notes](relativity_notes.md) list the open decisions for angular charts.
2. **Deformable registration needs no change to the core value objects.** Source and target
   frames are Cartesian; a field is a transform between them.
3. **Registration and curvilinear grids share storage, not interpolation.** A displacement field
   and a 2-D latitude/longitude grid are both a regular grid of stored values with a bounded
   domain. Stage 3 reuses stage 1's storage, grid evaluation and validity machinery, but angular
   values need a seam-aware interpolant.

Only stage 1 has a consumer (Pirana). Stages 2 to 4 are general xarrayrf work; the rules below
exist so that stage 1 does not box them in, not to schedule them.

## Current Decision

### Staging

| Stage | Capability | Unlocks | Core change |
|---|---|---|---|
| 1 | [Field-backed transforms](field_transform_design.md) between Cartesian frames | Deformable registration (ITK/ANTs/FSL warps, B-splines, NGFF 0.6 `displacements`, `coordinates`, `bijection`) | New transform classes; per-point validity in `resample`; array-valued persistence |
| 2 | Angular charts and chart transitions | Rectilinear longitude/latitude grids (most GCM and reanalysis output), celestial coordinates, lon/lat ↔ Earth-centred Cartesian within one frame | Chart declarations (representation, period, bounds); explicit transitions that keep frame identity |
| 3 | Curvilinear coordinate fields | 2-D longitude/latitude grids (ocean models, cubed-sphere tiles, satellite swaths), Cartesian curvilinear grids | Array-coordinate transforms backed by coordinate arrays; a source locator for approximate lookup; 2-D coordinates in the binding |
| 4 | Provider adapters | Reprojection between CRSs (pyproj), celestial WCS (Astropy, APE 14), CF `grid_mapping` | Adapters only |

Curvilinear coordinate fields come before the WCS provider adapters. They are far more common in
xarray workloads than FITS WCS, and a provider adapter is thin once angular systems exist,
because the provider does the mathematics. For earth-system data, however, curvilinear grids
are almost always longitude/latitude, so stage 3 depends on stage 2; a Cartesian-only stage 3
would serve swaths in projected coordinates and distortion-corrected imaging but not GCM grids.

A CF adapter (`grid_mapping`, 1-D and 2-D `lat`/`lon`, `bounds`) grows with stages 2 to 4
rather than being a stage of its own: CF is how most earth-system data declares its geometry.

### Structural rules

Settled in the 2026-10-08 review (Codex gpt-6-astra, high) except where marked open. They
constrain designs; none of them is implemented before its stage.

**Frame identity across chart changes.** Identity already survives a change of coordinate
system: `ReferenceFrame.with_coordinate_system` keeps identifier, definition and context, and
`is_equivalent_frame` ignores the coordinate system (`src/xarrayrf/_frame.py`). What is
missing is narrower: `coordinate_system_change` derives only signed permutations, and
`rf.assume_frame` and adapter `frame=` adopt only affine changes. The rule:

- A frame is expressed in one chart. Invariant identity, definition and context survive chart
  changes; projection, coordinate convention and chart-domain parameters belong to the chart
  declaration, not to frame identity.
- A conversion that cannot be derived needs an explicit transition transform. There is no
  transition registry and no automatic discovery; `with_coordinate_system` does not carry
  transform graphs.
- Strict endpoint equality and literal alignment stay: equivalent frames do not make
  coordinates interchangeable.
- A shared radius or datum alone never implies shared identity.
- 3-D lon/lat/height ↔ Earth-centred Cartesian is a chart change of one frame. 2-D lon/lat into
  3-D is a surface embedding, not a bijection, and is a transform between frames.

Consequence for existing code: `geotiff.crs_frame` puts the whole projected CRS into frame
identity, so two projections of one Earth are unrelated frames. Whether to migrate it is open
decision 1.

**Metrics.** A metric is optional. In its absence, calculations that need one use the identity
matrix as a documented coordinate-space convention; that is today's behaviour, and for
Cartesian axes sharing one length unit it is the physical Euclidean answer. `Lattice.spacing`
and `CoordinateSystem.axis_codes` already refuse axes with different units. Two qualifications:

- Axes sharing a unit are not thereby Euclidean. `(ct, x)` in metres has spacing computed with
  Euclidean norms, and a Lorentz boost classifies as generic `affine`
  (`tests/test_classification.py`); both are correct as numeric statements and wrong as
  physical ones. Only a declared metric changes that.
- An angular chart's Euclidean geometry is not the identity in chart coordinates: a sphere of
  radius R in radians has the induced metric R²·diag(cos²φ, 1). Identity-matrix calculations
  are refused on angular axes rather than defaulted.

Where a metric is declared, it belongs to the frame, with components tied to a chart, not to the
reusable axis-only `CoordinateSystem`. It may be constant or a field; under a chart transition
`y = F(x)` it transforms as `g_y = J^-T g_x J^-1`, so a constant metric can become a field. It is
not transported through arbitrary registrations. `affine_class` stays a numeric classifier;
metric preservation (`J^T g_t(F(x)) J = g_s(x)`) is a separate query, and sampling it at points
establishes residuals there, not global isometry. Whether this qualified rule or an unqualified
"absent means Euclidean" is adopted is open decision 2.

**Angular axes.** Period, representative interval, bounds and angular roles belong to the chart
declaration of angular representations only, so Cartesian users see no new parameters.
Intrinsic periodicity is distinct from sample coverage: a regional longitude grid does not wrap.
Seam policy covers duplicate endpoints, interpolation neighbours across the seam and regional
antimeridian coverage. Bounds and singularities belong to operations: a forward map may be
evaluable where its inverse or Jacobian fails.

**Tensors and frame fields.** Per-index variance applies `J` to contravariant slots and `J^-T` to
covariant slots, requiring invertibility where needed. A tetrad is a separate `FrameField`
(fibre map) value object with a base chart, component bases, a domain and pointwise matrices; it
need not pass `check_transform`, so no exception to the point-transform rule is needed. As
standalone operations both leave `Grid`, `Geometry`, the binding and scalar `resample`
unchanged. Tensor-aware resampling would not: it needs component metadata, persistence and a
defined order of conversion and interpolation, because with varying bases converting before
interpolating differs from converting after. Neither a metric nor a tetrad specifies parallel
transport.

### Stage 2: angular charts (outline)

- New `representation` values for longitude/latitude charts (spherical and geodetic) and the
  celestial sphere, with the chart declaration above: a periodic interval for longitude and
  bounded latitude.
- Explicit chart transitions that keep frame identity, designed together with the
  representations: lon/lat/height ↔ Earth-centred Cartesian is the first case.
- Label alignment stays literal (0 and 360 never align implicitly). Resampling across the
  longitude seam is explicit and supported, because a global grid is unusable without it;
  evaluation at the poles is refused or reported per point.
- No metric is inferred from angles. Area weights, great-circle distances and conservative
  regridding stay with dedicated tools (xESMF, xarray-regrid, pyresample).
- A model's spherical Earth and the WGS 84 ellipsoid are different frames; treating model
  latitude as geodetic latitude is a known source of errors of tens of kilometres. Declared
  identity (datum or sphere radius in the frame definition) keeps them apart.
- A minimal CF reader belongs in this stage, not stage 4: rectilinear `lat`/`lon` with CF units,
  and the earth figure from `grid_mapping` (`latitude_longitude` with `earth_radius`,
  `semi_major_axis` and related attributes) in the frame definition. Without it the GCM
  acceptance case has no import path.
- Order: chart declarations and transitions, then seam-aware sampling, then the CF reader.
- Acceptance cases: a rectilinear global model grid with a declared spherical earth read through
  that reader and resampled across the longitude seam, a regional grid crossing the
  antimeridian, a lon/lat/height ↔ Earth-centred round trip within one frame, and a celestial
  example checked against Astropy.

### Stage 3: curvilinear coordinate fields (outline)

- A transform from `ArrayCoordinates` to a frame backed by coordinate arrays (the 2-D `lat` and
  `lon` of an ocean or swath grid), reusing stage 1's storage, grid evaluation and validity
  machinery with stage 2's seam-aware interpolant.
- Forward evaluation is exact at nodes. The inverse (frame point to array position) is a search.
  `resample`'s source locator requires an exact inverse today (`locator` in `_positions.py`), so
  this stage adds a separate source-location contract for approximate lookup, with its own
  failure reporting; it does not impersonate `SupportsInverse`.
- `resample` currently refuses multidimensional coordinate fields; this stage lifts that for
  declared coordinate-field transforms.
- **2-D coordinates in the binding.** The binding owns one index per source axis and `rf.frame`
  refuses coordinates that share a dimension. Probe the simpler representation first: keep 1-D
  index coordinates in the binding and hold 2-D lon/lat as transform parameters. Only if the
  operation inventory shows required workflows failing does the binding own auxiliary 2-D
  coordinates, which is a redesign.
- Cubed-sphere tiles are curvilinear per tile; tile connectivity is deferred.

### Stage 4: provider adapters (outline)

- pyproj: a transform between two CRS frames (projected or, after stage 2, geographic), wrapping
  an immutable snapshot of the `Transformer` configuration.
- Astropy: an APE 14 WCS as a transform from array coordinates to a celestial frame.
- CF `grid_mapping` beyond stage 2: projected mappings through pyproj, and rotated-pole grids.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Geographic frames first | Needs angular systems before any value; rioxarray, odc-geo, xproj and rasterix already serve projected rasters, and xarrayrf sits beside them |
| Astropy WCS first | A thin adapter once angular systems exist; a small xarray audience on its own |
| Curvilinear grids before angular systems | Serves Cartesian curvilinear data only; GCM grids are longitude/latitude |
| Angular axes as stage 2 alone, chart transitions later | Lon/lat and Earth-centred Cartesian are charts of one Earth; transitions designed afterwards would have to retrofit identity |
| A transition registry with automatic discovery | Hidden graph search over frames; explicit transitions are inspectable and sufficient |
| Metric values on `CoordinateSystem` | A coordinate system is a reusable axis declaration; a metric is a property of the space expressed in a chart |
| Metric-aware `affine_class` | Changes a numeric classifier's meaning for existing users; a separate preservation query adds the physics without that |
| Mandatory pyproj dependency | Imposes engine conventions; adapters and dev-only oracles suffice |

## Deferred Work

- Implementation of the metric declaration, the preservation query, per-index tensor rules and
  `FrameField`: no consumer yet.
- Unstructured meshes (UGRID, MPAS, ICON) and cubed-sphere connectivity.
- Parametric vertical coordinates (CF `formula_terms`, hybrid sigma-pressure), which make the
  vertical coordinate depend on other variables such as surface pressure.
- Time-dependent transforms.

## Open Decisions

1. Whether `geotiff.crs_frame` migrates so that projections of one Earth share an identity
   (projection moving into the chart declaration) or identity stays CRS-specific. Needed when
   stage 2 starts; stage 1 does not depend on it.
2. The metric rule: the qualified rule above, or "absent means Euclidean" unqualified, with
   physical geometry documented as Euclidean-only.
3. Whether stage 2 includes seam-aware resampling (proposed: yes, for global grids) or only
   declarations and evaluation, as the relativity notes originally recommended.
4. Whether the stage-3 binding probe starts alongside stage 2.

## Next Steps

1. Stage 1 per [its design](field_transform_design.md).
2. Before stage 2, a reviewed design for chart declarations and transitions, settling open
   decisions 1–3.
3. Before stage 3, the binding probe with the operation inventory.
