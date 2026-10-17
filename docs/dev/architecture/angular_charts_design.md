# Angular charts and chart transitions

**Status:** Implemented (2026-10-12; stages 1 to 4 per their plans; section 4 amended by stage 3)
**Last updated:** 2026-10-17
**Scope:** Stage 2 of the [nonlinear geometry plan](nonlinear_geometry_plan.md): angular coordinate
systems (longitude/latitude, the celestial sphere), transitions between charts of one frame,
seam-aware resampling, and a minimal CF reader. Settles that plan's open decisions 1 to 3. It
is the prerequisite for pointwise bases (`FrameField`), stage 4 of the
[value geometry design](value_geometry_design.md).

## Context

`CoordinateSystem` admits only `representation="cartesian"` and refuses angular units
(`check_units(..., reject_angular=True)` in `src/xarrayrf/_validation.py`). Longitude/latitude
data, which is most GCM and reanalysis output, therefore cannot be declared, and the only
workaround (`units=None` on the angles) loses periodicity and the poles. The
[special-relativity note](relativity_notes.md) and the [curved spacetime review](curved_spacetime_review.md)
(gap 3) list the same prerequisite.

The plan fixes the structural rules this design must follow: a frame is expressed in one chart;
identity, definition and context survive chart changes; a conversion that cannot be derived
needs an explicit transition transform, with no registry; label alignment stays literal; no
metric is inferred from angles; identity-matrix calculations are refused on angular axes; a
spherical model Earth and WGS 84 are different frames.

What exists to build on:

- `ReferenceFrame.with_coordinate_system` keeps identity; `is_equivalent_frame` ignores the
  coordinate system (`src/xarrayrf/_frame.py`).
- `coordinate_system_change` (`src/xarrayrf/_orientation.py`) derives signed permutations
  between oriented systems of one frame.
- Transforms are structural protocols (`SupportsPoints`, `SupportsJacobian`, `SupportsInverse`)
  with pointwise Jacobians; non-affine transforms are already admitted.
- Source location (`locator` in `src/xarrayrf/_positions.py`) inverts the source geometry's
  transform and converts frame coordinates to fractional positions per axis
  (`coordinate_to_position` in `_sampling.py`); interpolation is `scipy.ndimage.map_coordinates`
  with `mode="nearest"` and clipped positions (`_interpolate` in `_resampling.py`).

## Current Decision

### 1. Chart declaration: angular representations on `CoordinateSystem`

The chart declaration is the `CoordinateSystem`, extended only for non-Cartesian
representations. Cartesian construction and equality are unchanged.

| Representation | Angular axes | Third spatial axis | Typical use |
|---|---|---|---|
| `"spherical"` | longitude, latitude (geocentric) | optional `"radius"` (distance from centre) | celestial sphere, spherical model Earth, spherical-polar physics charts |
| `"geodetic"` | longitude, latitude (ellipsoid normal) | optional `"height"` (above the ellipsoid) | WGS 84 and other ellipsoidal Earths |

New keyword `axis_roles`: one of `"longitude"`, `"latitude"`, `"radius"`, `"height"` or `None`
per axis. Rules:

- Required for, and only accepted with, a non-Cartesian representation. Exactly one
  `"longitude"` and one `"latitude"`; at most one of `"radius"` (spherical only) or `"height"`
  (geodetic only). Axes with role `None` (time, pressure level, channel) are ordinary,
  non-angular axes and keep today's Cartesian unit rules.
- Angular axes must declare a recognized degree or radian unit. The period is derived from the
  unit: 360 for degree spellings (including CF `degrees_east`, `degrees_north`), 2π for
  `"rad"`/`"radian"`/`"radians"`. Other angular spellings (arcmin, arcsec) are refused on
  angular axes for now, since nothing needs them and their period is a unit table, not a rule.
- Angular axes cannot be oriented (no direction tokens): a direction vocabulary names constant
  directions, which longitude and latitude are not.
- Latitude bounds are ±90° (±π/2) and are a property of the representation, not a parameter.
- Equality and hashing include representation and roles.

There is no representative longitude interval in the chart. The plan listed one; this design
drops it because coordinate labels stay literal (0 and 360 never align), so a chart that
constrained labels to `[-180, 180)` would reject valid regional grids crossing the
antimeridian (labels 170 to 190), and a chart that only recorded a preferred branch would make
two otherwise identical systems unequal and so make their frames unequal transform
endpoints. The branch is instead a parameter of the operations that *produce* longitudes:
inverse transitions (section 3) take `longitude_start` (default −180 in degrees, −π in
radians), and resampling never normalizes labels.

### 2. Refusals on angular axes (open decision 2)

Adopt the plan's qualified metric rule. No metric is implemented in this stage; the rule is
enforced as refusals:

- `Lattice.spacing` and `CoordinateSystem.axis_codes` refuse when any angular axis is involved
  (they already refuse mixed units; the new refusal names the angular axis and points to
  dedicated regridding tools).
- `affine_class` stays numeric and unchanged.
- Box methods (`step`, `overlap_mean`) refuse a latitude slab axis: a mean over degrees of
  latitude is not an area mean. Longitude slab axes are allowed, since at fixed latitude a
  degree mean is an arc-length mean; this is stated, not inferred. Conservative regridding stays
  with xESMF, xarray-regrid and pyresample.
- `is_coincident` measures in index steps and needs no change.

"Absent means Euclidean" unqualified is rejected because it would give lon/lat grids a
Euclidean spacing in degrees, which the plan names as the failure to avoid.

### 3. Chart transitions within one frame

`coordinate_system_change` keeps its contract (signed permutations, no unit rescaling). It
gains the angular cases it can still derive exactly: reordering axes between two systems with
the same representation, roles and units (a permutation). Between Cartesian and angular
representations it raises, naming the transition classes below.

Two public transition classes, each implementing `SupportsPoints`, `SupportsJacobian` and
`SupportsInverse`, with source and target required to be equivalent frames
(`is_equivalent_frame`) in the stated representations:

| Class | Source chart | Target chart | Parameters |
|---|---|---|---|
| `SphericalToCartesian` | spherical (lon, lat, radius) | Cartesian 3-axis | none |
| `GeodeticToCartesian` | geodetic (lon, lat, height) | Cartesian 3-axis (Earth-centred) | `semi_major_axis`, `inverse_flattening` (`0.0` means a sphere, as in pyproj; `inf` is refused) |

- The source must have exactly the three spatial roles and no other axes; a time or level axis
  is composed alongside by the caller (a block-diagonal composite is deferred). 2-D lon/lat
  into 3-D is an embedding between frames, per the plan, and is not a transition.
- Units: the length unit of `radius`/`height` must equal every Cartesian axis's unit, and
  `semi_major_axis` is in that unit; nothing is rescaled.
- Cartesian axis order is taken from the target system; axes are unoriented or oriented in any
  vocabulary, and the class fixes the convention x toward (lon 0, lat 0), z toward the north
  pole, right-handed. The target system declares which of its axes is which through axis order
  `(x, y, z)`; a different order composes with `coordinate_system_change`.
- `inverse()` returns the reverse transition with `longitude_start`. Geodetic inverse uses a
  closed-form or iterated solution with a stated accuracy (Bowring or Vermeille; settled in
  the stage plan with a pyproj oracle).
- Poles and the centre use only today's contract (`transform_point` returns coordinates or
  raises); stage 2 does not depend on the masked evaluation of
  [field-backed transforms stage 1c](field_transform_design.md). 1c landed on 2026-10-17 for
  the built-in field types only; transitions still raise (Deferred Work).
  Separately per operation:
  - *Forward points:* the forward domain is the inverse's image, so every accepted point
    round-trips: latitude in ±90°; a radius > 0; a geodetic height above `−N(φ)(1 − e²)`
    (negative heights included, down to where the normal reaches the equatorial plane or the
    axis; `−a` for a sphere), with the image outside the evolute bound below. Anything else
    raises: deeper points have a nearer surface foot elsewhere (amended in the
    [stage 2 plan](angular_charts_stage2_plan.md) review, 2026-10-12).
  - *Inverse points:* on the polar axis every longitude names the same point, so the inverse
    returns `longitude_start` there as an ordinary, valid result. Spherical: only the centre
    has no latitude, and there the inverse raises. Geodetic: the inverse is unique and meets
    its stated accuracy outside the ellipsoid's evolute. The test is the conservative enclosing
    radius `e² · a² / b` (the evolute's polar semi-axis, about 42.84 km for WGS 84; its
    equatorial semi-axis `e² · a` is smaller); points closer to the centre raise.
  - *Jacobian:* the forward Jacobian is finite everywhere and singular at the poles and the
    centre, where it is returned as is. The inverse Jacobian does not exist on the polar axis
    or at the centre, and `jacobian(at=...)` raises there, naming the point.
  - *Resampling:* scalar resampling needs only points, so pole samples and polar targets are
    usable. Declared tensor components need the Jacobian. Today the component path
    (`apply_pointwise` in `_component_resampling.py`) calls `jacobian_map.jacobian` outside
    the `try` that reports the singular target sample, so a raising Jacobian would escape
    unnamed. Stage 2 moves that call inside the handler, catching `ValueError` from it as
    well, and on that error path finds the first failing target sample by evaluating the
    block's Jacobians point by point, so both a singular matrix and a raised Jacobian fail with
    the error that names the target sample.
- The figure (sphere radius, or ellipsoid) is a constructor parameter, not read from the frame
  definition. The frame definition still carries it for identity (section 5); a transition
  whose parameters disagree with a definition using the reserved keys is refused at
  construction. This keeps the core from parsing open `definition` metadata except for those
  reserved keys.

A spherical model Earth with height above the sphere uses `geodetic` with
`inverse_flattening=0.0` (geodetic and geocentric latitude coincide on a sphere); `spherical`
with `radius` is for charts measured from the centre.

### 4. Seam-aware resampling (open decision 3: included)

A global longitude grid is unusable without it, so it is in this stage.

**Amended 2026-10-12 by the [stage 3 plan](angular_charts_stage3_plan.md):** location is exact
(ported from xarraybc's `periodic_bracket_array`), so nonuniform and float32 global grids are
included and the uniformity requirement below no longer applies; coverage comes from declared
intervals or, conservatively, from labels; and latitude crosses the poles on global grids with
antipodal longitude pairs ("latitude never wraps" below is superseded). The rest of this section
is the original design.

**Where wrapping applies.** On a source geometry whose frame has an angular representation, the
longitude frame axis wraps when the source geometry's transform is affine, that frame axis
depends on exactly one source axis with scale `s`, and that source axis affects no other frame
axis (its row and its column of the matrix each have one non-zero entry). A coupled map such as
`lon = u, lat = u + v` does not wrap, since shifting `u` by a period would move latitude. The period in source coordinates is then `P / |s|`. Any other
transform (rotated pole through a general affine, non-affine) gets no wrapping: positions
outside the labels are outside, as today, and the docs say so.

**Coverage.** For the wrapping source axis with labels `l_0 … l_{n-1}` (strictly monotonic,
span less than one period):

- *Global*: labels are uniform (within the existing uniform-step tolerance) and `n · |step|`
  equals the period; `uniform_step` returns a signed step, and descending labels are
  admitted. The position of a coordinate `c` is `((c − l_0) / step) mod n`, with the signed
  step, exact everywhere including the seam gap, and interpolation between `n − 1` and `0`
  crosses the seam.
- *Global with duplicate endpoint*: `(n − 1) · step` equals the period and the first and last
  samples are one period apart. The last sample is dropped for location and interpolation; its
  values are not checked against the first.
- *Regional*: anything else. A target coordinate is shifted by whole periods to the branch
  starting one cell extent before `l_0` and then located as today, so a regional grid crossing
  the antimeridian (labels 170 to 190, or −10 to 10) is found from targets labelled −175 or 355.
- Nonuniform longitude labels are never treated as global in this stage, even when declared
  intervals tile the period: today's location extrapolates past the last centre with the last
  step, which is wrong inside a nonuniform seam gap. They are located as regional. (Gaussian
  grids are nonuniform in latitude, not longitude, so this costs little.)

**Interpolation across the seam.** For a global axis the source block is extended periodically
by a halo of samples on both sides along that axis before prefiltering and `map_coordinates`,
and positions are offset by the halo. Linear and nearest need a halo of 1; cubic uses a halo
long enough that the spline prefilter's boundary error is below the existing roundoff allowance
(fixed and tested in the stage plan). `scipy`'s `grid-wrap` mode is not used because it applies
to every axis.

**Latitude and poles.** Latitude never wraps. Targets beyond the outer latitude samples are
outside, as on any axis. Pole rows are ordinary samples; nothing is averaged across the pole.

**Execution path.** A wrapping axis bypasses the lattice fast path: `_lattice_map` feeds
`ndimage.affine_transform` directly and the planner crops the source window first
(`_crop_window`), and neither a modulo nor a branch jump is affine, while a crop can drop
samples needed across the seam. Wrapped sources take the general block path without cropping along the
wrapping axis. The periodic step happens inside location, before any domain check:
`coordinate_to_position` turns out-of-range positions into NaN, so a shift applied after
`locator` could not recover seam targets. `locator` inverts the transform as today; for the
wrapping axis it then uses its own position function instead of `coordinate_to_position`: the
global modulo formula above, with no domain check, or, for a regional axis, the branch shift
in source coordinates followed by `coordinate_to_position` as today. Interpolation then runs
`map_coordinates` on the halo-extended block. A periodic fast path is deferred until a
measurement shows the general path is too slow.

**Labels.** Output coordinates are the target's labels unchanged. Nothing is normalized; `xr.align`
behaviour is unchanged.

### 5. Frame identity for Earth figures and the CF reader

**Reserved definition keys.** For frames used with angular charts, adapters write the figure
into `definition` under reserved keys: `figure` (`"sphere"` or `"ellipsoid"`),
`semi_major_axis` (the radius for a sphere), `inverse_flattening` (ellipsoid only; a sphere
omits it, so no non-finite value is stored, which frame metadata refuses), `length_unit`, and
optionally `datum_name`. One canonical form per figure keeps identity comparison exact. Shared
values do not make frames equivalent: identity remains the identifier, so two files from one
model share a frame only when the caller supplies it.

**CF reader** (`xarrayrf.cf`, optional extra, no pyproj): `cf.from_dataset(ds, variable, *,
frame=None)` (a `CfGeometry`: grid, frame and report) and `cf.frame_variable` (the variable
framed, data untouched) read rectilinear 1-D `lat`/`lon` (CF `units` or `standard_name`),
the earth figure from `grid_mapping` `latitude_longitude` (`earth_radius`, or
`semi_major_axis` plus `inverse_flattening`/`semi_minor_axis`), and `bounds` as intervals. It
returns the variable bound to a frame in a `geodetic` chart (a sphere when no flattening is
given). With no `grid_mapping` it refuses unless the caller passes `frame=`, since CF leaves the
figure unspecified. Projected mappings and rotated poles are stage 4.

### 6. `geotiff.crs_frame` (open decision 1: unchanged in stage 2)

Projected CRS identity stays CRS-specific in this stage. Stage 2 adds no projected chart, so a
migration would change identity for existing users without a transition to use it. The question
returns with stage 4 (pyproj), where a projected ↔ geodetic transition would make projections
of one datum charts of one frame. Recorded in the plan's open decisions.

### 7. Persistence

`CoordinateSystem` records gain an optional `axis_roles` field; Cartesian records are
unchanged. Transition classes encode as data-only records (kind, source and target frames,
figure parameters, `longitude_start`). Both are additive kinds within schema 1, as decided for
value geometry stage 2 (2026-10-11).

### Acceptance cases

1. A rectilinear global 2.5° model grid with `grid_mapping` `earth_radius`, read through the CF
   reader, resampled linearly and cubically onto a grid offset by half a cell and onto one
   labelled −180…180 from 0…357.5: values at the seam match an analytic periodic field within
   the stated tolerance, and no target near 0/360 is fill.
2. A regional grid with labels 170…190 resampled onto targets labelled −185…−175 and 175…185.
3. A duplicate-endpoint grid (0…360 inclusive) and a descending global grid (357.5…0) behave
   as global.
4. `GeodeticToCartesian` round trip on WGS 84 against pyproj (dev-only oracle), including
   heights of −10 km and 1000 km and points within 1e−9 rad of the poles; Jacobian checked by
   finite differences; at the poles the inverse returns `longitude_start`, the inverse Jacobian
   raises, scalar resampling of a pole row succeeds and component resampling there raises the
   error naming the target sample; a point within the evolute raises.
5. `SphericalToCartesian` on an ICRS (ra, dec, distance) chart against Astropy's
   `SkyCoord.cartesian` (dev-only oracle).
6. Refusals: an angular unit on a Cartesian axis (existing), a Cartesian unit on an angular axis,
   orientation on an angular axis, `spacing`/`axis_codes` with an angular axis, box methods on a
   latitude slab axis, a transition between non-equivalent frames, and a figure disagreeing with
   the frame definition.
7. Persistence round trip for an angular `CoordinateSystem` and both transitions.

### Stages

1. Chart declaration, refusals and persistence (sections 1, 2, 7).
2. Transitions (section 3) with oracles.
3. Seam-aware resampling (section 4).
4. CF reader (section 5).

Each gets a short plan reviewed before implementation, as for value geometry.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| A separate `Chart` object wrapping `CoordinateSystem` | Two objects for one declaration; frames already hold exactly one coordinate system, and Cartesian users would see a new layer |
| Representative longitude interval in the chart | Rejects valid regional grids or makes equal systems unequal; see section 1 |
| Normalizing longitude labels on read | Breaks literal alignment, which the plan keeps |
| Reading the figure from frame `definition` only | Makes the core parse open adapter metadata; explicit parameters checked against reserved keys are inspectable |
| Transitions derived inside `coordinate_system_change` | Its result type is `AffineTransform`; a nonlinear result would change its contract |
| `mode="grid-wrap"` in `map_coordinates` | Wraps every axis, including latitude and non-geometry axes |
| Seam-aware resampling deferred | Global grids, the main consumer, are unusable without it |
| A `"celestial"` representation | The celestial sphere is a spherical chart; the frame (ICRS, galactic) carries the difference |

## Deferred Work

- Projected charts, pyproj transitions, rotated pole, and the `crs_frame` identity question:
  stage 4 of the plan.
- Block-diagonal composition of a transition with pass-through axes (time, level).
- Arcminute and arcsecond units on angular axes.
- A periodic lattice fast path with seam-aware cropping.
- Box methods on 2-D longitude/latitude sources: they refuse every such source, latitude
  always, even point-sampled; section 2 reads as if longitude slabs worked. Narrow the refusal
  or document it (2026-10-17 review finding M1; maintainer: deferred).
- Masked (per-point validity) evaluation of transitions: stage 1c's internal mechanism
  (`_validity.py`) admits them by giving the transition types a `_masked` evaluation.
- Metric declaration and preservation query (plan deferred work).

## Next Steps

1. Stage 1, declarations, refusals and persistence ([plan](angular_charts_stage1_plan.md)).
   Done.
2. Stage 2, transitions ([plan](angular_charts_stage2_plan.md)). Done.
3. Stage 3, seam- and pole-aware resampling ([plan](angular_charts_stage3_plan.md)). Done.
4. Stage 4, the minimal CF reader ([plan](angular_charts_stage4_plan.md)). Done. All stages
   of this design are implemented; value geometry stage 4 (`FrameField`) is unblocked.
