# Angular charts stage 2: chart transitions

**Status:** Implemented (2026-10-12; plan and implementation reviewed by Codex gpt-6-sol)
**Last updated:** 2026-10-12
**Scope:** Core: four transition transforms between an angular chart and a Cartesian chart of
one frame, their persistence, angular reordering in `coordinate_system_change`, and the
component-resampling error path at singular Jacobians. Implements section 3 of the
[angular charts design](angular_charts_design.md) (and the transition records of section 7).
No seam handling, no CF reader.

## Context

[Stage 1](angular_charts_stage1_plan.md) declares spherical and geodetic charts and makes
`coordinate_system_change` refuse any pair that differs in representation or roles. A
lon/lat/height chart and an Earth-centred Cartesian chart of the same frame therefore have no
conversion yet. Transforms are structural protocols (`SupportsPoints`, `SupportsJacobian`,
`SupportsInverse` in `_transform.py`); `AffineTransform` (`_affine.py`) sets the idioms:
`_points` validation, read-only results, exact equality, `repr`. `resample` accepts an explicit
`transform` between equivalent frames (`_frame_map` in `_resampling.py`), so a transition can be
passed there once it exists.

## Current Decision

### Classes

New module `src/xarrayrf/_chart_transitions.py`, exported from `xarrayrf`:

| Class | Source chart | Target chart | Figure |
|---|---|---|---|
| `SphericalToCartesian` | spherical, roles longitude, latitude, radius | Cartesian, 3 axes | none |
| `CartesianToSpherical` | Cartesian, 3 axes | spherical, as above | none |
| `GeodeticToCartesian` | geodetic, roles longitude, latitude, height | Cartesian, 3 axes | `semi_major_axis`, `inverse_flattening` |
| `CartesianToGeodetic` | Cartesian, 3 axes | geodetic, as above | as above |

```python
to_ecef = rf.GeodeticToCartesian(
    geodetic_frame, ecef_frame, semi_major_axis=6378137.0, inverse_flattening=298.257223563
)
lon_lat_h = to_ecef.inverse().transform_point(points)  # a CartesianToGeodetic
```

Each implements `SupportsPoints`, `SupportsJacobian` and `SupportsInverse`; `inverse()` returns
the partner class with the same endpoints exchanged and the same parameters. All four take
`longitude_start` (keyword, default `None`, meaning −180 for a degree longitude and −π for a
radian one). It applies only where longitudes are produced (the `CartesianTo*` direction) but is
carried by both so that `inverse()` round-trips exactly; it joins equality.

### Endpoint and parameter validation (constructor)

- Source and target are `ReferenceFrame`s with `is_equivalent_frame` true (`ValueError`
  otherwise, naming both identifiers).
- The angular chart has exactly three axes with roles longitude, latitude and radius
  (spherical) or height (geodetic), in any order; the Cartesian chart has exactly three axes, read
  as `(x, y, z)` in its declared order (x toward longitude 0 on the equator, z toward the north
  pole, right-handed). A different Cartesian order composes with `coordinate_system_change`.
- The radius or height unit equals every Cartesian unit (no rescaling); `semi_major_axis` is in
  that unit. The angular units come from the chart, each with its period from stage 1.
- `semi_major_axis` finite and positive; `inverse_flattening` finite and either `0.0` (sphere)
  or greater than 1; `longitude_start` finite. `TypeError` for non-real values.
- Figure against the frame definition: when the definition has `figure`, the reserved keys must
  agree exactly (`"sphere"` with `inverse_flattening == 0.0`, `semi_major_axis`, and
  `length_unit` equal to the length unit), otherwise `ValueError` naming the disagreeing key. A
  definition without `figure` is not checked.

### Evaluation

- Points: validated like `AffineTransform._points` (finite real, trailing length 3). Results
  are new arrays; non-finite results raise.
- Forward (`*ToCartesian`): latitude outside ±90° (±π/2) or a negative radius raises
  `ValueError` naming the first offending point index. The forward domain is the inverse's
  image, so every accepted point round-trips: a geodetic height must satisfy
  `h > −N(φ)(1 − e²)` (the point stays on its surface foot's side of the equatorial plane and
  of the polar axis; for a sphere, `h > −a`), and the mapped point must lie outside the
  inverse's evolute bound below; otherwise it raises. Deeper points have another nearest foot
  (on a sphere, `h = −2a` at longitude 0 is the surface point at longitude 180°), so accepting
  them would break `inverse()`. A prototype round-trips 400,000 random points of this domain
  to 3e−14 rad and 5e−9 m.
- Exact poles: a latitude equal to ±90 (degrees) or ±`math.pi / 2` (radians) uses
  `cos φ = 0` and `sin φ = ±1` exactly, so the image lies exactly on the polar axis
  (`x = y = 0`) rather than `cos(π/2) ≈ 6e−17` off it; the inverse then returns
  `longitude_start` and the inverse Jacobian raises there.
- Inverse (`CartesianTo*`): longitude is `atan2(y, x)` shifted into
  `[longitude_start, longitude_start + period)`; on the polar axis (`x = y = 0`) it is
  `longitude_start`. Spherical: the centre raises. Geodetic: Vermeille's closed form
  (J. Geodesy 76, 2002), valid outside the evolute; points with `x² + y² + z² <=
  (e² a² / b)²` raise. For a sphere (`inverse_flattening == 0`) the bound is 0 and only the
  centre raises.
- Jacobian (`jacobian(at=...)`, `at` required, shape `(..., 3, 3)`, target axes by source
  axes): forward analytic, finite everywhere, singular at the poles and the centre (returned as
  is). Inverse: the matrix inverse of the forward Jacobian at the mapped point, computed
  analytically; on the polar axis or at the centre it raises `ValueError` naming the first
  offending point index.
- Equality: endpoints and parameters exactly; hashable; `repr` names class, endpoints and
  parameters. Slots; pickling and deep copy by value.

### `coordinate_system_change` reordering

Two angular systems with the same representation and the same multiset of roles derive a
permutation: angular axes match by role, units must be equal per matched pair, and axes with
role `None` follow today's unoriented rule (same name, unit and type at the same position).
The stage-1 guard then refuses only differing representations or role sets, and its message
names the transition classes.

### Component resampling at singular Jacobians

`apply_pointwise` in `_component_resampling.py` moves the `jacobian_map.jacobian(...)` call
inside the existing `try`, catching `ValueError` from it too. On that path it evaluates the
block's Jacobians one point at a time to find the first failing target sample, then raises the
existing error naming it. Scalar resampling is unaffected (it never asks for a Jacobian).

### Persistence

Four built-in kinds, `spherical_to_cartesian`, `cartesian_to_spherical`, `geodetic_to_cartesian`,
`cartesian_to_geodetic`, with fields `source`, `target`, `longitude_start` and, for the geodetic
pair, `semi_major_axis` and `inverse_flattening`. Additive within schema 1, like stage 1.
Composites containing them encode as today.

### Oracles and tests

New `tests/test_chart_transitions.py`:

1. WGS 84 `GeodeticToCartesian` and its inverse against `pyproj.Transformer` (EPSG:4979 to
   EPSG:4978, `always_xy`), `pytest.importorskip("pyproj")`: a grid of latitudes including ±90
   and 1e−9 rad from the poles, longitudes across ±180, heights −10 km, 0, 1000 km. Tolerances
   set from measured agreement and stated in the test (expected sub-millimetre and about 1e−11°).
2. `SphericalToCartesian` on an ICRS-like (ra, dec, distance) chart against an Astropy fixture:
   `tools/build_astropy_spherical_fixture.py` (run with `uv run --with astropy`) writes
   `tests/fixtures/astropy_spherical.json` (inputs, `SkyCoord(...).cartesian` outputs, Astropy
   version). The test needs no Astropy.
3. Round trips in degrees and radians, with lon/lat/radial roles in non-default axis orders and
   `longitude_start` of −180, 0 and 180.
4. Jacobians against central finite differences away from singularities; inverse Jacobian times
   forward Jacobian is the identity; raise at the poles and the centre, naming the point.
   Exact pole inputs in degrees and radians map to `x = y = 0`, invert to `longitude_start`,
   and the inverse Jacobian there raises.
5. Forward domain boundaries for sphere and WGS 84: heights just above `−N(φ)(1 − e²)` and
   just outside the evolute bound round-trip; heights at or below the limit, and points mapping
   inside the bound, raise.
6. Every constructor refusal in the validation section, one test each, including figure
   disagreement with each reserved key.
7. `resample` of a lon/lat/height grid onto an Earth-centred Cartesian grid through
   `CartesianToGeodetic` passed as `transform=` (scalar, linear), against the analytic field;
   component resampling through a polar target raises the error naming the sample.
8. Equality, hashing, `repr`, pickling, encoding round trips including inside a
   `CompositeTransform`; `coordinate_system_change` reordering cases and its refusals.

### Documentation

`docs/core_interface.md` (the four classes, conventions, domains, the reserved definition keys),
`docs/design.md` representation row, changelog entry, this plan's status, the design's next
steps.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Two classes with a direction flag | The inverse is a different map with a different domain; separate classes keep `repr`, encoding and refusals specific |
| Iterative geodetic inverse (Bowring) | Needs an iteration count and convergence test; Vermeille's closed form is exact outside the evolute, which is already the stated domain |
| Inverse Jacobian by numeric inversion of the forward one | Hides the polar singularity as an ill-conditioned matrix instead of raising at the axis |
| Reading the figure from the frame definition | Rejected in the design: explicit parameters, checked against reserved keys |
| Astropy as a test dependency | Heavy for one oracle; a generated fixture with its provenance suffices |

## Deferred Work

- Celestial frames with a 2-D (ra, dec) chart: an embedding into 3-D, between frames.
- Block-diagonal composition with pass-through axes (time, level).
- Seam handling (stage 3), CF reader (stage 4).

## Next Steps

Done. Implementation review added one fix: the inverse Jacobian takes its longitude scale from
the Cartesian distance to the axis, since a latitude computed just off the axis can round to
exactly ±90°. Next: the stage 3 plan (seam-aware resampling).
