# Angular charts stage 1: declarations, refusals and persistence

**Status:** Implemented (2026-10-12; plan and implementation reviewed by Codex gpt-6-sol)
**Last updated:** 2026-10-12
**Scope:** Core only: angular representations and `axis_roles` on `CoordinateSystem`, the
refusals of Euclidean calculations on angular axes, and persistence. Implements sections 1, 2
and 7 (the `CoordinateSystem` part) of the [angular charts design](angular_charts_design.md).
No transitions, no seam handling, no CF reader.

## Context

`CoordinateSystem` (`src/xarrayrf/_coordinate_system.py`) accepts only `"cartesian"` and calls
`check_units(..., reject_angular=True)` on every axis. `Lattice.spacing` and `.direction`
(`_lattice.py`) take Euclidean norms whenever all axes share a unit, so `("deg", "deg")` would
pass. Box methods (`_box_operator` in `_resampling.py`) average over slab intervals in
coordinate units. Encoding (`_encoding.py`) writes a fixed field set per kind and `_expect`
requires exactly that set.

## Current Decision

### API

```python
rf.CoordinateSystem(
    ("time", "lat", "lon"),
    ("days since 1850-01-01", "degrees_north", "degrees_east"),
    representation=rf.SPHERICAL,  # or rf.GEODETIC
    axis_roles=(None, "latitude", "longitude"),
)
```

- New constants `SPHERICAL = "spherical"`, `GEODETIC = "geodetic"`, exported beside
  `CARTESIAN`; `AxisRole = Literal["longitude", "latitude", "radius", "height"]`.
- `axis_roles`: keyword, default `None`. Property `axis_roles` returns one entry per axis
  (all `None` for Cartesian).

### Validation (in `CoordinateSystem.__init__`)

| Condition | Error |
|---|---|
| `axis_roles` given with `"cartesian"` | `ValueError` |
| non-Cartesian without `axis_roles` | `ValueError` |
| `axis_roles` not a non-string sequence of the four roles or `None`, or wrong length | `TypeError` / `ValueError` |
| not exactly one `longitude` and one `latitude` | `ValueError` |
| `radius` in a geodetic system, `height` in a spherical one, or more than one of either | `ValueError` |
| an angular role with a unit outside the accepted set below, or `None` | `ValueError` |
| any orientation token in a non-Cartesian system | `ValueError` |

Accepted angular units, kept in `_validation.py` beside `ANGULAR_UNITS`, with the period each
implies (exposed internally as `angular_period(unit) -> float`, used by later stages):

- longitude: `deg`, `degree`, `degrees`, and the CF east spellings (`degrees_east`,
  `degree_east`, `degrees_E`, `degree_E`, `degreesE`, `degreeE`); period 360;
- latitude: the same plain degree spellings and the CF north spellings; period 360;
- either: `rad`, `radian`, `radians`; period 2π.

West and south spellings are refused: they flip the axis sign, and the core never converts.
Arcminutes, arcseconds and the degree sign are refused on angular axes (deferred). Axes with
role `None` keep today's rule (`reject_angular=True`), as do radius and height axes.

Orientation is refused on every axis of a non-Cartesian system, not only the angular ones:
a direction vocabulary names constant directions, and a "height up" token would invite
`axis_codes` to mix it with angles. Revisit with a consumer.

Equality, hashing and `repr` include `axis_roles`; Cartesian `repr` is unchanged.

### Refusals of Euclidean calculations

- `Lattice._common_unit` (used by `spacing` and `direction`) raises first when the frame's
  representation is not Cartesian: "spacing is a Euclidean length; frame axes ('lat', 'lon')
  are angular, so it has none. Use matrix, whose columns are the steps axis by axis; area and
  distance on a sphere need a dedicated tool." `matrix`, `origin`, `affine` and
  `transform_point` stay available.
- `CoordinateSystem.axis_codes` on a non-Cartesian system raises a message naming the
  representation (it would already raise for having no oriented axis; the message changes).
- Box methods: in `_box_operator`, after the separability check, if the source frame is
  non-Cartesian and the column of `source.transform.matrix` for any non-pass-through source
  axis has a significant entry in the latitude row, raise `ValueError` naming the axis: a mean
  over latitude degrees is not an area mean. Longitude slab axes stay allowed. Pass-through
  axes (exact index matches) are unaffected.
- `coordinate_system_change` matches unoriented axes by name, unit and position and ignores
  representation, so spherical and geodetic systems with identical `("deg", "deg")` axes, or a
  role-swapped pair, would derive an identity, which `resample` then applies. It gains a guard:
  when either system is non-Cartesian, the representations and `axis_roles` must be equal,
  otherwise `ValueError` (stage 2 adds the transitions and angular reordering). Cartesian
  systems cannot carry angular units, so a Cartesian/angular pair is refused by the guard too.
  Tests cover spherical/geodetic, role-swapped and Cartesian/angular pairs, and equal angular
  systems giving the identity.
- `affine_class` and `is_coincident` are unchanged.

### Persistence

The `coordinate_system` record gains an `axis_roles` field written only for non-Cartesian
systems, so Cartesian records stay byte-identical and existing fixtures pass unchanged. The
reader accepts the record with or without it (`_expect` against either set); a non-Cartesian
record without it fails construction and surfaces as `MalformedDataError`. Schema version
stays 1, as for the value geometry additions; older readers refuse the new field with their
existing "exactly the fields" error.

### Documentation

- `docs/core_interface.md`: `CoordinateSystem` signature, the role and unit table, the
  refusals; `SPHERICAL`, `GEODETIC` in the constants list.
- `docs/design.md` §coordinate representation: "Implemented: Cartesian, spherical and geodetic
  declarations; transitions and seam handling follow."
- `docs/dev/changelog.md` entry.

### Tests

`tests/test_coordinate_system.py` (construction, validation table row by row, equality and
hashing including roles, `repr`, `axis_codes` refusal), `tests/test_lattice.py` (`spacing` and
`direction` refuse on a lon/lat frame; `matrix` works), `tests/test_box_resample.py` (latitude
slab refused; longitude slab and pass-through latitude allowed), `tests/test_orientation.py`
(the guard's refusals and the equal-system identity), `tests/test_encoding.py` (round trip of spherical and
geodetic systems and of a frame using one; Cartesian record has no `axis_roles` key; a
non-Cartesian record missing it is `MalformedDataError`). Plus a test that a `Grid` or
`Geometry` over a global 2.5° lon/lat frame constructs and resamples between two lon/lat grids
inside the labels with a point method (no wrapping yet; targets outside the labels are fill).

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Orientation allowed on non-angular axes of an angular system | No consumer; mixing direction tokens with angles invites wrong `axis_codes` answers |
| Accept west/south CF spellings and negate | The core never converts values; the CF reader (stage 4) normalizes |
| Always write `axis_roles` | Changes every existing Cartesian record and fixture for no reader benefit |
| Refuse all box methods on angular frames | Longitude slabs are arc-length means at fixed latitude; refusing them loses a valid case |

## Deferred Work

- Transitions and the `coordinate_system_change` reordering case: stage 2.
- Seam handling: stage 3. CF reader: stage 4.
- Arcminute, arcsecond and degree-sign units on angular axes.

## Next Steps

Done. Next: the stage 2 plan (transitions), per the
[angular charts design](angular_charts_design.md).
