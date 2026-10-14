# East/north pole halos

**Status:** Implemented (2026-10-14; plan reviewed by Codex gpt-6-astra)
**Last updated:** 2026-10-14
**Scope:** Resampling of declared tensor components in an `EastNorthUp` basis across a pole of a
global longitude/latitude source. Closes the deferred row "Pole halos with sign flips for
`EastNorthUp`" of [value geometry stage 4](value_geometry_stage4_plan.md) and the matching row of
the [value geometry design](value_geometry_design.md).

## Context

Angular charts stage 3 ([plan](angular_charts_stage3_plan.md), "Poles") extends a global,
antipodal source across a pole: halo rows beyond the outermost latitude row hold the mirrored
rows of the antipodal meridian (`PeriodicAxes.extend` in `_periodic.py`). Scalars are continuous
there. Value geometry stage 4 widened the gate (`pole_halos=` in `_resample.py`) to components
whose every slot basis is a constant Cartesian `ReferenceFrame`. Components in an `EastNorthUp`
basis still get no halo, so targets in the polar cap (between the outermost row and the pole)
are outside, and the documented route is Earth-centred: change basis, resample, change back.

Across a pole, east and north reverse. Continuing a meridian `λ` over the pole onto `λ + 180°`,
the frame that varies smoothly along the path is `(east, north, up)` on the near side and
`(−east, −north, up)` of the far side. So a halo row taken from the antipodal meridian is correct
in the near side's continued frame once its east and north components are negated. Derivation
through the chart: the mirror `m(λ, φ) = (λ + 180°, 180° − φ)` has Jacobian `diag(1, −1, 1)`; on
the extended chart `cos φ` changes sign, so the east scale `(N + h) cos φ` does too, and
`S_ext m_* S⁻¹ = diag(−1, −1, 1)`.

The stage 4 plan rejected automatic flips as "correct only for this field". That holds: the rule
belongs to the field, so only the built-in `EastNorthUp` gets it, not user `FrameField`s.

## Current Decision

### 1. Which declarations get a pole halo

A pole halo is built for declared components when every slot's basis is either

- a `ReferenceFrame` with a Cartesian representation (sign `+1` for every component; today's
  rule), or
- an `EastNorthUp` (whose `base` is the source frame, as stage 4 already requires): sign `−1`
  for `east` and `north`, `+1` for `up`. Component order is fixed by role (stage 4, section 2),
  so the signs are `(−1, −1, +1)` for a 3-D field and `(−1, −1)` for a 2-D one.

Other bases (the chart's coordinate basis, `ArrayCoordinates`, user `FrameField`s) get no halo;
polar-cap targets stay outside, as now. The chart basis is excluded on purpose: its longitude
component is per unit of longitude and diverges like `1 / cos φ` at the pole, so interpolating it
across the pole is meaningless whatever the sign.

### 2. Sign of each stored component

Each slot's sign vector is diagonal, so the halo transform is elementwise. A stored component
entry gets the product of its slots' signs: for an unpacked rank-`r` tensor the outer product of
the slot sign vectors over its component dims; for a packed symmetric declaration, entry `(i, j)`
gets `s_i s_j` through the packing's entry order. The map `diag(±1)` has determinant `+1` in both
the 2-D and 3-D cases (a half-turn about `up`), so pseudo-tensor and density weights contribute
no factor; the plan states this rather than calling the kernel. Slots may differ: an ordinary tensor may
have an `EastNorthUp` slot and a Cartesian slot (stage 4 refuses separate maps only for
pseudotensors and densities), so the helper always forms the per-slot outer product rather than
reusing one slot's signs. In practice such a declaration is refused later by the component
maps (a Cartesian slot has no constant map to a geodetic chart), so the mixed case is not
observable through `resample`; the helper stays general because it costs nothing. A declaration
whose signs are all `+1` takes today's path unchanged.

A private helper on the component side (`_component_resampling.py`) returns the sign array over
the declaration's component dims, or `None` when a slot's basis has no pole rule. `_resample`
uses it to decide `pole_halos` (no components, or a sign array exists) and passes the signs on
when any is `−1`. `EastNorthUp` exposes its per-component signs privately (`_pole_signs`); the
`FrameField` protocol is unchanged.

### 3. Applying the signs

Signs vary over component dims, which are non-geometry ("leading") dims of `execute`, and with
Dask a task sees one chunk of them. So the signs travel as data: `_resample` passes a second
`apply_ufunc` input, a ±1 `DataArray` over the component dims (broadcast by xarray, chunk-aligned
by Dask), and `execute(plan, values, signs)` takes it broadcast to the leading shape.
`PeriodicAxes.extend(volume, ..., sign)` multiplies only the mirrored halo rows (`before` and
`after`) by the slice's sign; longitude halo columns and interior rows are untouched. Without
signs (`None`), behaviour and cost are today's.

The sign input is omitted when the plan has no target samples (`total == 0`, where rechunking is
disabled and no halo is built), and for a Dask source it is created with the source's chunks along
the component dims, so the two inputs never need rechunking to align.

Negation happens in the dtype the slices are interpolated in. Floating and complex values negate
exactly. Signed integers (only `method="nearest"` keeps an integer dtype) negate exactly except
the minimum value, which is refused by name if present in a negated halo row. Unsigned and
boolean components with a `−1` sign are refused up front: east/north in such a dtype cannot be
negated across the pole.

### 4. Accuracy statement

Interpolation stays in the source basis (design section 3), so in the cap it mixes east/north
components of neighbouring meridians exactly as it already does everywhere else on the grid. For
fields smooth in east/north (solid-body rotation about the polar axis) the cap is as accurate as
the interior. For flow across the pole, east/north components vary like `sin λ`, `cos λ` near
the pole, and the Earth-centred route remains the more accurate one; the docs say so instead of
calling it the only route.

### Tests (`tests/test_frame_fields.py` and the angular resampling tests)

Fixture: a constant Earth-centred vector `V = (1, 0.3, 0.5)` expressed in east/north/up at each
sample (`D(x)ᵀ V`). It is smooth everywhere, its east and north components vary like `sin λ`,
`cos λ` near the pole (the hard case), and its `up` component is about 0.5 there, so every sign
is observable. Tolerances are the linear-interpolation bound `|V| (Δλ² + Δφ²) / 8` (radians) for
the grid step, or the data's precision where the check is exact.

1. **Signs are applied.** A cell-centred global grid (pole midway between the last row and its
   mirror) with the fixture. Linear resampling onto a target sample at the pole on meridian `λ` returns the
   mean of `u(λ, b)` and `diag(−1, −1, 1) u(λ + 180°, b)` to roundoff; without the flip the
   east/north mean would be near zero, so this discriminates.
2. **Cap accuracy.** Targets inside both caps (pole-sample and cell-centred grids, linear and
   cubic) match the analytic east/north/up within the bound; a polar-axis rotation (smooth in
   east/north) matches as tightly as interior targets.
3. **Rank 2 and packing.** `V ⊗ V` in east/north/up, unpacked and packed symmetric with a
   reordered entry order and a non-unit off-diagonal scale: signs are `+1` on east–east,
   east–north, north–north, up–up and `−1` on east–up and north–up (each about 0.5 near the pole,
   so a missing flip fails); cap values match within the bound. A pseudovector variant of the
   fixture shows no determinant factor is applied. (A rank-2 tensor with one `EastNorthUp` and
   one Cartesian slot is refused by the component maps, so it is not tested here.)
4. **2-D geodetic field** on a (longitude, latitude) chart: signs `(−1, −1)`.
5. **Dask:** the same result with the component dim chunked one per chunk and with an extra
   chunked non-geometry dim (time); an empty target with split component chunks still returns
   an empty result.
6. **Unchanged:** chart-basis and user-`FrameField` components keep cap targets outside;
   Earth-centred components unchanged; scalars unchanged.
7. **Refusals:** unsigned-integer east/north with `method="nearest"` is refused, naming the
   dtype; a negated signed-integer minimum is refused.
8. The stage 4 test asserting "east/north components there stay outside" is updated to the new
   behaviour, and the Earth-centred recipe test is kept.

### Documentation

`docs/core_interface.md` (resampling of declared components: the polar cap and the two routes),
`CHANGELOG.md`, the dev changelog, the value geometry design's Deferred Work row and the stage 4
plan's Deferred Work and section 6 (pointing here), this plan's status.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Rotate halo rows to the near side's frame by the full pointwise map | The continued frame is exactly `diag(−1, −1, 1)` of the far frame; a pointwise map adds cost for the same numbers |
| A pole rule on the `FrameField` protocol | No second field needs it yet; a protocol member is public API to keep |
| Halos for chart-basis components with the Jacobian `diag(1, −1, 1)` | The longitude component diverges at the pole, so cap values would be meaningless |
| Signs as a plan constant indexed by global leading position | Wrong under Dask, where a task sees a chunk of the component dims |
| Leave the Earth-centred recipe as the only route | Common east/north data (CF wind) would have holes in the cap for every same-frame regrid |

## Deferred Work

- A pole rule for user `FrameField`s (a protocol member) once a second field needs one.
- Tetrads and other built-in fields over spherical charts.

## Next Steps

None; implemented with the tests above in `tests/test_frame_fields.py`. A mutation check
(halo signs disabled) fails every new test.
