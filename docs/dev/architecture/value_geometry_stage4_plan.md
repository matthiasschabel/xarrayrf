# Value geometry stage 4: pointwise bases (`FrameField`)

**Status:** Implemented (2026-10-12; plan reviewed by Codex gpt-6-sol)
**Last updated:** 2026-10-14
**Scope:** Core and binding: a `FrameField` protocol for component bases that vary by position,
one built-in field (`EastNorthUp` on longitude/latitude charts), pointwise basis maps in
`rf.change_component_basis` and in resampling of declared components, their persistence, and
pole crossing for components in a constant Cartesian basis. Implements section 4 of the
[value geometry design](value_geometry_design.md) and its Deferred Work rows "Pointwise bases"
and "Non-affine bindings … (pointwise `M_in`, `M_out`)" as far as frame fields and chart
transitions need them.

## Context

A declaration's slot basis is a `ReferenceFrame` (components along its axes) or the binding's
`ArrayCoordinates` (`_tensor_component_declaration.py`). Both give a constant basis: maps between
them are matrices, refused when non-affine (`_frame_map`, `_binding_matrix` in
`_tensor_components.py`). Resampling composes `output_map @ push(y) @ input_map` with constant
`input_map` and `output_map` (`resample_components` in `_component_resampling.py`); only the
push is pointwise, and the kernel `transform_declared_components` already accepts maps shaped
`(..., n, n)`.

Earth-system vectors are almost never in a constant basis. Wind and current come as eastward and
northward components (CF `eastward_wind`, `northward_wind`) in m/s, a basis that rotates with
position, has no meaning at the poles, and is not the chart's coordinate basis (which is per
degree, and shrinks with `cos φ`). The angular charts design (implemented 2026-10-12) supplies
the charts, the Earth-centred transitions and their Jacobians that such a basis is defined
against. Stage 3 of that design left components outside the polar cap, because east and north
reverse across a pole.

## Current Decision

### 1. `FrameField`

A structural protocol, like the transform protocols, so tetrads and other user fields need no
registration:

```python
class FrameField(Protocol):
    @property
    def base(self) -> ReferenceFrame: ...  # the chart the field is defined over
    @property
    def axes(self) -> tuple[str, ...]: ...  # component labels
    @property
    def units(self) -> tuple[str | None, ...]: ...  # one per basis vector
    def matrix(self, at: npt.ArrayLike) -> npt.NDArray[np.float64]: ...
    def inverse_matrix(self, at: npt.ArrayLike) -> npt.NDArray[np.float64]: ...
```

`matrix(at=points)`, with points in `base` coordinates shaped `(..., k)`, returns `(..., k, n)`:
column `j` is basis vector `j` in `base`'s coordinate basis, so components `u` in the field are
`M u` in the chart's coordinate basis. Square in this stage (`n = k`); `inverse_matrix` is its
inverse, and both raise a `ValueError` naming the first point where the field is undefined.
Equality and hashing are by declaration, since declarations compare. Not a `Transform`: it maps
tangent spaces at a point, with no point map, as the curved-spacetime review's gap 5 requires.

### 2. `EastNorthUp`

The built-in field: the local orthonormal frame of a spherical or geodetic chart.

| Chart | Labels | Parameters |
|---|---|---|
| geodetic (longitude, latitude, height) | `east`, `north`, `up` | `semi_major_axis`, `inverse_flattening` |
| geodetic (longitude, latitude) | `east`, `north` (on the surface, height 0) | as above |
| spherical (longitude, latitude, radius) | `east`, `north`, `up` | none |

With the chart's Jacobian into Earth-centred Cartesian factored as `J = D S` (orthonormal
columns `D` along east, north and up; `S` the diagonal scale factors `(N + h) cos φ`,
`M + h` and 1, times the angle units' radians per unit), components `u` in east/north/up give
Earth-centred components `D u = J S⁻¹ u`, so the field's matrix in the chart's coordinate basis
is `S⁻¹`. Component order is fixed by role, whatever the chart's axis order: `east`, `north` and
`up` belong to the longitude, latitude and height (or radius) roles, so the matrix is a scaled
permutation `P S_role⁻¹`, where `P` places the role-ordered components on the chart's axes. A
chart declared as (latitude, longitude, height) still reads its first field component as east. The scales are the ones the chart transitions already
compute (`_angular_jacobian`), moved to a shared function; the figure is checked against the
frame definition's reserved keys exactly as the transitions do. `units` is the length unit for
every vector (`length_unit=` for a 2-D geodetic chart, which has no length axis to take it from;
the definition's `length_unit` must agree when present).

At a pole the field is defined as its limit along the sample's meridian: the directions `D` are
continuous along a meridian, and gridded data store pole rows that way (one value per longitude
label). What does not exist there is the chart-basis matrix, since `S` has a zero scale, so
`matrix` and `inverse_matrix` raise at the poles, naming the point. Conversions that do not pass
through the chart basis are defined there (section 4). A spherical chart without a radius axis is refused (no length
scale), as is a non-angular chart. Labels can be renamed with `axes=` (default as in the table).

### 3. Declarations

`ComponentBasis` becomes `ReferenceFrame | ArrayCoordinates | FrameField`. Labels are the
field's `axes`. A field basis is checked at operation time, like `ArrayCoordinates`: its `base`
must equal the array's frame exactly (the same coordinate system too). A field over another
chart of the frame needs its own declaration; converting between charts of one frame first is
the caller's step.

### 4. Pointwise basis maps in `change_component_basis`

Every basis map is assembled at the array's sample points `x` (its geometry points in its frame
`A`) from two halves, *to `A`'s coordinate basis* and *from it*:

| Basis | To `A` at `x` |
|---|---|
| `ArrayCoordinates` (the binding source) | the binding matrix (constant; non-affine bindings stay refused) |
| `A` itself, or `A` in another coordinate system | identity, or the derived change (constant) |
| a `FrameField` with base `A` | `field.matrix(x)` |
| another frame `F`, with `transform=T` | see below |

and *from `A`* is the inverse of the corresponding row (`field.inverse_matrix(x)`).

`transform=` keeps today's meaning and direction: it maps the frame on the source basis's side
toward the frame on the target basis's side (`_frame_map`). An affine `T` between any two
frames works exactly as today. A non-affine `T` that provides a Jacobian (a chart transition) is
new; one of its endpoints must be `A`, because only `A`'s points are known. Its Jacobian is
evaluated at the sample's location in `T`'s source frame: at `x` when `T.source` is `A`
(components moving from `A`'s side to `F`), and at `T⁻¹(x)` when `T.target` is `A` (components
moving from `F` to `A`; `T` must then provide an inverse).

**East/north/up to its Earth-centred chart.** When the source basis is an `EastNorthUp` over
`A` and `T` is the matching `GeodeticToCartesian` or `SphericalToCartesian` (same figure), the
map is the orthonormal directions `D(x)` directly, not `J_T(x) · S⁻¹(x)`; the reverse direction
uses `D(x)ᵀ`. Mathematically equal away from the poles, this is also defined at the poles, so
the polar recipe (section 6) works on data with pole rows. When source and target bases are both constant, nothing is evaluated at
points and behaviour is exactly today's. A pointwise map is applied per sample by the kernel;
`check_common_map` compares maps per sample. A failing point (a pole) is reported by sample, as
in resampling.

Typical uses:

```python
wind = wind.rf.declare_tensor_components(VECTOR, component_dims=("component",), bases=enu)
earth_centred = wind.rf.change_component_basis(ecef, transform=to_ecef)  # metres to metres
chart = wind.rf.change_component_basis(wind.rf.reference_frame, units="degree s-1")
```

### 5. Resampling

The input map is evaluated at the source-frame points `T(y)` of the target samples, the output
map at the target-frame points `y`, so `output_map(y) @ push(y) @ input_map(T(y))` per block,
taking the pointwise path whenever any of the three varies. `T(y)` comes from the frame map
already applied for location. Interpolation stays in the source basis (design section 3).

Shortcut: when the source and output bases are the same basis and the source and target frames
are the same frame (no frame map), every map is the identity and nothing is evaluated at points.
This is exact, not an approximation that skips validation: an `EastNorthUp` field is defined at
the poles (its meridian limit, section 2), so a same-frame regrid of east/north components onto
pole rows is defined, while the chart-basis detour would pass through `S⁻¹`, which is not.

### 6. Poles

Stage 3's `pole_halos=components is None` widens to: no components, or every slot's basis is a
`ReferenceFrame` whose representation is Cartesian. Components in a constant Cartesian basis
(Earth-centred x, y, z) are continuous across a pole, so the mirrored rows are correct as they
stand. East/north (and chart-basis) components still get no pole halo: they reverse across the
pole. The documented recipe for polar caps is to change to the Earth-centred basis, resample, and
change back, which section 4 makes a two-line operation; an automatic sign flip in the halo for
`EastNorthUp` was deferred and is now implemented ([plan](east_north_pole_halo_plan.md)).

### 7. Persistence

`EastNorthUp` encodes as a built-in kind (`east_north_up`: base frame, axes, figure parameters,
`length_unit`), additive within schema 1; declaration records encode each basis through
`_encode`, and their decoder accepts a field where it accepts frames and array coordinates today.
User `FrameField`s are not persisted in this stage: the extension mechanism is for point
transforms (it requires `SupportsPoints` and transform endpoints), so `encode` refuses a
declaration with a user field, naming it. A field-extension record is deferred.

### Tests (`tests/test_frame_fields.py`, plus additions to the component tests)

1. `EastNorthUp` matrices: `J(x) @ M(x)` has orthonormal columns pointing east, north and up
   (checked against `GeodeticToCartesian` and `SphericalToCartesian` Jacobians at random points,
   3-D, degree and radian charts; a 2-D field against the first two tangent columns of the
   matching 3-D chart at height 0); with the chart in (latitude, longitude, height) order the
   first component is still east; the inverse matches; poles raise naming the point; refusals (2-D spherical, Cartesian chart, figure disagreeing with the
   definition, missing `length_unit` on a 2-D chart).
2. `change_component_basis`: east/north/up to the chart basis and back (round trip); to
   Earth-centred Cartesian through `GeodeticToCartesian` and back; a solid-body rotation wind
   (`v = ω × r`) whose east component is `ω (N + h) cos φ` and north component 0 maps to the
   analytic Earth-centred vector, including on pole rows (the direct `D` map); the reverse
   direction through `T⁻¹(x)`; today's constant `transform=` direction unchanged; units required
   exactly when they change; a field whose base is not the array's frame is refused; a
   chart-basis conversion at a pole sample raises naming the sample.
3. Resampling: east/north components between two lon/lat grids of one frame (identity
   shortcut, across the seam); onto a grid in another chart through a transition; output in an
   `EastNorthUp` of the target; Earth-centred components cross the poles and match the analytic
   solid-body field in the polar cap; east/north components there stay outside.
4. A user `FrameField` (a constant rotation written as a protocol implementation) works through
   both operations.
5. Persistence: declarations with `EastNorthUp` round-trip; a declaration with a user field is
   refused by `encode`, naming it.
6. Existing constant-basis tests unchanged (no point evaluation when both bases are constant).

### Documentation

`docs/core_interface.md` (tensor transformation and resampling sections: `FrameField`,
`EastNorthUp`, pointwise maps, the polar recipe), `CHANGELOG.md`, the value geometry design's
section 4 and Deferred Work, this plan's status, the dev changelog.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| `FrameField` as a `Transform` | It has no point map; making it one would need the exception to `check_transform` that the design rejected |
| East/north as an `ArrayCoordinates` basis with scaling | `ArrayCoordinates` is per unit of coordinate value by definition; east/north is per metre and orthonormal |
| A matrix field into Earth-centred Cartesian rather than the chart basis | Ties every field to an embedding; the chart basis is what the binding and resampling already speak, and the Earth-centred form is one `transform=` away |
| Automatic pole sign flips for east/north | Correct only for this field; the Earth-centred route is exact for any constant Cartesian basis and composes with section 4 |
| Constant-only maps with a precomputed per-sample matrix array as a basis | Loses the declaration's meaning (east/north) and its persistence |

## Deferred Work

- Pole halos with sign flips for `EastNorthUp` (a fiber twist, as in xarraybc): implemented,
  [plan](east_north_pole_halo_plan.md).
- Non-square fields (2-D east/north into a 3-D embedding) and spherical charts without radius.
- Persisting user `FrameField`s (a field-extension record).
- Fields over another chart of the array's frame (composition with the coordinate change).
- Non-affine bindings (`ArrayCoordinates` bases under field-backed transforms): with nonlinear
  plan stage 1.
- Tetrad and metric helpers for curved spacetimes (the protocol admits user tetrads now).

## Next Steps

1. Plan review (Codex), implement, implementation review, QA, commit and push.
