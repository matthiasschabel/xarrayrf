# Value geometry: vectors, tensors, pseudovectors and densities

**Status:** Active (design; not started)
**Last updated:** 2026-10-11
**Scope:** Core: a tensor type and transform law for basis-dependent values, a component
index carrying it, passive basis change, resampling of component variables and persistence.
Supersedes the quantity-rule and component-binding sections of
[the core model design](core_model_design.md), gaps 4 and 5 of
[the curved-spacetime review](curved_spacetime_review.md), and the open ordering question in
[the nonlinear geometry plan](nonlinear_geometry_plan.md) ("Tensors and frame fields").

## Context

xarrayrf transforms positions, never values. That is right for scalar fields and wrong for any
field whose values have a direction: wind and ocean currents, electromagnetic fields, stress,
strain and diffusion tensors, flow velocities, angular momentum. Re-expressing such a field in
a rotated or reflected frame, or resampling it through a registration, without transforming
its components returns wrong-direction values and no error. Meteorology's grid-relative versus
earth-relative wind flag in GRIB is the long-standing example.

Before this note the pieces were scattered and partly inconsistent: five quantity-rule
callables specified but unimplemented, a "component binding" with room reserved and no
behaviour, per-index placement and `FrameField` proposed for spacetime, and the
napari compositional model's `FieldGeometry`/`convert_field_basis`, which covers only passive
change between orthonormal bases ([comparison notes](../napari-3783/compositional_model_comparison_notes.md)).
Nothing covered parity (pseudovectors), densities or the carrier.

The library is domain-neutral: geoscience, continuum mechanics, electromagnetism, relativity
and medical imaging are equal acceptance cases, and domain reorientation policies (diffusion
PPD) stay downstream.

## Current Decision

The math (cross-product and two-form pseudovector laws, symmetric-packing restriction) was
checked numerically. The design was refined with a Codex reviewer (gpt-6-astra, reasoning
effort high) over two passes: five blocking findings and one introduced in revision, all
accepted; no open disagreements.

### 1. Tensor type and one transform law

A `TensorType` value object declares, per variable:

- `index_variance`: tuple of `"contravariant"` (transforms with `J`) or `"covariant"`
  (with `J^-T`), one per component axis, in order (empty = scalar);
- `reflection_parity`: `"even"` or `"odd"` (odd changes sign under orientation reversal);
- `determinant_exponent`: real `w`, the exponent of `|det J|`;
- `index_symmetry`: `None`, `"symmetric"` or `"antisymmetric"`, under exchange of the two indices: order 2 only, and only for two
  slots of equal variance in the same basis (mixed `J T J^-1` does not preserve matrix
  symmetry: shear maps `diag(2,1)` to `[[2,-1],[0,1]]`).

With `J` the components of the linear map from the source basis to the target basis
(`J[a, b] = d target_a / d source_b`):

    T' = sign(det J)^[parity odd] * |det J|^w
         * (J on each contravariant slot) (J^-T on each covariant slot) T

The existing named rules become shortcuts: `VECTOR = (contravariant,)`,
`COVECTOR = (covariant,)`, `SECOND_ORDER_CONTRAVARIANT_TENSOR = (contravariant, contravariant)`; mass
density = scalar with `w = -1`.

**Pseudovectors are not one law under non-orthogonal maps.** Using
`(Mx) x (My) = det(M) M^-T (x x y)`: a cross product of two vectors (torque, angular momentum)
transforms as `det(J) J^-T b` = `(covariant,)`, odd, `w = 1`; the axial dual of a 2-form (magnetic
flux density as a flux form) transforms as `det(J)^-1 J b` = `(contravariant,)`, odd, `w = -1`. Both
reduce to `sign(det R) R b` for orthogonal `R`. So the `ORTHOGONAL_PSEUDOVECTOR` shortcut is defined only
for orthogonal `J` (refusing otherwise, with a message naming the two explicit declarations),
and general maps require the explicit type. The two origins have their own shortcuts,
`CROSS_PRODUCT_PSEUDOVECTOR` and `TWO_FORM_DUAL_PSEUDOVECTOR`, which are plain types, not
persisted categories. A pseudoscalar with `w = 0` is `sign(det J)` for any invertible map.

`polar_orthogonal_factor` is not a type: it is a choice of which linear map to apply (the
orthogonal factor `U` of `J`'s polar decomposition `J = U P`, the finite-strain reorientation;
improper when `J` is). The kernel therefore takes
`(values, linear_map, tensor_type)`; callers or policies choose `J`, `polar(J)` or a domain
policy (PPD, downstream).

Kernel: `transform_tensor_components(values, linear_map, tensor_type, *, component_axes)` in the
NumPy-only geometry core, vectorised over leading axes and over pointwise maps
(`linear_map` shape `(..., n, n)`), chunk-friendly. Refuses singular maps where a covariant slot or
nonzero `w` needs `det`/inverse.

### 2. Where components live: a component index, not attrs

Attrs are not a carrier: the binding design rejected them because arithmetic keeps the first
operand's attrs, so a vector plus a covector would keep a valid-looking wrong type, and
`resample` copies attrs today. Component metadata therefore uses the binding's mechanism.

- **Order >= 1: a component index (`TensorComponentIndex`).** Each slot is a named component dimension with a label
  coordinate; one custom xarray index spans the variable's component coordinates (one per
  slot, or one packed coordinate for two symmetric slots) and declares, per slot, its
  variance and basis, plus `reflection_parity`, `determinant_exponent`, `requires_orthogonal_map`, `index_symmetry` and
  `packing_convention`. Slots are named, so
  transpose cannot exchange variance. Equality includes the whole declaration, so aligning or
  combining two variables with different declarations over the same dimensions refuses
  (as conflicting bindings do, including the binding's `join` compatibility and
  `check_override` refusals, not equality alone), and dropping or resetting the index leaves plain labels with
  no claimed type. In a Dataset, variables sharing component dimensions share their type
  (the shared-grid rule applied to components); velocity and vorticity use different
  component dimensions.
- **Bases.** A slot's basis is a `ReferenceFrame` (components along its axes, labels equal to
  its axis names in order) or the binding's `ArrayCoordinates` (components per unit of each
  source coordinate value, not per sample step: coordinates `[0, 2, 4]` under an identity
  binding have unit basis vectors, and striding leaves the declaration true; conversion uses
  the binding transform's Jacobian). Sample-position bases are not offered.
- **Order 0 (scalar densities, pseudoscalars): per-call declaration.** No dimension exists to
  hold an index, and an attr would be advisory only, so no carrier is claimed. Operations
  that would change their values (`change_component_basis` with `det J != 1` or improper `J`, `resample`
  through a volume-changing or orientation-reversing map) take `tensor_type=` explicitly; an
  undeclared scalar is an ordinary scalar. Point sampling with the `|det J|^w` factor is
  distinct from conservative remapping, which is deferred: the box methods average over
  covered target support and over overlapping sources, and mapping intervals applies no
  determinant factor, so they give support integration but no conservation guarantee.
- **Symmetric packing**: a packed component coordinate with labels `xx, yy, zz, xy, xz, yz` and
  a declared `packing_convention`, `"voigt"` (plain) or `"mandel"` (off-diagonals scaled by sqrt 2).
  Unpack, transform, repack; never transform packed components directly.
- **Units.** `J` includes unit conversion (affine basis vectors already do). A basis change
  whose basis units differ rewrites the variable's `units` only when the caller supplies the
  output unit; otherwise refuses.

User-facing operation: `da.rf.change_component_basis(basis)`: passive change of basis at the same sample
locations; constant for affine-related frames, pointwise for array-coordinate bases under a
nonlinear binding.

**Guard from stage 2.** Once a component index exists, `resample` and `rf.resample_to` refuse
arrays carrying one until stage 3 defines their law, so no stage ships typed arrays that
existing resampling would treat as scalars.

### 3. Resampling

`resample(source, target, transform=T)` maps target-frame points `y` to source-frame points
`T(y)`. For a variable whose components are in source basis `S`, with requested output basis
`O` (`output_component_basis=`, required), the result at `y` is: interpolate the source components at
`T(y)` in `S`; convert `S` to source-frame components (constant, or the source binding's
Jacobian at `T(y)` for an array-coordinate basis); push to the target frame with
`L = (D T(y))^-1`, or its polar rotation; convert to `O`. All four maps are applied per slot
by the law. `component_reorientation=` is required: `"jacobian"`, `"polar_orthogonal_factor"` (only when source and
target frames and both bases are orthonormal Cartesian with equal units), or a callable
supplied by a domain policy. For affine `T` and constant bases, interpolating before or after
the conversion agrees for linear interpolation; with a varying `S` (array-coordinate basis
under a nonlinear binding) interpolation happens in `S` as stated, and the alternative is not
offered. Interpolation of constrained values (SPD tensors log-Euclidean, rotations by slerp)
is a policy hook, not core; linear interpolation of SPD components stays SPD, cubic does not.

### 4. Pointwise bases

Components in a basis that varies per point (east/north wind on a sphere, a tetrad) need a
`FrameField` (base chart, component bases, pointwise matrices; not a `Transform`). Deferred to
after angular charts (nonlinear plan stage 2); the kernel already accepts pointwise maps.

### 5. Persistence

One new schema kind: the component index declaration (per-slot variance and basis,
`reflection_parity`, `determinant_exponent`, `requires_orthogonal_map`, `index_symmetry`,
`packing_convention`), encoded like the binding; dropping
`requires_orthogonal_map` would decode `ORTHOGONAL_PSEUDOVECTOR` as an unrestricted odd vector. Order-0 declarations are call
arguments and do not persist. Custom reorientation callables do not persist. Subject to the same
schema-1 freeze rule as transform edge semantics: the freeze records whether additive kinds may
land after it.

### Out of scope

Spinors, spherical-harmonic coefficients (Wigner-D), quaternion orientation fields, general
order > 2 symmetry classes. A domain adapter may supply its own rule through the kernel's
`linear_map` policy hook, but not a new persisted type in the first stages.

### Acceptance cases

Stage 1: the shear oracles in `prior_art_coverage_study.md` (covariant `(1,-1,0)`, which
matches ITK; `J T J^T` = `[[5,2,0],[2,2,0],[0,0,1]]`, which deliberately does not match ITK's
`diag(3,2,1)`); rigid `second_rank_tensor` equals the rotation case; under the reflection
`diag(-1,1,1)` a vector `(1,1,1)` becomes `(-1,1,1)` and a pseudovector `(1,1,1)` becomes
`(1,-1,-1)`; `(Ju) x (Jv)` equals the declared cross-product type and `(J^-T F J^-1)`'s dual
equals the two-form type under a random shear; a scalar density at `w = -1` scales by
`1/|det J|`; mixed-variance symmetric packing refuses; Lorentz boost of `F^{mu nu}` (relativity
notes). Stage 2: component-index alignment refuses mixed declarations; transpose keeps slots;
`resample` refuses typed arrays; encode/decode round trip. Stage 3: DTI rotation with
`"polar_orthogonal_factor"`; a non-unit source basis under identity registration (factor carried through).
Stage 4: GRIB grid-relative to earth-relative wind.

### Stages

1. `TensorType`, the law and the NumPy kernel with shortcuts; tests above that need no
   binding. No binding or `resample` change.
2. Component index, `rf.change_component_basis`, the `resample` guard, persistence; an operation probe
   for the component index on the same lanes as the binding inventory.
3. `resample` integration with `output_component_basis=` and `component_reorientation=`.
4. Pointwise bases with angular charts.

## Alternatives Considered

- **Naming the type `QuantityType`.** Rejected: pint's and astropy's `Quantity` (and
  xarrayrf's own `units` module) mean a value with units, so the name suggests dimensional
  analysis. "A tensor of type (p, q)" is the standard term; densities and pseudotensors are
  tensor types with a determinant power and odd parity. The per-slot field is `index_variance`
  with values `"contravariant"` / `"covariant"`: bare `variance` reads as the statistical
  concept, `indices` would collide with the component index that carries the type, and
  the physicists' `"up"` / `"down"` are shorthand that needs the textbook. `determinant_exponent`
  is used rather than "weight", whose sign convention varies by author. Generally, names
  favour descriptive, technically grounded forms over terse ones: `reflection_parity` (physical
  parity is behaviour under reflection), `index_symmetry`, `order` rather than `rank` (which
  collides with matrix rank), `polar_orthogonal_factor` rather than `polar_rotation` (it is a
  reflection for improper `J`), `change_component_basis` rather than `express_in`.
- **Attrs as the type carrier.** Rejected for the reason the binding design rejected them:
  arithmetic keeps the first operand's attrs, so a vector plus a covector keeps a
  valid-looking wrong type, and `resample` copies attrs. Order >= 1 uses an index instead.
- **One `pseudovector` law for all maps.** Rejected: a cross product of vectors and the dual
  of a two-form agree only for orthogonal maps. The shortcut refuses non-orthogonal maps.
- **Origin categories (`cross_product`, `two_form`) as persisted types.** Rejected: they are
  constructors returning plain types; persisting them would add vocabulary the law does not
  need.
- **A tensor-density "weight".** Rejected for `determinant_exponent`, an explicit exponent of `|det J|`,
  because the literature's weight sign conventions conflict.
- **`polar_orthogonal_factor` as a tensor type.** Rejected: it selects which linear map to apply
  (finite-strain reorientation), so it is a reorientation policy beside `"jacobian"`.
- **Diffusion PPD or another fixed tensor-reorientation strategy in core.** Rejected, as
  before: competing strategies are domain policy, supplied through `component_reorientation=`.
- **Passive orthonormal basis change only** (the napari model's scope). Rejected as the whole
  design: registrations need general `J`, covariant slots, parity and densities.
- **Interpolating after converting components** for varying source bases. Not offered; one
  order (interpolate in the source basis, then convert) is defined.
- **Conservative density transfer through the existing box methods.** Rejected: they average
  over covered support and overlapping sources and apply no determinant factor, so they
  conserve nothing in general.
- **Sample-position bases** (components per sample step). Not offered: they would change
  meaning under striding and reversal.

## Deferred Work

- Conservative remapping of densities, with explicit measure handling and coverage and overlap
  constraints.
- Pointwise bases (`FrameField`), after angular charts.
- Order > 2 symmetry classes; spinors, spherical-harmonic coefficients and quaternion fields
  (out of scope above).
- Persisting custom reorientation policies.

## Next Steps

1. Stage 1: `TensorType`, the law, the NumPy kernel and shortcuts, with the stage-1
   acceptance cases as tests. No binding change, so it does not wait on the schema freeze.
   Planned and reviewed in [the stage 1 plan](value_geometry_stage1_plan.md).
2. Before stage 2: the schema-1 freeze decision records whether additive record kinds may land
   after it; stage 2 then adds the component index, `rf.change_component_basis`, the `resample` guard,
   persistence and the operation probe.
3. Stage 3 after stage 2's probe passes on the patched lane.
