# Value geometry stage 3: resampling declared tensor components

**Status:** Implemented
**Last updated:** 2026-10-12
**Scope:** The xarray binding layer: `resample` and `rf.resample_to` for sources that declare
tensor components, replacing stage 2's refusal. Small additions to the planner
(`_resampling.py`) to expose the target-to-source frame map and target sample points.
Implements section 3 of [the value geometry design](value_geometry_design.md) and its stage-3
acceptance cases. Builds on stages 1, 2 and 2b (implemented).

## Context

Stage 2 makes `resample` and `rf.resample_to` refuse a declared source, because interpolating
components as independent scalars ignores how they change between the source and target bases.
The design fixes the rule: interpolate the components at `T(y)` in their source basis, then
convert source basis to source frame, push to the target frame with `L = (D T(y))^-1` (or its
polar factor), and convert to the requested output basis, all by the stage 1 law.

`T` here is the planner's frame map, from target-frame points to source-frame points: the
caller's `transform`, a derived coordinate-system change, or identity. Its Jacobian `D T(y)` is
`(source axes x target axes)`, so `L = (D T(y))^-1` maps source-frame components to target-frame
components (a pushforward along `T^-1`).

## Current Decision

### API

```python
resample(source, target, *, ..., output_component_basis=None, component_reorientation=None,
         units=None)
array.rf.resample_to(target, *, ..., output_component_basis=None,
                     component_reorientation=None, units=None)
```

- For a declared source, `output_component_basis` and `component_reorientation` are both
  **required**; for an undeclared source all three are refused (no silent no-ops).
- `output_component_basis`: one basis for every slot. A `ReferenceFrame` related to the target
  frame (the frame itself, or an equivalent frame in another coordinate system), or, for
  `rf.resample_to` only, the target's binding source `ArrayCoordinates`. Plain `resample`
  returns an unframed array, which cannot hold an `ArrayCoordinates` declaration, so it refuses
  one. Mixed output bases are a later `rf.change_component_basis`.
- `component_reorientation`:
  - `"jacobian"`: `L` as is.
  - `"polar_orthogonal_factor"`: `polar_orthogonal_factor(L)`, the finite-strain
    reorientation (DTI's preservation-of-principal-direction alternatives are domain policy).
    Allowed only when every axis of the source frame and the target frame has the same unit
    string, so the polar factor of `L` is a physical rotation or reflection rather than an
    artefact of unit scaling; and only for `determinant_exponent == 0`, since a density's volume
    factor is not a reorientation. The source and output bases may be any basis: their
    conversions are basis changes applied around the reorientation, not part of it. (This
    narrows the design's "both bases orthonormal" to what the polar factor actually needs.)
  - a callable `f(L) -> R`, receiving a batch of maps shaped `(..., n, n)` (one `(n, n)` map
    for a constant `L`; a block of target samples, in no promised order, for a pointwise one)
    and returning the same shape, real and finite; a domain policy's hook. It must act on each
    map independently. Not persisted (nothing here is).
  - The composed map still passes the kernel's own checks, including `requires_orthogonal_map`.
- `units`: the new `attrs["units"]`, required exactly when the source bases' units differ from
  the output basis's (the stage 2 `_check_units` rule). Plain `resample` drops attrs today, so
  its result carries only this one. `rf.resample_to` restores the source attrs onto its result,
  then applies `units` last, so the supplied unit wins over the source's.
- `fill_value` must be NaN or 0 for a declared source: both are fixed points of every linear
  map and density factor, while any other constant becomes a different, valid-looking tensor
  after the map. Refused otherwise.

### Maps

Per slot `k`, with source basis `S_k`, source frame `F_s`, target frame `F_t` and output basis
`O`:

```
J_k(y) = M_out · R(L(y)) · M_in,k
M_in,k = basis_map(S_k -> F_s)   (binding matrix for the source ArrayCoordinates;
                                  coordinate_system_change for an equivalent frame)
M_out  = basis_map(F_t -> O)     (inverse target binding matrix for target ArrayCoordinates)
```

- `S_k` must be the source's binding source, `F_s` or a frame equivalent to it; any other frame
  is refused (no transform for it is available; change basis first). Every map must be square
  and invertible, as in stage 2. Only affine bindings (constant `M_in`, `M_out`) are supported.
- `L(y)`: constant when the frame map is identity or `SupportsAffine`. Otherwise the frame map
  must be `SupportsJacobian`, and `D T` is evaluated at the target-frame point of each target
  sample (target `Sampling` -> target binding). A frame map without a Jacobian is refused.
- **Pointwise maps are evaluated in blocks** of `block_points` target samples inside the map
  function, per Dask task: points, Jacobians, `L`, the reorientation and the composed maps
  exist for one block at a time, so peak extra memory is about `block_points * n^2` floats per
  map, not `prod(target_shape) * n^2` (9 GiB for one 3x3 map over 512^3). `block_points`
  thus also bounds reorientation; it is recomputed per non-geometry slice in a task, as
  positions are.
- Pseudotensors and densities need one map across slots, checked as in stage 2: slots whose
  `M_in` differ refuse for odd parity or `w != 0`.
- Packed declarations unpack, transform and repack, as `change_component_basis` does.

The existing interpolation runs unchanged on the components as non-geometry dimensions; the maps
are then applied to its output at each target sample. That is the design's order: interpolate
in `S`, then convert. For constant maps and linear methods the order does not matter; for
pointwise `L` it does, and only this order is offered.

### Implementation

- `_resampling.plan` records its `frame_map` on `_Plan`, and a helper returns target-frame
  points for the target samples (the computation `_general_map` already performs).
- `_tensor_components`: refactor `basis_map` to take a binding *transform* (Geometry and Grid
  carry transforms, not `BindingIndex`), and extract the stage 2 per-slot apply function
  (common map, per-slot maps, packed) so `change_basis` and resampling share it. Per-slot maps
  with a batch insert singleton axes for the other component axes, so the kernel's
  broadcasting rule holds.
- `_resample_with_intervals`: detach the declaration (labels stay as plain coordinates),
  resample, apply the maps with `apply_ufunc` (core dims: target geometry dims and component
  dims; Dask stays lazy), and return the new declaration; `resample` and `rf.resample_to`
  attach it, the latter after framing so an `ArrayCoordinates` output basis validates.
- **Chunking:** the interpolation leaves component dims as loop dims, so they may be split
  across chunks. Before the map step the component dims and target geometry dims are
  consolidated into one chunk each (`chunk({dim: -1})`), and the map gufunc runs with
  `allow_rechunk=False`; non-geometry, non-component dims keep their chunks. An empty target
  skips the map step (nothing to map; Dask's gufunc rechunking divides by zero on empty core
  dims, as the resampler already notes).
- Output dtype: `result_type(resampled dtype, float64)`; nearest-neighbour integer sources
  become float.

### Tests and probe

Acceptance (design): DTI-style rotation of a symmetric packed tensor with
`"polar_orthogonal_factor"` under a rotation-plus-scale registration (equal to `R T R^T`);
a non-unit source basis (anisotropic `ArrayCoordinates`) under identity registration, with the
binding factor carried through. Also: a vector under a pure rotation agrees with rotating the
interpolated values; `"jacobian"` and polar differ under a shear; covariant slots use
`D T^T`; an odd pseudovector under a reflection; a density gets `|det J|^w`; mixed slot bases;
a pointwise `SupportsJacobian` frame map (a composite with a non-affine member, or a test
transform) against an independent per-point computation; a callable policy receiving the
documented shape; `rf.resample_to` with the target's `ArrayCoordinates` output; `units=` rules;
fill rules (NaN and 0 kept, others refused); Dask with components and geometry dims split
across chunks, eager and computed, packed and unpacked, matching the eager result; an empty
target; a pointwise map with `block_points` smaller than the target, equal to the unblocked
result; `rf.resample_to` with a source whose `attrs["units"]` differs from `units=`; every
refusal (missing or
superfluous arguments, polar with unequal units or a density, unrelated source basis, plain
`resample` with an `ArrayCoordinates` output, no Jacobian, a callable returning a wrong shape).
The probe's `resample_to` row splits: without the new arguments it refuses; with them it keeps a
declaration in the output basis.

## Alternatives Considered

- **Defaulting `component_reorientation` to `"jacobian"`.** Rejected, as in the design: for
  diffusion tensors the common choice is a rotation, for flux vectors the Jacobian, and a silent
  default is wrong for one of them.
- **Converting to the source frame before interpolating.** Not offered (design): with a varying
  basis the two orders differ, and one order must be defined.
- **Applying a non-NaN fill after the maps** (masking outside samples). Rejected for now: the
  resampler does not expose its outside mask per sample for every method; NaN or 0 already
  covers the cases, and a later explicit mask can lift the restriction.
- **Polar reorientation for densities.** Refused rather than defined: dropping the volume factor
  changes what a density means.
- **Per-slot output bases.** Deferred; `rf.change_component_basis` after resampling does it.

## Deferred Work

- Non-affine bindings (pointwise `M_in`, `M_out`): with pointwise bases (design stage 4).
- Undeclared scalar densities through `resample` (`tensor_type=` as in `change_component_basis`).
- Constrained-value interpolation (log-Euclidean SPD, rotations): a policy hook, not core.
- Conservative remapping of densities (design Deferred Work).

## Review

Codex (gpt-6-astra, reasoning effort high), first pass: math, polar precondition and fill rule
confirmed. B1 (the map gufunc needs an explicit component and geometry chunk policy, and empty
targets), B2 (`rf.resample_to` replaced attrs after the helper, so a helper-set unit would be
lost), N1 (full-target map allocation; blocked evaluation chosen, so no allocation decision is
needed). All accepted and folded in above.

Convergence pass: Accept.

Implementation review, same reviewer: B1 (a nonlinear registration onto a fully scalar target
failed building its indices; the target-point helper now mirrors `_block_positions`) and N1
(batched per-slot maps and a callable's returned map were not covered; a blocked nonlinear
mixed-variance test with distinct slot bases and a `2 * L` callable test added). The second
pass recommended Accept, noting that the mixed-slot test fits one block; multiple block
boundaries are covered by the vector case with `block_points=4`.

## Outcome

`src/xarrayrf/_component_resampling.py` (integration layer), hooks on `_Plan` in
`_resampling.py`, arguments on `resample` and `rf.resample_to`, the shared per-slot apply and
`basis_map(binding_transform=)` refactor in `_tensor_components.py`; 45 tests in
`tests/test_component_resampling.py`. Suites: stock 2289 passed (47 xfailed), patched 2336
passed. The probe's `resample_to` row splits into a refusal without the arguments and a kept
declaration with them; totals 49 ok / 14 limitations (stock), 51 / 12 (patched), 48 / 15
(upstream), no holes.

## Next Steps

1. Stage 4, pointwise bases (`FrameField`), after angular charts (nonlinear plan stage 2).
