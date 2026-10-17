# Field transform partial domains (stage 1c)

**Status:** Implemented (Codex plan review, two rounds, gpt-6-astra)
**Last updated:** 2026-10-17
**Scope:** Per-point validity in `resample` for fields declared `outside="raise"`, and
`PositionFieldTransform`. Implements the [field transform design](field_transform_design.md),
sections "Validity (1c)" and "Representations". NGFF field import is not part of 1c (deferred
by the maintainer on 2026-10-17; see the design's Deferred Work).

## Context

A field declared `"raise"` makes `resample` fail as soon as one target maps outside its
support. Three places raise:

- The general path maps target points with `frame_map.transform_point` (`_general_map`,
  `_resampling.py`), which raises in the field, or in a later composite member or the locator
  if a row were passed on as non-finite (`real_float_array` refuses NaN).
- The fused lattice path calls `field._displacement_rows` and raises `field._outside_error` for
  the first column outside the support (`_lattice_field_map`, `_resampling.py`).
- Component (vector/tensor) resampling evaluates `frame_map.jacobian` and
  `frame_map.transform_point` at every target point (`maps_at`, `_component_resampling.py`).

The SimpleITK engine is not involved: `_field_engine` takes only an identity exterior, or a
`"raise"`/`"nearest"` field whose every target-box corner lies inside the support.

NaN source positions are already the fill channel: the locator returns NaN outside the domain,
the fused path sets NaN columns, and interpolation fills NaN positions with `fill_value` on the
general, fused, threaded and Dask paths.

## Current Decision

### Validity as NaN source positions

The design's contract stays: values are finite wherever valid; a member without masked
evaluation is called through its ordinary `transform_point` and its exceptions propagate; direct
`transform_point` and `jacobian` calls still raise. This plan fixes how it is realized:

- **Internal only.** No public masked protocol. A public one would double the paired types
  again (`PairedTransform` already splits into `_PairedTransformWithJacobian` so that protocol
  checks stay truthful, `_declared_inverse.py`). Validity is offered only by built-in field types,
  recognized by exact type as `_chain` recognizes composites and pairs (a subclass may override
  `transform_point`).
- **`_FieldTransform._masked(points, *, jacobian)`** returns `(values, valid)`, with `valid`
  shaped `points.shape[:-1]` and values NaN where invalid. Only `"raise"` produces invalid rows;
  `"identity"` and `"nearest"` are valid everywhere. It shares `_evaluate`'s classification:
  `_evaluate` gains a `masked` flag, and `_displacement_rows` returns the outside mask instead of
  the first column when masked. The refusal path and its message are unchanged.
- **`masked_points(transform, points)`** and **`masked_jacobian(transform, points)`** (new
  private module `_validity.py`) walk `_chain(transform)`: each member is evaluated only on the
  rows still valid (compressed, then scattered back), so no invalid row reaches a later member.
  Built-in field members use `_masked`; every other member uses `transform_point` (and
  `jacobian`), whose exceptions propagate. The Jacobian is the chain rule over valid rows, each
  member's Jacobian taken where the earlier members place the point, as
  `CompositeTransform.jacobian` does.
- **General path:** `positions` maps through `masked_points`, locates only valid rows, and
  leaves the rest NaN.
- **Fused path:** under `"raise"`, columns outside the support get NaN displacement, so their
  source positions are NaN and filled. Boundary decisions keep the documented agreement with
  the general path (within about 1e-9 index units).
- **Component path: one validity decision for both stages.** Interpolation and the component
  stage must agree on which rows are valid; a row filled by one and transformed by the other
  would return a fill value pushed through a component matrix, or an untransformed value. The
  fused path classifies field indices from target indices and may differ from pointwise
  evaluation by roundoff, so a request with declared components and a `"raise"` field in its
  frame map takes the general path. There, interpolation (`_general_map`) and the component
  stage (`target_frame_points`) compute bit-identical target points
  (`target.transform.transform_point(sample_columns(...))`) and pass them through the same
  `masked_points`, whose classification is row-wise, so the masks agree whatever the blocking.
  `maps_at` maps through `masked_points` and `masked_jacobian`; matrices and the
  pole-refusal check are computed at valid rows only; invalid rows are set to `fill_value`
  explicitly (`fill_value` is passed into `resample_components`), not left as found. A block
  with no valid row calls no later member, Jacobian, frame field or reorientation. Fields
  declared `"identity"` or `"nearest"`, and requests without components, keep the fused path.
- **`CompositeTransform`, `PairedTransform`:** public behaviour unchanged. Filtering between
  members happens inside `masked_points`, which is what the design's "CompositeTransform
  filters rows between members" requires of `resample`.
- **Chart transitions** keep raising beyond a pole or within the centre's radius. The same
  mechanism admits them later by giving the built-in transition types a `_masked`; recorded in
  the angular charts design's Deferred Work, not done here.

### `PositionFieldTransform`

`PositionFieldTransform(*, source, target, field, grid, outside="raise")`: `y = f(x)`, the stored
value is the target position, interpolated linearly. SPM's DARTEL and Shoot deformations
(`y_`/`iy_`) are this form; NGFF `coordinates` would be.

- **Endpoints:** both Cartesian, coordinate systems may differ in axes and units, but the axis
  counts must be equal. Resampling's component path needs a square Jacobian, and no producer in
  scope changes dimension; relaxing it later breaks nothing.
- **Support:** the sample hull, continuous index `[0, n - 1]` per axis, closed. A displacement
  field's ITK support `[-0.5, n - 0.5)` extends edge samples as constants, which for a
  displacement is a translation but for a position collapses the half-cell onto one point (a
  zero Jacobian column). Every axis therefore needs at least two samples; a single-sample axis
  is refused, since it has no derivative. Cells and intervals are not required.
- **Exterior:** `"raise"` (default) or `"identity"`, the latter only when source and target
  coordinate systems are equal (it is `y = x`). `"nearest"` is refused: a constant extension has
  a zero Jacobian, so its push is singular.
- **Jacobian:** `∂f/∂x`, the multilinear derivative chained with the world-to-index map, with
  the dense field's tie rule on cell faces; the identity at an identity exterior point.
- **Class layout:** a third `_FieldTransform` subclass. `_check_endpoints` takes a flag for the
  equal-system requirement; `_FieldGrid` takes the hull support; evaluation reuses
  `_sampled_values` and `_multilinear_derivative`, with the displacement step (`x +`) moved
  into the subclass so a position is not added to the point.
- **Fused lattice path:** not extended. Its index formula `s = Q i + K u(p) + q` assumes a
  displacement; position fields take the general path. Recorded in Deferred Work, to be measured
  first.
- **Not added:** a `sample_position_field` conversion and any importer (SPM files read through
  nibabel are a caller's few lines; NGFF is deferred). `jacobian_determinant_report` and
  `round_trip_residual_report` accept it as they accept any `SupportsJacobian`/point transform.
- **Persistence:** new kind `position_field_transform` within schema 1, fields `source`,
  `target`, `grid`, `outside`, `field`, following `displacement_field_transform` in
  `_encoding.py`. Declared pairs encode their members by kind and need no change.

### Documentation

- `docs/core_interface.md`: field transforms section gains `PositionFieldTransform`;
  resampling states that targets outside a `"raise"` field receive `fill_value`, while direct
  calls still raise.
- `field_transform_design.md`: the "Persistence (1b)" bullets are rewritten to match 1b as built
  (the record carries endpoints, grid, interpolant or order, exterior and a content-digest
  reference covering dtype, shape and bytes; Dataset and Zarr assembly is in `xarrayrf.native`),
  replacing the stale text rather than adding another revision note. Status and "Validity (1c)"
  updated to point here.
- Both changelogs.

## Tests

- Existing tests that expect `resample` to refuse a `"raise"` field change to fill assertions,
  keeping their direct-call refusal checks and later-block coverage:
  `tests/test_field_transforms.py` (general-path refusal, around line 764) and
  `tests/test_resample_fields.py` (fused refusal, around 173; later-block refusal, around 326).
- `tests/test_field_transform_validity.py` (resampling through `"raise"` fields), each for a
  dense field and a B-spline (their support edges differ: half-open cells, closed `[1, n - 2]`):
  1. Acceptance test 10: a composite `affine → "raise" field → affine` with a user-defined last
     member that records every row it receives; no invalid row reaches it or the locator, and
     invalid targets get `fill_value`.
  2. A partly valid `"raise"` field on the fused lattice path (uniform target and source) and on
     the general path (nonuniform target) gives the same values and the same fill pattern.
  3. Component resampling (a declared vector) through a partly valid field: valid targets match
     the fully valid result computed on a subset, invalid ones are `fill_value`; with targets on
     and immediately beside support faces, an oblique target grid and a large origin, and both
     `fill_value=0` and NaN. The expected mask is the field's pointwise support classification
     of the same target points (the fused scalar path may differ by roundoff at a face, so it
     is not the oracle there); fused and general results are compared only away from faces.
  3a. Cumulative masking: a chain with two `"raise"` fields, the second invalidating further rows;
     a block whose rows are all invalid; a recording user member and a recording reorientation
     see only valid rows and are not called for an empty block.
  3b. A valid row with a singular Jacobian still raises (validity is not invertibility).
  4. Threaded and Dask-chunked resampling give the same result as serial.
  5. A user transform that raises still raises from `resample` (not reinterpreted).
  6. Direct `transform_point` and `jacobian` on the field still raise with the existing message.
- `tests/test_field_transforms.py` (`PositionFieldTransform`):
  7. A position field sampled from an affine map reproduces it under linear interpolation,
     values and Jacobians; source and target with different axis names and units.
  8. Support edges: `0` and `n - 1` inside, just beyond them raises or is identity; finite
     differences match the Jacobian inside.
  9. Refusals, one test per raised error: `"nearest"`; `"identity"` with unequal systems;
     unequal axis counts; a single-sample axis; non-Cartesian endpoints.
  10. Resampling an image through it matches resampling through the equivalent affine.
- `tests/test_encoding.py`, `tests/test_native_encoding.py`: round trip of the new kind alone,
  in a declared pair and through netCDF and Zarr.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| A public `SupportsMaskedPoints` protocol | Doubles the paired-transform types again for protocol truthfulness; no user transform needs it yet |
| Interpreting a user transform's `ValueError` as invalid | Swallows bugs; the design forbids reinterpretation |
| Masking inside `resample` only, after `CompositeTransform.transform_point` | An invalid row fails at the next member first |
| `PositionFieldTransform` with ITK's cell support and constant extension | Collapses the outer half-cells to a point; zero Jacobian columns |
| Allowing `"nearest"` for positions | Constant extension, singular push in component resampling |
| Extending the fused path to position fields now | A different index formula; measure the general path first |
| Carrying the fused path's outside mask into the component stage | Threads a mask through every interpolation path for one case; the general path already gives identical points |

## Deferred Work

- Masked evaluation of chart transitions (pole, centre).
- Fused lattice path for component requests through a `"raise"` field, if measurement shows the
  general path is too slow there.
- Fused lattice path for `PositionFieldTransform`, after measurement.
- Dimension-changing position fields, when a producer needs them.
- NGFF `displacements`, `coordinates` and `bijection` import (maintainer decision 2026-10-17).

## Next Steps

1. Codex plan review; revise.
2. Implement validity (internal evaluation, three paths) with tests 1–6; `make check` and the
   three test lanes.
3. Implement `PositionFieldTransform` and its encoding with tests 7–10 and the encoding tests.
4. Codex implementation review, docs and both changelogs, commit and push to `main`.
