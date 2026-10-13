# Field transforms, step 2: declared inverse pairs

**Status:** Implemented (2026-10-13; plan and implementation reviewed by Codex gpt-6-sol)
**Last updated:** 2026-10-13
**Scope:** Core only: the declared inverse pair, its two directions, the
`SupportsDeclaredInverse` capability, `CompositeTransform.declared_inverse()` and the
round-trip residual report. The second of the five steps of stage 1a in the
[field transform design](field_transform_design.md) (step table in the
[step 1 plan](field_transform_step1_plan.md)).

## Context

The design's Inverse section fixes the contract:

- `inverse()` stays exact. A field transform has none.
- A producer-declared pair (ANTs `1Warp` and `1InverseWarp`; later an NGFF `bijection`) is one
  immutable object owning both directions. Each direction exposes the other through
  `declared_inverse()`, which returns it without verifying it. One owner avoids mutually
  referencing transforms, which would make equality and encoding cyclic.
- `CompositeTransform.inverse()` keeps refusing a chain with a field.
  `CompositeTransform.declared_inverse()` is offered when every member is exactly invertible
  or has a declared inverse, and returns the reversed chain.
- A helper reports the round-trip residual of a pair, or of a chain and its declared inverse,
  on caller-chosen points.

The pair class name was left to this step. What exists: `SupportsInverse` and
`SupportsJacobian` are runtime-checkable protocols that the core tests with `isinstance`
(`_component_resampling.py:211`, `_tensor_components.py:490,526`), so a wrapper must not claim
a method its wrapped transform lacks. `CompositeTransform.inverse()` (`_composite.py:182`)
reverses members. `jacobian_determinant_report` (`_jacobian_determinant.py`) is the pattern for
a frozen report over points or a `Grid`.

## Current Decision

### Names

| Name | What |
|---|---|
| `DeclaredInversePair(forward, backward)` | The owner of both directions |
| `PairedTransform` | One direction: a transform that carries its declared inverse |
| `SupportsDeclaredInverse` | Protocol: `declared_inverse() -> Transform` |
| `CompositeTransform.declared_inverse()` | The reversed chain of exact or declared inverses |
| `RoundTripResidualReport`, `round_trip_residual_report(transform, at)` | The check |

All are exported from `xarrayrf`: the pair and its directions in
`src/xarrayrf/_declared_inverse.py`, the report in `src/xarrayrf/_round_trip_residual.py`, the
protocol in `_transform.py`.

### The pair

`DeclaredInversePair(forward, backward)`, keyword or positional:

- Both members pass `check_transform`. `forward.source == backward.target` and
  `forward.target == backward.source`, exactly, or `ValueError` naming the mismatched ends.
- A member that is itself a `PairedTransform` is refused (`TypeError`): it already has a
  declared inverse, and re-pairing it would give it two.
- A `SupportsAffine` member is refused (`TypeError`): it has an exact inverse, and wrapping it
  would hide its affine and exact-inverse capabilities from the paths that use them.
- Nothing is verified numerically; the name and the residual report carry that.
- Attributes: `forward` and `backward`, each a `PairedTransform`, created once at construction
  so `pair.forward is pair.forward`.
- Equality: the members' declared equality, in order (`forward == forward` and
  `backward == backward`); hash accordingly. Field members compare by identity, so two pairs
  built from the same field objects are equal.
- Immutable (`__slots__`, no setters); pickled through `__reduce__` as the constructor call on
  its two members. Pickle's memo keeps a half's pair shared with the half.

### A direction

`PairedTransform` is never constructed by callers (its constructor is private to the pair;
direct construction raises `TypeError` naming `DeclaredInversePair`). It has:

- `transform`: the wrapped member; `pair`; `direction` (`"forward"` or `"backward"`).
- `source`, `target`, `transform_point`: delegated.
- `declared_inverse()`: the pair's other direction, unverified.
- `jacobian(at=None)`: delegated, present only when the member is `SupportsJacobian`. The pair
  instantiates a private subclass with `jacobian` in that case, so `isinstance(half,
  SupportsJacobian)` is true exactly when the member's is. The wrapper is never
  `SupportsInverse`, so code requiring an exact inverse keeps refusing it.
- Equality: same direction of equal pairs. A half equal to another therefore has an equal
  declared inverse, which makes halves substitutable; the design's "a half's equality is its
  own identity, independent of its partner" is revised here, because equal halves with unequal
  partners would let two equal chains have unequal `declared_inverse()` results. There is still
  no cycle: the half compares its pair, the pair compares its members.
- `repr`: `DeclaredInversePair(forward=..., backward=...).forward`.
- Pickled as `(getattr, (pair, direction))`.

### Composite

`CompositeTransform.declared_inverse()` walks the members in reverse and takes, for each:

1. a nested `CompositeTransform`: its `declared_inverse()`;
2. a `SupportsInverse` member: `inverse()` (exact wins where both exist);
3. a `SupportsDeclaredInverse` member: `declared_inverse()`;
4. otherwise it is missing.

Missing members raise `TypeError` listing their indices, as `inverse()` does, and saying that
a chain with an unpaired field has no declared inverse. Errors raised by a member's `inverse()`
(a non-square affine) propagate. Each result passes `check_transform`, and its source and
target must be its member's target and source exactly (`ValueError` naming the member
otherwise); `_check_chain` compares only adjacent members, so it would not catch a wrong
inverse of a one-member chain or of the first or last member. For an all-exact chain the result
equals `inverse()`. `chain.declared_inverse().declared_inverse() == chain` when every member is
a `PairedTransform` or an affine whose inverse's inverse compares equal (an affine round trip
is numerically close but not guaranteed equal; tests use halves for the equality claim).

The class docstring gains one sentence: the declared inverse follows the same rule with
declared inverses admitted, unverified.

### Round-trip residual report

`round_trip_residual_report(transform, at) -> RoundTripResidualReport`:

- `transform` must be `SupportsPoints` (via `check_transform`) and `SupportsDeclaredInverse`
  or `SupportsInverse`; its inverse is chosen by
  the composite's rule (a composite's `declared_inverse()`, else `inverse()`, else
  `declared_inverse()`); otherwise `TypeError`.
- The source must be a Cartesian `ReferenceFrame` whose axes share one unit, so a Euclidean
  distance means something; otherwise `ValueError`. (Every field transform's endpoints are
  Cartesian; the shared-unit rule refuses, for example, a frame mixing millimetres and
  seconds.) Axes that all leave the unit undeclared (`None`) share it, and the report's unit
  is then `None`; a mix of `None` and a unit is refused.
- `at`: points shaped `(..., D)` or a `Grid` in the source frame, as in
  `jacobian_determinant_report`; empty input refused.
- The inverse's source must equal the transform's target and its target the transform's
  source (`ValueError` otherwise, so a user-defined `declared_inverse()` cannot return an
  unrelated transform silently).
- Computes `residual = inverse(transform(x)) - x`. Each direction's result is checked to be
  finite and real with shape `(..., D)` before use, since a user transform's result is
  otherwise unchecked and a wrongly shaped one would broadcast into a plausible residual. Points that either direction refuses
  propagate its error (a `"raise"` field outside its support); the caller chooses the points.
- Report fields: `residual` (read-only float64, `(..., D)`), `distance` (read-only, `(...)`,
  Euclidean norm in the source unit), `maximum`, `root_mean_square`, and `unit` (the source
  unit). Bytes-backed storage, as the determinant report.

### Encoding

`encode` refuses `DeclaredInversePair` halves with the stage 1b message, as it refuses fields
(the design encodes a pair as one record in 1b). A composite containing a half reaches that
refusal through its members.

### Tests (`tests/test_declared_inverse.py`)

- A smooth analytic pair: a dense field sampled from `u(p) = a·sin(p)` (small amplitude) and
  its inverse sampled by fixed-point iteration on the same grid; the residual report on
  interior points is below a tolerance derived from the grid spacing (linear interpolation
  error, O(h²·|u''|)), and an exact pair of opposite translations (constant fields) has
  residual ≤ 1e-12.
- Endpoint mismatch, re-pairing a half, an affine member, and direct `PairedTransform`
  construction are refused with their exact types.
- `declared_inverse()` returns the other half by identity; `half.declared_inverse()
  .declared_inverse() is half`.
- Capability honesty: a half over a field is `SupportsJacobian` and its Jacobian equals the
  field's; a half over a user transform without `jacobian` is not, and the composite Jacobian
  and component resampling refuse it with their existing messages; no half is
  `SupportsInverse` or `SupportsAffine`.
- Composite: the ANTs shape (fixed → intermediate field half, then an affine to moving)
  inverts to (inverse affine, backward half); points round-trip; nested composites; an
  unpaired field is refused with its index; exact chains give `declared_inverse() ==
  inverse()`; a non-square affine's error propagates; `inverse()` still refuses a chain with a
  half; a user-defined declared inverse with the wrong endpoints is refused in a one-member
  chain.
- Report: residual values for a known constant mismatch, `Grid` input, wrong-frame grid,
  empty input, non-Cartesian and mixed-unit sources, undeclared units, missing inverse, an
  inverse with wrong endpoints, a malformed (wrongly shaped) user result, a `"raise"` field
  outside its support propagating, immutability.
- Equality, hash, pickling (a pickled half's `declared_inverse()` is the other half of the
  same unpickled pair), repr, encode refusal.
- Resampling through a half (general path) and declared vector components through a half's
  Jacobian.

### Documentation

`docs/core_interface.md` "Field transforms" gains a "Declared inverses" subsection with an
executed example (ANTs-shaped chain, warp both ways, residual report). CHANGELOG Unreleased
and `docs/dev/changelog.md` entries. The design's Inverse section notes the equality revision.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| `declared_inverse` attribute on the field classes | Mutually referencing transforms: cyclic equality, encoding and pickling |
| Half equality by its own member only (the design's wording) | Equal halves with unequal partners; `declared_inverse()` would not respect equality |
| Halves that always expose `jacobian` and raise | Lies to the runtime protocols the core dispatches on |
| Requiring both members to have Jacobians | Excludes user-defined pairs for no reason; the conditional subclass is a few lines |
| Using a declared inverse where the core needs an exact one (`_tensor_components.py:490`) | Silently promotes an unverified assertion; callers can pass the declared direction explicitly |
| A residual that also runs the other direction | The caller passes `pair.backward` for that; one direction per report keeps it simple |

## Refinement Record

Plan review (Codex gpt-6-sol, high, 2026-10-13): Revise.

| Finding | Disposition |
|---|---|
| B1: the report must validate both directions' results | Accepted: shape and finiteness checked before subtraction |
| B2: `declared_inverse()` must check each inverse reverses its member | Accepted: per-member endpoint check |
| N1: undeclared units | Accepted: all-`None` shares a unit; the report's unit is `None` |
| N2: wrapping an affine hides its capabilities | Accepted: affine members are refused |

Implementation review (Codex gpt-6-sol, high, 2026-10-13): Accept; all four findings resolved,
no new findings.

## Deferred Work

- Encoding a pair as one record (1b).
- NGFF `bijection` import (1c).
- Numerical inversion (the approximate-inverse capability).

## Next Steps

1. Done: plan review, implementation, implementation review.
2. Next: the step 3 plan (`xarrayrf.elastix`).
