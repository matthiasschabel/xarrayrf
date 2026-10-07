# Curved spacetime as a test of the core

**Status:** Active (assessment; no implementation planned)
**Last updated:** 2026-10-01
**Scope:** Whether the frame, transform and sampling model can carry data on a curved,
pseudo-Riemannian spacetime (general relativity) without redesign; which layers already fit,
which assumptions break, and the smallest changes that would make it first class. Companion to
[the special-relativity note](relativity_notes.md), which covers the flat affine case, and to
[the nonlinear geometry plan](nonlinear_geometry_plan.md).

## Context

[The special-relativity note](relativity_notes.md) showed that Lorentz and Poincaré maps fit the
core as ordinary affines, with no physics added. General relativity is the next outside test:
charts are curvilinear, chart transitions are nonlinear, the geometry carries a metric of
indefinite signature, and physical "frames" (observers, tetrads) are not coordinate systems at
all. The question is architectural, not whether to build GR support. If the layering is sound,
the sampling and transform layers should carry GR data unchanged and the gaps should sit in the
frame-identity and quantity layers only.

Terminology: GR is pseudo-Riemannian (Lorentzian), not non-Riemannian. The distinction between
Riemannian and Lorentzian matters below only where a metric is implicitly used. Genuinely
non-Riemannian geometries (Finsler, metric-affine with torsion) add connection data and change
nothing at the storage or chart level, so the same assessment applies.

## Current Decision

### What the model is

The current design models an **affine space with named Cartesian charts**, not a **manifold with
an atlas**. A `ReferenceFrame` is a chart with identity; `CoordinateSystem` names its axes;
`coordinate_system_change` derives only signed permutations between two systems of the same
frame; any other relation is an explicit transform between *different* frames. The design is
metric-free throughout: no frame carries a metric, Euclidean or otherwise. Euclid enters
implicitly in exactly two places, listed under gaps.

### Mapping onto existing pieces

| Physics | xarrayrf today |
|---|---|
| Coordinate chart (Schwarzschild, Boyer-Lindquist, a 3+1 slicing) | `ReferenceFrame` with a 4-axis `CoordinateSystem`; angular axes need the deferred representation (gap 3) |
| Chart transition (Schwarzschild to Eddington-Finkelstein) | A user `SupportsPoints` + `SupportsJacobian` + `SupportsInverse` object between two frames |
| Chart domain (horizon, poles, r = 0) | `transform_point` raises outside its domain and never extrapolates; no domain object yet |
| Field on a product grid in one chart | `Grid` or `Geometry` over `(t, r, theta, phi)`, each a 1-D nonuniform coordinate |
| Multipatch output (cubed sphere) | One frame per patch, explicit transitions, `resample_to` across patches |
| Tangent vector, covector | `jacobian(at=x) @ v`, `J^-T` (specified quantity rules) |
| Metric, stress-energy with lowered indices | Not representable: the specified `second_rank_tensor` rule is `J T J^T` only |
| Observer frame field (tetrad) | Excluded: `check_transform` requires a point map |

### What already fits

1. **Transforms are chart transitions.** The protocols are structural, non-affine maps are
   admitted, `jacobian(at=...)` is pointwise, `inverse()` is exact, and out-of-domain
   evaluation raises. `CompositeTransform` applies the chain rule. No change is needed to carry
   a chart transition; the [nonlinear geometry plan](nonlinear_geometry_plan.md) already relies
   on this.
2. **Grid and Geometry are chart-local and metric-free.** A grid is a product of 1-D source
   coordinates plus a map into the frame. That is how numerical-relativity output exists on
   disk: one patch, one chart, product coordinates. Nonuniform spacing, declared intervals,
   `points_at`, `positions_at` and `resample` work in coordinate space and read no metric.
3. **Axes are symmetric.** Time is a geometry dimension with its own unit; nothing privileges
   three spatial axes. The special-relativity tests already exercise this.
4. **Coincidence is signature-agnostic.** `is_coincident` measures in index steps, not
   distances, so it is unaffected by the metric's signature.
5. **Quantity rules already split `vector` from `covariant_vector`**, the index placement GR
   needs, and are unimplemented, so the remaining gap (rank-2 placement) is a design choice
   rather than a retrofit.

### Where it breaks

1. **Frame identity is affine-chart identity.** `is_equivalent_frame`,
   `with_coordinate_system` and `coordinate_system_change` assume two coordinate systems of one
   frame differ by a signed permutation without rescaling; `rf.assume_frame` and adapter
   `frame=` refuse a non-affine system change explicitly. In GR the invariant object is the
   manifold and the chart is incidental, so a nonlinear chart change forces the manifold
   identity to be dropped and two charts of one spacetime become two unrelated worlds. This is
   the one architectural gap: the model has no notion of "same manifold, different chart" beyond
   signed permutations.
2. **Euclid is assumed where "rigid" and "spacing" are computed.** `affine_class` tests
   `M^T M = I` for the rotation, rigid and similarity classes; `Lattice.spacing` takes column
   norms. A Lorentz boost is an isometry of `eta = diag(-1, 1, 1, 1)` (`M^T eta M = eta`) and
   classifies as a generic `affine`. Harmless for storage, wrong for any geometric query. Any
   future world-distance or nearest-in-world query (named in [the design](../../design.md) §7
   as requiring a metric) inherits the same problem.
3. **Spherical charts are refused.** `CoordinateSystem` admits only `"cartesian"` and refuses
   angular units, so `(t, r, theta, phi)` cannot be declared honestly; the workaround is
   `units=None` on the angles, which [the special-relativity note](relativity_notes.md) already
   rejects for losing periodicity and singularities. Angular representations and declared
   domains are on that note's deferred list and on the
   [prior-art study](prior_art_coverage_study.md)'s candidate list; they are prerequisites here.
4. **Rank-2 tensors have one rule.** `J T J^T` is fully contravariant. The metric and the
   stress-energy tensor with lowered indices need per-index placement, evaluated at each
   sample's point through the geometry.
5. **Frame fields are excluded by decision.** `check_transform` requires `SupportsPoints`; a
   Jacobian-only object is not a transform. A tetrad is a pointwise linear map on tangent
   spaces with no point map, so an observer frame cannot be expressed. Supporting it means
   either admitting fibre maps as a separate kind or adding a `FrameField` value object. This
   cuts against an explicit rule in [the core interface](../../core_interface.md).

### Riemannian versus pseudo-Riemannian

Nothing at the chart, grid or transform level depends on signature. It bites only in gap 2 and
in any future metric-dependent query. The core's choice to apply no metric is what makes this
true; the cost is that there is nothing to extend from when a metric is wanted.

### Smallest changes for first-class support

None of these touch `Grid`, `Geometry`, the binding or `resample`, which is the evidence that
the layering is sound.

| Gap | Minimal change |
|---|---|
| 1 | Let a frame keep its identity across a user-supplied nonlinear chart transition (for example `with_coordinate_system(system, transition=...)`), with the signed-permutation derivation kept as the fast path. Composition rules and the `ArrayCoordinates`-only-at-ends rule are unchanged. |
| 2 | An optional constant or field metric on a frame (`eta` for Minkowski, identity implied when absent); `affine_class` tests isometry against it. The special-relativity note rejected "a metric nothing reads"; `affine_class` and `Lattice.spacing` are the readers. |
| 3 | The angular representation and declared-domain stages already listed in [the special-relativity note](relativity_notes.md). |
| 4 | Specify quantity rules as `tensor(values, indices=("up", "down", ...))` over trailing component axes, with `J` evaluated per sample from the geometry. |
| 5 | Decide whether fibre maps are a second transform kind or a separate value object; either way, `check_transform`'s rule needs an explicit exception. |

## Alternatives Considered

| Alternative | Why not |
|---|---|
| A `"minkowski"` or `"schwarzschild"` representation tag | A representation names a chart kind, not a metric; gap 2 needs a tensor, and gap 1 needs a transition, which a tag supplies neither of |
| Modelling every chart as a separate frame and accepting lost identity | Works today for resampling, but makes "is this the same spacetime?" unanswerable, which is the question frames exist to answer |
| Metric classes or curved-spacetime ray tracing in the core | Non-goals, as in the special-relativity note; the core should carry the data and the chart transitions, and leave geodesics to user code |
| Angles as unitless Cartesian axes | Rejected in the special-relativity note for the same reasons |

## Deferred Work

- Whether gap 1 is worth solving before any user needs it. The nonlinear geometry plan's
  deformable-registration stage does not need it: a displacement field relates two frames.
  Geographic and celestial charts (lon/lat to projected, ICRS to galactic) do, so it is likely to
  arrive from that direction rather than from relativity.
- Whether gap 2's metric belongs on `ReferenceFrame` or on `CoordinateSystem`. A metric is a
  property of the space, expressed in the chart, which suggests the frame holds it and a
  coordinate system change transforms it.
- Any demonstrator (a Schwarzschild field resampled between two charts through public APIs)
  waits for the angular-representation stage.

## Next Steps

None scheduled. Revisit this note when the angular-representation stage, the quantity rules or
a nonlinear `with_coordinate_system` is designed, and update the gap table as each closes.
