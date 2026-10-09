# Special relativity as a test of the core

**Status:** Deferred
**Last updated:** 2026-10-09
**Scope:** Lorentz and Poincaré transformations on spacetime arrays (implemented as examples and
tests, no physics in the core), measured contraction in 1+1-D and 3+1-D, and the deferred design
for relativistic viewing and angular coordinates. Executable companions:
`tests/test_relativity.py` and `examples/relativity.ipynb`.
The curved-spacetime follow-up is [the curved spacetime review](curved_spacetime_review.md).

## Context

The core was designed from imaging, microscopy, astronomy and GIS. Special relativity is an
outside test: a 4-D frame whose time axis is geometry, a linear but non-Euclidean symmetry group,
and quantities with textbook transformation rules. If it fits without new machinery, the
abstractions are general rather than tuned to images. It does, with three fixes it exposed.

## Current Decision

### Mapping onto existing pieces

| Physics | xarrayrf |
|---|---|
| Inertial frame | `ReferenceFrame` with `CoordinateSystem(("ct", "x", "y", "z"))` in metres (SI also works) |
| Lorentz boost / Poincaré transformation | `AffineTransform.from_matrix(source=lab, target=rocket, matrix=Λ, translation=a)` |
| Inverse, successive boosts | `inverse()` (exact); `compose` reproduces velocity addition `(a + b)/(1 + ab)` |
| 4-velocity, 4-momentum | `jacobian() @ u` |
| Wave vector, gradient (covectors) | `J^-T` |
| Stress-energy | `J T J^T` |
| Field on a `t, z, y, x` grid | `Geometry(..., dims=("t", "z", "y", "x"))`, time a geometry dimension |
| Field seen by another observer | `resample` or `rf.resample_to` with the target-to-source transform |

The core applies no metric: Minkowski metric, index raising, proper time and invariants are user
code. Tests cover interval invariance, velocity addition, the 4-velocity of a particle at rest
(`(1.25, -0.75, 0, 0)` at β = 0.6), frame independence of `k_μ x^μ`, and exact linear resampling
of a spacetime field. Nothing in composition, inversion, `points()`, `lattice()` or `resample()`
assumes three spatial dimensions. Reciprocal-space axes (`kx` in `1/mm`, `omega` in `rad/s`)
are declared the same way; a real-space affine M acts on them by `M^-T`.

Fixes the example forced into the core:

1. **Unit-independent inverse conditioning.** An SI boost has a raw condition number near 5e16;
   `inverse()` now measures and inverts after row and column equilibration.
2. **`Lattice.spacing` and `.direction` require one shared unit**; mixed-unit lattices report
   steps through `matrix`.
3. **`Geometry(..., dims=...)`** names geometry dimensions, which may include time (formerly
   `spatial_dims`).

### Measured contraction, 1+1-D

A scalar marker field (not a conserved density) represents one object's worldtube: an asymmetric
sum of three Gaussians plus two clock-pulse channels on a non-geometry `feature` dimension. It is
stationary in its rest frame C and synthesized on A's grid for `u = 0` and `u = 0.25c`. A to B is
a Poincaré boost with β = 0.6 and translation `(0.35, -0.25)` m.
`source.rf.resample_to(target, transform=A_to_B.inverse())` gives B's values; the transform maps
target to source. B's `ct = 0` row is simultaneous in B and corresponds to different A times, so
the whole spacetime field must be resampled before slicing. `rf.assume_frame` only relabels and
cannot implement a boost.

Length is `L = 2 sqrt(Σ ρ (x - x̄)² / Σ ρ)`, which scales as `1/γ_rel` for any stationary
integrable profile. The two clocks are four metres of proper `ct` apart; their B centroids differ
by `4 γ_rel`. For `u = 0`, `γ_rel = 1.25`: analytic width 1.80587080 m from proper 2.25733850 m,
tick 5 m. For `u = 0.25`, `w ≈ -0.411765`, `γ_rel ≈ 1.097345`: about 2.05709 m and 4.38938 m.
Predictions below use the measured C values at the same spacing and method.

| `u/c` | `h` (m) | method | width error (m) | tick error (m ct) |
|---:|---:|---|---:|---:|
| 0 | 0.20 | nearest | −3.6e−2 | +1.1e−3 |
| 0 | 0.20 | linear | −2.8e−3 | −2.2e−9 |
| 0 | 0.05 | linear | +2.8e−4 | −2.9e−9 |
| 0 | 0.05 | cubic | <1e−9 | −2.6e−9 |
| 0.25 | 0.20 | nearest | −8.6e−2 | +2.2e−3 |
| 0.25 | 0.20 | linear | −1.1e−2 | +1.1e−3 |
| 0.25 | 0.05 | linear | +1.4e−4 | +1.6e−6 |
| 0.25 | 0.05 | cubic | +9.5e−7 | +6.5e−10 |

The `h = 0.05` linear tests allow 5e−4 m and 1e−5 m of `ct`; these are empirical for this field,
not interpolation bounds. The source spans `[-11, 11]` m and the target `[-5, 5]` m, so every
target event lies inside the source; non-finite results raise rather than contaminate moments.
Tiny negative cubic overshoots are clipped before moments.

### Measured shape, 3+1-D

`n = (1, 2, 3)/sqrt(14)` in `(z, y, x)`, β = 0.8, γ = 5/3, with a nonzero origin translation. For
rest covariance C the prediction is `C' = A C A^T`, `A = I + (1/γ - 1) n n^T`: longitudinal
variance contracts by 0.36, RMS width by 0.6, the transverse block is unchanged and mixed moments
scale by 0.6. This does not claim that principal axes of an asymmetric object stay fixed.

| Source samples | Spacing (m) | Max tensor error (m²) | Parallel RMS ratio | Transverse RMS ratios |
|---|---:|---:|---:|---|
| 33³ | 0.4375 | 0.0304 | 0.6210 | 1.0204, 1.0371 |
| 63³ | 0.2258 | 0.00809 | 0.6056 | 1.0055, 1.0101 |

Linear interpolation adds about h²/6 variance per axis (0.00850 m² at the fine spacing), which
explains the error and its refinement ratio 0.266 ≈ (32/62)². Tests allow 0.009 m² tensor error
and 0.012 RMS-ratio error. The source spans `[-7, 7]` m; the target cube (half-width 3.6 m) is
centred on the moving position of the rest origin at `ct' = 0`. The rest-time interval covering
the whole target grid is about `[-7.907, 7.487]` m of `ct`; the static source uses two time
samples at those endpoints, for which linear time interpolation is exact.

### Relativistic viewing (deferred design)

The measured shape is a simultaneity slice, not what reaches a camera. A camera image needs:

1. **Past-light-cone sampling** from a reception event O: emission events E with
   `(E-O)·(E-O) = 0` before O (flat spacetime, straight rays). This is a different hypersurface
   from a simultaneity slice.
2. **Aberration:** Lorentz maps act linearly on null vectors; dividing by the time component gives
   a nonlinear (Möbius, in stereographic coordinates) map of directions.
3. **Terrell-Penrose appearance** emerges from sampling the cone, not from contracting an image
   and adding a fitted rotation.
4. **Doppler factor** `D = (-u_obs·k)/(-u_em·k)` is per-sample physics; specific intensity obeys
   `I_ν/ν³` invariance. A generic resampler must not apply a beaming power.
5. **Spectral colour** needs a spectrum model; RGB alone does not determine one.

In current terms, a target `Geometry` over `(theta, phi, r)` in `("rad", "rad", "m")` with a
user-defined `SupportsPoints` transform `F(θ, φ, r) = (ct_O - r, O + r n(θ, φ))` into the
observer's Cartesian spacetime frame is legal today: `ArrayCoordinates` accept angular units, and
the general resample path needs an exact inverse only of the **source** transform. So a bounded
cone gather into a Cartesian source is possible by source inspection (not yet run). The radial
reduction (first hit, or emission integral) is application code, after which the array should be
unframed and given a separately declared sky geometry.

Limits of the current path: the general path cannot locate source retained scalars, coordinate
fields or non-monotonic axes; `fill_value` handles points outside the source domain but not
transform exceptions, which fail the batch; `block_points` bounds coordinate temporaries (a 4-D
block of 2^20 points uses 32 MiB for positions) but not the whole job, and `rf.resample_to` does
not expose it.

Missing general capabilities, shared with astronomy and geography:

| Capability | Minimal proposal |
|---|---|
| Angular, spherical, geodetic systems | Immutable chart declarations: axis roles, periodic interval, bounded latitude or colatitude; no metric inferred from angles |
| Periodicity | Period and chart origin declared; never equate 0 and 360 in alignment or wrap implicitly |
| Nonlinear maps | Keep the existing protocols; engine wrappers (pyproj, Astropy) snapshot immutable configuration |
| Numerical inverse | Separately named, opt-in, with tolerances, iteration limits and per-point status; `inverse()` stays exact |
| Declared valid domains | Distinct from sample domain and visibility; raise by default, masked evaluation only on request |
| Projective maps | Only when several adapters need a homogeneous endpoint type |
| APE 14 | A view over `Geometry`, never a replacement for frames |

Periodicity is not invertibility: longitude is undetermined at a pole, directions collapse at
r = 0, geodetic latitude is not geocentric latitude. These must survive in declarations, not hide
in an open `axis_types` string.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| `LorentzTransform` class, `"minkowski"` representation, or a metric on `CoordinateSystem` | A boost is an affine map; a label or metric nothing reads is domain policy |
| Hand-coded rod endpoints, or worldline samples only | Verify event transforms but not resampling of a sampled object |
| A single A-time spatial image | Lacks the different A times one B slice needs |
| A core renderer | Visibility, material and spectral policy are not geometry |
| Contraction plus a fitted rotation | Ignores differing emission times |
| Angles as unitless Cartesian axes | Loses periodicity and singularities |
| Requiring global invertibility | Excludes embeddings; gathering needs only the source inverse |
| Mandatory pyproj or Astropy backend | Imposes engine conventions and dependencies |

## Deferred Work

Open decisions before implementing viewing or angular support:

1. First angular representations and chart metadata; whether the first stage includes
   seam-aware sampling (recommended: declarations and valid-chart evaluation only).
2. First astronomy and geography cases (recommended: lon/lat and the celestial sphere).
3. Numerical inverse result and failure contract; who chooses branches and initial guesses.
4. *(Resolved 2026-10-09 in the [field-backed transforms design](field_transform_design.md):
   exceptions by default, validity only when `resample` requests it.)*
5. Whether a shared homogeneous representation has enough consumers for a core type.
6. The light-cone demonstrator's observable (recommended: an optically thin emission integral).

Non-goals: renderer, visibility engine, radiative transfer, spectral uplifting, metric classes,
curved-spacetime ray tracing, a universal observer model. Also deferred: promoting a short
single-spacing contraction example to user documentation.

## Next Steps

Stages, each with acceptance criteria before claiming support:

1. **Angular systems:** celestial and geographic examples construct without disguising angles as
   Cartesian; invalid latitude, seams, poles, r = 0 and conflicting charts have explicit outcomes;
   angular geometry never acquires Euclidean spacing.
2. **Nonlinear transforms and engines:** analytic round trips within a chart; singular and
   out-of-domain points fail or report requested validity; encodings stay data-only.
3. **Light-cone example:** small `(θ, φ, r)` target through public APIs; β = 0 matches an
   analytic ray case; aberration and Doppler signs checked independently. A small cone
   feasibility experiment can precede stage 1, since angular inputs are already legal.
