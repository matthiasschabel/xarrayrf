# Support-aware resampling

**Status:** Active (proposed; no code yet)
**Last updated:** 2026-10-08
**Scope:** what a sample value means on an axis with declared intervals, and the resampling
operators that respect it: core `resample`, `rf.resample_to`, `Grid(intervals=)`. Extends
[geometry, cells and resampling](geometry_and_resampling_design.md), whose current behaviour is
unchanged by default. First consumer: MRI slice stacks in Pirana, where thickness differs from
spacing, possibly per slice.

## Context

Declared intervals exist (`Grid(intervals=)`, attached by the DICOM adapter from
`SliceThickness`), but resampling uses them only for the outer domain and a single slice's step.
Interpolation treats every sample as a point at its coordinate, and native resampling attaches the
target's intervals to point-interpolated output (`native.py:538-545`), which reads as a claim the
values do not make. The motivating case: a 1 mm axis resampled onto 5 mm. With point samples the
answer is interpolation; with 5 mm target slabs it should be an average; with gapped or
overlapping source slabs neither "interpolate the centres" nor "average the overlaps" is obviously
right.

Reviewed in three passes with Codex (gpt-6-astra, high); the abstraction below incorporates them.
Counterexamples quoted here were reproduced numerically.

## Current Decision

### Principle: simple by default, every operator reachable

Fast interpolation and support-correct operators trade efficiency against fidelity. The library
does not legislate that trade. The default stays today's point interpolation (`support="point"`),
so no existing result changes silently, and every operator below is reachable by naming it. Each
method documents its contract: consistent, local, monotone, identity on its own geometry, cost
class. Readers that declare intervals do not switch the default.

### What a value means

- **Point sample** (`sample_offset=None`): the value of f at a location.
- **Declared slab** `S_i = [a_i, b_i]` with value `y_i`: the mean of f over `S_i` (box profile,
  nominal; MRI slice profiles are not boxes and thickness is a nominal FWHM). It does **not** mean
  f equals `y_i` on `S_i`. What a thin slice through a slab shows is a reconstruction choice.
- **Unknown support** (cells without intervals, e.g. DICOM with `SliceThickness` missing): not a
  point declaration. Implied cell widths remain a tiling convention for domain and display only;
  operations that need widths refuse unless the caller supplies them.

**Profiles.** In general a sample is `y_i = ⟨f, φ_i⟩` for an analysis kernel `φ_i` (an MRI slice
profile; slice-selective excitation gives shoulders and side lobes, and DICOM does not record it).
The box is the special case used first. A declared interval is therefore the sample's *nominal
extent* (FWHM for MRI), not necessarily its support; the profile shape is a separate declaration,
box unless stated. Coverage and gaps are defined on the nominal extent. Profiles must integrate to
1, so constants are preserved.

The interval is not defined as an enclosed fraction of the signal: FWHM (what scanners report as
thickness) encloses about 76% for a Gaussian and a different fraction for other shapes, and profiles
with negative side lobes have a non-monotone cumulative integral, so "the range enclosing p%" can be
ambiguous or exceed 100%. Instead each shape is defined in units of its interval,
`φ(x) = (1/w) p((x - c)/w)` with `w = hi - lo` and `p` a unit-width shape whose definition states
its convention (box: unit support; Gaussian: unit FWHM). Interval plus shape name determine the
kernel; enclosed fraction is derived. Infinite-support shapes need an explicit, documented
truncation (a width multiple or enclosed fraction) for coverage, gap detection, cropping and
quadrature; it sets accuracy, not meaning. For asymmetric shapes the sample sits at the kernel's
centroid, which must agree with `sample_offset` (validation rule owed).

Consequence: linear interpolation through slab centres is the right reconstruction for points and
an approximation for slabs: its slab means differ from `y_i` unless the data is locally linear.

### Two orthogonal choices

1. **Reconstruction** of f from the source (`method`).
2. **Target functional** (`support`): `"point"` evaluates the reconstruction at target
   coordinates; `"average"` takes its mean over each declared target interval (box). `average`
   requires declared target intervals on the integrated axes.

Output metadata follows the functional: an `average` result declares the target intervals; a
`point` result does not claim them as measurement support.

### Consistency and identity

A reconstruction `f̂ = R y` is *consistent* if `A f̂ = y`, where `A` is the source slab-averaging
operator. Resampling is `y_T = A_T R y`.

- If `R` is consistent, resampling with `support="average"` onto the source's own intervals
  (`A_T = A`) returns `y` exactly: identity, hence idempotence, with no coincidence shortcut.
  `support="point"` at the slab centres does not: a smooth `f̂` need not equal its mean at the
  centre.
- Identity for arbitrary data needs the slab indicators to be linearly independent (`A` full row
  rank). Dependent slabs can reproduce only compatible data: `[0,1]`, `[1,2]`, `[0,2]` with means
  `(0, 2, 1)` are reproducible, `(0, 2, 9)` are not. Merging repeated or contradictory slabs changes
  the input and is an explicit caller step, never implicit.

### Information: coverage and resolution

Two separate quantities, both continuous:

- **Coverage**: how much of the axis carries measurements. 0 for point samples, 1 for contiguous
  slabs, above 1 where slabs overlap.
- **Resolution**: how localised each measurement is, set by thickness.

As thickness shrinks relative to spacing, slabs approach point samples continuously, so policy must
not switch abruptly at "has gaps". Overlaps are not mere redundancy: slabs offset by less than their
thickness carry sub-thickness information (the basis of MRI super-resolution from shifted thick
slices). Spending that redundancy on noise reduction or on resolution is the user's choice.

### Operator family

| Method | Reconstruction | Consistent | Local | Gaps | Notes |
|---|---|---|---|---|---|
| `nearest`/`linear`/`cubic` | interpolant through centres (today) | points only | yes | bridged | default; for slabs an approximation |
| `step` | piecewise constant f (linear cumulative integral F) | non-overlapping slabs | yes | fill | conservative remap on partitions (xESMF `conservative_normed` analogue) |
| `pchip` (on F) | monotone cubic of `F` at slab edges | non-overlapping slabs | yes | bridged by a defined heuristic | real data only; nonnegative for nonnegative data |
| `overlap_mean` | pointwise mean of covering slabs | no | yes | fill | cheap choice for overlapping stacks; blurs twice |
| `smooth` | smoothest consistent f | yes, given independence | no (banded solve per axis) | bridged by the prior | unifies points, slabs, gaps, overlaps |

Details per member:

- **Cumulative integral.** `F(x) = ∫ f`; slab constraints are `F(b_i) - F(a_i) = y_i |S_i|`.
  Point targets evaluate `F'`, box targets `(F(b) - F(a)) / (b - a)`. Linear `F` is `step`.
- **`pchip`.** In gaps, `F`'s increment is unknown; it is estimated from slab means interpolated
  across the gap (rule to be fixed precisely), leaving slab constraints untouched, so identity
  holds. It does **not** converge to an interpolant of the point values as thickness goes to zero:
  slabs at -1, 0, 1 with means 0, 1, 0 give `f̂(0) → 1.125`. It is a distinct reconstruction, not a
  thick-slice version of point interpolation. Complex data refuses: PCHIP is real-only and,
  being nonlinear, not phase-invariant (complex callers use `step` or `smooth`, which are linear).
- **`overlap_mean`.** Pointwise normalised: at each x, the mean of the slabs covering x, then a
  uniform average over the covered part of the target. Per-slab weights `|S_i ∩ T|` differ when
  coverage multiplicity varies inside the target (slabs `[0,1.5]`, `[1,2.5]` onto `[0,2]`: 0.6/0.4
  per-slab vs 0.625/0.375 pointwise); pointwise keeps coverage a union length, never above 1. The
  effective profile is target box convolved with source box: 2 mm slabs every 1 mm onto `[0,4]`
  give `(v0 + 2v1 + 2v2 + 2v3 + v4)/8` over 6 mm although `(v1 + v3)/2` is exact, and
  self-resampling smooths (`(v1 + 2v2 + v3)/4`). Not called "conservative": with overlaps
  `Σ y_i |S_i|` double-counts, so nothing is conserved.
- **`smooth`.** `f̂ = argmin ∫ |f''|² subject to A f = y`, with point samples as point
  constraints. It is the only member that generalises beyond box profiles: by the representer
  theorem the solution is a combination of each `φ_i` convolved with the spline's Green's function,
  so it is specified in those terms with the profile as a parameter; `step`, `pchip` and
  `overlap_mean` are box-only fast paths (they rely on slab values being differences of `F`).
  Specification still owed (astra B6): the function space and integration domain,
  boundary conditions (free ends give `f'' = f''' = 0` for box constraints; the point limit needs
  the matching treatment), complex handling (`|f''|²`), and the reconstruction operator written
  as `R`, not an unspecified pseudoinverse. Overlapping uniform stacks split into independent
  chains of edges (2 mm every 1 mm: `F_{j+2} - F_j = 2 y_j` couples even and odd edges
  separately); the prior fixes their relative constant. With well-posed constraints and distinct
  centres, the point limit is a natural cubic spline. Costs: deconvolution in disguise (ringing,
  negative values: slab means 0 and 1 across a 1 mm gap force f̂ < 0 inside the first slab), noise
  amplification on non-source targets, a global solve along the axis (one factorisation shared by
  all columns without missing data).

### Nullspace rule

The curvature penalty does not see straight lines, so it cannot choose a slope the data leaves
open: a single slab is satisfied by every `f = y + c (x - centre)` at zero cost. Among minimisers
of `∫ |f''|²`, choose the one minimising `∫ |f'|²` (equivalently `∫ |f''|² + ε ∫ |f'|²` with
`ε → 0`). It is inert whenever two or more independent slabs determine the affine part, and gives a
constant for a single slab, where every member of the family agrees. A constant is the least
committal estimate, not a confident one: the mean is exact, the value at a symmetric slab's centre
is pinned (every consistent line passes through `y` there), and uncertainty about the slope is
unbounded. Outside the slab the domain reach rule applies.

### Gaps

A target lying entirely inside source gaps has no measured support; any value there is an
inference from a prior, with the same standing as a linear interpolant between two point samples.
Refusing by coverage would be incoherent, since point samples are zero-thickness slabs whose gap is
the whole spacing.

- Methods without a prior (`step`, `overlap_mean`) return fill in gaps. `min_coverage` applies to
  them only; their partial results are means over covered support.
- Methods with a prior (interpolants, `pchip`, `smooth`) answer in interior gaps.
- The stack exterior keeps the existing domain reach rule.
- An optional diagnostic output reports, per target, the covered fraction and the distance to
  measured support (mm), so measured and inferred values are distinguishable without encoding the
  distinction as fill. Coverage alone cannot separate a 1 mm gap from a 20 mm one; distance can.

### Scope of the first implementation

- Integration only along an axis that maps to a single source axis after composing all
  transforms (permutation, reversal, scale, unit change; no shear or rotation coupling), checked in
  physical coordinates so nonuniform per-slice spacing and thickness qualify. Otherwise refuse
  before reading pixels.
- No source cropping on the new path; a cropped/full equivalence test precedes any optimisation.
  Centre-based cropping (`_resampling.py:150-156`) must not apply to it.
- Real and complex continuous values (except `pchip`, real only); integers only after explicit
  conversion; label maps refuse `average` and every slab method.
- Missing values: `propagate` works throughout; `average` + `omit` is deferred (the `omit` ratio
  of interpolants has no piecewise-polynomial closed form: `2 log 2 - 1` versus `1/3` for a ratio of
  integrals). Legacy cubic behaviour with missing values (shortcut gathers finite values, the
  general path gives NaN, `tests/test_resample.py:904-908`) is an explicit exception to the
  single-reconstruction guarantee until decided.
- Convergence of shrinking averages to the point value holds at continuity points with positive
  coverage only; point evaluation on a shared slab boundary uses a fixed ownership rule.
- Dask: geometry dimensions are already core dimensions with rechunking
  (`_resample.py:191-204`), so a global solve along an axis is a memory and cost contract to
  document, not an incompatibility.

### First slices: interval claims and the box methods

Decided 2026-10-08 for Next Steps 3 and 4.

**Axis classes.** Every first-slice decision is per target axis, from the composed affine map in
*coordinate* space (target coordinates → target frame → frame map → source coordinates), so
nonuniform axes qualify. The map is separable when its linear part is a scaled permutation within
the existing roundoff allowance; then each target axis `j` (including a retained scalar axis) maps
to one source axis `i` by `c_s = a_j c_t + b_j`. A non-affine transform or a non-separable map has
no axis classes. A separable target axis is:

- **pass-through** if every target coordinate maps onto a source coordinate value (the 1-D operator
  is a selection), and, for `support="average"`, the target declares no intervals there or declares
  the mapped source intervals;
- **slab** otherwise.

**Interval claims (slice 1).** Output intervals state what the values are, per axis:

- pass-through axis: the source's declared intervals at the selected samples, mapped into target
  coordinates (none if the source declares none), whatever intervals the target declared, provided
  they are declarable under the target's own `sample_offset` (validated as `Grid(intervals=)`
  validates them); otherwise none. The output keeps the target's transform unchanged, so it stays
  bindable with arrays on the target: a point-sampled target axis, or an offset the source's
  intervals disagree with, loses the claim (support unknown) rather than altering the target's
  declaration;
- `support="average"` slab axis: the target's declared intervals;
- any other axis, and every axis when there are no axis classes: none.

This replaces attaching the target's intervals to every result (`native.py:538-545`). It keeps a
thick-slice stack's thickness through an in-plane resampling and through self-resampling, and stops
point interpolation claiming support it does not have. Core `resample` stays unframed; the claims
are exposed to the native layer through a private return path.

**Box methods (slice 2).** `method="overlap_mean"` and `method="step"` (`Method` gains both) are one
operator: on each slab axis, the reconstruction at x is the mean of the source slabs whose closed
declared intervals contain x. `step` additionally refuses overlapping source intervals (beyond the
interval roundoff allowance), so a caller expecting a partition gets an error rather than a
silently different operator. Closed intervals fix boundary ownership: a point on a shared edge
takes the mean of both slabs, which is also the limit of shrinking symmetric averages.

- The operator is a tensor product of 1-D weight matrices, one per target axis (pass-through
  axes are selections), applied along the source's geometry axes; v1 requires every target axis to
  be pass-through or a slab axis whose source axis declares intervals, and refuses otherwise
  (resample the other axes first with a point method). Non-separable maps and non-affine
  transforms refuse before reading pixels.
- `support="point"`: weights from the covering slabs at the target coordinate; uncovered → fill.
- `support="average"`: requires declared target intervals on slab axes. Weights integrate the
  reconstruction over the target interval: split it at every source edge; a piece covered by the
  set `C` gives each slab in `C` weight `length / |C|`; normalise by the covered length. Coverage is
  the covered length over the target length, per axis, multiplied across axes; below
  `min_coverage` the value is `fill_value`. `min_coverage` defaults to 0.5 (confirmed
  2026-10-08) and must lie in `(0, 1]`. Rationale: interior MRI gaps sit near 80-90% coverage, so
  the threshold acts mainly at the stack exterior, where 0.5 matches the cells domain's half-cell
  reach. Known costs: a cliff at the threshold, and partial values are means over covered support
  (biased toward what was measured). Any-coverage (xESMF `conservative_normed`) was rejected as
  the default because slivers at the edge would produce values.
- Coverage output, opt-in: the native layer can return the per-target coverage fraction alongside
  the values (private plumbing, public opt-in on `rf.resample_to`; shape and name decided in the
  slice), so partial-coverage values are visible.
- Missing values propagate: a NaN component with positive weight gives NaN (real and imaginary
  parts independently, as for linear).
- Real and complex floating sources only; integer and boolean sources refuse (convert explicitly).
  Output float64 or complex128.
- `domain` does not apply (declared support is the domain); a non-default `domain` refuses.
- `support="average"` with `nearest`/`linear`/`cubic` refuses for now.
- No cropping; the same-grid gather is not used (pass-through axes are selections already).

### Prior theory

This is generalized sampling. Unser and Aldroubi's *consistent sampling* (1994; Unser, "Sampling:
50 years after Shannon", Proc. IEEE 2000) requires exactly the consistency property above and
constructs it as an oblique projection; Aldroubi and Gröchenig (SIAM Review 2001) treat nonuniform
sample positions. Wahba (*Spline Models for Observational Data*, 1990) defines smoothing splines for
arbitrary bounded linear functionals, equivalently Gaussian process regression with integral
observations, which is `smooth` with its posterior variance. MRI super-resolution from thick slices
models the slice profile as a point-spread function, often Gaussian with FWHM equal to the nominal
thickness (Greenspan; Plenge et al. 2012; SMORE). Implementations should be tested against these
formulations rather than derived afresh.

## Alternatives Considered

- **Infer the functional from target intervals.** Rejected: a target built from a reader would
  silently switch from interpolation to averaging.
- **Missing thickness means point.** Rejected: it asserts physics the file does not state.
- **Interpolate through slab centres, then average over the target** (pass-2 answer for overlaps).
  Rejected: same double blur and self-smoothing as `overlap_mean`, plus a second code path.
- **Average overlaps with per-slab weights.** Kept as the reasoning behind `overlap_mean`, but
  pointwise normalisation chosen so coverage stays a fraction.
- **Exact-coincidence shortcut for identity.** Rejected as the mechanism: idempotent but
  discontinuous (a target shifted by 1e-9 gets the smoothed answer). Consistency gives identity
  without it.
- **Coverage threshold for every method.** Rejected: incoherent in the point limit (see Gaps).
- **`max_gap`** (xarray `interpolate_na` precedent). Not in the first implementation; the distance
  diagnostic serves the purpose without a threshold.
- **Full slice-to-volume super-resolution** (NiftyMIC, SVRTK). Out of scope: the endpoint this
  family should not grow into.

## Deferred Work

- Smoothing parameter `λ` for `smooth` (`λ = 0` consistent, `λ > 0` trades exactness for noise
  suppression; scipy `make_smoothing_spline` precedent): the principled answer for noisy
  redundant slabs.
- Posterior variance as the information diagnostic: `smooth` is the posterior mean of Gaussian
  process regression with integral observations (spline–GP equivalence), and its variance grows
  smoothly away from measured support and shrinks with overlap.
- Non-box profiles (a `profile` declaration on source intervals and on the `average` target
  functional; a Gaussian target kernel simulates a thicker acquisition); `average` + `omit`; oblique or coupled integration (supersampled
  quadrature); width-corrected averaging (average over `sqrt(t² - s²)`) as an approximate
  deconvolution.

## Next Steps

1. Human decision: the legacy cubic-with-missing-values behaviour (proposed: refuse, matching
   Pirana). `min_coverage=0.5` was confirmed 2026-10-08.
2. Close the `smooth` specification (B6 items) and the `pchip` gap rule before implementing either.
3. Slice 1: per-axis interval claims (above).
4. Slice 2: the box methods with `support="point"|"average"` (above); then `pchip`, then `smooth`.
