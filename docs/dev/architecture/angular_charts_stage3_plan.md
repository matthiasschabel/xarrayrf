# Angular charts stage 3: seam- and pole-aware resampling

**Status:** Implemented (2026-10-12; plan reviewed by Codex gpt-6-sol)
**Last updated:** 2026-10-12
**Scope:** `resample` (and `rf.resample_to`) on an angular source: global longitude wraps,
regional longitude is found from targets on another branch, and global grids interpolate across
the poles. Implements section 4 of the [angular charts design](angular_charts_design.md),
amended below. No change to labels, alignment, `positions_at`/`points_at`, box methods or
Cartesian behaviour.

## Context

Point methods locate each target point among the source samples: the source geometry's inverse
gives source coordinates, then `coordinate_to_position` (`_sampling.py`) turns each coordinate
into a fractional position, NaN outside the domain (`locator`, `_positions.py`). Interpolation
is `ndimage.map_coordinates` with clipped positions (`_interpolate`) on the general path, or
`ndimage.affine_transform` on a cropped window on the lattice path (`_lattice_map`,
`_crop_window`, `_lattice_block` in `_resampling.py`). Neither knows that longitude is periodic
or that latitude continues across a pole, so a target at 358.75° on a 0…357.5° grid is fill,
and on a grid with cell-centred latitudes (±89.5°) every target in the polar cap is fill.

**Prior art: xarraybc** (`~/GitHub/xarraybc`, a sibling project for topology-aware indexing).
Its `GridGeometry.periodic_bracket_array` locates a coordinate exactly on nonuniform labels with
an explicit period, canonicalizing the seam's rounding, and its `topologies.sphere` crosses a
pole by reflecting latitude and shifting longitude by half a period. This plan ports both ideas;
it does not depend on xarraybc, whose index-remapping accessor is NumPy-only, linear at most and
treats every periodic axis as closed (it would interpolate across a regional grid's 340° gap).
Its `Fiber.PSEUDOSCALAR` seam sign is the one-dimensional case of what value geometry handles
with tensor types and, later, `FrameField`.

**Amendments to the design.** Section 4 required uniform labels for a global axis and deferred
nonuniform ones, and said latitude never wraps. Exact location removes the uniformity
requirement (and the float32 allowance the first review round needed), and pole halos let
latitude continue across a pole on global grids. Both are recorded in the design's section 4 when
this plan is implemented.

## Current Decision

### Which axes take part

From metadata only, once per plan, in a helper `_periodic_axes(source)`:

- The source frame's coordinate system is spherical or geodetic.
- The source geometry's transform is affine, and the frame's longitude row has exactly one
  significant entry, in source axis `s_lon`, whose column has no other significant entry
  ("significant" as in `_classify_axes`). The same test on the latitude row gives `s_lat`. A
  coupled map (rotated pole) takes part in neither.
- Each is a dimension coordinate. Its storage axis is `k = source_dims.index(source.axes[s].dim)`,
  the reordering `_general_map` already applies to located positions.

Periods in source coordinates: `P_c = P / |m_lon|` for longitude, with `P` from the longitude
unit (`angular_period`) and `m_lon` the matrix entry. Latitude has its own unit and its own
affine row `φ = m_lat · c + t_lat`, so each pole is located in source coordinates through that
row: `c_pole = (±P_lat / 4 − t_lat) / m_lat`, with `P_lat` from the latitude unit. Mixed
degree/radian charts and scaled or translated latitude coordinates are therefore handled.

### Longitude coverage

Work in ascending order: if labels descend, reverse them (and map positions back, below). With
ascending labels `a_0 < … < a_{n−1}`, neighbouring steps `g_i`, and seam gap
`g_seam = a_0 + P_c − a_{n−1}`:

Coverage cannot always be inferred from labels: a nonuniform grid missing one sample can look
like a global grid with a wide seam step. Two sources decide it, in order:

1. **Declared intervals** (CF `lon_bnds`, `Grid(..., intervals=)`), when the axis has them:
   global when the intervals tile the period (each upper bound meets the next lower bound, and
   the last meets the first plus `P_c`, within `δ`); otherwise regional. Exact, no inference.
2. **Labels alone**, conservatively:

| Case | Test | Effective samples |
|---|---|---|
| Duplicate endpoint | `|g_seam| <= δ` (the first and last labels are one period apart) | `n_eff = n − 1`; the duplicate (the last in ascending order) is never read |
| Global | `δ < g_seam <= max(g) + δ`: the seam is no wider than the widest step anywhere in the grid | `n_eff = n` |
| Regional | anything else, including `n = 1` and labels spanning more than a period | no wrapping |

`δ = min(4 · ulp₃₂(max |a_i|), 0.01 · min(g)) + LATTICE_TOLERANCE · max(g)`. The float32 term
absorbs a float32-stored endpoint (about 1.2e−4° at 360°) but never exceeds 1 % of the smallest
step, so a valid float64 grid with 0.0001° steps keeps its last sample. A grid missing one
sample has a seam gap of two steps, wider than any single step unless the grid's steps vary by
more than a factor of two; that residual case is stated in the docs, with declared intervals as
the remedy. Uniform grids, float32 decimal steps (0.1°, 0.01°) and alternating 2°/3° steps are
global; a uniform grid one cell short is regional.

### Exact location (port of `periodic_bracket_array`)

For a global axis, a coordinate `c` is reduced to `u = a_0 + remainder(c − a_0, P_c)`, with
`u >= a_0 + P_c` canonicalized to `a_0`. `j = searchsorted(a, u, "right") − 1`; the right label is
`a_{j+1}`, or `a_0 + P_c` across the seam; the ascending position is
`p = j + (u − a_j) / (right − a_j)`, in `[0, n_eff)`. For descending storage the position is
`(n_eff − 1 − p) mod n_eff`. This is piecewise-linear inversion like `coordinate_to_position`,
exact for nonuniform labels and float32 noise alike. There is no domain check: every longitude
is inside.

For a regional axis, `c` is shifted by whole periods into `[lo, lo + P_c)` with
`lo = a_0 − before` (`before` is the domain's reach below the lowest label in coordinate units:
0 for `"samples"`, the cell extent for `"cells"`), then located by `coordinate_to_position` as
today.

Both happen inside `locator`, after the inverse and before any domain check, through a keyword
`periodic: PeriodicAxes | None` (a frozen record of the decisions above) that only
`_general_map` passes. `positions_at`, `points_at`, coincidence and `Geometry.lattice` keep
literal labels.

### Poles

A pole halo is built when longitude is global, its labels come in antipodal pairs, and the
latitude axis takes part. Antipodal pairs: `n_eff` is even and for every `i`,
`a_{(i + n_eff/2) mod n_eff}` equals `a_i + P_c / 2` modulo `P_c` within `δ`, so the half-turn is
the index shift `n_eff / 2` (a nonuniform grid such as 0, 80, 170, 260 fails this and gets no
pole halo). Per pole, in latitude source coordinates, with the outermost label `b`, its
neighbour's step `g` and the pole `c_pole` from the latitude row above:

| Case | Test | Mirror |
|---|---|---|
| Pole sample | `|c_pole − b| <= δ` | odd: label `b − x` mirrors to `b + x`; the pole row itself is not repeated |
| Cell-centred | `δ < 2 · |c_pole − b| <= max(g_lat) + δ` (the gap across the pole is no wider than the widest latitude step) | even: label `l` mirrors to `2 · c_pole − l` |
| Off the pole | otherwise | no halo; targets beyond `b` are outside, as today |

Halo rows take their values from the mirrored row at longitude index `(i + n_eff / 2) mod n_eff`;
the extended latitude labels are the mirrored labels, so location on that axis is
`coordinate_to_position` over the extended, still monotonic labels. Targets with
`|latitude| > 90°` remain outside: the extended labels must not admit them, so positions beyond
the pole's own label are refused explicitly.

A pole halo applies to scalar values only. Components declared in a basis that varies with
position flip across a pole (east and north reverse); until `FrameField` (value geometry
stage 4), component resampling builds no pole halo and polar-cap targets stay outside. Components
in a fixed Cartesian basis are continuous across the pole, but they are not distinguished in this
stage. `_resample` detaches the declaration before planning, so it passes a keyword
`pole_halos=components is None` to `_plan`, which gates the pole decision; seam wrapping is
unaffected by it.

### Execution path

Any periodic axis forces the general block path: `plan` skips `_lattice_map` (and so
`_crop_window` and the same-grid gather), since neither a modulo nor a mirror is affine and a crop
could drop samples across the seam.

The source block is extended before interpolation, in storage order:

1. Longitude (global): `np.take(volume, np.arange(−h, n_eff + h) % n_eff, axis=k_lon)`, so
   `n_eff < h` wraps repeatedly; positions on that axis are offset by `h`.
2. Latitude (each pole with a halo): up to `h` mirrored rows gathered from the
   longitude-extended block with the half-turn shift, capped by the rows available (`n − 1` for
   odd, `n` for even); positions on that axis are relocated against the extended labels.

`source_last` per axis becomes the extended size minus one. Halo `h`: 1 for nearest and linear;
24 for cubic, since the cubic B-spline prefilter's boundary influence decays as
`(2 − √3)^h ≈ 0.268^h`, and `0.268^24 ≈ 2e−14`; with fewer rows available at a pole the error
there is bounded by the same factor at the available count and stated. The NaN-mask path for
linear (`_linear_inputs`) runs on the extended block.

### Labels, outputs and box methods

Output coordinates are the target's labels unchanged; nothing is normalized; `xr.align` is
unchanged; interval claims are unaffected. Box methods do not wrap or cross poles in this stage.

### Documentation

`docs/core_interface.md` resampling section (when wrapping and pole crossing apply, the coverage
and pole cases, halos, the general path's cost, components and box methods excluded),
`CHANGELOG.md` Added entry, the design's section 4 amendments and next steps, changelog entry
with xarraybc credited as prior art.

### Tests (`tests/test_seam_resampling.py`)

1. Global 2.5° grid, smooth periodic field: linear and cubic onto half-cell-offset targets and
   onto targets labelled −180…177.5; no seam target is fill; values match the same resampling
   of the grid rotated by half a period (no seam in range) to roundoff.
2. Descending (357.5…0) and duplicate-endpoint (0…360 inclusive, last sample set to a value that
   must never appear) grids match the ascending grid.
3. Nonuniform global longitude (steps alternating 2° and 3°) locates exactly: a linear field in
   longitude within each cell is reproduced, across the seam too.
4. Coverage: float32 0.1° and 0.01° grids are global; a float64 0.0001° grid keeps its last
   sample; a uniform grid one cell short is regional; a 2°/3° grid missing a 2° sample is
   regional; declared intervals that tile the period make a grid global and ones that leave a
   gap make it regional, whatever the labels suggest.
5. Regional 170…190 found from −185…−175 and 175…185; −10…10 from 355; `domain="cells"` shifts
   into the branch that includes the outer cells.
6. Coupled (rotated) source affine: no wrapping, targets beyond labels are fill.
7. Storage transposed relative to source axes (`("lat", "lon")` storage, `("lon", "lat")` axes)
   wraps and mirrors the right axes.
8. Cubic with `n_eff = 8` (fewer samples than the halo) still wraps correctly.
9. Poles: cell-centred (±89.5°) and pole-sample (±90°) global grids, odd `n_eff` and
   non-antipodal nonuniform longitudes (no halo, cap is fill), latitude stopping at 60° (no
   halo), radian latitude with degree longitude, and latitude source coordinates scaled and
   translated (`φ = 2c + 10`); a field `f(x, y, z)` smooth on the sphere,
   sampled at the grid, is reproduced in the polar cap within interpolation error, and a target
   exactly at the pole gets one value whatever its longitude label; `|latitude| > 90` targets are
   fill.
10. Declared components on a global grid: seam wrapping works; polar-cap targets are fill
    (the planner received `pole_halos=False`).
11. NaN near the seam (linear) spreads only to its neighbours.
12. Cartesian and non-participating angular sources keep the lattice path (plan-level assertion
    that `lattice_map` is set), so their results are unchanged.
13. Replace stage 1's pinning test (a seam target is fill) with its wrapping counterpart.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Depend on xarraybc | Index-remapping accessor, nearest/linear only, NumPy only, closed periodic axes; coupling release cycles for about 60 lines of algorithm |
| Mean-step location with a float32 allowance (this plan's first version) | Needs uniformity tolerances that float32 decimal steps break; excludes nonuniform global grids |
| `mode="grid-wrap"` in `map_coordinates` | Wraps every axis, including latitude and non-geometry axes |
| A periodic lattice fast path | Needs cropping across the seam and a non-affine index map; measure the general path first |
| Wrapping in `positions_at` too | Its fractional positions feed label-based selection, which does not wrap |
| Pole crossing for components now | Needs the pointwise basis change at the pole (`FrameField`) |
| Pole crossing for odd `n_eff` | The half-turn falls between samples; would need interpolated halo rows |

## Deferred Work

- Box methods across the seam and poles.
- Pole crossing for declared components (after `FrameField`), and for odd longitude counts.
- A periodic lattice fast path, after measurement.
- An optional adapter producing xarraybc topologies from angular charts.

## Next Steps

1. Plan review (Codex), implement with the tests above, implementation review, QA, commit and
   push.
2. Stage 4: the minimal CF reader.
