# Resampling performance: closing the gap with SimpleITK

**Status:** Active (Steps 0–2 implemented; Step 3 next)
**Last updated:** 2026-10-13
**Scope:** Point-method resampling (`nearest`, `linear`, `cubic`) on the general path in
`_resampling.py`; field transform evaluation in `_field_transform.py`; source location in
`_positions.py` and `_sampling.py`; output dtype policy; an optional SimpleITK engine. Box
methods only through the dtype policy.

## Context

The maintainer ranks resampling performance as high value. Workload: 256³ float32 moving image
onto a 256³ grid through a 128³ float32 dense `DisplacementFieldTransform` (16.7M points),
linear, Apple M5.

| Path | s | Mpt/s |
|---|---|---|
| `resample_to` (one thread) | 3.9 | 4.3 |
| SimpleITK `Resample`, 1 / 10 threads | 0.46 / 0.08 | 37 / 198 |

cProfile (3.98 s): SciPy `map_coordinates` 1.39 s over 784 calls (3 field components × 256
field tiles of `_BLOCK = 2**16`, plus 16 image blocks); everything else, 2.6 s, is array
plumbing between generic stages.

Two causes, reviewed with Codex (gpt-6-astra, high):

1. **A SciPy floor.** Four generic `map_coordinates` passes per point (three field components
   recompute the same weights; ITK reads the vector pixel once and fuses the chain in C++).
   Zero-overhead SciPy composition tops out near 12 Mpt/s single-threaded.
2. **Layering overhead, about 65% of the time.** Each stage takes whole arrays, re-validates
   and re-materializes them:
   - `real_float_array` (finiteness scan and copy) at the field's input, at `locate`'s input,
     and `result = flat.copy()` in `_FieldTransform._evaluate`;
   - three clamps: field clamp, `coordinate_to_position` clip with slack and NaN, then
     `_interpolate` re-scans NaN, re-clips and writes through a mask;
   - gather and scatter of every row in `_evaluate` even when all rows are inside (Codex B1);
   - generated integer indices converted to float64 and back (`_block_positions`,
     `_general_map`; Codex B2);
   - per-axis `coordinate_to_position` plus `np.stack`, and a separate affine call per frame hop
     (0.75 s in `AffineTransform.transform_point`).

   The genuinely marginal-case machinery (singleton roundoff, periodic axes, NaN weights) is
   already skipped when inactive. The overhead is redundancy between layers, not edge cases.

Output dtype: `_output_dtype` promotes every float input to float64 (complex to complex128),
doubling output memory and bandwidth. Float32 data declares float32 precision acceptable for
the data; geometry and accumulators keep float64 where cheap.

## Current Decision

Six steps, each its own commit with an implementation review. Every step keeps the full test
suite green, including the ITK fixture, edge-probe and golden resampling tests.

### Step 0: benchmark hardening

`benchmarks/field_transform_benchmark.py` keeps field evaluation; a new
`benchmarks/field_resample_benchmark.py` measures whole resampling against SimpleITK:

- warmed, median of three timed runs; `tracemalloc` peak in a separate untimed run;
- record native thread limits (set `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` to 1 for
  single-thread lines) and output dtype;
- compare like for like: SimpleITK with `domain="cells"` semantics and the same `fill_value`;
- lines: dense field, B-spline field, affine-only (lattice path reference), 2-D field; each at
  float32 and float64 input; worker counts 1, 2, 4, 8 once Step 4 lands.

Baseline before Step 1 (`field_resample_benchmark.py`, median of three warmed runs, native
pools pinned to one thread; 16.7M points; Mpt/s; max difference from SimpleITK):

| Case | xarrayrf | SimpleITK 1 thread | SimpleITK 10 threads | max diff |
|---|---|---|---|---|
| 3-D dense field, float32 | 3.95 | 36.3 | 190 | 3.0e-8 |
| 3-D dense field, float64 | 3.66 | 36.6 | 181 | 7.9e-14 |
| 3-D B-spline field, float32 | 1.52 | — | — | — |
| 3-D affine (lattice path), float32 | 34.6 | 214 | 899 | 3.0e-8 |
| 2-D dense field, float32 | 6.55 | 68.4 | 311 | 3.0e-8 |

The agreement confirms the like-for-like setup (the float32 difference is SimpleITK's float32
output). The affine lattice path, `scipy.ndimage.affine_transform`, is itself 6× behind
SimpleITK single-threaded; Step 4's threads and Step 5's engine apply to it too.

### Step 1: output dtype follows the data

- Real floating input keeps its dtype (float16 becomes float32: SciPy does not interpolate
  float16); complex64 stays complex64; integers and booleans unchanged (float64 unless
  nearest-representable). Dask `output_dtypes` follows.
- `fill_value` conversion: NaN and explicit ±inf are kept; a finite value that rounds (0.1 in
  float32) is accepted rounded; a finite value that overflows to ±inf in the output dtype raises
  ValueError (today it would silently become ±inf).
- A caller who wants float64 results casts the input (`array.astype(np.float64)`); the
  docstring says so.
- Precision stays where cheap: positions, index maps and support checks in float64; per-block
  map outputs computed in float64 only where an accumulation needs it (missing-data weights
  compared with `ROUNDING_ALLOWANCE`, box sums), then cast at the write. Cubic coefficients are
  stored in the data dtype (SciPy filters each line in double internally; to verify against a
  float64 reference at a stated tolerance).
- Breaking change: `resample` docstring Returns, `docs/core_interface.md` (output dtype
  paragraph), CHANGELOG.

### Step 2: remove redundant plumbing (all transforms)

- Validate once at the resample boundary. Built-in providers (`AffineTransform`,
  `CompositeTransform`, coordinate changes, field transforms) gain a private
  `_transform_trusted(points)` that skips `real_float_array`; the public methods validate and
  delegate. Foreign `SupportsPoints` providers keep the public call.
- Integer target indices stay integer (B2); for regular target axes compute coordinates
  arithmetically instead of `sample_columns` gathers.
- Field evaluation (B1): when a block is wholly inside, or the exterior is `"nearest"`, slice
  instead of gather and scatter; `map_coordinates` writes straight into result columns; field
  blocks follow the resample block for dense values (the `_BLOCK` bound exists for B-spline
  taps in Jacobians, not values).
- Two clamps with two contracts, done once each: `locate` classifies against the declared
  domain (cell reach from sample offsets and intervals, NaN outside) and keeps its public
  semantics; a resampling-specific preparation then clamps valid positions to the sample bounds
  (which stops cubic extrapolation in outer cells) and builds the inside mask. The prepared
  buffer and mask are shared across slices (N3, local part) instead of being recomputed in
  `_interpolate` per slice. Cubic outer-cell tests (`cells_domain_holds`, `rotated_cells`)
  guard it.
- Merge the per-axis `coordinate_to_position` loop into one vector expression for uniform axes.

Target: at least 8 Mpt/s single-threaded on the workload, results bitwise equal or within
roundoff of today's.

**As implemented (measurement changed the plan):**

- Validation is not the cost: `real_float_array` is 0.05 s of 4.8 s, and a finiteness scan
  costs 0.5 ms per million points. The private trusted entry points were dropped; they would
  add API surface for about 1%.
- Memory layout was the cost. Reductions along the point axis of an `(N, 3)` array take
  9.4 ms per million points against 0.4 ms on `(3, N)` rows; SciPy interpolates `(D, N)` rows;
  interpolating one component of an interleaved `(..., 3)` field is twice as slow as a
  contiguous one (74 against 36 ms per million random points). So:
  - field transforms store components contiguously (`_frozen_values`; `field` and
    `coefficients` stay `(..., C)` read-only views);
  - the value path computes index rows `(D, N)` (`_FieldGrid.columns`), with one composed
    affine when every grid step is uniform, checks bounds per block, and, for a block wholly
    inside the support (the common case), clamps only if needed and interpolates each
    component straight into its output row: no classification, gather or scatter;
  - `locate` builds `(D, ...)` rows and returns their `(..., D)` view, so the resampler's
    transpose is contiguous; `_interpolate` skips the inside mask and the clip when no row
    needs them;
  - affine evaluation uses `einsum` (14–17 ms per million points against 24 for `matmul` on
    a narrow array; equal to 4e-16).
- The composed field-index map subtracts the first sample's frame point (translation plus the
  matrix applied to the first coordinate values) before scaling; folding an origin into the
  offset cancels local coordinates at large origins (Step 2 review B1: a 0.1-spaced axis at
  1e15 read index 2 for 1.25; coordinate values at 1e15 read 0.5 for 1/3). Regressions
  `test_a_large_grid_origin_keeps_local_indices`, `test_large_coordinate_values_keep_local_indices`.
- Integer target indices stay integer (B2). Value blocks are `2**20` points; Jacobians keep
  `2**16`.

Measured after Step 2 (Mpt/s, float32): 3-D dense field 5.22 (3.95 before), 2-D dense field
10.0 (6.55), 3-D B-spline 1.61 (1.52; cubic taps dominate), affine 32.0 (34.6, noise). The
profile is now 3.0 s, of which SciPy interpolation is 1.52 s and the three affine stages
0.44 s, which Step 3 fuses. The 8 Mpt/s target moves to Step 3.

### Step 3: a fused regular-grid path for one field between affines

When source and target are regular lattices, the frame map flattens to `A_post ∘ F ∘ A_pre`
(affines and coordinate changes around exactly one field transform `F`), and `F`'s grid is
itself regular (every axis has a uniform step under `EXACT_STEP_TOLERANCE`, so frame point →
field index is affine; nonuniform dense fields are supported and take the general path, with a
regression test of regular images through a nonuniform field):

- precompute `P`: target index → `F`'s continuous grid index, `Q0`: target index → source index
  through `F`'s source point, and `L`: the linear part of moving frame → source index;
- every composed map is anchored: integer indices relative to the block's first index, and
  frame points relative to the grids' translations, so no large origin is multiplied before
  it cancels (Step 2 review B1);
- per block: `p = P i` by slab broadcasting (no `unravel_index`), `u = F.values(p)` (three
  `map_coordinates` into one buffer), `s = Q0 i + L u`, then interpolate;
- support classification closed-form: the image under `P` of a block's index box is a
  parallelepiped; if it lies inside the support the block skips per-point classification, if
  it lies outside under `"identity"` the block is affine; only straddling blocks classify per
  point. Under `"raise"`, the whole target box is checked once at plan time.

**Boundary policy (decision point).** Reassociating the arithmetic can flip a point lying within
roundoff of a support face or source edge. Proposed contract: both paths approximate exact
arithmetic; differential tests assert agreement everywhere except points within `1e-9` index
units of a face, where either side's answer is accepted. The existing exact edge-probe tests
must keep passing on both paths.

Target: 1.4–2× over Step 2.

### Step 4: threads

- A bounded thread pool over blocks on the general and fused paths. Eligibility checks the
  frame map recursively: every member of a `CompositeTransform` must be an exact built-in type
  (`type(t) is ...`, so subclasses and foreign providers run single-threaded; they have no
  thread-safety contract).
- Control: a keyword `workers: int | None = None` on `resample` and `rf.resample_to`; `None`
  means automatic, `1` forces single-threaded. One worker inside a Dask task. In-flight blocks
  are bounded to `workers × block_points`, and the automatic count is also capped so in-flight
  position buffers stay within `POSITION_CACHE_BYTES` (256 MiB).
- Default: automatic, as above (maintainer decision).
- Deterministic errors: the lowest failing block's exception is raised.
- Target: 2–4× over single-threaded, measured at 1, 2, 4, 8 workers.

### Step 5: SimpleITK engine where valid

Automatic when SimpleITK is importable (new extra `xarrayrf[simpleitk]`); otherwise Steps 2–4.
Selection has two stages so Dask stays lazy: a candidate engine is chosen at plan time from
metadata (geometry, transforms, method, domain, dtype), and `execute` confirms it per concrete
block from pixels (no NaN under linear), falling back per block otherwise.

Domain and support compatibility are separate, explicit predicates, because `domain="cells"`
is not ITK's domain in general: cell reach comes from each axis's sample offset and declared
intervals, and point-sampled axes reach only their samples. ITK's half-cell domain matches
only when every source axis is cell-sampled with `sample_offset = 0.5` and no declared
intervals; the dense field likewise needs `sample_offset = 0.5` cells for ITK's field support.
Anything else (asymmetric offsets, point-sampled axes, intervals) uses design (b) or falls back.
Differential tests cover asymmetric offsets `(0, 0.25, 1)`, point-sampled axes and declared
intervals, run with the engine selected automatically and with the fallback forced.

Two candidate designs, chosen by measurement:

| Design | Valid for | Cost |
|---|---|---|
| (a) `sitk.Resample` end to end | 2-D/3-D lattices; frame map affines plus at most one dense field on a regular grid; `nearest`/`linear`; `domain="cells"` with centered cells on every source axis (above); no NaN in the block; field exterior `"identity"` or `"raise"` passing the plan-time box check; real float data | Copies the field to a float64 vector image; axis order mapped into ITK direction matrices |
| (b) SimpleITK positions only (`TransformToDisplacementField` per target slab), our location and interpolation | Same transforms (centered-cell field support); any source domain, method and NaN handling | One extra `(slab, 3)` float64 pass; memory bounded by slab |

Default `domain="samples"` falls outside (a), which is why (b) is listed. A private switch
forces the NumPy path for differential tests; the Step 3 boundary policy applies. Thread count
follows Step 4's rule (one inside Dask).

## Decisions for the Maintainer

1. Precision: float32 data gives float32 output (Step 1). Approved in discussion (2026-10-13).
2. Boundary tolerance: accelerated paths may disagree with the reference within `1e-9` index
   units of a support face or domain edge (Step 3; precedent in
   `tests/test_field_transforms.py:115–137`). Approved (2026-10-13).
3. Threading default (Step 4): automatic. Approved (2026-10-13).

## Refinement Record

Plan review (Codex gpt-6-astra, high, 2026-10-13): Revise.

| Finding | Disposition |
|---|---|
| B1: the two clamps enforce different contracts | Accepted: Step 2 keeps both, each done once, in resampling-specific preparation |
| B2: fusion needs a regular field grid | Accepted: eligibility requires uniform field steps; nonuniform fallback regression |
| B3: `domain="cells"` is not ITK's domain in general | Accepted: separate domain and field-support predicates, centered cells only; differential tests over offsets, point axes, intervals |
| N1: NaN eligibility needs pixels | Accepted: two-stage selection, confirmed per block in `execute` |
| N2: thread resource control | Accepted: `workers` keyword, recursive exact-type eligibility, cap by position budget; default escalated |
| N3: fill conversion rules | Accepted: NaN/inf kept, rounding allowed, overflow raises; float64 by casting input |

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Numba or a C/Cython kernel | A compiled dependency or build for a pure-Python package; revisit if Steps 2–5 fall short |
| PyTorch `grid_sample` | Fast and threaded, but a very large dependency and float32-only geometry |
| float32 positions | Large origins (projected geospatial, ~10⁶ m) lose sub-pixel accuracy; breaks exact support checks |
| Make SimpleITK required | Core stays domain-neutral and light; the SciPy path must remain fast on its own |
| Only the SimpleITK engine | Leaves installs without it at 4 Mpt/s and every non-valid case unimproved |

## Deferred Work

- Reusing prepared positions across slices for cubic and NaN-containing linear workloads beyond
  the cache budget (Codex N3, design part).
- B-spline field in the SimpleITK engine (ITK file support is deferred; boundary rules differ).
- Box methods' performance.

## Next Steps

1. Done: Codex plan review (gpt-6-astra, high); revised.
2. Done: maintainer decisions.
3. Done: Steps 0 and 1 (implementation review, Codex gpt-6-astra high: B1 missing weights kept
   double, B2 cubic coefficients kept double near the dtype maximum, N1 field benchmark
   hardened, N2 docs; accepted on the second pass).
4. Done: Step 2 (implementation review, Codex gpt-6-astra high: B1 large-origin cancellation
   in the composed index map, fixed for translations on the first pass and for coordinate
   values after the convergence pass maintained it; both regressions pass).
5. Steps 3–5 in order, each with an implementation review, docs and changelog.
