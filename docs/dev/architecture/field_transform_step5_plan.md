# Field transforms, step 5: benchmark and evaluation backends

**Status:** Implemented (2026-10-13; plan and implementation reviewed by Codex gpt-6-sol; N1 awaits the maintainer)
**Last updated:** 2026-10-13
**Scope:** Design acceptance test 8 (`benchmarks/field_transform_benchmark.py`) and the choice
of evaluation backends for `DisplacementFieldTransform` and `BSplineTransform`. The last of the
five steps of stage 1a in the [field transform design](field_transform_design.md).

## Context

Step 1 evaluates both representations with one NumPy kernel each, which computes values and
derivatives together on every call. The design left the B-spline backend to this benchmark,
naming `map_coordinates(..., order=3, prefilter=False)` and `NdBSpline` (knots
`arange(n + 4) - 2`) as candidates.

**Measured before the change** (Apple M5, 16.7M points of a 256³ image grid through 128³
float32 fields, one thread; memory is `tracemalloc`'s peak of Python-visible allocations, not
process memory):

| Path | s | Mpt/s | tracemalloc peak MiB |
|---|---|---|---|
| Prepared grid lookup | 0.46 | 36 | 1152 |
| Dense, own kernel: values | 7.7 | 2.2 | 410 |
| Dense, own kernel: values and Jacobians | 17.1 | 1.0 | 1562 |
| B-spline, own kernel: values | 57.1 | 0.29 | 419 |
| B-spline, own kernel: values and Jacobians | 117.4 | 0.14 | 1571 |
| B-spline, `map_coordinates`: values | 5.8 | 2.9 | 192 |
| B-spline, `NdBSpline`: values (1/16 of the points) | 0.24 | 4.3 | 24 |
| B-spline, `NdBSpline`: values and three derivatives (1/16) | 0.99 | 1.06 | 96 |
| `resample_to` through the dense field | 10.2 | 1.65 | 300 |
| SimpleITK `Resample` through the same field | 0.47 | 36 | — |

On random points (cache-hostile) the ratios hold: dense values 1.8 (own) against 5.7
(`map_coordinates`, order 1) Mpt/s. A values-only NumPy multilinear without derivative work
reaches 4.5 Mpt/s. SimpleITK's C++ loop stays an order of magnitude ahead; matching it is not a
goal of a NumPy library.

## Current Decision

Evaluation needs SciPy, loaded lazily, as resampling already does; constructing, comparing,
pickling and inspecting field transforms do not. Evaluating without SciPy raises `ImportError`
naming `xarrayrf[resample]`. One backend per quantity, no environment-dependent fallback:

| Quantity | Dense field | B-spline |
|---|---|---|
| Values | `map_coordinates(order=1, mode="nearest", output=float64)` per component, on positions clamped to the samples as now | `map_coordinates(order=3, prefilter=False, mode="grid-constant", output=float64)` per component, on positions clamped to the support |
| Jacobians | the own multilinear kernel (its tie rules on cell faces are the contract) | `NdBSpline` with `nu` per axis, on the same clamped positions; its float64 copy of the coefficients is built on first use and kept |

- Values and derivatives no longer share a call: `transform_point` computes values only, and
  `jacobian` computes derivatives only (the multilinear kernel gains a derivatives-only mode).
  The exterior, clamping, raise message and zeroed derivative along clamped axes are unchanged.
- A cubic B-spline's derivative is continuous, so `NdBSpline` and the removed own kernel agree
  everywhere; dense values are continuous, so `map_coordinates` and the own kernel agree to
  roundoff. The own cubic kernel is removed; the multilinear kernel stays for Jacobians.
- `grid-constant` at the B-spline's upper face: at `p = n - 2` the fourth tap has weight
  `β₃(2) = 0`, so the padding value never contributes.
- Step 1's tests (ITK fixture, `map_coordinates` oracle, central differences, tie rules) stay
  and pin the new backends; a test checks the `ImportError` with SciPy hidden.

### Measured after the change

Same machine and workload; "public" lines are `transform_point` and `transform_point` plus
`jacobian`, so they measure exactly the selected calls on all points, with classification and
clamping included:

| Path | s | Mpt/s | tracemalloc peak MiB |
|---|---|---|---|
| Dense: values | 1.9 | 8.9 | 394 |
| Dense: values and Jacobians | 9.5 | 1.8 | 1559 |
| B-spline: values | 9.1 | 1.8 | 394 |
| B-spline: values and Jacobians | 23.8 | 0.71 | 1607 |
| `resample_to` through the dense field | 3.9 | 4.3 | 300 |
| SimpleITK `Resample`, 1 / 10 threads | 0.46 / 0.08 | 37 / 198 | — |

Speed-ups over step 1: dense values 4×, B-spline values 6×, B-spline Jacobians 5×,
resampling through a dense field 2.6×.

### Benchmark

`benchmarks/field_transform_benchmark.py` (development-only) prints one line per measurement:
lookup, values and values plus Jacobians per representation through the public methods, the
alternative backends for reference, `resample_to`, and SimpleITK `Resample` with one thread and
with its default count when SimpleITK is importable. The numbers above and after the change are
recorded here.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Keep NumPy kernels, optimized | Measured ceiling about 4.5 Mpt/s for dense values and far below SciPy for cubic taps; the design already accepted SciPy for B-splines |
| SciPy when importable, NumPy otherwise | Two paths whose results differ by roundoff depending on the environment, and twice the tests |
| `NdBSpline` for B-spline values too | Comparable speed to `map_coordinates`; one fewer backend for values, but `map_coordinates` handles float32 coefficients without a float64 copy |
| A compiled extension (Numba, C) | A build or heavy dependency for a library that ships pure Python |

## Refinement Record

Plan review (Codex gpt-6-sol, high, 2026-10-13): Revise.

| Finding | Disposition |
|---|---|
| B1: benchmark the selected calls | Accepted: the public-method lines measure exactly the selected calls on all points (table above) |
| N1: SciPy for evaluation is a user-visible decision | Decided by the implementer under the maintainer's go-ahead for stage 1, and flagged in the report to the maintainer: field evaluation sits beside resampling, which needs SciPy, and the ITK and ANTs importers need it too; reversible by restoring a NumPy kernel |
| N2: Jacobians at exact support faces | Accepted: a quadratic-reproduction test at both faces of a four-point axis, with `"raise"` and `"nearest"` |
| N3: label memory as traced | Accepted |

Implementation review (Codex gpt-6-sol, high, 2026-10-13): B1, N2, N3 resolved; N1 maintained
and escalated to the maintainer (whether core-only installations may lose field evaluation).

## Deferred Work

- Multithreaded evaluation (blocks are independent; SciPy releases the GIL in
  `map_coordinates`).
- Fusing the field map with image interpolation in `resample`, which is where SimpleITK's lead
  comes from.

## Next Steps

1. Done: plan review, implementation, implementation review.
2. Open: the maintainer's decision on N1 (SciPy for evaluation).
3. Next: the pole sign-flip halo plan, then stages 1b and 1c.
