# Field transforms, step 1: dense and B-spline displacement transforms

**Status:** Implemented (2026-10-12; plan and implementation reviewed by Codex gpt-6-sol)
**Last updated:** 2026-10-12
**Scope:** Core only: `DisplacementFieldTransform` and `BSplineTransform` on a `Grid`, their
exterior behaviour, analytic Jacobians, the Jacobian-determinant diagnostic and an explicit
rasterization into a dense field. The first of the five steps of stage 1a in the
[field transform design](field_transform_design.md), which this plan implements without
changing its decisions.

## Context

The design is reviewed (refinement record of 2026-10-09). Its open decisions are settled with
the maintainer (2026-10-12):

1. Names stay as the design has them: `DisplacementFieldTransform`, `BSplineTransform`,
   `declared_inverse()`, and `jacobian_determinant_report`. The pair class is named in step 2.
2. ITK `.h5` and `.tfm` B-spline files are deferred until a producer needs them.
3. Importers live in one subpackage per engine: `xarrayrf.elastix`, `xarrayrf.itk` and
   `xarrayrf.ants`. Where the container-level persistence API lives is decided in 1b.
4. Whether stationary velocity fields or simplicial meshes follow 1a is decided after 1a.

Stage 1a in steps, each with its own plan, review and commit:

| Step | Content | Check |
|---|---|---|
| 1 (this plan) | The two transforms, exterior behaviour, Jacobians, determinant report, rasterization | Analytic, finite-difference and ITK boundary fixtures; resample through a field |
| 2 | The declared inverse pair, `CompositeTransform.declared_inverse()`, round-trip residual | A smooth analytic pair; a chain with an affine inverts |
| 3 | `xarrayrf.elastix`: Pirana's parameter-map parser | elastix fixtures (transformix) |
| 4 | `xarrayrf.itk` and `xarrayrf.ants` through nitransforms | ANTs registration fixture, both directions |
| 5 | Benchmark (design acceptance test 8) and backend choice | Recorded numbers |

What exists. Pirana's `pirana.spatial._transforms` has `DisplacementField` and
`BSplineTransform`. Both are 3-D only, use a units string rather than frame endpoints, evaluate
through `interpn` and `map_coordinates`, and have no Jacobian. xarrayrf has:

- a `Grid` with frozen, possibly nonuniform coordinates and intervals;
- `coordinate_to_position` and `cell_extent` (`_sampling.py`), the per-axis inversion and cell
  reach used by every query;
- `frozen_float_array` (`_validation.py`), immutable storage;
- the general resampling path, which maps target points through any `SupportsPoints` frame map
  (`_general_map` in `_resampling.py`);
- `CompositeTransform.jacobian`, which applies the chain rule to any `SupportsJacobian` members.

**ITK oracle, measured** with ITK 5.4.7 (Python) on a unit-spaced grid:

- A 4-sample `DisplacementFieldTransform` displaces points on continuous index `[-0.5, 3.5)`,
  including at exactly −0.5 and at 3.4999999. It is the identity at 3.5 and at −0.500001.
- A `BSplineTransform` with domain `[0, 3]` and mesh size 3 (6 control points per axis, so the
  control grid starts at −1) displaces points on `[0, 3]` closed, that is on control index
  `[1, n − 2]` closed. It is the identity at −1e-9 and at 3 + 1e-9.

## Current Decision

### Module and names

`src/xarrayrf/_field_transform.py` holds the shared grid preparation and the two classes;
`src/xarrayrf/_jacobian_determinant.py` holds the report. All of these are exported from
`xarrayrf`:

- `DisplacementFieldTransform`
- `BSplineTransform`
- `jacobian_determinant_report`
- `JacobianDeterminantReport`
- `sample_displacement_field`

Neither class implements `SupportsInverse` or `SupportsAffine`.

### Construction

```python
DisplacementFieldTransform(*, source, target, field, grid, outside="raise")
BSplineTransform(*, source, target, coefficients, grid, order=3, outside="raise")
```

**Endpoints.**
- `source` and `target` must be `ReferenceFrame`s with Cartesian coordinate systems, and the two
  systems must be equal (`coordinate_system ==`: axes, units and types). A displacement is added
  to the point, so the target axes must be the source axes. Distinct frames are the normal case;
  equal frames are allowed (a motion model within one frame).
- An angular or non-Cartesian system is refused with a message that names stage 2's
  seam-aware interpolant.

**Grid admissibility**, checked once at construction:
- `grid.frame == source` exactly.
- `grid.transform` is `SupportsAffine`, square (one varying dimension per source axis) and
  invertible. An `AffineTransform` gives a cached inverse.
- No source axis is a retained scalar.
- Every coordinate is nonempty and strictly monotonic. Descending coordinates are accepted, since
  positions are indices.
- **Dense fields:** every axis must declare cells. Either the grid's `ArrayCoordinates` gives a
  `sample_offset` for it, or the grid has intervals on it. The support is the union of cells,
  so an axis of point samples has no defined support and is refused. A single-sample axis must
  carry an interval, which gives the slab its thickness (design: "explicit intervals on a
  single-slice axis"). Declared intervals must tile their axis.
  - Rows are taken in ascending coordinate order, so descending axes work. With coordinates
    `[1, 0]`, the rows `[[0.5, 1.5], [-0.5, 0.5]]` tile.
  - Each row's upper bound must equal the next row's lower bound within `LATTICE_TOLERANCE`
    (1e-6) of the smaller of the two cell widths.
  - A grid with larger gaps or overlaps is refused.
  - A tolerated mismatch counts as rounding in the declared bounds, not as a gap: the support is
    the outer extent from `cell_extent`, and a point in a tolerated sliver is inside.
- **B-splines:** every axis has at least four samples and coordinates uniform within
  `EXACT_STEP_TOLERANCE` (1e-12 steps). That is the bound under which `coordinate_to_position`
  uses one constant step. Positions are `(c − c₀) / step`, so `dp/dc` is one constant per axis
  and the world-space Jacobian keeps the spline's continuity. A grid that is uniform only to
  the looser `LATTICE_TOLERANCE` is refused, because piecewise inversion would make `dp/dc`
  jump at samples. Cells are not required.

**Values.**
- `field` and `coefficients` have shape `(*grid sizes in grid.dims order, D)`. The trailing axis
  holds components ordered as the target axes.
- float32 and float64 are kept as given. Other real integer dtypes are stored as float64.
  Boolean, complex, masked and non-finite values are refused.
- The values are copied once into a `bytes`-backed buffer, in a dtype-preserving variant of
  `frozen_float_array`, and exposed through read-only `field` and `coefficients` views.

**`outside`** is one of `"raise"` (the default), `"identity"` or `"nearest"`, for both classes.
B-splines get `"nearest"` too, so the vocabulary is the same for both. It clamps to the B-spline
support `[1, n − 2]`.

**`order`.** Only 3 is accepted; any other value raises `NotImplementedError`, as in Pirana.

### Locating a point

Each transform prepares, at construction:
- the inverse of the grid affine, which maps frame points to source-axis coordinates;
- the per-dimension coordinate arrays;
- each dimension's step for `coordinate_to_position`: `singleton_step(sampling, "cells")`, which
  supplies a single-sample slab's interval width;
- the support bounds in continuous index.

`transform_point` maps a point `x` to coordinates `c = A⁻¹(x − b)`. Each dimension's position
`p_k` is then `coordinate_to_position(values_k, c_axis(k), step_k, extrapolate=True)`. This is
exact for uniform axes and piecewise linear for nonuniform ones, extended beyond the outer
samples by the outer steps.

**Support**, in continuous index per dimension:

| Representation | Support | Edges |
|---|---|---|
| Dense | `[-before, n − 1 + after)`, from `cell_extent`; `[-0.5, n − 0.5)` for centred cells | Lower closed, upper open, as ITK |
| B-spline | `[1, n − 2]` | Both closed, as ITK |

Edges are compared exactly, without slack, as ITK compares them:
- dense: `lower ≤ p < upper`;
- B-spline: `lower ≤ p ≤ upper`.

A point is inside only if it is inside along every dimension. The fixture shows ITK does the
same: an exact lower edge whose computed index rounds to −0.5000000000000003 on an oblique grid is
outside in ITK. `POSITION_SLACK`, which snaps query positions in `coordinate_to_position`, is not
used: it would admit points just below a closed edge and exclude points just below an open
one. Positions come from `coordinate_to_position(..., extrapolate=True)`, which applies no
slack.

**Exterior:**
- `"raise"` raises `ValueError`, naming the first outside point by its index and giving the
  support per named dimension in continuous index (exact for any grid; frame-space corners
  of an oblique grid add eight points of noise).
- `"identity"` returns the point unchanged and an identity Jacobian.
- `"nearest"` clamps each outside position to the closed support and evaluates there.

### Evaluation

One separable kernel per representation serves both values and Jacobians. Points are processed
in blocks of `2**16` so memory stays bounded. Gathered coefficients are converted to float64
before weighting, so the result is float64 whatever the stored dtype.

**Dense: multilinear interpolation.**
- Positions are clamped to `[0, n − 1]`, the edge clamping inside the cells.
- The base index is `floor(p)` clipped to `[0, n − 2]`, with weights `(1 − t, t)` over the
  `2^D` corners. A single-sample axis has one corner and weight 1.
- `∂u/∂p_k` replaces axis k's weights by `(−1, +1)`.
- **Tie rule:** on an interior sample face the derivative is the upper cell's; at `n − 1` it is
  the lower cell's.
- Along a clamped axis (beyond the outer samples, or clamped by `"nearest"`) and along a
  single-sample axis, `∂u/∂p_k = 0`, the derivative of the constant extension.

**B-spline: cardinal cubic, without prefiltering, over the `4^D` neighbouring coefficients.**
- The weights are the uniform cubic B-spline basis at `t = p − floor(p)`; `∂/∂p_k` uses its
  derivative.
- Values must equal `scipy.ndimage.map_coordinates(c, p, order=3, prefilter=False)` for
  positions in the support. That is the oracle in tests, and it is Pirana's evaluator.
- At the upper edge `p = n − 2` the base is clipped so the four taps stay in range.
- The cubic basis is C², so values and derivatives are continuous and the face tie rule does not
  matter.

**Jacobian.** `jacobian(at)` returns `I + (∂u/∂p) · diag(dp/dc) · P · A⁻¹`. Here:
- `dp_k/dc` is `1 / local step`, using the same cell as the value (the outer step beyond the
  samples).
- `P` selects each dimension's source axis.
- At an `"identity"` exterior point the Jacobian is `I`.
- `at` is required; omitting it raises `ValueError`, since the transform is not affine.

### Rasterization

`sample_displacement_field(transform, grid, *, outside="raise")` evaluates any `SupportsPoints`
transform at a grid's sample points and returns a float64 `DisplacementFieldTransform` on that
grid, with `u = T(x) − x`.

- The transform's endpoints must be Cartesian `ReferenceFrame`s with equal coordinate systems,
  and `grid.frame` must equal its source.
- The grid must satisfy the dense construction rules, including declared cells. A B-spline
  control grid without cells is refused, with the dense rule's message.
- This is the explicit B-spline → dense conversion the design requires. It also serves an affine
  (Pirana's `to_displacement_field`).
- No method is added to `AffineTransform`. One function covers every transform.

### Jacobian-determinant report

`jacobian_determinant_report(transform, at)` takes a `SupportsJacobian` transform with a square
Jacobian. `at` is either points shaped `(..., D)` in the source frame or a `Grid` whose frame is
the transform's source; a grid is evaluated at its sample points.

It returns a frozen `JacobianDeterminantReport` with:
- `determinant`: a read-only float64 array with the points' leading shape, or the grid's sizes in
  `grid.dims` order;
- `minimum`;
- `folding_fraction`: the fraction of points with determinant ≤ 0.

It raises `TypeError` without a Jacobian and `ValueError` for a non-square one. It replaces
Pirana's finite-difference `jacobian_measures`.

### Equality, hashing, pickling and repr

- **Equality is identity.** No `__eq__` is defined, so a hash consistent with it comes for free.
  Grids and composites holding a field are therefore hashable.
- **Pickling** reconstructs through the constructor, so dask's distributed scheduler can ship a
  resampling through a field. The result is a new, unequal object, which identity equality
  permits.
- **`repr`** shows endpoints, grid sizes, dtype and `outside`, never the array.

### Integration

- `resample(source, target, transform=field)` uses the general path with the field as the frame
  map. A field is a `SupportsPoints` map from the target image's frame to the source image's:
  the pull direction, which the design states. A `"raise"` field that excludes a target point
  fails the resampling until 1c brings per-point validity; the message names the transform.
- Declared vector components resample through a field's pointwise Jacobian on the existing
  component path. A test confirms it. If a refusal appears instead, it is documented, not
  worked around.
- `compose` keeps collapsing only affine runs. `CompositeTransform.jacobian` chains a field
  member's Jacobian.
- `encode` of anything holding a field refuses. This is a small change to `_encoding.py`: an
  explicit check before the generic fallback (`_encoding.py:308`), whose message says that
  field transforms are not persisted until stage 1b. A test pins it.

### Tests

The tests go in `tests/test_field_transforms.py`. Pirana's tests (`test_transforms.py`) are
adapted, not copied, where they cover these classes. Pirana stays untouched.

**Correctness against analytic and computed results:**
- A constant field equals the matching `AffineTransform` translation at random in-domain points.
- A field sampled from an affine map by `sample_displacement_field` reproduces it exactly under
  linear interpolation inside the samples.
- Jacobians match central differences for both classes on oblique, nonuniform and single-slab
  grids, and on float32 and float64 storage.
- B-spline values match `map_coordinates(prefilter=False)` on random points in the support.

**Boundaries and exterior:**
- The ITK boundary fixture (oblique dense grid, single-slice slab, B-spline), at the support
  edges and 1e-6 either side. Values are compared wherever ITK and xarrayrf both classify a
  point as inside. Classification is compared wherever the probe is at least 1e-9 from an
  edge, since an exact-edge probe on an oblique grid depends on each library's roundoff.
- On unit grids, where positions are exact: each edge and its immediate floating-point
  neighbours (`np.nextafter`) on both sides, and offsets within 1e-9, against the exact rule.
- Refusal of gapped and overlapping intervals (`interval_gap`), and of a B-spline grid uniform
  only to `LATTICE_TOLERANCE` (`nearly_uniform`).
- Every exterior word, and the `"raise"` message.
- Pirana's nonuniform-slice and single-slice cases, with the deliberate upper-edge difference
  from Pirana's closed domain stated in the test.

**Integration:**
- `resample` through a constant field equals `resample` through the translation.
- A field and a composite of affine and field in `resample`.
- Vector components through a field.
- `jacobian_determinant_report` flags a folded field and reports a smooth one as fold-free.

**Contracts:**
- Every construction refusal, and every raised error, by type and message.
- Immutable storage: views cannot be made writeable, and the input array is copied.
- Pickle round trip, repr, hashability and the `encode` refusal.

The ITK boundary fixture is `tests/fixtures/itk_field_boundaries.json`, generated by
`tools/build_itk_field_fixture.py`:
- The script needs only an `itk` installation, and records the ITK and Python versions.
- It writes only with `--apply`, through a sibling temp file and a rename.
- CI needs no ITK.

### Documentation

- `docs/core_interface.md` gets a section on field transforms: construction, support, exterior,
  evaluation, Jacobian, the report and rasterization, with executed examples.
- `CHANGELOG.md` gets an Unreleased Added entry.
- `docs/dev/changelog.md` records the step.
- The field transform design's open decisions 1 to 3 are marked settled, and its Status becomes
  "Active (stage 1a in progress)".

## Alternatives Considered

| Alternative | Why not |
|---|---|
| `map_coordinates` for values and a separate path for Jacobians | Two evaluators can disagree at faces; one kernel serves both, and the benchmark in step 5 may move values back to `map_coordinates` |
| `NdBSpline` for B-spline derivatives | Copies coefficients to float64 and adds a second backend before the benchmark justifies it |
| The resampling `locator` for world → index | It returns NaN outside and handles periodic axes and singleton roundoff that fields do not need; the field also needs `A⁻¹` and the local steps for its Jacobian |
| Closed dense support (Pirana) | ITK, the producer family, is half-open; Pirana is not an ITK boundary oracle (design) |
| A `domain=` keyword choosing samples or cells | No producer uses a sample-only dense support; requiring cells keeps one rule |
| B-splines without an exterior keyword (design signature) | Direct construction must default to `"raise"`, so the keyword is needed; one vocabulary for both classes |
| `AffineTransform.to_displacement_field` as in Pirana | One function for every transform, with no method on the affine core type |

## Deferred Work

- Steps 2 to 5, as tabulated above.
- Nearest and cubic dense-field interpolants (design Deferred Work).
- Lazy fields.
- Per-point validity (1c).

## Refinement Record

Plan review by Codex (gpt-6-sol, high), 2026-10-12:

| ID | Finding | Disposition |
|---|---|---|
| B1 | Edge slack contradicts half-open and closed edges | Accepted: exact comparisons as in ITK; tests at `nextafter` neighbours |
| B2 | `cell_extent` is a bounding extent, not a union | Accepted: intervals must tile their axis, compared in ascending order; tolerated slivers are rounding and lie inside (convergence pass: direction and tolerance policy made explicit) |
| B3 | Nearly uniform B-spline grids break continuity | Accepted: B-splines require `EXACT_STEP_TOLERANCE` and use one constant step |
| N1 | The encode message needs code | Accepted: an explicit refusal in `_encoding.py` |
| N2 | Rasterization grids must admit dense fields | Accepted: stated, with a refusal test |

Implementation review by Codex (gpt-6-sol, high), 2026-10-12. A convergence pass accepted it.

| ID | Finding | Disposition |
|---|---|---|
| B1 | The report's determinant could be made writeable again | Accepted: bytes-backed storage, tested |
| N1 | The `"raise"` message gives index bounds only | Plan revised: per-dimension index bounds are exact; frame corners of an oblique grid add noise |
| N2 | No test with grid dimensions permuted against the source axes | Accepted: a permuted, unequal-size grid checks values and Jacobians |

## Next Steps

1. Codex convergence review of this plan: B1, B3, N1 and N2 resolved; B2's direction and
   tolerance rule corrected as the reviewer specified.
2. Implemented with tests; all three lanes and `make check` pass; implementation reviewed.
3. Next: the step 2 plan (declared inverse pairs).
