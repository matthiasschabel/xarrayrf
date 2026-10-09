# Transform adapters

**Status:** Active (ITK-family codec scheduled with stage 1a of the
[field-backed transforms](../architecture/field_transform_design.md); CASTalign deferred)
**Last updated:** 2026-10-09
**Scope:** A proposed family of optional adapters that turn registration results from external
engines (CASTalign, ITK/SimpleITK, ANTs, elastix, and later voxel-space fields such as
VoxelMorph's) into xarrayrf transforms. Nothing is implemented yet. CASTalign specifics come from
the [CASTalign comparison](../castalign/castalign_comparison_review.md), a source reading at
`e12f509`.

## Context

[design.md](../../design.md) puts registration outside the core: "A downstream consumer supplies
a validated relation and owns its optimizer." That still leaves a gap. Users run an engine,
get a transform object or file, and must hand-translate it into a relation between frames,
getting the axis order, handedness, row or column vectors, units and direction right. Those
translations are where silent errors come from, and xarrayrf can own them without owning any
optimizer.

A registration extra that wrapped the optimizers behind one interface was considered and
rejected (see Alternatives). This note records the smaller design that replaced it.

## Current Decision

**Translate transforms; never run an optimizer.** Each engine gets one optional subpackage
(`xarrayrf.castalign`, `xarrayrf.itk`, …) under the shared contract in
[adapters_design.md](adapters_design.md): its own extra, core-only imports, and a report of
every normalization. The adapter wraps the engine's result as a core transform. The caller
always names both endpoint frames, because no engine records which frames its result connects.

```python
from xarrayrf import castalign as xca

frames = xca.frames(graph, units="1")  # one local ReferenceFrame per node, axes (z, y, x)
t = xca.transform(
    graph, "invivo", "fish3d", resolved_frames=frames
)  # CompositeTransform of wrapped edges
arr = xca.frame(img, frames["fish3d"])  # ndarray_shifted -> framed DataArray
xrf.resample(invivo_arr.rf.geometry, fish_arr.rf.geometry, transform=t.inverse())
```

### Each adapter is a codec

An adapter is a small, declarative mapping from the engine's conventions to the core's:

| Convention | CASTalign | ITK / SimpleITK / ANTs / elastix |
|---|---|---|
| Space | Unitless voxel space per graph node, `(z, y, x)` | Physical LPS, units assumed mm |
| Axis order | `(z, y, x)`, positional | `(x, y, z)`, fastest-varying first |
| Affine | Row vectors: `p @ matrix - shift`, so `matrix.T` and `-shift` | Column vectors, centre of rotation folded in |
| Direction | Edge `frm→to` pushes points from `frm` to `to` | Result maps fixed-image points to moving-image points |
| Source of transforms | `Graph` edges, `Transform` objects | Transform objects; `.tfm`, `.h5`, `.mat`; displacement-field NIfTI |

The codec handles nothing beyond this. Getting one row wrong, above all the row/column
convention, silently transposes the matrix, so each row gets a point-oracle test against the
engine's own `transform`/`TransformPoint`.

### Capabilities follow the wrapped class

The wrapper claims only what the engine computes exactly, so a transform never advertises a
capability its methods contradict (design.md §7):

- **Affine classes** (CASTalign `Rigid`, `Affine`, parametric variants; ITK `AffineTransform`,
  `Euler3DTransform`, …) become an exact `AffineTransform` with `SupportsInverse`.
- **CASTalign `Triangulation` and `LaminarTriangulation`** become a point transform. Both
  directions are piecewise affine on the same Delaunay connectivity, so `inverse()` can be
  claimed, subject to a round-trip test. No Jacobian is claimed. Out-of-domain points raise
  `AssertionError` in CASTalign (and pass through as NaN under `python -O`). The wrapper
  converts that to `ValueError` and checks results for NaN itself. `resample` passes a
  frame-to-frame result straight on to the source's inverse, so the wrapper is the place to
  check.
- **`PointTransformNoAnalyticInverse`** inverts numerically, with a 1000-point cap. Only its
  fast direction is wrapped; `SupportsInverse` is never claimed.
- **ITK B-spline and displacement-field transforms** become the
  [field-backed transforms](../architecture/field_transform_design.md): a field or control grid
  with its own lattice in the source frame, a declared interpolant and exterior behaviour, and
  a declared inverse pair where the engine writes both directions, never an inferred one.
  Voxel-space fields such as VoxelMorph's fit the same types. A lattice in the source frame
  avoids a chain through `ArrayCoordinates`, which the core refuses.

`resample` needs only `SupportsPoints` for the frame-to-frame step, written from the target's
frame to the source's. A nonlinear edge without an exact inverse is therefore usable only in the
direction it was wrapped, and `resample` refuses the other direction through the endpoint check.

### Composition stays on the xarrayrf side

`xca.transform` wraps each graph edge separately and composes a `CompositeTransform`. It does
not call `Graph.get_transform`, for two reasons. The chain keeps each member's capabilities.
And `Graph.add_edge` stores `invert()` for every edge, numerical or not, so a CASTalign-composed
chain can hide an inverse that should not be trusted. When the graph has more than one path
between the requested nodes, the adapter raises instead of choosing (CASTalign only warns). An
explicit xarrayrf-side frame graph is a separate, deferred question (CASTalign comparison,
synergy 2).

### Checks at the boundary

The adapter refuses rather than guesses when:

- the array has non-geometry dimensions the engine cannot carry;
- frame units do not match the engine's space (for example µm into an ITK space assumed mm), unless
  the caller supplies the conversion;
- the fixed and moving frames are the same frame. A non-identity map from a frame to itself
  contradicts frame identity. Motion correction within one DICOM Frame of Reference must first
  give the moving image a new local frame.

Direction errors need no extra check. Named endpoints turn a reversed transform into the
`ValueError` `resample` already raises for wrong endpoints.

## Alternatives Considered

- **A registration extra wrapping the optimizers.** ANTs stages and metrics, elastix parameter
  maps and ITK's optimizer and metric objects have little in common. A unified API would be
  either a lowest common denominator or a pass-through of engine keywords that adds nothing.
  It would also tie the release cadence to ITK, ANTs and TensorFlow/PyTorch stacks.
- **A `Registrar` protocol now.** `register(fixed, moving) -> SupportsPoints` has one
  implementation today. Extract a result contract only if three hand-written integrations
  converge on one.
- **nitransforms as the engine.** Rejected for evaluation (float32 points, fixed cubic
  interpolation, silent identity outside the domain; see the
  [field-backed transforms design](../architecture/field_transform_design.md)). Its `io` layer is
  adopted as the ITK-family codec's reader for formats Pirana's existing elastix and ANTs
  parsers do not cover, and it is a test oracle.

## Deferred Work

- A framed `DataArray` ↔ `SimpleITK.Image`/`itk.Image` bridge. It is part of the ITK adapter,
  not a separate module.
- Writing transforms back out (`to_engine`, ITK transform files), once a consumer needs it.
- The same-frame reframing helper for motion correction, if the manual step proves error-prone.

## Next Steps

1. Build the CASTalign adapter as the example prototype listed in the CASTalign comparison's
   deferred work: affine edges, `LaminarTriangulation`, graph composition, `ndarray_shifted`
   framing, with point-oracle tests against CASTalign's own `transform`. It exercises the
   `Transform` protocol with a third-party nonlinear transform and needs no core change.
2. Promote it to `src/xarrayrf/castalign/` with an extra only if CASTalign is interested. Its
   required dependencies include a Qt GUI stack, so the extra's weight needs checking first.
3. The ITK-family codec (ITK, ANTs, elastix; affine and field transforms) is built with stage 1a
   of the [field-backed transforms](../architecture/field_transform_design.md), starting from
   Pirana's elastix and ANTs parsers. It no longer waits for CASTalign.
