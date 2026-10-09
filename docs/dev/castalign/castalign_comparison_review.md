# CASTalign comparison

**Status:** Implemented
**Last updated:** 2026-10-07
**Scope:** How CASTalign models spaces, transforms and resampling; where it overlaps xarrayrf,
where it differs, and which pieces could be combined. Informs the
[prior-art study](../architecture/prior_art_coverage_study.md) and any registration or
frame-graph work layered above the core.

## Context

Max Shinn, CASTalign's author, commented on napari/napari#3783 that xarrayrf "sounds very
similar to the architecture behind CASTalign" (https://github.com/mwshinn/castalign). This note
records a source reading of CASTalign at commit `e12f509` (July 2026, MIT). Nothing here was run;
behaviours come from reading the cited source. The napari wish-list assessment is separate:
[napari #3783 coverage review](../napari-3783/transformations_wishlist_review.md).

CASTalign registers 3D microscopy volumes (coppaFISH, immunofluorescence, in vivo two-photon)
with interactive napari GUIs. It has three layers:

- **Transforms** (`castalign/base.py`). A 3D class hierarchy in fixed `(z, y, x)` order.
  Parametric transforms (translate, rigid, affine, rescale, flip, matrix) and point-fitted ones
  (`Rigid`, `Affine`, `LaminarAffine` for thin sections, and the nonlinear Delaunay
  `Triangulation` and `LaminarTriangulation`). `a + b` composes: affine chains collapse into
  one matrix, nonlinear chains compose point-wise. Every transform must provide `invert()`;
  `PointTransformNoAnalyticInverse` falls back to numerical inversion, refusing more than 1000
  points. Resampling is pull-based `scipy.ndimage.map_coordinates` in threaded z-chunks, with
  an optional CuPy path.
- **Graph** (`castalign/graph.py`). Nodes are named spaces, usually one per image; edges are
  transforms. `add_edge` stores the inverse automatically, so edges are bidirectional.
  `get_transform` composes along a breadth-first shortest path; `g["a":"c"]` is shorthand. A
  cycle prints a warning and is otherwise accepted. Graphs persist as SQLite with compressed
  images inside.
- **GUI** (`castalign/gui.py`). Landmark picking and parameter sliders in napari.
  `GraphViewer` resamples each node's image into a chosen display space and adds the resampled
  array as a layer, with `ndarray_shifted.origin` as the layer translate.

## Current Decision

CASTalign and xarrayrf overlap in vocabulary (named spaces, composable transforms, pull
resampling) but sit at different layers. CASTalign is a registration application with a model
inside it; xarrayrf is the model with no registration. Treat CASTalign as a potential consumer
and a source of transforms, not as a competitor or a design to absorb.

### Comparison

| Concern | CASTalign | xarrayrf |
|---|---|---|
| Named spaces | Strings (graph node names) | `ReferenceFrame` identity, `CoordinateSystem` axes, units and directions, `ArrayCoordinates` |
| Transform endpoints | Implicit in the graph edge | Declared `source` and `target` on every transform |
| Composition | `a + b`; affine chains collapsed | `CompositeTransform`, endpoints checked |
| Path-finding between spaces | Yes, shortest path | No, by decision ([core model](../architecture/core_model_design.md): "Implicit composition or a frame graph: hides direction and completeness errors") |
| Registration, landmark fitting | Yes; the purpose of the library | Out of scope |
| Nonlinear transforms | Piecewise-affine triangulation | Protocol allows them; none implemented |
| Invertibility | Required on every transform | A capability (`SupportsInverse`); refusal when absent |
| Units and spacing | None; spaces are voxel indices, voxel size is a `RescaleParametric` edge between nodes | Required CF/UDUNITS strings |
| Dimensionality | Exactly 3, positional | Any, by name; nonspatial dimensions by rule |
| Geometry through array operations | `ndarray_shifted` carries an origin only | Binding index carries or refuses |
| Persistence | `repr()` read back with `eval()`; SQLite graph | `rf.encode()` attributes; NGFF 0.6 |
| Lazy data | In memory | Dask |
| Dependencies | napari, PyQt6, imageio-ffmpeg required even headless | NumPy core; extras optional |

### Model-layer problems in CASTalign

Recorded because they are the failure modes xarrayrf's contract exists to prevent, and because
any adapter must not inherit them.

- **Stale origin after crop or transpose.** `ndarray_shifted.__array_finalize__`
  (`ndarray_shifted.py:55`) copies `origin` onto every view, so `img[5:]` and `img.T` keep the
  parent's origin. Derived from NumPy subclass semantics, not run.
- **`eval()` on load.** `Transform.load` (`base.py:179`) and `Graph._load_sqlite`
  (`graph.py:349-351`, `837`) evaluate file contents, so opening a shared `.tf` or `.db`
  executes arbitrary code. It also couples the file format to class names, which `compat.py`
  remaps for older files.
- **Silent path choice in cycles.** With a cycle, two paths between nodes may compose to
  different transforms; shortest path picks one after a printed warning.
- **Affine convention.** `points @ matrix - shift` (`base.py:619`): row vectors and a negated
  shift, the transpose of xarrayrf's `matrix @ x + translation`.

### Synergies

1. **CASTalign transforms as xarrayrf transforms.** `resample(..., transform=)` needs only
   `SupportsPoints` for the frame-to-frame step (target to source). A wrapper supplying
   `source`/`target` endpoints and `transform_point`, plus `inverse()` where CASTalign has an
   analytic one, would let fitted CASTalign transforms drive `rf.resample_to`. This fills
   xarrayrf's "no registration" and "no nonlinear transform" gaps without either entering the
   core. The wrapper must bind `zyx` to axis names explicitly and convert affines to
   `matrix.T` and `-shift`. Triangulation raises outside its simplices, which already matches
   the no-extrapolation rule.
2. **An explicit graph above the core.** The frame-graph rejection is about composition
   happening implicitly during arithmetic. An application-level graph whose nodes are
   `ReferenceFrame`s, whose edges are endpoint-checked transforms, and which returns a
   `CompositeTransform` on request is compatible with it, and could refuse ambiguous paths or
   check cycle consistency instead of warning. CASTalign's Graph is the working precedent.
3. **xarrayrf as CASTalign's geometry layer.** Units and physical spacing instead of voxel
   spaces joined by Rescale edges, named n-D dimensions, frame identity, geometry that survives
   crops, Dask, NIfTI/DICOM/NGFF input and data-only persistence.
4. **Resampling kernels.** Both pull through `map_coordinates`; CASTalign has a CuPy path,
   xarrayrf has Dask chunking and the `affine_transform` fast path. Shared code is possible
   but not worth pursuing before a release.

## Alternatives Considered

- **Depend on CASTalign for registration in the core:** its required dependencies include a Qt
  GUI stack, and its transforms are 3D-positional. An adapter outside the core keeps the NumPy
  floor.
- **Adopt a frame graph in the core because CASTalign shows it is useful:** usefulness was
  never in question; the objection is implicit composition. An explicit graph belongs in an
  application or a separate module, per synergy 2.

## Deferred Work

- An adapter prototype as an example (not in `src/`): wrap `LaminarTriangulation` and resample
  a framed array onto a target through it. Designed in the
  [transform adapters note](../adapters/transform_adapters_design.md).
- Whether an explicit frame graph should ship as an optional xarrayrf module or stay with
  applications. Decide when a second consumer asks for one.
- A CASTalign row in the [prior-art study](../architecture/prior_art_coverage_study.md).

## Next Steps

Reply on napari/napari#3783 proposing the layering above. If Max is
interested, build the adapter prototype first; it tests the `Transform` protocol against a
third-party nonlinear transform with an external inverse convention.
