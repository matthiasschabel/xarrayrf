# napari #3783: transformations wish-list coverage

**Status:** Implemented
**Last updated:** 2026-10-06
**Scope:** How far xarrayrf covers the model side of napari/napari#3783 ("Transformations
Architecture") and the requests raised in its thread, and which parts belong to the viewer.
The work list that follows from it is the [viewer boundary plan](../architecture/viewer_boundary_plan.md).

## Context

napari/napari#3783 is jni's brain dump for napari's transform architecture: napari has four
explicit spaces (data, physical, world, grid), four implicit ones (tile, worldslice, vispy
world, camera), a data-to-physical transform limited to scale/translate/rotate/shear, an affine
physical-to-world transform, ad hoc accessors (`layer.data_to_world`, `layer._world_to_data`,
`layer.transforms[1:3]`), and slicing outside the transform machinery. The thread later asked
for a shared annotated-array type, with xarray plus flexible indexes as the preferred base.
xarrayrf was announced in the thread as that model layer; CASTalign was raised in reply and is
compared in [the CASTalign note](../castalign/castalign_comparison_review.md).

The test applied here is hygiene: a wish-list item is model-side when it concerns what the data
and its transforms mean, and viewer-side when it concerns rendering, camera, canvas or
interaction.

## Current Decision

xarrayrf covers the model side of the wish list except a named, sliceable chain container, and
stays viewer-free. The viewer-side remainder is napari's, as listed in the viewer boundary plan.

### The pitch items

| Wish-list item | xarrayrf (model side) | Remainder and owner |
|---|---|---|
| Everything in the transform chain, even slicing and viewing | Slicing: yes. A selected plane keeps its position as a scalar coordinate, giving a rectangular data-to-world affine. Viewing: excluded by design | napari: the chain through worldslice, canvas and vispy spaces |
| Elements add to the chain as their transforms become available; each layer's chain copied to the canvas | Supplies the data-to-frame and frame-to-frame links, immutable and composable (`CompositeTransform`) | napari: assembling and copying the chain across viewer spaces |
| All spaces named; the chain reports its extent; `layer.transforms['data':'world']` | Naming: stronger than asked; endpoints carry frame identity, axes, units and directions. Extent queries and slicing by space name: no, because there is no chain object | Open. napari's chain, or an explicit graph above the core (CASTalign's `g["a":"b"]` is the precedent) |
| All transforms invertible except the camera projection | Invertibility is a capability (`SupportsInverse`); exact inverse or refusal. An embedded plane refuses; checked on-plane inversion is boundary plan item 4 | napari: how picking handles a refusal |
| Users and plugins supply any transform satisfying invertibility and providing GLSL `map`/`imap` | Arbitrary transforms: yes, structural protocols with no registry. GLSL: excluded (see Alternatives in the boundary plan) | napari: shader generation from transform objects |
| Non-orthogonal slicing: a z/y shear updates the sliced translation per slice | Yes; a plane from a sheared volume carries the correct per-slice offset without special handling | napari: per-slice GLSL for out-of-memory non-orthogonal slicing |

### Requests raised in the thread

| Request | xarrayrf | Remainder |
|---|---|---|
| Per-frame or per-tile (n−k)-D affines: video stabilization, montages, EM section alignment (jojoelfe, jni) | Refused in v1; boundary plan item 8 (contextual transforms) | GPU storage of per-frame parameters: napari |
| Explicit dimension names and types (andy-sweet; napari #1904, #2487, #2348) | Yes; spatial dimensions named by the binding, others nonspatial by rule, transforms applied by axis name | None |
| Resampling for processing, not only display (andy-sweet's template-space plugin case) | Yes; `resample` and `rf.resample_to`, lazy on Dask | None |
| Physical vs world as separate spaces (the NGFF discussion) | Yes; an explicit frame-to-frame transform. NGFF 0.4 to 0.6 reader | None |
| Per-voxel displacement fields (jojoelfe) | Deferred with field-backed transforms | Bounds and inverse policy, when implemented |
| Tiled and chunked sources; a translation per tile (perlman) | Dask-backed arrays carry the binding; pixel-free domain and source footprint are boundary plan item 3 | Chunk scheduling: napari |
| Slicing as a rectangular matrix (perlman, andy-sweet re #3410) | Yes; a rectangular affine keeps the dropped axis's position | None |
| Data-to-canvas direction not needed, only canvas-to-data (nclack) | Both directions where they exist; neither required of a user transform beyond `transform_point` | None |
| Grid mode | Not model-side | napari; andy-sweet notes it depends on layer data and the preceding chain |
| One annotated-array type across fields (jni: ndcube, scipp) | Built on xarray with a custom index, as jni argued | xarray upstream fixes before release |

### Gap on the model side

The only model-side item not covered is the chain container: extent queries and slicing by
space name. xarrayrf provides typed links but deliberately no graph or chain that composes them
implicitly. Whether napari owns that container, or an explicit (never implicit) graph ships above
the core, is undecided; the CASTalign note records the precedent and the constraint.

## Alternatives Considered

- **Model napari's viewer spaces (worldslice, canvas) as xarrayrf frames:** they are
  per-canvas, mutable and camera-dependent; frames are immutable identities of the spaces data
  lives in.
- **Require invertibility, as the wish list does:** would exclude embedded planes and
  projections that are legitimate data geometry; refusal at the point of inversion is enough for
  picking.

## Deferred Work

- Decide the owner of the named chain container (napari, or an optional explicit graph).
- Revisit the thread when napari opens a NAP for multi-canvas frame selection.

## Next Steps

The boundary plan's items 1 and 2 (selection overhead and backend neutrality) are the
measurements napari would need before depending on xarrayrf; they are unchanged by this review.
