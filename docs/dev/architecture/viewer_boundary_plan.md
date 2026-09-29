# Viewer boundary: what xarrayrf owes napari

**Status:** Active
**Last updated:** 2026-09-29
**Scope:** Core (`Geometry`, `resample`, units, endpoints) and the NGFF reader, as needed by a
napari integration. napari's own work is listed only to fix the boundary.

## Context

napari's transformations issue (napari/napari#3783) asks for named spaces, arbitrary
data-to-world transforms, dimension roles, slicing that keeps geometry, and resampling outside the
GPU. xarrayrf already covers the data side: endpoint-named transforms, nonspatial dimensions by
rule, rectangular affines for a plane in a volume, scalar selection that keeps the plane's
position, `positions_at` and `rf.resample_to`; the item-by-item assessment is the
[napari #3783 coverage review](../napari-3783/transformations_wishlist_review.md). The items below
are what only xarrayrf can supply.

## Current Decision

xarrayrf stays viewer-free: no camera, canvas, shader, render loop or chunk scheduling. It answers
geometry questions a viewer can ask cheaply. Work, highest priority first:

1. **Measure selection overhead.** napari selects a plane on every slider step. Benchmark scalar
   `isel` and crop on a framed array against a plain `DataArray` (a case beside
   `benchmarks/resample_benchmark.py`). The result decides whether napari slices through xarray
   or reads the geometry once and indexes the backing array.
2. **Show the core is backend-neutral.** The binding owns coordinates, not pixels, and the core
   imports no Dask. Unproven: the binding lifecycle over a lazily indexed non-Dask array (xarray's
   zarr engine with `chunks=None`) and over one array-API duck array. Add those test lanes.
   `resample` on a non-Dask lazy source currently reads the whole source.
3. **Pixel-free target domain and source footprint.** A target domain (origin, directions,
   spacing, shape) constructible without an array, and a query for the source index box it needs,
   so `resample` reads only that box. This enables oblique and out-of-core reslicing: napari picks
   the plane and fetches chunks; xarrayrf says which and can evaluate values on the CPU.
4. **Checked inversion on an embedded plane.** `positions_at` refuses to invert a plane embedded
   in a volume. A pick ray intersected with that plane needs the on-plane point mapped back to
   positions with an explicit on-plane tolerance, refusing off-plane points, which keeps the
   no-silent-projection rule.
5. **Unit normalization helper.** Strings stay the carrier. An optional `xarrayrf[pint]` helper
   would normalize spellings and convert at construction; napari already depends on pint. First
   check whether pint-xarray's index on quantified coordinates collides with the binding index
   owning the same coordinates.
6. **Value-positioned point sets.** Points, vectors and landmarks carry positions as values
   `(point, axis)`, not sample coordinates. They need a second binding kind declaring the endpoint
   the values are in, with transforms rewriting values. Ragged geometry is excluded; xvec is the
   precedent if needed.
7. **Multiscale read.** `ngff.open` returns one level. Read a pyramid as a `DataTree` with one
   binding per level, each level's mapping kept, never inferred from shape ratios.
8. **Time-parameterized transforms.** One transform per value of a nonspatial dimension (motion
   correction, video stabilization, EM section alignment): the contextual model the architecture
   reserves; v1 refuses it.

napari's side: rendering and shaders; deriving a plane from camera and dims; level choice and
chunk scheduling; ray casting; the transform chain across viewer spaces; which frame each canvas
shows and the default frame for layers that declare none; grid mode; the live editable store
behind Points and Shapes layers.

## Alternatives Considered

- **Shader code in xarrayrf transforms:** ties the core to one renderer; napari generates shaders
  from transform objects.
- **pint as the unit carrier:** quantities from different registries do not mix, and xarray
  carries strings.
- **Points and shapes as xarray objects throughout napari:** each insertion concatenates and
  rebuilds indexes; kept for interchange and analysis, not live editing.

## Deferred Work

Items 5 to 8 wait for a napari integration. The release gate (upstream xarray fixes and a
release) is a prerequisite for any of this reaching napari users; it is tracked under
`docs/dev/xarray-upstream/`.

## Next Steps

Items 1 and 2 are measurements and can start now; item 3 follows from item 2's result.
