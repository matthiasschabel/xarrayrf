# Prior-art and coverage study

**Status:** Implemented
**Last updated:** 2026-10-07
**Scope:** How existing libraries model frames, coordinate systems and transforms; which of their
use cases xarrayrf represents; what to reuse and what to avoid. Companion to the
[core model](core_model_design.md).

## Context

The goal is coverage, not replication or dependency: xarrayrf should represent what these
libraries do, convert to and from their representations through adapters, and keep well-founded
deliberate differences. Each library was installed in a scratch environment, probed and read at
source. Behaviours marked *verified* were run; the rest come from reading the cited source.

| Group | Versions | Licenses |
|---|---|---|
| Astropy modeling, GWCS, coordinates | astropy 8.0.1, gwcs 1.0.3, asdf 5.4.0, asdf-astropy 0.11.0 | BSD-3 |
| Geospatial | rioxarray 0.23.0, xproj 0.2.1, rasterix 0.2.2, pyproj 3.8.0, rasterio 1.5.1 (GDAL 3.12.4) | Apache-2.0 (xproj helpers from geopandas, BSD-3) |
| Imaging | nibabel 5.4.2, SimpleITK 2.5.6, itk 5.4.7, VTK 9.7.0 | MIT, Apache-2.0, BSD-3 |
| NGFF | spatialdata 0.8.0, ngff-zarr 0.47.0, ome-zarr-models 1.8.1, NGFF 0.6 | BSD-3, MIT, MIT |

Probes ran against NumPy 2.5 and xarray 2026.7.0. Probe scripts were not kept.
These are dated observations from the 2026-09-28 study, not claims about subsequent releases.

## Current Decision

### Findings that changed xarrayrf

1. **CF angular spellings passed the Cartesian check** (verified): `degrees_east`,
   `degrees_north`, `degree_east`, `degree_N`, `arc_degree`. The angular spelling list was
   extended.
2. **ITK's tensor rules differ from J·T·Jᵀ.** `TransformSymmetricSecondRankTensor` computes
   J·T·J⁻¹ (correct for mixed (1,1) tensors): under shear, `diag(3,2,1)` comes back unchanged,
   whereas J·T·Jᵀ gives `xx=5, xy=2` (verified). `TransformDiffusionTensor3D` applies PPD with
   the inverse Jacobian because ITK maps fixed to moving. xarrayrf's `second_rank_tensor`
   deliberately differs; an ITK adapter swaps endpoints or inverts explicitly.

### Failure modes the design already forbids

| Failure | Where observed (verified) |
|---|---|
| Stale transform after slicing | rioxarray (GeoTransform after strided `isel`); spatialdata (`attrs["transform"]` after crop, arithmetic, `mean`) |
| Conflicting frames combine silently | rioxarray (EPSG:32633 + EPSG:4326 keeps the first CRS); xproj (two CRS coordinates under different names) |
| Identity lost, then "missing equals anything" | rasterix (joins and concat return CRS `None`, which matches any CRS) |
| `join="override"` rewrites identity | xproj (EPSG:4326 replaced CRS84) |
| Tolerant or identity-only equality | spatialdata (`np.allclose`, unhashable); rasterix (`rtol`); GWCS (frames unequal to themselves after ASDF round trip) |
| Implicit unit conversion | Astropy (`Shift(1 mm)(1 m) = 1001 mm`; `u.spectral()` reinterprets THz as µm) |
| Unchecked or infinite inverses | Astropy (user inverse unchecked; `Scale(0).inverse` gives `inf`) |
| Regular grid inferred from two samples | rasterix `assign_index` rewrites irregular coordinates and deletes the caller's `GeoTransform` |
| Unmentioned axes pass through | spatialdata `to_affine_matrix`; ome-zarr-models `ByDimension` |
| Composition order opposite to reading order | ITK, SimpleITK, VTK PreMultiply apply the last-added transform first |
| Bare orientation codes disagree | ITK legacy `RAI` equals DICOM `LPS` |
| Orientation label hides obliquity | nibabel `aff2axcodes` labels a column 71.6° from its axis; ngff-zarr takes the dominant component without collision check or residual |

### Coverage

"Yes" means represented in the current core; "adapter" is downstream conversion.

| Use case | Core | Missing core capability | Adapter |
|---|---|---|---|
| Voxel/pixel-to-world affine (NIfTI, ITK, VTK, GeoTransform, NGFF, Astropy linear) | Yes | | Corner-to-centre and index conventions applied once |
| Axis permutation and projection (NGFF `mapAxis`/`projectAxis`, Astropy `Mapping`) | Yes, 0/±1 or non-square affine | | Re-emit compact forms on write |
| LPS↔RAS, PSL→RAS, CRS84↔EPSG:4326 axis order | Yes, `coordinate_system_change` | | Vocabularies |
| nibabel `ornt` arrays | Yes, `M[ornt[i,0], i] = ornt[i,1]` | | Conversion |
| Sequences (NGFF `sequence`, Astropy `\|`, GWCS pipeline), ITK/VTK composites | Yes, `CompositeTransform` | | Synthesize frames for anonymous NGFF intermediates; reverse ITK order |
| Point, vector, covariant vector, J·T·Jᵀ, polar rotation | Jacobian only | Rule callables (specified, not implemented) | |
| ITK J·T·J⁻¹ and PPD-with-J⁻¹ | | | Downstream rules |
| Frame identity (EPSG/WKT/PROJJSON, NIfTI xform codes, NGFF names) | Yes, `(namespace, value)` + `definition` | | Canonical identifier; equivalence only by explicit normalization |
| Projected CRS axes in metres | Yes | | Unit spelling from pyproj `cs_to_cf` |
| Geographic lon/lat, celestial sky | No: Cartesian refuses angles | Geodetic/spherical representations | Identity from pyproj or `CelestialFrame` |
| Spectral, temporal, Stokes axes; NGFF axis types; unitless axes | Yes, `axis_types` and `None` units | | NGFF `discrete`, `longName` deferred |
| Mixed frames (sky + spectrum) | No | Mixed-representation systems | GWCS `CompositeFrame` |
| Per-axis transforms (Astropy `&`, NGFF `byDimension`) | Only when every part is affine | Product transform | |
| Declared valid domain (GWCS bounding box) | Transforms may raise; no domain object | Declared domain that raises outside | NaN fill stays adapter policy |
| Analytic inverse | Affine and composite | | |
| Numerical or declared inverse (GWCS, NGFF `bijection`) | No | Separately named approximate inverse | |
| Nonlinear CRS change, displacement fields, B-splines, GCP/RPC | Protocol accommodates | `jacobian(at)` implementations; by-reference arrays in persistence | pyproj, ITK wrappers |
| Perspective (`vtkPerspectiveTransform`) | No | Projective transform | |
| Separability matrix | No | Optional query from Jacobian sparsity | |
| APE 14 shared WCS API | No | | View over `Geometry` |
| Persistence (ASDF, CF `grid_mapping`, NGFF 0.6) | `encode`/`decode`; NGFF via `xarrayrf.ngff` | | ASDF and CF converters unwritten |
| Reprojection and tolerant joins | No, by design | | Explicit resampling |
| Live pipelines (VTK) | Deliberately unsupported | | |

### Reuse candidates

| Source | What | License | Use |
|---|---|---|---|
| rasterix | Per-axis `AxisAffineTransformIndex` slicing | Apache-2.0 | Reference for binding selection |
| xproj | `ProjIndexMixin`: duck-typed `crs`, set/convert hooks, override refused unless allowed | Apache-2.0 | Model for a frame-aware index hook |
| ome-zarr-models | Pydantic NGFF 0.6 models | MIT | NGFF serialization layer |
| ngff-zarr | RFC-4 values and validation; ITK↔NGFF conversion | MIT | NGFF and RFC-4 adapters |
| Astropy | Separability concept; `fix_inputs`; ASDF data-only tags; UCD1+ types | BSD-3 | Design references; ASDF adapter |
| ITK, VTK, nibabel | Numerical cases below | Apache-2.0, BSD-3, MIT | Development-only oracles |

Astropy modeling could sit behind the `Transform` protocol as an optional adapter: points always,
inverse only for analytic inverses without a user override, Jacobian only for recognized linear
leaves. Its models compare by identity, so a bound model needs an adapter-derived structural key.

### Deliberate differences

- Exact structural equality and hashing; equivalence only through explicit normalization.
- Endpoints checked by frame and axes, not arity (Astropy) or addition order (ITK, VTK).
- First-to-last composition (Astropy `|`, NGFF `sequence`, spatialdata `Sequence`).
- No implicit unit conversion.
- Input Jacobians and quantity rules, which only ITK has, with different tensor semantics.
- Unnormalized covariant vectors (ITK), not VTK's normalized normals.
- `inverse()` only where determined; raise outside a domain rather than extrapolate.
- Transforms act on coordinate values, so crop and stride keep them valid, removing the stale
  transform failure of rioxarray and spatialdata.
- Frame identity separate from coordinate system, which NGFF, spatialdata and pyproj fuse.
- Per-vector axis codes with residual angles; immutable value objects; no live pipelines.

### Development-only oracles

All verified against the named libraries:

1. `CompositeTransform(translate(+1 x), scale(2))` at (1,0,0) gives 4; ITK, SimpleITK and VTK
   PreMultiply give 3 for the same list, VTK PostMultiply gives 4.
2. Shear J = [[1,1,0],[0,1,0],[0,0,1]]: `covariant_vector((1,0,0)) = (1,−1,0)` (ITK); VTK's
   normalized form is (1,−1,0)/√2; `vector((0,1,0)) = (1,1,0)`.
3. Same shear, T = diag(3,2,1): `second_rank_tensor` = [[5,2,0],[2,2,0],[0,0,1]]; ITK's
   `TransformSymmetricSecondRankTensor` returns diag(3,2,1).
4. Same shear, T = diag(2,3,1): `polar_rotation` upper triangle [2.2, 0.4, 0, 2.8, 0, 1]; PPD
   with J gives xy = +0.5, ITK's PPD with J⁻¹ gives xy = −0.5.
5. Index (1,2,3) with origin (10,20,30), spacing (2,3,4), direction [[0,−1,0],[1,0,0],[0,0,1]]
   maps to (4,22,42) in ITK and VTK.
6. Rotation 90° about z with centre (10,0,0) and ITK translation (1,2,3) has offset (11,−8,3):
   xarrayrf's `translation` equals ITK's offset.
7. nibabel `ornt_transform(PSL→RAS)` = [[1,−1],[2,1],[0,−1]] matches the xarrayrf signed
   permutation.
8. 60° rotation about z: `aff2axcodes` gives ('A','L','S'); `axis_codes` of column 0 ranks
   anterior first at 30° and x second at 60°.

### xarray flexible indexes

Reviewed against xarray 2026.7.0 and the "Flexible indexing" blog post (2025-09-02).
`xarray.indexes.CoordinateTransform` maps integer positions to labels and back, carries the shape,
is experimental, and backs `CoordinateTransformIndex`: lazy coordinates, `sel` only with
`method="nearest"`, exact alignment, dropped on any `isel`. `RangeIndex` survives slicing and
compares with `np.isclose` by default. This is index machinery, not a frame model: no identity,
units, orientation or frame-to-frame transforms, and position-based, sometimes tolerant equality.

- **Adopted:** `Geometry.frame_coordinates()`, lazy coordinates through a
  `CoordinateTransformIndex` subclass that survives slicing and aligns exactly.
- **Adopted:** a `RangeIndex` coordinate contributes its declared step to lattices and lookups.
- **For the binding:** default `RangeIndex` equality is tolerant, so slightly different arrays
  align as identical under plain xarray.
- **Confirmed:** transforms on coordinate values need no per-axis index rebuild under selection.
- **Not adopted:** subclassing `CoordinateTransform` in the core (imports xarray, carries a shape,
  inherits tolerant equality).

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Astropy modeling or GWCS as the engine | No input Jacobians or quantity rules; arity-only composition; identity equality; implicit unit conversion |
| ITK or VTK | Large binaries, no named axes; different tensor semantics; VTK objects are live |
| spatialdata transformations | Go stale under xarray operations; tolerant unhashable equality; fixed axis names; units dropped |
| pyproj CRS as frame identity | Fuses identity with axis order; equivalence depends on the PROJ database |
| Wrap all of them | A dependency tangle; contradicts the NumPy-only core |

## Deferred Work

- Candidate core capabilities, unprioritized: geodetic and spherical representations, product
  transform, declared domain, named approximate inverse, separability query.
- Adapters: CF `grid_mapping`, pyproj, ITK/SimpleITK, nibabel, Astropy/ASDF, APE 14 view.
- A frame-aware index hook (xproj pattern) with the binding work.

## Next Steps

1. Choose the priority of the candidate core capabilities and record it in the
   [core model](core_model_design.md). Angular representations are the likely first stage (see
   [relativity notes](relativity_notes.md)).
