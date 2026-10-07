# Cross-domain acceptance cases

**Status:** Implemented
**Last updated:** 2026-10-06
**Scope:** The DICOM, OME-NGFF, astronomy and GIS cases the model was tested against before its
API and persistence schema became commitments, and what they established. DICOM, NIfTI, NGFF
and GeoTIFF adapters are implemented; astronomy is not. No package release has shipped. [The core interface](../../core_interface.md)
is normative; this note records the cases and their worked numbers.

## Context

The model must work without importing a medical class hierarchy or requiring every mapping to be
a 3-D affine. Four producer ecosystems were used as acceptance cases:

| Concept | DICOM | NGFF | Astronomy | GIS |
|---|---|---|---|---|
| Named reference system | Frame of Reference UID | Name in container/group scope | Frame family plus required attributes | Validated CRS, possibly authority-identified |
| Coordinate representation | Ordered LPS in mm | Ordered typed axes and units | Spherical or Cartesian; angular conventions | CRS component order and units |
| Sample-to-world mapping | Orientation, spacing, per-frame positions | Selected transforms | Pixel-to-world WCS or sampled fields | Raster transform or calibrated coordinates |
| Nonspatial axes | Echo, time, signal component | Channel, time | Spectral, time (explicit roles) | Band, acquisition time |
| Domain responsibility | Ingest, calibration, export | Container hierarchy, encoding | Frame transforms, projection, distortion | CRS resolution, projection, raster resampling |

The table identifies analogous responsibilities, not identical terminology or equality semantics.

## Current Decision

### Architecture the cases support

A producer or optional adapter interprets domain metadata once and builds immutable declarations
(frame plus a transform from array coordinates). A DataArray owns its current coordinates and
values; the binding owns the lifecycle; adapters handle domain math and IO at explicit import and
export boundaries and are never called back after every operation. Pixel buffers stay lazy. No
plugin registry.

### What the cases established

- **Transforms act on retained coordinate values, not rebased indices.** After a crop the
  retained labels still feed the same transform; reusing it on fresh zero-based indices is wrong.
- **xarray indexes decide correspondence; the transform evaluates.** A transform may read
  auxiliary coordinates (for example slice offsets) that are not the alignment index. The DICOM
  example therefore uses physical offsets `(0, 2, 5)` mm as the slice labels: bare labels
  `slice=[0, 2]` then select offsets 0 and 2. With ordinal labels and an auxiliary offset
  coordinate, the same labels select offsets 0 and 5 (verified). Callers declare which labels they
  mean; there is no implicit world-coordinate join. Restricting transforms to dimension
  coordinates was rejected because it excludes curvilinear coordinates and useful representations.
- **Identity plus matching labels is not compatibility.** Same UID with different origins, and
  different UIDs with identical numbers, both refuse to combine.
- **Units:** `attrs["units"]` is the explicit unit channel and must match exactly; Cartesian
  frames refuse angular units.
- **A genuine single plane uses a 3x2 mapping** and needs no synthetic slice thickness. A stack
  with nonuniform offsets is still an affine in those offsets. Per-slice orientations need a
  richer mapping or separate arrays, never a fitted single affine.
- **Absent identity is minted locally,** never a universal "unknown" token.
- **NGFF geometry round trip and exact label round trip are different targets.** Standard export
  may rebase positional labels while composing the mapping so every sample location survives;
  keeping the original labels needs a coordinate-array encoding, and an exact-label request must
  refuse when unavailable.
- **NGFF identity:** independent stores with equal local system names stay unrelated; reuse is an
  explicit caller assertion (`frame=`), validated against available definition and axis data,
  never a grid-equality claim.
- **GIS import consumes the redundant carrier.** While framed, the binding is authoritative: import
  returns a new array without the active CF grid mapping, `spatial_ref`/GeoTransform carrier or
  conflicting transform attributes; export regenerates them from current geometry. A carrier
  shared by other variables needs Dataset-aware handling, never blind deletion. Pixel-corner
  versus centre offsets belong to the conversion. Reprojection goes through the provider and back.
- **Astronomy:** a sky frame is distinct from the WCS mapping. The adapter validates agreement,
  resolves parameterized frames, handles FITS axis order and pixel origin at the boundary, and
  snapshots a mutable WCS (including referenced distortion tables) into a versioned data form.
  Projection singularities and multi-solution inverses stay visible; nothing wraps angles.
  GWCS/ASDF codecs wait for an enumerated trusted data-tag contract.
- **Rasterix** (source inspection at `7899c7a2`): `AxisAffineTransformIndex.isel` keeps slices but
  drops scalar and fancy cases; `RasterIndex.isel` drops coupled x/y indexes; construction is
  width/height and corner-affine based; joins use bounding boxes and tolerances. Its separation of
  coordinate generation from index lifecycle is reusable; its equality and tolerance policy is not.

### Worked examples

These are the numbers `tests/test_point_oracles.py` checks (`rtol=0`, `atol=1e-12`, a
floating-point check, not a physical tolerance).

**Generic 2-D affine (A1).** Source `ArrayCoordinates(("row", "column"), ("1", "1"))`, target
`(x, y)` in mm, `matrix=((0, 0.5), (0.5, 0))`, `translation=(10, 20)`. Crop rows `10:30:2`; crop
index `(1, 3)` retains `row=12, column=3` and maps to `(x, y) = (11.5, 26.0)` mm. Evaluating the
rebased index `row=1` would wrongly give `y=20.5`.

**DICOM oblique stack (A3).** Target is a declared frame
`("dicom-frame-of-reference", uid)` with axes `L, P, S` in mm oriented
`right-to-left, anterior-to-posterior, inferior-to-superior`. Source `("slice", "row", "column")`
in `("mm", "1", "1")`; slice coordinates are physical offsets. The matrix columns are
`(normal, row_spacing * column_direction, column_spacing * row_direction)` with `translation` the
first pixel's position; increasing array rows follow the column direction, as DICOM names it.
With `row_direction = (0, 0, -1)` and `column_direction = (0.6, 0.8, 0)` the normal
`cross(row_direction, column_direction)` is `(0.8, -0.6, 0)`. With `row_spacing = 0.5`,
`column_spacing = 0.75` and `translation=(-10, 20, 30)`: `(slice=5, row=3, column=2)` maps to
`(-5.1, 18.2, 28.5)`, and offsets `(0, 2, 5)` at `row=column=0` map to `L = (-10, -8.4, -6)`,
`P = (20, 18.8, 17)`, `S = 30`: the third slice sits at offset 5, not an inferred spacing.

**NGFF channel affine (N1).** The NGFF transformation example maps `(c, j, i)` to `(c, y, x)` with
identity channel and `y = j + 2i + 3`, `x = 4j + 5i + 6`. After `isel(j=slice(10, 30, 2))`, new
row `r` is old `j = 10 + 2r`, so `y = 13 + 2r + 2i`, `x = 46 + 8r + 5i`. The point `(r=1, i=2)`
lands on `(y, x) = (19, 64)`; reusing the stored affine on `r` gives `(8, 20)`, which is wrong.

**Index conventions.** First sample centre 100 mm, spacing 2.5 mm: zero-based `i=3`, one-based
`j=4` and a lower-boundary declaration all give 107.5 mm when each convention is encoded once in
the declaration; no global half-voxel correction is applied.

### Acceptance matrix

| ID | Case | Criterion |
|---|---|---|
| A1 | 2-D affine; crop, transpose, scalar threshold in both orders | World points as above; placement and lazy pixels retained |
| A2 | Embedded plane from an oblique volume | Retained points match vector calculation; fixed coordinate survives; no silent off-plane projection |
| A3 | Slice offsets (0,2,5) mm; component algebra | Offsets read as declared; geometry kept only when compatible |
| A4 | Same UID, different origins; different UIDs, same numbers | Both refuse to combine |
| A5 | Mixed framed/bare operands, `where`, broadcasting, labelled subsets | Declarations checked before xarray loses evidence; no pixel computation during metadata handling |
| A6 | Required outcomes table in design §5 | Specified errors, both operand orders |
| A7 | Two overlapping crops of one array | Inner join keeps common labels, binding and correct points |
| A8 | Physical versus ordinal slice labels | As described above; no implicit substitution of auxiliary coordinates |
| A9 | Angular Cartesian units; explicit coordinate units | Refused; conflicting or unknown units refused |
| N1 | NGFF channel affine and crop | `(19, 64)`; export/import keeps world position |
| N2 | Two levels with nonzero translations | Declared transforms kept, never shape-derived scale |
| N3 | Equal local names in independent stores | Unrelated unless explicitly shared |
| N4 | Irregular gather; 2-D in 3-D | Exact round trip or format-specific refusal |
| W1 | Nonlinear `F(u, v) = (u, v + u²)`, `-1 <= u <= 1` | `(0.5, 0.25) -> (0.5, 0.5)`; outside the domain refused |
| W2 | Same frame family, different context; angles 0 vs 360 | No implicit equivalence, wrapping or reprojection |
| W3 | Sampled field with holes after outer alignment | No invented positions |
| G1 | Calibrated raster with CRS; identity round trip | Centres unchanged; no double scale; carrier regenerated on export |
| G2 | Rotated raster with corner-based affine | Centre positions preserved or explicit refusal |
| P1 | Valid, unknown, malformed payloads; missing provider | Distinct errors; no executable deserialization |
| P2 | Dataset with differently framed and unrelated variables | Association only for relevant variables |

## Alternatives Considered

- **Cloning a medical class hierarchy or making the NGFF image profile the core object:** both
  impose restrictions the other three domains do not share.
- **Banning auxiliary transform inputs:** simplifies rectilinear cases but excludes curvilinear
  coordinates and does not solve lifecycle enforcement.
- **A general unit engine, universal transform graph or domain-framework dependency:** not needed
  for the contract. Canonical alternatives and risks are in [the architecture](../../design.md).

## Deferred Work

- Astronomy adapter and the W cases; geodetic and angular representations first (see
  [prior-art study](prior_art_coverage_study.md)).
- Which A, N, G and P cases have executable tests beyond `tests/test_point_oracles.py` and the
  adapter suites was not audited for this note.
- Nonlinear and ASDF provider snapshots; exact-label NGFF round trip.
- Out of scope: a full astronomy or GIS stack in the core, a registration engine, silent
  cross-frame arithmetic, automatic DICOM export.

## Next Steps

1. Map each acceptance case to its test, and add tests for any case left uncovered.
