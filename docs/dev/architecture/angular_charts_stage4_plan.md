# Angular charts stage 4: a minimal CF reader

**Status:** Implemented (2026-10-12; plan reviewed by Codex gpt-6-sol)
**Last updated:** 2026-10-12
**Scope:** A new adapter, `xarrayrf.cf`, that frames a variable of an xarray Dataset following
the CF conventions: rectilinear 1-D longitude and latitude, the earth figure from a
`latitude_longitude` grid mapping, and cell bounds as intervals. Implements section 5 of the
[angular charts design](angular_charts_design.md). Projected grid mappings, rotated poles and 2-D
(curvilinear) coordinates are refused, not supported.

## Context

Stages 1 to 3 let a longitude/latitude grid be declared, converted to Earth-centred Cartesian
and resampled across seams and poles, but a user must build the frame and `Grid` by hand. Most
such data arrives as CF netCDF or Zarr opened with `xarray.open_dataset`: the coordinates and
their `units`, `standard_name` and `bounds` are already on the Dataset, and the figure is on a
grid-mapping variable. The reader therefore takes a Dataset, not a file, and needs no format
library (xarray is already a dependency), so it has no extra.

Adapter conventions (`docs/core_interface.md`, Adapters): metadata import without reading
pixels, `frame=` to adopt an existing identity through `adopt_frame`, a `Report` of every
normalization, the core never importing the adapter.

## Current Decision

### API

```python
import xarrayrf.cf as rfcf

ds = xr.open_dataset("tas_Amon_model_historical.nc")
tas = rfcf.frame_variable(ds, "tas")  # framed DataArray, data still lazy
geometry = rfcf.from_dataset(ds, "tas")  # metadata only
tas2 = rfcf.frame_variable(other_ds, "tas", frame=tas)  # same model: share the identity
```

- `from_dataset(dataset, variable, *, frame=None) -> CfGeometry`: reads metadata only.
  `CfGeometry` holds `grid` (the `Grid` over the longitude and latitude dimensions), `frame`,
  `longitude` and `latitude` (coordinate names) and `report`.
- `frame_variable(dataset, variable, *, frame=None) -> xr.DataArray`: the Dataset's variable,
  framed on that grid with `replace_coordinates=False`, so its other coordinates (time,
  pressure level), attributes and lazy data are kept.

### Finding longitude and latitude

Among the variable's one-dimensional coordinates, dimension coordinates and CF auxiliary
coordinates alike (xarray decodes the `coordinates` attribute into the variable's coords), a
coordinate is longitude when its `standard_name` is `"longitude"` or its `units` is a CF
longitude spelling (east or west); latitude likewise (`"latitude"`, north or south). Exactly one
of each is required, each along one of the variable's dimensions, and the two along different
dimensions; a name may differ from its dimension (`Grid` maps coordinate names to dimensions).
The reason is named otherwise:

| Situation | Outcome |
|---|---|
| none, or more than one, of either | `ValueError` naming the candidates |
| `standard_name` `grid_longitude`/`grid_latitude`, or grid mapping `rotated_latitude_longitude` | `ValueError`: rotated poles are not supported |
| longitude/latitude present only as 2-D auxiliary coordinates | `ValueError`: curvilinear grids are not supported |
| units missing or not an accepted angular spelling | `ValueError` naming the coordinate and its units |

West and south units (`degrees_west`, `degrees_south` and variants) keep their labels: the
array coordinate takes that unit and the transform's matrix entry is −1 into the frame's
east/north axis, so labels are never rewritten and the frame stays east/north. Reported as a
normalization.

### Frame and figure

The frame is geodetic, with two axes named after the coordinates, in (longitude, latitude)
order, units `degrees_east` and `degrees_north` (or `rad` when the coordinates are in radians),
roles longitude and latitude. A spherical figure is geodetic with zero flattening (design,
section 3), so one representation serves both.

The figure comes from the variable's `grid_mapping` attribute, the name of a variable whose
`grid_mapping_name` is `"latitude_longitude"`:

| Attributes | Figure (definition keys) |
|---|---|
| `earth_radius` | `figure="sphere"`, `semi_major_axis=earth_radius` |
| `semi_major_axis` and `inverse_flattening` | `figure="ellipsoid"`, both (sphere when `inverse_flattening` is 0) |
| `semi_major_axis` and `semi_minor_axis` | ellipsoid with `inverse_flattening = a / (a − b)`, or sphere when equal |
| `semi_major_axis` alone | sphere |
| `semi_major_axis`, `semi_minor_axis` and `inverse_flattening` | as above, when they agree: equal semi-axes with `inverse_flattening` 0 is a sphere (decided before any ratio, which would divide by zero); otherwise `a / (a − b)` must match `inverse_flattening` to 1e−9 relative; refused otherwise |
| none of these (CF defines `latitude_longitude` without mandatory parameters) | no figure: treated as a missing figure below |

`length_unit` is `"m"` (CF figure parameters are in metres). `horizontal_datum_name` becomes
`datum_name` and `reference_ellipsoid_name` becomes `ellipsoid_name`: CF defines them as
different properties, and both join the frame definition when present. A nonzero
`longitude_of_prime_meridian`, the extended `grid_mapping` syntax (`"crs: lat lon"`), a missing
grid-mapping variable, a projected mapping or contradictory parameters (`earth_radius` with
`semi_major_axis`) are refused with the attribute named.

Identity: the frame is anonymous (`ReferenceFrame.anonymous`), carrying the figure in its
definition. A shared figure never implies a shared frame (design, plan structural rules); two
files of one model share a frame only when the caller passes `frame=`.

Without `grid_mapping`, or with a `latitude_longitude` mapping that declares no figure, CF leaves
the figure unspecified, so the reader refuses unless `frame=` is given, and says so. With `frame=`, the imported (figure-less) frame is replaced through
`adopt_frame`, which requires the caller's frame to be an equivalent coordinate system.

### Bounds

A coordinate's `bounds` attribute names an `(n, 2)` variable. `Grid` intervals need each row as
`(low, high)` and one sample offset per axis (`label = low + offset · (high − low)` for every
cell, as `freeze_intervals` checks), while CF orders bounds with the coordinate (descending
coordinates have descending bounds). The reader therefore:

1. sorts each pair to `(low, high)`;
2. computes each cell's offset `(label − low) / (high − low)`, and uses their mean as the
   axis's `sample_offset` when every cell agrees within `freeze_intervals`' tolerance;
3. otherwise declares no intervals for that axis and reports `bounds-off-centre`: labels at
   varying positions within their cells (Gaussian latitudes are the common case) cannot be
   expressed by one offset, and dropping the bounds is safer than refusing common data.

Missing bounds are not inferred. Declared bounds are what stage 3 uses to decide global coverage
exactly; an axis whose bounds were dropped falls back to the label rule.

### Report

Entries `(code, message)`: `figure` (where it came from), `west-longitude` / `south-latitude`
(sign normalizations), `no-bounds` per axis without bounds, `bounds-off-centre` per axis whose
bounds were dropped, `figure-ignored` when `frame=` replaces a declared figure.

### Documentation

`docs/core_interface.md` Adapters section (the CF reader, its scope and refusals),
`CHANGELOG.md` Added entry, design next steps, plan status, changelog entry; the import-boundary
test's adapter list.

### Tests (`tests/test_cf.py`)

1. A synthetic CMIP-style dataset (`time × lat × lon`, 2.5° global, `lat_bnds`/`lon_bnds`,
   `grid_mapping` with `earth_radius = 6371229`): framed, the frame is geodetic with the sphere
   in its definition, time is carried, data stays lazy (dask), and a resample across the seam
   onto a −180…177.5 grid matches the analytic field (design acceptance case 1).
2. Each figure form in the table, including `inverse_flattening = 0`, equal semi-axes, all three
   ellipsoid parameters agreeing and disagreeing, and `datum_name` and `ellipsoid_name` kept
   apart.
3. Each refusal in the two tables, with the attribute or coordinate named.
4. `degrees_west` longitude and `degrees_south` latitude: labels kept, the frame east/north,
   values match the east/north equivalent at the same points.
5. Descending latitude (90…−90) with CF-descending bounds; off-centre bounds (Gaussian-style
   latitudes) dropped and reported; `standard_name`-only identification; 1-D auxiliary
   longitude/latitude named differently from their dimensions.
6. No `grid_mapping`, and a `latitude_longitude` mapping without figure parameters: refused
   without `frame=`; with `frame=` adopted, and `figure-ignored` reported when a figure was
   declared. A caller frame whose units are spelled differently (`deg` rather than
   `degrees_east`) is refused by adoption, which never relabels units.
7. Two datasets with the same figure get different frames; `frame=` shares one, and resampling
   between them needs no transform.
8. A netCDF round trip through the scipy engine (no extra dependency) reads back identically.
9. The import-boundary test covers `xarrayrf.cf` (adapter imports only the core).

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Reading files (`open(path)`) | xarray already opens CF netCDF, Zarr and OPeNDAP; taking a Dataset avoids engine choices |
| A spherical representation for sphere figures | Height is measured from the surface in model data; geodetic with zero flattening is the same chart and keeps one code path |
| Rewriting west/south labels to east/north | Breaks literal labels; a −1 matrix entry expresses the same thing |
| Inferring bounds from labels | Bounds decide coverage and conservative remapping; guessing them would hide the ambiguity stage 3 documents |
| Shared identity from a shared figure | Rejected by the plan's structural rules: a datum or radius does not identify a frame |
| pyproj for the figure | Not needed for `latitude_longitude`; pyproj belongs to the stage-4 provider adapters of the nonlinear plan |

## Deferred Work

- Projected grid mappings and rotated poles (pyproj; nonlinear plan stage 4).
- 2-D latitude/longitude (curvilinear; nonlinear plan stage 3).
- Vertical coordinates and `formula_terms`.
- CF export (writing `grid_mapping` and bounds from a frame).

## Next Steps

1. Plan review (Codex), implement with the tests above, implementation review, QA, commit and
   push.
2. Then value geometry stage 4 (`FrameField`), now unblocked.
