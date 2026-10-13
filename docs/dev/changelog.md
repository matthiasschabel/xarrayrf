# Changelog

## [2026-10-13] — Field transforms, step 4: ITK and ANTs importers

- **Problem**: ITK and ANTs registration results (transform files, warps, inverse warps) could
  not be read into xarrayrf; the design's choice of `nitransforms.io` turned out to parse `.tfm`
  parameters and centres as float32, ignore text transform types and support 3-D only.
- **Resolution**:
  - `xarrayrf.itk` (`_files.py`, `_import.py`): direct float64 readers for text, MATLAB, HDF5
    and NIfTI files, composites in ITK's last-added-first order, linear types and displacement
    fields, NIfTI units and xform rules.
  - `xarrayrf.ants`: the warp-then-affine chain with the inverse warp as a
    `DeclaredInversePair`, so `declared_inverse()` is ANTs' inverse.
  - `_transform_import.py`: the frame joining shared by the elastix, ITK and ANTs importers; a
    single imported member is returned as itself.
  - SimpleITK and antspyx fixture from `tools/build_itk_transform_fixture.py`.
  - Plan reviewed by Codex: NIfTI units and xform rules, the 2-D projection, and a fixture
    matrix per reader.
- **Files affected**: `src/xarrayrf/itk/`, `src/xarrayrf/ants/`, `src/xarrayrf/_transform_import.py`,
  `src/xarrayrf/elastix/_import.py`, `tests/test_itk.py`, `tests/test_ants.py`,
  `tests/test_elastix.py`, `tests/test_import_boundary.py`, `tests/fixtures/itk/`,
  `tools/build_itk_transform_fixture.py`, `pyproject.toml`, `uv.lock`, docs.
- **Reviewed by**: Codex gpt-6-sol (reasoning effort high)

## [2026-10-13] — Field transforms, step 3: the elastix importer

- **Problem**: elastix results could only be applied through Pirana or ITK; xarrayrf had no
  importer, and its B-spline could not express elastix's support.
- **Resolution**:
  - `xarrayrf.elastix` (`_parameter_files.py`, `_import.py`), adapted from Pirana's parser:
    parameter files and chains, translation, 2-D and 3-D Euler, affine and cubic B-spline
    stages, `space` (LPS mm by default in 3-D) joined to the caller's frames by
    `coordinate_system_change`, local intermediate frames, and a report.
  - `BSplineTransform(support="half-open")`, measured against transformix: elastix excludes the
    upper face.
  - transformix fixture from `tools/build_elastix_fixture.py` (itk-elastix 0.25.4).
  - Found while building it: elastix's `RecursiveBSplineTransform` evaluates a nonzero
    `GridIndex` inconsistently, so it is refused.
  - Plan reviewed by Codex: refuse a first map naming an initial transform; face probes cover
    the B-spline stage alone.
- **Files affected**: `src/xarrayrf/elastix/`, `src/xarrayrf/_field_transform.py`,
  `tests/test_elastix.py`, `tests/test_field_transforms.py`, `tests/test_import_boundary.py`,
  `tests/fixtures/elastix_transforms.json`, `tools/build_elastix_fixture.py`,
  `docs/core_interface.md`, `README.md`, `CHANGELOG.md`.
- **Reviewed by**: Codex gpt-6-sol (reasoning effort high)

## [2026-10-13] — Field transforms, step 2: declared inverse pairs

- **Problem**: A field transform has no exact inverse, so a registration result could be
  applied in one direction only, although tools such as ANTs write the inverse warp too.
- **Resolution**:
  - `DeclaredInversePair` owns both directions (`_declared_inverse.py`), so neither transform
    refers to the other. Its `PairedTransform` directions expose a Jacobian exactly when the
    member does, through a private subclass, so the runtime capability checks stay truthful.
  - `SupportsDeclaredInverse`; `CompositeTransform.declared_inverse()` prefers exact inverses
    and checks that each inverse reverses its member's endpoints.
  - `round_trip_residual_report` (`_round_trip_residual.py`) validates both directions'
    results before subtracting.
  - Plan reviewed by Codex. Findings: validate user results, check each inverse's endpoints,
    accept undeclared units as one unit, refuse affine members. The design's half equality is
    revised: directions compare their pairs, so equal directions have equal declared inverses.
- **Files affected**: `src/xarrayrf/_declared_inverse.py`, `_round_trip_residual.py`,
  `_composite.py`, `_transform.py`, `_encoding.py`, `__init__.py`,
  `tests/test_declared_inverse.py`, `docs/core_interface.md`, `CHANGELOG.md`.
- **Reviewed by**: Codex gpt-6-sol (reasoning effort high)

## [2026-10-12] — Field transforms, step 1: dense and B-spline displacement transforms

- **Problem**: Only affine transforms existed. Deformable registration results could not be
  applied, differentiated or checked for folding (stage 1a of the field transform design).
- **Resolution**:
  - `DisplacementFieldTransform` and `BSplineTransform` (`_field_transform.py`) on an admissible
    `Grid`.
  - ITK supports, compared exactly: dense `[lo, hi)` over tiling cells, B-spline `[1, n − 2]`.
  - `"raise"`, `"identity"` and `"nearest"` exteriors.
  - One separable float64 kernel per representation for both values and Jacobians, in blocks of
    2**16 points.
  - Bytes-backed float32 and float64 storage.
  - `jacobian_determinant_report`, `sample_displacement_field`, and an explicit `encode`
    refusal until 1b.
  - ITK 5.4.7 boundary fixture from `tools/build_itk_field_fixture.py`.
  - Plan and implementation reviewed by Codex. Findings: exact edges rather than slack,
    direction-aware interval tiling, B-splines restricted to exact-step grids, and an immutable
    report.
- **Verification**: stock 2,570 passed plus 47 xfailed; patched 2,617; upstream 2,575 plus 42
  xfailed; ruff and mypy pass.
- **Files affected**: `src/xarrayrf/_field_transform.py`, `src/xarrayrf/_jacobian_determinant.py`,
  `src/xarrayrf/_encoding.py`, `src/xarrayrf/__init__.py`, `tests/test_field_transforms.py`,
  `tests/fixtures/itk_field_boundaries.json`, `tools/build_itk_field_fixture.py`,
  `pyproject.toml`, `docs/core_interface.md`, `CHANGELOG.md`,
  `docs/dev/architecture/field_transform_design.md`,
  `docs/dev/architecture/field_transform_step1_plan.md`.
- **Reviewed by**: Codex (`gpt-6-sol`, reasoning effort high).

## [2026-10-12] — Independent QA of value geometry stage 4 (FrameField)

- **Problem**: Review pointwise bases (`c8d704b`) for regressions, edge cases and error paths
  beyond the Codex rounds.
- **Resolution**: Probes confirmed pickling, `isel`, Dataset membership, persistence and lazy
  component resampling. Three gaps fixed. (1) A per-sample change of basis put the geometry
  dimensions among the core dimensions, so a Dask array chunked along them (spatially chunked
  wind) raised xarray's core-dimension error: the maps are now passed as their own inputs over
  the geometry dimensions, which stay loop dimensions. (2) An undeclared scalar with a frame
  field as `from_basis` failed with an unrelated "two different ArrayCoordinates" message; the
  constant basis map now refuses fields with the reason. (3) After `assume_frame`, the stale
  basis error named only `ArrayCoordinates`; it now covers frame fields. Regression tests for
  each.
- **Verification**: stock 2,500 passed plus 47 xfailed; patched 2,547; upstream 2,505 plus 42
  xfailed; ruff and mypy pass.
- **Files affected**: `src/xarrayrf/_tensor_components.py`, `src/xarrayrf/native.py`,
  `tests/test_frame_fields.py`, `tests/test_tensor_components.py`, `docs/core_interface.md`.
- **Reviewed by**: Claude Opus 5.5 (`claude-opus-5-5`) applying QAEngineer.

## [2026-10-12] — Value geometry stage 4: pointwise bases (FrameField)

- **Problem**: Component bases were constant (a frame or the binding's array coordinates), so
  eastward/northward wind, the most common earth-system vector, could not be declared, and
  declared components could not cross the poles when resampling.
- **Resolution**: `FrameField` protocol (`_frame_field.py`) and `EastNorthUp` (scaled role
  permutation `S⁻¹` into the chart basis; meridian limit at the poles; `directions` in
  Earth-centred axes). Shared tangent-frame and figure checks factored out of
  `_chart_transitions.py`. `change_component_basis` assembles per-sample maps through the
  array's frame (`point_basis_map`), including chart transitions as `transform=` with today's
  direction rule, and maps east/north/up to its own Earth-centred chart by its directions so
  pole rows convert; failures name the sample. Component resampling evaluates field bases at
  `T(y)` and `y`, with an identity shortcut for one frame and one basis. Pole halos extend to
  components in constant Cartesian bases. `east_north_up` persists; user fields are refused.
- **Verification**: stock 2,494 passed plus 47 xfailed; patched 2,541; upstream 2,499 plus 42
  xfailed; ruff and mypy pass.
- **Files affected**: `src/xarrayrf/_frame_field.py`, `_chart_transitions.py`,
  `_tensor_components.py`, `_tensor_component_declaration.py`, `_component_resampling.py`,
  `_resample.py`, `_encoding.py`, `__init__.py`; `tests/test_frame_fields.py`,
  `tests/test_encoding.py`, `tests/test_component_resampling.py`; `docs/core_interface.md`,
  `CHANGELOG.md`, the value geometry design and stage 4 plan.
- **Reviewed by**: Codex (gpt-6-sol, high), plan and implementation.

## [2026-10-12] — Angular charts stage 4: a minimal CF reader

- **Problem**: A CF longitude/latitude dataset had no import path: users built the frame, its
  figure and the `Grid` by hand.
- **Resolution**: `xarrayrf.cf` (`from_dataset`, `frame_variable`) identifies one-dimensional
  longitude and latitude (dimension or auxiliary) by `standard_name` or `units`, reads the figure
  from a `latitude_longitude` grid mapping into the reserved definition keys (with
  `datum_name` and `ellipsoid_name` kept apart), keeps west/south labels through a negating
  transform, converts bounds to intervals with one derived offset (dropping and reporting
  off-centre bounds), mints an anonymous geodetic frame, and adopts `frame=` when given (required
  when no figure is declared). Refuses rotated poles, curvilinear coordinates, projected
  mappings, the extended grid-mapping syntax and a nonzero prime meridian.
- **Verification**: stock 2,466 passed plus 47 xfailed; ruff and mypy pass; import-boundary
  test extended to `cf`.
- **Files affected**: `src/xarrayrf/cf/__init__.py`, `tests/test_cf.py`,
  `tests/test_import_boundary.py`, `docs/core_interface.md`, `CHANGELOG.md`, design and plan
  notes.
- **Reviewed by**: Codex (gpt-6-sol, high), plan and implementation.

## [2026-10-12] — Angular charts stage 3: seam- and pole-aware resampling

- **Problem**: Resampling treated longitude/latitude labels literally: a target at 358.75° on a
  0…357.5° grid was fill, a regional 170…190 grid was not found from −175, and every target in
  the polar cap of a cell-centred grid (±89.5°) was fill.
- **Resolution**: `_periodic.py` decides, from metadata and labels, whether the longitude axis
  wraps (global, by declared intervals or a conservative seam-gap rule; closing duplicates
  recognized) or is regional (branch shift), and whether latitude crosses the poles (global
  longitude with antipodal pairs; odd or even mirror chosen from the labels; beyond ±90
  refused). Location is exact, ported from xarraybc's `periodic_bracket_array`; the pole
  crossing follows its `sphere` topology. `locator` applies it before any domain check;
  `plan` skips the lattice path for such sources and extends each block with halos (1 sample,
  24 for cubic). Declared components get no pole halo (`pole_halos=False`).
- **Verification**: stock 2,424 passed plus 47 xfailed; patched and upstream lanes pass; ruff
  and mypy pass. A 1440 × 720 global grid resamples in about 0.13 s, linear or cubic.
- **Files affected**: `src/xarrayrf/_periodic.py`, `_positions.py`, `_resampling.py`,
  `_resample.py`; `tests/test_seam_resampling.py`, `tests/test_resample.py`;
  `docs/core_interface.md`, `CHANGELOG.md`, the angular charts design and stage 3 plan.
- **Reviewed by**: Codex (gpt-6-sol, high), plan and implementation.

## [2026-10-12] — Independent QA of angular charts steps 1 and 2

- **Problem**: Cross-model QA (Codex gpt-6-astra, after gpt-6-sol's reviews) of the angular
  declarations and chart transitions (`58ddfac..0464ef5`).
- **Resolution**: Four findings, all confirmed by reproduction and fixed. (B1) Jacobians could
  return `inf` instead of raising (spherical inverse at `[1e-307, 0, 1]`); `jacobian` now raises
  a `ValueError` naming the first non-finite point, and `transform_point` reports overflow as
  an error rather than a warning. (B2) An accepted flattening near 1 divided by zero, then
  cancelled at the poles: `1 - e²` is carried as `(1 - f)²` and `1 - e² sin²φ` as
  `cos²φ + (1 - f)² sin²φ`, so no subtraction of e² remains. (N1) Tests added: real polar
  scalar and component resampling through `CartesianToGeodetic`, shapes, empty batches,
  overflow, mixed degree/radian charts, a chart missing its radial role, a malformed record.
  (N2) Docs overstated round trips (longitude returns modulo the period, and as
  `longitude_start` on the axis) and the PROJ comparison: PROJ's own geocentric inverse is off
  by up to 7.5 mm at 1000 km, while this inverse recovers PROJ's forward inputs to ~1e-9 m.
- **Verification**: stock 2,394 passed plus 47 xfailed; ruff and mypy pass. Two convergence
  passes; accepted.
- **Files affected**: `src/xarrayrf/_chart_transitions.py`, `tests/test_chart_transitions.py`,
  `docs/core_interface.md`, `CHANGELOG.md`.
- **Reviewed by**: Codex (gpt-6-astra, high); Claude Opus 5.5 (`claude-opus-5-5`) applying
  QAEngineer.

## [2026-10-12] — Angular charts stage 2: chart transitions

- **Problem**: A lon/lat/height chart and an Earth-centred Cartesian chart of one frame had no
  conversion: `coordinate_system_change` derives only permutations, and stage 1 made it refuse
  angular/Cartesian pairs.
- **Resolution**: `SphericalToCartesian`, `CartesianToSpherical`, `GeodeticToCartesian` and
  `CartesianToGeodetic` (`_chart_transitions.py`): equivalent-frame endpoints, explicit figure
  checked against reserved definition keys, Vermeille's closed-form geodetic inverse, forward
  domain equal to the inverse image (height above `-N(1 - e²)`, outside the evolute bound), exact
  poles on the axis, pointwise Jacobians with an analytic inverse whose longitude scale comes
  from the Cartesian distance to the axis. Four schema-1 record kinds.
  `coordinate_system_change` matches angular axes by role. Component resampling names the target
  sample when a Jacobian raises, not only when it is singular. Oracles: pyproj live (dev extra),
  Astropy as a generated fixture (`tools/build_astropy_spherical_fixture.py`).
- **Verification**: stock 2,384 passed plus 47 xfailed; patched and upstream lanes pass; ruff
  and mypy pass.
- **Files affected**: `src/xarrayrf/_chart_transitions.py`, `_encoding.py`, `_orientation.py`,
  `_component_resampling.py`, `__init__.py`; `tests/test_chart_transitions.py`,
  `test_encoding.py`, `test_orientation.py`, `test_component_resampling.py`;
  `tests/fixtures/astropy_spherical.json`; `tools/build_astropy_spherical_fixture.py`;
  `pyproject.toml` (mypy: astropy stubs); `CHANGELOG.md`; `docs/core_interface.md`,
  `docs/design.md`.
- **Reviewed by**: Codex (gpt-6-sol, high), plan and implementation.

## [2026-10-12] — Independent QA of angular charts stage 1

- **Problem**: Review the angular declarations, refusals and persistence (`515b9e7`, design
  `f48b888`) for root-cause soundness, regression risk, edge cases and coverage.
- **Resolution**: The declaration model, the `coordinate_system_change` guard and the optional
  `axis_roles` field hold; probes confirmed pickling, deep copy, native encode/decode of a framed
  lon/lat array, `isel`, and NIfTI export refusal. One gap fixed: NGFF export
  (`to_transform`, `to_multiscale_level`) wrote an angular frame as plain axes with degree
  units, reporting only the identity loss, so representation and roles vanished and our own
  reader would refuse the result; both now refuse angular frames. Also: the angular-unit error
  now names the axis and its role (it said "Cartesian axis" for a time or radius axis of an
  angular system); `check_units(reject_angular=)` lost its last caller and was removed; tests
  added for an angle on a radius axis and a Cartesian record carrying roles. Residual,
  accepted: other adapters build Cartesian systems only, so an angular frame never enters them.
- **Verification**: stock 2,333 passed plus 47 xfailed; patched 2,380; upstream 2,338 plus 42
  xfailed; ruff and mypy pass.
- **Files affected**: `src/xarrayrf/ngff/_export.py`, `src/xarrayrf/_coordinate_system.py`,
  `src/xarrayrf/_validation.py`, `tests/test_ngff_export.py`, `tests/test_coordinate_system.py`,
  `tests/test_encoding.py`, `docs/core_interface.md`.
- **Reviewed by**: Claude Opus 5.5 (`claude-opus-5-5`) applying QAEngineer.

## [2026-10-12] — Angular charts stage 1: declarations, refusals, persistence

- **Problem**: Longitude/latitude data could not be declared: `CoordinateSystem` admitted only
  Cartesian axes and refused angular units, so GCM grids and celestial coordinates had to
  disguise angles as unitless Cartesian axes.
- **Resolution**: `SPHERICAL` and `GEODETIC` representations with `axis_roles` (longitude,
  latitude, radius, height); degree, CF east/north and radian units on angular axes only;
  orientation refused on angular systems. Euclidean quantities refused on angular frames
  (`Lattice.spacing`/`direction`, `axis_codes`, box averaging along latitude);
  `coordinate_system_change` refuses differing angular declarations, which would otherwise
  have derived an identity from matching axis names and units. `axis_roles` persists as an
  optional schema-1 field written only for angular systems, so Cartesian records are
  unchanged. No wrapping at the longitude seam yet.
- **Verification**: stock 2,330 passed plus 47 xfailed; patched 2,377; upstream 2,335 plus 42
  xfailed; ruff and mypy pass.
- **Files affected**: `src/xarrayrf/_coordinate_system.py`, `_validation.py`, `_orientation.py`,
  `_lattice.py`, `_resampling.py`, `_encoding.py`, `__init__.py`; tests for each;
  `docs/core_interface.md`, `docs/design.md`.
- **Reviewed by**: Codex (gpt-6-sol, high), plan and implementation.

## [2026-10-12] — Independent QA of value geometry stages 1 to 3

- **Problem**: Review the value geometry work (tensor types and kernel, component declarations,
  symmetric packing, resampling of declared components) for root-cause soundness, regression
  risk, edge cases and coverage before it is considered done.
- **Resolution**: The transform law, per-slot composition and pushforward direction hold;
  probes confirmed equivalent-frame resampling without a transform (derived coordinate-system
  change), pickling of packed declarations and arrays, and float output for integer nearest
  sources. Two gaps fixed. (1) A singular or non-finite frame-map Jacobian (a folding
  registration) raised the block-relative stack index from the inverse, with "zero row"
  wording for NaN, so the reported location was wrong for any block after the first; it now
  names the target sample by dimension. (2) Box methods (`overlap_mean` with
  `return_coverage`) on declared components had no test; one now checks averaged values, the
  output declaration and an undeclared coverage array. Residual, accepted: one singular
  Jacobian anywhere in the target aborts the whole resampling (explicit failure rather than
  per-sample fill); stage 1 and 2 behaviour is unchanged.
- **Verification**: stock 2,291 passed plus 47 xfailed; patched 2,338 passed; upstream 409 on
  the tensor, packing and resampling files; ruff and mypy pass. Each stage was also reviewed by
  Codex (gpt-6-astra, high) at plan and implementation.
- **Files affected**: `src/xarrayrf/_component_resampling.py`,
  `tests/test_component_resampling.py`.
- **Reviewed by**: Claude Opus 5.5 (`claude-opus-5-5`) applying QAEngineer.

## [2026-10-07] — Independent QA of the core layering refactor

- **Problem**: Verify that moving resampling planning, frame adoption, `Grid.isel`, binding
  diagnostics and interval arithmetic out of the xarray modules preserves public behavior,
  performance and the binding lifecycle, and that the stated import boundary holds.
- **Resolution**: One defect fixed. The deferred-revalidation size view in `_resample` revalidated
  the source Geometry on every planner size read (13 coordinate checks per call versus 9 at the
  base), making small resamples about 18% slower. It now revalidates on the first read and
  reuses those sizes; small resamples are now at or below base time, large ones unchanged.
  Stale design text claiming Grid selection and BindingIndex policy were still pending was
  corrected; the layering is recorded in
  [core_layering_design.md](architecture/core_layering_design.md). The `Grid.isel` error-type
  change for invalid indexers is recorded in `CHANGELOG.md`.
- **Verification**: Full suites: stock 1,904 plus 47 xfailed; patched 1,951; minimum-core (no
  SciPy) 1,648 plus 88 skipped/47 xfailed; minimum-resample 1,904 plus 47 xfailed; current
  Python 3.13/xarray 2026.9.0 1,908 plus 43 xfailed. Ruff, mypy (stock, floor, current) pass.
  `tools/binding_operation_probe.py` output is byte-identical to `da5f627`. Anonymous-frame
  diagnostic messages match `da5f627` for RangeIndex-step and int64/float cases. Implementation
  by Codex (gpt-6-astra) in three rounds, each reviewed by Claude Opus 5.5.
- **Files affected**: `src/xarrayrf/_resample.py` (size caching); design notes and changelogs.
- **Reviewed by**: Claude Opus 5.5 (`claude-opus-5-5`) applying QAEngineer.

## [2026-10-07] — Independent QA before committing repository cleanup

- **Problem**: Verify that the dependency-floor corrections, publishing smoke and documentation
  consolidation preserve public behavior, optional-dependency coverage and unique audit evidence.
- **Resolution**: No functional correction was needed. Clarified the metadata comment about
  lazy SciPy loading and rewrapped the touched Grid rationale. Kept the separate curved-spacetime
  assessment and its links outside the commit scope.
- **Verification**: Fresh full suites pass: stock 1,796 plus 47 xfailed; patched 1,843;
  minimum-core 1,559 plus 86 skipped/47 xfailed; minimum-resample 1,796 plus 47 xfailed;
  current Python 3.13/xarray 2026.9.0 1,800 plus 43 xfailed. Original source reproduces all
  three selected regressions and eight NumPy-floor mypy errors. Nine independent missing-extra
  probes verify validation, exception causes and zero Dask pixel tasks; interpolation kernels
  are unchanged after removing typing-only casts/aliases. Stock/floor/current mypy, Ruff,
  dependency consistency, helper/quoted-path checks and scoped Markdown paths/anchors pass.
  Distribution build/payload, strict metadata, sdist wheel rebuild and exact clean-wheel
  publishing smoke pass without SciPy. The original publishing block fails on its removed
  constructor keyword. Hosted CI and upload are separate from this local QA.
- **Files affected**: Cleanup changes in core resampling, tests, maintainer commands, publishing
  workflow and documentation; QA wording in `pyproject.toml` and Grid rationale.
- **Reviewed by**: Codex GPT-6 applying QAEngineer (exact serving model ID not exposed).
  Implementation review attribution remains Claude Opus 5.5 (`claude-opus-5-5`), as recorded
  in the preceding entry; no additional external review was required for wording-only changes.

Maintainer QA and validation provenance. User-visible behavior belongs in `CHANGELOG.md`;
completed audit decisions and per-stage evidence are in [the audit record](codebase_audit_review.md).

## [2026-10-06] — Repository documentation, dependency floors and publishing smoke

- **Problem**: The publishing smoke used a removed affine constructor; minimum-core tests
  loaded an absent optional dependency before validation; NumPy-floor stubs exposed eight
  typing errors. Completed audit transcripts and status snapshots contradicted current code.
- **Resolution**: Use the named-basis smoke declaration, load SciPy after resampling validation,
  guard actual sampling/NetCDF tests, and retain core-only frame/adoption assertions. Narrow
  numeric annotations without changing interpolation; consolidate completed audit history,
  refresh source-of-truth links and current/deferred status, and make local source paths explicit.
  `xrpr` defaults to the repository environment with an `XARRAY_PYTHON` override.
- **Verification**: The missing-extra validation regression fails before the import-order fix
  and passes after it. Exact floor environments reproduce the original 60 test failures/eight
  mypy errors and now pass: core 1,559 passed/86 skipped/47 xfailed; resample 1,796 passed/47 xfailed.
  Stock 1,796 passed/47 xfailed; patched 1,843 passed. Stock and floor mypy pass 82 files; Ruff
  lint/format and whitespace pass. Current Python 3.13/xarray 2026.9.0 passes 1,800 tests with
  43 xfailed and current mypy passes 82 files. Distribution build/payload and the exact
  clean-wheel publishing smoke pass on Python 3.12 without SciPy. Local helper and Markdown
  link/anchor checks pass. Opus accepted correctness convergence and documentation, with useful
  test/wording follow-ups resolved. Hosted CI has not rerun these unpushed changes; TestPyPI
  was not dispatched. Supported-floor warning totals remain in the test logs; none failed checks.
- **Files affected**: `src/xarrayrf/_resample.py`; optional-dependency/README/grid tests;
  `.github/workflows/publish.yml`; Makefile, `tools/xrpr`, binding probe; contributor/user docs
  and the maintainer documentation tree.
- **Reviewed by**: Codex GPT-6 applying QAEngineer; Claude Opus 5.5 (`claude-opus-5-5`) through
  collaborative-refinement. Model selection is recorded by invocation; the CLI did not announce
  its serving model separately.

## [2026-10-05–06] — Completed five-stage audit

- **Problem**: Incorrect identity/support/cardinality, valid-input failures, export/interpolation
  semantics, ambiguous public declarations and partial DICOM output were confirmed through
  public interfaces.
- **Resolution**: Five independently reviewable stages, including fresh Unicode-destination QA,
  are summarized in [codebase_audit_review.md](codebase_audit_review.md). That record preserves
  numerical calibration, commit provenance, original-source failures, independent probes and
  the uncommitted Stage 5 correction evidence. Original committed transcripts remain in Git.
- **Files affected**: Core geometry/sampling/native code, adapters, DICOM tool, public regression
  tests and normative docs; see the audit stage/commit table for scope.
- **Reviewed by**: Codex GPT-6 applying QAEngineer; Claude Opus 5.5 (`claude-opus-5-5`).

## [2026-10-04] — QA of named-basis affine construction

- **Problem**: Matrix-only construction left source-column and target-component ordering
  implicit. The implementation review also caught two accidentally migrated SimpleITK
  constructors; the QA pass found one remaining old-signature example in the relativity notes.
- **Resolution**: The default constructor uses source-named `basis_vectors` and requires
  `target_axes` to match the target exactly. Each vector is validated before assembly; both
  construction paths share immutable coefficients and retain schema 1. Matrix callers use
  `from_matrix`. The SimpleITK helper is byte-for-byte unchanged from the baseline, and the
  stale documentation example is corrected. QA found no remaining functional defect.
- **Verification**: 150 focused tests pass; stock xarray has 1617 passing tests and 47 expected
  failures for existing hook gaps; patched xarray has 1664 passing tests. Lint, formatting,
  mypy and `git diff --check` pass. An AST audit confirms 35 migrated Python files changed
  only the constructor spelling. The implementation pass also executed all 10 relativity
  notebook cells and checked the tour's placement against an independent point calculation.
  The full network-backed tour and optional SimpleITK registration helper were not executed.
- **Files affected**: `src/xarrayrf/_affine.py`; matrix callers in the core, anatomy and
  DICOM, NIfTI, NGFF and GeoTIFF adapters; affine, encoding, README and integration tests;
  `README.md`, `CHANGELOG.md`, `docs/core_interface.md`, the core-model and relativity notes,
  binding-operation inventory; notebook examples, their builders, the binding probe and
  resampling benchmark.
- **Reviewed by**: Codex GPT-6 (exact serving model ID not exposed); implementation reviews
  by Claude Opus 5.5 (`claude-opus-5-5`).

## [2026-10-01] — QA fixes for anonymous frames (grid plan stage 5)

- **Problem**: A QA pass over stage 5 found that a framed array written to NetCDF and read back
  could no longer combine with its original on stock xarray: the reload reordered the bound
  coordinates, and stock xarray keys alignment by coordinate order, so two equal bindings
  conflicted (this affected complete frames as well and predated stage 5; patched xarray was
  unaffected). It also found that two identically framed anonymous arrays sliced to zero length
  were told to resample although `assume_frame` alone sufficed, and following that advice
  raised `IndexError` inside resampling.
- **Resolution**: Every framing door and `rf.decode` produce bound coordinates and indexes in one
  canonical order, keeping the adapter orders already pinned by tests and leaving other
  coordinates and metadata untouched. Equal adopted bindings recommend `assume_frame` alone
  regardless of the numerical point check. Resampling onto an empty target returns an empty
  framed result, and an empty source with a non-empty target raises a documented `ValueError`.
- **Files affected**: `src/xarrayrf/native.py`, `_frame_compatibility.py`, `_resample.py`;
  `tests/test_grid.py`, `tests/test_native.py`, `tests/test_native_encoding.py`;
  `docs/core_interface.md`, `CHANGELOG.md`.
- **Reviewed by**: Codex gpt-6-astra (reasoning effort high), with fixes implemented by Codex
  gpt-6.1-sol and verified by Claude Opus 5.5 (claude-opus-5-5)

## [2026-10-01] — QA fixes for anatomy on grids (grid plan stage 4)

- **Problem**: A QA pass over `xarrayrf.anatomy` found that `cardinal_grid` always assigned
  source dimensions to frame axes, so a source whose assignment is ambiguous (a 45° volume)
  was refused even with explicit `spacing` and `dims`, although the ambiguity error tells
  callers to resample to a cardinal grid; and that the roundoff bound on non-anatomical frame
  axes multiplied the largest coefficient by the largest source span independently, so
  rescaling one source coordinate loosened it enough to accept a real one-second time
  variation. Committed tests also did not resample onto a cardinal grid, and the reflection
  offset bound was worded inconsistently.
- **Resolution**: Output directions come from the requested orientation and the frame;
  the source assignment is used only for default spacing and dimension names, which refuse an
  ambiguous source with advice to pass them explicitly. The roundoff bound uses each source
  column's own physical scale (coefficient times span), so equivalent representations decide
  alike. Integration tests resample an oblique DICOM-like volume and a thick single slice onto
  cardinal grids against a linear oracle. The offset bound is stated as `np.spacing(1.0)`.
- **Files affected**: `src/xarrayrf/anatomy.py`; `tests/test_anatomy.py`,
  `tests/test_dicom_import.py`; `docs/core_interface.md`, `docs/dev/architecture/grid_plan.md`,
  `CHANGELOG.md`.
- **Reviewed by**: Codex gpt-6-astra (reasoning effort high), with fixes implemented by Codex
  gpt-6.1-sol and verified by Claude Opus 5.5 (claude-opus-5-5)

## [2026-09-30] — QA fixes for declared intervals (grid plan stage 3)

- **Problem**: A QA pass over declared intervals found that an empty varying axis with declared
  intervals could not round-trip through either persistence API (its `(0, 2)` rows encoded as
  `[]` and decoded with the wrong shape); that the magnitude-aware offset/interval agreement
  tolerance could exceed a narrow interval's width, admitting a sample outside its own cell, which
  the cells domain then refused to locate; and that binding-index equality raised pandas'
  `InvalidIndexError` in one operand order when labels repeated, while the reverse returned
  `False`.
- **Resolution**: Empty interval lists decode as `(0, 2)` rows for empty varying axes, with strict
  rejection kept elsewhere. Construction checks exact containment `lo <= sample <= hi` in addition
  to the approximate offset agreement. `BindingIndex.equals` compares coordinate indexes before
  any label lookup, so it returns a bool in both orders and merges raise xarray's `MergeError`.
- **Files affected**: `src/xarrayrf/_encoding.py`, `_sampling.py`, `_binding.py`;
  `tests/test_intervals.py`; `docs/core_interface.md`,
  `docs/dev/architecture/geometry_and_resampling_design.md`, `CHANGELOG.md`.
- **Reviewed by**: Codex gpt-6-astra (reasoning effort high), with fixes implemented by Codex
  gpt-6.1-sol and verified by Claude Opus 5.5 (claude-opus-5-5)

## [2026-09-30] — QA fixes for freestanding grids (grid plan stages 1 and 2)

- **Problem**: A QA pass over the `Grid` value and its doors found:
  - copied and unpickled grids exposed writable coordinates, and returned arrays let callers
    re-type the stored bytes;
  - `Grid` query paths accepted malformed custom-transform output that `Geometry.points`
    already refused;
  - fractional interpolation lost stored endpoints to cancellation across large magnitudes
    (the first fix then overflowed when extrapolating near the float limit);
  - integer-only input was promoted to float64 and rounded before validation;
  - a `Geometry` resample target overwrote the source's non-geometry scalar coordinates,
    unlike a `Grid` target;
  - `Grid.isel` duplicated xarray's indexing and the binding's slicing and lacked list,
    array and mask indexers; zero-row queries on empty grids raised.
- **Resolution**: Copies and pickles rebuild through the constructor and coordinates are
  returned as read-only views. Transform-result validation is shared by every `Grid` and
  `Geometry` query. Interpolation uses a weighted form inside a cell (exact at samples) and
  extends from the nearer endpoint outside it. Integer input is detected and range-checked
  before promotion. Resampling takes only geometry coordinates from the target and refuses
  name collisions with source context. `Grid.isel` and the new `Grid.sel` run xarray's
  selection on a coordinate-only Dataset carrying the binding index. Zero-row queries return
  empty results. The lost `RangeIndex` step on snapshots is documented as a parity difference.
- **Files affected**: `src/xarrayrf/_grid.py`, `_positions.py`, `_sampling.py`, `_validation.py`,
  `_resample.py`, `_binding.py`, `_geometry.py`, `native.py`; `tests/test_grid.py`,
  `tests/test_resample.py`, `tests/test_import_boundary.py`; `docs/core_interface.md`,
  `docs/dev/architecture/geometry_and_resampling_design.md`, `CHANGELOG.md`.
- **Reviewed by**: Codex gpt-6-astra (reasoning effort high), with fixes implemented by Codex
  gpt-6.1-sol and verified by Claude Opus 5.5 (claude-opus-5-5)
