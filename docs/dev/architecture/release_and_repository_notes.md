# Release, packaging and repository notes

**Status:** Active
**Last updated:** 2026-10-24
**Scope:** Package metadata, CI lanes, dependency floors, the manual TestPyPI workflow and the
hosted steps still outstanding. The maintainer procedure is [releasing.md](../../releasing.md).

## Context

xarrayrf is a standalone repository (`matthiasschabel/xarrayrf`, public, MIT). Runtime
dependencies are NumPy and xarray; domain producers such as DICOM applications are downstream
clients or optional adapter extras. Version `0.0.1.dev0` is on TestPyPI from the
2026-10-24 rehearsal (below); nothing has been published to PyPI. Native-operation gaps on stock
xarray remain release blockers ([the architecture](../../design.md) §5), so a TestPyPI upload is a
packaging rehearsal, not a release. Clean fixes belong in xarray itself where appropriate,
maintained locally as small upstream-sized changes ([upstream workflow](../../upstream.md)).

## Current Decision

### Packaging

- PEP 621 metadata with Hatchling, an explicit MIT license file, repository and issue URLs, and
  the `Development Status :: 2 - Pre-Alpha` classifier, matching the README. An Alpha classifier
  would claim more than the native-operation gate supports.
- The sdist includes only `src/xarrayrf`, `README.md`, `CHANGELOG.md`, `LICENSE` and
  `pyproject.toml` (root-anchored patterns, so nested development READMEs cannot leak).
  Development notes ship in neither artifact.
- `tools/check_distribution.py` requires `xarrayrf/__init__.py` and `py.typed` in both artifacts;
  a negative control confirmed that omitting package code fails the check.
- `requires-python >= 3.12`; core `numpy >= 1.26`, `xarray >= 2026.7.0`. The `resample` and `dev`
  extras require `scipy >= 1.18`, which requires NumPy 2.

### Why SciPy 1.18

SciPy 1.16.3 and 1.17.1 mishandle the boundary of the nearest-mode spline prefilter:
`spline_filter([10., 20., 30.], order=3, mode="nearest")` then
`map_coordinates(..., [[0., 2.]], order=3, mode="nearest", prefilter=False)` returns about
9.99705 for the first value instead of 10. This broke cubic cells-domain edge values by about
1.8e-4 against a 1e-10 test tolerance. The fix is scipy/scipy#24615, in SciPy 1.18.0. The floor
was raised rather than loosening tolerances.

### CI lanes (`.github/workflows/ci.yml`)

| Job | Python | Dependencies | Runs |
|---|---|---|---|
| locked | 3.12 | `uv sync --locked --extra dev` | `make check test build` |
| minimum-resample | 3.12 | NumPy 2.0.0, SciPy 1.18.0, xarray 2026.7.0 | `make check test build` |
| current | 3.13 | latest resolution of `.[dev]` | `make check test build` |
| minimum-core | 3.12 | NumPy 1.26.0, xarray 2026.7.0, adapter extras, pytest, Dask; SciPy absent | `pytest tests` |
| patched | 3.12 | `uv sync --locked --extra dev --group patched` in `.venv-patched` | Full `pytest` suite, binding probe JSON gate |

Non-locked lanes resolve the full editable extra in a clean environment (downgrading only NumPy
had left SciPy and Zarr versions requiring NumPy 2) and every lane runs `uv pip check`.
minimum-core asserts SciPy is absent. Actual sampling and SciPy-NetCDF tests skip there;
frame/adoption refusals validate before loading SciPy and remain covered by core-only tests.
Local worktree targets have overrideable source paths; `make test-pinned` uses an isolated
pinned environment. `make probe-pinned` syncs that environment from the lock and runs the
binding probe plus `tools/check_binding_probe.py`. The new `patched` job uses the same
environment layout and requires a nonempty case list, every classification `ok`, and the
resolved `xarray_module` path inside that environment. It logs the imported module path,
version and installed distribution's `direct_url.json` git commit ID: this records the revision
actually installed, independently of the configured pretend version or a movable tag.

**Hosted validation rechecked 2026-10-24:** both floor lanes had been failing since mid-October
while the locked lane stayed green. minimum-resample's mypy run rejected annotations that only
NumPy 2.0's stubs treat as errors, and minimum-core ran 135 field-transform and resampling tests
without SciPy. After `548f700`, [run 40113369020](https://github.com/matthiasschabel/xarrayrf/actions/runs/40113369020) passes all four jobs again. Only the locked lane runs
locally by default, so check new numeric annotations and SciPy-dependent tests against both
floors before relying on a green local run.

**Hosted validation checked 2026-10-07:**
[run 37668735524](https://github.com/matthiasschabel/xarrayrf/actions/runs/37668735524) on
`da5f627d7e822de647273b14afb42d659c97d3bb` passes all four jobs: locked, minimum-core,
minimum-resample and current. The three `check` jobs pass dependency consistency, lint,
formatting, mypy, tests and distribution build/payload checks; minimum-core passes dependency
consistency, the absent-SciPy assertion and tests. The preceding cleanup revision `43eed8c`
also passes [hosted checks](https://github.com/matthiasschabel/xarrayrf/actions/runs/37668711828).

Historical [run 37538303719](https://github.com/matthiasschabel/xarrayrf/actions/runs/37538303719)
on `a0b7e6f` failed minimum-core (60 missing-SciPy test failures) and minimum-resample (eight
NumPy-floor typing errors). These failures are resolved by the cleanup commits. Fresh local
QA on 2026-10-07 passed core 1,559/86 skipped/47 xfailed, resample 1,796/47 xfailed, stock
1,796/47 xfailed, patched 1,843, and current Python 3.13/xarray 2026.9.0 1,800/43 xfailed;
details belong in the [maintainer changelog](../changelog.md).

The patched-pin job is configured but still awaits hosted evidence and a bad-pin negative
control after an approved push; R3 remains open. Stock expected failures still represent native
operation gaps; green dependency-floor CI does not certify the release invariant.

### Publishing workflow (`.github/workflows/publish.yml`)

Manual dispatch, restricted to `main`. The build job runs `make check test build`, Twine
metadata validation (Twine at a fixed version, not a dev dependency), a wheel rebuilt from the
sdist and a clean wheel-install smoke test. Only the publish job has `id-token: write`; it uploads
the build artifact through OIDC Trusted Publishing in the `testpypi` environment. Trusted
Publishing avoids storing an API token. The manual installation check in releasing.md takes
core dependencies from PyPI and the exact xarrayrf version from TestPyPI with `--no-deps`, then
runs `pip check`, so a mixed index cannot select the wrong source.

The wheel smoke uses the current named-basis affine constructor; its older `matrix=` call
failed after that constructor was replaced. Local wheel validation is recorded in the
[maintainer changelog](../changelog.md). Procedures remain in releasing.md.

Hosted setup rechecked 2026-10-07: the `testpypi` environment exists with a selected-branch
policy allowing only `main`; the GitHub API lists no publishing workflow runs. PyPI and TestPyPI
project JSON endpoints both return 404 for `xarrayrf`. A pending publisher does not establish a
published project or reserve the name.

Pending publishers rechecked 2026-10-24 by the maintainer: `xarrayrf` is pending on both TestPyPI
and PyPI. The TestPyPI publisher's fields are confirmed (project `xarrayrf`, owner
`matthiasschabel`, repository `xarrayrf`, workflow `publish.yml`, environment `testpypi`). The
PyPI publisher is unused until a production job exists; that job must match its workflow and
environment fields.

TestPyPI rehearsal completed 2026-10-24 on `3bc80f2`: local checks in a fresh clone (2994 passed,
47 stock-lane xfails; build, payload check and `twine check --strict` passed), then [run 40111808490](https://github.com/matthiasschabel/xarrayrf/actions/runs/40111808490)
built and uploaded through OIDC, creating the TestPyPI project. Uploaded files:
`xarrayrf-0.0.1.dev0-py3-none-any.whl` (sha256 `3ea29138039cdfa8…`) and
`xarrayrf-0.0.1.dev0.tar.gz` (`36df9e58479f76b9…`). A fresh Python 3.12 environment outside the
checkout installed NumPy and xarray 2026.9.0 from PyPI and xarrayrf from TestPyPI with
`--no-deps`; `pip check` and the workflow smoke test passed. The metadata shows MIT,
`>=3.12`, the Pre-Alpha classifier and public repository/issue links; the sdist contains LICENSE,
README and CHANGELOG and no `docs/dev`. The run warned that `actions/checkout@v4`,
`upload-artifact@v4` and `setup-uv@v6` target the deprecated Node.js 20 runtime. Both workflows
now use Node.js 24 releases: `checkout@v7`, `upload-artifact@v7`, `download-artifact@v8` and
`setup-uv@v10.3.2`. setup-uv no longer publishes floating major tags, so it is pinned exactly.
download-artifact v8 fails on a digest mismatch instead of warning.

## Alternatives Considered

- Keeping development inside a downstream application: couples its dependencies and history to
  one consumer.
- Combining with a boundary-topology package: confuses reference geometry with topology.
- An API token secret instead of Trusted Publishing: needless credential handling.
- A production publishing job now: deferred until the native-operation release gate is met.
- Loosening cubic tolerances for older SciPy: would hide a real edge-value error.

## Deferred Work

- A later version number and a production publishing policy, after the release gate.
- Keep both minimum lanes current as dependency floors move.

## Next Steps

The hosted patched-pin job is R3 of the [release plan](../release_plan.md), which holds its
acceptance checks and approvals; the TestPyPI rehearsal (R6) is complete.
