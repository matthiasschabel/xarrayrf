# Contributing

## Principles

- **Reference frames, coordinate mappings and sampling are separate concepts.** A frame names a
  space, a transform maps an array's coordinates into it, and xarray's own coordinates describe
  the current samples. Array size is not geometry.
- **Native xarray objects.** Framed arrays are ordinary `DataArray`s and `Dataset`s. Do not add
  an array wrapper, a provider registry or a parallel mutable copy of xarray coordinates to work
  around an operation-lifecycle problem.
- **Producers attach geometry once.** Generic operations never call back into the code that
  created an array.
- **Keep the core small.** The core imports NumPy only (xarray for the `.rf` integration).
  Format adapters (`nifti`, `dicom`, `ngff`, `geotiff`) are optional extras that import only the
  core, `xarrayrf.anatomy` and their own format library; the import-boundary test enforces this.
  Extend the core only for a general capability that more than one adapter or use case needs.
- **Fix xarray where the problem belongs to xarray.** Prefer a small, separately reviewable
  xarray change over a downstream workaround, monkey patch or weaker guarantee (see
  [xarray patches](#xarray-patches)).
- **No wrong answers.** An operation that would return a valid-looking but incorrect binding
  blocks a release, even if documented as unsupported. The minimum set of native operations that
  must work is defined in [docs/design.md](docs/design.md) and is not narrowed to make a test
  pass.

## Code conventions

- Public names and terms follow [docs/core_interface.md](docs/core_interface.md), which is
  normative. Architecture notes explain the decisions; [the roadmap](docs/dev/roadmap.md)
  owns the current work queue.
- Python 3.12+, src layout, NumPy typing, immutable value objects, composition over inheritance,
  standard-library exceptions, explicit validation at public boundaries and Google-style
  docstrings. Line length is 100.
- No compatibility aliases for APIs that have not been released.
- Tests exercise public behavior, with explicit tolerances for floating-point comparisons.
  Expected failures document known gaps; they are not supported behavior.

## Repository layout

| Path | Contents |
|---|---|
| `src/xarrayrf/` | The shipped package |
| `tests/` | Tests of the public API |
| `benchmarks/` | Development-only performance measurements |
| `examples/` | Example notebooks and their data helpers |
| `tools/` | Notebook builders, the native-operation probe, upstream reproducers and other maintainer scripts |
| `docs/` | Design, interface specification and development notes |

## Development setup

```sh
uv sync --locked --extra dev   # released xarray from PyPI
make check                     # ruff and mypy
make test                      # tests against released xarray
make test-pinned               # tests against the pinned xarray patch series, in .venv-patched
make build                     # build and check the distribution payload
```

`make test-pinned` installs the patched xarray into a separate environment through the `patched`
dependency group, so the main environment is never changed. With pip instead of uv:

```sh
pip install "xarray @ git+https://github.com/matthiasschabel/xarray@xarrayrf-patches-4"
pip install -e ".[dev]"
```

`make tour` and `make relativity` rebuild the example notebooks; `make probe` prints the
native-operation measurements recorded in
[the operation inventory](docs/dev/binding/binding_operation_inventory.md).

## xarray patches

When a correct solution needs a change in xarray, the change is made in xarray rather than worked
around here. Each fix is kept as a separately reviewable commit on the
[xarray fork](https://github.com/matthiasschabel/xarray), with an xarray-only regression test,
and proposed upstream. The current series, its base commit and the state of each upstream pull
request are recorded in [the patch manifest](docs/dev/xarray-upstream/xarray_patches.md); the
workflow is described in [docs/upstream.md](docs/upstream.md).

Maintainers developing patches can select their worktrees with
`make test-upstream XARRAY_UPSTREAM=/path/to/xarray` and
`make test-patched XARRAY_PATCHED=/path/to/patched-xarray`. The shared reproducible command
is `make test-pinned`; local worktree targets use the existing stock test environment.

## Development notes

Maintainer notes live under [docs/dev/](docs/dev/README.md). Each opens with a Status, Last
updated and Scope header, followed by Context, Current Decision, Alternatives Considered,
Deferred Work and Next Steps. Stable decisions move into [docs/design.md](docs/design.md) or
[docs/core_interface.md](docs/core_interface.md). README claims must match implemented behavior.
Shared documentation and its linked files belong in the same change; a file present only in
a maintainer's working tree must not be required by committed navigation. Local session state
stays outside the shared documentation index. Completed reviews retain decisions and evidence,
with remaining work moved to the roadmap rather than duplicated in execution transcripts.

## Further reading

- [Design](docs/design.md) and [core interface](docs/core_interface.md)
- [Development notes](docs/dev/README.md)
- [Release rehearsal](docs/releasing.md)
