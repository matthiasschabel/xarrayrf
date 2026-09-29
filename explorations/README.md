# Feasibility experiments

These files are development artifacts and are excluded from distributions. The scalar guard
is a known-insufficient candidate, retained to reproduce xarray lifecycle gaps. It does not implement xarrayrf's public API.

Run `make probe` for JSON observations and `make test` for conformance experiments. Strict
xfails name selected known contract violations; changes to those tested cases force review rather than silently
turning an old failure expectation into proof of production readiness. Dask tests check that
scalar arithmetic, two-operand alignment and selection build lazy results without executing
pixel tasks through the local scheduler. Other operations and distributed schedulers are not covered.

`index_hook_probe.py` adds hook-level recording on top of the same rejected guard. It answers
which public `Index` hooks stock xarray calls for coordinate and Dataset variable extraction,
binary ops, NumPy ufuncs, `apply_ufunc`, `where`, selection and rename, and whether
`should_add_coord_to_array` receives anything that identifies the owning data variable. It
asserts nothing and grants nothing; see [the core model design](../docs/dev/architecture/core_model_design.md).

`joint_coordinate_index.py` holds a second rejected-until-measured candidate: an index that
owns `y`, `x` and the marker jointly, delegating every one-dimensional label operation to a
public `PandasIndex` per axis. `ownership_comparison_probe.py` runs one operation matrix
against the bare, scalar-guard and joint carriers; `operation_trace_probe.py` records the
scalar baseline's real call chains with a temporary `sys.setprofile` hook that observes
metadata only and is always restored. Predictions and the empty results table are in
[the ownership note](../docs/dev/binding/binding_design.md); neither candidate is adopted.

The production value objects in `src/xarrayrf` are tested under `tests/`, not here. These
experiments never import them.

`operand_check_probe.py` is an xarray-only matrix of scalar custom-index, plain-coordinate and
missing-coordinate operands. It reports each operation's outcome on a chosen xarray source; its
test pins observed classes on the locked dependency. See
[the operand measurement](../docs/dev/binding/binding_design.md) for both-lane results.

Choose the eventual mechanism from these results and the reuse options in docs/design.md.
When a fix belongs in xarray, follow docs/upstream.md instead of adding a workaround here.
