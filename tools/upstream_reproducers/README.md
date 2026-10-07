# Upstream PR reproducers

One script per pydata/xarray bug-fix PR, plus the issue #11607 operand-conflict measurement.
[Upstream PR notes](../../docs/dev/xarray-upstream/upstream_prs.md) own current merge/review
status; the [patch manifest](../../docs/dev/xarray-upstream/xarray_patches.md) owns the pinned
series. Every script prints the imported xarray path before its observed behavior.

Run against any selected xarray source checkout:

```sh
tools/xrpr /path/to/xarray tools/upstream_reproducers/pr_11621.py
XARRAY_PYTHON=/path/to/environment/bin/python tools/xrpr /path/to/xarray -c 'import xarray; print(xarray.__file__)'
```

Without a command, `xrpr` opens a console with NumPy and xarray preloaded; a script is run with
`python -i`. The default interpreter is this repository's `.venv/bin/python`. Keep source
revision and interpreter selection explicit; a local checkout is not necessarily current
upstream or the posted PR head.

`_raster_index.py` reproduces the public custom-index guide's RasterIndex and the scalar-keeping
variant needed by the scalar-drop repro. `issue_11607_operand_check.py` prints JSON for indexed,
plain-coordinate and missing-coordinate operands. Merged regression reproducers remain useful
for verifying a series base or a released build.
