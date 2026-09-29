# Upstream PR reproducers

One script per pydata/xarray PR (#11613, #11615, #11616, #11617, #11621), each printing which
xarray tree it imported and the behaviour the PR changes. #11613 and #11615 are merged;
run them against `main`. `_raster_index.py` is the `RasterIndex` from
xarray's custom-index guide (Meta-indexes), plus a scalar-keeping variant for #11617.

Run them against a tree with `tools/xrpr` (see its header): `base` is the PR base commit
`dfd25c7252e71a47e789aa24b8a06dd911a56461` (worktree `~/GitHub/xarray-base`);
the open PR numbers #11616, #11617 and #11621 map to their worktrees.

`issue_11607_operand_check.py` is the measurement behind issue #11607: an xarray-only matrix of
scalar custom-index, plain-coordinate and missing-coordinate operands. It prints JSON describing
each operation's outcome on whichever xarray it imports.
