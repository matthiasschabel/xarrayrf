# Developing and maintaining xarray fixes

Prefer the layer that owns the problem. A missing xarray lifecycle hook or incorrect index
behavior should be addressed in xarray when that is cleaner than a downstream workaround.
Frame identity, mapping representations and DICOM policy remain outside xarray core.

## Local patch workflow

1. Reproduce the defect on an unmodified, identified xarray revision. Search existing issues
   and PRs, including closed reports, by symptom as well as proposed mechanism; record findings.
2. Work in a separate xarray checkout on a dedicated branch. One conceptual fix per commit;
   keep an xarray-only regression test with the fix. Do not alter installed site-packages.
3. Record the upstream URL, full base commit, ordered patch commit IDs, owning local checkout,
   rationale, dependencies between patches and validation commands in a patch manifest under
   docs/dev/. Empty manifests must not claim any patch exists or is required.
4. Use a dedicated environment to install the exact patched source, then run xarray's focused
   regression tests and xarrayrf's operation contract tests. Preserve the normal uv.lock as the
   unmodified dependency baseline. Record actual import path and revision in test results.
5. Keep a stock dependency CI lane. When a patch becomes necessary, add a separate patched lane
   sourcing immutable, CI-accessible commits. A local path override is a developer convenience,
   not a reproducible CI dependency. Do not publish private commits to make CI work without
   authorization. Mark stock expected failures narrowly; patched behavior must pass normally.
6. Rebase deliberately onto a recorded new base, retain reviewable commit boundaries, and run
   the focused regressions before broad compatibility tests. Do not combine unrelated fixes
   merely because the reference-frame implementation needs them together.
7. Once reference-frame behavior stabilizes, prepare small upstream issues/PRs following the
   current xarray contribution and AI policies. New APIs require maintainer discussion before
   a substantial PR. Obtain explicit user authorization before any external submission.
8. Once a development series rebases onto an upstream commit containing a merged fix, retire
   the duplicate local patch and update the manifest. After that fix is released and verified
   on stock xarray, raise the supported floor if appropriate. Never silently retain an obsolete
   patch alongside the upstream implementation.

## Testing an upstream PR branch locally

Use a separate checkout/worktree for each candidate change. Choose the interpreter and source
checkout explicitly; local worktrees may lag posted PR heads. Current remote status belongs in
[upstream PR notes](dev/xarray-upstream/upstream_prs.md), while immutable series membership
belongs in [the patch manifest](dev/xarray-upstream/xarray_patches.md).

```sh
tools/xrpr /path/to/baseline-xarray tools/upstream_reproducers/pr_11621.py
tools/xrpr /path/to/pr-xarray tools/upstream_reproducers/pr_11621.py
XARRAY_PYTHON=/path/to/xarray-environment/bin/python tools/xrpr /path/to/xarray -c 'import xarray; print(xarray.__file__)'
make test-upstream XARRAY_UPSTREAM=/path/to/upstream-xarray
make test-patched XARRAY_PATCHED=/path/to/patched-xarray
```

`tools/xrpr` defaults to this repository's `.venv/bin/python`. It puts the selected source first
on `PYTHONPATH`, opens a NumPy/xarray console without a command, and runs scripts with `python -i`.
Every [reproducer](../tools/upstream_reproducers/README.md) prints its imported xarray path.
Keep merged reproducers when they remain useful for verifying baseline/released behavior.
`make test-pinned` remains the portable shared command for the immutable published series.

To run xarray's tests, work inside the selected tree and point at its source:

```sh
cd /path/to/xarray
PYTHONPATH="$PWD" /path/to/xarray-environment/bin/python -m pytest xarray/tests/test_dataset.py -q -p no:cacheprovider
```

Without the source path, an editable install elsewhere may answer. Reuse a chosen environment;
do not let `uv run` create an unnoticed environment in every PR worktree. Format using the ruff
version pinned in upstream's `.pre-commit-config.yaml`, and compare typing checks with an
untouched baseline when dependency stubs are incomplete.

## Ownership

The extension owns reference-frame semantics and acceptance tests. The separate xarray checkout
owns xarray fixes and their regression tests. The maintainer manifest links those two histories;
there is no vendored xarray tree, import-time monkey patch or second copy of its implementation.

The project does not require waiting for upstream acceptance to develop against a local fix.
A clean upstream-layer solution is preferred even when maintaining that fix has a short-term
cost. If no viable mechanism meets the native-operation contract, report the concrete conflict
rather than quietly replacing native arithmetic with accessor-only calls.
