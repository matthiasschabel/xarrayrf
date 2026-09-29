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

Each candidate upstream PR lives in its own worktree of `~/GitHub/xarray-upstream`, on a
branch `pr/<name>` cut from upstream `main`. A detached `~/GitHub/xarray-base` worktree at
`dfd25c7252e71a47e789aa24b8a06dd911a56461` gives the "before" tree for the three open
PRs. A reproducer can run on both sides without touching the `main` checkout that
`make test-upstream` uses.

`tools/xrpr` runs the upstream environment's Python with a chosen tree first on `PYTHONPATH`, so
that tree's source shadows the installed xarray:

```sh
tools/xrpr base explorations/upstream_reproducers/pr_11621.py    # before: run, then stay in the console
tools/xrpr 11621 explorations/upstream_reproducers/pr_11621.py   # after: same script on the PR branch
tools/xrpr 11616                                                 # bare console, np and xr preloaded
tools/xrpr main explorations/upstream_reproducers/pr_11615.py   # merged fix; no PR worktree
tools/xrpr base -c "import xarray; print(xarray.__file__)"       # confirm which tree answers
```

Trees are named by PR number (the mapping is in the script header), plus `base`, `main` (the
unmodified comparison checkout) and `patched` (the local series 4 tree, `index-hooks-4`), or
an absolute path to any xarray checkout. Every reproducer prints `xarray.__file__` first.

`explorations/upstream_reproducers/` holds one self-contained script per PR, including merged
#11613 and #11615, written around a public example (the `RasterIndex` from xarray's
custom-index guide) so a maintainer can paste it. Add one when preparing a PR and record its
before/after output in the PR's section of `docs/dev/xarray-upstream/upstream_prs.md`.

To run xarray's own tests against a tree, use the same interpreter from inside that worktree:

```sh
cd ~/GitHub/xarray-pr-<name>
PYTHONPATH=$PWD ~/GitHub/xarray-upstream/.venv/bin/pytest xarray/tests/test_dataset.py -q -p no:cacheprovider
PYTHONPATH=$PWD ~/GitHub/xarray-upstream/.venv/bin/pytest xarray/tests -n 6 -q -p no:cacheprovider   # full suite
```

`PYTHONPATH=$PWD` is required; without it the editable install in the environment answers, not the
worktree. Do not create a `.venv` inside a PR worktree (`uv run` will, if asked); it is gitignored but
it shadows nothing and wastes space. Format with the ruff version pinned in the upstream
`.pre-commit-config.yaml` (`uvx ruff@<version>`), and compare any mypy run against the baseline of an
untouched tree, since the environment lacks some stubs.

## Ownership

The extension owns reference-frame semantics and acceptance tests. The separate xarray checkout
owns xarray fixes and their regression tests. The maintainer manifest links those two histories;
there is no vendored xarray tree, import-time monkey patch or second copy of its implementation.

The project does not require waiting for upstream acceptance to develop against a local fix.
A clean upstream-layer solution is preferred even when maintaining that fix has a short-term
cost. If no viable mechanism meets the native-operation contract, report the concrete conflict
rather than quietly replacing native arithmetic with accessor-only calls.
