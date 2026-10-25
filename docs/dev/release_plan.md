# Release plan

**Status:** Active
**Last updated:** 2026-10-24
**Scope:** The remaining release gates in the [roadmap](roadmap.md#release-criteria), their
upstream xarray dependencies and the packaging rehearsal. Packaging, CI lanes and the publishing
workflow themselves are in [release and repository notes](architecture/release_and_repository_notes.md);
the procedure is [releasing.md](../releasing.md).

## Context

### What "done" means

[Design §5](../design.md): required native operations preserve correct geometry, incompatible
operands refuse, and binding loss happens only at explicitly permitted exits. A valid-looking
wrong binding blocks release even for an operation described as unsupported. Green tests with
stock-lane strict xfails are not release approval. Viewer, nonlinear geometry and convenience
APIs are separate workstreams, not release gates.

### Status on 2026-10-24

R1 (audit), R3 (patched-pin CI) and R6 (TestPyPI rehearsal) are complete. R2's series 5 is
published and pinned; its extension evidence and drafts remain open, as do R5 and R7. R4 is
dropped for the pre-alpha release. Criterion 4
(dependency-floor CI) is met. Upstream: #11613, #11615 and #11616 are merged (`b335cea`,
`dfd25c7`, `23c9dd1`); #11613 and #11615 are in v2026.09.0, #11616 is on xarray `main` only.
The audit stages 1–5 are complete ([audit record](codebase_audit_review.md)).

### Relation to the nonlinear geometry work

The release gates are independent of field transforms stages 1b and 1c and of the twist halo,
with one exception:

- **R1 may add field-transform cases.** A binding can hold a field transform, so the audit
  matrix should include at least one field-backed binding where the operation paths differ
  from the affine case.

## Current Decision

Labels R1–R7 are local to this note; the roadmap criterion each one serves is in brackets.

### Ready to start

Before the first plan review, confirm the review workflow with the maintainer
(collaborative refinement; which reviewer model).

**R1: extend the native-operation audit** [criterion 2]. Re-run the
[93-case probe](binding/binding_operation_inventory.md) on stock, current upstream and
series 4 first; the September totals are historical. Propose a small, non-Cartesian matrix
covering mixed framed/plain operand order, partial-coordinate alignment and merge, Dataset
update/extraction, composed selection and reduction, coordinate-ownership changes,
scalar/empty/singleton geometry, declaration order differing from pixel order,
complete/anonymous identity and eager/Dask where the paths differ. Write the expected
preserve/refuse/unframe outcome for each case before running it. Check world points
independently of the result's own transform. Replace the probe's broad "may refuse" exception
acceptance with assertions on the actual error. Assess the three unfiled bug candidates in
[upstream PR notes, Deferred Work](xarray-upstream/upstream_prs.md) where the matrix reaches
their paths.
*Done when* the reviewed matrix runs, every failure has an owner and release impact in the
inventory, and confirmed blockers are in the roadmap. Do not loosen strict xfails to get there.
*Done 2026-10-24:* the [matrix](binding/native_operation_audit_matrix.md) (35 cases, N09
corrected in review) and exact expectations for the 93 legacy cases run in the probe and
`tests/test_native.py`. The run found one series-4 blocker, N33 (broadcast with an excluded
geometry dimension), fixed in series 5, and led to the maintainer decision that `interp`
refuses at the operation. Series 5 is 128/128; every stock and upstream hole has an owner in
the [inventory](binding/binding_operation_inventory.md).

**R2: next xarray series and consolidated hooks** [criterion 1]. Record a new upstream base in
a fresh worktree. Drop patch 6 (the base will include #11616). Re-verify that patches 2 and 7
match the posted #11617/#11621 code: they matched on 2026-10-08, but #11621's head has since
moved to `e52d7fa4` by two merges of xarray `main`. Keep patch 7b, because ordinary update
refusal and the hook's opt-in preservation are separate behaviors. Split the generic halves of
patches 4 (custom-index preservation through broadcast/alignment) and 9 (unaffected indexes
through coarsen/construct) from their hook additions, and prove each with an xarray-only
regression. Patch 4 may still need the hook; if so, report the dependency rather than claim it
stands alone. Following the [hook design](xarray-upstream/index_hook_design.md), fold the
stack/pad/coarsen vetoes into one preflight method and analyze joins, override and unindexed
conflicts at both the alignment and merge call sites (merge `compat` is not alignment `join`).
Prove the two-method contract or document the concrete need for a third. `swap_dims` stays
separate. Update `BindingIndex`, capability detection and the xarray regressions together, then
run the xarray suites, xarrayrf suites and probe and compare skips/xfails/xpasses against
series 4. Aim for two independently evidenced extension cases per hook (time-box
pint-xarray/cf-xarray). Check the opt-in conflict handling of the
[#11607](https://github.com/pydata/xarray/issues/11607) prototype, which awaits maintainer
feedback. Write the manifest and the design-issue and generic-fix drafts for the maintainer.
`index-hooks-4` and `xarrayrf-patches-4` stay unchanged; a new tag and pin bump need
publication approval, after which update the patched group pin, pretend version and lock
together.
*Series 5 done 2026-10-24:* tag `xarrayrf-patches-5` (`a6576179`, [manifest](xarray-upstream/xarray_patches.md))
implements the two-hook design on `b2acc77a` with the fix halves as separate commits, and the
pin, lock and `BindingIndex` moved to it. Still open: the extension evidence
(pint-xarray/cf-xarray), the #11607 prototype check, and the design-issue and fix-PR drafts;
the maintainer decisions on the hook contract are in the
[hook design](xarray-upstream/index_hook_design.md).

**R3: hosted CI for the patched pin** [supports criterion 1]. Add a separate job using the
`patched` group (`make test-pinned`). Log the imported xarray path, version and commit. Run the
full suite and the probe, and require a nonempty case set with every case `classification ==
"ok"`; a zero exit code alone is not enough. Show that a bad pin fails the job. Leave the four
existing jobs unchanged. Pushing needs approval.
*Done 2026-10-24:* `4ced97e` adds the `patched` job and
`tools/check_binding_probe.py`. [run 40114374947](https://github.com/matthiasschabel/xarrayrf/actions/runs/40114374947) installs series 4 (`5db75d53`), passes 3,059 tests and
gates 93/93 `ok`; the negative control [run 40114398033](https://github.com/matthiasschabel/xarrayrf/actions/runs/40114398033) pins the unpatched base `dfd25c72` and fails at
the gate, naming each hole. On the series-5 pin,
[run 40152700415](https://github.com/matthiasschabel/xarrayrf/actions/runs/40152700415) installs
`a6576179`, passes 3,108 tests and gates 128/128.

### Waiting on others

**R5: upstream merges and a released-xarray check** [criterion 1].
[#11617](https://github.com/pydata/xarray/pull/11617) is open at `81a8f22c` with no reviews;
the open questions are scalar `drop=True` and partial shared-index refusal.
[#11621](https://github.com/pydata/xarray/pull/11621) (head `e52d7fa4`) has changes requested:
the 2026-09-28 review asks that `ds["nir"] = nir` with a conflicting default index raise rather
than replace the custom index, with `drop_indexes()` as the explicit resolution; re-review was
requested on 2026-10-05 and nothing newer has arrived. When fixes ship, run the xarray-only
reproducers, xarrayrf contract tests and probe against the installed wheel before proposing a
floor bump. If upstream rejects a fix or the hooks, that needs a maintainer decision; do not
weaken the contract.

### Waiting on the maintainer

**R4: dropped for the pre-alpha release** [criterion 3]. The maintainer decided on
2026-10-14 not to freeze persistence schema 1 now: xarrayrf is pre-alpha with no released
versions, so no files need protecting. Schema 1 stays provisional; new record kinds (field
transforms stage 1b among them) are added within it. The freeze returns before the first release
users are expected to keep files from, with this preparation: compare `encode`/`decode` and the
native binding metadata against the [persistence design](architecture/persistence_design.md);
check numeric/dtype handling, integers above `2**53`, empty/scalar Grid coordinates, intervals,
anonymous identity, named-basis encoding and every kind added since schema 1 was written; re-run
the encoding and NGFF tests with real NetCDF/Zarr round trips and add representative-record
compatibility tests; record the date, fields and versioning policy.

**R6: TestPyPI rehearsal** [criterion 5]. The maintainer verifies the pending-publisher fields
against [releasing.md, One-time hosted setup](../releasing.md). Run its local checks in an
isolated checkout, then present the SHA and version (`0.0.1.dev0`) for dispatch approval. After
upload, install core dependencies from PyPI and xarrayrf from TestPyPI with `--no-deps` outside
the checkout, then run `pip check` and the smoke test and inspect the metadata. Uploaded
filenames cannot be reused, so inventory the published files before any retry.
*Done 2026-10-24:* [run 40111808490](https://github.com/matthiasschabel/xarrayrf/actions/runs/40111808490) on `3bc80f2` uploaded `0.0.1.dev0`; install,
`pip check`, smoke test and metadata review passed
([release notes](architecture/release_and_repository_notes.md)). Re-run it with a new version
when the packaging changes before R7.

**R7: production approval**, once R1–R3, R5 and R6 close. Compile one readiness record, rebuild and
install the final artifacts, and recheck floors, changelog and classifier. Production Trusted
Publishing does not exist yet; the workflow only uploads to TestPyPI. The PyPI pending
publisher is registered; the production job must match its workflow and environment fields. A patched-xarray
distribution is a separate product decision.

### Fixed constraints

- Native xarray objects and explicit identity only: no wrapper, registry, monkey patch or
  weakened required subset.
- Keep stock `.venv` and `.venv-patched` separate, and identify a lane by source path plus
  commit (version metadata goes stale under `PYTHONPATH`). Don't create environments inside
  the PR worktrees.
- No public action (tag, pin publication, CI push, TestPyPI dispatch, upstream post) without
  specific approval.

### Local commands

From the xarrayrf root, after `uv sync --locked --extra dev`:

```sh
make check test
.venv/bin/python tools/binding_operation_probe.py
make test-patched XARRAY_PATCHED="$HOME/GitHub/xarray-hooks-5"
PYTHONPATH="$HOME/GitHub/xarray-hooks-5" .venv/bin/python tools/binding_operation_probe.py
make test-pinned   # provisions .venv-patched from the published pin
```

Check the probe's reported `xarray_module` and the exit code as well as the case results.
Worktree roles are in the [patch manifest](xarray-upstream/xarray_patches.md) (series 4) and
[upstream PR notes](xarray-upstream/upstream_prs.md) (PR worktrees); recheck their heads before
acting.

## Alternatives Considered

- Treating green stock CI as release readiness: the strict xfails and missing patched evidence
  leave the contract unverified.
- Blocking all work until upstream merges: R1–R3 can proceed now; only final closure waits.

## Deferred Work

The [roadmap's Parked table](roadmap.md#parked-with-what-would-bring-each-item-back) owns
non-release work.

## Next Steps

1. Finish R2's evidence and drafts; settle the hook-contract decisions with the maintainer.
2. Keep status in the roadmap and evidence in the area notes; update this note when a thread
   closes and mark it `Implemented` after R7.
