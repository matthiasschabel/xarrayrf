# Completed codebase audit

**Status:** Implemented
**Last updated:** 2026-10-06
**Scope:** Five-stage correctness and API audit of xarrayrf at baseline `de48d8a`, completed
through `a0b7e6f` plus the Unicode DICOM destination correction.

## Context

The audit emphasized incorrect identities and sample associations, valid-input failures,
export semantics, comprehensible public contracts and output reliability. All five stages
are complete. The [roadmap](roadmap.md) owns remaining work; there is no agreed audit Stage 6.
The [interface](../core_interface.md) is normative and [CHANGELOG](../../CHANGELOG.md) records
user-visible behavior. This record replaces the audit plan, refinement transcripts, five
implementation reviews and obsolete Stage 5 handoff.

Codex independently reproduced the reported failures and applied QAEngineer review. Claude
reviewed plans and implementations through collaborative-refinement with explicit model
selection `claude-opus-5-5`; the CLI did not independently announce its model. Stages 2–5 were
implemented by Codex `gpt-6-astra`; Stage 1 and subsequent QA corrections by the root Codex
agent. Committed history retains the original review reports and detailed execution records.
No unresolved blocking review disagreement remained when each stage completed.

## Current Decision

Retain native xarray binding, explicit identity and registration, and the separation of frames,
transforms and sampling. A wrapper hierarchy or global frame registry is not warranted.

| Stage | Durable outcome | Implementation commits |
|---|---|---|
| 1 — identity/support/cardinality | Local NGFF readers canonicalize filesystem and symlink paths; singleton matching uses float64 roundoff with affine cancellation terms; DICOM validates original pixel-stack cardinality before selection. | `299eda4`, `b490720`, `3d97352` |
| 2 — valid-input failures | Scalar resampling targets work in affine/general and eager/Dask paths; `Geometry.points(axis_dim=..., units_coord=...)` avoids explicit output-name conflicts without changing `point_at` indexers. | `74e0021`, `f4a7d6a` |
| 3 — export/interpolation | NGFF intrinsic axis types/units survive permutations; mixed columns require explicit same-unit spatial axes. Linear interpolation excludes zero/roundoff-weight NaNs component-wise. | `e302e92`, `b1fd1b3` |
| 4 — API contracts | Queries use declared geometry order after pixel transposition. Grid framing refuses coordinate conflicts unless `replace_coordinates=True`. Extension-transform stability and conditional hashing are explicit. Indexed assignment replaces destination pixels and preserves its binding under xarray coordinate checks. | `aaa95d9`, `0f0e00a`, `846da58` |
| 5 — tools/documentation | Complete DICOM batch preflight, sibling staging and per-series publication prevent partial final series; failures report planned/published counts and permit independent later series. Cells interpolate interior gaps; singleton docs match runtime. | `febbe89`, `a0b7e6f` |

### Numerical and representation decisions

- Singleton allowance is a measured small multiple of float64 epsilon, including intermediate
  affine terms. Epoch-scale round trips differed by one ulp; a scale of `0.1234567` plus `1e9`
  translation produced source errors near `5.7e-7` and `9.5e-7`. Final-coordinate ulps alone
  cannot cover cancellation. A non-affine inverse resets propagated allowance; arbitrary
  providers do not inherit an affine error certificate.
- Linear missing weights and integral gather classification share `1e-9` source-position
  allowance. Large-origin oblique tests measured errors up to `4.66e-10`. Larger positive
  missing weights propagate NaN; complex components are independent. Cubic prefilter NaN
  spreading and SciPy infinity behavior remain documented limitations.
- Scalar affine kernels temporarily use a one-sample computational shape, retaining public
  scalar shape, target coordinates and cropped reads. General kernels use explicit zero-row
  position arrays. Global planner shape promotion was unnecessary.
- Imported DICOM `source_count` is retained before sorting/selection. Manually constructed
  metadata can leave its default unknown count, retaining index validation; selected indices
  cannot establish the original enhanced stack's cardinality.
- `dims` order is geometry declaration order, independent of pixel storage. NGFF explicitly
  requests storage-order columns. Replacing Grid coordinates changes declarations, not pixels
  or units; matching metadata survives and conflicting declarations require opt-in.
- Grid copies/freezes coordinates and intervals but retains transforms by reference. Extensions
  must keep endpoints, behavior and scalar equality stable; hashing requires a compatible hash.
- Indexed assignment is destination-frame payload replacement. Shared dimension coordinates
  must match; retained scalar coordinates on selected slices are ignored by xarray's check.

### DICOM tool limits and subsequent Unicode correction

Batch preflight refuses source/output overlap, duplicate or symlink-aliased inputs, conflicting
names, existing/dangling targets, invalid ancestors and empty sources. Every immediate regular
file, including dotfiles, is selected. Required valid UIDs and class/instance agreement with
file metadata are checked explicitly. Narrow expected errors clean unpublished staging while
independent later series continue. UID remapping is deterministic across saved/retried output.

Fresh QA found composed/decomposed Unicode destination names can alias on the local filesystem.
The comparison key now uses NFC then casefold, without changing supplied path spelling. Its
regression fails on `a0b7e6f`; the corrected tool has 29 passing CLI regressions. This correction
was uncommitted at consolidation and its evidence is preserved here rather than assumed to be
recoverable from Git history.

Publication is per series and assumes one writer; it is not a whole-batch transaction or crash
durability guarantee. An interrupt immediately after rename can leave a complete target before
its completion status is recorded; reconcile such a target with the manifest. The tool remains
an example-data copier, not a general confidentiality profile.

### Validation provenance

These are dated historical results, not a claim that the hosted dependency-floor lanes passed.

| Point | Stock xarray | Patched xarray | Independent evidence |
|---|---|---|---|
| Audit baseline | 1,617 passed, 47 xfailed | 1,664 passed | F1–F8 public reproductions; disputed reviewer interpretations checked against source. |
| Stage 1 QA | 1,638 passed, 47 xfailed | 1,685 passed | Eight failures against original source, six protective cases; 178 focused passes and mixed-axis probes. |
| Stage 2 QA | 1,696 passed, 47 xfailed | 1,743 passed | Scalar affine/general targets, eager/Dask, empty/scalar output and naming collisions. |
| Stage 3 QA | 1,739 passed, 47 xfailed | 1,786 passed | 34 failures against original source; 32 independent interpolation-oracle cases and NGFF semantic round trips. |
| Stage 4 final QA | 1,765 passed, 47 xfailed | 1,812 passed | 15 original-source failures; 72 ordering/export probes, coordinate replacement and scalar assignment boundaries. |
| Stage 5 final QA, with Unicode correction | 1,794 passed, 47 xfailed | 1,841 passed | Original 28 CLI regressions fail on `df385d3`; Unicode regression fails on `a0b7e6f`; 12 CLI probes, seven malformed-metadata cases, source preservation, gap oracle and unchanged executable Geometry/Grid AST. |

Final per-stage static checks passed Ruff lint/format, mypy and whitespace checks; Stage 5
checked 82 files and distribution payloads. Full stock/patched stages reported three existing
dependency warnings each. Later repository-cleanup validation belongs in the
[maintainer changelog](changelog.md), avoiding another duplicate current-status ledger.

## Alternatives Considered

- Metadata-only NGFF identities remain caller-resolved; neither cwd canonicalization nor
  wholesale migration of existing persisted identifiers to file URIs is justified.
- Missing-data correction remains bounded linear interpolation, not a generic spline rewrite.
- Output dimension names are explicit rather than automatically suffixed. `point_at` retains
  keyword indexers; custom naming there would take valid coordinate names.
- No implicit spatial typing for time/unit mixtures in NGFF, upstream assignment guard,
  transform-freezing framework or pre-release compatibility aliases.
- Reviewer source claims withdrawn after verification included supposed metadata reader
  adoption failure and reuse of NumPy-only `Grid.points` for labelled output naming.

## Deferred Work

**A4 — adapter Grid/report convenience.** Metadata results already carry dimensions,
coordinates, transforms and reports; DICOM ordering/intervals, NGFF levels and GeoTIFF nodata
must remain format-specific. Ordinary readers return DataArrays with `array.rf.grid`; no
README/tour/real-data consumer required a report channel. Reopen for an actual metadata-only
Grid or reader-report consumer, preserving provenance, DataArray returns and pixel laziness.
Do not put adapter reports in core geometry merely to make them visible.

**A6 — construction and mixed coincidence.** `frame_array` requires coordinates declaring
non-geometric dimension sizes. `xr.DataArray(data, dims=("channel", "i")).rf.frame(grid)`
already handles an unlabeled channel without pixel computation. `grid.is_coincident(geometry)`
refuses the concrete type; `grid.is_coincident(array.rf.grid)` provides the comparison.
Snapshots can materialize field coordinates. Reopen for a blocked construction/comparison
workflow, with focused shape, laziness, identity and tolerance checks; no new protocol hierarchy.

General-path many-context interpolation routing optimization needs profiling. Cubic missing-data
policy changes and Grid snapshot `RangeIndex` step parity need demonstrated use cases. Viewer,
nonlinear geometry, upstream releases and packaging gates remain separate roadmap work.

## Next Steps

No audit stage remains. Follow the [roadmap](roadmap.md), and reopen deferred conveniences only
when their usage triggers arise. Preserve the regression tests and normative contracts when
changing the affected modules; old execution transcripts are history, not a work queue.
