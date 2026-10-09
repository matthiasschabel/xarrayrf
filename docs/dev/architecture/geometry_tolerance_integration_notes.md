# Geometry tolerances and pirana integration

**Status:** Deferred
**Last updated:** 2026-10-09
**Scope:** Generic sampling queries, DICOM geometry import, and future pirana consumers

## Context

A pirana geometry audit considered configurable absolute and relative tolerances after
highdicom substituted nearby source coordinates during ParametricMap construction.
The current xarrayrf implementation already provides configurable sampling coincidence
and lattice approximation. No new xarrayrf API is demonstrated necessary by that case.

## Current Decision

Reuse existing APIs before adding tolerance abstractions:

| API | Current policy |
|---|---|
| `Geometry.is_coincident` / `Grid.is_coincident` | Default `1e-6` of a local sample step per dimension; valid `[0, 0.5)`; checks both operand directions |
| `Geometry.lattice` / `Grid.lattice` | Default `1e-6` of a step for uniform-coordinate approximation |
| `dicom.from_datasets` / `dicom.from_enhanced` | Orientation tolerance `1e-4`; slice tolerance `0.01` in steps for slice residuals and in pixel spacings for in-plane shifts |

The first two APIs require no millimetre convention. Coincidence requires equal or equivalent
frames and matching dimension names/sizes, pairs corresponding samples by dimension name,
and uses roundoff allowances on singleton dimensions rather than inventing a step. Its
per-axis sampling bound is not a Euclidean distance in physical units. Exact transform/index
compatibility remains distinct: approximate coincidence is not a transitive equivalence
relation and must not silently widen automatic arithmetic/alignment compatibility.

Core sampling queries belong in xarrayrf. Consumers choose their acceptance policy and
whether coincidence permits replacing a requested grid or skipping resampling. Generic DICOM
geometry import belongs in `xarrayrf.dicom`; repeated acquisition-slice grouping across
components, echoes, or time points and source-image lineage belong in pirana's reader.
The importer rejects near-duplicate positions in a single spatial stack; it must not replace
pirana's multidimensional acquisition assembly without an adapter-level partitioning step.

Pirana's highdicom evidence match radius stays with its export adapter: changing an xarrayrf
comparison tolerance cannot configure highdicom's source-plane matching or substitution.

Grounding files: `src/xarrayrf/_geometry.py`, `_grid.py`, `_coincidence.py`, `_sampling.py`,
`_positions.py`, `_affine.py`, and `dicom/__init__.py`. Detailed pirana/external library
values live in the sibling
[pirana audit](../../../../pirana-lab/docs/spatial/dev/geometry_tolerance_notes.md).
This is an integration proposal, not a change to current comparison contracts.

## Alternatives Considered

- **Global tolerance:** rejected. Sampling fractions, directions, physical distances,
  serialization precision, and approximation residuals have different meanings.
- **New absolute/relative position knobs immediately:** deferred. Existing step-based
  coincidence is origin-independent and unit-independent in its sampling metric.
- **Tolerance in equality or binding alignment:** rejected. Approximate comparison is
  non-transitive and does not establish transform identity.
- **Physical maximum-displacement query:** potentially useful, but a separate contract.
  A caller must define the norm, compatible units/space, sample correspondence, and
  singleton behavior before selecting this API. It is not implied by coincidence.

## Deferred Work

No required xarrayrf implementation change is identified. Reconsider only after a concrete
pirana integration case fails or requires awkward caller code:

- A physical-distance query if callers need a millimetre bound on corresponding samples.
- Separate DICOM importer tolerances if the existing slice tolerance's roles (uniformity,
  in-plane shifts, and duplicate refusal) need different values in a demonstrated workflow.
- Migration of pirana's spacing classifier, after comparing its absolute/relative gap rule
  and accumulated residual guard with xarrayrf's end-to-end-step residual criterion.

## Next Steps

Implement a pirana producer adapter and one consumer later. Validate anisotropic spacing,
nonuniform slices, singletons, LPS/RAS equivalent frames, unrelated spaces, repeated
acquisition slices, and requested-grid versus source-grid substitution. Preserve pirana's
current acceptance policies until differences are explicitly reviewed.

The documentation was checked against the source on 2026-10-09. No runtime code changes
are included; integration and new APIs remain deferred.
