# Data-only persistence of the core value objects

**Status:** Active
**Last updated:** 2026-09-30
**Scope:** `xarrayrf.encode` and `xarrayrf.decode` for vocabularies, coordinate systems, frames,
array coordinates, grids, affine and composite transforms, and user-defined transforms. Native binding
persistence wraps this encoding; NGFF, CF and NIfTI formats are adapters.

## Context

[The architecture](../../design.md) §8 requires versioned, data-only persistence: no pickled live
objects, no import path, callback or network lookup driven by the data, and distinct failures for
an unknown schema or kind, a malformed declaration and an unavailable provider. The core
interface requires a bound transform to persist only through such an encoding.

## Current Decision

- `encode(value)` returns `{"xarrayrf": 1, "value": {"kind": ..., ...}}`: plain JSON whose fields
  mirror constructor arguments. `json.dumps` round-trips it exactly (finite float64 via `repr`;
  constructors refuse non-finite values).
- `decode(data, *, decoders=None)` validates the whole payload as JSON types first (string keys,
  finite numbers), checks numbers strictly (no strings or booleans coerced), then rebuilds through
  the public constructors. Unknown fields are refused so newer data is never silently truncated;
  additions after schema freezing bump the version.
- Identity survives: the identifier is decoded with `ReferenceFrame.declared`, so a local frame
  decodes equal to the original. Metadata keeps the type distinctions the core compares.
- Schema 1 gains the `grid` kind before freezing: `transform` uses the existing transform
  encoding, `dims` lists the dimension order (JSON key order is not relied on), and
  `coordinates` maps names to `{"dim": ..., "values": [...], "dtype": ...}` for varying axes or
  `{"value": ..., "dtype": ...}` for retained scalars. `intervals` is always emitted as an
  object mapping source axis names to `[lo, hi]` rows (possibly empty). Decoding
  checks exact coordinate fields and numeric types, then rebuilds through `Grid`; constructor
  refusals are chained `MalformedDataError`. Stage 2 requires an explicit `"dtype": "int64"`
  or `"dtype": "float64"` on every coordinate record, including scalars and empty arrays.
  int64 accepts only JSON integers fitting in int64; float64 accepts finite numbers, including
  integral JSON numbers. Numeric kind survives tools rewriting `2.0` to `2`. Integer values
  never pass through float64, including values above 2**53. Missing or unknown dtypes refuse;
  there is no inferred kind or fallback for older records without dtype. Equality and hashing
  include the kind, and both kinds round-trip exactly. These changes remain within provisional
  schema 1.
- Native binding JSON has exactly `transform`, `dims` and `intervals`. Interval rows are
  checked as JSON arrays of numbers, refusing strings, booleans, non-finite values and unknown
  fields. Grid and binding decoding rebuild through validated constructors; invalid shapes,
  bounds, axis names, point-sampled declarations and offset disagreement become chained
  `MalformedDataError`. Selected scalars retain a single `(2,)` row. No version bump: schema 1
  remains provisional.
- A vocabulary stores the lexicographically smaller token of each antipodal pair.
- User transforms implement `SupportsEncoding`: a namespaced `kind` (`"package:name"`), their own
  integer `version`, and `to_data()`. They decode only through caller-supplied `decoders`, which
  receive the data, version and decoded endpoints; the result must be a point transform whose full
  endpoint declarations equal the decoded ones.
- Errors are `EncodingError` (a `ValueError`) subclasses: `UnsupportedVersionError`;
  `UnknownKindError` (not built in, not namespaced); `MissingDecoderError` (namespaced without a
  decoder, the unavailable-provider case); `MalformedDataError` (bad fields or types, or a
  constructor refusal, cause chained); `DecoderResultError`. `encode` raises `TypeError` for an
  unencodable value.

### NGFF gate

§8 asks that NGFF be evaluated before a private schema is published. The gate in
`tests/test_ngff_persistence_gate.py` passes for the lossless list (`affine2d2d`, `affine2d3d`,
`byDimension1`, `byDimension2`, `identity`, `mapAxis1`, `projectAxis`, `projectAxis2`, `rotation`,
`scale`, `sequence`, `translation`, the one-level `multiscales_transformations` fixture, and a
synthetic oblique level exported as scale/translation with a level affine). Each imported value
survives schema-1 `encode`/`decode`; exported v0.6 affines reimport within 1e-12 and named
systems stay equal. Lossy and refused cases also behave as specified: channel and discrete drops
are reported; nonlinear, invalid `byDimension` and array-backed displacement cases refuse.

In reverse, export reports what v0.6 cannot hold: frame identity, definition, context, role,
display, vocabulary, orientation, non-centred sample offsets, and array-coordinate units (other
than `"1"`, including `None`) and axis types. Composite endpoints and non-affine transforms
refuse export.

Schema 1 remains provisional and is not frozen. Freezing it is an explicit maintainer decision.

## Alternatives Considered

| Alternative | Rejected because |
|---|---|
| Pickle | Executes code on load |
| NGFF 0.6 JSON as the native format | Loses identity, context, vocabularies, sample offsets and undeclared units; stays an adapter target |
| ASDF tags | A dependency in the NumPy-only core; an ASDF converter can wrap `encode`/`decode` |
| A registry of kinds or import paths | Code selected by data |
| `to_data`/`from_data` on every class | Spreads version handling; one module owns the schema |
| Propagating constructor errors unchanged | Callers could not tell malformed data from other failures |

## Deferred Work

- Migration from older schema versions, once version 1 is frozen.
- Opaque preservation of undecodable extension payloads without activating them (§8 permits it).

## Next Steps

None until the maintainer decides to freeze schema 1.
