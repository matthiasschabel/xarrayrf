# Field transform persistence (stage 1b)

**Status:** Implemented
**Last updated:** 2026-10-15
**Scope:** `xarrayrf.encode`/`decode`, `DataArray.rf.encode`/`decode`, a container-level
Dataset API in `xarrayrf.native`, and the persistence-related planning docs. Implements the
[field transform design](field_transform_design.md), section "Persistence (1b)".

## Context

`encode` refuses `DisplacementFieldTransform`, `BSplineTransform`, `DeclaredInversePair` and
`PairedTransform` (`_encoding.py`, "stage 1b" messages). Schema-1 records are inline JSON;
`rf.encode()` returns one DataArray and cannot carry a companion array. A binding accepts any
point transform, so a framed array can be bound through a field, and a registration result (a
composite or a declared pair between frames) is commonly persisted on its own.

**Schema decision (maintainer, 2026-10-14):** no schema bump and no freeze. 1b adds new kinds
within schema 1, leaves every existing kind and the `{"xarrayrf": 1, "value": ...}` envelope
unchanged, and schema 1 stays provisional. xarrayrf is pre-alpha with no released versions, so
there are no files to protect; a freeze belongs to the first release users are expected to keep
files from. This replaces the design's "schema bump" bullet.

## Current Decision

### Records (new kinds in schema 1)

| Kind | Fields |
|---|---|
| `displacement_field_transform` | `source`, `target`, `grid`, `outside`, `field` |
| `bspline_transform` | `source`, `target`, `grid`, `order`, `outside`, `support`, `coefficients` |
| `declared_inverse_pair` | `forward`, `backward` (each a full transform record) |
| `paired_transform` | `pair` (a `declared_inverse_pair` record), `direction` (`"forward"`/`"backward"`) |

`grid` uses the existing `grid` kind. `field`/`coefficients` is an array reference
`{"reference": "sha256:<64 hex>", "dtype": "float32" | "float64"}`. Shape is not recorded: the
constructor checks it against the grid sizes and the target's axis count.

**References are content digests.** The reference is SHA-256 of a header naming the dtype and
shape (`"float32:2,6,2\n"`) followed by the array's C-order, little-endian bytes. The header
keeps arrays with equal bytes but another shape or dtype (zero fields, say) from sharing a
reference (Codex B1). One string is both locator and digest, so identical arrays deduplicate,
two encodes into one store never collide, and decoding verifies data by recomputing it. Digests
are computed at encode and decode, never at construction (equality stays identity, as decided
for 1a). The stored view is component-last over component-major storage, so the bytes are
hashed in C-order blocks of at most 16 MiB: a leading axis is split recursively when one of its
rows is larger (a 1×2048×2048 slab), and the last axis holds only the components (Codex N1).
Measured against a whole-array `tobytes` hash, digests equal: 256³×3 float64 (400 MB) 0.19 s,
16 MB peak; 1×2048×2048×3 float64 0.04 s, 17 MB peak.

A pair is one record owning both directions; a `PairedTransform` encodes its whole pair plus the
direction, so decoding a half returns that direction of a freshly decoded pair with its
original source and target. Pair members can be any encodable non-affine transform (a field, a
user extension), encoded recursively. Decoded field objects are new objects and compare unequal
to the originals (identity equality); round-trip tests compare re-encoded records and arrays.

### Core API (`_encoding.py`)

- `encode(value, *, arrays=None)`: `arrays` is a caller-supplied `MutableMapping[str, ndarray]`.
  Each field array is stored under its reference as the transform's read-only view (no copy); an
  existing key is left as is (same digest, same content). A field with `arrays=None` raises
  `TypeError` naming the keyword. Values without fields encode exactly as today.
- `decode(data, *, decoders=None, arrays=None)`: `arrays` is a `Mapping[str, ArrayLike]` (a
  dict, an `xr.Dataset`-derived mapping, a Zarr group): the resolver. Per reference:
  1. Missing `arrays`, or reference absent: new `MissingArrayError(EncodingError)`, the
     unavailable-provider case; nothing is guessed.
  2. A `numpy.ma.MaskedArray` entry is `MalformedDataError` before conversion, since
     `np.asarray` would drop the mask the constructor refuses (Codex N2). `np.asarray` the
     entry; its dtype must be a floating dtype of the recorded item size in
     either byte order (netCDF readers may return big-endian), else `MalformedDataError`. It is
     converted to native byte order, since the constructor keeps only native `float32`/`float64`
     and would otherwise widen to float64.
  3. Recompute the digest; a mismatch is `MalformedDataError`.
  4. Construct through the public constructor, which copies into immutable storage; its
     refusals (shape, finiteness) become chained `MalformedDataError` as for every kind.
- The encoder threads `arrays` through recursion (composites, pairs, grids are unaffected).
  `_FIELDS` gains the four kinds; `_BUILT_IN` lists them.
- New public names: `MissingArrayError` (exported with the other `EncodingError` subclasses).

### Native API (`native.py`)

- `DataArray.rf.encode(*, arrays=None)` and `rf.decode(*, decoders=None, arrays=None)` pass
  `arrays` through. Without `arrays`, a binding containing a field raises `TypeError` naming
  `arrays=` and `rf.to_dataset()`.
- **Container level**, in `xarrayrf.native` (settles the design's open decision on location):
  - `DataArray.rf.to_dataset(*, name=None) -> xr.Dataset`: `rf.encode(arrays=...)` stored as a
    data variable (`name`, else `array.name`; neither is `ValueError`, as in xarray), plus one
    reserved data variable per referenced array.
  - `native.array_from_dataset(dataset, *, name=None, decoders=None) -> xr.DataArray`: picks
    `name` or the single non-reserved data variable (none or several is `ValueError`), builds the
    resolver from the reserved variables and calls `rf.decode`.
  - `native.value_to_dataset(value) -> xr.Dataset` and
    `native.value_from_dataset(dataset, *, decoders=None)`: a standalone value (a registration
    chain, a pair) as canonical JSON in `dataset.attrs["xarrayrf_value"]` plus reserved array
    variables. This is the single save path for a transform that is not bound to an array.
  - Reserved array variables are named `xarrayrf_array_<first 16 hex of the digest>` with dims
    `<name>_dim<i>`, and carry `attrs["xarrayrf_reference"]` (the full reference). Content-derived
    names let datasets holding the same field merge with `xr.merge`. Writing refuses any
    reserved name already present among the array's coords or the dataset's variables or dims
    (`ValueError`), and two different references sharing a 16-hex prefix (`ValueError`, never
    expected). Reading maps references from the attribute, never from names; a reserved variable
    without a string reference attribute, or two variables claiming one reference, is
    `MalformedDataError`.
  - Users write the Dataset with `to_netcdf` or `to_zarr`; xarrayrf imports neither backend.
    Reading is lazy: only referenced arrays are loaded.

### Schema documentation

- `persistence_design.md`: add the four kinds to the introduced-kinds table (schema 1, field
  transforms stage 1b, unreleased); describe array references, the resolver and
  `MissingArrayError`; state that schema 1 stays provisional and the evolution rule binds from
  the first release users are expected to keep files from.
- `field_transform_design.md`: replace "schema bump" in the 1b split and Persistence sections;
  record the content-digest reference and the container API location; close open decision 3's
  remaining question.
- `roadmap.md` criterion 3 and `release_plan.md` R4: no freeze gate for the pre-alpha release;
  remove the R4↔1b coupling.

## Tests (`tests/test_encoding.py`, `tests/test_native_encoding.py`)

1. Each field kind round-trips through `encode(arrays=)`/`decode(arrays=)`: re-encoded record
   equal, arrays bit-equal, dtype kept (float32 and float64), `support="half-open"` and each
   `outside` kept, points map identically; `json.dumps` round trip of the record.
2. A composite `[affine, displacement, affine]` and both halves of a declared pair (field
   members) decode with original endpoints and `declared_inverse()` returning the other
   direction; a pair with a user-extension member decodes through `decoders`.
3. Refusals, each asserting its class: no `arrays` at encode (`TypeError`); missing resolver and
   missing reference (`MissingArrayError`); wrong dtype, digest mismatch, wrong shape (chained
   `MalformedDataError`); a masked entry whose data matches the digest (`MalformedDataError`);
   arrays with equal bytes but different shapes or dtypes encode into one store and each
   decodes; identical arrays deduplicate; unknown fields and bad `direction`/`outside` (`MalformedDataError`).
   A big-endian float32 array decodes and stays float32.
4. Container: framed array bound through `CompositeTransform(grid affine, displacement)` via
   `rf.to_dataset` → `to_netcdf(engine="scipy")` → `open_dataset` → `array_from_dataset`; same
   through `to_zarr`/`open_zarr`; a pair through `value_to_dataset`/`value_from_dataset` on both
   backends (acceptance test 9). Affine-only arrays and values use the same path. Reserved-name
   collisions, ambiguous variable selection and malformed reference attributes refuse.
5. Existing schema-1 records still decode unchanged (existing tests stay green).

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Schema 2 for records with references | No existing kind or envelope changes; pre-alpha, no files to protect (maintainer, 2026-10-14) |
| Caller-chosen or sequential reference names | Collide across encodes into one store; a separate digest field would still be needed |
| A separate `encode_with_arrays` returning a tuple | Two encode entry points; `arrays=` mirrors `decode(arrays=)` and lets several values share a store |
| A callable resolver | A `Mapping` already covers dicts, Datasets and Zarr groups and matches `decoders` |
| Recording shape in the reference | Redundant with the grid; the constructor checks it |
| Memoizing pairs so both halves in one payload share a pair | No consumer encodes both halves of one pair in one value; identity equality permits separate pairs |
| Container API in a new subpackage | `native` already owns the DataArray persistence and is the integration layer the core layering allows to import xarray |

## Deferred Work

- Lazy (Dask-backed) fields remain deferred with the design; decoding loads the array.
- Opaque preservation of undecodable records, and migration once a schema is frozen.
- NGFF export of field transforms (NGFF adapter, after 1c import).

## Next Steps

Done. Plan accepted by Codex (gpt-6-astra) after two convergence passes; the implementation
review accepted with two non-blocking findings, both fixed: `array_from_dataset` keeps
non-string variable names, and the laziness test now uses arrays that fail if computed.
