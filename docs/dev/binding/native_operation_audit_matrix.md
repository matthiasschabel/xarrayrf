# Native-operation audit matrix

**Status:** Implemented
**Last updated:** 2026-10-25
**Scope:** R1a design for the R1 native-operation release gate; 35 additional cases, with
contract-derived expectations written before execution. Implemented in the probe and
`tests/test_native.py`; results are in the [inventory](binding_operation_inventory.md).

## Context

[R1](../release_plan.md#current-decision) extends the
[93-case inventory](binding_operation_inventory.md) beyond its single 2-D affine fixture.
This note specifies new inputs and composed paths, not replacement measurements. Expectations
come from [design §§1–5](../../design.md), the
[binding design](binding_design.md), and code at `6508422712bfe288e0f84c632feedd0c9bed4971`.
The existing notes were checked against that code: joint ownership, shared-grid extraction,
fixed terms, non-affine rename refusal and optional hooks remain implemented as described.
The inventory's historical totals and the upstream notes' remote PR status are not fresh evidence.

Done for R1a means a reviewed matrix with explicit oracles and named open questions. Running
it, assigning failures and closing R1 are subsequent work. A correct-looking but misplaced
binding is a release blocker; an unexpected exception or loss on a preserving path is also
a failure, even when an older lane is known to lack a patch.

## Current Decision

Use these 35 cases without cross-multiplying fixtures, joins, operand orders or backends.
The construction column extends the shared setup below; each row starts with fresh objects.
Python here is specification only, not executed setup or a new implementation.

### Fixtures and notation

Reuse `raw`, `image`, `transform`, `frame` and imports from
`tools/binding_operation_probe.py:116`: `raw` has dimensions `(y, x)`, labels
`y=[0,2,4]`, `x=[0,3,6,9]`, and coordinate units `"1"`; `image` binds the identity affine
from `(y,x)` into the shared local frame `(a,b)` in mm. In the rows, `f = image`.

```python
f = image
px = xr.DataArray([10, 20, 30, 40], dims="x", coords={"x": [3, 6, 9, 12]}, name="profile")
px_equal = px.assign_coords(x=[0, 3, 6, 9])
bad = raw.assign_coords(x=[1, 4, 7, 10]).drop_indexes("x")  # y equal/indexed; x unequal/unindexed
shifted = raw.assign_coords(x=[3, 6, 9, 12])  # unequal *indexed* labels: ordinary alignment
plane = f.isel(y=1)
channel = f.expand_dims(channel=[0, 1])
point = f.isel(y=1, x=2)

# Source-component order and declared geometry order both differ from pixel storage order.
ordered_transform = AffineTransform.from_matrix(
    source=ArrayCoordinates(("x", "y"), ("1", "1")),
    target=frame,
    matrix=[[2, 0.5], [-1, 3]],
    translation=[11, -7],
)
ordered = raw.rf.frame(ordered_transform, dims=("x", "y"))

system = frame.coordinate_system
anonymous_frame = ReferenceFrame.anonymous(system)
anonymous_transform = transform.with_endpoints(target=anonymous_frame)
anonymous = raw.rf.frame(anonymous_transform, dims=("y", "x"))
independent = raw.rf.frame(
    transform.with_endpoints(target=ReferenceFrame.local(system)),
    dims=("y", "x"),
)
anonymous_other = raw.rf.frame(
    transform.with_endpoints(target=ReferenceFrame.anonymous(system)),
    dims=("y", "x"),
)
lazy = raw.copy(data=da.from_array(raw.data, chunks=(2, 2))).rf.frame(
    transform,
    dims=("y", "x"),
)
```

The complete identity control uses the deliberately shared `frame`; separately minting a
local frame does not relate two images. Missing units on `px` inherit the framed coordinate's
units under design §3. `bad` deliberately changes a protected, unindexed coordinate, whereas
`shifted` supplies alignment keys. Do not conflate those assertions.

For the field rows, use a displacement field inside a composition: field endpoints themselves
must be reference frames, while a native binding starts at array coordinates
(`src/xarrayrf/_field_transform.py:677`, `src/xarrayrf/_composite.py:25`).

```python
from xarrayrf import CompositeTransform, DisplacementFieldTransform, Grid

field_grid_transform = AffineTransform.from_matrix(
    source=ArrayCoordinates(("y", "x"), ("1", "1"), sample_offset=(0.5, 0.5)),
    target=frame,
    matrix=np.eye(2),
    translation=np.zeros(2),
)
field_grid = Grid(field_grid_transform, {"y": ("y", [0, 2, 4]), "x": ("x", [0, 3, 6, 9])})
yy, xx = np.meshgrid(
    np.array([0, 2, 4], dtype=np.float64), np.array([0, 3, 6, 9], dtype=np.float64), indexing="ij"
)
warp = DisplacementFieldTransform(
    source=frame,
    target=ReferenceFrame.local(system),
    field=np.stack((yy * xx / 100, xx / 10), axis=-1),
    grid=field_grid,
    outside="raise",
)
field_transform = CompositeTransform(transform, warp)
field_image = raw.rf.frame(field_transform, dims=("y", "x"))
```

All field-row samples remain inside support. At original labels the independent analytic
oracle is `(y + y*x/100, x + x/10)` in the warp target frame. Construction needs no SciPy;
world-point evaluation later requires the resample extra. Missing that dependency is a reported
unrun case, never a passing refusal.

### Lane annotations

The **outcome** column is the contract in every lane. The following codes predict where
implementations may fail it; they are not alternate passing outcomes. Patch numbers refer to
the [series-4 manifest](../xarray-upstream/xarray_patches.md#series-4-current-published).
For planning, S means stock 2026.7.0 (the floor), U means the recorded upstream `4929aa90`
with #11613/#11615/#11616, and P means series 4 `5db75d53`. Before execution, identify the actual
stock wheel and current upstream commit and revise predictions for newly merged patches.

| Code | Expected lane distinction and deciding patch |
|---|---|
| — | No known lane distinction; no patch prerequisite identified. Still an unmeasured case. |
| J | Mixed/partial alignment: S may fail before merged #11532; U can lose ownership, especially in reverse order. P uses patches **3+4** (`join_overlapping` and alignment refinements). |
| C | Protected unindexed conflicts: S/U can discard evidence. P must refuse via patch **3**, with alignment integration in **4**; #11607's prototype is not a prerequisite. |
| U | Dataset update: S can fail alignment; U can replace the binding. Patch **7** alone refuses an index-type replacement; **7b**, depending on **3+7**, permits compatible labelled updates. Do not count patch 7's refusal as success for a preserve row. |
| R | Dataset geometry reduction: S can split ownership; U has #11616; P has patch **6**. |
| D | Partial scalar `drop=True`: S/U may retain the unwanted scalar; P patch **2** must raise. |
| E | Empty roll: S can raise `ZeroDivisionError`; U/P include #11613 (retired patch **8**). |
| W | Dimension swap: S/U can split ownership; P patch **5** must keep it whole. |
| B | Broadcast with one geometry dimension excluded: S lacks #11615; U has the unfiled `CoordinateValidationError` candidate. P patch **4** is the preservation candidate, but exclusion coverage is unproved; no manifest entry establishes that this subcase is fixed. |

### Cases

Source abbreviations: **D§n** = `docs/design.md` section n; **B:section** = the named section
of `docs/dev/binding/binding_design.md`; **I:line** = `src/xarrayrf/_binding.py:line`.
Every `refuse` names the exact class (not all its subclasses) and a literal message substring.
Observe the exception at the operation, not merely when inspecting its damaged result.

| ID | Construction beyond fixtures | Operation | Expected outcome and contract source | Lanes |
|---|---|---|---|---|
| N01 | `f`, `px` share only x | `f + px` | **preserve** at x `[3,6,9]`, all original y; complete frame inherited. D§3, §5 minimum subset; I:288. | J |
| N02 | Same as N01, fresh inputs | `px + f` | **preserve**, same labels/world points as N01 despite reversed operand/storage order. D§3; I:288. | J |
| N03 | `f`, `px` | `xr.align(f, px, join="outer")[0]` | **preserve**, x `[0,3,6,9,12]`; the original affine locates new labels too. D§3–4; I:288. | J |
| N04 | `px`, `f` | `xr.align(px, f, join="exact")` | **refuse** `ValueError`, `coordinate 'x' differs for join='exact'`. I:317–319; D§3. | J |
| N05 | `left = f.to_dataset(name="owner")`; `right = px.to_dataset()` | `xr.merge([right, left], join="inner")` | **preserve** `owner` at x `[3,6,9]`; **unframe** `profile`, which lacks y and makes no full-grid claim. D§3; B:Dataset: shared-grid rule. | J |
| N06 | `bad` retains equal indexed y but conflicting unindexed x | `xr.merge([f.to_dataset(name="owner"), bad.to_dataset(name="incoming")])` | **refuse** `ValueError`, `conflicting unindexed coordinate 'x'`. D§5 negative cases; I:333–346. | C, J |
| N07 | `ds = f.to_dataset(name="owner")`; `incoming = (raw * 2).to_dataset(name="extra")` | `ds.update(incoming)`; inspect `ds["owner"]`, `ds["extra"]` | **preserve** both on the destination grid; same labels supply no new geometry claim. D§3; B:Dataset: shared-grid rule. | U |
| N08 | `ds = f.to_dataset(name="owner")` | `ds.update(bad.to_dataset(name="incoming"))` | **refuse** `ValueError`, `conflicting unindexed coordinate 'x'`; equal y must not hide unequal x. D§5 negative cases; I:333–346. | C, U |
| N09 | `ds = f.to_dataset(name="owner")` | `ds.update(shifted.to_dataset(name="incoming"))` | **refuse** `AlignmentError`, `cannot update Dataset: coordinate`; the message must also contain `Drop the incoming index first`, and `ds` must be unchanged. Incoming `PandasIndex`es would replace the Dataset's binding; patch 7 (#11621, per its maintainer review) raises, and 7b accepts only equal labels, so unequal indexed labels are not a silent left join. B:Dataset: shared-grid rule; I:320–321 (`how="left"` applies only after 7b accepts). | U |
| N10 | `ds = xr.Dataset({"owner": ordered, "series": (("t", "y", "x"), np.ones((2, 3, 4)))})` | `ds["series"].isel(t=1)` | **preserve**, declared geometry order `(x,y)` despite pixel order `(y,x)`; a full-grid sibling inherits geometry. D§1; B:Dataset: shared-grid rule. | — |
| N11 | `ds = xr.Dataset({"owner": f, "profile": ("x", np.arange(4))})` | `ds["profile"]` | **unframe**: the variable lacks geometry dimension y; no frame may be borrowed from its owner sibling. D§5; B:Dataset: shared-grid rule. | — |
| N12 | `ds = f.to_dataset(name="owner")` | `ds.coords["x"]` | **unframe**: coordinate extraction is explicitly excluded from Dataset shared-grid inheritance. D§4–5; B:Dataset: shared-grid rule. | — |
| N13 | `channel` | `channel.sel(y=2, x=[9,3]).mean("channel")` | **preserve**, fixed y=2 and x `[9,3]`; nonspatial averaging must not erase the selected plane. D§1, §4; B:Index behaviour. | — |
| N14 | `ds = channel.to_dataset(name="owner")` | `ds.sel(y=2).mean("x")["owner"]` | **unframe**: averaging the remaining geometry axis is not selection of a point; even retained scalar y must lose its binding. D§4; B:Dataset: shared-grid rule. | R |
| N15 | `point` has no geometry dimensions and two fixed terms | `point + 3` | **preserve**, exactly one world point `(2,6)`, not an empty geometry or a newly minted identity. D§1, §3; I:140–169. | — |
| N16 | `point` | `point.sel(y=2)` | **refuse** `KeyError`, `source axis 'y' is fixed and cannot be selected by label`. Fixed terms are not label indexes. B:Index behaviour; I:134. | — |
| N17 | `empty = f.isel(y=slice(1,2), x=slice(0,0))` | `empty.roll(x=1, roll_coords=True)` | **preserve**, sizes y=1/x=0, no world samples; retain identity, transform and ownership rather than accepting vacuous numerical equality. D§1, §4; B:Coarsen, stack, pad, roll; I:392. | E |
| N18 | `one = channel.isel(y=slice(1,2), x=slice(2,3))` | `one.squeeze(("y", "x")).mean("channel")` | **preserve**, both geometry terms fixed at `(2,6)`; reducing channel is still nonspatial. D§1, §4; I:140–169. | — |
| N19 | `plane` | `plane.assign_coords(y=2)` | **refuse** `ValueError`, `would corrupt the following index`. Even equal-valued partial reassignment is not explicit re-framing. D§4 coordinate replacement; B:Invariants enforced in code. | — |
| N20 | `plane` | `plane.drop_vars("y")` | **refuse** `ValueError`, `would corrupt the following index`. Removing a fixed term loses the plane's location. D§1, §4; B:Invariants enforced in code. | — |
| N21 | `selected = ordered.isel(y=1)` | `selected.rename(y="row", x="column")` | **preserve**, rename both fixed and varying source references; compare column to original x and row to original y. D§4; B:Index behaviour; I:406. | — |
| N22 | `plane` | `plane.drop_indexes(["y", "x"])` | **unframe**: deliberate removal of the entire joint index is a documented exit; coordinates remain, no residual binding. D§5; B:Invariants enforced in code. | — |
| N23 | `ordered` | `ordered.isel(x=[3,1], y=slice(1,None)).transpose("y", "x")` | **preserve**, geometry order `(x,y)`, pixel order `(y,x)` and asymmetric affine components respected. D§1, §4; I:140, 186. | — |
| N24 | `anonymous`, `px` | `px + anonymous` | **preserve** the original anonymous identity at x `[3,6,9]`; plain values make no rival identity assertion. D§2–3; `src/xarrayrf/_frame.py:166`; I:288. | J |
| N25 | `f`, `independent` have numerically identical placements but distinct complete identities | `f + independent` | **refuse** `ValueError`, `incompatible framed coordinate mappings`. Numerical coincidence cannot establish identity. D§2, §5; I:198–204. | — |
| N26 | `derived = anonymous.isel(x=[1,2])` | `anonymous + derived` | **preserve** the shared anonymous identity on x `[3,6]`; selecting a descendant does not create a new world. D§2–3; `src/xarrayrf/_frame.py:166`; I:186, 279. | — |
| N27 | `anonymous`, `anonymous_other` | `anonymous + anonymous_other` | **refuse** `ValueError`, `incompatible framed coordinate mappings`. Separately minted anonymous identities remain unrelated. D§2, §5; `src/xarrayrf/_frame.py:166`; I:198–204. | — |
| N28 | `anonymous`, `f` | `anonymous + f` | **refuse** `ValueError`, `left operand is anonymous`. A complete frame must not silently replace anonymous identity. D§2, §5; I:198–204; `src/xarrayrf/_frame_compatibility.py:18` (`binding_difference`). | — |
| N29 | `lazy_channel = lazy.expand_dims(channel=[0,1]).chunk({"channel": 1})` | `lazy_channel.sel(y=2, x=[9,3]).mean("channel")`, then `.compute()` | **preserve** before and after compute, identical to N13's geometry; no pixel tasks before explicit compute. D§4–5 (lazy pixels). Dask reduction/graph construction differs from eager N13. | — |
| N30 | `ds = lazy.to_dataset(name="owner")`; `incoming = (raw * 2).chunk({"x": 2}).to_dataset(name="extra")` | `ds.update(incoming)`; inspect both variables, then `ds.compute()` | **preserve** as N07, with no pixel computation during update or geometry checks. D§3, §5; B:Dataset: shared-grid rule. Alignment must not eagerly compare pixel payloads. | U |
| N31 | `field_image` | `field_image.rename(x="column")` | **refuse** `ValueError`, `this transform cannot rename source axes`. The affine rename rewrite is unavailable for the field-backed composition. B:Index behaviour; I:413–414. | — |
| N32 | `field_image` | `field_image.swap_dims({"x": "column"}).isel(column=[3,1])` | **preserve**, x remains an auxiliary source coordinate along column; source names and the field transform stay unchanged. Check the nonlinear analytic oracle above. D§4; B:swap_dims; I:406, 439. | W |
| N33 | `band = xr.DataArray([1,2], dims="channel")` | `xr.broadcast(f, band, exclude={"x"})` | **preserve** first output (all geometry dims plus channel); **unframe** second output (channel/y only), which lacks x. Neither output may acquire an x dimension through an excluded index. D§3–4; B:Dataset: shared-grid rule (full geometry required). | B |
| N34 | `one = f.isel(y=slice(1,2))` | `one.isel(y=0, drop=True)` | **refuse** `ValueError`, `would corrupt the following index`. Singleton size does not permit loss of fixed y while x remains. D§5 scalar-drop rule; B:Invariants enforced in code; patch 2. | D |
| N35 | `selected = ordered.isel(y=1)` | `selected.isel(x=xr.Variable("points", [3,1]))` | **refuse** `ValueError`, `vectorized indexing changes geometry dimensions; call rf.unframe()`. Renaming the sample dimension is not a supported point-set conversion. B:Index behaviour; I:150–154. | — |

N01–06 cover partial coordinate participation without making two framed grids falsely
compatible. N07–12 exercise Dataset ownership; N13–18 composition and degenerate geometry;
N19–23 ownership and order; N24–28 identity; N29–30 only the distinct lazy paths; N31–32 the
field/affine distinction; N33–35 excluded dimensions and precise refusals. These rows add
coverage to the probe, though some controls already have focused tests in `tests/test_native.py`.

### Independent oracle and strict acceptance

Extend the approach of `tools/binding_operation_probe.py:38` (`_correct`), without deriving
expected points or expected axes from the result's transform:

1. Before the operation, snapshot each relevant source's transform, target identity, source
   axis names/order, declared geometry dimensions, coordinate labels/fixed terms, and world
   points keyed by the full source-label tuple. The expected domain and any rename map come
   from the row, not from the returned binding. Keep snapshots outside mutable update inputs.
2. Independently assert the result's sizes, coordinate values, fixed terms, dimension roles and
   target identity. Broadcast result coordinate **values** in the snapshotted source order
   (undo N21's name changes); evaluate the **original** transform on them. Compare that to
   `result.rf.geometry.points()`, reordered by named geometry dimensions, and compare matching
   source tuples to the saved original points. Never let a result's wrong source-axis order
   define the oracle. For N03's new x=12 labels the original unbounded affine supplies points;
   for N32 also use the independent analytic expression above.
3. Use `rtol=0, atol=1e-12` mm, matching the probe: these small synthetic affines and bilinear
   field values incur rounding far below unit-spaced labels. Identity/names/dimension roles
   are exact structural assertions, not approximate numerical comparisons. An empty result
   must pass those checks and the expected empty point shape; a zero-iteration label loop
   alone proves nothing. Scalar geometry must have one point with both fixed components.
4. Audit every specified output separately. N05/N33 deliberately mix framed and unframed
   outputs; a single aggregate `all_framed` flag is insufficient. `update` returns no useful
   array: audit the mutated Dataset, including owner and incoming variables. In N07/N30,
   also check incoming pixel values at matching labels/missing positions; correct geometry
   cannot reveal silent loss of update values. For refusals, verify destination inputs remain
   unchanged. For lazy rows, count Dask pixel tasks around construction/operation/geometry
   inspection and require zero; computation is a separate, explicit phase.

Replace `may_refuse`/`must_refuse` acceptance with one outcome per row/output. A refusal must
raise the specified exact type and contain the literal substring above; a broad
`except Exception` may record diagnostics but never confer a pass. In a later pytest
implementation, pair `pytest.raises(ExpectedError, match=re.escape(fragment))` with
`type(exc.value) is ExpectedError`. Check the named offending coordinate as well for generic
corrupt-index fragments (y in N19/N20/N34). A setup error is a broken fixture, not the row's
expected refusal. Inspection-time split-binding errors on preserve/unframe rows remain failures.

For old probe rows still labelled `may_refuse`, first derive the specific supported result or
specific refusal from their contracts (for example vectorized selection uses I:150–154,
geometry concat I:445–446). Leave undecided rows pending review; do not accept whatever was
observed. Unexpected S/U exceptions are recorded as holes with their actual types, even if a
strict xfail names the responsible patch. A new type/message requires contract review before
changing its assertion. Preserve and unframe rows accept no exceptions. Unframe requires no
remaining binding index on any coordinate, and `rf.is_framed` must return false without raising.

### Placement

Extend `tools/binding_operation_probe.py` with these stable case IDs and per-output expectations
after design review: it is the existing cross-lane inventory entry point and can report setup,
operation and inspection failures distinctly. Put regression assertions in `tests/test_native.py`,
beside its existing mixed-operation, Dataset, order, identity and non-affine rename tests;
reuse/extend those controls rather than duplicate them. Dask callbacks belong with its current
`pixel_tasks` helper. Field cases test native lifecycle behavior, so they belong there too,
with a small local fixture and an explicit optional-dependency skip for point evaluation.
No new test module or generic scenario framework is justified by 35 small cases. Both probe
and tests must use the same reviewed outcome specification; neither imports the other's runner
or uses observed lane outcomes as its expected behavior.

## Alternatives Considered

- A Cartesian matrix would mostly repeat affine arithmetic and obscure distinct ownership
  paths. Pair only N01/N02 for order and N13/N29, N07/N30 for eager/lazy paths.
- Keeping the probe's any-exception acceptance would certify missing dependencies, unrelated
  alignment failures and broken fixtures as valid refusals.
- Testing only binding presence or re-evaluating the result transform would miss stale fixed
  terms, swapped source axes and wrong identity. Source-label and structural checks are required.
- A field crop alone follows the affine `isel` path. N31/N32 instead distinguish source-name
  rewriting from dimension-only rewriting in the same non-affine binding.

### Addendum: N36-N39 (2026-10-25)

The extension survey found a `RasterIndex` split when a 1-D array with equal `x` labels comes
first; the same path splits the binding, which N01/N02 (unequal labels, so reindexed) and the
legacy reverse cases (plain operand covering both dimensions) never reached. Contract as for
N02 (design §3: framed plus plain is framed whatever the order):

| ID | Operation (`px_equal = px.assign_coords(x=[0, 3, 6, 9])`) | Expected |
|---|---|---|
| N36 | `px_equal + f` | **preserve** on the full grid |
| N37 | `xr.where(px_equal > 0, px_equal, f)` | **preserve** on the full grid |
| N38 | `xr.Dataset({"profile": px_equal, "owner": f})` | **preserve** `owner`; **unframe** `profile` |
| N39 | `xr.merge([px_equal.to_dataset(), f.to_dataset(name="owner")])` | as N38 |

## Deferred Work

Disposition of the three unfiled candidates in
[upstream PR notes, Deferred Work](../xarray-upstream/upstream_prs.md#deferred-work):

| Candidate | Reach and expectation |
|---|---|
| Shared custom index, equal incoming a and unequal unindexed b silently dropped by update | **N08** instantiates a=y and b=x, owning both in one binding. It must refuse the x conflict, not preserve an update that discarded it. N06 reaches the adjacent merge path. Patches 3/4 must check evidence before 7/7b's type-priority decision; a patch-7-only `AlignmentError` is not proof that the unindexed conflict hook was reached. |
| Transform-backed Dataset silently reindexes an unequal incoming coordinate away | **Exact candidate not established by this matrix.** The note supplies neither a reproducer nor the index class, coordinate role or alignment mode. `BindingIndex` delegates source labels to per-axis `PandasIndex` (I:82, 288); attaching a field transform does not make it xarray's `CoordinateTransformIndex`. N09 covers the indexed-label update path (a refusal) and N08 the protected-unindexed refusal. A separate transform-index reproducer is needed before claiming reach or choosing its expected exception; see Q1. |
| Broadcast excludes one dimension of a multi-dimension index and raises `CoordinateValidationError` | **N33** directly reaches the excluded-x/multi-coordinate-index path. Required result is the stated preserve/unframe pair, never that exception. #11615 enables ordinary broadcast; patch 4 must be inspected for exclusion handling. A failure on P would need a new fix, not an assumed existing patch owner. |

**Q2 — check at execution:** N04 expects `ValueError` from `join_overlapping`; if xarray's
exact-join check raises `AlignmentError` (a `ValueError` subclass) before the hook runs, record
the observed path and bring it back for a contract decision instead of loosening the class.

**Q1 — reviewer decision needed:** recover the second candidate's original construction and
determine whether its unequal coordinate is an alignment label or protected mapping input.
The present contract does not decide that underspecified report. Do not invent a refusal type
or claim N09 reproduces the bug. If it needs a generic xarray transform-index fixture, keep
that reproducer outside this native-binding matrix and decide its contract before executing it.

The actual lane heads, all measurements, any new generic xarray fixes and revision of the old
inventory are deferred to R1 execution. No upstream status was refreshed or external service
accessed for this note. No source, test, probe or other documentation changes are part of R1a.

## Next Steps

1. Review the 35 constructions, exact refusals, output-specific expectations and Q1 against
   the cited contracts. Resolve any changed expectation before running a case.
2. Implement the accepted rows in the probe and focused native tests; check fixture setup
   separately, then run the narrow relevant groups on identified stock, upstream and series-4
   lanes. Re-run the original 93 cases as required by R1.
3. Record each failure, owner and release impact in the inventory. Preserve strict xfails;
   do not turn unexpected exceptions or missing field/Dask coverage into passes. Close R1
   only after its execution and release criteria, not on acceptance of this design note.
