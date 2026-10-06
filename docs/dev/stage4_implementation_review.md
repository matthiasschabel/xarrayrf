# Stage 4 implementation and refinement record

**Status:** Implemented
**Last updated:** 2026-10-06
**Scope:** A1 declared geometry ordering, A2 explicit coordinate replacement, A5 transform value contract and F5 assignment

## Context

The user explicitly removed backward-compatibility and deprecation constraints for the current
development version. Stage 4 therefore changes contracts directly, without legacy aliases,
warning machinery or parallel behavior modes. The base is
`a721688ffe180f0d5d3ee5801326fca22405c1d6`. Opus plan/implementation reviews explicitly select
`claude-opus-5-5`, and gpt-6-astra implements in an isolated detached worktree. Codex GPT-6
orchestrates and independently verifies. The unrelated user documentation edits stay intact
and outside commits. No remote publication occurs.

Reviewer CLI controls remain read-only. Runner escalation allows macOS authentication/network
access, so review isolation rests on application controls. Codex implementation uses its
worktree workspace-write sandbox. The initial implementer timeout is deliberately 900 seconds
because changes cross several query, binding and adapter consumers; default resumability stays.

## Current Decision

Declared query ordering is committed as `aaa95d9` and explicit coordinate replacement as
`0f0e00a`. Shared value contracts, assignment tests, public docs and QA records accompany
them in a separate commit.

A1: Declared geometry dimension order is the logical sampling order. Geometry.dims, positional
query inputs/outputs, dense point axes, default lattice columns, frame-coordinate fields and
Grid snapshots agree. Pixel storage layout is separate: transposing a DataArray does not
silently change its binding's sampling order. Explicit lattice dimensions and xarray transpose
are sufficient for consumer-specific order. NGFF export explicitly extracts storage-order
columns so metadata still locates on-disk pixels correctly.

A2: The Grid form of rf.frame refuses conflicting source coordinates by default.
replace_coordinates=True explicitly adopts the Grid values/dimensions/dtype/unit declarations,
including when previous explicit units are malformed. Compatible coordinates retain metadata;
missing coordinates are supplied. Pixel values are shared, not resampled or unit-converted.

A5: Coordinates/intervals are copied and frozen, but transforms are retained by reference.
Extension transforms must keep endpoints, behavior and scalar equality stable. Grid hashing
is conditional on equality-consistent transform hashing. Unhashable transforms remain usable
for Grid queries/binding. Composite stability and hashability depend on members.

F5: Indexed assignment replaces destination-frame payload when xarray's dimension-label checks
allow it. It does not register/resample data or adopt the source frame. Conflicting dimension
labels may raise IndexError. No new assignment API or upstream guard is added.

### Plan finding dispositions

Opus recommended Accept with revisions and no blockers. Each local revision was incorporated
before implementation; no second plan pass was needed.

| ID | Disposition | Decision |
|---|---|---|
| N1 | Accepted | Check NGFF dimension coverage before explicit storage-order lattice extraction; remove unused array-order helper. |
| N2 | Accepted | Explicit replacement includes malformed units; default raises TypeError for malformed unit attrs. This follows the already-authorized use of the Grid declaration. |
| N3 | Accepted | Fix unconditional binding/hash and composite-immutability claims. |
| N4 | Accepted and verified | Independent distinct-frame array assignment rejects conflicting dimension labels, while matching labels allow replacement and retain destination frame. |
| N5 | Accepted | State ordering changes and direct transpose/lattice recipes in public docs/changelog; check affected tests semantically. |

### Implementation review dispositions

Both implementation passes recommended **Accept** with no blockers. gpt-6-astra handled the
two useful diagnostic/coverage followups; the final pass confirmed resolution and withdrew
the informational precedence observation. The final local test run satisfies its acceptance
condition.

| ID | Disposition | Decision |
|---|---|---|
| N1 | Accepted | Preserve exact unit diagnostics and short per-coordinate dimension/dtype-kind/value reasons, without printing arrays or adding an API. |
| N2 | Accepted | Add an indexed dimension-coordinate case: refusal preserves the exact original index/input and opt-in binds the Grid while sharing pixels. |
| N3 | Acknowledged; reviewer withdrew | Coverage preflight intentionally precedes lattice extraction; the public contract already states it. |


### Implementation and independent QA

The initial gpt-6-astra CLI used high reasoning effort. Targeted followups used the same
explicit model. Early test/README assertions and typing/lint errors were corrected before the
reported final runs; no failed check remains unresolved. Implementation changed 13 scoped
files. Independent root review checked source and public tests rather than accepting agent
agreement as proof.

| Check | Result |
|---|---|
| New public regressions against original source snapshot | 15 failures reproduce the old ordering/framing contracts; snapshot at `a721688` leaves the shared checkout untouched. |
| Initial root focused suite | 486 passed, 41 expected stock failures, two existing warnings. |
| gpt-6-astra followup Grid suite | 199 passed on patched xarray; diagnostic and indexed-coordinate cases included. |
| Independent root ordering probes | Eager/Dask pixel transpose preserves declared order across point queries, dense points, lattices, snapshots, native encode/decode and NGFF physical mapping. |
| Independent root replacement probes | Refusal preserves caller state; opt-in shares pixels, retains compatible attrs and replaces malformed units. |
| Custom-transform/assignment checks | Immutable custom value equality/hash works; public tests cover unhashable binding/query values. Distinct-frame assignment preserves destination binding when dimension labels agree and rejects mismatched labels. |
| Final root stock suite on main | 1,764 passed, 47 expected failures, three existing dependency warnings. |
| Final root patched suite on main | 1,811 passed, three existing dependency warnings. |
| Ruff lint/format, mypy and whitespace | Passed; mypy checks 81 source files. |

All 13 reviewed implementation files were integrated byte-for-byte. QA checked root causes,
public error paths, scalar/empty behavior in the full suite, Dask laziness and pixel identity,
unit/context metadata, constructor contracts and adapter consumers. There are no deprecated
aliases or backward-compatibility behavior branches. The three user-owned documentation file
hashes, main branch and original empty index were verified before integration and commit.
Maintainer records add only the audit plan, internal changelog and this refinement record.


### Subsequent QAEngineer review

The fresh user-requested QA pass reviewed stage 4 at `826fe76` against original source
`a721688`, preserving the checkout, index and user edits. It confirmed the root causes,
focused scope, public error paths and documentation contracts. No production correction
was required; one assignment documentation boundary was clarified and tested.

| Check | Result |
|---|---|
| Selected new tests against original source snapshot | 15 failed, confirming original ordering/framing behavior. |
| Initial focused Geometry/Grid/native/encoding/frame-coordinate/NGFF suite | 490 passed, 41 expected stock failures, two existing warnings. |
| Independent 3-D order and export probes | 72 passed: all six declaration orders × six storage orders × eager/Dask; analytic shear/origin checks across points, inverse positions, lattices, fields, native round trips and NGFF/NIfTI exports. |
| Independent boundary replacement probes | Eight passed: empty/scalar/eager-Dask combinations retain input indexes, shared pixels, context and intervals. |
| Additional semantic probes | NGFF time/heterogeneous spatial units remain correct with different declaration/storage order; frame-coordinate reverse nearest selection agrees. Unhashable custom and composite transforms remain queryable/bindable and explicitly refuse hashing. |
| Assignment clarification | New scalar-slice contract test and existing varying-dimension conflict test pass. |
| Final stock/patched full suites | 1,765 passed and 47 expected failures / 1,812 passed; three existing warnings each. |
| Ruff lint/format, mypy, links and whitespace | Passed; mypy checks 81 source files. |

Xarray's installed `assert_coordinate_consistent` checks coordinates only for dimensions
remaining on the indexed object. A scalar selection removes that dimension, so its retained
scalar coordinate is ignored even if the right operand's position differs. Assignment still
replaces destination-frame payload and keeps the destination binding. Public README/interface
and release notes now state this boundary explicitly; the new public test verifies both
changed pixel values and unchanged destination Grid. No assignment guard or compatibility
mechanism was introduced.

QA documentation and the public clarification/test are committed to main. The original
user-owned documentation files remain byte-for-byte intact and excluded.

## Alternatives Considered

Making current array order authoritative would require changing established Grid snapshots,
binding/encoding order and positional queries. Declared order already supplies one coherent
sampling value contract. New order flags or compatibility defaults would create parallel
interpretations. Arbitrary transform deep copies/freezers cannot guarantee extension-object
immutability. Runtime assignment guards would change a documented payload operation rather
than clarify it.

## Deferred Work

Stage 5 remains pending. Previously documented cubic NaN/Inf behavior and optional stage 3
per-task slice-routing performance work remain separate. No encoding schema bump follows
from these contracts.

## Next Steps

Stage 4 is complete, cross-reviewed, QA-verified and committed to local main in scoped units.
Stage 5 is the remaining queue. Commits retain `Co-Authored-By: Codex (gpt-6-astra)` and
`Reviewed-By: Claude (claude-opus-5-5)` attribution. No remote push is authorized or performed.
The task-owned worktree is removed after integration and final verification.

## Open Disagreements

None blocking. API decisions are made within the user's explicit clean-development scope;
compatibility scaffolding is intentionally excluded.

## Reviewer Reports

The following complete reviewer reports are retained as quotations.

### Plan review

> ## Artifact Summary
> The plan covers four changes. A1 makes the declaration order of geometry dimensions the single logical order. A2 makes coordinate replacement in the Grid form of `rf.frame` an explicit opt-in. A5 states honestly what Grid equality and hashing require of extension transforms. F5 documents what assignment does. I checked the source and the direction holds up. Using declaration order in A1 is the right call: `Grid` has no pixels, so declaration order is the only order it can carry (`_geometry.py:521` already transposes the snapshot to `self._dims`). I found no blockers. The gaps below are about what the implementer needs to be told.
>
> ## Blocking Findings
> None.
>
> ## Non-Blocking Findings
>
> ### N1: NGFF export must check dimension coverage before calling the lattice with explicit order
> - Confidence: High
> - Evidence: `ngff/_export.py:188-193` calls `geometry.lattice()` inside a `try` that turns any `ValueError` into "requires a regular affine lattice". It checks `len(lattice.dims) != len(geometry.array.dims)` only afterwards. Suppose the fix passes `geometry.array.dims` directly and the array has a non-geometry dimension. `lattice` would then raise a `dims` mismatch, and the user would see the misleading regular-lattice message. When `dims` is supplied, `_array_order` (`_geometry.py:422`) is not needed. The NGFF check already requires every array dimension to be a geometry dimension.
> - Suggested change: in NGFF, run the every-dimension check first, then call `geometry.lattice(tuple(map(str, geometry.array.dims)))`. Remove `_array_order` once `points`, `lattice` and `frame_coordinates` (lines 459, 574, 665) no longer use it. Update the docstring at line 558 ("Defaults to the array's own order") and the `points` docstring at line 428.
> - Would change my mind: an existing test showing that an array with an extra non-geometry dimension already gets the clearer error.
>
> ### N2: A2 must count the unit attribute in "compatible", and decide what happens to malformed attributes when the caller opts in
> - Confidence: High
> - Evidence: today a coordinate with matching values keeps `current`, attributes included (`native.py:337-342`). Validation then fails inside `Geometry(...)` through `check_coordinate_unit` (`_geometry.py:110-135`). If the plan's predicate compares only dimensions, values and dtype kind, `replace_coordinates=True` would still raise on a coordinate whose values match but whose unit conflicts. Today any mismatch in values, dimensions or dtype kind is replaced silently (`replaced` at lines 333-354). So the refusal is the real behaviour change, and its tests should cover the old silent path.
>
>   The plan says malformed unit types "retain a clear TypeError". It doesn't say whether this still applies when `replace_coordinates=True`. Opting in naturally means "use the Grid's declaration", which would replace the malformed attribute rather than raise.
> - Suggested change: define compatible as same dimensions, same dtype kind, `array_equal` values, and an absent or equal `units` string. With opt-in, anything else is replaced, malformed units included. **Decision point for the user:** confirm that malformed units are replaced under opt-in.
> - Would change my mind: a stated reason why malformed attributes should block even an explicit opt-in.
>
> ### N3: A5 conflicts with a documented binding requirement
> - Confidence: High
> - Evidence: `docs/core_interface.md:279` says "A transform used in a binding must implement exact structural `__eq__` and `__hash__`". At runtime the binding only compares (`_binding.py:168`), and the source never calls `hash()` on a transform outside `__hash__` methods. Lines 249 and 337 (immutable, hashable `Grid`) are also in scope. `CompositeTransform.__hash__` (`_composite.py:210`) hashes its members, so a composite that wraps an unhashable extension transform is unhashable too. That means "built-in composite declarations are immutable" only holds when the members are.
> - Suggested change: rewrite line 279 to match the A5 contract. Add one sentence saying that composite immutability and hashability depend on the members.
> - Would change my mind: a code path that hashes a binding's transform (`grep -rn "hash(" src/xarrayrf`).
>
> ### N4: F5 "replacement" may raise in xarray when dimension coordinates conflict
> - Confidence: Medium (speculation, not run)
> - Evidence: I expect xarray's `DataArray.__setitem__` to call `assert_coordinate_consistent`. That should raise `IndexError` when the right operand's dimension coordinate conflicts with the indexed destination's. Two frames with different coordinate values on dimension-named axes would then fail rather than replace.
> - Suggested change: document that values are replaced only when shared dimension coordinates agree, and that xarray raises otherwise. Build the example with matching coordinate values and different frames. The binding itself is not a dimension coordinate.
> - Would change my mind: the command below completing silently.
>
> ### N5: The positional break for dense consumers belongs in the changelog
> - Confidence: High
> - Evidence: after A1, `geometry.points().values` on a transposed array no longer lines up positionally with `array.values`. Alignment by name still works. Default-`lattice()` tests in `tests/test_geometry.py`, `test_lattice.py`, `test_frame_coordinates.py`, `test_synthetic_workflows.py` and `test_ngff_export.py` need their expectations checked, not just reordered.
> - Suggested change: add a CHANGELOG line covering `points`, `lattice` and `frame_coordinates`, with "use `.transpose(*array.dims)`" as the recipe.
> - Would change my mind: none needed; this is a documentation completeness point.
>
> On the plan's own checks, the A1 four-way agreement test (`points_at`, `point_at`, `points().isel`, `lattice().transform_point`) is the right regression and is enough. The A2 test list is proportionate.
>
> ## Verification Commands
> - N1: `rg -n "lattice\(" src/xarrayrf` (after implementation, confirm the only no-argument caller is gone), then run an NGFF export on an array with an extra non-geometry dimension and read the error message.
> - N3: `rg -n "hash\(" src/xarrayrf/*.py | grep -v "def __hash__"`
> - N4: `python -c "import numpy as np, xarray as xr; a=xr.DataArray(np.zeros(3),dims='x',coords={'x':[0,1,2]}); b=xr.DataArray(np.ones(2),dims='x',coords={'x':[5,6]}); a[0:2]=b; print(a.values)"`. Repeat with `a.rf.frame(...)` and `b.rf.frame(...)` bound to different frames.
>
> ## Recommendation
> Accept with the revisions above. N2's handling of malformed attributes under opt-in needs a user decision, and N4 needs the command run before the F5 example is written.

### Initial implementation review

> ## Artifact Summary
>
> I recommend accepting this change. It has no blocking findings and three optional ones. I checked it against the source in the worktree without changing anything.
>
> - **A1 (declared order):** `Geometry.points`, the default `lattice()` and `frame_coordinates` now use `self.dims`. `_array_order` is gone and nothing else used it.
> - **Lattice consumers:** I checked every caller of `.lattice(`.
>   - NIfTI passes explicit `dims` (`nifti/__init__.py:371`).
>   - Resampling passes `source.dims` (`_resample.py:183`).
>   - NGFF now checks first that every array dimension is a geometry dimension, then asks for the lattice in storage order (`ngff/_export.py:188-191`). This is correct because the NGFF scale/affine columns describe on-disk axes.
>   - GeoTIFF, DICOM and NGFF import go through `frame_array`, which pairs `dims` with the pixels explicitly, so they are unaffected.
> - **A2 (coordinate replacement):** the logic in `native.py:349-377` is sound.
>   - The flag is checked to be a real bool before any other check.
>   - Passing `True` with a transform is refused.
>   - A malformed unit with the default raises `TypeError` that keeps the original cause and names the opt-in.
>   - Unit conflicts mark the coordinate as conflicting, and the refusal happens before `assign_coords`, so the caller's array is untouched.
>   - Compatible coordinates keep the caller's existing variable, attributes included.
>   - Pixels are shared; the Dask test confirms the same graph and that nothing was computed.
> - **A5 (extension transforms):** confirmed that the binding never hashes the transform. Only `Grid.__hash__` (`_grid.py:382`) and `CompositeTransform.__hash__` (`_composite.py:208`) do, so "hashing is optional for binding" is true. The custom-transform test covers both the hashable and unhashable cases.
> - **F5 (assignment):** documentation and one test only, no runtime change, as planned.
>
> ## Blocking Findings
> None.
>
> ## Non-Blocking Findings
>
> ### N1: The default unit-conflict refusal no longer says what the units disagree about
> - Confidence: High
> - Evidence: `native.py:363-364` catches the specific `ValueError` from `check_coordinate_unit` (`_geometry.py:131-136`, e.g. `'m'` vs `'mm'`) and throws it away. The caller now only sees "source coordinate(s) ['offset'] conflict with the Grid declarations". Before this change, the caller got the exact unit mismatch (the old test matched `"declares"`). The same loss applies to value, dtype-kind and dimension conflicts: the message never says which of these failed.
> - Suggested change: record a short reason per conflicting coordinate (unit message, "dims", "dtype kind", "values") and put it in the refusal message. This is a few lines and needs no new API.
> - Would change my mind: a deliberate decision to keep refusal messages generic, for example to avoid printing large values.
>
> ### N2: A conflicting coordinate that carries an index is not tested on the default-refusal path
> - Confidence: Medium (the gap is in coverage; I did not find a bug)
> - Evidence: the new refusal tests use `offset`, and `u` along `time`, neither of which has an index. Index preservation on refusal is asserted only in `test_grid_frame_replaces_scalar_and_dimension_conflicts_without_computing_pixels`, where `u` has no index. The opt-in path calls `drop_indexes` and then `drop_vars(replaced)` (`native.py:384-385`). The case that matters is a conflicting indexed coordinate, such as a dimension coordinate `i` whose values differ from the Grid's.
> - Suggested change: add one parametrize case where the conflicting coordinate is a dimension coordinate with an index. Assert that a default refusal leaves `xindexes` unchanged and that `replace_coordinates=True` produces a binding index.
> - Would change my mind: an existing test that already covers this; I did not search the whole suite.
>
> ### N3: The NGFF preflight changes which error wins when both apply
> - Confidence: Medium
> - Evidence: `ngff/_export.py:188` now rejects missing geometry dimensions before the lattice is built. Previously, a non-affine or irregular geometry that also had a non-geometry dimension produced the "regular affine lattice" error first; now the dimension-coverage error comes first. This order is better. I have not checked whether any test pins the old order, though the full suite passing suggests none does.
> - Suggested change: none required. Optionally add a one-line CHANGELOG note if error precedence counts as public behaviour.
> - Would change my mind: nothing; I raise this only for completeness.
>
> The rest checks out:
> - The README/core_interface recipe for pixel order uses the public `Geometry.array` (`_geometry.py:317`).
> - The test changes are meaningful rather than just reordered expectations:
>   - In `test_frame_coordinates.py`, a pixel transpose now gives identical indexes, while a `Geometry` with reordered `dims` still differs.
>   - The four-way consistency test covers eager and Dask arrays, with and without a retained scalar.
>
> ## Verification Commands
> - N1: `python -c "import numpy as np, xarray as xr; from tests.test_grid import transform; import xarrayrf as xrf; g=xrf.Grid(transform(),{'offset':('i',[0,2])}); a=xr.DataArray(np.zeros(2),dims='i',coords=dict(g.coordinates)); a.offset.attrs['units']='m'; a.rf.frame(g)"`. The message should include the declared/actual units; currently it does not.
> - N2: `pytest tests/test_grid.py -k "replace or conflict" -q`, after adding the indexed-coordinate case.
>
> ## Recommendation
> Accept. N1 is the one change I would make before committing, because it brings back unit-mismatch detail that callers had before this change.

### Implementation convergence

> ## Convergence Assessment
> | ID | Status | Rationale |
> |---|---|---|
> | N1 | Resolved | `native.py:353-363` now keeps the exact text of the `check_coordinate_unit` error. For a mismatch, the `ValueError` from `_geometry.py:126-136` becomes the reason for that coordinate. A malformed unit with the default flag raises `TypeError` chained `from error`; with `replace_coordinates=True` it becomes the reason. When the unit check passes, the code adds short reasons for dims, dtype kind and values (`native.py:364-372`), and no array contents are printed. `test_grid.py` checks the unit text (`'m'.*'mm'`) and the TypeError cause, and runs one parametrized case for each of the three non-unit reasons. |
> | N2 | Resolved | `test_grid_frame_replaces_conflicting_indexed_dimension_coordinate` adds an indexed dimension coordinate `i` whose values differ from the Grid's. On default refusal, the caller's array is unchanged and its index is the same object. With opt-in, `result.rf.grid == grid` (which needs the binding index), the index is new, pixels are shared, and the caller's original is still untouched afterwards. |
> | N3 | Withdrawn | It was raised for completeness and asked for no change. The designer says the coverage check intentionally runs before lattice extraction, and the docs say so. I accept that. |
>
> ## New Findings
> None. Points I checked:
> - `units[name]` (`native.py:356`) cannot raise `KeyError`. `grid_coordinates` emits only variables keyed by source axes, and the binding has no declaration coordinate.
> - The conflict loop is pure, and the refusal fires before `assign_coords`, so the "caller untouched" guarantee still holds.
> - One nit, not a defect: when the unit check fails it short-circuits the dims, dtype and values checks, so a coordinate that conflicts in several ways reports only the unit reason. That is enough to tell the caller what to fix.
>
> I did not run the suite. To confirm: `pytest tests/test_grid.py -k "replace or conflict or extension" -q && pytest -q`.
>
> ## Open Disagreements
> | ID | Designer / Implementor Position | Critic / Reviewer Position | Human Decision Needed? |
> |---|---|---|---|
> | — | — | — | No |
>
> ## Recommendation
> Accept, as long as the local test run passes.
