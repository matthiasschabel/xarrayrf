# Core model: frames, coordinate systems and transforms

**Status:** Implemented
**Last updated:** 2026-10-04
**Scope:** The NumPy-only core: `ReferenceFrame`, `CoordinateSystem`, roles, direction
vocabularies, the `Transform` protocols and endpoints, quantity rules, units, affine classes and
`Lattice.affine`. [The core interface](../../core_interface.md) is normative; this note records
why it looks the way it does. Quantity rules are specified but not implemented (see Deferred Work).

## Context

The first prototype had `ReferenceSpace` (identity plus a free `convention` string) and an affine
class named `ReferenceFrame`. Checked against DICOM PS3.3, OME-NGFF RFC-4, ITK, VTK and Astropy,
that surface had four defects:

1. **Names inverted domain usage.** DICOM's Frame of Reference, Astropy frames and robotics
   frames name the *space*; the prototype gave that name to the *mapping*.
2. **`convention` conflated three facts:** axes defined by a subject, a subject's orientation
   within an independent frame, and derived direction labels. They have different owners.
3. **LPS and RAS of one frame were reported as a contradiction.**
4. **Mappings were affine-only by construction**, with no seam for sampled fields, provider
   transforms, user transforms or frame-to-frame transforms.

Established vocabulary separates these ideas. Geodesy and Astropy distinguish a *reference frame*
(identity, anchoring) from a *coordinate system* (axes, order, direction, units). Graphics and
robotics distinguish object frames from world frames, related by a pose. ITK and VTK distinguish
transforming points, vectors and covariant vectors. DICOM has the same structure: the patient
coordinate system (PS3.3 C.7.6.2.1.1), a separately identified equipment frame (Equipment Frame
of Reference UID (300A,0675), macro 10.39) related by the rigid Image to Equipment Mapping Matrix
(0028,9520), and registrations between frames. Each is a transform between frames, not a
property of one frame.

## Current Decision

### Vocabulary

| Term | Meaning | Precedent |
|---|---|---|
| Reference frame | Identity and anchoring | DICOM Frame of Reference, Astropy frame, geodetic datum |
| Coordinate system | Axis names, order, units, representation, orientation | ITK/VTK, NGFF `coordinateSystems` |
| Role | Descriptive `"world"` or `"object"` | Graphics world vs model space; robotics world vs body |
| Transform | Mapping from a source endpoint to a target endpoint | ITK `Transform`, VTK `vtkAbstractTransform` |
| Point / vector / covariant vector | Transformed with translation / by J / by J^-T | ITK `TransformPoint`/`TransformVector`/`TransformCovariantVector` |
| Origin / spacing / direction | Regular-lattice geometry | ITK image geometry, VTK `vtkImageData` |

"World" is preferred to ITK's "physical" and is used only for the role. `CoordinateTransform`
is avoided because `xarray.indexes.CoordinateTransform` names array position to label; "Mapping"
is avoided because Astropy's `Mapping` means axis reordering.

### Value objects and comparisons

`CoordinateSystem` is immutable: axis names, one unit per axis (`None` means undeclared, not a
non-quantity), optional open axis-type strings, representation (Cartesian only), and optional
orientation in a vocabulary. It carries no identity.

`ReferenceFrame` is immutable: identifier `(namespace, value)`, `coordinate_system`, optional
role, and `definition`, `context` and `display` metadata. `local()` mints an identity;
`declared()` adopts an explicit one. Neither resolves a name. A frame value is a frame *expressed
in* one coordinate system.

| Question | Call | Compares |
|---|---|---|
| Same frame? | `a.is_equivalent_frame(b)` | Identifier, definition, context (Astropy semantics) |
| Contradictory? | `a.conflicts_with(b)` | Same identifier, different definition or context |
| Same axes? | `a.coordinate_system == b.coordinate_system` | Names, units, types, representation, orientation |
| Identical? | `a == b` | All of the above; never role or display |

`==` is strict because composition and binding need exact axis order and sign; a loose `==`
would let mismatched axes compose silently. Metadata comparison is type-tagged, so `True`, `1`
and `1.0` differ.

Stage-1 value-object rules still hold: coefficients are copied into immutable buffers and
returned as snapshots; numeric input must have a real integer or floating dtype (numeric strings,
complex and boolean arrays are refused; a mixed Python sequence follows NumPy dtype inference);
masked arrays are refused because conversion would drop the mask. Private or dunder mutation is
unsupported rather than defended against.

### Coordinate-system change

`coordinate_system_change(a, b)` returns the exact transform between two coordinate systems of
one frame, or raises `ValueError`. It is derivable only if the frames are equivalent, the
representations are equal, both use the same vocabulary with identical sets of antipodal pairs,
oriented axes match by token pair (never by name) with equal units, and unoriented axes appear in
both with identical name, unit and position. The result is a signed permutation that may rename
axes (`x,y,z` to `L,P,S`). It never rescales; a unit change is an explicit scaling transform.
Differing coordinate systems are not a contradiction. `resample` applies this change
automatically between equivalent frames.

### Roles

Role is descriptive metadata like `display`: excluded from equality, hashing and contradiction,
never inferred, and no generic operation branches on it. "World" is relative to an analysis: a
donor body is an object in the scanner and a world for a biopsy. Excluding role from equality is
what lets `CompositeTransform(biopsy_to_donor, donor_to_scanner)` compose when the two donor
declarations carry different roles. Atlas spaces (MNI, Allen CCF) are object frames of a
canonical subject, related to a subject only through registration. View, camera and canvas
frames are ordinary frames defined by a viewer; the core has no display concept.

Index space is not a frame. Array positions and labels are xarray's; a transform from
`ArrayCoordinates` maps coordinate values into a frame.

Several subjects in one array (specimens on a slide, a biopsy in its donor) are several object
frames with their own transforms. The one rule that keeps this cheap later: orientation and pose
attach to frames and transforms, never to an array or binding as a "subject orientation".

### Orientation vocabularies

Two layers with different owners:

- **Geometric (core).** A coordinate system may give each axis a direction token from a
  `DirectionVocabulary`: an identifier plus antipodal token pairs. The core validates distinct
  pairs and derives signed permutations; it interprets no token. LPS to RAS is diag(-1, -1, 1)
  purely from the declared pairs. Tokens use RFC-4's explicit from-to form (`left-to-right`).
  Bare codes such as `RAI` are refused: ITK's legacy codes name the direction an axis points
  *from*, the opposite of the DICOM/NIfTI reading, so an unqualified code is ambiguous. Adapters
  normalize. `CoordinateSystem.axis_codes(vector)` reports the nearest signed oriented axes with
  angular residuals, measured within oriented axes only; unlike nibabel's `aff2axcodes` it keeps
  the residual visible. Handedness is not modelled.
- **Naming (adapters).** Names for people may depend on body region. DICOM quadrupeds
  (PS3.3 C.7.6.2.1.1) name +y dorsal or cranial and +z cranial, rostral or proximal by region,
  while the tokens stay constant. The canonical anatomical vocabulary (RFC-4 pairs) and regional
  naming live in `xarrayrf.anatomy`; the core ships no vocabulary.

A DICOM adapter maps the Frame of Reference UID to the patient object frame (LPS). Patient
Position is reported on the adapter's result (`DicomGeometry.patient_position`), not recorded as
frame context: it describes how the patient lay, not which space the coordinates refer to, and
series sharing a Frame of Reference UID share one frame by the standard's definition. Relating
series from different frames uses an explicit transform, never an automatic one.

### Transforms and endpoints

- Every transform has required `source` and `target` endpoints, each a `ReferenceFrame` or
  `ArrayCoordinates` (axis names, units and sample offsets, no identity). Input axes and units are
  the source's, so there is no separate `inputs` field to disagree with it.
- Capabilities are structural protocols: `SupportsPoints` (required by `check_transform`),
  `SupportsJacobian`, `SupportsAffine` (constant `matrix` and `translation`), `SupportsInverse`
  (exact only). A capability an instance cannot honour raises; it never approximates.
- `ArrayCoordinates` may only begin or end a chain: it has no identity, so composing through it
  would connect unrelated arrays whose coordinates happen to compare equal.
- `CompositeTransform` applies first to last (Astropy `|`, NGFF `sequence`), the opposite of
  ITK and VTK PreMultiply. Its Jacobian follows the chain rule; its inverse reverses the members'.
  `compose` collapses an all-affine chain into one `AffineTransform`.
- Points are positional arrays in source-axis order; names belong to `transform_named` and the
  xarray layer.
- User transforms need exact structural `__eq__`/`__hash__` to be bound, and persist only through
  the data-only encoding ([persistence](persistence_design.md)). There is no registry.
- Transforms are never applied during arithmetic; resampling is a separate explicit operation.
- `AffineTransform.inverse()` requires a square matrix whose 2-norm condition number, after row
  and column equilibration, is within `INVERSE_CONDITION_LIMIT` (1/(10·eps), about 4.5e14).
  Equilibration makes the test unit-independent: an SI Lorentz boost has entries spanning about
  17 orders of magnitude and a raw condition number near 5e16, yet is well conditioned.
- Orientation-only poses were removed: a transform with unknown translation cannot map points.

### Affine construction

The default `AffineTransform` constructor takes `basis_vectors` keyed by source axis and a
required `target_axes` assertion for vector and translation component order. The assertion
must match the target endpoint exactly; dictionary order is ignored and matrix columns are
assembled in source-axis order. Vectors describe unit coordinate-value increments, include
scale, and may be nonorthogonal or dependent. They are properties of the transform, so two
arrays with different sampling can still share one reference frame and coordinate system.

`AffineTransform.from_matrix` remains public for adapters, codecs and numerical operations
that already have coefficients. Both paths share validation and immutable matrix/translation
storage, preserving structural identity and schema 1. Per-vector validation happens before
assembly so dtype promotion between vectors cannot hide a boolean or masked vector. The
pre-release API changes directly, without a deprecation shim, by explicit project decision.

An ordered vector list would retain source-order ambiguity; nested component mappings were
rejected as unnecessary complexity. Explicit target-axis assertion catches a mistaken order,
but cannot verify that the caller assigned the right numbers to those declared components.

### Quantity rules

Specified in the core interface as plain callables `rule(transform, values, *, at)` over a
trailing component axis in source-axis order, acting independently over leading axes so they
apply chunk by chunk:

| Rule | Transformation | Needs |
|---|---|---|
| `point` | `transform_point` | points |
| `vector` | J v | Jacobian |
| `covariant_vector` | J^-T n (unnormalized, ITK semantics) | square invertible J |
| `second_rank_tensor` | J T J^T (contravariant) | Jacobian |
| `polar_rotation` | R T R^T, R from J's polar decomposition | Jacobian |

ITK differs by design and must not be mirrored: `TransformSymmetricSecondRankTensor` computes
J T J^-1, and `TransformDiffusionTensor3D` applies preservation of principal direction (PPD) with
the inverse Jacobian because ITK maps fixed to moving. Domain rules such as PPD live downstream.
A rule needing a Jacobian refuses a transform without one; finite differences only on explicit
request.

A future *component binding* will declare which array dimension indexes which frame's axes and
which rule its values follow (NIfTI diffusion tensors in voxel axes, DICOM gradient directions in
patient axes). Components in the array's own axes reference the binding transform, so index
space still needs no frame. The binding carrier must reserve room for zero or more of these per
variable.

### Units

Open CF/UDUNITS strings, compared exactly and case-sensitively, never converted implicitly. A
unit is a non-empty string without surrounding whitespace; nothing else is parsed. Different
spellings (`mm`, `millimeter`) compare unequal, the conservative failure; adapters normalize with
`xarrayrf.units.canonical`. Cartesian axes refuse known angular spellings, including CF
`degrees_east` and `degrees_north`; this check is best-effort. A closed vocabulary was dropped
because it could not follow CF and NGFF spellings and never prevented mislabelling (nothing stops
`"mm"` on values in metres). Pint objects are never stored: registries do not compare across
instances and strings persist. Conversion is the caller's explicit scaling.

### Package layering

Frames, coordinate systems, transforms, vocabularies and `transform_named` form a NumPy-only core,
useful for meshes, point sets and tables as well as arrays. `Geometry`, `resample`, bindings and
the accessor form the xarray layer. The core never imports xarray, enforced by an AST scan in
`tests/test_import_boundary.py`; without xarray, integration names raise `ImportError` from the
package `__getattr__`. xarray remains a required dependency for now. The goal is coverage: represent
and convert what existing libraries model ([prior-art study](prior_art_coverage_study.md)) without
depending on them; converters live in adapters.

The core's value objects are independent of how a binding survives xarray operations. That
lifecycle is governed by [the architecture](../../design.md) §5: the target is lifecycle
ownership over a named minimum native subset, and every operation must enforce, reject, or leave a
detectably invalid declaration. A scalar-index carrier alone was measured and rejected: a
fixed-plane sum silently lost a bound coordinate while keeping the declaration, and a guarded
declaration silently won over a conflicting unguarded one. Binding work is recorded under `docs/dev/binding/`.

### Affine classes

An affine map `x -> A x + t` from D to M dimensions is one `(M+1)x(D+1)` homogeneous matrix with
last row `[0 ... 0 1]`. "Homogeneous" names the form, not a class of motion; DICOM's Frame of
Reference Transformation Matrix Type distinguishes `RIGID`, `RIGID_SCALE` and `AFFINE`, all 4x4
with that last row (PS3.3 C.20.2.1.1).

| Class (`affine_class` name) | Constraint | DOF 2-D / 3-D | Common names |
|---|---|---|---|
| `identity` | A = I, t = 0 | 0 / 0 | |
| `translation` | A = I | 2 / 3 | ITK `TranslationTransform`, NGFF `translation` |
| `rotation` | A orthogonal, t = 0 | 1 / 3 | SO(n); direction cosines |
| `rigid` | A orthogonal | 3 / 6 | SE(n), pose; DICOM `RIGID`; FLIRT 6-DOF |
| `similarity` | A = sR | 4 / 7 | ITK `Similarity3DTransform`; FLIRT 7-DOF |
| `scaled_rigid` | A = R diag(s) | 5 / 9 | DICOM `RIGID_SCALE`; NIfTI qform; ITK `direction · diag(spacing)` |
| `affine` | A invertible | 6 / 12 | DICOM `AFFINE`; NIfTI sform; ITK `AffineTransform` |
| `singular`, `rectangular` | any A | | embeddings (a 2-D slice in 3-D), projections |
| projective (not modelled) | free last row | 8 / 15 | homography, camera matrix |

`affine_class(affine, *, tolerance=1e-6, offset_tolerance=1e-9)` accepts any `SupportsAffine` or
a `Lattice`. Reflection is the `proper` flag, not a class. The containment order is `identity`
in both `translation` and `rotation`, which meet at `rigid`, then `rigid ⊂ similarity ⊂
scaled_rigid ⊂ affine`; `at_most` ignores `proper`, so DICOM `RIGID` also tests it. With `N` the
column-normalized matrix, the decision runs broadest first: `M != D` is `rectangular`; a zero
column or `rank(N) < D` is `singular` (rank on normalized columns is scale-invariant, so
`diag(1e-16, 1, 1)` stays `scaled_rigid`); non-orthogonal `N` is `affine`; unequal column norms
is `scaled_rigid`; failing `|A^T A - I|` or `||det A| - 1|` is `similarity` (the DICOM `RIGID`
test, so `sqrt(1 + 0.8 tol) I` is never `identity`); otherwise identity, translation, rotation
or rigid by `A ≈ I` and `|t|`. Each class satisfies all its ancestors' constraints, so `at_most`
is trustworthy; the cost is that DICOM `RIGID` became stricter within tolerance² of its boundary.
`residual` reports how close the call was; `same_units=False` warns that a lattice's "rigid"
describes the matrix, not a physical motion (indices are unitless). The NIfTI qform check and
DICOM `RIGID`/`RIGID_SCALE` checks are built on it.

The query classifies the resulting matrix; it cannot tell how the map was estimated. A Lorentz
boost is `affine`: it preserves the Minkowski interval, not Euclidean distance.

### Lattice naming

`Lattice.affine` is the `(M+1)x(D+1)` index-to-frame matrix (4x4 for a volume, rectangular for a
plane in a volume), named after nibabel `img.affine`, napari `affine=` and DICOM `AFFINE`. It is
not rigid in general, so `rigid_matrix` or `pose_matrix` would be wrong. `matrix` is the linear
part; `origin` the frame point of position zero (ITK `Origin`); `spacing` the column norms
(ITK spacing only when unsheared); `direction` the unit-norm columns, which are not orthogonal
for a sheared lattice (ITK documents an orthonormal direction but does not enforce it).
`spacing` and `direction` are refused unless all frame axes share one unit; `matrix` remains.
A voxel-to-world matrix does not scale the object: `A = R diag(spacing)` puts the grid's
placement in R and t and the meaning of one index step in the spacing.

## Alternatives Considered

| Alternative | Why not |
|---|---|
| Free `convention` string | Cannot tell convertible from contradictory |
| Class hierarchy (spatial, anatomical, DICOM frame) | Puts domain policy (determinant sign, regular slices) in core types |
| Subject orientation on the array or binding | Blocks multiple subjects and nested objects |
| Role in equality, or behaviour by role | Breaks nested composition; makes roles a policy switch |
| Core-interpreted anatomical tokens | Needs domain knowledge; fails for regional quadruped naming |
| Transform registry or plugin discovery | Structural protocols suffice |
| One fixed tensor rule, or diffusion PPD in core | Tensor reorientation has competing strategies; PPD is domain policy |
| ITK, Astropy modeling or GWCS as the engine | See the prior-art study; used as oracles or optional adapters |
| Loose `==` ignoring the coordinate system | Mismatched axes would compose silently |
| Closed unit vocabulary | Cannot follow CF/NGFF and never prevented mislabelling |
| Implicit composition or a frame graph | Hides direction and completeness errors |
| Transform subclasses per group (`RigidTransform`) | No consumer needs type enforcement; `affine_class` answers the question |
| Refusing sheared lattices for `direction` | Would remove supported behaviour; documented instead |

## Deferred Work

- Quantity-rule callables (`vector`, `covariant_vector`, `second_rank_tensor`,
  `polar_rotation`): no implementation exists in `src/xarrayrf/` as of this date; callers apply
  `jacobian()` themselves (as the relativity example does). The acceptance oracles are in the
  [prior-art study](prior_art_coverage_study.md).
- Component bindings: carrier room reserved, behaviour unspecified.
- Projective and field-backed transforms, including DICOM Deformable Spatial Registration.
- Finite-difference Jacobians behind an explicit request.
- Approximate and declared inverses, under a separate name.
- Pint-backed conversion helper.
- A single-label axis-code query, when a consumer (reorientation, display labels, export) sets
  its shape. Known constraints: a required `max_angle` strictly below π/4, which keeps labels
  unique for orthogonal columns, plus a collision check for sheared lattices such as
  gantry-tilted CT.
- Classifying non-affine transforms by local Jacobian; export helpers choosing the tightest
  DICOM matrix type.

## Next Steps

1. Implement the quantity rules when a consumer needs them, with the prior-art oracles as tests
   and the rigid case (`second_rank_tensor` equals `polar_rotation`) as an acceptance check.
2. Keep this note and the core interface in step when either changes.
