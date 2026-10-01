"""Public behavior of :class:`xarrayrf.ReferenceFrame`."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from typing import Any, cast

import pytest

from xarrayrf import LOCAL_NAMESPACE, CoordinateSystem, ReferenceFrame, Role

UID = "1.2.826.0.1.3680043.8.498.10101010101010101010101010101010"
CONTEXT: Mapping[str, str] = {"realization": "a"}
DEFINITION: Mapping[str, str] = {"schema": "v1"}
PATIENT_AXES = CoordinateSystem(("L", "P", "S"), ("mm", "mm", "mm"))
PLANE = CoordinateSystem(("x", "y"), ("mm", "mm"))


def patient_frame(
    *,
    identifier: tuple[str, str] = ("dicom-frame-of-reference", UID),
    coordinate_system: CoordinateSystem = PATIENT_AXES,
    role: Role | None = None,
    definition: Mapping[str, str] | None = DEFINITION,
    context: Mapping[str, str] | None = CONTEXT,
    display: Mapping[str, str] | None = None,
) -> ReferenceFrame:
    """Build a DICOM-shaped declared frame with selected fields overridden."""
    return ReferenceFrame.declared(
        identifier,
        coordinate_system,
        role=role,
        definition=definition,
        context=context,
        display=display,
    )


def test_local_mints_a_distinct_identity_each_call() -> None:
    first = ReferenceFrame.local(PLANE)
    second = ReferenceFrame.local(PLANE)
    assert first.identifier[0] == LOCAL_NAMESPACE
    assert first.identifier != second.identifier
    assert first != second
    assert not first.is_equivalent_frame(second)
    assert not first.conflicts_with(second)


def test_sharing_one_local_frame_is_how_arrays_declare_one_frame() -> None:
    shared = ReferenceFrame.local(PLANE)
    alias = shared
    independent = ReferenceFrame.local(PLANE)
    assert alias == shared
    assert hash(alias) == hash(shared)
    assert len({shared, alias, independent}) == 2


def test_declared_identity_is_reconstructible_without_a_domain_package() -> None:
    first = patient_frame()
    second = patient_frame()
    assert first == second
    assert hash(first) == hash(second)
    assert first.identifier == ("dicom-frame-of-reference", UID)
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from xarrayrf import CoordinateSystem, ReferenceFrame; "
            "ReferenceFrame.declared(('dicom', '1.2.3'), CoordinateSystem(('x',), ('mm',))); "
            "assert 'pydicom' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr


def test_declared_can_readopt_a_persisted_local_identity() -> None:
    """Restoring a stored frame is adoption, not minting.

    A persisted local identifier has to come back as the same frame, so the local
    namespace is not fenced off from ``declared()``.
    """
    minted = ReferenceFrame.local(PLANE)
    restored = ReferenceFrame.declared(minted.identifier, PLANE)
    assert restored == minted
    assert hash(restored) == hash(minted)
    assert not restored.conflicts_with(minted)


def test_local_never_reuses_an_identity_it_was_given_elsewhere() -> None:
    adopted = ReferenceFrame.declared((LOCAL_NAMESPACE, "persisted"), PLANE)
    minted = ReferenceFrame.local(PLANE)
    assert minted.identifier[0] == LOCAL_NAMESPACE
    assert minted.identifier != adopted.identifier
    assert minted != adopted


@pytest.mark.parametrize(
    "other",
    [
        pytest.param(patient_frame(context={"realization": "b"}), id="context"),
        pytest.param(patient_frame(definition={"schema": "v2"}), id="definition"),
        pytest.param(patient_frame(context=None), id="missing-context"),
    ],
)
def test_same_identifier_with_different_definition_or_context_conflicts(
    other: ReferenceFrame,
) -> None:
    declared = patient_frame()
    assert declared != other
    assert not declared.is_equivalent_frame(other)
    assert declared.conflicts_with(other)
    assert other.conflicts_with(declared)


def test_one_frame_in_another_coordinate_system_is_equivalent_not_equal() -> None:
    """The Astropy question and the strict question have separate answers."""
    declared = patient_frame()
    other = declared.with_coordinate_system(CoordinateSystem(("x", "y", "z"), ("m", "m", "m")))
    assert declared.is_equivalent_frame(other)
    assert not declared.conflicts_with(other)
    assert declared.coordinate_system != other.coordinate_system
    assert declared != other
    assert len({declared, other}) == 2


def test_with_coordinate_system_keeps_everything_else() -> None:
    declared = patient_frame(role="object", display={"label": "supine"})
    other = declared.with_coordinate_system(PLANE)
    assert other.identifier == declared.identifier
    assert other.role == "object"
    assert dict(other.definition) == dict(declared.definition)
    assert dict(other.context) == dict(declared.context)
    assert other.display["label"] == "supine"
    assert other.coordinate_system is PLANE


def test_different_identifiers_with_identical_declarations_are_unrelated() -> None:
    declared = patient_frame()
    other = patient_frame(identifier=("dicom-frame-of-reference", UID + ".2"))
    assert declared != other
    assert not declared.is_equivalent_frame(other)
    assert not declared.conflicts_with(other)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(None, "world", id="unset-world"),
        pytest.param("world", "object", id="world-object"),
    ],
)
def test_role_is_excluded_from_every_comparison(first: Role | None, second: Role | None) -> None:
    """A donor is an object in the scanner and a world for its biopsy: one frame."""
    declared = patient_frame(role=first)
    other = patient_frame(role=second)
    assert declared == other
    assert hash(declared) == hash(other)
    assert declared.is_equivalent_frame(other)
    assert not declared.conflicts_with(other)
    assert other.role == second


@pytest.mark.parametrize(("role", "error"), [("canvas", ValueError), (["world"], TypeError)])
def test_unknown_roles_are_refused(role: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="role must be"):
        patient_frame(role=cast(Role, role))


def test_display_metadata_is_excluded_from_equality() -> None:
    plain = patient_frame()
    labelled = patient_frame(display={"label": "head first supine", "source": "scanner"})
    assert plain == labelled
    assert hash(plain) == hash(labelled)
    assert not plain.conflicts_with(labelled)
    assert labelled.display["label"] == "head first supine"


def test_metadata_key_order_does_not_change_identity() -> None:
    forward = patient_frame(context={"epoch": "J2000", "realization": "a"})
    reversed_order = patient_frame(context={"realization": "a", "epoch": "J2000"})
    assert forward == reversed_order
    assert hash(forward) == hash(reversed_order)
    assert tuple(forward.context) == ("epoch", "realization")


def test_comparison_with_a_non_frame_is_not_an_equality() -> None:
    other: object = ("dicom-frame-of-reference", UID)
    assert patient_frame() != other
    with pytest.raises(TypeError, match="ReferenceFrame"):
        patient_frame().conflicts_with(cast(ReferenceFrame, "other"))
    with pytest.raises(TypeError, match="ReferenceFrame"):
        patient_frame().is_equivalent_frame(cast(ReferenceFrame, "other"))


def test_declaration_is_immutable() -> None:
    frame = patient_frame(display={"label": "head first supine"})
    for attribute, value in (("coordinate_system", PLANE), ("extra", 1)):
        with pytest.raises(AttributeError):
            setattr(frame, attribute, value)
    view = cast(dict[str, object], frame.display)
    with pytest.raises(TypeError):
        view["label"] = "other"
    assert frame.display["label"] == "head first supine"


def test_coordinate_system_must_be_a_coordinate_system() -> None:
    with pytest.raises(TypeError, match="coordinate_system must be a CoordinateSystem"):
        ReferenceFrame.local(cast(CoordinateSystem, ("x", "y")))


@pytest.mark.parametrize(
    "identifier",
    [("dicom-frame-of-reference",), ("dicom-frame-of-reference", UID, "extra")],
)
def test_identifier_must_be_a_pair(identifier: object) -> None:
    with pytest.raises(ValueError, match="namespace, value"):
        ReferenceFrame.declared(cast("tuple[str, str]", identifier), PLANE)


@pytest.mark.parametrize(
    ("identifier", "message"),
    [
        pytest.param(["namespace", UID], "namespace, value", id="list"),
        pytest.param((1, UID), "namespace must be a string", id="non-string"),
    ],
)
def test_identifier_types_are_validated(identifier: object, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        ReferenceFrame.declared(cast("tuple[str, str]", identifier), PLANE)


def typed_frame(field: str, entry: str | int | float | bool | None) -> ReferenceFrame:
    """Build the reference declaration with one typed entry in the named metadata field."""
    metadata = {"entry": entry}
    return ReferenceFrame.declared(
        ("dicom-frame-of-reference", UID),
        PATIENT_AXES,
        definition=metadata if field == "definition" else None,
        context=metadata if field == "context" else None,
        display=metadata if field == "display" else None,
    )


@pytest.mark.parametrize("field", ["context", "definition"])
@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(True, 1, id="bool-int"),
        pytest.param(False, 0, id="false-zero"),
        pytest.param(1, 1.0, id="int-float"),
        pytest.param(True, "True", id="bool-str"),
        pytest.param(0, None, id="zero-none"),
    ],
)
def test_metadata_scalars_compare_by_type_as_well_as_value(
    field: str,
    first: str | int | float | bool | None,
    second: str | int | float | bool | None,
) -> None:
    """A boolean flag must not satisfy a numeric one, or a conflict would go unseen."""
    declared = typed_frame(field, first)
    other = typed_frame(field, second)
    assert declared != other
    assert declared.conflicts_with(other)
    assert len({declared, other}) == 2


@pytest.mark.parametrize("field", ["context", "definition"])
@pytest.mark.parametrize("entry", [True, False, 1, 1.0, "1", None])
def test_metadata_of_one_type_stays_equal_and_hashes_consistently(
    field: str,
    entry: str | int | float | bool | None,
) -> None:
    declared = typed_frame(field, entry)
    same = typed_frame(field, entry)
    assert declared == same
    assert hash(declared) == hash(same)
    assert not declared.conflicts_with(same)


def test_metadata_values_are_returned_as_declared() -> None:
    """The type tag is an internal comparison detail, not part of the stored value."""
    frame = typed_frame("context", True)
    assert frame.context["entry"] is True
    assert typed_frame("definition", 1).definition["entry"] == 1
    assert typed_frame("display", 1.5).display["entry"] == 1.5


def test_display_metadata_stays_excluded_from_typed_comparison() -> None:
    assert typed_frame("display", True) == typed_frame("display", 1)
    assert not typed_frame("display", True).conflicts_with(typed_frame("display", 1))


def test_metadata_must_be_flat_finite_data() -> None:
    with pytest.raises(TypeError, match="context"):
        ReferenceFrame.local(PLANE, context={"nested": cast(str, {"a": 1})})
    with pytest.raises(ValueError, match="finite"):
        ReferenceFrame.local(PLANE, context={"epoch": float("nan")})
    with pytest.raises(TypeError, match="mapping"):
        ReferenceFrame.local(PLANE, definition=cast(Mapping[str, str], [("schema", "v1")]))


def test_anonymous_identity_metadata_views_and_persistence() -> None:
    from xarrayrf import decode, encode

    frame = ReferenceFrame.anonymous(
        PLANE,
        role="world",
        definition={"space": "unknown"},
        context={"epoch": 1},
        display={"label": "synthetic"},
    )
    assert frame.is_anonymous
    assert frame.identifier[0] == "xarrayrf.anonymous"
    assert frame != ReferenceFrame.anonymous(
        PLANE, definition=frame.definition, context=frame.context
    )
    assert not ReferenceFrame.local(PLANE).is_anonymous
    assert not patient_frame().is_anonymous
    assert "anonymous=True" in repr(frame)
    restored = decode(encode(frame))
    assert restored == frame
    assert hash(restored) == hash(frame)
    assert restored.is_anonymous
    view = frame.with_coordinate_system(PATIENT_AXES)
    assert view.is_anonymous
    assert view.is_equivalent_frame(frame)
    assert view.definition == frame.definition
    assert view.context == frame.context
    assert view.display == frame.display
    assert view.role == frame.role
    assert (
        ReferenceFrame.declared(
            frame.identifier, PLANE, definition=frame.definition, context=frame.context
        )
        == frame
    )


@pytest.mark.parametrize("constructor", [ReferenceFrame.anonymous, ReferenceFrame.local])
def test_minted_frames_validate_declarations(constructor: Any) -> None:
    with pytest.raises(TypeError, match="coordinate_system must be"):
        constructor("bad")
    with pytest.raises(ValueError, match="role must be"):
        constructor(PLANE, role="bad")
    with pytest.raises(TypeError, match="definition"):
        constructor(PLANE, definition={"nested": []})
    with pytest.raises(ValueError, match="finite"):
        constructor(PLANE, context={"epoch": float("nan")})
