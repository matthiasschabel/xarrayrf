"""Public behavior of :func:`xarrayrf.encode` and :func:`xarrayrf.decode`."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import xarrayrf as xrf

ANATOMY = xrf.DirectionVocabulary(
    "test-anatomy", ("right-to-left", "anterior-to-posterior", "inferior-to-superior")
)
LPS = xrf.CoordinateSystem(
    ("x", "y", "z", "t"),
    ("mm", "mm", "mm", "s"),
    axis_types=("space", "space", "space", "time"),
    vocabulary=ANATOMY,
    orientation=("right-to-left", "anterior-to-posterior", "inferior-to-superior", None),
)
PATIENT = xrf.ReferenceFrame.declared(
    ("dicom-frame-of-reference", "1.2.3"),
    LPS,
    role="world",
    definition={"system": "patient"},
    context={"flag": True, "count": 1, "scale": 1.0, "note": None},
    display={"label": "Patient"},
)
ARRAY = xrf.ArrayCoordinates(
    ("i", "j", "k", "c"),
    ("1", "1", "mm", None),
    axis_types=(None, None, "space", "channel"),
    sample_offset=(0.5, 0.5, 0.5, None),
)


def round_trip(value: object, **kwargs: Any) -> Any:
    return xrf.decode(json.loads(json.dumps(xrf.encode(value))), **kwargs)


def affine(source: xrf.Endpoint, target: xrf.Endpoint) -> xrf.AffineTransform:
    size_out, size_in = len(target.axes), len(source.axes)
    matrix = np.full((size_out, size_in), 0.1) + np.eye(size_out, size_in) / 3.0
    return xrf.AffineTransform.from_matrix(
        source=source, target=target, matrix=matrix, translation=np.full(size_out, 1e-300)
    )


@pytest.mark.parametrize(
    "value",
    [
        ANATOMY,
        LPS,
        xrf.CoordinateSystem(("kx", "ω"), ("1/mm", "rad/s")),
        PATIENT,
        xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x",), ("m",))),
        ARRAY,
        affine(ARRAY, PATIENT),
    ],
    ids=["vocabulary", "oriented-system", "reciprocal", "frame", "local-frame", "array", "affine"],
)
def test_values_round_trip_through_json(value: object) -> None:
    decoded = round_trip(value)
    assert decoded == value
    assert type(decoded) is type(value)


def test_a_local_frame_keeps_its_identity() -> None:
    local = xrf.ReferenceFrame.local(LPS)
    decoded = round_trip(local)
    assert decoded == local and decoded.identifier[0] == xrf.LOCAL_NAMESPACE


def test_metadata_keeps_the_types_the_core_compares() -> None:
    decoded = round_trip(PATIENT)
    assert decoded.context == PATIENT.context
    assert type(decoded.context["flag"]) is bool and type(decoded.context["scale"]) is float
    assert not decoded.conflicts_with(PATIENT)


def test_affine_coefficients_round_trip_bitwise() -> None:
    values = np.array([0.1, 1.0 / 3.0, 1e-300, 5e-324, -2.5e17, np.nextafter(1.0, 2.0)])
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("a",), ("1",)),
        target=xrf.ReferenceFrame.local(xrf.CoordinateSystem(tuple("uvwxyz"), ("m",) * 6)),
        matrix=values.reshape(6, 1),
        translation=values[::-1],
    )
    decoded = round_trip(transform)
    assert decoded.matrix.tobytes() == transform.matrix.tobytes()
    assert decoded.translation.tobytes() == transform.translation.tobytes()


def test_named_basis_uses_the_existing_matrix_encoding() -> None:
    source = xrf.ArrayCoordinates(("j", "i"), ("1", "1"))
    target = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y", "z"), ("mm",) * 3))
    named = xrf.AffineTransform(
        source=source,
        target=target,
        target_axes=("x", "y", "z"),
        basis_vectors={"i": (-1.0, 3.0, -0.0), "j": (2.0, 0.5, 0.0)},
        translation=(1.0, -1.0, -0.0),
    )
    coefficients = xrf.AffineTransform.from_matrix(
        source=source,
        target=target,
        matrix=((2.0, -1.0), (0.5, 3.0), (0.0, 0.0)),
        translation=(1.0, -1.0, 0.0),
    )
    assert json.dumps(xrf.encode(named)) == json.dumps(xrf.encode(coefficients))
    assert round_trip(named) == coefficients
    assert hash(round_trip(named)) == hash(coefficients)


def test_a_composite_round_trips_with_its_array_coordinate_end() -> None:
    moving = xrf.ReferenceFrame.declared(("test", "moving"), LPS)
    chain = xrf.CompositeTransform(affine(ARRAY, PATIENT), affine(PATIENT, moving))
    assert round_trip(chain) == chain


class Shift:
    """A user transform: adds an offset along the first axis."""

    kind = "example:shift"
    version = 2

    def __init__(self, source: xrf.Endpoint, target: xrf.Endpoint, offset: float) -> None:
        self._source, self._target, self.offset = source, target, offset

    @property
    def source(self) -> xrf.Endpoint:
        return self._source

    @property
    def target(self) -> xrf.Endpoint:
        return self._target

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        result = np.array(points, dtype=np.float64)
        result[..., 0] += self.offset
        return result

    def to_data(self) -> Mapping[str, Any]:
        return {"offset": self.offset}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Shift) and (self.source, self.target, self.offset) == (
            other.source,
            other.target,
            other.offset,
        )

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.offset))


def decode_shift(
    data: Mapping[str, Any], *, version: int, source: xrf.Endpoint, target: xrf.Endpoint
) -> Shift:
    assert version == 2
    return Shift(source, target, data["offset"])


def test_a_user_transform_decodes_only_through_a_caller_decoder() -> None:
    moving = xrf.ReferenceFrame.declared(("test", "moving"), LPS)
    shift = Shift(PATIENT, moving, 0.25)
    assert isinstance(shift, xrf.SupportsEncoding)
    assert round_trip(shift, decoders={"example:shift": decode_shift}) == shift
    chain = xrf.CompositeTransform(affine(ARRAY, PATIENT), shift)
    assert round_trip(chain, decoders={"example:shift": decode_shift}) == chain
    with pytest.raises(xrf.MissingDecoderError, match="example:shift"):
        round_trip(shift)


def test_a_decoder_must_return_the_encoded_transform() -> None:
    moving = xrf.ReferenceFrame.declared(("test", "moving"), LPS)
    shift = Shift(PATIENT, moving, 0.25)

    def elsewhere(data: Mapping[str, Any], **_: Any) -> Shift:
        return Shift(moving, PATIENT, data["offset"])

    def nothing(data: Mapping[str, Any], **_: Any) -> object:
        return data

    for decoder in (elsewhere, nothing):
        with pytest.raises(xrf.DecoderResultError):
            round_trip(shift, decoders={"example:shift": decoder})


def test_extensions_must_be_namespaced_versioned_and_json() -> None:
    moving = xrf.ReferenceFrame.declared(("test", "moving"), LPS)
    for attribute, value, message in (
        ("kind", "shift", "namespaced"),
        ("version", -1, "non-negative integer"),
        ("version", True, "non-negative integer"),
    ):
        shift = Shift(PATIENT, moving, 0.25)
        setattr(shift, attribute, value)
        with pytest.raises(xrf.EncodingError, match=message):
            xrf.encode(shift)
    for offset, message in ((float("nan"), "non-finite"), (np.float32(1.0), "not JSON data")):
        with pytest.raises(xrf.EncodingError, match=message):
            xrf.encode(Shift(PATIENT, moving, offset))  # type: ignore[arg-type]


def test_encode_refuses_values_it_cannot_encode() -> None:
    with pytest.raises(TypeError, match="cannot be encoded"):
        xrf.encode(object())


def envelope(value: object) -> dict[str, Any]:
    return {"xarrayrf": xrf.SCHEMA_VERSION, "value": value}


@pytest.mark.parametrize(
    ("data", "error", "message"),
    [
        pytest.param(
            {"xarrayrf": 2, "value": {}}, xrf.UnsupportedVersionError, "version 2", id="version"
        ),
        pytest.param(
            {"xarrayrf": "1", "value": {}}, xrf.MalformedDataError, "integer", id="version-type"
        ),
        pytest.param({"value": {}}, xrf.MalformedDataError, "exactly the keys", id="envelope"),
        pytest.param([], xrf.MalformedDataError, "JSON object", id="not-object"),
        pytest.param(
            envelope({"kind": "projective"}), xrf.UnknownKindError, "projective", id="kind"
        ),
        pytest.param(envelope({"axes": []}), xrf.MalformedDataError, "string 'kind'", id="no-kind"),
        pytest.param(
            envelope({"kind": "array_coordinates", "axes": ["i"], "units": ["1"]}),
            xrf.MalformedDataError,
            "missing",
            id="missing-field",
        ),
        pytest.param(
            envelope({**xrf.encode(ARRAY)["value"], "extra": 1}),
            xrf.MalformedDataError,
            "extra",
            id="extra-field",
        ),
        pytest.param(
            envelope({**xrf.encode(ARRAY)["value"], "axes": [1, 2, 3, 4]}),
            xrf.MalformedDataError,
            "not a valid declaration",
            id="constructor-type",
        ),
        pytest.param(
            envelope({**xrf.encode(ARRAY)["value"], "units": "1"}),
            xrf.MalformedDataError,
            "JSON array",
            id="field-type",
        ),
    ],
)
def test_decode_classifies_failures(data: object, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        xrf.decode(data)


def test_constructor_errors_are_chained() -> None:
    data = envelope({**xrf.encode(ARRAY)["value"], "sample_offset": [2.0, 0.5, 0.5, None]})
    with pytest.raises(xrf.MalformedDataError) as caught:
        xrf.decode(data)
    assert isinstance(caught.value.__cause__, ValueError)


def test_a_transform_endpoint_must_be_an_endpoint() -> None:
    encoded = xrf.encode(affine(ARRAY, PATIENT))
    encoded["value"]["source"] = xrf.encode(ANATOMY)["value"]
    with pytest.raises(xrf.MalformedDataError, match="endpoint"):
        xrf.decode(encoded)


def test_a_decoder_must_keep_the_full_endpoint_declarations() -> None:
    moving = xrf.ReferenceFrame.declared(("test", "moving"), LPS, display={"label": "Moving"})
    relabelled = xrf.ReferenceFrame.declared(("test", "moving"), LPS, display={"label": "Other"})
    assert relabelled == moving
    shift = Shift(PATIENT, moving, 0.25)

    def relabel(data: Mapping[str, Any], *, source: xrf.Endpoint, **_: Any) -> Shift:
        return Shift(source, relabelled, data["offset"])

    with pytest.raises(xrf.DecoderResultError, match="other endpoints"):
        round_trip(shift, decoders={"example:shift": relabel})


@pytest.mark.parametrize("coefficient", ["1", True], ids=["string", "boolean"])
def test_affine_coefficients_are_not_coerced(coefficient: object) -> None:
    encoded = xrf.encode(affine(ARRAY, PATIENT))
    encoded["value"]["matrix"][0][0] = coefficient
    with pytest.raises(xrf.MalformedDataError, match="must hold numbers"):
        xrf.decode(encoded)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda data: data["value"]["data"].update({"offset": {1: 0.25}}),
        lambda data: data["value"]["data"].update({"offset": float("nan")}),
        lambda data: data["value"]["data"].update({"offset": object()}),
        lambda data: data.update({1: 0}),
    ],
    ids=["non-string-key", "non-finite", "non-json", "envelope-key"],
)
def test_decode_refuses_data_that_is_not_json(corrupt: Any) -> None:
    moving = xrf.ReferenceFrame.declared(("test", "moving"), LPS)
    encoded = xrf.encode(Shift(PATIENT, moving, 0.25))
    corrupt(encoded)
    with pytest.raises(xrf.MalformedDataError):
        xrf.decode(encoded, decoders={"example:shift": decode_shift})


def test_every_built_in_kind_encodes_exactly_its_schema_fields() -> None:
    """The decoder's field table and the encoder's output must not drift apart."""
    from xarrayrf._encoding import _FIELDS

    vocabulary = xrf.DirectionVocabulary("test-dirs", ("down-to-up",))
    system = xrf.CoordinateSystem(
        ("a",), ("mm",), vocabulary=vocabulary, orientation=("down-to-up",), axis_types=("space",)
    )
    frame = xrf.ReferenceFrame.local(system)
    array = xrf.ArrayCoordinates(("i",), ("1",), sample_offset=(0.5,))
    affine = xrf.AffineTransform.from_matrix(
        source=array, target=frame, matrix=[[2.0]], translation=[1.0]
    )
    frame_b = xrf.ReferenceFrame.local(system)
    composite = xrf.CompositeTransform(
        affine,
        xrf.AffineTransform.from_matrix(
            source=frame, target=frame_b, matrix=[[1.0]], translation=[0.0]
        ),
    )
    seen = set()
    for value in (
        vocabulary,
        system,
        frame,
        array,
        affine,
        composite,
        xrf.Grid(affine, {"i": ("i", [0, 1])}),
    ):
        encoded = xrf.encode(value)["value"]
        kind = encoded["kind"]
        assert set(encoded) - {"kind"} == _FIELDS[kind], kind
        seen.add(kind)
    assert seen == set(_FIELDS)
