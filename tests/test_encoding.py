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


PACKING = xrf.SymmetricPacking.voigt(3)
PACKED = xrf.TensorComponentDeclaration(
    xrf.SYMMETRIC_SECOND_ORDER_CONTRAVARIANT_TENSOR,
    ("component",),
    xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y", "z"), ("m", "m", "m"))),
    packing=PACKING,
)


def test_packings_and_packed_declarations_round_trip() -> None:
    for value in (PACKING, xrf.SymmetricPacking.mandel(2), PACKED):
        decoded = xrf.decode(json.loads(json.dumps(xrf.encode(value))))
        assert decoded == value and hash(decoded) == hash(value)


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
        pytest.param(
            envelope({**xrf.encode(PACKING)["value"], "index_pairs": [[0, 0], 1]}),
            xrf.MalformedDataError,
            "JSON array",
            id="packing-pairs",
        ),
        pytest.param(
            envelope({**xrf.encode(PACKING)["value"], "index_pairs": [[0.0, 0]]}),
            xrf.MalformedDataError,
            "not a valid declaration",
            id="packing-index-type",
        ),
        pytest.param(
            envelope({**xrf.encode(PACKING)["value"], "off_diagonal_scale": True}),
            xrf.MalformedDataError,
            "must be a number",
            id="packing-scale",
        ),
        pytest.param(
            envelope({**xrf.encode(PACKED)["value"], "packing": xrf.encode(ARRAY)["value"]}),
            xrf.MalformedDataError,
            "must be a symmetric packing",
            id="packed-packing",
        ),
        pytest.param(
            envelope({**xrf.encode(PACKED)["value"], "tensor_type": xrf.encode(ARRAY)["value"]}),
            xrf.MalformedDataError,
            "must be a tensor type",
            id="packed-type",
        ),
        pytest.param(
            envelope({**xrf.encode(PACKED)["value"], "component_dim": ["component"]}),
            xrf.MalformedDataError,
            "component_dim must be a string",
            id="packed-dim",
        ),
        pytest.param(
            envelope(
                {**xrf.encode(PACKED)["value"], "tensor_type": xrf.encode(xrf.VECTOR)["value"]}
            ),
            xrf.MalformedDataError,
            "only a symmetric order-2 type can be packed",
            id="packed-unsymmetric",
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


def _transitions() -> list[object]:
    """One of each chart transition, spherical and geodetic, in both directions, and a field."""
    spherical = xrf.ReferenceFrame.local(
        xrf.CoordinateSystem(
            ("lon", "lat", "r"),
            ("deg", "deg", "m"),
            representation="spherical",
            axis_roles=("longitude", "latitude", "radius"),
        )
    )
    geodetic = xrf.ReferenceFrame.local(
        xrf.CoordinateSystem(
            ("lon", "lat", "h"),
            ("deg", "deg", "m"),
            representation="geodetic",
            axis_roles=("longitude", "latitude", "height"),
        )
    )
    cartesian = xrf.CoordinateSystem(("x", "y", "z"), ("m", "m", "m"))
    to_sphere = xrf.SphericalToCartesian(spherical, spherical.with_coordinate_system(cartesian))
    to_ellipsoid = xrf.GeodeticToCartesian(
        geodetic,
        geodetic.with_coordinate_system(cartesian),
        semi_major_axis=6378137.0,
        inverse_flattening=298.257223563,
    )
    field = xrf.EastNorthUp(geodetic, semi_major_axis=6378137.0, inverse_flattening=298.257223563)
    return [to_sphere, to_sphere.inverse(), to_ellipsoid, to_ellipsoid.inverse(), field]


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
        xrf.CROSS_PRODUCT_PSEUDOVECTOR,
        xrf.TensorComponentDeclaration(xrf.COVECTOR, ("component",), frame),
        xrf.SymmetricPacking.mandel(2),
        *_transitions(),
        xrf.TensorComponentDeclaration(
            xrf.SYMMETRIC_SECOND_ORDER_COVARIANT_TENSOR,
            ("component",),
            frame,
            packing=xrf.SymmetricPacking.upper_triangular(1),
        ),
    ):
        encoded = xrf.encode(value)["value"]
        kind = encoded["kind"]
        assert set(encoded) - {"kind"} == _FIELDS[kind], kind
        seen.add(kind)
    pair = xrf.DeclaredInversePair(dense_field(np.float32), backward_field())
    for value in (dense_field(np.float64), bspline_field(), pair, pair.backward):
        encoded = xrf.encode(value, arrays={})["value"]
        kind = encoded["kind"]
        assert set(encoded) - {"kind"} == _FIELDS[kind], kind
        seen.add(kind)
    assert seen == set(_FIELDS)


GEODETIC_SYSTEM = xrf.CoordinateSystem(
    ("time", "lat", "lon", "h"),
    ("s", "degrees_north", "degrees_east", "m"),
    representation="geodetic",
    axis_roles=(None, "latitude", "longitude", "height"),
)


@pytest.mark.parametrize(
    "value",
    [
        GEODETIC_SYSTEM,
        xrf.CoordinateSystem(
            ("ra", "dec"),
            ("rad", "rad"),
            representation="spherical",
            axis_roles=("longitude", "latitude"),
        ),
        xrf.ReferenceFrame.declared(
            ("test-earth", "sphere-6371229"),
            GEODETIC_SYSTEM,
            definition={"figure": "sphere", "semi_major_axis": 6371229.0, "length_unit": "m"},
        ),
    ],
    ids=["geodetic", "spherical-radians", "frame"],
)
def test_angular_declarations_round_trip(value: object) -> None:
    decoded = round_trip(value)
    assert decoded == value
    assert type(decoded) is type(value)


def test_axis_roles_are_written_only_for_angular_systems() -> None:
    assert "axis_roles" not in xrf.encode(LPS)["value"]
    assert xrf.encode(GEODETIC_SYSTEM)["value"]["axis_roles"] == [
        None,
        "latitude",
        "longitude",
        "height",
    ]


def test_an_angular_record_without_roles_is_malformed() -> None:
    data = xrf.encode(GEODETIC_SYSTEM)
    del data["value"]["axis_roles"]
    with pytest.raises(xrf.MalformedDataError, match="needs axis_roles"):
        xrf.decode(data)


def test_a_cartesian_record_with_roles_is_malformed() -> None:
    data = xrf.encode(xrf.CoordinateSystem(("x",), ("m",)))
    data["value"]["axis_roles"] = [None]
    with pytest.raises(xrf.MalformedDataError, match="a Cartesian coordinate system has none"):
        xrf.decode(data)


# --- Field transforms and declared pairs: arrays by reference ---------------------------------

FIELD_SYSTEM = xrf.CoordinateSystem(("x", "y", "z"), ("mm",) * 3)
FIXED = xrf.ReferenceFrame.declared(("test", "fixed"), FIELD_SYSTEM)
MOVING = xrf.ReferenceFrame.declared(("test", "moving"), FIELD_SYSTEM)
FIELD_INDEX = xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3, sample_offset=(0.5,) * 3)
FIELD_POINTS = np.random.default_rng(3).uniform(0.0, 3.0, size=(20, 3))


def field_grid(shape: tuple[int, ...], frame: xrf.ReferenceFrame = FIXED) -> xrf.Grid:
    transform = xrf.AffineTransform.from_matrix(
        source=FIELD_INDEX, target=frame, matrix=np.diag([1.0, 0.8, 1.2]), translation=[0.1, 0, 0]
    )
    return xrf.Grid(
        transform, {axis: (axis, np.arange(n)) for axis, n in zip("ijk", shape, strict=True)}
    )


def dense_field(
    dtype: type = np.float64, shape: tuple[int, ...] = (4, 5, 3), outside: Any = "identity"
) -> xrf.DisplacementFieldTransform:
    values: npt.NDArray[Any] = (
        np.random.default_rng(1).normal(scale=0.2, size=(*shape, 3)).astype(dtype)
    )
    return xrf.DisplacementFieldTransform(
        source=FIXED, target=MOVING, field=values, grid=field_grid(shape), outside=outside
    )


def backward_field() -> xrf.DisplacementFieldTransform:
    values = np.random.default_rng(2).normal(scale=0.2, size=(4, 5, 3, 3))
    return xrf.DisplacementFieldTransform(
        source=MOVING, target=FIXED, field=values, grid=field_grid((4, 5, 3), MOVING)
    )


def bspline_field(support: Any = "half-open") -> xrf.BSplineTransform:
    values = np.random.default_rng(4).normal(scale=0.2, size=(5, 6, 5, 3)).astype(np.float32)
    return xrf.BSplineTransform(
        source=FIXED,
        target=MOVING,
        coefficients=values,
        grid=field_grid((5, 6, 5)),
        outside="nearest",
        support=support,
    )


def by_reference(value: object, **kwargs: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Encode through JSON text with a fresh store; return the decoded value, data and store."""
    arrays: dict[str, Any] = {}
    data = json.loads(json.dumps(xrf.encode(value, arrays=arrays)))
    return xrf.decode(data, arrays=arrays, **kwargs), data, arrays


@pytest.mark.parametrize(
    "value",
    [
        dense_field(np.float32),
        dense_field(np.float64, outside="raise"),
        dense_field(outside="nearest"),
        bspline_field(),
        bspline_field("closed"),
    ],
    ids=["dense-float32", "dense-raise", "dense-nearest", "bspline-half-open", "bspline-closed"],
)
def test_field_transforms_round_trip_by_reference(value: Any) -> None:
    decoded, data, arrays = by_reference(value)
    assert type(decoded) is type(value)
    # Equality is identity, so the decoded transform is new; its declaration and data are not.
    assert decoded != value
    assert xrf.encode(decoded, arrays={}) == data
    stored, restored = (
        (value.field, decoded.field)
        if isinstance(value, xrf.DisplacementFieldTransform)
        else (value.coefficients, decoded.coefficients)
    )
    assert restored.dtype == stored.dtype and np.array_equal(restored, stored)
    assert not restored.flags.writeable
    reference = data["value"]["field" if "field" in data["value"] else "coefficients"]
    assert reference == {"reference": xrf_reference(stored), "dtype": stored.dtype.name}
    assert list(arrays) == [reference["reference"]]
    assert decoded.outside == value.outside and decoded.grid == value.grid
    assert_points_equal(decoded, value)


def xrf_reference(values: npt.NDArray[np.floating]) -> str:
    """The documented digest, computed independently of the encoder's blocked hashing."""
    import hashlib

    header = f"{values.dtype.name}:{','.join(map(str, values.shape))}\n".encode("ascii")
    body = values.astype(values.dtype.newbyteorder("<")).tobytes(order="C")
    return "sha256:" + hashlib.sha256(header + body).hexdigest()


def assert_points_equal(decoded: Any, original: Any) -> None:
    np.testing.assert_array_equal(
        decoded.transform_point(FIELD_POINTS), original.transform_point(FIELD_POINTS)
    )


def test_a_composite_with_a_field_round_trips() -> None:
    chain = xrf.CompositeTransform(
        affine(FIELD_INDEX, FIXED), dense_field(np.float32), affine(MOVING, FIXED)
    )
    decoded, _, arrays = by_reference(chain)
    assert [type(member) for member in decoded.transforms] == [
        type(member) for member in chain.transforms
    ]
    assert decoded.transforms[0] == chain.transforms[0] and len(arrays) == 1
    assert_points_equal(decoded, chain)


def test_a_declared_pair_and_each_direction_round_trip() -> None:
    pair = xrf.DeclaredInversePair(dense_field(np.float32), backward_field())
    decoded, _, arrays = by_reference(pair)
    assert isinstance(decoded, xrf.DeclaredInversePair) and len(arrays) == 2
    assert_points_equal(decoded.forward, pair.forward)
    assert_points_equal(decoded.backward, pair.backward)
    for direction in ("forward", "backward"):
        half = getattr(pair, direction)
        decoded, data, _ = by_reference(half)
        assert data["value"]["direction"] == direction
        assert decoded.direction == direction
        assert (decoded.source, decoded.target) == (half.source, half.target)
        assert_points_equal(decoded, half)
        assert_points_equal(decoded.declared_inverse(), half.declared_inverse())


def test_a_pair_of_user_transforms_needs_no_arrays() -> None:
    pair = xrf.DeclaredInversePair(Shift(FIXED, MOVING, 0.25), Shift(MOVING, FIXED, -0.25))
    data = json.loads(json.dumps(xrf.encode(pair.forward)))
    decoded = xrf.decode(data, decoders={"example:shift": decode_shift})
    assert decoded == pair.forward and decoded.declared_inverse() == pair.backward


def test_a_shared_store_keeps_arrays_with_equal_bytes_apart() -> None:
    """Zeros of another shape or dtype have the same bytes; the header keeps them apart."""
    zeros = [
        xrf.DisplacementFieldTransform(
            source=FIXED, target=MOVING, field=np.zeros((*shape, 3), dtype), grid=field_grid(shape)
        )
        for shape, dtype in (
            ((2, 6, 2), np.float32),
            ((3, 4, 2), np.float32),
            ((2, 3, 2), np.float64),
        )
    ]
    arrays: dict[str, Any] = {}
    records = [xrf.encode(value, arrays=arrays) for value in zeros]
    assert len(arrays) == 3
    for value, record in zip(zeros, records, strict=True):
        decoded = xrf.decode(record, arrays=arrays)
        assert decoded.field.shape == value.field.shape and decoded.field.dtype == value.field.dtype
    twin = dense_field(np.float32)
    xrf.encode(twin, arrays=arrays)
    xrf.encode(dense_field(np.float32), arrays=arrays)
    assert len(arrays) == 4


def test_a_big_endian_array_decodes_and_keeps_its_dtype() -> None:
    value = dense_field(np.float32)
    arrays: dict[str, Any] = {}
    data = xrf.encode(value, arrays=arrays)
    ((reference, stored),) = arrays.items()
    arrays[reference] = stored.astype(">f4")
    decoded = xrf.decode(data, arrays=arrays)
    assert decoded.field.dtype == np.float32 and np.array_equal(decoded.field, value.field)


def test_encoding_a_field_needs_a_store() -> None:
    for value in (
        dense_field(),
        bspline_field(),
        xrf.CompositeTransform(dense_field(), affine(MOVING, FIXED)),
        xrf.DeclaredInversePair(dense_field(), backward_field()).backward,
    ):
        with pytest.raises(TypeError, match="pass arrays="):
            xrf.encode(value)


def field_record() -> tuple[dict[str, Any], dict[str, Any], str]:
    arrays: dict[str, Any] = {}
    data = xrf.encode(dense_field(np.float32), arrays=arrays)
    return data, arrays, data["value"]["field"]["reference"]


def test_decoding_a_reference_needs_its_array() -> None:
    data, _, reference = field_record()
    with pytest.raises(xrf.MissingArrayError, match="pass arrays="):
        xrf.decode(data)
    with pytest.raises(xrf.MissingArrayError, match=reference):
        xrf.decode(data, arrays={})


def test_a_resolved_array_must_match_its_record() -> None:
    data, arrays, reference = field_record()
    stored = arrays[reference]
    altered = np.array(stored)
    altered[0, 0, 0, 0] += 1
    masked = np.ma.masked_array(stored, mask=np.zeros(stored.shape, bool))
    masked[1, 1, 1, 1] = np.ma.masked
    for entry, message in (
        (stored.astype(np.float64), "recorded as float32"),
        (stored.astype(np.int32), "recorded as float32"),
        (altered, "does not match its digest"),
        (stored.reshape(5, 4, 3, 3), "does not match its digest"),
        (masked, "masked array"),
    ):
        with pytest.raises(xrf.MalformedDataError, match=message):
            xrf.decode(data, arrays={reference: entry})


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda v: v["field"].update(reference="md5:00"), "64 lowercase hex"),
        (lambda v: v["field"].update(reference="sha256:" + "A" * 64), "64 lowercase hex"),
        (lambda v: v["field"].update(dtype="float16"), "float32 or float64"),
        (lambda v: v["field"].update(shape=[4, 5, 3, 3]), "extra"),
        (lambda v: v.update(outside="clamp"), "outside must be one of"),
        (lambda v: v.update(grid=v["source"]), "grid must be a grid"),
        (lambda v: v.update(source=v["grid"]), "endpoints must be reference frames"),
        (lambda v: v.pop("outside"), "missing"),
    ],
)
def test_malformed_field_records_are_refused(edit: Any, message: str) -> None:
    data, arrays, _ = field_record()
    edit(data["value"])
    with pytest.raises(xrf.MalformedDataError, match=message):
        xrf.decode(data, arrays=arrays)


def test_a_wrongly_shaped_record_is_a_chained_constructor_refusal() -> None:
    data, arrays, _ = field_record()
    data["value"]["grid"] = xrf.encode(field_grid((4, 5, 4)))["value"]
    with pytest.raises(xrf.MalformedDataError, match="must have shape") as caught:
        xrf.decode(data, arrays=arrays)
    assert isinstance(caught.value.__cause__, ValueError)


def test_malformed_pair_records_are_refused() -> None:
    arrays: dict[str, Any] = {}
    pair = xrf.DeclaredInversePair(dense_field(), backward_field())
    data = xrf.encode(pair.forward, arrays=arrays)
    data["value"]["direction"] = "inverse"
    with pytest.raises(xrf.MalformedDataError, match="'forward' or 'backward'"):
        xrf.decode(data, arrays=arrays)
    data["value"]["direction"] = "forward"
    data["value"]["pair"] = data["value"]["pair"]["forward"]
    with pytest.raises(xrf.MalformedDataError, match="must be a declared inverse pair"):
        xrf.decode(data, arrays=arrays)
    swapped = xrf.encode(pair, arrays=arrays)
    swapped["value"]["backward"] = swapped["value"]["forward"]
    with pytest.raises(xrf.MalformedDataError, match="backward must map"):
        xrf.decode(swapped, arrays=arrays)


def test_large_arrays_digest_in_bounded_blocks() -> None:
    """A slab whose single leading row exceeds the block is split, with the same digest."""
    from xarrayrf._encoding import _DIGEST_BLOCK_BYTES, array_reference

    rows = _DIGEST_BLOCK_BYTES // (8 * 3) + 7
    values = np.moveaxis(np.arange(3.0 * 2 * rows).reshape(3, 1, 2, rows), 0, -1)
    assert values[:1].nbytes > _DIGEST_BLOCK_BYTES and not values.flags.c_contiguous
    assert array_reference(values) == xrf_reference(values)
