"""Data-only encoding of the core value objects, versioned and validated on decode.

The encoded form is plain JSON data: dicts with string keys, lists, strings, finite numbers,
booleans and ``None``. ``json.dumps`` of it round-trips exactly, because finite float64 values are
written by ``repr``. Decoding rebuilds every object through its public constructor, so every
invariant is checked again. Nothing in the data names code: there is no registry and no import
path, and a user-defined transform decodes only through a decoder the caller passes in.

The schema is provisional until the NGFF round-trip fixtures pass; see
``docs/dev/architecture/persistence_design.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from ._affine import AffineTransform
from ._array_coordinates import ArrayCoordinates
from ._composite import CompositeTransform
from ._coordinate_system import CoordinateSystem
from ._frame import ReferenceFrame
from ._grid import Coordinate, Grid
from ._transform import Endpoint, SupportsPoints, check_transform
from ._vocabulary import DirectionVocabulary

SCHEMA_VERSION: Final = 1
"""The version :func:`encode` writes and the only one :func:`decode` reads."""

_ENVELOPE: Final = "xarrayrf"


class EncodingError(ValueError):
    """Base class of the failures to decode data, or to encode an extension's data."""


class UnsupportedVersionError(EncodingError):
    """The data is from a schema version this release does not read."""


class UnknownKindError(EncodingError):
    """A kind that is neither built in nor a namespaced extension kind."""


class MissingDecoderError(EncodingError):
    """A namespaced extension kind for which the caller supplied no decoder."""


class MalformedDataError(EncodingError):
    """Data of a known kind that is not a valid declaration of it."""


class DecoderResultError(EncodingError):
    """A caller's decoder returned something that is not the encoded transform."""


@runtime_checkable
class SupportsEncoding(Protocol):
    """A user-defined transform that can be encoded.

    ``kind`` is namespaced (``"package:name"``), so it never collides with a built-in kind and a
    missing decoder is distinguishable from unknown data. ``version`` is the extension's own
    payload version, which its decoder interprets. ``to_data`` returns the transform's
    parameters as JSON data; its endpoints are encoded separately.
    """

    kind: str
    version: int

    def to_data(self) -> Mapping[str, Any]:
        """Return the transform's parameters as JSON data, without its endpoints."""
        ...


type Decoder = Callable[..., SupportsPoints]
"""Called as ``decoder(data, version=..., source=..., target=...)``; returns the transform."""

_BUILT_IN: Final = frozenset(
    {
        "direction_vocabulary",
        "coordinate_system",
        "reference_frame",
        "array_coordinates",
        "affine_transform",
        "composite_transform",
        "grid",
    }
)


def encode(value: object) -> dict[str, Any]:
    """Encode a core value object as versioned JSON data.

    Args:
        value: A :class:`DirectionVocabulary`, :class:`CoordinateSystem`,
            :class:`ReferenceFrame`, :class:`ArrayCoordinates`, :class:`AffineTransform`,
            :class:`CompositeTransform`, :class:`Grid`, or a user-defined transform implementing
            :class:`SupportsEncoding`.

    Returns:
        ``{"xarrayrf": SCHEMA_VERSION, "value": {...}}``.

    Raises:
        TypeError: If the value, or a transform inside it, cannot be encoded.
        EncodingError: If an extension's kind is not namespaced, its version is not a
            non-negative integer, or its ``to_data`` does not return JSON data.
    """
    return {_ENVELOPE: SCHEMA_VERSION, "value": _encode(value)}


def decode(data: object, *, decoders: Mapping[str, Decoder] | None = None) -> Any:
    """Rebuild a value from :func:`encode` data, validating it completely.

    Args:
        data: The encoded data, typically from ``json.loads``.
        decoders: Decoders for namespaced extension kinds, by kind. They are the caller's
            code; nothing in the data selects code to run.

    Returns:
        The decoded value, equal to the one encoded.

    Raises:
        UnsupportedVersionError: If the schema version is not :data:`SCHEMA_VERSION`.
        UnknownKindError: If a kind is neither built in nor namespaced.
        MissingDecoderError: If a namespaced kind has no decoder in ``decoders``.
        MalformedDataError: If fields are missing, extra or of the wrong type, or a
            constructor refuses the declaration (the constructor's error is chained).
        DecoderResultError: If a decoder returns something other than a point transform
            between the encoded endpoints.
    """
    try:
        _check_json(data, where="the encoded data")
    except EncodingError as error:
        raise MalformedDataError(str(error)) from error
    envelope = _mapping(data, "the encoded data")
    if set(envelope) != {_ENVELOPE, "value"}:
        raise MalformedDataError(
            f"encoded data must have exactly the keys {_ENVELOPE!r} and 'value', got "
            f"{sorted(envelope)}",
        )
    version = envelope[_ENVELOPE]
    if isinstance(version, bool) or not isinstance(version, int):
        raise MalformedDataError(f"the schema version must be an integer, got {version!r}")
    if version != SCHEMA_VERSION:
        raise UnsupportedVersionError(
            f"schema version {version} is not supported; this release reads version "
            f"{SCHEMA_VERSION}",
        )
    return _Decoder(dict(decoders or {})).value(envelope["value"])


def _encode(value: object) -> dict[str, Any]:
    if isinstance(value, Grid):
        return {
            "kind": "grid",
            "transform": _encode(value.transform),
            "dims": list(value.dims),
            "coordinates": {
                name: {
                    "dim": entry[0],
                    "values": np.asarray(entry[1]).tolist(),
                    "dtype": str(np.asarray(entry[1]).dtype),
                }
                if isinstance(entry, tuple)
                else {"value": entry, "dtype": "int64" if isinstance(entry, int) else "float64"}
                for name, entry in value.coordinates.items()
            },
        }
    if isinstance(value, DirectionVocabulary):
        representatives = sorted({min(token, value.opposite(token)) for token in value.directions})
        return {
            "kind": "direction_vocabulary",
            "identifier": value.identifier,
            "directions": representatives,
        }
    if isinstance(value, CoordinateSystem):
        return {
            "kind": "coordinate_system",
            "axes": list(value.axes),
            "units": list(value.units),
            "axis_types": list(value.axis_types),
            "representation": value.representation,
            "vocabulary": None if value.vocabulary is None else _encode(value.vocabulary),
            "orientation": list(value.orientation),
        }
    if isinstance(value, ReferenceFrame):
        return {
            "kind": "reference_frame",
            "identifier": list(value.identifier),
            "coordinate_system": _encode(value.coordinate_system),
            "role": value.role,
            "definition": dict(value.definition),
            "context": dict(value.context),
            "display": dict(value.display),
        }
    if isinstance(value, ArrayCoordinates):
        return {
            "kind": "array_coordinates",
            "axes": list(value.axes),
            "units": list(value.units),
            "axis_types": list(value.axis_types),
            "sample_offset": list(value.sample_offset),
        }
    if isinstance(value, AffineTransform):
        return {
            "kind": "affine_transform",
            "source": _encode(value.source),
            "target": _encode(value.target),
            "matrix": value.matrix.tolist(),
            "translation": value.translation.tolist(),
        }
    if isinstance(value, CompositeTransform):
        return {
            "kind": "composite_transform",
            "transforms": [_encode(member) for member in value.transforms],
        }
    if isinstance(value, SupportsEncoding) and isinstance(value, SupportsPoints):
        return _encode_extension(value)
    raise TypeError(
        f"{type(value).__name__} cannot be encoded; a user-defined transform must implement "
        "SupportsEncoding (kind, version and to_data)",
    )


def _encode_extension(value: SupportsEncoding) -> dict[str, Any]:
    kind = value.kind
    if not isinstance(kind, str) or not _is_namespaced(kind):
        raise EncodingError(
            f"an extension kind must be namespaced as 'package:name', got {kind!r}",
        )
    version = value.version
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise EncodingError(
            f"extension {kind!r} must declare a non-negative integer version, got {version!r}",
        )
    data = value.to_data()
    _check_json(data, where=f"{kind} data")
    transform = check_transform(value)
    return {
        "kind": kind,
        "version": version,
        "source": _encode(transform.source),
        "target": _encode(transform.target),
        "data": _plain(data),
    }


def _is_namespaced(kind: str) -> bool:
    namespace, separator, name = kind.partition(":")
    return bool(separator and namespace and name and ":" not in name)


def _check_json(value: object, *, where: str) -> None:
    """Refuse anything ``json`` would not write back exactly."""
    if value is None or isinstance(value, bool | str | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EncodingError(f"{where} holds a non-finite number, which JSON cannot represent")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EncodingError(f"{where} has a non-string key {key!r}")
            _check_json(item, where=f"{where}[{key!r}]")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _check_json(item, where=f"{where}[{index}]")
        return
    raise EncodingError(f"{where} holds {type(value).__name__}, which is not JSON data")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MalformedDataError(f"{where} must be a JSON object, got {type(value).__name__}")
    return value


_FIELDS: Final = {
    "direction_vocabulary": {"identifier", "directions"},
    "coordinate_system": {
        "axes",
        "units",
        "axis_types",
        "representation",
        "vocabulary",
        "orientation",
    },
    "reference_frame": {
        "identifier",
        "coordinate_system",
        "role",
        "definition",
        "context",
        "display",
    },
    "array_coordinates": {"axes", "units", "axis_types", "sample_offset"},
    "affine_transform": {"source", "target", "matrix", "translation"},
    "composite_transform": {"transforms"},
    "grid": {"transform", "dims", "coordinates"},
}

_EXTENSION_FIELDS: Final = {"version", "source", "target", "data"}


class _Decoder:
    def __init__(self, decoders: dict[str, Decoder]) -> None:
        self._decoders = decoders

    def value(self, data: object) -> Any:
        record = _mapping(data, "an encoded value")
        kind = record.get("kind")
        if not isinstance(kind, str):
            raise MalformedDataError(f"an encoded value needs a string 'kind', got {kind!r}")
        fields = {key: item for key, item in record.items() if key != "kind"}
        if kind in _BUILT_IN:
            self._expect(kind, fields, _FIELDS[kind])
            try:
                return getattr(self, kind)(fields)
            except EncodingError:
                raise
            except (TypeError, ValueError) as error:
                raise MalformedDataError(f"{kind} is not a valid declaration: {error}") from error
        if not _is_namespaced(kind):
            raise UnknownKindError(
                f"kind {kind!r} is not a kind this release knows, and not a namespaced "
                "extension kind ('package:name')",
            )
        return self._extension(kind, fields)

    @staticmethod
    def _expect(kind: str, fields: Mapping[str, Any], expected: set[str]) -> None:
        if set(fields) != expected:
            missing, extra = sorted(expected - set(fields)), sorted(set(fields) - expected)
            raise MalformedDataError(
                f"{kind} must have exactly the fields {sorted(expected)}; missing {missing}, "
                f"extra {extra}",
            )

    def _endpoint(self, data: object) -> Endpoint:
        endpoint = self.value(data)
        if not isinstance(endpoint, ReferenceFrame | ArrayCoordinates):
            raise MalformedDataError(
                f"a transform endpoint must be a reference frame or array coordinates, got "
                f"{type(endpoint).__name__}",
            )
        return endpoint

    @staticmethod
    def _list(value: object, field: str) -> list[Any]:
        if not isinstance(value, list):
            raise MalformedDataError(f"{field} must be a JSON array, got {type(value).__name__}")
        return value

    @classmethod
    def _numbers(cls, value: object, field: str) -> list[Any]:
        """A nested JSON array of numbers; NumPy would coerce strings and booleans silently."""
        entries = cls._list(value, field)
        for entry in entries:
            if isinstance(entry, list):
                cls._numbers(entry, field)
            elif isinstance(entry, bool) or not isinstance(entry, int | float):
                raise MalformedDataError(f"{field} must hold numbers, got {entry!r}")
        return entries

    def direction_vocabulary(self, fields: Mapping[str, Any]) -> DirectionVocabulary:
        return DirectionVocabulary(
            fields["identifier"], self._list(fields["directions"], "directions")
        )

    def coordinate_system(self, fields: Mapping[str, Any]) -> CoordinateSystem:
        vocabulary = fields["vocabulary"]
        if vocabulary is not None:
            vocabulary = self.value(vocabulary)
            if not isinstance(vocabulary, DirectionVocabulary):
                raise MalformedDataError("a coordinate system's vocabulary must be a vocabulary")
        return CoordinateSystem(
            self._list(fields["axes"], "axes"),
            self._list(fields["units"], "units"),
            axis_types=self._list(fields["axis_types"], "axis_types"),
            representation=fields["representation"],
            vocabulary=vocabulary,
            orientation=self._list(fields["orientation"], "orientation"),
        )

    def reference_frame(self, fields: Mapping[str, Any]) -> ReferenceFrame:
        identifier = self._list(fields["identifier"], "identifier")
        system = self.value(fields["coordinate_system"])
        if not isinstance(system, CoordinateSystem):
            raise MalformedDataError("a frame's coordinate_system must be a coordinate system")
        return ReferenceFrame.declared(
            tuple(identifier),
            system,
            role=fields["role"],
            definition=_mapping(fields["definition"], "definition"),
            context=_mapping(fields["context"], "context"),
            display=_mapping(fields["display"], "display"),
        )

    def array_coordinates(self, fields: Mapping[str, Any]) -> ArrayCoordinates:
        return ArrayCoordinates(
            self._list(fields["axes"], "axes"),
            self._list(fields["units"], "units"),
            axis_types=self._list(fields["axis_types"], "axis_types"),
            sample_offset=self._list(fields["sample_offset"], "sample_offset"),
        )

    def affine_transform(self, fields: Mapping[str, Any]) -> AffineTransform:
        return AffineTransform(
            source=self._endpoint(fields["source"]),
            target=self._endpoint(fields["target"]),
            matrix=self._numbers(fields["matrix"], "matrix"),
            translation=self._numbers(fields["translation"], "translation"),
        )

    def grid(self, fields: Mapping[str, Any]) -> Grid:
        coordinates: dict[str, Coordinate] = {}
        for name, data in _mapping(fields["coordinates"], "grid coordinates").items():
            record = _mapping(data, f"grid coordinate {name!r}")
            if "value" in record:
                self._expect("scalar coordinate", record, {"value", "dtype"})
                values = [record["value"]]
            else:
                self._expect("varying coordinate", record, {"dim", "values", "dtype"})
                values = self._list(record["values"], "values")
            dtype = record["dtype"]
            if dtype not in ("int64", "float64"):
                raise MalformedDataError("coordinate dtype must be int64 or float64")
            for value in values:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise MalformedDataError(f"coordinate {name!r} values must be numbers")
                if dtype == "int64":
                    if not isinstance(value, int):
                        raise MalformedDataError("int64 coordinate values must be integers")
                    if not -(2**63) <= value < 2**63:
                        raise MalformedDataError("integer coordinate values must fit in int64")
            try:
                decoded_values = np.asarray(values, dtype=dtype)
            except OverflowError as error:
                raise MalformedDataError("float64 coordinate values must be finite") from error
            if not np.all(np.isfinite(decoded_values)):
                raise MalformedDataError("coordinate values must be finite")
            if "value" in record:
                coordinates[name] = decoded_values[0].item()
            else:
                coordinates[name] = (record["dim"], decoded_values)
        dims = self._list(fields["dims"], "grid dims")
        varying = {
            entry[0]: name for name, entry in coordinates.items() if isinstance(entry, tuple)
        }
        if not all(isinstance(dim, str) for dim in dims) or sorted(dims) != sorted(varying):
            raise MalformedDataError(
                f"grid dims {dims!r} must name each varying coordinate's dimension once"
            )
        # JSON object order is not preserved by every tool, so dims, not key order, fixes the order.
        ordered = {varying[dim]: coordinates[varying[dim]] for dim in dims}
        ordered.update({name: entry for name, entry in coordinates.items() if name not in ordered})
        return Grid(self.value(fields["transform"]), ordered)

    def composite_transform(self, fields: Mapping[str, Any]) -> CompositeTransform:
        members = [self.value(item) for item in self._list(fields["transforms"], "transforms")]
        return CompositeTransform(*members)

    def _extension(self, kind: str, fields: Mapping[str, Any]) -> SupportsPoints:
        decoder = self._decoders.get(kind)
        if decoder is None:
            raise MissingDecoderError(
                f"extension kind {kind!r} needs a decoder; pass decoders={{{kind!r}: ...}}",
            )
        self._expect(kind, fields, _EXTENSION_FIELDS)
        version = fields["version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise MalformedDataError(f"{kind} version must be a non-negative integer")
        source = self._endpoint(fields["source"])
        target = self._endpoint(fields["target"])
        data = _mapping(fields["data"], f"{kind} data")
        result = decoder(data, version=version, source=source, target=target)
        try:
            transform = check_transform(result)
        except (TypeError, ValueError) as error:
            raise DecoderResultError(
                f"the decoder for {kind!r} returned {type(result).__name__}, not a point transform",
            ) from error
        # Frame equality ignores role and display, so compare the full declarations.
        if _encode(transform.source) != _encode(source) or _encode(transform.target) != _encode(
            target
        ):
            raise DecoderResultError(
                f"the decoder for {kind!r} returned a transform between other endpoints than "
                "the encoded ones",
            )
        return transform
