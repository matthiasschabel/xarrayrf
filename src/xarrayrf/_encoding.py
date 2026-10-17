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

import hashlib
import math
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ._affine import AffineTransform
from ._array_coordinates import ArrayCoordinates
from ._chart_transitions import (
    CartesianToGeodetic,
    CartesianToSpherical,
    GeodeticToCartesian,
    SphericalToCartesian,
)
from ._composite import CompositeTransform
from ._coordinate_system import CARTESIAN, CoordinateSystem
from ._declared_inverse import DeclaredInversePair, PairedTransform
from ._field_transform import BSplineTransform, DisplacementFieldTransform, PositionFieldTransform
from ._frame import ReferenceFrame
from ._frame_field import EastNorthUp, FrameField
from ._grid import Coordinate, Grid
from ._symmetric_packing import SymmetricPacking
from ._tensor_component_declaration import ComponentBasis, TensorComponentDeclaration
from ._tensor_type import TensorType
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


class MissingArrayError(EncodingError):
    """An array reference that the caller's ``arrays`` does not resolve."""


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
        "spherical_to_cartesian",
        "cartesian_to_spherical",
        "geodetic_to_cartesian",
        "cartesian_to_geodetic",
        "east_north_up",
        "grid",
        "tensor_type",
        "tensor_component_declaration",
        "symmetric_packing",
        "packed_tensor_component_declaration",
        "displacement_field_transform",
        "position_field_transform",
        "bspline_transform",
        "declared_inverse_pair",
        "paired_transform",
    }
)

_REFERENCE_PREFIX: Final = "sha256:"
"""Array references are content digests: SHA-256 of a dtype and shape header, then the C-order,
little-endian bytes."""

_DIGEST_BLOCK_BYTES: Final = 2**24
"""Bytes copied at once to hash a non-contiguous array."""

type ArrayStore = MutableMapping[str, npt.NDArray[np.floating]]
"""Where :func:`encode` puts the arrays a record references, keyed by reference."""


def encode(value: object, *, arrays: ArrayStore | None = None) -> dict[str, Any]:
    """Encode a core value object as versioned JSON data.

    Field transforms hold arrays that an inline record cannot carry. Their records reference
    each array by its content digest (``"sha256:<hex>"``), and the array itself goes into
    ``arrays`` under that reference, as the transform's read-only view.

    Args:
        value: A :class:`DirectionVocabulary`, :class:`CoordinateSystem`,
            :class:`ReferenceFrame`, :class:`ArrayCoordinates`, :class:`AffineTransform`,
            :class:`CompositeTransform`, :class:`Grid`, :class:`TensorType`,
            :class:`SymmetricPacking`, :class:`TensorComponentDeclaration`,
            :class:`DisplacementFieldTransform`, :class:`PositionFieldTransform`,
            :class:`BSplineTransform`, :class:`DeclaredInversePair` or one of its directions, or a user-defined transform
            implementing :class:`SupportsEncoding`.
        arrays: A mutable mapping that receives the referenced arrays; required when the value
            holds a field transform. Several values may share one mapping: equal arrays have
            equal references.

    Returns:
        ``{"xarrayrf": SCHEMA_VERSION, "value": {...}}``.

    Raises:
        TypeError: If the value, or a transform inside it, cannot be encoded, or it holds a
            field transform and ``arrays`` is None.
        EncodingError: If an extension's kind is not namespaced, its version is not a
            non-negative integer, or its ``to_data`` does not return JSON data.
    """
    return {_ENVELOPE: SCHEMA_VERSION, "value": _encode(value, arrays)}


def decode(
    data: object,
    *,
    decoders: Mapping[str, Decoder] | None = None,
    arrays: Mapping[str, Any] | None = None,
) -> Any:
    """Rebuild a value from :func:`encode` data, validating it completely.

    Args:
        data: The encoded data, typically from ``json.loads``.
        decoders: Decoders for namespaced extension kinds, by kind. They are the caller's
            code; nothing in the data selects code to run.
        arrays: Resolves the array references of field transform records: a mapping from
            reference to an array-like (a dict, a Zarr group, the arrays a Dataset holds). Each
            array is loaded, checked against its recorded dtype and digest, and copied into the
            transform's immutable storage.

    Returns:
        The decoded value, equal to the one encoded.

    Raises:
        UnsupportedVersionError: If the schema version is not :data:`SCHEMA_VERSION`.
        UnknownKindError: If a kind is neither built in nor namespaced.
        MissingDecoderError: If a namespaced kind has no decoder in ``decoders``.
        MissingArrayError: If a referenced array is not in ``arrays``, or ``arrays`` is None.
        MalformedDataError: If fields are missing, extra or of the wrong type, a referenced
            array has another dtype or digest, or a constructor refuses the declaration (the
            constructor's error is chained).
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
    return _Decoder(dict(decoders or {}), arrays).value(envelope["value"])


def _encode(value: object, arrays: ArrayStore | None = None) -> dict[str, Any]:
    """Encode a value; ``arrays`` reaches every transform that may hold a field."""
    if isinstance(value, Grid):
        return {
            "kind": "grid",
            "intervals": {name: rows.tolist() for name, rows in value.intervals.items()},
            "transform": _encode(value.transform, arrays),
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
            # Written only where roles exist, so Cartesian records are unchanged.
            **({"axis_roles": list(value.axis_roles)} if value.representation != CARTESIAN else {}),
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
    if isinstance(value, SphericalToCartesian | CartesianToSpherical):
        return {
            "kind": _TRANSITION_KINDS[type(value)],
            "source": _encode(value.source),
            "target": _encode(value.target),
            "longitude_start": value.longitude_start,
        }
    if isinstance(value, GeodeticToCartesian | CartesianToGeodetic):
        return {
            "kind": _TRANSITION_KINDS[type(value)],
            "source": _encode(value.source),
            "target": _encode(value.target),
            "semi_major_axis": value.semi_major_axis,
            "inverse_flattening": value.inverse_flattening,
            "longitude_start": value.longitude_start,
        }
    if isinstance(value, CompositeTransform):
        return {
            "kind": "composite_transform",
            "transforms": [_encode(member, arrays) for member in value.transforms],
        }
    if isinstance(value, TensorType):
        return {
            "kind": "tensor_type",
            "index_variance": list(value.index_variance),
            "reflection_parity": value.reflection_parity,
            "determinant_exponent": value.determinant_exponent,
            "index_symmetry": value.index_symmetry,
            "requires_orthogonal_map": value.requires_orthogonal_map,
        }
    if isinstance(value, SymmetricPacking):
        return {
            "kind": "symmetric_packing",
            "index_pairs": [list(pair) for pair in value.index_pairs],
            "off_diagonal_scale": value.off_diagonal_scale,
        }
    if isinstance(value, EastNorthUp):
        geodetic = value.base.coordinate_system.representation != "spherical"
        return {
            "kind": "east_north_up",
            "base": _encode(value.base),
            "axes": list(value.axes),
            "semi_major_axis": value.semi_major_axis if geodetic else None,
            "inverse_flattening": value.inverse_flattening if geodetic else None,
            "length_unit": value.length_unit,
        }
    if isinstance(value, TensorComponentDeclaration):
        for basis in value.bases:
            if isinstance(basis, FrameField) and not isinstance(basis, EastNorthUp):
                raise TypeError(
                    f"the declaration's basis {type(basis).__name__} is a user frame field, "
                    "which cannot be encoded yet; only EastNorthUp is persisted"
                )
    if isinstance(value, TensorComponentDeclaration) and value.packing is not None:
        # Its own kind, so a release without packing refuses it rather than misreading it.
        return {
            "kind": "packed_tensor_component_declaration",
            "tensor_type": _encode(value.tensor_type),
            "component_dim": value.component_dims[0],
            "basis": _encode(value.bases[0]),
            "packing": _encode(value.packing),
        }
    if isinstance(value, TensorComponentDeclaration):
        return {
            "kind": "tensor_component_declaration",
            "tensor_type": _encode(value.tensor_type),
            "component_dims": list(value.component_dims),
            "bases": [_encode(basis) for basis in value.bases],
        }
    if isinstance(value, DisplacementFieldTransform):
        return {
            "kind": "displacement_field_transform",
            "source": _encode(value.source),
            "target": _encode(value.target),
            "grid": _encode(value.grid, arrays),
            "outside": value.outside,
            "field": _store(value.field, arrays, value),
        }
    if isinstance(value, PositionFieldTransform):
        return {
            "kind": "position_field_transform",
            "source": _encode(value.source),
            "target": _encode(value.target),
            "grid": _encode(value.grid, arrays),
            "outside": value.outside,
            "field": _store(value.field, arrays, value),
        }
    if isinstance(value, BSplineTransform):
        return {
            "kind": "bspline_transform",
            "source": _encode(value.source),
            "target": _encode(value.target),
            "grid": _encode(value.grid, arrays),
            "order": value.order,
            "outside": value.outside,
            "support": value.support,
            "coefficients": _store(value.coefficients, arrays, value),
        }
    if isinstance(value, DeclaredInversePair):
        return {
            "kind": "declared_inverse_pair",
            "forward": _encode(value.forward.transform, arrays),
            "backward": _encode(value.backward.transform, arrays),
        }
    if isinstance(value, PairedTransform):
        # The whole pair, so the decoded direction keeps its declared inverse.
        return {
            "kind": "paired_transform",
            "pair": _encode(value.pair, arrays),
            "direction": value.direction,
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


def _store(
    values: npt.NDArray[np.floating], arrays: ArrayStore | None, owner: object
) -> dict[str, Any]:
    if arrays is None:
        raise TypeError(
            f"{type(owner).__name__} holds an array, which an inline record cannot carry; pass "
            "arrays= (a mutable mapping that receives it), or use DataArray.rf.to_dataset() or "
            "xarrayrf.native.value_to_dataset() to write a Dataset"
        )
    reference = array_reference(values)
    if reference not in arrays:
        arrays[reference] = values
    return {"reference": reference, "dtype": str(values.dtype)}


def array_reference(values: npt.NDArray[np.floating]) -> str:
    """Return ``"sha256:<hex>"`` of ``values``' dtype, shape and C-order, little-endian bytes.

    The header ``"float32:2,6,2\n"`` keeps arrays with equal bytes but another shape or dtype
    (zeros, say) from sharing a reference. The bytes are hashed in C-order blocks of at most
    ``_DIGEST_BLOCK_BYTES``, so a non-contiguous view (the transforms' storage is
    component-major) is never copied whole, even when one leading-axis row is larger.
    """
    header = f"{values.dtype.name}:{','.join(map(str, values.shape))}\n"
    digest = hashlib.sha256(header.encode("ascii"))
    _hash_blocks(digest, values, values.dtype.newbyteorder("<"))
    return _REFERENCE_PREFIX + digest.hexdigest()


def _hash_blocks(digest: Any, values: npt.NDArray[np.floating], little: np.dtype[Any]) -> None:
    """Feed ``values`` in C order, splitting leading axes until each block fits the bound.

    The last axis holds the components (two or three), so a 1-D block is always small.
    """
    if values.nbytes <= _DIGEST_BLOCK_BYTES or values.ndim == 1:
        digest.update(np.ascontiguousarray(values, dtype=little).data)
        return
    step = _DIGEST_BLOCK_BYTES // values[:1].nbytes
    if step == 0:
        for row in values:
            _hash_blocks(digest, row, little)
        return
    for start in range(0, len(values), step):
        digest.update(np.ascontiguousarray(values[start : start + step], dtype=little).data)


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


def decode_intervals(data: object) -> dict[str, Any]:
    """Check JSON interval mappings without coercing strings or booleans into numbers."""
    intervals = _mapping(data, "intervals")

    def check_rows(value: object) -> None:
        if not isinstance(value, list):
            raise MalformedDataError("interval rows must be JSON arrays")
        for item in value:
            if isinstance(item, list):
                check_rows(item)
            elif isinstance(item, bool) or not isinstance(item, int | float):
                raise MalformedDataError("interval values must be numbers")

    for name, rows in intervals.items():
        if not isinstance(name, str):
            raise MalformedDataError("interval axis names must be strings")
        check_rows(rows)
    # JSON drops the trailing dimension of (0, 2) rows; coordinate validation checks the shape.
    return {name: rows if rows else np.empty((0, 2)) for name, rows in intervals.items()}


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
    "east_north_up": {"base", "axes", "semi_major_axis", "inverse_flattening", "length_unit"},
    "spherical_to_cartesian": {"source", "target", "longitude_start"},
    "cartesian_to_spherical": {"source", "target", "longitude_start"},
    "geodetic_to_cartesian": {
        "source",
        "target",
        "semi_major_axis",
        "inverse_flattening",
        "longitude_start",
    },
    "cartesian_to_geodetic": {
        "source",
        "target",
        "semi_major_axis",
        "inverse_flattening",
        "longitude_start",
    },
    "grid": {"transform", "dims", "coordinates", "intervals"},
    "tensor_type": {
        "index_variance",
        "reflection_parity",
        "determinant_exponent",
        "index_symmetry",
        "requires_orthogonal_map",
    },
    "tensor_component_declaration": {"tensor_type", "component_dims", "bases"},
    "symmetric_packing": {"index_pairs", "off_diagonal_scale"},
    "packed_tensor_component_declaration": {"tensor_type", "component_dim", "basis", "packing"},
    "displacement_field_transform": {"source", "target", "grid", "outside", "field"},
    "position_field_transform": {"source", "target", "grid", "outside", "field"},
    "bspline_transform": {
        "source",
        "target",
        "grid",
        "order",
        "outside",
        "support",
        "coefficients",
    },
    "declared_inverse_pair": {"forward", "backward"},
    "paired_transform": {"pair", "direction"},
}

_TRANSITION_KINDS: Final[dict[type, str]] = {
    SphericalToCartesian: "spherical_to_cartesian",
    CartesianToSpherical: "cartesian_to_spherical",
    GeodeticToCartesian: "geodetic_to_cartesian",
    CartesianToGeodetic: "cartesian_to_geodetic",
}

_OPTIONAL_FIELDS: Final = {"coordinate_system": {"axis_roles"}}
"""Fields a record of that kind may omit: added within schema 1 and written only when used."""

_EXTENSION_FIELDS: Final = {"version", "source", "target", "data"}


class _Decoder:
    def __init__(self, decoders: dict[str, Decoder], arrays: Mapping[str, Any] | None) -> None:
        self._decoders = decoders
        self._arrays = arrays

    def value(self, data: object) -> Any:
        record = _mapping(data, "an encoded value")
        kind = record.get("kind")
        if not isinstance(kind, str):
            raise MalformedDataError(f"an encoded value needs a string 'kind', got {kind!r}")
        fields = {key: item for key, item in record.items() if key != "kind"}
        if kind in _BUILT_IN:
            self._expect(
                kind, fields, _FIELDS[kind] | (_OPTIONAL_FIELDS.get(kind, set()) & set(fields))
            )
            try:
                return getattr(self, kind)(fields)
            except EncodingError:
                raise
            except (TypeError, ValueError, OverflowError) as error:
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
            axis_roles=(
                self._list(fields["axis_roles"], "axis_roles") if "axis_roles" in fields else None
            ),
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
        return AffineTransform.from_matrix(
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
        return Grid(
            self.value(fields["transform"]),
            ordered,
            intervals=decode_intervals(fields["intervals"]),
        )

    def tensor_type(self, fields: Mapping[str, Any]) -> TensorType:
        exponent = fields["determinant_exponent"]
        if isinstance(exponent, bool) or not isinstance(exponent, int | float):
            raise MalformedDataError(f"determinant_exponent must be a number, got {exponent!r}")
        return TensorType(
            self._list(fields["index_variance"], "index_variance"),
            reflection_parity=fields["reflection_parity"],
            determinant_exponent=exponent,
            index_symmetry=fields["index_symmetry"],
            requires_orthogonal_map=fields["requires_orthogonal_map"],
        )

    def tensor_component_declaration(self, fields: Mapping[str, Any]) -> TensorComponentDeclaration:
        tensor_type = self.value(fields["tensor_type"])
        if not isinstance(tensor_type, TensorType):
            raise MalformedDataError("a declaration's tensor_type must be a tensor type")
        bases = [self._basis(item) for item in self._list(fields["bases"], "bases")]
        return TensorComponentDeclaration(
            tensor_type, self._list(fields["component_dims"], "component_dims"), bases
        )

    def symmetric_packing(self, fields: Mapping[str, Any]) -> SymmetricPacking:
        pairs = self._list(fields["index_pairs"], "index_pairs")
        for pair in pairs:
            self._list(pair, "index_pairs entries")
        scale = fields["off_diagonal_scale"]
        if isinstance(scale, bool) or not isinstance(scale, int | float):
            raise MalformedDataError(f"off_diagonal_scale must be a number, got {scale!r}")
        return SymmetricPacking([tuple(pair) for pair in pairs], off_diagonal_scale=scale)

    def packed_tensor_component_declaration(
        self, fields: Mapping[str, Any]
    ) -> TensorComponentDeclaration:
        tensor_type = self.value(fields["tensor_type"])
        if not isinstance(tensor_type, TensorType):
            raise MalformedDataError("a declaration's tensor_type must be a tensor type")
        packing = self.value(fields["packing"])
        if not isinstance(packing, SymmetricPacking):
            raise MalformedDataError("a packed declaration's packing must be a symmetric packing")
        dim = fields["component_dim"]
        if not isinstance(dim, str):
            raise MalformedDataError(f"component_dim must be a string, got {dim!r}")
        return TensorComponentDeclaration(
            tensor_type, (dim,), self._basis(fields["basis"]), packing=packing
        )

    def east_north_up(self, fields: Mapping[str, Any]) -> EastNorthUp:
        base = self.value(fields["base"])
        if not isinstance(base, ReferenceFrame):
            raise MalformedDataError("an east/north/up field's base must be a reference frame")
        return EastNorthUp(
            base,
            semi_major_axis=fields["semi_major_axis"],
            inverse_flattening=fields["inverse_flattening"],
            length_unit=fields["length_unit"],
            axes=tuple(self._list(fields["axes"], "axes")),
        )

    def _basis(self, data: object) -> ComponentBasis:
        basis = self.value(data)
        if not isinstance(basis, ReferenceFrame | ArrayCoordinates | EastNorthUp):
            raise MalformedDataError(
                "a component basis must be a reference frame, array coordinates or an "
                f"east/north/up field, got {type(basis).__name__}"
            )
        return basis

    def displacement_field_transform(self, fields: Mapping[str, Any]) -> DisplacementFieldTransform:
        source, target = self._frames(fields, "a field transform")
        return DisplacementFieldTransform(
            source=source,
            target=target,
            field=self._array(fields["field"], "field"),
            grid=self._grid(fields["grid"]),
            outside=fields["outside"],
        )

    def position_field_transform(self, fields: Mapping[str, Any]) -> PositionFieldTransform:
        source, target = self._frames(fields, "a field transform")
        return PositionFieldTransform(
            source=source,
            target=target,
            field=self._array(fields["field"], "field"),
            grid=self._grid(fields["grid"]),
            outside=fields["outside"],
        )

    def bspline_transform(self, fields: Mapping[str, Any]) -> BSplineTransform:
        source, target = self._frames(fields, "a field transform")
        return BSplineTransform(
            source=source,
            target=target,
            coefficients=self._array(fields["coefficients"], "coefficients"),
            grid=self._grid(fields["grid"]),
            order=fields["order"],
            outside=fields["outside"],
            support=fields["support"],
        )

    def declared_inverse_pair(self, fields: Mapping[str, Any]) -> DeclaredInversePair:
        return DeclaredInversePair(self.value(fields["forward"]), self.value(fields["backward"]))

    def paired_transform(self, fields: Mapping[str, Any]) -> PairedTransform:
        pair = self.value(fields["pair"])
        if not isinstance(pair, DeclaredInversePair):
            raise MalformedDataError("a paired transform's pair must be a declared inverse pair")
        direction = fields["direction"]
        if direction not in ("forward", "backward"):
            raise MalformedDataError(
                f"a paired transform's direction must be 'forward' or 'backward', got {direction!r}"
            )
        return pair.forward if direction == "forward" else pair.backward

    def _grid(self, data: object) -> Grid:
        grid = self.value(data)
        if not isinstance(grid, Grid):
            raise MalformedDataError(
                f"a field transform's grid must be a grid, got {type(grid).__name__}"
            )
        return grid

    def _array(self, data: object, field: str) -> npt.NDArray[np.floating]:
        """Resolve a reference, check its dtype and digest; the constructor checks the shape."""
        record = _mapping(data, f"{field} array reference")
        self._expect(f"{field} array reference", record, {"reference", "dtype"})
        reference, dtype = record["reference"], record["dtype"]
        digest = reference.removeprefix(_REFERENCE_PREFIX) if isinstance(reference, str) else ""
        if (
            not isinstance(reference, str)
            or not reference.startswith(_REFERENCE_PREFIX)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise MalformedDataError(
                f"{field} reference must be 'sha256:' and 64 lowercase hex digits, got {reference!r}"
            )
        if dtype not in ("float32", "float64"):
            raise MalformedDataError(f"{field} dtype must be float32 or float64, got {dtype!r}")
        if self._arrays is None:
            raise MissingArrayError(
                f"the {field} array {reference} is stored by reference; pass arrays= to resolve it"
            )
        try:
            entry = self._arrays[reference]
        except KeyError:
            raise MissingArrayError(
                f"arrays has no entry for the {field} array {reference}"
            ) from None
        if isinstance(entry, np.ma.MaskedArray):
            # np.asarray would drop the mask the constructor refuses.
            raise MalformedDataError(f"the {field} array {reference} is a masked array")
        loaded = np.asarray(entry)
        expected = np.dtype(dtype)
        # Either byte order: netCDF readers can return big-endian data. The constructor keeps
        # only native float32/float64, so convert rather than let it widen.
        if loaded.dtype.kind != "f" or loaded.dtype.itemsize != expected.itemsize:
            raise MalformedDataError(
                f"the {field} array {reference} has dtype {loaded.dtype}, recorded as {dtype}"
            )
        loaded = loaded.astype(expected, copy=False)
        if array_reference(loaded) != reference:
            raise MalformedDataError(
                f"the {field} array resolved for {reference} does not match its digest"
            )
        return loaded

    def composite_transform(self, fields: Mapping[str, Any]) -> CompositeTransform:
        members = [self.value(item) for item in self._list(fields["transforms"], "transforms")]
        return CompositeTransform(*members)

    # Constructors validate the frames and numbers; their TypeError and ValueError surface as
    # MalformedDataError through the dispatcher.
    def spherical_to_cartesian(self, fields: Mapping[str, Any]) -> SphericalToCartesian:
        return SphericalToCartesian(
            *self._frames(fields, "a chart transition"), longitude_start=fields["longitude_start"]
        )

    def cartesian_to_spherical(self, fields: Mapping[str, Any]) -> CartesianToSpherical:
        return CartesianToSpherical(
            *self._frames(fields, "a chart transition"), longitude_start=fields["longitude_start"]
        )

    def geodetic_to_cartesian(self, fields: Mapping[str, Any]) -> GeodeticToCartesian:
        return GeodeticToCartesian(
            *self._frames(fields, "a chart transition"), **self._figure(fields)
        )

    def cartesian_to_geodetic(self, fields: Mapping[str, Any]) -> CartesianToGeodetic:
        return CartesianToGeodetic(
            *self._frames(fields, "a chart transition"), **self._figure(fields)
        )

    def _frames(
        self, fields: Mapping[str, Any], what: str
    ) -> tuple[ReferenceFrame, ReferenceFrame]:
        source, target = self.value(fields["source"]), self.value(fields["target"])
        if not isinstance(source, ReferenceFrame) or not isinstance(target, ReferenceFrame):
            raise MalformedDataError(f"{what}'s endpoints must be reference frames")
        return source, target

    @staticmethod
    def _figure(fields: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "semi_major_axis": fields["semi_major_axis"],
            "inverse_flattening": fields["inverse_flattening"],
            "longitude_start": fields["longitude_start"],
        }

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
