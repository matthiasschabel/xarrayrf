"""Shared validation for the immutable reference-frame value objects.

Units are open strings following the CF/UDUNITS convention used by xarray ``attrs["units"]``
and OME-NGFF. They are compared exactly and never parsed or converted here; adapters
normalize spellings and perform explicit conversions before construction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt


def _cf_geographic_degrees() -> frozenset[str]:
    """CF/UDUNITS spellings of geographic degrees, such as ``degrees_east`` or ``degree_N``."""
    spellings = set()
    for stem in ("degree", "degrees"):
        for full, letter in (("east", "E"), ("west", "W"), ("north", "N"), ("south", "S")):
            spellings.update({f"{stem}_{full}", f"{stem}_{letter}", f"{stem}{letter}"})
    return frozenset(spellings)


ANGULAR_UNITS: Final = frozenset(
    {
        "rad",
        "radian",
        "radians",
        "deg",
        "degree",
        "degrees",
        "arc_degree",
        "angular_degree",
        "\N{DEGREE SIGN}",
        "arcmin",
        "arcminute",
        "arcminutes",
        "arc_minute",
        "angular_minute",
        "arcsec",
        "arcsecond",
        "arcseconds",
        "arc_second",
        "angular_second",
    }
    | _cf_geographic_degrees(),
)
"""Known angular spellings a Cartesian axis refuses, including CF geographic degrees.

Best effort: with open unit strings only these spellings are recognized as angular.
"""

REAL_KINDS: Final = frozenset({"i", "u", "f"})
"""NumPy dtype kinds accepted as coordinate values: signed, unsigned and floating."""

type MetadataValue = str | int | float | bool | None
type MetadataItem = tuple[str, str, MetadataValue]
type MetadataItems = tuple[MetadataItem, ...]


def check_str(value: object, *, field: str) -> str:
    """Validate a non-empty string.

    Args:
        value: Candidate value.
        field: Field name used in error messages.

    Returns:
        The validated string.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If ``value`` is empty.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must not be empty")
    return value


def check_optional_str(value: object, *, field: str) -> str | None:
    """Validate an optional non-empty string.

    Args:
        value: Candidate value or ``None``.
        field: Field name used in error messages.

    Returns:
        The validated string, or ``None``.

    Raises:
        TypeError: If ``value`` is neither ``None`` nor a string.
        ValueError: If ``value`` is an empty string.
    """
    if value is None:
        return None
    return check_str(value, field=field)


def _check_str_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            f"{field} must be a sequence of strings, got {type(value).__name__}",
        )
    entries = tuple(value)
    for entry in entries:
        if not isinstance(entry, str):
            raise TypeError(f"{field} entries must be strings, got {type(entry).__name__}")
        if not entry:
            raise ValueError(f"{field} entries must not be empty strings")
    return entries


def check_names(value: object, *, field: str, allow_empty: bool = False) -> tuple[str, ...]:
    """Validate an ordered sequence of unique non-empty names.

    Args:
        value: Candidate sequence of names.
        field: Field name used in error messages.
        allow_empty: Accept an empty sequence, for a fully selected point with no varying
            geometry dimension left.

    Returns:
        The names as a tuple, in the given order.

    Raises:
        TypeError: If ``value`` is not a non-string sequence of strings.
        ValueError: If the sequence is empty (unless allowed) or repeats a name.
    """
    names = _check_str_sequence(value, field=field)
    if not names and not allow_empty:
        raise ValueError(f"{field} must declare at least one name")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"{field} names must be unique; repeated {tuple(repeated)}")
    return names


def _check_optional_str_sequence(value: object, *, field: str) -> tuple[str | None, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            f"{field} must be a sequence of strings or None, got {type(value).__name__}",
        )
    entries = tuple(value)
    for entry in entries:
        if entry is None:
            continue
        if not isinstance(entry, str):
            raise TypeError(f"{field} entries must be strings or None, got {type(entry).__name__}")
        if not entry:
            raise ValueError(
                f"{field} entries must not be empty strings; use None for an undeclared entry",
            )
        if entry != entry.strip():
            raise ValueError(f"{field} entry {entry!r} has leading or trailing whitespace")
    return entries


def check_units(
    value: object, *, field: str, reject_angular: bool = False
) -> tuple[str | None, ...]:
    """Validate an ordered sequence of unit strings, ``None`` where no unit is declared.

    Units are compared exactly, so ``"mm"`` and ``"millimeter"`` differ; adapters choose one
    spelling. Units need not be unique: several axes legitimately share one. ``None`` declares
    no unit and says nothing more: it is not ``"1"`` (dimensionless), and no kind of axis is
    inferred from it.

    Args:
        value: Candidate sequence of unit strings or ``None``.
        field: Field name used in error messages.
        reject_angular: Refuse the known angular spellings in :data:`ANGULAR_UNITS`. A
            Cartesian axis declares no periodicity or branch, so an angle there would be
            relabelled rather than represented.

    Returns:
        The units as a tuple, in the given order.

    Raises:
        TypeError: If ``value`` is not a non-string sequence of strings and ``None``.
        ValueError: If the sequence is empty, a unit is empty or has leading or trailing
            whitespace, or an angular unit is given when ``reject_angular`` is set.
    """
    units = _check_optional_str_sequence(value, field=field)
    if not units:
        raise ValueError(f"{field} must declare at least one unit")
    for unit in units:
        if reject_angular and unit in ANGULAR_UNITS:
            raise ValueError(
                f"{field} unit {unit!r} is angular; a Cartesian axis declares no periodicity "
                "or branch, so an angle would be relabelled rather than represented",
            )
    return units


def check_axis_types(value: object, *, count: int) -> tuple[str | None, ...]:
    """Validate one optional axis type per axis: an open string, or ``None`` when undeclared.

    Raises:
        TypeError: If ``value`` is not ``None`` or a non-string sequence of strings and ``None``.
        ValueError: If the count differs, or a type is empty or padded.
    """
    if value is None:
        return (None,) * count
    types = _check_optional_str_sequence(value, field="axis_types")
    if len(types) != count:
        raise ValueError(
            f"axis_types must declare one type per axis; got {len(types)} for {count} axes",
        )
    return types


def check_identifier(value: object, *, field: str = "identifier") -> tuple[str, str]:
    """Validate a ``(namespace, value)`` reference-frame identifier.

    Args:
        value: Candidate identifier.
        field: Field name used in error messages.

    Returns:
        The validated identifier.

    Raises:
        TypeError: If ``value`` is not a tuple of two strings.
        ValueError: If the tuple has the wrong length or holds an empty string.
    """
    if not isinstance(value, tuple):
        raise TypeError(
            f"{field} must be a (namespace, value) tuple, got {type(value).__name__}",
        )
    if len(value) != 2:
        raise ValueError(
            f"{field} must be a (namespace, value) tuple, got {len(value)} entries",
        )
    namespace = check_str(value[0], field=f"{field} namespace")
    name = check_str(value[1], field=f"{field} value")
    return (namespace, name)


def _scalar_kind(entry: object, *, field: str, name: str) -> str:
    """Return the type tag a metadata value is compared under.

    ``bool`` is checked before ``int`` because it is a subclass of it.
    """
    if entry is None:
        return "none"
    if isinstance(entry, bool):
        return "bool"
    if isinstance(entry, str):
        return "str"
    if isinstance(entry, int):
        return "int"
    if isinstance(entry, float):
        return "float"
    raise TypeError(
        f"{field}[{name!r}] must be a string, integer, boolean, float or None, "
        f"got {type(entry).__name__}",
    )


def canonical_metadata(value: object, *, field: str) -> MetadataItems:
    """Canonicalize a flat, data-only metadata mapping.

    The result is sorted by key so that two declarations written in different key orders
    compare and hash identically. Values are restricted to exactly comparable scalars:
    the core stores declarations, never live provider objects or executable payloads.

    Each value carries a type tag, so comparison is by type *and* value. Python's numeric
    tower would otherwise make ``True`` equal ``1`` and ``1`` equal ``1.0``, which would
    let a boolean context entry silently satisfy a numeric one and hide a conflict two
    adapters must resolve. Distinguishing them is conservative in the only direction that
    is safe here: it reports a conflict rather than concealing one, and a caller that
    wants two spellings to agree normalizes them explicitly before construction.

    Args:
        value: Candidate mapping, or ``None`` for absent metadata.
        field: Field name used in error messages.

    Returns:
        Key-sorted ``(key, type tag, value)`` triples; empty when ``value`` is ``None``.

    Raises:
        TypeError: If ``value`` is not a mapping, or holds a non-string key or a value
            that is not a string, integer, boolean, float or ``None``.
        ValueError: If a key is empty or a float value is not finite.
    """
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping or None, got {type(value).__name__}")
    items: list[MetadataItem] = []
    for key, entry in value.items():
        name = check_str(key, field=f"{field} key")
        kind = _scalar_kind(entry, field=field, name=name)
        if kind == "float" and not math.isfinite(entry):
            raise ValueError(f"{field}[{name!r}] must be finite, got {entry!r}")
        items.append((name, kind, entry))
    return tuple(sorted(items, key=lambda item: item[0]))


def metadata_view(items: MetadataItems) -> Mapping[str, MetadataValue]:
    """Return a read-only mapping view of canonical metadata items.

    The type tags are an internal comparison detail; the view returns the values as they
    were declared.

    Args:
        items: Key-sorted metadata triples.

    Returns:
        A read-only mapping; the underlying dictionary is not reachable by the caller.
    """
    return MappingProxyType({name: entry for name, _, entry in items})


def real_float_array(value: npt.ArrayLike, *, field: str) -> npt.NDArray[np.float64]:
    """Read ``value`` as finite float64 without coercing a non-real dtype.

    NumPy's own conversion parses numeric strings and drops the imaginary part of a
    complex array with a warning, both of which would change the caller's numbers on the
    way in. The dtype is therefore checked before the conversion, not after it.

    The contract is on the dtype NumPy infers, not on the Python objects supplied. A
    mixed sequence such as ``[True, 2.0]`` is promoted to ``float64`` by ordinary NumPy
    inference and is accepted on that dtype; only an input whose *inferred* dtype is
    boolean is refused. Nothing walks a container looking for individual elements.

    A masked array is refused before the conversion, because ``np.asarray`` drops the
    mask without warning and would read each masked entry as whatever sits underneath it.

    Args:
        value: Candidate numeric values.
        field: Field name used in error messages.

    Returns:
        A finite float64 array; a new array whenever conversion was needed.

    Raises:
        TypeError: If ``value`` is a masked array or its inferred dtype is not real integer
            or floating.
        ValueError: If ``value`` is not a rectangular numeric array or is not finite.
    """
    if isinstance(value, np.ma.MaskedArray):
        raise TypeError(
            f"{field} must not be a masked array; converting one discards the mask "
            "silently and reads masked entries as their underlying data. Resolve the mask "
            "explicitly before calling, for example with filled() or compressed()",
        )
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field} must be convertible to finite float values; NumPy could not read it "
            "as a rectangular numeric array",
        ) from error
    if array.dtype.kind not in REAL_KINDS:
        raise TypeError(
            f"{field} must hold real integer or floating values, got dtype {array.dtype}; "
            "numeric strings are not parsed, the real part of a complex value is not taken, "
            "and a boolean dtype is not read as 0 and 1; convert explicitly before calling",
        )
    converted = np.asarray(array, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise ValueError(f"{field} must contain only finite values")
    return converted


def frozen_float_array(value: npt.ArrayLike, *, field: str) -> npt.NDArray[np.float64]:
    """Store ``value`` as a finite float array backed by an immutable buffer.

    A read-only flag alone does not make a published array safe: an array that owns its
    memory can have ``flags.writeable`` set back to ``True``, and the transform's coefficients
    (and its hash) would change underneath it. Backing the storage with ``bytes`` moves
    the guarantee from a flag to ownership, so neither the returned array nor anything
    reachable through its base chain can be made writeable.
    """
    array = real_float_array(value, field=field)
    # Normalize negative zero so exact structural equality stays consistent with hashing,
    # which compares the raw buffer. Adding +0.0 is exact for every other finite value,
    # and produces a new array rather than writing into the caller's input.
    normalized = array + 0.0
    frozen = np.frombuffer(normalized.tobytes(), dtype=np.float64)
    return frozen.reshape(normalized.shape)
