"""Canonical unit spellings shared by the format adapters.

The core compares unit strings exactly and never converts them. Adapters therefore have to
agree on one spelling for each unit they emit, or frames read from different formats never
compare equal. This table maps the UDUNITS-2 names OME-NGFF prefers (``"micrometer"``) and
their plurals to the CF symbols the DICOM, NIfTI and GeoTIFF adapters emit (``"um"``), and
back. Spellings outside the table pass through unchanged.
"""

from __future__ import annotations

from types import MappingProxyType

_SYMBOLS: dict[str, str] = {
    "meter": "m",
    "metre": "m",
    "millimeter": "mm",
    "millimetre": "mm",
    "micrometer": "um",
    "micrometre": "um",
    "micron": "um",
    "nanometer": "nm",
    "nanometre": "nm",
    "kilometer": "km",
    "kilometre": "km",
    "centimeter": "cm",
    "centimetre": "cm",
    "foot": "ft",
    "second": "s",
    "millisecond": "ms",
    "microsecond": "us",
    "nanosecond": "ns",
    "minute": "min",
    "hour": "h",
    "hertz": "Hz",
    "kilohertz": "kHz",
    "megahertz": "MHz",
    "radian": "rad",
    "degree": "degree",
}
_SYMBOLS.update({f"{name}s": symbol for name, symbol in list(_SYMBOLS.items())})
del _SYMBOLS["foots"]
_SYMBOLS.update(
    {"feet": "ft", "US survey foot": "US_survey_foot", "US survey feet": "US_survey_foot"}
)

# The UDUNITS-2 name OME-NGFF prefers for each symbol; the first spelling listed above wins.
_NAMES: dict[str, str] = {}
for _name, _symbol in _SYMBOLS.items():
    _NAMES.setdefault(_symbol, _name)

SYMBOLS = MappingProxyType(_SYMBOLS)
"""Read-only view of the spelling table: UDUNITS-2 names and plurals to CF symbols."""


def canonical(unit: str | None) -> str | None:
    """Return the CF symbol for a UDUNITS-2 unit name, or the input when it is not in the table.

    Args:
        unit: A unit string, or ``None`` for an axis without a declared unit.

    Returns:
        The canonical spelling adapters emit, so ``"micrometer"`` becomes ``"um"``; ``None``
        stays ``None`` and unknown spellings stay as given.

    Raises:
        TypeError: If ``unit`` is neither a string nor ``None``.
    """
    if unit is None:
        return None
    if not isinstance(unit, str):
        raise TypeError(f"unit must be a string or None, got {type(unit).__name__}")
    return _SYMBOLS.get(unit, unit)


def udunits_name(unit: str | None) -> str | None:
    """Return the UDUNITS-2 name for a canonical symbol, or the input when it is not in the table.

    The inverse of :func:`canonical` for export to formats that prefer long names (OME-NGFF):
    ``"um"`` becomes ``"micrometer"``.

    Args:
        unit: A unit string, or ``None``.

    Raises:
        TypeError: If ``unit`` is neither a string nor ``None``.
    """
    if unit is None:
        return None
    if not isinstance(unit, str):
        raise TypeError(f"unit must be a string or None, got {type(unit).__name__}")
    return _NAMES.get(unit, unit)
