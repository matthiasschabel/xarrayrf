"""Canonical unit spellings shared by the adapters."""

from __future__ import annotations

import pytest

from xarrayrf import CoordinateSystem, units


@pytest.mark.parametrize(
    ("name", "symbol"),
    [
        ("micrometer", "um"),
        ("micrometers", "um"),
        ("millimetre", "mm"),
        ("second", "s"),
        ("hertz", "Hz"),
        ("feet", "ft"),
        ("US survey foot", "US_survey_foot"),
    ],
)
def test_udunits_names_map_to_cf_symbols_and_back(name: str, symbol: str) -> None:
    assert units.canonical(name) == symbol
    assert units.canonical(symbol) == symbol
    assert units.canonical(units.udunits_name(symbol)) == symbol


def test_unknown_spellings_and_none_pass_through() -> None:
    assert units.canonical("furlong") == "furlong"
    assert units.canonical("foots") == "foots"
    assert units.udunits_name("furlong") == "furlong"
    assert units.canonical(None) is None
    assert units.udunits_name(None) is None
    with pytest.raises(TypeError, match="unit must be a string"):
        units.canonical(3)  # type: ignore[arg-type]


def test_canonical_spellings_make_systems_comparable() -> None:
    ngff_style = CoordinateSystem(
        ("y", "x"), tuple(units.canonical(u) for u in ("micrometer", "micrometer"))
    )
    dicom_style = CoordinateSystem(("y", "x"), ("um", "um"))
    assert ngff_style == dicom_style
