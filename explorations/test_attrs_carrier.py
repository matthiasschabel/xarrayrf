"""Measured behaviour of the provisional attrs carrier under xarray 2026.7.0 operations.

Each case records what happens to the carried declaration, checked against sample locations
computed independently from the original geometry. Outcomes:

- ``correct``: the declaration survives and locates the result's samples correctly;
- ``refuses``: it survives but no longer describes the array, and ``Geometry`` refuses it;
- ``lost``: it is dropped, so the result is silently unframed;
- ``unsafe``: it survives on a result it does not truthfully describe (mixed frames, or
  coordinates the caller changed), and nothing notices.

These pin measurements; they are not a contract. ``unsafe`` and ``lost`` on a cross-frame
combination are the failures a real binding must turn into refusals.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import attrs_carrier  # noqa: F401  # registers the accessor
import numpy as np
import pytest
import synthetic
import xarray as xr

ARRAY, GEOMETRY = synthetic.volume()
FRAMED = ARRAY.rf_probe.attach(GEOMETRY.transform, GEOMETRY.dims)
SAME = synthetic.volume(frame=GEOMETRY.frame)
SAME_FRAMED = SAME[0].rf_probe.attach(SAME[1].transform, SAME[1].dims)
OTHER = synthetic.volume()
OTHER_FRAMED = OTHER[0].rf_probe.attach(OTHER[1].transform, OTHER[1].dims)
POINTS = GEOMETRY.points()


def outcome(result: object, expected: xr.DataArray | None) -> str:
    if not isinstance(result, xr.DataArray) or not result.rf_probe.attached:
        return "lost"
    try:
        points = result.rf_probe.geometry().points()
    except ValueError:
        return "refuses"
    if expected is None:
        return "unsafe"
    points = points.transpose(*expected.dims)
    if points.shape == expected.shape and np.allclose(points.values, expected.values):
        return "correct"
    return "unsafe"


def netcdf(array: xr.DataArray, directory: Path) -> xr.DataArray:
    path = os.fspath(directory / "framed.nc")
    array.to_netcdf(path)
    with xr.open_dataarray(path) as opened:
        return opened.load()


CASES: list[tuple[str, Callable[[], object], xr.DataArray | None, str]] = [
    ("isel slice", lambda: FRAMED.isel(k=slice(1, 5)), POINTS.isel(k=slice(1, 5)), "correct"),
    (
        "isel stride",
        lambda: FRAMED.isel(j=slice(None, None, 3)),
        POINTS.isel(j=slice(None, None, 3)),
        "correct",
    ),
    ("isel gather", lambda: FRAMED.isel(i=[5, 0, 7]), POINTS.isel(i=[5, 0, 7]), "correct"),
    ("sel slice", lambda: FRAMED.sel(k=slice(1, 4)), POINTS.sel(k=slice(1, 4)), "correct"),
    ("transpose", lambda: FRAMED.transpose("i", "k", "j"), POINTS, "correct"),
    ("scalar arithmetic", lambda: FRAMED * 2, POINTS, "correct"),
    ("same frame a + b", lambda: FRAMED + SAME_FRAMED, POINTS, "correct"),
    ("framed + unframed", lambda: FRAMED + ARRAY, POINTS, "correct"),
    ("unframed + framed", lambda: ARRAY + FRAMED, POINTS, "correct"),
    (
        "concat along k",
        lambda: xr.concat([FRAMED.isel(k=slice(0, 4)), FRAMED.isel(k=slice(4, 8))], "k"),
        POINTS,
        "correct",
    ),
    ("dataset extraction", lambda: FRAMED.to_dataset()["signal"], POINTS, "correct"),
    ("coarsen mean", lambda: FRAMED.coarsen(i=2).mean(), POINTS.coarsen(i=2).mean(), "correct"),
    # The declaration names a dimension or coordinate the result no longer has.
    ("integer isel", lambda: FRAMED.isel(k=2), None, "refuses"),
    ("integer isel drop", lambda: FRAMED.isel(k=2, drop=True), None, "refuses"),
    ("rename", lambda: FRAMED.rename(k="s"), None, "refuses"),
    ("mean over k", lambda: FRAMED.mean("k"), None, "refuses"),
    # Cross-frame combination: dropped silently, or kept from the first operand.
    ("different frames a + c", lambda: FRAMED + OTHER_FRAMED, None, "lost"),
    ("different frames c + a", lambda: OTHER_FRAMED + FRAMED, None, "lost"),
    ("np.add different frames", lambda: np.add(FRAMED, OTHER_FRAMED), None, "unsafe"),
    (
        "xr.where different frames",
        lambda: xr.where(FRAMED > 0, FRAMED, OTHER_FRAMED),
        None,
        "unsafe",
    ),
    ("concat different frames", lambda: xr.concat([FRAMED, OTHER_FRAMED], "stack"), None, "unsafe"),
    # Coordinates reassigned by the caller: the declaration now locates other points.
    ("reassigned coordinates", lambda: FRAMED.assign_coords(i=FRAMED.i + 1), None, "unsafe"),
]


@pytest.mark.parametrize(
    ("name", "operation", "expected", "measured"), CASES, ids=[c[0] for c in CASES]
)
def test_attrs_carrier_outcome(
    name: str, operation: Callable[[], object], expected: xr.DataArray | None, measured: str
) -> None:
    assert outcome(operation(), expected) == measured


def test_netcdf_round_trip_keeps_the_declaration_as_metadata_only(tmp_path: Path) -> None:
    """The data-only declaration survives; nothing is enforced on the reopened array."""
    pytest.importorskip("scipy")
    reopened = netcdf(FRAMED, tmp_path)
    assert outcome(reopened, POINTS) == "correct"
    assert outcome(reopened + OTHER_FRAMED, None) == "lost"
