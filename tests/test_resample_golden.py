"""Recorded outputs of :func:`xarrayrf.resample`, guarding restructuring.

The fixture ``tests/fixtures/resample_golden.npz`` holds the results the implementation
produced on 2026-09-27 for every scenario below. Changes beyond floating-point rounding are
behaviour changes and must be deliberate: regenerate with
``.venv/bin/python tests/test_resample_golden.py`` after confirming the new numbers are right,
and say so in the changelog.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Protocol, cast

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from test_resample import LPS, RAS, Shift, ramp, volume

import xarrayrf as xrf

pytest.importorskip("scipy", minversion="1.18")

FIXTURE = Path(__file__).parent / "fixtures" / "resample_golden.npz"
# Linux/macOS drift reached 2.13e-14 absolute and 1.36e-15 relative on these float64 fixtures.
FLOAT_RTOL = 5e-15
FLOAT_ATOL = 5e-14


class _SaveArrays(Protocol):
    def __call__(self, file: Path, **arrays: npt.ArrayLike) -> None: ...


def _rotation(degrees: float) -> npt.NDArray[np.float64]:
    angle = np.deg2rad(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )


def _offset_source() -> xrf.Geometry:
    data = ramp((5, 6, 7))
    array = xr.DataArray(
        data,
        dims=("k", "j", "i"),
        coords={"k": np.arange(5)[::-1].astype(float), "j": np.arange(6), "i": np.arange(7)},
    )
    coordinates = xrf.ArrayCoordinates(("i", "j", "k"), ("1", "1", "1"), sample_offset=(0, 0.25, 1))
    return xrf.Geometry(
        array,
        xrf.AffineTransform.from_matrix(
            source=coordinates, target=LPS, matrix=_rotation(20.0), translation=(0, 0, 0)
        ),
        dims=("k", "j", "i"),
    )


def _nonuniform_stack() -> xrf.Geometry:
    values = np.stack([np.full((2, 2), value) for value in (10.0, 20.0, 50.0)])
    array = xr.DataArray(
        values,
        dims=("slice", "row", "column"),
        coords={"slice_offset": ("slice", [0.0, 2.0, 5.0]), "row": [0, 1], "column": [0, 1]},
    )
    return xrf.Geometry(
        array,
        xrf.AffineTransform.from_matrix(
            source=xrf.ArrayCoordinates(("column", "row", "slice_offset"), ("1", "1", "mm")),
            target=LPS,
            matrix=np.eye(3),
            translation=(0.0, 0.0, 0.0),
        ),
        dims=("slice", "row", "column"),
    )


def scenarios() -> Iterator[tuple[str, Callable[[], xr.DataArray]]]:
    """Name and a thunk producing the resampled array, one per code path worth guarding."""
    rng = np.random.default_rng(20260927)
    noisy = rng.random((4, 5, 6))
    source = volume(noisy)
    rotated_target = volume(
        np.zeros((5, 6, 7)), matrix=_rotation(10.0), translation=(0.5, -0.25, 0.75)
    )
    shifted_target = volume(np.zeros((4, 5, 6)), translation=(1.0, 1.0, 1.0))
    offset_target = volume(
        np.zeros((9, 11, 12)), translation=(-3.0, -1.5, -1.8), matrix=np.eye(3) * 0.8
    )
    general = Shift(LPS, LPS, 0.0)
    for method in ("nearest", "linear", "cubic"):
        yield f"lattice-{method}", partial(xrf.resample, source, rotated_target, method=method)
        yield (
            f"general-{method}",
            partial(xrf.resample, source, rotated_target, transform=general, method=method),
        )
        yield (
            f"same-grid-shift-{method}",
            partial(xrf.resample, source, shifted_target, method=method),
        )
        yield (
            f"cells-lattice-{method}",
            partial(xrf.resample, _offset_source(), offset_target, method=method, domain="cells"),
        )
        yield (
            f"cells-general-{method}",
            partial(
                xrf.resample,
                _offset_source(),
                offset_target,
                transform=general,
                method=method,
                domain="cells",
            ),
        )
    yield (
        "fill-value-linear",
        lambda: xrf.resample(source, shifted_target, method="linear", fill_value=-1.0),
    )
    yield (
        "ras-target",
        lambda: xrf.resample(
            source, volume(np.zeros((4, 5, 6)), frame=RAS, matrix=np.diag([-1.0, -1.0, 1.0]))
        ),
    )
    yield (
        "nonuniform-slices",
        lambda: xrf.resample(
            _nonuniform_stack(), volume(np.zeros((1, 2, 2)), translation=(0.0, 0.0, 3.5))
        ),
    )
    yield (
        "echo-dimension-linear",
        lambda: xrf.resample(volume(np.stack([noisy, 2.0 * noisy])), rotated_target),
    )
    chunked = da.from_array(np.stack([noisy, 2.0 * noisy]), chunks=(1, 4, 5, 6))  # type: ignore[no-untyped-call]
    yield (
        "dask-echo-cubic",
        lambda: xrf.resample(volume(chunked), rotated_target, method="cubic").compute(),
    )
    yield (
        "cropped-window-linear",
        lambda: xrf.resample(
            volume(rng.random((12, 12, 12))),
            volume(np.zeros((2, 2, 2)), translation=(5.2, 6.1, 7.3), matrix=np.eye(3) * 0.5),
        ),
    )
    yield (
        "small-blocks-general-cubic",
        lambda: xrf.resample(
            source, rotated_target, transform=Shift(LPS, LPS, 0.0), method="cubic", block_points=7
        ),
    )
    yield (
        "integer-nearest",
        lambda: xrf.resample(
            volume(rng.integers(0, 100, (4, 5, 6))), shifted_target, method="nearest", fill_value=-1
        ),
    )
    yield (
        "complex-linear",
        lambda: xrf.resample(volume(noisy + 1j * noisy[::-1]), rotated_target, method="linear"),
    )


def _record() -> dict[str, npt.ArrayLike]:
    return {name: np.asarray(thunk().values) for name, thunk in scenarios()}


@pytest.mark.parametrize(
    ("name", "thunk"), list(scenarios()), ids=lambda item: item if isinstance(item, str) else ""
)
def test_resample_matches_recorded_output(name: str, thunk: Callable[[], xr.DataArray]) -> None:
    golden = np.load(FIXTURE)
    assert name in golden.files, f"{name} missing from the fixture; regenerate it deliberately"
    actual = np.asarray(thunk().values)
    expected = golden[name]
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if np.issubdtype(expected.dtype, np.inexact):
        assert_array_equal(np.isnan(actual.real), np.isnan(expected.real))
        assert_array_equal(np.isnan(actual.imag), np.isnan(expected.imag))
        assert_allclose(actual, expected, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
    else:
        assert_array_equal(actual, expected)


if __name__ == "__main__":
    # New NumPy stubs reserve allow_pickle when checking arbitrary **array names.
    save_arrays = cast(_SaveArrays, np.savez)
    save_arrays(FIXTURE, **_record())
    print(f"wrote {len(list(scenarios()))} scenarios to {FIXTURE}")
