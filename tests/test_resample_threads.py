"""Threaded resampling: results, error order, eligibility and the Dask default."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from test_resample_fields import FIXED, MOVING, ROTATION, Opaque, case, field, geometry

import xarrayrf as xrf
import xarrayrf._resampling as resampling

pytest.importorskip("scipy", minversion="1.18")

LATTICE_ATOL = 1e-12
"""A slab's offset is reassociated (offset + linear[:, 0] * row); values are of order one."""


@pytest.fixture
def no_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if resampling starts a thread pool."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("expected inline execution")

    monkeypatch.setattr(resampling, "ThreadPoolExecutor", refuse)


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize("path", ["fused", "general", "lattice"])
def test_threads_give_the_inline_result(method: xrf.Method, path: str) -> None:
    source, target = case(0.0, None)
    if path == "lattice":
        transform: Any = xrf.AffineTransform.from_matrix(
            source=FIXED, target=MOVING, matrix=ROTATION, translation=(0.5, 0.0, 0.0)
        )
    else:
        transform = field("dense", FIXED, MOVING, np.zeros(3), "identity")
    if path == "general":
        # Nonuniform source coordinates leave only the general path, with built-in transforms.
        array = source.array.assign_coords(i=np.cumsum(np.linspace(1.0, 1.5, 11)))
        source = xrf.Geometry(array, source.transform, dims=source.dims)
    inline = xrf.resample(source, target, transform=transform, method=method, workers=1).values
    threaded = xrf.resample(source, target, transform=transform, method=method, workers=4).values
    assert_array_equal(np.isnan(inline), np.isnan(threaded))
    atol = LATTICE_ATOL if path == "lattice" else 0.0
    assert_allclose(threaded, inline, rtol=0, atol=atol, equal_nan=True)


def test_threads_with_missing_values_and_several_slices() -> None:
    source, target = case(0.0, None)
    values = np.stack([source.array.values, 2 * source.array.values])
    values[0, 4, 5, 6] = np.nan
    stacked = xrf.Geometry(
        xr.DataArray(values, dims=("echo", *source.array.dims), coords=source.array.coords),
        source.transform,
        dims=source.dims,
    )
    transform = field("dense", FIXED, MOVING, np.zeros(3), "identity")
    inline = xrf.resample(stacked, target, transform=transform, workers=1).values
    threaded = xrf.resample(stacked, target, transform=transform, workers=3).values
    assert_array_equal(threaded, inline)


def test_the_first_failing_block_is_reported_whatever_the_threads() -> None:
    source, _ = case(0.0, None)
    target = geometry(np.zeros((8, 9, 10)), FIXED, 1.15 * ROTATION, (1.0, 1.0, 1.0))
    transform = field("dense", FIXED, MOVING, np.zeros(3), "raise")
    messages = []
    for workers in (1, 4, 8):
        with pytest.raises(ValueError, match="outside the displacement field") as error:
            xrf.resample(source, target, transform=transform, block_points=7, workers=workers)
        messages.append(str(error.value))
    assert messages[1:] == messages[:1] * 2


@pytest.mark.usefixtures("no_threads")
def test_a_foreign_transform_runs_inline() -> None:
    source, target = case(0.0, None)
    transform = Opaque(field("dense", FIXED, MOVING, np.zeros(3), "identity"))
    xrf.resample(source, target, transform=transform, workers=4)


def test_a_dask_source_defaults_to_one_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    # Automatic threads would start for this small target on a four-core machine; a Dask
    # source must still run inline.
    monkeypatch.setattr(resampling, "MIN_THREAD_POINTS", 1)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    started: list[int] = []

    def spy(*, max_workers: int) -> ThreadPoolExecutor:
        started.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(resampling, "ThreadPoolExecutor", spy)
    source, target = case(0.0, None)
    transform = field("dense", FIXED, MOVING, np.zeros(3), "identity")
    xrf.resample(source, target, transform=transform)
    assert started == [4]
    lazy = xrf.Geometry(source.array.chunk(), source.transform, dims=source.dims)
    xrf.resample(lazy, target, transform=transform).compute()
    assert started == [4]


@pytest.mark.usefixtures("no_threads")
def test_one_worker_runs_inline() -> None:
    source, target = case(0.0, None)
    transform = field("dense", FIXED, MOVING, np.zeros(3), "identity")
    xrf.resample(source, target, transform=transform, workers=1)


def test_threads_are_used_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    source, target = case(0.0, None)
    transform = field("dense", FIXED, MOVING, np.zeros(3), "identity")
    started: list[int] = []

    def spy(*, max_workers: int) -> ThreadPoolExecutor:
        started.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(resampling, "ThreadPoolExecutor", spy)
    xrf.resample(source, target, transform=transform, workers=3)
    assert started and max(started) == 3


@pytest.mark.parametrize(
    "workers, error, match",
    [
        (0, ValueError, "positive integer"),
        (-2, ValueError, "positive integer"),
        (True, TypeError, "positive integer"),
        (2.0, TypeError, "positive integer"),
    ],
)
def test_workers_must_be_a_positive_integer(
    workers: Any, error: type[Exception], match: str
) -> None:
    source, target = case(0.0, None)
    with pytest.raises(error, match=match):
        xrf.resample(
            source,
            target,
            transform=field("dense", FIXED, MOVING, np.zeros(3), "identity"),
            workers=workers,
        )


def test_submission_is_bounded_and_the_lowest_failure_wins() -> None:
    import threading
    import time
    from types import SimpleNamespace

    submitted: list[int] = []
    gate = threading.Event()

    def task(item: int) -> None:
        submitted.append(item)
        if item == 1:
            # The lower item fails last, after a higher one has already failed.
            gate.wait(1.0)
            raise ValueError("item 1")
        if item == 2:
            gate.set()
            raise ValueError("item 2")
        time.sleep(0.001)

    plan: Any = SimpleNamespace(workers=2)
    with pytest.raises(ValueError, match="item 1"):
        resampling._run(plan, task, range(10_000))
    # Two items per worker in flight: nothing near the 10 000 is ever queued.
    assert max(submitted) < 16


@pytest.mark.parametrize("kind", ["dense", "bspline"])
def test_threaded_fields_of_both_kinds(kind: str) -> None:
    source, target = case(0.0, None)
    transform = field(kind, FIXED, MOVING, np.zeros(3), "nearest")
    inline = xrf.resample(source, target, transform=transform, method="cubic", workers=1)
    threaded = xrf.resample(source, target, transform=transform, method="cubic", workers=4)
    assert_array_equal(threaded.values, inline.values)


def test_threaded_affine_slabs_with_complex_missing_values() -> None:
    source, target = case(0.0, None)
    values = source.array.values.astype(np.complex128) * (1 + 1j)
    values[3, 4, 5] = complex(np.nan, 1.0)
    values[5, 2, 7] = complex(1.0, np.nan)
    complex_source = xrf.Geometry(
        source.array.copy(data=values), source.transform, dims=source.dims
    )
    transform = xrf.AffineTransform.from_matrix(
        source=FIXED, target=MOVING, matrix=ROTATION, translation=(0.5, 0.0, 0.0)
    )
    inline = xrf.resample(complex_source, target, transform=transform, workers=1).values
    threaded = xrf.resample(complex_source, target, transform=transform, workers=4).values
    assert_array_equal(np.isnan(threaded.real), np.isnan(inline.real))
    assert_array_equal(np.isnan(threaded.imag), np.isnan(inline.imag))
    assert_allclose(threaded, inline, rtol=0, atol=LATTICE_ATOL, equal_nan=True)


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 9, 10), (8, 1, 1)])
def test_threaded_small_and_one_row_targets(shape: tuple[int, int, int]) -> None:
    source, _ = case(0.0, None)
    target = geometry(np.zeros(shape), FIXED, 1.15 * ROTATION, (1.0, 1.0, 1.0))
    for transform in (
        field("dense", FIXED, MOVING, np.zeros(3), "identity"),
        xrf.AffineTransform.from_matrix(
            source=FIXED, target=MOVING, matrix=ROTATION, translation=(0.5, 0.0, 0.0)
        ),
    ):
        inline = xrf.resample(source, target, transform=transform, workers=1).values
        threaded = xrf.resample(source, target, transform=transform, workers=4).values
        assert_allclose(threaded, inline, rtol=0, atol=LATTICE_ATOL, equal_nan=True)


def test_threaded_periodic_sources() -> None:
    from test_seam_resampling import GLOBAL_LON, LAT, grid, source

    src = source(GLOBAL_LON, LAT)
    target = grid(np.linspace(-20.0, 380.0, 400), np.linspace(-30.0, 40.0, 60))
    inline = xrf.resample(src.rf.geometry, target, method="cubic", workers=1).values
    threaded = xrf.resample(src.rf.geometry, target, method="cubic", workers=4).values
    assert_array_equal(threaded, inline)


class ShiftedTarget(xrf.CompositeTransform):
    """A target grid transform whose override has no thread-safety contract."""


@pytest.mark.usefixtures("no_threads")
def test_a_foreign_target_transform_runs_inline() -> None:
    source, target = case(0.0, None)
    grid = target.grid()
    foreign = xrf.Grid(ShiftedTarget(grid.transform), grid.coordinates)
    transform = field("dense", FIXED, MOVING, np.zeros(3), "identity")
    xrf.resample(source, foreign, transform=transform, workers=4)
