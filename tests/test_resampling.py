"""Storage order at the private sampling planner boundary."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from xarrayrf import AffineTransform, ArrayCoordinates, CoordinateSystem, Grid, ReferenceFrame
from xarrayrf._resampling import execute, plan

pytest.importorskip("scipy")


@pytest.mark.parametrize("general", [False, True])
def test_resampling_storage_orders_differ_from_sampling_dims(general: bool) -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("x", "y"), ("mm", "mm")))
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("u", "v"), ("mm", "mm")),
        target=frame,
        matrix=np.eye(2),
        translation=(0.0, 0.0),
    )
    source_v = np.array([0.0, 2.0, 5.0 if general else 4.0])
    target_v = np.array([1.0, 3.5 if general else 3.0])
    source = Grid(transform, {"u": ("i", [0, 1, 2]), "v": ("j", source_v)})
    target = Grid(transform, {"v": ("j", target_v), "u": ("i", [0.5, 1.5])})
    prepared = plan(
        source._sampling(),
        target._sampling(),
        source_order=("j", "i"),
        target_order=("i", "j"),
        dtype=np.dtype(np.float64),
        transform=None,
        method="linear",
        fill_value=np.nan,
        domain="samples",
        block_points=1,
        other_dims=(),
    )
    values = 10 * source_v[:, None] + np.arange(3)[None, :]
    if prepared.window is not None:
        values = values[prepared.window]
    expected = np.array([0.5, 1.5])[:, None] + 10 * target_v[None, :]
    # Small synthetic coordinates accumulate only float64 interpolation roundoff.
    result = cast(npt.NDArray[np.float64], execute(prepared, values))
    assert_allclose(result, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("which", ["source_order", "target_order"])
@pytest.mark.parametrize("storage_order", [("other",), (), ("i", "i")])
def test_resampling_refuses_invalid_storage_orders(
    which: str, storage_order: tuple[str, ...]
) -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("x",), ("mm",)))
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("u",), ("mm",)), target=frame, matrix=[[1]], translation=[0]
    )
    grid = Grid(transform, {"u": ("i", [0, 1, 2])})
    orders: dict[str, tuple[str, ...]] = {"source_order": ("i",), "target_order": ("i",)}
    orders[which] = storage_order
    with pytest.raises(ValueError, match=which):
        plan(
            grid._sampling(),
            grid._sampling(),
            source_order=orders["source_order"],
            target_order=orders["target_order"],
            dtype=np.dtype(np.float64),
            transform=None,
            method="linear",
            fill_value=np.nan,
            domain="samples",
            block_points=1,
            other_dims=(),
        )
