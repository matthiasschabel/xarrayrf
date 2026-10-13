"""Resampling through dense fields with SimpleITK evaluating the field, against NumPy alone."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import xarrayrf as xrf
import xarrayrf._sitk_engine as engine

pytest.importorskip("scipy", minversion="1.18")
pytest.importorskip("SimpleITK")

ATOL = 1e-9
"""Index-space float64 arithmetic in ITK against reassociated NumPy, values of order one."""


def frames(dims: int) -> tuple[xrf.ReferenceFrame, xrf.ReferenceFrame]:
    system = xrf.CoordinateSystem(("x", "y", "z")[:dims], ("mm",) * dims)
    return (
        xrf.ReferenceFrame.declared(("engine", f"fixed{dims}"), system),
        xrf.ReferenceFrame.declared(("engine", f"moving{dims}"), system),
    )


def geometry(
    values: npt.NDArray[Any],
    frame: xrf.ReferenceFrame,
    matrix: npt.ArrayLike,
    origin: npt.ArrayLike,
    offsets: tuple[float, ...] | None = None,
) -> xrf.Geometry:
    dims = ("k", "j", "i")[-values.ndim :]
    axes = ("i", "j", "k")[: values.ndim]
    coords = {d: np.arange(n) for d, n in zip(dims, values.shape, strict=True)}
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(axes, ("1",) * values.ndim, sample_offset=offsets),
        target=frame,
        matrix=matrix,
        translation=origin,
    )
    return xrf.Geometry(xr.DataArray(values, dims=dims, coords=coords), transform, dims=dims)


def warp(
    fixed: xrf.ReferenceFrame,
    moving: xrf.ReferenceFrame,
    shape: tuple[int, ...],
    spacing: float,
    origin: npt.ArrayLike,
    outside: Any = "identity",
) -> xrf.DisplacementFieldTransform:
    dims = len(shape)
    axes = ("i", "j", "k")[:dims]
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(axes, ("1",) * dims, sample_offset=(0.5,) * dims),
        target=fixed,
        matrix=spacing * np.eye(dims),
        translation=origin,
    )
    grid = xrf.Grid(transform, {d: (d, np.arange(n)) for d, n in zip(axes, shape, strict=True)})
    values = 0.8 * np.random.default_rng(7).standard_normal((*shape, dims))
    return xrf.DisplacementFieldTransform(
        source=fixed, target=moving, field=values, grid=grid, outside=outside
    )


def rotation(dims: int, angle: float, reflect: bool = False) -> npt.NDArray[np.float64]:
    matrix = np.eye(dims)
    matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    if reflect:
        matrix[:, -1] *= -1
    return matrix


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count SimpleITK evaluations."""
    counted: list[int] = []
    original = engine.FieldEngine.index_displacement

    def counting(self: engine.FieldEngine, first: int, stop: int, threads: int) -> Any:
        counted.append(stop - first)
        return original(self, first, stop, threads)

    monkeypatch.setattr(engine.FieldEngine, "index_displacement", counting)
    return counted


def both(
    source: xrf.Geometry, target: xrf.Geometry, transform: Any, **options: Any
) -> tuple[Any, Any]:
    with_engine = xrf.resample(source, target, transform=transform, **options).values
    engine.ENABLED = False
    try:
        without = xrf.resample(source, target, transform=transform, **options).values
    finally:
        engine.ENABLED = True
    return with_engine, without


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize("domain, offsets", [("samples", None), ("cells", (0.5, 0.5, 0.5))])
@pytest.mark.parametrize("origin", [0.0, 1e6])
@pytest.mark.parametrize("reflect", [False, True])
def test_the_engine_matches_numpy(
    calls: list[int],
    method: xrf.Method,
    domain: xrf.Domain,
    offsets: Any,
    origin: float,
    reflect: bool,
) -> None:
    fixed, moving = frames(3)
    corner = np.full(3, origin)
    source = geometry(
        np.random.default_rng(2).random((9, 10, 11)), moving, np.eye(3), corner, offsets
    )
    # Rotated (and reflected) about z, overhanging the field so its exterior acts.
    target = geometry(np.zeros((8, 9, 10)), fixed, 1.1 * rotation(3, 0.3, reflect), corner + 1.0)
    transform = warp(fixed, moving, (6, 7, 8), 1.5, corner - 0.4)
    fused, numpy_only = both(source, target, transform, method=method, domain=domain)
    assert calls, "expected SimpleITK to evaluate the field"
    assert_array_equal(np.isnan(fused), np.isnan(numpy_only))
    assert np.isfinite(fused).any()
    assert_allclose(fused, numpy_only, rtol=0, atol=ATOL, equal_nan=True)


def test_the_engine_in_two_dimensions(calls: list[int]) -> None:
    fixed, moving = frames(2)
    source = geometry(
        np.random.default_rng(3).random((20, 22)), moving, 0.9 * np.eye(2), (0.0, 0.0)
    )
    target = geometry(np.zeros((17, 19)), fixed, rotation(2, -0.4), (1.0, 2.0))
    transform = warp(fixed, moving, (9, 11), 2.0, (-0.5, -0.5))
    fused, numpy_only = both(source, target, transform)
    assert calls
    assert_allclose(fused, numpy_only, rtol=0, atol=ATOL, equal_nan=True)


def test_permuted_target_axes(calls: list[int]) -> None:
    fixed, moving = frames(3)
    source = geometry(np.random.default_rng(4).random((9, 10, 11)), moving, np.eye(3), np.zeros(3))
    # The target's first storage axis runs along the field's last.
    permutation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    target = geometry(np.zeros((8, 7, 6)), fixed, 1.2 * permutation, np.full(3, 0.5))
    transform = warp(fixed, moving, (6, 7, 8), 1.5, np.zeros(3))
    fused, numpy_only = both(source, target, transform)
    assert calls
    assert_allclose(fused, numpy_only, rtol=0, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("outside", ["raise", "nearest"])
def test_a_target_inside_the_support_uses_the_engine_for_any_exterior(
    calls: list[int], outside: Any
) -> None:
    fixed, moving = frames(3)
    source = geometry(np.random.default_rng(5).random((9, 10, 11)), moving, np.eye(3), np.zeros(3))
    target = geometry(np.zeros((5, 5, 5)), fixed, 1.1 * rotation(3, 0.2), np.full(3, 2.0))
    transform = warp(fixed, moving, (8, 8, 8), 1.5, np.zeros(3), outside=outside)
    fused, numpy_only = both(source, target, transform)
    assert calls
    assert_allclose(fused, numpy_only, rtol=0, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("outside", ["raise", "nearest"])
def test_an_overhanging_target_keeps_numpy_for_other_exteriors(
    calls: list[int], outside: Any
) -> None:
    fixed, moving = frames(3)
    source = geometry(np.random.default_rng(6).random((9, 10, 11)), moving, np.eye(3), np.zeros(3))
    target = geometry(np.zeros((8, 9, 10)), fixed, 1.1 * rotation(3, 0.3), np.full(3, 1.0))
    transform = warp(fixed, moving, (6, 7, 8), 1.5, np.zeros(3), outside=outside)
    if outside == "raise":
        with pytest.raises(ValueError, match="outside the displacement field's support"):
            xrf.resample(source, target, transform=transform)
    else:
        fused, numpy_only = both(source, target, transform)
        assert_allclose(fused, numpy_only, rtol=0, atol=0, equal_nan=True)
    assert not calls


def test_an_anisotropic_oblique_target_keeps_numpy(calls: list[int]) -> None:
    # Field index columns of a rotated target on an anisotropic field grid are not orthogonal,
    # so no ITK image direction and spacing describe them.
    fixed, moving = frames(3)
    source = geometry(np.random.default_rng(8).random((9, 10, 11)), moving, np.eye(3), np.zeros(3))
    target = geometry(
        np.zeros((8, 9, 10)), fixed, rotation(3, 0.3) @ np.diag([1.0, 2.0, 1.0]), np.ones(3)
    )
    transform = warp(fixed, moving, (6, 7, 8), 1.5, np.zeros(3))
    stretched = xrf.DisplacementFieldTransform(
        source=fixed,
        target=moving,
        field=np.asarray(transform.field),
        grid=xrf.Grid(
            xrf.AffineTransform.from_matrix(
                source=transform.grid.transform.source,
                target=fixed,
                matrix=np.diag([1.5, 1.0, 2.0]),
                translation=np.zeros(3),
            ),
            {d: (d, np.arange(n)) for d, n in zip("ijk", (6, 7, 8), strict=True)},
        ),
        outside="identity",
    )
    fused, numpy_only = both(source, target, stretched)
    assert not calls
    assert_allclose(fused, numpy_only, rtol=0, atol=0, equal_nan=True)


def test_threads_with_the_engine(calls: list[int]) -> None:
    fixed, moving = frames(3)
    source = geometry(np.random.default_rng(9).random((9, 10, 11)), moving, np.eye(3), np.zeros(3))
    target = geometry(np.zeros((8, 9, 10)), fixed, 1.1 * rotation(3, 0.3), np.full(3, 1.0))
    transform = warp(fixed, moving, (6, 7, 8), 1.5, np.zeros(3))
    inline = xrf.resample(source, target, transform=transform, workers=1).values
    threaded = xrf.resample(source, target, transform=transform, workers=4).values
    assert calls
    assert_array_equal(threaded, inline)


@pytest.mark.parametrize("dims", [1, 4])
def test_ranks_simpleitk_lacks_keep_numpy(calls: list[int], dims: int) -> None:
    names = ("i", "j", "k", "l")[:dims]
    system = xrf.CoordinateSystem(("x", "y", "z", "t")[:dims], ("mm",) * dims)
    fixed = xrf.ReferenceFrame.declared(("engine", f"fixed-rank{dims}"), system)
    moving = xrf.ReferenceFrame.declared(("engine", f"moving-rank{dims}"), system)

    def framed(values: npt.NDArray[Any], frame: xrf.ReferenceFrame, offsets: Any) -> xrf.Geometry:
        transform = xrf.AffineTransform.from_matrix(
            source=xrf.ArrayCoordinates(names, ("1",) * dims, sample_offset=offsets),
            target=frame,
            matrix=np.eye(dims),
            translation=np.zeros(dims),
        )
        coords = {d: np.arange(n) for d, n in zip(names, values.shape, strict=True)}
        return xrf.Geometry(xr.DataArray(values, dims=names, coords=coords), transform, dims=names)

    shape = (5,) * dims
    grid_transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(names, ("1",) * dims, sample_offset=(0.5,) * dims),
        target=fixed,
        matrix=np.eye(dims),
        translation=np.zeros(dims),
    )
    transform = xrf.DisplacementFieldTransform(
        source=fixed,
        target=moving,
        field=0.2 * np.ones((*shape, dims)),
        grid=xrf.Grid(grid_transform, {d: (d, np.arange(5)) for d in names}),
        outside="identity",
    )
    source = framed(np.random.default_rng(1).random(shape), moving, None)
    target = framed(np.zeros((4,) * dims), fixed, None)
    fused, numpy_only = both(source, target, transform)
    assert not calls
    assert_allclose(fused, numpy_only, rtol=0, atol=0, equal_nan=True)


def test_blocks_keep_their_bound_when_a_row_exceeds_it(
    calls: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    import xarrayrf._resampling as resampling

    fixed, moving = frames(3)
    source = geometry(np.ones((10, 10, 10)), moving, np.eye(3), np.zeros(3))
    target = geometry(np.zeros((8, 9, 10)), fixed, np.eye(3), np.zeros(3))
    transform = warp(fixed, moving, (6, 7, 8), 1.0, np.zeros(3))
    sizes: list[int] = []
    original = resampling._block_positions

    def recorded(plan: Any, start: int, stop: int) -> Any:
        sizes.append(stop - start)
        return original(plan, start, stop)

    monkeypatch.setattr(resampling, "_block_positions", recorded)
    xrf.resample(source, target, transform=transform, block_points=7, workers=1)
    # A 90-sample row does not fit 7-sample blocks: they keep their bound and use NumPy.
    assert max(sizes) <= 7
    assert not calls
    sizes.clear()
    xrf.resample(source, target, transform=transform, block_points=200, workers=1)
    # Two whole rows fit: blocks round down to 180 and SimpleITK evaluates them.
    assert max(sizes) == 180 and calls


def exact_case(
    outside: Any = "identity", offsets: Any = (0.5, 0.5, 0.5), intervals: Any = None
) -> tuple[xrf.Geometry, xrf.DisplacementFieldTransform, xrf.ReferenceFrame]:
    """A unit field grid at the origin with x-displacement 0.25 (1 + x index): exact values."""
    fixed, moving = frames(3)
    shape = (4, 4, 1) if intervals else (4, 4, 4)
    values = np.zeros((*shape, 3))
    values[..., 0] = 0.25 * (1 + np.arange(4.0))[:, None, None]
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3, sample_offset=offsets),
        target=fixed,
        matrix=np.eye(3),
        translation=np.zeros(3),
    )
    coordinates = {d: (d, np.arange(n)) for d, n in zip("ijk", shape, strict=True)}
    grid = xrf.Grid(transform, coordinates, intervals=intervals)
    warp_ = xrf.DisplacementFieldTransform(
        source=fixed, target=moving, field=values, grid=grid, outside=outside
    )
    # The source holds its own x coordinate, so the result reads off the mapped x exactly.
    xs = np.broadcast_to(np.arange(-2.0, 7.0)[None, None, :], (9, 9, 9)).copy()
    source = geometry(xs, moving, np.eye(3), np.array([-2.0, -2.0, -2.0]))
    return source, warp_, fixed


def test_support_faces_and_outer_half_cells(calls: list[int]) -> None:
    source, transform, fixed = exact_case()
    # Target x on the lower face and in the outer half cell: -0.75 to 0.25 in quarter steps.
    target = geometry(np.zeros((2, 2, 5)), fixed, np.diag([0.25, 1.0, 1.0]), (-0.75, 1.0, 1.0))
    fused, numpy_only = both(source, target, transform, domain="cells")
    assert calls
    xs = -0.75 + 0.25 * np.arange(5)
    # Outside [-0.5, 3.5) the identity; inside, the edge held over the outer half cell.
    displacement = 0.25 * (1 + np.clip(xs, 0.0, 3.0))
    expected = np.broadcast_to(xs + np.where(xs < -0.5, 0.0, displacement), (2, 2, 5))
    assert_array_equal(fused, expected)
    assert_array_equal(numpy_only, expected)


def test_the_upper_face_is_excluded(calls: list[int]) -> None:
    source, transform, fixed = exact_case()
    target = geometry(np.zeros((2, 2, 4)), fixed, np.diag([0.25, 1.0, 1.0]), (3.0, 1.0, 1.0))
    fused, numpy_only = both(source, target, transform, domain="cells")
    assert calls
    xs = 3.0 + 0.25 * np.arange(4)
    expected = np.broadcast_to(xs + np.where(xs < 3.5, 1.0, 0.0), (2, 2, 4))
    assert_array_equal(fused, expected)
    assert_array_equal(numpy_only, expected)


def test_a_single_slab_with_a_declared_interval(calls: list[int]) -> None:
    source, transform, fixed = exact_case(intervals={"k": np.array([[-0.5, 0.5]])})
    target = geometry(np.zeros((3, 3, 3)), fixed, np.eye(3), (0.25, 0.25, -0.75))
    fused, numpy_only = both(source, target, transform, domain="cells")
    assert calls
    assert_allclose(fused, numpy_only, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("offsets", [(0.0, 0.5, 0.5), (1.0, 0.5, 0.5)])
def test_uncentred_field_cells_keep_numpy(calls: list[int], offsets: Any) -> None:
    source, transform, fixed = exact_case(offsets=offsets)
    target = geometry(np.zeros((3, 3, 3)), fixed, np.eye(3), (0.25, 0.25, 0.25))
    fused, numpy_only = both(source, target, transform, domain="cells")
    assert not calls
    assert_array_equal(fused, numpy_only)


@pytest.mark.parametrize("clearance, used", [(2e-9, True), (0.5e-9, False)])
def test_the_corner_margin_decides_a_raising_field(
    calls: list[int], clearance: float, used: bool
) -> None:
    source, transform, fixed = exact_case(outside="raise")
    # The target box's lowest corner clears the support's lower face by `clearance`.
    corner = -0.5 + clearance
    target = geometry(np.zeros((2, 2, 2)), fixed, np.eye(3), (corner, corner, corner))
    xrf.resample(source, target, transform=transform, domain="cells")
    assert bool(calls) is used


def test_a_rotation_with_input_roundoff_still_uses_the_engine(calls: list[int]) -> None:
    # A rotation read from a file, orthogonal only to about 1e-11, is still an ITK direction.
    fixed, moving = frames(3)
    source = geometry(np.random.default_rng(10).random((9, 10, 11)), moving, np.eye(3), np.zeros(3))
    matrix = 1.1 * rotation(3, 0.3) + 1e-11 * np.random.default_rng(11).standard_normal((3, 3))
    target = geometry(np.zeros((8, 9, 10)), fixed, matrix, np.full(3, 1.0))
    fused, numpy_only = both(source, target, warp(fixed, moving, (6, 7, 8), 1.5, np.zeros(3)))
    assert calls
    assert_allclose(fused, numpy_only, rtol=0, atol=ATOL, equal_nan=True)
