"""Resampling through fields declared ``outside="raise"``: targets they do not map are filled."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from test_resample_fields import FIXED, MOVING, ROTATION, SYSTEM, WARPED, Opaque, case, geometry

import xarrayrf as xrf
import xarrayrf._resampling as resampling
from xarrayrf.native import frame_array

pytest.importorskip("scipy", minversion="1.18")

SOURCE = xrf.ReferenceFrame.declared(("test", "source"), SYSTEM)
FILLS = [0.0, np.nan]


@pytest.fixture
def general_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every resampling off the fused field path."""
    monkeypatch.setattr(resampling, "_field_lattice_map", lambda *args, **kwargs: None)


def field_grid(
    frame: xrf.ReferenceFrame, shape: tuple[int, ...], origin: npt.ArrayLike, *, cells: bool
) -> xrf.Grid:
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(
            ("i", "j", "k"), ("1",) * 3, sample_offset=(0.5,) * 3 if cells else None
        ),
        target=frame,
        matrix=np.eye(3),
        translation=origin,
    )
    return xrf.Grid(transform, {d: (d, np.arange(n)) for d, n in zip("ijk", shape, strict=True)})


def warp(
    kind: str,
    source: xrf.ReferenceFrame,
    target: xrf.ReferenceFrame,
    *,
    shape: tuple[int, ...] = (6, 7, 8),
    origin: npt.ArrayLike = (0.0, 0.0, 0.0),
    outside: Any = "raise",
    seed: int = 1,
) -> Any:
    values = 0.3 * np.random.default_rng(seed).standard_normal((*shape, 3))
    if kind == "dense":
        return xrf.DisplacementFieldTransform(
            source=source,
            target=target,
            field=values,
            grid=field_grid(source, shape, origin, cells=True),
            outside=outside,
        )
    return xrf.BSplineTransform(
        source=source,
        target=target,
        coefficients=values,
        grid=field_grid(source, shape, origin, cells=False),
        outside=outside,
    )


def pointwise_validity(target: xrf.Geometry, transform: Any) -> npt.NDArray[np.bool_]:
    """Which target samples a field maps, classified point by point (dims k, j, i)."""
    shape = target.array.shape
    indices = np.indices(shape).reshape(3, -1)[::-1].T.astype(float)
    _, valid = transform._masked(target.transform.transform_point(indices), jacobian=False)
    return np.asarray(valid).reshape(shape)


class Recorder:
    """A user transform that records the rows it receives and returns them unchanged."""

    def __init__(self, source: xrf.ReferenceFrame, target: xrf.ReferenceFrame) -> None:
        self.source, self.target = source, target
        self.calls: list[npt.NDArray[np.float64]] = []

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        values = np.array(points, dtype=np.float64)
        self.calls.append(values)
        return values

    def jacobian(self, at: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        rows = np.asarray(at)
        self.calls.append(rows)
        return np.broadcast_to(np.eye(3), (*rows.shape[:-1], 3, 3)).copy()


def moved(source: xrf.Geometry, frame: xrf.ReferenceFrame) -> xrf.Geometry:
    transform = cast(xrf.AffineTransform, source.transform)
    rebound = xrf.AffineTransform.from_matrix(
        source=transform.source,
        target=frame,
        matrix=transform.matrix,
        translation=transform.translation,
    )
    return xrf.Geometry(source.array, rebound, dims=source.dims)


@pytest.mark.parametrize("kind", ["dense", "bspline"])
def test_no_invalid_row_reaches_a_later_member(kind: str) -> None:
    # Acceptance test 10: affine -> "raise" field -> user member; invalid rows are filled.
    source, _ = case(0.0, None)
    source = moved(source, SOURCE)
    target = geometry(np.zeros((8, 9, 10)), FIXED, 1.15 * ROTATION, (1.0, 1.0, 1.0))
    before = xrf.AffineTransform.from_matrix(
        source=FIXED, target=WARPED, matrix=np.eye(3), translation=(0.25, 0.0, 0.0)
    )
    field = warp(kind, WARPED, MOVING)
    recorder = Recorder(MOVING, SOURCE)
    chain = xrf.CompositeTransform(before, field, recorder)
    result = xrf.resample(source, target, transform=chain, fill_value=-7.0, block_points=11)
    received = np.concatenate(recorder.calls)
    assert np.isfinite(received).all()
    assert all(len(call) for call in recorder.calls)
    filled = result.values == -7.0
    assert filled.any() and not filled.all()
    assert len(received) == filled.size - filled.sum()


@pytest.mark.parametrize("kind", ["dense", "bspline"])
def test_masking_accumulates_and_skips_empty_blocks(kind: str) -> None:
    source, _ = case(0.0, None)
    source = moved(source, SOURCE)
    target = geometry(np.zeros((8, 9, 10)), FIXED, 1.15 * ROTATION, (1.0, 1.0, 1.0))
    first = warp(kind, FIXED, WARPED)
    # A smaller second field invalidates rows the first one kept.
    second = warp(kind, WARPED, MOVING, shape=(5, 5, 8), seed=2)
    recorder = Recorder(MOVING, SOURCE)
    chain = xrf.CompositeTransform(first, second, recorder)
    only_first = xrf.resample(
        source, target, transform=xrf.CompositeTransform(first, Recorder(WARPED, SOURCE))
    )
    result = xrf.resample(source, target, transform=chain, block_points=7)
    assert np.isnan(result.values).sum() > np.isnan(only_first.values).sum()
    # Blocks whose rows are all invalid call no later member.
    assert all(len(call) for call in recorder.calls)
    assert len(np.concatenate(recorder.calls)) == np.isfinite(result.values).sum()


@pytest.mark.usefixtures("general_only")
@pytest.mark.parametrize("kind", ["dense", "bspline"])
def test_the_general_path_fills_as_the_fused_path(kind: str) -> None:
    source, _ = case(0.0, None)
    target = geometry(np.zeros((8, 9, 10)), FIXED, 1.15 * ROTATION, (1.0, 1.0, 1.0))
    field = warp(kind, FIXED, MOVING)
    general = xrf.resample(source, target, transform=field, fill_value=-7.0).values
    valid = pointwise_validity(target, field)
    assert_array_equal(general == -7.0, ~valid)


@pytest.mark.parametrize("kind", ["dense", "bspline"])
def test_the_paths_agree_away_from_the_faces(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    source, _ = case(0.0, None)
    target = geometry(np.zeros((8, 9, 10)), FIXED, 1.15 * ROTATION, (1.0, 1.0, 1.0))
    field = warp(kind, FIXED, MOVING)
    fused = xrf.resample(source, target, transform=field, fill_value=-7.0).values
    monkeypatch.setattr(resampling, "_field_lattice_map", lambda *args, **kwargs: None)
    general = xrf.resample(source, target, transform=field, fill_value=-7.0).values
    assert_array_equal(fused == -7.0, general == -7.0)
    assert_allclose(fused, general, rtol=0, atol=1e-9)


def test_dask_chunks_fill_as_a_whole_array() -> None:
    pytest.importorskip("dask")
    source, _ = case(0.0, None)
    target = geometry(np.zeros((8, 9, 10)), FIXED, 1.15 * ROTATION, (1.0, 1.0, 1.0))
    field = warp("dense", FIXED, MOVING)
    whole = xrf.resample(source, target, transform=field).values
    chunked = xrf.Geometry(source.array.chunk({"k": 4}), source.transform, dims=source.dims)
    assert_array_equal(xrf.resample(chunked, target, transform=field).values, whole)


def test_a_user_transform_that_raises_still_raises() -> None:
    source, target = case(0.0, None)
    target = geometry(np.zeros((8, 9, 10)), FIXED, 1.15 * ROTATION, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="outside the displacement field's support"):
        xrf.resample(source, target, transform=Opaque(warp("dense", FIXED, MOVING)))


def test_direct_calls_still_raise() -> None:
    field = warp("bspline", FIXED, MOVING)
    outside = np.full((1, 3), 100.0)
    with pytest.raises(ValueError, match="outside the B-spline"):
        field.transform_point(outside)
    with pytest.raises(ValueError, match="outside the B-spline"):
        field.jacobian(outside)


# Declared components


def vectors(origin: float) -> xr.DataArray:
    rng = np.random.default_rng(3)
    values = rng.standard_normal((12, 12, 12, 3))
    binding = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3),
        target=MOVING,
        matrix=np.eye(3),
        translation=np.full(3, origin - 2.0),
    )
    framed = frame_array(
        values,
        xrf.Grid(binding, {d: (d, np.arange(12)) for d in "ijk"}),
        dims=("i", "j", "k", "c"),
        coords={"c": ("c", np.array(["x", "y", "z"]), {})},
    )
    return cast(
        xr.DataArray,
        framed.rf.declare_tensor_components(xrf.VECTOR, component_dims=("c",), bases=MOVING),
    )


def face_target(origin: float) -> xrf.Geometry:
    """Samples every half unit from -1: on the dense support's faces (-0.5, n - 0.5) and beside."""
    return geometry(np.zeros((9, 16, 14)), FIXED, 0.5 * np.eye(3), np.full(3, origin - 1.0))


def oblique_target(origin: float) -> xrf.Geometry:
    return geometry(np.zeros((8, 9, 10)), FIXED, 1.15 * ROTATION, np.full(3, origin + 1.0))


@pytest.mark.parametrize("fill", FILLS, ids=["zero", "nan"])
@pytest.mark.parametrize("origin", [0.0, 1e6])
@pytest.mark.parametrize("make_target", [face_target, oblique_target], ids=["faces", "oblique"])
@pytest.mark.parametrize("kind", ["dense", "bspline"])
def test_components_fill_exactly_where_the_field_does_not_map(
    kind: str, make_target: Any, origin: float, fill: float
) -> None:
    source = vectors(origin)
    target = make_target(origin)
    raising = warp(kind, FIXED, MOVING, origin=np.full(3, origin))
    identity = warp(kind, FIXED, MOVING, origin=np.full(3, origin), outside="identity")
    options: dict[str, Any] = {
        "output_component_basis": FIXED,
        "component_reorientation": "jacobian",
        "fill_value": fill,
    }
    result = xrf.resample(source.rf.geometry, target, transform=raising, **options)
    reference = xrf.resample(source.rf.geometry, target, transform=identity, **options)
    values = result.transpose("k", "j", "i", "c").values
    expected = reference.transpose("k", "j", "i", "c").values
    valid = pointwise_validity(target, raising)
    assert valid.any() and not valid.all()
    invalid = values[~valid]
    if np.isnan(fill):
        assert np.isnan(invalid).all()
    else:
        assert_array_equal(invalid, fill)
    assert np.isfinite(values[valid]).all()
    # The reference may take the fused path, whose positions agree with the general path's to
    # about 1e-9 of a sample at a 1e6 origin; values vary by tens per sample here.
    assert_allclose(values[valid], expected[valid], rtol=0, atol=1e-7)


def test_a_fully_unmapped_target_calls_no_reorientation() -> None:
    source = vectors(0.0)
    target = geometry(np.zeros((3, 3, 3)), FIXED, np.eye(3), np.full(3, 500.0))
    calls: list[npt.NDArray[np.float64]] = []

    def reorientation(push: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        calls.append(push)
        return push

    result = xrf.resample(
        source.rf.geometry,
        target,
        transform=warp("dense", FIXED, MOVING),
        output_component_basis=FIXED,
        component_reorientation=reorientation,
        fill_value=0.0,
    )
    assert_array_equal(result.values, 0.0)
    assert not calls


def test_a_valid_singular_jacobian_still_raises() -> None:
    # Validity is "this point maps", not invertibility: u_x = -x folds every point to x = 0.
    shape = (6, 7, 8)
    values = np.zeros((*shape, 3))
    values[..., 0] = -np.arange(shape[0])[:, None, None]
    field = xrf.DisplacementFieldTransform(
        source=FIXED,
        target=MOVING,
        field=values,
        grid=field_grid(FIXED, shape, np.zeros(3), cells=True),
    )
    target = geometry(np.zeros((3, 3, 3)), FIXED, np.eye(3), np.full(3, 1.0))
    with pytest.raises(ValueError, match="singular"):
        xrf.resample(
            vectors(0.0).rf.geometry,
            target,
            transform=field,
            output_component_basis=FIXED,
            component_reorientation="jacobian",
        )


class Malformed:
    """A user member whose results a float conversion would silently accept."""

    def __init__(self, kind: str) -> None:
        self.source, self.target, self.kind = WARPED, SOURCE, kind

    def transform_point(self, points: npt.ArrayLike) -> Any:
        values = np.asarray(points, dtype=np.float64)
        if self.kind == "masked":
            return np.ma.array(values, mask=np.ones(values.shape, dtype=bool))
        return values + 0j


@pytest.mark.parametrize("kind, error", [("masked", TypeError), ("complex", TypeError)])
def test_a_user_members_result_is_validated_after_a_raising_field(kind: str, error: type) -> None:
    source, _ = case(0.0, None)
    source = moved(source, SOURCE)
    target = geometry(np.zeros((2, 2, 2)), FIXED, np.eye(3), np.ones(3))
    chain = xrf.CompositeTransform(warp("dense", FIXED, WARPED), Malformed(kind))
    with pytest.raises(error) as direct:
        chain.transform_point([[1.0, 1.0, 1.0]])
    with pytest.raises(error) as resampled:
        xrf.resample(source, target, transform=chain)
    assert str(resampled.value) == str(direct.value)


def chain_validity(target: xrf.Geometry, members: list[Any]) -> npt.NDArray[np.bool_]:
    """Pointwise validity of a chain of fields (dims k, j, i), member by member."""
    shape = target.array.shape
    indices = np.indices(shape).reshape(3, -1)[::-1].T.astype(float)
    points = target.transform.transform_point(indices)
    valid = np.ones(len(points), dtype=bool)
    for member in members:
        mapped, kept = member._masked(points[valid], jacobian=False)
        rows = np.flatnonzero(valid)
        valid[rows[~kept]] = False
        points[rows[kept]] = mapped[kept]
    return valid.reshape(shape)


@pytest.mark.parametrize("fill", FILLS, ids=["zero", "nan"])
def test_components_through_two_raising_fields_follow_the_chain_rule(fill: float) -> None:
    # Two fields with noncommuting Jacobians; the second invalidates rows the first kept. The
    # reference declares both "identity" and takes CompositeTransform's own chain rule.
    source = vectors(0.0)
    target = oblique_target(0.0)
    first = warp("dense", FIXED, WARPED, seed=4)
    second = warp("bspline", WARPED, MOVING, shape=(6, 5, 8), seed=5)
    reference_chain = xrf.CompositeTransform(
        warp("dense", FIXED, WARPED, seed=4, outside="identity"),
        warp("bspline", WARPED, MOVING, shape=(6, 5, 8), seed=5, outside="identity"),
    )
    options: dict[str, Any] = {
        "output_component_basis": FIXED,
        "component_reorientation": "jacobian",
        "fill_value": fill,
    }
    chain = xrf.CompositeTransform(first, second)
    values = (
        xrf.resample(source.rf.geometry, target, transform=chain, block_points=13, **options)
        .transpose("k", "j", "i", "c")
        .values
    )
    expected = (
        xrf.resample(source.rf.geometry, target, transform=reference_chain, **options)
        .transpose("k", "j", "i", "c")
        .values
    )
    valid = chain_validity(target, [first, second])
    assert valid.sum() < pointwise_validity(target, first).sum()
    assert valid.any()
    invalid = values[~valid]
    assert np.isnan(invalid).all() if np.isnan(fill) else (invalid == fill).all()
    assert_allclose(values[valid], expected[valid], rtol=0, atol=1e-9)


def position_warp(outside: Any = "raise") -> xrf.PositionFieldTransform:
    grid = field_grid(FIXED, (6, 7, 8), np.zeros(3), cells=False)
    jitter = 0.2 * np.random.default_rng(6).standard_normal((6, 7, 8, 3))
    return xrf.PositionFieldTransform(
        source=FIXED, target=MOVING, field=grid.points() + jitter, grid=grid, outside=outside
    )


def test_a_position_field_fills_beyond_its_sample_hull() -> None:
    source, _ = case(0.0, None)
    # Samples every half unit from -1 reach the hull's upper faces (5, 6, 7) exactly.
    target = geometry(np.zeros((18, 16, 14)), FIXED, 0.5 * np.eye(3), np.full(3, -1.0))
    field = position_warp()
    valid = pointwise_validity(target, field)
    upper_face = (2 * 7 + 2, 2 * 6 + 2, 2 * 5 + 2)  # (k, j, i) of the point (5, 6, 7)
    assert valid[upper_face] and not valid[upper_face[0] + 1, upper_face[1], upper_face[2]]
    filled = xrf.resample(source, target, transform=field, fill_value=-7.0).values
    reference = xrf.resample(source, target, transform=position_warp("identity")).values
    assert_array_equal(filled[~valid], -7.0)
    assert_allclose(filled[valid], reference[valid], rtol=0, atol=1e-12)


def test_components_through_a_position_field_fill_beyond_its_hull() -> None:
    source = vectors(0.0)
    target = geometry(np.zeros((18, 16, 14)), FIXED, 0.5 * np.eye(3), np.full(3, -1.0))
    options: dict[str, Any] = {
        "output_component_basis": FIXED,
        "component_reorientation": "jacobian",
    }
    values = (
        xrf.resample(source.rf.geometry, target, transform=position_warp(), **options)
        .transpose("k", "j", "i", "c")
        .values
    )
    expected = (
        xrf.resample(source.rf.geometry, target, transform=position_warp("identity"), **options)
        .transpose("k", "j", "i", "c")
        .values
    )
    valid = pointwise_validity(target, position_warp())
    assert np.isnan(values[~valid]).all()
    assert_allclose(values[valid], expected[valid], rtol=0, atol=1e-12)
