"""Resampling through field transforms: the fused lattice path against the general path."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import xarrayrf as xrf
import xarrayrf._resampling as resampling

pytest.importorskip("scipy", minversion="1.18")

SYSTEM = xrf.CoordinateSystem(("x", "y", "z"), ("mm",) * 3)
FIXED = xrf.ReferenceFrame.declared(("test", "fixed"), SYSTEM)
WARPED = xrf.ReferenceFrame.declared(("test", "warped"), SYSTEM)
MOVING = xrf.ReferenceFrame.declared(("test", "moving"), SYSTEM)

ATOL = 1e-9
"""Reassociated float64 arithmetic on values of order one at origins up to 1e6."""

ROTATION = np.array(
    [[np.cos(0.2), -np.sin(0.2), 0.0], [np.sin(0.2), np.cos(0.2), 0.0], [0.0, 0.0, 1.0]]
)


def geometry(
    values: npt.NDArray[Any],
    frame: xrf.ReferenceFrame,
    matrix: npt.ArrayLike,
    origin: npt.ArrayLike,
    offsets: tuple[float, ...] | None = None,
) -> xrf.Geometry:
    dims = ("k", "j", "i")
    coords = {d: np.arange(n) for d, n in zip(dims, values.shape, strict=True)}
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3, sample_offset=offsets),
        target=frame,
        matrix=matrix,
        translation=origin,
    )
    return xrf.Geometry(xr.DataArray(values, dims=dims, coords=coords), transform, dims=dims)


def field_grid(
    frame: xrf.ReferenceFrame, shape: tuple[int, ...], origin: npt.ArrayLike, *, cells: bool
) -> xrf.Grid:
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(
            ("i", "j", "k"), ("1",) * 3, sample_offset=(0.5,) * 3 if cells else None
        ),
        target=frame,
        matrix=np.diag([1.5, 1.25, 1.0]),
        translation=origin,
    )
    return xrf.Grid(transform, {d: (d, np.arange(n)) for d, n in zip("ijk", shape, strict=True)})


def field(
    kind: str,
    source: xrf.ReferenceFrame,
    target: xrf.ReferenceFrame,
    origin: npt.ArrayLike,
    outside: Any,
) -> Any:
    shape = (6, 7, 8)
    values = 0.6 * np.random.default_rng(1).standard_normal((*shape, 3))
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


class Opaque:
    """The same mapping without the field's type, so resampling takes the general path."""

    def __init__(self, transform: Any) -> None:
        self.source = transform.source
        self.target = transform.target
        self._transform = transform

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        mapped: npt.NDArray[np.float64] = self._transform.transform_point(points)
        return mapped


def case(origin: float, offsets: tuple[float, ...] | None) -> tuple[xrf.Geometry, xrf.Geometry]:
    corner = np.full(3, origin)
    source = geometry(
        np.random.default_rng(2).random((9, 10, 11)),
        MOVING,
        np.diag([1.1, 0.9, 1.2]),
        corner - 1.0,
        offsets,
    )
    # The target overhangs the field and the source on purpose, so every exterior acts.
    target = geometry(np.zeros((8, 9, 10)), FIXED, 1.15 * ROTATION, corner - 2.0)
    return source, target


@pytest.fixture
def fused_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if a fused-eligible resampling falls back to the general path."""
    general = resampling._general_map

    def refuse(*args: Any, **kwargs: Any) -> Any:
        if not isinstance(args[4], Opaque):
            raise AssertionError("expected the fused field path")
        return general(*args, **kwargs)

    monkeypatch.setattr(resampling, "_general_map", refuse)


@pytest.mark.usefixtures("fused_only")
@pytest.mark.parametrize("kind", ["dense", "bspline"])
@pytest.mark.parametrize("outside", ["identity", "nearest"])
@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize(
    "domain, offsets", [("samples", None), ("cells", (0.5, 0.5, 0.5)), ("cells", (0.0, 0.25, 1.0))]
)
@pytest.mark.parametrize("origin", [0.0, 1e6])
def test_fused_path_matches_the_general_path(
    kind: str, outside: str, method: xrf.Method, domain: xrf.Domain, offsets: Any, origin: float
) -> None:
    source, target = case(origin, offsets)
    transform = field(kind, FIXED, MOVING, np.full(3, origin), outside)
    fused = xrf.resample(source, target, transform=transform, method=method, domain=domain)
    general = xrf.resample(
        source, target, transform=Opaque(transform), method=method, domain=domain
    )
    assert_array_equal(np.isnan(fused.values), np.isnan(general.values))
    assert np.isfinite(fused.values).any() and np.isnan(fused.values).any()
    assert_allclose(fused.values, general.values, rtol=0, atol=ATOL, equal_nan=True)


@pytest.mark.usefixtures("fused_only")
@pytest.mark.parametrize("paired", [False, True])
def test_fused_path_opens_chains_and_declared_pairs(paired: bool) -> None:
    source, target = case(0.0, None)
    before = xrf.AffineTransform.from_matrix(
        source=FIXED, target=WARPED, matrix=ROTATION.T, translation=(0.5, -0.25, 0.75)
    )
    warp = field("dense", WARPED, WARPED, np.zeros(3), "identity")
    if paired:
        inverse = field("dense", WARPED, WARPED, np.zeros(3), "identity")
        warp = xrf.DeclaredInversePair(warp, inverse).forward
    after = xrf.AffineTransform.from_matrix(
        source=WARPED, target=MOVING, matrix=np.diag([1.0, 1.1, 0.9]), translation=(-0.5, 0, 0)
    )
    chain = xrf.CompositeTransform(before, warp, after)
    fused = xrf.resample(source, target, transform=chain).values
    general = xrf.resample(source, target, transform=Opaque(chain)).values
    assert_array_equal(np.isnan(fused), np.isnan(general))
    assert_allclose(fused, general, rtol=0, atol=ATOL, equal_nan=True)


@pytest.mark.usefixtures("fused_only")
def test_fused_path_refuses_points_outside_a_raising_field() -> None:
    source, target = case(0.0, None)
    transform = field("dense", FIXED, MOVING, np.zeros(3), "raise")
    with pytest.raises(ValueError, match=r"point \(\d+, \d+, \d+\) lies outside the displacement"):
        xrf.resample(source, target, transform=transform)


def test_a_nonuniform_field_grid_takes_the_general_path() -> None:
    source, target = case(0.0, None)
    shape = (6, 7, 8)
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3, sample_offset=(0.5,) * 3),
        target=FIXED,
        matrix=np.eye(3),
        translation=np.zeros(3),
    )
    coordinates = {
        "i": ("i", np.cumsum([0.0, 1.0, 1.5, 1.0, 2.0, 1.0])),
        "j": ("j", np.arange(7)),
        "k": ("k", np.arange(8)),
    }
    warp = xrf.DisplacementFieldTransform(
        source=FIXED,
        target=MOVING,
        field=0.3 * np.random.default_rng(4).standard_normal((*shape, 3)),
        grid=xrf.Grid(transform, coordinates),
        outside="identity",
    )
    assert (
        resampling._field_lattice_map(
            source._sampling(), target._sampling(), source.dims, target.dims, warp, "samples"
        )
        is None
    )
    fused = xrf.resample(source, target, transform=warp).values
    general = xrf.resample(source, target, transform=Opaque(warp)).values
    assert_allclose(fused, general, rtol=0, atol=0, equal_nan=True)


def zero_warp(outside: Any = "identity") -> xrf.DisplacementFieldTransform:
    return xrf.DisplacementFieldTransform(
        source=FIXED,
        target=MOVING,
        field=np.zeros((6, 7, 8, 3)),
        grid=field_grid(FIXED, (6, 7, 8), np.zeros(3), cells=True),
        outside=outside,
    )


def test_a_nearly_uniform_source_takes_the_general_path() -> None:
    # 1.0000005 is uniform to the affine lattice path's 1e-6 but not to the general path's
    # exact steps; fusing it would move the half-way target onto the other nearest sample.
    values = np.broadcast_to(np.arange(3.0), (3, 3, 3)).copy()
    regular = geometry(values, MOVING, np.eye(3), np.zeros(3))
    irregular = xrf.Geometry(
        xr.DataArray(
            values,
            dims=("k", "j", "i"),
            coords={"i": [0, 1.0000005, 2], "j": range(3), "k": range(3)},
        ),
        regular.transform,
        dims=("k", "j", "i"),
    )
    target = geometry(np.zeros((1, 1, 1)), FIXED, np.eye(3), (0.5, 0, 0))
    fused = xrf.resample(irregular, target, transform=zero_warp(), method="nearest").values
    general = xrf.resample(irregular, target, transform=Opaque(zero_warp()), method="nearest")
    assert_array_equal(fused, general.values)
    # The same for a nearly uniform target.
    nearly = xrf.Geometry(
        xr.DataArray(
            np.zeros((3, 3, 3)),
            dims=("k", "j", "i"),
            coords={"i": [0.25, 1.2500005, 2.25], "j": range(3), "k": range(3)},
        ),
        geometry(np.zeros((3, 3, 3)), FIXED, np.eye(3), np.zeros(3)).transform,
        dims=("k", "j", "i"),
    )
    fused = xrf.resample(regular, nearly, transform=zero_warp(), method="nearest").values
    general = xrf.resample(regular, nearly, transform=Opaque(zero_warp()), method="nearest")
    assert_array_equal(fused, general.values)


class ShiftedChain(xrf.CompositeTransform):
    """A composite subclass whose override must survive resampling."""

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        shifted: npt.NDArray[np.float64] = super().transform_point(points) + np.array(
            [0.25, 0.0, 0.0]
        )
        return shifted


class ShiftedWarp(xrf.DisplacementFieldTransform):
    """A field subclass whose override must survive resampling."""

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        shifted: npt.NDArray[np.float64] = super().transform_point(points) + np.array(
            [0.25, 0.0, 0.0]
        )
        return shifted


@pytest.mark.parametrize("subclass", ["composite", "field"])
def test_subclass_overrides_are_honoured(subclass: str) -> None:
    source, target = case(0.0, None)
    if subclass == "composite":
        transform: Any = ShiftedChain(zero_warp())
    else:
        transform = ShiftedWarp(
            source=FIXED,
            target=MOVING,
            field=np.zeros((6, 7, 8, 3)),
            grid=field_grid(FIXED, (6, 7, 8), np.zeros(3), cells=True),
            outside="identity",
        )
    shifted = xrf.resample(source, target, transform=transform).values
    general = xrf.resample(source, target, transform=Opaque(transform)).values
    assert_allclose(shifted, general, rtol=0, atol=0, equal_nan=True)


@pytest.mark.usefixtures("fused_only")
def test_large_coordinate_labels_under_oblique_affines() -> None:
    # Origins in the coordinate labels (not the translation) on every grid, with rotated
    # affines around the field, cancel before scaling.
    label = 1e6
    source = xrf.Geometry(
        xr.DataArray(
            np.random.default_rng(5).random((9, 10, 11)),
            dims=("k", "j", "i"),
            coords={"i": label + np.arange(11.0), "j": np.arange(10.0), "k": np.arange(9.0)},
        ),
        xrf.AffineTransform.from_matrix(
            source=xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3),
            target=MOVING,
            matrix=np.diag([1.1, 0.9, 1.2]),
            translation=(-1.1 * label, 0.0, 0.0),
        ),
        dims=("k", "j", "i"),
    )
    target = geometry(np.zeros((8, 9, 10)), FIXED, 1.15 * ROTATION, (-2.0, -2.0, -2.0))
    transform = xrf.AffineTransform.from_matrix(
        source=FIXED, target=WARPED, matrix=ROTATION, translation=(0.25, 0.0, 0.0)
    )
    after = xrf.AffineTransform.from_matrix(
        source=WARPED, target=MOVING, matrix=ROTATION.T, translation=(0.0, 0.5, 0.0)
    )
    chain = xrf.CompositeTransform(
        transform, field("dense", WARPED, WARPED, np.zeros(3), "identity"), after
    )
    fused = xrf.resample(source, target, transform=chain).values
    general = xrf.resample(source, target, transform=Opaque(chain)).values
    assert np.isfinite(fused).any()
    assert_array_equal(np.isnan(fused), np.isnan(general))
    assert_allclose(fused, general, rtol=0, atol=ATOL, equal_nan=True)


@pytest.mark.usefixtures("fused_only")
def test_a_refusal_in_a_later_block_names_its_target_sample() -> None:
    source, _ = case(0.0, None)
    # The first rows of target samples lie inside the field; later ones leave it.
    target = geometry(np.zeros((8, 9, 10)), FIXED, 1.15 * ROTATION, (1.0, 1.0, 1.0))
    transform = field("dense", FIXED, MOVING, np.zeros(3), "raise")
    with pytest.raises(ValueError, match=r"point \((\d+), (\d+), (\d+)\)") as error:
        xrf.resample(source, target, transform=transform, block_points=7)
    named = tuple(int(n) for n in error.value.args[0].split("(")[1].split(")")[0].split(","))
    assert np.ravel_multi_index(named, (8, 9, 10)) >= 7
    # The named target sample's point lies outside the field: transform_point refuses it.
    point = target.transform.transform_point(np.array(named[::-1], dtype=float))
    with pytest.raises(ValueError, match="outside the displacement field's support"):
        transform.transform_point(point)


def test_an_empty_target_through_a_field() -> None:
    source, _ = case(0.0, None)
    empty = geometry(np.zeros((0, 3, 3)), FIXED, np.eye(3), np.zeros(3))
    result = xrf.resample(source, empty, transform=zero_warp())
    assert result.shape == (0, 3, 3)
