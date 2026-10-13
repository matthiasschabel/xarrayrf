"""Public behaviour of declared inverse pairs, chain inversion and the round-trip residual."""

from __future__ import annotations

import pickle
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

import xarrayrf as xrf

ATOL = 1e-12
"""Roundoff allowance for float64 evaluation of small synthetic fields."""

SYSTEM = xrf.CoordinateSystem(("x", "y", "z"), ("mm",) * 3)
FIXED = xrf.ReferenceFrame.declared(("test", "fixed"), SYSTEM)
WARPED = xrf.ReferenceFrame.declared(("test", "warped"), SYSTEM)
MOVING = xrf.ReferenceFrame.declared(("test", "moving"), SYSTEM)
INDEX = xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3, sample_offset=(0.5,) * 3)
SIZE = 16


def grid(frame: xrf.ReferenceFrame, size: int = SIZE) -> xrf.Grid:
    """A unit-spaced cube of ``size`` samples per axis located in ``frame``."""
    transform = xrf.AffineTransform.from_matrix(
        source=INDEX, target=frame, matrix=np.eye(3), translation=np.zeros(3)
    )
    return xrf.Grid(transform, {axis: (axis, np.arange(size)) for axis in "ijk"})


def field(
    values: npt.ArrayLike, source: xrf.ReferenceFrame, target: xrf.ReferenceFrame
) -> xrf.DisplacementFieldTransform:
    return xrf.DisplacementFieldTransform(
        source=source, target=target, field=values, grid=grid(source)
    )


def constant_pair(offset: npt.ArrayLike) -> xrf.DeclaredInversePair:
    """Opposite constant fields: an exact pair."""
    shift = np.broadcast_to(np.asarray(offset, dtype=float), (SIZE, SIZE, SIZE, 3))
    return xrf.DeclaredInversePair(field(shift, FIXED, WARPED), field(-shift, WARPED, FIXED))


AMPLITUDE = 0.5
WAVENUMBER = 2 * np.pi / SIZE


def displacement(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A smooth, invertible warp along x: u(p) = a sin(k x), with |u'| = a k < 1."""
    shift = np.zeros_like(points)
    shift[..., 0] = AMPLITUDE * np.sin(WAVENUMBER * points[..., 0])
    return shift


def smooth_pair() -> xrf.DeclaredInversePair:
    """The warp and its inverse, each sampled on its own grid as a registration tool writes them.

    The inverse displacement v(q) solves v = -u(q + v); the fixed-point iteration contracts
    because |u'| = a k ≈ 0.2.
    """
    points = grid(FIXED).points()
    v = np.zeros_like(points)
    for _ in range(60):
        v = -displacement(points + v)
    return xrf.DeclaredInversePair(
        field(displacement(points), FIXED, WARPED), field(v, WARPED, FIXED)
    )


INTERIOR = np.random.default_rng(3).uniform(3.0, SIZE - 4.0, size=(200, 3))
"""Points whose round trip stays inside both fields' supports."""

INTERPOLATION_TOLERANCE = 0.03
"""Linear interpolation of each field errs by at most h²/8 · max|u''| = a k² / 8 ≈ 0.0096 mm at
unit spacing; the round trip adds both errors, the first scaled by at most 1 + a k ≈ 1.2."""


class PointsOnly:
    """A user transform with points but no Jacobian."""

    def __init__(self, source: xrf.ReferenceFrame, target: xrf.ReferenceFrame, shift: float):
        self.source, self.target, self.shift = source, target, shift

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(points, dtype=np.float64) + self.shift


def translation(
    source: xrf.ReferenceFrame, target: xrf.ReferenceFrame, offset: npt.ArrayLike
) -> xrf.AffineTransform:
    return xrf.AffineTransform.from_matrix(
        source=source, target=target, matrix=np.eye(3), translation=offset
    )


# --- The pair ---------------------------------------------------------------------------------


def test_each_direction_maps_with_its_member_and_returns_the_other() -> None:
    pair = constant_pair([0.5, -0.25, 1.0])
    forward, backward = pair.forward, pair.backward
    assert forward.direction == "forward" and backward.direction == "backward"
    assert forward.pair is pair and forward is pair.forward
    assert (forward.source, forward.target) == (FIXED, WARPED)
    assert (backward.source, backward.target) == (WARPED, FIXED)
    assert forward.declared_inverse() is backward
    assert backward.declared_inverse().declared_inverse() is backward
    point = np.array([4.0, 5.0, 6.0])
    assert_allclose(forward.transform_point(point), point + np.array([0.5, -0.25, 1.0]), atol=ATOL)
    assert_allclose(
        forward.transform_point(point), forward.transform.transform_point(point), atol=0
    )


def test_a_smooth_pair_round_trips_within_interpolation_error() -> None:
    pair = smooth_pair()
    for direction in (pair.forward, pair.backward):
        report = xrf.round_trip_residual_report(direction, INTERIOR)
        assert report.maximum < INTERPOLATION_TOLERANCE
        assert report.unit == "mm"
    # Without the inverse the error would be the displacement itself.
    assert np.max(np.abs(displacement(INTERIOR)[..., 0])) > 10 * INTERPOLATION_TOLERANCE


def test_an_exact_pair_round_trips_to_roundoff() -> None:
    report = xrf.round_trip_residual_report(constant_pair([0.5, -0.25, 1.0]).forward, INTERIOR)
    assert report.maximum <= ATOL
    assert report.residual.shape == INTERIOR.shape
    assert report.distance.shape == INTERIOR.shape[:-1]


def test_mismatched_endpoints_are_refused() -> None:
    shift = np.zeros((SIZE, SIZE, SIZE, 3))
    forward = field(shift, FIXED, WARPED)
    with pytest.raises(ValueError, match="backward must map"):
        xrf.DeclaredInversePair(forward, field(shift, WARPED, MOVING))
    with pytest.raises(ValueError, match="backward must map"):
        xrf.DeclaredInversePair(forward, field(shift, FIXED, WARPED))


def test_a_direction_cannot_be_paired_again_or_built_directly() -> None:
    pair = constant_pair([1.0, 0.0, 0.0])
    with pytest.raises(TypeError, match="already a direction"):
        xrf.DeclaredInversePair(pair.forward, pair.backward.transform)
    with pytest.raises(TypeError, match="already a direction"):
        xrf.DeclaredInversePair(pair.forward.transform, pair.backward)
    with pytest.raises(TypeError, match="created by DeclaredInversePair"):
        xrf.PairedTransform()
    with pytest.raises(TypeError, match="SupportsPoints"):
        xrf.DeclaredInversePair(cast(Any, object()), pair.backward.transform)


def test_capabilities_follow_the_member() -> None:
    pair = constant_pair([1.0, 0.0, 0.0])
    assert isinstance(pair.forward, xrf.SupportsJacobian)
    assert isinstance(pair.forward, xrf.SupportsDeclaredInverse)
    assert not isinstance(pair.forward, xrf.SupportsInverse)
    assert not isinstance(pair.forward, xrf.SupportsAffine)
    point = np.array([[4.0, 5.0, 6.0]])
    member = pair.forward.transform
    assert isinstance(member, xrf.DisplacementFieldTransform)
    assert_allclose(pair.forward.jacobian(point), member.jacobian(point), atol=0)
    plain = xrf.DeclaredInversePair(PointsOnly(FIXED, WARPED, 1.0), PointsOnly(WARPED, FIXED, -1.0))
    assert not isinstance(plain.forward, xrf.SupportsJacobian)
    assert not hasattr(plain.forward, "jacobian")
    chain = xrf.CompositeTransform(plain.forward, translation(WARPED, MOVING, [0, 0, 0]))
    with pytest.raises(TypeError, match="provide no Jacobian"):
        chain.jacobian(point)


def test_an_affine_member_is_refused() -> None:
    affine = translation(FIXED, WARPED, [1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="affine and has an exact inverse"):
        xrf.DeclaredInversePair(affine, PointsOnly(WARPED, FIXED, -1.0))


def test_a_user_pair_is_measured_not_trusted() -> None:
    pair = xrf.DeclaredInversePair(PointsOnly(FIXED, WARPED, 1.0), PointsOnly(WARPED, FIXED, 0.0))
    report = xrf.round_trip_residual_report(pair.forward, [[0.0, 0.0, 0.0]])
    assert_allclose(report.residual, [[1.0, 1.0, 1.0]], atol=ATOL)
    assert_allclose(report.maximum, np.sqrt(3.0), rtol=1e-15)


# --- Chains -----------------------------------------------------------------------------------


def ants_chain(pair: xrf.DeclaredInversePair) -> xrf.CompositeTransform:
    """ANTs' forward output: the warp on the fixed point, then the affine to moving."""
    affine = xrf.AffineTransform.from_matrix(
        source=WARPED,
        target=MOVING,
        matrix=[[1.1, 0.1, 0.0], [0.0, 0.9, 0.0], [0.05, 0.0, 1.0]],
        translation=[2.0, -1.0, 0.5],
    )
    return xrf.CompositeTransform(pair.forward, affine)


def test_a_chain_with_a_pair_and_an_affine_inverts() -> None:
    pair = smooth_pair()
    chain = ants_chain(pair)
    inverse = chain.declared_inverse()
    assert (inverse.source, inverse.target) == (MOVING, FIXED)
    affine = chain.transforms[1]
    assert isinstance(affine, xrf.AffineTransform)
    assert inverse.transforms == (affine.inverse(), pair.backward)
    assert inverse.declared_inverse().transforms[0] is pair.forward
    report = xrf.round_trip_residual_report(chain, INTERIOR)
    assert report.maximum < INTERPOLATION_TOLERANCE
    # The exact inverse is still refused.
    with pytest.raises(TypeError, match="provide no inverse"):
        chain.inverse()


def test_nested_chains_invert_recursively() -> None:
    pair = constant_pair([0.5, 0.0, 0.0])
    inner = ants_chain(pair)
    outer = xrf.CompositeTransform(
        translation(xrf.ReferenceFrame.declared(("test", "a"), SYSTEM), FIXED, [1.0, 0.0, 0.0]),
        inner,
    )
    inverse = outer.declared_inverse()
    assert inverse.transforms[0] == inner.declared_inverse()
    report = xrf.round_trip_residual_report(outer, INTERIOR - [1.0, 0.0, 0.0])
    assert report.maximum <= 1e-10


def test_an_exact_chain_declares_its_exact_inverse() -> None:
    chain = xrf.CompositeTransform(
        translation(FIXED, WARPED, [1.0, 0.0, 0.0]), translation(WARPED, MOVING, [0.0, 2.0, 0.0])
    )
    assert chain.declared_inverse() == chain.inverse()


def test_an_unpaired_field_has_no_declared_inverse() -> None:
    shift = np.zeros((SIZE, SIZE, SIZE, 3))
    chain = xrf.CompositeTransform(
        translation(MOVING, FIXED, [0.0, 0.0, 0.0]), field(shift, FIXED, WARPED)
    )
    with pytest.raises(TypeError, match=r"transforms \[1\] have neither"):
        chain.declared_inverse()


def test_a_members_inverse_error_propagates() -> None:
    plane = xrf.ReferenceFrame.declared(
        ("test", "plane"), xrf.CoordinateSystem(("u", "v"), ("mm",) * 2)
    )
    projection = xrf.AffineTransform.from_matrix(
        source=FIXED, target=plane, matrix=np.eye(3)[:2], translation=[0.0, 0.0]
    )
    with pytest.raises(ValueError):
        xrf.CompositeTransform(projection).declared_inverse()


# --- Report -----------------------------------------------------------------------------------


def test_the_report_accepts_a_grid_and_freezes_its_values() -> None:
    pair = constant_pair([0.25, 0.0, 0.0])
    g = xrf.Grid(
        xrf.AffineTransform.from_matrix(
            source=INDEX, target=FIXED, matrix=np.eye(3), translation=[4.0, 4.0, 4.0]
        ),
        {axis: (axis, np.arange(3)) for axis in "ijk"},
    )
    report = xrf.round_trip_residual_report(pair.forward, g)
    assert report.residual.shape == (3, 3, 3, 3)
    assert report.distance.shape == (3, 3, 3)
    for values in (report.residual, report.distance):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    with pytest.raises(ValueError, match="evaluate it in the transform's source frame"):
        xrf.round_trip_residual_report(pair.forward, grid(WARPED, 3))


def test_the_report_refuses_what_it_cannot_measure() -> None:
    pair = constant_pair([0.25, 0.0, 0.0])
    with pytest.raises(ValueError, match="holds no points"):
        xrf.round_trip_residual_report(pair.forward, np.empty((0, 3)))
    with pytest.raises(ValueError, match="trailing axis of length 3"):
        xrf.round_trip_residual_report(pair.forward, [1.0, 2.0])
    with pytest.raises(TypeError, match="neither an exact nor a declared inverse"):
        xrf.round_trip_residual_report(pair.forward.transform, INTERIOR)
    mixed = xrf.ReferenceFrame.declared(
        ("test", "mixed"), xrf.CoordinateSystem(("x", "t"), ("mm", "s"))
    )
    mixed_pair = xrf.DeclaredInversePair(
        PointsOnly(mixed, mixed, 1.0), PointsOnly(mixed, mixed, -1.0)
    )
    with pytest.raises(ValueError, match="sharing one unit"):
        xrf.round_trip_residual_report(mixed_pair.forward, [[0.0, 0.0]])
    index = xrf.AffineTransform.from_matrix(
        source=INDEX, target=FIXED, matrix=np.eye(3), translation=[0.0, 0.0, 0.0]
    )
    with pytest.raises(ValueError, match="array coordinates"):
        xrf.round_trip_residual_report(index, [[0.0, 0.0, 0.0]])


def test_an_inverse_with_wrong_endpoints_is_refused() -> None:
    class Liar(PointsOnly):
        def declared_inverse(self) -> PointsOnly:
            return PointsOnly(WARPED, MOVING, 0.0)

    with pytest.raises(ValueError, match="the inverse maps"):
        xrf.round_trip_residual_report(Liar(FIXED, WARPED, 1.0), [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="the inverse of transform 0 maps"):
        xrf.CompositeTransform(Liar(FIXED, WARPED, 1.0)).declared_inverse()


def test_a_malformed_user_result_is_refused() -> None:
    class OnePoint(PointsOnly):
        def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return np.zeros(3)

    pair = xrf.DeclaredInversePair(OnePoint(FIXED, WARPED, 0.0), PointsOnly(WARPED, FIXED, 0.0))
    with pytest.raises(ValueError, match=r"the transform's result has shape \(3,\)"):
        xrf.round_trip_residual_report(pair.forward, np.zeros((4, 3)))
    with pytest.raises(ValueError, match=r"the inverse's result has shape \(3,\)"):
        xrf.round_trip_residual_report(pair.backward, np.zeros((4, 3)))


def test_undeclared_units_are_one_shared_unit() -> None:
    plain = xrf.ReferenceFrame.declared(
        ("test", "unitless"), xrf.CoordinateSystem(("x", "y"), (None, None))
    )
    pair = xrf.DeclaredInversePair(PointsOnly(plain, plain, 1.0), PointsOnly(plain, plain, -1.0))
    report = xrf.round_trip_residual_report(pair.forward, [[0.0, 0.0]])
    assert report.unit is None and report.maximum <= ATOL
    half = xrf.ReferenceFrame.declared(
        ("test", "half"), xrf.CoordinateSystem(("x", "y"), ("mm", None))
    )
    pair = xrf.DeclaredInversePair(PointsOnly(half, half, 1.0), PointsOnly(half, half, -1.0))
    with pytest.raises(ValueError, match="sharing one unit"):
        xrf.round_trip_residual_report(pair.forward, [[0.0, 0.0]])


def test_a_raise_field_outside_its_support_propagates() -> None:
    pair = constant_pair([0.25, 0.0, 0.0])
    with pytest.raises(ValueError, match="outside the displacement field's support"):
        xrf.round_trip_residual_report(pair.forward, [[100.0, 0.0, 0.0]])


# --- Identity, pickling, repr, encoding -------------------------------------------------------


def test_equality_and_hash_follow_the_members() -> None:
    shift = np.zeros((SIZE, SIZE, SIZE, 3))
    forward, backward = field(shift, FIXED, WARPED), field(shift, WARPED, FIXED)
    first, second = (
        xrf.DeclaredInversePair(forward, backward),
        xrf.DeclaredInversePair(forward, backward),
    )
    assert first == second and hash(first) == hash(second)
    assert first.forward == second.forward and hash(first.forward) == hash(second.forward)
    assert first.forward != second.backward
    other = xrf.DeclaredInversePair(forward, field(shift, WARPED, FIXED))
    assert first != other and first.forward != other.forward
    assert first.forward != forward


def test_pickling_keeps_both_directions_on_one_pair() -> None:
    pair = constant_pair([0.25, 0.0, 0.0])
    forward = pickle.loads(pickle.dumps(pair.forward))
    assert isinstance(forward, xrf.PairedTransform)
    assert forward.direction == "forward"
    assert forward.declared_inverse().declared_inverse() is forward
    assert forward.declared_inverse() is forward.pair.backward
    assert isinstance(forward, xrf.SupportsJacobian)
    point = np.array([4.0, 5.0, 6.0])
    assert_allclose(
        cast(xrf.PairedTransform, forward).transform_point(point),
        pair.forward.transform_point(point),
        atol=0,
    )


def test_repr_names_the_pair_and_direction() -> None:
    pair = constant_pair([0.25, 0.0, 0.0])
    assert repr(pair).startswith("DeclaredInversePair(forward=DisplacementFieldTransform(")
    assert repr(pair.backward) == f"{pair!r}.backward"


def test_encoding_is_refused_until_stage_1b() -> None:
    pair = constant_pair([0.25, 0.0, 0.0])
    for value in (pair, pair.forward):
        with pytest.raises(TypeError, match=r"declared inverse pair .* stage 1b"):
            xrf.encode(cast(Any, value))


# --- Resampling -------------------------------------------------------------------------------


def framed(values: npt.ArrayLike, frame: xrf.ReferenceFrame, *, extra: tuple[str, ...] = ()) -> Any:
    """A framed (i, j, k, *extra) array on unit index coordinates."""
    xr = pytest.importorskip("xarray")
    import xarrayrf.native  # noqa: F401  (registers the .rf accessor)

    data = np.asarray(values)
    binding = xrf.AffineTransform.from_matrix(
        source=INDEX, target=frame, matrix=np.eye(3), translation=np.zeros(3)
    )
    array = xr.DataArray(
        data,
        dims=("i", "j", "k", *extra),
        coords={
            dim: (dim, np.arange(n), {"units": "1"})
            for dim, n in zip("ijk", data.shape, strict=False)
        },
    )
    return array.rf.frame(binding, dims=("i", "j", "k"))


def test_resampling_both_ways_through_a_pair() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    i, j, k = np.indices((SIZE,) * 3, dtype=np.float64)
    offset = np.array([0.75, 0.5, 1.25])
    pair = constant_pair(offset)
    moving = framed(1.0 + 2.0 * i + 3.0 * j + 5.0 * k, WARPED)
    fixed = framed(np.zeros((6, 6, 6)), FIXED)
    expected = moving.rf.resample_to(fixed, transform=translation(FIXED, WARPED, offset))
    assert_allclose(
        moving.rf.resample_to(fixed, transform=pair.forward).values, expected.values, atol=1e-10
    )
    # The other way: a fixed-frame image onto a warped-frame grid through the declared inverse.
    back = framed(1.0 + 2.0 * i + 3.0 * j + 5.0 * k, FIXED).rf.resample_to(
        framed(np.zeros((6, 6, 6)), WARPED), transform=pair.forward.declared_inverse()
    )
    assert_allclose(back.values[2, 2, 2], 1.0 + 2.0 * 1.25 + 3.0 * 1.5 + 5.0 * 0.75, atol=1e-10)


def test_declared_vectors_resample_through_a_directions_jacobian() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    matrix = np.array([[1.2, 0.3, 0.0], [0.0, 0.9, 0.1], [0.2, 0.0, 1.1]])
    affine = xrf.AffineTransform.from_matrix(
        source=FIXED, target=WARPED, matrix=matrix, translation=(1.0, 1.0, 1.0)
    )
    sampled = xrf.sample_displacement_field(affine, grid(FIXED, 4))
    pair = xrf.DeclaredInversePair(sampled, PointsOnly(WARPED, FIXED, 0.0))
    source = framed(np.broadcast_to([1.0, 2.0, 3.0], (8, 8, 8, 3)).copy(), WARPED, extra=("c",))
    source = source.rf.declare_tensor_components(xrf.VECTOR, component_dims=("c",), bases=WARPED)
    target = framed(np.zeros((3, 3, 3)), FIXED)
    options = {"output_component_basis": FIXED, "component_reorientation": "jacobian"}
    through_pair = source.rf.resample_to(target, transform=pair.forward, **options)
    expected = source.rf.resample_to(target, transform=affine, **options)
    assert_allclose(
        through_pair.transpose("i", "j", "k", "c").values,
        expected.transpose("i", "j", "k", "c").values,
        atol=1e-10,
    )
