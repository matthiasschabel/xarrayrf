"""Inverses, :class:`xarrayrf.CompositeTransform` and :func:`xarrayrf.compose`."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

import xarrayrf as xrf

ATOL = 1e-12
"""Rounding allowance for a few products of small exact matrices."""

ANATOMY = xrf.DirectionVocabulary(
    "test-anatomy", ("right-to-left", "anterior-to-posterior", "inferior-to-superior")
)


def frame(name: str, axes: tuple[str, ...] = ("x", "y", "z")) -> xrf.ReferenceFrame:
    return xrf.ReferenceFrame.declared(
        ("test", name), xrf.CoordinateSystem(axes, ("mm",) * len(axes))
    )


A, B, C = frame("a"), frame("b"), frame("c")
IJK = xrf.ArrayCoordinates(("i", "j", "k"), ("1", "1", "1"))


def affine(
    source: xrf.Endpoint, target: xrf.Endpoint, matrix: npt.ArrayLike, translation: npt.ArrayLike
) -> xrf.AffineTransform:
    return xrf.AffineTransform(source=source, target=target, matrix=matrix, translation=translation)


def rotation_z(degrees: float) -> npt.NDArray[np.float64]:
    angle = np.radians(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1.0]]
    )


# Inverses


def test_a_rigid_inverse_round_trips() -> None:
    """Case 10: a rigid 4x4 in the Image to Equipment Mapping Matrix form."""
    patient_to_equipment = affine(A, B, rotation_z(30.0), (10.0, -5.0, 2.0))
    back = patient_to_equipment.inverse()
    assert back.source == B
    assert back.target == A
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]])
    assert_allclose(
        back.transform_point(patient_to_equipment.transform_point(points)), points, atol=ATOL
    )


def test_the_inverse_of_a_sample_locator_ends_at_array_coordinates() -> None:
    locator = affine(IJK, A, np.diag([0.5, 0.5, 2.0]), (10.0, 20.0, 30.0))
    lookup = locator.inverse()
    assert lookup.source == A
    assert lookup.target == IJK
    assert_allclose(lookup.transform_point([11.0, 21.0, 34.0]), [2.0, 2.0, 2.0], atol=ATOL)


def test_a_coordinate_system_change_inverts_to_the_reverse_change() -> None:
    lps = xrf.ReferenceFrame.declared(
        ("test", "patient"),
        xrf.CoordinateSystem(
            ("x", "y", "z"),
            ("mm",) * 3,
            vocabulary=ANATOMY,
            orientation=("right-to-left", "anterior-to-posterior", "inferior-to-superior"),
        ),
    )
    ras = lps.with_coordinate_system(
        xrf.CoordinateSystem(
            ("x", "y", "z"),
            ("mm",) * 3,
            vocabulary=ANATOMY,
            orientation=("left-to-right", "posterior-to-anterior", "inferior-to-superior"),
        )
    )
    assert xrf.coordinate_system_change(lps, ras).inverse() == xrf.coordinate_system_change(
        ras, lps
    )


def test_a_rectangular_affine_has_no_inverse() -> None:
    plane = xrf.ArrayCoordinates(("row", "column"), ("1", "1"))
    embedding = affine(plane, A, [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], (0.0, 0.0, 0.0))
    assert isinstance(embedding, xrf.SupportsInverse)
    with pytest.raises(ValueError, match="different dimension and has no inverse"):
        embedding.inverse()


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        pytest.param(np.diag([1.0, 1.0, 0.0]), "zero row", id="zero-row"),
        pytest.param(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [2.0, 0.0, 3.0]], "zero column", id="zero-column"
        ),
        pytest.param(
            [[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-15, 0.0], [0.0, 0.0, 1.0]],
            "singular or ill-conditioned",
            id="nearly-parallel-rows",
        ),
    ],
)
def test_singular_or_ill_conditioned_affines_have_no_inverse(
    matrix: npt.ArrayLike, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        affine(A, B, matrix, (0.0, 0.0, 0.0)).inverse()


def test_the_inverse_does_not_depend_on_the_choice_of_units() -> None:
    """Axes in km, mm and nm scale rows and columns; equilibration sees through the scaling.

    The 2-norm condition number of this matrix is about 1e18, far beyond the limit, although
    it is a rotation written in badly matched units. Its exact inverse is known.
    """
    left, right = np.diag([1e3, 1e-3, 1e-9]), np.diag([1e-6, 1.0, 1e6])
    rotation = rotation_z(30.0)
    transform = affine(A, B, left @ rotation @ right, (0.0, 0.0, 0.0))
    expected = np.linalg.inv(right) @ rotation.T @ np.linalg.inv(left)
    assert_allclose(transform.inverse().matrix, expected, rtol=1e-12, atol=0)


def test_the_condition_limit_is_one_over_ten_epsilon() -> None:
    assert pytest.approx(1.0 / (10.0 * np.finfo(float).eps)) == xrf.INVERSE_CONDITION_LIMIT


# Composition


def test_composition_applies_first_to_last() -> None:
    """Prior-art oracle: ITK, SimpleITK and VTK PreMultiply give 3; first-to-last gives 4."""
    translate = affine(A, B, np.eye(3), (1.0, 0.0, 0.0))
    scale = affine(B, C, 2.0 * np.eye(3), (0.0, 0.0, 0.0))
    chain = xrf.CompositeTransform(translate, scale)
    assert_allclose(chain.transform_point([1.0, 0.0, 0.0]), [4.0, 0.0, 0.0], atol=ATOL)
    assert chain.source == A
    assert chain.target == C


def test_compose_collapses_a_chain_of_affines_into_one() -> None:
    first = affine(IJK, A, np.diag([0.5, 0.5, 2.0]), (10.0, 20.0, 30.0))
    second = affine(A, B, rotation_z(90.0), (1.0, 2.0, 3.0))
    collapsed = xrf.compose(first, second)
    assert isinstance(collapsed, xrf.AffineTransform)
    assert isinstance(collapsed, xrf.SupportsAffine)
    assert collapsed.source == IJK
    assert collapsed.target == B
    assert_allclose(collapsed.matrix, second.matrix @ first.matrix, atol=ATOL)
    assert_allclose(
        collapsed.translation, second.matrix @ first.translation + second.translation, atol=ATOL
    )
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    assert_allclose(
        collapsed.transform_point(points),
        second.transform_point(first.transform_point(points)),
        atol=ATOL,
    )


def test_compose_collapses_nested_affine_chains() -> None:
    first = affine(IJK, A, np.diag([0.5, 0.5, 2.0]), (10.0, 20.0, 30.0))
    second = affine(A, B, rotation_z(90.0), (1.0, 2.0, 3.0))
    third = affine(B, C, np.eye(3), (0.0, 0.0, 1.0))
    flat = xrf.compose(first, second, third)
    nested = xrf.compose(xrf.CompositeTransform(first, second), third)
    assert isinstance(nested, xrf.AffineTransform)
    assert nested == flat


def test_mismatched_endpoints_are_refused() -> None:
    with pytest.raises(ValueError, match="each target must equal the next source exactly"):
        xrf.CompositeTransform(
            affine(A, B, np.eye(3), (0, 0, 0)), affine(C, A, np.eye(3), (0, 0, 0))
        )


def test_array_coordinates_may_not_sit_inside_a_chain() -> None:
    """Two images with equally named coordinates must not be joined through them."""
    image_a = affine(IJK, A, np.eye(3), (0.0, 0.0, 0.0))
    image_b = affine(IJK, A, 2.0 * np.eye(3), (5.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="may only begin or end a chain"):
        xrf.compose(image_a.inverse(), image_b)


def test_nested_roles_compose() -> None:
    """Case 14: the donor is a world for the biopsy and an object in the scanner."""
    biopsy = frame("biopsy")
    donor_as_world = xrf.ReferenceFrame.declared(
        ("test", "donor"), A.coordinate_system, role="world"
    )
    donor_as_object = xrf.ReferenceFrame.declared(
        ("test", "donor"), A.coordinate_system, role="object"
    )
    scanner = frame("scanner")
    chain = xrf.compose(
        affine(biopsy, donor_as_world, np.eye(3), (1.0, 0.0, 0.0)),
        affine(donor_as_object, scanner, np.eye(3), (0.0, 2.0, 0.0)),
    )
    assert_allclose(chain.transform_point([0.0, 0.0, 0.0]), [1.0, 2.0, 0.0], atol=ATOL)


def test_a_composition_needs_a_transform() -> None:
    with pytest.raises(ValueError, match="at least one transform"):
        xrf.CompositeTransform()


def test_members_must_be_transforms() -> None:
    with pytest.raises(TypeError, match="must implement SupportsPoints"):
        xrf.CompositeTransform(affine(A, B, np.eye(3), (0, 0, 0)), "scale")  # type: ignore[arg-type]


class Bend:
    """A user-defined nonlinear transform: (x, y, z) -> (x, y + x**2, z), with its Jacobian."""

    def __init__(self, source: xrf.Endpoint, target: xrf.Endpoint) -> None:
        self._source = source
        self._target = target

    @property
    def source(self) -> xrf.Endpoint:
        return self._source

    @property
    def target(self) -> xrf.Endpoint:
        return self._target

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        bent: npt.NDArray[np.float64] = values.copy()
        bent[..., 1] += values[..., 0] ** 2
        return bent

    def jacobian(self, at: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        if at is None:
            raise ValueError("Bend needs at")
        values = np.asarray(at, dtype=np.float64)
        result = np.broadcast_to(np.eye(3), (*values.shape[:-1], 3, 3)).copy()
        result[..., 1, 0] = 2.0 * values[..., 0]
        return result


class BendWithoutJacobian(Bend):
    """Setting the method to None makes the structural protocol check report it absent."""

    jacobian = None  # type: ignore[assignment]


def test_a_nonlinear_chain_stays_a_composite() -> None:
    chain = xrf.compose(affine(A, B, 2.0 * np.eye(3), (1.0, 0.0, 0.0)), Bend(B, C))
    assert isinstance(chain, xrf.CompositeTransform)
    assert_allclose(chain.transform_point([1.0, 0.0, 0.0]), [3.0, 9.0, 0.0], atol=ATOL)


def test_the_chain_rule_evaluates_later_jacobians_where_points_land() -> None:
    """J = J_bend(A x) @ A; at x = (1, 0, 0) the scaled point is (3, 0, 0), so dy/dx = 2*3*2."""
    chain = xrf.CompositeTransform(affine(A, B, 2.0 * np.eye(3), (1.0, 0.0, 0.0)), Bend(B, C))
    jacobian = chain.jacobian(np.array([[1.0, 0.0, 0.0]]))
    assert jacobian.shape == (1, 3, 3)
    assert_allclose(jacobian[0], [[2.0, 0.0, 0.0], [12.0, 2.0, 0.0], [0.0, 0.0, 2.0]], atol=ATOL)
    with pytest.raises(ValueError, match="at is required unless every member"):
        chain.jacobian()


def test_an_affine_chain_has_a_constant_jacobian() -> None:
    chain = xrf.CompositeTransform(
        affine(A, B, 2.0 * np.eye(3), (1.0, 0.0, 0.0)), affine(B, C, rotation_z(90.0), (0, 0, 0))
    )
    assert_allclose(chain.jacobian(), rotation_z(90.0) @ (2.0 * np.eye(3)), atol=ATOL)


def test_a_member_without_a_jacobian_leaves_the_chain_without_one() -> None:
    chain = xrf.CompositeTransform(affine(A, B, np.eye(3), (0, 0, 0)), BendWithoutJacobian(B, C))
    with pytest.raises(TypeError, match="provide no Jacobian"):
        chain.jacobian(np.zeros((1, 3)))


def test_a_chain_inverts_member_by_member_in_reverse() -> None:
    first = affine(A, B, rotation_z(30.0), (1.0, 2.0, 3.0))
    second = affine(B, C, np.diag([2.0, 3.0, 4.0]), (0.0, -1.0, 0.0))
    chain = xrf.CompositeTransform(first, second)
    back = chain.inverse()
    assert back.transforms == (second.inverse(), first.inverse())
    points = np.array([[1.0, 2.0, 3.0]])
    assert_allclose(back.transform_point(chain.transform_point(points)), points, atol=ATOL)


def test_a_member_whose_inverse_is_undetermined_raises_through_the_chain() -> None:
    plane = xrf.ArrayCoordinates(("row", "column"), ("1", "1"))
    embedding = affine(plane, A, [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="no inverse"):
        xrf.CompositeTransform(embedding, affine(A, B, np.eye(3), (0, 0, 0))).inverse()


@pytest.mark.parametrize("at", [1.0, [[1.0, 2.0]]])
def test_jacobian_points_must_match_the_source_axes(at: object) -> None:
    chain = xrf.CompositeTransform(affine(A, B, np.eye(3), (0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="trailing axis of length 3"):
        chain.jacobian(at)  # type: ignore[arg-type]


class Narrow(Bend):
    """Returns one value per point although its target has three axes."""

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(points, dtype=np.float64)[..., :1]


def test_a_member_returning_the_wrong_width_is_refused() -> None:
    chain = xrf.CompositeTransform(affine(A, B, np.eye(3), (0, 0, 0)), Narrow(B, C))
    with pytest.raises(ValueError, match="trailing axis of length 3"):
        chain.transform_point([[1.0, 2.0, 3.0]])


def test_a_member_without_an_inverse_leaves_the_chain_without_one() -> None:
    chain = xrf.CompositeTransform(affine(A, B, np.eye(3), (0, 0, 0)), Bend(B, C))
    with pytest.raises(TypeError, match="provide no inverse"):
        chain.inverse()


def test_chains_compare_by_members() -> None:
    first = affine(A, B, np.eye(3), (0.0, 0.0, 0.0))
    second = affine(B, C, np.eye(3), (1.0, 0.0, 0.0))
    assert xrf.CompositeTransform(first, second) == xrf.CompositeTransform(first, second)
    assert hash(xrf.CompositeTransform(first, second)) == hash(
        xrf.CompositeTransform(first, second)
    )
    assert xrf.CompositeTransform(first, second) != xrf.CompositeTransform(first)
    assert "CompositeTransform(" in repr(xrf.CompositeTransform(first))


def test_a_chain_passes_masked_values_to_its_first_member_intact() -> None:
    chain = xrf.CompositeTransform(affine(A, B, np.eye(3), (0.0, 0.0, 0.0)))
    masked_array = cast(Callable[..., npt.NDArray[np.float64]], np.ma.masked_array)
    with pytest.raises(TypeError, match="masked array"):
        chain.transform_point(masked_array([[1.0, 2.0, 3.0]], mask=[[True, False, False]]))


def test_subclass_results_are_still_validated() -> None:
    class Broken(xrf.AffineTransform):
        def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return np.full_like(super().transform_point(points), np.nan)

    first = affine(IJK, A, np.eye(3), (0.0, 0.0, 0.0))
    broken = Broken(source=A, target=B, matrix=np.eye(3), translation=np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        xrf.CompositeTransform(first, broken).transform_point(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        xrf.transform_named(broken, {name: 0.0 for name in A.axes})
