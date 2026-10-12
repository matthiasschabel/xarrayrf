"""Tensor types and the component transformation law (value geometry stage 1)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from test_relativity import boost_matrix

import xarrayrf as xrf

RTOL = ATOL = 1e-12
"""A few float64 operations on O(1) numbers."""

SHEAR = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
MIRROR = np.diag([-1.0, 1.0, 1.0])
SCALED_MIRROR = np.diag([-2.0, 1.0, 1.0])

EPSILON = np.zeros((3, 3, 3))
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    EPSILON[_i, _j, _k] = 1.0
    EPSILON[_i, _k, _j] = -1.0


def random_map(seed: int) -> npt.NDArray[np.float64]:
    """A well-conditioned non-orthogonal map scaled to ``|det| = 2``, so its exponents show."""
    matrix = np.eye(3) + 0.4 * np.random.default_rng(seed).standard_normal((3, 3))
    scaled: npt.NDArray[np.float64] = matrix * (2.0 / abs(np.linalg.det(matrix))) ** (1.0 / 3.0)
    return scaled


def assert_close(actual: Any, desired: Any) -> None:
    np.testing.assert_allclose(actual, desired, rtol=RTOL, atol=ATOL)


def transform(
    values: npt.ArrayLike, linear_map: npt.ArrayLike, tensor_type: xrf.TensorType, **kwargs: object
) -> npt.NDArray[Any]:
    return xrf.transform_tensor_components(values, linear_map, tensor_type, **kwargs)  # type: ignore[arg-type]


# Oracles


def test_shear_oracles_from_the_prior_art_study() -> None:
    assert_close(transform([1.0, 0.0, 0.0], SHEAR, xrf.COVECTOR), [1, -1, 0])
    assert_close(transform([0.0, 1.0, 0.0], SHEAR, xrf.VECTOR), [1, 1, 0])
    assert_close(
        transform(np.diag([3.0, 2.0, 1.0]), SHEAR, xrf.SECOND_ORDER_CONTRAVARIANT_TENSOR),
        [[5, 2, 0], [2, 2, 0], [0, 0, 1]],
    )


def test_polar_orthogonal_factor_reorients_the_prior_art_tensor() -> None:
    factor = xrf.polar_orthogonal_factor(SHEAR)
    expected = np.array([[2.0, 1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, np.sqrt(5.0)]]) / np.sqrt(5)
    assert_close(factor, expected)
    reoriented = transform(np.diag([2.0, 3.0, 1.0]), factor, xrf.SECOND_ORDER_CONTRAVARIANT_TENSOR)
    assert_close(reoriented[np.triu_indices(3)], [2.2, 0.4, 0, 2.8, 0, 1])


def test_polar_orthogonal_factor_of_an_improper_map_is_a_reflection() -> None:
    matrix = SHEAR @ SCALED_MIRROR
    factor = xrf.polar_orthogonal_factor(matrix)
    assert_close(factor.T @ factor, np.eye(3))
    assert np.linalg.det(factor) == pytest.approx(-1.0, abs=1e-12)
    positive = factor.T @ matrix
    assert_close(positive, positive.T)
    assert np.all(np.linalg.eigvalsh(positive) > 0)


def test_polar_orthogonal_factor_is_batched() -> None:
    stack = np.stack([SHEAR, 2.0 * np.eye(3), SHEAR @ MIRROR])
    expected = np.stack([xrf.polar_orthogonal_factor(matrix) for matrix in stack])
    assert_close(xrf.polar_orthogonal_factor(stack), expected)


def test_rotation_transforms_a_tensor_as_r_t_r_transpose() -> None:
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    tensor = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, 0.2], [0.5, 0.2, 1.0]])
    assert_close(
        transform(tensor, rotation, xrf.SECOND_ORDER_CONTRAVARIANT_TENSOR),
        rotation @ tensor @ rotation.T,
    )


def test_mixed_tensor_transforms_as_j_t_j_inverse() -> None:
    tensor = np.diag([3.0, 2.0, 1.0])
    mixed = xrf.TensorType(("contravariant", "covariant"))
    assert_close(transform(tensor, SHEAR, mixed), SHEAR @ tensor @ np.linalg.inv(SHEAR))


def test_order_three_tensor_matches_an_index_by_index_contraction() -> None:
    matrix = random_map(3)
    inverse = np.linalg.inv(matrix)
    tensor = np.random.default_rng(4).standard_normal((3, 3, 3))
    expected = np.einsum("ai,bj,ck,ibc->ajk", matrix, inverse, inverse, tensor)
    order_three = xrf.TensorType(("contravariant", "covariant", "covariant"))
    assert_close(transform(tensor, matrix, order_three), expected)


def test_reflection_negates_a_vector_and_a_pseudovector_differently() -> None:
    assert_close(transform([1.0, 1.0, 1.0], MIRROR, xrf.VECTOR), [-1, 1, 1])
    assert_close(transform([1.0, 1.0, 1.0], MIRROR, xrf.ORTHOGONAL_PSEUDOVECTOR), [1, -1, -1])
    assert_close(transform(2.0, MIRROR, xrf.PSEUDOSCALAR), -2.0)


@pytest.mark.parametrize("matrix", [random_map(1), SCALED_MIRROR], ids=["random", "scaled-mirror"])
def test_the_two_pseudovector_origins_follow_their_own_laws(
    matrix: npt.NDArray[np.float64],
) -> None:
    rng = np.random.default_rng(2)
    u, v = rng.standard_normal((2, 3))
    cross = transform(np.cross(u, v), matrix, xrf.CROSS_PRODUCT_PSEUDOVECTOR)
    assert_close(cross, np.cross(matrix @ u, matrix @ v))

    form = rng.standard_normal((3, 3))
    form = form - form.T
    inverse = np.linalg.inv(matrix)
    dual = 0.5 * np.einsum("ijk,jk->i", EPSILON, form)
    expected = 0.5 * np.einsum("ijk,jk->i", EPSILON, inverse.T @ form @ inverse)
    flux = transform(dual, matrix, xrf.TWO_FORM_DUAL_PSEUDOVECTOR)
    assert_close(flux, expected)

    assert not np.allclose(transform(u, matrix, xrf.CROSS_PRODUCT_PSEUDOVECTOR), flux)
    with pytest.raises(ValueError, match=r"CROSS_PRODUCT_PSEUDOVECTOR.*TWO_FORM_DUAL_PSEUDOVECTOR"):
        transform(u, matrix, xrf.ORTHOGONAL_PSEUDOVECTOR)


def test_densities_and_pseudoscalars_follow_the_determinant() -> None:
    matrix = random_map(5) @ MIRROR
    determinant = np.linalg.det(matrix)
    density = xrf.TensorType((), determinant_exponent=-1)
    assert_close(transform(6.0, matrix, density), 6.0 / abs(determinant))
    np.testing.assert_allclose(transform(1.0, matrix, xrf.PSEUDOSCALAR), np.sign(determinant))


def test_an_even_density_is_not_negated_by_a_reflection() -> None:
    density = xrf.TensorType((), determinant_exponent=-1)
    assert_close(transform(3.0, SCALED_MIRROR, density), 1.5)


def test_a_lorentz_boost_transforms_the_field_tensor() -> None:
    electric = np.array([1.0, 2.0, 3.0])
    magnetic = np.array([4.0, 5.0, 6.0])
    field = np.zeros((4, 4))
    field[0, 1:] = electric
    field[1:, 0] = -electric
    field[1:, 1:] = np.einsum("ijk,k->ij", EPSILON, magnetic)
    field_tensor = xrf.TensorType(
        ("contravariant", "contravariant"), index_symmetry="antisymmetric"
    )
    boosted = transform(field, boost_matrix(0.6), field_tensor)
    assert_close(boosted[0, 1:], [1.0, -2.0, 7.5])
    assert_close(0.5 * np.einsum("ijk,jk->i", EPSILON, boosted[1:, 1:]), [4.0, 8.5, 6.0])


def test_an_ordinary_scalar_is_unchanged_even_by_a_singular_map() -> None:
    values = np.array([1.0, 2.0])
    result = transform(values, np.zeros((3, 3)), xrf.SCALAR)
    np.testing.assert_array_equal(result, values)
    assert result is not values


def test_a_vector_accepts_a_singular_projection() -> None:
    projection = np.diag([1.0, 1.0, 0.0])
    np.testing.assert_allclose(transform([1.0, 2.0, 3.0], projection, xrf.VECTOR), [1, 2, 0])


@pytest.mark.parametrize("scale", [1e-120, 1e120])
def test_determinant_factors_stay_representable_where_det_does_not(scale: float) -> None:
    cube_root_density = xrf.TensorType((), determinant_exponent=1 / 3)
    factor = transform(1.0, scale * np.eye(3), cube_root_density)
    np.testing.assert_allclose(factor, scale, rtol=1e-12, atol=0)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_extreme_but_well_conditioned_scales_are_accepted(scale: float) -> None:
    cube_root_density = xrf.TensorType((), determinant_exponent=1 / 3)
    np.testing.assert_allclose(
        transform(1.0, scale * np.eye(3), cube_root_density), scale, rtol=1e-12
    )
    np.testing.assert_allclose(
        transform([1.0, 0.0, 0.0], scale * np.eye(3), xrf.COVECTOR), [1 / scale, 0, 0], rtol=1e-12
    )


def test_batched_maps_give_each_point_its_own_determinant_factor() -> None:
    maps = np.stack([2.0 * np.eye(3), SCALED_MIRROR, random_map(21), random_map(22) @ MIRROR])
    determinants = np.linalg.det(maps)
    assert len(set(np.sign(determinants))) == 2 and len(set(np.round(abs(determinants), 6))) > 1
    density = xrf.TensorType((), determinant_exponent=-1)
    assert_close(transform(np.full(4, 6.0), maps, density), 6.0 / abs(determinants))
    assert_close(transform(np.ones(4), maps, xrf.PSEUDOSCALAR), np.sign(determinants))
    vectors = np.random.default_rng(23).standard_normal((4, 3))
    expected = np.stack(
        [np.linalg.det(m) * np.linalg.inv(m).T @ v for m, v in zip(maps, vectors, strict=True)]
    )
    assert_close(transform(vectors, maps, xrf.CROSS_PRODUCT_PSEUDOVECTOR), expected)
    # Component axis leading: the factor must follow the batch axis, not the component axis.
    leading = transform(vectors.T, maps, xrf.CROSS_PRODUCT_PSEUDOVECTOR, component_axes=(0,))
    assert_close(leading, expected.T)


@pytest.mark.parametrize("exponent", [1.0, -1.0], ids=["overflow", "underflow"])
def test_a_determinant_factor_out_of_float64_range_is_refused(exponent: float) -> None:
    density = xrf.TensorType((), determinant_exponent=exponent)
    with pytest.raises(ValueError, match="over- or underflows"):
        transform(1.0, 1e120 * np.eye(3), density)


# Arrays: batching, axes, dtypes


def test_a_stack_of_maps_applies_one_map_per_point() -> None:
    maps = np.stack([SHEAR, MIRROR, 2.0 * np.eye(3), random_map(6)])
    vectors = np.random.default_rng(7).standard_normal((4, 3))
    expected = np.stack([m @ v for m, v in zip(maps, vectors, strict=True)])
    assert_close(transform(vectors, maps, xrf.VECTOR), expected)
    covectors = np.stack([np.linalg.inv(m).T @ v for m, v in zip(maps, vectors, strict=True)])
    assert_close(transform(vectors, maps, xrf.COVECTOR), covectors)


def test_one_map_broadcasts_over_every_point() -> None:
    values = np.random.default_rng(8).standard_normal((5, 4, 3))
    assert_close(transform(values, SHEAR, xrf.VECTOR), values @ SHEAR.T)


def test_a_batch_broadcasts_along_matching_trailing_axes() -> None:
    maps = np.stack([SHEAR, MIRROR, 2.0 * np.eye(3), random_map(9)])
    values = np.random.default_rng(10).standard_normal((5, 4, 3))
    expected = np.einsum("pab,qpb->qpa", maps, values)
    assert_close(transform(values, maps, xrf.VECTOR), expected)


@pytest.mark.parametrize("shape", [(1, 3), (3,)], ids=["singleton", "single-vector"])
def test_maps_never_enlarge_the_values(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="without enlarging it"):
        transform(np.ones(shape), np.stack([SHEAR] * 4), xrf.VECTOR)


def test_an_incompatible_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="without enlarging it"):
        transform(np.ones((5, 3)), np.stack([SHEAR] * 4), xrf.VECTOR)


def test_non_trailing_component_axes_keep_the_storage_order() -> None:
    tensor = np.random.default_rng(11).standard_normal((3, 7, 3))
    maps = np.stack([random_map(seed) for seed in range(12, 19)])
    trailing = np.moveaxis(tensor, 1, 0)
    expected = np.moveaxis(transform(trailing, maps, xrf.SECOND_ORDER_CONTRAVARIANT_TENSOR), 0, 1)
    for axes in ((0, 2), (-3, -1)):
        result = transform(tensor, maps, xrf.SECOND_ORDER_CONTRAVARIANT_TENSOR, component_axes=axes)
        assert result.shape == tensor.shape
        assert_close(result, expected)


def test_component_axes_follow_index_order() -> None:
    tensor = np.random.default_rng(19).standard_normal((3, 3))
    mixed = xrf.TensorType(("contravariant", "covariant"))
    swapped = transform(tensor, SHEAR, mixed, component_axes=(1, 0))
    assert_close(swapped, transform(tensor.T, SHEAR, mixed).T)


def test_an_empty_batch_returns_an_empty_result() -> None:
    result = transform(np.zeros((0, 3)), np.zeros((0, 3, 3)), xrf.COVECTOR)
    assert result.shape == (0, 3)


def test_complex_values_transform_by_their_real_and_imaginary_parts() -> None:
    rng = np.random.default_rng(20)
    real, imaginary = rng.standard_normal((2, 4, 3))
    result = transform(real + 1j * imaginary, SHEAR, xrf.COVECTOR)
    assert result.dtype == np.complex128
    expected = transform(real, SHEAR, xrf.COVECTOR) + 1j * transform(imaginary, SHEAR, xrf.COVECTOR)
    assert_close(result, expected)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(np.int64, np.float64), (np.float32, np.float64), (np.complex64, np.complex128)],
)
def test_output_dtype_promotes_to_at_least_float64(dtype: type, expected: type) -> None:
    assert transform(np.ones(3, dtype=dtype), SHEAR, xrf.VECTOR).dtype == expected


def test_non_finite_values_propagate() -> None:
    result = transform([[np.nan, 1.0, 0.0], [0.0, 1.0, 0.0]], SHEAR, xrf.VECTOR)
    assert np.all(np.isnan(result[0]))  # 0 * NaN is NaN, so the whole contraction is
    np.testing.assert_allclose(result[1], [1.0, 1.0, 0.0])


def test_a_singular_matrix_in_a_stack_is_named_by_index() -> None:
    maps = np.stack(
        [
            np.stack([SHEAR, SHEAR]),
            np.stack([SHEAR, np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])]),
        ]
    )
    with pytest.raises(ValueError, match=r"at stack index \(1, 1\) is singular or ill-conditioned"):
        transform(np.ones((2, 2, 3)), maps, xrf.COVECTOR)


def test_a_zero_row_in_a_stack_is_named_by_index() -> None:
    maps = np.stack([SHEAR, np.diag([1.0, 0.0, 1.0])])
    with pytest.raises(ValueError, match=r"at stack index \(1,\) has a zero row"):
        transform(np.ones((2, 3)), maps, xrf.COVECTOR)


@pytest.mark.parametrize(
    "tensor_type",
    [xrf.COVECTOR, xrf.PSEUDOSCALAR, xrf.TensorType((), determinant_exponent=2)],
    ids=["covariant", "odd", "exponent"],
)
def test_laws_needing_an_inverse_or_determinant_refuse_singular_maps(
    tensor_type: xrf.TensorType,
) -> None:
    with pytest.raises(ValueError, match="singular"):
        transform(np.ones((3,) * tensor_type.order), np.diag([1.0, 1.0, 0.0]), tensor_type)


# TensorType


def test_shortcuts_are_plain_tensor_types() -> None:
    assert xrf.TensorType(("contravariant",)) == xrf.VECTOR
    assert xrf.COVECTOR.index_variance == ("covariant",)
    assert xrf.SECOND_ORDER_CONTRAVARIANT_TENSOR.order == 2
    assert xrf.SCALAR.order == 0
    assert (
        xrf.TensorType(["covariant"], reflection_parity="odd", determinant_exponent=1)
        == xrf.CROSS_PRODUCT_PSEUDOVECTOR
    )
    assert xrf.TWO_FORM_DUAL_PSEUDOVECTOR.determinant_exponent == -1.0


def test_equality_and_hashing_are_exact() -> None:
    integer = xrf.TensorType((), determinant_exponent=-1)
    floating = xrf.TensorType((), determinant_exponent=-1.0)
    assert integer == floating and hash(integer) == hash(floating)
    unrestricted = xrf.TensorType(("contravariant",), reflection_parity="odd")
    assert unrestricted != xrf.ORTHOGONAL_PSEUDOVECTOR
    assert xrf.ORTHOGONAL_PSEUDOVECTOR.requires_orthogonal_map
    assert len({xrf.VECTOR, xrf.TensorType(("contravariant",)), xrf.COVECTOR}) == 2
    assert xrf.VECTOR != ("contravariant",)


def test_tensor_types_are_immutable() -> None:
    with pytest.raises(AttributeError):
        xrf.VECTOR.reflection_parity = "odd"  # type: ignore[misc]


def test_repr_names_non_default_fields() -> None:
    assert repr(xrf.VECTOR) == "TensorType(('contravariant',))"
    assert repr(xrf.ORTHOGONAL_PSEUDOVECTOR) == (
        "TensorType(('contravariant',), reflection_parity='odd', requires_orthogonal_map=True)"
    )
    antisymmetric = xrf.TensorType(
        ("covariant", "covariant"), determinant_exponent=0.5, index_symmetry="antisymmetric"
    )
    assert repr(antisymmetric) == (
        "TensorType(('covariant', 'covariant'), determinant_exponent=0.5, "
        "index_symmetry='antisymmetric')"
    )


def test_symmetry_is_accepted_on_equal_variance_order_two() -> None:
    assert xrf.TensorType(("covariant", "covariant"), index_symmetry="symmetric").index_symmetry


# Refusals


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"index_variance": "contravariant"}, TypeError, "sequence of 'contravariant'"),
        ({"index_variance": 3}, TypeError, "sequence of 'contravariant'"),
        ({"index_variance": ("up",)}, ValueError, "not 'up'"),
        ({"index_variance": ("down",)}, ValueError, "not 'down'"),
        ({"index_variance": ("sideways",)}, ValueError, "index_variance must be one of"),
        ({"index_variance": (1,)}, TypeError, "index_variance must be a string"),
        ({"reflection_parity": "positive"}, ValueError, "reflection_parity must be one of"),
        ({"reflection_parity": 1}, TypeError, "reflection_parity must be a string"),
        ({"determinant_exponent": True}, TypeError, "determinant_exponent must be a real"),
        ({"determinant_exponent": "1"}, TypeError, "determinant_exponent must be a real"),
        ({"determinant_exponent": float("nan")}, ValueError, "must be finite"),
        ({"determinant_exponent": float("inf")}, ValueError, "must be finite"),
        ({"index_symmetry": "skew"}, ValueError, "index_symmetry must be one of"),
        ({"requires_orthogonal_map": 1}, TypeError, "requires_orthogonal_map must be a bool"),
    ],
)
def test_invalid_tensor_types_are_refused(
    kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    arguments: dict[str, object] = {"index_variance": ("contravariant",), **kwargs}
    with pytest.raises(error, match=match):
        xrf.TensorType(**arguments)  # type: ignore[arg-type]


def test_symmetry_requires_order_two() -> None:
    with pytest.raises(ValueError, match="order-2 types only"):
        xrf.TensorType(("contravariant",), index_symmetry="symmetric")


def test_symmetry_requires_equal_variance() -> None:
    with pytest.raises(ValueError, match="equal variance"):
        xrf.TensorType(("contravariant", "covariant"), index_symmetry="symmetric")


@pytest.mark.parametrize(
    ("values", "error", "match"),
    [
        (np.ma.masked_array([1.0, 2.0, 3.0], mask=[False, True, False]), TypeError, "masked array"),
        (np.array([True, False, True]), TypeError, "boolean dtype"),
        (np.array(["1", "2", "3"]), TypeError, "dtype <U1"),
        (np.array([1, 2, 3], dtype=object), TypeError, "dtype object"),
        ([[1.0, 2.0], [3.0]], TypeError, "rectangular numeric array"),
    ],
    ids=["masked", "boolean", "string", "object", "ragged"],
)
def test_invalid_values_are_refused(values: object, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        transform(values, SHEAR, xrf.VECTOR)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("linear_map", "error", "match"),
    [
        (np.eye(3)[:2], ValueError, "square matrices"),
        (np.ones(3), ValueError, "square matrices"),
        (np.zeros((0, 0)), ValueError, "at least one row"),
        (np.full((3, 3), np.nan), ValueError, "finite"),
        (np.eye(3) * 1j, TypeError, "real integer or floating"),
        (np.eye(2), ValueError, "has length 3, but linear_map is 2 by 2"),
    ],
    ids=["rectangular", "one-dimensional", "empty", "non-finite", "complex", "size"],
)
def test_invalid_maps_are_refused(linear_map: object, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        transform(np.ones(3), linear_map, xrf.VECTOR)  # type: ignore[arg-type]


def test_tensor_type_must_be_a_tensor_type() -> None:
    with pytest.raises(TypeError, match="tensor_type must be a TensorType"):
        transform(np.ones(3), SHEAR, ("contravariant",))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("component_axes", "error", "match"),
    [
        (0, TypeError, "sequence of integers"),
        ((True,), TypeError, "entries must be integers"),
        ((0.0,), TypeError, "entries must be integers"),
        ((2,), ValueError, "out of range"),
        ((-3,), ValueError, "out of range"),
        ((0, 1), ValueError, "one axis per component index"),
        ((), ValueError, "one axis per component index"),
    ],
)
def test_invalid_component_axes_are_refused(
    component_axes: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        transform(np.ones((2, 3)), SHEAR, xrf.VECTOR, component_axes=component_axes)


def test_repeated_component_axes_are_refused() -> None:
    with pytest.raises(ValueError, match="distinct"):
        transform(
            np.ones((3, 3)), SHEAR, xrf.SECOND_ORDER_CONTRAVARIANT_TENSOR, component_axes=(0, -2)
        )


def test_too_few_axes_for_the_type_are_refused() -> None:
    with pytest.raises(ValueError, match="fewer than the 2 component axes"):
        transform(np.ones(3), SHEAR, xrf.SECOND_ORDER_CONTRAVARIANT_TENSOR)


def test_requires_orthogonal_map_refuses_a_shear() -> None:
    with pytest.raises(ValueError, match="only for orthogonal maps"):
        transform(np.ones(3), SHEAR, xrf.ORTHOGONAL_PSEUDOVECTOR)


def test_requires_orthogonal_map_accepts_rotations_within_tolerance() -> None:
    nearly = MIRROR + 1e-9
    np.testing.assert_allclose(
        transform([1.0, 1.0, 1.0], nearly, xrf.ORTHOGONAL_PSEUDOVECTOR), [1, -1, -1], atol=1e-8
    )


@pytest.mark.parametrize(
    ("linear_map", "error", "match"),
    [
        (np.diag([1.0, 1.0, 0.0]), ValueError, "singular"),
        (np.eye(3)[:2], ValueError, "non-empty square"),
        (np.eye(3) * 1j, TypeError, "real integer or floating"),
    ],
    ids=["singular", "rectangular", "complex"],
)
def test_polar_orthogonal_factor_refuses_invalid_maps(
    linear_map: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        xrf.polar_orthogonal_factor(linear_map)  # type: ignore[arg-type]


def test_docstring_examples_run() -> None:
    import doctest

    import xarrayrf._tensor_type as module

    results = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.attempted > 0 and results.failed == 0
