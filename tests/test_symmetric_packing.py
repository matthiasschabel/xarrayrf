"""Symmetric packing: explicit component order and off-diagonal scale (value geometry stage 2b)."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

import xarrayrf as xrf

RTOL = ATOL = 1e-12
"""A few float64 operations on O(1) numbers."""

PACKINGS = {
    "voigt-3": xrf.SymmetricPacking.voigt(3),
    "voigt-2": xrf.SymmetricPacking.voigt(2),
    "voigt-strain": xrf.SymmetricPacking.voigt(3, off_diagonal_scale=2.0),
    "mandel-3": xrf.SymmetricPacking.mandel(3),
    "upper-1": xrf.SymmetricPacking.upper_triangular(1),
    "upper-4": xrf.SymmetricPacking.upper_triangular(4),
    "lower-3": xrf.SymmetricPacking.lower_triangular(3),
    "mrtrix": xrf.SymmetricPacking(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))),
}


def symmetric(n: int, batch: tuple[int, ...] = (5,), seed: int = 0) -> npt.NDArray[np.float64]:
    matrices = np.random.default_rng(seed).normal(size=(*batch, n, n))
    return matrices + np.swapaxes(matrices, -1, -2)


@pytest.mark.parametrize(
    ("packing", "labels"),
    [
        (xrf.SymmetricPacking.voigt(3), ("xx", "yy", "zz", "yz", "xz", "xy")),
        (xrf.SymmetricPacking.voigt(2), ("xx", "yy", "xy")),
        (xrf.SymmetricPacking.mandel(3), ("xx", "yy", "zz", "yz", "xz", "xy")),
        (xrf.SymmetricPacking.upper_triangular(3), ("xx", "xy", "xz", "yy", "yz", "zz")),
        (xrf.SymmetricPacking.lower_triangular(3), ("xx", "yx", "yy", "zx", "zy", "zz")),
    ],
    ids=["voigt-3", "voigt-2", "mandel", "upper (FSL)", "lower (NIfTI)"],
)
def test_constructors_give_the_documented_order(
    packing: xrf.SymmetricPacking, labels: tuple[str, ...]
) -> None:
    assert packing.labels(("x", "y", "z")[: packing.dimension]) == labels
    assert packing.component_count == len(labels)


def test_scales_are_one_two_and_root_two() -> None:
    assert xrf.SymmetricPacking.voigt(3).off_diagonal_scale == 1.0
    assert xrf.SymmetricPacking.voigt(3, off_diagonal_scale=2).off_diagonal_scale == 2.0
    assert xrf.SymmetricPacking.mandel(2).off_diagonal_scale == math.sqrt(2.0)


@pytest.mark.parametrize("packing", PACKINGS.values(), ids=PACKINGS.keys())
def test_pack_and_unpack_round_trip(packing: xrf.SymmetricPacking) -> None:
    matrices = symmetric(packing.dimension, batch=(2, 3))
    packed = packing.pack(matrices)
    assert packed.shape == (2, 3, packing.component_count)
    np.testing.assert_allclose(packing.unpack(packed), matrices, rtol=RTOL, atol=ATOL)


def test_off_diagonals_are_scaled_and_diagonals_are_not() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(xrf.SymmetricPacking.voigt(2).pack(matrix), [1.0, 3.0, 2.0])
    np.testing.assert_allclose(
        xrf.SymmetricPacking.voigt(2, off_diagonal_scale=2.0).pack(matrix), [1.0, 3.0, 4.0]
    )
    np.testing.assert_allclose(
        xrf.SymmetricPacking.mandel(2).pack(matrix), [1.0, 3.0, 2.0 * math.sqrt(2.0)]
    )


def test_mandel_keeps_the_frobenius_norm_and_voigt_does_not() -> None:
    matrices = symmetric(3)
    frobenius = np.linalg.norm(matrices, axis=(-2, -1))
    mandel = np.linalg.norm(xrf.SymmetricPacking.mandel(3).pack(matrices), axis=-1)
    voigt = np.linalg.norm(xrf.SymmetricPacking.voigt(3).pack(matrices), axis=-1)
    np.testing.assert_allclose(mandel, frobenius, rtol=RTOL)
    assert not np.allclose(voigt, frobenius)


def test_integer_and_complex_values_keep_a_floating_dtype() -> None:
    packing = xrf.SymmetricPacking.voigt(2)
    assert packing.unpack(np.array([1, 2, 3])).dtype == np.float64
    assert packing.pack(np.array([[1, 2], [2, 3]])).dtype == np.float64
    complex_matrix = np.array([[1, 2j], [2j, 3]])
    assert packing.pack(complex_matrix).dtype == np.complex128
    np.testing.assert_allclose(packing.unpack(packing.pack(complex_matrix)), complex_matrix)


def test_rounding_level_asymmetry_is_packed_as_the_mean() -> None:
    matrix = np.array([[1.0, 2.0], [2.0 + 1e-12, 3.0]])
    np.testing.assert_allclose(
        xrf.SymmetricPacking.voigt(2).pack(matrix), [1.0, 3.0, 2.0 + 5e-13], rtol=0, atol=1e-15
    )


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 2.0], [2.5, 3.0]],
        [[0.0, np.nan], [1.0, 0.0]],
        [[0.0, np.inf], [1.0, 0.0]],
        [[0.0, np.inf], [-np.inf, 0.0]],
        [[np.nan, 1.0], [2.0, 0.0]],
        [[np.inf, 1.0], [2.0, 0.0]],
    ],
    ids=[
        "finite",
        "missing against known",
        "inf against finite",
        "inf against -inf",
        "NaN diagonal hides nothing",
        "inf diagonal hides nothing",
    ],
)
def test_pack_refuses_mirrored_entries_that_disagree(matrix: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="the matrix is not symmetric"):
        xrf.SymmetricPacking.voigt(2).pack(np.array(matrix))


def test_pack_names_the_first_asymmetric_matrix() -> None:
    matrices = symmetric(2, batch=(2, 3))
    matrices[1, 2, 0, 1] += 1.0
    with pytest.raises(ValueError, match=r"matrix \(1, 2\) is not symmetric"):
        xrf.SymmetricPacking.voigt(2).pack(matrices)


def test_pack_accepts_mirrored_missing_and_infinite_entries() -> None:
    packing = xrf.SymmetricPacking.voigt(2)
    np.testing.assert_array_equal(
        packing.pack(np.array([[np.nan, np.inf], [np.inf, 1.0]])), [np.nan, 1.0, np.inf]
    )
    assert np.isnan(packing.pack(np.full((2, 2), np.nan))).all()


def test_large_finite_entries_do_not_overflow_when_packed() -> None:
    packing = xrf.SymmetricPacking.voigt(2)
    matrix = np.array([[0.0, 1e308], [1e308, 0.0]])
    np.testing.assert_array_equal(packing.unpack(packing.pack(matrix)), matrix)
    uneven = np.array([[0.0, 1.5e308], [1.5e308 * (1 + 1e-9), 0.0]])
    assert np.isfinite(packing.pack_symmetric_part(uneven)).all()


def test_the_symmetric_part_is_packed_without_a_check() -> None:
    matrix = np.array([[1.0, 2.0], [4.0, 3.0]])
    np.testing.assert_allclose(
        xrf.SymmetricPacking.voigt(2).pack_symmetric_part(matrix), [1.0, 3.0, 3.0]
    )


def test_equality_and_hashing_are_exact() -> None:
    voigt = xrf.SymmetricPacking.voigt(3)
    assert voigt == xrf.SymmetricPacking(voigt.index_pairs)
    assert hash(voigt) == hash(xrf.SymmetricPacking(voigt.index_pairs))
    assert voigt != xrf.SymmetricPacking.mandel(3)
    assert voigt != xrf.SymmetricPacking.upper_triangular(3)
    assert voigt != "voigt"
    assert "off_diagonal_scale=1.0" in repr(voigt)


def test_coinciding_labels_are_refused() -> None:
    with pytest.raises(ValueError, match="coinciding packed labels"):
        # (1, 0) and (0, 2) both read "aba".
        packing = xrf.SymmetricPacking(((0, 0), (1, 1), (2, 2), (1, 0), (0, 2), (1, 2)))
        packing.labels(("a", "ab", "ba"))


def test_labels_need_one_axis_per_dimension() -> None:
    with pytest.raises(ValueError, match="2-dimensional, but the basis has 3 axes"):
        xrf.SymmetricPacking.voigt(2).labels(("x", "y", "z"))


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ({"index_pairs": "xy"}, TypeError, "non-empty sequence"),
        ({"index_pairs": ()}, TypeError, "non-empty sequence"),
        ({"index_pairs": ((0, 0, 0),)}, TypeError, "pair of integers"),
        ({"index_pairs": ((0.0, 0),)}, TypeError, "pair of integers"),
        ({"index_pairs": ((True, 0),)}, TypeError, "pair of integers"),
        ({"index_pairs": ((-1, 0),)}, ValueError, "non-negative"),
        ({"index_pairs": ((0, 0), (1, 1), (0, 1), (1, 0))}, ValueError, "repeats"),
        ({"index_pairs": ((0, 0), (1, 1))}, ValueError, "packs into 3 components, got 2"),
        ({"index_pairs": ((0, 0),), "off_diagonal_scale": True}, TypeError, "real number"),
        ({"index_pairs": ((0, 0),), "off_diagonal_scale": 0.0}, ValueError, "positive"),
        ({"index_pairs": ((0, 0),), "off_diagonal_scale": math.inf}, ValueError, "finite"),
    ],
)
def test_invalid_packings_are_refused(
    arguments: dict[str, object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        xrf.SymmetricPacking(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda: xrf.SymmetricPacking.voigt(4), ValueError, "dimension 2 or 3"),
        (lambda: xrf.SymmetricPacking.upper_triangular(0), ValueError, "at least 1"),
        (
            lambda: xrf.SymmetricPacking.lower_triangular(2.0),  # type: ignore[arg-type]
            TypeError,
            "integer",
        ),
    ],
)
def test_invalid_constructor_arguments_are_refused(
    call: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        call()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda p: p.unpack(np.ones(4)), ValueError, "3 packed components"),
        (lambda p: p.unpack(np.float64(1.0)), ValueError, "3 packed components"),
        (lambda p: p.pack(np.ones((3, 3))), ValueError, r"2 x 2 matrices"),
        (lambda p: p.pack(np.ones(2)), ValueError, r"2 x 2 matrices"),
        (lambda p: p.pack(np.ones((2, 2), dtype=bool)), TypeError, "boolean"),
        (lambda p: p.unpack(np.ma.masked_array(np.ones(3))), TypeError, "masked"),
    ],
)
def test_invalid_values_are_refused(call: object, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        call(xrf.SymmetricPacking.voigt(2))  # type: ignore[operator]


def test_docstring_examples_run() -> None:
    import doctest

    import xarrayrf._symmetric_packing as module

    results = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.attempted > 0 and results.failed == 0
