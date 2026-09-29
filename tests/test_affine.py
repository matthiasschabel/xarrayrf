"""Public behavior of :class:`xarrayrf.AffineTransform`."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from xarrayrf import (
    AffineTransform,
    ArrayCoordinates,
    CoordinateSystem,
    ReferenceFrame,
    transform_named,
)

ATOL = 1e-12
"""Floating-point arithmetic tolerance for these small synthetic affines.

This is a rounding allowance for a handful of multiply-add operations, not a physical
equivalence tolerance; the reference values are exact rationals.
"""


@pytest.fixture
def plane_frame() -> ReferenceFrame:
    """A minted 2-D Cartesian frame in millimetres."""
    return ReferenceFrame.local(CoordinateSystem(("x", "y"), ("mm", "mm")))


@pytest.fixture
def plane_transform(plane_frame: ReferenceFrame) -> AffineTransform:
    """A generic, non-diagonal 2-to-2 affine on row/column index coordinates."""
    return AffineTransform(
        target=plane_frame,
        source=ArrayCoordinates(("row", "column"), ("1", "1")),
        matrix=((2.0, -1.0), (0.5, 3.0)),
        translation=(1.0, -1.0),
    )


def test_generic_affine_evaluates_both_target_axes(plane_transform: AffineTransform) -> None:
    point = transform_named(plane_transform, {"row": 4.0, "column": 5.0})
    assert tuple(point) == ("x", "y")
    # x = 1 + 2*4 - 1*5 = 4; y = -1 + 0.5*4 + 3*5 = 16
    assert_allclose(point["x"], 4.0, rtol=0, atol=ATOL)
    assert_allclose(point["y"], 16.0, rtol=0, atol=ATOL)
    assert point["x"].shape == ()


def test_two_to_three_embedding_places_a_plane_in_a_volume() -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("L", "P", "S"), ("mm", "mm", "mm")))
    transform = AffineTransform(
        target=frame,
        source=ArrayCoordinates(("row", "column"), ("1", "1")),
        matrix=((0.6, 0.0), (0.8, 0.0), (0.0, -1.0)),
        translation=(1.0, 2.0, 3.0),
    )
    point = transform_named(transform, {"row": 2.0, "column": 4.0})
    # L = 1 + 0.6*2 = 2.2; P = 2 + 0.8*2 = 3.6; S = 3 - 1*4 = -1
    assert_allclose(point["L"], 2.2, rtol=0, atol=ATOL)
    assert_allclose(point["P"], 3.6, rtol=0, atol=ATOL)
    assert_allclose(point["S"], -1.0, rtol=0, atol=ATOL)
    assert transform.matrix.shape == (3, 2)


def test_non_square_mapping_accepts_more_source_axes_than_target_axes() -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("u", "v"), ("mm", "mm")))
    transform = AffineTransform(
        target=frame,
        source=ArrayCoordinates(("slice", "row", "column"), ("mm", "1", "1")),
        matrix=((1.0, 0.5, 0.0), (0.0, 0.0, 2.0)),
        translation=(0.0, -1.0),
    )
    point = transform_named(transform, {"slice": 5.0, "row": 4.0, "column": 3.0})
    # u = 0 + 5 + 0.5*4 = 7; v = -1 + 2*3 = 5
    assert_allclose(point["u"], 7.0, rtol=0, atol=ATOL)
    assert_allclose(point["v"], 5.0, rtol=0, atol=ATOL)
    assert transform.matrix.shape == (2, 3)


def test_nonuniform_sample_coordinates_are_read_as_given() -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("S",), ("mm",)))
    transform = AffineTransform(
        target=frame,
        source=ArrayCoordinates(("slice",), ("mm",)),
        matrix=((1.0,),),
        translation=(100.0,),
    )
    point = transform_named(transform, {"slice": [0.0, 2.0, 5.0]})
    # No spacing is inferred: the third sample sits at its declared 5 mm offset.
    assert_allclose(point["S"], [100.0, 102.0, 105.0], rtol=0, atol=ATOL)


def test_scalar_and_array_source_axes_broadcast(plane_transform: AffineTransform) -> None:
    rows = np.array([[0.0], [1.0], [2.0]])
    columns = np.array([0.0, 1.0, 2.0, 3.0])
    point = transform_named(plane_transform, {"row": rows, "column": columns})
    assert point["x"].shape == (3, 4)
    expected_x = 1.0 + 2.0 * rows - 1.0 * columns
    expected_y = -1.0 + 0.5 * rows + 3.0 * columns
    assert_allclose(point["x"], expected_x, rtol=0, atol=ATOL)
    assert_allclose(point["y"], expected_y, rtol=0, atol=ATOL)


def test_a_fixed_scalar_coordinate_broadcasts_against_varying_ones() -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("L", "P", "S"), ("mm", "mm", "mm")))
    transform = AffineTransform(
        target=frame,
        source=ArrayCoordinates(("slice", "row", "column"), ("mm", "1", "1")),
        matrix=((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        translation=(0.0, 0.0, 0.0),
    )
    point = transform_named(transform, {"slice": 5.0, "row": [0.0, 1.0], "column": [2.0, 2.0]})
    assert_allclose(point["S"], [5.0, 5.0], rtol=0, atol=ATOL)
    assert_allclose(point["L"], [0.0, 1.0], rtol=0, atol=ATOL)


def test_one_frame_can_be_shared_by_several_transforms(plane_frame: ReferenceFrame) -> None:
    first = AffineTransform(
        target=plane_frame,
        source=ArrayCoordinates(("row", "column"), ("1", "1")),
        matrix=((1.0, 0.0), (0.0, 1.0)),
        translation=(0.0, 0.0),
    )
    second = AffineTransform(
        target=plane_frame,
        source=ArrayCoordinates(("row", "column"), ("1", "1")),
        matrix=((1.0, 0.0), (0.0, 1.0)),
        translation=(5.0, 0.0),
    )
    assert first.target is second.target
    assert first.target == second.target
    assert first != second


def test_transforms_in_independently_minted_frames_are_unrelated() -> None:
    first = AffineTransform(
        target=ReferenceFrame.local(CoordinateSystem(("x",), ("mm",))),
        source=ArrayCoordinates(("row",), ("1",)),
        matrix=((1.0,),),
        translation=(0.0,),
    )
    second = AffineTransform(
        target=ReferenceFrame.local(CoordinateSystem(("x",), ("mm",))),
        source=ArrayCoordinates(("row",), ("1",)),
        matrix=((1.0,),),
        translation=(0.0,),
    )
    assert first != second


def test_transform_owns_its_coefficients(plane_frame: ReferenceFrame) -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    translation = np.array([1.0, 2.0])
    transform = AffineTransform(
        target=plane_frame,
        source=ArrayCoordinates(("row", "column"), ("1", "1")),
        matrix=matrix,
        translation=translation,
    )
    matrix[0, 0] = 99.0
    translation[0] = 99.0
    assert_allclose(transform.matrix, [[1.0, 0.0], [0.0, 1.0]], rtol=0, atol=ATOL)
    assert_allclose(transform.translation, [1.0, 2.0], rtol=0, atol=ATOL)
    assert not transform.matrix.flags.writeable
    assert not transform.translation.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        transform.matrix[0, 0] = 99.0


def test_a_caller_cannot_reopen_the_published_coefficients(
    plane_transform: AffineTransform,
) -> None:
    """A read-only flag is not the guarantee; ownership of the buffer is.

    Flipping ``writeable`` back on is exactly how an owning array leaks, and it would
    change both the coefficients and the transform's hash.
    """
    before = hash(plane_transform)
    for published in (plane_transform.matrix, plane_transform.translation):
        assert not published.flags.writeable
        assert not published.flags.owndata
        with pytest.raises(ValueError):
            published.flags.writeable = True
        # Walk the base chain: an owning, writeable array anywhere in it is the same leak.
        base: object = published.base
        while isinstance(base, np.ndarray):
            assert not base.flags.writeable
            with pytest.raises(ValueError):
                base.flags.writeable = True
            base = base.base
        # The chain ends in the immutable bytes buffer, not in an array that owns memory.
        assert base is not None
        assert not isinstance(base, np.ndarray)
    assert_allclose(plane_transform.matrix, ((2.0, -1.0), (0.5, 3.0)), rtol=0, atol=ATOL)
    assert_allclose(plane_transform.translation, (1.0, -1.0), rtol=0, atol=ATOL)
    assert hash(plane_transform) == before


@pytest.mark.parametrize("attribute", ("matrix", "translation"))
def test_published_coefficients_are_fresh_per_access(
    plane_transform: AffineTransform,
    attribute: str,
) -> None:
    """Two accesses share no ndarray, so an edit to one cannot reach the other."""
    first = cast(npt.NDArray[np.float64], getattr(plane_transform, attribute))
    second = cast(npt.NDArray[np.float64], getattr(plane_transform, attribute))
    assert first is not second
    assert first.base is not second.base
    assert_allclose(first, second, rtol=0, atol=ATOL)


# The regression below deliberately performs the dtype assignment NumPy has deprecated,
# because that assignment is the leak being guarded against. The filter is scoped to this
# one warning: if NumPy removes the mutation outright, the raised error still fails here.
@pytest.mark.filterwarnings("ignore:Setting the .*dtype.*:DeprecationWarning")
@pytest.mark.parametrize("attribute", ("matrix", "translation"))
def test_published_coefficients_are_isolated_snapshots(
    plane_transform: AffineTransform,
    attribute: str,
) -> None:
    """An immutable buffer stops writes; it does not freeze the array header.

    Reinterpreting the dtype of a published array, or of anything in its base chain, is a
    header edit that a read-only flag permits. If the transform handed out a view of its own
    storage, that edit would land on the transform's coefficients and change what it maps to.
    """
    before_hash = hash(plane_transform)
    before = transform_named(plane_transform, {"row": 4.0, "column": 5.0})
    published = cast(npt.NDArray[np.float64], getattr(plane_transform, attribute))
    expected_shape = published.shape

    # setattr, not assignment, because the dtype setter is absent from the NumPy stubs.
    target: object = published
    while isinstance(target, np.ndarray):
        setattr(target, "dtype", np.dtype(np.float32))  # noqa: B010
        target = target.base

    refetched = cast(npt.NDArray[np.float64], getattr(plane_transform, attribute))
    assert refetched.shape == expected_shape
    assert refetched.dtype == np.dtype(np.float64)
    assert hash(plane_transform) == before_hash
    after = transform_named(plane_transform, {"row": 4.0, "column": 5.0})
    for axis, values in before.items():
        assert_allclose(after[axis], values, rtol=0, atol=ATOL)


def test_evaluation_does_not_write_into_the_caller_values(plane_transform: AffineTransform) -> None:
    rows = np.array([1.0, 2.0])
    columns = np.array([3.0, 4.0])
    transform_named(plane_transform, {"row": rows, "column": columns})
    assert_allclose(rows, [1.0, 2.0], rtol=0, atol=ATOL)
    assert_allclose(columns, [3.0, 4.0], rtol=0, atol=ATOL)


def test_structural_equality_is_exact(plane_frame: ReferenceFrame) -> None:
    def build(corner: float) -> AffineTransform:
        return AffineTransform(
            target=plane_frame,
            source=ArrayCoordinates(("row", "column"), ("1", "1")),
            matrix=((corner, 0.0), (0.0, 1.0)),
            translation=(0.0, 0.0),
        )

    assert build(1.0) == build(1.0)
    assert hash(build(1.0)) == hash(build(1.0))
    # A tolerance here would make compatibility non-transitive, so a perturbation
    # far below any physical significance still has to compare unequal.
    assert build(1.0) != build(1.0 + 1e-15)


def test_negative_zero_does_not_split_identity(plane_frame: ReferenceFrame) -> None:
    def build(sign: float) -> AffineTransform:
        return AffineTransform(
            target=plane_frame,
            source=ArrayCoordinates(("row", "column"), ("1", "1")),
            matrix=((1.0, sign), (0.0, 1.0)),
            translation=(sign, 0.0),
        )

    assert build(0.0) == build(-0.0)
    assert hash(build(0.0)) == hash(build(-0.0))


def test_transform_is_immutable(plane_transform: AffineTransform) -> None:
    for attribute, value in (("source", ("a", "b")), ("extra", 1)):
        with pytest.raises(AttributeError):
            setattr(plane_transform, attribute, value)


def test_comparison_with_a_non_transform_is_not_an_equality(
    plane_transform: AffineTransform,
) -> None:
    other: object = plane_transform.matrix.tolist()
    assert plane_transform != other


@pytest.mark.parametrize(
    ("matrix", "translation", "message"),
    [
        pytest.param(((1.0, 0.0),), (0.0, 0.0), "matrix must be 2-by-2", id="rows"),
        pytest.param(((1.0,), (0.0,)), (0.0, 0.0), "matrix must be 2-by-2", id="columns"),
        pytest.param([[[1.0, 0.0], [0.0, 1.0]]], (0.0, 0.0), "matrix must be", id="rank-3"),
        pytest.param(
            ((1.0, 0.0), (0.0, 1.0)),
            (0.0, 0.0, 0.0),
            "one value per target axis",
            id="translation-length",
        ),
        pytest.param(
            ((1.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (0.0, 0.0)),
            "one value per target axis",
            id="translation-rank",
        ),
        pytest.param(
            ((1.0, 0.0), (0.0, float("inf"))),
            (0.0, 0.0),
            "matrix must contain only finite values",
            id="infinite",
        ),
        pytest.param(
            ((1.0, 0.0), (0.0, float("nan"))),
            (0.0, 0.0),
            "matrix must contain only finite values",
            id="nan",
        ),
        pytest.param(
            ((1.0, 0.0), (0.0,)),
            (0.0, 0.0),
            "convertible to finite float values",
            id="ragged",
        ),
    ],
)
def test_coefficient_shapes_and_values_are_validated(
    plane_frame: ReferenceFrame,
    matrix: npt.ArrayLike,
    translation: npt.ArrayLike,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AffineTransform(
            target=plane_frame,
            source=ArrayCoordinates(("row", "column"), ("1", "1")),
            matrix=matrix,
            translation=translation,
        )


NON_REAL_COEFFICIENTS = [
    pytest.param("identity", id="string-scalar"),
    pytest.param((("1.0", "0.0"), ("0.0", "1.0")), id="numeric-strings"),
    pytest.param(np.array([["1.0", "0.0"], ["0.0", "1.0"]]), id="string-array"),
    pytest.param(((1.0 + 2.0j, 0.0), (0.0, 1.0)), id="complex"),
    pytest.param(np.eye(2, dtype=np.complex128), id="complex-array"),
    pytest.param(((True, False), (False, True)), id="bool"),
    pytest.param(np.eye(2, dtype=bool), id="bool-array"),
    pytest.param(np.array([[None, None], [None, None]], dtype=object), id="object-array"),
]
"""Coefficient values NumPy would silently reinterpret if the dtype were not checked."""


@pytest.mark.parametrize("matrix", NON_REAL_COEFFICIENTS)
def test_matrix_rejects_non_real_dtypes(
    plane_frame: ReferenceFrame,
    matrix: npt.ArrayLike,
) -> None:
    """A numeric string must not be parsed and an imaginary part must not be dropped.

    Both used to pass: the strings were converted and the complex values lost their
    imaginary part with only a ``ComplexWarning``, so the transform held numbers the caller
    never declared.
    """
    with pytest.raises(TypeError, match="real integer or floating values"):
        AffineTransform(
            target=plane_frame,
            source=ArrayCoordinates(("row", "column"), ("1", "1")),
            matrix=matrix,
            translation=(0.0, 0.0),
        )


@pytest.mark.parametrize(
    "translation",
    [
        pytest.param(("0.0", "1.0"), id="numeric-strings"),
        pytest.param((1.0 + 2.0j, 0.0), id="complex"),
        pytest.param((True, False), id="bool"),
        pytest.param(np.array([None, None], dtype=object), id="object-array"),
    ],
)
def test_translation_rejects_non_real_dtypes(
    plane_frame: ReferenceFrame,
    translation: npt.ArrayLike,
) -> None:
    with pytest.raises(TypeError, match="real integer or floating values"):
        AffineTransform(
            target=plane_frame,
            source=ArrayCoordinates(("row", "column"), ("1", "1")),
            matrix=((1.0, 0.0), (0.0, 1.0)),
            translation=translation,
        )


def test_integer_coefficients_are_accepted(plane_frame: ReferenceFrame) -> None:
    """Rejecting non-real dtypes must not reject ordinary integer arithmetic."""
    transform = AffineTransform(
        target=plane_frame,
        source=ArrayCoordinates(("row", "column"), ("1", "1")),
        matrix=np.array([[2, -1], [0, 3]]),
        translation=(1, -1),
    )
    assert transform.matrix.dtype == np.dtype(np.float64)
    point = transform_named(transform, {"row": 4, "column": 5})
    assert_allclose(point["x"], 4.0, rtol=0, atol=ATOL)
    assert_allclose(point["y"], 14.0, rtol=0, atol=ATOL)


def test_mixed_python_sequences_follow_numpy_promotion(plane_frame: ReferenceFrame) -> None:
    """The dtype contract is checked after inference, not on the Python objects supplied.

    ``[True, 2.0]`` infers ``float64`` under ordinary NumPy promotion, so it is accepted
    and ``True`` contributes 1.0. Only a value whose inferred dtype is boolean is refused;
    nothing walks a container hunting for individual boolean elements. Stating this here
    keeps the decision explicit rather than incidental.
    """
    transform = AffineTransform(
        target=plane_frame,
        source=ArrayCoordinates(("row", "column"), ("1", "1")),
        matrix=[[True, 2.0], [0.0, 1.0]],
        translation=(0.0, 0.0),
    )
    assert transform.matrix.dtype == np.dtype(np.float64)
    assert_allclose(transform.matrix, [[1.0, 2.0], [0.0, 1.0]], rtol=0, atol=ATOL)
    point = transform_named(transform, {"row": [True, 2.0], "column": 0.0})
    assert_allclose(point["x"], [1.0, 2.0], rtol=0, atol=ATOL)


@pytest.mark.parametrize("field", ("matrix", "translation"))
def test_coefficients_reject_a_masked_array(plane_frame: ReferenceFrame, field: str) -> None:
    """A mask is the caller's decision to make, and conversion would erase it silently.

    ``np.asarray`` on a masked array returns the underlying data with no warning, so a
    masked coefficient would become whatever value happened to sit beneath the mask.
    """
    matrix: npt.ArrayLike = ((1.0, 0.0), (0.0, 1.0))
    translation: npt.ArrayLike = (0.0, 0.0)
    masked_invalid = cast(
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], np.ma.masked_invalid
    )
    if field == "matrix":
        matrix = masked_invalid(np.array([[1.0, np.nan], [0.0, 1.0]]))
    else:
        translation = masked_invalid(np.array([np.nan, 0.0]))
    with pytest.raises(TypeError, match=f"{field} must not be a masked array"):
        AffineTransform(
            target=plane_frame,
            source=ArrayCoordinates(("row", "column"), ("1", "1")),
            matrix=matrix,
            translation=translation,
        )


def test_transform_named_rejects_a_masked_coordinate_array(
    plane_transform: AffineTransform,
) -> None:
    masked_invalid = cast(
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], np.ma.masked_invalid
    )
    masked = masked_invalid(np.array([1.0, np.nan]))
    with pytest.raises(TypeError, match=r"coordinate 'row'.*must not be a masked array"):
        transform_named(plane_transform, {"row": masked, "column": 0.0})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), [1.0, float("-inf")]])
def test_transform_named_rejects_non_finite_source_axes(
    plane_transform: AffineTransform,
    value: npt.ArrayLike,
) -> None:
    with pytest.raises(ValueError, match="only finite values"):
        transform_named(plane_transform, {"row": value, "column": 0.0})


def test_transform_named_rejects_unconvertible_source_axes(
    plane_transform: AffineTransform,
) -> None:
    with pytest.raises(ValueError, match="convertible to finite float values"):
        transform_named(plane_transform, {"row": [[1.0, 2.0], [3.0]], "column": 0.0})


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("origin", id="string-scalar"),
        pytest.param(["1.0", "2.0"], id="numeric-strings"),
        pytest.param(np.array(["1.0", "2.0"]), id="string-array"),
        pytest.param(1.0 + 2.0j, id="complex"),
        pytest.param(np.array([1.0 + 2.0j, 3.0]), id="complex-array"),
        pytest.param(True, id="bool"),
        pytest.param(np.array([True, False]), id="bool-array"),
        pytest.param(np.array([None, None], dtype=object), id="object-array"),
    ],
)
def test_transform_named_rejects_non_real_coordinate_dtypes(
    plane_transform: AffineTransform,
    value: npt.ArrayLike,
) -> None:
    """Evaluation is the second boundary where a coerced value would change the answer."""
    with pytest.raises(TypeError, match="real integer or floating values"):
        transform_named(plane_transform, {"row": value, "column": 0.0})


def test_transform_named_names_the_offending_coordinate(plane_transform: AffineTransform) -> None:
    with pytest.raises(TypeError, match="coordinate 'column'"):
        transform_named(plane_transform, {"row": 0.0, "column": "origin"})


def test_transform_named_accepts_integer_and_empty_coordinate_arrays(
    plane_transform: AffineTransform,
) -> None:
    integers = transform_named(plane_transform, {"row": np.array([1, 2]), "column": 3})
    assert integers["x"].dtype == np.dtype(np.float64)
    assert_allclose(integers["x"], [0.0, 2.0], rtol=0, atol=ATOL)

    empty = transform_named(plane_transform, {"row": np.array([], dtype=np.float64), "column": 0.0})
    assert empty["x"].shape == (0,)
    assert empty["y"].shape == (0,)


def test_transform_named_rejects_non_broadcastable_source_axes(
    plane_transform: AffineTransform,
) -> None:
    with pytest.raises(ValueError, match="do not broadcast"):
        transform_named(plane_transform, {"row": [1.0, 2.0], "column": [1.0, 2.0, 3.0]})


def test_transform_named_rejects_an_overflowing_evaluation(plane_frame: ReferenceFrame) -> None:
    transform = AffineTransform(
        target=plane_frame,
        source=ArrayCoordinates(("row",), ("1",)),
        matrix=((1e300,), (1.0,)),
        translation=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="non-finite coordinates"):
        transform_named(transform, {"row": 1e300})


def test_transform_point_maps_trailing_axis_points(plane_transform: AffineTransform) -> None:
    points = np.array([[[3.0, 4.0], [0.0, 0.0]]])
    mapped = plane_transform.transform_point(points)
    assert mapped.shape == (1, 2, 2)
    # x = 1 + 2*3 - 4 = 3, y = -1 + 0.5*3 + 3*4 = 12.5; the origin maps to the translation.
    assert_allclose(mapped, [[[3.0, 12.5], [1.0, -1.0]]], rtol=0, atol=ATOL)


def test_transform_point_agrees_with_named_evaluation(plane_transform: AffineTransform) -> None:
    rows = np.array([0.0, 1.5, -2.0])
    columns = np.array([4.0, 0.25, 7.0])
    named = transform_named(plane_transform, {"row": rows, "column": columns})
    stacked = plane_transform.transform_point(np.stack([rows, columns], axis=-1))
    assert_allclose(stacked[..., 0], named["x"], rtol=0, atol=ATOL)
    assert_allclose(stacked[..., 1], named["y"], rtol=0, atol=ATOL)


@pytest.mark.parametrize("points", [5.0, [1.0, 2.0, 3.0], [[1.0], [2.0]]])
def test_transform_point_requires_one_trailing_value_per_source_axis(
    plane_transform: AffineTransform, points: npt.ArrayLike
) -> None:
    with pytest.raises(ValueError, match="trailing axis of length 2"):
        plane_transform.transform_point(points)


def test_transform_point_rejects_non_real_points(plane_transform: AffineTransform) -> None:
    with pytest.raises(TypeError, match="real integer or floating"):
        plane_transform.transform_point(np.array([[1 + 2j, 0.0]]))


def test_jacobian_is_the_constant_matrix(plane_transform: AffineTransform) -> None:
    assert_allclose(plane_transform.jacobian(), [[2.0, -1.0], [0.5, 3.0]], rtol=0, atol=ATOL)
    assert not plane_transform.jacobian().flags.writeable


def test_jacobian_at_points_broadcasts_to_their_leading_shape(
    plane_transform: AffineTransform,
) -> None:
    jacobian = plane_transform.jacobian(np.zeros((4, 3, 2)))
    assert jacobian.shape == (4, 3, 2, 2)
    assert_allclose(jacobian[2, 1], plane_transform.matrix, rtol=0, atol=ATOL)


def test_jacobian_validates_its_points(plane_transform: AffineTransform) -> None:
    with pytest.raises(ValueError, match="trailing axis of length 2"):
        plane_transform.jacobian(np.zeros((3, 3)))


def test_a_sample_locating_affine_starts_at_array_coordinates(
    plane_transform: AffineTransform,
) -> None:
    assert plane_transform.source == ArrayCoordinates(("row", "column"), ("1", "1"))


@pytest.mark.parametrize(
    ("points", "error", "message"),
    [
        pytest.param([[float("nan"), 0.0]], ValueError, "only finite", id="nan"),
        pytest.param([[1e308, 1e308]], ValueError, "non-finite coordinates", id="overflow"),
        pytest.param(
            cast(Callable[..., npt.NDArray[np.float64]], np.ma.masked_array)(
                [[1.0, 2.0]], mask=[[True, False]]
            ),
            TypeError,
            "masked array",
            id="masked",
        ),
    ],
)
def test_transform_point_refuses_unusable_points(
    plane_transform: AffineTransform,
    points: npt.ArrayLike,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        plane_transform.transform_point(points)


def test_jacobian_rejects_non_real_points(plane_transform: AffineTransform) -> None:
    with pytest.raises(TypeError, match="real integer or floating"):
        plane_transform.jacobian(np.array([[1 + 2j, 0.0]]))


def test_array_coordinates_may_be_angular(plane_frame: ReferenceFrame) -> None:
    """Source coordinates are values; an angle is a legitimate source coordinate."""
    transform = AffineTransform(
        source=ArrayCoordinates(("azimuth", "distance"), ("deg", "micrometer")),
        target=plane_frame,
        matrix=((1.0, 0.0), (0.0, 1.0)),
        translation=(0.0, 0.0),
    )
    assert transform.source.units == ("deg", "micrometer")


@pytest.mark.parametrize("field", ["source", "target"])
def test_endpoints_must_be_frames_or_array_coordinates(
    plane_frame: ReferenceFrame, field: str
) -> None:
    endpoints: dict[str, object] = {
        "source": ArrayCoordinates(("row", "column"), ("1", "1")),
        "target": plane_frame,
    }
    endpoints[field] = ("x", "y")
    with pytest.raises(TypeError, match=f"{field} must be a ReferenceFrame or ArrayCoordinates"):
        AffineTransform(
            source=cast(ReferenceFrame, endpoints["source"]),
            target=cast(ReferenceFrame, endpoints["target"]),
            matrix=((1.0, 0.0), (0.0, 1.0)),
            translation=(0.0, 0.0),
        )


def test_a_transform_between_frames_uses_the_frame_axes(plane_frame: ReferenceFrame) -> None:
    moved = AffineTransform(
        source=plane_frame,
        target=plane_frame,
        matrix=((1.0, 0.0), (0.0, 1.0)),
        translation=(5.0, 0.0),
    )
    from_array = AffineTransform(
        source=ArrayCoordinates(("x", "y"), ("mm", "mm")),
        target=plane_frame,
        matrix=((1.0, 0.0), (0.0, 1.0)),
        translation=(5.0, 0.0),
    )
    assert moved.source is plane_frame
    assert moved.source.axes == ("x", "y")
    assert moved != from_array
    assert transform_named(moved, {"x": 1.0, "y": 2.0})["x"] == pytest.approx(6.0, abs=ATOL)
    with pytest.raises(ValueError, match="matrix must be 2-by-3"):
        AffineTransform(
            source=ReferenceFrame.local(CoordinateSystem(("a", "b", "c"), ("mm",) * 3)),
            target=plane_frame,
            matrix=((1.0, 0.0), (0.0, 1.0)),
            translation=(0.0, 0.0),
        )


def test_with_endpoints_keeps_coefficients_and_validates_shape() -> None:
    a = ReferenceFrame.local(CoordinateSystem(("p", "q"), ("mm", "mm")))
    b = ReferenceFrame.local(CoordinateSystem(("p", "q"), ("mm", "mm")))
    transform = AffineTransform(
        source=ArrayCoordinates(("i", "j"), ("1", "1")),
        target=a,
        matrix=[[2.0, 0.5], [0.0, 3.0]],
        translation=[4.0, -1.0],
    )
    moved = transform.with_endpoints(target=b)
    assert moved.target == b and moved.source == transform.source
    assert np.array_equal(moved.matrix, transform.matrix)
    assert np.array_equal(moved.translation, transform.translation)
    renamed = transform.with_endpoints(source=ArrayCoordinates(("row", "col"), ("1", "1")))
    assert renamed.source.axes == ("row", "col") and renamed.target == a
    with pytest.raises(ValueError, match="matrix must be"):
        transform.with_endpoints(source=ArrayCoordinates(("i",), ("1",)))
    with pytest.raises(TypeError):
        transform.with_endpoints(target="frame")  # type: ignore[arg-type]
