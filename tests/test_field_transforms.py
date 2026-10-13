"""Public behaviour of the field transforms, their Jacobians and the determinant report."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

import xarrayrf as xrf

ATOL = 1e-12
"""Roundoff allowance for float64 evaluation of small synthetic fields."""

FIXTURE_ATOL = 1e-9
"""Agreement with ITK, which evaluates the same arithmetic in a different order."""

DIFFERENCE_STEP = 1e-6
"""Central-difference step in millimetres; its truncation error on these smooth fields is far
below the comparison tolerance below."""

DIFFERENCE_ATOL = 1e-7
"""Central-difference agreement: roundoff of about eps / step, and the step's truncation."""

SYSTEM = xrf.CoordinateSystem(("x", "y", "z"), ("mm",) * 3)
FIXED = xrf.ReferenceFrame.declared(("test", "fixed"), SYSTEM)
MOVING = xrf.ReferenceFrame.declared(("test", "moving"), SYSTEM)
INDEX = xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3, sample_offset=(0.5,) * 3)
POINTS = xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3)
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "itk_field_boundaries.json").read_text(encoding="utf-8")
)


def grid(
    shape: tuple[int, ...] = (4, 5, 3),
    *,
    frame: xrf.ReferenceFrame = FIXED,
    matrix: npt.ArrayLike | None = None,
    translation: npt.ArrayLike | None = None,
    coordinates: dict[str, npt.ArrayLike] | None = None,
    intervals: dict[str, npt.ArrayLike] | None = None,
    source: xrf.ArrayCoordinates = INDEX,
) -> xrf.Grid:
    """A grid with dims (i, j, k) along source axes (i, j, k), unit index coordinates by default."""
    transform = xrf.AffineTransform.from_matrix(
        source=source,
        target=frame,
        matrix=np.eye(3) if matrix is None else matrix,
        translation=np.zeros(3) if translation is None else translation,
    )
    values: dict[str, npt.ArrayLike] = {
        axis: np.arange(n) for axis, n in zip("ijk", shape, strict=True)
    }
    values.update(coordinates or {})
    return xrf.Grid(transform, {axis: (axis, values[axis]) for axis in "ijk"}, intervals=intervals)


def field_grid(case: dict[str, Any]) -> xrf.Grid:
    """The fixture's ITK grid: physical = origin + direction · diag(spacing) · index."""
    matrix = np.asarray(case["direction"]) * np.asarray(case["spacing"])
    shape = tuple(case["shape"])
    intervals: dict[str, npt.ArrayLike] | None = {"k": [[-0.5, 0.5]]} if shape[2] == 1 else None
    return grid(shape, matrix=matrix, translation=case["origin"], intervals=intervals)


def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def oblique() -> npt.NDArray[np.float64]:
    angle = 0.4
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    return cast(npt.NDArray[np.float64], rotation @ np.diag([0.8, 1.3, 2.0]))


def dense(
    field: npt.ArrayLike, g: xrf.Grid, outside: Any = "raise"
) -> xrf.DisplacementFieldTransform:
    return xrf.DisplacementFieldTransform(
        source=FIXED, target=MOVING, field=field, grid=g, outside=outside
    )


def spline(
    coefficients: npt.ArrayLike, g: xrf.Grid, outside: Any = "raise"
) -> xrf.BSplineTransform:
    return xrf.BSplineTransform(
        source=FIXED, target=MOVING, coefficients=coefficients, grid=g, outside=outside
    )


def central_differences(transform: Any, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    columns = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = DIFFERENCE_STEP
        columns.append(
            (transform.transform_point(points + step) - transform.transform_point(points - step))
            / (2 * DIFFERENCE_STEP)
        )
    return np.stack(columns, axis=-1)


# --- Agreement with ITK -----------------------------------------------------------------------


def _away_from_edges(
    index: npt.NDArray[np.float64], low: float, high: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Probes at least 1e-9 from every edge, whose classification no roundoff can change."""
    return np.all((np.abs(index - low) > 1e-9) & (np.abs(index - high) > 1e-9), axis=1)


@pytest.mark.parametrize("case", ["dense", "slab"])
def test_a_dense_field_matches_itk_inside_and_at_its_support_edges(case: str) -> None:
    data = FIXTURE[case]
    transform = dense(data["field"], field_grid(data), outside="identity")
    points, expected = np.asarray(data["points"]), np.asarray(data["mapped"])
    index = np.asarray(data["index"])
    keep = _away_from_edges(index, -0.5, np.asarray(data["shape"]) - 0.5)
    assert keep.sum() >= 30
    assert_allclose(transform.transform_point(points[keep]), expected[keep], atol=FIXTURE_ATOL)
    # Exact-edge probes on an oblique grid depend on each library's roundoff; where both call
    # them inside, the values agree.
    mapped = transform.transform_point(points[~keep])
    both = np.any(mapped != points[~keep], axis=1) & np.any(
        expected[~keep] != points[~keep], axis=1
    )
    assert_allclose(mapped[both], expected[~keep][both], atol=FIXTURE_ATOL)


def test_a_bspline_matches_itk_inside_and_at_its_support_edges() -> None:
    data = FIXTURE["bspline"]
    transform = spline(data["coefficients"], field_grid(data), outside="identity")
    points, expected = np.asarray(data["points"]), np.asarray(data["mapped"])
    keep = _away_from_edges(np.asarray(data["index"]), 1.0, np.asarray(data["shape"]) - 2.0)
    assert_allclose(transform.transform_point(points[keep]), expected[keep], atol=FIXTURE_ATOL)
    # On this axis-aligned grid the exact edges are computed exactly: closed at both ends.
    assert_allclose(transform.transform_point(points[~keep]), expected[~keep], atol=FIXTURE_ATOL)


def test_bspline_values_match_unfiltered_cubic_interpolation() -> None:
    ndimage = pytest.importorskip("scipy.ndimage")
    coefficients = rng().normal(size=(6, 7, 5, 3))
    g = grid((6, 7, 5), matrix=oblique(), translation=(3.0, -1.0, 2.0), source=POINTS)
    transform = spline(coefficients, g)
    index = rng().uniform([1, 1, 1], [4, 5, 3], size=(200, 3))
    points = index @ oblique().T + np.array([3.0, -1.0, 2.0])
    expected = points + np.stack(
        [
            ndimage.map_coordinates(coefficients[..., c], index.T, order=3, prefilter=False)
            for c in range(3)
        ],
        axis=-1,
    )
    assert_allclose(transform.transform_point(points), expected, atol=1e-10)


# --- Exact support edges ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("coordinate", "inside"),
    [
        (-0.5, True),
        (np.nextafter(-0.5, -np.inf), False),
        (-0.5 - 1e-10, False),
        (np.nextafter(3.5, -np.inf), True),
        (3.5 - 1e-10, True),
        (3.5, False),
    ],
)
def test_a_dense_support_is_closed_below_and_open_above(coordinate: float, inside: bool) -> None:
    transform = dense(np.ones((4, 4, 4, 3)), grid((4, 4, 4)))
    point = np.array([coordinate, 1.0, 1.0])
    if inside:
        assert_allclose(transform.transform_point(point), point + 1.0)
    else:
        with pytest.raises(ValueError, match="outside the displacement field's support"):
            transform.transform_point(point)


@pytest.mark.parametrize(
    ("coordinate", "inside"),
    [
        (1.0, True),
        (np.nextafter(1.0, -np.inf), False),
        (1.0 - 1e-10, False),
        (4.0, True),
        (np.nextafter(4.0, np.inf), False),
        (4.0 + 1e-10, False),
    ],
)
def test_a_bspline_support_is_closed_at_both_ends(coordinate: float, inside: bool) -> None:
    transform = spline(np.ones((6, 6, 6, 3)), grid((6, 6, 6), source=POINTS))
    point = np.array([2.5, coordinate, 2.5])
    if inside:
        assert_allclose(transform.transform_point(point), point + 1.0, atol=ATOL)
    else:
        with pytest.raises(ValueError, match="outside the B-spline transform's support"):
            transform.transform_point(point)


@pytest.mark.parametrize(
    ("coordinate", "inside"),
    [
        (1.0, True),
        (np.nextafter(1.0, -np.inf), False),
        (np.nextafter(4.0, -np.inf), True),
        (4.0, False),
    ],
)
def test_a_half_open_bspline_support_excludes_its_upper_face(
    coordinate: float, inside: bool
) -> None:
    transform = xrf.BSplineTransform(
        source=FIXED,
        target=MOVING,
        coefficients=np.ones((6, 6, 6, 3)),
        grid=grid((6, 6, 6), source=POINTS),
        support="half-open",
    )
    assert transform.support == "half-open"
    point = np.array([2.5, coordinate, 2.5])
    if inside:
        assert_allclose(transform.transform_point(point), point + 1.0, atol=ATOL)
    else:
        with pytest.raises(ValueError, match=r"j: \[1, 4\)"):
            transform.transform_point(point)
    # "nearest" still evaluates on the closed face.
    nearest = xrf.BSplineTransform(
        source=FIXED,
        target=MOVING,
        coefficients=np.ones((6, 6, 6, 3)),
        grid=grid((6, 6, 6), source=POINTS),
        outside="nearest",
        support="half-open",
    )
    assert_allclose(nearest.transform_point(point), point + 1.0, atol=ATOL)


def test_the_support_declaration_is_checked_kept_and_shown() -> None:
    options: dict[str, Any] = {
        "source": FIXED,
        "target": MOVING,
        "coefficients": np.ones((6, 6, 6, 3)),
        "grid": grid((6, 6, 6), source=POINTS),
    }
    assert xrf.BSplineTransform(**options).support == "closed"
    assert "support" not in repr(xrf.BSplineTransform(**options))
    with pytest.raises(ValueError, match="support must be 'closed' or 'half-open'"):
        xrf.BSplineTransform(**options, support=cast(Any, "open"))
    half = xrf.BSplineTransform(**options, support="half-open")
    assert repr(half).endswith("outside='raise', support='half-open')")
    assert pickle.loads(pickle.dumps(half)).support == "half-open"


# --- Analytic agreement -----------------------------------------------------------------------


def test_a_constant_field_is_the_translation() -> None:
    offset = np.array([1.5, -2.0, 0.25])
    transform = dense(np.broadcast_to(offset, (4, 5, 3, 3)), grid())
    translation = xrf.AffineTransform.from_matrix(
        source=FIXED, target=MOVING, matrix=np.eye(3), translation=offset
    )
    points = rng().uniform(-0.5, [3.4, 4.4, 2.4], size=(50, 3))
    assert_allclose(transform.transform_point(points), translation.transform_point(points))


def test_a_field_sampled_from_an_affine_reproduces_it_between_the_samples() -> None:
    affine = xrf.AffineTransform.from_matrix(
        source=FIXED, target=MOVING, matrix=oblique(), translation=(1.0, 2.0, 3.0)
    )
    g = grid((5, 6, 4), matrix=np.diag([1.5, 1.0, 2.0]), translation=(-2.0, 0.0, 1.0))
    transform = xrf.sample_displacement_field(affine, g, outside="identity")
    assert transform.field.dtype == np.float64
    points = np.array([-2.0, 0.0, 1.0]) + rng().uniform(0, [6.0, 5.0, 6.0], size=(100, 3))
    assert_allclose(transform.transform_point(points), affine.transform_point(points), atol=1e-12)
    assert_allclose(transform.jacobian(points), np.broadcast_to(oblique(), (100, 3, 3)), atol=1e-12)


def test_rasterizing_a_bspline_agrees_at_the_samples() -> None:
    coefficients = rng().normal(scale=0.3, size=(6, 6, 6, 3))
    source = spline(coefficients, grid((6, 6, 6), source=POINTS))
    target_grid = grid((3, 3, 3), translation=(1.5, 1.5, 1.5))
    sampled = xrf.sample_displacement_field(source, target_grid)
    points = target_grid.points().reshape(-1, 3)
    assert_allclose(sampled.transform_point(points), source.transform_point(points), atol=ATOL)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_dense_jacobians_match_central_differences(dtype: type) -> None:
    field: npt.NDArray[Any] = rng().normal(scale=0.3, size=(5, 4, 4, 3)).astype(dtype)
    # Nonuniform along i, descending along k, oblique: every term of the chain rule is exercised.
    g = grid(
        (5, 4, 4),
        matrix=oblique(),
        translation=(1.0, 2.0, -1.0),
        coordinates={"i": [0.0, 1.0, 3.0, 4.5, 7.0], "k": [3, 2, 1, 0]},
    )
    transform = dense(field, g)
    assert transform.field.dtype == dtype
    # Cell interiors, away from the faces where the one-sided derivative changes.
    cells = rng().integers(0, 3, size=(40, 3))
    fractions = rng().uniform(0.2, 0.8, size=(40, 3))
    i_values = np.array([0.0, 1.0, 3.0, 4.5, 7.0])
    i = i_values[cells[:, 0]] + fractions[:, 0] * np.diff(i_values)[cells[:, 0]]
    coordinates = np.stack(
        (i, cells[:, 1] + fractions[:, 1], 3 - cells[:, 2] - fractions[:, 2]), -1
    )
    points = coordinates @ oblique().T + np.array([1.0, 2.0, -1.0])
    assert_allclose(
        transform.jacobian(points), central_differences(transform, points), atol=DIFFERENCE_ATOL
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_bspline_jacobians_match_central_differences(dtype: type) -> None:
    coefficients: npt.NDArray[Any] = rng().normal(scale=0.3, size=(6, 7, 5, 3)).astype(dtype)
    g = grid((6, 7, 5), matrix=oblique(), translation=(0.5, 0.0, 2.0), source=POINTS)
    transform = spline(coefficients, g)
    index = rng().uniform([1.01, 1.01, 1.01], [3.99, 4.99, 2.99], size=(60, 3))
    points = index @ oblique().T + np.array([0.5, 0.0, 2.0])
    assert_allclose(
        transform.jacobian(points), central_differences(transform, points), atol=DIFFERENCE_ATOL
    )


def test_a_slab_jacobian_has_no_derivative_across_the_slab() -> None:
    data = FIXTURE["slab"]
    transform = dense(data["field"], field_grid(data))
    point = np.array([1.3, 2.2, -4.0 + 0.4 * 2.5])
    jacobian = transform.jacobian(point)
    assert_allclose(jacobian[:, 2], [0.0, 0.0, 1.0])
    assert_allclose(
        jacobian[:, :2], central_differences(transform, point)[:, :2], atol=DIFFERENCE_ATOL
    )


def test_a_dense_derivative_uses_the_upper_cell_on_a_face_and_the_lower_at_the_end() -> None:
    field = np.zeros((3, 1, 1, 3))
    field[:, 0, 0, 0] = [0.0, 1.0, 3.0]
    transform = dense(field, grid((3, 1, 1), intervals={"j": [[-0.5, 0.5]], "k": [[-0.5, 0.5]]}))
    assert transform.jacobian(np.array([1.0, 0.0, 0.0]))[0, 0] == pytest.approx(3.0)
    assert transform.jacobian(np.array([2.0, 0.0, 0.0]))[0, 0] == pytest.approx(3.0)
    assert transform.jacobian(np.array([0.0, 0.0, 0.0]))[0, 0] == pytest.approx(2.0)
    # Beyond the outer sample, inside the cell, the edge value extends as a constant.
    assert transform.jacobian(np.array([2.3, 0.0, 0.0]))[0, 0] == pytest.approx(1.0)
    assert_allclose(transform.transform_point(np.array([2.3, 0.0, 0.0])), [5.3, 0.0, 0.0])


def test_values_follow_the_grid_dimension_order_not_the_source_axis_order() -> None:
    transform = xrf.AffineTransform.from_matrix(
        source=INDEX, target=FIXED, matrix=oblique(), translation=(1.0, -2.0, 0.5)
    )
    sizes = {"i": 5, "j": 3, "k": 4}
    permuted = xrf.Grid(transform, {axis: (axis, np.arange(sizes[axis])) for axis in "kij"})
    assert permuted.dims == ("k", "i", "j")
    # u_x grows along i only, u_y along k only, so a swapped axis changes both value and Jacobian.
    k, i, _ = np.meshgrid(np.arange(4), np.arange(5), np.arange(3), indexing="ij")
    field = np.stack((2.0 * i, -1.0 * k, np.zeros(k.shape)), axis=-1)
    warp = dense(field, permuted)
    index = rng().uniform(0, [4, 2, 3], size=(30, 3))  # (i, j, k), inside the samples
    points = index @ oblique().T + np.array([1.0, -2.0, 0.5])
    expected = points + np.stack((2.0 * index[:, 0], -1.0 * index[:, 2], 0 * index[:, 0]), -1)
    assert_allclose(warp.transform_point(points), expected, atol=1e-12)
    gradient = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    jacobian = np.eye(3) + gradient @ np.linalg.inv(oblique())
    assert_allclose(warp.jacobian(points), np.broadcast_to(jacobian, (30, 3, 3)), atol=1e-12)


# --- Exterior ---------------------------------------------------------------------------------


def test_the_raise_exterior_names_the_point_and_the_support() -> None:
    transform = dense(np.zeros((4, 5, 3, 3)), grid())
    points = np.zeros((2, 2, 3))
    points[1, 0] = (10.0, 0.0, 0.0)
    with pytest.raises(ValueError, match=r"point \(1, 0\) lies outside .* i: \[-0\.5, 3\.5\)"):
        transform.transform_point(points)


def test_the_identity_exterior_leaves_points_and_jacobians_unchanged() -> None:
    transform = dense(np.ones((4, 5, 3, 3)), grid(), outside="identity")
    points = np.array([[10.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert_allclose(transform.transform_point(points), [[10.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    assert_allclose(transform.jacobian(points[0]), np.eye(3))


def test_the_nearest_exterior_extends_the_edge_displacement() -> None:
    field = rng().normal(size=(4, 5, 3, 3))
    transform = dense(field, grid(), outside="nearest")
    point = np.array([12.0, 2.0, 1.0])
    assert_allclose(transform.transform_point(point), point + field[3, 2, 1])
    jacobian = transform.jacobian(point)
    assert_allclose(jacobian[:, 0], [1.0, 0.0, 0.0])


def test_the_nearest_exterior_of_a_bspline_evaluates_at_the_support_edge() -> None:
    coefficients = rng().normal(size=(6, 6, 6, 3))
    transform = spline(coefficients, grid((6, 6, 6), source=POINTS), outside="nearest")
    edge = transform.transform_point(np.array([4.0, 2.0, 3.0])) - np.array([4.0, 2.0, 3.0])
    beyond = transform.transform_point(np.array([9.0, 2.0, 3.0])) - np.array([9.0, 2.0, 3.0])
    assert_allclose(beyond, edge, atol=ATOL)
    assert_allclose(transform.jacobian(np.array([9.0, 2.0, 3.0]))[:, 0], [1.0, 0.0, 0.0])


def test_the_jacobian_needs_points() -> None:
    with pytest.raises(ValueError, match="at is required"):
        dense(np.zeros((4, 5, 3, 3)), grid()).jacobian()


@pytest.mark.parametrize("points", [np.zeros(2), np.zeros((3, 2))])
def test_points_need_one_value_per_source_axis(points: npt.NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="trailing axis of length 3"):
        dense(np.zeros((4, 5, 3, 3)), grid()).transform_point(points)


def test_points_must_be_finite_and_real() -> None:
    transform = dense(np.zeros((4, 5, 3, 3)), grid())
    with pytest.raises(ValueError, match="finite"):
        transform.transform_point([np.nan, 0.0, 0.0])
    with pytest.raises(TypeError, match="real"):
        transform.transform_point(np.zeros(3, dtype=complex))


# --- Construction -----------------------------------------------------------------------------


def test_a_descending_axis_with_tiling_intervals_is_accepted() -> None:
    g = grid(
        (2, 1, 1),
        coordinates={"i": [1, 0]},
        intervals={"i": [[0.5, 1.5], [-0.5, 0.5]], "j": [[-0.5, 0.5]], "k": [[-0.5, 0.5]]},
    )
    transform = dense(np.ones((2, 1, 1, 3)), g)
    assert_allclose(transform.transform_point([1.4, 0.0, 0.0]), [2.4, 1.0, 1.0])


@pytest.mark.parametrize(
    "rows", [[[-0.5, 0.5], [0.6, 1.4]], [[-0.6, 0.6], [0.5, 1.5]]], ids=["gap", "overlap"]
)
def test_interval_gaps_and_overlaps_are_refused(rows: list[list[float]]) -> None:
    g = grid((2, 1, 1), intervals={"i": rows, "j": [[-0.5, 0.5]], "k": [[-0.5, 0.5]]})
    with pytest.raises(ValueError, match="leave gaps or overlap"):
        dense(np.zeros((2, 1, 1, 3)), g)


def test_a_nearly_uniform_bspline_grid_is_refused() -> None:
    values = np.arange(6.0)
    values[3] += 1e-9
    g = grid((6, 6, 6), coordinates={"i": values}, source=POINTS)
    with pytest.raises(ValueError, match="not uniform"):
        spline(np.zeros((6, 6, 6, 3)), g)


@pytest.mark.parametrize(
    ("make", "error", "match"),
    [
        (lambda: grid(source=POINTS), ValueError, "declares point samples along 'i'"),
        (lambda: grid((4, 5, 1)), ValueError, "single sample along 'k'"),
        (lambda: grid(coordinates={"i": [0, 2, 1, 3]}), ValueError, "not strictly monotonic"),
        (lambda: grid(frame=MOVING), ValueError, "sampled in its source frame"),
        (lambda: grid(matrix=np.zeros((3, 3))), ValueError, "zero row"),
    ],
)
def test_inadmissible_dense_grids_are_refused(make: Any, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        dense(np.zeros((4, 5, 3, 3)), make())


def test_a_grid_with_a_retained_scalar_is_refused() -> None:
    transform = xrf.AffineTransform.from_matrix(
        source=INDEX, target=FIXED, matrix=np.eye(3), translation=np.zeros(3)
    )
    g = xrf.Grid(transform, {"i": ("i", np.arange(4)), "j": ("j", np.arange(5)), "k": 0})
    with pytest.raises(ValueError, match="retains scalar axes"):
        dense(np.zeros((4, 5, 3)), g)


def test_a_non_square_grid_is_refused() -> None:
    plane = xrf.ArrayCoordinates(("i", "j"), ("1",) * 2, sample_offset=(0.5,) * 2)
    transform = xrf.AffineTransform.from_matrix(
        source=plane, target=FIXED, matrix=np.eye(3)[:, :2], translation=np.zeros(3)
    )
    g = xrf.Grid(transform, {"i": ("i", np.arange(4)), "j": ("j", np.arange(5))})
    with pytest.raises(ValueError, match="must be square"):
        dense(np.zeros((4, 5, 3)), g)


def test_a_non_affine_grid_transform_is_refused() -> None:
    affine = xrf.AffineTransform.from_matrix(
        source=INDEX, target=FIXED, matrix=np.eye(3), translation=np.zeros(3)
    )
    g = xrf.Grid(xrf.CompositeTransform(affine), {axis: (axis, np.arange(4)) for axis in "ijk"})
    with pytest.raises(ValueError, match="transform must be affine"):
        dense(np.zeros((4, 4, 4, 3)), g)


def test_a_grid_must_be_a_grid() -> None:
    with pytest.raises(TypeError, match="grid must be a Grid"):
        dense(np.zeros((4, 5, 3, 3)), cast(xrf.Grid, object()))


def test_bspline_control_grids_need_four_points_and_order_three() -> None:
    with pytest.raises(ValueError, match="at least four control points"):
        spline(np.zeros((3, 6, 6, 3)), grid((3, 6, 6), source=POINTS))
    g = grid((6, 6, 6), source=POINTS)
    with pytest.raises(NotImplementedError, match="order 3"):
        xrf.BSplineTransform(
            source=FIXED, target=MOVING, coefficients=np.zeros((6, 6, 6, 3)), grid=g, order=2
        )
    with pytest.raises(TypeError, match="order must be an integer"):
        xrf.BSplineTransform(
            source=FIXED,
            target=MOVING,
            coefficients=np.zeros((6, 6, 6, 3)),
            grid=g,
            order=cast(int, True),
        )


def test_endpoints_must_be_cartesian_frames_with_equal_systems() -> None:
    g = grid()
    values = np.zeros((4, 5, 3, 3))
    with pytest.raises(TypeError, match="target must be a ReferenceFrame"):
        xrf.DisplacementFieldTransform(
            source=FIXED, target=cast(xrf.ReferenceFrame, INDEX), field=values, grid=g
        )
    metres = xrf.ReferenceFrame.declared(
        ("test", "metres"), xrf.CoordinateSystem(("x", "y", "z"), ("m",) * 3)
    )
    with pytest.raises(ValueError, match="equal coordinate systems"):
        xrf.DisplacementFieldTransform(source=FIXED, target=metres, field=values, grid=g)
    sphere = xrf.ReferenceFrame.declared(
        ("test", "sphere"),
        xrf.CoordinateSystem(
            ("lon", "lat", "r"),
            ("degrees_east", "degrees_north", "m"),
            representation=xrf.SPHERICAL,
            axis_roles=("longitude", "latitude", "radius"),
        ),
    )
    with pytest.raises(ValueError, match="spherical coordinate system"):
        xrf.DisplacementFieldTransform(source=sphere, target=sphere, field=values, grid=g)


@pytest.mark.parametrize(
    ("values", "error", "match"),
    [
        (np.zeros((4, 5, 3)), ValueError, r"must have shape \(4, 5, 3, 3\)"),
        (np.full((4, 5, 3, 3), np.inf), ValueError, "finite"),
        (np.zeros((4, 5, 3, 3), dtype=complex), TypeError, "real integer or floating"),
        (np.zeros((4, 5, 3, 3), dtype=bool), TypeError, "real integer or floating"),
        (np.ma.zeros((4, 5, 3, 3)), TypeError, "masked"),
        ([[1, 2], [3]], ValueError, "rectangular"),
    ],
)
def test_field_values_are_validated(values: Any, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        dense(values, grid())


def test_an_unknown_exterior_is_refused() -> None:
    with pytest.raises(ValueError, match="outside must be one of"):
        dense(np.zeros((4, 5, 3, 3)), grid(), outside=cast(Any, "extrapolate"))


def test_integer_fields_are_stored_as_float64() -> None:
    assert dense(np.zeros((4, 5, 3, 3), dtype=np.int16), grid()).field.dtype == np.float64


def test_stored_values_are_copied_and_cannot_be_made_writeable() -> None:
    values = np.zeros((4, 5, 3, 3))
    transform = dense(values, grid())
    values[...] = 9.0
    assert transform.field.max() == 0.0
    view = transform.field
    with pytest.raises(ValueError):
        view[0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        view.flags.writeable = True
    coefficients = spline(np.zeros((6, 6, 6, 3)), grid((6, 6, 6), source=POINTS)).coefficients
    with pytest.raises(ValueError):
        coefficients.flags.writeable = True


# --- Identity, pickling, repr, encoding -------------------------------------------------------


def test_equality_is_identity_and_transforms_are_hashable() -> None:
    first = dense(np.zeros((4, 5, 3, 3)), grid())
    second = dense(np.zeros((4, 5, 3, 3)), grid())
    assert first == first
    assert first != second
    assert len({first, second, first}) == 2


def test_a_pickled_field_maps_the_same_points() -> None:
    field = rng().normal(size=(4, 5, 3, 3)).astype(np.float32)
    transform = dense(field, grid(), outside="nearest")
    copy = pickle.loads(pickle.dumps(transform))
    assert copy is not transform and copy != transform
    assert copy.field.dtype == np.float32 and copy.outside == "nearest"
    points = rng().uniform(-1, 4, size=(10, 3))
    assert_allclose(copy.transform_point(points), transform.transform_point(points))
    bspline = spline(rng().normal(size=(6, 6, 6, 3)), grid((6, 6, 6), source=POINTS))
    restored = pickle.loads(pickle.dumps(bspline))
    assert_allclose(restored.coefficients, bspline.coefficients)
    assert restored.order == 3


def test_repr_names_the_declaration_without_the_values() -> None:
    text = repr(dense(np.zeros((4, 5, 3, 3), dtype=np.float32), grid()))
    assert text.startswith("DisplacementFieldTransform(source=")
    assert "float32" in text and "'raise'" in text and "0." not in text


def test_field_transforms_are_not_encoded_yet() -> None:
    transform = dense(np.zeros((4, 5, 3, 3)), grid())
    with pytest.raises(TypeError, match="stage 1b"):
        xrf.encode(transform)
    affine = xrf.AffineTransform.from_matrix(
        source=MOVING, target=FIXED, matrix=np.eye(3), translation=np.zeros(3)
    )
    with pytest.raises(TypeError, match="stage 1b"):
        xrf.encode(xrf.CompositeTransform(transform, affine))


# --- Composition and the determinant report ---------------------------------------------------


def test_a_composite_chains_a_field_jacobian() -> None:
    field = rng().normal(scale=0.2, size=(4, 5, 3, 3))
    transform = dense(field, grid())
    affine = xrf.AffineTransform.from_matrix(
        source=MOVING, target=FIXED, matrix=oblique(), translation=np.zeros(3)
    )
    chain = xrf.compose(transform, affine)
    assert isinstance(chain, xrf.CompositeTransform)
    point = np.array([1.3, 2.4, 0.6])
    assert_allclose(chain.jacobian(point), oblique() @ transform.jacobian(point))


def test_the_report_flags_a_folded_field() -> None:
    field = np.zeros((4, 5, 3, 3))
    field[..., 0] = -2.0 * np.arange(4)[:, None, None]  # u_x = -2x: J_xx = -1 everywhere
    report = xrf.jacobian_determinant_report(dense(field, grid()), grid())
    assert report.determinant.shape == (4, 5, 3)
    assert report.minimum == pytest.approx(-1.0)
    assert report.folding_fraction == 1.0
    with pytest.raises(ValueError):
        report.determinant.flags.writeable = True


def test_the_report_finds_a_smooth_field_fold_free() -> None:
    field = rng().normal(scale=0.05, size=(4, 5, 3, 3))
    points = rng().uniform(0, [3, 4, 2], size=(7, 2, 3))
    report = xrf.jacobian_determinant_report(dense(field, grid()), points)
    assert report.determinant.shape == (7, 2)
    assert report.folding_fraction == 0.0 and report.minimum > 0.5


def test_the_report_validates_its_inputs() -> None:
    transform = dense(np.zeros((4, 5, 3, 3)), grid())
    with pytest.raises(TypeError, match="provides no Jacobian"):
        xrf.jacobian_determinant_report(cast(Any, object()), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="evaluate it in the transform's source frame"):
        xrf.jacobian_determinant_report(transform, grid(frame=MOVING))
    with pytest.raises(ValueError, match="no points"):
        xrf.jacobian_determinant_report(transform, np.zeros((0, 3)))
    plane = xrf.ReferenceFrame.declared(
        ("test", "plane"), xrf.CoordinateSystem(("u", "v"), ("mm",) * 2)
    )
    projection = xrf.AffineTransform.from_matrix(
        source=FIXED, target=plane, matrix=np.eye(3)[:2], translation=np.zeros(2)
    )
    with pytest.raises(ValueError, match="square Jacobian"):
        xrf.jacobian_determinant_report(projection, np.zeros((1, 3)))


def test_rasterization_validates_its_inputs() -> None:
    bspline = spline(np.zeros((6, 6, 6, 3)), grid((6, 6, 6), source=POINTS))
    with pytest.raises(ValueError, match="declares point samples"):
        xrf.sample_displacement_field(bspline, bspline.grid)
    with pytest.raises(ValueError, match="sample in the transform's source frame"):
        xrf.sample_displacement_field(bspline, grid(frame=MOVING))
    with pytest.raises(TypeError, match="grid must be a Grid"):
        xrf.sample_displacement_field(bspline, cast(xrf.Grid, None))
    with pytest.raises(TypeError, match="SupportsPoints"):
        xrf.sample_displacement_field(cast(Any, object()), grid())


# --- Resampling through a field ---------------------------------------------------------------


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


def test_resampling_through_a_constant_field_is_resampling_through_the_translation() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    i, j, k = np.indices((6, 6, 6), dtype=np.float64)
    source = framed(1.0 + 2.0 * i + 3.0 * j + 5.0 * k, MOVING)
    target = framed(np.zeros((4, 4, 4)), FIXED)
    offset = np.array([0.75, 0.5, 1.25])
    field = dense(np.broadcast_to(offset, (4, 4, 4, 3)), grid((4, 4, 4)))
    translation = xrf.AffineTransform.from_matrix(
        source=FIXED, target=MOVING, matrix=np.eye(3), translation=offset
    )
    through_field = source.rf.resample_to(target, transform=field)
    expected = source.rf.resample_to(target, transform=translation)
    assert_allclose(through_field.values, expected.values, atol=1e-10)
    # A chain of a field and an affine resamples like the collapsed translation.
    within = xrf.ReferenceFrame.declared(("test", "intermediate"), SYSTEM)
    half = xrf.DisplacementFieldTransform(
        source=FIXED,
        target=within,
        field=np.broadcast_to(offset / 2, (4, 4, 4, 3)),
        grid=grid((4, 4, 4)),
    )
    rest = xrf.AffineTransform.from_matrix(
        source=within, target=MOVING, matrix=np.eye(3), translation=offset / 2
    )
    chained = source.rf.resample_to(target, transform=xrf.compose(half, rest))
    assert_allclose(chained.values, expected.values, atol=1e-10)


def test_declared_vectors_follow_the_fields_pointwise_jacobian() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    matrix = np.array([[1.2, 0.3, 0.0], [0.0, 0.9, 0.1], [0.2, 0.0, 1.1]])
    affine = xrf.AffineTransform.from_matrix(
        source=FIXED, target=MOVING, matrix=matrix, translation=(1.0, 1.0, 1.0)
    )
    # Linear interpolation reproduces the affine exactly, so its Jacobian is the matrix.
    field = xrf.sample_displacement_field(affine, grid((4, 4, 4)))
    source = framed(np.broadcast_to([1.0, 2.0, 3.0], (8, 8, 8, 3)).copy(), MOVING, extra=("c",))
    source = source.rf.declare_tensor_components(xrf.VECTOR, component_dims=("c",), bases=MOVING)
    target = framed(np.zeros((3, 3, 3)), FIXED)
    options = {"output_component_basis": FIXED, "component_reorientation": "jacobian"}
    through_field = source.rf.resample_to(target, transform=field, **options)
    expected = source.rf.resample_to(target, transform=affine, **options)
    assert_allclose(
        through_field.transpose("i", "j", "k", "c").values,
        expected.transpose("i", "j", "k", "c").values,
        atol=1e-10,
    )


def test_a_raise_field_refuses_a_target_point_outside_its_support() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    source = framed(np.zeros((6, 6, 6)), MOVING)
    target = framed(np.zeros((6, 6, 6)), FIXED)
    field = dense(np.zeros((4, 4, 4, 3)), grid((4, 4, 4)))
    with pytest.raises(ValueError, match="outside the displacement field's support"):
        source.rf.resample_to(target, transform=field)


def test_evaluation_without_scipy_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    transform = dense(np.zeros((4, 5, 3, 3)), grid())
    real_import = builtins.__import__

    def without_scipy(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    with pytest.raises(ImportError, match=r"xarrayrf\[resample\]"):
        transform.transform_point([1.0, 1.0, 1.0])
    # Construction and inspection need no SciPy.
    assert dense(np.zeros((4, 5, 3, 3)), grid()).field.shape == (4, 5, 3, 3)


@pytest.mark.parametrize("outside", ["raise", "nearest"])
def test_bspline_jacobians_are_exact_at_the_support_faces(outside: str) -> None:
    # Cardinal cubic B-splines reproduce quadratics: coefficients k**2 give p**2 + 1/3, whose
    # derivative 2p holds on the whole support, faces included. Four control points per axis put
    # both faces at a tap boundary.
    k = np.arange(4.0)
    coefficients = np.zeros((4, 4, 4, 3))
    coefficients[..., 0] = k[:, None, None] ** 2
    coefficients[..., 1] = k[None, :, None] ** 2
    coefficients[..., 2] = k[None, None, :] ** 2
    spacing = np.array([1.5, 2.0, 0.5])
    transform = spline(
        coefficients, grid((4, 4, 4), matrix=np.diag(spacing), source=POINTS), outside
    )
    faces = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 2.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    points = faces * spacing
    expected = np.eye(3) + np.einsum("na,ab->nab", 2 * faces, np.diag(1 / spacing))
    assert_allclose(transform.jacobian(points), expected, atol=ATOL)
    assert_allclose(transform.transform_point(points), points + faces**2 + 1 / 3, atol=ATOL)


def test_a_large_grid_origin_keeps_local_indices() -> None:
    # A 0.1-spaced axis at 1e15: folding the translation into the index affine would round
    # 1e15 + 0.125 to index 2 instead of 1.25, and move 1e15 + 0.25 (index 2.5, the excluded
    # upper face) inside.
    g = grid((3, 3, 3), matrix=np.diag([0.1, 1.0, 1.0]), translation=(1e15, 0.0, 0.0))
    field = np.zeros((3, 3, 3, 3))
    field[..., 1] = np.arange(3.0)[:, None, None]
    points = np.array([[1e15 + 0.125, 0.0, 0.0], [1e15 + 0.25, 0.0, 0.0]])
    mapped = dense(field, g, outside="identity").transform_point(points)
    # Exact: linear interpolation of a linear field at representable positions.
    assert_allclose(mapped[:, 1], [1.25, 0.0], rtol=0, atol=0)
    with pytest.raises(ValueError, match="outside the displacement field's support"):
        dense(field, g).transform_point(points)
    jacobian = dense(field, g, outside="identity").jacobian(points[:1])
    assert_allclose(jacobian[0, 1, 0], 10.0, rtol=1e-12)


def test_large_coordinate_values_keep_local_indices() -> None:
    # The origin can sit in the coordinate values rather than the translation; the index map
    # must cancel it before scaling by 1 / 0.375 too.
    g = grid((3, 3, 3), coordinates={"i": 1e15 + np.array([0.0, 0.375, 0.75])})
    field = np.zeros((3, 3, 3, 3))
    field[..., 1] = np.arange(3.0)[:, None, None]
    mapped = dense(field, g).transform_point(np.array([[1e15 + 0.125, 0.0, 0.0]]))
    # 0.125 / 0.375 rounds once.
    assert_allclose(mapped[:, 1], [1 / 3], rtol=1e-15, atol=0)
