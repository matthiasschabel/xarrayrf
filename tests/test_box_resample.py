"""Public box resampling, including support metadata and lazy execution."""

from typing import Any, Literal

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose
from test_intervals import mapping
from test_resample import IJK, LPS, RAS, PointOnly

import xarrayrf as xrf
from xarrayrf.native import frame_array

ATOL = 1e-12  # Roundoff for small synthetic weighted means and affine maps.


def slab_grid(rows: npt.ArrayLike, *, scalar: bool = False) -> xrf.Grid:
    bounds = np.asarray(rows, dtype=float)
    centers = bounds.mean(axis=-1)
    return xrf.Grid(
        mapping(),
        {"z": float(centers[0]) if scalar else ("slice", centers)},
        intervals={"z": bounds[0] if scalar else bounds},
    )


@pytest.mark.parametrize("method", ["step", "overlap_mean"])
@pytest.mark.parametrize(
    "target_rows, expected", [([[0, 5], [5, 10]], [2, 7]), ([[0.5, 2]], [2 / 3])]
)
def test_box_average_weights(method: xrf.Method, target_rows: Any, expected: Any) -> None:
    source = frame_array(np.arange(10.0), slab_grid(np.arange(10)[:, None] + [0, 1]))
    result = xrf.resample(
        source.rf.geometry, slab_grid(target_rows), method=method, support="average"
    )
    assert_allclose(result, expected, atol=ATOL, rtol=0)
    assert result.dtype == np.float64


@pytest.mark.parametrize("method", ["step", "overlap_mean"])
@pytest.mark.parametrize("rows", [[[0, 1], [1, 3], [3, 6]], [[0, 1], [2, 4], [7, 10]]])
def test_box_average_identity(method: xrf.Method, rows: Any) -> None:
    source = frame_array(np.array([2.0, 11.0, 7.0]), slab_grid(rows))
    result = source.rf.resample_to(source, method=method, support="average")
    assert_allclose(result, source, atol=ATOL, rtol=0)
    assert_allclose(result.rf.grid.intervals["z"], rows, atol=ATOL, rtol=0)


@pytest.mark.parametrize(
    "rows, target, expected",
    [
        (np.arange(5)[:, None] + [-1, 1], [[0, 4]], [1 / 8, 2 / 8, 2 / 8, 2 / 8, 1 / 8]),
        ([[0, 1.5], [1, 2.5]], [[0, 2]], [0.625, 0.375]),
    ],
)
def test_overlap_pointwise_normalisation(rows: Any, target: Any, expected: Any) -> None:
    grid = slab_grid(rows)
    source = frame_array(
        np.eye(len(rows)),
        grid,
        dims=("echo", *grid.dims),
        coords={"echo": ("echo", np.arange(len(rows)), {})},
    )
    result = source.rf.resample_to(slab_grid(target), method="overlap_mean", support="average")
    assert_allclose(result.values[:, 0], expected, atol=ATOL, rtol=0)
    with pytest.raises(ValueError, match="overlapping source intervals"):
        source.rf.resample_to(slab_grid(target), method="step", support="average")


@pytest.mark.parametrize("support", ["point", "average"])
@pytest.mark.parametrize("method", ["step", "overlap_mean"])
def test_box_gaps_are_fill(method: xrf.Method, support: Literal["point", "average"]) -> None:
    source = frame_array(np.array([2.0, 7.0]), slab_grid([[0, 5], [6, 11]]))
    result, coverage = source.rf.resample_to(
        slab_grid([[5.1, 5.9]]), method=method, support=support, fill_value=-9, return_coverage=True
    )
    assert_allclose(result, [-9], atol=ATOL, rtol=0)
    assert_allclose(coverage, [0], atol=ATOL, rtol=0)


@pytest.mark.parametrize("threshold, expected", [(0.5, [2, 7]), (0.9, [-9, -9])])
def test_partial_coverage_and_threshold(threshold: float, expected: Any) -> None:
    source = frame_array(np.array([2.0, 7.0]), slab_grid([[0, 5], [6, 11]]))
    result, coverage = source.rf.resample_to(
        slab_grid([[0, 6], [6, 12]]),
        method="step",
        support="average",
        min_coverage=threshold,
        fill_value=-9,
        return_coverage=True,
    )
    assert_allclose(result, expected, atol=ATOL, rtol=0)
    assert_allclose(coverage, [5 / 6, 5 / 6], atol=ATOL, rtol=0)


def test_closed_interval_point_edges() -> None:
    source = frame_array(np.array([2.0, 6.0, 10.0]), slab_grid([[0, 1], [1, 2], [3, 4]]))
    target = xrf.Grid(mapping(), {"z": ("slice", [0.25, 1, 2.5])})
    result = source.rf.resample_to(target, method="step", fill_value=-9)
    assert_allclose(result, [2, 4, -9], atol=ATOL, rtol=0)
    assert not result.rf.grid.intervals


def test_scalar_average_target() -> None:
    source = frame_array(np.arange(4.0), slab_grid(np.arange(4)[:, None] + [0, 1]))
    result, coverage = source.rf.resample_to(
        slab_grid([[0, 4]], scalar=True), method="step", support="average", return_coverage=True
    )
    assert result.dims == coverage.dims == ()
    assert_allclose(result, 1.5, atol=ATOL, rtol=0)
    assert_allclose(coverage, 1, atol=ATOL, rtol=0)
    assert_allclose(result.rf.grid.intervals["z"], [0, 4], atol=ATOL, rtol=0)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.1, np.nan, np.inf, "0.5", 1j])
def test_min_coverage_refusals(invalid: Any) -> None:
    source = frame_array(np.array([2.0, 7.0]), slab_grid([[0, 1], [1, 2]]))
    with pytest.raises(ValueError, match=r"min_coverage.*real number"):
        source.rf.resample_to(source, min_coverage=invalid)


@pytest.mark.parametrize(
    "options, error, match",
    [
        ({"support": "wrong"}, ValueError, "support must"),
        ({"support": "average"}, ValueError, "not yet supported.*step or overlap_mean"),
        ({"method": "step", "domain": "cells"}, ValueError, "declared support is the domain"),
        ({"return_coverage": True}, ValueError, "return_coverage requires"),
    ],
)
def test_box_option_refusals(options: Any, error: type[Exception], match: str) -> None:
    source = frame_array(np.array([2.0, 7.0]), slab_grid([[0, 1], [1, 2]]))
    with pytest.raises(error, match=match):
        source.rf.resample_to(source, **options)


@pytest.mark.parametrize("dtype", [np.int32, np.bool_])
def test_box_integer_refusal(dtype: Any) -> None:
    source = frame_array(np.ones(2, dtype=dtype), slab_grid([[0, 1], [1, 2]]))
    with pytest.raises(TypeError, match="convert explicitly"):
        source.rf.resample_to(source, method="step")


@pytest.mark.parametrize("missing", ["source", "target"])
def test_box_requires_declared_slabs(missing: str) -> None:
    source_grid = slab_grid([[0, 1], [1, 2]])
    target = slab_grid([[0, 2]])
    if missing == "source":
        source_grid = xrf.Grid(source_grid.transform, source_grid.coordinates)
    else:
        target = xrf.Grid(target.transform, target.coordinates)
    source = frame_array(np.array([2.0, 7.0]), source_grid)
    with pytest.raises(
        ValueError,
        match="resample that axis first" if missing == "source" else "target axis.*intervals",
    ):
        source.rf.resample_to(target, method="step", support="average")


@pytest.mark.parametrize("complex_values", [False, True])
def test_box_nan_components_only_propagate_positive_weights(complex_values: bool) -> None:
    values = np.array([2.0, np.nan, 6.0, 8.0])
    if complex_values:
        values = values.astype(complex)
        values.imag = [np.nan, 4, 6, 8]
    source = frame_array(values, slab_grid(np.arange(4)[:, None] + [0, 1]))
    result = source.rf.resample_to(
        slab_grid([[0.0, 0.5], [0.0, 2.0], [1.0, 2.0], [2.0, 4.0]]),
        method="step",
        support="average",
    )
    assert_allclose(result.values.real, [2, np.nan, np.nan, 7], atol=ATOL, rtol=0)
    if complex_values:
        assert result.dtype == np.complex128
        assert_allclose(result.values.imag, [np.nan, np.nan, 4, 7], atol=ATOL, rtol=0)


@pytest.mark.parametrize("variant", ["plain", "reverse", "units", "ras", "permutation"])
def test_box_spatial_mapping_and_coverage(variant: str) -> None:
    coordinates = xrf.ArrayCoordinates(IJK.axes, ("mm",) * 3, sample_offset=(0.5,) * 3)
    transform = xrf.AffineTransform.from_matrix(
        source=coordinates, target=LPS, matrix=np.eye(3), translation=[0, 0, 0]
    )
    grid = xrf.Grid(
        transform,
        {"i": ("i", [0.0, 1.0]), "j": ("j", [0.0, 1.0, 2.0]), "k": ("k", [0.5, 1.5, 2.5, 3.5])},
        intervals={"k": np.arange(4)[:, None] + [0, 1], "i": [[-0.5, 0.5], [0.5, 1.5]]},
    )
    values = np.arange(48.0).reshape(2, 4, 3, 2)
    source = frame_array(
        values, grid, dims=("echo", "k", "j", "i"), coords={"echo": ("echo", np.arange(2), {})}
    )
    if variant == "reverse":
        source = source.isel(k=slice(None, None, -1))
    matrix = np.diag([-1.0, -1.0, 1.0]) if variant == "ras" else np.eye(3)
    target_coords = dict(grid.coordinates)
    target_coords["k"] = ("k", [1.0, 3.0])
    target_rows = np.array([[0.0, 2.0], [2.0, 4.0]])
    if variant == "units":
        coordinates = xrf.ArrayCoordinates(IJK.axes, ("1",) * 3, sample_offset=(0.5,) * 3)
        matrix[2, 2] = -2
        target_coords["k"] = ("k", [-0.5, -1.5])
        target_rows = np.array([[-1.0, 0.0], [-2.0, -1.0]])
    slab_axis = "k"
    if variant == "permutation":
        matrix = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        target_coords["i"], target_coords["k"] = ("i", [1.0, 3.0]), ("k", [0.0, 1.0])
        slab_axis = "i"
    target_transform = xrf.AffineTransform.from_matrix(
        source=coordinates,
        target=RAS if variant == "ras" else LPS,
        matrix=matrix,
        translation=[0, 0, 0],
    )
    target = xrf.Grid(target_transform, target_coords, intervals={slab_axis: target_rows})
    target = (
        target.transpose("k", "j", "i")
        if variant != "permutation"
        else target.transpose("i", "j", "k")
    )
    result, coverage = source.rf.resample_to(
        target, method="step", support="average", return_coverage=True
    )
    assert_allclose(result, values.reshape(2, 2, 2, 3, 2).mean(axis=2), atol=ATOL, rtol=0)
    assert coverage.dims == target.dims
    assert coverage.dtype == np.float64
    assert coverage.rf.coordinate_transform is target.transform
    assert not coverage.rf.grid.intervals
    assert_allclose(coverage, np.ones((2, 3, 2)), atol=ATOL, rtol=0)
    assert slab_axis in result.rf.grid.intervals


def test_box_lazy_context_dimensions() -> None:
    grid = slab_grid(np.arange(4)[:, None] + [0, 1])
    values = np.arange(8.0).reshape(2, 4)
    source = frame_array(
        values,
        grid,
        dims=("echo", *grid.dims),
        coords={"echo": ("echo", np.arange(2), {})},
    )
    source = source.chunk({"echo": 1, "slice": 2})
    result = source.rf.resample_to(slab_grid([[0, 2], [2, 4]]), method="step", support="average")
    assert isinstance(result.data, da.Array)
    assert result.dims == ("echo", "slice")
    assert_allclose(result.compute(), values.reshape(2, 2, 2).mean(axis=-1), atol=ATOL, rtol=0)


@pytest.mark.parametrize("non_affine", [False, True])
def test_box_refuses_unclassifiable_maps(non_affine: bool) -> None:
    transform = xrf.AffineTransform.from_matrix(
        source=IJK, target=LPS, matrix=np.eye(3), translation=[0, 0, 0]
    )
    source = frame_array(
        np.ones((2, 2, 2)), xrf.Grid(transform, {d: (d, [0.0, 1.0]) for d in IJK.axes})
    )
    mapping = xrf.AffineTransform.from_matrix(
        source=IJK,
        target=LPS,
        matrix=[[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]],
        translation=[0, 0, 0],
    )
    target = xrf.Grid(PointOnly(mapping) if non_affine else mapping, source.rf.grid.coordinates)
    with pytest.raises(ValueError, match="non-affine" if non_affine else "non-separable"):
        source.rf.resample_to(target, method="step")


@pytest.mark.parametrize("threshold, expected", [(0.25, 7), (0.5, -9), (1.0, -9)])
def test_coverage_multiplies_across_slab_axes(threshold: float, expected: float) -> None:
    coordinates = xrf.ArrayCoordinates(("x", "y"), ("mm", "mm"), sample_offset=(0.5, 0.5))
    transform = xrf.AffineTransform.from_matrix(
        source=coordinates,
        target=xrf.ReferenceFrame.local(xrf.CoordinateSystem(("X", "Y"), ("mm", "mm"))),
        matrix=np.eye(2),
        translation=[0, 0],
    )
    source = frame_array(
        np.array([[7.0]]),
        xrf.Grid(
            transform,
            {"x": ("x", [0.5]), "y": ("y", [0.5])},
            intervals={"x": [[0, 1]], "y": [[0, 1]]},
        ),
    )
    target = xrf.Grid(
        transform, {"x": ("x", [1.0]), "y": ("y", [1.0])}, intervals={"x": [[0, 2]], "y": [[0, 2]]}
    )
    result, coverage = source.rf.resample_to(
        target,
        method="step",
        support="average",
        min_coverage=threshold,
        fill_value=-9,
        return_coverage=True,
    )
    assert_allclose(result, [[expected]], atol=ATOL, rtol=0)
    assert_allclose(coverage, [[0.25]], atol=ATOL, rtol=0)


@pytest.mark.parametrize("overlap, refuses", [(5e-10, False), (2e-9, True)])
def test_step_overlap_uses_interval_roundoff(overlap: float, refuses: bool) -> None:
    source = frame_array(np.ones(2), slab_grid([[0.0, 1.0], [1.0 - overlap, 2.0 - overlap]]))
    target = slab_grid([[0.0, 2.0]])
    if refuses:
        with pytest.raises(ValueError, match="overlapping source intervals"):
            source.rf.resample_to(target, method="step", support="average")
    else:
        result = source.rf.resample_to(target, method="step", support="average")
        assert_allclose(result, [1], atol=ATOL, rtol=0)


def test_box_matches_linear_interpolation_on_a_ramp() -> None:
    # Slab means of a linear signal equal its value at slab centres, so the box average onto
    # whole-slab targets and linear interpolation at target centres must agree exactly.
    rows = np.array([[0, 1], [1, 3], [3, 4], [4, 7], [7, 8], [8, 12]], dtype=float)
    centres = rows.mean(axis=-1)
    source = frame_array(3 * centres + 1, slab_grid(rows))
    target = slab_grid([[0, 3], [3, 8], [1, 7]])
    target_centres = np.array([1.5, 5.5, 4.0])
    for method in ("step", "overlap_mean"):
        box = source.rf.resample_to(target, method=method, support="average")
        assert_allclose(box, 3 * target_centres + 1, atol=ATOL, rtol=0)
    point_target = xrf.Grid(target.transform, {"z": ("slice", centres[1:-1])})
    linear = source.rf.resample_to(point_target, method="linear")
    assert_allclose(linear, 3 * centres[1:-1] + 1, atol=ATOL, rtol=0)


def test_box_stays_close_to_the_signal_for_thin_slabs() -> None:
    # Thin contiguous slabs of width h hold exact means of sin. Averaging onto unaligned
    # targets of length L treats only the two cut edge slabs as constant, an error below
    # h**2 * max|f'| / L; evaluating the step at a point errs by at most h / 2 * max|f'|.
    h, length = 0.05, 1.0
    edges = np.arange(0, 10 + h / 2, h)
    rows = np.column_stack([edges[:-1], edges[1:]])
    means = (np.cos(rows[:, 0]) - np.cos(rows[:, 1])) / h
    source = frame_array(means, slab_grid(rows))
    starts = np.array([0.37, 2.913, 5.5, 8.01])
    targets = np.column_stack([starts, starts + length])
    expected = (np.cos(targets[:, 0]) - np.cos(targets[:, 1])) / length
    box = source.rf.resample_to(slab_grid(targets), method="overlap_mean", support="average")
    assert_allclose(box, expected, atol=h**2 / length, rtol=0)
    centres = targets.mean(axis=-1)
    point_target = xrf.Grid(slab_grid(targets).transform, {"z": ("slice", centres)})
    step = source.rf.resample_to(point_target, method="step")
    linear = source.rf.resample_to(point_target, method="linear")
    assert_allclose(step, np.sin(centres), atol=h / 2, rtol=0)
    assert_allclose(step, linear, atol=h, rtol=0)


def test_point_on_an_edge_within_roundoff_takes_both_slabs() -> None:
    source = frame_array(np.array([0.0, 2.0]), slab_grid([[0, 1], [1, 2]]))
    for offset in (-1e-13, 0.0, 1e-13):
        target = xrf.Grid(slab_grid([[0, 1]]).transform, {"z": ("slice", [1.0 + offset])})
        assert_allclose(source.rf.resample_to(target, method="step"), [1.0], atol=ATOL, rtol=0)
