"""Shared anatomical vocabulary and patient coordinate systems."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations, product

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from xarrayrf import (
    AffineTransform,
    ArrayCoordinates,
    CoordinateSystem,
    DirectionVocabulary,
    Grid,
    ReferenceFrame,
    coordinate_system_change,
)
from xarrayrf.anatomy import (
    LPS,
    RAS,
    VOCABULARY,
    cardinal_grid,
    orientation_codes,
    patient_coordinate_system,
    reoriented,
)

ATOL = 1e-12  # Signed permutations have exact coefficients; allow float evaluation slack.


def test_rfc4_vocabulary_is_exact_and_bidirectional() -> None:
    """Canonical list from https://ngff.openmicroscopy.org/rfc/4/ ."""
    pairs = (
        "left-to-right",
        "anterior-to-posterior",
        "inferior-to-superior",
        "dorsal-to-ventral",
        "proximal-to-distal",
        "dorsal-to-palmar",
        "dorsal-to-plantar",
        "rostral-to-caudal",
        "cranial-to-caudal",
        "superficial-to-deep",
        "apical-to-basal",
        "apex-to-base",
    )
    expected = set(pairs)
    expected.update("-to-".join(reversed(token.split("-to-"))) for token in pairs)
    assert VOCABULARY.identifier == "ome-ngff:rfc-4:anatomical"
    assert VOCABULARY.directions == expected
    assert len(expected) == 24
    assert all(VOCABULARY.opposite(token) in VOCABULARY.directions for token in expected)


def test_ras_lps_patient_systems_share_an_exact_derived_flip() -> None:
    ras = patient_coordinate_system(RAS, "mm")
    lps = patient_coordinate_system(LPS, "mm")
    assert ras.axes == lps.axes == ("x", "y", "z")
    assert ras.axis_types == lps.axis_types == ("space",) * 3
    frame = ReferenceFrame.local(ras)
    change = coordinate_system_change(frame, frame.with_coordinate_system(lps))
    assert_allclose(change.matrix, np.diag([-1.0, -1.0, 1.0]), rtol=0, atol=ATOL)


def test_patient_system_can_name_its_axes() -> None:
    system = patient_coordinate_system(RAS, "um", axes=("R", "A", "S"))
    assert system.axes == ("R", "A", "S")
    assert system.units == ("um",) * 3


# Grids use auxiliary coordinate names to exercise the axis/dimension distinction.
def volume(
    *,
    orientation: tuple[str, ...] = RAS,
    angle: float = 0,
    values: tuple[list[float], ...] = ([0, 1, 2], [0, 1, 2, 3], [0, 1]),
    offsets: tuple[float | None, ...] = (0.5, 0.5, 0.5),
    intervals: Mapping[str, npt.ArrayLike] | None = None,
) -> Grid:
    theta = np.deg2rad(angle)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("u", "v", "w"), ("1",) * 3, sample_offset=offsets),
        target=ReferenceFrame.local(patient_coordinate_system(orientation, "mm")),
        matrix=rotation @ np.diag([2.0, 3.0, 4.0]),
        translation=[10, -20, 30],
    )
    return Grid(
        transform,
        {
            axis: (dim, coord)
            for axis, dim, coord in zip(("u", "v", "w"), ("a", "b", "c"), values, strict=True)
        },
        intervals=intervals,
    )


def corners(grid: Grid, cover: str) -> npt.NDArray[np.float64]:
    bounds = []
    for j, axis in enumerate(grid.transform.source.axes):
        entry = grid.coordinates[axis]
        assert isinstance(entry, tuple)
        values = np.asarray(entry[1], dtype=float)
        if cover == "samples":
            bounds.append((values.min(), values.max()))
        elif axis in grid.intervals:
            rows = grid.intervals[axis]
            bounds.append((rows[:, 0].min(), rows[:, 1].max()))
        else:
            offset = grid.transform.source.sample_offset[j]  # type: ignore[union-attr]
            assert offset is not None
            ordered = np.sort(values)
            bounds.append(
                (
                    ordered[0] - offset * (ordered[1] - ordered[0]),
                    ordered[-1] + (1 - offset) * (ordered[-1] - ordered[-2]),
                )
            )
    return grid.transform.transform_point(np.array(list(product(*bounds))))


@pytest.mark.parametrize("system,codes", [(RAS, "RAS"), (LPS, "LPS")])
@pytest.mark.parametrize("angle", [0, 20])
def test_orientation_in_patient_frames(system: tuple[str, ...], codes: str, angle: float) -> None:
    grid = volume(orientation=system, angle=angle)
    assert orientation_codes(grid) == codes
    assert orientation_codes(grid.isel(a=slice(None, None, -1))) == (
        {"R": "L", "L": "R"}[codes[0]] + codes[1:]
    )
    assert orientation_codes(grid.isel(a=0)) == codes[1:]
    assert orientation_codes(grid.isel(a=0, b=0, c=0)) == ""


@pytest.mark.parametrize("order", list(permutations("RAS")))
@pytest.mark.parametrize("signs", list(product([False, True], repeat=3)))
@pytest.mark.parametrize("singleton", [False, True])
def test_all_signed_permutations_preserve_points_and_cells(
    order: tuple[str, ...],
    signs: tuple[bool, ...],
    singleton: bool,
) -> None:
    values = ([2.0] if singleton else [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0])
    offsets = (0.25, 0.5, 0.75)
    intervals = {
        name: [[v - offset * 1.5, v + (1 - offset) * 1.5] for v in vals]
        for name, vals, offset in zip(("u", "v", "w"), values, offsets, strict=True)
    }
    grid = volume(angle=20, values=values, offsets=offsets, intervals=intervals)
    opposites = {"R": "L", "A": "P", "S": "I"}
    requested = "".join(
        opposites[token] if flip else token for token, flip in zip(order, signs, strict=True)
    )
    result = reoriented(grid, requested)
    assert orientation_codes(result) == requested
    assert result.frame == grid.frame
    original = grid.points()
    expected = original.transpose(*("RAS".index(token) for token in order), 3)
    for j, flip in enumerate(signs):
        if flip:
            expected = np.flip(expected, axis=j)
    assert_allclose(result.points(), expected, rtol=0, atol=ATOL)
    assert_allclose(
        np.sort(corners(result, "cells"), axis=0),
        np.sort(corners(grid, "cells"), axis=0),
        rtol=0,
        atol=ATOL,
    )
    assert reoriented(result, "RAS") == grid


def test_assignment_is_global_when_nearest_axes_collide() -> None:
    grid = volume()
    transform = AffineTransform.from_matrix(
        source=grid.transform.source,
        target=grid.frame,
        matrix=np.array([[1, 0.8, 0], [0.1, 0.6, 0], [0, 0, 1]]),
        translation=[0, 0, 0],
    )
    assert orientation_codes(Grid(transform, grid.coordinates)) == "RAS"


@pytest.mark.parametrize("sign", [1, -1])
def test_assignment_refuses_parallel_anatomical_directions(sign: int) -> None:
    grid = volume()
    transform = AffineTransform.from_matrix(
        source=grid.transform.source,
        target=grid.frame,
        matrix=[[1, sign, 0], [1, sign, 0], [0, 0, 1]],
        translation=[0, 0, 0],
    )
    singular = Grid(transform, grid.coordinates)
    with pytest.raises(ValueError, match=r"dimensions 'a' and 'b' point along the same direction"):
        orientation_codes(singular)
    for function in (reoriented, cardinal_grid):
        with pytest.raises(ValueError, match=r"dimensions .* point along the same direction"):
            function(singular, "RAS")


def test_assignment_accepts_45_degree_shear() -> None:
    grid = volume()
    transform = AffineTransform.from_matrix(
        source=grid.transform.source,
        target=grid.frame,
        matrix=[[1, 1, 0], [0, 1, 0], [0, 0, 1]],
        translation=[0, 0, 0],
    )
    assert orientation_codes(Grid(transform, grid.coordinates)) == "RAS"


def test_nearest_axis_assignment_flips_across_45_degrees() -> None:
    with pytest.raises(ValueError, match=r"ambiguous"):
        orientation_codes(volume(angle=45))
    assert orientation_codes(volume(angle=44.99)) == "RAS"
    assert orientation_codes(volume(angle=45.01)) == "ALS"


def test_non_patient_anatomical_tokens_round_trip() -> None:
    tokens = ("dorsal-to-ventral", "proximal-to-distal", "apical-to-basal")
    grid = volume(orientation=tokens)
    assert orientation_codes(grid) == tokens
    requested = tuple(VOCABULARY.opposite(token) for token in tokens[::-1])
    assert orientation_codes(reoriented(grid, requested)) == requested
    assert orientation_codes(cardinal_grid(grid, requested)) == requested


@pytest.mark.parametrize("system", [LPS, RAS])
@pytest.mark.parametrize("angle", [0, 20])
@pytest.mark.parametrize("cover", ["cells", "samples"])
@pytest.mark.parametrize(
    "plane,expected",
    [("axial", "SPL"), ("transverse", "SPL"), ("coronal", "PIL"), ("sagittal", "RIP")],
)
@pytest.mark.parametrize("spacing", [None, 1.25, (2.25, 3.25, 4.25)])
def test_cardinal_planes_cover_exact_source_corners(
    system: tuple[str, ...],
    angle: float,
    cover: str,
    plane: str,
    expected: str,
    spacing: float | tuple[float, ...] | None,
) -> None:
    grid = volume(orientation=system, angle=angle)
    output = cardinal_grid(grid, plane, spacing=spacing, cover=cover)
    assert orientation_codes(output) == expected
    lattice = output.lattice()
    assert_allclose(np.abs(lattice.direction).sum(axis=0), np.ones(3), rtol=0, atol=ATOL)
    desired = (
        np.array(
            [4, 3, 2]
            if plane in ("axial", "transverse")
            else [3, 4, 2]
            if plane == "coronal"
            else [2, 4, 3]
        )
        if spacing is None
        else np.broadcast_to(spacing, (3,))
    )
    assert_allclose(lattice.spacing, desired, rtol=0, atol=ATOL)
    input_corners = corners(grid, cover) @ lattice.direction
    output_corners = corners(output, cover) @ lattice.direction
    assert np.all(output_corners.min(axis=0) <= input_corners.min(axis=0) + ATOL)
    assert np.all(output_corners.max(axis=0) >= input_corners.max(axis=0) - ATOL)
    first = output.point_at(**dict.fromkeys(output.dims, 0)) @ lattice.direction
    assert_allclose(
        first,
        input_corners.min(axis=0) + (desired / 2 if cover == "cells" else 0),
        rtol=0,
        atol=ATOL,
    )
    expected_names = {"R": "a", "L": "a", "A": "b", "P": "b", "S": "c", "I": "c"}
    assert output.dims == tuple(expected_names[token] for token in expected)
    assert output.transform.source.sample_offset == (  # type: ignore[union-attr]
        (0.5,) * 3 if cover == "cells" else (None,) * 3
    )
    assert all(
        np.asarray(entry[1]).dtype == np.int64
        for entry in output.coordinates.values()
        if isinstance(entry, tuple)
    )


def test_explicit_dims_and_nonuniform_spacing() -> None:
    grid = volume(values=([0, 1, 3], [4, 2, 0], [0, 1]))
    with pytest.raises(ValueError, match=r"nonuniform.*explicit spacing"):
        cardinal_grid(grid, "axial")
    output = cardinal_grid(grid, "axial", dims=("slice", "row", "column"), spacing=[1, 2, 3])
    assert output.dims == ("slice", "row", "column")
    assert_allclose(output.lattice().spacing, [1, 2, 3], rtol=0, atol=ATOL)
    assert_allclose(
        output.positions_at(corners(grid, "cells"), domain="cells", outside="extrapolate").min(
            axis=0
        ),
        [-0.5] * 3,
        rtol=0,
        atol=ATOL,
    )


@pytest.mark.parametrize("plane", ["axial", "coronal"])
@pytest.mark.parametrize("spacing", [None, 10.0])
def test_single_thick_slice_retains_slab(plane: str, spacing: float | None) -> None:
    grid = volume(values=([0, 1, 2], [0, 1, 2, 3], [2]), intervals={"w": [[1.25, 2.75]]})
    output = cardinal_grid(grid, plane, spacing=spacing)
    assert orientation_codes(output) == {"axial": "SPL", "coronal": "PIL"}[plane]
    axis = "c"
    assert output.sizes[axis] == 1
    assert_allclose(
        output.intervals[axis],
        [
            [
                -0.5 * 6 / (6 if spacing is None else spacing),
                0.5 * 6 / (6 if spacing is None else spacing),
            ]
        ],
        rtol=0,
        atol=ATOL,
    )
    source_bounds = corners(grid, "cells")
    target_bounds = corners(output, "cells")
    assert_allclose(
        [target_bounds[:, 2].min(), target_bounds[:, 2].max()],
        [source_bounds[:, 2].min(), source_bounds[:, 2].max()],
        rtol=0,
        atol=ATOL,
    )
    result = cardinal_grid(grid, "axial", cover="samples")
    assert result.sizes["c"] == 1
    assert "c" not in result.intervals
    assert_allclose(
        result.point_at(c=0, b=0, a=0)[2], grid.point_at(a=0, b=0, c=0)[2], rtol=0, atol=ATOL
    )


@pytest.mark.parametrize("cover", ["cells", "samples"])
def test_singleton_output_declares_intervals_only_for_cells(cover: str) -> None:
    grid = volume(values=([0, 1e-10], [0, 1], [0, 1]))
    output = cardinal_grid(grid, "RAS", spacing=1, cover=cover)
    assert output.sizes["a"] == 1
    if cover == "cells":
        assert_allclose(output.intervals["a"], [[-2e-10, 2e-10]], rtol=0, atol=ATOL)
    else:
        assert not output.intervals
        assert output.transform.source.sample_offset == (None,) * 3  # type: ignore[union-attr]
        with pytest.raises(ValueError, match=r"point-sampled|no cells"):
            cardinal_grid(output, "RAS", spacing=1, cover="cells")


def test_rotated_singleton_cells_are_centred_on_covered_extent() -> None:
    grid = volume(angle=20)
    output = cardinal_grid(grid, "RAS", spacing=100)
    assert tuple(output.sizes.values()) == (1, 1, 1)
    bounds = corners(grid, "cells")
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    assert_allclose(output.point_at(a=0, b=0, c=0), (low + high) / 2, rtol=0, atol=ATOL)
    for j, name in enumerate(output.dims):
        width = (high[j] - low[j]) / 100
        assert_allclose(output.intervals[name], [[-width / 2, width / 2]], rtol=0, atol=ATOL)
    covered = corners(output, "cells")
    assert_allclose(covered.min(axis=0), low, rtol=0, atol=ATOL)
    assert_allclose(covered.max(axis=0), high, rtol=0, atol=ATOL)


def test_default_spacing_uses_oblique_anatomical_column_norms() -> None:
    grid = volume(angle=20, values=([0, 2, 4], [0, 3, 6], [0, 4]))
    output = cardinal_grid(grid, "RAS")
    assert_allclose(output.lattice().spacing, [4, 9, 16], rtol=0, atol=ATOL)


def test_cardinal_grid_resamples_ambiguous_orientation_with_explicit_spacing_and_dims() -> None:
    from xarrayrf.native import frame_array

    grid = volume(angle=45)
    with pytest.raises(ValueError, match=r"ambiguous.*cardinal grid"):
        orientation_codes(grid)
    target = cardinal_grid(grid, "RAS", spacing=1, dims=("x", "y", "z"))
    assert orientation_codes(target) == "RAS"
    coefficients = np.array([2.0, -3.0, 0.5])
    source = frame_array(7 + grid.points() @ coefficients, grid=grid)
    result = source.rf.resample_to(target)
    transform = grid.transform
    assert isinstance(transform, AffineTransform)
    positions = (target.points() - transform.translation) @ np.linalg.inv(transform.matrix).T
    inside = np.all((positions >= -ATOL) & (positions <= np.array([2, 3, 1]) + ATOL), axis=-1)
    assert inside.any() and (~inside).any()
    np.testing.assert_array_equal(np.isnan(result.values), ~inside)
    assert_allclose(
        result.values[inside], (7 + target.points() @ coefficients)[inside], rtol=0, atol=ATOL
    )
    assert result.rf.grid == target


@pytest.mark.parametrize("spacing,dims", [(None, None), (1, None), (None, ("x", "y", "z"))])
def test_ambiguous_cardinal_defaults_request_explicit_spacing_and_dims(
    spacing: float | None, dims: tuple[str, ...] | None
) -> None:
    with pytest.raises(ValueError, match=r"ambiguous.*pass explicit spacing and dims"):
        cardinal_grid(volume(angle=45), "RAS", spacing=spacing, dims=dims)


def test_explicit_cardinal_grid_still_requires_rank_three_anatomical_directions() -> None:
    grid = volume()
    transform = AffineTransform.from_matrix(
        source=grid.transform.source,
        target=grid.frame,
        matrix=[[1, 0, 1], [0, 1, 1], [0, 0, 0]],
        translation=[0, 0, 0],
    )
    with pytest.raises(ValueError, match="rank three"):
        cardinal_grid(Grid(transform, grid.coordinates), "RAS", spacing=1, dims=("x", "y", "z"))


def test_cardinal_self_orientation_resamples_by_exact_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scipy import ndimage

    from xarrayrf.native import frame_array

    grid = volume()
    values = np.arange(24).reshape(3, 4, 2)
    array = frame_array(values, grid=grid)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("same-grid resampling interpolated")

    monkeypatch.setattr(ndimage, "affine_transform", forbidden)
    output = array.rf.resample_to(cardinal_grid(grid, "RAS"))
    np.testing.assert_array_equal(output.values, values)


@pytest.mark.parametrize(
    "orientation", ["ras", "XYZ", "", "RRR", "RA", ("invalid",), ("left-to-right",) * 3]
)
def test_invalid_orientation_refuses(orientation: str | tuple[str, ...]) -> None:
    for function in (reoriented, cardinal_grid):
        with pytest.raises(ValueError):
            function(volume(), orientation)


def test_unavailable_axes_and_retained_axes_refuse() -> None:
    grid = volume().isel(a=0)
    with pytest.raises(ValueError, match=r"available.*AS"):
        reoriented(grid, "LS")
    with pytest.raises(ValueError, match=r"three varying dims.*retained"):
        cardinal_grid(grid, "axial")
    with pytest.raises(ValueError, match=r"unavailable.*frame directions"):
        cardinal_grid(volume(), ("dorsal-to-ventral", "left-to-right", "inferior-to-superior"))


@pytest.mark.parametrize(
    "spacing", [0, -1, [1, 0, 2], [1, 2], [[1, 2, 3]], float("nan"), float("inf")]
)
def test_invalid_spacing_refuses(spacing: npt.ArrayLike) -> None:
    with pytest.raises(ValueError, match=r"spacing"):
        cardinal_grid(volume(), "axial", spacing=spacing)


@pytest.mark.parametrize("dims", [("a", "a", "c"), ("a", "b")])
def test_invalid_dims_refuse(dims: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match=r"unique|three"):
        cardinal_grid(volume(), "axial", dims=dims)


def test_invalid_cover_and_missing_cells_refuse() -> None:
    with pytest.raises(ValueError, match=r"cover"):
        cardinal_grid(volume(), "axial", cover="invalid")
    with pytest.raises(ValueError, match=r"no cells.*samples"):
        cardinal_grid(volume(offsets=(None, 0.5, 0.5)), "axial")
    grid = volume(values=([0, 1], [0, 1], [0]))
    with pytest.raises(ValueError, match=r"undetermined.*explicit spacing"):
        cardinal_grid(grid, "axial")
    with pytest.raises(ValueError, match=r"single sample"):
        cardinal_grid(grid, "axial", spacing=1)
    assert cardinal_grid(grid, "axial", spacing=1, cover="samples").sizes["c"] == 1


@pytest.mark.parametrize(
    "values,match", [([], "empty"), ([0, 1, 1], "monotonic"), ([0, 2, 1], "monotonic")]
)
def test_undefined_step_direction_refuses(values: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        orientation_codes(volume(values=(values, [0, 1], [0, 1])))


def test_frame_and_zero_column_refusals() -> None:
    grid = volume()
    systems = [
        CoordinateSystem(("x", "y", "z"), ("mm",) * 3),
        CoordinateSystem(
            ("x", "y", "z"),
            ("mm", "cm", "mm"),
            axis_types=("space",) * 3,
            vocabulary=VOCABULARY,
            orientation=RAS,
        ),
        CoordinateSystem(("x", "y", "z"), ("mm",) * 3, vocabulary=VOCABULARY, orientation=RAS),
        CoordinateSystem(
            ("x", "y", "z"),
            ("mm",) * 3,
            axis_types=("space",) * 3,
            vocabulary=DirectionVocabulary("other", RAS),
            orientation=RAS,
        ),
    ]
    for system in systems:
        transform = AffineTransform.from_matrix(
            source=grid.transform.source,
            target=ReferenceFrame.local(system),
            matrix=np.eye(3),
            translation=[0, 0, 0],
        )
        with pytest.raises(ValueError, match=r"spatial axes|share one unit"):
            orientation_codes(Grid(transform, grid.coordinates))
    transform = AffineTransform.from_matrix(
        source=grid.transform.source,
        target=grid.frame,
        matrix=np.diag([0, 1, 1]),
        translation=[0, 0, 0],
    )
    with pytest.raises(ValueError, match=r"nonzero anatomical direction"):
        orientation_codes(Grid(transform, grid.coordinates))


def test_type_refusals() -> None:
    with pytest.raises(TypeError, match=r"Grid"):
        orientation_codes(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"orientation"):
        reoriented(volume(), ["R", "A", "S"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"spacing"):
        cardinal_grid(volume(), "axial", spacing="1")
    with pytest.raises(TypeError, match=r"sequence"):
        cardinal_grid(volume(), "axial", dims="abc")


def test_one_and_two_dimensional_assignments_ignore_retained_scalars() -> None:
    grid = volume()
    line = grid.isel(a=1, b=1)
    assert orientation_codes(reoriented(line, "I")) == "I"
    plane = grid.isel(c=1)
    assert orientation_codes(reoriented(plane, ("anterior-to-posterior", "right-to-left"))) == "PL"
    assert_allclose(reoriented(line, "I").points(), line.points()[::-1], rtol=0, atol=ATOL)


def test_singleton_integer_reflection_and_column_direction() -> None:
    grid = volume(
        values=([float(np.iinfo(np.int64).min)], [0, 1], [0, 1]), offsets=(None, 0.5, 0.5)
    )
    coordinates = dict(grid.coordinates)
    coordinates["u"] = ("a", [np.iinfo(np.int64).min])
    grid = Grid(grid.transform, coordinates)
    result = reoriented(grid, "LAS")
    assert orientation_codes(result) == "LAS"
    assert_allclose(result.points(), grid.points(), rtol=0, atol=ATOL)
    assert orientation_codes(volume(values=([7], [0, 1], [0, 1]))) == "RAS"


def test_singleton_without_interval_keeps_noncentred_offset_on_reflection() -> None:
    grid = volume(values=([2], [0, 1], [0, 1]), offsets=(0.1, 0.5, 0.5))
    reflected = reoriented(grid, "LAS")
    assert orientation_codes(reflected) == "LAS"
    assert reflected.transform.source.sample_offset == (0.1, 0.5, 0.5)  # type: ignore[union-attr]
    np.testing.assert_array_equal(reflected.points(), grid.points())
    assert reoriented(reflected, "RAS") == grid
    with pytest.raises(ValueError, match=r"single sample"):
        grid.points_at([[0, 0, 0]], domain="cells")


@pytest.mark.parametrize("offset", [0.1, 0.3])
def test_singleton_interval_reflection_preserves_exact_support(offset: float) -> None:
    grid = volume(
        values=([2], [0, 1], [0, 1]),
        offsets=(offset, 0.5, 0.5),
        intervals={"u": [[2 - offset, 3 - offset]]},
    )
    reflected = reoriented(grid, "LAS")
    assert isinstance(reflected.transform.source, ArrayCoordinates)
    reflected_offset = reflected.transform.source.sample_offset[0]
    assert reflected_offset is not None
    assert_allclose(
        reflected_offset,
        1 - offset,
        rtol=0,
        atol=np.spacing(1.0),
    )
    np.testing.assert_array_equal(reflected.points(), grid.points())
    np.testing.assert_array_equal(reflected.intervals["u"], -grid.intervals["u"][:, ::-1])
    restored = reoriented(reflected, "RAS")
    np.testing.assert_array_equal(restored.points(), grid.points())
    for name in grid.coordinates:
        original_entry, restored_entry = grid.coordinates[name], restored.coordinates[name]
        assert isinstance(original_entry, tuple) and isinstance(restored_entry, tuple)
        assert restored_entry[0] == original_entry[0]
        np.testing.assert_array_equal(restored_entry[1], original_entry[1])
    for name in grid.intervals:
        np.testing.assert_array_equal(restored.intervals[name], grid.intervals[name])
    assert isinstance(restored.transform.source, ArrayCoordinates)
    restored_offset = restored.transform.source.sample_offset[0]
    assert restored_offset is not None
    assert abs(restored_offset - offset) <= np.spacing(1.0)


def test_nonlinear_transform_refuses() -> None:
    grid = volume()

    class Warp:
        source = grid.transform.source
        target = grid.transform.target

        def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return np.asarray(points, dtype=np.float64) ** 2

    warped = Grid(Warp(), grid.coordinates)
    with pytest.raises(TypeError, match=r"affine"):
        orientation_codes(warped)
    for function in (reoriented, cardinal_grid):
        with pytest.raises(TypeError, match=r"affine"):
            function(warped, "RAS")


def test_too_many_dims_and_non_anatomical_displacement_refuse() -> None:
    system = CoordinateSystem(
        ("x", "y", "z", "t"),
        ("mm", "mm", "mm", "s"),
        axis_types=("space", "space", "space", "time"),
        vocabulary=VOCABULARY,
        orientation=(*RAS, None),
    )
    frame = ReferenceFrame.local(system)
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("a", "b", "c", "d"), ("1",) * 4),
        target=frame,
        matrix=np.eye(4),
        translation=np.zeros(4),
    )
    grid = Grid(transform, {dim: (dim, [0, 1]) for dim in "abcd"})
    with pytest.raises(ValueError, match=r"at most three"):
        orientation_codes(grid)
    with pytest.raises(ValueError, match=r"retained scalar"):
        cardinal_grid(grid.isel(d=0), "RAS", spacing=1, cover="samples")
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("a", "b", "c"), ("1",) * 3),
        target=frame,
        matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]),
        translation=np.zeros(4),
    )
    grid = Grid(transform, {dim: (dim, [0, 1]) for dim in "abc"})
    assert orientation_codes(grid) == "RAS"
    with pytest.raises(ValueError, match=r"non-anatomical"):
        cardinal_grid(grid, "RAS", spacing=1, cover="samples")


@pytest.mark.parametrize("cover", ["cells", "samples"])
@pytest.mark.parametrize("time_origin", [0.0, 1e15])
@pytest.mark.parametrize("cross_step", [1e-17, 1e-4])
@pytest.mark.parametrize("spatial_step", [1.0, 1000.0])
def test_cardinal_non_anatomical_cross_rows_allow_only_roundoff(
    cover: str, time_origin: float, cross_step: float, spatial_step: float
) -> None:
    system = CoordinateSystem(
        ("x", "y", "z", "t"),
        ("mm", "mm", "mm", "s"),
        axis_types=("space", "space", "space", "time"),
        vocabulary=VOCABULARY,
        orientation=(*RAS, None),
    )
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("a", "b", "c"), ("1",) * 3, sample_offset=(0.5,) * 3),
        target=ReferenceFrame.local(system),
        # The same anatomy in finer units must not loosen the check on the time axis.
        matrix=np.array(
            [[spatial_step, 0, 0], [0, spatial_step, 0], [0, 0, spatial_step], [0, 0, cross_step]]
        ),
        translation=[0, 0, 0, time_origin],
    )
    grid = Grid(transform, {dim: (dim, [0, 1]) for dim in "abc"})
    if cross_step > 1e-6:
        with pytest.raises(ValueError, match=r"non-anatomical"):
            cardinal_grid(grid, "RAS", spacing=spatial_step, cover=cover)
    else:
        output = cardinal_grid(grid, "RAS", spacing=spatial_step, cover=cover)
        assert orientation_codes(output) == "RAS"
        assert isinstance(output.transform, AffineTransform)
        np.testing.assert_array_equal(output.transform.matrix[3], [0, 0, 0])


@pytest.mark.parametrize("coordinate_scale", [1.0, 1e15])
@pytest.mark.parametrize("spatial_step,unit", [(1.0, "mm"), (1000.0, "um")])
@pytest.mark.parametrize("cross_step", [1.0, 1e-17])
def test_cardinal_cross_row_bound_matches_each_column_to_its_coordinate_span(
    coordinate_scale: float, spatial_step: float, unit: str, cross_step: float
) -> None:
    system = CoordinateSystem(
        ("x", "y", "z", "t"),
        (unit, unit, unit, "s"),
        axis_types=("space", "space", "space", "time"),
        vocabulary=VOCABULARY,
        orientation=(*RAS, None),
    )
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("a", "b", "c"), ("1",) * 3),
        target=ReferenceFrame.local(system),
        matrix=np.array(
            [[spatial_step, 0, 0], [0, spatial_step, 0], [0, 0, spatial_step], [0, 0, cross_step]]
        )
        / [coordinate_scale, 1, 1],
        translation=np.zeros(4),
    )
    grid = Grid(
        transform, {"a": ("a", [0, coordinate_scale]), "b": ("b", [0, 1]), "c": ("c", [0, 1])}
    )
    if cross_step > 1e-6:
        with pytest.raises(ValueError, match="non-anatomical"):
            cardinal_grid(grid, "RAS", spacing=spatial_step, dims=("x", "y", "z"), cover="samples")
    else:
        output = cardinal_grid(
            grid, "RAS", spacing=spatial_step, dims=("x", "y", "z"), cover="samples"
        )
        assert isinstance(output.transform, AffineTransform)
        np.testing.assert_array_equal(output.transform.matrix[3], [0, 0, 0])
        assert_allclose(output.lattice().matrix[:3], np.eye(3) * spatial_step, rtol=0, atol=ATOL)


def test_coverage_count_roundoff_keeps_exact_spacing() -> None:
    grid = volume(values=([0, 1, 2 + 1e-10], [0, 1], [0, 1]))
    output = cardinal_grid(grid, "RAS", spacing=2, cover="samples")
    assert output.sizes["a"] == 3
    assert_allclose(output.lattice().spacing, [2, 2, 2], rtol=0, atol=ATOL)
    grid = volume(values=([0, 1, 2.25], [0, 1], [0, 1]))
    output = cardinal_grid(grid, "RAS", spacing=2, cover="samples")
    assert output.sizes["a"] == 4
    assert_allclose(output.lattice().spacing, [2, 2, 2], rtol=0, atol=ATOL)
    with pytest.raises(ValueError, match=r"too many int64 samples"):
        cardinal_grid(grid, "RAS", spacing=1e-310, cover="samples")


@pytest.mark.parametrize(
    "plane,expected",
    [("axial", "SPL"), ("transverse", "SPL"), ("coronal", "PIL"), ("sagittal", "RIP")],
)
def test_reorientation_accepts_named_planes(plane: str, expected: str) -> None:
    grid = volume(orientation=LPS)
    assert orientation_codes(reoriented(grid, plane)) == expected
    assert reoriented(reoriented(grid, plane), "LPS") == grid


def test_descending_nonuniform_noncentred_cells_are_covered() -> None:
    grid = volume(angle=20, values=([5, 3, 0], [0, 2, 5], [4, 0]), offsets=(0.25, 0.75, 0.5))
    output = cardinal_grid(grid, "sagittal", spacing=1)
    positions = output.positions_at(corners(grid, "cells"), domain="cells")
    assert_allclose(positions.min(axis=0), [-0.5] * 3, rtol=0, atol=ATOL)
    assert np.all(positions.max(axis=0) <= np.array(list(output.sizes.values())) - 0.5 + ATOL)


def test_declared_cell_widths_and_gaps_determine_coverage() -> None:
    grid = volume(angle=20, intervals={"u": np.array([[-2, 2], [0.5, 1.5], [0, 4]])})
    output = cardinal_grid(grid, "coronal", spacing=1)
    positions = output.positions_at(corners(grid, "cells"), domain="cells")
    assert_allclose(positions.min(axis=0), [-0.5] * 3, rtol=0, atol=ATOL)
    assert np.all(positions.max(axis=0) <= np.array(list(output.sizes.values())) - 0.5 + ATOL)


def test_reflected_slab_preserves_requested_spacing_in_perpendicular_plane() -> None:
    grid = volume(values=([0, 1, 2], [0, 1, 2], [2]), intervals={"w": [[1.25, 2.75]]})
    grid = reoriented(grid, "RAI")
    output = cardinal_grid(grid, "coronal", spacing=1)
    assert output.sizes["c"] == 6
    assert not output.intervals
    assert_allclose(output.lattice().spacing, [1, 1, 1], rtol=0, atol=ATOL)
    assert_allclose(
        output.positions_at(corners(grid, "cells"), domain="cells").min(axis=0),
        [-0.5] * 3,
        rtol=0,
        atol=ATOL,
    )
