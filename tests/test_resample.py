"""Public behavior of :func:`xarrayrf.resample`."""

from __future__ import annotations

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from xarray.indexes import RangeIndex

import xarrayrf as xrf

pytest.importorskip("scipy", minversion="1.18")
ndimage = pytest.importorskip("scipy.ndimage")

ATOL = 1e-10
"""Allowance for interpolation arithmetic on small exact synthetic values."""

ANATOMY = xrf.DirectionVocabulary(
    "test-anatomy", ("right-to-left", "anterior-to-posterior", "inferior-to-superior")
)
LPS = xrf.ReferenceFrame.declared(
    ("test", "patient"),
    xrf.CoordinateSystem(
        ("x", "y", "z"),
        ("mm",) * 3,
        vocabulary=ANATOMY,
        orientation=("right-to-left", "anterior-to-posterior", "inferior-to-superior"),
    ),
)
RAS = LPS.with_coordinate_system(
    xrf.CoordinateSystem(
        ("x", "y", "z"),
        ("mm",) * 3,
        vocabulary=ANATOMY,
        orientation=("left-to-right", "posterior-to-anterior", "inferior-to-superior"),
    )
)
IJK = xrf.ArrayCoordinates(("i", "j", "k"), ("1", "1", "1"))


def volume(
    values: npt.ArrayLike,
    frame: xrf.ReferenceFrame = LPS,
    matrix: npt.ArrayLike = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    translation: npt.ArrayLike = (0.0, 0.0, 0.0),
    extra: dict[str, object] | None = None,
) -> xrf.Geometry:
    data = np.asarray(values)
    shape = data.shape[-3:]
    dims = (*(("echo",) if data.ndim == 4 else ()), "k", "j", "i")
    coords: dict[str, object] = {
        "k": np.arange(shape[0]),
        "j": np.arange(shape[1]),
        "i": np.arange(shape[2]),
    }
    coords.update(extra or {})
    array = xr.DataArray(data, dims=dims, coords=coords)
    transform = xrf.AffineTransform.from_matrix(
        source=IJK, target=frame, matrix=matrix, translation=translation
    )
    return xrf.Geometry(array, transform, dims=("k", "j", "i"))


def ramp(shape: tuple[int, int, int] = (4, 5, 6)) -> npt.NDArray[np.float64]:
    """A field linear in i, j, k, so linear interpolation reproduces it exactly."""
    k, j, i = np.indices(shape, dtype=np.float64)
    field: npt.NDArray[np.float64] = 1.0 + 2.0 * i + 3.0 * j + 5.0 * k
    return field


class Shift:
    """A non-affine-typed point transform between frames, forcing the general path."""

    def __init__(self, source: xrf.ReferenceFrame, target: xrf.ReferenceFrame, offset: float):
        self._source = source
        self._target = target
        self._offset = offset

    @property
    def source(self) -> xrf.Endpoint:
        return self._source

    @property
    def target(self) -> xrf.Endpoint:
        return self._target

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        shifted: npt.NDArray[np.float64] = np.asarray(points, dtype=np.float64) + np.array(
            [self._offset, 0.0, 0.0]
        )
        return shifted


class PointOnly:
    """Expose an affine transform only as a point mapping to force the general path."""

    def __init__(self, transform: xrf.AffineTransform):
        self.source = transform.source
        self.target = transform.target
        self._transform = transform

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return self._transform.transform_point(points)


@pytest.mark.parametrize(
    ("shape", "matrix", "translation"),
    [
        ((4, 5, 6), np.eye(3), (0.0, 0.0, 0.0)),
        ((2, 3, 4), np.eye(3), (1.0, 1.0, 1.0)),
        ((4, 5, 6), np.diag([-1.0, 1.0, 1.0]), (5.0, 0.0, 0.0)),
        ((6, 5, 4), np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), (0.0, 0.0, 0.0)),
        ((4, 5, 7), np.eye(3), (-1.0, 0.0, 0.0)),
    ],
)
@pytest.mark.parametrize("domain", ["samples", "cells"])
def test_same_grid_matches_general_path_without_affine_interpolation(
    shape: tuple[int, int, int],
    matrix: npt.ArrayLike,
    translation: tuple[float, ...],
    domain: xrf.Domain,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = volume(ramp())
    target = volume(np.zeros(shape), matrix=matrix, translation=translation)
    assert isinstance(target.transform, xrf.AffineTransform)
    general = xrf.Geometry(target.array, PointOnly(target.transform), dims=target.dims)
    expected = xrf.resample(source, general, domain=domain, fill_value=-9.0).values

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("same-grid path called affine_transform")

    monkeypatch.setattr(ndimage, "affine_transform", forbidden)
    actual = xrf.resample(source, target, domain=domain, fill_value=-9.0).values
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("scale,shape", [(1.0, (4, 5, 6)), (1.0 + 5e-10, (4, 5, 5000))])
def test_nonintegral_or_drifting_grid_uses_interpolation(
    scale: float,
    shape: tuple[int, int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation = (0.5, 0.0, 0.0) if scale == 1.0 else (0.0, 0.0, 0.0)
    target = volume(np.zeros(shape), matrix=np.diag([scale, 1.0, 1.0]), translation=translation)
    called = []
    original = ndimage.affine_transform

    def counted(*args: object, **kwargs: object) -> object:
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(ndimage, "affine_transform", counted)
    xrf.resample(volume(ramp()), target)
    assert called


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_resampling_onto_the_same_samples_is_the_identity(method: xrf.Method) -> None:
    source = volume(ramp())
    result = xrf.resample(source, source, method=method)
    assert not np.isnan(result.values).any()
    assert_allclose(result.values, source.array.values, rtol=0, atol=ATOL)


def test_a_shift_moves_values_and_fills_outside() -> None:
    """Target sample i sits at x = i + 1, which is source sample i + 1."""
    source = volume(ramp())
    target = volume(np.zeros((4, 5, 6)), translation=(1.0, 0.0, 0.0))
    result = xrf.resample(source, target)
    assert_allclose(result.values[..., :5], source.array.values[..., 1:], rtol=0, atol=ATOL)
    assert np.isnan(result.values[..., 5]).all()


def test_linear_resampling_is_exact_on_a_linear_field_under_rotation() -> None:
    angle = np.radians(20.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1.0]]
    )
    source = volume(ramp((6, 12, 12)))
    target = volume(np.zeros((6, 4, 4)), matrix=0.5 * rotation, translation=(5.0, 5.0, 0.0))
    result = xrf.resample(source, target)
    points = target.points().values
    expected = 1.0 + 2.0 * points[..., 0] + 3.0 * points[..., 1] + 5.0 * points[..., 2]
    assert_allclose(result.values, expected, rtol=0, atol=ATOL)


def test_the_general_path_agrees_with_the_lattice_path() -> None:
    other = xrf.ReferenceFrame.declared(("test", "moved"), LPS.coordinate_system)
    source = volume(ramp((5, 6, 7)))
    target = volume(np.zeros((5, 6, 7)), frame=other)
    affine = xrf.AffineTransform.from_matrix(
        source=other, target=LPS, matrix=np.eye(3), translation=(0.25, 0.0, 0.0)
    )
    lattice_result = xrf.resample(source, target, transform=affine)
    general_result = xrf.resample(source, target, transform=Shift(other, LPS, 0.25))
    assert_allclose(general_result.values, lattice_result.values, rtol=0, atol=ATOL, equal_nan=True)


def test_nonuniform_slice_offsets_are_interpolated_by_position() -> None:
    """Slices at 0, 2 and 5 mm; a target slice at 3.5 mm lies halfway between the last two."""
    values = np.stack([np.full((2, 2), value) for value in (10.0, 20.0, 50.0)])
    array = xr.DataArray(
        values,
        dims=("slice", "row", "column"),
        coords={
            "slice_offset": ("slice", [0.0, 2.0, 5.0]),
            "row": [0, 1],
            "column": [0, 1],
        },
    )
    stack = xrf.Geometry(
        array,
        xrf.AffineTransform.from_matrix(
            source=xrf.ArrayCoordinates(("column", "row", "slice_offset"), ("1", "1", "mm")),
            target=LPS,
            matrix=np.eye(3),
            translation=(0.0, 0.0, 0.0),
        ),
        dims=("slice", "row", "column"),
    )
    target = volume(np.zeros((1, 2, 2)), translation=(0.0, 0.0, 3.5))
    result = xrf.resample(stack, target)
    assert_allclose(result.values, 35.0, rtol=0, atol=ATOL)


def test_equivalent_frames_in_other_coordinate_systems_convert_automatically() -> None:
    """A RAS target sample at (-1, -2, 3) is the LPS source point (1, 2, 3)."""
    source = volume(ramp())
    target = volume(np.zeros((4, 5, 6)), frame=RAS, matrix=-np.diag([1.0, 1.0, -1.0]))
    result = xrf.resample(source, target)
    assert_allclose(result.values, source.array.values, rtol=0, atol=ATOL)


def test_different_frames_need_a_transform() -> None:
    other = xrf.ReferenceFrame.declared(("test", "other"), LPS.coordinate_system)
    source = volume(ramp())
    target = volume(np.zeros((4, 5, 6)), frame=other)
    with pytest.raises(ValueError, match="different frames; supply the transform"):
        xrf.resample(source, target)
    registration = xrf.AffineTransform.from_matrix(
        source=other, target=LPS, matrix=np.eye(3), translation=(0.0, 0.0, 0.0)
    )
    assert_allclose(
        xrf.resample(source, target, transform=registration).values,
        source.array.values,
        atol=ATOL,
    )
    with pytest.raises(ValueError, match="applies only to the frames it was computed between"):
        xrf.resample(source, target, transform=registration.inverse())


def test_non_geometry_dimensions_are_carried_through() -> None:
    echoes = np.stack([ramp(), 10.0 * ramp()])
    source = volume(echoes, extra={"echo": [10.0, 20.0]})
    target = volume(np.zeros((4, 5, 6)), translation=(1.0, 0.0, 0.0))
    result = xrf.resample(source, target)
    assert result.dims == ("echo", "k", "j", "i")
    assert list(result.echo.values) == [10.0, 20.0]
    assert_allclose(result.values[1], 10.0 * result.values[0], rtol=0, atol=ATOL, equal_nan=True)


def test_a_dask_source_is_resampled_lazily_per_non_geometry_chunk() -> None:
    echoes = np.stack([ramp(), 10.0 * ramp()])
    eager = volume(echoes)
    lazy = xrf.Geometry(eager.array.chunk({"echo": 1}), eager.transform, dims=eager.dims)
    result = xrf.resample(lazy, volume(np.zeros((4, 5, 6)), translation=(0.5, 0.0, 0.0)))
    assert isinstance(result.data, da.Array)
    expected = xrf.resample(eager, volume(np.zeros((4, 5, 6)), translation=(0.5, 0.0, 0.0)))
    assert_allclose(result.values, expected.values, rtol=0, atol=ATOL, equal_nan=True)


def test_nearest_keeps_integer_labels_when_the_fill_fits() -> None:
    labels = volume(np.arange(120, dtype=np.int16).reshape(4, 5, 6))
    target = volume(np.zeros((4, 5, 6)), translation=(0.4, 0.0, 0.0))
    kept = xrf.resample(labels, target, method="nearest", fill_value=0)
    assert kept.dtype == np.int16
    assert xrf.resample(labels, target, method="linear").dtype == np.float64
    assert xrf.resample(labels, target, method="nearest").dtype == np.float64


def test_small_blocks_give_the_same_result() -> None:
    other = xrf.ReferenceFrame.declared(("test", "moved"), LPS.coordinate_system)
    source = volume(ramp())
    target = volume(np.zeros((4, 5, 6)), frame=other)
    whole = xrf.resample(source, target, transform=Shift(other, LPS, 0.5))
    blocked = xrf.resample(source, target, transform=Shift(other, LPS, 0.5), block_points=7)
    assert_allclose(blocked.values, whole.values, rtol=0, atol=ATOL, equal_nan=True)


def test_the_target_coordinates_are_carried() -> None:
    source = volume(ramp())
    target = volume(np.zeros((2, 5, 6)), translation=(0.0, 0.0, 1.0))
    result = xrf.resample(source, target)
    assert list(result.k.values) == [0, 1]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        pytest.param({"method": "quintic"}, ValueError, "method must be one of", id="method"),
        pytest.param({"block_points": 0}, ValueError, "positive integer", id="block"),
    ],
)
def test_arguments_are_validated(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    source = volume(ramp())
    with pytest.raises(error, match=message):
        xrf.resample(source, source, **kwargs)  # type: ignore[arg-type]


def test_arguments_must_be_geometries() -> None:
    with pytest.raises(TypeError, match="source must be Geometry"):
        xrf.resample(volume(ramp()).array, volume(ramp()))  # type: ignore[arg-type]


def test_a_target_dimension_may_not_be_a_source_non_geometry_dimension() -> None:
    source = volume(np.stack([ramp(), ramp()]), extra={"echo": [1.0, 2.0]})
    array = xr.DataArray(
        np.zeros((2, 5, 6)),
        dims=("echo", "j", "i"),
        coords={"echo": [0, 1], "j": np.arange(5), "i": np.arange(6)},
    )
    target = xrf.Geometry(
        array,
        xrf.AffineTransform.from_matrix(
            source=xrf.ArrayCoordinates(("i", "j", "echo"), ("1", "1", "1")),
            target=LPS,
            matrix=np.eye(3),
            translation=(0.0, 0.0, 0.0),
        ),
        dims=("echo", "j", "i"),
    )
    with pytest.raises(ValueError, match="non-geometry dimensions of the source"):
        xrf.resample(source, target)


def test_a_plane_source_cannot_be_located_in_a_volume() -> None:
    plane = xr.DataArray(
        np.ones((5, 6)), dims=("j", "i"), coords={"j": np.arange(5), "i": np.arange(6)}
    )
    source = xrf.Geometry(
        plane,
        xrf.AffineTransform.from_matrix(
            source=xrf.ArrayCoordinates(("i", "j"), ("1", "1")),
            target=LPS,
            matrix=[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            translation=(0.0, 0.0, 0.0),
        ),
        dims=("j", "i"),
    )
    with pytest.raises(ValueError, match="no inverse"):
        xrf.resample(source, volume(np.zeros((4, 5, 6))))


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_complex_values_are_resampled(method: xrf.Method) -> None:
    source = volume(ramp() + 1j * ramp())
    result = xrf.resample(source, source, method=method)
    assert result.dtype == np.complex128
    assert_allclose(result.values, source.array.values, rtol=0, atol=ATOL)


def test_cubic_agrees_between_the_two_paths() -> None:
    other = xrf.ReferenceFrame.declared(("test", "moved"), LPS.coordinate_system)
    source = volume(np.random.default_rng(1).random((5, 6, 7)))
    target = volume(np.zeros((5, 6, 7)), frame=other)
    affine = xrf.AffineTransform.from_matrix(
        source=other, target=LPS, matrix=np.eye(3), translation=(0.3, 0.0, 0.0)
    )
    fast = xrf.resample(source, target, transform=affine, method="cubic")
    general = xrf.resample(source, target, transform=Shift(other, LPS, 0.3), method="cubic")
    blocked = xrf.resample(
        source, target, transform=Shift(other, LPS, 0.3), method="cubic", block_points=11
    )
    assert_allclose(general.values, fast.values, rtol=0, atol=ATOL, equal_nan=True)
    assert_allclose(blocked.values, fast.values, rtol=0, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("general", [False, True])
def test_a_finite_fill_value_marks_outside_samples(general: bool) -> None:
    other = xrf.ReferenceFrame.declared(("test", "moved"), LPS.coordinate_system)
    source = volume(ramp())
    target = volume(np.zeros((4, 5, 6)), frame=other)
    transform: xrf.SupportsPoints = (
        Shift(other, LPS, 1.0)
        if general
        else xrf.AffineTransform.from_matrix(
            source=other, target=LPS, matrix=np.eye(3), translation=(1.0, 0.0, 0.0)
        )
    )
    result = xrf.resample(source, target, transform=transform, fill_value=-7.0)
    assert (result.values[..., 5] == -7.0).all()
    assert_allclose(result.values[..., :5], source.array.values[..., 1:], rtol=0, atol=ATOL)


def test_a_single_slice_target_is_resampled_like_a_slab() -> None:
    source = volume(ramp())
    target = volume(np.zeros((1, 5, 6)), translation=(0.0, 0.0, 2.0))
    result = xrf.resample(source, target)
    assert result.shape == (1, 5, 6)
    assert_allclose(result.values[0], source.array.values[2], rtol=0, atol=ATOL)


def test_a_plane_target_samples_a_volume() -> None:
    source = volume(ramp())
    plane = xr.DataArray(
        np.zeros((5, 6)), dims=("j", "i"), coords={"j": np.arange(5), "i": np.arange(6)}
    )
    target = xrf.Geometry(
        plane,
        xrf.AffineTransform.from_matrix(
            source=xrf.ArrayCoordinates(("i", "j"), ("1", "1")),
            target=LPS,
            matrix=[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            translation=(0.0, 0.0, 3.0),
        ),
        dims=("j", "i"),
    )
    result = xrf.resample(source, target)
    assert result.dims == ("j", "i")
    assert_allclose(result.values, source.array.values[3], rtol=0, atol=ATOL)


def test_descending_coordinates_are_located() -> None:
    """Stored k runs 3, 2, 1, 0, so array position 0 holds the sample at k = 3."""
    data = ramp()[::-1]
    array = xr.DataArray(
        data, dims=("k", "j", "i"), coords={"k": [3, 2, 1, 0], "j": np.arange(5), "i": np.arange(6)}
    )
    descending = xrf.Geometry(
        array,
        xrf.AffineTransform.from_matrix(
            source=IJK, target=LPS, matrix=np.eye(3), translation=(0.0, 0.0, 0.0)
        ),
        dims=("k", "j", "i"),
    )
    shifted = xrf.resample(descending, volume(np.zeros((4, 5, 6)), frame=LPS), transform=None)
    assert_allclose(shifted.values, ramp(), rtol=0, atol=ATOL)


def test_a_dask_source_follows_the_general_path_lazily() -> None:
    other = xrf.ReferenceFrame.declared(("test", "moved"), LPS.coordinate_system)
    echoes = volume(np.stack([ramp(), 2.0 * ramp()]))
    lazy = xrf.Geometry(echoes.array.chunk({"echo": 1}), echoes.transform, dims=echoes.dims)
    target = volume(np.zeros((4, 5, 6)), frame=other)
    result = xrf.resample(lazy, target, transform=Shift(other, LPS, 0.5))
    assert isinstance(result.data, da.Array)
    eager = xrf.resample(echoes, target, transform=Shift(other, LPS, 0.5))
    assert_allclose(result.values, eager.values, rtol=0, atol=ATOL, equal_nan=True)


def test_a_boolean_mask_keeps_its_dtype_under_nearest() -> None:
    mask = volume(ramp() > 20.0)
    result = xrf.resample(mask, mask, method="nearest", fill_value=0)
    assert result.dtype == np.bool_
    assert (result.values == mask.array.values).all()


def test_the_transform_argument_must_be_a_transform() -> None:
    with pytest.raises(TypeError, match="must implement SupportsPoints"):
        xrf.resample(volume(ramp()), volume(ramp()), transform="identity")  # type: ignore[arg-type]


def test_a_boolean_block_size_is_refused() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        xrf.resample(volume(ramp()), volume(ramp()), block_points=True)


CENTRED = xrf.ArrayCoordinates(("i", "j", "k"), ("1", "1", "1"), sample_offset=(0.5, 0.5, 0.5))


def centred(geometry: xrf.Geometry) -> xrf.Geometry:
    transform = geometry.transform
    assert isinstance(transform, xrf.AffineTransform)
    return xrf.Geometry(
        geometry.array,
        xrf.AffineTransform.from_matrix(
            source=CENTRED,
            target=transform.target,
            matrix=transform.matrix,
            translation=transform.translation,
        ),
        dims=geometry.dims,
    )


@pytest.mark.parametrize("general", [False, True])
@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_the_cells_domain_holds_the_edge_value_half_a_step_out(
    method: xrf.Method, general: bool
) -> None:
    """Target x runs -0.5 to 6.5; the source samples run 0 to 5 and their cells -0.5 to 5.5."""
    source = centred(volume(ramp()))
    target = volume(np.zeros((4, 5, 8)), translation=(-0.5, 0.0, 0.0))
    transform = Shift(LPS, LPS, 0.0) if general else None
    samples = xrf.resample(source, target, transform=transform, method=method)
    cells = xrf.resample(source, target, transform=transform, method=method, domain="cells")
    assert np.isnan(samples.isel(i=[0, 6, 7])).all()
    edges = source.array.isel(i=[0, 5]).values
    assert_allclose(cells.isel(i=[0, 6]).values, edges, rtol=0, atol=ATOL)
    assert np.isnan(cells.isel(i=7)).all()
    assert_allclose(cells.isel(i=slice(1, 6)), samples.isel(i=slice(1, 6)), rtol=0, atol=1e-9)


def test_the_cells_domain_refuses_a_single_sample_with_cells_and_unknown_domains() -> None:
    source = centred(volume(ramp((1, 5, 6))))
    with pytest.raises(ValueError, match="declares cells but has a single sample"):
        xrf.resample(source, source, domain="cells")
    with pytest.raises(ValueError, match="domain must be"):
        xrf.resample(source, source, domain="voxels")  # type: ignore[arg-type]


def test_point_sampled_axes_keep_the_samples_domain_under_cells() -> None:
    source = volume(ramp())
    target = volume(np.zeros((4, 5, 8)), translation=(-0.5, 0.0, 0.0))
    samples = xrf.resample(source, target).values
    assert_array_equal(
        np.isnan(xrf.resample(source, target, domain="cells").values), np.isnan(samples)
    )


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_both_paths_agree_on_rotated_cells_with_asymmetric_offsets(method: xrf.Method) -> None:
    """Offsets 0, 0.25 and 1 on a rotated, descending-in-k source; the target reaches every face."""
    angle = np.deg2rad(20.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    data = ramp((5, 6, 7))
    array = xr.DataArray(
        data,
        dims=("k", "j", "i"),
        coords={"k": np.arange(5)[::-1].astype(float), "j": np.arange(6), "i": np.arange(7)},
    )
    coordinates = xrf.ArrayCoordinates(("i", "j", "k"), ("1", "1", "1"), sample_offset=(0, 0.25, 1))
    source = xrf.Geometry(
        array,
        xrf.AffineTransform.from_matrix(
            source=coordinates, target=LPS, matrix=rotation, translation=(0, 0, 0)
        ),
        dims=("k", "j", "i"),
    )
    target = volume(np.zeros((9, 11, 12)), translation=(-3.0, -1.5, -1.8), matrix=np.eye(3) * 0.8)
    lattice = xrf.resample(source, target, method=method, domain="cells").values
    general = xrf.resample(
        source, target, transform=Shift(LPS, LPS, 0.0), method=method, domain="cells"
    ).values
    assert_array_equal(np.isnan(lattice), np.isnan(general))
    assert_allclose(lattice, general, rtol=0, atol=1e-9, equal_nan=True)
    samples = xrf.resample(source, target, method=method).values
    shell = np.isnan(samples) & ~np.isnan(lattice)
    assert shell.any()
    assert lattice[shell].min() >= data.min() - 1e-9 and lattice[shell].max() <= data.max() + 1e-9


def test_a_dask_source_holds_the_edge_value_in_the_cells() -> None:
    source = centred(volume(ramp()))
    lazy = xrf.Geometry(source.array.chunk({"k": 2}), source.transform, dims=source.dims)
    target = volume(np.zeros((4, 5, 8)), translation=(-0.5, 0.0, 0.0))
    eager = xrf.resample(source, target, method="cubic", domain="cells")
    result = xrf.resample(lazy, target, method="cubic", domain="cells")
    assert isinstance(result.data, da.Array)
    assert_allclose(result.values, eager.values, rtol=0, atol=ATOL, equal_nan=True)


def test_cropping_the_source_commutes_with_resampling() -> None:
    """A crop keeps its coordinates, so the cropped source's samples keep their places.

    In the cells domain the cropped source also answers half a step beyond its outer samples,
    holding their values where the whole source still has neighbours.
    """
    source = centred(volume(ramp((6, 5, 6))))
    target = volume(np.zeros((6, 5, 6)), translation=(0.3, -0.2, 0.4))
    cropped = xrf.Geometry(
        source.array.isel(k=slice(1, 5), i=slice(2, 6)), source.transform, dims=source.dims
    )
    whole = xrf.resample(source, target).values
    part = xrf.resample(cropped, target).values
    inside = ~np.isnan(part)
    assert inside.any()
    assert_allclose(part[inside], whole[inside], rtol=0, atol=ATOL)
    cells = xrf.resample(cropped, target, domain="cells").values
    assert_allclose(cells[inside], part[inside], rtol=0, atol=ATOL)
    assert (~np.isnan(cells) & ~inside).any()


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize("domain", ["samples", "cells"])
@pytest.mark.parametrize(
    ("shape", "matrix", "translation"),
    [
        ((4, 6, 5), np.eye(3), (7.3, 11.6, 2.2)),  # interior crop, non-integral offset
        ((5, 7, 6), [[0.0, 0.9, 0.3], [-0.9, 0.0, 0.2], [0.1, 0.0, 1.1]], (20.03, 8.07, 3.01)),
        ((3, 9, 9), -np.eye(3), (12.37, 30.21, 40.13)),  # reversed, partly outside the source
    ],
)
def test_cropping_the_source_to_the_target_footprint_changes_no_value(
    method: xrf.Method,
    domain: xrf.Domain,
    shape: tuple[int, int, int],
    matrix: npt.ArrayLike,
    translation: tuple[float, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(3)
    source = volume(rng.normal(size=(16, 40, 36)))
    target = volume(np.zeros(shape), matrix=matrix, translation=translation)
    cropped = xrf.resample(source, target, method=method, domain=domain, fill_value=-9.0).values
    monkeypatch.setattr("xarrayrf._resample._crop_window", lambda *args: None)
    whole = xrf.resample(source, target, method=method, domain=domain, fill_value=-9.0).values
    # The cubic prefilter margin leaves ~2e-14 of a boundary change (CROP_MARGIN). Offsets
    # avoid exact half-sample ties, where nearest may pick either equidistant neighbour.
    assert_allclose(cropped, whole, rtol=0, atol=1e-11)


class _RecordingArray:
    """A lazily indexable source that records every read."""

    shape = (8, 64, 64)
    dtype = np.dtype(np.float64)
    ndim = 3

    def __init__(self) -> None:
        self.reads: list[tuple[slice, ...]] = []

    def __getitem__(self, key: tuple[slice, ...]) -> npt.NDArray[np.float64]:
        self.reads.append(key)
        return np.ones(self.shape)[key]


def test_a_small_target_reads_only_the_source_chunks_it_touches() -> None:
    recording = _RecordingArray()
    lazy = da.from_array(recording, chunks=(8, 16, 16))  # type: ignore[no-untyped-call]
    source = volume(np.zeros((8, 64, 64)))
    source = xrf.Geometry(source.array.copy(data=lazy), source.transform, dims=source.dims)
    target = volume(np.zeros((4, 8, 8)), translation=(2.0, 3.0, 2.0))
    xrf.resample(source, target, method="linear").compute()
    reads = [key for key in recording.reads if all(part.stop for part in key)]  # not probes
    assert 0 < len(reads) <= 4  # of 16 chunks: the target and its margin fit in one or two


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize(
    ("position", "domain"),
    [(2.25, "samples"), (2.0, "samples"), (-0.25, "samples"), (-0.25, "cells"), (-0.75, "cells")],
)
def test_scalar_volume_target_matches_singleton_target(
    method: xrf.Method, position: float, domain: xrf.Domain
) -> None:
    source = centred(volume(ramp()))
    lazy = xrf.Geometry(source.array.chunk({"k": 2}), source.transform, dims=source.dims)
    for transform in (source.transform, xrf.CompositeTransform(source.transform)):
        scalar = xrf.Grid(transform, {"i": position, "j": 1.0, "k": 1.0})
        singleton = xrf.Grid(
            transform, {"i": ("i", [position]), "j": ("j", [1.0]), "k": ("k", [1.0])}
        )
        result = xrf.resample(source, scalar, method=method, domain=domain, fill_value=-9)
        expected = xrf.resample(source, singleton, method=method, domain=domain, fill_value=-9)
        assert result.dims == ()
        assert_allclose(result, expected.values.item(), rtol=0, atol=ATOL)
        assert_allclose(result.i, position, rtol=0, atol=ATOL)
        chunked = xrf.resample(lazy, scalar, method=method, domain=domain, fill_value=-9)
        assert chunked.chunks == ()
        assert_allclose(chunked.compute(), result, rtol=0, atol=ATOL)
        if position < 0:
            edge = xrf.Grid(transform, {"i": ("i", [0.0]), "j": ("j", [1.0]), "k": ("k", [1.0])})
            held = xrf.resample(source, edge, method=method).values.item()
            value = held if domain == "cells" and position > -0.5 else -9
            assert_allclose(result, value, rtol=0, atol=ATOL)


def test_scalar_affine_target_reads_only_nearby_source_chunks() -> None:
    recording = _RecordingArray()
    lazy = da.from_array(recording, chunks=(8, 16, 16))  # type: ignore[no-untyped-call]
    source = volume(np.zeros(recording.shape))
    source = xrf.Geometry(source.array.copy(data=lazy), source.transform, dims=source.dims)
    target = xrf.Grid(source.transform, {"i": 2.5, "j": 3.5, "k": 2.5})
    recording.reads.clear()
    result = xrf.resample(source, target)
    assert not recording.reads
    assert result.chunks == ()
    assert_allclose(result.compute(), 1.0, rtol=0, atol=ATOL)
    assert len(recording.reads) == 1


@pytest.mark.parametrize("domain", ["samples", "cells"])
def test_scalar_target_does_not_enable_scalar_source_inversion(domain: xrf.Domain) -> None:
    source = volume(ramp())
    target = xrf.Grid(source.transform, {"i": 1.0, "j": 1.0, "k": 1.0})
    for selection in ({"i": 1}, {"i": 1, "j": 1, "k": 1}):
        selected = source.array.isel(selection)
        geometry = xrf.Geometry(
            selected,
            source.transform,
            dims=tuple(dim for dim in source.dims if dim not in selection),
        )
        with pytest.raises(ValueError, match=r"retained scalar.*projection policy"):
            xrf.resample(geometry, target, domain=domain)


def test_a_transform_applies_only_to_the_frames_it_was_computed_between() -> None:
    other = xrf.ReferenceFrame.declared(("test", "other patient"), LPS.coordinate_system)
    source = volume(ramp(), frame=other)
    target = volume(np.zeros((2, 2, 2)))
    registration = xrf.AffineTransform.from_matrix(
        source=LPS, target=other, matrix=np.eye(3), translation=np.zeros(3)
    )
    xrf.resample(source, target, transform=registration)  # computed between these frames
    third = xrf.ReferenceFrame.declared(("test", "third patient"), LPS.coordinate_system)
    with pytest.raises(
        ValueError, match=r"maps test:patient to test:other patient.*computed between"
    ):
        xrf.resample(volume(ramp(), frame=third), target, transform=registration)
    in_ras = xrf.AffineTransform.from_matrix(
        source=RAS, target=other, matrix=np.eye(3), translation=np.zeros(3)
    )
    with pytest.raises(ValueError, match="same frames, different coordinate systems"):
        xrf.resample(source, target, transform=in_ras)


def test_geometry_target_keeps_nongeometry_coordinate_attrs() -> None:
    source = volume(
        np.stack([ramp(), ramp()]),
        extra={
            "echo": ("echo", [0, 1], {"description": "echo labels"}),
            "delay": ("echo", [0.25, 0.75], {"units": "s"}),
            "context": ((), 7, {"description": "scan"}),
        },
    )
    target = volume(np.zeros((2, 3, 4)))
    result = xrf.resample(source, target)
    for name in ("echo", "delay", "context"):
        xr.testing.assert_identical(result.coords[name], source.array.coords[name])


def test_resampling_target_context_does_not_replace_source_context() -> None:
    source = volume(ramp(), extra={"context": ((), 7, {"description": "source context"})})
    target = volume(np.zeros((2, 3, 4)), extra={"context": ((), 9, {})})
    for axis, unit in zip(target.transform.source.axes, target.transform.source.units, strict=True):
        target.array.coords[axis].attrs["units"] = unit
    by_geometry = xrf.resample(source, target)
    by_grid = xrf.resample(source, target.grid())
    xr.testing.assert_identical(by_geometry, by_grid)
    xr.testing.assert_identical(by_geometry.coords["context"], source.array.coords["context"])


@pytest.mark.parametrize("target_kind", ["geometry", "grid"])
def test_target_geometry_coordinate_cannot_replace_source_context(target_kind: str) -> None:
    source = volume(ramp(), extra={"context": ((), 7, {})})
    target = volume(np.zeros((2, 3, 4)))
    coordinates = xrf.ArrayCoordinates(("context", "j", "k"), target.transform.source.units)
    assert isinstance(target.transform, xrf.AffineTransform)
    target = xrf.Geometry(
        target.array.drop_vars("i").assign_coords(context=("i", target.array.coords["i"].values)),
        target.transform.with_endpoints(source=coordinates),
        dims=target.dims,
    )
    with pytest.raises(
        ValueError, match=r"target geometry coordinate 'context'.*source non-geometry"
    ):
        xrf.resample(source, target.grid() if target_kind == "grid" else target)


@pytest.mark.parametrize("target_kind", ["geometry", "grid"])
def test_resampling_keeps_nongeometry_custom_index(target_kind: str) -> None:
    source = volume(np.stack([ramp(), ramp()]))
    index = RangeIndex.arange(2, dim="echo")
    array = source.array.assign_coords(xr.Coordinates.from_xindex(index))
    array.coords["echo"].attrs["description"] = "echo labels"
    source = xrf.Geometry(array, source.transform, dims=source.dims)
    target = volume(np.zeros((2, 3, 4)))
    result = xrf.resample(source, target.grid() if target_kind == "grid" else target)
    assert isinstance(result.xindexes["echo"], RangeIndex)
    assert result.xindexes["echo"].equals(index)
    xr.testing.assert_identical(result.coords["echo"], source.array.coords["echo"])


def _nan_line(values: npt.ArrayLike) -> xrf.Geometry:
    frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x",), ("mm",)))
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("i",), ("1",), sample_offset=(0.5,)),
        target=frame,
        matrix=[[1.0]],
        translation=[0.0],
    )
    return xrf.Geometry(
        xr.DataArray(values, dims="i", coords={"i": np.arange(len(np.asarray(values)))}),
        transform,
        dims=("i",),
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("path", ["affine", "composite", "points"])
def test_linear_nan_exact_and_fractional_queries(lazy: bool, path: str) -> None:
    source = _nan_line([1.0, np.nan, 3.0, 4.0])
    if lazy:
        source = xrf.Geometry(source.array.chunk({"i": 2}), source.transform, dims=source.dims)
    assert isinstance(source.transform, xrf.AffineTransform)
    transform = (
        xrf.CompositeTransform(source.transform)
        if path == "composite"
        else PointOnly(source.transform)
        if path == "points"
        else source.transform
    )
    target = xrf.Grid(transform, {"i": ("i", [0.0, 1.0, 2.0, 3.0])})
    assert_allclose(xrf.resample(source, target), [1.0, np.nan, 3.0, 4.0], rtol=0, atol=ATOL)
    for position, expected in [
        (-0.75, -9.0),
        (-0.25, 1.0),
        (0.0, 1.0),
        (0.5, np.nan),
        (1e-8, np.nan),
        (2.0, 3.0),
        (2.5, 3.5),
    ]:
        scalar = xrf.Grid(transform, {"i": position})
        actual = xrf.resample(source, scalar, domain="cells", fill_value=-9)
        assert_allclose(actual, expected, rtol=0, atol=ATOL)


@pytest.mark.parametrize("general", [False, True])
def test_linear_nan_partial_integral_queries(general: bool) -> None:
    values = ramp()
    values[1, 2, 2] = np.nan
    source = volume(values)
    transform = xrf.CompositeTransform(source.transform) if general else source.transform
    # j is integral: the missing sample in the next row has zero weight.
    for j, expected in [(1.0, 14.0), (1.5, np.nan)]:
        target = xrf.Grid(transform, {"i": 2.5, "j": j, "k": 1.0})
        assert_allclose(xrf.resample(source, target), expected, rtol=0, atol=ATOL)
    grid = xrf.Grid(
        transform, {"i": ("i", [2.0, 2.5]), "j": ("j", [1.0, 1.5]), "k": ("k", [1.0, 2.0])}
    )
    result = xrf.resample(source, grid).transpose("k", "j", "i").values
    assert_allclose(result[0, 0], [13.0, 14.0], rtol=0, atol=ATOL)
    assert np.isnan(result[0, 1]).all()
    assert np.isfinite(result[1]).all()


@pytest.mark.parametrize("general", [False, True])
def test_linear_nan_complex_components_remain_independent(general: bool) -> None:
    values = np.array([complex(1, np.nan), complex(np.nan, 2), complex(3, 4), complex(5, 6)])
    source = _nan_line(values)
    transform = xrf.CompositeTransform(source.transform) if general else source.transform
    for position, real, imag in [
        (0.0, 1.0, np.nan),
        (1.0, np.nan, 2.0),
        (2.0, 3.0, 4.0),
        (1.5, np.nan, 3.0),
        (2.5, 4.0, 5.0),
    ]:
        result = xrf.resample(source, xrf.Grid(transform, {"i": position})).values
        assert_allclose(result.real, real, rtol=0, atol=ATOL)
        assert_allclose(result.imag, imag, rtol=0, atol=ATOL)


@pytest.mark.parametrize("origin", [(-250.0, 300.0), (-13700.0, 8456.0), (100000.0, -120000.0)])
@pytest.mark.parametrize("spacing", [(0.5, 0.7), (0.037, 0.081)])
def test_linear_nan_oblique_round_trip_at_large_origins(
    origin: tuple[float, float], spacing: tuple[float, float]
) -> None:
    angle = np.deg2rad(31.0)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ np.diag(
        spacing
    )
    frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("mm", "mm")))
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("j", "i"), ("1", "1")),
        target=frame,
        matrix=matrix,
        translation=origin,
    )
    values = np.ones((13, 17))
    values[5, 7] = np.nan
    source = xrf.Geometry(
        xr.DataArray(values, dims=("j", "i"), coords={"j": np.arange(13), "i": np.arange(17)}),
        transform,
        dims=("j", "i"),
    )
    for mapping in (transform, xrf.CompositeTransform(transform)):
        target = xrf.Grid(mapping, {"j": ("j", np.arange(13)), "i": ("i", np.arange(17))})
        result = xrf.resample(source, target, block_points=19).values
        assert_array_equal(np.isnan(result), np.isnan(values))
        # Measured public index round-trip errors reach 4.66e-10; no renormalization.
        assert_allclose(result, values, rtol=0, atol=1e-9)

    # A 1e-8 pixel shift exceeds the 1e-9 gather bound even at the largest origin.
    # Missing weight from this shift must propagate, while cross-axis roundoff must not.
    shifted = xrf.Grid(transform, {"j": ("j", np.arange(12) + 1e-8), "i": ("i", np.arange(16))})
    result = xrf.resample(source, shifted).values
    expected = values[:12, :16].copy()
    expected[4, 7] = np.nan
    assert_array_equal(np.isnan(result), np.isnan(expected))
    assert_allclose(result, expected, rtol=0, atol=1e-9)


def test_cubic_nan_prefilter_spreads_missing_values_except_on_gather() -> None:
    source = _nan_line([1.0, np.nan, 3.0, 4.0])
    assert_allclose(xrf.resample(source, source, method="cubic"), source.array, rtol=0, atol=ATOL)
    target = xrf.Grid(xrf.CompositeTransform(source.transform), {"i": ("i", [0.0, 1.0, 2.0, 3.0])})
    assert np.isnan(xrf.resample(source, target, method="cubic")).all()


@pytest.mark.parametrize("general", [False, True])
@pytest.mark.parametrize("offset", [1e-8, 5e-10])
def test_linear_missing_weight_uses_measured_rounding_allowance(
    general: bool, offset: float
) -> None:
    source = _nan_line([1.0, np.nan, 3.0, 4.0])
    transform = xrf.CompositeTransform(source.transform) if general else source.transform
    target = xrf.Grid(transform, {"i": ("i", np.arange(3) + offset)})
    result = xrf.resample(source, target).values
    assert_array_equal(np.isnan(result), [offset > 1e-9, True, False])
    if offset <= 1e-9:
        assert_allclose(result[0], 1.0, rtol=0, atol=1e-9)
    # A scalar avoids the gather path and demonstrates the unnormalized finite deficit.
    scalar = xrf.resample(source, xrf.Grid(transform, {"i": offset})).values
    assert_allclose(scalar, np.nan if offset > 1e-9 else 1.0 - offset, rtol=0, atol=1e-15)


@pytest.mark.parametrize("cache_bytes", [0, 1 << 20])
def test_linear_nan_buffers_are_reused_per_slice_and_released(
    monkeypatch: pytest.MonkeyPatch, cache_bytes: int
) -> None:
    import weakref

    monkeypatch.setattr("xarrayrf._resample.POSITION_CACHE_BYTES", cache_bytes)
    values = np.stack([ramp(), ramp() + 20, ramp() + 40])
    values[:, 1, 2, 2] = np.nan
    source = volume(values, extra={"context": 7, "echo": [0, 1, 2]})
    assert isinstance(source.transform, xrf.AffineTransform)
    target = xrf.Geometry(source.array, PointOnly(source.transform), dims=source.dims)
    references: list[weakref.ReferenceType[npt.NDArray[np.generic]]] = []
    calls = 0
    original = ndimage.map_coordinates

    def recording(data: npt.NDArray[np.generic], *args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if not any(ref() is data for ref in references):
            references.append(weakref.ref(data))
        # The public resample call may hold only one slice's value and mask buffers.
        assert sum(ref() is not None for ref in references) <= 2
        return original(data, *args, **kwargs)

    monkeypatch.setattr(ndimage, "map_coordinates", recording)
    result = xrf.resample(source, target, block_points=11)
    assert_allclose(result, source.array, rtol=0, atol=ATOL)
    assert len(references) == 6  # one value buffer and one mask per source slice
    assert calls > len(references)  # reused across blocks
    assert all(ref() is None for ref in references)
    xr.testing.assert_identical(result.context, source.array.context)
    xr.testing.assert_identical(result.echo, source.array.echo)


def test_linear_nan_lazy_crop_reads_only_nearby_chunks() -> None:
    class MissingArray(_RecordingArray):
        def __getitem__(self, key: tuple[slice, ...]) -> npt.NDArray[np.float64]:
            self.reads.append(key)
            values = np.ones(self.shape)
            values[2, 4, 2] = np.nan
            return values[key]

    recording = MissingArray()
    lazy = da.from_array(recording, chunks=(8, 16, 16))  # type: ignore[no-untyped-call]
    base = volume(np.zeros(recording.shape))
    source = xrf.Geometry(base.array.copy(data=lazy), base.transform, dims=base.dims)
    target = xrf.Grid(source.transform, {"i": 2.5, "j": 3.0, "k": 2.0})
    recording.reads.clear()
    result = xrf.resample(source, target)
    assert not recording.reads
    assert_allclose(result.compute(), 1.0, rtol=0, atol=ATOL)
    assert len(recording.reads) == 1


@pytest.mark.parametrize("general", [False, True])
def test_linear_infinities_retain_scipy_behavior(general: bool) -> None:
    values = np.array([1.0, np.inf, 3.0, 4.0])
    source = _nan_line(values)
    transform = xrf.CompositeTransform(source.transform) if general else source.transform
    positions = np.array([0.0, 0.5, 2.0, 2.5])
    target = xrf.Grid(transform, {"i": ("i", positions)})
    expected = ndimage.map_coordinates(
        values, positions[None, :], order=1, mode="nearest", prefilter=False
    )
    assert_allclose(xrf.resample(source, target), expected, rtol=0, atol=ATOL)


@pytest.mark.parametrize("general", [False, True])
def test_linear_nan_cells_hold_missing_edges_and_outside_fill(general: bool) -> None:
    source = _nan_line([np.nan, 1.0, 2.0, np.nan])
    transform = xrf.CompositeTransform(source.transform) if general else source.transform
    target = xrf.Grid(transform, {"i": ("i", [-0.75, -0.25, 3.25, 3.75])})
    result = xrf.resample(source, target, domain="cells", fill_value=-9)
    assert_allclose(result, [-9.0, np.nan, np.nan, -9.0], rtol=0, atol=ATOL)
