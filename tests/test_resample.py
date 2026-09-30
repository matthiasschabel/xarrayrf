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
    transform = xrf.AffineTransform(
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
    affine = xrf.AffineTransform(
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
        xrf.AffineTransform(
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
    registration = xrf.AffineTransform(
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
        xrf.AffineTransform(
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
        xrf.AffineTransform(
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
    affine = xrf.AffineTransform(
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
        else xrf.AffineTransform(
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
        xrf.AffineTransform(
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
        xrf.AffineTransform(source=IJK, target=LPS, matrix=np.eye(3), translation=(0.0, 0.0, 0.0)),
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
        xrf.AffineTransform(
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
        xrf.AffineTransform(source=coordinates, target=LPS, matrix=rotation, translation=(0, 0, 0)),
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


def test_a_transform_applies_only_to_the_frames_it_was_computed_between() -> None:
    other = xrf.ReferenceFrame.declared(("test", "other patient"), LPS.coordinate_system)
    source = volume(ramp(), frame=other)
    target = volume(np.zeros((2, 2, 2)))
    registration = xrf.AffineTransform(
        source=LPS, target=other, matrix=np.eye(3), translation=np.zeros(3)
    )
    xrf.resample(source, target, transform=registration)  # computed between these frames
    third = xrf.ReferenceFrame.declared(("test", "third patient"), LPS.coordinate_system)
    with pytest.raises(
        ValueError, match=r"maps test:patient to test:other patient.*computed between"
    ):
        xrf.resample(volume(ramp(), frame=third), target, transform=registration)
    in_ras = xrf.AffineTransform(
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
