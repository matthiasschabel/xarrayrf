"""Resampling declared tensor components (value geometry stage 3)."""

from __future__ import annotations

from typing import Any, cast

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

import xarrayrf as xrf
import xarrayrf.native  # register the DataArray accessor

pytest.importorskip("scipy")

RTOL = ATOL = 1e-10
"""Interpolating a constant field and a few 3x3 products on O(1) numbers."""

COMPASS = xrf.DirectionVocabulary("test-compass", ("west-to-east", "south-to-north", "down-to-up"))
SYSTEM = xrf.CoordinateSystem(
    ("x", "y", "z"),
    ("mm", "mm", "mm"),
    vocabulary=COMPASS,
    orientation=("west-to-east", "south-to-north", "down-to-up"),
)
MOVING = xrf.ReferenceFrame.declared(("test", "moving"), SYSTEM)
FIXED = xrf.ReferenceFrame.declared(("test", "fixed"), SYSTEM)
FIXED_FLIPPED = FIXED.with_coordinate_system(
    xrf.CoordinateSystem(
        ("w", "y", "z"),
        ("mm", "mm", "mm"),
        vocabulary=COMPASS,
        orientation=("east-to-west", "south-to-north", "down-to-up"),
    )
)
FIXED_IN_CM = FIXED.with_coordinate_system(
    xrf.CoordinateSystem(
        ("x", "y", "z"),
        ("cm", "cm", "cm"),
        vocabulary=COMPASS,
        orientation=("west-to-east", "south-to-north", "down-to-up"),
    )
)
SOURCE_AXES = xrf.ArrayCoordinates(("k", "j", "i"), ("1", "1", "1"))
TARGET_AXES = xrf.ArrayCoordinates(("k", "j", "i"), ("1", "1", "1"))

ROTATION = np.array(
    [[np.cos(0.3), -np.sin(0.3), 0.0], [np.sin(0.3), np.cos(0.3), 0.0], [0.0, 0.0, 1.0]]
)
SHEAR = np.array([[1.0, 0.4, 0.0], [0.0, 1.2, 0.0], [0.1, 0.0, 0.8]])
VECTOR_VALUE = np.array([1.0, -2.0, 0.5])


def assert_close(actual: Any, desired: Any) -> None:
    np.testing.assert_allclose(actual, desired, rtol=RTOL, atol=ATOL)


def grid_array(
    values: npt.ArrayLike,
    frame: xrf.ReferenceFrame,
    *,
    spacing: npt.ArrayLike = (1.0, 1.0, 1.0),
    extra: tuple[str, ...] = (),
) -> xr.DataArray:
    """A framed (k, j, i, *extra) array centred on the frame origin."""
    data = np.asarray(values)
    shape = data.shape[:3]
    steps = np.asarray(spacing, dtype=np.float64)
    binding = xrf.AffineTransform.from_matrix(
        source=SOURCE_AXES,
        target=frame,
        matrix=np.diag(steps),
        translation=-steps * (np.array(shape) - 1) / 2,
    )
    array = xr.DataArray(
        data,
        dims=("k", "j", "i", *extra),
        coords={
            dim: (dim, np.arange(n), {"units": "1"}) for dim, n in zip("kji", shape, strict=True)
        },
    )
    return cast(xr.DataArray, array.rf.frame(binding, dims=("k", "j", "i")))


def constant_field(
    value: npt.ArrayLike,
    tensor_type: xrf.TensorType = xrf.VECTOR,
    *,
    bases: Any = MOVING,
    packing: xrf.SymmetricPacking | None = None,
    spacing: npt.ArrayLike = (1.0, 1.0, 1.0),
) -> xr.DataArray:
    """A 7^3 source holding one tensor everywhere, so interpolation reproduces it exactly."""
    components = np.asarray(value, dtype=np.float64)
    names = ("component",) if components.ndim == 1 else ("row", "column")
    data = np.broadcast_to(components, (7, 7, 7, *components.shape)).copy()
    array = grid_array(data, MOVING, spacing=spacing, extra=names)
    if isinstance(bases, str):
        bases = array.rf.coordinate_transform.source
    return cast(
        xr.DataArray,
        array.rf.declare_tensor_components(
            tensor_type, component_dims=names, bases=bases, packing=packing
        ),
    )


TARGET = grid_array(np.zeros((3, 3, 3)), FIXED, spacing=(0.5, 0.5, 0.5))
"""Target samples within one unit of the origin: inside the source under every map used."""


def registration(matrix: npt.ArrayLike) -> xrf.AffineTransform:
    """A transform from the fixed (target) frame to the moving (source) frame."""
    return xrf.AffineTransform.from_matrix(
        source=FIXED, target=MOVING, matrix=matrix, translation=(0.0, 0.0, 0.0)
    )


def resample(field: xr.DataArray, matrix: npt.ArrayLike, **options: Any) -> xr.DataArray:
    options.setdefault("output_component_basis", FIXED)
    options.setdefault("component_reorientation", "jacobian")
    return cast(
        xr.DataArray, field.rf.resample_to(TARGET, transform=registration(matrix), **options)
    )


def geometry_first(result: xr.DataArray) -> npt.NDArray[Any]:
    """Values as (k, j, i, *components); resampling puts non-geometry dimensions first."""
    return result.transpose("k", "j", "i", ...).values


def at_every_target(result: xr.DataArray, expected: npt.ArrayLike) -> None:
    """The result holds ``expected`` at every target sample."""
    values = geometry_first(result)
    assert_close(values, np.broadcast_to(np.asarray(expected), values.shape))


def test_a_vector_follows_the_inverse_of_the_registration() -> None:
    result = resample(constant_field(VECTOR_VALUE), ROTATION)
    at_every_target(result, ROTATION.T @ VECTOR_VALUE)
    declaration = result.rf.tensor_component_declaration
    assert declaration.bases == (FIXED,) and declaration.tensor_type == xrf.VECTOR
    assert result.component.values.tolist() == ["x", "y", "z"]
    assert result.rf.reference_frame == FIXED


def test_a_rotation_gives_the_same_vector_under_both_reorientations() -> None:
    field = constant_field(VECTOR_VALUE)
    jacobian = resample(field, 2.0 * ROTATION)
    polar = resample(field, 2.0 * ROTATION, component_reorientation="polar_orthogonal_factor")
    at_every_target(jacobian, ROTATION.T @ VECTOR_VALUE / 2.0)
    at_every_target(polar, ROTATION.T @ VECTOR_VALUE)


def test_a_shear_separates_the_jacobian_from_its_polar_factor() -> None:
    field = constant_field(VECTOR_VALUE)
    push = np.linalg.inv(SHEAR)
    u, _, vt = np.linalg.svd(push)
    at_every_target(resample(field, SHEAR), push @ VECTOR_VALUE)
    at_every_target(
        resample(field, SHEAR, component_reorientation="polar_orthogonal_factor"),
        u @ vt @ VECTOR_VALUE,
    )


def test_a_covector_follows_the_transpose_of_the_registration() -> None:
    result = resample(constant_field(VECTOR_VALUE, xrf.COVECTOR), SHEAR)
    at_every_target(result, SHEAR.T @ VECTOR_VALUE)


def test_a_pseudovector_gains_the_sign_of_a_reflection() -> None:
    mirror = np.diag([-1.0, 1.0, 1.0])
    result = resample(constant_field(VECTOR_VALUE, xrf.ORTHOGONAL_PSEUDOVECTOR), mirror)
    at_every_target(result, -(mirror @ VECTOR_VALUE))


def test_a_density_scales_with_the_volume_change() -> None:
    flux_density = xrf.TensorType(("contravariant",), determinant_exponent=-1.0)
    result = resample(constant_field(VECTOR_VALUE, flux_density), SHEAR)
    push = np.linalg.inv(SHEAR)
    at_every_target(result, push @ VECTOR_VALUE / abs(np.linalg.det(push)))


def test_a_packed_diffusion_tensor_is_rotated_by_the_polar_factor() -> None:
    """The design's acceptance case: rotation plus scale, reoriented as a rotation."""
    diffusion = np.diag([3.0, 1.0, 0.5])
    diffusion[0, 1] = diffusion[1, 0] = 0.2
    packing = xrf.SymmetricPacking.mandel(3)
    field = constant_field(
        packing.pack(diffusion), xrf.SYMMETRIC_SECOND_ORDER_CONTRAVARIANT_TENSOR, packing=packing
    )
    result = resample(field, 1.5 * ROTATION, component_reorientation="polar_orthogonal_factor")
    expected = ROTATION.T @ diffusion @ ROTATION
    at_every_target(result, packing.pack(expected))
    assert result.rf.tensor_component_declaration.packing == packing
    assert result.component.values.tolist() == ["xx", "yy", "zz", "yz", "xz", "xy"]


def test_an_array_coordinate_basis_carries_the_binding_factor() -> None:
    """The design's acceptance case: anisotropic voxels under identity registration."""
    spacing = np.array([2.0, 1.0, 0.5])
    field = constant_field(VECTOR_VALUE, bases="array", spacing=spacing)
    result = resample(field, np.eye(3), units="mm/s")
    at_every_target(result, spacing * VECTOR_VALUE)


def test_slots_in_different_bases_each_get_their_own_input_map() -> None:
    spacing = np.array([2.0, 1.0, 0.5])
    tensor = np.arange(9.0).reshape(3, 3) - 4.0
    field = constant_field(
        tensor, xrf.SECOND_ORDER_CONTRAVARIANT_TENSOR, spacing=spacing, bases="array"
    )
    field = field.rf.change_component_basis({"column": MOVING}, units="mm")
    result = resample(field, ROTATION, units="mm2")
    # The column slot already took the spacing when it changed to the frame; the row slot
    # takes it now, through its input map.
    expected = ROTATION.T @ np.diag(spacing) @ tensor @ np.diag(spacing) @ ROTATION
    at_every_target(result, expected)
    assert result.rf.tensor_component_declaration.bases == (FIXED, FIXED)


def test_components_can_be_returned_in_the_target_array_coordinates() -> None:
    spacing = np.array([2.0, 1.0, 0.5])
    target = grid_array(np.zeros((3, 3, 3)), FIXED, spacing=spacing)
    result = constant_field(VECTOR_VALUE).rf.resample_to(
        target,
        transform=registration(np.eye(3)),
        output_component_basis=target.rf.coordinate_transform.source,
        component_reorientation="jacobian",
        units="1",
    )
    at_every_target(result, VECTOR_VALUE / spacing)
    assert result.rf.tensor_component_declaration.bases == (TARGET_AXES,)
    assert result.attrs["units"] == "1"


def test_an_equivalent_output_frame_uses_the_coordinate_system_change() -> None:
    result = resample(constant_field(VECTOR_VALUE), np.eye(3), output_component_basis=FIXED_FLIPPED)
    at_every_target(result, VECTOR_VALUE * np.array([-1.0, 1.0, 1.0]))
    assert result.component.values.tolist() == ["w", "y", "z"]


def test_plain_resample_declares_the_result_in_a_frame() -> None:
    field = constant_field(VECTOR_VALUE)
    result = xrf.resample(
        field.rf.geometry,
        TARGET.rf.grid,
        transform=registration(ROTATION),
        output_component_basis=FIXED,
        component_reorientation="jacobian",
    )
    at_every_target(result, ROTATION.T @ VECTOR_VALUE)
    assert result.rf.tensor_component_declaration.bases == (FIXED,)


class Bend:
    """A nonlinear fixed-to-moving transform, (x, y, z) -> (x, y + 0.2 x^2, z), with Jacobian."""

    source = FIXED
    target = MOVING

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        bent: npt.NDArray[np.float64] = values.copy()
        bent[..., 1] += 0.2 * values[..., 0] ** 2
        return bent

    def jacobian(self, at: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        assert at is not None
        values = np.asarray(at, dtype=np.float64)
        result = np.broadcast_to(np.eye(3), (*values.shape[:-1], 3, 3)).copy()
        result[..., 1, 0] = 0.4 * values[..., 0]
        return result


class BendWithoutJacobian(Bend):
    jacobian = None  # type: ignore[assignment]


def expected_under_bend() -> npt.NDArray[np.float64]:
    """``(D T(y))^-1 v`` at every target sample, computed independently per point."""
    x = (np.arange(3) - 1) * 0.5  # the target's k coordinate maps to frame x
    expected = np.empty((3, 3, 3, 3))
    for k in range(3):
        jacobian = np.eye(3)
        jacobian[1, 0] = 0.4 * x[k]
        expected[k] = np.linalg.solve(jacobian, VECTOR_VALUE)
    return expected


@pytest.mark.parametrize("block_points", [4, 100_000])
def test_a_nonlinear_registration_maps_each_target_point(block_points: int) -> None:
    field = constant_field(VECTOR_VALUE)
    result = field.rf.resample_to(
        TARGET,
        transform=Bend(),
        output_component_basis=FIXED,
        component_reorientation="jacobian",
    )
    assert_close(geometry_first(result), expected_under_bend())
    blocked = xrf.resample(
        field.rf.geometry,
        TARGET.rf.grid,
        transform=Bend(),
        output_component_basis=FIXED,
        component_reorientation="jacobian",
        block_points=block_points,
    )
    assert_close(geometry_first(blocked), expected_under_bend())


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "dask"])
def test_a_nonlinear_registration_reaches_a_single_point_target(lazy: bool) -> None:
    field = constant_field(VECTOR_VALUE)
    source = field.chunk({"component": 1, "i": 3}) if lazy else field
    options: dict[str, Any] = {
        "transform": Bend(),
        "output_component_basis": FIXED,
        "component_reorientation": "jacobian",
    }
    point = source.rf.resample_to(TARGET.rf.grid.isel(k=1, j=1, i=1), **options).compute()
    singleton = source.rf.resample_to(TARGET.rf.grid.isel(k=[1], j=[1], i=[1]), **options)
    assert point.dims == ("component",)
    assert_close(point.values, VECTOR_VALUE)  # x = 0, where the bend has unit Jacobian
    assert_close(point.values, singleton.values.reshape(3))


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "dask"])
def test_mixed_variance_slots_in_different_bases_map_per_point(lazy: bool) -> None:
    """A contravariant slot in voxel components and a covariant one in the frame, bent."""
    spacing = np.array([2.0, 1.0, 0.5])
    mixed = xrf.TensorType(("contravariant", "covariant"))
    tensor = np.arange(9.0).reshape(3, 3) - 4.0
    data = np.broadcast_to(tensor, (7, 7, 7, 2, 3, 3)).copy()
    array = grid_array(data, MOVING, spacing=spacing, extra=("time", "row", "column"))
    field = array.rf.declare_tensor_components(
        mixed,
        component_dims=("row", "column"),
        bases=(array.rf.coordinate_transform.source, MOVING),
    )
    source = field.chunk({"time": 1, "row": 1}) if lazy else field
    result = source.rf.resample_to(
        TARGET,
        transform=Bend(),
        output_component_basis=FIXED,
        component_reorientation="jacobian",
        units="mm",
    )
    if lazy:
        assert isinstance(result.data, da.Array)
    values = result.transpose("time", "k", "j", "i", "row", "column").values
    x = (np.arange(3) - 1) * 0.5
    for k in range(3):
        jacobian = np.eye(3)
        jacobian[1, 0] = 0.4 * x[k]
        push = np.linalg.inv(jacobian)
        expected = push @ np.diag(spacing) @ tensor @ np.linalg.inv(push)
        assert_close(values[:, k], np.broadcast_to(expected, values[:, k].shape))


class Fold(Bend):
    """A registration that folds space at x = 0.5: its Jacobian is singular there."""

    def jacobian(self, at: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        result = super().jacobian(at)
        values = np.asarray(at, dtype=np.float64)
        result[..., 0, 0] = np.where(np.isclose(values[..., 0], 0.5), 0.0, 1.0)
        return result


def test_a_singular_jacobian_names_the_target_sample() -> None:
    with pytest.raises(ValueError, match=r"target sample \{'k': 2, 'j': 0, 'i': 0\}"):
        xrf.resample(
            constant_field(VECTOR_VALUE).rf.geometry,
            TARGET.rf.grid,
            transform=Fold(),
            output_component_basis=FIXED,
            component_reorientation="jacobian",
            block_points=5,
        )


class Undefined(Bend):
    """A map whose Jacobian raises at x = 0.5, as a chart transition does on the polar axis."""

    def jacobian(self, at: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        values = np.asarray(at, dtype=np.float64)
        if np.isclose(values[..., 0], 0.5).any():
            raise ValueError("no Jacobian on the axis")
        return super().jacobian(at)


def test_a_raised_jacobian_names_the_target_sample() -> None:
    with pytest.raises(ValueError, match=r"undefined at target sample \{'k': 2, 'j': 0, 'i': 0\}"):
        xrf.resample(
            constant_field(VECTOR_VALUE).rf.geometry,
            TARGET.rf.grid,
            transform=Undefined(),
            output_component_basis=FIXED,
            component_reorientation="jacobian",
            block_points=5,
        )


def test_box_methods_average_components_and_keep_the_declaration() -> None:
    from test_box_resample import slab_grid

    source_grid = slab_grid(np.arange(6)[:, None] + [0, 1])
    frame = source_grid.transform.target
    plain = xr.DataArray(
        np.arange(6.0)[:, None],
        dims=(*source_grid.dims, "component"),
        coords={"component": list(frame.axes)},
    )
    source = cast(xr.DataArray, plain.rf.frame(source_grid))
    declared = source.rf.declare_tensor_components(
        xrf.VECTOR, component_dims=("component",), bases=frame
    )
    result, coverage = declared.rf.resample_to(
        slab_grid([[0, 2], [2, 6]]),
        method="overlap_mean",
        support="average",
        return_coverage=True,
        output_component_basis=frame,
        component_reorientation="jacobian",
    )
    assert_close(result.transpose(..., "component").values[:, 0], [0.5, 3.5])
    assert result.rf.tensor_component_declaration.bases == (frame,)
    assert_close(coverage.values, [1.0, 1.0])
    assert not coverage.rf.has_tensor_components


def test_a_callable_reorientation_result_is_applied() -> None:
    result = resample(constant_field(VECTOR_VALUE), SHEAR, component_reorientation=lambda m: 2 * m)
    at_every_target(result, 2 * np.linalg.inv(SHEAR) @ VECTOR_VALUE)


def test_a_callable_reorientation_receives_batches_of_maps() -> None:
    shapes: list[tuple[int, ...]] = []

    def record(maps: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        shapes.append(maps.shape)
        return maps

    field = constant_field(VECTOR_VALUE)
    resample(field, SHEAR, component_reorientation=record)
    assert shapes == [(3, 3)]
    shapes.clear()
    xrf.resample(
        field.rf.geometry,
        TARGET.rf.grid,
        transform=Bend(),
        output_component_basis=FIXED,
        component_reorientation=record,
        block_points=10,
    )
    assert shapes == [(10, 3, 3), (10, 3, 3), (7, 3, 3)]


@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
@pytest.mark.parametrize("transform", ["affine", "nonlinear"])
def test_split_chunks_stay_lazy_and_match_the_eager_result(packed: bool, transform: str) -> None:
    if packed:
        packing = xrf.SymmetricPacking.voigt(3)
        diffusion = np.diag([3.0, 1.0, 0.5])
        field = constant_field(
            packing.pack(diffusion),
            xrf.SYMMETRIC_SECOND_ORDER_CONTRAVARIANT_TENSOR,
            packing=packing,
        )
    else:
        field = constant_field(VECTOR_VALUE)
    frame_map: Any = registration(SHEAR) if transform == "affine" else Bend()
    options = {
        "transform": frame_map,
        "output_component_basis": FIXED,
        "component_reorientation": "jacobian",
    }
    eager = field.rf.resample_to(TARGET, **options)
    lazy = field.chunk({"component": 1, "i": 3}).rf.resample_to(TARGET, **options)
    assert isinstance(lazy.data, da.Array)
    computed = lazy.compute()
    assert_close(computed.values, eager.values)
    assert computed.rf.tensor_component_declaration == eager.rf.tensor_component_declaration


def test_an_empty_target_is_declared_without_mapping() -> None:
    empty = grid_array(np.zeros((0, 3, 3)), FIXED)
    for field in (constant_field(VECTOR_VALUE), constant_field(VECTOR_VALUE).chunk({"i": 2})):
        result = field.rf.resample_to(
            empty,
            transform=registration(ROTATION),
            output_component_basis=FIXED,
            component_reorientation="jacobian",
        )
        assert result.sizes == {"component": 3, "k": 0, "j": 3, "i": 3}
        assert result.rf.tensor_component_declaration.bases == (FIXED,)


@pytest.mark.parametrize("fill_value", [0.0, np.nan])
def test_nan_and_zero_fill_stay_fill(fill_value: float) -> None:
    far = grid_array(np.zeros((2, 1, 1)), FIXED, spacing=(100.0, 1.0, 1.0))
    result = constant_field(VECTOR_VALUE).rf.resample_to(
        far,
        transform=registration(SHEAR),
        output_component_basis=FIXED,
        component_reorientation="jacobian",
        fill_value=fill_value,
    )
    np.testing.assert_array_equal(result.values, np.full((3, 2, 1, 1), fill_value))


def test_units_are_required_and_replace_the_source_units_on_the_accessor() -> None:
    field = constant_field(VECTOR_VALUE).assign_attrs(units="mm/s", long_name="velocity")
    with pytest.raises(ValueError, match="pass units="):
        resample(field, np.eye(3), output_component_basis=TARGET_AXES)
    result = resample(field, np.eye(3), output_component_basis=TARGET_AXES, units="1/s")
    assert result.attrs == {"units": "1/s", "long_name": "velocity"}


def test_undeclared_sources_are_unchanged() -> None:
    plain = constant_field(VECTOR_VALUE).rf.clear_tensor_components()
    result = plain.rf.resample_to(TARGET, transform=registration(ROTATION))
    at_every_target(result, VECTOR_VALUE)
    assert not result.rf.has_tensor_components


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"output_component_basis": None}, ValueError, "needs output_component_basis"),
        ({"component_reorientation": None}, ValueError, "needs component_reorientation"),
        ({"component_reorientation": "rotation"}, ValueError, "must be one of"),
        ({"component_reorientation": 3}, TypeError, "or a callable"),
        (
            {"output_component_basis": "fixed"},
            TypeError,
            "ReferenceFrame, ArrayCoordinates or FrameField",
        ),
        ({"output_component_basis": MOVING}, ValueError, "output_component_basis must be"),
        ({"fill_value": -1.0}, ValueError, "valid-looking tensor"),
        ({"units": "m/s"}, ValueError, "only accepted when the basis units change"),
        (
            {"component_reorientation": lambda maps: maps[..., :2]},
            ValueError,
            "one map per map",
        ),
        (
            {"component_reorientation": lambda maps: np.full_like(maps, np.inf)},
            ValueError,
            "real, finite maps",
        ),
    ],
)
def test_invalid_component_arguments_are_refused(
    options: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        resample(constant_field(VECTOR_VALUE), ROTATION, **options)


def test_component_arguments_need_a_declared_source() -> None:
    plain = constant_field(VECTOR_VALUE).rf.clear_tensor_components()
    with pytest.raises(ValueError, match="apply only to a source that declares"):
        plain.rf.resample_to(
            TARGET, transform=registration(ROTATION), component_reorientation="jacobian"
        )


def test_the_polar_factor_needs_equal_units_and_no_density() -> None:
    millimetre_target = grid_array(np.zeros((3, 3, 3)), FIXED_IN_CM)
    to_moving = xrf.AffineTransform.from_matrix(
        source=FIXED_IN_CM, target=MOVING, matrix=10.0 * np.eye(3), translation=(0.0, 0.0, 0.0)
    )
    with pytest.raises(ValueError, match="one unit on every axis"):
        constant_field(VECTOR_VALUE).rf.resample_to(
            millimetre_target,
            transform=to_moving,
            output_component_basis=FIXED_IN_CM,
            component_reorientation="polar_orthogonal_factor",
            units="cm/s",
        )
    density = xrf.TensorType(("contravariant",), determinant_exponent=-1.0)
    with pytest.raises(ValueError, match="drops the volume change"):
        resample(
            constant_field(VECTOR_VALUE, density),
            ROTATION,
            component_reorientation="polar_orthogonal_factor",
        )


def test_a_source_basis_unrelated_to_its_frame_is_refused() -> None:
    other = xrf.ReferenceFrame.declared(("test", "other"), SYSTEM)
    with pytest.raises(ValueError, match="change their basis first"):
        resample(constant_field(VECTOR_VALUE, bases=other), ROTATION)


def test_plain_resample_refuses_an_array_coordinates_output() -> None:
    with pytest.raises(ValueError, match=r"use rf\.resample_to"):
        xrf.resample(
            constant_field(VECTOR_VALUE).rf.geometry,
            TARGET.rf.grid,
            transform=registration(ROTATION),
            output_component_basis=TARGET_AXES,
            component_reorientation="jacobian",
        )


def test_a_frame_map_without_a_jacobian_is_refused() -> None:
    with pytest.raises(ValueError, match="needs the frame map's Jacobian"):
        constant_field(VECTOR_VALUE).rf.resample_to(
            TARGET,
            transform=BendWithoutJacobian(),
            output_component_basis=FIXED,
            component_reorientation="jacobian",
        )


def test_an_orthogonal_only_law_refuses_a_shear() -> None:
    with pytest.raises(ValueError, match="orthogonal"):
        resample(constant_field(VECTOR_VALUE, xrf.ORTHOGONAL_PSEUDOVECTOR), SHEAR)
