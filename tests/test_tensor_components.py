"""Tensor component declarations on xarray variables (value geometry stage 2)."""

from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

import xarrayrf as xrf
import xarrayrf.native  # register the DataArray accessor

RTOL = ATOL = 1e-12
"""A few float64 operations on O(1) numbers."""

COMPASS = xrf.DirectionVocabulary("test-compass", ("west-to-east", "south-to-north", "down-to-up"))


def site(*orientation: str, axes: tuple[str, ...] = ("a", "b", "c")) -> xrf.ReferenceFrame:
    """One site frame expressed in a coordinate system with the given axis orientation."""
    system = xrf.CoordinateSystem(
        axes, ("m", "m", "m"), vocabulary=COMPASS, orientation=orientation
    )
    return xrf.ReferenceFrame.declared(("test", "site"), system)


ENU = site("west-to-east", "south-to-north", "down-to-up", axes=("e", "n", "u"))
WSU = site("east-to-west", "north-to-south", "down-to-up", axes=("w", "s", "u"))
WNU = site("east-to-west", "south-to-north", "down-to-up", axes=("w", "n", "u"))

PLANE = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("mm", "mm")))
OTHER_PLANE = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("p", "q"), ("mm", "mm")))
SPACING = np.array([[2.0, 0.0], [0.0, 3.0]])
"""The binding's matrix: anisotropic spacing, so array and frame components differ."""


def assert_close(actual: Any, desired: Any) -> None:
    np.testing.assert_allclose(actual, desired, rtol=RTOL, atol=ATOL)


def framed_field(extra: dict[str, int], values: npt.ArrayLike | None = None) -> xr.DataArray:
    """A field on a framed (j, i) grid with additional non-geometry dimensions."""
    source = xrf.ArrayCoordinates(("j", "i"), ("1", "1"))
    binding = xrf.AffineTransform.from_matrix(
        source=source, target=PLANE, matrix=SPACING, translation=(0.0, 0.0)
    )
    shape = (2, 3, *extra.values())
    data = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) if values is None else values
    raw = xr.DataArray(
        data,
        dims=("j", "i", *extra),
        coords={"j": ("j", [0, 1], {"units": "1"}), "i": ("i", [0, 1, 2], {"units": "1"})},
    )
    return cast(xr.DataArray, raw.rf.frame(binding, dims=("j", "i")))


@pytest.fixture
def source() -> xrf.ArrayCoordinates:
    return xrf.ArrayCoordinates(("j", "i"), ("1", "1"))


@pytest.fixture
def wind() -> xr.DataArray:
    """A vector per time step in the ENU frame, unframed."""
    raw = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        dims=("time", "component"),
        attrs={"units": "m/s"},
    )
    return cast(
        xr.DataArray,
        raw.rf.declare_tensor_components(xrf.VECTOR, component_dims=("component",), bases=ENU),
    )


@pytest.fixture
def velocity(source: xrf.ArrayCoordinates) -> xr.DataArray:
    """A vector field in the array's own coordinates, on a framed grid."""
    field = framed_field({"component": 2})
    return cast(
        xr.DataArray,
        field.rf.declare_tensor_components(xrf.VECTOR, component_dims=("component",), bases=source),
    )


# Declaration


def test_declaring_creates_labels_and_shares_data(wind: xr.DataArray) -> None:
    assert wind.rf.has_tensor_components
    assert wind.component.values.tolist() == ["e", "n", "u"]
    declaration = wind.rf.tensor_component_declaration
    assert declaration == xrf.TensorComponentDeclaration(xrf.VECTOR, ("component",), ENU)
    raw = xr.DataArray(np.ones((2, 3)), dims=("time", "component"))
    declared = raw.rf.declare_tensor_components(
        xrf.VECTOR, component_dims=("component",), bases=ENU
    )
    assert np.shares_memory(declared.values, raw.values)


def test_existing_labels_are_kept_with_their_attrs() -> None:
    raw = xr.DataArray(
        np.ones(3),
        dims="component",
        coords={"component": ("component", ["e", "n", "u"], {"long_name": "axis"})},
    )
    declared = raw.rf.declare_tensor_components(
        xrf.COVECTOR, component_dims=("component",), bases=ENU
    )
    assert declared.component.attrs == {"long_name": "axis"}


def test_an_order_two_declaration_spans_two_dimensions(velocity: xr.DataArray) -> None:
    raw = xr.DataArray(np.ones((4, 3, 3)), dims=("time", "row", "column"))
    stress = raw.rf.declare_tensor_components(
        xrf.SECOND_ORDER_CONTRAVARIANT_TENSOR, component_dims=("row", "column"), bases=ENU
    )
    assert stress.rf.tensor_component_declaration.component_dims == ("row", "column")
    assert stress.row.values.tolist() == stress.column.values.tolist() == ["e", "n", "u"]


def test_declaring_keeps_dask_lazy() -> None:
    raw = xr.DataArray(da.ones((4, 3), chunks=(2, 3)), dims=("time", "component"))
    declared = raw.rf.declare_tensor_components(
        xrf.VECTOR, component_dims=("component",), bases=ENU
    )
    assert isinstance(declared.data, da.Array)
    changed = declared.rf.change_component_basis(WNU)
    assert isinstance(changed.data, da.Array)
    assert_close(changed.values, np.tile([-1.0, 1.0, 1.0], (4, 1)))


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (
            lambda a: a.rf.declare_tensor_components(xrf.SCALAR, component_dims=(), bases=ENU),
            "scalar",
        ),
        (
            lambda a: a.rf.declare_tensor_components(
                xrf.VECTOR, component_dims=("absent",), bases=ENU
            ),
            "not a dimension",
        ),
        (
            lambda a: a.rf.declare_tensor_components(
                xrf.VECTOR, component_dims=("time",), bases=ENU
            ),
            "length 2",
        ),
        (
            lambda a: a.rf.declare_tensor_components(
                xrf.VECTOR, component_dims=("component",), bases=PLANE
            ),
            "length 3",
        ),
    ],
    ids=["order-0", "absent", "wrong-length", "basis-size"],
)
def test_invalid_declarations_are_refused(make: Any, match: str) -> None:
    raw = xr.DataArray(np.ones((2, 3)), dims=("time", "component"))
    with pytest.raises(ValueError, match=match):
        make(raw)


def test_labels_must_be_the_basis_axes_in_order() -> None:
    raw = xr.DataArray(np.ones(3), dims="component", coords={"component": ["n", "e", "u"]})
    with pytest.raises(ValueError, match=r"must be the basis axes \('e', 'n', 'u'\)"):
        raw.rf.declare_tensor_components(xrf.VECTOR, component_dims=("component",), bases=ENU)
    numeric = xr.DataArray(np.ones(3), dims="component", coords={"component": [0, 1, 2]})
    with pytest.raises(ValueError, match="must be the basis axes"):
        numeric.rf.declare_tensor_components(xrf.VECTOR, component_dims=("component",), bases=ENU)


def test_declaring_twice_or_over_an_encoded_declaration_is_refused(wind: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="already declares"):
        wind.rf.declare_tensor_components(xrf.VECTOR, component_dims=("component",), bases=ENU)
    encoded = wind.rf.encode()
    with pytest.raises(ValueError, match=r"encoded declaration.*rf\.decode\(\)"):
        encoded.rf.declare_tensor_components(xrf.VECTOR, component_dims=("component",), bases=ENU)


def test_a_component_dimension_cannot_be_a_geometry_dimension() -> None:
    field = framed_field({})
    with pytest.raises(ValueError, match="geometry dimension"):
        field.rf.declare_tensor_components(
            xrf.VECTOR,
            component_dims=("j",),
            bases=xrf.ReferenceFrame.local(xrf.CoordinateSystem(("p", "q"), ("m", "m"))),
        )


def test_an_array_coordinates_basis_must_be_the_binding_source(
    source: xrf.ArrayCoordinates,
) -> None:
    raw = xr.DataArray(np.ones(2), dims="component")
    with pytest.raises(ValueError, match="source of this array's frame binding"):
        raw.rf.declare_tensor_components(xrf.VECTOR, component_dims=("component",), bases=source)
    field = framed_field({"component": 2})
    with pytest.raises(ValueError, match="source of this array's frame binding"):
        field.rf.declare_tensor_components(
            xrf.VECTOR,
            component_dims=("component",),
            bases=xrf.ArrayCoordinates(("j", "i"), ("1", "mm")),
        )


def test_set_xindex_cannot_build_a_declaration(wind: xr.DataArray) -> None:
    from xarrayrf._tensor_components import TensorComponentIndex

    raw = xr.DataArray(np.ones(3), dims="component", coords={"component": ["e", "n", "u"]})
    with pytest.raises(ValueError, match=r"rf\.declare_tensor_components"):
        raw.drop_indexes("component").set_xindex("component", TensorComponentIndex)


def test_reading_an_undeclared_array_is_refused() -> None:
    raw = xr.DataArray(np.ones(3), dims="component")
    assert not raw.rf.has_tensor_components
    with pytest.raises(ValueError, match="declares no tensor components"):
        _ = raw.rf.tensor_component_declaration
    with pytest.raises(ValueError, match="declares no tensor components"):
        raw.rf.clear_tensor_components()


def test_clearing_restores_ordinary_labels(wind: xr.DataArray) -> None:
    cleared = wind.rf.clear_tensor_components()
    assert not cleared.rf.has_tensor_components
    assert cleared.sel(component="n").values.tolist() == [2.0, 5.0]


def test_the_declaration_value_object_validates_alone() -> None:
    with pytest.raises(ValueError, match="one dimension per component index"):
        xrf.TensorComponentDeclaration(xrf.VECTOR, ("a", "b"), ENU)
    with pytest.raises(ValueError, match="one basis per component index"):
        xrf.TensorComponentDeclaration(xrf.VECTOR, ("a",), (ENU, ENU))
    with pytest.raises(TypeError, match="each basis must be"):
        xrf.TensorComponentDeclaration(xrf.VECTOR, ("a",), ("e",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bases must be"):
        xrf.TensorComponentDeclaration(xrf.VECTOR, ("a",), 3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tensor_type must be"):
        xrf.TensorComponentDeclaration("vector", ("a",), ENU)  # type: ignore[arg-type]
    symmetric = xrf.TensorType(("contravariant", "contravariant"), index_symmetry="symmetric")
    with pytest.raises(ValueError, match="both indices in one basis"):
        xrf.TensorComponentDeclaration(symmetric, ("r", "c"), (ENU, WNU))
    declaration = xrf.TensorComponentDeclaration(symmetric, ("r", "c"), ENU)
    assert declaration.bases == (ENU, ENU) and hash(declaration) == hash(
        xrf.TensorComponentDeclaration(symmetric, ["r", "c"], [ENU, ENU])
    )


# Structural operations


@pytest.mark.parametrize(
    "select",
    [
        lambda a: a.isel(component=0),
        lambda a: a.isel(component=[0, 1]),
        lambda a: a.isel(component=[2, 1, 0]),
        lambda a: a.sel(component="n"),
    ],
    ids=["scalar", "subset", "permutation", "label"],
)
def test_selecting_part_of_the_components_drops_the_declaration(
    wind: xr.DataArray, select: Any
) -> None:
    result = select(wind)
    assert not result.rf.has_tensor_components
    assert "component" in result.coords


@pytest.mark.parametrize(
    "select",
    [
        lambda a: a.isel(time=0),
        lambda a: a.isel(component=slice(None)),
        lambda a: a.isel(component=[0, 1, 2]),
        lambda a: a.sel(component=["e", "n", "u"]),
        lambda a: a.transpose(),
    ],
    ids=["other-dim", "full-slice", "identity", "labels", "transpose"],
)
def test_keeping_every_component_in_order_keeps_the_declaration(
    wind: xr.DataArray, select: Any
) -> None:
    assert select(wind).rf.tensor_component_declaration == wind.rf.tensor_component_declaration


def test_renaming_a_component_dimension_follows(wind: xr.DataArray) -> None:
    renamed = wind.rename(component="axis")
    assert renamed.rf.tensor_component_declaration.component_dims == ("axis",)


def test_arithmetic_with_an_equal_declaration_keeps_it(wind: xr.DataArray) -> None:
    for result in (
        wind + wind,
        wind - 2 * wind,
        wind.mean("time"),
        xr.concat([wind, wind], "time"),
    ):
        assert result.rf.tensor_component_declaration == wind.rf.tensor_component_declaration


def test_different_laws_cannot_be_combined(wind: xr.DataArray) -> None:
    covector = wind.rf.clear_tensor_components().rf.declare_tensor_components(
        xrf.COVECTOR, component_dims=("component",), bases=ENU
    )
    with pytest.raises(ValueError, match="follow different laws"):
        wind + covector


def test_different_bases_cannot_be_combined(wind: xr.DataArray) -> None:
    flipped = wind.rf.change_component_basis(WNU).assign_coords(component=["e", "n", "u"])
    assert not flipped.rf.has_tensor_components  # reassigning labels is a coordinate exit
    other = wind.rf.change_component_basis(WNU)
    with pytest.raises(ValueError):
        wind + other


def test_concatenating_along_a_component_dimension_is_refused(wind: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="concatenate along a component dimension"):
        xr.concat([wind, wind], "component")


def test_reducing_over_a_component_dimension_drops_the_declaration(wind: xr.DataArray) -> None:
    assert not wind.sum("component").rf.has_tensor_components


def test_copies_and_pickles_keep_the_declaration(wind: xr.DataArray) -> None:
    for result in (
        wind.copy(),
        wind.copy(deep=True),
        pickle.loads(pickle.dumps(wind)),
        wind.astype(np.float32),
    ):
        assert result.rf.tensor_component_declaration == wind.rf.tensor_component_declaration


def test_dataset_variables_on_the_component_dimension_share_the_declaration(
    wind: xr.DataArray,
) -> None:
    gust = (2 * wind).rename("gust")
    dataset = xr.Dataset({"wind": wind, "gust": gust})
    assert dataset["gust"].rf.tensor_component_declaration == wind.rf.tensor_component_declaration
    assert dataset["wind"].rf.has_tensor_components


def test_resampling_a_declared_source_is_refused(velocity: xr.DataArray) -> None:
    target = velocity.isel(component=0, drop=True).rf.grid
    with pytest.raises(ValueError, match="declares tensor components"):
        velocity.rf.resample_to(target)
    with pytest.raises(ValueError, match="declares tensor components"):
        xrf.resample(velocity.rf.geometry, target)


def test_unframing_refuses_components_declared_in_array_coordinates(velocity: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="own coordinates, which need the binding"):
        velocity.rf.unframe()
    assert not velocity.rf.clear_tensor_components().rf.unframe().rf.is_framed
    in_frame = velocity.rf.change_component_basis(PLANE, units="mm/s")
    assert in_frame.rf.unframe().rf.has_tensor_components


def test_assume_frame_keeps_array_coordinate_components(velocity: xr.DataArray) -> None:
    renamed = xrf.ReferenceFrame.local(PLANE.coordinate_system)
    assert velocity.rf.assume_frame(renamed).rf.has_tensor_components


# Changing basis


def test_array_coordinates_to_frame_scales_vectors_and_covectors_oppositely(
    velocity: xr.DataArray, source: xrf.ArrayCoordinates
) -> None:
    ones = xr.ones_like(velocity)
    vector = ones.rf.change_component_basis(PLANE, units="mm/s")
    assert vector.component.values.tolist() == ["x", "y"]
    assert_close(vector.isel(j=0, i=0).values, [2.0, 3.0])
    assert vector.attrs["units"] == "mm/s"
    assert vector.rf.tensor_component_declaration.bases == (PLANE,)
    covector = ones.rf.clear_tensor_components().rf.declare_tensor_components(
        xrf.COVECTOR, component_dims=("component",), bases=source
    )
    gradient = covector.rf.change_component_basis(PLANE, units="1/mm")
    assert_close(gradient.isel(j=0, i=0).values, [0.5, 1 / 3])


def test_a_round_trip_restores_the_components(velocity: xr.DataArray) -> None:
    there = velocity.rf.change_component_basis(PLANE, units="mm/s")
    back = there.rf.change_component_basis(velocity.rf.coordinate_transform.source, units="1/s")
    assert back.rf.tensor_component_declaration == velocity.rf.tensor_component_declaration
    assert_close(back.values, velocity.values)


def test_a_non_commuting_round_trip_through_another_frame(velocity: xr.DataArray) -> None:
    shear = np.array([[1.0, 0.5], [0.2, 1.0]])
    to_other = xrf.AffineTransform.from_matrix(
        source=PLANE, target=OTHER_PLANE, matrix=shear, translation=(0.0, 0.0)
    )
    there = velocity.rf.change_component_basis(OTHER_PLANE, transform=to_other, units="mm/s")
    expected = np.einsum("ab,jib->jia", shear @ SPACING, velocity.values)
    assert_close(there.values, expected)
    back_transform = xrf.AffineTransform.from_matrix(
        source=OTHER_PLANE, target=PLANE, matrix=np.linalg.inv(shear), translation=(0.0, 0.0)
    )
    back = there.rf.change_component_basis(
        velocity.rf.coordinate_transform.source, transform=back_transform, units="1/s"
    )
    assert_close(back.values, velocity.values)


def test_a_proper_flip_moves_vectors_and_pseudovectors_alike(wind: xr.DataArray) -> None:
    flipped = wind.rf.change_component_basis(WSU)
    assert flipped.component.values.tolist() == ["w", "s", "u"]
    assert_close(flipped.isel(time=0).values, [-1.0, -2.0, 3.0])
    spin = wind.rf.clear_tensor_components().rf.declare_tensor_components(
        xrf.ORTHOGONAL_PSEUDOVECTOR, component_dims=("component",), bases=ENU
    )
    assert_close(spin.rf.change_component_basis(WSU).isel(time=0).values, [-1.0, -2.0, 3.0])


def test_an_improper_flip_gives_a_pseudovector_its_extra_sign(wind: xr.DataArray) -> None:
    assert_close(wind.rf.change_component_basis(WNU).isel(time=0).values, [-1.0, 2.0, 3.0])
    spin = wind.rf.clear_tensor_components().rf.declare_tensor_components(
        xrf.ORTHOGONAL_PSEUDOVECTOR, component_dims=("component",), bases=ENU
    )
    assert_close(spin.rf.change_component_basis(WNU).isel(time=0).values, [1.0, -2.0, -3.0])


def test_slots_in_different_bases_change_one_at_a_time(source: xrf.ArrayCoordinates) -> None:
    field = framed_field({"row": 2, "column": 2})
    mixed = field.rf.declare_tensor_components(
        xrf.SECOND_ORDER_CONTRAVARIANT_TENSOR,
        component_dims=("row", "column"),
        bases=(source, PLANE),
    )
    changed = mixed.rf.change_component_basis(PLANE, units="mm2")
    assert_close(changed.values, np.einsum("ab,jibc->jiac", SPACING, field.values))
    assert changed.rf.tensor_component_declaration.bases == (PLANE, PLANE)
    partial = mixed.rf.change_component_basis({"row": PLANE}, units="mm2")
    assert_close(partial.values, changed.values)


def test_a_density_changes_through_the_scalar_route(source: xrf.ArrayCoordinates) -> None:
    field = framed_field({})
    density = xrf.TensorType((), determinant_exponent=-1)
    changed = field.rf.change_component_basis(
        PLANE, tensor_type=density, from_basis=source, units="1/mm2"
    )
    assert_close(changed.values, field.values / 6.0)
    assert not changed.rf.has_tensor_components and changed.rf.is_framed


def test_a_same_frame_unit_change_needs_an_explicit_transform() -> None:
    metres = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("m", "m")))
    millimetres = xrf.ReferenceFrame.declared(
        metres.identifier, xrf.CoordinateSystem(("x", "y"), ("mm", "mm"))
    )
    raw = xr.DataArray(np.array([1.0, 2.0]), dims="component")
    declared = raw.rf.declare_tensor_components(
        xrf.VECTOR, component_dims=("component",), bases=metres
    )
    with pytest.raises(ValueError, match="Pass transform="):
        declared.rf.change_component_basis(millimetres, units="mm/s")
    scaling = xrf.AffineTransform.from_matrix(
        source=metres, target=millimetres, matrix=1000 * np.eye(2), translation=(0.0, 0.0)
    )
    changed = declared.rf.change_component_basis(millimetres, transform=scaling, units="mm/s")
    assert_close(changed.values, [1000.0, 2000.0])


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        (
            lambda a: a.rf.change_component_basis(WNU, units="m/s"),
            ValueError,
            "only accepted when the basis units change",
        ),
        (
            lambda a: a.rf.change_component_basis(WNU, tensor_type=xrf.SCALAR, from_basis=ENU),
            ValueError,
            "for undeclared scalars",
        ),
        (
            lambda a: a.rf.change_component_basis({"time": WNU}),
            ValueError,
            "not component dimensions",
        ),
        (lambda a: a.rf.change_component_basis(PLANE), ValueError, "no map from"),
        (
            lambda a: a.astype(str).rf.change_component_basis(WNU),
            TypeError,
            "real integer, floating or complex",
        ),
    ],
    ids=["units-unneeded", "scalar-args", "unknown-dim", "no-map", "dtype"],
)
def test_invalid_basis_changes_are_refused(
    wind: xr.DataArray, change: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        change(wind)


def test_a_change_of_units_requires_units(velocity: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="pass units="):
        velocity.rf.change_component_basis(PLANE)


def test_a_singular_map_is_refused_even_for_a_vector() -> None:
    raw = xr.DataArray(np.array([1.0, 2.0]), dims="component")
    declared = raw.rf.declare_tensor_components(
        xrf.VECTOR, component_dims=("component",), bases=PLANE
    )
    projection = xrf.AffineTransform.from_matrix(
        source=PLANE, target=OTHER_PLANE, matrix=[[1.0, 0.0], [0.0, 0.0]], translation=(0.0, 0.0)
    )
    with pytest.raises(ValueError, match="singular"):
        declared.rf.change_component_basis(OTHER_PLANE, transform=projection)


def test_a_transform_with_other_endpoints_is_not_used(wind: xr.DataArray) -> None:
    unrelated = xrf.AffineTransform.from_matrix(
        source=PLANE, target=OTHER_PLANE, matrix=np.eye(2), translation=(0.0, 0.0)
    )
    with pytest.raises(ValueError, match="but this change needs"):
        wind.rf.change_component_basis(PLANE, transform=unrelated)


def test_symmetric_slots_cannot_end_in_different_bases() -> None:
    symmetric = xrf.TensorType(("contravariant", "contravariant"), index_symmetry="symmetric")
    raw = xr.DataArray(np.eye(3), dims=("row", "column"))
    declared = raw.rf.declare_tensor_components(
        symmetric, component_dims=("row", "column"), bases=ENU
    )
    with pytest.raises(ValueError, match="both indices in one basis"):
        declared.rf.change_component_basis({"row": WNU})


def test_pseudotensor_slots_cannot_change_by_different_maps() -> None:
    odd = xrf.TensorType(("contravariant", "contravariant"), reflection_parity="odd")
    raw = xr.DataArray(np.eye(3), dims=("row", "column"))
    declared = raw.rf.declare_tensor_components(odd, component_dims=("row", "column"), bases=ENU)
    with pytest.raises(ValueError, match="needs one map for det J"):
        declared.rf.change_component_basis({"row": WNU})


def test_orthogonal_only_laws_are_checked_slot_by_slot(source: xrf.ArrayCoordinates) -> None:
    restricted = xrf.TensorType(("contravariant", "contravariant"), requires_orthogonal_map=True)
    field = framed_field({"row": 2, "column": 2})
    declared = field.rf.declare_tensor_components(
        restricted, component_dims=("row", "column"), bases=(source, PLANE)
    )
    with pytest.raises(ValueError, match="only for orthogonal maps"):
        declared.rf.change_component_basis(PLANE, units="mm2")


def test_the_scalar_route_needs_both_arguments(wind: xr.DataArray) -> None:
    plain = wind.rf.clear_tensor_components()
    with pytest.raises(ValueError, match="pass tensor_type="):
        plain.rf.change_component_basis(WNU)
    with pytest.raises(ValueError, match="order-0"):
        plain.rf.change_component_basis(WNU, tensor_type=xrf.VECTOR, from_basis=ENU)
    with pytest.raises(TypeError, match="one target basis"):
        plain.rf.change_component_basis({"component": WNU}, tensor_type=xrf.SCALAR, from_basis=ENU)


def test_a_renamed_binding_source_makes_the_basis_stale(velocity: xr.DataArray) -> None:
    renamed = velocity.rename(j="row")
    with pytest.raises(ValueError, match="binding changed"):
        renamed.rf.change_component_basis(PLANE, units="mm/s")


# Persistence


def test_encoding_round_trips_the_binding_and_the_declaration(velocity: xr.DataArray) -> None:
    encoded = velocity.rf.encode()
    assert not encoded.rf.is_framed and not encoded.rf.has_tensor_components
    record = json.loads(encoded.attrs["xarrayrf_tensor_components"])
    assert record["value"]["kind"] == "tensor_component_declaration"
    decoded = encoded.rf.decode()
    assert decoded.rf.tensor_component_declaration == velocity.rf.tensor_component_declaration
    assert decoded.rf.coordinate_transform == velocity.rf.coordinate_transform
    assert "xarrayrf_tensor_components" not in decoded.attrs


def test_an_unframed_declaration_round_trips_through_netcdf(wind: xr.DataArray) -> None:
    payload = wind.rf.encode().to_netcdf(engine="scipy")
    with xr.open_dataarray(io.BytesIO(payload), engine="scipy") as raw:
        decoded = raw.load().rf.decode()
    assert decoded.rf.tensor_component_declaration == wind.rf.tensor_component_declaration
    assert_close(decoded.values, wind.values)


def test_a_declaration_round_trips_through_zarr(tmp_path: Path, velocity: xr.DataArray) -> None:
    path = tmp_path / "velocity.zarr"
    velocity.rf.encode().to_zarr(path)
    with xr.open_dataarray(path, engine="zarr") as raw:
        decoded = raw.load().rf.decode()
    assert decoded.rf.tensor_component_declaration == velocity.rf.tensor_component_declaration


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (3, "must be JSON text"),
        ("{", "not valid JSON"),
        (json.dumps(xrf.encode(xrf.VECTOR)), "must hold a tensor component declaration"),
        (
            json.dumps(xrf.encode(xrf.TensorComponentDeclaration(xrf.VECTOR, ("component",), WNU))),
            "not valid here",
        ),
    ],
    ids=["not-text", "not-json", "wrong-kind", "stale-labels"],
)
def test_malformed_declarations_are_reported(
    wind: xr.DataArray, payload: object, match: str
) -> None:
    encoded = wind.rf.encode()
    encoded.attrs["xarrayrf_tensor_components"] = payload
    with pytest.raises(xrf.MalformedDataError, match=match):
        encoded.rf.decode()


def test_decoding_a_declared_array_or_nothing_is_refused(wind: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="no 'xarrayrf_binding' attribute"):
        wind.rf.clear_tensor_components().rf.decode()
    stale = wind.copy()
    stale.attrs["xarrayrf_tensor_components"] = wind.rf.encode().attrs["xarrayrf_tensor_components"]
    with pytest.raises(ValueError, match="already declares"):
        stale.rf.decode()


def test_clearing_drops_a_stale_encoded_declaration(wind: xr.DataArray) -> None:
    stale = wind.copy()
    stale.attrs["xarrayrf_tensor_components"] = "{}"
    assert "xarrayrf_tensor_components" not in stale.rf.clear_tensor_components().attrs


def test_docstring_examples_run() -> None:
    import doctest

    import xarrayrf._tensor_component_declaration as declaration_module
    from xarrayrf.native import _ReferenceFrameAccessor

    finder, runner = doctest.DocTestFinder(), doctest.DocTestRunner()
    for test in finder.find(_ReferenceFrameAccessor.declare_tensor_components) + finder.find(
        declaration_module
    ):
        runner.run(test)
    results = runner.summarize(verbose=False)
    assert results.attempted > 0 and results.failed == 0


def test_a_transform_that_does_not_fit_is_refused_even_when_a_map_is_derivable(
    wind: xr.DataArray,
) -> None:
    unrelated = xrf.AffineTransform.from_matrix(
        source=PLANE, target=OTHER_PLANE, matrix=np.eye(2), translation=(0.0, 0.0)
    )
    assert_close(wind.rf.change_component_basis(WNU).isel(time=0).values, [-1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="but this change needs"):
        wind.rf.change_component_basis(WNU, transform=unrelated)


@pytest.mark.parametrize("route", ["transform", "grid"])
def test_framing_cannot_make_a_component_dimension_a_geometry_dimension(
    wind: xr.DataArray, route: str
) -> None:
    field = wind.assign_coords(sample=("component", [0.0, 1.0, 2.0]))
    line = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("s",), ("m",)))
    binding = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("sample",), ("1",)),
        target=line,
        matrix=[[1.0]],
        translation=[0.0],
    )
    with pytest.raises(ValueError, match="hold declared tensor components"):
        if route == "transform":
            field.rf.frame(binding, dims=("component",))
        else:
            field.rf.frame(xrf.Grid(binding, {"sample": ("component", [0.0, 1.0, 2.0])}))


def test_a_stale_array_coordinates_basis_cannot_be_unframed_or_encoded(
    velocity: xr.DataArray,
) -> None:
    stale = velocity.rename(j="row")
    with pytest.raises(ValueError, match="own coordinates, which need the binding"):
        stale.rf.unframe()
    with pytest.raises(ValueError, match="no longer this array's binding source"):
        stale.rf.encode()
    assert stale.rf.clear_tensor_components().rf.encode().rf.decode().rf.is_framed


def test_mixed_variance_slots_each_change_by_their_own_map(source: xrf.ArrayCoordinates) -> None:
    shear = np.array([[1.0, 0.5], [0.2, 1.0]])
    from_other = xrf.AffineTransform.from_matrix(
        source=OTHER_PLANE, target=PLANE, matrix=shear, translation=(0.0, 0.0)
    )
    field = framed_field({"row": 2, "column": 2})
    mixed = field.rf.declare_tensor_components(
        xrf.TensorType(("contravariant", "covariant")),
        component_dims=("row", "column"),
        bases=(source, OTHER_PLANE),
    )
    changed = mixed.rf.change_component_basis(PLANE, transform=from_other, units="1")
    # Written per index, independently of the slot-by-slot route.
    expected = np.einsum("ab,dc,jibd->jiac", SPACING, np.linalg.inv(shear), field.values)
    assert_close(changed.values, expected)
    assert changed.rf.tensor_component_declaration.bases == (PLANE, PLANE)
