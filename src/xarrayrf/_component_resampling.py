"""Resampling declared tensor components: interpolate in the source basis, then map.

At each target point ``y`` the planner's frame map ``T`` (target frame to source frame) gives
``L = (D T(y))^-1``, which pushes source-frame components to the target frame. Per slot the
applied map is ``M_out @ R(L) @ M_in``: source basis to source frame, the reorientation ``R`` of
``L``, then target frame to the output basis. The design and the measured decisions are in
``docs/dev/architecture/value_geometry_stage3_plan.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr

from ._affine import equilibrated_inverse
from ._array_coordinates import ArrayCoordinates
from ._frame import ReferenceFrame
from ._tensor_component_declaration import ComponentBasis, TensorComponentDeclaration
from ._tensor_components import (
    TensorComponentIndex,
    _check_units,
    basis_map,
    check_common_map,
    declaration_of,
    transform_declared_components,
)
from ._tensor_type import polar_orthogonal_factor
from ._transform import SupportsAffine, SupportsJacobian, SupportsPoints

type ComponentReorientation = (
    Literal["jacobian", "polar_orthogonal_factor"]
    | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]
)
_NAMED_REORIENTATIONS = ("jacobian", "polar_orthogonal_factor")


@dataclass(frozen=True)
class ComponentRequest:
    """A declared source's validated component arguments."""

    index: TensorComponentIndex
    output_basis: ComponentBasis
    reorientation: ComponentReorientation
    units: str | None


def component_request(
    array: xr.DataArray,
    *,
    output_component_basis: ComponentBasis | None,
    component_reorientation: ComponentReorientation | None,
    units: str | None,
    fill_value: float,
    framed_result: bool,
) -> ComponentRequest | None:
    """Validate the component arguments against the source's declaration, before planning."""
    index = declaration_of(array)
    given = [
        name
        for name, value in (
            ("output_component_basis", output_component_basis),
            ("component_reorientation", component_reorientation),
            ("units", units),
        )
        if value is not None
    ]
    if index is None:
        if given:
            raise ValueError(
                f"{', '.join(given)} apply only to a source that declares tensor components; "
                "declare them with rf.declare_tensor_components"
            )
        return None
    missing = [
        name
        for name, value in (
            ("output_component_basis", output_component_basis),
            ("component_reorientation", component_reorientation),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"the source declares tensor components, so resampling needs {' and '.join(missing)}: "
            "the basis of the result's components, and 'jacobian', 'polar_orthogonal_factor' or "
            "a callable for how they follow the transform. To resample them as independent "
            "scalars deliberately, call rf.clear_tensor_components() first"
        )
    assert output_component_basis is not None and component_reorientation is not None
    if not isinstance(output_component_basis, ReferenceFrame | ArrayCoordinates):
        raise TypeError(
            "output_component_basis must be a ReferenceFrame or ArrayCoordinates, got "
            f"{type(output_component_basis).__name__}"
        )
    if isinstance(output_component_basis, ArrayCoordinates) and not framed_result:
        raise ValueError(
            "resample returns an unframed array, which cannot declare components in "
            "ArrayCoordinates; use rf.resample_to, or a ReferenceFrame output basis"
        )
    if isinstance(component_reorientation, str):
        if component_reorientation not in _NAMED_REORIENTATIONS:
            raise ValueError(
                f"component_reorientation must be one of {_NAMED_REORIENTATIONS} or a callable, "
                f"got {component_reorientation!r}"
            )
    elif not callable(component_reorientation):
        raise TypeError(
            "component_reorientation must be 'jacobian', 'polar_orthogonal_factor' or a "
            f"callable, got {type(component_reorientation).__name__}"
        )
    tensor_type = index.declaration.tensor_type
    if component_reorientation == "polar_orthogonal_factor" and tensor_type.determinant_exponent:
        raise ValueError(
            "polar_orthogonal_factor drops the volume change, which a density's factor "
            "|det J|^w needs; use component_reorientation='jacobian'"
        )
    value = complex(fill_value)
    if not (math.isnan(value.real) or math.isnan(value.imag) or value == 0):
        raise ValueError(
            f"fill_value {fill_value!r} would become a different, valid-looking tensor under the "
            "component maps; use NaN or 0 for declared components"
        )
    return ComponentRequest(index, output_component_basis, component_reorientation, units)


def resample_components(
    result: xr.DataArray,
    request: ComponentRequest,
    *,
    source_transform: SupportsPoints,
    target_transform: SupportsPoints,
    frame_map: SupportsPoints | None,
    target_dims: Sequence[str],
    target_frame_points: Callable[[int, int], npt.NDArray[np.float64]],
    block_points: int,
) -> tuple[xr.DataArray, TensorComponentDeclaration]:
    """Map resampled source-basis components into the output basis, and relabel them.

    Returns the relabelled values without a declaration index, and the declaration to attach
    once the result is framed.
    """
    declaration = request.index.declaration
    tensor_type = declaration.tensor_type
    source_frame, target_frame = source_transform.target, target_transform.target
    assert isinstance(source_frame, ReferenceFrame) and isinstance(target_frame, ReferenceFrame)
    output = request.output_basis
    inputs = [_input_map(basis, source_frame, source_transform) for basis in declaration.bases]
    check_common_map(tensor_type, inputs)
    try:
        output_map = basis_map(
            target_frame, output, binding_transform=target_transform, transform=None
        )
    except ValueError as error:
        raise ValueError(
            f"output_component_basis must be the target frame, an equivalent frame in another "
            f"coordinate system, or the target's ArrayCoordinates: {error}"
        ) from error
    targets = (output,) * len(declaration.bases)
    _check_units(declaration.bases, targets, request.units)
    if request.reorientation == "polar_orthogonal_factor":
        units = set(source_frame.coordinate_system.units) | set(
            target_frame.coordinate_system.units
        )
        if len(units) != 1:
            raise ValueError(
                "polar_orthogonal_factor needs one unit on every axis of the source and target "
                f"frames, so its factor is a rotation rather than unit scaling; got "
                f"{sorted(map(str, units))}"
            )
    new_declaration = TensorComponentDeclaration(
        tensor_type, declaration.component_dims, targets, packing=declaration.packing
    )
    dims = list(declaration.component_dims)
    if frame_map is not None and not isinstance(frame_map, SupportsAffine | SupportsJacobian):
        raise ValueError(
            "resampling declared components needs the frame map's Jacobian; this transform "
            "provides none"
        )

    def compose(push: npt.NDArray[np.float64]) -> list[npt.NDArray[np.float64]]:
        reoriented = _reorient(push, request.reorientation)
        composed: dict[int, npt.NDArray[np.float64]] = {}
        maps = []
        for matrix in inputs:
            key = next((k for k, m in enumerate(inputs) if np.array_equal(m, matrix)), 0)
            if key not in composed:
                composed[key] = output_map @ reoriented @ matrix
            maps.append(composed[key])
        return maps

    total = int(np.prod([result.sizes[dim] for dim in target_dims], dtype=np.int64))
    dtype = np.result_type(result.dtype, np.float64)
    pointwise = frame_map is not None and not isinstance(frame_map, SupportsAffine)
    if total == 0:
        mapped = result.astype(dtype)
    elif not pointwise:
        n = len(target_frame.axes)
        push = (
            np.eye(n)
            if frame_map is None
            else _push(np.asarray(cast(SupportsAffine, frame_map).matrix, dtype=np.float64))
        )
        maps = compose(push)
        mapped = _apply(
            result,
            lambda values: transform_declared_components(values, declaration, maps),
            dims,
            dtype,
        )
    else:
        jacobian_map = cast(SupportsJacobian, frame_map)
        shape = tuple(result.sizes[dim] for dim in target_dims)
        component_ndim = len(dims)

        def apply_pointwise(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
            leading = values.shape[: values.ndim - len(shape) - component_ndim]
            components = values.shape[values.ndim - component_ndim :]
            flat = values.reshape(*leading, total, *components)
            out = np.empty(flat.shape, dtype=dtype)
            for start in range(0, total, block_points):
                stop = min(start + block_points, total)
                jacobians = np.asarray(
                    jacobian_map.jacobian(target_frame_points(start, stop)), dtype=np.float64
                )
                block = (slice(None),) * len(leading) + (slice(start, stop),)
                try:
                    push = _push(jacobians)
                except ValueError as error:
                    # The inverse names a block-relative index; report the target sample.
                    sample = _first_singular(jacobians, start, shape, target_dims)
                    raise ValueError(
                        f"the frame map's Jacobian is singular or not finite at target sample "
                        f"{sample}, so components there have no image; a folding or "
                        "undefined registration cannot carry tensor components"
                    ) from error
                out[block] = transform_declared_components(flat[block], declaration, compose(push))
            return out.reshape(values.shape)

        mapped = _apply(result, apply_pointwise, [*target_dims, *dims], dtype)
    names = [name for name in dims if name in mapped.coords]
    relabelled = mapped.drop_indexes([name for name in names if name in mapped.xindexes]).drop_vars(
        names
    )
    relabelled = relabelled.assign_coords(
        {
            dim: (dim, np.asarray(new_declaration.labels(position)))
            for position, dim in enumerate(dims)
        }
    )
    for dim in names:
        relabelled.coords[dim].attrs.update(result.coords[dim].attrs)
    return relabelled.drop_indexes(dims), new_declaration


def _input_map(
    basis: ComponentBasis, source_frame: ReferenceFrame, source_transform: SupportsPoints
) -> npt.NDArray[np.float64]:
    try:
        return basis_map(basis, source_frame, binding_transform=source_transform, transform=None)
    except ValueError as error:
        raise ValueError(
            "the source components must be in the source's ArrayCoordinates, its frame, or an "
            f"equivalent frame in another coordinate system; change their basis first: {error}"
        ) from error


def _push(jacobian: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``L = (D T)^-1``: the frame map's Jacobian is source axes by target axes."""
    if jacobian.ndim < 2 or jacobian.shape[-1] != jacobian.shape[-2]:
        raise ValueError(
            f"the frame map's Jacobian has shape {jacobian.shape}; component maps need source "
            "and target frames of one dimension"
        )
    return equilibrated_inverse(jacobian)


def _first_singular(
    jacobians: npt.NDArray[np.float64],
    start: int,
    shape: tuple[int, ...],
    target_dims: Sequence[str],
) -> dict[str, int]:
    """The target sample of the first Jacobian in a block that cannot be inverted."""
    for offset, matrix in enumerate(jacobians):
        try:
            equilibrated_inverse(matrix)
        except ValueError:
            position = np.unravel_index(start + offset, shape) if shape else ()
            return {str(dim): int(index) for dim, index in zip(target_dims, position, strict=True)}
    return {}


def _reorient(
    push: npt.NDArray[np.float64], reorientation: ComponentReorientation
) -> npt.NDArray[np.float64]:
    if reorientation == "jacobian":
        return push
    if reorientation == "polar_orthogonal_factor":
        return polar_orthogonal_factor(push)
    assert callable(reorientation)
    returned = np.asarray(reorientation(push))
    if returned.shape != push.shape:
        raise ValueError(
            f"component_reorientation returned shape {returned.shape} for maps shaped "
            f"{push.shape}; it must return one map per map"
        )
    if returned.dtype.kind not in "iuf" or not np.isfinite(returned).all():
        raise ValueError("component_reorientation must return real, finite maps")
    return returned.astype(np.float64, copy=False)


def _apply(
    array: xr.DataArray,
    function: Callable[[npt.NDArray[Any]], npt.NDArray[Any]],
    core_dims: Sequence[str],
    dtype: np.dtype[Any],
) -> xr.DataArray:
    core = list(core_dims)
    if array.chunks is not None:
        # The interpolation leaves component dims as loop dims, so they may be split; the maps
        # need each point's components and each block's target samples in one chunk.
        array = array.chunk(dict.fromkeys(core, -1))
    result = xr.apply_ufunc(
        function,
        array,
        input_core_dims=[core],
        output_core_dims=[core],
        dask="parallelized",
        output_dtypes=[dtype],
        keep_attrs=True,
    )
    return cast(xr.DataArray, result.transpose(*array.dims))
