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
from ._coordinate_system import CARTESIAN
from ._frame import ReferenceFrame
from ._frame_field import EastNorthUp, FrameField
from ._tensor_component_declaration import ComponentBasis, TensorComponentDeclaration
from ._tensor_components import (
    TensorComponentIndex,
    _check_units,
    basis_map,
    check_common_map,
    declaration_of,
    refuse_separate_maps,
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
    if not isinstance(output_component_basis, ReferenceFrame | ArrayCoordinates | FrameField):
        raise TypeError(
            "output_component_basis must be a ReferenceFrame, ArrayCoordinates or FrameField, got "
            f"{type(output_component_basis).__name__}"
        )
    if isinstance(output_component_basis, ArrayCoordinates | FrameField) and not framed_result:
        # Both mean something only at the result's own sample points, which an unframed
        # array does not have.
        raise ValueError(
            "resample returns an unframed array, which cannot declare components in "
            f"{type(output_component_basis).__name__}; use rf.resample_to, or a ReferenceFrame "
            "output basis"
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


def pole_signs(
    declaration: TensorComponentDeclaration, source_frame: ReferenceFrame
) -> npt.NDArray[np.float64] | None:
    """Signs over the component dims that carry halo rows across a pole, or ``None``.

    ``None`` when some slot's basis has no pole rule, and then no pole halo is built: the chart
    basis diverges at a pole (its longitude component goes like 1 / cos φ), and a user frame
    field's behaviour there is unknown. A constant Cartesian basis is continuous across a pole
    (sign +1); an east/north/up field over the source frame reverses east and north. Each stored
    entry takes the product of its slots' signs; every map is diagonal with determinant +1, so
    pseudotensor and density factors are 1. The plan is east_north_pole_halo_plan.md.
    """
    slots: list[npt.NDArray[np.float64]] = []
    for basis in declaration.bases:
        if (
            isinstance(basis, ReferenceFrame)
            and basis.coordinate_system.representation == CARTESIAN
        ):
            slots.append(np.ones(len(basis.axes)))
        elif isinstance(basis, EastNorthUp) and basis.base == source_frame:
            slots.append(np.asarray(basis._pole_signs()))
        else:
            return None
    packing = declaration.packing
    if packing is not None:
        # Symmetry requires one basis for both slots.
        signs = slots[0]
        return np.array([signs[i] * signs[j] for i, j in packing.index_pairs])
    product = np.ones(())
    for signs in slots:
        product = np.multiply.outer(product, signs)
    return product


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
    # One frame and one basis: every map is the identity, and nothing is evaluated at points.
    # Exact even at a pole, where an east/north field is its meridian limit, and for a basis
    # (Earth-centred axes on a longitude/latitude chart) with no constant map to the frame.
    # With one frame and one basis the basis maps cancel around the reorientation of an identity
    # push. The named reorientations return the identity; a callable is evaluated once, and if
    # it returns a multiple of the identity that multiple commutes with the basis maps, so it
    # applies directly. Anything else goes through the evaluated path.
    same_basis = frame_map is None and all(basis == output for basis in declaration.bases)
    factor: float | None = None
    if same_basis:
        n = len(target_frame.axes)
        reoriented_identity = np.asarray(_reorient(np.eye(n), request.reorientation))
        candidate = float(reoriented_identity[0, 0]) if reoriented_identity.shape == (n, n) else 0.0
        if np.array_equal(reoriented_identity, candidate * np.eye(n)):
            if candidate == 0.0 or not math.isfinite(candidate):
                raise ValueError(
                    f"component_reorientation returned {candidate} times the identity; a "
                    "component map must be invertible and finite"
                )
            factor = candidate
    identity = factor is not None
    inputs: list[npt.NDArray[np.float64] | FrameField] = (
        [np.eye(len(basis.axes)) for basis in declaration.bases]
        if identity
        else [_input_side(basis, source_frame, source_transform) for basis in declaration.bases]
    )
    constant_inputs = [side for side in inputs if not isinstance(side, FrameField)]
    if len(constant_inputs) == len(inputs):
        check_common_map(tensor_type, constant_inputs)
    elif len(set(declaration.bases)) > 1:
        # A frame field varies by point, so slots in different bases change by different maps.
        refuse_separate_maps(tensor_type)
    output_side = (
        np.eye(len(output.axes))
        if identity
        else _output_side(output, target_frame, target_transform)
    )
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

    total = int(np.prod([result.sizes[dim] for dim in target_dims], dtype=np.int64))
    # The resampled values already carry the output dtype; the maps apply in float64.
    dtype = result.dtype if result.dtype.kind in "fc" else np.result_type(result.dtype, np.float64)
    fields = any(isinstance(side, FrameField) for side in (*inputs, output_side))
    pointwise = fields or (frame_map is not None and not isinstance(frame_map, SupportsAffine))
    if total == 0 or factor == 1.0:
        mapped = result.astype(dtype)
    elif factor is not None:
        scaled = [factor * np.eye(len(basis.axes)) for basis in declaration.bases]
        mapped = _apply(
            result,
            lambda values: transform_declared_components(values, declaration, scaled),
            dims,
            dtype,
        )
    elif not pointwise:
        n = len(target_frame.axes)
        push = (
            np.eye(n)
            if frame_map is None
            else _push(np.asarray(cast(SupportsAffine, frame_map).matrix, dtype=np.float64))
        )
        reoriented = _reorient(push, request.reorientation)
        constant_output = cast(npt.NDArray[np.float64], output_side)
        maps = [constant_output @ reoriented @ cast(npt.NDArray[np.float64], m) for m in inputs]
        mapped = _apply(
            result,
            lambda values: transform_declared_components(values, declaration, maps),
            dims,
            dtype,
        )
    else:
        shape = tuple(result.sizes[dim] for dim in target_dims)
        component_ndim = len(dims)

        def maps_at(start: int, stop: int) -> list[npt.NDArray[np.float64]]:
            target_points = target_frame_points(start, stop)
            if frame_map is None:
                push = np.eye(len(target_frame.axes))
                source_points = target_points
            else:
                if isinstance(frame_map, SupportsAffine):
                    push = _push(np.asarray(frame_map.matrix, dtype=np.float64))
                else:
                    jacobian = np.asarray(
                        cast(SupportsJacobian, frame_map).jacobian(target_points), dtype=np.float64
                    )
                    push = _push(jacobian)
                source_points = frame_map.transform_point(target_points)
            reoriented = _reorient(push, request.reorientation)
            out = (
                output_side.inverse_matrix(target_points)
                if isinstance(output_side, FrameField)
                else output_side
            )
            return [
                np.matmul(
                    np.matmul(out, reoriented),
                    side.matrix(source_points) if isinstance(side, FrameField) else side,
                )
                for side in inputs
            ]

        def apply_pointwise(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
            leading = values.shape[: values.ndim - len(shape) - component_ndim]
            components = values.shape[values.ndim - component_ndim :]
            flat = values.reshape(*leading, total, *components)
            out = np.empty(flat.shape, dtype=dtype)
            for start in range(0, total, block_points):
                stop = min(start + block_points, total)
                block = (slice(None),) * len(leading) + (slice(start, stop),)
                try:
                    # A transform or frame field may raise where its matrix does not exist (a
                    # pole) rather than return a singular matrix.
                    maps = maps_at(start, stop)
                    for matrix in maps:
                        equilibrated_inverse(np.asarray(matrix, dtype=np.float64))
                except ValueError as error:
                    # Both failures name a block-relative index; report the target sample.
                    sample = _first_failing(maps_at, start, stop, shape, target_dims)
                    raise ValueError(
                        f"the component maps (the frame map's Jacobian, or a frame-field basis) "
                        f"are singular, not finite or undefined at target sample {sample}, so "
                        "components there have no image; a folding or undefined registration, "
                        "or a pole of a chart transition or frame field, cannot carry tensor "
                        "components"
                    ) from error
                out[block] = transform_declared_components(flat[block], declaration, maps)
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


def _input_side(
    basis: ComponentBasis, source_frame: ReferenceFrame, source_transform: SupportsPoints
) -> npt.NDArray[np.float64] | FrameField:
    """The map from a slot's basis into the source frame: a matrix, or a field to evaluate."""
    if isinstance(basis, FrameField):
        if basis.base != source_frame:
            raise ValueError(
                "a frame-field basis must be over the source's frame in its own coordinate "
                f"system; this field is over {basis.base.axes}"
            )
        return basis
    try:
        return basis_map(basis, source_frame, binding_transform=source_transform, transform=None)
    except ValueError as error:
        raise ValueError(
            "the source components must be in the source's ArrayCoordinates, its frame, an "
            "equivalent frame in another coordinate system, or a frame field over its frame; "
            f"change their basis first: {error}"
        ) from error


def _output_side(
    output: ComponentBasis, target_frame: ReferenceFrame, target_transform: SupportsPoints
) -> npt.NDArray[np.float64] | FrameField:
    """The map from the target frame into the output basis: a matrix, or a field to evaluate."""
    if isinstance(output, FrameField):
        if output.base != target_frame:
            raise ValueError(
                "an output frame field must be over the target frame in its own coordinate "
                f"system; this field is over {output.base.axes}"
            )
        return output
    try:
        return basis_map(target_frame, output, binding_transform=target_transform, transform=None)
    except ValueError as error:
        raise ValueError(
            f"output_component_basis must be the target frame, an equivalent frame in another "
            f"coordinate system, the target's ArrayCoordinates, or a frame field over the target "
            f"frame: {error}"
        ) from error


def _push(jacobian: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``L = (D T)^-1``: the frame map's Jacobian is source axes by target axes."""
    if jacobian.ndim < 2 or jacobian.shape[-1] != jacobian.shape[-2]:
        raise ValueError(
            f"the frame map's Jacobian has shape {jacobian.shape}; component maps need source "
            "and target frames of one dimension"
        )
    return equilibrated_inverse(jacobian)


def _first_failing(
    maps_at: Callable[[int, int], list[npt.NDArray[np.float64]]],
    start: int,
    stop: int,
    shape: tuple[int, ...],
    target_dims: Sequence[str],
) -> dict[str, int]:
    """The target sample of the first point in a block whose maps fail or are singular.

    Runs only on the error path, one point at a time, so the cost is paid once per failure.
    """
    for offset in range(stop - start):
        try:
            for matrix in maps_at(start + offset, start + offset + 1):
                equilibrated_inverse(np.asarray(matrix).reshape(-1, *matrix.shape[-2:])[0])
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

    def typed(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        # The maps apply in float64; the result takes the resampled values' dtype.
        return function(values).astype(dtype, copy=False)

    result = xr.apply_ufunc(
        typed,
        array,
        input_core_dims=[core],
        output_core_dims=[core],
        dask="parallelized",
        output_dtypes=[dtype],
        keep_attrs=True,
    )
    return cast(xr.DataArray, result.transpose(*array.dims))
