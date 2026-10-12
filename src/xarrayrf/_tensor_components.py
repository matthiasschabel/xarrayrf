"""Private xarray index carrying a tensor component declaration, and basis changes through it.

The declaration is advisory with respect to values: the index is consulted by structural
operations (selection, alignment, concatenation, renaming) and by xarrayrf, but an xarray index
lives with the coordinates and never sees element-wise operations or payload replacement. The
measured boundary is in ``docs/dev/architecture/value_geometry_stage2_plan.md``.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from xarray.core.indexing import IndexSelResult
from xarray.indexes import PandasIndex

from ._affine import equilibrated_inverse
from ._array_coordinates import ArrayCoordinates
from ._binding import BindingIndex
from ._frame import ReferenceFrame
from ._frame_field import EastNorthUp, FrameField
from ._orientation import coordinate_system_change
from ._symmetric_packing import SymmetricPacking
from ._tensor_component_declaration import ComponentBasis, TensorComponentDeclaration
from ._tensor_type import TensorType, transform_tensor_components
from ._transform import (
    SupportsAffine,
    SupportsInverse,
    SupportsJacobian,
    SupportsPoints,
    check_transform,
)

DECLARATION_ATTR = "xarrayrf_tensor_components"


class TensorComponentIndex(xr.Index):
    """Own each component dimension's label coordinate and the declaration they share."""

    def __init__(
        self, declaration: TensorComponentDeclaration, axes: Mapping[str, PandasIndex]
    ) -> None:
        self.declaration = declaration
        self.axes = dict(axes)

    @classmethod
    def from_variables(
        cls, variables: Mapping[Any, xr.Variable], *, options: Mapping[str, Any]
    ) -> TensorComponentIndex:
        raise ValueError(
            "TensorComponentIndex cannot be constructed with set_xindex; use "
            "rf.declare_tensor_components()"
        )

    def create_variables(
        self, variables: Mapping[Any, xr.Variable] | None = None
    ) -> dict[str, xr.Variable]:
        created: dict[str, xr.Variable] = {}
        for name, index in self.axes.items():
            source = (
                {name: variables[name]} if variables is not None and name in variables else None
            )
            created.update(index.create_variables(source))
        return created

    def _position_of(self, name: str) -> int:
        return self.declaration.component_dims.index(str(self.axes[name].dim))

    def sel(
        self, labels: dict[Any, Any], method: str | None = None, tolerance: Any = None
    ) -> IndexSelResult:
        indexers: dict[Any, Any] = {}
        for name, label in labels.items():
            indexers.update(
                self.axes[name].sel({name: label}, method=method, tolerance=tolerance).dim_indexers
            )
        return IndexSelResult(indexers)

    def isel(self, indexers: Mapping[Any, Any]) -> TensorComponentIndex | None:
        axes: dict[str, PandasIndex] = {}
        for name, index in self.axes.items():
            if index.dim not in indexers:
                axes[name] = index
                continue
            indexer = indexers[index.dim]
            if isinstance(indexer, xr.Variable) and indexer.dims != (index.dim,):
                return None
            reduced = index.isel({index.dim: indexer})
            # A scalar, a subset or a permutation of components is not this tensor: drop the
            # whole declaration and leave the labels as plain coordinates.
            if reduced is None or tuple(reduced.index) != self.declaration.labels(
                self._position_of(name)
            ):
                return None
            axes[name] = reduced
        return type(self)(self.declaration, axes)

    def equals(self, other: xr.Index, *, exclude: frozenset[Hashable] | None = None) -> bool:
        if not isinstance(other, TensorComponentIndex):
            return False
        return (
            self.declaration == other.declaration
            and self.axes.keys() == other.axes.keys()
            and all(index.equals(other.axes[name]) for name, index in self.axes.items())
        )

    def _require_compatible(self, other: xr.Index) -> TensorComponentIndex:
        if not isinstance(other, TensorComponentIndex):
            raise ValueError(
                "cannot combine declared tensor components with untyped ones; declare the other "
                "operand with rf.declare_tensor_components or clear this one with "
                "rf.clear_tensor_components"
            )
        if not self.equals(other):
            raise ValueError(
                f"incompatible tensor component declarations: {self._difference(other)}"
            )
        return other

    def _difference(self, other: TensorComponentIndex) -> str:
        mine, theirs = self.declaration, other.declaration
        if mine.tensor_type != theirs.tensor_type:
            return (
                f"{mine.tensor_type!r} and {theirs.tensor_type!r} follow different laws, so "
                "their components cannot be combined"
            )
        if mine.component_dims != theirs.component_dims:
            return f"component dimensions {mine.component_dims} and {theirs.component_dims} differ"
        if mine.bases != theirs.bases:
            return (
                "the components are in different bases; express one in the other's with "
                "rf.change_component_basis"
            )
        if mine.packing != theirs.packing:
            return (
                f"the components are packed differently ({mine.packing!r} and "
                f"{theirs.packing!r}); unpack both with rf.unpack_symmetric_components"
            )
        return "the component labels differ; select them in basis order"

    def join(self, other: xr.Index, how: str = "inner") -> TensorComponentIndex:
        self._require_compatible(other)
        return self

    def reindex_like(
        self, other: xr.Index, method: str | None = None, tolerance: Any = None
    ) -> dict[Any, Any]:
        compatible = self._require_compatible(other)
        indexers: dict[Any, Any] = {}
        for name, index in self.axes.items():
            indexers.update(
                index.reindex_like(compatible.axes[name], method=method, tolerance=tolerance)
            )
        return indexers

    def join_overlapping(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse adopting this declaration for plain coordinates with component labels."""
        raise ValueError(
            "cannot combine declared tensor components with untyped ones carrying the same "
            "labels; shared labels do not assert the same transformation law"
        )

    def check_unindexed_coord_conflicts(self, variables: Mapping[Hashable, xr.Variable]) -> None:
        """Refuse adopting this declaration for unindexed component labels."""
        raise ValueError(
            "cannot combine declared tensor components with untyped ones carrying the same "
            "labels; shared labels do not assert the same transformation law"
        )

    def check_override(self, other: xr.Index) -> None:
        """Refuse replacement under ``join='override'``."""
        raise ValueError("join='override' is unsupported for declared tensor components")

    def check_stack(self, dims: Sequence[Hashable]) -> None:
        """Refuse stacking a component dimension."""
        raise ValueError("cannot stack component dimensions; call rf.clear_tensor_components()")

    def check_pad(self, pad_width: Mapping[Hashable, int | tuple[int, int]]) -> None:
        """Refuse padding a component dimension."""
        raise ValueError("cannot pad component dimensions; call rf.clear_tensor_components()")

    def check_coarsen(self, windows: Mapping[Hashable, int]) -> None:
        """Refuse coarsening a component dimension."""
        raise ValueError("cannot coarsen component dimensions; call rf.clear_tensor_components()")

    def roll(self, shifts: Mapping[Any, int]) -> TensorComponentIndex:
        """Refuse rolling labels along a component dimension, which permutes components."""
        raise ValueError("cannot roll component dimensions; call rf.clear_tensor_components()")

    @classmethod
    def concat(cls, indexes: Any, dim: Any, positions: Any = None) -> TensorComponentIndex:
        raise ValueError(
            "cannot concatenate along a component dimension; call rf.clear_tensor_components()"
        )

    def rename(
        self, name_dict: Mapping[Any, Any], dims_dict: Mapping[Any, Any]
    ) -> TensorComponentIndex:
        dims = tuple(str(dims_dict.get(dim, dim)) for dim in self.declaration.component_dims)
        names = {name: str(name_dict.get(name, name)) for name in self.axes}
        if dims == self.declaration.component_dims and all(a == b for a, b in names.items()):
            return self
        declaration = TensorComponentDeclaration(
            self.declaration.tensor_type,
            dims,
            self.declaration.bases,
            packing=self.declaration.packing,
        )
        axes = {
            names[name]: index.rename(name_dict, dims_dict)  # type: ignore[no-untyped-call]
            for name, index in self.axes.items()
        }
        return type(self)(declaration, axes)

    def _copy(self, deep: bool = True, memo: dict[int, Any] | None = None) -> TensorComponentIndex:
        return type(self)(
            self.declaration, {name: index.copy(deep=deep) for name, index in self.axes.items()}
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return type(self), (self.declaration, self.axes)

    def __repr__(self) -> str:
        return f"TensorComponentIndex({self.declaration!r})"


def declaration_of(array: xr.DataArray, *, verify: bool = True) -> TensorComponentIndex | None:
    """Return the array's component index, or None when undeclared.

    With ``verify``, the index must still own every coordinate it claims; a split index (one
    whose coordinates another index took over) raises so it cannot be read as a declaration.
    """
    owners = {
        name: index
        for name, index in array.xindexes.items()
        if isinstance(index, TensorComponentIndex)
    }
    indexes = {id(index): index for index in owners.values()}
    if not indexes:
        return None
    if len(indexes) != 1:
        raise ValueError("array has more than one tensor component declaration")
    index = next(iter(indexes.values()))
    if verify and set(index.axes) != set(owners):
        lost = sorted(map(str, set(index.axes) - set(owners)))
        raise ValueError(
            f"the tensor component declaration no longer owns its label coordinate(s) {lost}; "
            "call rf.clear_tensor_components() and declare again"
        )
    return index


def declared_coordinates(
    array: xr.DataArray, declaration: TensorComponentDeclaration, binding: BindingIndex | None
) -> xr.Coordinates:
    """Validate a declaration against an array and build its labelled, indexed coordinates."""
    geometry_dims = binding.dims if binding is not None else ()
    variables: dict[str, xr.Variable] = {}
    for position, dim in enumerate(declaration.component_dims):
        basis = declaration.bases[position]
        labels = declaration.labels(position)
        if dim not in array.dims:
            raise ValueError(f"component dimension {dim!r} is not a dimension of the array")
        if dim in geometry_dims:
            raise ValueError(
                f"component dimension {dim!r} is a geometry dimension of the frame binding; "
                "components need a dimension of their own"
            )
        if array.sizes[dim] != len(labels):
            held = (
                f"the packing holds {len(labels)} components"
                if declaration.packing is not None
                else f"its basis has {len(labels)} axes"
            )
            raise ValueError(
                f"component dimension {dim!r} has length {array.sizes[dim]}, but {held}"
            )
        if isinstance(basis, FrameField) and (
            binding is None or basis.base != binding.transform.target
        ):
            raise ValueError(
                "a frame field basis must be over this array's frame, in its own coordinate "
                "system: the field is evaluated at the array's sample points"
            )
        if isinstance(basis, ArrayCoordinates) and (
            binding is None or basis != binding.transform.source
        ):
            raise ValueError(
                "an ArrayCoordinates basis must be the source of this array's frame "
                "binding: components per unit of the array's own coordinate values"
            )
        if dim in array.coords:
            current = array.coords[dim].variable
            if current.dims != (dim,):
                raise ValueError(f"the coordinate {dim!r} must vary along {dim!r} alone")
            if current.dtype.kind not in "UO" or tuple(current.values.tolist()) != labels:
                raise ValueError(
                    f"the labels of component dimension {dim!r} must be the basis axes "
                    f"{labels}, in order; got {tuple(current.values.tolist())}"
                )
            variables[dim] = xr.Variable((dim,), np.asarray(labels), attrs=dict(current.attrs))
        else:
            variables[dim] = xr.Variable((dim,), np.asarray(labels))
    axes = {
        name: PandasIndex.from_variables({name: variable}, options={})
        for name, variable in variables.items()
    }
    index = TensorComponentIndex(declaration, axes)
    return xr.Coordinates(variables, indexes={name: index for name in variables})


def attach(array: xr.DataArray, coordinates: xr.Coordinates) -> xr.DataArray:
    """Replace the component coordinates with the declaration's labelled, indexed ones.

    The old coordinates are dropped first: assigning over an unindexed coordinate of the same
    name would ask this index whether to adopt it (patched xarray), which it refuses.
    ``declared_coordinates`` already carried their attributes over.
    """
    indexed = [name for name in coordinates if name in array.xindexes]
    present = [name for name in coordinates if name in array.coords]
    return array.drop_indexes(indexed).drop_vars(present).assign_coords(coordinates)


def detach(array: xr.DataArray, index: TensorComponentIndex) -> xr.DataArray:
    """Remove the declaration and restore default indexes on the component dimensions."""
    result = array.drop_indexes(list(index.axes))
    for name in index.axes:
        if name in result.dims and result.coords[name].dims == (name,):
            result = result.set_xindex(name)
    return result


def _frame_map(
    source: ReferenceFrame, target: ReferenceFrame, transform: SupportsPoints | None
) -> npt.NDArray[np.float64]:
    if source == target:
        return np.eye(len(source.axes))
    if transform is not None:
        # A supplied transform is the caller's statement of this hop; never substitute a
        # derived map for one that does not fit.
        if transform.source != source or transform.target != target:
            raise ValueError(
                f"transform maps {_name(transform.source)} to {_name(transform.target)}, but this "
                f"change needs {_name(source)} to {_name(target)}"
            )
        if not isinstance(transform, SupportsAffine):
            raise ValueError(
                "change_component_basis supports constant (affine) maps only; pointwise maps "
                "arrive with field transforms"
            )
        return np.asarray(transform.matrix, dtype=np.float64)
    try:
        return coordinate_system_change(source, target).matrix
    except ValueError as error:
        raise ValueError(
            f"no map from {source.identifier} to {target.identifier}: {error}. Pass "
            "transform= an affine transform from the one to the other"
        ) from error


def _name(endpoint: object) -> str:
    if isinstance(endpoint, ReferenceFrame):
        return f"frame {endpoint.identifier} in axes {endpoint.axes}"
    return repr(endpoint)


def stale_array_bases(index: TensorComponentIndex, binding: BindingIndex | None) -> bool:
    """Whether a declared ArrayCoordinates basis no longer matches the binding source."""
    source = binding.transform.source if binding is not None else None
    frame = binding.transform.target if binding is not None else None
    return any(
        (isinstance(basis, ArrayCoordinates) and basis != source)
        or (isinstance(basis, FrameField) and basis.base != frame)
        for basis in index.declaration.bases
    )


def refuse_component_geometry(array: xr.DataArray, dims: Sequence[str]) -> None:
    """Refuse a binding whose geometry dimensions are declared component dimensions."""
    index = declaration_of(array, verify=False)
    if index is None:
        return
    overlap = sorted(set(dims) & set(index.declaration.component_dims))
    if overlap:
        raise ValueError(
            f"dimension(s) {overlap} hold declared tensor components and cannot also be geometry "
            "dimensions"
        )


def _binding_matrix(
    transform: SupportsPoints | None, basis: ArrayCoordinates
) -> tuple[npt.NDArray[np.float64], ReferenceFrame]:
    if transform is None or transform.source != basis:
        raise ValueError(
            "an ArrayCoordinates basis must be the source of this array's frame binding; the "
            "binding changed since the components were declared"
        )
    if not isinstance(transform, SupportsAffine):
        raise ValueError(
            "change_component_basis supports affine frame bindings only; pointwise maps arrive "
            "with field transforms"
        )
    frame = transform.target
    assert isinstance(frame, ReferenceFrame)  # a binding maps array coordinates to a frame
    return np.asarray(transform.matrix, dtype=np.float64), frame


def basis_map(
    source: ComponentBasis,
    target: ComponentBasis,
    *,
    binding_transform: SupportsPoints | None,
    transform: SupportsPoints | None,
) -> npt.NDArray[np.float64]:
    """The linear map taking components in ``source`` to components in ``target``.

    With the binding ``B: A -> F`` and a frame map ``C`` toward the target, ``A -> O`` is
    ``C @ B`` and ``S -> A`` is ``B^-1 @ C`` with ``C: S -> F``. Every map is square and
    passes the equilibrated invertibility test: a passive change of basis is a bijection.
    """
    if source == target:
        matrix = np.eye(len(source.axes))
    elif isinstance(source, ArrayCoordinates) and isinstance(target, ReferenceFrame):
        bound, frame = _binding_matrix(binding_transform, source)
        matrix = _frame_map(frame, target, transform) @ bound
    elif isinstance(source, ReferenceFrame) and isinstance(target, ArrayCoordinates):
        bound, frame = _binding_matrix(binding_transform, target)
        if bound.shape[0] != bound.shape[1]:
            raise ValueError("the frame binding's matrix is not square, so it has no inverse")
        matrix = equilibrated_inverse(bound) @ _frame_map(source, frame, transform)
    elif isinstance(source, ReferenceFrame) and isinstance(target, ReferenceFrame):
        matrix = _frame_map(source, target, transform)
    else:
        raise ValueError(
            "components can change only between the array's own ArrayCoordinates and frames; "
            "two different ArrayCoordinates are unrelated"
        )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"a change of basis must be square; the map from {source.axes} to {target.axes} "
            f"has shape {matrix.shape}"
        )
    equilibrated_inverse(matrix)  # refuses singular maps whatever the law
    return matrix


def needs_points(
    sources: Sequence[ComponentBasis],
    targets: Sequence[ComponentBasis],
    transform: SupportsPoints | None,
) -> bool:
    """Whether a change of basis varies by position: a frame field, or a non-affine transform."""
    fields = any(isinstance(basis, FrameField) for basis in (*sources, *targets))
    return fields or (transform is not None and not isinstance(transform, SupportsAffine))


def _to_frame(
    basis: ComponentBasis,
    frame: ReferenceFrame,
    points: npt.NDArray[np.float64],
    binding_transform: SupportsPoints | None,
    transform: SupportsPoints | None,
) -> npt.NDArray[np.float64]:
    """Components in ``basis`` to components in ``frame``'s coordinate basis, at ``points``."""
    if isinstance(basis, FrameField):
        _require_base(basis, frame)
        return np.asarray(basis.matrix(points), dtype=np.float64)
    if isinstance(basis, ArrayCoordinates):
        bound, bound_frame = _binding_matrix(binding_transform, basis)
        if bound_frame != frame:
            raise ValueError("the binding's frame is not the frame components are mapped through")
        return bound
    if basis == frame:
        return np.eye(len(frame.axes))
    if transform is not None and transform.source == basis and transform.target == frame:
        if isinstance(transform, SupportsAffine):
            return np.asarray(transform.matrix, dtype=np.float64)
        # Components move from the transform's source frame, so its Jacobian is evaluated at
        # each sample's location there.
        if not isinstance(transform, SupportsInverse):
            raise ValueError(
                "a pointwise transform toward the array's frame needs an inverse to locate the "
                "samples in its source frame"
            )
        located = check_transform(transform.inverse()).transform_point(points)
        return np.asarray(_jacobian(transform).jacobian(at=located), dtype=np.float64)
    return _frame_map(basis, frame, None)


def _from_frame(
    basis: ComponentBasis,
    frame: ReferenceFrame,
    points: npt.NDArray[np.float64],
    binding_transform: SupportsPoints | None,
    transform: SupportsPoints | None,
) -> npt.NDArray[np.float64]:
    """Components in ``frame``'s coordinate basis to components in ``basis``, at ``points``."""
    if isinstance(basis, FrameField):
        _require_base(basis, frame)
        return np.asarray(basis.inverse_matrix(points), dtype=np.float64)
    if isinstance(basis, ArrayCoordinates):
        bound, bound_frame = _binding_matrix(binding_transform, basis)
        if bound_frame != frame or bound.shape[0] != bound.shape[1]:
            raise ValueError("the binding's matrix does not map this frame square and invertibly")
        return equilibrated_inverse(bound)
    if basis == frame:
        return np.eye(len(frame.axes))
    if transform is not None and transform.source == frame and transform.target == basis:
        if isinstance(transform, SupportsAffine):
            return np.asarray(transform.matrix, dtype=np.float64)
        return np.asarray(_jacobian(transform).jacobian(at=points), dtype=np.float64)
    return _frame_map(frame, basis, None)


def _jacobian(transform: SupportsPoints) -> SupportsJacobian:
    if not isinstance(transform, SupportsJacobian):
        raise ValueError("a pointwise change of basis needs the transform's Jacobian")
    return transform


def _require_base(field: FrameField, frame: ReferenceFrame) -> None:
    if field.base != frame:
        raise ValueError(
            f"a frame field's base must be the array's frame {frame.identifier} in its own "
            f"coordinate system {frame.axes}; this field is over {field.base.axes}"
        )


def point_basis_map(
    source: ComponentBasis,
    target: ComponentBasis,
    *,
    frame: ReferenceFrame,
    points: npt.NDArray[np.float64],
    binding_transform: SupportsPoints | None,
    transform: SupportsPoints | None,
) -> npt.NDArray[np.float64]:
    """The map from ``source`` to ``target`` components at ``points`` of ``frame``.

    Assembled through ``frame``'s coordinate basis: to it from ``source``, then from it to
    ``target``, each piece constant or evaluated at the points (``(*batch, n, n)``). A
    pointwise ``transform`` must have ``frame`` as one endpoint, since only its points are
    known. East/north/up to (or from) its own Earth-centred chart maps by the field's
    directions directly, which exist at the poles where the chart basis does not.
    """
    if source == target:
        return np.eye(len(source.axes))
    if isinstance(source, EastNorthUp) and source.matches(transform):
        assert transform is not None
        if transform.target == target and source.base == frame:
            return source.directions(points)
    if isinstance(target, EastNorthUp) and target.matches(transform):
        assert transform is not None
        if transform.source == source and target.base == frame:
            return np.swapaxes(target.directions(points), -1, -2)
    if (
        transform is not None
        and not isinstance(transform, SupportsAffine)
        and frame not in (transform.source, transform.target)
    ):
        raise ValueError(
            "a pointwise transform must start or end at the array's frame, whose points are "
            "the only ones known"
        )
    inward = _to_frame(source, frame, points, binding_transform, transform)
    outward = _from_frame(target, frame, points, binding_transform, transform)
    return np.asarray(np.matmul(outward, inward), dtype=np.float64)


def change_basis(
    array: xr.DataArray,
    index: TensorComponentIndex,
    targets: Sequence[ComponentBasis],
    *,
    binding: BindingIndex | None,
    transform: SupportsPoints | None,
    units: str | None,
) -> xr.DataArray:
    """Re-express the declared components in ``targets``, one basis per slot."""
    declaration = index.declaration
    tensor_type = declaration.tensor_type
    packing = declaration.packing
    new_declaration = TensorComponentDeclaration(
        tensor_type, declaration.component_dims, tuple(targets), packing=packing
    )
    binding_transform = binding.transform if binding is not None else None
    dims = list(declaration.component_dims)
    if needs_points(declaration.bases, targets, transform):
        maps, geometry_dims = _sample_maps(
            array, declaration.bases, targets, binding=binding, transform=transform
        )
        core = [*geometry_dims, *dims]
    else:
        maps = [
            basis_map(source, target, binding_transform=binding_transform, transform=transform)
            for source, target in zip(declaration.bases, targets, strict=True)
        ]
        core = dims
    check_common_map(tensor_type, maps)
    _check_units(declaration.bases, targets, units)
    result = _apply_over(
        array, lambda values: transform_declared_components(values, declaration, maps), core
    )
    result = result.drop_indexes(list(index.axes)).drop_vars(list(index.axes))
    labelled = result.assign_coords(
        {
            dim: (dim, np.asarray(new_declaration.labels(position)))
            for position, dim in enumerate(dims)
        }
    )
    for dim in dims:
        if dim in array.coords and array.coords[dim].attrs:
            labelled.coords[dim].attrs.update(array.coords[dim].attrs)
    coordinates = declared_coordinates(labelled.drop_indexes(dims), new_declaration, binding)
    typed = attach(labelled, coordinates)
    if units is not None:
        typed.attrs["units"] = units
    return typed


def _sample_maps(
    array: xr.DataArray,
    sources: Sequence[ComponentBasis],
    targets: Sequence[ComponentBasis],
    *,
    binding: BindingIndex | None,
    transform: SupportsPoints | None,
) -> tuple[list[npt.NDArray[np.float64]], tuple[str, ...]]:
    """Per-slot maps at every sample of a framed array, batched over its geometry dims."""
    if binding is None:
        raise ValueError(
            "a change of basis that varies by position needs the samples' points; frame the "
            "array first"
        )
    frame = binding.transform.target
    assert isinstance(frame, ReferenceFrame)  # a binding maps array coordinates to a frame
    geometry = array.rf.geometry
    points = np.asarray(geometry.points(axis_dim="__axis__").values, dtype=np.float64)
    maps = []
    for source, target in zip(sources, targets, strict=True):

        def evaluate(
            at: npt.NDArray[np.float64],
            source: ComponentBasis = source,
            target: ComponentBasis = target,
        ) -> npt.NDArray[np.float64]:
            matrix = point_basis_map(
                source,
                target,
                frame=frame,
                points=at,
                binding_transform=binding.transform,
                transform=transform,
            )
            if matrix.shape[-1] == matrix.shape[-2]:
                # A passive change of basis is a bijection at every sample; a user field could
                # return a finite but singular matrix.
                equilibrated_inverse(matrix)
            return matrix

        try:
            matrix = evaluate(points)
        except ValueError as error:
            sample = _first_failing_sample(evaluate, points, geometry.dims)
            if sample is None:
                raise
            raise ValueError(f"at sample {sample}: {error}") from error
        if matrix.ndim == 2:
            matrix = np.broadcast_to(matrix, (*points.shape[:-1], *matrix.shape))
        if matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(
                f"a change of basis must be square; the map from {source.axes} to {target.axes} "
                f"has shape {matrix.shape[-2:]}"
            )
        maps.append(matrix)
    return maps, tuple(geometry.dims)


def _first_failing_sample(
    evaluate: Any, points: npt.NDArray[np.float64], dims: Sequence[str]
) -> dict[str, int] | None:
    """The first sample whose evaluation raises, by bisection over the flattened samples."""
    flat = points.reshape(-1, points.shape[-1])

    def fails(stop: int) -> bool:
        try:
            evaluate(flat[:stop])
        except ValueError:
            return True
        return False

    if not fails(len(flat)):
        return None
    low, high = 0, len(flat)  # the prefix of length high fails, of length low does not
    while high - low > 1:
        middle = (low + high) // 2
        if fails(middle):
            high = middle
        else:
            low = middle
    position = np.unravel_index(high - 1, points.shape[:-1])
    return {str(dim): int(index) for dim, index in zip(dims, position, strict=True)}


def check_common_map(tensor_type: TensorType, maps: Sequence[npt.NDArray[np.float64]]) -> None:
    """Refuse per-slot maps that differ for a type whose determinant factor needs one map."""
    common = all(np.array_equal(matrix, maps[0]) for matrix in maps)
    if not common:
        refuse_separate_maps(tensor_type)


def refuse_separate_maps(tensor_type: TensorType) -> None:
    """Refuse a pseudotensor or density whose slots would change by different maps."""
    if tensor_type.reflection_parity == "odd" or tensor_type.determinant_exponent:
        raise ValueError(
            "a pseudotensor or density needs one map for det J, but its slots change basis by "
            "different maps; give every slot the same basis"
        )


def transform_declared_components(
    values: npt.NDArray[Any],
    declaration: TensorComponentDeclaration,
    maps: Sequence[npt.NDArray[np.float64]],
) -> npt.NDArray[Any]:
    """Apply one map per slot to components on the trailing axes (one axis when packed).

    Each map is ``(n, n)`` or a batch ``(*batch, n, n)`` aligned with the axes just before the
    component axes. Equal maps apply as one, which pseudotensors and densities require.
    """
    tensor_type = declaration.tensor_type
    packing = declaration.packing
    if packing is not None:
        # Never transform packed components directly: unpack, transform, repack. The result is
        # symmetric up to rounding, which the repack averages away.
        matrices = transform_tensor_components(packing.unpack(values), maps[0], tensor_type)
        return packing.pack_symmetric_part(matrices)
    if all(matrix is maps[0] or np.array_equal(matrix, maps[0]) for matrix in maps):
        return transform_tensor_components(values, maps[0], tensor_type)
    order = tensor_type.order
    for slot, matrix in enumerate(maps):
        slot_type = TensorType(
            (tensor_type.index_variance[slot],),
            requires_orthogonal_map=tensor_type.requires_orthogonal_map,
        )
        # The kernel aligns a map batch with the values minus this slot's axis, which still
        # include the other component axes: give the batch unit axes for them.
        batch = matrix.shape[:-2]
        if batch:
            matrix = matrix.reshape(*batch, *(1,) * (order - 1), *matrix.shape[-2:])
        values = transform_tensor_components(
            values, matrix, slot_type, component_axes=(values.ndim - order + slot,)
        )
    return values


def change_scalar_basis(
    array: xr.DataArray,
    tensor_type: TensorType,
    source: ComponentBasis,
    target: ComponentBasis,
    *,
    binding: BindingIndex | None,
    transform: SupportsPoints | None,
    units: str | None,
) -> xr.DataArray:
    """Re-express an undeclared scalar (density or pseudoscalar) under a change of basis."""
    matrix = basis_map(
        source,
        target,
        binding_transform=binding.transform if binding is not None else None,
        transform=transform,
    )
    _check_units((source,), (target,), units)
    result = _apply_over(
        array, lambda values: transform_tensor_components(values, matrix, tensor_type), []
    )
    if units is not None:
        result.attrs["units"] = units
    return result


def _check_units(
    sources: Sequence[ComponentBasis], targets: Sequence[ComponentBasis], units: str | None
) -> None:
    changed = any(
        source.units != target.units for source, target in zip(sources, targets, strict=True)
    )
    if changed and units is None:
        raise ValueError(
            "the basis units change, so the components' units change too; pass units= the "
            "new attrs['units'] (unit strings are never parsed)"
        )
    if not changed and units is not None:
        raise ValueError("units= is only accepted when the basis units change")
    if units is not None and not isinstance(units, str):
        raise TypeError(f"units must be a string, got {type(units).__name__}")


def unpack_components(
    array: xr.DataArray,
    index: TensorComponentIndex,
    component_dims: Sequence[str],
    *,
    binding: BindingIndex | None,
) -> xr.DataArray:
    """Replace a packed component dimension by two, declared unpacked in the same bases."""
    declaration = index.declaration
    packing = declaration.packing
    if packing is None:
        raise ValueError("the components are not packed; there is nothing to unpack")
    unpacked = TensorComponentDeclaration(
        declaration.tensor_type, component_dims, declaration.bases
    )
    (packed_dim,) = declaration.component_dims
    _refuse_taken_names(array, unpacked.component_dims, replaced=())
    n = packing.dimension
    plain = _without_components(array, index)
    result = _apply_over(
        plain,
        packing.unpack,
        [packed_dim],
        output_dims=list(unpacked.component_dims),
        output_sizes=dict.fromkeys(unpacked.component_dims, n),
    )
    at = array.dims.index(packed_dim)
    order = [*array.dims[:at], *unpacked.component_dims, *array.dims[at + 1 :]]
    result = result.transpose(*order)
    return attach(result, declared_coordinates(result, unpacked, binding))


def pack_components(
    array: xr.DataArray,
    index: TensorComponentIndex,
    packing: SymmetricPacking,
    component_dim: str,
    *,
    binding: BindingIndex | None,
) -> xr.DataArray:
    """Replace the two dimensions of a symmetric declaration by one packed dimension."""
    declaration = index.declaration
    if declaration.packing is not None:
        raise ValueError("the components are already packed; unpack them first to repack")
    if not isinstance(packing, SymmetricPacking):
        raise TypeError(f"packing must be a SymmetricPacking, got {type(packing).__name__}")
    packed = TensorComponentDeclaration(
        declaration.tensor_type, (component_dim,), declaration.bases, packing=packing
    )
    rows, columns = declaration.component_dims
    _refuse_taken_names(array, packed.component_dims, replaced=(rows, columns))
    plain = _without_components(array, index)
    result = _apply_over(
        plain,
        packing.pack,
        [rows, columns],
        output_dims=[component_dim],
        output_sizes={component_dim: packing.component_count},
    )
    first = min(array.dims.index(rows), array.dims.index(columns))
    kept = [dim for dim in array.dims if dim not in (rows, columns)]
    order = [*kept[:first], component_dim, *kept[first:]]
    result = result.transpose(*order)
    return attach(result, declared_coordinates(result, packed, binding))


def _refuse_taken_names(
    array: xr.DataArray, names: Sequence[str], *, replaced: Sequence[str]
) -> None:
    taken = sorted(
        name
        for name in names
        if name not in replaced and (name in array.dims or name in array.coords)
    )
    if taken:
        raise ValueError(
            f"{taken} already name a dimension or coordinate of the array; choose new component "
            "dimension names"
        )


def _without_components(array: xr.DataArray, index: TensorComponentIndex) -> xr.DataArray:
    """The array without its component label coordinates or their index."""
    names = list(index.axes)
    return array.drop_indexes(names).drop_vars(names)


def _apply_over(
    array: xr.DataArray,
    function: Any,
    dims: Sequence[str],
    *,
    output_dims: Sequence[str] | None = None,
    output_sizes: Mapping[str, int] | None = None,
) -> xr.DataArray:
    if array.dtype.kind not in "iufc":
        raise TypeError(
            f"components must hold real integer, floating or complex numbers, got dtype "
            f"{array.dtype}"
        )
    dtype = np.result_type(array.dtype, np.float64)
    outputs = list(dims) if output_dims is None else list(output_dims)
    result = xr.apply_ufunc(
        function,
        array,
        input_core_dims=[list(dims)],
        output_core_dims=[outputs],
        dask="parallelized",
        output_dtypes=[dtype],
        dask_gufunc_kwargs={"output_sizes": dict(output_sizes)} if output_sizes else None,
        # Packing may reuse a replaced dimension's name at a new size.
        exclude_dims=set(dims) & set(outputs) if output_dims is not None else frozenset(),
        keep_attrs=True,
    )
    if output_dims is not None:
        return cast(xr.DataArray, result)
    transposed: xr.DataArray = result.transpose(*array.dims)
    return transposed
