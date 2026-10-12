"""Private xarray index carrying a tensor component declaration, and basis changes through it.

The declaration is advisory with respect to values: the index is consulted by structural
operations (selection, alignment, concatenation, renaming) and by xarrayrf, but an xarray index
lives with the coordinates and never sees element-wise operations or payload replacement. The
measured boundary is in ``docs/dev/architecture/value_geometry_stage2_plan.md``.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from xarray.core.indexing import IndexSelResult
from xarray.indexes import PandasIndex

from ._affine import equilibrated_inverse
from ._array_coordinates import ArrayCoordinates
from ._binding import BindingIndex
from ._frame import ReferenceFrame
from ._orientation import coordinate_system_change
from ._tensor_component_declaration import ComponentBasis, TensorComponentDeclaration
from ._tensor_type import TensorType, transform_tensor_components
from ._transform import SupportsAffine, SupportsPoints

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

    def _slot_of(self, name: str) -> int:
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
                self._slot_of(name)
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
            self.declaration.tensor_type, dims, self.declaration.bases
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
    for slot, dim in enumerate(declaration.component_dims):
        basis = declaration.bases[slot]
        labels = declaration.labels(slot)
        if dim not in array.dims:
            raise ValueError(f"component dimension {dim!r} is not a dimension of the array")
        if dim in geometry_dims:
            raise ValueError(
                f"component dimension {dim!r} is a geometry dimension of the frame binding; "
                "components need a dimension of their own"
            )
        if array.sizes[dim] != len(labels):
            raise ValueError(
                f"component dimension {dim!r} has length {array.sizes[dim]}, but its basis has "
                f"{len(labels)} axes"
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
    return any(
        isinstance(basis, ArrayCoordinates) and basis != source for basis in index.declaration.bases
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
    binding: BindingIndex | None, basis: ArrayCoordinates
) -> tuple[npt.NDArray[np.float64], ReferenceFrame]:
    if binding is None or binding.transform.source != basis:
        raise ValueError(
            "an ArrayCoordinates basis must be the source of this array's frame binding; the "
            "binding changed since the components were declared"
        )
    transform = binding.transform
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
    binding: BindingIndex | None,
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
        bound, frame = _binding_matrix(binding, source)
        matrix = _frame_map(frame, target, transform) @ bound
    elif isinstance(source, ReferenceFrame) and isinstance(target, ArrayCoordinates):
        bound, frame = _binding_matrix(binding, target)
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
    new_declaration = TensorComponentDeclaration(
        tensor_type, declaration.component_dims, tuple(targets)
    )
    maps = [
        basis_map(source, target, binding=binding, transform=transform)
        for source, target in zip(declaration.bases, targets, strict=True)
    ]
    common = all(np.array_equal(matrix, maps[0]) for matrix in maps)
    if not common and (tensor_type.reflection_parity == "odd" or tensor_type.determinant_exponent):
        raise ValueError(
            "a pseudotensor or density needs one map for det J, but its slots change basis by "
            "different maps; give every slot the same basis"
        )
    _check_units(declaration.bases, targets, units)
    dims = list(declaration.component_dims)

    def apply(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if common:
            return transform_tensor_components(values, maps[0], tensor_type)
        order = tensor_type.order
        for slot, matrix in enumerate(maps):
            slot_type = TensorType(
                (tensor_type.index_variance[slot],),
                requires_orthogonal_map=tensor_type.requires_orthogonal_map,
            )
            values = transform_tensor_components(
                values, matrix, slot_type, component_axes=(values.ndim - order + slot,)
            )
        return values

    result = _apply_over(array, apply, dims)
    result = result.drop_indexes(list(index.axes)).drop_vars(list(index.axes))
    labelled = result.assign_coords(
        {dim: (dim, np.asarray(target.axes)) for dim, target in zip(dims, targets, strict=True)}
    )
    for dim in dims:
        if dim in array.coords and array.coords[dim].attrs:
            labelled.coords[dim].attrs.update(array.coords[dim].attrs)
    coordinates = declared_coordinates(labelled.drop_indexes(dims), new_declaration, binding)
    typed = attach(labelled, coordinates)
    if units is not None:
        typed.attrs["units"] = units
    return typed


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
    matrix = basis_map(source, target, binding=binding, transform=transform)
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


def _apply_over(array: xr.DataArray, function: Any, dims: Sequence[str]) -> xr.DataArray:
    if array.dtype.kind not in "iufc":
        raise TypeError(
            f"components must hold real integer, floating or complex numbers, got dtype "
            f"{array.dtype}"
        )
    dtype = np.result_type(array.dtype, np.float64)
    result = xr.apply_ufunc(
        function,
        array,
        input_core_dims=[list(dims)],
        output_core_dims=[list(dims)],
        dask="parallelized",
        output_dtypes=[dtype],
        keep_attrs=True,
    )
    transposed: xr.DataArray = result.transpose(*array.dims)
    return transposed
