"""Private xarray index that carries a transform with its source coordinates."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr
from xarray.core.indexing import IndexSelResult
from xarray.indexes import PandasIndex

from ._affine import AffineTransform
from ._array_coordinates import ArrayCoordinates
from ._frame import ReferenceFrame
from ._geometry import check_coordinate_unit
from ._transform import SupportsPoints


class BindingIndex(xr.Index):
    """Own every source axis as either a labelled axis or a retained scalar."""

    def __init__(
        self,
        axes: Mapping[str, PandasIndex],
        fixed: Mapping[str, xr.Variable],
        transform: SupportsPoints,
        dims: tuple[str, ...],
    ) -> None:
        self.axes = dict(axes)
        self.fixed = dict(fixed)
        self.transform = transform
        self.dims = dims
        self.units = dict(zip(transform.source.axes, transform.source.units, strict=True))

    @classmethod
    def from_variables(
        cls, variables: Mapping[Any, xr.Variable], *, options: Mapping[str, Any]
    ) -> BindingIndex:
        raise ValueError("BindingIndex cannot be constructed with set_xindex; use rf.frame()")

    def create_variables(
        self, variables: Mapping[Any, xr.Variable] | None = None
    ) -> dict[str, xr.Variable]:
        created: dict[str, xr.Variable] = {}
        for name, index in self.axes.items():
            source = (
                {name: variables[name]} if variables is not None and name in variables else None
            )
            created.update(index.create_variables(source))
        for name, variable in self.fixed.items():
            # A scalar selection rebuilds the fixed term from labels alone; the coordinate
            # variable xarray passes back still carries its attributes, such as units.
            if variables is not None and name in variables and not variable.attrs:
                variable = variable.copy(deep=False)
                variable.attrs = dict(variables[name].attrs)
            created[name] = variable
        return created

    def sel(
        self, labels: dict[Any, Any], method: str | None = None, tolerance: Any = None
    ) -> IndexSelResult:
        indexers: dict[Any, Any] = {}
        for name, label in labels.items():
            if name not in self.axes:
                raise KeyError(f"source axis {name!r} is fixed and cannot be selected by label")
            indexers.update(
                self.axes[name].sel({name: label}, method=method, tolerance=tolerance).dim_indexers
            )
        return IndexSelResult(indexers)

    def isel(self, indexers: Mapping[Any, Any]) -> BindingIndex:
        axes: dict[str, PandasIndex] = {}
        fixed = dict(self.fixed)
        dims = list(self.dims)
        for name, index in self.axes.items():
            dim = index.dim
            if dim not in indexers:
                axes[name] = index
                continue
            indexer = indexers[dim]
            if isinstance(indexer, xr.Variable) and indexer.dims != (dim,):
                raise ValueError(
                    "vectorized indexing changes geometry dimensions; call rf.unframe()"
                )
            reduced = index.isel({dim: indexer})
            if reduced is not None:
                axes[name] = reduced
            elif not isinstance(indexer, slice) and np.ndim(indexer) == 0:
                position = indexer.item() if hasattr(indexer, "item") else indexer
                fixed[name] = xr.Variable((), index.index[position])
                if dim in dims:
                    dims.remove(str(dim))
            else:
                raise ValueError("indexer changes geometry dimensions; call rf.unframe()")
        return type(self)(axes, fixed, self.transform, tuple(dims))

    def equals(self, other: xr.Index, *, exclude: frozenset[Hashable] | None = None) -> bool:
        if not isinstance(other, BindingIndex):
            return False
        if not self._same_binding(other):
            return False
        return all(
            index.equals(other.axes[name], exclude=exclude)
            for name, index in self.axes.items()
            if exclude is None or index.dim not in exclude
        )

    def _same_binding(self, other: BindingIndex) -> bool:
        return (
            self.transform == other.transform
            and self.dims == other.dims
            and self.axes.keys() == other.axes.keys()
            and self.fixed.keys() == other.fixed.keys()
            and all(
                self.fixed[name].equals(other.fixed[name])  # type: ignore[no-untyped-call]
                for name in self.fixed
            )
        )

    def _require_compatible(self, other: xr.Index) -> BindingIndex:
        if not isinstance(other, BindingIndex):
            raise ValueError(
                "incompatible framed coordinate mappings; unframe or resample an operand"
            )
        if not self._same_binding(other):
            raise ValueError(f"incompatible framed coordinate mappings: {self._difference(other)}")
        return other

    def _difference(self, other: BindingIndex) -> str:
        """Say why two bindings cannot combine, naming the corrective action."""
        mine = getattr(self.transform, "target", None)
        theirs = getattr(other.transform, "target", None)
        if (
            isinstance(mine, ReferenceFrame)
            and isinstance(theirs, ReferenceFrame)
            and not mine.is_equivalent_frame(theirs)
        ):
            return (
                f"the operands are in different reference frames ({mine.identifier[0]}:"
                f"{mine.identifier[1]} and {theirs.identifier[0]}:{theirs.identifier[1]}); "
                "resample one onto the other with a transform between the frames, or use "
                "rf.assume_frame if they are the same space"
            )
        return (
            "the operands sample the same frame on different grids; resample one onto the "
            "other with rf.resample_to"
        )

    def join(self, other: xr.Index, how: str = "inner") -> BindingIndex:
        compatible = self._require_compatible(other)
        axes = {
            name: index.join(compatible.axes[name], how=how) for name, index in self.axes.items()
        }
        return type(self)(axes, self.fixed, self.transform, self.dims)

    def join_overlapping(
        self,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: str,
    ) -> tuple[BindingIndex, dict[Hashable, xr.Index]]:
        """Join plain indexes on source axes while retaining this binding's ownership.

        Args:
            other_indexes: Indexes keyed by their overlapping source coordinate.
            other_variables: Coordinate variables belonging to those indexes.
            how: Alignment join, relative to this binding as the left operand.

        Returns:
            The joined binding and target indexes for each overlapping coordinate.

        Raises:
            ValueError: If an overlap covers a fixed term or uses an unsupported index.
        """
        axes = dict(self.axes)
        targets: dict[Hashable, xr.Index] = {}
        for name, other in other_indexes.items():
            if name in self.fixed:
                raise ValueError(f"fixed source coordinate {name!r} cannot be joined")
            if name not in axes or not isinstance(other, PandasIndex):
                raise ValueError(f"source coordinate {name!r} requires a PandasIndex")
            check_coordinate_unit(other_variables[name], name=str(name), declared=self.units[name])
            axis = axes[name]
            if how == "exact":
                if not axis.equals(other):
                    raise ValueError(f"coordinate {name!r} differs for join='exact'")
                target = axis
            elif how == "left":
                target = axis
            elif how == "right":
                target = other
            else:
                target = axis.join(other, how=how)
            axes[name] = target
            targets[name] = target
        return type(self)(axes, self.fixed, self.transform, self.dims), targets

    def check_unindexed_coord_conflicts(self, variables: Mapping[Hashable, xr.Variable]) -> None:
        """Reject discarded plain coordinates that contradict bound source coordinates.

        Args:
            variables: Same-name, same-dimension unindexed coordinates to replace.

        Raises:
            ValueError: If a coordinate has different values or source units.
            TypeError: If a coordinate's ``units`` attribute is not a string.
        """
        owned = self.create_variables()
        for name, variable in variables.items():
            if name not in owned or not variable.equals(owned[name]):  # type: ignore[no-untyped-call]
                raise ValueError(f"conflicting unindexed coordinate {name!r}")
            check_coordinate_unit(variable, name=str(name), declared=self.units[name])

    def check_override(self, other: xr.Index) -> None:
        """Refuse replacement of this binding under ``join='override'``.

        Raises:
            ValueError: Always, because override discards coordinate evidence.
        """
        raise ValueError("join='override' is unsupported for framed coordinates")

    def check_stack(self, dims: Sequence[Hashable]) -> None:
        """Refuse stacking dimensions of this binding's source coordinates.

        Raises:
            ValueError: Always, because native stacking changes geometry dimensions.
        """
        raise ValueError("cannot stack geometry dimensions; call rf.unframe()")

    def check_pad(self, pad_width: Mapping[Hashable, int | tuple[int, int]]) -> None:
        """Refuse padding dimensions of this binding's source coordinates.

        Raises:
            ValueError: Always, because native padding does not declare sample positions.
        """
        raise ValueError("cannot pad geometry dimensions; call rf.unframe()")

    def check_coarsen(self, windows: Mapping[Hashable, int]) -> None:
        """Refuse coarsening dimensions of this binding's source coordinates.

        Raises:
            ValueError: Always, because native coarsening has no geometry aggregation policy.
        """
        raise ValueError("cannot coarsen geometry dimensions; call rf.unframe()")

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

    def roll(self, shifts: Mapping[Any, int]) -> BindingIndex:
        """Roll owned coordinate labels with their corresponding geometry axes."""
        axes = {
            name: index.roll({index.dim: shifts[index.dim]}) if index.dim in shifts else index
            for name, index in self.axes.items()
        }
        return type(self)(axes, self.fixed, self.transform, self.dims)

    def rename(self, name_dict: Mapping[Any, Any], dims_dict: Mapping[Any, Any]) -> BindingIndex:
        names = {name: name_dict.get(name, name) for name in self.transform.source.axes}
        new_dims = tuple(dims_dict.get(dim, dim) for dim in self.dims)
        if all(name == renamed for name, renamed in names.items()) and new_dims == self.dims:
            return self
        transform = self.transform
        if any(name != renamed for name, renamed in names.items()):
            if not isinstance(transform, AffineTransform):
                raise ValueError("this transform cannot rename source axes; call rf.unframe()")
            source = transform.source
            if not isinstance(source, ArrayCoordinates):
                raise ValueError("binding source must be ArrayCoordinates")
            transform = transform.with_endpoints(
                source=ArrayCoordinates(
                    tuple(names.values()),
                    source.units,
                    axis_types=source.axis_types,
                    sample_offset=source.sample_offset,
                )
            )
        axes = {
            names[name]: index.rename(name_dict, dims_dict)  # type: ignore[no-untyped-call]
            for name, index in self.axes.items()
        }
        fixed = {names[name]: variable for name, variable in self.fixed.items()}
        return type(self)(axes, fixed, transform, new_dims)

    def swap_dims(self, dims_dict: Mapping[Any, Any]) -> BindingIndex:
        # Opt-in xarray hook: keep the binding whole, with the source coordinate of a swapped
        # dimension now a non-dimension coordinate along the replacement dimension.
        return self.rename({}, dims_dict)

    @classmethod
    def concat(cls, indexes: Any, dim: Any, positions: Any = None) -> BindingIndex:
        raise ValueError("concatenation of framed arrays is unsupported; call rf.unframe()")

    def __repr__(self) -> str:
        return f"BindingIndex(axes={tuple(self.axes)}, fixed={tuple(self.fixed)}, dims={self.dims})"
