"""A development-only index that owns ``("y", "x", MARKER)`` jointly.

The scalar guard in ``scalar_binding_probe`` is dimensionless, so ``Index.isel`` never fires for
it and the declaration survives selection by being ignored. This candidate instead owns the
mapping-input coordinates, the arrangement ``docs/design.md`` requires to be measured against
ordinary ``PandasIndex`` label and ``method="nearest"`` selection before adoption.

Every one-dimensional label operation delegates to a public ``xarray.indexes.PandasIndex`` per
axis; no label-search arithmetic lives here. The only original logic is bookkeeping: which axes
are still one-dimensional, which collapsed to a retained fixed term, and how the declaration's
input names follow a rename. It is not a production carrier, wrapper or accessor, and it does
not replace the scalar baseline, which stays available and unmodified.
"""

import json

import numpy as np
import xarray as xr
from scalar_binding_probe import MARKER, framed
from xarray.core.indexing import IndexSelResult
from xarray.indexes import PandasIndex

# ``IndexSelResult`` has no public export in xarray 2026.7.0. Rasterix imports it from the same
# private module, so a custom index cannot implement ``sel`` without it. A recorded reuse cost.


class JointFrameIndex(xr.Index):
    """Own the mapping-input coordinates and the declaration in one index.

    Args:
        axes: Per-axis ``PandasIndex`` objects keyed by coordinate name.
        fixed: Zero-dimensional coordinate variables retained by a scalar selection.
        marker: Name of the declaration coordinate.
        declaration: The zero-dimensional declaration variable.
    """

    def __init__(self, axes, fixed, marker, declaration):
        self._axes = dict(axes)
        self._fixed = dict(fixed)
        self._marker = marker
        self._declaration = declaration

    @classmethod
    def from_variables(cls, variables, *, options):
        marker = options.get("marker", MARKER)
        axes, fixed, declaration = {}, {}, None
        for name, var in variables.items():
            if name == marker:
                declaration = var
            elif var.ndim == 1:
                axes[name] = PandasIndex.from_variables({name: var}, options={})
            elif var.ndim == 0:
                fixed[name] = var
            else:
                raise ValueError(f"mapping input {name!r} must be 0- or 1-dimensional")
        if declaration is None:
            raise ValueError(f"joint ownership requires the declaration coordinate {marker!r}")
        if not axes and not fixed:
            raise ValueError("joint ownership requires at least one mapping-input coordinate")
        return cls(axes, fixed, marker, declaration)

    def _replace(self, axes, fixed=None, marker=None, declaration=None):
        return type(self)(
            axes,
            self._fixed if fixed is None else fixed,
            self._marker if marker is None else marker,
            self._declaration if declaration is None else declaration,
        )

    def create_variables(self, variables=None):
        created = {}
        for name, index in self._axes.items():
            source = {name: variables[name]} if variables and name in variables else None
            created.update(index.create_variables(source))
        created.update(self._fixed)
        created[self._marker] = self._declaration
        return created

    def sel(self, labels, method=None, tolerance=None):
        """Delegate each axis label query to that axis's ``PandasIndex``."""
        dim_indexers = {}
        for name, label in labels.items():
            if name == self._marker:
                raise KeyError(f"{name!r} is a declaration, not a selectable label")
            if name not in self._axes:
                raise KeyError(f"{name!r} was fixed by an earlier selection and cannot be sliced")
            result = self._axes[name].sel({name: label}, method=method, tolerance=tolerance)
            dim_indexers.update(result.dim_indexers)
        return IndexSelResult(dim_indexers)

    def isel(self, indexers):
        """Reconstruct per axis, keeping a scalar selection as a retained fixed term."""
        axes, fixed = {}, dict(self._fixed)
        for name, index in self._axes.items():
            if index.dim not in indexers:
                axes[name] = index
                continue
            indexer = indexers[index.dim]
            reduced = index.isel({index.dim: indexer})
            if reduced is not None:
                axes[name] = reduced
            elif not isinstance(indexer, slice) and np.ndim(indexer) == 0:
                position = indexer.item() if hasattr(indexer, "item") else indexer
                fixed[name] = xr.Variable((), index.index[position])
            else:
                # An indexer that introduces a new dimension leaves no 1-D axis to rebuild
                # and nothing here reconstructs one. Returning None drops the index;
                # the declaration may survive unguarded, an observed contract failure.
                return None
        return self._replace(axes, fixed=fixed)

    def equals(self, other, *, exclude=None):
        excluded = frozenset() if exclude is None else exclude
        if not isinstance(other, JointFrameIndex):
            return False
        if self._declaration.item() != other._declaration.item():
            return False
        if set(self._axes) != set(other._axes) or set(self._fixed) != set(other._fixed):
            return False
        if any(var.item() != other._fixed[name].item() for name, var in self._fixed.items()):
            return False
        return all(
            index.equals(other._axes[name], exclude=exclude)
            for name, index in self._axes.items()
            if index.dim not in excluded
        )

    def _require_same_declaration(self, other):
        if not isinstance(other, JointFrameIndex):
            raise ValueError("incompatible coordinate mappings: not a joint declaration")
        if self._declaration.item() != other._declaration.item():
            raise ValueError("incompatible coordinate mappings")
        if set(self._axes) != set(other._axes) or set(self._fixed) != set(other._fixed):
            raise ValueError("incompatible coordinate mappings: different retained inputs")
        if any(var.item() != other._fixed[name].item() for name, var in self._fixed.items()):
            raise ValueError("incompatible coordinate mappings: different fixed terms")

    def join(self, other, how="inner"):
        self._require_same_declaration(other)
        axes = {name: idx.join(other._axes[name], how=how) for name, idx in self._axes.items()}
        return self._replace(axes)

    def reindex_like(self, other, method=None, tolerance=None):
        self._require_same_declaration(other)
        indexers = {}
        for name, index in self._axes.items():
            indexers.update(
                index.reindex_like(other._axes[name], method=method, tolerance=tolerance)
            )
        return indexers

    def rename(self, name_dict, dims_dict):
        owned = set(self._axes) | set(self._fixed) | {self._marker}
        owned |= {index.dim for index in self._axes.values()}
        if not owned & (set(name_dict) | set(dims_dict)):
            return self
        axes = {
            name_dict.get(name, name): index.rename(name_dict, dims_dict)
            for name, index in self._axes.items()
        }
        fixed = {name_dict.get(name, name): var for name, var in self._fixed.items()}
        payload = json.loads(self._declaration.item())
        payload["inputs"] = [name_dict.get(name, name) for name in payload["inputs"]]
        declaration = xr.Variable((), json.dumps(payload, sort_keys=True))
        marker = name_dict.get(self._marker, self._marker)
        return self._replace(axes, fixed=fixed, marker=marker, declaration=declaration)

    def __repr__(self):
        return f"JointFrameIndex(axes={sorted(self._axes)}, fixed={sorted(self._fixed)})"


def jointly_framed(origin: int = 10) -> xr.DataArray:
    """Build the ``scalar_binding_probe`` fixture with joint ownership instead of the guard."""
    stripped = framed(origin).drop_indexes([MARKER, "y", "x"])
    return stripped.set_xindex(["y", "x", MARKER], JointFrameIndex)
