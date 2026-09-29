"""The RasterIndex from xarray's custom-index guide (Meta-indexes section), minus its asserts,
plus an optional scalar-keeping ``isel`` used by the drop=True reproducer."""

import numpy as np
import xarray as xr
from xarray import Index, Variable
from xarray.core.indexes import PandasIndex


class RasterIndex(Index):
    def __init__(self, xy_indexes, scalars=None):
        self._xy_indexes = dict(xy_indexes)
        self._scalars = dict(scalars or {})

    @classmethod
    def from_variables(cls, variables, *, options):
        return cls(
            {k: PandasIndex.from_variables({k: v}, options=options) for k, v in variables.items()}
        )

    def create_variables(self, variables=None):
        out = dict(self._scalars)
        for index in self._xy_indexes.values():
            out.update(index.create_variables(variables))
        return out


class ScalarKeepingRasterIndex(RasterIndex):
    def isel(self, indexers):
        axes, scalars = {}, dict(self._scalars)
        for k, idx in self._xy_indexes.items():
            if idx.dim in indexers:
                new = idx.isel({idx.dim: indexers[idx.dim]})
                if new is None:
                    scalars[k] = Variable((), idx.index[indexers[idx.dim]])
                else:
                    axes[k] = new
            else:
                axes[k] = idx
        return type(self)(axes, scalars)


def raster(index_cls=RasterIndex):
    da = xr.DataArray(
        np.zeros((100, 50)), coords={"x": np.arange(50), "y": np.arange(100)}, dims=("y", "x")
    )
    return da.set_xindex(["x", "y"], index_cls)
