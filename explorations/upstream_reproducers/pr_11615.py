"""#11615 broadcast with an index spanning several dimensions. main: ValueError; PR: dims (y, x, band)."""

import xarray as xr
from _raster_index import raster

print(xr.__file__)
result, _ = xr.broadcast(raster(), xr.DataArray([1, 2], dims="band"))
print(result.dims)
print(result.xindexes)
