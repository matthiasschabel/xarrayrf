"""#11616 Dataset reductions. main: RasterIndex left on y (still spanning x); PR: no index."""

import xarray as xr
from _raster_index import raster

print(xr.__file__)
ds = raster().to_dataset(name="red")
print(ds.mean("x").xindexes)
