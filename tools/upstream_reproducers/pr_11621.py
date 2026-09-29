"""#11621 Dataset.update replacing a custom index.

main: the RasterIndex is silently replaced by two PandasIndex.
PR: the assignment raises; dropping the incoming indexes keeps the RasterIndex.
"""

import numpy as np
import xarray as xr
from _raster_index import raster

print(xr.__file__)
ds = raster().to_dataset(name="red")
nir = xr.DataArray(
    np.ones((100, 50)), coords={"x": np.arange(50), "y": np.arange(100)}, dims=("y", "x")
)
try:
    ds["nir"] = nir
except xr.AlignmentError as error:
    print("raised:", error)
    ds["nir"] = nir.drop_indexes(["x", "y"])
print(ds.xindexes)
