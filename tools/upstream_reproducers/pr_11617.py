"""#11617 drop=True with index-owned scalars. main: x and y kept; PR: empty coords, and a
single-axis selection with drop=True raises the corrupt-index error."""

import xarray as xr
from _raster_index import ScalarKeepingRasterIndex, raster

print(xr.__file__)
r = raster(ScalarKeepingRasterIndex)
print("pixel:", list(r.isel(x=1, y=0, drop=True).coords))
try:
    print("row:", list(r.isel(y=0, drop=True).coords))
except ValueError as e:
    print("row: ValueError:", str(e)[:70])
