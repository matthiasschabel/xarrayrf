"""#11613 roll on an empty dimension. main: ZeroDivisionError; PR: empty result unchanged."""

import xarray as xr

print(xr.__file__)
print(xr.DataArray([], coords={"x": range(0)}, dims="x").roll(x=1, roll_coords=True))
