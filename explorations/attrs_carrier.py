"""A provisional attrs carrier for probing, not a binding.

``da.rf_probe.attach(transform, dims)`` stores the encoded transform and geometry dims as a JSON
string in ``attrs["xarrayrf_probe"]``; ``da.rf_probe.geometry()`` decodes it into a ``Geometry``.
There are no guards: xarray decides whether attrs survive an operation, and nothing checks that a
surviving declaration is still true. The accessor name is deliberately not ``rf``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import xarray as xr

import xarrayrf as xrf

KEY = "xarrayrf_probe"


@xr.register_dataarray_accessor("rf_probe")
class AttrsCarrier:
    def __init__(self, array: xr.DataArray) -> None:
        self._array = array

    def attach(self, transform: xrf.SupportsPoints, dims: Sequence[str]) -> xr.DataArray:
        """Return a copy of the array carrying the declaration; validates it once."""
        xrf.Geometry(self._array, transform, dims=dims)
        payload = json.dumps({"transform": xrf.encode(transform), "dims": list(dims)})
        return self._array.assign_attrs({KEY: payload})

    @property
    def attached(self) -> bool:
        return KEY in self._array.attrs

    def geometry(self) -> xrf.Geometry:
        """Rebuild the pairing from the carried declaration, as the array now stands."""
        payload = json.loads(self._array.attrs[KEY])
        return xrf.Geometry(self._array, xrf.decode(payload["transform"]), dims=payload["dims"])
