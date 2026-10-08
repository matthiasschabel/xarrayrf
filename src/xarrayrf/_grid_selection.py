"""xarray label-selection convenience for the otherwise independent Grid."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from ._binding import binding_coordinates, grid_from_binding, grid_variables
from ._grid import Grid


def select_grid_labels(grid: Grid, indexers: Mapping[str, Any]) -> Grid:
    """Select labels on a coordinate-only Dataset carrying the Grid's native binding."""
    coordinates = binding_coordinates(
        grid_variables(grid), grid.transform, grid.dims, grid.intervals
    )
    selected = xr.Dataset(coords=coordinates).sel(indexers)
    return grid_from_binding(selected.coords)
