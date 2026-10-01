"""Read GeoTIFF pixels through independent rasterio window tasks."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, cast

import dask.array as da
import numpy as np
import rasterio  # type: ignore[import-untyped]
import xarray as xr
from dask import delayed  # type: ignore[attr-defined]
from rasterio.windows import Window  # type: ignore[import-untyped]

from xarrayrf import ReferenceFrame


def _read_window(path: str, bands: tuple[int, ...], window: Window) -> np.ndarray[Any, Any]:
    with rasterio.open(path) as dataset:
        return cast(np.ndarray[Any, Any], dataset.read(bands, window=window))


AUTO_CHUNK_PIXELS = 2048
"""Target side of an automatic chunk; a crop of a remote COG then fetches a few tiles."""


def _tile_aligned_chunks(dataset: Any, bands: int) -> tuple[int, int, int]:
    """Chunks of whole internal tiles (or strips), about ``AUTO_CHUNK_PIXELS`` on a side."""
    tile_rows, tile_columns = dataset.block_shapes[0]
    rows = max(tile_rows, AUTO_CHUNK_PIXELS // tile_rows * tile_rows)
    columns = max(tile_columns, AUTO_CHUNK_PIXELS // tile_columns * tile_columns)
    return bands, min(rows, dataset.height), min(columns, dataset.width)


def open(
    path: str | os.PathLike[str],
    *,
    frame: ReferenceFrame | xr.DataArray | None = None,
    bands: Sequence[int] | None = None,
    chunks: Any = "auto",
) -> xr.DataArray:
    """Open a projected GeoTIFF as a framed DataArray.

    Nodata, masks and scale/offset are not applied. Each lazy chunk opens its own
    rasterio dataset and reads exactly one window.

    Args:
        path: Local or rasterio virtual filesystem path.
        frame: Frame or framed array to adopt, overriding the imported CRS identity.
        bands: Optional sequence of 1-based band indices.
        chunks: Dask chunk specification; ``None`` reads eagerly. ``"auto"`` uses whole internal
            tiles (or strips), about 2048 pixels on a side, so a crop of a remote COG fetches
            only the tiles it covers.

    Returns:
        Framed pixels with a 1-based band dimension.

    Raises:
        TypeError: If a public argument has an invalid type.
        ValueError: If band selection or metadata cannot be represented.
        FileNotFoundError: If the path does not exist.
    """
    from . import from_profile, to_dataarray

    if not isinstance(path, str | os.PathLike):
        raise TypeError(f"path must be a filesystem path, got {type(path).__name__}")
    filename = os.fspath(path)
    try:
        dataset = rasterio.open(filename)
    except rasterio.errors.RasterioIOError as exc:
        if not filename.startswith("/vsi") and not os.path.exists(filename):
            raise FileNotFoundError(filename) from exc
        raise
    with dataset:
        if bands is None:
            selected = tuple(range(1, dataset.count + 1))
        else:
            if isinstance(bands, str | bytes) or not isinstance(bands, Sequence):
                raise TypeError("bands must be a sequence of 1-based integers or None")
            selected = tuple(bands)
            if any(isinstance(b, bool) or not isinstance(b, int) for b in selected):
                raise TypeError("bands must contain 1-based integers")
            if (
                not selected
                or len(set(selected)) != len(selected)
                or any(b < 1 or b > dataset.count for b in selected)
            ):
                raise ValueError(f"bands must be distinct indices from 1 to {dataset.count}")
        profile = dataset.profile.copy()
        profile["count"] = len(selected)
        geometry = from_profile(
            profile, frame=frame, area_or_point=dataset.tags().get("AREA_OR_POINT", "Area")
        )
        if chunks is None:
            pixels = dataset.read(selected)
        else:
            shape = (len(selected), dataset.height, dataset.width)
            if isinstance(chunks, str) and chunks == "auto":
                chunks = _tile_aligned_chunks(dataset, len(selected))
            normalized = da.core.normalize_chunks(  # type: ignore[no-untyped-call]
                chunks, shape=shape, dtype=dataset.dtypes[0]
            )
            blocks = []
            band_start = 0
            for band_size in normalized[0]:
                rows = []
                row_start = 0
                for row_size in normalized[1]:
                    columns = []
                    column_start = 0
                    for column_size in normalized[2]:
                        window = Window(column_start, row_start, column_size, row_size)
                        block = da.from_delayed(  # type: ignore[no-untyped-call]
                            delayed(_read_window)(
                                filename, selected[band_start : band_start + band_size], window
                            ),
                            shape=(band_size, row_size, column_size),
                            dtype=dataset.dtypes[0],
                        )
                        columns.append(block)
                        column_start += column_size
                    rows.append(da.concatenate(columns, axis=2))  # type: ignore[no-untyped-call]
                    row_start += row_size
                blocks.append(da.concatenate(rows, axis=1))  # type: ignore[no-untyped-call]
                band_start += band_size
            pixels = da.concatenate(blocks, axis=0)  # type: ignore[no-untyped-call]
    array = to_dataarray(geometry, pixels)
    if bands is not None:
        array = array.assign_coords(band=list(selected))
    return array
