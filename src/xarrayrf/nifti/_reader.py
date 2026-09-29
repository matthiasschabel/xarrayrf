"""Read NIfTI pixels and bind the existing header geometry."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal, cast

import dask.array as da
import nibabel as nib
import numpy as np
import xarray as xr

from xarrayrf import ReferenceFrame

if TYPE_CHECKING:
    from . import NiftiGeometry


def open(
    path: str | os.PathLike[str],
    *,
    frame: ReferenceFrame | xr.DataArray | None = None,
    template: str | None = None,
    xform: Literal["best", "sform", "qform"] = "best",
    spatial_unit: str | None = "mm",
    time: bool = False,
    chunks: Any = "auto",
) -> xr.DataArray:
    """Open a NIfTI image with its header geometry.

    The nibabel array proxy applies stored slope and intercept when sliced. Its
    resulting dtype follows nibabel proxy semantics and may differ from the stored dtype.

    Args:
        path: Local NIfTI file path.
        frame: Optional known frame identity or framed DataArray.
        template: TemplateFlow/BIDS space label for a coded aligned xform.
        xform: Coded transform to use; ``best`` prefers sform.
        spatial_unit: Fallback for an unknown header unit.
        time: Include a declared time axis in geometry.
        chunks: Dask chunk specification, ``auto`` by default; ``None`` reads eagerly.

    Returns:
        Framed image in header dimension order.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If the header or chunk specification is invalid.
        FileNotFoundError: If the file does not exist.
    """
    from . import from_header, to_dataarray

    if not isinstance(path, str | os.PathLike):
        raise TypeError("path must be a filesystem path")
    filename = os.fspath(path)
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)
    image = cast(nib.Nifti1Image | nib.Nifti2Image, nib.load(filename))
    geometry: NiftiGeometry = from_header(
        image.header,
        frame=frame,
        template=template,
        xform=xform,
        spatial_unit=spatial_unit,
        time=time,
    )
    if chunks is None:
        pixels = np.asarray(image.dataobj)
    else:
        proxy = cast(nib.arrayproxy.ArrayProxy, image.dataobj)
        dtype = nib.volumeutils.apply_read_scaling(
            np.empty(0, dtype=proxy.dtype), proxy.slope, proxy.inter
        ).dtype
        meta = np.empty((0,) * len(proxy.shape), dtype=dtype)
        pixels = da.from_array(proxy, chunks=chunks, meta=meta).astype(dtype)  # type: ignore[no-untyped-call]
    return to_dataarray(geometry, pixels)
