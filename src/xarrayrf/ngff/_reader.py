"""Open OME-Zarr levels and translate older multiscale declarations."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from ome_zarr_models.v04.multiscales import Multiscale as MultiscaleV04
from ome_zarr_models.v05.multiscales import Multiscale as MultiscaleV05


def _older_multiscale(value: Mapping[str, Any], version: str) -> dict[str, Any]:
    for dataset in value.get("datasets", ()):
        for item in dataset.get("coordinateTransformations", ()):
            if "path" in item:
                raise ValueError("path-based 0.4/0.5 transforms are unsupported")
    for item in value.get("coordinateTransformations", ()):
        if "path" in item:
            raise ValueError("path-based 0.4/0.5 transforms are unsupported")
    model = (MultiscaleV04 if version == "0.4" else MultiscaleV05).model_validate(value)
    # The model conversion creates an extra output system for the multiscale-level
    # transform. Earlier specifications have one physical space, so fold it into
    # each dataset transform before passing the result to the v06 importer.
    converted = model.to_version("0.6").model_dump(exclude_none=True)
    system = converted["coordinateSystems"][0]
    system["name"] = model.name if isinstance(model.name, str) and model.name else "intrinsic"
    converted["coordinateSystems"] = [system]
    extra = converted.pop("coordinateTransformations", ())
    members = list(extra[0]["transformations"]) if extra else []
    for dataset in converted["datasets"]:
        existing = dataset["coordinateTransformations"][0]
        # A lone scale converts to a bare transform rather than a one-member sequence.
        own = (
            list(existing["transformations"])
            if existing.get("type") == "sequence"
            else [{k: v for k, v in existing.items() if k not in ("input", "output")}]
        )
        transformations = own + members
        if any(item.get("path") is not None for item in transformations):
            raise ValueError("path-based 0.4/0.5 transforms are unsupported")
        dataset["coordinateTransformations"] = [
            {
                "type": "sequence",
                "input": existing["input"],
                "output": {"name": system["name"]},
                "transformations": transformations,
            }
        ]
    return converted


def open(
    store: Any,
    *,
    group: str = "",
    multiscale: int | str | None = None,
    level: str | None = None,
    chunks: Any = "auto",
) -> xr.DataArray:
    """Open one OME-Zarr level as a framed DataArray.

    For v06, the array is bound to its intrinsic system. Additional transforms
    remain available through ``from_multiscale(...).transforms``. v04/v05 are
    translated into that same intrinsic system; path-based transforms are refused.

    Args:
        store: Local path or store accepted by ``zarr.open_group``.
        group: Relative group containing multiscales.
        multiscale: Index or name; omitted only when exactly one is present.
        level: Dataset path; omitted selects the first, full resolution level.
        chunks: Dask chunks, ``auto`` by default; ``None`` reads eagerly.

    Returns:
        Framed level pixels in declared axis order.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If metadata, selection or transforms cannot be represented.
        FileNotFoundError: If a local store does not exist.
    """
    from . import from_multiscale, to_dataarray

    if not isinstance(group, str):
        raise TypeError("group must be a string")
    if (
        group.startswith("/")
        or group.endswith("/")
        or any(part in ("", ".", "..") for part in group.split("/") if group)
    ):
        raise ValueError("group must be a relative path without empty, '.' or '..' segments")
    if multiscale is not None and (
        isinstance(multiscale, bool) or not isinstance(multiscale, int | str)
    ):
        raise TypeError("multiscale must be an index, name or None")
    if level is not None and not isinstance(level, str):
        raise TypeError("level must be a path string or None")
    if isinstance(store, str) and "://" in store:
        # A URL (https://, s3://, ...) goes to zarr's fsspec-backed stores; it names the
        # store for frame identity just as a local path does.
        store_arg: Any = store
        identity: str | None = store.rstrip("/")
    elif isinstance(store, str | os.PathLike):
        path = os.fspath(store)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        store_arg = path
        identity = path.rstrip("/")
    else:
        store_arg = store
        identity = None
    root = zarr.open_group(store_arg, mode="r")
    node = root[group] if group else root
    attrs = dict(node.attrs)
    ome = attrs.get("ome", {})
    if not isinstance(ome, Mapping):
        raise ValueError("ome attributes must be an object")
    if ome.get("version") in ("0.5", "0.6"):
        version = ome["version"]
        scales = ome.get("multiscales", attrs.get("multiscales"))
    else:
        scales = attrs.get("multiscales")
        versions = {item.get("version") for item in scales} if isinstance(scales, list) else set()
        version = next(iter(versions)) if versions in ({"0.4"}, {"0.6"}) else None
    if version not in ("0.4", "0.5", "0.6") or not isinstance(scales, list):
        raise ValueError("unknown OME-Zarr multiscale version")
    if not scales:
        raise ValueError("no multiscales found")
    listing = ", ".join(f"{i}: {item.get('name', '<unnamed>')}" for i, item in enumerate(scales))
    if multiscale is None:
        if len(scales) != 1:
            raise ValueError(f"multiple multiscales: {listing}; select multiscale")
        index = 0
    elif isinstance(multiscale, int):
        if multiscale < 0 or multiscale >= len(scales):
            raise ValueError(f"unknown multiscale {multiscale}; available: {listing}")
        index = multiscale
    else:
        matches = [i for i, item in enumerate(scales) if item.get("name") == multiscale]
        if len(matches) != 1:
            raise ValueError(
                f"unknown or ambiguous multiscale {multiscale!r}; available: {listing}"
            )
        index = matches[0]
    selected = scales[index]
    metadata = _older_multiscale(selected, version) if version in ("0.4", "0.5") else selected
    paths = [item["path"] for item in metadata["datasets"]]
    if level is None:
        level = paths[0]
    elif level not in paths:
        raise ValueError(f"unknown level {level!r}; available: {', '.join(paths)}")
    arrays: dict[str, zarr.Array[Any]] = {}
    for path in paths:
        candidate = node[path]
        if not isinstance(candidate, zarr.Array):
            raise ValueError(f"dataset path {path!r} must refer to a zarr array")
        arrays[path] = candidate
    shapes = {path: array.shape for path, array in arrays.items()}
    imported = from_multiscale(metadata, shapes=shapes, store=identity, group=group)
    array = arrays[level]
    pixels = (
        np.asarray(array[...]) if chunks is None else da.from_zarr(array)  # type: ignore[no-untyped-call]
    )
    if chunks is not None and chunks != "auto":
        pixels = pixels.rechunk(chunks)  # type: ignore[union-attr]
    return to_dataarray(imported.levels[level], pixels)
