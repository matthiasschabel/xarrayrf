"""Dtype-preserving positional selection and per-axis sampling updates."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def positional_indexer(indexer: Any, *, dim: str) -> Any:
    """Normalize one dimension's indexer, refusing changes to geometry dimensions."""
    if indexer is Ellipsis:
        raise ValueError(
            "dimensions () must have the same length as the number of data dimensions, ndim=1"
        )
    if isinstance(indexer, tuple):
        if len(indexer) not in (2, 3, 4):
            raise ValueError(f"Tuple {indexer} is not in the form (dims, data[, attrs])")
        dims = (indexer[0],) if isinstance(indexer[0], str) else indexer[0]
        if not isinstance(dims, tuple | list) or not all(isinstance(name, str) for name in dims):
            raise TypeError(
                "Variable None: Could not convert tuple of form "
                f"(dims, data[, attrs, encoding]): {indexer} to Variable."
            )
        if tuple(dims) != (dim,):
            raise ValueError("vectorized indexing changes geometry dimensions; call rf.unframe()")
        indexer = np.asarray(indexer[1])
    if hasattr(indexer, "dims"):
        if indexer.dims != (dim,):
            raise ValueError("vectorized indexing changes geometry dimensions; call rf.unframe()")
        indexer = indexer.data
    if isinstance(indexer, slice):
        return indexer
    if indexer is None or isinstance(indexer, bool | np.bool_):
        raise ValueError(
            "Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer supported. "
            "Convert to a numpy array before indexing instead."
        )
    if isinstance(indexer, int | np.integer):
        return indexer
    values = np.asarray(indexer)
    if values.ndim > 1:
        raise IndexError(f"Unlabeled multi-dimensional array cannot be used for indexing: {dim}")
    if values.ndim == 0:
        return positional_indexer(values.item(), dim=dim) if values.dtype.kind in "biu" else indexer
    if isinstance(indexer, list) and not indexer:
        return np.array([], dtype=np.intp)
    return values


def select_axis[T: np.generic](
    values: npt.NDArray[T],
    indexer: Any,
    *,
    step: float | None = None,
    intervals: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[T], float | None, npt.NDArray[np.float64] | None]:
    """Select coordinate values and interval rows, multiplying an exact step by slice stride."""
    # NumPy accepts a size-0 mask on any axis; xarray, and so the native binding, refuses it.
    if (
        isinstance(indexer, np.ndarray)
        and indexer.dtype == bool
        and not indexer.size
        and values.size
    ):
        raise ValueError(
            "The length of the boolean indexer cannot be 0 when the Index has length greater than 0."
        )
    selected = np.asarray(values[indexer])
    stride = indexer.indices(values.size)[2] if isinstance(indexer, slice) else None
    return (
        selected,
        step * stride if step is not None and stride is not None else None,
        intervals[indexer] if intervals is not None else None,
    )
