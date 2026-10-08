"""Time resample() on synthetic volumes; development-only, never shipped.

Compares both resampling paths against scipy.ndimage called directly with the same map, so the
overhead xarrayrf adds is visible. Run with ``uv run python benchmarks/resample_benchmark.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from functools import partial

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy import ndimage

import xarrayrf as xrf
from xarrayrf._positions import extents
from xarrayrf._resampling import _lattice_map

FRAME = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y", "z"), ("mm",) * 3))


def geometry(
    values: npt.NDArray[np.float64],
    matrix: npt.NDArray[np.float64],
    translation: tuple[float, ...],
    offsets: npt.NDArray[np.float64] | None = None,
) -> xrf.Geometry:
    size = values.shape[0]
    coords: dict[str, object] = {
        "k": np.arange(size),
        "j": np.arange(values.shape[1]),
        "i": np.arange(values.shape[2]),
    }
    axes = ("i", "j", "k")
    if offsets is not None:
        coords["slice_offset"] = ("k", offsets)
        axes = ("i", "j", "slice_offset")
    array = xr.DataArray(values, dims=("k", "j", "i"), coords=coords)
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(axes, ("1", "1", "1")),
        target=FRAME,
        matrix=matrix,
        translation=translation,
    )
    return xrf.Geometry(array, transform, dims=("k", "j", "i"))


def timed(label: str, function: Callable[[], object]) -> None:
    start = time.perf_counter()
    function()
    print(f"{label:<40} {time.perf_counter() - start:8.3f} s", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128, help="edge length of the cubic volumes")
    size = parser.parse_args().size
    rng = np.random.default_rng(0)
    angle = np.radians(10.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]]
    )
    source = geometry(rng.random((size,) * 3), np.diag([1.0, 1.0, 2.0]), (0.0, 0.0, 0.0))
    target = geometry(np.zeros((size,) * 3), rotation, (5.0, -3.0, 10.0))
    offsets = np.cumsum(rng.uniform(1.5, 2.5, size))
    nonuniform = geometry(rng.random((size,) * 3), np.eye(3), (0.0, 0.0, 0.0), offsets - offsets[0])
    print(f"{size}^3 volumes, 10 degree rotation", file=sys.stderr)
    methods: tuple[tuple[xrf.Method, int], ...] = (("nearest", 0), ("linear", 1), ("cubic", 3))
    # The baseline is scipy alone applying the same composed map resample() uses, with the same
    # edge mode and a separately timed prefilter, so the difference is xarrayrf's overhead.
    composed = _lattice_map(
        source._sampling(),
        target._sampling(),
        source.dims,
        ("k", "j", "i"),
        None,
        extents(source._sampling(), "samples"),
    )
    assert composed is not None
    values = source.array.values
    for method, order in methods:
        timed(f"lattice path, {method}", partial(xrf.resample, source, target, method=method))
        timed(
            f"nonuniform path, {method}", partial(xrf.resample, nonuniform, target, method=method)
        )
        timed(
            f"scipy alone, same map, order {order}",
            partial(_scipy_alone, values, composed.linear, composed.offset, order),
        )


def _scipy_alone(
    values: npt.NDArray[np.float64],
    linear: npt.NDArray[np.float64],
    offset: npt.NDArray[np.float64],
    order: int,
) -> npt.NDArray[np.float64]:
    coefficients = (
        ndimage.spline_filter(values, order=order, mode="nearest") if order > 1 else values
    )
    result: npt.NDArray[np.float64] = ndimage.affine_transform(
        coefficients, linear, offset=offset, order=order, mode="nearest", prefilter=False
    )
    return result


if __name__ == "__main__":
    main()
