"""Time field transforms on 16.7M points; development-only, never shipped.

Design acceptance test 8 of the field transform design: a 256³ image through a 128³ field.
Measures the prepared grid lookup, values alone and values plus Jacobians for the dense field and
through the public methods, and each B-spline backend directly (``map_coordinates`` without prefiltering,
``NdBSpline``), with peak traced memory, and the whole resampling against SimpleITK's
``Resample`` when SimpleITK is importable:

    uv run --no-sync python benchmarks/field_transform_benchmark.py
    uv run --no-sync --with simpleitk python benchmarks/field_transform_benchmark.py

``--image`` and ``--field`` shrink the problem for a quick run. Results go to stdout as one line
per measurement; progress goes to stderr.
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy import ndimage
from scipy.interpolate import NdBSpline

import xarrayrf as xrf
import xarrayrf.native

SYSTEM = xrf.CoordinateSystem(("x", "y", "z"), ("mm",) * 3)
FIXED = xrf.ReferenceFrame.declared(("benchmark", "fixed"), SYSTEM)
MOVING = xrf.ReferenceFrame.declared(("benchmark", "moving"), SYSTEM)
CELLS = xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3, sample_offset=(0.5,) * 3)
CONTROL = xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3)


def measure(label: str, function: Callable[[], object], *, points: int) -> object:
    """Run once, print seconds, million points per second and traced peak memory."""
    print(f"running {label}", file=sys.stderr)
    tracemalloc.start()
    start = time.perf_counter()
    result = function()
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rate = points / seconds / 1e6
    print(f"{label:<52} {seconds:8.2f} s {rate:8.2f} Mpt/s {peak / 2**20:9.0f} MiB peak")
    return result


def grid(size: int, spacing: float, source: xrf.ArrayCoordinates) -> xrf.Grid:
    transform = xrf.AffineTransform.from_matrix(
        source=source, target=FIXED, matrix=np.eye(3) * spacing, translation=np.zeros(3)
    )
    return xrf.Grid(transform, {axis: (axis, np.arange(size)) for axis in "ijk"})


def target_points(image: int, field: int, spacing: float) -> npt.NDArray[np.float64]:
    """An image grid spanning the field's interior, as (N, 3) points in the fixed frame."""
    axis = np.linspace(1.5 * spacing, (field - 2.5) * spacing, image)
    i, j, k = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([i.ravel(), j.ravel(), k.ravel()], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=int, default=256, help="image samples per axis")
    parser.add_argument("--field", type=int, default=128, help="field samples per axis")
    args = parser.parse_args()
    image, size, spacing = args.image, args.field, 2.0
    rng = np.random.default_rng(0)
    smooth = ndimage.gaussian_filter(rng.normal(size=(size, size, size, 3)), (4, 4, 4, 0))
    values = (smooth / np.abs(smooth).max() * 1.5).astype(np.float32)
    dense = xrf.DisplacementFieldTransform(
        source=FIXED, target=MOVING, field=values, grid=grid(size, spacing, CELLS)
    )
    spline = xrf.BSplineTransform(
        source=FIXED, target=MOVING, coefficients=values, grid=grid(size, spacing, CONTROL)
    )
    points = target_points(image, size, spacing)
    count = len(points)
    print(f"{count:,} points; field {size}³ float32; machine-dependent numbers", file=sys.stderr)

    lookup: Any = spline._lookup  # the prepared world-to-index map both kinds share
    positions = measure("prepared grid lookup", lambda: lookup.positions(points), points=count)
    assert isinstance(positions, np.ndarray)
    measure("dense (public): values", lambda: dense.transform_point(points), points=count)
    measure(
        "dense (public): values and Jacobians",
        lambda: (dense.transform_point(points), dense.jacobian(points)),
        points=count,
    )
    measure("B-spline (public): values", lambda: spline.transform_point(points), points=count)
    measure(
        "B-spline (public): values and Jacobians",
        lambda: (spline.transform_point(points), spline.jacobian(points)),
        points=count,
    )
    coordinates = np.ascontiguousarray(positions.T)
    measure(
        "B-spline, map_coordinates: values",
        lambda: [
            ndimage.map_coordinates(values[..., c], coordinates, order=3, prefilter=False)
            for c in range(3)
        ],
        points=count,
    )
    knots = tuple(np.arange(size + 4, dtype=np.float64) - 2.0 for _ in range(3))
    subset = positions[: max(count // 16, 1)]
    ndb = NdBSpline(knots, values.astype(np.float64), 3)
    measure("B-spline, NdBSpline: values (1/16 of points)", lambda: ndb(subset), points=len(subset))
    measure(
        "B-spline, NdBSpline: values and 3 derivatives (1/16)",
        lambda: [
            ndb(subset),
            *(ndb(subset, nu=tuple(int(a == b) for b in range(3))) for a in range(3)),
        ],
        points=len(subset),
    )

    # The whole resampling: an image^3 moving image onto an image^3 fixed grid through the field.
    moving_values = rng.random((image, image, image), dtype=np.float32)

    def framed(
        data: npt.NDArray[np.float32], frame: xrf.ReferenceFrame, step: float, origin: float
    ) -> Any:
        array = xr.DataArray(
            data,
            dims=("i", "j", "k"),
            coords={d: (d, np.arange(image), {"units": "1"}) for d in "ijk"},
        )
        binding = xrf.AffineTransform.from_matrix(
            source=xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3),
            target=frame,
            matrix=np.eye(3) * step,
            translation=np.full(3, origin),
        )
        return array.rf.frame(binding, dims=("i", "j", "k"))

    step = (size - 4.0) * spacing / (image - 1)
    moving_image = framed(moving_values, MOVING, step, 1.5 * spacing)
    fixed_image = framed(np.zeros((image,) * 3, dtype=np.float32), FIXED, step, 1.5 * spacing)
    measure(
        "resample_to through the dense field (linear)",
        lambda: moving_image.rf.resample_to(fixed_image, transform=dense, method="linear"),
        points=count,
    )
    try:
        import SimpleITK as sitk  # noqa: N813 (SimpleITK's own spelling)
    except ImportError:
        print("SimpleITK not importable; skipping its Resample", file=sys.stderr)
        return
    field_image = sitk.GetImageFromArray(
        np.transpose(values.astype(np.float64), (2, 1, 0, 3)), isVector=True
    )
    field_image.SetSpacing((spacing,) * 3)
    transform = sitk.DisplacementFieldTransform(field_image)
    moving_itk = sitk.GetImageFromArray(np.transpose(moving_values, (2, 1, 0)))
    moving_itk.SetSpacing((step,) * 3)
    moving_itk.SetOrigin((1.5 * spacing,) * 3)
    reference = sitk.Image(image, image, image, sitk.sitkFloat32)
    reference.SetSpacing((step,) * 3)
    reference.SetOrigin((1.5 * spacing,) * 3)
    available = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    for threads in (1, available):
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
        label = f"SimpleITK Resample through the field ({threads} threads)"
        measure(
            label,
            lambda: sitk.Resample(moving_itk, reference, transform, sitk.sitkLinear),
            points=count,
        )


if __name__ == "__main__":
    main()
