"""Time resample_to against SimpleITK Resample on like-for-like inputs; development-only.

The comparison workload of the resampling performance plan: a moving image onto a fixed grid of
the same size through a dense field, a B-spline field (xarrayrf only) or an affine, in 3-D and
2-D, for float32 and float64 data. Both sides use the same semantics: cell-sampled images with
centred cells, ``domain="cells"`` (ITK's half-cell domain holding the edge value), a dense field
with ``outside="identity"`` (ITK's zero displacement outside the field) and ``fill_value=0``.
Each line is the median of ``--repeat`` warmed runs and reports the result dtype and, where
SimpleITK ran, the largest difference from it:

    uv run --no-sync python benchmarks/field_resample_benchmark.py
    uv run --no-sync --with simpleitk python benchmarks/field_resample_benchmark.py --memory

Native thread pools (OpenMP, BLAS) are pinned to one thread unless the environment sets them;
SimpleITK runs with one thread and with its default count. Results go to stdout, progress to
stderr.
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402 (the thread pins must precede NumPy's import)
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import xarray as xr  # noqa: E402
from scipy import ndimage  # noqa: E402

import xarrayrf as xrf  # noqa: E402
import xarrayrf.native  # noqa: E402 (registers the rf accessor)

FILL = 0.0
SPACING = 2.0


@dataclass(frozen=True)
class Options:
    repeat: int
    memory: bool


def measure(label: str, function: Callable[[], Any], *, points: int, options: Options) -> Any:
    """Print the median of warmed runs, the result dtype and, optionally, traced peak memory."""
    print(f"running {label}", file=sys.stderr)
    result = function()
    seconds = []
    for _ in range(options.repeat):
        start = time.perf_counter()
        result = function()
        seconds.append(time.perf_counter() - start)
    median = statistics.median(seconds)
    memory = ""
    if options.memory:
        tracemalloc.start()
        function()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory = f" {peak / 2**20:7.0f} MiB traced peak"
    # A DataArray has a dtype; a SimpleITK image names its pixel type.
    dtype = getattr(result, "dtype", None) or result.GetPixelIDTypeAsString()
    print(f"{label:<54} {median:7.3f} s {points / median / 1e6:7.2f} Mpt/s {dtype!s:>8}{memory}")
    return result


def frames(dims: int) -> tuple[xrf.ReferenceFrame, xrf.ReferenceFrame]:
    axes = ("x", "y", "z")[:dims]
    system = xrf.CoordinateSystem(axes, ("mm",) * dims)
    return (
        xrf.ReferenceFrame.declared(("benchmark", f"fixed{dims}d"), system),
        xrf.ReferenceFrame.declared(("benchmark", f"moving{dims}d"), system),
    )


def index_dims(dims: int) -> tuple[str, ...]:
    return ("i", "j", "k")[:dims]


def field_grid(size: int, dims: int, frame: xrf.ReferenceFrame, *, cells: bool) -> xrf.Grid:
    names = index_dims(dims)
    source = xrf.ArrayCoordinates(
        names, ("1",) * dims, sample_offset=(0.5,) * dims if cells else None
    )
    transform = xrf.AffineTransform.from_matrix(
        source=source, target=frame, matrix=np.eye(dims) * SPACING, translation=np.zeros(dims)
    )
    return xrf.Grid(transform, {name: (name, np.arange(size)) for name in names})


def framed(
    data: npt.NDArray[np.floating], frame: xrf.ReferenceFrame, step: float, origin: float
) -> Any:
    """A cell-sampled image with centred cells, sample ``n`` at ``origin + n * step``."""
    names = index_dims(data.ndim)
    array = xr.DataArray(
        data,
        dims=names,
        coords={
            d: (d, np.arange(s), {"units": "1"}) for d, s in zip(names, data.shape, strict=True)
        },
    )
    binding = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(names, ("1",) * data.ndim, sample_offset=(0.5,) * data.ndim),
        target=frame,
        matrix=np.eye(data.ndim) * step,
        translation=np.full(data.ndim, origin),
    )
    return array.rf.frame(binding, dims=names)


def sitk_image(sitk: Any, data: npt.NDArray[np.floating], step: float, origin: float) -> Any:
    """SimpleITK stores x fastest, so our (i, j, k) array is reversed into its (k, j, i)."""
    image = sitk.GetImageFromArray(np.ascontiguousarray(data.T))
    image.SetSpacing((step,) * data.ndim)
    image.SetOrigin((origin,) * data.ndim)
    return image


def sitk_field(sitk: Any, field: npt.NDArray[np.floating]) -> Any:
    dims = field.ndim - 1
    reversed_axes = (*range(dims - 1, -1, -1), dims)
    image = sitk.GetImageFromArray(
        np.ascontiguousarray(np.transpose(field.astype(np.float64), reversed_axes)), isVector=True
    )
    image.SetSpacing((SPACING,) * dims)
    return sitk.DisplacementFieldTransform(image)


def smooth_field(rng: np.random.Generator, size: int, dims: int) -> npt.NDArray[np.float32]:
    noise = rng.normal(size=(size,) * dims + (dims,))
    smooth: npt.NDArray[np.float64] = ndimage.gaussian_filter(noise, (4,) * dims + (0,))
    scaled: npt.NDArray[np.float64] = smooth / float(np.abs(smooth).max()) * 1.5
    return scaled.astype(np.float32)


def run_case(
    name: str,
    *,
    dims: int,
    image: int,
    field: int,
    dtype: type[np.floating],
    kind: str,
    sitk: Any,
    options: Options,
) -> None:
    rng = np.random.default_rng(0)
    fixed_frame, moving_frame = frames(dims)
    step = (field - 4.0) * SPACING / (image - 1)
    origin = 1.5 * SPACING
    values = rng.random((image,) * dims).astype(dtype)
    moving = framed(values, moving_frame, step, origin)
    fixed = framed(np.zeros((image,) * dims, dtype=dtype), fixed_frame, step, origin)
    displacement = smooth_field(rng, field, dims)
    sitk_transform: Any = None
    if kind == "dense":
        transform: Any = xrf.DisplacementFieldTransform(
            source=fixed_frame,
            target=moving_frame,
            field=displacement,
            grid=field_grid(field, dims, fixed_frame, cells=True),
            outside="identity",
        )
        if sitk is not None:
            sitk_transform = sitk_field(sitk, displacement)
    elif kind == "bspline":
        transform = xrf.BSplineTransform(
            source=fixed_frame,
            target=moving_frame,
            coefficients=displacement,
            grid=field_grid(field, dims, fixed_frame, cells=False),
            outside="identity",
        )
    else:
        angle = np.radians(5.0)
        rotation = np.eye(dims)
        rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        centre = np.full(dims, origin + step * (image - 1) / 2)
        translation = centre - rotation @ centre
        transform = xrf.AffineTransform.from_matrix(
            source=fixed_frame, target=moving_frame, matrix=rotation, translation=translation
        )
        if sitk is not None:
            sitk_transform = sitk.AffineTransform(dims)
            sitk_transform.SetMatrix(rotation.ravel().tolist())
            sitk_transform.SetTranslation(translation.tolist())
    points = image**dims
    label = f"{name}, {np.dtype(dtype).name}"
    ours = measure(
        f"xarrayrf {label}",
        lambda: moving.rf.resample_to(
            fixed, transform=transform, method="linear", domain="cells", fill_value=FILL
        ),
        points=points,
        options=options,
    )
    if sitk_transform is None:
        return
    moving_itk = sitk_image(sitk, values, step, origin)
    reference = sitk_image(sitk, np.zeros((image,) * dims, dtype=dtype), step, origin)
    available = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    for threads in (1, available):
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
        theirs = measure(
            f"SimpleITK {label}, {threads} threads",
            lambda: sitk.Resample(moving_itk, reference, sitk_transform, sitk.sitkLinear, FILL),
            points=points,
            options=options,
        )
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(available)
    difference = np.abs(
        np.asarray(ours.values, dtype=np.float64) - sitk.GetArrayFromImage(theirs).T
    )
    print(f"{'':<54} max |xarrayrf - SimpleITK| = {difference.max():.3g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=int, default=256, help="3-D image samples per axis")
    parser.add_argument("--field", type=int, default=128, help="3-D field samples per axis")
    parser.add_argument("--repeat", type=int, default=3, help="timed runs after one warm-up")
    parser.add_argument("--memory", action="store_true", help="add a traced-memory run per line")
    args = parser.parse_args()
    options = Options(args.repeat, args.memory)
    try:
        import SimpleITK as sitk  # noqa: N813 (SimpleITK's own spelling)
    except ImportError:
        sitk = None
        print("SimpleITK not importable; xarrayrf lines only", file=sys.stderr)
    pins = ", ".join(f"{v}={os.environ[v]}" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    print(f"native threads: {pins}; {os.cpu_count()} cores", file=sys.stderr)
    # The 2-D case keeps the 3-D point count: image and field edges to the power 3/2.
    image2, field2 = round(args.image**1.5), round(args.field**1.5)
    for dtype in (np.float32, np.float64):
        for name, dims, image, field, kind in (
            ("3-D dense field", 3, args.image, args.field, "dense"),
            ("3-D B-spline field", 3, args.image, args.field, "bspline"),
            ("3-D affine", 3, args.image, args.field, "affine"),
            ("2-D dense field", 2, image2, field2, "dense"),
        ):
            run_case(
                name,
                dims=dims,
                image=image,
                field=field,
                dtype=dtype,
                kind=kind,
                sitk=sitk,
                options=options,
            )


if __name__ == "__main__":
    main()
