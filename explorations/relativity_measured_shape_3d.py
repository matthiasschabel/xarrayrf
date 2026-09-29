"""Measure an oblique Lorentz contraction of a static 3-D scalar marker field.

Run with ``PYTHONPATH=src python explorations/relativity_measured_shape_3d.py``.
All four coordinates (ct, z, y, x) are in metres. This is a simultaneous
measurement, not a photograph or a conserved-density transformation.

At ct'=0, x' = A x + b, where A = I + (1/gamma - 1) n n.T and
b = a_space + beta*a_ct*n for the Poincare translation a. Thus the centered
second moment is C' = A C A.T: longitudinal variance contracts by 1/gamma**2,
RMS width by 1/gamma, transverse covariance is unchanged, and mixed moments
contract by 1/gamma. Principal eigenvectors need not stay fixed.

The source has only two time planes. Inverse-mapped target corners give the
minimal rest-time interval covering the whole target box; linear interpolation
between identical static planes is exact in time. One rest-time plane would
not cover the relativity-of-simultaneity tilt. Default source storage is
2*63**3 float64 values (3.82 MiB); target storage is 1*63**3 (1.91 MiB).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import cast

import numpy as np
import numpy.typing as npt
import xarray as xr

import xarrayrf as xrf
import xarrayrf.native  # Register the public DataArray .rf accessor.

type Array = npt.NDArray[np.float64]

AXES = ("ct", "z", "y", "x")
SYSTEM = xrf.CoordinateSystem(AXES, ("m",) * 4)
BETA = 0.8
GAMMA = 1.0 / np.sqrt(1.0 - BETA**2)
DIRECTION = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)  # z, y, x order
TRANSLATION = np.array([0.35, -0.2, 0.15, 0.1])
SOURCE_LIMIT = 7.0
TARGET_LIMIT = 3.6
CENTERS = np.array([[-0.8, 0.35, 0.5], [0.55, -0.65, -0.4], [0.1, 0.8, -0.7]])
WIDTHS = np.array([[0.55, 0.4, 0.65], [0.35, 0.6, 0.45], [0.5, 0.35, 0.3]])
AMPLITUDES = np.array([1.0, 0.7, 0.45])


@dataclass(frozen=True)
class Measurement:
    """Centered moments in z/y/x order, widths, and sampling diagnostics."""

    rest_covariance: Array
    moving_covariance: Array
    predicted_covariance: Array
    rms_ratios: Array
    rest_time_range: Array
    source_spacing: float
    boundary_mass_fraction: float


def analytic_rest_covariance() -> Array:
    """Return the infinite-domain Gaussian mixture's centered second moment."""
    masses = AMPLITUDES * np.prod(WIDTHS, axis=1)
    weights = masses / masses.sum()
    center = weights @ CENTERS
    offsets = CENTERS - center
    return np.diag(weights @ WIDTHS**2) + (offsets.T * weights) @ offsets


def motion_basis() -> Array:
    """Return orthonormal rows: motion direction and two transverse directions."""
    transverse = np.cross(DIRECTION, np.array([0.0, 0.0, 1.0]))
    transverse /= np.linalg.norm(transverse)
    return np.stack([DIRECTION, transverse, np.cross(DIRECTION, transverse)])


def _framed(values: Array, coordinates: list[Array], frame: xrf.ReferenceFrame) -> xr.DataArray:
    array = xr.DataArray(
        values,
        dims=AXES,
        coords={axis: (axis, c, {"units": "m"}) for axis, c in zip(AXES, coordinates, strict=True)},
        name="scalar_marker",
    )
    mapping = xrf.AffineTransform(
        source=xrf.ArrayCoordinates(AXES, ("m",) * 4),
        target=frame,
        matrix=np.eye(4),
        translation=np.zeros(4),
    )
    return cast(xr.DataArray, array.rf.frame(mapping, dims=AXES))


def _moments(slice_: xr.DataArray) -> tuple[Array, float]:
    weights = np.asarray(slice_.values, dtype=np.float64).reshape(-1)
    # A fill value must never masquerade as empty space in the measurement.
    if not np.isfinite(weights).all():
        raise ValueError("the target extends outside the sampled source domain")
    points = np.stack(np.meshgrid(*(slice_[a].values for a in AXES[1:]), indexing="ij"))
    points = points.reshape(3, -1)
    offsets = points - (points @ weights / weights.sum())[:, None]
    covariance = (offsets * weights) @ offsets.T / weights.sum()
    boundary = np.ones(slice_.shape, dtype=bool)
    boundary[1:-1, 1:-1, 1:-1] = False
    return covariance, float(weights[boundary.reshape(-1)].sum() / weights.sum())


def measure(spatial_samples: int = 63) -> Measurement:
    """Resample the worldtube and measure its ct'=0 slice.

    Args:
        spatial_samples: Samples per source spatial axis, from 17 through 64.
            The target always uses 63 samples per spatial axis.

    Returns:
        Measured tensors in square metres and RMS width ratios in motion order.

    Raises:
        TypeError: If spatial_samples is not an integer.
        ValueError: If spatial_samples is outside [17, 64], or resampling fills
            any target sample instead of locating it inside the source.
    """
    if isinstance(spatial_samples, bool) or not isinstance(spatial_samples, int):
        raise TypeError("spatial_samples must be an integer")
    if not 17 <= spatial_samples <= 64:
        raise ValueError("spatial_samples must be in [17, 64]")
    rest = xrf.ReferenceFrame.declared(("relativity-demo", "rest-3d"), SYSTEM)
    moving = xrf.ReferenceFrame.declared(("relativity-demo", "moving-3d"), SYSTEM)
    boost = np.eye(4)
    boost[0, 0] = GAMMA
    boost[0, 1:] = boost[1:, 0] = -GAMMA * BETA * DIRECTION
    boost[1:, 1:] += (GAMMA - 1.0) * np.outer(DIRECTION, DIRECTION)
    rest_to_moving = xrf.AffineTransform(
        source=rest, target=moving, matrix=boost, translation=TRANSLATION
    )
    moving_to_rest = rest_to_moving.inverse()

    shift = TRANSLATION[1:] + BETA * TRANSLATION[0] * DIRECTION
    target_axes = [np.linspace(-TARGET_LIMIT, TARGET_LIMIT, 63) + b for b in shift]
    corners = np.array(list(product(*((a[0], a[-1]) for a in target_axes))))
    events = moving_to_rest.transform_point(np.column_stack([np.zeros(8), corners]))
    time_range = np.array([events[:, 0].min(), events[:, 0].max()])
    # Pad only roundoff at the time boundary, not missing physical support.
    time_range += np.array([-1.0, 1.0]) * 1e-10
    axis = np.linspace(-SOURCE_LIMIT, SOURCE_LIMIT, spatial_samples, dtype=np.float64)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    body = np.zeros((spatial_samples,) * 3, dtype=np.float64)
    for amplitude, center, width in zip(AMPLITUDES, CENTERS, WIDTHS, strict=True):
        exponent = sum(((p - c) / w) ** 2 for p, c, w in zip((z, y, x), center, width, strict=True))
        body += amplitude * np.exp(-0.5 * exponent)
    source = _framed(np.stack([body, body]), [time_range, axis, axis, axis], rest)
    target = _framed(np.zeros((1, 63, 63, 63)), [np.array([0.0]), *target_axes], moving)
    resampled = source.rf.resample_to(target, transform=moving_to_rest, method="linear")
    rest_covariance, _ = _moments(source.isel(ct=0))
    moving_covariance, boundary_mass = _moments(resampled.isel(ct=0))
    contraction = np.eye(3) + (1.0 / GAMMA - 1.0) * np.outer(DIRECTION, DIRECTION)
    predicted = contraction @ rest_covariance @ contraction.T
    basis = motion_basis()
    ratios = np.sqrt(
        np.diag(basis @ moving_covariance @ basis.T) / np.diag(basis @ rest_covariance @ basis.T)
    )
    return Measurement(
        rest_covariance,
        moving_covariance,
        predicted,
        ratios,
        time_range,
        float(axis[1] - axis[0]),
        boundary_mass,
    )


def main() -> None:
    """Print the measured tensors and contraction ratios at two resolutions."""
    start = perf_counter()
    for samples in (33, 63):
        result = measure(samples)
        print(f"source 2 x {samples}^3, target 1 x 63^3; h={result.source_spacing:.6f} m")
        print(f"rest ct range (m): {result.rest_time_range}")
        print("moving covariance (m^2):\n", result.moving_covariance)
        print("predicted covariance (m^2):\n", result.predicted_covariance)
        print(
            f"max tensor error: {np.max(np.abs(result.moving_covariance - result.predicted_covariance)):.6g} m^2"
        )
        print(f"RMS ratios (parallel, transverse 1, transverse 2): {result.rms_ratios}")
        print(
            f"expected: {(1 / GAMMA, 1.0, 1.0)}; boundary mass: {result.boundary_mass_fraction:.3g}"
        )
    print(f"elapsed: {perf_counter() - start:.3f} s")


if __name__ == "__main__":
    main()
