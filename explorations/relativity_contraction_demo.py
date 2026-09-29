"""Measure Lorentz contraction and clock dilation in sampled spacetime fields.

Run with ``PYTHONPATH=src python explorations/relativity_contraction_demo.py``.
Coordinates use natural units: both ``ct`` and ``x`` are in metres, so speeds
are fractions of ``c`` and a clock interval is a ``c Δt`` distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import xarray as xr

import xarrayrf as xrf
import xarrayrf.native  # Register the public DataArray .rf accessor.

type Array = npt.NDArray[np.float64]

AXES = ("ct", "x")
SYSTEM = xrf.CoordinateSystem(AXES, ("m", "m"))
FEATURES = ("body", "early_tick", "late_tick")
CENTERS = np.array([-1.40, 0.05, 1.15])
WIDTHS = np.array([0.34, 0.58, 0.27])
AMPLITUDES = np.array([0.95, 0.42, 0.75])
TICK_PROPER_TIMES = (-2.0, 2.0)
TICK_SPATIAL_WIDTH = 0.25
TICK_TEMPORAL_WIDTH = 0.25
TRANSLATION = (0.35, -0.25)


@dataclass(frozen=True)
class Measurement:
    """Measured and predicted lengths and tick intervals for one sampling choice."""

    object_speed: float
    observer_speed: float
    spacing: float
    method: str
    proper_length: float
    observed_length: float
    predicted_length: float
    proper_tick_interval: float
    observed_tick_interval: float
    predicted_tick_interval: float

    @property
    def length_error(self) -> float:
        """Return observed minus predicted contracted length, in metres."""
        return self.observed_length - self.predicted_length

    @property
    def tick_error(self) -> float:
        """Return observed minus predicted dilated tick interval, in metres of ct."""
        return self.observed_tick_interval - self.predicted_tick_interval


def _axis(limit: float, spacing: float) -> Array:
    count = round(2 * limit / spacing)
    return np.linspace(-limit, limit, count + 1, dtype=np.float64)


def _frame(name: str) -> xrf.ReferenceFrame:
    return xrf.ReferenceFrame.declared(("inertial", name), SYSTEM)


def _boost(
    source: xrf.ReferenceFrame,
    target: xrf.ReferenceFrame,
    speed: float,
    translation: tuple[float, float] = (0.0, 0.0),
) -> xrf.AffineTransform:
    gamma = 1.0 / np.sqrt(1.0 - speed**2)
    return xrf.AffineTransform(
        source=source,
        target=target,
        matrix=np.array([[gamma, -gamma * speed], [-gamma * speed, gamma]]),
        translation=np.array(translation),
    )


def _coordinate_transform(frame: xrf.ReferenceFrame) -> xrf.AffineTransform:
    return xrf.AffineTransform(
        source=xrf.ArrayCoordinates(AXES, ("m", "m")),
        target=frame,
        matrix=np.eye(2),
        translation=np.zeros(2),
    )


def _body_profile(rest_x: Array) -> Array:
    components = AMPLITUDES[:, None, None] * np.exp(
        -0.5 * ((rest_x[None, ...] - CENTERS[:, None, None]) / WIDTHS[:, None, None]) ** 2
    )
    result: Array = components.sum(axis=0)
    return result


def _source(spacing: float, speed: float, frame: xrf.ReferenceFrame) -> xr.DataArray:
    ct_axis = _axis(11.0, spacing)
    x_axis = _axis(11.0, spacing)
    ct, x = np.meshgrid(ct_axis, x_axis, indexing="ij")
    gamma = 1.0 / np.sqrt(1.0 - speed**2)
    proper_time = gamma * (ct - speed * x)
    rest_x = gamma * (x - speed * ct)
    body = _body_profile(rest_x)
    clock_space = np.exp(-0.5 * (rest_x / TICK_SPATIAL_WIDTH) ** 2)
    ticks = [
        clock_space * np.exp(-0.5 * ((proper_time - tick) / TICK_TEMPORAL_WIDTH) ** 2)
        for tick in TICK_PROPER_TIMES
    ]
    field = xr.DataArray(
        np.stack([body, *ticks]),
        dims=("feature", "ct", "x"),
        coords={"feature": list(FEATURES), "ct": ct_axis, "x": x_axis},
        name="object_and_clock",
    )
    return cast(xr.DataArray, field.rf.frame(_coordinate_transform(frame), dims=AXES))


def _target(spacing: float, frame: xrf.ReferenceFrame) -> xr.DataArray:
    ct_axis = _axis(5.0, spacing)
    x_axis = _axis(5.0, spacing)
    grid = xr.DataArray(
        np.zeros((ct_axis.size, x_axis.size), dtype=np.float64),
        dims=AXES,
        coords={"ct": ct_axis, "x": x_axis},
    )
    return cast(xr.DataArray, grid.rf.frame(_coordinate_transform(frame), dims=AXES))


def _length_and_ticks(field: xr.DataArray) -> tuple[float, float]:
    """Use a spatial second moment and each labelled clock pulse's time centroid."""
    x = np.asarray(field.coords["x"].values, dtype=np.float64)
    ct = np.asarray(field.coords["ct"].values, dtype=np.float64)
    at_zero = int(np.argmin(np.abs(ct)))
    if abs(ct[at_zero]) > 1e-12:
        raise ValueError("the target grid must contain ct=0 for the length measurement")
    values = np.asarray(field.values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("the target grid extends outside the sampled source domain")
    density = np.maximum(values[0, at_zero], 0.0)
    center = np.sum(x * density) / np.sum(density)
    variance = np.sum((x - center) ** 2 * density) / np.sum(density)
    length = 2.0 * np.sqrt(variance)
    tick_times = []
    for pulse in values[1:]:
        weights = np.maximum(pulse, 0.0)
        tick_times.append(float(np.sum(ct[:, None] * weights) / np.sum(weights)))
    return float(length), tick_times[1] - tick_times[0]


def analytic_proper_length() -> float:
    """Return twice the standard deviation of the defined rest profile."""
    masses = AMPLITUDES * WIDTHS
    center = float(np.sum(masses * CENTERS) / np.sum(masses))
    variance = float(np.sum(masses * (WIDTHS**2 + (CENTERS - center) ** 2)) / np.sum(masses))
    return float(2.0 * np.sqrt(variance))


def measure(
    spacing: float, method: str, *, object_speed: float, observer_speed: float = 0.6
) -> Measurement:
    """Resample one asymmetric object into its rest and observer frames.

    The object's worldtube is created in frame A. If it moves in A, the rest
    frame C is obtained by a second, independent resampling of that field.
    The B target's ``ct`` coordinates are B times; its row at zero is therefore
    a simultaneous B slice, even though the corresponding A events have
    different A times.
    """
    lab, rest, observer = _frame("A"), _frame("C"), _frame("B")
    source = _source(spacing, object_speed, lab)
    rest_grid = _target(spacing, rest)
    observer_grid = _target(spacing, observer)
    rest_to_lab = _boost(lab, rest, object_speed).inverse()
    observer_to_lab = _boost(lab, observer, observer_speed, TRANSLATION).inverse()
    rest_field = source.rf.resample_to(rest_grid, transform=rest_to_lab, method=method)
    observed = source.rf.resample_to(observer_grid, transform=observer_to_lab, method=method)
    proper_length, proper_ticks = _length_and_ticks(rest_field)
    observed_length, observed_ticks = _length_and_ticks(observed)
    relative_speed = (object_speed - observer_speed) / (1.0 - object_speed * observer_speed)
    relative_gamma = 1.0 / np.sqrt(1.0 - relative_speed**2)
    return Measurement(
        object_speed=object_speed,
        observer_speed=observer_speed,
        spacing=spacing,
        method=method,
        proper_length=proper_length,
        observed_length=observed_length,
        predicted_length=proper_length / relative_gamma,
        proper_tick_interval=proper_ticks,
        observed_tick_interval=observed_ticks,
        predicted_tick_interval=proper_ticks * relative_gamma,
    )


def main() -> None:
    """Print measured versus predicted observables and signed errors."""
    print("Natural units: ct and x in m; length = 2 * scalar-profile standard deviation")
    print(f"Analytic proper length: {analytic_proper_length():.8f} m; tick interval: 4 m of ct")
    print("u_A  h     method   L_B       L_pred    error_L     Δct_B     Δct_pred  error_ct")
    for object_speed in (0.0, 0.25):
        for spacing in (0.2, 0.1, 0.05):
            for method in ("nearest", "linear", "cubic"):
                result = measure(spacing, method, object_speed=object_speed)
                print(
                    f"{object_speed:3.2f} {spacing:4.2f} {method:7s} "
                    f"{result.observed_length:9.6f} {result.predicted_length:9.6f} "
                    f"{result.length_error:+10.3e} {result.observed_tick_interval:9.6f} "
                    f"{result.predicted_tick_interval:9.6f} {result.tick_error:+10.3e}"
                )


if __name__ == "__main__":
    main()
