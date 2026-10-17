"""Longitude seams and pole crossings for resampling an angular source.

Resampling locates target points among source labels, which are literal: 0 and 360 are
different labels. On a longitude/latitude chart the space is not literal, though. Longitude is
periodic, so a global grid continues across its seam, and latitude continues across a pole on
the antipodal meridian. This module decides, from metadata and labels only, which source axes
take part, locates coordinates on them, and extends the source block with halos so that
``map_coordinates`` interpolates across the seam and the poles without wrapping every axis.

The exact periodic location follows ``GridGeometry.periodic_bracket_array`` in xarraybc (a
sibling project for topology-aware indexing), and the pole crossing its ``sphere`` topology:
reflect latitude and shift longitude by half a turn. Unlike xarraybc, coverage is decided rather
than assumed: a regional grid never interpolates across its gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from ._sampling import (
    LATTICE_TOLERANCE,
    POSITION_SLACK,
    AxisSampling,
    Domain,
    Sampling,
    coordinate_to_position,
    singleton_step,
)
from ._transform import SupportsAffine
from ._validation import angular_period

SIGNIFICANCE: float = 1e-9
"""Relative size below which a matrix entry is treated as zero, as in ``_classify_axes``."""

CUBIC_HALO: int = 24
"""Halo for cubic splines: the prefilter's boundary influence decays as 0.268**h, 2e-14 at 24."""

Kind = Literal["global", "regional"]


def _tolerance(labels: npt.NDArray[np.float64], steps: npt.NDArray[np.float64]) -> float:
    """Allowance for label equality: a float32-stored endpoint, but never 1 % of a step."""
    largest = float(np.max(np.abs(labels)))
    float32_rounding = 4.0 * float(np.spacing(np.float32(largest)))
    return min(float32_rounding, 0.01 * float(np.min(steps))) + LATTICE_TOLERANCE * float(
        np.max(steps)
    )


@dataclass(frozen=True)
class Longitude:
    """How one longitude source axis is located and extended."""

    axis: int
    """The source coordinate axis (column of the inverse's output)."""

    kind: Kind
    period: float
    """One turn in source coordinates."""

    labels: npt.NDArray[np.float64]
    """The labels in storage order."""

    count: int
    """Samples taking part (``n_eff``): the duplicate of a closing endpoint is excluded."""

    offset: int
    """Storage index of effective index 0 (1 when a descending grid's duplicate leads)."""

    descending: bool
    low: float
    """Regional only: the lowest coordinate the domain reaches; targets shift into [low, low + period)."""

    step: float | None = None
    """Regional only: the step ``coordinate_to_position`` would use (``singleton_step``)."""

    def locate(
        self, coordinates: npt.NDArray[np.float64], extent: tuple[float, float]
    ) -> npt.NDArray[np.float64]:
        """Positions in effective storage indices; global positions lie in ``[0, count)``."""
        if self.kind == "regional":
            shifted = self.low + np.remainder(coordinates - self.low, self.period)
            return coordinate_to_position(self.labels, shifted, self.step, extent)
        ascending = self.labels[self.offset : self.offset + self.count]
        if self.descending:
            ascending = ascending[::-1]
        origin = float(ascending[0])
        end = origin + self.period
        local = origin + np.remainder(coordinates - origin, self.period)
        # Rounding can land on the excluded end of the half-open cycle.
        local = np.where(local >= end, origin, local)
        left = np.clip(np.searchsorted(ascending, local, side="right") - 1, 0, self.count - 1)
        right_label = np.where(
            left + 1 == self.count, end, ascending[np.minimum(left + 1, self.count - 1)]
        )
        position = left + (local - ascending[left]) / (right_label - ascending[left])
        if self.descending:
            position = np.remainder(self.count - 1 - position, self.count)
        return np.asarray(position, dtype=np.float64)


@dataclass(frozen=True)
class Pole:
    """A mirror across one pole at one end of the latitude labels."""

    at_start: bool
    """Whether the pole lies beyond storage index 0 (else beyond the last index)."""

    odd: bool
    """A sample sits on the pole (not repeated); else the pole lies between a row and its image."""

    center: float
    """The pole in latitude source coordinates."""

    step: float
    """Absolute latitude step at this end; sets the roundoff band beyond the pole."""


@dataclass(frozen=True)
class Latitude:
    """A latitude axis extended across one or both poles."""

    axis: int
    labels: npt.NDArray[np.float64]
    """Extended labels in storage order, halo rows included."""

    before: npt.NDArray[np.intp]
    """Source rows mirrored before storage index 0, in extended order."""

    after: npt.NDArray[np.intp]
    """Source rows mirrored after the last index, in extended order."""

    poles: tuple[Pole, ...]

    def locate(
        self, coordinates: npt.NDArray[np.float64], extent: tuple[float, float]
    ) -> npt.NDArray[np.float64]:
        """Positions among the extended labels; beyond a pole is outside."""
        start_halo = any(pole.at_start for pole in self.poles)
        end_halo = any(not pole.at_start for pole in self.poles)
        ascending = self.labels[-1] > self.labels[0]
        beyond = np.zeros(coordinates.shape, dtype=bool)
        for pole in self.poles:
            # A pole bounds latitude: the halo serves interpolation, not targets past the pole.
            # Within POSITION_SLACK of a step past it, as at every domain edge, a target is
            # roundoff and is located on the pole.
            high_side = pole.at_start != ascending
            past = coordinates - pole.center if high_side else pole.center - coordinates
            band = POSITION_SLACK * pole.step
            coordinates = np.where((past > 0.0) & (past <= band), pole.center, coordinates)
            beyond |= past > band
        before, after = extent
        positions = coordinate_to_position(
            self.labels,
            coordinates,
            None,
            (0.0 if start_halo else before, 0.0 if end_halo else after),
        )
        return np.where(beyond, np.nan, positions)


@dataclass(frozen=True)
class PeriodicAxes:
    """The angular axes of a source that take part in resampling, decided once per plan."""

    longitude: Longitude
    latitude: Latitude | None
    halo: int
    """Longitude halo; latitude halos are capped by the rows available."""

    def extend(
        self,
        volume: npt.NDArray[np.generic],
        longitude_dim: int,
        latitude_dim: int | None,
        sign: float = 1.0,
    ) -> npt.NDArray[np.generic]:
        """Extend one source block (in storage order) by its halos.

        ``sign`` multiplies the rows mirrored across a pole: -1 for a component that reverses
        there (east or north), +1 otherwise.
        """
        lon = self.longitude
        if lon.kind != "global":
            return volume
        h, n = self.halo, lon.count
        columns = lon.offset + np.remainder(np.arange(-h, n + h), n)
        extended = np.take(volume, columns, axis=longitude_dim)
        lat = self.latitude
        if lat is None or latitude_dim is None:
            return extended
        # Across a pole the antipodal meridian continues: shift longitude by half a turn.
        antipodal = np.take(
            volume, lon.offset + np.remainder(columns - lon.offset + n // 2, n), axis=longitude_dim
        )
        before = np.take(antipodal, lat.before, axis=latitude_dim)
        after = np.take(antipodal, lat.after, axis=latitude_dim)
        if sign < 0:
            before, after = _negated(before), _negated(after)
        return np.concatenate([before, extended, after], axis=latitude_dim)

    def locate(
        self, axis: int, coordinates: npt.NDArray[np.float64], extent: tuple[float, float]
    ) -> npt.NDArray[np.float64] | None:
        """Positions in the extended block along ``axis``, or ``None`` if it does not take part."""
        lon = self.longitude
        if axis == lon.axis:
            positions = lon.locate(coordinates, extent)
            return positions + self.halo if lon.kind == "global" else positions
        if self.latitude is not None and axis == self.latitude.axis:
            return self.latitude.locate(coordinates, extent)
        return None

    def extended_size(self, axis: int, size: int) -> int:
        """The extended block's length along a source coordinate axis."""
        if axis == self.longitude.axis and self.longitude.kind == "global":
            return self.longitude.count + 2 * self.halo
        if self.latitude is not None and axis == self.latitude.axis:
            return int(self.latitude.labels.size)
        return size


def _negated(rows: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """Negate halo rows exactly; a signed integer's minimum has no negation in its dtype.

    Unsigned and boolean components are refused before planning.
    """
    if rows.dtype.kind == "i" and rows.size:
        lowest = np.iinfo(rows.dtype.str).min
        if rows.min() == lowest:
            raise ValueError(
                f"a component crossing a pole is negated there, and {rows.dtype} cannot "
                f"represent -({lowest}); convert the components to a wider or floating dtype"
            )
    return np.negative(rows)


def _single_source_axis(matrix: npt.NDArray[np.float64], row: int) -> int | None:
    """The one source axis a frame row depends on, if it feeds no other frame axis."""
    values = np.abs(matrix)
    significant = values > SIGNIFICANCE * values.max(axis=1, keepdims=True)
    columns = np.flatnonzero(significant[row])
    if columns.size != 1 or np.count_nonzero(significant[:, columns[0]]) != 1:
        return None
    return int(columns[0])


def _intervals_tile(intervals: npt.NDArray[np.float64], period: float, tolerance: float) -> bool:
    bounds = np.sort(intervals.reshape(-1, 2), axis=1)
    bounds = bounds[np.argsort(bounds[:, 0])]
    gaps = bounds[1:, 0] - bounds[:-1, 1]
    closing = bounds[0, 0] + period - bounds[-1, 1]
    return bool(np.all(np.abs(gaps) <= tolerance) and abs(closing) <= tolerance)


def _longitude(
    sampling: AxisSampling, axis: int, period: float, extent: tuple[float, float], domain: Domain
) -> Longitude:
    labels = np.asarray(sampling.values, dtype=np.float64)
    n = labels.size
    descending = n > 1 and labels[-1] < labels[0]
    differences = np.diff(labels)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        # Checked here because coverage is computed from the steps before any location.
        raise ValueError(
            "coordinate values are not strictly monotonic, so a coordinate does not determine "
            "a unique position",
        )

    def regional() -> Longitude:
        step = singleton_step(sampling, domain)
        reach = _lowest_reach(labels, extent, step)
        return Longitude(axis, "regional", period, labels, n, 0, descending, reach, step)

    if n < 2:
        return regional()
    ascending = labels[::-1] if descending else labels
    steps = np.diff(ascending)
    tolerance = _tolerance(labels, steps)
    seam = float(ascending[0] + period - ascending[-1])
    if sampling.intervals is not None:
        if not _intervals_tile(np.asarray(sampling.intervals), period, tolerance):
            return regional()
        return Longitude(axis, "global", period, labels, n, 0, descending, 0.0)
    if abs(seam) <= tolerance and n > 2:
        # A closing duplicate: the last ascending label repeats the first one turn later.
        return Longitude(
            axis, "global", period, labels, n - 1, 1 if descending else 0, descending, 0.0
        )
    if tolerance < seam <= float(np.max(steps)) + tolerance:
        return Longitude(axis, "global", period, labels, n, 0, descending, 0.0)
    return regional()


def _lowest_reach(
    labels: npt.NDArray[np.float64], extent: tuple[float, float], step: float | None
) -> float:
    """The lowest coordinate a domain reaches: the outer cell edge below the smallest label."""
    if labels.size == 1:
        # A lone sample has a step only from its declared interval.
        return float(labels[0]) - (extent[0] * abs(step) if step is not None else 0.0)
    before, after = extent
    first_step = float(labels[1] - labels[0])
    last_step = float(labels[-1] - labels[-2])
    edges = (float(labels[0]) - before * first_step, float(labels[-1]) + after * last_step)
    return min(edges)


def _antipodal(lon: Longitude) -> bool:
    if lon.kind != "global" or lon.count % 2:
        return False
    labels = lon.labels[lon.offset : lon.offset + lon.count]
    steps = np.abs(np.diff(labels))
    tolerance = _tolerance(labels, steps)
    shifted = np.roll(labels, -(lon.count // 2))
    difference = np.remainder(shifted - labels - lon.period / 2, lon.period)
    difference = np.minimum(difference, lon.period - difference)
    return bool(np.all(difference <= tolerance))


def _latitude(
    sampling: AxisSampling, axis: int, poles: tuple[float, float], halo: int
) -> Latitude | None:
    labels = np.asarray(sampling.values, dtype=np.float64)
    n = labels.size
    if n < 2:
        return None
    steps = np.abs(np.diff(labels))
    tolerance = _tolerance(labels, steps)
    widest = float(np.max(steps))
    found: list[Pole] = []
    before = np.empty(0, dtype=np.intp)
    after = np.empty(0, dtype=np.intp)
    for at_start in (True, False):
        end = float(labels[0] if at_start else labels[-1])
        for center in poles:
            gap = abs(center - end)
            odd = gap <= tolerance
            if not odd and not (tolerance < 2.0 * gap <= widest + tolerance):
                continue
            # The pole must lie outside the labels, beyond this end.
            inward = float(labels[1] - labels[0]) if at_start else float(labels[-2] - labels[-1])
            if not odd and (center - end) * inward > 0:
                continue
            rows = min(halo, n - 1 if odd else n)
            # Rows nearest the pole first, counted inward from this end.
            nearest = np.arange(1, rows + 1) if odd else np.arange(rows)
            source_rows = nearest if at_start else n - 1 - nearest
            if at_start:
                before = source_rows[::-1].astype(np.intp)
            else:
                after = source_rows.astype(np.intp)
            found.append(Pole(at_start, odd, center, abs(inward)))
            break
    if not found:
        return None
    mirrored_before = np.array([2.0 * _center(found, True) - labels[i] for i in before])
    mirrored_after = np.array([2.0 * _center(found, False) - labels[i] for i in after])
    extended = np.concatenate((mirrored_before, labels, mirrored_after))
    return Latitude(axis, extended, before, after, tuple(found))


def _center(poles: list[Pole], at_start: bool) -> float:
    for pole in poles:
        if pole.at_start == at_start:
            return pole.center
    return 0.0  # unused: no rows are mirrored at an end without a pole


def periodic_axes(
    source: Sampling,
    *,
    halo: int,
    pole_halos: bool,
    extents: dict[int, tuple[float, float]],
    domain: Domain,
) -> PeriodicAxes | None:
    """Decide which axes of an angular source wrap or cross a pole; ``None`` when none do.

    ``extents`` gives, per source coordinate axis, the domain's reach beyond the outer labels in
    positions (``reach`` in ``_positions``).
    """
    system = source.frame.coordinate_system
    roles = system.axis_roles
    if "longitude" not in roles or not isinstance(source.transform, SupportsAffine):
        return None
    matrix = np.asarray(source.transform.matrix, dtype=np.float64)
    translation = np.asarray(source.transform.translation, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        return None
    lon_row = roles.index("longitude")
    lon_axis = _single_source_axis(matrix, lon_row)
    axes = source.axes
    if lon_axis is None or axes[lon_axis].dim is None:
        return None
    lon_unit = system.units[lon_row]
    assert lon_unit is not None  # angular roles declare a unit
    period = angular_period(lon_unit) / abs(float(matrix[lon_row, lon_axis]))
    longitude = _longitude(
        axes[lon_axis], lon_axis, period, extents.get(lon_axis, (0.0, 0.0)), domain
    )
    latitude = None
    if pole_halos and _antipodal(longitude):
        lat_row = roles.index("latitude")
        lat_axis = _single_source_axis(matrix, lat_row)
        if lat_axis is not None and axes[lat_axis].dim is not None:
            lat_unit = system.units[lat_row]
            assert lat_unit is not None
            quarter = angular_period(lat_unit) / 4.0
            scale = float(matrix[lat_row, lat_axis])
            offset = float(translation[lat_row])
            poles = ((quarter - offset) / scale, (-quarter - offset) / scale)
            latitude = _latitude(axes[lat_axis], lat_axis, poles, halo)
    return PeriodicAxes(longitude, latitude, halo)
