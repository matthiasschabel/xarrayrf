"""Coordinate systems: the axes in which a reference frame's coordinates are expressed."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

from ._validation import check_axis_types, check_names, check_str, check_units, real_float_array
from ._vocabulary import DirectionVocabulary

CARTESIAN: Final = "cartesian"
"""The only coordinate representation implemented so far."""

_REPRESENTATIONS: Final = frozenset({CARTESIAN})
# Adding a representation must revisit the angular-unit refusal below, ``axis_codes`` and
# ``coordinate_system_change``, which assume Cartesian axes because nothing else exists.


class AxisCode(NamedTuple):
    """How closely a vector follows one oriented axis."""

    axis: str
    """The axis name."""

    direction: str
    """The axis's direction token, or its opposite when the vector points the other way."""

    angle: float
    """Angle in radians between the vector and ``direction``; 0 is exact alignment."""


class CoordinateSystem:
    """Ordered axes with units, a representation and, optionally, axis orientation.

    A coordinate system carries no identity. The same reference frame can be expressed in
    several coordinate systems, for example LPS and RAS for one patient, and a frame compared
    with :meth:`~xarrayrf.ReferenceFrame.is_equivalent_frame` ignores which one it uses.

    Orientation assigns an axis the direction it increases toward, taken from a
    :class:`~xarrayrf.DirectionVocabulary`. Axes may be left unoriented. Orientation is what
    lets :func:`~xarrayrf.coordinate_system_change` derive a conversion and
    :meth:`axis_codes` name a vector's direction; axis names never
    are, so ``x`` and ``L`` can be the same axis.
    """

    __slots__ = ("_axes", "_axis_types", "_orientation", "_representation", "_units", "_vocabulary")

    _axes: tuple[str, ...]
    _units: tuple[str | None, ...]
    _axis_types: tuple[str | None, ...]
    _representation: str
    _vocabulary: DirectionVocabulary | None
    _orientation: tuple[str | None, ...]

    def __init__(
        self,
        axes: Sequence[str],
        units: Sequence[str | None],
        *,
        axis_types: Sequence[str | None] | None = None,
        representation: str = CARTESIAN,
        vocabulary: DirectionVocabulary | None = None,
        orientation: Sequence[str | None] | None = None,
    ) -> None:
        """Validate and freeze a coordinate system.

        Args:
            axes: Ordered axis names, unique and non-empty.
            units: One unit string per axis, compared exactly, or ``None`` where no unit is
                declared. ``None`` is not ``"1"`` (dimensionless) and implies nothing else.
            axis_types: One open type string per axis, or ``None`` where undeclared; omitted,
                none is declared. NGFF's ``"space"``, ``"time"`` and ``"channel"`` are the
                recommended spellings. Declarative only: nothing here depends on a type.
            representation: Coordinate representation kind; ``"cartesian"`` so far.
            vocabulary: The vocabulary orientation tokens come from. Required when any axis
                is oriented, and discarded when none is.
            orientation: One direction token or ``None`` per axis, naming the direction the
                axis increases toward. Omitted means no axis is oriented.

        Raises:
            TypeError: If an argument has the wrong Python type.
            ValueError: If names repeat, counts disagree, a unit is malformed or angular on
                a Cartesian axis, an axis type is empty or padded, the representation is
                unsupported, orientation is given without a vocabulary, a token is not in the
                vocabulary, two axes use the same antipodal pair, or an oriented axis declares
                no unit.
        """
        self._axes = check_names(axes, field="axes")
        self._representation = check_str(representation, field="representation")
        if self._representation not in _REPRESENTATIONS:
            raise ValueError(
                f"representation {self._representation!r} is not implemented; supported "
                f"representations are {tuple(sorted(_REPRESENTATIONS))}",
            )
        self._units = check_units(units, field="units", reject_angular=True)
        if len(self._units) != len(self._axes):
            raise ValueError(
                f"units must declare one unit per axis; got {len(self._units)} units for "
                f"{len(self._axes)} axes",
            )
        self._axis_types = check_axis_types(axis_types, count=len(self._axes))
        if vocabulary is not None and not isinstance(vocabulary, DirectionVocabulary):
            raise TypeError(
                f"vocabulary must be a DirectionVocabulary, got {type(vocabulary).__name__}",
            )
        self._vocabulary = vocabulary
        self._orientation = self._check_orientation(orientation)
        if all(token is None for token in self._orientation):
            # A vocabulary with no oriented axis carries no meaning; dropping it keeps two
            # unoriented systems equal whether or not a caller passed one.
            self._vocabulary = None

    def _check_orientation(self, orientation: object) -> tuple[str | None, ...]:
        if orientation is None:
            return (None,) * len(self._axes)
        if isinstance(orientation, str) or not isinstance(orientation, Sequence):
            raise TypeError(
                "orientation must be a sequence of direction tokens or None, got "
                f"{type(orientation).__name__}",
            )
        entries = tuple(orientation)
        if len(entries) != len(self._axes):
            raise ValueError(
                f"orientation must give one entry per axis; got {len(entries)} for "
                f"{len(self._axes)} axes",
            )
        if any(entry is not None for entry in entries) and self._vocabulary is None:
            raise ValueError("orientation tokens require the vocabulary they come from")
        pairs: dict[frozenset[str], str] = {}
        checked: list[str | None] = []
        for index, (axis, entry) in enumerate(zip(self._axes, entries, strict=True)):
            if entry is None:
                checked.append(None)
                continue
            assert self._vocabulary is not None  # established by the check above
            if self._units[index] is None:
                raise ValueError(
                    f"axis {axis!r} is oriented but declares no unit; a direction needs the "
                    "axis's quantity, so declare its unit",
                )
            token = self._vocabulary.check(entry)
            pair = self._vocabulary.pair(token)
            if pair in pairs:
                raise ValueError(
                    f"axes {pairs[pair]!r} and {axis!r} are both oriented along "
                    f"{sorted(pair)}; each axis needs its own direction pair",
                )
            pairs[pair] = axis
            checked.append(token)
        return tuple(checked)

    @property
    def axes(self) -> tuple[str, ...]:
        """Ordered axis names."""
        return self._axes

    @property
    def units(self) -> tuple[str | None, ...]:
        """One unit string per axis, ``None`` where no unit is declared."""
        return self._units

    @property
    def axis_types(self) -> tuple[str | None, ...]:
        """One type string per axis, ``None`` where undeclared."""
        return self._axis_types

    @property
    def representation(self) -> str:
        """Coordinate representation kind."""
        return self._representation

    @property
    def vocabulary(self) -> DirectionVocabulary | None:
        """The vocabulary orientation tokens come from; ``None`` when no axis is oriented."""
        return self._vocabulary

    @property
    def orientation(self) -> tuple[str | None, ...]:
        """One direction token or ``None`` per axis."""
        return self._orientation

    def axis_codes(self, vector: npt.ArrayLike) -> tuple[AxisCode, ...]:
        """Rank the oriented axes by how closely they follow a vector.

        The reverse of reading a direction off an axis, in the manner of nibabel's
        ``aff2axcodes``: which named directions does this vector point along? The first entry
        is the nearest; its angle is the residual that a label such as "anterior" would hide.
        Reading only ``codes[0].direction`` discards that residual and reproduces
        ``aff2axcodes``, which labels a vector however oblique it is and breaks a 45° tie by axis
        order. A caller that needs one label must bound ``codes[0].angle`` itself. A bound below
        45° (``math.pi / 4``) also makes the label unique, since across orthonormal axes at most
        one angle can be under 45°.
        Angles are measured within the oriented axes only. Components along unoriented axes,
        such as time, may be in other units, so they are ignored rather than mixed into a norm.

        Args:
            vector: Finite real values, one per axis in this system's order, nonzero along
                some oriented axis.

        Returns:
            One code per oriented axis, nearest first; ties keep axis order. A value of zero
            along an axis reports the axis's own direction at a right angle.

        Raises:
            TypeError: If ``vector`` has a non-real dtype or is a masked array.
            ValueError: If the system has no oriented axis, or its oriented
                axes use different units, or ``vector`` has the wrong length, is not finite,
                or is zero along every oriented axis.
        """
        vocabulary = self._vocabulary
        oriented = [(i, token) for i, token in enumerate(self._orientation) if token is not None]
        if vocabulary is None or not oriented:
            raise ValueError("the coordinate system has no oriented axis to report")
        units = {self._units[i] for i, _ in oriented}
        if len(units) != 1:
            raise ValueError(
                f"oriented axes use different units {sorted(map(str, units))}; angles between them are "
                "not meaningful",
            )
        values = real_float_array(vector, field="vector")
        if values.shape != (len(self._axes),):
            raise ValueError(
                f"vector must have one value per axis {self._axes}, got shape {values.shape}",
            )
        norm = float(np.linalg.norm([values[i] for i, _ in oriented]))
        if norm == 0.0:
            raise ValueError(
                "the vector is zero along every oriented axis, so it has no direction to report",
            )
        codes = []
        for i, token in oriented:
            cosine = float(values[i]) / norm
            direction = token if cosine >= 0.0 else vocabulary.opposite(token)
            codes.append(AxisCode(self._axes[i], direction, math.acos(min(1.0, abs(cosine)))))
        return tuple(sorted(codes, key=lambda code: code.angle))

    def _key(self) -> tuple[object, ...]:
        return (
            self._axes,
            self._units,
            self._axis_types,
            self._representation,
            self._vocabulary,
            self._orientation,
        )

    def __eq__(self, other: object) -> bool:
        """Compare axes, units, axis types, representation, vocabulary and orientation exactly."""
        if not isinstance(other, CoordinateSystem):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        """Hash consistently with :meth:`__eq__`."""
        return hash((CoordinateSystem, self._key()))

    def __repr__(self) -> str:
        """Return a representation naming axes, units and orientation."""
        text = f"CoordinateSystem(axes={self._axes!r}, units={self._units!r}"
        if any(kind is not None for kind in self._axis_types):
            text += f", axis_types={self._axis_types!r}"
        if self._representation != CARTESIAN:
            text += f", representation={self._representation!r}"
        if self._vocabulary is not None:
            text += (
                f", vocabulary={self._vocabulary.identifier!r}, orientation={self._orientation!r}"
            )
        return text + ")"
