"""Array coordinates: an array's own coordinate values, as a transform endpoint."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ._validation import check_axis_types, check_names, check_units


def _check_sample_offsets(value: object, *, count: int) -> tuple[float | None, ...]:
    """Validate one sample offset per axis: ``None``, or a real number in ``[0, 1]``."""
    if value is None:
        return (None,) * count
    if isinstance(value, np.ma.MaskedArray):
        raise TypeError("sample_offset must not be a masked array; use None for a point sample")
    if isinstance(value, np.ndarray) and value.ndim == 1:
        value = value.tolist()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            f"sample_offset must be a sequence of numbers or None, got {type(value).__name__}",
        )
    if len(value) != count:
        raise ValueError(
            f"sample_offset must declare one offset per axis; got {len(value)} for {count} axes",
        )
    offsets: list[float | None] = []
    for offset in value:
        if offset is None:
            offsets.append(None)
            continue
        if isinstance(offset, bool) or not isinstance(
            offset, int | float | np.integer | np.floating
        ):
            raise TypeError(
                f"a sample offset must be a real number or None, got {type(offset).__name__}",
            )
        if not (math.isfinite(float(offset)) and 0.0 <= float(offset) <= 1.0):
            raise ValueError(f"a sample offset must lie in [0, 1], got {offset!r}")
        offsets.append(float(offset))
    return tuple(offsets)


class ArrayCoordinates:
    """The coordinate values of an array along named axes, with units and no identity.

    This is the endpoint a transform starts from when it locates an array's samples: the NGFF
    array coordinate system, napari's data coordinates. Axis names are the names of the
    array's coordinates a transform reads, which may be dimension coordinates or auxiliary
    ones, such as a slice offset along a slice dimension. Coordinates are values, not
    positions: at import they usually equal positions, and after a crop xarray keeps them, so
    a transform from array coordinates stays valid for the cropped array.

    A coordinate value is where its sample is. Whether each sample stands for a cell (a voxel,
    a time bin) and where it sits in that cell is a separate declaration, ``sample_offset``:
    per axis, ``None`` for point samples with no cell, or the sample's fractional place in its
    cell in position units, measured from the cell's edge at lower coordinate values: the cell of
    position ``p`` spans positions ``p - s`` to ``p + 1 - s`` (reversed where coordinates
    descend), mapped to coordinates as positions are, piecewise linearly for nonuniform values.
    ``0.5`` is a centred voxel,
    as DICOM, NIfTI, ITK and NGFF specify; ``0`` and ``1`` are cells labelled at their start
    and end. Measuring by coordinate value rather than by position keeps the declaration true
    when an array is reversed, cropped or strided. The cells themselves are recomputed from the
    current samples, so striding widens them; they describe the current sampling, not a region
    fixed to each element. xarray operations that recompute coordinates, such as ``coarsen``,
    keep only a centred offset true. Physical slice thickness and gaps between slices are not
    described here.

    Array coordinates carry no identity. Two arrays whose coordinates have the same names,
    units, axis types and sample offsets compare equal, which is why a chain of transforms may begin or
    end at array coordinates but never pass through them.
    """

    __slots__ = ("_axes", "_axis_types", "_sample_offset", "_units")

    _axes: tuple[str, ...]
    _units: tuple[str | None, ...]
    _axis_types: tuple[str | None, ...]
    _sample_offset: tuple[float | None, ...]

    def __init__(
        self,
        axes: Sequence[str],
        units: Sequence[str | None],
        *,
        axis_types: Sequence[str | None] | None = None,
        sample_offset: (
            Sequence[float | None] | npt.NDArray[np.floating[Any] | np.integer[Any]] | None
        ) = None,
    ) -> None:
        """Validate and freeze array coordinates.

        Args:
            axes: Ordered coordinate names, unique and non-empty.
            units: One unit string per axis, compared exactly, or ``None`` where no unit is
                declared.
            axis_types: One open type string per axis, or ``None`` where undeclared, as for
                :class:`~xarrayrf.CoordinateSystem`.
            sample_offset: One entry per axis: ``None`` for point samples, or where each
                sample sits in its cell as a fraction in ``[0, 1]`` from the cell's lower
                coordinate edge. Omitted, every axis is point-sampled.

        Raises:
            TypeError: If an argument is not a sequence of strings, or an offset is not a real
                number or ``None``.
            ValueError: If names repeat, a unit is empty or padded, the counts differ, or an
                offset lies outside ``[0, 1]``.
        """
        self._axes = check_names(axes, field="axes")
        self._units = check_units(units, field="units")
        if len(self._units) != len(self._axes):
            raise ValueError(
                f"units must declare one unit per axis; got {len(self._units)} units for "
                f"{len(self._axes)} axes",
            )
        self._axis_types = check_axis_types(axis_types, count=len(self._axes))
        self._sample_offset = _check_sample_offsets(sample_offset, count=len(self._axes))

    @property
    def axes(self) -> tuple[str, ...]:
        """Ordered coordinate names."""
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
    def sample_offset(self) -> tuple[float | None, ...]:
        """Per axis, ``None`` for point samples or the sample's fractional place in its cell."""
        return self._sample_offset

    def __eq__(self, other: object) -> bool:
        """Compare axes, units, axis types and sample offsets exactly."""
        if not isinstance(other, ArrayCoordinates):
            return NotImplemented
        return self._key() == other._key()

    def _key(self) -> tuple[object, ...]:
        return (self._axes, self._units, self._axis_types, self._sample_offset)

    def __hash__(self) -> int:
        """Hash consistently with :meth:`__eq__`."""
        return hash((ArrayCoordinates, self._key()))

    def __repr__(self) -> str:
        """Return a representation naming axes and units, and any declared sample offsets."""
        offsets = (
            ""
            if all(offset is None for offset in self._sample_offset)
            else f", sample_offset={self._sample_offset!r}"
        )
        types = (
            ""
            if all(kind is None for kind in self._axis_types)
            else f", axis_types={self._axis_types!r}"
        )
        return f"ArrayCoordinates(axes={self._axes!r}, units={self._units!r}{types}{offsets})"
