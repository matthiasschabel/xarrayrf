"""Packing symmetric matrices into ``n(n+1)/2`` components, in an explicit order and scale.

Conventions differ in two independent ways: which matrix entry each packed component holds,
and how off-diagonal entries are scaled (1 for Voigt stress, 2 for Voigt engineering strain,
``sqrt(2)`` for Mandel). A :class:`SymmetricPacking` states both, so nothing is inferred from a
convention's name.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from ._tensor_type import _value_array

SYMMETRY_TOLERANCE: Final = 1e-6
"""Largest ``|T[i, j] - T[j, i]|``, relative to the largest finite ``|T|`` of the matrix, that
:meth:`SymmetricPacking.pack` accepts as symmetric. Matches
:data:`~xarrayrf.ORTHOGONALITY_TOLERANCE`."""

_VOIGT_PAIRS: Final = {
    2: ((0, 0), (1, 1), (0, 1)),
    3: ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)),
}


class SymmetricPacking:
    """The order and off-diagonal scale of a packed symmetric matrix.

    Packed component ``k`` holds matrix entry ``index_pairs[k]``; off-diagonal components are
    the entry times ``off_diagonal_scale``, diagonal ones the entry itself. Each unordered pair
    of indices appears once, so an ``n x n`` matrix packs into ``n(n+1)/2`` components. The
    orientation of a pair names its label: ``(2, 1)`` over axes ``("x", "y", "z")`` is ``"zy"``.

    Equality and hashing are exact over the pairs and the scale.

    Example:
        Mandel packing keeps the Frobenius norm:

        >>> import numpy as np
        >>> import xarrayrf as xrf
        >>> packing = xrf.SymmetricPacking.mandel(2)
        >>> packing.index_pairs
        ((0, 0), (1, 1), (0, 1))
        >>> packing.pack(np.array([[1.0, 2.0], [2.0, 3.0]])).round(4)
        array([1.    , 3.    , 2.8284])
    """

    __slots__ = ("_dimension", "_index_pairs", "_off_diagonal_scale")

    _index_pairs: tuple[tuple[int, int], ...]
    _off_diagonal_scale: float
    _dimension: int

    def __init__(
        self, index_pairs: Sequence[tuple[int, int]], *, off_diagonal_scale: float = 1.0
    ) -> None:
        """Validate and freeze a packing.

        Args:
            index_pairs: The matrix entry ``(i, j)`` each packed component holds, in packed
                order. Every unordered pair over ``0 .. n-1`` must appear exactly once.
            off_diagonal_scale: The factor multiplying off-diagonal entries; finite and
                positive.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If the pairs repeat, miss an entry, or the scale is not finite and
                positive.
        """
        self._index_pairs = _check_pairs(index_pairs)
        self._dimension = 1 + max(max(pair) for pair in self._index_pairs)
        expected = self._dimension * (self._dimension + 1) // 2
        if len(self._index_pairs) != expected:
            raise ValueError(
                f"index_pairs must cover every entry of the upper triangle once: a "
                f"{self._dimension} x {self._dimension} matrix packs into {expected} components, "
                f"got {len(self._index_pairs)}",
            )
        if isinstance(off_diagonal_scale, bool) or not isinstance(off_diagonal_scale, int | float):
            raise TypeError(
                f"off_diagonal_scale must be a real number, got {type(off_diagonal_scale).__name__}",
            )
        scale = float(off_diagonal_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"off_diagonal_scale must be finite and positive, got {scale!r}")
        self._off_diagonal_scale = scale

    @classmethod
    def voigt(cls, dimension: int, *, off_diagonal_scale: float = 1.0) -> SymmetricPacking:
        """Voigt order: ``11, 22, 33, 23, 13, 12`` in 3-D, ``11, 22, 12`` in 2-D.

        The scale is 1 for stress-like tensors and 2 for engineering shear strain. Software
        differs on the order under the same name (Abaqus uses ``11, 22, 33, 12, 13, 23``); give
        explicit ``index_pairs`` for another order.

        Raises:
            ValueError: If ``dimension`` is not 2 or 3.
        """
        if dimension not in _VOIGT_PAIRS:
            raise ValueError(
                f"Voigt order is defined for dimension 2 or 3, got {dimension!r}; use "
                "upper_triangular() or explicit index_pairs",
            )
        return cls(_VOIGT_PAIRS[dimension], off_diagonal_scale=off_diagonal_scale)

    @classmethod
    def mandel(cls, dimension: int) -> SymmetricPacking:
        """Mandel (Kelvin) packing: Voigt order, off-diagonals scaled by ``sqrt(2)``.

        The packed vector's Euclidean norm equals the matrix's Frobenius norm.
        """
        return cls.voigt(dimension, off_diagonal_scale=math.sqrt(2.0))

    @classmethod
    def upper_triangular(
        cls, dimension: int, *, off_diagonal_scale: float = 1.0
    ) -> SymmetricPacking:
        """The upper triangle row by row (``numpy.triu_indices``): ``11, 12, 13, 22, 23, 33``."""
        rows, columns = np.triu_indices(_check_dimension(dimension))
        return cls(
            tuple(zip(rows.tolist(), columns.tolist(), strict=True)),
            off_diagonal_scale=off_diagonal_scale,
        )

    @classmethod
    def lower_triangular(
        cls, dimension: int, *, off_diagonal_scale: float = 1.0
    ) -> SymmetricPacking:
        """The lower triangle row by row (NIfTI ``SYM_MATRIX``): ``11, 21, 22, 31, 32, 33``."""
        rows, columns = np.tril_indices(_check_dimension(dimension))
        return cls(
            tuple(zip(rows.tolist(), columns.tolist(), strict=True)),
            off_diagonal_scale=off_diagonal_scale,
        )

    @property
    def index_pairs(self) -> tuple[tuple[int, int], ...]:
        """The matrix entry each packed component holds, in packed order."""
        return self._index_pairs

    @property
    def off_diagonal_scale(self) -> float:
        """The factor multiplying off-diagonal entries."""
        return self._off_diagonal_scale

    @property
    def dimension(self) -> int:
        """The matrix size ``n``."""
        return self._dimension

    @property
    def component_count(self) -> int:
        """The number of packed components, ``n(n+1)/2``."""
        return len(self._index_pairs)

    def labels(self, axes: Sequence[str]) -> tuple[str, ...]:
        """The packed component labels over basis axis names: ``axes[i] + axes[j]`` per pair.

        Raises:
            ValueError: If the axis count is not the dimension, or two labels coincide.
        """
        names = tuple(axes)
        if len(names) != self._dimension:
            raise ValueError(
                f"the packing is {self._dimension}-dimensional, but the basis has {len(names)} "
                f"axes {names}",
            )
        labels = tuple(names[i] + names[j] for i, j in self._index_pairs)
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"axis names {names} give coinciding packed labels {labels}; rename the axes",
            )
        return labels

    def unpack(self, components: npt.ArrayLike) -> npt.NDArray[np.floating | np.complexfloating]:
        """Expand packed components on the last axis into symmetric matrices on the last two.

        Raises:
            TypeError: If the values are masked, boolean or not numeric.
            ValueError: If the last axis does not have ``n(n+1)/2`` entries.
        """
        values = _value_array(components)
        if values.ndim == 0 or values.shape[-1] != self.component_count:
            raise ValueError(
                f"the last axis must hold the {self.component_count} packed components, got "
                f"shape {values.shape}",
            )
        n = self._dimension
        dtype = np.result_type(values.dtype, np.float64)
        rows, columns, scales = self._entries()
        entries = values.astype(dtype, copy=False) / scales
        matrices = np.empty((*values.shape[:-1], n, n), dtype=dtype)
        matrices[..., rows, columns] = entries
        matrices[..., columns, rows] = entries
        return matrices

    def pack(self, matrices: npt.ArrayLike) -> npt.NDArray[np.floating | np.complexfloating]:
        """Pack symmetric matrices on the last two axes into components on the last axis.

        Mirrored entries must agree: both missing (NaN) or neither, identical when either is
        infinite, and within :data:`SYMMETRY_TOLERANCE` of the largest finite ``|T|`` of their
        matrix otherwise. Packing stores the mean of each mirrored pair.

        Raises:
            TypeError: If the values are masked, boolean or not numeric.
            ValueError: If the last two axes are not ``n x n``, or a matrix is not symmetric.
        """
        raw = self._matrix_array(matrices)
        values: npt.NDArray[np.floating | np.complexfloating] = raw.astype(
            np.result_type(raw.dtype, np.float64), copy=False
        )
        rows, columns, _ = self._entries()
        upper, lower = values[..., rows, columns], values[..., columns, rows]
        with np.errstate(invalid="ignore"):
            missing = np.isnan(upper) != np.isnan(lower)
            infinite = (np.isinf(upper) | np.isinf(lower)) & (upper != lower)
            finite = np.isfinite(upper) & np.isfinite(lower)
            magnitude = np.abs(np.where(np.isfinite(values), values, 0)).max(axis=(-2, -1))
            difference = np.abs(upper - lower)
        asymmetric = finite & (difference > SYMMETRY_TOLERANCE * magnitude[..., None])
        bad = (missing | infinite | asymmetric).any(axis=-1)
        if bad.any():
            where = tuple(int(i) for i in np.argwhere(bad)[0])
            subject = f"matrix {where}" if where else "the matrix"
            raise ValueError(
                f"{subject} is not symmetric: mirrored entries differ in value, in "
                f"missingness or in an infinity beyond tolerance {SYMMETRY_TOLERANCE}; packing "
                "would lose one of them",
            )
        return self._pack_mean(values)

    def pack_symmetric_part(
        self, matrices: npt.ArrayLike
    ) -> npt.NDArray[np.floating | np.complexfloating]:
        """Pack the symmetric part ``(T + T^T) / 2``, without checking symmetry.

        For matrices symmetric by construction, such as a symmetric tensor after a change of
        basis, where only rounding separates mirrored entries.
        """
        return self._pack_mean(self._matrix_array(matrices))

    def _pack_mean(
        self, values: npt.NDArray[np.generic]
    ) -> npt.NDArray[np.floating | np.complexfloating]:
        rows, columns, scales = self._entries()
        dtype = np.result_type(values.dtype, np.float64)
        upper = values[..., rows, columns].astype(dtype, copy=False)
        lower = values[..., columns, rows].astype(dtype, copy=False)
        # Halving before adding cannot overflow; equal entries (the usual case) are kept exactly.
        with np.errstate(invalid="ignore"):
            midpoint = np.where(upper == lower, upper, upper / 2 + lower / 2)
        mean = np.where(rows == columns, upper, midpoint)
        packed: npt.NDArray[np.floating | np.complexfloating] = mean * scales
        return packed

    def _matrix_array(self, matrices: npt.ArrayLike) -> npt.NDArray[np.generic]:
        values = _value_array(matrices)
        n = self._dimension
        if values.ndim < 2 or values.shape[-2:] != (n, n):
            raise ValueError(
                f"the last two axes must hold {n} x {n} matrices, got shape {values.shape}",
            )
        return values

    def _entries(
        self,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.float64]]:
        rows = np.array([i for i, _ in self._index_pairs], dtype=np.intp)
        columns = np.array([j for _, j in self._index_pairs], dtype=np.intp)
        scales = np.where(rows == columns, 1.0, self._off_diagonal_scale)
        return rows, columns, scales

    def _key(self) -> tuple[object, ...]:
        return (self._index_pairs, self._off_diagonal_scale)

    def __eq__(self, other: object) -> bool:
        """Compare the pairs and the scale exactly."""
        if not isinstance(other, SymmetricPacking):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        """Hash consistently with :meth:`__eq__`."""
        return hash((SymmetricPacking, self._key()))

    def __repr__(self) -> str:
        """Return a representation naming the pairs and the scale."""
        return (
            f"SymmetricPacking({self._index_pairs!r}, "
            f"off_diagonal_scale={self._off_diagonal_scale!r})"
        )


def _check_dimension(dimension: object) -> int:
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError(f"dimension must be an integer, got {type(dimension).__name__}")
    if dimension < 1:
        raise ValueError(f"dimension must be at least 1, got {dimension}")
    return dimension


def _check_pairs(value: object) -> tuple[tuple[int, int], ...]:
    if isinstance(value, str) or not isinstance(value, Sequence) or not value:
        raise TypeError("index_pairs must be a non-empty sequence of (i, j) integer pairs")
    pairs: list[tuple[int, int]] = []
    seen: set[frozenset[int]] = set()
    for entry in value:
        if isinstance(entry, str) or not isinstance(entry, Sequence) or len(entry) != 2:
            raise TypeError(f"each index pair must be a pair of integers, got {entry!r}")
        for index in entry:
            if isinstance(index, bool) or not isinstance(index, int | np.integer):
                raise TypeError(f"each index pair must be a pair of integers, got {entry!r}")
            if index < 0:
                raise ValueError(f"matrix indices must be non-negative, got {entry!r}")
        pair = (int(entry[0]), int(entry[1]))
        key = frozenset(pair)
        if key in seen:
            raise ValueError(
                f"index pair {pair} repeats an entry already packed; each unordered pair once",
            )
        seen.add(key)
        pairs.append(pair)
    return tuple(pairs)
