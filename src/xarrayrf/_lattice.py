"""Regular lattices: array positions mapped affinely into a reference frame."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ._frame import ReferenceFrame
from ._validation import check_names, frozen_float_array, real_float_array


class Lattice:
    """A regular sampling of a reference frame: ``point = origin + matrix @ position``.

    Positions are zero-based integer indices along ``dims``. This is the ITK/VTK image geometry
    (origin, spacing, direction) and the NIfTI/nibabel voxel-to-world affine, derived from an
    array whose coordinates are uniformly spaced; it is available only when they are. Obtain one
    from :meth:`~xarrayrf.Geometry.lattice`.
    """

    __slots__ = ("_dims", "_frame", "_matrix", "_origin")

    _frame: ReferenceFrame
    _dims: tuple[str, ...]
    _origin: npt.NDArray[np.float64]
    _matrix: npt.NDArray[np.float64]

    def __init__(
        self,
        *,
        frame: ReferenceFrame,
        dims: Sequence[str],
        origin: npt.ArrayLike,
        matrix: npt.ArrayLike,
    ) -> None:
        """Validate the declaration: ``matrix`` is frame axes by dims, ``origin`` one per axis.

        Raises:
            TypeError: If ``frame`` is not a reference frame, ``dims`` is not a sequence of
                strings, or ``origin``/``matrix`` are not real arrays.
            ValueError: If ``dims`` repeats a name, or ``origin`` and ``matrix`` do not have
                shapes ``(M,)`` and ``(M, D)`` for ``M`` frame axes and ``D`` dims (``D`` may
                be zero for a fully selected point), or a coefficient is not finite.
        """
        if not isinstance(frame, ReferenceFrame):
            raise TypeError(f"frame must be a ReferenceFrame, got {type(frame).__name__}")
        self._frame = frame
        self._dims = check_names(dims, field="dims", allow_empty=True)
        # Frozen buffers normalize signed zero, so equality and the hash agree (as in _affine).
        self._origin = frozen_float_array(origin, field="origin")
        self._matrix = frozen_float_array(matrix, field="matrix")
        expected = (len(frame.axes), len(self._dims))
        if self._matrix.shape != expected:
            raise ValueError(
                f"matrix must have shape {expected} (frame axes by dims), got {self._matrix.shape}"
            )
        if self._origin.shape != (expected[0],):
            raise ValueError(
                f"origin must have shape {(expected[0],)} (one value per frame axis), got "
                f"{self._origin.shape}"
            )

    @property
    def frame(self) -> ReferenceFrame:
        """The reference frame the lattice samples."""
        return self._frame

    @property
    def dims(self) -> tuple[str, ...]:
        """The array dimensions, in the order of the matrix columns."""
        return self._dims

    @property
    def origin(self) -> npt.NDArray[np.float64]:
        """The point at position zero, one value per frame axis."""
        return self._origin.copy()

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Frame axes by dims: column ``j`` is the step for one position along ``dims[j]``."""
        return self._matrix.copy()

    def _common_unit(self, name: str) -> None:
        units = self._frame.units
        if None in units:
            raise ValueError(
                f"{name} is a length in the frame's common unit, but axes "
                f"{tuple(axis for axis, unit in zip(self._frame.axes, units, strict=True) if unit is None)} "
                "declare no unit. Use matrix, whose columns are the steps axis by axis",
            )
        if len(set(units)) != 1:
            raise ValueError(
                f"{name} is a Euclidean length across the frame's axes, which use different units "
                f"{units}; it is defined only when every axis shares one unit. Use "
                "matrix, whose columns are the steps axis by axis",
            )

    @property
    def spacing(self) -> npt.NDArray[np.float64]:
        """The length of one step along each dim, in the frame's common unit.

        Defined only when every frame axis has the same unit, such as millimetres for a patient
        frame or ``1/mm`` for reciprocal space. A frame mixing seconds and metres, or ``1/mm``
        and ``rad/s``, has no single length.

        Raises:
            ValueError: If the frame's axes use different units, or any declares none.
        """
        self._common_unit("spacing")
        spacing: npt.NDArray[np.float64] = np.linalg.norm(self._matrix, axis=0)
        return spacing

    @property
    def direction(self) -> npt.NDArray[np.float64]:
        """Unit vectors of each dim's step, frame axes by dims.

        For an unsheared lattice the columns are orthonormal and this is ITK's direction matrix
        (the lattice's orientation, with ``matrix == direction @ diag(spacing)``). For a sheared
        lattice the columns are unit length but not orthogonal.

        Raises:
            ValueError: If the frame's axes use different units, as for :attr:`spacing`.
        """
        self._common_unit("direction")
        direction: npt.NDArray[np.float64] = self._matrix / self.spacing
        return direction

    @property
    def affine(self) -> npt.NDArray[np.float64]:
        """The index-to-frame affine matrix in homogeneous form, as nibabel's ``img.affine``.

        ``(M + 1)``-by-``(D + 1)``: ``[[matrix, origin], [0 ... 0, 1]]``, mapping a position
        ``(p, 1)`` to the frame point ``(x, 1)``. A 3-D image in a 3-D frame gives the familiar
        4x4 (NIfTI's sform, napari's ``affine=``); a plane embedded in a volume gives a
        rectangular matrix. It is affine, not rigid: ``matrix`` includes the step lengths (and
        any shear), factoring as ``direction @ diag(spacing)`` for an unsheared lattice.

        Reorder dims with :meth:`~xarrayrf.Geometry.lattice` to choose the column order, for
        example ``("i", "j", "k")`` for ITK and NIfTI rather than array order ``("k", "j", "i")``.
        """
        rows, columns = self._matrix.shape
        result = np.zeros((rows + 1, columns + 1))
        result[:rows, :columns] = self._matrix
        result[:rows, columns] = self._origin
        result[rows, columns] = 1.0
        return result

    def transform_point(self, positions: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map positions shaped ``(..., D)`` in ``dims`` order to frame points ``(..., M)``.

        Fractional positions are allowed; this is an affine map, not a lookup.

        Raises:
            TypeError: If the positions have a non-real dtype or are a masked array.
            ValueError: If the trailing length is not the number of dims, or a value is not
                finite.
        """
        values = real_float_array(positions, field="positions")
        if values.ndim == 0 or values.shape[-1] != len(self._dims):
            raise ValueError(
                f"positions must have a trailing axis of length {len(self._dims)} ordered as "
                f"{self._dims}, got shape {values.shape}",
            )
        mapped: npt.NDArray[np.float64] = values @ self._matrix.T + self._origin
        return mapped

    def __eq__(self, other: object) -> bool:
        """Compare frame, dims, origin and matrix exactly, as the other value objects do."""
        if not isinstance(other, Lattice):
            return NotImplemented
        return (
            self._frame == other._frame
            and self._dims == other._dims
            and bool(np.array_equal(self._origin, other._origin))
            and bool(np.array_equal(self._matrix, other._matrix))
        )

    def __hash__(self) -> int:
        """Hash the declaration, consistent with :meth:`__eq__`."""
        return hash(
            (Lattice, self._frame, self._dims, self._origin.tobytes(), self._matrix.tobytes())
        )

    def __repr__(self) -> str:
        """Return a representation naming the frame, dims, origin and matrix."""
        return (
            f"Lattice(frame={self._frame.identifier!r}, dims={self._dims!r}, "
            f"origin={self._origin.tolist()!r}, matrix={self._matrix.tolist()!r})"
        )
