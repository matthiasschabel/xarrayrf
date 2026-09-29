"""The immutable affine transform."""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from ._transform import Endpoint, check_endpoint
from ._validation import frozen_float_array, real_float_array

INVERSE_CONDITION_LIMIT: Final = 1.0 / (10.0 * float(np.finfo(np.float64).eps))
"""Largest condition number, after equilibration, for which an inverse is computed.

About 4.5e14 in the 2-norm (``numpy.linalg.cond``). The matrix is first equilibrated: rows, then columns, scaled to unit norm. Changing the unit of any source or
target axis scales a row or column, so equilibration makes the test independent of the choice of
units: a boost written in seconds and metres, or a transform mixing millimetres and metres, is
judged by its geometry rather than by the size of the numbers. A genuinely singular matrix stays
singular after equilibration and is still refused.
"""


def equilibrated_inverse(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Invert a square matrix through row and column equilibration.

    With ``B = R @ matrix @ C`` for diagonal ``R`` and ``C`` that give ``B`` unit rows and then
    unit columns, ``inv(matrix) = C @ inv(B) @ R``. Computing through ``B`` keeps the conditioning
    test and the arithmetic independent of the axes' units.

    Raises:
        ValueError: If the matrix has a zero row or column, or ``B`` is too ill-conditioned for
            an inverse to be meaningful in float64.
    """
    row_norms = np.linalg.norm(matrix, axis=1)
    if not np.all(row_norms > 0.0):
        raise ValueError("the matrix has a zero row, so it is singular and has no inverse")
    scaled = matrix / row_norms[:, np.newaxis]
    column_norms = np.linalg.norm(scaled, axis=0)
    if not np.all(column_norms > 0.0):
        raise ValueError("the matrix has a zero column, so it is singular and has no inverse")
    balanced = scaled / column_norms
    condition = float(np.linalg.cond(balanced))
    if not np.isfinite(condition) or condition > INVERSE_CONDITION_LIMIT:
        raise ValueError(
            f"the matrix is singular or ill-conditioned (condition number {condition:.3g} after "
            f"equilibration exceeds {INVERSE_CONDITION_LIMIT:.3g}), so its inverse is not "
            "meaningful",
        )
    inverse: npt.NDArray[np.float64] = (
        np.linalg.inv(balanced) / column_norms[:, np.newaxis] / row_norms[np.newaxis, :]
    )
    return inverse


class AffineTransform:
    """An affine transform ``x -> matrix @ x + translation`` between two endpoints.

    Each endpoint is a :class:`~xarrayrf.ReferenceFrame` or :class:`~xarrayrf.ArrayCoordinates`.
    From array coordinates, the transform locates an array's samples: it is evaluated on
    coordinate *values*, not positions, so a crop or stride keeps it valid because xarray keeps
    the original coordinate values. Between frames, it is a registration, a relation such as
    patient to equipment, or a coordinate-system change.

    ``matrix`` has one row per target axis and one column per source axis, with no requirement
    to be square, orthonormal or invertible. The transform carries no array, shape or sample
    ownership. It implements :class:`~xarrayrf.SupportsAffine` and
    :class:`~xarrayrf.SupportsInverse`; :meth:`inverse` raises where no inverse is determined.

    Equality is exact and structural. Two differently expressed affines that happen to agree on
    some overlap are not equal; normalizing them is an explicit adapter step.

    Immutability covers supported public use: the declaration is validated once, no public
    method or attribute mutates it, and the published coefficient arrays are isolated
    snapshots. Reinitializing an existing instance by calling ``__init__`` again, or assigning
    to a private slot, is outside the supported API and is not defended against.
    """

    __slots__ = ("_matrix", "_source", "_target", "_translation")

    _source: Endpoint
    _target: Endpoint
    _matrix: npt.NDArray[np.float64]
    _translation: npt.NDArray[np.float64]

    def __init__(
        self,
        *,
        source: Endpoint,
        target: Endpoint,
        matrix: npt.ArrayLike,
        translation: npt.ArrayLike,
    ) -> None:
        """Validate and freeze an affine transform.

        Affine in the coordinate values does not imply uniformly sampled array positions: the
        source coordinates may be nonuniform offsets, physical distances, one-based labels or a
        scalar retained by a selection. Any unit conversion from source to target units is
        carried by ``matrix``.

        Args:
            source: The endpoint points are mapped from; its axes order the matrix columns.
            target: The endpoint points are mapped to; its axes order the matrix rows.
            matrix: Real coefficients shaped (target axes, source axes). A mixed Python
                sequence follows ordinary NumPy promotion, so its inferred dtype is what must
                be real integer or floating; a masked array is refused rather than unmasked.
            translation: One real value per target axis: where the zero source point lands,
                not where the first sample lands.

        Raises:
            TypeError: If an endpoint is not a :class:`~xarrayrf.ReferenceFrame` or
                :class:`~xarrayrf.ArrayCoordinates`, or the coefficients have a non-real
                inferred dtype or are a masked array.
            ValueError: If the coefficients are not a rectangular numeric array, their shape
                disagrees with the endpoints' axes, or any coefficient is not finite.
        """
        self._source = check_endpoint(source, field="source")
        self._target = check_endpoint(target, field="target")
        expected = (len(target.axes), len(source.axes))
        self._matrix = frozen_float_array(matrix, field="matrix")
        if self._matrix.shape != expected:
            raise ValueError(
                f"matrix must be {expected[0]}-by-{expected[1]} for target axes "
                f"{target.axes} and source axes {source.axes}, got shape {self._matrix.shape}",
            )
        self._translation = frozen_float_array(translation, field="translation")
        if self._translation.shape != (expected[0],):
            raise ValueError(
                f"translation must have one value per target axis {target.axes}, got shape "
                f"{self._translation.shape}",
            )

    @property
    def source(self) -> Endpoint:
        """The endpoint points are mapped from."""
        return self._source

    @property
    def target(self) -> Endpoint:
        """The endpoint points are mapped to."""
        return self._target

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """A read-only (target axes, source axes) snapshot of the linear part.

        Each access returns a fresh array over a fresh buffer, sharing no ndarray with the
        transform's storage. An immutable buffer stops writes, but it does not freeze the array
        *header*: assigning to ``shape`` or ``dtype`` on a shared view would change the
        transform's own coefficients. The snapshot isolates those edits to the caller's copy.
        It is not writeable; copy it to obtain a mutable array.
        """
        return np.frombuffer(self._matrix.tobytes(), dtype=np.float64).reshape(self._matrix.shape)

    @property
    def translation(self) -> npt.NDArray[np.float64]:
        """A read-only snapshot of the offset, one value per target axis.

        A fresh isolated snapshot per access, like :attr:`matrix`.
        """
        return np.frombuffer(self._translation.tobytes(), dtype=np.float64).reshape(
            self._translation.shape
        )

    def _points(self, points: npt.ArrayLike, *, field: str) -> npt.NDArray[np.float64]:
        values = real_float_array(points, field=field)
        axes = self._source.axes
        if values.ndim == 0 or values.shape[-1] != len(axes):
            raise ValueError(
                f"{field} must have a trailing axis of length {len(axes)} ordered as {axes}, "
                f"got shape {values.shape}",
            )
        return values

    def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map points whose trailing axis is ordered as the source axes.

        Use :func:`~xarrayrf.transform_named` to supply coordinates by axis name instead.

        Args:
            points: Finite real values shaped ``(..., K)`` for ``K`` source axes.

        Returns:
            A new array shaped ``(..., M)``, ordered as the target axes.

        Raises:
            TypeError: If the inferred dtype is not real integer or floating, or the input is
                a masked array.
            ValueError: If the input is not a rectangular numeric array, is not finite, has
                the wrong trailing length, or evaluation overflows to a non-finite result.
        """
        values = self._points(points, field="points")
        mapped = values @ self._matrix.T + self._translation
        if not np.isfinite(mapped).all():
            raise ValueError(
                "affine evaluation produced non-finite coordinates; check the input "
                "magnitudes and the declared coefficients",
            )
        return mapped

    def jacobian(self, at: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
        """Return the constant Jacobian, which is :attr:`matrix` everywhere.

        Args:
            at: Optional points shaped ``(..., K)``. When given, the result is broadcast to
                ``(..., M, K)`` so affine and location-dependent transforms share a layout.

        Returns:
            A read-only snapshot shaped ``(M, K)``, or ``(..., M, K)`` when ``at`` is given.

        Raises:
            TypeError: If ``at`` has a non-real dtype or is a masked array.
            ValueError: If ``at`` has the wrong trailing length or is not finite.
        """
        if at is None:
            return self.matrix
        leading = self._points(at, field="at").shape[:-1]
        return np.broadcast_to(self.matrix, (*leading, *self._matrix.shape))

    def inverse(self) -> AffineTransform:
        """Return the inverse affine, with source and target exchanged.

        The inverse is determined only for a square matrix whose condition number, measured after
        equilibration so that the choice of units does not matter, is within
        :data:`INVERSE_CONDITION_LIMIT`. A rectangular affine, such as a plane embedded in a
        volume, has no inverse: a point off the plane has no preimage, and choosing one is a
        separately named operation, never this method.

        The inverse of a transform from array coordinates maps its frame back into those array
        coordinates: the exact part of locating a frame point among an array's samples.

        Returns:
            The inverse transform, ``matrix = M^-1`` and ``translation = -M^-1 t``.

        Raises:
            ValueError: If the matrix is not square, or is singular or too ill-conditioned for
                the inverse to be meaningful in float64.
        """
        rows, columns = self._matrix.shape
        if rows != columns:
            raise ValueError(
                f"a {rows}-by-{columns} affine maps between spaces of different dimension and has "
                "no inverse; projecting a point onto its image is a separate operation",
            )
        inverse = equilibrated_inverse(self._matrix)
        return AffineTransform(
            source=self._target,
            target=self._source,
            matrix=inverse,
            translation=-(inverse @ self._translation),
        )

    def with_endpoints(
        self, *, source: Endpoint | None = None, target: Endpoint | None = None
    ) -> AffineTransform:
        """Return the same coefficients between replaced endpoints.

        Renaming source axes or adopting another frame's identity changes what the endpoints
        say, not the map, so the matrix and translation are reused as they are.

        Args:
            source: Replacement source endpoint, or ``None`` to keep the current one.
            target: Replacement target endpoint, or ``None`` to keep the current one.

        Raises:
            TypeError: If a replacement is not an endpoint.
            ValueError: If a replacement's axis count disagrees with the coefficients.
        """
        return AffineTransform(
            source=self._source if source is None else source,
            target=self._target if target is None else target,
            matrix=self._matrix,
            translation=self._translation,
        )

    def __eq__(self, other: object) -> bool:
        """Compare endpoints and coefficients exactly.

        Exact comparison is deliberate. A tolerance here would make compatibility a
        non-transitive relation and would accept two transforms that merely agree where they
        were sampled.
        """
        if not isinstance(other, AffineTransform):
            return NotImplemented
        if (self._source, self._target) != (other._source, other._target):
            return False
        return bool(
            np.array_equal(self._matrix, other._matrix)
            and np.array_equal(self._translation, other._translation)
        )

    def __hash__(self) -> int:
        """Hash the declaration, consistent with :meth:`__eq__`."""
        return hash(
            (
                AffineTransform,
                self._source,
                self._target,
                self._matrix.tobytes(),
                self._translation.tobytes(),
            )
        )

    def __repr__(self) -> str:
        """Return an unambiguous representation naming the endpoints and coefficients."""
        return (
            f"AffineTransform(source={self._source!r}, target={self._target!r}, "
            f"matrix={self._matrix.tolist()!r}, translation={self._translation.tolist()!r})"
        )
