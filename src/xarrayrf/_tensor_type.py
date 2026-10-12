"""Tensor types and the law that transforms tensor components under a linear map.

Positions move with a transform; values with a direction must be re-expressed. Every value
handled here obeys one law, given the components ``J[a, b] = d target_a / d source_b`` of the
linear map from the source basis to the target basis::

    T' = sign(det J)^[odd reflection parity] * |det J|^w
         * (J on each contravariant slot) (J^-T on each covariant slot) T

A :class:`TensorType` declares the slots, the reflection parity and the determinant exponent
``w``; :func:`transform_tensor_components` applies the law to NumPy arrays. Neither knows about
frames or bases: choosing ``J`` (a Jacobian, a basis change, or its
:func:`polar_orthogonal_factor`) is the caller's decision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, Literal, cast

import numpy as np
import numpy.typing as npt

from ._affine import equilibrated_inverse
from ._validation import real_float_array

type IndexVariance = Literal["contravariant", "covariant"]
type ReflectionParity = Literal["even", "odd"]
type IndexSymmetry = Literal["symmetric", "antisymmetric"]

_VARIANCES: Final = frozenset({"contravariant", "covariant"})
_PARITIES: Final = frozenset({"even", "odd"})
_SYMMETRIES: Final = frozenset({"symmetric", "antisymmetric"})

ORTHOGONALITY_TOLERANCE: Final = 1e-6
"""Largest ``max |J^T J - I|`` for which a map counts as orthogonal.

Matches the default constrained-class tolerance of :func:`~xarrayrf.affine_class`. The kernel
sees numbers, not units, so orthogonality is judged on the matrix as given.
"""

_VALUE_KINDS: Final = frozenset({"i", "u", "f", "c"})


class TensorType:
    """How a value's components respond to a change of basis.

    A tensor of order ``k`` has ``k`` component indices (slots). Each slot is
    ``"contravariant"``, transforming with ``J`` like the components of a displacement or
    velocity, or ``"covariant"``, transforming with ``J^-T`` like the components of a gradient
    or a surface normal. Changing metres to centimetres along ``x`` multiplies a velocity's
    ``x`` component by 100 and a gradient's by 1/100.

    Two further factors cover pseudotensors and densities. Odd reflection parity multiplies
    by ``sign(det J)``, so the value changes sign under an orientation-reversing map. The
    determinant exponent ``w`` multiplies by ``|det J|^w``; a mass density, whose value
    falls as volume grows, has ``w = -1``. The exponent is explicit because the sign
    convention of a tensor density's "weight" varies between authors.

    Equality and hashing are exact over all fields.

    Example:
        A reflection negates a vector's ``x`` component, but a pseudovector's other two:

        >>> import numpy as np
        >>> import xarrayrf as xrf
        >>> mirror = np.diag([-1.0, 1.0, 1.0])
        >>> xrf.transform_tensor_components([1.0, 1.0, 1.0], mirror, xrf.VECTOR)
        array([-1.,  1.,  1.])
        >>> xrf.transform_tensor_components([1.0, 1.0, 1.0], mirror, xrf.ORTHOGONAL_PSEUDOVECTOR)
        array([ 1., -1., -1.])
    """

    __slots__ = (
        "_determinant_exponent",
        "_index_symmetry",
        "_index_variance",
        "_reflection_parity",
        "_requires_orthogonal_map",
    )

    _index_variance: tuple[IndexVariance, ...]
    _reflection_parity: ReflectionParity
    _determinant_exponent: float
    _index_symmetry: IndexSymmetry | None
    _requires_orthogonal_map: bool

    def __init__(
        self,
        index_variance: Sequence[IndexVariance],
        *,
        reflection_parity: ReflectionParity = "even",
        determinant_exponent: float = 0.0,
        index_symmetry: IndexSymmetry | None = None,
        requires_orthogonal_map: bool = False,
    ) -> None:
        """Validate and freeze a tensor type.

        Args:
            index_variance: ``"contravariant"`` or ``"covariant"`` for each component index,
                in order; empty for a scalar. ``"up"`` and ``"down"`` are not accepted.
            reflection_parity: ``"odd"`` for a pseudotensor, which changes sign under an
                orientation-reversing map; ``"even"`` otherwise.
            determinant_exponent: The exponent ``w`` of the ``|det J|^w`` factor; ``0`` for an
                ordinary tensor, ``-1`` for a density per unit volume.
            index_symmetry: ``"symmetric"`` or ``"antisymmetric"`` under exchange of the two
                indices of an order-2 type with equal variance; ``None`` otherwise. Declarative:
                the components are not checked against it.
            requires_orthogonal_map: Refuse maps that are not orthogonal, for a type whose
                law is only defined there (:data:`ORTHOGONAL_PSEUDOVECTOR`).

        Raises:
            TypeError: If an argument has the wrong Python type.
            ValueError: If a variance, parity or symmetry is not one of its spellings, the
                exponent is not finite, or a symmetry is declared on a type that is not order 2
                with equal variance.
        """
        self._index_variance = _check_index_variance(index_variance)
        self._reflection_parity = cast(
            ReflectionParity, _check_choice(reflection_parity, _PARITIES, "reflection_parity")
        )
        self._determinant_exponent = _check_exponent(determinant_exponent)
        if index_symmetry is not None:
            index_symmetry = cast(
                IndexSymmetry, _check_choice(index_symmetry, _SYMMETRIES, "index_symmetry")
            )
            if len(self._index_variance) != 2:
                raise ValueError(
                    f"index_symmetry applies to order-2 types only; this type has order "
                    f"{len(self._index_variance)}",
                )
            if self._index_variance[0] != self._index_variance[1]:
                raise ValueError(
                    "index_symmetry needs two indices of equal variance: a mixed tensor "
                    "J T J^-1 does not stay symmetric (a shear maps diag(2, 1) to "
                    "[[2, -1], [0, 1]])",
                )
        self._index_symmetry = index_symmetry
        if not isinstance(requires_orthogonal_map, bool):
            raise TypeError(
                "requires_orthogonal_map must be a bool, got "
                f"{type(requires_orthogonal_map).__name__}",
            )
        self._requires_orthogonal_map = requires_orthogonal_map

    @property
    def index_variance(self) -> tuple[IndexVariance, ...]:
        """``"contravariant"`` or ``"covariant"`` for each component index, in order."""
        return self._index_variance

    @property
    def reflection_parity(self) -> ReflectionParity:
        """``"odd"`` if the value changes sign under an orientation-reversing map."""
        return self._reflection_parity

    @property
    def determinant_exponent(self) -> float:
        """The exponent ``w`` of the ``|det J|^w`` factor."""
        return self._determinant_exponent

    @property
    def index_symmetry(self) -> IndexSymmetry | None:
        """Declared symmetry under exchange of the two indices, or ``None``."""
        return self._index_symmetry

    @property
    def requires_orthogonal_map(self) -> bool:
        """Whether the type's law is defined only for orthogonal maps."""
        return self._requires_orthogonal_map

    @property
    def order(self) -> int:
        """The number of component indices; 0 for a scalar."""
        return len(self._index_variance)

    def _key(self) -> tuple[object, ...]:
        return (
            self._index_variance,
            self._reflection_parity,
            self._determinant_exponent,
            self._index_symmetry,
            self._requires_orthogonal_map,
        )

    def __eq__(self, other: object) -> bool:
        """Compare every field exactly."""
        if not isinstance(other, TensorType):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        """Hash consistently with :meth:`__eq__`."""
        return hash((TensorType, self._key()))

    def __repr__(self) -> str:
        """Return a representation naming every field that differs from its default."""
        text = f"TensorType({self._index_variance!r}"
        if self._reflection_parity != "even":
            text += f", reflection_parity={self._reflection_parity!r}"
        if self._determinant_exponent != 0.0:
            text += f", determinant_exponent={self._determinant_exponent!r}"
        if self._index_symmetry is not None:
            text += f", index_symmetry={self._index_symmetry!r}"
        if self._requires_orthogonal_map:
            text += ", requires_orthogonal_map=True"
        return text + ")"


def _check_choice(value: object, choices: frozenset[str], field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string, got {type(value).__name__}")
    if value not in choices:
        raise ValueError(f"{field} must be one of {tuple(sorted(choices))}, got {value!r}")
    return value


def _check_index_variance(value: object) -> tuple[IndexVariance, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(
            "index_variance must be a sequence of 'contravariant' or 'covariant', got "
            f"{type(value).__name__}",
        )
    entries = []
    for entry in value:
        if entry in ("up", "down"):
            raise ValueError(
                f"index_variance entries are spelled 'contravariant' or 'covariant', not {entry!r}",
            )
        entries.append(cast(IndexVariance, _check_choice(entry, _VARIANCES, "index_variance")))
    return tuple(entries)


def _check_exponent(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(
            f"determinant_exponent must be a real number, got {type(value).__name__}",
        )
    exponent = float(value)
    if not math.isfinite(exponent):
        raise ValueError(f"determinant_exponent must be finite, got {exponent!r}")
    return exponent


SCALAR: Final = TensorType(())
"""An ordinary scalar: unchanged by any map."""

PSEUDOSCALAR: Final = TensorType((), reflection_parity="odd")
"""A scalar that only changes sign under orientation reversal, such as a handedness.

A signed volume also scales with ``|det J|``: declare it with ``determinant_exponent=1``."""

VECTOR: Final = TensorType(("contravariant",))
"""Displacement, velocity, flux: components transform with ``J``."""

COVECTOR: Final = TensorType(("covariant",))
"""Gradient, surface normal, wave vector: components transform with ``J^-T``."""

SECOND_ORDER_CONTRAVARIANT_TENSOR: Final = TensorType(("contravariant", "contravariant"))
"""Two contravariant indices, transforming as ``J T J^T``; stress and diffusion tensors are
usually declared this way."""

SYMMETRIC_SECOND_ORDER_CONTRAVARIANT_TENSOR: Final = TensorType(
    ("contravariant", "contravariant"), index_symmetry="symmetric"
)
"""A symmetric ``J T J^T`` tensor, such as stress, diffusion or a covariance; packable with a
:class:`~xarrayrf.SymmetricPacking`."""

SYMMETRIC_SECOND_ORDER_COVARIANT_TENSOR: Final = TensorType(
    ("covariant", "covariant"), index_symmetry="symmetric"
)
"""A symmetric ``J^-T T J^-1`` tensor, such as a metric or a small strain read as a bilinear
form; packable with a :class:`~xarrayrf.SymmetricPacking`."""

CROSS_PRODUCT_PSEUDOVECTOR: Final = TensorType(
    ("covariant",), reflection_parity="odd", determinant_exponent=1.0
)
"""The cross product of two vectors (torque, angular momentum): ``det(J) J^-T b``, since
``(J u) x (J v) = det(J) J^-T (u x v)``."""

TWO_FORM_DUAL_PSEUDOVECTOR: Final = TensorType(
    ("contravariant",), reflection_parity="odd", determinant_exponent=-1.0
)
"""The axial dual of a 2-form, such as magnetic flux density as a flux: ``det(J)^-1 J b``."""

ORTHOGONAL_PSEUDOVECTOR: Final = TensorType(
    ("contravariant",), reflection_parity="odd", requires_orthogonal_map=True
)
"""A pseudovector under orthogonal maps only, ``sign(det R) R b``.

The two pseudovector origins agree for rotations and reflections and differ under any other
map, so a map that is not orthogonal is refused; declare
:data:`CROSS_PRODUCT_PSEUDOVECTOR` or :data:`TWO_FORM_DUAL_PSEUDOVECTOR` instead."""


def transform_tensor_components(
    values: npt.ArrayLike,
    linear_map: npt.ArrayLike,
    tensor_type: TensorType,
    *,
    component_axes: Sequence[int] | None = None,
) -> npt.NDArray[np.floating | np.complexfloating]:
    """Apply the tensor transformation law to components under a linear map.

    Args:
        values: Components, real integer, floating or complex. Non-finite entries are allowed
            and propagate.
        linear_map: ``J`` shaped ``(*batch, n, n)``, with ``J[a, b] = d target_a / d source_b``.
            ``batch`` must broadcast to the shape of ``values`` without its component axes,
            without enlarging it: ``(n, n)`` applies one map everywhere, ``(4, n, n)`` against
            values ``(4, n)`` one map per point. Broadcast ``values`` first if expansion is
            wanted.
        tensor_type: The law to apply.
        component_axes: One axis of ``values`` per component index, in index order, each of
            length ``n``; negative axes count from the end. Defaults to the trailing
            ``tensor_type.order`` axes. Empty for a scalar.

    Returns:
        A new array with the shape and axis order of ``values``, of dtype
        ``result_type(values.dtype, float64)``.

    Raises:
        TypeError: If ``values`` is masked or not numeric, ``linear_map`` is not real,
            ``tensor_type`` is not a :class:`TensorType`, or an axis is not an integer.
        ValueError: If ``linear_map`` is not finite, not square or its batch does not fit the
            values; component axes are repeated, out of range, the wrong number or not of
            length ``n``; a map is singular where the law needs ``J^-T`` or ``det J``; the
            ``|det J|^w`` factor over- or underflows; or the type requires an orthogonal map
            and a map is not orthogonal.

    Example:
        A shear moves a vector and a covector in different ways:

        >>> import numpy as np
        >>> import xarrayrf as xrf
        >>> shear = np.array([[1.0, 1.0], [0.0, 1.0]])
        >>> xrf.transform_tensor_components([0.0, 1.0], shear, xrf.VECTOR)
        array([1., 1.])
        >>> xrf.transform_tensor_components([1.0, 0.0], shear, xrf.COVECTOR)
        array([ 1., -1.])
    """
    if not isinstance(tensor_type, TensorType):
        raise TypeError(f"tensor_type must be a TensorType, got {type(tensor_type).__name__}")
    array = _value_array(values)
    matrix = real_float_array(linear_map, field="linear_map")
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"linear_map must be shaped (..., n, n) with square matrices, got shape {matrix.shape}",
        )
    size = matrix.shape[-1]
    if size == 0:
        raise ValueError("linear_map must have at least one row and column")
    axes = _component_axes(component_axes, tensor_type.order, array.ndim)
    for axis in axes:
        if array.shape[axis] != size:
            raise ValueError(
                f"component axis {axis} of values has length {array.shape[axis]}, but "
                f"linear_map is {size} by {size}",
            )
    rest = tuple(length for axis, length in enumerate(array.shape) if axis not in axes)
    batch = matrix.shape[:-2]
    try:
        fits = np.broadcast_shapes(batch, rest) == rest
    except ValueError:
        fits = False
    if not fits:
        raise ValueError(
            f"linear_map batch shape {batch} must broadcast to the values' non-component "
            f"shape {rest} without enlarging it; broadcast values explicitly to expand them",
        )

    covariant = "covariant" in tensor_type.index_variance
    odd = tensor_type.reflection_parity == "odd"
    exponent = tensor_type.determinant_exponent
    inverse = None
    if covariant or odd or exponent != 0.0:
        # Validates invertibility with the unit-independent test even when only det J is used.
        inverse = equilibrated_inverse(matrix)
    if tensor_type.requires_orthogonal_map:
        _check_orthogonal(matrix)

    dtype = np.result_type(array.dtype, np.float64)
    if tensor_type.order == 0 and not odd and exponent == 0.0:
        return array.astype(dtype, copy=True)  # an ordinary scalar is unchanged
    result = np.moveaxis(array.astype(dtype, copy=False), axes, range(-len(axes), 0))
    # Broadcast the map over the non-component axes only; the component axes follow at the end.
    order = len(axes)
    for slot, variance in enumerate(tensor_type.index_variance):
        if variance == "contravariant":
            slot_map = matrix
        else:
            assert inverse is not None  # computed above for every covariant type
            slot_map = np.swapaxes(inverse, -1, -2)
        result = _apply_on_slot(result, slot_map, slot, order)
    if odd or exponent != 0.0:
        result = result * _scalar_factor(matrix, odd=odd, exponent=exponent, order=order)
    transformed: npt.NDArray[np.floating | np.complexfloating] = np.moveaxis(
        result, range(-len(axes), 0), axes
    )
    return transformed


def _value_array(values: npt.ArrayLike) -> npt.NDArray[np.generic]:
    if isinstance(values, np.ma.MaskedArray):
        raise TypeError(
            "values must not be a masked array; converting one discards the mask silently "
            "and reads masked entries as their underlying data. Resolve the mask explicitly "
            "before calling, for example with filled()",
        )
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise TypeError("values must be a rectangular numeric array") from error
    if array.dtype.kind not in _VALUE_KINDS:
        raise TypeError(
            f"values must hold real integer, floating or complex numbers, got dtype "
            f"{array.dtype}; a boolean dtype is not read as 0 and 1",
        )
    return array


def _component_axes(component_axes: Sequence[int] | None, order: int, ndim: int) -> tuple[int, ...]:
    if component_axes is None:
        if order > ndim:
            raise ValueError(
                f"values have {ndim} axes, fewer than the {order} component axes of the type",
            )
        return tuple(range(ndim - order, ndim))
    if isinstance(component_axes, str) or not isinstance(component_axes, Sequence):
        raise TypeError(
            f"component_axes must be a sequence of integers, got {type(component_axes).__name__}",
        )
    axes = []
    for axis in component_axes:
        if isinstance(axis, bool) or not isinstance(axis, int | np.integer):
            raise TypeError(f"component_axes entries must be integers, got {type(axis).__name__}")
        if not -ndim <= axis < ndim:
            raise ValueError(f"component axis {axis} is out of range for values with {ndim} axes")
        axes.append(int(axis) % ndim)
    if len(axes) != order:
        raise ValueError(
            f"component_axes must name one axis per component index; got {len(axes)} for a "
            f"type of order {order}",
        )
    if len(set(axes)) != len(axes):
        raise ValueError(f"component_axes must be distinct, got {tuple(component_axes)}")
    return tuple(axes)


def _apply_on_slot(
    components: npt.NDArray[np.generic],
    slot_map: npt.NDArray[np.float64],
    slot: int,
    order: int,
) -> npt.NDArray[np.generic]:
    """Contract ``slot_map`` with one trailing component axis, keeping the others in place."""
    position = -order + slot
    moved = np.moveaxis(components, position, -1)
    # The map's batch axes align with the non-component axes; the other order - 1 component
    # axes sit between them and the contracted axis, so the map gets singleton axes there.
    expanded = slot_map.reshape(slot_map.shape[:-2] + (1,) * (order - 1) + slot_map.shape[-2:])
    contracted = np.matmul(expanded, moved[..., np.newaxis])[..., 0]
    return np.moveaxis(contracted, -1, position)


def _scalar_factor(
    matrix: npt.NDArray[np.float64], *, odd: bool, exponent: float, order: int
) -> npt.NDArray[np.float64]:
    """``sign(det J)`` for odd parity times ``|det J|^w``, shaped to broadcast over values."""
    sign, logabsdet = np.linalg.slogdet(matrix)
    # exp(w log|det|) stays representable where det itself under- or overflows, as for
    # 1e120 * I_3 with w = 1/3 (det 1e360, factor 1e120).
    log_factor = exponent * logabsdet
    with np.errstate(over="ignore", under="ignore"):  # refused below with a clear message
        factor = np.exp(log_factor)
    bad = ~np.isfinite(factor) | ((factor == 0.0) & (log_factor != 0.0))
    if np.any(bad):
        batch = matrix.shape[:-2]
        first = np.unravel_index(int(np.argmax(bad)), batch) if batch else ()
        position = f" at stack index {tuple(int(i) for i in first)}" if batch else ""
        raise ValueError(
            f"the |det J|^w factor{position} over- or underflows float64 "
            f"(w log|det J| = {float(np.asarray(log_factor)[first]):.6g})",
        )
    if odd:
        factor = factor * sign
    return np.asarray(factor).reshape(np.shape(factor) + (1,) * order)


def _check_orthogonal(matrix: npt.NDArray[np.float64]) -> None:
    size = matrix.shape[-1]
    residual = np.abs(np.swapaxes(matrix, -1, -2) @ matrix - np.eye(size))
    worst = np.max(residual, axis=(-2, -1))
    if worst.size and np.max(worst) > ORTHOGONALITY_TOLERANCE:
        raise ValueError(
            f"this tensor type is defined only for orthogonal maps, and linear_map is not "
            f"orthogonal (max |J^T J - I| = {float(np.max(worst)):.3g} exceeds "
            f"{ORTHOGONALITY_TOLERANCE:g}); a cross product of vectors transforms as "
            "CROSS_PRODUCT_PSEUDOVECTOR and the dual of a 2-form as TWO_FORM_DUAL_PSEUDOVECTOR, "
            "which differ under such a map",
        )


def polar_orthogonal_factor(linear_map: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the orthogonal factor ``U`` of the polar decomposition ``J = U P``.

    ``U`` is the orthogonal matrix closest to ``J``, the finite-strain reorientation of a
    deformation. For an orientation-reversing ``J`` it is a reflection, not a rotation. Batched
    over ``(..., n, n)``.

    Args:
        linear_map: ``J`` shaped ``(..., n, n)``, finite and real.

    Returns:
        ``U`` with the shape of ``linear_map``.

    Raises:
        TypeError: If ``linear_map`` is not real.
        ValueError: If it is not finite, not square, or singular, where ``U`` is not unique.

    Example:
        >>> import numpy as np
        >>> import xarrayrf as xrf
        >>> np.round(xrf.polar_orthogonal_factor([[2.0, 0.0], [0.0, 3.0]]), 12)
        array([[1., 0.],
               [0., 1.]])
    """
    matrix = real_float_array(linear_map, field="linear_map")
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2] or matrix.shape[-1] == 0:
        raise ValueError(
            f"linear_map must be shaped (..., n, n) with non-empty square matrices, got shape "
            f"{matrix.shape}",
        )
    equilibrated_inverse(matrix)  # U is unique only for an invertible J
    left, _, right = np.linalg.svd(matrix)
    factor: npt.NDArray[np.float64] = left @ right
    return factor
