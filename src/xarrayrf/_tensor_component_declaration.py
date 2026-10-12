"""Which array dimensions hold a value's tensor components, and in which basis."""

from __future__ import annotations

from collections.abc import Sequence

from ._array_coordinates import ArrayCoordinates
from ._frame import ReferenceFrame
from ._tensor_type import TensorType
from ._validation import check_names

type ComponentBasis = ReferenceFrame | ArrayCoordinates


class TensorComponentDeclaration:
    """A tensor type with the dimension and basis of each of its component indices.

    The components along ``component_dims[k]`` are the coordinates of slot ``k`` in
    ``bases[k]``: along a :class:`~xarrayrf.ReferenceFrame`'s axes, or per unit of each
    coordinate value of :class:`~xarrayrf.ArrayCoordinates`. The labels of each component
    dimension are the basis axis names, in order.

    The declaration describes how the stored components change with the basis. It is advisory
    with respect to values: an operation that changes the values without consulting it (a
    square, a component-dependent mask) leaves it in place. Equality and hashing are exact.
    """

    __slots__ = ("_bases", "_component_dims", "_tensor_type")

    _tensor_type: TensorType
    _component_dims: tuple[str, ...]
    _bases: tuple[ComponentBasis, ...]

    def __init__(
        self,
        tensor_type: TensorType,
        component_dims: Sequence[str],
        bases: ComponentBasis | Sequence[ComponentBasis],
    ) -> None:
        """Validate and freeze a declaration.

        Args:
            tensor_type: The transformation law; order at least 1. Scalars (densities,
                pseudoscalars) have no component dimension and are declared per call instead.
            component_dims: One distinct dimension name per component index, in index order.
            bases: One basis per component index, or a single basis for all of them.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If the order is 0, the counts disagree, dimensions repeat, or an index
                symmetry is declared for slots in different bases.
        """
        if not isinstance(tensor_type, TensorType):
            raise TypeError(f"tensor_type must be a TensorType, got {type(tensor_type).__name__}")
        if tensor_type.order == 0:
            raise ValueError(
                "a scalar has no component dimension to declare; pass tensor_type= to the "
                "operation instead",
            )
        dims = check_names(component_dims, field="component_dims")
        if len(dims) != tensor_type.order:
            raise ValueError(
                f"component_dims must name one dimension per component index; got {len(dims)} "
                f"for a type of order {tensor_type.order}",
            )
        if isinstance(bases, ReferenceFrame | ArrayCoordinates):
            entries: tuple[object, ...] = (bases,) * tensor_type.order
        elif isinstance(bases, Sequence) and not isinstance(bases, str):
            entries = tuple(bases)
        else:
            raise TypeError(
                "bases must be a ReferenceFrame, ArrayCoordinates or a sequence of them, got "
                f"{type(bases).__name__}",
            )
        if len(entries) != tensor_type.order:
            raise ValueError(
                f"bases must give one basis per component index; got {len(entries)} for a type "
                f"of order {tensor_type.order}",
            )
        for basis in entries:
            if not isinstance(basis, ReferenceFrame | ArrayCoordinates):
                raise TypeError(
                    f"each basis must be a ReferenceFrame or ArrayCoordinates, got "
                    f"{type(basis).__name__}",
                )
        checked: tuple[ComponentBasis, ...] = entries  # type: ignore[assignment]
        if tensor_type.index_symmetry is not None and checked[0] != checked[1]:
            raise ValueError(
                f"an {tensor_type.index_symmetry} declaration needs both indices in one basis; "
                "symmetry is not preserved when the slots change basis separately",
            )
        self._tensor_type = tensor_type
        self._component_dims = dims
        self._bases = checked

    @property
    def tensor_type(self) -> TensorType:
        """The transformation law."""
        return self._tensor_type

    @property
    def component_dims(self) -> tuple[str, ...]:
        """One dimension name per component index, in index order."""
        return self._component_dims

    @property
    def bases(self) -> tuple[ComponentBasis, ...]:
        """One basis per component index, in index order."""
        return self._bases

    def labels(self, slot: int) -> tuple[str, ...]:
        """The component labels of one slot: its basis axis names, in order."""
        return self._bases[slot].axes

    def _key(self) -> tuple[object, ...]:
        return (self._tensor_type, self._component_dims, self._bases)

    def __eq__(self, other: object) -> bool:
        """Compare the law, dimensions and bases exactly."""
        if not isinstance(other, TensorComponentDeclaration):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        """Hash consistently with :meth:`__eq__`."""
        return hash((TensorComponentDeclaration, self._key()))

    def __repr__(self) -> str:
        """Return a representation naming the law, dimensions and bases."""
        return (
            f"TensorComponentDeclaration({self._tensor_type!r}, "
            f"component_dims={self._component_dims!r}, bases={self._bases!r})"
        )
