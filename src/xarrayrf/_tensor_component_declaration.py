"""Which array dimensions hold a value's tensor components, and in which basis."""

from __future__ import annotations

from collections.abc import Sequence

from ._array_coordinates import ArrayCoordinates
from ._frame import ReferenceFrame
from ._symmetric_packing import SymmetricPacking
from ._tensor_type import TensorType
from ._validation import check_names

type ComponentBasis = ReferenceFrame | ArrayCoordinates


class TensorComponentDeclaration:
    """A tensor type with the dimension and basis of each of its component indices.

    The components along ``component_dims[k]`` are the coordinates of slot ``k`` in
    ``bases[k]``: along a :class:`~xarrayrf.ReferenceFrame`'s axes, or per unit of each
    coordinate value of :class:`~xarrayrf.ArrayCoordinates`. The labels of each component
    dimension are the basis axis names, in order.

    A symmetric order-2 type may instead be packed: one component dimension holds both indices
    in the order and off-diagonal scale of a :class:`~xarrayrf.SymmetricPacking`, labelled
    ``axes[i] + axes[j]`` per packed entry.

    The declaration describes how the stored components change with the basis. It is advisory
    with respect to values: an operation that changes the values without consulting it (a
    square, a component-dependent mask) leaves it in place. Equality and hashing are exact.
    """

    __slots__ = ("_bases", "_component_dims", "_packing", "_tensor_type")

    _tensor_type: TensorType
    _component_dims: tuple[str, ...]
    _bases: tuple[ComponentBasis, ...]
    _packing: SymmetricPacking | None

    def __init__(
        self,
        tensor_type: TensorType,
        component_dims: Sequence[str],
        bases: ComponentBasis | Sequence[ComponentBasis],
        *,
        packing: SymmetricPacking | None = None,
    ) -> None:
        """Validate and freeze a declaration.

        Args:
            tensor_type: The transformation law; order at least 1. Scalars (densities,
                pseudoscalars) have no component dimension and are declared per call instead.
            component_dims: One distinct dimension name per component index, in index order;
                with ``packing``, the one dimension holding both indices.
            bases: One basis per component index, or a single basis for all of them.
            packing: The packing of a symmetric order-2 type into one dimension, or ``None``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If the order is 0, the counts disagree, dimensions repeat, an index
                symmetry is declared for slots in different bases, or a packing is given for a
                type that is not symmetric of order 2, or does not fit the basis.
        """
        if not isinstance(tensor_type, TensorType):
            raise TypeError(f"tensor_type must be a TensorType, got {type(tensor_type).__name__}")
        if tensor_type.order == 0:
            raise ValueError(
                "a scalar has no component dimension to declare; pass tensor_type= to the "
                "operation instead",
            )
        if packing is not None and not isinstance(packing, SymmetricPacking):
            raise TypeError(f"packing must be a SymmetricPacking, got {type(packing).__name__}")
        if packing is not None and tensor_type.index_symmetry != "symmetric":
            raise ValueError(
                f"only a symmetric order-2 type can be packed; {tensor_type!r} is not declared "
                "index_symmetry='symmetric'",
            )
        dims = check_names(component_dims, field="component_dims")
        if packing is not None and len(dims) != 1:
            raise ValueError(
                f"a packed declaration names one component dimension for both indices; got "
                f"{len(dims)}",
            )
        if packing is None and len(dims) != tensor_type.order:
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
        if packing is not None:
            packing.labels(checked[0].axes)  # refuses a dimension mismatch or coinciding labels
        self._tensor_type = tensor_type
        self._component_dims = dims
        self._bases = checked
        self._packing = packing

    @property
    def tensor_type(self) -> TensorType:
        """The transformation law."""
        return self._tensor_type

    @property
    def component_dims(self) -> tuple[str, ...]:
        """One dimension name per component index, in index order; one when packed."""
        return self._component_dims

    @property
    def bases(self) -> tuple[ComponentBasis, ...]:
        """One basis per component index, in index order."""
        return self._bases

    @property
    def packing(self) -> SymmetricPacking | None:
        """The packing of both indices into one dimension, or ``None``."""
        return self._packing

    def labels(self, position: int) -> tuple[str, ...]:
        """The labels of ``component_dims[position]``: its basis axis names, or packed labels."""
        if self._packing is not None:
            return self._packing.labels(self._bases[0].axes)
        return self._bases[position].axes

    def _key(self) -> tuple[object, ...]:
        return (self._tensor_type, self._component_dims, self._bases, self._packing)

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
            f"component_dims={self._component_dims!r}, bases={self._bases!r}"
            + (f", packing={self._packing!r}" if self._packing is not None else "")
            + ")"
        )
