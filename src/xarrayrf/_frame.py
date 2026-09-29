"""Immutable reference-frame declarations."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Final, Literal, get_args

from ._coordinate_system import CoordinateSystem
from ._validation import (
    MetadataItems,
    MetadataValue,
    canonical_metadata,
    check_identifier,
    metadata_view,
)

LOCAL_NAMESPACE: Final = "xarrayrf.local"
"""Identifier namespace stamped on minted reference-frame identities.

Only :meth:`ReferenceFrame.local` mints a value in it, and every call mints a new one.
The namespace is not fenced off from :meth:`ReferenceFrame.declared`: re-adopting a
persisted local identity is how a stored frame comes back as the same frame.
"""

type Role = Literal["world", "object"]

_ROLES: Final = frozenset(get_args(Role.__value__))  # a PEP 695 alias wraps the Literal


class ReferenceFrame:
    """A particular reference frame, expressed in one coordinate system.

    The identifier names the frame: what coordinates are relative to, such as one patient's
    DICOM Frame of Reference or one microscope stage. ``definition`` and ``context`` say what
    is needed to interpret it. The :class:`~xarrayrf.CoordinateSystem` says which axes the
    coordinates use. One frame can be expressed in several coordinate systems, so the
    comparisons are separate:

    - :meth:`is_equivalent_frame`: the same frame, whatever the coordinate system.
    - :meth:`conflicts_with`: the same identifier, but a contradictory definition or context.
    - ``a.coordinate_system == b.coordinate_system``: the same axes.
    - ``a == b``: all of these, as transform endpoints and hashing require.

    ``role`` (``"world"`` or ``"object"``) and ``display`` are descriptive. They are excluded
    from every comparison and from hashing, because a role is relative to an analysis: a
    donor body is an object in a scanner and a world for a biopsy taken from it.

    A reference frame owns no array, no sample domain and no shape. Sharing one is the
    supported way for several arrays to claim the same frame.

    Immutability covers supported public use: the declaration is validated once and no
    public method or attribute mutates it. Reinitializing an existing instance by calling
    ``__init__`` again, or assigning to a private slot, is outside the supported API and is
    not defended against.
    """

    __slots__ = (
        "_context",
        "_coordinate_system",
        "_definition",
        "_display",
        "_identifier",
        "_role",
    )

    _identifier: tuple[str, str]
    _coordinate_system: CoordinateSystem
    _role: Role | None
    _definition: MetadataItems
    _context: MetadataItems
    _display: MetadataItems

    def __init__(
        self,
        *,
        identifier: tuple[str, str],
        coordinate_system: CoordinateSystem,
        role: Role | None = None,
        definition: Mapping[str, MetadataValue] | None = None,
        context: Mapping[str, MetadataValue] | None = None,
        display: Mapping[str, MetadataValue] | None = None,
    ) -> None:
        """Validate and freeze a reference-frame declaration.

        Prefer :meth:`local` or :meth:`declared`, which make the identity decision
        explicit. This constructor applies the same validation.

        Args:
            identifier: ``(namespace, value)`` naming a particular frame.
            coordinate_system: The axes this frame's coordinates are expressed in.
            role: ``"world"``, ``"object"`` or ``None``. Descriptive; excluded from
                comparison and hashing.
            definition: Optional flat, data-only declaration an adapter resolved and
                validated for an externally defined system.
            context: Optional defining context, such as an epoch or realization, needed
                to interpret the frame. Acquisition timestamps are not defining context.
            display: Optional human-readable labels and provenance. Excluded from
                comparison and hashing.

        Raises:
            TypeError: If any argument has the wrong Python type.
            ValueError: If the identifier is malformed, the role is unknown, or metadata
                holds an empty key or a non-finite float.
        """
        self._identifier = check_identifier(identifier)
        if not isinstance(coordinate_system, CoordinateSystem):
            raise TypeError(
                "coordinate_system must be a CoordinateSystem, got "
                f"{type(coordinate_system).__name__}",
            )
        self._coordinate_system = coordinate_system
        if role is not None and not isinstance(role, str):
            raise TypeError(f"role must be a string or None, got {type(role).__name__}")
        if role is not None and role not in _ROLES:
            raise ValueError(f"role must be 'world', 'object' or None, got {role!r}")
        self._role = role
        self._definition = canonical_metadata(definition, field="definition")
        self._context = canonical_metadata(context, field="context")
        self._display = canonical_metadata(display, field="display")

    @classmethod
    def local(
        cls,
        coordinate_system: CoordinateSystem,
        *,
        role: Role | None = None,
        definition: Mapping[str, MetadataValue] | None = None,
        context: Mapping[str, MetadataValue] | None = None,
        display: Mapping[str, MetadataValue] | None = None,
    ) -> ReferenceFrame:
        """Mint a private identity for a frame that has no external name.

        Every call mints a distinct identity, without exception, so two independently
        constructed local frames are unrelated even when their declarations are identical.
        Reuse the returned object to declare that several arrays share one frame, or
        re-adopt its identifier through :meth:`declared` to restore a persisted one.

        Args:
            coordinate_system: The axes this frame's coordinates are expressed in.
            role: Optional descriptive role.
            definition: Optional flat, data-only system declaration.
            context: Optional defining context.
            display: Optional display metadata, excluded from comparison.

        Returns:
            A reference frame with a freshly minted identifier.

        Raises:
            TypeError: If any argument has the wrong Python type.
            ValueError: If the declaration is invalid.
        """
        return cls(
            identifier=(LOCAL_NAMESPACE, uuid.uuid4().hex),
            coordinate_system=coordinate_system,
            role=role,
            definition=definition,
            context=context,
            display=display,
        )

    @classmethod
    def declared(
        cls,
        identifier: tuple[str, str],
        coordinate_system: CoordinateSystem,
        *,
        role: Role | None = None,
        definition: Mapping[str, MetadataValue] | None = None,
        context: Mapping[str, MetadataValue] | None = None,
        display: Mapping[str, MetadataValue] | None = None,
    ) -> ReferenceFrame:
        """Adopt an already existing identity, such as a DICOM Frame of Reference UID.

        The identifier is opaque: nothing here parses, resolves or looks up a name, and
        no domain package is imported. A shared name is not evidence that two declarations
        agree; :meth:`conflicts_with` reports when they do not.

        Args:
            identifier: ``(namespace, value)`` naming a particular frame. Any namespace may
                be adopted, including :data:`LOCAL_NAMESPACE`: re-adopting an identity that
                :meth:`local` minted earlier states that a persisted frame is this same
                frame. Only :meth:`local` mints new values, so adoption cannot collide with
                a mint.
            coordinate_system: The axes this frame's coordinates are expressed in.
            role: Optional descriptive role.
            definition: Optional flat, data-only system declaration an adapter validated.
            context: Optional defining context.
            display: Optional display metadata, excluded from comparison.

        Returns:
            A reference frame carrying the declared identity.

        Raises:
            TypeError: If any argument has the wrong Python type.
            ValueError: If the declaration is invalid.
        """
        return cls(
            identifier=identifier,
            coordinate_system=coordinate_system,
            role=role,
            definition=definition,
            context=context,
            display=display,
        )

    def with_coordinate_system(self, coordinate_system: CoordinateSystem) -> ReferenceFrame:
        """Return this frame expressed in another coordinate system.

        Identity, role, definition, context and display are kept. No coordinates are
        converted: use :func:`~xarrayrf.coordinate_system_change` for the transform between
        the two.

        Raises:
            TypeError: If ``coordinate_system`` is not a :class:`~xarrayrf.CoordinateSystem`.
        """
        return type(self)(
            identifier=self._identifier,
            coordinate_system=coordinate_system,
            role=self._role,
            definition=metadata_view(self._definition),
            context=metadata_view(self._context),
            display=metadata_view(self._display),
        )

    @property
    def identifier(self) -> tuple[str, str]:
        """``(namespace, value)`` naming this particular frame."""
        return self._identifier

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """The axes this frame's coordinates are expressed in."""
        return self._coordinate_system

    @property
    def axes(self) -> tuple[str, ...]:
        """The coordinate system's axes, so a frame reads like any transform endpoint."""
        return self._coordinate_system.axes

    @property
    def units(self) -> tuple[str | None, ...]:
        """The coordinate system's units, so a frame reads like any transform endpoint."""
        return self._coordinate_system.units

    @property
    def role(self) -> Role | None:
        """Descriptive role: ``"world"``, ``"object"`` or ``None``."""
        return self._role

    @property
    def definition(self) -> Mapping[str, MetadataValue]:
        """Read-only external system declaration; empty when none was supplied."""
        return metadata_view(self._definition)

    @property
    def context(self) -> Mapping[str, MetadataValue]:
        """Read-only defining context; empty when none was supplied."""
        return metadata_view(self._context)

    @property
    def display(self) -> Mapping[str, MetadataValue]:
        """Read-only display metadata, excluded from comparison."""
        return metadata_view(self._display)

    def _frame_key(self) -> tuple[object, ...]:
        return (self._identifier, self._definition, self._context)

    def is_equivalent_frame(self, other: ReferenceFrame) -> bool:
        """Report whether two declarations describe the same frame.

        Compares identifier, definition and context, ignoring the coordinate system, role
        and display, as Astropy's ``is_equivalent_frame`` ignores representation.

        Raises:
            TypeError: If ``other`` is not a reference frame.
        """
        if not isinstance(other, ReferenceFrame):
            raise TypeError(f"other must be a ReferenceFrame, got {type(other).__name__}")
        return self._frame_key() == other._frame_key()

    def conflicts_with(self, other: ReferenceFrame) -> bool:
        """Report whether two declarations name the same frame but contradict each other.

        A conflict is not inequality. Different identifiers are merely unrelated frames,
        and one frame in two coordinate systems is not a contradiction. The same identifier
        with a different definition or defining context is, and a caller must resolve it.

        Raises:
            TypeError: If ``other`` is not a reference frame.
        """
        if not isinstance(other, ReferenceFrame):
            raise TypeError(f"other must be a ReferenceFrame, got {type(other).__name__}")
        if self._identifier != other._identifier:
            return False
        return (self._definition, self._context) != (other._definition, other._context)

    def __eq__(self, other: object) -> bool:
        """Compare frame and coordinate system exactly; role and display are excluded.

        Comparison is exact and structural. Approximate geometric comparison is a separate
        explicit query, never an equality.
        """
        if not isinstance(other, ReferenceFrame):
            return NotImplemented
        return (self._frame_key(), self._coordinate_system) == (
            other._frame_key(),
            other._coordinate_system,
        )

    def __hash__(self) -> int:
        """Hash consistently with :meth:`__eq__`."""
        return hash((ReferenceFrame, self._frame_key(), self._coordinate_system))

    def __repr__(self) -> str:
        """Return an unambiguous representation naming identity and coordinate system."""
        role = f", role={self._role!r}" if self._role is not None else ""
        return (
            f"ReferenceFrame(identifier={self._identifier!r}, "
            f"coordinate_system={self._coordinate_system!r}{role})"
        )
