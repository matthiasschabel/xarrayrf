"""Direction vocabularies: data that names the directions coordinate axes may point."""

from __future__ import annotations

from collections.abc import Iterable

from ._validation import check_str

_SEPARATOR = "-to-"


def _reverse(token: str) -> str:
    start, end = token.split(_SEPARATOR)
    return f"{end}{_SEPARATOR}{start}"


def _check_token(token: object, *, field: str) -> str:
    """Require the explicit ``<from>-to-<to>`` form with two distinct, non-empty ends."""
    text = check_str(token, field=field)
    ends = text.split(_SEPARATOR)
    if (
        text != text.strip()
        or len(ends) != 2
        or not all(end.strip() == end and end for end in ends)
        or ends[0] == ends[1]
    ):
        raise ValueError(
            f"{field} {text!r} is not an explicit direction such as 'left-to-right'; bare "
            "codes such as 'RAI' are refused because libraries disagree about whether a "
            "letter names where an axis points from or where it points to",
        )
    return text


class DirectionVocabulary:
    """A named set of directions, each paired with its opposite.

    A vocabulary is data supplied by a domain adapter, such as anatomical terms following
    OME-NGFF RFC-4 or geographic east/north/up. The core never interprets a token; it only
    uses the pairing to match axes between two coordinate systems and to tell a reversed
    axis from an unrelated one.

    Each direction is written in the explicit form ``<from>-to-<to>``, and its opposite is
    ``<to>-to-<from>``, so the pairing is part of the spelling and cannot be declared
    inconsistently. Human-readable names, including names that depend on body region, belong
    to the adapter that supplies the vocabulary.
    """

    __slots__ = ("_directions", "_identifier")

    _identifier: str
    _directions: frozenset[str]

    def __init__(self, identifier: str, directions: Iterable[str]) -> None:
        """Validate and freeze a vocabulary.

        Args:
            identifier: Name distinguishing this vocabulary from others that might reuse a
                token with a different meaning.
            directions: One token per antipodal pair, in the form ``<from>-to-<to>``. Its
                opposite is implied; listing both directions of a pair is refused, so each
                pair has one canonical spelling in the declaration.

        Raises:
            TypeError: If the identifier or a token is not a string, or ``directions`` is a
                string or not iterable.
            ValueError: If the identifier is empty, no direction is given, a token is not in
                the explicit form, or a pair is listed twice.
        """
        self._identifier = check_str(identifier, field="identifier")
        if isinstance(directions, str) or not isinstance(directions, Iterable):
            raise TypeError(
                f"directions must be an iterable of tokens, got {type(directions).__name__}",
            )
        tokens = [_check_token(token, field="direction") for token in directions]
        if not tokens:
            raise ValueError("a vocabulary must declare at least one direction")
        seen: set[str] = set()
        for token in tokens:
            if token in seen or _reverse(token) in seen:
                raise ValueError(
                    f"direction {token!r} repeats a pair already declared; list each pair once",
                )
            seen.add(token)
        self._directions = frozenset(seen | {_reverse(token) for token in seen})

    @property
    def identifier(self) -> str:
        """Name distinguishing this vocabulary."""
        return self._identifier

    @property
    def directions(self) -> frozenset[str]:
        """Every direction in the vocabulary, including each implied opposite."""
        return self._directions

    def opposite(self, token: str) -> str:
        """Return the direction opposite ``token``.

        Raises:
            ValueError: If ``token`` is not in this vocabulary.
        """
        self.check(token)
        return _reverse(token)

    def pair(self, token: str) -> frozenset[str]:
        """Return the unordered antipodal pair ``token`` belongs to.

        Raises:
            ValueError: If ``token`` is not in this vocabulary.
        """
        return frozenset({token, self.opposite(token)})

    def check(self, token: object) -> str:
        """Return ``token`` if it is a direction of this vocabulary.

        Raises:
            TypeError: If ``token`` is not a string.
            ValueError: If ``token`` is not in the explicit form or not in this vocabulary.
        """
        text = _check_token(token, field="direction")
        if text not in self._directions:
            raise ValueError(
                f"direction {text!r} is not in vocabulary {self._identifier!r}",
            )
        return text

    def __eq__(self, other: object) -> bool:
        """Compare identifier and directions exactly."""
        if not isinstance(other, DirectionVocabulary):
            return NotImplemented
        return (self._identifier, self._directions) == (other._identifier, other._directions)

    def __hash__(self) -> int:
        """Hash consistently with :meth:`__eq__`."""
        return hash((DirectionVocabulary, self._identifier, self._directions))

    def __repr__(self) -> str:
        """Return a representation naming the vocabulary and its directions."""
        return f"DirectionVocabulary({self._identifier!r}, {sorted(self._directions)!r})"
