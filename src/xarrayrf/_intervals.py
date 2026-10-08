"""Interval-row arithmetic after a binding has matched its coordinate labels."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def missing_interval_positions(positions: npt.NDArray[np.intp]) -> npt.NDArray[np.bool_]:
    """Identify missing labels in an already-matched positional indexer."""
    return positions < 0


def intervals_equal(
    mine: npt.NDArray[np.float64] | None,
    theirs: npt.NDArray[np.float64] | None,
    positions: npt.NDArray[np.intp] | None = None,
) -> bool:
    """Compare declarations exactly, or only the rows with matched labels.

    ``positions`` maps each row of ``mine`` into ``theirs``; missing labels do not conflict.
    Without positions, compare entire declarations, including retained scalar rows.
    """
    if mine is None or theirs is None:
        return mine is None and theirs is None
    if positions is None:
        return bool(np.array_equal(mine, theirs))
    matched = ~missing_interval_positions(positions)
    return bool(np.array_equal(mine[matched], theirs[positions[matched]]))


def interval_rows(
    mine: npt.NDArray[np.float64],
    positions: npt.NDArray[np.intp],
    *,
    theirs: npt.NDArray[np.float64] | None = None,
    other_positions: npt.NDArray[np.intp] | None = None,
) -> npt.NDArray[np.float64]:
    """Assemble target rows from matched positions, filling missing rows from another binding.

    The caller must resolve every missing position before assembly. ``other_positions`` maps
    only the missing target rows into ``theirs``.
    """
    missing = missing_interval_positions(positions)
    rows = np.empty((len(positions), 2), dtype=np.float64)
    rows[~missing] = mine[positions[~missing]]
    if missing.any():
        assert theirs is not None and other_positions is not None
        assert not missing_interval_positions(other_positions).any()
        rows[missing] = theirs[other_positions]
    return rows
