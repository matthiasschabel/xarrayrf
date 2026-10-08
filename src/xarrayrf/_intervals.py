"""Interval-row arithmetic after a binding has matched its coordinate labels.

Positions follow ``pandas.Index.get_indexer``: ``-1`` marks a label with no match.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


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
    matched = positions >= 0
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
    missing = positions < 0
    rows = np.empty((len(positions), 2), dtype=np.float64)
    rows[~missing] = mine[positions[~missing]]
    if missing.any():
        assert theirs is not None and other_positions is not None
        assert (other_positions >= 0).all()
        rows[missing] = theirs[other_positions]
    return rows
