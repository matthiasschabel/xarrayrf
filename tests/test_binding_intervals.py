"""Interval arithmetic on positional matches supplied by a binding."""

from __future__ import annotations

import numpy as np
import pytest

from xarrayrf._intervals import interval_rows, intervals_equal


@pytest.mark.parametrize(
    ("mine", "theirs", "expected"),
    [
        (None, None, True),
        (None, [[0, 1]], False),
        ([[0, 1]], None, False),
        ([[0, 1]], [[0, 1]], True),
        ([[0, 1]], [[0, 2]], False),
        ([[0, 1]], [[0, 1], [1, 2]], False),
        ([0, 1], [0, 1], True),
        ([0, 1], [0, 2], False),
        ([], [], True),
    ],
)
def test_intervals_equal_entire_declarations(
    mine: list[list[int]] | list[int] | None,
    theirs: list[list[int]] | list[int] | None,
    expected: bool,
) -> None:
    assert (
        intervals_equal(
            None if mine is None else np.asarray(mine, dtype=np.float64),
            None if theirs is None else np.asarray(theirs, dtype=np.float64),
        )
        is expected
    )


@pytest.mark.parametrize(
    ("positions", "expected"),
    [([1, 0], True), ([0, 1], False), ([-1, -1], True), ([1, -1], True), ([], True)],
)
def test_intervals_equal_matched_and_disjoint_rows(positions: list[int], expected: bool) -> None:
    mine = np.array([[0, 1], [2, 3]], dtype=np.float64)[: len(positions)]
    theirs = np.array([[2, 3], [0, 1]], dtype=np.float64)
    assert intervals_equal(mine, theirs, np.asarray(positions, dtype=np.intp)) is expected


def test_intervals_equal_repeated_target_labels() -> None:
    mine = np.array([[2, 3], [2, 3], [0, 1]], dtype=np.float64)
    theirs = np.array([[0, 1], [2, 3]], dtype=np.float64)
    positions = np.array([1, 1, 0], dtype=np.intp)
    assert intervals_equal(mine, theirs, positions)
    mine[1, 1] = 4
    assert not intervals_equal(mine, theirs, positions)


@pytest.mark.parametrize("positions", [[1, 0], [1, 1, 0], []])
def test_interval_rows_reorders_and_repeats(positions: list[int]) -> None:
    mine = np.array([[0, 1], [2, 3]], dtype=np.float64)
    result = interval_rows(mine, np.asarray(positions, dtype=np.intp))
    np.testing.assert_array_equal(result, mine[positions])
    assert result.shape == (len(positions), 2)
    assert result.dtype == np.float64
    assert not np.shares_memory(result, mine)


@pytest.mark.parametrize("positions", [[1, -1, 0, -1], [-1, -1, -1, -1]])
def test_interval_rows_carries_missing_and_disjoint_labels(positions: list[int]) -> None:
    mine = np.array([[0, 1], [2, 3]], dtype=np.float64)
    theirs = np.array([[4, 5], [6, 7]], dtype=np.float64)
    indexer = np.asarray(positions, dtype=np.intp)
    missing = indexer < 0
    other_positions = np.resize(np.array([1, 0], dtype=np.intp), np.count_nonzero(missing))
    result = interval_rows(mine, indexer, theirs=theirs, other_positions=other_positions)
    np.testing.assert_array_equal(result[~missing], mine[indexer[~missing]])
    np.testing.assert_array_equal(result[missing], theirs[other_positions])
