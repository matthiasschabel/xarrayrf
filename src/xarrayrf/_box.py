"""Sparse tensor operators for declared box support; no array-binding dependencies."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._sampling import INTERVAL_ROUNDOFF_FACTOR, INTERVAL_TOLERANCE

type Rows = list[tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]]


def interval_tolerance(rows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.maximum(
        INTERVAL_TOLERANCE * (rows[..., 1] - rows[..., 0]),
        INTERVAL_ROUNDOFF_FACTOR * np.finfo(float).eps * np.max(np.abs(rows), axis=-1),
    )


def weights(
    source: npt.NDArray[np.float64], target: npt.NDArray[np.float64], *, average: bool
) -> tuple[Rows, npt.NDArray[np.float64]]:
    """Sweep edges in O(edges + emitted weights) after sorting.

    The primitive of 1 / covering-count gives each intersecting slab's contribution;
    the primitive of covered/not-covered gives the target's union coverage.
    """
    targets = target.reshape(-1, 2) if average else target.reshape(-1, 1)
    events = [(float(lo), 0, i) for i, (lo, _) in enumerate(source)]
    events += [(float(hi), 2, i) for i, (_, hi) in enumerate(source)]
    events += [(float(row[0]), 1, j) for j, row in enumerate(targets)]
    if average:
        events += [(float(row[1]), 3, j) for j, row in enumerate(targets)]
        edges = np.unique(np.concatenate((source.ravel(), targets.ravel())))
        delta = np.zeros(edges.size)
        np.add.at(delta, np.searchsorted(edges, source[:, 0]), 1)
        np.add.at(delta, np.searchsorted(edges, source[:, 1]), -1)
        counts = np.cumsum(delta)[:-1]
        lengths = np.diff(edges)
        reciprocal = np.divide(lengths, counts, out=np.zeros_like(lengths), where=counts > 0)
        primitive = dict(zip(edges, np.r_[0, np.cumsum(reciprocal)], strict=True))
        union = np.r_[0, np.cumsum(lengths * (counts > 0))]
        ends = np.searchsorted(edges, targets)
        covered = union[ends[:, 1]] - union[ends[:, 0]]
        coverage = np.clip(covered / np.diff(targets, axis=1)[:, 0], 0, 1)
    else:
        covered = np.ones(len(targets))
        coverage = np.zeros(len(targets))
    entries: list[list[tuple[int, float]]] = [[] for _ in targets]
    active_sources: set[int] = set()
    active_targets: set[int] = set()

    def add(i: int, j: int) -> None:
        if average:
            lo = max(source[i, 0], targets[j, 0])
            hi = min(source[i, 1], targets[j, 1])
            weight = primitive[hi] - primitive[lo]
            if weight > 0:
                entries[j].append((i, weight / covered[j]))
        else:
            entries[j].append((i, 1.0 / len(active_sources)))
            coverage[j] = 1.0

    # Starts precede queries and ends, giving closed intervals at shared point edges.
    for _, kind, index in sorted(events):
        if kind == 0:
            active_sources.add(index)
            for j in active_targets:
                add(index, j)
        elif kind == 1:
            for i in active_sources:
                add(i, index)
            if average:
                active_targets.add(index)
        elif kind == 2:
            active_sources.remove(index)
        else:
            active_targets.remove(index)
    rows = [
        (np.array([i for i, _ in row], dtype=np.intp), np.array([w for _, w in row]))
        for row in entries
    ]
    return rows, coverage


def as_operator(rows: Rows, sources: int) -> Any:
    """CSR weights ``(targets, sources)``; only stored entries multiply, so NaN never meets 0."""
    from scipy import sparse

    lengths = [indices.size for indices, _ in rows]
    indptr = np.r_[0, np.cumsum(lengths, dtype=np.intp)]
    indices = np.concatenate([i for i, _ in rows]) if rows else np.empty(0, dtype=np.intp)
    data = np.concatenate([w for _, w in rows]) if rows else np.empty(0)
    return sparse.csr_array((data, indices, indptr), shape=(len(rows), sources))


@dataclass
class BoxOperator:
    """Per source axis: target dim (None for a retained scalar) and either selected indices
    (pass-through) or CSR weights (slab axis)."""

    axes: list[tuple[int, str | None, Any]]
    coverage: npt.NDArray[np.float64]
    min_coverage: float

    def apply(
        self,
        values: npt.NDArray[Any],
        source_ndim: int,
        target_dims: tuple[str, ...],
        fill_value: float,
    ) -> npt.NDArray[Any]:
        leading = values.ndim - source_ndim
        # Weights are real: complex parts are reduced separately so a NaN in one part cannot
        # reach the other through complex multiplication.
        components = (values.real, values.imag) if np.iscomplexobj(values) else (values,)
        outputs = []
        for component in components:
            result = np.asarray(component, dtype=np.float64)
            for axis, _, operator in self.axes:
                position = leading + axis
                if isinstance(operator, np.ndarray):
                    if not np.array_equal(operator, np.arange(result.shape[position])):
                        result = np.take(result, operator, axis=position)
                    continue
                moved = np.moveaxis(result, position, -1)
                flat = moved.reshape(-1, moved.shape[-1])
                reduced = (operator @ flat.T).T
                result = np.moveaxis(
                    reduced.reshape(*moved.shape[:-1], operator.shape[0]), -1, position
                )
            order = [next(i for i, dim, _ in self.axes if dim == d) for d in target_dims]
            scalars = [i for i, dim, _ in self.axes if dim is None]
            result = result.transpose([*range(leading), *(leading + i for i in order + scalars)])
            outputs.append(result.reshape((*result.shape[: leading + len(order)],)))
        output = outputs[0]
        if len(outputs) == 2:
            output = output.astype(np.complex128)
            output.imag = outputs[1]
        return np.where(self.coverage < self.min_coverage, fill_value, output)
