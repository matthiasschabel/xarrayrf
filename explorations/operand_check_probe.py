"""Measure how stock xarray resolves scalar index and plain-coordinate operands."""

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import xarray as xr


class TagConflict(ValueError):  # noqa: N818 - named by the measurement plan
    """Two indexed tag declarations disagree."""


class TagIndex(xr.Index):
    """A scalar index whose combination methods refuse unequal tags."""

    def __init__(self, name: str, variable: xr.Variable):
        self.name = name
        self.variable = variable

    @classmethod
    def from_variables(
        cls, variables: Mapping[str, xr.Variable], *, options: Mapping
    ) -> "TagIndex":
        name, variable = next(iter(variables.items()))
        return cls(name, variable)

    def create_variables(self, variables: Mapping[str, xr.Variable] | None = None) -> dict:
        return {self.name: self.variable}

    def equals(self, other: xr.Index, **kwargs: object) -> bool:
        return isinstance(other, TagIndex) and self.variable.item() == other.variable.item()

    def isel(self, indexers: Mapping) -> "TagIndex":
        return self

    def _agree(self, other: xr.Index) -> None:
        if not self.equals(other):
            raise TagConflict("conflicting indexed tags")

    def join(self, other: xr.Index, how: str = "inner") -> "TagIndex":
        self._agree(other)
        return self

    def reindex_like(self, other: xr.Index, **kwargs: object) -> dict:
        self._agree(other)
        return {}

    @classmethod
    def concat(cls, indexes: list[xr.Index], dim: str, positions: list | None = None) -> "TagIndex":
        first = indexes[0]
        for other in indexes[1:]:
            first._agree(other)
        return first


def tagged(value: int = 10, *, indexed: bool = True) -> xr.DataArray:
    """Minimal xarray-only indexed/plain tag pair for an upstream reproducer."""
    array = xr.DataArray(np.array([1, 2]), dims="x", coords={"x": [0, 1], "tag": value})
    return array.set_xindex("tag", TagIndex) if indexed else array


# A-F are the plan's operand cases. The second operand is reversed in every operation.
CASES = ("A", "B", "C", "D", "E", "F")
ORDERS = ("forward", "reverse")
PAIR_OPERATIONS = ("binary", "numpy", "apply_ufunc", "concat", "merge")
WHERE_PAIRS = (("cond", "x"), ("cond", "y"), ("x", "y"))
JOINS = ("inner", "exact", "override")


def _operands(case: str):
    first = tagged(10, indexed=case != "F")
    if case == "A":
        second = tagged(10)
    elif case == "B":
        second = tagged(20)
    elif case in ("C", "D"):
        second = tagged(20 if case == "C" else 10, indexed=False)
    elif case == "E":
        second = xr.DataArray([1, 2], dims="x", coords={"x": [0, 1]})
    else:
        second = tagged(20, indexed=False)
    return first, second


def _summary(array: xr.DataArray | xr.Dataset) -> dict[str, object]:
    return {
        "tag": array.coords["tag"].values.tolist() if "tag" in array.coords else None,
        "tag_indexed": isinstance(array.xindexes.get("tag"), TagIndex),
    }


def _outcome(call: Callable[[], object]) -> dict[str, object]:
    try:
        result = call()
    except Exception as exc:
        message = str(exc).splitlines()[0] if str(exc) else ""
        if isinstance(exc, TagConflict):
            outcome = "index-refused"
        elif isinstance(exc, NotImplementedError):
            outcome = "unmeasured"
        elif any(word in message.lower() for word in ("conflict", "different", "align", "index")):
            outcome = "xarray-refused"
        else:
            outcome = "unmeasured"
        return {"outcome": outcome, "error": type(exc).__name__, "message": message}
    if isinstance(result, tuple):
        return {"outcome": "silent", "operands": [_summary(item) for item in result]}
    return {"outcome": "silent", **_summary(result)}


def _where(pair: tuple[str, str], first: xr.DataArray, second: xr.DataArray):
    roles = {
        "cond": xr.DataArray([True, False], dims="x", coords={"x": [0, 1]}),
        "x": xr.DataArray([1, 2], dims="x", coords={"x": [0, 1]}),
        "y": xr.DataArray([3, 4], dims="x", coords={"x": [0, 1]}),
    }
    roles[pair[0]], roles[pair[1]] = first, second
    roles["cond"] = roles["cond"].astype(bool)
    return xr.where(roles["cond"], roles["x"], roles["y"])


def _git_revision() -> str | None:
    source = Path(xr.__file__).resolve()
    checkout = source.parent.parent
    if not (checkout / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def observations() -> dict[str, object]:
    """Return every operation/case/order observation without asserting support."""
    cells: dict[str, dict[str, object]] = {}
    for case in CASES:
        for order in ORDERS:
            first, second = _operands(case)
            if order == "reverse":
                first, second = second, first
            calls: dict[str, Callable[[], object]] = {
                "binary": lambda a=first, b=second: a + b,
                "numpy": lambda a=first, b=second: np.add(a, b),
                "apply_ufunc": lambda a=first, b=second: xr.apply_ufunc(np.add, a, b),
                "concat": lambda a=first, b=second: xr.concat([a, b], "x"),
                "merge": lambda a=first, b=second: xr.merge(
                    [a.to_dataset(name="left"), b.to_dataset(name="right")]
                ),
            }
            for name, call in calls.items():
                cells[f"{name}/{case}/{order}"] = _outcome(call)
            for pair in WHERE_PAIRS:
                a, b = _operands(case)
                if order == "reverse":
                    a, b = b, a
                name = f"where_{pair[0]}_{pair[1]}"
                cells[f"{name}/{case}/{order}"] = _outcome(
                    lambda roles=pair, left=a, right=b: _where(roles, left, right)
                )
            for join in JOINS:
                cells[f"align_{join}/{case}/{order}"] = _outcome(
                    lambda a=first, b=second, how=join: xr.align(a, b, join=how)
                )
    indexed = tagged()
    for name, call in {
        "scalar_right": lambda: indexed + 2,
        "scalar_left": lambda: 2 + indexed,
        "scalar_numpy": lambda: np.add(indexed, 2),
    }.items():
        cells[name] = _outcome(call)
    for name in (
        PAIR_OPERATIONS
        + tuple(f"where_{a}_{b}" for a, b in WHERE_PAIRS)
        + tuple(f"align_{join}" for join in JOINS)
    ):
        for case in CASES:
            forward = cells[f"{name}/{case}/forward"]
            reverse = cells[f"{name}/{case}/reverse"]
            for result, other in ((forward, reverse), (reverse, forward)):
                left = result.get("operands")
                right = other.get("operands")
                if left is not None and right is not None:
                    result["order_dependent"] = left != list(reversed(right))
                else:
                    result["order_dependent"] = {
                        key: value for key, value in result.items() if key != "order_dependent"
                    } != {key: value for key, value in other.items() if key != "order_dependent"}
    return {
        "xarray_source": xr.__file__,
        "xarray_version": xr.__version__,
        "xarray_git_revision": _git_revision(),
        "cells": cells,
    }


if __name__ == "__main__":
    print(json.dumps(observations(), indent=2))
