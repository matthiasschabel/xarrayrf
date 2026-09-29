"""Exercise a deliberately insufficient scalar reference-binding candidate."""

import json
from collections.abc import Callable

import numpy as np
import pandas as pd
import xarray as xr

import xarrayrf as xrf


class FrameIndex(xr.Index):
    def __init__(self, variables):
        self.variables = variables
        self.name, self.variable = next(iter(variables.items()))

    @classmethod
    def from_variables(cls, variables, *, options):
        return cls(dict(variables))

    def create_variables(self, variables=None):
        return self.variables

    def to_pandas_index(self):
        return pd.Index([self.variable.item()], name=self.name)

    def equals(self, other, **kwargs):
        return self.variable.item() == other.variable.item()

    def join(self, other, how="inner"):
        if not self.equals(other):
            raise ValueError("incompatible coordinate mappings")
        return self

    def reindex_like(self, other, **kwargs):
        if not self.equals(other):
            raise ValueError("incompatible coordinate mappings")
        return {}


MARKER = "_frame"


def framed(origin: int = 10) -> xr.DataArray:
    """Make an independently constructed array for the scalar-guard experiment."""
    return xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("y", "x"),
        coords={
            "y": [0, 2, 4],
            "x": [0, 3, 6, 9],
            MARKER: json.dumps({"origin": origin, "inputs": ["y", "x"]}, sort_keys=True),
        },
    ).set_xindex(MARKER, FrameIndex)


def unguarded(origin: int = 20) -> xr.DataArray:
    """Make a declaration-carrying operand with no index.

    No ``Index`` method can be called for this operand, in any operation, because it has no
    index. That is the shape of the counterexample a stronger index cannot reach.
    """
    return framed(origin).drop_indexes(MARKER)


PLANE_SPACE = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("L", "P", "S"), ("mm", "mm", "mm")))
"""A 3-D world for the embedded-plane cases; one identity, minted once per process."""

PLANE_TRANSFORM = xrf.AffineTransform(
    target=PLANE_SPACE,
    source=xrf.ArrayCoordinates(("row", "column"), ("1", "1")),
    matrix=((0.0, 0.5), (0.25, 0.0), (0.0, 0.25)),
    translation=(10.0, 20.0, -5.0),
)
"""A real 2-to-3 mapping: ``L=10+c/2``, ``P=20+r/4``, ``S=-5+c/4``.

The column direction moves in both L and S, so the sampled plane is genuinely oblique in the
3-D space rather than axis-aligned, and a lost fixed row term cannot hide behind a zero.
"""


def embedded_plane() -> xr.DataArray:
    """Build a one-based 2-D sample plane carrying the same scalar guard.

    The labels are one-based deliberately. ``PLANE_TRANSFORM`` maps coordinate *values*, so the
    one-based convention is absorbed by the declaration once and needs no index-origin
    correction anywhere downstream.
    """
    return xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("row", "column"),
        coords={
            "row": [1, 2, 3],
            "column": [1, 2, 3, 4],
            MARKER: json.dumps({"origin": 0, "inputs": ["row", "column"]}, sort_keys=True),
        },
    ).set_xindex(MARKER, FrameIndex)


def plane_world(array: xr.DataArray) -> dict[str, np.ndarray]:
    """Evaluate ``PLANE_TRANSFORM`` on an array's current coordinates as a row-major grid.

    Reads coordinates only, so no pixel buffer is touched. A scalar coordinate retained by a
    selection broadcasts against the varying one, which is how a fixed plane term is checked.
    """
    row = np.asarray(array.coords["row"].values)
    column = np.asarray(array.coords["column"].values)
    shaped = row.reshape(row.shape + (1,) * column.ndim)
    return xrf.transform_named(PLANE_TRANSFORM, {"row": shaped, "column": column})


def observations() -> dict[str, object]:
    """Report native outcomes without treating them as supported behavior."""
    a = framed()
    bare = xr.DataArray(np.ones((3, 4)), dims=("y", "x"))
    shifted = bare.assign_coords(y=[0, 2, 4], x=[3, 6, 9, 12])
    operations: dict[str, Callable[[], xr.DataArray]] = {
        "scalar comparison": lambda: a > 5,
        "numpy forward": lambda: a + np.ones((3, 4)),
        "numpy reverse": lambda: np.ones((3, 4)) + a,
        "label intersection": lambda: a + shifted,
        "conflicting mappings": lambda: a + framed(20),
        "crop stride": lambda: a.isel(x=slice(1, None, 2)),
        "transpose": lambda: a.transpose(),
        "plane": lambda: a.isel(y=1),
        "plane drop": lambda: a.isel(y=1, drop=True),
        "mean": lambda: a.mean("y"),
        "different planes": lambda: a.isel(y=0) + a.isel(y=1),
        "plane volume": lambda: a.isel(y=0) + a,
        "incidental scalar": lambda: (
            a.isel(y=0) + xr.DataArray(np.ones(4), dims="x", coords={"y": 2})
        ),
        "lost guard": lambda: a + framed(20).drop_indexes(MARKER),
        "coarsen": lambda: a.coarsen(x=2).mean(),
        "where": lambda: xr.where(a > 5, a, 0),
        "where conflict": lambda: xr.where(a > 5, a, framed(20)),
        "rename": lambda: a.rename(x="column"),
        "override": lambda: xr.align(a, framed(20), join="override")[1],
        "coordinate extraction": lambda: a.x,
        "dataset extraction": lambda: xr.Dataset(
            {"image": a, "other": xr.DataArray([1, 2], dims="echo")}
        )["other"],
    }
    report: dict[str, object] = {"xarray_version": xr.__version__}
    for name, operation in operations.items():
        try:
            result = operation()
        except (ValueError, TypeError, NotImplementedError) as exc:
            report[name] = {"error": type(exc).__name__, "message": str(exc)}
        else:
            report[name] = {
                "sizes": dict(result.sizes),
                "declaration": MARKER in result.coords,
                "guard": MARKER in result.xindexes,
                "required_coordinates": all(c in result.coords for c in ("y", "x")),
            }
    return report


if __name__ == "__main__":
    print(json.dumps(observations(), indent=2))
