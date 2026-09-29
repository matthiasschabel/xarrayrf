"""Compare three carriers of the same declaration across one operation matrix.

The three carriers differ only in what owns the declaration coordinate:

* ``bare`` - no index at all on the marker; stock ``PandasIndex`` on ``y`` and ``x``.
* ``scalar`` - the rejected scalar guard from ``scalar_binding_probe``.
* ``joint`` - the candidate in ``joint_coordinate_index`` that owns ``y``, ``x`` and the marker.

Nothing here changes a global xarray option or patches xarray. Alignment joins are requested
per call through ``xr.align(..., join=...)`` so the process-wide ``arithmetic_join`` stays at
its default. Every case reports the outcome it actually produced, including a failure that
makes the joint candidate unusable; no relabelling or special case hides one.
"""

import json
from collections.abc import Callable

import numpy as np
import xarray as xr
from joint_coordinate_index import jointly_framed
from scalar_binding_probe import MARKER, framed, unguarded

CARRIERS: dict[str, Callable[[int], xr.DataArray]] = {
    "bare": unguarded,
    "scalar": framed,
    "joint": jointly_framed,
}

LABELS = {"y": [0, 2, 4], "x": [0, 3, 6, 9]}


def _bare_partner() -> xr.DataArray:
    """An ordinary labelled array with stock per-axis indexes and no declaration."""
    return xr.DataArray(np.ones((3, 4)), dims=("y", "x"), coords=LABELS)


def _shifted_partner() -> xr.DataArray:
    return xr.DataArray(
        np.ones((3, 4)), dims=("y", "x"), coords={"y": [0, 2, 4], "x": [3, 6, 9, 12]}
    )


def _strip_guard(array: xr.DataArray) -> xr.DataArray:
    """Remove whatever index owns the declaration, leaving the declaration in place."""
    if MARKER not in array.xindexes:
        return array
    owned = [name for name, index in array.xindexes.items() if index is array.xindexes[MARKER]]
    return array.drop_indexes(owned)


def _cases(make: Callable[[int], xr.DataArray]) -> dict[str, Callable[[], object]]:
    array = make(10)
    return {
        "sel exact": lambda: array.sel(x=6),
        "sel nearest": lambda: array.sel(x=5, method="nearest"),
        "isel crop": lambda: array.isel(x=slice(1, 3)),
        "isel stride": lambda: array.isel(x=slice(None, None, 2)),
        "isel gather": lambda: array.isel(x=[3, 0, 1]),
        "isel vectorized": lambda: array.isel(x=xr.Variable("z", [0, 1])),
        "isel scalar": lambda: array.isel(y=1),
        "isel scalar drop": lambda: array.isel(y=1, drop=True),
        "different fixed planes": lambda: array.isel(y=0) + array.isel(y=1),
        "transpose": lambda: array.transpose(),
        "rename": lambda: array.rename(x="column"),
        "binary forward": lambda: array + _bare_partner(),
        "binary reflected": lambda: _bare_partner() + array,
        "inner alignment shifted": lambda: xr.align(array, _shifted_partner(), join="inner")[0],
        "dataset owner": lambda: xr.Dataset(
            {"image": array, "sibling": xr.DataArray(np.zeros((3, 4)), dims=("y", "x"))}
        )["image"],
        "dataset sibling": lambda: xr.Dataset(
            {"image": array, "sibling": xr.DataArray(np.zeros((3, 4)), dims=("y", "x"))}
        )["sibling"],
        "missing guard scalar": lambda: _strip_guard(array) + 1,
    }


def _summarize(result: object) -> dict[str, object]:
    if not isinstance(result, xr.DataArray):
        return {"result_type": type(result).__name__}
    declaration = result.coords[MARKER].item() if MARKER in result.coords else None
    return {
        "sizes": dict(result.sizes),
        "coords": sorted(str(name) for name in result.coords),
        "indexed": sorted(str(name) for name in result.xindexes),
        "index_types": sorted({type(index).__name__ for index in result.xindexes.values()}),
        "declaration": declaration,
        "mapping_inputs_present": (
            all(name in result.coords for name in json.loads(declaration)["inputs"])
            if declaration is not None
            else False
        ),
    }


def observations() -> dict[str, object]:
    """Report every carrier/operation outcome, errors included."""
    report: dict[str, object] = {"xarray_version": xr.__version__, "xarray_source": xr.__file__}
    for carrier, make in CARRIERS.items():
        outcomes: dict[str, object] = {}
        for name, case in _cases(make).items():
            try:
                outcomes[name] = _summarize(case())
            except Exception as exc:
                # A probe reports a failure as an observation instead of aborting the run.
                outcomes[name] = {"error": type(exc).__name__, "message": str(exc)}
        report[carrier] = outcomes
    return report


if __name__ == "__main__":
    print(json.dumps(observations(), indent=2, default=str))
