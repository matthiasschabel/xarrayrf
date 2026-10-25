# The compared libraries are optional (the "priorart" dependency group) and mostly unannotated.
# mypy: disable-error-code="import-untyped,import-not-found,no-any-return,misc,no-untyped-call"
"""Measure which xarray extension failures the series-5 ``Index`` hooks address.

Each case runs one native operation on an extension's own index and records what happens.
Where a hook applies, the same operation also runs on a subclass of that index implementing
only that hook, on the same xarray tree, and the case records whether the hook was reached.
Credit goes to a hook only when the original index fails and the subclass, with the hook
reached, refuses or gives the correct result. Run it in the ``priorart`` environment on each
lane (``PYTHONPATH`` selects a source tree); it prints JSON::

    UV_PROJECT_ENVIRONMENT=.venv-priorart uv sync --extra dev --group priorart
    .venv-priorart/bin/python tools/extension_hook_survey.py
    PYTHONPATH=$HOME/GitHub/xarray-hooks-5 .venv-priorart/bin/python tools/extension_hook_survey.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

# Each code path names the function on upstream main b2acc77a that drops or replaces the index.
_STACK = "xarray/core/dataset.py:Dataset._stack_once"
_PAD = "xarray/core/dataset.py:Dataset.pad"
_COARSEN = "xarray/computation/rolling.py:DataArrayCoarsen/DatasetCoarsen reductions"
_INTERP = "xarray/core/dataset.py:Dataset.interp"
_OVERRIDE = "xarray/structure/alignment.py:Aligner.override_indexes"
_UNINDEXED = "xarray/structure/alignment.py:Aligner._get_indexes_and_vars"
_OVERLAP = "xarray/structure/alignment.py:Aligner.align_indexes"
_BROADCAST = "xarray/structure/alignment.py:_broadcast_helper"


@dataclass
class Run:
    """One operation on one index class."""

    index: str
    outcome: str
    failed: bool
    hook_calls: list[str] = field(default_factory=list)


@dataclass
class Case:
    """One extension failure and, when a hook applies, its same-tree control."""

    case: str
    library: str
    operation: str
    expected: str
    hook: str
    code_path: str
    original: Run
    hooked: Run | None = None
    verdict: str = ""


class HookRefusalError(ValueError):
    """Raised by the survey's hook subclasses so a refusal is unambiguous."""


def _hooked(base: type[xr.Index], *, hooks: Sequence[str]) -> tuple[type[xr.Index], list[str]]:
    """Subclass ``base`` with only the named hooks, recording each call."""
    calls: list[str] = []
    namespace: dict[str, Any] = {}

    if "check_drop" in hooks:

        def check_drop(self: xr.Index, operation: str, dims: Sequence[Hashable]) -> None:
            calls.append(f"check_drop({operation!r})")
            raise HookRefusalError(f"{base.__name__} refuses {operation} of its dimensions")

        namespace["check_drop"] = check_drop

    if {"override", "validate", "join"} & set(hooks):

        def join_overlapping(
            self: xr.Index,
            other_indexes: Mapping[Hashable, xr.Index],
            *,
            other_variables: Mapping[Hashable, xr.Variable],
            how: str | None,
        ) -> Any:
            calls.append(f"join_overlapping(how={how!r})")
            if how == "override" and "override" in hooks:
                raise HookRefusalError(f"{base.__name__} refuses join='override'")
            if how is None and "validate" in hooks:
                own = self.create_variables()
                for name, variable in other_variables.items():
                    mine = np.asarray(getattr(own[name].data, "magnitude", own[name].data))
                    theirs = np.asarray(getattr(variable.data, "magnitude", variable.data))
                    if not np.array_equal(mine, theirs):
                        raise HookRefusalError(f"conflicting unindexed coordinate {name!r}")
                return self, {}
            if how not in (None, "override") and "join" in hooks:
                # Equal labels only: enough to show the joint index can own the result.
                own = self.create_variables()
                if all(
                    np.array_equal(np.asarray(own[name]), np.asarray(other_variables[name]))
                    for name in other_indexes
                ):
                    return self, dict(other_indexes)
            return None

        namespace["join_overlapping"] = join_overlapping

    return type(f"Hooked{base.__name__}", (base,), namespace), calls


def _run(label: str, operation: Callable[[], str | None], calls: list[str] | None = None) -> Run:
    """Run an operation; ``operation`` returns a failure description or ``None`` when correct."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = operation()
    except HookRefusalError as error:
        return Run(label, f"refused: {error}", False, list(calls or []))
    except Exception as error:  # every outcome is recorded, none is hidden
        return Run(
            label,
            f"{type(error).__name__}: {str(error).splitlines()[0][:120]}",
            True,
            list(calls or []),
        )
    if problem is None:
        return Run(label, "correct", False, list(calls or []))
    return Run(label, problem, True, list(calls or []))


def _verdict(case: Case) -> str:
    if case.hooked is None:
        return "fixed" if not case.original.failed else "fails; no hook applies"
    if not case.original.failed:
        return "original index already correct"
    if case.hooked.hook_calls and not case.hooked.failed:
        return "hook resolves"
    if not case.hooked.hook_calls:
        return "hook not reached"
    return "hook reached but not sufficient"


# --- pint-xarray ---------------------------------------------------------------------------


def _pint(index_class: Any = None, units: str = "m") -> xr.DataArray:
    import pint_xarray  # noqa: F401 - registers the pint accessor

    array = xr.DataArray(np.arange(4.0), dims="x", coords={"x": [0, 1, 2, 3]}).pint.quantify(
        {"x": units}
    )
    if index_class is None:
        return array
    original = array.xindexes["x"]
    hooked = index_class(index=original.index, units=original.units)
    return array.assign_coords(xr.Coordinates.from_xindex(hooked))


def _units_survive(result: xr.DataArray, name: str = "x") -> str | None:
    index = type(result.xindexes[name]).__name__ if name in result.xindexes else "no index"
    if getattr(result[name].data, "units", None) is None:
        return f"units of the indexed coordinate dropped (index now {index})"
    return None


_PINT_DROPS: dict[str, tuple[Callable[[xr.DataArray], xr.DataArray], str]] = {
    "stack": (lambda a: a.expand_dims(y=2).stack(z=("y", "x")).unstack("z"), _STACK),
    "pad": (lambda a: a.pad(x=1), _PAD),
    "coarsen": (lambda a: a.coarsen(x=2).reduce(np.mean), _COARSEN),
    "interp": (lambda a: a.interp(x=[0.5]), _INTERP),
}


def _pint_cases() -> list[Case]:
    from pint_xarray.index import PintIndex

    cases = []
    for name, (operation, path) in _PINT_DROPS.items():
        hooked, calls = _hooked(PintIndex, hooks=["check_drop"])
        cases.append(
            Case(
                f"pint_{name}",
                "pint-xarray",
                name,
                "keep the units, or refuse",
                "A (check_drop)",
                path,
                _run("PintIndex", lambda op=operation: _units_survive(op(_pint()))),
                _run(
                    hooked.__name__,
                    lambda op=operation, h=hooked: _units_survive(op(_pint(h))),
                    calls,
                ),
            )
        )

    def override(index_class: type[xr.Index] | None) -> str | None:
        meters, kilometers = _pint(index_class, "m"), _pint(None, "km")
        aligned = xr.align(kilometers, meters, join="override")[1]
        if str(aligned.x.data.units) != "meter":
            return f"metre labels relabelled as {aligned.x.data.units}"
        return None

    hooked, calls = _hooked(PintIndex, hooks=["override"])
    cases.append(
        Case(
            "pint_override_units",
            "pint-xarray",
            "align(km, m, join='override')",
            "refuse, or keep each operand's units",
            "B (join_overlapping, how='override')",
            _OVERRIDE,
            _run("PintIndex", lambda: override(None)),
            _run(hooked.__name__, lambda: override(hooked), calls),
        )
    )

    def unindexed(index_class: type[xr.Index] | None) -> str | None:
        other = xr.DataArray(np.ones(4), dims="x", coords={"x": ("x", [9, 9, 9, 9])}).drop_indexes(
            "x"
        )
        result = _pint(index_class) + other
        return (
            "unindexed x with different values silently discarded"
            if "x" in result.xindexes
            else None
        )

    hooked, calls = _hooked(PintIndex, hooks=["validate"])
    cases.append(
        Case(
            "pint_unindexed_conflict",
            "pint-xarray",
            "quantified + plain array whose unindexed x differs",
            "refuse the conflicting coordinate",
            "B (join_overlapping, how=None)",
            _UNINDEXED,
            _run("PintIndex", lambda: unindexed(None)),
            _run(hooked.__name__, lambda: unindexed(hooked), calls),
        )
    )

    def broadcast() -> str | None:
        result = xr.broadcast(_pint(), xr.DataArray([1, 2], dims="c"))[0]
        return _units_survive(result)

    cases.append(
        Case(
            "pint_broadcast",
            "pint-xarray",
            "broadcast against a coordinate-free array",
            "keep the PintIndex and units",
            "broadcast fix (no hook)",
            _BROADCAST,
            _run("PintIndex", broadcast),
        )
    )
    return cases


# --- rasterix ------------------------------------------------------------------------------


def _raster(index_class: Any = None) -> xr.DataArray:
    from rasterio.transform import Affine
    from rasterix import RasterIndex

    cls = index_class or RasterIndex
    index = cls.from_transform(Affine(2, 0, 100, 0, -3, 200), width=4, height=3, crs="EPSG:32633")
    data = xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("y", "x"))
    return data.assign_coords(xr.Coordinates.from_xindex(index))


def _raster_survives(result: xr.DataArray) -> str | None:
    from rasterix import RasterIndex

    kinds = sorted({type(index).__name__ for index in result.xindexes.values()})
    if not any(isinstance(index, RasterIndex) for index in result.xindexes.values()):
        return f"RasterIndex (affine and CRS) replaced by {kinds}"
    return None


_RASTER_DROPS: dict[str, tuple[Callable[[xr.DataArray], xr.DataArray], str]] = {
    "stack": (lambda r: r.stack(p=("y", "x")), _STACK),
    "pad": (lambda r: r.pad(x=1), _PAD),
    "coarsen": (lambda r: r.coarsen(x=2).reduce(np.mean), _COARSEN),
    "interp": (lambda r: r.interp(x=[101.0]), _INTERP),
}


def _rasterix_cases() -> list[Case]:
    from rasterix import RasterIndex

    cases = []
    for name, (operation, path) in _RASTER_DROPS.items():
        hooked, calls = _hooked(RasterIndex, hooks=["check_drop"])
        cases.append(
            Case(
                f"rasterix_{name}",
                "rasterix",
                name,
                "keep a valid RasterIndex, or refuse",
                "A (check_drop)",
                path,
                _run("RasterIndex", lambda op=operation: _raster_survives(op(_raster()))),
                _run(
                    hooked.__name__,
                    lambda op=operation, h=hooked: _raster_survives(op(_raster(h))),
                    calls,
                ),
            )
        )
    cases.append(
        Case(
            "rasterix_broadcast",
            "rasterix",
            "broadcast against a coordinate-free array",
            "keep the RasterIndex",
            "broadcast fix (no hook)",
            _BROADCAST,
            _run(
                "RasterIndex",
                lambda: _raster_survives(
                    xr.broadcast(_raster(), xr.DataArray([1, 2], dims="c"))[0]
                ),
            ),
        )
    )

    def profile_first(index_class: type[xr.Index] | None) -> str | None:
        raster = _raster(index_class)
        profile = xr.DataArray(np.ones(4), dims="x", coords={"x": raster.x.values})
        result = profile + raster
        if result.xindexes["x"] is not result.xindexes["y"]:
            kinds = sorted({type(index).__name__ for index in result.xindexes.values()})
            return f"RasterIndex split into {kinds}"
        return None

    hooked, calls = _hooked(RasterIndex, hooks=["join"])
    cases.append(
        Case(
            "rasterix_profile_first",
            "rasterix",
            "1-D array on x with equal labels + raster",
            "keep the RasterIndex over both coordinates",
            "B (join_overlapping, join)",
            _OVERLAP,
            _run("RasterIndex", lambda: profile_first(None)),
            _run(hooked.__name__, lambda: profile_first(hooked), calls),
        )
    )

    def join_crs() -> str | None:
        first = RasterIndex.from_transform(_raster_affine(), width=4, height=3, crs="EPSG:32633")
        second = RasterIndex.from_transform(_raster_affine(), width=4, height=3, crs="EPSG:32633")
        joined = first.join(second, how="inner")
        return None if joined.crs is not None else "RasterIndex.join returns crs=None"

    cases.append(
        Case(
            "rasterix_join_crs",
            "rasterix",
            "RasterIndex.join of two EPSG:32633 indexes",
            "keep the shared CRS",
            "not a hook: the extension's own join",
            "rasterix RasterIndex.join",
            _run("RasterIndex", join_crs),
        )
    )
    return cases


def _raster_affine() -> Any:
    from rasterio.transform import Affine

    return Affine(2, 0, 100, 0, -3, 200)


# --- xproj ---------------------------------------------------------------------------------


def _xproj_cases() -> list[Case]:
    import pyproj
    import xproj  # noqa: F401 - registers the proj accessor
    from xproj import CRSIndex

    def array(crs: str, value: float, index_class: type[xr.Index] | None) -> xr.DataArray:
        result = xr.DataArray([value], dims="x", coords={"x": [0]}).proj.assign_crs(spatial_ref=crs)
        if index_class is not None:
            result = result.drop_indexes("spatial_ref").set_xindex(
                "spatial_ref", index_class, crs=crs
            )
        return result

    def override(index_class: type[xr.Index] | None) -> str | None:
        first = array("EPSG:4326", 1.0, index_class)
        second = array("OGC:CRS84", 2.0, None)
        aligned = xr.align(first, second, join="override")[1]
        if aligned.proj.crs == pyproj.CRS.from_user_input("EPSG:4326"):
            return "OGC:CRS84 operand relabelled EPSG:4326"
        return None

    hooked, calls = _hooked(CRSIndex, hooks=["override"])
    cases = [
        Case(
            "xproj_override_crs",
            "xproj",
            "align(EPSG:4326, OGC:CRS84, join='override')",
            "refuse, or keep each operand's CRS",
            "B (join_overlapping, how='override')",
            _OVERRIDE,
            _run("CRSIndex", lambda: override(None)),
            _run(hooked.__name__, lambda: override(hooked), calls),
        )
    ]

    def renamed() -> str | None:
        first = array("EPSG:4326", 1.0, None)
        second = xr.DataArray([2.0], dims="x", coords={"x": [0]}).proj.assign_crs(crs="OGC:CRS84")
        result = first + second
        names = sorted(
            name for name, index in result.xindexes.items() if isinstance(index, CRSIndex)
        )
        return f"two CRS coordinates coexist: {names}" if len(names) > 1 else None

    cases.append(
        Case(
            "xproj_two_crs_names",
            "xproj",
            "EPSG:4326 on spatial_ref + OGC:CRS84 on crs",
            "refuse mixed CRSs",
            "not a hook: different coordinate names never overlap",
            "xarray/structure/merge.py:merge_coordinates_without_align",
            _run("CRSIndex", renamed),
        )
    )
    return cases


# --- cf-xarray -----------------------------------------------------------------------------


def _cf_cases() -> list[Case]:
    import cf_xarray  # noqa: F401 - registers the cf accessor

    def coarsen_bounds() -> str | None:
        ds = xr.Dataset({"t": (("x",), np.arange(4.0))}, coords={"x": [0.0, 1, 2, 3]})
        ds.x.attrs["axis"] = "X"
        result = ds.cf.add_bounds("x").coarsen(x=2).reduce(np.mean)
        bounds = np.asarray(result["x_bounds"])
        if not np.allclose(bounds[0], [-0.5, 1.5]):
            return f"bounds averaged to {bounds[0].tolist()} instead of [-0.5, 1.5]"
        return None

    return [
        Case(
            "cf_coarsen_bounds",
            "cf-xarray",
            "coarsen(x=2).mean() with x_bounds",
            "combine cell bounds, or refuse",
            "not a hook today: bounds are attribute-associated; A would apply to an index owning x and x_bounds",
            _COARSEN,
            _run("PandasIndex + attrs", coarsen_bounds),
        )
    ]


# --- driver --------------------------------------------------------------------------------


def _provenance() -> dict[str, Any]:
    module = Path(xr.__file__).resolve()
    checkout = module.parent.parent
    # Only a source tree counts: a site-packages install can sit inside another repository.
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--show-toplevel", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    toplevel, _, head = commit.stdout.strip().partition("\n")
    is_checkout = commit.returncode == 0 and Path(toplevel).resolve() == checkout
    return {
        "python": sys.executable,
        "xarray_module": str(module),
        "xarray_version": xr.__version__,
        "xarray_commit": head if is_checkout else None,
        "has_check_drop": hasattr(xr.Index, "check_drop"),
        "has_join_overlapping": hasattr(xr.Index, "join_overlapping"),
        "libraries": {
            name: version(name)
            for name in ("pint-xarray", "pint", "rasterix", "xproj", "cf-xarray", "pyproj")
        },
    }


def run() -> dict[str, Any]:
    """Measure every case on the imported xarray."""
    cases = [*_pint_cases(), *_rasterix_cases(), *_xproj_cases(), *_cf_cases()]
    for case in cases:
        case.verdict = _verdict(case)
    return {**_provenance(), "cases": [asdict(case) for case in cases]}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
