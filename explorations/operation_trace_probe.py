"""Trace the scalar baseline's actual xarray call chains with ``sys.setprofile``.

Stage 1 inferred the binary-op and ufunc chains from source and recorded that inference as
unproved. This probe records which of the named functions are really entered, in what order,
for ``a + b``, ``np.add`` in both orders, ``xr.where`` and ``join="override"``.

It records metadata only: qualnames, operand counts, coordinate *names* and element counts per
merge group. No array is read, repr'd or dumped, and no runtime behavior is modified; profiling
observes, it does not patch. The profile hook is always restored in a ``finally``.
"""

import json
import sys

import numpy as np
import xarray as xr
from scalar_binding_probe import MARKER, framed, unguarded

TARGETS = frozenset(
    {
        "DataArray._binary_op",
        "Coordinates._merge_raw",
        "merge_coordinates_without_align",
        "merge_collected",
        "build_output_coords_and_indexes",
        "apply_dataarray_vfunc",
        "align",
        "deep_align",
        "Aligner.align",
        "Aligner.override_indexes",
    }
)

_CALLS: list[dict[str, object]] = []


def _coord_names(value):
    """Name-only view of a coordinate group; never touches values."""
    keys = getattr(value, "keys", None)
    return sorted(str(key) for key in keys()) if callable(keys) else None


def _operand_groups(objects):
    return [_coord_names(getattr(obj, "coords", None)) for obj in objects]


def _metadata(qualname, local):
    """Describe a traced frame from its bound arguments, using metadata only."""
    match qualname:
        case "DataArray._binary_op":
            other = local.get("other")
            return {
                "reflexive": bool(local.get("reflexive")),
                "operands": [type(local.get("self")).__name__, type(other).__name__],
                "other_coords": _coord_names(getattr(other, "coords", None)),
            }
        case "Coordinates._merge_raw":
            return {
                "self_coords": _coord_names(local.get("self")),
                "other_coords": _coord_names(local.get("other")),
            }
        case "merge_coordinates_without_align":
            return {"operand_groups": [_coord_names(obj) for obj in local.get("objects") or []]}
        case "merge_collected":
            grouped = local.get("grouped") or {}
            return {
                "elements_per_name": {str(name): len(items) for name, items in grouped.items()},
                "declaration_elements": len(grouped.get(MARKER, ())),
            }
        case "build_output_coords_and_indexes":
            args = local.get("args") or []
            return {"operand_count": len(args), "operand_groups": _operand_groups(args)}
        case "apply_dataarray_vfunc":
            args = local.get("args") or ()
            return {
                "operand_count": len(args),
                "operand_types": [type(obj).__name__ for obj in args],
                "operand_groups": _operand_groups(args),
            }
        case "Aligner.align" | "Aligner.override_indexes":
            aligner = local.get("self")
            return {"operand_count": len(getattr(aligner, "objects", ()))}
        case _:
            objects = local.get("objects") or ()
            return {"operand_count": len(objects), "operand_groups": _operand_groups(objects)}


def _profile(frame, event, arg):
    if event != "call":
        return
    code = frame.f_code
    qualname = code.co_qualname
    if qualname not in TARGETS:
        return
    module = frame.f_globals.get("__name__", "")
    if not module.startswith("xarray"):
        return
    _CALLS.append({"function": qualname, "module": module, **_metadata(qualname, frame.f_locals)})


def trace(operation):
    """Run ``operation`` under the profiler and always restore the previous profile hook."""
    previous = sys.getprofile()
    _CALLS.clear()
    sys.setprofile(_profile)
    try:
        operation()
    finally:
        sys.setprofile(previous)
    return list(_CALLS)


def observations() -> dict[str, object]:
    """Record the real call chain per operation; never treat any of it as support."""
    previous = sys.getprofile()
    image = framed()
    bare = xr.DataArray(
        np.ones((3, 4)), dims=("y", "x"), coords={"y": [0, 2, 4], "x": [0, 3, 6, 9]}
    )
    cases = {
        "binary forward": lambda: image + bare,
        "binary reflected": lambda: bare + image,
        "numpy add forward": lambda: np.add(image, bare),
        "numpy add reflected": lambda: np.add(bare, image),
        "where": lambda: xr.where(image > 5, image, bare),
        "conflicting unindexed forward": lambda: unguarded(10) + unguarded(20),
        "conflicting unindexed reflected": lambda: unguarded(20) + unguarded(10),
        "guarded conflicting unindexed forward": lambda: framed(10) + unguarded(20),
        "guarded conflicting unindexed reflected": lambda: unguarded(20) + framed(10),
        "guarded scalar": lambda: image + 1,
        "missing guard scalar": lambda: unguarded() + 1,
        "align override": lambda: xr.align(image, framed(20), join="override"),
    }
    report: dict[str, object] = {"xarray_version": xr.__version__, "xarray_source": xr.__file__}
    for name, case in cases.items():
        error = None
        try:
            calls = trace(case)
        except Exception as exc:
            # A probe reports a failure as an observation instead of aborting the run.
            calls = list(_CALLS)
            error = {"type": type(exc).__name__, "message": str(exc)}
        report[name] = {
            "chain": [str(call["function"]) for call in calls],
            "reached_merge_collected": any(c["function"] == "merge_collected" for c in calls),
            "calls": calls,
            **({"error": error} if error else {}),
        }
    report["profile_restored"] = sys.getprofile() is previous
    return report


if __name__ == "__main__":
    print(json.dumps(observations(), indent=2, default=str))
