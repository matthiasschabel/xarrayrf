"""Record which xarray Index hooks fire at operation boundaries.

Two stage-2 unknowns motivate this probe:

* whether ``Index.should_add_coord_to_array`` can tell which data variable owns a
  binding, or only which dimensions the result has;
* whether any existing public hook observes every operand of a binary op, NumPy ufunc,
  ``apply_ufunc`` or ``where`` before xarray discards a conflicting declaration.

It reuses the rejected scalar guard from ``scalar_binding_probe`` and only adds
recording, so the observed behavior is that of stock xarray. Nothing here is a supported
mechanism, and nothing here modifies xarray or its installed source.
"""

import json
from collections.abc import Callable

import numpy as np
import xarray as xr
from scalar_binding_probe import MARKER, FrameIndex, framed

_CALLS: list[str] = []


def _describe(value: object) -> str:
    """Render a hook argument compactly enough to compare across cases."""
    if isinstance(value, FrameIndex):
        return f"{type(value).__name__}(declaration={value.variable.item()})"
    if isinstance(value, xr.Variable):
        return f"Variable(dims={value.dims}, shape={value.shape})"
    if isinstance(value, set | frozenset):
        return "{" + ", ".join(sorted(str(item) for item in value)) + "}"
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


class RecordingFrameIndex(FrameIndex):
    """The scalar guard, instrumented to report which hooks xarray actually calls."""

    def _record(self, hook: str, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        rendered = [_describe(value) for value in args]
        rendered += [f"{key}={_describe(value)}" for key, value in kwargs.items()]
        _CALLS.append(f"{hook}({', '.join(rendered)})")

    def should_add_coord_to_array(self, *args, **kwargs):
        self._record("should_add_coord_to_array", args, kwargs)
        return super().should_add_coord_to_array(*args, **kwargs)

    def equals(self, *args, **kwargs):
        self._record("equals", args, kwargs)
        return super().equals(*args, **kwargs)

    def join(self, *args, **kwargs):
        self._record("join", args, kwargs)
        return super().join(*args, **kwargs)

    def reindex_like(self, *args, **kwargs):
        self._record("reindex_like", args, kwargs)
        return super().reindex_like(*args, **kwargs)

    def isel(self, *args, **kwargs):
        self._record("isel", args, kwargs)
        return super().isel(*args, **kwargs)

    def rename(self, *args, **kwargs):
        self._record("rename", args, kwargs)
        return super().rename(*args, **kwargs)


def recording(origin: int = 10) -> xr.DataArray:
    """Build the probe array carrying the recording index."""
    return framed(origin).drop_indexes(MARKER).set_xindex(MARKER, RecordingFrameIndex)


def _summarize(result: object) -> dict[str, object]:
    if isinstance(result, xr.DataArray):
        return {
            "sizes": dict(result.sizes),
            "declaration": MARKER in result.coords,
            "guard": MARKER in result.xindexes,
        }
    return {"result_type": type(result).__name__}


def _ownership_answer(report: dict[str, object]) -> str:
    owner = report.get("owning variable extraction")
    sibling = report.get("sibling variable extraction")
    if not isinstance(owner, dict) or not isinstance(sibling, dict):
        return "not observed"
    if owner.get("hooks") == sibling.get("hooks"):
        return "no: the owning and unrelated variable produced identical hook arguments"
    return "hook arguments differ between the two variables; read the recorded calls"


def observations() -> dict[str, object]:
    """Report hooks and outcomes per case without treating any of it as support."""
    image = recording()
    other = recording(20)
    unguarded = recording(20).drop_indexes(MARKER)
    coords = {"y": [0, 2, 4], "x": [0, 3, 6, 9]}
    bare = xr.DataArray(np.ones((3, 4)), dims=("y", "x"), coords=coords)
    sibling = xr.DataArray(np.zeros((3, 4)), dims=("y", "x"), coords=coords)

    dataset = None
    dataset_error = None
    try:
        dataset = xr.Dataset({"image": image, "sibling": sibling})
    except Exception as exc:
        # A probe reports a failure as an observation instead of aborting the run.
        dataset_error = f"{type(exc).__name__}: {exc}"

    cases: dict[str, Callable[[], object]] = {
        "coordinate extraction": lambda: image.x,
        "binary op guarded conflict": lambda: image + other,
        "binary op unguarded conflict": lambda: image + unguarded,
        "binary op unguarded conflict reflected": lambda: unguarded + image,
        "numpy ufunc bare operand": lambda: np.add(image, bare),
        "numpy ufunc unguarded conflict": lambda: np.add(image, unguarded),
        "where unguarded branch": lambda: xr.where(image > 5, image, unguarded),
        "apply_ufunc unguarded conflict": lambda: xr.apply_ufunc(np.add, image, unguarded),
        "isel crop stride": lambda: image.isel(x=slice(1, None, 2)),
        "isel scalar selection": lambda: image.isel(y=1),
        "rename": lambda: image.rename(x="column"),
    }
    if dataset is not None:
        cases["owning variable extraction"] = lambda: dataset["image"]
        cases["sibling variable extraction"] = lambda: dataset["sibling"]

    report: dict[str, object] = {"xarray_version": xr.__version__}
    if dataset_error is not None:
        report["dataset_construction_error"] = dataset_error
    for name, case in cases.items():
        _CALLS.clear()
        try:
            result = case()
        except Exception as exc:
            report[name] = {
                "error": type(exc).__name__,
                "message": str(exc),
                "hooks": list(_CALLS),
            }
        else:
            report[name] = {**_summarize(result), "hooks": list(_CALLS)}
    report["should_add_coord_to_array_distinguishes_owner"] = _ownership_answer(report)
    return report


if __name__ == "__main__":
    print(json.dumps(observations(), indent=2))
