# xarray.where and dask.array.from_array are unannotated upstream.
# mypy: disable-error-code="no-untyped-call"
"""Inventory native xarray operations against a tensor component declaration.

Each case states what the advisory contract requires (value_geometry_stage2_plan.md):

- ``keep``: the result keeps the original declaration (renames update its dimensions);
- ``drop``: the result carries no declaration (a scalar, subset or reduction of components);
- ``refuse``: the operation raises;
- ``safe``: raising or dropping are both acceptable, keeping is a documented limitation;
- ``limitation``: a documented path where the declaration survives a value change.

Classification: ``ok`` when the outcome meets the requirement, ``limitation`` when a ``safe`` or
``limitation`` case keeps the declaration, ``hole`` otherwise. Run once per xarray lane.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import cast

import dask.array as da
import numpy as np
import xarray as xr

import xarrayrf as xrf
import xarrayrf.native  # register the DataArray accessor

COMPASS = xrf.DirectionVocabulary("probe-compass", ("west-to-east", "south-to-north", "down-to-up"))


def _frame(*orientation: str, axes: tuple[str, ...]) -> xrf.ReferenceFrame:
    system = xrf.CoordinateSystem(axes, ("m",) * 3, vocabulary=COMPASS, orientation=orientation)
    return xrf.ReferenceFrame.declared(("probe", "site"), system)


ENU = _frame("west-to-east", "south-to-north", "down-to-up", axes=("e", "n", "u"))
WNU = _frame("east-to-west", "south-to-north", "down-to-up", axes=("w", "n", "u"))


def _outcome(value: object, original: xrf.TensorComponentDeclaration) -> tuple[str, bool | None]:
    arrays = (
        [value]
        if isinstance(value, xr.DataArray)
        else [value[name] for name in value.data_vars]
        if isinstance(value, xr.Dataset)
        else [item for item in value if isinstance(item, xr.DataArray)]
        if isinstance(value, tuple | list)
        else []
    )
    declared = [array for array in arrays if array.rf.has_tensor_components]
    if not declared:
        return "dropped", None
    same = all(
        array.rf.tensor_component_declaration.tensor_type == original.tensor_type
        and array.rf.tensor_component_declaration.bases == original.bases
        and array.rf.tensor_component_declaration.packing == original.packing
        for array in declared
    )
    return "kept", same


def _measure(
    name: str,
    call: Callable[[], object],
    original: xrf.TensorComponentDeclaration,
    requirement: str,
) -> dict[str, object]:
    row: dict[str, object] = {"case": name, "requirement": requirement}
    try:
        outcome, same = _outcome(call(), original)
        row["outcome"] = outcome
        if same is not None:
            row["same_declaration"] = same
    except Exception as exc:
        outcome = "raises"
        row.update(
            outcome=outcome, exception=type(exc).__name__, message=str(exc).splitlines()[0][:140]
        )
    if requirement == "keep":
        ok = outcome == "kept" and row.get("same_declaration") is True
    elif requirement == "drop":
        ok = outcome == "dropped"
    elif requirement == "refuse":
        ok = outcome == "raises"
    else:  # safe or limitation
        ok = outcome in ("raises", "dropped")
    row["classification"] = (
        "ok" if ok else "limitation" if requirement in ("safe", "limitation") else "hole"
    )
    return row


def run() -> dict[str, object]:
    raw = xr.DataArray(
        np.arange(6.0).reshape(2, 3), dims=("time", "component"), coords={"time": [0, 1]}
    )
    wind = raw.rf.declare_tensor_components(xrf.VECTOR, component_dims=("component",), bases=ENU)
    covector = raw.rf.declare_tensor_components(
        xrf.COVECTOR, component_dims=("component",), bases=ENU
    )
    flipped = wind.rf.change_component_basis(WNU)
    plain = wind.rf.clear_tensor_components()
    unindexed = plain.drop_indexes("component")
    bare = xr.DataArray(np.ones((2, 3)), dims=("time", "component"), coords={"time": [0, 1]})
    original = wind.rf.tensor_component_declaration

    source = xrf.ArrayCoordinates(("j", "i"), ("1", "1"))
    plane = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("mm", "mm")))
    binding = xrf.AffineTransform.from_matrix(
        source=source, target=plane, matrix=np.diag([2.0, 3.0]), translation=(0.0, 0.0)
    )
    image = (
        xr.DataArray(
            np.ones((2, 3, 2)),
            dims=("j", "i", "component"),
            coords={"j": ("j", [0, 1], {"units": "1"}), "i": ("i", [0, 1, 2], {"units": "1"})},
        )
        .rf.frame(binding, dims=("j", "i"))
        .rf.declare_tensor_components(xrf.VECTOR, component_dims=("component",), bases=source)
    )
    image_declaration = image.rf.tensor_component_declaration

    def payload_replaced() -> xr.DataArray:
        result = wind.copy()
        result.data = wind.data**2
        return cast(xr.DataArray, result)

    def stale_basis() -> xr.DataArray:
        return cast(
            xr.DataArray, image.rename(j="row").rf.change_component_basis(plane, units="mm/s")
        )

    cases: list[tuple[str, Callable[[], object], str]] = [
        ("isel_identity", lambda: wind.isel(component=[0, 1, 2]), "keep"),
        ("isel_scalar", lambda: wind.isel(component=0), "drop"),
        ("isel_subset", lambda: wind.isel(component=[0, 1]), "drop"),
        ("isel_permutation", lambda: wind.isel(component=[2, 1, 0]), "drop"),
        ("isel_other_dim", lambda: wind.isel(time=0), "keep"),
        ("sel_label", lambda: wind.sel(component="n"), "drop"),
        ("sel_all_labels", lambda: wind.sel(component=["e", "n", "u"]), "keep"),
        ("transpose", lambda: wind.transpose(), "keep"),
        ("rename_component_dim", lambda: wind.rename(component="axis"), "keep"),
        ("add_equal", lambda: wind + wind, "keep"),
        ("add_scalar", lambda: wind + 1.0, "keep"),
        ("scale", lambda: 2.0 * wind, "keep"),
        ("add_different_law", lambda: wind + covector, "refuse"),
        ("add_different_basis", lambda: wind + flipped, "refuse"),
        ("add_default_index", lambda: wind + plain, "safe"),
        ("add_default_index_reverse", lambda: plain + wind, "safe"),
        ("add_unindexed_labels", lambda: wind + unindexed, "safe"),
        ("add_unindexed_labels_reverse", lambda: unindexed + wind, "safe"),
        ("add_bare_dimension", lambda: wind + bare, "safe"),
        ("add_bare_dimension_reverse", lambda: bare + wind, "safe"),
        ("square", lambda: wind**2, "limitation"),
        ("abs", lambda: np.abs(wind), "limitation"),
        ("component_product", lambda: wind * wind, "limitation"),
        ("where_component_mask", lambda: wind.where(wind.component != "n", 0.0), "limitation"),
        ("where_constant_mask", lambda: wind.where(wind.time > 0, 0.0), "keep"),
        ("fillna", lambda: wind.where(wind > 1).fillna(0.0), "limitation"),
        ("nanmean_missing", lambda: wind.where(wind > 1).mean("time"), "limitation"),
        ("roll_values", lambda: wind.roll(component=1), "limitation"),
        ("roll_coords", lambda: wind.roll(component=1, roll_coords=True), "refuse"),
        ("shift", lambda: wind.shift(component=1), "limitation"),
        ("payload_replacement", payload_replaced, "limitation"),
        ("mean_time", lambda: wind.mean("time"), "keep"),
        ("sum_component", lambda: wind.sum("component"), "drop"),
        ("align_inner", lambda: xr.align(wind, wind.isel(time=[1]))[0], "keep"),
        ("align_outer", lambda: xr.align(wind, wind.isel(time=[1]), join="outer")[1], "keep"),
        ("align_override", lambda: xr.align(wind, flipped, join="override")[1], "safe"),
        ("concat_time", lambda: xr.concat([wind, wind], "time"), "keep"),
        ("concat_component", lambda: xr.concat([wind, wind], "component"), "refuse"),
        ("stack_component", lambda: wind.stack(flat=("time", "component")), "safe"),
        ("pad_component", lambda: wind.pad(component=(1, 0)), "safe"),
        ("coarsen_component", lambda: wind.coarsen(component=3).mean(), "safe"),
        ("expand_dims", lambda: wind.expand_dims(ensemble=2), "keep"),
        ("dataset_sibling", lambda: xr.Dataset({"a": wind, "b": wind * 2})["b"], "keep"),
        ("copy", lambda: wind.copy(deep=True), "keep"),
        ("astype", lambda: wind.astype(np.float32), "keep"),
        ("chunk", lambda: wind.chunk({"time": 1}), "keep"),
        ("compute", lambda: wind.chunk({"time": 1}).compute(), "keep"),
        ("dask_add", lambda: (wind.chunk({"time": 1}) + wind).compute(), "keep"),
        ("pickle", lambda: pickle.loads(pickle.dumps(wind)), "keep"),
        ("encode_decode", lambda: wind.rf.encode().rf.decode(), "keep"),
    ]
    rows = [_measure(name, call, original, requirement) for name, call, requirement in cases]

    mandel = xrf.SymmetricPacking.mandel(3)
    stress = xr.DataArray(
        np.zeros((2, 6)), dims=("time", "component"), coords={"time": [0, 1]}
    ).rf.declare_tensor_components(
        xrf.SYMMETRIC_SECOND_ORDER_CONTRAVARIANT_TENSOR,
        component_dims=("component",),
        bases=ENU,
        packing=mandel,
    )
    voigt_stress = stress.rf.clear_tensor_components().rf.declare_tensor_components(
        xrf.SYMMETRIC_SECOND_ORDER_CONTRAVARIANT_TENSOR,
        component_dims=("component",),
        bases=ENU,
        packing=xrf.SymmetricPacking.voigt(3),
    )
    packed = stress.rf.tensor_component_declaration
    packed_cases: list[tuple[str, Callable[[], object], str]] = [
        ("packed_isel_identity", lambda: stress.isel(component=list(range(6))), "keep"),
        ("packed_isel_subset", lambda: stress.isel(component=[0, 1, 2]), "drop"),
        ("packed_isel_permutation", lambda: stress.isel(component=[5, 4, 3, 2, 1, 0]), "drop"),
        ("packed_rename", lambda: stress.rename(component="entry"), "keep"),
        ("packed_add_equal", lambda: stress + stress, "keep"),
        ("packed_add_other_packing", lambda: stress + voigt_stress, "refuse"),
        ("packed_concat_component", lambda: xr.concat([stress, stress], "component"), "refuse"),
        ("packed_encode_decode", lambda: stress.rf.encode().rf.decode(), "keep"),
        (
            "packed_unpack_repack",
            lambda: stress.rf.unpack_symmetric_components(
                component_dims=("row", "column")
            ).rf.pack_symmetric_components(mandel, component_dim="component"),
            "keep",
        ),
    ]
    rows += [_measure(name, call, packed, requirement) for name, call, requirement in packed_cases]
    rows.append(
        _measure(
            "resample_to", lambda: image.rf.resample_to(image.rf.grid), image_declaration, "refuse"
        )
    )
    resampled = xrf.TensorComponentDeclaration(xrf.VECTOR, ("component",), plane)
    rows.append(
        _measure(
            "resample_to_with_components",
            lambda: image.rf.resample_to(
                image.rf.grid,
                output_component_basis=plane,
                component_reorientation="jacobian",
                units="mm/s",
            ),
            resampled,
            "keep",
        )
    )
    rows.append(_measure("stale_array_basis", stale_basis, image_declaration, "refuse"))
    rows.append(
        _measure(
            "dask_from_array",
            lambda: xr.DataArray(da.from_array(wind.values), dims=wind.dims, coords=wind.coords),
            original,
            "keep",
        )
    )
    counts = {
        label: sum(row["classification"] == label for row in rows)
        for label in ("ok", "limitation", "hole")
    }
    return {
        "xarray": xr.__version__,
        "xarray_path": str(Path(xr.__file__).parent),
        "counts": counts,
        "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
