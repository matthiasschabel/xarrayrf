# xarray.where and dask.array.from_array are unannotated upstream.
# mypy: disable-error-code="no-untyped-call"
"""Inventory native xarray operations against the private reference-frame binding.

The absolute 1e-12 tolerance is far below the unit-spaced synthetic grid and covers
floating-point affine evaluation without treating a displaced sample as correct.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from functools import partial
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

import xarrayrf.native  # noqa: F401  # register the DataArray accessor
from xarrayrf import AffineTransform, ArrayCoordinates, CoordinateSystem, ReferenceFrame

ATOL = 1e-12


def _case_result(value: object) -> list[xr.DataArray]:
    if isinstance(value, xr.DataArray):
        return [value]
    if isinstance(value, xr.Dataset):
        return [value[name] for name in value.data_vars]
    if isinstance(value, tuple | list):
        return [array for item in value for array in _case_result(item)]
    return []


def _correct(original: xr.DataArray, result: xr.DataArray) -> bool:
    """Check points independently, including matching source coordinate labels."""
    try:
        actual = result.rf.geometry.points()
        source = original.rf.coordinate_transform.source.axes
        result_source = result.rf.coordinate_transform.source.axes

        def inputs(array: xr.DataArray, points: xr.DataArray, names: tuple[str, ...]) -> np.ndarray:
            grid = points.isel(axis=0, drop=True)
            components = [
                xr.DataArray(array.coords[name].data, dims=array.coords[name].dims)
                .broadcast_like(grid)
                .transpose(*grid.dims)
                .values
                for name in names
            ]
            return np.stack(components, axis=-1)

        expected_input = inputs(result, actual, result_source)
        expected = original.rf.coordinate_transform.transform_point(expected_input)
        assert_allclose(actual.values, expected, rtol=0, atol=ATOL)
        old_points = original.rf.geometry.points()
        old_input = inputs(original, old_points, source)
        old_by_label = {
            tuple(labels): point
            for labels, point in zip(
                old_input.reshape(-1, len(source)),
                old_points.values.reshape(-1, old_points.sizes["axis"]),
                strict=True,
            )
        }
        for labels, point in zip(
            expected_input.reshape(-1, len(source)),
            actual.values.reshape(-1, actual.sizes["axis"]),
            strict=True,
        ):
            previous = old_by_label.get(tuple(labels))
            if previous is not None:
                assert_allclose(point, previous, rtol=0, atol=ATOL)
        return True
    except (AssertionError, KeyError, TypeError, ValueError):
        return False


def _measure(
    name: str, call: Callable[[], object], original: xr.DataArray, requirement: str
) -> dict[str, object]:
    row: dict[str, object] = {"case": name, "requirement": requirement}
    try:
        arrays = _case_result(call())
        framed = [array for array in arrays if array.rf.is_framed]
        row["outcome"] = "preserved" if arrays and len(framed) == len(arrays) else "unframed"
        if framed:
            row["correct"] = all(_correct(original, array) for array in framed)
    except Exception as exc:
        row.update(
            outcome="raises",
            exception=type(exc).__name__,
            message=str(exc).splitlines()[0][:160],
        )
    outcome = row["outcome"]
    row["classification"] = (
        "ok"
        if (
            (outcome == "preserved" and row.get("correct") is True)
            or (outcome == "raises" and requirement in ("may_refuse", "must_refuse"))
            or (outcome == "unframed" and requirement in ("no_geometry", "no_binding"))
        )
        else "hole"
    )
    if requirement in ("must_refuse", "no_binding") and outcome == "preserved":
        row["classification"] = "hole"
    return row


def run() -> dict[str, object]:
    """Run the fixed stage-3 matrix and return machine-readable observations."""
    frame = ReferenceFrame.local(CoordinateSystem(("a", "b"), ("mm", "mm")))
    transform = AffineTransform(
        source=ArrayCoordinates(("y", "x"), ("1", "1")),
        target=frame,
        matrix=np.eye(2),
        translation=np.zeros(2),
    )
    raw = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("y", "x"),
        coords={"y": [0, 2, 4], "x": [0, 3, 6, 9]},
        name="pixels",
    )
    raw = raw.assign_coords(y=raw.y.assign_attrs(units="1"), x=raw.x.assign_attrs(units="1"))
    image = raw.rf.frame(transform, dims=("y", "x"))
    plane = image.isel(y=1)
    shifted_transform = AffineTransform(
        source=transform.source,
        target=frame,
        matrix=transform.matrix,
        translation=(1, 0),
    )
    other = raw.rf.frame(shifted_transform, dims=("y", "x"))
    shifted = raw.assign_coords(x=[1, 4, 7, 10]).rf.frame(transform, dims=("y", "x"))
    labelled = raw.copy(deep=False)
    labelled_shifted = raw.assign_coords(x=[3, 6, 9, 12])
    labelled_wrong_units = raw.assign_coords(x=raw.x.assign_attrs(units="cm"))
    conflict = xr.DataArray(np.ones(4), dims="x", coords={"x": [0, 3, 6, 9], "y": 4})
    channel = image.expand_dims(channel=[0, 1])
    lazy = raw.copy(data=da.from_array(raw.data, chunks=(2, 2))).rf.frame(
        transform, dims=("y", "x")
    )
    cases: list[tuple[str, Callable[[], object], xr.DataArray, str]] = []

    def add(
        name: str, call: Callable[[], object], requirement: str = "keep", base: xr.DataArray = image
    ) -> None:
        cases.append((name, call, base, requirement))

    add("isel_scalar", lambda: image.isel(y=1))
    add("isel_slice", lambda: image.isel(x=slice(1, None, 2)))
    add("isel_array", lambda: image.isel(x=[3, 0]))
    add("isel_vectorized", lambda: image.isel(x=xr.Variable("points", [0, 1])), "may_refuse")
    add("isel_drop_true", lambda: image.isel(y=1, drop=True), "must_refuse")
    add("sel_scalar", lambda: image.sel(y=2))
    add("sel_slice", lambda: image.sel(x=slice(3, 9)))
    add("sel_nearest", lambda: image.sel(x=5, method="nearest"))
    add("sel_array", lambda: image.sel(x=[9, 0]))
    add("sel_vectorized", lambda: image.sel(x=xr.DataArray([0, 3], dims="points")), "may_refuse")
    add("sel_drop_true", lambda: image.sel(y=2, drop=True), "must_refuse")
    add("head", lambda: image.head(x=2))
    add("tail", lambda: image.tail(x=2))
    add("thin", lambda: image.thin(x=2))
    add("transpose", lambda: image.transpose("x", "y"))
    add("rename", lambda: image.rename(x="column"))
    add("swap_dims", lambda: image.swap_dims({"x": "column"}))
    add("expand_dims", lambda: image.expand_dims(channel=[0, 1]))
    add("squeeze", lambda: plane.expand_dims(channel=[0]).squeeze("channel"), base=plane)
    add("squeeze_geometry", lambda: image.isel(y=slice(1, 2)).squeeze("y"))
    add("add_framed", lambda: image + image)
    add("add_different_transform", lambda: image + other, "must_refuse")
    add("add_unframed", lambda: image + labelled)
    add("add_unframed_reverse", lambda: labelled + image)
    add("add_unframed_shifted_labels", lambda: image + labelled_shifted)
    add("add_unframed_conflict", lambda: plane + conflict, "must_refuse", plane)
    add("add_unframed_conflict_reverse", lambda: conflict + plane, "must_refuse", plane)
    add("add_unframed_unit_conflict", lambda: image + labelled_wrong_units, "must_refuse")
    add("add_unframed_unit_conflict_reverse", lambda: labelled_wrong_units + image, "must_refuse")
    unindexed_wrong_units = labelled_wrong_units.drop_indexes("x")
    add("add_unindexed_unit_conflict", lambda: image + unindexed_wrong_units, "must_refuse")
    add(
        "add_unindexed_unit_conflict_reverse",
        lambda: unindexed_wrong_units + image,
        "must_refuse",
    )
    add("add_scalar", lambda: image + 1)
    add("add_scalar_reverse", lambda: 1 + image)
    add("ufunc_framed", lambda: np.add(image, image))
    add("ufunc_unframed", lambda: np.add(image, labelled))
    add("ufunc_unframed_reverse", lambda: np.add(labelled, image))
    add("ufunc_scalar", lambda: np.add(image, 1))
    add("ufunc_scalar_reverse", lambda: np.add(1, image))
    add("ufunc_unary", lambda: np.negative(image))
    add("ufunc_unframed_conflict", lambda: np.add(plane, conflict), "must_refuse", plane)
    add("ufunc_unframed_conflict_reverse", lambda: np.add(conflict, plane), "must_refuse", plane)
    add("where_method", lambda: image.where(image > 2))
    add("where_function", lambda: xr.where(image > 2, image, 0))
    add("where_unframed_x", lambda: xr.where(image > 2, labelled, image))
    add("where_unframed_y", lambda: xr.where(image > 2, image, labelled))
    add("where_unframed_condition", lambda: xr.where(image > 2, labelled, labelled))
    add("where_framed_x", lambda: xr.where(labelled > 2, image, labelled))
    add("where_framed_y", lambda: xr.where(labelled > 2, labelled, image))
    add(
        "where_conflicting_coordinate",
        lambda: xr.where(conflict > 0, plane, 0),
        "must_refuse",
        plane,
    )
    for how in ("inner", "outer", "exact", "override"):
        add(
            f"align_{how}",
            partial(xr.align, image, shifted, join=how),
            "must_refuse" if how == "override" else "may_refuse",
        )
    add("align_unframed", lambda: xr.align(image, labelled))
    add("align_unframed_reverse", lambda: xr.align(labelled, image))
    add("align_unframed_shifted_right", lambda: xr.align(image, labelled_shifted, join="right"))
    add(
        "align_override_framed_first",
        lambda: xr.align(image, labelled, join="override"),
        "must_refuse",
    )
    add(
        "align_override_framed_second",
        lambda: xr.align(labelled, image, join="override"),
        "must_refuse",
    )
    add(
        "broadcast",
        lambda: xr.broadcast(image, xr.DataArray([1, 2], dims="channel"))[0],
    )
    add("broadcast_plane_volume", lambda: xr.broadcast(plane, image), "must_refuse", plane)
    add("sum_geometry", lambda: image.sum("y"), "no_geometry")
    add("sum_nongeometry", lambda: channel.sum("channel"), base=channel)
    add("coarsen", lambda: image.coarsen(x=2).mean(), "must_refuse")
    add("rolling", lambda: image.rolling(x=2).mean(), "may_refuse")
    add("groupby", lambda: image.groupby("y").mean(), "may_refuse")
    add(
        "resample",
        lambda: (
            image.assign_coords(time=("y", np.array([0, 1, 2], dtype="datetime64[D]")))
            .resample(time="2D")
            .mean()
        ),
        "no_geometry",
    )
    add("stack", lambda: image.stack(pixel=("y", "x")), "must_refuse")
    add(
        "unstack",
        lambda: channel.expand_dims(echo=[0, 1]).stack(acquisition=("channel", "echo")).unstack(),
    )
    add("interp", lambda: image.interp(x=[1.5, 4.5]), "may_refuse")
    add("reindex", lambda: image.reindex(x=[0, 3, 6, 12]), "may_refuse")
    add("reindex_like", lambda: image.reindex_like(shifted), "may_refuse")
    add("shift", lambda: image.shift(x=1))
    add("roll_values", lambda: image.roll(x=1))
    add("roll_coords", lambda: image.roll(x=1, roll_coords=True))
    add("roll_empty", lambda: image.isel(x=slice(0, 0)).roll(x=1, roll_coords=True))
    add("pad", lambda: image.pad(x=(1, 1)), "must_refuse")
    add("concat_geometry", lambda: xr.concat([image, image], dim="y"), "may_refuse")
    add("concat_nongeometry", lambda: xr.concat([image, image], dim="channel"), "may_refuse")
    add(
        "merge",
        lambda: xr.merge([image.to_dataset(), image.rename("other").to_dataset()]),
        "may_refuse",
    )
    add("to_dataset_extract_owner", lambda: image.to_dataset()["pixels"])
    add("dataset_extract_sibling", lambda: xr.Dataset({"owner": image, "sibling": raw})["sibling"])
    add("dataset_mean_geometry", lambda: xr.Dataset({"owner": image}).mean("x"), "no_geometry")
    add(
        "dataset_quantile_geometry",
        lambda: xr.Dataset({"owner": image}).quantile(0.5, dim="x"),
        "no_geometry",
    )
    add("dataset_setitem_labelled", lambda: xr.Dataset({"owner": image}).assign(sibling=raw * 2))
    add("assign_coords", lambda: image.assign_coords(x=[0, 4, 8, 12]), "may_refuse")
    add("drop_vars", lambda: image.drop_vars("x"), "may_refuse")
    add("copy", lambda: image.copy())
    add("astype", lambda: image.astype(float))
    add("chunk", lambda: image.chunk({"x": 2}))
    add("compute", lambda: lazy.compute(), base=lazy)
    add("dask_isel", lambda: lazy.isel(x=slice(1, None)), base=lazy)
    add("dask_add", lambda: lazy + 1, base=lazy)
    add("pickle", lambda: pickle.loads(pickle.dumps(image)))
    return {
        "xarray_version": xr.__version__,
        "xarray_module": str(Path(xr.__file__).resolve()),
        "absolute_tolerance": ATOL,
        "cases": [
            _measure(name, call, base, requirement) for name, call, base, requirement in cases
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
