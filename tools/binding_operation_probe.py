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
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np
import numpy.typing as npt
import xarray as xr
from dask.callbacks import Callback
from numpy.testing import assert_allclose

import xarrayrf.native  # noqa: F401  # register the DataArray accessor
from xarrayrf import (
    AffineTransform,
    ArrayCoordinates,
    CompositeTransform,
    CoordinateSystem,
    DisplacementFieldTransform,
    Grid,
    ReferenceFrame,
)
from xarrayrf._binding import BindingIndex

ATOL = 1e-12


def _case_result(value: object) -> list[xr.DataArray]:
    if isinstance(value, xr.DataArray):
        return [value]
    if isinstance(value, xr.Dataset):
        return [value[name] for name in value.data_vars]
    if isinstance(value, tuple | list):
        return [array for item in value for array in _case_result(item)]
    return []


@dataclass
class Output:
    """One output's domain and pre-operation geometry oracle."""

    domain: xr.DataArray
    source: xr.DataArray | None
    rename: dict[str, str] = field(default_factory=dict)
    dims_map: dict[str, str] = field(default_factory=dict)
    pixels: bool = False
    analytic_field: bool = False
    transform: Any = field(init=False, default=None)
    axes: tuple[str, ...] = field(init=False, default=())
    geometry_dims: tuple[str, ...] = field(init=False, default=())
    original_points: dict[tuple[float, ...], Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.domain = self.domain.copy(deep=True)
        if self.source is None:
            return
        # A shallow pixel copy is sufficient: only eager coordinate values are read.
        self.source = self.source.copy(deep=False)
        self.transform = self.source.rf.coordinate_transform
        self.axes = self.transform.source.axes
        self.geometry_dims = tuple(
            self.dims_map.get(dim, self.rename.get(dim, dim))
            for dim in self.source.rf.geometry_dims
            if self.dims_map.get(dim, self.rename.get(dim, dim)) in self.domain.dims
        )
        labels = _inputs(self.source, self.axes, self.source.rf.geometry_dims)
        points = self.transform.transform_point(labels)
        self.original_points = dict(
            zip(
                map(tuple, labels.reshape(-1, len(self.axes))),
                points.reshape(-1, len(self.transform.target.axes)),
                strict=True,
            )
        )

    def inspect(self, result: xr.DataArray, *, computed: bool = False) -> None:
        assert dict(result.sizes) == dict(self.domain.sizes), "output sizes differ"
        for name, coord in self.domain.coords.items():
            assert name in result.coords, f"missing coordinate {name}"
            assert result.coords[name].dims == coord.dims, f"coordinate role differs: {name}"
            np.testing.assert_array_equal(result.coords[name].values, coord.values)
        if self.source is None:
            assert not result.rf.is_framed, "expected unframed output"
            assert not any(isinstance(index, BindingIndex) for index in result.xindexes.values())
        else:
            assert result.rf.is_framed, "expected preserved binding"
            assert result.rf.geometry_dims == self.geometry_dims, "geometry order differs"
            assert result.rf.reference_frame == self.transform.target, "target identity differs"
            names = tuple(self.rename.get(name, name) for name in self.axes)
            actual_transform = result.rf.coordinate_transform
            assert actual_transform.source.axes == names, "source order differs"
            expected_transform = self.transform
            if self.rename:
                source = self.transform.source
                expected_transform = self.transform.with_endpoints(
                    source=ArrayCoordinates(
                        names,
                        source.units,
                        axis_types=source.axis_types,
                        sample_offset=source.sample_offset,
                    )
                )
            assert actual_transform == expected_transform, "transform differs"
            indexes = [result.xindexes[name] for name in names]
            assert all(index is indexes[0] for index in indexes), "split binding"
            assert isinstance(indexes[0], BindingIndex)
            assert set(indexes[0].fixed) == {
                name for name in names if self.domain.coords[name].ndim == 0
            }, "fixed terms differ"
            owned = indexes[0].create_variables()
            for name in names:
                assert owned[name].dims == self.domain.coords[name].dims
                np.testing.assert_array_equal(owned[name].values, self.domain.coords[name].values)
            lazy_points = result.rf.geometry.points()
            assert lazy_points.dims == (*self.geometry_dims, "axis")
            if isinstance(result.data, da.Array):
                assert isinstance(lazy_points.data, da.Array), "geometry became eager"
            labels = _inputs(result, names, self.geometry_dims)
            expected = self.transform.transform_point(labels)
            # Evaluate coordinate-only geometry eagerly; lazy pixels stay untouched.
            geometry_source = result.copy(data=np.empty(result.shape))
            actual = geometry_source.rf.geometry.points().transpose(*self.geometry_dims, "axis")
            assert actual.shape == (
                *[result.sizes[d] for d in self.geometry_dims],
                len(self.transform.target.axes),
            )
            assert_allclose(actual.values, expected, rtol=0, atol=ATOL)
            if self.analytic_field:
                y, x = labels[..., 0], labels[..., 1]
                assert_allclose(
                    actual.values,
                    np.stack((y + y * x / 100, x + x / 10), axis=-1),
                    rtol=0,
                    atol=ATOL,
                )
            for label, point in zip(
                labels.reshape(-1, len(names)),
                actual.values.reshape(-1, actual.sizes["axis"]),
                strict=True,
            ):
                previous = self.original_points.get(tuple(label))
                if previous is not None:
                    assert_allclose(point, previous, rtol=0, atol=ATOL)
        if self.pixels and (computed or not isinstance(result.data, da.Array)):
            assert_allclose(result.values, self.domain.values, rtol=0, atol=ATOL, equal_nan=True)


def _inputs(array: xr.DataArray, names: tuple[str, ...], dims: tuple[str, ...]) -> npt.NDArray[Any]:
    coords = [xr.DataArray(array.coords[n].values, dims=array.coords[n].dims) for n in names]
    return np.stack([c.transpose(*dims).values for c in xr.broadcast(*coords)], axis=-1)


@dataclass
class Case:
    """An operation separated from its setup and output inspections."""

    call: Callable[[], object]
    outputs: list[Output] = field(default_factory=list)
    error: type[Exception] | None = None
    fragments: tuple[str, ...] = ()
    pending: str | None = None
    unchanged: list[xr.DataArray | xr.Dataset] = field(default_factory=list)
    compute: bool = False


def _exception(exc: Exception, phase: str) -> dict[str, object]:
    return {"phase": phase, "exception": type(exc).__name__, "message": str(exc)}


def _measure(name: str, setup: Callable[[], Case]) -> dict[str, Any]:
    row: dict[str, Any] = {"case": name, "classification": "hole"}
    tasks: list[object] = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        try:
            case = setup()
            before = [value.copy(deep=True) for value in case.unchanged]
            row["expectation"] = (
                {"outcome": "refuse", "exception": case.error.__name__, "fragments": case.fragments}
                if case.error
                else [
                    {
                        "outcome": "preserve" if output.source is not None else "unframe",
                        "sizes": dict(output.domain.sizes),
                        "geometry_dims": output.geometry_dims,
                    }
                    for output in case.outputs
                ]
            )
            if case.pending:
                row["pending_reason"] = case.pending
                row["expectation"] = {"outcome": "pending_review", "reason": case.pending}
        except Exception as exc:
            row.update(_exception(exc, "setup"), classification="setup_error")
            return row
        try:
            value = case.call()
        except Exception as exc:
            row.update(_exception(exc, "operation"), outcome="raises")
            if (
                case.error is not None
                and type(exc) is case.error
                and all(f in str(exc) for f in case.fragments)
            ):
                row["classification"] = "ok"
            try:
                for old, current in zip(before, case.unchanged, strict=True):
                    xr.testing.assert_identical(old, current)
                    assert set(old.xindexes) == set(current.xindexes)
                    for coord, index in old.xindexes.items():
                        assert type(current.xindexes[coord]) is type(index)
                        assert index.equals(current.xindexes[coord])
            except Exception as inspection:
                row["refusal_inspection"] = _exception(inspection, "inspection")
                row["classification"] = "hole"
        else:
            row.update(outcome="returned", phase="inspection")
            try:
                arrays = _case_result(value)
                observations = []
                for position, array in enumerate(arrays):
                    observation: dict[str, object] = {
                        "sizes": dict(array.sizes),
                        "coordinates": list(array.coords),
                        "name": array.name,
                    }
                    try:
                        framed = array.rf.is_framed
                        observation["outcome"] = "preserved" if framed else "unframed"
                        if position < len(case.outputs):
                            case.outputs[position].inspect(array)
                            observation["classification"] = "ok"
                        else:
                            observation["classification"] = "unexpected_output"
                    except Exception as exc:
                        observation.update(_exception(exc, "inspection"), classification="hole")
                    observations.append(observation)
                row["outputs"] = observations
                if (
                    not case.error
                    and len(arrays) == len(case.outputs)
                    and all(o["classification"] == "ok" for o in observations)
                ):
                    row["classification"] = "ok"
            except Exception as exc:
                row.update(_exception(exc, "inspection"))
        row["pixel_tasks_before_compute"] = len(tasks)
        if tasks and name != "compute":
            row.update(
                classification="hole", task_failure="pixel tasks ran before explicit compute"
            )
    if case.compute and row["outcome"] == "returned":
        try:
            arrays = _case_result(value)
            for array, output in zip(arrays, case.outputs, strict=True):
                assert isinstance(array.data, da.Array), "lazy output became eager"
                output.inspect(array.compute(), computed=True)
            row["compute"] = "ok"
        except Exception as exc:
            row.update(classification="hole", compute=_exception(exc, "compute"))
    if case.pending:
        row["classification"] = "pending_review"
    return row


def _legacy_cases() -> list[tuple[str, Callable[[], object], xr.DataArray]]:
    """Run the fixed stage-3 matrix and return machine-readable observations."""
    frame = ReferenceFrame.local(CoordinateSystem(("a", "b"), ("mm", "mm")))
    transform = AffineTransform.from_matrix(
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
    shifted_transform = AffineTransform.from_matrix(
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
    cases: list[tuple[str, Callable[[], object], xr.DataArray]] = []

    def add(name: str, call: Callable[[], object], base: xr.DataArray = image) -> None:
        cases.append((name, call, base))

    add("isel_scalar", lambda: image.isel(y=1))
    add("isel_slice", lambda: image.isel(x=slice(1, None, 2)))
    add("isel_array", lambda: image.isel(x=[3, 0]))
    add("isel_vectorized", lambda: image.isel(x=xr.Variable("points", [0, 1])))
    add("isel_drop_true", lambda: image.isel(y=1, drop=True))
    add("sel_scalar", lambda: image.sel(y=2))
    add("sel_slice", lambda: image.sel(x=slice(3, 9)))
    add("sel_nearest", lambda: image.sel(x=5, method="nearest"))
    add("sel_array", lambda: image.sel(x=[9, 0]))
    add("sel_vectorized", lambda: image.sel(x=xr.DataArray([0, 3], dims="points")))
    add("sel_drop_true", lambda: image.sel(y=2, drop=True))
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
    add("add_different_transform", lambda: image + other)
    add("add_unframed", lambda: image + labelled)
    add("add_unframed_reverse", lambda: labelled + image)
    add("add_unframed_shifted_labels", lambda: image + labelled_shifted)
    add("add_unframed_conflict", lambda: plane + conflict, plane)
    add("add_unframed_conflict_reverse", lambda: conflict + plane, plane)
    add("add_unframed_unit_conflict", lambda: image + labelled_wrong_units)
    add("add_unframed_unit_conflict_reverse", lambda: labelled_wrong_units + image)
    unindexed_wrong_units = labelled_wrong_units.drop_indexes("x")
    add("add_unindexed_unit_conflict", lambda: image + unindexed_wrong_units)
    add(
        "add_unindexed_unit_conflict_reverse",
        lambda: unindexed_wrong_units + image,
    )
    add("add_scalar", lambda: image + 1)
    add("add_scalar_reverse", lambda: 1 + image)
    add("ufunc_framed", lambda: np.add(image, image))
    add("ufunc_unframed", lambda: np.add(image, labelled))
    add("ufunc_unframed_reverse", lambda: np.add(labelled, image))
    add("ufunc_scalar", lambda: np.add(image, 1))
    add("ufunc_scalar_reverse", lambda: np.add(1, image))
    add("ufunc_unary", lambda: np.negative(image))
    add("ufunc_unframed_conflict", lambda: np.add(plane, conflict), plane)
    add("ufunc_unframed_conflict_reverse", lambda: np.add(conflict, plane), plane)
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
        plane,
    )
    for how in ("inner", "outer", "exact", "override"):
        add(
            f"align_{how}",
            partial(xr.align, image, shifted, join=how),
        )
    add("align_unframed", lambda: xr.align(image, labelled))
    add("align_unframed_reverse", lambda: xr.align(labelled, image))
    add("align_unframed_shifted_right", lambda: xr.align(image, labelled_shifted, join="right"))
    add(
        "align_override_framed_first",
        lambda: xr.align(image, labelled, join="override"),
    )
    add(
        "align_override_framed_second",
        lambda: xr.align(labelled, image, join="override"),
    )
    add(
        "broadcast",
        lambda: xr.broadcast(image, xr.DataArray([1, 2], dims="channel"))[0],
    )
    add("broadcast_plane_volume", lambda: xr.broadcast(plane, image), plane)
    add("sum_geometry", lambda: image.sum("y"))
    add("sum_nongeometry", lambda: channel.sum("channel"), base=channel)
    add("coarsen", lambda: image.coarsen(x=2).mean())
    add("rolling", lambda: image.rolling(x=2).mean())
    add("groupby", lambda: image.groupby("y").mean())
    add(
        "resample",
        lambda: (
            image.assign_coords(time=("y", np.array([0, 1, 2], dtype="datetime64[D]")))
            .resample(time="2D")
            .mean()
        ),
    )
    add("stack", lambda: image.stack(pixel=("y", "x")))
    add(
        "unstack",
        lambda: channel.expand_dims(echo=[0, 1]).stack(acquisition=("channel", "echo")).unstack(),
    )
    add("interp", lambda: image.interp(x=[1.5, 4.5]))
    add("reindex", lambda: image.reindex(x=[0, 3, 6, 12]))
    add("reindex_like", lambda: image.reindex_like(shifted))
    add("shift", lambda: image.shift(x=1))
    add("roll_values", lambda: image.roll(x=1))
    add("roll_coords", lambda: image.roll(x=1, roll_coords=True))
    add("roll_empty", lambda: image.isel(x=slice(0, 0)).roll(x=1, roll_coords=True))
    add("pad", lambda: image.pad(x=(1, 1)))
    add("concat_geometry", lambda: xr.concat([image, image], dim="y"))
    add("concat_nongeometry", lambda: xr.concat([image, image], dim="channel"))
    add(
        "merge",
        lambda: xr.merge([image.to_dataset(), image.rename("other").to_dataset()]),
    )
    add("to_dataset_extract_owner", lambda: image.to_dataset()["pixels"])
    add("dataset_extract_sibling", lambda: xr.Dataset({"owner": image, "sibling": raw})["sibling"])
    add("dataset_mean_geometry", lambda: xr.Dataset({"owner": image}).mean("x"))
    add(
        "dataset_quantile_geometry",
        lambda: xr.Dataset({"owner": image}).quantile(0.5, dim="x"),
    )
    add("dataset_setitem_labelled", lambda: xr.Dataset({"owner": image}).assign(sibling=raw * 2))
    add("assign_coords", lambda: image.assign_coords(x=[0, 4, 8, 12]))
    add("drop_vars", lambda: image.drop_vars("x"))
    add("copy", lambda: image.copy())
    add("astype", lambda: image.astype(float))
    add("chunk", lambda: image.chunk({"x": 2}))
    add("compute", lambda: lazy.compute(), base=lazy)
    add("dask_isel", lambda: lazy.isel(x=slice(1, None)), base=lazy)
    add("dask_add", lambda: lazy + 1, base=lazy)
    add("pickle", lambda: pickle.loads(pickle.dumps(image)))
    return cases


def _legacy_case(name: str) -> Case:
    # Contracts: design §§3-5; binding_design Index behaviour / shared-grid / vetoes.
    _, call, source = next(row for row in _legacy_cases() if row[0] == name)
    raw = source.rf.unframe()
    refusals: dict[str, tuple[type[Exception], tuple[str, ...]]] = {
        "isel_vectorized": (
            ValueError,
            ("vectorized indexing changes geometry dimensions; call rf.unframe()",),
        ),
        "sel_vectorized": (
            ValueError,
            ("vectorized indexing changes geometry dimensions; call rf.unframe()",),
        ),
        "isel_drop_true": (ValueError, ("would corrupt the following index", "coordinate(s) 'y'")),
        "sel_drop_true": (ValueError, ("would corrupt the following index", "coordinate(s) 'y'")),
        "add_different_transform": (ValueError, ("incompatible framed coordinate mappings",)),
        # Reviewed: xarray refuses at alignment before the binding is consulted; no binding results.
        "broadcast_plane_volume": (
            xr.AlignmentError,
            ("cannot align objects on coordinate 'y' because of conflicting indexes",),
        ),
        "align_exact": (xr.AlignmentError, ("cannot align objects with join='exact'",)),
        "coarsen": (ValueError, ("cannot coarsen geometry dimensions; call rf.unframe()",)),
        "stack": (ValueError, ("cannot stack geometry dimensions; call rf.unframe()",)),
        # Reviewed: grouping by a bound coordinate is refused before any stacking.
        "groupby": (ValueError, ("would corrupt the following index", "coordinate(s) 'y'")),
        "pad": (ValueError, ("cannot pad geometry dimensions; call rf.unframe()",)),
        "concat_geometry": (
            ValueError,
            ("concatenation of framed arrays is unsupported; call rf.unframe()",),
        ),
        "assign_coords": (ValueError, ("would corrupt the following index", "coordinate(s) 'x'")),
        "drop_vars": (ValueError, ("would corrupt the following index", "coordinate(s) 'x'")),
        # Maintainer decision 2026-10-24: refuse at the operation through Index.check_drop.
        "interp": (
            ValueError,
            ("cannot interpolate geometry dimensions; use rf.resample_to or call rf.unframe()",),
        ),
        "reindex": (
            xr.AlignmentError,
            ("cannot align objects on coordinate 'x' because of conflicting indexes",),
        ),
    }
    for key in ("align_override", "align_override_framed_first", "align_override_framed_second"):
        refusals[key] = (ValueError, ("join='override' is unsupported for framed coordinates",))
    for key in (
        "add_unframed_conflict",
        "add_unframed_conflict_reverse",
        "ufunc_unframed_conflict",
        "ufunc_unframed_conflict_reverse",
        "where_conflicting_coordinate",
    ):
        refusals[key] = (ValueError, ("conflicting unindexed coordinate 'y'",))
    for key in (
        "add_unframed_unit_conflict",
        "add_unframed_unit_conflict_reverse",
        "add_unindexed_unit_conflict",
        "add_unindexed_unit_conflict_reverse",
    ):
        refusals[key] = (ValueError, ("coordinate 'x' declares attrs['units'] = 'cm'",))
    if name in refusals:
        error, fragments = refusals[name]
        return Case(call, error=error, fragments=fragments, unchanged=[source])
    # These domains come from selection/alignment semantics, never the framed result.
    domains = {
        "isel_scalar": lambda: raw.isel(y=1),
        "isel_slice": lambda: raw.isel(x=slice(1, None, 2)),
        "isel_array": lambda: raw.isel(x=[3, 0]),
        "sel_scalar": lambda: raw.sel(y=2),
        "sel_slice": lambda: raw.sel(x=slice(3, 9)),
        "sel_nearest": lambda: raw.sel(x=6),
        "sel_array": lambda: raw.sel(x=[9, 0]),
        "head": lambda: raw.isel(x=slice(0, 2)),
        "tail": lambda: raw.isel(x=slice(2, None)),
        "thin": lambda: raw.isel(x=slice(None, None, 2)),
        "rename": lambda: raw.rename(x="column"),
        "swap_dims": lambda: raw.swap_dims(x="column"),
        "expand_dims": lambda: raw.expand_dims(channel=[0, 1]),
        "squeeze_geometry": lambda: raw.isel(y=1),
        "add_unframed_shifted_labels": lambda: raw.sel(x=[3, 6, 9]),
        "sum_nongeometry": lambda: raw.sum("channel"),
        "broadcast": lambda: raw.expand_dims(channel=2),
        "unstack": lambda: raw.expand_dims(channel=[0, 1], echo=[0, 1]),
        "reindex_like": lambda: raw.reindex(x=[1, 4, 7, 10]),
        "roll_coords": lambda: raw.roll(x=1, roll_coords=True),
        "roll_empty": lambda: raw.isel(x=slice(0, 0)),
        "concat_nongeometry": lambda: raw.expand_dims(channel=2),
        "dask_isel": lambda: raw.isel(x=slice(1, None)),
    }
    unframed = {
        "sum_geometry": lambda: raw.sum("y"),
        "dataset_mean_geometry": lambda: raw.mean("x"),
        "dataset_quantile_geometry": lambda: raw.quantile(0.5, dim="x"),
        "resample": lambda: (
            raw.assign_coords(time=("y", np.array([0, 1, 2], dtype="datetime64[D]")))
            .resample(time="2D")
            .mean()
        ),
    }
    if name in unframed:
        return Case(call, [Output(unframed[name](), None)])
    if name in ("align_inner", "align_outer"):
        labels = [] if name == "align_inner" else [0, 1, 3, 4, 6, 7, 9, 10]
        domain = raw.reindex(x=labels)
        return Case(call, [Output(domain, source), Output(domain, source)])
    if name.startswith("align_unframed"):
        domain = raw.reindex(x=[3, 6, 9, 12]) if name.endswith("shifted_right") else raw
        return Case(call, [Output(domain, source), Output(domain, source)])
    domain = domains[name]() if name in domains else raw
    output = Output(
        domain,
        source,
        rename={"x": "column"} if name == "rename" else {},
        dims_map={"x": "column"} if name == "swap_dims" else {},
    )
    outputs = (
        [output, Output(domain, source)]
        if name in ("merge", "dataset_setitem_labelled")
        else [output]
    )
    return Case(call, outputs)


def _native_case(number: int) -> Case:
    raw = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("y", "x"),
        coords={"y": [0, 2, 4], "x": [0, 3, 6, 9]},
        name="pixels",
    )
    raw = raw.assign_coords(y=raw.y.assign_attrs(units="1"), x=raw.x.assign_attrs(units="1"))
    frame = ReferenceFrame.local(CoordinateSystem(("a", "b"), ("mm", "mm")))
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("y", "x"), ("1", "1")),
        target=frame,
        matrix=np.eye(2),
        translation=np.zeros(2),
    )
    f = raw.rf.frame(transform, dims=("y", "x"))
    px = xr.DataArray([10, 20, 30, 40], dims="x", coords={"x": [3, 6, 9, 12]}, name="profile")
    bad = raw.assign_coords(x=[1, 4, 7, 10]).drop_indexes("x")
    if number in (36, 37, 38, 39):
        # The lower-dimensional plain operand comes first with labels equal to the binding's,
        # so alignment needs no reindexing (the survey's RasterIndex finding).
        px_equal = px.assign_coords(x=[0, 3, 6, 9])
        if number in (36, 37):
            operation = {
                36: lambda: px_equal + f,
                37: lambda: xr.where(px_equal > 0, px_equal, f),
            }[number]
            return Case(operation, [Output(raw, f)])
        build = {
            38: lambda: xr.Dataset({"profile": px_equal, "owner": f}),
            39: lambda: xr.merge([px_equal.to_dataset(), f.to_dataset(name="owner")]),
        }[number]
        return Case(build, [Output(px_equal, None), Output(raw, f)])
    if number in (1, 2, 3, 4, 5):
        domain = raw.reindex(x=[0, 3, 6, 9, 12]) if number == 3 else raw.sel(x=[3, 6, 9])
        if number == 4:
            return Case(
                lambda: xr.align(px, f, join="exact"),
                error=ValueError,
                fragments=("coordinate 'x' differs for join='exact'",),
                unchanged=[px, f],
            )
        if number == 5:
            left, right = f.to_dataset(name="owner"), px.to_dataset()
            # Dataset variable order follows merge input order: profile, owner.
            return Case(
                lambda: xr.merge([right, left], join="inner"),
                [Output(px.sel(x=[3, 6, 9]), None), Output(domain, f)],
            )
        operations = {
            1: lambda: f + px,
            2: lambda: px + f,
            3: lambda: xr.align(f, px, join="outer")[0],
        }
        return Case(operations[number], [Output(domain, f)])
    if number == 6:
        left, right = f.to_dataset(name="owner"), bad.to_dataset(name="incoming")
        return Case(
            lambda: xr.merge([left, right]),
            error=ValueError,
            fragments=("conflicting unindexed coordinate 'x'",),
            unchanged=[left, right],
        )
    if number in (7, 8, 9, 30):
        if number == 30:
            f = raw.copy(data=da.from_array(raw.data, chunks=(2, 2))).rf.frame(
                transform, dims=("y", "x")
            )
        ds = cast(xr.Dataset, f.to_dataset(name="owner"))
        incoming = (
            bad if number == 8 else raw.assign_coords(x=[3, 6, 9, 12]) if number == 9 else raw * 2
        ).to_dataset(name="incoming" if number in (8, 9) else "extra")
        if number == 30:
            incoming = incoming.chunk({"x": 2})

        def update() -> xr.Dataset:
            ds.update(incoming)
            return ds

        if number in (8, 9):
            return Case(
                update,
                error=ValueError if number == 8 else xr.AlignmentError,
                fragments=("conflicting unindexed coordinate 'x'",)
                if number == 8
                else ("cannot update Dataset: coordinate", "Drop the incoming index first"),
                unchanged=[ds, incoming],
            )
        return Case(
            update,
            [Output(raw, f, pixels=True), Output(raw * 2, f, pixels=True)],
            compute=number == 30,
        )
    if number in (10, 21, 23, 35):
        ordered_transform = AffineTransform.from_matrix(
            source=ArrayCoordinates(("x", "y"), ("1", "1")),
            target=frame,
            matrix=[[2.0, 0.5], [-1.0, 3.0]],
            translation=[11, -7],
        )
        ordered = raw.rf.frame(ordered_transform, dims=("x", "y"))
        if number == 10:
            ds = xr.Dataset({"owner": ordered, "series": (("t", "y", "x"), np.ones((2, 3, 4)))})
            return Case(lambda: ds["series"].isel(t=1), [Output(raw, ordered)])
        selected = ordered.isel(y=1)
        if number == 21:
            return Case(
                lambda: selected.rename(y="row", x="column"),
                [
                    Output(
                        raw.isel(y=1).rename(y="row", x="column"),
                        ordered,
                        rename={"y": "row", "x": "column"},
                    )
                ],
            )
        if number == 23:
            return Case(
                lambda: ordered.isel(x=[3, 1], y=slice(1, None)).transpose("y", "x"),
                [Output(raw.isel(x=[3, 1], y=slice(1, None)), ordered)],
            )
        return Case(
            lambda: selected.isel(x=xr.Variable("points", [3, 1])),
            error=ValueError,
            fragments=("vectorized indexing changes geometry dimensions; call rf.unframe()",),
            unchanged=[selected],
        )
    if number == 11:
        ds = xr.Dataset({"owner": f, "profile": ("x", np.arange(4))})
        return Case(lambda: ds["profile"], [Output(raw.x, None)])
    if number == 12:
        ds = cast(xr.Dataset, f.to_dataset(name="owner"))
        return Case(lambda: ds.coords["x"], [Output(raw.x, None)])
    if number in (13, 14, 18, 29):
        if number == 29:
            f = raw.copy(data=da.from_array(raw.data, chunks=(2, 2))).rf.frame(
                transform, dims=("y", "x")
            )
        channel = f.expand_dims(channel=[0, 1])
        if number == 29:
            channel = channel.chunk({"channel": 1})
        if number in (13, 29):
            return Case(
                lambda: channel.sel(y=2, x=[9, 3]).mean("channel"),
                [Output(raw.sel(y=2, x=[9, 3]), f)],
                compute=number == 29,
            )
        if number == 14:
            ds = channel.to_dataset(name="owner")
            return Case(
                lambda: ds.sel(y=2).mean("x")["owner"],
                [Output(raw.expand_dims(channel=[0, 1]).sel(y=2).mean("x"), None)],
            )
        one = channel.isel(y=slice(1, 2), x=slice(2, 3))
        return Case(
            lambda: one.squeeze(("y", "x")).mean("channel"), [Output(raw.isel(y=1, x=2), f)]
        )
    if number in (15, 16):
        point = f.isel(y=1, x=2)
        if number == 15:
            return Case(lambda: point + 3, [Output(raw.isel(y=1, x=2), f)])
        return Case(
            lambda: point.sel(y=2),
            error=KeyError,
            fragments=("source axis 'y' is fixed and cannot be selected by label",),
            unchanged=[point],
        )
    if number == 17:
        empty = f.isel(y=slice(1, 2), x=slice(0, 0))
        return Case(
            lambda: empty.roll(x=1, roll_coords=True),
            [Output(raw.isel(y=slice(1, 2), x=slice(0, 0)), f)],
        )
    if number in (19, 20, 22, 34):
        plane = f.isel(y=1)
        if number == 22:
            return Case(lambda: plane.drop_indexes(["y", "x"]), [Output(raw.isel(y=1), None)])
        one = f.isel(y=slice(1, 2))
        operations = {
            19: lambda: plane.assign_coords(y=2),
            20: lambda: plane.drop_vars("y"),
            34: lambda: one.isel(y=0, drop=True),
        }
        return Case(
            operations[number],
            error=ValueError,
            fragments=("would corrupt the following index", "coordinate(s) 'y'"),
            unchanged=[one if number == 34 else plane],
        )
    if number in (24, 25, 26, 27, 28):
        system = frame.coordinate_system
        anonymous = raw.rf.frame(
            transform.with_endpoints(target=ReferenceFrame.anonymous(system)), dims=("y", "x")
        )
        if number == 24:
            return Case(lambda: px + anonymous, [Output(raw.sel(x=[3, 6, 9]), anonymous)])
        if number == 26:
            derived = anonymous.isel(x=[1, 2])
            return Case(lambda: anonymous + derived, [Output(raw.sel(x=[3, 6]), anonymous)])
        independent = raw.rf.frame(
            transform.with_endpoints(target=ReferenceFrame.local(system)), dims=("y", "x")
        )
        anonymous_other = raw.rf.frame(
            transform.with_endpoints(target=ReferenceFrame.anonymous(system)), dims=("y", "x")
        )
        left, right = (
            (f, independent)
            if number == 25
            else (anonymous, anonymous_other if number == 27 else f)
        )
        return Case(
            lambda: left + right,
            error=ValueError,
            fragments=("left operand is anonymous",)
            if number == 28
            else ("incompatible framed coordinate mappings",),
            unchanged=[left, right],
        )
    if number in (31, 32):
        grid_transform = AffineTransform.from_matrix(
            source=ArrayCoordinates(("y", "x"), ("1", "1"), sample_offset=(0.5, 0.5)),
            target=frame,
            matrix=np.eye(2),
            translation=np.zeros(2),
        )
        grid = Grid(grid_transform, {"y": ("y", [0, 2, 4]), "x": ("x", [0, 3, 6, 9])})
        yy, xx = np.meshgrid(
            np.array([0, 2, 4], dtype=np.float64),
            np.array([0, 3, 6, 9], dtype=np.float64),
            indexing="ij",
        )
        warp = DisplacementFieldTransform(
            source=frame,
            target=ReferenceFrame.local(frame.coordinate_system),
            field=np.stack((yy * xx / 100, xx / 10), axis=-1),
            grid=grid,
            outside="raise",
        )
        field_image = raw.rf.frame(CompositeTransform(transform, warp), dims=("y", "x"))
        if number == 31:
            return Case(
                lambda: field_image.rename(x="column"),
                error=ValueError,
                fragments=("this transform cannot rename source axes",),
                unchanged=[field_image],
            )
        return Case(
            lambda: field_image.swap_dims({"x": "column"}).isel(column=[3, 1]),
            [
                Output(
                    raw.swap_dims(x="column").isel(column=[3, 1]),
                    field_image,
                    dims_map={"x": "column"},
                    analytic_field=True,
                )
            ],
        )
    if number == 33:
        band = xr.DataArray([1, 2], dims="channel")
        return Case(
            lambda: xr.broadcast(f, band, exclude={"x"}),
            [
                Output(raw.expand_dims(channel=2), f),
                Output(
                    xr.DataArray(np.empty((2, 3)), dims=("channel", "y"), coords={"y": raw.y}), None
                ),
            ],
        )
    raise ValueError(f"unknown native row {number}")


def run() -> dict[str, object]:
    """Report the 93 original cases and reviewed N01-N35 without relaxing outcomes."""
    names = [name for name, *_ in _legacy_cases()]
    rows = [_measure(name, partial(_legacy_case, name)) for name in names]
    rows += [_measure(f"N{number:02}", partial(_native_case, number)) for number in range(1, 40)]
    return {
        "xarray_version": xr.__version__,
        "xarray_module": str(Path(xr.__file__).resolve()),
        "absolute_tolerance": ATOL,
        "cases": rows,
        "totals": {
            label: sum(row["classification"] == label for row in rows)
            for label in ("ok", "hole", "pending_review", "setup_error")
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
