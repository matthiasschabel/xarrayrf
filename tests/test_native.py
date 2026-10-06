"""Public native binding behavior and named xarray integration gaps."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal, cast

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from dask.callbacks import Callback
from numpy.testing import assert_allclose, assert_array_equal
from xarray.indexes import PandasIndex

import xarrayrf.native  # register the DataArray accessor
from xarrayrf import (
    AffineTransform,
    ArrayCoordinates,
    CompositeTransform,
    CoordinateSystem,
    Geometry,
    Grid,
    ReferenceFrame,
)
from xarrayrf._binding import BindingIndex


@pytest.fixture
def transform() -> AffineTransform:
    frame = ReferenceFrame.local(CoordinateSystem(("a", "b"), ("mm", "mm")))
    return AffineTransform.from_matrix(
        source=ArrayCoordinates(("y", "x"), ("1", "1")),
        target=frame,
        matrix=np.eye(2),
        translation=np.zeros(2),
    )


@pytest.fixture
def image() -> xr.DataArray:
    return xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("y", "x"),
        coords={"y": [0, 2, 4], "x": [0, 3, 6, 9]},
    )


@pytest.fixture
def framed(image: xr.DataArray, transform: AffineTransform) -> xr.DataArray:
    return cast(xr.DataArray, image.rf.frame(transform, dims=("y", "x")))


@contextmanager
def pixel_tasks() -> Iterator[list[object]]:
    tasks: list[object] = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        yield tasks


def test_import_registers_accessor_only_in_native_module() -> None:
    # Registration is tested in a fresh interpreter by the import-boundary suite.
    assert hasattr(xr.DataArray(np.ones(1)), "rf")
    # Native framing helpers share the accessor binding path.
    assert set(xarrayrf.native.__all__) == {
        "CoordinateSpec",
        "DuckArray",
        "Report",
        "frame_array",
        "grid_coordinates",
        "index_coordinate",
    }
    assert not hasattr(xarrayrf.native, "bind")
    assert not hasattr(xarrayrf.native, "decode")


def test_bind_inspect_and_unframe(
    image: xr.DataArray, transform: AffineTransform, framed: xr.DataArray
) -> None:
    assert framed.data is image.data
    assert framed.rf.is_framed
    assert framed.rf.coordinate_transform is transform
    assert framed.rf.reference_frame is transform.target
    assert framed.rf.geometry_dims == ("y", "x")
    assert isinstance(framed.rf.geometry, Geometry)
    assert set(framed.xindexes) == {"x", "y"}
    assert framed.xindexes["x"] is framed.xindexes["y"]
    plain = framed.rf.unframe()
    assert not plain.rf.is_framed
    assert all(type(index).__name__ == "PandasIndex" for index in plain.xindexes.values())
    assert_array_equal(plain.coords["x"], image.coords["x"])


def test_resample_to_binds_target_and_keeps_source_metadata(
    framed: xr.DataArray,
    transform: AffineTransform,
) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    source = framed.expand_dims(echo=[10, 20]).rename("signal")
    source.attrs = {"kind": "synthetic"}
    target = framed.rf.unframe().assign_coords(x=[0, 1, 2, 3]).rf.frame(transform, dims=("y", "x"))
    result = source.rf.resample_to(target)
    assert result.rf.is_framed
    assert result.rf.coordinate_transform == target.rf.coordinate_transform
    assert result.rf.geometry_dims == target.rf.geometry_dims
    assert result.name == "signal"
    assert result.attrs == {"kind": "synthetic"}
    assert result.dims == ("echo", "y", "x")
    assert_array_equal(result.echo, [10, 20])
    assert_array_equal(result.x, target.x)
    via_geometry = source.rf.resample_to(target.rf.geometry)
    assert_allclose(via_geometry.values, result.values, rtol=0, atol=1e-12, equal_nan=True)


def test_resample_to_is_lazy_and_validates_frames(
    framed: xr.DataArray,
    image: xr.DataArray,
) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    lazy = framed.chunk({"y": 2})
    with pixel_tasks() as tasks:
        result = lazy.rf.resample_to(framed)
    assert not tasks
    assert isinstance(result.data, da.Array)
    with pytest.raises(ValueError, match="unframed"):
        image.rf.resample_to(framed)
    with pytest.raises(ValueError, match="unframed"):
        framed.rf.resample_to(image)
    with pytest.raises(TypeError, match="target must be"):
        framed.rf.resample_to(3)


@pytest.mark.parametrize("target_kind", ["grid", "array", "geometry", "general"])
@pytest.mark.parametrize("echo_size", [None, 2, 0])
@pytest.mark.parametrize("lazy", [False, True])
def test_resample_to_scalar_target_preserves_binding_and_context(
    target_kind: str, echo_size: int | None, lazy: bool
) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    frame = ReferenceFrame.local(CoordinateSystem(("position",), ("mm",)))
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("x",), ("mm",)),
        target=frame,
        matrix=[[1.0]],
        translation=[0.0],
    )
    source = xr.DataArray(
        [0.0, 10.0, 20.0],
        dims="x",
        coords={"x": [0.0, 1.0, 2.0], "context": "scan"},
        name="signal",
        attrs={"kind": "synthetic"},
    )
    if echo_size is not None:
        source = source.expand_dims(echo=np.arange(echo_size))
    if lazy:
        source = source.chunk({"x": -1, **({"echo": 1} if echo_size else {})})
    source = source.rf.frame(transform, dims=("x",))
    selected = source.isel(x=1)
    target = {
        "grid": Grid(transform, {"x": 0.5}),
        "array": selected,
        "geometry": selected.rf.geometry,
        "general": Grid(CompositeTransform(transform), {"x": 0.5}),
    }[target_kind]
    with pixel_tasks() as tasks:
        result = source.rf.resample_to(target)
    assert not tasks
    assert result.dims == (() if echo_size is None else ("echo",))
    assert result.rf.is_framed
    assert result.rf.geometry_dims == ()
    assert result.rf.coordinate_transform == (
        target.transform if isinstance(target, Grid | Geometry) else target.rf.coordinate_transform
    )
    assert result.name == "signal"
    assert result.attrs == {"kind": "synthetic"}
    assert result.context.item() == "scan"
    position = 0.5 if target_kind in {"grid", "general"} else 1.0
    assert_allclose(result.x, position, rtol=0, atol=1e-12)
    if echo_size is not None:
        xr.testing.assert_identical(result.echo.variable, source.echo.variable)
    if lazy:
        assert isinstance(result.data, da.Array)
        assert result.chunks == (() if echo_size is None else (source.chunksizes["echo"],))
    expected = np.full(() if echo_size is None else (echo_size,), 10 * position)
    assert_allclose(result.compute(), expected, rtol=0, atol=1e-12)


def test_resample_to_accepts_transform_across_frames(framed: xr.DataArray) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    other = ReferenceFrame.local(framed.rf.reference_frame.coordinate_system)
    original = framed.rf.coordinate_transform
    assert isinstance(original, AffineTransform)
    target_transform = AffineTransform.from_matrix(
        source=original.source,
        target=other,
        matrix=original.matrix,
        translation=original.translation,
    )
    target = framed.rf.unframe().rf.frame(target_transform, dims=("y", "x"))
    frame_map = AffineTransform.from_matrix(
        source=other,
        target=framed.rf.reference_frame,
        matrix=np.eye(2),
        translation=np.zeros(2),
    )
    with pytest.raises(ValueError, match="different frames"):
        framed.rf.resample_to(target)
    result = framed.rf.resample_to(target, transform=frame_map)
    assert result.rf.reference_frame == other
    assert_allclose(result.values, framed.values, rtol=0, atol=1e-12)


def test_assume_frame_adopts_declaration_and_checks_mapping(framed: xr.DataArray) -> None:
    current = framed.rf.reference_frame
    assumed = ReferenceFrame.declared(
        ("test", "shared"),
        current.coordinate_system,
        definition={"space": "shared"},
        context={"cohort": "synthetic"},
    )
    rebound = framed.rf.assume_frame(assumed)
    assert rebound.rf.reference_frame == assumed
    assert rebound.rf.reference_frame.definition == assumed.definition
    assert rebound.rf.reference_frame.context == assumed.context
    with pytest.raises(ValueError, match=r"conflicting indexes|incompatible"):
        _ = framed + rebound
    matching = framed.rf.assume_frame(rebound)
    assert_allclose((rebound + matching).values, 2 * framed.values, rtol=0, atol=1e-12)
    original = framed.rf.coordinate_transform
    assert isinstance(original, AffineTransform)
    scaled = AffineTransform.from_matrix(
        source=original.source,
        target=current,
        matrix=2 * original.matrix,
        translation=original.translation,
    )
    other = framed.rf.unframe().rf.frame(scaled, dims=("y", "x")).rf.assume_frame(rebound)
    with pytest.raises(ValueError, match=r"conflicting indexes|incompatible"):
        _ = rebound + other
    with pytest.raises(ValueError, match="cannot adopt frame: unoriented axis"):
        framed.rf.assume_frame(ReferenceFrame.local(CoordinateSystem(("q", "r"), ("mm", "mm"))))
    with pytest.raises(TypeError, match="other must be"):
        framed.rf.assume_frame("bad")


def test_assume_frame_refuses_non_affine_mapping(
    image: xr.DataArray, transform: AffineTransform
) -> None:
    class PointOnly:
        def __init__(self, affine: AffineTransform) -> None:
            self.source = affine.source
            self.target = affine.target
            self._affine = affine

        def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return self._affine.transform_point(points)

    non_affine = image.rf.frame(PointOnly(transform), dims=("y", "x"))
    with pytest.raises(ValueError, match="requires an affine"):
        non_affine.rf.assume_frame(transform.target)


def test_assume_frame_accepts_a_structural_affine(
    image: xr.DataArray, transform: AffineTransform
) -> None:
    class AffineProxy:
        def __init__(self, affine: AffineTransform) -> None:
            self.source, self.target = affine.source, affine.target
            self.matrix, self.translation = affine.matrix, affine.translation
            self._affine = affine

        def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return self._affine.transform_point(points)

        def jacobian(self, at: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
            return self._affine.jacobian(at)

    other = ReferenceFrame.local(cast(ReferenceFrame, transform.target).coordinate_system)
    adopted = image.rf.frame(AffineProxy(transform), dims=("y", "x")).rf.assume_frame(other)
    assert adopted.rf.reference_frame == other
    assert adopted.rf.coordinate_transform == transform.with_endpoints(target=other)


@pytest.mark.parametrize(
    "member",
    ["coordinate_transform", "reference_frame", "geometry_dims", "geometry", "unframe", "encode"],
)
def test_unframed_access_raises(image: xr.DataArray, member: str) -> None:
    assert not image.rf.is_framed
    with pytest.raises(ValueError, match="unframed"):
        value = getattr(image.rf, member)
        if callable(value):
            value()


def test_frame_is_dataarray_only_and_refuses_rebinding(
    image: xr.DataArray, transform: AffineTransform, framed: xr.DataArray
) -> None:
    with pytest.raises(AttributeError, match="rf"):
        _ = image.to_dataset(name="pixels").rf
    with pytest.raises(ValueError, match="already framed"):
        framed.rf.frame(transform, dims=("y", "x"))


def test_bind_refuses_multidimensional_source_coordinate(transform: AffineTransform) -> None:
    image = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", "x"),
        coords={"y": (("y", "x"), np.ones((2, 3))), "x": [0, 1, 2]},
    )
    with pytest.raises(ValueError, match="0-D or 1-D"):
        image.rf.frame(transform, dims=("y", "x"))


def test_bind_replaces_existing_index(image: xr.DataArray, transform: AffineTransform) -> None:
    bound = image.rf.frame(transform, dims=("y", "x"))
    assert bound.rf.geometry.dims == ("y", "x")


def test_selection_and_rename_keep_valid_geometry(framed: xr.DataArray) -> None:
    assert_array_equal(framed.sel(x=5, method="nearest").coords["x"], 6)
    assert_array_equal(framed.sel(x=slice(3, 9)).coords["x"], [3, 6, 9])
    assert_array_equal(framed.isel(x=slice(1, None, 2)).coords["x"], [3, 9])
    assert_array_equal(framed.isel(x=[3, 0]).coords["x"], [9, 0])
    point = framed.isel(y=1)
    assert point.rf.geometry_dims == ("x",)
    assert point.coords["y"].item() == 2
    assert point.rf.geometry.dims == ("x",)
    renamed = framed.rename(x="column")
    assert renamed.rf.coordinate_transform.source.axes == ("y", "column")
    assert renamed.rf.geometry_dims == ("y", "column")
    assert renamed.rf.geometry.dims == ("y", "column")
    assert framed.transpose("x", "y").rf.geometry.dims == ("y", "x")


def test_binding_that_lost_a_coordinate_is_neither_framed_nor_unframed(
    framed: xr.DataArray,
) -> None:
    broken = framed.isel(y=1).expand_dims("y")  # y gains a default index; x stays bound
    assert isinstance(broken.xindexes["x"], BindingIndex)
    assert not isinstance(broken.xindexes["y"], BindingIndex)
    for member in ("is_framed", "geometry_dims", "coordinate_transform", "geometry"):
        with pytest.raises(ValueError, match=r"no longer owns.*\['y'\]"):
            getattr(broken.rf, member)
    with pytest.raises(ValueError, match="no longer owns"):
        broken.rf.frame(framed.rf.coordinate_transform, dims=("y", "x"))
    plain = broken.rf.unframe()
    assert not plain.rf.is_framed
    assert all(isinstance(index, PandasIndex) for index in plain.xindexes.values())


def test_vectorized_selection_requires_unframe(framed: xr.DataArray) -> None:
    with pytest.raises(ValueError, match=r"rf\.unframe\(\)"):
        framed.isel(x=xr.Variable("points", [0, 1]))


def test_incompatible_framed_operands_raise(framed: xr.DataArray) -> None:
    with pytest.raises(ValueError, match=r"conflicting indexes|incompatible"):
        _ = framed.isel(y=0) + framed.isel(y=1)
    with pytest.raises(ValueError, match=r"conflicting indexes|incompatible"):
        _ = framed.isel(y=0) + framed


def test_join_reindex_and_concat_rejections(framed: xr.DataArray) -> None:
    other = (
        framed.rf.unframe()
        .assign_coords(x=[1, 4, 7, 10])
        .rf.frame(
            framed.rf.coordinate_transform,
            dims=("y", "x"),
        )
    )
    aligned = xr.align(framed, other, join="inner")
    assert all(value.rf.is_framed for value in aligned)
    assert_array_equal(aligned[0].coords["x"], [])
    with pytest.raises(ValueError, match="concatenation"):
        xr.concat([framed, framed], dim="y")


def test_arithmetic_comparison_where_and_ufunc(framed: xr.DataArray) -> None:
    results: tuple[xr.DataArray, ...] = (
        framed + 1,
        1 + framed,
        framed > 2,
        cast(xr.DataArray, np.negative(framed)),
        xr.where(framed > 2, framed, 0),  # type: ignore[no-untyped-call]
    )
    for result in results:
        assert result.rf.is_framed
        assert result.rf.geometry.dims == ("y", "x")


_HAS_INDEX_HOOKS = hasattr(xr.Index, "join_overlapping")


def _supports_mixed_index_alignment() -> bool:
    """Probe #11532 without relying on a version or checkout path."""
    frame = ReferenceFrame.local(CoordinateSystem(("a", "b"), ("mm", "mm")))
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("y", "x"), ("1", "1")),
        target=frame,
        matrix=np.eye(2),
        translation=np.zeros(2),
    )
    plain = xr.DataArray(np.ones((2, 2)), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]})
    try:
        xr.align(plain.rf.frame(transform, dims=("y", "x")), plain)
    except xr.AlignmentError:
        return False
    return True


_HAS_MIXED_ALIGNMENT = _supports_mixed_index_alignment()
_NEEDS_SWAP_DIMS_HOOK = pytest.mark.xfail(
    not hasattr(xr.Index, "swap_dims"),
    strict=True,
    reason="this xarray lane's swap_dims splits multi-coordinate indexes (no Index.swap_dims hook)",
)
_NEEDS_11532 = pytest.mark.xfail(
    not _HAS_MIXED_ALIGNMENT,
    strict=True,
    raises=xr.AlignmentError,
    reason="this xarray lane lacks #11532 mixed joint/PandasIndex alignment",
)
_NEEDS_HOOKS = pytest.mark.xfail(
    not _HAS_INDEX_HOOKS,
    strict=True,
    reason="this xarray lane lacks the optional Index hooks for stage 3b",
)


@_NEEDS_11532
def test_mixed_labelled_arithmetic_and_alignment(framed: xr.DataArray, image: xr.DataArray) -> None:
    for result in (framed + image, xr.align(framed, image)[0]):
        assert result.rf.is_framed
        assert result.rf.geometry.dims == ("y", "x")


@pytest.mark.xfail(
    not _HAS_MIXED_ALIGNMENT,
    strict=True,
    raises=xr.AlignmentError,
    reason="this xarray lane lacks #11532 mixed joint/PandasIndex alignment",
)
@pytest.mark.xfail(
    _HAS_MIXED_ALIGNMENT and not _HAS_INDEX_HOOKS,
    strict=True,
    raises=AssertionError,
    reason="this xarray lane lacks overlapping Index joins that preserve reverse operand order",
)
def test_reflected_mixed_arithmetic_keeps_binding(
    framed: xr.DataArray, image: xr.DataArray
) -> None:
    result = image + framed
    assert result.rf.is_framed
    assert result.rf.geometry.dims == ("y", "x")


@_NEEDS_HOOKS
def test_mixed_ufuncs_keep_binding_in_both_orders(
    framed: xr.DataArray, image: xr.DataArray
) -> None:
    for result in (
        cast(xr.DataArray, np.add(framed, image)),
        cast(xr.DataArray, np.add(image, framed)),
    ):
        assert result.rf.is_framed
        assert result.rf.geometry.dims == ("y", "x")


@_NEEDS_HOOKS
def test_where_keeps_binding_in_every_operand_position(
    framed: xr.DataArray, image: xr.DataArray
) -> None:
    for result in (
        xr.where(framed > 2, image, image),  # type: ignore[no-untyped-call]
        xr.where(image > 2, framed, image),  # type: ignore[no-untyped-call]
        xr.where(image > 2, image, framed),  # type: ignore[no-untyped-call]
    ):
        assert result.rf.is_framed
        assert result.rf.geometry.dims == ("y", "x")


@_NEEDS_HOOKS
@pytest.mark.parametrize(
    ("how", "expected"),
    [
        ("inner", [3, 6, 9]),
        ("outer", [0, 3, 6, 9, 12]),
        ("left", [0, 3, 6, 9]),
        ("right", [3, 6, 9, 12]),
    ],
)
def test_shifted_plain_labels_join(
    framed: xr.DataArray,
    image: xr.DataArray,
    how: Literal["inner", "outer", "left", "right"],
    expected: list[int],
) -> None:
    shifted = image.assign_coords(x=[3, 6, 9, 12])
    first, second = xr.align(framed, shifted, join=how)
    for result in (first, second):
        assert result.rf.is_framed
        assert_array_equal(result.coords["x"], expected)


@_NEEDS_HOOKS
@pytest.mark.parametrize("reverse", [False, True])
def test_align_plain_operand_keeps_binding(
    framed: xr.DataArray, image: xr.DataArray, reverse: bool
) -> None:
    operands = (image, framed) if reverse else (framed, image)
    for result in xr.align(*operands):
        assert result.rf.is_framed
        assert result.rf.geometry.dims == ("y", "x")


@_NEEDS_HOOKS
def test_broadcast_nonspatial_array_keeps_binding(framed: xr.DataArray) -> None:
    expanded, _ = xr.broadcast(framed, xr.DataArray([1, 2], dims="channel"))
    assert expanded.rf.is_framed
    assert expanded.rf.geometry.dims == ("y", "x")
    assert expanded.shape == (3, 4, 2)


@_NEEDS_HOOKS
def test_drop_true_on_geometry_dim_raises(framed: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="corrupt"):
        framed.isel(y=1, drop=True)
    with pytest.raises(ValueError, match="corrupt"):
        framed.sel(y=2, drop=True)


@_NEEDS_HOOKS
@pytest.mark.parametrize("reverse", [False, True])
def test_unframed_coordinate_conflict_raises(framed: xr.DataArray, reverse: bool) -> None:
    plane = framed.isel(y=0)
    other = xr.DataArray(np.ones(4), dims="x", coords={"y": 2})
    with pytest.raises(ValueError, match="coordinate"):
        _ = other + plane if reverse else plane + other


@_NEEDS_HOOKS
@pytest.mark.parametrize("reverse", [False, True])
def test_unframed_unit_conflict_raises(framed: xr.DataArray, reverse: bool) -> None:
    other = framed.rf.unframe().drop_indexes("x")
    other.coords["x"].attrs["units"] = "cm"
    with pytest.raises(ValueError, match=r"coordinate 'x'.*units"):
        _ = other + framed if reverse else framed + other


@_NEEDS_HOOKS
@pytest.mark.parametrize("reverse", [False, True])
def test_indexed_plain_unit_conflict_raises(framed: xr.DataArray, reverse: bool) -> None:
    other = framed.rf.unframe()
    other.coords["x"].attrs["units"] = "cm"
    with pytest.raises(ValueError, match=r"coordinate 'x'.*units"):
        _ = other + framed if reverse else framed + other


@_NEEDS_HOOKS
@pytest.mark.parametrize(
    ("units", "error", "message"),
    [(3, TypeError, "must be a string"), ("mm", ValueError, "declares no unit")],
)
def test_plain_operand_unit_attribute_is_validated(
    image: xr.DataArray, units: object, error: type[Exception], message: str
) -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("a", "b"), ("mm", "mm")))
    undeclared = AffineTransform.from_matrix(
        source=ArrayCoordinates(("y", "x"), (None, None)),
        target=frame,
        matrix=np.eye(2),
        translation=np.zeros(2),
    )
    framed = image.rf.frame(undeclared, dims=("y", "x"))
    other = framed.rf.unframe()
    other.coords["x"].attrs["units"] = units
    with pytest.raises(error, match=message):
        _ = framed + other


@_NEEDS_HOOKS
def test_unindexed_axis_label_conflict_raises(framed: xr.DataArray) -> None:
    other = framed.rf.unframe().assign_coords(x=[1, 4, 7, 10]).drop_indexes("x")
    with pytest.raises(ValueError, match="coordinate 'x'"):
        _ = framed + other


@_NEEDS_HOOKS
@pytest.mark.parametrize("order", ["framed_first", "framed_second", "both_framed"])
def test_align_override_raises(framed: xr.DataArray, order: str) -> None:
    plain = framed.rf.unframe()
    operands = {
        "framed_first": (framed, plain),
        "framed_second": (plain, framed),
        "both_framed": (framed, framed),
    }[order]
    with pytest.raises(ValueError, match="override"):
        xr.align(*operands, join="override")


def _dataset_reduction_drops_whole_binding() -> bool:
    """Probe the xarray fix that drops a multi-coordinate index whole in Dataset reductions."""
    frame = ReferenceFrame.local(CoordinateSystem(("a", "b"), ("mm", "mm")))
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("y", "x"), ("1", "1")),
        target=frame,
        matrix=np.eye(2),
        translation=np.zeros(2),
    )
    plain = xr.DataArray(np.ones((2, 2)), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]})
    reduced = xr.Dataset({"v": plain.rf.frame(transform, dims=("y", "x"))}).mean("x")
    return not reduced.xindexes


def _dataset_update_keeps_binding() -> bool:
    """Probe the xarray fix that keeps a Dataset's own index when update supplies labels."""
    frame = ReferenceFrame.local(CoordinateSystem(("a", "b"), ("mm", "mm")))
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("y", "x"), ("1", "1")),
        target=frame,
        matrix=np.eye(2),
        translation=np.zeros(2),
    )
    plain = xr.DataArray(np.ones((2, 2)), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]})
    dataset = xr.Dataset({"v": plain.rf.frame(transform, dims=("y", "x"))})
    try:
        dataset["w"] = plain
    except xr.AlignmentError:
        return False
    return bool(dataset["v"].rf.is_framed)


_NEEDS_WHOLE_INDEX_REDUCTION = pytest.mark.xfail(
    not _dataset_reduction_drops_whole_binding(),
    strict=True,
    raises=ValueError,  # the split binding is refused as neither framed nor unframed
    reason="this xarray lane's Dataset reductions keep a multi-coordinate index on part of it",
)
_NEEDS_UPDATE_INDEX_PRIORITY = pytest.mark.xfail(
    not _dataset_update_keeps_binding(),
    strict=True,
    raises=(AssertionError, xr.AlignmentError),
    reason="this xarray lane lets update replace a Dataset's index with incoming PandasIndexes",
)


def test_dataset_variables_on_the_grid_share_the_binding(framed: xr.DataArray) -> None:
    dataset = xr.Dataset(
        {
            "owner": framed,
            "sibling": xr.DataArray(np.ones((3, 4)), dims=("y", "x")),
            "series": (("t", "y", "x"), np.ones((2, 3, 4))),
            "profile": ("x", np.arange(4.0)),
            "curve": ("t", [1.0, 2.0]),
            "echo_time": 0.03,
        }
    )
    for name in ("owner", "sibling", "series"):
        assert dataset[name].rf.is_framed
        assert dataset[name].rf.geometry_dims == ("y", "x")
    for name in ("profile", "curve", "echo_time"):
        assert not dataset[name].rf.is_framed
    assert not dataset.x.rf.is_framed


@_NEEDS_11532
def test_labelled_dataset_sibling_shares_the_binding(
    framed: xr.DataArray, image: xr.DataArray
) -> None:
    dataset = xr.Dataset({"owner": framed, "sibling": image * 2})
    assert dataset["sibling"].rf.is_framed
    assert_allclose(
        dataset["sibling"].rf.geometry.points().values,
        framed.rf.geometry.points().values,
        rtol=0,
        atol=1e-12,
    )


def test_dataset_holds_distinct_grids_and_refuses_conflicting_frames(
    framed: xr.DataArray, image: xr.DataArray, transform: AffineTransform
) -> None:
    other_grid = image.rename(y="v", x="u").rf.frame(
        AffineTransform.from_matrix(
            source=ArrayCoordinates(("v", "u"), ("1", "1")),
            target=transform.target,
            matrix=np.eye(2),
            translation=np.ones(2),
        ),
        dims=("v", "u"),
    )
    dataset = xr.Dataset({"first": framed, "second": other_grid})
    assert dataset["first"].rf.geometry_dims == ("y", "x")
    assert dataset["second"].rf.geometry_dims == ("v", "u")
    shifted = framed.rf.unframe().rf.frame(
        AffineTransform.from_matrix(
            source=transform.source,
            target=transform.target,
            matrix=np.eye(2),
            translation=np.ones(2),
        ),
        dims=("y", "x"),
    )
    with pytest.raises(ValueError, match="incompatible framed"):
        xr.Dataset({"first": framed, "shifted": shifted})


@_NEEDS_WHOLE_INDEX_REDUCTION
def test_dataset_reduction_over_geometry_unframes_only_that_grid(
    framed: xr.DataArray, image: xr.DataArray, transform: AffineTransform
) -> None:
    other_grid = image.rename(y="v", x="u").rf.frame(
        AffineTransform.from_matrix(
            source=ArrayCoordinates(("v", "u"), ("1", "1")),
            target=transform.target,
            matrix=np.eye(2),
            translation=np.ones(2),
        ),
        dims=("v", "u"),
    )
    dataset = xr.Dataset({"first": framed, "second": other_grid})
    for reduced in (dataset.mean("x"), dataset.quantile(0.5, dim="x")):
        assert not reduced["first"].rf.is_framed
        assert "y" not in reduced.xindexes
        assert reduced["second"].rf.geometry_dims == ("v", "u")


@_NEEDS_UPDATE_INDEX_PRIORITY
def test_assigning_a_labelled_variable_keeps_the_dataset_binding(
    framed: xr.DataArray, image: xr.DataArray
) -> None:
    dataset = xr.Dataset({"owner": framed})
    dataset["mask"] = image > 3
    assigned = dataset.assign(scaled=image * 2)
    updated = dataset.copy()
    updated.update(xr.Dataset({"extra": image}))
    for result in (dataset, assigned, updated):
        for name in result.data_vars:
            assert result[name].rf.is_framed, name


def test_dask_pixel_tasks_stay_lazy(transform: AffineTransform) -> None:
    pixels = da.ones((3, 4), chunks=(2, 2))
    image = xr.DataArray(pixels, dims=("y", "x"), coords={"y": [0, 2, 4], "x": [0, 3, 6, 9]})
    with pixel_tasks() as tasks:
        bound = image.rf.frame(transform, dims=("y", "x"))
        selected = bound.isel(y=slice(1, None))
        added = selected + 2
        transposed = added.transpose("x", "y")
    assert not tasks
    assert isinstance(transposed.data, da.Array)
    assert transposed.rf.geometry.dims == ("y", "x")


def test_pickle_round_trip(framed: xr.DataArray) -> None:
    restored: Any = pickle.loads(pickle.dumps(framed))
    assert restored.rf.is_framed
    assert restored.rf.coordinate_transform == framed.rf.coordinate_transform
    assert restored.rf.geometry.dims == framed.rf.geometry.dims


def test_fixed_and_auxiliary_source_coordinates(transform: AffineTransform) -> None:
    source = ArrayCoordinates(("y", "x"), ("1", "1"))
    mapped = AffineTransform.from_matrix(
        source=source,
        target=transform.target,
        matrix=transform.matrix,
        translation=transform.translation,
    )
    image = xr.DataArray(
        np.ones(4),
        dims="column",
        coords={"y": 2, "x": ("column", [0, 3, 6, 9])},
    )
    bound = image.rf.frame(mapped, dims=("column",))
    assert bound.rf.geometry_dims == ("column",)
    assert bound.rf.geometry.dims == ("column",)
    selected = bound.isel(column=1)
    assert selected.rf.geometry_dims == ()
    assert selected.coords["x"].item() == 3
    assert selected.coords["y"].item() == 2
    assert selected.rf.geometry.dims == ()


def test_index_constructor_is_private(framed: xr.DataArray) -> None:
    with pytest.raises(ValueError, match=r"rf\.frame"):
        type(framed.xindexes["x"]).from_variables({}, options={})


def test_non_affine_source_rename_is_refused(
    image: xr.DataArray, transform: AffineTransform
) -> None:
    from xarrayrf import CompositeTransform

    bound = image.rf.frame(CompositeTransform(transform), dims=("y", "x"))
    with pytest.raises(ValueError, match="cannot rename source axes"):
        bound.rename(x="column")


def test_transform_mismatch_is_refused(
    image: xr.DataArray, transform: AffineTransform, framed: xr.DataArray
) -> None:
    different = AffineTransform.from_matrix(
        source=transform.source,
        target=transform.target,
        matrix=transform.matrix,
        translation=(1, 0),
    )
    other = image.rf.frame(different, dims=("y", "x"))
    with pytest.raises(ValueError, match="incompatible"):
        _ = framed + other


def test_unframe_removes_binding_without_deleting_coordinates(framed: xr.DataArray) -> None:
    plane = framed.isel(y=1)
    plain = plane.rf.unframe()
    assert not plain.rf.is_framed
    assert plain.coords["y"].item() == 2
    assert "x" in plain.xindexes


def test_rename_retained_fixed_axis(framed: xr.DataArray) -> None:
    plane = framed.isel(y=1)
    renamed = plane.rename(y="row", x="column")
    assert renamed.rf.coordinate_transform.source.axes == ("row", "column")
    assert renamed.rf.geometry_dims == ("column",)
    assert renamed.coords["row"].item() == 2
    assert renamed.rf.geometry.dims == ("column",)


def test_package_import_does_not_register_accessor() -> None:
    import subprocess
    import sys

    script = "import xarray as xr; import xarrayrf; assert not hasattr(xr.DataArray, 'rf')"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_scalar_selection_keeps_fixed_term_attributes(
    image: xr.DataArray, transform: AffineTransform
) -> None:
    labelled = image.assign_coords(
        y=image.y.assign_attrs(units="1"), x=image.x.assign_attrs(units="1")
    )
    plane = labelled.rf.frame(transform, dims=("y", "x")).isel(y=1)
    assert plane.coords["y"].attrs == {"units": "1"}
    assert plane.rf.geometry.dims == ("x",)


def test_points_support_joint_binding_after_scalar_selection(framed: xr.DataArray) -> None:
    points = framed.isel(y=1).rf.geometry.points()
    assert points.dims == ("x", "axis")
    assert_allclose(points.values, [[2, 0], [2, 3], [2, 6], [2, 9]], rtol=0, atol=1e-12)


def test_roll_coordinates_keeps_binding_and_world_points(framed: xr.DataArray) -> None:
    rolled = framed.roll(x=1, roll_coords=True)
    assert rolled.rf.is_framed
    assert_array_equal(rolled.coords["x"], [9, 0, 3, 6])
    assert_allclose(
        rolled.rf.geometry.points().values[..., 1],
        np.broadcast_to([9, 0, 3, 6], (3, 4)),
        rtol=0,
        atol=1e-12,
    )


def _supports_indexed_empty_roll() -> bool:
    empty = xr.DataArray(np.empty(0), dims="x", coords={"x": []})
    try:
        for roll_coords in (False, True):
            empty.roll(x=1, roll_coords=roll_coords)
    except ZeroDivisionError:
        return False
    return True


@pytest.mark.xfail(
    not _supports_indexed_empty_roll(),
    strict=True,
    raises=ZeroDivisionError,
    reason="this xarray lane divides by zero when rolling an empty indexed dimension",
)
@pytest.mark.parametrize("roll_coords", [False, True])
def test_roll_empty_keeps_binding_and_geometry(framed: xr.DataArray, roll_coords: bool) -> None:
    empty = framed.isel(x=slice(0, 0))
    result = empty.roll(x=1, roll_coords=roll_coords)
    assert result.rf.is_framed
    assert result.rf.coordinate_transform is empty.rf.coordinate_transform
    assert result.rf.geometry_dims == ("y", "x")
    assert result.rf.geometry.sizes == {"y": 3, "x": 0}
    assert result.sizes == empty.sizes
    for name in ("y", "x"):
        xr.testing.assert_identical(result.coords[name], empty.coords[name])
    points = result.rf.geometry.points()
    assert points.dims == ("y", "x", "axis")
    assert points.shape == (3, 0, 2)
    xr.testing.assert_identical(points, empty.rf.geometry.points())


def _assert_same_binding_and_points(result: xr.DataArray, original: xr.DataArray) -> None:
    assert result.rf.is_framed
    assert result.rf.coordinate_transform is original.rf.coordinate_transform
    assert result.rf.reference_frame is original.rf.reference_frame
    assert result.rf.geometry_dims == original.rf.geometry_dims
    for name in original.rf.coordinate_transform.source.axes:
        assert result.xindexes[name].equals(original.xindexes[name])
        xr.testing.assert_identical(result.coords[name], original.coords[name])
    # Far below the synthetic grid spacing; allows only affine rounding error.
    assert_allclose(
        result.rf.geometry.points().values,
        original.rf.geometry.points().values,
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.xfail(
    not hasattr(xr.Index, "check_stack"),
    strict=True,
    raises=pytest.fail.Exception,
    reason="this xarray lane lacks the Index.check_stack preflight",
)
@pytest.mark.parametrize("dims", [("y", "x"), ("x", "echo")])
def test_stack_geometry_is_refused(framed: xr.DataArray, dims: tuple[str, str]) -> None:
    source = framed.expand_dims(echo=[0, 1])
    with pytest.raises(ValueError, match=r"stack.*geometry dimensions.*rf\.unframe\(\)"):
        source.stack(sample=dims)


def test_stack_unstack_nongeometry_keeps_binding(framed: xr.DataArray) -> None:
    source = framed.expand_dims(echo=[0, 1], channel=[0, 1])
    stacked = source.stack(acquisition=("echo", "channel"))
    _assert_same_binding_and_points(stacked, source)
    restored = stacked.unstack("acquisition")
    _assert_same_binding_and_points(restored, source)
    xr.testing.assert_identical(restored.transpose(*source.dims), source)


@pytest.mark.xfail(
    not hasattr(xr.Index, "check_pad"),
    strict=True,
    raises=pytest.fail.Exception,
    reason="this xarray lane lacks the Index.check_pad preflight",
)
@pytest.mark.parametrize("width", [(0, 0), (1, 2)])
def test_pad_geometry_is_refused(framed: xr.DataArray, width: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match=r"pad.*geometry dimensions.*rf\.unframe\(\)"):
        framed.pad(x=width)


def test_pad_nongeometry_keeps_binding(framed: xr.DataArray) -> None:
    source = framed.expand_dims(echo=[0, 1])
    result = source.pad(echo=(1, 2))
    assert result.sizes["echo"] == 5
    _assert_same_binding_and_points(result, source)


@pytest.mark.xfail(
    not hasattr(xr.Index, "check_coarsen"),
    strict=True,
    raises=pytest.fail.Exception,
    reason="this xarray lane lacks the Index.check_coarsen preflight",
)
@pytest.mark.parametrize("dataset", [False, True], ids=["dataarray", "dataset"])
def test_coarsen_geometry_is_refused(framed: xr.DataArray, dataset: bool) -> None:
    source = framed.to_dataset(name="signal") if dataset else framed
    with pytest.raises(ValueError, match=r"coarsen.*geometry dimensions.*rf\.unframe\(\)"):
        source.coarsen(x=2)


@pytest.mark.xfail(
    not hasattr(xr.Index, "check_coarsen"),
    strict=True,
    raises=AssertionError,
    reason="this xarray lane lacks coarsen retention of unaffected indexes",
)
@pytest.mark.parametrize("dataset", [False, True], ids=["dataarray", "dataset"])
@pytest.mark.parametrize("operation", ["mean", "construct"])
def test_coarsen_nongeometry_keeps_binding(
    framed: xr.DataArray, dataset: bool, operation: str
) -> None:
    array = framed.expand_dims(echo=[0, 1, 2, 3])
    array = array.copy(data=framed.values + np.arange(4)[:, None, None])
    source = array.to_dataset(name="signal") if dataset else array
    coarsened = source.coarsen(echo=2)
    plain_array = array.rf.unframe()
    plain_source = plain_array.to_dataset(name="signal") if dataset else plain_array
    plain = plain_source.coarsen(echo=2)
    if operation == "mean":
        # xarray adds coarsen reductions dynamically, so their stubs differ between releases.
        result = cast(Any, coarsened).mean()
        expected = plain.mean()
    else:
        result = coarsened.construct(echo=("group", "member"))
        expected = plain.construct(echo=("group", "member"))
    result_array = result["signal"] if dataset else result
    expected_array = expected["signal"] if dataset else expected
    assert isinstance(result_array, xr.DataArray)
    _assert_same_binding_and_points(result_array, array)
    for name in ("y", "x"):
        assert result_array.xindexes[name] is array.xindexes[name]
    xr.testing.assert_identical(result_array.rf.unframe(), expected_array)


def test_bind_refuses_source_coordinates_sharing_a_dimension() -> None:
    frame = ReferenceFrame.local(CoordinateSystem(("a", "b"), ("mm", "mm")))
    transform = AffineTransform.from_matrix(
        source=ArrayCoordinates(("x", "y"), ("1", "1")),
        target=frame,
        matrix=np.eye(2),
        translation=np.zeros(2),
    )
    curve = xr.DataArray(np.ones(2), dims="p", coords={"x": ("p", [0, 1]), "y": ("p", [0, 1])})
    with pytest.raises(ValueError, match="share dimension"):
        curve.rf.frame(transform, dims=("p",))


@_NEEDS_SWAP_DIMS_HOOK
def test_swap_dims_keeps_binding_on_the_replacement_dimension(framed: xr.DataArray) -> None:
    swapped = framed.assign_coords(column=("x", [10, 11, 12, 13])).swap_dims(x="column")
    assert swapped.rf.geometry_dims == ("y", "column")
    assert swapped.coords["x"].dims == ("column",)
    assert swapped.xindexes["x"] is swapped.xindexes["y"]
    assert isinstance(swapped.xindexes["column"], PandasIndex)
    assert_allclose(
        swapped.rf.geometry.points().values,
        framed.rf.geometry.points().values,
        rtol=0,
        atol=1e-12,
    )
    for result in (swapped.isel(column=[3, 0]), swapped.sel(column=11)):
        assert result.rf.is_framed
    assert_array_equal(swapped.sel(column=11).coords["x"], 3)


@_NEEDS_SWAP_DIMS_HOOK
def test_swap_dims_onto_a_new_name_keeps_binding(framed: xr.DataArray) -> None:
    swapped = framed.swap_dims(y="row")
    assert swapped.rf.geometry_dims == ("row", "x")
    assert swapped.coords["y"].dims == ("row",)
    assert_allclose(
        swapped.rf.geometry.points().values,
        framed.rf.geometry.points().values,
        rtol=0,
        atol=1e-12,
    )


def test_incompatible_bindings_say_why(framed: xr.DataArray, transform: AffineTransform) -> None:
    other_frame = ReferenceFrame.local(framed.rf.reference_frame.coordinate_system)
    elsewhere = framed.rf.assume_frame(other_frame)
    with pytest.raises(ValueError, match=r"different reference frames.*rf\.assume_frame"):
        _ = framed + elsewhere
    regridded = framed.rf.unframe().rf.frame(
        AffineTransform.from_matrix(
            source=transform.source,
            target=transform.target,
            matrix=2 * np.eye(2),
            translation=np.zeros(2),
        ),
        dims=("y", "x"),
    )
    with pytest.raises(ValueError, match=r"same frame on different grids.*rf\.resample_to"):
        _ = framed + regridded


def test_frame_array_keeps_array_api_storage_and_refuses_lists(
    transform: AffineTransform,
) -> None:
    class Namespaced:
        """Minimal array-API object: no NumPy protocols, only ``__array_namespace__``."""

        def __init__(self, values: npt.NDArray[np.float64]) -> None:
            self._values = values

        ndim = property(lambda self: self._values.ndim)
        shape = property(lambda self: self._values.shape)
        dtype = property(lambda self: self._values.dtype)

        def __array_namespace__(self, api_version: str | None = None) -> Any:
            return np

        def __getitem__(self, key: Any) -> Namespaced:
            return Namespaced(self._values[key])

        def __array__(self, dtype: Any = None, copy: Any = None) -> npt.NDArray[Any]:
            return np.asarray(self._values, dtype=dtype)

    coords = {
        "y": xarrayrf.native.index_coordinate("y", 3),
        "x": xarrayrf.native.index_coordinate("x", 4),
    }
    grid = xarrayrf.Grid(transform, {name: (name, spec[1]) for name, spec in coords.items()})
    storage = Namespaced(np.arange(12.0).reshape(3, 4))
    framed = xarrayrf.native.frame_array(storage, grid)
    assert framed.data is storage  # accepted as is, no eager conversion
    assert framed.rf.geometry_dims == ("y", "x")
    with pytest.raises(TypeError, match="duck array, got list"):
        xarrayrf.native.frame_array([[0.0] * 4] * 3, grid)


def test_core_grid_resampling_does_not_register_accessor() -> None:
    import subprocess
    import sys

    script = """
import numpy as np
import xarray as xr
import xarrayrf as xrf
frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x",), ("mm",)))
transform = xrf.AffineTransform.from_matrix(
    source=xrf.ArrayCoordinates(("i",), ("1",)), target=frame,
    matrix=((2.0,),), translation=(0.0,),
)
grid = xrf.Grid(transform, {"i": ("i", [0, 1, 2])})
array = xr.DataArray(np.zeros(3), dims="i", coords=dict(grid.coordinates))
source = xrf.Geometry(array, transform, dims=("i",))
result = xrf.resample(source, grid)
assert not hasattr(result, "rf")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_anonymous_adoption_keeps_different_grids_and_names_resampling(
    framed: xr.DataArray,
) -> None:
    system = framed.rf.reference_frame.coordinate_system
    first = framed.rf.assume_frame(ReferenceFrame.anonymous(system))
    original = first.rf.coordinate_transform
    assert isinstance(original, AffineTransform)
    shifted = original.with_endpoints(target=ReferenceFrame.anonymous(system))
    shifted = AffineTransform.from_matrix(
        source=shifted.source,
        target=shifted.target,
        matrix=shifted.matrix,
        translation=shifted.translation + shifted.matrix @ [2, 0],
    )
    second = framed.rf.unframe().rf.frame(shifted, dims=framed.rf.geometry_dims)
    with pytest.raises(
        ValueError,
        match=r"are anonymous.*grids differ.*frame=.*rf.assume_frame.*then use rf.resample_to",
    ):
        _ = first + second
    with pytest.raises(ValueError, match=r"are anonymous.*grids differ.*then use rf.resample_to"):
        first.rf.resample_to(second)
    adopted = second.rf.assume_frame(first)
    with pytest.raises(ValueError, match=r"same frame on different grids.*rf.resample_to"):
        _ = first + adopted
    onto_first = adopted.rf.resample_to(first)
    onto_second = first.rf.resample_to(adopted)
    assert onto_first.rf.grid == first.rf.grid
    assert onto_second.rf.grid == adopted.rf.grid
    assert_allclose(onto_first.values[1:, :], framed.values[:-1, :], rtol=0, atol=1e-12)


@pytest.mark.parametrize("anonymous_left", [True, False])
def test_anonymous_refusal_identifies_one_operand(
    framed: xr.DataArray, anonymous_left: bool
) -> None:
    anonymous = framed.rf.assume_frame(
        ReferenceFrame.anonymous(framed.rf.reference_frame.coordinate_system)
    )
    left, right = (anonymous, framed) if anonymous_left else (framed, anonymous)
    label = "left" if anonymous_left else "right"
    with pytest.raises(
        ValueError, match=rf"{label} operand is anonymous.*points match numerically.*alone suffices"
    ):
        _ = left + right
    with pytest.raises(
        ValueError,
        match=rf"{'source' if anonymous_left else 'target'} operand is anonymous.*alone suffices",
    ):
        left.rf.resample_to(right)


def test_empty_anonymous_bindings_recommend_adoption_alone(framed: xr.DataArray) -> None:
    first = framed.isel(x=slice(0, 0)).rf.assume_frame(
        ReferenceFrame.anonymous(framed.rf.reference_frame.coordinate_system)
    )
    second = first.rf.assume_frame(
        ReferenceFrame.anonymous(first.rf.reference_frame.coordinate_system)
    )
    operations: tuple[Callable[[], xr.DataArray], ...] = (
        lambda: first + second,
        lambda: second.rf.resample_to(first),
    )
    for operation in operations:
        with pytest.raises(ValueError, match=r"rf.assume_frame alone suffices") as error:
            operation()
        assert "then use rf.resample_to" not in str(error.value)
    adopted = second.rf.assume_frame(first)
    assert adopted.rf.grid == first.rf.grid
    assert (first + adopted).rf.grid == first.rf.grid


@pytest.mark.parametrize("domain", ["samples", "cells"])
@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize("empty_source", [True, False])
@pytest.mark.parametrize("lazy", [True, False])
def test_resample_to_empty_target(
    framed: xr.DataArray, domain: str, method: str, empty_source: bool, lazy: bool
) -> None:
    source = framed.expand_dims(echo=[10, 20]).rename("signal").assign_attrs(note="retained")
    target = framed.isel(x=slice(0, 0))
    if empty_source:
        source = source.isel(x=slice(0, 0))
    if lazy:
        source = source.chunk({"echo": 1, "y": 2, "x": 2})
    with pixel_tasks() as tasks:
        result = source.rf.resample_to(target, method=method, domain=domain)
        assert not tasks
    if lazy:
        assert isinstance(result.data, da.Array)
        result = result.compute()
    assert result.rf.grid == target.rf.grid
    assert result.shape == (2, 3, 0)
    assert result.name == source.name
    assert result.attrs == source.attrs
    xr.testing.assert_identical(result.echo, source.echo)


@pytest.mark.parametrize("domain", ["samples", "cells"])
def test_resample_to_empty_source_refuses_nonempty_target(
    framed: xr.DataArray, domain: str
) -> None:
    source = framed.isel(x=slice(0, 0))
    with pytest.raises(ValueError, match=r"empty source.*nothing to sample from"):
        source.rf.resample_to(framed, domain=domain, fill_value=-9)


def test_anonymous_underivable_change_names_reason_without_assumption_remedy(
    framed: xr.DataArray,
) -> None:
    frame = ReferenceFrame.anonymous(CoordinateSystem(("q", "r"), ("mm", "mm")))
    other = framed.rf.unframe().rf.frame(
        cast(AffineTransform, framed.rf.coordinate_transform).with_endpoints(target=frame),
        dims=framed.rf.geometry_dims,
    )
    operations: tuple[Callable[[], xr.DataArray], ...] = (
        lambda: framed + other,
        lambda: framed.rf.resample_to(other),
    )
    for operation in operations:
        with pytest.raises(ValueError, match=r"not derivable.*unoriented axis") as error:
            operation()
        assert "rf.assume_frame" not in str(error.value)
        assert "frame=" not in str(error.value)
    with pytest.raises(ValueError, match="cannot adopt frame: unoriented axis"):
        framed.rf.assume_frame(other)


def test_numerically_matching_points_do_not_bypass_binding_checks(framed: xr.DataArray) -> None:
    first = framed.rf.assume_frame(
        ReferenceFrame.anonymous(framed.rf.reference_frame.coordinate_system)
    )
    original = cast(AffineTransform, first.rf.coordinate_transform)
    second = (
        first.rf.unframe()
        .assign_coords(y=first.y / 2, x=first.x / 2)
        .rf.frame(
            AffineTransform.from_matrix(
                source=original.source,
                target=ReferenceFrame.anonymous(first.rf.reference_frame.coordinate_system),
                matrix=2 * original.matrix,
                translation=original.translation,
            ),
            dims=first.rf.geometry_dims,
        )
    )
    with pytest.raises(
        ValueError,
        match=r"points match numerically but their bindings differ.*then use rf.resample_to",
    ):
        _ = first + second
    assumed = second.rf.assume_frame(first)
    with pytest.raises(ValueError, match="different grids"):
        _ = first + assumed
    assert_allclose(assumed.rf.resample_to(first), first, rtol=0, atol=1e-12)


def test_complete_frame_refusal_wording_is_unchanged(framed: xr.DataArray) -> None:
    other = framed.rf.assume_frame(
        ReferenceFrame.local(framed.rf.reference_frame.coordinate_system)
    )
    mine, theirs = framed.rf.reference_frame.identifier, other.rf.reference_frame.identifier
    expected = (
        f"incompatible framed coordinate mappings: the operands are in different reference frames "
        f"({mine[0]}:{mine[1]} and {theirs[0]}:{theirs[1]}); resample one onto the other "
        "with a transform between the frames, or use rf.assume_frame if they are the same space"
    )
    with pytest.raises(ValueError) as error:
        _ = framed + other
    assert str(error.value) == expected
    with pytest.raises(ValueError) as error:
        framed.rf.resample_to(other)
    assert str(error.value) == (
        f"the target frame {theirs} and the source frame {mine} are "
        "different frames; supply the transform between them, such as a registration result"
    )


def test_anonymous_non_affine_refusal_names_the_mapping_limit(
    image: xr.DataArray, transform: AffineTransform
) -> None:
    class PointOnly:
        def __init__(self, affine: AffineTransform) -> None:
            self.source, self.target = affine.source, affine.target
            self._affine = affine

        def transform_point(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
            return self._affine.transform_point(points)

    original = image.rf.frame(transform, dims=("y", "x"))
    frame = ReferenceFrame.anonymous(original.rf.reference_frame.coordinate_system)
    other = image.rf.frame(PointOnly(transform.with_endpoints(target=frame)), dims=("y", "x"))
    with pytest.raises(
        ValueError, match=r"right operand is anonymous.*requires an affine"
    ) as error:
        _ = original + other
    assert "rf.assume_frame" not in str(error.value)
    with pytest.raises(ValueError, match=r"target operand is anonymous.*requires an affine"):
        original.rf.resample_to(other)


def test_anonymous_adoption_retains_declared_support_checks(framed: xr.DataArray) -> None:
    original = framed.rf.coordinate_transform
    assert isinstance(original, AffineTransform)
    original = original.with_endpoints(
        source=ArrayCoordinates(
            original.source.axes,
            original.source.units,
            sample_offset=(0.5, 0.5),
        )
    )
    first = (
        framed.rf.unframe()
        .rf.frame(original, dims=framed.rf.geometry_dims)
        .rf.assume_frame(ReferenceFrame.anonymous(framed.rf.reference_frame.coordinate_system))
    )
    original = cast(AffineTransform, first.rf.coordinate_transform)
    second = framed.rf.unframe().rf.frame(
        original.with_endpoints(
            target=ReferenceFrame.anonymous(first.rf.reference_frame.coordinate_system)
        ),
        dims=first.rf.geometry_dims,
        intervals={"y": [[-1, 1], [1, 3], [3, 5]]},
    )
    with pytest.raises(
        ValueError, match=r"points match numerically but their bindings differ.*rf.resample_to"
    ):
        _ = first + second
    adopted = second.rf.assume_frame(first)
    assert "y" in adopted.rf.grid.intervals
    with pytest.raises(ValueError, match="incompatible declared intervals"):
        _ = first + adopted
    onto_first = adopted.rf.resample_to(first)
    assert onto_first.rf.grid == first.rf.grid
    assert_allclose(onto_first, first, rtol=0, atol=1e-12)
