"""OME-Zarr 0.6 export through the public adapter."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from ome_zarr_models.v06 import coordinate_transforms as ct
from ome_zarr_models.v06.multiscales import Multiscale

from xarrayrf import (
    AffineTransform,
    ArrayCoordinates,
    CompositeTransform,
    CoordinateSystem,
    DirectionVocabulary,
    Geometry,
    ReferenceFrame,
)
from xarrayrf.anatomy import RAS, VOCABULARY
from xarrayrf.ngff import from_multiscale, to_multiscale_level, to_transform


def _geometry(*, diagonal: bool = True, offset: float = 0.5) -> Geometry:
    frame = ReferenceFrame.local(
        CoordinateSystem(("y", "x"), ("um", "um"), axis_types=("space", "space"))
    )
    matrix = np.diag([2.0, 3.0]) if diagonal else np.array([[2.0, 0.5], [0.0, 3.0]])
    source = ArrayCoordinates(("j", "i"), ("1", "1"), sample_offset=(offset, offset))
    array = xr.DataArray(
        np.zeros((3, 4)),
        dims=("j", "i"),
        coords={dim: (dim, np.arange(size), {"units": "1"}) for dim, size in (("j", 3), ("i", 4))},
    )
    return Geometry(
        array,
        AffineTransform(source=source, target=frame, matrix=matrix, translation=[4.0, 5.0]),
        dims=("j", "i"),
    )


@pytest.mark.parametrize("diagonal", [True, False])
def test_level_validates_and_preserves_lattice(diagonal: bool) -> None:
    geometry = _geometry(diagonal=diagonal)
    metadata, report = to_multiscale_level(geometry, path="s0")
    assert isinstance(metadata, Multiscale)
    assert Multiscale.model_validate(metadata.model_dump()) == metadata
    assert metadata.coordinateSystems[0].name == "intrinsic"
    assert metadata.coordinateSystems[0].axes[0].unit == "micrometer"  # NGFF long name on export
    level = metadata.datasets[0].coordinateTransformations[0]
    assert isinstance(level, ct.Sequence)
    assert [member.type for member in level.transformations] == ["scale", "translation"]
    assert level.input == ct.CoordinateSystemIdentifier(path="s0")
    assert level.output == ct.CoordinateSystemIdentifier(name="intrinsic")
    frame_name = "intrinsic" if diagonal else "physical"
    assert report[0] == (
        "frame-identity",
        f"frame {frame_name!r}: identity {geometry.frame.identifier!r} is not retained",
    )
    assert [code for code, _ in report] == (
        ["frame-identity"] if diagonal else ["frame-identity", "intrinsic-synthesized"]
    )
    if diagonal:
        assert metadata.coordinateTransformations is None
    if not diagonal:
        assert tuple(axis.name for axis in metadata.coordinateSystems[0].axes) == ("j", "i")
        assert tuple(axis.type for axis in metadata.coordinateSystems[0].axes) == ("space", "space")
        assert metadata.coordinateSystems[1].name == "physical"
        assert tuple(axis.name for axis in metadata.coordinateSystems[1].axes) == ("y", "x")
        assert {axis.unit for axis in metadata.coordinateSystems[1].axes} == {"micrometer"}
        assert metadata.coordinateTransformations is not None
        assert len(metadata.coordinateTransformations) == 1
        assert metadata.coordinateTransformations[0].type == "affine"


def test_oblique_level_round_trip_preserves_sample_points() -> None:
    geometry = _geometry(diagonal=False)
    metadata, _ = to_multiscale_level(geometry, path="s0")
    restored = from_multiscale(metadata, shapes={"s0": geometry.array.shape}, store="file:///test")
    assert restored.report == ()
    assert len(restored.transforms) == 1
    level = restored.levels["s0"].transform
    extra = restored.transforms[0]
    assert level.target == extra.source
    points = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(
        extra.transform_point(level.transform_point(points)),
        geometry.lattice().transform_point(points),
        rtol=1e-12,
        atol=1e-12,
    )


def test_shared_names_declare_every_transform_endpoint() -> None:
    geometry = _geometry(diagonal=False)
    other = ReferenceFrame.local(
        CoordinateSystem(("u", "v"), ("um", "um"), axis_types=("space", "space"))
    )
    metadata, _ = to_multiscale_level(
        geometry, path="s0", name="array-space", frame_name="physical"
    )
    names = {geometry.frame: "physical", other: "other"}
    frame_transform = AffineTransform(
        source=geometry.frame, target=other, matrix=np.eye(2), translation=np.zeros(2)
    )
    exported, _ = to_transform(frame_transform, names=names)
    systems = (
        *metadata.coordinateSystems,
        ct.CoordinateSystem(
            name="other",
            axes=(
                ct.Axis(name="u", unit="um", type="space"),
                ct.Axis(name="v", unit="um", type="space"),
            ),
        ),
    )
    combined = Multiscale(
        coordinateSystems=systems,
        datasets=metadata.datasets,
        coordinateTransformations=(*(metadata.coordinateTransformations or ()), exported),
    )
    assert Multiscale.model_validate(combined.model_dump()) == combined
    declared = {system.name for system in systems}
    endpoints = (
        *combined.datasets[0].coordinateTransformations,
        *(combined.coordinateTransformations or ()),
    )
    assert {
        endpoint.output.name for endpoint in endpoints if endpoint.output is not None
    } <= declared
    assert {
        endpoint.input.name
        for endpoint in endpoints
        if endpoint.input is not None and endpoint.input.name is not None
    } <= declared
    assert exported.input == ct.CoordinateSystemIdentifier(name="physical")
    assert exported.output == ct.CoordinateSystemIdentifier(name="other")


def test_local_frame_transform_needs_explicit_names() -> None:
    geometry = _geometry()
    frame_transform = AffineTransform(
        source=geometry.frame, target=geometry.frame, matrix=np.eye(2), translation=np.zeros(2)
    )
    with pytest.raises(ValueError, match="needs a name in names for NGFF export"):
        to_transform(frame_transform)


def test_level_reports_every_unrepresented_declaration() -> None:
    oriented = CoordinateSystem(
        ("x", "y", "z"),
        ("mm",) * 3,
        axis_types=("space",) * 3,
        vocabulary=DirectionVocabulary("other-anatomy", RAS),
        orientation=RAS,
    )
    frame = ReferenceFrame.local(
        oriented,
        role="world",
        definition={"maker": "test"},
        context={"epoch": 1},
        display={"label": "scan"},
    )
    dims = ("i", "j", "k")
    array = xr.DataArray(
        np.zeros((2, 2, 2)),
        dims=dims,
        coords={dim: (dim, np.arange(2), {"units": "1"}) for dim in dims},
    )
    geometry = Geometry(
        array,
        AffineTransform(
            source=ArrayCoordinates(dims, ("1",) * 3, sample_offset=(0.0,) * 3),
            target=frame,
            matrix=np.eye(3),
            translation=np.zeros(3),
        ),
        dims=dims,
    )
    _, report = to_multiscale_level(geometry, path="s0")
    codes = [code for code, _ in report]
    assert codes == [
        "frame-identity",
        "definition",
        "context",
        "role",
        "display",
        "vocabulary",
        "orientation",
        "sample-offset",
        "sample-offset",
        "sample-offset",
    ]


@pytest.mark.parametrize("diagonal", [True, False])
def test_level_round_trip_preserves_rfc4_orientation(diagonal: bool) -> None:
    base = _geometry(diagonal=diagonal)
    assert isinstance(base.transform, AffineTransform)
    system = CoordinateSystem(
        base.frame.axes,
        base.frame.units,
        axis_types=("space", "space"),
        vocabulary=VOCABULARY,
        orientation=RAS[:2],
    )
    geometry = Geometry(
        base.array,
        AffineTransform(
            source=base.transform.source,
            target=base.frame.with_coordinate_system(system),
            matrix=base.transform.matrix,
            translation=base.transform.translation,
        ),
        dims=base.dims,
    )
    metadata, report = to_multiscale_level(geometry, path="s0")
    assert not {"orientation", "vocabulary"} & {code for code, _ in report}
    dumped = metadata.model_dump()
    frame_system = dumped["coordinateSystems"][0 if diagonal else 1]
    assert tuple(axis["orientation"] for axis in frame_system["axes"]) == RAS[:2]
    if not diagonal:
        assert all("orientation" not in axis for axis in dumped["coordinateSystems"][0]["axes"])
    restored = from_multiscale(dumped, shapes={"s0": base.array.shape})
    frame = restored.frame if diagonal else restored.transforms[0].target
    assert isinstance(frame, ReferenceFrame)
    assert frame.coordinate_system == system
    assert restored.report == ()


def test_to_transform_exports_rectangular_affine_and_losses() -> None:
    source = ReferenceFrame.declared(
        ("ome-zarr", "file:///store#source"), CoordinateSystem(("i", "j"), (None, None))
    )
    target = ReferenceFrame.declared(
        ("ome-zarr", "file:///store#target"), CoordinateSystem(("x", "y", "z"), (None, None, None))
    )
    transform = AffineTransform(
        source=source, target=target, matrix=[[1, 0], [0, 2], [3, 4]], translation=[5, 6, 7]
    )
    value, report = to_transform(transform)
    assert isinstance(value, ct.Affine)
    assert ct.Affine.model_validate(value.model_dump()) == value
    assert value.input is not None and value.input.name == "source"
    assert value.output is not None and value.output.name == "target"
    assert value.affine is not None
    np.testing.assert_allclose(
        value.affine, [[1, 0, 5], [0, 2, 6], [3, 4, 7]], rtol=1e-12, atol=1e-12
    )
    assert [code for code, _ in report] == ["frame-identity", "frame-identity"]


def test_export_refusals_have_specific_type_and_message() -> None:
    geometry = _geometry()
    with pytest.raises(TypeError, match="geometry must be a Geometry, got str"):
        to_multiscale_level("bad", path="s0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"path '../s0' must be relative"):
        to_multiscale_level(geometry, path="../s0")
    with pytest.raises(TypeError, match="name must be a string, got int"):
        to_multiscale_level(geometry, path="s0", name=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name must be nonempty"):
        to_multiscale_level(geometry, path="s0", name="")
    with pytest.raises(TypeError, match="frame_name must be a string"):
        to_multiscale_level(geometry, path="s0", frame_name=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="frame_name must be nonempty"):
        to_multiscale_level(geometry, path="s0", frame_name="")
    with pytest.raises(ValueError, match="store must be a resolved URI without a trailing"):
        to_multiscale_level(geometry, path="s0", store="file:///store/")
    with pytest.raises(TypeError, match="t must be an AffineTransform, got str"):
        to_transform("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="to_transform requires ReferenceFrame source and target"):
        to_transform(geometry.transform)  # type: ignore[arg-type]


def test_non_lattice_geometry_refuses() -> None:
    geometry = _geometry()
    geometry.array.coords["i"] = ("i", [0, 1, 4, 9], {"units": "1"})
    with pytest.raises(ValueError, match="NGFF export requires a regular affine lattice"):
        to_multiscale_level(geometry, path="s0")


def test_non_affine_geometry_refuses() -> None:
    base = _geometry()
    chained = Geometry(
        base.array,
        CompositeTransform(base.transform),
        dims=base.dims,
    )
    with pytest.raises(ValueError, match="NGFF export requires a regular affine lattice"):
        to_multiscale_level(chained, path="s0")


def test_non_geometry_dimension_refuses() -> None:
    base = _geometry()
    array = base.array.expand_dims(channel=[0, 1])
    geometry = Geometry(array, base.transform, dims=base.dims)
    with pytest.raises(
        ValueError, match="NGFF level requires every array dimension to be a geometry dimension"
    ):
        to_multiscale_level(geometry, path="s0")


def test_rectangular_level_refuses() -> None:
    base = _geometry()
    frame = ReferenceFrame.local(
        CoordinateSystem(("z", "y", "x"), ("um", "um", "um"), axis_types=("space",) * 3)
    )
    geometry = Geometry(
        base.array,
        AffineTransform(
            source=base.transform.source,
            target=frame,
            matrix=[[1, 0], [0, 2], [3, 4]],
            translation=[0, 0, 0],
        ),
        dims=base.dims,
    )
    with pytest.raises(ValueError, match="NGFF level needs one array dimension per intrinsic axis"):
        to_multiscale_level(geometry, path="s0")


def test_array_unit_loss_includes_undeclared_unit() -> None:
    base = _geometry()
    array = base.array.copy()
    array.coords["j"].attrs.pop("units")
    geometry = Geometry(
        array,
        AffineTransform(
            source=ArrayCoordinates(("j", "i"), (None, "1"), sample_offset=(0.5, 0.5)),
            target=base.frame,
            matrix=np.diag([2, 3]),
            translation=[4, 5],
        ),
        dims=("j", "i"),
    )
    _, report = to_multiscale_level(geometry, path="s0")
    assert ("array-unit", "axis 'j': array unit None is not represented") in report


def test_array_axis_type_loss_is_reported() -> None:
    base = _geometry()
    geometry = Geometry(
        base.array,
        AffineTransform(
            source=ArrayCoordinates(
                ("j", "i"), ("1", "1"), axis_types=("space", None), sample_offset=(0.5, 0.5)
            ),
            target=base.frame,
            matrix=np.diag([2, 3]),
            translation=[4, 5],
        ),
        dims=("j", "i"),
    )
    _, report = to_multiscale_level(geometry, path="s0")
    assert ("array-axis-type", "axis 'j': array axis type 'space' is not represented") in report


def test_more_invalid_location_arguments_refuse() -> None:
    geometry = _geometry()
    with pytest.raises(TypeError, match="store must be a resolved URI string or None"):
        to_multiscale_level(geometry, path="s0", store=3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="path must be a relative string, got int"):
        to_multiscale_level(geometry, path=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="store must be a resolved URI without a trailing"):
        to_multiscale_level(geometry, path="s0", store="")
    with pytest.raises(ValueError, match=r"path 's0/' must be relative"):
        to_multiscale_level(geometry, path="s0/")


def test_orientation_on_a_non_space_axis_is_reported_not_written() -> None:
    from xarrayrf import anatomy

    system = CoordinateSystem(
        ("t", "x", "y"),
        ("s", "um", "um"),
        axis_types=("time", "space", "space"),
        vocabulary=anatomy.VOCABULARY,
        orientation=("inferior-to-superior", "left-to-right", "posterior-to-anterior"),
    )
    frame = ReferenceFrame.local(system)
    array = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("t", "j", "i"),
        coords={
            dim: (dim, np.arange(n), {"units": "1"}) for dim, n in (("t", 2), ("j", 3), ("i", 4))
        },
    )
    geometry = Geometry(
        array,
        AffineTransform(
            source=ArrayCoordinates(("t", "i", "j"), ("1", "1", "1")),
            target=frame,
            matrix=np.eye(3),
            translation=np.zeros(3),
        ),
        dims=("t", "j", "i"),
    )
    metadata, report = to_multiscale_level(geometry, path="s0")
    written = {
        axis.name: (axis.model_extra or {}).get("orientation")
        for axis in metadata.coordinateSystems[-1].axes
    }
    assert written == {"t": None, "x": "left-to-right", "y": "posterior-to-anterior"}
    assert (
        "orientation",
        "frame 'physical': orientation of non-space axis 't' is not represented",
    ) in report
