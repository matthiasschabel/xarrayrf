"""OME-Zarr 0.6 metadata import, including the vendored specification examples."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest
from dask.callbacks import Callback
from ome_zarr_models.v06 import coordinate_transforms as ct
from ome_zarr_models.v06.multiscales import Multiscale
from ome_zarr_models.v06.scene import SceneAttrs

from xarrayrf import ArrayCoordinates, ReferenceFrame, anatomy
from xarrayrf.ngff import (
    NAMESPACE,
    coordinate_system,
    from_multiscale,
    from_scene,
    to_dataarray,
    to_multiscale_level,
    transform,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ngff_0_6"
BINDING_MULTISCALE = {
    "coordinateSystems": [
        {
            "name": "intrinsic",
            "axes": [
                {"name": "y", "type": "space", "unit": "um"},
                {"name": "x", "type": "space", "unit": "um"},
            ],
        }
    ],
    "datasets": [
        {
            "path": "s0",
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [2, 3],
                    "input": {"path": "s0"},
                    "output": {"name": "intrinsic"},
                }
            ],
        }
    ],
}


def test_to_dataarray_binds_and_exports_level() -> None:
    level = from_multiscale(BINDING_MULTISCALE, shapes={"s0": (3, 4)}).levels["s0"]
    pixels = np.arange(12).reshape(3, 4)
    array = to_dataarray(level, pixels)
    assert array.data is pixels
    assert array.rf.coordinate_transform == level.transform
    for point in ((0, 0), (1, 2), (2, 3)):
        position = dict(zip(level.dims, point, strict=True))
        np.testing.assert_allclose(
            np.asarray(array.rf.geometry.point_at(**position)),
            level.transform.transform_point(point),
            rtol=1e-12,
            atol=1e-12,
        )
    exported, _ = to_multiscale_level(array.rf.geometry, path="s0")
    restored = from_multiscale(exported, shapes={"s0": pixels.shape})
    np.testing.assert_allclose(
        restored.levels["s0"].transform.transform_point([1, 2]),
        level.transform.transform_point([1, 2]),
        rtol=1e-12,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match=r"data shape \(3, 3\).*geometry shape \(3, 4\)"):
        to_dataarray(level, np.zeros((3, 3)))


def test_to_dataarray_keeps_ngff_dask_pixels_lazy() -> None:
    level = from_multiscale(BINDING_MULTISCALE, shapes={"s0": (3, 4)}).levels["s0"]
    pixels = da.ones((3, 4), chunks=(2, 2))
    tasks: list[object] = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        array = to_dataarray(level, pixels)
    assert not tasks
    assert array.data is pixels


@pytest.mark.parametrize("pixels", [np.zeros((3, 4)).tolist(), 1])
def test_to_dataarray_refuses_non_duck_pixels(pixels: object) -> None:
    level = from_multiscale(BINDING_MULTISCALE, shapes={"s0": (3, 4)}).levels["s0"]
    with pytest.raises(TypeError, match=r"duck array.*np.asarray"):
        to_dataarray(level, pixels)  # type: ignore[arg-type]


def read_json(path: Path) -> dict[str, Any]:
    text = path.read_text()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return cast(dict[str, Any], json.loads(text))


# The points are taken from the examples' numbers; the expected results are calculated
# independently of ome-zarr-models' affine conversion, which lacks rectangular support.
POINTS: dict[str, list[tuple[tuple[float, ...], tuple[float, ...]]]] = {
    "affine2d2d": [((1, 2), (8, 20))],
    "affine2d2d_with_channel": [((1, 2), (8, 20))],
    "affine2d3d": [((1, 2), (1, 12, 24))],
    "byDimension1": [((1, 2), (2, 1))],
    "byDimension2": [((1, 2, 3, 4), (4, 4.5, 4.5))],
    "identity": [((1, 2), (1, 2))],
    "mapAxis1": [((1, 2), (1, 2)), ((1, 2), (2, 1))],
    "projectAxis": [((1, 2), (0, 0, 1, 2))],
    "projectAxis2": [((1, 2, 3), (0, 2, 3))],
    "rotation": [((1, 2), (-2, 1))],
    "scale": [((1, 2), (2, 6.24))],
    "scale_with_discrete": [((1, 2), (3.12, 4))],
    "sequence": [((1, 2), (2.2, 8.7))],
    "translation": [((1, 2), (10, 0.58))],
}
REFUSALS = {
    "bijection": "coordinates is nonlinear",
    "byDimensionCoordinates": "coordinates is nonlinear",
    "byDimensionInvalid1": "byDimension inputAxes/outputAxes indices are out of range",
    "byDimensionInvalid2": "byDimension must cover every output axis exactly once",
    "byDimensionXarray": "coordinates is nonlinear",
    "xarrayLike": "coordinates is nonlinear",
}


@pytest.mark.parametrize(
    "path", sorted((FIXTURES / "transformations").glob("*.json")), ids=lambda p: p.stem
)
def test_transformation_examples(path: Path) -> None:
    data = read_json(path)
    if path.stem == "bijection_verbose":
        systems = [{"name": name, "axes": [{"name": "x"}]} for name in ("src", "tgt")]
        with pytest.raises(ValueError, match="coordinates is nonlinear"):
            transform(data, systems)
        return
    assert "coordinateTransformations" in data
    declarations = data["coordinateSystems"]
    assert isinstance(declarations, list)
    transformations = data["coordinateTransformations"]
    assert isinstance(transformations, list)
    if path.stem in REFUSALS:
        for item in transformations:
            with pytest.raises(ValueError, match=re.escape(REFUSALS[path.stem])):
                transform(item, declarations)
        return
    for item, (point, expected) in zip(transformations, POINTS[path.stem], strict=True):
        result, report = transform(item, declarations)
        if path.stem in ("affine2d2d_with_channel", "scale_with_discrete"):
            assert any(code == "discrete-axis-dropped" for code, _ in report)
        np.testing.assert_allclose(
            result.matrix @ np.asarray(point) + result.translation,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )


def test_coordinate_system_identity_and_local_frames() -> None:
    declaration = {"name": "physical / x", "axes": [{"name": "x", "unit": "mm", "type": "space"}]}
    first, first_report = coordinate_system(declaration, store="file:///data", group="image")
    second, second_report = coordinate_system(
        ct.CoordinateSystem.model_validate(declaration), store="file:///data", group="image"
    )
    assert first == second
    assert first_report == second_report == ()
    assert first.identifier == (NAMESPACE, "file:///data/image#physical%20%2F%20x")
    assert first.coordinate_system.axis_types == ("space",)
    assert coordinate_system(declaration)[0] != coordinate_system(declaration)[0]


def test_coordinate_system_reports_unrepresented_axis_metadata() -> None:
    _, report = coordinate_system(
        {
            "name": "intrinsic",
            "axes": [
                {"name": "c", "type": "channel", "discrete": True, "longName": "channel"},
                {"name": "x", "type": "space"},
            ],
        }
    )
    assert report == (
        ("discrete-deferred", "axis 'c': discrete is not retained"),
        ("long-name-deferred", "axis 'c': longName is not retained"),
    )


@pytest.mark.parametrize("token", ["west-to-east", "unknown"])
def test_coordinate_system_refuses_unknown_orientation(token: str) -> None:
    with pytest.raises(ValueError, match=rf"{token!r}.*ome-ngff:rfc-4:anatomical"):
        coordinate_system(
            {
                "name": "intrinsic",
                "axes": [{"name": "x", "type": "space", "unit": "um", "orientation": token}],
            }
        )


@pytest.mark.parametrize("kind", ["time", "channel", None])
def test_coordinate_system_refuses_non_space_orientation(kind: str | None) -> None:
    with pytest.raises(ValueError, match="axis 'a': orientation requires type 'space'"):
        coordinate_system(
            {
                "name": "intrinsic",
                "axes": [{"name": "a", "type": kind, "unit": "um", "orientation": anatomy.RAS[0]}],
            }
        )


def test_multiscale_reuses_oriented_frame_after_channel_drop() -> None:
    metadata = {
        "coordinateSystems": [
            {
                "name": "intrinsic",
                "axes": [
                    {"name": "t", "type": "time", "unit": "second"},
                    {"name": "c", "type": "channel"},
                    {"name": "x", "type": "space", "unit": "um", "orientation": anatomy.RAS[0]},
                    {"name": "y", "type": "space", "unit": "um"},
                ],
            }
        ],
        "datasets": [
            {
                "path": "0",
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [2, 1, 3, 4],
                        "input": {"path": "0"},
                        "output": {"name": "intrinsic"},
                    }
                ],
            }
        ],
    }
    first = from_multiscale(metadata, shapes={"0": (2, 3, 4, 5)})
    assert first.frame.coordinate_system.vocabulary == anatomy.VOCABULARY
    assert first.frame.coordinate_system.orientation == (None, anatomy.RAS[0], None)
    assert {code for code, _ in first.report} == {"discrete-axis-dropped"}
    reused = from_multiscale(
        metadata, shapes={"0": (2, 3, 4, 5)}, frames={("", "intrinsic"): first.frame}
    )
    assert reused.frame == first.frame
    assert reused.levels["0"].transform.target == first.frame


def test_path_only_endpoint_uses_supplied_dims_and_centred_cells() -> None:
    systems = [{"name": "intrinsic", "axes": [{"name": "y"}, {"name": "x"}]}]
    result, report = transform(
        {
            "type": "scale",
            "scale": [2, 3],
            "input": {"path": "s0"},
            "output": {"name": "intrinsic"},
        },
        systems,
        dims={"s0": ("j", "i")},
    )
    assert isinstance(result.source, ArrayCoordinates)
    assert result.source.axes == ("j", "i")
    assert result.source.units == ("1", "1")
    assert result.source.sample_offset == (0.5, 0.5)
    assert report == ()
    with pytest.raises(TypeError, match="array path 's0' dims must be a sequence"):
        transform(
            {
                "type": "scale",
                "scale": [2, 3],
                "input": {"path": "s0"},
                "output": {"name": "intrinsic"},
            },
            systems,
            dims={"s0": "ji"},
        )


def test_all_entry_points_accept_v06_models() -> None:
    declarations = [
        {
            "name": "intrinsic",
            "axes": [
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
        }
    ]
    transform_json = {
        "type": "scale",
        "scale": [2, 3],
        "input": {"path": "s0"},
        "output": {"name": "intrinsic"},
    }
    model = Multiscale.model_validate(
        {
            "coordinateSystems": declarations,
            "datasets": [{"path": "s0", "coordinateTransformations": [transform_json]}],
        }
    )
    result = from_multiscale(model, shapes={"s0": (4, 5)})
    direct, direct_report = transform(
        ct.Scale.model_validate(transform_json),
        [ct.CoordinateSystem.model_validate(declarations[0])],
        dims={"s0": ("y", "x")},
    )
    np.testing.assert_allclose(
        result.levels["s0"].transform.matrix, direct.matrix, rtol=1e-12, atol=1e-12
    )
    assert direct_report == ()
    scene = SceneAttrs.model_validate(
        {
            "coordinateSystems": declarations,
            "coordinateTransformations": [
                {
                    "type": "identity",
                    "input": {"name": "intrinsic"},
                    "output": {"name": "intrinsic"},
                }
            ],
        }
    )
    imported_scene = from_scene(scene, systems={})
    assert len(imported_scene.transforms) == 1
    assert imported_scene.report == ()


def test_bijection_checks_inverse() -> None:
    systems = [{"name": name, "axes": [{"name": "x"}]} for name in ("in", "out")]
    item = {
        "type": "bijection",
        "input": {"name": "in"},
        "output": {"name": "out"},
        "forward": {"type": "translation", "translation": [2]},
        "inverse": {"type": "translation", "translation": [-2]},
    }
    imported, report = transform(item, systems)
    assert imported.translation[0] == pytest.approx(2, abs=1e-12)
    assert report == (
        ("declared-inverse-deferred", "bijection inverse was checked but not retained"),
    )
    item["inverse"] = {"type": "translation", "translation": [-1]}
    with pytest.raises(ValueError, match="bijection inverse disagrees"):
        transform(item, systems)


def test_sequence_changes_dimension_before_later_member() -> None:
    systems = [
        {"name": "in", "axes": [{"name": "i"}, {"name": "j"}]},
        {"name": "out", "axes": [{"name": "x"}, {"name": "y"}, {"name": "z"}]},
    ]
    item = {
        "type": "sequence",
        "input": {"name": "in"},
        "output": {"name": "out"},
        "transformations": [
            {"type": "projectAxis", "createdOutputs": [0]},
            {"type": "translation", "translation": [1, 2, 3]},
            {"type": "scale", "scale": [2, 3, 4]},
        ],
    }
    result, report = transform(item, systems)
    assert report == ()
    np.testing.assert_allclose(
        result.matrix @ (5, 6) + result.translation, (2, 21, 36), rtol=1e-12, atol=1e-12
    )
    item["transformations"] = []
    with pytest.raises(ValueError, match="sequence output axis count disagrees"):
        transform(item, systems)


@pytest.mark.parametrize(
    "name", ["multiscales_example", "multiscales_example_relative", "multiscales_transformations"]
)
def test_multiscale_examples(name: str) -> None:
    data = read_json(FIXTURES / "multiscales_strict" / f"{name}.json")
    metadata = data.get("attributes", {}).get("ome", data)
    multiscale = metadata["multiscales"][0]
    count = len(multiscale["coordinateSystems"][0]["axes"])
    shapes = {item["path"]: (8,) * count for item in multiscale["datasets"]}
    result = from_multiscale(multiscale, shapes=shapes, store="file:///data", group="image")
    assert all(level.transform.target == result.frame for level in result.levels.values())
    assert all(
        np.asarray(level.coords[level.dims[0]][1]).dtype == np.int64
        for level in result.levels.values()
    )
    assert result.frame.identifier == (
        NAMESPACE,
        f"file:///data/image#{multiscale['datasets'][0]['coordinateTransformations'][0]['output']['name']}",
    )
    if name == "multiscales_example_relative":
        assert result.frame.axes == ("t", "z", "y", "x")
        assert any(code == "discrete-axis-dropped" for code, _ in result.report)
        np.testing.assert_allclose(
            result.levels["s1"].transform.translation, (0, 0.5, 0.5, 0.5), rtol=1e-12, atol=1e-12
        )


def test_multiscale_levels_share_one_local_intrinsic_frame() -> None:
    data = read_json(FIXTURES / "multiscales_strict" / "multiscales_example.json")
    ms = data["attributes"]["ome"]["multiscales"][0]
    count = len(ms["coordinateSystems"][0]["axes"])
    shapes = {path: (4,) * count for path in ("s0", "s1", "s2")}
    imported = from_multiscale(ms, shapes=shapes)
    assert all(level.transform.target == imported.frame for level in imported.levels.values())
    assert imported.frame != from_multiscale(ms, shapes=shapes).frame


def test_multiscale_reference_to_undeclared_image_refuses() -> None:
    data = read_json(FIXTURES / "multiscales_strict" / "multiscales_reference_to_label.json")
    multiscale = data["ome"]["multiscales"][0]
    with pytest.raises(
        ValueError,
        match="unresolved coordinate system 'physical' at path 'labels/cell_segmentation'",
    ):
        from_multiscale(multiscale, shapes={"s0": (4, 4, 4)})


@pytest.mark.parametrize(
    "name,reason",
    [
        ("multiscales", "displacements is nonlinear"),
        ("displacement_field", "The length of axes does not match the dimensionality of the scale"),
    ],
)
def test_displacement_examples_refuse(name: str, reason: str) -> None:
    data = read_json(FIXTURES / "transformations" / "displacements" / f"{name}.json")
    multiscale = data["ome"]["multiscales"][0]
    shape = (4,) * len(multiscale["coordinateSystems"][0]["axes"])
    with pytest.raises(ValueError, match=reason):
        from_multiscale(multiscale, shapes={"s0": shape})


def test_scene_examples_and_image_frame_identity() -> None:
    scene = read_json(FIXTURES / "scene" / "scene_stitching.json")["attributes"]["ome"]["scene"]
    physical = {
        "name": "physical",
        "axes": [
            {"name": "y", "type": "space", "unit": "mm"},
            {"name": "x", "type": "space", "unit": "mm"},
        ],
    }
    systems = {f"tile_{i}": [physical] for i in range(4)}
    imported = from_scene(scene, systems=systems, store="file:///data", group="tiles")
    assert len(imported.transforms) == 4
    assert imported.report == (
        ("discrete-deferred", "axis 'x': discrete is not retained"),
        ("discrete-deferred", "axis 'y': discrete is not retained"),
    )
    multiscale = {
        "coordinateSystems": [physical],
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [
                    {
                        "type": "identity",
                        "input": {"path": "s0"},
                        "output": {"name": "physical"},
                    }
                ],
            }
        ],
    }
    image = from_multiscale(
        multiscale, shapes={"s0": (3, 4)}, store="file:///data", group="tiles/tile_0"
    )
    assert imported.transforms[0].source == image.frame
    np.testing.assert_allclose(
        imported.transforms[3].translation, (276, 348), rtol=1e-12, atol=1e-12
    )
    registration = read_json(FIXTURES / "scene" / "scene_registration.json")["attributes"]["ome"][
        "scene"
    ]
    with pytest.raises(ValueError, match="displacements is nonlinear"):
        from_scene(registration, systems={"JRC2018F": [physical], "FCWB": [physical]})


def test_scene_reuses_supplied_local_frames() -> None:
    physical = {"name": "physical", "axes": [{"name": "x"}]}
    known, report = coordinate_system(physical)
    assert report == ()
    scene = {
        "coordinateTransformations": [
            {
                "type": "identity",
                "input": {"path": "tile", "name": "physical"},
                "output": {"name": "world"},
            }
        ],
        "coordinateSystems": [{"name": "world", "axes": [{"name": "x"}]}],
    }
    result = from_scene(scene, systems={"tile": [physical]}, frames={("tile", "physical"): known})
    assert isinstance(result.transforms[0].source, ReferenceFrame)
    assert result.transforms[0].source == known
    assert result.report == ()


def test_scene_reuses_geometry_frame_after_channel_drop() -> None:
    physical = {
        "name": "physical",
        "axes": [
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
    }
    ms = {
        "coordinateSystems": [physical],
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1, 1, 1],
                        "input": {"path": "s0"},
                        "output": {"name": "physical"},
                    }
                ],
            }
        ],
    }
    image = from_multiscale(ms, shapes={"s0": (2, 4, 4)}, group="tile")
    scene = {
        "coordinateSystems": [{"name": "world", "axes": physical["axes"]}],
        "coordinateTransformations": [
            {
                "type": "identity",
                "input": {"path": "tile", "name": "physical"},
                "output": {"name": "world"},
            }
        ],
    }
    result = from_scene(
        scene, systems={"tile": [physical]}, frames={("tile", "physical"): image.frame}
    )
    assert result.transforms[0].source == image.frame
    assert result.transforms[0].source.axes == ("y", "x")
    assert ("discrete-axis-dropped", "identity-mapped axis 'c'") in result.report


def test_scene_refuses_undeclared_image_system() -> None:
    scene = {
        "coordinateSystems": [{"name": "world", "axes": [{"name": "x"}]}],
        "coordinateTransformations": [
            {
                "type": "identity",
                "input": {"path": "missing", "name": "physical"},
                "output": {"name": "world"},
            }
        ],
    }
    with pytest.raises(
        ValueError, match="unresolved coordinate system 'physical' at path 'missing'"
    ):
        from_scene(scene, systems={})


@pytest.mark.parametrize(
    "value,pattern",
    [
        ({"name": "bad", "axes": [{"name": None}]}, "axis without a name"),
        ({"name": "bad", "axes": [{"name": "x", "unit": 1}]}, "axis 'x' has a non-string unit"),
    ],
)
def test_unrepresentable_axes(value: dict[str, object], pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        coordinate_system(value)


def test_public_model_argument_types() -> None:
    with pytest.raises(TypeError, match="coordinate system must be a v06 CoordinateSystem"):
        coordinate_system(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="transform must be a v06 transform"):
        transform(1, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="systems must be a sequence"):
        transform({"type": "identity"}, "bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dims must map array paths"):
        transform({"type": "identity"}, [], dims=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="frames must map"):
        transform({"type": "identity"}, [], frames=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="systems must map image paths"):
        from_scene({}, systems=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="frames must map"):
        from_scene({}, systems={}, frames=[])  # type: ignore[arg-type]


def test_unvalidated_transform_models_still_fail_clearly() -> None:
    # model_construct skips v06 validation; public import still rejects unsupported values.
    systems = [{"name": name, "axes": [{"name": "i"}, {"name": "j"}]} for name in ("a", "b")]
    source = ct.CoordinateSystemIdentifier(name="a")
    target = ct.CoordinateSystemIdentifier(name="b")
    bad_endpoint = ct.CoordinateSystemIdentifier.model_construct(name=None, path=None)
    with pytest.raises(ValueError, match="transform endpoint requires a name or array path"):
        transform(ct.Identity.model_construct(input=bad_endpoint, output=target), systems)
    with pytest.raises(ValueError, match="rotation array-backed path is unsupported"):
        transform(
            ct.Rotation.model_construct(path="rotation", input=source, output=target), systems
        )
    with pytest.raises(ValueError, match="rotation must have shape"):
        transform(
            ct.Rotation.model_construct(rotation=((1, 0),), input=source, output=target), systems
        )
    with pytest.raises(ValueError, match="mapAxis indices must name the input axes"):
        transform(ct.MapAxis.model_construct(mapAxis=(0, 2), input=source, output=target), systems)


def test_by_dimension_must_cover_every_output() -> None:
    systems = [{"name": name, "axes": [{"name": "i"}, {"name": "j"}]} for name in ("a", "b")]
    item = {
        "type": "byDimension",
        "input": {"name": "a"},
        "output": {"name": "b"},
        "transformations": [
            {"transformation": {"type": "identity"}, "inputAxes": [0], "outputAxes": [0]}
        ],
    }
    with pytest.raises(ValueError, match="byDimension must cover every output axis exactly once"):
        transform(item, systems)


def test_unknown_transform_subclass_refuses() -> None:
    class Unsupported(ct.Transform):
        type: str = "unsupported"

        @property
        def has_inverse(self) -> bool:
            return False

        def get_inverse(self) -> ct.Transform:
            raise NotImplementedError

        def transform_point(self, point: Sequence[float]) -> tuple[float, ...]:
            return tuple(point)

        def as_affine(self) -> ct.Affine:
            raise NotImplementedError

    system = {"name": "a", "axes": [{"name": "x"}]}
    with pytest.raises(ValueError, match="unsupported NGFF transform 'unsupported'"):
        transform(
            Unsupported(
                input=ct.CoordinateSystemIdentifier(name="a"),
                output=ct.CoordinateSystemIdentifier(name="a"),
            ),
            [system],
        )


def test_mixed_discrete_axis_refuses() -> None:
    systems = [
        {"name": name, "axes": [{"name": "c", "type": "channel"}, {"name": "x", "type": "space"}]}
        for name in ("a", "b")
    ]
    item = {
        "type": "affine",
        "affine": [[1, 1, 0], [0, 1, 0]],
        "input": {"name": "a"},
        "output": {"name": "b"},
    }
    with pytest.raises(ValueError, match="discrete axis 'c' mixes with other axes"):
        transform(item, systems)
    systems[1] = {
        "name": "b",
        "axes": [{"name": "c", "type": "space"}, {"name": "x", "type": "space"}],
    }
    with pytest.raises(ValueError, match="discrete axes must be identity-mapped on both endpoints"):
        transform({"type": "identity", "input": {"name": "a"}, "output": {"name": "b"}}, systems)


def test_only_discrete_axes_have_no_geometry() -> None:
    systems = [{"name": name, "axes": [{"name": "c", "type": "channel"}]} for name in ("a", "b")]
    with pytest.raises(ValueError, match="transform has no continuous geometry axes"):
        transform({"type": "identity", "input": {"name": "a"}, "output": {"name": "b"}}, systems)


def test_array_backed_parameters_refuse() -> None:
    systems = [{"name": name, "axes": [{"name": "x"}]} for name in ("a", "b")]
    for kind in ("affine", "rotation"):
        item = {"type": kind, "path": "matrix", "input": {"name": "a"}, "output": {"name": "b"}}
        with pytest.raises(ValueError, match=f"{kind} array-backed path is unsupported"):
            transform(item, systems)


@pytest.mark.parametrize(
    "kwargs,error,pattern",
    [
        ({"store": 3}, TypeError, "store must be a resolved URI string"),
        ({"store": "file:///data/"}, ValueError, "store must be a resolved URI"),
        ({"group": 3}, TypeError, "group must be a path string"),
        ({"group": "a/../b"}, ValueError, "group must have no leading/trailing"),
    ],
)
def test_location_validation(
    kwargs: dict[str, object], error: type[Exception], pattern: str
) -> None:
    with pytest.raises(error, match=pattern):
        coordinate_system({"name": "physical", "axes": [{"name": "x"}]}, **kwargs)  # type: ignore[arg-type]


def test_frame_mapping_must_agree_with_declared_axes() -> None:
    known, _ = coordinate_system({"name": "a", "axes": [{"name": "y"}]})
    systems = [{"name": name, "axes": [{"name": "x"}]} for name in ("a", "b")]
    with pytest.raises(ValueError, match=r"frame for .* disagrees with coordinate system axes"):
        transform(
            {"type": "identity", "input": {"name": "a"}, "output": {"name": "b"}},
            systems,
            frames={("", "a"): known},
        )
    with pytest.raises(ValueError, match="identity inconsistent with store"):
        transform(
            {"type": "identity", "input": {"name": "a"}, "output": {"name": "b"}},
            systems,
            store="file:///data",
            frames={("", "a"): coordinate_system(systems[0])[0]},
        )
    with pytest.raises(TypeError, match="must be a ReferenceFrame"):
        transform(
            {"type": "identity", "input": {"name": "a"}, "output": {"name": "b"}},
            systems,
            frames={("", "a"): "bad"},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize(
    "kind,parameters,expected",
    [
        ("identity", {}, "identity requires equal input and output axis counts"),
        ("scale", {"scale": [2]}, "scale needs one factor per input and output axis"),
        (
            "translation",
            {"translation": [1]},
            "translation needs one value per input and output axis",
        ),
        ("affine", {"affine": [[1, 2], [3, 4]]}, "affine must have shape"),
        ("projectAxis", {"createdOutputs": [3]}, "projectAxis indices must be within"),
        (
            "projectAxis",
            {"createdOutputs": [0, 1]},
            "projectAxis retained input and output axis counts must agree",
        ),
    ],
)
def test_transform_shape_errors(kind: str, parameters: dict[str, object], expected: str) -> None:
    systems = [
        {"name": "in", "axes": [{"name": "i"}, {"name": "j"}]},
        {"name": "out", "axes": [{"name": "x"}, {"name": "y"}, {"name": "z"}]},
    ]
    item = {"type": kind, "input": {"name": "in"}, "output": {"name": "out"}, **parameters}
    with pytest.raises(ValueError, match=expected):
        transform(item, systems)


def test_missing_endpoints_and_array_dims_refuse() -> None:
    with pytest.raises(ValueError, match="transform requires input and output endpoints"):
        transform({"type": "identity"}, [])
    with pytest.raises(ValueError, match="array path 's0' needs dims"):
        transform(
            {"type": "identity", "input": {"path": "s0"}, "output": {"name": "out"}},
            [{"name": "out", "axes": [{"name": "x"}]}],
        )


def test_duplicate_systems_and_bad_relative_path_refuse() -> None:
    system = {"name": "out", "axes": [{"name": "x"}]}
    item = {"type": "identity", "input": {"name": "out"}, "output": {"name": "out"}}
    with pytest.raises(ValueError, match="systems must have unique names"):
        transform(item, [system, system])
    item["input"] = {"path": "../s0"}
    with pytest.raises(ValueError, match=re.escape("path '../s0' must be relative")):
        transform(item, [system], dims={"../s0": ("x",)})


def test_multiscale_shape_and_output_validation() -> None:
    systems = [
        {
            "name": "intrinsic",
            "axes": [{"name": "x", "type": "space"}, {"name": "y", "type": "space"}],
        }
    ]
    level = {
        "path": "s0",
        "coordinateTransformations": [
            {
                "type": "scale",
                "scale": [1, 1],
                "input": {"path": "s0"},
                "output": {"name": "intrinsic"},
            }
        ],
    }
    ms = {"coordinateSystems": systems, "datasets": [level]}
    with pytest.raises(TypeError, match="shapes must map dataset paths"):
        from_multiscale(ms, shapes=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dims must map dataset paths"):
        from_multiscale(ms, shapes={"s0": (2, 2)}, dims=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="frames must map"):
        from_multiscale(ms, shapes={"s0": (2, 2)}, frames=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dataset 's0' needs a shape"):
        from_multiscale(ms, shapes={})
    with pytest.raises(TypeError, match="dataset 's0' dims must be a sequence"):
        from_multiscale(ms, shapes={"s0": (2, 2)}, dims={"s0": "xy"})
    with pytest.raises(ValueError, match="dataset 's0' dims must uniquely name"):
        from_multiscale(ms, shapes={"s0": (2, 2)}, dims={"s0": ("x", "x")})
    with pytest.raises(ValueError, match="Tuple should have at most 1 item"):
        from_multiscale(
            {
                **ms,
                "datasets": [
                    {
                        **level,
                        "coordinateTransformations": [
                            *level["coordinateTransformations"],
                            *level["coordinateTransformations"],
                        ],
                    }
                ],
            },
            shapes={"s0": (2, 2)},
        )
    external_output = {
        **level,
        "coordinateTransformations": [
            {
                "type": "scale",
                "scale": [1, 1],
                "input": {"path": "s0"},
                "output": {"path": "other", "name": "intrinsic"},
            }
        ],
    }
    with pytest.raises(ValueError, match="dataset 's0' must output to a named intrinsic system"):
        from_multiscale({**ms, "datasets": [external_output]}, shapes={"s0": (2, 2)})


def test_multiscale_rejects_different_intrinsic_systems() -> None:
    axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    datasets = [
        {
            "path": path,
            "coordinateTransformations": [
                {
                    "type": "identity",
                    "input": {"path": path},
                    "output": {"name": "intrinsic"},
                }
            ],
        }
        for path in ("s0", "s1")
    ]
    model = Multiscale.model_validate(
        {
            "coordinateSystems": [{"name": name, "axes": axes} for name in ("intrinsic", "other")],
            "datasets": datasets,
        }
    )
    second = model.datasets[1].model_copy(
        update={
            "coordinateTransformations": (
                ct.Identity.model_validate(
                    {
                        "type": "identity",
                        "input": {"path": "s1"},
                        "output": {"name": "other"},
                    }
                ),
            )
        }
    )
    inconsistent = model.model_copy(update={"datasets": (model.datasets[0], second)})
    with pytest.raises(
        ValueError, match="dataset 's1' must output to intrinsic system 'intrinsic'"
    ):
        from_multiscale(inconsistent, shapes={"s0": (2, 2), "s1": (2, 2)})


def test_multiscale_reports_unrepresented_axis_metadata() -> None:
    ms = {
        "metadata": {"type": "rotation", "path": "descriptive-only"},
        "coordinateSystems": [
            {
                "name": "intrinsic",
                "axes": [
                    {"name": "x", "type": "space", "discrete": False, "longName": "horizontal"},
                    {"name": "y", "type": "space"},
                ],
            }
        ],
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [
                    {"type": "identity", "input": {"path": "s0"}, "output": {"name": "intrinsic"}}
                ],
            }
        ],
    }
    result = from_multiscale(ms, shapes={"s0": (2, 2)})
    assert ("discrete-deferred", "axis 'x': discrete is not retained") in result.report
    assert ("long-name-deferred", "axis 'x': longName is not retained") in result.report


def test_multiscale_reports_checked_declared_inverse() -> None:
    axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    ms = {
        "coordinateSystems": [{"name": name, "axes": axes} for name in ("intrinsic", "physical")],
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [
                    {
                        "type": "identity",
                        "input": {"path": "s0"},
                        "output": {"name": "intrinsic"},
                    }
                ],
            }
        ],
        "coordinateTransformations": [
            {
                "type": "bijection",
                "input": {"name": "intrinsic"},
                "output": {"name": "physical"},
                "forward": {"type": "translation", "translation": [2, 3]},
                "inverse": {"type": "translation", "translation": [-2, -3]},
            }
        ],
    }
    result = from_multiscale(ms, shapes={"s0": (3, 4)})
    assert (
        "declared-inverse-deferred",
        "bijection inverse was checked but not retained",
    ) in result.report


def test_multiscale_rejects_duplicate_dataset_paths() -> None:
    cs = {
        "name": "intrinsic",
        "axes": [
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
    }
    dataset = {
        "path": "s0",
        "coordinateTransformations": [
            {
                "type": "identity",
                "input": {"path": "s0"},
                "output": {"name": "intrinsic"},
            }
        ],
    }
    with pytest.raises(ValueError, match="duplicate dataset path 's0'"):
        from_multiscale(
            {"coordinateSystems": [cs], "datasets": [dataset, dataset]},
            shapes={"s0": (2, 2)},
        )


def test_multiscale_additional_transform_starts_at_intrinsic() -> None:
    axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    ms = {
        "coordinateSystems": [{"name": name, "axes": axes} for name in ("intrinsic", "other")],
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [
                    {
                        "type": "identity",
                        "input": {"path": "s0"},
                        "output": {"name": "intrinsic"},
                    }
                ],
            }
        ],
        "coordinateTransformations": [
            {"type": "identity", "input": {"name": "other"}, "output": {"name": "intrinsic"}}
        ],
    }
    with pytest.raises(
        ValueError, match="multiscale additional transforms must start at the intrinsic frame"
    ):
        from_multiscale(ms, shapes={"s0": (2, 2)})
