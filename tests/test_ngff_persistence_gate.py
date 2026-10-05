"""Boundary fixtures for the provisional persistence schema and NGFF 0.6."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

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
    Geometry,
    ReferenceFrame,
    decode,
    encode,
)
from xarrayrf.ngff import from_multiscale, to_multiscale_level, to_transform, transform

FIXTURES = Path(__file__).parent / "fixtures" / "ngff_0_6"

# These lists make the boundary explicit: every transformation example is classified.
LOSSLESS = (
    "affine2d2d",
    "affine2d3d",
    "byDimension1",
    "byDimension2",
    "identity",
    "mapAxis1",
    "projectAxis",
    "projectAxis2",
    "rotation",
    "scale",
    "sequence",
    "translation",
)
LOSSLESS_LEVELS = ("oblique_level",)
LOSSY_OR_REFUSED = {
    "affine2d2d_with_channel": "discrete-axis-dropped",
    "bijection": "coordinates is nonlinear",
    "bijection_verbose": "coordinates is nonlinear",
    "byDimensionCoordinates": "coordinates is nonlinear",
    "byDimensionInvalid1": "byDimension inputAxes/outputAxes indices are out of range",
    "byDimensionInvalid2": "byDimension must cover every output axis exactly once",
    "byDimensionXarray": "coordinates is nonlinear",
    "displacements/multiscales": "displacements is nonlinear",
    "scale_with_discrete": "discrete-axis-dropped",
    "xarrayLike": "coordinates is nonlinear",
}


def _read(path: Path) -> dict[str, Any]:
    text = path.read_text()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return cast(dict[str, Any], json.loads(text))


def _round_trip(value: Any) -> Any:
    restored = decode(json.loads(json.dumps(encode(value))))
    assert restored == value
    return restored


@pytest.mark.parametrize("name", LOSSLESS)
def test_lossless_transform_fixtures(name: str) -> None:
    fixture = _read(FIXTURES / "transformations" / f"{name}.json")
    systems = fixture["coordinateSystems"]
    for declaration in fixture["coordinateTransformations"]:
        imported, import_report = transform(declaration, systems, store="file:///fixture")
        assert import_report == ()
        restored = _round_trip(imported)
        assert _round_trip(imported.source) == imported.source
        assert _round_trip(imported.target) == imported.target
        assert isinstance(imported.source, ReferenceFrame)
        assert isinstance(imported.target, ReferenceFrame)
        assert _round_trip(imported.source.coordinate_system) == imported.source.coordinate_system
        assert _round_trip(imported.target.coordinate_system) == imported.target.coordinate_system
        exported, export_report = to_transform(restored)
        assert ct.Affine.model_validate(exported.model_dump()) == exported
        assert [code for code, _ in export_report] == ["frame-identity", "frame-identity"]
        assert exported.input is not None and exported.input.name == declaration["input"]["name"]
        assert exported.output is not None and exported.output.name == declaration["output"]["name"]
        reimported, _ = transform(exported, systems, store="file:///fixture")
        np.testing.assert_allclose(reimported.matrix, imported.matrix, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            reimported.translation, imported.translation, rtol=1e-12, atol=1e-12
        )
        assert isinstance(reimported.source, ReferenceFrame)
        assert isinstance(reimported.target, ReferenceFrame)
        assert isinstance(imported.source, ReferenceFrame)
        assert isinstance(imported.target, ReferenceFrame)
        assert reimported.source.coordinate_system == imported.source.coordinate_system
        assert reimported.target.coordinate_system == imported.target.coordinate_system


@pytest.mark.parametrize("name, expected", LOSSY_OR_REFUSED.items())
def test_lossy_or_refused_transform_fixtures(name: str, expected: str) -> None:
    fixture = _read(FIXTURES / "transformations" / f"{name}.json")
    if name == "displacements/multiscales":
        item = fixture["ome"]["multiscales"][0]
        declarations = item["coordinateTransformations"]
        systems = item["coordinateSystems"]
    elif name == "bijection_verbose":
        declarations = [fixture]
        systems = [{"name": value, "axes": [{"name": "x"}]} for value in ("src", "tgt")]
    else:
        declarations = fixture["coordinateTransformations"]
        systems = fixture["coordinateSystems"]
    for declaration in declarations:
        if expected == "discrete-axis-dropped":
            _, report = transform(declaration, systems, store="file:///fixture")
            assert expected in [code for code, _ in report]
            assert sum(code == "discrete-deferred" for code, _ in report) == 6
        else:
            with pytest.raises(ValueError, match=re.escape(expected)):
                transform(declaration, systems, store="file:///fixture")


def test_multiscale_lossless_boundary_fixture() -> None:
    fixture = _read(FIXTURES / "multiscales_strict" / "multiscales_transformations.json")
    original = fixture["attributes"]["ome"]["multiscales"][0]
    imported = from_multiscale(original, shapes={"s0": (4, 5)}, store="file:///fixture")
    assert imported.report == ()
    level = imported.levels["s0"]
    assert _round_trip(imported.frame) == imported.frame
    assert _round_trip(imported.frame.coordinate_system) == imported.frame.coordinate_system
    assert _round_trip(level.transform) == level.transform
    assert _round_trip(level.transform.source) == level.transform.source
    for extra in imported.transforms:
        restored = _round_trip(extra)
        exported, report = to_transform(restored)
        assert [code for code, _ in report] == ["frame-identity", "frame-identity"]
        reimported, _ = transform(exported, original["coordinateSystems"], store="file:///fixture")
        np.testing.assert_allclose(reimported.matrix, extra.matrix, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            reimported.translation, extra.translation, rtol=1e-12, atol=1e-12
        )
    array = xr.DataArray(np.zeros((4, 5)), dims=level.dims, coords=level.coords)
    metadata, report = to_multiscale_level(
        Geometry(array, level.transform, dims=level.dims), path="s0"
    )
    assert isinstance(metadata, Multiscale)
    assert Multiscale.model_validate(metadata.model_dump()) == metadata
    assert [code for code, _ in report] == ["frame-identity"]
    again = from_multiscale(metadata, shapes={"s0": (4, 5)}, store="file:///fixture")
    np.testing.assert_allclose(
        again.levels["s0"].transform.matrix, level.transform.matrix, rtol=1e-12, atol=1e-12
    )
    assert metadata.coordinateSystems[0].axes == tuple(
        ct.Axis.model_validate(axis) for axis in original["coordinateSystems"][1]["axes"]
    )


@pytest.mark.parametrize("path", LOSSLESS_LEVELS)
def test_oblique_level_lossless_boundary(path: str) -> None:
    dims = ("j", "i")
    array = xr.DataArray(
        np.zeros((3, 4)),
        dims=dims,
        coords={
            dim: (dim, np.arange(size), {"units": "1"})
            for dim, size in zip(dims, (3, 4), strict=True)
        },
    )
    frame = ReferenceFrame.local(
        CoordinateSystem(("y", "x"), ("um", "um"), axis_types=("space", "space"))
    )
    geometry = Geometry(
        array,
        AffineTransform.from_matrix(
            source=ArrayCoordinates(dims, ("1", "1"), sample_offset=(0.5, 0.5)),
            target=frame,
            matrix=[[2.0, 0.5], [0.0, 3.0]],
            translation=[4.0, 5.0],
        ),
        dims=dims,
    )
    metadata, report = to_multiscale_level(geometry, path=path)
    assert [code for code, _ in report] == ["frame-identity", "intrinsic-synthesized"]
    assert Multiscale.model_validate(metadata.model_dump()) == metadata
    imported = from_multiscale(metadata, shapes={path: array.shape}, store="file:///fixture")
    level = _round_trip(imported.levels[path].transform)
    extra = _round_trip(imported.transforms[0])
    assert level.target == extra.source
    points = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(
        extra.transform_point(level.transform_point(points)),
        geometry.lattice().transform_point(points),
        rtol=1e-12,
        atol=1e-12,
    )


def test_composite_endpoint_is_preserved_by_schema_and_refused_by_ngff() -> None:
    fixture = _read(FIXTURES / "transformations" / "scale.json")
    affine, _ = transform(
        fixture["coordinateTransformations"][0],
        fixture["coordinateSystems"],
        store="file:///fixture",
    )
    composite = CompositeTransform(affine)
    assert _round_trip(composite) == composite
    with pytest.raises(TypeError, match="t must be an AffineTransform, got CompositeTransform"):
        to_transform(composite)  # type: ignore[arg-type]
