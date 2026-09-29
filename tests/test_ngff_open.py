"""Synthetic OME-Zarr reader checks for older and current metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest
import zarr
from dask.callbacks import Callback
from numpy.testing import assert_allclose

from xarrayrf import anatomy, coordinate_system_change
from xarrayrf.ngff import open

AXES = [
    {"name": "t", "type": "time", "unit": "second"},
    {"name": "c", "type": "channel"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]


def test_ngff_rfc4_orientation_relates_to_lps(tmp_path: Path) -> None:
    path = tmp_path / "oriented.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=3)
    group.create_array("0", data=np.zeros((2, 3, 4, 5), dtype=np.float64))
    group.attrs["ome"] = {
        "version": "0.6",
        "multiscales": [
            {
                "coordinateSystems": [
                    {
                        "name": "intrinsic",
                        "axes": [
                            {"name": "c", "type": "channel"},
                            *[
                                {
                                    "name": name,
                                    "type": "space",
                                    "unit": "micrometer",
                                    "orientation": token,
                                }
                                for name, token in zip(("x", "y", "z"), anatomy.RAS, strict=True)
                            ],
                        ],
                    }
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [1, 2, 3, 4],
                                "input": {"path": "0"},
                                "output": {"name": "intrinsic"},
                            }
                        ],
                    }
                ],
            }
        ],
    }
    frame = open(path).rf.geometry.frame
    assert frame.coordinate_system.vocabulary == anatomy.VOCABULARY
    assert frame.coordinate_system.orientation == anatomy.RAS
    lps = frame.with_coordinate_system(anatomy.patient_coordinate_system(anatomy.LPS, "um"))
    change = coordinate_system_change(frame, lps)
    # The expected change is an exact signed permutation; allow only rounding error.
    assert_allclose(change.matrix, np.diag([-1.0, -1.0, 1.0]), rtol=0, atol=1e-12)
    assert_allclose(change.translation, 0.0, rtol=0, atol=1e-12)


def make_store(tmp_path: Path, version: str) -> str:
    path = str(tmp_path / f"image-{version}.zarr")
    group = zarr.open_group(path, mode="w", zarr_format=2 if version == "0.4" else 3)
    for name, shape in (("0", (3, 2, 4, 5)), ("1", (3, 2, 2, 3))):
        group.create_array(name, data=np.arange(np.prod(shape)).reshape(shape))
    if version in ("0.4", "0.5"):
        scale: dict[str, Any] = {
            "name": "image",
            "axes": AXES,
            "datasets": [
                {
                    "path": path,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [2, 1, 3 * factor, 4 * factor]},
                        {"type": "translation", "translation": [5, 0, 7, 8]},
                    ],
                }
                for path, factor in (("0", 1), ("1", 2))
            ],
            "coordinateTransformations": [
                {"type": "scale", "scale": [1, 1, 1, 1]},
                {"type": "translation", "translation": [11, 0, 13, 17]},
            ],
        }
        if version == "0.4":
            scale["version"] = "0.4"
            group.attrs["multiscales"] = [scale]
        else:
            group.attrs["ome"] = {"version": "0.5", "multiscales": [scale]}
    else:
        scale = {
            "name": "image",
            "coordinateSystems": [{"name": "intrinsic", "axes": AXES}],
            "datasets": [
                {
                    "path": path,
                    "coordinateTransformations": [
                        {
                            "type": "sequence",
                            "input": {"path": path},
                            "output": {"name": "intrinsic"},
                            "transformations": [
                                {"type": "scale", "scale": [2, 1, 3 * factor, 4 * factor]},
                                {"type": "translation", "translation": [5, 0, 7, 8]},
                            ],
                        }
                    ],
                }
                for path, factor in (("0", 1), ("1", 2))
            ],
        }
        group.attrs["ome"] = {"version": "0.6", "multiscales": [scale]}
    return path


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_ngff_levels_points_and_laziness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    path = make_store(tmp_path, version)
    tasks = []
    reads = []
    original = zarr.Array.__getitem__

    def counted(self: Any, selection: Any) -> Any:
        if not (
            isinstance(selection, tuple)
            and all(isinstance(part, slice) and part.start == part.stop == 0 for part in selection)
        ):
            reads.append(selection)
        return original(self, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", counted)
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        image = open(path)
    assert not tasks and not reads
    assert isinstance(image.data, da.Array)
    assert image.dims == ("t", "c", "y", "x")
    assert image.shape == (3, 2, 4, 5)
    expected = [16, 20, 25] if version in ("0.4", "0.5") else [5, 7, 8]
    assert_allclose(
        image.rf.geometry.points().isel(t=0, y=0, x=0).data, expected, rtol=0, atol=1e-12
    )
    assert not reads
    image.isel(t=0, c=0, y=0, x=0).compute()
    assert reads
    selected = open(path, level="1", chunks=None)
    assert isinstance(selected.data, np.ndarray)
    assert selected.shape == (3, 2, 2, 3)
    assert_allclose(
        selected.rf.geometry.points().isel(t=1, y=1, x=1).data,
        np.add(expected, [2, 6, 8]),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_ngff_scale_only_older_datasets(tmp_path: Path, version: str) -> None:
    path = make_store(tmp_path, version)
    group = zarr.open_group(path, mode="a")
    attrs = cast(dict[str, Any], group.attrs.asdict())
    scale = attrs["multiscales"][0] if version == "0.4" else attrs["ome"]["multiscales"][0]
    for dataset in scale["datasets"]:
        del dataset["coordinateTransformations"][1]
    del scale["coordinateTransformations"]
    group.attrs.update(attrs)
    image = open(path)
    assert_allclose(
        image.rf.geometry.points().isel(t=1, y=1, x=1).data, [2, 3, 4], rtol=0, atol=1e-12
    )


def test_ngff_selection_and_errors(tmp_path: Path) -> None:
    path = make_store(tmp_path, "0.4")
    group = zarr.open_group(path, mode="a")
    scales = cast(list[dict[str, Any]], group.attrs["multiscales"])
    group.attrs["multiscales"] = [scales[0], {**scales[0], "name": "second"}]
    with pytest.raises(ValueError, match=r"multiple multiscales.*0: image.*1: second"):
        open(path)
    assert open(path, multiscale="second").shape == (3, 2, 4, 5)
    with pytest.raises(ValueError, match=r"unknown level.*0, 1"):
        open(path, multiscale=0, level="bad")
    with pytest.raises(ValueError, match="unknown multiscale"):
        open(path, multiscale=3)
    with pytest.raises(TypeError, match="multiscale"):
        open(path, multiscale=1.2)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        open(tmp_path / "absent.zarr")
    group.attrs["multiscales"] = [{**scales[0], "version": "0.2"}]
    with pytest.raises(ValueError, match="unknown OME-Zarr"):
        open(path)


def test_ngff_path_transform_and_argument_errors(tmp_path: Path) -> None:
    path = make_store(tmp_path, "0.4")
    group = zarr.open_group(path, mode="a")
    with pytest.raises(TypeError, match="group"):
        open(path, group=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="group"):
        open(path, group="../image")
    with pytest.raises(TypeError, match="level"):
        open(path, level=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown or ambiguous multiscale"):
        open(path, multiscale="missing")
    scale = cast(list[dict[str, Any]], group.attrs["multiscales"])[0]
    scale["datasets"][0]["coordinateTransformations"][0] = {"type": "scale", "path": "scale"}
    group.attrs["multiscales"] = [scale]
    with pytest.raises(ValueError, match="path-based"):
        open(path)
    current = make_store(tmp_path, "0.6")
    other = zarr.open_group(current, mode="a")
    other.attrs["ome"] = {"version": "0.6", "multiscales": []}
    with pytest.raises(ValueError, match="no multiscales"):
        open(current)


def test_ngff_urls_go_to_zarr_and_name_the_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = make_store(tmp_path, "0.4")
    opened: list[object] = []
    original = zarr.open_group

    def redirect(store: object, **kwargs: Any) -> object:
        opened.append(store)
        return original(local, **kwargs)  # stand in for the remote store

    monkeypatch.setattr(zarr, "open_group", redirect)
    image = open("https://example.org/data/image.zarr/")
    assert opened == ["https://example.org/data/image.zarr/"]
    assert image.rf.reference_frame.identifier[1].startswith("https://example.org/data/image.zarr#")


def test_store_units_are_canonicalized(tmp_path: Path) -> None:
    opened = open(make_store(tmp_path, "0.5"))
    assert opened.rf.reference_frame.coordinate_system.units == ("s", "um", "um")
