"""Generate the transformix oracle for the elastix importer (tests/test_elastix.py).

elastix is not a test dependency, so transformix's answers are recorded once with their
provenance. Each case is a stage chain of hand-built parameter maps, as elastix writes them,
evaluated by transformix as a deformation field over an oblique output grid; B-spline cases
also probe points on and beside the control grid's support faces, one transformix call per
point (a one-voxel output grid placed at the point).

    uv run --no-project --with itk-elastix --with numpy python tools/build_elastix_fixture.py
    uv run --no-project --with itk-elastix --with numpy python tools/build_elastix_fixture.py --apply

Without ``--apply`` the JSON goes to stdout and nothing is written.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import itk
import numpy as np

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "elastix_transforms.json"

# Offsets from a support face, in control-grid index: on it, and a thousandth either side.
FACE_OFFSETS = (-1e-3, 0.0, 1e-3)

Map = dict[str, list[str]]


def _text(values: Any) -> list[str]:
    return [repr(float(v)) if isinstance(v, float) else str(v) for v in np.ravel(values)]


def _common(dimension: int) -> Map:
    return {
        "HowToCombineTransforms": ["Compose"],
        "InitialTransformParameterFileName": ["NoInitialTransform"],
        "FixedImageDimension": [str(dimension)],
        "MovingImageDimension": [str(dimension)],
        "FixedInternalImagePixelType": ["float"],
        "MovingInternalImagePixelType": ["float"],
        "UseDirectionCosines": ["true"],
        "ResampleInterpolator": ["FinalBSplineInterpolator"],
        "FinalBSplineInterpolationOrder": ["3"],
        "Resampler": ["DefaultResampler"],
        "DefaultPixelValue": ["0"],
        "ResultImagePixelType": ["float"],
        "WriteResultImage": ["false"],
    }


def _rotation(angles: tuple[float, ...]) -> np.ndarray:
    if len(angles) == 1:
        c, s = np.cos(angles[0]), np.sin(angles[0])
        return np.array([[c, -s], [s, c]])
    rotation = np.eye(3)
    for axis, angle in enumerate(angles):
        c, s = np.cos(angle), np.sin(angle)
        step = np.eye(3)
        a, b = [n for n in range(3) if n != axis]
        step[a, a], step[a, b], step[b, a], step[b, b] = c, -s, s, c
        rotation = step @ rotation
    return rotation


def _output_grid(dimension: int) -> dict[str, Any]:
    """An oblique, anisotropic output grid whose voxel centres are the dense query points."""
    if dimension == 3:
        return {
            "size": [5, 6, 4],
            "spacing": [1.7, 1.3, 2.1],
            "origin": [-3.0, -4.0, -2.5],
            "direction": _rotation((0.2, -0.1, 0.35)),
        }
    return {
        "size": [7, 5],
        "spacing": [1.4, 1.9],
        "origin": [-4.0, -3.0],
        "direction": _rotation((0.25,)),
    }


def _output_keys(grid: dict[str, Any]) -> Map:
    return {
        "Size": _text(grid["size"]),
        "Index": ["0"] * len(grid["size"]),
        "Spacing": _text(np.asarray(grid["spacing"], dtype=float)),
        "Origin": _text(np.asarray(grid["origin"], dtype=float)),
        # ITK writes Direction row by row of the matrix's transpose: column-major.
        "Direction": _text(np.asarray(grid["direction"]).ravel(order="F")),
    }


def _bspline(
    rng: np.random.Generator,
    dimension: int,
    *,
    name: str = "BSplineTransform",
    size: tuple[int, ...],
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    index: tuple[int, ...],
    direction: np.ndarray,
) -> Map:
    # Component-major parameters, each component's control grid with the first index fastest.
    coefficients = rng.normal(scale=0.6, size=(dimension, *reversed(size)))
    return {
        "Transform": [name],
        "NumberOfParameters": [str(coefficients.size)],
        "TransformParameters": _text(coefficients),
        "BSplineTransformSplineOrder": ["3"],
        "UseCyclicTransform": ["false"],
        "GridSize": _text(size),
        "GridIndex": _text(index),
        "GridSpacing": _text(np.asarray(spacing, dtype=float)),
        "GridOrigin": _text(np.asarray(origin, dtype=float)),
        "GridDirection": _text(direction.ravel(order="F")),
        **_common(dimension),
    }


def _euler(dimension: int, values: list[float], center: list[float], *, zyx: bool) -> Map:
    entry = {
        "Transform": ["EulerTransform"],
        "NumberOfParameters": [str(len(values))],
        "TransformParameters": _text(np.asarray(values, dtype=float)),
        "CenterOfRotationPoint": _text(np.asarray(center, dtype=float)),
        **_common(dimension),
    }
    if dimension == 3:
        entry["ComputeZYX"] = ["true" if zyx else "false"]
    return entry


def _affine(dimension: int, matrix: Any, translation: list[float], center: list[float]) -> Map:
    values = np.concatenate([np.asarray(matrix, dtype=float).ravel(), translation])
    return {
        "Transform": ["AffineTransform"],
        "NumberOfParameters": [str(values.size)],
        "TransformParameters": _text(values),
        "CenterOfRotationPoint": _text(np.asarray(center, dtype=float)),
        **_common(dimension),
    }


def _translation(dimension: int, values: list[float]) -> Map:
    return {
        "Transform": ["TranslationTransform"],
        "NumberOfParameters": [str(dimension)],
        "TransformParameters": _text(np.asarray(values, dtype=float)),
        **_common(dimension),
    }


def _transformix(maps: list[Map], grid: dict[str, Any], directory: Path) -> np.ndarray:
    """Displacements at the output grid's points, shaped like the grid (first index fastest)."""
    dimension = len(grid["size"])
    parameters = itk.ParameterObject.New()
    for entry in maps:
        parameters.AddParameterMap({**entry, **_output_keys(grid)})
    image = itk.image_from_array(np.zeros(tuple(reversed(grid["size"])), dtype=np.float32))
    transformix = itk.TransformixFilter.New(image)
    transformix.SetOutputDirectory(str(directory))
    transformix.SetTransformParameterObject(parameters)
    transformix.SetComputeDeformationField(True)
    transformix.SetLogToConsole(False)
    transformix.SetLogToFile(False)
    transformix.UpdateLargestPossibleRegion()
    field = np.asarray(itk.array_from_image(transformix.GetOutputDeformationField()), dtype=float)
    # itk arrays index the last axis fastest; return the first index fastest.
    return np.transpose(field, (*reversed(range(dimension)), dimension))


def _grid_points(grid: dict[str, Any]) -> np.ndarray:
    dimension = len(grid["size"])
    index = np.moveaxis(np.indices(grid["size"], dtype=float), 0, -1)
    matrix = np.asarray(grid["direction"]) * np.asarray(grid["spacing"])
    points = np.asarray(grid["origin"]) + index @ matrix.T
    return np.asarray(points, dtype=float).reshape(-1, dimension)


def _face_probes(entry: Map, directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Points on and beside each support face of a B-spline's control grid, and their images."""
    dimension = int(entry["FixedImageDimension"][0])
    size = np.asarray(entry["GridSize"], dtype=float)
    index = np.asarray(entry["GridIndex"], dtype=float)
    spacing = np.asarray(entry["GridSpacing"], dtype=float)
    origin = np.asarray(entry["GridOrigin"], dtype=float)
    direction = np.asarray(entry["GridDirection"], dtype=float).reshape(
        dimension, dimension, order="F"
    )
    centre = (size - 1) / 2
    probes = []
    for axis in range(dimension):
        for face in (1.0, size[axis] - 2):
            for offset in FACE_OFFSETS:
                probe = centre.copy()
                probe[axis] = face + offset
                probes.append(probe)
    points = origin + ((np.asarray(probes) + index) * spacing) @ direction.T
    mapped = []
    for point in points:
        grid = {
            "size": [1] * dimension,
            "spacing": [1.0] * dimension,
            "origin": point,
            "direction": np.eye(dimension),
        }
        mapped.append(point + _transformix([entry], grid, directory).reshape(dimension))
    return points, np.asarray(mapped)


def _case(maps: list[Map], directory: Path, *, faces: bool = False) -> dict[str, Any]:
    dimension = int(maps[0]["FixedImageDimension"][0])
    grid = _output_grid(dimension)
    points = _grid_points(grid)
    mapped = points + _transformix(maps, grid, directory).reshape(-1, dimension)
    files = []
    for position, entry in enumerate(maps):
        path = directory / f"TransformParameters.{position}.txt"
        written = dict(entry)
        if position:
            # elastix records the preceding stage's file; relative here, resolved by the reader.
            written["InitialTransformParameterFileName"] = [
                f"TransformParameters.{position - 1}.txt"
            ]
        single = itk.ParameterObject.New()
        single.AddParameterMap(written)
        single.WriteParameterFile(str(path))
        files.append(path.read_text(encoding="utf-8"))
    case: dict[str, Any] = {
        "maps": maps,
        "files": files,
        "points": points.tolist(),
        "mapped": mapped.tolist(),
    }
    if faces:
        face_points, face_mapped = _face_probes(maps[-1], directory)
        case["face_points"] = face_points.tolist()
        case["face_mapped"] = face_mapped.tolist()
    return case


def build() -> dict[str, Any]:
    rng = np.random.default_rng(20261013)
    cases: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        spline_3d: dict[str, Any] = {
            "size": (6, 7, 8),
            "spacing": (2.5, 2.0, 2.75),
            "origin": (-9.0, -8.0, -12.5),
            "index": (2, -3, 1),
            "direction": _rotation((0.19, -0.31, 0.43)),
        }
        cases["rigid_bspline_3d"] = _case(
            [
                _euler(3, [0.1, -0.2, 0.15, 1.5, -0.5, 0.75], [1.0, 2.0, -1.0], zyx=False),
                _bspline(rng, 3, **spline_3d),
            ],
            directory,
            faces=True,
        )
        cases["zyx_affine_translation_3d"] = _case(
            [
                _euler(3, [0.2, 0.1, -0.3, 0.5, 1.0, -1.5], [-2.0, 0.5, 3.0], zyx=True),
                _affine(
                    3,
                    [[1.1, 0.1, -0.05], [0.02, 0.95, 0.1], [0.0, -0.08, 1.05]],
                    [1.0, -2.0, 0.5],
                    [0.5, -1.0, 2.0],
                ),
                _translation(3, [0.25, -0.75, 1.25]),
            ],
            directory,
        )
        # elastix's recursive B-spline takes its support from GridIndex but evaluates as if it
        # were zero (itk-elastix 0.25.4), so it is recorded with the zero index elastix writes.
        cases["recursive_bspline_3d"] = _case(
            [
                _bspline(
                    rng, 3, name="RecursiveBSplineTransform", **{**spline_3d, "index": (0, 0, 0)}
                )
            ],
            directory,
            faces=True,
        )
        cases["euler_bspline_2d"] = _case(
            [
                _euler(2, [0.3, 1.0, -0.5], [0.5, 1.5], zyx=False),
                _bspline(
                    rng,
                    2,
                    size=(7, 6),
                    spacing=(2.0, 2.5),
                    origin=(-8.0, -7.5),
                    index=(1, -2),
                    direction=_rotation((-0.2,)),
                ),
            ],
            directory,
            faces=True,
        )
        cases["affine_2d"] = _case(
            [_affine(2, [[0.9, 0.2], [-0.1, 1.1]], [2.0, -1.0], [1.0, 1.0])], directory
        )
    return {
        "itk_version": itk.Version.GetITKVersion(),
        "itk_elastix_version": version("itk-elastix"),
        "python_version": platform.python_version(),
        "note": "points and mapped are physical (x, y, ...); maps are in application order",
        **cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help=f"write {FIXTURE}")
    args = parser.parse_args()
    text = json.dumps(build(), indent=1) + "\n"
    if not args.apply:
        sys.stdout.write(text)
        return 0
    # Write beside the target and rename, so a failure never leaves a partial fixture.
    with tempfile.NamedTemporaryFile(
        "w", dir=FIXTURE.parent, suffix=".tmp", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(text)
    os.replace(handle.name, FIXTURE)
    print(f"wrote {FIXTURE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
