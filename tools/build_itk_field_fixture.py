"""Generate the ITK oracle for field transforms (tests/test_field_transforms.py).

ITK is not a test dependency, so its answers are recorded once with their provenance: a dense
``DisplacementFieldTransform`` on an oblique, anisotropic grid, the same on a single-slice slab,
and a cubic ``BSplineTransform``, each queried at random interior points and at continuous
indices on and just beside its support edges.

    uv run --no-project --with itk --with numpy python tools/build_itk_field_fixture.py
    uv run --no-project --with itk --with numpy python tools/build_itk_field_fixture.py --apply

Without ``--apply`` the JSON goes to stdout and nothing is written.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import itk
import numpy as np
import numpy.typing as npt

FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "itk_field_boundaries.json"
)

# Offsets from an edge, in continuous index: on it, and one millionth of a sample either side.
EDGE_OFFSETS = (-1e-6, 0.0, 1e-6)


def _rotation(angle: float) -> npt.NDArray[np.float64]:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _probes(
    rng: np.random.Generator, shape: tuple[int, ...], low: list[float], high: list[float]
) -> npt.NDArray[np.float64]:
    """Continuous indices: random interior points, then each axis's edges with the others inside."""
    lows, highs = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    interior = lows + (highs - lows) * rng.uniform(0.05, 0.95, size=(24, len(shape)))
    centre = (lows + highs) / 2
    edges = []
    for axis in range(len(shape)):
        for edge in (lows[axis], highs[axis]):
            for offset in EDGE_OFFSETS:
                point = centre.copy()
                point[axis] = edge + offset
                edges.append(point)
    return np.concatenate([interior, np.array(edges)])


def _physical(
    origin: npt.ArrayLike,
    direction: npt.ArrayLike,
    spacing: npt.ArrayLike,
    index: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.asarray(np.asarray(origin) + index @ (np.asarray(direction) * np.asarray(spacing)).T)


def _dense(
    rng: np.random.Generator,
    shape: tuple[int, int, int],
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    direction: npt.NDArray[np.float64],
) -> dict[str, object]:
    field = rng.normal(scale=0.5, size=(*shape, 3))
    image = itk.image_from_array(
        np.ascontiguousarray(np.transpose(field, (2, 1, 0, 3))), is_vector=True
    )
    image.SetSpacing(list(spacing))
    image.SetOrigin(list(origin))
    image.SetDirection(itk.matrix_from_array(np.asarray(direction, dtype=float)))
    transform = itk.DisplacementFieldTransform[itk.D, 3].New()
    transform.SetDisplacementField(image)
    low = [-0.5] * 3
    high = [size - 0.5 for size in shape]
    index = _probes(rng, shape, low, high)
    points = _physical(origin, direction, spacing, index)
    mapped = np.array([list(transform.TransformPoint(list(point))) for point in points])
    return {
        "shape": list(shape),
        "spacing": list(spacing),
        "origin": list(origin),
        "direction": np.asarray(direction).tolist(),
        "field": field.tolist(),
        "index": index.tolist(),
        "points": points.tolist(),
        "mapped": mapped.tolist(),
    }


def _bspline(rng: np.random.Generator) -> dict[str, object]:
    mesh = 3
    origin = [1.5, -2.0, 0.25]
    spacing = [0.8, 1.2, 1.0]
    transform = itk.BSplineTransform[itk.D, 3, 3].New()
    transform.SetTransformDomainOrigin(origin)
    transform.SetTransformDomainPhysicalDimensions([s * mesh for s in spacing])
    transform.SetTransformDomainMeshSize([mesh] * 3)
    size = mesh + 3
    coefficients = rng.normal(scale=0.4, size=(size, size, size, 3))
    # ITK's parameters are component-major, each component image in x-fastest order.
    parameters = np.concatenate(
        [np.transpose(coefficients[..., c], (2, 1, 0)).ravel() for c in range(3)]
    )
    flat = itk.OptimizerParameters[itk.D](parameters.size)
    for position, value in enumerate(parameters):
        flat[position] = float(value)
    transform.SetParameters(flat)
    # The control grid starts one spacing before the domain origin.
    grid_origin = [o - s for o, s in zip(origin, spacing, strict=True)]
    index = _probes(rng, (size,) * 3, [1.0] * 3, [size - 2.0] * 3)
    points = _physical(grid_origin, np.eye(3), spacing, index)
    mapped = np.array([list(transform.TransformPoint(list(point))) for point in points])
    return {
        "shape": [size] * 3,
        "spacing": spacing,
        "origin": grid_origin,
        "direction": np.eye(3).tolist(),
        "coefficients": coefficients.tolist(),
        "index": index.tolist(),
        "points": points.tolist(),
        "mapped": mapped.tolist(),
    }


def build() -> dict[str, object]:
    rng = np.random.default_rng(20261012)
    return {
        "itk_version": itk.Version.GetITKVersion(),
        "python_version": platform.python_version(),
        "note": "index columns and field axes are (x, y, z); field arrays are indexed [x, y, z, component]",
        "dense": _dense(rng, (4, 5, 3), (0.7, 1.1, 1.3), (2.0, -3.0, 5.0), _rotation(0.3)),
        "slab": _dense(rng, (4, 3, 1), (1.0, 0.9, 2.5), (0.0, 1.0, -4.0), np.eye(3)),
        "bspline": _bspline(rng),
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
