"""Generate the SimpleITK and ANTs oracle for the ITK and ANTs importers (tests/test_itk.py,
tests/test_ants.py).

Neither ITK nor ANTs is a test dependency, so their files and answers are recorded once with
their provenance: one transform file per supported ITK type and dimension, written by
SimpleITK; a composite HDF5 file; NIfTI displacement fields; and a small ANTs SyN registration.
Each is queried with SimpleITK's double-precision ``TransformPoint``.

    uv run --no-project --with simpleitk --with antspyx --with numpy \
        python tools/build_itk_transform_fixture.py
    uv run --no-project --with simpleitk --with antspyx --with numpy \
        python tools/build_itk_transform_fixture.py --apply

Without ``--apply`` the JSON goes to stdout and nothing is written. ANTs registration is not
bit-reproducible across machines, so its files are regenerated only on purpose.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import ants
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk  # noqa: N813 (SimpleITK's own spelling)

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
DIRECTORY = FIXTURES / "itk"


def _rotation(angles: tuple[float, float, float]) -> npt.NDArray[np.float64]:
    a, b, c = angles
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return np.asarray(rz @ ry @ rx, dtype=float)


def _query(transform: sitk.Transform, points: npt.NDArray[np.float64]) -> list[list[float]]:
    return [list(transform.TransformPoint([float(v) for v in point])) for point in points]


def _field_image(
    rng: np.random.Generator,
    size: tuple[int, ...],
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    direction: npt.NDArray[np.float64],
) -> sitk.Image:
    vectors = rng.normal(scale=0.7, size=(*reversed(size), len(size)))
    image = sitk.GetImageFromArray(vectors, isVector=True)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection([float(v) for v in direction.ravel()])
    return image


def _read_field(path: Path) -> sitk.DisplacementFieldTransform:
    return sitk.DisplacementFieldTransform(sitk.ReadImage(str(path), sitk.sitkVectorFloat64))


def _inside(
    rng: np.random.Generator, image: sitk.Image, count: int = 40
) -> npt.NDArray[np.float64]:
    """Physical points well inside a field image's samples."""
    size = np.array(image.GetSize(), dtype=float)
    index = rng.uniform(0.2, 1.0, size=(count, len(size))) * (size - 1.4)
    return np.array(
        [image.TransformContinuousIndexToPhysicalPoint([float(v) for v in row]) for row in index]
    )


def _linear_cases(rng: np.random.Generator) -> dict[str, tuple[sitk.Transform, str]]:
    cases: dict[str, tuple[sitk.Transform, str]] = {}
    affine3 = sitk.AffineTransform(3)
    affine3.SetMatrix(
        [float(v) for v in (_rotation((0.1, -0.2, 0.3)) @ np.diag([1.1, 0.9, 1.05])).ravel()]
    )
    affine3.SetTranslation((1.5, -2.0, 0.75))
    affine3.SetCenter((10.0, -5.0, 3.0))
    cases["affine_3d.tfm"] = (affine3, "tfm")
    cases["affine_3d.mat"] = (affine3, "mat")
    cases["affine_3d.h5"] = (affine3, "h5")
    affine2 = sitk.AffineTransform(2)
    affine2.SetMatrix((0.9, 0.2, -0.15, 1.1))
    affine2.SetTranslation((2.0, -1.0))
    affine2.SetCenter((4.0, 3.0))
    cases["affine_2d.tfm"] = (affine2, "tfm")
    cases["affine_2d.mat"] = (affine2, "mat")
    translation = sitk.TranslationTransform(3, (0.5, -1.25, 2.0))
    cases["translation_3d.tfm"] = (translation, "tfm")
    euler2 = sitk.Euler2DTransform((1.0, 2.0), 0.35, (0.5, -0.75))
    cases["euler_2d.tfm"] = (euler2, "tfm")
    euler3 = sitk.Euler3DTransform((2.0, -1.0, 4.0), 0.2, -0.1, 0.3, (1.0, 2.0, -0.5))
    cases["euler_3d.tfm"] = (euler3, "tfm")
    euler3zyx = sitk.Euler3DTransform((2.0, -1.0, 4.0), 0.2, -0.1, 0.3, (1.0, 2.0, -0.5))
    euler3zyx.SetComputeZYX(True)
    cases["euler_3d_zyx.tfm"] = (euler3zyx, "tfm")
    versor = sitk.VersorRigid3DTransform()
    versor.SetCenter((3.0, 1.0, -2.0))
    versor.SetRotation((0.2, -0.3, 0.5), 0.4)
    versor.SetTranslation((0.25, 0.5, -1.0))
    cases["versor_rigid_3d.tfm"] = (versor, "tfm")
    similarity = sitk.Similarity3DTransform()
    similarity.SetCenter((-1.0, 2.0, 0.5))
    similarity.SetRotation((0.6, 0.1, -0.2), -0.3)
    similarity.SetTranslation((1.0, -0.5, 0.25))
    similarity.SetScale(1.15)
    cases["similarity_3d.tfm"] = (similarity, "tfm")
    cases["euler_3d.h5"] = (euler3, "h5")
    del rng
    return cases


def euler_for_composite() -> sitk.Euler3DTransform:
    return sitk.Euler3DTransform((0.5, 1.0, -1.5), -0.2, 0.15, 0.4, (2.0, -1.0, 0.5))


def _ants_registration(directory: Path) -> dict[str, Any]:
    z, y, x = np.indices((16, 18, 15), dtype=np.float32)

    def blob(cx: float, cy: float, cz: float) -> npt.NDArray[np.float64]:
        main = np.exp(-((x - cx) ** 2 / 12 + (y - cy) ** 2 / 16 + (z - cz) ** 2 / 14))
        side = np.exp(-((x - cx - 3) ** 2 / 5 + (y - cy + 3) ** 2 / 6 + (z - cz) ** 2 / 5))
        return np.asarray(main + 0.6 * side, dtype=np.float32)

    geometry = {"origin": (10.0, -20.0, 5.0), "spacing": (1.2, 1.0, 1.1)}
    fixed = ants.from_numpy(blob(7, 9, 8).transpose(2, 1, 0).copy(), **geometry)
    moving = ants.from_numpy(blob(8, 8.5, 8.5).transpose(2, 1, 0).copy(), **geometry)
    with tempfile.TemporaryDirectory() as scratch:
        result = ants.registration(
            fixed,
            moving,
            type_of_transform="SyN",
            outprefix=os.path.join(scratch, "reg_"),
            random_seed=1,
        )
        names = {
            "warp": "ants_1Warp.nii.gz",
            "affine": "ants_0GenericAffine.mat",
            "inverse_warp": "ants_1InverseWarp.nii.gz",
        }
        sources = {
            "warp": result["fwdtransforms"][0],
            "affine": result["fwdtransforms"][1],
            "inverse_warp": result["invtransforms"][1],
        }
        for key, name in names.items():
            shutil.copyfile(sources[key], directory / name)
    affine = sitk.ReadTransform(str(directory / names["affine"]))
    warp = sitk.DisplacementFieldTransform(
        sitk.ReadImage(str(directory / names["warp"]), sitk.sitkVectorFloat64)
    )
    inverse_warp = sitk.DisplacementFieldTransform(
        sitk.ReadImage(str(directory / names["inverse_warp"]), sitk.sitkVectorFloat64)
    )
    rng = np.random.default_rng(11)
    lower = np.array(geometry["origin"]) + 2.0
    upper = lower + np.array([12.0, 13.0, 12.0])
    points = rng.uniform(lower, upper, size=(40, 3))
    forward = [affine.TransformPoint(warp.TransformPoint([float(v) for v in p])) for p in points]
    inverse_affine = affine.GetInverse()
    backward_points = np.array(forward)
    backward = [
        inverse_warp.TransformPoint(inverse_affine.TransformPoint([float(v) for v in q]))
        for q in backward_points
    ]
    return {
        "files": names,
        "affine_type": affine.GetName(),
        "points": points.tolist(),
        "forward": [list(p) for p in forward],
        "backward": [list(p) for p in backward],
    }


def build(directory: Path) -> dict[str, Any]:
    rng = np.random.default_rng(20261013)
    cases: dict[str, Any] = {}
    for name, (transform, _) in _linear_cases(rng).items():
        sitk.WriteTransform(transform, str(directory / name))
        dimension = transform.GetDimension()
        points = rng.uniform(-10.0, 10.0, size=(12, dimension))
        cases[name] = {"points": points.tolist(), "mapped": _query(transform, points)}
    oblique = _rotation((0.15, -0.1, 0.25))
    field3 = _field_image(rng, (6, 7, 5), (1.5, 1.2, 2.0), (-3.0, 2.0, 1.0), oblique)
    sitk.WriteImage(field3, str(directory / "field_3d.nii.gz"))
    # Query the field as ITK reads it back: NIfTI stores its geometry in float32.
    field_transform = _read_field(directory / "field_3d.nii.gz")
    points = _inside(rng, field3)
    cases["field_3d.nii.gz"] = {
        "points": points.tolist(),
        "mapped": _query(field_transform, points),
    }
    field2 = _field_image(
        rng, (7, 6), (1.25, 0.8), (1.0, -2.0), np.array([[0.96, -0.28], [0.28, 0.96]])
    )
    sitk.WriteImage(field2, str(directory / "field_2d.nii.gz"))
    field2_transform = _read_field(directory / "field_2d.nii.gz")
    points = _inside(rng, field2)
    cases["field_2d.nii.gz"] = {
        "points": points.tolist(),
        "mapped": _query(field2_transform, points),
    }
    # A composite: an affine added first and a field added last, so the field applies first.
    inner = _field_image(rng, (6, 6, 6), (1.0, 1.1, 0.9), (-2.0, -2.0, -2.0), np.eye(3))
    affine = sitk.AffineTransform(3)
    affine.SetMatrix([float(v) for v in _rotation((0.05, 0.1, -0.08)).ravel()])
    affine.SetTranslation((0.5, 0.25, -0.75))
    affine.SetCenter((1.0, 1.0, 1.0))
    composite = sitk.CompositeTransform(
        [affine, sitk.DisplacementFieldTransform(sitk.Cast(inner, sitk.sitkVectorFloat64))]
    )
    sitk.WriteTransform(composite, str(directory / "composite_3d.h5"))
    points = _inside(rng, inner)
    cases["composite_3d.h5"] = {"points": points.tolist(), "mapped": _query(composite, points)}
    # A text composite of linear members: the Euler added last applies first.
    text_composite = sitk.CompositeTransform([affine, euler_for_composite()])
    sitk.WriteTransform(text_composite, str(directory / "composite_3d.tfm"))
    points = rng.uniform(-10.0, 10.0, size=(12, 3))
    cases["composite_3d.tfm"] = {
        "points": points.tolist(),
        "mapped": _query(text_composite, points),
    }
    return {
        "simpleitk_version": sitk.Version.VersionString(),
        "antspyx_version": version("antspyx"),
        "python_version": platform.python_version(),
        "note": "points and mapped are ITK physical (LPS) coordinates; files are in tests/fixtures/itk",
        "files": cases,
        "ants": _ants_registration(directory),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help=f"write {DIRECTORY} and its JSON")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        staging = Path(scratch) / "itk"
        staging.mkdir()
        record = build(staging)
        record_text = json.dumps(record, indent=1) + "\n"
        if not args.apply:
            sys.stdout.write(record_text)
            return 0
        (staging / "oracle.json").write_text(record_text, encoding="utf-8")
        for path in staging.iterdir():
            path.chmod(0o644)
        # Replace the directory only once everything was generated, so a failure never leaves a
        # partial fixture.
        replaced = DIRECTORY.with_name("itk.old")
        if DIRECTORY.exists():
            DIRECTORY.rename(replaced)
        shutil.copytree(staging, DIRECTORY)
        if replaced.exists():
            shutil.rmtree(replaced)
    print(f"wrote {DIRECTORY}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
