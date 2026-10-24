"""The elastix importer against transformix, its refusals, and the parameter-file reader."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

import xarrayrf as xrf
from xarrayrf import elastix as rfe
from xarrayrf.anatomy import LPS, RAS, patient_coordinate_system

TRANSFORMIX_ATOL = 1e-5
"""transformix writes its deformation field as float32: about 1e-7 relative on these
displacements of a few millimetres, so 1e-5 mm with margin."""

ATOL = 1e-12
"""Roundoff allowance for float64 evaluation."""

FACE_ROUNDOFF = 1e-9
"""A probe this close to a support face, in control index, may land on either side by roundoff."""

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "elastix_transforms.json").read_text(encoding="utf-8")
)
CASES = [name for name, case in FIXTURE.items() if isinstance(case, dict)]
LPS_MM = patient_coordinate_system(LPS, "mm")
RAS_MM = patient_coordinate_system(RAS, "mm")
PLANE = xrf.CoordinateSystem(("x", "y"), ("mm", "mm"))


def frames(system: xrf.CoordinateSystem) -> tuple[xrf.ReferenceFrame, xrf.ReferenceFrame]:
    return (
        xrf.ReferenceFrame.declared(("test", "fixed"), system),
        xrf.ReferenceFrame.declared(("test", "moving"), system),
    )


def space_of(case: dict[str, Any]) -> xrf.CoordinateSystem | None:
    return None if len(case["points"][0]) == 3 else PLANE


def imported(maps: Any, case: dict[str, Any], **options: Any) -> rfe.ElastixImport:
    space = space_of(case)
    fixed, moving = frames(space or LPS_MM)
    return rfe.transform_from_parameter_maps(
        maps, fixed=fixed, moving=moving, space=space, **options
    )


def stage(
    name: str = "TranslationTransform", dimension: int = 3, **keys: Sequence[str] | str
) -> dict[str, tuple[str, ...]]:
    """A minimal parameter map; keys override."""
    entry: dict[str, tuple[str, ...]] = {
        "Transform": (name,),
        "FixedImageDimension": (str(dimension),),
        "MovingImageDimension": (str(dimension),),
        "TransformParameters": ("1",) * dimension,
    }
    for key, value in keys.items():
        entry[key] = (value,) if isinstance(value, str) else tuple(value)
    return entry


def spline_stage(**keys: Sequence[str] | str) -> dict[str, tuple[str, ...]]:
    """A 4-point-per-axis aligned unit B-spline map with constant coefficients (1, 2, 3)."""
    coefficients = np.concatenate([np.full(64, value) for value in (1.0, 2.0, 3.0)])
    entry = stage(
        "BSplineTransform",
        TransformParameters=[str(v) for v in coefficients],
        GridSize=["4"] * 3,
        GridIndex=["0"] * 3,
        GridSpacing=["1"] * 3,
        GridOrigin=["0"] * 3,
        GridDirection=[str(v) for v in np.eye(3).ravel()],
    )
    entry.update(
        {key: ((value,) if isinstance(value, str) else tuple(value)) for key, value in keys.items()}
    )
    return entry


# --- Agreement with transformix ---------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_imported_chains_match_transformix(name: str) -> None:
    case = FIXTURE[name]
    result = imported(case["maps"], case)
    if "bspline" in name:
        pytest.importorskip("scipy", minversion="1.18")
    mapped = result.transform.transform_point(np.asarray(case["points"]))
    assert_allclose(mapped, case["mapped"], atol=TRANSFORMIX_ATOL, rtol=0)
    stages = len(case["maps"])
    if all(
        entry["Transform"][0] not in ("BSplineTransform", "RecursiveBSplineTransform")
        for entry in case["maps"]
    ):
        assert isinstance(result.transform, xrf.AffineTransform)
    elif stages == 1:
        assert isinstance(result.transform, xrf.BSplineTransform)
    else:
        assert isinstance(result.transform, xrf.CompositeTransform)
        assert len(result.transform.transforms) == stages
    assert [code for code, _ in result.report].count("stage") == stages


@pytest.mark.parametrize("name", CASES)
def test_parameter_files_round_trip_through_the_reader(name: str, tmp_path: Path) -> None:
    case = FIXTURE[name]
    for position, text in enumerate(case["files"]):
        (tmp_path / f"TransformParameters.{position}.txt").write_text(text, encoding="utf-8")
    maps = rfe.read_parameter_files(tmp_path / f"TransformParameters.{len(case['files']) - 1}.txt")
    assert len(maps) == len(case["maps"])
    for read, given in zip(maps, case["maps"], strict=True):
        assert read["Transform"] == tuple(given["Transform"])
        assert_allclose(
            np.asarray(read["TransformParameters"], dtype=float),
            np.asarray(given["TransformParameters"], dtype=float),
            rtol=1e-15,
        )
    if "bspline" in name:
        pytest.importorskip("scipy", minversion="1.18")
    mapped = imported(maps, case).transform.transform_point(np.asarray(case["points"]))
    assert_allclose(mapped, case["mapped"], atol=TRANSFORMIX_ATOL, rtol=0)


@pytest.mark.parametrize("name", [name for name in CASES if "face_points" in FIXTURE[name]])
def test_bspline_faces_match_transformix_up_to_roundoff(name: str) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    case = FIXTURE[name]
    # The generator probes the B-spline stage alone.
    member = spline = imported(case["maps"][-1:], case).transform
    assert isinstance(member, xrf.BSplineTransform)
    assert member.support == "half-open" and member.outside == "identity"
    points = np.asarray(case["face_points"])
    mapped = spline.transform_point(points)
    to_grid = member.grid.transform
    assert isinstance(to_grid, xrf.AffineTransform)
    index = to_grid.inverse().transform_point(points)
    sizes = np.array([member.grid.sizes[dim] for dim in member.grid.dims])
    faces = np.minimum(np.abs(index - 1.0), np.abs(index - (sizes - 2.0))).min(axis=-1)
    agree = np.all(np.abs(mapped - np.asarray(case["face_mapped"])) <= TRANSFORMIX_ATOL, axis=-1)
    assert np.all(agree | (faces <= FACE_ROUNDOFF))
    assert (
        agree[faces > FACE_ROUNDOFF].all() and (faces > FACE_ROUNDOFF).sum() >= len(points) * 2 // 3
    )


def test_a_parameter_object_is_read_in_index_order() -> None:
    case = FIXTURE["zyx_affine_translation_3d"]

    class ParameterObject:
        def GetNumberOfParameterMaps(self) -> int:  # noqa: N802 (itk-elastix's name)
            return len(case["maps"])

        def GetParameterMap(self, index: int) -> Mapping[str, Sequence[str]]:  # noqa: N802
            return cast(Mapping[str, Sequence[str]], case["maps"][index])

    mapped = imported(ParameterObject(), case).transform.transform_point(np.asarray(case["points"]))
    assert_allclose(mapped, case["mapped"], atol=TRANSFORMIX_ATOL, rtol=0)


# --- Frames and coordinate systems ------------------------------------------------------------


def test_ras_frames_are_joined_by_coordinate_changes() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    case = FIXTURE["rigid_bspline_3d"]
    fixed, moving = frames(RAS_MM)
    result = rfe.transform_from_parameter_maps(case["maps"], fixed=fixed, moving=moving)
    assert isinstance(result.transform, xrf.CompositeTransform)
    first, *_, last = result.transform.transforms
    assert (first.source, last.target) == (fixed, moving)
    assert first.target == fixed.with_coordinate_system(LPS_MM)
    assert [code for code, _ in result.report].count("coordinate-system-change") == 2
    flip = np.array([-1.0, -1.0, 1.0])
    lps = np.asarray(case["points"])
    assert_allclose(
        result.transform.transform_point(lps * flip),
        np.asarray(case["mapped"]) * flip,
        atol=TRANSFORMIX_ATOL,
        rtol=0,
    )


def test_an_all_linear_chain_in_ras_collapses_to_one_affine() -> None:
    case = FIXTURE["zyx_affine_translation_3d"]
    fixed, moving = frames(RAS_MM)
    result = rfe.transform_from_parameter_maps(case["maps"], fixed=fixed, moving=moving)
    assert isinstance(result.transform, xrf.AffineTransform)
    assert (result.transform.source, result.transform.target) == (fixed, moving)


def test_intermediate_frames_can_be_named() -> None:
    case = FIXTURE["zyx_affine_translation_3d"]
    fixed, moving = frames(LPS_MM)
    between = [xrf.ReferenceFrame.declared(("test", f"stage{n}"), LPS_MM) for n in (1, 2)]
    maps = case["maps"]
    linear = rfe.transform_from_parameter_maps(
        [*maps, FIXTURE["rigid_bspline_3d"]["maps"][1]],
        fixed=fixed,
        moving=moving,
        intermediate_frames=[*between, xrf.ReferenceFrame.declared(("test", "stage3"), LPS_MM)],
    ).transform
    assert isinstance(linear, xrf.CompositeTransform)
    assert linear.transforms[0].target == between[0]
    with pytest.raises(ValueError, match="one frame per boundary"):
        rfe.transform_from_parameter_maps(
            maps, fixed=fixed, moving=moving, intermediate_frames=between[:1]
        )
    with pytest.raises(ValueError, match="coordinate system space"):
        rfe.transform_from_parameter_maps(
            maps,
            fixed=fixed,
            moving=moving,
            intermediate_frames=[xrf.ReferenceFrame.declared(("test", "r"), RAS_MM), between[1]],
        )


def test_space_is_required_beyond_three_dimensions_and_must_fit() -> None:
    case = FIXTURE["affine_2d"]
    fixed, moving = frames(PLANE)
    with pytest.raises(ValueError, match="space is required for 2-D maps"):
        rfe.transform_from_parameter_maps(case["maps"], fixed=fixed, moving=moving)
    with pytest.raises(ValueError, match="space has 3 axes but the elastix transform is 2-D"):
        rfe.transform_from_parameter_maps(case["maps"], fixed=fixed, moving=moving, space=LPS_MM)
    with pytest.raises(TypeError, match="space must be a CoordinateSystem"):
        rfe.transform_from_parameter_maps(
            case["maps"], fixed=fixed, moving=moving, space=cast(Any, "LPS")
        )


def test_an_unoriented_frame_needs_an_explicit_space() -> None:
    plain = xrf.CoordinateSystem(("x", "y", "z"), ("mm",) * 3)
    fixed, moving = frames(plain)
    with pytest.raises(ValueError):
        rfe.transform_from_parameter_maps([stage()], fixed=fixed, moving=moving)
    result = rfe.transform_from_parameter_maps([stage()], fixed=fixed, moving=moving, space=plain)
    assert_allclose(result.transform.transform_point([0.0, 0.0, 0.0]), [1.0, 1.0, 1.0], atol=ATOL)
    assert all(code == "stage" for code, _ in result.report)


def test_frames_are_checked() -> None:
    fixed, _ = frames(LPS_MM)
    with pytest.raises(ValueError, match="same frame"):
        rfe.transform_from_parameter_maps([stage()], fixed=fixed, moving=fixed)
    with pytest.raises(TypeError, match="moving must be a ReferenceFrame"):
        rfe.transform_from_parameter_maps([stage()], fixed=fixed, moving=cast(Any, None))


# --- Refusals ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "error", "match"),
    [
        (
            stage("SimilarityTransform"),
            NotImplementedError,
            "'SimilarityTransform' is not supported",
        ),
        (stage(HowToCombineTransforms="Add"), ValueError, "'Compose'"),
        (stage(UseDirectionCosines="false"), ValueError, "UseDirectionCosines must be true"),
        (stage(NumberOfParameters="4"), ValueError, "NumberOfParameters is 4"),
        (stage(TransformParameters=["1", "nan", "1"]), ValueError, "must be finite"),
        (stage(TransformParameters=["1", "one", "1"]), ValueError, "must hold numbers"),
        (stage(TransformParameters=["1", "1"]), ValueError, "must hold 3 values"),
        (
            stage("AffineTransform", TransformParameters=["1"] * 12),
            ValueError,
            "CenterOfRotationPoint is missing",
        ),
        (
            stage(
                "EulerTransform",
                TransformParameters=["0"] * 6,
                CenterOfRotationPoint=["0"] * 3,
                ComputeZYX="yes",
            ),
            ValueError,
            "ComputeZYX must be 'true' or 'false'",
        ),
        (stage(MovingImageDimension="2"), ValueError, "MovingImageDimension is 2"),
        (
            stage(InitialTransformParameterFileName="TransformParameters.0.txt"),
            ValueError,
            "read_parameter_files",
        ),
        (spline_stage(BSplineTransformSplineOrder="1"), NotImplementedError, "SplineOrder 1"),
        (spline_stage(UseCyclicTransform="true"), ValueError, "UseCyclicTransform must be false"),
        (spline_stage(GridSize=["3", "4", "4"]), ValueError, "at least 4"),
        (spline_stage(GridIndex=["0.5", "0", "0"]), ValueError, "must hold integers"),
        (spline_stage(GridSpacing=["1", "0", "1"]), ValueError, "GridSpacing must be positive"),
        (
            spline_stage(Transform="RecursiveBSplineTransform", GridIndex=["1", "0", "0"]),
            NotImplementedError,
            "GridIndex",
        ),
    ],
)
def test_malformed_or_unsupported_stages_are_refused(
    entry: dict[str, tuple[str, ...]], error: type[Exception], match: str
) -> None:
    fixed, moving = frames(LPS_MM)
    with pytest.raises(error, match=match):
        rfe.transform_from_parameter_maps([entry], fixed=fixed, moving=moving)


def test_stage_errors_name_the_stage_and_maps_are_checked() -> None:
    fixed, moving = frames(LPS_MM)
    with pytest.raises(ValueError, match="stage 1: TransformParameters"):
        rfe.transform_from_parameter_maps(
            [stage(), stage(TransformParameters=["1"])], fixed=fixed, moving=moving
        )
    with pytest.raises(ValueError, match="no stages"):
        rfe.transform_from_parameter_maps([], fixed=fixed, moving=moving)
    with pytest.raises(TypeError, match="wrap a single map in a list"):
        rfe.transform_from_parameter_maps(stage(), fixed=fixed, moving=moving)
    with pytest.raises(TypeError, match="not a parameter map"):
        rfe.transform_from_parameter_maps([cast(Any, 3)], fixed=fixed, moving=moving)
    with pytest.raises(TypeError, match="sequence of parameter maps"):
        rfe.transform_from_parameter_maps(cast(Any, 3), fixed=fixed, moving=moving)


def test_a_bspline_stage_declares_elastix_support() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    fixed, moving = frames(LPS_MM)
    result = rfe.transform_from_parameter_maps([spline_stage()], fixed=fixed, moving=moving)
    assert "bspline-support" in [code for code, _ in result.report]
    # Control index [1, 2) on a 4-point axis: lower face displaced, upper face not.
    assert_allclose(result.transform.transform_point([1.0, 1.5, 1.5]), [2.0, 3.5, 4.5], atol=ATOL)
    assert_allclose(result.transform.transform_point([2.0, 1.5, 1.5]), [2.0, 1.5, 1.5], atol=ATOL)


# --- Parameter files --------------------------------------------------------------------------


def test_the_reader_parses_comments_quotes_and_bare_words() -> None:
    text = """
    // elastix parameter file
    (Transform "BSplineTransform")   // trailing comment
    (Name "a path // with slashes" "two words")
    (Numbers 1 -2.5e-3
       7)
    (Empty)
    """
    assert rfe.parse_parameter_text(text) == {
        "Transform": ("BSplineTransform",),
        "Name": ("a path // with slashes", "two words"),
        "Numbers": ("1", "-2.5e-3", "7"),
        "Empty": (),
    }


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("(A 1)\n(A 2)", r"<text>:2: key 'A' repeats"),
        ("(A 1)\ngarbage", r"<text>:2: expected '\('"),
        ("(A 1", "not terminated"),
        ("(A 1\n(B 2)", "not terminated"),
        ('(A "open)', "unterminated quoted value"),
        ('("A" 1)', "unquoted key"),
        ("()", "unquoted key"),
    ],
)
def test_the_reader_refuses_malformed_text(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        rfe.parse_parameter_text(text)


def test_the_chain_follows_relative_and_absolute_names(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text('(Transform "A")\n', encoding="utf-8")
    (tmp_path / "b.txt").write_text(
        '(Transform "B")\n(InitialTransformParameterFileName "sub/a.txt")\n', encoding="utf-8"
    )
    (tmp_path / "c.txt").write_text(
        f'(Transform "C")\n(InitialTransformParameterFileName "{tmp_path / "b.txt"}")\n',
        encoding="utf-8",
    )
    chain = rfe.read_parameter_files(tmp_path / "c.txt")
    assert [entry["Transform"] for entry in chain] == [("A",), ("B",), ("C",)]
    assert rfe.read_parameter_file(tmp_path / "b.txt")["Transform"] == ("B",)


def test_the_chain_refuses_missing_files_and_cycles(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text(
        '(InitialTransformParameterFileName "gone.txt")\n', encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match=r"a\.txt names the initial transform 'gone\.txt'"):
        rfe.read_parameter_files(tmp_path / "a.txt")
    (tmp_path / "b.txt").write_text(
        '(InitialTransformParameterFileName "c.txt")\n', encoding="utf-8"
    )
    (tmp_path / "c.txt").write_text(
        '(InitialTransformParameterFileName "b.txt")\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="named twice"):
        rfe.read_parameter_files(tmp_path / "b.txt")
    (tmp_path / "d.txt").write_text(
        '(InitialTransformParameterFileName "x" "y")\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="one file name"):
        rfe.read_parameter_files(tmp_path / "d.txt")
    with pytest.raises(FileNotFoundError):
        rfe.read_parameter_file(tmp_path / "missing.txt")


# --- Resampling -------------------------------------------------------------------------------


def test_an_imported_chain_resamples_the_moving_image_onto_the_fixed_grid() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    xr = pytest.importorskip("xarray")
    import xarrayrf.native  # noqa: F401  (registers the .rf accessor)

    fixed, moving = frames(LPS_MM)
    index = xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3)

    def framed(values: npt.NDArray[np.float64], frame: xrf.ReferenceFrame) -> Any:
        array = xr.DataArray(
            values,
            dims=("i", "j", "k"),
            coords={
                dim: (dim, np.arange(n), {"units": "1"})
                for dim, n in zip("ijk", values.shape, strict=True)
            },
        )
        binding = xrf.AffineTransform.from_matrix(
            source=index, target=frame, matrix=np.eye(3), translation=np.zeros(3)
        )
        return array.rf.frame(binding, dims=("i", "j", "k"))

    i, j, k = np.indices((10, 10, 10), dtype=float)
    moving_image = framed(1.0 + 2.0 * i + 3.0 * j + 5.0 * k, moving)
    fixed_image = framed(np.zeros((4, 4, 4)), fixed)
    shift = ["1.5", "0.5", "2.0"]
    imported = rfe.transform_from_parameter_maps(
        [stage(TransformParameters=shift)], fixed=fixed, moving=moving
    )
    moved = moving_image.rf.resample_to(fixed_image, transform=imported.transform)
    fi, fj, fk = np.indices((4, 4, 4), dtype=float)
    expected = 1.0 + 2.0 * (fi + 1.5) + 3.0 * (fj + 0.5) + 5.0 * (fk + 2.0)
    assert_allclose(moved.transpose("i", "j", "k").values, expected, atol=1e-10)
