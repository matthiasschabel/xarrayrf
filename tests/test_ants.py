"""The ANTs registration importer against SimpleITK, with its declared inverse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import xarrayrf as xrf
from xarrayrf.anatomy import LPS, RAS, patient_coordinate_system

pytest.importorskip("nibabel")
pytest.importorskip("scipy")
from xarrayrf import ants as rfa

DIRECTORY = Path(__file__).parent / "fixtures" / "itk"
REGISTRATION = json.loads((DIRECTORY / "oracle.json").read_text(encoding="utf-8"))["ants"]
FILES = {key: DIRECTORY / name for key, name in REGISTRATION["files"].items()}

NIFTI_ATOL = 1e-6
"""NIfTI geometry is float32 and ITK re-orthonormalizes the direction (see tests/test_itk.py)."""

ROUND_TRIP_LIMIT = 0.5
"""SyN's inverse warp is computed alongside the forward one and is not exact; on this small
registration its round trip is measured at about 0.07 mm, against displacements of millimetres."""

LPS_MM = patient_coordinate_system(LPS, "mm")
RAS_MM = patient_coordinate_system(RAS, "mm")
POINTS = np.asarray(REGISTRATION["points"])


def frames(system: xrf.CoordinateSystem = LPS_MM) -> tuple[xrf.ReferenceFrame, xrf.ReferenceFrame]:
    return (
        xrf.ReferenceFrame.declared(("test", "fixed"), system),
        xrf.ReferenceFrame.declared(("test", "moving"), system),
    )


def full(**options: object) -> rfa.AntsImport:
    fixed, moving = frames()
    return rfa.transform_from_registration(
        fixed=fixed,
        moving=moving,
        affine=FILES["affine"],
        warp=FILES["warp"],
        inverse_warp=FILES["inverse_warp"],
        **options,  # type: ignore[arg-type]
    )


def test_the_forward_chain_matches_simpleitk() -> None:
    result = full()
    mapped = result.transform.transform_point(POINTS)
    assert_allclose(mapped, REGISTRATION["forward"], atol=NIFTI_ATOL, rtol=0)
    assert isinstance(result.transform, xrf.CompositeTransform)
    warp, affine = result.transform.transforms
    assert isinstance(warp, xrf.PairedTransform)
    assert isinstance(warp.transform, xrf.DisplacementFieldTransform)
    assert isinstance(affine, xrf.AffineTransform)


def test_the_declared_inverse_is_ants_inverse_chain() -> None:
    result = full()
    assert isinstance(result.transform, xrf.CompositeTransform)
    inverse = result.transform.declared_inverse()
    mapped = inverse.transform_point(np.asarray(REGISTRATION["forward"]))
    assert_allclose(mapped, REGISTRATION["backward"], atol=NIFTI_ATOL, rtol=0)
    report = xrf.round_trip_residual_report(result.transform, POINTS)
    assert 0.0 < report.maximum < ROUND_TRIP_LIMIT
    with pytest.raises(TypeError, match="provide no inverse"):
        result.transform.inverse()


def test_without_an_inverse_warp_there_is_no_declared_inverse() -> None:
    fixed, moving = frames()
    result = rfa.transform_from_registration(
        fixed=fixed, moving=moving, affine=FILES["affine"], warp=FILES["warp"]
    )
    assert isinstance(result.transform, xrf.CompositeTransform)
    with pytest.raises(TypeError, match="neither an exact nor a declared inverse"):
        result.transform.declared_inverse()
    assert_allclose(
        result.transform.transform_point(POINTS), REGISTRATION["forward"], atol=NIFTI_ATOL, rtol=0
    )


def test_an_affine_or_a_warp_alone() -> None:
    fixed, moving = frames()
    affine = rfa.transform_from_registration(fixed=fixed, moving=moving, affine=FILES["affine"])
    assert isinstance(affine.transform, xrf.AffineTransform)
    assert (affine.transform.source, affine.transform.target) == (fixed, moving)
    warp = rfa.transform_from_registration(fixed=fixed, moving=moving, warp=FILES["warp"])
    assert isinstance(warp.transform, xrf.DisplacementFieldTransform)


def test_ras_frames_and_a_named_intermediate() -> None:
    fixed, moving = frames(RAS_MM)
    between = xrf.ReferenceFrame.declared(("test", "between"), LPS_MM)
    result = rfa.transform_from_registration(
        fixed=fixed,
        moving=moving,
        affine=FILES["affine"],
        warp=FILES["warp"],
        inverse_warp=FILES["inverse_warp"],
        intermediate=between,
    )
    flip = np.array([-1.0, -1.0, 1.0])
    mapped = result.transform.transform_point(POINTS * flip)
    assert_allclose(mapped, np.asarray(REGISTRATION["forward"]) * flip, atol=NIFTI_ATOL, rtol=0)
    assert isinstance(result.transform, xrf.CompositeTransform)
    assert result.transform.transforms[1].target == between
    codes = [code for code, _ in result.report]
    assert codes.count("coordinate-system-change") == 2 and "declared-inverse" in codes
    back = result.transform.declared_inverse()
    assert (back.source, back.target) == (moving, fixed)


def test_refusals() -> None:
    fixed, moving = frames()
    with pytest.raises(ValueError, match="needs an affine, a warp or both"):
        rfa.transform_from_registration(fixed=fixed, moving=moving)
    with pytest.raises(ValueError, match="warp is required"):
        rfa.transform_from_registration(
            fixed=fixed, moving=moving, affine=FILES["affine"], inverse_warp=FILES["inverse_warp"]
        )
    with pytest.raises(ValueError, match="give both or neither"):
        rfa.transform_from_registration(
            fixed=fixed, moving=moving, affine=FILES["affine"], intermediate=fixed
        )
    with pytest.raises(ValueError, match="not a displacement field"):
        rfa.transform_from_registration(fixed=fixed, moving=moving, warp=FILES["affine"])
    with pytest.raises(ValueError, match="not an affine transform"):
        rfa.transform_from_registration(fixed=fixed, moving=moving, affine=FILES["warp"])
