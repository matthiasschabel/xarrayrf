"""An ANTs registration's files as a chain from the fixed frame to the moving frame."""

from __future__ import annotations

import os
from dataclasses import dataclass

from xarrayrf._affine import AffineTransform
from xarrayrf._coordinate_system import CoordinateSystem
from xarrayrf._declared_inverse import DeclaredInversePair
from xarrayrf._field_transform import DisplacementFieldTransform
from xarrayrf._frame import ReferenceFrame
from xarrayrf._transform import SupportsPoints
from xarrayrf._transform_import import joined, stage_frames
from xarrayrf.anatomy import LPS, patient_coordinate_system
from xarrayrf.itk import read_transform

type Report = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class AntsImport:
    """An imported ANTs registration.

    Attributes:
        transform: Fixed-frame points to moving-frame points: the warp, then the affine, as
            ``antsApplyTransforms`` applies ``[1Warp, 0GenericAffine]`` (the transform that
            resamples the moving image onto the fixed grid). With an inverse warp, the warp is
            the forward direction of a :class:`~xarrayrf.DeclaredInversePair`, so
            ``transform.declared_inverse()`` is ANTs' inverse.
        report: ``(code, message)`` for every normalization made.
    """

    transform: SupportsPoints
    report: Report


def _field(
    path: str | os.PathLike[str],
    source: ReferenceFrame,
    target: ReferenceFrame,
    space: CoordinateSystem,
    report: list[tuple[str, str]],
) -> DisplacementFieldTransform:
    imported = read_transform(path, fixed=source, moving=target, space=space)
    if not isinstance(imported.transform, DisplacementFieldTransform):
        raise ValueError(
            f"{path} is not a displacement field; an ANTs warp is a NIfTI vector image"
        )
    report.extend(entry for entry in imported.report if entry[0] != "stage")
    return imported.transform


def transform_from_registration(
    *,
    fixed: ReferenceFrame,
    moving: ReferenceFrame,
    affine: str | os.PathLike[str] | None = None,
    warp: str | os.PathLike[str] | None = None,
    inverse_warp: str | os.PathLike[str] | None = None,
    intermediate: ReferenceFrame | None = None,
    space: CoordinateSystem | None = None,
) -> AntsImport:
    """Import an ANTs registration's affine, warp and inverse warp.

    Args:
        fixed: The fixed image's frame; the result maps from it.
        moving: The moving image's frame; the result maps to it.
        affine: ``0GenericAffine.mat``, from the intermediate frame (or ``fixed`` without a warp)
            to ``moving``.
        warp: ``1Warp.nii.gz``, from ``fixed`` to the intermediate frame (or ``moving`` without an
            affine).
        inverse_warp: ``1InverseWarp.nii.gz``, ANTs' declared inverse of ``warp``.
        intermediate: The frame between the warp and the affine, in ``space``. Defaults to a
            fresh local frame; only meaningful with both a warp and an affine.
        space: The coordinate system of ANTs' physical points; defaults to the LPS patient system
            in millimetres (required for 2-D registrations).

    Returns:
        The chain and a report of every normalization.

    Raises:
        TypeError: If a frame or ``space`` has the wrong type.
        ValueError: If neither ``affine`` nor ``warp`` is given, ``inverse_warp`` comes without
            ``warp``, ``intermediate`` without both, a warp file is not a displacement field, the
            affine file is not an affine, or
            a file is malformed (see :func:`xarrayrf.itk.read_transform`).
    """
    if affine is None and warp is None:
        raise ValueError("an ANTs registration needs an affine, a warp or both")
    if inverse_warp is not None and warp is None:
        raise ValueError("inverse_warp is the inverse of warp, so warp is required with it")
    if intermediate is not None and (affine is None or warp is None):
        raise ValueError("intermediate lies between a warp and an affine; give both or neither")
    if space is None:
        space = patient_coordinate_system(LPS, "mm")
    stages = (warp is not None) + (affine is not None)
    frames = stage_frames(
        stages,
        fixed=fixed,
        moving=moving,
        space=space,
        intermediate_frames=None if intermediate is None else [intermediate],
    )
    report: list[tuple[str, str]] = []
    members: list[SupportsPoints] = []
    if warp is not None:
        forward = _field(warp, frames[0], frames[1], space, report)
        report.append(
            (
                "stage",
                f"warp {os.fspath(warp)} from {frames[0].identifier} to {frames[1].identifier}",
            )
        )
        if inverse_warp is not None:
            backward = _field(inverse_warp, frames[1], frames[0], space, report)
            members.append(DeclaredInversePair(forward, backward).forward)
            report.append(
                (
                    "declared-inverse",
                    f"{os.fspath(inverse_warp)} is ANTs' declared inverse of the warp, unverified",
                )
            )
        else:
            members.append(forward)
    if affine is not None:
        imported = read_transform(affine, fixed=frames[-2], moving=frames[-1], space=space)
        if not isinstance(imported.transform, AffineTransform):
            raise ValueError(
                f"{affine} is not an affine transform; an ANTs affine is 0GenericAffine.mat"
            )
        members.append(imported.transform)
        report.append(
            (
                "stage",
                f"affine {os.fspath(affine)} from {frames[-2].identifier} to {frames[-1].identifier}",
            )
        )
    transform, changes = joined(members, fixed=fixed, moving=moving, space=space, engine="ANTs")
    return AntsImport(transform=transform, report=(*report, *changes))
