"""ITK transform files, read in float64: text, MATLAB, HDF5 and NIfTI displacement fields.

Each reader returns stages in application order. ITK applies a composite's last-added member
first, so a composite's members are returned reversed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FORM_TOLERANCE = 1e-6
"""Largest sform/qform difference, relative to the voxel size, accepted as the same geometry."""

_TYPE = re.compile(r"^(?P<name>[A-Za-z0-9]+)_(?P<precision>double|float)_(?P<a>\d)_(?P<b>\d)$")


@dataclass(frozen=True, slots=True)
class LinearStage:
    """One parametric ITK transform: its type name, dimension and parameters."""

    name: str
    dimension: int
    parameters: npt.NDArray[np.float64]
    fixed_parameters: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FieldStage:
    """A displacement field in ITK physical (LPS) space, vectors indexed ``[i, j, (k,) c]``."""

    vectors: npt.NDArray[np.floating[Any]]
    origin: npt.NDArray[np.float64]
    spacing: npt.NDArray[np.float64]
    direction: npt.NDArray[np.float64]
    notes: tuple[tuple[str, str], ...] = ()

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[-1])


type Stage = LinearStage | FieldStage


def split_type(text: str, *, where: str) -> tuple[str, int]:
    """``AffineTransform_double_3_3`` -> ``("AffineTransform", 3)``."""
    match = _TYPE.match(text.strip())
    if match is None:
        raise ValueError(
            f"{where}: {text!r} is not an ITK transform type such as 'AffineTransform_double_3_3'"
        )
    if match["a"] != match["b"]:
        raise ValueError(f"{where}: {text!r} maps between different dimensions")
    return match["name"], int(match["a"])


def _numbers(text: str, *, where: str) -> npt.NDArray[np.float64]:
    try:
        values = np.array([float(word) for word in text.split()], dtype=np.float64)
    except ValueError:
        raise ValueError(f"{where}: expected numbers, got {text.strip()!r}") from None
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{where}: values must be finite")
    return values


def _field_stage(
    name: str,
    dimension: int,
    parameters: npt.NDArray[np.float64],
    fixed: npt.NDArray[np.float64],
    *,
    where: str,
) -> FieldStage:
    """ITK's DisplacementFieldTransform: fixed ``size, origin, spacing, direction (row-major)``."""
    if fixed.size != 3 * dimension + dimension * dimension:
        raise ValueError(
            f"{where}: a {dimension}-D {name} needs {3 * dimension + dimension**2} fixed "
            f"parameters (size, origin, spacing, direction), got {fixed.size}"
        )
    size = fixed[:dimension]
    if np.any(size != np.round(size)) or np.any(size < 1):
        raise ValueError(
            f"{where}: the field size must hold positive integers, got {size.tolist()}"
        )
    shape = tuple(int(n) for n in size)
    if parameters.size != dimension * int(np.prod(shape)):
        raise ValueError(
            f"{where}: a field of size {shape} needs {dimension * int(np.prod(shape))} "
            f"parameters, got {parameters.size}"
        )
    # Interleaved vectors with the first index fastest.
    vectors = np.moveaxis(
        parameters.reshape(*reversed(shape), dimension), range(dimension), range(dimension)[::-1]
    )
    return FieldStage(
        vectors=np.ascontiguousarray(vectors),
        origin=fixed[dimension : 2 * dimension],
        spacing=fixed[2 * dimension : 3 * dimension],
        direction=fixed[3 * dimension :].reshape(dimension, dimension),
    )


def _stage(
    type_text: str,
    parameters: npt.NDArray[np.float64],
    fixed: npt.NDArray[np.float64],
    *,
    where: str,
) -> Stage:
    name, dimension = split_type(type_text, where=where)
    if name == "DisplacementFieldTransform":
        return _field_stage(name, dimension, parameters, fixed, where=where)
    return LinearStage(name, dimension, parameters, fixed)


def read_text(path: Path) -> list[Stage]:
    """An Insight Transform File: one transform, or a ``CompositeTransform`` and its members."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines or not lines[0].startswith("#Insight Transform File"):
        raise ValueError(
            f"{path}: not an Insight Transform File (no '#Insight Transform File' header)"
        )
    blocks: list[dict[str, str]] = []
    for number, line in enumerate(lines[1:], start=2):
        if line.startswith("#Transform"):
            blocks.append({})
            continue
        if line.startswith("#"):
            continue
        if not blocks or ":" not in line:
            raise ValueError(f"{path}:{number}: expected 'Key: value' inside a '#Transform' block")
        key, _, value = line.partition(":")
        if key in blocks[-1]:
            raise ValueError(f"{path}:{number}: {key!r} repeats in one transform")
        blocks[-1][key.strip()] = value
    if not blocks:
        raise ValueError(f"{path}: holds no transform")
    stages: list[Stage] = []
    for index, block in enumerate(blocks):
        where = f"{path} transform {index}"
        if "Transform" not in block:
            raise ValueError(f"{where}: no 'Transform:' line")
        if index == 0 and block["Transform"].strip().startswith("CompositeTransform"):
            continue
        for key in ("Parameters", "FixedParameters"):
            if key not in block:
                raise ValueError(f"{where}: no '{key}:' line")
        stages.append(
            _stage(
                block["Transform"],
                _numbers(block["Parameters"], where=where),
                _numbers(block["FixedParameters"], where=where),
                where=where,
            )
        )
    composite = blocks[0]["Transform"].strip().startswith("CompositeTransform")
    if not composite and len(stages) > 1:
        raise ValueError(
            f"{path}: holds {len(stages)} transforms without a CompositeTransform header; ITK reads "
            "that as a list whose order of application the file does not state"
        )
    if composite and not stages:
        raise ValueError(f"{path}: the composite holds no transforms")
    return stages[::-1] if composite else stages


def read_mat(path: Path) -> list[Stage]:
    """A MATLAB file as ITK and ANTs write it: one affine-family transform and ``fixed``."""
    from scipy.io import loadmat

    contents = loadmat(str(path))
    names = [key for key in contents if not key.startswith("__") and key != "fixed"]
    if len(names) != 1 or "fixed" not in contents:
        raise ValueError(
            f"{path}: expected one transform variable and 'fixed', got {sorted(contents)}"
        )
    where = str(path)
    name, dimension = split_type(names[0], where=where)
    parameters = np.asarray(contents[names[0]], dtype=np.float64).ravel()
    fixed = np.asarray(contents["fixed"], dtype=np.float64).ravel()
    if not (np.all(np.isfinite(parameters)) and np.all(np.isfinite(fixed))):
        raise ValueError(f"{where}: values must be finite")
    return [LinearStage(name, dimension, parameters, fixed)]


def read_hdf5(path: Path) -> list[Stage]:
    """ITK's HDF5 transform file: ``TransformGroup/0`` and, for a composite, its members."""
    import h5py

    with h5py.File(str(path), "r") as handle:
        if "TransformGroup" not in handle:
            raise ValueError(f"{path}: no TransformGroup; not an ITK transform file")
        group = handle["TransformGroup"]
        keys = sorted(group, key=lambda key: int(key) if key.isdigit() else -1)
        if not keys or not all(key.isdigit() for key in keys):
            raise ValueError(f"{path}: TransformGroup must hold numbered transforms, got {keys}")

        def text(member: object, key: str) -> str:
            value = np.asarray(member[key][()]).ravel()  # type: ignore[index]
            word = value[0]
            return word.decode() if isinstance(word, bytes) else str(word)

        def values(member: object, key: str, where: str) -> npt.NDArray[np.float64]:
            for spelling in (key, key.replace("Transform", "Tranform", 1)):
                if spelling in member:  # type: ignore[operator]
                    array = np.asarray(member[spelling][()], dtype=np.float64).ravel()  # type: ignore[index]
                    if not np.all(np.isfinite(array)):
                        raise ValueError(f"{where}: {key} must be finite")
                    return array
            raise ValueError(f"{where}: no {key}")

        first = text(group[keys[0]], "TransformType")
        composite = first.startswith("CompositeTransform")
        members = keys[1:] if composite else keys[:1]
        if not composite and len(keys) > 1:
            raise ValueError(f"{path}: several transforms without a CompositeTransform at 0")
        if composite and not members:
            raise ValueError(f"{path}: the composite holds no transforms")
        stages: list[Stage] = []
        for key in members:
            member = group[key]
            where = f"{path} TransformGroup/{key}"
            stages.append(
                _stage(
                    text(member, "TransformType"),
                    values(member, "TransformParameters", where),
                    values(member, "TransformFixedParameters", where),
                    where=where,
                )
            )
    return stages[::-1] if composite else stages


def read_nifti_field(path: Path) -> list[Stage]:
    """An ITK displacement field in NIfTI: ``(x, y, z, 1, 3)`` or ``(x, y, 1, 1, 2)``, LPS vectors.

    The header's geometry is RAS; it is converted to LPS so field and vectors share ITK's space.
    """
    import nibabel as nib

    # nibabel's stubs type load() as the base image; a NIfTI file gives a NIfTI image.
    image: Any = nib.load(str(path))
    shape = tuple(int(n) for n in image.shape)
    if len(shape) != 5 or shape[3] != 1 or shape[4] not in (2, 3):
        raise ValueError(
            f"{path}: an ITK displacement field is five-dimensional, (x, y, z, 1, D) with D = 2 or "
            f"3; got shape {shape}"
        )
    dimension = shape[4]
    if dimension == 2 and shape[2] != 1:
        raise ValueError(f"{path}: a 2-D field must have one sample along z, got {shape[2]}")
    header = image.header
    notes: list[tuple[str, str]] = []
    if header.get_intent()[0] != "vector":
        notes.append(
            (
                "nifti-intent",
                f"{path.name}: intent {header.get_intent()[0]!r} read as a vector field",
            )
        )
    unit = header.get_xyzt_units()[0]
    if unit == "unknown":
        notes.append(
            ("nifti-units", f"{path.name}: no spatial unit; read as millimetres, as ITK does")
        )
    elif unit != "mm":
        raise ValueError(
            f"{path}: the spatial unit is {unit!r}; ITK would rescale it to millimetres, so convert "
            "the field to millimetres first"
        )
    sform, sform_code = header.get_sform(coded=True)
    qform, qform_code = header.get_qform(coded=True)
    if sform is None and qform is None:
        raise ValueError(f"{path}: neither sform nor qform is coded, so the field has no geometry")
    affine = np.asarray(sform if sform is not None else qform, dtype=np.float64)
    if sform is not None and qform is not None:
        voxel = float(np.min(np.linalg.norm(affine[:3, :3], axis=0)))
        if np.max(np.abs(np.asarray(sform) - np.asarray(qform))) > FORM_TOLERANCE * voxel:
            raise ValueError(
                f"{path}: sform (code {int(sform_code)}) and qform (code {int(qform_code)}) disagree; "
                "ITK and nibabel would place the field differently"
            )
    # RAS header to LPS: negate the first two world rows.
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine
    if dimension == 2:
        if np.any(lps[:2, 2] != 0) or np.any(lps[2, :2] != 0):
            raise ValueError(f"{path}: a 2-D field's z axis must not mix with x and y")
        linear, origin = lps[:2, :2], lps[:2, 3]
    else:
        linear, origin = lps[:3, :3], lps[:3, 3]
    spacing = np.linalg.norm(linear, axis=0)
    data = np.asarray(image.dataobj)
    vectors = data[:, :, 0, 0, :] if dimension == 2 else data[:, :, :, 0, :]
    if vectors.dtype not in (np.float32, np.float64):
        vectors = vectors.astype(np.float64)
    return [
        FieldStage(
            vectors=np.ascontiguousarray(vectors),
            origin=origin,
            spacing=spacing,
            direction=linear / spacing,
            notes=tuple(notes),
        )
    ]


def read_stages(path: Path) -> list[Stage]:
    """Dispatch on the suffix; stages in application order."""
    suffixes = "".join(path.suffixes[-2:]).lower()
    if suffixes.endswith((".nii", ".nii.gz")):
        return read_nifti_field(path)
    suffix = path.suffix.lower()
    if suffix in (".tfm", ".txt"):
        return read_text(path)
    if suffix == ".mat":
        return read_mat(path)
    if suffix in (".h5", ".hdf5"):
        return read_hdf5(path)
    raise ValueError(f"{path}: unknown suffix; expected .tfm, .txt, .mat, .h5, .nii or .nii.gz")
