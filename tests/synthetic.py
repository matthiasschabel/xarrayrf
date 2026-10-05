"""Synthetic arrays in local frames, each returned with its explicit ``Geometry`` pairing.

Exploration only: generators for trying xarrayrf end to end without real data.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import xarray as xr

import xarrayrf as xrf

ANATOMY = xrf.DirectionVocabulary(
    "synthetic-anatomy", ("right-to-left", "anterior-to-posterior", "inferior-to-superior")
)
LPS = ("right-to-left", "anterior-to-posterior", "inferior-to-superior")


def patient_frame() -> xrf.ReferenceFrame:
    """A fresh local frame with LPS-oriented axes in millimetres."""
    return xrf.ReferenceFrame.local(
        xrf.CoordinateSystem(
            ("x", "y", "z"),
            ("mm",) * 3,
            axis_types=("space",) * 3,
            vocabulary=ANATOMY,
            orientation=LPS,
        ),
        role="world",
    )


def rotation_z(degrees: float) -> npt.NDArray[np.float64]:
    angle = np.deg2rad(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )


def volume(
    shape: tuple[int, int, int] = (8, 10, 12),
    spacing: tuple[float, float, float] = (0.5, 0.5, 2.0),
    rotation: npt.ArrayLike | None = None,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    frame: xrf.ReferenceFrame | None = None,
) -> tuple[xr.DataArray, xrf.Geometry]:
    """A ``(k, j, i)`` volume of a smooth field; ``spacing`` is along ``(i, j, k)``, as x, y, z."""
    frame = patient_frame() if frame is None else frame
    k, j, i = np.indices(shape, dtype=np.float64)
    array = xr.DataArray(
        np.sin(0.3 * i) + np.cos(0.2 * j) + 0.1 * k,
        dims=("k", "j", "i"),
        coords={"k": np.arange(shape[0]), "j": np.arange(shape[1]), "i": np.arange(shape[2])},
        name="signal",
    )
    matrix = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("i", "j", "k"), ("1",) * 3, sample_offset=(0.5,) * 3),
        target=frame,
        matrix=matrix @ np.diag(spacing),
        translation=origin,
    )
    return array, xrf.Geometry(array, transform, dims=("k", "j", "i"))


def time_series(
    shape: tuple[int, int, int, int] = (5, 4, 6, 7), interval: float = 2.0
) -> tuple[xr.DataArray, xrf.Geometry]:
    """A ``(t, k, j, i)`` series whose time axis is part of the geometry, in seconds."""
    frame = xrf.ReferenceFrame.local(
        xrf.CoordinateSystem(
            ("x", "y", "z", "time"),
            ("mm", "mm", "mm", "s"),
            axis_types=("space", "space", "space", "time"),
            vocabulary=ANATOMY,
            orientation=(*LPS, None),
        )
    )
    t, k, j, i = np.indices(shape, dtype=np.float64)
    array = xr.DataArray(
        np.cos(0.5 * t) * (i + j + k),
        dims=("t", "k", "j", "i"),
        coords={
            name: np.arange(size) for name, size in zip(("t", "k", "j", "i"), shape, strict=True)
        },
        name="bold",
    )
    matrix = np.zeros((4, 4))
    matrix[:3, :3] = np.diag([1.0, 1.0, 3.0])
    matrix[3, 3] = interval
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(
            ("i", "j", "k", "t"),
            ("1",) * 4,
            axis_types=("space", "space", "space", "time"),
            sample_offset=(0.5, 0.5, 0.5, None),
        ),
        target=frame,
        matrix=matrix,
        translation=(0.0, 0.0, 0.0, 10.0),
    )
    return array, xrf.Geometry(array, transform, dims=("t", "k", "j", "i"))


def reciprocal(shape: tuple[int, int, int] = (6, 6, 5)) -> tuple[xr.DataArray, xrf.Geometry]:
    """A ``(kx, ky, w)`` spectrum on centred frequency grids, in ``1/mm`` and ``rad/s``."""
    frame = xrf.ReferenceFrame.local(
        xrf.CoordinateSystem(
            ("kx", "ky", "omega"),
            ("1/mm", "1/mm", "rad/s"),
            axis_types=("spatial-frequency", "spatial-frequency", "angular-frequency"),
        )
    )
    coords = {
        "kx": ("kx", np.fft.fftshift(np.fft.fftfreq(shape[0], d=0.5))),
        "ky": ("ky", np.fft.fftshift(np.fft.fftfreq(shape[1], d=0.5))),
        "w": ("w", np.linspace(-np.pi, np.pi, shape[2])),
    }
    array = xr.DataArray(
        np.random.default_rng(0).random(shape), dims=("kx", "ky", "w"), coords=coords, name="power"
    )
    transform = xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(("kx", "ky", "w"), ("1/mm", "1/mm", "rad/s")),
        target=frame,
        matrix=np.eye(3),
        translation=np.zeros(3),
    )
    return array, xrf.Geometry(array, transform, dims=("kx", "ky", "w"))
