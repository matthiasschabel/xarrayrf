"""Lorentz and Poincaré transformations as ordinary transforms between inertial frames.

The executable companion of ``docs/dev/architecture/relativity_notes.md``: nothing here is specific to
physics in the core; every step uses the same frames, transforms, composition and resampling as
medical and microscopy data. Units are natural (``ct`` in metres).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import xarrayrf as xrf

ATOL = 1e-12
"""Rounding allowance for products of small matrices with entries near one."""

SPACETIME = xrf.CoordinateSystem(("ct", "x", "y", "z"), ("m", "m", "m", "m"))
MINKOWSKI = np.diag([-1.0, 1.0, 1.0, 1.0])
"""The metric lives in user code: the core applies no metric to spacetime points."""


def inertial(name: str) -> xrf.ReferenceFrame:
    return xrf.ReferenceFrame.declared(("inertial", name), SPACETIME)


def boost_matrix(beta: float) -> npt.NDArray[np.float64]:
    """A boost along x by velocity ``beta`` (in units of c)."""
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    return np.array(
        [
            [gamma, -gamma * beta, 0.0, 0.0],
            [-gamma * beta, gamma, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def boost(
    source: xrf.ReferenceFrame, target: xrf.ReferenceFrame, beta: float
) -> xrf.AffineTransform:
    return xrf.AffineTransform.from_matrix(
        source=source, target=target, matrix=boost_matrix(beta), translation=np.zeros(4)
    )


def interval(events: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    result: npt.NDArray[np.float64] = np.einsum("...i,ij,...j->...", events, MINKOWSKI, events)
    return result


LAB, ROCKET, PROBE = inertial("lab"), inertial("rocket"), inertial("probe")


def test_a_boost_preserves_the_interval_and_inverts() -> None:
    lab_to_rocket = boost(LAB, ROCKET, 0.6)
    events = np.array([[3.0, 1.0, 0.0, 0.0], [5.0, -2.0, 1.0, 4.0]])
    moved = lab_to_rocket.transform_point(events)
    assert_allclose(interval(moved), interval(events), rtol=0, atol=ATOL)
    assert_allclose(lab_to_rocket.inverse().transform_point(moved), events, rtol=0, atol=ATOL)


def test_collinear_boosts_compose_by_relativistic_velocity_addition() -> None:
    """Composition is ordinary transform composition; velocities add as (a + b) / (1 + ab)."""
    chain = xrf.compose(boost(LAB, ROCKET, 0.6), boost(ROCKET, PROBE, 0.5))
    assert isinstance(chain, xrf.AffineTransform)
    assert chain.source == LAB
    assert chain.target == PROBE
    assert_allclose(chain.matrix, boost_matrix((0.6 + 0.5) / (1 + 0.6 * 0.5)), rtol=0, atol=ATOL)


def test_a_poincare_transformation_is_a_boost_with_a_translation() -> None:
    """Moving the origin event: the interval between two events is still preserved."""
    poincare = xrf.AffineTransform.from_matrix(
        source=LAB, target=ROCKET, matrix=boost_matrix(0.8), translation=(10.0, -3.0, 2.0, 1.0)
    )
    first, second = np.array([1.0, 0.5, 0.0, 0.0]), np.array([4.0, 2.0, 1.0, -1.0])
    separation = poincare.transform_point(second) - poincare.transform_point(first)
    assert_allclose(interval(separation), interval(second - first), rtol=0, atol=ATOL)


def test_four_vectors_and_covectors_follow_the_jacobian() -> None:
    """A 4-velocity is a vector (J u); a wave 4-vector k_mu is a covector (J^-T k)."""
    lab_to_rocket = boost(LAB, ROCKET, 0.6)
    at_rest = np.array([1.0, 0.0, 0.0, 0.0])
    assert_allclose(lab_to_rocket.jacobian() @ at_rest, [1.25, -0.75, 0.0, 0.0], atol=ATOL)
    phase_gradient = np.array([-2.0, 1.0, 0.0, 0.0])
    covector = np.linalg.inv(lab_to_rocket.jacobian()).T @ phase_gradient
    # The phase k_mu x^mu of an event is frame independent.
    event = np.array([3.0, 1.0, 0.5, 0.0])
    assert_allclose(
        covector @ lab_to_rocket.transform_point(event), phase_gradient @ event, rtol=0, atol=ATOL
    )


def test_a_spacetime_field_is_resampled_into_a_boosted_frame() -> None:
    """A field sampled on a t/z/y/x grid in the lab, seen on the rocket's grid.

    A field linear in the lab coordinates stays linear, so linear interpolation reproduces it
    exactly wherever the rocket's samples fall inside the lab's.
    """
    pytest.importorskip("scipy", minversion="1.18")
    size = 6
    axis = np.arange(size) * 0.5
    ct, z, y, x = np.meshgrid(axis, axis, axis, axis, indexing="ij")
    field = xr.DataArray(
        1.0 + 2.0 * ct + 3.0 * x - y + 0.5 * z,
        dims=("t", "z", "y", "x"),
        coords={"t": axis, "z": axis, "y": axis, "x": axis},
    )

    def locate(frame: xrf.ReferenceFrame) -> xrf.AffineTransform:
        return xrf.AffineTransform.from_matrix(
            source=xrf.ArrayCoordinates(("t", "x", "y", "z"), ("m",) * 4),
            target=frame,
            matrix=np.eye(4),
            translation=np.zeros(4),
        )

    dims = ("t", "z", "y", "x")
    lab = xrf.Geometry(field, locate(LAB), dims=dims)
    rocket = xrf.Geometry(xr.zeros_like(field), locate(ROCKET), dims=dims)
    rocket_to_lab = boost(LAB, ROCKET, 0.3).inverse()
    seen = xrf.resample(lab, rocket, transform=rocket_to_lab)
    events = rocket_to_lab.transform_point(rocket.points().values)
    expected = (
        1.0 + 2.0 * events[..., 0] + 3.0 * events[..., 1] - events[..., 2] + 0.5 * events[..., 3]
    )
    inside = ~np.isnan(seen.values)
    assert inside.any()
    assert_allclose(seen.values[inside], expected[inside], rtol=0, atol=1e-10)


def test_a_boost_in_si_units_inverts() -> None:
    """Seconds and metres make the matrix entries span 17 orders of magnitude; still invertible."""
    c = 299_792_458.0
    si = xrf.CoordinateSystem(("t", "x", "y", "z"), ("s", "m", "m", "m"))
    lab = xrf.ReferenceFrame.declared(("inertial", "lab-si"), si)
    rocket = xrf.ReferenceFrame.declared(("inertial", "rocket-si"), si)
    beta = 0.6
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    matrix = np.array(
        [
            [gamma, -gamma * beta / c, 0.0, 0.0],
            [-gamma * beta * c, gamma, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    lab_to_rocket = xrf.AffineTransform.from_matrix(
        source=lab, target=rocket, matrix=matrix, translation=np.zeros(4)
    )
    event = np.array([1e-8, 3.0, 0.0, 0.0])
    moved = lab_to_rocket.transform_point(event)
    assert_allclose(lab_to_rocket.inverse().transform_point(moved), event, rtol=1e-12, atol=0)
    scale = np.array([c, 1.0, 1.0, 1.0])
    assert_allclose(interval(moved * scale), interval(event * scale), rtol=1e-12, atol=0)
