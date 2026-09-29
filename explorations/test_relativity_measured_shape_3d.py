"""Exploratory measurements of a simultaneous slice, not optical appearance."""

from __future__ import annotations

import numpy as np
import pytest
from relativity_measured_shape_3d import (
    DIRECTION,
    GAMMA,
    analytic_rest_covariance,
    measure,
    motion_basis,
)


def test_oblique_simultaneous_slice_contracts_only_along_motion() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    fine = measure()
    coarse = measure(33)
    # Rest-domain tails and Gaussian quadrature at h=14/62 are negligible.
    np.testing.assert_allclose(
        fine.rest_covariance, analytic_rest_covariance(), rtol=0.0, atol=1e-9
    )
    contraction = np.eye(3) + (1 / GAMMA - 1) * np.outer(DIRECTION, DIRECTION)
    expected = contraction @ analytic_rest_covariance() @ contraction.T
    # Linear interpolation convolves well-sampled data with a triangular kernel:
    # variance h^2/6 = 0.00850 m^2 per rest axis here. Target quadrature adds
    # grid-phase error. Measured max tensor error is 0.00810 m^2; 0.009 allows
    # modest drift without hiding contraction in a transverse direction.
    np.testing.assert_allclose(fine.moving_covariance, expected, rtol=0.0, atol=0.009)
    basis = motion_basis()
    rest = basis @ fine.rest_covariance @ basis.T
    moving = basis @ fine.moving_covariance @ basis.T
    # Check the full transverse block and the mixed longitudinal moments, too.
    np.testing.assert_allclose(moving[1:, 1:], rest[1:, 1:], rtol=0.0, atol=0.009)
    np.testing.assert_allclose(moving[0, 1:], rest[0, 1:] / GAMMA, rtol=0.0, atol=0.001)
    # Observed RMS errors: 0.00564, 0.00546, 0.01007; empirical tolerances
    # for this field/grid, not universal interpolation bounds.
    np.testing.assert_allclose(fine.rms_ratios, [1 / GAMMA, 1.0, 1.0], rtol=0.0, atol=0.012)
    fine_error = np.max(np.abs(fine.moving_covariance - fine.predicted_covariance))
    coarse_error = np.max(np.abs(coarse.moving_covariance - coarse.predicted_covariance))
    # The measured error ratio is 0.266, close to the h^2 ratio (32/62)^2.
    assert fine_error < 0.3 * coarse_error
    assert fine.boundary_mass_fraction < 3e-7  # Observed 1.39e-7 on the outer target shell.


@pytest.mark.parametrize("samples", [16, 65])
def test_measure_rejects_out_of_range_sample_counts(samples: int) -> None:
    with pytest.raises(ValueError, match=r"spatial_samples must be in \[17, 64\]"):
        measure(samples)


def test_measure_rejects_noninteger_sample_counts() -> None:
    with pytest.raises(TypeError, match="spatial_samples must be an integer"):
        measure(33.5)  # type: ignore[arg-type]
