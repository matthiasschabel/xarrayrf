"""Numerical checks for the sampled asymmetric object and its clock."""

from __future__ import annotations

import pytest
from relativity_contraction_demo import analytic_proper_length, measure


@pytest.mark.parametrize("object_speed", [0.0, 0.25])
def test_fine_grid_linear_resampling_recovers_contraction_and_dilation(
    object_speed: float,
) -> None:
    pytest.importorskip("scipy", minversion="1.18")
    result = measure(0.05, "linear", object_speed=object_speed)
    # At h=0.05 m, the worst measured length error is 2.8e-4 m and the
    # worst tick error is 1.6e-6 m of ct; these limits allow modest platform drift.
    assert result.length_error == pytest.approx(0.0, abs=5e-4)
    assert result.tick_error == pytest.approx(0.0, abs=1e-5)
    assert result.proper_length == pytest.approx(analytic_proper_length(), abs=5e-4)
    assert result.proper_tick_interval == pytest.approx(4.0, abs=1e-5)


def test_moving_object_refines_from_nearest_to_cubic() -> None:
    pytest.importorskip("scipy", minversion="1.18")
    coarse_nearest = measure(0.2, "nearest", object_speed=0.25)
    coarse_linear = measure(0.2, "linear", object_speed=0.25)
    coarse_cubic = measure(0.2, "cubic", object_speed=0.25)
    fine_linear = measure(0.05, "linear", object_speed=0.25)
    # These are comparisons of measured physical observables, not samples or internals.
    assert abs(coarse_cubic.length_error) < abs(coarse_linear.length_error)
    assert abs(coarse_linear.length_error) < abs(coarse_nearest.length_error)
    assert abs(fine_linear.length_error) < abs(coarse_linear.length_error)
    assert abs(fine_linear.tick_error) < abs(coarse_linear.tick_error)
