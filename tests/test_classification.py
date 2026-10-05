"""Public affine classification and containment behavior."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

import xarrayrf as xrf


def transform(
    matrix: npt.NDArray[np.float64],
    translation: tuple[float, ...] | None = None,
    *,
    source_units: tuple[str, ...] | None = None,
    target_units: tuple[str, ...] | None = None,
) -> xrf.AffineTransform:
    rows, columns = matrix.shape
    return xrf.AffineTransform.from_matrix(
        source=xrf.ArrayCoordinates(
            tuple(f"a{i}" for i in range(columns)), source_units or ("mm",) * columns
        ),
        target=xrf.ArrayCoordinates(
            tuple(f"b{i}" for i in range(rows)), target_units or ("mm",) * rows
        ),
        matrix=matrix,
        translation=(0.0,) * rows if translation is None else translation,
    )


@pytest.mark.parametrize(
    ("matrix", "translation", "name"),
    [
        (np.eye(2), None, "identity"),
        (np.eye(3), (1.0, 0.0, 0.0), "translation"),
        (np.array([[0.0, -1.0], [1.0, 0.0]]), None, "rotation"),
        (np.array([[0.0, -1.0], [1.0, 0.0]]), (1.0, 0.0), "rigid"),
        (2.0 * np.eye(3), None, "similarity"),
        (np.diag([2.0, 3.0, 4.0]), None, "scaled_rigid"),
        (np.array([[1.0, 0.5], [0.0, 1.0]]), None, "affine"),
        (np.array([[1.0, 1.0], [0.0, 0.0]]), None, "singular"),
        (np.ones((3, 2)), None, "rectangular"),
    ],
)
def test_each_class(
    matrix: npt.NDArray[np.float64], translation: tuple[float, ...] | None, name: str
) -> None:
    result = xrf.affine_class(transform(matrix, translation))
    assert result.name == name
    assert result.same_units
    if name in {"affine", "singular", "rectangular"}:
        assert result.scales is None
        assert result.residual is None
    else:
        assert result.scales == pytest.approx(tuple(np.linalg.norm(matrix, axis=0)), rel=1e-12)
        assert result.residual is not None
        assert result.residual >= 0


def test_reflection_is_improper_but_still_orthogonal() -> None:
    result = xrf.affine_class(transform(np.diag([-1.0, 1.0, 1.0])))
    assert result.name == "rotation"
    assert result.proper is False
    assert result.at_most("rigid")
    with pytest.raises(FrozenInstanceError):
        result.proper = True  # type: ignore[misc]


@pytest.mark.parametrize("step", [1e-8, 1e-16])
def test_tiny_nonzero_orthogonal_steps_are_scaled_rigid(step: float) -> None:
    result = xrf.affine_class(transform(np.diag([step, 1.0, 1.0])))
    assert result.name == "scaled_rigid"
    assert result.scales == pytest.approx((step, 1.0, 1.0), rel=1e-12)
    assert result.proper is True


def test_rigid_determinant_constraint_precedes_identity() -> None:
    tolerance = 1e-4
    matrix = np.sqrt(1 + 0.8 * tolerance) * np.eye(3)
    result = xrf.affine_class(transform(matrix), tolerance=tolerance)
    assert result.name == "similarity"
    assert not result.at_most("rigid")
    assert result.residual == pytest.approx(0.0, abs=1e-12)


def test_tolerance_and_offset_boundaries() -> None:
    tolerance = 1e-3
    inside = np.sqrt(1 + 0.8 * tolerance) * np.eye(2)
    outside = np.sqrt(1 + 1.2 * tolerance) * np.eye(2)
    assert xrf.affine_class(transform(inside), tolerance=tolerance).name == "identity"
    assert xrf.affine_class(transform(outside), tolerance=tolerance).name == "similarity"
    assert xrf.affine_class(transform(np.eye(2), (1e-9, 0.0))).name == "identity"
    assert xrf.affine_class(transform(np.eye(2), (1.1e-9, 0.0))).name == "translation"


def test_orthogonality_and_uniformity_boundaries() -> None:
    tolerance = 1e-3
    # The 2-D normalized Gram matrix has off-diagonal residual sin(angle).
    within = np.array([[1.0, 0.8 * tolerance], [0.0, 1.0]])
    outside = np.array([[1.0, 1.2 * tolerance], [0.0, 1.0]])
    assert xrf.affine_class(transform(within), tolerance=tolerance).at_most("scaled_rigid")
    assert xrf.affine_class(transform(outside), tolerance=tolerance).name == "affine"
    within = np.diag([1 + 0.8 * tolerance, 1 - 0.8 * tolerance])
    outside = np.diag([1 + 1.2 * tolerance, 1 - 1.2 * tolerance])
    assert xrf.affine_class(transform(within), tolerance=tolerance).name == "similarity"
    assert xrf.affine_class(transform(outside), tolerance=tolerance).name == "scaled_rigid"


def test_lattice_units_spacing_and_shear() -> None:
    frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y", "z"), ("mm",) * 3))
    lattice = xrf.Lattice(
        frame=frame,
        dims=("i", "j", "k"),
        origin=np.zeros(3),
        matrix=np.diag([2.0, 3.0, 4.0]),
    )
    result = xrf.affine_class(lattice)
    assert result.name == "scaled_rigid"
    assert result.scales == pytest.approx((2.0, 3.0, 4.0), rel=1e-12)
    assert result.same_units is False
    sheared = xrf.Lattice(
        frame=frame,
        dims=("i", "j", "k"),
        origin=np.zeros(3),
        matrix=np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    assert xrf.affine_class(sheared).name == "affine"
    dimensionless = xrf.Lattice(
        frame=xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("1", "1"))),
        dims=("i", "j"),
        origin=np.zeros(2),
        matrix=np.eye(2),
    )
    assert xrf.affine_class(dimensionless).same_units is True


def test_same_units_describes_endpoints_not_physical_motion() -> None:
    beta = 0.5
    gamma = 1 / np.sqrt(1 - beta * beta)
    boost = gamma * np.array([[1.0, -beta], [-beta, 1.0]])
    result = xrf.affine_class(transform(boost, source_units=("m", "m"), target_units=("m", "m")))
    assert result.name == "affine"
    assert result.same_units is True
    assert xrf.affine_class(transform(np.eye(2), target_units=("m", "m"))).same_units is False


def test_partial_order_including_incomparable_and_outside_classes() -> None:
    identity = xrf.affine_class(transform(np.eye(2)))
    translation = xrf.affine_class(transform(np.eye(2), (1.0, 0.0)))
    rotation = xrf.affine_class(transform(np.diag([-1.0, 1.0])))
    singular = xrf.affine_class(transform(np.diag([0.0, 1.0])))
    rectangular = xrf.affine_class(transform(np.ones((3, 2))))
    assert identity.at_most("translation") and identity.at_most("rotation")
    assert translation.at_most("rigid") and not translation.at_most("rotation")
    assert rotation.at_most("rigid") and not rotation.at_most("translation")
    assert singular.at_most("singular") and not singular.at_most("affine")
    assert rectangular.at_most("rectangular") and not rectangular.at_most("affine")
    assert singular.proper is None and rectangular.proper is None
    with pytest.raises(ValueError, match="unknown affine class"):
        identity.at_most("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown affine class"):
        identity.at_most([])  # type: ignore[arg-type]


def test_invalid_input_and_thresholds() -> None:
    with pytest.raises(TypeError, match="SupportsAffine or be a Lattice"):
        xrf.affine_class(cast(xrf.AffineTransform, object()))
    for field in ("tolerance", "offset_tolerance"):
        for invalid in (-1.0, float("nan"), float("inf"), 10**400):
            kwargs = {field: invalid}
            with pytest.raises(ValueError, match=f"{field} must be finite and >= 0"):
                xrf.affine_class(transform(np.eye(2)), **kwargs)


def test_malformed_affine_coefficients_are_rejected() -> None:
    class WrongMatrix(xrf.AffineTransform):
        @property
        def matrix(self) -> npt.NDArray[np.float64]:
            return np.eye(3)

    class WrongTranslation(xrf.AffineTransform):
        @property
        def translation(self) -> npt.NDArray[np.float64]:
            return np.zeros(3)

    valid = transform(np.eye(2))
    with pytest.raises(ValueError, match=r"affine\.matrix must have shape"):
        xrf.affine_class(
            WrongMatrix.from_matrix(
                source=valid.source,
                target=valid.target,
                matrix=np.eye(2),
                translation=np.zeros(2),
            )
        )
    with pytest.raises(ValueError, match="affine translation must have shape"):
        xrf.affine_class(
            WrongTranslation.from_matrix(
                source=valid.source,
                target=valid.target,
                matrix=np.eye(2),
                translation=np.zeros(2),
            )
        )


def test_tolerance_type_and_range_errors_are_distinct() -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        xrf.affine_class(transform(np.eye(3)), tolerance="loose")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be finite"):
        xrf.affine_class(transform(np.eye(3)), tolerance=-1.0)
