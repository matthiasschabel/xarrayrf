"""Frame adoption by assertion of shared identity and exact coordinate-system conversion."""

from __future__ import annotations

from ._affine import AffineTransform
from ._composite import compose
from ._frame import ReferenceFrame
from ._orientation import coordinate_system_change
from ._transform import SupportsAffine, SupportsPoints


def adopt_frame(transform: SupportsPoints, other: ReferenceFrame) -> AffineTransform:
    """Assert a shared world, composing its exact coordinate-system change when derivable."""
    if not isinstance(transform, SupportsAffine):
        raise ValueError("adopting a frame requires an affine coordinate transform")
    if not isinstance(transform.target, ReferenceFrame):
        raise ValueError("adopting a frame requires a ReferenceFrame target")
    affine = (
        transform
        if isinstance(transform, AffineTransform)
        else AffineTransform.from_matrix(
            source=transform.source,
            target=transform.target,
            matrix=transform.matrix,
            translation=transform.translation,
        )
    )
    view = other.with_coordinate_system(transform.target.coordinate_system)
    retargeted = affine.with_endpoints(target=view)
    if view.coordinate_system == other.coordinate_system:
        return retargeted.with_endpoints(target=other)
    try:
        change = coordinate_system_change(view, other)
    except ValueError as error:
        raise ValueError(f"cannot adopt frame: {error}") from error
    result = compose(retargeted, change)
    assert isinstance(result, AffineTransform)
    return result
