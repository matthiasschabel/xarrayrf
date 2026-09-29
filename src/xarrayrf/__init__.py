"""Experimental xarray reference-frame extension.

The core is NumPy-only and useful for arrays, meshes, point sets and tables alike:

- :class:`ReferenceFrame` names a particular frame and is expressed in a
  :class:`CoordinateSystem`, whose axes may be oriented with a :class:`DirectionVocabulary`.
- :class:`ArrayCoordinates` names an array's own coordinate values; it and a frame are the two
  kinds of transform endpoint.
- :class:`AffineTransform` maps points between endpoints and implements the :class:`Transform`
  capability protocols, which user-defined transforms can satisfy too.
- :func:`transform_named` and :func:`coordinate_system_change` evaluate and derive transforms;
  :meth:`CoordinateSystem.axis_codes` names a vector's direction.
- :func:`affine_class` classifies a constant affine transform or lattice by its matrix geometry.

:class:`Geometry` is the xarray integration: a read-through view answering where an
array's current samples sit, given a transform the caller supplies at each construction. It
attaches nothing, and an array it was built from gains no geometry that an xarray operation
could carry. The core never imports xarray; ``Geometry`` is available when xarray is.

Import :mod:`xarrayrf.native` to register the DataArray ``.rf`` accessor. Use
``array.rf.frame(coordinate_transform, dims=...)`` to attach a transform from the
array's ``ArrayCoordinates`` into its target ``ReferenceFrame``. Importing
:mod:`xarrayrf` alone leaves the accessor unregistered. Native-operation lifecycle
support remains under validation; see ``docs/design.md``.
"""

from typing import Any

from ._affine import INVERSE_CONDITION_LIMIT, AffineTransform
from ._array_coordinates import ArrayCoordinates
from ._classification import AffineClass, affine_class
from ._composite import CompositeTransform, compose
from ._coordinate_system import CARTESIAN, AxisCode, CoordinateSystem
from ._encoding import (
    SCHEMA_VERSION,
    DecoderResultError,
    EncodingError,
    MalformedDataError,
    MissingDecoderError,
    SupportsEncoding,
    UnknownKindError,
    UnsupportedVersionError,
    decode,
    encode,
)
from ._frame import LOCAL_NAMESPACE, ReferenceFrame, Role
from ._lattice import Lattice
from ._orientation import coordinate_system_change
from ._transform import (
    Endpoint,
    SupportsAffine,
    SupportsInverse,
    SupportsJacobian,
    SupportsPoints,
    Transform,
    check_transform,
    transform_named,
)
from ._vocabulary import DirectionVocabulary

_INTEGRATION = frozenset({"Domain", "Geometry", "Method", "resample"})

__all__ = [
    "CARTESIAN",
    "INVERSE_CONDITION_LIMIT",
    "LOCAL_NAMESPACE",
    "SCHEMA_VERSION",
    "AffineClass",
    "AffineTransform",
    "ArrayCoordinates",
    "AxisCode",
    "CompositeTransform",
    "CoordinateSystem",
    "DecoderResultError",
    "DirectionVocabulary",
    "EncodingError",
    "Endpoint",
    "Lattice",
    "MalformedDataError",
    "MissingDecoderError",
    "ReferenceFrame",
    "Role",
    "SupportsAffine",
    "SupportsEncoding",
    "SupportsInverse",
    "SupportsJacobian",
    "SupportsPoints",
    "Transform",
    "UnknownKindError",
    "UnsupportedVersionError",
    "affine_class",
    "check_transform",
    "compose",
    "coordinate_system_change",
    "decode",
    "encode",
    "transform_named",
]

try:
    import xarray as _xarray  # noqa: F401  # only probes availability for the integration layer
except ImportError:
    pass
else:
    from ._geometry import Geometry
    from ._resample import Method, resample
    from ._sampling import Domain

    # Listed only when importable, so a star import still works without xarray.
    __all__ += ["Domain", "Geometry", "Method", "resample"]


def __getattr__(name: str) -> Any:
    """Explain a missing integration name instead of reporting a bare AttributeError."""
    if name in _INTEGRATION:
        raise ImportError(f"xarrayrf.{name} is part of the xarray integration; install xarray")
    raise AttributeError(f"module 'xarrayrf' has no attribute {name!r}")
