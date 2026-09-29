"""Optional native DataArray binding; importing this module registers ``.rf``."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from xarray.indexes import PandasIndex

from ._affine import AffineTransform
from ._binding import BindingIndex
from ._encoding import Decoder, MalformedDataError, encode
from ._encoding import decode as decode_value
from ._frame import ReferenceFrame
from ._geometry import Geometry
from ._resample import Method, resample
from ._sampling import Domain
from ._transform import SupportsAffine, SupportsPoints

_BINDING_ATTR = "xarrayrf_binding"
__all__ = ["CoordinateSpec", "DuckArray", "Report", "frame_dataarray", "index_coordinate"]


class DuckArray(Protocol):
    """Pixel storage xarray accepts as is: a NumPy or Dask array, never a list.

    Structural, like xarray's own check: ``ndim``, ``shape`` and ``dtype`` plus either the NumPy
    ``__array_function__``/``__array_ufunc__`` pair or ``__array_namespace__``.
    """

    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype[Any]: ...
    def __getitem__(self, key: Any, /) -> Any: ...


type CoordinateSpec = tuple[
    str | tuple[()], npt.NDArray[np.int64] | npt.NDArray[np.float64] | int, Mapping[str, str]
]
"""An xarray coordinate declaration ``(dims, values, attrs)`` as the adapters build them."""

type Report = tuple[tuple[str, str], ...]
"""Import decisions an adapter records as ``(code, message)`` pairs."""

_INDEX_UNITS: Mapping[str, str] = MappingProxyType({"units": "1"})


def index_coordinate(dim: str, size: int, *, start: int = 0) -> CoordinateSpec:
    """Declare a dimensionless integer index coordinate along ``dim``.

    Args:
        dim: The dimension name, also the coordinate name.
        size: Number of samples.
        start: First index value; ``1`` for one-based conventions such as GeoTIFF bands.
    """
    return (dim, np.arange(start, start + size, dtype=np.int64), _INDEX_UNITS)


def frame_dataarray(
    data: object,
    *,
    dims: Sequence[str],
    coords: Mapping[str, CoordinateSpec],
    transform: SupportsPoints,
    attrs: Mapping[str, Any] | None = None,
) -> xr.DataArray:
    """Bind caller-supplied pixels to imported geometry without evaluating them.

    The adapters' shared final step: check the pixels are a duck array of the shape the
    dimension coordinates declare, build the DataArray, and frame it on the dimensions the
    transform's source axes depend on.

    Args:
        data: NumPy or dask array in ``dims`` order.
        dims: Dimension names.
        coords: Coordinate declarations, at least one per dimension.
        transform: The array's coordinate transform into its frame.
        attrs: Optional attributes for the DataArray.

    Returns:
        A framed DataArray sharing the supplied pixels.

    Raises:
        TypeError: If ``data`` is not a duck array.
        ValueError: If its shape differs from the declared dimension sizes.
    """
    duck = _require_duck_array(data)
    sizes: dict[str, int] = {}
    for coordinate_dims, values, _ in coords.values():
        along = (coordinate_dims,) if isinstance(coordinate_dims, str) else coordinate_dims
        if len(along) == 1:
            sizes[along[0]] = len(cast(Sequence[Any], values))
    unsized = [dim for dim in dims if dim not in sizes]
    if unsized:
        raise ValueError(f"no coordinate declares the size of dimension(s) {unsized}")
    shape = tuple(sizes[dim] for dim in dims)
    if duck.shape != shape:
        raise ValueError(f"data shape {duck.shape} does not match geometry shape {shape}")
    array = xr.DataArray(duck, dims=tuple(dims), coords=coords, attrs=attrs)
    used = {dim for axis in transform.source.axes for dim in array.coords[axis].dims}
    frame_dims = tuple(dim for dim in dims if dim in used)
    return cast(xr.DataArray, array.rf.frame(transform, dims=frame_dims))


def _binding(array: xr.DataArray, *, verify: bool = True) -> BindingIndex | None:
    """Return the array's binding index, or None when unframed.

    With ``verify``, the index must still own every coordinate it claims: an operation such
    as ``expand_dims`` on a retained scalar can hand one of them to a default index, leaving
    an index that is neither absent nor a valid binding. That state raises so it cannot be
    read as a binding; ``rf.unframe()`` removes it.
    """
    owners = {
        name: index for name, index in array.xindexes.items() if isinstance(index, BindingIndex)
    }
    indexes = {id(index): index for index in owners.values()}
    if not indexes:
        return None
    if len(indexes) != 1:
        raise ValueError("array has multiple reference-frame bindings")
    index = next(iter(indexes.values()))
    claimed = set(index.axes) | set(index.fixed)
    if verify and claimed != set(owners):
        lost = sorted(map(str, claimed - set(owners)))
        raise ValueError(
            f"the binding no longer owns its source coordinate(s) {lost}; the array is neither "
            "framed nor unframed. Call rf.unframe() and frame it again"
        )
    return index


def _require_duck_array(data: object) -> DuckArray:
    """Require pixel input that xarray treats as a duck array (xarray's own rule, restated)."""
    array_like = isinstance(data, np.ndarray) or (
        all(hasattr(data, name) for name in ("ndim", "shape", "dtype"))
        and (
            (hasattr(data, "__array_function__") and hasattr(data, "__array_ufunc__"))
            or hasattr(data, "__array_namespace__")
        )
    )
    if not array_like:
        raise TypeError(
            f"data must be a duck array, got {type(data).__name__}; use np.asarray(data)"
        )
    return cast(DuckArray, data)


@xr.register_dataarray_accessor("rf")  # type: ignore[no-untyped-call]
class _ReferenceFrameAccessor:
    """Frame, inspect and persist a DataArray's reference-frame binding."""

    def __init__(self, array: xr.DataArray) -> None:
        self._array = array

    def _require_binding(self) -> BindingIndex:
        index = _binding(self._array)
        if index is None:
            raise ValueError("array is unframed; call rf.frame() first")
        return index

    def frame(self, coordinate_transform: SupportsPoints, *, dims: Sequence[str]) -> xr.DataArray:
        """Frame this array with a transform from its ArrayCoordinates to a ReferenceFrame.

        Args:
            coordinate_transform: Mapping from the array's coordinates (an
                ``ArrayCoordinates`` source) to its target ``ReferenceFrame``.
            dims: Geometry dimensions, in their intended order.

        Returns:
            A DataArray sharing the original pixel data and carrying a private binding index.

        Raises:
            TypeError: If geometry inputs have invalid types.
            ValueError: If the array is already framed, still carries an encoded binding
                attribute, or a source coordinate is invalid.
        """
        array = self._array
        if _binding(array) is not None:
            raise ValueError("array is already framed; call rf.unframe() before binding again")
        if _BINDING_ATTR in array.attrs:
            # A live binding is the only geometry authority; an encoded declaration left
            # beside it would be decoded later as if it were current.
            raise ValueError(
                f"array carries an encoded binding in attrs[{_BINDING_ATTR!r}]; call "
                "rf.decode() to restore it, or drop the attribute before framing"
            )
        geometry = Geometry(array, coordinate_transform, dims=dims)
        names = coordinate_transform.source.axes
        axes: dict[str, PandasIndex] = {}
        fixed: dict[str, xr.Variable] = {}
        for name, coordinate_dims in geometry.coordinate_dependencies.items():
            variable = array.coords[name].variable
            if len(coordinate_dims) == 1:
                axes[name] = PandasIndex.from_variables({name: variable}, options={})
            elif not coordinate_dims:
                fixed[name] = variable
            else:
                raise ValueError(f"source coordinate {name!r} must be 0-D or 1-D")
        shared = sorted(
            str(dim)
            for dim, count in Counter(index.dim for index in axes.values()).items()
            if count > 1
        )
        if shared:
            # Per-axis label indexes over one dimension would each write that dimension's
            # indexer, so a selection or alignment could satisfy one label and not the other.
            raise ValueError(
                f"source coordinates share dimension(s) {shared}; binding several "
                "coordinates that vary along one dimension is not supported"
            )
        index = BindingIndex(axes, fixed, coordinate_transform, geometry.dims)
        stripped = array.drop_indexes([name for name in names if name in array.xindexes])
        coords = xr.Coordinates(
            {name: stripped.coords[name].variable for name in names},
            indexes={name: index for name in names},
        )
        return stripped.assign_coords(coords)

    def decode(self, *, decoders: Mapping[str, Decoder] | None = None) -> xr.DataArray:
        """Restore a binding from this array's reserved JSON attribute.

        The decoded coordinate transform maps the array's coordinates (an
        ``ArrayCoordinates`` source) to its target ``ReferenceFrame``.

        Args:
            decoders: Decoders for user-defined transform kinds, passed to the value decoder.

        Returns:
            A DataArray with the attribute removed and its binding validated against current
            coordinates. Pixel data is shared with this array.

        Raises:
            ValueError: If the array is already framed or has no encoded binding.
            MalformedDataError: If the binding attribute is not valid JSON binding data.
            EncodingError: If decoding the transform fails.
        """
        array = self._array
        if _binding(array) is not None:
            raise ValueError("array is already framed")
        if _BINDING_ATTR not in array.attrs:
            raise ValueError(f"array has no {_BINDING_ATTR!r} attribute")
        encoded = array.attrs[_BINDING_ATTR]
        if not isinstance(encoded, str):
            raise MalformedDataError(f"{_BINDING_ATTR!r} must be JSON text")
        try:
            payload = json.loads(encoded)
        except ValueError as error:
            raise MalformedDataError(f"{_BINDING_ATTR!r} is not valid JSON: {error}") from error
        if not isinstance(payload, dict) or set(payload) != {"transform", "dims"}:
            raise MalformedDataError("encoded binding must have exactly 'transform' and 'dims'")
        dims = payload["dims"]
        if not isinstance(dims, list) or any(not isinstance(dim, str) for dim in dims):
            raise MalformedDataError("encoded binding 'dims' must be a JSON array of strings")
        coordinate_transform = decode_value(payload["transform"], decoders=decoders)
        if not isinstance(coordinate_transform, SupportsPoints):
            raise MalformedDataError("encoded binding 'transform' must be a point transform")
        plain = array.copy(deep=False)
        plain.attrs = {key: value for key, value in array.attrs.items() if key != _BINDING_ATTR}
        return cast(xr.DataArray, plain.rf.frame(coordinate_transform, dims=dims))

    @property
    def is_framed(self) -> bool:
        """Whether this array carries a reference-frame binding.

        Raises:
            ValueError: If a binding index has lost coordinates to another index.
        """
        return _binding(self._array) is not None

    @property
    def coordinate_transform(self) -> SupportsPoints:
        """The transform from the array's ArrayCoordinates to its target ReferenceFrame."""
        return self._require_binding().transform

    @property
    def reference_frame(self) -> ReferenceFrame:
        """The target reference frame of the current binding."""
        return cast(ReferenceFrame, self.coordinate_transform.target)

    @property
    def geometry_dims(self) -> tuple[str, ...]:
        """The dimensions locating samples in the bound frame."""
        return self._require_binding().dims

    @property
    def geometry(self) -> Geometry:
        """A fresh, validated view of this array's current geometry."""
        index = self._require_binding()
        return Geometry(self._array, index.transform, dims=index.dims)

    def resample_to(
        self,
        target: xr.DataArray | Geometry,
        *,
        transform: SupportsPoints | None = None,
        method: Method = "linear",
        fill_value: float = np.nan,
        domain: Domain = "samples",
    ) -> xr.DataArray:
        """Resample values onto a target geometry and bind them to its frame.

        Args:
            target: Framed array or geometry whose samples and binding define the result.
            transform: Mapping from the target frame to the source frame, if needed.
            method: ``"nearest"``, ``"linear"`` or ``"cubic"`` interpolation.
            fill_value: Value outside the source domain.
            domain: ``"samples"`` or ``"cells"``.

        Returns:
            Framed values with target geometry, source non-geometry coordinates, name and
            ordinary attributes. A reserved ``xarrayrf_binding`` attribute is not carried.

        Raises:
            TypeError: If target is neither a DataArray nor a Geometry.
            ValueError: If either array is unframed or core resampling rejects the geometry.
        """
        source = self.geometry
        if isinstance(target, xr.DataArray):
            target_geometry = target.rf.geometry
        elif isinstance(target, Geometry):
            target_geometry = target
        else:
            raise TypeError("target must be a framed DataArray or Geometry")
        result = resample(
            source,
            target_geometry,
            transform=transform,
            method=method,
            fill_value=fill_value,
            domain=domain,
        )
        result.name = self._array.name
        result.attrs = {k: v for k, v in self._array.attrs.items() if k != _BINDING_ATTR}
        return cast(
            xr.DataArray,
            result.rf.frame(target_geometry.transform, dims=target_geometry.dims),
        )

    def assume_frame(self, other: ReferenceFrame | xr.DataArray) -> xr.DataArray:
        """Adopt another frame's complete declaration without changing sample positions.

        Args:
            other: Reference frame or framed array that declares the intended identity.

        Returns:
            This array rebound to the other frame with its affine mapping unchanged.

        Raises:
            TypeError: If other is neither a frame nor a DataArray.
            ValueError: If an array is unframed, the systems differ, or this mapping is non-affine.
        """
        binding = self._require_binding()
        if isinstance(other, xr.DataArray):
            other_frame = other.rf.reference_frame
        elif isinstance(other, ReferenceFrame):
            other_frame = other
        else:
            raise TypeError("other must be a ReferenceFrame or framed DataArray")
        current = self.reference_frame
        if current.coordinate_system != other_frame.coordinate_system:
            raise ValueError("frames must have equal coordinate systems")
        transform = binding.transform
        if not isinstance(transform, SupportsAffine):
            raise ValueError("assume_frame requires an affine coordinate transform")
        affine = (
            transform
            if isinstance(transform, AffineTransform)
            else AffineTransform(
                source=transform.source,
                target=transform.target,
                matrix=transform.matrix,
                translation=transform.translation,
            )
        )
        replacement = affine.with_endpoints(target=other_frame)
        return cast(xr.DataArray, self.unframe().rf.frame(replacement, dims=binding.dims))

    def unframe(self) -> xr.DataArray:
        """Remove the binding and restore default indexes on dimension coordinates.

        Also clears a binding that has lost coordinates to another index, which every other
        accessor member refuses to read, and drops any encoded binding left in ``attrs``: the
        live binding was the authority, so a declaration beside it is stale.
        """
        index = _binding(self._array, verify=False)
        if index is None:
            raise ValueError("array is unframed; call rf.frame() first")
        names = index.transform.source.axes
        result = self._array.drop_indexes(list(names))
        if _BINDING_ATTR in result.attrs:
            result = result.copy(deep=False)
            result.attrs = {k: v for k, v in result.attrs.items() if k != _BINDING_ATTR}
        for name in names:
            if name in result.dims and result.coords[name].dims == (name,):
                result = result.set_xindex(name)
        return result

    def encode(self) -> xr.DataArray:
        """Return an unframed, pixel-sharing copy with a JSON binding attribute.

        The reserved ``xarrayrf_binding`` attribute replaces any existing value. Call
        ``rf.decode()`` after reading the array to restore and validate its binding.
        """
        index = self._require_binding()
        result = self.unframe().copy(deep=False)
        result.attrs = {
            **result.attrs,
            _BINDING_ATTR: json.dumps(
                {"transform": encode(index.transform), "dims": list(index.dims)},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        }
        return result
