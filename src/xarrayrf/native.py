"""Optional native DataArray binding; importing this module registers ``.rf``."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from ._binding import BindingIndex, binding_coordinates, grid_from_binding, grid_variables
from ._encoding import Decoder, MalformedDataError, decode_intervals, encode
from ._encoding import decode as decode_value
from ._frame import ReferenceFrame
from ._geometry import Geometry, adopt_frame, check_coordinate_unit
from ._grid import Coordinate, Grid
from ._resample import Method, _resample_with_intervals
from ._resampling import BOX_METHODS, Support, check_box_values, check_support
from ._sampling import Domain
from ._transform import SupportsPoints
from ._validation import check_names

_BINDING_ATTR = "xarrayrf_binding"
_DIMS_UNSET = object()
__all__ = [
    "CoordinateSpec",
    "DuckArray",
    "Report",
    "frame_array",
    "grid_coordinates",
    "index_coordinate",
]


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


def _grid_and_coords(
    transform: SupportsPoints,
    coords: Mapping[str, CoordinateSpec],
    *,
    intervals: Mapping[str, npt.ArrayLike] | None = None,
) -> tuple[Grid, dict[str, CoordinateSpec]]:
    """Split adapter coordinate specs without discarding geometry metadata."""
    units = dict(zip(transform.source.axes, transform.source.units, strict=True))
    geometry: dict[str, Coordinate] = {}
    other: dict[str, CoordinateSpec] = {}
    for name, spec in coords.items():
        if name not in units:
            other[name] = spec
            continue
        dim, values, attrs = spec
        if attrs and attrs != {"units": units[name]}:
            raise ValueError(
                f"internal adapter contract: geometry coordinate {name!r} attrs must be empty "
                "or contain only the declared units"
            )
        geometry[name] = (dim, values) if isinstance(dim, str) else values
    return Grid(transform, geometry, intervals=intervals), other


def grid_coordinates(grid: Grid) -> xr.Coordinates:
    """Convert a Grid to coordinates carrying its native binding index.

    Assign these coordinates to an array with matching geometry dimensions and sizes.
    Declared source units become coordinate ``units`` attributes.

    Raises:
        TypeError: If grid is not a Grid.
    """
    if not isinstance(grid, Grid):
        raise TypeError(f"grid must be a Grid, got {type(grid).__name__}")
    return binding_coordinates(grid_variables(grid), grid.transform, grid.dims, grid.intervals)


def _assign_binding(array: xr.DataArray, coords: xr.Coordinates) -> xr.DataArray:
    """Assign a binding and canonicalize only the bound coordinate slots."""
    bound = array.assign_coords(coords)
    names = iter(coords)
    ordered_names = [next(names) if name in coords else name for name in bound.coords]
    # xarray's matching-index key uses coordinate order before calling index equality.
    variables = {name: bound.coords[name].variable for name in ordered_names}
    indexes = {name: bound.xindexes[name] for name in ordered_names if name in bound.xindexes}
    ordered = xr.Coordinates(variables, indexes=indexes)
    result = xr.DataArray(bound.variable, coords=ordered, name=bound.name)
    result.encoding = dict(bound.encoding)
    return result


def frame_array(
    data: object,
    grid: Grid,
    *,
    dims: Sequence[str] | None = None,
    coords: Mapping[str, CoordinateSpec] | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> xr.DataArray:
    """Frame duck-array pixels on a Grid without evaluating them.

    ``dims`` names every pixel dimension (default: grid dimensions). ``coords`` declares
    non-geometry coordinates; geometry coordinates and their source units come from the grid.
    Every dimension needs a coordinate declaring its size.

    Raises:
        TypeError: If grid, data, dims or coords has an invalid type.
        ValueError: If coords redefines a grid coordinate, dims omits a grid dimension,
            a dimension has no declared size, a coordinate has the wrong length along a grid
            dimension, or data has the wrong shape.
    """
    if not isinstance(grid, Grid):
        raise TypeError(f"grid must be a Grid, got {type(grid).__name__}")
    names = grid.dims if dims is None else check_names(dims, field="dims", allow_empty=True)
    if coords is not None and not isinstance(coords, Mapping):
        raise TypeError("coords must be a mapping or None")
    extra = dict(coords or {})
    redefined = set(extra) & set(grid.coordinates)
    if redefined:
        raise ValueError(f"coords must not redefine grid coordinates {sorted(redefined)}")
    absent = set(grid.dims) - set(names)
    if absent:
        raise ValueError(f"dims must include grid dimensions {sorted(absent)}")
    duck = _require_duck_array(data)
    sizes = dict(grid.sizes)
    for name, (coordinate_dims, values, _) in extra.items():
        along = (coordinate_dims,) if isinstance(coordinate_dims, str) else coordinate_dims
        if len(along) == 1:
            size = len(cast(Sequence[Any], values))
            if along[0] in grid.sizes and size != grid.sizes[along[0]]:
                raise ValueError(
                    f"coordinate {name!r} along grid dimension {along[0]!r} has length {size}; "
                    f"expected {grid.sizes[along[0]]}"
                )
            sizes.setdefault(along[0], size)
    unsized = [dim for dim in names if dim not in sizes]
    if unsized:
        raise ValueError(f"no coordinate declares the size of dimension(s) {unsized}")
    shape = tuple(sizes[dim] for dim in names)
    if duck.shape != shape:
        raise ValueError(f"data shape {duck.shape} does not match geometry shape {shape}")
    # Place coordinates along array dimensions first, preserving the declared order of extras.
    declared: dict[str, xr.Variable | CoordinateSpec] = {**grid_variables(grid), **extra}

    def dims_of(name: str) -> tuple[str, ...]:
        entry = declared[name]
        if isinstance(entry, xr.Variable):
            return tuple(map(str, entry.dims))
        coordinate_dims = entry[0]
        return (coordinate_dims,) if isinstance(coordinate_dims, str) else tuple(coordinate_dims)

    ordered = [name for dim in names for name in declared if dims_of(name) == (dim,)]
    ordered += [name for name in declared if name not in ordered]
    extra_names = iter(extra)
    ordered = [next(extra_names) if name in extra else name for name in ordered]
    array = xr.DataArray(
        duck, dims=names, coords={name: declared[name] for name in ordered}, attrs=attrs
    )
    # Keep the accessor's validation, including refusal of a stale encoded declaration.
    return cast(xr.DataArray, array.rf.frame(grid))


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

    @overload
    def frame(
        self, coordinate_transform: Grid, *, replace_coordinates: bool = False
    ) -> xr.DataArray: ...

    @overload
    def frame(
        self,
        coordinate_transform: SupportsPoints,
        *,
        dims: Sequence[str],
        intervals: Mapping[str, npt.ArrayLike] | None = None,
        replace_coordinates: Literal[False] = False,
    ) -> xr.DataArray: ...

    def frame(
        self,
        coordinate_transform: SupportsPoints | Grid,
        *,
        dims: object = _DIMS_UNSET,
        intervals: Mapping[str, npt.ArrayLike] | None = None,
        replace_coordinates: bool = False,
    ) -> xr.DataArray:
        """Frame this array with a Grid or an ArrayCoordinates-to-ReferenceFrame transform.

        Args:
            coordinate_transform: Grid, or mapping from the array's coordinates (an
                ``ArrayCoordinates`` source) to its target ``ReferenceFrame``.
            dims: Required with a transform; forbidden with a Grid.
            intervals: Optional declared cell bounds for a transform; supplied by a Grid.
            replace_coordinates: With a Grid, replace conflicting source coordinates using
                its declarations. Matching coordinates retain their attrs; missing ones are
                supplied. This changes labels, never pixels, with no unit conversion or
                resampling. Must be False with a transform.

        Returns:
            A DataArray sharing the original pixel data and carrying a private binding index.

        Raises:
            TypeError: If geometry inputs have invalid types.
            ValueError: If the array is already framed, still carries an encoded binding
                attribute, a source coordinate is invalid or conflicts with the Grid without
                replacement opt-in, or replacement is requested with a transform.
        """
        if not isinstance(replace_coordinates, bool):
            raise TypeError("replace_coordinates must be a bool")
        if replace_coordinates and not isinstance(coordinate_transform, Grid):
            raise ValueError(
                "replace_coordinates=True requires a Grid; a transform cannot replace coordinates"
            )
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
        if isinstance(coordinate_transform, Grid):
            if dims is not _DIMS_UNSET:
                raise TypeError("dims must not be supplied with a Grid")
            if intervals is not None:
                raise TypeError("intervals must not be supplied with a Grid")
            grid = coordinate_transform
            for dim, size in grid.sizes.items():
                if dim not in array.dims or array.sizes[dim] != size:
                    raise ValueError(
                        f"grid dimension {dim!r} must be an array dimension of size {size}"
                    )
            coords = grid_coordinates(grid)
            variables = {str(name): variable for name, variable in coords.variables.items()}
            conflicts: dict[str, str] = {}
            units = dict(zip(grid.transform.source.axes, grid.transform.source.units, strict=True))
            for name, variable in variables.items():
                if name in array.coords:
                    current = array.coords[name].variable
                    reason = ""
                    try:
                        check_coordinate_unit(current, name=name, declared=units[name])
                    except TypeError as error:
                        if not replace_coordinates:
                            raise TypeError(
                                f"{error}; use replace_coordinates=True to use the Grid declaration"
                            ) from error
                        reason = str(error)
                    except ValueError as error:
                        reason = str(error)
                    if not reason:
                        if current.dims != variable.dims:
                            reason = f"dims {current.dims!r} differ from {variable.dims!r}"
                        elif current.dtype.kind != variable.dtype.kind:
                            reason = (
                                f"dtype kind {current.dtype.kind!r} differs from "
                                f"{variable.dtype.kind!r}"
                            )
                        elif not np.array_equal(current.data, variable.data):
                            reason = "values differ"
                    if reason:
                        conflicts[name] = reason
                    else:
                        variables[name] = current
            if conflicts and not replace_coordinates:
                details = "; ".join(f"{name!r}: {reason}" for name, reason in conflicts.items())
                raise ValueError(
                    f"source coordinates conflict with the Grid declarations ({details}); "
                    "use replace_coordinates=True to replace them"
                )
            Geometry(
                array.assign_coords(variables),
                grid.transform,
                dims=grid.dims,
                intervals=grid.intervals,
            )
            coords = binding_coordinates(variables, grid.transform, grid.dims, grid.intervals)
            names = grid.transform.source.axes
            stripped = array.drop_indexes([name for name in names if name in array.xindexes])
            return _assign_binding(stripped.drop_vars(list(conflicts)), coords)
        if dims is _DIMS_UNSET:
            raise TypeError("dims is required with a coordinate transform")
        geometry = Geometry(
            array, coordinate_transform, dims=cast(Sequence[str], dims), intervals=intervals
        )
        names = coordinate_transform.source.axes
        variables = {name: array.coords[name].variable for name in names}
        for name, variable in variables.items():
            if variable.ndim > 1:
                raise ValueError(f"source coordinate {name!r} must be 0-D or 1-D")
        shared = sorted(
            str(dim)
            for dim, count in Counter(
                variable.dims[0] for variable in variables.values() if variable.ndim == 1
            ).items()
            if count > 1
        )
        if shared:
            # Per-axis label indexes over one dimension would each write that dimension's
            # indexer, so a selection or alignment could satisfy one label and not the other.
            raise ValueError(
                f"source coordinates share dimension(s) {shared}; binding several "
                "coordinates that vary along one dimension is not supported"
            )
        coords = binding_coordinates(
            variables, coordinate_transform, geometry.dims, geometry.intervals
        )
        stripped = array.drop_indexes([name for name in names if name in array.xindexes])
        return _assign_binding(stripped, coords)

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
        if not isinstance(payload, dict) or set(payload) != {"transform", "dims", "intervals"}:
            raise MalformedDataError(
                "encoded binding must have exactly 'transform', 'dims' and 'intervals'"
            )
        dims = payload["dims"]
        if not isinstance(dims, list) or any(not isinstance(dim, str) for dim in dims):
            raise MalformedDataError("encoded binding 'dims' must be a JSON array of strings")
        coordinate_transform = decode_value(payload["transform"], decoders=decoders)
        if not isinstance(coordinate_transform, SupportsPoints):
            raise MalformedDataError("encoded binding 'transform' must be a point transform")
        plain = array.copy(deep=False)
        plain.attrs = {key: value for key, value in array.attrs.items() if key != _BINDING_ATTR}
        intervals = decode_intervals(payload["intervals"])
        try:
            return cast(
                xr.DataArray, plain.rf.frame(coordinate_transform, dims=dims, intervals=intervals)
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise MalformedDataError(
                f"encoded binding is not a valid declaration: {error}"
            ) from error

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
        return Geometry(self._array, index.transform, dims=index.dims, intervals=index.intervals)

    @property
    def grid(self) -> Grid:
        """A frozen coordinate snapshot retaining this array's transform by reference."""
        self._require_binding()
        return grid_from_binding(self._array.coords)

    @overload
    def resample_to(
        self,
        target: xr.DataArray | Geometry | Grid,
        *,
        transform: SupportsPoints | None = None,
        method: Method = "linear",
        fill_value: float = np.nan,
        domain: Domain = "samples",
        support: Support = "point",
        min_coverage: float = 0.5,
        return_coverage: Literal[False] = False,
    ) -> xr.DataArray: ...

    @overload
    def resample_to(
        self,
        target: xr.DataArray | Geometry | Grid,
        *,
        transform: SupportsPoints | None = None,
        method: Method = "linear",
        fill_value: float = np.nan,
        domain: Domain = "samples",
        support: Support = "point",
        min_coverage: float = 0.5,
        return_coverage: Literal[True],
    ) -> tuple[xr.DataArray, xr.DataArray]: ...

    @overload
    def resample_to(
        self,
        target: xr.DataArray | Geometry | Grid,
        *,
        transform: SupportsPoints | None = None,
        method: Method = "linear",
        fill_value: float = np.nan,
        domain: Domain = "samples",
        support: Support = "point",
        min_coverage: float = 0.5,
        return_coverage: bool,
    ) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]: ...

    def resample_to(
        self,
        target: xr.DataArray | Geometry | Grid,
        *,
        transform: SupportsPoints | None = None,
        method: Method = "linear",
        fill_value: float = np.nan,
        domain: Domain = "samples",
        support: Support = "point",
        min_coverage: float = 0.5,
        return_coverage: bool = False,
    ) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
        """Resample values onto a target geometry and bind them to its frame.

        Empty targets return empty framed values. Empty sources with non-empty targets raise
        ValueError because there is nothing to sample from.

        With a separable affine map, axes whose target samples coincide with source samples
        retain mapped source intervals only when valid under the target's sample_offset;
        average slab axes claim target intervals and point slab axes declare none. The target
        transform is unchanged.

        Box methods reconstruct the mean of covering slabs, treating each source slab value
        as a mean over its declared interval. ``step`` refuses overlapping source slabs.
        ``support="average"`` averages over covered target support; uncovered samples and
        coverage below ``min_coverage`` receive fill. These methods require floating or complex
        values, separable affine maps and source intervals on slab axes (also target intervals
        for averaging). Non-default ``domain`` refuses. Average support for nearest, linear
        and cubic is not yet supported. Defaults are unchanged.
        Unlike point interpolation, box methods assume nothing between slabs: targets in gaps
        are fill rather than bridged, and values reach the outer slab edges rather than
        stopping at the outer slab centres.

        Args:
            target: Grid, framed array or geometry whose samples define the result.
            transform: Mapping from the target frame to the source frame, if needed.
            method: ``"nearest"``, ``"linear"``, ``"cubic"``, ``"step"`` or ``"overlap_mean"``.
            fill_value: Value outside the source domain.
            domain: ``"samples"`` or ``"cells"``.
            support: ``"point"`` (default) or ``"average"`` over declared target intervals.
            min_coverage: Minimum covered fraction in ``(0, 1]``; default 0.5. Validated
                for every method, but used only by box methods.
            return_coverage: Return ``(values, coverage)`` for box methods. Coverage is a
                framed float64 array over target geometry dimensions only, bound to the target
                transform without intervals. Fractions multiply across axes; pass-through
                axes contribute 1 and point slab axes contribute 0 or 1.

        Returns:
            Framed values with target geometry, source non-geometry coordinates, name and
            ordinary attributes. A reserved ``xarrayrf_binding`` attribute is not carried.

        Raises:
            TypeError: If target is not a DataArray, Geometry or Grid, or a box method
                receives non-floating source values.
            ValueError: If either array is unframed or core resampling rejects the geometry.
        """
        check_support(method, support, min_coverage)
        check_box_values(method, self._array.dtype, domain)
        if return_coverage and method not in BOX_METHODS:
            raise ValueError("return_coverage requires step or overlap_mean")
        source = self.geometry
        if isinstance(target, xr.DataArray):
            target_geometry = target.rf.geometry
        elif isinstance(target, Geometry | Grid):
            target_geometry = target
        else:
            raise TypeError("target must be a framed DataArray, Geometry or Grid")
        result, intervals, coverage = _resample_with_intervals(
            source,
            target_geometry,
            transform=transform,
            method=method,
            fill_value=fill_value,
            domain=domain,
            support=support,
            min_coverage=min_coverage,
        )
        result.name = self._array.name
        result.attrs = {k: v for k, v in self._array.attrs.items() if k != _BINDING_ATTR}
        framed = cast(
            xr.DataArray,
            result.rf.frame(
                target_geometry.transform,
                dims=target_geometry.dims,
                intervals=intervals,
            ),
        )

        if return_coverage:
            assert coverage is not None
            return framed, cast(
                xr.DataArray,
                coverage.rf.frame(
                    target_geometry.transform,
                    dims=target_geometry.dims,
                ),
            )
        return framed

    def assume_frame(self, other: ReferenceFrame | xr.DataArray) -> xr.DataArray:
        """Adopt another frame's declaration and derivable coordinate-system change.

        Args:
            other: Reference frame or framed array that declares the intended identity.

        Returns:
            This array rebound to the other frame, preserving samples and grid checks.

        Raises:
            TypeError: If other is neither a frame nor a DataArray.
            ValueError: If an array is unframed, the system change is underivable, or this mapping is non-affine.
        """
        binding = self._require_binding()
        replacement = adopt_frame(binding.transform, other)
        return cast(
            xr.DataArray,
            self.unframe().rf.frame(replacement, dims=binding.dims, intervals=binding.intervals),
        )

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
                {
                    "transform": encode(index.transform),
                    "dims": list(index.dims),
                    "intervals": {name: rows.tolist() for name, rows in index.intervals.items()},
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        }
        return result
