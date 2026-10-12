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

from ._array_coordinates import ArrayCoordinates
from ._binding import BindingIndex, binding_coordinates, grid_from_binding, grid_variables
from ._component_resampling import ComponentReorientation, component_request
from ._encoding import Decoder, MalformedDataError, decode_intervals, encode
from ._encoding import decode as decode_value
from ._frame import ReferenceFrame
from ._geometry import Geometry, adopt_frame, check_coordinate_unit
from ._grid import Coordinate, Grid
from ._resample import Method, _resample_with_intervals
from ._resampling import BOX_METHODS, Support, check_box_values, check_support
from ._sampling import Domain
from ._symmetric_packing import SymmetricPacking
from ._tensor_component_declaration import ComponentBasis, TensorComponentDeclaration
from ._tensor_components import (
    DECLARATION_ATTR,
    TensorComponentIndex,
    attach,
    change_basis,
    change_scalar_basis,
    declaration_of,
    declared_coordinates,
    detach,
    pack_components,
    refuse_component_geometry,
    stale_array_bases,
    unpack_components,
)
from ._tensor_type import TensorType
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


def _unframe(array: xr.DataArray, index: BindingIndex) -> xr.DataArray:
    names = index.transform.source.axes
    result = array.drop_indexes(list(names))
    if _BINDING_ATTR in result.attrs:
        result = result.copy(deep=False)
        result.attrs = {k: v for k, v in result.attrs.items() if k != _BINDING_ATTR}
    for name in names:
        if name in result.dims and result.coords[name].dims == (name,):
            result = result.set_xindex(name)
    return result


def _canonical_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decode_declaration(array: xr.DataArray) -> xr.DataArray:
    encoded = array.attrs[DECLARATION_ATTR]
    if not isinstance(encoded, str):
        raise MalformedDataError(f"{DECLARATION_ATTR!r} must be JSON text")
    try:
        payload = json.loads(encoded)
    except ValueError as error:
        raise MalformedDataError(f"{DECLARATION_ATTR!r} is not valid JSON: {error}") from error
    declaration = decode_value(payload)
    if not isinstance(declaration, TensorComponentDeclaration):
        raise MalformedDataError(
            f"{DECLARATION_ATTR!r} must hold a tensor component declaration, got "
            f"{type(declaration).__name__}"
        )
    plain = array.copy(deep=False)
    plain.attrs = {key: value for key, value in array.attrs.items() if key != DECLARATION_ATTR}
    try:
        return attach(plain, declared_coordinates(plain, declaration, _binding(plain)))
    except (TypeError, ValueError) as error:
        raise MalformedDataError(
            f"encoded tensor component declaration is not valid here: {error}"
        ) from error


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
            refuse_component_geometry(array, grid.dims)
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
        refuse_component_geometry(array, geometry.dims)
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
        """Restore a binding and a tensor component declaration from reserved JSON attributes.

        The decoded coordinate transform maps the array's coordinates (an
        ``ArrayCoordinates`` source) to its target ``ReferenceFrame``. The binding is restored
        first, so a declaration in the array's own coordinates validates against it.

        Args:
            decoders: Decoders for user-defined transform kinds, passed to the value decoder.

        Returns:
            A DataArray with the attributes removed and both declarations validated against
            current coordinates. Pixel data is shared with this array.

        Raises:
            ValueError: If the array is already framed or declared, or has neither encoded
                attribute.
            MalformedDataError: If an attribute is not valid JSON declaration data.
            EncodingError: If decoding a value fails.
        """
        array = self._array
        if _BINDING_ATTR not in array.attrs and DECLARATION_ATTR not in array.attrs:
            raise ValueError(
                f"array has no {_BINDING_ATTR!r} attribute and no {DECLARATION_ATTR!r} attribute"
            )
        if declaration_of(array) is not None:
            raise ValueError("array already declares tensor components")
        if _BINDING_ATTR in array.attrs:
            array = self._decode_binding(array, decoders)
        if DECLARATION_ATTR in array.attrs:
            array = _decode_declaration(array)
        return array

    def _decode_binding(
        self, array: xr.DataArray, decoders: Mapping[str, Decoder] | None
    ) -> xr.DataArray:
        if _binding(array) is not None:
            raise ValueError("array is already framed")
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
        output_component_basis: ComponentBasis | None = None,
        component_reorientation: ComponentReorientation | None = None,
        units: str | None = None,
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
        output_component_basis: ComponentBasis | None = None,
        component_reorientation: ComponentReorientation | None = None,
        units: str | None = None,
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
        output_component_basis: ComponentBasis | None = None,
        component_reorientation: ComponentReorientation | None = None,
        units: str | None = None,
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
        output_component_basis: ComponentBasis | None = None,
        component_reorientation: ComponentReorientation | None = None,
        units: str | None = None,
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
            output_component_basis: For declared tensor components: the result's basis, the
                target frame, an equivalent frame, or the target's ``ArrayCoordinates``.
            component_reorientation: For declared tensor components: ``"jacobian"``,
                ``"polar_orthogonal_factor"`` or a callable; see :func:`xarrayrf.resample`.
            units: For declared tensor components: the result's ``attrs["units"]``, required
                exactly when the source and output bases' units differ; it replaces the
                source's.

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
        request = component_request(
            self._array,
            output_component_basis=output_component_basis,
            component_reorientation=component_reorientation,
            units=units,
            fill_value=fill_value,
            framed_result=True,
        )
        result, intervals, coverage, declaration = _resample_with_intervals(
            source,
            target_geometry,
            transform=transform,
            method=method,
            fill_value=fill_value,
            domain=domain,
            support=support,
            min_coverage=min_coverage,
            components=request,
        )
        result.name = self._array.name
        result.attrs = {
            k: v for k, v in self._array.attrs.items() if k not in (_BINDING_ATTR, DECLARATION_ATTR)
        }
        framed = cast(
            xr.DataArray,
            result.rf.frame(
                target_geometry.transform,
                dims=target_geometry.dims,
                intervals=intervals,
            ),
        )
        if declaration is not None:
            framed = attach(framed, declared_coordinates(framed, declaration, _binding(framed)))
        if units is not None:
            framed.attrs["units"] = units  # after the source attrs, so the new unit wins

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
            _unframe(self._array, binding).rf.frame(
                replacement, dims=binding.dims, intervals=binding.intervals
            ),
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
        components = declaration_of(self._array, verify=False)
        if components is not None and any(
            isinstance(basis, ArrayCoordinates) for basis in components.declaration.bases
        ):
            raise ValueError(
                "tensor components are declared in this array's own coordinates, which need the "
                "binding; change them to a frame with rf.change_component_basis, or call "
                "rf.clear_tensor_components(), before unframing"
            )
        return _unframe(self._array, index)

    def encode(self) -> xr.DataArray:
        """Return an undeclared, pixel-sharing copy with JSON declaration attributes.

        The frame binding goes to the reserved ``xarrayrf_binding`` attribute and a tensor
        component declaration to ``xarrayrf_tensor_components``, each replacing any existing
        value. Call ``rf.decode()`` after reading the array to restore and validate both.

        Raises:
            ValueError: If the array has neither a binding nor a tensor component declaration.
        """
        index = _binding(self._array)
        components = declaration_of(self._array)
        if index is None and components is None:
            raise ValueError(
                "array is unframed and declares no tensor components; there is nothing to encode"
            )
        if components is not None and stale_array_bases(components, index):
            raise ValueError(
                "the tensor components are declared in ArrayCoordinates that are no longer this "
                "array's binding source (a source axis was renamed or the binding changed); "
                "call rf.clear_tensor_components() and declare them again"
            )
        result = self._array
        if components is not None:
            result = detach(result, components)
        if index is not None:
            result = _unframe(result, index)
        result = result.copy(deep=False)
        attrs = dict(result.attrs)
        if index is not None:
            attrs[_BINDING_ATTR] = _canonical_json(
                {
                    "transform": encode(index.transform),
                    "dims": list(index.dims),
                    "intervals": {name: rows.tolist() for name, rows in index.intervals.items()},
                }
            )
        if components is not None:
            attrs[DECLARATION_ATTR] = _canonical_json(encode(components.declaration))
        result.attrs = attrs
        return result

    @property
    def has_tensor_components(self) -> bool:
        """Whether this array declares tensor components.

        Raises:
            ValueError: If the declaration's index has lost coordinates to another index.
        """
        return declaration_of(self._array) is not None

    @property
    def tensor_component_declaration(self) -> TensorComponentDeclaration:
        """The tensor component declaration: law, component dimensions and bases."""
        return self._require_components().declaration

    def _require_components(self) -> TensorComponentIndex:
        index = declaration_of(self._array)
        if index is None:
            raise ValueError(
                "array declares no tensor components; call rf.declare_tensor_components() first"
            )
        return index

    def declare_tensor_components(
        self,
        tensor_type: TensorType,
        *,
        component_dims: Sequence[str],
        bases: ComponentBasis | Sequence[ComponentBasis],
        packing: SymmetricPacking | None = None,
    ) -> xr.DataArray:
        """Declare which dimensions hold this array's tensor components, and in which bases.

        The labels of each component dimension are its basis's axis names, in order; they are
        created when the dimension has no coordinate. A basis is a ``ReferenceFrame`` or this
        array's binding source ``ArrayCoordinates`` (components per unit of each coordinate
        value). Scalars (densities, pseudoscalars) have no component dimension; pass
        ``tensor_type=`` to ``rf.change_component_basis`` instead.

        The declaration is enforced on selection (a scalar, subset or permutation of
        components drops it, leaving plain labels), alignment and concatenation between
        declared operands (different declarations refuse), renaming, and every xarrayrf
        operation: ``resample`` and ``rf.resample_to`` refuse a declared source unless given
        ``output_component_basis`` and ``component_reorientation``. It is
        **advisory with respect to values**: element-wise operations that are not linear in
        the components (``v ** 2``, ``abs``, a component-by-component product, a mask varying
        along a component dimension, NaN-skipping reductions with components missing at
        different samples), ``roll`` and ``shift`` along a component dimension, and payload
        replacement leave it in place; an operand with the component dimension but no
        coordinate adopts it. Sums, differences and scalings of equal declarations are
        correct.

        A symmetric order-2 type can be declared packed: ``packing`` gives the order and
        off-diagonal scale of the ``n(n+1)/2`` components along one dimension, labelled
        ``axes[i] + axes[j]``. ``rf.change_component_basis`` unpacks, transforms and repacks;
        ``rf.unpack_symmetric_components`` and ``rf.pack_symmetric_components`` convert.

        Args:
            tensor_type: The transformation law; order at least 1.
            component_dims: One array dimension per component index, in index order; with
                ``packing``, the one dimension holding the packed components.
            bases: One basis per component index, or one for all of them.
            packing: The packing of a symmetric order-2 type, or ``None``.

        Returns:
            A DataArray sharing this array's data, with an index on the component
            coordinates.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If the array already declares components or carries an encoded
                declaration, the declaration is invalid, a component dimension is absent, a
                geometry dimension or of the wrong length, its labels are not the basis axes,
                or an ``ArrayCoordinates`` basis is not this array's binding source.

        Example:
            >>> import numpy as np
            >>> import xarray as xr
            >>> import xarrayrf as xrf
            >>> import xarrayrf.native
            >>> frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("m", "m")))
            >>> wind = xr.DataArray(np.array([[1.0, 2.0]]), dims=("time", "component"))
            >>> wind = wind.rf.declare_tensor_components(
            ...     xrf.VECTOR, component_dims=("component",), bases=frame
            ... )
            >>> wind.component.values.tolist()
            ['x', 'y']
        """
        array = self._array
        if declaration_of(array) is not None:
            raise ValueError(
                "array already declares tensor components; call rf.clear_tensor_components() first"
            )
        if DECLARATION_ATTR in array.attrs:
            raise ValueError(
                f"array carries an encoded declaration in attrs[{DECLARATION_ATTR!r}]; call "
                "rf.decode() to restore it, or drop the attribute before declaring"
            )
        declaration = TensorComponentDeclaration(
            tensor_type, component_dims, bases, packing=packing
        )
        return attach(array, declared_coordinates(array, declaration, _binding(array)))

    def unpack_symmetric_components(self, *, component_dims: Sequence[str]) -> xr.DataArray:
        """Expand packed symmetric components into two component dimensions.

        The two new dimensions take the packed dimension's place, in order, and are declared
        unpacked in the same bases and type. Coordinates on other dimensions, attrs and the
        frame binding are kept; coordinates along the packed dimension other than its labels
        are dropped.

        Args:
            component_dims: The names of the two new dimensions, neither an existing
                dimension or coordinate.

        Raises:
            ValueError: If the array declares no packed components or a name is taken.

        Example:
            >>> import numpy as np
            >>> import xarray as xr
            >>> import xarrayrf as xrf
            >>> import xarrayrf.native
            >>> frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x", "y"), ("m", "m")))
            >>> stress = xr.DataArray(np.array([1.0, 3.0, 2.0]), dims=("component",))
            >>> stress = stress.rf.declare_tensor_components(
            ...     xrf.SYMMETRIC_SECOND_ORDER_CONTRAVARIANT_TENSOR,
            ...     component_dims=("component",),
            ...     bases=frame,
            ...     packing=xrf.SymmetricPacking.voigt(2),
            ... )
            >>> stress.component.values.tolist()
            ['xx', 'yy', 'xy']
            >>> stress.rf.unpack_symmetric_components(component_dims=("row", "column")).values
            array([[1., 2.],
                   [2., 3.]])
        """
        index = self._require_components()
        return unpack_components(self._array, index, component_dims, binding=_binding(self._array))

    def pack_symmetric_components(
        self, packing: SymmetricPacking, *, component_dim: str
    ) -> xr.DataArray:
        """Pack the two component dimensions of a symmetric declaration into one.

        The packed dimension takes the place of the first of the two. Mirrored entries must
        agree as :meth:`SymmetricPacking.pack` requires; under Dask the check runs, and may
        refuse, at compute time. Coordinates on other dimensions, attrs and the frame binding
        are kept.

        Args:
            packing: The order and off-diagonal scale of the packed components.
            component_dim: The packed dimension's name: one of the two it replaces, or a name
                that is not a dimension or coordinate.

        Raises:
            TypeError: If ``packing`` is not a ``SymmetricPacking``.
            ValueError: If the declaration is not an unpacked symmetric one, the packing does
                not fit the basis, the name is taken, or a matrix is not symmetric.
        """
        index = self._require_components()
        return pack_components(
            self._array, index, packing, component_dim, binding=_binding(self._array)
        )

    def clear_tensor_components(self) -> xr.DataArray:
        """Remove the tensor component declaration; the labels stay as ordinary coordinates.

        Also clears a declaration whose index lost coordinates to another index, and drops an
        encoded declaration left in ``attrs``.
        """
        index = declaration_of(self._array, verify=False)
        if index is None:
            raise ValueError("array declares no tensor components")
        result = detach(self._array, index)
        if DECLARATION_ATTR in result.attrs:
            result = result.copy(deep=False)
            result.attrs = {k: v for k, v in result.attrs.items() if k != DECLARATION_ATTR}
        return result

    def change_component_basis(
        self,
        basis: ComponentBasis | Mapping[str, ComponentBasis],
        *,
        transform: SupportsPoints | None = None,
        units: str | None = None,
        tensor_type: TensorType | None = None,
        from_basis: ComponentBasis | None = None,
    ) -> xr.DataArray:
        """Express the components in another basis, at the same sample locations.

        Maps between bases: this array's binding source and its frame (the binding matrix and
        its inverse), one frame in two coordinate systems (``coordinate_system_change``), or
        ``transform`` between two frames, taking precedence when its endpoints match. Each map
        must be square and invertible. Only constant (affine) maps are supported. Labels
        become the new basis's axis names.

        Args:
            basis: The target basis for every slot, or a mapping from component dimension to
                basis.
            transform: An affine transform between the frames on the path, from the slot's
                side toward the target's.
            units: The new ``attrs["units"]``, required exactly when the basis units change.
            tensor_type: For an undeclared scalar only: its order-0 law (a density or a
                pseudoscalar).
            from_basis: For an undeclared scalar only: the basis it is expressed in.

        Returns:
            A new array; declared components stay declared in the new bases. Dask inputs stay
            lazy; a component dimension split across chunks raises xarray's core-dimension
            error.

        Raises:
            TypeError: If an argument has the wrong type or the values are not numeric.
            ValueError: If no map is derivable, a map is singular or not square, slots of a
                pseudotensor or density change by different maps, a symmetric declaration
                would end in mixed bases, ``units`` is missing or not needed, or the scalar
                arguments are given with a declaration or missing without one.
        """
        index = declaration_of(self._array)
        binding = _binding(self._array)
        if index is None:
            if tensor_type is None or from_basis is None:
                raise ValueError(
                    "array declares no tensor components; declare them, or pass tensor_type= "
                    "(order 0) and from_basis= for a scalar"
                )
            if not isinstance(tensor_type, TensorType) or tensor_type.order != 0:
                raise ValueError("tensor_type= here must be an order-0 TensorType")
            if isinstance(basis, Mapping):
                raise TypeError("a scalar takes one target basis, not a mapping")
            return change_scalar_basis(
                self._array,
                tensor_type,
                from_basis,
                basis,
                binding=binding,
                transform=transform,
                units=units,
            )
        if tensor_type is not None or from_basis is not None:
            raise ValueError(
                "tensor_type= and from_basis= are for undeclared scalars; this array declares "
                "its components"
            )
        dims = index.declaration.component_dims
        if isinstance(basis, Mapping):
            unknown = set(basis) - set(dims)
            if unknown:
                raise ValueError(f"{sorted(unknown)} are not component dimensions {dims}")
            # A packed dimension holds both slots, so its target sets both.
            slot_dims = dims * len(index.declaration.bases) if len(dims) == 1 else dims
            targets = tuple(
                basis.get(dim, current)
                for dim, current in zip(slot_dims, index.declaration.bases, strict=True)
            )
        else:
            targets = (basis,) * len(index.declaration.bases)
        return change_basis(
            self._array, index, targets, binding=binding, transform=transform, units=units
        )
