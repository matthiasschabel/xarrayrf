"""A read-through view of where an array's current samples sit in a reference frame."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from xarray.indexes import RangeIndex

from ._affine import AffineTransform
from ._array_coordinates import ArrayCoordinates
from ._coincidence import is_coincident
from ._composite import compose
from ._frame import ReferenceFrame
from ._grid import Grid
from ._lattice import Lattice
from ._orientation import coordinate_system_change
from ._positions import (
    Outside,
    check_position,
    check_tolerance,
    lattice,
    points_at,
    positions_at,
    transform_points,
)
from ._sampling import LATTICE_TOLERANCE, AxisSampling, Domain, Sampling, freeze_intervals
from ._transform import SupportsAffine, SupportsPoints, check_transform, transform_named
from ._validation import REAL_KINDS, _check_str_sequence, check_names, real_float_array

AXIS_DIM: Final = "axis"
"""Dimension that labels a point's axes in results returned by :class:`Geometry`."""

_SOURCE_AXIS: Final = "__xarrayrf_source_axis__"


def adopt_frame(
    transform: SupportsPoints, other: ReferenceFrame | xr.DataArray, *, name: str = "other"
) -> AffineTransform:
    """Assert a shared world, composing its exact coordinate-system change when derivable.

    ``name`` is the caller's parameter name, used in the type error.
    """
    if isinstance(other, xr.DataArray):
        other = other.rf.reference_frame
    if not isinstance(other, ReferenceFrame):
        raise TypeError(f"{name} must be a ReferenceFrame or framed DataArray")
    if not isinstance(transform, SupportsAffine):
        raise ValueError("adopting a frame requires an affine coordinate transform")
    if not isinstance(transform.target, ReferenceFrame):
        raise ValueError("adopting a frame requires a ReferenceFrame target")
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


def _sample_axes(array: xr.DataArray, axes: tuple[str, ...]) -> tuple[AxisSampling, ...]:
    """Read each source axis's coordinate values, refusing multidimensional coordinate fields.

    Raises:
        ValueError: If a coordinate depends on more than one dimension, or is not finite.
    """
    samplings = []
    for axis in axes:
        coordinate = array.coords[axis]
        if coordinate.ndim > 1:
            raise ValueError(
                f"coordinate {axis!r} depends on {coordinate.dims}; a multidimensional coordinate "
                "field has no per-dimension spacing or inverse lookup here",
            )
        values = real_float_array(np.asarray(coordinate.values), field=f"coordinate {axis!r}")
        dim = str(coordinate.dims[0]) if coordinate.ndim == 1 else None
        index = array.xindexes.get(axis)
        step = float(index.step) if isinstance(index, RangeIndex) and values.size > 1 else None
        samplings.append(AxisSampling(axis, dim, values, step))
    return tuple(samplings)


def _check_dims(value: object) -> tuple[str, ...]:
    """Validate an ordered, unique, possibly empty sequence of dimension names.

    Empty is meaningful: a fully selected point has no varying geometry dimension left.
    """
    return check_names(value, field="dims", allow_empty=True)


def check_coordinate_unit(
    coordinate: xr.DataArray | xr.Variable, *, name: str, declared: str | None
) -> None:
    """Require an explicit ``attrs['units']`` to agree exactly with the transform's token.

    An absent key is not a conflict: the transform's source already declares the axis's
    unit. A present key is a second statement about the same values, so it must say the
    same thing; for an axis declaring no unit, any ``units`` attribute contradicts it.
    Nothing is converted and no token is guessed from the values.
    """
    if "units" not in coordinate.attrs:
        return
    unit = coordinate.attrs["units"]
    if not isinstance(unit, str):
        raise TypeError(
            f"coordinate {name!r} has attrs['units'] of type {type(unit).__name__}; a unit "
            "declaration must be a string token",
        )
    if declared is None:
        raise ValueError(
            f"coordinate {name!r} declares attrs['units'] = {unit!r}, but the transform's source "
            "declares no unit for that axis; declare the unit in the source, or drop the "
            "attribute",
        )
    if unit != declared:
        raise ValueError(
            f"coordinate {name!r} declares attrs['units'] = {unit!r}, but the transform's source "
            f"declares that axis in {declared!r}; convert and relabel the values explicitly, "
            "because nothing here converts them",
        )


def _check_coordinate_dtype(coordinate: xr.DataArray, *, name: str) -> None:
    """Require a real integer or floating dtype, checked on metadata alone.

    The dtype is available without reading values, including for a chunked coordinate, so
    this costs nothing. It cannot recover what xarray already normalized away: a masked
    array whose mask xarray resolved into NaN on construction arrives here as an ordinary
    float coordinate, and this view does not pretend otherwise. A mask that does survive is
    refused when the selected value reaches the transform.
    """
    if coordinate.dtype.kind not in REAL_KINDS:
        raise TypeError(
            f"coordinate {name!r} must hold real integer or floating values, got dtype "
            f"{coordinate.dtype}; boolean, complex, string, object and datetime coordinates "
            "are not read as numbers",
        )


class Geometry:
    """Where an array's current samples sit, re-read from the array at every access.

    The array is the authoritative sample domain. This view holds a reference to it, never
    a copy of its shape, coordinate values or validity, so a caller who assigns new
    coordinates or edits a unit attribute sees the consequence on the next access instead
    of a remembered answer. Every geometry-dependent property and query revalidates.

    This is a query object, not an attachment. Constructing it binds nothing to the array
    and gives it no geometry an xarray operation could carry: a crop or a sum produces a
    plain DataArray, and a view built on the original says nothing about the result. A
    future ``.rf.geometry`` accessor can delegate here, but the snapshot ownership and
    operation-lifecycle enforcement it needs live outside this object and do not exist yet.
    Nothing here certifies what an array has been through.

    Pixels are never read, no dense mesh of points is built and no whole coordinate field is
    materialized. A chunked coordinate is validated on metadata at construction; only the
    individual values a point query selects are ever evaluated, so finiteness is a property
    of the selected values, not a promise about the field they came from.
    """

    __slots__ = ("_array", "_dims", "_frame", "_intervals", "_transform")

    _array: xr.DataArray
    _transform: SupportsPoints
    _frame: ReferenceFrame
    _dims: tuple[str, ...]

    def __init__(
        self,
        array: xr.DataArray,
        transform: SupportsPoints,
        *,
        dims: Sequence[str],
        intervals: Mapping[str, npt.ArrayLike] | None = None,
    ) -> None:
        """Validate that ``transform`` describes ``array``'s current geometry coordinates.

        Args:
            array: The array whose coordinates locate the samples. Kept by reference.
            transform: A point transform from :class:`~xarrayrf.ArrayCoordinates` naming
                the array's coordinates into a :class:`~xarrayrf.ReferenceFrame`, such as an
                :class:`~xarrayrf.AffineTransform`.
            intervals: Optional per-source-axis cell bounds, validated as on Grid.
                Declared coordinates are read to check agreement with their sample offsets.
            dims: The dimensions the source axes are allowed to depend on. They
                must exist on ``array``, be unique, and be exactly the dimensions some
                source axis actually depends on: a declared dimension no source axis uses is refused
                rather than silently treated as part of the geometry. Empty is valid for a fully
                selected point whose source axes are all retained scalars.

        A transform that is not a transform, or whose endpoints are not endpoints at all, is a
        ``TypeError``. A well-formed transform with the wrong kind of endpoint for this use, such
        as one between two frames, is a ``ValueError``.

        Raises:
            TypeError: If ``array``, ``transform`` or ``dims`` has the wrong Python
                type, a coordinate's dtype is not real integer or floating, or a
                coordinate's ``attrs['units']`` is not a string.
            ValueError: If ``dims`` repeats a name or names a dimension the array
                does not have, a source axis is not a coordinate of the array, a
                coordinate depends on a dimension outside ``dims``, a declared
                geometry dimension carries no source axis, or a coordinate's
                ``attrs['units']`` disagrees with the unit the transform's source declares,
                the transform's source is not array coordinates or its target is not a
                reference frame, or the transform fails :func:`~xarrayrf.check_transform`.
        """
        if not isinstance(array, xr.DataArray):
            raise TypeError(f"array must be an xarray.DataArray, got {type(array).__name__}")
        check_transform(transform)
        if not isinstance(transform.source, ArrayCoordinates):
            raise ValueError(
                "Geometry needs a transform from ArrayCoordinates, the array's own coordinate "
                f"values; this transform maps from {type(transform.source).__name__}",
            )
        if not isinstance(transform.target, ReferenceFrame):
            raise ValueError(
                "Geometry needs a transform into a ReferenceFrame; this transform maps into "
                f"{type(transform.target).__name__}",
            )
        self._dims = _check_dims(dims)
        self._array = array
        self._transform = transform
        self._frame = transform.target
        self._dependencies()
        if intervals is not None and not isinstance(intervals, Mapping):
            raise TypeError("intervals must be a mapping from source axis names to rows")
        self._intervals = freeze_intervals(
            transform.source,
            {name: array.coords[name].values for name in intervals or {} if name in array.coords},
            intervals,
        )

    @property
    def intervals(self) -> Mapping[str, npt.NDArray[np.float64]]:
        """Read-only declared support, keyed by source axis name."""
        return MappingProxyType({name: rows.view() for name, rows in self._intervals.items()})

    def _sample_axes(self) -> tuple[AxisSampling, ...]:
        axes = _sample_axes(self._array, self._transform.source.axes)
        assert isinstance(self._transform.source, ArrayCoordinates)
        intervals = freeze_intervals(
            self._transform.source, {axis.axis: axis.values for axis in axes}, self._intervals
        )
        return tuple(
            AxisSampling(axis.axis, axis.dim, axis.values, axis.step, intervals.get(axis.axis))
            for axis in axes
        )

    def _dependencies(self) -> dict[str, tuple[str, ...]]:
        """Revalidate the current coordinate structure and report source-axis dependencies.

        This is the one structural check. It runs at construction and again at every
        geometry-dependent access, because the array it reads is the caller's and may have
        changed since the last one.
        """
        dimensions = set(self._array.dims)
        absent = tuple(dim for dim in self._dims if dim not in dimensions)
        if absent:
            raise ValueError(
                f"geometry dimensions {absent} are not dimensions of the array, which has "
                f"{tuple(self._array.dims)}",
            )
        dependencies: dict[str, tuple[str, ...]] = {}
        used: set[Hashable] = set()
        for name, declared in zip(
            self._transform.source.axes, self._transform.source.units, strict=True
        ):
            if name not in self._array.coords:
                raise ValueError(
                    f"source axis {name!r} is not a coordinate of the array, which carries "
                    f"{tuple(self._array.coords)}; a data variable is not a coordinate, and "
                    "nothing here creates the missing coordinate",
                )
            coordinate = self._array.coords[name]
            surplus = tuple(dim for dim in coordinate.dims if dim not in self._dims)
            if surplus:
                raise ValueError(
                    f"coordinate {name!r} depends on dimensions {surplus} outside the declared "
                    f"geometry dimensions {self._dims}; a source axis parameterized by "
                    "a non-geometry dimension such as time or channel is not supported",
                )
            check_coordinate_unit(coordinate, name=name, declared=declared)
            _check_coordinate_dtype(coordinate, name=name)
            # The surplus check above proves every dim is one of the validated str names.
            dependencies[name] = cast(tuple[str, ...], coordinate.dims)
            used.update(coordinate.dims)
        unused = tuple(dim for dim in self._dims if dim not in used)
        if unused:
            raise ValueError(
                f"geometry dimensions {unused} carry no source axis; listing a dimension in dims "
                "does not make it part of the geometry",
            )
        return dependencies

    @property
    def transform(self) -> SupportsPoints:
        """The transform this view evaluates. A declaration, not a measurement."""
        return self._transform

    @property
    def array(self) -> xr.DataArray:
        """The array whose samples this view locates, held by reference."""
        return self._array

    @property
    def frame(self) -> ReferenceFrame:
        """The reference frame the samples are located in: the transform's target."""
        return self._frame

    @property
    def dims(self) -> tuple[str, ...]:
        """The declared geometry dimensions, in the order given at construction."""
        return self._dims

    @property
    def sizes(self) -> Mapping[str, int]:
        """The array's current size along each geometry dimension.

        Derived afresh and returned read-only, so it cannot become a stale second opinion
        about a shape the array owns.

        Raises:
            TypeError: If a coordinate's dtype or unit attribute is unacceptable.
            ValueError: If the current coordinate structure no longer matches the transform.
        """
        self._dependencies()
        return MappingProxyType({dim: self._array.sizes[dim] for dim in self._dims})

    @property
    def coordinate_dependencies(self) -> Mapping[str, tuple[str, ...]]:
        """Each source axis's current coordinate dimensions, in the coordinate's order.

        An empty tuple means the input is a retained scalar, such as the fixed position of
        a selected plane. Derived afresh and returned read-only.

        Raises:
            TypeError: If a coordinate's dtype or unit attribute is unacceptable.
            ValueError: If the current coordinate structure no longer matches the transform.
        """
        return MappingProxyType(self._dependencies())

    def point_at(self, /, **indexers: int | np.integer[Any]) -> xr.DataArray:
        """Locate one sample in the target reference frame.

        The positions index the geometry coordinates, not the data values, and are resolved
        through the array's actual named coordinate values: a one-based label, a nonuniform
        physical offset and a multidimensional coordinate field are all read as they stand,
        with no spacing inferred and no origin assumed. A retained scalar source axis, such as
        the fixed offset of a selected plane, is read from the array and needs no position.

        Only the selected coordinate values are evaluated. For a chunked coordinate that
        means the requested chunks and nothing else; pixel data is never touched.

        This is the forward direction only. There is no inverse, no nearest-sample lookup,
        no fractional position and no interpolation between samples.

        Args:
            **indexers: One zero-based, non-negative position per geometry dimension.
                Exactly the current geometry dimensions must be supplied. NumPy integers are
                accepted alongside Python integers; a boolean is not.

        Returns:
            The sample's point in the target frame: a one-dimensional ``DataArray`` over the
            dimension ``"axis"``, labelled by the frame's axis names, with a ``units``
            coordinate giving each axis's unit; an undeclared unit is missing there (xarray
            stores ``None`` as NaN).

        Raises:
            TypeError: If a position is not a Python or NumPy integer, if a position is a
                boolean, or if a coordinate's dtype or unit attribute is unacceptable.
            ValueError: If a geometry dimension is missing from ``indexers`` or an
                unexpected name is supplied, if the current coordinate structure no longer
                matches the transform, or if a selected coordinate value is not finite.
            IndexError: If a position is negative or not below the dimension's current
                size.
        """
        dependencies = self._dependencies()
        missing = tuple(dim for dim in self._dims if dim not in indexers)
        if missing:
            raise ValueError(
                f"missing positions for geometry dimensions {missing}; a single-sample query "
                f"must supply every varying geometry dimension {self._dims}",
            )
        unexpected = tuple(name for name in indexers if name not in self._dims)
        if unexpected:
            raise ValueError(
                f"unexpected positions {unexpected}; this geometry indexes only the geometry "
                f"dimensions {self._dims}",
            )
        positions = {
            dim: check_position(indexers[dim], dim=dim, size=self._array.sizes[dim])
            for dim in self._dims
        }
        values = {
            name: self._array.coords[name].isel({dim: positions[dim] for dim in dims}).data
            for name, dims in dependencies.items()
        }
        mapped = transform_named(self._transform, values)
        axes = self._frame.axes
        return xr.DataArray(
            np.array([float(mapped[axis]) for axis in axes]),
            dims=(AXIS_DIM,),
            coords={AXIS_DIM: list(axes), "units": (AXIS_DIM, list(self._frame.units))},
        )

    def _array_order(self) -> tuple[str, ...]:
        return tuple(str(dim) for dim in self._array.dims if dim in self._dims)

    def points(self) -> xr.DataArray:
        """Return every sample's point in the target frame.

        The result has the geometry dimensions in the array's order followed by ``"axis"``,
        labelled by the frame's axis names with a ``units`` coordinate, and carries the array's
        index coordinates along the geometry dimensions. Coordinates are broadcast and evaluated;
        pixels are never read. When the array is chunked with Dask the result is chunked the same
        way along the geometry dimensions and nothing is computed until it is requested.

        Returns:
            One point per sample of the geometry dimensions.

        Raises:
            TypeError: If a coordinate's dtype or unit attribute is unacceptable.
            ValueError: If the coordinate structure no longer matches the transform, or the
                transform returns points of the wrong shape or non-finite values.
        """
        self._dependencies()
        transform = self._transform
        frame_axes = self._frame.axes
        coordinates = [
            xr.DataArray(self._array.coords[axis].variable.data, dims=self._array.coords[axis].dims)
            for axis in transform.source.axes
        ]
        broadcast = list(xr.broadcast(*coordinates))
        order = [dim for dim in self._array_order() if dim in broadcast[0].dims]
        if self._array.chunks is not None:
            chunks = {dim: self._array.chunksizes[dim] for dim in order}
            broadcast = [coordinate.chunk(chunks) for coordinate in broadcast]
        stacked = xr.concat(broadcast, dim=_SOURCE_AXIS).transpose(*order, _SOURCE_AXIS)
        if self._array.chunks is not None:
            stacked = stacked.chunk({_SOURCE_AXIS: -1})

        def evaluate(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return transform_points(transform, values)

        result: xr.DataArray = xr.apply_ufunc(
            evaluate,
            stacked,
            input_core_dims=[[_SOURCE_AXIS]],
            output_core_dims=[[AXIS_DIM]],
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={"output_sizes": {AXIS_DIM: len(frame_axes)}},
        )
        index_coordinates = {
            dim: self._array.coords[dim].variable for dim in order if dim in self._array.xindexes
        }
        return result.assign_coords(
            {AXIS_DIM: list(frame_axes), "units": (AXIS_DIM, list(self._frame.units))}
        ).assign_coords(index_coordinates)

    def _sampling(self) -> Sampling:
        """Revalidate and read the per-axis coordinates needed by sampling queries."""
        self._dependencies()
        return Sampling(
            self._transform,
            self._dims,
            self.sizes,
            self._sample_axes,
        )

    def grid(self) -> Grid:
        """Snapshot the current 0-D/1-D coordinates as an immutable Grid.

        Raises:
            ValueError: If a coordinate is multidimensional or dimensions share source axes.
        """
        dependencies = self._dependencies()
        if any(len(dims) > 1 for dims in dependencies.values()):
            raise ValueError("multidimensional coordinates cannot be snapshotted as a Grid")
        coordinates = {
            name: (dims[0], self._array.coords[name].data)
            if dims
            else self._array.coords[name].data
            for name, dims in dependencies.items()
        }
        return Grid(self._transform, coordinates, intervals=self._intervals).transpose(*self._dims)

    def points_at(
        self,
        positions: npt.ArrayLike,
        *,
        domain: Domain = "samples",
        outside: Outside = "raise",
    ) -> npt.NDArray[np.float64]:
        """Map fractional positions (..., D), in dims order, to frame points (..., M).

        Coordinates interpolate piecewise linearly and extrapolate by the outer steps.
        Retained scalar axes are read from the array.

        Raises:
            TypeError: If positions or transformed points have a non-real dtype.
            ValueError: If positions have the wrong shape, domain or outside is unknown,
                coordinates are multidimensional, two axes share a dimension, an empty or
                single-sample axis has no step, a position is outside the chosen domain,
                or transformed points have the wrong shape or non-finite values.
        """
        return points_at(self._sampling(), positions, domain=domain, outside=outside)

    def lattice(
        self, dims: Sequence[str] | None = None, *, tolerance: float = LATTICE_TOLERANCE
    ) -> Lattice:
        """Return the regular lattice the samples form, when they form one.

        The samples form a lattice when the transform is affine and every source axis is either
        a retained scalar or a one-dimensional coordinate uniformly spaced along one geometry
        dimension. Coordinate steps are composed with the transform's matrix, so the lattice
        maps positions directly into the frame: origin, spacing and direction in ITK's sense,
        and the homogeneous voxel-to-frame matrix NIfTI and elastix use.

        Args:
            dims: The geometry dimensions in the order the lattice's columns should follow, such
                as ``("i", "j", "k")`` for ITK when the array is stored ``("k", "j", "i")``.
                Defaults to the array's own order. Must name every geometry dimension.
            tolerance: Largest deviation of a coordinate from uniform spacing, as a fraction of
                one step. Scanner positions often need a looser value than the default; nothing
                is snapped silently. Coordinates defined by an xarray ``RangeIndex`` are uniform
                by construction; their exact start and step are used without a test.

        Returns:
            The lattice.

        Raises:
            TypeError: If the transform is not affine, or ``dims`` is not a sequence of strings.
            ValueError: If ``dims`` does not name exactly the geometry dimensions, a coordinate is
                a multidimensional field, a dimension has a single sample (its step is not
                determined), a coordinate is not uniformly spaced within ``tolerance``, or a
                dimension's step maps to no displacement in the frame.
        """
        order = self._array_order() if dims is None else dims
        return lattice(self._sampling(), order, tolerance=tolerance)

    def frame_coordinates(
        self,
        names: Sequence[str] | None = None,
        *,
        domain: Domain = "samples",
    ) -> xr.Coordinates:
        """Return lazy xarray coordinates giving each sample's frame coordinates.

        The coordinates are backed by an xarray ``CoordinateTransformIndex`` using the exact
        affine and source coordinates, including nonuniform strictly monotonic coordinates.
        Values are computed on access and never stored. Assign them
        with ``array.assign_coords(geometry.frame_coordinates())`` to let xarray, plotting and
        viewer libraries read frame coordinates, and to select with
        ``.sel(x=..., y=..., z=..., method="nearest")`` using ``DataArray`` labels, as xarray's
        transform indexes require. Slicing keeps the index; other selections drop it while the
        values already produced stay correct. Two arrays whose frame coordinates differ do not
        align silently. A frame point outside the domain, or not finite, is refused rather than
        rounded to a wrong sample; the domain is the one :meth:`positions_at` uses, so
        ``domain="cells"`` admits points in the outer samples' cells as declared by each source
        axis's declared intervals or :attr:`~xarrayrf.ArrayCoordinates.sample_offset`,
        selecting the edge sample there.
        Reverse mapping agrees exactly with :meth:`positions_at`, with admitted outer-cell
        positions clipped to the edge sample before rounding.

        This is a snapshot of the current geometry and an interoperability aid, not an
        attachment: xarrayrf does not read these coordinates back.

        Args:
            names: Coordinate names, one per frame axis. Defaults to the frame's axis names;
                give others when those names are already used by the array, as when an NGFF
                array's dimensions are called ``z``, ``y``, ``x`` like its frame axes.
            domain: ``"samples"`` or ``"cells"``, as for :meth:`positions_at`.

        Returns:
            Coordinates over the geometry dimensions, with a ``units`` attribute on each axis
            that declares a unit.

        Raises:
            TypeError: If the transform is not affine, or ``names`` is not a sequence of
                strings.
            ValueError: If a source axis is a multidimensional field, more than one source
                axis varies along a dimension, coordinates are not strictly monotonic,
                ``domain`` is unknown, a cells-domain axis has a single sample without an interval, ``names`` has
                the wrong length or repeats a name, or a name is already a coordinate or
                dimension of the array. Retained scalars and single-sample dimensions support
                forward mapping; retained scalars have no reverse selection.
        """
        from ._frame_coordinates import FrameCoordinateIndex, FrameCoordinateTransform
        from ._positions import check_domain, reach
        from ._sampling import coordinate_to_position

        self._dependencies()
        transform = self._transform
        if not isinstance(transform, SupportsAffine):
            raise TypeError(
                f"frame_coordinates needs an affine transform; {type(transform).__name__} is not one"
            )
        check_domain(domain)
        source = transform.source
        assert isinstance(source, ArrayCoordinates)
        samplings = self._sample_axes()
        by_dim = {}
        for axis, sampling in enumerate(samplings):
            if sampling.dim is None:
                continue
            if sampling.dim in by_dim:
                raise ValueError(
                    f"dimension {sampling.dim!r} carries more than one source axis, so a "
                    "position is not determined by a single coordinate"
                )
            # Inverting the coordinates onto themselves raises now, not at the first sel(), if
            # they are not strictly monotonic.
            coordinate_to_position(sampling.values, sampling.values, sampling.step)
            by_dim[sampling.dim] = (sampling, axis, source.sample_offset[axis])
        chosen = self._frame.axes if names is None else _check_str_sequence(names, field="names")
        if len(chosen) != len(self._frame.axes) or len(set(chosen)) != len(chosen):
            raise ValueError(
                f"names must give one distinct name per frame axis {self._frame.axes}, got "
                f"{chosen}",
            )
        taken = sorted(
            set(chosen) & (set(map(str, self._array.coords)) | set(map(str, self._array.dims)))
        )
        if taken:
            raise ValueError(
                f"names {taken} are already coordinates or dimensions of the array; pass other "
                "names",
            )
        sizes = {dim: self._array.sizes[dim] for dim in self._array_order()}
        extents_by_dim = tuple(reach([by_dim[dim] for dim in sizes], domain))
        index = FrameCoordinateIndex(
            FrameCoordinateTransform(transform, samplings, chosen, sizes, extents_by_dim, domain)
        )
        coordinates = xr.Coordinates.from_xindex(index)
        for name, unit in zip(chosen, self._frame.units, strict=True):
            if unit is not None:
                coordinates.variables[name].attrs["units"] = unit
        return coordinates

    def positions_at(
        self,
        points: npt.ArrayLike,
        *,
        outside: Outside = "raise",
        domain: Domain = "samples",
    ) -> npt.NDArray[np.float64]:
        """Locate frame points among the samples as fractional positions.

        The transform is inverted exactly, giving source coordinates, and each coordinate is
        converted to a position along its dimension: by arithmetic for uniform coordinates and
        by monotonic interpolation for nonuniform ones such as DICOM slice offsets.

        Args:
            points: Frame points shaped ``(..., M)`` in the frame's axis order.
            outside: ``"raise"`` refuses a point outside the domain; ``"nan"`` returns NaN
                for every position of such a point, as resampling needs. "extrapolate" extends
                beyond the domain by the outer coordinate steps.
            domain: ``"samples"`` admits points between the outer samples, as xarray's
                ``interp`` does. ``"cells"`` also admits points in the outer samples' cells, as
                ITK does, using each source axis's declared
                :attr:`~xarrayrf.ArrayCoordinates.sample_offset`; a centred voxel reaches half
                a step beyond its sample, and a point-sampled axis no further than its samples. Positions there lie outside ``[0, n - 1]``, such as
                ``-0.5``, and are returned as they are.

        Returns:
            Positions shaped ``(..., D)``, ordered as :attr:`dims`.

        Raises:
            TypeError: If the transform has no inverse, or the points have a non-real dtype.
            ValueError: If the inverse is not determined (such as a plane embedded in a volume),
                a source axis is a retained scalar or a multidimensional field, a geometry
                dimension carries more than one source axis, coordinates are not strictly
                monotonic, ``domain`` is unknown, ``domain="cells"`` and a source axis declaring
                cells has a single sample without an interval, or ``outside="raise"`` and a point lies outside the
                domain.
        """
        return positions_at(self._sampling(), points, domain=domain, outside=outside)

    def is_coincident(self, other: Geometry, *, tolerance: float = LATTICE_TOLERANCE) -> bool:
        """Return whether both arrays sample the same points, element for element.

        When they do, one array's values stand for the other's without resampling. Frames must
        be the same frame: equal, or equivalent in another coordinate system (LPS and RAS),
        compared through the derived :func:`~xarrayrf.coordinate_system_change`; identity is
        never tolerant, so different frames are never coincident. Geometry dimensions must have
        the same names and sizes; elements are paired by dimension name, whatever each array's
        storage order.

        The tolerance is in steps, not coordinate units, as ITK's coordinate tolerance is a
        fraction of spacing: each sample of either array, located among the other's samples,
        must lie within ``tolerance`` of its own position along every dimension. A step is the
        local spacing, so nonuniform slice offsets and mixed-unit frames compare the same way.
        When every transform is affine the check is exact and costs time linear in the samples
        per dimension, not in the samples; otherwise every sample is checked, in blocks. A
        single-sample dimension has no step, so its coordinates must agree to rounding (1e-9
        relative); a tolerance there needs a declared cell width, which is designed but not
        implemented. This is a sample-location query: sample offsets, cells, non-geometry
        dimensions and values are not compared.

        Args:
            other: The geometry to compare with.
            tolerance: Largest deviation, as a fraction of a step, in ``[0, 0.5)``.

        Returns:
            Whether every sample coincides within ``tolerance``.

        Raises:
            TypeError: If ``other`` is not a Geometry, ``tolerance`` is not a real number, or a
                transform has no inverse.
            ValueError: If ``tolerance`` is outside ``[0, 0.5)``, the frames are equivalent but
                no coordinate-system change is derivable, or either geometry cannot locate
                points (a retained scalar or multidimensional source axis, or no determined
                inverse).
        """
        if not isinstance(other, Geometry):
            raise TypeError(f"other must be a Geometry, got {type(other).__name__}")
        return is_coincident(self._sampling(), other._sampling(), check_tolerance(tolerance))

    def __repr__(self) -> str:
        """Return a representation naming the target frame and the geometry dimensions."""
        return (
            f"Geometry(source={self._transform.source.axes!r}, "
            f"target={self._frame.identifier!r}, "
            f"dims={self._dims!r})"
        )
