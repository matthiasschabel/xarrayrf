"""Exact lazy frame coordinates as an xarray ``CoordinateTransformIndex``.

Interoperability only: the xarrayrf core does not depend on this. It lets xarray, and the
plotting and viewer libraries that read coordinates, see where samples sit in a frame without
materializing a coordinate array, and select samples by frame point with ``method="nearest"``.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from xarray.indexes import CoordinateTransform, CoordinateTransformIndex

from ._sampling import AxisSampling, Domain, coordinate_to_position
from ._transform import SupportsAffine, SupportsInverse, check_transform


class FrameCoordinateTransform(CoordinateTransform):
    """Map sample positions to frame coordinates through the actual source coordinates.

    The transform and samplings are a snapshot of the geometry. Reverse mapping uses the
    same coordinate inversion and domain as :meth:`~xarrayrf.Geometry.positions_at`; a plane
    in a volume has no reverse mapping.
    """

    def __init__(
        self,
        transform: SupportsAffine,
        samplings: tuple[AxisSampling, ...],
        names: tuple[str, ...],
        sizes: Mapping[str, int],
        reach: tuple[tuple[float, float], ...],
        domain: Domain = "samples",
    ) -> None:
        super().__init__(names, sizes, dtype=np.float64)
        self.transform = transform
        self.samplings = tuple(
            replace(sampling, values=sampling.values.copy()) for sampling in samplings
        )
        # xarray renames self.dims without updating the source samplings.
        self._axis_dims = tuple(
            None if sampling.dim is None else self.dims.index(sampling.dim)
            for sampling in samplings
        )
        self.reach = reach
        self.domain = domain

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        """Map positions along each dim to frame coordinates, one array per frame axis."""
        columns = []
        for sampling, dim_index in zip(self.samplings, self._axis_dims, strict=True):
            if dim_index is None:
                columns.append(sampling.values)
                continue
            positions = np.asarray(dim_positions[self.dims[dim_index]])
            if positions.dtype.kind in "iu":
                columns.append(sampling.values[positions])
            else:
                columns.append(
                    np.interp(positions, np.arange(sampling.values.size), sampling.values)
                )
        values = np.stack(np.broadcast_arrays(*columns), axis=-1)
        points = values @ self.transform.matrix.T + self.transform.translation
        return {name: points[..., index] for index, name in enumerate(self.coord_names)}

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        """Map frame coordinates to fractional positions along each dim."""
        if len(self.dims) != len(self.coord_names) or None in self._axis_dims:
            raise ValueError(
                f"a geometry with {len(self.dims)} dims in a {len(self.coord_names)}-axis frame "
                "has no reverse mapping; a frame point off the geometry has no position",
            )
        if not isinstance(self.transform, SupportsInverse):
            raise TypeError(
                f"{type(self.transform).__name__} has no inverse, so points cannot be located"
            )
        broadcast = np.broadcast_arrays(
            *[np.asarray(coord_labels[name], dtype=np.float64) for name in self.coord_names]
        )
        points = np.stack(broadcast, axis=-1)
        hint = (
            ", or build the coordinates with domain='cells' to admit the outer samples' cells"
            if self.domain == "samples"
            else ""
        )
        outside_message = (
            f"frame points lie outside the {self.domain} domain (or are not finite), so no "
            f"sample there carries a value; select inside the geometry{hint}"
        )
        if not np.isfinite(points).all() or any(size == 0 for size in self.dim_size.values()):
            raise ValueError(outside_message)
        coordinates = check_transform(self.transform.inverse()).transform_point(points)
        result = {}
        for axis, (sampling, dim_index) in enumerate(
            zip(self.samplings, self._axis_dims, strict=True)
        ):
            assert dim_index is not None
            positions = coordinate_to_position(
                sampling.values, coordinates[..., axis], sampling.step, self.reach[dim_index]
            )
            if not np.isfinite(positions).all():
                raise ValueError(outside_message)
            # Admitted outer-cell positions must not round to a wrapping negative index.
            dim = self.dims[dim_index]
            result[dim] = np.clip(positions, 0.0, self.dim_size[dim] - 1)
        return result

    def equals(self, other: CoordinateTransform, **kwargs: Any) -> bool:
        """Compare frame, names, dimensions, domain, affine and sampling exactly.

        ``exclude`` is not honoured: each dimension's coordinates affect the frame points, so
        a partial comparison would be wrong. Aligning with an indexed dimension excluded is refused.
        """
        if not isinstance(other, FrameCoordinateTransform):
            return False
        return (
            self.transform.target == other.transform.target
            and self.coord_names == other.coord_names
            and self.dims == other.dims
            and self.dim_size == other.dim_size
            and self.reach == other.reach
            and self.domain == other.domain
            and self.transform == other.transform
            and self._axis_dims == other._axis_dims
            and all(
                np.array_equal(first.values, second.values)
                for first, second in zip(self.samplings, other.samplings, strict=True)
            )
        )

    def sliced(self, slices: Mapping[str, slice]) -> FrameCoordinateTransform:
        """Slice source coordinates, retaining the affine and reversing cell reach as needed."""
        samplings = []
        sizes = dict(self.dim_size)
        reach = list(self.reach)
        for sampling, dim_index in zip(self.samplings, self._axis_dims, strict=True):
            if dim_index is None:
                samplings.append(sampling)
                continue
            dim = self.dims[dim_index]
            selection = slices.get(dim, slice(None))
            _, _, step = selection.indices(sizes[dim])
            values = sampling.values[selection]
            sizes[dim] = values.size
            samplings.append(
                replace(
                    sampling,
                    dim=dim,
                    values=values,
                    step=sampling.step * step if sampling.step is not None else None,
                )
            )
            if step < 0:
                reach[dim_index] = (reach[dim_index][1], reach[dim_index][0])
        return FrameCoordinateTransform(
            self.transform,
            tuple(samplings),
            tuple(map(str, self.coord_names)),
            sizes,
            tuple(reach),
            self.domain,
        )


class FrameCoordinateIndex(CoordinateTransformIndex):
    """A ``CoordinateTransformIndex`` that survives slicing along its dims.

    Other selections (integer, fancy or vectorized) drop the index, as the base class does; the
    coordinate values already produced stay correct because they are computed from positions.
    Alignment is exact: two arrays with different frame coordinates do not align silently.
    """

    transform: FrameCoordinateTransform

    def isel(self, indexers: Mapping[Any, Any]) -> FrameCoordinateIndex | None:
        """Keep the index for slices along its dims; drop it for any other indexer."""
        relevant = {dim: indexer for dim, indexer in indexers.items() if dim in self.transform.dims}
        if not all(isinstance(indexer, slice) for indexer in relevant.values()):
            return None
        return FrameCoordinateIndex(self.transform.sliced(relevant))
