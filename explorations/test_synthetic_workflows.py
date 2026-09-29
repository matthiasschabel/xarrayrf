"""End-to-end workflows on synthetic arrays in local frames, with explicit ``Geometry`` pairing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import synthetic
import xarray as xr
from numpy.testing import assert_allclose

import xarrayrf as xrf


def test_each_kind_of_array_locates_its_samples() -> None:
    _, volume = synthetic.volume()
    assert_allclose(volume.lattice().spacing, [2.0, 0.5, 0.5])
    _, series = synthetic.time_series()
    assert_allclose(series.point_at(t=3, k=0, j=0, i=0).sel(axis="time").item(), 16.0)
    array, spectrum = synthetic.reciprocal()
    assert_allclose(
        spectrum.point_at(kx=0, ky=0, w=0).values, [array.kx[0], array.ky[0], array.w[0]]
    )


def test_a_rotated_volume_resamples_onto_another_frame_through_a_registration() -> None:
    """Target frame = source frame moved by a rigid registration; values come back unchanged."""
    pytest.importorskip("scipy", minversion="1.18")
    source_array, source = synthetic.volume(spacing=(1.0, 1.0, 1.0))
    moved = synthetic.patient_frame()
    rotation = synthetic.rotation_z(90.0)
    registration = xrf.AffineTransform(
        source=moved, target=source.frame, matrix=rotation.T, translation=np.zeros(3)
    )
    target_array, target = synthetic.volume(spacing=(1.0, 1.0, 1.0), rotation=rotation, frame=moved)
    result = xrf.resample(source, target, transform=registration, method="nearest")
    assert result.dims == target_array.dims
    assert_allclose(result.values, source_array.values, atol=1e-12)


def test_independently_paired_crops_coincide_and_shifted_ones_do_not() -> None:
    array, geometry = synthetic.volume()
    first = xrf.Geometry(array.isel(k=slice(2, 6)), geometry.transform, dims=geometry.dims)
    again, again_geometry = synthetic.volume(frame=geometry.frame)
    second = xrf.Geometry(again.isel(k=slice(2, 6)), again_geometry.transform, dims=geometry.dims)
    shifted = xrf.Geometry(again.isel(k=slice(3, 7)), again_geometry.transform, dims=geometry.dims)
    assert first.is_coincident(second)
    assert not first.is_coincident(shifted)


def test_netcdf_round_trip_restores_the_pairing_only_by_explicit_decode(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    array, geometry = synthetic.volume()
    stored = array.assign_attrs(xarrayrf_transform=json.dumps(xrf.encode(geometry.transform)))
    path = tmp_path / "volume.nc"
    stored.to_netcdf(path)
    with xr.open_dataarray(path) as opened:
        reopened = opened.load()
    transform = xrf.decode(json.loads(reopened.attrs["xarrayrf_transform"]))
    restored = xrf.Geometry(reopened, transform, dims=geometry.dims)
    assert transform == geometry.transform
    assert restored.is_coincident(geometry)
    assert_allclose(reopened.values, array.values)
