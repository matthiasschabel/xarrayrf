"""Conformance evidence for a rejected candidate, not production support claims."""

import dask.array as da
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from dask.callbacks import Callback
from scalar_binding_probe import MARKER, framed


@pytest.mark.parametrize("reflected", [False, True])
def test_mixed_numpy_preserves_binding(reflected):
    image = framed()
    other = np.ones(image.shape, dtype=np.int64)
    result = other + image if reflected else image + other
    assert result.coords[MARKER].item() == image.coords[MARKER].item()
    assert MARKER in result.xindexes
    npt.assert_array_equal(result.data, image.data + 1)


def test_inner_alignment_preserves_current_coordinates():
    image = framed()
    bare = xr.DataArray(
        np.ones((3, 4)), dims=("y", "x"), coords={"y": [0, 2, 4], "x": [3, 6, 9, 12]}
    )
    with xr.set_options(arithmetic_join="inner"):
        result = image + bare
    npt.assert_array_equal(result.coords["x"].data, [3, 6, 9])
    assert result.coords[MARKER].item() == image.coords[MARKER].item()


def test_conflicting_guarded_mappings_raise():
    with pytest.raises(ValueError, match="incompatible coordinate mappings"):
        framed() + framed(20)


@pytest.mark.parametrize("operation", ["scalar", "alignment", "selection"])
def test_supported_dask_probes_do_not_compute_pixels(operation):
    image = framed()
    lazy = image.copy(data=da.from_array(image.data, chunks=(1, 2)))
    tasks = []
    with (
        Callback(pretask=lambda key, *args: tasks.append(key)),
        xr.set_options(arithmetic_join="inner"),
    ):
        match operation:
            case "scalar":
                result = lazy + 2
            case "alignment":
                result = lazy + lazy.isel(x=slice(1, None))
            case "selection":
                result = lazy.isel(x=slice(1, None, 2))
            case _:
                raise AssertionError(operation)
    assert isinstance(result.data, da.Array)
    assert not tasks
    assert MARKER in result.xindexes


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="scalar guard does not compare fixed mapped coordinates",
)
def test_fixed_plane_conflicts_must_raise():
    image = framed()
    with pytest.raises(ValueError, match="incompatible"):
        image.isel(y=0) + image.isel(y=1)


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="scalar guard permits framed plane broadcasting through volume",
)
def test_plane_volume_conflicts_must_raise():
    image = framed()
    with pytest.raises(ValueError, match="incompatible"):
        image.isel(y=0) + image


def test_where_preserves_matching_binding():
    image = framed()
    result = xr.where(image > 5, image, 0)
    assert result.coords[MARKER].item() == image.coords[MARKER].item()
    npt.assert_array_equal(result.data, np.where(image.data > 5, image.data, 0))


def test_where_rejects_conflicting_framed_branch():
    image = framed()
    with pytest.raises(ValueError, match=r"cannot align|incompatible"):
        xr.where(image > 5, image, framed(20))
