"""NIfTI file reader behavior with synthetic images."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pytest
from dask.callbacks import Callback
from numpy.testing import assert_allclose

from xarrayrf.nifti import from_header, open, to_dataarray


@pytest.mark.parametrize(
    "shape,time", [((3, 4, 2), False), ((3, 4, 2, 5), False), ((3, 4, 2, 5), True)]
)
def test_nifti_open_scaling_geometry_and_laziness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], time: bool
) -> None:
    raw = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    image = nib.Nifti1Image(raw, np.diag([2.0, 3.0, 4.0, 1.0]))  # type: ignore[no-untyped-call]
    image.header.set_slope_inter(2.5, 7.0)  # type: ignore[no-untyped-call]
    image.header.set_xyzt_units("mm", "sec")  # type: ignore[no-untyped-call]
    filename = tmp_path / "image.nii"
    image.to_filename(filename)
    reads = []
    original = nib.arrayproxy.ArrayProxy.__getitem__

    def counted(self: Any, slicer: Any) -> Any:
        reads.append(slicer)
        return original(self, slicer)  # type: ignore[no-untyped-call]

    monkeypatch.setattr(nib.arrayproxy.ArrayProxy, "__getitem__", counted)
    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):  # type: ignore[no-untyped-call]
        array = open(filename, time=time)
    assert not reads and not tasks
    assert array.dtype == np.float64
    loaded = cast(nib.Nifti1Image, nib.load(filename))
    expected = to_dataarray(from_header(loaded.header, time=time), np.asarray(loaded.dataobj))
    assert array.dims == expected.dims
    assert_allclose(
        array.rf.geometry.points().data, expected.rf.geometry.points().data, rtol=0, atol=1e-12
    )
    reads.clear()
    assert_allclose(array.compute().data, raw * 2.5 + 7.0, rtol=0, atol=1e-12)
    assert reads
    eager = open(filename, time=time, chunks=None)
    assert isinstance(eager.data, np.ndarray)
    assert_allclose(eager.data, array.data, rtol=0, atol=1e-12)


def test_nifti_open_input_errors(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="path"):
        open(3)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        open(tmp_path / "missing.nii")
