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


@pytest.mark.parametrize("remedy", ["frame", "assume_frame"])
def test_anonymous_opens_require_shared_world_and_survive_reload(
    tmp_path: Path, remedy: str
) -> None:
    import xarray as xr

    from xarrayrf import resample

    pixels = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    image = nib.Nifti1Image(pixels, np.diag([2.0, 3.0, 4.0, 1.0]))  # type: ignore[no-untyped-call]
    image.header.set_sform(image.affine, code=1)  # type: ignore[no-untyped-call]
    image.header.set_xyzt_units("mm")  # type: ignore[no-untyped-call]
    filename = tmp_path / "anonymous.nii"
    image.to_filename(filename)
    first, second = open(filename), open(filename)
    assert first.rf.reference_frame.is_anonymous
    assert second.rf.reference_frame.is_anonymous
    assert first.rf.reference_frame != second.rf.reference_frame
    assert not first.rf.grid.is_coincident(second.rf.grid)
    assert "anonymous=True" in repr(first.rf.reference_frame)
    assert "anonymous=True" in repr(first.rf.grid)
    assert "anonymous=True" in repr(first.xindexes["i"])
    message = r"left operand and right operand are anonymous.*points match numerically.*frame=.*rf.assume_frame alone suffices"
    with pytest.raises(ValueError, match=message):
        _ = first + second
    with pytest.raises(ValueError, match=message):
        xr.align(first, second, join="outer")
    with pytest.raises(ValueError, match=message):
        first.xindexes["i"].join(second.xindexes["i"])
    with pytest.raises(
        ValueError, match=r"source operand and target operand are anonymous.*alone suffices"
    ):
        first.rf.resample_to(second)
    with pytest.raises(
        ValueError, match=r"source operand and target operand are anonymous.*alone suffices"
    ):
        resample(first.rf.geometry, second.rf.grid)
    related = open(filename, frame=first) if remedy == "frame" else second.rf.assume_frame(first)
    assert_allclose((first + related).compute(), 2 * pixels, rtol=0, atol=1e-12)
    restored = related.rf.encode().rf.decode()
    assert restored.rf.reference_frame.is_anonymous
    assert restored.rf.reference_frame == first.rf.reference_frame
    assert_allclose((first + restored).compute(), 2 * pixels, rtol=0, atol=1e-12)


@pytest.mark.parametrize("array_target", [False, True])
def test_nifti_adopts_lps_dicom_through_both_remedies(tmp_path: Path, array_target: bool) -> None:
    import xarray as xr

    from xarrayrf import AffineTransform, ArrayCoordinates
    from xarrayrf.dicom import patient_frame

    pixels = np.ones((2, 3, 4), dtype=np.float32)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [10.0, 20.0, 30.0]
    image = nib.Nifti1Image(pixels, affine)  # type: ignore[no-untyped-call]
    image.header.set_sform(affine, code=1)  # type: ignore[no-untyped-call]
    image.header.set_xyzt_units("mm")  # type: ignore[no-untyped-call]
    filename = tmp_path / "ras.nii"
    image.to_filename(filename)
    frame = patient_frame("1.2.3.4")
    target = xr.DataArray(
        pixels, dims=("i", "j", "k"), coords={"i": range(2), "j": range(3), "k": range(4)}
    ).rf.frame(
        AffineTransform(
            source=ArrayCoordinates(("i", "j", "k"), ("1",) * 3),
            target=frame,
            matrix=np.eye(3),
            translation=np.zeros(3),
        ),
        dims=("i", "j", "k"),
    )
    other = target if array_target else frame
    anonymous = open(filename)
    assert anonymous.rf.reference_frame.is_anonymous
    loaded = open(filename, frame=other)
    assumed = anonymous.rf.assume_frame(other)
    assert loaded.rf.coordinate_transform == assumed.rf.coordinate_transform
    assert loaded.rf.reference_frame == frame
    assert_allclose(loaded.rf.grid.points(), assumed.rf.grid.points(), rtol=0, atol=1e-12)
    assert_allclose(
        loaded.rf.grid.points(), anonymous.rf.grid.points() * [-1, -1, 1], rtol=0, atol=1e-12
    )
    assert_allclose((loaded + assumed).compute(), 2 * pixels, rtol=0, atol=1e-12)
