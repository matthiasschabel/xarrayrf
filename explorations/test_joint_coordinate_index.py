"""Observations of the joint-ownership candidate; none of this is a production guarantee.

Tests named ``test_joint_index_*`` check the candidate's *intended* per-axis delegation: if one
fails, the candidate is wrong. Tests named ``test_observed_*`` record what stock xarray does
with joint ownership. These assertions record measurements on pinned xarray and an upstream source checkout;
a changed observation must be recorded in ``docs/dev/binding/binding_design.md`` rather than
suppressed with an xfail.
"""

import json
import sys

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from joint_coordinate_index import JointFrameIndex, jointly_framed
from operation_trace_probe import trace
from scalar_binding_probe import MARKER, framed


def test_joint_index_delegates_exact_label_selection():
    result = jointly_framed().sel(x=6)
    assert result.coords["x"].item() == 6
    npt.assert_array_equal(result.data, [2, 6, 10])


def test_joint_index_delegates_nearest_label_selection():
    result = jointly_framed().sel(x=5, method="nearest")
    assert result.coords["x"].item() == 6


def test_joint_index_delegates_slice_label_selection():
    result = jointly_framed().sel(x=slice(3, 9))
    npt.assert_array_equal(result.coords["x"].data, [3, 6, 9])


def test_joint_index_keeps_the_crop_stride_labels():
    result = jointly_framed().isel(x=slice(1, None, 2))
    npt.assert_array_equal(result.coords["x"].data, [3, 9])
    assert isinstance(result.xindexes[MARKER], JointFrameIndex)


def test_joint_index_keeps_the_gathered_labels_in_order():
    result = jointly_framed().isel(x=[3, 0, 1])
    npt.assert_array_equal(result.coords["x"].data, [9, 0, 3])


def test_joint_index_retains_the_scalar_selection_term():
    result = jointly_framed().isel(y=1)
    assert result.coords["y"].item() == 2
    assert MARKER in result.xindexes
    assert "x" in result.xindexes


def test_joint_index_rewrites_declaration_inputs_on_rename():
    result = jointly_framed().rename(x="column")
    assert json.loads(result.coords[MARKER].item())["inputs"] == ["y", "column"]


def test_joint_index_rejects_selection_on_the_declaration():
    with pytest.raises(KeyError):
        jointly_framed().sel(**{MARKER: "anything"})


def test_scalar_baseline_is_unchanged():
    """The candidate must not displace the baseline it is being compared against."""
    assert MARKER in framed().xindexes
    assert not isinstance(framed().xindexes[MARKER], JointFrameIndex)


def test_trace_restores_the_previous_profile_function():
    previous = sys.getprofile()
    trace(lambda: framed() + 1)
    assert sys.getprofile() is previous


def test_trace_restores_the_profile_function_after_a_failure():
    previous = sys.getprofile()
    with pytest.raises(ZeroDivisionError):
        trace(lambda: 1 / 0)
    assert sys.getprofile() is previous


def test_observed_shifted_joint_ownership_conflicts_with_ordinary_indexes():
    """Shifted labels conflict on both tested xarray revisions: two index keys claim 'y', so alignment refuses before any join.

    ``Aligner._collect_indexes`` keys an index by its coordinate names, their dimensions and
    the index type, so a joint ``(y, x, marker)`` index never matches an ordinary array's two
    separate ``PandasIndex`` entries, and ``align_indexes`` raises on the duplicate name.
    """
    bare = xr.DataArray(
        np.ones((3, 4)), dims=("y", "x"), coords={"y": [0, 2, 4], "x": [3, 6, 9, 12]}
    )
    with pytest.raises(xr.AlignmentError, match="conflicting indexes"):
        _ = jointly_framed() + bare


def test_observed_scalar_selection_survives_drop_true():
    """Record the old behavior and the patched partial-index-drop refusal.

    Stock xarray keeps the index-owned coordinate. The patched lane refuses to drop only
    some of a multi-coordinate index's coordinates.
    """
    if hasattr(xr.Index, "join_overlapping"):
        with pytest.raises(ValueError, match="corrupt"):
            jointly_framed().isel(y=1, drop=True)
        return
    result = jointly_framed().isel(y=1, drop=True)
    assert result.coords["y"].item() == 2


def test_observations_report_restoration_with_an_existing_profiler():
    from operation_trace_probe import observations

    previous = sys.getprofile()

    def existing(frame, event, arg):
        pass

    sys.setprofile(existing)
    try:
        assert observations()["profile_restored"] is True
        assert sys.getprofile() is existing
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("guarded", [False, True])
def test_observed_conflicting_declarations_reach_merge_together(reverse, guarded):
    from scalar_binding_probe import unguarded

    first = framed(10) if guarded else unguarded(10)
    second = unguarded(20)
    left, right = (second, first) if reverse else (first, second)
    calls = trace(lambda: left + right)
    merges = [call for call in calls if call["function"] == "merge_collected"]
    assert merges
    assert merges[-1]["declaration_elements"] == 2


def test_observed_vectorized_selection_leaves_an_unguarded_declaration():
    result = jointly_framed().isel(x=xr.Variable("z", [0, 1]))
    assert MARKER in result.coords
    assert MARKER not in result.xindexes
