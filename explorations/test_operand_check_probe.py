"""Pin locked-xarray observations, not an xarrayrf support contract."""

import pytest
import xarray as xr
from operand_check_probe import CASES, JOINS, ORDERS, PAIR_OPERATIONS, WHERE_PAIRS, observations

# Each six-character row lists the observed classes for cases A-F in either order.
CLASSES = {"S": "silent", "I": "index-refused", "X": "xarray-refused"}
EXPECTED_ROWS = {
    "binary": "SISSSS",
    "numpy": "SISSSS",
    "apply_ufunc": "SXSSSS",
    "concat": "SISSSS",
    "merge": "SISSSX",
    "where_cond_x": "SXSSSS",
    "where_cond_y": "SXSSSS",
    "where_x_y": "SXSSSS",
    "align_inner": "SISSSS",
    "align_exact": "SXSSSS",
    "align_override": "SSSSSS",
}


def test_locked_operand_outcome_classes():
    if xr.__version__ != "2026.7.0" or hasattr(xr.Index, "join_overlapping"):
        pytest.skip("locked stock 2026.7.0 observation; source lanes are separate")
    report = observations()
    assert report["xarray_version"] == "2026.7.0"
    assert report["xarray_git_revision"] is None
    cells = report["cells"]
    operations = (
        PAIR_OPERATIONS
        + tuple(f"where_{a}_{b}" for a, b in WHERE_PAIRS)
        + tuple(f"align_{join}" for join in JOINS)
    )
    assert set(EXPECTED_ROWS) == set(operations)
    expected = {
        f"{operation}/{case}/{order}": CLASSES[code]
        for operation, row in EXPECTED_ROWS.items()
        for case, code in zip(CASES, row, strict=True)
        for order in ORDERS
    }
    expected.update({name: "silent" for name in ("scalar_right", "scalar_left", "scalar_numpy")})
    assert set(cells) == set(expected)
    assert {name: cell["outcome"] for name, cell in cells.items()} == expected
