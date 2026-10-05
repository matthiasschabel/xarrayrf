"""The README's code examples run as written and show the values their comments state."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

README = Path(__file__).parent.parent / "README.md"
NEEDS_FILES = "# Needs the data files"


def _examples() -> list[str]:
    blocks = re.findall(r"^ *```python\n(.*?)^ *```", README.read_text(), flags=re.S | re.M)
    dedented = []
    for block in blocks:
        lines = block.splitlines()
        indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
        dedented.append("\n".join(line[indent:] for line in lines))
    return [block for block in dedented if NEEDS_FILES not in block]


def test_readme_examples_run_and_match_their_comments() -> None:
    pytest.importorskip("scipy")
    namespace: dict[str, Any] = {}
    for block in _examples():
        exec(compile(block, str(README), "exec"), namespace)

    crop = namespace["crop"]
    np.testing.assert_allclose(crop.rf.geometry.point_at(j=0, i=0).values, [10.5, 20.0])
    grid = namespace["grid"]
    anatomy = namespace["anatomy"]
    np.testing.assert_allclose(grid.point_at(k=1, j=0, i=0), [-2.5, -2.0, 3.0])
    np.testing.assert_allclose(grid.positions_at([[0.0, 0.0, 4.5]]), [[1.5, 2.0, 2.5]])
    assert anatomy.orientation_codes(grid) == "SPL"
    assert anatomy.orientation_codes(namespace["resliced"].rf.grid) == "RIP"
    assert not bool(namespace["resliced"].isnull().any())
    assert "anonymous" in namespace["refusal"]
    assert "rf.assume_frame alone suffices" in namespace["refusal"]
    assert namespace["combined"].rf.is_framed


def test_readme_has_runnable_examples() -> None:
    assert len(_examples()) >= 3
