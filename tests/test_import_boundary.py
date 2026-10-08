"""The NumPy-only core imports and works without xarray (acceptance case 6)."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import xarrayrf

INTEGRATION_MODULES = frozenset(
    {"_geometry", "_resample", "_frame_coordinates", "_binding", "_grid_selection"}
)
"""Modules allowed to import xarray; everything else in the package is core."""

CORE_WITHOUT_XARRAY = textwrap.dedent(
    """
    import sys
    sys.modules["xarray"] = None  # any import of xarray now raises ImportError

    import xarrayrf as xrf

    frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x",), ("mm",)))
    transform = xrf.AffineTransform.from_matrix(
        target=frame, source=xrf.ArrayCoordinates(("i",), ("1",)), matrix=((2.0,),), translation=(1.0,)
    )
    assert float(xrf.transform_named(transform, {"i": 3.0})["x"]) == 7.0
    import numpy as np
    grid = xrf.Grid(transform, {"i": ("i", [0, 1, 3])})
    np.testing.assert_allclose(grid.point_at(i=2), [7.0], atol=1e-12)
    np.testing.assert_allclose(grid.points_at([[1.5]]), [[5.0]], atol=1e-12)
    np.testing.assert_allclose(grid.positions_at([[5.0]]), [[1.5]], atol=1e-12)
    assert xrf.decode(xrf.encode(grid)) == grid
    assert hash(grid) == hash(xrf.decode(xrf.encode(grid)))
    assert grid.is_coincident(grid)
    from xarrayrf._frame_adoption import adopt_frame
    from xarrayrf._frame_compatibility import binding_difference
    from xarrayrf._intervals import interval_rows, intervals_equal
    from xarrayrf._resampling import plan, execute
    assert adopt_frame(transform, frame) == transform
    assert "same frame on different grids" in binding_difference(
        transform, transform, samplings=lambda: (grid._sampling(), grid._sampling()),
        adoption_suffices=lambda _: False,
    )
    rows = np.array([[0, 1], [1, 2]], dtype=np.float64)
    positions = np.array([1, 0], dtype=np.intp)
    assert intervals_equal(rows, rows)
    np.testing.assert_array_equal(interval_rows(rows, positions), rows[::-1])
    assert not any(name.split(".")[0] == "scipy" for name in sys.modules)
    for indexer, expected in (
        (1, {"i": 1}),
        (slice(None, None, -1), {"i": ("i", [3, 1, 0])}),
        ([2, 0], {"i": ("i", [3, 0])}),
        (np.array([True, False, True]), {"i": ("i", [0, 3])}),
    ):
        assert grid.isel(i=indexer) == xrf.Grid(transform, expected)
    try:
        grid.sel(i=1)
    except ImportError as error:
        assert "Grid.sel requires xarray" in str(error)
    else:
        raise AssertionError("Grid.sel must need xarray")
    assert grid.transpose().dims == ("i",)
    uniform = xrf.Grid(transform, {"i": ("i", [0, 1, 2])})
    np.testing.assert_allclose(uniform.lattice().origin, [1.0], atol=1e-12)
    try:
        xrf.Geometry
    except ImportError as error:
        assert "install xarray" in str(error)
    else:
        raise AssertionError("Geometry must need xarray")
    loaded = sorted(name for name in sys.modules if name.split(".")[0] == "xarray")
    assert loaded == ["xarray"], loaded  # only the blocking sentinel
    namespace = {}
    exec("from xarrayrf import *", namespace)
    assert "ReferenceFrame" in namespace and "Grid" in namespace and "Geometry" not in namespace
    """
)


def test_core_imports_and_evaluates_without_xarray() -> None:
    result = subprocess.run(
        [sys.executable, "-c", CORE_WITHOUT_XARRAY], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def xarray_imports(source: str) -> list[int]:
    """Line numbers of any import of xarray, at any depth, including inside functions."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(name == "xarray" or name.startswith("xarray.") for name in names):
            lines.append(node.lineno)
    return lines


def test_core_modules_never_import_xarray_even_lazily() -> None:
    """Catches integration imports at any depth, including never executed functions."""
    package = Path(xarrayrf.__file__).parent
    core = [
        path
        for path in package.glob("_*.py")
        if path.stem not in INTEGRATION_MODULES and path.name != "__init__.py"
    ]
    assert core, "no core modules found; the check would pass vacuously"
    assert package / "_resampling.py" in core
    assert package / "_intervals.py" in core
    offenders = {}
    for path in core:
        offenders[path.name] = sampling_imports_with_grid_exemption(path.stem, path.read_text())
    assert {name: imports for name, imports in offenders.items() if imports} == {}


def sampling_imports_with_grid_exemption(module: str, source: str) -> set[str]:
    tree = ast.parse(source)
    if module == "_grid":
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "Grid":
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "sel":
                        method.body = [ast.Pass()]
    return forbidden_sampling_imports(ast.unparse(tree))


def test_grid_imports_integration_only_inside_grid_sel() -> None:
    source = (Path(xarrayrf.__file__).parent / "_grid.py").read_text()
    assert xarray_imports(source) == []
    assert forbidden_sampling_imports(source) == {"._grid_selection"}
    assert sampling_imports_with_grid_exemption("_grid", source) == set()
    assert sampling_imports_with_grid_exemption("_grid", "def sel():\n    import xarray\n") == {
        "xarray"
    }
    assert sampling_imports_with_grid_exemption(
        "_grid", "class Other:\n    def sel(self):\n        import xarray\n"
    ) == {"xarray"}
    assert sampling_imports_with_grid_exemption(
        "_grid", "class Grid:\n    def isel(self):\n        import xarray\n"
    ) == {"xarray"}


def test_the_import_scan_detects_an_xarray_import() -> None:
    """Prove the scan can fail, including for a lazy import inside a function."""
    assert xarray_imports("def f():\n    import xarray as xr\n") == [2]
    assert xarray_imports("from xarray.core import indexing\n") == [1]
    assert xarray_imports("import numpy\n") == []


def import_names(source: str) -> set[str]:
    """Find imported modules, including relative imports and imports inside functions."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            if node.module:
                names.add(prefix + node.module)
            else:
                names.update(prefix + alias.name for alias in node.names)
    return names


def forbidden_core_imports(source: str) -> set[str]:
    forbidden = {"anatomy", "nifti", "dicom", "ngff", "geotiff"}
    return {
        name
        for name in import_names(source)
        if name.removeprefix("xarrayrf.").lstrip(".").split(".")[0] in forbidden
    }


def forbidden_sampling_imports(source: str) -> set[str]:
    forbidden = INTEGRATION_MODULES | {
        "xarray",
        "pandas",
        "native",
        "anatomy",
        "nifti",
        "dicom",
        "ngff",
        "geotiff",
    }
    names = import_names(source)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == "xarrayrf":
            names.update("xarrayrf." + alias.name for alias in node.names)
    return {
        name
        for name in names
        if name.removeprefix("xarrayrf.").lstrip(".").split(".")[0] in forbidden
    }


def test_sampling_import_scan_can_fail() -> None:
    for name in sorted(INTEGRATION_MODULES | {"native", "nifti", "dicom", "ngff", "geotiff"}):
        assert forbidden_sampling_imports(f"def f():\n    from .{name} import value\n") == {
            f".{name}"
        }
        assert forbidden_sampling_imports(f"from xarrayrf import {name}\n") == {f"xarrayrf.{name}"}
    assert forbidden_sampling_imports("def f():\n    import pandas as pd\n") == {"pandas"}
    assert forbidden_sampling_imports("from xarray.core import indexing\n") == {"xarray.core"}
    assert forbidden_sampling_imports("from ._sampling import Sampling\n") == set()


def test_core_resampling_without_xarray() -> None:
    pytest.importorskip("scipy")
    script = textwrap.dedent(
        """
        import sys
        sys.modules["xarray"] = None
        sys.modules["pandas"] = None
        import numpy as np
        import xarrayrf as xrf
        from xarrayrf._resampling import plan, execute

        frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x",), ("mm",)))
        transform = xrf.AffineTransform.from_matrix(
            source=xrf.ArrayCoordinates(("u",), ("mm",)), target=frame, matrix=((1.0,),), translation=(0.0,)
        )
        source = xrf.Grid(transform, {"u": ("i", [0, 2, 4])})
        target = xrf.Grid(transform, {"u": ("j", [1, 3])})
        prepared = plan(
            source._sampling(), target._sampling(), source_order=("i",), target_order=("j",),
            dtype=np.dtype(np.float64), transform=None, method="linear", fill_value=np.nan,
            domain="samples", block_points=1, other_dims=()
        )
        values = np.array([2.0, 6.0, 10.0])
        if prepared.window is not None:
            values = values[prepared.window]
        # Each target is halfway between neighbours: (2 + 6)/2 and (6 + 10)/2.
        np.testing.assert_allclose(execute(prepared, values), [4.0, 8.0], rtol=0, atol=1e-12)
        assert not any(name.startswith("xarray.") for name in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def forbidden_adapter_imports(source: str, *, anatomy: bool, dicom: bool = False) -> set[str]:
    allowed = sys.stdlib_module_names | {"numpy"}
    if not anatomy:
        allowed |= ({"pydicom", "xarray"} if dicom else {"nibabel", "xarray"}) | {"dask"}
    return {
        name
        for name in import_names(source)
        if not (
            name.split(".")[0] in allowed
            or name == "xarrayrf"
            or name.startswith("xarrayrf._")
            or (not anatomy and name in {"xarrayrf.anatomy", "xarrayrf.native", "xarrayrf.units"})
            or name.startswith("._")
            or (not anatomy and name.startswith(".") and not name.startswith(".."))
        )
    }


def forbidden_ngff_imports(source: str) -> set[str]:
    """Keep the metadata adapter within the core and its declared format dependencies."""
    allowed = sys.stdlib_module_names | {
        "numpy",
        "xarray",
        "ome_zarr_models",
        "pydantic",
        "dask",
        "zarr",
    }
    return {
        name
        for name in import_names(source)
        if not (
            name.split(".")[0] in allowed
            or name == "xarrayrf"
            or name in {"xarrayrf.anatomy", "xarrayrf.native", "xarrayrf.units"}
            or name.startswith("xarrayrf._")
            or (name.startswith(".") and not name.startswith(".."))
        )
    }


def forbidden_geotiff_imports(source: str) -> set[str]:
    """Allow only core, xarray and the GeoTIFF adapter's format dependencies."""
    allowed = sys.stdlib_module_names | {"numpy", "xarray", "rasterio", "pyproj", "dask"}
    return {
        name
        for name in import_names(source)
        if not (
            name.split(".")[0] in allowed
            or name == "xarrayrf"
            or name == "xarrayrf.native"
            or name.startswith("xarrayrf._")
            or (name.startswith(".") and not name.startswith(".."))
        )
    }


def test_core_never_imports_anatomy_or_nifti() -> None:
    package = Path(xarrayrf.__file__).parent
    core = [package / "__init__.py", *package.glob("_*.py")]
    assert all(path.exists() for path in core)
    offenders = {path.name: forbidden_core_imports(path.read_text()) for path in core}
    assert {name: imports for name, imports in offenders.items() if imports} == {}


def test_adapter_imports_stay_within_their_boundaries() -> None:
    package = Path(xarrayrf.__file__).parent
    anatomy = (package / "anatomy.py").read_text()
    nifti = sorted((package / "nifti").glob("*.py"))
    dicom = sorted((package / "dicom").glob("*.py"))
    ngff = sorted((package / "ngff").glob("*.py"))
    geotiff = sorted((package / "geotiff").glob("*.py"))
    assert nifti, "no nifti modules found; the check would pass vacuously"
    assert dicom, "no dicom modules found; the check would pass vacuously"
    assert ngff, "no ngff modules found; the check would pass vacuously"
    assert geotiff, "no geotiff modules found; the check would pass vacuously"
    assert forbidden_adapter_imports(anatomy, anatomy=True) == set()
    offenders = {
        path.name: forbidden_adapter_imports(path.read_text(), anatomy=False) for path in nifti
    }
    assert {name: imports for name, imports in offenders.items() if imports} == {}
    dicom_offenders = {
        path.name: forbidden_adapter_imports(path.read_text(), anatomy=False, dicom=True)
        for path in dicom
    }
    assert {name: imports for name, imports in dicom_offenders.items() if imports} == {}
    ngff_offenders = {path.name: forbidden_ngff_imports(path.read_text()) for path in ngff}
    assert {name: imports for name, imports in ngff_offenders.items() if imports} == {}
    geotiff_offenders = {path.name: forbidden_geotiff_imports(path.read_text()) for path in geotiff}
    assert {name: imports for name, imports in geotiff_offenders.items() if imports} == {}


def test_adapter_import_scan_can_fail() -> None:
    assert forbidden_core_imports("def f():\n    import xarrayrf.nifti\n") == {"xarrayrf.nifti"}
    assert forbidden_core_imports("from . import anatomy\n") == {".anatomy"}
    assert forbidden_core_imports("import xarrayrf.dicom\n") == {"xarrayrf.dicom"}
    assert forbidden_core_imports("import xarrayrf.ngff\n") == {"xarrayrf.ngff"}
    assert forbidden_core_imports("import xarrayrf.geotiff\n") == {"xarrayrf.geotiff"}
    assert forbidden_adapter_imports("from someapp import data\n", anatomy=False) == {"someapp"}
    assert forbidden_adapter_imports("import xarrayrf.nifti\n", anatomy=True) == {"xarrayrf.nifti"}
    assert forbidden_adapter_imports("import nibabel\n", anatomy=False, dicom=True) == {"nibabel"}
    assert forbidden_adapter_imports("from .. import dicom\n", anatomy=False) == {"..dicom"}
    assert forbidden_ngff_imports("import xarrayrf.dicom\n") == {"xarrayrf.dicom"}
    assert forbidden_ngff_imports("from .. import dicom\n") == {"..dicom"}
    assert forbidden_ngff_imports("def f():\n    import zarr\n") == set()
    assert forbidden_ngff_imports("from ome_zarr_models.v06 import scene\n") == set()
    assert forbidden_geotiff_imports("import rasterio\nimport pyproj\nimport dask\n") == set()
    assert forbidden_geotiff_imports("import nibabel\n") == {"nibabel"}
