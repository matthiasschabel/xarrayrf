"""The NumPy-only core imports and works without xarray (acceptance case 6)."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import xarrayrf

INTEGRATION_MODULES = frozenset({"_geometry", "_resample", "_frame_coordinates", "_binding"})
"""Modules allowed to import xarray; everything else in the package is core."""

CORE_WITHOUT_XARRAY = textwrap.dedent(
    """
    import sys
    sys.modules["xarray"] = None  # any import of xarray now raises ImportError

    import xarrayrf as xrf

    frame = xrf.ReferenceFrame.local(xrf.CoordinateSystem(("x",), ("mm",)))
    transform = xrf.AffineTransform(
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
    for method, indexers in (("isel", {"i": slice(None, None, -1)}), ("sel", {"i": 1})):
        try:
            getattr(grid, method)(**indexers)
        except ImportError as error:
            assert f"Grid.{method} requires xarray" in str(error)
        else:
            raise AssertionError(f"Grid.{method} must need xarray")
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
    """Catches imports the runtime check would miss: inside functions or never executed."""
    package = Path(xarrayrf.__file__).parent
    core = [
        path
        for path in package.glob("_*.py")
        if path.stem not in INTEGRATION_MODULES | {"_grid"} and path.name != "__init__.py"
    ]
    assert core, "no core modules found; the check would pass vacuously"
    offenders = {path.name: xarray_imports(path.read_text()) for path in core}
    assert {name: lines for name, lines in offenders.items() if lines} == {}


def test_grid_imports_xarray_only_inside_selection_methods() -> None:
    source = (Path(xarrayrf.__file__).parent / "_grid.py").read_text()
    allowed: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name in ("isel", "sel"):
            imports = xarray_imports(ast.unparse(node))
            assert len(imports) == 1
            allowed.extend(
                child.lineno
                for child in ast.walk(node)
                if isinstance(child, ast.Import)
                and any(alias.name == "xarray" for alias in child.names)
            )
    assert len(allowed) == 2
    assert xarray_imports(source) == sorted(allowed)


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
