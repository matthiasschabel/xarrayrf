"""Check that built artifacts contain the intended payload only."""

import tarfile
import tomllib
import zipfile
from pathlib import Path


def main() -> None:
    """Verify the wheel and source distribution after a build."""
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    stem = f"{project['name']}-{project['version']}"
    with zipfile.ZipFile(root / "dist" / f"{stem}-py3-none-any.whl") as archive:
        names = archive.namelist()
        unexpected = [
            name for name in names if not name.startswith(("xarrayrf/", f"{stem}.dist-info/"))
        ]
        required = {"xarrayrf/__init__.py", "xarrayrf/py.typed"}
        if unexpected or not required.issubset(names):
            raise ValueError(f"invalid wheel payload: unexpected={unexpected}, files={names}")
    allowed = {".gitignore", "CHANGELOG.md", "LICENSE", "README.md", "pyproject.toml", "PKG-INFO"}
    with tarfile.open(root / "dist" / f"{stem}.tar.gz") as archive:
        names = [name.removeprefix(f"{stem}/") for name in archive.getnames()]
        unexpected = [
            name for name in names if name not in allowed and not name.startswith("src/xarrayrf/")
        ]
        required = {"CHANGELOG.md", "src/xarrayrf/__init__.py", "src/xarrayrf/py.typed"}
        if unexpected or not required.issubset(names):
            raise ValueError(f"invalid source payload: unexpected={unexpected}, files={names}")
    print("Wheel and source distribution contain only the intended payload")


if __name__ == "__main__":
    main()
