from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "check_binding_probe.py"


def cli(environment: Path, report: object, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--environment", str(environment), *args],
        input=json.dumps(report),
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.fixture
def report(tmp_path: Path) -> dict[str, object]:
    module = tmp_path / ".venv-patched/lib/python3.12/site-packages/xarray/__init__.py"
    module.parent.mkdir(parents=True)
    module.touch()
    return {
        "xarray_module": str(module),
        "cases": [{"case": "isel_scalar", "classification": "ok"}],
    }


@pytest.mark.parametrize("source", ["stdin", "file"])
def test_all_ok(tmp_path: Path, report: dict[str, object], source: str) -> None:
    args: list[str] = []
    if source == "file":
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))
        args.append(str(path))
    result = cli(tmp_path / ".venv-patched", report, *args)
    assert result.returncode == 0, result.stderr
    assert "1 cases" in result.stderr


def test_reports_every_failing_case(tmp_path: Path, report: dict[str, object]) -> None:
    report["cases"] = [
        {"case": "isel_scalar", "classification": "ok"},
        {"case": "align", "classification": "hole"},
        {"case": "concat", "classification": "unknown"},
        {"case": "missing_classification"},
    ]
    result = cli(tmp_path / ".venv-patched", report)
    assert result.returncode == 1
    for name in ("align", "concat", "missing_classification"):
        assert f"case {name}: classification" in result.stderr
    assert "isel_scalar" not in result.stderr


@pytest.mark.parametrize("cases", [[], None, {}, "ok"])
def test_rejects_empty_or_invalid_cases(
    tmp_path: Path, report: dict[str, object], cases: object
) -> None:
    report["cases"] = cases
    result = cli(tmp_path / ".venv-patched", report)
    assert result.returncode == 1
    assert "cases must be a nonempty list" in result.stderr


@pytest.mark.parametrize("location", ["stock", "prefix", "parent", "symlink"])
def test_rejects_wrong_module_path(
    tmp_path: Path, report: dict[str, object], location: str
) -> None:
    environment = tmp_path / ".venv-patched"
    outside = tmp_path / ".venv/lib/python3.12/site-packages/xarray/__init__.py"
    outside.parent.mkdir(parents=True)
    outside.touch()
    module = outside
    if location == "prefix":
        module = tmp_path / ".venv-patched-other/xarray/__init__.py"
    elif location == "parent":
        module = environment / ".." / outside.relative_to(tmp_path)
    elif location == "symlink":
        module = environment / "linked.py"
        module.symlink_to(outside)
    report["xarray_module"] = str(module)
    result = cli(environment, report)
    assert result.returncode == 1
    assert "resolves outside patched environment" in result.stderr


@pytest.mark.parametrize("module", [None, "", "relative/xarray/__init__.py"])
def test_rejects_invalid_module_path(
    tmp_path: Path, report: dict[str, object], module: object
) -> None:
    report["xarray_module"] = module
    result = cli(tmp_path / ".venv-patched", report)
    assert result.returncode == 1
    assert "xarray_module must be a nonempty absolute path" in result.stderr


def test_rejects_nonobject_case(tmp_path: Path, report: dict[str, object]) -> None:
    report["cases"] = [None]
    result = cli(tmp_path / ".venv-patched", report)
    assert result.returncode == 1
    assert "case #1: expected an object" in result.stderr


def test_rejects_nonobject_report(tmp_path: Path) -> None:
    result = cli(tmp_path, [])
    assert result.returncode == 1
    assert "probe report must be a JSON object" in result.stderr


@pytest.mark.parametrize("exists", [True, False])
def test_rejects_unreadable_json(tmp_path: Path, exists: bool) -> None:
    path = tmp_path / "report.json"
    if exists:
        path.write_text("{truncated")
    result = cli(tmp_path, None, str(path))
    assert result.returncode == 1
    assert "cannot read probe JSON" in result.stderr
