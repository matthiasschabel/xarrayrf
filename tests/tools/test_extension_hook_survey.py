from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "extension_hook_survey.py"


@pytest.fixture(scope="module")
def survey() -> ModuleType:
    spec = importlib.util.spec_from_file_location("extension_hook_survey", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _case(survey: ModuleType, original: object, hooked: object | None) -> object:
    return survey.Case("c", "lib", "op", "expected", "hook", "path", original, hooked)


def test_a_hook_that_refuses_resolves_the_case(survey: ModuleType) -> None:
    calls: list[str] = []

    def refuse() -> None:
        calls.append("check_drop('stack')")
        raise survey.HookRefusalError("refused")

    original = survey._run("Base", lambda: "lost")
    hooked = survey._run("Hooked", refuse, calls)
    assert survey._verdict(_case(survey, original, hooked)) == "hook resolves"


def test_an_error_after_the_hook_keeps_the_recorded_call(survey: ModuleType) -> None:
    calls: list[str] = []

    def fail_after_hook() -> None:
        calls.append("join_overlapping(how='inner')")
        raise RuntimeError("later failure")

    original = survey._run("Base", lambda: "split")
    hooked = survey._run("Hooked", fail_after_hook, calls)
    assert hooked.hook_calls == ["join_overlapping(how='inner')"]
    assert survey._verdict(_case(survey, original, hooked)) == "hook reached but not sufficient"


def test_an_unreached_hook_is_reported(survey: ModuleType) -> None:
    original = survey._run("Base", lambda: "lost")
    hooked = survey._run("Hooked", lambda: "lost", [])
    assert survey._verdict(_case(survey, original, hooked)) == "hook not reached"
