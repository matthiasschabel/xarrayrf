"""Gate binding-probe JSON on case classifications and xarray import location."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    """Read a probe report from a file or stdin; return nonzero on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", default="-", help="JSON file, or - for stdin")
    parser.add_argument("--environment", required=True, type=Path, help="Patched environment root")
    args = parser.parse_args()
    try:
        text = sys.stdin.read() if args.report == "-" else Path(args.report).read_text()
        report = json.loads(text)
    except (OSError, ValueError) as exc:
        print(f"cannot read probe JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(report, dict):
        print("probe report must be a JSON object", file=sys.stderr)
        return 1

    failures: list[str] = []
    environment = args.environment.resolve()
    module = report.get("xarray_module")
    if not isinstance(module, str) or not module or not Path(module).is_absolute():
        failures.append("xarray_module must be a nonempty absolute path")
    elif not Path(module).resolve().is_relative_to(environment):
        failures.append(
            f"xarray_module {module!r} resolves outside patched environment {environment}"
        )

    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        failures.append("cases must be a nonempty list")
    else:
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                failures.append(f"case #{index + 1}: expected an object")
            elif case.get("classification") != "ok":
                name = case.get("case", f"#{index + 1}")
                failures.append(
                    f"case {name}: classification {case.get('classification')!r} != 'ok'"
                )

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    assert isinstance(cases, list)
    print(f"Binding probe passed: {len(cases)} cases; xarray_module: {module}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
