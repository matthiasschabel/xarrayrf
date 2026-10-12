"""Generate the Astropy oracle for ``SphericalToCartesian`` (tests/test_chart_transitions.py).

Astropy is not a test dependency, so its answers are recorded once with their provenance:
ICRS right ascension, declination and distance, and ``SkyCoord(...).cartesian`` for each.

    uv run --no-project --with astropy --with numpy python tools/build_astropy_spherical_fixture.py
    uv run --no-project --with astropy --with numpy python tools/build_astropy_spherical_fixture.py --apply

Without ``--apply`` the JSON goes to stdout and nothing is written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import astropy
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "astropy_spherical.json"


def build() -> dict[str, object]:
    # Right ascension spans the 0/360 seam; declination includes both poles and the equator.
    ra = np.array([0.0, 10.684708, 83.633083, 179.999, 180.0, 266.416837, 359.9999, 45.0, 300.0])
    dec = np.array([0.0, 41.26875, 22.0145, -89.5, 90.0, -29.007811, 0.5, -90.0, 60.0])
    distance = np.array([1.0, 778.0, 2000.0, 1.0e-3, 5.0, 8178.0, 1.0e6, 2.5, 0.75])
    cartesian = SkyCoord(
        ra=ra * u.deg, dec=dec * u.deg, distance=distance * u.pc, frame="icrs"
    ).cartesian
    return {
        "astropy_version": astropy.__version__,
        "frame": "icrs",
        "units": {"angle": "deg", "length": "pc"},
        "spherical": np.stack((ra, dec, distance), axis=-1).tolist(),
        "cartesian": np.stack(
            (cartesian.x.to_value(u.pc), cartesian.y.to_value(u.pc), cartesian.z.to_value(u.pc)),
            axis=-1,
        ).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help=f"write {FIXTURE}")
    args = parser.parse_args()
    text = json.dumps(build(), indent=1) + "\n"
    if not args.apply:
        sys.stdout.write(text)
        return 0
    # Write beside the target and rename, so a failure never leaves a partial fixture.
    with tempfile.NamedTemporaryFile(
        "w", dir=FIXTURE.parent, suffix=".tmp", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(text)
    os.replace(handle.name, FIXTURE)
    print(f"wrote {FIXTURE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
