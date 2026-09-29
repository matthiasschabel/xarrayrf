"""Shared anatomical vocabulary and patient coordinate systems."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from xarrayrf import ReferenceFrame, coordinate_system_change
from xarrayrf.anatomy import LPS, RAS, VOCABULARY, patient_coordinate_system

ATOL = 1e-12  # Signed permutations have exact coefficients; allow float evaluation slack.


def test_rfc4_vocabulary_is_exact_and_bidirectional() -> None:
    """Canonical list from https://ngff.openmicroscopy.org/rfc/4/ ."""
    pairs = (
        "left-to-right",
        "anterior-to-posterior",
        "inferior-to-superior",
        "dorsal-to-ventral",
        "proximal-to-distal",
        "dorsal-to-palmar",
        "dorsal-to-plantar",
        "rostral-to-caudal",
        "cranial-to-caudal",
        "superficial-to-deep",
        "apical-to-basal",
        "apex-to-base",
    )
    expected = set(pairs)
    expected.update("-to-".join(reversed(token.split("-to-"))) for token in pairs)
    assert VOCABULARY.identifier == "ome-ngff:rfc-4:anatomical"
    assert VOCABULARY.directions == expected
    assert len(expected) == 24
    assert all(VOCABULARY.opposite(token) in VOCABULARY.directions for token in expected)


def test_ras_lps_patient_systems_share_an_exact_derived_flip() -> None:
    ras = patient_coordinate_system(RAS, "mm")
    lps = patient_coordinate_system(LPS, "mm")
    assert ras.axes == lps.axes == ("x", "y", "z")
    assert ras.axis_types == lps.axis_types == ("space",) * 3
    frame = ReferenceFrame.local(ras)
    change = coordinate_system_change(frame, frame.with_coordinate_system(lps))
    assert_allclose(change.matrix, np.diag([-1.0, -1.0, 1.0]), rtol=0, atol=ATOL)


def test_patient_system_can_name_its_axes() -> None:
    system = patient_coordinate_system(RAS, "um", axes=("R", "A", "S"))
    assert system.axes == ("R", "A", "S")
    assert system.units == ("um",) * 3
