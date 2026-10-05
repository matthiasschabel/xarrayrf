"""World points checked against independently calculated reference values.

Each case states its expected coordinates in the target frame as literal numbers derived by hand from
the worked examples in ``docs/design.md`` and ``docs/dev/architecture/cross_domain_cases.md``, so a
sign, ordering or offset error in the mapping shows up as a numeric mismatch rather than
as agreement between two copies of the same expression.

These are arithmetic oracles for the coordinate contract. Nothing here reads a DICOM
object, opens an NGFF store or validates a wire format: no file is parsed and no codec is
exercised, so passing says nothing about reader or writer conformance.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from xarrayrf import (
    AffineTransform,
    ArrayCoordinates,
    CoordinateSystem,
    DirectionVocabulary,
    ReferenceFrame,
    transform_named,
)

ATOL = 1e-12
"""Floating-point arithmetic tolerance; the reference values are exact rationals."""


def test_index_origin_and_sample_anchor_live_in_the_declaration() -> None:
    """Three conventions for the same sample must produce the same point.

    First sample centre 100 mm, spacing 2.5 mm, fourth sample. Zero-based centres,
    one-based labels and a lower cell boundary are encoded once by the adapter; no global
    half-voxel correction is applied anywhere in the core.
    """
    centre = 100.0
    spacing = 2.5
    boundary = centre - spacing / 2
    target_frame = ReferenceFrame.local(CoordinateSystem(("S",), ("mm",)))

    def build(name: str, offset: float) -> AffineTransform:
        return AffineTransform.from_matrix(
            target=target_frame,
            source=ArrayCoordinates((name,), ("1",)),
            matrix=((spacing,),),
            translation=(offset,),
        )

    # C + s*i with zero-based i, C + s*(j-1) with one-based j, B + s*(i+0.5) from the
    # lower cell boundary; the last folds the half-sample shift into the constant.
    zero_based = build("i", centre)
    one_based = build("j", centre - spacing)
    from_boundary = build("i", boundary + spacing / 2)

    assert_allclose(transform_named(zero_based, {"i": 3.0})["S"], 107.5, rtol=0, atol=ATOL)
    assert_allclose(transform_named(one_based, {"j": 4.0})["S"], 107.5, rtol=0, atol=ATOL)
    assert_allclose(transform_named(from_boundary, {"i": 3.0})["S"], 107.5, rtol=0, atol=ATOL)
    # The declared boundary is not itself a sample position: reading it as one would
    # move the point by half a sample.
    naive = boundary + spacing * 3
    assert abs(float(transform_named(from_boundary, {"i": 3.0})["S"]) - naive) > 1.0


def generic_plane_transform() -> AffineTransform:
    """The 2-D example from ``docs/dev/architecture/cross_domain_cases.md``."""
    target_frame = ReferenceFrame.local(CoordinateSystem(("x", "y"), ("mm", "mm")))
    return AffineTransform.from_matrix(
        target=target_frame,
        source=ArrayCoordinates(("row", "column"), ("1", "1")),
        matrix=((0.0, 0.5), (0.5, 0.0)),
        translation=(10.0, 20.0),
    )


def test_crop_uses_retained_coordinate_values_not_rebased_indices() -> None:
    """Acceptance case A1: crop rows ``10:30:2``, then read crop index ``(1, 3)``.

    The array keeps its original row label 12, so the unchanged mapping still locates the
    sample. Rebasing to a fresh zero-based index would silently move it.
    """
    transform = generic_plane_transform()
    retained = transform_named(transform, {"row": 10.0 + 2.0 * 1.0, "column": 3.0})
    assert_allclose(retained["x"], 11.5, rtol=0, atol=ATOL)
    assert_allclose(retained["y"], 26.0, rtol=0, atol=ATOL)
    rebased = transform_named(transform, {"row": 1.0, "column": 3.0})
    assert_allclose(rebased["y"], 20.5, rtol=0, atol=ATOL)


def test_dicom_style_direction_matrix_places_an_oblique_stack() -> None:
    """A patient-target_frame affine built the way ``docs/dev/architecture/cross_domain_cases.md`` builds it.

    The DICOM axis names are stated explicitly, because a self-consistent affine proves
    nothing about the convention it claims to follow:

    - ``row_direction`` is ImageOrientationPatient (0020,0037) elements 1-3, the direction
      cosines of the first row, followed as the *column* index increases;
    - ``column_direction`` is elements 4-6, the direction cosines of the first column,
      followed as the *row* index increases;
    - PixelSpacing (0028,0030) is (between-row spacing, between-column spacing), so
      ``row_spacing`` scales ``column_direction`` and ``column_spacing`` scales
      ``row_direction``;
    - the positive slice normal is ``cross(row_direction, column_direction)``, the
      right-handed normal DICOM readers use to order a stack. Reversing the cross product
      mirrors the stack through the image plane and leaves every point at slice offset 0
      exactly where it was, so only the values at a nonzero offset detect the error.

    No DICOM file or library is involved: the orientation vectors, spacings and first
    pixel position are literals, and the expected point is the hand-computed sum
    ``position + 5*normal + 3*0.5*column_direction + 2*0.75*row_direction``.
    """
    anatomy = DirectionVocabulary(
        "test-anatomy",
        ("right-to-left", "anterior-to-posterior", "inferior-to-superior"),
    )
    target_frame = ReferenceFrame.declared(
        ("dicom-frame-of-reference", "1.2.826.0.1.3680043.8.498.99"),
        CoordinateSystem(
            ("L", "P", "S"),
            ("mm", "mm", "mm"),
            vocabulary=anatomy,
            orientation=("right-to-left", "anterior-to-posterior", "inferior-to-superior"),
        ),
    )
    row_direction = np.array([0.0, 0.0, -1.0])
    column_direction = np.array([0.6, 0.8, 0.0])
    normal = np.cross(row_direction, column_direction)
    assert_allclose(normal, [0.8, -0.6, 0.0], rtol=0, atol=ATOL)
    row_spacing, column_spacing = 0.5, 0.75
    transform = AffineTransform.from_matrix(
        target=target_frame,
        source=ArrayCoordinates(("slice", "row", "column"), ("mm", "1", "1")),
        matrix=np.column_stack(
            (normal, row_spacing * column_direction, column_spacing * row_direction)
        ),
        translation=(-10.0, 20.0, 30.0),
    )
    point = transform_named(transform, {"slice": 5.0, "row": 3.0, "column": 2.0})
    assert_allclose(point["L"], -5.1, rtol=0, atol=ATOL)
    assert_allclose(point["P"], 18.2, rtol=0, atol=ATOL)
    assert_allclose(point["S"], 28.5, rtol=0, atol=ATOL)

    # Acceptance case A3: nonuniform offsets (0, 2, 5) mm are read as offsets. The third
    # slice sits at 5 mm, not at twice the gap between the first two.
    stack = transform_named(transform, {"slice": [0.0, 2.0, 5.0], "row": 0.0, "column": 0.0})
    assert_allclose(stack["L"], [-10.0, -8.4, -6.0], rtol=0, atol=ATOL)
    assert_allclose(stack["P"], [20.0, 18.8, 17.0], rtol=0, atol=ATOL)
    assert_allclose(stack["S"], [30.0, 30.0, 30.0], rtol=0, atol=ATOL)


def test_ngff_channel_example_reaches_the_published_point() -> None:
    """Acceptance case N1: the NGFF 2-D affine with an independent channel axis.

    The published example is ``y = j + 2*i + 3`` and ``x = 4*j + 5*i + 6``. After the
    crop ``isel(j=slice(10, 30, 2))``, new row ``r = 1`` is old ``j = 12``, and the point
    ``(r=1, i=2)`` must land on ``(y=19, x=64)``.

    This is the arithmetic from the specification example, not a test of NGFF metadata
    encoding or of any reader. The target axes are dimensionless because the cited
    example declares no units.
    """
    target_frame = ReferenceFrame.local(CoordinateSystem(("y", "x"), ("1", "1")))
    stored = AffineTransform.from_matrix(
        target=target_frame,
        source=ArrayCoordinates(("j", "i"), ("1", "1")),
        matrix=((1.0, 2.0), (4.0, 5.0)),
        translation=(3.0, 6.0),
    )
    assert "c" not in stored.source.axes
    retained = transform_named(stored, {"j": 12.0, "i": 2.0})
    assert_allclose(retained["y"], 19.0, rtol=0, atol=ATOL)
    assert_allclose(retained["x"], 64.0, rtol=0, atol=ATOL)

    # Reusing the stored affine on a fresh zero-based crop index is the error the design
    # calls out: it silently relocates the sample.
    naive = transform_named(stored, {"j": 1.0, "i": 2.0})
    assert_allclose(naive["y"], 8.0, rtol=0, atol=ATOL)
    assert_allclose(naive["x"], 20.0, rtol=0, atol=ATOL)

    # An adapter that rebases labels must compose the crop into the mapping instead:
    # y = 13 + 2r + 2i, x = 46 + 8r + 5i.
    composed = AffineTransform.from_matrix(
        target=target_frame,
        source=ArrayCoordinates(("r", "i"), ("1", "1")),
        matrix=((2.0, 2.0), (8.0, 5.0)),
        translation=(13.0, 46.0),
    )
    rebased = transform_named(composed, {"r": 1.0, "i": 2.0})
    assert_allclose(rebased["y"], 19.0, rtol=0, atol=ATOL)
    assert_allclose(rebased["x"], 64.0, rtol=0, atol=ATOL)
    # Agreeing at a shared point is not equality: the two charts differ structurally.
    assert stored != composed
