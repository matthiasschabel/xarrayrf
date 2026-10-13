"""Import an ANTs registration as one chain with its declared inverse.

``antsRegistration`` writes an affine (``0GenericAffine.mat``) and, for deformable stages, a warp
(``1Warp.nii.gz``) and its inverse (``1InverseWarp.nii.gz``). The forward transforms map a fixed
point through the warp, then the affine, to the moving image; the inverse warp is ANTs' own
declared inverse. Install ``xarrayrf[ants]``.
"""

from __future__ import annotations

from ._import import AntsImport, Report, transform_from_registration

__all__ = ["AntsImport", "Report", "transform_from_registration"]
