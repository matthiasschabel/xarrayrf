"""Affine registration of the tour's subject T1w to the MNI152NLin2009cAsym template.

Registration is not xarrayrf's job; this script records how the tour's MNI-to-subject affine was
computed so it can be reproduced. It prints the transform in RAS millimetres, the coordinate
system xarrayrf's NIfTI adapter uses, ready to paste into examples/tour_data.py.

Usage: uv run --no-project --with SimpleITK python tools/register_tour_subject.py SUBJECT TEMPLATE
(SUBJECT: OpenNeuro ds000001 sub-01_T1w.nii.gz; TEMPLATE: tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz)
"""

from __future__ import annotations

import sys

import numpy as np
import SimpleITK as sitk  # type: ignore[import-not-found]  # noqa: N813 (conventional alias)


def main() -> int:
    subject_path, template_path = sys.argv[1:3]
    fixed = sitk.ReadImage(template_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(subject_path, sitk.sitkFloat32)

    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.05, seed=1)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=300, relaxationFactor=0.7
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    registration.SetInitialTransform(initial, inPlace=False)
    result = registration.Execute(fixed, moving)
    affine = sitk.AffineTransform(
        result.GetNthTransform(0) if result.GetName() == "CompositeTransform" else result
    )

    # ITK maps fixed (template) points to moving (subject) points in LPS: p' = A (p - c) + c + t.
    matrix_lps = np.array(affine.GetMatrix()).reshape(3, 3)
    centre = np.array(affine.GetCenter())
    offset_lps = centre + np.array(affine.GetTranslation()) - matrix_lps @ centre
    flip = np.diag([-1.0, -1.0, 1.0])  # LPS <-> RAS
    matrix_ras = flip @ matrix_lps @ flip
    offset_ras = flip @ offset_lps
    print(f"final metric {registration.GetMetricValue():.4f}", file=sys.stderr)
    np.set_printoptions(precision=6, suppress=True)
    print("MNI_TO_SUBJECT_MATRIX =", repr(matrix_ras.round(6).tolist()))
    print("MNI_TO_SUBJECT_TRANSLATION =", repr(offset_ras.round(4).tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
