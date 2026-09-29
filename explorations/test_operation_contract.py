"""A bounded operation contract: required outcomes stated beside stock-xarray observations.

These cases describe the scalar-carrier observations that remain relevant under route (ii);
missing-guard cases belonged only to the rejected route (i). A strict xfail records that the
deliberately insufficient scalar guard does not produce it. The mark is narrow on purpose: it accepts only
a missing refusal, so an unexpected exception, or a refusal of the wrong type, is a failure
rather than something that could be mistaken for conformance.

The placement cases evaluate the real ``AffineTransform`` on the coordinates an operation
actually retained and compare against independently calculated world points, so they check
sample placement rather than the presence of a marker. They observe the coordinate model of
stock xarray. None of this certifies reference-frame support, which does not exist.
"""

from collections.abc import Callable
from dataclasses import dataclass

import dask.array as da
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from dask.callbacks import Callback
from scalar_binding_probe import MARKER, embedded_plane, framed, plane_world

TOLERANCE = {"rtol": 0.0, "atol": 1e-12}
"""A floating-point arithmetic check in declared millimetres, not a physical tolerance."""


@dataclass(frozen=True)
class ContractCase:
    """One required rejection, today's stock observation, and the operation that shows it."""

    name: str
    required: str
    observed: str
    operate: Callable[[], object]
    error: type[Exception] = ValueError


REJECTION_CASES = (
    ContractCase(
        name="protected scalar contradicted by an unframed operand",
        required="ValueError; a bound fixed coordinate is protected against silent dropping",
        observed="the conflicting non-index scalar is dropped and the plane position is lost",
        operate=lambda: framed().isel(y=0) + xr.DataArray(np.ones(4), dims="x", coords={"y": 2}),
    ),
    ContractCase(
        name="join=override on compatible frames",
        required="ValueError in v1 even here; a caller override does not replace the checks",
        observed="override accepts compatible framed inputs without a geometry refusal",
        operate=lambda: xr.align(framed(), framed(), join="override"),
    ),
    ContractCase(
        name="isel drop=True removing a mapped coordinate",
        required="ValueError; the fixed term cannot be preserved once its coordinate is gone",
        observed="the declaration survives while the input coordinate it needs does not",
        operate=lambda: framed().isel(y=1, drop=True),
    ),
    ContractCase(
        name="unsupported spatial reduction",
        required="ValueError in v1; a reduction cannot silently retain stale geometry",
        observed="spatial mean preserves a declaration whose input coordinate is gone",
        operate=lambda: framed().mean("y"),
    ),
    ContractCase(
        name="unsupported native coarsening",
        required="ValueError in v1 until sampling and support semantics are implemented",
        observed="coarsening returns a declaration with no runtime guard",
        operate=lambda: framed().coarsen(x=2).mean(),
    ),
    ContractCase(
        name="unsupported native concatenation",
        required="ValueError in v1 until binding reconstruction is implemented",
        observed="concatenation returns a valid-looking guarded declaration without preflight",
        operate=lambda: xr.concat([framed(), framed()], dim="batch"),
    ),
)


def _expected_failure(case: ContractCase) -> pytest.MarkDecorator:
    """Narrow the xfail to a missing refusal, so a wrong exception type still fails."""
    return pytest.mark.xfail(strict=True, raises=pytest.fail.Exception, reason=case.observed)


REJECTION_PARAMS = [
    pytest.param(case, id=case.name, marks=_expected_failure(case)) for case in REJECTION_CASES
]


@pytest.mark.parametrize("case", REJECTION_PARAMS)
def test_required_rejection(case):
    """Assert the outcome the design requires; the xfail records the guard's known deficiency."""
    with pytest.raises(case.error):
        case.operate()


def test_drop_true_observation_retains_a_declaration_without_its_input():
    """Stock observation of the release-invariant counterexample above, stated on its own.

    The result looks structurally fine and is not: the declaration names an input coordinate
    the array no longer carries. Accidentally losing the whole declaration would also violate
    the release invariant.
    """
    result = framed().isel(y=1, drop=True)
    assert MARKER in result.coords
    assert "y" not in result.coords


def _dataset() -> xr.Dataset:
    """One framed variable beside an unrelated variable over the same dimensions."""
    coords = {"y": [0, 2, 4], "x": [0, 3, 6, 9]}
    sibling = xr.DataArray(np.zeros((3, 4)), dims=("y", "x"), coords=coords)
    return xr.Dataset({"image": framed(), "sibling": sibling})


def test_owning_variable_extraction_keeps_its_declaration():
    """Stock observation: the owner does keep the declaration it actually owns."""
    assert MARKER in _dataset()["image"].coords


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="a dimensionless scalar coordinate is added to every same-dimensional variable",
)
def test_unrelated_variable_must_not_inherit_the_declaration():
    """Required: association follows the variable, not the presence of a scalar coordinate."""
    assert MARKER not in _dataset()["sibling"].coords


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="coordinate extraction inherits the scalar binding despite not being a measurement",
)
def test_coordinate_extraction_must_be_unframed():
    assert MARKER not in framed().x.coords


def _placed_lazily(operation: Callable[[xr.DataArray], xr.DataArray]) -> xr.DataArray:
    """Apply a geometry-only operation to chunked pixels and prove none were computed."""
    image = embedded_plane()
    lazy = image.copy(data=da.from_array(image.data, chunks=(2, 2)))
    tasks: list[object] = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        result = operation(lazy)
    assert isinstance(result.data, da.Array)
    assert not tasks
    assert MARKER in result.xindexes
    return result


def test_crop_stride_keeps_the_retained_column_labels():
    """Columns 2 and 4 keep their own world positions; nothing is rebased to a new origin."""
    world = plane_world(_placed_lazily(lambda a: a.isel(column=slice(1, None, 2))))
    npt.assert_allclose(world["L"], [[11.0, 12.0]] * 3, **TOLERANCE)
    npt.assert_allclose(world["P"], [[20.25] * 2, [20.5] * 2, [20.75] * 2], **TOLERANCE)
    npt.assert_allclose(world["S"], [[-4.5, -4.0]] * 3, **TOLERANCE)


def test_gather_reorders_samples_without_moving_them():
    """A gather of rows 3 then 1 reports those rows' positions, in that order."""
    world = plane_world(_placed_lazily(lambda a: a.isel(row=[2, 0])))
    npt.assert_allclose(world["P"], [[20.75] * 4, [20.25] * 4], **TOLERANCE)
    npt.assert_allclose(world["L"], [[10.5, 11.0, 11.5, 12.0]] * 2, **TOLERANCE)
    npt.assert_allclose(world["S"], [[-4.75, -4.5, -4.25, -4.0]] * 2, **TOLERANCE)


def test_transpose_keeps_named_coordinate_placement():
    """Compare retained coordinates in canonical row/column order, not output storage order."""
    result = _placed_lazily(lambda a: a.transpose())
    assert result.dims == ("column", "row")
    world = plane_world(result)
    npt.assert_allclose(world["P"], [[20.25] * 4, [20.5] * 4, [20.75] * 4], **TOLERANCE)
    npt.assert_allclose(world["S"], [[-4.75, -4.5, -4.25, -4.0]] * 3, **TOLERANCE)


def test_scalar_selection_retains_the_fixed_plane_term():
    """Row 2 survives as a scalar coordinate, so its fixed world offset survives with it."""
    result = _placed_lazily(lambda a: a.isel(row=1))
    assert result.coords["row"].item() == 2
    world = plane_world(result)
    npt.assert_allclose(world["P"], [20.5] * 4, **TOLERANCE)
    npt.assert_allclose(world["L"], [10.5, 11.0, 11.5, 12.0], **TOLERANCE)
    npt.assert_allclose(world["S"], [-4.75, -4.5, -4.25, -4.0], **TOLERANCE)
