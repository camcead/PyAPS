"""apply_redshift_to_fwhm_corrected: a scalar wavelength must give a float on NumPy 2.5.

The redshifted LSF function did ``float(func(np.atleast_1d(wave)) / (1 + z))``: the L1 LSF splines return a
1-element array, and ``float()`` of a 1-element array is an error on NumPy 2.5 ("only 0-dimensional arrays
can be converted to Python scalars"). MOSExGalEMIPPXF.prepare_emission_template calls ``LSF_Data(line_wave)``
for every emission line, caught the TypeError ("ERROR in batch template creation") and every fibre of the
MOS EMIPPXF stage failed (OB 26133: 'emi kinematics analysis failed for 553 targets', FLUX_CLEAN_EMI all NaN,
so the MOS line strengths had nothing to measure).
"""
import numpy as np
import pytest

from PyAPS.aps_utils import apply_redshift_to_fwhm_corrected


def _spline_like(x):
    """What the L1 LSF functions do: always return an array, also for a 1-element input."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return 2.0 + 0.0002 * x


@pytest.mark.parametrize("wave", [3000.0, np.float64(5000.0), np.array(4000.0), np.array([4500.0])])
def test_scalar_wavelength_gives_a_float(wave):
    f = apply_redshift_to_fwhm_corrected(_spline_like, 0.5)
    out = f(wave)
    assert isinstance(out, float)
    assert out == pytest.approx(float(_spline_like(np.ravel(wave)[0])[0]) / 1.5)


def test_array_wavelength_gives_an_array_and_the_list_form_works():
    waves = np.array([3000.0, 4000.0, 5000.0])
    f = apply_redshift_to_fwhm_corrected(_spline_like, [0.1, 0.0])
    assert np.allclose(f(waves), _spline_like(waves) / 1.1)
    fl = apply_redshift_to_fwhm_corrected([_spline_like, _spline_like], 0.0)
    assert isinstance(fl, list) and isinstance(fl[1](3000.0), float)


def test_the_per_line_pattern_of_prepare_emission_template():
    f = apply_redshift_to_fwhm_corrected(_spline_like, 0.0)
    lines = np.array([3727.0, 4861.0, 6563.0])
    arr = np.array([f(lw) for lw in lines])
    assert arr.shape == (3,)
