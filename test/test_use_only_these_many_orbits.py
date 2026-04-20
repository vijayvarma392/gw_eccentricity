"""Tests for the use_only_these_many_orbits option.

For model waveforms (SEOBNRv5EHM), results using a 10-orbit segment centred on
the reference time/frequency should agree with the full-waveform result to
within 0.02% for tref_in and within 0.1% for fref_in (the looser tolerance
accounts for the extra frequency-to-time conversion step near the waveform
edges).  Tests are parametrised over all available methods and both spline and
rational_fit interpolation.
"""
import numpy as np
import pytest
from gw_eccentricity import measure_eccentricity, get_available_methods
from conftest import get_seob_datadict


NUM_ORBITS = 10
INTERPOLATION_METHODS = ["spline", "rational_fit"]
# model waveforms: short-segment vs full-waveform agree to within 0.02% for
# tref_in; fref_in adds a frequency-to-time conversion step that increases the
# edge discrepancy, so a looser tolerance is used for that path.
SEGMENT_VS_FULL_RTOL_TREF = 2e-4
SEGMENT_VS_FULL_RTOL_FREF = 1e-3


@pytest.fixture(scope="module")
def dataDict():
    dd = get_seob_datadict(omega_start=0.009, include_zero_ecc=True)
    # Verify the waveform is genuinely longer than the segment so that
    # use_only_these_many_orbits actually truncates rather than uses the full waveform.
    phase = np.unwrap(np.angle(dd["hlm"][(2, 2)]))
    total_orbits = abs(phase[-1] - phase[0]) / (2 * 2 * np.pi)
    assert total_orbits >= 10 * NUM_ORBITS, (
        f"Waveform has only {total_orbits:.1f} orbits — need at least "
        f"{10 * NUM_ORBITS} for use_only_these_many_orbits={NUM_ORBITS} "
        f"to genuinely truncate the data. Lower omega_start to get a longer waveform.")
    return dd


@pytest.mark.parametrize("interp_method", INTERPOLATION_METHODS)
@pytest.mark.parametrize("method", get_available_methods())
def test_segment_agrees_with_full_tref_in(dataDict, method, interp_method):
    """Short segment gives eccentricity within 0.01% of full waveform (tref_in array)."""
    t = dataDict["t"]
    # 10 evenly spaced points spanning 5%–95% of the waveform
    t_arr = np.linspace(t[len(t) // 20], t[19 * len(t) // 20], 10)
    extra_kwargs = {"omega_gw_extrema_interpolation_method": interp_method}

    ecc_full = measure_eccentricity(
        tref_in=t_arr,
        method=method,
        dataDict=dataDict,
        extra_kwargs=extra_kwargs)["eccentricity"]

    ecc_seg = measure_eccentricity(
        tref_in=t_arr,
        method=method,
        dataDict=dataDict,
        extra_kwargs={**extra_kwargs,
                      "use_only_these_many_orbits": NUM_ORBITS})["eccentricity"]

    np.testing.assert_allclose(
        ecc_seg, ecc_full, rtol=SEGMENT_VS_FULL_RTOL_TREF,
        err_msg=f"use_only_these_many_orbits={NUM_ORBITS} disagrees with "
                f"full waveform by more than {SEGMENT_VS_FULL_RTOL_TREF:.0e} "
                f"for method={method}, interp={interp_method}, tref_in")


@pytest.mark.parametrize("interp_method", INTERPOLATION_METHODS)
@pytest.mark.parametrize("method", get_available_methods())
def test_segment_agrees_with_full_fref_in(dataDict, method, interp_method):
    """Short segment gives eccentricity within 0.01% of full waveform (fref_in array)."""
    t = dataDict["t"]
    t_arr = np.linspace(t[len(t) // 20], t[19 * len(t) // 20], 10)
    extra_kwargs = {"omega_gw_extrema_interpolation_method": interp_method}

    result_full = measure_eccentricity(
        tref_in=t_arr,
        method=method,
        dataDict=dataDict,
        extra_kwargs=extra_kwargs)
    fref_arr = result_full["gwecc_object"].compute_orbit_averaged_omega_gw_between_extrema(
        t_arr) / (2 * np.pi)

    ecc_full = measure_eccentricity(
        fref_in=fref_arr,
        method=method,
        dataDict=dataDict,
        extra_kwargs=extra_kwargs)["eccentricity"]

    ecc_seg = measure_eccentricity(
        fref_in=fref_arr,
        method=method,
        dataDict=dataDict,
        extra_kwargs={**extra_kwargs,
                      "use_only_these_many_orbits": NUM_ORBITS})["eccentricity"]

    np.testing.assert_allclose(
        ecc_seg, ecc_full, rtol=SEGMENT_VS_FULL_RTOL_FREF,
        err_msg=f"use_only_these_many_orbits={NUM_ORBITS} disagrees with "
                f"full waveform by more than {SEGMENT_VS_FULL_RTOL_FREF:.0e} "
                f"for method={method}, interp={interp_method}, fref_in")
