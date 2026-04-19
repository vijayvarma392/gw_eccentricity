"""Tests for the use_only_these_many_orbits option.

For model waveforms (SEOBNRv5EHM), results using a short segment should agree
with the full-waveform result to within 0.01% (per the documentation).
"""
import numpy as np
import pytest
from gw_eccentricity import measure_eccentricity, get_available_methods
from conftest import get_seob_datadict


NUM_ORBITS = 10
INTERPOLATION_METHODS = ["spline", "rational_fit"]
# model waveforms: short-segment vs full-waveform agree to within 0.01%
SEGMENT_VS_FULL_RTOL = 1e-4


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
    """Short segment gives eccentricity within 0.01% of full waveform (tref_in)."""
    t = dataDict["t"]
    t_mid = t[len(t) // 2]
    extra_kwargs = {"omega_gw_extrema_interpolation_method": interp_method}

    ecc_full = measure_eccentricity(
        tref_in=t_mid,
        method=method,
        dataDict=dataDict,
        extra_kwargs=extra_kwargs)["eccentricity"]

    ecc_seg = measure_eccentricity(
        tref_in=t_mid,
        method=method,
        dataDict=dataDict,
        extra_kwargs={**extra_kwargs,
                      "use_only_these_many_orbits": NUM_ORBITS})["eccentricity"]

    np.testing.assert_allclose(
        ecc_seg, ecc_full, rtol=SEGMENT_VS_FULL_RTOL,
        err_msg=f"use_only_these_many_orbits={NUM_ORBITS} disagrees with "
                f"full waveform by more than {SEGMENT_VS_FULL_RTOL:.0e} "
                f"for method={method}, interp={interp_method}, tref_in")


@pytest.mark.parametrize("interp_method", INTERPOLATION_METHODS)
@pytest.mark.parametrize("method", get_available_methods())
def test_segment_agrees_with_full_fref_in(dataDict, method, interp_method):
    """Short segment gives eccentricity within 0.01% of full waveform (fref_in)."""
    t = dataDict["t"]
    t_mid = t[len(t) // 2]
    extra_kwargs = {"omega_gw_extrema_interpolation_method": interp_method}

    result_full = measure_eccentricity(
        tref_in=t_mid,
        method=method,
        dataDict=dataDict,
        extra_kwargs=extra_kwargs)
    fref = result_full["gwecc_object"].compute_orbit_averaged_omega_gw_between_extrema(
        t_mid) / (2 * np.pi)

    ecc_full = measure_eccentricity(
        fref_in=fref,
        method=method,
        dataDict=dataDict,
        extra_kwargs=extra_kwargs)["eccentricity"]

    ecc_seg = measure_eccentricity(
        fref_in=fref,
        method=method,
        dataDict=dataDict,
        extra_kwargs={**extra_kwargs,
                      "use_only_these_many_orbits": NUM_ORBITS})["eccentricity"]

    np.testing.assert_allclose(
        ecc_seg, ecc_full, rtol=SEGMENT_VS_FULL_RTOL,
        err_msg=f"use_only_these_many_orbits={NUM_ORBITS} disagrees with "
                f"full waveform by more than {SEGMENT_VS_FULL_RTOL:.0e} "
                f"for method={method}, interp={interp_method}, fref_in")
