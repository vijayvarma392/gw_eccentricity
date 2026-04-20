"""Tests for omega_gw_extrema_interpolation_method="rational_fit"."""
import numpy as np
from gw_eccentricity import measure_eccentricity, get_available_methods
from conftest import get_seob_datadict


def test_rational_fit_gives_monotonic_eccentricity():
    """rational_fit should give monotonically decreasing egw(t) where spline does not.

    The q=5, chi1z=0.4, chi2z=0.3, ecc=0.1, leob=1.7 SEOBNRv5EHM waveform is
    a known case where spline interpolation of the omega extrema introduces
    artificial oscillations that produce a non-monotonic eccentricity, while
    rational_fit remains monotonic.
    """
    dataDict = get_seob_datadict(include_zero_ecc=True)
    tref_in = dataDict["t"]

    # --- spline: expected to be non-monotonic ---
    gwecc_spline = measure_eccentricity(
        tref_in=tref_in,
        method="ResidualAmplitude",
        dataDict=dataDict,
        num_orbits_to_exclude_before_merger=0,
        extra_kwargs={"omega_gw_extrema_interpolation_method": "spline"})
    obj_spline = gwecc_spline["gwecc_object"]
    obj_spline.check_monotonicity_and_convexity()
    assert np.any(obj_spline.decc_dt_for_checks > 0), (
        "Spline should produce non-monotonic eccentricity for q=5, "
        "chi1z=0.4, chi2z=0.3 SEOBNRv5EHM waveform")

    # --- rational_fit: expected to be monotonically decreasing ---
    gwecc_ratfit = measure_eccentricity(
        tref_in=tref_in,
        method="ResidualAmplitude",
        dataDict=dataDict,
        num_orbits_to_exclude_before_merger=0,
        extra_kwargs={"omega_gw_extrema_interpolation_method": "rational_fit"})
    obj_ratfit = gwecc_ratfit["gwecc_object"]
    obj_ratfit.check_monotonicity_and_convexity()
    assert np.all(obj_ratfit.decc_dt_for_checks <= 0), (
        "rational_fit should produce monotonically decreasing eccentricity")


def test_rational_fit_all_methods():
    """rational_fit runs without error for all available methods, scalar and array tref."""
    dataDict = get_seob_datadict(include_zero_ecc=True)
    t = dataDict["t"]
    t_mid = t[len(t) // 2]
    # five evenly spaced times in the safe interior of the waveform
    t_arr = t[len(t) // 4 : 3 * len(t) // 4 : len(t) // 20]

    available_methods = get_available_methods()
    for method in available_methods:
        for tref_in in (t_mid, t_arr):
            result = measure_eccentricity(
                tref_in=tref_in,
                method=method,
                dataDict=dataDict,
                extra_kwargs={"omega_gw_extrema_interpolation_method": "rational_fit"})
            assert np.isfinite(result["eccentricity"]).all(), (
                f"rational_fit returned non-finite eccentricity for "
                f"method={method}, tref_in={'scalar' if np.isscalar(tref_in) else 'array'}")


def test_rational_fit_at_frequency():
    """rational_fit works for scalar and array fref."""
    dataDict = get_seob_datadict(include_zero_ecc=True)

    # frequency array spanning the mid-band of the waveform
    omega_gw = np.abs(np.diff(np.unwrap(np.angle(
        dataDict["hlm"][(2, 2)]))) / np.diff(dataDict["t"]))
    f_median = float(np.median(omega_gw)) / (2 * np.pi)
    f_arr = np.linspace(0.8 * f_median, 1.2 * f_median, 5)

    available_methods = get_available_methods()
    for method in available_methods:
        for fref_in in (f_median, f_arr):
            result = measure_eccentricity(
                fref_in=fref_in,
                method=method,
                dataDict=dataDict,
                extra_kwargs={"omega_gw_extrema_interpolation_method": "rational_fit"})
            assert np.isfinite(result["eccentricity"]).all(), (
                f"rational_fit returned non-finite eccentricity for "
                f"method={method}, fref_in={'scalar' if np.isscalar(fref_in) else 'array'}")
