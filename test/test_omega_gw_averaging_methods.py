"""Tests for omega_gw_averaging_method options."""
import numpy as np
import pytest
from gw_eccentricity import measure_eccentricity, get_available_methods
from conftest import get_seob_datadict


AVERAGING_METHODS = [
    "orbit_averaged_omega_gw",
    "mean_of_extrema_interpolants",
    "omega_gw_zeroecc",
]


@pytest.fixture(scope="module")
def dataDict():
    return get_seob_datadict(include_zero_ecc=True)


def _f_mid(dataDict):
    """Return the median GW frequency of the waveform."""
    omega_gw = np.abs(np.diff(np.unwrap(np.angle(
        dataDict["hlm"][(2, 2)]))) / np.diff(dataDict["t"]))
    return float(np.median(omega_gw)) / (2 * np.pi)


@pytest.mark.parametrize("avg_method", AVERAGING_METHODS)
@pytest.mark.parametrize("ecc_method", get_available_methods())
def test_averaging_method_returns_finite(dataDict, ecc_method, avg_method):
    """Every combination of ecc method and averaging method returns finite eccentricity
    when using fref_in (the path where omega_gw_averaging_method is actually used)."""
    result = measure_eccentricity(
        fref_in=_f_mid(dataDict),
        method=ecc_method,
        dataDict=dataDict,
        extra_kwargs={"omega_gw_averaging_method": avg_method})
    assert np.isfinite(result["eccentricity"]).all(), (
        f"Non-finite eccentricity for ecc_method={ecc_method}, "
        f"avg_method={avg_method}")


@pytest.mark.parametrize("avg_method", AVERAGING_METHODS)
def test_averaging_methods_agree(dataDict, avg_method):
    """All three averaging methods return eccentricity within 10% of orbit_averaged
    when using fref_in."""
    f_mid = _f_mid(dataDict)

    ecc_ref = measure_eccentricity(
        fref_in=f_mid,
        method="Amplitude",
        dataDict=dataDict,
        extra_kwargs={"omega_gw_averaging_method": "orbit_averaged_omega_gw"})["eccentricity"]

    ecc = measure_eccentricity(
        fref_in=f_mid,
        method="Amplitude",
        dataDict=dataDict,
        extra_kwargs={"omega_gw_averaging_method": avg_method})["eccentricity"]

    np.testing.assert_allclose(
        ecc, ecc_ref, rtol=0.10,
        err_msg=f"avg_method={avg_method} differs >10% from orbit_averaged_omega_gw")
