"""Tests for precessing=True with frame="inertial" and frame="coprecessing"."""
import os
import sys
import json
import numpy as np
import pytest
from gw_eccentricity import measure_eccentricity
from gw_eccentricity.load_data import (
    load_waveform, get_coprecessing_data_dict, download_sxs_waveform)

# ---------------------------------------------------------------------------
# SXS test cases
#   SXS:BBH:2859  — nearly quasi-circular, precessing  [active]
#   SXS:BBH:xxxx  — high-eccentricity, precessing      [placeholder; all
#                   eccentric tests skipped until a suitable simulation is
#                   available in the public SXS catalog]
# ---------------------------------------------------------------------------

DATA_DIR = "/tmp/gwecc_test_data"
SXS_CASES = {
    "eccentric":     ("SXS:BBH:xxxx", 0),
    "quasicircular": ("SXS:BBH:2859", 4),
}
ALL_ELL2_MODES = [(2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]


def _data_dir(sxs_id, lev):
    sxs_id_safe = sxs_id.replace(":", "_")
    return os.path.join(DATA_DIR, sxs_id_safe, f"Lev{lev}")


def _load(sxs_id, lev):
    """Load all ell=2 modes from a downloaded SXS waveform."""
    return load_waveform(
        origin="SXSCatalog",
        data_dir=_data_dir(sxs_id, lev),
        mode_array=ALL_ELL2_MODES,
        num_orbits_to_remove_as_junk=4)


def _ref_ecc(sxs_id, lev):
    """Read reference eccentricity from the downloaded metadata.json."""
    path = os.path.join(_data_dir(sxs_id, lev), "metadata.json")
    with open(path) as f:
        return float(json.load(f)["reference_eccentricity"])


# ---------------------------------------------------------------------------
# Fixtures — download data on first run, skip only if sxs_id is a placeholder
# ---------------------------------------------------------------------------

def _ensure_data(sxs_id, lev):
    """Download SXS waveform if not already present; skip if sxs_id is a placeholder
    or if running on Python < 3.10 where the sxs download API is not fully supported."""
    if "xxxx" in sxs_id:
        pytest.skip(
            f"SXS ID {sxs_id} is a placeholder — no suitable simulation available yet")
    if not os.path.isfile(os.path.join(_data_dir(sxs_id, lev), "Strain_N2.h5")):
        if sys.version_info < (3, 10):
            pytest.skip("NR precessing tests require Python >= 3.10 for SXS download support")
        download_sxs_waveform(sxs_id, lev, DATA_DIR)


def _make_fixture(key):
    """Return a module-scoped fixture that downloads and loads one SXS case."""
    @pytest.fixture(scope="module")
    def _fixture():
        sxs_id, lev = SXS_CASES[key]
        _ensure_data(sxs_id, lev)
        return _load(sxs_id, lev), _ref_ecc(sxs_id, lev)
    return _fixture


eccentric_dataDict = _make_fixture("eccentric")
quasicircular_dataDict = _make_fixture("quasicircular")


@pytest.fixture(scope="module", params=list(SXS_CASES.keys()))
def sxs_dataDict(request):
    """Module-scoped dataDict for each SXS case; downloads data if not present."""
    key = request.param
    sxs_id, lev = SXS_CASES[key]
    _ensure_data(sxs_id, lev)
    return key, _load(sxs_id, lev), _ref_ecc(sxs_id, lev)


# ---------------------------------------------------------------------------
# Test: frame="inertial" triggers internal coprecessing transform
# ---------------------------------------------------------------------------

def test_inertial_frame_gives_finite_eccentricity(sxs_dataDict):
    """measure_eccentricity with precessing=True, frame='inertial' returns finite result."""
    key, dataDict, _ = sxs_dataDict
    t = dataDict["t"]
    t_mid = t[len(t) // 2]

    result = measure_eccentricity(
        tref_in=t_mid,
        method="AmplitudeFits",
        dataDict=dataDict,
        precessing=True,
        frame="inertial")
    assert np.isfinite(result["eccentricity"]).all(), (
        f"Non-finite eccentricity for {key} with frame='inertial'")


# ---------------------------------------------------------------------------
# Test: frame="coprecessing" (pre-rotated modes) matches frame="inertial"
# ---------------------------------------------------------------------------

def test_coprecessing_frame_matches_inertial(sxs_dataDict):
    """frame='coprecessing' and frame='inertial' agree to within numerical precision."""
    key, dataDict, _ = sxs_dataDict
    t = dataDict["t"]
    t_mid = t[len(t) // 2]

    result_inertial = measure_eccentricity(
        tref_in=t_mid,
        method="AmplitudeFits",
        dataDict=dataDict,
        precessing=True,
        frame="inertial")

    coprec_dataDict = get_coprecessing_data_dict(dataDict)
    result_coprec = measure_eccentricity(
        tref_in=t_mid,
        method="AmplitudeFits",
        dataDict=coprec_dataDict,
        precessing=True,
        frame="coprecessing")

    np.testing.assert_allclose(
        result_coprec["eccentricity"],
        result_inertial["eccentricity"],
        rtol=1e-10,
        err_msg=f"coprecessing vs inertial mismatch for {key}")


# ---------------------------------------------------------------------------
# Test: spin_filter is exercised for quasicircular precessing data.
#       Eccentricity evolution should be monotonically decreasing and the
#       initial value should be within a factor of 2 of the metadata reference.
#
#       NOTE: spin_filter is not triggered for high-eccentricity waveforms
#       because the eccentricity signal dominates spin-induced oscillations.
# ---------------------------------------------------------------------------

def test_spin_filter_gives_monotonic_eccentricity(quasicircular_dataDict):
    """For quasicircular precessing NR data (SXS:BBH:2859), verify that:

    1. The eccentricity evolution is monotonically decreasing (spin_filter path
       exercised — spin-induced oscillations are removed before measurement).
    2. The initial eccentricity is within a factor of 2 of the SXS metadata
       reference value.
    """
    dataDict, ecc_ref = quasicircular_dataDict
    t = dataDict["t"]

    result = measure_eccentricity(
        tref_in=t,
        method="AmplitudeFits",
        dataDict=dataDict,
        precessing=True,
        frame="inertial")

    ecc = result["eccentricity"]
    assert np.isfinite(ecc).all(), "Non-finite eccentricity for quasicircular case"

    # use the built-in monotonicity check which populates decc_dt_for_checks
    obj = result["gwecc_object"]
    obj.check_monotonicity_and_convexity()
    assert np.all(obj.decc_dt_for_checks <= 0), (
        "Eccentricity is not monotonically decreasing for quasicircular case")

    # initial eccentricity vs SXS metadata reference (within factor of 2)
    ecc_initial = float(ecc[0])
    assert 0.5 * ecc_ref <= ecc_initial <= 2.0 * ecc_ref, (
        f"Initial eccentricity {ecc_initial:.6f} not within factor of 2 "
        f"of metadata reference {ecc_ref:.6f}")


@pytest.mark.skip(
    reason="No suitable high-eccentricity precessing SXS simulation is currently "
           "available in the public catalog. TODO: re-enable and tighten to rtol=0.05 "
           "once such a simulation is added.")
def test_eccentric_precessing_eccentricity_value(eccentric_dataDict):
    """For high-eccentricity precessing NR data, verify that the initial
    eccentricity is within 5% of the SXS metadata reference value.

    spin_filter is not triggered for high-eccentricity waveforms; only the
    eccentricity value is checked here.
    """
    dataDict, ecc_ref = eccentric_dataDict
    t = dataDict["t"]

    result = measure_eccentricity(
        tref_in=t,
        method="AmplitudeFits",
        dataDict=dataDict,
        precessing=True,
        frame="inertial")

    ecc = result["eccentricity"]
    assert np.isfinite(ecc).all(), "Non-finite eccentricity for eccentric case"

    ecc_initial = float(ecc[0])
    np.testing.assert_allclose(
        ecc_initial, ecc_ref, rtol=0.05,
        err_msg=f"Initial eccentricity {ecc_initial:.4f} differs from "
                f"metadata reference {ecc_ref:.4f} by more than 5%")
