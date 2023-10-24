import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity
import numpy as np


def test_set_failures_to_zero():
    """Test that the interface works with set_failures_to_zero for waveforms
    with small or zero ecc.

    In certain situations, the waveform may have zero eccentricity or a very
    small eccentricity, making it difficult for the given method to identify
    any extrema. In cases where such a situation occurs, and if the user has
    set 'set_failures_to_zero' to `True` in the 'extra_kwargs' parameter, both
    the eccentricity and mean anomaly will be set to zero.
    """
    # The Amplitude and Frequency methods usually fail to detect any extrema
    # for eccentricities less than about 1e-3. Therefore, to test whether we
    # are setting eccentricity to zero when using these two methods, we use an
    # EccentricTD waveform with an initial eccentricity of 1e-4. However, since
    # the Residual and the Fits methods can detect extrema for the same
    # eccentricity, we instead use a quasicircular waveform to test the
    # handling of failures due to insufficient extrema for these methods.
    lal_kwargs_ecc = {"approximant": "EccentricTD",
                      "q": 3.0,
                      "chi1": [0.0, 0.0, 0.0],
                      "chi2": [0.0, 0.0, 0.0],
                      "Momega0": 0.01,
                      "ecc": 1e-4,
                      "mean_ano": 0.0,
                      "include_zero_ecc": True}
    lal_kwargs_qc = lal_kwargs_ecc.copy()
    lal_kwargs_qc.update({"approximant": "IMRPhenomT",
                          "ecc": 0.0})
    # Create the dataDict for Amplitude and Frequency method
    dataDict_ecc = load_data.load_waveform(**lal_kwargs_ecc)
    # Create the dataDict for Residual and Fits method
    dataDict_qc = load_data.load_waveform(**lal_kwargs_qc)
    available_methods = gw_eccentricity.get_available_methods()
    # The following will set ecc and mean ano to zero
    # if no extrema are found.
    extra_kwargs = {"set_failures_to_zero": True}

    # We want to test it with both a single reference point
    # as well as an array of reference points
    tref_in = {"scalar": -8000.0,
               "array": np.arange(-8000.0, 0.)}
    fref_in = {"scalar": 0.05,
               "array": np.arange(0.02, 0.05, 0.005)}
    for method in available_methods:
        if method in ["Amplitude", "Frequency"]:
            dataDict = dataDict_ecc
        else:
            dataDict = dataDict_qc
        # test for reference time
        for ref in tref_in:
            gwecc_dict = measure_eccentricity(
                tref_in=tref_in[ref],
                method=method,
                dataDict=dataDict,
                extra_kwargs=extra_kwargs)
            tref_out = gwecc_dict["tref_out"]
            ecc_ref = gwecc_dict["eccentricity"]
            meanano_ref = gwecc_dict["mean_anomaly"]
            np.testing.assert_allclose(
                ecc_ref, 0.0 if ref == "scalar"
                else np.zeros(len(tref_in[ref])))
            np.testing.assert_allclose(
                meanano_ref, 0.0 if ref == "scalar"
                else np.zeros(len(tref_in[ref])))
        # test for reference frequency
        for ref in fref_in:
            gwecc_dict = measure_eccentricity(
                fref_in=fref_in[ref],
                method=method,
                dataDict=dataDict,
                extra_kwargs=extra_kwargs)
            fref_out = gwecc_dict["fref_out"]
            ecc_ref = gwecc_dict["eccentricity"]
            meanano_ref = gwecc_dict["mean_anomaly"]
            np.testing.assert_allclose(
                ecc_ref, 0.0 if ref == "scalar"
                else np.zeros(len(fref_in[ref])))
            np.testing.assert_allclose(
                meanano_ref, 0.0 if ref == "scalar"
                else np.zeros(len(fref_in[ref])))


def test_set_failures_to_zero_for_decayed_eccentricity():
    """Test that the interface works with set_failures_to_zero for tref/fref
    greater than the tmax/fmax.

    In certain situations, the maximum allowed time `tmax` or the maximum
    allowed frequency `fmax` could be much smaller compared to the merger
    time/frequency. Therefore, measurement of eccentricity closer to merger,
    i.e., tref_in > tmax or fref_in > fmax would fail.

    In cases where such a situation occurs, and if the user has
    set 'set_failures_to_zero' to `True` in the 'extra_kwargs' parameter, both
    the eccentricity and mean anomaly will be set to zero.
    """
    # We will use an EccentricTD waveform such that the Amplitude/Frequency
    # method can find a few initial pericenetrs/apoceneters but not all of them
    # so that tmax/fmax will be much earlier than the merger time/frequency.
    lal_kwargs = {"approximant": "EccentricTD",
                  "q": 1.0,
                  "chi1": [0.0, 0.0, 0.0],
                  "chi2": [0.0, 0.0, 0.0],
                  "Momega0": 0.01,
                  "ecc": 0.01,
                  "mean_ano": 0.0,
                  "include_zero_ecc": True}
    # We want to test it with both a single reference point
    # as well as an array of reference points
    tref_in = {"scalar": -2000.0,
               "array": np.arange(-2000.0, 0.)}
    fref_in = {"scalar": 0.01,
               "array": np.arange(0.01, 0.02, 0.005)}
    dataDict = load_data.load_waveform(**lal_kwargs)
    extra_kwargs = {"set_failures_to_zero": True}
    for method in ["Amplitude", "Frequency"]:
        for ref in tref_in:
            gwecc_dict = measure_eccentricity(
                tref_in=tref_in[ref],
                method=method,
                dataDict=dataDict,
                extra_kwargs=extra_kwargs)
            tref_out = gwecc_dict["tref_out"]
            ecc_ref = gwecc_dict["eccentricity"]
            meanano_ref = gwecc_dict["mean_anomaly"]
            np.testing.assert_allclose(
                ecc_ref, 0.0 if ref == "scalar"
                else np.zeros(len(tref_in[ref])))
            np.testing.assert_allclose(
                meanano_ref, 0.0 if ref == "scalar"
                else np.zeros(len(tref_in[ref])))
        # test for reference frequency
        for ref in fref_in:
            gwecc_dict = measure_eccentricity(
                fref_in=fref_in[ref],
                method=method,
                dataDict=dataDict,
                extra_kwargs=extra_kwargs)
            fref_out = gwecc_dict["fref_out"]
            ecc_ref = gwecc_dict["eccentricity"]
            meanano_ref = gwecc_dict["mean_anomaly"]
            np.testing.assert_allclose(
                ecc_ref, 0.0 if ref == "scalar"
                else np.zeros(len(fref_in[ref])))
            np.testing.assert_allclose(
                meanano_ref, 0.0 if ref == "scalar"
                else np.zeros(len(fref_in[ref])))
