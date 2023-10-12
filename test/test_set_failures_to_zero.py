import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity
import numpy as np


def test_set_failures_to_zero():
    """Tests handling of failures due to insufficient extrema.

    In certain situations, the waveform may have zero eccentricity or a very
    small eccentricity, making it difficult for the given method to identify
    any extrema. In cases where such a situation occurs, and if the user has
    configured the 'set_failures_to_zero' to `True` in the 'extra_kwargs'
    parameter, both the eccentricity and mean anomaly will be set to zero.
    """
    # The Amplitude and Frequency methods usually fail to detect any extrema
    # for eccentricities less than about 1e-3. Therefore, to test whether we
    # are setting eccentricity to zero when using these two methods, we use an
    # EccentricTD waveform with an initial eccentricity of 1e-4. However, since
    # the Residual and the Fits methods can detect extrema for the same
    # eccentricity, we use a quasicircular waveform to test them with these
    # methods.
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
    tref_in = -8000
    fref_in = 0.005
    # The following will set ecc and mean ano to zero
    # if no extrema are found.
    extra_kwargs = {"set_failures_to_zero": True}
    for method in available_methods:
        if method in ["Amplitude", "Frequency"]:
            dataDict = dataDict_ecc
        else:
            dataDict = dataDict_qc
        gwecc_dict = measure_eccentricity(
            tref_in=tref_in,
            method=method,
            dataDict=dataDict,
            extra_kwargs=extra_kwargs)
        tref_out = gwecc_dict["tref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        np.testing.assert_allclose(ecc_ref, 0.0)
        np.testing.assert_allclose(meanano_ref, 0.0)

        gwecc_dict = measure_eccentricity(
            fref_in=fref_in,
            method=method,
            dataDict=dataDict,
            extra_kwargs=extra_kwargs)
        fref_out = gwecc_dict["fref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        np.testing.assert_allclose(ecc_ref, 0.0)
        np.testing.assert_allclose(meanano_ref, 0.0)
