import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity
from gw_eccentricity.exceptions import InsufficientExtrema
import numpy as np


def test_set_failures_to_zero():
    """ Tests that failures handling due to insufficient extrema.

    Amplitude and Frequency method usually fails to detect extrema when the
    eccentricity is below 10^-3. If the waveform has enough orbits and yet
    these methods fail to detect any extrema then it is probably because of the
    very small eccentricity. In such cases of failures, if the user set the key
    `set_failures_to_zero` in extra_kwargs, then eccentricity and mean anomaly
    are set to zero.
    """
    # Load test waveform
    lal_kwargs = {"approximant": "EccentricTD",
                  "q": 3.0,
                  "chi1": [0.0, 0.0, 0.0],
                  "chi2": [0.0, 0.0, 0.0],
                  "Momega0": 0.01,
                  "ecc": 1e-4,
                  "mean_ano": 0,
                  "include_zero_ecc": True}
    dataDict = load_data.load_waveform(**lal_kwargs)
    available_methods = gw_eccentricity.get_available_methods()
    tref_in = -8000
    fref_in = 0.005

    extra_kwargs = {"set_failures_to_zero": True}
    for method in available_methods:
        gwecc_dict = measure_eccentricity(
            tref_in=tref_in,
            method=method,
            dataDict=dataDict,
            extra_kwargs=extra_kwargs)
        tref_out = gwecc_dict["tref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        if method in ["Amplitude", "Frequency"]:
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
        if method in ["Amplitude", "Frequency"]:
            np.testing.assert_allclose(ecc_ref, 0.0)
            np.testing.assert_allclose(meanano_ref, 0.0)
