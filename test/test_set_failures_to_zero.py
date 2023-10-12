import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity
import numpy as np


def test_set_failures_to_zero():
    """ Tests that failures handling due to insufficient extrema.

    In certain situations, the waveform may have zero eccentricity or a very
    small eccentricity, making it difficult for the given method to identify
    any extrema. In cases where such a situation occurs, and if the user has
    configured the 'set_failures_to_zero' to `True` in the 'extra_kwargs'
    parameter, both the eccentricity and mean anomaly will be forcibly set to
    zero.
    """
    # Load test waveform
    # We use a quasicircular waveform model to test if the eccentricity and
    # mean anomaly are correctly set to zero.
    lal_kwargs = {"approximant": "IMRPhenomT",
                  "q": 3.0,
                  "chi1": [0.0, 0.0, 0.0],
                  "chi2": [0.0, 0.0, 0.0],
                  "Momega0": 0.01,
                  "ecc": 0,
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
