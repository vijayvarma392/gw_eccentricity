import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity
import numpy as np


def test_interface():
    """ Tests that the measure_eccentricity interface is working for all
    implemented methods.
    """
    # Load test waveform
    lal_kwargs = {"approximant": "EccentricTD",
                  "q": 1.0,
                  "chi1": [0.0, 0.0, 0.0],
                  "chi2": [0.0, 0.0, 0.0],
                  "Momega0": 0.01,
                  "ecc": 0.1,
                  "mean_ano": 0,
                  "include_zero_ecc": True}
    dataDict = load_data.load_waveform(**lal_kwargs)

    # List of all available methods
    available_methods = gw_eccentricity.get_available_methods()
    for method in available_methods:
        # Try evaluating at a single time
        gwecc_dict = measure_eccentricity(
            tref_in=-12000,
            method=method,
            dataDict=dataDict)
        tref_out = gwecc_dict["tref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]

        # Try evaluating at an array of times
        gwecc_dict = measure_eccentricity(
            tref_in=dataDict["t"],
            method=method,
            dataDict=dataDict)
        tref_out = gwecc_dict["tref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        gwecc_object = gwecc_dict["gwecc_object"]

        # Make diagnostic plots
        gwecc_object.make_diagnostic_plots(usetex=False)

        # Try evaluating at single frequency
        gwecc_dict = measure_eccentricity(
            fref_in=0.025 / (2 * np.pi),
            method=method,
            dataDict=dataDict)
        fref_out = gwecc_dict["fref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]

        # Try evaluating at an array of frequencies
        gwecc_dict = measure_eccentricity(
            fref_in=np.arange(0.025, 0.035, 0.001) / (2 * np.pi),
            method=method,
            dataDict=dataDict)
        fref_out = gwecc_dict["fref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        gwecc_object = gwecc_dict["gwecc_object"]
