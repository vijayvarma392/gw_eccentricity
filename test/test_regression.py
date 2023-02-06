"""Regression test."""
import numpy as np
import json
import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity

# locally it passs without problem but on github action, we need to set this tolerance
atol = 1e-9

def test_regression():
    """Regression test using all methods."""
    # List of all available methods
    available_methods = gw_eccentricity.get_available_methods()
    for method in available_methods:
        # Load regression data
        regression_data_file = f"test/regression_data/{method}_regression_data.json"
        # Load the regression data
        fl = open(regression_data_file, "r")
        regression_data = json.load(fl)
        fl.close()
        # waveform kwargs
        lal_kwargs = regression_data["waveform_kwargs"]
        # load waveform data
        dataDict = load_data.load_waveform(**lal_kwargs)
        extra_kwargs = regression_data["extra_kwargs"]

        # Try evaluating at times where regression data are saved
        regression_data_at_tref = regression_data["tref"]
        tref_in = regression_data_at_tref["time"]
        gwecc_dict = measure_eccentricity(
            tref_in=tref_in,
            method=method,
            dataDict=dataDict,
            extra_kwargs=extra_kwargs)
        tref_out = gwecc_dict["tref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        # Compare the measured data with the saved data
        np.testing.assert_allclose(
            ecc_ref, regression_data_at_tref["eccentricity"],
            atol=atol,
            err_msg="measured and saved eccentricity at saved times do not match.")
        np.testing.assert_allclose(
            meanano_ref, regression_data_at_tref["mean_anomaly"],
            atol=atol,
            err_msg="measured and saved mean anomaly at saved times do not match.")

        # Try evaluating at frequencies where regression data are saved
        regression_data_at_fref = regression_data["fref"]
        fref_in = regression_data_at_fref["frequency"]
        gwecc_dict = measure_eccentricity(
            fref_in=fref_in,
            method=method,
            dataDict=dataDict,
            extra_kwargs=extra_kwargs)
        fref_out = gwecc_dict["fref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        # Compare the measured data with the saved data
        np.testing.assert_allclose(
            ecc_ref, regression_data_at_fref["eccentricity"],
            atol=atol,
            err_msg="measured and saved eccentricity at saved frequencies do not match.")
        np.testing.assert_allclose(
            meanano_ref, regression_data_at_fref["mean_anomaly"],
            atol=atol,
            err_msg="measured and saved mean anomaly at saved frequencies do not match.")
