"""Generate data for regression test."""
import numpy as np
import json
import subprocess
import os
import sys
git_home = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                                   text=True).strip('\n')
sys.path.append(git_home)
import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity

data_dir = git_home + "/test/regression_data/"


def generate_regression_data():
    """Generate data for regression test using all methods."""
    # Load test waveform
    lal_kwargs = {"approximant": "EccentricTD",
                  "q": 1.0,
                  "chi1": [0.0, 0.0, 0.0],
                  "chi2": [0.0, 0.0, 0.0],
                  "Momega0": 0.01,
                  "ecc": 0.1,
                  "mean_ano": 0,
                  "include_zero_ecc": True}

    # Make a dictionary  to contain data we want to save for regression
    regression_data = {"waveform_kwargs": lal_kwargs}
    dataDict = load_data.load_waveform(**lal_kwargs)

    # List of all available methods
    available_methods = gw_eccentricity.get_available_methods()
    extra_kwargs = {"debug": False}
    regression_data.update({"extra_kwargs": extra_kwargs})
    for method in available_methods:
        # Try evaluating at an array of times
        gwecc_dict = measure_eccentricity(
            tref_in=dataDict["t"],
            method=method,
            dataDict=dataDict,
            extra_kwargs=extra_kwargs)
        tref_out = gwecc_dict["tref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        # For each method we save the measured data 3 reference times
        n = len(tref_out)
        dict_tref =  {"time": [tref_out[0], tref_out[n//4], tref_out[n//2]],
                      "eccentricity": [ecc_ref[0], ecc_ref[n//4], ecc_ref[n//2]],
                      "mean_anomaly": [meanano_ref[0], meanano_ref[n//4], meanano_ref[n//2]]}
        
        # Try evaluating at an array of frequencies
        gwecc_dict = measure_eccentricity(
            fref_in=np.arange(0.025, 0.035, 0.001) / (2 * np.pi),
            method=method,
            dataDict=dataDict,
            extra_kwargs=extra_kwargs)
        fref_out = gwecc_dict["fref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        n = len(fref_out)
        dict_fref = {"frequency": [fref_out[0], fref_out[n//4], fref_out[n//2]],
                     "eccentricity": [ecc_ref[0], ecc_ref[n//4], ecc_ref[n//2]],
                     "mean_anomaly": [meanano_ref[0], meanano_ref[n//4], meanano_ref[n//2]]}
        regression_data.update({method: {"tref": dict_tref,
                                         "fref": dict_fref}})

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # save to a json file
    fl = open(f"{data_dir}/regression_data.json", "w")
    json.dump(regression_data, fl)
    fl.close()

# generate regression data
generate_regression_data()
