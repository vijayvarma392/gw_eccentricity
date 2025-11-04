"""Generate data for regression test.

Run it from the base directory.
"""
import numpy as np
import json
import os
import argparse
import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity

data_dir = "test/regression_data/"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--method", "-m",
    type=str,
    required=True,
    help="EccDefinition method to save the regression data for.")
parser.add_argument(
    "--interp_method",
    type=str,
    required=True,
    help="omega_gw_extrema_interpolation_method to save the regression data for.")
parser.add_argument("--no-segment", action="store_false", dest="use_segment", help="Disable use of segment, use full waveform.")

args = parser.parse_args()


def generate_regression_data(method, interp_method, use_segment):
    """Generate data for regression test using a method."""
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
    if method not in available_methods:
        raise Exception(f"method {method} is not available. Must be one of "
                        f"{available_methods}")

    extra_kwargs = {"omega_gw_extrema_interpolation_method": interp_method,
                    "use_segment": use_segment}
    user_kwargs = extra_kwargs.copy()
    regression_data.update({"extra_kwargs": extra_kwargs})
    # Try evaluating at an array of times
    gwecc_dict = measure_eccentricity(
        tref_in=dataDict["t"],
        method=method,
        dataDict=dataDict,
        extra_kwargs=user_kwargs)
    tref_out = gwecc_dict["tref_out"]
    ecc_ref = gwecc_dict["eccentricity"]
    meanano_ref = gwecc_dict["mean_anomaly"]
    # We save the measured data 3 reference times
    n = len(tref_out)
    dict_tref =  {"time": [tref_out[n//8], tref_out[n//4], tref_out[n//2]],
                  "eccentricity": [ecc_ref[n//8], ecc_ref[n//4], ecc_ref[n//2]],
                  "mean_anomaly": [meanano_ref[n//8], meanano_ref[n//4], meanano_ref[n//2]]}
        
    # Try evaluating at an array of frequencies
    gwecc_dict = measure_eccentricity(
        fref_in=np.arange(0.025, 0.035, 0.001) / (2 * np.pi),
        method=method,
        dataDict=dataDict,
        extra_kwargs=user_kwargs)
    fref_out = gwecc_dict["fref_out"]
    ecc_ref = gwecc_dict["eccentricity"]
    meanano_ref = gwecc_dict["mean_anomaly"]
    n = len(fref_out)
    dict_fref = {"frequency": [fref_out[n//8], fref_out[n//4], fref_out[n//2]],
                 "eccentricity": [ecc_ref[n//8], ecc_ref[n//4], ecc_ref[n//2]],
                 "mean_anomaly": [meanano_ref[n//8], meanano_ref[n//4], meanano_ref[n//2]]}
    regression_data.update({"tref": dict_tref,
                            "fref": dict_fref})

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # save to a json file
    fl = open(f"{data_dir}/{method}_{interp_method}_use_segment_{use_segment}_regression_data.json", "w")
    json.dump(regression_data, fl)
    fl.close()

# generate regression data
generate_regression_data(args.method, args.interp_method, args.use_segment)
