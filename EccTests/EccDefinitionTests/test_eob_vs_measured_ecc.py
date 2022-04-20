"""Test to compare eob model eccentricity with measured eccentricity."""

import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from argparse import RawTextHelpFormatter
import warnings
sys.path.append("../../")
from measureEccentricity import measure_eccentricity, get_available_methods
from measureEccentricity.load_data import load_waveform

parser = argparse.ArgumentParser(
    description=("This test is designed to check how well different"
                 " eccentricity definition works. The waveforms that are"
                 " used for this test are generated"
                 " from a fixed initial frequency and eccentricities varying "
                 " from 1e-5 to 0.5. We try to measure the eccentricity"
                 " from these waveforms using the definitions. For a given"
                 " waveform, we measure the eccentriicities at times in the"
                 " range [max(first_peak, first_trough), min(last_peak,"
                 " last_trough)] and collect the very first value of the "
                 " eccentricity in this range and then do the same looping"
                 " over all the eob waveforms. Finally we plot these"
                 " eccentricites vs the eccentricities that was used to"
                 " generate the eob waveforms.\n\n"
                 "The goal is to check that the measured eccentricity"
                 " varies smoothly with the eccentricities of the model."),
    formatter_class=RawTextHelpFormatter)
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help=("Base directory where waveform files are stored."))
parser.add_argument(
    "--method",
    type=str,
    nargs="+",
    default="all",
    help=("EccDefinition method to test.\n"
          "Could be one or more of the methods in"
          " measureEccentricity.get_available_methods.\n"
          "Default is 'all'.\n"
          "usage: --method 'Amplitude' 'Frequency'"))
parser.add_argument(
    "--set",
    type=str,
    default="all",
    nargs="+",
    help=("Run test on the set of parameters.\n"
          "Possible choices are one or more of 1, 2, 3, 4 or all.\n"
          "1: q=1, chi1z=chi2z=0.\n"
          "2: q=2, chi1z=chi2z=0.5\n"
          "3: q=4, chi1z=chi2z=-0.6\n"
          "4: q=6, chi1z=0.4, chi2z=-0.4.\n"
          "Default is 'all'.\n"
          "usage: --set 1 2"))
parser.add_argument(
    "--fig_dir",
    type=str,
    required=False,
    help="Directory to save figure.")
parser.add_argument(
    "--plot_format",
    type=str,
    default="png",
    help=("Format to save the plot. Default is 'png'. "
          "Could be any format that matplotlib supports."))

args = parser.parse_args()

EOBeccs = 10**np.linspace(-5, np.log10(0.5), 100)
param_sets = {"1": [1, 0, 0],
              "2": [2, 0.5, 0.5],
              "3": [4, -0.6, -0.6],
              "4": [6, 0.4, -0.4]}
data_dir = args.data_dir + "Non-Precessing/EOB/"
# don't want to raise warnings when length of data for interpolaion
# in the monotonicity check is too long
extra_kwargs = {"debug": False}


def plot_waveform_ecc_vs_model_ecc(method, set_key, ax):
    # We will loop over waveforms with ecc in EOBeccs
    # Howver many eccentricity definitions might not work
    # for all of these waveforms, for example, due to small
    # eccentricity. Therefore we need to track only those
    # eob model eccentricity for which the particular
    # definition works.
    waveform_eccs = []  # ecc as measured by the definition
    model_eccs = []  # ecc that goes in to the waveform generation
    q, chi1z, chi2z = param_sets[set_key]
    for ecc in tqdm(EOBeccs):
        fileName = (f"{data_dir}/EccTest_q{q:.2f}_chi1z{chi1z:.2f}_"
                    f"chi2z{chi2z:.2f}_EOBecc{ecc:.7f}.h5")
        kwargs = {"filepath": fileName}
        if method == "ResidualAmplitude":
            fileName_zero_ecc = (f"{data_dir}/EccTest_q{q:.2f}_chi1z"
                                 f"{chi1z:.2f}_"
                                 f"chi2z{chi2z:.2f}_EOBecc{0:.7f}.h5")
            kwargs.update({"filepath_zero_ecc": fileName_zero_ecc,
                           "include_zero_ecc": True})
        dataDict = load_waveform(catalog="EOB", **kwargs)
        tref_in = dataDict["t"]
        try:
            tref_out, measured_ecc, mean_ano = measure_eccentricity(tref_in,
                                                                    dataDict,
                                                                    method)
            waveform_eccs.append(measured_ecc[0])
            model_eccs.append(ecc)
        except Exception:
            warnings.warn("Exception raised. Probably too small eccentricity"
                          "to detect any extrema.")

    ax.loglog(model_eccs, waveform_eccs, marker="o", label=f"{method}")


if "all" in args.method:
    args.method = get_available_methods()
    method_str = "all"
else:
    method_str = "_".join(args.method)

if "all" in args.set:
    args.set = param_sets
else:
    sets = {}
    for key in args.set:
        sets.update({key: param_sets[key]})
    args.set = sets

fig_dir = args.fig_dir if args.fig_dir else "./"

for key in args.set:
    fig, ax = plt.subplots()
    fig_name = f"{fig_dir}/EccTest_set{key}_{method_str}.{args.plot_format}"
    for idx, method in enumerate(args.method):
        plot_waveform_ecc_vs_model_ecc(method, key, ax)
    ax.legend()
    ax.grid()
    ax.set_xlabel("EOB Eccentricity")
    ax.set_ylabel("Measured Eccentricity")
    fig.savefig(f"{fig_name}")
