import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import warnings
sys.path.append("../../")
from measureEccentricity import measure_eccentricity, get_available_methods
from measureEccentricity.load_data import load_waveform

parser = argparse.ArgumentParser()
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
    help=("EccDefinition method to test."
          " Could be one or more of the methods in"
          " measureEccentricity.get_available_methods."
          " Default is all."
          " usage: --method Amplitude Frequency"))
parser.add_argument(
    "--set",
    type=str,
    default="all",
    nargs="+",
    help=("Run test on the set of parameters. "
          "Possible choices are one or more of 1, 2, 3, 4 or all."
          " 1: q=1, chi1z=chi2z=0."
          " 2: q=2, chi1z=chi2z=0.5"
          " 3: q=4, chi1z=chi2z=-0.6"
          " 4: q=6, chi1z=0.4, chi2z=-0.4."
          " Default is all."
          " usage: --set 1 2"))
parser.add_argument(
    "--fig_dir",
    type=str,
    required=False,
    help="Directory to save figure.")

args = parser.parse_args()

EOBeccs = 10**np.linspace(-5, np.log10(0.5), 100)
param_sets = {"1": [1, 0, 0],
              "2": [2, 0.5, 0.5],
              "3": [4, -0.6, -0.6],
              "4": [6, 0.4, -0.4]}
data_dir = args.data_dir + "Non-Precessing/EOB/"

markers = ["o", "v", "^", "<", ">", "d", "+", "x"]


def plot_for(method, param_set, marker, ax):
    MeasuredEccs = []
    WaveformEccs = []
    q, chi1z, chi2z = param_sets[param_set]
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
        tref_in = tref_in[tref_in < 0]
        try:
            tref_out, measured_ecc, mean_ano = measure_eccentricity(tref_in,
                                                                    dataDict,
                                                                    method)
            MeasuredEccs.append(measured_ecc[0])
            WaveformEccs.append(ecc)
        except Exception:
            warnings.warn("Exception raised. Probably too small eccentricity"
                          "to detect any extrema.")

    ax.loglog(WaveformEccs, MeasuredEccs, marker=marker, label=f"{method}")


if args.method[0] == "all":
    methods = get_available_methods()
    method_str = args.method[0]
else:
    methods = args.method
    method_str = "_".join(args.method)

if args.set[0] == "all":
    sets = param_sets
else:
    sets = {}
    for key in args.set:
        sets.update({key: param_sets[key]})

fig_dir = args.fig_dir if args.fig_dir else "./"

for param_set in sets:
    fig, ax = plt.subplots()
    fig_name = f"{fig_dir}/EccTest_set{param_set}_{method_str}.png"
    for idx, method in enumerate(methods):
        plot_for(method, param_set, markers[idx], ax)
    ax.legend()
    ax.grid()
    ax.set_xlabel("EOB Eccentricity")
    ax.set_ylabel("Measured Eccentricity")
    fig.savefig(f"{fig_name}")
