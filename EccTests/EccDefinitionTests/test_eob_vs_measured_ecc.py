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
    help=("Directory where EOB Files are stored"))
parser.add_argument(
    "--method",
    type=str,
    default="Amplitude",
    help=("EccDefinition method to use."))
parser.add_argument(
    "--q",
    type=float,
    default=1.,
    help="mass ratio of the system.")
parser.add_argument(
    "--chi1z",
    type=float,
    default=0.0,
    help="chi1z of the system.")
parser.add_argument(
    "--chi2z",
    type=float,
    default=0.0,
    help="chi2z of the system.")
parser.add_argument(
    "--fig_dir",
    type=str,
    required=False,
    help="Directory to save figure.")

args = parser.parse_args()

EOBeccs = 10**np.linspace(-5, np.log10(0.5), 100)
q = args.q
chi1z = args.chi1z
chi2z = args.chi2z
data_dir = args.data_dir
method = args.method

if not args.fig_dir:
    fig_dir = "."
else:
    fig_dir = args.fig_dir
fig_name = f"{fig_dir}/EccTest_q{q:.2f}_chi1z{chi1z:.2f}_chi2z{chi2z:.2f}_{method}.pdf"

fig, ax = plt.subplots()

markers = ["o", "v", "^", "<", ">", "d", "+", "x"]


def plot_for_method(method, marker, ax):
    MeasuredEccs = []
    WaveformEccs = []
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


if method == "All":
    methods = get_available_methods()
    for idx, method in enumerate(methods):
        plot_for_method(method, markers[idx], ax)
else:
    plot_for_method(method, markers[0], ax)

ax.legend()
ax.grid()
ax.set_xlabel("EOB Eccentricity")
ax.set_ylabel("Measured Eccentricity")
fig.savefig(f"{fig_name}")
