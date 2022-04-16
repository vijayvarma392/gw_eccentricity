import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
sys.path.append("../")
from measureEccentricity import measure_eccentricity
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

EOBeccs = 10**np.linspace(-5, np.log10(0.5), 100)[80:]
q = args.q
chi1z = args.chi1z
chi2z = args.chi2z
data_dir = args.data_dir
method = args.method

if not args.fig_dir:
    fig_dir = "."
else:
    fig_dir = args.fig_dir
fig_name = f"{fig_dir}/EccTest_q{q:.2f}_chi1z{chi1z:.2f}_chi2z{chi2z:.2f}.pdf"


MeasuredEccs = []

for ecc in tqdm(EOBeccs):
    fileName = f"{data_dir}/EccTest_q{q:.2f}_chi1z{chi1z:.2f}_chi2z{chi2z:.2f}_EOBecc{ecc:.7f}.h5"
    kwargs = {"filepath": fileName}
    dataDict = load_waveform(catalog="EOB", **kwargs)
    tref_in = dataDict["t"]
    tref_in = tref_in[tref_in < 0]
    tref_out, ecc, mean_ano = measure_eccentricity(tref_in, dataDict, method)
    MeasuredEccs.append(ecc[0])


fig, ax = plt.subplots()
ax.plot(EOBeccs, MeasuredEccs)
fig.savefig(f"{fig_name}")
