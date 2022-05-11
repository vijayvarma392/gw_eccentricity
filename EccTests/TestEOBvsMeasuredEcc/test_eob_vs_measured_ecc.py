__doc__ = """This test checks whether different eccentricity definitions vary
smoothly as a function of the internal eccentricity definition used by Toni's
EOB model. The EOB waveforms that are used for this test are generated from a
fixed mass ratio, spins, and Momega0 (the initial dimless orbital frequency),
with eccentricity varying from 1e-5 to 0.5. We try to measure the eccentricity
from these waveforms using different eccentricity definitions. For each
waveform, we measure the eccentricity at the very first extrema (periastron or
apastron). That way, the measured eccentricity is also at (nearly) Momega0.
Finally, we plot the measured eccentricity vs the internal EOB eccentricity.
You should check visually whether there are glitchy features in these plots.
Usage:
python test_eob_vs_measured_ecc.py -d ecc_waveforms -m 'Amplitude' 'ResidualAmplitude' -p 'all'
python test_eob_vs_measured_ecc.py -d ecc_waveforms -m 'all' -p 'all'
"""

import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import warnings
sys.path.append("../../")
from measureEccentricity import measure_eccentricity, get_available_methods
from measureEccentricity.load_data import load_waveform
from measureEccentricity.utils import SmartFormatter

parser = argparse.ArgumentParser(
    description=(__doc__),
    formatter_class=SmartFormatter)

parser.add_argument(
    "--data_dir", "-d",
    type=str,
    required=True,
    help=("Base directory where waveform files are stored. You can get this "
          "from home/md.shaikh/ecc_waveforms on CIT."))
parser.add_argument(
    "--method", "-m",
    type=str,
    nargs="+",
    default="all",
    help=("EccDefinition method to test. Can be 'all' OR one or more of the "
          f"methods in {list(get_available_methods())}."))
parser.add_argument(
    "--param_set_key", "-p",
    type=str,
    default="all",
    nargs="+",
    help=("Run test for this set of parameters kept fixed.\n"
          "Possible choices are 'all' OR one or more of 1, 2, 3, 4.\n"
          "1: q=1, chi1z=chi2z=0.\n"
          "2: q=2, chi1z=chi2z=0.5\n"
          "3: q=4, chi1z=chi2z=-0.6\n"
          "4: q=6, chi1z=0.4, chi2z=-0.4.\n"))
parser.add_argument(
    "--fig_dir",
    "-f",
    type=str,
    default='.',
    help="Directory to save figure.")
parser.add_argument(
    "--plot_format",
    type=str,
    default="png",
    help=("Format to save the plot. "
          "Can be any format that matplotlib supports."))
parser.add_argument(
    "--example",
    action="store_true",
    help=("This will override the figure name (that contains the "
          "information about parameter set, method used and so on)"
          " and uses a figure name which is of the form test_name_example.png"
          "where test_name is the name of the test."))

args = parser.parse_args()

EOBeccs = 10**np.linspace(-5, np.log10(0.5), 100)

# Format: [q, chi1z, chi2z]
available_param_sets = {
    "1": [1, 0, 0],
    "2": [2, 0.5, 0.5],
    "3": [4, -0.6, -0.6],
    "4": [6, 0.4, -0.4]}
data_dir = args.data_dir + "/Non-Precessing/EOB/"

# Avoid raising warnings when length of the data for interpolation
# in monotonicity check is too long
extra_kwargs = {"debug": False}


def plot_waveform_ecc_vs_model_ecc(method, set_key, ax):
    # We will loop over waveforms with ecc in EOBeccs
    # However many eccentricity definitions might not work
    # for all of these waveforms, for example, due to small
    # eccentricity. Therefore we need to track only those
    # eob model eccentricity for which the particular
    # definition works.
    waveform_eccs = []  # ecc as measured by the definition
    model_eccs = []  # ecc that goes in to the waveform generation
    q, chi1z, chi2z = available_param_sets[set_key]
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
            tref_out, measured_ecc, mean_ano = measure_eccentricity(
                tref_in,
                dataDict,
                method,
                extra_kwargs=extra_kwargs)
            # Get the measured eccentricity at the first available index.
            # This corresponds to the first extrema that occurs after the
            # initial time.
            waveform_eccs.append(measured_ecc[0])
            model_eccs.append(ecc)
        except Exception:
            warnings.warn("Exception raised. Probably too small eccentricity "
                          "to detect any extrema.")

    ax.loglog(model_eccs, waveform_eccs, marker=".", label=f"{method}")
    ax.set_title(f"$q$={q:.3f}, $\chi_{{1z}}$={chi1z:.3f}, $\chi_{{2z}}$={chi2z:.3f}")


if "all" in args.method:
    args.method = get_available_methods()
    # method_str is used in the filename for the output figure
    method_str = "all"
else:
    method_str = "_".join(args.method)

if "all" in args.param_set_key:
    args.param_set_key = list(available_param_sets.keys())

for key in args.param_set_key:
    fig, ax = plt.subplots()
    if args.example:
        fig_name = f"{args.fig_dir}/test_eob_vs_measured_ecc_example.png"
    else:
        fig_name = f"{args.fig_dir}/EccTest_set{key}_{method_str}.{args.plot_format}"
    for idx, method in enumerate(args.method):
        plot_waveform_ecc_vs_model_ecc(method, key, ax)
    ax.legend()
    ax.grid()
    ax.set_xlabel("EOB Eccentricity")
    ax.set_ylabel("Measured Eccentricity")
    fig.savefig(f"{fig_name}")
